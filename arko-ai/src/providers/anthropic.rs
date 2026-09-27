use super::{GenerationResponse, Provider, StreamEvent, take_refused_output};
use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::Provider as ProviderType,
    reasoning::{
        AnthropicReasoningEffort, AnthropicReasoningFragment, ReasoningArtifact, ReasoningConfig,
    },
    stop::{Refusal, RefusalSource, StopReason},
    tool::{ToolCall, ToolChoice},
    pricing::ServiceTier,
    usage::{ModelCallUsage, UsageInfo},
};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone)]
pub struct AnthropicClient {
    api_key: String,
    http_client: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http_client: super::default_http_client(),
        }
    }

    fn has_anthropic_reasoning_config(config: &GenerationConfig) -> bool {
        matches!(
            config.reasoning,
            Some(ReasoningConfig::Anthropic { .. })
                | Some(ReasoningConfig::AnthropicAdaptive { .. })
        )
    }

    /// Manual extended thinking (`budget_tokens`) is in effect.
    fn uses_manual_thinking(config: &GenerationConfig) -> bool {
        !Self::thinking_explicitly_disabled(config)
            && matches!(
                config.reasoning,
                Some(ReasoningConfig::Anthropic { budget_tokens, .. }) if budget_tokens > 0
            )
    }

    /// Fast mode, data residency and cache lifetime options.
    fn validate_request_options(
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<(), Error> {
        if config.anthropic_fast_mode == Some(true)
            && !conversation.model.supports_anthropic_fast_mode()
        {
            return Err(Error::InvalidRequest(format!(
                "Fast mode is not supported for model '{}' (supported: Claude Opus 5.5, Opus 5 and Opus 4.8; Opus 4.6 would silently run at standard speed).",
                conversation.model.id
            )));
        }

        if config.anthropic_inference_geo.is_some()
            && !conversation.model.supports_anthropic_inference_geo()
        {
            return Err(Error::InvalidRequest(format!(
                "inference_geo is only supported on Claude 4.6 and later models, not '{}'.",
                conversation.model.id
            )));
        }

        if config.anthropic_cache_ttl.is_some() && !Self::caching_enabled(conversation, config) {
            return Err(Error::InvalidRequest(format!(
                "An Anthropic cache TTL requires prompt caching to be enabled for model '{}'.",
                conversation.model.id
            )));
        }

        Ok(())
    }

    fn caching_enabled(conversation: &Conversation, config: &GenerationConfig) -> bool {
        config.prompt_caching.unwrap_or(conversation.enable_cache)
            && conversation.model.supports_cache
    }

    /// Top-level automatic caching; `ttl` defaults to 5 minutes when omitted.
    /// https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    fn apply_cache_control(
        json: &mut Value,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) {
        if !Self::caching_enabled(conversation, config) {
            return;
        }
        let mut cache_control = serde_json::json!({"type": "ephemeral"});
        if let Some(ttl) = config.anthropic_cache_ttl {
            cache_control["ttl"] = serde_json::to_value(ttl).unwrap_or(Value::Null);
        }
        json["cache_control"] = cache_control;
    }

    fn thinking_explicitly_disabled(config: &GenerationConfig) -> bool {
        config.anthropic_thinking_enabled == Some(false)
    }

    fn thinking_explicitly_enabled(config: &GenerationConfig) -> bool {
        config.anthropic_thinking_enabled == Some(true)
    }

    fn uses_adaptive_thinking(conversation: &Conversation, config: &GenerationConfig) -> bool {
        if Self::thinking_explicitly_disabled(config) {
            return false;
        }

        match &config.reasoning {
            Some(ReasoningConfig::AnthropicAdaptive { .. }) => true,
            Some(ReasoningConfig::Anthropic { .. }) => false,
            _ if Self::thinking_explicitly_enabled(config) => {
                conversation.model.supports_anthropic_adaptive_thinking()
            }
            _ => conversation.model.defaults_anthropic_adaptive_thinking(),
        }
    }

    fn has_anthropic_reasoning(conversation: &Conversation, config: &GenerationConfig) -> bool {
        if Self::thinking_explicitly_disabled(config) {
            return false;
        }

        match &config.reasoning {
            Some(ReasoningConfig::Anthropic { budget_tokens, .. }) => *budget_tokens > 0,
            Some(ReasoningConfig::AnthropicAdaptive { .. }) => true,
            _ if Self::thinking_explicitly_enabled(config) => {
                conversation.model.supports_anthropic_adaptive_thinking()
            }
            _ => conversation.model.defaults_anthropic_adaptive_thinking(),
        }
    }

    /// Beta headers. The 1M-context beta (`context-1m-2025-08-07`, retired for
    /// Sonnet 4/4.5 on 2026-04-30) and the effort betas (`effort-2025-11-24`,
    /// not required since 2026-02-05; `max-effort-2026-01-24`, undocumented) are
    /// no longer sent. https://platform.claude.com/docs/en/release-notes/overview
    fn compute_betas(conversation: &Conversation, config: &GenerationConfig) -> Vec<&'static str> {
        let mut betas = Vec::new();

        if config.anthropic_fast_mode == Some(true) {
            betas.push("fast-mode-2026-02-01");
        }

        let adaptive = Self::uses_adaptive_thinking(conversation, config);
        let interleaved =
            Self::is_interleaved_thinking_active(config, adaptive, &conversation.model);

        if interleaved {
            let needs_header = !conversation.model.auto_enables_interleaved_thinking() || !adaptive;

            if needs_header {
                betas.push("interleaved-thinking-2025-05-14");
            }
        }

        betas
    }

    fn is_interleaved_thinking_active(
        config: &GenerationConfig,
        adaptive: bool,
        model: &crate::model::Model,
    ) -> bool {
        match config.interleaved_thinking {
            Some(enabled) => enabled,
            None => adaptive && model.auto_enables_interleaved_thinking(),
        }
    }

    fn build_beta_header(conversation: &Conversation, config: &GenerationConfig) -> Option<String> {
        let betas = Self::compute_betas(conversation, config);

        if betas.is_empty() {
            None
        } else {
            Some(betas.join(","))
        }
    }

    fn validate_config(
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<(), Error> {
        if config.long_context == Some(true) && !conversation.model.supports_long_context() {
            if conversation.model.had_retired_anthropic_long_context_beta() {
                return Err(Error::InvalidRequest(format!(
                    "The 1M-context beta for model '{}' was retired on 2026-04-30; requests over 200K tokens now fail. Use a Claude 4.6 or later model for 1M context.",
                    conversation.model.id
                )));
            }
            return Err(Error::InvalidRequest(format!(
                "Long context configuration is not supported for model '{}'.",
                conversation.model.id
            )));
        }

        let forced_tool_choice = matches!(
            config.tool_config.as_ref().and_then(|tools| tools.tool_choice.as_ref()),
            Some(ToolChoice::Required | ToolChoice::Specific(_))
        );

        if forced_tool_choice && conversation.model.rejects_anthropic_forced_tool_choice() {
            return Err(Error::InvalidRequest(format!(
                "Forced tool use is not supported for model '{}'. Use Auto or None tool choice.",
                conversation.model.id
            )));
        }

        if forced_tool_choice && Self::uses_manual_thinking(config) {
            return Err(Error::InvalidRequest(
                "Forced tool use is not supported with manual extended thinking (budget_tokens). Use Auto or None tool choice."
                    .to_string(),
            ));
        }

        Self::validate_request_options(conversation, config)?;

        if Self::thinking_explicitly_disabled(config) {
            if conversation.model.always_uses_anthropic_adaptive_thinking() {
                return Err(Error::InvalidRequest(format!(
                    "Thinking cannot be disabled for model '{}'.",
                    conversation.model.id
                )));
            }

            if Self::has_anthropic_reasoning_config(config) {
                return Err(Error::InvalidRequest(
                    "Anthropic thinking cannot be disabled while an Anthropic reasoning configuration is present."
                        .to_string(),
                ));
            }

            if config.anthropic_summarized_thinking == Some(true) {
                return Err(Error::InvalidRequest(
                    "Anthropic summarized thinking cannot be requested while thinking is disabled."
                        .to_string(),
                ));
            }
        }

        if Self::thinking_explicitly_enabled(config)
            && !Self::has_anthropic_reasoning_config(config)
            && !conversation.model.supports_anthropic_adaptive_thinking()
        {
            return Err(Error::InvalidRequest(format!(
                "Model '{}' requires an Anthropic reasoning configuration to enable thinking.",
                conversation.model.id
            )));
        }

        let adaptive = Self::uses_adaptive_thinking(conversation, config);

        // `thinking.display` works in both adaptive and manual modes; where
        // summarized text is already the default it is a no-op.
        // https://platform.claude.com/docs/en/build-with-claude/thinking
        if config.anthropic_summarized_thinking == Some(true)
            && !Self::has_anthropic_reasoning(conversation, config)
        {
            return Err(Error::InvalidRequest(format!(
                "Anthropic summarized thinking requires thinking to be enabled for model '{}'.",
                conversation.model.id
            )));
        }

        if let Some(reasoning) = &config.reasoning {
            if matches!(reasoning, ReasoningConfig::Anthropic { .. })
                && conversation.model.rejects_anthropic_manual_thinking()
            {
                return Err(Error::InvalidRequest(format!(
                    "Model '{}' does not support budget_tokens-based extended thinking. Use AnthropicAdaptive thinking instead.",
                    conversation.model.id
                )));
            }

            if matches!(reasoning, ReasoningConfig::AnthropicAdaptive { .. })
                && !conversation.model.supports_anthropic_adaptive_thinking()
            {
                return Err(Error::InvalidRequest(format!(
                    "Adaptive thinking is not supported for model '{}'.",
                    conversation.model.id
                )));
            }

            let effort = match reasoning {
                ReasoningConfig::Anthropic { effort, .. } => effort.as_ref(),
                ReasoningConfig::AnthropicAdaptive { effort } => effort.as_ref(),
                _ => None,
            };

            if let Some(effort) = effort {
                if !conversation.model.supports_anthropic_effort() {
                    return Err(Error::InvalidRequest(format!(
                        "Reasoning effort is not supported for model '{}'.",
                        conversation.model.id
                    )));
                }

                if matches!(effort, AnthropicReasoningEffort::XHigh)
                    && !conversation.model.supports_anthropic_xhigh_effort()
                {
                    return Err(Error::InvalidRequest(format!(
                        "xhigh reasoning effort is not supported for model '{}'.",
                        conversation.model.id
                    )));
                }

                if matches!(effort, AnthropicReasoningEffort::Max)
                    && !conversation.model.supports_anthropic_max_effort()
                {
                    return Err(Error::InvalidRequest(format!(
                        "Max reasoning effort is not supported for model '{}'.",
                        conversation.model.id
                    )));
                }
            }
        }

        Self::validate_interleaved_thinking(conversation, config, adaptive)?;

        let has_reasoning = Self::has_anthropic_reasoning(conversation, config);

        if has_reasoning
            && !conversation.model.omits_anthropic_sampling_parameters()
            && config.temperature_specified
            && config.temperature != 1.0
        {
            return Err(Error::InvalidRequest(
                "Anthropic models with reasoning enabled must use temperature=1.0.".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_interleaved_thinking(
        conversation: &Conversation,
        config: &GenerationConfig,
        adaptive: bool,
    ) -> Result<(), Error> {
        match config.interleaved_thinking {
            Some(true) => {
                if conversation.model.provider != ProviderType::Anthropic {
                    return Err(Error::InvalidRequest(
                        "Interleaved thinking is only supported for Anthropic models.".to_string(),
                    ));
                }

                if !conversation.model.supports_interleaved_thinking() {
                    return Err(Error::InvalidRequest(format!(
                        "Interleaved thinking is not supported for model '{}'.",
                        conversation.model.id
                    )));
                }

                if !Self::has_anthropic_reasoning(conversation, config) {
                    return Err(Error::InvalidRequest(
                        "Interleaved thinking requires Anthropic reasoning to be enabled."
                            .to_string(),
                    ));
                }

                // Opus 4.6 has no interleaved thinking in manual mode; the beta
                // header would be a silent no-op.
                // https://platform.claude.com/docs/en/build-with-claude/extended-thinking
                if conversation.model.is_opus_4_6() && Self::uses_manual_thinking(config) {
                    return Err(Error::InvalidRequest(
                        "Claude Opus 4.6 does not interleave thinking with manual budget_tokens; use adaptive thinking instead."
                            .to_string(),
                    ));
                }
            }
            Some(false) if adaptive && conversation.model.auto_enables_interleaved_thinking() => {
                return Err(Error::InvalidRequest(format!(
                    "Interleaved thinking cannot be disabled for '{}' while adaptive thinking is active.",
                    conversation.model.id
                )));
            }
            Some(false) => {}
            None => {}
        }

        Ok(())
    }

    fn with_stream_stop_metadata(usage: ModelCallUsage, delta: &Value) -> ModelCallUsage {
        usage.with_stop_metadata(
            delta.get("stop_reason").and_then(Value::as_str).map(str::to_owned),
            delta.get("stop_details").cloned(),
        )
    }

    fn generation_response(
        response_data: AnthropicResponse,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> GenerationResponse {
        let (mut message, _) = Self::parse_response(&response_data);
        let mut usage = Self::calculate_usage(&response_data.usage);
        usage.calculate_cost(&conversation.model, Some(config));

        let model_call_usage = ModelCallUsage::new(
            response_data.id.clone(),
            conversation.model.provider.as_telemetry_str().to_string(),
            response_data.model.clone(),
            usage.input_tokens,
            usage.output_tokens,
            conversation.model.max_input_tokens,
        ).with_stop_metadata(
            response_data.stop_reason.clone(),
            response_data.stop_details.clone(),
        );

        let trailing_tool_call = matches!(
            response_data.content.last(),
            Some(AnthropicResponseContent::ToolUse { .. })
        );
        let (stop_reason, refusal) = Self::finish_turn(
            &mut message,
            response_data.stop_reason.as_deref(),
            response_data.stop_details.as_ref(),
            trailing_tool_call,
        );

        GenerationResponse {
            message,
            usage,
            model_call_usage,
            stop_reason,
            refusal,
        }
    }

    /// Normalizes a finished turn. `trailing_tool_call` says the last content
    /// block was a completed `tool_use` block, which a token limit may have
    /// cut short. Stop reasons:
    /// https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
    fn finish_turn(
        message: &mut Message,
        stop: Option<&str>,
        stop_details: Option<&Value>,
        trailing_tool_call: bool,
    ) -> (StopReason, Option<Refusal>) {
        let stop_reason = match stop {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("stop_sequence") => StopReason::StopSequence,
            Some("pause_turn") => StopReason::PauseTurn,
            Some("refusal") => StopReason::Refusal,
            Some("model_context_window_exceeded") => StopReason::ContextWindowExceeded,
            Some(other) => StopReason::Other(other.to_string()),
            None => StopReason::unknown(),
        };

        if stop_reason.is_truncated() && trailing_tool_call {
            message.tool_calls.pop();
        }

        // Streaming classifier refusals: partial output must be discarded and
        // the turn removed before continuing.
        // https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals
        let refusal = (stop_reason == StopReason::Refusal).then(|| {
            let details = stop_details.filter(|details| !details.is_null());
            let field = |name: &str| {
                details
                    .and_then(|details| details.get(name))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            };
            Refusal {
                source: RefusalSource::SafetyClassifier,
                category: field("category"),
                explanation: field("explanation"),
                partial_output: take_refused_output(message),
                provider_details: details.cloned().unwrap_or(Value::Null),
            }
        });

        (stop_reason, refusal)
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicClient {
    async fn generate_text(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        if conversation.model.provider != ProviderType::Anthropic {
            return Err(Error::ProviderNotSupported(
                "Model is not from Anthropic".to_string(),
            ));
        }

        Self::validate_config(conversation, config)?;

        let request = self.build_request(conversation, config)?;
        let mut json = serde_json::to_value(request).map_err(Error::JsonError)?;
        Self::apply_cache_control(&mut json, conversation, config);

        let mut request = self
            .http_client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if let Some(beta_header) = Self::build_beta_header(conversation, config) {
            request = request.header("anthropic-beta", beta_header);
        }

        let response = request.json(&json).send().await?;

        if !response.status().is_success() {
            return Err(Error::from_response("anthropic API error", response).await);
        }

        let response_data: AnthropicResponse = response.json().await?;
        Ok(Self::generation_response(response_data, conversation, config))
    }

    async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        if conversation.model.provider != ProviderType::Anthropic {
            return Err(Error::ProviderNotSupported(
                "Model is not from Anthropic".to_string(),
            ));
        }

        Self::validate_config(conversation, config)?;

        let request = self.build_request(conversation, config)?;
        let mut json = serde_json::to_value(request).map_err(Error::JsonError)?;
        Self::apply_cache_control(&mut json, conversation, config);

        json["stream"] = Value::Bool(true);

        let mut request = self
            .http_client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if let Some(beta_header) = Self::build_beta_header(conversation, config) {
            request = request.header("anthropic-beta", beta_header);
        }

        let response = request.json(&json).send().await?;

        if !response.status().is_success() {
            return Err(Error::from_response("anthropic API error", response).await);
        }

        Ok(Self::stream_events(
            response.bytes_stream(),
            conversation.model.clone(),
            config.clone(),
        ))
    }
}

impl AnthropicClient {
    /// Parses an Anthropic Messages SSE byte stream. `End` is emitted only for
    /// the `message_delta` event that carries the stop reason; a stream that
    /// ends earlier yields [`Error::StreamTruncated`].
    fn stream_events<S, B, E>(
        bytes: S,
        model: crate::model::Model,
        cost_config: GenerationConfig,
    ) -> BoxStream<'static, Result<StreamEvent, Error>>
    where
        S: futures::Stream<Item = Result<B, E>> + Send + 'static,
        B: AsRef<[u8]> + Send + 'static,
        E: std::fmt::Display + Send + 'static,
    {
        let stream = try_stream! {
            let mut event_stream = Box::pin(bytes.eventsource());
            let mut fragments = Vec::new();
            let mut current_thinking: Option<(String, Option<String>)> = None;
            let mut full_text = String::new();
            // `message_start` usage, updated by the final `message_delta`
            // (which carries cumulative counts).
            let mut usage_fields = serde_json::Map::new();
            let mut tool_calls = Vec::new();
            let mut current_tool: Option<(String, String, String)> = None;
            let mut tool_call_index = 0usize;
            let mut last_block_type = String::new();
            let mut response_id = String::new();
            let mut response_model = String::new();
            let mut finished = false;

            while let Some(event) = event_stream.next().await {
                let event =
                    event.map_err(|error| Error::ApiError(format!("SSE error: {error}")))?;

                let value: Value =
                    serde_json::from_str(&event.data).map_err(Error::JsonError)?;

                match value.get("type").and_then(Value::as_str) {
                    Some("message_start") => {
                        if let Some(message) = value.get("message") {
                            response_id = message
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();

                            response_model = message
                                .get("model")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();

                            if let Some(usage_value) = message.get("usage") {
                                merge_usage(&mut usage_fields, usage_value);
                            }
                        }
                    }
                    Some("content_block_start") => {
                        let block = &value["content_block"];
                        last_block_type = block["type"].as_str().unwrap_or("").to_string();

                        match last_block_type.as_str() {
                            "thinking" => {
                                current_thinking = Some((String::new(), None));
                            }
                            // Encrypted; kept for replay but never shown as reasoning text.
                            "redacted_thinking" => {
                                fragments.push(
                                    AnthropicReasoningFragment::RedactedThinking {
                                        data: block["data"].as_str().unwrap_or("").to_string(),
                                    },
                                );
                            }
                            "tool_use" => {
                                let id = block["id"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();

                                let name = block["name"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();

                                current_tool = Some((id, name.clone(), String::new()));

                                yield StreamEvent::ToolCallStart {
                                    index: tool_call_index,
                                    name,
                                };
                            }
                            _ => {}
                        }
                    }
                    Some("content_block_delta") => {
                        let delta = &value["delta"];

                        match delta["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                let text = delta["text"].as_str().unwrap_or("");

                                full_text.push_str(text);

                                yield StreamEvent::TextChunk(text.to_string());
                            }
                            "thinking_delta" => {
                                if let Some((thinking, _)) = &mut current_thinking {
                                    let text = delta["thinking"]
                                        .as_str()
                                        .unwrap_or("");

                                    thinking.push_str(text);

                                    yield StreamEvent::ReasoningChunk(text.to_string());
                                }
                            }
                            "signature_delta" => {
                                if let Some((_, signature)) = &mut current_thinking {
                                    *signature = Some(
                                        delta["signature"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string(),
                                    );
                                }
                            }
                            "input_json_delta" => {
                                if let Some((_, _, arguments)) = &mut current_tool {
                                    let partial = delta["partial_json"]
                                        .as_str()
                                        .unwrap_or("");

                                    arguments.push_str(partial);

                                    yield StreamEvent::ToolCallArgumentsDelta {
                                        index: tool_call_index,
                                        delta: partial.to_string(),
                                    };
                                }
                            }
                            _ => {}
                        }
                    }
                    Some("content_block_stop") => {
                        if let Some((thinking, signature)) = current_thinking.take() {
                            fragments.push(
                                AnthropicReasoningFragment::Thinking {
                                    thinking,
                                    signature: signature.unwrap_or_default(),
                                },
                            );
                        }

                        if let Some((id, name, arguments)) = current_tool.take() {
                            tool_calls.push(ToolCall::from_argument_text(
                                Some(id),
                                name,
                                &arguments,
                            ));

                            tool_call_index += 1;
                        }
                    }
                    Some("message_delta") => {
                        if value
                            .get("delta")
                            .and_then(|delta| delta.get("stop_reason"))
                            .is_some_and(|reason| !reason.is_null())
                            && let Some(usage_value) = value.get("usage")
                        {
                            merge_usage(&mut usage_fields, usage_value);
                            let reported: AnthropicUsage = serde_json::from_value(
                                Value::Object(std::mem::take(&mut usage_fields)),
                            )
                            .map_err(Error::JsonError)?;
                            let mut usage = Self::calculate_usage(&reported);
                            usage.calculate_cost(&model, Some(&cost_config));

                            let model_call_usage = ModelCallUsage::new(
                                response_id.clone(),
                                model.provider.as_telemetry_str().to_string(),
                                response_model.clone(),
                                usage.input_tokens,
                                usage.output_tokens,
                                model.max_input_tokens,
                            );
                            let model_call_usage = Self::with_stream_stop_metadata(
                                model_call_usage, &value["delta"],
                            );

                            let mut message = Message {
                                role: Role::Assistant,
                                content: vec![Content::Text(std::mem::take(&mut full_text))],
                                reasoning_artifact: if fragments.is_empty() {
                                    None
                                } else {
                                    Some(ReasoningArtifact::AnthropicReasoning {
                                        fragments: std::mem::take(&mut fragments),
                                    })
                                },
                                tool_calls: std::mem::take(&mut tool_calls),
                                id: None,
                            };

                            // A tool block still open here never finished, so it
                            // was never added; only a closed trailing one may be cut.
                            let trailing_tool_call =
                                last_block_type == "tool_use" && current_tool.is_none();
                            let (stop_reason, refusal) = Self::finish_turn(
                                &mut message,
                                value["delta"]["stop_reason"].as_str(),
                                value["delta"].get("stop_details"),
                                trailing_tool_call,
                            );

                            if let Some(refusal) = &refusal {
                                yield StreamEvent::Refusal(refusal.clone());
                            }

                            yield StreamEvent::End(Box::new(GenerationResponse {
                                message,
                                usage,
                                model_call_usage,
                                stop_reason,
                                refusal,
                            }));

                            finished = true;
                            break;
                        }
                    }
                    Some("error") => {
                        let message = value["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown");

                        Err(Error::provider_error(&format!("Anthropic stream error: {message}"), None, None, &value.to_string()))?;

                        break;
                    }
                    _ => {}
                }
            }

            if !finished {
                Err(Error::StreamTruncated(
                    "Anthropic stream ended before a stop reason was reported".to_string(),
                ))?;
            }
        };

        stream.boxed()
    }

    fn build_request(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<AnthropicRequest, Error> {
        // Omitted unless set, so the provider default applies (1.0, which is
        // also what thinking requires; validate_config rejects anything else).
        let temperature = (config.temperature_specified
            && !conversation.model.omits_anthropic_sampling_parameters())
        .then_some(config.temperature);

        let mut request = AnthropicRequest {
            model: conversation.model.id.clone(),
            max_tokens: config
                .max_tokens
                .unwrap_or(conversation.model.max_output_tokens),
            temperature,
            system: conversation.system_message.as_ref().map(|system| {
                vec![AnthropicSystemContent {
                    type_: "text".to_string(),
                    text: system.clone(),
                }]
            }),
            messages: Vec::new(),
            thinking: None,
            output_config: None,
            tools: None,
            tool_choice: None,
            speed: (config.anthropic_fast_mode == Some(true)).then_some("fast"),
            inference_geo: config.anthropic_inference_geo,
        };

        // Summarized thinking text must be requested where the model defaults to
        // omitted text; elsewhere it is already the default.
        let display = (config.anthropic_summarized_thinking == Some(true)
            && conversation.model.supports_anthropic_summarized_thinking_display())
        .then(|| "summarized".to_string());
        let max_tokens = request.max_tokens;

        if Self::thinking_explicitly_disabled(config) {
            request.thinking = Some(AnthropicThinking {
                type_: "disabled".to_string(),
                display: None,
                budget_tokens: None,
            });
        } else {
            match &config.reasoning {
                Some(ReasoningConfig::Anthropic {
                    budget_tokens,
                    effort,
                }) => {
                    if conversation.model.rejects_anthropic_manual_thinking() {
                        return Err(Error::InvalidRequest(format!(
                            "Model '{}' does not support budget_tokens-based extended thinking. Use AnthropicAdaptive thinking instead.",
                            conversation.model.id
                        )));
                    }

                    if *budget_tokens < 1024 {
                        return Err(Error::InvalidRequest(
                            "Anthropic reasoning budget must be at least 1024 tokens.".to_string(),
                        ));
                    }

                    // budget_tokens must be below max_tokens unless interleaved
                    // thinking lets thinking span tool calls.
                    // https://platform.claude.com/docs/en/build-with-claude/extended-thinking
                    let interleaved = Self::is_interleaved_thinking_active(
                        config,
                        false,
                        &conversation.model,
                    );
                    if !interleaved && *budget_tokens >= max_tokens {
                        return Err(Error::InvalidRequest(format!(
                            "Anthropic reasoning budget ({budget_tokens}) must be less than max_tokens ({max_tokens}) unless interleaved thinking is enabled."
                        )));
                    }

                    // These caps come from older Anthropic guidance for manual
                    // budgets and are no longer in the current docs; they are kept
                    // deliberately for models on the numeric-budget path.
                    if (conversation.model.is_opus_4_5() || conversation.model.is_sonnet_4_6())
                        && *budget_tokens > 52_000
                    {
                        return Err(Error::InvalidRequest(
                            "Claude Opus 4.5 and Claude Sonnet 4.6 reasoning budgets must be at most 52000 tokens."
                                .to_string(),
                        ));
                    }

                    if conversation.model.is_opus_4_6() && *budget_tokens > 102_400 {
                        return Err(Error::InvalidRequest(
                            "Claude Opus 4.6 reasoning budget must be at most 102400 tokens."
                                .to_string(),
                        ));
                    }

                    request.thinking = Some(AnthropicThinking {
                        type_: "enabled".to_string(),
                        display: display.clone(),
                        budget_tokens: Some(*budget_tokens),
                    });

                    if let Some(effort) = effort {
                        request.output_config = Some(AnthropicOutputConfig {
                            effort: effort_to_str(effort).to_string(),
                        });
                    }
                }
                Some(ReasoningConfig::AnthropicAdaptive { effort }) => {
                    request.thinking = Some(AnthropicThinking {
                        type_: "adaptive".to_string(),
                        display: display.clone(),
                        budget_tokens: None,
                    });

                    if let Some(effort) = effort {
                        request.output_config = Some(AnthropicOutputConfig {
                            effort: effort_to_str(effort).to_string(),
                        });
                    }
                }
                _ => {
                    // Explicitly enabled adaptive thinking, or a summarized display
                    // on a model whose adaptive thinking is on by default.
                    let explicit = Self::thinking_explicitly_enabled(config)
                        && conversation.model.supports_anthropic_adaptive_thinking();
                    let default_with_display = display.is_some()
                        && conversation.model.defaults_anthropic_adaptive_thinking();
                    if explicit || default_with_display {
                        request.thinking = Some(AnthropicThinking {
                            type_: "adaptive".to_string(),
                            display: display.clone(),
                            budget_tokens: None,
                        });
                    }
                }
            }
        }

        if let Some(tool_config) = &config.tool_config {
            request.tools = Some(
                tool_config
                    .tools
                    .iter()
                    .map(|tool| AnthropicToolDef {
                        type_: "custom".to_string(),
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        input_schema: tool.parameters.clone(),
                        strict: tool.strict,
                    })
                    .collect(),
            );

            let disable_parallel = tool_config.parallel_tool_calls == Some(false);

            request.tool_choice = Some(match &tool_config.tool_choice {
                Some(ToolChoice::Required) => AnthropicToolChoiceValue {
                    type_: "any".to_string(),
                    name: None,
                    disable_parallel_tool_use: disable_parallel.then_some(true),
                },
                Some(ToolChoice::Specific(name)) => AnthropicToolChoiceValue {
                    type_: "tool".to_string(),
                    name: Some(name.clone()),
                    disable_parallel_tool_use: disable_parallel.then_some(true),
                },
                Some(ToolChoice::None) => AnthropicToolChoiceValue {
                    type_: "none".to_string(),
                    name: None,
                    disable_parallel_tool_use: None,
                },
                Some(ToolChoice::Auto) | None => AnthropicToolChoiceValue {
                    type_: "auto".to_string(),
                    name: None,
                    disable_parallel_tool_use: disable_parallel.then_some(true),
                },
            });
        }

        for message in &conversation.messages {
            request
                .messages
                .push(self.convert_message_to_anthropic(message)?);
        }

        Ok(request)
    }

    fn convert_message_to_anthropic(&self, message: &Message) -> Result<AnthropicMessage, Error> {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => {
                return Err(Error::InvalidRequest(
                    "System messages should be set in the system field.".to_string(),
                ));
            }
        };

        let mut converted = AnthropicMessage {
            role: role.to_string(),
            content: Vec::new(),
        };

        if role == "assistant"
            && let Some(reasoning_artifact) = &message.reasoning_artifact
        {
            match reasoning_artifact {
                ReasoningArtifact::AnthropicReasoning { fragments } => {
                    for fragment in fragments {
                        match fragment {
                            AnthropicReasoningFragment::Thinking {
                                thinking,
                                signature,
                            } => {
                                converted.content.push(AnthropicContent::Thinking(
                                    AnthropicThinkingContent {
                                        type_: "thinking".to_string(),
                                        thinking: thinking.clone(),
                                        signature: signature.clone(),
                                    },
                                ));
                            }
                            AnthropicReasoningFragment::RedactedThinking { data } => {
                                converted.content.push(AnthropicContent::RedactedThinking(
                                    AnthropicRedactedThinkingContent {
                                        type_: "redacted_thinking".to_string(),
                                        data: data.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }
                _ => {
                    return Err(Error::InvalidRequest(
                        "Invalid reasoning artifact type for Anthropic.".to_string(),
                    ));
                }
            }
        }

        for content in &message.content {
            match content {
                Content::Text(text) => {
                    if role == "assistant" && text.trim().is_empty() {
                        continue;
                    }

                    converted
                        .content
                        .push(AnthropicContent::Text(AnthropicTextContent {
                            type_: "text".to_string(),
                            text: text.clone(),
                        }));
                }
                Content::Image(image) => {
                    converted.content.push(AnthropicContent::Image(
                        self.convert_image_to_anthropic(image)?,
                    ));
                }
                Content::ToolResult(result) => {
                    converted.content.push(AnthropicContent::ToolResult(
                        AnthropicToolResultContent {
                            type_: "tool_result".to_string(),
                            tool_use_id: result.call_id.clone().unwrap_or_default(),
                            content: result.output.clone(),
                            is_error: result.is_error.then_some(true),
                        },
                    ));
                }
            }
        }

        if role == "assistant" {
            for tool_call in &message.tool_calls {
                converted
                    .content
                    .push(AnthropicContent::ToolUse(AnthropicToolUseContent {
                        type_: "tool_use".to_string(),
                        id: tool_call.id.clone().unwrap_or_default(),
                        name: tool_call.name.clone(),
                        input: tool_call.replay_arguments(),
                    }));
            }
        }

        if role == "assistant" && converted.content.is_empty() {
            log::warn!(
                "Assistant message had no content blocks after conversion. Inserting a placeholder."
            );

            converted
                .content
                .push(AnthropicContent::Text(AnthropicTextContent {
                    type_: "text".to_string(),
                    text: "[Processing]".to_string(),
                }));
        }

        Ok(converted)
    }

    fn convert_image_to_anthropic(&self, image: &Image) -> Result<AnthropicImageContent, Error> {
        match image {
            Image::Url(_) => Err(Error::UnsupportedFeature(
                "Anthropic does not support image URLs, only base64 data.".to_string(),
            )),
            Image::Base64 { data, mime_type } => Ok(AnthropicImageContent {
                type_: "image".to_string(),
                source: AnthropicImageSource {
                    type_: "base64".to_string(),
                    media_type: mime_type.clone(),
                    data: data.clone(),
                },
            }),
        }
    }

    fn parse_response(response: &AnthropicResponse) -> (Message, Option<ReasoningArtifact>) {
        let mut text = String::new();
        let mut fragments = Vec::new();
        let mut tool_calls = Vec::new();

        for content in &response.content {
            match content {
                AnthropicResponseContent::Text { text: content, .. } => {
                    text.push_str(content);
                }
                AnthropicResponseContent::Thinking {
                    thinking,
                    signature,
                } => {
                    fragments.push(AnthropicReasoningFragment::Thinking {
                        thinking: thinking.clone(),
                        signature: signature.clone(),
                    });
                }
                AnthropicResponseContent::RedactedThinking { data } => {
                    fragments
                        .push(AnthropicReasoningFragment::RedactedThinking { data: data.clone() });
                }
                AnthropicResponseContent::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: Some(id.clone()),
                        name: name.clone(),
                        arguments: input.clone(),
                        thought_signature: None,
                        provider_id: None,
                        invalid_arguments: None,
                    });
                }
                AnthropicResponseContent::Unknown => {
                    log::debug!("Skipping an Anthropic content block of an unsupported type.");
                }
            }
        }

        let reasoning_artifact = if fragments.is_empty() {
            None
        } else {
            Some(ReasoningArtifact::AnthropicReasoning { fragments })
        };

        let message = Message {
            role: Role::Assistant,
            content: vec![Content::Text(text)],
            reasoning_artifact: reasoning_artifact.clone(),
            tool_calls,
            id: None,
        };

        (message, reasoning_artifact)
    }

    /// Anthropic's `input_tokens` counts only uncached input; arko's
    /// `UsageInfo::input_tokens` is the total, so cache reads and writes are
    /// added back. The 1-hour share of cache writes, fast mode (`speed`) and
    /// `inference_geo` select price variants.
    /// https://platform.claude.com/docs/en/build-with-claude/prompt-caching#tracking-cache-performance
    fn calculate_usage(reported: &AnthropicUsage) -> UsageInfo {
        let cache_read = reported.cache_read_input_tokens;
        let cache_write = reported.cache_creation_input_tokens;
        let mut usage = UsageInfo::new(
            reported.input_tokens + cache_read.unwrap_or(0) + cache_write.unwrap_or(0),
            reported.output_tokens,
        );
        usage.cached_tokens = cache_read;
        usage.cache_write_tokens = cache_write;
        usage.cache_write_1h_tokens = reported
            .cache_creation
            .as_ref()
            .and_then(|breakdown| breakdown.ephemeral_1h_input_tokens)
            .filter(|tokens| *tokens > 0);
        usage.service_tier = ServiceTier::from_anthropic(
            reported.service_tier.as_deref(),
            reported.speed.as_deref(),
        );
        usage.inference_geo = reported.inference_geo.clone();
        usage
    }
}

fn effort_to_str(effort: &AnthropicReasoningEffort) -> &'static str {
    match effort {
        AnthropicReasoningEffort::Low => "low",
        AnthropicReasoningEffort::Medium => "medium",
        AnthropicReasoningEffort::High => "high",
        AnthropicReasoningEffort::XHigh => "xhigh",
        AnthropicReasoningEffort::Max => "max",
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<AnthropicSystemContent>>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoiceValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_geo: Option<crate::config::AnthropicInferenceGeo>,
}

#[derive(Debug, Serialize)]
struct AnthropicToolDef {
    #[serde(rename = "type")]
    type_: String,
    name: String,
    description: String,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AnthropicToolChoiceValue {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disable_parallel_tool_use: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AnthropicSystemContent {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    display: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct AnthropicOutputConfig {
    effort: String,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(AnthropicTextContent),
    Image(AnthropicImageContent),
    Thinking(AnthropicThinkingContent),
    RedactedThinking(AnthropicRedactedThinkingContent),
    ToolUse(AnthropicToolUseContent),
    ToolResult(AnthropicToolResultContent),
}

#[derive(Debug, Serialize)]
struct AnthropicTextContent {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicImageContent {
    #[serde(rename = "type")]
    type_: String,
    source: AnthropicImageSource,
}

#[derive(Debug, Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    type_: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicThinkingContent {
    #[serde(rename = "type")]
    type_: String,
    thinking: String,
    signature: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRedactedThinkingContent {
    #[serde(rename = "type")]
    type_: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicToolUseContent {
    #[serde(rename = "type")]
    type_: String,
    id: String,
    name: String,
    input: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicToolResultContent {
    #[serde(rename = "type")]
    type_: String,
    tool_use_id: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type", default)]
    type_: String,
    #[serde(default)]
    role: String,
    model: String,
    content: Vec<AnthropicResponseContent>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<Value>,
    #[serde(default)]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Block types this client doesn't handle (for example server tool
    /// blocks or server-side fallback markers) are skipped.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Default, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: usize,
    #[serde(default)]
    output_tokens: usize,
    #[serde(default)]
    cache_creation_input_tokens: Option<usize>,
    #[serde(default)]
    cache_read_input_tokens: Option<usize>,
    /// Cache writes split by TTL.
    #[serde(default)]
    cache_creation: Option<AnthropicCacheCreation>,
    #[serde(default)]
    service_tier: Option<String>,
    /// `"fast"` when fast mode ran.
    #[serde(default)]
    speed: Option<String>,
    /// Where inference ran (`"global"` or `"us"`).
    #[serde(default)]
    inference_geo: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AnthropicCacheCreation {
    #[serde(default)]
    ephemeral_1h_input_tokens: Option<usize>,
}

/// Overlays usage fields from a later stream event; `message_delta` carries
/// cumulative counts that replace the `message_start` values.
fn merge_usage(fields: &mut serde_json::Map<String, Value>, update: &Value) {
    if let Some(update) = update.as_object() {
        for (key, value) in update {
            if !value.is_null() {
                fields.insert(key.clone(), value.clone());
            }
        }
    }
}

pub fn list_models() -> Vec<crate::model::Model> {
    crate::model::Model::list_anthropic_models()
}

#[cfg(test)]
mod tests {
    use super::AnthropicClient;
    use crate::{
        config::GenerationConfig,
        conversation::Conversation,
        error::Error,
        model::{Model, Provider},
        reasoning::{AnthropicReasoningEffort, ReasoningConfig},
        tool::{ToolChoice, ToolConfig, ToolDefinition},
    };

    #[test]
    fn tool_images_serialize_after_all_correlated_results_with_base64_source() {
        use crate::{Content, Image, Message, ToolCall, ToolResult};
        let client = AnthropicClient::new("test");
        let mut conversation = Conversation::new(Model::get(Provider::Anthropic, "claude-fable-5-1").unwrap());
        let calls = ["first", "second"].map(|id| ToolCall { id: Some(id.into()), name: "view_image".into(), arguments: serde_json::json!({"path":"fixture.png"}), thought_signature: None, provider_id: None, invalid_arguments: None });
        let mut assistant = Message::new_assistant("Inspecting images."); assistant.tool_calls = calls.to_vec(); conversation.messages.push(assistant);
        let mut content: Vec<_> = calls.iter().map(|call| Content::ToolResult(ToolResult::from_tool_call(call, "fixture.png, 12 × 8"))).collect();
        content.extend([Content::Text("Image from first call: fixture.png".into()), Content::Image(Image::new_base64("cGl4ZWxz", "image/png"))]);
        conversation.add_user_message_with_contents(content);
        let saved = serde_json::to_string(&conversation).unwrap();
        let conversation: Conversation = serde_json::from_str(&saved).unwrap();
        let value = serde_json::to_value(client.build_request(&conversation, &GenerationConfig::default()).unwrap()).unwrap();
        let blocks = value["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result"); assert_eq!(blocks[0]["tool_use_id"], "first");
        assert_eq!(blocks[1]["type"], "tool_result"); assert_eq!(blocks[1]["tool_use_id"], "second");
        assert_eq!(blocks[2]["type"], "text"); assert_eq!(blocks[3]["type"], "image");
        assert_eq!(blocks[3]["source"]["type"], "base64");
        assert_eq!(blocks[3]["source"]["media_type"], "image/png");
        assert_eq!(blocks[3]["source"]["data"], "cGl4ZWxz");
        assert_eq!(blocks[0]["content"], "fixture.png, 12 × 8");
    }

    #[test]
    fn streamed_refusal_stop_metadata_survives_usage_serialization() {
        let event = serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": null, "explanation": null}
            },
            "usage": {"output_tokens": 17}
        });
        let usage = crate::usage::ModelCallUsage::new(
            "msg_test".into(), "anthropic".into(), "test-model".into(), 42, 17, 200000,
        );
        let saved = serde_json::to_value(AnthropicClient::with_stream_stop_metadata(
            usage, &event["delta"],
        ).with_input_delta(3)).unwrap();
        assert_eq!(saved["stop_reason"], "refusal");
        assert_eq!(saved["stop_details"], event["delta"]["stop_details"]);
        assert_eq!(saved["output_tokens"], 17);
        assert_eq!(saved["input_delta_from_previous_call"], 3);
    }

    #[test]
    fn non_streaming_response_preserves_stop_details_and_normal_stops() {
        for reason in ["end_turn", "tool_use", "max_tokens", "refusal", "future_reason"] {
            let response: super::AnthropicResponse = serde_json::from_value(serde_json::json!({
                "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
                "content": [], "stop_reason": reason, "stop_sequence": null,
                "stop_details": {"type": "refusal", "category": "test", "explanation": "Fixture explanation"},
                "usage": {"input_tokens": 42, "output_tokens": 17}
            })).unwrap();
            let usage = crate::usage::ModelCallUsage::new(
                response.id, "anthropic".into(), response.model, 42, 17, 200000,
            ).with_stop_metadata(response.stop_reason, response.stop_details);
            assert_eq!(usage.stop_reason.as_deref(), Some(reason));
            assert_eq!(usage.stop_details.unwrap()["explanation"], "Fixture explanation");
        }
    }

    #[test]
    fn old_usage_records_and_null_stop_details_remain_compatible() {
        let old = serde_json::json!({
            "call_id": "old", "provider": "anthropic", "model": "test-model",
            "input_tokens": 10, "output_tokens": 2
        });
        let usage: crate::usage::ModelCallUsage = serde_json::from_value(old).unwrap();
        assert!(usage.stop_reason.is_none());
        assert!(usage.stop_details.is_none());
        let usage = AnthropicClient::with_stream_stop_metadata(
            usage, &serde_json::json!({"stop_reason": "end_turn", "stop_details": null}),
        );
        let saved = serde_json::to_value(usage).unwrap();
        assert_eq!(saved["stop_reason"], "end_turn");
        assert!(saved.get("stop_details").is_none());
    }

    #[test]
    fn latest_claude_models_build_adaptive_requests_without_beta_headers() {
        let client = AnthropicClient::new("test");
        for id in ["claude-fable-5-1", "claude-opus-5"] {
            let conversation = Conversation::new(Model::get(Provider::Anthropic, id).unwrap());
            let config = GenerationConfig::default()
                .with_long_context(true)
                .with_anthropic_summarized_thinking(true);
            AnthropicClient::validate_config(&conversation, &config).unwrap();
            assert!(AnthropicClient::build_beta_header(&conversation, &config).is_none());

            let request = client.build_request(&conversation, &config).unwrap();
            let value = serde_json::to_value(request).unwrap();
            assert_eq!(value["thinking"]["type"], "adaptive");
            assert_eq!(value["thinking"]["display"], "summarized");
            assert!(value.get("temperature").is_none());
            assert!(value["thinking"].get("budget_tokens").is_none());

            for effort in [
                AnthropicReasoningEffort::XHigh,
                AnthropicReasoningEffort::Max,
            ] {
                let config = config
                    .clone()
                    .with_reasoning(ReasoningConfig::anthropic_adaptive_with_effort(effort));
                AnthropicClient::validate_config(&conversation, &config).unwrap();
                let request = client.build_request(&conversation, &config).unwrap();
                let value = serde_json::to_value(request).unwrap();
                assert_eq!(
                    value["output_config"]["effort"],
                    serde_json::to_value(effort).unwrap()
                );
            }

            let config =
                GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(4096));
            assert!(matches!(
                AnthropicClient::validate_config(&conversation, &config),
                Err(Error::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn fable_5_1_rejects_disabled_thinking_and_forced_tool_use() {
        let conversation = Conversation::new(Model::anthropic_claude_fable_5_1());
        let config = GenerationConfig::default().with_anthropic_thinking_enabled(false);
        assert!(matches!(
            AnthropicClient::validate_config(&conversation, &config),
            Err(Error::InvalidRequest(_))
        ));

        for (choice, allowed) in [
            (ToolChoice::Auto, true),
            (ToolChoice::None, true),
            (ToolChoice::Required, false),
            (ToolChoice::Specific("lookup".to_string()), false),
        ] {
            let tool = ToolDefinition::new(
                "lookup",
                "Look up a value",
                serde_json::json!({"type": "object"}),
            );
            let config = GenerationConfig::default()
                .with_tool_config(ToolConfig::new(vec![tool]).with_tool_choice(choice));
            assert_eq!(
                AnthropicClient::validate_config(&conversation, &config).is_ok(),
                allowed
            );
        }

        let client = AnthropicClient::new("test");
        let conversation = Conversation::new(Model::anthropic_claude_opus_5());
        let config = GenerationConfig::default().with_anthropic_thinking_enabled(false);
        AnthropicClient::validate_config(&conversation, &config).unwrap();
        let request = client.build_request(&conversation, &config).unwrap();
        assert_eq!(
            serde_json::to_value(request).unwrap()["thinking"]["type"],
            "disabled"
        );
    }

    use crate::{
        providers::{GenerationResponse, StreamEvent},
        stop::{RefusalSource, StopReason},
    };
    use futures::StreamExt;
    use serde_json::{Value, json};

    fn sse(events: &[Value]) -> String {
        events.iter().map(|event| format!("data: {event}\n\n")).collect()
    }

    async fn stream_events(events: &[Value]) -> (Vec<StreamEvent>, Option<Error>) {
        let bytes = futures::stream::iter([Ok::<_, std::io::Error>(sse(events))]);
        let mut stream = AnthropicClient::stream_events(
            bytes,
            Model::anthropic_claude_sonnet_4_5(),
            GenerationConfig::default(),
        );
        let mut collected = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => collected.push(event),
                Err(error) => return (collected, Some(error)),
            }
        }
        (collected, None)
    }

    fn end(events: &[StreamEvent]) -> &GenerationResponse {
        match events.last() {
            Some(StreamEvent::End(response)) => response,
            other => panic!("expected End last, got {other:?}"),
        }
    }

    fn message_start() -> Value {
        json!({"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5",
            "usage":{"input_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}})
    }

    fn text_block(index: u64, text: &str) -> [Value; 3] {
        [
            json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
            json!({"type":"content_block_stop","index":index}),
        ]
    }

    fn tool_block(index: u64, id: &str, json_text: &str, close: bool) -> Vec<Value> {
        let mut events = vec![
            json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":id,"name":"fixture_tool","input":{}}}),
            json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":json_text}}),
        ];
        if close {
            events.push(json!({"type":"content_block_stop","index":index}));
        }
        events
    }

    fn message_delta(stop_reason: &str, stop_details: Value) -> Value {
        json!({"type":"message_delta","delta":{"stop_reason":stop_reason,"stop_details":stop_details},
            "usage":{"output_tokens":5}})
    }

    #[tokio::test]
    async fn streamed_classifier_refusal_discards_partial_output_from_the_message() {
        let mut events = vec![message_start()];
        events.extend(text_block(0, "Hello.."));
        events.push(message_delta(
            "refusal",
            json!({"type":"refusal","category":"cyber","explanation":"Declined."}),
        ));
        let (events, error) = stream_events(&events).await;
        assert!(error.is_none());
        assert!(matches!(events[events.len() - 2], StreamEvent::Refusal(_)));
        let response = end(&events);
        assert_eq!(response.stop_reason, StopReason::Refusal);
        let refusal = response.refusal.as_ref().unwrap();
        assert_eq!(refusal.source, RefusalSource::SafetyClassifier);
        assert_eq!(refusal.category.as_deref(), Some("cyber"));
        assert_eq!(refusal.explanation.as_deref(), Some("Declined."));
        assert_eq!(refusal.partial_output.as_deref(), Some("Hello.."));
        assert_eq!(response.text(), "");
        assert!(response.message.tool_calls.is_empty());
        assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("refusal"));

        // Such a turn is not added to the conversation.
        let mut conversation = Conversation::new(Model::anthropic_claude_sonnet_4_5());
        conversation.add_user_message("hi");
        conversation.add_assistant_message_from_response(response.clone());
        assert_eq!(conversation.messages.len(), 1);
    }

    #[test]
    fn non_streaming_refusal_with_null_details_is_still_a_refusal() {
        let response: super::AnthropicResponse = serde_json::from_value(json!({
            "id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5",
            "content":[{"type":"text","text":"partial"}],"stop_reason":"refusal",
            "stop_details":{"type":"refusal","category":null,"explanation":null},
            "usage":{"input_tokens":1,"output_tokens":1}
        })).unwrap();
        let response = AnthropicClient::generation_response(
            response,
            &Conversation::new(Model::anthropic_claude_sonnet_4_5()),
            &GenerationConfig::default(),
        );
        let refusal = response.refusal.unwrap();
        assert_eq!(refusal.category, None);
        assert_eq!(refusal.explanation, None);
        assert_eq!(refusal.partial_output.as_deref(), Some("partial"));
        assert!(matches!(
            response.message.content.as_slice(),
            [crate::Content::Text(text)] if text.is_empty()
        ));
    }

    #[tokio::test]
    async fn max_tokens_drops_a_trailing_or_unfinished_tool_call() {
        // A closed trailing tool block may have been cut by the limit.
        let mut events = vec![message_start()];
        events.extend(tool_block(0, "toolu_a", "{\"n\":1}", true));
        events.extend(tool_block(1, "toolu_b", "{\"n\":2}", true));
        events.push(message_delta("max_tokens", Value::Null));
        let (events, _) = stream_events(&events).await;
        let response = end(&events);
        assert_eq!(response.stop_reason, StopReason::MaxTokens);
        let ids: Vec<_> = response.message.tool_calls.iter().map(|call| call.id.as_deref().unwrap()).collect();
        assert_eq!(ids, ["toolu_a"]);

        // An unfinished block was never added; earlier calls stay.
        let mut events = vec![message_start()];
        events.extend(tool_block(0, "toolu_a", "{\"n\":1}", true));
        events.extend(tool_block(1, "toolu_b", "{\"n\":", false));
        events.push(message_delta("max_tokens", Value::Null));
        let (events, _) = stream_events(&events).await;
        let ids: Vec<_> = end(&events).message.tool_calls.iter().map(|call| call.id.as_deref().unwrap()).collect();
        assert_eq!(ids, ["toolu_a"]);

        // Normal tool use keeps every call.
        let mut events = vec![message_start()];
        events.extend(tool_block(0, "toolu_a", "{\"n\":1}", true));
        events.push(message_delta("tool_use", Value::Null));
        let (events, _) = stream_events(&events).await;
        assert_eq!(end(&events).stop_reason, StopReason::ToolUse);
        assert_eq!(end(&events).message.tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_json_is_marked_not_replaced() {
        let mut events = vec![message_start()];
        events.extend(tool_block(0, "toolu_a", "{broken", true));
        events.push(message_delta("tool_use", Value::Null));
        let (events, _) = stream_events(&events).await;
        let call = &end(&events).message.tool_calls[0];
        assert_eq!(call.invalid_arguments.as_deref(), Some("{broken"));
        assert!(call.arguments.is_null());
        assert_eq!(call.replay_arguments(), json!({}));
    }

    #[tokio::test]
    async fn redacted_thinking_is_kept_but_never_streamed_as_reasoning() {
        let events = vec![
            message_start(),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"ENCRYPTED"}}),
            json!({"type":"content_block_stop","index":0}),
            message_delta("end_turn", Value::Null),
        ];
        let (events, _) = stream_events(&events).await;
        assert!(!events.iter().any(|event| matches!(event, StreamEvent::ReasoningChunk(_))));
        let response = end(&events);
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert!(matches!(
            &response.message.reasoning_artifact,
            Some(crate::ReasoningArtifact::AnthropicReasoning { fragments }) if fragments.len() == 1
        ));
    }

    #[tokio::test]
    async fn a_stream_without_a_stop_reason_is_truncated() {
        let mut events = vec![message_start()];
        events.extend(text_block(0, "partial"));
        let (events, error) = stream_events(&events).await;
        assert!(matches!(events.last(), Some(StreamEvent::TextChunk(_))));
        assert!(matches!(error, Some(Error::StreamTruncated(_))));
    }

    #[test]
    fn unknown_content_blocks_and_stop_reasons_are_tolerated() {
        let response: super::AnthropicResponse = serde_json::from_value(json!({
            "id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5",
            "content":[
                {"type":"fallback","from":{"model":"a"},"to":{"model":"b"}},
                {"type":"text","text":"ok"}
            ],
            "stop_reason":"compaction","usage":{"input_tokens":1,"output_tokens":1}
        })).unwrap();
        let response = AnthropicClient::generation_response(
            response,
            &Conversation::new(Model::anthropic_claude_sonnet_4_5()),
            &GenerationConfig::default(),
        );
        assert_eq!(response.text(), "ok");
        assert_eq!(response.stop_reason, StopReason::Other("compaction".into()));
    }

    #[test]
    fn temperature_is_sent_only_when_specified() {
        let client = AnthropicClient::new("test");
        let conversation = Conversation::new(Model::anthropic_claude_sonnet_4_5());

        let request = client.build_request(&conversation, &GenerationConfig::default()).unwrap();
        assert!(serde_json::to_value(request).unwrap().get("temperature").is_none());

        let request = client.build_request(&conversation, &GenerationConfig::new(0.2)).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap()["temperature"], 0.2);

        // Thinking no longer forces an explicit temperature.
        let config = GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(2048));
        let request = client.build_request(&conversation, &config).unwrap();
        assert!(serde_json::to_value(request).unwrap().get("temperature").is_none());
    }

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn non_streaming_usage_reports_total_input_and_prices_cache_ttls() {
        let response: super::AnthropicResponse = serde_json::from_value(json!({
            "id": "msg_usage", "type": "message", "role": "assistant", "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100, "output_tokens": 100,
                "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200,
                "cache_creation": {"ephemeral_5m_input_tokens": 50, "ephemeral_1h_input_tokens": 150},
                "service_tier": "standard", "inference_geo": "global"
            }
        }))
        .unwrap();
        let conversation = Conversation::new(Model::anthropic_claude_opus_4_6());
        let result = AnthropicClient::generation_response(
            response,
            &conversation,
            &GenerationConfig::default(),
        );
        let usage = &result.usage;
        assert_eq!(usage.input_tokens, 1300);
        assert_eq!(result.model_call_usage.input_tokens, 1300);
        assert_eq!(usage.cached_tokens, Some(1000));
        assert_eq!(usage.cache_write_tokens, Some(200));
        assert_eq!(usage.cache_write_1h_tokens, Some(150));
        assert_eq!(usage.service_tier, Some(crate::pricing::ServiceTier::Standard));
        // Opus 4.6: $5 input, $0.50 reads, $6.25 5m writes, $10 1h writes, $25 output.
        assert!(close(usage.input_cost, 100.0 * 5.0 / 1e6));
        assert!(close(usage.cache_read_cost.unwrap(), 1000.0 * 0.5 / 1e6));
        assert!(close(
            usage.cache_write_cost.unwrap(),
            (50.0 * 6.25 + 150.0 * 10.0) / 1e6
        ));
        assert!(close(usage.output_cost, 100.0 * 25.0 / 1e6));
        assert!(close(usage.billed(), usage.total_cost));
    }

    #[tokio::test]
    async fn streamed_usage_uses_the_final_cumulative_counts() {
        let mut events = vec![message_start()];
        events.extend(text_block(0, "Hi"));
        events.push(json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},
            "usage":{"input_tokens":12,"cache_read_input_tokens":100,"output_tokens":50}}));
        let (events, error) = stream_events(&events).await;
        assert!(error.is_none(), "{error:?}");
        let response = end(&events);
        assert_eq!(response.usage.input_tokens, 112);
        assert_eq!(response.usage.cached_tokens, Some(100));
        assert_eq!(response.usage.cache_write_tokens, Some(0));
        assert_eq!(response.usage.output_tokens, 50);
        assert_eq!(response.model_call_usage.input_tokens, 112);
        assert_eq!(response.model_call_usage.output_tokens, 50);
    }

    #[test]
    fn fast_mode_and_us_inference_are_parsed_and_priced() {
        let reported: super::AnthropicUsage = serde_json::from_value(json!({
            "input_tokens": 1_000_000, "output_tokens": 1_000_000,
            "speed": "fast", "inference_geo": "us"
        }))
        .unwrap();
        let mut usage = AnthropicClient::calculate_usage(&reported);
        assert_eq!(usage.service_tier, Some(crate::pricing::ServiceTier::Fast));
        assert_eq!(usage.inference_geo.as_deref(), Some("us"));
        // Opus 4.8: $5 + $25 standard; fast 2x and US-only 1.1x stack.
        usage.calculate_cost(&Model::anthropic_claude_opus_4_8(), None);
        assert!(close(usage.total_cost, 30.0 * 2.0 * 1.1));
        // Opus 4.5 offers neither modifier.
        usage.calculate_cost(&Model::anthropic_claude_opus_4_5(), None);
        assert!(close(usage.total_cost, 30.0));
    }
}

#[cfg(test)]
mod model_rule_tests {
    use super::AnthropicClient;
    use crate::{
        config::{AnthropicCacheTtl, AnthropicInferenceGeo, GenerationConfig},
        conversation::Conversation,
        error::Error,
        model::Model,
        reasoning::{AnthropicReasoningEffort, ReasoningConfig},
        tool::{ToolChoice, ToolConfig, ToolDefinition},
    };
    use serde_json::{Value, json};

    fn request(model: Model, config: &GenerationConfig) -> Result<Value, Error> {
        let conversation = Conversation::new(model);
        AnthropicClient::validate_config(&conversation, config)?;
        let request = AnthropicClient::new("test").build_request(&conversation, config)?;
        let mut value = serde_json::to_value(request).unwrap();
        AnthropicClient::apply_cache_control(&mut value, &conversation, config);
        Ok(value)
    }

    fn betas(model: Model, config: &GenerationConfig) -> Option<String> {
        AnthropicClient::build_beta_header(&Conversation::new(model), config)
    }

    fn forced(config: GenerationConfig) -> GenerationConfig {
        let tool = ToolDefinition::new("lookup", "Look up a value", json!({"type": "object"}));
        config.with_tool_config(
            ToolConfig::new(vec![tool]).with_tool_choice(ToolChoice::Specific("lookup".into())),
        )
    }

    fn invalid(result: Result<Value, Error>) -> String {
        match result {
            Err(Error::InvalidRequest(message)) => message,
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn opus_5_5_is_always_adaptive_and_rejects_forced_tools_and_manual_budgets() {
        let model = Model::anthropic_claude_opus_5_5();
        let value = request(model.clone(), &GenerationConfig::default()).unwrap();
        assert!(value.get("thinking").is_none(), "adaptive is the server default");
        assert!(betas(model.clone(), &GenerationConfig::default()).is_none());

        let summarized = GenerationConfig::default().with_anthropic_summarized_thinking(true);
        let value = request(model.clone(), &summarized).unwrap();
        assert_eq!(value["thinking"]["type"], "adaptive");
        assert_eq!(value["thinking"]["display"], "summarized");

        for effort in [AnthropicReasoningEffort::XHigh, AnthropicReasoningEffort::Max] {
            let config = GenerationConfig::default()
                .with_reasoning(ReasoningConfig::anthropic_adaptive_with_effort(effort));
            request(model.clone(), &config).unwrap();
        }

        invalid(request(
            model.clone(),
            &GenerationConfig::default().with_anthropic_thinking_enabled(false),
        ));
        invalid(request(
            model.clone(),
            &GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(4096)),
        ));
        invalid(request(model.clone(), &forced(GenerationConfig::default())));

        // Non-default sampling parameters return 400 on Opus 5.5, so none is sent.
        let value = request(model, &GenerationConfig::new(0.3)).unwrap();
        assert!(value.get("temperature").is_none());
    }

    #[test]
    fn retired_sonnet_long_context_beta_is_rejected_and_never_sent() {
        let config = GenerationConfig::default().with_long_context(true);
        for model in [Model::anthropic_claude_sonnet_4_5(), Model::anthropic_claude_sonnet_4()] {
            let message = invalid(request(model.clone(), &config));
            assert!(message.contains("2026-04-30"), "{message}");
            assert!(betas(model.clone(), &config).is_none());
            assert!(model.pricing.large_input_threshold.is_none());
        }
        for model in [Model::anthropic_claude_sonnet_4_6(), Model::anthropic_claude_opus_4_6()] {
            request(model.clone(), &config).unwrap();
            assert!(betas(model, &config).is_none());
        }
    }

    #[test]
    fn opus_4_5_effort_needs_no_beta_header() {
        let config = GenerationConfig::default().with_reasoning(
            ReasoningConfig::anthropic_with_effort(4096, AnthropicReasoningEffort::High),
        );
        let value = request(Model::anthropic_claude_opus_4_5(), &config).unwrap();
        assert_eq!(value["output_config"]["effort"], "high");
        assert!(betas(Model::anthropic_claude_opus_4_5(), &config).is_none());
    }

    #[test]
    fn forced_tool_choice_is_rejected_with_manual_thinking_on_any_model() {
        let manual = GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(4096));
        let message = invalid(request(Model::anthropic_claude_sonnet_4_5(), &forced(manual)));
        assert!(message.contains("manual extended thinking"), "{message}");
        // Without manual thinking the same model may force a tool.
        request(Model::anthropic_claude_sonnet_4_5(), &forced(GenerationConfig::default()))
            .unwrap();
        // Fable 5.1 rejects forced tools regardless.
        invalid(request(
            Model::anthropic_claude_fable_5_1(),
            &forced(GenerationConfig::default()),
        ));
    }

    #[test]
    fn manual_budgets_stay_below_max_tokens_unless_interleaved_and_keep_legacy_caps() {
        let model = Model::anthropic_claude_sonnet_4_5();
        let config = GenerationConfig::default()
            .with_max_tokens(8_000)
            .with_reasoning(ReasoningConfig::anthropic(8_000));
        let message = invalid(request(model.clone(), &config));
        assert!(message.contains("less than max_tokens"), "{message}");
        request(model.clone(), &config.clone().with_interleaved_thinking(true)).unwrap();
        request(
            model,
            &GenerationConfig::default()
                .with_max_tokens(8_000)
                .with_reasoning(ReasoningConfig::anthropic(7_999)),
        )
        .unwrap();

        // Caps kept deliberately from older Anthropic guidance.
        let over = GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(52_001));
        invalid(request(Model::anthropic_claude_opus_4_5(), &over));
        invalid(request(Model::anthropic_claude_sonnet_4_6(), &over));
        request(Model::anthropic_claude_opus_4_6(), &over).unwrap();
        invalid(request(
            Model::anthropic_claude_opus_4_6(),
            &GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(102_401)),
        ));
    }

    #[test]
    fn summarized_display_works_in_both_modes_and_is_a_no_op_where_default() {
        // Opus 4.6: summarized is already the default.
        let adaptive = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::anthropic_adaptive())
            .with_anthropic_summarized_thinking(true);
        let value = request(Model::anthropic_claude_opus_4_6(), &adaptive).unwrap();
        assert_eq!(value["thinking"]["type"], "adaptive");
        assert!(value["thinking"].get("display").is_none());

        let manual = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::anthropic(4096))
            .with_anthropic_summarized_thinking(true);
        let value = request(Model::anthropic_claude_sonnet_4_5(), &manual).unwrap();
        assert_eq!(value["thinking"]["type"], "enabled");
        assert!(value["thinking"].get("display").is_none());

        // Summaries need thinking to be on.
        invalid(request(
            Model::anthropic_claude_sonnet_4_5(),
            &GenerationConfig::default().with_anthropic_summarized_thinking(true),
        ));
    }

    #[test]
    fn opus_4_6_manual_budgets_do_not_interleave() {
        let config = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::anthropic(4096))
            .with_interleaved_thinking(true);
        invalid(request(Model::anthropic_claude_opus_4_6(), &config));
        // Opus 4.5 manual mode still interleaves with the beta header.
        request(Model::anthropic_claude_opus_4_5(), &config).unwrap();
        assert_eq!(
            betas(Model::anthropic_claude_opus_4_5(), &config).as_deref(),
            Some("interleaved-thinking-2025-05-14")
        );
    }

    #[test]
    fn fast_mode_sends_speed_and_beta_only_on_supported_models() {
        let fast = GenerationConfig::default().with_anthropic_fast_mode(true);
        for model in [
            Model::anthropic_claude_opus_5_5(),
            Model::anthropic_claude_opus_5(),
            Model::anthropic_claude_opus_4_8(),
        ] {
            let value = request(model.clone(), &fast).unwrap();
            assert_eq!(value["speed"], "fast");
            assert_eq!(betas(model, &fast).as_deref(), Some("fast-mode-2026-02-01"));
        }
        for model in [Model::anthropic_claude_opus_4_7(), Model::anthropic_claude_opus_4_6()] {
            invalid(request(model, &fast));
        }
        let value =
            request(Model::anthropic_claude_opus_5_5(), &GenerationConfig::default()).unwrap();
        assert!(value.get("speed").is_none());
        let off = GenerationConfig::default().with_anthropic_fast_mode(false);
        let value = request(Model::anthropic_claude_opus_4_7(), &off).unwrap();
        assert!(value.get("speed").is_none());
        assert!(betas(Model::anthropic_claude_opus_4_7(), &off).is_none());
    }

    #[test]
    fn inference_geo_is_sent_only_on_claude_4_6_and_later() {
        let us =
            GenerationConfig::default().with_anthropic_inference_geo(AnthropicInferenceGeo::Us);
        let value = request(Model::anthropic_claude_opus_5_5(), &us).unwrap();
        assert_eq!(value["inference_geo"], "us");
        let global = GenerationConfig::default()
            .with_anthropic_inference_geo(AnthropicInferenceGeo::Global);
        let value = request(Model::anthropic_claude_sonnet_4_6(), &global).unwrap();
        assert_eq!(value["inference_geo"], "global");
        for model in [
            Model::anthropic_claude_opus_4_5(),
            Model::anthropic_claude_sonnet_4_5(),
            Model::anthropic_claude_haiku_4_5(),
        ] {
            invalid(request(model, &us));
        }
        let value =
            request(Model::anthropic_claude_opus_5_5(), &GenerationConfig::default()).unwrap();
        assert!(value.get("inference_geo").is_none());
    }

    #[test]
    fn cache_ttl_sets_the_automatic_cache_control_lifetime() {
        let model = Model::anthropic_claude_opus_5_5();
        let value = request(model.clone(), &GenerationConfig::default()).unwrap();
        assert_eq!(value["cache_control"], json!({"type": "ephemeral"}));

        let hour =
            GenerationConfig::default().with_anthropic_cache_ttl(AnthropicCacheTtl::OneHour);
        let value = request(model.clone(), &hour).unwrap();
        assert_eq!(value["cache_control"], json!({"type": "ephemeral", "ttl": "1h"}));

        let five =
            GenerationConfig::default().with_anthropic_cache_ttl(AnthropicCacheTtl::FiveMinutes);
        assert_eq!(request(model.clone(), &five).unwrap()["cache_control"]["ttl"], "5m");

        invalid(request(model.clone(), &hour.clone().with_prompt_caching(false)));
        let value =
            request(model, &GenerationConfig::default().with_prompt_caching(false)).unwrap();
        assert!(value.get("cache_control").is_none());
    }

    #[test]
    fn opus_5_5_reported_speed_geo_and_hour_writes_drive_the_price() {
        let reported: super::AnthropicUsage = serde_json::from_value(json!({
            "input_tokens": 1_000_000, "output_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1_000_000},
            "speed": "fast", "inference_geo": "us"
        }))
        .unwrap();
        let mut usage = AnthropicClient::calculate_usage(&reported);
        usage.calculate_cost(&Model::anthropic_claude_opus_5_5(), None);
        // $4 input + $20 output + $8 1-hour writes; fast 2x and US-only 1.1x stack.
        let expected = (4.0 + 20.0 + 8.0) * 2.0 * 1.1;
        assert!((usage.total_cost - expected).abs() < 1e-6, "{}", usage.total_cost);

        // The response, not the request, decides: standard speed, global inference.
        let standard: super::AnthropicUsage = serde_json::from_value(json!({
            "input_tokens": 1_000_000, "output_tokens": 0,
            "speed": "standard", "inference_geo": "global"
        }))
        .unwrap();
        let mut usage = AnthropicClient::calculate_usage(&standard);
        usage.calculate_cost(&Model::anthropic_claude_opus_5_5(), None);
        assert!((usage.total_cost - 4.0).abs() < 1e-6, "{}", usage.total_cost);
    }
}
