use super::{GenerationResponse, Provider, StreamEvent};
use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::Provider as ProviderType,
    reasoning::{
        AnthropicReasoningEffort, AnthropicReasoningFragment, ReasoningArtifact, ReasoningConfig,
    },
    tool::{ToolCall, ToolChoice},
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

    fn compute_betas(conversation: &Conversation, config: &GenerationConfig) -> Vec<&'static str> {
        let mut betas = Vec::new();

        if config.long_context == Some(true) && conversation.model.requires_long_context_beta() {
            betas.push("context-1m-2025-08-07");
        }

        let effort = match &config.reasoning {
            Some(ReasoningConfig::AnthropicAdaptive { effort }) => effort.as_ref(),
            Some(ReasoningConfig::Anthropic { effort, .. }) => effort.as_ref(),
            _ => None,
        };

        if let Some(effort) = effort
            && conversation.model.requires_anthropic_effort_beta()
        {
            betas.push("effort-2025-11-24");

            if matches!(effort, AnthropicReasoningEffort::Max) {
                betas.push("max-effort-2026-01-24");
            }
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
            return Err(Error::InvalidRequest(format!(
                "Long context configuration is not supported for model '{}'.",
                conversation.model.id
            )));
        }

        if conversation.model.is_fable_5_1()
            && matches!(
                config.tool_config.as_ref().and_then(|tools| tools.tool_choice.as_ref()),
                Some(ToolChoice::Required | ToolChoice::Specific(_))
            )
        {
            return Err(Error::InvalidRequest(format!(
                "Forced tool use is not supported for model '{}'. Use Auto or None tool choice.",
                conversation.model.id
            )));
        }

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

        if config.anthropic_summarized_thinking == Some(true) {
            if !adaptive {
                return Err(Error::InvalidRequest(
                    "Anthropic summarized thinking requires adaptive thinking.".to_string(),
                ));
            }

            if !conversation
                .model
                .supports_anthropic_summarized_thinking_display()
            {
                return Err(Error::InvalidRequest(format!(
                    "Summarized thinking display is not supported for model '{}'.",
                    conversation.model.id
                )));
            }
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

    fn unified_input_tokens(usage: &AnthropicUsage) -> usize {
        usage.input_tokens
            + usage.cache_creation_input_tokens.unwrap_or(0)
            + usage.cache_read_input_tokens.unwrap_or(0)
    }

    fn with_stream_stop_metadata(usage: ModelCallUsage, delta: &Value) -> ModelCallUsage {
        usage.with_stop_metadata(
            delta.get("stop_reason").and_then(Value::as_str).map(str::to_owned),
            delta.get("stop_details").cloned(),
        )
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

        let caching_enabled = config.prompt_caching.unwrap_or(conversation.enable_cache);

        if caching_enabled && conversation.model.supports_cache {
            json["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }

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
        let (message, _) = Self::parse_response(&response_data)?;
        let mut usage = Self::calculate_usage(&response_data);
        usage.calculate_cost(&conversation.model, Some(config));

        let model_call_usage = ModelCallUsage::new(
            response_data.id.clone(),
            conversation.model.provider.as_telemetry_str().to_string(),
            response_data.model.clone(),
            Self::unified_input_tokens(&response_data.usage),
            response_data.usage.output_tokens,
            conversation.model.max_input_tokens,
        ).with_stop_metadata(
            Some(response_data.stop_reason.clone()),
            response_data.stop_details.clone(),
        );

        Ok(GenerationResponse {
            message,
            usage,
            model_call_usage,
        })
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

        let caching_enabled = config.prompt_caching.unwrap_or(conversation.enable_cache);

        if caching_enabled && conversation.model.supports_cache {
            json["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }

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

        let byte_stream = response.bytes_stream().eventsource();
        let model = conversation.model.clone();
        let cost_config = (*config).clone();

        let stream = try_stream! {
            let mut event_stream = byte_stream;
            let mut fragments = Vec::new();
            let mut current_thinking: Option<(String, Option<String>)> = None;
            let mut full_text = String::new();
            let mut initial_usage: Option<UsageInfo> = None;
            let mut tool_calls = Vec::new();
            let mut current_tool: Option<(String, String, String)> = None;
            let mut tool_call_index = 0usize;
            let mut response_id = String::new();
            let mut response_model = String::new();

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
                                let input_tokens = usage_value
                                    .get("input_tokens")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(0) as usize;

                                let mut usage = UsageInfo::new(input_tokens, 0);

                                if let Some(cached) = usage_value
                                    .get("cache_read_input_tokens")
                                    .and_then(Value::as_u64)
                                {
                                    usage.cached_tokens = Some(cached as usize);
                                }

                                if let Some(written) = usage_value
                                    .get("cache_creation_input_tokens")
                                    .and_then(Value::as_u64)
                                {
                                    usage.cache_write_tokens = Some(written as usize);
                                }

                                initial_usage = Some(usage);
                            }
                        }
                    }
                    Some("content_block_start") => {
                        let block = &value["content_block"];

                        match block["type"].as_str().unwrap_or("") {
                            "thinking" => {
                                current_thinking = Some((String::new(), None));
                            }
                            "redacted_thinking" => {
                                let data = block["data"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();

                                fragments.push(
                                    AnthropicReasoningFragment::RedactedThinking {
                                        data: data.clone(),
                                    },
                                );

                                yield StreamEvent::ReasoningChunk(data);
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
                            let arguments =
                                serde_json::from_str(&arguments).unwrap_or(
                                    Value::Object(Default::default()),
                                );

                            tool_calls.push(ToolCall {
                                id: Some(id),
                                name,
                                arguments,
                                thought_signature: None,
                                provider_id: None,
                            });

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
                            let output_tokens = usage_value
                                .get("output_tokens")
                                .and_then(Value::as_u64)
                                .unwrap_or(0) as usize;

                            let mut usage = initial_usage
                                .take()
                                .unwrap_or_else(|| UsageInfo::new(0, 0));

                            usage.output_tokens = output_tokens;

                            let unified_input = usage.input_tokens
                                + usage.cached_tokens.unwrap_or(0)
                                + usage.cache_write_tokens.unwrap_or(0);

                            usage.calculate_cost(&model, Some(&cost_config));

                            let model_call_usage = ModelCallUsage::new(
                                response_id.clone(),
                                model.provider.as_telemetry_str().to_string(),
                                response_model.clone(),
                                unified_input,
                                output_tokens,
                                model.max_input_tokens,
                            );
                            let model_call_usage = Self::with_stream_stop_metadata(
                                model_call_usage, &value["delta"],
                            );

                            let message = Message {
                                role: Role::Assistant,
                                content: vec![Content::Text(full_text.clone())],
                                reasoning_artifact: if fragments.is_empty() {
                                    None
                                } else {
                                    Some(ReasoningArtifact::AnthropicReasoning {
                                        fragments: fragments.clone(),
                                    })
                                },
                                tool_calls: tool_calls.clone(),
                                id: None,
                            };

                            yield StreamEvent::End(Box::new(GenerationResponse {
                                message,
                                usage,
                                model_call_usage,
                            }));

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
        };

        Ok(stream.boxed())
    }
}

impl AnthropicClient {
    fn build_request(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<AnthropicRequest, Error> {
        let has_reasoning = Self::has_anthropic_reasoning(conversation, config);

        let temperature = if conversation.model.omits_anthropic_sampling_parameters() {
            None
        } else if has_reasoning {
            Some(1.0)
        } else {
            Some(config.temperature)
        };

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
        };

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
                        display: None,
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
                        display: if config.anthropic_summarized_thinking == Some(true) {
                            Some("summarized".to_string())
                        } else {
                            None
                        },
                        budget_tokens: None,
                    });

                    if let Some(effort) = effort {
                        request.output_config = Some(AnthropicOutputConfig {
                            effort: effort_to_str(effort).to_string(),
                        });
                    }
                }
                _ => {
                    if Self::thinking_explicitly_enabled(config)
                        && conversation.model.supports_anthropic_adaptive_thinking()
                    {
                        request.thinking = Some(AnthropicThinking {
                            type_: "adaptive".to_string(),
                            display: if config.anthropic_summarized_thinking == Some(true) {
                                Some("summarized".to_string())
                            } else {
                                None
                            },
                            budget_tokens: None,
                        });
                    } else if config.anthropic_summarized_thinking == Some(true)
                        && conversation.model.defaults_anthropic_adaptive_thinking()
                    {
                        request.thinking = Some(AnthropicThinking {
                            type_: "adaptive".to_string(),
                            display: Some("summarized".to_string()),
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
                        input: tool_call.arguments.clone(),
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

    fn parse_response(
        response: &AnthropicResponse,
    ) -> Result<(Message, Option<ReasoningArtifact>), Error> {
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
                    });
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

        Ok((message, reasoning_artifact))
    }

    fn calculate_usage(response: &AnthropicResponse) -> UsageInfo {
        let mut usage = UsageInfo::new(response.usage.input_tokens, response.usage.output_tokens);

        if let Some(cache_read) = response.usage.cache_read_input_tokens {
            usage = usage.with_cached_tokens(cache_read);
        }

        if let Some(cache_creation) = response.usage.cache_creation_input_tokens {
            usage = usage.with_cache_write_tokens(cache_creation);
        }

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
    #[serde(rename = "type")]
    type_: String,
    role: String,
    model: String,
    content: Vec<AnthropicResponseContent>,
    stop_reason: String,
    #[serde(default)]
    stop_details: Option<Value>,
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
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
    cache_creation_input_tokens: Option<usize>,
    cache_read_input_tokens: Option<usize>,
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
        let calls = ["first", "second"].map(|id| ToolCall { id: Some(id.into()), name: "view_image".into(), arguments: serde_json::json!({"path":"fixture.png"}), thought_signature: None, provider_id: None });
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
            ).with_stop_metadata(Some(response.stop_reason), response.stop_details);
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
}
