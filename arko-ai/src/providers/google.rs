use super::{GenerationResponse, ProviderClient, StreamEvent};
use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::{ModelType, Provider as ProviderType},
    reasoning::{GeminiThinkingLevel, GoogleReasoningFragment, ReasoningArtifact, ReasoningConfig},
    stop::{Refusal, RefusalSource, StopReason},
    tool::{ToolCall, ToolChoice},
    usage::{ModelCallUsage, UsageInfo},
};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default Interactions API base URL; requests go to `{base}/interactions`.
const GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1";

#[derive(Clone)]
pub struct GoogleClient {
    api_key: String,
    http_client: reqwest::Client,
    base_url: String,
}

impl GoogleClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http_client: super::default_http_client(),
            base_url: GOOGLE_BASE_URL.to_string(),
        }
    }

    /// Sends requests to `{base_url}/interactions` instead of
    /// `https://generativelanguage.googleapis.com/v1/interactions`, for example
    /// a proxy or a test server.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub(crate) fn interactions_url(&self) -> String {
        format!("{}/interactions", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait::async_trait]
impl ProviderClient for GoogleClient {
    async fn generate_text(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        if conversation.model.provider != ProviderType::Google {
            return Err(Error::ProviderNotSupported(
                "Model is not from Google".to_string(),
            ));
        }

        let request = self.build_request(conversation, config, false)?;

        let response = self
            .http_client
            .post(self.interactions_url())
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(response_error(response).await);
        }

        let interaction: InteractionResponse = response.json().await?;

        if interaction.status.as_deref() == Some("failed") {
            return Err(interaction.failure());
        }

        let message = Self::parse_steps(&interaction.steps);
        let status = interaction.status.as_deref().unwrap_or("completed");
        let trailing_tool_call = interaction
            .steps
            .last()
            .and_then(|step| step.get("type"))
            .and_then(Value::as_str)
            == Some("function_call");

        let usage_data = interaction.usage.clone().unwrap_or_default();
        Ok(Self::generation_response(
            message,
            status,
            trailing_tool_call,
            &usage_data,
            interaction.id.clone(),
            &conversation.model,
            config,
        ))
    }

    async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        if conversation.model.provider != ProviderType::Google {
            return Err(Error::ProviderNotSupported(
                "Model is not from Google".to_string(),
            ));
        }

        let request = self.build_request(conversation, config, true)?;
        let url = format!("{}?alt=sse", self.interactions_url());

        let response = self
            .http_client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(response_error(response).await);
        }

        Ok(Self::stream_events(
            response.bytes_stream(),
            conversation.model.clone(),
            config.clone(),
        ))
    }
}

impl GoogleClient {
    /// Parses an Interactions API SSE byte stream. `End` is emitted only after
    /// the interaction reports a terminal status; a stream that ends earlier
    /// yields [`Error::StreamTruncated`].
    fn stream_events<S, B, E>(
        bytes: S,
        model: crate::model::Model,
        config: GenerationConfig,
    ) -> BoxStream<'static, Result<StreamEvent, Error>>
    where
        S: futures::Stream<Item = Result<B, E>> + Send + 'static,
        B: AsRef<[u8]> + Send + 'static,
        E: std::fmt::Display + Send + 'static,
    {
        let emit_reasoning = Self::requests_thinking_summaries(&config);

        let stream = try_stream! {
            let mut event_stream = Box::pin(bytes.eventsource());
            let mut full_text = String::new();
            let mut reasoning_fragments: Vec<GoogleReasoningFragment> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_call_index = 0usize;
            let mut current_thought: Option<PendingThought> = None;
            let mut pending_tool: Option<PendingFunctionCall> = None;
            let mut last_step_type = String::new();
            let mut interaction_id = String::new();
            let mut final_usage: Option<InteractionUsage> = None;
            let mut terminal_status: Option<String> = None;

            while let Some(event) = event_stream.next().await {
                let event =
                    event.map_err(|error| Error::ApiError(format!("SSE error: {error}")))?;

                if event.data == "[DONE]" {
                    break;
                }

                let value: Value =
                    serde_json::from_str(&event.data).map_err(Error::JsonError)?;

                if let Some(total_usage) = value.pointer("/metadata/total_usage")
                    && !total_usage.is_null()
                    && let Ok(parsed) =
                        serde_json::from_value::<InteractionUsage>(total_usage.clone())
                {
                    final_usage = Some(parsed);
                }

                match value
                    .get("event_type")
                    .or_else(|| value.get("type"))
                    .and_then(Value::as_str)
                {
                    Some("interaction.created")
                    | Some("interaction.in_progress") => {
                        if let Some(id) = value
                            .pointer("/interaction/id")
                            .and_then(Value::as_str)
                        {
                            interaction_id = id.to_string();
                        }
                    }
                    Some("step.start") => {
                        let step = &value["step"];
                        last_step_type = step
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();

                        match last_step_type.as_str() {
                            // A step can start with its first content: text on
                            // `model_output`, a summary on `thought`.
                            // https://ai.google.dev/gemini-api/docs/interactions/thinking
                            "model_output" => {
                                let mut text = String::new();
                                Self::collect_content_text(step.get("content"), &mut text);
                                if !text.is_empty() {
                                    full_text.push_str(&text);
                                    yield StreamEvent::TextChunk(text);
                                }
                            }
                            "thought" => {
                                let mut summary = String::new();
                                Self::collect_content_text(step.get("summary"), &mut summary);
                                current_thought = Some(PendingThought {
                                    summary: summary.clone(),
                                    signature: step
                                        .get("signature")
                                        .and_then(Value::as_str)
                                        .map(String::from),
                                });
                                if emit_reasoning && !summary.is_empty() {
                                    yield StreamEvent::ReasoningChunk(summary);
                                }
                            }
                            "function_call" => {
                                let name = step
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();

                                pending_tool = Some(PendingFunctionCall {
                                    id: step
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .map(String::from),
                                    name: name.clone(),
                                    signature: step
                                        .get("signature")
                                        .and_then(Value::as_str)
                                        .map(String::from),
                                    partial_arguments: String::new(),
                                    initial_arguments: step
                                        .get("arguments")
                                        .cloned()
                                        .filter(|arguments| !arguments.is_null()),
                                });

                                yield StreamEvent::ToolCallStart {
                                    index: tool_call_index,
                                    name,
                                };
                            }
                            _ => {}
                        }
                    }
                    Some("step.delta") => {
                        let delta = &value["delta"];

                        match delta.get("type").and_then(Value::as_str) {
                            Some("text") => {
                                let text = delta
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or("");

                                full_text.push_str(text);

                                yield StreamEvent::TextChunk(text.to_string());
                            }
                            Some("thought_summary")
                            | Some("thought") => {
                                let text = delta
                                    .pointer("/content/text")
                                    .and_then(Value::as_str)
                                    .or_else(|| {
                                        delta
                                            .get("text")
                                            .and_then(Value::as_str)
                                    })
                                    .unwrap_or("")
                                    .to_string();

                                if let Some(thought) = &mut current_thought {
                                    thought.summary.push_str(&text);
                                } else {
                                    current_thought = Some(PendingThought {
                                        summary: text.clone(),
                                        signature: None,
                                    });
                                }

                                if emit_reasoning {
                                    yield StreamEvent::ReasoningChunk(text);
                                }
                            }
                            Some("thought_signature") => {
                                let signature = delta
                                    .get("signature")
                                    .and_then(Value::as_str)
                                    .map(String::from);

                                if let Some(thought) = &mut current_thought {
                                    thought.signature = signature;
                                } else {
                                    current_thought = Some(PendingThought {
                                        summary: String::new(),
                                        signature,
                                    });
                                }
                            }
                            Some("function_call_signature") => {
                                if let Some(pending) = &mut pending_tool {
                                    pending.signature = delta
                                        .get("signature")
                                        .and_then(Value::as_str)
                                        .map(String::from);
                                }
                            }
                            Some("arguments_delta")
                            | Some("arguments") => {
                                let partial = delta
                                    .get("arguments")
                                    .and_then(Value::as_str)
                                    .or_else(|| {
                                        delta
                                            .get("partial_arguments")
                                            .and_then(Value::as_str)
                                    })
                                    .unwrap_or("")
                                    .to_string();

                                if let Some(pending) = &mut pending_tool {
                                    pending.partial_arguments.push_str(&partial);
                                }

                                yield StreamEvent::ToolCallArgumentsDelta {
                                    index: tool_call_index,
                                    delta: partial,
                                };
                            }
                            _ => {}
                        }
                    }
                    Some("step.stop") => {
                        if let Some(usage_value) = value.get("usage")
                            && !usage_value.is_null()
                            && let Ok(parsed) =
                                serde_json::from_value::<InteractionUsage>(
                                    usage_value.clone(),
                                )
                        {
                            final_usage = Some(parsed);
                        }

                        if let Some(thought) = current_thought.take() {
                            reasoning_fragments.push(thought.into_fragment());
                        }

                        if let Some(pending) = pending_tool.take() {
                            tool_calls.push(pending.into_tool_call());
                            tool_call_index += 1;
                        }
                    }
                    Some(event_type @ ("interaction.completed" | "interaction.requires_action")) => {
                        let interaction = &value["interaction"];

                        if let Some(id) = interaction
                            .get("id")
                            .and_then(Value::as_str)
                        {
                            interaction_id = id.to_string();
                        }

                        if let Some(usage_value) = interaction.get("usage")
                            && !usage_value.is_null()
                            && let Ok(parsed) =
                                serde_json::from_value::<InteractionUsage>(
                                    usage_value.clone(),
                                )
                        {
                            final_usage = Some(parsed);
                        }

                        let default_status = if event_type == "interaction.requires_action" {
                            "requires_action"
                        } else {
                            "completed"
                        };
                        let status = interaction
                            .get("status")
                            .and_then(Value::as_str)
                            .unwrap_or(default_status);
                        if status == "failed" {
                            Err(interaction_error(
                                "Google Interactions stream completed with a failed interaction",
                                first_error(interaction).or_else(|| first_error(&value)),
                                &full_text,
                            ))?;
                        }
                        terminal_status = Some(status.to_string());

                        break;
                    }
                    Some("interaction.status_update") => {
                        if let Some(id) = value
                            .get("interaction_id")
                            .and_then(Value::as_str)
                        {
                            interaction_id = id.to_string();
                        }

                        match value
                            .get("status")
                            .and_then(Value::as_str)
                        {
                            Some("failed") => {
                                Err(interaction_error(
                                    "Google Interactions stream reported a failed interaction",
                                    first_error(&value)
                                        .or_else(|| value.get("interaction").and_then(first_error)),
                                    &full_text,
                                ))?;

                                break;
                            }
                            Some(status @ ("completed"
                            | "requires_action"
                            | "incomplete"
                            | "cancelled")) => {
                                terminal_status = Some(status.to_string());
                                break;
                            }
                            _ => {}
                        }
                    }
                    Some("interaction.failed") => {
                        Err(interaction_error(
                            "Google Interactions stream failed",
                            value.get("interaction").and_then(first_error),
                            &full_text,
                        ))?;

                        break;
                    }
                    Some("error") => {
                        Err(interaction_error(
                            "Google Interactions stream error",
                            first_error(&value),
                            &full_text,
                        ))?;

                        break;
                    }
                    _ => {}
                }
            }

            let status = terminal_status.ok_or_else(|| {
                Error::StreamTruncated(
                    "Google Interactions stream ended before the interaction finished".to_string(),
                )
            })?;

            if let Some(thought) = current_thought.take() {
                reasoning_fragments.push(thought.into_fragment());
            }

            let trailing_tool_call = last_step_type == "function_call" && pending_tool.is_none();

            // A call still being written when the interaction ended incomplete was cut off.
            if let Some(pending) = pending_tool.take()
                && status != "incomplete"
            {
                tool_calls.push(pending.into_tool_call());
            }

            let message = Message {
                role: Role::Assistant,
                content: vec![Content::Text(full_text)],
                reasoning_artifact: (!reasoning_fragments.is_empty())
                    .then(|| ReasoningArtifact::new_google_reasoning(reasoning_fragments)),
                tool_calls,
                id: None,
                configuration_update: None,
            };

            yield StreamEvent::End(Box::new(GoogleClient::generation_response(
                message,
                &status,
                trailing_tool_call,
                &final_usage.unwrap_or_default(),
                interaction_id,
                &model,
                &config,
            )));
        };

        stream.boxed()
    }

    fn generation_response(
        mut message: Message,
        status: &str,
        trailing_tool_call: bool,
        usage_data: &InteractionUsage,
        interaction_id: String,
        model: &crate::model::Model,
        config: &GenerationConfig,
    ) -> GenerationResponse {
        let mut usage = Self::calculate_usage(usage_data);
        usage.calculate_cost(model, Some(config));

        let model_call_usage = ModelCallUsage::new(
            interaction_id,
            model.provider.as_telemetry_str().to_string(),
            model.id.clone(),
            usage.input_tokens,
            usage.output_tokens,
            model.max_input_tokens,
        )
        .with_stop_metadata(Some(status.to_string()), None);

        let stop_reason = Self::finish_turn(&mut message, status, trailing_tool_call);

        GenerationResponse {
            message,
            usage,
            model_call_usage,
            stop_reason,
            refusal: None,
        }
    }

    /// Maps an interaction status. The Interactions API reports only
    /// `status` (in_progress, requires_action, completed, failed, cancelled,
    /// incomplete) with no finish or safety reason; `incomplete` is
    /// documented as "e.g. hitting max_tokens", so it maps to `MaxTokens`.
    /// https://ai.google.dev/api/interactions-api-v1
    fn finish_turn(message: &mut Message, status: &str, trailing_tool_call: bool) -> StopReason {
        match status {
            "completed" | "requires_action" if !message.tool_calls.is_empty() => {
                StopReason::ToolUse
            }
            "completed" => StopReason::EndTurn,
            "incomplete" => {
                if trailing_tool_call {
                    message.tool_calls.pop();
                }
                StopReason::MaxTokens
            }
            other => StopReason::Other(other.to_string()),
        }
    }

    /// Per-model defaults, checked 2026-09-27 against
    /// https://ai.google.dev/gemini-api/docs/thinking
    fn default_thinking_level(model: &crate::model::Model) -> GeminiThinkingLevel {
        if model.is_gemini_3_5_flash()
            || model.is_gemini_3_6_flash()
            || model.is_gemini_3_7_flash()
            || model.is_gemini_3_8_flash()
        {
            GeminiThinkingLevel::Medium
        } else if model.is_gemini_3_1_flash_lite() || model.is_gemini_3_5_flash_lite() {
            GeminiThinkingLevel::Minimal
        } else {
            GeminiThinkingLevel::High
        }
    }

    fn thinking_level_request_value(level: GeminiThinkingLevel) -> &'static str {
        match level {
            GeminiThinkingLevel::Minimal => "minimal",
            GeminiThinkingLevel::Low => "low",
            GeminiThinkingLevel::Medium => "medium",
            GeminiThinkingLevel::High => "high",
        }
    }

    fn thinking_level_for_budget(budget: isize) -> Option<GeminiThinkingLevel> {
        if budget < 0 {
            None
        } else if budget == 0 {
            Some(GeminiThinkingLevel::Minimal)
        } else if budget <= 4_096 {
            Some(GeminiThinkingLevel::Low)
        } else if budget <= 16_384 {
            Some(GeminiThinkingLevel::Medium)
        } else {
            Some(GeminiThinkingLevel::High)
        }
    }

    fn requests_thinking_summaries(config: &GenerationConfig) -> bool {
        match &config.reasoning {
            Some(ReasoningConfig::Gemini {
                include_summaries, ..
            })
            | Some(ReasoningConfig::Google {
                include_summaries, ..
            }) => *include_summaries,
            _ => false,
        }
    }

    fn build_request(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        stream: bool,
    ) -> Result<InteractionsRequest, Error> {
        super::ensure_supported_inputs(conversation)?;
        if let Some(tool_config) = &config.tool_config
            && tool_config.parallel_tool_calls == Some(false)
        {
            return Err(Error::UnsupportedFeature(
                "Google Gemini does not support disabling parallel tool calls.".to_string(),
            ));
        }

        // temperature, top_p and top_k are no longer part of the Interactions
        // API generation_config (deprecated 2026-07-21, absent from the v1
        // spec). Reject rather than silently ignore an explicit value.
        // https://ai.google.dev/gemini-api/docs/changelog
        // https://ai.google.dev/static/api/interactions-v1.openapi.json
        if config.temperature_specified {
            return Err(Error::UnsupportedFeature(
                "The Gemini Interactions API does not accept a temperature.".to_string(),
            ));
        }

        let (thinking_level, thinking_summaries) = self.resolve_thinking(conversation, config)?;

        let tool_choice = config
            .tool_config
            .as_ref()
            .and_then(|tool_config| tool_config.tool_choice.as_ref())
            .map(|choice| {
                let (mode, tools) = match choice {
                    ToolChoice::Auto => ("auto", None),
                    ToolChoice::Required => ("any", None),
                    ToolChoice::None => ("none", None),
                    ToolChoice::Specific(name) => ("any", Some(vec![name.clone()])),
                };

                InteractionsToolChoice {
                    allowed_tools: InteractionsAllowedTools {
                        mode: mode.to_string(),
                        tools,
                    },
                }
            });

        let generation_config = InteractionsGenerationConfig {
            max_output_tokens: Some(
                config
                    .max_tokens
                    .unwrap_or(conversation.model.max_output_tokens),
            ),
            thinking_level,
            thinking_summaries,
            tool_choice,
        };

        let tools = config.tool_config.as_ref().map(|tool_config| {
            tool_config
                .tools
                .iter()
                .map(|tool| InteractionsFunctionTool {
                    type_: "function".to_string(),
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.parameters.clone(),
                })
                .collect()
        });

        Ok(InteractionsRequest {
            model: conversation.model.id.clone(),
            store: false,
            stream: stream.then_some(true),
            system_instruction: conversation.system_message.clone(),
            input: self.convert_messages(conversation)?,
            generation_config: Some(generation_config),
            tools,
        })
    }

    fn resolve_thinking(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<(Option<String>, Option<String>), Error> {
        let summaries_value =
            |requested: bool| Some(if requested { "auto" } else { "none" }.to_string());

        if conversation.model.uses_gemini_thinking_levels() {
            match &config.reasoning {
                Some(ReasoningConfig::Gemini {
                    thinking_level,
                    include_summaries,
                }) => {
                    self.validate_gemini_thinking_level(&conversation.model, *thinking_level)?;

                    Ok((
                        Some(Self::thinking_level_request_value(*thinking_level).to_string()),
                        summaries_value(*include_summaries),
                    ))
                }
                Some(ReasoningConfig::Google {
                    thinking_budget,
                    include_summaries,
                }) => {
                    let level = Self::thinking_level_for_budget(*thinking_budget);
                    if let Some(level) = level {
                        self.validate_gemini_thinking_level(&conversation.model, level)?;
                    }

                    Ok((
                        level.map(|level| Self::thinking_level_request_value(level).to_string()),
                        summaries_value(*include_summaries),
                    ))
                }
                _ => Ok((
                    Some(
                        Self::thinking_level_request_value(Self::default_thinking_level(
                            &conversation.model,
                        ))
                        .to_string(),
                    ),
                    summaries_value(false),
                )),
            }
        } else {
            let thinking_config = matches!(
                config.reasoning,
                Some(ReasoningConfig::Google { .. } | ReasoningConfig::Gemini { .. })
            );
            if thinking_config && conversation.model.model_type != ModelType::Reasoning {
                return Err(Error::InvalidRequest(format!(
                    "Model '{}' does not support thinking.",
                    conversation.model.id
                )));
            }
            match &config.reasoning {
                Some(ReasoningConfig::Google {
                    thinking_budget,
                    include_summaries,
                }) => Ok((
                    Self::budget_level_without_thinking_levels(&conversation.model, *thinking_budget)?
                        .map(|level| Self::thinking_level_request_value(level).to_string()),
                    summaries_value(*include_summaries),
                )),
                // Gemini 2.5 on the Interactions API takes LOW, MEDIUM and HIGH.
                // https://ai.google.dev/gemini-api/docs/thinking
                Some(ReasoningConfig::Gemini {
                    thinking_level,
                    include_summaries,
                }) if conversation.model.is_gemini_2_5() => {
                    if matches!(thinking_level, GeminiThinkingLevel::Minimal) {
                        return Err(Error::InvalidRequest(format!(
                            "Gemini 2.5 models only support LOW, MEDIUM, and HIGH thinking levels, got {}.",
                            thinking_level.as_str()
                        )));
                    }
                    Ok((
                        Some(Self::thinking_level_request_value(*thinking_level).to_string()),
                        summaries_value(*include_summaries),
                    ))
                }
                Some(ReasoningConfig::Gemini { .. }) => Err(Error::InvalidRequest(format!(
                    "Thinking levels are not supported for model '{}'.",
                    conversation.model.id
                ))),
                _ => Ok((None, None)),
            }
        }
    }

    /// Gemini 2.5 models on the Interactions API accept only LOW, MEDIUM and
    /// HIGH, so a zero budget (thinking off) has no level. 2.5 Pro can't turn
    /// thinking off and 2.5 Flash has no documented off switch there, so they
    /// are rejected; 2.5 Flash-Lite thinks only when asked, so the level is
    /// omitted. https://ai.google.dev/gemini-api/docs/thinking
    fn budget_level_without_thinking_levels(
        model: &crate::model::Model,
        budget: isize,
    ) -> Result<Option<GeminiThinkingLevel>, Error> {
        match Self::thinking_level_for_budget(budget) {
            Some(GeminiThinkingLevel::Minimal) if model.is_gemini_2_5_flash_lite() => Ok(None),
            Some(GeminiThinkingLevel::Minimal)
                if model.is_gemini_2_5_pro() || model.is_gemini_2_5_flash() =>
            {
                Err(Error::InvalidRequest(format!(
                    "Thinking can't be disabled for '{}' on the Interactions API; use a positive budget, or -1 for the model default.",
                    model.id
                )))
            }
            level => Ok(level),
        }
    }

    fn validate_gemini_thinking_level(
        &self,
        model: &crate::model::Model,
        level: GeminiThinkingLevel,
    ) -> Result<(), Error> {
        if model.is_gemini_3_pro()
            && matches!(
                level,
                GeminiThinkingLevel::Minimal | GeminiThinkingLevel::Medium
            )
        {
            return Err(Error::InvalidRequest(format!(
                "Gemini 3 Pro only supports HIGH and LOW thinking levels, got {}.",
                level.as_str()
            )));
        }

        if model.is_gemini_3_1_pro() && matches!(level, GeminiThinkingLevel::Minimal) {
            return Err(Error::InvalidRequest(format!(
                "Gemini 3.1 Pro only supports LOW, MEDIUM, and HIGH thinking levels, got {}.",
                level.as_str()
            )));
        }

        // Gemini 3.7 and 3.8 Flash return an error for MINIMAL.
        // https://ai.google.dev/gemini-api/docs/thinking
        if (model.is_gemini_3_7_flash() || model.is_gemini_3_8_flash())
            && matches!(level, GeminiThinkingLevel::Minimal)
        {
            return Err(Error::InvalidRequest(format!(
                "Model '{}' only supports LOW, MEDIUM, and HIGH thinking levels, got {}.",
                model.id,
                level.as_str()
            )));
        }

        Ok(())
    }

    fn convert_messages(
        &self,
        conversation: &Conversation,
    ) -> Result<Vec<InteractionInputStep>, Error> {
        let mut steps = Vec::new();

        for message in &conversation.messages {
            match message.role {
                Role::System => {
                    return Err(Error::InvalidRequest(
                        "System messages should be set in the system_instruction field."
                            .to_string(),
                    ));
                }
                Role::User => {
                    let mut blocks = Vec::new();

                    for content in &message.content {
                        match content {
                            Content::Text(text) => {
                                blocks.push(InteractionContentBlock::Text { text: text.clone() });
                            }
                            Content::Image(image) => {
                                blocks.push(self.convert_image_block(image)?);
                            }
                            Content::ToolResult(result) => {
                                steps.push(InteractionInputStep::FunctionResult {
                                    name: result.name.clone(),
                                    call_id: result.call_id.clone().unwrap_or_default(),
                                    is_error: result.is_error.then_some(true),
                                    result: vec![InteractionContentBlock::Text {
                                        text: result.output.clone(),
                                    }],
                                });
                            }
                        }
                    }

                    if !blocks.is_empty() {
                        steps.push(InteractionInputStep::UserInput { content: blocks });
                    }
                }
                Role::Assistant => {
                    if let Some(artifact) = &message.reasoning_artifact {
                        match artifact {
                            ReasoningArtifact::GoogleReasoning { fragments } => {
                                for fragment in fragments {
                                    let summary = if fragment.summary.is_empty() {
                                        Vec::new()
                                    } else {
                                        vec![InteractionContentBlock::Text {
                                            text: fragment.summary.clone(),
                                        }]
                                    };

                                    steps.push(InteractionInputStep::Thought {
                                        summary,
                                        signature: (!fragment.signature.is_empty())
                                            .then(|| fragment.signature.clone()),
                                    });
                                }
                            }
                            _ => {
                                return Err(Error::InvalidRequest(
                                    "Invalid reasoning artifact type for Google.".to_string(),
                                ));
                            }
                        }
                    }

                    let mut blocks = Vec::new();

                    for content in &message.content {
                        match content {
                            Content::Text(text) => {
                                if !text.trim().is_empty() {
                                    blocks
                                        .push(InteractionContentBlock::Text { text: text.clone() });
                                }
                            }
                            Content::Image(image) => {
                                blocks.push(self.convert_image_block(image)?);
                            }
                            Content::ToolResult(result) => {
                                steps.push(InteractionInputStep::FunctionResult {
                                    name: result.name.clone(),
                                    call_id: result.call_id.clone().unwrap_or_default(),
                                    is_error: result.is_error.then_some(true),
                                    result: vec![InteractionContentBlock::Text {
                                        text: result.output.clone(),
                                    }],
                                });
                            }
                        }
                    }

                    if !blocks.is_empty() {
                        steps.push(InteractionInputStep::ModelOutput { content: blocks });
                    }

                    for tool_call in &message.tool_calls {
                        steps.push(InteractionInputStep::FunctionCall {
                            id: tool_call.id.clone().unwrap_or_default(),
                            signature: tool_call
                                .thought_signature
                                .clone()
                                .filter(|signature| !signature.is_empty()),
                            name: tool_call.name.clone(),
                            arguments: tool_call.replay_arguments(),
                        });
                    }
                }
            }
        }

        Ok(steps)
    }

    fn convert_image_block(&self, image: &Image) -> Result<InteractionContentBlock, Error> {
        match image {
            Image::Url(_) => Err(Error::UnsupportedFeature(
                "Google does not support image URLs, only base64 data.".to_string(),
            )),
            Image::Base64 { data, mime_type } => Ok(InteractionContentBlock::Image {
                mime_type: mime_type.clone(),
                data: data.clone(),
            }),
        }
    }

    fn parse_steps(steps: &[Value]) -> Message {
        let mut text = String::new();
        let mut reasoning_fragments = Vec::new();
        let mut tool_calls = Vec::new();

        for step in steps {
            match step.get("type").and_then(Value::as_str) {
                Some("model_output") => {
                    Self::collect_content_text(step.get("content"), &mut text);
                }
                Some("thought") => {
                    let mut thought_text = String::new();

                    Self::collect_content_text(
                        step.get("summary").or_else(|| step.get("content")),
                        &mut thought_text,
                    );

                    reasoning_fragments.push(GoogleReasoningFragment {
                        summary: thought_text,
                        signature: step
                            .get("signature")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    });
                }
                Some("function_call") => {
                    let id = step.get("id").and_then(Value::as_str).map(String::from);
                    let name = step
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let mut call = match step.get("arguments") {
                        Some(Value::String(text)) => ToolCall::from_argument_text(id, name, text),
                        Some(arguments) if !arguments.is_null() => {
                            let mut call = ToolCall::from_argument_text(id, name, "");
                            call.arguments = arguments.clone();
                            call
                        }
                        _ => ToolCall::from_argument_text(id, name, ""),
                    };
                    call.thought_signature = step
                        .get("signature")
                        .and_then(Value::as_str)
                        .filter(|signature| !signature.is_empty())
                        .map(String::from);
                    tool_calls.push(call);
                }
                _ => {}
            }
        }

        let reasoning_artifact = if reasoning_fragments.is_empty() {
            None
        } else {
            Some(ReasoningArtifact::new_google_reasoning(reasoning_fragments))
        };

        Message {
            role: Role::Assistant,
            content: vec![Content::Text(text)],
            reasoning_artifact,
            tool_calls,
            id: None,
            configuration_update: None,
        }
    }

    fn collect_content_text(content: Option<&Value>, buffer: &mut String) {
        match content {
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    if let Some(block_text) = block.get("text").and_then(Value::as_str) {
                        buffer.push_str(block_text);
                    }
                }
            }
            Some(Value::Object(block)) => {
                if let Some(block_text) = block.get("text").and_then(Value::as_str) {
                    buffer.push_str(block_text);
                }
            }
            Some(Value::String(content)) => {
                buffer.push_str(content);
            }
            _ => {}
        }
    }

    /// `total_input_tokens` includes cached input. Output is visible output
    /// plus thought tokens, both billed as output. `total_tokens` is not used
    /// because it also counts other internal tokens such as tool-use prompts.
    /// https://ai.google.dev/api/interactions-api-v1
    fn calculate_usage(usage: &InteractionUsage) -> UsageInfo {
        let input_tokens = usage.total_input_tokens as usize;
        let output_tokens = (usage.total_output_tokens + usage.total_thought_tokens) as usize;

        let mut info = UsageInfo::new(input_tokens, output_tokens);
        info.reasoning_tokens =
            (usage.total_thought_tokens > 0).then_some(usage.total_thought_tokens as usize);

        if let Some(cached_tokens) = usage.total_cached_tokens
            && cached_tokens > 0
        {
            info = info.with_cached_tokens(cached_tokens as usize);
        }

        info
    }
}

/// Gemini codes for output blocked by policy, safety or content
/// restrictions. They arrive in the standard error object of HTTP and SSE
/// responses. https://ai.google.dev/gemini-api/docs/api-errors
const GENERATION_BLOCKED_CODES: &[&str] = &[
    "safety",
    "recitation",
    "language",
    "prohibited_content",
    "spii",
    "blocklist",
    "image_safety",
    "image_prohibited_content",
    "image_recitation",
    "image_other",
    "content_blocked",
];

/// A content-filter refusal for a generation-blocked error, with the code as
/// its category.
fn blocked_generation(error: &Value) -> Option<Refusal> {
    let code = error.get("code").and_then(Value::as_str)?;
    GENERATION_BLOCKED_CODES.contains(&code).then(|| {
        let mut refusal = Refusal::new(RefusalSource::ContentFilter);
        refusal.category = Some(code.to_string());
        refusal.explanation = error.get("message").and_then(Value::as_str).map(str::to_owned);
        refusal.provider_details = error.clone();
        refusal
    })
}

/// The error recorded on a failed interaction or error event: the v1
/// `Interaction.errors` list (the first entry classifies it), or `error`.
/// https://ai.google.dev/static/api/interactions-v1.openapi.json
fn first_error(value: &Value) -> Option<&Value> {
    value
        .get("errors")
        .and_then(Value::as_array)
        .and_then(|errors| errors.first())
        .or_else(|| value.get("error"))
        .filter(|error| !error.is_null())
}

/// Classifies an interaction or stream error. A generation-blocked code is a
/// [`Error::Refusal`] (keeping any text streamed before it as
/// `partial_output`); otherwise the `code` (string or numeric HTTP-style) and
/// `status` (e.g. `UNAVAILABLE`) decide. An error without either stays a
/// generic, boundedly retryable API error.
fn interaction_error(label: &str, error: Option<&Value>, partial_output: &str) -> Error {
    let Some(error) = error else {
        return Error::ApiError(format!("{label}: unknown error"));
    };
    if let Some(mut refusal) = blocked_generation(error) {
        refusal.partial_output = (!partial_output.is_empty()).then(|| partial_output.to_string());
        return Error::Refusal(Box::new(refusal));
    }
    if error.get("code").is_some() || error.get("status").is_some() {
        Error::provider_error(label, None, None, &serde_json::json!({ "error": error }).to_string())
    } else {
        Error::ApiError(format!("{label}: {error}"))
    }
}

/// An HTTP error response, with generation-blocked codes as refusals.
async fn response_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    let retry_after = crate::error::retry_after_from_headers(status, response.headers());
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<unavailable error body>".into());
    http_error(status, retry_after, &body)
}

fn http_error(status: u16, retry_after: Option<std::time::Duration>, body: &str) -> Error {
    let value = serde_json::from_str::<Value>(body).unwrap_or_default();
    if let Some(refusal) = first_error(&value).and_then(blocked_generation) {
        return Error::Refusal(Box::new(refusal));
    }
    Error::provider_error("google API error", Some(status), retry_after, body)
}

struct PendingThought {
    summary: String,
    signature: Option<String>,
}

impl PendingThought {
    fn into_fragment(self) -> GoogleReasoningFragment {
        GoogleReasoningFragment {
            summary: self.summary,
            signature: self.signature.unwrap_or_default(),
        }
    }
}

struct PendingFunctionCall {
    id: Option<String>,
    name: String,
    signature: Option<String>,
    partial_arguments: String,
    initial_arguments: Option<Value>,
}

impl PendingFunctionCall {
    fn into_tool_call(self) -> ToolCall {
        let mut call = if self.partial_arguments.is_empty() {
            let mut call = ToolCall::from_argument_text(self.id, self.name, "");
            if let Some(arguments) = self.initial_arguments {
                call.arguments = arguments;
            }
            call
        } else {
            ToolCall::from_argument_text(self.id, self.name, &self.partial_arguments)
        };
        call.thought_signature = self.signature.filter(|signature| !signature.is_empty());
        call
    }
}

#[derive(Debug, Serialize)]
struct InteractionsRequest {
    model: String,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<String>,
    input: Vec<InteractionInputStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<InteractionsGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<InteractionsFunctionTool>>,
}

#[derive(Debug, Serialize)]
struct InteractionsGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_summaries: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<InteractionsToolChoice>,
}

#[derive(Debug, Serialize)]
struct InteractionsToolChoice {
    allowed_tools: InteractionsAllowedTools,
}

#[derive(Debug, Serialize)]
struct InteractionsAllowedTools {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct InteractionsFunctionTool {
    #[serde(rename = "type")]
    type_: String,
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InteractionInputStep {
    UserInput {
        content: Vec<InteractionContentBlock>,
    },
    Thought {
        #[serde(skip_serializing_if = "Vec::is_empty")]
        summary: Vec<InteractionContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ModelOutput {
        content: Vec<InteractionContentBlock>,
    },
    FunctionCall {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        name: String,
        arguments: Value,
    },
    FunctionResult {
        name: String,
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        result: Vec<InteractionContentBlock>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InteractionContentBlock {
    Text { text: String },
    Image { mime_type: String, data: String },
}

#[derive(Debug, Deserialize)]
struct InteractionResponse {
    #[serde(default)]
    id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    steps: Vec<Value>,
    #[serde(default)]
    usage: Option<InteractionUsage>,
    /// Errors recorded on a failed interaction (v1 `errors`).
    #[serde(default)]
    errors: Vec<Value>,
    /// Single-error form, kept for tolerance.
    #[serde(default)]
    error: Option<Value>,
}

impl InteractionResponse {
    fn failure(&self) -> Error {
        interaction_error(
            "Google Interactions API returned a failed interaction",
            self.errors
                .first()
                .or(self.error.as_ref())
                .filter(|error| !error.is_null()),
            "",
        )
    }
}

#[derive(Debug, Deserialize, Default, Clone)]
struct InteractionUsage {
    #[serde(default)]
    total_input_tokens: u64,
    #[serde(default)]
    total_output_tokens: u64,
    #[serde(default)]
    total_thought_tokens: u64,
    #[serde(default)]
    total_cached_tokens: Option<u64>,
}

pub fn list_models() -> Vec<crate::model::Model> {
    crate::model::Model::list_google_models()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use serde_json::json;

    async fn stream_events(events: &[Value]) -> (Vec<StreamEvent>, Option<Error>) {
        let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
        stream_body(body).await
    }

    async fn stream_body(body: String) -> (Vec<StreamEvent>, Option<Error>) {
        let bytes = futures::stream::iter([Ok::<_, std::io::Error>(body)]);
        let mut stream = GoogleClient::stream_events(
            bytes,
            Model::google_gemini_3_5_flash(),
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

    fn text_step() -> [Value; 3] {
        [
            json!({"event_type":"step.start","index":0,"step":{"type":"model_output"}}),
            json!({"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}),
            json!({"event_type":"step.stop","index":0}),
        ]
    }

    fn call_step(index: u64, id: &str, arguments: &str, stop: bool) -> Vec<Value> {
        let mut events = vec![
            json!({"event_type":"step.start","index":index,"step":{"type":"function_call","id":id,"name":"fixture_tool"}}),
            json!({"event_type":"step.delta","index":index,"delta":{"type":"arguments_delta","arguments":arguments}}),
        ];
        if stop {
            events.push(json!({"event_type":"step.stop","index":index}));
        }
        events
    }

    fn completed(status: &str) -> Value {
        json!({"event_type":"interaction.completed","interaction":{"id":"int_1","status":status,
            "usage":{"total_input_tokens":3,"total_output_tokens":2}}})
    }

    #[test]
    fn usage_counts_thoughts_as_output_and_ignores_other_total_tokens() {
        let usage: InteractionUsage = serde_json::from_value(json!({
            "total_input_tokens": 1000, "total_output_tokens": 200,
            "total_thought_tokens": 300, "total_cached_tokens": 400,
            "total_tool_use_tokens": 50, "total_tokens": 1550
        }))
        .unwrap();
        let info = GoogleClient::calculate_usage(&usage);
        assert_eq!(info.input_tokens, 1000);
        assert_eq!(info.output_tokens, 500);
        assert_eq!(info.reasoning_tokens, Some(300));
        assert_eq!(info.cached_tokens, Some(400));
    }

    #[tokio::test]
    async fn stream_errors_are_classified_by_code_and_status() {
        let (_, error) = stream_events(&[json!({"event_type":"error",
            "error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}})])
        .await;
        let error = error.unwrap();
        assert!(error.is_capacity_error(), "{error:?}");
        let (_, error) = stream_events(&[json!({"event_type":"error",
            "error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}})])
        .await;
        assert!(!error.unwrap().is_retryable());
        // A failure without details stays boundedly retryable.
        let (_, error) = stream_events(&[json!({"event_type":"interaction.failed",
            "interaction":{"id":"int_1","status":"failed"}})])
        .await;
        assert!(matches!(error, Some(Error::ApiError(_))));
    }

    #[tokio::test]
    async fn interaction_error_codes_are_classified_and_blocked_output_is_a_refusal() {
        // SSE error events carry the documented snake_case codes.
        // https://ai.google.dev/gemini-api/docs/api-errors
        for (code, class) in [
            ("service_unavailable", crate::RetryClass::Capacity),
            ("deadline_exceeded", crate::RetryClass::Transient),
            ("api_error", crate::RetryClass::Transient),
            ("quota_exceeded", crate::RetryClass::Permanent),
            ("not_found", crate::RetryClass::Permanent),
        ] {
            let (_, error) = stream_events(&[json!({"event_type":"error",
                "error":{"code":code,"message":"m"}})])
            .await;
            let error = error.unwrap();
            assert!(matches!(error, Error::ProviderError { .. }), "{error:?}");
            assert_eq!(error.retry_class(), class, "{code}");
        }
        // A generation-blocked code after some text is a content-filter
        // refusal that keeps the streamed text aside.
        let mut events = vec![
            json!({"event_type":"step.start","index":0,"step":{"type":"model_output"}}),
            json!({"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Once upon"}}),
        ];
        events.push(json!({"event_type":"error","error":{"code":"recitation","message":"Blocked for recitation."}}));
        let (_, error) = stream_events(&events).await;
        let Some(Error::Refusal(refusal)) = error else {
            panic!("expected a refusal, got {error:?}");
        };
        assert_eq!(refusal.source, RefusalSource::ContentFilter);
        assert_eq!(refusal.category.as_deref(), Some("recitation"));
        assert_eq!(refusal.explanation.as_deref(), Some("Blocked for recitation."));
        assert_eq!(refusal.partial_output.as_deref(), Some("Once upon"));
        assert!(!Error::Refusal(refusal).is_retryable());
        // The same codes in an HTTP error body.
        let error = http_error(400, None, r#"{"error":{"code":"prohibited_content","message":"m"}}"#);
        assert!(matches!(&error, Error::Refusal(refusal)
            if refusal.category.as_deref() == Some("prohibited_content") && refusal.partial_output.is_none()));
        let error = http_error(429, None, r#"{"error":{"code":"rate_limit_exceeded","message":"m"}}"#);
        assert!(matches!(error, Error::ProviderError { status: Some(429), .. }));
    }

    #[tokio::test]
    async fn a_completed_event_with_a_failed_status_is_an_error() {
        let mut events = text_step().to_vec();
        events.push(json!({"event_type":"interaction.completed",
            "interaction":{"id":"int_1","status":"failed"}}));
        let (events, error) = stream_events(&events).await;
        assert!(!events.iter().any(|event| matches!(event, StreamEvent::End(_))));
        assert!(matches!(error, Some(Error::ApiError(_))), "{error:?}");
        let mut events = text_step().to_vec();
        events.push(json!({"event_type":"interaction.completed",
            "interaction":{"id":"int_1","status":"failed",
                "errors":[{"code":"service_unavailable","message":"overloaded"}]}}));
        let (_, error) = stream_events(&events).await;
        assert!(error.unwrap().is_capacity_error());
    }

    #[test]
    fn failed_non_streaming_interactions_read_the_errors_list() {
        let failed = |body: Value| serde_json::from_value::<InteractionResponse>(body).unwrap().failure();
        let error = failed(json!({"id":"int_1","status":"failed","steps":[],
            "errors":[{"code":"service_unavailable","message":"The service is temporarily overloaded."}]}));
        assert!(error.is_capacity_error(), "{error:?}");
        let error = failed(json!({"id":"int_1","status":"failed",
            "errors":[{"code":"safety","message":"Blocked."},{"code":"api_error","message":"x"}]}));
        assert!(matches!(&error, Error::Refusal(refusal) if refusal.category.as_deref() == Some("safety")));
        // The single-error form still works; no details stays a generic error.
        let error = failed(json!({"id":"int_1","status":"failed","error":{"code":"invalid_request","message":"m"}}));
        assert_eq!(error.retry_class(), crate::RetryClass::Permanent);
        assert!(matches!(failed(json!({"id":"int_1","status":"failed"})), Error::ApiError(_)));
    }

    #[tokio::test]
    async fn content_carried_on_step_start_is_kept() {
        // Event shapes from https://ai.google.dev/gemini-api/docs/interactions/thinking
        let events = [
            json!({"index":0,"step":{"signature":"","summary":[{"text":"**Evaluating the clues**","type":"text"}],"type":"thought"},"event_type":"step.start"}),
            json!({"index":0,"delta":{"content":{"text":" further","type":"text"},"type":"thought_summary"},"event_type":"step.delta"}),
            json!({"index":0,"delta":{"signature":"EpoGCpcG","type":"thought_signature"},"event_type":"step.delta"}),
            json!({"index":0,"event_type":"step.stop"}),
            json!({"index":1,"step":{"content":[{"text":"Based on the clues provided, here","type":"text"}],"type":"model_output"},"event_type":"step.start"}),
            json!({"index":1,"delta":{"text":" is the answer.","type":"text"},"event_type":"step.delta"}),
            json!({"index":1,"event_type":"step.stop"}),
            completed("completed"),
        ];
        let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
        let config = GenerationConfig {
            reasoning: Some(ReasoningConfig::Gemini {
                thinking_level: GeminiThinkingLevel::High,
                include_summaries: true,
            }),
            ..GenerationConfig::default()
        };
        let mut stream = GoogleClient::stream_events(
            futures::stream::iter([Ok::<_, std::io::Error>(body)]),
            Model::google_gemini_3_5_flash(),
            config,
        );
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut response = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::TextChunk(chunk) => text.push_str(&chunk),
                StreamEvent::ReasoningChunk(chunk) => reasoning.push_str(&chunk),
                StreamEvent::End(end) => response = Some(end),
                _ => {}
            }
        }
        let response = response.unwrap();
        assert_eq!(text, "Based on the clues provided, here is the answer.");
        assert_eq!(response.text(), text);
        assert_eq!(reasoning, "**Evaluating the clues** further");
        let Some(ReasoningArtifact::GoogleReasoning { fragments }) = &response.message.reasoning_artifact else {
            panic!("expected Google reasoning");
        };
        assert_eq!(fragments[0].summary, "**Evaluating the clues** further");
        assert_eq!(fragments[0].signature, "EpoGCpcG");
    }

    #[tokio::test]
    async fn completed_interactions_end_the_turn_and_record_their_status() {
        let mut events = text_step().to_vec();
        events.push(completed("completed"));
        let (events, error) = stream_events(&events).await;
        assert!(error.is_none());
        let response = end(&events);
        assert_eq!(response.text(), "hi");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("completed"));
    }

    #[tokio::test]
    async fn function_calls_awaiting_results_are_tool_use() {
        let mut events = call_step(0, "call_a", "{\"n\":1}", true);
        events.push(json!({"event_type":"interaction.status_update","interaction_id":"int_1","status":"requires_action"}));
        let (events, _) = stream_events(&events).await;
        let response = end(&events);
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.message.tool_calls[0].arguments, json!({"n":1}));
        assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("requires_action"));
    }

    #[tokio::test]
    async fn incomplete_interactions_drop_cut_off_calls() {
        let mut events = call_step(0, "call_a", "{\"n\":1}", true);
        events.extend(call_step(1, "call_b", "{\"n\":2}", true));
        events.push(completed("incomplete"));
        let (events, _) = stream_events(&events).await;
        let response = end(&events);
        assert_eq!(response.stop_reason, StopReason::MaxTokens);
        let ids: Vec<_> = response.message.tool_calls.iter().map(|call| call.id.as_deref().unwrap()).collect();
        assert_eq!(ids, ["call_a"]);

        let mut events = call_step(0, "call_a", "{\"n\":1}", true);
        events.extend(call_step(1, "call_b", "{\"n\":", false));
        events.push(completed("incomplete"));
        let (events, _) = stream_events(&events).await;
        let ids: Vec<_> = end(&events).message.tool_calls.iter().map(|call| call.id.as_deref().unwrap()).collect();
        assert_eq!(ids, ["call_a"]);
    }

    #[tokio::test]
    async fn streams_without_a_terminal_status_are_truncated() {
        let (events, error) = stream_events(&text_step()).await;
        assert!(!events.iter().any(|event| matches!(event, StreamEvent::End(_))));
        assert!(matches!(error, Some(Error::StreamTruncated(_))));

        let body: String = text_step().iter().map(|event| format!("data: {event}\n\n")).collect();
        let (_, error) = stream_body(body + "data: [DONE]\n\n").await;
        assert!(matches!(error, Some(Error::StreamTruncated(_))));
    }

    #[tokio::test]
    async fn invalid_call_arguments_are_marked() {
        let mut events = call_step(0, "call_a", "{oops", true);
        events.push(completed("requires_action"));
        let (events, _) = stream_events(&events).await;
        let call = &end(&events).message.tool_calls[0];
        assert_eq!(call.invalid_arguments.as_deref(), Some("{oops"));
    }

    #[test]
    fn non_streaming_statuses_map_to_stop_reasons() {
        for (status, expected) in [
            ("completed", StopReason::EndTurn),
            ("incomplete", StopReason::MaxTokens),
            ("cancelled", StopReason::Other("cancelled".into())),
        ] {
            let response = GoogleClient::generation_response(
                GoogleClient::parse_steps(&[json!({"type":"model_output","content":[{"type":"text","text":"x"}]})]),
                status,
                false,
                &InteractionUsage::default(),
                "int_1".into(),
                &Model::google_gemini_3_5_flash(),
                &GenerationConfig::default(),
            );
            assert_eq!(response.stop_reason, expected);
            assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some(status));
        }
    }

    #[test]
    fn empty_signatures_are_not_sent() {
        let mut conversation = Conversation::new(Model::google_gemini_3_5_flash());
        conversation.add_user_message("go");
        let mut message = Message::new_assistant("");
        message.reasoning_artifact = Some(ReasoningArtifact::new_google_reasoning(vec![
            GoogleReasoningFragment { summary: "thought".into(), signature: String::new() },
        ]));
        let mut call = ToolCall::from_argument_text(Some("call_a".into()), "fixture_tool".into(), "{}");
        call.thought_signature = Some(String::new());
        message.tool_calls.push(call);
        conversation.messages.push(message);
        let steps = serde_json::to_value(
            GoogleClient::new("test").convert_messages(&conversation).unwrap(),
        )
        .unwrap();
        for step in steps.as_array().unwrap() {
            assert!(step.get("signature").is_none(), "{step}");
        }
    }

    #[test]
    fn budget_mapped_levels_are_validated_for_the_model() {
        let conversation = Conversation::new(Model::google_gemini_3_0_pro());
        let config = GenerationConfig::default().with_reasoning(ReasoningConfig::google(10_000, false));
        assert!(matches!(
            GoogleClient::new("test").build_request(&conversation, &config, false),
            Err(Error::InvalidRequest(_))
        ));
        let config = GenerationConfig::default().with_reasoning(ReasoningConfig::google(20_000, false));
        let request = GoogleClient::new("test").build_request(&conversation, &config, false).unwrap();
        assert_eq!(
            serde_json::to_value(request).unwrap()["generation_config"]["thinking_level"],
            "high"
        );

        // Gemini 3.1 Pro rejects MINIMAL, which a zero budget maps to.
        let conversation = Conversation::new(Model::google_gemini_3_1_pro());
        let config = GenerationConfig::default().with_reasoning(ReasoningConfig::google(0, false));
        assert!(matches!(
            GoogleClient::new("test").build_request(&conversation, &config, false),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn gemini_2_5_budgets_never_map_to_minimal() {
        let client = GoogleClient::new("test");
        let zero = GenerationConfig::default().with_reasoning(ReasoningConfig::google(0, false));
        for model in [Model::google_gemini_2_5_pro(), Model::google_gemini_2_5_flash()] {
            let conversation = Conversation::new(model);
            assert!(matches!(
                client.build_request(&conversation, &zero, false),
                Err(Error::InvalidRequest(_))
            ));
        }

        let conversation = Conversation::new(Model::google_gemini_2_5_flash_lite());
        let request = client.build_request(&conversation, &zero, false).unwrap();
        assert!(serde_json::to_value(request).unwrap()["generation_config"].get("thinking_level").is_none());

        let conversation = Conversation::new(Model::google_gemini_2_5_pro());
        let config = GenerationConfig::default().with_reasoning(ReasoningConfig::google(2_000, false));
        let request = client.build_request(&conversation, &config, false).unwrap();
        assert_eq!(
            serde_json::to_value(request).unwrap()["generation_config"]["thinking_level"],
            "low"
        );
    }

    #[test]
    fn temperature_is_rejected_because_the_api_no_longer_accepts_it() {
        let client = GoogleClient::new("test");
        let conversation = Conversation::new(Model::google_gemini_3_5_flash());
        assert!(matches!(
            client.build_request(&conversation, &GenerationConfig::new(0.5), false),
            Err(Error::UnsupportedFeature(_))
        ));
        let request = client
            .build_request(&conversation, &GenerationConfig::default(), false)
            .unwrap();
        assert!(serde_json::to_value(request).unwrap()["generation_config"].get("temperature").is_none());
    }
}

#[cfg(test)]
mod model_rule_tests {
    use super::GoogleClient;
    use crate::{
        config::GenerationConfig,
        conversation::Conversation,
        error::Error,
        model::Model,
        reasoning::{GeminiThinkingLevel, ReasoningConfig},
    };

    fn thinking(model: Model, config: &GenerationConfig) -> Result<(Option<String>, Option<String>), Error> {
        GoogleClient::new("test").resolve_thinking(&Conversation::new(model), config)
    }

    fn level(model: Model, config: &GenerationConfig) -> Option<String> {
        thinking(model, config).unwrap().0
    }

    #[test]
    fn new_gemini_models_use_their_documented_default_levels() {
        let default = GenerationConfig::default();
        for model in [
            Model::google_gemini_3_8_flash(),
            Model::google_gemini_3_7_flash(),
            Model::google_gemini_3_6_flash(),
            Model::google_gemini_3_5_flash(),
        ] {
            assert_eq!(level(model, &default).as_deref(), Some("medium"));
        }
        for model in [
            Model::google_gemini_3_5_flash_lite(),
            Model::google_gemini_3_1_flash_lite_stable(),
            Model::google_gemini_3_1_flash_lite(),
        ] {
            assert_eq!(level(model, &default).as_deref(), Some("minimal"));
        }
        assert_eq!(level(Model::google_gemini_3_1_pro(), &default).as_deref(), Some("high"));
    }

    #[test]
    fn gemini_3_7_and_3_8_flash_reject_minimal_explicitly_and_from_a_zero_budget() {
        let minimal =
            GenerationConfig::default().with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Minimal, false));
        let zero_budget = GenerationConfig::default().with_reasoning(ReasoningConfig::google(0, false));
        for model in [Model::google_gemini_3_7_flash(), Model::google_gemini_3_8_flash()] {
            assert!(matches!(thinking(model.clone(), &minimal), Err(Error::InvalidRequest(_))));
            assert!(matches!(thinking(model.clone(), &zero_budget), Err(Error::InvalidRequest(_))));
            let low = GenerationConfig::default()
                .with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Low, true));
            assert_eq!(thinking(model, &low).unwrap(), (Some("low".into()), Some("auto".into())));
        }
        // 3.6 Flash accepts minimal.
        assert_eq!(level(Model::google_gemini_3_6_flash(), &minimal).as_deref(), Some("minimal"));
    }

    #[test]
    fn gemini_2_5_takes_levels_but_not_minimal_and_2_0_has_no_thinking() {
        let medium = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Medium, false));
        assert_eq!(level(Model::google_gemini_2_5_flash(), &medium).as_deref(), Some("medium"));
        let minimal = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Minimal, false));
        assert!(matches!(
            thinking(Model::google_gemini_2_5_pro(), &minimal),
            Err(Error::InvalidRequest(_))
        ));
        for config in [medium, GenerationConfig::default().with_reasoning(ReasoningConfig::google(2048, false))] {
            assert!(matches!(
                thinking(Model::google_gemini_2_0_flash(), &config),
                Err(Error::InvalidRequest(_))
            ));
        }
        assert_eq!(thinking(Model::google_gemini_2_0_flash(), &GenerationConfig::default()).unwrap(), (None, None));
    }

    #[test]
    fn gemini_3_6_to_3_8_flash_promotional_pricing_ends_on_2027_01_01() {
        use std::time::{Duration, UNIX_EPOCH};
        let new_year = UNIX_EPOCH + Duration::from_secs(1_798_761_600);
        for model in [
            Model::google_gemini_3_6_flash(),
            Model::google_gemini_3_7_flash(),
            Model::google_gemini_3_8_flash(),
        ] {
            let before = model.pricing.at(new_year - Duration::from_secs(1));
            assert_eq!(before.input_cost_per_million, 0.75);
            assert_eq!(before.cached_input_cost_per_million, Some(0.075));
            let after = model.pricing.at(new_year);
            assert_eq!(after.input_cost_per_million, 1.50);
            assert_eq!(after.output_cost_per_million, 7.50);
            assert_eq!(after.cached_input_cost_per_million, Some(0.15));
        }
    }

    #[test]
    fn gemini_3_pro_preview_cache_reads_use_the_published_rates() {
        let pricing = Model::google_gemini_3_0_pro().pricing;
        assert_eq!(pricing.cached_input_cost_per_million, Some(0.20));
        assert_eq!(pricing.large_cached_input_cost_per_million, Some(0.40));
    }
}
