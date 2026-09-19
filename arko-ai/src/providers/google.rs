use super::{GenerationResponse, Provider, StreamEvent};
use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::Provider as ProviderType,
    reasoning::{GeminiThinkingLevel, GoogleReasoningFragment, ReasoningArtifact, ReasoningConfig},
    tool::{ToolCall, ToolChoice},
    usage::{ModelCallUsage, UsageInfo},
};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const INTERACTIONS_URL: &str = "https://generativelanguage.googleapis.com/v1/interactions";

#[derive(Clone)]
pub struct GoogleClient {
    api_key: String,
    http_client: reqwest::Client,
}

impl GoogleClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http_client: super::default_http_client(),
        }
    }
}

#[async_trait::async_trait]
impl Provider for GoogleClient {
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
            .post(INTERACTIONS_URL)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::from_response("google API error", response).await);
        }

        let interaction: InteractionResponse = response.json().await?;

        if interaction.status.as_deref() == Some("failed") {
            return Err(Error::ApiError(format!(
                "Google Interactions API returned a failed interaction: {}",
                interaction
                    .error
                    .as_ref()
                    .map(Value::to_string)
                    .unwrap_or_else(|| "unknown error".to_string())
            )));
        }

        let message = Self::parse_steps(&interaction.steps);

        let usage_data = interaction.usage.clone().unwrap_or_default();
        let mut usage = Self::calculate_usage(&usage_data);
        usage.calculate_cost(&conversation.model, Some(config));

        let model_call_usage = ModelCallUsage::new(
            interaction.id.clone(),
            conversation.model.provider.as_telemetry_str().to_string(),
            conversation.model.id.clone(),
            usage.input_tokens,
            usage.output_tokens,
            conversation.model.max_input_tokens,
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
        if conversation.model.provider != ProviderType::Google {
            return Err(Error::ProviderNotSupported(
                "Model is not from Google".to_string(),
            ));
        }

        let request = self.build_request(conversation, config, true)?;
        let url = format!("{INTERACTIONS_URL}?alt=sse");

        let response = self
            .http_client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::from_response("google API error", response).await);
        }

        let byte_stream = response.bytes_stream().eventsource();
        let model = conversation.model.clone();
        let config = config.clone();
        let emit_reasoning = Self::requests_thinking_summaries(&config);

        let stream = try_stream! {
            let mut event_stream = byte_stream;
            let mut full_text = String::new();
            let mut reasoning_fragments: Vec<GoogleReasoningFragment> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_call_index = 0usize;
            let mut current_thought: Option<PendingThought> = None;
            let mut pending_tool: Option<PendingFunctionCall> = None;
            let mut interaction_id = String::new();
            let mut final_usage: Option<InteractionUsage> = None;

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

                        match step.get("type").and_then(Value::as_str) {
                            Some("thought") => {
                                current_thought = Some(PendingThought {
                                    summary: String::new(),
                                    signature: step
                                        .get("signature")
                                        .and_then(Value::as_str)
                                        .map(String::from),
                                    signature_received: step.get("signature").is_some(),
                                });
                            }
                            Some("function_call") => {
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
                                        signature_received: false,
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
                                    .unwrap_or("")
                                    .to_string();

                                if let Some(thought) = &mut current_thought {
                                    thought.signature = Some(signature);
                                    thought.signature_received = true;
                                } else {
                                    current_thought = Some(PendingThought {
                                        summary: String::new(),
                                        signature: Some(signature),
                                        signature_received: true,
                                    });
                                }
                            }
                            Some("function_call_signature") => {
                                let signature = delta
                                    .get("signature")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();

                                if let Some(pending) = &mut pending_tool {
                                    pending.signature = Some(signature);
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
                            reasoning_fragments.push(GoogleReasoningFragment {
                                summary: thought.summary,
                                signature: thought.signature.unwrap_or_default(),
                            });
                        }

                        if let Some(pending) = pending_tool.take() {
                            let arguments = if !pending.partial_arguments.is_empty() {
                                serde_json::from_str(&pending.partial_arguments)
                                    .unwrap_or(Value::Object(Default::default()))
                            } else {
                                pending
                                    .initial_arguments
                                    .unwrap_or(Value::Object(Default::default()))
                            };

                            tool_calls.push(ToolCall {
                                id: pending.id,
                                name: pending.name,
                                arguments,
                                thought_signature: pending.signature,
                                provider_id: None,
                            });

                            tool_call_index += 1;
                        }
                    }
                    Some("interaction.completed")
                    | Some("interaction.requires_action") => {
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
                                Err(Error::ApiError(
                                    "Google Interactions stream reported a failed interaction."
                                        .to_string(),
                                ))?;

                                break;
                            }
                            Some("completed")
                            | Some("requires_action")
                            | Some("incomplete")
                            | Some("cancelled") => {
                                break;
                            }
                            _ => {}
                        }
                    }
                    Some("interaction.failed") => {
                        Err(Error::ApiError(format!(
                            "Google Interactions stream failed: {}",
                            value["interaction"]
                        )))?;

                        break;
                    }
                    Some("error") => {
                        let message = value
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown");

                        let code = value
                            .pointer("/error/code")
                            .and_then(Value::as_str)
                            .unwrap_or("error");

                        Err(Error::ApiError(format!(
                            "Google Interactions stream error {code}: {message}"
                        )))?;

                        break;
                    }
                    _ => {}
                }
            }

            if let Some(thought) = current_thought.take() {
                reasoning_fragments.push(GoogleReasoningFragment {
                    summary: thought.summary,
                    signature: thought.signature.unwrap_or_default(),
                });
            }

            if let Some(pending) = pending_tool.take() {
                let arguments = if !pending.partial_arguments.is_empty() {
                    serde_json::from_str(&pending.partial_arguments)
                        .unwrap_or(Value::Object(Default::default()))
                } else {
                    pending
                        .initial_arguments
                        .unwrap_or(Value::Object(Default::default()))
                };

                tool_calls.push(ToolCall {
                    id: pending.id,
                    name: pending.name,
                    arguments,
                    thought_signature: pending.signature,
                    provider_id: None,
                });
            }

            let reasoning_artifact = if reasoning_fragments.is_empty() {
                None
            } else {
                Some(ReasoningArtifact::new_google_reasoning(
                    reasoning_fragments,
                ))
            };

            let message = Message {
                role: Role::Assistant,
                content: vec![Content::Text(full_text)],
                reasoning_artifact,
                tool_calls,
                id: None,
            };

            let usage_data = final_usage.unwrap_or_default();
            let mut usage = GoogleClient::calculate_usage(&usage_data);
            usage.calculate_cost(&model, Some(&config));

            let model_call_usage = ModelCallUsage::new(
                interaction_id,
                model.provider.as_telemetry_str().to_string(),
                model.id.clone(),
                usage.input_tokens,
                usage.output_tokens,
                model.max_input_tokens,
            );

            yield StreamEvent::End(Box::new(GenerationResponse {
                message,
                usage,
                model_call_usage,
            }));
        };

        Ok(stream.boxed())
    }
}

impl GoogleClient {
    fn default_thinking_level(model: &crate::model::Model) -> GeminiThinkingLevel {
        if model.is_gemini_3_5_flash() {
            GeminiThinkingLevel::Medium
        } else if model.is_gemini_3_1_flash_lite() {
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

    fn thinking_level_for_budget(budget: isize) -> Option<&'static str> {
        if budget < 0 {
            None
        } else if budget == 0 {
            Some("minimal")
        } else if budget <= 4_096 {
            Some("low")
        } else if budget <= 16_384 {
            Some("medium")
        } else {
            Some("high")
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
        if let Some(tool_config) = &config.tool_config
            && tool_config.parallel_tool_calls == Some(false)
        {
            return Err(Error::UnsupportedFeature(
                "Google Gemini does not support disabling parallel tool calls.".to_string(),
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
            temperature: config.temperature_specified.then_some(config.temperature),
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
                }) => Ok((
                    Self::thinking_level_for_budget(*thinking_budget).map(String::from),
                    summaries_value(*include_summaries),
                )),
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
        } else if let Some(ReasoningConfig::Google {
            thinking_budget,
            include_summaries,
        }) = &config.reasoning
        {
            Ok((
                Self::thinking_level_for_budget(*thinking_budget).map(String::from),
                summaries_value(*include_summaries),
            ))
        } else {
            Ok((None, None))
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
                                        signature: Some(fragment.signature.clone()),
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
                            signature: tool_call.thought_signature.clone(),
                            name: tool_call.name.clone(),
                            arguments: tool_call.arguments.clone(),
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
                    tool_calls.push(ToolCall {
                        id: step.get("id").and_then(Value::as_str).map(String::from),
                        name: step
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        arguments: step
                            .get("arguments")
                            .cloned()
                            .unwrap_or(Value::Object(Default::default())),
                        thought_signature: step
                            .get("signature")
                            .and_then(Value::as_str)
                            .map(String::from),
                        provider_id: None,
                    });
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

    fn calculate_usage(usage: &InteractionUsage) -> UsageInfo {
        let input_tokens = usage.total_input_tokens as usize;

        let output_tokens = usage
            .total_tokens
            .map(|total| (total as usize).saturating_sub(input_tokens))
            .unwrap_or((usage.total_output_tokens + usage.total_thought_tokens) as usize);

        let mut info = UsageInfo::new(input_tokens, output_tokens);

        if let Some(cached_tokens) = usage.total_cached_tokens
            && cached_tokens > 0
        {
            info = info.with_cached_tokens(cached_tokens as usize);
        }

        info
    }
}

struct PendingThought {
    summary: String,
    signature: Option<String>,
    signature_received: bool,
}

struct PendingFunctionCall {
    id: Option<String>,
    name: String,
    signature: Option<String>,
    partial_arguments: String,
    initial_arguments: Option<Value>,
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
    temperature: Option<f64>,
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
    #[serde(default)]
    error: Option<Value>,
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
    total_tokens: Option<u64>,
    #[serde(default)]
    total_cached_tokens: Option<u64>,
}

pub fn list_models() -> Vec<crate::model::Model> {
    crate::model::Model::list_google_models()
}
