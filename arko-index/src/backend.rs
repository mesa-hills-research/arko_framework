//! Provider boundary. Tests and offline applications can supply deterministic backends.
use crate::{EmbeddingProfile, Result, error::invalid};
use arko_ai::{
    EmbeddingClient, EmbeddingConfig, EmbeddingModelDetails, EmbeddingProviderIdentifier,
    GenerationConfig, GenerationResponse, LLMClient, Model, OpenAIEffort, ReasoningConfig,
    StopReason, StreamEvent, TokenCounter, ToolCall, ToolChoice, ToolConfig, ToolDefinition,
    UsageInfo, Verbosity, VoyageInputType,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRequest {
    pub system: String,
    pub prompt: String,
    pub tool: String,
    pub schema: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReply {
    pub value: Value,
    /// The provider's call record, such as a serialized
    /// [`arko_ai::ModelCallUsage`] (with the raw stop reason). The index
    /// wraps it with the attempt history before saving it.
    pub usage: Value,
    /// Token usage and cost of the call, when the backend reports them.
    /// Summed into [`JobUsage`](crate::JobUsage); replies saved before this
    /// field existed load as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_info: Option<UsageInfo>,
}
#[derive(Clone, Debug)]
pub struct VectorReply {
    pub vectors: Vec<Vec<f32>>,
    pub billed_tokens: usize,
    /// Token usage and cost of the request, when the backend reports them.
    /// Summed into [`JobUsage`](crate::JobUsage).
    pub usage_info: Option<UsageInfo>,
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub item_tokens: usize,
    pub batch_tokens: usize,
    pub batch_items: usize,
}
#[async_trait::async_trait]
pub trait Backend: Send + Sync + 'static {
    async fn chunk(&self, request: ModelRequest) -> Result<ModelReply>;
    async fn embed(&self, texts: Vec<String>, query: bool) -> Result<VectorReply>;
    /// Exact selected-model token count. Implementations must disable truncation.
    fn tokens(&self, text: &str) -> Result<usize>;
    /// Token count of the submitted payload for provider-enforced input limits.
    fn input_tokens(&self, text: &str, _query: bool) -> Result<usize> {
        self.tokens(text)
    }
    fn limits(&self) -> Limits;
    fn profile(&self) -> &EmbeddingProfile;
    fn output_tokenizer(&self) -> String;
    /// The chunk model, for job provenance. `None` (the default) records no
    /// model.
    fn chunk_model_info(&self) -> Option<ChunkModelInfo> {
        None
    }
}

/// The one call to `tool` in a completed turn. Fails like
/// [`GenerationResponse::single_tool_call`] (refusals, no call, several calls,
/// another tool, unparseable arguments) and, even when the output holds a
/// complete call, for a turn that ended any other way than
/// [`StopReason::EndTurn`] or [`StopReason::ToolUse`].
pub(crate) fn completed_tool_call<'a>(
    response: &'a GenerationResponse,
    tool: &str,
) -> std::result::Result<&'a ToolCall, arko_ai::Error> {
    let call = response.single_tool_call(tool)?;
    match response.stop_reason {
        StopReason::EndTurn | StopReason::ToolUse => Ok(call),
        _ => Err(arko_ai::Error::UnexpectedToolCalls {
            expected: tool.to_owned(),
            stop_reason: response.stop_reason.clone(),
            found: vec![call.name.clone()],
        }),
    }
}

/// The generation settings [`ArkoBackend`] uses with its default chunk model,
/// GPT-5.6 Luna: high reasoning effort with summaries, low verbosity.
pub fn default_chunk_config() -> GenerationConfig {
    GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, true))
        .with_verbosity(Verbosity::Low)
}

/// `config` with the boundary tool for `request`: forced where `model` allows
/// forcing a tool with `config`, and with parallel calls disabled where
/// supported. Replaces any tool configuration in `config`.
fn chunk_generation_config(
    model: &Model,
    config: &GenerationConfig,
    request: &ModelRequest,
) -> GenerationConfig {
    let mut tools = ToolConfig::new(vec![
        ToolDefinition::new(
            &request.tool,
            "Submit the requested source boundaries or repair choices.",
            request.schema.clone(),
        )
        .with_strict(true),
    ])
    // Without a forced call the reply must still be exactly one call to the tool.
    .with_tool_choice(if model.supports_forced_tool_choice(config) {
        ToolChoice::Specific(request.tool.clone())
    } else {
        ToolChoice::Auto
    });
    if model.supports_disabling_parallel_tool_calls() {
        tools = tools.with_parallel_tool_calls(false);
    }
    config.clone().with_tool_config(tools)
}

/// The model a backend selects chunk boundaries with, recorded as `model` and
/// `reasoning` in each job's `snapshot` event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkModelInfo {
    /// The model id.
    pub model: String,
    /// The reasoning setting: the effort name for OpenAI reasoning, otherwise
    /// the serialized reasoning configuration (`null` when none is set).
    pub reasoning: Value,
}
impl ChunkModelInfo {
    pub fn new(model: &Model, config: &GenerationConfig) -> Self {
        Self {
            model: model.id.clone(),
            reasoning: match &config.reasoning {
                Some(ReasoningConfig::OpenAI { effort, .. }) => serde_json::to_value(effort),
                other => serde_json::to_value(other),
            }
            .unwrap_or_default(),
        }
    }
}

pub struct ArkoBackend {
    llm: LLMClient,
    embeddings: EmbeddingClient,
    tokenizer: TokenCounter,
    profile: EmbeddingProfile,
    limits: Limits,
    chunk_model: Model,
    chunk_config: GenerationConfig,
}
impl ArkoBackend {
    /// Loads the pinned Hugging Face tokenizer on a blocking worker. Authentication
    /// and rotated OAuth credential persistence remain with the application.
    ///
    /// Chunk boundaries come from GPT-5.6 Luna with [`default_chunk_config`];
    /// use [`ArkoBackend::with_chunk_model`] for another model.
    pub async fn new(
        llm: LLMClient,
        embeddings: EmbeddingClient,
        profile: EmbeddingProfile,
    ) -> Result<Self> {
        let details =
            EmbeddingModelDetails::get(EmbeddingProviderIdentifier::Voyage, &profile.model)?;
        if !details.supported_dimensions.contains(&profile.dimensions) {
            return Err(invalid("unsupported embedding dimension"));
        }
        let limits = Limits {
            item_tokens: details.max_input_tokens_per_item,
            batch_tokens: details
                .max_batch_total_tokens
                .ok_or_else(|| invalid("model has no verified aggregate request limit"))?,
            batch_items: details
                .max_batch_size
                .ok_or_else(|| invalid("model has no verified item-count limit"))?,
        };
        let tokenizer = embeddings
            .token_counter(
                &EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, &profile.model)
                    .with_tokenizer_revision(&profile.tokenizer_revision),
            )
            .await?;
        Ok(Self {
            llm,
            embeddings,
            tokenizer,
            profile,
            limits,
            chunk_model: Model::openai_gpt_5_6_luna(),
            chunk_config: default_chunk_config(),
        })
    }

    /// Selects chunk boundaries with `model` and `config` instead of GPT-5.6
    /// Luna with [`default_chunk_config`]. The client passed to
    /// [`ArkoBackend::new`] must hold credentials for the model's provider.
    ///
    /// `config` holds the generation settings (reasoning, verbosity, output
    /// limit, …). The backend owns the tool configuration: any `tool_config`
    /// in `config` is replaced by the boundary tool, forced when the model
    /// supports forcing it with `config`
    /// ([`Model::supports_forced_tool_choice`]) and with parallel calls
    /// disabled where supported. Either way a reply must be exactly one call
    /// to that tool in a completed turn. Changing the model does not
    /// invalidate boundaries reused from earlier generations; the source
    /// validation is the same for every model.
    pub fn with_chunk_model(mut self, model: Model, config: GenerationConfig) -> Self {
        self.chunk_model = model;
        self.chunk_config = config;
        self
    }
}
#[async_trait::async_trait]
impl Backend for ArkoBackend {
    async fn chunk(&self, request: ModelRequest) -> Result<ModelReply> {
        if !self.chunk_model.reports_stop_reason() {
            return Err(invalid(format!(
                "chunk model '{}' does not report completion metadata, which boundary validation requires",
                self.chunk_model.id
            )));
        }
        let config = chunk_generation_config(&self.chunk_model, &self.chunk_config, &request);
        let mut conversation = self
            .llm
            .new_conversation(self.chunk_model.clone())
            .with_system_message(&request.system);
        conversation.add_user_message(&request.prompt);
        let mut stream = self.llm.generate_stream(&conversation, config).await?;
        while let Some(event) = stream.next().await {
            if let StreamEvent::End(response) = event? {
                // Refusals and token limits are permanent failures; a missing,
                // extra or malformed call is resampled as a transient failure.
                let call = completed_tool_call(&response, &request.tool)?;
                return Ok(ModelReply {
                    value: call.arguments.clone(),
                    // Keeps the provider's raw stop reason with the call record.
                    usage: serde_json::to_value(&response.model_call_usage)?,
                    usage_info: Some(response.usage.clone()),
                });
            }
        }
        Err(invalid("chunk model stream ended without a completed response"))
    }
    fn chunk_model_info(&self) -> Option<ChunkModelInfo> {
        Some(ChunkModelInfo::new(&self.chunk_model, &self.chunk_config))
    }
    async fn embed(&self, texts: Vec<String>, query: bool) -> Result<VectorReply> {
        let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, &self.profile.model)
            .with_output_dimensionality(self.profile.dimensions)
            .with_voyage_truncation(false)
            .with_tokenizer_revision(&self.profile.tokenizer_revision)
            .with_voyage_input_type(if query {
                VoyageInputType::Query
            } else {
                VoyageInputType::Document
            });
        let result = self.embeddings.embed(texts, &config).await?;
        Ok(VectorReply {
            vectors: result.embeddings,
            billed_tokens: result.usage.input_tokens,
            usage_info: Some(result.usage),
        })
    }
    fn tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer
            .count(text)
            .map_err(|e| invalid(e.to_string()))
    }
    fn limits(&self) -> Limits {
        self.limits
    }
    fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }
    fn output_tokenizer(&self) -> String {
        format!(
            "voyageai/{}@{}",
            self.profile.model, self.profile.tokenizer_revision
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arko_ai::{Message, ModelCallUsage, Refusal, RefusalSource, RetryClass};
    use serde_json::json;

    fn response(calls: &[&str], stop_reason: StopReason) -> GenerationResponse {
        let mut message = Message::new_assistant("");
        for (n, name) in calls.iter().enumerate() {
            message.tool_calls.push(ToolCall {
                id: Some(format!("call_{n}")),
                name: (*name).into(),
                arguments: json!({"f0": {"cuts": []}}),
                thought_signature: None,
                provider_id: None,
                invalid_arguments: None,
            });
        }
        GenerationResponse {
            message,
            usage: UsageInfo::new(100, 20),
            model_call_usage: ModelCallUsage::new(
                "call".into(),
                "openai".into(),
                "test".into(),
                100,
                20,
                1000,
            )
            .with_stop_metadata(Some("completed".into()), None),
            stop_reason,
            refusal: None,
            openai_turn_state: None,
        }
    }
    /// How the scheduler treats the rejection of `response`.
    fn rejection(response: &GenerationResponse) -> RetryClass {
        crate::Error::from(completed_tool_call(response, "submit").unwrap_err())
            .retry_class()
            .unwrap()
    }

    fn request() -> ModelRequest {
        ModelRequest {
            system: "system".into(),
            prompt: "prompt".into(),
            tool: "submit_cuts".into(),
            schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn default_chunk_request_is_unchanged() {
        // The request the backend built before the chunk model was configurable.
        let request = request();
        let before = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, true))
            .with_verbosity(Verbosity::Low)
            .with_tool_config(
                ToolConfig::new(vec![
                    ToolDefinition::new(
                        &request.tool,
                        "Submit the requested source boundaries or repair choices.",
                        request.schema.clone(),
                    )
                    .with_strict(true),
                ])
                .with_tool_choice(ToolChoice::Specific(request.tool.clone()))
                .with_parallel_tool_calls(false),
            );
        let luna = Model::openai_gpt_5_6_luna();
        let now = chunk_generation_config(&luna, &default_chunk_config(), &request);
        assert_eq!(
            serde_json::to_value(&now).unwrap(),
            serde_json::to_value(&before).unwrap()
        );
        // The provenance the snapshot event recorded before.
        assert_eq!(
            serde_json::to_value(ChunkModelInfo::new(&luna, &default_chunk_config())).unwrap(),
            json!({"model": "gpt-5.6-luna", "reasoning": "high"})
        );
    }

    #[test]
    fn chunk_tool_is_forced_and_serial_only_where_supported() {
        let request = request();
        let tools = |model: &Model, config: &GenerationConfig| {
            let config = chunk_generation_config(model, config, &request);
            serde_json::to_value(config.tool_config.unwrap()).unwrap()
        };
        // The backend's tool replaces any tool configuration in the settings.
        let unrelated = GenerationConfig::default().with_tool_config(
            ToolConfig::new(vec![ToolDefinition::new("other", "other", json!({}))])
                .with_tool_choice(ToolChoice::None),
        );
        let openai = tools(&Model::openai_gpt_5_5(), &unrelated);
        assert_eq!(openai["tools"].as_array().unwrap().len(), 1);
        assert_eq!(openai["tools"][0]["name"], "submit_cuts");
        assert_eq!(openai["tool_choice"], json!({"Specific": "submit_cuts"}));
        assert_eq!(openai["parallel_tool_calls"], false);
        // Claude Opus 5.5 rejects a forced tool: the reply is still validated.
        let opus = tools(&Model::anthropic_claude_opus_5_5(), &GenerationConfig::default());
        assert_eq!(opus["tool_choice"], "Auto");
        assert_eq!(opus["parallel_tool_calls"], false);
        // Gemini can't disable parallel calls; a second call is rejected instead.
        let gemini = tools(&Model::google_gemini_3_5_flash(), &GenerationConfig::default());
        assert_eq!(gemini["tool_choice"], json!({"Specific": "submit_cuts"}));
        assert!(gemini.get("parallel_tool_calls").is_none_or(Value::is_null));
        // Provenance of a configured model.
        let adaptive = GenerationConfig::default().with_reasoning(ReasoningConfig::AnthropicAdaptive {
            effort: None,
        });
        let info = ChunkModelInfo::new(&Model::anthropic_claude_opus_5_5(), &adaptive);
        assert_eq!(info.model, "claude-opus-5-5");
        assert_eq!(info.reasoning, json!({"AnthropicAdaptive": {"effort": null}}));
        let plain = ChunkModelInfo::new(&Model::openai_gpt_5_5(), &GenerationConfig::default());
        assert_eq!(plain.reasoning, Value::Null);
    }

    #[test]
    fn chunk_replies_need_one_call_in_a_completed_turn() {
        for stop in [StopReason::ToolUse, StopReason::EndTurn] {
            let reply = response(&["submit"], stop);
            let call = completed_tool_call(&reply, "submit").unwrap();
            assert_eq!(call.arguments, json!({"f0": {"cuts": []}}));
        }
        // A token or context limit needs a configuration change, even when the
        // output holds a complete call.
        for stop in [StopReason::MaxTokens, StopReason::ContextWindowExceeded] {
            assert_eq!(
                rejection(&response(&["submit"], stop.clone())),
                RetryClass::Permanent
            );
            assert_eq!(rejection(&response(&[], stop)), RetryClass::Permanent);
        }
        let mut refused = response(&["submit"], StopReason::Refusal);
        refused.refusal = Some(Refusal::new(RefusalSource::Model));
        assert!(matches!(
            completed_tool_call(&refused, "submit"),
            Err(arko_ai::Error::Refusal(_))
        ));
        assert_eq!(rejection(&refused), RetryClass::Permanent);
        // A missing, extra, foreign or unparseable call, or a turn that ended
        // some other way, is resampled.
        for reply in [
            response(&[], StopReason::EndTurn),
            response(&["other"], StopReason::ToolUse),
            response(&["submit", "submit"], StopReason::ToolUse),
            response(&["submit"], StopReason::Other("incomplete".into())),
        ] {
            assert_eq!(
                rejection(&reply),
                RetryClass::Transient,
                "{:?}",
                reply.stop_reason
            );
        }
        let mut malformed = response(&["submit"], StopReason::ToolUse);
        malformed.message.tool_calls[0].invalid_arguments = Some("{\"f0\":".into());
        assert_eq!(rejection(&malformed), RetryClass::Transient);
        // The raw stop reason no longer matters; the normalized one does.
        let mut raw = response(&["submit"], StopReason::ToolUse);
        raw.model_call_usage.stop_reason = Some("requires_action".into());
        assert!(completed_tool_call(&raw, "submit").is_ok());
    }
}
