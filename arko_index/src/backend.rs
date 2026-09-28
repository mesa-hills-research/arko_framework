//! Provider boundary. Tests and offline applications can supply deterministic backends.
use crate::{EmbeddingProfile, Error, Result, error::invalid};
use arko_ai::{
    EmbeddingClient, EmbeddingConfig, EmbeddingModelDetails, EmbeddingProviderIdentifier,
    GenerationConfig, GenerationResponse, LLMClient, Model, OpenAIEffort, Provider,
    ReasoningConfig, StopReason, StreamEvent, TokenCounter, ToolCall, ToolChoice, ToolConfig,
    ToolDefinition, UsageInfo, Verbosity, VoyageInputType,
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
    /// The prompt-cache key of the job the request belongs to: the same for
    /// all of a job's requests, so they reach one cache. [`ArkoBackend`] uses
    /// it as the conversation's `prompt_cache_key`, which ChatGPT OAuth
    /// requests also send as the `session-id` header the backend routes
    /// cache affinity by. `None` gives each request a new key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
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
/// The tokenizer that measures chunk requests for a chunk model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkTokenizer {
    /// OpenAI's `o200k_base` encoding, used by GPT-4o and later, GPT-5.x and
    /// GPT-6. Counted locally with the encoding built into the crate.
    O200kBase,
}
/// What sizes chunk requests: the chunk model's tokenizer, and the most
/// input tokens it accepts on the auth path in use. Either may be unknown;
/// see [`ChunkingOptions`](crate::ChunkingOptions).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkInput {
    pub tokenizer: Option<ChunkTokenizer>,
    pub max_input_tokens: Option<usize>,
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
    /// The chunk model's tokenizer and input limit, which size chunk
    /// requests. The default reports neither: requests are measured in bytes
    /// against a share of
    /// [`FALLBACK_MAX_INPUT_TOKENS`](crate::budget::FALLBACK_MAX_INPUT_TOKENS).
    async fn chunk_input(&self) -> ChunkInput {
        ChunkInput::default()
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

/// The tokenizer that measures requests to `model`: `o200k_base` for the
/// OpenAI provider (every registry model there uses it; GPT-6 reports the
/// same input tokens as GPT-5.6), none for other providers.
pub fn chunk_tokenizer(model: &Model) -> Option<ChunkTokenizer> {
    (model.provider == Provider::OpenAI).then_some(ChunkTokenizer::O200kBase)
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

/// The per-request token limit assumed for a Voyage model whose limit Voyage
/// does not publish (`voyage-code-4`): 120,000, the smallest per-request
/// limit Voyage publishes for any current model.
pub const UNPUBLISHED_BATCH_TOKENS: usize = 120_000;

/// The request limits of a Voyage embedding model: its published per-input,
/// per-request token and per-request input limits, with
/// [`UNPUBLISHED_BATCH_TOKENS`] where no per-request token limit is
/// published.
pub fn embedding_limits(details: &EmbeddingModelDetails) -> Result<Limits> {
    Ok(Limits {
        item_tokens: details.max_input_tokens_per_item,
        batch_tokens: details
            .max_batch_total_tokens
            .unwrap_or(UNPUBLISHED_BATCH_TOKENS),
        batch_items: details
            .max_batch_size
            .ok_or_else(|| invalid("model has no verified item-count limit"))?,
    })
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
        let limits = embedding_limits(&details)?;
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
/// Sends `request` to `model` and returns its boundary tool call. The
/// shared instructions go first (the system message), then the files, so
/// requests of one job share their prompt prefix; the job's cache key routes
/// them to one cache.
async fn chunk_with(
    llm: &LLMClient,
    model: &Model,
    chunk_config: &GenerationConfig,
    request: ModelRequest,
) -> Result<ModelReply> {
    if !model.reports_stop_reason() {
        return Err(invalid(format!(
            "chunk model '{}' does not report completion metadata, which boundary validation requires",
            model.id
        )));
    }
    let config = chunk_generation_config(model, chunk_config, &request);
    let mut conversation = llm
        .new_conversation(model.clone())
        .with_system_message(&request.system);
    if let Some(key) = &request.cache_key {
        conversation = conversation.with_prompt_cache_key(key);
    }
    conversation.add_user_message(&request.prompt);
    let mut stream = llm.generate_stream(&conversation, config).await?;
    while let Some(event) = stream.next().await {
        if let StreamEvent::End(response) = event? {
            // Refusals and token limits are permanent failures; a missing,
            // extra or malformed call is resampled as a transient failure.
            // Either way the response was billed, so its usage goes along.
            let call = completed_tool_call(&response, &request.tool)
                .map_err(|error| Error::from(error).with_usage(response.usage.clone()))?;
            return Ok(ModelReply {
                value: call.arguments.clone(),
                // Keeps the provider's raw stop reason with the call record.
                usage: serde_json::to_value(&response.model_call_usage)?,
                usage_info: Some(response.usage.clone()),
            });
        }
    }
    Err(invalid(
        "chunk model stream ended without a completed response",
    ))
}
#[async_trait::async_trait]
impl Backend for ArkoBackend {
    async fn chunk(&self, request: ModelRequest) -> Result<ModelReply> {
        chunk_with(&self.llm, &self.chunk_model, &self.chunk_config, request).await
    }
    fn chunk_model_info(&self) -> Option<ChunkModelInfo> {
        Some(ChunkModelInfo::new(&self.chunk_model, &self.chunk_config))
    }
    /// OpenAI chunk models are measured in `o200k_base` tokens; others in
    /// bytes. The input limit is the model's on the client's auth path
    /// ([`LLMClient::max_input_tokens`]).
    async fn chunk_input(&self) -> ChunkInput {
        ChunkInput {
            tokenizer: chunk_tokenizer(&self.chunk_model),
            max_input_tokens: self.llm.max_input_tokens(&self.chunk_model).await,
        }
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
            cache_key: None,
        }
    }

    /// Answers `count` requests with an error, recording each request's head.
    async fn recording_server(count: usize) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let mut heads = vec![];
            for _ in 0..count {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut received = vec![];
                let mut buffer = [0u8; 4096];
                let head_end = loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    received.extend_from_slice(&buffer[..read]);
                    if let Some(at) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let head = String::from_utf8_lossy(&received[..head_end]).to_string();
                let length: usize = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while received.len() < head_end + length {
                    let read = socket.read(&mut buffer).await.unwrap();
                    received.extend_from_slice(&buffer[..read]);
                }
                let body = r#"{"error":{"message":"rejected by the test server","type":"invalid_request_error","code":"invalid_request_error"}}"#;
                let reply = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
                heads.push(head);
            }
            heads
        });
        (url, task)
    }

    fn header(head: &str, name: &str) -> Option<String> {
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_owned())
        })
    }

    #[tokio::test]
    async fn the_job_cache_key_is_the_chatgpt_oauth_session_id() {
        let (url, server) = recording_server(3).await;
        // An access token that expires in 2100.
        let token = "header.eyJleHAiOjQxMDI0NDQ4MDB9.signature";
        let llm = LLMClient::new().with_openai_client(
            arko_ai::providers::openai::OpenAIClient::new_chatgpt_oauth(
                token,
                Some("refresh"),
                "account",
            )
            .with_base_url(url),
        );
        let luna = Model::openai_gpt_5_6_luna();
        for key in [Some("job-key"), Some("job-key"), None] {
            let request = ModelRequest {
                cache_key: key.map(Into::into),
                ..request()
            };
            // The test server rejects every request; only its head matters.
            assert!(
                chunk_with(&llm, &luna, &default_chunk_config(), request)
                    .await
                    .is_err()
            );
        }
        let heads = server.await.unwrap();
        let sessions: Vec<_> = heads
            .iter()
            .map(|head| header(head, "session-id").unwrap())
            .collect();
        assert_eq!(sessions[0], "job-key");
        assert_eq!(sessions[1], "job-key");
        assert_ne!(sessions[2], "job-key");
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
        let opus = tools(
            &Model::anthropic_claude_opus_5_5(),
            &GenerationConfig::default(),
        );
        assert_eq!(opus["tool_choice"], "Auto");
        assert_eq!(opus["parallel_tool_calls"], false);
        // Gemini can't disable parallel calls; a second call is rejected instead.
        let gemini = tools(
            &Model::google_gemini_3_5_flash(),
            &GenerationConfig::default(),
        );
        assert_eq!(gemini["tool_choice"], json!({"Specific": "submit_cuts"}));
        assert!(gemini.get("parallel_tool_calls").is_none_or(Value::is_null));
        // Provenance of a configured model.
        let adaptive = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::AnthropicAdaptive { effort: None });
        let info = ChunkModelInfo::new(&Model::anthropic_claude_opus_5_5(), &adaptive);
        assert_eq!(info.model, "claude-opus-5-5");
        assert_eq!(
            info.reasoning,
            json!({"AnthropicAdaptive": {"effort": null}})
        );
        let plain = ChunkModelInfo::new(&Model::openai_gpt_5_5(), &GenerationConfig::default());
        assert_eq!(plain.reasoning, Value::Null);
    }

    #[test]
    fn voyage_code_4_gets_a_conservative_request_limit() {
        let code = embedding_limits(&EmbeddingModelDetails::voyage_code_4()).unwrap();
        assert_eq!(
            (code.item_tokens, code.batch_tokens, code.batch_items),
            (32_000, 120_000, 1_000)
        );
        let published = embedding_limits(&EmbeddingModelDetails::voyage_4()).unwrap();
        assert_eq!(
            (
                published.item_tokens,
                published.batch_tokens,
                published.batch_items
            ),
            (32_000, 320_000, 1_000)
        );
    }

    #[test]
    fn openai_chunk_models_are_measured_in_o200k_tokens() {
        for model in [
            Model::openai_gpt_5_6_luna(),
            Model::openai_gpt_6_luna(),
            Model::openai_gpt_5_5(),
        ] {
            assert_eq!(chunk_tokenizer(&model), Some(ChunkTokenizer::O200kBase));
        }
        for model in [
            Model::anthropic_claude_opus_5_5(),
            Model::google_gemini_3_5_flash(),
            Model::openrouter("openai/gpt-5.6-luna"),
        ] {
            assert_eq!(chunk_tokenizer(&model), None, "{}", model.id);
        }
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
