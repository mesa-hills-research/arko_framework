//! Provider boundary. Tests and offline applications can supply deterministic backends.
use crate::{EmbeddingProfile, Result, error::invalid};
use arko_ai::{
    EmbeddingClient, EmbeddingConfig, EmbeddingModelDetails, EmbeddingProviderIdentifier,
    GenerationConfig, LLMClient, Model, OpenAIEffort, ReasoningConfig, StreamEvent, TokenCounter,
    ToolChoice, ToolConfig, ToolDefinition, Verbosity, VoyageInputType,
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
    pub usage: Value,
}
#[derive(Clone, Debug)]
pub struct VectorReply {
    pub vectors: Vec<Vec<f32>>,
    pub billed_tokens: usize,
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
}

pub struct ArkoBackend {
    llm: LLMClient,
    embeddings: EmbeddingClient,
    tokenizer: TokenCounter,
    profile: EmbeddingProfile,
    limits: Limits,
}
impl ArkoBackend {
    /// Loads the pinned Hugging Face tokenizer on a blocking worker. Authentication
    /// and rotated OAuth credential persistence remain with the application.
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
        })
    }
}
#[async_trait::async_trait]
impl Backend for ArkoBackend {
    async fn chunk(&self, request: ModelRequest) -> Result<ModelReply> {
        let config = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, true))
            .with_verbosity(Verbosity::Low)
            .with_tool_config(
                ToolConfig::new(vec![
                    ToolDefinition::new(
                        &request.tool,
                        "Submit the requested source boundaries or repair choices.",
                        request.schema,
                    )
                    .with_strict(true),
                ])
                .with_tool_choice(ToolChoice::Specific(request.tool.clone()))
                .with_parallel_tool_calls(false),
            );
        let mut conversation = self
            .llm
            .new_conversation(Model::openai_gpt_5_6_luna())
            .with_system_message(&request.system);
        conversation.add_user_message(&request.prompt);
        let mut stream = self.llm.generate_stream(&conversation, config).await?;
        while let Some(event) = stream.next().await {
            if let StreamEvent::End(response) = event? {
                if response.model_call_usage.stop_reason.as_deref() != Some("completed") {
                    return Err(invalid(format!(
                        "Luna response incomplete/refused: {:?}",
                        response.model_call_usage.stop_reason
                    )));
                }
                if response.message.tool_calls.len() != 1
                    || response.message.tool_calls[0].name != request.tool
                {
                    return Err(invalid(
                        "Luna did not complete exactly one requested tool call",
                    ));
                }
                return Ok(ModelReply {
                    value: response.message.tool_calls[0].arguments.clone(),
                    usage: serde_json::to_value(&response.model_call_usage)?,
                });
            }
        }
        Err(invalid("Luna stream ended without a completed response"))
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
