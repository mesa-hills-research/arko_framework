//! Provider boundary. Tests and offline applications can supply deterministic backends.
use crate::{EmbeddingProfile, Result, error::invalid};
use arko_ai::{
    EmbeddingClient, EmbeddingConfig, EmbeddingModelDetails, EmbeddingProviderIdentifier,
    GenerationConfig, GenerationResponse, LLMClient, Model, OpenAIEffort, ReasoningConfig,
    StopReason, StreamEvent, TokenCounter, ToolCall, ToolChoice, ToolConfig, ToolDefinition,
    Verbosity, VoyageInputType,
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
                // Refusals and token limits are permanent failures; a missing,
                // extra or malformed call is resampled as a transient failure.
                let call = completed_tool_call(&response, &request.tool)?;
                return Ok(ModelReply {
                    value: call.arguments.clone(),
                    // Keeps the provider's raw stop reason with the call record.
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

#[cfg(test)]
mod tests {
    use super::*;
    use arko_ai::{Message, ModelCallUsage, Refusal, RefusalSource, RetryClass, UsageInfo};
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
