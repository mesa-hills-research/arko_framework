//! LLM-backed, context-aware tool-result distillation.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use arko_agents::{InterceptedResult, TaskContext, ToolError, ToolResultInterceptor};

use crate::context::SharedContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    pub name: String,
    pub model: arko_ai::Model,
    pub target_agents: HashSet<String>,
    pub target_tools: HashSet<String>,
    pub min_chars: usize,
    pub max_output_tokens: usize,
    pub instructions: String,
    /// Optional reasoning configuration for the distillation model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<arko_ai::ReasoningConfig>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use arko_ai::{Model, OpenAIEffort, ReasoningConfig};

    #[test]
    fn reasoning_reaches_generation_config_and_legacy_configs_still_load() {
        let config = DistillationConfig::new("bulk", Model::openai_gpt_5_6_luna())
            .reasoning(ReasoningConfig::openai(OpenAIEffort::High, false))
            .max_output_tokens(2_000);
        let mut json = serde_json::to_value(&config).unwrap();
        let restored: DistillationConfig = serde_json::from_value(json.clone()).unwrap();
        restored.validate().unwrap();
        let generation = restored.generation_config();
        assert_eq!(generation.max_tokens, Some(2_000));
        assert!(matches!(
            generation.reasoning,
            Some(ReasoningConfig::OpenAI {
                effort: OpenAIEffort::High,
                include_summaries: false,
            })
        ));
        json.as_object_mut().unwrap().remove("reasoning");
        let legacy: DistillationConfig = serde_json::from_value(json).unwrap();
        assert!(legacy.generation_config().reasoning.is_none());
    }

    #[test]
    fn invalid_distillers_are_rejected_during_workflow_build() {
        for config in [
            DistillationConfig::new("zero", Model::openai_gpt_5_6_luna()).max_output_tokens(0),
            DistillationConfig::new("mismatch", Model::openai_gpt_5_6_luna())
                .reasoning(ReasoningConfig::anthropic(1_024)),
        ] {
            assert!(matches!(
                crate::Workflow::builder().distiller(config).build(),
                Err(crate::WorkflowError::BuilderInvalid(_))
            ));
        }
    }

    #[test]
    fn thresholds_and_target_filters_are_composed() {
        let config = DistillationConfig::new("bulk", Model::openai_gpt_5_6_luna())
            .min_chars(5)
            .target_agent("researcher")
            .target_tool("search");
        assert!(!config.applies("researcher", "search", "éééé".chars().count()));
        assert!(config.applies("researcher", "search", "ééééé".chars().count()));
        assert!(!config.applies("builder", "search", 100));
        assert!(!config.applies("researcher", "read_file", 100));
        assert!(
            DistillationConfig::new("all", Model::openai_gpt_5_6_luna())
                .min_chars(0)
                .applies("any", "any", 0)
        );
    }

    #[test]
    fn the_default_output_limit_leaves_room_for_reasoning_within_the_model_limit() {
        assert_eq!(
            DistillationConfig::new("a", Model::anthropic_claude_opus_5_5()).max_output_tokens,
            16_000
        );
        assert_eq!(
            DistillationConfig::new("b", Model::anthropic_claude_3_5_sonnet_20241022())
                .max_output_tokens,
            8_192
        );
        // An OpenRouter model without catalog metadata has no known limit.
        assert_eq!(
            DistillationConfig::new("c", Model::openrouter("anthropic/claude-sonnet-5"))
                .max_output_tokens,
            16_000
        );
        for model in Model::all() {
            let config = DistillationConfig::new("d", model.clone());
            assert!(config.max_output_tokens > 0, "{}", model.id);
            assert!(
                model.max_output_tokens == 0 || config.max_output_tokens <= model.max_output_tokens,
                "{}",
                model.id
            );
        }
    }

    fn response(
        text: &str,
        stop_reason: arko_ai::StopReason,
        refusal: Option<arko_ai::Refusal>,
    ) -> arko_ai::GenerationResponse {
        arko_ai::GenerationResponse {
            message: arko_ai::Message::new_assistant(text),
            usage: arko_ai::UsageInfo::new(0, 0),
            model_call_usage: arko_ai::ModelCallUsage::new(
                "id".into(),
                "test".into(),
                "test".into(),
                0,
                0,
                0,
            ),
            stop_reason,
            refusal,
            openai_turn_state: None,
        }
    }

    #[test]
    fn only_a_finished_unrefused_summary_with_text_is_usable() {
        use arko_ai::{Refusal, RefusalSource, StopReason};
        for stop in [StopReason::EndTurn, StopReason::StopSequence] {
            assert_eq!(
                complete_summary(&response("short summary", stop, None)).unwrap(),
                "short summary"
            );
        }
        // (response, reason, retryable): refusals and token-limit cuts would
        // most likely repeat, so they are not retried.
        let rejected = [
            (
                response("The failing test is in", StopReason::MaxTokens, None),
                "summary truncated",
                false,
            ),
            (
                response("The failing", StopReason::ContextWindowExceeded, None),
                "summary truncated",
                false,
            ),
            (
                response("Partial summary", StopReason::PauseTurn, None),
                "summary incomplete",
                true,
            ),
            (
                response(
                    "Partial summary",
                    StopReason::Other("cancelled".into()),
                    None,
                ),
                "summary incomplete",
                true,
            ),
            // No stop reason: a live response without one (OpenRouter
            // `finish_reason: null`, Anthropic `stop_reason: null`) or one
            // saved before stop reasons existed. Completeness can't be told.
            (
                response("Summary", StopReason::Other("unknown".into()), None),
                "summary incomplete",
                true,
            ),
            (
                response("Summary", StopReason::ToolUse, None),
                "summary incomplete",
                true,
            ),
            (
                response("  \n", StopReason::EndTurn, None),
                "returned no text",
                true,
            ),
            // A model refusal keeps the turn, so text can accompany it.
            (
                response(
                    "Here is a summ",
                    StopReason::Refusal,
                    Some(Refusal::new(RefusalSource::Model)),
                ),
                "refused by model",
                false,
            ),
            // Even with a clean stop reason, a refusal is never a summary.
            (
                response(
                    "Here is a summary",
                    StopReason::EndTurn,
                    Some(Refusal::new(RefusalSource::Model)),
                ),
                "refused by model",
                false,
            ),
            (
                response(
                    "",
                    StopReason::Refusal,
                    Some(Refusal::new(RefusalSource::SafetyClassifier)),
                ),
                "refused by safety classifier",
                false,
            ),
            (
                response(
                    "",
                    StopReason::Refusal,
                    Some(Refusal::new(RefusalSource::ContentFilter)),
                ),
                "refused by content filter",
                false,
            ),
        ];
        for (response, reason, retryable) in rejected {
            let rejection = complete_summary(&response).unwrap_err();
            assert!(
                rejection.reason.contains(reason),
                "{rejection:?} for {response:?}"
            );
            assert_eq!(
                rejection.retryable, retryable,
                "{rejection:?} for {response:?}"
            );
        }
    }

    #[test]
    fn records_saved_before_rejections_were_recorded_still_load() {
        let record = DistillationRecord {
            timestamp: chrono::Utc::now(),
            task_id: uuid::Uuid::new_v4(),
            agent_name: "researcher".into(),
            tool_name: "search".into(),
            config_name: "bulk".into(),
            input_chars: 10,
            output_chars: 5,
            cost: 0.5,
            rejection: None,
        };
        let json = serde_json::to_value(&record).unwrap();
        assert!(json.get("rejection").is_none());
        let legacy: DistillationRecord = serde_json::from_value(json).unwrap();
        assert!(legacy.rejection.is_none());
        let rejected = DistillationRecord {
            rejection: Some("refused by model".into()),
            ..record
        };
        let restored: DistillationRecord =
            serde_json::from_value(serde_json::to_value(&rejected).unwrap()).unwrap();
        assert_eq!(restored.rejection.as_deref(), Some("refused by model"));
    }

    /// A local HTTP server answering every request with the same JSON body
    /// until dropped.
    pub(crate) struct MockServer {
        pub url: String,
        requests: Arc<std::sync::atomic::AtomicUsize>,
        task: tokio::task::JoinHandle<()>,
    }

    impl MockServer {
        pub async fn start(body: serde_json::Value) -> Self {
            Self::start_with_status(200, body).await
        }

        pub async fn start_with_status(status: u16, body: serde_json::Value) -> Self {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = requests.clone();
            let body = body.to_string();
            let task = tokio::spawn(async move {
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    loop {
                        let mut chunk = [0; 8192];
                        let n = socket.read(&mut chunk).await.unwrap();
                        assert!(n > 0, "connection closed before the request was complete");
                        request.extend_from_slice(&chunk[..n]);
                        let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                            continue;
                        };
                        let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let length: usize = head
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .map_or(0, |value| value.trim().parse().unwrap());
                        if request.len() >= end + 4 + length {
                            break;
                        }
                    }
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let response = format!(
                        "HTTP/1.1 {status} Test\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    socket.write_all(response.as_bytes()).await.unwrap();
                }
            });
            Self {
                url,
                requests,
                task,
            }
        }

        pub fn requests(&self) -> usize {
            self.requests.load(std::sync::atomic::Ordering::SeqCst)
        }

        pub fn anthropic_client(&self) -> arko_ai::LLMClient {
            arko_ai::LLMClient::new().with_anthropic_client(
                arko_ai::providers::anthropic::AnthropicClient::new("test-key")
                    .with_base_url(&self.url),
            )
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    pub(crate) fn anthropic_reply(text: &str, stop_reason: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason,
            "usage": {"input_tokens": 10_000, "output_tokens": 4_000}
        })
    }

    fn workflow(client: arko_ai::LLMClient) -> crate::Workflow {
        let config = arko_agents::AgentConfig::builder(
            "researcher".into(),
            "Researcher".into(),
            arko_agents::AgentLLMConfig {
                model: Model::anthropic_claude_opus_5_5(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        crate::Workflow::builder()
            .name("distillation-tests")
            .workspace(std::env::temp_dir())
            .llm_client(Arc::new(client))
            .agent("researcher", config)
            .ephemeral()
            .build()
            .unwrap()
    }

    /// Runs one interception of a large tool result against `model`, served
    /// by `client`, returning the outcome and the context it charged.
    async fn distill(
        client: arko_ai::LLMClient,
        model: Model,
    ) -> (Result<InterceptedResult, ToolError>, Arc<SharedContext>) {
        let workflow = workflow(client.clone());
        let interceptor = DistillationInterceptor {
            agent_name: "researcher".to_string(),
            config: DistillationConfig::new("bulk", model).min_chars(10),
            client: Arc::new(client),
            context: workflow.context().clone(),
        };
        let task = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "researcher",
            initial_task_prompt: "find the failing test",
            current_working_directory: &workflow.context().workspace,
        };
        let outcome = interceptor
            .intercept(
                "search",
                &"raw tool output ".repeat(10),
                &HashMap::new(),
                &task,
            )
            .await;
        (outcome, workflow.context().clone())
    }

    #[tokio::test]
    async fn truncated_and_refused_summaries_keep_the_tool_output_without_a_retry_but_are_charged()
    {
        for (reply, reason, output_chars) in [
            (
                anthropic_reply("The failing test is in", "max_tokens"),
                "summary truncated (stop reason MaxTokens)",
                "The failing test is in".len(),
            ),
            // The classifier's partial output moves out of the message and
            // is still counted.
            (
                anthropic_reply("The failing test", "refusal"),
                "refused by safety classifier",
                "The failing test".len(),
            ),
        ] {
            let server = MockServer::start(reply).await;
            let (outcome, context) = distill(
                server.anthropic_client(),
                Model::anthropic_claude_opus_5_5(),
            )
            .await;
            assert!(
                matches!(outcome, Ok(InterceptedResult::Unchanged)),
                "a {reason} summary keeps the tool output and is not retried"
            );
            assert_eq!(server.requests(), 1);
            let spent = context.budget.total_spent();
            assert!(spent > 0.0, "the billed call is charged");
            let records = context.distillation_records.read();
            assert_eq!(records.len(), 1);
            assert!(records[0].rejection.as_deref().unwrap().contains(reason));
            assert_eq!(records[0].output_chars, output_chars);
            assert!((records[0].cost - spent).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn an_incomplete_or_empty_summary_is_a_retryable_error_and_charged() {
        for (reply, reason) in [
            (anthropic_reply("", "end_turn"), "returned no text"),
            (
                anthropic_reply("Summary", "pause_turn"),
                "summary incomplete (stop reason PauseTurn)",
            ),
        ] {
            let server = MockServer::start(reply).await;
            let (outcome, context) = distill(
                server.anthropic_client(),
                Model::anthropic_claude_opus_5_5(),
            )
            .await;
            let Err(ToolError::ExecutionFailed(error)) = outcome else {
                panic!("a {reason} summary is an interceptor error");
            };
            assert_eq!(error, format!("distiller 'bulk' result not used: {reason}"));
            assert!(context.budget.total_spent() > 0.0);
            assert_eq!(
                context.distillation_records.read()[0].rejection.as_deref(),
                Some(reason)
            );
        }
    }

    #[tokio::test]
    async fn a_model_refusal_with_text_does_not_replace_the_tool_output() {
        // OpenRouter reports a model refusal next to kept message text.
        let server = MockServer::start(serde_json::json!({
            "id": "gen-1", "object": "chat.completion", "created": 1,
            "model": "anthropic/claude-sonnet-5", "provider": "Anthropic",
            "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "end_turn",
                "message": {"role": "assistant", "content": "Here is a summ",
                    "refusal": "I can't help with that."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6, "cost": 0.25}
        }))
        .await;
        let client = arko_ai::LLMClient::new().with_openrouter_client(
            arko_ai::providers::openrouter::OpenRouterClient::new("sk-or-test")
                .with_base_url(format!("{}/api/v1", server.url)),
        );
        let (outcome, context) =
            distill(client, Model::openrouter("anthropic/claude-sonnet-5")).await;
        assert!(matches!(outcome, Ok(InterceptedResult::Unchanged)));
        assert_eq!(server.requests(), 1);
        assert_eq!(context.budget.total_spent(), 0.25);
        let records = context.distillation_records.read();
        // The explanation is not counted; the kept message text is.
        assert_eq!(records[0].output_chars, "Here is a summ".len());
        assert_eq!(
            records[0].rejection.as_deref(),
            Some("refused by model: I can't help with that.")
        );
    }

    #[tokio::test]
    async fn a_refusal_reported_as_an_error_keeps_the_tool_output_without_a_retry() {
        let server = MockServer::start_with_status(
            403,
            serde_json::json!({"error": {"code": 403,
                "message": "This content was flagged for possible cybersecurity risk.",
                "metadata": {"error_type": "refusal", "provider_code": "cyber_policy"}}}),
        )
        .await;
        let client = arko_ai::LLMClient::new().with_openrouter_client(
            arko_ai::providers::openrouter::OpenRouterClient::new("sk-or-test")
                .with_base_url(format!("{}/api/v1", server.url)),
        );
        let (outcome, context) = distill(client, Model::openrouter("openai/gpt-6-sol")).await;
        assert!(matches!(outcome, Ok(InterceptedResult::Unchanged)));
        assert_eq!(server.requests(), 1);
        // Nothing was generated or billed.
        assert_eq!(context.budget.total_spent(), 0.0);
        assert!(context.distillation_records.read().is_empty());
    }

    #[tokio::test]
    async fn a_finished_summary_replaces_the_tool_output() {
        let server =
            MockServer::start(anthropic_reply("The failing test is x.rs:12.", "end_turn")).await;
        let (outcome, context) = distill(
            server.anthropic_client(),
            Model::anthropic_claude_opus_5_5(),
        )
        .await;
        let Ok(InterceptedResult::Replaced(summary)) = outcome else {
            panic!("a complete summary replaces the output");
        };
        assert_eq!(summary, "The failing test is x.rs:12.");
        let records = context.distillation_records.read();
        assert_eq!(records.len(), 1);
        assert!(records[0].rejection.is_none());
        assert_eq!(records[0].output_chars, summary.len());
        assert!(context.budget.total_spent() > 0.0);
    }
}

/// The default [`DistillationConfig::max_output_tokens`], lowered to the
/// model's output limit when that is smaller. Reasoning (thinking) tokens
/// count against the limit, and models such as Claude Opus 5.5 (adaptive
/// thinking) or GPT-5.5 and later (medium effort) reason by default, so the
/// limit leaves room for reasoning as well as the summary. It is a cap, not
/// a target: a summary cut off at the limit is billed but not used.
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 16_000;

impl DistillationConfig {
    pub fn new(name: impl Into<String>, model: arko_ai::Model) -> Self {
        // 0 means the limit is unknown (an OpenRouter model without catalog
        // metadata).
        let max_output_tokens = match model.max_output_tokens {
            0 => DEFAULT_MAX_OUTPUT_TOKENS,
            limit => DEFAULT_MAX_OUTPUT_TOKENS.min(limit),
        };
        Self {
            name: name.into(),
            model,
            target_agents: HashSet::new(),
            target_tools: HashSet::new(),
            min_chars: 8_000,
            max_output_tokens,
            instructions: "Preserve every fact relevant to the task, exact errors, paths, identifiers, constraints, and actionable details. Remove repetition and irrelevant bulk. Never invent facts.".to_string(),
            reasoning: None,
        }
    }

    /// Distill only this agent's tool results (by default, every agent's).
    /// `agent` is the name the agent is registered under with
    /// [`WorkflowBuilder::agent`](crate::WorkflowBuilder::agent), not its
    /// `AgentConfig::name`.
    pub fn target_agent(mut self, agent: impl Into<String>) -> Self {
        self.target_agents.insert(agent.into());
        self
    }

    pub fn target_tool(mut self, tool: impl Into<String>) -> Self {
        self.target_tools.insert(tool.into());
        self
    }

    pub fn min_chars(mut self, chars: usize) -> Self {
        self.min_chars = chars;
        self
    }

    pub fn max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    pub fn reasoning(mut self, reasoning: arko_ai::ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), crate::WorkflowError> {
        if self.max_output_tokens == 0 {
            return Err(crate::WorkflowError::BuilderInvalid(format!(
                "distiller '{}' output token limit must be positive",
                self.name
            )));
        }
        if self
            .reasoning
            .as_ref()
            .is_some_and(|reasoning| reasoning.provider() != self.model.provider)
        {
            return Err(crate::WorkflowError::BuilderInvalid(format!(
                "distiller '{}' reasoning provider does not match its model",
                self.name
            )));
        }
        Ok(())
    }

    fn generation_config(&self) -> arko_ai::GenerationConfig {
        let mut config =
            arko_ai::GenerationConfig::default().with_max_tokens(self.max_output_tokens);
        config.reasoning = self.reasoning.clone();
        config
    }

    /// Whether the distiller targets the agent registered as `agent`.
    pub(crate) fn targets_agent(&self, agent: &str) -> bool {
        self.target_agents.is_empty() || self.target_agents.contains(agent)
    }

    fn applies(&self, agent: &str, tool: &str, chars: usize) -> bool {
        chars >= self.min_chars
            && self.targets_agent(agent)
            && (self.target_tools.is_empty() || self.target_tools.contains(tool))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub task_id: uuid::Uuid,
    pub agent_name: String,
    pub tool_name: String,
    pub config_name: String,
    pub input_chars: usize,
    pub output_chars: usize,
    pub cost: f64,
    /// Why the distilled text was not used, or `None` when it replaced the
    /// tool output. Records saved before this field existed were written
    /// only for replacements, so they load as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection: Option<String>,
}

/// Why a distillation response can't replace the tool output.
#[derive(Debug)]
struct Rejection {
    reason: String,
    /// Whether another attempt could turn out differently. A refusal or a
    /// token-limit cut would most likely repeat, at full price, so those
    /// keep the tool output without being retried.
    retryable: bool,
}

/// The response's text content, joined as a summary is.
fn response_text(response: &arko_ai::GenerationResponse) -> String {
    response
        .message
        .content
        .iter()
        .filter_map(arko_ai::Content::as_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The distilled text when `response` is a complete summary, else why it
/// can't stand in for the tool output: a refusal (whatever text accompanies
/// it), a turn cut short by a token limit, a turn that ended any other way
/// than the model finishing (including an unreported stop reason), or no
/// text at all.
fn complete_summary(response: &arko_ai::GenerationResponse) -> Result<String, Rejection> {
    let rejection = |reason: String, retryable: bool| Err(Rejection { reason, retryable });
    if let Some(refusal) = &response.refusal {
        return rejection(refusal.to_string(), false);
    }
    match &response.stop_reason {
        arko_ai::StopReason::EndTurn | arko_ai::StopReason::StopSequence => {}
        reason if reason.is_truncated() => {
            return rejection(format!("summary truncated (stop reason {reason:?})"), false);
        }
        // `Other("unknown")` is a response without a stop reason (OpenRouter
        // `finish_reason: null`, an Anthropic `stop_reason: null`, or a
        // response saved before stop reasons existed): whether the summary
        // is complete can't be told, so it is not used.
        reason => {
            return rejection(format!("summary incomplete (stop reason {reason:?})"), true);
        }
    }
    let text = response_text(response);
    if text.trim().is_empty() {
        return rejection("returned no text".to_string(), true);
    }
    Ok(text)
}

pub(crate) struct DistillationInterceptor {
    /// The name the agent is registered under in the workflow, which target
    /// filters and records use; `TaskContext::agent_name` is the
    /// configuration's display name.
    pub agent_name: String,
    pub config: DistillationConfig,
    pub client: Arc<arko_ai::LLMClient>,
    pub context: Arc<SharedContext>,
}

#[async_trait]
impl ToolResultInterceptor for DistillationInterceptor {
    async fn intercept(
        &self,
        tool_name: &str,
        raw_result: &str,
        params: &HashMap<String, Value>,
        task_context: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        // A stopped run's agent is ending: no summary is paid for.
        if self.context.run_control.is_stopped()
            || !self
                .config
                .applies(&self.agent_name, tool_name, raw_result.chars().count())
        {
            return Ok(InterceptedResult::Unchanged);
        }

        let mut conversation = arko_ai::Conversation::new(self.config.model.clone())
            .with_system_message(&format!(
                "You distill tool output for another agent. {}",
                self.config.instructions
            ))
            .with_cache_disabled();
        conversation.add_user_message(&format!(
            "Agent task:\n{}\n\nTool: {}\nParameters: {}\n\nRaw tool result:\n{}",
            task_context.initial_task_prompt,
            tool_name,
            serde_json::to_string(params).unwrap_or_else(|_| "{}".to_string()),
            raw_result
        ));
        let response = match self
            .client
            .generate_text(&conversation, self.config.generation_config())
            .await
        {
            Ok(response) => response,
            // A refusal reported as an error (such as an OpenRouter
            // moderation 403) would repeat like any other refusal.
            Err(arko_ai::Error::Refusal(refusal)) => {
                log::warn!(
                    "Task {}: distiller '{}' request for tool '{}' was {refusal}; keeping the tool output",
                    task_context.task_id,
                    self.config.name,
                    tool_name
                );
                return Ok(InterceptedResult::Unchanged);
            }
            Err(error) => {
                return Err(ToolError::ExecutionFailed(format!(
                    "distiller '{}' failed: {error}",
                    self.config.name
                )));
            }
        };
        // A refused or cut-off summary is billed like a complete one, so
        // every returned response is charged and recorded, used or not.
        let summary = complete_summary(&response);
        let cost = response.usage.billed();
        let cost_result = self.context.budget.add_manual_cost(
            cost,
            &format!("distillation: {}", self.config.name),
            Some(format!(
                "agent={}, task={}, tool={}",
                self.agent_name, task_context.task_id, tool_name
            )),
        );
        if cost_result.warning_triggered {
            crate::workflow::engine::fire_hooks_on_context(
                &self.context,
                crate::hooks::HookPoint::OnBudgetWarning,
                |hook| hook.budget_spent = cost_result.new_total,
            )
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        }
        if cost_result.exceeded {
            crate::workflow::engine::fire_hooks_on_context(
                &self.context,
                crate::hooks::HookPoint::OnBudgetExceeded,
                |hook| hook.budget_spent = cost_result.new_total,
            )
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        }
        // The text the model generated, joined as a summary is: the message
        // text, plus the partial output a refusal moved out of the message
        // (safety classifier, content filter). A refusal's explanation is not
        // counted.
        let partial_output = response
            .refusal
            .as_ref()
            .and_then(|refusal| refusal.partial_output.as_deref())
            .unwrap_or_default();
        let output_chars =
            response_text(&response).chars().count() + partial_output.chars().count();
        self.context
            .distillation_records
            .write()
            .push(DistillationRecord {
                timestamp: chrono::Utc::now(),
                task_id: task_context.task_id,
                agent_name: self.agent_name.clone(),
                tool_name: tool_name.to_string(),
                config_name: self.config.name.clone(),
                input_chars: raw_result.chars().count(),
                output_chars,
                cost,
                rejection: summary
                    .as_ref()
                    .err()
                    .map(|rejection| rejection.reason.clone()),
            });
        match summary {
            Ok(distilled) => Ok(InterceptedResult::Replaced(distilled)),
            // Keeps the tool output without a retry.
            Err(Rejection {
                reason,
                retryable: false,
            }) => {
                log::warn!(
                    "Task {}: distiller '{}' result for tool '{}' not used ({reason}); keeping the tool output",
                    task_context.task_id,
                    self.config.name,
                    tool_name
                );
                Ok(InterceptedResult::Unchanged)
            }
            // The interception failure policy may retry; the tool output is
            // kept once it gives up.
            Err(Rejection { reason, .. }) => Err(ToolError::ExecutionFailed(format!(
                "distiller '{}' result not used: {reason}",
                self.config.name
            ))),
        }
    }
}
