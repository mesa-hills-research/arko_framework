use crate::agent_config::PlanUsageAction;
use crate::task::{ApprovalReason, PendingToolCallInfo, RefusedInput, TaskState};
use arko_ai::{ModelCallUsage, PlanUsage, Refusal, UsageInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Events emitted by the agent runtime during task execution.
// Events are built once and sent through channels, so the size of the largest
// variant costs little, and boxing it would change the public variants.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentEvent {
    /// The task state has changed.
    TaskStateChanged(TaskState),
    /// A chunk of the assistant's response text. The chunks after a
    /// `ModelRequestStarted` of the conversation's own requests (`subagent`
    /// `None`) belong to that request's response, which ends with its
    /// `UsageReported`. When it fails after streaming, a
    /// `StreamedOutputDiscarded` follows its last chunk.
    AssistantResponseChunk(String),
    /// A chunk of the assistant's reasoning (for models that support it),
    /// belonging to a response as `AssistantResponseChunk` does.
    ReasoningChunk(String),
    /// The response the latest `ModelRequestStarted` began failed after
    /// streaming part of it. The `AssistantResponseChunk` and
    /// `ReasoningChunk` events since then belong to no response, and a host
    /// that showed them drops them (`text_bytes` and `reasoning_bytes` say
    /// how much of each). A retry streams its response anew, after its own
    /// `ModelRequestStarted`. Claude Code may ask for it again without
    /// streaming instead, and that response's text then comes whole, with no
    /// `ModelRequestStarted` before it. The task keeps nothing of the failed
    /// one.
    StreamedOutputDiscarded {
        text_bytes: usize,
        reasoning_bytes: usize,
    },
    /// Images of the task's conversation are gone (their saved files were
    /// deleted or damaged), each now a line saying so in its place, with one
    /// reason per image. Reported as the run starts.
    ImagesUnavailable { reasons: Vec<String> },

    /// A tool call has been initiated.
    ToolCallInitiated {
        tool_call_id: Uuid,
        tool_name: String,
    },
    /// A parameter name has been parsed (granular event, output-based mode only).
    ToolParamName {
        tool_call_id: Uuid,
        param_name: String,
    },
    /// A chunk of a parameter value has been parsed (granular event, output-based mode only).
    ToolParamValueChunk {
        tool_call_id: Uuid,
        param_name: String,
        chunk: String,
    },
    /// A parameter has finished being parsed (granular event, output-based mode only).
    ToolParamEnd {
        tool_call_id: Uuid,
        param_name: String,
    },
    /// A complete tool call has been parsed and formatted.
    ToolCallFormatted {
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
    },
    /// A single tool call is awaiting user approval.
    ToolCallAwaitingApproval {
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
        tool_description: String,
        /// Why the call awaits approval (not recorded in events saved by an
        /// older version).
        #[serde(default, skip_serializing_if = "ApprovalReason::is_unrecorded")]
        approval_reason: ApprovalReason,
    },
    /// A batch of tool calls is awaiting user approval.
    BatchToolCallAwaitingApproval {
        batch_id: Uuid,
        pending_calls: Vec<PendingToolCallInfo>,
    },
    /// The agent has asked questions and is awaiting user response.
    UserQuestionsAsked {
        /// The questions asked by the agent.
        questions: String,
    },
    /// The agent sent the user a message without ending its turn. It called
    /// the built-in `message_user` tool, or, with
    /// [`AgentToolConfig::text_reply_as_message`](crate::agent_config::AgentToolConfig::text_reply_as_message),
    /// replied in plain text without a tool call. Nothing waits for an
    /// answer, and the task's state doesn't change. For a call, it follows
    /// the call's `ToolCallFormatted` and comes before its `ToolResult`. A
    /// reply in plain text has already streamed as `AssistantResponseChunk`
    /// events.
    MessageToUser {
        /// The message, as the agent wrote it.
        message: String,
        /// The `message_user` call that sent it. `None` for a reply in
        /// plain text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<Uuid>,
    },
    /// A tool has finished executing with a result.
    ///
    /// Contains the **original** tool output before any interception.
    /// If a `ToolResultInterceptor` modified the result, a separate
    /// `ToolResultIntercepted` event follows with both values.
    ToolResult {
        tool_call_id: Uuid,
        tool_name: String,
        output: String,
    },
    /// A tool result was modified by a `ToolResultInterceptor`.
    ///
    /// Emitted immediately after the corresponding `ToolResult` event
    /// when interception produced a different result. The `intercepted_output`
    /// is what the agent's conversation will contain. `original_output`
    /// matches the preceding `ToolResult.output`.
    ToolResultIntercepted {
        tool_call_id: Uuid,
        tool_name: String,
        original_output: String,
        intercepted_output: String,
    },
    /// A tool was rejected by the user.
    ToolRejected {
        tool_call_id: Uuid,
        tool_name: String,
        reason: Option<String>,
    },
    /// An error was encountered during execution.
    ErrorEncountered {
        error_message: String,
        requires_approval_to_retry: bool,
        tool_call_id: Option<Uuid>,
        /// The kind of a failed model request, when it was classified
        /// ([`arko_ai::Error::kind`]). `None` for other errors, unclassified
        /// ones, and in events saved by an older version.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_kind: Option<arko_ai::ErrorKind>,
    },
    /// A tool result was shortened before it went into the conversation
    /// by the overflow guard, by an opt-in cap, or to recover from a
    /// request the provider found over the model's input limit (see
    /// [`ToolResultLimits`](crate::agent_config::ToolResultLimits)). The
    /// `ToolResult` event before it has the whole output. The conversation
    /// holds the head and tail kept and a notice.
    ToolResultShortened {
        /// The provider's id of the call the result answers.
        call_id: Option<String>,
        tool_name: String,
        /// The result's estimated size, in tokens.
        original_tokens: usize,
        /// The estimated size of what was kept, in tokens.
        kept_tokens: usize,
        reason: ShortenReason,
    },
    /// The model or a filter around it refused the turn. Emitted before the
    /// refusal is handled as a non-retryable error (an `ErrorEncountered`
    /// event and task state change follow). For a model's own refusal
    /// ([`arko_ai::RefusalSource::Model`]) the turn stays in history with the
    /// refusal explanation as its text. Classifier and content-filter
    /// refusals are removed from history. With the opt-in refusal fallback,
    /// `RefusalFallback` follows instead and the task continues.
    RefusalReceived(Refusal),
    /// The input the refused turn answered (follows `RefusalReceived` when
    /// the task stops for the refusal). The host can change it with
    /// `UserCommand::ResolveRefusal` (or [`Task::resolve_refusal`](crate::task::Task::resolve_refusal)
    /// on a returned task) and continue.
    RefusedInput(RefusedInput),
    /// The refused turn is sent again, once, to the opt-in fallback model
    /// ([`AgentConfig::refusal_fallback`](crate::agent_config::AgentConfig::refusal_fallback)).
    /// Follows `RefusalReceived`. The refused turn is not kept.
    RefusalFallback { model_id: String },
    /// A retry is scheduled. Delay includes any provider minimum.
    RetryScheduled {
        attempt: u32,
        delay_ms: u64,
        elapsed_ms: u64,
        /// The retry waits out a provider rate limit (`false` in events
        /// saved before this field existed).
        #[serde(default)]
        rate_limit: bool,
    },
    /// The model called a tool the task can't run. In native mode, this can be
    /// a tool it is not offered (see
    /// [`AgentToolConfig::available_tool_names`](crate::agent_config::AgentToolConfig::available_tool_names)),
    /// or in either mode an offered tool the registry doesn't hold. The call
    /// was not run.
    UnavailableToolAttempted { tool_name: String },
    /// The task stream has ended.
    TaskStreamEnd,
    /// A model request of the task is going out. In arko's own runtime it
    /// is emitted just before each request is sent, for every attempt, a retry
    /// and a refusal fallback's request included. A Claude Code model's
    /// requests are made by Claude Code, which shows one when its response
    /// starts (its `message_start`), so it is emitted then, for the
    /// conversation's own requests and its subagents'. A prompt cache is
    /// refreshed by each request that reads or writes it, so a host times
    /// the cache's lifetime from the latest one (its
    /// `WorkflowEvent::timestamp` in arko_workflows). The
    /// `ModelCallUsageReported` that follows tells whether it read or
    /// wrote the cache.
    ModelRequestStarted {
        /// The model the request goes to, given by the configured model's id, or the
        /// refusal fallback's, or the model Claude Code reports.
        model: String,
        /// For a request of a subagent the model runs itself (Claude Code's
        /// `Agent` tool), the id of the tool call that started it. `None`
        /// for the conversation's own requests.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subagent: Option<String>,
        /// The request's size by section, as the runtime knows it. This covers what it
        /// sends in arko's own runtime or, for a Claude Code model, the task's
        /// conversation, system prompt and arko tools, without what Claude
        /// Code adds of its own. `None` for a subagent's request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sizes: Option<arko_ai::RequestSizes>,
        /// The bytes per token of the task's earlier requests (their text
        /// bytes over the input tokens the provider reported, see
        /// [`calibrated_bytes_per_token`](crate::runtime::calibrated_bytes_per_token)),
        /// which turns `sizes` into estimated tokens, or
        /// [`DEFAULT_BYTES_PER_TOKEN`](crate::runtime::DEFAULT_BYTES_PER_TOKEN)
        /// until one was answered.
        bytes_per_token: f64,
    },
    /// LLM usage information for a turn (cost and raw token counts).
    UsageReported(UsageInfo),
    /// Unified model call usage metrics for a turn.
    ///
    /// Includes provider-normalized input token count, context usage percentage,
    /// and `input_delta_from_previous_call` (computed by the agent runtime).
    /// Emitted once per LLM call, immediately after `UsageReported`.
    ModelCallUsageReported(ModelCallUsage),
    /// Subscription plan usage the provider reported with a model call (the
    /// ChatGPT backend on ChatGPT OAuth calls, see
    /// [`ModelCallUsage::plan_usage`]). Emitted after
    /// `ModelCallUsageReported` for every call that reports it.
    PlanUsageReported(PlanUsage),
    /// Plan usage reached the opt-in threshold
    /// ([`MonitoringConfig::plan_usage`](crate::agent_config::MonitoringConfig::plan_usage)),
    /// before a model request. Reported once per window and reset. `action`
    /// follows.
    PlanUsageThresholdReached {
        /// `"primary"` or `"secondary"`.
        window: String,
        used_percent: f64,
        threshold_percent: f64,
        /// When the window resets, in Unix milliseconds, when reported.
        resets_at_unix_ms: Option<u64>,
        action: PlanUsageAction,
    },
    /// The provider served a model call with another model than the one
    /// requested ([`ModelCallUsage::served_model_changed`]) through an OpenRouter
    /// fallback, router or alias, or a ChatGPT backend reroute. Emitted after
    /// `ModelCallUsageReported`.
    ModelRerouted {
        provider: String,
        requested_model: String,
        served_model: String,
    },
    /// The provider refused the reasoning summaries the run's configuration
    /// asks for ([`ModelCallUsage::reasoning_summaries_refused`], for an OpenAI
    /// organization that isn't verified for them), and the request was sent
    /// again without them. The run asks for none from then on (its saved
    /// configuration too), nor does any request of that provider's client,
    /// so the model's thinking no longer streams. Emitted after
    /// `ModelCallUsageReported`. A host can tell the user once.
    ReasoningSummariesRefused { provider: String, model: String },
    /// The conversation's prompt-cache efficiency after a model call (opt-in
    /// [`MonitoringConfig::cache`](crate::agent_config::MonitoringConfig::cache)).
    /// This is cache reads divided by the most that could have been read, for the
    /// call and over the last `calls` judged calls.
    CacheEfficiencyReported {
        call_efficiency: f64,
        rolling_efficiency: f64,
        calls: usize,
    },
    /// The rolling cache efficiency fell below a monitor threshold
    /// (`warn_below`, or `pause_below` with `pausing` set so the task pauses
    /// before its next request). Reported once until it recovers.
    CacheEfficiencyLow {
        rolling_efficiency: f64,
        threshold: f64,
        pausing: bool,
    },
    /// A model response ended at its output-token limit
    /// ([`arko_ai::StopReason::MaxTokens`]). Its text may be cut short, and a
    /// tool call still being written when the limit was reached was dropped.
    /// Emitted after the call's usage events. The response is then handled
    /// as usual (a turn without a tool call gets the runtime's note). A host
    /// can tell the user, who can ask the agent to continue. `max_tokens` in
    /// the model's generation settings raises the limit where the model
    /// allows.
    OutputTokenLimitReached {
        /// The model that answered, as the provider reported it.
        model: String,
        /// Output tokens the response used, as the provider reported them.
        output_tokens: usize,
    },
    /// A plan usage limit was reached (ChatGPT `usage_limit_reached`, or a
    /// rate limit whose reset is further off than
    /// [`LLMRateLimitRetryConfig::max_wait_ms`](crate::agent_config::LLMRateLimitRetryConfig::max_wait_ms)).
    /// Under [`UsageLimitPolicy::WaitUntilReset`](crate::agent_config::UsageLimitPolicy::WaitUntilReset)
    /// the runtime waits until `waiting_until_unix_ms` and retries. Otherwise
    /// it stops retrying (a task state change and `ErrorEncountered` follow).
    UsageLimitReached {
        /// When the limit resets, in Unix milliseconds, when the provider
        /// said.
        resets_at_unix_ms: Option<u64>,
        /// When the runtime retries, in Unix milliseconds, if it waits.
        waiting_until_unix_ms: Option<u64>,
        error_message: String,
    },
    /// A handover started. The agent was asked for its handover report,
    /// and its tool calls other than `complete_task` (and `handover_report`,
    /// where it is offered) are rejected until it gives it. See
    /// [`crate::handover`].
    HandoverRequested {
        /// The input size of the latest request of the conversation being
        /// replaced. 0 when a host asked before any request.
        input_tokens: usize,
    },
    /// A context update joined the conversation, holding content the host sent with
    /// [`InjectContext`](crate::runtime::command::UserCommand::InjectContext)
    /// and its turn context source's context
    /// ([`TurnContextSource`](crate::runtime::TurnContextSource)), added as
    /// one user message headed `# Workflow context update`. Emitted once the
    /// message is in the conversation (for a Claude Code agent, once it is
    /// sent to the session), so the task has received what it carries.
    ContextInjected {
        /// The host's identifiers of the items the update holds
        /// ([`TurnContext::ids`](crate::runtime::TurnContext::ids), and
        /// [`TurnContextSource::identify`](crate::runtime::TurnContextSource::identify)
        /// for injected content), in the order they were added. Empty when
        /// the host identified none. arko_workflows uses `note:<seq>` for a
        /// note, `message:<id>` for an agent message and `mail:<id>` for a
        /// subagent mailbox item.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        ids: Vec<String>,
        /// The update's size in bytes, without its heading.
        #[serde(default)]
        bytes: usize,
    },
    /// The conversation was replaced with a fresh one that starts from the
    /// task prompt and the report ([`Task::handovers`](crate::task::Task::handovers)
    /// holds the record).
    ConversationHandedOver {
        /// The report the agent wrote.
        report: String,
        /// The input size of the latest request of the replaced
        /// conversation.
        input_tokens: usize,
        /// How many handovers the task has made, this one included.
        handovers: usize,
    },
    /// Claude Code compacted the conversation of an agent whose model runs
    /// in it, at the runtime's request (see
    /// [`HandoverConfig::compact_in_claude_code`](crate::handover::HandoverConfig::compact_in_claude_code))
    /// or on its own. Its summary replaced the earlier messages in the
    /// session, and the agent goes on from it. The task's conversation
    /// keeps every message.
    ConversationCompacted {
        /// The conversation's size before, in tokens, as Claude Code
        /// reported it.
        input_tokens: usize,
        /// Its size after, as Claude Code reported it, when it did.
        compacted_tokens: Option<usize>,
    },
    /// The host cut the conversation back to just before message
    /// `message_index` ([`Task::cut_back`](crate::task::Task::cut_back)),
    /// and `removed_messages` messages were removed for an edit or a regenerate. A
    /// workflow reports it as the run goes on
    /// (`ContinueOptions::cut_back_to`), before the run's own events.
    ConversationCutBack {
        message_index: usize,
        removed_messages: usize,
    },
    /// The host continued the task under another configuration
    /// ([`Task::reconfigure`](crate::task::Task::reconfigure)), such as a
    /// switch between a chat and an agent. The `chat` field says whether it is a chat
    /// now, and `tools` how many tools of its own it is offered. A workflow
    /// reports it as the run goes on (`ContinueOptions::reconfigure`),
    /// before the run's own events.
    Reconfigured { chat: bool, tools: usize },
}

/// Why a tool result was shortened ([`AgentEvent::ToolResultShortened`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortenReason {
    /// The overflow guard shortened the turn's results because they would
    /// have put the next request over the model's input limit.
    Overflow,
    /// The result was over its opt-in cap
    /// ([`ToolResultLimits::max_result_tokens`](crate::agent_config::ToolResultLimits::max_result_tokens)
    /// or the tool's own).
    ResultCap,
    /// The provider found the request over the model's input limit, and
    /// the result was shortened so the request could be sent again.
    OverflowRecovery,
}
