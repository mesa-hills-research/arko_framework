use crate::agent_config::PlanUsageAction;
use crate::task::{PendingToolCallInfo, RefusedInput, TaskState};
use arko_ai::{ModelCallUsage, PlanUsage, Refusal, UsageInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Events emitted by the agent runtime during task execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentEvent {
    /// The task state has changed.
    TaskStateChanged(TaskState),
    /// A chunk of the assistant's response text.
    AssistantResponseChunk(String),
    /// A chunk of the assistant's reasoning (for models that support it).
    ReasoningChunk(String),

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
    /// is what the agent's conversation will contain; `original_output`
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
    },
    /// The model or a filter around it refused the turn. Emitted before the
    /// refusal is handled as a non-retryable error (an `ErrorEncountered`
    /// event and task state change follow). For a model's own refusal
    /// ([`arko_ai::RefusalSource::Model`]) the turn stays in history with the
    /// refusal explanation as its text; classifier and content-filter
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
    /// Follows `RefusalReceived`; the refused turn is not kept.
    RefusalFallback { model_id: String },
    /// A retry is scheduled; delay includes any provider minimum.
    RetryScheduled {
        attempt: u32,
        delay_ms: u64,
        elapsed_ms: u64,
        /// The retry waits out a provider rate limit (`false` in events
        /// saved before this field existed).
        #[serde(default)]
        rate_limit: bool,
    },
    /// The LLM attempted to use a tool that is not available.
    UnavailableToolAttempted { tool_name: String },
    /// A general log message from the agent.
    TaskLog(String),
    /// The task stream has ended.
    TaskStreamEnd,
    /// LLM usage information for a turn (cost and raw token counts).
    UsageReported(UsageInfo),
    /// Unified model call usage metrics for a turn.
    ///
    /// Includes provider-normalized input token count, context usage percentage,
    /// and `input_delta_from_previous_call` (computed by the agent runtime).
    /// Emitted once per LLM call, immediately after `UsageReported`.
    ModelCallUsageReported(ModelCallUsage),
    /// Subscription plan usage the provider reported with a model call (the
    /// ChatGPT backend on ChatGPT OAuth calls; see
    /// [`ModelCallUsage::plan_usage`]). Emitted after
    /// `ModelCallUsageReported` for every call that reports it.
    PlanUsageReported(PlanUsage),
    /// Plan usage reached the opt-in threshold
    /// ([`MonitoringConfig::plan_usage`](crate::agent_config::MonitoringConfig::plan_usage)),
    /// before a model request. Reported once per window and reset; `action`
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
    /// requested ([`ModelCallUsage::served_model_changed`]): an OpenRouter
    /// fallback, router or alias, or a ChatGPT backend reroute. Emitted after
    /// `ModelCallUsageReported`.
    ModelRerouted {
        provider: String,
        requested_model: String,
        served_model: String,
    },
    /// The conversation's prompt-cache efficiency after a model call (opt-in
    /// [`MonitoringConfig::cache`](crate::agent_config::MonitoringConfig::cache)):
    /// cache reads divided by the most that could have been read, for the
    /// call and over the last `calls` judged calls.
    CacheEfficiencyReported {
        call_efficiency: f64,
        rolling_efficiency: f64,
        calls: usize,
    },
    /// The rolling cache efficiency fell below a monitor threshold
    /// (`warn_below`, or `pause_below` with `pausing` set: the task pauses
    /// before its next request). Reported once until it recovers.
    CacheEfficiencyLow {
        rolling_efficiency: f64,
        threshold: f64,
        pausing: bool,
    },
    /// A plan usage limit was reached (ChatGPT `usage_limit_reached`, or a
    /// rate limit whose reset is further off than
    /// [`LLMRateLimitRetryConfig::max_wait_ms`](crate::agent_config::LLMRateLimitRetryConfig::max_wait_ms)).
    /// Under [`UsageLimitPolicy::WaitUntilReset`](crate::agent_config::UsageLimitPolicy::WaitUntilReset)
    /// the runtime waits until `waiting_until_unix_ms` and retries; otherwise
    /// it stops retrying (a task state change and `ErrorEncountered` follow).
    UsageLimitReached {
        /// When the limit resets, in Unix milliseconds, when the provider
        /// said.
        resets_at_unix_ms: Option<u64>,
        /// When the runtime retries, in Unix milliseconds, if it waits.
        waiting_until_unix_ms: Option<u64>,
        error_message: String,
    },
}
