use crate::task::{PendingToolCallInfo, TaskState};
use arko_ai::{ModelCallUsage, Refusal, UsageInfo};
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
    /// refusals are removed from history.
    RefusalReceived(Refusal),
    /// A retry is scheduled; delay includes any provider minimum.
    RetryScheduled {
        attempt: u32,
        delay_ms: u64,
        elapsed_ms: u64,
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
}
