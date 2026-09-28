use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a decision made by the user for a tool call requiring approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolApprovalDecision {
    /// Approve the tool call.
    Approve {
        /// Optional: If the downstream code executes the tool itself as part of the approval workflow,
        /// it can provide the output string here. If `None`, the agent runtime
        /// will call `tool.execute()`.
        custom_tool_output: Option<String>,
        /// Optional message from the user to include with the tool result,
        /// which will be sent to the LLM.
        message_to_agent: Option<String>,
    },
    /// Reject the tool call.
    Reject { reason: Option<String> },
}

/// Commands that can be sent to the agent runtime by the user/application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserCommand {
    /// Queue application-provided context for the agent. The runtime appends it
    /// as a user message at the next safe turn boundary; an in-flight model
    /// stream is never mutated.
    InjectContext { content: String },

    /// Approve the use of a single tool that was awaiting approval.
    ApproveToolUse {
        tool_call_id: Uuid,
        /// Optional: If the downstream code executes the tool, provide output.
        custom_tool_output: Option<String>,
        /// Optional message from the user to include with the tool result.
        message_to_agent: Option<String>,
    },

    /// Reject the use of a single tool that was awaiting approval.
    RejectToolUse {
        tool_call_id: Uuid,
        reason: Option<String>,
    },

    /// Submit decisions for a batch of tool calls that were awaiting granular approval.
    SubmitBatchToolApprovals {
        /// Matches the batch_id from AgentEvent/TaskState.
        batch_id: Uuid,
        /// Key is the individual tool_call_id from PendingToolCallInfo.
        decisions: HashMap<Uuid, ToolApprovalDecision>,
    },

    /// Submit a response to questions asked by the agent via `ask_user` tool.
    SubmitUserResponse {
        /// The user's response to the questions.
        response: String,
    },

    /// Approve recovery after an error (e.g., retry a failed tool execution or LLM call).
    /// The model is told that the previous attempt resulted in the error and
    /// that the user approved continuing, followed by `message` when given.
    ApproveErrorRecovery { message: Option<String> },

    /// Continue after an error-recovery pause as the host's own decision,
    /// without the user-approval note of
    /// [`ApproveErrorRecovery`](Self::ApproveErrorRecovery): with no `note`
    /// the failed request is sent again unchanged (with the tool results the
    /// pause left waiting, which report their errors); with a `note`, it is
    /// added as the runtime's text before the next request. Like an
    /// approval, it starts a new recovery and clears a refused input.
    ContinueAfterError { note: Option<String> },

    /// Reject recovery after an error, potentially leading to task failure.
    RejectErrorRecovery { message: Option<String> },

    /// Change the input of a refused turn (reported with
    /// `AgentEvent::RefusedInput`) and continue: the refused turn is removed
    /// and the changed input is sent. Accepted while the refusal awaits
    /// error-recovery approval; see [`crate::task::Task::resolve_refusal`].
    /// The changes apply together, or none does: an invalid one is reported
    /// with `ErrorEncountered` and the task keeps waiting.
    ResolveRefusal {
        changes: Vec<crate::task::RefusalResolution>,
    },

    /// Request to gracefully cancel the currently running task.
    CancelTask,

    /// Stop the runtime while preserving its current suspended state so the
    /// caller can persist and resume it later.
    SuspendTask,
}
