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
    ApproveErrorRecovery { message: Option<String> },

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
