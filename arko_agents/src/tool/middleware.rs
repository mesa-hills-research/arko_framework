use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use super::ToolError;

/// Stable task information exposed at tool execution boundaries.
#[derive(Debug, Clone, Copy)]
pub struct ToolExecutionContext<'a> {
    pub task_id: Uuid,
    pub agent_name: &'a str,
    pub initial_task_prompt: &'a str,
    pub current_working_directory: &'a Path,
}

/// A fully parsed tool call about to be executed.
#[derive(Debug)]
pub struct ToolCallContext<'a> {
    pub tool_call_id: Uuid,
    pub tool_name: &'a str,
    pub parameters: &'a HashMap<String, Value>,
}

/// The decision returned before a tool executes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolExecutionDecision {
    Continue,
    Reject {
        message_to_agent: String,
    },
    /// Permanently reject the call and fail the task. This is intended for
    /// policy gates where continuing would be unsafe or an attempt budget has
    /// been exhausted.
    FailTask {
        reason: String,
    },
    /// Reject the call and stop the task as
    /// [`Cancelled`](crate::task::TaskState::Cancelled): the caller is
    /// shutting the task down, for example because the run it belongs to was
    /// cancelled.
    CancelTask {
        reason: String,
    },
    /// Don't run the call, and pause the task for its host's decision: the
    /// call is answered with `message_to_agent` (and reported rejected), and
    /// once the other calls of its batch have run the task awaits
    /// error-recovery approval
    /// ([`AwaitingErrorRecoveryApproval`](crate::task::TaskState::AwaitingErrorRecoveryApproval))
    /// with the cause
    /// [`ErrorRecoveryCause::Middleware`](crate::task::ErrorRecoveryCause::Middleware)
    /// and `reason` as its error. `ContinueAfterError` goes on (the agent
    /// reads its note next), `RejectErrorRecovery` fails the task. For a
    /// decision the middleware can't make, such as a completion check that
    /// could not be carried out: the call is neither run nor refused.
    PauseTask {
        message_to_agent: String,
        reason: String,
    },
}

/// The observed outcome of a tool execution.
#[derive(Debug, Clone, Copy)]
pub enum ToolExecutionOutcome<'a> {
    Success(&'a str),
    Error(&'a str),
    /// The caller supplied an output during approval, so the runtime did not
    /// execute the registered tool itself.
    CallerProvided(&'a str),
}

/// Approval-independent middleware around actual tool execution.
///
/// Unlike approval events, this boundary is used for auto-approved, manually
/// approved, native, output-based, single, and batched calls alike.
#[async_trait]
pub trait ToolExecutionMiddleware: Send + Sync {
    async fn before(
        &self,
        context: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError>;

    async fn after(
        &self,
        context: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
        outcome: ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError>;
}
