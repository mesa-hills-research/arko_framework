//! Context a host adds at a task's safe turn boundaries.

use uuid::Uuid;

/// Supplies context a task receives at each safe turn boundary: before each
/// model request, where queued
/// [`InjectContext`](crate::runtime::command::UserCommand::InjectContext)
/// content is added, as part of the same "Workflow context update" user
/// message. Set with
/// [`AgentRuntime::with_turn_context`](crate::runtime::AgentRuntime::with_turn_context).
///
/// Content is derived when it is added, from the host's state at that
/// moment, rather than queued when something happens.
pub trait TurnContextSource: Send + Sync {
    /// Context for the task's next model request, or `None`. Called once
    /// before each request (not before a refusal fallback request); what it
    /// returns is added to the conversation.
    fn turn_context(&self, task_id: Uuid) -> Option<String>;

    /// Whether to hold the task's next model request, and why: with
    /// `Some(reason)` the request isn't made and the task pauses for
    /// error-recovery approval with
    /// [`ErrorRecoveryCause::BudgetExceeded`](crate::task::ErrorRecoveryCause::BudgetExceeded)
    /// and `reason` as its error, whatever the agent's LLM error policy
    /// (a workflow holds the requests of agents over their budget this
    /// way). Continuing the pause asks again. Called before each request,
    /// refusal fallbacks included; a Claude Code agent is asked before each
    /// turn and at each message of a turn, and a turn it holds is
    /// interrupted. The default holds nothing.
    fn hold_request(&self, _task_id: Uuid) -> Option<String> {
        None
    }
}
