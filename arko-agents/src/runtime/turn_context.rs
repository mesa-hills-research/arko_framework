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
}
