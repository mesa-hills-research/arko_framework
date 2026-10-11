//! Context a host adds at a task's safe turn boundaries.

use uuid::Uuid;

/// Context for a task's next model request, with the identifiers of what it
/// holds: [`TurnContextSource::context_update`] returns it, and the
/// identifiers are reported in
/// [`AgentEvent::ContextInjected`](crate::runtime::event::AgentEvent::ContextInjected)
/// once the content is in the conversation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnContext {
    /// The text added to the conversation.
    pub content: String,
    /// The host's identifiers of the items in `content` (such as note
    /// numbers or message ids), in any form it chooses; may be empty.
    pub ids: Vec<String>,
}

impl From<String> for TurnContext {
    fn from(content: String) -> Self {
        Self {
            content,
            ids: Vec::new(),
        }
    }
}

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
    /// before each request (not before a refusal fallback request) through
    /// [`context_update`](Self::context_update)'s default; what it returns
    /// is added to the conversation.
    fn turn_context(&self, task_id: Uuid) -> Option<String>;

    /// [`turn_context`](Self::turn_context) with the identifiers of what it
    /// holds, which the runtime reports in
    /// [`AgentEvent::ContextInjected`](crate::runtime::event::AgentEvent::ContextInjected)
    /// once the content is in the conversation. This is what the runtime
    /// calls; the default returns `turn_context`'s content without
    /// identifiers.
    fn context_update(&self, task_id: Uuid) -> Option<TurnContext> {
        self.turn_context(task_id).map(TurnContext::from)
    }

    /// The host's identifier of `content`, a payload it sent with
    /// [`InjectContext`](crate::runtime::command::UserCommand::InjectContext),
    /// for [`AgentEvent::ContextInjected`](crate::runtime::event::AgentEvent::ContextInjected);
    /// `None` (the default) for content it doesn't identify.
    fn identify(&self, _content: &str) -> Option<String> {
        None
    }

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
