//! Cancelling a workflow run.
//!
//! A run stops when a [`WorkflowCancelHandle`] cancels it or a hook returns
//! [`HookResult::Abort`](crate::hooks::HookResult::Abort). Either way every
//! running agent is told to stop, and every run call in progress or made
//! later returns the stop's error: [`WorkflowError::Cancelled`] or
//! [`WorkflowError::AbortedByHook`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use arko_agents::runtime::command::UserCommand;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::hooks::HookPoint;

/// How long a stopped run waits for each running agent to stop before it
/// aborts the agent; see [`WorkflowBuilder::cancel_grace_period`](crate::WorkflowBuilder::cancel_grace_period).
pub const DEFAULT_CANCEL_GRACE_PERIOD: Duration = Duration::from_secs(30);

/// Why a workflow run was stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelCause {
    /// A [`WorkflowCancelHandle`] cancelled the run.
    Requested,
    /// A hook returned [`HookResult::Abort`](crate::hooks::HookResult::Abort).
    AbortedByHook { hook_point: String, reason: String },
}

impl CancelCause {
    /// The error run calls return once the run stopped for this cause.
    pub fn to_error(&self) -> WorkflowError {
        match self {
            Self::Requested => WorkflowError::Cancelled,
            Self::AbortedByHook { hook_point, reason } => WorkflowError::AbortedByHook {
                hook_point: hook_point.clone(),
                reason: reason.clone(),
            },
        }
    }

    /// The reason recorded on the agents the stop cancels.
    pub fn agent_reason(&self) -> String {
        match self {
            Self::Requested => "Workflow cancelled.".to_string(),
            Self::AbortedByHook { hook_point, reason } => {
                format!("Workflow aborted by {hook_point} hook: {reason}")
            }
        }
    }
}

/// A stopped run: why, and when. Saved in checkpoints as
/// [`WorkflowState::cancellation`](crate::persistence::state::WorkflowState::cancellation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cancellation {
    pub cause: CancelCause,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// The stop state of one run, shared by the engine, tools and handles.
pub(crate) struct RunControl {
    state: watch::Sender<Option<Cancellation>>,
    /// Command channels of the run's agents, to tell them to stop at once.
    agents: parking_lot::Mutex<HashMap<uuid::Uuid, mpsc::WeakSender<UserCommand>>>,
    grace_period: Duration,
    /// When the run stopped, for the drivers' wind-down deadlines.
    stopped_at: parking_lot::Mutex<Option<tokio::time::Instant>>,
    stop_saved: AtomicBool,
}

impl RunControl {
    pub fn new(grace_period: Duration) -> Self {
        Self {
            state: watch::Sender::new(None),
            agents: parking_lot::Mutex::new(HashMap::new()),
            grace_period,
            stopped_at: parking_lot::Mutex::new(None),
            stop_saved: AtomicBool::new(false),
        }
    }

    pub fn grace_period(&self) -> Duration {
        self.grace_period
    }

    /// Stops the run for `cause` and sends every registered agent
    /// `CancelTask`, unless the run is already stopped (the first cause
    /// stands). Returns whether this call stopped the run.
    pub fn stop(&self, cause: CancelCause) -> bool {
        let stopped = self.state.send_if_modified(|state| {
            if state.is_some() {
                return false;
            }
            *state = Some(Cancellation {
                cause,
                timestamp: chrono::Utc::now(),
            });
            true
        });
        if stopped {
            *self.stopped_at.lock() = Some(tokio::time::Instant::now());
            // Queued before any agent takes its next step; the driver's own
            // (awaited) CancelTask covers a full command channel.
            for sender in self.agents.lock().values() {
                if let Some(sender) = sender.upgrade() {
                    let _ = sender.try_send(UserCommand::CancelTask);
                }
            }
        }
        stopped
    }

    pub fn cancellation(&self) -> Option<Cancellation> {
        self.state.borrow().clone()
    }

    /// When the run stopped, once it has.
    pub fn stopped_at(&self) -> Option<tokio::time::Instant> {
        *self.stopped_at.lock()
    }

    pub fn is_stopped(&self) -> bool {
        self.state.borrow().is_some()
    }

    /// The stop's error, once the run is stopped.
    pub fn error(&self) -> Option<WorkflowError> {
        self.state
            .borrow()
            .as_ref()
            .map(|cancellation| cancellation.cause.to_error())
    }

    /// `Err` with the stop's error once the run is stopped.
    pub fn check(&self) -> Result<(), WorkflowError> {
        self.error().map_or(Ok(()), Err)
    }

    /// Resolves once the run is stopped.
    pub async fn stopped(&self) {
        let mut receiver = self.state.subscribe();
        let _ = receiver.wait_for(Option::is_some).await;
    }

    /// Registers a started agent's command channel; an agent started after
    /// the stop is sent `CancelTask` at once.
    pub fn register(&self, task_id: uuid::Uuid, sender: &mpsc::Sender<UserCommand>) {
        {
            let mut agents = self.agents.lock();
            agents.retain(|_, sender| sender.strong_count() > 0);
            agents.insert(task_id, sender.downgrade());
        }
        if self.is_stopped() {
            let _ = sender.try_send(UserCommand::CancelTask);
        }
    }

    pub fn unregister(&self, task_id: uuid::Uuid) {
        self.agents.lock().remove(&task_id);
    }

    /// True once, for the first caller after the run stopped: the caller
    /// saves the checkpoint recording the stop.
    pub fn claim_stop_save(&self) -> bool {
        self.is_stopped() && !self.stop_saved.swap(true, Ordering::AcqRel)
    }
}

/// Cancels a workflow run from any task or thread.
///
/// Obtained from [`Workflow::cancel_handle`](crate::Workflow::cancel_handle);
/// clones cancel the same run. [`cancel`](Self::cancel) tells every running
/// agent of the run to stop (`CancelTask`) and makes run calls in progress,
/// and every later one, return [`WorkflowError::Cancelled`]. Each call in
/// progress returns once its agents have stopped, or have been aborted after
/// the run's grace period.
#[derive(Clone)]
pub struct WorkflowCancelHandle {
    control: Arc<RunControl>,
    context: Weak<SharedContext>,
}

impl WorkflowCancelHandle {
    pub(crate) fn new(context: &Arc<SharedContext>) -> Self {
        Self {
            control: context.run_control.clone(),
            context: Arc::downgrade(context),
        }
    }

    /// Cancels the run and fires [`HookPoint::OnWorkflowCancelled`] (its
    /// result is ignored). Returns `false`, changing nothing, when the run
    /// was already cancelled or aborted by a hook.
    pub fn cancel(&self) -> bool {
        if !self.control.stop(CancelCause::Requested) {
            return false;
        }
        if let Some(context) = self.context.upgrade() {
            crate::workflow::engine::fire_hooks_ignoring_result(
                &context,
                HookPoint::OnWorkflowCancelled,
                |_| {},
            );
        }
        true
    }

    /// Whether the run is stopped, by cancellation or a hook's abort.
    pub fn is_cancelled(&self) -> bool {
        self.control.is_stopped()
    }

    /// Why and when the run stopped, once it has.
    pub fn cancellation(&self) -> Option<Cancellation> {
        self.control.cancellation()
    }

    /// Resolves once the run is stopped, for example to stop work of the
    /// application's own running alongside it.
    pub async fn cancelled(&self) {
        self.control.stopped().await
    }
}

impl std::fmt::Debug for WorkflowCancelHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkflowCancelHandle")
            .field("cancellation", &self.control.cancellation())
            .finish()
    }
}

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod tests;
