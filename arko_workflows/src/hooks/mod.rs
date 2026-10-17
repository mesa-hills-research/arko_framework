//! Lifecycle hook system for workflow events.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod context;

/// Lifecycle points where hooks can be registered.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HookPoint {
    PreWorkflow,
    PostWorkflow,
    PreAgentSpawn,
    PostAgentCompletion,
    PostAgentInterruption,
    /// An agent run ended failed. A failure at an error-recovery pause the
    /// error recovery policy (or the caller) rejected has the pause's cause
    /// in the metadata's `cause` (`non_retryable`, `refusal`, …).
    PostAgentFailure,
    /// An agent run ended cancelled: stopped with the workflow run, or by the
    /// workflow over its budget.
    PostAgentCancellation,
    OnUserInputRequired,
    /// An agent paused for a decision the workflow's approval or error
    /// recovery policy leaves to the caller (`AgentRunStatus::Paused`):
    /// fires once per pause. `metadata["pause"]` is `"tool_approval"` or
    /// `"error_recovery"`. An error-recovery pause also sets
    /// `metadata["cause"]` and `error_message`, a tool-approval pause
    /// `metadata["tools"]`.
    OnAgentPaused,
    PreToolExecution,
    PostToolExecution,
    PostToolResultInterception,
    OnWorkflowPause,
    OnWorkflowResume,
    /// A run call failed. Fires at most once per call, and not for a pause
    /// (an agent waiting for input or paused), a cancellation, or a stop.
    OnWorkflowFailed,
    /// A [`WorkflowCancelHandle`](crate::workflow::cancel::WorkflowCancelHandle)
    /// cancelled or [stopped](crate::Workflow::stop) the run. Fires once,
    /// when the run is stopped (after a graceful stop's wrap-up), from the
    /// task that stopped it. For a graceful stop its metadata has
    /// `stop_kind` and `reason`.
    OnWorkflowCancelled,
    OnBudgetExceeded,
    OnBudgetWarning,
    OnAgentMessage,
    /// A tracked write made another actor's read of the same context stale.
    #[serde(alias = "OnMesiConflict")]
    OnFreshnessConflict,
}

/// The result of a hook invocation, directing the engine how to proceed.
#[derive(Debug, Clone)]
pub enum HookResult {
    /// Continue normal execution.
    Continue,
    /// Abort the workflow run with this reason: every running agent is
    /// cancelled, and every run call in progress or made later returns
    /// [`WorkflowError::AbortedByHook`](crate::WorkflowError::AbortedByHook).
    Abort(String),
    /// Skip the current operation. At `PreToolExecution` the call is
    /// rejected with the message `Skipped by PreToolExecution hook`.
    Skip,
    /// Skip the current operation, with the reason. At `PreToolExecution`
    /// the call is rejected and the agent is told this message, so it knows
    /// why and what to do instead. At other points it acts as `Skip`.
    Reject(String),
}

/// A synchronous hook callback.
pub type HookCallback = Arc<dyn Fn(&context::HookContext) -> HookResult + Send + Sync + 'static>;

/// Registry of lifecycle hooks.
pub struct HookRegistry {
    hooks: HashMap<HookPoint, Vec<HookCallback>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
        }
    }

    /// Register a hook at the given lifecycle point.
    pub fn register(&mut self, point: HookPoint, callback: HookCallback) {
        self.hooks.entry(point).or_default().push(callback);
    }

    /// Get hooks registered at the given point.
    pub fn get(&self, point: &HookPoint) -> Option<&Vec<HookCallback>> {
        self.hooks.get(point)
    }

    /// Returns true if any hooks are registered at the given point.
    pub fn has_hooks(&self, point: &HookPoint) -> bool {
        self.hooks.get(point).is_some_and(|v| !v.is_empty())
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_former_conflict_hook_name_loads_as_the_freshness_hook() {
        let point: HookPoint = serde_json::from_str("\"OnMesiConflict\"").unwrap();
        assert_eq!(point, HookPoint::OnFreshnessConflict);
        assert_eq!(
            serde_json::to_string(&HookPoint::OnFreshnessConflict).unwrap(),
            "\"OnFreshnessConflict\""
        );
    }
}
