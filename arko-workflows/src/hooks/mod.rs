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
    PostAgentFailure,
    OnUserInputRequired,
    PreToolExecution,
    PostToolExecution,
    PostToolResultInterception,
    OnWorkflowPause,
    OnWorkflowResume,
    OnWorkflowFailed,
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
    /// Abort the workflow with this reason.
    Abort(String),
    /// Skip the current operation.
    Skip,
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
