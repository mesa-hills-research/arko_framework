//! Context provided to hook callbacks.

use std::collections::HashMap;

use serde_json::Value;

/// Context provided to hooks. Fields are populated based on the hook point.
#[derive(Debug, Clone)]
pub struct HookContext {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub hook_point: super::HookPoint,
    pub workflow_run_id: uuid::Uuid,
    pub workflow_name: String,
    pub agent_run_id: Option<uuid::Uuid>,
    pub agent_name: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<uuid::Uuid>,
    pub tool_params: Option<HashMap<String, Value>>,
    pub tool_result: Option<String>,
    pub original_tool_result: Option<String>,
    pub effective_tool_result: Option<String>,
    pub agent_summary: Option<String>,
    pub error_message: Option<String>,
    pub budget_spent: f64,
    pub budget_limit: Option<f64>,
    pub metadata: HashMap<String, Value>,
}

impl HookContext {
    pub(crate) fn base(
        workflow_run_id: uuid::Uuid,
        workflow_name: &str,
        hook_point: super::HookPoint,
        budget_spent: f64,
        budget_limit: Option<f64>,
    ) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            hook_point,
            workflow_run_id,
            workflow_name: workflow_name.to_string(),
            agent_run_id: None,
            agent_name: None,
            tool_name: None,
            tool_call_id: None,
            tool_params: None,
            tool_result: None,
            original_tool_result: None,
            effective_tool_result: None,
            agent_summary: None,
            error_message: None,
            budget_spent,
            budget_limit,
            metadata: HashMap::new(),
        }
    }
}
