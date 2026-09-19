//! Late-bound services that let explicitly authorized tools call into a live workflow.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::persistence::PersistenceManager;
use crate::workflow::engine::WorkflowEngine;
use crate::workflow::result::AgentResult;

pub const RUN_WORKFLOW_AGENT_TOOL_NAME: &str = "run_workflow_agent";

#[derive(Debug, Clone)]
pub struct WorkflowServicesConfig {
    pub allowed_callers: HashSet<String>,
    pub allowed_targets: HashSet<String>,
    pub max_active_calls: u32,
}

impl Default for WorkflowServicesConfig {
    fn default() -> Self {
        Self {
            allowed_callers: HashSet::new(),
            allowed_targets: HashSet::new(),
            max_active_calls: 4,
        }
    }
}

impl WorkflowServicesConfig {
    pub fn allow_caller(mut self, agent: impl Into<String>) -> Self {
        self.allowed_callers.insert(agent.into());
        self
    }

    /// Restrict workflow-agent delegation to explicitly named targets.
    /// An empty target set preserves the backwards-compatible allow-all policy.
    pub fn allow_target(mut self, agent: impl Into<String>) -> Self {
        self.allowed_targets.insert(agent.into());
        self
    }

    pub fn max_active_calls(mut self, max: u32) -> Self {
        self.max_active_calls = max.max(1);
        self
    }
}

#[derive(Clone)]
pub struct WorkflowServices {
    context: Weak<SharedContext>,
    client: Arc<arko_ai::LLMClient>,
    persistence: Option<Arc<PersistenceManager>>,
    config: WorkflowServicesConfig,
    active_calls: Arc<AtomicU32>,
}

impl WorkflowServices {
    pub(crate) fn new(
        context: Weak<SharedContext>,
        client: Arc<arko_ai::LLMClient>,
        persistence: Option<Arc<PersistenceManager>>,
        config: WorkflowServicesConfig,
    ) -> Self {
        Self {
            context,
            client,
            persistence,
            config,
            active_calls: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn is_allowed(&self, caller: &str) -> bool {
        self.config.allowed_callers.is_empty() || self.config.allowed_callers.contains(caller)
    }

    pub fn is_target_allowed(&self, target: &str) -> bool {
        self.config.allowed_targets.is_empty() || self.config.allowed_targets.contains(target)
    }

    fn allowed_target_names(&self) -> Vec<String> {
        let mut targets = self
            .config
            .allowed_targets
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        targets.sort();
        targets
    }

    pub async fn run_agent(
        &self,
        caller: &str,
        agent_name: &str,
        prompt: &str,
    ) -> Result<AgentResult, WorkflowError> {
        if !self.is_allowed(caller) {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{caller}' is not authorized to invoke workflow services"
            )));
        }
        if !self.is_target_allowed(agent_name) {
            let allowed = self.allowed_target_names();
            let suffix = if allowed.is_empty() {
                String::new()
            } else {
                format!("; allowed targets: {}", allowed.join(", "))
            };
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{caller}' is not authorized to invoke workflow target '{agent_name}'{suffix}"
            )));
        }
        let previous = self.active_calls.fetch_add(1, Ordering::AcqRel);
        if previous >= self.config.max_active_calls {
            self.active_calls.fetch_sub(1, Ordering::AcqRel);
            return Err(WorkflowError::Channel(format!(
                "workflow service active-call limit ({}) reached",
                self.config.max_active_calls
            )));
        }
        struct CallGuard(Arc<AtomicU32>);
        impl Drop for CallGuard {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _guard = CallGuard(self.active_calls.clone());
        let context = self.context.upgrade().ok_or_else(|| {
            WorkflowError::Channel("workflow services outlived their workflow".to_string())
        })?;
        WorkflowEngine::from_shared(self.client.clone(), self.persistence.clone())
            .run_agent(agent_name, prompt, &context)
            .await
    }
}

pub(crate) struct RunWorkflowAgentTool {
    services: WorkflowServices,
    caller: String,
}

impl std::fmt::Debug for RunWorkflowAgentTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunWorkflowAgentTool")
            .field("caller", &self.caller)
            .finish_non_exhaustive()
    }
}

impl RunWorkflowAgentTool {
    pub fn new(services: WorkflowServices, caller: String) -> Self {
        Self { services, caller }
    }
}

#[async_trait]
impl Tool for RunWorkflowAgentTool {
    fn definition(&self) -> ToolDefinition {
        let allowed_targets = self.services.allowed_target_names();
        let target_guidance = if allowed_targets.is_empty() {
            "Use only a configured agent name.".to_string()
        } else {
            format!("Allowed targets: {}.", allowed_targets.join(", "))
        };
        ToolDefinition {
            name: RUN_WORKFLOW_AGENT_TOOL_NAME.to_string(),
            description: "Run another configured workflow agent for a bounded subtask and return its verified completion summary. Use only when delegation is materially useful.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition { name: "agent".to_string(), description: target_guidance, param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None },
                ToolParameterDefinition { name: "prompt".to_string(), description: "Concrete bounded task for that agent.".to_string(), param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None },
            ],
            supports_batched_calls: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let agent = params.get("agent").and_then(Value::as_str).ok_or_else(|| {
            ToolError::MissingParameter {
                param_name: "agent".to_string(),
            }
        })?;
        let prompt = params
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "prompt".to_string(),
            })?;
        self.services
            .run_agent(&self.caller, agent, prompt)
            .await
            .map(|result| result.summary)
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_allowlist_is_opt_in_and_independent_from_callers() {
        let unrestricted = WorkflowServicesConfig::default();
        assert!(unrestricted.allowed_targets.is_empty());
        let unrestricted_services = WorkflowServices::new(
            Weak::new(),
            Arc::new(arko_ai::LLMClient::new()),
            None,
            unrestricted,
        );
        assert!(unrestricted_services.is_target_allowed("any-agent"));

        let restricted = WorkflowServicesConfig::default()
            .allow_caller("orchestrator")
            .allow_target("coder")
            .allow_target("researcher");
        assert!(restricted.allowed_callers.contains("orchestrator"));
        assert!(restricted.allowed_targets.contains("coder"));
        assert!(restricted.allowed_targets.contains("researcher"));
        assert!(!restricted.allowed_targets.contains("final_verifier"));
        let restricted_services = WorkflowServices::new(
            Weak::new(),
            Arc::new(arko_ai::LLMClient::new()),
            None,
            restricted,
        );
        assert!(restricted_services.is_target_allowed("coder"));
        assert!(!restricted_services.is_target_allowed("final_verifier"));
    }
}
