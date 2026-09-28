//! Late-bound services that let explicitly authorized tools call into a live workflow.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
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

/// Which agents may run which other agents with `run_workflow_agent`.
///
/// Callers and targets are named explicitly: with none named, no agent is
/// offered the tool, and a caller may run no agent. The opt-ins
/// [`allow_all_callers`](Self::allow_all_callers) and
/// [`allow_all_targets`](Self::allow_all_targets) admit every agent except
/// completion-verifier agents (and, as a target, the caller itself).
#[derive(Debug, Clone)]
pub struct WorkflowServicesConfig {
    /// Agents offered `run_workflow_agent`.
    pub allowed_callers: HashSet<String>,
    /// Agents a caller may run.
    pub allowed_targets: HashSet<String>,
    /// Opt-in: every agent is a caller, except verifier agents.
    pub all_callers: bool,
    /// Opt-in: a caller may run every agent, except itself and verifier
    /// agents.
    pub all_targets: bool,
    /// Calls running at once, workflow-wide (nested delegation included);
    /// a call over the limit fails at once. Default
    /// [`DEFAULT_MAX_ACTIVE_CALLS`].
    pub max_active_calls: NonZeroU32,
    /// `max_active_calls(0)` was asked for; `WorkflowBuilder::build`
    /// reports it.
    zero_max_active_calls: bool,
}

/// The default [`WorkflowServicesConfig::max_active_calls`]: 4.
pub const DEFAULT_MAX_ACTIVE_CALLS: NonZeroU32 = NonZeroU32::new(4).unwrap();

impl Default for WorkflowServicesConfig {
    fn default() -> Self {
        Self {
            allowed_callers: HashSet::new(),
            allowed_targets: HashSet::new(),
            all_callers: false,
            all_targets: false,
            max_active_calls: DEFAULT_MAX_ACTIVE_CALLS,
            zero_max_active_calls: false,
        }
    }
}

impl WorkflowServicesConfig {
    /// Offer `run_workflow_agent` to `agent`.
    pub fn allow_caller(mut self, agent: impl Into<String>) -> Self {
        self.allowed_callers.insert(agent.into());
        self
    }

    /// Let callers run `agent`.
    pub fn allow_target(mut self, agent: impl Into<String>) -> Self {
        self.allowed_targets.insert(agent.into());
        self
    }

    /// Opt-in: offer `run_workflow_agent` to every agent except
    /// completion-verifier agents.
    pub fn allow_all_callers(mut self) -> Self {
        self.all_callers = true;
        self
    }

    /// Opt-in: let a caller run every agent except itself and
    /// completion-verifier agents.
    pub fn allow_all_targets(mut self) -> Self {
        self.all_targets = true;
        self
    }

    /// How many calls may run at once, workflow-wide (default 4). A `max` of
    /// 0 makes `WorkflowBuilder::build` fail.
    pub fn max_active_calls(mut self, max: u32) -> Self {
        match NonZeroU32::new(max) {
            Some(max) => {
                self.max_active_calls = max;
                self.zero_max_active_calls = false;
            }
            None => self.zero_max_active_calls = true,
        }
        self
    }

    /// Every agent the configuration names must be registered, and the
    /// call limit must be at least 1.
    pub(crate) fn validate(&self, agents: &HashSet<String>) -> Result<(), WorkflowError> {
        if self.zero_max_active_calls {
            return Err(WorkflowError::BuilderInvalid(
                "max_active_calls needs at least 1; the default is 4".to_string(),
            ));
        }
        for (role, names) in [
            ("caller", &self.allowed_callers),
            ("target", &self.allowed_targets),
        ] {
            let mut names = names.iter().collect::<Vec<_>>();
            names.sort();
            if let Some(unknown) = names.into_iter().find(|name| !agents.contains(*name)) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "workflow services {role} '{unknown}' is not a registered agent"
                )));
            }
        }
        Ok(())
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

    /// Whether `agent` is a completion-verifier agent of the workflow.
    fn is_verifier(&self, agent: &str) -> bool {
        self.context.upgrade().is_some_and(|context| {
            context
                .completion_verifiers
                .iter()
                .any(|verifier| verifier.verifier_agent == agent)
        })
    }

    /// Whether `caller` is offered `run_workflow_agent`: named, or every
    /// agent but verifier agents under `allow_all_callers`.
    pub fn is_allowed(&self, caller: &str) -> bool {
        self.config.allowed_callers.contains(caller)
            || (self.config.all_callers && !self.is_verifier(caller))
    }

    /// Whether `caller` may run `target`: named, or every agent but the
    /// caller and verifier agents under `allow_all_targets`.
    pub fn is_target_allowed(&self, caller: &str, target: &str) -> bool {
        self.config.allowed_targets.contains(target)
            || (self.config.all_targets && target != caller && !self.is_verifier(target))
    }

    /// The agents `caller` may run, sorted.
    fn allowed_target_names(&self, caller: &str) -> Vec<String> {
        let Some(context) = self.context.upgrade() else {
            return Vec::new();
        };
        let mut targets = context
            .agent_configs
            .keys()
            .filter(|name| self.is_target_allowed(caller, name))
            .cloned()
            .collect::<Vec<_>>();
        targets.sort();
        targets
    }

    /// The `agent` parameter's description for `caller`: the agents it may
    /// run, with their roles. Configuration only, so it doesn't change
    /// during a task.
    fn target_guidance(&self, caller: &str) -> String {
        let Some(context) = self.context.upgrade() else {
            return "Use only a configured agent name.".to_string();
        };
        let targets = self
            .allowed_target_names(caller)
            .into_iter()
            .map(|name| {
                let role = context
                    .agent_configs
                    .get(&name)
                    .map_or("", |config| config.role_description.as_str());
                format!("`{name}` ({role})")
            })
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return "No agent is configured for you to run.".to_string();
        }
        format!("The agent to run, one of: {}.", targets.join("; "))
    }

    pub async fn run_agent(
        &self,
        caller: &str,
        agent_name: &str,
        prompt: &str,
    ) -> Result<AgentResult, WorkflowError> {
        self.run_agent_at(caller, agent_name, prompt, 0).await
    }

    /// Runs `agent_name` for `caller` at `depth`: inside the tool call of a
    /// run at `depth - 1`.
    pub(crate) async fn run_agent_at(
        &self,
        caller: &str,
        agent_name: &str,
        prompt: &str,
        depth: u32,
    ) -> Result<AgentResult, WorkflowError> {
        if !self.is_allowed(caller) {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{caller}' is not authorized to invoke workflow services"
            )));
        }
        if !self.is_target_allowed(caller, agent_name) {
            let allowed = self.allowed_target_names(caller);
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
        if previous >= self.config.max_active_calls.get() {
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
            .run_nested(agent_name, prompt, &context, depth)
            .await
    }
}

pub(crate) struct RunWorkflowAgentTool {
    services: WorkflowServices,
    caller: String,
    /// The depth of the caller's run; the agent it runs is one deeper.
    depth: u32,
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
    pub fn new(services: WorkflowServices, caller: String, depth: u32) -> Self {
        Self {
            services,
            caller,
            depth,
        }
    }
}

#[async_trait]
impl Tool for RunWorkflowAgentTool {
    fn definition(&self) -> ToolDefinition {
        let target_guidance = self.services.target_guidance(&self.caller);
        ToolDefinition {
            name: RUN_WORKFLOW_AGENT_TOOL_NAME.to_string(),
            description: "Run another configured workflow agent for a bounded subtask and return its verified completion summary. Use only when delegation is materially useful.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition { name: "agent".to_string(), description: target_guidance, param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None },
                ToolParameterDefinition { name: "prompt".to_string(), description: "Concrete bounded task for that agent.".to_string(), param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    /// The call runs another agent to completion, however long that takes.
    fn uses_host_timeout(&self) -> bool {
        false
    }

    /// When the calling agent is stopped, the workflow winds the agent this
    /// call runs down (within its grace period) and records it; the call
    /// then returns.
    fn interruptible(&self) -> bool {
        false
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
            .run_agent_at(&self.caller, agent, prompt, self.depth + 1)
            .await
            .map(|result| result.summary)
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(name: &str) -> arko_agents::AgentConfig {
        crate::test_support::agent_config(name)
    }

    fn build(services: WorkflowServicesConfig) -> crate::Workflow {
        crate::Workflow::builder()
            .name("services-test")
            .workspace(std::env::temp_dir())
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("orchestrator", agent("Orchestrator"))
            .agent("coder", agent("Coder"))
            .agent("reviewer", agent("Reviewer"))
            .completion_verifier(
                crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
                    .target("coder"),
            )
            .workflow_services(services)
            .build()
            .unwrap()
    }

    /// Callers and targets are named explicitly: none named means nobody.
    #[test]
    fn empty_caller_and_target_sets_mean_nobody() {
        let workflow = build(WorkflowServicesConfig::default());
        let services = workflow.context().workflow_services.get().unwrap();
        for agent in ["orchestrator", "coder", "reviewer"] {
            assert!(!services.is_allowed(agent));
            assert!(!services.is_target_allowed("orchestrator", agent));
        }

        let workflow = build(
            WorkflowServicesConfig::default()
                .allow_caller("orchestrator")
                .allow_target("coder"),
        );
        let services = workflow.context().workflow_services.get().unwrap();
        assert!(services.is_allowed("orchestrator"));
        assert!(!services.is_allowed("coder"));
        assert!(services.is_target_allowed("orchestrator", "coder"));
        assert!(!services.is_target_allowed("orchestrator", "reviewer"));
        assert_eq!(services.allowed_target_names("orchestrator"), ["coder"]);
    }

    /// The opt-ins admit every agent but verifier agents, and as a target
    /// not the caller itself.
    #[test]
    fn allow_all_admits_every_agent_but_self_and_verifiers() {
        let workflow = build(
            WorkflowServicesConfig::default()
                .allow_all_callers()
                .allow_all_targets(),
        );
        let services = workflow.context().workflow_services.get().unwrap();
        assert!(services.is_allowed("orchestrator"));
        assert!(services.is_allowed("coder"));
        assert!(!services.is_allowed("reviewer"));
        assert_eq!(
            services.allowed_target_names("orchestrator"),
            ["coder"],
            "not itself, not the verifier"
        );
        assert_eq!(services.allowed_target_names("coder"), ["orchestrator"]);
        assert!(services.target_guidance("orchestrator").contains("`coder`"));
    }

    /// `max_active_calls(0)` fails the build (it once built a workflow
    /// whose every call was refused, "active-call limit (0) reached"); the
    /// field can't hold 0, and the default is 4.
    #[test]
    fn zero_max_active_calls_fails_the_build() {
        assert_eq!(WorkflowServicesConfig::default().max_active_calls.get(), 4);
        let error = crate::Workflow::builder()
            .name("services-test")
            .workspace(std::env::temp_dir())
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("orchestrator", agent("Orchestrator"))
            .agent("coder", agent("Coder"))
            .workflow_services(
                WorkflowServicesConfig::default()
                    .allow_caller("orchestrator")
                    .allow_target("coder")
                    .max_active_calls(0),
            )
            .build()
            .err()
            .unwrap();
        assert!(
            matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("max_active_calls needs at least 1")),
            "{error}"
        );
        // A later valid value replaces the rejected one.
        let config = WorkflowServicesConfig::default()
            .max_active_calls(0)
            .max_active_calls(2);
        assert_eq!(config.max_active_calls.get(), 2);
        assert!(config.validate(&HashSet::new()).is_ok());
    }

    /// Names must be registered agents.
    #[test]
    fn unknown_callers_and_targets_are_rejected_at_build() {
        for config in [
            WorkflowServicesConfig::default().allow_caller("orchestator"),
            WorkflowServicesConfig::default().allow_target("cdoer"),
        ] {
            let error = crate::Workflow::builder()
                .name("services-test")
                .workspace(std::env::temp_dir())
                .llm_client(Arc::new(arko_ai::LLMClient::new()))
                .agent("orchestrator", agent("Orchestrator"))
                .agent("coder", agent("Coder"))
                .workflow_services(config)
                .build()
                .err()
                .unwrap();
            assert!(
                matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("is not a registered agent")),
                "{error}"
            );
        }
    }
}
