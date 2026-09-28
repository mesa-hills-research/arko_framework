//! Python bindings via PyO3 (feature-gated behind `python`).

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pyo3::create_exception!(
    arko_workflows,
    WorkflowCancelled,
    pyo3::exceptions::PyException,
    "The workflow run was cancelled."
);

#[cfg(feature = "python")]
pyo3::create_exception!(
    arko_workflows,
    UserInputRequired,
    pyo3::exceptions::PyException,
    "An agent asked a question and waits for an answer: args are (message, agent_run_id). Answer with PyWorkflow.continue_agent."
);

#[cfg(feature = "python")]
pyo3::create_exception!(
    arko_workflows,
    AgentPaused,
    pyo3::exceptions::PyException,
    "An agent paused for a tool approval or error-recovery decision: args are (message, agent_run_id). Decide with PyWorkflow.resume_agent."
);

#[cfg(feature = "python")]
pyo3::create_exception!(
    arko_workflows,
    VerifierFailed,
    pyo3::exceptions::PyException,
    "A completion verifier could not review the agent's completion, which was neither accepted nor rejected; the agent is left paused: args are (message, agent_run_id). Continue it with PyWorkflow.continue_agent on PyWorkflow.agent_result(agent_run_id)."
);

#[cfg(feature = "python")]
#[pymodule]
fn arko_workflows(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkflowBuilder>()?;
    m.add_class::<PyWorkflow>()?;
    m.add_class::<PyAgentResult>()?;
    m.add("WorkflowCancelled", m.py().get_type::<WorkflowCancelled>())?;
    m.add("UserInputRequired", m.py().get_type::<UserInputRequired>())?;
    m.add("AgentPaused", m.py().get_type::<AgentPaused>())?;
    m.add("VerifierFailed", m.py().get_type::<VerifierFailed>())?;
    Ok(())
}

/// A cancelled run raises `WorkflowCancelled`; an agent waiting for input
/// `UserInputRequired`, a paused agent `AgentPaused`, and an agent whose
/// completion a verifier could not review (left paused) `VerifierFailed`,
/// each with the agent run ID as the second argument; other errors raise
/// `RuntimeError`.
#[cfg(feature = "python")]
fn run_error(error: crate::WorkflowError) -> PyErr {
    match error {
        crate::WorkflowError::Cancelled => WorkflowCancelled::new_err(error.to_string()),
        crate::WorkflowError::UserInputRequired { task_id, .. } => {
            UserInputRequired::new_err((error.to_string(), task_id.to_string()))
        }
        crate::WorkflowError::AgentPaused { task_id, .. } => {
            AgentPaused::new_err((error.to_string(), task_id.to_string()))
        }
        crate::WorkflowError::VerifierFailed { task_id, .. } => {
            VerifierFailed::new_err((error.to_string(), task_id.to_string()))
        }
        error => pyo3::exceptions::PyRuntimeError::new_err(format!("Workflow error: {error}")),
    }
}

#[cfg(feature = "python")]
fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyWorkflowBuilder {
    name: Option<String>,
    workspace: Option<String>,
    agents: Vec<(String, String)>,
    budget: Option<f64>,
    persistence_dir: Option<String>,
    auto_checkpoint: bool,
    retain_checkpoints: Option<usize>,
    event_log: crate::persistence::EventLog,
    resume_from: Option<String>,
    memory_tiers: Vec<(String, Option<usize>)>,
    memory_renames: Vec<(String, String)>,
    dropped_memory_tiers: Vec<String>,
    approval_policy: crate::policy::ApprovalPolicy,
    error_recovery_policy: crate::policy::ErrorRecoveryPolicy,
    max_parallel_agents: Option<usize>,
    fail_fast: bool,
    cancel_grace_period: Option<f64>,
    /// `None`: the automatic limit.
    wait_for_message_timeout: Option<f64>,
    agent_wait_for_message_timeouts: Vec<(String, Option<f64>)>,
}

/// A `wait_for_message` limit from Python: `None` is automatic, otherwise
/// seconds, more than zero.
#[cfg(feature = "python")]
fn wait_timeout_seconds(seconds: Option<f64>) -> PyResult<Option<f64>> {
    match seconds {
        Some(seconds) if !seconds.is_finite() || seconds <= 0.0 => Err(
            pyo3::exceptions::PyValueError::new_err("the timeout must be finite and positive"),
        ),
        seconds => Ok(seconds),
    }
}

#[cfg(feature = "python")]
fn wait_timeout(seconds: Option<f64>) -> PyResult<crate::messaging::wait::WaitForMessageTimeout> {
    match seconds {
        None => Ok(crate::messaging::wait::WaitForMessageTimeout::Auto),
        Some(seconds) => std::time::Duration::try_from_secs_f64(seconds)
            .map(Into::into)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string())),
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWorkflowBuilder {
    #[new]
    fn new() -> Self {
        Self {
            name: None,
            workspace: None,
            agents: Vec::new(),
            budget: None,
            persistence_dir: None,
            auto_checkpoint: true,
            retain_checkpoints: None,
            event_log: crate::persistence::EventLog::Full,
            resume_from: None,
            memory_tiers: Vec::new(),
            memory_renames: Vec::new(),
            dropped_memory_tiers: Vec::new(),
            approval_policy: crate::policy::ApprovalPolicy::ApproveAll,
            error_recovery_policy: crate::policy::ErrorRecoveryPolicy::Automatic,
            max_parallel_agents: None,
            fail_fast: false,
            cancel_grace_period: None,
            wait_for_message_timeout: None,
            agent_wait_for_message_timeouts: Vec::new(),
        }
    }

    fn name(&mut self, name: String) -> PyResult<()> {
        self.name = Some(name);
        Ok(())
    }

    fn workspace(&mut self, path: String) -> PyResult<()> {
        self.workspace = Some(path);
        Ok(())
    }

    fn agent(&mut self, name: String, config_path: String) -> PyResult<()> {
        self.agents.push((name, config_path));
        Ok(())
    }

    fn budget(&mut self, limit: f64) -> PyResult<()> {
        self.budget = Some(limit);
        Ok(())
    }

    /// Persist the run in `checkpoint_dir`. As in Rust: a checkpoint after
    /// every finalized agent run (`auto_checkpoint=True`), every checkpoint
    /// and task version kept (`retain_checkpoints=None`; a count of at least
    /// 1 keeps the newest checkpoints and the task versions they refer to,
    /// and 0 raises `ValueError`), and every event logged
    /// (`event_log="full"`; `"lifecycle_only"` leaves out streamed
    /// fragments).
    #[pyo3(signature = (checkpoint_dir, auto_checkpoint=true, retain_checkpoints=None, event_log="full"))]
    fn persistence(
        &mut self,
        checkpoint_dir: String,
        auto_checkpoint: bool,
        retain_checkpoints: Option<usize>,
        event_log: &str,
    ) -> PyResult<()> {
        if retain_checkpoints == Some(0) {
            return Err(value_error(
                "retain_checkpoints needs at least 1; every checkpoint is kept by default",
            ));
        }
        self.event_log = match event_log {
            "full" => crate::persistence::EventLog::Full,
            "lifecycle_only" => crate::persistence::EventLog::LifecycleOnly,
            other => {
                return Err(value_error(format!(
                    "event_log must be \"full\" or \"lifecycle_only\", not {other:?}"
                )));
            }
        };
        self.persistence_dir = Some(checkpoint_dir);
        self.auto_checkpoint = auto_checkpoint;
        self.retain_checkpoints = retain_checkpoints;
        Ok(())
    }

    /// On resume, restore the checkpoint's memory tier `old` into the
    /// configured tier `new`.
    fn rename_memory_tier_on_resume(&mut self, old: String, new: String) -> PyResult<()> {
        self.memory_renames.push((old, new));
        Ok(())
    }

    /// On resume, discard the checkpoint's memory tier `name`.
    fn drop_memory_tier_on_resume(&mut self, name: String) -> PyResult<()> {
        self.dropped_memory_tiers.push(name);
        Ok(())
    }

    /// `"approve_all"` (the default: every tool call is approved) or
    /// `"respect_agent_config"` (a call the agent's configuration marks for
    /// approval pauses the agent: `AgentPaused`, decided with
    /// `resume_agent`).
    fn approval_policy(&mut self, policy: &str) -> PyResult<()> {
        self.approval_policy = match policy {
            "approve_all" => crate::policy::ApprovalPolicy::ApproveAll,
            "respect_agent_config" => crate::policy::ApprovalPolicy::RespectAgentConfig,
            other => {
                return Err(value_error(format!(
                    "the approval policy must be \"approve_all\" or \"respect_agent_config\", not {other:?}"
                )));
            }
        };
        Ok(())
    }

    /// `"automatic"` (the default: recover without an attempt limit from
    /// errors that may pass), `"limited"` with `max` automatic recoveries,
    /// or `"return_to_caller"` (every error-recovery pause raises
    /// `AgentPaused`).
    #[pyo3(signature = (policy, max=None))]
    fn error_recovery_policy(&mut self, policy: &str, max: Option<u32>) -> PyResult<()> {
        self.error_recovery_policy = match (policy, max) {
            ("automatic", None) => crate::policy::ErrorRecoveryPolicy::Automatic,
            ("limited", Some(max)) => crate::policy::ErrorRecoveryPolicy::Limited { max },
            ("return_to_caller", None) => crate::policy::ErrorRecoveryPolicy::ReturnToCaller,
            ("limited", None) => return Err(value_error("\"limited\" needs max")),
            (other, _) => {
                return Err(value_error(format!(
                    "the error recovery policy must be \"automatic\", \"limited\" (with max), or \"return_to_caller\", not {other:?}"
                )));
            }
        };
        Ok(())
    }

    /// Run at most `max` agents of a `run_agents_parallel` call at once
    /// (default: no cap).
    fn max_parallel_agents(&mut self, max: usize) -> PyResult<()> {
        self.max_parallel_agents = Some(max);
        Ok(())
    }

    /// In `run_agents_parallel`, stop the other agents at the first failure
    /// (default off).
    fn fail_fast(&mut self, enabled: bool) -> PyResult<()> {
        self.fail_fast = enabled;
        Ok(())
    }

    fn resume_from(&mut self, checkpoint: String) -> PyResult<()> {
        self.resume_from = Some(checkpoint);
        Ok(())
    }

    #[pyo3(signature = (name, max_bytes=None))]
    fn global_memory_tier(&mut self, name: String, max_bytes: Option<usize>) -> PyResult<()> {
        self.memory_tiers.push((name, max_bytes));
        Ok(())
    }

    /// Seconds a cancelled run waits for each agent to stop before
    /// aborting it (default 30).
    fn cancel_grace_period(&mut self, seconds: f64) -> PyResult<()> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the grace period must be finite and non-negative",
            ));
        }
        self.cancel_grace_period = Some(seconds);
        Ok(())
    }

    /// Seconds `wait_for_message` waits at most; `None` (the default)
    /// selects the automatic limit, just inside each agent's prompt-cache
    /// lifetime.
    #[pyo3(signature = (seconds=None))]
    fn wait_for_message_timeout(&mut self, seconds: Option<f64>) -> PyResult<()> {
        self.wait_for_message_timeout = wait_timeout_seconds(seconds)?;
        Ok(())
    }

    /// The same for one agent, in place of the workflow's setting.
    #[pyo3(signature = (agent, seconds=None))]
    fn agent_wait_for_message_timeout(
        &mut self,
        agent: String,
        seconds: Option<f64>,
    ) -> PyResult<()> {
        let seconds = wait_timeout_seconds(seconds)?;
        self.agent_wait_for_message_timeouts.push((agent, seconds));
        Ok(())
    }

    fn build(&self) -> PyResult<PyWorkflow> {
        let mut builder = crate::Workflow::builder();
        if let Some(seconds) = self.cancel_grace_period {
            let grace_period = std::time::Duration::try_from_secs_f64(seconds)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
            builder = builder.cancel_grace_period(grace_period);
        }
        builder = builder.wait_for_message_timeout(wait_timeout(self.wait_for_message_timeout)?);
        for (agent, seconds) in &self.agent_wait_for_message_timeouts {
            builder =
                builder.agent_wait_for_message_timeout(agent.clone(), wait_timeout(*seconds)?);
        }

        if let Some(ref name) = self.name {
            builder = builder.name(name.clone());
        }
        if let Some(ref ws) = self.workspace {
            builder = builder.workspace(ws.clone());
        }
        if let Some(b) = self.budget {
            builder = builder.budget(b);
        }

        if let Some(ref directory) = self.persistence_dir {
            let mut config = crate::persistence::PersistenceConfig::new(directory)
                .auto_checkpoint(self.auto_checkpoint)
                .event_log(self.event_log);
            if let Some(count) = self.retain_checkpoints {
                config = config.keep_last_checkpoints(count);
            }
            builder = builder.persistence(config);
        }
        for (old, new) in &self.memory_renames {
            builder = builder.rename_memory_tier_on_resume(old.clone(), new.clone());
        }
        for name in &self.dropped_memory_tiers {
            builder = builder.drop_memory_tier_on_resume(name.clone());
        }
        builder = builder
            .approval_policy(self.approval_policy.clone())
            .error_recovery_policy(self.error_recovery_policy.clone())
            .fail_fast(self.fail_fast);
        if let Some(max) = self.max_parallel_agents {
            builder = builder.max_parallel_agents(max);
        }
        if let Some(ref checkpoint) = self.resume_from {
            builder = builder.resume_from(checkpoint);
        }
        if !self.memory_tiers.is_empty() {
            let mut memory = crate::memory::MemoryConfig::new();
            for (name, max_bytes) in &self.memory_tiers {
                memory = match max_bytes {
                    Some(limit) => memory.global_tier_with_limit(name, *limit),
                    None => memory.global_tier(name),
                };
            }
            builder = builder.memory(memory);
        }

        let llm_client = std::sync::Arc::new(arko_ai::LLMClient::from_env().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to configure model providers from the environment: {}",
                e
            ))
        })?);
        builder = builder.llm_client(llm_client);

        for (name, config_path) in &self.agents {
            let config =
                arko_agents::AgentConfig::load_from_path(std::path::Path::new(config_path))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to load agent config: {}",
                            e
                        ))
                    })?;
            builder = builder.agent(name.clone(), config);
        }

        let workflow = builder.build().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to build workflow: {}", e))
        })?;

        Ok(PyWorkflow {
            inner: std::sync::Arc::new(workflow),
        })
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyWorkflow {
    inner: std::sync::Arc<crate::Workflow>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWorkflow {
    fn run_agent<'py>(
        &self,
        py: Python<'py>,
        agent_name: String,
        prompt: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = workflow
                .run_agent(&agent_name, &prompt)
                .await
                .map_err(run_error)?;
            Ok(PyAgentResult::from(result))
        })
    }

    fn run_agents_parallel<'py>(
        &self,
        py: Python<'py>,
        agents: Vec<(String, String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let borrowed = agents
                .iter()
                .map(|(name, prompt)| (name.as_str(), prompt.clone()))
                .collect();
            workflow
                .run_agents_parallel(borrowed)
                .await
                .map(|results| {
                    results
                        .into_iter()
                        .map(PyAgentResult::from)
                        .collect::<Vec<_>>()
                })
                .map_err(run_error)
        })
    }

    /// Continue a completed agent run with a follow-up `prompt`, answer an
    /// agent waiting for input (`UserInputRequired`) with it, or continue an
    /// agent whose completion a verifier could not review (`VerifierFailed`)
    /// with it.
    fn continue_agent<'py>(
        &self,
        py: Python<'py>,
        previous: PyAgentResult,
        prompt: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            workflow
                .continue_agent(&previous.inner, &prompt)
                .await
                .map(PyAgentResult::from)
                .map_err(run_error)
        })
    }

    /// Continue a paused agent (`AgentPaused`): `"approve"` or `"reject"`
    /// every call awaiting approval, or `"continue"` (the failed request is
    /// sent again; `note`, when given, is added for the model first) or
    /// `"fail"` (`note` is the reason) after an error-recovery pause.
    #[pyo3(signature = (previous, decision, note=None))]
    fn resume_agent<'py>(
        &self,
        py: Python<'py>,
        previous: PyAgentResult,
        decision: String,
        note: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use arko_agents::runtime::command::ToolApprovalDecision;
        let calls = match crate::policy::AgentPause::of(&previous.inner.task.current_state) {
            Some(crate::policy::AgentPause::ToolApproval { calls }) => calls,
            _ => Vec::new(),
        };
        let resume = match decision.as_str() {
            "approve" | "reject" => crate::policy::AgentResume::ToolApprovals(
                calls
                    .iter()
                    .map(|call| {
                        let decision = if decision == "approve" {
                            ToolApprovalDecision::Approve {
                                custom_tool_output: None,
                                message_to_agent: note.clone(),
                            }
                        } else {
                            ToolApprovalDecision::Reject {
                                reason: note.clone(),
                            }
                        };
                        (call.tool_call_id, decision)
                    })
                    .collect(),
            ),
            "continue" => crate::policy::AgentResume::ContinueAfterError { note },
            "fail" => crate::policy::AgentResume::FailAfterError { reason: note },
            other => {
                return Err(value_error(format!(
                    "the decision must be \"approve\", \"reject\", \"continue\", or \"fail\", not {other:?}"
                )));
            }
        };
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            workflow
                .resume_agent(&previous.inner, resume)
                .await
                .map(PyAgentResult::from)
                .map_err(run_error)
        })
    }

    /// The recorded result of agent run `agent_run_id` (as `UserInputRequired`,
    /// `AgentPaused`, and `VerifierFailed` give it), or None.
    fn agent_result(&self, agent_run_id: String) -> PyResult<Option<PyAgentResult>> {
        let id =
            uuid::Uuid::parse_str(&agent_run_id).map_err(|error| value_error(error.to_string()))?;
        Ok(self.inner.agent_result(id).map(PyAgentResult::from))
    }

    /// Finish the run: fire `PostWorkflow`, save a checkpoint, and return
    /// the recorded agent results.
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            workflow
                .finish()
                .await
                .map(|result| {
                    result
                        .agent_results
                        .into_iter()
                        .map(PyAgentResult::from)
                        .collect::<Vec<_>>()
                })
                .map_err(run_error)
        })
    }

    /// Cancels the run: running agents stop, and pending and later runs
    /// raise `WorkflowCancelled`. Returns False when it was already stopped.
    fn cancel(&self) -> bool {
        self.inner.cancel()
    }

    /// Whether the run was cancelled or aborted by a hook.
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    #[pyo3(signature = (amount, description, metadata=None))]
    fn add_manual_cost(
        &self,
        amount: f64,
        description: String,
        metadata: Option<String>,
    ) -> PyResult<f64> {
        self.inner
            .add_manual_cost(amount, &description, metadata)
            .map(|result| result.new_total)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn cost_ledger_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.cost_ledger())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            workflow
                .checkpoint()
                .await
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (agent_name, tier, max_chars=None))]
    fn read_memory(
        &self,
        agent_name: String,
        tier: String,
        max_chars: Option<usize>,
    ) -> PyResult<String> {
        self.inner
            .context()
            .memory
            .read_window(&agent_name, &tier, max_chars)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn write_memory(&self, agent_name: String, tier: String, content: String) -> PyResult<()> {
        self.inner
            .context()
            .memory
            .write(&agent_name, &tier, &content)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyAgentResult {
    #[pyo3(get)]
    pub agent_name: String,
    #[pyo3(get)]
    pub summary: String,
    #[pyo3(get)]
    pub total_cost: f64,
    /// `"Completed"`, `"Failed"`, `"Interrupted"`, `"Cancelled"`,
    /// `"WaitingForInput"`, or `"Paused"`.
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub agent_run_id: String,
    /// The failure or cancellation reason, the questions, or the pause.
    #[pyo3(get)]
    pub error: Option<String>,
    /// The full result, to continue or resume the run.
    inner: crate::AgentResult,
}

#[cfg(feature = "python")]
impl From<crate::AgentResult> for PyAgentResult {
    fn from(result: crate::AgentResult) -> Self {
        Self {
            agent_name: result.agent_name.clone(),
            summary: result.summary.clone(),
            total_cost: result.total_cost(),
            status: format!("{:?}", result.status),
            agent_run_id: result.agent_run_id.to_string(),
            error: result.error.clone(),
            inner: result,
        }
    }
}
