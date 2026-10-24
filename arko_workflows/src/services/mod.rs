//! Late-bound services that let explicitly authorized tools call into a live workflow.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::{
    TaskContext, TaskParent, Tool, ToolDefinition, ToolError, ToolParameterDefinition,
    ToolParameterType, ToolRecovery, ToolTypeDetails,
};

use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::persistence::PersistenceManager;
use crate::policy::{AgentPause, AgentResume};
use crate::workflow::engine::{Recovery, WorkflowEngine, is_orphaned, recorded_result};
use crate::workflow::result::{AgentResult, AgentRunStatus};
use crate::workflow::stop::StopRecord;

pub const RUN_WORKFLOW_AGENT_TOOL_NAME: &str = "run_workflow_agent";

/// `run_workflow_agent`'s optional parameter naming an agent run to resume
/// instead of starting a new one.
pub const RESUME_RUN_PARAMETER: &str = "resume_run";

/// Which agents may run which other agents with `run_workflow_agent`.
///
/// Callers and targets are named explicitly. With none named, no agent is
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
    /// When enabled, every agent is a caller, except verifier agents.
    pub all_callers: bool,
    /// When enabled, a caller may run every agent, except itself and verifier
    /// agents.
    pub all_targets: bool,
    /// Calls running at once, workflow-wide (nested delegation included).
    /// A call over the limit fails at once, or waits for a place with
    /// [`queue_active_calls`](Self::queue_active_calls). Default
    /// [`DEFAULT_MAX_ACTIVE_CALLS`].
    pub max_active_calls: NonZeroU32,
    /// When enabled, no limit on calls running at once
    /// ([`max_active_calls`](Self::max_active_calls) is ignored).
    pub unlimited_active_calls: bool,
    /// When enabled, a call over [`max_active_calls`](Self::max_active_calls)
    /// waits until one ends, instead of failing.
    pub queue_active_calls: bool,
    /// `max_active_calls(0)` was asked for. `WorkflowBuilder::build`
    /// reports it.
    zero_max_active_calls: bool,
}

/// The default [`WorkflowServicesConfig::max_active_calls`] of 4.
pub const DEFAULT_MAX_ACTIVE_CALLS: NonZeroU32 = NonZeroU32::new(4).unwrap();

impl Default for WorkflowServicesConfig {
    fn default() -> Self {
        Self {
            allowed_callers: HashSet::new(),
            allowed_targets: HashSet::new(),
            all_callers: false,
            all_targets: false,
            max_active_calls: DEFAULT_MAX_ACTIVE_CALLS,
            unlimited_active_calls: false,
            queue_active_calls: false,
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

    /// Opt in to offering `run_workflow_agent` to every agent except
    /// completion-verifier agents.
    pub fn allow_all_callers(mut self) -> Self {
        self.all_callers = true;
        self
    }

    /// Opt in to letting a caller run every agent except itself and
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

    /// Opt in to unlimited calls running at once.
    pub fn unlimited_active_calls(mut self) -> Self {
        self.unlimited_active_calls = true;
        self
    }

    /// When enabled, a call over [`max_active_calls`](Self::max_active_calls)
    /// waits until one ends, instead of failing.
    pub fn queue_active_calls(mut self) -> Self {
        self.queue_active_calls = true;
        self
    }

    /// Every agent the configuration names must be registered, and the
    /// call limit must be at least 1.
    pub(crate) fn validate(&self, agents: &HashSet<String>) -> Result<(), WorkflowError> {
        if self.zero_max_active_calls {
            return Err(WorkflowError::BuilderInvalid(
                "max_active_calls needs at least 1. The default is 4".to_string(),
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
    /// Notified when a call ends, for the calls waiting for a place.
    call_ended: Arc<tokio::sync::Notify>,
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
            call_ended: Arc::new(tokio::sync::Notify::new()),
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

    /// Whether `caller` is offered `run_workflow_agent`, when named or every
    /// agent but verifier agents under `allow_all_callers`.
    pub fn is_allowed(&self, caller: &str) -> bool {
        self.config.allowed_callers.contains(caller)
            || (self.config.all_callers && !self.is_verifier(caller))
    }

    /// Whether `caller` may run `target`, when named or every agent but the
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
    pub(crate) fn target_guidance(&self, caller: &str) -> String {
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
        self.run_agent_at(caller, agent_name, prompt, 0, None).await
    }

    /// Runs `agent_name` for `caller` at `depth`: inside tool call `parent`
    /// of a run at `depth - 1`.
    pub(crate) async fn run_agent_at(
        &self,
        caller: &str,
        agent_name: &str,
        prompt: &str,
        depth: u32,
        parent: Option<TaskParent>,
    ) -> Result<AgentResult, WorkflowError> {
        self.authorize(caller, agent_name)?;
        let _guard = self.enter_call(true).await?;
        let context = self.context()?;
        WorkflowEngine::from_shared(self.client.clone(), self.persistence.clone())
            .run_nested(agent_name, prompt, &context, depth, parent)
            .await
    }

    /// Counts a call running until the guard is dropped, as a test's
    /// stand-in for one.
    #[cfg(test)]
    pub(crate) fn hold_call(&self) -> CallGuard {
        self.guard().counted()
    }

    fn guard(&self) -> CallGuard {
        CallGuard {
            count: self.active_calls.clone(),
            ended: self.call_ended.clone(),
        }
    }

    /// Counts a call until the guard is dropped. With `refuse`, fails when
    /// [`WorkflowServicesConfig::max_active_calls`] are running (or waits
    /// for a place, with [`WorkflowServicesConfig::queue_active_calls`]).
    async fn enter_call(&self, refuse: bool) -> Result<CallGuard, WorkflowError> {
        if !refuse || self.config.unlimited_active_calls {
            self.active_calls.fetch_add(1, Ordering::AcqRel);
            return Ok(self.guard());
        }
        let max = self.config.max_active_calls.get();
        loop {
            // Listening before the check, so an end between the two counts.
            let ended = self.call_ended.notified();
            tokio::pin!(ended);
            ended.as_mut().enable();
            let running = self.active_calls.load(Ordering::Acquire);
            if running < max {
                if self
                    .active_calls
                    .compare_exchange(running, running + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(self.guard());
                }
                continue;
            }
            if !self.config.queue_active_calls {
                return Err(WorkflowError::Channel(format!(
                    "workflow service active-call limit ({max}) reached"
                )));
            }
            ended.await;
        }
    }

    /// Recovers call `parent` of `caller`, which was running `agent_name`
    /// with `prompt` (resuming run `resume_run`, if set) at `depth` when the
    /// workflow stopped: the agent run the call ran ends as the call would
    /// have. A run that completed returns its result without running again
    /// (the caller's task wasn't saved with it). An interrupted one is
    /// recovered and waited for (as is another recovery of it already
    /// running). A run that ended otherwise returns its error. When the call
    /// ran no run that was saved with it (it can't have run a tool: runs are
    /// saved before their first), the call runs now: `agent_name` starts,
    /// or run `resume_run` is resumed. The call was admitted before the
    /// stop, so the active-call limit counts it without refusing it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn recover_call(
        &self,
        caller: &str,
        agent_name: &str,
        prompt: &str,
        resume_run: Option<uuid::Uuid>,
        depth: u32,
        parent: TaskParent,
    ) -> Result<AgentResult, WorkflowError> {
        self.authorize(caller, agent_name)?;
        let _guard = self.enter_call(false).await?;
        let context = self.context()?;
        let engine = WorkflowEngine::from_shared(self.client.clone(), self.persistence.clone());
        let started = context
            .agent_results
            .read()
            .iter()
            .rev()
            .find(|result| result.info().parent == Some(parent))
            .cloned();
        match started {
            // Retired by the host in its wrap-up when the workflow stopped:
            // it stays stopped, and the call fails naming it.
            Some(run)
                if run.status == AgentRunStatus::Interrupted
                    && run.stopped_by.as_ref().is_some_and(StopRecord::is_retired) =>
            {
                Err(WorkflowError::AgentStopped {
                    agent: run.agent_name.clone(),
                    task_id: run.agent_run_id,
                    reason: run
                        .stopped_by
                        .as_ref()
                        .map(StopRecord::run_reason)
                        .unwrap_or_default(),
                })
            }
            Some(run) if run.status == AgentRunStatus::Interrupted => {
                log::info!(
                    "Recovering agent run {} of agent '{}', which agent '{caller}' was running when the workflow stopped",
                    run.agent_run_id,
                    run.agent_name
                );
                let recovery = Recovery {
                    depth,
                    wait: true,
                    ..Recovery::default()
                };
                engine.recover_nested(&run, &context, recovery).await
            }
            Some(run) => crate::workflow::engine::run_outcome(run),
            None => match resume_run {
                Some(run_id) => {
                    self.resume_admitted(
                        parent.task_id,
                        agent_name,
                        run_id,
                        prompt,
                        depth,
                        Some(parent),
                    )
                    .await
                }
                None => {
                    engine
                        .run_nested(agent_name, prompt, &context, depth, Some(parent))
                        .await
                }
            },
        }
    }

    /// Resumes agent run `run_id` of `agent_name` for `caller`, whose run
    /// is `caller_run`, at `depth` inside tool call `parent`, with `message`
    /// (see [`resume_admitted`](Self::resume_admitted)). The call counts
    /// against the active-call limit, as a new run does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn resume_run(
        &self,
        caller: &str,
        caller_run: uuid::Uuid,
        agent_name: &str,
        run_id: uuid::Uuid,
        message: &str,
        depth: u32,
        parent: Option<TaskParent>,
    ) -> Result<AgentResult, WorkflowError> {
        self.authorize(caller, agent_name)?;
        let _guard = self.enter_call(true).await?;
        self.resume_admitted(caller_run, agent_name, run_id, message, depth, parent)
            .await
    }

    /// Resumes agent run `run_id` of `agent_name`, which did not finish, for
    /// the caller's run `caller_run` (authorized to run `agent_name`): it continues its own task, with
    /// `message` as a user message, instead of a new run starting. The run
    /// must be one `caller_run` started (its parent link names it), or an
    /// orphaned run ([`is_orphaned`]: the run that started it can't resume
    /// it any more). An interrupted run is recovered (the calls it was
    /// running are settled first). A completed run, or one waiting for an
    /// answer, continues. A run paused after an error continues with the
    /// message as the note. Its parent link then points at `parent`, so
    /// that a later stop finds it in this call.
    #[allow(clippy::too_many_arguments)]
    async fn resume_admitted(
        &self,
        caller_run: uuid::Uuid,
        agent_name: &str,
        run_id: uuid::Uuid,
        message: &str,
        depth: u32,
        parent: Option<TaskParent>,
    ) -> Result<AgentResult, WorkflowError> {
        let refuse = |reason: String| {
            Err(WorkflowError::BuilderInvalid(format!(
                "agent run {run_id} can't be resumed: {reason}"
            )))
        };
        let context = self.context()?;
        let Some(run) = recorded_result(&context, run_id) else {
            return refuse("the workflow has no such agent run".to_string());
        };
        if run.agent_name != agent_name {
            return refuse(format!(
                "it is a run of agent '{}', not '{agent_name}'",
                run.agent_name
            ));
        }
        match run.info().parent {
            None => {
                return refuse(
                    "run_workflow_agent didn't start it; the workflow's host continues it"
                        .to_string(),
                );
            }
            Some(link) if link.task_id == caller_run || is_orphaned(&context, &run) => {}
            Some(link) => {
                return refuse(format!(
                    "agent run {} started it and can still resume it",
                    link.task_id
                ));
            }
        }
        if context.registry.is_running(run_id) {
            return refuse("it is running".to_string());
        }
        if run.stopped_by.as_ref().is_some_and(StopRecord::is_retired) {
            return refuse("the workflow's host retired it".to_string());
        }
        let engine = WorkflowEngine::from_shared(self.client.clone(), self.persistence.clone());
        match run.status {
            AgentRunStatus::Interrupted => {
                let recovery = Recovery {
                    depth,
                    parent: Some(parent),
                    message: Some(message),
                    ..Recovery::default()
                };
                engine.recover_nested(&run, &context, recovery).await
            }
            AgentRunStatus::Cancelled if crate::workflow::engine::is_superseded(&run) => {
                refuse("a run that replaced it after a restart superseded it".to_string())
            }
            // Cancelled (by the budget, say): reopened where it stopped.
            AgentRunStatus::Cancelled => {
                let message = crate::workflow::engine::reopened_message(message);
                let recovery = Recovery {
                    depth,
                    parent: Some(parent),
                    message: Some(&message),
                    reopen: true,
                    ..Recovery::default()
                };
                engine.recover_nested(&run, &context, recovery).await
            }
            AgentRunStatus::Completed | AgentRunStatus::WaitingForInput => {
                engine
                    .continue_nested(&run, message, &context, depth, parent)
                    .await
            }
            AgentRunStatus::Paused => match run
                .task()
                .ok()
                .and_then(|task| AgentPause::of(&task.current_state))
            {
                Some(AgentPause::ErrorRecovery { .. }) => {
                    let resume = AgentResume::ContinueAfterError {
                        note: Some(message.to_string()),
                    };
                    engine
                        .resume_nested(&run, resume, &context, depth, parent)
                        .await
                }
                _ => {
                    refuse("it waits for the workflow's host to decide its tool calls".to_string())
                }
            },
            AgentRunStatus::Failed => refuse(format!(
                "it ended {}{}; run the agent again instead",
                format!("{:?}", run.status).to_lowercase(),
                run.error
                    .as_deref()
                    .map_or_else(String::new, |error| format!(" ({error})"))
            )),
        }
    }

    /// What a `run_workflow_agent` call that failed tells `caller` about
    /// agent run `run`, which it ran and which did not finish (or what a
    /// call that succeeded adds to the summary of a run that completed
    /// after a stop's wrap-up notice): the run's id and agent, and how to
    /// resume it (or that the host recovers it, when the caller may not run
    /// its agent). `None` for a run that can't be resumed with a message,
    /// or is running.
    fn unfinished_note(&self, caller: &str, run: &AgentResult) -> Option<String> {
        let context = self.context.upgrade()?;
        if context.registry.is_running(run.agent_run_id) {
            return None;
        }
        let state = match run.status {
            // Stopped alone by the host: it says whether it may go on.
            AgentRunStatus::Cancelled | AgentRunStatus::Completed | AgentRunStatus::Interrupted
                if run
                    .stopped_by
                    .as_ref()
                    .is_some_and(|stop| stop.intent.is_some()) =>
            {
                return Some(self.stopped_alone_note(caller, run));
            }
            AgentRunStatus::Interrupted => "was interrupted and did not finish",
            AgentRunStatus::Paused
                if matches!(
                    run.task()
                        .ok()
                        .and_then(|task| AgentPause::of(&task.current_state)),
                    Some(AgentPause::ErrorRecovery { .. })
                ) =>
            {
                "is paused after an error"
            }
            AgentRunStatus::WaitingForInput => "is waiting for an answer to its questions",
            // Reopened when the caller is restarted after the stop.
            AgentRunStatus::Cancelled
                if run.stopped_by.is_some() && !crate::workflow::engine::is_superseded(run) =>
            {
                "was stopped with the workflow and did not finish"
            }
            // Completed after the wrap-up notice told it to stop: its
            // summary is a progress report its completion verifiers didn't
            // review, and `stopped_runs` leaves a nested run to its caller.
            AgentRunStatus::Completed if run.stopped_by.is_some() => {
                "stopped with the workflow before it finished: its summary is a progress report that no completion verifier reviewed"
            }
            _ => return None,
        };
        let (id, agent) = (run.agent_run_id, &run.agent_name);
        Some(format!(
            "Agent run {id} of agent '{agent}' {state}. {}",
            self.how_to_resume(caller, run)
        ))
    }

    /// What a caller is told about agent run `run`, which the workflow's
    /// host stopped alone (`Workflow::stop_run`): why, and whether it can
    /// be resumed (not when the host retired it).
    fn stopped_alone_note(&self, caller: &str, run: &AgentResult) -> String {
        let (id, agent) = (run.agent_run_id, &run.agent_name);
        let Some(stop) = &run.stopped_by else {
            return String::new();
        };
        let reason = stop.reason.trim().trim_end_matches('.');
        let why = if reason.is_empty() {
            String::new()
        } else {
            format!(" ({reason})")
        };
        let state = if run.status == AgentRunStatus::Completed {
            format!(
                "was stopped by the workflow's host{why} before it finished: its summary is a progress report that no completion verifier reviewed"
            )
        } else {
            format!("was stopped by the workflow's host{why} and did not finish")
        };
        let how = if stop.is_retired() {
            "The host retired it, so it can't be resumed.".to_string()
        } else {
            self.how_to_resume(caller, run)
        };
        format!("Agent run {id} of agent '{agent}' {state}. {how}")
    }

    /// How `caller` can resume agent run `run`, or that the host recovers
    /// it when the caller may not run its agent.
    fn how_to_resume(&self, caller: &str, run: &AgentResult) -> String {
        let (id, agent) = (run.agent_run_id, &run.agent_name);
        if self.is_allowed(caller) && self.is_target_allowed(caller, agent) {
            format!(
                "You can resume it instead of starting over: call {RUN_WORKFLOW_AGENT_TOOL_NAME} with agent \"{agent}\", resume_run \"{id}\" and a prompt telling it what happened and what to do next; it continues its own task."
            )
        } else {
            "You may not run that agent, so it stays unfinished for the workflow's host to recover."
                .to_string()
        }
    }

    /// An engine running agents for the workflow the services serve.
    pub(crate) fn engine(&self) -> WorkflowEngine {
        WorkflowEngine::from_shared(self.client.clone(), self.persistence.clone())
    }

    /// The workflow the services serve.
    fn context(&self) -> Result<Arc<SharedContext>, WorkflowError> {
        self.context.upgrade().ok_or_else(|| {
            WorkflowError::Channel("workflow services outlived their workflow".to_string())
        })
    }

    /// Fails unless `caller` may run `agent_name`.
    pub(crate) fn authorize(&self, caller: &str, agent_name: &str) -> Result<(), WorkflowError> {
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
        Ok(())
    }
}

/// Counts a running call until dropped.
pub(crate) struct CallGuard {
    count: Arc<AtomicU32>,
    ended: Arc<tokio::sync::Notify>,
}

impl CallGuard {
    #[cfg(test)]
    fn counted(self) -> Self {
        self.count.fetch_add(1, Ordering::AcqRel);
        self
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
        self.ended.notify_waiters();
    }
}

pub(crate) struct RunWorkflowAgentTool {
    services: WorkflowServices,
    caller: String,
    /// The depth of the caller's run. The agent it runs is one deeper.
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
            description: "Run another configured workflow agent for a bounded subtask and return its verified completion summary. Use only when delegation is materially useful. When a result names an agent run that did not finish and says you can resume it, pass its id as `resume_run` (with the same `agent`) to continue that run's own task instead of starting over; `prompt` is then your message to it.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition { name: "agent".to_string(), description: target_guidance, param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None },
                ToolParameterDefinition { name: "prompt".to_string(), description: "Concrete bounded task for that agent; with `resume_run`, your message to the resumed run: what happened and what to do next.".to_string(), param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None },
                ToolParameterDefinition { name: RESUME_RUN_PARAMETER.to_string(), description: "Optional: the id of an unfinished run of `agent` to resume (one a result told you can be resumed), instead of starting a new run.".to_string(), param_type: ToolParameterType::String, required: false, is_path: false, flag: None, default_value: None },
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
    /// call runs down (within its grace period) and records it. The call
    /// then returns.
    fn interruptible(&self) -> bool {
        false
    }

    /// A call running when the workflow stopped recovers the agent run it
    /// ran, or returns its result if it had finished (see
    /// [`WorkflowServices::recover_call`]).
    fn recovery_mode(&self) -> ToolRecovery {
        ToolRecovery::Resume
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
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
        let resume_run =
            match params
                .get(RESUME_RUN_PARAMETER)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
            {
                Some(id) => Some(uuid::Uuid::parse_str(id).map_err(|error| {
                    ToolError::ParameterValidation {
                        param_name: RESUME_RUN_PARAMETER.to_string(),
                        message: format!("not an agent run id: {error}"),
                    }
                })?),
                None => None,
            };
        // A run being stopped alone wraps up: it starts no agent runs.
        if !context.recovering
            && self
                .services
                .context
                .upgrade()
                .is_some_and(|workflow| workflow.run_control.is_run_stopping(context.task_id))
        {
            return Err(ToolError::ExecutionFailed(
                "Your run is being stopped, so it can't start or resume agent runs. Save your progress and call complete_task.".to_string(),
            ));
        }
        // The agent runs inside this call, which its task records.
        let parent = context.tool_call_id.map(|tool_call_id| TaskParent {
            task_id: context.task_id,
            tool_call_id,
        });
        let depth = self.depth + 1;
        let result = match (parent, resume_run) {
            (Some(parent), _) if context.recovering => {
                self.services
                    .recover_call(&self.caller, agent, prompt, resume_run, depth, parent)
                    .await
            }
            (_, Some(run_id)) => {
                self.services
                    .resume_run(
                        &self.caller,
                        context.task_id,
                        agent,
                        run_id,
                        prompt,
                        depth,
                        parent,
                    )
                    .await
            }
            (_, None) => {
                self.services
                    .run_agent_at(&self.caller, agent, prompt, depth, parent)
                    .await
            }
        };
        // A run a stop's wrap-up ended early: named after its progress
        // report, so that the caller can resume it.
        let result = result.map(|result| {
            let note = self.services.unfinished_note(&self.caller, &result);
            match note {
                Some(note) => format!("{}\n\n{note}", result.summary),
                None => result.summary,
            }
        });
        result.map_err(|error| {
            // The run the call ran, if it did not finish: named, so that
            // the caller can resume it rather than start over.
            let ran =
                self.services
                    .context
                    .upgrade()
                    .and_then(|workflow| match (resume_run, parent) {
                        (Some(run_id), _) => {
                            recorded_result(&workflow, run_id).filter(|run| run.agent_name == agent)
                        }
                        (None, Some(parent)) => workflow
                            .agent_results
                            .read()
                            .iter()
                            .rev()
                            .find(|result| result.info().parent == Some(parent))
                            .cloned(),
                        (None, None) => None,
                    });
            let note = ran.and_then(|run| self.services.unfinished_note(&self.caller, &run));
            ToolError::ExecutionFailed(match note {
                Some(note) => format!("{error}\n\n{note}"),
                None => error.to_string(),
            })
        })
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
    /// whose every call was refused, "active-call limit (0) reached"). The
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

    /// Over the limit a call fails, or waits for a place with
    /// `queue_active_calls`. `unlimited_active_calls` lifts the limit.
    #[tokio::test]
    async fn calls_over_the_limit_fail_wait_or_run() {
        let services = |config: WorkflowServicesConfig| {
            WorkflowServices::new(
                Weak::new(),
                Arc::new(arko_ai::LLMClient::new()),
                None,
                config.max_active_calls(1),
            )
        };
        let failing = services(WorkflowServicesConfig::default());
        let held = failing.enter_call(true).await.unwrap();
        let error = failing.enter_call(true).await.err().unwrap();
        assert!(error.to_string().contains("active-call limit (1) reached"));
        drop(held);
        assert!(failing.enter_call(true).await.is_ok());

        let queueing = Arc::new(services(
            WorkflowServicesConfig::default().queue_active_calls(),
        ));
        let held = queueing.enter_call(true).await.unwrap();
        let waiting = tokio::spawn({
            let queueing = queueing.clone();
            async move { queueing.enter_call(true).await.map(|_| ()) }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!waiting.is_finished(), "it waits for a place");
        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        let unlimited = services(WorkflowServicesConfig::default().unlimited_active_calls());
        let mut guards = Vec::new();
        for _ in 0..8 {
            guards.push(unlimited.enter_call(true).await.unwrap());
        }
        assert_eq!(unlimited.active_calls.load(Ordering::Acquire), 8);
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
