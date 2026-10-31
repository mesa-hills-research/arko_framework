//! Result types returned after workflow and agent execution.
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::budget::CostLedger;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub workflow_run_id: uuid::Uuid,
    pub agent_name: String,
    pub agent_run_id: uuid::Uuid,
    pub event: arko_agents::runtime::event::AgentEvent,
}

/// Terminal or suspended outcome of one configured-agent invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentRunStatus {
    Completed,
    Failed,
    Interrupted,
    /// The task ended `TaskState::Cancelled`, stopped with the workflow run
    /// (cancelled, or aborted by a hook), or by the workflow (over budget).
    Cancelled,
    /// Suspended at an `ask_user` question. `Workflow::continue_agent`
    /// answers it.
    WaitingForInput,
    /// Suspended awaiting a tool approval or error-recovery decision the
    /// workflow's policies leave to the caller (see
    /// [`AgentPause`](crate::policy::AgentPause)).
    /// `Workflow::resume_agent` continues it.
    Paused,
}

impl AgentRunStatus {
    /// A pause (`WaitingForInput` or `Paused`) from which the run can go on.
    pub fn is_paused(&self) -> bool {
        matches!(self, Self::WaitingForInput | Self::Paused)
    }

    /// The status a run whose task ended in `state` records.
    pub fn of_state(state: &arko_agents::TaskState) -> Self {
        use arko_agents::TaskState;
        match state {
            TaskState::Completed { .. } => Self::Completed,
            TaskState::Cancelled { .. } => Self::Cancelled,
            TaskState::Failed { .. } => Self::Failed,
            TaskState::AwaitingUserResponse { .. } => Self::WaitingForInput,
            TaskState::AwaitingToolApproval { .. }
            | TaskState::AwaitingBatchToolApproval { .. }
            | TaskState::AwaitingErrorRecoveryApproval { .. } => Self::Paused,
            _ => Self::Interrupted,
        }
    }
}

/// Puts `result` in `results`, replacing an earlier result of its task.
pub(crate) fn upsert_result(results: &mut Vec<AgentResult>, result: AgentResult) {
    match results
        .iter_mut()
        .find(|existing| existing.agent_run_id == result.agent_run_id)
    {
        Some(existing) => *existing = result,
        None => results.push(result),
    }
}

impl AgentResult {
    /// The result a run whose task ended as `task` records, with its
    /// status, summary and error from the task's state.
    pub fn of_task(agent_name: String, task: arko_agents::Task) -> Self {
        use arko_agents::TaskState;
        let status = AgentRunStatus::of_state(&task.current_state);
        let (summary, error) = match &task.current_state {
            TaskState::Completed { summary } => (summary.clone(), None),
            TaskState::Cancelled { reason }
            | TaskState::Failed { reason, .. }
            | TaskState::Interrupted { reason } => (String::new(), Some(reason.clone())),
            TaskState::AwaitingUserResponse { questions } => {
                (String::new(), Some(questions.clone()))
            }
            TaskState::AwaitingToolApproval { .. }
            | TaskState::AwaitingBatchToolApproval { .. }
            | TaskState::AwaitingErrorRecoveryApproval { .. } => (
                String::new(),
                crate::policy::AgentPause::of(&task.current_state)
                    .map(|pause| format!("Paused, {pause}")),
            ),
            state => (
                String::new(),
                Some(format!("Agent ended in non-terminal state: {state:?}")),
            ),
        };
        Self {
            agent_run_id: task.id,
            agent_name,
            status,
            summary,
            error,
            stopped_by: None,
            task: Arc::new(RunTask::loaded(task)),
        }
    }

    /// The result of `record`, whose task is `task`.
    pub(crate) fn of_record(
        record: &crate::persistence::state::AgentRunRecord,
        task: RunTask,
    ) -> Self {
        Self {
            agent_run_id: record.agent_run_id,
            agent_name: record.agent_name.clone(),
            status: record.status.clone(),
            summary: record.summary.clone(),
            error: record.error.clone(),
            stopped_by: record.stopped_by.clone(),
            task: Arc::new(task),
        }
    }
}

/// What a run's task holds besides its conversation, which lists, notices
/// and a resume's checks read without loading the task. See
/// [`AgentResult::info`].
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    /// The configuration the task runs with (`Task::agent_config`).
    pub agent_config: arko_agents::AgentConfig,
    /// The task's first prompt (`Task::initial_task_prompt`).
    pub initial_task_prompt: String,
    /// When the task started (`Task::start_time`), to the second once saved.
    #[serde(with = "chrono::serde::ts_seconds")]
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// When its last turn ended (`Task::end_time`), to the second once saved.
    #[serde(
        with = "chrono::serde::ts_seconds_option",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// The task and tool call the run ran inside (`Task::parent`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<arko_agents::TaskParent>,
    /// Whether the host gave the run another configuration
    /// (`Task::reconfigured`).
    #[serde(default)]
    pub reconfigured: bool,
    /// Its model calls' usage added up (`Task::total_usage`), with the
    /// cached tokens, the costs and the billing kind. Its `billed()` is
    /// [`AgentResult::total_cost`].
    pub usage: arko_ai::UsageInfo,
    /// How many model calls `usage` adds up (`Task::usage_history`'s length).
    pub model_calls: usize,
    /// What the task knows of its current conversation's latest completed
    /// model request (`Task::latest_request`), which its prompt cache lasts
    /// from. A checkpoint saved before it existed reads as
    /// [`RequestRecord::Unrecorded`](arko_agents::RequestRecord::Unrecorded),
    /// as its task does.
    #[serde(
        default,
        skip_serializing_if = "arko_agents::RequestRecord::is_unrecorded"
    )]
    pub latest_request: arko_agents::RequestRecord,
}

impl TaskInfo {
    /// What `task` holds besides its conversation.
    pub(crate) fn of(task: &arko_agents::Task) -> Self {
        Self {
            agent_config: task.agent_config.clone(),
            initial_task_prompt: task.initial_task_prompt.clone(),
            start_time: task.start_time,
            end_time: task.end_time,
            parent: task.parent,
            reconfigured: task.reconfigured,
            usage: task.total_usage(),
            model_calls: task.usage_history.len(),
            latest_request: task.latest_request.clone(),
        }
    }
}

/// A run's task, given, or read from its task file the first time it is
/// needed. Clones of the result share it.
pub(crate) struct RunTask {
    info: TaskInfo,
    task: std::sync::OnceLock<Result<arko_agents::Task, String>>,
    /// Where the task is read from, for one not given.
    file: Option<crate::persistence::state::TaskFile>,
}

impl RunTask {
    pub(crate) fn loaded(task: arko_agents::Task) -> Self {
        Self {
            info: TaskInfo::of(&task),
            task: std::sync::OnceLock::from(Ok(task)),
            file: None,
        }
    }

    /// The task in `file`, read when it is first needed.
    pub(crate) fn in_file(info: TaskInfo, file: crate::persistence::state::TaskFile) -> Self {
        Self {
            info,
            task: std::sync::OnceLock::new(),
            file: Some(file),
        }
    }

    fn get(&self) -> Result<&arko_agents::Task, String> {
        self.task
            .get_or_init(|| {
                let Some(file) = &self.file else {
                    unreachable!("a task without a file is given");
                };
                let mut warnings = Vec::new();
                let task = file.read(&mut warnings);
                if let Ok(task) = &task {
                    warnings.extend(file.state_warning(task));
                }
                for warning in warnings {
                    log::warn!("{warning}");
                }
                task
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl std::fmt::Debug for RunTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("RunTask");
        debug.field("info", &self.info);
        match (self.task.get(), &self.file) {
            (Some(Ok(task)), _) => debug.field("task", task),
            (Some(Err(error)), _) => debug.field("error", error),
            (None, Some(file)) => debug.field("file", &file.path),
            (None, None) => debug.field("task", &None::<()>),
        };
        debug.finish()
    }
}

/// Result of a single agent run within a workflow.
///
/// A result holds its run's task. One restored from a checkpoint
/// ([`WorkflowBuilder::resume_from`](crate::WorkflowBuilder::resume_from),
/// [`WorkflowState::agent_results_in`](crate::persistence::state::WorkflowState::agent_results_in))
/// for a run that completed reads its task from its task file only when
/// [`task`](Self::task) is first called. [`info`](Self::info) has what the
/// task holds besides its conversation without reading it. Clones share
/// the task, so a clone is cheap.
///
/// It is serialized with its task (the task is read for that), as `task`
/// next to the fields.
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    pub status: AgentRunStatus,
    pub summary: String,
    pub error: Option<String>,
    /// The stop that ended the run, if one did. See
    /// [`stopped_by`](Self::stopped_by). `None` in results recorded by an
    /// older version.
    pub stopped_by: Option<crate::workflow::stop::StopRecord>,
    task: Arc<RunTask>,
}

/// An [`AgentResult`] as it is serialized.
#[derive(Serialize)]
struct SavedResult<'a> {
    agent_run_id: uuid::Uuid,
    agent_name: &'a str,
    status: &'a AgentRunStatus,
    summary: &'a str,
    error: &'a Option<String>,
    task: &'a arko_agents::Task,
    #[serde(skip_serializing_if = "Option::is_none")]
    stopped_by: Option<&'a crate::workflow::stop::StopRecord>,
}

/// An [`AgentResult`] as it is deserialized.
#[derive(Deserialize)]
struct LoadedResult {
    agent_run_id: uuid::Uuid,
    agent_name: String,
    status: AgentRunStatus,
    summary: String,
    error: Option<String>,
    task: arko_agents::Task,
    #[serde(default)]
    stopped_by: Option<crate::workflow::stop::StopRecord>,
}

impl Serialize for AgentResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let task = self.task().map_err(serde::ser::Error::custom)?;
        SavedResult {
            agent_run_id: self.agent_run_id,
            agent_name: &self.agent_name,
            status: &self.status,
            summary: &self.summary,
            error: &self.error,
            task,
            stopped_by: self.stopped_by.as_ref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AgentResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let loaded = LoadedResult::deserialize(deserializer)?;
        Ok(Self {
            agent_run_id: loaded.agent_run_id,
            agent_name: loaded.agent_name,
            status: loaded.status,
            summary: loaded.summary,
            error: loaded.error,
            stopped_by: loaded.stopped_by,
            task: Arc::new(RunTask::loaded(loaded.task)),
        })
    }
}

impl AgentResult {
    /// The run's task, with its conversation. A result restored from a
    /// checkpoint may read it from its task file now, once (the first
    /// caller reads it, from any clone or thread, and later ones get it),
    /// and then keeps it. Fails with [`WorkflowError::Persistence`](crate::WorkflowError::Persistence)
    /// when the file can't be read, the same error each time.
    pub fn task(&self) -> Result<&arko_agents::Task, crate::WorkflowError> {
        self.task.get().map_err(crate::WorkflowError::Persistence)
    }

    /// What the run's task holds besides its conversation, without reading
    /// the task.
    pub fn info(&self) -> &TaskInfo {
        &self.task.info
    }

    /// Returns the total cost of this agent run.
    pub fn total_cost(&self) -> f64 {
        self.info().usage.billed()
    }

    /// Returns the total input tokens consumed.
    pub fn total_input_tokens(&self) -> usize {
        self.info().usage.input_tokens
    }

    /// Returns the total output tokens generated.
    pub fn total_output_tokens(&self) -> usize {
        self.info().usage.output_tokens
    }

    /// The stop that ended this run, when a stop the host asked for ended
    /// it. The run was cancelled by
    /// [`Workflow::stop`](crate::Workflow::stop),
    /// [`Workflow::cancel`](crate::Workflow::cancel) or its own stop
    /// ([`Workflow::stop_run`](crate::Workflow::stop_run)) (status
    /// `Cancelled`, including a run aborted after the grace period), or it
    /// completed after its stop's wrap-up notice told it to finish (status
    /// `Completed`). A run stopped alone carries its stop also when it was
    /// interrupted in its wrap-up (status `Interrupted`, such as a run the
    /// process ended), and its record's
    /// [`intent`](crate::workflow::stop::StopRecord::intent) says what the
    /// host means to do with it. A stopped run is blameless. Its outcome
    /// says nothing about the agent's work, so don't count it as a failure
    /// (or a success). [`Workflow::stopped_runs`](crate::Workflow::stopped_runs)
    /// lists it for a restart unless the host retired it. `None` for every
    /// other run, a run a hook's abort ended, and runs recorded by an older
    /// version.
    pub fn stopped_by(&self) -> Option<&crate::workflow::stop::StopRecord> {
        self.stopped_by.as_ref()
    }

    /// Why the run failed, when it failed at an error-recovery pause the
    /// error recovery policy (or a caller) rejected, with the pause's cause, as
    /// its task recorded it (`TaskState::Failed::cause`). For example
    /// [`ErrorRecoveryCause::NonRetryable`](arko_agents::ErrorRecoveryCause::NonRetryable) means
    /// the model request needs a new credential (an expired login) or a
    /// configuration change, so starting the agent again only helps once
    /// that is fixed. `None` for a run that didn't fail, failed otherwise,
    /// or was saved by an older version.
    pub fn failure_cause(&self) -> Option<&arko_agents::ErrorRecoveryCause> {
        if self.status != AgentRunStatus::Failed {
            return None;
        }
        match &self.task().ok()?.current_state {
            arko_agents::TaskState::Failed { cause, .. } => cause.as_ref(),
            _ => None,
        }
    }

    /// The kind of the model-request error the run failed on, when it
    /// failed at an error-recovery pause that recorded one
    /// (`TaskState::Failed::error_kind`), for example
    /// [`ErrorKind::Authentication`](arko_ai::ErrorKind::Authentication) for
    /// a lost login, which fails every run on that credential until it is
    /// renewed, or [`ErrorKind::ContextOverflow`](arko_ai::ErrorKind::ContextOverflow)
    /// for a request over the model's input limit, which fails only this
    /// run. A host decides by it instead of matching error text. `None` for
    /// a run that didn't fail, failed otherwise or on an unclassified
    /// error, or was saved by an older version.
    pub fn failure_kind(&self) -> Option<arko_ai::ErrorKind> {
        if self.status != AgentRunStatus::Failed {
            return None;
        }
        match &self.task().ok()?.current_state {
            arko_agents::TaskState::Failed { error_kind, .. } => *error_kind,
            _ => None,
        }
    }
}

/// Aggregated result of the entire workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_run_id: uuid::Uuid,
    pub workflow_name: String,
    pub metadata: crate::workflow::config::WorkflowMetadata,
    pub total_cost: f64,
    /// Complete cost ledger containing all agent and manual cost entries
    /// in chronological order.
    pub cost_ledger: CostLedger,
    pub agent_results: Vec<AgentResult>,
    /// Set when the run was stopped (cancelled, or aborted by a hook).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<crate::workflow::cancel::Cancellation>,
}
