//! Serializable workflow state for checkpointing.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::budget::CostLedger;
use crate::freshness::FreshnessSnapshot;
use crate::memory::MemoryTier;
use crate::messaging::AgentMessage;
use crate::workflow::result::{AgentResult, AgentRunStatus, RunTask, TaskInfo};

/// A serializable snapshot of the entire workflow state.
#[derive(Serialize, Deserialize)]
pub struct WorkflowState {
    #[serde(default = "uuid::Uuid::nil")]
    pub workflow_run_id: uuid::Uuid,
    pub workflow_name: String,
    #[serde(default)]
    pub workflow_metadata: crate::workflow::config::WorkflowMetadata,
    pub checkpoint_id: uuid::Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memory_snapshot: Vec<MemoryTier>,
    pub message_queues: HashMap<String, Vec<AgentMessage>>,
    pub budget_spent: f64,
    /// Complete cost ledger at the time of the checkpoint.
    ///
    /// Defaults to an empty ledger when deserializing checkpoints
    /// created before the ledger was introduced.
    #[serde(default)]
    pub cost_ledger: CostLedger,
    // Older checkpoints may contain `oauth_exempt_agents`; it is ignored now
    // that subscription calls record a billed amount of zero.
    #[serde(default)]
    pub freshness_snapshot: Option<FreshnessSnapshot>,
    /// Finalized agent results with their tasks, as checkpoints before
    /// [`agent_runs`](Self::agent_runs) held them. Empty in newer ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_results: Vec<AgentResult>,
    /// Finalized agent results, each referring to the version of its task
    /// the checkpoint recorded instead of holding the task (a version is
    /// stored once, however many checkpoints refer to it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_runs: Vec<AgentRunRecord>,
    /// The agent runs whose latest task state could not be saved. Their
    /// records in [`agent_runs`](Self::agent_runs) refer to their last
    /// saved state instead, and a run with none is left out. Resuming from
    /// the checkpoint restores them as last saved, with a warning.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_task_states: Vec<MissingTaskState>,
    #[serde(default)]
    pub verification_records: Vec<crate::verification::VerificationRecord>,
    #[serde(default)]
    pub distillation_records: Vec<crate::distillation::DistillationRecord>,
    #[serde(default)]
    pub completed_agents: Vec<SerializableAgentResult>,
    /// Set when the run was stopped (cancelled, or aborted by a hook) before
    /// this checkpoint. Resuming from the checkpoint starts the run again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<crate::workflow::cancel::Cancellation>,
    /// Agent statuses and what each agent run has been told about the
    /// agents it may message. Absent in checkpoints from before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_registry: Option<crate::messaging::registry::RegistrySnapshot>,
    /// The notes posted to the run's agents, what each agent run has been
    /// given, and how much of the control folder has been read. Absent
    /// while there is nothing to record, and in checkpoints from before
    /// notes existed. A resume then starts with none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<crate::notes::NotesSnapshot>,
    /// The budget limits in force at the checkpoint, which may have been
    /// changed while the run was live
    /// ([`Workflow::set_budget`](crate::Workflow::set_budget)). Absent in
    /// checkpoints from before it existed. See
    /// [`WorkflowBuilder::resume_from`](crate::WorkflowBuilder::resume_from)
    /// for how a resume combines them with the builder's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_limits: Option<BudgetLimits>,
    /// The plan budget's limits in force at the checkpoint and how much of
    /// each plan window the run had used, from the levels it first saw
    /// (see [`WorkflowBuilder::plan_budget`](crate::WorkflowBuilder::plan_budget)).
    /// Absent in checkpoints from before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_budget: Option<crate::budget::PlanBudgetState>,
    /// The background subagents, their mailboxes and the run ids handed out
    /// (see [`crate::subagents`]). Absent while there is nothing to record,
    /// and in checkpoints from before subagents existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagents: Option<crate::subagents::SubagentsSnapshot>,
}

/// The budget limits a checkpoint records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BudgetLimits {
    /// The workflow's limit in dollars. `None` means no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    /// The agents' own limits in dollars, by agent name.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub agent_limits: std::collections::BTreeMap<String, f64>,
}

/// A finalized agent result in a checkpoint, with its task in a task
/// version file in the run directory.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentRunRecord {
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    pub status: AgentRunStatus,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The version of the task the checkpoint recorded, relative to the
    /// run directory, `agents/<agent_run_id>/<content hash>.json`, or, for
    /// a run that was running, its live file (`agents/<agent_run_id>/live.json`)
    /// in the live checkpoint.
    pub task_file: String,
    /// For a run that was running (status `Interrupted`), how many entries
    /// of its task's `usage_history` the checkpoint's cost ledger has
    /// charged. Resuming charges the rest. Absent for a finished run, and in
    /// checkpoints from before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_charged: Option<usize>,
    /// The stop that ended the run
    /// ([`AgentResult::stopped_by`](crate::AgentResult::stopped_by)).
    /// Absent for other runs, and in checkpoints from before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_by: Option<crate::workflow::stop::StopRecord>,
    /// What the run's task holds besides its conversation
    /// ([`AgentResult::info`]), so that a resume reads the task file of a
    /// run that completed only when its task is needed. Absent for a run
    /// that was running, and in checkpoints from before it existed, whose
    /// task files are all read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<TaskInfo>,
}

impl AgentRunRecord {
    pub(crate) fn of(result: &AgentResult, task_file: String) -> Self {
        Self {
            agent_run_id: result.agent_run_id,
            agent_name: result.agent_name.clone(),
            status: result.status.clone(),
            summary: result.summary.clone(),
            error: result.error.clone(),
            task_file,
            // A recorded interrupted run's usage is all in the ledger.
            usage_charged: (result.status == AgentRunStatus::Interrupted)
                .then_some(result.info().model_calls),
            stopped_by: result.stopped_by.clone(),
            info: Some(result.info().clone()),
        }
    }

    /// Whether a resume reads this run's task only when it is needed, for a
    /// run that completed (not in a stop's wrap-up) whose task the
    /// checkpoint describes. Every other run's task is read at once.
    fn loads_on_demand(&self) -> bool {
        self.info.is_some() && self.status == AgentRunStatus::Completed && self.stopped_by.is_none()
    }
}

/// The task file of a checkpoint's agent run.
pub(crate) struct TaskFile {
    pub(crate) path: std::path::PathBuf,
    images: std::sync::Arc<super::images::ImageFiles>,
    agent_name: String,
    agent_run_id: uuid::Uuid,
    /// The status the checkpoint recorded, and whether it charged part of
    /// the run's usage (a run that was running).
    status: AgentRunStatus,
    usage_charged: bool,
}

impl TaskFile {
    fn of(
        record: &AgentRunRecord,
        run_dir: &std::path::Path,
        images: &std::sync::Arc<super::images::ImageFiles>,
    ) -> Self {
        Self {
            path: run_dir.join(&record.task_file),
            images: images.clone(),
            agent_name: record.agent_name.clone(),
            agent_run_id: record.agent_run_id,
            status: record.status.clone(),
            usage_charged: record.usage_charged.is_some(),
        }
    }

    /// Reads the task, with a warning in `warnings` for its images no longer
    /// available. Fails with what a [`WorkflowError::Persistence`](crate::WorkflowError::Persistence)
    /// says.
    pub(crate) fn read(&self, warnings: &mut Vec<String>) -> Result<arko_agents::Task, String> {
        std::fs::read_to_string(&self.path)
            .map_err(crate::WorkflowError::from)
            .and_then(|json| {
                let mut task = self
                    .images
                    .with(|| serde_json::from_str::<arko_agents::Task>(&json))?;
                // An image file gone or damaged doesn't keep the task from
                // loading: the image becomes a line saying so.
                let gone = task.settle_unavailable_images();
                if !gone.is_empty() {
                    warnings.push(format!(
                        "{} image(s) of agent '{}' ({}) are no longer available, each replaced by a line saying so: {}",
                        gone.len(),
                        self.agent_name,
                        self.agent_run_id,
                        gone.join("; ")
                    ));
                }
                Ok(task)
            })
            .map_err(|error: crate::WorkflowError| {
                format!(
                    "the checkpoint's task file {} for agent '{}' can't be loaded: {error}",
                    self.path.display(),
                    self.agent_name
                )
            })
    }

    /// The warning when `task` holds another state than the checkpoint
    /// recorded.
    pub(crate) fn state_warning(&self, task: &arko_agents::Task) -> Option<String> {
        let now = AgentRunStatus::of_state(&task.current_state);
        // A cancellation older versions recorded as a failure loads as
        // cancelled: the same outcome. A run that was running is
        // interrupted whatever state its task was saved in (a wait the
        // workflow was answering, say).
        let same = now == self.status
            || (self.status == AgentRunStatus::Failed && now == AgentRunStatus::Cancelled)
            || (self.status == AgentRunStatus::Interrupted && self.usage_charged);
        (!same).then(|| {
            format!(
                "The task file {} of agent '{}' holds another state ({now:?}) than the checkpoint recorded ({:?}).",
                self.path.display(),
                self.agent_name,
                self.status
            )
        })
    }
}

/// An agent run whose latest task state a checkpoint lacks, because it
/// could not be saved.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MissingTaskState {
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    /// The status of the state that is missing.
    pub status: AgentRunStatus,
    /// The task version the checkpoint refers to instead, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_saved: Option<String>,
    /// Why it could not be saved.
    pub reason: String,
}

impl WorkflowState {
    /// The checkpoint's agent results, including those it holds with their
    /// tasks (a checkpoint written before task files were referenced), then
    /// those whose tasks are in the task files of `run_dir`, the directory
    /// holding the checkpoint. The task of a run that completed is read
    /// from its file when [`AgentResult::task`] is first called, when the
    /// checkpoint describes it ([`AgentRunRecord::info`]), and every other
    /// task now. A run whose latest state the checkpoint lacks
    /// ([`missing_task_states`](Self::missing_task_states)) is restored as
    /// last saved, and one never saved isn't. Each is logged as a warning.
    pub fn agent_results_in(
        &self,
        run_dir: &std::path::Path,
    ) -> Result<Vec<AgentResult>, crate::WorkflowError> {
        let (results, warnings) = self.agent_results_with_warnings(run_dir)?;
        for warning in warnings {
            log::warn!("{warning}");
        }
        Ok(results)
    }

    /// [`agent_results_in`](Self::agent_results_in), with its warnings.
    pub(crate) fn agent_results_with_warnings(
        &self,
        run_dir: &std::path::Path,
    ) -> Result<(Vec<AgentResult>, Vec<String>), crate::WorkflowError> {
        let mut results = self.agent_results.clone();
        let mut warnings = Vec::new();
        // Absolute, as the tasks read on demand are read later, maybe after
        // the process changed its working directory.
        let run_dir = &std::path::absolute(if run_dir.as_os_str().is_empty() {
            std::path::Path::new(".")
        } else {
            run_dir
        })?;
        // The tasks' images are in the run directory's image files.
        let images = super::images::ImageFiles::in_run_dir(run_dir);
        for record in &self.agent_runs {
            let file = TaskFile::of(record, run_dir, &images);
            let missing = self
                .missing_task_states
                .iter()
                .find(|missing| missing.agent_run_id == record.agent_run_id);
            if let (Some(info), None) = (&record.info, missing)
                && record.loads_on_demand()
            {
                let task = RunTask::in_file(info.clone(), file);
                results.push(AgentResult::of_record(record, task));
                continue;
            }
            let task = file
                .read(&mut warnings)
                .map_err(crate::WorkflowError::Persistence)?;
            if let Some(missing) = missing {
                let mut result = AgentResult::of_task(record.agent_name.clone(), task);
                result.stopped_by = record.stopped_by.clone();
                warnings.push(format!(
                    "The checkpoint lacks the latest state of agent '{}' ({}), which could not be saved ({}). It was {:?}, but resumes as last saved, {:?}, from {}.",
                    record.agent_name,
                    record.agent_run_id,
                    missing.reason,
                    missing.status,
                    result.status,
                    file.path.display()
                ));
                results.push(result);
                continue;
            }
            warnings.extend(file.state_warning(&task));
            results.push(AgentResult::of_record(record, RunTask::loaded(task)));
        }
        for missing in &self.missing_task_states {
            if missing.last_saved.is_none() {
                warnings.push(format!(
                    "The checkpoint lacks agent '{}' ({}). No state of its task could be saved ({}), so it is not restored.",
                    missing.agent_name, missing.agent_run_id, missing.reason
                ));
            }
        }
        Ok((results, warnings))
    }
}

/// Lightweight serializable agent result for checkpoints.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SerializableAgentResult {
    pub agent_name: String,
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness::{ActorId, ContextFreshnessConfig, ResourceId, ResourceTracker};
    use std::path::PathBuf;

    #[test]
    fn workflow_state_with_populated_freshness_is_json_serializable() {
        let workspace = PathBuf::from("/workspace");
        let tracker = ResourceTracker::new(workspace.clone(), ContextFreshnessConfig::default());
        let reader = ActorId::new(uuid::Uuid::new_v4(), "reviewer");
        let writer = ActorId::new(uuid::Uuid::new_v4(), "coder");
        let resource = ResourceId::File(workspace.join("src/lib.rs"));
        tracker.record_read(&reader, resource.clone());
        tracker.record_write(&writer, resource);
        let run_id = uuid::Uuid::new_v4();
        let state = WorkflowState {
            workflow_run_id: run_id,
            workflow_name: "checkpoint-test".to_string(),
            workflow_metadata: crate::workflow::config::WorkflowMetadata::default(),
            checkpoint_id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            memory_snapshot: Vec::new(),
            message_queues: HashMap::new(),
            budget_spent: 0.0,
            cost_ledger: CostLedger::default(),
            freshness_snapshot: Some(tracker.snapshot()),
            agent_results: Vec::new(),
            agent_runs: Vec::new(),
            missing_task_states: Vec::new(),
            verification_records: Vec::new(),
            distillation_records: Vec::new(),
            completed_agents: Vec::new(),
            cancellation: None,
            agent_registry: None,
            notes: None,
            budget_limits: None,
            plan_budget: None,
            subagents: None,
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: WorkflowState = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.workflow_run_id, run_id);
        assert!(restored.freshness_snapshot.is_some());
    }

    /// A checkpoint from an early version still loads, with fields this
    /// version doesn't know: the removed OAuth exemption set, and settings
    /// of the removed MESI layer an application may have stored alongside.
    #[test]
    fn an_early_checkpoint_with_removed_fields_still_loads() {
        let checkpoint = serde_json::json!({
            "workflow_name": "legacy",
            "checkpoint_id": uuid::Uuid::new_v4(),
            "timestamp": "2026-09-01T12:00:00Z",
            "memory_snapshot": [],
            "message_queues": {},
            "budget_spent": 0.25,
            "oauth_exempt_agents": ["coder"],
            "mesi_config": {
                "block_on_conflict": true,
                "write_tools": ["write_file"],
                "read_tools": ["read_file"]
            },
            "file_tracker": {"files": {}}
        });
        let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
        assert_eq!(state.workflow_name, "legacy");
        assert_eq!(state.workflow_run_id, uuid::Uuid::nil());
        assert!(state.agent_results.is_empty());

        assert!(state.cancellation.is_none());

        let config: crate::workflow::config::WorkflowConfig =
            serde_json::from_value(serde_json::json!({
                "name": "legacy",
                "workspace": "/workspace",
                "mesi_config": {"block_on_conflict": false}
            }))
            .unwrap();
        assert_eq!(config.name, "legacy");
    }

    /// A checkpoint from before the task journal (its run records without
    /// `usage_charged`, its tasks without `parent` or running calls) still
    /// loads. A record of a run that was running loads interrupted, without
    /// a warning, whatever state its task was saved in.
    #[test]
    fn records_from_before_the_task_journal_still_load() {
        let workspace = crate::test_support::Workspace::new();
        let mut task = arko_agents::Task::new(
            crate::test_support::agent_config("Coder"),
            "prompt".into(),
            workspace.0.clone(),
            None,
        );
        task.set_state(arko_agents::TaskState::Completed {
            summary: "done".into(),
        });
        let mut task_json = serde_json::to_value(&task).unwrap();
        let fields = task_json.as_object_mut().unwrap();
        assert!(!fields.contains_key("parent"));
        assert!(!fields.contains_key("running_tool_calls"));
        std::fs::create_dir_all(workspace.0.join(format!("agents/{}", task.id))).unwrap();
        let old_file = format!("agents/{}/old.json", task.id);
        std::fs::write(workspace.0.join(&old_file), task_json.to_string()).unwrap();
        let mut running = task.clone();
        running.id = uuid::Uuid::new_v4();
        running.set_state(arko_agents::TaskState::AwaitingErrorRecoveryApproval {
            error_message: "overloaded".into(),
            cause: arko_agents::ErrorRecoveryCause::Unknown,
            error_kind: None,
        });
        std::fs::create_dir_all(workspace.0.join(format!("agents/{}", running.id))).unwrap();
        let live_file = format!("agents/{}/live.json", running.id);
        std::fs::write(
            workspace.0.join(&live_file),
            serde_json::to_string(&running).unwrap(),
        )
        .unwrap();
        let checkpoint = serde_json::json!({
            "workflow_run_id": uuid::Uuid::new_v4(),
            "workflow_name": "journal",
            "checkpoint_id": uuid::Uuid::new_v4(),
            "timestamp": "2026-09-01T12:00:00Z",
            "memory_snapshot": [],
            "message_queues": {},
            "budget_spent": 0.0,
            "agent_runs": [
                {
                    "agent_run_id": task.id,
                    "agent_name": "coder",
                    "status": "Completed",
                    "summary": "done",
                    "task_file": old_file
                },
                {
                    "agent_run_id": running.id,
                    "agent_name": "coder",
                    "status": "Interrupted",
                    "summary": "",
                    "error": crate::persistence::RUNNING_AT_CHECKPOINT,
                    "task_file": live_file,
                    "usage_charged": 0
                }
            ]
        });
        let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
        assert_eq!(state.agent_runs[0].usage_charged, None);
        assert_eq!(state.agent_runs[1].usage_charged, Some(0));
        let (results, warnings) = state.agent_results_with_warnings(&workspace.0).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(results[0].status, AgentRunStatus::Completed);
        assert!(results[0].task().unwrap().parent.is_none());
        assert_eq!(results[1].status, AgentRunStatus::Interrupted);
        assert_eq!(
            results[1].error.as_deref(),
            Some(crate::persistence::RUNNING_AT_CHECKPOINT)
        );
    }

    /// An agent cancelled before cancellation had its own task state was
    /// saved as failed with the runtime's cancellation reason. It loads as
    /// cancelled.
    #[test]
    fn a_checkpointed_legacy_cancellation_loads_as_cancelled() {
        let config = arko_agents::AgentConfig::builder(
            "coder".into(),
            "Coder".into(),
            arko_agents::AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let task = arko_agents::Task::new(config, "prompt".into(), PathBuf::from("/w"), None);
        let mut task_json = serde_json::to_value(&task).unwrap();
        task_json["current_state"] =
            serde_json::json!({"Failed": {"reason": "Task cancelled by user."}});
        let checkpoint = serde_json::json!({
            "workflow_run_id": uuid::Uuid::new_v4(),
            "workflow_name": "legacy",
            "checkpoint_id": uuid::Uuid::new_v4(),
            "timestamp": "2026-09-01T12:00:00Z",
            "memory_snapshot": [],
            "message_queues": {},
            "budget_spent": 0.0,
            "agent_results": [{
                "agent_run_id": task.id,
                "agent_name": "coder",
                "status": "Cancelled",
                "summary": "",
                "error": "Task cancelled by user.",
                "task": task_json
            }]
        });
        let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
        let result = &state.agent_results[0];
        assert_eq!(
            result.status,
            crate::workflow::result::AgentRunStatus::Cancelled
        );
        assert!(matches!(
            result.task().unwrap().current_state,
            arko_agents::TaskState::Cancelled { .. }
        ));
        assert!(state.cancellation.is_none());
    }

    /// A checkpoint of one completed run of agent `coder`, whose task file is
    /// in `run_dir` and whose record describes its task (`info`) or not.
    fn one_completed_run(run_dir: &std::path::Path, info: bool) -> WorkflowState {
        let mut task = arko_agents::Task::new(
            crate::test_support::agent_config("Coder"),
            "prompt".into(),
            run_dir.into(),
            None,
        );
        task.set_state(arko_agents::TaskState::Completed {
            summary: "done".into(),
        });
        let file = format!("agents/{}/version.json", task.id);
        std::fs::create_dir_all(run_dir.join(&file).parent().unwrap()).unwrap();
        std::fs::write(run_dir.join(&file), serde_json::to_string(&task).unwrap()).unwrap();
        let mut record = AgentRunRecord::of(&AgentResult::of_task("coder".into(), task), file);
        if !info {
            record.info = None;
        }
        serde_json::from_value(serde_json::json!({
            "workflow_name": "state-test",
            "checkpoint_id": uuid::Uuid::new_v4(),
            "timestamp": "2026-10-01T00:00:00Z",
            "memory_snapshot": [],
            "message_queues": {},
            "budget_spent": 0.0,
            "agent_runs": [record],
        }))
        .unwrap()
    }

    /// A task file that can't be read fails with the same error whether its
    /// task is read at once (a record without `info`) or on demand.
    #[test]
    fn a_task_file_that_cant_be_read_fails_alike_at_once_and_on_demand() {
        let workspace = crate::test_support::Workspace::new();
        let described = one_completed_run(&workspace.0, true);
        let older = one_completed_run(&workspace.0, false);
        std::fs::remove_dir_all(workspace.0.join("agents")).unwrap();
        let results = described.agent_results_in(&workspace.0).unwrap();
        let on_demand = results[0].task().unwrap_err();
        let at_once = older.agent_results_in(&workspace.0).unwrap_err();
        let message = |error: &crate::WorkflowError| match error {
            crate::WorkflowError::Persistence(message) => message
                .replace(&described.agent_runs[0].task_file, "TASK FILE")
                .replace(&older.agent_runs[0].task_file, "TASK FILE"),
            error => panic!("{error}"),
        };
        assert_eq!(message(&on_demand), message(&at_once));
        assert_eq!(
            on_demand.to_string().matches("Persistence error").count(),
            1
        );
    }

    /// Set for [`the_child_of_the_directory_test`].
    const DIRECTORY_CHILD: &str = "ARKO_TEST_DIRECTORY_CHILD";

    /// A task read on demand is read from the run directory the checkpoint
    /// was read from, also after the process changed its working directory.
    /// The process here is a child of the test, which changes its directory.
    #[test]
    fn a_task_read_on_demand_is_read_from_the_run_directory_given() {
        let module = module_path!().split_once("::").unwrap().1;
        let test = format!("{module}::the_child_of_the_directory_test");
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test.as_str(), "--exact", "--nocapture", "--test-threads=1"])
            .env(DIRECTORY_CHILD, "1")
            .output()
            .unwrap();
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{report}");
        assert!(
            report.contains("1 passed"),
            "the child test didn't run: {report}"
        );
    }

    /// The part of the test above that changes the working directory, in a
    /// process of its own. Alone it does nothing.
    #[test]
    fn the_child_of_the_directory_test() {
        if std::env::var_os(DIRECTORY_CHILD).is_none() {
            return;
        }
        let workspace = crate::test_support::Workspace::new();
        let state = one_completed_run(&workspace.0, true);
        let start = std::env::current_dir().unwrap();
        std::env::set_current_dir(workspace.0.parent().unwrap()).unwrap();
        let relative = std::path::Path::new(workspace.0.file_name().unwrap());
        let results = state.agent_results_in(relative).unwrap();
        std::env::set_current_dir(&workspace.0).unwrap();
        let read = results[0]
            .task()
            .map(|task| task.initial_task_prompt.clone());
        std::env::set_current_dir(start).unwrap();
        assert_eq!(read.unwrap(), "prompt");
    }
}
