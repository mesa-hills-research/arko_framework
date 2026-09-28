/// Serializable workflow state for checkpointing.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::budget::CostLedger;
use crate::freshness::FreshnessSnapshot;
use crate::memory::MemoryTier;
use crate::messaging::AgentMessage;
use crate::workflow::result::{AgentResult, AgentRunStatus};

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
    /// [`agent_runs`](Self::agent_runs) held them; empty in newer ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_results: Vec<AgentResult>,
    /// Finalized agent results, each referring to its task file instead of
    /// holding the task (the conversation is stored once, in the task file).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_runs: Vec<AgentRunRecord>,
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
}

/// A finalized agent result in a checkpoint, with its task in the run
/// directory's task file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentRunRecord {
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    pub status: AgentRunStatus,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The task file, relative to the run directory:
    /// `agents/<agent_run_id>.json`.
    pub task_file: String,
}

impl AgentRunRecord {
    pub(crate) fn of(result: &AgentResult) -> Self {
        Self {
            agent_run_id: result.agent_run_id,
            agent_name: result.agent_name.clone(),
            status: result.status.clone(),
            summary: result.summary.clone(),
            error: result.error.clone(),
            task_file: super::task_file(result.agent_run_id),
        }
    }
}

impl WorkflowState {
    /// The checkpoint's agent results, with their tasks: those it holds (a
    /// checkpoint written before task files were referenced), then those
    /// whose tasks are in the task files of `run_dir`, the directory
    /// holding the checkpoint.
    pub fn agent_results_in(
        &self,
        run_dir: &std::path::Path,
    ) -> Result<Vec<AgentResult>, crate::WorkflowError> {
        let mut results = self.agent_results.clone();
        for record in &self.agent_runs {
            let path = run_dir.join(&record.task_file);
            let task = std::fs::read_to_string(&path)
                .map_err(crate::WorkflowError::from)
                .and_then(|json| Ok(serde_json::from_str::<arko_agents::Task>(&json)?))
                .map_err(|error| {
                    crate::WorkflowError::Persistence(format!(
                        "the checkpoint's task file {} for agent '{}' can't be loaded: {error}",
                        path.display(),
                        record.agent_name
                    ))
                })?;
            results.push(AgentResult {
                agent_run_id: record.agent_run_id,
                agent_name: record.agent_name.clone(),
                status: record.status.clone(),
                summary: record.summary.clone(),
                error: record.error.clone(),
                task,
            });
        }
        Ok(results)
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
            verification_records: Vec::new(),
            distillation_records: Vec::new(),
            completed_agents: Vec::new(),
            cancellation: None,
            agent_registry: None,
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

    /// An agent cancelled before cancellation had its own task state was
    /// saved as failed with the runtime's cancellation reason; it loads as
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
            result.task.current_state,
            arko_agents::TaskState::Cancelled { .. }
        ));
        assert!(state.cancellation.is_none());
    }
}
