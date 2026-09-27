/// Serializable workflow state for checkpointing.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::budget::CostLedger;
use crate::freshness::FreshnessSnapshot;
use crate::memory::MemoryTier;
use crate::messaging::AgentMessage;
use crate::workflow::result::AgentResult;

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
    #[serde(default)]
    pub agent_results: Vec<AgentResult>,
    #[serde(default)]
    pub verification_records: Vec<crate::verification::VerificationRecord>,
    #[serde(default)]
    pub distillation_records: Vec<crate::distillation::DistillationRecord>,
    #[serde(default)]
    pub completed_agents: Vec<SerializableAgentResult>,
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
            verification_records: Vec::new(),
            distillation_records: Vec::new(),
            completed_agents: Vec::new(),
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: WorkflowState = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.workflow_run_id, run_id);
        assert!(restored.freshness_snapshot.is_some());
    }

    /// A checkpoint from an early version, with fields since removed (the
    /// OAuth exemption set, MESI file tracking settings), still loads.
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

        let config: crate::workflow::config::WorkflowConfig =
            serde_json::from_value(serde_json::json!({
                "name": "legacy",
                "workspace": "/workspace",
                "mesi_config": {"block_on_conflict": false}
            }))
            .unwrap();
        assert_eq!(config.name, "legacy");
    }
}
