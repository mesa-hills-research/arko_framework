/// Result types returned after workflow and agent execution.
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
    Cancelled,
    WaitingForInput,
}

/// Result of a single agent run within a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    pub status: AgentRunStatus,
    pub summary: String,
    pub error: Option<String>,
    pub task: arko_agents::Task,
}

impl AgentResult {
    /// Returns the total cost of this agent run.
    pub fn total_cost(&self) -> f64 {
        self.task.total_usage_cost()
    }

    /// Returns the total input tokens consumed.
    pub fn total_input_tokens(&self) -> usize {
        self.task.total_input_tokens()
    }

    /// Returns the total output tokens generated.
    pub fn total_output_tokens(&self) -> usize {
        self.task.total_output_tokens()
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
}
