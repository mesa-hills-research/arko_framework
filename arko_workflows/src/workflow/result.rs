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
    /// The task ended `TaskState::Cancelled`: stopped with the workflow run
    /// (cancelled, or aborted by a hook), or by the workflow (over budget).
    Cancelled,
    /// Suspended at an `ask_user` question; `Workflow::continue_agent`
    /// answers it.
    WaitingForInput,
    /// Suspended awaiting a tool approval or error-recovery decision the
    /// workflow's policies leave to the caller (see
    /// [`AgentPause`](crate::policy::AgentPause));
    /// `Workflow::resume_agent` continues it.
    Paused,
}

impl AgentRunStatus {
    /// A pause (`WaitingForInput` or `Paused`): the run can go on.
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
    /// The result a run whose task ended as `task` records.
    pub(crate) fn of_task(agent_name: String, task: arko_agents::Task) -> Self {
        use arko_agents::TaskState;
        let status = AgentRunStatus::of_state(&task.current_state);
        let (summary, error) = match &task.current_state {
            TaskState::Completed { summary } => (summary.clone(), None),
            TaskState::Cancelled { reason }
            | TaskState::Failed { reason }
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
            task,
        }
    }
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
    /// Set when the run was stopped (cancelled, or aborted by a hook).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<crate::workflow::cancel::Cancellation>,
}
