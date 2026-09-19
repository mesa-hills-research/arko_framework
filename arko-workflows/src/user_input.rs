//! Optional human-in-the-loop routing for an agent's `ask_user` tool.

use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct UserInputRequest {
    pub workflow_run_id: uuid::Uuid,
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    pub questions: String,
}

#[async_trait]
pub trait UserInputHandler: Send + Sync {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String>;
}
