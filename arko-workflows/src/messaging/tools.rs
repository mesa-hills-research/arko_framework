//! Send-message tool for inter-agent communication.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

use crate::context::SharedContext;

pub const SEND_MESSAGE_TOOL_NAME: &str = "send_message";

/// Tool allowing agents to send messages to other agents in the workflow.
/// It lists, and accepts, only the recipients the workflow's messaging
/// policy permits the agent.
#[derive(Debug)]
pub struct SendMessageTool {
    context: Arc<SharedContext>,
    agent_name: String,
}

impl SendMessageTool {
    pub fn new(context: Arc<SharedContext>, agent_name: String) -> Self {
        Self {
            context,
            agent_name,
        }
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: SEND_MESSAGE_TOOL_NAME.to_string(),
            description: format!(
                "Send a message to another agent in the workflow. Active agents receive it at their next safe turn boundary. Valid recipients: {}.",
                self.context.message_recipients(&self.agent_name).join(", ")
            ),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "to".to_string(),
                    description: "The name of the recipient agent.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "attachments_json".to_string(),
                    description: "Optional JSON array of typed context references: file, memory_tier, transcript, or text.".to_string(),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "content".to_string(),
                    description: "The message content.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            supports_batched_calls: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        task_context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let to = params.get("to").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::MissingParameter {
                param_name: "to".to_string(),
            }
        })?;
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "content".to_string(),
            })?;
        let attachments = params
            .get("attachments_json")
            .and_then(Value::as_str)
            .map(serde_json::from_str::<Vec<crate::messaging::MessageAttachment>>)
            .transpose()
            .map_err(|error| ToolError::ParameterValidation {
                param_name: "attachments_json".to_string(),
                message: error.to_string(),
            })?
            .unwrap_or_default();

        let recipients = self.context.message_recipients(&self.agent_name);
        if !recipients.iter().any(|recipient| recipient == to) {
            return Err(ToolError::ParameterValidation {
                param_name: "to".to_string(),
                message: format!(
                    "'{to}' is not a recipient you may message; valid recipients: {}",
                    recipients.join(", ")
                ),
            });
        }

        let message = self
            .context
            .messages
            .send(
                &self.agent_name,
                Some(task_context.task_id),
                to,
                content.to_string(),
                attachments,
            )
            .await
            .map_err(ToolError::ExecutionFailed)?;

        crate::workflow::engine::fire_hooks_on_context(
            &self.context,
            crate::hooks::HookPoint::OnAgentMessage,
            |hook| {
                hook.agent_name = Some(self.agent_name.clone());
                hook.agent_run_id = Some(task_context.task_id);
                hook.metadata
                    .insert("message_id".to_string(), serde_json::json!(message.id));
                hook.metadata
                    .insert("recipient".to_string(), serde_json::json!(to));
                hook.metadata
                    .insert("content".to_string(), serde_json::json!(content));
            },
        )
        .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;

        Ok(format!("Message {} sent to agent '{}'.", message.id, to))
    }
}
