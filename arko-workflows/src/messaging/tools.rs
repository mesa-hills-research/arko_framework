//! Tools for inter-agent communication: `send_message` and
//! `list_recipients`, and the start notices agents receive.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

use crate::context::SharedContext;
use crate::messaging::registry::AgentStatus;

pub const SEND_MESSAGE_TOOL_NAME: &str = "send_message";
pub const LIST_RECIPIENTS_TOOL_NAME: &str = "list_recipients";

/// Tool allowing agents to send messages to other agents in the workflow.
/// It accepts only recipients the workflow's messaging policy permits the
/// agent that are running or waiting. Its definition names no agent, so it
/// never changes during a task.
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
            description: "Send a message to another workflow agent that is running (or waiting); it receives the message at its next safe turn boundary. list_recipients reports the agents you can message and their current status; workflow context updates tell you when one starts.".to_string(),
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

        // Only permitted recipients are named, never other agents.
        let recipients = self.context.message_recipients(&self.agent_name);
        if !recipients.iter().any(|recipient| recipient == to) {
            return Err(ToolError::ParameterValidation {
                param_name: "to".to_string(),
                message: format!(
                    "'{to}' is not an agent you can message. The agents you can message: {}.",
                    recipients.join(", ")
                ),
            });
        }
        let status = self.context.recipient_status(to);
        match &status {
            AgentStatus::NotStarted => {
                return Err(ToolError::ExecutionFailed(format!(
                    "Agent '{to}' has not started, so it can't receive messages. A workflow context update will tell you when it starts."
                )));
            }
            AgentStatus::Finished(_) => {
                return Err(ToolError::ExecutionFailed(format!(
                    "Agent '{to}' is {}; it no longer receives messages.",
                    status.describe()
                )));
            }
            AgentStatus::Running | AgentStatus::Waiting(_) => {}
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

        Ok(match status {
            AgentStatus::Waiting(_) => format!(
                "Message {} sent to agent '{to}', which is {}; it receives the message when it continues.",
                message.id,
                status.describe()
            ),
            _ => format!("Message {} sent to agent '{to}'.", message.id),
        })
    }
}

/// Tool reporting the agents an agent may message, with their roles and
/// current statuses. Offered with `send_message`.
#[derive(Debug)]
pub struct ListRecipientsTool {
    context: Arc<SharedContext>,
    agent_name: String,
}

impl ListRecipientsTool {
    pub fn new(context: Arc<SharedContext>, agent_name: String) -> Self {
        Self {
            context,
            agent_name,
        }
    }
}

#[async_trait]
impl Tool for ListRecipientsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LIST_RECIPIENTS_TOOL_NAME.to_string(),
            description: "List the agents you can message with send_message, with each one's role and current status: not_started, running, waiting (with the reason and, when known, until when), or finished (with the outcome). Running and waiting agents receive messages.".to_string(),
            output_based_example: None,
            parameters: vec![],
            supports_batched_calls: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _task_context: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let statuses = self.context.recipient_statuses(&self.agent_name);
        if statuses.is_empty() {
            return Ok("There are no agents you can message.".to_string());
        }
        Ok(render_recipients(&self.context, &statuses))
    }
}

/// One line per agent: its name, role, and status.
pub(crate) fn render_recipients(
    context: &SharedContext,
    statuses: &[(String, AgentStatus)],
) -> String {
    statuses
        .iter()
        .map(|(agent, status)| {
            format!(
                "- `{agent}` ({}): {}",
                role(context, agent),
                status.describe()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn role<'a>(context: &'a SharedContext, agent: &str) -> &'a str {
    context
        .agent_configs
        .get(agent)
        .map_or("", |config| config.role_description.as_str())
}

/// The start notices a running agent receives at each safe turn boundary:
/// one line for every agent it may message that started after the registry
/// version it has seen.
pub(crate) struct StartNotices {
    pub context: Arc<SharedContext>,
    pub agent_name: String,
}

impl arko_agents::TurnContextSource for StartNotices {
    fn turn_context(&self, task_id: uuid::Uuid) -> Option<String> {
        let recipients = self.context.message_recipients(&self.agent_name);
        if recipients.is_empty() {
            return None;
        }
        let started = self
            .context
            .registry
            .started_since_seen(task_id, &recipients);
        (!started.is_empty()).then(|| {
            started
                .iter()
                .map(|agent| start_notice(&self.context, agent))
                .collect::<Vec<_>>()
                .join("\n")
        })
    }
}

/// The notice that `agent` started.
pub(crate) fn start_notice(context: &SharedContext, agent: &str) -> String {
    format!(
        "Agent `{agent}` (role: {}) is now running. You can message it with send_message.",
        role(context, agent)
    )
}
