use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

pub const MESSAGE_USER_TOOL_NAME: &str = "message_user";
pub const MESSAGE_USER_MESSAGE_PARAM_NAME: &str = "message";

/// The result the model gets for a `message_user` call.
pub const MESSAGE_USER_RESULT: &str = "Your message was sent to the user. Carry on with your work.";

/// Shows the user a message without ending the agent's turn: `ask_user`
/// without the wait. The runtime sends the message to the host as
/// [`AgentEvent::MessageToUser`](crate::AgentEvent::MessageToUser) and the
/// task carries on. Its state doesn't change.
///
/// A host's own tool registered under the same name replaces this one: it
/// runs as any other tool, and the runtime sends no `MessageToUser` for its
/// calls.
#[derive(Debug, Clone)]
pub struct MessageUserTool;

impl MessageUserTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MessageUserTool {
    fn default() -> Self {
        Self::new()
    }
}

/// The message a finished call of `tool`, registered under `name`, sends
/// the user, given the call's `message` argument: the argument's text when
/// the tool is the built-in `message_user` (the name is `message_user` and
/// the tool's type is [`ToolTypeDetails::BuiltIn`]), and `None` for any
/// other tool, a host's tool registered under the name included.
pub(crate) fn message_to_user(
    name: &str,
    tool: &dyn Tool,
    message: Option<&Value>,
) -> Option<String> {
    if name != MESSAGE_USER_TOOL_NAME
        || tool.definition().tool_type_details != ToolTypeDetails::BuiltIn
    {
        return None;
    }
    message.and_then(Value::as_str).map(str::to_string)
}

#[async_trait]
impl Tool for MessageUserTool {
    /// Sending again only shows the message again: a call running when its
    /// task stopped is run again.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: MESSAGE_USER_TOOL_NAME.to_string(),
            description: "Show the user a message now, without ending your turn or waiting for \
                a reply: the answer to something the user asked while you work (a question, a \
                request for progress), or something they must know before you finish. Then \
                carry on with your work. Your final result still goes in complete_task."
                .to_string(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: MESSAGE_USER_MESSAGE_PARAM_NAME.to_string(),
                description: "The message to show the user.".to_string(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let message = params
            .get(MESSAGE_USER_MESSAGE_PARAM_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: MESSAGE_USER_MESSAGE_PARAM_NAME.to_string(),
            })?;

        if message.trim().is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: MESSAGE_USER_MESSAGE_PARAM_NAME.to_string(),
                message: "The message cannot be empty.".to_string(),
            });
        }

        Ok(MESSAGE_USER_RESULT.to_string())
    }
}
