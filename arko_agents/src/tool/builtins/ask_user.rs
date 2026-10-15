use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

pub const ASK_USER_TOOL_NAME: &str = "ask_user";
pub const ASK_USER_QUESTIONS_PARAM_NAME: &str = "questions";

#[derive(Debug, Clone)]
pub struct AskUserTool;

impl AskUserTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AskUserTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for AskUserTool {
    /// Asking again only returns the questions: a call running when its
    /// task stopped asks them again.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: ASK_USER_TOOL_NAME.to_string(),
            description: "Asks the user and waits for their answer. Ask only what you can't \
                reasonably decide yourself. Put several questions in one call, numbered."
                .to_string(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
                description: "The questions: what you need to know and why, with example \
                        answers or options when they help."
                    .to_string(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let questions = params
            .get(ASK_USER_QUESTIONS_PARAM_NAME)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
            })?;

        if questions.trim().is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
                message: "Questions cannot be empty.".to_string(),
            });
        }

        Ok(questions.to_string())
    }
}
