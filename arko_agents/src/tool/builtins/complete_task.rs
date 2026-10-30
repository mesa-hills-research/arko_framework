use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

pub const COMPLETE_TASK_TOOL_NAME: &str = "complete_task";
pub const COMPLETE_TASK_SUMMARY_PARAM_NAME: &str = "summary";

#[derive(Debug, Clone)]
pub struct CompleteTaskTool;

impl CompleteTaskTool {
    pub fn new() -> Self {
        Self {}
    }

    pub fn default_summary_description() -> String {
        "The task's result, in full: what you did and found, and anything left undone.".to_string()
    }
}

impl Default for CompleteTaskTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CompleteTaskTool {
    /// Completing again completes the task once (its host's checks, such as
    /// a workflow's completion review, run again).
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: COMPLETE_TASK_TOOL_NAME.to_string(),
            description: "Ends the task. Call it when the task is done, or when you can't go further. The summary is the task's result, so write it in full.".to_string(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: COMPLETE_TASK_SUMMARY_PARAM_NAME.to_string(),
                description: Self::default_summary_description(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let summary = params
            .get(COMPLETE_TASK_SUMMARY_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: COMPLETE_TASK_SUMMARY_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: COMPLETE_TASK_SUMMARY_PARAM_NAME.to_string(),
                message: "Summary must be a string.".to_string(),
            })?
            .to_string();

        Ok(summary)
    }
}
