use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

pub const LIST_FILES_TOOL_NAME: &str = "list_files";
const DIRECTORY_PARAM_NAME: &str = "directory";

#[derive(Debug, Clone)]
pub struct ListFilesTool;

impl ListFilesTool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ListFilesTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ListFilesTool {
    /// Listing has no side effects. A call running when its task stopped
    /// lists again.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn allow_result_interception(&self) -> bool {
        false
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LIST_FILES_TOOL_NAME.to_string(),
            description: "Lists the files and directories in a directory, at every depth, as \
                a tree. What .gitignore files leave out is left out.".to_string(),
            output_based_example: Some(
                "<list_files>\n<directory>src</directory>\n</list_files>".to_string()
            ),
            parameters: vec![ToolParameterDefinition {
                name: DIRECTORY_PARAM_NAME.to_string(),
                description: "The directory to list (relative to the workspace root, e.g., 'src' or 'src/core'). Use '.' for the workspace root.".to_string(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: true,
                flag: None,
                default_value: None,
            }],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::BuiltIn,
            read_only: true,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let directory_path_str = params
            .get(DIRECTORY_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: DIRECTORY_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: DIRECTORY_PARAM_NAME.to_string(),
                message: format!("Parameter '{}' must be a string.", DIRECTORY_PARAM_NAME),
            })?
            .trim()
            .to_string();

        if directory_path_str.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: DIRECTORY_PARAM_NAME.to_string(),
                message: "Directory path cannot be empty.".to_string(),
            });
        }

        let target_path = super::resolve_path_parameter(
            DIRECTORY_PARAM_NAME,
            &directory_path_str,
            context.current_working_directory,
        )?;

        log::debug!(
            "ListFilesTool: Resolved directory '{}' to target path: {:?}",
            directory_path_str,
            target_path
        );

        if !target_path.exists() {
            return Err(ToolError::ExecutionFailed(format!(
                "Directory '{}' does not exist.",
                directory_path_str
            )));
        }

        if !target_path.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "Path '{}' is not a directory.",
                directory_path_str
            )));
        }

        let canonical_target = target_path.canonicalize().map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to canonicalize directory path: {}", e))
        })?;

        let file_tree_output =
            crate::utils::get_file_tree(&canonical_target, None).map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to generate file tree: {}", e))
            })?;

        Ok(file_tree_output)
    }
}
