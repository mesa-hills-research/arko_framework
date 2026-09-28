use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

pub const WRITE_FILE_TOOL_NAME: &str = "write_file";
const PATH_PARAM_NAME: &str = "path";
const CONTENT_PARAM_NAME: &str = "content";

#[derive(Debug, Clone)]
pub struct WriteFileTool;

impl WriteFileTool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: WRITE_FILE_TOOL_NAME.to_string(),
            description: "Create a new file in the workspace or overwrite an existing one. \
                NOTE - if you are using this tool to update existing code, you MUST output all \
                unchanged content as well, as the entire value of the content parameter will \
                become the new content of the file.".to_string(),
            output_based_example: Some(
                "<write_file>\n\
                <path>src/config.rs</path>\n\
                <content>pub const VERSION: &str = \"1.0.0\";</content>\n\
                </write_file>".to_string()
            ),
            parameters: vec![
                ToolParameterDefinition {
                    name: PATH_PARAM_NAME.to_string(),
                    description: "The path to the file to write (relative to the current working directory, e.g. 'src/main.rs').".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: true,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: CONTENT_PARAM_NAME.to_string(),
                    description: "The content to write to the file.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let file_path_str = params
            .get(PATH_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: PATH_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: format!("Parameter '{}' must be a string.", PATH_PARAM_NAME),
            })?
            .trim()
            .to_string();

        let content = params
            .get(CONTENT_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: CONTENT_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: CONTENT_PARAM_NAME.to_string(),
                message: format!("Parameter '{}' must be a string.", CONTENT_PARAM_NAME),
            })?
            .to_string();

        if file_path_str.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "File path cannot be empty.".to_string(),
            });
        }

        let absolute_path = super::resolve_path_parameter(
            PATH_PARAM_NAME,
            &file_path_str,
            context.current_working_directory,
        )?;

        log::debug!(
            "WriteFileTool: Attempting to write to file at absolute path: {:?}",
            absolute_path
        );

        if let Some(parent_dir) = absolute_path.parent()
            && !parent_dir.exists()
        {
            log::debug!(
                "WriteFileTool: Parent directory {:?} does not exist, attempting to create.",
                parent_dir
            );
            tokio::fs::create_dir_all(parent_dir)
                .await
                .map_err(|e| ToolError::Io {
                    tool_name: self.name(),
                    source: e,
                })?;
        }

        let content_bytes = content.as_bytes();
        tokio::fs::write(&absolute_path, content_bytes)
            .await
            .map_err(|e| ToolError::Io {
                tool_name: self.name(),
                source: e,
            })?;

        Ok(format!(
            "Successfully wrote {} bytes to file '{}'",
            content_bytes.len(),
            file_path_str
        ))
    }
}
