use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

pub const READ_FILE_TOOL_NAME: &str = "read_file";
const PATH_PARAM_NAME: &str = "path";

#[derive(Debug, Clone)]
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: READ_FILE_TOOL_NAME.to_string(),
            description: "Reads the content of a file from the workspace, prepending line numbers to each line (e.g., '1 | line content').".to_string(),
            output_based_example: Some(
                "<read_file>\n<path>src/main.rs</path>\n</read_file>".to_string()
            ),
            parameters: vec![ToolParameterDefinition {
                name: PATH_PARAM_NAME.to_string(),
                description: "The path to the file to read (relative to the current working directory, e.g. 'src/main.rs').".to_string(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: true,
                flag: None,
                default_value: None,
            }],
            supports_batched_calls: true,
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

        if file_path_str.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "File path cannot be empty.".to_string(),
            });
        }

        let absolute_path = context.current_working_directory.join(&file_path_str);

        log::debug!(
            "ReadFileTool: Attempting to read file at absolute path: {:?}",
            absolute_path
        );

        let bytes = tokio::fs::read(&absolute_path)
            .await
            .map_err(|e| ToolError::Io {
                tool_name: self.name(),
                source: e,
            })?;

        let content_bytes = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            bytes[3..].to_vec()
        } else {
            bytes
        };

        let content = String::from_utf8(content_bytes)
            .map_err(|_| ToolError::ExecutionFailed("File is not valid UTF-8".to_string()))?;
        let normalized = content.replace("\r\n", "\n");
        Ok(normalized
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{} | {}", i + 1, line))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}
