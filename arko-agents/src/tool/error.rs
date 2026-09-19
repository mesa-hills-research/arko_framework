use thiserror::Error;

/// Errors that can occur during tool execution or configuration.
#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Parameter validation failed for parameter '{param_name}': {message}")]
    ParameterValidation { param_name: String, message: String },

    #[error("Missing required parameter: '{param_name}'")]
    MissingParameter { param_name: String },

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Configuration error for tool '{tool_name}': {message}")]
    Configuration { tool_name: String, message: String },

    #[error(
        "The path for parameter '{param_name}' must be within the workspace '{workspace_path}'."
    )]
    PathAccessRejected {
        param_name: String,
        workspace_path: String,
    },

    #[error(
        "Shell command execution failed for tool '{tool_name}'. Exit code: {exit_code:?}. Stdout: {stdout}. Stderr: {stderr}"
    )]
    ShellCommandFailed {
        tool_name: String,
        exit_code: Option<i32>,
        stdout: String,
        stderr: String,
    },

    #[error("Shell command template error for tool '{tool_name}': {message}")]
    ShellCommandTemplate { tool_name: String, message: String },

    #[error("I/O error during tool execution for tool '{tool_name}': {source}")]
    Io {
        tool_name: String,
        #[source]
        source: std::io::Error,
    },
}
