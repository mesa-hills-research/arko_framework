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

    /// The call ran longer than the host's time limit and was stopped
    /// ([`AgentToolConfig::tool_timeout_seconds`](crate::agent_config::AgentToolConfig::tool_timeout_seconds)).
    #[error(
        "The tool did not finish within the host's time limit of {} and was stopped; anything it had started may be incomplete.",
        describe_limit(*limit)
    )]
    TimedOut { limit: std::time::Duration },

    #[error("I/O error during tool execution for tool '{tool_name}': {source}")]
    Io {
        tool_name: String,
        #[source]
        source: std::io::Error,
    },
}

/// `20 minutes`, `1 hour`, `90 seconds`.
fn describe_limit(limit: std::time::Duration) -> String {
    let seconds = limit.as_secs();
    let (count, unit) = if seconds >= 3600 && seconds.is_multiple_of(3600) {
        (seconds / 3600, "hour")
    } else if seconds >= 60 && seconds.is_multiple_of(60) {
        (seconds / 60, "minute")
    } else {
        (seconds, "second")
    };
    if count == 1 {
        format!("1 {unit}")
    } else {
        format!("{count} {unit}s")
    }
}
