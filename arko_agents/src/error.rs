use std::path::PathBuf;
use thiserror::Error;

use crate::parsing::ToolAssemblyError;
use crate::tool::ToolError;

#[derive(Error, Debug)]
pub enum ArkoAgentsError {
    #[error("LLM Error: {0}")]
    LLM(#[from] arko_ai::Error),

    #[error("Tool Error: {0}")]
    Tool(#[from] ToolError),

    #[error("I/O Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Configuration Error: {0}")]
    Configuration(String),

    #[error("Parsing Error: {0}")]
    Parsing(String),

    #[error("Tool Assembly Error: {0}")]
    AssemblyError(#[from] ToolAssemblyError),

    #[error("Runtime Error: {0}")]
    Runtime(String),

    #[error("Task Execution Error: {0}")]
    TaskExecution(String),

    #[error("Tool not found in registry: {0}")]
    ToolNotFound(String),

    #[error("Invalid Task State Transition: {0}")]
    InvalidTaskState(String),

    #[error("Channel send error: {0}")]
    ChannelSend(String),

    #[error("Channel receive error: {0}")]
    ChannelReceive(String),

    #[error("File operation failed for path {path:?}: {source}")]
    FileOperation {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("UUID generation error: {0}")]
    UuidError(#[from] uuid::Error),

    #[error("WalkDir error: {0}")]
    WalkDir(#[from] walkdir::Error),

    #[error("Ignore Build error: {0}")]
    IgnoreError(#[from] ignore::Error),

    #[error("Tokio JoinError: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    #[error("Builder error for {struct_name}: {message}")]
    BuilderError {
        struct_name: String,
        message: String,
    },

    #[error("Skill error: {0}")]
    Skill(#[from] crate::skill::SkillLoadError),
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for ArkoAgentsError {
    fn from(e: tokio::sync::mpsc::error::SendError<T>) -> Self {
        ArkoAgentsError::ChannelSend(e.to_string())
    }
}
