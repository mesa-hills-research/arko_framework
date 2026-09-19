use std::time::Duration;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("blocked: {0}")]
    Blocked(String),
    #[error("paused")]
    Paused,
    #[error("cancelled")]
    Cancelled,
    #[error("stale reference: {0}")]
    Stale(String),
    #[error("provider: {message}")]
    Provider {
        message: String,
        transient: bool,
        retry_after: Option<Duration>,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("worker: {0}")]
    Worker(String),
}
pub type Result<T> = std::result::Result<T, Error>;
impl From<arko_ai::Error> for Error {
    fn from(e: arko_ai::Error) -> Self {
        Self::Provider {
            transient: e.is_retryable(),
            retry_after: e.retry_after(),
            message: e.to_string(),
        }
    }
}
impl From<tokio::task::JoinError> for Error {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Worker(e.to_string())
    }
}
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
