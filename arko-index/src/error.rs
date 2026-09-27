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
    /// A model or embedding provider failure. `source` keeps the underlying
    /// `arko_ai::Error` (status, code, retry class) when there is one.
    #[error("provider: {message}")]
    Provider {
        message: String,
        transient: bool,
        retry_after: Option<Duration>,
        #[source]
        source: Option<Box<arko_ai::Error>>,
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
            source: Some(Box::new(e)),
        }
    }
}
impl Error {
    /// Recovery class of a provider failure: the source error's class, or
    /// transient/permanent from `transient` for backends without one. `None`
    /// for non-provider errors.
    pub fn retry_class(&self) -> Option<arko_ai::RetryClass> {
        match self {
            Self::Provider {
                source: Some(source),
                ..
            } => Some(source.retry_class()),
            Self::Provider { transient, .. } => Some(if *transient {
                arko_ai::RetryClass::Transient
            } else {
                arko_ai::RetryClass::Permanent
            }),
            _ => None,
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
