use std::time::Duration;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("blocked: {0}")]
    Blocked(String),
    #[error("paused")]
    Paused,
    /// A plan usage limit stopped the job; the message says when the limit
    /// resets, when the provider said. The job is paused.
    #[error("{0}")]
    UsageLimit(String),
    #[error("cancelled")]
    Cancelled,
    #[error("stale reference: {0}")]
    Stale(String),
    /// A model or embedding provider failure. `source` keeps the underlying
    /// `arko_ai::Error` (status, code, retry class) when there is one.
    /// `usage` is the usage of a response the provider returned (and may
    /// have billed) that the backend then rejected, such as a refused or
    /// truncated reply; it is counted in [`JobUsage`](crate::JobUsage).
    /// Failures without a response have none.
    #[error("provider: {message}")]
    Provider {
        message: String,
        transient: bool,
        retry_after: Option<Duration>,
        #[source]
        source: Option<Box<arko_ai::Error>>,
        usage: Option<Box<arko_ai::UsageInfo>>,
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
            usage: None,
        }
    }
}
impl Error {
    /// Attaches the usage of the rejected response this provider failure is
    /// about (see [`Error::Provider`]). Other errors are returned unchanged.
    pub fn with_usage(mut self, response_usage: arko_ai::UsageInfo) -> Self {
        if let Self::Provider { usage, .. } = &mut self {
            *usage = Some(Box::new(response_usage));
        }
        self
    }
    /// The usage of the rejected response a provider failure is about.
    pub fn usage(&self) -> Option<&arko_ai::UsageInfo> {
        match self {
            Self::Provider { usage, .. } => usage.as_deref(),
            _ => None,
        }
    }
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
/// The [`Error::UsageLimit`] for a provider failure that is a plan usage
/// limit, naming when it resets; `None` for other errors.
pub(crate) fn usage_limit(error: &Error) -> Option<Error> {
    let Error::Provider { retry_after, .. } = error else {
        return None;
    };
    if error.retry_class() != Some(arko_ai::RetryClass::UsageLimit) {
        return None;
    }
    let when = match retry_after.and_then(|delay| {
        chrono::Duration::from_std(delay)
            .ok()
            .and_then(|delta| chrono::Utc::now().checked_add_signed(delta))
            .map(|at| (delay, at))
    }) {
        Some((delay, at)) => format!(
            "resets at {} (in about {})",
            at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            describe_duration(delay)
        ),
        None => "the provider did not say when it resets".into(),
    };
    Some(Error::UsageLimit(format!(
        "usage limit reached; {when}. Resume the job after the reset. {error}"
    )))
}
/// `3h 12m`, `5m 3s`, `2d 4h`.
fn describe_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (days, hours, minutes, rest) = (
        seconds / 86_400,
        seconds % 86_400 / 3_600,
        seconds % 3_600 / 60,
        seconds % 60,
    );
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {rest}s")
    } else {
        format!("{rest}s")
    }
}
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
