use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    /// A provider failure with machine-readable recovery information.
    #[error("Provider error (status {status:?}, code {code:?}): {message}")]
    ProviderError {
        message: String,
        status: Option<u16>,
        code: Option<String>,
        retry_after: Option<Duration>,
    },

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Provider not supported: {0}")]
    ProviderNotSupported(String),

    #[error("Model not supported: {0}")]
    ModelNotSupported(String),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    #[error("Missing API key for provider: {0}")]
    MissingApiKey(String),

    #[error("Base64 error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Input text exceeds model's token limit: {0}")]
    TokenLimitExceeded(String),

    #[error("Missing HuggingFace token for tokenizer: {0}")]
    MissingHuggingFaceToken(String),

    #[error("Batch size limit exceeded: {0}")]
    BatchSizeLimitExceeded(String),

    #[error("Invalid embedding dimension: {0}")]
    InvalidDimension(String),
}

impl Error {
    /// Explicit temporary capacity signals, excluding permanent status/code failures.
    pub fn is_capacity_error(&self) -> bool {
        self.is_retryable()
            && matches!(
                self,
                Self::ProviderError { code: Some(code), .. }
                    if matches!(code.as_str(), "server_is_overloaded" | "overloaded_error" | "service_unavailable_error")
            )
    }

    /// Whether another model/refresh request may recover without changing configuration.
    /// Unknown legacy API errors retain bounded retries for compatibility.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::ProviderError { status, code, .. } => {
                if matches!(
                    code.as_deref(),
                    Some(
                        "insufficient_quota"
                            | "billing_hard_limit_reached"
                            | "invalid_api_key"
                            | "invalid_grant"
                            | "invalid_client"
                            | "unauthorized_client"
                            | "authentication_error"
                            | "permission_error"
                            | "permission_denied"
                            | "invalid_request_error"
                            | "context_length_exceeded"
                            | "model_not_found"
                    )
                ) {
                    return false;
                }
                if let Some(status) = status {
                    return matches!(status, 408 | 409 | 429 | 500..=599);
                }
                matches!(
                    code.as_deref(),
                    Some(
                        "rate_limit_exceeded"
                            | "rate_limit_error"
                            | "server_error"
                            | "internal_error"
                            | "overloaded_error"
                            | "server_is_overloaded"
                            | "service_unavailable_error"
                            | "temporarily_unavailable"
                            | "RESOURCE_EXHAUSTED"
                            | "UNAVAILABLE"
                    )
                )
            }
            Self::HttpError(error) => error.status().map_or_else(
                || {
                    error.is_connect()
                        || error.is_timeout()
                        || error.is_body()
                        || error.is_request()
                },
                |status| matches!(status.as_u16(), 408 | 409 | 429 | 500..=599),
            ),
            Self::ApiError(_) => true,
            _ => false,
        }
    }

    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::ProviderError { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    pub(crate) fn provider_error(
        label: &str,
        status: Option<u16>,
        retry_after: Option<Duration>,
        body: &str,
    ) -> Self {
        let value = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
        let error = value.get("error").unwrap_or(&value);
        let code = error
            .get("code")
            .and_then(|v| v.as_str())
            .or_else(|| error.get("type").and_then(|v| v.as_str()))
            .or_else(|| error.get("status").and_then(|v| v.as_str()))
            .or_else(|| error.as_str())
            .map(str::to_owned);
        Self::ProviderError {
            message: format!("{label}: {body}"),
            status,
            code,
            retry_after,
        }
    }

    pub(crate) async fn from_response(label: &str, response: reqwest::Response) -> Self {
        let status = response.status().as_u16();
        let retry_after = retry_after_from_headers(response.headers());
        // Preserve the HTTP status even when reading the error body fails.
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable error body>".into());
        Self::provider_error(label, Some(status), retry_after, &body)
    }
}

pub(crate) fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value).ok().map(|deadline| {
        deadline
            .duration_since(std::time::SystemTime::now())
            .unwrap_or_default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_and_error_codes_distinguish_outages_from_configuration_errors() {
        for status in [408, 409, 429, 500, 502, 503, 504, 529] {
            assert!(
                Error::provider_error("test", Some(status), None, "{}").is_retryable(),
                "{status}"
            );
        }
        for status in [400, 401, 403, 404, 422] {
            assert!(
                !Error::provider_error("test", Some(status), None, "{}").is_retryable(),
                "{status}"
            );
        }
        assert!(
            !Error::provider_error(
                "test",
                Some(429),
                None,
                r#"{"error":{"code":"insufficient_quota"}}"#
            )
            .is_retryable()
        );
        assert!(
            !Error::provider_error("test", Some(503), None, r#"{"error":"invalid_grant"}"#)
                .is_retryable()
        );
        assert!(
            Error::provider_error(
                "test",
                None,
                None,
                r#"{"error":{"type":"overloaded_error"}}"#
            )
            .is_retryable()
        );
        assert!(
            !Error::provider_error("test", None, None, r#"{"code":"invalid_api_key"}"#)
                .is_retryable()
        );
        assert!(!Error::InvalidRequest("bad model".into()).is_retryable());
        assert!(!Error::MissingApiKey("test".into()).is_retryable());
        assert!(!Error::AuthenticationFailed("revoked".into()).is_retryable());
    }

    #[test]
    fn stream_capacity_errors_retry_without_an_http_status() {
        for body in [
            r#"{"error":{"code":"server_is_overloaded","message":"Our servers are currently overloaded. Please try again later.","param":null,"type":"service_unavailable_error"},"sequence_number":2,"type":"error"}"#,
            r#"{"error":{"type":"service_unavailable_error"}}"#,
        ] {
            let error = Error::provider_error("OpenAI stream error", None, None, body);
            assert!(error.is_retryable(), "{error}");
            assert!(error.is_capacity_error(), "{error}");
        }
        assert!(
            !Error::provider_error(
                "test",
                Some(401),
                None,
                r#"{"error":{"code":"server_is_overloaded"}}"#
            )
            .is_capacity_error()
        );
        assert!(
            !Error::provider_error(
                "test",
                Some(429),
                None,
                r#"{"error":{"code":"insufficient_quota"}}"#
            )
            .is_capacity_error()
        );
        // An explicit permanent code still takes precedence over the error type.
        assert!(
            !Error::provider_error(
                "OpenAI stream error",
                None,
                None,
                r#"{"error":{"code":"insufficient_quota","type":"service_unavailable_error"}}"#,
            )
            .is_retryable()
        );
    }

    #[test]
    fn retry_after_accepts_seconds_http_dates_and_rejects_invalid_values() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "60".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(&headers),
            Some(Duration::from_secs(60))
        );
        let deadline = std::time::SystemTime::now() + Duration::from_secs(60);
        headers.insert(
            reqwest::header::RETRY_AFTER,
            httpdate::fmt_http_date(deadline).parse().unwrap(),
        );
        let delay = retry_after_from_headers(&headers).unwrap();
        assert!(delay >= Duration::from_secs(58) && delay <= Duration::from_secs(60));
        for value in ["-1", "NaN", "nonsense"] {
            headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
            assert_eq!(retry_after_from_headers(&headers), None);
        }
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Thu, 01 Jan 1970 00:00:00 GMT".parse().unwrap(),
        );
        assert_eq!(retry_after_from_headers(&headers), Some(Duration::ZERO));
    }
}
