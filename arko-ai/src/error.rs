use crate::stop::{Refusal, StopReason};
use serde_json::Value;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// How a failed request may be recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Retrying the same request cannot succeed; a request, credential or
    /// configuration change is required.
    Permanent,
    /// A temporary failure (rate limit, timeout, transient server error) that
    /// a bounded retry may recover.
    Transient,
    /// An explicit provider capacity signal (overloaded / unavailable). Also
    /// retryable; callers may choose a longer wait.
    Capacity,
}

impl RetryClass {
    pub fn is_retryable(self) -> bool {
        !matches!(self, Self::Permanent)
    }
}

/// Codes meaning the request itself must change. They take precedence over a
/// retryable HTTP status. Includes OpenAI/Anthropic error codes and types,
/// OpenRouter `error.metadata.error_type` values
/// (https://openrouter.ai/docs/api_reference/errors-and-debugging) and Google
/// RPC status names.
const PERMANENT_CODES: &[&str] = &[
    "insufficient_quota",
    "billing_hard_limit_reached",
    "invalid_api_key",
    "invalid_grant",
    "invalid_client",
    "unauthorized_client",
    "authentication_error",
    "authentication",
    "permission_error",
    "permission_denied",
    "invalid_request_error",
    "invalid_request",
    "invalid_prompt",
    "context_length_exceeded",
    "max_tokens_exceeded",
    "token_limit_exceeded",
    "string_too_long",
    "model_not_found",
    "not_found",
    "precondition_failed",
    "payload_too_large",
    "unprocessable",
    "content_policy_violation",
    "refusal",
    "INVALID_ARGUMENT",
    "FAILED_PRECONDITION",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "NOT_FOUND",
];

/// Explicit temporary capacity signals (Gemini reports an overloaded model as
/// 503 `UNAVAILABLE`).
const CAPACITY_CODES: &[&str] = &[
    "server_is_overloaded",
    "overloaded_error",
    "service_unavailable_error",
    "provider_overloaded",
    "UNAVAILABLE",
];

/// Temporary failures recognised without an HTTP status (stream errors).
const TRANSIENT_CODES: &[&str] = &[
    "rate_limit_exceeded",
    "rate_limit_error",
    "server_error",
    "internal_error",
    "temporarily_unavailable",
    "provider_unavailable",
    "timeout",
    "server",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
];

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

    /// The model or a filter around it declined the request.
    #[error("Model refused the request: {0}")]
    Refusal(Box<Refusal>),

    /// The response stream ended before the provider reported completion.
    #[error("Response stream ended before completion: {0}")]
    StreamTruncated(String),

    /// A response expected to contain exactly one call to `expected` did not.
    #[error("Expected one call to `{expected}`; the response stopped with {stop_reason:?} and called {found:?}")]
    UnexpectedToolCalls {
        expected: String,
        stop_reason: StopReason,
        found: Vec<String>,
    },

    /// The model produced tool arguments that are not valid JSON.
    #[error("Tool `{name}` received arguments that are not valid JSON: {raw}")]
    InvalidToolArguments { name: String, raw: String },
}

impl Error {
    /// How this failure may be recovered. [`Error::is_retryable`] and
    /// [`Error::is_capacity_error`] are views of this classification.
    pub fn retry_class(&self) -> RetryClass {
        match self {
            Self::ProviderError {
                status,
                code,
                retry_after,
                ..
            } => {
                let code = code.as_deref();
                // OpenRouter reports an exhausted prepaid in-flight budget as a
                // 402 with Retry-After; any other payment failure is permanent.
                if code == Some("payment_required") {
                    return if retry_after.is_some() {
                        RetryClass::Transient
                    } else {
                        RetryClass::Permanent
                    };
                }
                if code.is_some_and(|code| PERMANENT_CODES.contains(&code)) {
                    return RetryClass::Permanent;
                }
                let capacity = code.is_some_and(|code| CAPACITY_CODES.contains(&code));
                if let Some(status) = status {
                    return if !matches!(status, 408 | 409 | 429 | 500..=599) {
                        RetryClass::Permanent
                    } else if capacity {
                        RetryClass::Capacity
                    } else {
                        RetryClass::Transient
                    };
                }
                if capacity {
                    RetryClass::Capacity
                } else if code.is_some_and(|code| TRANSIENT_CODES.contains(&code)) {
                    RetryClass::Transient
                } else {
                    RetryClass::Permanent
                }
            }
            Self::HttpError(error) => {
                let retryable = error.status().map_or_else(
                    || {
                        error.is_connect()
                            || error.is_timeout()
                            || error.is_body()
                            || error.is_request()
                    },
                    |status| matches!(status.as_u16(), 408 | 409 | 429 | 500..=599),
                );
                if retryable {
                    RetryClass::Transient
                } else {
                    RetryClass::Permanent
                }
            }
            // Unknown legacy API errors retain bounded retries for compatibility.
            Self::ApiError(_) | Self::StreamTruncated(_) | Self::InvalidToolArguments { .. } => {
                RetryClass::Transient
            }
            // A missing or wrong call may be resampled; a token or context
            // limit needs a configuration change.
            Self::UnexpectedToolCalls { stop_reason, .. } if !stop_reason.is_truncated() => {
                RetryClass::Transient
            }
            _ => RetryClass::Permanent,
        }
    }

    /// Explicit temporary capacity signals, excluding permanent status/code failures.
    pub fn is_capacity_error(&self) -> bool {
        self.retry_class() == RetryClass::Capacity
    }

    /// Whether another model/refresh request may recover without changing configuration.
    /// Unknown legacy API errors retain bounded retries for compatibility.
    pub fn is_retryable(&self) -> bool {
        self.retry_class().is_retryable()
    }

    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::ProviderError { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Builds a provider error from an error body. Understands string codes
    /// (`error.code`, `error.type`, `error.status`), OpenRouter's typed
    /// `error.metadata.error_type`, and numeric HTTP-style codes
    /// (`error.code: 429`, used by OpenRouter and Google), which stand in for
    /// the status when the HTTP status is absent or 2xx, as on stream errors.
    /// A delay hint in the body is used when no header delay is supplied.
    pub(crate) fn provider_error(
        label: &str,
        status: Option<u16>,
        retry_after: Option<Duration>,
        body: &str,
    ) -> Self {
        let value = serde_json::from_str::<Value>(body).unwrap_or_default();
        let error = value.get("error").unwrap_or(&value);
        let numeric_code = error
            .get("code")
            .and_then(Value::as_u64)
            .filter(|code| (100..=599).contains(code))
            .map(|code| code as u16);
        let code = error
            .pointer("/metadata/error_type")
            .and_then(Value::as_str)
            .or_else(|| error.get("code").and_then(Value::as_str))
            .or_else(|| error.get("type").and_then(Value::as_str))
            .or_else(|| error.get("status").and_then(Value::as_str))
            .or_else(|| error.as_str())
            .map(str::to_owned);
        let status = status
            .filter(|status| !(200..300).contains(status))
            .or(numeric_code);
        Self::ProviderError {
            message: format!("{label}: {body}"),
            status,
            code,
            retry_after: retry_after.or_else(|| retry_after_from_value(&value)),
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

/// Delay requested by response headers: `Retry-After` (seconds or an HTTP
/// date), else the longer of OpenAI-style `x-ratelimit-reset-requests` /
/// `x-ratelimit-reset-tokens` durations such as `1m2.5s`.
pub(crate) fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    retry_after_header(headers).or_else(|| {
        ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]
            .iter()
            .filter_map(|name| headers.get(*name))
            .filter_map(|value| value.to_str().ok())
            .filter_map(parse_relative_duration)
            .max()
    })
}

fn retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
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
            .duration_since(SystemTime::now())
            .unwrap_or_default()
    })
}

pub(crate) fn parse_relative_duration(value: &str) -> Option<Duration> {
    let mut remainder = value.trim();
    let mut total_millis = 0_f64;
    let mut parsed_component = false;

    while !remainder.is_empty() {
        let numeric_end = remainder
            .char_indices()
            .find_map(|(index, character)| {
                (!character.is_ascii_digit() && character != '.').then_some(index)
            })
            .unwrap_or(remainder.len());
        if numeric_end == 0 {
            return None;
        }
        let amount = remainder[..numeric_end].parse::<f64>().ok()?;
        remainder = &remainder[numeric_end..];
        let (multiplier, suffix_bytes) = if remainder.starts_with("ms") {
            (1_f64, 2)
        } else if remainder.starts_with('s') {
            (1_000_f64, 1)
        } else if remainder.starts_with('m') {
            (60_000_f64, 1)
        } else if remainder.starts_with('h') {
            (3_600_000_f64, 1)
        } else {
            return None;
        };
        total_millis += amount * multiplier;
        parsed_component = true;
        remainder = remainder[suffix_bytes..].trim_start();
    }

    parsed_component.then(|| Duration::from_millis(total_millis.ceil() as u64))
}

fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse::<f64>().ok()))
}

fn find_numeric_field(value: &Value, names: &[&str]) -> Option<f64> {
    match value {
        Value::Object(object) => {
            for name in names {
                if let Some(number) = object
                    .get(*name)
                    .and_then(value_as_f64)
                    .filter(|n| n.is_finite() && *n >= 0.0)
                {
                    return Some(number);
                }
            }
            object
                .values()
                .find_map(|child| find_numeric_field(child, names))
        }
        Value::Array(array) => array
            .iter()
            .find_map(|child| find_numeric_field(child, names)),
        _ => None,
    }
}

/// Delay hints anywhere in an error body: `retry_after_ms`/`retryAfterMs`,
/// `retry_after`/`retryAfter` (seconds), or a reset time
/// `resets_at`/`reset_at`/`resetsAt`/`resetAt` (epoch seconds or milliseconds).
pub(crate) fn retry_after_from_value(value: &Value) -> Option<Duration> {
    if let Some(milliseconds) =
        find_numeric_field(value, &["retry_after_ms", "retryAfterMs"])
    {
        return Some(Duration::from_millis(milliseconds.ceil() as u64));
    }
    if let Some(seconds) = find_numeric_field(value, &["retry_after", "retryAfter"])
    {
        return Some(Duration::from_millis((seconds * 1_000_f64).ceil() as u64));
    }
    let reset_at = find_numeric_field(
        value,
        &["resets_at", "reset_at", "resetsAt", "resetAt"],
    )?;
    let reset_millis = if reset_at >= 10_000_000_000_f64 {
        reset_at
    } else {
        reset_at * 1_000_f64
    };
    let now_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_millis() as f64;
    Some(Duration::from_millis(
        (reset_millis - now_millis).max(1_000_f64).ceil() as u64,
    ))
}

pub(crate) fn retry_after_from_body(body: &str) -> Option<Duration> {
    let value: Value = serde_json::from_str(body).ok()?;
    retry_after_from_value(&value)
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
    fn numeric_codes_stand_in_for_a_missing_or_successful_http_status() {
        // OpenRouter mid-stream error chunk: HTTP 200 already sent.
        let error = Error::provider_error(
            "stream",
            Some(200),
            None,
            r#"{"error":{"code":429,"message":"Rate limit exceeded"}}"#,
        );
        assert!(matches!(error, Error::ProviderError { status: Some(429), .. }));
        assert_eq!(error.retry_class(), RetryClass::Transient);
        let error = Error::provider_error("stream", None, None, r#"{"error":{"code":400}}"#);
        assert_eq!(error.retry_class(), RetryClass::Permanent);
        // A real HTTP status is not replaced by the body code.
        let error = Error::provider_error("http", Some(503), None, r#"{"error":{"code":400}}"#);
        assert!(matches!(error, Error::ProviderError { status: Some(503), .. }));
        // gRPC-style small codes are not statuses.
        let error = Error::provider_error("grpc", None, None, r#"{"error":{"code":8}}"#);
        assert!(matches!(error, Error::ProviderError { status: None, .. }));
    }

    #[test]
    fn openrouter_error_types_take_precedence_and_classify() {
        let body = |status: u16, error_type: &str| {
            format!(
                r#"{{"error":{{"code":{status},"message":"m","metadata":{{"error_type":"{error_type}","provider_name":"X"}}}}}}"#
            )
        };
        for (status, error_type, class) in [
            (429, "rate_limit_exceeded", RetryClass::Transient),
            (503, "provider_overloaded", RetryClass::Capacity),
            (502, "provider_unavailable", RetryClass::Transient),
            (400, "context_length_exceeded", RetryClass::Permanent),
            (403, "refusal", RetryClass::Permanent),
            (403, "content_policy_violation", RetryClass::Permanent),
            (401, "authentication", RetryClass::Permanent),
        ] {
            // Mid-stream (no HTTP status) and pre-stream classify the same way.
            for http in [None, Some(status)] {
                let error = Error::provider_error("or", http, None, &body(status, error_type));
                assert!(
                    matches!(&error, Error::ProviderError { code: Some(code), .. } if code == error_type)
                );
                assert_eq!(error.retry_class(), class, "{error_type} {http:?}");
            }
        }
        // A 402 is retryable only when the provider says when to retry.
        let payment = body(402, "payment_required");
        assert_eq!(
            Error::provider_error("or", Some(402), None, &payment).retry_class(),
            RetryClass::Permanent
        );
        assert_eq!(
            Error::provider_error("or", Some(402), Some(Duration::from_secs(3)), &payment)
                .retry_class(),
            RetryClass::Transient
        );
    }

    #[test]
    fn google_status_names_classify_overload_as_capacity() {
        let unavailable = r#"{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}"#;
        let error = Error::provider_error("google", None, None, unavailable);
        assert!(matches!(&error, Error::ProviderError { status: Some(503), code: Some(code), .. } if code == "UNAVAILABLE"));
        assert!(error.is_capacity_error());
        let exhausted = r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}"#;
        assert_eq!(
            Error::provider_error("google", None, None, exhausted).retry_class(),
            RetryClass::Transient
        );
        let invalid = r#"{"error":{"code":400,"status":"INVALID_ARGUMENT"}}"#;
        assert_eq!(
            Error::provider_error("google", Some(400), None, invalid).retry_class(),
            RetryClass::Permanent
        );
    }

    #[test]
    fn body_delay_hints_apply_when_no_header_delay_is_given() {
        let error = Error::provider_error(
            "any",
            Some(429),
            None,
            r#"{"error":{"message":"slow down","retry_after_ms":1500}}"#,
        );
        assert_eq!(error.retry_after(), Some(Duration::from_millis(1500)));
        let error = Error::provider_error(
            "any",
            Some(429),
            Some(Duration::from_secs(9)),
            r#"{"retry_after":1}"#,
        );
        assert_eq!(error.retry_after(), Some(Duration::from_secs(9)));
    }

    #[test]
    fn retry_classes_cover_non_provider_errors() {
        assert_eq!(Error::ApiError("x".into()).retry_class(), RetryClass::Transient);
        assert_eq!(
            Error::StreamTruncated("x".into()).retry_class(),
            RetryClass::Transient
        );
        assert_eq!(
            Error::InvalidRequest("x".into()).retry_class(),
            RetryClass::Permanent
        );
        assert_eq!(
            Error::UnexpectedToolCalls {
                expected: "t".into(),
                stop_reason: StopReason::MaxTokens,
                found: vec![],
            }
            .retry_class(),
            RetryClass::Permanent
        );
    }

    #[test]
    fn rate_limit_reset_headers_are_used_without_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-reset-requests", "1s".parse().unwrap());
        headers.insert("x-ratelimit-reset-tokens", "1m2.5s".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(&headers),
            Some(Duration::from_millis(62_500))
        );
        headers.insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
        assert_eq!(retry_after_from_headers(&headers), Some(Duration::from_secs(3)));
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
