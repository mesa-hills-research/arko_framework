use crate::stop::{Refusal, StopReason};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// What kind of failure an error is, beside how it may be retried
/// ([`RetryClass`], which this doesn't change). A host decides by it what a
/// failure means for the work around the request: a lost login stops
/// everything that uses the credential, a request over the model's input
/// limit can be made smaller, and an invalid request fails one task. Read
/// it with [`Error::kind`].
///
/// A provider's error is classified from its codes first (its error code
/// or type, then its HTTP status), then from the documented phrases of its
/// message; anything else is [`Unknown`](Self::Unknown), which callers
/// treat as before kinds existed. Temporary failures (rate limits,
/// capacity, server and network errors) are `Unknown`: their
/// [`RetryClass`] says what they are.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The credential is missing, invalid, expired or revoked: an HTTP 401,
    /// an API key the provider rejects, a ChatGPT sign-in whose refresh
    /// failed, a Claude Code CLI that isn't signed in. A new credential (or
    /// sign-in) is needed; every request that uses it fails until then.
    Authentication,
    /// The credential works but may not use this model, feature or region
    /// (an HTTP 403).
    Permission,
    /// A credit, spend or quota wall (an HTTP 402, OpenAI
    /// `insufficient_quota`, Anthropic `billing_error`, a Gemini daily
    /// quota). Every request on the account fails until it is lifted.
    Billing,
    /// The request's input is over the model's input limit.
    ContextOverflow,
    /// The request body, or one value in it, is over a size limit (an HTTP
    /// 413, OpenAI `string_above_max_length`).
    RequestTooLarge,
    /// Any other request the provider rejects: a bad parameter, schema,
    /// image or history.
    InvalidRequest,
    /// The model or resource doesn't exist or isn't available (an HTTP 404,
    /// `model_not_found`).
    NotFound,
    /// The model or a filter around it declined the request (a policy
    /// block).
    Refusal,
    /// A feature the provider or arko doesn't support.
    Unsupported,
    /// Not classified. Also what a kind added by a later version loads as.
    #[serde(other)]
    Unknown,
}

/// Codes that name an error's kind on their own, whatever the HTTP status.
/// Generic codes (`invalid_request_error`, `INVALID_ARGUMENT`, ...) are not
/// here: their message or status decides. Sources: OpenAI error codes
/// (https://developers.openai.com/api/docs/guides/error-codes) and the
/// Codex client (codex-rs/codex-api/src/sse/responses_error.rs,
/// codex-rs/codex-api/src/api_bridge.rs, codex-rs/login/src/auth/manager.rs);
/// Anthropic error types (https://platform.claude.com/docs/en/api/errors);
/// the Gemini Interactions API codes
/// (https://ai.google.dev/gemini-api/docs/api-errors) and Google RPC
/// statuses; OpenRouter `error.metadata.error_type` values
/// (https://openrouter.ai/docs/api/reference/errors-and-debugging).
const KIND_CODES: &[(&str, ErrorKind)] = &[
    // Credentials.
    ("invalid_api_key", ErrorKind::Authentication),
    ("invalid_grant", ErrorKind::Authentication),
    ("invalid_client", ErrorKind::Authentication),
    ("unauthorized_client", ErrorKind::Authentication),
    ("token_expired", ErrorKind::Authentication),
    ("refresh_token_expired", ErrorKind::Authentication),
    ("refresh_token_reused", ErrorKind::Authentication),
    ("refresh_token_invalidated", ErrorKind::Authentication),
    ("authentication_error", ErrorKind::Authentication),
    ("authentication", ErrorKind::Authentication),
    ("unauthorized", ErrorKind::Authentication),
    ("UNAUTHENTICATED", ErrorKind::Authentication),
    // Permissions.
    ("permission_error", ErrorKind::Permission),
    ("permission_denied", ErrorKind::Permission),
    ("PERMISSION_DENIED", ErrorKind::Permission),
    ("forbidden", ErrorKind::Permission),
    (
        "unsupported_country_region_territory",
        ErrorKind::Permission,
    ),
    ("data_residency_mismatch", ErrorKind::Permission),
    // Credit, spend and quota walls.
    ("insufficient_quota", ErrorKind::Billing),
    ("billing_hard_limit_reached", ErrorKind::Billing),
    ("billing_error", ErrorKind::Billing),
    ("payment_required", ErrorKind::Billing),
    ("quota_exceeded", ErrorKind::Billing),
    ("credit_balance_exhausted", ErrorKind::Billing),
    ("organization_spend_limit_exceeded", ErrorKind::Billing),
    ("organization_usage_limit_exceeded", ErrorKind::Billing),
    ("project_spend_limit_exceeded", ErrorKind::Billing),
    ("usage_not_included", ErrorKind::Billing),
    // Gemini Interactions API: "Billing or account prerequisite".
    ("failed_precondition", ErrorKind::Billing),
    ("FAILED_PRECONDITION", ErrorKind::Billing),
    // The input is over the model's limit.
    ("context_length_exceeded", ErrorKind::ContextOverflow),
    ("token_limit_exceeded", ErrorKind::ContextOverflow),
    // Voyage AI: a request over the model's total tokens per request.
    ("TOO_MANY_TOKENS_IN_BATCH", ErrorKind::ContextOverflow),
    // Size limits on the body or one value in it.
    ("request_too_large", ErrorKind::RequestTooLarge),
    ("payload_too_large", ErrorKind::RequestTooLarge),
    ("content_too_large", ErrorKind::RequestTooLarge),
    ("string_above_max_length", ErrorKind::RequestTooLarge),
    ("string_too_long", ErrorKind::RequestTooLarge),
    // Missing models and resources.
    ("model_not_found", ErrorKind::NotFound),
    ("not_found", ErrorKind::NotFound),
    ("not_found_error", ErrorKind::NotFound),
    ("NOT_FOUND", ErrorKind::NotFound),
    ("no_eligible_provider", ErrorKind::NotFound),
    // Policy blocks.
    ("content_policy_violation", ErrorKind::Refusal),
    ("refusal", ErrorKind::Refusal),
    ("bio_policy", ErrorKind::Refusal),
    ("cyber_policy", ErrorKind::Refusal),
    ("misalignment_policy_violation", ErrorKind::Refusal),
    ("image_content_policy_violation", ErrorKind::Refusal),
    // Unsupported operations.
    ("unimplemented", ErrorKind::Unsupported),
    ("not_implemented", ErrorKind::Unsupported),
    ("UNIMPLEMENTED", ErrorKind::Unsupported),
];

/// Codes of a rejected request that say no more than that: the message or
/// the HTTP status may tell more (Anthropic reports an overlong prompt as
/// `invalid_request_error`, Gemini an invalid API key as
/// `INVALID_ARGUMENT`).
const GENERIC_INVALID_REQUEST_CODES: &[&str] = &[
    "invalid_request_error",
    "invalid_request",
    "INVALID_ARGUMENT",
    "bad_request",
];

/// Documented message phrases, matched in lowercase when the code didn't
/// decide. Sources:
/// - OpenAI Responses (and the ChatGPT backend, codex-rs
///   `sse/responses.rs`): "Your input exceeds the context window of this
///   model."; Chat Completions and OpenRouter: "This model's maximum
///   context length is N tokens", "This endpoint's maximum context length
///   is N tokens"; OpenAI per-value limits: "string too long. Expected a
///   string with maximum length N".
/// - Anthropic (https://platform.claude.com/docs/en/api/errors): "prompt is
///   too long: N tokens > M maximum"; "input length and max_tokens exceed
///   context limit".
/// - Gemini: "The input token count (N) exceeds the maximum number of
///   tokens allowed (M)."; an invalid key: "API key not valid", reason
///   `API_KEY_INVALID`.
const KIND_PHRASES: &[(&str, ErrorKind)] = &[
    ("context_length_exceeded", ErrorKind::ContextOverflow),
    ("exceeds the context window", ErrorKind::ContextOverflow),
    ("maximum context length", ErrorKind::ContextOverflow),
    ("prompt is too long", ErrorKind::ContextOverflow),
    ("exceed context limit", ErrorKind::ContextOverflow),
    (
        "exceeds the maximum number of tokens allowed",
        ErrorKind::ContextOverflow,
    ),
    ("string_above_max_length", ErrorKind::RequestTooLarge),
    ("string too long", ErrorKind::RequestTooLarge),
    ("api key not valid", ErrorKind::Authentication),
    ("api_key_invalid", ErrorKind::Authentication),
];

/// The kind an HTTP status names on its own.
fn status_kind(status: u16) -> Option<ErrorKind> {
    match status {
        401 => Some(ErrorKind::Authentication),
        402 => Some(ErrorKind::Billing),
        403 => Some(ErrorKind::Permission),
        404 => Some(ErrorKind::NotFound),
        413 => Some(ErrorKind::RequestTooLarge),
        501 => Some(ErrorKind::Unsupported),
        _ => None,
    }
}

/// The kind of a provider error with `status`, `code` and `message` (the
/// label and the error body); see [`ErrorKind`].
fn provider_error_kind(status: Option<u16>, code: Option<&str>, message: &str) -> ErrorKind {
    if let Some(kind) = code.and_then(|code| {
        KIND_CODES
            .iter()
            .find_map(|(known, kind)| (*known == code).then_some(*kind))
    }) {
        return kind;
    }
    let status = status.or_else(|| code.and_then(code_status));
    if let Some(kind) = status.and_then(status_kind) {
        return kind;
    }
    // Phrases only for a request the provider rejected, never for an
    // outage whose text happens to mention a limit.
    let rejected = status.is_none_or(|status| (400..500).contains(&status));
    if rejected && let Some(kind) = message_kind(message) {
        return kind;
    }
    let generic = code.is_some_and(|code| GENERIC_INVALID_REQUEST_CODES.contains(&code));
    match status {
        _ if generic => ErrorKind::InvalidRequest,
        // Other rejected requests, except timeouts, conflicts a retry may
        // clear, rate limits and cancelled requests.
        Some(status)
            if (400..500).contains(&status) && !matches!(status, 408 | 409 | 429 | 499) =>
        {
            ErrorKind::InvalidRequest
        }
        // Any other permanent code (images, parameters, history).
        _ if code.is_some_and(|code| PERMANENT_CODES.contains(&code)) => ErrorKind::InvalidRequest,
        _ => ErrorKind::Unknown,
    }
}

/// The kind a documented phrase in `message` names, if any.
fn message_kind(message: &str) -> Option<ErrorKind> {
    let message = message.to_ascii_lowercase();
    KIND_PHRASES
        .iter()
        .find_map(|(phrase, kind)| message.contains(phrase).then_some(*kind))
}

impl ErrorKind {
    /// The kind's name as it is saved: `"authentication"`,
    /// `"context_overflow"`, ...
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::Permission => "permission",
            Self::Billing => "billing",
            Self::ContextOverflow => "context_overflow",
            Self::RequestTooLarge => "request_too_large",
            Self::InvalidRequest => "invalid_request",
            Self::NotFound => "not_found",
            Self::Refusal => "refusal",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
        }
    }

    /// The kind of a failed Claude Code CLI turn, from what the CLI reports:
    /// the assistant message's `error` field (`authentication_failed`,
    /// `billing_error`, `invalid_request`, ...; the Claude Agent SDK's
    /// `SDKAssistantMessageError`), the result's `api_error_status` (the
    /// HTTP status of the failed API request) and `terminal_reason`
    /// (`prompt_too_long`, ...), and the result text, whose Anthropic error
    /// phrases are read as for an API error. Temporary failures
    /// (`rate_limit`, `overloaded`, `server_error`) and unrecognised values
    /// are [`Unknown`](Self::Unknown).
    pub fn from_claude_code(
        error: Option<&str>,
        api_error_status: Option<u16>,
        terminal_reason: Option<&str>,
        text: &str,
    ) -> Self {
        let named = match error {
            Some("authentication_failed" | "verification_required" | "cloud_credential_error") => {
                Some(Self::Authentication)
            }
            Some("oauth_org_not_allowed") => Some(Self::Permission),
            Some("billing_error" | "account_on_hold") => Some(Self::Billing),
            Some("model_not_found") => Some(Self::NotFound),
            _ => None,
        };
        if let Some(kind) = named.or_else(|| api_error_status.and_then(status_kind)) {
            return kind;
        }
        if terminal_reason == Some("prompt_too_long") {
            return Self::ContextOverflow;
        }
        let rejected = error == Some("invalid_request")
            || api_error_status.is_some_and(|status| (400..500).contains(&status));
        match message_kind(text) {
            Some(kind) if rejected || kind == Self::ContextOverflow => kind,
            _ if rejected && !matches!(api_error_status, Some(408 | 409 | 429)) => {
                Self::InvalidRequest
            }
            _ => Self::Unknown,
        }
    }
}

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
    /// A plan usage limit (ChatGPT `usage_limit_reached`): requests fail
    /// until the limit resets, typically hours or days later
    /// ([`Error::retry_after`] holds the time left when the provider said).
    /// Retryable after that wait, not within a short recovery budget.
    UsageLimit,
}

impl RetryClass {
    pub fn is_retryable(self) -> bool {
        !matches!(self, Self::Permanent)
    }
}

/// Codes meaning the request itself must change. They take precedence over a
/// retryable HTTP status. Includes OpenAI/Anthropic error codes and types
/// (OpenAI Responses `ResponseError.code` values:
/// https://developers.openai.com/api/reference/resources/responses/streaming-events),
/// OpenRouter `error.metadata.error_type` values
/// (https://openrouter.ai/docs/api_reference/errors-and-debugging), Gemini
/// Interactions API codes (https://ai.google.dev/gemini-api/docs/api-errors)
/// and Google RPC status names.
const PERMANENT_CODES: &[&str] = &[
    "insufficient_quota",
    "billing_hard_limit_reached",
    // Gemini: "You have exceeded your daily quota. Wait until the quota
    // resets or request a quota increase." A daily wall, like OpenAI's
    // `insufficient_quota`, not a short rate limit.
    "quota_exceeded",
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
    "failed_precondition",
    "parameter_unknown",
    "invalid_prompt",
    "context_length_exceeded",
    "max_tokens_exceeded",
    "token_limit_exceeded",
    "string_too_long",
    "model_not_found",
    "not_found",
    "not_found_error",
    // A 409 conflict that needs resolving, unlike Gemini's retryable `aborted`.
    "already_exists",
    "conflict_error",
    // A 501: the operation is not supported.
    "unimplemented",
    "not_implemented",
    "precondition_failed",
    "payload_too_large",
    "request_too_large",
    "unprocessable",
    "content_policy_violation",
    "refusal",
    "data_residency_mismatch",
    "bio_policy",
    // ChatGPT (Codex backend) `response.failed` codes that the official
    // Codex client treats as fatal (codex-rs/codex-api/src/sse/responses_error.rs):
    // quota/spend walls, a plan without API usage, and a policy block.
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    // A quota wall Codex reports like the ones above when it arrives with an
    // HTTP 429 (codex-rs/codex-api/src/api_bridge.rs).
    "organization_usage_limit_exceeded",
    "project_spend_limit_exceeded",
    "usage_not_included",
    // Flex processing has no capacity: Codex does not retry it (its
    // `ApiError::FlexUnavailable`, also sent with a 429); a request can use
    // another service tier instead.
    "flex_unavailable",
    "cyber_policy",
    "misalignment_policy_violation",
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "invalid_image_mode",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "image_content_policy_violation",
    "image_file_too_large",
    "image_file_not_found",
    "unsupported_image_media_type",
    "empty_image_file",
    "failed_to_download_image",
    // Gemini: the replayed history lacks a required thought signature.
    "missing_thought_signature",
    // arko's code for OpenRouter's "no available model provider meets your
    // routing requirements": provider preferences exclude every endpoint.
    "no_eligible_provider",
    "INVALID_ARGUMENT",
    "FAILED_PRECONDITION",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "NOT_FOUND",
    "UNIMPLEMENTED",
];

/// Explicit temporary capacity signals (Gemini reports an overloaded model as
/// 503 `UNAVAILABLE` on generateContent and `service_unavailable` on the
/// Interactions API: "The service is temporarily overloaded or down").
const CAPACITY_CODES: &[&str] = &[
    "server_is_overloaded",
    "overloaded_error",
    "service_unavailable_error",
    "service_unavailable",
    "provider_overloaded",
    "UNAVAILABLE",
];

/// HTTP statuses of documented string codes, used to classify errors that
/// arrive without one: a stream error event or failed interaction reported
/// after the HTTP 200. Sources: Gemini Interactions API
/// (https://ai.google.dev/gemini-api/docs/api-errors; unlisted codes fall back
/// to the snake_case HTTP status name, e.g. `gateway_timeout`), Anthropic
/// (https://platform.claude.com/docs/en/api/errors) and Google RPC names.
const CODE_STATUSES: &[(&str, u16)] = &[
    // Gemini Interactions API.
    ("invalid_request", 400),
    ("failed_precondition", 400),
    ("parameter_unknown", 400),
    ("authentication", 401),
    ("payment_required", 402),
    ("permission_denied", 403),
    ("not_found", 404),
    ("model_not_found", 404),
    ("already_exists", 409),
    ("aborted", 409),
    ("out_of_range", 416),
    ("rate_limit_exceeded", 429),
    ("quota_exceeded", 429),
    ("too_many_requests", 429),
    ("cancelled", 499),
    ("api_error", 500),
    ("unimplemented", 501),
    ("service_unavailable", 503),
    ("deadline_exceeded", 504),
    // snake_case HTTP status names (Gemini fallback codes).
    ("bad_request", 400),
    ("unauthorized", 401),
    ("forbidden", 403),
    ("method_not_allowed", 405),
    ("request_timeout", 408),
    ("conflict", 409),
    ("gone", 410),
    ("payload_too_large", 413),
    ("content_too_large", 413),
    ("unprocessable_entity", 422),
    ("unprocessable_content", 422),
    ("internal_server_error", 500),
    ("not_implemented", 501),
    ("bad_gateway", 502),
    ("gateway_timeout", 504),
    ("not_acceptable", 406),
    ("length_required", 411),
    ("precondition_failed", 412),
    ("uri_too_long", 414),
    ("unsupported_media_type", 415),
    ("range_not_satisfiable", 416),
    ("expectation_failed", 417),
    ("misdirected_request", 421),
    ("locked", 423),
    ("failed_dependency", 424),
    ("too_early", 425),
    ("upgrade_required", 426),
    ("precondition_required", 428),
    ("request_header_fields_too_large", 431),
    ("unavailable_for_legal_reasons", 451),
    ("client_closed_request", 499),
    ("http_version_not_supported", 505),
    ("insufficient_storage", 507),
    ("network_authentication_required", 511),
    // Anthropic error types (`api_error` is shared with Gemini).
    ("invalid_request_error", 400),
    ("authentication_error", 401),
    ("billing_error", 402),
    ("permission_error", 403),
    ("not_found_error", 404),
    ("conflict_error", 409),
    ("request_too_large", 413),
    ("rate_limit_error", 429),
    ("timeout_error", 504),
    ("overloaded_error", 529),
    // Google RPC status names.
    ("INVALID_ARGUMENT", 400),
    ("FAILED_PRECONDITION", 400),
    ("OUT_OF_RANGE", 400),
    ("UNAUTHENTICATED", 401),
    ("PERMISSION_DENIED", 403),
    ("NOT_FOUND", 404),
    ("ALREADY_EXISTS", 409),
    ("ABORTED", 409),
    ("RESOURCE_EXHAUSTED", 429),
    ("CANCELLED", 499),
    ("INTERNAL", 500),
    ("UNKNOWN", 500),
    ("UNIMPLEMENTED", 501),
    ("UNAVAILABLE", 503),
    ("DEADLINE_EXCEEDED", 504),
];

/// Plan usage limits (the ChatGPT backend's HTTP 429 `usage_limit_reached`,
/// with the reset time in `resets_at`; codex-rs `api_bridge.rs`).
const USAGE_LIMIT_CODES: &[&str] = &["usage_limit_reached"];

/// Short request-rate limits besides HTTP 429 itself: OpenAI/Codex
/// `rate_limit_exceeded` and `slow_down` (sent with a 503 or on a stream
/// error), and the 429 names of other providers.
const RATE_LIMIT_CODES: &[&str] = &[
    "rate_limit_exceeded",
    "slow_down",
    "too_many_requests",
    "rate_limit_error",
    "RESOURCE_EXHAUSTED",
];

fn code_status(code: &str) -> Option<u16> {
    CODE_STATUSES
        .iter()
        .find_map(|(known, status)| (*known == code).then_some(*status))
}

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
    #[error(
        "Expected one call to `{expected}`; the response stopped with {stop_reason:?} and called {found:?}"
    )]
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
                if code.is_some_and(|code| USAGE_LIMIT_CODES.contains(&code)) {
                    return RetryClass::UsageLimit;
                }
                let capacity = code.is_some_and(|code| CAPACITY_CODES.contains(&code));
                // Without an HTTP status (an error reported after the 200), a
                // documented code stands in for it.
                if let Some(status) = status.or_else(|| code.and_then(code_status)) {
                    return if !matches!(status, 408 | 409 | 429 | 500..=599) {
                        RetryClass::Permanent
                    } else if capacity {
                        RetryClass::Capacity
                    } else {
                        RetryClass::Transient
                    };
                }
                // Any other error reported after the 200 is bounded-transient:
                // the request was accepted, so an unrecognised code (for
                // example OpenAI `server_error`, a Gemini generation error such
                // as `malformed_function_call`, or an event with no code) is
                // most likely a server-side failure. Permanent codes were
                // handled above.
                if capacity {
                    RetryClass::Capacity
                } else {
                    RetryClass::Transient
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

    /// A plan usage limit; see [`RetryClass::UsageLimit`].
    pub fn is_usage_limit(&self) -> bool {
        self.retry_class() == RetryClass::UsageLimit
    }

    /// A short request-rate limit: a retryable HTTP 429 (or a 429 code on an
    /// error reported after the response started), or `rate_limit_exceeded`
    /// / `slow_down`. Such errors are [`RetryClass::Transient`]; this
    /// refines that class. Quota walls (`insufficient_quota`, ...) are
    /// permanent and plan usage limits are [`RetryClass::UsageLimit`], so
    /// neither is a rate limit. [`retry_after`](Self::retry_after) holds the
    /// provider's delay when it sent one.
    pub fn is_rate_limit(&self) -> bool {
        if self.retry_class() != RetryClass::Transient {
            return false;
        }
        match self {
            Self::ProviderError { status, code, .. } => {
                let code = code.as_deref();
                code.is_some_and(|code| RATE_LIMIT_CODES.contains(&code))
                    || status.or_else(|| code.and_then(code_status)) == Some(429)
            }
            Self::HttpError(error) => error.status().is_some_and(|status| status.as_u16() == 429),
            _ => false,
        }
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

    /// What kind of failure this is; see [`ErrorKind`]. A provider error is
    /// classified from its code, then its HTTP status, then documented
    /// phrases of its message. arko's own errors: a rejected or missing
    /// credential is [`ErrorKind::Authentication`], a local request check
    /// [`ErrorKind::InvalidRequest`], an unsupported provider, model or
    /// feature [`ErrorKind::Unsupported`], a refusal
    /// [`ErrorKind::Refusal`], and a response cut off at the model's
    /// context window [`ErrorKind::ContextOverflow`].
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::ProviderError {
                message,
                status,
                code,
                ..
            } => provider_error_kind(*status, code.as_deref(), message),
            Self::HttpError(error) => error
                .status()
                .and_then(|status| status_kind(status.as_u16()))
                .unwrap_or(ErrorKind::Unknown),
            Self::AuthenticationFailed(_)
            | Self::MissingApiKey(_)
            | Self::MissingHuggingFaceToken(_) => ErrorKind::Authentication,
            Self::InvalidRequest(_) | Self::InvalidDimension(_) => ErrorKind::InvalidRequest,
            Self::ProviderNotSupported(_)
            | Self::ModelNotSupported(_)
            | Self::UnsupportedFeature(_) => ErrorKind::Unsupported,
            Self::TokenLimitExceeded(_) => ErrorKind::ContextOverflow,
            Self::BatchSizeLimitExceeded(_) => ErrorKind::RequestTooLarge,
            Self::Refusal(_) => ErrorKind::Refusal,
            Self::UnexpectedToolCalls {
                stop_reason: StopReason::ContextWindowExceeded,
                ..
            } => ErrorKind::ContextOverflow,
            _ => ErrorKind::Unknown,
        }
    }

    /// Builds a provider error from an error body. Understands string codes
    /// (`error.code`, `error.type`, `error.status`), OpenRouter's typed
    /// `error.metadata.error_type`, and numeric HTTP-style codes
    /// (`error.code: 429`, used by OpenRouter and Google), which stand in for
    /// the status when the HTTP status is absent or 2xx, as on stream errors.
    /// A delay hint in the body is used when no header delay is supplied, and
    /// a rate limit without either takes the wait its message names ("Please
    /// try again in 2.5s"), as Codex does.
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
            // A flat stream event (OpenAI `{"type":"error","code":null,...}`)
            // has the event name in `type`, not an error type.
            .or_else(|| {
                error
                    .get("type")
                    .and_then(Value::as_str)
                    .filter(|kind| *kind != "error")
            })
            .or_else(|| error.get("status").and_then(Value::as_str))
            .or_else(|| error.as_str())
            .map(str::to_owned);
        let status = status
            .filter(|status| !(200..300).contains(status))
            .or(numeric_code);
        let mut provider_error = Self::ProviderError {
            message: format!("{label}: {body}"),
            status,
            code,
            retry_after: retry_after.or_else(|| retry_after_from_value(&value)),
        };
        // OpenAI names the wait only in a rate limit's message; Codex reads it
        // from there (codex-rs/codex-api/src/sse/responses_error.rs).
        if provider_error.retry_after().is_none()
            && provider_error.is_rate_limit()
            && let Some(delay) = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .and_then(retry_delay_from_message)
            && let Self::ProviderError { retry_after, .. } = &mut provider_error
        {
            *retry_after = Some(delay);
        }
        provider_error
    }

    pub(crate) async fn from_response(label: &str, response: reqwest::Response) -> Self {
        let status = response.status().as_u16();
        let retry_after = retry_after_from_headers(status, response.headers());
        // Preserve the HTTP status even when reading the error body fails.
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable error body>".into());
        Self::provider_error(label, Some(status), retry_after, &body)
    }
}

/// Delay requested by response headers: `Retry-After` (seconds or an HTTP
/// date) on any status; on a 429 without it, OpenAI-style
/// `x-ratelimit-reset-requests` / `x-ratelimit-reset-tokens` durations such
/// as `1m2.5s`.
///
/// Those headers are "the time until the rate limit ... resets to its
/// initial state" (https://developers.openai.com/api/docs/guides/rate-limits):
/// a full refill, sent on every response, so they say nothing about a
/// non-rate-limit failure and overstate the wait for a limit that was not
/// hit. Only the limits whose `x-ratelimit-remaining-*` is 0 count (the
/// longest of them, since each must recover). When no remaining count is 0
/// or none is reported, the limit that was hit is unknown and the shorter
/// reset is used: it is the earliest point a limit is known to have fully
/// recovered, and callers treat the hint as a minimum under their own
/// backoff, so underestimating costs one early retry while the longer reset
/// (often minutes for tokens) could stall a request that would already
/// succeed.
///
/// Capped at [`MAX_SERVER_DELAY_HINT`](crate::retry::MAX_SERVER_DELAY_HINT).
pub(crate) fn retry_after_from_headers(
    status: u16,
    headers: &reqwest::header::HeaderMap,
) -> Option<Duration> {
    header_delay(status, headers).map(cap_server_hint)
}

/// A delay hint from a response, capped so callers can add it to the current
/// time; see [`MAX_SERVER_DELAY_HINT`](crate::retry::MAX_SERVER_DELAY_HINT).
fn cap_server_hint(delay: Duration) -> Duration {
    delay.min(crate::retry::MAX_SERVER_DELAY_HINT)
}

fn header_delay(status: u16, headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    retry_after_header(headers).or_else(|| {
        if status != 429 {
            return None;
        }
        let header = |name: String| {
            headers
                .get(name.as_str())
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
        };
        let limits: Vec<(Option<bool>, Duration)> = ["requests", "tokens"]
            .iter()
            .filter_map(|limit| {
                let reset = parse_relative_duration(header(format!("x-ratelimit-reset-{limit}"))?)?;
                let exhausted = header(format!("x-ratelimit-remaining-{limit}"))
                    .and_then(|remaining| remaining.parse::<f64>().ok())
                    .map(|remaining| remaining <= 0.0);
                Some((exhausted, reset))
            })
            .collect();
        limits
            .iter()
            .filter(|(exhausted, _)| *exhausted == Some(true))
            .map(|(_, reset)| *reset)
            .max()
            .or_else(|| limits.iter().map(|(_, reset)| *reset).min())
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
/// `resets_at`/`reset_at`/`resetsAt`/`resetAt` (epoch seconds or milliseconds)
/// or OpenRouter's `error.metadata.headers.X-RateLimit-Reset` (epoch
/// milliseconds). Capped at
/// [`MAX_SERVER_DELAY_HINT`](crate::retry::MAX_SERVER_DELAY_HINT).
pub(crate) fn retry_after_from_value(value: &Value) -> Option<Duration> {
    body_delay(value).map(cap_server_hint)
}

fn body_delay(value: &Value) -> Option<Duration> {
    if let Some(milliseconds) = find_numeric_field(value, &["retry_after_ms", "retryAfterMs"]) {
        return Some(Duration::from_millis(milliseconds.ceil() as u64));
    }
    if let Some(seconds) = find_numeric_field(value, &["retry_after", "retryAfter"]) {
        return Some(Duration::from_millis((seconds * 1_000_f64).ceil() as u64));
    }
    let reset_at = find_numeric_field(value, &["resets_at", "reset_at", "resetsAt", "resetAt"])
        .or_else(|| openrouter_rate_limit_reset(value))?;
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

/// OpenRouter passes a rate limit's headers along in its error body:
/// `error.metadata.headers."X-RateLimit-Reset"`, in epoch milliseconds
/// (usually a string).
fn openrouter_rate_limit_reset(value: &Value) -> Option<f64> {
    value
        .pointer("/error/metadata/headers")?
        .as_object()?
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-ratelimit-reset"))
        .and_then(|(_, reset)| value_as_f64(reset))
        .filter(|reset| reset.is_finite() && *reset >= 0.0)
}

/// The wait a rate-limit message names, read as the Codex CLI reads it
/// (`(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)`): "Please try
/// again in 2.5s", "try again in 350ms", "Try again in 11 seconds". Capped at
/// [`MAX_SERVER_DELAY_HINT`](crate::retry::MAX_SERVER_DELAY_HINT).
fn retry_delay_from_message(message: &str) -> Option<Duration> {
    const PHRASE: &str = "try again in";
    let digits =
        |text: &str| text.len() - text.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    // ASCII lowercasing keeps byte offsets.
    let lowercase = message.to_ascii_lowercase();
    let mut rest = lowercase.as_str();
    while let Some(start) = rest.find(PHRASE) {
        rest = &rest[start + PHRASE.len()..];
        let text = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        let mut end = digits(text);
        if end == 0 {
            continue;
        }
        if let Some(fraction) = text[end..].strip_prefix('.')
            && digits(fraction) > 0
        {
            end += 1 + digits(fraction);
        }
        let Ok(amount) = text[..end].parse::<f64>() else {
            continue;
        };
        let unit = text[end..].trim_start_matches(|c: char| c.is_ascii_whitespace());
        let seconds = if unit.starts_with("ms") {
            amount / 1_000_f64
        } else if unit.starts_with('s') {
            amount
        } else {
            continue;
        };
        return Some(
            Duration::try_from_secs_f64(seconds)
                .map_or(crate::retry::MAX_SERVER_DELAY_HINT, cap_server_hint),
        );
    }
    None
}

/// [`retry_after_from_value`] for an unparsed body. Error bodies reach it
/// through [`Error::provider_error`], so only tests call it directly.
#[cfg(test)]
pub(crate) fn retry_after_from_body(body: &str) -> Option<Duration> {
    let value: Value = serde_json::from_str(body).ok()?;
    retry_after_from_value(&value)
}

#[cfg(test)]
#[path = "error_kind_tests.rs"]
mod kind_tests;

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
        assert!(matches!(
            error,
            Error::ProviderError {
                status: Some(429),
                ..
            }
        ));
        assert_eq!(error.retry_class(), RetryClass::Transient);
        let error = Error::provider_error("stream", None, None, r#"{"error":{"code":400}}"#);
        assert_eq!(error.retry_class(), RetryClass::Permanent);
        // A real HTTP status is not replaced by the body code.
        let error = Error::provider_error("http", Some(503), None, r#"{"error":{"code":400}}"#);
        assert!(matches!(
            error,
            Error::ProviderError {
                status: Some(503),
                ..
            }
        ));
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
        let unavailable =
            r#"{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}"#;
        let error = Error::provider_error("google", None, None, unavailable);
        assert!(
            matches!(&error, Error::ProviderError { status: Some(503), code: Some(code), .. } if code == "UNAVAILABLE")
        );
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
    fn gemini_interaction_codes_classify_by_their_documented_status() {
        use RetryClass::{Capacity, Permanent, Transient};
        // SSE error event shape from https://ai.google.dev/gemini-api/docs/api-errors;
        // stream errors carry the same codes as HTTP errors.
        for (code, class) in [
            ("invalid_request", Permanent),
            ("failed_precondition", Permanent),
            ("out_of_range", Permanent),
            ("parameter_unknown", Permanent),
            ("authentication", Permanent),
            ("payment_required", Permanent),
            ("permission_denied", Permanent),
            ("not_found", Permanent),
            ("model_not_found", Permanent),
            ("already_exists", Permanent),
            ("aborted", Transient),
            ("rate_limit_exceeded", Transient),
            ("quota_exceeded", Permanent),
            ("too_many_requests", Transient),
            ("cancelled", Permanent),
            ("api_error", Transient),
            ("unimplemented", Permanent),
            ("service_unavailable", Capacity),
            ("deadline_exceeded", Transient),
            // Fallback codes: snake_case HTTP status names.
            ("gateway_timeout", Transient),
            ("bad_gateway", Transient),
            ("unprocessable_entity", Permanent),
            // Generation errors: resampling may produce a valid output...
            ("malformed_function_call", Transient),
            ("malformed_tool_call", Transient),
            ("unexpected_tool_call", Transient),
            ("too_many_tool_calls", Transient),
            ("no_image", Transient),
            // ...but a missing signature is in the request history.
            ("missing_thought_signature", Permanent),
        ] {
            let event =
                format!(r#"{{"event_type":"error","error":{{"code":"{code}","message":"m"}}}}"#);
            let error = Error::provider_error("google stream", None, None, &event);
            assert_eq!(error.retry_class(), class, "{code}");
        }
        // The daily quota wall stays permanent with its HTTP 429; an overload
        // is a capacity signal with or without its 503.
        let quota = r#"{"error":{"code":"quota_exceeded","message":"You have exceeded your daily quota."}}"#;
        assert_eq!(
            Error::provider_error("google", Some(429), None, quota).retry_class(),
            Permanent
        );
        let overloaded = r#"{"error":{"code":"service_unavailable","message":"m"}}"#;
        assert_eq!(
            Error::provider_error("google", Some(503), None, overloaded).retry_class(),
            Capacity
        );
    }

    #[test]
    fn anthropic_stream_error_types_classify_like_their_http_status() {
        use RetryClass::{Capacity, Permanent, Transient};
        // https://platform.claude.com/docs/en/api/errors
        for (kind, class) in [
            ("api_error", Transient),
            ("timeout_error", Transient),
            ("rate_limit_error", Transient),
            ("overloaded_error", Capacity),
            ("invalid_request_error", Permanent),
            ("authentication_error", Permanent),
            ("billing_error", Permanent),
            ("permission_error", Permanent),
            ("not_found_error", Permanent),
            ("conflict_error", Permanent),
            ("request_too_large", Permanent),
        ] {
            let event = format!(
                r#"{{"type":"error","error":{{"type":"{kind}","message":"m"}},"request_id":"req_1"}}"#
            );
            let error = Error::provider_error("Anthropic stream error", None, None, &event);
            assert_eq!(error.retry_class(), class, "{kind}");
        }
    }

    #[test]
    fn unknown_errors_after_a_success_are_bounded_transient_but_permanent_codes_win() {
        // OpenAI's flat `error` stream event; `code` may be null and `type`
        // is the event name, not an error type.
        let event = r#"{"type":"error","code":null,"message":"An error occurred while processing your request.","param":null,"sequence_number":4}"#;
        let error = Error::provider_error("OpenAI stream error", None, None, event);
        assert!(matches!(
            &error,
            Error::ProviderError {
                status: None,
                code: None,
                ..
            }
        ));
        assert_eq!(error.retry_class(), RetryClass::Transient);
        for (code, class) in [
            ("server_error", RetryClass::Transient),
            ("vector_store_timeout", RetryClass::Transient),
            ("invalid_prompt", RetryClass::Permanent),
            ("image_content_policy_violation", RetryClass::Permanent),
        ] {
            let event = format!(
                r#"{{"type":"error","code":"{code}","message":"m","param":null,"sequence_number":1}}"#
            );
            assert_eq!(
                Error::provider_error("OpenAI stream error", None, None, &event).retry_class(),
                class,
                "{code}"
            );
        }
        // `response.failed` carries the error inside the response object.
        let failed =
            r#"{"id":"resp_1","status":"failed","error":{"code":"invalid_image","message":"m"}}"#;
        assert_eq!(
            Error::provider_error("OpenAI stream error", None, None, failed).retry_class(),
            RetryClass::Permanent
        );
        // Every permanent code stays permanent without an HTTP status.
        for code in PERMANENT_CODES {
            let body = format!(r#"{{"error":{{"code":"{code}"}}}}"#);
            assert_eq!(
                Error::provider_error("stream", None, None, &body).retry_class(),
                RetryClass::Permanent,
                "{code}"
            );
        }
        // A body-less failure after the 200 is retried (boundedly).
        assert_eq!(
            Error::provider_error("stream", None, None, "{}").retry_class(),
            RetryClass::Transient
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
        assert_eq!(
            Error::ApiError("x".into()).retry_class(),
            RetryClass::Transient
        );
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
    fn rate_limit_reset_headers_apply_only_to_an_exhausted_limit_on_a_429() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-ratelimit-reset-requests", "1s".parse().unwrap());
        headers.insert("x-ratelimit-reset-tokens", "1m2.5s".parse().unwrap());
        // Without remaining counts the limit that was hit is unknown.
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::from_secs(1))
        );
        // Reset headers accompany every response; they are no delay hint for
        // other failures.
        for status in [500, 503, 529, 400] {
            assert_eq!(retry_after_from_headers(status, &headers), None, "{status}");
        }
        // Only the exhausted limit counts.
        headers.insert("x-ratelimit-remaining-requests", "12".parse().unwrap());
        headers.insert("x-ratelimit-remaining-tokens", "0".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::from_millis(62_500))
        );
        headers.insert("x-ratelimit-remaining-requests", "0".parse().unwrap());
        headers.insert("x-ratelimit-remaining-tokens", "5000".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::from_secs(1))
        );
        // Both exhausted: both must recover.
        headers.insert("x-ratelimit-remaining-tokens", "0".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::from_millis(62_500))
        );
        // Retry-After wins on any status.
        headers.insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::from_secs(3))
        );
        assert_eq!(
            retry_after_from_headers(503, &headers),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn retry_after_accepts_seconds_http_dates_and_rejects_invalid_values() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "60".parse().unwrap());
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::from_secs(60))
        );
        let deadline = std::time::SystemTime::now() + Duration::from_secs(60);
        headers.insert(
            reqwest::header::RETRY_AFTER,
            httpdate::fmt_http_date(deadline).parse().unwrap(),
        );
        let delay = retry_after_from_headers(429, &headers).unwrap();
        assert!(delay >= Duration::from_secs(58) && delay <= Duration::from_secs(60));
        for value in ["-1", "NaN", "nonsense"] {
            headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
            assert_eq!(retry_after_from_headers(429, &headers), None);
        }
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Thu, 01 Jan 1970 00:00:00 GMT".parse().unwrap(),
        );
        assert_eq!(
            retry_after_from_headers(429, &headers),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn huge_server_delay_hints_are_capped() {
        let cap = crate::retry::MAX_SERVER_DELAY_HINT;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            u64::MAX.to_string().parse().unwrap(),
        );
        assert_eq!(retry_after_from_headers(429, &headers), Some(cap));
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Fri, 31 Dec 9999 23:59:59 GMT".parse().unwrap(),
        );
        assert_eq!(retry_after_from_headers(429, &headers), Some(cap));
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-ratelimit-reset-requests",
            "99999999999999999999h".parse().unwrap(),
        );
        assert_eq!(retry_after_from_headers(429, &headers), Some(cap));
        for body in [
            r#"{"retry_after_ms":1e300}"#,
            r#"{"retry_after":18446744073709551615}"#,
            r#"{"resets_at":1e300}"#,
        ] {
            assert_eq!(retry_after_from_body(body), Some(cap), "{body}");
        }
        // A capped hint can be added to the current time.
        let _ = std::time::Instant::now() + cap;
        let _ = std::time::SystemTime::now() + cap;
    }

    #[test]
    fn plan_usage_limits_and_short_rate_limits_are_told_apart() {
        // The ChatGPT backend's usage limit: a 429 with the reset time.
        let resets_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3 * 3600;
        let body = format!(
            r#"{{"error":{{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"pro","resets_at":{resets_at}}}}}"#
        );
        let error = Error::provider_error("OpenAI request failed", Some(429), None, &body);
        assert_eq!(error.retry_class(), RetryClass::UsageLimit);
        assert!(error.is_usage_limit() && error.is_retryable());
        assert!(!error.is_rate_limit() && !error.is_capacity_error());
        let wait = error.retry_after().unwrap();
        assert!(wait > Duration::from_secs(3 * 3600 - 60) && wait <= Duration::from_secs(3 * 3600));

        for (status, body) in [
            (Some(429), r#"{"error":{"message":"Rate limit exceeded"}}"#),
            (
                Some(503),
                r#"{"error":{"code":"slow_down","message":"Slow down"}}"#,
            ),
            (
                None,
                r#"{"error":{"code":"rate_limit_exceeded","message":"Try again in 11s"}}"#,
            ),
            (
                None,
                r#"{"type":"error","error":{"type":"rate_limit_error"}}"#,
            ),
        ] {
            let error = Error::provider_error("test", status, Some(Duration::from_secs(36)), body);
            assert!(error.is_rate_limit(), "{body}");
            assert_eq!(error.retry_class(), RetryClass::Transient, "{body}");
            assert!(!error.is_usage_limit());
        }
        // Quota walls, capacity signals and other outages are not rate limits.
        for (status, body) in [
            (Some(429), r#"{"error":{"code":"insufficient_quota"}}"#),
            (None, r#"{"error":{"code":"server_is_overloaded"}}"#),
            (Some(503), "{}"),
            (Some(400), r#"{"error":{"code":"rate_limit_exceeded"}}"#),
        ] {
            assert!(
                !Error::provider_error("test", status, None, body).is_rate_limit(),
                "{body}"
            );
        }
    }

    #[test]
    fn codex_quota_walls_and_flex_unavailability_are_permanent_even_on_a_429() {
        for body in [
            r#"{"error":{"code":"organization_usage_limit_exceeded","message":"m"}}"#,
            r#"{"error":{"code":"flex_unavailable","message":"Flex capacity unavailable."}}"#,
        ] {
            for status in [Some(429), None] {
                let error = Error::provider_error("OpenAI request failed", status, None, body);
                assert_eq!(
                    error.retry_class(),
                    RetryClass::Permanent,
                    "{body} {status:?}"
                );
                assert!(!error.is_rate_limit(), "{body}");
            }
        }
    }

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    #[test]
    fn openrouter_rate_limit_reset_metadata_is_a_delay_hint() {
        let reset = now_millis() + 90_000;
        // OpenRouter's 429 body passes the limit's headers along (the reset in
        // epoch milliseconds, as a string).
        for reset in [format!(r#""{reset}""#), reset.to_string()] {
            let body = format!(
                r#"{{"error":{{"code":429,"message":"Rate limit exceeded: free-models-per-min. ","metadata":{{"headers":{{"X-RateLimit-Limit":"20","X-RateLimit-Remaining":"0","X-RateLimit-Reset":{reset}}},"provider_name":null}}}},"user_id":"user_1"}}"#
            );
            for status in [Some(429), None] {
                let error = Error::provider_error("OpenRouter request failed", status, None, &body);
                assert!(error.is_rate_limit(), "{body}");
                let wait = error.retry_after().unwrap();
                assert!(
                    wait > Duration::from_secs(80) && wait <= Duration::from_secs(90),
                    "{wait:?}"
                );
            }
        }
        // A header delay still wins.
        let body = format!(
            r#"{{"error":{{"code":429,"metadata":{{"headers":{{"x-ratelimit-reset":"{}"}}}}}}}}"#,
            now_millis() + 90_000
        );
        let error = Error::provider_error("or", Some(429), Some(Duration::from_secs(4)), &body);
        assert_eq!(error.retry_after(), Some(Duration::from_secs(4)));
        let error = Error::provider_error("or", Some(429), None, &body);
        assert!(error.retry_after().unwrap() > Duration::from_secs(80));
    }

    #[test]
    fn try_again_in_messages_are_delay_hints_for_rate_limits_only() {
        for (message, expected) in [
            (
                "Rate limit reached for gpt-5 in organization org-x on tokens per min (TPM): Limit 30000, Used 29000, Requested 2000. Please try again in 2.5s. Visit https://platform.openai.com/account/rate-limits to learn more.",
                Duration::from_millis(2_500),
            ),
            ("Please try again in 350ms.", Duration::from_millis(350)),
            ("Try again in 11 seconds", Duration::from_secs(11)),
            ("TRY AGAIN IN 1 second.", Duration::from_secs(1)),
            ("try again in7s", Duration::from_secs(7)),
            // The first phrase without a number does not stop the search.
            (
                "You can try again in a moment; try again in 3s",
                Duration::from_secs(3),
            ),
        ] {
            let message = serde_json::to_string(message).unwrap();
            // An OpenAI stream error event, a `response.failed` error and a 429.
            for (status, body) in [
                (
                    None,
                    format!(
                        r#"{{"type":"error","code":"rate_limit_exceeded","message":{message},"param":null,"sequence_number":3}}"#
                    ),
                ),
                (
                    None,
                    format!(r#"{{"error":{{"code":"slow_down","message":{message}}}}}"#),
                ),
                (Some(429), format!(r#"{{"error":{{"message":{message}}}}}"#)),
            ] {
                let error = Error::provider_error("test", status, None, &body);
                assert!(error.is_rate_limit(), "{body}");
                assert_eq!(error.retry_after(), Some(expected), "{body}");
            }
        }
        // Other hints win over the message.
        let body = r#"{"error":{"code":"rate_limit_exceeded","message":"try again in 5s","retry_after_ms":800}}"#;
        assert_eq!(
            Error::provider_error("test", None, None, body).retry_after(),
            Some(Duration::from_millis(800))
        );
        assert_eq!(
            Error::provider_error("test", Some(429), Some(Duration::from_secs(9)), body)
                .retry_after(),
            Some(Duration::from_secs(9))
        );
        // Only rate limits read their message.
        for (status, body) in [
            (
                Some(503),
                r#"{"error":{"message":"Please try again in 5s"}}"#,
            ),
            (
                None,
                r#"{"error":{"code":"server_error","message":"Please try again in 5s"}}"#,
            ),
            (
                Some(429),
                r#"{"error":{"code":"insufficient_quota","message":"try again in 5s"}}"#,
            ),
            (
                Some(400),
                r#"{"error":{"code":"rate_limit_exceeded","message":"try again in 5s"}}"#,
            ),
        ] {
            assert_eq!(
                Error::provider_error("test", status, None, body).retry_after(),
                None,
                "{body}"
            );
        }
        // No number or unit: no hint. A huge one is capped.
        for message in [
            "try again in a moment",
            "try again in 5 minutes",
            "try again in .5s",
            "try again in 5",
        ] {
            let body =
                format!(r#"{{"error":{{"code":"rate_limit_exceeded","message":"{message}"}}}}"#);
            assert_eq!(
                Error::provider_error("test", None, None, &body).retry_after(),
                None,
                "{message}"
            );
        }
        let body = format!(
            r#"{{"error":{{"code":"rate_limit_exceeded","message":"try again in 1{}s"}}}}"#,
            "0".repeat(400)
        );
        assert_eq!(
            Error::provider_error("test", None, None, &body).retry_after(),
            Some(crate::retry::MAX_SERVER_DELAY_HINT)
        );
    }

    #[test]
    fn codex_backend_fatal_stream_codes_and_status_names_are_permanent() {
        for code in [
            "credit_balance_exhausted",
            "organization_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
            "project_spend_limit_exceeded",
            "usage_not_included",
            "flex_unavailable",
            "cyber_policy",
            "unsupported_media_type",
            "uri_too_long",
            "client_closed_request",
        ] {
            // The shape openai.rs passes for `response.failed` (the event's
            // `response` object) and for wrapped `error` events.
            let body = format!(r#"{{"error":{{"code":"{code}","message":"x"}}}}"#);
            let error = Error::provider_error("stream", None, None, &body);
            assert_eq!(error.retry_class(), RetryClass::Permanent, "{code}");
        }
    }
}
