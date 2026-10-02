//! Error kinds of real provider error bodies, as each provider client hands
//! them to [`Error::provider_error`] (its label, HTTP status when there is
//! one, and body).

use super::*;
use ErrorKind::*;

fn kind(label: &str, status: Option<u16>, body: &str) -> ErrorKind {
    Error::provider_error(label, status, None, body).kind()
}

#[track_caller]
fn assert_kinds(label: &str, cases: &[(Option<u16>, &str, ErrorKind)]) {
    for (status, body, expected) in cases {
        assert_eq!(kind(label, *status, body), *expected, "{status:?} {body}");
    }
}

#[test]
fn openai_api_errors() {
    assert_kinds(
        "OpenAI request failed",
        &[
            // Responses API overflow, as a `response.failed` event's response
            // (codex-rs/codex-api/src/sse/responses.rs test fixture).
            (
                None,
                r#"{"status":"failed","error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again."}}"#,
                ContextOverflow,
            ),
            // Chat Completions overflow.
            (
                Some(400),
                r#"{"error":{"message":"This model's maximum context length is 4096 tokens. However, your messages resulted in 4239 tokens. Please reduce the length of the messages.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"#,
                ContextOverflow,
            ),
            // The same message without a code.
            (
                Some(400),
                r#"{"error":{"message":"Your input exceeds the context window of this model. Please adjust your input and try\nagain.","type":"invalid_request_error","param":null,"code":null}}"#,
                ContextOverflow,
            ),
            // A value over its own size limit
            // (https://community.openai.com/t/compaction-encrypted-content-too-long-400-error/1379862).
            (
                Some(400),
                r#"{"error":{"message":"Invalid 'input[1535].encrypted_content': string too long. Expected a string with maximum length 10485760, but got a string with length 17599352 instead.","type":"invalid_request_error","param":"input[1535].encrypted_content","code":"string_above_max_length"}}"#,
                RequestTooLarge,
            ),
            (
                Some(401),
                r#"{"error":{"message":"Incorrect API key provided: sk-proj-****. You can find your API key at https://platform.openai.com/account/api-keys.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}"#,
                Authentication,
            ),
            // No code: the 401 decides.
            (
                Some(401),
                r#"{"error":{"message":"You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).","type":"invalid_request_error","param":null,"code":null}}"#,
                Authentication,
            ),
            (
                Some(403),
                r#"{"error":{"code":"unsupported_country_region_territory","message":"Country, region, or territory not supported","param":null,"type":"request_forbidden"}}"#,
                Permission,
            ),
            (
                Some(404),
                r#"{"error":{"message":"The model `gpt-9` does not exist or you do not have access to it.","type":"invalid_request_error","param":null,"code":"model_not_found"}}"#,
                NotFound,
            ),
            (
                Some(429),
                r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#,
                Billing,
            ),
            (
                Some(400),
                r#"{"error":{"message":"Invalid prompt: your prompt was flagged as potentially violating our usage policy.","type":"invalid_request_error","param":null,"code":"invalid_prompt"}}"#,
                InvalidRequest,
            ),
            (
                Some(400),
                r#"{"error":{"message":"Invalid 'tools[0].name': string does not match pattern.","type":"invalid_request_error","param":"tools[0].name","code":"invalid_value"}}"#,
                InvalidRequest,
            ),
            // Temporary failures have no kind.
            (
                Some(429),
                r#"{"error":{"message":"Rate limit reached for gpt-5.1 in organization org-AAA on tokens per min (TPM): Limit 30000, Used 22999, Requested 12528. Please try again in 11.054s.","type":"tokens","param":null,"code":"rate_limit_exceeded"}}"#,
                Unknown,
            ),
            (
                None,
                r#"{"type":"error","code":"server_error","message":"The server had an error while processing your request.","param":null,"sequence_number":3}"#,
                Unknown,
            ),
            (Some(500), "{}", Unknown),
            (
                Some(503),
                r#"{"error":{"message":"maximum context length is unrelated here","code":"server_is_overloaded"}}"#,
                Unknown,
            ),
        ],
    );
}

#[test]
fn chatgpt_oauth_backend_errors() {
    // The ChatGPT (Codex) backend: `response.failed` codes Codex treats as
    // fatal (codex-rs/codex-api/src/sse/responses_error.rs) and its 429
    // bodies (codex-rs/codex-api/src/api_bridge.rs).
    assert_kinds(
        "OpenAI stream error",
        &[
            (
                None,
                r#"{"status":"failed","error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again."}}"#,
                ContextOverflow,
            ),
            (
                None,
                r#"{"error":{"code":"credit_balance_exhausted","message":"m"}}"#,
                Billing,
            ),
            (
                None,
                r#"{"error":{"code":"usage_not_included","message":"m"}}"#,
                Billing,
            ),
            (
                Some(429),
                r#"{"error":{"type":"organization_usage_limit_exceeded","message":"m"}}"#,
                Billing,
            ),
            (
                None,
                r#"{"error":{"code":"cyber_policy","message":"This request was flagged for cyber policy."}}"#,
                Refusal,
            ),
        ],
    );
    // A token the backend rejects, and the refresh codes of the Codex login
    // (codex-rs/login/src/auth/manager.rs).
    assert_kinds(
        "OpenAI request failed",
        &[
            (
                Some(401),
                r#"{"error":{"message":"Provided authentication token is expired. Please try signing in again.","type":null,"code":"token_expired","param":null},"status":401}"#,
                Authentication,
            ),
            (
                Some(401),
                r#"{"error":{"code":"refresh_token_reused","message":"Your refresh token has already been used to generate a new access token. Please try signing in again."}}"#,
                Authentication,
            ),
            (Some(400), r#"{"error":"invalid_grant"}"#, Authentication),
        ],
    );
    // The plan usage limit stays a retryable limit with no kind.
    let body = r#"{"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"pro","resets_at":1790000000}}"#;
    assert_eq!(kind("OpenAI request failed", Some(429), body), Unknown);
    // arko's own sign-in failures.
    assert_eq!(
        Error::AuthenticationFailed("refresh failed".into()).kind(),
        Authentication
    );
    assert_eq!(Error::MissingApiKey("OpenAI".into()).kind(), Authentication);
}

#[test]
fn anthropic_errors() {
    // https://platform.claude.com/docs/en/api/errors
    assert_kinds(
        "anthropic API error",
        &[
            // https://github.com/anthropics/claude-code/issues/12186
            (
                Some(400),
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208310 tokens > 200000 maximum"},"request_id":"req_011CVQe1dLrGNMJyE1ghG1av"}"#,
                ContextOverflow,
            ),
            // https://github.com/anthropics/claude-code/issues/476
            (
                Some(400),
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"input length and max_tokens exceed context limit: 186433 + 20000 > 200000, decrease input"}}"#,
                ContextOverflow,
            ),
            (
                Some(400),
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"This model does not support assistant message prefill. The conversation must end with a user message."},"request_id":"req_1"}"#,
                InvalidRequest,
            ),
            (
                Some(401),
                r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"},"request_id":"req_1"}"#,
                Authentication,
            ),
            (
                Some(402),
                r#"{"type":"error","error":{"type":"billing_error","message":"m"},"request_id":"req_1"}"#,
                Billing,
            ),
            (
                Some(403),
                r#"{"type":"error","error":{"type":"permission_error","message":"m"},"request_id":"req_1"}"#,
                Permission,
            ),
            (
                Some(404),
                r#"{"type":"error","error":{"type":"not_found_error","message":"The requested resource could not be found."},"request_id":"req_1"}"#,
                NotFound,
            ),
            (
                Some(413),
                r#"{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."},"request_id":"req_1"}"#,
                RequestTooLarge,
            ),
            // Cloudflare's 413 has no JSON body.
            (
                Some(413),
                "<html><head><title>413 Request Entity Too Large</title></head></html>",
                RequestTooLarge,
            ),
            (
                Some(429),
                r#"{"type":"error","error":{"type":"rate_limit_error","message":"m"},"request_id":"req_1"}"#,
                Unknown,
            ),
            (
                Some(529),
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_1"}"#,
                Unknown,
            ),
        ],
    );
    // The same error as a stream event after the HTTP 200 (anthropic.rs puts
    // the message in the label).
    let event = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 1000500 tokens > 1000000 maximum"}}"#;
    assert_eq!(
        kind(
            "Anthropic stream error: prompt is too long: 1000500 tokens > 1000000 maximum",
            None,
            event
        ),
        ContextOverflow
    );
}

#[test]
fn gemini_errors() {
    assert_kinds(
        "google API error",
        &[
            // https://github.com/google-gemini/gemini-cli/issues/10393
            (
                Some(400),
                r#"{"error":{"code":400,"message":"The input token count (185586) exceeds the maximum number of tokens allowed (131072).","status":"INVALID_ARGUMENT"}}"#,
                ContextOverflow,
            ),
            // An invalid key is a 400, told apart by its message and reason.
            (
                Some(400),
                r#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID","domain":"googleapis.com"}]}}"#,
                Authentication,
            ),
            (
                Some(400),
                r#"{"error":{"code":400,"message":"Invalid JSON payload received.","status":"INVALID_ARGUMENT"}}"#,
                InvalidRequest,
            ),
            (
                Some(403),
                r#"{"error":{"code":403,"message":"Your API key was reported as leaked. Please use another API key.","status":"PERMISSION_DENIED"}}"#,
                Permission,
            ),
            (
                Some(404),
                r#"{"error":{"code":404,"message":"models/gemini-pro is not found for API version v1beta, or is not supported for generateContent.","status":"NOT_FOUND"}}"#,
                NotFound,
            ),
            // Interactions API codes (https://ai.google.dev/gemini-api/docs/api-errors).
            (
                Some(401),
                r#"{"error":{"code":"authentication","message":"The API key is missing, invalid, or expired."}}"#,
                Authentication,
            ),
            (
                Some(402),
                r#"{"error":{"code":"payment_required","message":"Your Prepay credit balance is depleted."}}"#,
                Billing,
            ),
            (
                Some(429),
                r#"{"error":{"code":"quota_exceeded","message":"You have exceeded your daily quota."}}"#,
                Billing,
            ),
            (
                Some(400),
                r#"{"error":{"code":"invalid_request","message":"The value 'invalid_tool_type_xyz' is not supported for 'type' at 'tools[0]'."}}"#,
                InvalidRequest,
            ),
            (
                Some(429),
                r#"{"error":{"code":"rate_limit_exceeded","message":"m"}}"#,
                Unknown,
            ),
            (
                Some(503),
                r#"{"error":{"code":"service_unavailable","message":"The service is temporarily overloaded or down."}}"#,
                Unknown,
            ),
        ],
    );
    // A stream error event (no HTTP status) keeps its kind.
    let event = r#"{"error":{"code":"model_not_found","message":"m"}}"#;
    assert_eq!(kind("google stream error", None, event), NotFound);
    let event = r#"{"error":{"code":"invalid_request","message":"The input token count (1100000) exceeds the maximum number of tokens allowed (1048576)."}}"#;
    assert_eq!(kind("google stream error", None, event), ContextOverflow);
}

#[test]
fn openrouter_errors() {
    // https://openrouter.ai/docs/api/reference/errors-and-debugging
    let typed = |status: u16, error_type: &str, message: &str| {
        format!(
            r#"{{"error":{{"code":{status},"message":"{message}","metadata":{{"error_type":"{error_type}","provider_name":"X"}}}}}}"#
        )
    };
    for (status, error_type, expected) in [
        (400, "context_length_exceeded", ContextOverflow),
        (400, "token_limit_exceeded", ContextOverflow),
        (400, "string_too_long", RequestTooLarge),
        (413, "payload_too_large", RequestTooLarge),
        (401, "authentication", Authentication),
        (403, "permission_denied", Permission),
        (402, "payment_required", Billing),
        (404, "not_found", NotFound),
        (403, "content_policy_violation", Refusal),
        (422, "unprocessable", InvalidRequest),
        (400, "invalid_request", InvalidRequest),
        (400, "max_tokens_exceeded", InvalidRequest),
        (400, "invalid_image", InvalidRequest),
        (429, "rate_limit_exceeded", Unknown),
        (503, "provider_overloaded", Unknown),
        (502, "provider_unavailable", Unknown),
    ] {
        let body = typed(status, error_type, "m");
        // Before the stream (with the HTTP status) and mid-stream (without).
        for http in [Some(status), None] {
            assert_eq!(
                kind("OpenRouter request failed", http, &body),
                expected,
                "{error_type} {http:?}"
            );
        }
    }
    // An upstream message without a typed error
    // (https://github.com/RooCodeInc/Roo-Code/issues/11998).
    let body = r#"{"error":{"code":400,"message":"This endpoint's maximum context length is 256000 tokens. However, you requested about 256493 tokens (194895 of text input, 10398 of tool input, 51200 in the output). Please reduce the length of either one, or use the \"middle-out\" transform to compress your prompt automatically."}}"#;
    assert_eq!(
        kind("OpenRouter request failed", Some(400), body),
        ContextOverflow
    );
    // The codes openrouter.rs sets itself.
    let mut payment = Error::provider_error(
        "OpenRouter request failed",
        Some(402),
        Some(Duration::from_secs(3)),
        r#"{"error":{"code":402,"message":"Insufficient credits"}}"#,
    );
    if let Error::ProviderError { code, .. } = &mut payment {
        *code = Some("payment_required".into());
    }
    // A billing wall even when the in-flight budget clears by itself.
    assert!(payment.is_retryable());
    assert_eq!(payment.kind(), Billing);
    let mut unroutable = Error::provider_error(
        "OpenRouter request failed",
        Some(404),
        None,
        r#"{"error":{"code":404,"message":"No endpoints found that can handle the requested parameters."}}"#,
    );
    if let Error::ProviderError { code, .. } = &mut unroutable {
        *code = Some("no_eligible_provider".into());
    }
    assert_eq!(unroutable.kind(), NotFound);
}

#[test]
fn claude_code_cli_errors() {
    // The assistant message's `error` values (the Claude Agent SDK's
    // `SDKAssistantMessageError`), the result's `api_error_status` and
    // `terminal_reason`, and the result text.
    let cases = [
        (
            Some("authentication_failed"),
            Some(401),
            None,
            "Invalid API key · Please run /login",
            Authentication,
        ),
        (
            Some("authentication_failed"),
            None,
            None,
            "",
            Authentication,
        ),
        (None, Some(401), None, "", Authentication),
        (Some("oauth_org_not_allowed"), None, None, "", Permission),
        (Some("billing_error"), Some(400), None, "", Billing),
        (Some("model_not_found"), Some(404), None, "", NotFound),
        (
            None,
            None,
            Some("prompt_too_long"),
            "Prompt is too long",
            ContextOverflow,
        ),
        (
            Some("invalid_request"),
            Some(400),
            None,
            r#"API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208310 tokens > 200000 maximum"}}"#,
            ContextOverflow,
        ),
        (
            Some("invalid_request"),
            Some(400),
            None,
            "API Error: 400",
            InvalidRequest,
        ),
        (
            Some("rate_limit"),
            Some(429),
            None,
            "API Error: 429",
            Unknown,
        ),
        (Some("overloaded"), Some(529), None, "", Unknown),
        (Some("server_error"), Some(500), None, "", Unknown),
        (Some("max_output_tokens"), None, None, "", Unknown),
        (Some("unknown"), None, None, "something failed", Unknown),
        (None, None, None, "Claude Code exited", Unknown),
    ];
    for (error, status, terminal_reason, text, expected) in cases {
        assert_eq!(
            ErrorKind::from_claude_code(error, status, terminal_reason, text),
            expected,
            "{error:?} {status:?} {terminal_reason:?} {text}"
        );
    }
}

#[test]
fn arko_errors_and_retry_classes_are_unchanged() {
    assert_eq!(Error::InvalidRequest("x".into()).kind(), InvalidRequest);
    assert_eq!(Error::ModelNotSupported("x".into()).kind(), Unsupported);
    assert_eq!(Error::UnsupportedFeature("x".into()).kind(), Unsupported);
    assert_eq!(Error::ApiError("x".into()).kind(), Unknown);
    assert_eq!(Error::StreamTruncated("x".into()).kind(), Unknown);
    assert_eq!(
        Error::Refusal(Box::new(crate::stop::Refusal::new(
            crate::stop::RefusalSource::Model
        )))
        .kind(),
        Refusal
    );
    assert_eq!(
        Error::UnexpectedToolCalls {
            expected: "t".into(),
            stop_reason: StopReason::ContextWindowExceeded,
            found: vec![],
        }
        .kind(),
        ContextOverflow
    );
    // Every permanent code has a kind other than Unknown.
    for code in PERMANENT_CODES {
        let body = format!(r#"{{"error":{{"code":"{code}"}}}}"#);
        let error = Error::provider_error("stream", None, None, &body);
        assert_eq!(error.retry_class(), RetryClass::Permanent, "{code}");
        assert_ne!(error.kind(), Unknown, "{code}");
    }
    // A kind a later version adds loads as Unknown; known kinds round-trip.
    assert_eq!(
        serde_json::from_str::<ErrorKind>(r#""from_the_future""#).unwrap(),
        Unknown
    );
    for kind in [
        Authentication,
        Permission,
        Billing,
        ContextOverflow,
        RequestTooLarge,
        InvalidRequest,
        NotFound,
        Refusal,
        Unsupported,
        Unknown,
    ] {
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(serde_json::from_str::<ErrorKind>(&json).unwrap(), kind);
        assert_eq!(json, format!(r#""{}""#, kind.as_str()));
    }
    assert_eq!(
        serde_json::to_string(&ContextOverflow).unwrap(),
        r#""context_overflow""#
    );
}
