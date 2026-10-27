use super::{GenerationResponse, ProviderClient, StreamEvent, take_refused_output};
use crate::{
    config::{GenerationConfig, OpenAIServiceTier},
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::{ModelType, Provider as ProviderType},
    pricing::ServiceTier,
    reasoning::{OpenAIEffort, OpenAIReasoningMode, ReasoningArtifact},
    retry::BackoffPolicy,
    stop::{Refusal, RefusalSource, StopReason},
    tool::{ToolCall, ToolChoice},
    trace::{RequestTrace, TraceCall},
    usage::{BillingKind, ModelCallUsage, UsageInfo},
};
use async_stream::try_stream;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use reqwest::header::{ACCEPT, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::{OwnedRwLockWriteGuard, RwLock};

#[cfg(test)]
mod oauth_recovery_tests;

#[cfg(test)]
mod refusal_tests;

#[cfg(test)]
mod summaries_tests;

#[cfg(test)]
mod trace_tests;

#[cfg(test)]
mod codex_parity_tests;

#[cfg(test)]
mod reasoning_stream_tests;

mod request_hook;
pub use request_hook::OpenAIRequestHook;

mod credential_store;
pub use credential_store::ChatGptOAuthStore;

mod accounts;
pub use accounts::{
    CHATGPT_USAGE_URL, ChatGptAccount, ChatGptAccountEvent, ChatGptAccountStatus, ChatGptAccounts,
    ChatGptUsageStatus, DEFAULT_STATUS_CHECK_INTERVAL, SwitchReason, UsableReason,
};

pub(crate) mod spent_tokens;

type StreamingRefusals = BTreeMap<(String, u64), String>;

const CHATGPT_OAUTH_DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CHATGPT_OAUTH_DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// Default Responses API base URLs. Requests go to `{base}/responses`.
const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// ChatGPT backend sticky-routing token. See [`Conversation::openai_turn_state`].
const TURN_STATE_HEADER: &str = "x-codex-turn-state";
/// The ChatGPT backend's routing hint, `model=<model>` plus `;tier=<tier>`
/// when a tier is sent, which Codex sends on every Codex-backend request
/// (openai/codex `codex-rs/core/src/client.rs`, `build_routing_hint_header`).
const ROUTING_HINT_HEADER: &str = "x-codex-routing-hint";
/// The models whose Codex catalog entry offers Fast mode (`service_tiers`
/// with id `priority`, "2x speed, increased usage"): the only ones Codex
/// sends `service_tier: "priority"` for over ChatGPT sign-in
/// (`ModelInfo::service_tier_for_request` in
/// `codex-rs/protocol/src/openai_models.rs`). From
/// `codex-rs/models-manager/models.json` at openai/codex ac9b5b8, less
/// Codex's hidden approval-review model (`codex-auto-review`).
const CHATGPT_FAST_MODE_MODELS: &[&str] = &[
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
];

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatGptOAuthCredentials {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub account_id: String,
}

/// Stands in for a secret in `Debug` output.
const REDACTED: &str = "<redacted>";

fn redacted(secret: Option<&str>) -> Option<&'static str> {
    secret.map(|_| REDACTED)
}

// Tokens and the account id never appear in `Debug` output (logs, panics,
// error reports).
impl std::fmt::Debug for ChatGptOAuthCredentials {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatGptOAuthCredentials")
            .field("access_token", &REDACTED)
            .field("refresh_token", &redacted(self.refresh_token.as_deref()))
            .field("account_id", &REDACTED)
            .finish()
    }
}

#[derive(Clone)]
pub enum OpenAIAuth {
    ApiKey(String),
    ChatGptOAuth(ChatGptOAuthCredentials),
}

impl OpenAIAuth {
    fn is_chatgpt_oauth(&self) -> bool {
        matches!(self, OpenAIAuth::ChatGptOAuth(_))
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OAuthRefreshResult {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
}

impl std::fmt::Debug for OAuthRefreshResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OAuthRefreshResult")
            .field("access_token", &REDACTED)
            .field("refresh_token", &redacted(self.refresh_token.as_deref()))
            .field("id_token", &redacted(self.id_token.as_deref()))
            .finish()
    }
}

struct StreamingToolCall {
    /// Position in `ToolCallStart`/`ToolCallArgumentsDelta` events.
    index: usize,
    name: String,
    arguments: String,
    item_id: Option<String>,
    call_id: Option<String>,
    output_index: Option<u64>,
    /// Set once the provider reports the arguments as finished.
    completed: bool,
}

impl StreamingToolCall {
    /// The call an argument event belongs to, matched by `item_id` and then
    /// `output_index`, so interleaved parallel calls stay separate.
    fn find<'a>(calls: &'a mut [Self], event: &Value) -> Option<&'a mut Self> {
        let item_id = event.get("item_id").and_then(Value::as_str);
        let output_index = event.get("output_index").and_then(Value::as_u64);
        let position = calls
            .iter()
            .position(|call| item_id.is_some() && call.item_id.as_deref() == item_id)
            .or_else(|| {
                calls
                    .iter()
                    .position(|call| output_index.is_some() && call.output_index == output_index)
            })
            // Events without either identifier can only mean the latest call.
            .or_else(|| {
                (item_id.is_none() && output_index.is_none())
                    .then(|| calls.len().checked_sub(1))
                    .flatten()
            })?;
        calls.get_mut(position)
    }

    fn into_tool_call(self) -> ToolCall {
        let mut call = ToolCall::from_argument_text(self.call_id, self.name, &self.arguments);
        call.provider_id = self.item_id;
        call
    }

    /// The `function_call` output item for a call whose item never arrived.
    fn replay_item(&self) -> Value {
        let mut item = serde_json::json!({
            "type": "function_call", "call_id": self.call_id, "name": self.name,
            "arguments": self.arguments,
        });
        if let Some(id) = &self.item_id {
            item["id"] = Value::String(id.clone());
        }
        item
    }
}

/// What a streamed response delivered besides its terminal event.
#[derive(Default)]
struct StreamedOutput {
    /// `response.output_item.done` items with their `output_index`, in
    /// arrival order.
    items: Vec<(Option<u64>, Value)>,
    refusals: StreamingRefusals,
    /// Calls whose arguments finished streaming.
    calls: Vec<StreamingToolCall>,
    text: String,
    /// The model named by `response.created`.
    created_model: Option<String>,
}

/// What a response's HTTP headers say about the call.
#[derive(Debug, Default)]
struct ResponseHead {
    /// The `model` the request asked for.
    requested_model: String,
    /// `openai-model` (or `x-openai-model`) header.
    served_model: Option<String>,
    /// `x-codex-turn-state` header.
    turn_state: Option<String>,
    /// The ChatGPT backend's `x-codex-*` plan usage headers.
    plan_usage: Option<crate::usage::PlanUsage>,
}

impl ResponseHead {
    fn new(requested_model: &str, headers: &HeaderMap) -> Self {
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(String::from)
        };
        Self {
            requested_model: requested_model.to_string(),
            served_model: header("openai-model").or_else(|| header("x-openai-model")),
            turn_state: header(TURN_STATE_HEADER),
            plan_usage: crate::usage::PlanUsage::from_codex_headers(headers),
        }
    }
}

fn base64_url_decode(input: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(input.trim_end_matches('=')).ok()
}

pub fn parse_jwt_expiry(token: &str) -> Option<u64> {
    let mut parts = token.splitn(4, '.');
    let _header = parts.next()?;
    let payload_b64 = parts.next()?;
    let _signature = parts.next()?;

    if parts.next().is_some() {
        return None;
    }

    let payload_bytes = base64_url_decode(payload_b64)?;
    let claims: Value = serde_json::from_slice(&payload_bytes).ok()?;
    claims.get("exp")?.as_u64()
}

pub fn is_token_expired(token: &str, margin: std::time::Duration) -> bool {
    let Some(exp) = parse_jwt_expiry(token) else {
        return true;
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    exp <= now + margin.as_secs()
}

fn retry_after_from_headers(response: &reqwest::Response) -> Option<Duration> {
    crate::error::retry_after_from_headers(response.status().as_u16(), response.headers())
}

fn response_error(
    label: &str,
    status: reqwest::StatusCode,
    header_delay: Option<Duration>,
    body: &str,
) -> Error {
    Error::provider_error(label, Some(status.as_u16()), header_delay, body)
}

/// Whether an error body (of a 400) refuses the request's reasoning
/// summaries, as OpenAI does for an organization that isn't verified for
/// them: "Your organization must be verified to generate reasoning
/// summaries. ...", with `param` `reasoning.summary`.
fn refuses_reasoning_summaries(body: &str) -> bool {
    let Ok(body) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let error = &body["error"];
    error["param"].as_str() == Some("reasoning.summary")
        || error["message"]
            .as_str()
            .is_some_and(|message| message.to_lowercase().contains("reasoning summaries"))
}

/// Retries of a rate-limited OAuth refresh when rate-limit retries are
/// enabled: about an hour of backoff before the 429 is returned.
const OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES: u32 = 8;

/// Timing of the opt-in OAuth refresh rate-limit loop.
#[derive(Debug, Clone, Copy)]
struct OAuthRefreshRetry {
    /// 30 s doubling to a 15-minute cap. A server hint is a minimum.
    backoff: BackoffPolicy,
    /// The longest server hint one wait honors (the backoff's own cap), far
    /// below [`MAX_SERVER_DELAY_HINT`](crate::retry::MAX_SERVER_DELAY_HINT):
    /// each retry reads the store again, and another program may refresh
    /// meanwhile.
    max_server_hint: Duration,
}

const OAUTH_REFRESH_RETRY: OAuthRefreshRetry = OAuthRefreshRetry {
    backoff: BackoffPolicy::new(Duration::from_secs(30), Duration::from_secs(900)),
    max_server_hint: Duration::from_secs(900),
};

impl OAuthRefreshRetry {
    /// The wait before rate-limit retry `attempt` (0 for the first), for a
    /// uniformly random `sample` (see [`crate::retry::random_sample`]).
    fn delay(&self, server_hint: Option<Duration>, attempt: u32, sample: u64) -> Duration {
        let server_hint = server_hint.map(|hint| hint.min(self.max_server_hint));
        self.backoff
            .delay(attempt.saturating_add(1), server_hint, sample)
    }
}

fn is_rate_limited_refresh(error: &Error) -> bool {
    matches!(
        error,
        Error::ProviderError {
            status: Some(429),
            ..
        }
    )
}

/// How long one OAuth refresh request may take, from connecting to the end
/// of the response body. The shared HTTP client has no total deadline (a
/// generation can stream for a long time), but a refresh holds the
/// credentials while it runs, so a stalled token endpoint must not hold them
/// for long. A timeout is a transient [`Error::HttpError`].
///
/// The trade-off: Codex sets no timeout here. A server that rotates the
/// token but answers after 60 s has its answer dropped, so that rotation is
/// lost and the next refresh with the old token is refused, just as when the
/// connection drops mid-answer. The timeout exists so a stalled endpoint
/// can't hold the credential locks (and every request waiting on them)
/// indefinitely.
const OAUTH_REFRESH_TIMEOUT: Duration = Duration::from_secs(60);

/// Exchanges a ChatGPT OAuth refresh token for new tokens, once. An HTTP
/// failure (including a 429) is returned to the caller with its retry
/// metadata. `client_id` and `token_url` default to the Codex CLI's. The
/// request times out after 60 seconds (a transient [`Error::HttpError`]).
///
/// Build `http_client` with [`crate::http_client`] or
/// [`crate::http_client_builder`]. `reqwest::Client::new()` panics in
/// applications that don't install a process-wide rustls crypto provider,
/// since this crate builds reqwest without one.
///
/// The refresh token rotates. Once this succeeds, the old one stops working,
/// so store the returned tokens (see [`crate::CodexAuthFile`]). A later call
/// (or any other refresh in the process) with a refresh token that a refresh
/// here already exchanged for a new one, or that the server refused as
/// reused, expired or revoked, fails with
/// [`Error::AuthenticationFailed`] without a request (a second use would
/// sign the user out). Refresh tokens are unique, so this holds whatever
/// `token_url` either refresh used.
pub async fn refresh_chatgpt_oauth_token(
    http_client: &reqwest::Client,
    refresh_token: &str,
    client_id: Option<&str>,
    token_url: Option<&str>,
) -> Result<OAuthRefreshResult, Error> {
    let timeout = Some(OAUTH_REFRESH_TIMEOUT);
    request_oauth_refresh(
        http_client,
        refresh_token,
        None,
        client_id,
        token_url,
        timeout,
    )
    .await
}

/// [`refresh_chatgpt_oauth_token`] with the request timing out after
/// `timeout`, if given. `account_id`, when known, is the account the refresh
/// token belongs to (a refusal names that account's recovery file, if any).
async fn request_oauth_refresh(
    http_client: &reqwest::Client,
    refresh_token: &str,
    account_id: Option<&str>,
    client_id: Option<&str>,
    token_url: Option<&str>,
    timeout: Option<Duration>,
) -> Result<OAuthRefreshResult, Error> {
    let resolved_client_id = client_id.unwrap_or(CHATGPT_OAUTH_DEFAULT_CLIENT_ID);
    let resolved_token_url = token_url.unwrap_or(CHATGPT_OAUTH_DEFAULT_TOKEN_URL);
    // Every refresh request passes here: a token this process knows is spent
    // is never sent again.
    if spent_tokens::is_spent(refresh_token) {
        return Err(crate::codex_auth::spent_refresh_token_error(account_id));
    }

    let body = serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": resolved_client_id,
        "scope": "openid profile email offline_access",
    });

    let mut request = http_client
        .post(resolved_token_url)
        .header("Content-Type", "application/json");
    if let Some(timeout) = timeout {
        request = request.timeout(timeout);
    }
    let response = request.json(&body).send().await?;
    let status = response.status();
    if !status.is_success() {
        let header_delay = retry_after_from_headers(&response);
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable error body>".into());
        // Reused, expired or revoked: the token can never be used again.
        if spent_tokens::refused_for_good(&error_text) {
            spent_tokens::record(refresh_token);
        }
        return Err(response_error(
            "OAuth token refresh failed",
            status,
            header_delay,
            &error_text,
        ));
    }

    let data: Value = response.json().await?;

    let access_token = data["access_token"]
        .as_str()
        .ok_or_else(|| {
            Error::ApiError("Missing access_token in OAuth refresh response".to_string())
        })?
        .to_string();

    let result = OAuthRefreshResult {
        access_token,
        refresh_token: data["refresh_token"].as_str().map(String::from),
        id_token: data["id_token"].as_str().map(String::from),
    };
    // The server rotated the refresh token: the one sent is spent. (Without a
    // new one in the answer, the old one stays in use.)
    if result
        .refresh_token
        .as_deref()
        .is_some_and(|new| new != refresh_token)
    {
        spent_tokens::record(refresh_token);
    }
    Ok(result)
}

#[derive(Clone)]
pub struct OpenAIClient {
    auth: Arc<RwLock<OpenAIAuth>>,
    http_client: reqwest::Client,
    retry_oauth_rate_limits: bool,
    base_url: Option<String>,
    oauth_token_url: Option<String>,
    trace: Option<RequestTrace>,
    request_hook: Option<Arc<dyn OpenAIRequestHook>>,
    originator: Option<String>,
    user_agent: Option<String>,
    credential_store: Option<Arc<dyn ChatGptOAuthStore>>,
    /// See [`with_stream_idle_timeout`](Self::with_stream_idle_timeout).
    stream_idle_timeout: Option<Duration>,
    /// Warn (once) when a refresh rotates tokens that no store keeps: set for
    /// credentials taken from the environment by `LLMClient::from_env`.
    warn_unsaved_rotation: bool,
    /// Counts replacements of `auth`, shared by clones. Read before waiting
    /// for the credential lock, it tells a refresh whether the credentials
    /// were replaced while it waited.
    auth_epoch: Arc<AtomicU64>,
    /// [`OAUTH_REFRESH_TIMEOUT`]. Tests shorten it, and turn it off under a
    /// paused clock (which jumps ahead while test I/O is under way).
    oauth_refresh_timeout: Option<Duration>,
    /// Timing of the opt-in rate-limit retries of a refresh. Tests shorten it.
    oauth_refresh_retry: OAuthRefreshRetry,
    /// The organization refused reasoning summaries (it isn't verified for
    /// them): requests ask for none from then on. Shared by clones.
    summaries_refused: Arc<AtomicBool>,
}

/// What a refresh replaces. Checked once the credential lock is held.
#[derive(Clone)]
enum RefreshTarget {
    /// The credentials held when [`auth_epoch`](OpenAIClient::auth_epoch) had
    /// this value: an explicit refresh.
    Since(u64),
    /// The credentials a request was rejected with (HTTP 401 `token_expired`).
    Rejected(ChatGptOAuthCredentials),
    /// The credentials a request was rejected with by any other HTTP 401:
    /// replaced only by newer ones from the store, never refreshed.
    Reload(ChatGptOAuthCredentials),
    /// Whatever the client holds, if its access token expires within this
    /// margin (after the store has been read).
    Expiring(Duration),
}

impl OpenAIClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            auth: Arc::new(RwLock::new(OpenAIAuth::ApiKey(api_key.to_string()))),
            http_client: super::default_http_client(),
            retry_oauth_rate_limits: false,
            base_url: None,
            oauth_token_url: None,
            trace: None,
            request_hook: None,
            originator: None,
            user_agent: None,
            credential_store: None,
            warn_unsaved_rotation: false,
            stream_idle_timeout: None,
            auth_epoch: Arc::new(AtomicU64::new(0)),
            oauth_refresh_timeout: Some(OAUTH_REFRESH_TIMEOUT),
            oauth_refresh_retry: OAUTH_REFRESH_RETRY,
            summaries_refused: Arc::default(),
        }
    }

    pub fn new_chatgpt_oauth(
        access_token: &str,
        refresh_token: Option<&str>,
        account_id: &str,
    ) -> Self {
        Self {
            auth: Arc::new(RwLock::new(OpenAIAuth::ChatGptOAuth(
                ChatGptOAuthCredentials {
                    access_token: access_token.to_string(),
                    refresh_token: refresh_token.map(String::from),
                    account_id: account_id.to_string(),
                },
            ))),
            http_client: super::default_http_client(),
            retry_oauth_rate_limits: false,
            base_url: None,
            oauth_token_url: None,
            trace: None,
            request_hook: None,
            originator: None,
            user_agent: None,
            credential_store: None,
            warn_unsaved_rotation: false,
            stream_idle_timeout: None,
            auth_epoch: Arc::new(AtomicU64::new(0)),
            oauth_refresh_timeout: Some(OAUTH_REFRESH_TIMEOUT),
            oauth_refresh_retry: OAUTH_REFRESH_RETRY,
            summaries_refused: Arc::default(),
        }
    }

    pub fn with_auth(auth: OpenAIAuth) -> Self {
        Self {
            auth: Arc::new(RwLock::new(auth)),
            http_client: super::default_http_client(),
            retry_oauth_rate_limits: false,
            base_url: None,
            oauth_token_url: None,
            trace: None,
            request_hook: None,
            originator: None,
            user_agent: None,
            credential_store: None,
            warn_unsaved_rotation: false,
            stream_idle_timeout: None,
            auth_epoch: Arc::new(AtomicU64::new(0)),
            oauth_refresh_timeout: Some(OAUTH_REFRESH_TIMEOUT),
            oauth_refresh_retry: OAUTH_REFRESH_RETRY,
            summaries_refused: Arc::default(),
        }
    }

    /// This client with `signed_in`'s credentials (and what belongs to
    /// them: its store, its epoch, whether summaries are refused), keeping
    /// this client's settings. For an entry of a ChatGPT account list that
    /// signs in as another account.
    pub(crate) fn with_sign_in_of(&self, signed_in: &OpenAIClient) -> OpenAIClient {
        OpenAIClient {
            auth: signed_in.auth.clone(),
            credential_store: signed_in.credential_store.clone(),
            warn_unsaved_rotation: signed_in.warn_unsaved_rotation,
            auth_epoch: signed_in.auth_epoch.clone(),
            summaries_refused: signed_in.summaries_refused.clone(),
            ..self.clone()
        }
    }

    /// Sends Responses requests to `{base_url}/responses` instead of the
    /// default for the auth mode (`https://api.openai.com/v1` for API keys,
    /// `https://chatgpt.com/backend-api/codex` for ChatGPT OAuth), for example
    /// a proxy or a test server.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sends ChatGPT OAuth token refreshes to `url` instead of
    /// `https://auth.openai.com/oauth/token`.
    pub fn with_oauth_token_url(mut self, url: impl Into<String>) -> Self {
        self.oauth_token_url = Some(url.into());
        self
    }

    /// Configures whether an explicit
    /// [`refresh_oauth_token`](Self::refresh_oauth_token) call retries HTTP
    /// 429 responses (off by default). When enabled, it retries up to 8 times,
    /// waiting for the advertised reset (at most 15 minutes per wait) when
    /// available and otherwise backing off from 30 seconds to 15 minutes,
    /// before the 429 is returned. No lock is held while waiting, so other
    /// callers keep using and refreshing the credentials. A retry finding
    /// them replaced returns them instead of refreshing.
    ///
    /// The refreshes requests make on their own
    /// (after a request rejected as expired, and
    /// [`refresh_oauth_token_if_expiring`](Self::refresh_oauth_token_if_expiring),
    /// which the arko_agents runtime calls before each request) and
    /// [`refresh_oauth_token_once`](Self::refresh_oauth_token_once) never
    /// wait. A 429 from them is returned as the request's error, which an
    /// agent's own rate-limit recovery then handles. So turning this on for
    /// a client that only runs agents changes nothing.
    pub fn with_oauth_rate_limit_retries(mut self, enabled: bool) -> Self {
        self.retry_oauth_rate_limits = enabled;
        self
    }

    /// Keeps ChatGPT OAuth credentials in step with `store`, which other
    /// programs share (for a Codex `auth.json`, see
    /// [`CodexAuthFile::openai_client`](crate::codex_auth::CodexAuthFile::openai_client)).
    /// Every refresh first re-reads the store and adopts tokens another
    /// program already refreshed, and rotated tokens are stored as soon as a
    /// refresh succeeds. A request rejected with any other HTTP 401 re-reads
    /// the store too and is sent once more if it holds newer tokens. See
    /// [`ChatGptOAuthStore`]. Clones share the store.
    pub fn with_credential_store(mut self, store: Option<Arc<dyn ChatGptOAuthStore>>) -> Self {
        self.credential_store = store;
        self
    }

    /// Logs a warning (once per process) when a refresh rotates tokens that
    /// no credential store keeps, for credentials whose source (the
    /// environment) can't be updated.
    pub(crate) fn with_unsaved_rotation_warning(mut self) -> Self {
        self.warn_unsaved_rotation = true;
        self
    }

    /// Whether a credential store is installed.
    #[cfg(test)]
    pub(crate) fn has_credential_store(&self) -> bool {
        self.credential_store.is_some()
    }

    /// Whether [`with_unsaved_rotation_warning`](Self::with_unsaved_rotation_warning) was set.
    #[cfg(test)]
    pub(crate) fn warns_unsaved_rotation(&self) -> bool {
        self.warn_unsaved_rotation
    }

    /// Cuts a streamed response off (including every ChatGPT OAuth response,
    /// since that backend always streams), with a transient
    /// [`Error::StreamTruncated`], once no data has arrived for `timeout`.
    /// `None` (the default) waits as long as the connection stays open. See
    /// [`LLMClient::with_stream_idle_timeout`](crate::client::LLMClient::with_stream_idle_timeout)
    /// before setting one. Models can think for a long time without sending
    /// anything, so a value must be generous.
    pub fn with_stream_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    /// Writes every request and response to `trace`. See [`RequestTrace`].
    pub fn with_request_trace(mut self, trace: Option<RequestTrace>) -> Self {
        self.trace = trace;
        self
    }

    /// Installs a hook that can adjust each request. See [`OpenAIRequestHook`].
    pub fn with_request_hook(mut self, hook: Option<Arc<dyn OpenAIRequestHook>>) -> Self {
        self.request_hook = hook;
        self
    }

    /// Sends `originator: <originator>` with every request, naming your
    /// application to the ChatGPT backend. Nothing is sent by default. Use
    /// your own application's name, never another client's.
    pub fn with_originator(mut self, originator: Option<String>) -> Self {
        self.originator = originator;
        self
    }

    /// Sends `User-Agent: <user_agent>` with every request. Nothing is sent
    /// by default.
    pub fn with_user_agent(mut self, user_agent: Option<String>) -> Self {
        self.user_agent = user_agent;
        self
    }

    pub async fn update_auth(&self, auth: OpenAIAuth) {
        let mut slot = self.auth.write().await;
        self.install_auth(&mut slot, auth);
    }

    /// Replaces the credentials. `slot` is the locked `auth`.
    fn install_auth(&self, slot: &mut OpenAIAuth, auth: OpenAIAuth) {
        *slot = auth;
        self.auth_epoch.fetch_add(1, Ordering::SeqCst);
    }

    pub async fn is_chatgpt_oauth(&self) -> bool {
        self.auth.read().await.is_chatgpt_oauth()
    }

    /// Returns a snapshot of the current ChatGPT OAuth credentials.
    ///
    /// The snapshot includes any access or refresh token installed by the
    /// most recent successful refresh. A refresh in flight on this client or
    /// a clone is waited for because it holds the credentials until its new
    /// tokens are installed and stored (see [`refresh_oauth_token`](Self::refresh_oauth_token)
    /// for awaiting this before a runtime shutdown). API-key clients return
    /// `None`.
    pub async fn chatgpt_oauth_credentials(&self) -> Option<ChatGptOAuthCredentials> {
        match &*self.auth.read().await {
            OpenAIAuth::ChatGptOAuth(credentials) => Some(credentials.clone()),
            OpenAIAuth::ApiKey(_) => None,
        }
    }

    pub async fn is_oauth_token_expired(&self, margin: std::time::Duration) -> bool {
        let auth = self.auth.read().await;

        match &*auth {
            OpenAIAuth::ChatGptOAuth(credentials) => {
                is_token_expired(&credentials.access_token, margin)
            }
            OpenAIAuth::ApiKey(_) => false,
        }
    }

    /// [`refresh_oauth_token`](Self::refresh_oauth_token) with a single
    /// refresh request. An HTTP 429 is returned at once, whatever
    /// [`with_oauth_rate_limit_retries`](Self::with_oauth_rate_limit_retries)
    /// says, so the caller owns retry timing and cancellation.
    pub async fn refresh_oauth_token_once(&self) -> Result<OAuthRefreshResult, Error> {
        self.refresh_explicitly(false).await
    }

    /// Refreshes the ChatGPT OAuth credentials, rotating the refresh token,
    /// and installs the new ones for every clone. With a credential store,
    /// tokens another program already refreshed are adopted instead of
    /// refreshing (see [`ChatGptOAuthStore`]). When the credentials are
    /// replaced while this call waits for them (by a concurrent refresh or
    /// [`update_auth`](Self::update_auth)), those are returned without
    /// another refresh, so concurrent calls rotate once.
    ///
    /// Once the refresh request is sent, the refresh completes (installing
    /// and storing the new tokens) even if this future is dropped. It runs in
    /// a Tokio task of its own. A runtime shutting down still cancels that
    /// task, and rotated tokens it had not stored yet are then lost with the
    /// process (the previous refresh token no longer works). A host that
    /// must not lose them stops starting refreshes (for example, finishes or
    /// cancels its agent runs) and then awaits
    /// [`chatgpt_oauth_credentials`](Self::chatgpt_oauth_credentials)
    /// before shutting the runtime down. A refresh in flight holds the
    /// credential lock until the new tokens are installed and stored, so
    /// that call returns only after it. Clients that don't share the
    /// credentials (such as two built separately from one Codex `auth.json`)
    /// are awaited one by one.
    ///
    /// The request times out after 60 seconds (a transient
    /// [`Error::HttpError`]), so a stalled token endpoint holds the
    /// credentials no longer than that.
    ///
    /// No refresh (here or on any other path) sends a refresh token that a
    /// refresh in this process already exchanged for a new one, or that the
    /// server refused as reused, expired or revoked, nor adopts stored
    /// credentials holding one. That fails with
    /// [`Error::AuthenticationFailed`] (a second use would sign the user
    /// out).
    pub async fn refresh_oauth_token(&self) -> Result<OAuthRefreshResult, Error> {
        self.refresh_explicitly(self.retry_oauth_rate_limits).await
    }

    /// Refreshes the ChatGPT OAuth credentials if the access token expires
    /// within `margin` (a token whose expiry can't be read counts as
    /// expiring), making at most one refresh request (an HTTP 429 is
    /// returned at once, as with [`refresh_oauth_token_once`](Self::refresh_oauth_token_once)).
    ///
    /// The decision is made under the credential lock, after the credential
    /// store (if any) has been read, so concurrent calls on one client, its
    /// clones, or clients sharing a store's
    /// [`refresh_lock`](ChatGptOAuthStore::refresh_lock) refresh once.
    ///
    /// Returns `Ok(Some(_))` when this call installed new credentials, either
    /// refreshed ones or ones another program stored (adopted without a
    /// refresh when they don't expire within `margin`, refreshed otherwise).
    /// Returns `Ok(None)` when the credentials held don't expire within
    /// `margin` (including ones another caller installed while this call
    /// waited), and for API-key clients.
    ///
    /// Credentials without a refresh token can't be refreshed. They are used
    /// until the server rejects them, so this returns `Ok(None)` for them,
    /// even once their access token's expiry has passed. A request the
    /// server then rejects as expired fails with
    /// [`Error::AuthenticationFailed`] saying there is no refresh token.
    pub async fn refresh_oauth_token_if_expiring(
        &self,
        margin: Duration,
    ) -> Result<Option<OAuthRefreshResult>, Error> {
        if !self.is_oauth_token_expired(margin).await {
            return Ok(None);
        }
        self.refresh(RefreshTarget::Expiring(margin), false).await
    }

    async fn refresh_explicitly(
        &self,
        retry_rate_limits: bool,
    ) -> Result<OAuthRefreshResult, Error> {
        // Read before waiting for the refresh locks: a refresh that is under
        // way now replaces the credentials, and its result is returned
        // instead of refreshing again.
        let since = self.auth_epoch.load(Ordering::SeqCst);
        self.refresh(RefreshTarget::Since(since), retry_rate_limits)
            .await?
            .ok_or_else(|| {
                Error::InvalidRequest(
                    "Cannot refresh OAuth token. Client is using API key auth".into(),
                )
            })
    }

    /// Refreshes the credentials a request was rejected with, unless another
    /// caller already replaced them.
    async fn refresh_oauth_token_for_auth(
        &self,
        expected: &OpenAIAuth,
        retry_rate_limits: bool,
    ) -> Result<OAuthRefreshResult, Error> {
        let OpenAIAuth::ChatGptOAuth(rejected) = expected else {
            return Err(Error::InvalidRequest(
                "Cannot refresh OAuth token. Client is using API key auth".to_string(),
            ));
        };
        self.refresh(RefreshTarget::Rejected(rejected.clone()), retry_rate_limits)
            .await?
            .ok_or_else(|| {
                Error::InvalidRequest(
                    "Cannot refresh OAuth token. Client is using API key auth".into(),
                )
            })
    }

    /// After an HTTP 401 other than `token_expired`, as Codex's unauthorized
    /// recovery does: reads the credential store and adopts newer
    /// credentials of the same account from it, without refreshing. Returns
    /// whether the credentials the request was rejected with have been
    /// replaced (from the store, or by another caller meanwhile), so the
    /// request can be sent again. A failure (a store that can't be read, or
    /// holds another account) is logged: the 401 stands.
    async fn reload_rejected_auth(&self, rejected: &OpenAIAuth) -> bool {
        let OpenAIAuth::ChatGptOAuth(rejected) = rejected else {
            return false;
        };
        match self
            .refresh(RefreshTarget::Reload(rejected.clone()), false)
            .await
        {
            Ok(replaced) => replaced.is_some(),
            Err(error) => {
                log::warn!(
                    "arko_ai: ChatGPT OAuth credentials were not reloaded after an HTTP 401: {error}"
                );
                false
            }
        }
    }

    /// Refresh attempts for `target`. When `retry_rate_limits` is on, an
    /// HTTP 429 is waited out (up to [`OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES`]
    /// times) with no lock held, so the wait can be cancelled and other
    /// callers go on. Each attempt takes the locks again and checks `target`
    /// again, since another caller may have refreshed meanwhile.
    async fn refresh(
        &self,
        target: RefreshTarget,
        retry_rate_limits: bool,
    ) -> Result<Option<OAuthRefreshResult>, Error> {
        let mut rate_limit_retries = 0_u32;
        loop {
            let retry =
                retry_rate_limits && rate_limit_retries < OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES;
            let error = match self.refresh_attempt(target.clone(), retry).await {
                Err(error) if retry && is_rate_limited_refresh(&error) => error,
                result => return result,
            };
            let delay = self.oauth_refresh_retry.delay(
                error.retry_after(),
                rate_limit_retries,
                crate::retry::random_sample(),
            );
            rate_limit_retries += 1;
            log::warn!(
                "ChatGPT OAuth refresh was rate limited. Waiting {delay:?} before retry {rate_limit_retries}"
            );
            tokio::time::sleep(delay).await;
        }
    }

    /// One refresh attempt. Takes the store's refresh lock (when it has
    /// one), then this client's credential lock (always in this order), and
    /// runs the rest in a task of its own that holds both: from the store
    /// read to installing and storing new tokens, nothing can be lost to the
    /// caller being dropped (a refresh token rotates when the request
    /// succeeds, whether or not anyone reads the response). Dropping the
    /// caller while it waits for the locks cancels the attempt. A runtime
    /// shutting down cancels the task too.
    ///
    /// The store's lock comes first so that a client waiting for another
    /// client's refresh of the same credentials does not hold its own
    /// credentials meanwhile: its requests, which only read them, go on with
    /// the token it holds. Once taken, the credential lock is held until the
    /// refresh has installed and stored the new tokens, so this client's
    /// requests (and [`chatgpt_oauth_credentials`](Self::chatgpt_oauth_credentials))
    /// wait for them.
    async fn refresh_attempt(
        &self,
        target: RefreshTarget,
        rate_limit_retry_follows: bool,
    ) -> Result<Option<OAuthRefreshResult>, Error> {
        let store_lock = match self
            .credential_store
            .as_ref()
            .and_then(|store| store.refresh_lock())
        {
            Some(lock) => Some(lock.lock_owned().await),
            None => None,
        };
        let auth = self.auth.clone().write_owned().await;
        let client = self.clone();
        let attempt = tokio::spawn(async move {
            let _store_lock = store_lock;
            client
                .refresh_locked(auth, &target, rate_limit_retry_follows)
                .await
        });
        match attempt.await {
            Ok(result) => result,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(_) => Err(Error::AuthenticationFailed(
                "The ChatGPT OAuth refresh was cancelled (the runtime is shutting down).".into(),
            )),
        }
    }

    /// [`refresh_attempt`](Self::refresh_attempt) once the locks are held.
    /// With `rate_limit_retry_follows`, an HTTP 429 is returned without
    /// reading the store again: the next attempt reads it first anyway.
    async fn refresh_locked(
        &self,
        mut auth: OwnedRwLockWriteGuard<OpenAIAuth>,
        target: &RefreshTarget,
        rate_limit_retry_follows: bool,
    ) -> Result<Option<OAuthRefreshResult>, Error> {
        let OpenAIAuth::ChatGptOAuth(held) = &*auth else {
            if matches!(target, RefreshTarget::Expiring(_)) {
                return Ok(None);
            }
            return Err(Error::InvalidRequest(
                "Cannot refresh OAuth token. Client is using API key auth".to_string(),
            ));
        };
        let mut current = held.clone();
        match target {
            RefreshTarget::Since(epoch) => {
                if self.auth_epoch.load(Ordering::SeqCst) != *epoch {
                    // Replaced while this caller waited.
                    return Ok(Some(Self::result_for(&current)));
                }
            }
            RefreshTarget::Rejected(rejected) | RefreshTarget::Reload(rejected) => {
                if current.account_id != rejected.account_id {
                    return Err(Error::AuthenticationFailed(
                        "OAuth account changed during the request. Start a new request.".into(),
                    ));
                }
                if current.access_token != rejected.access_token {
                    // Another caller already replaced the rejected token.
                    return Ok(Some(Self::result_for(&current)));
                }
            }
            RefreshTarget::Expiring(margin) => {
                if !is_token_expired(&current.access_token, *margin) {
                    return Ok(None);
                }
            }
        }

        // Another program sharing the credentials may have refreshed
        // already; refreshing again would reuse a rotated refresh token.
        // Never refresh without reading the store first.
        if let Some(store) = &self.credential_store
            && let Some(newer) = store.before_refresh(&current)?
        {
            let adopted = self.adopt_stored(&mut auth, &current, newer.clone())?;
            let expiring_too = matches!(target, RefreshTarget::Expiring(margin)
                if is_token_expired(&newer.access_token, *margin));
            if !expiring_too {
                return Ok(Some(adopted));
            }
            current = newer;
        }
        if matches!(target, RefreshTarget::Reload(_)) {
            return Ok(None);
        }
        let Some(refresh_token) = current.refresh_token.as_deref() else {
            return match target {
                // Without a refresh token the access token is used until the
                // server rejects it.
                RefreshTarget::Expiring(_) => Ok(None),
                RefreshTarget::Rejected(_) => {
                    Err(Error::AuthenticationFailed(
                        "The ChatGPT OAuth access token has expired and there is no refresh token to renew it. Sign in again and provide the new credentials, or provide a refresh token (for example by loading a Codex auth.json with CodexAuthFile)."
                            .to_string(),
                    ))
                }
                _ => Err(Error::AuthenticationFailed(
                    "There is no refresh token in the ChatGPT OAuth credentials to refresh them with. Provide a refresh_token when constructing the client."
                        .to_string(),
                )),
            };
        };
        let result = match request_oauth_refresh(
            &self.http_client,
            refresh_token,
            Some(&current.account_id),
            None,
            self.oauth_token_url.as_deref(),
            self.oauth_refresh_timeout,
        )
        .await
        {
            Ok(result) => result,
            Err(error) if rate_limit_retry_follows && is_rate_limited_refresh(&error) => {
                return Err(error);
            }
            Err(error) => {
                // The server rejects a refresh token another program rotated
                // in the meantime; its new tokens are then in the store.
                if let Some(store) = &self.credential_store
                    && let Ok(Some(newer)) = store.before_refresh(&current)
                {
                    return self.adopt_stored(&mut auth, &current, newer).map(Some);
                }
                return Err(error);
            }
        };

        let refreshed = ChatGptOAuthCredentials {
            access_token: result.access_token.clone(),
            refresh_token: result.refresh_token.clone().or(current.refresh_token),
            account_id: current.account_id,
        };
        self.install_auth(&mut auth, OpenAIAuth::ChatGptOAuth(refreshed.clone()));
        // The previous refresh token no longer works: store the new tokens
        // before anything else can fail.
        if let Some(store) = &self.credential_store {
            store.save_refreshed(&refreshed, result.id_token.as_deref())?;
        } else if self.warn_unsaved_rotation {
            crate::notice::warn_once("openai-oauth:unsaved-rotation", || {
                "arko_ai: the ChatGPT OAuth refresh token from OPENAI_OAUTH_REFRESH_TOKEN was rotated. The new tokens are kept in memory only, and the value in the environment no longer works. Save them from LLMClient::openai_chatgpt_oauth_credentials(), or load the credentials with CodexAuthFile and LLMClient::with_codex_auth_file so every rotation is written back.".to_string()
            });
        }
        Ok(Some(result))
    }

    fn result_for(credentials: &ChatGptOAuthCredentials) -> OAuthRefreshResult {
        OAuthRefreshResult {
            access_token: credentials.access_token.clone(),
            refresh_token: credentials.refresh_token.clone(),
            id_token: None,
        }
    }

    /// Installs credentials another program stored, in place of a refresh.
    fn adopt_stored(
        &self,
        auth: &mut OpenAIAuth,
        current: &ChatGptOAuthCredentials,
        stored: ChatGptOAuthCredentials,
    ) -> Result<OAuthRefreshResult, Error> {
        if stored.account_id != current.account_id {
            return Err(Error::AuthenticationFailed(
                "The credential store holds a different ChatGPT account. Sign in again.".into(),
            ));
        }
        if spent_tokens::credentials_spent(&stored) {
            return Err(crate::codex_auth::spent_refresh_token_error(Some(
                &stored.account_id,
            )));
        }
        let result = Self::result_for(&stored);
        self.install_auth(auth, OpenAIAuth::ChatGptOAuth(stored));
        Ok(result)
    }

    /// Replaces the ChatGPT OAuth credentials only if the client still holds
    /// `expected`. Returns whether it did.
    pub(crate) async fn replace_chatgpt_credentials(
        &self,
        expected: &ChatGptOAuthCredentials,
        new: ChatGptOAuthCredentials,
    ) -> bool {
        let mut auth = self.auth.write().await;
        match &*auth {
            OpenAIAuth::ChatGptOAuth(current) if current == expected => {
                self.install_auth(&mut auth, OpenAIAuth::ChatGptOAuth(new));
                true
            }
            _ => false,
        }
    }

    async fn snapshot_auth(&self) -> OpenAIAuth {
        self.auth.read().await.clone()
    }

    /// Sends `request`, refreshing ChatGPT credentials once on a 401 they
    /// can recover from. A 400 refusing the reasoning summaries it asks for
    /// (an organization not verified for them) sends it once more without
    /// them, and the client asks for none from then on: the third value is
    /// then `true`.
    async fn send_response_request(
        &self,
        request: &mut OpenAIRequest,
        mut auth: OpenAIAuth,
        conversation: &Conversation,
        trace: &mut Option<TraceCall>,
    ) -> Result<(reqwest::Response, OpenAIAuth, bool), Error> {
        let conversation_key = conversation.prompt_cache_key.as_str();
        let mut summaries_dropped = false;
        let mut refreshed = false;
        loop {
            let url = self.responses_url(&auth);
            let builder = self
                .http_client
                .post(url)
                .header("Content-Type", "application/json");
            let builder = self.apply_request_headers(&auth, builder, conversation, request);
            let http_request = self.prepare_http_request(
                Self::apply_auth_headers(&auth, builder).json(request),
                conversation_key,
            )?;
            if let Some(trace) = trace.as_mut() {
                trace.record_request(&http_request);
            }
            let response = match self.http_client.execute(http_request).await {
                Ok(response) => response,
                Err(error) => {
                    if let Some(trace) = trace.take() {
                        trace.fail(&error.to_string(), None);
                    }
                    return Err(error.into());
                }
            };
            if let Some(trace) = trace.as_mut() {
                trace.record_response_head(&response);
            }
            if let Some(hook) = &self.request_hook {
                hook.observe_response(conversation_key, response.headers());
            }
            if response.status().is_success() {
                return Ok((response, auth, summaries_dropped));
            }
            let status = response.status();
            let retry_after = retry_after_from_headers(&response);
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unavailable error body>".into());
            if status == reqwest::StatusCode::BAD_REQUEST
                && refuses_reasoning_summaries(&body)
                && let Some(reasoning) = request
                    .reasoning
                    .as_mut()
                    .filter(|reasoning| reasoning.summary.is_some())
            {
                // Nothing was streamed: the same request goes again without
                // the summaries, and no request asks for them again.
                log::warn!(
                    "OpenAI refused reasoning summaries for this organization (it isn't verified for them). Requests ask for none from now on."
                );
                self.summaries_refused.store(true, Ordering::Relaxed);
                reasoning.summary = None;
                summaries_dropped = true;
                continue;
            }
            let error = response_error("OpenAI request failed", status, retry_after, &body);
            let expired = matches!(&error, Error::ProviderError {
                status: Some(401), code: Some(code), ..
            } if code == "token_expired");
            let recoverable = !refreshed
                && status == reqwest::StatusCode::UNAUTHORIZED
                && auth.is_chatgpt_oauth();
            let recovered = if recoverable && expired {
                // A server can reject a token before its embedded expiry.
                // Refresh once; transient failures return to the agent's bounded
                // recovery policy. No stream or tool calls have been delivered.
                let refreshed = self.refresh_oauth_token_for_auth(&auth, false).await;
                if let Err(refresh_error) = refreshed {
                    if let Some(trace) = trace.take() {
                        trace.fail(&refresh_error.to_string(), None);
                    }
                    return Err(refresh_error);
                }
                true
            } else if recoverable && self.credential_store.is_some() {
                // Any other 401: another program may have refreshed or signed
                // in again (Codex's unauthorized recovery re-reads auth.json
                // first). Newer stored tokens get one more try; nothing is
                // refreshed.
                self.reload_rejected_auth(&auth).await
            } else {
                false
            };
            if recovered {
                let updated = self.snapshot_auth().await;
                if !matches!((&auth, &updated), (OpenAIAuth::ChatGptOAuth(old), OpenAIAuth::ChatGptOAuth(new)) if old.account_id == new.account_id)
                {
                    if let Some(trace) = trace.take() {
                        trace.fail(&error.to_string(), None);
                    }
                    return Err(error);
                }
                auth = updated;
                refreshed = true;
            } else {
                if let Some(trace) = trace.take() {
                    trace.fail(
                        &error.to_string(),
                        serde_json::from_str(&body).ok().as_ref(),
                    );
                }
                return Err(error);
            }
        }
    }

    /// Builds the HTTP request, letting an installed [`OpenAIRequestHook`]
    /// adjust it. The body is re-serialized only if the hook changed it.
    fn prepare_http_request(
        &self,
        builder: reqwest::RequestBuilder,
        conversation_key: &str,
    ) -> Result<reqwest::Request, Error> {
        let mut request = builder.build()?;
        if let Some(hook) = &self.request_hook {
            let bytes = request
                .body()
                .and_then(reqwest::Body::as_bytes)
                .unwrap_or_default();
            let original: Value = serde_json::from_slice(bytes).map_err(Error::JsonError)?;
            let mut body = original.clone();
            hook.prepare(conversation_key, request.headers_mut(), &mut body);
            if body != original {
                *request.body_mut() =
                    Some(serde_json::to_vec(&body).map_err(Error::JsonError)?.into());
            }
        }
        Ok(request)
    }

    fn begin_trace(
        &self,
        conversation: &Conversation,
        request: &OpenAIRequest,
        stream: bool,
    ) -> Option<TraceCall> {
        let trace = self.trace.as_ref()?;
        let body = serde_json::to_value(request).unwrap_or(Value::Null);
        Some(trace.begin(
            &conversation.prompt_cache_key,
            &conversation.model.id,
            &body,
            stream,
        ))
    }

    fn responses_url(&self, auth: &OpenAIAuth) -> String {
        let base = self.base_url.as_deref().unwrap_or(match auth {
            OpenAIAuth::ApiKey(_) => OPENAI_BASE_URL,
            OpenAIAuth::ChatGptOAuth(_) => CODEX_BASE_URL,
        });
        format!("{}/responses", base.trim_end_matches('/'))
    }

    fn apply_auth_headers(
        auth: &OpenAIAuth,
        builder: reqwest::RequestBuilder,
    ) -> reqwest::RequestBuilder {
        match auth {
            OpenAIAuth::ApiKey(key) => builder.header("Authorization", format!("Bearer {}", key)),
            OpenAIAuth::ChatGptOAuth(credentials) => builder
                .header(
                    "Authorization",
                    format!("Bearer {}", credentials.access_token),
                )
                .header("chatgpt-account-id", &credentials.account_id),
        }
    }

    /// Headers other than authentication. The application's `originator`
    /// and `User-Agent` go out only when configured. ChatGPT OAuth requests
    /// carry the conversation's identity as Codex does: `session-id` (the
    /// backend derives prompt-cache affinity from it), `thread-id` and
    /// `x-client-request-id`, all set to the conversation's
    /// `prompt_cache_key`, plus the turn's `x-codex-turn-state` when one is
    /// held (see [`Conversation::openai_turn_state`]) and Codex's
    /// `x-codex-routing-hint` (`model=<model>`, with `;tier=priority` in Fast
    /// mode).
    fn apply_request_headers(
        &self,
        auth: &OpenAIAuth,
        mut builder: reqwest::RequestBuilder,
        conversation: &Conversation,
        request: &OpenAIRequest,
    ) -> reqwest::RequestBuilder {
        fn header(
            builder: reqwest::RequestBuilder,
            name: &'static str,
            value: &str,
        ) -> reqwest::RequestBuilder {
            match HeaderValue::from_str(value) {
                Ok(value) => builder.header(name, value),
                Err(_) => {
                    log::warn!(
                        "arko_ai: Not sending the {name} header. Its value is not a valid header value"
                    );
                    builder
                }
            }
        }

        if let Some(originator) = &self.originator {
            builder = header(builder, "originator", originator);
        }
        if let Some(user_agent) = &self.user_agent {
            builder = header(builder, "user-agent", user_agent);
        }
        if auth.is_chatgpt_oauth() {
            let key = conversation.prompt_cache_key.as_str();
            builder = header(builder, "session-id", key);
            builder = header(builder, "thread-id", key);
            builder = header(builder, "x-client-request-id", key);
            if let Some(token) = conversation.current_openai_turn_state() {
                builder = header(builder, TURN_STATE_HEADER, token);
            }
            let routing_hint = match request.service_tier {
                Some(tier) => format!(
                    "model={};tier={}",
                    request.model,
                    Self::service_tier_wire(tier)
                ),
                None => format!("model={}", request.model),
            };
            builder = header(builder, ROUTING_HINT_HEADER, &routing_hint);
            if request.stream == Some(true) {
                builder = builder.header(ACCEPT, "text/event-stream");
            }
        }
        builder
    }
}

#[async_trait::async_trait]
impl ProviderClient for OpenAIClient {
    async fn generate_text(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        if conversation.model.provider != ProviderType::OpenAI {
            return Err(Error::ProviderNotSupported(
                "Model is not from OpenAI".to_string(),
            ));
        }

        if conversation.model.model_type == ModelType::Reasoning && config.temperature.is_some() {
            return Err(Error::InvalidRequest(
                "OpenAI reasoning models do not support temperature configuration.".to_string(),
            ));
        }

        let auth = self.snapshot_auth().await;
        if auth.is_chatgpt_oauth() {
            // Codex always streams on the ChatGPT backend; collect the stream.
            let mut stream = self.generate_stream(conversation, config).await?;
            while let Some(event) = stream.next().await {
                if let StreamEvent::End(response) = event? {
                    return Ok(*response);
                }
            }
            return Err(Error::StreamTruncated(
                "OpenAI stream ended before the response completed".to_string(),
            ));
        }
        let mut request = self.build_request(conversation, config, auth.is_chatgpt_oauth())?;

        let mut trace = self.begin_trace(conversation, &request, false);
        let (response, auth, summaries_dropped) = self
            .send_response_request(&mut request, auth, conversation, &mut trace)
            .await?;
        let head = ResponseHead::new(&request.model, response.headers());

        let response_data: OpenAIResponse = match trace.take() {
            None => response.json().await?,
            // A body that doesn't parse fails the call, so it is traced as
            // a failure.
            Some(trace) => {
                let value: Value = match response.json().await {
                    Ok(value) => value,
                    Err(error) => {
                        trace.fail(
                            &format!("failed to read or parse the response: {error}"),
                            None,
                        );
                        return Err(error.into());
                    }
                };
                match OpenAIResponse::deserialize(&value) {
                    Ok(response_data) => {
                        trace.finish(&value);
                        response_data
                    }
                    Err(error) => {
                        trace.fail(
                            &format!("failed to parse the response: {error}"),
                            Some(&value),
                        );
                        return Err(Error::JsonError(error));
                    }
                }
            }
        };
        let mut response = Self::generation_response(
            &response_data,
            &conversation.model,
            config,
            auth.is_chatgpt_oauth(),
            StreamedOutput::default(),
            &head,
        );
        response.model_call_usage.reasoning_summaries_refused = summaries_dropped;
        Ok(response)
    }

    async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        if conversation.model.provider != ProviderType::OpenAI {
            return Err(Error::ProviderNotSupported(
                "Model is not from OpenAI".to_string(),
            ));
        }

        if conversation.model.model_type == ModelType::Reasoning && config.temperature.is_some() {
            return Err(Error::InvalidRequest(
                "OpenAI reasoning models do not support temperature configuration.".to_string(),
            ));
        }

        let auth = self.snapshot_auth().await;
        let mut request = self.build_request(conversation, config, auth.is_chatgpt_oauth())?;
        request.stream = Some(true);

        let mut trace = self.begin_trace(conversation, &request, true);
        let (response, auth, summaries_dropped) = self
            .send_response_request(&mut request, auth, conversation, &mut trace)
            .await?;
        let head = ResponseHead::new(&request.model, response.headers());

        let (model, config) = (conversation.model.clone(), config.clone());
        let stream = super::idle::parse_with_idle_timeout(
            response.bytes_stream(),
            self.stream_idle_timeout,
            move |bytes| {
                Self::stream_events_traced(
                    bytes,
                    model,
                    config,
                    auth.is_chatgpt_oauth(),
                    head,
                    trace,
                )
            },
        );
        if !summaries_dropped {
            return Ok(stream);
        }
        // The response says the summaries were refused.
        Ok(stream
            .map(|event| match event {
                Ok(StreamEvent::End(mut response)) => {
                    response.model_call_usage.reasoning_summaries_refused = true;
                    Ok(StreamEvent::End(response))
                }
                other => other,
            })
            .boxed())
    }
}

impl OpenAIClient {
    /// Parses a Responses API SSE byte stream. `End` is emitted only for a
    /// `response.completed` or `response.incomplete` event. A stream that
    /// ends earlier yields [`Error::StreamTruncated`].
    #[cfg(test)]
    fn stream_events<S, B, E>(
        bytes: S,
        model: crate::model::Model,
        config: GenerationConfig,
        is_chatgpt_oauth: bool,
    ) -> BoxStream<'static, Result<StreamEvent, Error>>
    where
        S: futures::Stream<Item = Result<B, E>> + Send + 'static,
        B: AsRef<[u8]> + Send + 'static,
        E: std::fmt::Display + Send + 'static,
    {
        let head = ResponseHead {
            requested_model: model.id.clone(),
            ..ResponseHead::default()
        };
        Self::stream_events_traced(bytes, model, config, is_chatgpt_oauth, head, None)
    }

    /// [`stream_events`](Self::stream_events), recording each event in `trace`.
    ///
    /// The assistant output is assembled from `response.output_item.done`
    /// events, in stream order, as Codex does: the ChatGPT backend's
    /// `response.completed` usually carries no output items. Items only the
    /// terminal event holds are merged in (see [`Self::assemble_output`]).
    fn stream_events_traced<S, B, E>(
        bytes: S,
        model: crate::model::Model,
        config: GenerationConfig,
        is_chatgpt_oauth: bool,
        head: ResponseHead,
        mut trace: Option<TraceCall>,
    ) -> BoxStream<'static, Result<StreamEvent, Error>>
    where
        S: futures::Stream<Item = Result<B, E>> + Send + 'static,
        B: AsRef<[u8]> + Send + 'static,
        E: std::fmt::Display + Send + 'static,
    {
        let stream = try_stream! {
            let mut event_stream = Box::pin(bytes.eventsource());
            let mut streaming_tool_calls: Vec<StreamingToolCall> = Vec::new();
            let mut streamed = StreamedOutput::default();
            let mut finished = false;

            while let Some(event) = event_stream.next().await {
                let event = match event {
                    Ok(event) => event,
                    Err(error) => {
                        let message = format!("SSE error: {error}");
                        if let Some(trace) = trace.take() {
                            trace.fail(&message, None);
                        }
                        Err(Error::ApiError(message))?;
                        break;
                    }
                };

                let value: Value = match serde_json::from_str(&event.data) {
                    Ok(value) => value,
                    Err(error) => {
                        if let Some(trace) = trace.take() {
                            trace.fail(&format!("invalid event JSON: {error}"), None);
                        }
                        Err(Error::JsonError(error))?;
                        break;
                    }
                };

                if let Some(trace) = trace.as_mut() {
                    trace.on_event(value.get("type").and_then(Value::as_str).unwrap_or(""), &value);
                }

                match value.get("type").and_then(Value::as_str) {
                    Some("response.created") => {
                        streamed.created_model = value
                            .pointer("/response/model")
                            .and_then(Value::as_str)
                            .map(String::from);
                    }
                    Some("response.output_text.delta") => {
                        let delta = value["delta"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        streamed.text.push_str(&delta);

                        yield StreamEvent::TextChunk(delta);
                    }
                    // Refusal text is reported through `StreamEvent::Refusal`,
                    // never as ordinary text.
                    Some("response.refusal.delta" | "response.refusal.done") => {
                        Self::record_stream_refusal(&mut streamed.refusals, &value);
                    }
                    Some("response.reasoning_summary_text.delta") => {
                        let delta = value["delta"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();

                        yield StreamEvent::ReasoningChunk(delta);
                    }
                    Some("response.output_item.added") => {
                        if let Some(item) = value.get("item")
                            && item.get("type").and_then(Value::as_str)
                                == Some("function_call")
                        {
                            let name = item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            let index = streaming_tool_calls.len();

                            streaming_tool_calls.push(StreamingToolCall {
                                index,
                                name: name.clone(),
                                arguments: String::new(),
                                item_id: item
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .map(String::from),
                                call_id: item
                                    .get("call_id")
                                    .and_then(Value::as_str)
                                    .map(String::from),
                                output_index: value.get("output_index").and_then(Value::as_u64),
                                completed: false,
                            });

                            yield StreamEvent::ToolCallStart { index, name };
                        }
                    }
                    Some("response.function_call_arguments.delta") => {
                        let delta = value
                            .get("delta")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();

                        if let Some(tool_call) =
                            StreamingToolCall::find(&mut streaming_tool_calls, &value)
                        {
                            tool_call.arguments.push_str(&delta);
                            let index = tool_call.index;

                            yield StreamEvent::ToolCallArgumentsDelta { index, delta };
                        }
                    }
                    Some("response.function_call_arguments.done") => {
                        if let Some(tool_call) =
                            StreamingToolCall::find(&mut streaming_tool_calls, &value)
                        {
                            if let Some(arguments) =
                                value.get("arguments").and_then(Value::as_str)
                            {
                                tool_call.arguments = arguments.to_string();
                            }
                            tool_call.completed = true;
                        }
                    }
                    Some("response.output_item.done") => {
                        let item = &value["item"];
                        if item.get("type").and_then(Value::as_str) == Some("function_call")
                            && let Some(tool_call) = streaming_tool_calls.iter_mut().find(|call| {
                                call.item_id.is_some()
                                    && call.item_id.as_deref() == item.get("id").and_then(Value::as_str)
                            })
                        {
                            if let Some(arguments) =
                                item.get("arguments").and_then(Value::as_str)
                            {
                                tool_call.arguments = arguments.to_string();
                            }
                            tool_call.completed = matches!(
                                item.get("status").and_then(Value::as_str),
                                None | Some("completed")
                            );
                        }
                        if item.is_object() {
                            streamed.items.push((value.get("output_index").and_then(Value::as_u64), item.clone()));
                        }
                    }
                    Some("response.completed" | "response.incomplete") => {
                        #[derive(Deserialize)]
                        struct Completed {
                            response: OpenAIResponse,
                        }

                        let completed = Completed::deserialize(&value);
                        if let Some(trace) = trace.take() {
                            let response = value.get("response");
                            match &completed {
                                Ok(_) => trace.finish(response.unwrap_or(&Value::Null)),
                                Err(error) => trace.fail(
                                    &format!("failed to parse the response: {error}"),
                                    response,
                                ),
                            }
                        }
                        let completed = completed.map_err(Error::JsonError)?;

                        streamed.calls = streaming_tool_calls
                            .drain(..)
                            .filter(|tool_call| tool_call.completed)
                            .collect();

                        let response = Self::generation_response(
                            &completed.response,
                            &model,
                            &config,
                            is_chatgpt_oauth,
                            std::mem::take(&mut streamed),
                            &head,
                        );

                        if let Some(refusal) = &response.refusal {
                            yield StreamEvent::Refusal(refusal.clone());
                        }

                        yield StreamEvent::End(Box::new(response));

                        finished = true;
                        break;
                    }
                    Some("error" | "response.failed") => {
                        // `error` events are flat (`code` may be null) or wrap
                        // an `error` object; `response.failed` carries the
                        // error on its response. Classified in `provider_error`.
                        let value = value.get("response").unwrap_or(&value);
                        let message = value
                            .pointer("/error/message")
                            .or_else(|| value.get("message"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if let Some(trace) = trace.take() {
                            trace.fail(&format!("OpenAI stream error: {message}"), Some(value));
                        }

                        Err(Error::provider_error(
                            &format!("OpenAI stream error: {message}"),
                            None,
                            None,
                            &value.to_string(),
                        ))?;

                        break;
                    }
                    _ => {}
                }
            }

            if !finished {
                if let Some(trace) = trace.take() {
                    trace.fail("stream ended before the response completed", None);
                }
                Err(Error::StreamTruncated(
                    "OpenAI stream ended before the response completed".to_string(),
                ))?;
            }
        };

        stream.boxed()
    }

    fn generation_response(
        response: &OpenAIResponse,
        model: &crate::model::Model,
        config: &GenerationConfig,
        is_chatgpt_oauth: bool,
        streamed: StreamedOutput,
        head: &ResponseHead,
    ) -> GenerationResponse {
        let output = Self::assemble_output(&streamed.items, &response.output);
        let (mut message, mut refusal_texts, mut replay_items) = Self::parse_output(&output);
        let mut usage = Self::calculate_usage(response);
        // ChatGPT OAuth calls are covered by the subscription: the estimate
        // stays visible while nothing is billed per call.
        if is_chatgpt_oauth {
            usage.billing = BillingKind::Subscription;
            usage.service_tier = Some(Self::chatgpt_oauth_requested_tier(config));
        }
        usage.calculate_cost(model, Some(config));

        let model_call_usage = ModelCallUsage::new(
            response.id.clone(),
            model.provider.as_telemetry_str().to_string(),
            response.model.clone(),
            response.usage.input_tokens,
            response.usage.output_tokens,
            crate::input_limit::effective_max_input_tokens(model, is_chatgpt_oauth).unwrap_or(0),
        );
        let model_call_usage = Self::with_response_stop_metadata(
            model_call_usage,
            response,
            &output,
            &streamed.refusals,
        );
        let (served_model, rerouted) =
            Self::served_model(head, response, streamed.created_model.as_deref());
        let model_call_usage = model_call_usage
            .with_served_model(served_model, rerouted)
            .with_plan_usage(head.plan_usage.clone());

        if refusal_texts.is_empty() {
            refusal_texts = streamed.refusals.values().cloned().collect();
        }

        // A stream whose output items never arrived still delivered its text
        // and finished calls through delta events; keep them, with output
        // items that replay them.
        let has_message = replay_items
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) == Some("message"));
        if !has_message && refusal_texts.is_empty() && !streamed.text.is_empty() {
            replay_items.push(serde_json::json!({
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": streamed.text}],
            }));
            message.content = vec![Content::Text(streamed.text)];
        }
        if message.tool_calls.is_empty() && refusal_texts.is_empty() {
            for call in streamed.calls {
                replay_items.push(call.replay_item());
                message.tool_calls.push(call.into_tool_call());
            }
        }
        message.reasoning_artifact =
            (!replay_items.is_empty()).then_some(ReasoningArtifact::OpenAIOutputItems {
                items: replay_items,
            });

        let (stop_reason, refusal) = Self::finish_turn(
            &mut message,
            response,
            refusal_texts,
            model_call_usage.stop_details.clone(),
        );

        GenerationResponse {
            message,
            usage,
            model_call_usage,
            stop_reason,
            refusal,
            openai_turn_state: head.turn_state.clone(),
        }
    }

    /// The response's output items, in output order. Items streamed as
    /// `response.output_item.done` come first, as Codex builds its output
    /// only from those. Items only `response.completed.output` holds are
    /// merged in at their output index. A non-streaming response has only
    /// the latter.
    fn assemble_output(streamed: &[(Option<u64>, Value)], completed: &[Value]) -> Vec<Value> {
        if streamed.is_empty() {
            return completed.to_vec();
        }
        let same_item = |a: &Value, b: &Value| match (
            a.get("id").and_then(Value::as_str),
            b.get("id").and_then(Value::as_str),
        ) {
            (Some(a), Some(b)) => a == b,
            _ => a == b,
        };
        let mut output = streamed.to_vec();
        for (index, item) in completed.iter().enumerate() {
            if output.iter().any(|(_, streamed)| same_item(streamed, item)) {
                continue;
            }
            let index = index as u64;
            let position = output
                .iter()
                .position(|(streamed_index, _)| streamed_index.is_some_and(|i| i > index))
                .unwrap_or(output.len());
            output.insert(position, (Some(index), item.clone()));
        }
        output.into_iter().map(|(_, item)| item).collect()
    }

    /// The model the provider reports having served, and whether any
    /// reported model differs from the requested one.
    fn served_model(
        head: &ResponseHead,
        response: &OpenAIResponse,
        created_model: Option<&str>,
    ) -> (Option<String>, bool) {
        let reported = [
            head.served_model.as_deref(),
            Some(response.model.as_str()),
            created_model,
        ];
        let mut reported = reported
            .into_iter()
            .flatten()
            .filter(|model| !model.is_empty());
        let served = reported.clone().next().map(String::from);
        // Reported models are untrusted text: only ASCII ids are sliced.
        fn base(id: &str) -> &str {
            if id.is_ascii() {
                OpenAIClient::strip_openai_snapshot_suffix(id)
            } else {
                id
            }
        }
        let requested = base(&head.requested_model);
        let rerouted = !requested.is_empty()
            && reported.any(|model| !base(model).eq_ignore_ascii_case(requested));
        (served, rerouted)
    }

    /// Normalizes a finished response. Refusal parts are the model declining
    /// in its own words. `incomplete` responses carry their reason in
    /// `incomplete_details`.
    /// https://developers.openai.com/api/docs/guides/structured-outputs#refusals
    /// https://platform.openai.com/docs/api-reference/responses/object
    fn finish_turn(
        message: &mut Message,
        response: &OpenAIResponse,
        refusal_texts: Vec<String>,
        stop_details: Option<Value>,
    ) -> (StopReason, Option<Refusal>) {
        if !refusal_texts.is_empty() {
            let mut refusal = Refusal::new(RefusalSource::Model);
            refusal.explanation = Some(refusal_texts.join("\n"));
            refusal.provider_details = stop_details.unwrap_or(Value::Null);
            return (StopReason::Refusal, Some(refusal));
        }

        let incomplete_reason = response
            .incomplete_details
            .as_ref()
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str);

        match (response.status.as_str(), incomplete_reason) {
            ("incomplete", Some("content_filter")) => {
                let mut refusal = Refusal::new(RefusalSource::ContentFilter);
                refusal.partial_output = take_refused_output(message);
                refusal.provider_details =
                    response.incomplete_details.clone().unwrap_or(Value::Null);
                (StopReason::Refusal, Some(refusal))
            }
            ("incomplete", Some("max_output_tokens")) => (StopReason::MaxTokens, None),
            ("incomplete", reason) => (
                StopReason::Other(reason.unwrap_or("incomplete").to_string()),
                None,
            ),
            ("completed", _) if !message.tool_calls.is_empty() => (StopReason::ToolUse, None),
            ("completed", _) => (StopReason::EndTurn, None),
            (status, _) => (StopReason::Other(status.to_string()), None),
        }
    }

    fn record_stream_refusal(refusals: &mut StreamingRefusals, event: &Value) {
        let key = (
            event["item_id"].as_str().unwrap_or("").to_string(),
            event["content_index"].as_u64().unwrap_or(0),
        );
        match event["type"].as_str() {
            Some("response.refusal.delta") => {
                refusals
                    .entry(key)
                    .or_default()
                    .push_str(event["delta"].as_str().unwrap_or(""));
            }
            Some("response.refusal.done") => {
                refusals.insert(key, event["refusal"].as_str().unwrap_or("").to_string());
            }
            _ => {}
        }
    }

    fn with_response_stop_metadata(
        usage: ModelCallUsage,
        response: &OpenAIResponse,
        output: &[Value],
        streamed: &StreamingRefusals,
    ) -> ModelCallUsage {
        let mut refusals = Vec::new();
        for item in output {
            if let Ok(OpenAIOutput::Message { id, content, .. }) = OpenAIOutput::deserialize(item) {
                for (index, part) in content.iter().enumerate() {
                    if part.type_ == "refusal" {
                        refusals.push(serde_json::json!({
                            "item_id": id, "content_index": index, "refusal": part.refusal,
                        }));
                    }
                }
            }
        }
        if refusals.is_empty() {
            refusals.extend(streamed.iter().map(|((id, index), text)| {
                serde_json::json!({
                    "item_id": id, "content_index": index, "refusal": text,
                })
            }));
        }
        if !refusals.is_empty() {
            return usage.with_stop_metadata(
                Some("refusal".into()),
                Some(serde_json::json!({
                    "source": "openai_structured_refusal", "response_status": response.status,
                    "refusals": refusals,
                })),
            );
        }
        let reason = response
            .incomplete_details
            .as_ref()
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or(&response.status)
            .to_string();
        let details = response.incomplete_details.as_ref().map(|details| {
            serde_json::json!({
                "response_status": response.status, "incomplete_details": details,
            })
        });
        usage.with_stop_metadata(Some(reason), details)
    }

    /// The `service_tier` sent over ChatGPT OAuth, as Codex sends it there
    /// (`service_tier_for_request` in openai/codex
    /// `codex-rs/protocol/src/openai_models.rs`): Fast as `priority` for the
    /// models whose Codex catalog entry offers it, and `Default`, Codex's
    /// "standard routing" choice, as no tier at all. Fast on another model is
    /// an error rather than Codex's silent omission, as on the API-key path.
    /// Flex and Auto stay refused: no catalog entry offers them (Codex passes
    /// a configured `flex` through as "an API request option", but whether
    /// the ChatGPT backend honors it is unverified).
    fn chatgpt_oauth_service_tier(
        tier: OpenAIServiceTier,
        request_model: &str,
    ) -> Result<Option<OpenAIServiceTier>, Error> {
        match tier {
            OpenAIServiceTier::Fast if CHATGPT_FAST_MODE_MODELS.contains(&request_model) => {
                Ok(Some(tier))
            }
            OpenAIServiceTier::Fast => Err(Error::InvalidRequest(format!(
                "{request_model} does not offer Fast mode over ChatGPT OAuth. Codex offers it for {}.",
                CHATGPT_FAST_MODE_MODELS.join(", ")
            ))),
            OpenAIServiceTier::Default => Ok(None),
            OpenAIServiceTier::Flex | OpenAIServiceTier::Auto => {
                Err(Error::UnsupportedFeature(format!(
                    "The {tier:?} OpenAI service tier requires API-key authentication. Over ChatGPT OAuth only Fast (and Default, which sends no tier) can be requested."
                )))
            }
        }
    }

    /// The tier a ChatGPT OAuth call is reported and priced at: the one
    /// requested (Fast when `priority` was sent, standard otherwise), not the
    /// `service_tier` the response echoes. The ChatGPT backend echoes
    /// `default` or `auto` for Fast calls that are served fast, and Codex
    /// never reads the echo (`ResponseCompleted` in openai/codex
    /// `codex-rs/codex-api/src/sse/responses.rs` has no `service_tier`).
    /// A request that would send another tier fails before it is sent (see
    /// [`chatgpt_oauth_service_tier`](Self::chatgpt_oauth_service_tier)).
    fn chatgpt_oauth_requested_tier(config: &GenerationConfig) -> ServiceTier {
        match config.openai_service_tier {
            Some(OpenAIServiceTier::Fast) => ServiceTier::Fast,
            _ => ServiceTier::Standard,
        }
    }

    /// The wire value of `tier`, as serialized in `service_tier`.
    fn service_tier_wire(tier: OpenAIServiceTier) -> &'static str {
        match tier {
            OpenAIServiceTier::Auto => "auto",
            OpenAIServiceTier::Default => "default",
            OpenAIServiceTier::Flex => "flex",
            OpenAIServiceTier::Fast => "priority",
        }
    }

    fn request_model_id(model_id: &str, is_chatgpt_oauth: bool) -> String {
        if !is_chatgpt_oauth
            || !(model_id.starts_with("gpt-5.4") || model_id.starts_with("gpt-5.5"))
        {
            return model_id.to_string();
        }

        Self::strip_openai_snapshot_suffix(model_id).to_string()
    }

    fn strip_openai_snapshot_suffix(model_id: &str) -> &str {
        const SNAPSHOT_SUFFIX_LEN: usize = "-2026-04-23".len();

        if model_id.len() < SNAPSHOT_SUFFIX_LEN {
            return model_id;
        }

        let suffix_start = model_id.len() - SNAPSHOT_SUFFIX_LEN;
        let suffix = &model_id[suffix_start..];
        let suffix_bytes = suffix.as_bytes();

        let is_snapshot_suffix = suffix_bytes[0] == b'-'
            && suffix_bytes[1..5].iter().all(u8::is_ascii_digit)
            && suffix_bytes[5] == b'-'
            && suffix_bytes[6..8].iter().all(u8::is_ascii_digit)
            && suffix_bytes[8] == b'-'
            && suffix_bytes[9..11].iter().all(u8::is_ascii_digit);

        if is_snapshot_suffix {
            &model_id[..suffix_start]
        } else {
            model_id
        }
    }

    fn build_request(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        is_chatgpt_oauth: bool,
    ) -> Result<OpenAIRequest, Error> {
        super::ensure_supported_inputs(conversation)?;
        let sanitized = super::without_foreign_settings(&conversation.model, config);
        let config: &GenerationConfig = &sanitized;
        if is_chatgpt_oauth {
            Self::warn_oauth_ignored_settings(conversation, config);
        }
        // The ChatGPT backend rejects `max_output_tokens` ("Unsupported
        // parameter"), and Codex never sends it (openai/codex#7100).
        let resolved_max_tokens = if is_chatgpt_oauth {
            None
        } else {
            config
                .max_tokens
                .or(Some(conversation.model.max_output_tokens))
        };

        let request_model = Self::request_model_id(&conversation.model.id, is_chatgpt_oauth);
        let service_tier = match config.openai_service_tier {
            Some(tier) if is_chatgpt_oauth => {
                Self::chatgpt_oauth_service_tier(tier, &request_model)?
            }
            Some(tier) => {
                let priced = tier.pricing_tier();
                if priced != ServiceTier::Standard
                    && !conversation.model.pricing.offers_tier(&priced)
                {
                    return Err(Error::InvalidRequest(format!(
                        "{} does not offer the {:?} service tier.",
                        conversation.model.id, tier
                    )));
                }
                Some(tier)
            }
            None => None,
        };

        let mut request = OpenAIRequest {
            model: request_model,
            store: false,
            instructions: conversation.system_message.clone(),
            reasoning: None,
            text: None,
            include: None,
            input: Vec::new(),
            max_output_tokens: resolved_max_tokens,
            // Omitted unless set, so the model default applies. Reasoning
            // models reject it (checked before building).
            temperature: config
                .temperature
                .filter(|_| conversation.model.model_type != ModelType::Reasoning),
            // The ChatGPT backend is always streamed, as Codex does.
            stream: Some(is_chatgpt_oauth),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            // Every model: the key routes requests of one conversation to the
            // same prompt cache.
            prompt_cache_key: Some(conversation.prompt_cache_key.clone()),
            prompt_cache_options: None,
            service_tier,
        };

        if conversation.model.supports_openai_prompt_cache_options() && !is_chatgpt_oauth {
            let enabled = config.prompt_caching.unwrap_or(conversation.enable_cache);

            request.prompt_cache_options = Some(OpenAIPromptCacheOptions {
                mode: if enabled {
                    OpenAIPromptCacheMode::Implicit
                } else {
                    OpenAIPromptCacheMode::Explicit
                },
            });
        }

        if config.reasoning.is_some() && conversation.model.model_type != ModelType::Reasoning {
            return Err(Error::InvalidRequest(
                "Reasoning configuration is only supported for reasoning models.".to_string(),
            ));
        }

        // Pro models share prefixes with their base models, so they are checked
        // first below.
        let is_gpt5_pro = conversation.model.is_openai_gpt_5_pro();
        let is_gpt5_2_pro = conversation.model.is_openai_gpt_5_2_pro();
        let is_gpt5_5_pro = conversation.model.is_openai_gpt_5_5_pro();
        let is_gpt5 = conversation.model.id.starts_with("gpt-5-");
        let is_gpt5_1 = conversation.model.id.starts_with("gpt-5.1");
        let is_gpt5_2 = conversation.model.id.starts_with("gpt-5.2");
        let is_gpt5_3 = conversation.model.id.starts_with("gpt-5.3");
        let is_gpt5_4_pro = conversation.model.id.starts_with("gpt-5.4-pro");
        let is_gpt5_4 = conversation.model.id.starts_with("gpt-5.4");
        let is_gpt5_5 = conversation.model.id.starts_with("gpt-5.5");
        let is_gpt5_6 = conversation.model.is_openai_gpt_5_6();
        let is_gpt6_astra = conversation.model.is_openai_gpt_6_astra();
        let is_gpt6_sol_luna = conversation.model.is_openai_gpt_6_sol_or_luna();
        let is_gpt6 = is_gpt6_astra || is_gpt6_sol_luna;

        let is_legacy_reasoning =
            conversation.model.id.starts_with("o3-") || conversation.model.id.starts_with("o4-");

        let is_gpt5_family =
            is_gpt5 || is_gpt5_1 || is_gpt5_2 || is_gpt5_3 || is_gpt5_4 || is_gpt5_5 || is_gpt5_6;

        if config.openai_reasoning_mode.is_some() && !(is_gpt5_6 || is_gpt6) {
            return Err(Error::InvalidRequest(
                "The OpenAI reasoning mode parameter is only supported for GPT-5.6 and GPT-6 models."
                    .to_string(),
            ));
        }

        if let Some(verbosity) = config.verbosity {
            if !(is_gpt5_family || is_gpt6) {
                return Err(Error::InvalidRequest(
                    "The verbosity parameter is only supported for GPT-5 family and GPT-6 models."
                        .to_string(),
                ));
            }

            request.text = Some(OpenAITextConfig { verbosity });
        }

        if conversation.model.model_type == ModelType::Reasoning {
            request.include = Some(vec!["reasoning.encrypted_content".to_string()]);
        }

        // Render replayed reasoning from earlier turns too. The API documents
        // `all_turns` only for GPT-5.6 ("GPT-5.6 models support this value",
        // https://developers.openai.com/api/docs/guides/reasoning), its
        // default. Over ChatGPT OAuth, Codex also sends it for GPT-6 (and the
        // backend echoes it); with an API key GPT-6 gets nothing until it is
        // documented there. Earlier models default to `current_turn`.
        let reasoning_context = (conversation.model.model_type == ModelType::Reasoning
            && (is_gpt5_6 || (is_gpt6 && is_chatgpt_oauth)))
            .then_some(OpenAIReasoningContext::AllTurns);

        if let Some(crate::reasoning::ReasoningConfig::OpenAI {
            effort,
            include_summaries,
        }) = &config.reasoning
        {
            let (allowed, supported_label): (&[OpenAIEffort], &str) = if is_gpt6_astra {
                (
                    &[
                        OpenAIEffort::Low,
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                        OpenAIEffort::XHigh,
                        OpenAIEffort::Max,
                    ],
                    "low, medium, high, xhigh, max",
                )
            } else if is_gpt5_6 || is_gpt6_sol_luna {
                (
                    &[
                        OpenAIEffort::None,
                        OpenAIEffort::Low,
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                        OpenAIEffort::XHigh,
                        OpenAIEffort::Max,
                    ],
                    "none, low, medium, high, xhigh, max",
                )
            } else if is_gpt5_5_pro || is_gpt5_4_pro || is_gpt5_2_pro {
                (
                    &[
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                        OpenAIEffort::XHigh,
                    ],
                    "medium, high, xhigh",
                )
            } else if is_gpt5_pro {
                // gpt-5-pro defaults to, and only supports, high.
                // https://developers.openai.com/api/docs/models/gpt-5-pro
                (&[OpenAIEffort::High], "high")
            } else if is_gpt5_1 {
                // xhigh arrived with GPT-5.2; GPT-5.1 supports none through high.
                // https://developers.openai.com/api/docs/models/gpt-5.1
                (
                    &[
                        OpenAIEffort::None,
                        OpenAIEffort::Low,
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                    ],
                    "none, low, medium, high",
                )
            } else if is_gpt5_5 || is_gpt5_4 || is_gpt5_2 {
                (
                    &[
                        OpenAIEffort::None,
                        OpenAIEffort::Low,
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                        OpenAIEffort::XHigh,
                    ],
                    "none, low, medium, high, xhigh",
                )
            } else if is_gpt5_3 {
                (
                    &[
                        OpenAIEffort::Low,
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                        OpenAIEffort::XHigh,
                    ],
                    "low, medium, high, xhigh",
                )
            } else if is_gpt5 {
                (
                    &[
                        OpenAIEffort::Minimal,
                        OpenAIEffort::Low,
                        OpenAIEffort::Medium,
                        OpenAIEffort::High,
                    ],
                    "minimal, low, medium, high",
                )
            } else if is_legacy_reasoning {
                (
                    &[OpenAIEffort::Low, OpenAIEffort::Medium, OpenAIEffort::High],
                    "low, medium, high",
                )
            } else {
                return Err(Error::InvalidRequest(format!(
                    "Reasoning effort configuration is not supported for model '{}'.",
                    conversation.model.id
                )));
            };

            if !allowed.contains(effort) {
                return Err(Error::InvalidRequest(format!(
                    "Reasoning effort '{:?}' is not supported for model '{}'. Supported efforts: {}.",
                    effort, conversation.model.id, supported_label
                )));
            }

            request.reasoning = Some(OpenAIReasoning {
                mode: config.openai_reasoning_mode.filter(|_| !is_chatgpt_oauth),
                effort: Some(*effort),
                summary: if *include_summaries && !self.summaries_refused.load(Ordering::Relaxed) {
                    Some("auto".to_string())
                } else {
                    None
                },
                context: reasoning_context,
            });
        } else if is_chatgpt_oauth && (is_gpt5_family || is_gpt6) {
            request.reasoning = Some(OpenAIReasoning {
                mode: None,
                effort: Some(OpenAIEffort::Medium),
                summary: None,
                context: reasoning_context,
            });
        } else if config.openai_reasoning_mode.is_some() || reasoning_context.is_some() {
            // API key: without a configured effort the model's own default
            // applies (none on GPT-5.1/5.2/5.4, high on GPT-5.5 Pro, medium on
            // GPT-5.5/5.6/6). https://developers.openai.com/api/docs/guides/reasoning
            request.reasoning = Some(OpenAIReasoning {
                mode: config.openai_reasoning_mode,
                effort: None,
                summary: None,
                context: reasoning_context,
            });
        }

        if let Some(tool_config) = config.offered_tools() {
            request.tools = Some(
                tool_config
                    .tools
                    .iter()
                    .map(|tool| OpenAIToolDef {
                        type_: "function".to_string(),
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                        strict: tool.strict,
                    })
                    .collect(),
            );

            if let Some(choice) = &tool_config.tool_choice {
                request.tool_choice = Some(match choice {
                    ToolChoice::Auto => OpenAIToolChoice::String("auto".to_string()),
                    ToolChoice::Required => OpenAIToolChoice::String("required".to_string()),
                    ToolChoice::None => OpenAIToolChoice::String("none".to_string()),
                    ToolChoice::Specific(name) => {
                        OpenAIToolChoice::Object(OpenAIToolChoiceObject {
                            type_: "function".to_string(),
                            name: name.clone(),
                        })
                    }
                });
            }

            request.parallel_tool_calls = tool_config.parallel_tool_calls;
        }

        for message in &conversation.messages {
            if let Role::Assistant = message.role {
                if let Some(ReasoningArtifact::OpenAIOutputItems { items }) =
                    &message.reasoning_artifact
                {
                    let start = request.input.len();
                    Self::replay_output_items(&mut request.input, message, items);
                    // Text the items don't hold (the application's own, such
                    // as the explanation kept for a refused turn) follows them.
                    let holds_message = items
                        .iter()
                        .any(|item| item.get("type").and_then(Value::as_str) == Some("message"));
                    let has_text = message
                        .content
                        .iter()
                        .filter_map(Content::as_text)
                        .any(|text| !text.is_empty());
                    if !holds_message && has_text {
                        request.input.push(OpenAIInputEntry::Message(
                            self.convert_message_to_openai(message)?,
                        ));
                    }
                    // A reasoning item must be followed by the item it led
                    // to; the API rejects one replayed with nothing after it
                    // ("Item 'rs_…' of type 'reasoning' was provided without
                    // its required following item", openai/codex#1498). A
                    // response cut off after its reasoning (`incomplete`,
                    // such as at `max_output_tokens`), or whose calls the
                    // application dropped, ends that way, so its trailing
                    // reasoning is left out rather than padded with an empty
                    // assistant message. The items stay stored as received.
                    while request.input.len() > start
                        && matches!(
                            request.input.last(),
                            Some(OpenAIInputEntry::Item(item))
                                if item.get("type").and_then(Value::as_str) == Some("reasoning")
                        )
                    {
                        request.input.pop();
                    }
                    continue;
                }

                // Messages without output items (saved by earlier versions,
                // built by the application, or from another provider) are
                // reconstructed: reasoning, function calls, then the text.
                if let Some(ReasoningArtifact::OpenAIEncryptedReasoning { items }) =
                    &message.reasoning_artifact
                {
                    for item in items {
                        request
                            .input
                            .push(OpenAIInputEntry::Reasoning(OpenAIInputReasoning {
                                type_: "reasoning".to_string(),
                                encrypted_content: item.encrypted_content.clone(),
                                summary: item.summaries.as_ref().map(|summaries| {
                                    summaries
                                        .iter()
                                        .map(|summary| OpenAIReasoningSummary {
                                            type_: "summary_text".to_string(),
                                            text: summary.clone(),
                                        })
                                        .collect()
                                }),
                                id: item.id.clone(),
                            }));
                    }
                }

                for tool_call in &message.tool_calls {
                    request.input.push(Self::function_call_entry(tool_call));
                }

                // A tool-only turn has no message to replay.
                let has_content = message.content.iter().any(|content| match content {
                    Content::Text(text) => !text.is_empty(),
                    Content::Image(_) => true,
                    Content::ToolResult(_) => false,
                });
                if !message.tool_calls.is_empty() && !has_content {
                    continue;
                }
            }

            if let Role::User = message.role {
                let has_tool_results = message
                    .content
                    .iter()
                    .any(|content| content.as_tool_result().is_some());

                if has_tool_results {
                    let inline = conversation.model.accepts_tool_result_images();
                    for content in &message.content {
                        if let Content::ToolResult(result) = content {
                            request.input.push(OpenAIInputEntry::FunctionCallOutput(
                                OpenAIInputFunctionCallOutput {
                                    type_: "function_call_output".to_string(),
                                    call_id: result.call_id.clone().unwrap_or_default(),
                                    output: self.function_call_output(result, inline)?,
                                },
                            ));
                        }
                    }

                    let rest = super::tool_images::content_after_results(message, inline);
                    let filtered: Vec<&Content> = rest.iter().map(AsRef::as_ref).collect();

                    if !filtered.is_empty() {
                        let message = self.convert_filtered_to_openai_message(
                            "user",
                            &filtered,
                            message.id.as_deref(),
                        )?;

                        request.input.push(OpenAIInputEntry::Message(message));
                    }

                    continue;
                }
            }

            request.input.push(OpenAIInputEntry::Message(
                self.convert_message_to_openai(message)?,
            ));
        }

        Ok(request)
    }

    /// The request body [`build_request`](Self::build_request) produces,
    /// serialized as it is sent.
    #[cfg(test)]
    pub(crate) fn request_body(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        is_chatgpt_oauth: bool,
    ) -> Result<String, Error> {
        Ok(serde_json::to_string(&self.build_request(
            conversation,
            config,
            is_chatgpt_oauth,
        )?)?)
    }

    /// ChatGPT OAuth requests can't carry an output limit or turn caching off:
    /// the backend rejects `max_output_tokens`, and caches implicitly with no
    /// option to disable it. A set limit and caching turned off are ignored,
    /// with a warning logged once each.
    fn warn_oauth_ignored_settings(conversation: &Conversation, config: &GenerationConfig) {
        if let Some(max_tokens) = config.max_tokens {
            crate::notice::warn_once("openai-oauth:max_tokens", || {
                format!(
                    "arko_ai: the ChatGPT OAuth backend rejects max_output_tokens, so GenerationConfig::max_tokens ({max_tokens}) is not sent on ChatGPT OAuth requests. The model's own output limit applies."
                )
            });
        }
        if !config.prompt_caching.unwrap_or(conversation.enable_cache) {
            crate::notice::warn_once("openai-oauth:prompt_caching", || {
                "arko_ai: the ChatGPT OAuth backend caches prompts implicitly and can't turn caching off, so disabled prompt caching has no effect on ChatGPT OAuth requests.".to_string()
            });
        }
        if config.openai_reasoning_mode.is_some() {
            crate::notice::warn_once("openai-oauth:reasoning_mode", || {
                "arko_ai: the ChatGPT OAuth backend does not accept OpenAI reasoning.mode, so GenerationConfig::openai_reasoning_mode is not sent on ChatGPT OAuth requests.".to_string()
            });
        }
    }

    /// Replays a response's stored output items verbatim and in order (see
    /// [`ReasoningArtifact::OpenAIOutputItems`]). A `function_call` item is
    /// replayed only if the message still holds that call, so every
    /// replayed call gets an output (an application that drops a call, such
    /// as an agent keeping a refused turn, drops its item too). Calls the
    /// message holds without an item follow in the reconstructed shape.
    fn replay_output_items(input: &mut Vec<OpenAIInputEntry>, message: &Message, items: &[Value]) {
        let held: HashSet<&str> = message
            .tool_calls
            .iter()
            .filter_map(|call| call.id.as_deref())
            .collect();
        let mut replayed = HashSet::new();
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                let call_id = item.get("call_id").and_then(Value::as_str);
                if !call_id.is_some_and(|id| held.contains(id) && replayed.insert(id)) {
                    continue;
                }
            }
            input.push(OpenAIInputEntry::Item(item.clone()));
        }
        for tool_call in &message.tool_calls {
            if !tool_call
                .id
                .as_deref()
                .is_some_and(|id| replayed.contains(id))
            {
                input.push(Self::function_call_entry(tool_call));
            }
        }
    }

    /// A result's `function_call_output.output`: the output text, or with
    /// images the model takes inside the result, an array of `input_text`
    /// (the output, then each image's label) and `input_image` items. Empty
    /// texts are left out.
    fn function_call_output(
        &self,
        result: &crate::tool::ToolResult,
        inline: bool,
    ) -> Result<OpenAIContent, Error> {
        if !inline || result.images.is_empty() {
            return Ok(OpenAIContent::Text(result.output.clone()));
        }
        let text = |text: &str| OpenAIContentItem {
            type_: "input_text".to_string(),
            text: Some(text.to_string()),
            image_url: None,
        };
        let mut items = Vec::new();
        if !result.output.is_empty() {
            items.push(text(&result.output));
        }
        for image in &result.images {
            if !image.label.is_empty() {
                items.push(text(&image.label));
            }
            items.push(self.image_item(&image.image)?);
        }
        Ok(OpenAIContent::Array(items))
    }

    fn function_call_entry(tool_call: &ToolCall) -> OpenAIInputEntry {
        // Invalid arguments are replayed exactly as the model wrote them.
        let arguments = match &tool_call.invalid_arguments {
            Some(raw) => raw.clone(),
            None => {
                serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string())
            }
        };

        OpenAIInputEntry::FunctionCall(OpenAIInputFunctionCall {
            type_: "function_call".to_string(),
            id: tool_call.provider_id.clone(),
            call_id: tool_call.id.clone().unwrap_or_default(),
            name: tool_call.name.clone(),
            arguments,
        })
    }

    fn convert_filtered_to_openai_message(
        &self,
        role: &str,
        contents: &[&Content],
        id: Option<&str>,
    ) -> Result<OpenAIInputMessage, Error> {
        let mut input_message = OpenAIInputMessage {
            role: role.to_string(),
            type_: "message".to_string(),
            content: None,
            id: id.map(String::from),
        };

        if contents.len() == 1 {
            match contents[0] {
                Content::Text(text) => {
                    input_message.content = Some(OpenAIContent::Text(text.clone()));
                }
                Content::Image(image) => {
                    input_message.content =
                        Some(OpenAIContent::Array(vec![self.image_item(image)?]));
                }
                Content::ToolResult(_) => {}
            }
        } else {
            let mut content_items = Vec::new();

            for content in contents {
                match content {
                    Content::Text(text) => {
                        content_items.push(OpenAIContentItem {
                            type_: "input_text".to_string(),
                            text: Some(text.clone()),
                            image_url: None,
                        });
                    }
                    Content::Image(image) => {
                        content_items.push(OpenAIContentItem {
                            type_: "input_image".to_string(),
                            text: None,
                            image_url: Some(self.convert_image_to_openai(image)?),
                        });
                    }
                    Content::ToolResult(_) => {}
                }
            }

            input_message.content = Some(OpenAIContent::Array(content_items));
        }

        Ok(input_message)
    }

    fn convert_message_to_openai(&self, message: &Message) -> Result<OpenAIInputMessage, Error> {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => {
                return Err(Error::InvalidRequest(
                    "System messages should be set in the instructions field.".to_string(),
                ));
            }
        };

        let mut input_message = OpenAIInputMessage {
            role: role.to_string(),
            type_: "message".to_string(),
            content: None,
            id: message.id.clone(),
        };

        let content: Vec<&Content> = message
            .content
            .iter()
            .filter(|content| content.as_tool_result().is_none())
            .collect();

        if content.is_empty() {
            return Ok(input_message);
        }

        if content.len() == 1 {
            if let Some(text) = content[0].as_text() {
                if role == "assistant" {
                    input_message.content = Some(OpenAIContent::Array(vec![OpenAIContentItem {
                        type_: "output_text".to_string(),
                        text: Some(text.to_string()),
                        image_url: None,
                    }]));
                } else {
                    input_message.content = Some(OpenAIContent::Text(text.to_string()));
                }
            } else if let Some(image) = content[0].as_image() {
                input_message.content = Some(OpenAIContent::Array(vec![self.image_item(image)?]));
            }
        } else {
            let mut items = Vec::new();

            for content in content {
                match content {
                    Content::Text(text) => {
                        items.push(OpenAIContentItem {
                            type_: if role == "assistant" {
                                "output_text".to_string()
                            } else {
                                "input_text".to_string()
                            },
                            text: Some(text.clone()),
                            image_url: None,
                        });
                    }
                    Content::Image(image) => {
                        items.push(OpenAIContentItem {
                            type_: "input_image".to_string(),
                            text: None,
                            image_url: Some(self.convert_image_to_openai(image)?),
                        });
                    }
                    Content::ToolResult(_) => {}
                }
            }

            input_message.content = Some(OpenAIContent::Array(items));
        }

        Ok(input_message)
    }

    fn image_item(&self, image: &Image) -> Result<OpenAIContentItem, Error> {
        Ok(OpenAIContentItem {
            type_: "input_image".to_string(),
            text: None,
            image_url: Some(self.convert_image_to_openai(image)?),
        })
    }

    fn convert_image_to_openai(&self, image: &Image) -> Result<String, Error> {
        match image {
            Image::Url(url) => Ok(url.clone()),
            Image::Base64 { data, mime_type } => Ok(format!("data:{};base64,{}", mime_type, data)),
        }
    }

    /// Returns the assistant message, any refusal texts and the output items
    /// kept for replay. Refusal text is never part of the message text.
    /// Function calls that didn't finish (an `incomplete` response cut them
    /// off) are dropped. The caller stores the replay items on the message.
    fn parse_output(output: &[Value]) -> (Message, Vec<String>, Vec<Value>) {
        let mut text = String::new();
        let mut refusals = Vec::new();
        let mut replay_items = Vec::new();
        let mut message_id = None;
        let mut tool_calls = Vec::new();

        for item in output {
            if let Some(replay) = Self::replay_item(item) {
                replay_items.push(replay);
            }
            let parsed = match OpenAIOutput::deserialize(item) {
                Ok(parsed) => parsed,
                Err(error) => {
                    log::debug!("Skipping an OpenAI output item that doesn't parse: {error}");
                    continue;
                }
            };
            match parsed {
                OpenAIOutput::Message { content, id, .. } => {
                    message_id = Some(id);

                    for part in content {
                        match part.type_.as_str() {
                            "output_text" => text.push_str(&part.text),
                            "refusal" => refusals.push(part.refusal.unwrap_or_default()),
                            _ => {}
                        }
                    }
                }
                OpenAIOutput::Reasoning { .. } => {}
                OpenAIOutput::FunctionCall {
                    id,
                    call_id,
                    name,
                    arguments,
                    status,
                } => {
                    if !matches!(status.as_deref(), None | Some("completed")) {
                        continue;
                    }

                    let mut call = ToolCall::from_argument_text(Some(call_id), name, &arguments);
                    call.provider_id = Some(id);
                    tool_calls.push(call);
                }
                OpenAIOutput::Unknown => {
                    log::debug!("Skipping an OpenAI output item of an unsupported type.");
                }
            }
        }

        let message = Message {
            role: Role::Assistant,
            content: vec![Content::Text(text)],
            reasoning_artifact: None,
            tool_calls,
            id: message_id,
            configuration_update: None,
        };

        (message, refusals, replay_items)
    }

    /// The form of an output item that later requests replay, or `None` for
    /// items that aren't replayed: reasoning without `encrypted_content`
    /// (it can't be replayed with `store: false`), unfinished function
    /// calls, and item types arko doesn't request. Like Codex, `status` is
    /// dropped. Everything else is kept as sent.
    fn replay_item(item: &Value) -> Option<Value> {
        let replayable = match item.get("type").and_then(Value::as_str)? {
            "message" => true,
            "reasoning" => item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(|content| !content.is_empty()),
            "function_call" => {
                matches!(
                    item.get("status").and_then(Value::as_str),
                    None | Some("completed")
                )
            }
            _ => false,
        };
        if !replayable {
            return None;
        }
        let mut item = item.clone();
        item.as_object_mut()?.remove("status");
        Some(item)
    }

    /// `input_tokens` already includes cached and written tokens and
    /// `output_tokens` includes reasoning tokens. On the API, the response's
    /// `service_tier` is the tier actually used (for example `default` when a
    /// Fast request was downgraded). Over ChatGPT OAuth the requested tier
    /// replaces it (see [`chatgpt_oauth_requested_tier`](Self::chatgpt_oauth_requested_tier)).
    /// https://developers.openai.com/api/docs/guides/prompt-caching
    fn calculate_usage(response: &OpenAIResponse) -> UsageInfo {
        let mut usage = UsageInfo::new(response.usage.input_tokens, response.usage.output_tokens);

        if let Some(details) = &response.usage.input_tokens_details {
            if details.cached_tokens > 0 {
                usage = usage.with_cached_tokens(details.cached_tokens);
            }

            if details.cache_write_tokens > 0 {
                usage = usage.with_cache_write_tokens(details.cache_write_tokens);
            }
        }

        usage.reasoning_tokens = response
            .usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens);
        usage.service_tier = response
            .service_tier
            .as_deref()
            .map(ServiceTier::from_openai);

        usage
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIInputEntry {
    Message(OpenAIInputMessage),
    Reasoning(OpenAIInputReasoning),
    FunctionCall(OpenAIInputFunctionCall),
    FunctionCallOutput(OpenAIInputFunctionCallOutput),
    /// A stored output item, replayed verbatim.
    Item(Value),
}

#[derive(Debug, Serialize)]
struct OpenAIInputFunctionCall {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAIInputFunctionCallOutput {
    #[serde(rename = "type")]
    type_: String,
    call_id: String,
    /// A string, or with result images an array of content items.
    output: OpenAIContent,
}

#[derive(Debug, Serialize)]
struct OpenAIToolDef {
    #[serde(rename = "type")]
    type_: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIToolChoice {
    String(String),
    Object(OpenAIToolChoiceObject),
}

#[derive(Debug, Serialize)]
struct OpenAIToolChoiceObject {
    #[serde(rename = "type")]
    type_: String,
    name: String,
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    // The tier right after the model, ahead of a large `input`, as Codex
    // orders its routing fields: gateways may read the body incrementally
    // (openai/codex `codex-rs/codex-api/src/common.rs`, `ResponsesApiRequest`).
    // `stream` stays in place, so bodies without a tier are unchanged.
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<OpenAIServiceTier>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAIReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<OpenAITextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    input: Vec<OpenAIInputEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAIToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<OpenAIPromptCacheOptions>,
}

#[derive(Debug, Serialize)]
struct OpenAIPromptCacheOptions {
    mode: OpenAIPromptCacheMode,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum OpenAIPromptCacheMode {
    Implicit,
    Explicit,
}

#[derive(Debug, Serialize)]
struct OpenAIReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<OpenAIReasoningMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<OpenAIEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<OpenAIReasoningContext>,
}

/// Which earlier reasoning the model sees (`reasoning.context`).
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum OpenAIReasoningContext {
    AllTurns,
}

#[derive(Debug, Serialize)]
struct OpenAITextConfig {
    verbosity: crate::config::Verbosity,
}

#[derive(Debug, Serialize)]
struct OpenAIInputMessage {
    role: String,
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIInputReasoning {
    #[serde(rename = "type")]
    type_: String,
    encrypted_content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<Vec<OpenAIReasoningSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIReasoningSummary {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    Array(Vec<OpenAIContentItem>),
}

#[derive(Debug, Serialize)]
struct OpenAIContentItem {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIResponse {
    id: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    created_at: u64,
    status: String,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    incomplete_details: Option<Value>,
    #[serde(default)]
    model: String,
    /// Output items as sent. See [`OpenAIClient::assemble_output`].
    #[serde(default)]
    output: Vec<Value>,
    usage: OpenAIUsage,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    metadata: serde_json::Value,
    /// Processing tier actually used.
    #[serde(default)]
    service_tier: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum OpenAIOutput {
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(default)]
        status: Option<String>,
        content: Vec<OpenAIOutputContent>,
        #[serde(default)]
        role: Option<String>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        encrypted_content: Option<String>,
        #[serde(default)]
        summary: Option<Vec<OpenAISummary>>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default)]
        status: Option<String>,
    },
    /// Output items this client doesn't handle (for example built-in tool
    /// calls) are skipped.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIOutputContent {
    #[serde(rename = "type")]
    type_: String,
    #[serde(default)]
    annotations: Vec<serde_json::Value>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    refusal: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAISummary {
    #[serde(rename = "type")]
    type_: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    input_tokens: usize,
    #[serde(default)]
    input_tokens_details: Option<OpenAIInputTokensDetails>,
    output_tokens: usize,
    #[serde(default)]
    output_tokens_details: Option<OpenAIOutputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenAIOutputTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OpenAIInputTokensDetails {
    #[serde(default)]
    cached_tokens: usize,
    #[serde(default)]
    cache_write_tokens: usize,
}

pub fn list_models() -> Vec<crate::model::Model> {
    crate::model::Model::list_openai_models()
}

#[cfg(test)]
mod tests {
    use super::OpenAIClient;
    use crate::error::{parse_relative_duration, retry_after_from_body};
    use crate::{
        config::{GenerationConfig, Verbosity},
        conversation::Conversation,
        error::Error,
        model::{Model, Provider},
        reasoning::{OpenAIEffort, OpenAIReasoningMode, ReasoningConfig},
    };
    use std::time::Duration;

    #[test]
    fn tool_images_serialize_after_all_correlated_results_for_api_and_oauth() {
        use crate::{Content, Image, Message, ToolCall, ToolResult};
        let client = OpenAIClient::new("test");
        let mut conversation =
            Conversation::new(Model::get(Provider::OpenAI, "gpt-6-astra").unwrap());
        let calls = ["first", "second"].map(|id| ToolCall {
            id: Some(id.into()),
            name: "view_image".into(),
            arguments: serde_json::json!({"path":"fixture.png"}),
            thought_signature: None,
            provider_id: None,
            invalid_arguments: None,
        });
        let mut assistant = Message::new_assistant("Inspecting images.");
        assistant.tool_calls = calls.to_vec();
        conversation.messages.push(assistant);
        let mut content: Vec<_> = calls
            .iter()
            .map(|call| {
                Content::ToolResult(ToolResult::from_tool_call(call, "fixture.png, 12 × 8"))
            })
            .collect();
        content.extend([
            Content::Text("Image from first call: fixture.png".into()),
            Content::Image(Image::new_base64("cGl4ZWxz", "image/png")),
        ]);
        conversation.add_user_message_with_contents(content);
        let saved = serde_json::to_string(&conversation).unwrap();
        let conversation: Conversation = serde_json::from_str(&saved).unwrap();
        for oauth in [false, true] {
            let value = serde_json::to_value(
                client
                    .build_request(&conversation, &GenerationConfig::default(), oauth)
                    .unwrap(),
            )
            .unwrap();
            let input = value["input"].as_array().unwrap();
            let results: Vec<_> = input
                .iter()
                .enumerate()
                .filter(|(_, item)| item["type"] == "function_call_output")
                .collect();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].1["call_id"], "first");
            assert_eq!(results[1].1["call_id"], "second");
            let (image_index, image_message) = input
                .iter()
                .enumerate()
                .find(|(_, item)| {
                    item["content"].as_array().is_some_and(|content| {
                        content.iter().any(|block| block["type"] == "input_image")
                    })
                })
                .unwrap();
            assert!(image_index > results[1].0);
            assert_eq!(
                image_message["content"][1]["image_url"],
                "data:image/png;base64,cGl4ZWxz"
            );
            assert_eq!(results[0].1["output"], "fixture.png, 12 × 8");
        }
    }

    #[test]
    fn response_errors_preserve_status_body_delays_and_permanent_codes() {
        let error = super::response_error(
            "test",
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            None,
            r#"{"error":{"code":"server_error","retry_after_ms":1500}}"#,
        );
        assert!(error.is_retryable());
        assert_eq!(error.retry_after(), Some(Duration::from_millis(1500)));
        let error = super::response_error(
            "test",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            Some(Duration::from_secs(60)),
            r#"{"retry_after":1}"#,
        );
        assert_eq!(error.retry_after(), Some(Duration::from_secs(60)));
        let error = super::response_error(
            "refresh",
            reqwest::StatusCode::BAD_REQUEST,
            None,
            r#"{"error":"invalid_grant"}"#,
        );
        assert!(!error.is_retryable());
        for body in [
            r#"{"retry_after":-1}"#,
            r#"{"retry_after":"NaN"}"#,
            r#"{"retry_after":"inf"}"#,
        ] {
            assert_eq!(retry_after_from_body(body), None);
        }
    }

    #[tokio::test]
    async fn refresh_http_failure_returns_to_caller_with_retry_metadata() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/token", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            for (status, body) in [
                (
                    "429 Too Many Requests",
                    r#"{"error":"temporarily_unavailable"}"#,
                ),
                (
                    "200 OK",
                    r#"{"access_token":"synthetic-new-access","refresh_token":"synthetic-new-refresh"}"#,
                ),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut bytes = [0; 1024];
                    let count = socket.read(&mut bytes).await.unwrap();
                    assert!(count > 0);
                    request.extend_from_slice(&bytes[..count]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]);
                        let length: usize = headers
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse().unwrap())
                            })
                            .unwrap();
                        if request.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nRetry-After: 60\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let client = crate::providers::http_client_builder()
            .no_proxy()
            .build()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            let error = super::refresh_chatgpt_oauth_token(
                &client,
                "synthetic-refresh",
                Some("test-client"),
                Some(&url),
            )
            .await
            .unwrap_err();
            assert!(error.is_retryable());
            assert_eq!(error.retry_after(), Some(Duration::from_secs(60)));
            // The provider returned immediately. The caller explicitly decides
            // when to make the next request; no hidden sixty-second retry loop.
            let result = super::refresh_chatgpt_oauth_token(
                &client,
                "synthetic-refresh",
                Some("test-client"),
                Some(&url),
            )
            .await
            .unwrap();
            assert_eq!(result.access_token, "synthetic-new-access");
            assert_eq!(
                result.refresh_token.as_deref(),
                Some("synthetic-new-refresh")
            );
            server.await.unwrap();
        })
        .await
        .unwrap();
    }

    #[test]
    fn gpt_6_astra_builds_reasoning_and_cache_requests() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::get(Provider::OpenAI, "gpt-6-astra").unwrap());

        for effort in [
            OpenAIEffort::Low,
            OpenAIEffort::Medium,
            OpenAIEffort::High,
            OpenAIEffort::XHigh,
            OpenAIEffort::Max,
        ] {
            let config = GenerationConfig::default()
                .with_verbosity(Verbosity::Low)
                .with_openai_reasoning_mode(OpenAIReasoningMode::Pro)
                .with_reasoning(ReasoningConfig::openai(effort, true));

            for oauth in [false, true] {
                let request = client.build_request(&conversation, &config, oauth).unwrap();
                let value = serde_json::to_value(request).unwrap();

                assert_eq!(value["model"], "gpt-6-astra");
                // `all_turns` over OAuth only (Codex parity); undocumented
                // for GPT-6 with API keys.
                if oauth {
                    assert_eq!(value["reasoning"]["context"], "all_turns");
                } else {
                    assert!(value["reasoning"].get("context").is_none());
                }
                assert_eq!(value["stream"], oauth);
                assert_eq!(
                    value["reasoning"]["mode"],
                    if oauth {
                        serde_json::Value::Null
                    } else {
                        serde_json::json!("pro")
                    }
                );
                assert_eq!(
                    value["reasoning"]["effort"],
                    serde_json::to_value(effort).unwrap()
                );
                assert_eq!(value["reasoning"]["summary"], "auto");
                assert_eq!(value["text"]["verbosity"], "low");
                assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
                assert!(value.get("temperature").is_none());
                if oauth {
                    assert!(value.get("prompt_cache_options").is_none());
                } else {
                    assert_eq!(value["prompt_cache_options"]["mode"], "implicit");
                }
            }
        }

        for effort in [OpenAIEffort::None, OpenAIEffort::Minimal] {
            let config =
                GenerationConfig::default().with_reasoning(ReasoningConfig::openai(effort, false));
            assert!(matches!(
                client.build_request(&conversation, &config, false),
                Err(Error::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn gpt_6_sol_terra_and_luna_build_reasoning_verbosity_and_cache_requests() {
        let client = OpenAIClient::new("test");

        for id in ["gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"] {
            let conversation = Conversation::new(Model::get(Provider::OpenAI, id).unwrap());

            for effort in [
                OpenAIEffort::None,
                OpenAIEffort::Low,
                OpenAIEffort::Medium,
                OpenAIEffort::High,
                OpenAIEffort::XHigh,
                OpenAIEffort::Max,
            ] {
                let config = GenerationConfig::default()
                    .with_verbosity(Verbosity::Low)
                    .with_openai_reasoning_mode(OpenAIReasoningMode::Pro)
                    .with_reasoning(ReasoningConfig::openai(effort, true));

                for oauth in [false, true] {
                    let request = client.build_request(&conversation, &config, oauth).unwrap();
                    let value = serde_json::to_value(request).unwrap();

                    assert_eq!(value["model"], id);
                    if oauth {
                        assert_eq!(value["reasoning"]["context"], "all_turns");
                    } else {
                        assert!(value["reasoning"].get("context").is_none());
                    }
                    assert_eq!(
                        value["reasoning"]["mode"],
                        if oauth {
                            serde_json::Value::Null
                        } else {
                            serde_json::json!("pro")
                        }
                    );
                    assert_eq!(
                        value["reasoning"]["effort"],
                        serde_json::to_value(effort).unwrap()
                    );
                    assert_eq!(value["reasoning"]["summary"], "auto");
                    assert_eq!(value["text"]["verbosity"], "low");
                    assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
                    assert!(value.get("temperature").is_none());
                    if oauth {
                        assert!(value.get("prompt_cache_options").is_none());
                    } else {
                        assert_eq!(value["prompt_cache_options"]["mode"], "implicit");
                    }
                }
            }

            let config = GenerationConfig::default()
                .with_reasoning(ReasoningConfig::openai(OpenAIEffort::Minimal, false));
            assert!(matches!(
                client.build_request(&conversation, &config, false),
                Err(Error::InvalidRequest(_))
            ));

            // API key: no effort configured means the model default applies,
            // and no reasoning object is sent.
            let request = client
                .build_request(&conversation, &GenerationConfig::default(), false)
                .unwrap();
            let value = serde_json::to_value(request).unwrap();
            assert!(value.get("reasoning").is_none());

            // ChatGPT OAuth keeps sending medium (owned by the OAuth request path).
            let request = client
                .build_request(&conversation, &GenerationConfig::default(), true)
                .unwrap();
            let value = serde_json::to_value(request).unwrap();
            assert_eq!(value["reasoning"]["effort"], "medium");
            assert_eq!(value["reasoning"]["context"], "all_turns");
            assert!(value["reasoning"].get("mode").is_none());
        }
    }

    #[test]
    fn parses_retry_after_body_shapes() {
        assert_eq!(
            retry_after_from_body(r#"{"error":{"retry_after_ms":2500}}"#),
            Some(Duration::from_millis(2_500))
        );
        assert_eq!(
            retry_after_from_body(r#"{"retry_after":3}"#),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn parses_compound_reset_header_duration() {
        assert_eq!(
            parse_relative_duration("1m2.5s"),
            Some(Duration::from_millis(62_500))
        );
    }

    #[test]
    fn oauth_rate_limit_retries_are_opt_in() {
        let client =
            OpenAIClient::new_chatgpt_oauth("access-token", Some("refresh-token"), "account-id");
        assert!(!client.retry_oauth_rate_limits);

        let client = client.with_oauth_rate_limit_retries(true);
        assert!(client.retry_oauth_rate_limits);
    }

    #[test]
    fn oauth_request_model_id_strips_gpt_5_4_snapshot_suffixes() {
        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.4-2026-03-05", true),
            "gpt-5.4"
        );

        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.4-mini-2026-03-17", true),
            "gpt-5.4-mini"
        );

        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.4-nano-2026-03-17", true),
            "gpt-5.4-nano"
        );

        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.4-pro-2026-03-05", true),
            "gpt-5.4-pro"
        );
    }

    #[test]
    fn oauth_request_model_id_strips_gpt_5_5_snapshot_suffix() {
        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.5-2026-04-23", true),
            "gpt-5.5"
        );
    }

    #[test]
    fn request_model_id_preserves_api_key_and_shorthand_ids() {
        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.5-2026-04-23", false),
            "gpt-5.5-2026-04-23"
        );

        assert_eq!(OpenAIClient::request_model_id("gpt-5.5", true), "gpt-5.5");

        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.4-mini", true),
            "gpt-5.4-mini"
        );
    }

    #[test]
    fn gpt_5_6_defaults_to_implicit_prompt_caching() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default();

        let request = client.build_request(&conversation, &config, false).unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["prompt_cache_options"]["mode"], "implicit");
        assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
    }

    #[test]
    fn gpt_5_6_oauth_sends_cache_key_without_cache_options() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default();

        let request = client.build_request(&conversation, &config, true).unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
        assert!(value.get("prompt_cache_options").is_none());
    }

    /// The ChatGPT backend rejects `max_output_tokens` and caches implicitly:
    /// an explicit limit and caching turned off are dropped, each with a
    /// warning logged once.
    #[test]
    fn oauth_drops_an_output_limit_and_caching_off_with_a_warning() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_6_astra());
        let config = GenerationConfig::default()
            .with_max_tokens(16_000)
            .with_prompt_caching(false);
        let api =
            serde_json::to_value(client.build_request(&conversation, &config, false).unwrap())
                .unwrap();
        assert_eq!(api["max_output_tokens"], 16_000);
        assert_eq!(api["prompt_cache_options"]["mode"], "explicit");
        assert!(!crate::notice::warned("openai-oauth:max_tokens"));
        assert!(!crate::notice::warned("openai-oauth:prompt_caching"));

        let oauth =
            serde_json::to_value(client.build_request(&conversation, &config, true).unwrap())
                .unwrap();
        assert!(oauth.get("max_output_tokens").is_none(), "{oauth}");
        assert!(oauth.get("prompt_cache_options").is_none(), "{oauth}");
        assert!(crate::notice::warned("openai-oauth:max_tokens"));
        assert!(crate::notice::warned("openai-oauth:prompt_caching"));
    }

    #[test]
    fn oauth_defaults_warn_about_nothing() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_6_astra());
        client
            .build_request(&conversation, &GenerationConfig::default(), true)
            .unwrap();
        assert!(!crate::notice::warned("openai-oauth:max_tokens"));
        assert!(!crate::notice::warned("openai-oauth:prompt_caching"));
        // A conversation with caching disabled is caching turned off too.
        client
            .build_request(
                &conversation.clone().with_cache_disabled(),
                &GenerationConfig::default(),
                true,
            )
            .unwrap();
        assert!(crate::notice::warned("openai-oauth:prompt_caching"));
    }

    #[test]
    fn gpt_5_6_family_sends_cache_key_when_enabled() {
        let client = OpenAIClient::new("test");
        let models = [
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_5_6_terra(),
            Model::openai_gpt_5_6_luna(),
        ];
        let config = GenerationConfig::default().with_prompt_caching(true);

        for model in models {
            let conversation = Conversation::new(model);
            let request = client.build_request(&conversation, &config, false).unwrap();
            let value = serde_json::to_value(request).unwrap();

            assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
            assert_eq!(value["prompt_cache_options"]["mode"], "implicit");
        }
    }

    #[test]
    fn gpt_5_6_can_disable_prompt_caching() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default().with_prompt_caching(false);

        let request = client.build_request(&conversation, &config, false).unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["prompt_cache_options"]["mode"], "explicit");
    }

    #[test]
    fn older_openai_models_send_the_cache_key_without_cache_options() {
        let client = OpenAIClient::new("test");
        let config = GenerationConfig::default().with_prompt_caching(false);

        for model in [Model::openai_gpt_5_5(), Model::openai_gpt_4_1()] {
            let conversation = Conversation::new(model);
            for oauth in [false, true] {
                let request = client.build_request(&conversation, &config, oauth).unwrap();

                let value = serde_json::to_value(request).unwrap();

                assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
                assert!(value.get("prompt_cache_options").is_none());
                // Only GPT-5.6 (and GPT-6 over ChatGPT OAuth) get
                // `reasoning.context`.
                assert!(value.pointer("/reasoning/context").is_none());
            }
        }
    }

    #[test]
    fn gpt_5_6_pro_mode_serializes_for_all_model_tiers() {
        let client = OpenAIClient::new("test");
        let models = [
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_5_6_terra(),
            Model::openai_gpt_5_6_luna(),
        ];
        let config =
            GenerationConfig::default().with_openai_reasoning_mode(OpenAIReasoningMode::Pro);

        for model in models {
            let conversation = Conversation::new(model);

            let request = client.build_request(&conversation, &config, false).unwrap();

            let value = serde_json::to_value(request).unwrap();

            assert_eq!(value["reasoning"]["mode"], "pro");
            assert!(value["reasoning"].get("effort").is_none());
        }
    }

    #[test]
    fn gpt_5_6_standard_mode_can_be_explicit() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let config =
            GenerationConfig::default().with_openai_reasoning_mode(OpenAIReasoningMode::Standard);

        let request = client.build_request(&conversation, &config, false).unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["reasoning"]["mode"], "standard");
    }

    #[test]
    fn gpt_5_6_reasoning_mode_is_independent_of_effort() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default()
            .with_openai_reasoning_mode(OpenAIReasoningMode::Pro)
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::Max, true));

        let request = client.build_request(&conversation, &config, false).unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["reasoning"]["mode"], "pro");
        assert_eq!(value["reasoning"]["effort"], "max");
        assert_eq!(value["reasoning"]["summary"], "auto");
    }

    #[test]
    fn gpt_5_6_omits_reasoning_mode_by_default() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default();

        let request = client.build_request(&conversation, &config, false).unwrap();

        let value = serde_json::to_value(request).unwrap();

        // API key: no effort is sent, only the reasoning context.
        assert!(value["reasoning"].get("mode").is_none());
        assert!(value["reasoning"].get("effort").is_none());
        assert_eq!(value["reasoning"]["context"], "all_turns");
    }

    #[test]
    fn older_openai_models_reject_reasoning_mode() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_5_5());
        let config =
            GenerationConfig::default().with_openai_reasoning_mode(OpenAIReasoningMode::Pro);

        let error = client
            .build_request(&conversation, &config, false)
            .unwrap_err();

        match error {
            Error::InvalidRequest(message) => {
                assert_eq!(
                    message,
                    "The OpenAI reasoning mode parameter is only supported for GPT-5.6 and GPT-6 models."
                );
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn temperature_is_sent_only_when_specified() {
        let client = OpenAIClient::new("test");
        let conversation = Conversation::new(Model::openai_gpt_4_1());

        let request = client
            .build_request(&conversation, &GenerationConfig::default(), false)
            .unwrap();
        assert!(
            serde_json::to_value(request)
                .unwrap()
                .get("temperature")
                .is_none()
        );

        let request = client
            .build_request(
                &conversation,
                &GenerationConfig::default().with_temperature(0.3),
                false,
            )
            .unwrap();
        assert_eq!(serde_json::to_value(request).unwrap()["temperature"], 0.3);
    }

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn service_tier_requests_are_validated_per_model_and_auth() {
        use crate::config::OpenAIServiceTier;
        let client = OpenAIClient::new("test");
        let astra = Conversation::new(Model::openai_gpt_6_astra());
        let build = |oauth: bool, model: Model, tier: OpenAIServiceTier| {
            let config = GenerationConfig::default().with_openai_service_tier(tier);
            client
                .build_request(&Conversation::new(model), &config, oauth)
                .map(|request| serde_json::to_value(request).unwrap())
        };
        for (requested, wire) in [
            (OpenAIServiceTier::Flex, "flex"),
            (OpenAIServiceTier::Fast, "priority"),
            (OpenAIServiceTier::Auto, "auto"),
            (OpenAIServiceTier::Default, "default"),
        ] {
            let api_key = build(false, Model::openai_gpt_6_astra(), requested).unwrap();
            assert_eq!(api_key["service_tier"], wire);
        }
        let unset = client
            .build_request(&astra, &GenerationConfig::default(), false)
            .unwrap();
        assert!(
            serde_json::to_value(unset)
                .unwrap()
                .get("service_tier")
                .is_none()
        );

        // ChatGPT OAuth: Fast mode as Codex sends it, on every model its
        // catalog offers it for (GPT-5.5 by its undated ID)...
        for model in [
            Model::openai_gpt_6_astra(),
            Model::openai_gpt_6_sol(),
            Model::openai_gpt_6_1_sol(),
            Model::openai_gpt_6_1_terra(),
            Model::openai_gpt_6_luna(),
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_5_6_terra(),
            Model::openai_gpt_5_6_luna(),
            Model::openai_gpt_5_5(),
        ] {
            let id = model.id.clone();
            let oauth = build(true, model, OpenAIServiceTier::Fast).unwrap();
            assert_eq!(oauth["service_tier"], "priority", "{id}");
        }
        assert_eq!(
            build(true, Model::openai_gpt_5_5(), OpenAIServiceTier::Fast).unwrap()["model"],
            "gpt-5.5"
        );
        // ...and not on others, even where the API offers it.
        for model in [
            Model::openai_gpt_5_4(),
            Model::openai_gpt_5_4_nano(),
            Model::openai_gpt_5_6(),
            Model::openai_gpt_5_5_pro(),
        ] {
            let id = model.id.clone();
            assert!(
                matches!(
                    build(true, model, OpenAIServiceTier::Fast),
                    Err(Error::InvalidRequest(_))
                ),
                "{id}"
            );
        }
        // Default is standard routing, which Codex expresses by sending no
        // tier; Flex and Auto stay API-key only.
        let default = build(
            true,
            Model::openai_gpt_6_astra(),
            OpenAIServiceTier::Default,
        )
        .unwrap();
        assert!(default.get("service_tier").is_none());
        for refused in [OpenAIServiceTier::Flex, OpenAIServiceTier::Auto] {
            assert!(matches!(
                build(true, Model::openai_gpt_6_astra(), refused),
                Err(Error::UnsupportedFeature(_))
            ));
        }

        // The tier follows the model, ahead of the input, as Codex orders it.
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("hello");
        let config = GenerationConfig::default().with_openai_service_tier(OpenAIServiceTier::Fast);
        let body =
            serde_json::to_string(&client.build_request(&conversation, &config, true).unwrap())
                .unwrap();
        assert!(
            body.starts_with(r#"{"model":"gpt-6-astra","service_tier":"priority","#),
            "{body}"
        );
        // gpt-4.1 has no Flex; gpt-5.4-nano has no Fast mode.
        for (model, tier) in [
            (Model::openai_gpt_4_1(), OpenAIServiceTier::Flex),
            (Model::openai_gpt_5_4_nano(), OpenAIServiceTier::Fast),
        ] {
            let config = GenerationConfig::default().with_openai_service_tier(tier);
            assert!(matches!(
                client.build_request(&Conversation::new(model), &config, false),
                Err(Error::InvalidRequest(_))
            ));
        }
    }

    fn response(service_tier: Option<&str>) -> super::OpenAIResponse {
        serde_json::from_value(serde_json::json!({
            "id": "resp_1", "status": "completed", "model": "gpt-6-astra",
            "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 100_000, "output_tokens": 1_000_000,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 400_000}},
            "service_tier": service_tier,
        }))
        .unwrap()
    }

    fn priced(
        service_tier: Option<&str>,
        config: &GenerationConfig,
        oauth: bool,
    ) -> crate::usage::UsageInfo {
        OpenAIClient::generation_response(
            &response(service_tier),
            &Model::openai_gpt_6_astra(),
            config,
            oauth,
            super::StreamedOutput::default(),
            &super::ResponseHead::default(),
        )
        .usage
    }

    #[test]
    fn reported_service_tier_and_reasoning_tokens_price_the_call() {
        use crate::config::OpenAIServiceTier;
        let standard = GenerationConfig::default();
        let flex = GenerationConfig::default().with_openai_service_tier(OpenAIServiceTier::Flex);
        // GPT-6 Astra short context: 0.1M input at $10 + 1M output at $50.
        let usage = priced(None, &standard, false);
        assert_eq!(usage.reasoning_tokens, Some(400_000));
        assert!(close(usage.total_cost, 51.0));
        assert!(close(
            priced(Some("priority"), &standard, false).total_cost,
            102.0
        ));
        // The requested tier applies only when the response doesn't report one;
        // a Flex request served at the default tier is priced as standard.
        assert!(close(priced(None, &flex, false).total_cost, 25.5));
        let downgraded = priced(Some("default"), &flex, false);
        assert_eq!(
            downgraded.service_tier,
            Some(crate::pricing::ServiceTier::Standard)
        );
        assert!(close(downgraded.total_cost, 51.0));
    }

    #[test]
    fn chatgpt_oauth_usage_reports_the_backend_input_limit() {
        let usage = |oauth| {
            OpenAIClient::generation_response(
                &response(None),
                &Model::openai_gpt_6_astra(),
                &GenerationConfig::default(),
                oauth,
                super::StreamedOutput::default(),
                &super::ResponseHead::default(),
            )
            .model_call_usage
        };
        // 100,000 input tokens of GPT-6's 922,000 on the API...
        let api_key = usage(false);
        assert_eq!(api_key.max_input_tokens, Some(922_000));
        // ...and of the ChatGPT backend's 272,000 over OAuth.
        let oauth = usage(true);
        assert_eq!(
            oauth.max_input_tokens,
            Some(crate::input_limit::CHATGPT_OAUTH_MAX_INPUT_TOKENS)
        );
        assert!(close(
            oauth.context_usage_percent.unwrap(),
            100_000.0 / 272_000.0 * 100.0
        ));
    }

    #[test]
    fn oauth_refresh_backoff_doubles_to_a_cap_and_honors_server_hints() {
        let retry = super::OAUTH_REFRESH_RETRY;
        for sample in [0, 1, 7_777, u64::MAX / 3, u64::MAX] {
            let first = retry.delay(None, 0, sample);
            assert!(first >= Duration::from_secs(15) && first <= Duration::from_secs(30));
            let late = retry.delay(None, 20, sample);
            assert!(late >= Duration::from_secs(450) && late <= Duration::from_secs(900));
            // A body hint reaches the delay through the error's retry_after.
            let hint = Error::provider_error("refresh", Some(429), None, r#"{"retry_after":600}"#)
                .retry_after();
            let hinted = retry.delay(hint, 0, sample);
            assert!(hinted >= Duration::from_secs(600) && hinted <= Duration::from_secs(615));
            // One wait honors at most 15 minutes of a hint (plus jitter),
            // even for the 30-day cap on parsed hints.
            for hint in [
                Duration::from_secs(1200),
                crate::retry::MAX_SERVER_DELAY_HINT,
            ] {
                let capped = retry.delay(Some(hint), 0, sample);
                assert!(capped >= Duration::from_secs(900) && capped <= Duration::from_secs(915));
            }
        }
    }

    #[test]
    fn chatgpt_oauth_calls_keep_estimates_but_bill_nothing() {
        let usage = priced(None, &GenerationConfig::default(), true);
        assert_eq!(usage.billing, crate::usage::BillingKind::Subscription);
        assert!(close(usage.total_cost, 51.0));
        assert!(close(usage.input_cost, 1.0));
        assert_eq!(usage.billed(), 0.0);

        // Fast mode over OAuth: estimated at the API's Fast rates ("increased
        // usage" of the plan, per Codex), still not billed. The requested
        // tier is reported whatever the response echoes: the ChatGPT backend
        // echoes `default` or `auto` for Fast calls served fast, and Codex
        // never reads the echo.
        let fast = GenerationConfig::default()
            .with_openai_service_tier(crate::config::OpenAIServiceTier::Fast);
        for reported in [Some("priority"), None, Some("default"), Some("auto")] {
            let usage = priced(reported, &fast, true);
            assert_eq!(usage.service_tier, Some(crate::pricing::ServiceTier::Fast));
            assert_eq!(usage.billing, crate::usage::BillingKind::Subscription);
            assert!(close(usage.total_cost, 102.0));
            assert_eq!(usage.billed(), 0.0);
        }
        // Without Fast, no tier is sent: standard, even if `priority` is
        // echoed.
        for reported in [Some("priority"), None, Some("default")] {
            let usage = priced(reported, &GenerationConfig::default(), true);
            assert_eq!(
                usage.service_tier,
                Some(crate::pricing::ServiceTier::Standard)
            );
            assert!(close(usage.total_cost, 51.0));
        }
        // On the API the echo still wins: a Fast request served at the
        // default tier is priced as standard.
        let downgraded = priced(Some("default"), &fast, false);
        assert_eq!(
            downgraded.service_tier,
            Some(crate::pricing::ServiceTier::Standard)
        );
        assert!(close(downgraded.total_cost, 51.0));
    }
}

#[cfg(test)]
mod model_rule_tests {
    use super::OpenAIClient;
    use crate::{
        config::GenerationConfig,
        conversation::Conversation,
        error::Error,
        model::Model,
        reasoning::{OpenAIEffort, ReasoningConfig},
    };
    use serde_json::Value;

    fn request(model: Model, config: &GenerationConfig) -> Result<Value, Error> {
        let request =
            OpenAIClient::new("test").build_request(&Conversation::new(model), config, false)?;
        Ok(serde_json::to_value(request).unwrap())
    }

    fn accepts(model: &Model, effort: OpenAIEffort) -> bool {
        let config =
            GenerationConfig::default().with_reasoning(ReasoningConfig::openai(effort, false));
        request(model.clone(), &config).is_ok()
    }

    #[test]
    fn pro_models_have_their_own_effort_rules() {
        use OpenAIEffort::{High, Low, Max, Medium, Minimal, XHigh};
        for model in [
            Model::openai_gpt_5_5_pro(),
            Model::openai_gpt_5_2_pro(),
            Model::openai_gpt_5_4_pro(),
        ] {
            for effort in [Medium, High, XHigh] {
                assert!(accepts(&model, effort), "{} {effort:?}", model.id);
            }
            for effort in [OpenAIEffort::None, Minimal, Low, Max] {
                assert!(!accepts(&model, effort), "{} {effort:?}", model.id);
            }
        }
        let gpt_5_pro = Model::openai_gpt_5_pro();
        assert!(accepts(&gpt_5_pro, High));
        for effort in [Minimal, Low, Medium, XHigh] {
            assert!(!accepts(&gpt_5_pro, effort), "{effort:?}");
        }
    }

    #[test]
    fn gpt_5_1_supports_none_through_high_but_not_xhigh() {
        use OpenAIEffort::{High, Low, Medium, XHigh};
        let model = Model::openai_gpt_5_1();
        for effort in [OpenAIEffort::None, Low, Medium, High] {
            assert!(accepts(&model, effort), "{effort:?}");
        }
        assert!(!accepts(&model, XHigh));
        assert!(accepts(&Model::openai_gpt_5_2(), XHigh));
    }

    #[test]
    fn api_key_requests_without_effort_use_the_model_default() {
        for model in [
            Model::openai_gpt_5_1(),
            Model::openai_gpt_5_4(),
            Model::openai_gpt_5_5_pro(),
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_6_astra(),
        ] {
            let value = request(model.clone(), &GenerationConfig::default()).unwrap();
            assert!(value["reasoning"].get("effort").is_none(), "{}", model.id);
            // GPT-5.6 still asks for reasoning from all turns (documented for
            // API keys); the others send no reasoning object at all.
            if model.is_openai_gpt_5_6() {
                assert_eq!(
                    value["reasoning"],
                    serde_json::json!({"context": "all_turns"})
                );
            } else {
                assert!(value.get("reasoning").is_none(), "{}", model.id);
            }
            assert_eq!(value["include"][0], "reasoning.encrypted_content");
        }
    }

    #[test]
    fn gpt_5_6_alias_shares_sol_data_and_rules() {
        let alias = Model::openai_gpt_5_6();
        let sol = Model::openai_gpt_5_6_sol();
        assert_eq!(
            alias.pricing.input_cost_per_million,
            sol.pricing.input_cost_per_million
        );
        assert!(alias.is_openai_gpt_5_6());
        assert!(accepts(&alias, OpenAIEffort::Max));
    }
}

#[cfg(test)]
mod input_capability_tests {
    use super::OpenAIClient;
    use crate::{
        config::GenerationConfig,
        conversation::Conversation,
        error::Error,
        message::{Content, Image},
        model::Model,
    };

    #[test]
    fn images_are_rejected_before_sending_to_models_without_image_input() {
        let mut model = Model::openai_gpt_4_1();
        let image = || vec![Content::Image(Image::new_url("https://example.com/a.png"))];
        let mut conversation = Conversation::new(model.clone());
        conversation.add_user_message_with_contents(image());
        let client = OpenAIClient::new("test");
        assert!(
            client
                .build_request(&conversation, &GenerationConfig::default(), false)
                .is_ok()
        );

        model.supports_image_input = false;
        let mut conversation = Conversation::new(model);
        conversation.add_user_message_with_contents(image());
        assert!(matches!(
            client.build_request(&conversation, &GenerationConfig::default(), false),
            Err(Error::UnsupportedFeature(_))
        ));
    }

    #[test]
    fn partial_generation_configs_deserialize_with_defaults() {
        let config: GenerationConfig =
            serde_json::from_value(serde_json::json!({"max_tokens": 4096})).unwrap();
        assert_eq!(config.max_tokens, Some(4096));
        assert!(config.temperature.is_none());
        assert!(config.reasoning.is_none());
        let empty: GenerationConfig = serde_json::from_str("{}").unwrap();
        assert!(empty.max_tokens.is_none());
    }
}
