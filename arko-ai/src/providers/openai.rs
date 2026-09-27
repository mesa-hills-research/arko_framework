use super::{GenerationResponse, ProviderClient, StreamEvent, take_refused_output};
use crate::{
    config::{GenerationConfig, OpenAIServiceTier},
    pricing::ServiceTier,
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::{ModelType, Provider as ProviderType},
    reasoning::{
        OpenAIEffort, OpenAIReasoningItem, OpenAIReasoningMode, ReasoningArtifact,
    },
    retry::BackoffPolicy,
    stop::{Refusal, RefusalSource, StopReason},
    tool::{ToolCall, ToolChoice},
    usage::{BillingKind, ModelCallUsage, UsageInfo},
};
use async_stream::try_stream;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::sync::RwLock;

#[cfg(test)]
mod oauth_recovery_tests;

#[cfg(test)]
mod refusal_tests;

type StreamingRefusals = BTreeMap<(String, u64), String>;

const CHATGPT_OAUTH_DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CHATGPT_OAUTH_DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// Default Responses API base URLs; requests go to `{base}/responses`.
const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

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

// Tokens never appear in `Debug` output (logs, panics, error reports).
impl std::fmt::Debug for ChatGptOAuthCredentials {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatGptOAuthCredentials")
            .field("access_token", &REDACTED)
            .field("refresh_token", &redacted(self.refresh_token.as_deref()))
            .field("account_id", &self.account_id)
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

/// Backoff for the opt-in OAuth refresh rate-limit loop: 30 s doubling to a
/// 15-minute cap; a server hint is a minimum.
const OAUTH_REFRESH_BACKOFF: BackoffPolicy =
    BackoffPolicy::new(Duration::from_secs(30), Duration::from_secs(900));

fn oauth_retry_delay(
    header_delay: Option<Duration>,
    body: &str,
    attempt: u32,
) -> Duration {
    let hint = header_delay.or_else(|| crate::error::retry_after_from_body(body));
    OAUTH_REFRESH_BACKOFF.next_delay(attempt.saturating_add(1), hint)
}

/// Exchanges a ChatGPT OAuth refresh token for new tokens, once; an HTTP
/// failure (including a 429) is returned to the caller with its retry
/// metadata. `client_id` and `token_url` default to the Codex CLI's.
///
/// Build `http_client` with [`crate::http_client`] or
/// [`crate::http_client_builder`]. `reqwest::Client::new()` panics in
/// applications that don't install a process-wide rustls crypto provider,
/// since this crate builds reqwest without one.
///
/// The refresh token rotates: once this succeeds, the old one stops working,
/// so store the returned tokens (see [`crate::CodexAuthFile`]).
pub async fn refresh_chatgpt_oauth_token(
    http_client: &reqwest::Client,
    refresh_token: &str,
    client_id: Option<&str>,
    token_url: Option<&str>,
) -> Result<OAuthRefreshResult, Error> {
    refresh_chatgpt_oauth_token_with_policy(
        http_client,
        refresh_token,
        client_id,
        token_url,
        false,
    )
    .await
}

async fn refresh_chatgpt_oauth_token_with_policy(
    http_client: &reqwest::Client,
    refresh_token: &str,
    client_id: Option<&str>,
    token_url: Option<&str>,
    retry_rate_limits: bool,
) -> Result<OAuthRefreshResult, Error> {
    let resolved_client_id = client_id.unwrap_or(CHATGPT_OAUTH_DEFAULT_CLIENT_ID);
    let resolved_token_url = token_url.unwrap_or(CHATGPT_OAUTH_DEFAULT_TOKEN_URL);

    let body = serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": resolved_client_id,
        "scope": "openid profile email offline_access",
    });

    let mut rate_limit_attempts = 0_u32;
    let response = loop {
        let response = http_client
            .post(resolved_token_url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        if status.is_success() {
            break response;
        }

        let header_delay = retry_after_from_headers(&response);
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable error body>".into());
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS && retry_rate_limits {
            let delay = oauth_retry_delay(
                header_delay,
                &error_text,
                rate_limit_attempts,
            );
            rate_limit_attempts = rate_limit_attempts.saturating_add(1);
            log::warn!(
                "ChatGPT OAuth refresh was rate limited; waiting {:?} before retry {}",
                delay,
                rate_limit_attempts,
            );
            tokio::time::sleep(delay).await;
            continue;
        }

        return Err(response_error(
            "OAuth token refresh failed",
            status,
            header_delay,
            &error_text,
        ));
    };

    let data: Value = response.json().await?;

    let access_token = data["access_token"]
        .as_str()
        .ok_or_else(|| {
            Error::ApiError("Missing access_token in OAuth refresh response".to_string())
        })?
        .to_string();

    Ok(OAuthRefreshResult {
        access_token,
        refresh_token: data["refresh_token"].as_str().map(String::from),
        id_token: data["id_token"].as_str().map(String::from),
    })
}

#[derive(Clone)]
pub struct OpenAIClient {
    auth: Arc<RwLock<OpenAIAuth>>,
    http_client: reqwest::Client,
    retry_oauth_rate_limits: bool,
    base_url: Option<String>,
    oauth_token_url: Option<String>,
}

impl OpenAIClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            auth: Arc::new(RwLock::new(OpenAIAuth::ApiKey(
                api_key.to_string(),
            ))),
            http_client: super::default_http_client(),
            retry_oauth_rate_limits: false,
            base_url: None,
            oauth_token_url: None,
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
        }
    }

    pub fn with_auth(auth: OpenAIAuth) -> Self {
        Self {
            auth: Arc::new(RwLock::new(auth)),
            http_client: super::default_http_client(),
            retry_oauth_rate_limits: false,
            base_url: None,
            oauth_token_url: None,
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

    /// Configures whether OAuth refresh requests retry HTTP 429 responses.
    pub fn with_oauth_rate_limit_retries(mut self, enabled: bool) -> Self {
        self.retry_oauth_rate_limits = enabled;
        self
    }

    pub async fn update_auth(&self, auth: OpenAIAuth) {
        *self.auth.write().await = auth;
    }

    pub async fn is_chatgpt_oauth(&self) -> bool {
        self.auth.read().await.is_chatgpt_oauth()
    }

    /// Returns a snapshot of the current ChatGPT OAuth credentials.
    ///
    /// The snapshot includes any access or refresh token installed by the
    /// most recent successful refresh. API-key clients return `None`.
    pub async fn chatgpt_oauth_credentials(
        &self,
    ) -> Option<ChatGptOAuthCredentials> {
        match &*self.auth.read().await {
            OpenAIAuth::ChatGptOAuth(credentials) => Some(credentials.clone()),
            OpenAIAuth::ApiKey(_) => None,
        }
    }

    pub async fn is_oauth_token_expired(
        &self,
        margin: std::time::Duration,
    ) -> bool {
        let auth = self.auth.read().await;

        match &*auth {
            OpenAIAuth::ChatGptOAuth(credentials) => {
                is_token_expired(&credentials.access_token, margin)
            }
            OpenAIAuth::ApiKey(_) => false,
        }
    }

    /// Make one refresh request; caller owns retry timing and cancellation.
    pub async fn refresh_oauth_token_once(&self) -> Result<OAuthRefreshResult, Error> {
        self.clone()
            .with_oauth_rate_limit_retries(false)
            .refresh_oauth_token()
            .await
    }

    pub async fn refresh_oauth_token(&self) -> Result<OAuthRefreshResult, Error> {
        let expected = self.snapshot_auth().await;
        self.refresh_oauth_token_for_auth(&expected, self.retry_oauth_rate_limits)
            .await
    }

    async fn refresh_oauth_token_for_auth(
        &self,
        expected: &OpenAIAuth,
        retry_rate_limits: bool,
    ) -> Result<OAuthRefreshResult, Error> {
        // Serialize rotation across clones. Holding this lock also prevents
        // a refresh response from overwriting credentials installed by the host.
        let mut auth = self.auth.write().await;
        let (refresh_token, account_id) = {
            match (&*auth, expected) {
                (OpenAIAuth::ChatGptOAuth(credentials), OpenAIAuth::ChatGptOAuth(previous)) => {
                    if credentials.account_id != previous.account_id {
                        return Err(Error::AuthenticationFailed(
                            "OAuth account changed during the request; start a new request.".into(),
                        ));
                    }
                    if credentials.access_token != previous.access_token {
                        // Another caller already replaced the rejected token.
                        return Ok(OAuthRefreshResult {
                            access_token: credentials.access_token.clone(),
                            refresh_token: credentials.refresh_token.clone(),
                            id_token: None,
                        });
                    }
                    let refresh_token = credentials.refresh_token.as_ref().ok_or_else(|| {
                        Error::AuthenticationFailed(
                            "No refresh_token stored in ChatGPT OAuth credentials. Provide a refresh_token when constructing the client to enable automatic token refresh."
                                .to_string(),
                        )
                    })?;

                    (refresh_token.clone(), credentials.account_id.clone())
                }
                _ => {
                    return Err(Error::InvalidRequest(
                        "Cannot refresh OAuth token: client is using API key auth".to_string(),
                    ));
                }
            }
        };

        let token_url = self.oauth_token_url.as_deref();
        let result = refresh_chatgpt_oauth_token_with_policy(
            &self.http_client,
            &refresh_token,
            None,
            token_url,
            retry_rate_limits,
        )
        .await?;

        let new_refresh_token = result.refresh_token.clone().unwrap_or(refresh_token);

        let new_auth = OpenAIAuth::ChatGptOAuth(ChatGptOAuthCredentials {
            access_token: result.access_token.clone(),
            refresh_token: Some(new_refresh_token),
            account_id,
        });

        *auth = new_auth;

        Ok(result)
    }

    async fn snapshot_auth(&self) -> OpenAIAuth {
        self.auth.read().await.clone()
    }

    async fn send_response_request(
        &self,
        request: &OpenAIRequest,
        mut auth: OpenAIAuth,
    ) -> Result<(reqwest::Response, OpenAIAuth), Error> {
        for attempt in 0..2 {
            let url = self.responses_url(&auth);
            let builder = self
                .http_client
                .post(url)
                .header("Content-Type", "application/json");
            let response = Self::apply_auth_headers(&auth, builder)
                .json(request)
                .send()
                .await?;
            if response.status().is_success() {
                return Ok((response, auth));
            }
            let status = response.status();
            let retry_after = retry_after_from_headers(&response);
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unavailable error body>".into());
            let error = response_error("OpenAI request failed", status, retry_after, &body);
            let expired = matches!(&error, Error::ProviderError {
                status: Some(401), code: Some(code), ..
            } if code == "token_expired");
            if attempt == 0 && expired && auth.is_chatgpt_oauth() {
                // A server can reject a token before its embedded expiry.
                // Refresh once; transient failures return to the agent's bounded
                // recovery policy. No stream or tool calls have been delivered.
                self.refresh_oauth_token_for_auth(&auth, false).await?;
                let updated = self.snapshot_auth().await;
                if !matches!((&auth, &updated), (OpenAIAuth::ChatGptOAuth(old), OpenAIAuth::ChatGptOAuth(new)) if old.account_id == new.account_id)
                {
                    return Err(error);
                }
                auth = updated;
            } else {
                return Err(error);
            }
        }
        unreachable!("each request either returns or refreshes once")
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
            OpenAIAuth::ApiKey(key) => {
                builder.header("Authorization", format!("Bearer {}", key))
            }
            OpenAIAuth::ChatGptOAuth(credentials) => builder
                .header(
                    "Authorization",
                    format!("Bearer {}", credentials.access_token),
                )
                .header("chatgpt-account-id", &credentials.account_id)
                .header("OpenAI-Beta", "responses=experimental"),
        }
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

        if conversation.model.model_type == ModelType::Reasoning
            && config.temperature_specified
        {
            return Err(Error::InvalidRequest(
                "OpenAI reasoning models do not support temperature configuration."
                    .to_string(),
            ));
        }

        let auth = self.snapshot_auth().await;
        let request =
            self.build_request(conversation, config, auth.is_chatgpt_oauth())?;

        let (response, auth) = self.send_response_request(&request, auth).await?;

        let response_data: OpenAIResponse = response.json().await?;
        Ok(Self::generation_response(
            &response_data,
            &conversation.model,
            config,
            auth.is_chatgpt_oauth(),
            &StreamingRefusals::new(),
            Vec::new(),
        ))
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

        if conversation.model.model_type == ModelType::Reasoning
            && config.temperature_specified
        {
            return Err(Error::InvalidRequest(
                "OpenAI reasoning models do not support temperature configuration."
                    .to_string(),
            ));
        }

        let auth = self.snapshot_auth().await;
        let mut request =
            self.build_request(conversation, config, auth.is_chatgpt_oauth())?;
        request.stream = Some(true);

        let (response, auth) = self.send_response_request(&request, auth).await?;

        Ok(Self::stream_events(
            response.bytes_stream(),
            conversation.model.clone(),
            config.clone(),
            auth.is_chatgpt_oauth(),
        ))
    }
}

impl OpenAIClient {
    /// Parses a Responses API SSE byte stream. `End` is emitted only for a
    /// `response.completed` or `response.incomplete` event; a stream that
    /// ends earlier yields [`Error::StreamTruncated`].
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
        let stream = try_stream! {
            let mut event_stream = Box::pin(bytes.eventsource());
            let mut streaming_tool_calls: Vec<StreamingToolCall> = Vec::new();
            let mut streaming_refusals = StreamingRefusals::new();
            let mut finished = false;

            while let Some(event) = event_stream.next().await {
                let event =
                    event.map_err(|error| Error::ApiError(format!("SSE error: {error}")))?;

                let value: Value =
                    serde_json::from_str(&event.data).map_err(Error::JsonError)?;

                match value.get("type").and_then(Value::as_str) {
                    Some("response.output_text.delta") => {
                        let delta = value["delta"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();

                        yield StreamEvent::TextChunk(delta);
                    }
                    // Refusal text is reported through `StreamEvent::Refusal`,
                    // never as ordinary text.
                    Some("response.refusal.delta" | "response.refusal.done") => {
                        Self::record_stream_refusal(&mut streaming_refusals, &value);
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
                    }
                    Some("response.completed" | "response.incomplete") => {
                        #[derive(Deserialize)]
                        struct Completed {
                            response: OpenAIResponse,
                        }

                        let completed: Completed =
                            serde_json::from_value(value).map_err(Error::JsonError)?;

                        let streamed_calls = streaming_tool_calls
                            .drain(..)
                            .filter(|tool_call| tool_call.completed)
                            .map(StreamingToolCall::into_tool_call)
                            .collect();

                        let response = Self::generation_response(
                            &completed.response,
                            &model,
                            &config,
                            is_chatgpt_oauth,
                            &streaming_refusals,
                            streamed_calls,
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
        streamed_refusals: &StreamingRefusals,
        streamed_calls: Vec<ToolCall>,
    ) -> GenerationResponse {
        let (mut message, mut refusal_texts) = Self::parse_response(response);
        let mut usage = Self::calculate_usage(response);
        // ChatGPT OAuth calls are covered by the subscription: the estimate
        // stays visible while nothing is billed per call.
        if is_chatgpt_oauth {
            usage.billing = BillingKind::Subscription;
        }
        usage.calculate_cost(model, Some(config));

        let model_call_usage = ModelCallUsage::new(
            response.id.clone(),
            model.provider.as_telemetry_str().to_string(),
            response.model.clone(),
            response.usage.input_tokens,
            response.usage.output_tokens,
            model.max_input_tokens,
        );
        let model_call_usage =
            Self::with_response_stop_metadata(model_call_usage, response, streamed_refusals);

        if refusal_texts.is_empty() {
            refusal_texts = streamed_refusals.values().cloned().collect();
        }

        // The ChatGPT Codex backend can finish a stream without repeating its
        // output items, so use the calls whose arguments finished streaming.
        if message.tool_calls.is_empty() && refusal_texts.is_empty() {
            message.tool_calls = streamed_calls;
        }

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
        }
    }

    /// Normalizes a finished response. Refusal parts are the model declining
    /// in its own words; `incomplete` responses carry their reason in
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
            ("incomplete", reason) => {
                (StopReason::Other(reason.unwrap_or("incomplete").to_string()), None)
            }
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
                refusals.entry(key).or_default().push_str(event["delta"].as_str().unwrap_or(""));
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
        streamed: &StreamingRefusals,
    ) -> ModelCallUsage {
        let mut refusals = Vec::new();
        for output in &response.output {
            if let OpenAIOutput::Message { id, content, .. } = output {
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
            refusals.extend(streamed.iter().map(|((id, index), text)| serde_json::json!({
                "item_id": id, "content_index": index, "refusal": text,
            })));
        }
        if !refusals.is_empty() {
            return usage.with_stop_metadata(Some("refusal".into()), Some(serde_json::json!({
                "source": "openai_structured_refusal", "response_status": response.status,
                "refusals": refusals,
            })));
        }
        let reason = response.incomplete_details.as_ref()
            .and_then(|details| details.get("reason")).and_then(Value::as_str)
            .unwrap_or(&response.status).to_string();
        let details = response.incomplete_details.as_ref().map(|details| serde_json::json!({
            "response_status": response.status, "incomplete_details": details,
        }));
        usage.with_stop_metadata(Some(reason), details)
    }

    fn request_model_id(
        model_id: &str,
        is_chatgpt_oauth: bool,
    ) -> String {
        if !is_chatgpt_oauth
            || !(model_id.starts_with("gpt-5.4")
                || model_id.starts_with("gpt-5.5"))
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
        let resolved_max_tokens = if is_chatgpt_oauth {
            None
        } else {
            config
                .max_tokens
                .or(Some(conversation.model.max_output_tokens))
        };

        if let Some(tier) = config.openai_service_tier {
            if is_chatgpt_oauth {
                return Err(Error::UnsupportedFeature(
                    "OpenAI service tiers require API-key authentication; ChatGPT OAuth requests use the subscription's processing."
                        .to_string(),
                ));
            }
            let priced = tier.pricing_tier();
            if priced != ServiceTier::Standard
                && !conversation.model.pricing.offers_tier(&priced)
            {
                return Err(Error::InvalidRequest(format!(
                    "{} does not offer the {:?} service tier.",
                    conversation.model.id, tier
                )));
            }
        }

        let mut request = OpenAIRequest {
            model: Self::request_model_id(
                &conversation.model.id,
                is_chatgpt_oauth,
            ),
            store: false,
            instructions: conversation.system_message.clone(),
            reasoning: None,
            text: None,
            include: None,
            input: Vec::new(),
            max_output_tokens: resolved_max_tokens,
            // Omitted unless set, so the model default applies. Reasoning
            // models reject it (checked before building).
            temperature: (conversation.model.model_type != ModelType::Reasoning
                && config.temperature_specified)
                .then_some(config.temperature),
            stream: Some(false),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            prompt_cache_key: None,
            prompt_cache_options: None,
            service_tier: config.openai_service_tier,
        };

        if conversation.model.supports_openai_prompt_cache_options() {
            request.prompt_cache_key =
                Some(conversation.prompt_cache_key.clone());

            if !is_chatgpt_oauth {
                let enabled = config
                    .prompt_caching
                    .unwrap_or(conversation.enable_cache);

                request.prompt_cache_options =
                    Some(OpenAIPromptCacheOptions {
                        mode: if enabled {
                            OpenAIPromptCacheMode::Implicit
                        } else {
                            OpenAIPromptCacheMode::Explicit
                        },
                    });
            }
        }

        if config.reasoning.is_some()
            && conversation.model.model_type != ModelType::Reasoning
        {
            return Err(Error::InvalidRequest(
                "Reasoning configuration is only supported for reasoning models."
                    .to_string(),
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
        let is_gpt5_4_pro =
            conversation.model.id.starts_with("gpt-5.4-pro");
        let is_gpt5_4 = conversation.model.id.starts_with("gpt-5.4");
        let is_gpt5_5 = conversation.model.id.starts_with("gpt-5.5");
        let is_gpt5_6 = conversation.model.is_openai_gpt_5_6();
        let is_gpt6_astra = conversation.model.is_openai_gpt_6_astra();
        let is_gpt6_sol_luna = conversation.model.is_openai_gpt_6_sol_or_luna();
        let is_gpt6 = is_gpt6_astra || is_gpt6_sol_luna;

        let is_legacy_reasoning =
            conversation.model.id.starts_with("o3-")
                || conversation.model.id.starts_with("o4-");

        let is_gpt5_family = is_gpt5
            || is_gpt5_1
            || is_gpt5_2
            || is_gpt5_3
            || is_gpt5_4
            || is_gpt5_5
            || is_gpt5_6;

        if config.openai_reasoning_mode.is_some()
            && !(is_gpt5_6 || is_gpt6)
        {
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
            request.include =
                Some(vec!["reasoning.encrypted_content".to_string()]);
        }

        if let Some(crate::reasoning::ReasoningConfig::OpenAI {
            effort,
            include_summaries,
        }) = &config.reasoning
        {
            let (allowed, supported_label): (&[OpenAIEffort], &str) =
                if is_gpt6_astra {
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
                        &[
                            OpenAIEffort::Low,
                            OpenAIEffort::Medium,
                            OpenAIEffort::High,
                        ],
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
                mode: config.openai_reasoning_mode,
                effort: Some(*effort),
                summary: if *include_summaries {
                    Some("auto".to_string())
                } else {
                    None
                },
            });
        } else if is_chatgpt_oauth && (is_gpt5_family || is_gpt6) {
            request.reasoning = Some(OpenAIReasoning {
                mode: config.openai_reasoning_mode,
                effort: Some(OpenAIEffort::Medium),
                summary: None,
            });
        } else if config.openai_reasoning_mode.is_some() {
            // API key: without a configured effort the model's own default
            // applies (none on GPT-5.1/5.2/5.4, high on GPT-5.5 Pro, medium on
            // GPT-5.5/5.6/6). https://developers.openai.com/api/docs/guides/reasoning
            request.reasoning = Some(OpenAIReasoning {
                mode: config.openai_reasoning_mode,
                effort: None,
                summary: None,
            });
        }

        if let Some(tool_config) = &config.tool_config {
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
                    ToolChoice::Auto => {
                        OpenAIToolChoice::String("auto".to_string())
                    }
                    ToolChoice::Required => {
                        OpenAIToolChoice::String("required".to_string())
                    }
                    ToolChoice::None => {
                        OpenAIToolChoice::String("none".to_string())
                    }
                    ToolChoice::Specific(name) => {
                        OpenAIToolChoice::Object(OpenAIToolChoiceObject {
                            type_: "function".to_string(),
                            name: name.clone(),
                        })
                    }
                });
            }

            request.parallel_tool_calls =
                tool_config.parallel_tool_calls;
        }

        for message in &conversation.messages {
            if let Role::Assistant = message.role {
                if let Some(ReasoningArtifact::OpenAIEncryptedReasoning { items }) =
                    &message.reasoning_artifact
                {
                    for item in items {
                        request.input.push(OpenAIInputEntry::Reasoning(
                            OpenAIInputReasoning {
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
                            },
                        ));
                    }
                }

                for tool_call in &message.tool_calls {
                    // Invalid arguments are replayed exactly as the model wrote them.
                    let arguments = match &tool_call.invalid_arguments {
                        Some(raw) => raw.clone(),
                        None => serde_json::to_string(&tool_call.arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    };

                    request.input.push(OpenAIInputEntry::FunctionCall(
                        OpenAIInputFunctionCall {
                            type_: "function_call".to_string(),
                            id: tool_call.provider_id.clone(),
                            call_id: tool_call.id.clone().unwrap_or_default(),
                            name: tool_call.name.clone(),
                            arguments,
                        },
                    ));
                }
            }

            if let Role::User = message.role {
                let has_tool_results = message
                    .content
                    .iter()
                    .any(|content| content.as_tool_result().is_some());

                if has_tool_results {
                    for content in &message.content {
                        if let Content::ToolResult(result) = content {
                            request.input.push(
                                OpenAIInputEntry::FunctionCallOutput(
                                    OpenAIInputFunctionCallOutput {
                                        type_: "function_call_output"
                                            .to_string(),
                                        call_id: result
                                            .call_id
                                            .clone()
                                            .unwrap_or_default(),
                                        output: result.output.clone(),
                                    },
                                ),
                            );
                        }
                    }

                    let filtered: Vec<&Content> = message
                        .content
                        .iter()
                        .filter(|content| {
                            content.as_tool_result().is_none()
                        })
                        .collect();

                    if !filtered.is_empty() {
                        let message = self
                            .convert_filtered_to_openai_message(
                                "user",
                                &filtered,
                                message.id.as_deref(),
                            )?;

                        request
                            .input
                            .push(OpenAIInputEntry::Message(message));
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
                    input_message.content =
                        Some(OpenAIContent::Text(text.clone()));
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
                            image_url: Some(
                                self.convert_image_to_openai(image)?,
                            ),
                        });
                    }
                    Content::ToolResult(_) => {}
                }
            }

            input_message.content =
                Some(OpenAIContent::Array(content_items));
        }

        Ok(input_message)
    }

    fn convert_message_to_openai(
        &self,
        message: &Message,
    ) -> Result<OpenAIInputMessage, Error> {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => {
                return Err(Error::InvalidRequest(
                    "System messages should be set in the instructions field."
                        .to_string(),
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
                    input_message.content =
                        Some(OpenAIContent::Array(vec![OpenAIContentItem {
                            type_: "output_text".to_string(),
                            text: Some(text.to_string()),
                            image_url: None,
                        }]));
                } else {
                    input_message.content =
                        Some(OpenAIContent::Text(text.to_string()));
                }
            } else if let Some(image) = content[0].as_image() {
                input_message.content =
                    Some(OpenAIContent::Array(vec![self.image_item(image)?]));
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
                            image_url: Some(
                                self.convert_image_to_openai(image)?,
                            ),
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

    fn convert_image_to_openai(
        &self,
        image: &Image,
    ) -> Result<String, Error> {
        match image {
            Image::Url(url) => Ok(url.clone()),
            Image::Base64 { data, mime_type } => {
                Ok(format!("data:{};base64,{}", mime_type, data))
            }
        }
    }

    /// Returns the assistant message and any refusal texts. Refusal text is
    /// never part of the message text. Function calls that didn't finish
    /// (an `incomplete` response cut them off) are dropped.
    fn parse_response(response: &OpenAIResponse) -> (Message, Vec<String>) {
        let mut text = String::new();
        let mut refusals = Vec::new();
        let mut reasoning_items = Vec::new();
        let mut message_id = None;
        let mut tool_calls = Vec::new();

        for output in &response.output {
            match output {
                OpenAIOutput::Message { content, id, .. } => {
                    message_id = Some(id.clone());

                    for item in content {
                        match item.type_.as_str() {
                            "output_text" => text.push_str(&item.text),
                            "refusal" => refusals.push(item.refusal.clone().unwrap_or_default()),
                            _ => {}
                        }
                    }
                }
                OpenAIOutput::Reasoning {
                    encrypted_content: Some(encrypted_content),
                    summary,
                    id,
                } => reasoning_items.push(OpenAIReasoningItem {
                    encrypted_content: encrypted_content.clone(),
                    summaries: summary.as_ref().map(|summaries| {
                        summaries
                            .iter()
                            .map(|summary| summary.text.clone())
                            .collect()
                    }),
                    id: Some(id.clone()),
                }),
                // Without encrypted content a reasoning item can't be replayed
                // statelessly (store=false).
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

                    let mut call =
                        ToolCall::from_argument_text(Some(call_id.clone()), name.clone(), arguments);
                    call.provider_id = Some(id.clone());
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
            reasoning_artifact: (!reasoning_items.is_empty()).then_some(
                ReasoningArtifact::OpenAIEncryptedReasoning {
                    items: reasoning_items,
                },
            ),
            tool_calls,
            id: message_id,
            configuration_update: None,
        };

        (message, refusals)
    }

    /// `input_tokens` already includes cached and written tokens and
    /// `output_tokens` includes reasoning tokens. The response's
    /// `service_tier` is the tier actually used (for example `default` when a
    /// Fast request was downgraded).
    /// https://developers.openai.com/api/docs/guides/prompt-caching
    fn calculate_usage(response: &OpenAIResponse) -> UsageInfo {
        let mut usage = UsageInfo::new(
            response.usage.input_tokens,
            response.usage.output_tokens,
        );

        if let Some(details) = &response.usage.input_tokens_details {
            if details.cached_tokens > 0 {
                usage =
                    usage.with_cached_tokens(details.cached_tokens);
            }

            if details.cache_write_tokens > 0 {
                usage = usage
                    .with_cache_write_tokens(details.cache_write_tokens);
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
    output: String,
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
    model: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<OpenAIServiceTier>,
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
    model: String,
    output: Vec<OpenAIOutput>,
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
        reasoning::{
            OpenAIEffort, OpenAIReasoningMode, ReasoningConfig,
        },
    };
    use std::time::Duration;

    #[test]
    fn tool_images_serialize_after_all_correlated_results_for_api_and_oauth() {
        use crate::{Content, Image, Message, ToolCall, ToolResult};
        let client = OpenAIClient::new("test");
        let mut conversation = Conversation::new(Model::get(Provider::OpenAI, "gpt-6-astra").unwrap());
        let calls = ["first", "second"].map(|id| ToolCall { id: Some(id.into()), name: "view_image".into(), arguments: serde_json::json!({"path":"fixture.png"}), thought_signature: None, provider_id: None, invalid_arguments: None });
        let mut assistant = Message::new_assistant("Inspecting images."); assistant.tool_calls = calls.to_vec(); conversation.messages.push(assistant);
        let mut content: Vec<_> = calls.iter().map(|call| Content::ToolResult(ToolResult::from_tool_call(call, "fixture.png, 12 × 8"))).collect();
        content.extend([Content::Text("Image from first call: fixture.png".into()), Content::Image(Image::new_base64("cGl4ZWxz", "image/png"))]);
        conversation.add_user_message_with_contents(content);
        let saved = serde_json::to_string(&conversation).unwrap();
        let conversation: Conversation = serde_json::from_str(&saved).unwrap();
        for oauth in [false, true] {
            let value = serde_json::to_value(client.build_request(&conversation, &GenerationConfig::default(), oauth).unwrap()).unwrap();
            let input = value["input"].as_array().unwrap();
            let results: Vec<_> = input.iter().enumerate().filter(|(_, item)| item["type"] == "function_call_output").collect();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].1["call_id"], "first"); assert_eq!(results[1].1["call_id"], "second");
            let (image_index, image_message) = input.iter().enumerate().find(|(_, item)| item["content"].as_array().is_some_and(|content| content.iter().any(|block| block["type"] == "input_image"))).unwrap();
            assert!(image_index > results[1].0);
            assert_eq!(image_message["content"][1]["image_url"], "data:image/png;base64,cGl4ZWxz");
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
        let client = crate::providers::http_client_builder().no_proxy().build().unwrap();
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
                assert_eq!(value["reasoning"]["mode"], "pro");
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
    fn gpt_6_sol_and_luna_build_reasoning_verbosity_and_cache_requests() {
        let client = OpenAIClient::new("test");

        for id in ["gpt-6-sol", "gpt-6-luna"] {
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
                    assert_eq!(value["reasoning"]["mode"], "pro");
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

            // API key: no effort configured means the model default applies.
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
        let client = OpenAIClient::new_chatgpt_oauth(
            "access-token",
            Some("refresh-token"),
            "account-id",
        );
        assert!(!client.retry_oauth_rate_limits);

        let client = client.with_oauth_rate_limit_retries(true);
        assert!(client.retry_oauth_rate_limits);
    }

    #[test]
    fn oauth_request_model_id_strips_gpt_5_4_snapshot_suffixes() {
        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.4-2026-03-05",
                true
            ),
            "gpt-5.4"
        );

        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.4-mini-2026-03-17",
                true
            ),
            "gpt-5.4-mini"
        );

        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.4-nano-2026-03-17",
                true
            ),
            "gpt-5.4-nano"
        );

        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.4-pro-2026-03-05",
                true
            ),
            "gpt-5.4-pro"
        );
    }

    #[test]
    fn oauth_request_model_id_strips_gpt_5_5_snapshot_suffix() {
        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.5-2026-04-23",
                true
            ),
            "gpt-5.5"
        );
    }

    #[test]
    fn request_model_id_preserves_api_key_and_shorthand_ids() {
        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.5-2026-04-23",
                false
            ),
            "gpt-5.5-2026-04-23"
        );

        assert_eq!(
            OpenAIClient::request_model_id("gpt-5.5", true),
            "gpt-5.5"
        );

        assert_eq!(
            OpenAIClient::request_model_id(
                "gpt-5.4-mini",
                true
            ),
            "gpt-5.4-mini"
        );
    }

    #[test]
    fn gpt_5_6_defaults_to_implicit_prompt_caching() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default();

        let request = client
            .build_request(&conversation, &config, false)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(
            value["prompt_cache_options"]["mode"],
            "implicit"
        );
        assert_eq!(
            value["prompt_cache_key"],
            conversation.prompt_cache_key
        );
    }

    #[test]
    fn gpt_5_6_oauth_sends_cache_key_without_cache_options() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default();

        let request = client
            .build_request(&conversation, &config, true)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(
            value["prompt_cache_key"],
            conversation.prompt_cache_key
        );
        assert!(value.get("prompt_cache_options").is_none());
    }

    #[test]
    fn gpt_5_6_family_sends_cache_key_when_enabled() {
        let client = OpenAIClient::new("test");
        let models = [
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_5_6_terra(),
            Model::openai_gpt_5_6_luna(),
        ];
        let config =
            GenerationConfig::default().with_prompt_caching(true);

        for model in models {
            let conversation = Conversation::new(model);
            let request = client
                .build_request(&conversation, &config, false)
                .unwrap();
            let value = serde_json::to_value(request).unwrap();

            assert_eq!(
                value["prompt_cache_key"],
                conversation.prompt_cache_key
            );
            assert_eq!(
                value["prompt_cache_options"]["mode"],
                "implicit"
            );
        }
    }

    #[test]
    fn gpt_5_6_can_disable_prompt_caching() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_6_sol());
        let config =
            GenerationConfig::default().with_prompt_caching(false);

        let request = client
            .build_request(&conversation, &config, false)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(
            value["prompt_cache_options"]["mode"],
            "explicit"
        );
    }

    #[test]
    fn older_openai_models_omit_prompt_cache_options() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_5());
        let config =
            GenerationConfig::default().with_prompt_caching(false);

        let request = client
            .build_request(&conversation, &config, false)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert!(value.get("prompt_cache_key").is_none());
        assert!(value.get("prompt_cache_options").is_none());
    }

    #[test]
    fn gpt_5_6_pro_mode_serializes_for_all_model_tiers() {
        let client = OpenAIClient::new("test");
        let models = [
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_5_6_terra(),
            Model::openai_gpt_5_6_luna(),
        ];
        let config = GenerationConfig::default()
            .with_openai_reasoning_mode(OpenAIReasoningMode::Pro);

        for model in models {
            let conversation = Conversation::new(model);

            let request = client
                .build_request(&conversation, &config, false)
                .unwrap();

            let value = serde_json::to_value(request).unwrap();

            assert_eq!(value["reasoning"]["mode"], "pro");
            assert!(value["reasoning"].get("effort").is_none());
        }
    }

    #[test]
    fn gpt_5_6_standard_mode_can_be_explicit() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default()
            .with_openai_reasoning_mode(OpenAIReasoningMode::Standard);

        let request = client
            .build_request(&conversation, &config, false)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["reasoning"]["mode"], "standard");
    }

    #[test]
    fn gpt_5_6_reasoning_mode_is_independent_of_effort() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default()
            .with_openai_reasoning_mode(OpenAIReasoningMode::Pro)
            .with_reasoning(ReasoningConfig::openai(
                OpenAIEffort::Max,
                true,
            ));

        let request = client
            .build_request(&conversation, &config, false)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["reasoning"]["mode"], "pro");
        assert_eq!(value["reasoning"]["effort"], "max");
        assert_eq!(value["reasoning"]["summary"], "auto");
    }

    #[test]
    fn gpt_5_6_omits_reasoning_mode_by_default() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_6_sol());
        let config = GenerationConfig::default();

        let request = client
            .build_request(&conversation, &config, false)
            .unwrap();

        let value = serde_json::to_value(request).unwrap();

        assert!(value.get("reasoning").is_none());
    }

    #[test]
    fn older_openai_models_reject_reasoning_mode() {
        let client = OpenAIClient::new("test");
        let conversation =
            Conversation::new(Model::openai_gpt_5_5());
        let config = GenerationConfig::default()
            .with_openai_reasoning_mode(OpenAIReasoningMode::Pro);

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
        assert!(serde_json::to_value(request).unwrap().get("temperature").is_none());

        let request = client
            .build_request(&conversation, &GenerationConfig::new(0.3), false)
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
        for (tier, wire) in [
            (OpenAIServiceTier::Flex, "flex"),
            (OpenAIServiceTier::Fast, "priority"),
            (OpenAIServiceTier::Auto, "auto"),
            (OpenAIServiceTier::Default, "default"),
        ] {
            let config = GenerationConfig::default().with_openai_service_tier(tier);
            let request = client.build_request(&astra, &config, false).unwrap();
            assert_eq!(serde_json::to_value(request).unwrap()["service_tier"], wire);
            assert!(matches!(
                client.build_request(&astra, &config, true),
                Err(Error::UnsupportedFeature(_))
            ));
        }
        let unset = client
            .build_request(&astra, &GenerationConfig::default(), false)
            .unwrap();
        assert!(serde_json::to_value(unset).unwrap().get("service_tier").is_none());
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
            &super::StreamingRefusals::new(),
            Vec::new(),
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
        assert!(close(priced(Some("priority"), &standard, false).total_cost, 102.0));
        // The requested tier applies only when the response doesn't report one;
        // a Flex request served at the default tier is priced as standard.
        assert!(close(priced(None, &flex, false).total_cost, 25.5));
        let downgraded = priced(Some("default"), &flex, false);
        assert_eq!(downgraded.service_tier, Some(crate::pricing::ServiceTier::Standard));
        assert!(close(downgraded.total_cost, 51.0));
    }

    #[test]
    fn oauth_refresh_backoff_doubles_to_a_cap_and_honors_server_hints() {
        for _ in 0..50 {
            let first = super::oauth_retry_delay(None, "{}", 0);
            assert!(first >= Duration::from_secs(15) && first <= Duration::from_secs(30));
            let late = super::oauth_retry_delay(None, "{}", 20);
            assert!(late >= Duration::from_secs(450) && late <= Duration::from_secs(900));
            let hinted = super::oauth_retry_delay(None, r#"{"retry_after":1200}"#, 0);
            assert!(hinted >= Duration::from_secs(1200));
        }
    }

    #[test]
    fn chatgpt_oauth_calls_keep_estimates_but_bill_nothing() {
        let usage = priced(None, &GenerationConfig::default(), true);
        assert_eq!(usage.billing, crate::usage::BillingKind::Subscription);
        assert!(close(usage.total_cost, 51.0));
        assert!(close(usage.input_cost, 1.0));
        assert_eq!(usage.billed(), 0.0);
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
        let request = OpenAIClient::new("test").build_request(&Conversation::new(model), config, false)?;
        Ok(serde_json::to_value(request).unwrap())
    }

    fn accepts(model: &Model, effort: OpenAIEffort) -> bool {
        let config = GenerationConfig::default().with_reasoning(ReasoningConfig::openai(effort, false));
        request(model.clone(), &config).is_ok()
    }

    #[test]
    fn pro_models_have_their_own_effort_rules() {
        use OpenAIEffort::{High, Low, Max, Medium, Minimal, XHigh};
        for model in [Model::openai_gpt_5_5_pro(), Model::openai_gpt_5_2_pro(), Model::openai_gpt_5_4_pro()] {
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
            Model::openai_gpt_6_astra(),
        ] {
            let value = request(model.clone(), &GenerationConfig::default()).unwrap();
            assert!(value.get("reasoning").is_none(), "{}", model.id);
            assert_eq!(value["include"][0], "reasoning.encrypted_content");
        }
    }

    #[test]
    fn gpt_5_6_alias_shares_sol_data_and_rules() {
        let alias = Model::openai_gpt_5_6();
        let sol = Model::openai_gpt_5_6_sol();
        assert_eq!(alias.pricing.input_cost_per_million, sol.pricing.input_cost_per_million);
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
        assert!(client.build_request(&conversation, &GenerationConfig::default(), false).is_ok());

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
        assert!(!config.temperature_specified);
        assert!(config.reasoning.is_none());
        let empty: GenerationConfig = serde_json::from_str("{}").unwrap();
        assert!(empty.max_tokens.is_none());
    }
}
