//! OpenRouter provider on the native Chat Completions API
//! (`POST https://openrouter.ai/api/v1/chat/completions`) with OpenRouter's
//! routing, caching and reasoning extensions.
//!
//! Library-neutral by default: no attribution or metadata headers are sent
//! (see [`AppAttribution`] to opt in), no model fallbacks, no network call
//! beyond the requests you make. Models are dynamic slugs
//! ([`Model::openrouter`]); the catalog ([`OpenRouterClient::fetch_models`])
//! adds limits, prices and request validation.
//! https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion

mod catalog;
mod options;
mod request;
mod response;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod mock_server_tests;

#[cfg(test)]
mod live_tests;

pub use catalog::{CatalogSource, OpenRouterCatalog, OpenRouterModelInfo, OpenRouterReasoningInfo};
pub use options::{
    AppAttribution, AutoExactoPolicy, AutoRouterCostTier, AutoRouterOptions, DataCollection,
    MaxPrice, OpenRouterOptions, OpenRouterServiceTier, PerformanceThreshold, ProviderPreferences,
    ProviderSort, Quantization, SortBy, SortPartition,
};
pub(crate) use request::is_claude_family;

use super::{GenerationResponse, ProviderClient, StreamEvent};
use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::{Error, retry_after_from_headers},
    model::Model,
    stop::{Refusal, RefusalSource},
};
use futures::{StreamExt, stream::BoxStream};
use request::PreparedRequest;
use response::ResponseContext;
use serde_json::Value;
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

/// Default API base URL; chat requests go to `{base}/chat/completions`.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// A client for OpenRouter. Clones share the loaded catalog.
#[derive(Clone)]
pub struct OpenRouterClient {
    api_key: String,
    http_client: reqwest::Client,
    base_url: String,
    default_options: OpenRouterOptions,
    attribution: Option<AppAttribution>,
    catalog: Arc<RwLock<Option<Arc<OpenRouterCatalog>>>>,
    stream_idle_timeout: Option<Duration>,
}

impl std::fmt::Debug for OpenRouterClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterClient")
            .field("base_url", &self.base_url)
            .field("default_options", &self.default_options)
            .field("attribution", &self.attribution)
            .finish_non_exhaustive()
    }
}

impl OpenRouterClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http_client: super::default_http_client(),
            base_url: OPENROUTER_BASE_URL.to_string(),
            default_options: OpenRouterOptions::default(),
            attribution: None,
            catalog: Arc::new(RwLock::new(None)),
            stream_idle_timeout: None,
        }
    }

    /// Cuts a streamed response off, with a transient
    /// [`Error::StreamTruncated`], once no data has arrived for `timeout`.
    /// `None` (the default) waits as long as the connection stays open. See
    /// [`LLMClient::with_stream_idle_timeout`](crate::client::LLMClient::with_stream_idle_timeout)
    /// before setting one: models can think for a long time without sending
    /// anything, so a value must be generous.
    pub fn with_stream_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }

    /// Sends requests to `base_url` (for example a regional hostname, a proxy
    /// or a test server) instead of `https://openrouter.ai/api/v1`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Options applied to every request; per-request
    /// [`GenerationConfig::openrouter`] options override them field by field.
    pub fn with_default_options(mut self, options: OpenRouterOptions) -> Self {
        self.default_options = options;
        self
    }

    /// Sends app attribution headers (`HTTP-Referer`, `X-OpenRouter-Title`,
    /// ...) with chat requests. Off unless set: a library shouldn't credit
    /// itself, so binaries opt in with their own app.
    pub fn with_app_attribution(mut self, attribution: AppAttribution) -> Self {
        self.attribution = Some(attribution);
        self
    }

    pub fn default_options(&self) -> &OpenRouterOptions {
        &self.default_options
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url.trim_end_matches('/'))
    }

    /// Fetches the public catalog (`GET /models`, no API key sent) and keeps
    /// it in memory for validation and [`model`](Self::model). The catalog
    /// is only ever fetched by this call or [`fetch_user_models`](Self::fetch_user_models).
    pub async fn fetch_models(&self) -> Result<Arc<OpenRouterCatalog>, Error> {
        let response = self.http_client.get(self.url("models")).send().await?;
        self.store_catalog(response, CatalogSource::Public).await
    }

    /// Fetches the models this API key may use (`GET /models/user`), which
    /// reflects the account's provider preferences, privacy settings (such as
    /// ZDR) and guardrails, and keeps it in memory.
    pub async fn fetch_user_models(&self) -> Result<Arc<OpenRouterCatalog>, Error> {
        let response = self
            .http_client
            .get(self.url("models/user"))
            .bearer_auth(&self.api_key)
            .send()
            .await?;
        self.store_catalog(response, CatalogSource::User).await
    }

    async fn store_catalog(
        &self,
        response: reqwest::Response,
        source: CatalogSource,
    ) -> Result<Arc<OpenRouterCatalog>, Error> {
        let response = Self::check_status(response).await?;
        let catalog = Arc::new(OpenRouterCatalog::parse(&response.text().await?, source)?);
        self.set_catalog(Some(catalog.clone()));
        Ok(catalog)
    }

    /// Installs (or with `None` clears) the catalog used for validation, for
    /// example one parsed with [`OpenRouterCatalog::from_json`].
    pub fn set_catalog(&self, catalog: Option<Arc<OpenRouterCatalog>>) {
        *self
            .catalog
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = catalog;
    }

    /// The loaded catalog, if any.
    pub fn catalog(&self) -> Option<Arc<OpenRouterCatalog>> {
        self.catalog
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The model with this slug from the loaded catalog.
    pub fn model(&self, slug: &str) -> Option<Model> {
        self.catalog()?.get(slug)
    }

    /// The catalog model for `slug` when a catalog is loaded (failing for
    /// slugs it doesn't list; presets such as `@preset/name` are never
    /// listed and pass, and `model@preset/name` resolves to its model), else
    /// [`Model::openrouter`].
    pub fn resolve_model(&self, slug: &str) -> Result<Model, Error> {
        match self.catalog().filter(|_| !slug.starts_with('@')) {
            Some(catalog) => catalog.get(slug).ok_or_else(|| {
                Error::ModelNotSupported(format!(
                    "OpenRouter model '{slug}' is not in the loaded catalog ({:?}).",
                    catalog.source()
                ))
            }),
            None => Ok(Model::openrouter(slug)),
        }
    }

    async fn check_status(response: reqwest::Response) -> Result<reqwest::Response, Error> {
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status().as_u16();
        let retry_after = retry_after_from_headers(status, response.headers());
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable error body>".into());
        Err(openrouter_error(Some(status), retry_after, &body))
    }

    async fn send_chat(&self, prepared: &PreparedRequest) -> Result<reqwest::Response, Error> {
        let mut builder = self
            .http_client
            .post(self.url("chat/completions"))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json");
        if let Some(attribution) = &self.attribution {
            for (name, value) in attribution.headers() {
                builder = builder.header(name, value);
            }
        }
        if !prepared.anthropic_betas.is_empty() {
            builder = builder.header("x-anthropic-beta", prepared.anthropic_betas.join(","));
        }
        let response = builder.json(&prepared.body).send().await?;
        Self::check_status(response).await
    }
}

/// Converts an OpenRouter error body (from an HTTP error, a 200 response or
/// a stream chunk) into an [`Error`].
///
/// - `error.metadata.error_type` `refusal` (the provider refused as an error,
///   for example a GPT-6 `bio_policy` block) becomes [`Error::Refusal`] with
///   [`RefusalSource::SafetyClassifier`]; `content_policy_violation` and
///   moderation flags (`metadata.reasons` / `flagged_input` on a 403) become
///   [`RefusalSource::ContentFilter`]. Never retryable.
/// - A 402 is retryable only for the in-flight budget
///   (`limit_source: openrouter_in_flight_budget`) with a delay; exhausted
///   credits or key limits are permanent.
/// - "No available model provider meets your routing requirements" or "No
///   allowed providers" (503 or 404) is a permanent configuration error
///   (code `no_eligible_provider`), unless it carries a retry hint
///   (`Retry-After`, or a delay in the body) and a retryable status: OpenRouter
///   adds `Retry-After` when every attempted provider returned a retry hint,
///   so it is then transient and keeps the hint. A 404 stays permanent.
/// - Otherwise [`Error::ProviderError`], classified by `error_type` first,
///   then by status (a numeric `error.code` stands in for the status on
///   errors that arrive after a 200).
pub(crate) fn openrouter_error(
    status: Option<u16>,
    retry_after: Option<Duration>,
    body: &str,
) -> Error {
    let value: Value = serde_json::from_str(body).unwrap_or_default();
    let error = value.get("error").unwrap_or(&value);
    let metadata = error.get("metadata");
    let meta_str = |name: &str| metadata.and_then(|m| m.get(name)).and_then(Value::as_str);
    let error_type = meta_str("error_type");
    let message = error.get("message").and_then(Value::as_str);
    let effective_status = status
        .filter(|status| !(200..300).contains(status))
        .or_else(|| {
            error
                .get("code")
                .and_then(Value::as_u64)
                .map(|code| code as u16)
        });
    let moderation =
        metadata.is_some_and(|m| m.get("flagged_input").is_some() || m.get("reasons").is_some());

    if matches!(error_type, Some("refusal" | "content_policy_violation"))
        || (effective_status == Some(403) && moderation)
    {
        let source = if error_type == Some("refusal") {
            RefusalSource::SafetyClassifier
        } else {
            RefusalSource::ContentFilter
        };
        let mut refusal = Refusal::new(source);
        refusal.category = meta_str("provider_code").map(String::from).or_else(|| {
            metadata
                .and_then(|m| m.get("reasons"))
                .and_then(Value::as_array)
                .map(|reasons| {
                    reasons
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .filter(|reasons| !reasons.is_empty())
        });
        refusal.explanation = message.map(String::from);
        refusal.provider_details = error.clone();
        return Error::Refusal(Box::new(refusal));
    }

    let mut converted =
        Error::provider_error("OpenRouter request failed", status, retry_after, body);
    if effective_status == Some(402)
        && let Error::ProviderError {
            code, retry_after, ..
        } = &mut converted
    {
        *code = Some("payment_required".into());
        // Only the in-flight spending budget clears by itself.
        if meta_str("limit_source") != Some("openrouter_in_flight_budget") {
            *retry_after = None;
        }
    }
    let unroutable = message.is_some_and(|message| {
        let message = message.to_ascii_lowercase();
        message.contains("routing requirements")
            || message.contains("no allowed providers")
            || message.contains("no models match your request")
    });
    let outage = matches!(
        error_type,
        Some("provider_overloaded" | "provider_unavailable" | "rate_limit_exceeded")
    );
    // "When every attempted provider returned a retry hint, the error
    // response also carries a `Retry-After`": the providers are only
    // unavailable for now, and a retryable status keeps the hint.
    let waiting = converted.retry_after().is_some() && converted.is_retryable();
    if unroutable
        && !outage
        && !waiting
        && let Error::ProviderError { code, .. } = &mut converted
    {
        *code = Some("no_eligible_provider".into());
    }
    converted
}

#[async_trait::async_trait]
impl ProviderClient for OpenRouterClient {
    async fn generate_text(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        let prepared = self.build_request(conversation, config)?;
        let response = self.send_chat(&prepared).await?;
        let body = response.text().await?;
        let fallbacks = prepared.body.models.is_some();
        response::parse_response(
            &body,
            &ResponseContext::new(prepared.model, config.clone(), fallbacks),
        )
    }

    async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let mut prepared = self.build_request(conversation, config)?;
        prepared.body.stream = Some(true);
        let response = self.send_chat(&prepared).await?;
        let fallbacks = prepared.body.models.is_some();
        let context = ResponseContext::new(prepared.model, config.clone(), fallbacks);
        Ok(super::idle::parse_with_idle_timeout(
            response.bytes_stream(),
            self.stream_idle_timeout,
            move |bytes| response::stream_events(bytes, context).boxed(),
        ))
    }
}
