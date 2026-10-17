//! OpenRouter request options for provider routing preferences, the Auto
//! Router, model fallbacks, the Auto Exacto policy and app attribution.
//!
//! Every field is optional and omitted from the request unless set, so
//! OpenRouter's defaults apply. Client-level defaults
//! ([`OpenRouterClient::with_default_options`](super::OpenRouterClient::with_default_options))
//! and per-request options ([`GenerationConfig::openrouter`](crate::config::GenerationConfig::openrouter))
//! are merged field by field, the request winning.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Per-request OpenRouter options. See the module documentation for merging.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterOptions {
    /// Provider routing preferences (`provider`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    /// Models to try, in order, when the requested model fails (`models`).
    /// Not sent by default. Without it OpenRouter never switches models.
    /// With it, OpenRouter falls back on any error of the primary model,
    /// including rate limits, downtime and moderation or refusal errors raised
    /// before any output, so a refused prompt may be answered by another model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_models: Option<Vec<String>>,
    /// Auto Router settings, applied only when the model is `openrouter/auto`
    /// or `openrouter/auto-beta` and ignored for every other model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_router: Option<AutoRouterOptions>,
    /// How to handle Auto Exacto on tool requests. Defaults to
    /// [`AutoExactoPolicy::CacheAware`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_exacto: Option<AutoExactoPolicy>,
    /// Stable per-end-user identifier (`user`) for abuse isolation. Without
    /// it, requests share the account-level identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Processing tier (`service_tier`). Not sent by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenRouterServiceTier>,
}

impl OpenRouterOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider(mut self, provider: ProviderPreferences) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_fallback_models<I, S>(mut self, models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.fallback_models = Some(models.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_auto_router(mut self, auto_router: AutoRouterOptions) -> Self {
        self.auto_router = Some(auto_router);
        self
    }

    pub fn with_auto_exacto(mut self, policy: AutoExactoPolicy) -> Self {
        self.auto_exacto = Some(policy);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn with_service_tier(mut self, tier: OpenRouterServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Merges these options over `defaults`. Fields set here win. `provider` and
    /// `auto_router` merge field by field.
    pub fn merged_over(&self, defaults: &OpenRouterOptions) -> OpenRouterOptions {
        OpenRouterOptions {
            provider: merge_nested(
                &self.provider,
                &defaults.provider,
                ProviderPreferences::merged_over,
            ),
            fallback_models: self
                .fallback_models
                .clone()
                .or_else(|| defaults.fallback_models.clone()),
            auto_router: merge_nested(
                &self.auto_router,
                &defaults.auto_router,
                AutoRouterOptions::merged_over,
            ),
            auto_exacto: self.auto_exacto.or(defaults.auto_exacto),
            user: self.user.clone().or_else(|| defaults.user.clone()),
            service_tier: self.service_tier.or(defaults.service_tier),
        }
    }
}

fn merge_nested<T: Clone>(over: &Option<T>, base: &Option<T>, merge: fn(&T, &T) -> T) -> Option<T> {
    match (over, base) {
        (Some(over), Some(base)) => Some(merge(over, base)),
        (over, base) => over.clone().or_else(|| base.clone()),
    }
}

/// OpenRouter `service_tier` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterServiceTier {
    Auto,
    Default,
    Flex,
    Priority,
    Scale,
}

/// What arko does about Auto Exacto, OpenRouter's reordering of providers by
/// tool-calling quality on every request with tools. It overrides sticky
/// routing, so an agent loop can land on a different provider on each turn
/// and miss its prompt cache. Sorting providers by price turns it off.
/// <https://openrouter.ai/docs/guides/routing/auto-exacto>
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoExactoPolicy {
    /// Send `provider.sort: "price"` on tool requests with prompt caching
    /// enabled when cache misses are costly because the model charges for cache
    /// writes (catalog `input_cache_write` above zero, for example Claude and
    /// GPT-5.6 and later) or, without catalog metadata, is an `anthropic/` or
    /// `openai/` model. Otherwise OpenRouter's default routing applies.
    #[default]
    CacheAware,
    /// Never add a sort, so Auto Exacto runs on every tool request.
    AllowAutoExacto,
    /// Always send `provider.sort: "price"` on tool requests.
    DisableAutoExacto,
}

/// The `provider` routing object. Unset fields are omitted. OpenRouter defaults
/// to price-weighted load balancing across healthy providers,
/// fallbacks allowed, unsupported parameters silently dropped
/// (`require_parameters: false`), data collection allowed, no per-request ZDR
/// (account settings still apply), no sort (tool requests get Auto Exacto).
/// Provider slugs are lowercase (`anthropic`, `openai`, `google-vertex`,
/// `amazon-bedrock`, `azure`, ...). `GET /api/v1/providers` lists them.
/// <https://openrouter.ai/docs/guides/routing/provider-selection>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderPreferences {
    /// Providers to try first, in order. Disables sticky routing (and so
    /// prompt-cache affinity). Prefer [`only`](Self::only) to restrict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// Providers allowed for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Providers to skip for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Route only to providers supporting every parameter sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    /// Only Zero Data Retention endpoints (OR-ed with the account setting).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<PerformanceThreshold>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<PerformanceThreshold>,
    /// Provider-specific options keyed by provider slug, forwarded only to
    /// the provider that serves the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Map<String, Value>>,
}

impl ProviderPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    /// Only `slug` serves the request, with no fallback to other providers.
    /// Uses `only` rather than `order`, which would disable sticky routing.
    pub fn pinned(slug: impl Into<String>) -> Self {
        Self {
            only: Some(vec![slug.into()]),
            allow_fallbacks: Some(false),
            ..Self::default()
        }
    }

    /// Only Zero Data Retention endpoints. Under ZDR, Claude is served only by
    /// Amazon Bedrock and Google Vertex (Anthropic's own API is not ZDR) and
    /// OpenAI models mostly by Azure.
    pub fn zdr_only() -> Self {
        Self {
            zdr: Some(true),
            ..Self::default()
        }
    }

    pub fn with_order<I: IntoIterator<Item = S>, S: Into<String>>(mut self, slugs: I) -> Self {
        self.order = Some(slugs.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_only<I: IntoIterator<Item = S>, S: Into<String>>(mut self, slugs: I) -> Self {
        self.only = Some(slugs.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_ignore<I: IntoIterator<Item = S>, S: Into<String>>(mut self, slugs: I) -> Self {
        self.ignore = Some(slugs.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_allow_fallbacks(mut self, allow: bool) -> Self {
        self.allow_fallbacks = Some(allow);
        self
    }

    pub fn with_require_parameters(mut self, require: bool) -> Self {
        self.require_parameters = Some(require);
        self
    }

    pub fn with_data_collection(mut self, data_collection: DataCollection) -> Self {
        self.data_collection = Some(data_collection);
        self
    }

    pub fn with_zdr(mut self, zdr: bool) -> Self {
        self.zdr = Some(zdr);
        self
    }

    pub fn with_enforce_distillable_text(mut self, enforce: bool) -> Self {
        self.enforce_distillable_text = Some(enforce);
        self
    }

    pub fn with_quantizations(mut self, quantizations: Vec<Quantization>) -> Self {
        self.quantizations = Some(quantizations);
        self
    }

    pub fn with_sort(mut self, sort: impl Into<ProviderSort>) -> Self {
        self.sort = Some(sort.into());
        self
    }

    pub fn with_max_price(mut self, max_price: MaxPrice) -> Self {
        self.max_price = Some(max_price);
        self
    }

    pub fn with_preferred_min_throughput(mut self, threshold: PerformanceThreshold) -> Self {
        self.preferred_min_throughput = Some(threshold);
        self
    }

    pub fn with_preferred_max_latency(mut self, threshold: PerformanceThreshold) -> Self {
        self.preferred_max_latency = Some(threshold);
        self
    }

    /// Adds provider-specific options for `slug`.
    pub fn with_provider_options(mut self, slug: impl Into<String>, options: Value) -> Self {
        self.options
            .get_or_insert_with(Map::new)
            .insert(slug.into(), options);
        self
    }

    /// These preferences over `base`, field by field.
    pub fn merged_over(&self, base: &ProviderPreferences) -> ProviderPreferences {
        fn pick<T: Clone>(over: &Option<T>, base: &Option<T>) -> Option<T> {
            over.clone().or_else(|| base.clone())
        }
        ProviderPreferences {
            order: pick(&self.order, &base.order),
            only: pick(&self.only, &base.only),
            ignore: pick(&self.ignore, &base.ignore),
            allow_fallbacks: pick(&self.allow_fallbacks, &base.allow_fallbacks),
            require_parameters: pick(&self.require_parameters, &base.require_parameters),
            data_collection: pick(&self.data_collection, &base.data_collection),
            zdr: pick(&self.zdr, &base.zdr),
            enforce_distillable_text: pick(
                &self.enforce_distillable_text,
                &base.enforce_distillable_text,
            ),
            quantizations: pick(&self.quantizations, &base.quantizations),
            sort: pick(&self.sort, &base.sort),
            max_price: pick(&self.max_price, &base.max_price),
            preferred_min_throughput: pick(
                &self.preferred_min_throughput,
                &base.preferred_min_throughput,
            ),
            preferred_max_latency: pick(&self.preferred_max_latency, &base.preferred_max_latency),
            options: pick(&self.options, &base.options),
        }
    }

    /// Whether any field is set.
    pub fn is_empty(&self) -> bool {
        self == &ProviderPreferences::default()
    }
}

/// `provider.data_collection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    /// Allow providers that store data non-transiently and may train on it
    /// (OpenRouter's default).
    Allow,
    /// Only providers that don't collect user data.
    Deny,
}

/// `provider.quantizations` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Int4,
    Int8,
    Fp4,
    Mxfp4,
    Nvfp4,
    Fp6,
    Fp8,
    Mxfp8,
    Fp16,
    Bf16,
    Fp32,
    Unknown,
}

/// What to sort providers by (`provider.sort` or `provider.sort.by`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortBy {
    Price,
    Throughput,
    Latency,
    /// Tool-calling quality (what Auto Exacto does on tool requests).
    Exacto,
}

/// `provider.sort.partition`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortPartition {
    /// Sort endpoints within each model. Fallback models stay fallbacks
    /// (OpenRouter's default).
    Model,
    /// Sort all endpoints of all models together.
    None,
}

/// A `provider.sort` strategy (`"price"`) or an object with a partition.
/// When set, OpenRouter doesn't load-balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProviderSort {
    By(SortBy),
    Config {
        #[serde(skip_serializing_if = "Option::is_none")]
        by: Option<SortBy>,
        #[serde(skip_serializing_if = "Option::is_none")]
        partition: Option<SortPartition>,
    },
}

impl From<SortBy> for ProviderSort {
    fn from(by: SortBy) -> Self {
        ProviderSort::By(by)
    }
}

impl ProviderSort {
    pub fn by(&self) -> Option<SortBy> {
        match self {
            ProviderSort::By(by) => Some(*by),
            ProviderSort::Config { by, .. } => *by,
        }
    }
}

/// `provider.max_price`, in USD per million tokens. The `request`, `image`
/// and `audio` fields are per request, image and audio unit instead.
/// Requests with no endpoint under the limits fail. Sent as JSON numbers,
/// as in OpenRouter's routing
/// guide (the OpenAPI schema types them as strings).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaxPrice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<f64>,
}

impl MaxPrice {
    /// Prompt and completion limits in USD per million tokens.
    pub fn per_million_tokens(prompt: f64, completion: f64) -> Self {
        Self {
            prompt: Some(prompt),
            completion: Some(completion),
            ..Self::default()
        }
    }
}

/// `preferred_min_throughput` (tokens per second) or `preferred_max_latency`
/// (seconds), as a single p50 value or per-percentile cutoffs. Endpoints missing
/// the thresholds are deprioritized, not excluded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PerformanceThreshold {
    P50(f64),
    Percentiles {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p50: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p75: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p90: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p99: Option<f64>,
    },
}

/// Auto Router cost band (`cost_tier`). Without one, the router behaves
/// roughly as `Low`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AutoRouterCostTier {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// Settings for the Auto Router (`openrouter/auto`, `openrouter/auto-beta`),
/// ignored for every other model so they can be set once on the client.
/// The plugin id is chosen from the slug (`auto-router` or
/// `auto-beta-router`, the other id would be silently ignored).
/// <https://openrouter.ai/docs/guides/routing/routers/auto-router>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoRouterOptions {
    /// Model patterns the router may choose from (`anthropic/*`, `openai/gpt-5*`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_models: Option<Vec<String>>,
    /// Model patterns the router must not choose. Wins over `allowed_models`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded_models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_tier: Option<AutoRouterCostTier>,
    /// Provider preferences applied, over the request's own, only when the
    /// model is an Auto Router slug. How far the router's own task
    /// classifier honours ZDR is not documented. For a hard guarantee use an
    /// API key with a ZDR guardrail on OpenRouter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
}

impl AutoRouterOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Auto Router requests limited to Zero Data Retention endpoints with data
    /// collection denied. Other models are unaffected.
    pub fn zdr() -> Self {
        Self {
            provider: Some(
                ProviderPreferences::zdr_only().with_data_collection(DataCollection::Deny),
            ),
            ..Self::default()
        }
    }

    pub fn with_allowed_models<I: IntoIterator<Item = S>, S: Into<String>>(
        mut self,
        patterns: I,
    ) -> Self {
        self.allowed_models = Some(patterns.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_excluded_models<I: IntoIterator<Item = S>, S: Into<String>>(
        mut self,
        patterns: I,
    ) -> Self {
        self.excluded_models = Some(patterns.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_cost_tier(mut self, tier: AutoRouterCostTier) -> Self {
        self.cost_tier = Some(tier);
        self
    }

    pub fn with_provider(mut self, provider: ProviderPreferences) -> Self {
        self.provider = Some(provider);
        self
    }

    /// These options over `base`, field by field.
    pub fn merged_over(&self, base: &AutoRouterOptions) -> AutoRouterOptions {
        AutoRouterOptions {
            allowed_models: self
                .allowed_models
                .clone()
                .or_else(|| base.allowed_models.clone()),
            excluded_models: self
                .excluded_models
                .clone()
                .or_else(|| base.excluded_models.clone()),
            cost_tier: self.cost_tier.or(base.cost_tier),
            provider: merge_nested(
                &self.provider,
                &base.provider,
                ProviderPreferences::merged_over,
            ),
        }
    }

    /// The plugin entry for `plugin_id`, or `None` when no plugin setting is
    /// present.
    pub(crate) fn plugin(&self, plugin_id: &str) -> Option<Value> {
        if self.allowed_models.is_none()
            && self.excluded_models.is_none()
            && self.cost_tier.is_none()
        {
            return None;
        }
        let mut plugin = Map::new();
        plugin.insert("id".into(), Value::String(plugin_id.into()));
        if let Some(allowed) = &self.allowed_models {
            plugin.insert("allowed_models".into(), serde_json::json!(allowed));
        }
        if let Some(excluded) = &self.excluded_models {
            plugin.insert("excluded_models".into(), serde_json::json!(excluded));
        }
        if let Some(tier) = self.cost_tier {
            plugin.insert("cost_tier".into(), serde_json::json!(tier));
        }
        Some(Value::Object(plugin))
    }
}

/// App attribution headers for OpenRouter's rankings and analytics. Never
/// sent unless configured with
/// [`OpenRouterClient::with_app_attribution`](super::OpenRouterClient::with_app_attribution).
/// <https://openrouter.ai/docs/app-attribution>
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppAttribution {
    /// The app's URL (`HTTP-Referer`), which identifies the app.
    pub url: String,
    /// Display name (`X-OpenRouter-Title`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Marketplace categories (`X-OpenRouter-Categories`, comma-separated).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub categories: Vec<String>,
    /// Create the app hidden from public rankings
    /// (`X-OpenRouter-App-Visibility: hidden`). Applies only when the app is
    /// first created.
    pub hidden: bool,
}

impl AppAttribution {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Self::default()
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_categories<I: IntoIterator<Item = S>, S: Into<String>>(
        mut self,
        categories: I,
    ) -> Self {
        self.categories = categories.into_iter().map(Into::into).collect();
        self
    }

    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    pub(crate) fn headers(&self) -> Vec<(&'static str, String)> {
        let mut headers = vec![("HTTP-Referer", self.url.clone())];
        if let Some(title) = &self.title {
            headers.push(("X-OpenRouter-Title", title.clone()));
        }
        if !self.categories.is_empty() {
            headers.push(("X-OpenRouter-Categories", self.categories.join(",")));
        }
        if self.hidden {
            headers.push(("X-OpenRouter-App-Visibility", "hidden".to_string()));
        }
        headers
    }
}
