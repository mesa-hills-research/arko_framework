//! The OpenRouter model catalog (`GET /api/v1/models`, or
//! `/api/v1/models/user` for the models an API key may use) parsed into
//! [`Model`]s. Fetched only on request and cached in memory by the client.
//! https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties

use crate::{
    error::Error,
    model::{Model, ModelPricing, ModelType, Provider},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, time::SystemTime};

/// Catalog metadata for one OpenRouter model, kept on
/// [`Model::openrouter`](crate::model::Model::openrouter).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterModelInfo {
    /// Display name, for example `Anthropic: Claude Sonnet 5`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Dated slug the alias resolves to (for example
    /// `anthropic/claude-sonnet-5-20260630`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_slug: Option<String>,
    /// For `~…-latest` aliases, the concrete model's slug.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    /// The top provider's output limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    /// Request parameters the model accepts (`tools`, `reasoning`,
    /// `temperature`, ...). Empty when the catalog lists none (routers).
    pub supported_parameters: Vec<String>,
    /// Reasoning controls; `None` for non-reasoning models and routers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenRouterReasoningInfo>,
    /// Whether prices are published (routers report `-1`).
    pub pricing_known: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<String>,
}

impl OpenRouterModelInfo {
    /// Whether the catalog lists `parameter` as supported. A model with an
    /// empty list (routers, some aliases) is treated as accepting anything.
    pub fn supports_parameter(&self, parameter: &str) -> bool {
        self.supported_parameters.is_empty()
            || self.supported_parameters.iter().any(|p| p == parameter)
    }

    /// Whether the model accepts image input.
    pub fn accepts_images(&self) -> bool {
        self.input_modalities.iter().any(|m| m == "image")
    }
}

/// A model's reasoning controls from the catalog `reasoning` object.
/// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterReasoningInfo {
    /// Whether effort selection is exposed at all (`supported_efforts`
    /// present in the catalog).
    pub effort_selectable: bool,
    /// Accepted efforts, highest first; `None` with `effort_selectable` means
    /// every effort is accepted. Kept as strings so new levels don't break
    /// parsing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_efforts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_enabled: Option<bool>,
    /// Reasoning can't be turned off (`effort: "none"` and `enabled: false`
    /// are rejected).
    pub mandatory: bool,
    pub supports_max_tokens: bool,
}

/// Which catalog endpoint a catalog came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogSource {
    /// `GET /api/v1/models`: every public text model.
    Public,
    /// `GET /api/v1/models/user`: filtered by the key's provider preferences,
    /// privacy settings and guardrails.
    User,
    /// Loaded from JSON supplied by the caller.
    Supplied,
}

/// A parsed model catalog.
#[derive(Debug, Clone)]
pub struct OpenRouterCatalog {
    models: Vec<Model>,
    index: HashMap<String, usize>,
    source: CatalogSource,
    fetched_at: SystemTime,
}

impl OpenRouterCatalog {
    /// Parses a models-list response body (`{"data": [...]}`). Entries that
    /// can't be read are skipped; unknown fields are ignored.
    pub fn from_json(body: &str) -> Result<Self, Error> {
        Self::parse(body, CatalogSource::Supplied)
    }

    pub(crate) fn parse(body: &str, source: CatalogSource) -> Result<Self, Error> {
        let value: Value = serde_json::from_str(body)?;
        let entries = value.get("data").and_then(Value::as_array).ok_or_else(|| {
            Error::ApiError("OpenRouter models response has no `data` array".into())
        })?;
        let models: Vec<Model> = entries
            .iter()
            .filter_map(|entry| {
                let model = model_from_entry(entry);
                if model.is_none() {
                    log::debug!("arko_ai: skipping an OpenRouter catalog entry without an id");
                }
                model
            })
            .collect();
        Ok(Self::from_models(models, source))
    }

    /// A catalog of already-built models (for example saved snapshots).
    pub fn from_models(models: Vec<Model>, source: CatalogSource) -> Self {
        let mut index = HashMap::new();
        for (position, model) in models.iter().enumerate() {
            index.entry(model.id.clone()).or_insert(position);
        }
        for (position, model) in models.iter().enumerate() {
            if let Some(canonical) = model
                .openrouter
                .as_ref()
                .and_then(|info| info.canonical_slug.clone())
            {
                index.entry(canonical).or_insert(position);
            }
        }
        Self {
            models,
            index,
            source,
            fetched_at: SystemTime::now(),
        }
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn source(&self) -> CatalogSource {
        self.source
    }

    pub fn fetched_at(&self) -> SystemTime {
        self.fetched_at
    }

    /// The model with this slug or canonical slug. A variant suffix
    /// (`:floor`, `:nitro`, `:thinking`, ...) not listed on its own resolves
    /// to the base model; the returned model keeps the requested slug.
    pub fn get(&self, slug: &str) -> Option<Model> {
        if let Some(&position) = self.index.get(slug) {
            let mut model = self.models[position].clone();
            model.id = slug.to_string();
            return Some(model);
        }
        let (base, _) = slug.split_once(':')?;
        let &position = self.index.get(base)?;
        let mut model = self.models[position].clone();
        model.id = slug.to_string();
        Some(model)
    }

    /// Whether the catalog lists the slug (or its base without a variant).
    pub fn contains(&self, slug: &str) -> bool {
        self.index.contains_key(slug)
            || slug
                .split_once(':')
                .is_some_and(|(base, _)| self.index.contains_key(base))
    }
}

/// USD per token (as the catalog's decimal string) to USD per million
/// tokens. Negative values mark unpublished router prices.
fn per_million(value: Option<&Value>) -> Option<f64> {
    let per_token = match value? {
        Value::String(text) => text.trim().parse::<f64>().ok()?,
        Value::Number(number) => number.as_f64()?,
        _ => return None,
    };
    (per_token.is_finite() && per_token >= 0.0)
        .then(|| (per_token * 1_000_000.0 * 1e9).round() / 1e9)
}

fn usize_field(value: Option<&Value>) -> Option<usize> {
    value.and_then(Value::as_u64).map(|n| n as usize)
}

fn strings(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn reasoning_info(value: &Value) -> OpenRouterReasoningInfo {
    let efforts = value.get("supported_efforts");
    OpenRouterReasoningInfo {
        effort_selectable: efforts.is_some(),
        supported_efforts: efforts.filter(|v| v.is_array()).map(|v| strings(Some(v))),
        default_effort: value
            .get("default_effort")
            .and_then(Value::as_str)
            .map(String::from),
        default_enabled: value.get("default_enabled").and_then(Value::as_bool),
        mandatory: value
            .get("mandatory")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        supports_max_tokens: value
            .get("supports_max_tokens")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

/// Price keys of `pricing.overrides` entries (OpenAPI `PricingOverride`).
/// Entries use "the same keys and units as the base pricing object", so the
/// base object's keys are price keys too.
const OVERRIDE_PRICE_KEYS: &[&str] = &[
    "prompt",
    "completion",
    "input_cache_read",
    "input_cache_write",
    "input_cache_write_1h",
    "audio",
    "input_audio_cache",
];

/// The long-context tier: the first `pricing.overrides` entry whose only
/// condition is `min_prompt_tokens` (the entry applies when the prompt has
/// more tokens than that), with its threshold.
///
/// An entry applies only when all its condition fields match, and arko
/// evaluates only the prompt size, so entries with any other condition (the
/// `utc_start`/`utc_end` time window, `utc_days`, or a condition added
/// later) are skipped, as OpenRouter asks: "Consumers should skip entries
/// containing condition fields they do not recognize rather than apply their
/// prices". Keys that aren't prices count as conditions.
/// https://openrouter.ai/docs/guides/overview/models#pricing-overrides
fn long_context_override(pricing: &Value) -> Option<(&Value, usize)> {
    let is_price = |key: &str| {
        OVERRIDE_PRICE_KEYS.contains(&key)
            || (key != "overrides" && pricing.get(key).is_some_and(Value::is_string))
    };
    pricing
        .get("overrides")?
        .as_array()?
        .iter()
        .find_map(|entry| {
            let fields = entry.as_object()?;
            let threshold = fields
                .get("min_prompt_tokens")?
                .as_f64()
                .filter(|n| n.is_finite() && *n >= 0.0)?;
            fields
                .keys()
                .all(|key| key == "min_prompt_tokens" || is_price(key))
                .then_some((entry, threshold as usize))
        })
}

/// One catalog entry as a [`Model`]: context length, the top provider's
/// output limit, image input, prices (with the first prompt-size override as
/// the long-context tier) and the rest in [`OpenRouterModelInfo`].
pub(crate) fn model_from_entry(entry: &Value) -> Option<Model> {
    let id = entry.get("id")?.as_str()?.to_string();
    let pricing = entry.get("pricing").cloned().unwrap_or(Value::Null);
    let price = |name: &str| per_million(pricing.get(name));
    let prompt = price("prompt");
    let completion = price("completion");
    let cache_read = price("input_cache_read");
    let cache_write = price("input_cache_write");
    let long = long_context_override(&pricing);
    let long_price = |name: &str| long.and_then(|(o, _)| per_million(o.get(name)));

    let supported_parameters = strings(entry.get("supported_parameters"));
    let reasoning = entry
        .get("reasoning")
        .filter(|r| r.is_object())
        .map(reasoning_info);
    let architecture = entry.get("architecture");
    let input_modalities = strings(architecture.and_then(|a| a.get("input_modalities")));
    let top_provider = entry.get("top_provider");
    let context_length = usize_field(entry.get("context_length"))
        .or_else(|| usize_field(top_provider.and_then(|p| p.get("context_length"))));
    let max_completion_tokens =
        usize_field(top_provider.and_then(|p| p.get("max_completion_tokens")));

    let info = OpenRouterModelInfo {
        name: entry.get("name").and_then(Value::as_str).map(String::from),
        canonical_slug: entry
            .get("canonical_slug")
            .and_then(Value::as_str)
            .map(String::from),
        alias_target: entry
            .pointer("/alias_target/slug")
            .and_then(Value::as_str)
            .map(String::from),
        context_length,
        max_completion_tokens,
        output_modalities: strings(architecture.and_then(|a| a.get("output_modalities"))),
        supported_parameters: supported_parameters.clone(),
        reasoning: reasoning.clone(),
        pricing_known: prompt.is_some() && completion.is_some(),
        expiration_date: entry
            .get("expiration_date")
            .and_then(Value::as_str)
            .map(String::from),
        input_modalities,
    };

    Some(Model {
        provider: Provider::OpenRouter,
        id,
        model_type: if reasoning.is_some() || supported_parameters.iter().any(|p| p == "reasoning")
        {
            ModelType::Reasoning
        } else {
            ModelType::Standard
        },
        max_input_tokens: context_length.unwrap_or(0),
        max_output_tokens: max_completion_tokens.unwrap_or(0),
        supports_image_input: info.accepts_images(),
        supports_cache: cache_read.is_some() || cache_write.is_some(),
        pricing: ModelPricing {
            input_cost_per_million: prompt.unwrap_or(0.0),
            output_cost_per_million: completion.unwrap_or(0.0),
            cached_input_cost_per_million: cache_read,
            cache_write_cost_per_million: cache_write,
            cache_write_1h_cost_per_million: price("input_cache_write_1h"),
            large_input_threshold: long.map(|(_, threshold)| threshold),
            large_input_cost_per_million: long_price("prompt"),
            large_output_cost_per_million: long_price("completion"),
            large_cached_input_cost_per_million: long_price("input_cache_read"),
            large_cache_write_cost_per_million: long_price("input_cache_write"),
            large_cache_write_1h_cost_per_million: long_price("input_cache_write_1h"),
            ..ModelPricing::default()
        },
        openrouter: Some(info),
    })
}
