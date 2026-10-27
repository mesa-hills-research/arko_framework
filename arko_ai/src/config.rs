use crate::reasoning::{OpenAIReasoningMode, ReasoningConfig};
use crate::tool::ToolConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

/// Omitted fields take their defaults when deserializing, so a partial JSON or
/// YAML config (for example `{"max_tokens": 4096}`) is valid.
///
/// Configs saved by earlier versions carry a `temperature_specified` flag next
/// to `temperature`. They still load. `temperature_specified: false` means no
/// temperature, whatever `temperature` holds.
///
/// One config can serve models of several providers. A setting the model's
/// provider has no use for (an `openai_*` setting on Claude, an `anthropic_*`
/// one on GPT or Gemini, another provider's [`ReasoningConfig`] variant) is
/// dropped from the request, with a warning logged once per provider and
/// setting. A value the provider can't accept for a setting it does use is
/// an error.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(remote = "Self", default)]
pub struct GenerationConfig {
    /// Sampling temperature. `None` (the default) sends none, so the model's
    /// default applies. OpenAI reasoning models and Google's Interactions API
    /// take no temperature, so a set one is an error for requests to them.
    /// Claude models that take no sampling parameters (Fable 5.x, Mythos 5.1,
    /// Opus 4.7 and later, Sonnet 5 and later, Haiku 5.5) drop it, with a
    /// warning logged once per model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Output token limit. `None` (the default) lets the model produce up to
    /// its maximum. OpenAI (API key), Anthropic and Gemini requests send the
    /// model's registered maximum, since Anthropic requires a value. OpenRouter
    /// requests send none. ChatGPT OAuth requests never send one, because the
    /// backend rejects `max_output_tokens`. A set limit is ignored there, with
    /// a warning logged once.
    pub max_tokens: Option<usize>,
    pub verbosity: Option<Verbosity>,
    pub reasoning: Option<ReasoningConfig>,
    pub openai_reasoning_mode: Option<OpenAIReasoningMode>,
    /// Controls Anthropic thinking independently from the selected reasoning mode.
    ///
    /// `None` preserves existing behavior and uses the model's normal default.
    /// A supplied Anthropic reasoning configuration still enables thinking.
    ///
    /// `Some(true)` explicitly enables thinking. When no Anthropic reasoning
    /// configuration is supplied, adaptive thinking is selected for models that
    /// support it.
    ///
    /// `Some(false)` explicitly sends `thinking.type = "disabled"`. This is
    /// rejected for models such as Claude Fable 5 where adaptive thinking is
    /// always enabled.
    pub anthropic_thinking_enabled: Option<bool>,
    /// Turns provider prompt caching on or off for this request. `None`
    /// inherits [`Conversation::enable_cache`], which defaults to enabled.
    ///
    /// Caching is on by default. Turning it off takes effect where the
    /// provider offers a switch. Where caching is implicit it stays on:
    ///
    /// - On Anthropic and Claude models on OpenRouter, caching happens only
    ///   through the `cache_control` this client sends while caching is
    ///   enabled, so turning it off stops caching.
    /// - On OpenAI GPT-5.6 and GPT-6 with an API key, enabled sends
    ///   `prompt_cache_options.mode: "implicit"`, disabled `"explicit"` with
    ///   no breakpoints, so nothing is read from or written to the cache.
    /// - On OpenAI models before GPT-5.6, caching is automatic and can't be
    ///   turned off.
    /// - Over ChatGPT OAuth (any model), the backend caches implicitly and
    ///   takes no cache options. Disabled caching is ignored, with a warning
    ///   logged once.
    /// - Gemini uses implicit caching, managed by Google. It can't be turned
    ///   off.
    /// - On OpenRouter, enabled sends `prompt_cache_key` and, on Claude
    ///   models, a top-level `cache_control`. Disabled sends neither, and on
    ///   OpenAI models that bill cache writes (GPT-5.6 and later) also
    ///   `prompt_cache_options.mode: "explicit"`. Other upstreams that cache
    ///   implicitly (Gemini, earlier OpenAI models and others) keep caching.
    ///
    /// Every OpenAI request, and every OpenRouter request as `session_id`,
    /// carries the conversation's `prompt_cache_key` either way. It routes
    /// requests but doesn't enable caching by itself.
    ///
    /// [`Conversation::enable_cache`]: crate::conversation::Conversation::enable_cache
    pub prompt_caching: Option<bool>,
    /// Requests summarized Anthropic thinking text while thinking is enabled
    /// (adaptive or manual budgets).
    ///
    /// Claude Fable 5.1, Mythos 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7,
    /// Sonnet 5.5, Sonnet 5 and Haiku 5.5 default to omitted thinking text, so this
    /// sends `thinking.display: "summarized"`. On Opus 4.6, Sonnet 4.6 and earlier
    /// thinking models summarized text is already the default and nothing extra
    /// is sent.
    /// `None` and `Some(false)` keep the provider default.
    /// <https://platform.claude.com/docs/en/build-with-claude/thinking>
    pub anthropic_summarized_thinking: Option<bool>,
    /// Requests the 1M-token context window on Anthropic models.
    ///
    /// Claude 4.6 and later models include 1M context without a beta header, so
    /// the flag is accepted there and sends nothing. Their prices don't depend on
    /// it. Claude Haiku 5.5 charges more for any prompt over 100,000 tokens, and
    /// the other models charge the same rate across the window. The
    /// `context-1m-2025-08-07` beta that gave Claude Sonnet 4 and 4.5 1M context
    /// was retired on 2026-04-30, so the flag is rejected for those models and
    /// any other model without 1M context.
    pub long_context: Option<bool>,
    /// Tool calling configuration. When set, tool definitions are sent
    /// with the request and the model may respond with tool calls instead
    /// of or in addition to text.
    pub tool_config: Option<ToolConfig>,
    /// Enables interleaved thinking for supported Anthropic models.
    ///
    /// When enabled, Claude can produce thinking blocks after receiving tool
    /// results, allowing it to reason about intermediate results before
    /// deciding what to do next.
    ///
    /// `None` uses the model's normal behavior. Models with automatic
    /// interleaved thinking enable it whenever adaptive thinking is active.
    ///
    /// `Some(true)` explicitly enables the feature and requires supported
    /// Anthropic reasoning.
    ///
    /// `Some(false)` explicitly disables the feature. This is rejected when
    /// the selected model automatically requires interleaved thinking with its
    /// active adaptive-thinking mode.
    pub interleaved_thinking: Option<bool>,
    /// OpenAI processing tier (`service_tier`). With an API key, Flex and Fast
    /// must be offered by the model (see [`ModelPricing::service_tiers`]).
    /// Over ChatGPT OAuth, as in Codex, only Fast mode is available, on the
    /// models Codex offers it for (GPT-6 Astra, Sol, Luna, 6.1 Sol and 6.1
    /// Terra, GPT-5.6 Sol, Terra and Luna, GPT-5.5). `Default` sends no tier
    /// there, and Flex and Auto are [`Error::UnsupportedFeature`].
    /// `None` sends nothing, so the project (or plan) default applies.
    ///
    /// [`Error::UnsupportedFeature`]: crate::error::Error::UnsupportedFeature
    ///
    /// [`ModelPricing::service_tiers`]: crate::model::ModelPricing::service_tiers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_service_tier: Option<OpenAIServiceTier>,
    /// Anthropic prompt-cache lifetime for the automatic `cache_control` this
    /// client sends when caching is enabled. `None` sends no `ttl`, which is the
    /// provider's 5-minute default. 1-hour writes cost 2x base input (5-minute
    /// writes 1.25x). <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
    /// Also the `ttl` of the `cache_control` sent for Claude models on OpenRouter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_cache_ttl: Option<AnthropicCacheTtl>,
    /// The prompt-cache lifetime arko reports for these requests, in seconds,
    /// in place of the provider's
    /// ([`prompt_cache_lifetime`](crate::prompt_cache::prompt_cache_lifetime)),
    /// for a host that knows how long its route keeps the cache. `None` (the
    /// default) reports the provider's. Nothing is sent to the provider.
    /// The setting changes only what arko takes the lifetime to be, such as
    /// for the automatic limit of arko_workflows' `wait_for_message`. A
    /// request that isn't cached (a model without caching, or caching
    /// turned off where the provider allows it) is still reported as
    /// uncached, and `Some(0)` reports it so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_lifetime_seconds: Option<u64>,
    /// Anthropic fast mode (`speed: "fast"` with the `fast-mode-2026-02-01` beta
    /// header, research preview, Claude API only). Supported on Claude Opus 5.5,
    /// Opus 5 and Opus 4.8 and rejected locally for other models. Priced at the
    /// model's fast-mode rates. The response's `usage.speed` decides the price.
    /// `None` and `Some(false)` send nothing.
    /// <https://platform.claude.com/docs/en/build-with-claude/fast-mode>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_fast_mode: Option<bool>,
    /// Anthropic `inference_geo` (where inference runs), for Claude 4.6 and later.
    /// Older models reject it. `None` sends nothing, so the workspace default
    /// applies. US-only inference costs 1.1x. The response's `usage.inference_geo`
    /// decides the price. <https://platform.claude.com/docs/en/manage-claude/data-residency>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_inference_geo: Option<AnthropicInferenceGeo>,
    /// OpenRouter routing and request options for this request. Fields set
    /// here override the client's defaults
    /// ([`OpenRouterClient::with_default_options`]) field by field. See
    /// [`OpenRouterOptions`]. Other providers drop it with a warning.
    ///
    /// [`OpenRouterClient::with_default_options`]: crate::providers::openrouter::OpenRouterClient::with_default_options
    /// [`OpenRouterOptions`]: crate::providers::openrouter::OpenRouterOptions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openrouter: Option<crate::providers::openrouter::OpenRouterOptions>,
}

/// Anthropic prompt-cache lifetimes (`cache_control.ttl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

/// Anthropic `inference_geo` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicInferenceGeo {
    /// Inference may run in any available geography (the API default).
    Global,
    /// Inference runs only in US-based infrastructure (1.1x pricing).
    Us,
}

/// OpenAI `service_tier` request values.
/// <https://developers.openai.com/api/docs/guides/flex-processing>
/// <https://developers.openai.com/api/docs/guides/fast-mode>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIServiceTier {
    /// The project's default tier.
    Auto,
    /// Standard processing.
    Default,
    /// Discounted, lower-priority processing.
    Flex,
    /// Fast mode (formerly Priority processing). Sent as `priority`.
    #[serde(rename = "priority")]
    Fast,
}

impl OpenAIServiceTier {
    /// The tier used for pricing when the response does not report one.
    /// `Auto` prices at standard rates.
    pub fn pricing_tier(self) -> crate::pricing::ServiceTier {
        match self {
            Self::Auto | Self::Default => crate::pricing::ServiceTier::Standard,
            Self::Flex => crate::pricing::ServiceTier::Flex,
            Self::Fast => crate::pricing::ServiceTier::Fast,
        }
    }
}

impl Serialize for GenerationConfig {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        GenerationConfig::serialize(self, serializer)
    }
}

impl<'de> Deserialize<'de> for GenerationConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ConfigVisitor;

        impl<'de> serde::de::Visitor<'de> for ConfigVisitor {
            type Value = GenerationConfig;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a generation config")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                map: A,
            ) -> Result<GenerationConfig, A::Error> {
                let mut flag = None;
                let mut config = GenerationConfig::deserialize(
                    serde::de::value::MapAccessDeserializer::new(WithoutTemperatureFlag {
                        inner: map,
                        flag: &mut flag,
                    }),
                )?;
                match flag {
                    Some(false) => config.temperature = None,
                    // The old field defaulted to 0.0.
                    Some(true) => {
                        config.temperature.get_or_insert(0.0);
                    }
                    None => {}
                }
                Ok(config)
            }
        }

        deserializer.deserialize_map(ConfigVisitor)
    }
}

/// The map of a config, minus the `temperature_specified` flag that configs
/// saved by earlier versions carry next to `temperature` (which was sent only
/// with the flag set). The flag's value is kept in `flag`. Values pass through
/// unbuffered, so YAML tags and other format features keep working.
struct WithoutTemperatureFlag<'a, A> {
    inner: A,
    flag: &'a mut Option<bool>,
}

impl<'de, A: serde::de::MapAccess<'de>> serde::de::MapAccess<'de>
    for WithoutTemperatureFlag<'_, A>
{
    type Error = A::Error;

    fn next_key_seed<K: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        while let Some(key) = self.inner.next_key::<String>()? {
            if key == "temperature_specified" {
                *self.flag = self.inner.next_value()?;
                continue;
            }
            return seed
                .deserialize(serde::de::value::StringDeserializer::new(key))
                .map(Some);
        }
        Ok(None)
    }

    fn next_value_seed<V: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, A::Error> {
        self.inner.next_value_seed(seed)
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

impl GenerationConfig {
    /// A config that sends `temperature`.
    #[deprecated(note = "use `GenerationConfig::default().with_temperature(temperature)`")]
    pub fn new(temperature: f64) -> Self {
        Self::default().with_temperature(temperature)
    }

    /// Sets per-request OpenRouter options (merged over the client defaults).
    pub fn with_openrouter_options(
        mut self,
        options: crate::providers::openrouter::OpenRouterOptions,
    ) -> Self {
        self.openrouter = Some(options);
        self
    }

    /// Requests an OpenAI processing tier.
    pub fn with_openai_service_tier(mut self, tier: OpenAIServiceTier) -> Self {
        self.openai_service_tier = Some(tier);
        self
    }

    /// Sends `temperature`. Set [`temperature`](Self::temperature) to `None`
    /// to send none again.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_openai_reasoning_mode(mut self, mode: OpenAIReasoningMode) -> Self {
        self.openai_reasoning_mode = Some(mode);
        self
    }

    /// Explicitly enables or disables Anthropic thinking.
    ///
    /// See [`anthropic_thinking_enabled`](Self::anthropic_thinking_enabled)
    /// for model-specific behavior.
    pub fn with_anthropic_thinking_enabled(mut self, enabled: bool) -> Self {
        self.anthropic_thinking_enabled = Some(enabled);
        self
    }

    /// Enables or disables supported provider prompt caching.
    ///
    /// See [`prompt_caching`](Self::prompt_caching) for provider-specific
    /// behavior. When not specified, the conversation-level setting is used.
    pub fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = Some(enabled);
        self
    }

    /// Requests summarized Anthropic thinking text while thinking is enabled.
    pub fn with_anthropic_summarized_thinking(mut self, enabled: bool) -> Self {
        self.anthropic_summarized_thinking = Some(enabled);
        self
    }

    /// Sets the Anthropic prompt-cache lifetime.
    pub fn with_anthropic_cache_ttl(mut self, ttl: AnthropicCacheTtl) -> Self {
        self.anthropic_cache_ttl = Some(ttl);
        self
    }

    /// Sets the prompt-cache lifetime arko reports for these requests, in
    /// whole seconds ([`prompt_cache_lifetime_seconds`](Self::prompt_cache_lifetime_seconds)).
    pub fn with_prompt_cache_lifetime(mut self, lifetime: std::time::Duration) -> Self {
        self.prompt_cache_lifetime_seconds = Some(lifetime.as_secs());
        self
    }

    /// Requests Anthropic fast mode on supported models.
    pub fn with_anthropic_fast_mode(mut self, enabled: bool) -> Self {
        self.anthropic_fast_mode = Some(enabled);
        self
    }

    /// Sets where Anthropic inference runs (Claude 4.6 and later).
    pub fn with_anthropic_inference_geo(mut self, geo: AnthropicInferenceGeo) -> Self {
        self.anthropic_inference_geo = Some(geo);
        self
    }

    /// Requests the 1M-token context window on Anthropic models (see
    /// [`long_context`](Self::long_context)).
    pub fn with_long_context(mut self, enabled: bool) -> Self {
        self.long_context = Some(enabled);
        self
    }

    pub fn with_tool_config(mut self, tool_config: ToolConfig) -> Self {
        self.tool_config = Some(tool_config);
        self
    }

    /// The tool configuration a request sends, `tool_config` when it lists
    /// at least one tool. On every provider, a configuration with no tools
    /// omits the tool list, tool choice and parallel-call setting because
    /// providers refuse them without tools.
    pub fn offered_tools(&self) -> Option<&ToolConfig> {
        self.tool_config
            .as_ref()
            .filter(|tool_config| !tool_config.tools.is_empty())
    }

    /// Enables or disables interleaved thinking for supported Anthropic models.
    pub fn with_interleaved_thinking(mut self, enabled: bool) -> Self {
        self.interleaved_thinking = Some(enabled);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn load(value: serde_json::Value) -> GenerationConfig {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn configs_saved_with_the_temperature_flag_still_load() {
        // What earlier versions wrote for a default config and for one made
        // with `GenerationConfig::new(0.4)` plus a few settings.
        let saved_default = json!({
            "temperature": 0.0, "temperature_specified": false, "max_tokens": null,
            "verbosity": null, "reasoning": null, "openai_reasoning_mode": null,
            "anthropic_thinking_enabled": null, "prompt_caching": null,
            "anthropic_summarized_thinking": null, "long_context": null,
            "tool_config": null, "interleaved_thinking": null
        });
        let config = load(saved_default);
        assert_eq!(config.temperature, None);
        assert_eq!(config.max_tokens, None);

        let saved = json!({
            "temperature": 0.4, "temperature_specified": true, "max_tokens": 2048,
            "verbosity": "low", "reasoning": {"OpenAI": {"effort": "high", "include_summaries": true}},
            "openai_reasoning_mode": null, "anthropic_thinking_enabled": null,
            "prompt_caching": false, "anthropic_summarized_thinking": null,
            "long_context": null, "tool_config": null, "interleaved_thinking": null,
            "openai_service_tier": "flex"
        });
        let config = load(saved);
        assert_eq!(config.temperature, Some(0.4));
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.verbosity, Some(Verbosity::Low));
        assert!(matches!(
            config.reasoning,
            Some(ReasoningConfig::OpenAI {
                include_summaries: true,
                ..
            })
        ));
        assert_eq!(config.prompt_caching, Some(false));
        assert_eq!(config.openai_service_tier, Some(OpenAIServiceTier::Flex));

        // An unspecified temperature is never sent, whatever value it held.
        let config = load(json!({"temperature": 0.7, "temperature_specified": false}));
        assert_eq!(config.temperature, None);
        // The old field defaulted to 0.0.
        let config = load(json!({"temperature_specified": true}));
        assert_eq!(config.temperature, Some(0.0));
    }

    #[test]
    fn a_written_temperature_is_sent() {
        // Before, a hand-written temperature without the flag was dropped.
        assert_eq!(load(json!({"temperature": 0.2})).temperature, Some(0.2));
        assert_eq!(load(json!({"temperature": 1})).temperature, Some(1.0));
        assert_eq!(load(json!({"temperature": null})).temperature, None);
        assert_eq!(load(json!({})).temperature, None);
    }

    #[test]
    fn configs_round_trip_without_the_flag() {
        let default = serde_json::to_value(GenerationConfig::default()).unwrap();
        assert!(default.get("temperature").is_none(), "{default}");
        assert!(default.get("temperature_specified").is_none(), "{default}");

        let config = GenerationConfig::default()
            .with_temperature(0.3)
            .with_max_tokens(100)
            .with_prompt_caching(false);
        let saved = serde_json::to_value(&config).unwrap();
        assert_eq!(saved["temperature"], 0.3);
        assert!(saved.get("temperature_specified").is_none(), "{saved}");
        let restored: GenerationConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.temperature, Some(0.3));
        assert_eq!(restored.max_tokens, Some(100));
        assert_eq!(restored.prompt_caching, Some(false));
    }

    #[test]
    fn the_default_sends_no_temperature() {
        assert_eq!(GenerationConfig::default().temperature, None);
        #[allow(deprecated)]
        let old = GenerationConfig::new(0.5);
        assert_eq!(old.temperature, Some(0.5));
        assert_eq!(
            GenerationConfig::default()
                .with_temperature(0.5)
                .temperature,
            Some(0.5)
        );
    }
}
