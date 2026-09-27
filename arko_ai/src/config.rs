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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GenerationConfig {
    /// Sent only when `temperature_specified` is set; otherwise the model's
    /// default applies. Google's Interactions API has no temperature, so a
    /// specified one is an error for requests to Google.
    pub temperature: f64,
    pub temperature_specified: bool,
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
    /// Controls prompt caching when the provider exposes a convenient
    /// request-level mechanism.
    ///
    /// `None` inherits [`Conversation::enable_cache`], which defaults to
    /// enabled.
    ///
    /// For Anthropic, the resolved value controls ephemeral prompt caching.
    ///
    /// For GPT-5.6 models, enabled selects implicit prompt caching and disabled
    /// selects explicit-only mode without breakpoints, preventing prompt-cache
    /// reads and writes. Older OpenAI models retain their automatic provider
    /// behavior because they do not accept GPT-5.6 prompt-cache options.
    ///
    /// Gemini implicit caching remains provider-managed and cannot be disabled
    /// through this setting.
    ///
    /// For OpenRouter, enabled sends `prompt_cache_key` and, on Claude models,
    /// a top-level `cache_control`; disabled sends neither, and on OpenAI
    /// models the catalog shows billing cache writes it also sends
    /// `prompt_cache_options.mode: "explicit"`.
    ///
    /// [`Conversation::enable_cache`]: crate::conversation::Conversation::enable_cache
    pub prompt_caching: Option<bool>,
    /// Requests summarized Anthropic thinking text while thinking is enabled
    /// (adaptive or manual budgets).
    ///
    /// Claude Fable 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7 and Sonnet 5
    /// default to omitted thinking text, so this sends `thinking.display:
    /// "summarized"`. On Opus 4.6, Sonnet 4.6 and earlier thinking models
    /// summarized text is already the default and nothing extra is sent.
    /// `None` and `Some(false)` keep the provider default.
    /// https://platform.claude.com/docs/en/build-with-claude/thinking
    pub anthropic_summarized_thinking: Option<bool>,
    /// Requests the 1M-token context window on Anthropic models.
    ///
    /// Claude 4.6 and later models include 1M context at standard pricing, so the
    /// flag is accepted there without a beta header or price change. The
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
    /// OpenAI processing tier (`service_tier`). Flex and Fast must be offered
    /// by the model (see [`ModelPricing::service_tiers`]); API-key auth only.
    /// `None` sends nothing, so the project default applies.
    ///
    /// [`ModelPricing::service_tiers`]: crate::model::ModelPricing::service_tiers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_service_tier: Option<OpenAIServiceTier>,
    /// Anthropic prompt-cache lifetime for the automatic `cache_control` this
    /// client sends when caching is enabled. `None` sends no `ttl`, which is the
    /// provider's 5-minute default. 1-hour writes cost 2x base input (5-minute
    /// writes 1.25x). https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    /// Also the `ttl` of the `cache_control` sent for Claude models on OpenRouter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_cache_ttl: Option<AnthropicCacheTtl>,
    /// Anthropic fast mode (`speed: "fast"` with the `fast-mode-2026-02-01` beta
    /// header; research preview, Claude API only). Supported on Claude Opus 5.5,
    /// Opus 5 and Opus 4.8 and rejected locally for other models. Priced at the
    /// model's fast-mode rates; the response's `usage.speed` decides the price.
    /// `None` and `Some(false)` send nothing.
    /// https://platform.claude.com/docs/en/build-with-claude/fast-mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_fast_mode: Option<bool>,
    /// Anthropic `inference_geo` (where inference runs), for Claude 4.6 and later;
    /// older models reject it. `None` sends nothing, so the workspace default
    /// applies. US-only inference costs 1.1x; the response's `usage.inference_geo`
    /// decides the price. https://platform.claude.com/docs/en/manage-claude/data-residency
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_inference_geo: Option<AnthropicInferenceGeo>,
    /// OpenRouter routing and request options for this request. Fields set
    /// here override the client's defaults
    /// ([`OpenRouterClient::with_default_options`]) field by field; see
    /// [`OpenRouterOptions`]. Ignored by other providers.
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
/// https://developers.openai.com/api/docs/guides/flex-processing
/// https://developers.openai.com/api/docs/guides/fast-mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIServiceTier {
    /// The project's default tier.
    Auto,
    /// Standard processing.
    Default,
    /// Discounted, lower-priority processing.
    Flex,
    /// Fast mode (formerly Priority processing); sent as `priority`.
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

impl GenerationConfig {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            temperature_specified: true,
            max_tokens: None,
            verbosity: None,
            reasoning: None,
            openai_reasoning_mode: None,
            anthropic_thinking_enabled: None,
            prompt_caching: None,
            anthropic_summarized_thinking: None,
            long_context: None,
            tool_config: None,
            interleaved_thinking: None,
            openai_service_tier: None,
            anthropic_cache_ttl: None,
            anthropic_fast_mode: None,
            anthropic_inference_geo: None,
            openrouter: None,
        }
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

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self.temperature_specified = true;
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

    /// Enables or disables interleaved thinking for supported Anthropic models.
    pub fn with_interleaved_thinking(mut self, enabled: bool) -> Self {
        self.interleaved_thinking = Some(enabled);
        self
    }
}
