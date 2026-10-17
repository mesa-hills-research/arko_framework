use crate::pricing::{ResidencyPricing, ScheduledPricing, TierPricing};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    /// OpenRouter's Chat Completions API. Models are dynamic slugs such as
    /// `anthropic/claude-sonnet-5` ([`Model::openrouter`]).
    OpenRouter,
    /// Claude models run through the user's installed Claude Code CLI, under
    /// its login ([`Model::claude_code`], [`crate::claude_code`]). Claude
    /// Code runs its own agent loop, so these models are driven by
    /// arko_agents' Claude Code runtime, not [`LLMClient`](crate::LLMClient)
    /// requests.
    ClaudeCode,
}

impl Provider {
    pub fn as_telemetry_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Google => "gemini",
            Provider::OpenRouter => "openrouter",
            Provider::ClaudeCode => "claude_code",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Standard,
    Reasoning,
}

/// Standard list prices in USD per million tokens, plus documented price
/// variants (see [`crate::pricing`]).
///
/// Token classes without their own price (for example a cached-input price on
/// a model with no caching discount, or cache writes on models that do not
/// charge extra for them) are billed at the input rate. Constructors set the
/// fields they need and take the rest from `..ModelPricing::default()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_cost_per_million: f64,
    pub output_cost_per_million: f64,
    /// Cache reads.
    pub cached_input_cost_per_million: Option<f64>,
    /// Cache writes. For Anthropic this is the 5-minute TTL rate.
    pub cache_write_cost_per_million: Option<f64>,
    /// Input above which the `large_*` rates apply (inclusive of cached and
    /// written input tokens).
    pub large_input_threshold: Option<usize>,
    pub large_input_cost_per_million: Option<f64>,
    pub large_output_cost_per_million: Option<f64>,
    pub large_cached_input_cost_per_million: Option<f64>,
    pub large_cache_write_cost_per_million: Option<f64>,
    /// Unused. Reasoning tokens are billed as output by every provider.
    pub reasoning_output_cost_per_million: Option<f64>,
    /// Anthropic 1-hour TTL cache writes (2x base input).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h_cost_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub large_cache_write_1h_cost_per_million: Option<f64>,
    /// Processing tiers this model offers besides standard, and their prices.
    /// A tier listed here may also be requested (OpenAI `service_tier`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service_tiers: Vec<crate::pricing::TierPricing>,
    /// Data-residency multipliers keyed by the reported region.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub residency: Vec<crate::pricing::ResidencyPricing>,
    /// Dated price changes (for example the end of promotional pricing).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedule: Vec<crate::pricing::ScheduledPricing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub provider: Provider,
    pub id: String,
    pub model_type: ModelType,
    pub max_input_tokens: usize,
    pub max_output_tokens: usize,
    pub supports_image_input: bool,
    pub supports_cache: bool,
    pub pricing: ModelPricing,
    /// OpenRouter catalog metadata, for models built from the OpenRouter
    /// catalog. `None` for registry models and for [`Model::openrouter`]
    /// models created without a catalog.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openrouter: Option<crate::providers::openrouter::OpenRouterModelInfo>,
}

/// Model data comes from vendor docs:
/// <https://developers.openai.com/api/docs/pricing> (+ per-model pages),
/// <https://platform.claude.com/docs/en/about-claude/pricing> (+ model pages),
/// <https://ai.google.dev/gemini-api/docs/pricing> (+ model pages).
/// `max_input_tokens` is the vendor's published maximum input tokens where one is
/// given (OpenAI lists it separately for some models), otherwise the context window.
/// OpenAI publishes no separate maximum input for GPT-5.1, 5.2, 5.4, 5.4-pro, 5.5 and
/// 5.5-pro, so those entries use the context window.
///
/// GPT-5.4/5.5 long-context rates apply "for the full session" per OpenAI's model
/// pages. The docs don't define a session, so the rates are applied per request.
///
/// Price variants, from the same pricing pages:
/// - OpenAI `service_tiers` are the Flex and Fast tables as multipliers of the
///   Standard table. The tables round a few cached prices (for example gpt-5.4
///   Flex cached input $0.13 for 0.5 x $0.25). Fast mode on GPT-5.4/5.5 is
///   listed for short context only. Models without an entry do not offer the tier.
/// - Anthropic 1-hour cache writes cost 2x base input on every model. Fast mode
///   (Opus 5, Opus 4.8) is 2x across the full context window. US-only inference
///   (`inference_geo: "us"`) is 1.1x on Claude 4.6 and later. Claude Haiku 5.5
///   prices prompts over 100,000 tokens higher (the `large_*` rates).
/// - Gemini Flex/Priority require the v1beta endpoint, which this client does
///   not use, so no Gemini tiers are listed.
impl Model {
    /// Standard API pricing.
    /// <https://developers.openai.com/api/docs/pricing>
    pub fn openai_gpt_6_astra() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-6-astra".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 10.00,
                output_cost_per_million: 50.00,
                cached_input_cost_per_million: Some(1.00),
                cache_write_cost_per_million: Some(12.50),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(20.00),
                large_output_cost_per_million: Some(75.00),
                large_cached_input_cost_per_million: Some(2.00),
                large_cache_write_cost_per_million: Some(25.00),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Standard API pricing and limits.
    /// <https://developers.openai.com/api/docs/models/gpt-6-sol>
    pub fn openai_gpt_6_sol() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-6-sol".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: Some(2.50),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(15.00),
                large_cached_input_cost_per_million: Some(0.40),
                large_cache_write_cost_per_million: Some(5.00),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Standard API pricing and limits: cache reads are $0.10, half GPT-6
    /// Sol's. Other rates match it. Long-context cache rates are taken as 2x,
    /// as on GPT-6 Sol.
    /// <https://developers.openai.com/api/docs/models/gpt-6.1-sol>
    pub fn openai_gpt_6_1_sol() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-6.1-sol".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.10),
                cache_write_cost_per_million: Some(2.50),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(15.00),
                large_cached_input_cost_per_million: Some(0.20),
                large_cache_write_cost_per_million: Some(5.00),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Standard API pricing and limits.
    /// <https://developers.openai.com/api/docs/models/gpt-6-luna>
    pub fn openai_gpt_6_luna() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-6-luna".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.10,
                output_cost_per_million: 0.50,
                cached_input_cost_per_million: Some(0.01),
                cache_write_cost_per_million: Some(0.125),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(0.20),
                large_output_cost_per_million: Some(0.75),
                large_cached_input_cost_per_million: Some(0.02),
                large_cache_write_cost_per_million: Some(0.25),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Standard API pricing.
    /// Promotional pricing is available at least through November 21, 2026.
    /// <https://developers.openai.com/api/docs/pricing>
    pub fn openai_gpt_5_6_sol() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.6-sol".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 4.00,
                output_cost_per_million: 20.00,
                cached_input_cost_per_million: Some(0.40),
                cache_write_cost_per_million: Some(5.00),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(8.00),
                large_output_cost_per_million: Some(30.00),
                large_cached_input_cost_per_million: Some(0.80),
                large_cache_write_cost_per_million: Some(10.00),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Standard API pricing.
    /// <https://developers.openai.com/api/docs/pricing>
    pub fn openai_gpt_5_6_terra() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.6-terra".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 12.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: Some(2.50),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(18.00),
                large_cached_input_cost_per_million: Some(0.40),
                large_cache_write_cost_per_million: Some(5.00),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Standard API pricing.
    /// <https://developers.openai.com/api/docs/pricing>
    pub fn openai_gpt_5_6_luna() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.6-luna".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.20,
                output_cost_per_million: 1.20,
                cached_input_cost_per_million: Some(0.02),
                cache_write_cost_per_million: Some(0.25),
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(0.40),
                large_output_cost_per_million: Some(1.80),
                large_cached_input_cost_per_million: Some(0.04),
                large_cache_write_cost_per_million: Some(0.50),
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// The `gpt-5.6` alias routes requests to `gpt-5.6-sol` (same prices and limits).
    /// Responses report the served model.
    /// <https://developers.openai.com/api/docs/models/gpt-5.6-sol>
    pub fn openai_gpt_5_6() -> Self {
        Self {
            id: "gpt-5.6".to_string(),
            ..Self::openai_gpt_5_6_sol()
        }
    }

    /// Shuts down December 11, 2026 (replacement: gpt-5.6-sol).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_gpt_5() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-2025-08-07".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.25,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.125),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Defaults to, and only supports, `reasoning.effort: high`. No cached-input
    /// discount. Responses and Batch only. Shuts down December 11, 2026
    /// (replacement: gpt-5.6-sol with `reasoning.mode: "pro"`).
    /// <https://developers.openai.com/api/docs/models/gpt-5-pro>
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_gpt_5_pro() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-pro-2025-10-06".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 400_000,
            max_output_tokens: 272_000,
            supports_image_input: true,
            supports_cache: false,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 15.00,
                output_cost_per_million: 120.00,
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_1() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.1-2025-11-13".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 400_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.25,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.125),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_2() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.2".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 400_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.75,
                output_cost_per_million: 14.00,
                cached_input_cost_per_million: Some(0.175),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Reasoning effort medium, high or xhigh. No cached-input discount. Responses
    /// only. No Flex or Fast mode.
    /// <https://developers.openai.com/api/docs/models/gpt-5.2-pro>
    /// <https://developers.openai.com/api/docs/pricing>
    pub fn openai_gpt_5_2_pro() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.2-pro".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 400_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: false,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 21.00,
                output_cost_per_million: 168.00,
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_3_codex() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.3-codex".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.75,
                output_cost_per_million: 14.00,
                cached_input_cost_per_million: Some(0.175),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_4() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.4-2026-03-05".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_050_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.50,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.25),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(5.00),
                large_output_cost_per_million: Some(22.50),
                large_cached_input_cost_per_million: Some(0.50),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![
                    TierPricing::fast(2.0).short_context_only(),
                    TierPricing::flex(0.5),
                ],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_4_pro() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.4-pro-2026-03-05".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_050_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: false,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 30.00,
                output_cost_per_million: 180.00,
                cached_input_cost_per_million: None,
                cache_write_cost_per_million: None,
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(60.00),
                large_output_cost_per_million: Some(270.00),
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_5() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.5-2026-04-23".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_050_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 5.00,
                output_cost_per_million: 30.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(10.00),
                large_output_cost_per_million: Some(45.00),
                large_cached_input_cost_per_million: Some(1.00),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![
                    TierPricing::fast(2.5).short_context_only(),
                    TierPricing::flex(0.5),
                ],
                ..ModelPricing::default()
            },
        }
    }

    /// Reasoning effort medium, high (default) or xhigh. No cached-input discount.
    /// Responses and Batch only. Flex (Batch rates) for short context only, no Fast
    /// mode.
    /// <https://developers.openai.com/api/docs/models/gpt-5.5-pro>
    /// <https://developers.openai.com/api/docs/pricing>
    pub fn openai_gpt_5_5_pro() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.5-pro-2026-04-23".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_050_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: false,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 30.00,
                output_cost_per_million: 180.00,
                large_input_threshold: Some(272_000),
                large_input_cost_per_million: Some(60.00),
                large_output_cost_per_million: Some(270.00),
                service_tiers: vec![TierPricing::flex(0.5).short_context_only()],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_4_mini() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.4-mini-2026-03-17".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.75,
                output_cost_per_million: 4.50,
                cached_input_cost_per_million: Some(0.075),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_5_4_nano() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.4-nano-2026-03-17".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.20,
                output_cost_per_million: 1.25,
                cached_input_cost_per_million: Some(0.02),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Shuts down December 11, 2026 (replacement: gpt-5.6-terra).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_gpt_5_mini() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-mini-2025-08-07".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.25,
                output_cost_per_million: 2.00,
                cached_input_cost_per_million: Some(0.025),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(1.8), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Shuts down December 11, 2026 (replacement: gpt-5.6-luna).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_gpt_5_nano() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-nano-2025-08-07".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.05,
                output_cost_per_million: 0.40,
                cached_input_cost_per_million: Some(0.005),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Shuts down October 23, 2026 (replacement: gpt-5.6-terra).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_o4_mini() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "o4-mini-2025-04-16".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 100_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.10,
                output_cost_per_million: 4.40,
                cached_input_cost_per_million: Some(0.275),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(20.0 / 11.0), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Shuts down December 11, 2026 (replacement: gpt-5.6-sol).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_o3() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "o3-2025-04-16".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 100_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 8.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(1.75), TierPricing::flex(0.5)],
                ..ModelPricing::default()
            },
        }
    }

    /// Shuts down December 11, 2026 (replacement: gpt-5.6-sol with
    /// `reasoning.mode: "pro"`).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_o3_pro() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "o3-pro-2025-06-10".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 100_000,
            supports_image_input: true,
            supports_cache: false,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 20.00,
                output_cost_per_million: 80.00,
                cached_input_cost_per_million: None,
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_4_1() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-4.1-2025-04-14".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 1_047_576,
            max_output_tokens: 32_768,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 8.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(1.75)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn openai_gpt_4_1_mini() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-4.1-mini-2025-04-14".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 1_047_576,
            max_output_tokens: 32_768,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.40,
                output_cost_per_million: 1.60,
                cached_input_cost_per_million: Some(0.10),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(1.75)],
                ..ModelPricing::default()
            },
        }
    }

    /// Shuts down October 23, 2026 (replacement: gpt-5.6-luna).
    /// <https://developers.openai.com/api/docs/deprecations>
    pub fn openai_gpt_4_1_nano() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-4.1-nano-2025-04-14".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 1_047_576,
            max_output_tokens: 32_768,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.10,
                output_cost_per_million: 0.40,
                cached_input_cost_per_million: Some(0.025),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                service_tiers: vec![TierPricing::fast(2.0)],
                ..ModelPricing::default()
            },
        }
    }

    /// <https://platform.claude.com/docs/en/models/fable-5-1/overview>
    pub fn anthropic_claude_fable_5_1() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-fable-5-1".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 10.00,
                output_cost_per_million: 50.00,
                cached_input_cost_per_million: Some(0.25),
                cache_write_cost_per_million: Some(12.50),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(20.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Claude Fable 5.1 for organizations verified through one of Anthropic's
    /// verification programs, such as the Cyber Verification Program: the same
    /// model, limits and prices, so every Fable 5.1 rule applies to it too
    /// ([`Model::is_fable_or_mythos_5_1`]). Released 2026-09-01, retirement not
    /// sooner than 2027-09-01.
    /// <https://platform.claude.com/docs/en/models/mythos-5-1/overview>
    pub fn anthropic_claude_mythos_5_1() -> Self {
        Self {
            id: "claude-mythos-5-1".to_string(),
            ..Self::anthropic_claude_fable_5_1()
        }
    }

    pub fn anthropic_claude_fable_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-fable-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 10.00,
                output_cost_per_million: 50.00,
                cached_input_cost_per_million: Some(1.00),
                cache_write_cost_per_million: Some(12.50),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(20.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Released 2026-09-22. Retirement not sooner than 2027-09-22. Adaptive thinking
    /// is always on (default effort `medium`). `thinking` disabled or manual budgets,
    /// non-default sampling parameters, prefill and forced tool choice are rejected.
    /// Cache reads are 0.05x input. Fast mode is $8/$40 (2x).
    /// <https://platform.claude.com/docs/en/models/opus-5-5/overview>
    /// <https://platform.claude.com/docs/en/about-claude/pricing>
    /// <https://platform.claude.com/docs/en/build-with-claude/fast-mode>
    pub fn anthropic_claude_opus_5_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-5-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 4.00,
                output_cost_per_million: 20.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: Some(5.00),
                cache_write_1h_cost_per_million: Some(8.00),
                service_tiers: vec![TierPricing::fast(2.0)],
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// <https://platform.claude.com/docs/en/models/opus-5/overview>
    pub fn anthropic_claude_opus_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 5.00,
                output_cost_per_million: 25.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: Some(6.25),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(10.00),
                service_tiers: vec![TierPricing::fast(2.0)],
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn anthropic_claude_opus_4_8() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-8".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 5.00,
                output_cost_per_million: 25.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: Some(6.25),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(10.00),
                service_tiers: vec![TierPricing::fast(2.0)],
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Same prices as Sonnet 5 except cache reads, which cost $0.10 (0.05x input)
    /// since 2026-10-07, half the earlier $0.20. Adaptive thinking is on by default
    /// (default effort `high`, levels recalibrated from Sonnet 5). `thinking.type:
    /// "disabled"`, manual budgets, non-default sampling parameters, prefill and
    /// forced tool choice are rejected. No fast mode.
    /// <https://platform.claude.com/docs/en/about-claude/pricing>
    /// <https://www.anthropic.com/claude-haiku-5-5>
    pub fn anthropic_claude_sonnet_5_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-5-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.10),
                cache_write_cost_per_million: Some(2.50),
                cache_write_1h_cost_per_million: Some(4.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// $2/$10 is the standard price, the same as the introductory rate that ended
    /// August 31, 2026.
    /// <https://platform.claude.com/docs/en/about-claude/pricing>
    pub fn anthropic_claude_sonnet_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: Some(2.50),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(4.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Released 2026-10-07, retirement not sooner than 2027-10-07. Adaptive
    /// thinking is on by default (default effort `medium`, all five levels).
    /// `thinking.type: "disabled"` is accepted at `high` effort or below, and arko
    /// sends it only without an Anthropic reasoning config, so at the default
    /// effort. Manual budgets, non-default sampling parameters and prefill are
    /// rejected. A forced tool choice is accepted, and the response then starts
    /// with the tool call, without thinking. No fast mode.
    /// Prices depend on the prompt size: a prompt over 100,000 tokens (counting
    /// cached and written input) pays 5x on every token class.
    /// <https://platform.claude.com/docs/en/models/haiku-5-5/overview>
    /// <https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide>
    /// <https://platform.claude.com/docs/en/about-claude/pricing>
    pub fn anthropic_claude_haiku_5_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-haiku-5-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.10,
                output_cost_per_million: 0.50,
                cached_input_cost_per_million: Some(0.01),
                cache_write_cost_per_million: Some(0.125),
                cache_write_1h_cost_per_million: Some(0.20),
                large_input_threshold: Some(100_000),
                large_input_cost_per_million: Some(0.50),
                large_output_cost_per_million: Some(2.50),
                large_cached_input_cost_per_million: Some(0.05),
                large_cache_write_cost_per_million: Some(0.625),
                large_cache_write_1h_cost_per_million: Some(1.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Retired on the Claude API February 19, 2026 (requests return an error).
    /// Partner platforms set their own retirement dates. Max output is 64K: 128K
    /// required the `output-128k-2025-02-19` beta header, which this client never
    /// sends.
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_3_7_sonnet() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-3-7-sonnet-20250219".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(6.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Retired on the Claude API October 28, 2025. May remain on other cloud platforms.
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_3_5_sonnet_20241022() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-3-5-sonnet-20241022".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 200_000,
            max_output_tokens: 8192,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(6.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Retired on the Claude API October 28, 2025. May remain on other cloud platforms.
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_3_5_sonnet_20240620() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-3-5-sonnet-20240620".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 200_000,
            max_output_tokens: 8192,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(6.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Retired on the Claude API June 15, 2026. May remain on other cloud platforms.
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_opus_4() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-20250514".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 32_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 15.00,
                output_cost_per_million: 75.00,
                cached_input_cost_per_million: Some(1.50),
                cache_write_cost_per_million: Some(18.75),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(30.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Retired on the Claude API August 5, 2026. May remain on other cloud platforms.
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_opus_4_1() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-1-20250805".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 32_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 15.00,
                output_cost_per_million: 75.00,
                cached_input_cost_per_million: Some(1.50),
                cache_write_cost_per_million: Some(18.75),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(30.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Retirement not sooner than November 24, 2026 (no deprecation notice published).
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_opus_4_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-5-20251101".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 5.00,
                output_cost_per_million: 25.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: Some(6.25),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(10.00),
                ..ModelPricing::default()
            },
        }
    }

    pub fn anthropic_claude_opus_4_6() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-6".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 5.00,
                output_cost_per_million: 25.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: Some(6.25),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(10.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    pub fn anthropic_claude_opus_4_7() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-7".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 5.00,
                output_cost_per_million: 25.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: Some(6.25),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(10.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Retired on the Claude API June 15, 2026. May remain on other cloud platforms.
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    ///
    /// The 1M-context beta (`context-1m-2025-08-07`) was retired for Sonnet 4 and
    /// 4.5 on 2026-04-30: the header has no effect and requests over 200K tokens
    /// return an error. While it was available, input above 200K was priced at
    /// $6 input / $22.50 output, with caching multipliers on top ($0.60 reads, $7.50
    /// 5-minute and $12 1-hour writes). <https://platform.claude.com/docs/en/release-notes/overview>
    pub fn anthropic_claude_sonnet_4() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-4-20250514".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                cache_write_1h_cost_per_million: Some(6.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Max output is 128K.
    /// <https://platform.claude.com/docs/en/models/sonnet-4-6/overview>
    pub fn anthropic_claude_sonnet_4_6() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-4-6".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(6.00),
                residency: vec![ResidencyPricing::new("us", 1.1)],
                ..ModelPricing::default()
            },
        }
    }

    /// Retirement not sooner than September 29, 2026 (no deprecation notice published).
    /// The 1M-context beta (`context-1m-2025-08-07`) was retired for this model on
    /// 2026-04-30: the header has no effect and requests over 200K tokens return an
    /// error, so `long_context` is rejected locally. While it was available, input
    /// above 200K was priced at $6 input / $22.50 output, with caching multipliers
    /// on top ($0.60 reads, $7.50 5-minute and $12 1-hour writes).
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    /// <https://platform.claude.com/docs/en/build-with-claude/context-windows>
    /// <https://platform.claude.com/docs/en/release-notes/overview>
    pub fn anthropic_claude_sonnet_4_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-4-5-20250929".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                cache_write_1h_cost_per_million: Some(6.00),
                ..ModelPricing::default()
            },
        }
    }

    /// Retirement not sooner than October 15, 2026 (no deprecation notice published).
    /// <https://platform.claude.com/docs/en/about-claude/model-deprecations>
    pub fn anthropic_claude_haiku_4_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-haiku-4-5-20251001".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.00,
                output_cost_per_million: 5.00,
                cached_input_cost_per_million: Some(0.10),
                cache_write_cost_per_million: Some(1.25),
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                cache_write_1h_cost_per_million: Some(2.00),
                ..ModelPricing::default()
            },
        }
    }

    pub fn google_gemini_3_5_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3.5-flash".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.50,
                output_cost_per_million: 9.00,
                cached_input_cost_per_million: Some(0.15),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Since 2026-09-18 access is limited to existing users.
    /// <https://ai.google.dev/gemini-api/docs/pricing>
    /// <https://ai.google.dev/gemini-api/docs/models>
    pub fn google_gemini_2_5_pro() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.5-pro".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 1.25,
                output_cost_per_million: 10.00,
                cached_input_cost_per_million: Some(0.125),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(2.50),
                large_output_cost_per_million: Some(15.00),
                large_cached_input_cost_per_million: Some(0.25),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Since 2026-09-18 access is limited to existing users.
    /// <https://ai.google.dev/gemini-api/docs/pricing>
    /// <https://ai.google.dev/gemini-api/docs/models>
    pub fn google_gemini_2_5_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.5-flash".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.30,
                output_cost_per_million: 2.50,
                cached_input_cost_per_million: Some(0.03),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Since 2026-09-18 access is limited to existing users.
    /// <https://ai.google.dev/gemini-api/docs/pricing>
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite>
    pub fn google_gemini_2_5_flash_lite() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.5-flash-lite".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.10,
                output_cost_per_million: 0.40,
                cached_input_cost_per_million: Some(0.01),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Shut down on the Gemini API June 1, 2026.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-2.0-flash>
    /// <https://ai.google.dev/gemini-api/docs/deprecations>
    /// `supports_cache` is false: the Interactions API offers only implicit caching,
    /// which starts with Gemini 2.5 (<https://ai.google.dev/gemini-api/docs/caching>).
    pub fn google_gemini_2_0_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.0-flash".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 1_048_576,
            max_output_tokens: 8_192,
            supports_image_input: true,
            supports_cache: false,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.10,
                output_cost_per_million: 0.40,
                cached_input_cost_per_million: None,
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Shut down on the Gemini API March 9, 2026 (replacement: gemini-3.1-pro-preview).
    /// The changelog entry for that date also says the id "now points to"
    /// gemini-3.1-pro-preview. Both statements are recorded because the docs conflict.
    /// The current pricing page omits this model. Its cached-input prices ($0.20, or
    /// $0.40 above 200K) come from the launch-day and 2026-02-01 archived pricing pages.
    /// <https://ai.google.dev/gemini-api/docs/deprecations>
    /// <https://ai.google.dev/gemini-api/docs/changelog>
    /// <https://web.archive.org/web/20251118161504/https://ai.google.dev/gemini-api/docs/pricing>
    pub fn google_gemini_3_0_pro() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3-pro-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 12.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(18.00),
                large_cached_input_cost_per_million: Some(0.40),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Legacy model (replacement: gemini-3.6-flash). No shutdown date published.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview>
    /// <https://ai.google.dev/gemini-api/docs/deprecations>
    pub fn google_gemini_3_0_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3-flash-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.50,
                output_cost_per_million: 3.00,
                cached_input_cost_per_million: Some(0.05),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    pub fn google_gemini_3_1_pro() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3.1-pro-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 12.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(18.00),
                large_cached_input_cost_per_million: Some(0.40),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    pub fn google_gemini_3_1_pro_customtools() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3.1-pro-preview-customtools".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 12.00,
                cached_input_cost_per_million: Some(0.20),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(18.00),
                large_cached_input_cost_per_million: Some(0.40),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Shut down on the Gemini API May 25, 2026. The stable replacement is
    /// `gemini-3.1-flash-lite` (same published prices and limits).
    /// <https://ai.google.dev/gemini-api/docs/deprecations>
    pub fn google_gemini_3_1_flash_lite() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3.1-flash-lite-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing: ModelPricing {
                input_cost_per_million: 0.25,
                output_cost_per_million: 1.50,
                cached_input_cost_per_million: Some(0.025),
                cache_write_cost_per_million: None,
                large_input_threshold: None,
                large_input_cost_per_million: None,
                large_output_cost_per_million: None,
                large_cached_input_cost_per_million: None,
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
                ..ModelPricing::default()
            },
        }
    }

    /// Shared limits for Gemini 3.1+ Flash and Flash-Lite models: 1,048,576 input and
    /// 65,536 output tokens, image input, implicit caching, no >200K price tier.
    fn google_gemini_flash(id: &str, pricing: ModelPricing) -> Self {
        Self {
            provider: Provider::Google,
            id: id.to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            openrouter: None,
            pricing,
        }
    }

    /// Introductory $0.75/$3.75 (cached $0.075) through 2026-12-31, then $1.50/$7.50
    /// (cached $0.15) from 2027-01-01, for Gemini 3.6, 3.7 and 3.8 Flash.
    /// https://ai.google.dev/gemini-api/docs/pricing
    fn google_gemini_3_6_to_3_8_flash_pricing() -> ModelPricing {
        ModelPricing {
            input_cost_per_million: 0.75,
            output_cost_per_million: 3.75,
            cached_input_cost_per_million: Some(0.075),
            schedule: vec![ScheduledPricing::new(
                "2027-01-01",
                ModelPricing {
                    input_cost_per_million: 1.50,
                    output_cost_per_million: 7.50,
                    cached_input_cost_per_million: Some(0.15),
                    ..ModelPricing::default()
                },
            )],
            ..ModelPricing::default()
        }
    }

    /// GA 2026-09-02. Thinking levels low, medium (default) and high. `minimal`
    /// returns an error.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash>
    /// <https://ai.google.dev/gemini-api/docs/thinking>
    pub fn google_gemini_3_8_flash() -> Self {
        Self::google_gemini_flash(
            "gemini-3.8-flash",
            Self::google_gemini_3_6_to_3_8_flash_pricing(),
        )
    }

    /// GA 2026-08-13. Thinking levels low, medium (default) and high. `minimal`
    /// returns an error.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash>
    /// <https://ai.google.dev/gemini-api/docs/thinking>
    pub fn google_gemini_3_7_flash() -> Self {
        Self::google_gemini_flash(
            "gemini-3.7-flash",
            Self::google_gemini_3_6_to_3_8_flash_pricing(),
        )
    }

    /// GA 2026-07-21. Thinking levels minimal through high. Default medium.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash>
    /// <https://ai.google.dev/gemini-api/docs/thinking>
    pub fn google_gemini_3_6_flash() -> Self {
        Self::google_gemini_flash(
            "gemini-3.6-flash",
            Self::google_gemini_3_6_to_3_8_flash_pricing(),
        )
    }

    /// GA 2026-07-21. Thinking levels minimal (default) through high.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite>
    /// <https://ai.google.dev/gemini-api/docs/pricing>
    pub fn google_gemini_3_5_flash_lite() -> Self {
        Self::google_gemini_flash(
            "gemini-3.5-flash-lite",
            ModelPricing {
                input_cost_per_million: 0.30,
                output_cost_per_million: 2.50,
                cached_input_cost_per_million: Some(0.03),
                ..ModelPricing::default()
            },
        )
    }

    /// Stable release (GA 2026-05-07) of the retired preview. Earliest shutdown
    /// 2027-05-07 (replacement: gemini-3.5-flash-lite). Thinking levels minimal
    /// (default) through high.
    /// <https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite>
    /// <https://ai.google.dev/gemini-api/docs/pricing>
    /// <https://ai.google.dev/gemini-api/docs/deprecations>
    pub fn google_gemini_3_1_flash_lite_stable() -> Self {
        Self::google_gemini_flash(
            "gemini-3.1-flash-lite",
            ModelPricing {
                input_cost_per_million: 0.25,
                output_cost_per_million: 1.50,
                cached_input_cost_per_million: Some(0.025),
                ..ModelPricing::default()
            },
        )
    }

    pub(crate) fn list_openai_models() -> Vec<Self> {
        vec![
            Self::openai_gpt_6_astra(),
            Self::openai_gpt_6_1_sol(),
            Self::openai_gpt_6_sol(),
            Self::openai_gpt_6_luna(),
            Self::openai_gpt_5_6_sol(),
            Self::openai_gpt_5_6_terra(),
            Self::openai_gpt_5_6_luna(),
            Self::openai_gpt_5_6(),
            Self::openai_gpt_5_5_pro(),
            Self::openai_gpt_5_5(),
            Self::openai_gpt_5_4(),
            Self::openai_gpt_5_4_mini(),
            Self::openai_gpt_5_4_nano(),
            Self::openai_gpt_5_4_pro(),
            Self::openai_gpt_5_3_codex(),
            Self::openai_gpt_5_2_pro(),
            Self::openai_gpt_5_2(),
            Self::openai_gpt_5_1(),
            Self::openai_gpt_5_pro(),
            Self::openai_gpt_5(),
            Self::openai_gpt_5_mini(),
            Self::openai_gpt_5_nano(),
            Self::openai_o4_mini(),
            Self::openai_o3(),
            Self::openai_o3_pro(),
            Self::openai_gpt_4_1(),
            Self::openai_gpt_4_1_mini(),
            Self::openai_gpt_4_1_nano(),
        ]
    }

    pub(crate) fn list_anthropic_models() -> Vec<Self> {
        vec![
            Self::anthropic_claude_fable_5_1(),
            Self::anthropic_claude_mythos_5_1(),
            Self::anthropic_claude_fable_5(),
            Self::anthropic_claude_opus_5_5(),
            Self::anthropic_claude_opus_5(),
            Self::anthropic_claude_opus_4_8(),
            Self::anthropic_claude_sonnet_5_5(),
            Self::anthropic_claude_sonnet_5(),
            Self::anthropic_claude_haiku_5_5(),
            Self::anthropic_claude_opus_4_7(),
            Self::anthropic_claude_opus_4_6(),
            Self::anthropic_claude_opus_4_5(),
            Self::anthropic_claude_opus_4_1(),
            Self::anthropic_claude_opus_4(),
            Self::anthropic_claude_sonnet_4_6(),
            Self::anthropic_claude_sonnet_4_5(),
            Self::anthropic_claude_sonnet_4(),
            Self::anthropic_claude_haiku_4_5(),
            Self::anthropic_claude_3_7_sonnet(),
            Self::anthropic_claude_3_5_sonnet_20241022(),
            Self::anthropic_claude_3_5_sonnet_20240620(),
        ]
    }

    pub(crate) fn list_google_models() -> Vec<Self> {
        vec![
            Self::google_gemini_3_8_flash(),
            Self::google_gemini_3_7_flash(),
            Self::google_gemini_3_6_flash(),
            Self::google_gemini_3_5_flash(),
            Self::google_gemini_3_5_flash_lite(),
            Self::google_gemini_3_1_pro(),
            Self::google_gemini_3_1_pro_customtools(),
            Self::google_gemini_3_1_flash_lite_stable(),
            Self::google_gemini_3_1_flash_lite(),
            Self::google_gemini_3_0_pro(),
            Self::google_gemini_3_0_flash(),
            Self::google_gemini_2_5_pro(),
            Self::google_gemini_2_5_flash(),
            Self::google_gemini_2_5_flash_lite(),
            Self::google_gemini_2_0_flash(),
        ]
    }

    /// An OpenRouter model by slug, for example `anthropic/claude-sonnet-5`,
    /// `openrouter/auto`, `~anthropic/claude-sonnet-latest`,
    /// `openai/gpt-6-luna:floor` or `@preset/name`.
    ///
    /// No network call is made and nothing is known about the model: limits
    /// are 0 (unknown), prices are 0 (costs come from OpenRouter's reported
    /// `usage.cost`), and image input and caching are assumed so requests pass
    /// through unchanged. For limits, prices and request validation, load the
    /// catalog with [`OpenRouterClient::fetch_models`] and use
    /// [`OpenRouterClient::model`].
    ///
    /// [`OpenRouterClient::fetch_models`]: crate::providers::openrouter::OpenRouterClient::fetch_models
    /// [`OpenRouterClient::model`]: crate::providers::openrouter::OpenRouterClient::model
    pub fn openrouter(slug: impl Into<String>) -> Self {
        Self {
            provider: Provider::OpenRouter,
            id: slug.into(),
            model_type: ModelType::Standard,
            max_input_tokens: 0,
            max_output_tokens: 0,
            supports_image_input: true,
            supports_cache: true,
            pricing: ModelPricing::default(),
            openrouter: None,
        }
    }

    /// Every registered model for `provider`. OpenRouter models are dynamic
    /// and not registered (see [`Model::openrouter`]), so its list is empty.
    pub fn list(provider: Provider) -> Vec<Self> {
        match provider {
            Provider::OpenAI => Self::list_openai_models(),
            Provider::Anthropic => Self::list_anthropic_models(),
            Provider::Google => Self::list_google_models(),
            Provider::OpenRouter => Vec::new(),
            Provider::ClaudeCode => Self::list_anthropic_models()
                .into_iter()
                .map(Self::claude_code)
                .collect(),
        }
    }

    /// `model` (a registered Anthropic model) run through the user's Claude
    /// Code CLI: the same id, limits and list prices (for estimates, calls
    /// under a Claude subscription are billed nothing per call), with
    /// [`Provider::ClaudeCode`].
    pub fn claude_code(model: Model) -> Self {
        Self {
            provider: Provider::ClaudeCode,
            ..model
        }
    }

    /// Every registered model, across providers. Claude Code models share
    /// their ids with the Anthropic models and are listed only by
    /// [`Model::list`].
    pub fn all() -> Vec<Self> {
        [Provider::OpenAI, Provider::Anthropic, Provider::Google]
            .into_iter()
            .flat_map(Self::list)
            .collect()
    }

    /// The registered model with this id, from any provider. Ids are unique
    /// across the registry.
    pub fn find(model_id: &str) -> Option<Self> {
        Self::all().into_iter().find(|model| model.id == model_id)
    }

    /// The registered model with this id. OpenRouter models are not
    /// registered, so this fails for [`Provider::OpenRouter`]. Use
    /// [`Model::openrouter`] or the client's catalog instead.
    pub fn get(provider: Provider, model_id: &str) -> Result<Self, crate::error::Error> {
        if provider == Provider::OpenRouter {
            return Err(crate::error::Error::ModelNotSupported(format!(
                "OpenRouter model '{model_id}' is not in the static registry; OpenRouter models are dynamic (use Model::openrouter or OpenRouterClient::model)."
            )));
        }
        Self::list(provider.clone())
            .into_iter()
            .find(|model| model.id == model_id)
            .ok_or_else(|| {
                crate::error::Error::ModelNotSupported(format!(
                    "Model ID '{}' is not supported for provider {:?}.",
                    model_id, provider
                ))
            })
    }

    /// Whether a forced tool choice ([`ToolChoice::Required`] or
    /// [`ToolChoice::Specific`]) is accepted for a request with `config`.
    /// Anthropic rejects it on some models and, on every model, while manual
    /// extended thinking (`budget_tokens`) is enabled. The same rules apply to
    /// Claude models on OpenRouter, where OpenRouter reasoning on a Claude
    /// model without adaptive thinking, or a reasoning `max_tokens` budget,
    /// means manual thinking. Other OpenRouter models need `tool_choice` in
    /// their catalog parameters when catalog metadata is present.
    ///
    /// [`ToolChoice::Required`]: crate::tool::ToolChoice::Required
    /// [`ToolChoice::Specific`]: crate::tool::ToolChoice::Specific
    pub fn supports_forced_tool_choice(&self, config: &crate::config::GenerationConfig) -> bool {
        match self.provider {
            // Claude Code chooses its own tool calls.
            Provider::ClaudeCode => false,
            Provider::Anthropic => {
                let manual_thinking = config.anthropic_thinking_enabled != Some(false)
                    && matches!(
                        config.reasoning,
                        Some(crate::reasoning::ReasoningConfig::Anthropic { budget_tokens, .. })
                            if budget_tokens > 0
                    );
                !self.rejects_anthropic_forced_tool_choice() && !manual_thinking
            }
            Provider::OpenAI | Provider::Google => true,
            Provider::OpenRouter => {
                let listed = self
                    .openrouter
                    .as_ref()
                    .is_none_or(|info| info.supports_parameter("tool_choice"));
                let claude_allows = self.openrouter_claude_equivalent().is_none_or(|claude| {
                    !claude.rejects_anthropic_forced_tool_choice()
                        && !Self::openrouter_claude_manual_thinking(&claude, config)
                });
                listed && claude_allows
            }
        }
    }

    /// The registered Claude model an OpenRouter `anthropic/` slug (or a
    /// catalog alias's target) names, for Anthropic's model rules:
    /// `anthropic/claude-sonnet-4.5` is `claude-sonnet-4-5-20250929`.
    fn openrouter_claude_equivalent(&self) -> Option<Model> {
        let slug = self
            .openrouter
            .as_ref()
            .and_then(|info| info.alias_target.as_deref())
            .unwrap_or(&self.id);
        let name = slug.trim_start_matches('~').strip_prefix("anthropic/")?;
        // Strip a variant (`:floor`) and a preset (`@preset/x`) in either order.
        let name = name.split([':', '@']).next()?.replace('.', "-");
        let dated = format!("{name}-2");
        Self::list_anthropic_models()
            .into_iter()
            .find(|model| model.id == name || model.id.starts_with(&dated))
    }

    /// Whether OpenRouter reasoning in `config` runs Claude with a manual
    /// thinking budget.
    fn openrouter_claude_manual_thinking(
        claude: &Model,
        config: &crate::config::GenerationConfig,
    ) -> bool {
        let Some(crate::reasoning::ReasoningConfig::OpenRouter(reasoning)) = &config.reasoning
        else {
            return false;
        };
        if reasoning.enabled == Some(false)
            || reasoning.effort == Some(crate::reasoning::ReasoningEffort::None)
        {
            return false;
        }
        !claude.supports_anthropic_adaptive_thinking()
            || (reasoning.max_tokens.is_some() && !claude.rejects_anthropic_manual_thinking())
    }

    /// Whether `ToolConfig::with_parallel_tool_calls(false)` is accepted. The
    /// Gemini Interactions API has no switch to disable parallel calls.
    /// OpenRouter forwards `parallel_tool_calls` to the serving provider.
    pub fn supports_disabling_parallel_tool_calls(&self) -> bool {
        match self.provider {
            Provider::OpenAI | Provider::Anthropic | Provider::OpenRouter => true,
            Provider::Google | Provider::ClaudeCode => false,
        }
    }

    /// Whether responses carry a normalized
    /// [`StopReason`](crate::stop::StopReason) and the provider's raw stop
    /// metadata, so callers can tell complete turns from truncated or refused
    /// ones.
    pub fn reports_stop_reason(&self) -> bool {
        match self.provider {
            Provider::OpenAI
            | Provider::Anthropic
            | Provider::Google
            | Provider::OpenRouter
            | Provider::ClaudeCode => true,
        }
    }

    /// OpenRouter catalog metadata, when this model came from the catalog.
    pub fn openrouter_info(&self) -> Option<&crate::providers::openrouter::OpenRouterModelInfo> {
        self.openrouter.as_ref()
    }

    pub fn is_openai_gpt_6_astra(&self) -> bool {
        self.id == "gpt-6-astra"
    }

    /// GPT-6 Sol, GPT-6.1 Sol and GPT-6 Luna share GPT-5.6's reasoning
    /// controls. Unlike Astra, they accept `none`.
    pub fn is_openai_gpt_6_sol_or_luna(&self) -> bool {
        matches!(self.id.as_str(), "gpt-6-sol" | "gpt-6.1-sol" | "gpt-6-luna")
    }

    pub fn is_openai_gpt_5_6(&self) -> bool {
        matches!(
            self.id.as_str(),
            "gpt-5.6" | "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna"
        )
    }

    /// Whether images can go inside a tool result for this model
    /// ([`ToolResult::images`]): OpenAI (API key and ChatGPT OAuth) and
    /// Anthropic models, and Gemini 3 and later, whose `function_result` takes
    /// image content ("For Gemini 3 series models",
    /// <https://ai.google.dev/gemini-api/docs/function-calling#multimodal>).
    /// Gemini 2.x rejects images in function responses, and OpenRouter
    /// doesn't document how each upstream receives them, so those models,
    /// Gemini ids whose version can't be read, and models without image
    /// input get `false`: their result images follow the turn's results.
    ///
    /// [`ToolResult::images`]: crate::tool::ToolResult::images
    pub fn accepts_tool_result_images(&self) -> bool {
        self.supports_image_input
            && match self.provider {
                Provider::OpenAI | Provider::Anthropic => true,
                Provider::Google => gemini_major_version(&self.id).is_some_and(|major| major >= 3),
                // Its in-process tools return text.
                Provider::OpenRouter | Provider::ClaudeCode => false,
            }
    }

    /// Why `image` can't go to this model in a user message, or `None` when
    /// it can. A model without image input takes none. Anthropic and Google
    /// models take an image as data, not by URL (their requests refuse a
    /// URL), and Claude Code would pass a URL on as text. Google's image
    /// formats leave out GIF
    /// (<https://ai.google.dev/gemini-api/docs/image-understanding#supported-image-formats>).
    pub fn image_refusal(&self, image: &crate::message::Image) -> Option<String> {
        use crate::message::Image;
        if !self.supports_image_input {
            return Some(format!("model '{}' doesn't take images", self.id));
        }
        match image {
            Image::Url(_)
                if matches!(
                    self.provider,
                    Provider::Anthropic | Provider::Google | Provider::ClaudeCode
                ) =>
            {
                Some(format!(
                    "model '{}' takes images as data, not by URL",
                    self.id
                ))
            }
            Image::Base64 { mime_type, .. }
                if self.provider == Provider::Google
                    && mime_type.trim().eq_ignore_ascii_case("image/gif") =>
            {
                Some(format!(
                    "model '{}' doesn't take GIF images: convert it to PNG or JPEG",
                    self.id
                ))
            }
            _ => None,
        }
    }

    pub fn supports_openai_prompt_cache_options(&self) -> bool {
        self.provider == Provider::OpenAI
            && (self.is_openai_gpt_5_6()
                || self.is_openai_gpt_6_astra()
                || self.is_openai_gpt_6_sol_or_luna())
    }

    pub fn is_openai_gpt_5_5_pro(&self) -> bool {
        self.id.starts_with("gpt-5.5-pro")
    }

    pub fn is_openai_gpt_5_2_pro(&self) -> bool {
        self.id.starts_with("gpt-5.2-pro")
    }

    pub fn is_openai_gpt_5_pro(&self) -> bool {
        self.id.starts_with("gpt-5-pro")
    }

    pub fn is_fable_5_1(&self) -> bool {
        self.id == "claude-fable-5-1"
    }

    pub fn is_mythos_5_1(&self) -> bool {
        self.id == "claude-mythos-5-1"
    }

    /// Claude Fable 5.1 or Claude Mythos 5.1, the same model under two ids,
    /// so every Fable 5.1 rule applies to both.
    pub fn is_fable_or_mythos_5_1(&self) -> bool {
        self.is_fable_5_1() || self.is_mythos_5_1()
    }

    pub fn is_opus_5_5(&self) -> bool {
        self.id == "claude-opus-5-5"
    }

    pub fn is_opus_5(&self) -> bool {
        self.id == "claude-opus-5"
    }

    pub fn is_fable_5(&self) -> bool {
        self.id == "claude-fable-5"
    }

    pub fn is_opus_4_5(&self) -> bool {
        self.id == "claude-opus-4-5-20251101"
    }

    pub fn is_opus_4_6(&self) -> bool {
        self.id == "claude-opus-4-6"
    }

    pub fn is_opus_4_7(&self) -> bool {
        self.id == "claude-opus-4-7"
    }

    pub fn is_opus_4_8(&self) -> bool {
        self.id == "claude-opus-4-8"
    }

    pub fn is_sonnet_4_6(&self) -> bool {
        self.id == "claude-sonnet-4-6"
    }

    pub fn is_sonnet_5_5(&self) -> bool {
        self.id == "claude-sonnet-5-5"
    }

    pub fn is_sonnet_5(&self) -> bool {
        self.id == "claude-sonnet-5"
    }

    pub fn is_haiku_5_5(&self) -> bool {
        self.id == "claude-haiku-5-5"
    }

    /// Claude models whose 1M context window is standard (no beta header), which
    /// is also the "Claude 4.6 and later" set used by data residency.
    fn is_anthropic_4_6_or_later(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_opus_4_6()
            || self.is_sonnet_5_5()
            || self.is_sonnet_5()
            || self.is_sonnet_4_6()
            || self.is_haiku_5_5()
    }

    /// Thinking predicates follow
    /// <https://platform.claude.com/docs/en/build-with-claude/thinking> and
    /// <https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>
    pub fn supports_anthropic_adaptive_thinking(&self) -> bool {
        self.is_anthropic_4_6_or_later()
    }

    pub fn defaults_anthropic_adaptive_thinking(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_opus_5()
            || self.is_sonnet_5_5()
            || self.is_sonnet_5()
            || self.is_haiku_5_5()
    }

    /// Adaptive thinking can't be disabled (`thinking.type: "disabled"` returns 400).
    /// Sonnet 5.5's lowest setting, `thinking.type: "between_tools"`, is not sent
    /// by this client. Opus 5 and Haiku 5.5 accept `"disabled"` at `high` effort or
    /// below, and this client sends it only without an Anthropic reasoning config,
    /// so at the model's default effort.
    pub fn always_uses_anthropic_adaptive_thinking(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_sonnet_5_5()
    }

    pub fn rejects_anthropic_manual_thinking(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5_5()
            || self.is_sonnet_5()
            || self.is_haiku_5_5()
    }

    /// Non-default `temperature`, `top_p` or `top_k` return 400 on these models.
    pub fn omits_anthropic_sampling_parameters(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5_5()
            || self.is_sonnet_5()
            || self.is_haiku_5_5()
    }

    /// Models whose thinking display defaults to `"omitted"`, so summarized text must
    /// be requested with `thinking.display`. On Opus 4.6, Sonnet 4.6 and earlier
    /// thinking models summarized text is already the default.
    pub fn supports_anthropic_summarized_thinking_display(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5_5()
            || self.is_sonnet_5()
            || self.is_haiku_5_5()
    }

    /// Effort predicates follow
    /// <https://platform.claude.com/docs/en/build-with-claude/effort>
    pub fn supports_anthropic_effort(&self) -> bool {
        self.is_anthropic_4_6_or_later() || self.is_opus_4_5()
    }

    pub fn supports_anthropic_xhigh_effort(&self) -> bool {
        self.is_fable_or_mythos_5_1()
            || self.is_fable_5()
            || self.is_opus_5_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5_5()
            || self.is_sonnet_5()
            || self.is_haiku_5_5()
    }

    pub fn supports_anthropic_max_effort(&self) -> bool {
        self.is_anthropic_4_6_or_later()
    }

    /// Forced `tool_choice` (`any` or a named tool) returns 400 on every request to
    /// these models. It is also rejected on any model while manual extended thinking
    /// is enabled.
    /// <https://platform.claude.com/docs/en/build-with-claude/thinking>
    pub fn rejects_anthropic_forced_tool_choice(&self) -> bool {
        self.is_fable_or_mythos_5_1() || self.is_opus_5_5() || self.is_sonnet_5_5()
    }

    /// Fast mode (`speed: "fast"`, research preview): Opus 5.5, Opus 5 and Opus 4.8
    /// on the Claude API. Opus 4.7 rejects it. Opus 4.6 silently runs at standard
    /// speed.
    /// <https://platform.claude.com/docs/en/build-with-claude/fast-mode>
    pub fn supports_anthropic_fast_mode(&self) -> bool {
        self.is_opus_5_5() || self.is_opus_5() || self.is_opus_4_8()
    }

    /// `inference_geo` is accepted on Claude 4.6 and later. Older models return 400.
    /// <https://platform.claude.com/docs/en/manage-claude/data-residency>
    pub fn supports_anthropic_inference_geo(&self) -> bool {
        self.is_anthropic_4_6_or_later()
    }

    pub fn has_standard_anthropic_1m_context(&self) -> bool {
        self.is_anthropic_4_6_or_later()
    }

    /// Sonnet 4 and 4.5 reached 1M context through the `context-1m-2025-08-07` beta,
    /// retired on 2026-04-30 (the header now has no effect and requests over 200K
    /// tokens return an error). <https://platform.claude.com/docs/en/release-notes/overview>
    pub fn had_retired_anthropic_long_context_beta(&self) -> bool {
        self.id == "claude-sonnet-4-20250514" || self.id == "claude-sonnet-4-5-20250929"
    }

    pub fn supports_long_context(&self) -> bool {
        self.has_standard_anthropic_1m_context()
    }

    /// Claude 4.6 and later models, and the earlier Claude 4 models listed here.
    pub fn supports_interleaved_thinking(&self) -> bool {
        self.is_anthropic_4_6_or_later()
            || matches!(
                self.id.as_str(),
                "claude-opus-4-20250514"
                    | "claude-opus-4-1-20250805"
                    | "claude-opus-4-5-20251101"
                    | "claude-sonnet-4-20250514"
                    | "claude-sonnet-4-5-20250929"
            )
    }

    /// Interleaved thinking is automatic on every model that supports adaptive
    /// thinking, whenever adaptive thinking is active.
    pub fn auto_enables_interleaved_thinking(&self) -> bool {
        self.is_anthropic_4_6_or_later()
    }

    pub fn is_gemini_3_pro(&self) -> bool {
        self.id == "gemini-3-pro-preview"
    }

    pub fn is_gemini_3_flash(&self) -> bool {
        self.id == "gemini-3-flash-preview"
    }

    pub fn is_gemini_3_1_pro(&self) -> bool {
        self.id.starts_with("gemini-3.1-pro-preview")
    }

    /// The stable `gemini-3.1-flash-lite` and its retired preview.
    pub fn is_gemini_3_1_flash_lite(&self) -> bool {
        matches!(
            self.id.as_str(),
            "gemini-3.1-flash-lite" | "gemini-3.1-flash-lite-preview"
        )
    }

    pub fn is_gemini_3_5_flash(&self) -> bool {
        self.id == "gemini-3.5-flash"
    }

    pub fn is_gemini_3_5_flash_lite(&self) -> bool {
        self.id == "gemini-3.5-flash-lite"
    }

    pub fn is_gemini_3_6_flash(&self) -> bool {
        self.id == "gemini-3.6-flash"
    }

    pub fn is_gemini_3_7_flash(&self) -> bool {
        self.id == "gemini-3.7-flash"
    }

    pub fn is_gemini_3_8_flash(&self) -> bool {
        self.id == "gemini-3.8-flash"
    }

    pub fn is_gemini_2_5_pro(&self) -> bool {
        self.id == "gemini-2.5-pro"
    }

    pub fn is_gemini_2_5_flash(&self) -> bool {
        self.id == "gemini-2.5-flash"
    }

    pub fn is_gemini_2_5_flash_lite(&self) -> bool {
        self.id == "gemini-2.5-flash-lite"
    }

    pub fn is_gemini_2_5(&self) -> bool {
        self.is_gemini_2_5_pro() || self.is_gemini_2_5_flash() || self.is_gemini_2_5_flash_lite()
    }

    pub fn uses_gemini_thinking_levels(&self) -> bool {
        self.is_gemini_3_pro()
            || self.is_gemini_3_flash()
            || self.is_gemini_3_1_pro()
            || self.is_gemini_3_1_flash_lite()
            || self.is_gemini_3_5_flash()
            || self.is_gemini_3_5_flash_lite()
            || self.is_gemini_3_6_flash()
            || self.is_gemini_3_7_flash()
            || self.is_gemini_3_8_flash()
    }
}

/// The major version of a `gemini-<major>…` model id (`gemini-3.5-flash` → 3,
/// `gemini-3-pro-preview` → 3). `None` for ids of another form.
fn gemini_major_version(id: &str) -> Option<u32> {
    let rest = id.strip_prefix("gemini-")?;
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    match rest[digits..].chars().next() {
        None | Some('.' | '-') if digits > 0 => rest[..digits].parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_images_are_accepted_by_openai_anthropic_and_gemini_3_and_later() {
        for model in Model::list(Provider::OpenAI)
            .into_iter()
            .chain(Model::list(Provider::Anthropic))
        {
            assert!(model.accepts_tool_result_images(), "{}", model.id);
        }
        for model in Model::list(Provider::Google) {
            let gemini_3 = model.id.starts_with("gemini-3");
            assert_eq!(model.accepts_tool_result_images(), gemini_3, "{}", model.id);
        }
        let with_id = |id: &str| {
            let mut model = Model::google_gemini_3_5_flash();
            model.id = id.into();
            model.accepts_tool_result_images()
        };
        assert!(with_id("gemini-4-flash"));
        assert!(with_id("gemini-4.0-pro-preview"));
        assert!(with_id("gemini-10.1-flash"));
        assert!(!with_id("gemini-2.5-flash"));
        assert!(!with_id("gemini-exp-1206"));
        assert!(!with_id("gemini-3x"));
        assert!(!with_id("learnlm-2.0-flash"));
        assert!(!Model::openrouter("anthropic/claude-opus-5.5").accepts_tool_result_images());
        let mut blind = Model::openai_gpt_6_astra();
        blind.supports_image_input = false;
        assert!(!blind.accepts_tool_result_images());
    }
    use crate::{config::GenerationConfig, reasoning::ReasoningConfig};
    use std::collections::HashSet;

    #[test]
    fn registry_ids_are_unique_and_lookups_agree() {
        let all = Model::all();
        let ids: HashSet<_> = all.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids.len(), all.len(), "duplicate model ids");
        for model in &all {
            assert_eq!(Model::find(&model.id).unwrap().provider, model.provider);
            assert!(Model::get(model.provider.clone(), &model.id).is_ok());
            assert!(
                Model::list(model.provider.clone())
                    .iter()
                    .any(|m| m.id == model.id)
            );
        }
        assert!(Model::find("not-a-model").is_none());
        assert!(Model::get(Provider::OpenAI, "claude-opus-5-5").is_err());
    }

    #[test]
    fn every_model_has_sane_limits_and_prices() {
        for model in Model::all() {
            let id = &model.id;
            assert!(model.max_input_tokens >= 100_000, "{id} input limit");
            assert!(model.max_output_tokens >= 8_192, "{id} output limit");
            let pricing = &model.pricing;
            assert!(pricing.input_cost_per_million > 0.0, "{id} input price");
            assert!(
                pricing.output_cost_per_million >= pricing.input_cost_per_million,
                "{id} output price"
            );
            if let Some(cached) = pricing.cached_input_cost_per_million {
                assert!(
                    cached < pricing.input_cost_per_million,
                    "{id} cache reads cost less"
                );
            }
            if let Some(write) = pricing.cache_write_cost_per_million {
                assert!(
                    write >= pricing.input_cost_per_million,
                    "{id} cache writes cost more"
                );
            }
            if model.supports_cache && model.provider != Provider::Google {
                assert!(
                    pricing.cached_input_cost_per_million.is_some(),
                    "{id} cache read price"
                );
            }
            if let Some(threshold) = pricing.large_input_threshold {
                assert!(
                    threshold < model.max_input_tokens,
                    "{id} long-context threshold"
                );
                assert!(
                    pricing.large_input_cost_per_million.is_some(),
                    "{id} long-context price"
                );
            }
            for period in &pricing.schedule {
                assert!(period.effective_from_unix().is_some(), "{id} schedule date");
            }
            if model.provider == Provider::Anthropic {
                assert!(
                    pricing.cache_write_1h_cost_per_million.is_some(),
                    "{id} 1h writes"
                );
            }
        }
    }

    /// Cache reads cost 0.025x input on Fable 5.1 and Mythos 5.1, 0.05x on
    /// Opus 5.5 and Sonnet 5.5, and 0.1x on every other Claude model.
    /// https://platform.claude.com/docs/en/about-claude/pricing#prompt-caching
    #[test]
    fn claude_cache_reads_follow_the_documented_multipliers() {
        for model in Model::list(Provider::Anthropic) {
            let pricing = &model.pricing;
            let expected = match model.id.as_str() {
                "claude-fable-5-1" | "claude-mythos-5-1" => 0.025,
                "claude-opus-5-5" | "claude-sonnet-5-5" => 0.05,
                _ => 0.1,
            };
            let ratio =
                pricing.cached_input_cost_per_million.unwrap() / pricing.input_cost_per_million;
            assert!((ratio - expected).abs() < 1e-9, "{}: {ratio}", model.id);
            if let (Some(large), Some(threshold)) = (
                pricing.large_cached_input_cost_per_million,
                pricing.large_input_threshold,
            ) {
                let large_ratio = large / pricing.large_input_cost_per_million.unwrap();
                assert!(
                    (large_ratio - expected).abs() < 1e-9,
                    "{} over {threshold}: {large_ratio}",
                    model.id
                );
            }
        }
    }

    #[test]
    fn new_models_are_covered_by_their_feature_predicates() {
        let opus = Model::anthropic_claude_opus_5_5();
        assert!(opus.supports_anthropic_adaptive_thinking());
        assert!(opus.defaults_anthropic_adaptive_thinking());
        assert!(opus.always_uses_anthropic_adaptive_thinking());
        assert!(opus.rejects_anthropic_manual_thinking());
        assert!(opus.omits_anthropic_sampling_parameters());
        assert!(opus.supports_anthropic_summarized_thinking_display());
        assert!(opus.supports_anthropic_effort());
        assert!(opus.supports_anthropic_xhigh_effort());
        assert!(opus.supports_anthropic_max_effort());
        assert!(opus.rejects_anthropic_forced_tool_choice());
        assert!(opus.supports_anthropic_fast_mode());
        assert!(opus.supports_anthropic_inference_geo());
        assert!(opus.has_standard_anthropic_1m_context());
        assert!(opus.supports_interleaved_thinking());
        assert!(opus.auto_enables_interleaved_thinking());

        // Mythos 5.1 is Fable 5.1 under another id: every rule, limit and
        // price matches.
        let rules = |model: &Model| {
            [
                model.supports_anthropic_adaptive_thinking(),
                model.defaults_anthropic_adaptive_thinking(),
                model.always_uses_anthropic_adaptive_thinking(),
                model.rejects_anthropic_manual_thinking(),
                model.omits_anthropic_sampling_parameters(),
                model.supports_anthropic_summarized_thinking_display(),
                model.supports_anthropic_effort(),
                model.supports_anthropic_xhigh_effort(),
                model.supports_anthropic_max_effort(),
                model.rejects_anthropic_forced_tool_choice(),
                model.supports_anthropic_fast_mode(),
                model.supports_anthropic_inference_geo(),
                model.has_standard_anthropic_1m_context(),
                model.supports_interleaved_thinking(),
                model.auto_enables_interleaved_thinking(),
                model.supports_forced_tool_choice(&GenerationConfig::default()),
            ]
        };
        let fable = Model::anthropic_claude_fable_5_1();
        let mythos = Model::anthropic_claude_mythos_5_1();
        assert!(mythos.is_mythos_5_1() && mythos.is_fable_or_mythos_5_1());
        assert!(fable.is_fable_or_mythos_5_1() && !mythos.is_fable_5_1());
        assert_eq!(rules(&mythos), rules(&fable));
        assert_eq!(
            serde_json::to_value(&mythos.pricing).unwrap(),
            serde_json::to_value(&fable.pricing).unwrap()
        );
        assert_eq!(
            (mythos.max_input_tokens, mythos.max_output_tokens),
            (fable.max_input_tokens, fable.max_output_tokens)
        );

        let sonnet = Model::anthropic_claude_sonnet_5_5();
        assert!(sonnet.is_sonnet_5_5());
        assert!(!sonnet.is_sonnet_5());
        assert!(sonnet.supports_anthropic_adaptive_thinking());
        assert!(sonnet.defaults_anthropic_adaptive_thinking());
        assert!(sonnet.always_uses_anthropic_adaptive_thinking());
        assert!(sonnet.rejects_anthropic_manual_thinking());
        assert!(sonnet.omits_anthropic_sampling_parameters());
        assert!(sonnet.supports_anthropic_summarized_thinking_display());
        assert!(sonnet.supports_anthropic_effort());
        assert!(sonnet.supports_anthropic_xhigh_effort());
        assert!(sonnet.supports_anthropic_max_effort());
        assert!(sonnet.rejects_anthropic_forced_tool_choice());
        assert!(!sonnet.supports_anthropic_fast_mode());
        assert!(sonnet.supports_anthropic_inference_geo());
        assert!(sonnet.has_standard_anthropic_1m_context());
        assert!(sonnet.supports_interleaved_thinking());
        assert!(sonnet.auto_enables_interleaved_thinking());
        assert_eq!(
            Model::openrouter("anthropic/claude-sonnet-5.5")
                .openrouter_claude_equivalent()
                .map(|model| model.id),
            Some("claude-sonnet-5-5".to_string())
        );

        let haiku = Model::anthropic_claude_haiku_5_5();
        assert!(haiku.is_haiku_5_5());
        assert!(haiku.supports_anthropic_adaptive_thinking());
        assert!(haiku.defaults_anthropic_adaptive_thinking());
        // It accepts `thinking.type: "disabled"` at `high` effort or below.
        assert!(!haiku.always_uses_anthropic_adaptive_thinking());
        assert!(haiku.rejects_anthropic_manual_thinking());
        assert!(haiku.omits_anthropic_sampling_parameters());
        assert!(haiku.supports_anthropic_summarized_thinking_display());
        assert!(haiku.supports_anthropic_effort());
        assert!(haiku.supports_anthropic_xhigh_effort());
        assert!(haiku.supports_anthropic_max_effort());
        assert!(!haiku.rejects_anthropic_forced_tool_choice());
        assert!(!haiku.supports_anthropic_fast_mode());
        assert!(haiku.supports_anthropic_inference_geo());
        assert!(haiku.has_standard_anthropic_1m_context());
        assert!(haiku.supports_interleaved_thinking());
        assert!(haiku.auto_enables_interleaved_thinking());
        for (slug, id) in [
            ("anthropic/claude-haiku-5.5", "claude-haiku-5-5"),
            ("anthropic/claude-haiku-4.5", "claude-haiku-4-5-20251001"),
            ("anthropic/claude-mythos-5.1", "claude-mythos-5-1"),
        ] {
            assert_eq!(
                Model::openrouter(slug)
                    .openrouter_claude_equivalent()
                    .map(|model| model.id),
                Some(id.to_string())
            );
        }
        for id in ["claude-mythos-5-1", "claude-haiku-5-5"] {
            let model = Model::get(Provider::ClaudeCode, id).unwrap();
            assert_eq!(model.provider, Provider::ClaudeCode);
            assert_eq!(model.max_input_tokens, 1_000_000);
        }

        for model in [
            Model::google_gemini_3_8_flash(),
            Model::google_gemini_3_7_flash(),
            Model::google_gemini_3_6_flash(),
            Model::google_gemini_3_5_flash_lite(),
            Model::google_gemini_3_1_flash_lite_stable(),
        ] {
            assert!(model.uses_gemini_thinking_levels(), "{}", model.id);
        }
        assert!(Model::google_gemini_3_1_flash_lite_stable().is_gemini_3_1_flash_lite());

        for model in [
            Model::openai_gpt_5_5_pro(),
            Model::openai_gpt_5_2_pro(),
            Model::openai_gpt_5_pro(),
        ] {
            assert!(!model.supports_cache, "{}", model.id);
            assert!(model.pricing.cached_input_cost_per_million.is_none());
        }
        assert!(Model::openai_gpt_5_5_pro().is_openai_gpt_5_5_pro());
        assert!(!Model::openai_gpt_5_5().is_openai_gpt_5_5_pro());
        assert!(Model::openai_gpt_5_pro().is_openai_gpt_5_pro());
        assert!(!Model::openai_gpt_5().is_openai_gpt_5_pro());
    }

    #[test]
    fn retired_and_deprecated_models_stay_registered() {
        for id in [
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "gemini-3-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.0-flash",
            "gpt-5-pro-2025-10-06",
            "o3-2025-04-16",
            "o4-mini-2025-04-16",
            "gpt-4.1-nano-2025-04-14",
        ] {
            assert!(Model::find(id).is_some(), "{id}");
        }
    }

    /// The doc comment directly above the constructor that registers `id`.
    fn constructor_doc(id: &str) -> String {
        let source = include_str!("model.rs");
        let at = source
            .find(&format!("id: \"{id}\""))
            .unwrap_or_else(|| panic!("{id} constructor"));
        let lines: Vec<&str> = source[..at].lines().collect();
        let function = lines
            .iter()
            .rposition(|line| line.trim_start().starts_with("pub fn "))
            .unwrap();
        lines[..function]
            .iter()
            .rev()
            .take_while(|line| line.trim_start().starts_with("///"))
            .copied()
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn deprecated_models_document_their_retirement() {
        for id in [
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "gemini-3-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.0-flash",
            "gpt-5-pro-2025-10-06",
            "gpt-5-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "o3-2025-04-16",
            "o3-pro-2025-06-10",
            "o4-mini-2025-04-16",
            "gpt-4.1-nano-2025-04-14",
        ] {
            let doc = constructor_doc(id).to_lowercase();
            assert!(
                ["retired", "shut down", "shuts down", "retirement"]
                    .iter()
                    .any(|word| doc.contains(word)),
                "{id} lacks a retirement note: {doc}"
            );
        }
    }

    #[test]
    fn capability_flags_describe_provider_limits() {
        let default = GenerationConfig::default();
        let manual = GenerationConfig::default().with_reasoning(ReasoningConfig::anthropic(4096));
        assert!(Model::anthropic_claude_sonnet_4_5().supports_forced_tool_choice(&default));
        assert!(!Model::anthropic_claude_sonnet_4_5().supports_forced_tool_choice(&manual));
        assert!(
            Model::anthropic_claude_sonnet_4_5().supports_forced_tool_choice(
                &manual.clone().with_anthropic_thinking_enabled(false)
            )
        );
        assert!(!Model::anthropic_claude_fable_5_1().supports_forced_tool_choice(&default));
        assert!(!Model::anthropic_claude_opus_5_5().supports_forced_tool_choice(&default));
        assert!(!Model::anthropic_claude_sonnet_5_5().supports_forced_tool_choice(&default));
        assert!(!Model::anthropic_claude_mythos_5_1().supports_forced_tool_choice(&default));
        assert!(Model::anthropic_claude_haiku_5_5().supports_forced_tool_choice(&default));
        assert!(Model::anthropic_claude_sonnet_5().supports_forced_tool_choice(&default));
        assert!(Model::openai_gpt_6_astra().supports_forced_tool_choice(&default));
        assert!(Model::google_gemini_3_5_flash().supports_forced_tool_choice(&default));

        assert!(!Model::google_gemini_3_5_flash().supports_disabling_parallel_tool_calls());
        assert!(Model::openai_gpt_6_astra().supports_disabling_parallel_tool_calls());
        assert!(Model::anthropic_claude_opus_5().supports_disabling_parallel_tool_calls());

        assert!(Model::all().iter().all(Model::reports_stop_reason));
    }
}
