use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
}

impl Provider {
    pub fn as_telemetry_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Google => "gemini",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Standard,
    Reasoning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_cost_per_million: f64,
    pub output_cost_per_million: f64,
    pub cached_input_cost_per_million: Option<f64>,
    pub cache_write_cost_per_million: Option<f64>,
    pub large_input_threshold: Option<usize>,
    pub large_input_cost_per_million: Option<f64>,
    pub large_output_cost_per_million: Option<f64>,
    pub large_cached_input_cost_per_million: Option<f64>,
    pub large_cache_write_cost_per_million: Option<f64>,
    pub reasoning_output_cost_per_million: Option<f64>,
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
}

/// Model data audited 2026-09-27 against vendor docs:
/// https://developers.openai.com/api/docs/pricing (+ per-model pages),
/// https://platform.claude.com/docs/en/about-claude/pricing (+ model pages),
/// https://ai.google.dev/gemini-api/docs/pricing (+ model pages).
/// `max_input_tokens` is the vendor's published maximum input tokens where one is
/// given (OpenAI lists it separately for some models), otherwise the context window.
impl Model {
    /// Standard API pricing verified September 6, 2026.
    /// https://developers.openai.com/api/docs/pricing
    pub fn openai_gpt_6_astra() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-6-astra".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Standard API pricing verified September 6, 2026.
    /// Promotional pricing is available at least through November 21, 2026.
    /// https://developers.openai.com/api/docs/pricing
    pub fn openai_gpt_5_6_sol() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.6-sol".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Standard API pricing verified September 6, 2026.
    /// https://developers.openai.com/api/docs/pricing
    pub fn openai_gpt_5_6_terra() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.6-terra".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Standard API pricing verified September 6, 2026.
    /// https://developers.openai.com/api/docs/pricing
    pub fn openai_gpt_5_6_luna() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5.6-luna".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 922_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn openai_gpt_5() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-2025-08-07".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn openai_gpt_5_mini() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-mini-2025-08-07".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn openai_gpt_5_nano() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-5-nano-2025-08-07".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 272_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn openai_o4_mini() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "o4-mini-2025-04-16".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 100_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn openai_o3() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "o3-2025-04-16".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 100_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn openai_o3_pro() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "o3-pro-2025-06-10".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 100_000,
            supports_image_input: true,
            supports_cache: false,
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
            },
        }
    }

    pub fn openai_gpt_4_1_nano() -> Self {
        Self {
            provider: Provider::OpenAI,
            id: "gpt-4.1-nano-2025-04-14".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 1_047_576,
            max_output_tokens: 32_768,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// https://platform.claude.com/docs/en/models/fable-5-1/overview
    pub fn anthropic_claude_fable_5_1() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-fable-5-1".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
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
            },
        }
    }

    /// https://platform.claude.com/docs/en/models/opus-5/overview
    pub fn anthropic_claude_opus_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// $2/$10 was introductory pricing through August 31, 2026 and is now the
    /// standard price; the planned increase did not occur. Checked 2026-09-27 against
    /// https://platform.claude.com/docs/en/about-claude/pricing
    pub fn anthropic_claude_sonnet_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-5".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Retired on the Claude API February 19, 2026; may remain on other cloud platforms.
    /// https://platform.claude.com/docs/en/about-claude/model-deprecations
    pub fn anthropic_claude_3_7_sonnet() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-3-7-sonnet-20250219".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Retired on the Claude API October 28, 2025; may remain on other cloud platforms.
    /// https://platform.claude.com/docs/en/about-claude/model-deprecations
    pub fn anthropic_claude_3_5_sonnet_20241022() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-3-5-sonnet-20241022".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 200_000,
            max_output_tokens: 8192,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Retired on the Claude API October 28, 2025; may remain on other cloud platforms.
    /// https://platform.claude.com/docs/en/about-claude/model-deprecations
    pub fn anthropic_claude_3_5_sonnet_20240620() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-3-5-sonnet-20240620".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 200_000,
            max_output_tokens: 8192,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Retired on the Claude API June 15, 2026; may remain on other cloud platforms.
    /// https://platform.claude.com/docs/en/about-claude/model-deprecations
    pub fn anthropic_claude_opus_4() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-20250514".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 32_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Retired on the Claude API August 5, 2026; may remain on other cloud platforms.
    /// https://platform.claude.com/docs/en/about-claude/model-deprecations
    pub fn anthropic_claude_opus_4_1() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-1-20250805".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 32_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub fn anthropic_claude_opus_4_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-opus-4-5-20251101".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Retired on the Claude API June 15, 2026; may remain on other cloud platforms.
    /// https://platform.claude.com/docs/en/about-claude/model-deprecations
    pub fn anthropic_claude_sonnet_4() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-4-20250514".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(6.00),
                large_output_cost_per_million: Some(22.50),
                large_cached_input_cost_per_million: Some(0.60),
                large_cache_write_cost_per_million: Some(7.50),
                reasoning_output_cost_per_million: None,
            },
        }
    }

    /// Max output corrected to 128K, checked 2026-09-27 against
    /// https://platform.claude.com/docs/en/models/sonnet-4-6/overview
    pub fn anthropic_claude_sonnet_4_6() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-4-6".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_000_000,
            max_output_tokens: 128_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// The long-context rates below applied to the retired 1M-context beta; current
    /// docs list a 200K context window and no long-context tier (checked 2026-09-27,
    /// https://platform.claude.com/docs/en/build-with-claude/context-windows).
    pub fn anthropic_claude_sonnet_4_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-sonnet-4-5-20250929".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
            pricing: ModelPricing {
                input_cost_per_million: 3.00,
                output_cost_per_million: 15.00,
                cached_input_cost_per_million: Some(0.30),
                cache_write_cost_per_million: Some(3.75),
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(6.00),
                large_output_cost_per_million: Some(22.50),
                large_cached_input_cost_per_million: Some(0.60),
                large_cache_write_cost_per_million: Some(7.50),
                reasoning_output_cost_per_million: None,
            },
        }
    }

    pub fn anthropic_claude_haiku_4_5() -> Self {
        Self {
            provider: Provider::Anthropic,
            id: "claude-haiku-4-5-20251001".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 200_000,
            max_output_tokens: 64_000,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Cache-read prices corrected, checked 2026-09-27 against
    /// https://ai.google.dev/gemini-api/docs/pricing
    pub fn google_gemini_2_5_pro() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.5-pro".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Cache-read price corrected, checked 2026-09-27 against
    /// https://ai.google.dev/gemini-api/docs/pricing
    pub fn google_gemini_2_5_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.5-flash".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Cache-read price and token limits corrected, checked 2026-09-27 against
    /// https://ai.google.dev/gemini-api/docs/pricing
    /// https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite
    pub fn google_gemini_2_5_flash_lite() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.5-flash-lite".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Shut down on the Gemini API June 1, 2026. Output limit corrected from
    /// https://ai.google.dev/gemini-api/docs/models/gemini-2.0-flash
    /// https://ai.google.dev/gemini-api/docs/deprecations
    pub fn google_gemini_2_0_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-2.0-flash".to_string(),
            model_type: ModelType::Standard,
            max_input_tokens: 1_048_576,
            max_output_tokens: 8_192,
            supports_image_input: true,
            supports_cache: false,
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
            },
        }
    }

    /// Shut down on the Gemini API March 9, 2026 (replacement: gemini-3.1-pro-preview).
    /// Token limits corrected from the model page; pricing is no longer published.
    /// https://ai.google.dev/gemini-api/docs/deprecations
    pub fn google_gemini_3_0_pro() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3-pro-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
            pricing: ModelPricing {
                input_cost_per_million: 2.00,
                output_cost_per_million: 12.00,
                cached_input_cost_per_million: Some(0.50),
                cache_write_cost_per_million: None,
                large_input_threshold: Some(200_000),
                large_input_cost_per_million: Some(4.00),
                large_output_cost_per_million: Some(18.00),
                large_cached_input_cost_per_million: Some(1.00),
                large_cache_write_cost_per_million: None,
                reasoning_output_cost_per_million: None,
            },
        }
    }

    /// Input limit corrected, checked 2026-09-27 against
    /// https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview
    pub fn google_gemini_3_0_flash() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3-flash-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    /// Shut down on the Gemini API May 25, 2026; the stable replacement is
    /// `gemini-3.1-flash-lite` (same published prices and limits).
    /// https://ai.google.dev/gemini-api/docs/deprecations
    pub fn google_gemini_3_1_flash_lite() -> Self {
        Self {
            provider: Provider::Google,
            id: "gemini-3.1-flash-lite-preview".to_string(),
            model_type: ModelType::Reasoning,
            max_input_tokens: 1_048_576,
            max_output_tokens: 65_536,
            supports_image_input: true,
            supports_cache: true,
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
            },
        }
    }

    pub(crate) fn list_openai_models() -> Vec<Self> {
        vec![
            Self::openai_gpt_6_astra(),
            Self::openai_gpt_5_6_sol(),
            Self::openai_gpt_5_6_terra(),
            Self::openai_gpt_5_6_luna(),
            Self::openai_gpt_5_5(),
            Self::openai_gpt_5_4(),
            Self::openai_gpt_5_4_mini(),
            Self::openai_gpt_5_4_nano(),
            Self::openai_gpt_5_4_pro(),
            Self::openai_gpt_5_3_codex(),
            Self::openai_gpt_5_2(),
            Self::openai_gpt_5_1(),
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
            Self::anthropic_claude_fable_5(),
            Self::anthropic_claude_opus_5(),
            Self::anthropic_claude_opus_4_8(),
            Self::anthropic_claude_sonnet_5(),
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
            Self::google_gemini_3_5_flash(),
            Self::google_gemini_3_1_pro(),
            Self::google_gemini_3_1_pro_customtools(),
            Self::google_gemini_3_1_flash_lite(),
            Self::google_gemini_3_0_pro(),
            Self::google_gemini_3_0_flash(),
            Self::google_gemini_2_5_pro(),
            Self::google_gemini_2_5_flash(),
            Self::google_gemini_2_5_flash_lite(),
            Self::google_gemini_2_0_flash(),
        ]
    }

    pub fn get(provider: Provider, model_id: &str) -> Result<Self, crate::error::Error> {
        let models_to_search = match provider {
            Provider::OpenAI => Self::list_openai_models(),
            Provider::Anthropic => Self::list_anthropic_models(),
            Provider::Google => Self::list_google_models(),
        };

        for model in models_to_search {
            if model.id == model_id {
                return Ok(model);
            }
        }

        Err(crate::error::Error::ModelNotSupported(format!(
            "Model ID '{}' is not supported for provider {:?}.",
            model_id, provider
        )))
    }

    pub fn is_openai_gpt_6_astra(&self) -> bool {
        self.id == "gpt-6-astra"
    }

    pub fn is_openai_gpt_5_6(&self) -> bool {
        matches!(
            self.id.as_str(),
            "gpt-5.6" | "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna"
        )
    }

    pub fn supports_openai_prompt_cache_options(&self) -> bool {
        self.provider == Provider::OpenAI
            && (self.is_openai_gpt_5_6() || self.is_openai_gpt_6_astra())
    }

    pub fn is_fable_5_1(&self) -> bool {
        self.id == "claude-fable-5-1"
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

    pub fn is_sonnet_5(&self) -> bool {
        self.id == "claude-sonnet-5"
    }

    pub fn supports_anthropic_adaptive_thinking(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_opus_4_6()
            || self.is_sonnet_5()
            || self.is_sonnet_4_6()
    }

    pub fn defaults_anthropic_adaptive_thinking(&self) -> bool {
        self.is_fable_5_1() || self.is_fable_5() || self.is_opus_5() || self.is_sonnet_5()
    }

    pub fn always_uses_anthropic_adaptive_thinking(&self) -> bool {
        self.is_fable_5_1() || self.is_fable_5()
    }

    pub fn rejects_anthropic_manual_thinking(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5()
    }

    pub fn omits_anthropic_sampling_parameters(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5()
    }

    pub fn supports_anthropic_summarized_thinking_display(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5()
    }

    pub fn supports_anthropic_effort(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_opus_4_6()
            || self.is_opus_4_5()
            || self.is_sonnet_5()
            || self.is_sonnet_4_6()
    }

    pub fn supports_anthropic_xhigh_effort(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_sonnet_5()
    }

    pub fn supports_anthropic_max_effort(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_opus_4_6()
            || self.is_sonnet_5()
            || self.is_sonnet_4_6()
    }

    pub fn requires_anthropic_effort_beta(&self) -> bool {
        self.is_opus_4_5()
    }

    pub fn has_standard_anthropic_1m_context(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_opus_4_6()
            || self.is_sonnet_5()
            || self.is_sonnet_4_6()
    }

    pub fn requires_long_context_beta(&self) -> bool {
        self.id == "claude-sonnet-4-20250514" || self.id == "claude-sonnet-4-5-20250929"
    }

    pub fn supports_long_context(&self) -> bool {
        self.requires_long_context_beta() || self.has_standard_anthropic_1m_context()
    }

    pub fn supports_interleaved_thinking(&self) -> bool {
        matches!(
            self.id.as_str(),
            "claude-fable-5-1"
                | "claude-fable-5"
                | "claude-opus-5"
                | "claude-opus-4-20250514"
                | "claude-opus-4-1-20250805"
                | "claude-opus-4-5-20251101"
                | "claude-opus-4-6"
                | "claude-opus-4-7"
                | "claude-opus-4-8"
                | "claude-sonnet-4-20250514"
                | "claude-sonnet-4-5-20250929"
                | "claude-sonnet-4-6"
                | "claude-sonnet-5"
        )
    }

    pub fn auto_enables_interleaved_thinking(&self) -> bool {
        self.is_fable_5_1()
            || self.is_fable_5()
            || self.is_opus_5()
            || self.is_opus_4_8()
            || self.is_opus_4_7()
            || self.is_opus_4_6()
            || self.is_sonnet_5()
            || self.is_sonnet_4_6()
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

    pub fn is_gemini_3_1_flash_lite(&self) -> bool {
        self.id == "gemini-3.1-flash-lite-preview"
    }

    pub fn is_gemini_3_5_flash(&self) -> bool {
        self.id == "gemini-3.5-flash"
    }

    pub fn uses_gemini_thinking_levels(&self) -> bool {
        self.is_gemini_3_pro()
            || self.is_gemini_3_flash()
            || self.is_gemini_3_1_pro()
            || self.is_gemini_3_1_flash_lite()
            || self.is_gemini_3_5_flash()
    }
}
