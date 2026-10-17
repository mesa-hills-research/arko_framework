//! Settings meant for another provider.
//!
//! One `GenerationConfig` can be shared by agents on different providers, or
//! kept when an agent moves to another model. A setting the provider in use
//! has no equivalent for is dropped with a warning (logged once per provider
//! and setting), on every provider: an OpenAI service tier on Claude, an
//! Anthropic thinking option on Gemini, another provider's
//! `ReasoningConfig` variant. A setting the provider does take but whose
//! value it can't accept (an effort the model lacks, a temperature on an
//! OpenAI reasoning model) is still an error, raised where the request is
//! built.

use crate::{
    config::GenerationConfig,
    model::{Model, Provider},
    notice,
    providers::openrouter::is_claude_family,
    reasoning::ReasoningConfig,
};
use std::borrow::Cow;

/// A setting that belongs to some providers only.
struct Setting {
    name: &'static str,
    is_set: fn(&GenerationConfig) -> bool,
    clear: fn(&mut GenerationConfig),
    /// Where the setting applies.
    applies_to: &'static str,
}

static VERBOSITY: Setting = Setting {
    name: "verbosity",
    is_set: |config| config.verbosity.is_some(),
    clear: |config| config.verbosity = None,
    applies_to: "OpenAI GPT-5 and GPT-6 models and OpenRouter",
};
static OPENAI_REASONING_MODE: Setting = Setting {
    name: "openai_reasoning_mode",
    is_set: |config| config.openai_reasoning_mode.is_some(),
    clear: |config| config.openai_reasoning_mode = None,
    applies_to: "OpenAI GPT-5.6 and GPT-6 models (on OpenRouter, set OpenRouterReasoning::mode)",
};
static OPENAI_SERVICE_TIER: Setting = Setting {
    name: "openai_service_tier",
    is_set: |config| config.openai_service_tier.is_some(),
    clear: |config| config.openai_service_tier = None,
    applies_to: "the OpenAI provider with an API key (on OpenRouter, set OpenRouterOptions::service_tier)",
};
static ANTHROPIC_THINKING_ENABLED: Setting = Setting {
    name: "anthropic_thinking_enabled",
    is_set: |config| config.anthropic_thinking_enabled.is_some(),
    clear: |config| config.anthropic_thinking_enabled = None,
    applies_to: "the Anthropic provider (on OpenRouter, use OpenRouterReasoning::enabled or ::disabled)",
};
static ANTHROPIC_SUMMARIZED_THINKING: Setting = Setting {
    name: "anthropic_summarized_thinking",
    is_set: |config| config.anthropic_summarized_thinking.is_some(),
    clear: |config| config.anthropic_summarized_thinking = None,
    applies_to: "Claude models",
};
static ANTHROPIC_CACHE_TTL: Setting = Setting {
    name: "anthropic_cache_ttl",
    is_set: |config| config.anthropic_cache_ttl.is_some(),
    clear: |config| config.anthropic_cache_ttl = None,
    applies_to: "Claude models, directly or through OpenRouter",
};
static ANTHROPIC_FAST_MODE: Setting = Setting {
    name: "anthropic_fast_mode",
    is_set: |config| config.anthropic_fast_mode.is_some(),
    clear: |config| config.anthropic_fast_mode = None,
    applies_to: "the Anthropic provider",
};
static ANTHROPIC_INFERENCE_GEO: Setting = Setting {
    name: "anthropic_inference_geo",
    is_set: |config| config.anthropic_inference_geo.is_some(),
    clear: |config| config.anthropic_inference_geo = None,
    applies_to: "the Anthropic provider (on OpenRouter, use provider preferences)",
};
static LONG_CONTEXT: Setting = Setting {
    name: "long_context",
    is_set: |config| config.long_context.is_some(),
    clear: |config| config.long_context = None,
    applies_to: "Claude models",
};
static INTERLEAVED_THINKING: Setting = Setting {
    name: "interleaved_thinking",
    is_set: |config| config.interleaved_thinking.is_some(),
    clear: |config| config.interleaved_thinking = None,
    applies_to: "Claude models, directly or through OpenRouter",
};
static OPENROUTER: Setting = Setting {
    name: "openrouter",
    is_set: |config| config.openrouter.is_some(),
    clear: |config| config.openrouter = None,
    applies_to: "OpenRouter models",
};

/// The settings `model`'s provider has no use for.
fn foreign_settings(model: &Model) -> Vec<&'static Setting> {
    match model.provider {
        Provider::OpenAI => vec![
            &ANTHROPIC_THINKING_ENABLED,
            &ANTHROPIC_SUMMARIZED_THINKING,
            &ANTHROPIC_CACHE_TTL,
            &ANTHROPIC_FAST_MODE,
            &ANTHROPIC_INFERENCE_GEO,
            &LONG_CONTEXT,
            &INTERLEAVED_THINKING,
            &OPENROUTER,
        ],
        Provider::Anthropic => vec![
            &VERBOSITY,
            &OPENAI_REASONING_MODE,
            &OPENAI_SERVICE_TIER,
            &OPENROUTER,
        ],
        Provider::Google => vec![
            &VERBOSITY,
            &OPENAI_REASONING_MODE,
            &OPENAI_SERVICE_TIER,
            &ANTHROPIC_THINKING_ENABLED,
            &ANTHROPIC_SUMMARIZED_THINKING,
            &ANTHROPIC_CACHE_TTL,
            &ANTHROPIC_FAST_MODE,
            &ANTHROPIC_INFERENCE_GEO,
            &LONG_CONTEXT,
            &INTERLEAVED_THINKING,
            &OPENROUTER,
        ],
        // Claude Code takes the reasoning effort, fast mode and the cache
        // TTL; it decides the rest itself.
        Provider::ClaudeCode => vec![
            &VERBOSITY,
            &OPENAI_REASONING_MODE,
            &OPENAI_SERVICE_TIER,
            &ANTHROPIC_THINKING_ENABLED,
            &ANTHROPIC_SUMMARIZED_THINKING,
            &ANTHROPIC_INFERENCE_GEO,
            &LONG_CONTEXT,
            &INTERLEAVED_THINKING,
            &OPENROUTER,
        ],
        Provider::OpenRouter => {
            let mut settings = vec![
                &OPENAI_REASONING_MODE,
                &OPENAI_SERVICE_TIER,
                &ANTHROPIC_THINKING_ENABLED,
                &ANTHROPIC_FAST_MODE,
                &ANTHROPIC_INFERENCE_GEO,
            ];
            // Claude through OpenRouter takes a cache TTL and interleaved
            // thinking; summarized thinking and long context are what
            // OpenRouter already does for it.
            if !is_claude_family(&model.id) {
                settings.extend([
                    &ANTHROPIC_SUMMARIZED_THINKING,
                    &ANTHROPIC_CACHE_TTL,
                    &LONG_CONTEXT,
                    &INTERLEAVED_THINKING,
                ]);
            }
            settings
        }
    }
}

/// The `ReasoningConfig` variants `provider` takes.
fn takes_reasoning(provider: &Provider, reasoning: &ReasoningConfig) -> bool {
    match provider {
        Provider::ClaudeCode => reasoning.provider() == Provider::Anthropic,
        provider => reasoning.provider() == *provider,
    }
}

fn reasoning_hint(provider: &Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "OpenAI models take ReasoningConfig::OpenAI",
        Provider::Anthropic => {
            "Anthropic models take ReasoningConfig::Anthropic or ::AnthropicAdaptive"
        }
        Provider::Google => "Gemini models take ReasoningConfig::Gemini or ::Google",
        Provider::OpenRouter => {
            "OpenRouter models take ReasoningConfig::OpenRouter; convert another variant with ReasoningConfig::to_openrouter"
        }
        Provider::ClaudeCode => {
            "Claude Code models take ReasoningConfig::AnthropicAdaptive or ::Anthropic, whose effort becomes Claude Code's --effort"
        }
    }
}

/// `config` without the settings `model`'s provider has no use for, each
/// dropped with a warning logged once per provider and setting. Borrowed
/// unchanged when there are none.
pub(crate) fn without_foreign_settings<'a>(
    model: &Model,
    config: &'a GenerationConfig,
) -> Cow<'a, GenerationConfig> {
    let provider = &model.provider;
    let settings: Vec<_> = foreign_settings(model)
        .into_iter()
        .filter(|setting| (setting.is_set)(config))
        .collect();
    let foreign_reasoning = config
        .reasoning
        .as_ref()
        .filter(|reasoning| !takes_reasoning(provider, reasoning));
    if settings.is_empty() && foreign_reasoning.is_none() {
        return Cow::Borrowed(config);
    }
    let mut kept = config.clone();
    for setting in settings {
        notice::warn_once(&format!("foreign:{provider:?}:{}", setting.name), || {
            format!(
                "arko_ai: GenerationConfig::{} is not used by {provider:?} models (here '{}') and is ignored; it applies to {}.",
                setting.name, model.id, setting.applies_to
            )
        });
        (setting.clear)(&mut kept);
    }
    if let Some(reasoning) = foreign_reasoning {
        let variant = reasoning.provider();
        notice::warn_once(
            &format!("foreign:{provider:?}:reasoning:{variant:?}"),
            || {
                format!(
                    "arko_ai: a {variant:?} ReasoningConfig is not used by {provider:?} models (here '{}') and is ignored, so the model's default reasoning applies; {}.",
                    model.id,
                    reasoning_hint(provider)
                )
            },
        );
        kept.reasoning = None;
    }
    Cow::Owned(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{AnthropicCacheTtl, AnthropicInferenceGeo, OpenAIServiceTier, Verbosity},
        providers::openrouter::OpenRouterOptions,
        reasoning::{OpenAIEffort, OpenAIReasoningMode},
    };

    /// Every provider-specific setting set.
    fn everything() -> GenerationConfig {
        GenerationConfig::default()
            .with_verbosity(Verbosity::Low)
            .with_openai_reasoning_mode(OpenAIReasoningMode::Pro)
            .with_openai_service_tier(OpenAIServiceTier::Flex)
            .with_anthropic_thinking_enabled(true)
            .with_anthropic_summarized_thinking(true)
            .with_anthropic_cache_ttl(AnthropicCacheTtl::OneHour)
            .with_anthropic_fast_mode(true)
            .with_anthropic_inference_geo(AnthropicInferenceGeo::Us)
            .with_long_context(true)
            .with_interleaved_thinking(true)
            .with_openrouter_options(OpenRouterOptions::new())
            .with_max_tokens(100)
    }

    fn set(config: &GenerationConfig) -> Vec<&'static str> {
        let all = [
            &VERBOSITY,
            &OPENAI_REASONING_MODE,
            &OPENAI_SERVICE_TIER,
            &ANTHROPIC_THINKING_ENABLED,
            &ANTHROPIC_SUMMARIZED_THINKING,
            &ANTHROPIC_CACHE_TTL,
            &ANTHROPIC_FAST_MODE,
            &ANTHROPIC_INFERENCE_GEO,
            &LONG_CONTEXT,
            &INTERLEAVED_THINKING,
            &OPENROUTER,
        ];
        all.into_iter()
            .filter(|setting| (setting.is_set)(config))
            .map(|setting| setting.name)
            .collect()
    }

    #[test]
    fn each_provider_keeps_only_its_own_settings() {
        let config = everything();
        let cases = [
            (
                Model::openai_gpt_6_astra(),
                vec!["verbosity", "openai_reasoning_mode", "openai_service_tier"],
            ),
            (
                Model::anthropic_claude_opus_5_5(),
                vec![
                    "anthropic_thinking_enabled",
                    "anthropic_summarized_thinking",
                    "anthropic_cache_ttl",
                    "anthropic_fast_mode",
                    "anthropic_inference_geo",
                    "long_context",
                    "interleaved_thinking",
                ],
            ),
            (Model::google_gemini_3_5_flash(), vec![]),
            (
                Model::openrouter("anthropic/claude-opus-5.5"),
                vec![
                    "verbosity",
                    "anthropic_summarized_thinking",
                    "anthropic_cache_ttl",
                    "long_context",
                    "interleaved_thinking",
                    "openrouter",
                ],
            ),
            (
                Model::openrouter("openai/gpt-6-sol"),
                vec!["verbosity", "openrouter"],
            ),
        ];
        for (model, kept) in cases {
            let sanitized = without_foreign_settings(&model, &config);
            assert_eq!(set(&sanitized), kept, "{}", model.id);
            assert_eq!(sanitized.max_tokens, Some(100), "{}", model.id);
            for dropped in set(&config).into_iter().filter(|name| !kept.contains(name)) {
                assert!(
                    notice::warned(&format!("foreign:{:?}:{dropped}", model.provider)),
                    "{} {dropped}",
                    model.id
                );
            }
        }
    }

    #[test]
    fn another_provider_s_reasoning_is_dropped() {
        let openai = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, true));
        let claude = Model::anthropic_claude_opus_5_5();
        let sanitized = without_foreign_settings(&claude, &openai);
        assert!(sanitized.reasoning.is_none());
        assert!(notice::warned("foreign:Anthropic:reasoning:OpenAI"));
        // The provider's own variant is kept, and nothing is copied.
        let own = without_foreign_settings(&Model::openai_gpt_6_astra(), &openai);
        assert!(matches!(own, Cow::Borrowed(_)));
        let plain = GenerationConfig::default();
        assert!(matches!(
            without_foreign_settings(&claude, &plain),
            Cow::Borrowed(_)
        ));
    }
}

#[cfg(test)]
mod request_tests {
    use crate::{
        config::{AnthropicCacheTtl, GenerationConfig, OpenAIServiceTier, Verbosity},
        conversation::Conversation,
        model::Model,
        notice,
        providers::{anthropic::AnthropicClient, google::GoogleClient, openai::OpenAIClient},
        reasoning::{OpenAIEffort, OpenAIReasoningMode, ReasoningConfig},
    };

    fn conversation(model: Model) -> Conversation {
        let mut conversation = Conversation::new(model).with_prompt_cache_key("key");
        conversation.add_user_message("Hi.");
        conversation
    }

    /// Settings for another provider change nothing in the request. Before,
    /// the direct providers ignored them silently (and OpenAI rejected
    /// another provider's reasoning on non-reasoning models).
    #[test]
    fn direct_providers_send_the_request_they_would_without_foreign_settings() {
        let plain = GenerationConfig::default().with_max_tokens(1000);

        let openai = conversation(Model::openai_gpt_4_1());
        let foreign = plain
            .clone()
            .with_anthropic_fast_mode(true)
            .with_long_context(true)
            .with_reasoning(ReasoningConfig::anthropic_adaptive());
        let client = OpenAIClient::new("test");
        assert_eq!(
            client.request_body(&openai, &foreign, false).unwrap(),
            client.request_body(&openai, &plain, false).unwrap()
        );
        assert!(notice::warned("foreign:OpenAI:anthropic_fast_mode"));
        assert!(notice::warned("foreign:OpenAI:reasoning:Anthropic"));

        let claude = conversation(Model::anthropic_claude_opus_5_5());
        let foreign = plain
            .clone()
            .with_verbosity(Verbosity::Low)
            .with_openai_service_tier(OpenAIServiceTier::Flex)
            .with_openai_reasoning_mode(OpenAIReasoningMode::Pro)
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::Max, true));
        let client = AnthropicClient::new("test");
        assert_eq!(
            client.request_body(&claude, &foreign).unwrap(),
            client.request_body(&claude, &plain).unwrap()
        );
        assert!(notice::warned("foreign:Anthropic:verbosity"));
        assert!(notice::warned("foreign:Anthropic:reasoning:OpenAI"));

        let gemini = conversation(Model::google_gemini_3_5_flash());
        let foreign = plain
            .clone()
            .with_openai_service_tier(OpenAIServiceTier::Flex)
            .with_anthropic_cache_ttl(AnthropicCacheTtl::OneHour)
            .with_reasoning(ReasoningConfig::anthropic(2048));
        let client = GoogleClient::new("test");
        assert_eq!(
            client.request_body(&gemini, &foreign).unwrap(),
            client.request_body(&gemini, &plain).unwrap()
        );
        assert!(notice::warned("foreign:Google:anthropic_cache_ttl"));
        assert!(notice::warned("foreign:Google:reasoning:Anthropic"));
    }

    #[test]
    fn claude_models_without_sampling_parameters_drop_a_temperature_with_a_warning() {
        let claude = conversation(Model::anthropic_claude_opus_5_5());
        let client = AnthropicClient::new("test");
        let body = client
            .request_body(&claude, &GenerationConfig::default().with_temperature(0.2))
            .unwrap();
        assert!(!body.contains("temperature"), "{body}");
        assert!(notice::warned("anthropic:temperature:claude-opus-5-5"));
    }
}
