//! How long providers keep prompt-cache entries.
//!
//! A cache entry expires once it has gone unused for its lifetime; every
//! request that hits it restarts the clock. Anthropic: "the cache is
//! refreshed for no additional cost each time the cached content is used".
//! OpenAI: "Reusing the prefix refreshes its lifetime without another
//! cache-write charge." An agent that makes a request before the lifetime
//! runs out therefore keeps its prompt cached however long it stays idle in
//! between requests.
//!
//! Sources:
//! - Anthropic: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
//!   (5 minutes by default, 1 hour with `ttl: "1h"`); OpenRouter passes
//!   Claude models' `cache_control`, `ttl` included, through to Anthropic.
//! - OpenAI: <https://developers.openai.com/api/docs/guides/prompt-caching>.
//!   GPT-5.6 and later take `prompt_cache_options.ttl`, whose "only
//!   supported value, `30m`, is also the default". Earlier models keep
//!   entries "around 5 to 10 minutes of inactivity" in memory for
//!   organizations with Zero Data Retention; other organizations default to
//!   extended retention, which "typically keeps entries available for around
//!   30 minutes and can retain them for up to 24 hours".

use std::time::Duration;

use crate::config::{AnthropicCacheTtl, GenerationConfig};
use crate::model::{Model, Provider};

/// How long a provider keeps a request's prompt cache after its last use,
/// as far as the provider documents it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheLifetime {
    /// Entries last this long after their last use; every cache hit starts
    /// the lifetime again.
    Known(Duration),
    /// The provider may cache the prompt, but documents no lifetime for this
    /// model or route (Gemini, and OpenRouter models other than Claude).
    Unknown,
    /// The request is not cached: caching is turned off for it, or the model
    /// doesn't support caching.
    Disabled,
}

/// Anthropic's default cache lifetime (`cache_control` without a `ttl`, or
/// [`AnthropicCacheTtl::FiveMinutes`]).
pub const ANTHROPIC_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(5 * 60);

/// Anthropic's extended cache lifetime, [`AnthropicCacheTtl::OneHour`].
pub const ANTHROPIC_EXTENDED_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(60 * 60);

/// OpenAI's cache lifetime for GPT-5.6 and later models with an API key:
/// `prompt_cache_options.ttl` accepts only `30m`, which is also the default,
/// so this client sends none.
pub const OPENAI_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(30 * 60);

/// The cache lifetime assumed for OpenAI models before GPT-5.6. Their
/// retention depends on the organization, which a client can't see: Zero
/// Data Retention organizations get in-memory retention ("around 5 to 10
/// minutes of inactivity"), others extended retention (around 30 minutes,
/// up to 24 hours). The lower bound holds for both.
pub const OPENAI_IN_MEMORY_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(5 * 60);

/// The cache lifetime on the ChatGPT OAuth (Codex backend) path for GPT-5.6
/// and later models, a fixed property of the backend: it caches implicitly
/// only, and no request setting configures the lifetime (this client sends
/// it no `prompt_cache_options`). The backend documents none; this is the
/// API's 30 minutes, which a measurement with the `oauth_cache_probe`
/// example's `--idle-gap` option is confirming.
pub const CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(30 * 60);

impl AnthropicCacheTtl {
    /// How long the entry lasts after its last use.
    pub fn duration(self) -> Duration {
        match self {
            Self::FiveMinutes => ANTHROPIC_PROMPT_CACHE_LIFETIME,
            Self::OneHour => ANTHROPIC_EXTENDED_PROMPT_CACHE_LIFETIME,
        }
    }
}

/// The prompt-cache lifetime of requests for `model` with `config`, from a
/// conversation with caching enabled (the default; a conversation with
/// [`Conversation::enable_cache`] off matches `config.prompt_caching` set to
/// `Some(false)`). `openai_chatgpt_oauth` says whether OpenAI requests go
/// through ChatGPT OAuth ([`LLMClient::prompt_cache_lifetime`] reads it from
/// the client).
///
/// - A model without cache support: `Disabled`.
/// - Anthropic, and Claude models on OpenRouter: `Disabled` when
///   [`GenerationConfig::prompt_caching`] is off, since no `cache_control` is
///   sent; otherwise [`GenerationConfig::anthropic_cache_ttl`]'s lifetime, 5
///   minutes by default.
/// - OpenAI GPT-5.6 and later with an API key: `Disabled` when caching is off
///   (explicit mode without breakpoints), otherwise 30 minutes.
/// - OpenAI GPT-5.6 and later through ChatGPT OAuth:
///   [`CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME`]. The backend caches implicitly
///   only and can't be configured, so turning caching off doesn't change it.
/// - Earlier OpenAI models: 5 minutes, the lifetime every organization gets
///   at least; their caching is automatic and can't be turned off.
/// - Gemini (implicit caching the provider manages) and other OpenRouter
///   models: `Unknown`.
///
/// [`Conversation::enable_cache`]: crate::conversation::Conversation::enable_cache
/// [`LLMClient::prompt_cache_lifetime`]: crate::client::LLMClient::prompt_cache_lifetime
pub fn prompt_cache_lifetime(
    model: &Model,
    config: &GenerationConfig,
    openai_chatgpt_oauth: bool,
) -> PromptCacheLifetime {
    if !model.supports_cache {
        return PromptCacheLifetime::Disabled;
    }
    let enabled = config.prompt_caching.unwrap_or(true);
    let anthropic = || {
        if enabled {
            PromptCacheLifetime::Known(
                config
                    .anthropic_cache_ttl
                    .map_or(ANTHROPIC_PROMPT_CACHE_LIFETIME, AnthropicCacheTtl::duration),
            )
        } else {
            PromptCacheLifetime::Disabled
        }
    };
    match model.provider {
        Provider::Anthropic => anthropic(),
        Provider::OpenRouter if crate::providers::openrouter::is_claude_family(&model.id) => {
            anthropic()
        }
        Provider::OpenRouter | Provider::Google => PromptCacheLifetime::Unknown,
        Provider::OpenAI if !model.supports_openai_prompt_cache_options() => {
            PromptCacheLifetime::Known(OPENAI_IN_MEMORY_PROMPT_CACHE_LIFETIME)
        }
        Provider::OpenAI if openai_chatgpt_oauth => {
            PromptCacheLifetime::Known(CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME)
        }
        Provider::OpenAI if enabled => PromptCacheLifetime::Known(OPENAI_PROMPT_CACHE_LIFETIME),
        Provider::OpenAI => PromptCacheLifetime::Disabled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PromptCacheLifetime::{Disabled, Known, Unknown};

    const MINUTE: Duration = Duration::from_secs(60);

    fn lifetime(model: &Model, config: &GenerationConfig) -> PromptCacheLifetime {
        prompt_cache_lifetime(model, config, false)
    }

    #[test]
    fn anthropic_lifetimes_follow_the_cache_ttl_on_either_route() {
        let default = GenerationConfig::default();
        let hour = GenerationConfig::default().with_anthropic_cache_ttl(AnthropicCacheTtl::OneHour);
        let five =
            GenerationConfig::default().with_anthropic_cache_ttl(AnthropicCacheTtl::FiveMinutes);
        let off = GenerationConfig::default().with_prompt_caching(false);
        for model in [
            Model::anthropic_claude_opus_5_5(),
            Model::openrouter("anthropic/claude-opus-5.5"),
        ] {
            assert_eq!(
                lifetime(&model, &default),
                Known(5 * MINUTE),
                "{}",
                model.id
            );
            assert_eq!(lifetime(&model, &five), Known(5 * MINUTE));
            assert_eq!(lifetime(&model, &hour), Known(60 * MINUTE));
            assert_eq!(lifetime(&model, &off), Disabled);
            // The auth mode is OpenAI's only.
            assert_eq!(
                prompt_cache_lifetime(&model, &hour, true),
                Known(60 * MINUTE)
            );
        }
    }

    #[test]
    fn openai_lifetimes_depend_on_the_model_generation_and_auth() {
        let default = GenerationConfig::default();
        let off = GenerationConfig::default().with_prompt_caching(false);
        for model in [
            Model::openai_gpt_5_6_luna(),
            Model::openai_gpt_6_astra(),
            Model::get(Provider::OpenAI, "gpt-6-sol").unwrap(),
        ] {
            assert_eq!(
                lifetime(&model, &default),
                Known(30 * MINUTE),
                "{}",
                model.id
            );
            assert_eq!(lifetime(&model, &off), Disabled);
            // The ChatGPT backend's caching is implicit and fixed.
            for config in [&default, &off] {
                assert_eq!(
                    prompt_cache_lifetime(&model, config, true),
                    Known(CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME)
                );
            }
        }
        // Before GPT-5.6: automatic caching, the lifetime every organization
        // gets at least, on either auth.
        let older = Model::openai_gpt_5_5();
        for config in [&default, &off] {
            for oauth in [false, true] {
                assert_eq!(
                    prompt_cache_lifetime(&older, config, oauth),
                    Known(5 * MINUTE)
                );
            }
        }
    }

    #[test]
    fn undocumented_and_uncached_models() {
        let default = GenerationConfig::default();
        assert_eq!(lifetime(&Model::google_gemini_2_5_pro(), &default), Unknown);
        assert_eq!(
            lifetime(&Model::openrouter("openai/gpt-6-astra"), &default),
            Unknown
        );
        assert_eq!(
            lifetime(&Model::openrouter("deepseek/deepseek-v4"), &default),
            Unknown
        );
        for id in ["gpt-5.5-pro-2026-04-23", "o3-pro-2025-06-10"] {
            let model = Model::get(Provider::OpenAI, id).unwrap();
            assert!(!model.supports_cache);
            assert_eq!(lifetime(&model, &default), Disabled, "{id}");
        }
        let mut uncached = Model::openrouter("anthropic/claude-opus-5.5");
        uncached.supports_cache = false;
        assert_eq!(lifetime(&uncached, &default), Disabled);
    }
}
