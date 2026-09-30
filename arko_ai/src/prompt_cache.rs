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
//!   In-memory retention keeps entries "around 5 to 10 minutes of
//!   inactivity" (the default before GPT-5.6 for organizations with Zero Data
//!   Retention; others get extended retention, "around 30 minutes and ...
//!   up to 24 hours"). GPT-5.6 and later take `prompt_cache_options.ttl`,
//!   whose "only supported value, `30m`, is also the default". The ChatGPT
//!   OAuth (Codex) backend caches implicitly only; its lifetime is neither
//!   documented nor configurable. This module reports OpenAI as a
//!   conservative 5 minutes on every path, on purpose: a request after a
//!   longer idle time can miss the cache and pay for the whole prompt again
//!   (see [`OPENAI_PROMPT_CACHE_LIFETIME`]).

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

/// The cache lifetime reported for OpenAI models, with an API key or ChatGPT
/// OAuth: a conservative 5 minutes, the lower end of the documented
/// in-memory retention. Longer lifetimes exist (30 minutes for GPT-5.6 and
/// later on the API, extended retention for many organizations) but aren't
/// guaranteed on every path; the ChatGPT OAuth backend caches implicitly
/// only, with a lifetime no request can configure.
///
/// A live measurement over ChatGPT OAuth (GPT-6 Astra, 2026-09-27) found the
/// cache still warm after 4- and 40-minute idle gaps, but OpenAI only
/// documents "typically around 30 minutes" (API) and nothing for the OAuth
/// backend, so arko defaults to short waits. A longer fixed wait usually
/// still hits the cache but isn't guaranteed to.
pub const OPENAI_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(5 * 60);

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
/// - OpenAI: [`OPENAI_PROMPT_CACHE_LIFETIME`], a conservative 5 minutes, on
///   every model and auth; `Disabled` only for GPT-5.6 and later with an API
///   key and caching off (explicit mode without breakpoints). Earlier models
///   cache automatically, and the ChatGPT OAuth backend caches implicitly,
///   so turning caching off doesn't change them.
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
        // Claude Code sets its own cache lifetime.
        Provider::OpenRouter | Provider::Google | Provider::ClaudeCode => {
            PromptCacheLifetime::Unknown
        }
        Provider::OpenAI
            if !enabled
                && !openai_chatgpt_oauth
                && model.supports_openai_prompt_cache_options() =>
        {
            PromptCacheLifetime::Disabled
        }
        Provider::OpenAI => PromptCacheLifetime::Known(OPENAI_PROMPT_CACHE_LIFETIME),
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

    /// OpenAI is a conservative 5 minutes on every model and auth; only an
    /// API-key request that turns caching off is uncached.
    #[test]
    fn openai_lifetimes_are_a_conservative_five_minutes() {
        let default = GenerationConfig::default();
        let off = GenerationConfig::default().with_prompt_caching(false);
        for model in [
            Model::openai_gpt_5_6_luna(),
            Model::openai_gpt_6_astra(),
            Model::get(Provider::OpenAI, "gpt-6-sol").unwrap(),
        ] {
            assert_eq!(
                lifetime(&model, &default),
                Known(5 * MINUTE),
                "{}",
                model.id
            );
            assert_eq!(lifetime(&model, &off), Disabled);
            // The ChatGPT backend's caching is implicit and fixed.
            for config in [&default, &off] {
                assert_eq!(
                    prompt_cache_lifetime(&model, config, true),
                    Known(5 * MINUTE)
                );
            }
        }
        // Before GPT-5.6 caching is automatic, on either auth.
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
