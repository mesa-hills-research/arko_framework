//! How long providers keep prompt-cache entries.
//!
//! A cache entry expires once it has gone unused for its lifetime. Every
//! request that hits it restarts the clock. Anthropic says "the cache is
//! refreshed for no additional cost each time the cached content is used".
//! OpenAI says "Reusing the prefix refreshes its lifetime without another
//! cache-write charge." An agent that makes a request before the lifetime
//! runs out therefore keeps its prompt cached however long it stays idle in
//! between requests.
//!
//! The reported lifetimes follow these sources and defaults:
//! - Anthropic documents caching at <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
//!   (5 minutes by default, 1 hour with `ttl: "1h"`). OpenRouter passes
//!   Claude models' `cache_control`, `ttl` included, through to Anthropic.
//!   [`GenerationConfig::long_prompt_cache`] asks for the hour.
//! - OpenAI documents caching at <https://developers.openai.com/api/docs/guides/prompt-caching>.
//!   GPT-5.6 and later take `prompt_cache_options.ttl`, a minimum lifetime
//!   that every cache read extends, 30 minutes by default
//!   ([`OPENAI_STANDARD_PROMPT_CACHE_LIFETIME`]) and 12 hours with `"12h"`
//!   ([`OPENAI_EXTENDED_PROMPT_CACHE_LIFETIME`], asked for with
//!   [`GenerationConfig::long_prompt_cache`]). Earlier models only
//!   document typical lifetimes. In-memory retention keeps entries "around 5
//!   to 10 minutes of inactivity" (their default for organizations with Zero
//!   Data Retention), and extended retention "around 30 minutes and ... up to
//!   24 hours". This module reports them as a conservative 5 minutes
//!   ([`OPENAI_PROMPT_CACHE_LIFETIME`]), since a request after a longer idle
//!   time can miss the cache and pay for the whole prompt again.
//! - ChatGPT OAuth (the Codex backend) uses implicit caching only, with a
//!   lifetime no request configures. This module reports 30 minutes
//!   ([`CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME`]).
//!
//! A host that knows better for its route sets
//! [`GenerationConfig::prompt_cache_lifetime_seconds`], which replaces the
//! reported lifetime.

use std::time::Duration;

use crate::config::{AnthropicCacheTtl, GenerationConfig};
use crate::model::{Model, Provider};

/// How long a provider keeps a request's prompt cache after its last use,
/// as far as the provider documents it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheLifetime {
    /// Entries last this long after their last use. Every cache hit starts
    /// the lifetime again.
    Known(Duration),
    /// The provider may cache the prompt, but documents no lifetime for this
    /// model or route (Gemini, and OpenRouter models other than Claude).
    Unknown,
    /// The request is not cached because caching is turned off for it or the model
    /// doesn't support caching.
    Disabled,
}

/// Anthropic's default cache lifetime (`cache_control` without a `ttl`, or
/// [`AnthropicCacheTtl::FiveMinutes`]).
pub const ANTHROPIC_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(5 * 60);

/// Anthropic's extended cache lifetime, [`AnthropicCacheTtl::OneHour`].
pub const ANTHROPIC_EXTENDED_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(60 * 60);

/// The cache lifetime reported for OpenAI models before GPT-5.6 with an API
/// key, a conservative 5 minutes, the lower end of the documented in-memory
/// retention. Their extended retention, the default for many organizations,
/// typically lasts around 30 minutes, which OpenAI doesn't promise. A host
/// that knows its route keeps the cache longer sets
/// [`GenerationConfig::prompt_cache_lifetime_seconds`].
pub const OPENAI_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(5 * 60);

/// OpenAI's default cache lifetime on GPT-5.6 and later with an API key
/// (`prompt_cache_options.ttl: "30m"`), 30 minutes. OpenAI documents it as a
/// minimum, and every cache read extends it to at least 30 minutes after
/// that read.
pub const OPENAI_STANDARD_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(30 * 60);

/// OpenAI's extended cache lifetime on GPT-5.6 and later with an API key
/// (`prompt_cache_options.ttl: "12h"`, sent for
/// [`GenerationConfig::long_prompt_cache`]), 12 hours. OpenAI documents it as
/// a minimum too, and a read keeps the later of its expiry and 30 minutes
/// after the read.
pub const OPENAI_EXTENDED_PROMPT_CACHE_LIFETIME: Duration = Duration::from_secs(12 * 60 * 60);

/// The cache lifetime reported for OpenAI models over ChatGPT OAuth (the
/// Codex backend), 30 minutes. The backend caches implicitly, on every
/// model, with a lifetime no request configures. A live measurement over
/// ChatGPT OAuth (GPT-6 Astra) found the cache still warm after 4- and
/// 40-minute idle gaps.
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
/// conversation with caching enabled (the default, a conversation with
/// [`Conversation::enable_cache`] off matches `config.prompt_caching` set to
/// `Some(false)`). `openai_chatgpt_oauth` says whether OpenAI requests go
/// through ChatGPT OAuth ([`LLMClient::prompt_cache_lifetime`] reads it from
/// the client).
///
/// - A model without cache support returns `Disabled`.
/// - Anthropic and Claude models on OpenRouter return `Disabled` when
///   [`GenerationConfig::prompt_caching`] is off, since no `cache_control` is
///   sent. Otherwise the lifetime of [`GenerationConfig::claude_cache_ttl`]
///   (from `anthropic_cache_ttl` or `long_prompt_cache`), 5 minutes by
///   default.
/// - OpenAI with an API key returns [`OPENAI_STANDARD_PROMPT_CACHE_LIFETIME`],
///   30 minutes, on GPT-5.6 and later, or
///   [`OPENAI_EXTENDED_PROMPT_CACHE_LIFETIME`], 12 hours, with
///   [`GenerationConfig::long_prompt_cache`] on. It returns `Disabled` there
///   with caching off (explicit mode without breakpoints). Earlier models return
///   [`OPENAI_PROMPT_CACHE_LIFETIME`], a conservative 5 minutes. They cache
///   automatically, so turning caching off doesn't change them.
/// - OpenAI over ChatGPT OAuth returns [`CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME`], 30
///   minutes, on every model. The backend caches implicitly, so turning
///   caching off doesn't change it.
/// - Claude Code returns the lifetime of `claude_cache_ttl` when set (passed
///   on as its `promptCacheTtl`). Otherwise `Unknown`, since Claude Code chooses
///   1 hour on a Claude subscription within its usage limits, 5 minutes on
///   an API key.
/// - Gemini (implicit caching the provider manages) and other OpenRouter
///   models return `Unknown`.
///
/// [`GenerationConfig::prompt_cache_lifetime_seconds`], when set, replaces a
/// `Known` or `Unknown` lifetime (`Some(0)` reports `Disabled`). A request
/// that isn't cached stays `Disabled`.
///
/// [`Conversation::enable_cache`]: crate::conversation::Conversation::enable_cache
/// [`LLMClient::prompt_cache_lifetime`]: crate::client::LLMClient::prompt_cache_lifetime
pub fn prompt_cache_lifetime(
    model: &Model,
    config: &GenerationConfig,
    openai_chatgpt_oauth: bool,
) -> PromptCacheLifetime {
    match (
        provider_prompt_cache_lifetime(model, config, openai_chatgpt_oauth),
        config.prompt_cache_lifetime_seconds,
    ) {
        (PromptCacheLifetime::Disabled, _) | (_, Some(0)) => PromptCacheLifetime::Disabled,
        (_, Some(seconds)) => PromptCacheLifetime::Known(Duration::from_secs(seconds)),
        (lifetime, None) => lifetime,
    }
}

/// [`prompt_cache_lifetime`] as the provider sets it, without the host's
/// override.
fn provider_prompt_cache_lifetime(
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
                    .claude_cache_ttl()
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
        // Claude Code's main conversation uses the TTL asked for (its
        // `promptCacheTtl`). Otherwise it chooses, 1 hour on a Claude
        // subscription within its usage limits, 5 minutes on an API key.
        Provider::ClaudeCode => config
            .claude_cache_ttl()
            .map_or(PromptCacheLifetime::Unknown, |ttl| {
                PromptCacheLifetime::Known(ttl.duration())
            }),
        Provider::OpenRouter | Provider::Google => PromptCacheLifetime::Unknown,
        Provider::OpenAI if openai_chatgpt_oauth => {
            PromptCacheLifetime::Known(CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME)
        }
        Provider::OpenAI if model.supports_openai_prompt_cache_options() => {
            if !enabled {
                PromptCacheLifetime::Disabled
            } else if config.long_prompt_cache == Some(true) {
                PromptCacheLifetime::Known(OPENAI_EXTENDED_PROMPT_CACHE_LIFETIME)
            } else {
                PromptCacheLifetime::Known(OPENAI_STANDARD_PROMPT_CACHE_LIFETIME)
            }
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

    /// OpenAI with an API key is 30 minutes on GPT-5.6 and later, a
    /// conservative 5 minutes on earlier models, and over ChatGPT OAuth 30
    /// minutes. Only an API-key request that turns caching off is uncached.
    #[test]
    fn openai_lifetimes_depend_on_the_model_and_the_auth() {
        let default = GenerationConfig::default();
        let off = GenerationConfig::default().with_prompt_caching(false);
        for model in [
            Model::openai_gpt_5_6_luna(),
            Model::openai_gpt_5_6(),
            Model::openai_gpt_6_astra(),
            Model::openai_gpt_6_1_terra(),
            Model::get(Provider::OpenAI, "gpt-6-sol").unwrap(),
        ] {
            assert!(model.supports_openai_prompt_cache_options(), "{}", model.id);
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
                    Known(30 * MINUTE)
                );
            }
        }
        // Before GPT-5.6 caching is automatic, on either auth, and its
        // lifetime only typical.
        for older in [Model::openai_gpt_5_5(), Model::openai_gpt_5_4_mini()] {
            assert!(!older.supports_openai_prompt_cache_options());
            for config in [&default, &off] {
                assert_eq!(
                    prompt_cache_lifetime(&older, config, false),
                    Known(5 * MINUTE),
                    "{}",
                    older.id
                );
                assert_eq!(
                    prompt_cache_lifetime(&older, config, true),
                    Known(30 * MINUTE)
                );
            }
        }
        assert_eq!(OPENAI_PROMPT_CACHE_LIFETIME, 5 * MINUTE);
        assert_eq!(OPENAI_STANDARD_PROMPT_CACHE_LIFETIME, 30 * MINUTE);
        assert_eq!(CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME, 30 * MINUTE);
    }

    /// The long cache is an hour on every Claude route and 12 hours on
    /// OpenAI GPT-5.6 and later with an API key. Routes with one lifetime
    /// keep it, and an uncached request stays uncached.
    #[test]
    fn the_long_cache_lasts_as_long_as_each_route_allows() {
        let long = GenerationConfig::default().with_long_prompt_cache(true);
        let short = GenerationConfig::default().with_long_prompt_cache(false);
        let long_off = long.clone().with_prompt_caching(false);
        let opus = Model::anthropic_claude_opus_5_5();
        for model in [
            opus.clone(),
            Model::openrouter("anthropic/claude-opus-5.5"),
            Model::claude_code(opus.clone()),
        ] {
            assert_eq!(lifetime(&model, &long), Known(60 * MINUTE), "{}", model.id);
            assert_eq!(lifetime(&model, &short), Known(5 * MINUTE), "{}", model.id);
        }
        assert_eq!(lifetime(&opus, &long_off), Disabled);
        // Claude Code without either chooses for itself.
        assert_eq!(
            lifetime(
                &Model::claude_code(opus.clone()),
                &GenerationConfig::default()
            ),
            Unknown
        );
        // An explicit Claude TTL comes first.
        let five = long
            .clone()
            .with_anthropic_cache_ttl(AnthropicCacheTtl::FiveMinutes);
        assert_eq!(
            five.claude_cache_ttl(),
            Some(AnthropicCacheTtl::FiveMinutes)
        );
        assert_eq!(lifetime(&opus, &five), Known(5 * MINUTE));
        assert_eq!(GenerationConfig::default().claude_cache_ttl(), None);

        for model in [Model::openai_gpt_6_luna(), Model::openai_gpt_5_6_sol()] {
            assert_eq!(lifetime(&model, &long), Known(12 * 60 * MINUTE));
            assert_eq!(lifetime(&model, &short), Known(30 * MINUTE));
            assert_eq!(lifetime(&model, &long_off), Disabled);
            // The ChatGPT backend takes no lifetime.
            assert_eq!(
                prompt_cache_lifetime(&model, &long, true),
                Known(30 * MINUTE)
            );
        }
        assert_eq!(lifetime(&Model::openai_gpt_5_5(), &long), Known(5 * MINUTE));
        assert_eq!(lifetime(&Model::google_gemini_2_5_pro(), &long), Unknown);
        assert_eq!(
            lifetime(&Model::openrouter("openai/gpt-6-luna"), &long),
            Unknown
        );
        assert_eq!(OPENAI_EXTENDED_PROMPT_CACHE_LIFETIME, 12 * 60 * MINUTE);

        // Saved without it, a config loads without it, and it is written
        // only when set.
        let saved = serde_json::to_value(GenerationConfig::default()).unwrap();
        assert!(saved.get("long_prompt_cache").is_none(), "{saved}");
        let saved = serde_json::to_value(&long).unwrap();
        assert_eq!(saved["long_prompt_cache"], true);
        let loaded: GenerationConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(loaded.long_prompt_cache, Some(true));
    }

    /// The host's lifetime replaces the provider's, known or not, on any
    /// auth. An uncached request stays uncached, and zero reports none.
    #[test]
    fn the_hosts_lifetime_replaces_the_providers() {
        let hour = GenerationConfig::default().with_prompt_cache_lifetime(60 * MINUTE);
        assert_eq!(hour.prompt_cache_lifetime_seconds, Some(3600));
        let astra = Model::openai_gpt_6_astra();
        for oauth in [false, true] {
            assert_eq!(
                prompt_cache_lifetime(&astra, &hour, oauth),
                Known(60 * MINUTE)
            );
        }
        assert_eq!(
            lifetime(&Model::google_gemini_2_5_pro(), &hour),
            Known(60 * MINUTE)
        );
        assert_eq!(
            lifetime(
                &Model::anthropic_claude_opus_5_5(),
                &hour
                    .clone()
                    .with_anthropic_cache_ttl(AnthropicCacheTtl::FiveMinutes)
            ),
            Known(60 * MINUTE)
        );
        // Uncached requests: caching off with an API key, or a model without
        // caching.
        assert_eq!(
            lifetime(&astra, &hour.clone().with_prompt_caching(false)),
            Disabled
        );
        let uncached = Model::get(Provider::OpenAI, "o3-pro-2025-06-10").unwrap();
        assert_eq!(lifetime(&uncached, &hour), Disabled);
        let zero = GenerationConfig::default().with_prompt_cache_lifetime(Duration::ZERO);
        assert_eq!(prompt_cache_lifetime(&astra, &zero, true), Disabled);
        // Saved without it, a config loads without it, and it is written
        // only when set.
        let saved = serde_json::to_value(GenerationConfig::default()).unwrap();
        assert!(
            saved.get("prompt_cache_lifetime_seconds").is_none(),
            "{saved}"
        );
        let saved = serde_json::to_value(&hour).unwrap();
        assert_eq!(saved["prompt_cache_lifetime_seconds"], 3600);
        let loaded: GenerationConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(loaded.prompt_cache_lifetime_seconds, Some(3600));
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
