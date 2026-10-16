//! Opt-in live tests of the prompt cache refresh
//! ([`AgentRuntime::refresh_prompt_cache`]) per route: that the provider
//! takes its output limit with the route's thinking settings, and that it
//! reads the cached conversation. They make paid requests and need each
//! route's key (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`):
//!
//! ```text
//! ANTHROPIC_API_KEY=... cargo test -p arko_agents keep_warm_live -- --ignored --nocapture --test-threads=1
//! ```
//!
//! A test whose key isn't set passes without a request. Each prints the
//! refresh it saw: its usage, its output limit, and the error of a limit
//! the provider refused (the refresh then went again with a larger one).
//! The OpenRouter model can be set with `OPENROUTER_LIVE_CLAUDE`.

use std::sync::Arc;

use crate::agent_config::{AgentConfig, AgentLLMConfig, AgentToolConfig};
use crate::runtime::{AgentRuntime, CacheRefresh};
use crate::task::{Task, TaskState};
use crate::tool::ToolRegistry;

fn key(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(key) if !key.trim().is_empty() => Some(key.trim().to_string()),
        _ => {
            eprintln!("skipping: {name} is not set");
            None
        }
    }
}

/// Instructions long enough for every route to cache them (over 4,096
/// tokens).
fn instructions() -> String {
    (0..400)
        .map(|n| format!("Rule {n}: answer briefly, and stay on the topic of item {n}.\n"))
        .collect()
}

/// Runs a chat turn on `model`, then refreshes its prompt cache.
async fn run_then_refresh(
    client: arko_ai::LLMClient,
    model: arko_ai::Model,
    generation_config: arko_ai::GenerationConfig,
) -> CacheRefresh {
    let mut config = AgentConfig::builder(
        "chat".into(),
        "Chat".into(),
        AgentLLMConfig {
            model,
            generation_config,
        },
    )
    .role_description("A chat for a live test.".into())
    .instructions(instructions())
    .build()
    .unwrap();
    config.tool_config = AgentToolConfig::chat();
    let task = Task::new(
        config,
        "Say hello in one word.".into(),
        std::env::temp_dir(),
        None,
    );
    let runtime = AgentRuntime::new(Arc::new(client));
    let registry = Arc::new(ToolRegistry::new());
    let (mut events, _commands, handle) =
        runtime.run_task(task, registry.clone(), None, false, None, None, None);
    let drain = tokio::spawn(async move { while events.recv().await.is_some() {} });
    let task = handle.await.unwrap().unwrap();
    let _ = drain.await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    let refresh = runtime.refresh_prompt_cache(task, registry, None).await;
    eprintln!("{refresh:#?}");
    let refresh = refresh.unwrap();
    assert!(
        refresh.usage.cached_tokens.unwrap_or(0) > 0,
        "the refresh read nothing from the cache: {:?}",
        refresh.usage
    );
    refresh
}

/// Anthropic with adaptive thinking (Opus 5.5 always thinks adaptively):
/// a one-token limit with the thinking settings unchanged.
#[tokio::test]
#[ignore = "makes a paid request; needs ANTHROPIC_API_KEY"]
async fn keep_warm_live_anthropic_adaptive_thinking() {
    let Some(key) = key("ANTHROPIC_API_KEY") else {
        return;
    };
    let refresh = run_then_refresh(
        arko_ai::LLMClient::new().with_anthropic_api_key(&key),
        arko_ai::Model::anthropic_claude_opus_5_5(),
        arko_ai::GenerationConfig::default(),
    )
    .await;
    eprintln!(
        "cap {:?}, refused: {:?}",
        refresh.output_cap, refresh.refused_cap
    );
}

/// OpenRouter's Anthropic models without reasoning: a one-token limit.
#[tokio::test]
#[ignore = "makes a paid request; needs OPENROUTER_API_KEY"]
async fn keep_warm_live_openrouter_claude() {
    let Some(key) = key("OPENROUTER_API_KEY") else {
        return;
    };
    let slug = std::env::var("OPENROUTER_LIVE_CLAUDE")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string());
    let refresh = run_then_refresh(
        arko_ai::LLMClient::new().with_openrouter_api_key(&key),
        arko_ai::Model::openrouter(slug),
        arko_ai::GenerationConfig::default(),
    )
    .await;
    eprintln!(
        "cap {:?}, refused: {:?}",
        refresh.output_cap, refresh.refused_cap
    );
}

/// OpenAI's Responses API with reasoning: its smallest limit, 16.
#[tokio::test]
#[ignore = "makes a paid request; needs OPENAI_API_KEY"]
async fn keep_warm_live_openai() {
    let Some(key) = key("OPENAI_API_KEY") else {
        return;
    };
    let refresh = run_then_refresh(
        arko_ai::LLMClient::new().with_openai_api_key(&key),
        arko_ai::Model::openai_gpt_5_6_luna(),
        arko_ai::GenerationConfig::default(),
    )
    .await;
    assert_eq!(refresh.output_cap, Some(16));
}
