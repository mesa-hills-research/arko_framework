//! Opt-in live tests against OpenRouter for behavior the docs leave open.
//! They make paid requests and need `OPENROUTER_API_KEY`:
//!
//! ```text
//! OPENROUTER_API_KEY=... cargo test -p arko-ai openrouter::live_tests -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Models can be overridden with `OPENROUTER_LIVE_CLAUDE`,
//! `OPENROUTER_LIVE_OLD_CLAUDE`, `OPENROUTER_LIVE_GEMINI`,
//! `OPENROUTER_LIVE_GPT5` and `OPENROUTER_LIVE_GPT6`. Each test prints what
//! it observed (`--nocapture`) so undocumented behavior can be recorded.

use super::*;
use crate::{
    reasoning::{OpenRouterReasoning, ReasoningArtifact, ReasoningConfig, ReasoningEffort},
    tool::{ToolConfig, ToolDefinition, ToolResult},
};
use serde_json::json;

fn live() -> Option<OpenRouterClient> {
    match std::env::var("OPENROUTER_API_KEY") {
        Ok(key) if !key.trim().is_empty() => Some(OpenRouterClient::new(key.trim())),
        _ => {
            eprintln!("skipping: OPENROUTER_API_KEY is not set");
            None
        }
    }
}

fn slug(variable: &str, default: &str) -> String {
    std::env::var(variable).unwrap_or_else(|_| default.to_string())
}

fn claude() -> String {
    slug("OPENROUTER_LIVE_CLAUDE", "anthropic/claude-sonnet-5")
}

fn old_claude() -> String {
    slug("OPENROUTER_LIVE_OLD_CLAUDE", "anthropic/claude-haiku-4.5")
}

fn gemini() -> String {
    slug("OPENROUTER_LIVE_GEMINI", "google/gemini-3.5-flash")
}

fn gpt5() -> String {
    slug("OPENROUTER_LIVE_GPT5", "openai/gpt-5.5")
}

fn gpt6() -> String {
    slug("OPENROUTER_LIVE_GPT6", "openai/gpt-6-sol")
}

fn weather_tools() -> ToolConfig {
    ToolConfig::new(vec![ToolDefinition::new(
        "get_weather",
        "Current weather for a city. Always use this tool for weather questions.",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )])
}

fn reasoning(effort: ReasoningEffort) -> GenerationConfig {
    GenerationConfig::default()
        .with_max_tokens(4096)
        .with_reasoning(ReasoningConfig::openrouter_effort(effort))
        .with_tool_config(weather_tools())
}

async fn call(
    client: &OpenRouterClient,
    conversation: &Conversation,
    config: &GenerationConfig,
    stream: bool,
) -> Result<GenerationResponse, Error> {
    if !stream {
        return client.generate_text(conversation, config).await;
    }
    let mut events = client.generate_stream(conversation, config).await?;
    while let Some(event) = events.next().await {
        if let StreamEvent::End(response) = event? {
            return Ok(*response);
        }
    }
    Err(Error::StreamTruncated("no End event".into()))
}

fn details(response: &GenerationResponse) -> Vec<Value> {
    match &response.message.reasoning_artifact {
        Some(ReasoningArtifact::OpenRouter { details }) => details.clone(),
        _ => Vec::new(),
    }
}

fn describe(details: &[Value]) -> Vec<String> {
    details
        .iter()
        .map(|item| {
            let field = |name: &str| {
                item.get(name)
                    .and_then(Value::as_str)
                    .unwrap_or("-")
                    .to_string()
            };
            let data = field("data");
            format!(
                "{} format={} index={} signature={} data={}",
                field("type"),
                field("format"),
                item.get("index").map_or("-".into(), Value::to_string),
                item.get("signature")
                    .is_some_and(|s| s.as_str().is_some_and(|s| !s.is_empty())),
                if data == "-" {
                    data
                } else {
                    format!(
                        "{} chars{}",
                        data.len(),
                        if data.contains("[REDACTED]") {
                            " [REDACTED]"
                        } else {
                            ""
                        }
                    )
                },
            )
        })
        .collect()
}

/// Turn 1 asks for a tool call with reasoning on; turn 2 replays the
/// reasoning_details verbatim with the tool result and must be accepted.
async fn tool_continuation(client: &OpenRouterClient, model: &str, stream: bool) {
    let mut conversation = Conversation::new(Model::openrouter(model));
    conversation.add_user_message("What's the weather in Paris? Use the tool.");
    let config = reasoning(ReasoningEffort::Low);
    let first = call(client, &conversation, &config, stream).await.unwrap();
    let kept = details(&first);
    println!(
        "{model} stream={stream} turn 1: stop={:?} details={:#?}",
        first.stop_reason,
        describe(&kept)
    );
    assert_eq!(
        first.stop_reason,
        crate::stop::StopReason::ToolUse,
        "{model}: expected a tool call"
    );
    let calls = first.message.tool_calls.clone();
    conversation.add_assistant_message_from_response(first);
    conversation.add_tool_results(
        calls
            .iter()
            .map(|call| {
                ToolResult::from_tool_call(call, r#"{"temperature_c": 18, "sky": "clear"}"#)
            })
            .collect(),
    );
    let second = call(client, &conversation, &config, stream).await.unwrap_or_else(|error| {
        panic!("{model} stream={stream}: continuation with replayed reasoning_details failed: {error}")
    });
    println!(
        "{model} stream={stream} turn 2: {:?} {}",
        second.stop_reason,
        second.text()
    );
    assert!(!second.text().is_empty());
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn reasoning_details_round_trip_continues_tool_calls() {
    let Some(client) = live() else { return };
    for model in [claude(), gemini(), gpt5()] {
        for stream in [false, true] {
            tool_continuation(&client, &model, stream).await;
        }
    }
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn streamed_reasoning_details_have_the_non_streamed_shape() {
    let Some(client) = live() else { return };
    for model in [claude(), gemini(), gpt5()] {
        let mut conversation = Conversation::new(Model::openrouter(&model));
        conversation.add_user_message("Think briefly, then say which is larger: 9.11 or 9.9?");
        let config = GenerationConfig::default()
            .with_max_tokens(4096)
            .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::Low));
        let whole = describe(&details(
            &call(&client, &conversation, &config, false).await.unwrap(),
        ));
        let merged = describe(&details(
            &call(&client, &conversation, &config, true).await.unwrap(),
        ));
        println!("{model}\n  non-streamed: {whole:#?}\n  streamed+merged: {merged:#?}");
        let kinds = |items: &[String]| {
            items
                .iter()
                .map(|d| d.split(' ').next().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kinds(&whole),
            kinds(&merged),
            "{model}: merged stream items should match non-streamed item types"
        );
        assert!(
            !merged.iter().any(|d| d.contains("[REDACTED]")),
            "{model}: streamed encrypted data was redacted"
        );
    }
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn exclude_hides_reasoning_text_report_what_remains() {
    let Some(client) = live() else { return };
    for model in [claude(), gemini()] {
        let mut conversation = Conversation::new(Model::openrouter(&model));
        conversation.add_user_message("What's the weather in Oslo? Use the tool.");
        let config = GenerationConfig::default()
            .with_max_tokens(4096)
            .with_tool_config(weather_tools())
            .with_reasoning(ReasoningConfig::openrouter(
                OpenRouterReasoning::effort(ReasoningEffort::Low).with_exclude(true),
            ));
        let response = call(&client, &conversation, &config, false).await.unwrap();
        println!(
            "{model} exclude=true: reasoning_tokens={:?} details={:#?}",
            response.usage.reasoning_tokens,
            describe(&details(&response))
        );
    }
}

fn long_system_prompt() -> String {
    // Comfortably above every provider's minimum cacheable prefix (1024-4096 tokens).
    (0..900)
        .map(|n| format!("Rule {n}: answer concisely and cite rule numbers when relevant."))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn session_id_turns_hit_the_cache_and_prompt_tokens_include_cached() {
    let Some(client) = live() else { return };
    for zdr in [false, true] {
        let model = claude();
        let mut conversation =
            Conversation::new(Model::openrouter(&model)).with_system_message(&long_system_prompt());
        let mut config = GenerationConfig::default().with_max_tokens(256);
        if zdr {
            config = config.with_openrouter_options(
                OpenRouterOptions::new().with_provider(ProviderPreferences::zdr_only()),
            );
        }
        conversation.add_user_message("Say hello.");
        let first = client.generate_text(&conversation, &config).await.unwrap();
        conversation.add_assistant_message_from_response(first.clone());
        conversation.add_user_message("Say goodbye.");
        let second = client.generate_text(&conversation, &config).await.unwrap();
        let (usage, provider) = (&second.usage, &second.model_call_usage.upstream_provider);
        println!(
            "zdr={zdr} turn 1: prompt={} cached={:?} written={:?} via {:?}\n        turn 2: prompt={} cached={:?} written={:?} via {provider:?}",
            first.usage.input_tokens,
            first.usage.cached_tokens,
            first.usage.cache_write_tokens,
            first.model_call_usage.upstream_provider,
            usage.input_tokens,
            usage.cached_tokens,
            usage.cache_write_tokens
        );
        let cached = usage.cached_tokens.unwrap_or(0);
        assert!(cached > 0, "zdr={zdr}: turn 2 should read the cache");
        assert!(
            usage.input_tokens >= cached,
            "prompt_tokens should include cached tokens"
        );
    }
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn prompt_cache_key_is_forwarded_to_openai() {
    let Some(client) = live() else { return };
    let mut conversation = Conversation::new(Model::openrouter(gpt6()));
    conversation.add_user_message("Say hi.");
    let mut prepared = client
        .build_request(
            &conversation,
            &GenerationConfig::default().with_max_tokens(64),
        )
        .unwrap();
    prepared.body.stream = Some(true);
    let mut body = serde_json::to_value(&prepared.body).unwrap();
    // Debug echo works only when streaming and must not be used in production.
    body["debug"] = json!({"echo_upstream_body": true});
    let text = client
        .http_client
        .post(client.url("chat/completions"))
        .bearer_auth(&client.api_key)
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let upstream = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find_map(|chunk| chunk.pointer("/debug/echo_upstream_body").cloned())
        .expect("a debug chunk with the upstream body");
    println!("upstream body: {upstream:#}");
    assert_eq!(
        upstream.get("prompt_cache_key"),
        Some(&json!(conversation.prompt_cache_key))
    );
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn require_parameters_accepts_cache_control_and_session_id() {
    let Some(client) = live() else { return };
    let mut conversation = Conversation::new(Model::openrouter(claude()));
    conversation.add_user_message("Say hi.");
    let config = GenerationConfig::default()
        .with_max_tokens(64)
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_provider(ProviderPreferences::new().with_require_parameters(true)),
        );
    let response = client.generate_text(&conversation, &config).await;
    println!(
        "require_parameters with cache_control + session_id: {:?}",
        response.as_ref().map(|r| r.text())
    );
    response.unwrap();
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn auto_router_honours_zdr_and_allowed_models() {
    let Some(client) = live() else { return };
    let mut conversation = Conversation::new(Model::openrouter("openrouter/auto"));
    conversation.add_user_message("Write a haiku about routers.");
    let config = GenerationConfig::default()
        .with_max_tokens(512)
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_auto_router(AutoRouterOptions::zdr().with_allowed_models(["anthropic/*"])),
        );
    let response = client.generate_text(&conversation, &config).await.unwrap();
    let usage = &response.model_call_usage;
    println!(
        "auto router served {} via {:?}",
        usage.model, usage.upstream_provider
    );
    assert!(usage.model.starts_with("anthropic/"), "{}", usage.model);
    assert_eq!(usage.requested_model.as_deref(), Some("openrouter/auto"));
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn cache_hits_over_tool_turns_with_auto_exacto_and_price_sort() {
    let Some(client) = live() else { return };
    let turns = 4;
    for policy in [
        AutoExactoPolicy::AllowAutoExacto,
        AutoExactoPolicy::DisableAutoExacto,
    ] {
        let mut conversation = Conversation::new(Model::openrouter(claude()))
            .with_system_message(&long_system_prompt());
        let config = GenerationConfig::default()
            .with_max_tokens(1024)
            .with_tool_config(weather_tools())
            .with_openrouter_options(OpenRouterOptions::new().with_auto_exacto(policy));
        let (mut cached, mut prompt) = (0, 0);
        for turn in 0..turns {
            conversation.add_user_message(&format!("Weather in city number {turn}? Use the tool."));
            let response = client.generate_text(&conversation, &config).await.unwrap();
            println!(
                "{policy:?} turn {turn}: prompt={} cached={:?} via {:?}",
                response.usage.input_tokens,
                response.usage.cached_tokens,
                response.model_call_usage.upstream_provider
            );
            if turn > 0 {
                cached += response.usage.cached_tokens.unwrap_or(0);
                prompt += response.usage.input_tokens;
            }
            let calls = response.message.tool_calls.clone();
            conversation.add_assistant_message_from_response(response);
            if !calls.is_empty() {
                conversation.add_tool_results(
                    calls
                        .iter()
                        .map(|call| ToolResult::from_tool_call(call, "15C, cloudy"))
                        .collect(),
                );
                let answer = client.generate_text(&conversation, &config).await.unwrap();
                conversation.add_assistant_message_from_response(answer);
            }
        }
        println!(
            "{policy:?}: cache hit rate after turn 1 = {:.2}",
            cached as f64 / prompt.max(1) as f64
        );
    }
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn older_claude_accepts_effort_without_max_tokens() {
    let Some(client) = live() else { return };
    let mut conversation = Conversation::new(Model::openrouter(old_claude()));
    conversation.add_user_message("Is 221 prime? Answer in one sentence.");
    let config = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::Low));
    let response = client.generate_text(&conversation, &config).await.unwrap();
    println!(
        "{}: reasoning_tokens={:?} text={}",
        old_claude(),
        response.usage.reasoning_tokens,
        response.text()
    );
}

/// The catalog lists `parallel_tool_calls` for few models (not Claude, GPT
/// or Gemini); check that `false` still reaches them rather than being
/// dropped, since agents rely on it.
#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn parallel_tool_calls_false_reaches_major_models() {
    let Some(client) = live() else { return };
    for model in [claude(), gpt5(), gemini()] {
        let mut conversation = Conversation::new(Model::openrouter(&model));
        conversation.add_user_message(
            "What's the weather in Paris, Oslo and Rome? Call the tool for every city.",
        );
        let config = GenerationConfig::default()
            .with_max_tokens(2048)
            .with_tool_config(weather_tools().with_parallel_tool_calls(false));
        let response = client.generate_text(&conversation, &config).await.unwrap();
        let calls = response.message.tool_calls.len();
        println!("{model} parallel_tool_calls=false: {calls} tool call(s)");
        assert!(
            calls <= 1,
            "{model}: parallel_tool_calls=false was not applied"
        );
    }
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn max_price_is_accepted_as_numbers() {
    let Some(client) = live() else { return };
    let mut conversation = Conversation::new(Model::openrouter(gemini()));
    conversation.add_user_message("Say hi.");
    let config = GenerationConfig::default()
        .with_max_tokens(64)
        .with_openrouter_options(OpenRouterOptions::new().with_provider(
            ProviderPreferences::new().with_max_price(MaxPrice::per_million_tokens(100.0, 100.0)),
        ));
    client.generate_text(&conversation, &config).await.unwrap();
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn gpt6_accepts_a_mid_conversation_effort_update() {
    let Some(client) = live() else { return };
    let mut conversation = Conversation::new(Model::openrouter(gpt6()));
    conversation.add_user_message("Name a prime number.");
    let config = GenerationConfig::default()
        .with_max_tokens(1024)
        .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::Medium));
    let first = client.generate_text(&conversation, &config).await.unwrap();
    conversation.add_assistant_message_from_response(first);
    conversation.add_reasoning_effort_update(ReasoningEffort::Low);
    conversation.add_user_message("Name another.");
    let second = client.generate_text(&conversation, &config).await.unwrap();
    println!(
        "after update: reasoning_tokens={:?} cached={:?}",
        second.usage.reasoning_tokens, second.usage.cached_tokens
    );
}

#[tokio::test]
#[ignore = "live OpenRouter request; needs OPENROUTER_API_KEY"]
async fn catalogs_parse_and_responses_report_the_serving_provider() {
    let Some(client) = live() else { return };
    let public = client.fetch_models().await.unwrap();
    println!("public catalog: {} models", public.models().len());
    assert!(public.contains(&claude()));
    let user = client.fetch_user_models().await.unwrap();
    println!("user catalog: {} models", user.models().len());
    let mut conversation = Conversation::new(
        client
            .model(&gemini())
            .unwrap_or_else(|| Model::openrouter(gemini())),
    );
    conversation.add_user_message("Say hi.");
    let response = client
        .generate_text(
            &conversation,
            &GenerationConfig::default().with_max_tokens(64),
        )
        .await
        .unwrap();
    println!(
        "served {} via {:?}; cost={:?}",
        response.model_call_usage.model,
        response.model_call_usage.upstream_provider,
        response.usage.reported_cost
    );
    assert!(
        response.usage.reported_cost.is_some(),
        "usage.cost is always returned"
    );
}
