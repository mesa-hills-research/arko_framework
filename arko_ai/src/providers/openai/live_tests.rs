//! Opt-in live tests of OpenAI's prompt caching with an API key. They make
//! paid requests (well under a cent on GPT-6 Luna) and need `OPENAI_API_KEY`,
//! from the environment or from the `.env` file `OPENAI_LIVE_ENV_FILE` names:
//!
//! ```text
//! OPENAI_API_KEY=... cargo test -p arko_ai openai::live_tests -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `OPENAI_LIVE_MODEL` picks another model that takes `prompt_cache_options`.
//! Each test prints what it observed (`--nocapture`). The cache test also
//! traces its calls into `OPENAI_LIVE_TRACE` when set. A trace holds the
//! prompts and the responses, with credentials redacted.

use super::*;
use crate::{
    model::{Model, Provider},
    reasoning::{OpenAIEffort, ReasoningConfig},
};
use serde_json::json;

fn api_key() -> Option<String> {
    if let Ok(path) = std::env::var("OPENAI_LIVE_ENV_FILE") {
        // Variables already set are kept.
        dotenv::from_path(path).ok();
    }
    match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.trim().is_empty() => Some(key.trim().to_string()),
        _ => {
            eprintln!("skipping: OPENAI_API_KEY is not set");
            None
        }
    }
}

/// The model ids the key can use, from `GET /v1/models`.
#[tokio::test]
#[ignore = "live OpenAI request, needs OPENAI_API_KEY"]
async fn lists_the_models_the_key_can_use() {
    let Some(key) = api_key() else { return };
    let response = super::super::default_http_client()
        .get(format!("{OPENAI_BASE_URL}/models"))
        .bearer_auth(key)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    assert!(status.is_success(), "{status}");
    let mut ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect();
    ids.sort_unstable();
    println!("{}", json!({"status": status.as_u16(), "ids": ids}));
}

/// About 16,000 tokens of text that no other run shares.
fn long_prompt(nonce: &str) -> String {
    const WORDS: [&str; 16] = [
        "amber", "falcon", "crosses", "seven", "quiet", "rivers", "before", "dawn", "copper",
        "lantern", "gathers", "distant", "harbor", "winds", "under", "slate",
    ];
    let mut state: u32 = 0x9e37_79b9;
    let mut text = format!("Reference notes {nonce}.\n");
    for line in 0..1_300 {
        text.push_str(&format!("Note {line:04}:"));
        for _ in 0..11 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            text.push(' ');
            text.push_str(WORDS[(state % 16) as usize]);
        }
        text.push_str(".\n");
    }
    text
}

/// The raw `usage` of each traced call in `dir`, in order, with its HTTP
/// status.
fn traced_usage(dir: &std::path::Path) -> Vec<Value> {
    let calls = std::fs::read_to_string(dir.join("calls.jsonl")).unwrap();
    calls
        .lines()
        .map(|line| {
            let call: Value = serde_json::from_str(line).unwrap();
            let file = dir
                .join(call["conversation_dir"].as_str().unwrap())
                .join(format!(
                    "{:04}-response.json",
                    call["seq"].as_u64().unwrap()
                ));
            let response: Value =
                serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
            json!({
                "http_status": response["http_status"],
                "status": response["status"],
                "error": response["error"],
                "model": response["response"]["model"],
                "usage": response["response"]["usage"],
            })
        })
        .collect()
}

/// Request-level `prompt_cache_options.ttl: "12h"` is accepted, its writes
/// are reported as `ttl_12h_tokens` and read back on the next turn, and a
/// request without it writes `ttl_30m_tokens`. The usage and its cost are
/// checked against the raw response.
#[tokio::test]
#[ignore = "live OpenAI request, needs OPENAI_API_KEY"]
async fn the_twelve_hour_cache_is_accepted_written_read_and_priced() {
    let Some(key) = api_key() else { return };
    let id = std::env::var("OPENAI_LIVE_MODEL").unwrap_or_else(|_| "gpt-6-luna".into());
    let model = Model::get(Provider::OpenAI, &id).expect("a registered OpenAI model");
    assert!(model.supports_openai_prompt_cache_options(), "{id}");
    let trace_dir = std::env::var("OPENAI_LIVE_TRACE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!("arko-live-{}", uuid::Uuid::new_v4()))
        });
    let client = OpenAIClient::new(&key).with_request_trace(Some(RequestTrace::new(&trace_dir)));
    let config = |long: bool| {
        GenerationConfig::default()
            .with_max_tokens(64)
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::None, false))
            .with_long_prompt_cache(long)
    };
    let conversation = |nonce: &str| {
        let mut conversation =
            Conversation::new(model.clone()).with_system_message("Answer in one word.");
        conversation.add_user_message(&format!(
            "{}\nHow many notes are there? Reply with the number only.",
            long_prompt(nonce)
        ));
        conversation
    };
    let run = uuid::Uuid::new_v4().to_string();

    // A 12-hour write, then a read of it with a new turn after it.
    let mut twelve = conversation(&format!("{run} 12h"));
    let first = client.generate_text(&twelve, &config(true)).await.unwrap();
    twelve.add_assistant_message_from_response(first.clone());
    twelve.add_user_message("And how many words does each note have? Reply with the number only.");
    let second = client.generate_text(&twelve, &config(true)).await.unwrap();
    // A 30-minute write, on a prompt of its own.
    let thirty = conversation(&format!("{run} 30m"));
    let third = client.generate_text(&thirty, &config(false)).await.unwrap();

    let raw = traced_usage(&trace_dir);
    assert_eq!(raw.len(), 3, "{raw:#?}");
    for (response, raw) in [&first, &second, &third].into_iter().zip(&raw) {
        println!("{}", json!({"raw": raw, "parsed": response.usage}));
        assert_eq!(raw["http_status"], 200, "{raw}");
        let usage = &raw["usage"];
        let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);
        let (input, cached) = (
            count("/input_tokens"),
            count("/input_tokens_details/cached_tokens"),
        );
        let written = count("/input_tokens_details/cache_write_tokens");
        let split = "/input_tokens_details/cache_write_tokens_details";
        let (thirty_minutes, twelve_hours) = (
            count(&format!("{split}/ttl_30m_tokens")),
            count(&format!("{split}/ttl_12h_tokens")),
        );
        let parsed = &response.usage;
        assert_eq!(parsed.input_tokens as u64, input);
        assert_eq!(parsed.cached_tokens.unwrap_or(0) as u64, cached);
        assert_eq!(parsed.cache_write_tokens.unwrap_or(0) as u64, written);
        assert_eq!(
            parsed.cache_write_1h_tokens.unwrap_or(0) as u64,
            twelve_hours
        );
        assert_eq!(thirty_minutes + twelve_hours, written, "{usage}");
        // The guide's formula, at the model's prices (short context).
        let pricing = &model.pricing;
        let per_token = |rate: f64| rate / 1_000_000.0;
        let expected = (input - cached - written) as f64
            * per_token(pricing.input_cost_per_million)
            + cached as f64 * per_token(pricing.cached_input_cost_per_million.unwrap())
            + thirty_minutes as f64 * per_token(pricing.cache_write_cost_per_million.unwrap())
            + twelve_hours as f64 * per_token(pricing.cache_write_1h_cost_per_million.unwrap())
            + count("/output_tokens") as f64 * per_token(pricing.output_cost_per_million);
        assert!(
            (parsed.total_cost - expected).abs() < 1e-12,
            "{} against {expected}",
            parsed.total_cost
        );
    }
    let count = |raw: &Value, field: &str| {
        raw["usage"]["input_tokens_details"]["cache_write_tokens_details"][field]
            .as_u64()
            .unwrap_or(0)
    };
    assert!(count(&raw[0], "ttl_12h_tokens") > 10_000, "{}", raw[0]);
    assert!(
        raw[1]["usage"]["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0)
            > 10_000,
        "{}",
        raw[1]
    );
    assert!(count(&raw[2], "ttl_30m_tokens") > 10_000, "{}", raw[2]);
    assert_eq!(count(&raw[2], "ttl_12h_tokens"), 0, "{}", raw[2]);
    let cost: f64 = [&first, &second, &third]
        .iter()
        .map(|response| response.usage.total_cost)
        .sum();
    println!(
        "{}",
        json!({"model": id, "total_cost": cost, "trace": trace_dir})
    );
}
