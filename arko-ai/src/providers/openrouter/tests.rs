//! Unit tests: request JSON per mapping rule (including what is omitted),
//! catalog parsing and validation, response and stream parsing (fixtures
//! follow OpenRouter's documented examples), errors, usage and cost.

use super::request::{INTERLEAVED_THINKING_BETA, STRUCTURED_OUTPUTS_BETA};
use super::response::{self, DetailsMerger, ReportedUsage, ResponseContext};
use super::*;
use crate::{
    config::{AnthropicCacheTtl, OpenAIServiceTier, Verbosity},
    error::RetryClass,
    message::{ConfigurationUpdate, Content, Image, Message},
    model::Provider,
    reasoning::{
        AnthropicReasoningEffort, GeminiThinkingLevel, OpenAIEffort, OpenAIReasoningMode,
        OpenRouterReasoning, ReasoningArtifact, ReasoningConfig, ReasoningContext, ReasoningEffort,
        ReasoningSummary,
    },
    stop::{RefusalSource, StopReason},
    tool::{ToolCall, ToolChoice, ToolConfig, ToolDefinition, ToolResult},
    usage::BillingKind,
};
use serde_json::{Value, json};

const CLAUDE: &str = "anthropic/claude-sonnet-5";
const GEMINI: &str = "google/gemini-3.5-flash";
const GPT: &str = "openai/gpt-6-sol";

fn client() -> OpenRouterClient {
    OpenRouterClient::new("test-key")
}

fn conversation(slug: &str) -> Conversation {
    let mut conversation =
        Conversation::new(Model::openrouter(slug)).with_system_message("Be brief.");
    conversation.add_user_message("hi");
    conversation
}

fn prepare(
    client: &OpenRouterClient,
    conversation: &Conversation,
    config: &GenerationConfig,
) -> Result<(Value, Vec<&'static str>), Error> {
    let prepared = client.build_request(conversation, config)?;
    Ok((
        serde_json::to_value(&prepared.body).unwrap(),
        prepared.anthropic_betas,
    ))
}

fn body(slug: &str, config: &GenerationConfig) -> Value {
    prepare(&client(), &conversation(slug), config).unwrap().0
}

fn error(slug: &str, config: &GenerationConfig) -> String {
    prepare(&client(), &conversation(slug), config)
        .unwrap_err()
        .to_string()
}

fn tool_config() -> ToolConfig {
    ToolConfig::new(vec![ToolDefinition::new(
        "lookup",
        "Look something up",
        json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
    )])
}

fn with_tools() -> GenerationConfig {
    GenerationConfig::default().with_tool_config(tool_config())
}

fn keys(value: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort();
    keys
}

/// A catalog built from the shapes of OpenRouter's documented `GET /models`
/// response (entries trimmed to the fields arko reads).
fn catalog_json() -> String {
    json!({"data": [
        {
            "id": "anthropic/claude-sonnet-5", "canonical_slug": "anthropic/claude-sonnet-5-20260630",
            "name": "Anthropic: Claude Sonnet 5", "context_length": 1000000,
            "architecture": {"modality": "text+image+file->text", "input_modalities": ["text", "image", "file"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000002", "completion": "0.00001", "web_search": "0.01", "input_cache_read": "0.0000002", "input_cache_write": "0.0000025", "input_cache_write_1h": "0.000004"},
            "top_provider": {"context_length": 1000000, "max_completion_tokens": 128000, "is_moderated": true},
            "supported_parameters": ["include_reasoning", "max_completion_tokens", "max_tokens", "reasoning", "reasoning_effort", "response_format", "stop", "structured_outputs", "tool_choice", "tools", "verbosity"],
            "reasoning": {"mandatory": false, "default_enabled": true, "supported_efforts": ["max", "xhigh", "high", "medium", "low"], "default_effort": "high"}
        },
        {
            "id": "openai/gpt-6-astra", "canonical_slug": "openai/gpt-6-astra-20260903", "context_length": 1050000,
            "architecture": {"input_modalities": ["file", "image", "text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.00001", "completion": "0.00005", "input_cache_read": "0.000001", "input_cache_write": "0.0000125",
                "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.00002", "completion": "0.000075", "input_cache_read": "0.000002", "input_cache_write": "0.000025"}]},
            "top_provider": {"context_length": 1050000, "max_completion_tokens": 128000, "is_moderated": true},
            "supported_parameters": ["include_reasoning", "max_completion_tokens", "max_tokens", "reasoning", "reasoning_effort", "response_format", "seed", "structured_outputs", "tool_choice", "tools"],
            "reasoning": {"mandatory": true, "default_enabled": true, "supported_efforts": ["max", "xhigh", "high", "medium", "low"], "default_effort": "medium"}
        },
        {
            "id": "google/gemini-3.5-flash", "canonical_slug": "google/gemini-3.5-flash-20260519", "context_length": 1048576,
            "architecture": {"input_modalities": ["text", "image", "video", "file", "audio"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.0000015", "completion": "0.000009", "input_cache_read": "0.00000015", "input_cache_write": "0.0000000833333333333333"},
            "top_provider": {"context_length": 1048576, "max_completion_tokens": 65536},
            "supported_parameters": ["include_reasoning", "max_tokens", "reasoning", "reasoning_effort", "response_format", "seed", "stop", "structured_outputs", "temperature", "tool_choice", "tools", "top_p"],
            "reasoning": {"supported_efforts": ["high", "medium", "low", "minimal"], "default_effort": "medium", "default_enabled": true, "mandatory": true}
        },
        {
            "id": "deepseek/deepseek-v4-flash", "context_length": 131072,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008", "input_cache_read": "0.00000002"},
            "top_provider": {"context_length": 131072, "max_completion_tokens": null},
            "supported_parameters": ["max_tokens", "temperature", "tools", "tool_choice", "reasoning"],
            "reasoning": {"mandatory": false, "supported_efforts": null}
        },
        {
            "id": "~anthropic/claude-sonnet-latest", "canonical_slug": "~anthropic/claude-sonnet-latest",
            "alias_target": {"name": "Anthropic: Claude Sonnet 5", "slug": "anthropic/claude-sonnet-5"}, "context_length": 1000000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002", "input_cache_write": "0.0000025"},
            "top_provider": {"context_length": 1000000, "max_completion_tokens": 128000},
            "supported_parameters": ["max_tokens", "reasoning", "tools", "tool_choice"]
        },
        {
            "id": "openrouter/auto", "canonical_slug": "openrouter/auto", "context_length": 2000000,
            "architecture": {"input_modalities": ["text", "image", "audio", "file", "video"], "output_modalities": ["text", "image"]},
            "pricing": {"prompt": "-1", "completion": "-1"},
            "top_provider": {"context_length": null, "max_completion_tokens": null},
            "supported_parameters": ["max_tokens", "reasoning", "temperature", "tool_choice", "tools"]
        },
        {"name": "entry without an id is skipped"}
    ]})
    .to_string()
}

fn catalog_client() -> OpenRouterClient {
    let client = client();
    client.set_catalog(Some(Arc::new(
        OpenRouterCatalog::from_json(&catalog_json()).unwrap(),
    )));
    client
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

// ---------------------------------------------------------------- requests

#[test]
fn minimal_requests_send_only_model_messages_and_cache_routing_keys() {
    let conversation = conversation(GEMINI);
    let (value, betas) = prepare(&client(), &conversation, &GenerationConfig::default()).unwrap();
    // Everything else is omitted so OpenRouter and the provider apply their
    // defaults; usage/stream_options/transforms are never sent.
    assert_eq!(
        keys(&value),
        ["messages", "model", "prompt_cache_key", "session_id"]
    );
    assert!(betas.is_empty());
    assert_eq!(value["model"], GEMINI);
    assert_eq!(value["session_id"], conversation.prompt_cache_key);
    assert_eq!(value["prompt_cache_key"], conversation.prompt_cache_key);
    assert_eq!(
        value["messages"],
        json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
    );
}

#[test]
fn session_id_is_always_sent_and_caching_adds_the_cache_key_and_claude_cache_control() {
    let claude = body(CLAUDE, &GenerationConfig::default());
    assert_eq!(claude["cache_control"], json!({"type": "ephemeral"}));
    let hour = body(
        CLAUDE,
        &GenerationConfig::default().with_anthropic_cache_ttl(AnthropicCacheTtl::OneHour),
    );
    assert_eq!(
        hour["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
    let alias = body(
        "~anthropic/claude-sonnet-latest",
        &GenerationConfig::default(),
    );
    assert_eq!(alias["cache_control"], json!({"type": "ephemeral"}));
    assert!(
        body(GPT, &GenerationConfig::default())
            .get("cache_control")
            .is_none()
    );

    for disabled in [
        prepare(
            &client(),
            &conversation(CLAUDE).with_cache_disabled(),
            &GenerationConfig::default(),
        )
        .unwrap()
        .0,
        body(
            CLAUDE,
            &GenerationConfig::default().with_prompt_caching(false),
        ),
    ] {
        assert!(disabled.get("prompt_cache_key").is_none());
        assert!(disabled.get("cache_control").is_none());
        assert!(
            disabled["session_id"]
                .as_str()
                .unwrap()
                .starts_with("arko:conversation:")
        );
    }

    let ttl = GenerationConfig::default().with_anthropic_cache_ttl(AnthropicCacheTtl::FiveMinutes);
    assert!(error(GPT, &ttl).contains("TTL"));
    assert!(error(CLAUDE, &ttl.clone().with_prompt_caching(false)).contains("TTL"));

    // Disabling caching on an OpenAI model that bills cache writes (known
    // from the catalog) also turns off OpenAI's automatic breakpoints.
    let catalog = catalog_client();
    let off = GenerationConfig::default().with_prompt_caching(false);
    let (astra, _) = prepare(&catalog, &conversation("openai/gpt-6-astra"), &off).unwrap();
    assert_eq!(astra["prompt_cache_options"], json!({"mode": "explicit"}));
    assert!(astra.get("prompt_cache_key").is_none());
    let (on, _) = prepare(
        &catalog,
        &conversation("openai/gpt-6-astra"),
        &GenerationConfig::default(),
    )
    .unwrap();
    assert!(on.get("prompt_cache_options").is_none());
    assert!(
        body("openai/gpt-6-astra", &off)
            .get("prompt_cache_options")
            .is_none(),
        "unknown without a catalog"
    );
    let (claude_off, _) = prepare(&catalog, &conversation(CLAUDE), &off).unwrap();
    assert!(claude_off.get("prompt_cache_options").is_none());

    let long_key = conversation(GPT).with_prompt_cache_key("k".repeat(257));
    let rejected = prepare(&client(), &long_key, &GenerationConfig::default()).unwrap_err();
    assert!(rejected.to_string().contains("256"), "{rejected}");
}

#[test]
fn sampling_limits_and_verbosity_are_sent_only_when_set() {
    let value = body(
        GPT,
        &GenerationConfig::new(0.3)
            .with_max_tokens(900)
            .with_verbosity(Verbosity::Low),
    );
    assert_eq!(value["temperature"], 0.3);
    assert_eq!(value["max_tokens"], 900);
    assert_eq!(value["verbosity"], "low");
    let defaults = body(GPT, &GenerationConfig::default());
    for key in ["temperature", "max_tokens", "verbosity", "reasoning"] {
        assert!(defaults.get(key).is_none(), "{key}");
    }
}

#[test]
fn reasoning_is_sent_only_from_the_openrouter_variant() {
    let reasoning = OpenRouterReasoning::effort(ReasoningEffort::XHigh)
        .with_exclude(false)
        .with_summary(ReasoningSummary::Detailed)
        .with_context(ReasoningContext::AllTurns)
        .with_mode(OpenAIReasoningMode::Pro);
    let value = body(
        GPT,
        &GenerationConfig::default().with_reasoning(ReasoningConfig::openrouter(reasoning)),
    );
    assert_eq!(
        value["reasoning"],
        json!({"effort": "xhigh", "exclude": false, "summary": "detailed", "context": "all_turns", "mode": "pro"})
    );
    let budget = body(
        CLAUDE,
        &GenerationConfig::default().with_reasoning(ReasoningConfig::openrouter(
            OpenRouterReasoning::max_tokens(4096),
        )),
    );
    assert_eq!(budget["reasoning"], json!({"max_tokens": 4096}));
    let off = body(
        GEMINI,
        &GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openrouter(OpenRouterReasoning::disabled())),
    );
    assert_eq!(off["reasoning"], json!({"enabled": false}));

    let both = OpenRouterReasoning {
        effort: Some(ReasoningEffort::High),
        max_tokens: Some(2000),
        ..OpenRouterReasoning::default()
    };
    assert!(
        error(
            CLAUDE,
            &GenerationConfig::default().with_reasoning(ReasoningConfig::openrouter(both))
        )
        .contains("not both")
    );
    let foreign = error(
        GPT,
        &GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, true)),
    );
    assert!(foreign.contains("to_openrouter"), "{foreign}");

    // Claude needs max_tokens above the thinking budget.
    let budget = |max_tokens: usize| {
        GenerationConfig::default()
            .with_max_tokens(max_tokens)
            .with_reasoning(ReasoningConfig::openrouter(
                OpenRouterReasoning::max_tokens(4096),
            ))
    };
    assert!(error(CLAUDE, &budget(4096)).contains("must exceed"));
    assert_eq!(body(CLAUDE, &budget(4097))["max_tokens"], 4097);
    assert_eq!(
        body(GEMINI, &budget(4096))["reasoning"],
        json!({"max_tokens": 4096})
    );
}

#[test]
fn other_reasoning_configurations_convert_only_explicitly() {
    let convert = |config: ReasoningConfig| config.to_openrouter();
    assert_eq!(
        convert(ReasoningConfig::openai(OpenAIEffort::Minimal, true)).unwrap(),
        OpenRouterReasoning::effort(ReasoningEffort::Minimal).with_summary(ReasoningSummary::Auto)
    );
    assert_eq!(
        convert(ReasoningConfig::anthropic(8000)).unwrap(),
        OpenRouterReasoning::max_tokens(8000)
    );
    assert_eq!(
        convert(ReasoningConfig::anthropic(0)).unwrap(),
        OpenRouterReasoning::disabled()
    );
    assert!(
        convert(ReasoningConfig::anthropic_with_effort(
            8000,
            AnthropicReasoningEffort::High
        ))
        .is_err()
    );
    assert_eq!(
        convert(ReasoningConfig::anthropic_adaptive_with_effort(
            AnthropicReasoningEffort::Max
        ))
        .unwrap(),
        OpenRouterReasoning::effort(ReasoningEffort::Max)
    );
    assert_eq!(
        convert(ReasoningConfig::anthropic_adaptive()).unwrap(),
        OpenRouterReasoning::enabled()
    );
    assert_eq!(
        convert(ReasoningConfig::google(-1, false)).unwrap(),
        OpenRouterReasoning::enabled()
    );
    assert_eq!(
        convert(ReasoningConfig::google(0, true)).unwrap(),
        OpenRouterReasoning::disabled()
    );
    assert_eq!(
        convert(ReasoningConfig::google(2048, true)).unwrap(),
        OpenRouterReasoning::max_tokens(2048)
    );
    assert!(convert(ReasoningConfig::google(-5, true)).is_err());
    assert_eq!(
        convert(ReasoningConfig::gemini(GeminiThinkingLevel::Low, true)).unwrap(),
        OpenRouterReasoning::effort(ReasoningEffort::Low)
    );
    let config = ReasoningConfig::openrouter_effort(ReasoningEffort::None);
    assert_eq!(config.provider(), Provider::OpenRouter);
    assert!(!config.is_enabled());
    assert!(ReasoningConfig::openrouter(OpenRouterReasoning::enabled()).is_enabled());
}

#[test]
fn verbosity_and_effort_conflict_only_on_claude() {
    let config = GenerationConfig::default()
        .with_verbosity(Verbosity::High)
        .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::Low));
    assert!(error(CLAUDE, &config).contains("output_config.effort"));
    let value = body(GPT, &config);
    assert_eq!(value["verbosity"], "high");
    assert_eq!(value["reasoning"], json!({"effort": "low"}));
    // Verbosity alone is fine on Claude.
    assert_eq!(
        body(
            CLAUDE,
            &GenerationConfig::default().with_verbosity(Verbosity::Low)
        )["verbosity"],
        "low"
    );
}

#[test]
fn tools_choices_and_the_claude_strict_beta() {
    let value = body(GPT, &with_tools());
    assert_eq!(
        value["tools"],
        json!([{"type": "function", "function": {
            "name": "lookup", "description": "Look something up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }}])
    );
    assert!(value.get("tool_choice").is_none() && value.get("parallel_tool_calls").is_none());
    for (choice, expected) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::Required, json!("required")),
        (ToolChoice::None, json!("none")),
        (
            ToolChoice::Specific("lookup".into()),
            json!({"type": "function", "function": {"name": "lookup"}}),
        ),
    ] {
        let config = GenerationConfig::default().with_tool_config(
            tool_config()
                .with_tool_choice(choice)
                .with_parallel_tool_calls(false),
        );
        let value = body(GPT, &config);
        assert_eq!(value["tool_choice"], expected);
        assert_eq!(value["parallel_tool_calls"], false);
    }

    let mut strict = tool_config();
    strict.tools[0] = strict.tools[0].clone().with_strict(true);
    let config = GenerationConfig::default().with_tool_config(strict);
    let (claude, betas) = prepare(&client(), &conversation(CLAUDE), &config).unwrap();
    assert_eq!(claude["tools"][0]["function"]["strict"], true);
    assert_eq!(betas, [STRUCTURED_OUTPUTS_BETA]);
    let (_, betas) = prepare(&client(), &conversation(GPT), &config).unwrap();
    assert!(betas.is_empty(), "the beta header is Anthropic-only");

    let interleaved = config.clone().with_interleaved_thinking(true);
    let (_, betas) = prepare(&client(), &conversation(CLAUDE), &interleaved).unwrap();
    assert_eq!(betas, [INTERLEAVED_THINKING_BETA, STRUCTURED_OUTPUTS_BETA]);
    assert!(error(GPT, &interleaved).contains("interleaved_thinking"));
    assert!(
        error(
            CLAUDE,
            &GenerationConfig::default().with_interleaved_thinking(false)
        )
        .contains("interleaved_thinking")
    );
}

#[test]
fn options_from_other_providers_are_rejected_not_dropped() {
    let cases: Vec<(GenerationConfig, &str)> = vec![
        (
            GenerationConfig::default().with_openai_reasoning_mode(OpenAIReasoningMode::Pro),
            "openai_reasoning_mode",
        ),
        (
            GenerationConfig::default().with_anthropic_thinking_enabled(true),
            "anthropic_thinking_enabled",
        ),
        (
            GenerationConfig::default().with_openai_service_tier(OpenAIServiceTier::Flex),
            "openai_service_tier",
        ),
        (
            GenerationConfig::default().with_anthropic_fast_mode(true),
            "anthropic_fast_mode",
        ),
        (
            GenerationConfig::default()
                .with_anthropic_inference_geo(crate::config::AnthropicInferenceGeo::Us),
            "anthropic_inference_geo",
        ),
    ];
    for (config, field) in cases {
        let message = error(CLAUDE, &config);
        assert!(message.contains(field), "{message}");
    }
    // Documented no-ops: OpenRouter already summarizes Claude thinking and
    // enables extended context where the model has it.
    body(
        CLAUDE,
        &GenerationConfig::default()
            .with_anthropic_summarized_thinking(true)
            .with_long_context(true),
    );
}

#[test]
fn provider_preferences_merge_field_by_field_and_serialize_to_the_wire_shape() {
    let defaults = OpenRouterOptions::new()
        .with_provider(
            ProviderPreferences::new()
                .with_only(["anthropic", "google-vertex"])
                .with_data_collection(DataCollection::Deny)
                .with_zdr(true)
                .with_max_price(MaxPrice::per_million_tokens(3.0, 15.0)),
        )
        .with_user("user-1")
        .with_service_tier(OpenRouterServiceTier::Flex)
        .with_fallback_models(["anthropic/claude-opus-5"]);
    let client = client().with_default_options(defaults);
    let request = OpenRouterOptions::new().with_provider(
        ProviderPreferences::new()
            .with_zdr(false)
            .with_sort(ProviderSort::Config {
                by: Some(SortBy::Throughput),
                partition: Some(SortPartition::None),
            })
            .with_quantizations(vec![Quantization::Fp8, Quantization::Bf16])
            .with_preferred_min_throughput(PerformanceThreshold::Percentiles {
                p50: Some(100.0),
                p75: None,
                p90: Some(50.0),
                p99: None,
            })
            .with_preferred_max_latency(PerformanceThreshold::P50(5.0))
            .with_provider_options("anthropic", json!({"beta": true})),
    );
    let (value, _) = prepare(
        &client,
        &conversation(CLAUDE),
        &GenerationConfig::default().with_openrouter_options(request),
    )
    .unwrap();
    assert_eq!(
        value["provider"],
        json!({
            "only": ["anthropic", "google-vertex"], "data_collection": "deny", "zdr": false,
            "quantizations": ["fp8", "bf16"],
            "sort": {"by": "throughput", "partition": "none"},
            "max_price": {"prompt": 3.0, "completion": 15.0},
            "preferred_min_throughput": {"p50": 100.0, "p90": 50.0},
            "preferred_max_latency": 5.0,
            "options": {"anthropic": {"beta": true}},
        })
    );
    assert_eq!(value["user"], "user-1");
    assert_eq!(value["service_tier"], "flex");
    assert_eq!(value["models"], json!(["anthropic/claude-opus-5"]));

    assert_eq!(
        serde_json::to_value(ProviderPreferences::pinned("anthropic")).unwrap(),
        json!({"only": ["anthropic"], "allow_fallbacks": false})
    );
    assert_eq!(
        serde_json::to_value(ProviderPreferences::zdr_only()).unwrap(),
        json!({"zdr": true})
    );
    let sort: ProviderSort = serde_json::from_value(json!("latency")).unwrap();
    assert_eq!(sort, ProviderSort::By(SortBy::Latency));
    // Options round-trip through saved configs.
    let saved = serde_json::to_value(client.default_options()).unwrap();
    assert_eq!(
        &serde_json::from_value::<OpenRouterOptions>(saved).unwrap(),
        client.default_options()
    );
}

#[test]
fn auto_router_settings_apply_only_to_router_slugs_under_their_own_plugin_id() {
    let router = AutoRouterOptions::zdr()
        .with_allowed_models(["anthropic/*", "openai/gpt-6*"])
        .with_excluded_models(["openai/gpt-6-astra"])
        .with_cost_tier(AutoRouterCostTier::High);
    let config = GenerationConfig::default().with_openrouter_options(
        OpenRouterOptions::new()
            .with_auto_router(router)
            .with_provider(ProviderPreferences::new().with_ignore(["azure"])),
    );
    for (slug, id) in [
        ("openrouter/auto", "auto-router"),
        ("openrouter/auto-beta", "auto-beta-router"),
    ] {
        let value = body(slug, &config);
        assert_eq!(
            value["plugins"],
            json!([{"id": id, "allowed_models": ["anthropic/*", "openai/gpt-6*"], "excluded_models": ["openai/gpt-6-astra"], "cost_tier": "high"}])
        );
        assert_eq!(
            value["provider"],
            json!({"ignore": ["azure"], "data_collection": "deny", "zdr": true})
        );
    }
    let direct = body(CLAUDE, &config);
    assert!(direct.get("plugins").is_none());
    assert_eq!(direct["provider"], json!({"ignore": ["azure"]}));
    // Provider overrides alone add no plugin entry.
    let zdr_only = GenerationConfig::default().with_openrouter_options(
        OpenRouterOptions::new().with_auto_router(AutoRouterOptions::zdr()),
    );
    let value = body("openrouter/auto", &zdr_only);
    assert!(value.get("plugins").is_none());
    assert_eq!(
        value["provider"],
        json!({"data_collection": "deny", "zdr": true})
    );
}

#[test]
fn auto_exacto_policy_sorts_by_price_only_when_cache_misses_are_costly() {
    let sort = |slug: &str, config: &GenerationConfig| body(slug, config).get("provider").cloned();
    let price = Some(json!({"sort": "price"}));
    // Without a catalog: anthropic/ and openai/ models bill cache writes.
    assert_eq!(sort(CLAUDE, &with_tools()), price);
    assert_eq!(sort(GPT, &with_tools()), price);
    assert_eq!(sort(GEMINI, &with_tools()), None);
    assert_eq!(sort("openrouter/auto", &with_tools()), None);
    // Auto Exacto only runs on tool requests, and caching must be on.
    assert_eq!(sort(CLAUDE, &GenerationConfig::default()), None);
    assert_eq!(sort(CLAUDE, &with_tools().with_prompt_caching(false)), None);
    // Explicit routing decides.
    let explicit = |preferences: ProviderPreferences| {
        with_tools().with_openrouter_options(OpenRouterOptions::new().with_provider(preferences))
    };
    assert_eq!(
        sort(
            CLAUDE,
            &explicit(ProviderPreferences::new().with_sort(SortBy::Throughput))
        ),
        Some(json!({"sort": "throughput"}))
    );
    assert_eq!(
        sort(
            CLAUDE,
            &explicit(ProviderPreferences::new().with_order(["anthropic"]))
        ),
        Some(json!({"order": ["anthropic"]}))
    );
    let policy = |policy: AutoExactoPolicy| {
        with_tools().with_openrouter_options(OpenRouterOptions::new().with_auto_exacto(policy))
    };
    assert_eq!(
        sort(CLAUDE, &policy(AutoExactoPolicy::AllowAutoExacto)),
        None
    );
    assert_eq!(
        sort(GEMINI, &policy(AutoExactoPolicy::DisableAutoExacto)),
        price
    );
    assert_eq!(
        sort(
            GEMINI,
            &GenerationConfig::default().with_openrouter_options(
                OpenRouterOptions::new().with_auto_exacto(AutoExactoPolicy::DisableAutoExacto)
            )
        ),
        None
    );

    // With catalog metadata the price of cache writes decides.
    let client = catalog_client();
    let catalog_sort = |slug: &str| {
        prepare(&client, &conversation(slug), &with_tools())
            .unwrap()
            .0
            .get("provider")
            .cloned()
    };
    assert_eq!(
        catalog_sort(GEMINI),
        price,
        "Gemini 3.5 Flash lists a cache-write price"
    );
    assert_eq!(
        catalog_sort("deepseek/deepseek-v4-flash"),
        None,
        "no cache-write price"
    );
    assert_eq!(catalog_sort(CLAUDE), price);
}

#[test]
fn messages_convert_tool_turns_images_and_replay_reasoning_details_verbatim() {
    let details = vec![
        json!({"type": "reasoning.text", "text": "Plan.", "signature": "sig-1", "id": null, "format": "anthropic-claude-v1", "index": 0}),
        json!({"type": "reasoning.encrypted", "data": "opaque==", "format": "anthropic-claude-v1", "index": 1, "future_field": {"kept": true}}),
    ];
    let mut conversation = Conversation::new(Model::openrouter(CLAUDE));
    conversation.add_user_message_with_contents(vec![
        Content::Text("Describe".into()),
        Content::Image(Image::new_base64("QUJD", "image/png")),
        Content::Image(Image::new_url("https://example.com/a.png")),
    ]);
    let mut assistant = Message::new_assistant("");
    assistant.reasoning_artifact = Some(ReasoningArtifact::new_openrouter(details.clone()));
    assistant.tool_calls = vec![
        ToolCall::from_argument_text(Some("call_1".into()), "lookup".into(), r#"{"q":"a"}"#),
        ToolCall::from_argument_text(Some("call_2".into()), "lookup".into(), r#"{"q":"#),
    ];
    conversation.messages.push(assistant);
    conversation.add_tool_results_with_text(
        vec![
            ToolResult::new(Some("call_1".into()), "lookup", "A"),
            ToolResult::error(Some("call_2".into()), "lookup", "bad arguments"),
        ],
        "Thanks",
    );
    // Reasoning from another provider can't be replayed and is dropped.
    conversation.add_assistant_message_with_reasoning(
        "Done.",
        ReasoningArtifact::new_anthropic_thinking("t", "s"),
    );

    let (value, _) = prepare(&client(), &conversation, &GenerationConfig::default()).unwrap();
    assert_eq!(
        value["messages"],
        json!([
            {"role": "user", "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ]},
            {"role": "assistant", "content": null, "reasoning_details": details, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"a\"}"}},
                // Claude needs an object: invalid arguments become `{}`.
                {"id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "A"},
            {"role": "tool", "tool_call_id": "call_2", "content": "bad arguments"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Done."},
        ])
    );

    // OpenAI models take argument strings, so they get the text verbatim.
    conversation.model = Model::openrouter(GPT);
    let (value, _) = prepare(&client(), &conversation, &GenerationConfig::default()).unwrap();
    assert_eq!(
        value["messages"][1]["tool_calls"][1]["function"]["arguments"],
        "{\"q\":"
    );
}

#[test]
fn reasoning_effort_updates_use_the_content_less_system_message_form() {
    let mut conversation = conversation(GPT);
    conversation.add_assistant_message("Plan ready.");
    conversation.add_reasoning_effort_update(ReasoningEffort::High);
    // A second update before another turn replaces the pending one.
    conversation.add_reasoning_effort_update(ReasoningEffort::Low);
    conversation.add_user_message("Rename the config file.");
    assert_eq!(
        conversation
            .messages
            .iter()
            .filter(|m| m.is_configuration_update())
            .count(),
        1
    );

    let config = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::High));
    let (value, _) = prepare(&client(), &conversation, &config).unwrap();
    assert_eq!(
        value["messages"][3],
        json!({"role": "system", "content": "", "configuration_update": {"reasoning": {"effort": "low"}}})
    );
    assert_eq!(value["messages"][4]["content"], "Rename the config file.");
    assert_eq!(value["reasoning"], json!({"effort": "high"}));

    // Updates survive saving the conversation.
    let saved = serde_json::to_string(&conversation).unwrap();
    let restored: Conversation = serde_json::from_str(&saved).unwrap();
    assert_eq!(
        restored.messages[2].configuration_update,
        Some(ConfigurationUpdate::reasoning_effort(ReasoningEffort::Low))
    );

    // Adjacent updates written by hand are rejected before sending.
    let mut adjacent = conversation.clone();
    adjacent.messages.insert(
        2,
        Message::new_configuration_update(ConfigurationUpdate::reasoning_effort(
            ReasoningEffort::Max,
        )),
    );
    assert!(
        prepare(&client(), &adjacent, &config)
            .unwrap_err()
            .to_string()
            .contains("adjacent")
    );

    // Not with pro reasoning or -pro models.
    let pro = GenerationConfig::default().with_reasoning(ReasoningConfig::openrouter(
        OpenRouterReasoning::default().with_mode(OpenAIReasoningMode::Pro),
    ));
    assert!(
        prepare(&client(), &conversation, &pro)
            .unwrap_err()
            .to_string()
            .contains("pro")
    );
    let mut pro_model = conversation.clone();
    pro_model.model = Model::openrouter("openai/gpt-6-sol-pro");
    assert!(prepare(&client(), &pro_model, &GenerationConfig::default()).is_err());
    // The pro restriction is OpenAI's; other vendors' "-pro" models pass.
    pro_model.model = Model::openrouter("google/gemini-3-pro");
    prepare(&client(), &pro_model, &GenerationConfig::default()).unwrap();

    // An update between tool calls and their results would split the pair.
    let mut split = Conversation::new(Model::openrouter(GPT));
    split.add_user_message("Look it up.");
    let mut calling = Message::new_assistant("");
    calling.tool_calls = vec![ToolCall::from_argument_text(
        Some("call_1".into()),
        "lookup".into(),
        "{}",
    )];
    split.messages.push(calling);
    split.add_reasoning_effort_update(ReasoningEffort::Low);
    split.add_tool_results(vec![ToolResult::new(Some("call_1".into()), "lookup", "A")]);
    let rejected = prepare(&client(), &split, &GenerationConfig::default()).unwrap_err();
    assert!(rejected.to_string().contains("tool results"), "{rejected}");

    // Other providers reject the update rather than sending it as a system message.
    let mut direct = conversation.clone();
    direct.model = Model::openai_gpt_6_sol();
    let rejected = crate::providers::ensure_supported_inputs(&direct).unwrap_err();
    assert!(
        matches!(rejected, Error::UnsupportedFeature(_)),
        "{rejected}"
    );
}

// ----------------------------------------------------------------- catalog

#[test]
fn catalog_entries_become_models_with_limits_prices_and_metadata() {
    let catalog = OpenRouterCatalog::from_json(&catalog_json()).unwrap();
    assert_eq!(
        catalog.models().len(),
        6,
        "the entry without an id is skipped"
    );
    assert_eq!(catalog.source(), CatalogSource::Supplied);

    let claude = catalog.get(CLAUDE).unwrap();
    assert_eq!(claude.provider, Provider::OpenRouter);
    assert_eq!(
        (claude.max_input_tokens, claude.max_output_tokens),
        (1_000_000, 128_000)
    );
    assert!(claude.supports_image_input && claude.supports_cache);
    assert_eq!(claude.model_type, crate::model::ModelType::Reasoning);
    let pricing = &claude.pricing;
    assert!(
        close(pricing.input_cost_per_million, 2.0) && close(pricing.output_cost_per_million, 10.0)
    );
    assert!(close(pricing.cached_input_cost_per_million.unwrap(), 0.2));
    assert!(close(pricing.cache_write_cost_per_million.unwrap(), 2.5));
    assert!(close(pricing.cache_write_1h_cost_per_million.unwrap(), 4.0));
    let info = claude.openrouter_info().unwrap();
    assert_eq!(
        info.canonical_slug.as_deref(),
        Some("anthropic/claude-sonnet-5-20260630")
    );
    assert!(info.supports_parameter("verbosity") && !info.supports_parameter("temperature"));
    let reasoning = info.reasoning.as_ref().unwrap();
    assert!(reasoning.effort_selectable && !reasoning.mandatory);
    assert_eq!(reasoning.supported_efforts.as_ref().unwrap()[0], "max");
    assert_eq!(reasoning.default_effort.as_deref(), Some("high"));

    // Long-prompt overrides become the long-context tier.
    let astra = catalog.get("openai/gpt-6-astra").unwrap();
    assert_eq!(astra.pricing.large_input_threshold, Some(272_000));
    assert!(close(
        astra.pricing.large_input_cost_per_million.unwrap(),
        20.0
    ));
    assert!(close(
        astra.pricing.large_cache_write_cost_per_million.unwrap(),
        25.0
    ));
    assert!(
        astra
            .openrouter_info()
            .unwrap()
            .reasoning
            .as_ref()
            .unwrap()
            .mandatory
    );

    // `supported_efforts: null` means every effort; omitted means none selectable.
    let deepseek = catalog.get("deepseek/deepseek-v4-flash").unwrap();
    let reasoning = deepseek
        .openrouter_info()
        .unwrap()
        .reasoning
        .clone()
        .unwrap();
    assert!(reasoning.effort_selectable && reasoning.supported_efforts.is_none());
    assert_eq!(
        deepseek.max_output_tokens, 0,
        "a null output limit is unknown"
    );
    assert!(!deepseek.supports_image_input);

    // Routers publish no prices.
    let auto = catalog.get("openrouter/auto").unwrap();
    assert!(!auto.openrouter_info().unwrap().pricing_known);
    assert_eq!(auto.pricing.input_cost_per_million, 0.0);
    assert!(!auto.supports_cache);

    // Canonical slugs and variant suffixes resolve, keeping the requested id.
    assert_eq!(
        catalog
            .get("anthropic/claude-sonnet-5-20260630")
            .unwrap()
            .max_output_tokens,
        128_000
    );
    let floor = catalog.get("anthropic/claude-sonnet-5:floor").unwrap();
    assert_eq!(floor.id, "anthropic/claude-sonnet-5:floor");
    assert!(catalog.contains("openai/gpt-6-astra:nitro") && !catalog.contains("openai/gpt-7"));
    let alias = catalog.get("~anthropic/claude-sonnet-latest").unwrap();
    assert_eq!(
        alias.openrouter_info().unwrap().alias_target.as_deref(),
        Some(CLAUDE)
    );

    // Catalog models survive a saved snapshot.
    let restored: Model = serde_json::from_value(serde_json::to_value(&claude).unwrap()).unwrap();
    assert_eq!(restored.openrouter, claude.openrouter);
    assert!(OpenRouterCatalog::from_json("{}").is_err());
}

#[test]
fn only_prompt_size_overrides_become_the_long_context_tier() {
    // Override shapes from https://openrouter.ai/docs/guides/overview/models
    // ("Pricing Overrides"): an entry applies only when all its conditions
    // match, and entries with conditions arko doesn't evaluate are skipped.
    let model = |id: &str, pricing: Value| {
        json!({"id": id, "context_length": 1000000, "pricing": pricing,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}})
    };
    let catalog = OpenRouterCatalog::from_json(
        &json!({"data": [
            model("test/peak-long-context", json!({"prompt": "0.000002", "completion": "0.00001", "overrides": [
                {"min_prompt_tokens": 100000, "utc_start": 1630, "utc_end": 30, "prompt": "0.00001"},
                {"min_prompt_tokens": 200000, "prompt": "0.000004", "completion": "0.00002"},
            ]})),
            model("test/weekend-long-context", json!({"prompt": "0.000002", "completion": "0.00001", "overrides": [
                {"min_prompt_tokens": 100000, "utc_days": ["saturday", "sunday"], "prompt": "0.000001"},
            ]})),
            model("test/unknown-condition", json!({"prompt": "0.000002", "completion": "0.00001", "overrides": [
                {"min_prompt_tokens": 100000, "max_prompt_tokens": 500000, "prompt": "0.000004"},
            ]})),
            model("test/time-windows", json!({"prompt": "0.00000028", "completion": "0.00000042", "overrides": [
                {"utc_start": 30, "utc_end": 1630, "prompt": "0.00000028", "completion": "0.00000042"},
                {"utc_start": 1630, "utc_end": 30, "prompt": "0.00000014", "completion": "0.00000021"},
            ]})),
            // Price keys are the base pricing object's keys; the threshold is
            // a number, possibly written as a double.
            model("test/more-prices", json!({"prompt": "0.000002", "completion": "0.00001", "web_search": "0.01", "overrides": [
                {"min_prompt_tokens": 150000.0, "prompt": "0.000004", "web_search": "0.02", "input_audio_cache": "0.000001"},
            ]})),
        ]})
        .to_string(),
    )
    .unwrap();
    let tier = |id: &str| {
        let pricing = catalog.get(id).unwrap().pricing;
        (
            pricing.large_input_threshold,
            pricing.large_input_cost_per_million,
            pricing.input_cost_per_million,
        )
    };
    let (threshold, large_input, _) = tier("test/peak-long-context");
    assert_eq!(threshold, Some(200_000));
    assert!(close(large_input.unwrap(), 4.0));
    assert!(close(
        catalog
            .get("test/peak-long-context")
            .unwrap()
            .pricing
            .large_output_cost_per_million
            .unwrap(),
        20.0
    ));
    for id in [
        "test/weekend-long-context",
        "test/unknown-condition",
        "test/time-windows",
    ] {
        let (threshold, large_input, _) = tier(id);
        assert_eq!((threshold, large_input), (None, None), "{id}");
    }
    assert!(close(tier("test/time-windows").2, 0.28));
    let (threshold, large_input, _) = tier("test/more-prices");
    assert_eq!(threshold, Some(150_000));
    assert!(close(large_input.unwrap(), 4.0));
}

#[test]
fn a_loaded_catalog_validates_slugs_parameters_images_and_mandatory_reasoning() {
    let client = catalog_client();
    let check = |slug: &str, config: &GenerationConfig| {
        prepare(&client, &conversation(slug), config).map(|_| ())
    };
    let unknown = check("openai/gpt-7", &GenerationConfig::default()).unwrap_err();
    assert!(matches!(unknown, Error::ModelNotSupported(_)), "{unknown}");
    check("@preset/team-default", &GenerationConfig::default()).unwrap();
    check("openai/gpt-6-astra:floor", &GenerationConfig::default()).unwrap();

    // Parameters the catalog doesn't list would be silently dropped.
    assert!(
        check(CLAUDE, &GenerationConfig::new(0.2))
            .unwrap_err()
            .to_string()
            .contains("temperature")
    );
    check(GEMINI, &GenerationConfig::new(0.2)).unwrap();
    assert!(
        check(
            "openai/gpt-6-astra",
            &GenerationConfig::default().with_verbosity(Verbosity::Low)
        )
        .is_err()
    );
    let none = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::None));
    assert!(
        check("openai/gpt-6-astra", &none)
            .unwrap_err()
            .to_string()
            .contains("mandatory")
    );
    check(CLAUDE, &none).unwrap();
    let off = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openrouter(OpenRouterReasoning::disabled()));
    assert!(check(GEMINI, &off).is_err());

    // Images need image input.
    let mut images = Conversation::new(Model::openrouter("deepseek/deepseek-v4-flash"));
    images.add_user_message_with_image("look", Image::new_url("https://example.com/a.png"));
    let rejected = prepare(&client, &images, &GenerationConfig::default()).unwrap_err();
    assert!(
        matches!(rejected, Error::UnsupportedFeature(_)),
        "{rejected}"
    );

    // The client hands out catalog models; without a catalog, bare ones.
    assert_eq!(client.model(CLAUDE).unwrap().max_input_tokens, 1_000_000);
    assert!(client.resolve_model("openai/gpt-7").is_err());
    assert_eq!(
        client.resolve_model("@preset/team").unwrap().id,
        "@preset/team"
    );
    assert!(Model::openrouter(CLAUDE).supports_forced_tool_choice(&GenerationConfig::default()));
    client.set_catalog(None);
    assert_eq!(
        client
            .resolve_model("openai/gpt-7")
            .unwrap()
            .max_input_tokens,
        0
    );
    check("openai/gpt-7", &GenerationConfig::default()).unwrap();
}

#[test]
fn dynamic_models_are_not_registry_models() {
    let model = Model::openrouter("~anthropic/claude-sonnet-latest");
    assert_eq!(model.provider.as_telemetry_str(), "openrouter");
    assert!(Model::list(Provider::OpenRouter).is_empty());
    assert!(Model::get(Provider::OpenRouter, &model.id).is_err());
    assert!(Model::find(&model.id).is_none());
    assert!(model.reports_stop_reason() && model.supports_disabling_parallel_tool_calls());
}

#[test]
fn forced_tool_choice_follows_anthropic_rules_for_claude_slugs() {
    let default = GenerationConfig::default();
    let effort = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::High));
    let budget = GenerationConfig::default().with_reasoning(ReasoningConfig::openrouter(
        OpenRouterReasoning::max_tokens(4096),
    ));
    let off = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openrouter(OpenRouterReasoning::disabled()));
    let forced = |slug: &str, config: &GenerationConfig| {
        Model::openrouter(slug).supports_forced_tool_choice(config)
    };
    // Fable 5.1 and Opus 5.5 never take a forced tool.
    assert!(!forced("anthropic/claude-fable-5.1", &default));
    assert!(!forced("anthropic/claude-opus-5.5:nitro", &default));
    // Adaptive models: only a manual budget blocks it, on models that take
    // budgets (Sonnet 5 doesn't, so a budget becomes an effort there).
    assert!(forced("anthropic/claude-sonnet-4.6", &effort));
    assert!(!forced("anthropic/claude-sonnet-4.6", &budget));
    assert!(forced(CLAUDE, &budget));
    // Older Claude thinks with budgets whenever reasoning is on.
    assert!(forced("anthropic/claude-sonnet-4.5", &default));
    assert!(!forced("anthropic/claude-sonnet-4.5", &effort));
    assert!(forced("anthropic/claude-sonnet-4.5", &off));
    // Other vendors and unknown Claude slugs pass through.
    assert!(forced(GPT, &budget));
    assert!(forced("anthropic/claude-future-9", &budget));
    // Catalog aliases resolve through their target.
    let mut alias = Model::openrouter("~anthropic/claude-opus-latest");
    assert!(alias.supports_forced_tool_choice(&default));
    alias.openrouter = Some(OpenRouterModelInfo {
        alias_target: Some("anthropic/claude-opus-5.5".into()),
        ..OpenRouterModelInfo::default()
    });
    assert!(!alias.supports_forced_tool_choice(&default));
}

#[test]
fn sorting_variants_count_as_explicit_routing() {
    for slug in [
        "anthropic/claude-sonnet-5:exacto",
        "anthropic/claude-sonnet-5:nitro",
        "anthropic/claude-sonnet-5:floor",
    ] {
        assert!(
            body(slug, &with_tools()).get("provider").is_none(),
            "{slug}"
        );
    }
    assert_eq!(
        body("anthropic/claude-sonnet-5:thinking", &with_tools())["provider"],
        json!({"sort": "price"})
    );
}

// ---------------------------------------------------------------- responses

fn context(slug: &str) -> ResponseContext {
    ResponseContext::new(Model::openrouter(slug), GenerationConfig::default(), false)
}

/// The documented non-streaming example response.
fn documented_response() -> Value {
    json!({
        "id": "gen-xxxxxxxxxxxxxx",
        "choices": [{
            "finish_reason": "stop", "native_finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello there!"}
        }],
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
            "cost": 0.00014
        },
        "model": "openai/gpt-4o"
    })
}

fn parse(value: &Value, slug: &str) -> Result<crate::providers::GenerationResponse, Error> {
    response::parse_response(&value.to_string(), &context(slug))
}

#[test]
fn the_documented_response_parses_with_reported_cost_and_stop_metadata() {
    let response = parse(&documented_response(), "openai/gpt-4o").unwrap();
    assert_eq!(response.text(), "Hello there!");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(response.refusal.is_none());
    let usage = &response.usage;
    assert_eq!((usage.input_tokens, usage.output_tokens), (10, 4));
    assert_eq!(
        (usage.cached_tokens, usage.reasoning_tokens),
        (Some(0), Some(0))
    );
    assert_eq!(usage.billing, BillingKind::ProviderReported);
    assert_eq!(usage.reported_cost, Some(0.00014));
    assert!(close(usage.billed(), 0.00014));
    assert_eq!(
        usage.total_cost, 0.0,
        "no local estimate when the charge is reported"
    );
    let call = &response.model_call_usage;
    assert_eq!(
        (
            call.call_id.as_str(),
            call.provider.as_str(),
            call.model.as_str()
        ),
        ("gen-xxxxxxxxxxxxxx", "openrouter", "openai/gpt-4o")
    );
    assert_eq!(call.stop_reason.as_deref(), Some("stop"));
    assert_eq!(
        call.stop_details,
        Some(json!({"native_finish_reason": "stop"}))
    );
    assert!(!call.served_model_changed());
}

#[test]
fn reasoning_details_and_tool_calls_are_kept_from_non_streaming_responses() {
    // Documented reasoning_details example, plus the documented tool call shape.
    let details = json!([
        {"type": "reasoning.summary", "summary": "Analyzed the problem by breaking it into components", "id": "reasoning-summary-1", "format": "anthropic-claude-v1", "index": 0},
        {"type": "reasoning.text", "text": "Let me work through this systematically:\n1. First consideration...\n2. Second consideration...", "signature": null, "id": "reasoning-text-1", "format": "anthropic-claude-v1", "index": 1}
    ]);
    let mut value = documented_response();
    value["choices"][0] = json!({
        "finish_reason": "tool_calls", "native_finish_reason": "tool_use",
        "message": {
            "role": "assistant", "content": null, "reasoning": "Analyzed the problem", "reasoning_details": details,
            "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "search_gutenberg_books", "arguments": "{\"search_terms\": [\"James\", \"Joyce\"]}"}}]
        }
    });
    let response = parse(&value, "openai/gpt-4o").unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let call = response.single_tool_call("search_gutenberg_books").unwrap();
    assert_eq!(call.id.as_deref(), Some("call_abc123"));
    assert_eq!(call.arguments, json!({"search_terms": ["James", "Joyce"]}));
    let Some(ReasoningArtifact::OpenRouter { details: kept }) =
        &response.message.reasoning_artifact
    else {
        panic!("expected OpenRouter reasoning");
    };
    assert_eq!(Value::Array(kept.clone()), details);
    // Round-trips through serialization, as saved conversations do.
    let saved = serde_json::to_value(&response.message).unwrap();
    let restored: Message = serde_json::from_value(saved).unwrap();
    assert!(
        matches!(restored.reasoning_artifact, Some(ReasoningArtifact::OpenRouter { ref details }) if details.len() == 2)
    );
}

#[test]
fn finish_reasons_and_refusals_map_to_stop_reasons() {
    let reply = |finish: &str, native: Option<&str>, message: Value| {
        let mut value = documented_response();
        value["choices"][0] =
            json!({"finish_reason": finish, "native_finish_reason": native, "message": message});
        parse(&value, CLAUDE).unwrap()
    };
    let text = |content: &str| json!({"role": "assistant", "content": content});

    // Anthropic classifier refusal: the turn must be dropped from history.
    let classifier = reply(
        "content_filter",
        Some("refusal"),
        json!({"role": "assistant", "content": "Partial", "refusal": "Declined."}),
    );
    assert_eq!(classifier.stop_reason, StopReason::Refusal);
    let refusal = classifier.refusal.as_ref().unwrap();
    assert_eq!(refusal.source, RefusalSource::SafetyClassifier);
    assert_eq!(refusal.explanation.as_deref(), Some("Declined."));
    assert_eq!(refusal.partial_output.as_deref(), Some("Partial"));
    assert_eq!(classifier.text(), "");
    let mut history = Conversation::new(Model::openrouter(CLAUDE));
    history.add_assistant_message_from_response(classifier.clone());
    assert!(history.messages.is_empty());
    assert_eq!(
        classifier.model_call_usage.stop_details,
        Some(json!({"native_finish_reason": "refusal", "refusal": "Declined."}))
    );

    // The model's own refusal keeps the turn; the text isn't message text.
    let model = reply(
        "content_filter",
        Some("content_filter"),
        json!({"role": "assistant", "content": null, "refusal": "I can't help."}),
    );
    assert_eq!(model.refusal.as_ref().unwrap().source, RefusalSource::Model);
    assert_eq!(model.text(), "");
    let structured = reply(
        "stop",
        Some("stop"),
        json!({"role": "assistant", "content": "", "refusal": "No."}),
    );
    assert_eq!(structured.stop_reason, StopReason::Refusal);
    assert_eq!(
        structured.refusal.as_ref().unwrap().explanation.as_deref(),
        Some("No.")
    );

    // A filter around the model.
    let filtered = reply("content_filter", Some("SAFETY"), text("cut"));
    let refusal = filtered.refusal.as_ref().unwrap();
    assert_eq!(refusal.source, RefusalSource::ContentFilter);
    assert_eq!(refusal.partial_output.as_deref(), Some("cut"));
    assert_eq!(
        refusal.provider_details,
        json!({"finish_reason": "content_filter", "native_finish_reason": "SAFETY"})
    );

    assert_eq!(
        reply("stop", Some("stop_sequence"), text("x")).stop_reason,
        StopReason::StopSequence
    );
    assert_eq!(
        reply("weird", None, text("x")).stop_reason,
        StopReason::Other("weird".into())
    );
    let with_call = json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]});
    assert_eq!(
        reply("stop", Some("end_turn"), with_call).stop_reason,
        StopReason::ToolUse
    );

    // A token limit drops a tool call whose arguments were cut off.
    let cut = json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}},
        {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":"}},
    ]});
    let limited = reply("length", Some("max_tokens"), cut);
    assert_eq!(limited.stop_reason, StopReason::MaxTokens);
    assert_eq!(limited.message.tool_calls.len(), 1);
    assert_eq!(limited.message.tool_calls[0].id.as_deref(), Some("a"));
}

#[test]
fn errors_in_a_200_response_are_errors() {
    // No choices: the body holds only an error.
    let only_error = json!({"id": "gen-1", "error": {"code": 502, "message": "Provider returned error", "metadata": {"error_type": "provider_unavailable"}}});
    let error = parse(&only_error, CLAUDE).unwrap_err();
    assert!(
        matches!(&error, Error::ProviderError { status: Some(502), code: Some(code), .. } if code == "provider_unavailable"),
        "{error}"
    );
    assert!(error.is_retryable());
    // The documented partial-output form.
    let partial = json!({"choices": [{
        "message": {"role": "assistant", "content": "partial output..."}, "finish_reason": "error",
        "error": {"code": 502, "message": "Provider disconnected mid-stream", "metadata": {"error_type": "provider_unavailable"}}
    }]});
    assert!(parse(&partial, CLAUDE).unwrap_err().is_retryable());
    let bare = json!({"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "error"}]});
    assert!(parse(&bare, CLAUDE).unwrap_err().is_retryable());
    assert!(parse(&json!({"id": "x", "choices": []}), CLAUDE).is_err());
}

#[test]
fn usage_maps_totals_cache_counts_and_byok_charges() {
    // The documented usage example (BYOK amounts as given there).
    let usage = json!({
        "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 0},
        "cost": 0.95, "cost_details": {"upstream_inference_cost": 19},
        "prompt_tokens": 194, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 100, "audio_tokens": 0},
        "total_tokens": 196
    });
    let model = Model::openrouter(CLAUDE);
    let config = GenerationConfig::default();
    let mut byok = usage.clone();
    byok["is_byok"] = json!(true);
    let parsed = ReportedUsage::parse(&byok);
    assert_eq!(
        parsed,
        ReportedUsage {
            prompt_tokens: 194,
            completion_tokens: 2,
            cached_tokens: Some(0),
            cache_write_tokens: Some(100),
            reasoning_tokens: Some(0),
            cost: Some(0.95),
            is_byok: true,
            upstream_inference_cost: Some(19.0),
        }
    );
    let info = parsed.usage_info(&model, &config, Some("flex"));
    assert_eq!(info.input_tokens, 194, "prompt_tokens is the total input");
    assert_eq!(info.cache_write_tokens, Some(100));
    assert_eq!(info.reported_cost, Some(0.95));
    assert_eq!(info.upstream_cost, Some(19.0));
    assert!(
        close(info.billed(), 19.95),
        "BYOK bills OpenRouter's fee plus the provider's charge"
    );
    assert_eq!(info.service_tier, Some(crate::pricing::ServiceTier::Flex));

    // Without BYOK the upstream cost is OpenRouter's own, not billed to you.
    let info = ReportedUsage::parse(&usage).usage_info(&model, &config, None);
    assert_eq!(info.upstream_cost, None);
    assert!(close(info.billed(), 0.95));

    // No reported cost: estimated from catalog prices.
    let catalog = OpenRouterCatalog::from_json(&catalog_json()).unwrap();
    let priced = catalog.get(CLAUDE).unwrap();
    let mut no_cost = usage.clone();
    no_cost.as_object_mut().unwrap().remove("cost");
    let info = ReportedUsage::parse(&no_cost).usage_info(&priced, &config, None);
    assert_eq!(info.billing, BillingKind::Metered);
    // 94 uncached at $2/M, 100 written at $2.50/M, 2 output at $10/M.
    assert!(
        close(
            info.total_cost,
            (94.0 * 2.0 + 100.0 * 2.5 + 2.0 * 10.0) / 1e6
        ),
        "{}",
        info.total_cost
    );
    assert!(close(info.billed(), info.total_cost));

    // Sums keep both charges.
    let byok = ReportedUsage::parse(&byok).usage_info(&model, &config, None);
    let total: crate::usage::UsageInfo = [byok.clone(), byok].iter().sum();
    assert_eq!(total.upstream_cost, Some(38.0));
    assert!(close(total.billed(), 39.9));
}

#[test]
fn a_different_served_model_is_recorded() {
    let mut value = documented_response();
    value["model"] = json!("anthropic/claude-sonnet-4.5");
    value["provider"] = json!("Anthropic");
    let routed = parse(&value, "openrouter/auto").unwrap();
    assert_eq!(routed.model_call_usage.model, "anthropic/claude-sonnet-4.5");
    assert_eq!(
        routed.model_call_usage.requested_model.as_deref(),
        Some("openrouter/auto")
    );
    assert!(routed.model_call_usage.served_model_changed());
    assert_eq!(
        routed.model_call_usage.upstream_provider.as_deref(),
        Some("Anthropic")
    );

    // Dated snapshots, variants and, without a catalog, same-vendor
    // permaslugs (the documented `anthropic/claude-4.5-sonnet-20250929`) are
    // the same model.
    for (requested, served) in [
        (CLAUDE, "anthropic/claude-sonnet-5-20260630"),
        ("openai/gpt-6-luna:floor", "openai/gpt-6-luna"),
        ("openai/gpt-5.5", "openai/gpt-5.5-2026-04-23"),
        (
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-4.5-sonnet-20250929",
        ),
    ] {
        value["model"] = json!(served);
        assert!(
            !parse(&value, requested)
                .unwrap()
                .model_call_usage
                .served_model_changed(),
            "{requested}"
        );
    }
    // Without a catalog, a different vendor or configured fallbacks count.
    value["model"] = json!("openai/gpt-6-sol");
    assert!(
        parse(&value, CLAUDE)
            .unwrap()
            .model_call_usage
            .served_model_changed()
    );
    value["model"] = json!("anthropic/claude-opus-5");
    let with_fallbacks =
        ResponseContext::new(Model::openrouter(CLAUDE), GenerationConfig::default(), true);
    let response = response::parse_response(&value.to_string(), &with_fallbacks).unwrap();
    assert_eq!(
        response.model_call_usage.requested_model.as_deref(),
        Some(CLAUDE)
    );
    assert!(
        !parse(&value, CLAUDE)
            .unwrap()
            .model_call_usage
            .served_model_changed()
    );

    // With a catalog, the canonical slug is exact and an alias resolving to
    // its target is a change.
    let catalog = OpenRouterCatalog::from_json(&catalog_json()).unwrap();
    let known = ResponseContext::new(
        catalog.get(CLAUDE).unwrap(),
        GenerationConfig::default(),
        false,
    );
    for (served, changed) in [
        ("anthropic/claude-sonnet-5-20260630", false),
        ("anthropic/claude-opus-5", true),
    ] {
        value["model"] = json!(served);
        let response = response::parse_response(&value.to_string(), &known).unwrap();
        assert_eq!(
            response.model_call_usage.served_model_changed(),
            changed,
            "{served}"
        );
    }
    let alias = ResponseContext::new(
        catalog.get("~anthropic/claude-sonnet-latest").unwrap(),
        GenerationConfig::default(),
        false,
    );
    value["model"] = json!(CLAUDE);
    let response = response::parse_response(&value.to_string(), &alias).unwrap();
    assert_eq!(
        response.model_call_usage.requested_model.as_deref(),
        Some("~anthropic/claude-sonnet-latest")
    );
}

// ---------------------------------------------------------------- streaming

fn sse(chunks: &[Value]) -> String {
    chunks
        .iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect()
}

fn chunk(delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": "gen-abc123", "object": "chat.completion.chunk", "created": 1, "model": CLAUDE, "provider": "Anthropic",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "native_finish_reason": finish}]
    })
}

fn usage_chunk(finish: &str) -> Value {
    let mut value = chunk(json!({"content": "", "role": "assistant"}), Some(finish));
    value["usage"] = json!({"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17, "cost": 0.0001,
        "prompt_tokens_details": {"cached_tokens": 8}, "completion_tokens_details": {"reasoning_tokens": 3}});
    value
}

async fn collect(pieces: Vec<Vec<u8>>, slug: &str) -> (Vec<StreamEvent>, Option<Error>) {
    let bytes = futures::stream::iter(pieces.into_iter().map(Ok::<_, std::io::Error>));
    let mut stream = response::stream_events(bytes, context(slug));
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(error) => return (events, Some(error)),
        }
    }
    (events, None)
}

async fn parse_stream(body: String) -> (Vec<StreamEvent>, Option<Error>) {
    collect(vec![body.into_bytes()], CLAUDE).await
}

fn end(events: &[StreamEvent]) -> &crate::providers::GenerationResponse {
    match events.last() {
        Some(StreamEvent::End(response)) => response,
        other => panic!("expected End last, got {other:?}"),
    }
}

fn text_of(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextChunk(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn keep_alives_the_repeated_finish_reason_and_done_yield_one_end() {
    let body = format!(
        ": OPENROUTER PROCESSING\n\n{}: OPENROUTER PROCESSING\n\n{}data: [DONE]\n\n",
        sse(&[
            chunk(json!({"role": "assistant", "content": "Hel"}), None),
            chunk(json!({"content": "lo"}), None)
        ]),
        sse(&[
            chunk(json!({"content": ""}), Some("stop")),
            usage_chunk("stop")
        ]),
    );
    // Delivered in small pieces, as the network may split it.
    let pieces = body.as_bytes().chunks(7).map(<[u8]>::to_vec).collect();
    for (events, error) in [
        parse_stream(body.clone()).await,
        collect(pieces, CLAUDE).await,
    ] {
        assert!(error.is_none(), "{error:?}");
        assert_eq!(text_of(&events), "Hello");
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, StreamEvent::End(_)))
                .count(),
            1
        );
        let response = end(&events);
        assert_eq!(response.text(), "Hello");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(
            (response.usage.input_tokens, response.usage.output_tokens),
            (12, 5)
        );
        assert_eq!(
            (
                response.usage.cached_tokens,
                response.usage.reasoning_tokens
            ),
            (Some(8), Some(3))
        );
        assert_eq!(response.usage.reported_cost, Some(0.0001));
        assert_eq!(response.model_call_usage.call_id, "gen-abc123");
        assert_eq!(
            response.model_call_usage.upstream_provider.as_deref(),
            Some("Anthropic")
        );
    }
}

#[tokio::test]
async fn streams_without_a_terminal_are_truncated() {
    let partial = sse(&[chunk(json!({"content": "Hel"}), None)]);
    let (events, error) = parse_stream(partial.clone()).await;
    assert_eq!(text_of(&events), "Hel");
    assert!(matches!(error, Some(Error::StreamTruncated(_))));
    // A finish reason without its usage chunk or [DONE] is not complete either.
    let (_, error) = parse_stream(sse(&[chunk(json!({"content": "Hi"}), Some("stop"))])).await;
    assert!(matches!(error, Some(Error::StreamTruncated(_))));
    // [DONE] without any finish reason.
    let (_, error) = parse_stream(format!("{partial}data: [DONE]\n\n")).await;
    assert!(
        matches!(error, Some(Error::StreamTruncated(ref message)) if message.contains("finish"))
    );
    // The usage chunk completes a stream that closes before [DONE].
    let (events, error) = parse_stream(sse(&[
        chunk(json!({"content": "Hi"}), Some("stop")),
        usage_chunk("stop"),
    ]))
    .await;
    assert!(error.is_none());
    assert_eq!(end(&events).text(), "Hi");
}

#[tokio::test]
async fn mid_stream_error_chunks_are_errors_after_the_text_so_far() {
    // The documented mid-stream error chunk.
    let documented = json!({"id": "gen-abc123", "object": "chat.completion.chunk", "created": 1234567890, "model": "openai/gpt-4o", "provider": "OpenAI",
        "error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded"}},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]});
    let (events, error) = parse_stream(sse(&[
        chunk(json!({"content": "partial"}), None),
        documented,
    ]))
    .await;
    assert_eq!(text_of(&events), "partial");
    let error = error.unwrap();
    assert!(
        matches!(&error, Error::ProviderError { status: Some(429), code: Some(code), .. } if code == "rate_limit_exceeded"),
        "{error}"
    );
    assert_eq!(error.retry_class(), RetryClass::Transient);

    // An error can be the first and only event; the streaming guide's string code.
    let first = json!({"id": "cmpl-abc123", "object": "chat.completion.chunk", "created": 1234567890, "model": "openai/gpt-4o", "provider": "openai",
        "error": {"code": "server_error", "message": "Provider disconnected unexpectedly"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]});
    let (events, error) = parse_stream(sse(&[first])).await;
    assert!(events.is_empty());
    assert!(error.unwrap().is_retryable());

    // A GPT-6 policy block after the stream started is a refusal.
    let block = json!({"error": {"code": 403, "message": "This content was flagged for possible biological risk.",
        "metadata": {"error_type": "refusal", "provider_code": "bio_policy"}},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]});
    let (_, error) = parse_stream(sse(&[block])).await;
    let Some(Error::Refusal(refusal)) = error else {
        panic!("expected a refusal")
    };
    assert_eq!(refusal.source, RefusalSource::SafetyClassifier);
    assert_eq!(refusal.category.as_deref(), Some("bio_policy"));

    // finish_reason "error" on a choice, with or without its own error.
    let (_, error) = parse_stream(sse(&[chunk(json!({"content": ""}), Some("error"))])).await;
    assert!(error.unwrap().is_retryable());
}

#[tokio::test]
async fn interleaved_tool_call_deltas_are_assembled_by_index() {
    let call = |index: u64, id: Option<&str>, name: Option<&str>, arguments: &str| {
        let mut function = json!({"arguments": arguments});
        if let Some(name) = name {
            function["name"] = json!(name);
        }
        let mut value = json!({"index": index, "type": "function", "function": function});
        if let Some(id) = id {
            value["id"] = json!(id);
        }
        chunk(json!({"tool_calls": [value]}), None)
    };
    let body = format!(
        "{}data: [DONE]\n\n",
        sse(&[
            call(0, Some("call_a"), Some("lookup"), ""),
            call(1, Some("call_b"), Some("fetch"), "{\"url\":"),
            call(0, None, None, "{\"q\":"),
            call(1, None, None, "\"x\"}"),
            call(0, None, None, "\"y\"}"),
            chunk(json!({}), Some("tool_calls")),
            usage_chunk("tool_calls"),
        ])
    );
    let (events, error) = parse_stream(body).await;
    assert!(error.is_none(), "{error:?}");
    let starts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCallStart { index, name } => Some((*index, name.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(starts, [(0, "lookup"), (1, "fetch")]);
    let arguments = |wanted: usize| -> String {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ToolCallArgumentsDelta { index, delta } if *index == wanted => {
                    Some(delta.as_str())
                }
                _ => None,
            })
            .collect()
    };
    assert_eq!(arguments(0), "{\"q\":\"y\"}");
    assert_eq!(arguments(1), "{\"url\":\"x\"}");
    let response = end(&events);
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let calls = &response.message.tool_calls;
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (calls[0].id.as_deref(), calls[0].arguments.clone()),
        (Some("call_a"), json!({"q": "y"}))
    );
    assert_eq!(
        (calls[1].id.as_deref(), calls[1].arguments.clone()),
        (Some("call_b"), json!({"url": "x"}))
    );
}

/// A `delta.tool_calls` chunk; `index` is OpenRouter's raw index.
fn tool_call_chunk(
    index: Option<u64>,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> Value {
    let mut function = json!({"arguments": arguments});
    if let Some(name) = name {
        function["name"] = json!(name);
    }
    let mut value = json!({"type": "function", "function": function});
    if let Some(index) = index {
        value["index"] = json!(index);
    }
    if let Some(id) = id {
        value["id"] = json!(id);
    }
    chunk(json!({"tool_calls": [value]}), None)
}

async fn tool_call_stream(chunks: Vec<Value>, finish: &str) -> Vec<StreamEvent> {
    let mut chunks = chunks;
    chunks.push(chunk(json!({}), Some(finish)));
    chunks.push(usage_chunk(finish));
    let (events, error) = parse_stream(format!("{}data: [DONE]\n\n", sse(&chunks))).await;
    assert!(error.is_none(), "{error:?}");
    events
}

fn tool_call_starts(events: &[StreamEvent]) -> Vec<(usize, &str)> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolCallStart { index, name } => Some((*index, name.as_str())),
            _ => None,
        })
        .collect()
}

/// Every final tool call was announced at its position in
/// `message.tool_calls`, with the arguments streamed under that position.
fn assert_stream_positions_match_final_calls(events: &[StreamEvent]) {
    let starts = tool_call_starts(events);
    for (position, call) in end(events).message.tool_calls.iter().enumerate() {
        let announced: Vec<&str> = starts
            .iter()
            .filter(|(index, _)| *index == position)
            .map(|(_, name)| *name)
            .collect();
        assert_eq!(announced, [call.name.as_str()], "position {position}");
        let streamed: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ToolCallArgumentsDelta { index, delta } if *index == position => {
                    Some(delta.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&streamed).unwrap(),
            call.arguments,
            "position {position}"
        );
    }
}

#[tokio::test]
async fn a_new_id_at_an_index_in_use_starts_a_new_tool_call() {
    // Some upstreams number every call 0; the new id marks the next call.
    let events = tool_call_stream(
        vec![
            tool_call_chunk(Some(0), Some("call_a"), Some("lookup"), "{\"q\":"),
            tool_call_chunk(Some(0), None, None, "\"a\"}"),
            tool_call_chunk(Some(0), Some("call_b"), Some("fetch"), "{\"url\":"),
            tool_call_chunk(Some(0), None, None, "\"b\"}"),
            // A repeated id continues its call.
            tool_call_chunk(Some(0), Some("call_c"), Some("search"), "{\"s\":"),
            tool_call_chunk(Some(0), Some("call_c"), None, "\"c\"}"),
        ],
        "tool_calls",
    )
    .await;
    let calls = &end(&events).message.tool_calls;
    let summary: Vec<_> = calls
        .iter()
        .map(|call| {
            (
                call.id.as_deref(),
                call.name.as_str(),
                call.arguments.clone(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            (Some("call_a"), "lookup", json!({"q": "a"})),
            (Some("call_b"), "fetch", json!({"url": "b"})),
            (Some("call_c"), "search", json!({"s": "c"})),
        ]
    );
    assert_eq!(
        tool_call_starts(&events),
        [(0, "lookup"), (1, "fetch"), (2, "search")]
    );
    assert_stream_positions_match_final_calls(&events);
}

#[tokio::test]
async fn tool_call_events_carry_the_position_in_the_final_calls() {
    // Raw indexes needn't start at 0 or be contiguous (a proxy may pass
    // through the upstream's content-block index).
    let events = tool_call_stream(
        vec![
            tool_call_chunk(Some(1), Some("call_a"), Some("lookup"), "{\"q\":\"a\"}"),
            tool_call_chunk(Some(3), Some("call_b"), Some("fetch"), "{\"url\":"),
            tool_call_chunk(Some(3), None, None, "\"b\"}"),
        ],
        "tool_calls",
    )
    .await;
    assert_eq!(tool_call_starts(&events), [(0, "lookup"), (1, "fetch")]);
    assert_eq!(end(&events).message.tool_calls.len(), 2);
    assert_stream_positions_match_final_calls(&events);

    // A call whose name arrives after a later call started keeps the
    // position of its first delta.
    let events = tool_call_stream(
        vec![
            tool_call_chunk(Some(0), Some("call_a"), None, ""),
            tool_call_chunk(Some(1), Some("call_b"), Some("fetch"), "{}"),
            tool_call_chunk(Some(0), None, Some("lookup"), "{\"q\":\"a\"}"),
        ],
        "tool_calls",
    )
    .await;
    assert_eq!(tool_call_starts(&events), [(1, "fetch"), (0, "lookup")]);
    let names: Vec<_> = end(&events)
        .message
        .tool_calls
        .iter()
        .map(|call| call.name.as_str())
        .collect();
    assert_eq!(names, ["lookup", "fetch"]);
    assert_stream_positions_match_final_calls(&events);
}

#[tokio::test]
async fn calls_cut_off_by_the_token_limit_leave_the_kept_positions_intact() {
    // Interleaved calls: the middle one is unfinished when the limit hits.
    let events = tool_call_stream(
        vec![
            tool_call_chunk(Some(0), Some("call_a"), Some("lookup"), "{\"q\":\"a\"}"),
            tool_call_chunk(Some(1), Some("call_b"), Some("fetch"), "{\"url\":"),
            tool_call_chunk(Some(2), Some("call_c"), Some("search"), "{\"s\":\"c\"}"),
        ],
        "length",
    )
    .await;
    let response = end(&events);
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
    let ids: Vec<_> = response
        .message
        .tool_calls
        .iter()
        .map(|call| call.id.as_deref())
        .collect();
    // The unfinished call and every call after it are dropped.
    assert_eq!(ids, [Some("call_a")]);
    assert_stream_positions_match_final_calls(&events);

    // The same rule applies to non-streaming responses.
    let mut value = documented_response();
    value["choices"][0] = json!({"finish_reason": "length", "native_finish_reason": "max_tokens",
    "message": {"role": "assistant", "content": null, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}},
        {"id": "b", "type": "function", "function": {"name": "fetch", "arguments": "{\"url\":"}},
        {"id": "c", "type": "function", "function": {"name": "search", "arguments": "{}"}},
    ]}});
    let ids: Vec<_> = parse(&value, CLAUDE)
        .unwrap()
        .message
        .tool_calls
        .iter()
        .map(|call| call.id.clone().unwrap())
        .collect();
    assert_eq!(ids, ["a"]);
}

#[tokio::test]
async fn streamed_reasoning_details_merge_by_index_and_only_readable_text_is_shown() {
    let details = |items: Value, reasoning: Option<&str>| {
        let mut delta = json!({"reasoning_details": items});
        if let Some(text) = reasoning {
            delta["reasoning"] = json!(text);
        }
        chunk(delta, None)
    };
    let body = format!(
        "{}data: [DONE]\n\n",
        sse(&[
            details(
                json!([{"type": "reasoning.text", "text": "Let me think ", "signature": null, "id": "reasoning-text-1", "format": "anthropic-claude-v1", "index": 0}]),
                Some("Let me think ")
            ),
            details(
                json!([{"type": "reasoning.text", "text": "step by step.", "signature": null, "index": 0}]),
                Some("step by step.")
            ),
            details(
                json!([{"type": "reasoning.text", "text": "", "signature": "sig-abc", "index": 0, "format": "anthropic-claude-v1"}]),
                None
            ),
            details(
                json!([{"type": "reasoning.encrypted", "data": "abc", "format": "anthropic-claude-v1", "index": 1}]),
                None
            ),
            details(
                json!([{"type": "reasoning.encrypted", "data": "def", "index": 1}]),
                None
            ),
            // Only details, no `reasoning` string: readable text is still streamed.
            details(
                json!([{"type": "reasoning.summary", "summary": "Summary.", "format": "google-gemini-v1", "index": 2}]),
                None
            ),
            chunk(json!({"content": "Answer"}), Some("stop")),
            usage_chunk("stop"),
        ])
    );
    let (events, error) = parse_stream(body).await;
    assert!(error.is_none(), "{error:?}");
    let reasoning: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ReasoningChunk(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning,
        ["Let me think ", "step by step.", "Summary."],
        "encrypted data is never shown"
    );
    let response = end(&events);
    let Some(ReasoningArtifact::OpenRouter { details }) = &response.message.reasoning_artifact
    else {
        panic!("expected OpenRouter reasoning");
    };
    assert_eq!(
        Value::Array(details.clone()),
        json!([
            {"type": "reasoning.text", "text": "Let me think step by step.", "signature": "sig-abc", "id": "reasoning-text-1", "format": "anthropic-claude-v1", "index": 0},
            {"type": "reasoning.encrypted", "data": "abcdef", "format": "anthropic-claude-v1", "index": 1},
            {"type": "reasoning.summary", "summary": "Summary.", "format": "google-gemini-v1", "index": 2},
        ])
    );
    assert_eq!(response.text(), "Answer");
}

#[test]
fn detail_chunks_without_an_index_continue_the_previous_item() {
    let mut merger = DetailsMerger::default();
    merger.merge(&json!({"type": "reasoning.text", "text": "a"}));
    merger.merge(&json!({"type": "reasoning.text", "text": "b", "signature": "s"}));
    merger.merge(&json!({"type": "reasoning.encrypted", "data": "x"}));
    merger.merge(&json!("not an object"));
    assert_eq!(
        merger.finish(),
        [
            json!({"type": "reasoning.text", "text": "ab", "signature": "s"}),
            json!({"type": "reasoning.encrypted", "data": "x"}),
            json!("not an object")
        ]
    );

    // Continuations may omit `type`; `[REDACTED]` placeholders never mix
    // into real data.
    let mut merger = DetailsMerger::default();
    merger.merge(&json!({"type": "reasoning.text", "text": "a", "index": 0}));
    merger.merge(&json!({"text": "b", "signature": "s", "index": 0}));
    merger.merge(&json!({"type": "reasoning.encrypted", "data": "[REDACTED]", "index": 1}));
    merger.merge(&json!({"type": "reasoning.encrypted", "data": "real", "index": 1}));
    merger.merge(&json!({"type": "reasoning.encrypted", "data": "[REDACTED]", "index": 1}));
    merger.merge(&json!({"type": "reasoning.encrypted", "data": "[REDACTED]", "index": 2}));
    assert_eq!(
        merger.finish(),
        [
            json!({"type": "reasoning.text", "text": "ab", "signature": "s", "index": 0}),
            json!({"type": "reasoning.encrypted", "data": "real", "index": 1}),
            json!({"type": "reasoning.encrypted", "data": "[REDACTED]", "index": 2}),
        ]
    );
}

#[tokio::test]
async fn streamed_refusals_are_announced_once_and_never_streamed_as_text() {
    let body = format!(
        "{}data: [DONE]\n\n",
        sse(&[
            chunk(json!({"content": "Partial "}), None),
            chunk(json!({"refusal": "I can't "}), None),
            chunk(json!({"refusal": "help with that."}), None),
            chunk(json!({"content": ""}), Some("content_filter")),
            usage_chunk("content_filter"),
        ])
    );
    let (events, error) = parse_stream(body).await;
    assert!(error.is_none());
    assert_eq!(text_of(&events), "Partial ", "refusal text is not text");
    let refusals: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Refusal(_)))
        .collect();
    assert_eq!(refusals.len(), 1);
    assert!(
        matches!(events[events.len() - 2], StreamEvent::Refusal(_)),
        "Refusal precedes End"
    );
    let refusal = end(&events).refusal.as_ref().unwrap();
    assert_eq!(refusal.source, RefusalSource::Model);
    assert_eq!(
        refusal.explanation.as_deref(),
        Some("I can't help with that.")
    );
}

#[tokio::test]
async fn debug_and_choice_free_chunks_are_skipped() {
    let debug = json!({"id": "gen-xxxxx", "provider": "Anthropic", "model": "anthropic/claude-haiku-4.5", "object": "chat.completion.chunk",
        "created": 1234567890, "choices": [], "debug": {"echo_upstream_body": {"model": "claude-haiku-4-5-20251001", "stream": true}}});
    let body = format!(
        "{}data: [DONE]\n\n",
        sse(&[
            debug,
            chunk(json!({"content": "ok"}), Some("stop")),
            usage_chunk("stop")
        ])
    );
    let (events, error) = parse_stream(body).await;
    assert!(error.is_none());
    assert_eq!(end(&events).text(), "ok");
}

// ------------------------------------------------------------------- errors

fn or_error(status: Option<u16>, retry_after: Option<u64>, body: Value) -> Error {
    openrouter_error(
        status,
        retry_after.map(std::time::Duration::from_secs),
        &body.to_string(),
    )
}

#[test]
fn payment_errors_retry_only_for_the_in_flight_budget() {
    let in_flight = json!({"error": {"code": 402, "message": "This request would exceed your available credits given your current in-flight requests.",
        "metadata": {"reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget", "remedy_hint": "Retry after your in-flight requests settle."}}});
    let error = or_error(Some(402), Some(3), in_flight.clone());
    assert_eq!(error.retry_class(), RetryClass::Transient);
    assert_eq!(error.retry_after(), Some(std::time::Duration::from_secs(3)));
    assert_eq!(
        or_error(Some(402), None, in_flight).retry_class(),
        RetryClass::Permanent
    );
    for source in ["openrouter_credits", "openrouter_key_limit"] {
        let body =
            json!({"error": {"code": 402, "message": "m", "metadata": {"limit_source": source}}});
        let error = or_error(Some(402), Some(3), body);
        assert_eq!(error.retry_class(), RetryClass::Permanent, "{source}");
        assert_eq!(error.retry_after(), None);
    }
}

#[test]
fn refusal_and_moderation_errors_become_refusals() {
    let refusal = or_error(
        Some(403),
        None,
        json!({"error": {"code": 403, "message": "Claude refused.", "metadata": {"error_type": "refusal", "provider_name": "Anthropic"}}}),
    );
    let Error::Refusal(details) = &refusal else {
        panic!("{refusal}")
    };
    assert_eq!(details.source, RefusalSource::SafetyClassifier);
    assert_eq!(details.explanation.as_deref(), Some("Claude refused."));
    assert!(!refusal.is_retryable());

    let filtered = or_error(
        Some(403),
        None,
        json!({"error": {"code": 403, "message": "Blocked", "metadata": {"error_type": "content_policy_violation", "provider_code": "SAFETY"}}}),
    );
    assert!(
        matches!(&filtered, Error::Refusal(r) if r.source == RefusalSource::ContentFilter && r.category.as_deref() == Some("SAFETY"))
    );

    // Documented moderation metadata.
    let moderation = or_error(
        Some(403),
        None,
        json!({"error": {"code": 403, "message": "Input flagged", "metadata": {
        "reasons": ["harassment", "violence"], "flagged_input": "...", "provider_name": "OpenAI", "model_slug": "openai/gpt-4o"}}}),
    );
    assert!(
        matches!(&moderation, Error::Refusal(r) if r.source == RefusalSource::ContentFilter && r.category.as_deref() == Some("harassment, violence"))
    );

    // A guardrail block is a permission error, not a refusal.
    let guardrail = or_error(
        Some(403),
        None,
        json!({"error": {"code": 403, "message": "Blocked by guardrail", "metadata": {"error_type": "permission_denied"}}}),
    );
    assert!(matches!(
        guardrail,
        Error::ProviderError {
            status: Some(403),
            ..
        }
    ));
    assert!(!guardrail.is_retryable());
}

#[test]
fn routing_requirement_failures_are_permanent_but_outages_are_not() {
    let unroutable = or_error(
        Some(503),
        Some(1),
        json!({"error": {"code": 503, "message": "No available model provider that meets your routing requirements"}}),
    );
    assert!(
        matches!(&unroutable, Error::ProviderError { code: Some(code), .. } if code == "no_eligible_provider")
    );
    assert_eq!(unroutable.retry_class(), RetryClass::Permanent);
    let not_allowed = or_error(
        Some(404),
        None,
        json!({"error": {"code": 404, "message": "No allowed providers are available for the selected model."}}),
    );
    assert_eq!(not_allowed.retry_class(), RetryClass::Permanent);
    let overloaded = or_error(
        Some(503),
        Some(1),
        json!({"error": {"code": 503, "message": "Overloaded", "metadata": {"error_type": "provider_overloaded"}}}),
    );
    assert_eq!(overloaded.retry_class(), RetryClass::Capacity);
    let limited = or_error(
        Some(429),
        Some(60),
        json!({"error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded", "provider_code": "rate_limited"}}}),
    );
    assert_eq!(limited.retry_class(), RetryClass::Transient);
    assert_eq!(
        limited.retry_after(),
        Some(std::time::Duration::from_secs(60))
    );
    // Masked 500s keep their error type.
    let server = or_error(
        Some(500),
        None,
        json!({"error": {"code": 500, "message": "Internal Server Error", "metadata": {"error_type": "server"}}}),
    );
    assert!(server.is_retryable());
    let context = or_error(
        Some(400),
        None,
        json!({"error": {"code": 400, "message": "too long", "metadata": {"error_type": "context_length_exceeded"}}}),
    );
    assert!(!context.is_retryable());
}
