//! Requests whose tool configuration lists no tools: every provider sends
//! them as it sends a request without a tool configuration.

use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    model::{Model, Provider},
    providers::{
        anthropic::AnthropicClient, google::GoogleClient, openai::OpenAIClient,
        openrouter::OpenRouterClient,
    },
    reasoning::{GeminiThinkingLevel, ReasoningConfig},
    tool::{ToolChoice, ToolConfig},
};
use serde_json::Value;

fn conversation(model: Model) -> Conversation {
    let mut conversation = Conversation::new(model)
        .with_prompt_cache_key("pinned")
        .with_system_message("You are a helpful assistant.");
    conversation.add_user_message("Hello.");
    conversation
}

/// The request body for `model` under `config`, as sent.
fn body(model: Model, config: &GenerationConfig, oauth: bool) -> String {
    let conversation = conversation(model);
    match conversation.model.provider {
        Provider::OpenAI => OpenAIClient::new("test").request_body(&conversation, config, oauth),
        Provider::Anthropic => AnthropicClient::new("test").request_body(&conversation, config),
        Provider::Google => GoogleClient::new("test").request_body(&conversation, config),
        Provider::OpenRouter => OpenRouterClient::new("test")
            .build_request(&conversation, config)
            .map(|prepared| serde_json::to_string(&prepared.body).unwrap()),
        Provider::ClaudeCode => unreachable!("Claude Code models are run by the CLI"),
    }
    .unwrap()
}

/// Every route that sends tools in its request body.
fn routes() -> Vec<(Model, bool)> {
    vec![
        (Model::get(Provider::OpenAI, "gpt-6-astra").unwrap(), false),
        (Model::get(Provider::OpenAI, "gpt-6-astra").unwrap(), true),
        (
            Model::get(Provider::Anthropic, "claude-opus-5-5").unwrap(),
            false,
        ),
        (
            Model::get(Provider::Anthropic, "claude-fable-5-1").unwrap(),
            false,
        ),
        (
            Model::get(Provider::Google, "gemini-3.5-flash").unwrap(),
            false,
        ),
        (Model::openrouter("anthropic/claude-opus-5.5"), false),
        (Model::openrouter("openai/gpt-6-astra"), false),
    ]
}

/// The base configuration, with an explicit Gemini level so the body
/// doesn't depend on the default-level rule.
fn base(model: &Model) -> GenerationConfig {
    if model.uses_gemini_thinking_levels() {
        GenerationConfig::default()
            .with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Medium, false))
    } else {
        GenerationConfig::default()
    }
}

/// An empty tool list sends no `tools`, no tool choice and no parallel-call
/// setting (providers refuse a tool choice without tools): the request is
/// the one sent without a tool configuration, byte for byte. Gemini, which
/// can't turn parallel calls off, doesn't refuse the setting either when
/// there are no tools to call.
#[test]
fn an_empty_tool_list_sends_no_tool_fields() {
    for (model, oauth) in routes() {
        let without = body(model.clone(), &base(&model), oauth);
        for choice in [
            None,
            Some(ToolChoice::Auto),
            Some(ToolChoice::None),
            Some(ToolChoice::Required),
            Some(ToolChoice::Specific("lookup".into())),
        ] {
            let mut tools = ToolConfig::new(Vec::new()).with_parallel_tool_calls(false);
            tools.tool_choice = choice.clone();
            let config = base(&model).with_tool_config(tools);
            let sent = body(model.clone(), &config, oauth);
            assert_eq!(sent, without, "{} {choice:?}", model.id);
            let value: Value = serde_json::from_str(&sent).unwrap();
            for field in ["tools", "tool_choice", "parallel_tool_calls"] {
                assert!(value.get(field).is_none(), "{} {field}", model.id);
            }
        }
    }
}

/// `offered_tools` is the configuration only while it lists a tool.
#[test]
fn offered_tools_needs_a_tool() {
    let none = GenerationConfig::default();
    assert!(none.offered_tools().is_none());
    let empty = GenerationConfig::default().with_tool_config(ToolConfig::new(Vec::new()));
    assert!(empty.offered_tools().is_none());
    let one = GenerationConfig::default().with_tool_config(ToolConfig::new(vec![
        crate::tool::ToolDefinition::new(
            "lookup",
            "Look up",
            serde_json::json!({"type": "object"}),
        ),
    ]));
    assert_eq!(one.offered_tools().unwrap().tools.len(), 1);
}
