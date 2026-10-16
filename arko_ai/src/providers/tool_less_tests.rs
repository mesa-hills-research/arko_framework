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

/// A conversation that called a tool, continued without tools (an agent
/// switched to a chat): the ask, a reply with its reasoning, text and a
/// call, the call's result with an image and the user's next words beside
/// it, and the reply after.
fn conversation_with_tool_history(model: Model) -> Conversation {
    use crate::message::{Content, Image, Message};
    use crate::reasoning::ReasoningArtifact;
    use crate::tool::{ToolCall, ToolResult, ToolResultImage};

    let mut conversation = conversation(model);
    let call = ToolCall::from_argument_text(
        Some("call_a".into()),
        "read_file".into(),
        r#"{"path":"a.rs"}"#,
    );
    let mut reply = Message::new_assistant("Reading it.");
    reply.tool_calls = vec![call.clone()];
    // An OpenAI response's output items replay the call itself.
    if conversation.model.provider == Provider::OpenAI {
        reply.reasoning_artifact = Some(ReasoningArtifact::OpenAIOutputItems {
            items: vec![serde_json::json!({
                "type": "function_call", "id": "fc_a", "call_id": "call_a",
                "name": "read_file", "arguments": "{\"path\":\"a.rs\"}",
            })],
        });
    }
    conversation.messages.push(reply);
    let result =
        ToolResult::from_tool_call(&call, "fn main() {}").with_images(vec![ToolResultImage::new(
            "a.png, 1 KB",
            Image::new_base64("cGl4ZWxz", "image/png"),
        )]);
    conversation.add_user_message_with_contents(vec![
        Content::ToolResult(result),
        Content::Text("Thanks.".into()),
    ]);
    conversation.add_assistant_message("It is an empty program.");
    conversation.add_user_message("Explain it.");
    conversation
}

fn body_of(conversation: &Conversation, config: &GenerationConfig, oauth: bool) -> String {
    match conversation.model.provider {
        Provider::OpenAI => OpenAIClient::new("test").request_body(conversation, config, oauth),
        Provider::Anthropic => AnthropicClient::new("test").request_body(conversation, config),
        Provider::Google => GoogleClient::new("test").request_body(conversation, config),
        Provider::OpenRouter => OpenRouterClient::new("test")
            .build_request(conversation, config)
            .map(|prepared| serde_json::to_string(&prepared.body).unwrap()),
        Provider::ClaudeCode => unreachable!("Claude Code models are run by the CLI"),
    }
    .unwrap()
}

/// Written as text, a conversation that called tools goes to every
/// provider without tools: no call or result items, and no reasoning of the
/// turn that made the call (an OpenAI output item replays the call itself).
/// The result's image follows its text.
#[test]
fn tool_history_as_text_sends_no_call_or_result_items() {
    for (model, oauth) in routes() {
        let conversation = conversation_with_tool_history(model.clone());
        let sent = body_of(&conversation.tool_history_as_text(), &base(&model), oauth);
        for item in [
            "tool_use",
            "tool_result",
            "\"function_call\"",
            "function_call_output",
            "function_result",
            "tool_calls",
            "\"role\":\"tool\"",
            "fc_a",
        ] {
            assert!(!sent.contains(item), "{} {item}: {sent}", model.id);
        }
        for text in [
            "Reading it.",
            r#"[Tool call read_file: {\"path\":\"a.rs\"}]"#,
            r#"[Result of read_file]\nfn main() {}"#,
            "a.png, 1 KB",
            "cGl4ZWxz",
            "Thanks.",
        ] {
            assert!(sent.contains(text), "{} {text}: {sent}", model.id);
        }
    }
}

/// Only the copy is written as text; a conversation without tool calls is
/// used as it is.
#[test]
fn tool_history_as_text_leaves_the_conversation_as_it_is() {
    let model = Model::get(Provider::OpenAI, "gpt-6-astra").unwrap();
    let with_calls = conversation_with_tool_history(model.clone());
    let flattened = with_calls.tool_history_as_text().into_owned();
    assert_eq!(with_calls.messages[1].tool_calls.len(), 1);
    assert!(with_calls.messages[1].reasoning_artifact.is_some());
    let reply = &flattened.messages[1];
    assert!(reply.tool_calls.is_empty());
    assert!(reply.reasoning_artifact.is_none());
    let texts: Vec<_> = reply
        .content
        .iter()
        .filter_map(crate::message::Content::as_text)
        .collect();
    assert_eq!(
        texts,
        ["Reading it.", r#"[Tool call read_file: {"path":"a.rs"}]"#]
    );
    assert!(matches!(
        &flattened.messages[2].content[..],
        [
            crate::message::Content::Text(output),
            crate::message::Content::Text(label),
            crate::message::Content::Image(_),
            crate::message::Content::Text(thanks),
        ] if output == "[Result of read_file]\nfn main() {}" && label == "a.png, 1 KB" && thanks == "Thanks."
    ));
    let plain = conversation(model);
    assert!(matches!(
        plain.tool_history_as_text(),
        std::borrow::Cow::Borrowed(_)
    ));
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
