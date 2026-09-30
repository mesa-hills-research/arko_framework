//! Where tool-result images go in each provider's request (TI-1), pinned
//! against the requests sent before result images existed.

use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message},
    model::{Model, Provider},
    providers::{
        anthropic::AnthropicClient, google::GoogleClient, openai::OpenAIClient,
        openrouter::OpenRouterClient,
    },
    reasoning::{GeminiThinkingLevel, ReasoningConfig},
    tool::{ToolCall, ToolResult, ToolResultImage},
};
use serde_json::{Value, json};

fn calls() -> Vec<ToolCall> {
    ["call_a", "call_b"]
        .into_iter()
        .map(|id| {
            ToolCall::from_argument_text(
                Some(id.into()),
                "view_image".into(),
                r#"{"path":"a.png"}"#,
            )
        })
        .collect()
}

fn pixels() -> Image {
    Image::new_base64("cGl4ZWxz", "image/png")
}

/// How the tool-result message is built.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    /// Results only.
    ResultsOnly,
    /// Results, an answer text, and a labelled image as sibling content: the
    /// layout the arko_agents runtime writes, and conversations saved before
    /// result images existed hold.
    Siblings,
    /// The answer text as sibling content, and the image inside call_a's
    /// result.
    InResult,
}

fn conversation(model: Model, layout: Layout) -> Conversation {
    let mut conversation = Conversation::new(model).with_prompt_cache_key("pinned");
    conversation.add_user_message("Look at these.");
    let mut assistant = Message::new_assistant("Inspecting.");
    assistant.tool_calls = calls();
    conversation.messages.push(assistant);
    let calls = calls();
    let mut first = ToolResult::from_tool_call(&calls[0], "a.png, 12 × 8");
    if layout == Layout::InResult {
        first = first.with_images(vec![ToolResultImage::new("a.png", pixels())]);
    }
    let mut content = vec![
        Content::ToolResult(first),
        Content::ToolResult(ToolResult::error_from_tool_call(&calls[1], "not found")),
    ];
    match layout {
        Layout::ResultsOnly => {}
        Layout::Siblings => content.extend([
            Content::Text("Answer: yes".into()),
            Content::Text("Image from tool view_image, call call_a: a.png".into()),
            Content::Image(pixels()),
        ]),
        Layout::InResult => content.push(Content::Text("Answer: yes".into())),
    }
    conversation.add_user_message_with_contents(content);
    conversation.add_assistant_message("Done.");
    conversation
}

fn model(provider: Provider, id: &str) -> Model {
    Model::get(provider, id).unwrap()
}

fn gpt() -> Model {
    model(Provider::OpenAI, "gpt-6-astra")
}

fn claude() -> Model {
    model(Provider::Anthropic, "claude-opus-5-5")
}

fn gemini_3() -> Model {
    model(Provider::Google, "gemini-3.5-flash")
}

fn gemini_2() -> Model {
    model(Provider::Google, "gemini-2.5-pro")
}

/// Request bodies exactly as sent. Gemini 3 is asked for an explicit level
/// so the body doesn't depend on the default-level rule.
fn openai_body(conversation: &Conversation, oauth: bool) -> Result<String, Error> {
    OpenAIClient::new("test").request_body(conversation, &GenerationConfig::default(), oauth)
}

fn anthropic_body(conversation: &Conversation) -> Result<String, Error> {
    AnthropicClient::new("test").request_body(conversation, &GenerationConfig::default())
}

fn google_body(conversation: &Conversation) -> Result<String, Error> {
    let config = if conversation.model.uses_gemini_thinking_levels() {
        GenerationConfig::default()
            .with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Medium, false))
    } else {
        GenerationConfig::default()
    };
    GoogleClient::new("test").request_body(conversation, &config)
}

fn openrouter_body(conversation: &Conversation) -> Result<String, Error> {
    let prepared =
        OpenRouterClient::new("test").build_request(conversation, &GenerationConfig::default())?;
    Ok(serde_json::to_string(&prepared.body).unwrap())
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap()
}

// Captured from the base commit, before `ToolResult::images` existed.
const OPENAI_KEY: &str = r#"{"model":"gpt-6-astra","store":false,"include":["reasoning.encrypted_content"],"input":[{"role":"user","type":"message","content":"Look at these."},{"type":"function_call","call_id":"call_a","name":"view_image","arguments":"{\"path\":\"a.png\"}"},{"type":"function_call","call_id":"call_b","name":"view_image","arguments":"{\"path\":\"a.png\"}"},{"role":"assistant","type":"message","content":[{"type":"output_text","text":"Inspecting."}]},{"type":"function_call_output","call_id":"call_a","output":"a.png, 12 × 8"},{"type":"function_call_output","call_id":"call_b","output":"not found"},{"role":"user","type":"message","content":[{"type":"input_text","text":"Answer: yes"},{"type":"input_text","text":"Image from tool view_image, call call_a: a.png"},{"type":"input_image","image_url":"data:image/png;base64,cGl4ZWxz"}]},{"role":"assistant","type":"message","content":[{"type":"output_text","text":"Done."}]}],"max_output_tokens":128000,"stream":false,"prompt_cache_key":"pinned","prompt_cache_options":{"mode":"implicit"}}"#;
const OPENAI_OAUTH: &str = r#"{"model":"gpt-6-astra","store":false,"reasoning":{"effort":"medium","context":"all_turns"},"include":["reasoning.encrypted_content"],"input":[{"role":"user","type":"message","content":"Look at these."},{"type":"function_call","call_id":"call_a","name":"view_image","arguments":"{\"path\":\"a.png\"}"},{"type":"function_call","call_id":"call_b","name":"view_image","arguments":"{\"path\":\"a.png\"}"},{"role":"assistant","type":"message","content":[{"type":"output_text","text":"Inspecting."}]},{"type":"function_call_output","call_id":"call_a","output":"a.png, 12 × 8"},{"type":"function_call_output","call_id":"call_b","output":"not found"},{"role":"user","type":"message","content":[{"type":"input_text","text":"Answer: yes"},{"type":"input_text","text":"Image from tool view_image, call call_a: a.png"},{"type":"input_image","image_url":"data:image/png;base64,cGl4ZWxz"}]},{"role":"assistant","type":"message","content":[{"type":"output_text","text":"Done."}]}],"stream":true,"prompt_cache_key":"pinned"}"#;
const ANTHROPIC: &str = r#"{"model":"claude-opus-5-5","max_tokens":128000,"messages":[{"role":"user","content":[{"type":"text","text":"Look at these."}]},{"role":"assistant","content":[{"type":"text","text":"Inspecting."},{"type":"tool_use","id":"call_a","name":"view_image","input":{"path":"a.png"}},{"type":"tool_use","id":"call_b","name":"view_image","input":{"path":"a.png"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":"a.png, 12 × 8"},{"type":"tool_result","tool_use_id":"call_b","content":"not found","is_error":true},{"type":"text","text":"Answer: yes"},{"type":"text","text":"Image from tool view_image, call call_a: a.png"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"cGl4ZWxz"}}]},{"role":"assistant","content":[{"type":"text","text":"Done."}]}]}"#;
const GEMINI_2_5: &str = r#"{"model":"gemini-2.5-pro","store":false,"input":[{"type":"user_input","content":[{"type":"text","text":"Look at these."}]},{"type":"model_output","content":[{"type":"text","text":"Inspecting."}]},{"type":"function_call","id":"call_a","name":"view_image","arguments":{"path":"a.png"}},{"type":"function_call","id":"call_b","name":"view_image","arguments":{"path":"a.png"}},{"type":"function_result","name":"view_image","call_id":"call_a","result":[{"type":"text","text":"a.png, 12 × 8"}]},{"type":"function_result","name":"view_image","call_id":"call_b","is_error":true,"result":[{"type":"text","text":"not found"}]},{"type":"user_input","content":[{"type":"text","text":"Answer: yes"},{"type":"text","text":"Image from tool view_image, call call_a: a.png"},{"type":"image","mime_type":"image/png","data":"cGl4ZWxz"}]},{"type":"model_output","content":[{"type":"text","text":"Done."}]}],"generation_config":{"max_output_tokens":65536}}"#;
const GEMINI_3: &str = r#"{"model":"gemini-3.5-flash","store":false,"input":[{"type":"user_input","content":[{"type":"text","text":"Look at these."}]},{"type":"model_output","content":[{"type":"text","text":"Inspecting."}]},{"type":"function_call","id":"call_a","name":"view_image","arguments":{"path":"a.png"}},{"type":"function_call","id":"call_b","name":"view_image","arguments":{"path":"a.png"}},{"type":"function_result","name":"view_image","call_id":"call_a","result":[{"type":"text","text":"a.png, 12 × 8"}]},{"type":"function_result","name":"view_image","call_id":"call_b","is_error":true,"result":[{"type":"text","text":"not found"}]},{"type":"model_output","content":[{"type":"text","text":"Done."}]}],"generation_config":{"max_output_tokens":65536,"thinking_level":"medium","thinking_summaries":"none"}}"#;
const OPENROUTER: &str = r#"{"model":"anthropic/claude-opus-5.5","messages":[{"content":"Look at these.","role":"user"},{"content":"Inspecting.","role":"assistant","tool_calls":[{"function":{"arguments":"{\"path\":\"a.png\"}","name":"view_image"},"id":"call_a","type":"function"},{"function":{"arguments":"{\"path\":\"a.png\"}","name":"view_image"},"id":"call_b","type":"function"}]},{"content":"a.png, 12 × 8","role":"tool","tool_call_id":"call_a"},{"content":"not found","role":"tool","tool_call_id":"call_b"},{"content":[{"text":"Answer: yes","type":"text"},{"text":"Image from tool view_image, call call_a: a.png","type":"text"},{"image_url":{"url":"data:image/png;base64,cGl4ZWxz"},"type":"image_url"}],"role":"user"},{"content":"Done.","role":"assistant"}],"session_id":"pinned","prompt_cache_key":"pinned","cache_control":{"type":"ephemeral"}}"#;

/// Messages without result images, including conversations saved before
/// result images existed (sibling images), are sent byte for byte as
/// before.
#[test]
fn requests_without_result_images_are_unchanged() {
    let siblings = |model| conversation(model, Layout::Siblings);
    assert_eq!(openai_body(&siblings(gpt()), false).unwrap(), OPENAI_KEY);
    assert_eq!(openai_body(&siblings(gpt()), true).unwrap(), OPENAI_OAUTH);
    assert_eq!(anthropic_body(&siblings(claude())).unwrap(), ANTHROPIC);
    assert_eq!(google_body(&siblings(gemini_2())).unwrap(), GEMINI_2_5);
    assert_eq!(
        google_body(&conversation(gemini_3(), Layout::ResultsOnly)).unwrap(),
        GEMINI_3
    );
    assert_eq!(
        openrouter_body(&siblings(Model::openrouter("anthropic/claude-opus-5.5"))).unwrap(),
        OPENROUTER
    );
}

/// A conversation saved before result images existed loads, serializes
/// without the new field, and is sent as before.
#[test]
fn saved_conversations_without_result_images_load_and_send_as_before() {
    let saved = serde_json::to_value(conversation(gpt(), Layout::Siblings)).unwrap();
    let text = saved.to_string();
    assert!(!text.contains("\"images\""), "{text}");
    // An older result, written without the field.
    let result = &saved["messages"][2]["content"][0]["ToolResult"];
    assert_eq!(
        result,
        &json!({"call_id": "call_a", "name": "view_image", "output": "a.png, 12 × 8", "is_error": false})
    );
    let restored: Conversation = serde_json::from_value(saved).unwrap();
    assert!(
        restored.messages[2].content[0]
            .as_tool_result()
            .unwrap()
            .images
            .is_empty()
    );
    assert_eq!(openai_body(&restored, false).unwrap(), OPENAI_KEY);
}

fn output_of<'a>(input: &'a Value, call_id: &str) -> &'a Value {
    input
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call_output" && item["call_id"] == call_id)
        .map(|item| &item["output"])
        .unwrap()
}

/// OpenAI (API key and ChatGPT OAuth): the image goes in the call's
/// `function_call_output.output`, after the output and its label.
#[test]
fn openai_sends_result_images_inside_the_function_call_output() {
    for oauth in [false, true] {
        let body = json(&openai_body(&conversation(gpt(), Layout::InResult), oauth).unwrap());
        let input = &body["input"];
        assert_eq!(
            output_of(input, "call_a"),
            &json!([
                {"type": "input_text", "text": "a.png, 12 × 8"},
                {"type": "input_text", "text": "a.png"},
                {"type": "input_image", "image_url": "data:image/png;base64,cGl4ZWxz"},
            ])
        );
        assert_eq!(output_of(input, "call_b"), "not found");
        // Only the sibling text follows the results.
        let trailing = &input[6];
        assert_eq!(trailing["role"], "user");
        assert_eq!(trailing["content"], "Answer: yes");
    }
}

#[test]
fn anthropic_sends_result_images_inside_the_tool_result() {
    let mut conversation = conversation(claude(), Layout::InResult);
    // An empty output sends no empty text block.
    let result = conversation.messages[2].content[1].clone();
    let Content::ToolResult(mut empty) = result else {
        unreachable!()
    };
    empty.output.clear();
    empty.images = vec![ToolResultImage::new("", pixels())];
    conversation.messages[2].content[1] = Content::ToolResult(empty);
    let body = json(&anthropic_body(&conversation).unwrap());
    let blocks = &body["messages"][2]["content"];
    let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cGl4ZWxz"}});
    assert_eq!(
        blocks[0],
        json!({"type": "tool_result", "tool_use_id": "call_a", "content": [
            {"type": "text", "text": "a.png, 12 × 8"},
            {"type": "text", "text": "a.png"},
            image,
        ]})
    );
    assert_eq!(
        blocks[1],
        json!({"type": "tool_result", "tool_use_id": "call_b", "is_error": true, "content": [image]})
    );
    assert_eq!(blocks[2], json!({"type": "text", "text": "Answer: yes"}));
    assert_eq!(blocks.as_array().unwrap().len(), 3);
}

#[test]
fn gemini_3_sends_result_images_inside_the_function_result() {
    let body = json(&google_body(&conversation(gemini_3(), Layout::InResult)).unwrap());
    let input = body["input"].as_array().unwrap();
    assert_eq!(
        input[4]["result"],
        json!([
            {"type": "text", "text": "a.png, 12 × 8"},
            {"type": "text", "text": "a.png"},
            {"type": "image", "mime_type": "image/png", "data": "cGl4ZWxz"},
        ])
    );
    // The answer text is folded into the last result (TI-3).
    assert_eq!(
        input[5]["result"],
        json!([{"type": "text", "text": "not found\n\nAnswer: yes"}])
    );
    assert_eq!(input[6]["type"], "model_output");
}

/// Gemini 2.x, Gemini ids whose version can't be read, and OpenRouter take
/// no images in results: they follow the results after the runtime's label,
/// byte for byte the layout the runtime writes today (the answer text, then
/// the labelled image).
#[test]
fn other_models_get_result_images_after_the_results_as_before() {
    let mut unknown_gemini = gemini_3();
    unknown_gemini.id = "gemini-exp-1206".into();
    let same = |model: Model, body: fn(&Conversation) -> Result<String, Error>| {
        assert!(!model.accepts_tool_result_images(), "{}", model.id);
        assert_eq!(
            body(&conversation(model.clone(), Layout::InResult)).unwrap(),
            body(&conversation(model.clone(), Layout::Siblings)).unwrap(),
            "{}",
            model.id
        );
    };
    same(gemini_2(), google_body);
    same(unknown_gemini, google_body);
    same(
        Model::openrouter("anthropic/claude-opus-5.5"),
        openrouter_body,
    );
    same(Model::openrouter("openai/gpt-6-astra"), openrouter_body);
    assert_eq!(
        google_body(&conversation(gemini_2(), Layout::InResult)).unwrap(),
        GEMINI_2_5
    );
    // An image with no label still names its tool and call.
    let mut conversation = conversation(gemini_2(), Layout::ResultsOnly);
    let Content::ToolResult(result) = &mut conversation.messages[2].content[1] else {
        unreachable!()
    };
    result.call_id = None;
    result.images = vec![ToolResultImage::new("", pixels())];
    let body = json(&google_body(&conversation).unwrap());
    assert_eq!(
        body["input"][6]["content"][0]["text"],
        "Image from tool view_image, call unidentified: "
    );
}

/// Result images count as image input: a model without it rejects them
/// before sending (the only case where OpenAI and Anthropic would not take
/// them inside the result).
#[test]
fn models_without_image_input_reject_result_images() {
    for mut model in [
        gpt(),
        claude(),
        gemini_3(),
        Model::openrouter("openai/gpt-6-astra"),
    ] {
        model.supports_image_input = false;
        assert!(!model.accepts_tool_result_images());
        let conversation = conversation(model.clone(), Layout::InResult);
        let error = match model.provider {
            Provider::OpenAI => openai_body(&conversation, false),
            Provider::Anthropic => anthropic_body(&conversation),
            Provider::Google => google_body(&conversation),
            Provider::OpenRouter => openrouter_body(&conversation),
            Provider::ClaudeCode => unreachable!("not listed"),
        }
        .unwrap_err();
        assert!(
            matches!(error, Error::UnsupportedFeature(_)),
            "{}: {error}",
            model.id
        );
    }
}

/// Gemini 3.x: the rest of a tool-result message is folded into its last
/// result, with no `user_input` after the results (TI-3).
mod gemini_3_fold {
    use super::*;

    fn text(text: &str) -> Value {
        json!({"type": "text", "text": text})
    }

    fn image() -> Value {
        json!({"type": "image", "mime_type": "image/png", "data": "cGl4ZWxz"})
    }

    fn results_message(extra: Vec<Content>, results: usize) -> Conversation {
        let mut conversation = Conversation::new(gemini_3()).with_prompt_cache_key("pinned");
        conversation.add_user_message("Look at these.");
        let calls: Vec<ToolCall> = (0..results)
            .map(|index| {
                ToolCall::from_argument_text(Some(format!("call_{index}")), "tool".into(), "{}")
            })
            .collect();
        let mut assistant = Message::new_assistant("");
        assistant.tool_calls = calls.clone();
        conversation.messages.push(assistant);
        let mut content: Vec<Content> = calls
            .iter()
            .map(|call| {
                Content::ToolResult(ToolResult::from_tool_call(
                    call,
                    &format!("out {}", call.id.as_deref().unwrap()),
                ))
            })
            .collect();
        content.extend(extra);
        conversation.add_user_message_with_contents(content);
        conversation
    }

    /// The steps after the model's function calls.
    fn after_calls(conversation: &Conversation) -> Vec<Value> {
        let body = json(&google_body(conversation).unwrap());
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .skip_while(|step| step["type"] != "function_result")
            .cloned()
            .collect()
    }

    fn results_of(steps: &[Value]) -> Vec<Value> {
        assert!(
            steps.iter().all(|step| step["type"] == "function_result"),
            "no user_input after the results: {steps:?}"
        );
        steps.iter().map(|step| step["result"].clone()).collect()
    }

    #[test]
    fn text_is_appended_to_the_last_result_after_a_blank_line() {
        let conversation = results_message(
            vec![
                Content::Text("The user answered: yes".into()),
                Content::Text("# Reminder\nKeep going.".into()),
            ],
            1,
        );
        assert_eq!(
            results_of(&after_calls(&conversation)),
            [json!([text(
                "out call_0\n\nThe user answered: yes\n\n# Reminder\nKeep going."
            )])]
        );
    }

    #[test]
    fn images_follow_the_last_result_with_their_labels() {
        let conversation = results_message(
            vec![
                Content::Text("Image from tool tool, call call_0: a.png".into()),
                Content::Image(pixels()),
                Content::Image(pixels()),
            ],
            1,
        );
        assert_eq!(
            results_of(&after_calls(&conversation)),
            [json!([
                text("out call_0"),
                text("Image from tool tool, call call_0: a.png"),
                image(),
                image(),
            ])]
        );
    }

    #[test]
    fn text_and_images_keep_their_order_and_follow_the_result_s_own_images() {
        let mut conversation = results_message(
            vec![
                Content::Text("first".into()),
                Content::Text("label b".into()),
                Content::Image(pixels()),
                Content::Text("second".into()),
            ],
            1,
        );
        let Content::ToolResult(result) = &mut conversation.messages[2].content[0] else {
            unreachable!()
        };
        result.images = vec![ToolResultImage::new("own", pixels())];
        assert_eq!(
            results_of(&after_calls(&conversation)),
            [json!([
                text("out call_0\n\nfirst\n\nsecond"),
                text("own"),
                image(),
                text("label b"),
                image(),
            ])]
        );
    }

    #[test]
    fn only_the_last_of_several_results_takes_the_rest() {
        let conversation = results_message(vec![Content::Text("note".into())], 3);
        assert_eq!(
            results_of(&after_calls(&conversation)),
            [
                json!([text("out call_0")]),
                json!([text("out call_1")]),
                json!([text("out call_2\n\nnote")]),
            ]
        );
    }

    /// Gemini 2.x, and every other provider, keep the text after the results
    /// (the pinned bodies show it), in the results message and in the
    /// messages after it.
    #[test]
    fn other_models_are_not_folded() {
        let mut gemini_2_5 = results_message(vec![Content::Text("note".into())], 1);
        gemini_2_5.add_user_message("# Workflow context update\n\nfile changed");
        gemini_2_5.model = gemini_2();
        let steps = after_calls(&gemini_2_5);
        assert_eq!(steps[0]["result"], json!([text("out call_0")]));
        assert_eq!(
            steps[1],
            json!({"type": "user_input", "content": [text("note")]})
        );
        assert_eq!(
            steps[2],
            json!({"type": "user_input", "content": [text("# Workflow context update\n\nfile changed")]})
        );
    }

    /// A user message of text after the results message, before the model's
    /// next turn (a runtime context update), is folded into the last result
    /// after the message's own text, each after a blank line.
    #[test]
    fn text_messages_after_the_results_are_folded_in_order() {
        let mut one = results_message(vec![Content::Text("answer".into())], 2);
        one.add_user_message("# Workflow context update\n\nsrc/lib.rs changed");
        assert_eq!(
            results_of(&after_calls(&one)),
            [
                json!([text("out call_0")]),
                json!([text(
                    "out call_1\n\nanswer\n\n# Workflow context update\n\nsrc/lib.rs changed"
                )]),
            ]
        );

        let mut two = results_message(Vec::new(), 1);
        two.add_user_message("first update");
        two.add_user_message("second update");
        assert_eq!(
            results_of(&after_calls(&two)),
            [json!([text("out call_0\n\nfirst update\n\nsecond update")])]
        );
    }

    /// An image message after the results joins the last result's images,
    /// with the text before it in its own message as its label; a text at
    /// the end of the results message doesn't label it.
    #[test]
    fn an_image_message_after_the_results_is_folded_with_its_label() {
        let mut conversation = results_message(vec![Content::Text("note".into())], 1);
        conversation.add_user_message_with_contents(vec![Content::Image(pixels())]);
        conversation.add_user_message_with_contents(vec![
            Content::Text("screenshot".into()),
            Content::Image(pixels()),
        ]);
        assert_eq!(
            results_of(&after_calls(&conversation)),
            [json!([
                text("out call_0\n\nnote"),
                image(),
                text("screenshot"),
                image(),
            ])]
        );
    }

    /// A user message after the model's turn is a new input, not folded;
    /// neither is a later results message.
    #[test]
    fn a_user_message_after_an_assistant_message_is_not_folded() {
        let mut conversation = results_message(Vec::new(), 1);
        conversation.add_assistant_message("Done.");
        conversation.add_user_message("Next task.");
        let steps = after_calls(&conversation);
        assert_eq!(steps[0]["result"], json!([text("out call_0")]));
        assert_eq!(steps[1]["type"], "model_output");
        assert_eq!(
            steps[2],
            json!({"type": "user_input", "content": [text("Next task.")]})
        );
    }

    /// A conversation saved with the runtime's sibling layout (before result
    /// images existed) loads and is folded on Gemini 3.x.
    #[test]
    fn saved_conversations_with_sibling_images_load_and_fold() {
        let saved = serde_json::to_string(&conversation(gemini_3(), Layout::Siblings)).unwrap();
        let restored: Conversation = serde_json::from_str(&saved).unwrap();
        let steps = after_calls(&restored);
        assert_eq!(
            results_of(&steps[..2]),
            [
                json!([text("a.png, 12 × 8")]),
                json!([
                    text("not found\n\nAnswer: yes"),
                    text("Image from tool view_image, call call_a: a.png"),
                    image(),
                ]),
            ]
        );
        assert_eq!(steps[2]["type"], "model_output");
    }
}
