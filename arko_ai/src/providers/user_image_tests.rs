//! Where a user's images go in each provider's request: after the text of
//! their message, as the provider's image input, in a task's first message
//! and in a later one (a context update with a labelled image).

use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Image},
    model::{Model, Provider},
    providers::{
        anthropic::AnthropicClient, google::GoogleClient, openai::OpenAIClient,
        openrouter::OpenRouterClient,
    },
    reasoning::{GeminiThinkingLevel, ReasoningConfig},
};
use serde_json::{Value, json};

fn pixels() -> Image {
    Image::new_base64("cGl4ZWxz", "image/png")
}

/// A first message with an image, a reply, and a context update whose image
/// follows its label.
fn conversation(model: Model) -> Conversation {
    let mut conversation = Conversation::new(model).with_prompt_cache_key("pinned");
    conversation.add_user_message_with_contents(vec![
        Content::Text("What is this?".into()),
        Content::Image(pixels()),
    ]);
    conversation.add_assistant_message("A chart.");
    conversation.add_user_message_with_contents(vec![
        Content::Text("# Workflow context update\n\nA new note.".into()),
        Content::Text("The image of note 3:".into()),
        Content::Image(pixels()),
    ]);
    conversation
}

fn body(conversation: &Conversation, oauth: bool) -> Result<Value, Error> {
    let model = &conversation.model;
    let text = match model.provider {
        Provider::OpenAI => OpenAIClient::new("test").request_body(
            conversation,
            &GenerationConfig::default(),
            oauth,
        )?,
        Provider::Anthropic => {
            AnthropicClient::new("test").request_body(conversation, &GenerationConfig::default())?
        }
        Provider::Google => {
            let config = if model.uses_gemini_thinking_levels() {
                GenerationConfig::default()
                    .with_reasoning(ReasoningConfig::gemini(GeminiThinkingLevel::Medium, false))
            } else {
                GenerationConfig::default()
            };
            GoogleClient::new("test").request_body(conversation, &config)?
        }
        Provider::OpenRouter => {
            let prepared = OpenRouterClient::new("test")
                .build_request(conversation, &GenerationConfig::default())?;
            serde_json::to_string(&prepared.body).unwrap()
        }
        Provider::ClaudeCode => unreachable!("Claude Code sends no HTTP request"),
    };
    Ok(serde_json::from_str(&text).unwrap())
}

fn model(provider: Provider, id: &str) -> Model {
    Model::get(provider, id).unwrap()
}

/// The user messages of a request body, in order.
fn user_contents(body: &Value, provider: Provider) -> Vec<Value> {
    let (items, is_user): (&str, fn(&Value) -> bool) = match provider {
        Provider::OpenAI => ("input", |item| item["role"] == "user"),
        Provider::Anthropic | Provider::OpenRouter => ("messages", |item| item["role"] == "user"),
        Provider::Google => ("input", |item| item["type"] == "user_input"),
        Provider::ClaudeCode => unreachable!(),
    };
    body[items]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| is_user(item))
        .map(|item| item["content"].clone())
        .collect()
}

#[test]
fn openai_sends_user_images_after_their_text() {
    let image = json!({"type": "input_image", "image_url": "data:image/png;base64,cGl4ZWxz"});
    for oauth in [false, true] {
        let body = body(&conversation(model(Provider::OpenAI, "gpt-6-astra")), oauth).unwrap();
        assert_eq!(
            user_contents(&body, Provider::OpenAI),
            [
                json!([{"type": "input_text", "text": "What is this?"}, image]),
                json!([
                    {"type": "input_text", "text": "# Workflow context update\n\nA new note."},
                    {"type": "input_text", "text": "The image of note 3:"},
                    image,
                ]),
            ],
            "oauth: {oauth}"
        );
    }
}

#[test]
fn anthropic_sends_user_images_after_their_text() {
    let body = body(
        &conversation(model(Provider::Anthropic, "claude-opus-5-5")),
        false,
    )
    .unwrap();
    let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cGl4ZWxz"}});
    assert_eq!(
        user_contents(&body, Provider::Anthropic),
        [
            json!([{"type": "text", "text": "What is this?"}, image]),
            json!([
                {"type": "text", "text": "# Workflow context update\n\nA new note."},
                {"type": "text", "text": "The image of note 3:"},
                image,
            ]),
        ]
    );
}

#[test]
fn gemini_sends_user_images_after_their_text() {
    let image = json!({"type": "image", "mime_type": "image/png", "data": "cGl4ZWxz"});
    for id in ["gemini-3.5-flash", "gemini-2.5-pro"] {
        let body = body(&conversation(model(Provider::Google, id)), false).unwrap();
        assert_eq!(
            user_contents(&body, Provider::Google),
            [
                json!([{"type": "text", "text": "What is this?"}, image]),
                json!([
                    {"type": "text", "text": "# Workflow context update\n\nA new note."},
                    {"type": "text", "text": "The image of note 3:"},
                    image,
                ]),
            ],
            "{id}"
        );
    }
}

#[test]
fn openrouter_sends_user_images_after_their_text() {
    let image =
        json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,cGl4ZWxz"}});
    for id in ["anthropic/claude-opus-5.5", "openai/gpt-6-astra"] {
        let body = body(&conversation(Model::openrouter(id)), false).unwrap();
        assert_eq!(
            user_contents(&body, Provider::OpenRouter),
            [
                json!([{"type": "text", "text": "What is this?"}, image]),
                json!([
                    {"type": "text", "text": "# Workflow context update\n\nA new note."},
                    {"type": "text", "text": "The image of note 3:"},
                    image,
                ]),
            ],
            "{id}"
        );
    }
}

/// A model without image input refuses a request with a user's image
/// before sending it, as `Model::image_refusal` says.
#[test]
fn models_without_image_input_refuse_user_images() {
    for mut model in [
        model(Provider::OpenAI, "gpt-6-astra"),
        model(Provider::Anthropic, "claude-opus-5-5"),
        model(Provider::Google, "gemini-3.5-flash"),
        Model::openrouter("openai/gpt-6-astra"),
    ] {
        model.supports_image_input = false;
        assert!(model.image_refusal(&pixels()).is_some(), "{}", model.id);
        let error = body(&conversation(model.clone()), false).unwrap_err();
        assert!(
            matches!(error, Error::UnsupportedFeature(_)),
            "{}: {error}",
            model.id
        );
    }
}

/// What `Model::image_refusal` refuses: anything without image input, a URL
/// where the request takes data only (Anthropic and Google refuse it, and
/// Claude Code would pass it on as text), and GIF on Google.
#[test]
fn image_refusal_names_what_a_model_can_t_take() {
    let url = Image::new_url("https://example.com/a.png");
    let gif = Image::new_base64("R0lGODlh", "image/gif");
    let gpt = model(Provider::OpenAI, "gpt-6-astra");
    let claude = model(Provider::Anthropic, "claude-opus-5-5");
    let gemini = model(Provider::Google, "gemini-3.5-flash");
    let claude_code = Model::claude_code(claude.clone());
    let openrouter = Model::openrouter("anthropic/claude-opus-5.5");
    for model in [&gpt, &claude, &gemini, &claude_code, &openrouter] {
        assert_eq!(model.image_refusal(&pixels()), None, "{}", model.id);
    }
    assert_eq!(gpt.image_refusal(&url), None);
    assert_eq!(openrouter.image_refusal(&url), None);
    for model in [&claude, &gemini, &claude_code] {
        let reason = model.image_refusal(&url).unwrap();
        assert!(reason.contains("as data, not by URL"), "{reason}");
    }
    assert!(
        gemini
            .image_refusal(&gif)
            .unwrap()
            .contains("doesn't take GIF")
    );
    assert_eq!(claude.image_refusal(&gif), None);
    assert_eq!(gpt.image_refusal(&gif), None);
}
