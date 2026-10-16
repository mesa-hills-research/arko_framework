//! How large a request is, by section: what a host shows as the split of
//! a conversation's context, estimated from bytes without a tokenizer.

use serde::{Deserialize, Serialize};

use crate::config::GenerationConfig;
use crate::conversation::Conversation;
use crate::message::{Content, Image, Message, Role};
use crate::tool::ToolConfig;

/// The size of a request by section: bytes of text, which a bytes-per-token
/// ratio turns into tokens, and images, counted with their estimated
/// tokens. The text sections add up to the request's text as this crate
/// sends it, before each provider's own framing; reasoning a provider
/// replays and what a provider adds of its own are not counted.
///
/// Built with [`of`](Self::of) for a whole request, or kept up to date
/// message by message with [`add_message`](Self::add_message), which costs
/// only the new message's size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestSizes {
    /// Bytes of the system prompt.
    pub system_prompt: usize,
    /// Bytes of the tool definitions, as JSON.
    pub tool_definitions: usize,
    /// Bytes of the text in user messages: the user's own messages and
    /// everything else sent as one (notes, context updates, an agent
    /// runtime's messages).
    pub user_text: usize,
    /// Bytes of the model's text in assistant messages.
    pub assistant_text: usize,
    /// Bytes of the model's tool calls: each tool's name and arguments.
    pub tool_calls: usize,
    /// Bytes of tool results: each tool's name, output and image labels.
    pub tool_results: usize,
    /// How many images the request holds, in user messages and tool
    /// results.
    pub images: usize,
    /// Their estimated input tokens ([`Image::estimated_tokens`]).
    pub image_tokens: usize,
}

impl RequestSizes {
    /// A request with `system_prompt`, offering `tools` (none when `None`
    /// or empty), and no messages yet.
    pub fn new(system_prompt: Option<&str>, tools: Option<&ToolConfig>) -> Self {
        let tool_definitions = tools
            .filter(|tools| !tools.tools.is_empty())
            .map_or(0, |tools| {
                serde_json::to_string(&tools.tools).map_or(0, |json| json.len())
            });
        Self {
            system_prompt: system_prompt.map_or(0, str::len),
            tool_definitions,
            ..Self::default()
        }
    }

    /// The sizes of `conversation` sent with `tools`.
    pub fn of(conversation: &Conversation, tools: Option<&ToolConfig>) -> Self {
        Self::of_messages(conversation, &conversation.messages, tools)
    }

    /// The sizes of the request `config` sends with `conversation` (its
    /// [`offered_tools`](GenerationConfig::offered_tools)).
    pub fn of_request(conversation: &Conversation, config: &GenerationConfig) -> Self {
        Self::of(conversation, config.offered_tools())
    }

    /// The sizes of a request with `conversation`'s system prompt and only
    /// `messages`.
    pub fn of_messages(
        conversation: &Conversation,
        messages: &[Message],
        tools: Option<&ToolConfig>,
    ) -> Self {
        let mut sizes = Self::new(conversation.system_message.as_deref(), tools);
        for message in messages {
            sizes.add_message(message);
        }
        sizes
    }

    /// Adds a message's size to its sections.
    pub fn add_message(&mut self, message: &Message) {
        for content in &message.content {
            match content {
                Content::Text(text) if message.role == Role::Assistant => {
                    self.assistant_text += text.len()
                }
                Content::Text(text) => self.user_text += text.len(),
                Content::Image(image) => self.add_image(image),
                Content::ToolResult(result) => {
                    self.tool_results += result.name.len() + result.output.len();
                    for image in &result.images {
                        self.tool_results += image.label.len();
                        self.add_image(&image.image);
                    }
                }
            }
        }
        for call in &message.tool_calls {
            self.tool_calls += call.name.len() + call.arguments.to_string().len();
        }
    }

    fn add_image(&mut self, image: &Image) {
        self.images += 1;
        self.image_tokens += image.estimated_tokens();
    }

    /// The bytes of text in every section, the system prompt and the tool
    /// definitions included.
    pub fn text_bytes(&self) -> usize {
        self.system_prompt
            + self.tool_definitions
            + self.user_text
            + self.assistant_text
            + self.tool_calls
            + self.tool_results
    }

    /// The tokens `bytes` of text take at `bytes_per_token`, rounded up.
    pub fn tokens(bytes: usize, bytes_per_token: f64) -> usize {
        if bytes_per_token <= 0.0 {
            return 0;
        }
        (bytes as f64 / bytes_per_token).ceil() as usize
    }

    /// The request's estimated input tokens: its text at `bytes_per_token`,
    /// and its images' estimated tokens.
    pub fn estimated_tokens(&self, bytes_per_token: f64) -> usize {
        Self::tokens(self.text_bytes(), bytes_per_token) + self.image_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::tool::{ToolCall, ToolDefinition, ToolResult, ToolResultImage};

    /// 1×1 PNG: 20 bytes of header is all the size reads.
    fn png(width: u32, height: u32) -> Image {
        let mut bytes = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        bytes.extend(width.to_be_bytes());
        bytes.extend(height.to_be_bytes());
        bytes.extend([8, 6, 0, 0, 0]);
        Image::from_bytes(&bytes, "image/png")
    }

    /// Each section counts what it names, the sections add up to the
    /// request's text, and adding messages one at a time gives the same
    /// sizes as measuring the whole conversation.
    #[test]
    fn sections_add_up_and_grow_message_by_message() {
        let mut conversation =
            Conversation::new(Model::anthropic_claude_opus_5_5()).with_system_message("You help.");
        conversation.add_user_message_with_image("Look:", png(750, 100));
        let mut reply = Message::new_assistant("Reading.");
        reply.tool_calls.push(ToolCall::from_argument_text(
            Some("c1".into()),
            "read_file".into(),
            r#"{"path":"a"}"#,
        ));
        conversation.messages.push(reply);
        let result =
            ToolResult::from_tool_call(&conversation.messages[1].tool_calls[0], "contents")
                .with_images(vec![ToolResultImage::new("a.png", png(1500, 1000))]);
        conversation.add_tool_results(vec![result]);
        conversation.add_assistant_message("Done.");
        let tools = ToolConfig::new(vec![ToolDefinition::new(
            "read_file",
            "Reads",
            serde_json::json!({"type": "object"}),
        )]);

        let sizes = RequestSizes::of(&conversation, Some(&tools));
        assert_eq!(sizes.system_prompt, "You help.".len());
        assert_eq!(
            sizes.tool_definitions,
            serde_json::to_string(&tools.tools).unwrap().len()
        );
        assert_eq!(sizes.user_text, "Look:".len());
        assert_eq!(sizes.assistant_text, "Reading.Done.".len());
        assert_eq!(
            sizes.tool_calls,
            "read_file".len() + r#"{"path":"a"}"#.len()
        );
        assert_eq!(
            sizes.tool_results,
            "read_file".len() + "contents".len() + "a.png".len()
        );
        assert_eq!(sizes.images, 2);
        // 750 × 100 / 750, and 1500 × 1000 scaled to fit 1.15 megapixels.
        assert_eq!(sizes.image_tokens, 100 + 1534);
        assert_eq!(
            sizes.text_bytes(),
            sizes.system_prompt
                + sizes.tool_definitions
                + sizes.user_text
                + sizes.assistant_text
                + sizes.tool_calls
                + sizes.tool_results
        );
        assert_eq!(
            sizes.estimated_tokens(2.0),
            sizes.text_bytes().div_ceil(2) + sizes.image_tokens
        );

        let mut grown = RequestSizes::new(Some("You help."), Some(&tools));
        for message in &conversation.messages {
            grown.add_message(message);
        }
        assert_eq!(grown, sizes);
        // No tools, or an empty list, count nothing.
        assert_eq!(RequestSizes::of(&conversation, None).tool_definitions, 0);
        assert_eq!(
            RequestSizes::of(&conversation, Some(&ToolConfig::new(Vec::new()))).tool_definitions,
            0
        );
    }
}
