//! A conversation's tool calls and results written as text, for a request
//! that offers no tools: a conversation continued without tools (an agent
//! switched to a chat) keeps what its calls did, and no provider is sent
//! call or result items without tool definitions to go with them.

use std::borrow::Cow;

use crate::{
    conversation::Conversation,
    message::{Content, Message},
    tool::{ToolCall, ToolResult},
};

impl Conversation {
    /// This conversation as a request without tools sends it: unchanged
    /// (borrowed) when it holds no tool call or result; otherwise a copy in
    /// which each call becomes a line `[Tool call NAME: ARGUMENTS]` in its
    /// assistant message, whose reasoning is left out (it led to the calls,
    /// and an OpenAI response's output items would replay them), and each
    /// result becomes `[Result of NAME]` (or `[NAME failed]`) and its output,
    /// followed by its images, each after its label, in its user message.
    pub fn tool_history_as_text(&self) -> Cow<'_, Conversation> {
        if !self.messages.iter().any(has_tool_history) {
            return Cow::Borrowed(self);
        }
        let mut flattened = self.clone();
        for message in &mut flattened.messages {
            if has_tool_history(message) {
                as_text(message);
            }
        }
        Cow::Owned(flattened)
    }
}

fn has_tool_history(message: &Message) -> bool {
    !message.tool_calls.is_empty()
        || message
            .content
            .iter()
            .any(|content| content.as_tool_result().is_some())
}

fn as_text(message: &mut Message) {
    let calls = std::mem::take(&mut message.tool_calls);
    if !calls.is_empty() {
        message.reasoning_artifact = None;
    }
    let mut content: Vec<Content> = Vec::new();
    for part in std::mem::take(&mut message.content) {
        match part {
            Content::Text(text) if text.is_empty() => {}
            Content::ToolResult(result) => content.extend(result_content(result)),
            other => content.push(other),
        }
    }
    content.extend(calls.iter().map(|call| Content::Text(call_text(call))));
    if content.is_empty() {
        content.push(Content::Text(String::new()));
    }
    message.content = content;
}

/// `[Tool call read_file: {"path":"a.rs"}]`.
fn call_text(call: &ToolCall) -> String {
    let arguments = match &call.invalid_arguments {
        Some(text) => text.clone(),
        None => call.arguments.to_string(),
    };
    format!("[Tool call {}: {arguments}]", call.name)
}

/// `[Result of read_file]` (or `[read_file failed]`) and the output, then
/// each image after its label.
fn result_content(result: ToolResult) -> Vec<Content> {
    let heading = if result.is_error {
        format!("[{} failed]", result.name)
    } else {
        format!("[Result of {}]", result.name)
    };
    let mut content = vec![Content::Text(if result.output.is_empty() {
        heading
    } else {
        format!("{heading}\n{}", result.output)
    })];
    for image in result.images {
        if !image.label.is_empty() {
            content.push(Content::Text(image.label));
        }
        content.push(Content::Image(image.image));
    }
    content
}
