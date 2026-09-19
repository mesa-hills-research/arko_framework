use crate::{
    message::{Content, Message},
    model::Model,
    providers::GenerationResponse,
    reasoning::ReasoningArtifact,
    tool::ToolResult,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

fn new_prompt_cache_key() -> String {
    format!("arko:conversation:{}", Uuid::new_v4())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub model: Model,
    pub system_message: Option<String>,
    pub enable_cache: bool,
    /// Stable request-routing key for providers with prefix prompt caches.
    ///
    /// The key is generated once per conversation and serialized so restored
    /// conversations continue to reach the same cache lineage.
    #[serde(default = "new_prompt_cache_key")]
    pub prompt_cache_key: String,
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            system_message: None,
            enable_cache: true,
            prompt_cache_key: new_prompt_cache_key(),
            messages: Vec::new(),
        }
    }

    pub fn with_system_message(mut self, message: &str) -> Self {
        self.system_message = Some(message.to_string());
        self
    }

    pub fn with_cache_disabled(mut self) -> Self {
        self.enable_cache = false;
        self
    }

    /// Overrides the generated provider prompt-cache routing key.
    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = key.into();
        self
    }

    pub fn add_user_message(&mut self, text: &str) -> &mut Self {
        self.messages.push(Message::new_user(text));
        self
    }

    pub fn add_user_message_with_image(
        &mut self,
        text: &str,
        image: crate::message::Image,
    ) -> &mut Self {
        self.messages
            .push(Message::new_user_with_image(text, image));
        self
    }

    pub fn add_user_message_with_contents(&mut self, contents: Vec<Content>) -> &mut Self {
        self.messages
            .push(Message::new_user_with_contents(contents));
        self
    }

    pub fn add_assistant_message(&mut self, text: &str) -> &mut Self {
        self.messages.push(Message::new_assistant(text));
        self
    }

    pub fn add_assistant_message_with_reasoning(
        &mut self,
        text: &str,
        reasoning_artifact: ReasoningArtifact,
    ) -> &mut Self {
        self.messages.push(Message::new_assistant_with_reasoning(
            text,
            reasoning_artifact,
        ));
        self
    }

    pub fn add_assistant_message_from_response(
        &mut self,
        response: GenerationResponse,
    ) -> &mut Self {
        self.messages.push(response.message);
        self
    }

    /// Adds tool results as a user message in the conversation.
    ///
    /// This should be called after receiving an assistant message with
    /// tool calls (typically via [`add_assistant_message_from_response`])
    /// and after executing the requested tools locally.
    ///
    /// Each provider serializes tool results in its native format:
    /// - **OpenAI**: `function_call_output` input entries
    /// - **Anthropic**: `tool_result` content blocks in a user message
    /// - **Gemini**: `function_result` steps in the Interactions API
    ///   `input`, correlated via `call_id`
    ///
    /// [`add_assistant_message_from_response`]: Self::add_assistant_message_from_response
    pub fn add_tool_results(&mut self, results: Vec<ToolResult>) -> &mut Self {
        let contents = results.into_iter().map(Content::ToolResult).collect();
        self.messages.push(Message::new_user_with_contents(contents));
        self
    }

    /// Adds tool results followed by additional user text in a single
    /// turn. Useful when the user wants to provide feedback alongside
    /// the tool outputs.
    ///
    /// Tool results are placed before the text content, which matches
    /// the ordering requirements of all supported providers.
    pub fn add_tool_results_with_text(
        &mut self,
        results: Vec<ToolResult>,
        text: &str,
    ) -> &mut Self {
        let mut contents: Vec<Content> = results.into_iter().map(Content::ToolResult).collect();
        contents.push(Content::Text(text.to_string()));
        self.messages.push(Message::new_user_with_contents(contents));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_cache_keys_are_unique_per_conversation() {
        let first = Conversation::new(Model::openai_gpt_5_6_sol());
        let second = Conversation::new(Model::openai_gpt_5_6_sol());

        assert!(first.prompt_cache_key.starts_with("arko:conversation:"));
        assert_ne!(first.prompt_cache_key, second.prompt_cache_key);
    }

    #[test]
    fn prompt_cache_key_survives_serialization() {
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let serialized = serde_json::to_string(&conversation).unwrap();
        let restored: Conversation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(conversation.prompt_cache_key, restored.prompt_cache_key);
    }
}
