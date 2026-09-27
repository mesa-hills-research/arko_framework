use crate::{
    message::{ConfigurationUpdate, Content, Message},
    model::Model,
    providers::GenerationResponse,
    reasoning::{ReasoningArtifact, ReasoningEffort},
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
    /// conversations continue to reach the same cache lineage. OpenRouter
    /// always receives it as `session_id` (its sticky-routing key, at most 256
    /// characters) and, with caching enabled, as `prompt_cache_key`.
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

    /// Changes the reasoning effort from this point of the conversation
    /// onward. Call it just before adding the user turn it should apply to.
    ///
    /// The request-level effort stays the baseline for earlier turns, and the
    /// update stays in history at its position, so the prompt prefix before it
    /// (and the provider's prompt cache for it) is unchanged. Unlike editing
    /// the request-level effort, earlier turns keep their cache.
    ///
    /// Supported for OpenRouter models that accept mid-conversation effort
    /// changes (for example GPT-6 Astra, Sol and Luna, Claude Fable 5.1 and
    /// Claude Opus 5; OpenRouter rejects other models with a 400). Other
    /// providers reject conversations containing an update with
    /// [`Error::UnsupportedFeature`](crate::error::Error::UnsupportedFeature).
    /// Updates can't be combined with pro reasoning mode or `-pro` models.
    ///
    /// Providers reject two updates in a row, so calling this again before any
    /// other message replaces the pending update.
    pub fn add_reasoning_effort_update(&mut self, effort: ReasoningEffort) -> &mut Self {
        self.add_configuration_update(ConfigurationUpdate::reasoning_effort(effort))
    }

    /// Appends a [`ConfigurationUpdate`]; see
    /// [`add_reasoning_effort_update`](Self::add_reasoning_effort_update).
    pub fn add_configuration_update(&mut self, update: ConfigurationUpdate) -> &mut Self {
        match self.messages.last_mut() {
            Some(last) if last.is_configuration_update() => {
                last.configuration_update = Some(update);
            }
            _ => self.messages.push(Message::new_configuration_update(update)),
        }
        self
    }

    /// Appends the response's assistant message. A refused turn that must not
    /// remain in history (see [`RefusalSource::keeps_turn`]) is not appended;
    /// the caller decides what to do with the input that triggered it.
    ///
    /// [`RefusalSource::keeps_turn`]: crate::stop::RefusalSource::keeps_turn
    pub fn add_assistant_message_from_response(
        &mut self,
        response: GenerationResponse,
    ) -> &mut Self {
        if response
            .refusal
            .as_ref()
            .is_some_and(|refusal| !refusal.source.keeps_turn())
        {
            return self;
        }
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
    fn effort_updates_are_kept_in_place_and_adjacent_ones_collapse() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_sol());
        conversation.add_user_message("Plan.");
        conversation.add_assistant_message("Planned.");
        conversation.add_reasoning_effort_update(ReasoningEffort::High);
        conversation.add_reasoning_effort_update(ReasoningEffort::Low);
        conversation.add_user_message("Next.");
        let updates: Vec<_> = conversation
            .messages
            .iter()
            .filter_map(|message| message.configuration_update.as_ref())
            .collect();
        assert_eq!(updates, [&ConfigurationUpdate::reasoning_effort(ReasoningEffort::Low)]);
        assert!(conversation.messages[2].is_configuration_update());
        assert!(conversation.messages[2].content.is_empty());

        // Messages saved before updates existed still load.
        let mut saved = serde_json::to_value(&conversation).unwrap();
        saved["messages"][0].as_object_mut().unwrap().remove("configuration_update");
        let restored: Conversation = serde_json::from_value(saved).unwrap();
        assert!(!restored.messages[0].is_configuration_update());
        assert!(restored.messages[2].is_configuration_update());

        // Providers that can't send updates refuse rather than dropping them.
        let error = crate::providers::ensure_supported_inputs(&conversation).unwrap_err();
        assert!(matches!(error, crate::error::Error::UnsupportedFeature(_)), "{error}");
    }

    #[test]
    fn prompt_cache_key_survives_serialization() {
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let serialized = serde_json::to_string(&conversation).unwrap();
        let restored: Conversation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(conversation.prompt_cache_key, restored.prompt_cache_key);
    }
}
