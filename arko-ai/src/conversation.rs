use crate::{
    message::{ConfigurationUpdate, Content, Message, Role},
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
    /// conversations continue to reach the same cache lineage. OpenAI
    /// requests send it as `prompt_cache_key`; ChatGPT OAuth requests also
    /// send it as the `session-id`, `thread-id` and `x-client-request-id`
    /// headers, since the ChatGPT backend derives cache affinity from
    /// `session-id`. OpenRouter always receives it as `session_id` (its
    /// sticky-routing key, at most 256 characters) and, with caching enabled,
    /// as `prompt_cache_key`. Give each agent its own conversation (and key).
    #[serde(default = "new_prompt_cache_key")]
    pub prompt_cache_key: String,
    pub messages: Vec<Message>,
    /// The ChatGPT backend's sticky-routing token for the current user turn.
    ///
    /// The backend returns `x-codex-turn-state` on its responses. Like Codex,
    /// arko keeps the first token of a turn and echoes it, unchanged, on
    /// every later ChatGPT OAuth request of the same turn: tool-result
    /// follow-ups and retries. It is never sent in a different turn.
    ///
    /// A turn starts at a user message that holds no tool results. Adding
    /// one with [`add_user_message`](Self::add_user_message),
    /// [`add_user_message_with_image`](Self::add_user_message_with_image) or
    /// [`add_user_message_with_contents`](Self::add_user_message_with_contents)
    /// clears the token; [`add_tool_results`](Self::add_tool_results) and
    /// [`add_tool_results_with_text`](Self::add_tool_results_with_text) keep
    /// it. [`OpenAITurnState::turn_start`] also ties the token to its turn,
    /// so a message pushed onto `messages` directly can't carry it into the
    /// next turn. Serialized so a task resumed mid-turn keeps its routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_turn_state: Option<OpenAITurnState>,
}

/// A ChatGPT backend `x-codex-turn-state` token and the turn it belongs to;
/// see [`Conversation::openai_turn_state`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAITurnState {
    /// Opaque token, echoed unchanged.
    pub token: String,
    /// Index in [`Conversation::messages`] of the user message that started
    /// the turn (the last user message without tool results when the token
    /// arrived); `None` when there was none.
    pub turn_start: Option<usize>,
}

impl Conversation {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            system_message: None,
            enable_cache: true,
            prompt_cache_key: new_prompt_cache_key(),
            messages: Vec::new(),
            openai_turn_state: None,
        }
    }

    /// Index of the message that started the current turn: the last user
    /// message without tool results.
    fn turn_start(&self) -> Option<usize> {
        self.messages
            .iter()
            .rposition(|message| message.role == Role::User && !has_tool_results(&message.content))
    }

    /// The turn-state token the next request of the current turn should
    /// echo, if the backend issued one in this turn.
    pub fn current_openai_turn_state(&self) -> Option<&str> {
        self.openai_turn_state
            .as_ref()
            .filter(|state| state.turn_start == self.turn_start())
            .map(|state| state.token.as_str())
    }

    /// Records a turn-state token from a response. The first token of a
    /// turn is kept for the rest of it; a token from a new turn replaces the
    /// previous turn's.
    pub fn record_openai_turn_state(&mut self, token: &str) {
        if self.current_openai_turn_state().is_none() {
            self.openai_turn_state = Some(OpenAITurnState {
                token: token.to_string(),
                turn_start: self.turn_start(),
            });
        }
    }

    /// Adds a user message; one without tool results starts a new turn.
    fn push_turn_message(&mut self, message: Message) {
        if !has_tool_results(&message.content) {
            self.openai_turn_state = None;
        }
        self.messages.push(message);
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

    /// Adds a user message, which starts a new turn.
    pub fn add_user_message(&mut self, text: &str) -> &mut Self {
        self.push_turn_message(Message::new_user(text));
        self
    }

    /// Adds a user message with an image, which starts a new turn.
    pub fn add_user_message_with_image(
        &mut self,
        text: &str,
        image: crate::message::Image,
    ) -> &mut Self {
        self.push_turn_message(Message::new_user_with_image(text, image));
        self
    }

    /// Adds a user message. It starts a new turn unless it holds tool
    /// results.
    pub fn add_user_message_with_contents(&mut self, contents: Vec<Content>) -> &mut Self {
        self.push_turn_message(Message::new_user_with_contents(contents));
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
    /// changes. Support is per model: OpenRouter names Claude Fable 5.1 and
    /// OpenAI GPT-6 Astra as the first, and Claude Opus 5 as accepting them
    /// too, and rejects models without support with a 400
    /// (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#mid-conversation-effort).
    /// Claude rejects an update to [`ReasoningEffort::None`], so the
    /// OpenRouter provider rejects one for `anthropic/` models before
    /// sending. Other providers reject conversations containing an update
    /// with [`Error::UnsupportedFeature`](crate::error::Error::UnsupportedFeature).
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
    /// The response's turn-state token is recorded either way (see
    /// [`openai_turn_state`](Self::openai_turn_state)), so a retry in the
    /// same turn keeps its routing.
    ///
    /// [`RefusalSource::keeps_turn`]: crate::stop::RefusalSource::keeps_turn
    pub fn add_assistant_message_from_response(
        &mut self,
        response: GenerationResponse,
    ) -> &mut Self {
        if let Some(token) = &response.openai_turn_state {
            self.record_openai_turn_state(token);
        }
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

fn has_tool_results(content: &[Content]) -> bool {
    content.iter().any(|content| content.as_tool_result().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        stop::{Refusal, RefusalSource, StopReason},
        tool::ToolCall,
        usage::{ModelCallUsage, UsageInfo},
    };

    fn response(token: Option<&str>) -> GenerationResponse {
        let mut message = Message::new_assistant("");
        message.tool_calls.push(ToolCall::from_argument_text(Some("call_1".into()), "tool".into(), "{}"));
        GenerationResponse {
            message,
            usage: UsageInfo::new(0, 0),
            model_call_usage: ModelCallUsage::new("resp".into(), "openai".into(), "m".into(), 0, 0, 0),
            stop_reason: StopReason::ToolUse,
            refusal: None,
            openai_turn_state: token.map(String::from),
        }
    }

    fn result() -> ToolResult {
        ToolResult::from_tool_call(&ToolCall::from_argument_text(Some("call_1".into()), "tool".into(), "{}"), "ok")
    }

    #[test]
    fn turn_state_is_kept_for_its_turn_and_dropped_at_the_next_user_message() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("start");
        assert_eq!(conversation.current_openai_turn_state(), None);

        conversation.add_assistant_message_from_response(response(Some("t1")));
        conversation.add_tool_results(vec![result()]);
        assert_eq!(conversation.current_openai_turn_state(), Some("t1"));
        // Later tokens of the same turn don't replace the first (Codex keeps
        // it in a OnceLock for the whole turn).
        conversation.add_assistant_message_from_response(response(Some("t2")));
        conversation.add_tool_results_with_text(vec![result()], "and also");
        assert_eq!(conversation.current_openai_turn_state(), Some("t1"));
        // A response without the header keeps the turn's token.
        conversation.add_assistant_message_from_response(response(None));
        assert_eq!(conversation.current_openai_turn_state(), Some("t1"));

        conversation.add_user_message("next task");
        assert_eq!(conversation.openai_turn_state, None);
        assert_eq!(conversation.current_openai_turn_state(), None);
        conversation.add_assistant_message_from_response(response(Some("t3")));
        assert_eq!(conversation.current_openai_turn_state(), Some("t3"));

        // A user message pushed directly still ends the turn.
        conversation.messages.push(Message::new_user("pushed"));
        assert_eq!(conversation.current_openai_turn_state(), None);
        conversation.add_assistant_message_from_response(response(Some("t4")));
        assert_eq!(conversation.current_openai_turn_state(), Some("t4"));

        // Tool results added as contents keep the turn.
        conversation.add_user_message_with_contents(vec![Content::ToolResult(result())]);
        assert_eq!(conversation.current_openai_turn_state(), Some("t4"));
        conversation.add_user_message_with_contents(vec![Content::Text("new turn".into())]);
        assert_eq!(conversation.current_openai_turn_state(), None);
    }

    #[test]
    fn a_refused_turn_still_records_its_token_for_the_retry() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("start");
        let mut refused = response(Some("t1"));
        refused.refusal = Some(Refusal::new(RefusalSource::ContentFilter));
        conversation.add_assistant_message_from_response(refused);
        assert_eq!(conversation.messages.len(), 1);
        assert_eq!(conversation.current_openai_turn_state(), Some("t1"));
    }

    #[test]
    fn turn_state_survives_serialization_and_old_conversations_load() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("start");
        conversation.add_assistant_message_from_response(response(Some("t1")));
        let restored: Conversation =
            serde_json::from_str(&serde_json::to_string(&conversation).unwrap()).unwrap();
        assert_eq!(restored.current_openai_turn_state(), Some("t1"));

        let mut saved = serde_json::to_value(Conversation::new(Model::openai_gpt_6_astra())).unwrap();
        assert!(saved.get("openai_turn_state").is_none());
        saved.as_object_mut().unwrap().remove("prompt_cache_key");
        let restored: Conversation = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.openai_turn_state, None);
    }

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
