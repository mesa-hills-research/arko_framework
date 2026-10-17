use crate::{
    error::Error,
    message::{ConfigurationUpdate, Content, Message, Role},
    model::Model,
    providers::GenerationResponse,
    reasoning::{ReasoningArtifact, ReasoningEffort},
    tool::ToolResult,
};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// A random UUID, with nothing that names the library.
fn new_prompt_cache_key() -> String {
    Uuid::new_v4().to_string()
}

/// A saved model copy, replaced by the registry's entry for its id when there
/// is one, so a restored conversation prices calls and checks limits with
/// current data (the rule `AgentLLMConfig` follows too). Ids the registry
/// doesn't know (OpenRouter slugs, models added after this build) keep the
/// saved copy.
fn registry_model<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Model, D::Error> {
    let saved = Model::deserialize(deserializer)?;
    Ok(Model::get(saved.provider.clone(), &saved.id).unwrap_or(saved))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// The model the conversation is sent to. When a conversation is
    /// deserialized, a model the registry knows is replaced by its current
    /// registry entry (see [`Model::get`]), including a saved copy whose
    /// fields were customized. Other ids keep the saved copy. To keep custom
    /// pricing or limits, use an id the registry doesn't know.
    #[serde(deserialize_with = "registry_model")]
    pub model: Model,
    pub system_message: Option<String>,
    /// Whether requests use provider prompt caching, unless a request's
    /// [`GenerationConfig::prompt_caching`] says otherwise. On by default
    /// ([`with_cache_disabled`](Self::with_cache_disabled) turns it off).
    /// Some providers cache implicitly and can't turn caching off. See
    /// [`GenerationConfig::prompt_caching`] for each provider.
    ///
    /// [`GenerationConfig::prompt_caching`]: crate::config::GenerationConfig::prompt_caching
    pub enable_cache: bool,
    /// Stable request-routing key for providers with prefix prompt caches.
    ///
    /// A new conversation gets a random UUID, generated once and serialized
    /// so restored conversations continue to reach the same cache lineage
    /// (conversations saved with an older key form keep theirs). OpenAI
    /// requests send it as `prompt_cache_key`. ChatGPT OAuth requests also
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
    /// clears the token. [`add_tool_results`](Self::add_tool_results) and
    /// [`add_tool_results_with_text`](Self::add_tool_results_with_text) keep
    /// it.
    ///
    /// When `messages` is edited directly, the token is echoed only while
    /// the turn's starting message is still the last user message without
    /// tool results, at the same index and with the same serialized form
    /// ([`OpenAITurnState::turn_start`] and
    /// [`OpenAITurnState::turn_start_fingerprint`]). Pushing a new user
    /// message, or rewinding and pushing a different one at the same index,
    /// drops it. Rewinding to the unchanged starting message (a retry of the
    /// turn) keeps it. Serialized so a task resumed mid-turn keeps its
    /// routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_turn_state: Option<OpenAITurnState>,
}

/// A ChatGPT backend `x-codex-turn-state` token and the turn it belongs to.
/// See [`Conversation::openai_turn_state`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAITurnState {
    /// Opaque token, echoed unchanged.
    pub token: String,
    /// Index in [`Conversation::messages`] of the user message that started
    /// the turn (the last user message without tool results when the token
    /// arrived). `None` when there was none.
    pub turn_start: Option<usize>,
    /// Hex SHA-256 of that message's JSON serialization, so a different
    /// message at the same index is a different turn. `None` when there was
    /// no such message. A state saved with an index but no fingerprint is
    /// never echoed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_start_fingerprint: Option<String>,
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

    /// Index and fingerprint of the message that started the current turn:
    /// the last user message without tool results.
    fn turn_start(&self) -> (Option<usize>, Option<String>) {
        let index = self
            .messages
            .iter()
            .rposition(|message| message.role == Role::User && !has_tool_results(&message.content));
        let fingerprint = index.map(|index| {
            let serialized = serde_json::to_vec(&self.messages[index]).unwrap_or_default();
            Sha256::digest(&serialized)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        });
        (index, fingerprint)
    }

    /// The turn-state token the next request of the current turn should
    /// echo, if the backend issued one in this turn.
    pub fn current_openai_turn_state(&self) -> Option<&str> {
        let state = self.openai_turn_state.as_ref()?;
        let (index, fingerprint) = self.turn_start();
        (state.turn_start == index && state.turn_start_fingerprint == fingerprint)
            .then_some(state.token.as_str())
    }

    /// Records a turn-state token from a response. The first token of a
    /// turn is kept for the rest of it. A token from a new turn replaces the
    /// previous turn's.
    pub fn record_openai_turn_state(&mut self, token: &str) {
        if self.current_openai_turn_state().is_none() {
            let (turn_start, turn_start_fingerprint) = self.turn_start();
            self.openai_turn_state = Some(OpenAITurnState {
                token: token.to_string(),
                turn_start,
                turn_start_fingerprint,
            });
        }
    }

    /// Adds a user message. One without tool results starts a new turn.
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

    /// Turns provider prompt caching off for this conversation's requests
    /// where the provider allows it. See [`enable_cache`](Self::enable_cache).
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
    /// (<https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#mid-conversation-effort>).
    /// Claude rejects an update to [`ReasoningEffort::None`], so the
    /// OpenRouter provider rejects one for `anthropic/` models before
    /// sending. Other providers reject conversations containing an update
    /// with [`Error::UnsupportedFeature`].
    /// Updates can't be combined with pro reasoning mode or `-pro` models.
    ///
    /// Providers reject two updates in a row, so calling this again before any
    /// other message replaces the pending update.
    pub fn add_reasoning_effort_update(&mut self, effort: ReasoningEffort) -> &mut Self {
        self.add_configuration_update(ConfigurationUpdate::reasoning_effort(effort))
    }

    /// Appends a [`ConfigurationUpdate`]. See
    /// [`add_reasoning_effort_update`](Self::add_reasoning_effort_update).
    pub fn add_configuration_update(&mut self, update: ConfigurationUpdate) -> &mut Self {
        match self.messages.last_mut() {
            Some(last) if last.is_configuration_update() => {
                last.configuration_update = Some(update);
            }
            _ => self
                .messages
                .push(Message::new_configuration_update(update)),
        }
        self
    }

    /// Appends the response's assistant message. A refused turn that must not
    /// remain in history (see [`RefusalSource::keeps_turn`]) is not appended.
    /// The caller decides what to do with the input that triggered it.
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
        self.messages
            .push(Message::new_user_with_contents(contents));
        self
    }

    /// Adds tool results followed by additional user text in a single
    /// turn. Useful when the user wants to provide feedback alongside
    /// the tool outputs.
    ///
    /// Tool results are placed before the text content, which matches
    /// the ordering requirements of all supported providers. Gemini 3 and
    /// later get the text inside the last result instead, appended to its
    /// output after a blank line, as Google advises.
    pub fn add_tool_results_with_text(
        &mut self,
        results: Vec<ToolResult>,
        text: &str,
    ) -> &mut Self {
        let mut contents: Vec<Content> = results.into_iter().map(Content::ToolResult).collect();
        contents.push(Content::Text(text.to_string()));
        self.messages
            .push(Message::new_user_with_contents(contents));
        self
    }
}

/// Editing the input of a refused turn (or any earlier input) before sending
/// it again. A provider that refused a request refuses it again unchanged, so
/// the input has to change: rephrase or remove the user's message, or redact
/// a tool result the model or a filter objected to. Edits that end a turn's
/// history early also remove the messages after the edited one (the refused
/// turn and anything after it). Tool calls and their results always stay
/// paired. A ChatGPT turn-state token tied to an edited turn start is no
/// longer echoed (see [`openai_turn_state`](Self::openai_turn_state)).
impl Conversation {
    /// Index of the last user turn: the last user message holding no tool
    /// results.
    pub fn last_user_turn(&self) -> Option<usize> {
        self.messages
            .iter()
            .rposition(|message| message.role == Role::User && !has_tool_results(&message.content))
    }

    /// Replaces the text of the last user turn with `text` (images it holds
    /// are kept, after the text) and removes every message after it, so the
    /// turn can be sent again.
    pub fn rephrase_last_user_turn(&mut self, text: &str) -> Result<(), Error> {
        let index = self.last_user_turn().ok_or_else(|| {
            Error::InvalidRequest("The conversation has no user message to rephrase.".into())
        })?;
        self.messages.truncate(index + 1);
        let message = &mut self.messages[index];
        message
            .content
            .retain(|content| content.as_text().is_none());
        message.content.insert(0, Content::Text(text.to_string()));
        Ok(())
    }

    /// Removes the last user turn and every message after it, returning the
    /// removed user message. The conversation then ends where it did before
    /// that message was added.
    pub fn remove_last_user_turn(&mut self) -> Result<Message, Error> {
        let index = self.last_user_turn().ok_or_else(|| {
            Error::InvalidRequest("The conversation has no user message to remove.".into())
        })?;
        self.messages.truncate(index + 1);
        Ok(self.messages.remove(index))
    }

    /// Replaces the output of the tool result for `call_id` with
    /// `replacement`, keeping its call id, name and error flag, so the call
    /// and its result stay paired. Returns the index of the message holding
    /// it. Messages after it are kept. Truncate `messages` to drop them.
    ///
    /// Only the text is redacted: images the result holds
    /// ([`ToolResult::images`]) are kept. Remove those separately, with the
    /// message's other images, through [`Message::remove_images`].
    pub fn redact_tool_result(&mut self, call_id: &str, replacement: &str) -> Result<usize, Error> {
        for (index, message) in self.messages.iter_mut().enumerate().rev() {
            for content in &mut message.content {
                if let Content::ToolResult(result) = content
                    && result.call_id.as_deref() == Some(call_id)
                {
                    result.output = replacement.to_string();
                    return Ok(index);
                }
            }
        }
        Err(Error::InvalidRequest(format!(
            "The conversation has no tool result for call {call_id}."
        )))
    }
}

fn has_tool_results(content: &[Content]) -> bool {
    content
        .iter()
        .any(|content| content.as_tool_result().is_some())
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
        message.tool_calls.push(ToolCall::from_argument_text(
            Some("call_1".into()),
            "tool".into(),
            "{}",
        ));
        GenerationResponse {
            message,
            usage: UsageInfo::new(0, 0),
            model_call_usage: ModelCallUsage::new(
                "resp".into(),
                "openai".into(),
                "m".into(),
                0,
                0,
                0,
            ),
            stop_reason: StopReason::ToolUse,
            refusal: None,
            openai_turn_state: token.map(String::from),
        }
    }

    fn result() -> ToolResult {
        ToolResult::from_tool_call(
            &ToolCall::from_argument_text(Some("call_1".into()), "tool".into(), "{}"),
            "ok",
        )
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
    fn a_different_turn_start_at_the_same_index_drops_the_token() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("start");
        conversation.add_assistant_message("done");
        conversation.add_user_message("first task");
        let turn_start = conversation.messages.len() - 1;
        conversation.add_assistant_message_from_response(response(Some("t1")));
        conversation.add_tool_results(vec![result()]);
        assert_eq!(conversation.current_openai_turn_state(), Some("t1"));

        // The application rewinds and pushes a different user message
        // directly, at the same index: a new turn.
        conversation.messages.truncate(turn_start);
        conversation.messages.push(Message::new_user("second task"));
        assert_eq!(conversation.current_openai_turn_state(), None);
        conversation.add_assistant_message_from_response(response(Some("t2")));
        assert_eq!(conversation.current_openai_turn_state(), Some("t2"));

        // Rewinding to the same message is a retry of the same turn.
        conversation.messages.truncate(turn_start + 1);
        assert_eq!(conversation.current_openai_turn_state(), Some("t2"));
        // So is restoring it unchanged; the fingerprint survives serialization.
        let restored: Conversation =
            serde_json::from_str(&serde_json::to_string(&conversation).unwrap()).unwrap();
        assert_eq!(restored.current_openai_turn_state(), Some("t2"));
        // A state without a fingerprint is never echoed.
        let mut saved = serde_json::to_value(&conversation).unwrap();
        saved["openai_turn_state"]
            .as_object_mut()
            .unwrap()
            .remove("turn_start_fingerprint");
        let restored: Conversation = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.current_openai_turn_state(), None);
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

        let mut saved =
            serde_json::to_value(Conversation::new(Model::openai_gpt_6_astra())).unwrap();
        assert!(saved.get("openai_turn_state").is_none());
        saved.as_object_mut().unwrap().remove("prompt_cache_key");
        let restored: Conversation = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.openai_turn_state, None);
    }

    #[test]
    fn prompt_cache_keys_are_unique_per_conversation() {
        let first = Conversation::new(Model::openai_gpt_5_6_sol());
        let second = Conversation::new(Model::openai_gpt_5_6_sol());

        // A bare UUID, with nothing that names the library.
        let parsed = Uuid::parse_str(&first.prompt_cache_key).unwrap();
        assert_eq!(parsed.to_string(), first.prompt_cache_key);
        assert_ne!(first.prompt_cache_key, second.prompt_cache_key);
    }

    #[test]
    fn saved_conversations_keep_their_key() {
        let mut saved =
            serde_json::to_value(Conversation::new(Model::openai_gpt_6_astra())).unwrap();
        saved["prompt_cache_key"] = "arko:conversation:0b6f2c1e-5d0a-4c7e-9b1f-3a2d4e5f6a7b".into();
        let restored: Conversation = serde_json::from_value(saved).unwrap();
        assert_eq!(
            restored.prompt_cache_key,
            "arko:conversation:0b6f2c1e-5d0a-4c7e-9b1f-3a2d4e5f6a7b"
        );
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
        assert_eq!(
            updates,
            [&ConfigurationUpdate::reasoning_effort(ReasoningEffort::Low)]
        );
        assert!(conversation.messages[2].is_configuration_update());
        assert!(conversation.messages[2].content.is_empty());

        // Messages saved before updates existed still load.
        let mut saved = serde_json::to_value(&conversation).unwrap();
        saved["messages"][0]
            .as_object_mut()
            .unwrap()
            .remove("configuration_update");
        let restored: Conversation = serde_json::from_value(saved).unwrap();
        assert!(!restored.messages[0].is_configuration_update());
        assert!(restored.messages[2].is_configuration_update());

        // Providers that can't send updates refuse rather than dropping them.
        let error = crate::providers::ensure_supported_inputs(&conversation).unwrap_err();
        assert!(
            matches!(error, crate::error::Error::UnsupportedFeature(_)),
            "{error}"
        );
    }

    #[test]
    fn a_refused_user_turn_can_be_rephrased_or_removed() {
        let image = crate::message::Image::new_url("https://example.com/a.png");
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("first");
        conversation.add_assistant_message("ok");
        conversation.add_user_message_with_image("describe this", image);
        conversation.add_assistant_message_from_response(response(Some("t1")));
        conversation.add_tool_results(vec![result()]);
        conversation.add_assistant_message("I can't help with that.");
        assert_eq!(conversation.last_user_turn(), Some(2));

        let mut rephrased = conversation.clone();
        rephrased.rephrase_last_user_turn("summarize this").unwrap();
        assert_eq!(
            rephrased.messages.len(),
            3,
            "the replies to the turn are removed"
        );
        let turn = &rephrased.messages[2];
        assert_eq!(turn.content[0].as_text(), Some("summarize this"));
        assert!(turn.content[1].as_image().is_some(), "images are kept");
        assert_eq!(turn.content.len(), 2);
        // The turn's routing token belonged to the old text.
        assert_eq!(rephrased.current_openai_turn_state(), None);

        let mut removed = conversation.clone();
        let message = removed.remove_last_user_turn().unwrap();
        assert_eq!(message.content[0].as_text(), Some("describe this"));
        assert_eq!(removed.messages.len(), 2);
        assert_eq!(removed.messages[1].role, Role::Assistant);

        let mut empty = Conversation::new(Model::openai_gpt_6_astra());
        assert!(empty.rephrase_last_user_turn("x").is_err());
        assert!(empty.remove_last_user_turn().is_err());
    }

    #[test]
    fn a_tool_result_can_be_redacted_without_breaking_the_pairing() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.add_user_message("start");
        conversation.add_assistant_message_from_response(response(None));
        conversation.add_tool_results(vec![result()]);
        conversation.add_assistant_message("I can't help with that.");
        let index = conversation
            .redact_tool_result("call_1", "[removed]")
            .unwrap();
        assert_eq!(index, 2);
        let redacted = conversation.messages[2].content[0]
            .as_tool_result()
            .unwrap();
        assert_eq!(
            (
                redacted.call_id.as_deref(),
                redacted.name.as_str(),
                redacted.output.as_str()
            ),
            (Some("call_1"), "tool", "[removed]")
        );
        assert_eq!(
            conversation.messages.len(),
            4,
            "later messages are the caller's to drop"
        );
        assert!(
            conversation
                .redact_tool_result("call_unknown", "x")
                .is_err()
        );
    }

    #[test]
    fn restored_conversations_take_current_registry_data_for_known_models() {
        let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
        conversation.model.max_input_tokens = 1;
        conversation.model.pricing.input_cost_per_million = 999.0;
        let restored: Conversation =
            serde_json::from_str(&serde_json::to_string(&conversation).unwrap()).unwrap();
        let current = Model::openai_gpt_6_astra();
        assert_eq!(restored.model.max_input_tokens, current.max_input_tokens);
        assert_eq!(
            restored.model.pricing.input_cost_per_million,
            current.pricing.input_cost_per_million
        );

        // Ids the registry doesn't know keep their saved copy.
        let mut future = Model::openai_gpt_6_astra();
        future.id = "gpt-future".into();
        future.max_input_tokens = 7;
        let restored: Conversation =
            serde_json::from_str(&serde_json::to_string(&Conversation::new(future)).unwrap())
                .unwrap();
        assert_eq!(
            (restored.model.id.as_str(), restored.model.max_input_tokens),
            ("gpt-future", 7)
        );
        let mut routed = Model::openrouter("anthropic/claude-sonnet-5");
        routed.max_input_tokens = 12_345;
        let restored: Conversation =
            serde_json::from_str(&serde_json::to_string(&Conversation::new(routed)).unwrap())
                .unwrap();
        assert_eq!(restored.model.max_input_tokens, 12_345);
    }

    #[test]
    fn prompt_cache_key_survives_serialization() {
        let conversation = Conversation::new(Model::openai_gpt_5_6_sol());
        let serialized = serde_json::to_string(&conversation).unwrap();
        let restored: Conversation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(conversation.prompt_cache_key, restored.prompt_cache_key);
    }
}
