//! Refused turns: reporting the refused input, the opt-in fallback model's
//! answer to it, and OpenAI reasoning that can't be replayed to another model.

use super::*;

/// Whether OpenAI encrypted reasoning produced by `from` is replayed to `to`.
/// Codex replays history unchanged when it switches between models of one
/// compaction family and compacts it otherwise (codex-rs
/// `maybe_run_previous_model_inline_compact`, its bundled catalog,
/// models-manager/models.json, gives every GPT-5.6 and GPT-6 model the same
/// `comp_hash`). arko can't compact, so for other pairs the reasoning items
/// are dropped and the reply's messages and calls replayed without them.
pub(super) fn openai_reasoning_replays(from: &ai::Model, to: &ai::Model) -> bool {
    let family = |model: &ai::Model| {
        model.provider == ai::Provider::OpenAI
            && (model.is_openai_gpt_5_6()
                || model.is_openai_gpt_6_astra()
                || model.is_openai_gpt_6_sol_or_luna())
    };
    from.id == to.id || (family(from) && family(to))
}

/// Drops an assistant message's OpenAI reasoning (for a model that can't read
/// it, see [`openai_reasoning_replays`]). The response's other items then
/// lose their ids too: the Responses API rejects an item with an id (such as
/// `fc_…`) sent without the reasoning item that preceded it ("provided
/// without its required 'reasoning' item"), while ids are optional on input.
/// Codex never hits this, since it compacts rather than strip on such a
/// switch.
pub(super) fn strip_openai_reasoning(message: &mut ai::Message) {
    let stripped = match &mut message.reasoning_artifact {
        Some(ai::ReasoningArtifact::OpenAIOutputItems { items }) => {
            let before = items.len();
            items.retain(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
            let stripped = items.len() != before;
            if stripped {
                for item in items.iter_mut() {
                    if let Some(item) = item.as_object_mut() {
                        item.remove("id");
                    }
                }
            }
            if items.is_empty() {
                message.reasoning_artifact = None;
            }
            stripped
        }
        Some(ai::ReasoningArtifact::OpenAIEncryptedReasoning { .. }) => {
            message.reasoning_artifact = None;
            true
        }
        _ => false,
    };
    if stripped {
        // Calls and text rebuilt without items carry these ids otherwise.
        message.id = None;
        for call in &mut message.tool_calls {
            call.provider_id = None;
        }
    }
}

impl AgentLoop {
    /// Reports a refusal and stops recovery: the same input is expected to be
    /// refused again, so it is never retried unchanged. With the opt-in
    /// [`AgentConfig::refusal_fallback`], a refusal from the configured model
    /// instead sends the same input to the fallback model once. A failure
    /// of that request is recovered from by its own error, the fallback
    /// staying pending, and only the fallback's own refusal makes the
    /// refusal stand (see [`recover`](Self::recover)).
    ///
    /// A model's own refusal ([`ai::RefusalSource::keeps_turn`]) stays in
    /// history so the model can see what it declined, with the refusal
    /// explanation as its text when the turn has none (not when the fallback
    /// answers instead). Its tool calls are not run, so they are not kept
    /// either (a call without a result would invalidate the history, OpenAI
    /// output items stored with the turn replay only the calls it still
    /// holds), and a turn with no text at all is not kept. In output-based
    /// tool mode the turn's text holds tool-call markup that was never run:
    /// `plain_text` is the text without it, and only the refusal explanation
    /// (or else that plain text) is kept. Classifier and content-filter
    /// refusals are removed from history.
    ///
    /// The refused input is recorded in [`Task::refused_input`] and reported
    /// with `AgentEvent::RefusedInput`, for `UserCommand::ResolveRefusal`.
    pub(super) async fn handle_refusal(
        &mut self,
        refusal: ai::Refusal,
        mut message: ai::Message,
        plain_text: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        self.flush_throttled_reasoning().await?;
        self.flush_throttled_text().await?;
        let from_fallback = std::mem::take(&mut self.current_request_is_fallback);
        self.task.fallback_for = None;
        let fallback = self
            .task
            .agent_config
            .refusal_fallback
            .clone()
            .filter(|_| !from_fallback);
        let input = self.task.current_input();
        if refusal.source.keeps_turn() && fallback.is_none() {
            message.tool_calls.clear();
            let has_text = |message: &ai::Message| {
                message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .any(|text| !text.trim().is_empty())
            };
            if let Some(plain_text) = plain_text {
                let text = refusal
                    .explanation
                    .clone()
                    .filter(|explanation| !explanation.trim().is_empty())
                    .unwrap_or(plain_text);
                message
                    .content
                    .retain(|content| content.as_text().is_none());
                if !text.trim().is_empty() {
                    message.content.insert(0, ai::Content::Text(text));
                }
                // OpenAI message items would replay the original text,
                // markup included; the message's own text replaces them.
                if let Some(ai::ReasoningArtifact::OpenAIOutputItems { items }) =
                    &mut message.reasoning_artifact
                {
                    items
                        .retain(|item| item.get("type").and_then(Value::as_str) != Some("message"));
                }
            } else if !has_text(&message)
                && let Some(explanation) = refusal.explanation.as_deref()
            {
                message
                    .content
                    .retain(|content| content.as_text().is_none());
                message
                    .content
                    .insert(0, ai::Content::Text(explanation.to_string()));
            }
            if has_text(&message) {
                self.task.conversation.messages.push(message);
            }
        }
        // Recorded before any fallback, so a failed fallback leaves the
        // refusal resolvable.
        self.task.refused_input = input.clone();
        self.emit_event(AgentEvent::RefusalReceived(refusal.clone()))
            .await?;
        if let Some(input) = input {
            self.emit_event(AgentEvent::RefusedInput(input)).await?;
        }
        if let Some(fallback) = fallback {
            self.emit_event(AgentEvent::RefusalFallback {
                model_id: fallback.model.id,
            })
            .await?;
            self.task.fallback_for = Some(refusal);
            // The refusal was a response: the fallback's request starts a
            // recovery of its own.
            self.reset_llm_recovery();
            return Ok(());
        }
        self.handle_llm_error(ai::Error::Refusal(Box::new(refusal)), None)
            .await
    }

    /// A fallback model's response joins the configured model's
    /// conversation: its ChatGPT turn-state token is not kept (it routed the
    /// fallback's request, not the configured model's turn), OpenAI
    /// reasoning items are dropped unless the configured model can read them
    /// (see [`openai_reasoning_replays`]), and the cache monitor skips the
    /// call.
    pub(super) fn prepare_fallback_response(&mut self, response: &mut ai::GenerationResponse) {
        response.openai_turn_state = None;
        self.task.fallback_calls.push(self.task.usage_history.len());
        let Some(fallback) = &self.task.agent_config.refusal_fallback else {
            return;
        };
        if !openai_reasoning_replays(&fallback.model, &self.task.conversation.model) {
            strip_openai_reasoning(&mut response.message);
        }
    }
}
