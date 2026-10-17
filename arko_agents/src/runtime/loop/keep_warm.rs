//! A request that keeps the task's prompt cache warm between turns: the
//! task's next request with a short ping after its conversation, or its last
//! request again, its reply discarded.

use super::*;
use crate::runtime::CacheRefresh;

/// The cap a refresh falls back to once a provider refuses its smaller one
/// (Anthropic's smallest thinking budget, 1,024 tokens, is the usual reason
/// a small cap is refused).
const FALLBACK_OUTPUT_CAP: usize = 1_024;

impl AgentLoop {
    /// Sends the task's next request (system prompt, tools, conversation) with
    /// a short ping after it and the output capped where the provider
    /// allows, so the provider reads the cached prefix and keeps it. The
    /// reply is discarded and the task is unchanged. A task waiting with
    /// calls of the model's that have no results yet (paused for an
    /// approval) can't take a message after them: its last request goes
    /// again unchanged instead, ending where the cache entry the next request
    /// reads ends. A provider that refuses the cap gets the request once
    /// more with [`FALLBACK_OUTPUT_CAP`], which the result reports.
    ///
    /// Unsupported (`arko_ai::Error::UnsupportedFeature`) for Gemini (its
    /// implicit cache has no lifetime a request is known to extend),
    /// OpenRouter models other than Anthropic's, and Anthropic's with
    /// reasoning on (OpenRouter derives the thinking budget from the output
    /// cap, so a capped request would change the thinking parameters and
    /// miss the cached messages), a manual Anthropic thinking budget (the
    /// ping could spend it), a task that has made no request yet, and a task
    /// waiting with calls on the ChatGPT backend (its last request would go
    /// again uncapped).
    pub(crate) async fn refresh_prompt_cache(mut self) -> Result<CacheRefresh, ArkoAgentsError> {
        if self.task.conversation.system_message.is_none()
            || self.task.conversation.messages.is_empty()
        {
            return Err(unsupported(
                "the task has made no request with its current prompt yet, so nothing is cached",
            ));
        }
        let (mut conversation, mut config) = self.next_request();
        let model = &conversation.model;
        config.max_tokens = match model.provider {
            ai::Provider::Anthropic => {
                if matches!(
                    config.reasoning,
                    Some(ai::ReasoningConfig::Anthropic { budget_tokens, .. }) if budget_tokens > 0
                ) {
                    return Err(unsupported(
                        "a manual thinking budget must stay below the output limit, so the ping could spend the whole budget",
                    ));
                }
                Some(1)
            }
            ai::Provider::OpenRouter if model.id.starts_with("anthropic/") => {
                if config
                    .reasoning
                    .as_ref()
                    .is_some_and(openrouter_reasoning_on)
                {
                    return Err(unsupported(
                        "OpenRouter derives an Anthropic model's thinking budget from the output limit, so a capped request would change the thinking parameters and miss the cached conversation",
                    ));
                }
                Some(1)
            }
            // The ChatGPT backend takes no limit (the ping asks for a word).
            ai::Provider::OpenAI if self.llm_client.openai_uses_chatgpt_oauth().await => None,
            // The Responses API's smallest limit.
            ai::Provider::OpenAI => Some(16),
            _ => {
                return Err(unsupported(&format!(
                    "the prompt cache of {:?} model '{}' can't be kept warm by a request",
                    model.provider, model.id
                )));
            }
        };
        let resent = match unanswered_calls(&conversation) {
            // Without an output limit, the last request would have the
            // model write its whole turn again.
            Some(_) if config.max_tokens.is_none() => {
                return Err(unsupported(
                    "the run waits with calls that have no results yet, so its last request would go again, and the ChatGPT backend takes no output limit to keep it short",
                ));
            }
            Some(turn) => {
                conversation.messages.truncate(turn);
                true
            }
            None => {
                conversation.add_user_message(crate::runtime::CACHE_PING);
                false
            }
        };
        let mut refresh = CacheRefresh::new(ai::UsageInfo::default());
        refresh.resent_last_request = resent;
        refresh.output_cap = config.max_tokens;
        let response = match self
            .llm_client
            .generate_text(&conversation, config.clone())
            .await
        {
            Ok(response) => response,
            Err(error) if config.max_tokens.is_some() && refuses_output_cap(&error) => {
                log::warn!(
                    "Task {}: the prompt cache refresh's output limit of {} was refused ({error}); sent again with {FALLBACK_OUTPUT_CAP}.",
                    self.task.id,
                    config.max_tokens.unwrap_or_default()
                );
                refresh.refused_cap = Some(error.to_string());
                refresh.output_cap = Some(FALLBACK_OUTPUT_CAP);
                config.max_tokens = Some(FALLBACK_OUTPUT_CAP);
                self.llm_client.generate_text(&conversation, config).await?
            }
            Err(error) => return Err(error.into()),
        };
        refresh.usage = response.usage;
        Ok(refresh)
    }
}

/// The index of the model's turn whose tool calls have no results yet, if
/// the conversation ends waiting on them.
fn unanswered_calls(conversation: &ai::Conversation) -> Option<usize> {
    let turn = conversation
        .messages
        .iter()
        .rposition(|message| message.role == ai::Role::Assistant)?;
    let answered = conversation.messages[turn + 1..].iter().any(|message| {
        message
            .content
            .iter()
            .any(|content| matches!(content, ai::Content::ToolResult(_)))
    });
    (!conversation.messages[turn].tool_calls.is_empty() && !answered).then_some(turn)
}

/// Whether OpenRouter reasoning is on.
fn openrouter_reasoning_on(reasoning: &ai::ReasoningConfig) -> bool {
    match reasoning {
        ai::ReasoningConfig::OpenRouter(reasoning) => {
            reasoning.enabled != Some(false)
                && (reasoning.enabled == Some(true)
                    || reasoning.effort.is_some()
                    || reasoning.max_tokens.is_some())
        }
        _ => false,
    }
}

/// Whether `error` is a provider refusing the request's output limit.
fn refuses_output_cap(error: &ai::Error) -> bool {
    let text = error.to_string().to_ascii_lowercase();
    text.contains("max_tokens") || text.contains("max_output_tokens")
}

fn unsupported(reason: &str) -> ArkoAgentsError {
    ArkoAgentsError::LLM(ai::Error::UnsupportedFeature(format!(
        "The prompt cache can't be refreshed: {reason}."
    )))
}
