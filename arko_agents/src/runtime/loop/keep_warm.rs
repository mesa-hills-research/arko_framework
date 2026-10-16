//! A request that keeps the task's prompt cache warm between turns: the
//! task's next request with a short ping after its conversation, its reply
//! discarded.

use super::*;

impl AgentLoop {
    /// Sends the task's next request (system prompt, tools, conversation) with
    /// a short ping after it and the output capped where the provider
    /// allows, so the provider reads the cached prefix and keeps it; the
    /// reply is discarded and the task is unchanged. Returns the request's
    /// usage. Unsupported (`arko_ai::Error::UnsupportedFeature`) for Gemini
    /// (its implicit cache has no lifetime a request is known to extend),
    /// OpenRouter models other than Anthropic's, a manual Anthropic thinking
    /// budget (the ping could spend it), and a task that has made no request
    /// yet.
    pub(crate) async fn refresh_prompt_cache(mut self) -> Result<ai::UsageInfo, ArkoAgentsError> {
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
            ai::Provider::OpenRouter if model.id.starts_with("anthropic/") => Some(1),
            // The Responses API's smallest limit; the ChatGPT backend takes
            // none, and the ping asks for a word.
            ai::Provider::OpenAI => Some(16),
            _ => {
                return Err(unsupported(&format!(
                    "the prompt cache of {:?} model '{}' can't be kept warm by a request",
                    model.provider, model.id
                )));
            }
        };
        conversation.add_user_message(crate::runtime::CACHE_PING);
        let response = self.llm_client.generate_text(&conversation, config).await?;
        Ok(response.usage)
    }
}

fn unsupported(reason: &str) -> ArkoAgentsError {
    ArkoAgentsError::LLM(ai::Error::UnsupportedFeature(format!(
        "The prompt cache can't be refreshed: {reason}."
    )))
}
