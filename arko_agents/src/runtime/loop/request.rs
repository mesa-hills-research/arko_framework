//! The pass's model request: its conversation and settings (to the refusal
//! fallback's model while one is pending), and the attempt that makes it.

use super::*;

const OAUTH_REFRESH_MARGIN_SECS: u64 = 5 * 60;

#[cfg(test)]
pub(super) type ScriptedStreamResult =
    Result<futures::stream::BoxStream<'static, Result<ai::StreamEvent, ai::Error>>, ai::Error>;

impl AgentLoop {
    /// The tools of a native request: the agent's offered tools, with the
    /// configured tool choice (none, the provider's `auto`, by default: tool
    /// use is optional) and parallel-call setting. One call per turn
    /// ([`AgentConfig::asks_one_tool_call_per_turn`]) disables parallel calls
    /// where the provider can.
    pub(super) fn build_native_tool_config(
        &self,
        model: &ai::Model,
        configured: Option<&ai::ToolConfig>,
    ) -> ai::ToolConfig {
        let agent_config = &self.task.agent_config;
        let offered = |tool: Arc<dyn crate::tool::Tool>| {
            agent_config
                .offered_tool_definition(tool.definition())
                .to_arko_ai_tool_definition()
        };
        let mut ai_tools = Vec::new();
        for name in &agent_config.tool_config.available_tool_names {
            if let Some(tool) = self.tool_registry.get(name) {
                ai_tools.push(offered(tool));
            }
        }
        if !agent_config
            .tool_config
            .available_tool_names
            .iter()
            .any(|n| n == COMPLETE_TASK_TOOL_NAME)
            && let Some(ct) = self.tool_registry.get(COMPLETE_TASK_TOOL_NAME)
        {
            ai_tools.push(offered(ct));
        }

        let mut config = ai::ToolConfig::new(ai_tools);
        if let Some(configured) = configured {
            config.tool_choice = configured.tool_choice.clone();
            config.parallel_tool_calls = configured.parallel_tool_calls;
        }
        if agent_config.asks_one_tool_call_per_turn() {
            // Gemini can't turn parallel calls off; its calls run one at a
            // time instead.
            config.parallel_tool_calls = model
                .supports_disabling_parallel_tool_calls()
                .then_some(false);
        }
        config
    }

    pub(super) fn build_generation_config(&self) -> ai::GenerationConfig {
        self.generation_config_for(
            &self.task.conversation.model,
            &self.task.agent_config.llm_config.generation_config,
        )
    }

    fn generation_config_for(
        &self,
        model: &ai::Model,
        generation_config: &ai::GenerationConfig,
    ) -> ai::GenerationConfig {
        let mut config = generation_config.clone();
        if self.is_native_mode() {
            config = config.with_tool_config(
                self.build_native_tool_config(model, generation_config.tool_config.as_ref()),
            );
        }
        config
    }

    /// The conversation and settings of the next request: the task's, or
    /// after a refusal with [`AgentConfig::refusal_fallback`] set, the same
    /// conversation sent to the fallback model.
    fn next_request(&mut self) -> (ai::Conversation, ai::GenerationConfig) {
        let fallback = self
            .task
            .agent_config
            .refusal_fallback
            .clone()
            .filter(|_| self.task.fallback_for.is_some());
        self.current_request_is_fallback = fallback.is_some();
        let mut conversation = self.task.conversation.clone();
        #[cfg(test)]
        self.requested_models.lock().unwrap().push(
            fallback
                .as_ref()
                .map_or(&conversation.model.id, |fallback| &fallback.model.id)
                .clone(),
        );
        match fallback {
            Some(fallback) => {
                let config =
                    self.generation_config_for(&fallback.model, &fallback.generation_config);
                // The configured model's reasoning replays to the fallback
                // under the same rule as the other way around.
                if !openai_reasoning_replays(&conversation.model, &fallback.model) {
                    for message in &mut conversation.messages {
                        if message.role == ai::Role::Assistant {
                            strip_openai_reasoning(message);
                        }
                    }
                }
                conversation.model = fallback.model;
                // The ChatGPT backend's turn-state token routes the configured
                // model's turn; the fallback request starts its own.
                conversation.openai_turn_state = None;
                (conversation, config)
            }
            None => (conversation, self.build_generation_config()),
        }
    }

    /// Refresh and model stream initialization share one attempt. A request
    /// once sent runs until it completes or fails, or the host cancels or
    /// suspends the task: the elapsed recovery budget never cuts it off.
    pub(super) async fn run_llm_attempt<T>(
        &mut self,
        operation: impl std::future::Future<Output = Result<T, ai::Error>>,
    ) -> Result<Option<T>, ArkoAgentsError> {
        self.recovery.llm_interaction_attempts =
            self.recovery.llm_interaction_attempts.saturating_add(1);
        tokio::pin!(operation);
        loop {
            tokio::select! {
                biased;
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(None); }
                }
                result = &mut operation => {
                    return match result {
                        Ok(value) => Ok(Some(value)),
                        Err(ai::Error::Refusal(refusal)) => {
                            self.handle_refusal(*refusal, ai::Message::new_assistant(""), None).await?;
                            Ok(None)
                        }
                        Err(error) => { self.handle_llm_error(error, None).await?; Ok(None) }
                    };
                }
            }
        }
    }

    /// The pass's model request (to the refusal fallback's model while one
    /// is pending), made under the error policy's recovery, and its stream.
    pub(super) async fn model_turn(&mut self) -> Result<ModelTurn, ArkoAgentsError> {
        let llm_client = Arc::clone(&self.llm_client);
        let (conversation, gen_config) = self.next_request();
        #[cfg(test)]
        self.trace.request(&conversation, &gen_config);
        // What this request's usage is compared with, to estimate the
        // next one (not for the refusal fallback's model).
        self.sent_request = (!self.current_request_is_fallback).then(|| RequestMark {
            messages: conversation.messages.len(),
            bytes: result_budget::request_bytes(&conversation, &gen_config),
            conversation: self.task.conversation_usage_start,
        });
        #[cfg(test)]
        let scripted_response = self.scripted_responses.get_mut().unwrap().pop_front();
        let request = async move {
            #[cfg(test)]
            if let Some(response) = scripted_response {
                return response;
            }
            if conversation.model.provider == ai::Provider::OpenAI {
                // Decided under the credential lock, so tasks sharing a
                // client (or an auth.json) don't rotate twice.
                llm_client
                    .refresh_openai_oauth_token_if_expiring(Duration::from_secs(
                        OAUTH_REFRESH_MARGIN_SECS,
                    ))
                    .await?;
            }
            llm_client.generate_stream(&conversation, gen_config).await
        };
        let Some(response_stream) = self.run_llm_attempt(request).await? else {
            return Ok(ModelTurn::Ended(Flow::NextPass));
        };
        // The provider accepted the request: the elapsed budget stops
        // counting while the response runs, however long that is, and
        // resumes if it fails (the waits between attempts still count,
        // so failures that keep coming stay bounded).
        self.pause_ordinary_recovery_clock();
        self.stream_response(response_stream).await
    }
}
