//! The run's start: the configuration checked, the task in progress, what a
//! stopped run left settled, and the prompts written.

use super::*;

impl AgentLoop {
    /// Checks the configuration before anything is reported (the
    /// developer environment details the run can show need content), and
    /// estimates the saved conversation's latest request.
    pub(super) fn check_config(&mut self) -> Result<(), ArkoAgentsError> {
        self.task.agent_config.validate()?;
        self.answered_request = self.saved_request_mark();
        if self.is_native_mode()
            && self
                .task
                .agent_config
                .llm_config
                .generation_config
                .tool_config
                .as_ref()
                .is_some_and(|tools| !tools.tools.is_empty())
        {
            log::warn!(
                "Task {}: The model's tool_config lists tool definitions, which native tool calling doesn't send. The agent's tools come from its tool registry. Its tool_choice and parallel_tool_calls apply.",
                self.task.id
            );
        }
        // Every developer environment detail the run can show needs content:
        // details placed in every user message on every turn, and the others
        // while the system prompt and task prompt are still to be written.
        let writing_prompts = self.task.conversation.system_message.is_none();
        if let Some(error) = self
            .task
            .developer_env_details
            .iter()
            .flatten()
            .filter(|detail| {
                writing_prompts
                    || detail.placement
                        == crate::agent_config::EnvironmentDetailPlacement::EachUserMessage
            })
            .find_map(|detail| detail.missing_content_error())
        {
            return Err(ArkoAgentsError::Configuration(error));
        }
        log::info!(
            "Agent loop starting for task ID: {} (mode: {:?})",
            self.task.id,
            self.task.agent_config.tool_config.tool_calling_mode
        );
        Ok(())
    }

    /// The run's start: the task in progress (reported), what a stopped
    /// run left settled, and the prompts written.
    pub(super) async fn start(&mut self) -> Result<(), ArkoAgentsError> {
        let interrupted = self.start_state().await?;
        // Images found gone become lines saying so, and are reported.
        self.task.settle_unavailable_images();
        if !self.task.unavailable_images.is_empty() {
            let reasons = std::mem::take(&mut self.task.unavailable_images);
            self.emit_event(AgentEvent::ImagesUnavailable { reasons })
                .await?;
        }
        // A task saved while its tool calls ran (by a host that stopped)
        // settles them before anything else.
        let recovered = self.recover_running_calls().await?;
        if interrupted {
            // Older versions left a native turn without results when an
            // approval wait ended this way.
            self.settle_interrupted_wait(UNDECIDED_CALL, UNDECIDED_CALL)
                .await?;
            self.end_tool_batch().await?;
        } else {
            self.settle_open_turn(recovered).await?;
        }
        self.write_prompts()
    }

    /// Puts a task that is pending or was interrupted in progress, and one
    /// saved while approved calls ran out of its approval wait, and reports
    /// a task in progress. Returns whether the task was interrupted.
    async fn start_state(&mut self) -> Result<bool, ArkoAgentsError> {
        let interrupted = matches!(self.task.current_state, TaskState::Interrupted { .. });
        if matches!(self.task.current_state, TaskState::Pending)
            || matches!(self.task.current_state, TaskState::Interrupted { .. })
        {
            self.task.set_state(TaskState::InProgress);
            self.task.monitor_pause = false;
        }
        // Calls run only once their approval is decided: a task saved while
        // they ran no longer waits for it.
        if !self.task.running_tool_calls.is_empty()
            && matches!(
                self.task.current_state,
                TaskState::AwaitingToolApproval { .. }
                    | TaskState::AwaitingBatchToolApproval { .. }
            )
        {
            self.task.set_state(TaskState::InProgress);
        }

        if matches!(self.task.current_state, TaskState::InProgress) {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        Ok(interrupted)
    }

    /// Writes the system prompt and the task prompt, unless the
    /// conversation has them already.
    pub(super) fn write_prompts(&mut self) -> Result<(), ArkoAgentsError> {
        self.task.number_context();
        if self.task.conversation.system_message.is_none() {
            let tool_defs_for_prompt = self.prompt_tool_definitions();
            let prompt_builder = PromptBuilder::new(
                &self.task.agent_config,
                tool_defs_for_prompt.iter().collect(),
                &self.parsing_config,
                self.skill_prompt_context.as_ref(),
            )
            .tools_offered(self.offers_tools());
            let system_prompt = prompt_builder.build_system_prompt(&self.task)?;
            self.task.conversation.system_message = Some(system_prompt);

            if self.task.conversation.messages.is_empty()
                && !self.task.initial_task_prompt.is_empty()
            {
                let initial_user_prompt = prompt_builder
                    .format_initial_task_prompt(&self.task, &self.task.initial_task_prompt)?;
                // Moved: the first message holds them from now on.
                let content = crate::task::with_images(
                    initial_user_prompt,
                    std::mem::take(&mut self.task.initial_task_images),
                );
                self.task
                    .conversation
                    .add_user_message_with_contents(content);
                self.task.increment_user_message_count();
                self.task.record_initial_prompt();
                self.reset_llm_recovery();
            }
        }
        Ok(())
    }

    /// Writes the system prompt alone, from the task's configuration and
    /// the tools it offers (a handover's fresh conversation that changed
    /// the configuration adds its first message itself).
    pub(super) fn write_system_prompt(&mut self) -> Result<(), ArkoAgentsError> {
        let tool_defs_for_prompt = self.prompt_tool_definitions();
        let system_prompt = PromptBuilder::new(
            &self.task.agent_config,
            tool_defs_for_prompt.iter().collect(),
            &self.parsing_config,
            self.skill_prompt_context.as_ref(),
        )
        .tools_offered(self.offers_tools())
        .build_system_prompt(&self.task)?;
        self.task.conversation.system_message = Some(system_prompt);
        Ok(())
    }

    /// The definitions of the tools the system prompt lists: those the
    /// configuration lists that the registry has, and `complete_task`.
    fn prompt_tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut tool_defs_for_prompt: Vec<ToolDefinition> = Vec::new();
        for name in &self.task.agent_config.tool_config.available_tool_names {
            if let Some(tool) = self.tool_registry.get(name) {
                tool_defs_for_prompt.push(tool.definition());
            } else {
                log::warn!(
                    "Agent {} configured with tool '{}' but it was not found in the registry.",
                    self.task.agent_config.name,
                    name
                );
            }
        }
        if !tool_defs_for_prompt
            .iter()
            .any(|def| def.name == COMPLETE_TASK_TOOL_NAME)
            && let Some(ct_tool) = self.tool_registry.get(COMPLETE_TASK_TOOL_NAME)
        {
            tool_defs_for_prompt.push(ct_tool.definition());
        }
        tool_defs_for_prompt
    }
}
