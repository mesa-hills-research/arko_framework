use crate::agent_config::{
    AgentConfig, BuiltInEnvironmentDetail, DynamicContentProvider, EnvironmentDetailPlacement,
    ReminderInjectionFrequency, ToolCallingMode,
};
use crate::error::ArkoAgentsError;
use crate::skill::SkillPromptContext;
use crate::task::Task;
use crate::tool::Tool;
use crate::tool::ToolDefinition;
use crate::utils;

/// Builder for constructing prompts for LLM interactions.
pub struct PromptBuilder<'a> {
    agent_config: &'a AgentConfig,
    available_tools_defs: Vec<&'a ToolDefinition>,
    parsing_config: &'a super::ToolParsingConfig,
    skill_prompt_context: Option<&'a SkillPromptContext>,
}

impl<'a> PromptBuilder<'a> {
    /// Creates a new PromptBuilder.
    pub fn new(
        agent_config: &'a AgentConfig,
        available_tools_defs: Vec<&'a ToolDefinition>,
        parsing_config: &'a super::ToolParsingConfig,
        skill_prompt_context: Option<&'a SkillPromptContext>,
    ) -> Self {
        Self {
            agent_config,
            available_tools_defs,
            parsing_config,
            skill_prompt_context,
        }
    }

    /// Returns true if the agent is configured for native tool calling.
    fn is_native_mode(&self) -> bool {
        self.agent_config.tool_config.tool_calling_mode == ToolCallingMode::Native
    }

    /// Processes text that may contain `{{#output_based_only}}` markers.
    /// In native mode, strips the marked content. In output-based mode,
    /// removes just the markers but keeps the content.
    fn process_conditional_text(&self, text: &str) -> String {
        if self.is_native_mode() {
            super::strip_output_based_only_blocks(text)
        } else {
            super::remove_output_based_only_markers(text)
        }
    }

    /// Builds the system prompt for the agent.
    pub fn build_system_prompt(&self, task: &'a Task) -> Result<String, ArkoAgentsError> {
        let mut prompt = String::new();

        let role_description = self.process_conditional_text(&self.agent_config.role_description);
        let instructions = self.process_conditional_text(&self.agent_config.instructions);

        prompt.push_str(&format!("# Agent: {}\n", self.agent_config.name));
        prompt.push_str(&format!("## Role\n{}\n\n", role_description));
        prompt.push_str(&format!("## Instructions\n{}\n\n", instructions));

        if let Some(ctx) = self.skill_prompt_context
            && !ctx.auto_loaded_skills.is_empty()
        {
            prompt
                .push_str("## Active Skills\nThe following skills are pre-loaded and active:\n\n");
            for skill in &ctx.auto_loaded_skills {
                prompt.push_str(&format!(
                    "--- BEGIN SKILL: {} ---\n{}\n--- END SKILL: {} ---\n\n",
                    skill.name,
                    skill.content.trim(),
                    skill.name
                ));
            }
        }

        let env_details_system =
            self.gather_env_details_for_placement(task, EnvironmentDetailPlacement::SystemPrompt)?;
        if !env_details_system.is_empty() {
            prompt.push_str("## Environment Information\n");
            prompt.push_str(&env_details_system);
            prompt.push('\n');
        }

        if self.is_native_mode() {
            let calls = if self.agent_config.asks_one_tool_call_per_turn() {
                "Call at most one tool per response."
            } else {
                "You can call several tools in one response; calls that don't depend on each other can run at the same time, and each call gets its own result."
            };
            prompt.push_str(&format!(
                "## Tool Usage\n\
                You have tools available through the native tool calling interface. {calls} \
                The task ends only when you call `complete_task`: call it with a summary once the task is finished. \
                A response without a tool call does not end the task.\n\n"
            ));
        } else {
            prompt.push_str("## Tools Available\n");
            prompt.push_str(
                "You have access to the following tools. \
                To use a tool, respond with the tool call in the specified XML-like format. \
                You MUST use one tool call or one batch of tool calls in every assistant response, even if it's 'complete_task'.\n"
            );

            if let Some(ref escape_prefix) = self.parsing_config.escape_prefix
                && !escape_prefix.is_empty()
            {
                prompt.push_str(&format!(
                        "If you need to output text that exactly matches a tool tag or a parameter closing tag within a parameter's value, \
                        prefix the specific problematic tag with `{0}`.\n\n",
                        escape_prefix
                    ));
            }

            if self
                .available_tools_defs
                .iter()
                .any(|def| self.agent_config.allows_batched_calls_for_tool(def))
            {
                prompt.push_str(&format!(
                    "You can call the SAME tool multiple times in a single response using the \
                    <{0}>...</{0}> structure when its description says it supports batched calls.\n",
                    self.parsing_config.batched_tool_call_tag
                ));
            }
            prompt.push_str("\nExample tool call format:\n");
            prompt.push_str("<tool_name>\n  <param_name>value</param_name>\n</tool_name>\n\n");

            for tool_def in &self.available_tools_defs {
                prompt.push_str(&self.format_tool_definition_output_based(tool_def));
                prompt.push('\n');
            }

            if !self
                .available_tools_defs
                .iter()
                .any(|def| def.name == crate::tool::builtins::COMPLETE_TASK_TOOL_NAME)
            {
                log::warn!(
                    "'{}' tool not explicitly in available_tools for prompt generation, adding its definition.",
                    crate::tool::builtins::COMPLETE_TASK_TOOL_NAME
                );
                let ct_def = crate::tool::builtins::CompleteTaskTool::new().definition();
                prompt.push_str(&self.format_tool_definition_output_based(&ct_def));
                prompt.push('\n');
            }
        }

        if let Some(ctx) = self.skill_prompt_context
            && !ctx.on_demand_skills.is_empty()
        {
            prompt.push_str("## Available Skills (On-Demand)\n");
            prompt.push_str(
                    "You have access to additional skills that can be loaded using the `load_skill` tool. \
                     Only load a skill when its domain is relevant to your current task.\n\n"
                );
            for skill in &ctx.on_demand_skills {
                prompt.push_str(&format!("- **{}**: {}\n", skill.name, skill.description));
            }
            prompt.push('\n');
        }

        Ok(prompt)
    }

    /// Formats the initial task prompt, including environment details and any reminder.
    pub fn format_initial_task_prompt(
        &self,
        task: &'a Task,
        initial_prompt: &str,
    ) -> Result<String, ArkoAgentsError> {
        let processed_initial_prompt = self.process_conditional_text(initial_prompt);
        let mut full_prompt = processed_initial_prompt;

        let env_details_initial = self.gather_env_details_for_placement(
            task,
            EnvironmentDetailPlacement::InitialTaskPrompt,
        )?;
        let env_details_each_turn = self
            .gather_env_details_for_placement(task, EnvironmentDetailPlacement::EachUserMessage)?;

        let mut combined_details = String::new();
        combined_details.push_str(&env_details_initial);
        combined_details.push_str(&env_details_each_turn);

        if !combined_details.is_empty() {
            full_prompt.push_str("\n\n# Initial Environment Information\n");
            full_prompt.push_str(&combined_details);
        }

        if let Some(ref reminder) = self.agent_config.user_reminder
            && self.should_inject_reminder(1, &reminder.frequency)
        {
            full_prompt.push_str("\n\n# Reminder\n");
            full_prompt.push_str(&reminder.content);
        }

        Ok(full_prompt)
    }

    /// Augments a user message with environment details and optional reminder.
    pub fn augment_user_message_with_env_details_and_reminder(
        &self,
        task: &'a Task,
        user_message_content: &str,
        user_message_number: usize,
    ) -> Result<String, ArkoAgentsError> {
        let mut full_content = user_message_content.to_string();
        let env_details_each_turn = self
            .gather_env_details_for_placement(task, EnvironmentDetailPlacement::EachUserMessage)?;
        if !env_details_each_turn.is_empty() {
            full_content.push_str("\n\n# Current Environment Information (updated this turn)\n");
            full_content.push_str(&env_details_each_turn);
        }

        if let Some(ref reminder) = self.agent_config.user_reminder
            && self.should_inject_reminder(user_message_number, &reminder.frequency)
        {
            full_content.push_str("\n\n# Reminder\n");
            full_content.push_str(&reminder.content);
        }

        Ok(full_content)
    }

    /// The text that goes beside a turn's native tool results in their user
    /// message: `text` (an answer or a note) followed by the sections
    /// [`augment_user_message_with_env_details_and_reminder`](Self::augment_user_message_with_env_details_and_reminder)
    /// adds, or those sections alone; `None` when there is neither. Providers
    /// take it after the results in the same turn.
    pub fn text_beside_tool_results(
        &self,
        task: &'a Task,
        text: Option<&str>,
        user_message_number: usize,
    ) -> Result<Option<String>, ArkoAgentsError> {
        let augmented = self.augment_user_message_with_env_details_and_reminder(
            task,
            text.unwrap_or_default(),
            user_message_number,
        )?;
        Ok(match text {
            Some(_) => Some(augmented),
            None => {
                let sections = augmented.trim_start_matches("\n\n");
                (!sections.is_empty()).then(|| sections.to_string())
            }
        })
    }

    /// Determines if a reminder should be injected for the given user message number.
    fn should_inject_reminder(
        &self,
        user_message_number: usize,
        frequency: &ReminderInjectionFrequency,
    ) -> bool {
        match frequency {
            ReminderInjectionFrequency::InitialTaskPromptOnly => user_message_number == 1,
            ReminderInjectionFrequency::EveryUserMessage => true,
            ReminderInjectionFrequency::EveryNUserMessages(n) => {
                if *n == 0 {
                    return false;
                }
                if user_message_number == 1 {
                    return true;
                }
                let messages_after_initial = user_message_number - 1;
                messages_after_initial.is_multiple_of(*n)
            }
        }
    }

    /// Formats a tool definition for output-based tool calling prompts.
    fn format_tool_definition_output_based(&self, tool_def: &ToolDefinition) -> String {
        let tool_def = &self.agent_config.offered_tool_definition(tool_def.clone());
        let mut s = String::new();
        s.push_str(&format!("Tool: <{}>\n", tool_def.name));
        s.push_str(&format!("Description: {}\n", tool_def.description));

        if let Some(ref example) = tool_def.output_based_example {
            s.push_str(&format!("Example:\n{}\n", example));
        }

        if self.agent_config.allows_batched_calls_for_tool(tool_def) {
            s.push_str("Supports batched calls.\n");
        }

        if !tool_def.parameters.is_empty() {
            s.push_str("Parameters:\n");
            for param in &tool_def.parameters {
                s.push_str(&format!(
                    "  - <{}>: {} (Type: {:?}, Required: {}",
                    param.name, param.description, param.param_type, param.required
                ));
                if let Some(default_val) = &param.default_value {
                    s.push_str(&format!(", Default: {}", default_val));
                }
                if let Some(flag_val) = &param.flag {
                    s.push_str(&format!(", CLI Flag: {}", flag_val));
                }
                s.push_str(")\n");
            }
        }
        s
    }

    fn gather_env_details_for_placement(
        &self,
        task: &'a Task,
        placement: EnvironmentDetailPlacement,
    ) -> Result<String, ArkoAgentsError> {
        let mut details_str = String::new();
        let config = &self.agent_config.environment_details_config;
        let developer_env_details_override = task.developer_env_details.as_ref();

        let append_built_in_if_match = |detail_str: &mut String,
                                        name: &str,
                                        content_res: Result<String, ArkoAgentsError>,
                                        detail_conf: &BuiltInEnvironmentDetail|
         -> Result<(), ArkoAgentsError> {
            if detail_conf.include && detail_conf.placement == placement {
                match content_res {
                    Ok(content) => {
                        if !content.is_empty() {
                            detail_str.push_str(&format!("{}:\n{}\n", name, content.trim_end()));
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to gather built-in environment detail '{}': {:?}",
                            name,
                            e
                        );
                    }
                }
            }
            Ok(())
        };

        append_built_in_if_match(
            &mut details_str,
            "Current Working Directory",
            Ok(task
                .current_working_directory
                .to_string_lossy()
                .into_owned()),
            &config.current_working_directory,
        )?;

        let file_tree_depth = config.file_tree.depth.unwrap_or(3);
        if config.file_tree.include
            && config.file_tree.placement == placement
            && config.file_tree_mode == crate::agent_config::FileTreeMode::Full
        {
            append_built_in_if_match(
                &mut details_str,
                &format!("File Tree (max depth {})", file_tree_depth),
                utils::get_file_tree(&task.current_working_directory, Some(file_tree_depth)),
                &config.file_tree,
            )?;
        }

        // The system prompt is written once, when the task starts.
        let datetime_label = if placement == EnvironmentDetailPlacement::SystemPrompt {
            "Task Start Datetime (UTC)"
        } else {
            "Current Datetime (UTC)"
        };
        append_built_in_if_match(
            &mut details_str,
            datetime_label,
            Ok(utils::get_current_datetime_utc()),
            &config.current_datetime,
        )?;
        append_built_in_if_match(
            &mut details_str,
            "Operating System",
            Ok(utils::get_operating_system()),
            &config.operating_system,
        )?;

        if let Some(dev_details_list) = developer_env_details_override {
            for detail_config in dev_details_list {
                if detail_config.placement == placement {
                    let content_result = match detail_config.content_provider.as_ref() {
                        Some(DynamicContentProvider::Static(s)) => Ok(s.clone()),
                        Some(DynamicContentProvider::Dynamic(f)) => f(task).map_err(|e| {
                            log::warn!(
                                "Failed to gather dynamic developer environment detail '{}': {}",
                                detail_config.name,
                                e
                            );
                            ArkoAgentsError::Runtime(format!(
                                "Dynamic detail '{}' failed: {}",
                                detail_config.name, e
                            ))
                        }),
                        None => match &detail_config.static_content_for_serde {
                            Some(static_content) => Ok(static_content.clone()),
                            None => {
                                return Err(ArkoAgentsError::Configuration(
                                    detail_config.missing_content_error().unwrap_or_default(),
                                ));
                            }
                        },
                    };
                    match content_result {
                        Ok(raw_content_val) if !raw_content_val.is_empty() => {
                            details_str.push_str(&format!(
                                "{}:\n{}\n",
                                detail_config.name,
                                raw_content_val.trim_end()
                            ));
                        }
                        Ok(_) => {}
                        Err(_) => {}
                    }
                }
            }
        }

        Ok(details_str)
    }

    /// Formats a tool result as a user message.
    pub fn format_tool_result_as_user_message(
        &self,
        tool_name: &str,
        result: &str,
        user_message_on_approval: Option<&str>,
    ) -> String {
        let mut msg = format!(
            "Tool {} finished with the following result:\n{}",
            tool_name,
            result.trim()
        );
        if let Some(user_msg) = user_message_on_approval
            && !user_msg.trim().is_empty()
        {
            msg.push_str("\n\nUser provided additional message on approval:\n");
            msg.push_str(user_msg.trim());
        }
        msg
    }

    /// Formats a tool rejection as a user message.
    pub fn format_tool_rejection_as_user_message(
        &self,
        tool_name: &str,
        reason: Option<&str>,
    ) -> String {
        let reason_text = reason.unwrap_or("No specific reason provided.");
        format!(
            "Tool {} was rejected by the user. Reason: {}",
            tool_name,
            reason_text.trim()
        )
    }

    /// Formats a tool execution error as a user message.
    pub fn format_tool_execution_error_as_user_message(
        &self,
        tool_name: &str,
        error_message: &str,
    ) -> String {
        format!(
            "Tool {} failed during execution. Error: {}",
            tool_name,
            error_message.trim()
        )
    }

    /// Formats the user's response to ask_user questions as a user message.
    pub fn format_user_response_as_user_message(&self, questions: &str, response: &str) -> String {
        format!(
            "You asked the following questions:\n{}\n\nThe user responded:\n{}",
            questions.trim(),
            response.trim()
        )
    }
}
