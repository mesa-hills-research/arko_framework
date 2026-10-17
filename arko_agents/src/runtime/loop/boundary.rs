//! The turn boundary before a model request, and what it adds to the
//! conversation: context taken from the host, the file tree's changes, a user
//! turn to answer, and the runtime's own messages.

use super::*;

impl AgentLoop {
    /// Keeps context taken from the command channel with the task
    /// ([`Task::pending_context_injections`]) until it is added to the
    /// conversation, so a task that stops first never loses it.
    pub(super) fn queue_context_injection(&mut self, content: String) {
        if !content.trim().is_empty() {
            self.task.pending_context_injections.push(content);
        }
    }

    pub(super) fn observe_committed_tool_context(&mut self) {
        let tools = self
            .tool_registry
            .list_tool_names()
            .into_iter()
            .filter_map(|name| self.tool_registry.get(&name))
            .collect::<Vec<_>>();
        if self.tool_context_cursor > self.task.conversation.messages.len() {
            for tool in &tools {
                tool.reset_context(self.task.id);
            }
            self.tool_context_cursor = 0;
        }
        for message in &self.task.conversation.messages[self.tool_context_cursor..] {
            for tool in &tools {
                tool.observe_context(self.task.id, message);
            }
        }
        self.tool_context_cursor = self.task.conversation.messages.len();
    }

    pub(super) fn refresh_file_tree_context(&mut self) {
        if self.handover_report_due() {
            return;
        }
        let config = &self.task.agent_config.environment_details_config;
        if !config.file_tree.include
            || config.file_tree_mode != crate::agent_config::FileTreeMode::Diff
        {
            self.file_tree_baseline = None;
            self.file_tree_error = None;
            return;
        }
        let depth = config.file_tree.depth.unwrap_or(3);
        let message = match crate::file_tree::FileTreeSnapshot::capture(
            &self.task.current_working_directory,
            Some(depth),
        ) {
            Ok(snapshot) => {
                let recovered = self.file_tree_error.take().is_some();
                let update = match &self.file_tree_baseline {
                    Some(previous) => snapshot.changes_from(previous),
                    None => Some(format!("Full snapshot:\n{}", snapshot.full())),
                };
                self.file_tree_baseline = Some(snapshot);
                update
                    .map(|text| format!("# File tree (max depth {depth})\n{text}"))
                    .or_else(|| {
                        recovered
                            .then(|| "File-tree scan recovered; visible tree is unchanged.".into())
                    })
            }
            Err(error) => {
                let error = error.to_string();
                if self.file_tree_error.as_ref() == Some(&error) {
                    None
                } else {
                    self.file_tree_error = Some(error.clone());
                    Some(format!(
                        "File-tree scan failed; current inventory could not be checked: {error}"
                    ))
                }
            }
        };
        if let Some(message) = message {
            self.task.conversation.add_user_message(&message);
            self.task.mark_last_message_runtime();
            self.task.increment_user_message_count();
        }
    }

    /// At a safe turn boundary before a model request: the turn context
    /// source's context joins the context taken from the channel, and all of
    /// it goes into the conversation.
    pub(super) fn flush_context_injections(&mut self) -> Result<(), ArkoAgentsError> {
        if self.handover_report_due() {
            return Ok(());
        }
        if let Some(update) = self
            .turn_context
            .as_ref()
            .and_then(|source| source.context_update(self.task.id))
        {
            if !update.content.trim().is_empty() || !update.images.is_empty() {
                self.context_ids.extend(update.ids);
            }
            self.queue_context_injection(update.content);
            self.task.pending_context_images.extend(update.images);
        }
        self.add_taken_context()
    }

    /// Emits the `ContextInjected` events of the context updates added
    /// since the last call.
    pub(super) async fn emit_context_receipts(&mut self) -> Result<(), ArkoAgentsError> {
        for event in std::mem::take(&mut self.context_receipts) {
            self.emit_event(event).await?;
        }
        Ok(())
    }

    /// Adds the context taken from the command channel
    /// ([`Task::pending_context_injections`]) to the conversation as one
    /// user message headed `# Workflow context update`, in the order it was
    /// sent, with its images after the text. Only where a user message can
    /// go now (see
    /// [`context_can_be_added_now`](Self::context_can_be_added_now)).
    pub(super) fn add_taken_context(&mut self) -> Result<(), ArkoAgentsError> {
        // While the handover report is due, context waits for the fresh
        // conversation: added now, it would be read only through the report
        // and could be lost from it (a message that needs a reply, say).
        if (self.task.pending_context_injections.is_empty()
            && self.task.pending_context_images.is_empty())
            || self.handover_report_due()
        {
            return Ok(());
        }
        let mut content = self.task.pending_context_injections.join("\n\n");
        // Its images as the model can be sent them, within what a request
        // takes: the others are left out, with a line saying why.
        let images = std::mem::take(&mut self.task.pending_context_images);
        let (images, left_out) = crate::task::admit_context_images(
            &self.task.conversation.model,
            &self.task.conversation.messages,
            images,
        );
        for line in left_out {
            content.push_str("\n\n");
            content.push_str(&line);
        }
        let prompt_builder = self.prompts();
        let next_message_number = self.task.user_message_count + 1;
        // The message just before this one (tool results, say) already
        // carried this turn's environment details.
        let details_sent = self.details_sent_with.is_some()
            && self.details_sent_with == self.task.conversation.messages.len().checked_sub(1);
        let message = prompt_builder.augment_user_message(
            &self.task,
            &format!("# Workflow context update\n\n{content}"),
            next_message_number,
            !details_sent,
        )?;
        self.task
            .conversation
            .add_user_message_with_contents(crate::task::with_context_images(message, images));
        self.task.increment_user_message_count();
        // The host's identifiers of what it holds, for its receipt: those of
        // the content it sent, then the turn context source's.
        let mut ids = self
            .turn_context
            .as_ref()
            .map(|source| {
                self.task
                    .pending_context_injections
                    .iter()
                    .filter_map(|content| source.identify(content))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        ids.append(&mut self.context_ids);
        self.context_receipts.push(AgentEvent::ContextInjected {
            ids,
            bytes: content.len(),
        });
        self.task.pending_context_injections.clear();
        Ok(())
    }

    /// Whether continuing after an error-recovery pause sends the pending
    /// refusal fallback's request again, unchanged: a fallback is pending
    /// and no output-based results wait to be sent first.
    pub(super) fn fallback_resend(&self) -> bool {
        self.task.fallback_for.is_some()
            && (self.is_native_mode() || self.task.pending_text_tool_results.is_empty())
    }

    /// Whether a user message can join the conversation as the loop stops:
    /// no native call waits for its result (a user message between a call
    /// and its result breaks the next request for every provider), no
    /// command is awaited (whose answer, such as tool results, comes first),
    /// and no refusal is pending: a refusal fallback answers exactly the
    /// refused input, and [`Task::resolve_refusal`] needs the refused input
    /// to be the conversation's last input.
    pub(super) fn context_can_be_added_now(&self) -> bool {
        !self.results_pending()
            && self.task.fallback_for.is_none()
            && self.task.refused_input.is_none()
    }

    /// Whether the last turn's results are not all in the conversation yet:
    /// the task awaits a command (an approval, an answer, or a recovery
    /// decision), results wait with it, or the conversation ends on a native
    /// turn whose calls have none.
    pub(super) fn results_pending(&self) -> bool {
        let open_turn = self
            .task
            .conversation
            .messages
            .last()
            .is_some_and(|message| {
                message.role == ai::Role::Assistant && !message.tool_calls.is_empty()
            });
        open_turn
            || self.task.current_state.is_awaiting_approval()
            || !self.task.pending_native_tool_calls.is_empty()
            || !self.task.pending_native_tool_results.is_empty()
            || !self.task.pending_text_tool_results.is_empty()
    }

    /// As the loop stops, for any reason: context taken from the command
    /// channel goes into the conversation where it can
    /// ([`context_can_be_added_now`](Self::context_can_be_added_now)), so the
    /// saved transcript holds everything the host sent. Otherwise it stays in
    /// [`Task::pending_context_injections`], saved with the task, and is
    /// added at the first safe boundary after the task runs again. A
    /// completed task keeps it there too: no request followed it, and hosts
    /// such as arko_workflows take context in a completed task's
    /// conversation as read.
    pub(super) fn keep_taken_context(&mut self) -> Result<(), ArkoAgentsError> {
        if matches!(self.task.current_state, TaskState::Completed { .. })
            || !self.context_can_be_added_now()
        {
            return Ok(());
        }
        self.add_taken_context()
    }

    /// Records that the conversation's last message carries this turn's
    /// environment details and reminder.
    pub(super) fn mark_details_sent(&mut self) {
        self.details_sent_with = self.task.conversation.messages.len().checked_sub(1);
    }

    /// The turn boundary before a model request: a stop or the end of the
    /// task, the commands queued meanwhile, then the handover, context and
    /// file tree, a user turn to answer, and the monitors and the host's
    /// hold, any of which can end the pass instead of the request.
    pub(super) async fn turn_boundary(&mut self) -> Result<Flow, ArkoAgentsError> {
        // A suspension while approved calls ran stops the loop here.
        if self.stop_requested {
            return Ok(Flow::NextPass);
        }
        // Every command queued by now, also after an approved call ran,
        // so all context sent before this boundary (for example while a
        // tool call ran) reaches the next request together. The task is
        // not terminal here: a command or call that ended it broke out
        // of the loop above.
        if self.take_queued_commands().await? {
            return Ok(Flow::Exit);
        }

        // No path brings the task here in another state (a task that ended
        // or waits leaves the pass before): the check below is a guard.
        debug_assert!(
            matches!(self.task.current_state, TaskState::InProgress),
            "a turn boundary in state {:?}",
            self.task.current_state
        );
        if !matches!(self.task.current_state, TaskState::InProgress) {
            if self.task.current_state.is_terminal() {
                return Ok(Flow::Exit);
            }
            tokio_sleep(Duration::from_millis(100)).await;
            return Ok(Flow::NextPass);
        }

        // The fallback answers exactly the refused input: context updates
        // and file-tree changes wait for the next request.
        if self.task.fallback_for.is_none() {
            // First: context that arrives as a handover starts waits for
            // the fresh conversation.
            self.start_handover_if_due().await?;
            self.flush_context_injections()?;
            self.emit_context_receipts().await?;
            self.refresh_file_tree_context();
        }
        self.ensure_user_turn_before_request()?;
        if !self.apply_plan_usage_threshold().await? {
            return Ok(Flow::NextPass);
        }
        if let Some(message) = self.cache_monitor.pause_pending.take() {
            self.finish_monitor_pause(message).await?;
            return Ok(Flow::NextPass);
        }
        if let Some(reason) = self
            .turn_context
            .as_ref()
            .and_then(|source| source.hold_request(self.task.id))
        {
            self.hold_request(reason).await?;
            return Ok(Flow::NextPass);
        }
        self.observe_committed_tool_context();
        Ok(Flow::Proceed)
    }

    /// Adds a message of the runtime's own (a request to continue, a note
    /// on an error) as the next user message, with the environment details
    /// and reminder due with it.
    pub(super) fn add_runtime_message(&mut self, message: &str) -> Result<(), ArkoAgentsError> {
        let pb = self.prompts();
        let next_msg_num = self.task.user_message_count + 1;
        let augmented_message = pb.augment_user_message_with_env_details_and_reminder(
            &self.task,
            message,
            next_msg_num,
        )?;
        self.task.conversation.add_user_message(&augmented_message);
        self.task.mark_last_message_runtime();
        self.task.increment_user_message_count();
        Ok(())
    }

    /// Before a model request: a conversation that ends on an assistant
    /// turn (output kept from an interrupted stream, or a kept refusal, in a
    /// task resumed without a new message) gets a user message asking to
    /// continue, since several providers reject a request that ends on an
    /// assistant turn (Claude 4.6 and later reject prefill).
    fn ensure_user_turn_before_request(&mut self) -> Result<(), ArkoAgentsError> {
        let ends_on_assistant_text =
            self.task
                .conversation
                .messages
                .last()
                .is_some_and(|message| {
                    message.role == ai::Role::Assistant && message.tool_calls.is_empty()
                });
        if ends_on_assistant_text {
            self.add_runtime_message(CONTINUE_FROM_PARTIAL_TURN)?;
        }
        Ok(())
    }
}
