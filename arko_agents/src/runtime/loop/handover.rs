//! The in-place handover of a long conversation: the request for the agent's
//! report, the report turn, and the fresh conversation that starts from it.

use super::*;

impl AgentLoop {
    /// Notes a host's [`UserCommand::RequestHandover`]: the report is
    /// requested at the next boundary where a user message can go. Ignored
    /// while a handover is under way.
    pub(super) fn note_handover_request(&mut self) {
        if self.task.handover_requested.is_none() {
            self.task.handover_requested = Some(HandoverRequest {
                requested_at: chrono::Utc::now(),
                input_tokens: self.task.current_conversation_context_size().unwrap_or(0),
                report_requested: false,
            });
        }
    }

    /// The agent was asked for its handover report and hasn't written it.
    pub(super) fn handover_report_due(&self) -> bool {
        self.task
            .handover_requested
            .as_ref()
            .is_some_and(|request| request.report_requested)
    }

    /// Whether a call to `tool_name` is rejected because the handover report
    /// is due: every call except `complete_task`.
    pub(super) fn handover_blocks(&self, tool_name: &str) -> bool {
        self.handover_report_due() && tool_name != COMPLETE_TASK_TOOL_NAME
    }

    /// Before a model request (not a refusal fallback's): asks the agent for
    /// its handover report when a host asked for a handover or the current
    /// conversation reached the configured threshold, and a user message can
    /// go into the conversation.
    pub(super) async fn start_handover_if_due(&mut self) -> Result<(), ArkoAgentsError> {
        if self.handover_report_due() || !self.context_can_be_added_now() {
            return Ok(());
        }
        let context_size = self.task.current_conversation_context_size();
        let config = self.task.agent_config.handover.clone();
        if self.task.handover_requested.is_none() {
            let threshold = config.as_ref().and_then(|config| {
                config.threshold(self.task.agent_config.llm_config.model.max_input_tokens)
            });
            let Some((threshold, size)) = threshold.zip(context_size) else {
                return Ok(());
            };
            // The request about to be sent, not the one before it: what
            // the turn added since (its tool results, above all) counts.
            let coming = self.estimated_request_tokens().max(size);
            if coming < threshold {
                return Ok(());
            }
            // A conversation whose first request was already over the
            // threshold would start over the threshold again: handing over
            // can't make it smaller.
            let first = self
                .task
                .usage_history
                .get(self.task.conversation_usage_start)
                .map_or(0, |usage| usage.input_tokens);
            if first >= threshold {
                if !std::mem::replace(&mut self.handover_skip_warned, true) {
                    log::warn!(
                        "Task {}: No handover: the conversation's first request ({} input tokens) was already over the handover threshold ({}).",
                        self.task.id,
                        first,
                        threshold
                    );
                }
                return Ok(());
            }
        }
        let request = self
            .task
            .handover_requested
            .get_or_insert_with(|| HandoverRequest {
                requested_at: chrono::Utc::now(),
                input_tokens: context_size.unwrap_or(0),
                report_requested: false,
            });
        request.report_requested = true;
        let input_tokens = request.input_tokens;
        let report_request = config
            .as_ref()
            .map_or(crate::handover::DEFAULT_REPORT_REQUEST, |config| {
                config.report_request()
            });
        self.task.conversation.add_user_message(report_request);
        self.task.mark_last_message_runtime();
        self.task.increment_user_message_count();
        log::info!(
            "Task {}: Handover requested at {} input tokens.",
            self.task.id,
            input_tokens
        );
        self.emit_event(AgentEvent::HandoverRequested { input_tokens })
            .await
    }

    /// A reply without tool calls while the handover report is due: its
    /// text is the report, and the conversation is handed over; a reply
    /// without text is answered with a note, like a turn without a tool
    /// call.
    pub(super) async fn take_handover_report(
        &mut self,
        report: String,
    ) -> Result<(), ArkoAgentsError> {
        if report.trim().is_empty() {
            return self
                .handle_model_output_failure(NO_HANDOVER_REPORT.to_string())
                .await;
        }
        self.complete_handover(report).await
    }

    /// Replaces the conversation with a fresh one that starts from the
    /// seed (the [`HandoverHook`]'s, or the task's initial prompt) and the
    /// report, and resets what the loop keeps about the old one.
    async fn complete_handover(&mut self, report: String) -> Result<(), ArkoAgentsError> {
        let report = report.trim().to_string();
        let seed = match &self.handover_hook {
            Some(hook) => hook.on_handover(&self.task, &report),
            None => HandoverSeed::for_task(&self.task),
        };
        let input_tokens = self
            .task
            .handover_requested
            .take()
            .map_or(0, |request| request.input_tokens);
        let replaced_messages = self.task.start_fresh_conversation();
        self.task.handovers.push(HandoverRecord {
            at: chrono::Utc::now(),
            input_tokens,
            replaced_messages,
            report: report.clone(),
        });
        let prompt_builder = self.prompts();
        let first_message =
            prompt_builder.format_initial_task_prompt(&self.task, &seed.message(&report))?;
        self.task.conversation.add_user_message(&first_message);
        self.task.increment_user_message_count();

        // What the loop keeps about the replaced conversation: tools that
        // track what the conversation holds start over (see
        // `observe_committed_tool_context`), the file tree is sent in full
        // again, and the cache monitor judges the fresh conversation alone.
        self.tool_context_cursor = usize::MAX;
        self.file_tree_baseline = None;
        self.file_tree_error = None;
        self.details_sent_with = None;
        self.cache_warned = false;
        self.cache_pause_episode = false;
        self.cache_pause_pending = None;
        self.handover_skip_warned = false;
        self.reset_llm_recovery();

        log::info!(
            "Task {}: Conversation handed over ({} messages replaced).",
            self.task.id,
            replaced_messages
        );
        let handovers = self.task.handovers.len();
        // The fresh conversation replaces the saved one.
        self.report_task(false).await;
        self.emit_event(AgentEvent::ConversationHandedOver {
            report,
            input_tokens,
            handovers,
        })
        .await
    }
}
