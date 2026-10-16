//! The in-place handover of a long conversation: the request for the agent's
//! report, the report turn, and the fresh conversation that starts from it.

use super::*;
use crate::handover::HANDOVER_REPORT_PARAM_NAME;

impl AgentLoop {
    /// Notes a host's [`UserCommand::RequestHandover`]: the report is
    /// requested at the next boundary where a user message can go. Ignored
    /// while a handover is under way.
    pub(super) fn note_handover_request(&mut self) {
        if self.task.handover_requested.is_none() {
            self.task.handover_requested = Some(HandoverRequest::new(
                self.task.current_conversation_context_size().unwrap_or(0),
            ));
        }
    }

    /// The agent was asked for its handover report and the conversation
    /// hasn't been replaced yet (the report may have come with
    /// `handover_report`, for the next turn boundary).
    pub(super) fn handover_report_due(&self) -> bool {
        self.task
            .handover_requested
            .as_ref()
            .is_some_and(|request| request.report_requested)
    }

    /// Whether the agent gives its report with the `handover_report` tool,
    /// which its native requests then offer: native tool calling with
    /// `handover` set. Otherwise the report is a reply without a tool call.
    pub(super) fn offers_report_tool(&self) -> bool {
        self.is_native_mode() && self.task.agent_config.handover.is_some()
    }

    /// Whether a call to `tool_name` is rejected because the handover report
    /// is due: every call except `complete_task` (and `handover_report`,
    /// which is answered before this check).
    pub(super) fn handover_blocks(&self, tool_name: &str) -> bool {
        self.handover_report_due() && tool_name != COMPLETE_TASK_TOOL_NAME
    }

    /// The result of a call rejected because the handover report is due.
    pub(super) fn handover_report_due_reason(&self) -> &'static str {
        let completes = !self.task.agent_config.tool_config.ends_task_with_text();
        match (self.offers_report_tool(), completes) {
            (true, true) => HANDOVER_REPORT_TOOL_DUE,
            (true, false) => HANDOVER_REPORT_TOOL_DUE_NO_COMPLETION,
            (false, true) => HANDOVER_REPORT_DUE,
            (false, false) => HANDOVER_REPORT_DUE_NO_COMPLETION,
        }
    }

    /// Answers a `handover_report` call with `arguments`, offered to this
    /// agent: `Ok` with its result when the report is taken (kept in the
    /// task's handover request for the next turn boundary; a second report
    /// in the same turn is answered as received and ignored), `Err` with the
    /// reason it was rejected: no report is due, or the call has none.
    pub(super) fn take_report_call(
        &mut self,
        arguments: &HashMap<String, Value>,
    ) -> Result<&'static str, &'static str> {
        let Some(request) = self
            .task
            .handover_requested
            .as_mut()
            .filter(|request| request.report_requested)
        else {
            return Err(REPORT_NOT_REQUESTED);
        };
        let report = arguments
            .get(HANDOVER_REPORT_PARAM_NAME)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|report| !report.is_empty())
            .ok_or(EMPTY_REPORT_CALL)?;
        if request.report.is_none() {
            request.report = Some(report.to_string());
        }
        Ok(REPORT_RECEIVED)
    }

    /// Before a model request (not a refusal fallback's): replaces the
    /// conversation when the agent gave its report with `handover_report`
    /// and the turn's results are in; otherwise asks the agent for its
    /// handover report when a host asked for a handover or the current
    /// conversation reached the threshold (the host's, else the configured
    /// one), and a user message can go into the conversation.
    pub(super) async fn start_handover_if_due(&mut self) -> Result<(), ArkoAgentsError> {
        let received = self
            .task
            .handover_requested
            .as_ref()
            .and_then(|request| request.report.clone());
        if let Some(report) = received {
            if self.context_can_be_added_now() {
                self.complete_handover(report).await?;
            }
            return Ok(());
        }
        if self.handover_report_due() || !self.context_can_be_added_now() {
            return Ok(());
        }
        let context_size = self.task.current_conversation_context_size();
        let config = self.task.agent_config.handover.clone();
        if self.task.handover_requested.is_none() {
            let threshold = self.handover_threshold.of(&self.task.agent_config);
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
        let with_tool = self.offers_report_tool();
        let request = self
            .task
            .handover_requested
            .get_or_insert_with(|| HandoverRequest::new(context_size.unwrap_or(0)));
        request.report_requested = true;
        let input_tokens = request.input_tokens;
        let completes = !self.task.agent_config.tool_config.ends_task_with_text();
        let report_request = match &config {
            Some(config) => config.report_request_for(with_tool, completes),
            None => crate::handover::default_report_request_for(with_tool, completes),
        };
        if std::mem::take(&mut self.step_left_out) {
            self.task.conversation.add_user_message(&format!(
                "{}\n\n{report_request}",
                crate::handover::STEP_LEFT_OUT
            ));
        } else {
            self.task.conversation.add_user_message(report_request);
        }
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
    /// call. An agent offered `handover_report` gives the report with it, so
    /// its reply is answered with a note asking for that call.
    pub(super) async fn take_handover_report(
        &mut self,
        report: String,
    ) -> Result<(), ArkoAgentsError> {
        if self.offers_report_tool() {
            let note = if self.task.agent_config.tool_config.ends_task_with_text() {
                NO_REPORT_TOOL_CALL_NO_COMPLETION
            } else {
                NO_REPORT_TOOL_CALL
            };
            return self.handle_model_output_failure(note.to_string()).await;
        }
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
        self.cache_monitor = CacheMonitorState::default();
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
