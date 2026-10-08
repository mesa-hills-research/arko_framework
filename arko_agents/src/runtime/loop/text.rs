//! Output-based tool calling: a turn's parsed calls reported, checked and run
//! in order or held for approval, each call's result recorded as text, and the
//! turn's results sent as the next message.

use super::*;

/// What became of one call of an output-based turn; see
/// [`AgentLoop::handle_text_call`].
enum TextCall {
    /// Answered or run: the turn goes on with its next call.
    Handled,
    /// It waits for approval.
    NeedsApproval(PendingToolCallInfo),
    /// The task stopped or no longer runs calls: the turn's other calls
    /// don't run.
    EndsBatch,
}

impl AgentLoop {
    /// Records an output-based call's result (nothing for an empty one) in
    /// the task as it is produced ([`Task::pending_text_tool_results`]), so
    /// that a snapshot taken meanwhile holds it, and reports the task.
    pub(super) async fn record_text_result(&mut self, result: String) {
        if !result.is_empty() {
            self.task.pending_text_tool_results.push(result);
        }
        self.report_task(false).await;
    }

    /// Takes the output-based results recorded since `start` back out of
    /// the task, in order.
    pub(super) fn take_text_results(&mut self, start: usize) -> Vec<String> {
        let results = &mut self.task.pending_text_tool_results;
        results.split_off(start.min(results.len()))
    }

    fn queue_tool_images(
        &mut self,
        tool: &str,
        call_id: impl std::fmt::Display,
        images: Vec<ToolImage>,
    ) {
        self.task
            .pending_tool_images
            .extend(images.into_iter().map(|mut attachment| {
                attachment.description = format!(
                    "Image from tool {tool}, call {call_id}: {}",
                    attachment.description
                );
                attachment
            }));
    }

    /// Adds output-based results (or an answer, or a note) as the next
    /// user message, with the environment details and reminder due with it
    /// and the tool images waiting for it.
    pub(super) fn add_results_message(&mut self, text: &str) -> Result<(), ArkoAgentsError> {
        let message = self
            .prompts()
            .augment_user_message_with_env_details_and_reminder(
                &self.task,
                text,
                self.task.user_message_count + 1,
            )?;
        self.add_text_tool_results_to_conversation(&message);
        self.mark_details_sent();
        self.task.increment_user_message_count();
        Ok(())
    }

    pub(super) fn add_text_tool_results_to_conversation(&mut self, text: &str) {
        let mut content = vec![ai::Content::Text(text.to_string())];
        content.extend(self.task.take_tool_image_content());
        self.task
            .conversation
            .add_user_message_with_contents(content);
    }

    /// An output-based turn. The parser's events are reported, then the
    /// turn's one call, or batch of calls of one tool, is checked, and in
    /// order its calls run or wait for approval; their results go into the
    /// conversation as one message (or wait with the task). A turn without
    /// a call is the handover report, or a response without a tool call.
    pub(super) async fn text_turn(&mut self, turn: StreamedTurn) -> Result<Flow, ArkoAgentsError> {
        let StreamedTurn {
            response: final_generation_response,
            parser_events: raw_events_for_turn,
            ..
        } = turn;
        let mut parsed_call_ids = self.emit_parser_events(&raw_events_for_turn).await?;

        let mut turn_message = String::new();

        match assemble_tool_calls(raw_events_for_turn) {
            Ok((plain_text_from_assembler, mut tool_call_groups)) => {
                if self.simplified_tool_events && !plain_text_from_assembler.is_empty() {
                    self.text_buffer.push_str(&plain_text_from_assembler);
                    self.emit_throttled_text().await?;
                    self.flush_throttled_text().await?;
                }

                if tool_call_groups.len() > 1 {
                    // The turn's message is still empty.
                    debug_assert!(turn_message.is_empty());
                    turn_message.push_str(
                        "Error: Only one tool call or one batch of the same tool can be used per turn.",
                    );
                    // None of the turn's calls runs.
                    self.reject_announced_text_calls(&mut parsed_call_ids, &turn_message)
                        .await?;
                } else if tool_call_groups.is_empty() {
                    if self.handover_report_due() {
                        self.take_handover_report(plain_text_from_assembler.clone())
                            .await?;
                        return Ok(self.end_of_pass());
                    }
                    // The stream left the task in progress.
                    debug_assert!(!self.task.current_state.is_terminal());
                    self.handle_model_output_failure(NO_TEXT_TOOL_CALL.to_string())
                        .await?;
                    return Ok(self.end_of_pass());
                } else {
                    let current_group = tool_call_groups.remove(0);
                    self.handle_text_group(current_group, &mut parsed_call_ids, &mut turn_message)
                        .await?;
                    // Every call the parser announced belongs to this group.
                    debug_assert!(parsed_call_ids.is_empty());
                }
            }
            Err(e) => {
                log::error!(
                    "Failed to assemble tool calls for task {}: {}",
                    self.task.id,
                    e
                );
                let note = format!(
                    "Your tool call could not be read ({e}). Write it again in the format the system prompt describes."
                );
                // The calls announced before the turn proved unreadable
                // don't run.
                self.reject_announced_text_calls(&mut parsed_call_ids, &note)
                    .await?;
                self.handle_model_output_failure(note).await?;
                return Ok(self.end_of_pass());
            }
        }

        self.text_turn_message(turn_message, &final_generation_response)?;
        Ok(Flow::Proceed)
    }

    /// The calls of an output-based turn's group: each is checked and run
    /// or held for approval, in order, until one ends the batch; then, as
    /// in native mode, a turn a call stopped answers the calls awaiting
    /// approval and the calls it didn't reach as not run, and otherwise the
    /// calls awaiting approval wait for it. The batch's tool errors pause
    /// the task, and the results become the turn's message (for a batch of
    /// a tool that can't be batched, which runs no call, the error saying
    /// so, its announced calls rejected).
    async fn handle_text_group(
        &mut self,
        current_group: ToolCallGroup,
        parsed_call_ids: &mut VecDeque<(Uuid, String)>,
        turn_message: &mut String,
    ) -> Result<(), ArkoAgentsError> {
        let calls_to_process_with_ids =
            self.calls_of_group(current_group, parsed_call_ids, turn_message);
        if !turn_message.is_empty() {
            // The batch can't run: its announced calls get the error.
            self.reject_announced_text_calls(parsed_call_ids, turn_message)
                .await?;
        }

        let mut calls_needing_approval = Vec::new();
        // The results of the calls handled, recorded in
        // the task as they are produced.
        let start = self.task.pending_text_tool_results.len();
        let overall_batch_id = if calls_to_process_with_ids.len() > 1 {
            Some(Uuid::new_v4())
        } else {
            None
        };
        self.begin_tool_batch();

        let mut calls = calls_to_process_with_ids.into_iter();
        for (call, tool_call_id) in calls.by_ref() {
            match self.handle_text_call(&call, tool_call_id).await? {
                TextCall::Handled => {}
                TextCall::NeedsApproval(pending) => calls_needing_approval.push(pending),
                TextCall::EndsBatch => break,
            }
        }

        // A turn a call stopped (it ended the task or made it wait, or
        // the task was suspended while it ran) answers every other call
        // once, as native mode does: the calls awaiting approval are not
        // asked about, and the calls after it don't run.
        if self.stop_requested || !matches!(self.task.current_state, TaskState::InProgress) {
            let not_run = self.not_run_after_stop();
            let unanswered: Vec<(Uuid, String)> = calls_needing_approval
                .drain(..)
                .map(|pending| (pending.tool_call_id, pending.tool_name))
                .chain(calls.map(|(call, tool_call_id)| (tool_call_id, call.name)))
                .collect();
            for (tool_call_id, tool_name) in unanswered {
                self.answer_text_call_not_run(tool_call_id, &tool_name, &not_run)
                    .await?;
            }
        }
        if !calls_needing_approval.is_empty()
            && matches!(self.task.current_state, TaskState::InProgress)
        {
            self.await_approvals(calls_needing_approval, overall_batch_id)
                .await?;
        }
        // Pauses on the batch's tool errors now, or after
        // the calls still awaiting approval are decided.
        self.end_tool_batch().await?;
        // The results of the calls that ran, which wait
        // with the task below if it now waits. A batch that
        // couldn't run has none, and keeps the error saying why.
        let results = self.take_text_results(start).join("\n\n---\n\n");
        if !results.is_empty() {
            debug_assert!(turn_message.is_empty(), "a batch that ran has no error");
            *turn_message = results;
        }
        Ok(())
    }

    /// An output-based call that doesn't run (`reason` says why): it is
    /// reported rejected, and the turn's message says so.
    pub(super) async fn answer_text_call_not_run(
        &mut self,
        tool_call_id: Uuid,
        tool_name: &str,
        reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        self.emit_event(AgentEvent::ToolRejected {
            tool_call_id,
            tool_name: tool_name.to_string(),
            reason: Some(reason.to_string()),
        })
        .await?;
        self.record_text_result(format!("Tool {tool_name}: {reason}"))
            .await;
        Ok(())
    }

    /// Rejects, in order, the calls the parser announced
    /// (`ToolCallInitiated`) that the turn won't run, with `reason`, the
    /// error the turn's message gives: every announced call gets an
    /// outcome, as in native mode.
    async fn reject_announced_text_calls(
        &self,
        announced: &mut VecDeque<(Uuid, String)>,
        reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        for (tool_call_id, tool_name) in announced.drain(..) {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name,
                reason: Some(reason.to_string()),
            })
            .await?;
        }
        Ok(())
    }

    /// Reports an output-based turn's parser events (unless tool events
    /// are simplified): `ToolCallInitiated` and the parameters of each
    /// call, then the turn's plain text. Returns the ids given to the
    /// calls, with their tools' names, in order.
    async fn emit_parser_events(
        &mut self,
        raw_events_for_turn: &[RawParserEvent],
    ) -> Result<VecDeque<(Uuid, String)>, ArkoAgentsError> {
        let mut plain_text = String::new();
        let mut parsed_call_ids = VecDeque::new();

        if !self.simplified_tool_events {
            let mut current_tool_call_id_granular: Option<Uuid> = None;
            let mut current_param_name_granular: Option<String> = None;

            for raw_event in raw_events_for_turn {
                match raw_event {
                    RawParserEvent::Text(text) => {
                        plain_text.push_str(text);
                    }
                    RawParserEvent::Error(err) => {
                        log::warn!("Raw parser emitted error event: {}", err);
                    }
                    RawParserEvent::ToolCallStart(name) => {
                        let tool_call_id = Uuid::new_v4();
                        parsed_call_ids.push_back((tool_call_id, name.clone()));
                        current_tool_call_id_granular = Some(tool_call_id);
                        self.emit_event(AgentEvent::ToolCallInitiated {
                            tool_call_id,
                            tool_name: name.clone(),
                        })
                        .await?;
                    }
                    RawParserEvent::ParamName(p_name) => {
                        if let Some(id) = current_tool_call_id_granular {
                            current_param_name_granular = Some(p_name.clone());
                            self.emit_event(AgentEvent::ToolParamName {
                                tool_call_id: id,
                                param_name: p_name.clone(),
                            })
                            .await?;
                        }
                    }
                    RawParserEvent::ParamValueChunk(chunk) => {
                        if let Some(id) = current_tool_call_id_granular
                            && let Some(ref param_name) = current_param_name_granular
                        {
                            self.emit_event(AgentEvent::ToolParamValueChunk {
                                tool_call_id: id,
                                param_name: param_name.clone(),
                                chunk: chunk.clone(),
                            })
                            .await?;
                        }
                    }
                    RawParserEvent::ParamEnd => {
                        if let Some(id) = current_tool_call_id_granular
                            && let Some(ref param_name) = current_param_name_granular
                        {
                            self.emit_event(AgentEvent::ToolParamEnd {
                                tool_call_id: id,
                                param_name: param_name.clone(),
                            })
                            .await?;
                        }
                        current_param_name_granular = None;
                    }
                    RawParserEvent::ToolCallEnd(_) => {
                        current_tool_call_id_granular = None;
                        current_param_name_granular = None;
                    }
                    RawParserEvent::BatchedToolCallStart | RawParserEvent::BatchedToolCallEnd => {}
                }
            }
            if !plain_text.is_empty() {
                self.text_buffer.push_str(&plain_text);
                self.emit_throttled_text().await?;
            }
        }
        self.flush_throttled_text().await?;
        Ok(parsed_call_ids)
    }

    /// The calls of an output-based turn's group, with their ids (those
    /// the parser events gave, or new ones), and their boolean parameters
    /// made booleans. A batch of a tool that can't be batched has no calls:
    /// the turn's message says why.
    fn calls_of_group(
        &self,
        current_group: ToolCallGroup,
        parsed_call_ids: &mut VecDeque<(Uuid, String)>,
        turn_message: &mut String,
    ) -> Vec<(ParsedToolCall, Uuid)> {
        // The id the parser gave the next call, or a new one.
        let mut next_id = || {
            parsed_call_ids
                .pop_front()
                .map_or_else(Uuid::new_v4, |(id, _)| id)
        };
        let calls_to_process_with_ids: Vec<(ParsedToolCall, Uuid)> = match current_group {
            ToolCallGroup::Single(pc) => {
                vec![(pc, next_id())]
            }
            ToolCallGroup::Batch(pcs) => {
                if let Some(first_tool_name_in_batch) = pcs.first().map(|call| &call.name) {
                    if let Some(tool_def_arc) = self.tool_registry.get(first_tool_name_in_batch) {
                        if !self
                            .task
                            .agent_config
                            .allows_batched_calls_for_tool(&tool_def_arc.definition())
                        {
                            let err_msg = format!(
                                "Error: Tool '{}' cannot be used in a batch.",
                                first_tool_name_in_batch
                            );
                            // The turn's message is still empty.
                            debug_assert!(turn_message.is_empty());
                            turn_message.push_str(&err_msg);
                            Vec::new()
                        } else {
                            pcs.into_iter().map(|pc| (pc, next_id())).collect()
                        }
                    } else {
                        pcs.into_iter().map(|pc| (pc, next_id())).collect()
                    }
                } else {
                    Vec::new()
                }
            }
        };

        // Text calls carry every value as a string: give
        // boolean parameters booleans, as in native mode.
        let mut calls_to_process_with_ids = calls_to_process_with_ids;
        for (call, _) in &mut calls_to_process_with_ids {
            if let Some(tool) = self.tool_registry.get(&call.name) {
                tool.definition()
                    .coerce_text_arguments(&mut call.parameters);
            }
        }
        calls_to_process_with_ids
    }

    /// One call of an output-based turn: reported, then checked (a due
    /// handover report, a tool the run doesn't offer, its paths), then run
    /// or, if it needs approval, returned to wait for it. Its result is
    /// recorded in the task.
    async fn handle_text_call(
        &mut self,
        call: &ParsedToolCall,
        tool_call_id: Uuid,
    ) -> Result<TextCall, ArkoAgentsError> {
        if self.simplified_tool_events {
            self.emit_event(AgentEvent::ToolCallInitiated {
                tool_call_id,
                tool_name: call.name.clone(),
            })
            .await?;
        }
        self.emit_event(AgentEvent::ToolCallFormatted {
            tool_call_id,
            tool_name: call.name.clone(),
            parameters: call.parameters.clone(),
        })
        .await?;

        if self.handover_blocks(&call.name) {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: call.name.clone(),
                reason: Some(HANDOVER_REPORT_DUE.to_string()),
            })
            .await?;
            self.record_text_result(
                self.prompts()
                    .format_tool_execution_error_as_user_message(&call.name, HANDOVER_REPORT_DUE),
            )
            .await;
            return Ok(TextCall::Handled);
        }

        let tool_arc = match self.tool_registry.get(&call.name) {
            Some(t) => t,
            None => {
                self.emit_event(AgentEvent::UnavailableToolAttempted {
                    tool_name: call.name.clone(),
                })
                .await?;
                self.record_text_result(format!("Error: Tool '{}' is not available.", call.name))
                    .await;
                return Ok(TextCall::Handled);
            }
        };
        let tool_definition = tool_arc.definition();

        // The path parameter that forces an approval.
        let mut outside_workspace = None;
        let path_rejection = match self.path_verdict(&call.name, &tool_definition, &call.parameters)
        {
            PathVerdict::Allowed => None,
            PathVerdict::NeedsApproval(param_name) => {
                outside_workspace = Some(param_name);
                None
            }
            PathVerdict::Outside(param_name) => Some(crate::tool::ToolError::PathAccessRejected {
                param_name,
                workspace_path: self
                    .task
                    .current_working_directory
                    .to_string_lossy()
                    .to_string(),
            }),
            PathVerdict::Invalid(reason) => Some(crate::tool::ToolError::ExecutionFailed(reason)),
        };
        if let Some(tool_error) = path_rejection {
            let error_message_for_llm = self
                .handle_tool_exec_error(&call.name, tool_error, Some(tool_call_id))
                .await?;
            self.record_text_result(error_message_for_llm).await;
            // Under `FailTask` the rest of the batch stops.
            if self.task.current_state.is_terminal() {
                return Ok(TextCall::EndsBatch);
            }
            return Ok(TextCall::Handled);
        }

        let normal_approval_needed = !self.task.agent_config.should_auto_approve_tool(&call.name);

        if outside_workspace.is_some() || normal_approval_needed {
            Ok(TextCall::NeedsApproval(PendingToolCallInfo {
                tool_call_id,
                tool_name: call.name.clone(),
                parameters: call.parameters.clone(),
                tool_description: tool_definition.description.clone(),
                approval_reason: ApprovalReason {
                    tool_not_auto_approved: normal_approval_needed,
                    path_outside_workspace: outside_workspace,
                },
            }))
        } else {
            match self
                .execute_text_call(
                    &call.name,
                    call.parameters.clone(),
                    tool_call_id,
                    None,
                    None,
                    false,
                )
                .await
            {
                Ok(msg_part) => self.record_text_result(msg_part).await,
                Err(e) => log::error!(
                    "Task {}: Error during auto-approved tool execution: {}",
                    self.task.id,
                    e
                ),
            }
            if !matches!(self.task.current_state, TaskState::InProgress) || self.stop_requested {
                return Ok(TextCall::EndsBatch);
            }
            Ok(TextCall::Handled)
        }
    }

    /// The message after an output-based turn: its results (which wait with
    /// the task if it now waits), or a request to use a tool.
    fn text_turn_message(
        &mut self,
        mut turn_message: String,
        final_generation_response: &Option<ai::GenerationResponse>,
    ) -> Result<(), ArkoAgentsError> {
        if final_generation_response.is_some() && !self.task.current_state.is_awaiting_approval() {
            self.reset_llm_recovery();
        }

        // Results of calls that ran in a turn that now waits (for
        // other calls' approval, or on a pause) wait with it.
        if !turn_message.is_empty() && self.task.current_state.is_awaiting_approval() {
            self.task
                .pending_text_tool_results
                .push(std::mem::take(&mut turn_message));
        }
        if !turn_message.is_empty()
            && !self.task.current_state.is_terminal()
            && !self.task.current_state.is_awaiting_approval()
        {
            self.add_results_message(&turn_message)?;
        } else if turn_message.is_empty()
            && !self.task.current_state.is_terminal()
            && !self.task.current_state.is_awaiting_approval()
            && final_generation_response.is_some()
        {
            self.add_runtime_message(CONTINUE_WITH_A_TOOL)?;
        }
        Ok(())
    }

    /// Runs an output-based call (auto-approved or approved) and returns
    /// its result as text for the next user message. A cancellation or
    /// suspension while it runs stops it. `recovering`: the call runs again
    /// after the task stopped while it ran (see
    /// [`TaskContext::recovering`](crate::tool::TaskContext::recovering)).
    pub(super) async fn execute_text_call(
        &mut self,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        tool_call_id: Uuid,
        custom_tool_output: Option<String>,
        message_to_agent_on_approval: Option<String>,
        recovering: bool,
    ) -> Result<String, ArkoAgentsError> {
        let result = self
            .run_text_call(
                tool_name,
                parameters,
                tool_call_id,
                custom_tool_output,
                message_to_agent_on_approval,
                recovering,
            )
            .await;
        // Its result is the caller's to record.
        self.call_settled(tool_call_id);
        result
    }

    /// The call itself; see [`execute_text_call`](Self::execute_text_call).
    async fn run_text_call(
        &mut self,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        tool_call_id: Uuid,
        custom_tool_output: Option<String>,
        message_to_agent_on_approval: Option<String>,
        recovering: bool,
    ) -> Result<String, ArkoAgentsError> {
        // A call saved awaiting approval whose tool this run does not offer
        // is answered like any unavailable tool.
        let Some(tool_arc) = self.tool_registry.get(tool_name) else {
            self.emit_event(AgentEvent::UnavailableToolAttempted {
                tool_name: tool_name.to_string(),
            })
            .await?;
            return Ok(format!("Error: Tool '{tool_name}' is not available."));
        };

        let format_error = |this: &Self, message: &str| {
            this.prompts()
                .format_tool_execution_error_as_user_message(tool_name, message)
        };
        let mut run = self.tool_run(
            tool_call_id,
            tool_name,
            tool_arc.clone(),
            parameters,
            custom_tool_output,
        );
        run.message_to_agent = message_to_agent_on_approval.clone();
        run.recovering = recovering;
        let mut group = self.run_group(vec![run]).await;
        let outcome = match group.outcomes.pop().flatten() {
            Some(outcome) => outcome?,
            None => {
                let stop = group.stop.expect("a call without an outcome was stopped");
                self.stop_after_calls(stop).await?;
                let reason = stopped_call_reason(stop, &self.task.current_state);
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id,
                    tool_name: tool_name.to_string(),
                    reason: Some(reason.to_string()),
                })
                .await?;
                return Ok(format_error(self, reason));
            }
        };
        let result = self
            .finish_text_call(
                tool_name,
                tool_call_id,
                outcome,
                message_to_agent_on_approval,
            )
            .await;
        if let Some(stop) = group.stop {
            self.stop_after_calls(stop).await?;
        }
        result
    }

    /// The text result of an output-based call that ran; see
    /// [`execute_text_call`](Self::execute_text_call).
    async fn finish_text_call(
        &mut self,
        tool_name: &str,
        tool_call_id: Uuid,
        outcome: RunOutcome,
        message_to_agent_on_approval: Option<String>,
    ) -> Result<String, ArkoAgentsError> {
        let format_error = |this: &Self, message: &str| {
            this.prompts()
                .format_tool_execution_error_as_user_message(tool_name, message)
        };
        match outcome {
            RunOutcome::Rejected(message) => {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id,
                    tool_name: tool_name.to_string(),
                    reason: Some(message.clone()),
                })
                .await?;
                Ok(format_error(self, &message))
            }
            RunOutcome::MiddlewareErrorBefore { reason } => {
                let formatted = format_error(self, &reason);
                self.pause_by_middleware(tool_call_id, tool_name, &reason, reason.clone())
                    .await?;
                Ok(formatted)
            }
            RunOutcome::MiddlewareErrorAfter {
                output,
                effective,
                intercepted,
                reason,
                is_error,
            } => {
                let raw = output.text;
                self.emit_tool_result_events(
                    tool_call_id,
                    tool_name,
                    &raw,
                    &effective,
                    intercepted,
                )
                .await?;
                let formatted = if is_error {
                    format_error(self, &effective)
                } else {
                    self.prompts().format_tool_result_as_user_message(
                        tool_name,
                        &effective,
                        message_to_agent_on_approval.as_deref(),
                    )
                };
                self.queue_tool_images(tool_name, tool_call_id, output.images);
                self.pause_for_middleware_error(tool_call_id, tool_name, reason)
                    .await?;
                Ok(formatted)
            }
            RunOutcome::Stop { reason, cancel } => {
                let formatted = format_error(self, &reason);
                self.stop_by_middleware(tool_call_id, tool_name, &reason, cancel)
                    .await?;
                Ok(formatted)
            }
            RunOutcome::Pause { message, reason } => {
                let formatted = format_error(self, &message);
                self.pause_by_middleware(tool_call_id, tool_name, &message, reason)
                    .await?;
                Ok(formatted)
            }
            RunOutcome::Failed(error) => {
                self.handle_tool_exec_error(tool_name, error, Some(tool_call_id))
                    .await
            }
            RunOutcome::Finished {
                output,
                effective,
                intercepted,
            } => {
                self.finish_text_result(
                    tool_name,
                    tool_call_id,
                    output,
                    effective,
                    intercepted,
                    message_to_agent_on_approval,
                )
                .await
            }
        }
    }

    /// An output-based call that finished: `ask_user` awaits the answer
    /// (its result is the questions alone, and no message follows), and
    /// `complete_task` completes the task (reported with
    /// `TaskStateChanged`, as in native mode); the result is formatted with
    /// the host's message, and the tool's images wait for the next
    /// message.
    async fn finish_text_result(
        &mut self,
        tool_name: &str,
        tool_call_id: Uuid,
        output: crate::tool::ToolOutput,
        effective: String,
        intercepted: bool,
        message_to_agent_on_approval: Option<String>,
    ) -> Result<String, ArkoAgentsError> {
        let raw = output.text;
        // A call of a group another call already ended (possible only
        // for tools configured parallel-safe) changes no state.
        let open = !self.task.current_state.is_terminal();
        if tool_name == ASK_USER_TOOL_NAME && open {
            self.task.set_state(TaskState::AwaitingUserResponse {
                questions: raw.clone(),
            });
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
            self.emit_event(AgentEvent::UserQuestionsAsked {
                questions: raw.clone(),
            })
            .await?;
            self.emit_event(AgentEvent::ToolResult {
                tool_call_id,
                tool_name: tool_name.to_string(),
                output: raw,
            })
            .await?;
            self.queue_tool_images(tool_name, tool_call_id, output.images);
            return Ok(String::new());
        }
        self.emit_tool_result_events(tool_call_id, tool_name, &raw, &effective, intercepted)
            .await?;
        let formatted = self.prompts().format_tool_result_as_user_message(
            tool_name,
            &effective,
            message_to_agent_on_approval.as_deref(),
        );
        self.queue_tool_images(tool_name, tool_call_id, output.images);
        if tool_name == COMPLETE_TASK_TOOL_NAME {
            self.task.set_state(TaskState::Completed { summary: raw });
            // Reported as in native mode, after the call's result.
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        Ok(formatted)
    }
}
