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

    fn take_tool_image_content(&mut self) -> Vec<ai::Content> {
        self.task.take_tool_image_content()
    }

    pub(super) fn add_text_tool_results_to_conversation(&mut self, text: &str) {
        let mut content = vec![ai::Content::Text(text.to_string())];
        content.extend(self.take_tool_image_content());
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

        let mut next_user_message_str_accumulated = String::new();

        match assemble_tool_calls(raw_events_for_turn) {
            Ok((plain_text_from_assembler, mut tool_call_groups)) => {
                if self.simplified_tool_events && !plain_text_from_assembler.is_empty() {
                    self.text_buffer.push_str(&plain_text_from_assembler);
                    self.emit_throttled_text().await?;
                    self.flush_throttled_text().await?;
                }

                if tool_call_groups.len() > 1 {
                    let error_msg = "Error: Only one tool call or one batch of the same tool can be used per turn.".to_string();
                    if !next_user_message_str_accumulated.is_empty() {
                        next_user_message_str_accumulated.push_str("\n\n");
                    }
                    next_user_message_str_accumulated.push_str(&error_msg);
                } else if tool_call_groups.is_empty() {
                    if self.handover_report_due() {
                        self.take_handover_report(plain_text_from_assembler.clone())
                            .await?;
                        return Ok(self.end_of_pass());
                    }
                    if !self.task.current_state.is_terminal() {
                        self.handle_model_output_failure(NO_TEXT_TOOL_CALL.to_string())
                            .await?;
                        return Ok(self.end_of_pass());
                    }
                } else {
                    let current_group = tool_call_groups.remove(0);
                    self.handle_text_group(
                        current_group,
                        &mut parsed_call_ids,
                        &mut next_user_message_str_accumulated,
                    )
                    .await?;
                }
            }
            Err(e) => {
                log::error!(
                    "Failed to assemble tool calls for task {}: {}",
                    self.task.id,
                    e
                );
                self.handle_model_output_failure(format!(
                    "Your tool call could not be read ({e}). Write it again in the format the system prompt describes."
                ))
                .await?;
                return Ok(self.end_of_pass());
            }
        }

        self.text_turn_message(
            next_user_message_str_accumulated,
            &final_generation_response,
        )?;
        Ok(Flow::Proceed)
    }

    /// The calls of an output-based turn's group: each is checked and run
    /// or held for approval, in order, until one ends the batch; calls a
    /// suspension left are answered as not run, the others wait for
    /// approval, the batch's tool errors pause the task, and the results
    /// become the turn's message.
    async fn handle_text_group(
        &mut self,
        current_group: ToolCallGroup,
        parsed_call_ids: &mut VecDeque<Uuid>,
        next_user_message_str_accumulated: &mut String,
    ) -> Result<(), ArkoAgentsError> {
        let calls_to_process_with_ids = self.calls_of_group(
            current_group,
            parsed_call_ids,
            next_user_message_str_accumulated,
        );

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

        for (parsed_tool_call_ref, tool_call_id_ref) in &calls_to_process_with_ids {
            match self
                .handle_text_call(parsed_tool_call_ref, *tool_call_id_ref)
                .await?
            {
                TextCall::Handled => {}
                TextCall::NeedsApproval(pending) => calls_needing_approval.push(pending),
                TextCall::EndsBatch => break,
            }
        }

        // Suspended while a call ran: the calls
        // awaiting approval are not asked about.
        if self.stop_requested {
            let not_run = self.not_run_after_stop();
            for pending in calls_needing_approval.drain(..) {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: pending.tool_call_id,
                    tool_name: pending.tool_name.clone(),
                    reason: Some(not_run.clone()),
                })
                .await?;
                self.record_text_result(format!("Tool {}: {not_run}", pending.tool_name))
                    .await;
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
        // with the task below if it now waits.
        *next_user_message_str_accumulated = self.take_text_results(start).join("\n\n---\n\n");
        Ok(())
    }

    /// Reports an output-based turn's parser events (unless tool events
    /// are simplified): `ToolCallInitiated` and the parameters of each
    /// call, then the turn's plain text. Returns the ids given to the
    /// calls, in order.
    async fn emit_parser_events(
        &mut self,
        raw_events_for_turn: &[RawParserEvent],
    ) -> Result<VecDeque<Uuid>, ArkoAgentsError> {
        let mut assistant_text_from_raw_events_for_pass1 = String::new();
        let mut parsed_call_ids = VecDeque::new();

        if !self.simplified_tool_events {
            let mut current_tool_call_id_granular: Option<Uuid> = None;
            let mut current_param_name_granular: Option<String> = None;

            for raw_event in raw_events_for_turn {
                match raw_event {
                    RawParserEvent::Text(text) => {
                        assistant_text_from_raw_events_for_pass1.push_str(text);
                    }
                    RawParserEvent::Error(err) => {
                        log::warn!("Raw parser emitted error event: {}", err);
                    }
                    RawParserEvent::ToolCallStart(name) => {
                        let tool_call_id = Uuid::new_v4();
                        parsed_call_ids.push_back(tool_call_id);
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
            if !assistant_text_from_raw_events_for_pass1.is_empty() {
                self.text_buffer
                    .push_str(&assistant_text_from_raw_events_for_pass1);
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
        parsed_call_ids: &mut VecDeque<Uuid>,
        next_user_message_str_accumulated: &mut String,
    ) -> Vec<(ParsedToolCall, Uuid)> {
        let calls_to_process_with_ids: Vec<(ParsedToolCall, Uuid)> = match current_group {
            ToolCallGroup::Single(pc) => {
                vec![(pc, parsed_call_ids.pop_front().unwrap_or_else(Uuid::new_v4))]
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
                            if !next_user_message_str_accumulated.is_empty() {
                                next_user_message_str_accumulated.push_str("\n\n");
                            }
                            next_user_message_str_accumulated.push_str(&err_msg);
                            Vec::new()
                        } else {
                            pcs.into_iter()
                                .map(|pc| {
                                    let id =
                                        parsed_call_ids.pop_front().unwrap_or_else(Uuid::new_v4);
                                    (pc, id)
                                })
                                .collect()
                        }
                    } else {
                        pcs.into_iter()
                            .map(|pc| {
                                let id = parsed_call_ids.pop_front().unwrap_or_else(Uuid::new_v4);
                                (pc, id)
                            })
                            .collect()
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
        parsed_tool_call_ref: &ParsedToolCall,
        tool_call_id: Uuid,
    ) -> Result<TextCall, ArkoAgentsError> {
        if self.simplified_tool_events {
            self.emit_event(AgentEvent::ToolCallInitiated {
                tool_call_id,
                tool_name: parsed_tool_call_ref.name.clone(),
            })
            .await?;
        }
        self.emit_event(AgentEvent::ToolCallFormatted {
            tool_call_id,
            tool_name: parsed_tool_call_ref.name.clone(),
            parameters: parsed_tool_call_ref.parameters.clone(),
        })
        .await?;

        if self.handover_blocks(&parsed_tool_call_ref.name) {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: parsed_tool_call_ref.name.clone(),
                reason: Some(HANDOVER_REPORT_DUE.to_string()),
            })
            .await?;
            self.record_text_result(
                PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
                    .format_tool_execution_error_as_user_message(
                        &parsed_tool_call_ref.name,
                        HANDOVER_REPORT_DUE,
                    ),
            )
            .await;
            return Ok(TextCall::Handled);
        }

        let tool_arc = match self.tool_registry.get(&parsed_tool_call_ref.name) {
            Some(t) => t,
            None => {
                self.emit_event(AgentEvent::UnavailableToolAttempted {
                    tool_name: parsed_tool_call_ref.name.clone(),
                })
                .await?;
                self.record_text_result(format!(
                    "Error: Tool '{}' is not available.",
                    parsed_tool_call_ref.name
                ))
                .await;
                return Ok(TextCall::Handled);
            }
        };
        let tool_definition = tool_arc.definition();

        // The path parameter that forces an approval.
        let mut outside_workspace = None;
        let path_rejection = match self
            .check_path_parameters(&tool_definition, &parsed_tool_call_ref.parameters)
        {
            PathCheck::Inside => None,
            PathCheck::Invalid(reason) => Some(crate::tool::ToolError::ExecutionFailed(reason)),
            PathCheck::Outside(param_name) => match self
                .task
                .agent_config
                .get_path_safety_policy(&parsed_tool_call_ref.name)
            {
                PathSafetyPolicy::AutoReject => Some(crate::tool::ToolError::PathAccessRejected {
                    param_name,
                    workspace_path: self
                        .task
                        .current_working_directory
                        .to_string_lossy()
                        .to_string(),
                }),
                PathSafetyPolicy::RequireApproval => {
                    outside_workspace = Some(param_name);
                    None
                }
                PathSafetyPolicy::AutoApprove => None,
            },
        };
        if let Some(tool_error) = path_rejection {
            let error_message_for_llm = self
                .handle_tool_exec_error(&parsed_tool_call_ref.name, tool_error, Some(tool_call_id))
                .await?;
            self.record_text_result(error_message_for_llm).await;
            // Under `FailTask` the rest of the batch stops.
            if self.task.current_state.is_terminal() {
                return Ok(TextCall::EndsBatch);
            }
            return Ok(TextCall::Handled);
        }

        let normal_approval_needed = !self
            .task
            .agent_config
            .should_auto_approve_tool(&parsed_tool_call_ref.name);

        if outside_workspace.is_some() || normal_approval_needed {
            Ok(TextCall::NeedsApproval(PendingToolCallInfo {
                tool_call_id,
                tool_name: parsed_tool_call_ref.name.clone(),
                parameters: parsed_tool_call_ref.parameters.clone(),
                tool_description: tool_definition.description.clone(),
                approval_reason: ApprovalReason {
                    tool_not_auto_approved: normal_approval_needed,
                    path_outside_workspace: outside_workspace,
                },
            }))
        } else {
            match self
                .execute_tool_and_handle_result(
                    &parsed_tool_call_ref.name,
                    parsed_tool_call_ref.parameters.clone(),
                    tool_call_id,
                    None,
                    None,
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
        mut next_user_message_str_accumulated: String,
        final_generation_response: &Option<ai::GenerationResponse>,
    ) -> Result<(), ArkoAgentsError> {
        if final_generation_response.is_some() && !self.task.current_state.is_awaiting_approval() {
            self.reset_llm_recovery();
        }

        // Results of calls that ran in a turn that now waits (for
        // other calls' approval, or on a pause) wait with it.
        if !next_user_message_str_accumulated.is_empty()
            && self.task.current_state.is_awaiting_approval()
        {
            self.task
                .pending_text_tool_results
                .push(std::mem::take(&mut next_user_message_str_accumulated));
        }
        if !next_user_message_str_accumulated.is_empty()
            && !self.task.current_state.is_terminal()
            && !self.task.current_state.is_awaiting_approval()
        {
            let pb =
                PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
            let next_msg_num = self.task.user_message_count + 1;
            let augmented_user_message = pb.augment_user_message_with_env_details_and_reminder(
                &self.task,
                &next_user_message_str_accumulated,
                next_msg_num,
            )?;
            self.add_text_tool_results_to_conversation(&augmented_user_message);
            self.mark_details_sent();
            self.task.increment_user_message_count();
        } else if next_user_message_str_accumulated.is_empty()
            && !self.task.current_state.is_terminal()
            && !self.task.current_state.is_awaiting_approval()
            && final_generation_response.is_some()
        {
            self.add_runtime_message(
                "Please continue with the task or use the 'complete_task' tool if finished. Ensure you use a tool.",
            )?;
        }
        Ok(())
    }

    /// Runs an output-based call (auto-approved or approved) and returns
    /// its result as text for the next user message. A cancellation or
    /// suspension while it runs stops it.
    pub(super) async fn execute_tool_and_handle_result(
        &mut self,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        tool_call_id: Uuid,
        custom_tool_output: Option<String>,
        message_to_agent_on_approval: Option<String>,
    ) -> Result<String, ArkoAgentsError> {
        self.execute_text_call(
            tool_name,
            parameters,
            tool_call_id,
            custom_tool_output,
            message_to_agent_on_approval,
            false,
        )
        .await
    }

    /// [`execute_tool_and_handle_result`](Self::execute_tool_and_handle_result);
    /// `recovering`: the call runs again after the task stopped while it ran
    /// (see [`TaskContext::recovering`](crate::tool::TaskContext::recovering)).
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
            PromptBuilder::new(&this.task.agent_config, vec![], &this.parsing_config, None)
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
    /// [`execute_tool_and_handle_result`](Self::execute_tool_and_handle_result).
    async fn finish_text_call(
        &mut self,
        tool_name: &str,
        tool_call_id: Uuid,
        outcome: RunOutcome,
        message_to_agent_on_approval: Option<String>,
    ) -> Result<String, ArkoAgentsError> {
        let format_error = |this: &Self, message: &str| {
            PromptBuilder::new(&this.task.agent_config, vec![], &this.parsing_config, None)
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
                    PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
                        .format_tool_result_as_user_message(
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
    /// `complete_task` completes the task; the result is formatted with
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
        let formatted =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
                .format_tool_result_as_user_message(
                    tool_name,
                    &effective,
                    message_to_agent_on_approval.as_deref(),
                );
        self.queue_tool_images(tool_name, tool_call_id, output.images);
        if tool_name == COMPLETE_TASK_TOOL_NAME {
            self.task.set_state(TaskState::Completed { summary: raw });
        }
        Ok(formatted)
    }
}
