//! The decisions that end a wait: approvals and rejections of calls (one, or
//! a batch), answers to the agent's questions, and decisions on an
//! error-recovery pause.

use super::*;

/// What a decision on a wait leaves for after it; see
/// [`AgentLoop::after_decision`].
#[derive(Default)]
pub(super) struct FollowUp {
    /// The message the decision adds (output-based results, an answer, a
    /// note), empty for none.
    pub(super) next_user_message_from_command: String,
    /// That message is the runtime's.
    pub(super) next_message_is_runtime: bool,
    /// The decision needs no follow-up message (it added what it had to).
    pub(super) native_handled: bool,
}

/// How a command applied to the state a task waits in.
pub(super) enum Decision {
    /// It doesn't fit the wait: the pass ends, and the wait goes on.
    Ignored,
    /// It decided the wait.
    Decided(FollowUp),
}

impl AgentLoop {
    /// `ApproveToolUse` or `RejectToolUse` for the call awaiting approval:
    /// an approved call runs (with the calls held for it, in native
    /// mode), a rejected one is answered. A decision for another call is
    /// ignored (a rejection reports the state again), as is any other
    /// command but `CancelTask` and `SuspendTask`.
    pub(super) async fn decide_tool_approval(
        &mut self,
        cmd: &UserCommand,
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
    ) -> Result<Decision, ArkoAgentsError> {
        let follow_up = match *cmd {
            UserCommand::ApproveToolUse {
                tool_call_id: cmd_tool_id,
                ref custom_tool_output,
                ref message_to_agent,
            } => {
                if cmd_tool_id != tool_call_id {
                    log::warn!(
                        "Task {}: Received ApproveToolUse for unexpected tool_call_id. Ignoring.",
                        self.task.id
                    );
                    return Ok(Decision::Ignored);
                }
                self.approve_tool_use(
                    tool_call_id,
                    tool_name,
                    parameters,
                    custom_tool_output,
                    message_to_agent,
                )
                .await?
            }
            UserCommand::RejectToolUse {
                tool_call_id: cmd_tool_id,
                ref reason,
            } => {
                if cmd_tool_id != tool_call_id {
                    log::warn!(
                        "Task {}: Received RejectToolUse for unexpected tool_call_id. Ignoring.",
                        self.task.id
                    );
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                    return Ok(Decision::Ignored);
                }
                self.reject_tool_use(tool_call_id, tool_name, reason)
                    .await?
            }
            UserCommand::CancelTask => FollowUp::default(),
            UserCommand::SuspendTask => FollowUp::default(),
            _ => {
                log::warn!(
                    "Task {}: Received inappropriate command while AwaitingToolApproval. Ignoring.",
                    self.task.id
                );
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                return Ok(Decision::Ignored);
            }
        };
        Ok(Decision::Decided(follow_up))
    }

    /// The approved call runs: natively with the calls held for it, then
    /// the turn's results go into the conversation once due;
    /// output-based, its result is the next message.
    async fn approve_tool_use(
        &mut self,
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
        custom_tool_output: &Option<String>,
        message_to_agent: &Option<String>,
    ) -> Result<FollowUp, ArkoAgentsError> {
        let mut next_user_message_from_command = String::new();
        let mut native_handled = false;
        self.begin_tool_batch();
        // Decided: a snapshot taken while the
        // call runs no longer waits for it.
        self.task.set_state(TaskState::InProgress);
        if self.is_native_mode() {
            let mut approved = Vec::new();
            let mut results = Vec::new();
            if let Some(ai_call) = self.task.pending_native_tool_calls.remove(&tool_call_id) {
                let index = self.turn_position(&ai_call);
                match self.approved_native_run(
                    index,
                    &ai_call,
                    tool_call_id,
                    &tool_name,
                    parameters,
                    custom_tool_output.clone(),
                    message_to_agent.clone(),
                ) {
                    Ok(run) => approved.push(run),
                    Err(()) => results.push((index, self.unavailable_native_tool(&ai_call).await?)),
                }
            }
            self.run_decided_native_calls(approved, results).await?;
            if matches!(
                self.task.current_state,
                TaskState::AwaitingToolApproval { .. }
            ) {
                self.task.set_state(TaskState::InProgress);
            }
            self.end_tool_batch().await?;
            if self.native_results_are_due() {
                self.flush_native_results_to_conversation(None).await?;
            }
            native_handled = true;
        } else {
            let start = self.task.pending_text_tool_results.len();
            match self
                .execute_tool_and_handle_result(
                    &tool_name,
                    parameters,
                    tool_call_id,
                    custom_tool_output.clone(),
                    message_to_agent.clone(),
                )
                .await
            {
                Ok(msg_part) => self.record_text_result(msg_part).await,
                Err(e) => {
                    log::error!(
                        "Task {}: Error during approved tool execution: {}",
                        self.task.id,
                        e
                    );
                }
            }
            next_user_message_from_command = self.take_text_results(start).join("\n\n---\n\n");
            if matches!(self.task.current_state, TaskState::AwaitingToolApproval { tool_call_id: current_id, .. } if current_id == tool_call_id)
            {
                self.task.set_state(TaskState::InProgress);
            }
            self.end_tool_batch().await?;
        }
        Ok(FollowUp {
            next_user_message_from_command,
            native_handled,
            ..Default::default()
        })
    }

    /// The rejected call is answered with the reason: natively beside the
    /// held calls, which run now; output-based, as the next message.
    async fn reject_tool_use(
        &mut self,
        tool_call_id: Uuid,
        tool_name: String,
        reason: &Option<String>,
    ) -> Result<FollowUp, ArkoAgentsError> {
        let mut next_user_message_from_command = String::new();
        let mut native_handled = false;
        self.emit_event(AgentEvent::ToolRejected {
            tool_call_id,
            tool_name: tool_name.clone(),
            reason: (*reason).clone(),
        })
        .await?;

        if self.is_native_mode() {
            let mut results = Vec::new();
            if let Some(ai_call) = self.task.pending_native_tool_calls.remove(&tool_call_id) {
                let reason_str = reason.as_deref().unwrap_or("Rejected by user");
                results.push((
                    self.turn_position(&ai_call),
                    ai::ToolResult::error_from_tool_call(&ai_call, reason_str),
                ));
            }
            // Calls held for this one run now.
            self.begin_tool_batch();
            self.task.set_state(TaskState::InProgress);
            self.run_decided_native_calls(Vec::new(), results).await?;
            if matches!(
                self.task.current_state,
                TaskState::AwaitingToolApproval { .. }
            ) {
                self.task.set_state(TaskState::InProgress);
            }
            self.end_tool_batch().await?;
            if self.native_results_are_due() {
                self.flush_native_results_to_conversation(None).await?;
            }
            native_handled = true;
        } else {
            let pb =
                PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
            next_user_message_from_command =
                pb.format_tool_rejection_as_user_message(&tool_name, (*reason).as_deref());
            self.task.set_state(TaskState::InProgress);
        }
        Ok(FollowUp {
            next_user_message_from_command,
            native_handled,
            ..Default::default()
        })
    }

    /// `SubmitBatchToolApprovals` for the batch awaiting approval: the
    /// rejections and the calls without a decision are answered, and the
    /// approved calls run (natively with the held ones, in call order;
    /// output-based one at a time until one changes the state). A
    /// decision for another batch is ignored, and any other command but
    /// `CancelTask` and `SuspendTask` reports the state again.
    pub(super) async fn decide_batch_approval(
        &mut self,
        cmd: &UserCommand,
        batch_id: Uuid,
        pending_calls: &[PendingToolCallInfo],
    ) -> Result<Decision, ArkoAgentsError> {
        let mut follow_up = FollowUp::default();
        if let UserCommand::SubmitBatchToolApprovals {
            batch_id: cmd_batch_id,
            ref decisions,
        } = *cmd
        {
            if cmd_batch_id != batch_id {
                log::warn!(
                    "Task {}: Received SubmitBatchToolApprovals for unexpected batch_id. Ignoring.",
                    self.task.id
                );
                return Ok(Decision::Ignored);
            }

            self.begin_tool_batch();
            // Decided: a snapshot taken while the
            // calls run no longer waits for them.
            self.task.set_state(TaskState::InProgress);
            follow_up = if self.is_native_mode() {
                self.decide_native_batch(pending_calls, decisions).await?
            } else {
                self.decide_text_batch(pending_calls, decisions).await?
            };
        } else if matches!(*cmd, UserCommand::CancelTask | UserCommand::SuspendTask) {
        } else {
            log::warn!(
                "Task {}: Received inappropriate command while AwaitingBatchToolApproval. Ignoring.",
                self.task.id
            );
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
            return Ok(Decision::Ignored);
        }
        Ok(Decision::Decided(follow_up))
    }

    /// A native batch decided: rejections are answered at once, then the
    /// approved calls run like a turn's (parallel-safe ones at the same
    /// time) with the held ones, in the turn's order.
    async fn decide_native_batch(
        &mut self,
        pending_calls: &[PendingToolCallInfo],
        decisions: &HashMap<Uuid, ToolApprovalDecision>,
    ) -> Result<FollowUp, ArkoAgentsError> {
        // Rejections are answered at once;
        // the approved calls then run like a
        // turn's (parallel-safe ones at the
        // same time), in the batch's order.
        let mut results = Vec::new();
        let mut approved = Vec::new();
        for pending_call_info in pending_calls {
            let Some(ai_call) = self
                .task
                .pending_native_tool_calls
                .remove(&pending_call_info.tool_call_id)
            else {
                continue;
            };
            let index = self.turn_position(&ai_call);
            match decisions.get(&pending_call_info.tool_call_id) {
                Some(ToolApprovalDecision::Approve {
                    custom_tool_output,
                    message_to_agent,
                }) => match self.approved_native_run(
                    index,
                    &ai_call,
                    pending_call_info.tool_call_id,
                    &pending_call_info.tool_name,
                    pending_call_info.parameters.clone(),
                    custom_tool_output.clone(),
                    message_to_agent.clone(),
                ) {
                    Ok(run) => approved.push(run),
                    Err(()) => results.push((index, self.unavailable_native_tool(&ai_call).await?)),
                },
                Some(ToolApprovalDecision::Reject { reason }) => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: pending_call_info.tool_call_id,
                        tool_name: pending_call_info.tool_name.clone(),
                        reason: (*reason).clone(),
                    })
                    .await?;
                    let reason_str = reason.as_deref().unwrap_or("Rejected by user");
                    results.push((
                        index,
                        ai::ToolResult::error_from_tool_call(&ai_call, reason_str),
                    ));
                }
                None => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: pending_call_info.tool_call_id,
                        tool_name: pending_call_info.tool_name.clone(),
                        reason: Some("No decision provided".to_string()),
                    })
                    .await?;
                    results.push((
                        index,
                        ai::ToolResult::error_from_tool_call(
                            &ai_call,
                            "No decision provided, implicitly rejected",
                        ),
                    ));
                }
            }
        }
        self.run_decided_native_calls(approved, results).await?;
        if matches!(
            self.task.current_state,
            TaskState::AwaitingBatchToolApproval { .. }
        ) {
            self.task.set_state(TaskState::InProgress);
        }
        self.end_tool_batch().await?;
        if self.native_results_are_due() {
            self.flush_native_results_to_conversation(None).await?;
        }
        let native_handled = true;
        Ok(FollowUp {
            native_handled,
            ..Default::default()
        })
    }

    /// An output-based batch decided: in order, each call runs or is
    /// answered as rejected, until one changes the state; the results are
    /// the next message.
    async fn decide_text_batch(
        &mut self,
        pending_calls: &[PendingToolCallInfo],
        decisions: &HashMap<Uuid, ToolApprovalDecision>,
    ) -> Result<FollowUp, ArkoAgentsError> {
        let start = self.task.pending_text_tool_results.len();
        for pending_call_info in pending_calls {
            match decisions.get(&pending_call_info.tool_call_id) {
                Some(ToolApprovalDecision::Approve {
                    custom_tool_output,
                    message_to_agent,
                }) => {
                    match self
                        .execute_tool_and_handle_result(
                            &pending_call_info.tool_name,
                            pending_call_info.parameters.clone(),
                            pending_call_info.tool_call_id,
                            custom_tool_output.clone(),
                            message_to_agent.clone(),
                        )
                        .await
                    {
                        Ok(msg_part) => self.record_text_result(msg_part).await,
                        Err(e) => log::error!(
                            "Task {}: Error during approved batched tool execution: {}",
                            self.task.id,
                            e
                        ),
                    }
                }
                Some(ToolApprovalDecision::Reject { reason }) => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: pending_call_info.tool_call_id,
                        tool_name: pending_call_info.tool_name.clone(),
                        reason: (*reason).clone(),
                    })
                    .await?;
                    let pb = PromptBuilder::new(
                        &self.task.agent_config,
                        vec![],
                        &self.parsing_config,
                        None,
                    );
                    self.record_text_result(pb.format_tool_rejection_as_user_message(
                        &pending_call_info.tool_name,
                        (*reason).as_deref(),
                    ))
                    .await;
                }
                None => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: pending_call_info.tool_call_id,
                        tool_name: pending_call_info.tool_name.clone(),
                        reason: Some("No decision provided".to_string()),
                    })
                    .await?;
                    let pb = PromptBuilder::new(
                        &self.task.agent_config,
                        vec![],
                        &self.parsing_config,
                        None,
                    );
                    self.record_text_result(pb.format_tool_rejection_as_user_message(
                        &pending_call_info.tool_name,
                        Some("Implicitly rejected: No decision provided."),
                    ))
                    .await;
                }
            }
            // A call that changed the state (asked
            // the user, completed or ended the
            // task) ends the batch.
            if !matches!(self.task.current_state, TaskState::InProgress) || self.stop_requested {
                break;
            }
        }
        let next_user_message_from_command = self.take_text_results(start).join("\n\n---\n\n");
        self.end_tool_batch().await?;
        Ok(FollowUp {
            next_user_message_from_command,
            ..Default::default()
        })
    }

    /// `SubmitUserResponse` to the questions the task asked: the answer
    /// goes beside the turn's native results, or is the next
    /// output-based message.
    pub(super) async fn answer_question(
        &mut self,
        cmd: &UserCommand,
        questions: &str,
    ) -> Result<Decision, ArkoAgentsError> {
        let mut next_user_message_from_command = String::new();
        let mut native_handled = false;
        match *cmd {
            UserCommand::SubmitUserResponse { ref response } => {
                log::info!("Task {}: User response received.", self.task.id);
                let pb =
                    PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
                let formatted = pb.format_user_response_as_user_message(questions, response);
                self.task.set_state(TaskState::InProgress);

                if self.is_native_mode() {
                    self.flush_native_results_to_conversation(Some(&formatted))
                        .await?;
                    native_handled = true;
                } else {
                    next_user_message_from_command = formatted;
                }
            }
            UserCommand::CancelTask => {}
            UserCommand::SuspendTask => {}
            _ => {
                log::warn!(
                    "Task {}: Received inappropriate command while AwaitingUserResponse. Ignoring.",
                    self.task.id
                );
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                return Ok(Decision::Ignored);
            }
        }
        Ok(Decision::Decided(FollowUp {
            next_user_message_from_command,
            native_handled,
            ..Default::default()
        }))
    }

    /// A decision on an error-recovery pause: `ContinueAfterError` (with an
    /// optional note), `ApproveErrorRecovery` (with an error note unless a
    /// monitor paused), `ResolveRefusal` (which fails, and leaves the wait
    /// on, when the changes don't apply) or `RejectErrorRecovery`.
    pub(super) async fn decide_error_recovery(
        &mut self,
        cmd: &UserCommand,
        error_message: &str,
        cause: &ErrorRecoveryCause,
        error_kind: Option<ai::ErrorKind>,
    ) -> Result<Decision, ArkoAgentsError> {
        let follow_up = match *cmd {
            UserCommand::ContinueAfterError { ref note } => self.continue_after_error(note).await?,
            UserCommand::ApproveErrorRecovery {
                message: ref user_msg_on_recovery,
            } => {
                self.approve_error_recovery(user_msg_on_recovery, error_message)
                    .await?
            }
            UserCommand::ResolveRefusal { ref changes } => {
                match self.task.resolve_refusal(changes.clone()) {
                    Ok(()) => {
                        log::info!("Task {}: Refused input changed; continuing.", self.task.id);
                        self.task.monitor_pause = false;
                        self.task.fallback_for = None;
                        self.task.set_state(TaskState::InProgress);
                        self.reset_llm_recovery();
                        // Tools observed the old input:
                        // they see the changed history
                        // from the start again.
                        self.tool_context_cursor = usize::MAX;
                        // The changed input is sent as it is.
                        FollowUp {
                            native_handled: true,
                            ..Default::default()
                        }
                    }
                    Err(error) => {
                        self.emit_event(AgentEvent::ErrorEncountered {
                            error_message: format!("Could not resolve the refusal: {error}"),
                            requires_approval_to_retry: true,
                            tool_call_id: None,
                            error_kind: None,
                        })
                        .await?;
                        return Ok(Decision::Ignored);
                    }
                }
            }
            UserCommand::RejectErrorRecovery { ref message } => {
                log::info!("Task {}: Error recovery rejected by user.", self.task.id);
                // The failed task still records the
                // results its last turn holds.
                self.settle_interrupted_wait(UNREACHED_CALL, UNREACHED_CALL)
                    .await?;
                // The host's reason, when it gives one.
                let reason = match message.as_deref().filter(|message| !message.is_empty()) {
                    Some(message) => message.to_string(),
                    None => format!(
                        "Error recovery rejected by user for error: {}",
                        error_message
                    ),
                };
                // The pause's cause stays with the
                // failure.
                self.task.set_state(TaskState::Failed {
                    reason,
                    cause: Some(cause.clone()),
                    error_kind,
                });
                FollowUp::default()
            }
            UserCommand::CancelTask => FollowUp::default(),
            UserCommand::SuspendTask => FollowUp::default(),
            _ => {
                log::warn!(
                    "Task {}: Received inappropriate command while AwaitingErrorRecoveryApproval. Ignoring.",
                    self.task.id
                );
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                return Ok(Decision::Ignored);
            }
        };
        Ok(Decision::Decided(follow_up))
    }

    /// `ContinueAfterError`: a pending refusal fallback's request is sent
    /// again (a note waits as context), waiting native results go out with
    /// the note beside them, or the note is the next message; without one
    /// the failed request is sent again, or the waiting output-based
    /// results are the next message.
    async fn continue_after_error(
        &mut self,
        note: &Option<String>,
    ) -> Result<FollowUp, ArkoAgentsError> {
        let mut next_user_message_from_command = String::new();
        let mut next_message_is_runtime = false;
        let mut native_handled = false;
        log::info!(
            "Task {}: Continuing after an error-recovery pause.",
            self.task.id
        );
        self.task.monitor_pause = false;
        let note = note.as_deref().filter(|text| !text.is_empty());
        if self.is_native_mode() {
            // Calls an older version left
            // without results get them now.
            self.answer_open_native_calls(UNREACHED_CALL, UNREACHED_CALL)
                .await?;
        }
        if self.fallback_resend() {
            // The fallback answers exactly the
            // refused input: a note waits, as
            // context, until it has answered.
            if let Some(note) = note {
                self.queue_context_injection(note.to_string());
            }
            native_handled = true;
        } else if self.is_native_mode() && !self.task.pending_native_tool_results.is_empty() {
            self.flush_native_results_to_conversation(note).await?;
            if note.is_some() {
                // The note beside the results is the runtime's.
                self.task.mark_last_message_runtime_text();
            }
            native_handled = true;
        } else if let Some(note) = note {
            next_user_message_from_command = note.to_string();
            next_message_is_runtime = true;
        } else if self.is_native_mode() || self.task.pending_text_tool_results.is_empty() {
            // The failed request is sent again
            // as it was.
            native_handled = true;
        }
        // Otherwise the output-based results the
        // pause left waiting are the next message.
        self.task.refused_input = None;
        self.task.set_state(TaskState::InProgress);
        self.reset_llm_recovery();
        Ok(FollowUp {
            next_user_message_from_command,
            next_message_is_runtime,
            native_handled,
        })
    }

    /// `ApproveErrorRecovery`: like continuing, but a model error is noted
    /// for the model (with the host's message); a monitor's pause adds only
    /// the host's message.
    async fn approve_error_recovery(
        &mut self,
        user_msg_on_recovery: &Option<String>,
        error_message: &str,
    ) -> Result<FollowUp, ArkoAgentsError> {
        let mut next_user_message_from_command = String::new();
        let mut next_message_is_runtime = false;
        let mut native_handled = false;
        log::info!("Task {}: Error recovery approved by user.", self.task.id);
        let monitor_pause = std::mem::take(&mut self.task.monitor_pause);
        if self.fallback_resend() {
            // The fallback answers exactly the
            // refused input: no error note, and
            // the host's message waits, as
            // context, until it has answered.
            if self.is_native_mode() {
                // Calls an older version left
                // without results get them now.
                self.answer_open_native_calls(UNREACHED_CALL, UNREACHED_CALL)
                    .await?;
            }
            if let Some(text) = user_msg_on_recovery
                .as_deref()
                .filter(|text| !text.is_empty())
            {
                self.queue_context_injection(text.to_string());
            }
            native_handled = true;
        } else if monitor_pause {
            // A monitor paused the task, not a
            // model error: nothing to report to
            // the model beyond the host's note.
            match user_msg_on_recovery
                .as_deref()
                .filter(|text| !text.is_empty())
            {
                Some(text) => next_user_message_from_command = text.to_string(),
                None => native_handled = true,
            }
        } else {
            let mut recovery_prompt = format!(
                "The previous attempt resulted in an error: {}. The user has approved to continue.",
                error_message
            );
            if let Some(user_input) = user_msg_on_recovery
                && !(*user_input).is_empty()
            {
                recovery_prompt.push_str("\nUser provided additional context for recovery:\n");
                recovery_prompt.push_str(user_input);
            }

            if self.is_native_mode() {
                // Calls an older version left
                // without results get them now.
                self.answer_open_native_calls(UNREACHED_CALL, UNREACHED_CALL)
                    .await?;
            }
            if self.is_native_mode() && !self.task.pending_native_tool_results.is_empty() {
                self.flush_native_results_to_conversation(Some(&recovery_prompt))
                    .await?;
                // The note beside the results is the runtime's.
                self.task.mark_last_message_runtime_text();
                native_handled = true;
            } else {
                next_user_message_from_command = recovery_prompt;
                next_message_is_runtime = true;
            }
        }
        // Continuing without a change leaves no refused
        // input to resolve.
        self.task.refused_input = None;
        self.task.set_state(TaskState::InProgress);
        self.reset_llm_recovery();
        Ok(FollowUp {
            next_user_message_from_command,
            next_message_is_runtime,
            native_handled,
        })
    }
}
