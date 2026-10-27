//! Native tool calling: a turn's calls checked, run in order or held for
//! approval, the approved and held calls run once decided, each call's result
//! recorded, and the turn's results added to the conversation.

use super::*;

/// A tool's images as images of its result, after its text.
fn result_images(images: Vec<ToolImage>) -> Vec<ai::ToolResultImage> {
    images
        .into_iter()
        .map(|image| ai::ToolResultImage::new(image.description, image.image))
        .collect()
}

/// A native result's text with the host's message given with the call's
/// approval.
fn with_approval_message(effective: String, message_to_agent: Option<String>) -> String {
    match message_to_agent {
        Some(message) => {
            let mut text = effective;
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str("Message from user on approval:\n");
            text.push_str(&message);
            text
        }
        None => effective,
    }
}

/// A native call to run, with where its result goes. See
/// [`AgentLoop::run_native_calls`].
pub(super) struct NativeRun {
    /// Its position in the turn (or approval batch).
    pub(super) index: usize,
    pub(super) call: ai::ToolCall,
    pub(super) run: ToolRun,
    /// The host's message with its approval, added to its result.
    pub(super) message_to_agent: Option<String>,
}

/// The results of a turn's (or an approval's) native calls, recorded in the
/// task ([`Task::pending_native_tool_results`]) as they are produced, so
/// that a snapshot taken meanwhile holds them. [`finish`](Self::finish)
/// puts them in call order.
pub(super) struct NativeResults {
    /// Where the recorded results start in the task's pending results.
    pub(super) start: usize,
    /// Each recorded result's position in its turn, in the order recorded.
    pub(super) positions: Vec<usize>,
}

impl NativeResults {
    pub(super) fn new(task: &Task) -> Self {
        Self {
            start: task.pending_native_tool_results.len(),
            positions: Vec::new(),
        }
    }

    pub(super) fn record(&mut self, task: &mut Task, position: usize, result: ai::ToolResult) {
        task.pending_native_tool_results.push(result);
        self.positions.push(position);
    }

    /// The recorded results, with their positions.
    fn recorded<'a>(&'a self, task: &'a Task) -> impl Iterator<Item = (usize, &'a ai::ToolResult)> {
        self.positions
            .iter()
            .copied()
            .zip(task.pending_native_tool_results.iter().skip(self.start))
    }

    /// The positions of the recorded results that are errors (the calls
    /// that were rejected or failed).
    pub(super) fn failed(&self, task: &Task) -> std::collections::BTreeSet<usize> {
        self.recorded(task)
            .filter(|(_, result)| result.is_error)
            .map(|(position, _)| position)
            .collect()
    }

    /// Whether a call before `position` was rejected or failed.
    fn failed_before(&self, task: &Task, position: usize) -> bool {
        self.recorded(task)
            .any(|(recorded, result)| recorded < position && result.is_error)
    }

    /// Puts the recorded results in call order.
    pub(super) fn finish(self, task: &mut Task) {
        let results = &mut task.pending_native_tool_results;
        if results.len() != self.start + self.positions.len() {
            debug_assert!(false, "pending native results changed while recorded");
            return;
        }
        let mut recorded: Vec<_> = self
            .positions
            .into_iter()
            .zip(results.drain(self.start..))
            .collect();
        recorded.sort_by_key(|(position, _)| *position);
        results.extend(recorded.into_iter().map(|(_, result)| result));
    }
}

/// A native turn's calls once checked. See [`AgentLoop::native_turn`].
struct NativePlan {
    /// The results recorded so far, in the task.
    results: NativeResults,
    /// The calls awaiting approval, with their positions in the turn.
    calls_needing_approval: Vec<(usize, PendingToolCallInfo)>,
    /// The calls that run now.
    runs: Vec<NativeRun>,
    /// Auto-approved calls that must wait for a call awaiting approval:
    /// they run once it is decided.
    held: Vec<NativeRun>,
}

/// What a native call's checks decided. See
/// [`AgentLoop::classify_native_call`].
#[allow(
    clippy::large_enum_variant,
    reason = "returned once per call and taken apart at once: a box would only add an allocation"
)]
enum NativeCall {
    /// Answered at once: its result is recorded.
    Answered,
    /// It waits for approval. `exclusive`: it doesn't run beside other
    /// calls.
    NeedsApproval {
        pending: PendingToolCallInfo,
        exclusive: bool,
    },
    /// It runs without approval.
    Run(NativeRun),
}

impl AgentLoop {
    /// Runs native calls in order under the parallel-safety rule (see
    /// [`AgentToolConfig`](crate::agent_config::AgentToolConfig)): each group
    /// of [`tool_run::groups`] runs at the same time, after the one before.
    /// Once the task can no longer run calls (`keep_going` says so of its
    /// state, or it was suspended), the calls not yet run are answered as not
    /// run. `complete_task` after an earlier call of the turn that was
    /// rejected or failed (`failed` holds those calls' positions, and the
    /// calls here that fail join them) doesn't run: its result says so.
    /// Each call's result is recorded in `results` (in the task) as its group
    /// ends, and the task reported to the snapshot sink.
    pub(super) async fn run_native_calls(
        &mut self,
        runs: Vec<NativeRun>,
        keep_going: fn(&TaskState) -> bool,
        mut failed: std::collections::BTreeSet<usize>,
        results: &mut NativeResults,
    ) -> Result<(), ArkoAgentsError> {
        let mut unreached = Vec::new();
        for group in tool_run::groups(runs, |run| run.run.parallel_safe) {
            if !keep_going(&self.task.current_state) || self.stop_requested {
                unreached.extend(group);
                continue;
            }
            let mut to_run = Vec::new();
            for native in group {
                if native.call.name == COMPLETE_TASK_TOOL_NAME
                    && failed.range(..native.index).next().is_some()
                {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: native.run.tool_call_id,
                        tool_name: native.call.name.clone(),
                        reason: Some(COMPLETION_AFTER_FAILURE.to_string()),
                    })
                    .await?;
                    results.record(
                        &mut self.task,
                        native.index,
                        ai::ToolResult::from_tool_call(&native.call, COMPLETION_AFTER_FAILURE),
                    );
                    self.call_settled(native.run.tool_call_id);
                } else {
                    to_run.push(native);
                }
            }
            if to_run.is_empty() {
                continue;
            }
            self.run_native_group(to_run, &mut failed, results).await?;
        }
        if !unreached.is_empty() {
            let not_run = self.not_run_after_stop();
            for native in unreached {
                self.answer_not_run(
                    results,
                    native.index,
                    &native.call,
                    native.run.tool_call_id,
                    &not_run,
                )
                .await?;
                // A recovered call that doesn't run again.
                self.call_settled(native.run.tool_call_id);
            }
        }
        Ok(())
    }

    /// Runs one group of native calls at the same time (see
    /// [`run_group`](Self::run_group)), records each call's result (those
    /// that fail join `failed`), answers the calls a stop ended, and
    /// reports the task.
    async fn run_native_group(
        &mut self,
        to_run: Vec<NativeRun>,
        failed: &mut std::collections::BTreeSet<usize>,
        results: &mut NativeResults,
    ) -> Result<(), ArkoAgentsError> {
        let (calls, group_runs): (Vec<_>, Vec<_>) = to_run
            .into_iter()
            .map(|native| {
                let id = native.run.tool_call_id;
                let mut run = native.run;
                run.call_id = native.call.id.clone();
                run.message_to_agent = native.message_to_agent.clone();
                (
                    (native.index, native.call, id, native.message_to_agent),
                    run,
                )
            })
            .unzip();
        let group = self.run_group(group_runs).await;
        let mut stopped = Vec::new();
        for ((index, call, id, message), outcome) in calls.into_iter().zip(group.outcomes) {
            match outcome {
                Some(outcome) => {
                    let result = self
                        .finish_native_call(&call, id, outcome?, message)
                        .await?;
                    if result.is_error {
                        failed.insert(index);
                    }
                    results.record(&mut self.task, index, result);
                    self.call_settled(id);
                }
                None => stopped.push((index, call, id)),
            }
        }
        if let Some(stop) = group.stop {
            self.stop_after_calls(stop).await?;
            let reason = stopped_call_reason(stop, &self.task.current_state);
            for (index, call, id) in stopped {
                self.answer_not_run(results, index, &call, id, reason)
                    .await?;
                self.call_settled(id);
            }
        }
        self.report_task(false).await;
        Ok(())
    }

    /// The result of a native call that ran, with its events and the state
    /// changes it makes: `ask_user` awaits the answer, `complete_task`
    /// completes the task, a middleware's decision can end it, and a tool
    /// error is handled under the tool-error policy. The tool's images go
    /// inside its result, after its text. `message_to_agent` (given with an
    /// approval) is added to the result.
    async fn finish_native_call(
        &mut self,
        call: &ai::ToolCall,
        tool_call_id: Uuid,
        outcome: RunOutcome,
        message_to_agent: Option<String>,
    ) -> Result<ai::ToolResult, ArkoAgentsError> {
        let tool_name = call.name.as_str();
        match outcome {
            RunOutcome::Rejected(message) => {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id,
                    tool_name: tool_name.to_string(),
                    reason: Some(message.clone()),
                })
                .await?;
                Ok(ai::ToolResult::error_from_tool_call(call, &message))
            }
            RunOutcome::MiddlewareErrorBefore { reason } => {
                self.pause_by_middleware(tool_call_id, tool_name, &reason, reason.clone())
                    .await?;
                Ok(ai::ToolResult::error_from_tool_call(call, &reason))
            }
            RunOutcome::MiddlewareErrorAfter {
                output,
                effective,
                intercepted,
                reason,
                is_error,
            } => {
                let images = result_images(output.images);
                let raw = output.text;
                self.emit_tool_result_events(
                    tool_call_id,
                    tool_name,
                    &raw,
                    &effective,
                    intercepted,
                )
                .await?;
                let text = with_approval_message(effective, message_to_agent);
                self.pause_for_middleware_error(tool_call_id, tool_name, reason)
                    .await?;
                let result = if is_error {
                    ai::ToolResult::error_from_tool_call(call, &text)
                } else {
                    ai::ToolResult::from_tool_call(call, &text)
                };
                Ok(result.with_images(images))
            }
            RunOutcome::Stop { reason, cancel } => {
                self.stop_by_middleware(tool_call_id, tool_name, &reason, cancel)
                    .await?;
                Ok(ai::ToolResult::error_from_tool_call(call, &reason))
            }
            RunOutcome::Pause { message, reason } => {
                self.pause_by_middleware(tool_call_id, tool_name, &message, reason)
                    .await?;
                Ok(ai::ToolResult::error_from_tool_call(call, &message))
            }
            RunOutcome::Failed(error) => {
                let error_text = format!("Tool '{}' execution failed: {}", tool_name, error);
                self.handle_tool_exec_error(tool_name, error, Some(tool_call_id))
                    .await?;
                Ok(ai::ToolResult::error_from_tool_call(call, &error_text))
            }
            RunOutcome::Finished {
                output,
                effective,
                intercepted,
            } => {
                self.finish_native_result(
                    call,
                    tool_call_id,
                    output,
                    effective,
                    intercepted,
                    message_to_agent,
                )
                .await
            }
            // An ordinary result for now, which names the job. The call is
            // settled with it, and its result comes as a note.
            RunOutcome::Moved { text } => {
                self.finish_native_result(
                    call,
                    tool_call_id,
                    crate::tool::ToolOutput::text(text.clone()),
                    text,
                    false,
                    message_to_agent,
                )
                .await
            }
        }
    }

    /// A native call that finished: `ask_user` awaits the answer and
    /// `complete_task` completes the task (unless another call of its
    /// group already ended it), `message_user` sends its message to the
    /// host, the result events are sent, and the result carries the host's
    /// message and the tool's images.
    async fn finish_native_result(
        &mut self,
        call: &ai::ToolCall,
        tool_call_id: Uuid,
        output: crate::tool::ToolOutput,
        effective: String,
        intercepted: bool,
        message_to_agent: Option<String>,
    ) -> Result<ai::ToolResult, ArkoAgentsError> {
        let tool_name = call.name.as_str();
        let images = result_images(output.images);
        let mut raw = output.text;
        let mut effective = effective;
        // A call of a group another call already ended (possible only
        // for tools configured parallel-safe) changes no state.
        let open = !self.task.current_state.is_terminal();
        if open && let Some(result) = self.take_question_call(tool_name, &raw).await? {
            raw = result.clone();
            effective = result;
        }
        if let Some(message) = self.message_to_user(
            tool_name,
            call.arguments
                .get(crate::tool::builtins::message_user::MESSAGE_USER_MESSAGE_PARAM_NAME),
        ) {
            self.emit_event(AgentEvent::MessageToUser {
                message,
                tool_call_id: Some(tool_call_id),
            })
            .await?;
        }
        self.emit_tool_result_events(tool_call_id, tool_name, &raw, &effective, intercepted)
            .await?;
        let text = with_approval_message(effective, message_to_agent);
        if tool_name == COMPLETE_TASK_TOOL_NAME && open {
            self.task.set_state(TaskState::Completed { summary: raw });
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        Ok(ai::ToolResult::from_tool_call(call, &text).with_images(images))
    }

    /// An `ask_user` or `close_questions` call that finished with `raw`:
    /// its questions are numbered and recorded in the task's log, or closed
    /// there, and the host is told (`UserQuestionsAsked`,
    /// `QuestionsClosed`). An `ask_user` that waits sets the task awaiting
    /// the answer. Returns the call's result for the agent, or `None` for
    /// any other tool.
    pub(super) async fn take_question_call(
        &mut self,
        tool_name: &str,
        raw: &str,
    ) -> Result<Option<String>, ArkoAgentsError> {
        if tool_name == ASK_USER_TOOL_NAME {
            let (questions, wait) = crate::tool::builtins::ask_user::read_call(raw);
            let hook = self.handover_hook.clone();
            let last = crate::handover::last_question_number(
                hook.as_deref(),
                &self.task,
                questions.len() as u32,
                0,
            );
            let (asked, wait, result) = crate::questions::take_ask(
                &mut self.task.questions,
                last,
                questions,
                wait,
                self.task.agent_config.tool_config.async_questions,
            );
            if let Some(hook) = &hook {
                hook.questions_asked(&self.task, &asked, wait);
            }
            let questions = crate::questions::questions_text(&asked);
            if wait {
                self.task.set_state(TaskState::AwaitingUserResponse {
                    questions: questions.clone(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
            }
            self.emit_event(AgentEvent::UserQuestionsAsked {
                questions,
                asked,
                wait,
            })
            .await?;
            return Ok(Some(result));
        }
        if tool_name == crate::tool::builtins::CLOSE_QUESTIONS_TOOL_NAME
            && let Some((ids, why)) = crate::tool::builtins::close_questions::read_call(raw)
        {
            let (ids, result) = crate::questions::take_close(&mut self.task.questions, ids);
            if !ids.is_empty() {
                self.emit_event(AgentEvent::QuestionsClosed { ids, why })
                    .await?;
            }
            return Ok(Some(result));
        }
        Ok(None)
    }

    pub(super) fn tool_call_args_to_map(tc: &ai::ToolCall) -> HashMap<String, Value> {
        if let Value::Object(map) = &tc.arguments {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            HashMap::new()
        }
    }

    /// Adds the turn's native results to the conversation as one user
    /// message: the results, then `extra_text` (an answer or a note) and
    /// the environment details and reminder due with this user message, as
    /// text after the results in the same turn.
    pub(super) async fn flush_native_results_to_conversation(
        &mut self,
        extra_text: Option<&str>,
    ) -> Result<(), ArkoAgentsError> {
        if self.task.pending_native_tool_results.is_empty()
            && extra_text.is_none()
            && self.task.pending_tool_images.is_empty()
        {
            return Ok(());
        }
        self.limit_tool_results().await?;
        let text = self.prompts().text_beside_tool_results(
            &self.task,
            extra_text,
            self.task.user_message_count + 1,
        )?;
        self.task.add_pending_tool_results(text.as_deref());
        self.mark_details_sent();
        Ok(())
    }

    /// Whether an approved native turn's results go into the conversation
    /// now: the task no longer waits, or it waits for the answer to
    /// `ask_user`, which follows them (as when `ask_user` is auto-approved),
    /// so the answer can also come from
    /// [`Task::prepare_for_resumption_with_user_response`].
    pub(super) fn native_results_are_due(&self) -> bool {
        !self.task.current_state.is_awaiting_approval()
            || matches!(
                self.task.current_state,
                TaskState::AwaitingUserResponse { .. }
            )
    }

    /// Adds the turn's native results to the conversation if they are due
    /// now. See [`native_results_are_due`](Self::native_results_are_due).
    pub(super) async fn flush_native_results_if_due(&mut self) -> Result<(), ArkoAgentsError> {
        if self.native_results_are_due() {
            self.flush_native_results_to_conversation(None).await?;
        }
        Ok(())
    }

    /// A native turn. Its calls are checked, and in call order those that
    /// need no approval run (or are held behind a call awaiting approval)
    /// and those that need it wait for it. Every call gets one result. A
    /// turn without calls is the handover report, or a response without a
    /// tool call.
    pub(super) async fn native_turn(
        &mut self,
        turn: StreamedTurn,
    ) -> Result<Flow, ArkoAgentsError> {
        let StreamedTurn {
            response: final_generation_response,
            announced: mut native_stream_call_ids,
            ..
        } = turn;
        let tool_calls = final_generation_response
            .as_ref()
            .map(|r| r.message.tool_calls.clone())
            .unwrap_or_default();

        if tool_calls.is_empty() {
            return self
                .native_turn_without_calls(final_generation_response, native_stream_call_ids)
                .await;
        }

        self.reset_llm_recovery();
        let overall_batch_id = (tool_calls.len() > 1).then(Uuid::new_v4);
        self.begin_tool_batch();
        // Every call gets exactly one result, in the order the model
        // made them: checks answer some at once, the auto-approved
        // calls run, and the calls that need approval wait for it.
        let mut plan = NativePlan {
            results: NativeResults::new(&self.task),
            calls_needing_approval: Vec::new(),
            runs: Vec::new(),
            held: Vec::new(),
        };
        // Whether an earlier call of the turn awaits approval or is
        // held, and whether one of those isn't parallel-safe. A call
        // waiting for approval counts as not finished: a
        // parallel-safe call waits for every earlier call that isn't,
        // and any other call for every earlier call.
        let mut waiting_any = false;
        let mut waiting_exclusive = false;
        for (call_index, ai_tool_call) in tool_calls.iter().enumerate() {
            let call = self
                .classify_native_call(
                    call_index,
                    ai_tool_call,
                    &mut native_stream_call_ids,
                    &mut plan.results,
                )
                .await?;
            match call {
                NativeCall::Answered => {}
                NativeCall::NeedsApproval { pending, exclusive } => {
                    waiting_any = true;
                    waiting_exclusive |= exclusive;
                    plan.calls_needing_approval.push((call_index, pending));
                }
                NativeCall::Run(run) => {
                    let must_wait = if run.run.parallel_safe {
                        waiting_exclusive
                    } else {
                        waiting_any
                    };
                    if must_wait {
                        waiting_any = true;
                        waiting_exclusive |= !run.run.parallel_safe;
                        plan.held.push(run);
                    } else {
                        plan.runs.push(run);
                    }
                }
            }
        }
        self.run_planned_calls(&mut plan).await?;
        let NativePlan {
            results,
            calls_needing_approval,
            held,
            ..
        } = plan;
        // Held calls wait with the calls awaiting approval (and are
        // saved with them).
        for native in held {
            self.task
                .pending_native_tool_calls
                .insert(native.run.tool_call_id, native.call);
        }
        results.finish(&mut self.task);
        // Calls announced while streaming that the response doesn't
        // hold (for example cut off at the token limit) are not run.
        self.reject_announced_calls(&mut native_stream_call_ids, UNDELIVERED_CALL)
            .await?;

        if !calls_needing_approval.is_empty() {
            let calls_needing_approval: Vec<PendingToolCallInfo> = calls_needing_approval
                .into_iter()
                .map(|(_, pending)| pending)
                .collect();
            self.await_approvals(calls_needing_approval, overall_batch_id)
                .await?;
        }
        // Pauses on the turn's tool errors now, or after the calls
        // still awaiting approval are decided.
        self.end_tool_batch().await?;
        self.flush_native_results_if_due().await?;
        Ok(Flow::Proceed)
    }

    /// A native turn without calls. It is the handover report when one is
    /// due and the agent isn't offered `handover_report`, the final reply
    /// of an agent whose reply in text ends the task (which completes it),
    /// or else a response without a tool call, which gets a note and is
    /// asked again under the error policy. Calls the stream announced are
    /// rejected.
    async fn native_turn_without_calls(
        &mut self,
        final_generation_response: Option<ai::GenerationResponse>,
        mut native_stream_call_ids: HashMap<usize, (Uuid, String)>,
    ) -> Result<Flow, ArkoAgentsError> {
        self.reject_announced_calls(&mut native_stream_call_ids, UNDELIVERED_CALL)
            .await?;
        let text = final_generation_response
            .as_ref()
            .map(|response| message_text(&response.message))
            .unwrap_or_default();
        if self.handover_report_due() {
            self.take_handover_report(text).await?;
            return Ok(self.end_of_pass());
        }
        if self.task.agent_config.tool_config.ends_task_with_text() && !text.trim().is_empty() {
            self.complete_with_reply(text).await?;
            return Ok(self.end_of_pass());
        }
        if self.task.agent_config.tool_config.text_reply_as_message
            && !self.recovery.text_reply_sent
            && !text.trim().is_empty()
        {
            self.recovery.text_reply_sent = true;
            self.emit_event(AgentEvent::MessageToUser {
                message: text,
                tool_call_id: None,
            })
            .await?;
            self.add_runtime_message(TEXT_REPLY_SENT)?;
            return Ok(self.end_of_pass());
        }
        log::warn!(
            "LLM response for task {} did not contain a native tool call.",
            self.task.id
        );
        let note = if self.task.agent_config.tool_config.ends_task_with_text() {
            NO_NATIVE_TOOL_CALL_OR_REPLY
        } else {
            NO_NATIVE_TOOL_CALL
        };
        self.handle_model_output_failure(note.to_string()).await?;
        Ok(self.end_of_pass())
    }

    /// The final reply of an agent whose reply in text ends the task
    /// ([`AgentToolConfig::text_reply_ends_task`](crate::agent_config::AgentToolConfig::text_reply_ends_task)):
    /// the task completes with the reply as its summary, as a
    /// `complete_task` call with it would complete it.
    async fn complete_with_reply(&mut self, reply: String) -> Result<(), ArkoAgentsError> {
        self.reset_llm_recovery();
        self.task.set_state(TaskState::Completed {
            summary: reply.trim().to_string(),
        });
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await
    }

    /// A native call's checks, in order: arguments that aren't JSON,
    /// `handover_report` (answered by the runtime where it is offered), a
    /// handover report that is due, a tool the run doesn't offer, and its
    /// paths. A call they reject is answered (`ToolRejected` or
    /// `UnavailableToolAttempted`, and its result recorded). Otherwise it
    /// needs approval (it waits in the task's pending calls) or runs.
    async fn classify_native_call(
        &mut self,
        call_index: usize,
        ai_tool_call: &ai::ToolCall,
        native_stream_call_ids: &mut HashMap<usize, (Uuid, String)>,
        results: &mut NativeResults,
    ) -> Result<NativeCall, ArkoAgentsError> {
        let tool_call_id = native_stream_call_ids
            .remove(&call_index)
            .map_or_else(Uuid::new_v4, |(id, _)| id);
        let params_map = Self::tool_call_args_to_map(ai_tool_call);

        self.emit_event(AgentEvent::ToolCallFormatted {
            tool_call_id,
            tool_name: ai_tool_call.name.clone(),
            parameters: params_map.clone(),
        })
        .await?;

        if let Some(raw) = &ai_tool_call.invalid_arguments {
            let reason = format!(
                "The arguments for tool '{}' are not valid JSON, so the tool was not run. Call it again with a JSON object. Received: {}",
                ai_tool_call.name, raw
            );
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: ai_tool_call.name.clone(),
                reason: Some(reason.clone()),
            })
            .await?;
            results.record(
                &mut self.task,
                call_index,
                ai::ToolResult::error_from_tool_call(ai_tool_call, &reason),
            );
            return Ok(NativeCall::Answered);
        }

        if ai_tool_call.name == HANDOVER_REPORT_TOOL_NAME && self.offers_report_tool() {
            let result = match self.take_report_call(&params_map) {
                Ok(received) => {
                    self.emit_tool_result_events(
                        tool_call_id,
                        HANDOVER_REPORT_TOOL_NAME,
                        received,
                        received,
                        false,
                    )
                    .await?;
                    ai::ToolResult::from_tool_call(ai_tool_call, received)
                }
                Err(reason) => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id,
                        tool_name: ai_tool_call.name.clone(),
                        reason: Some(reason.to_string()),
                    })
                    .await?;
                    ai::ToolResult::error_from_tool_call(ai_tool_call, reason)
                }
            };
            results.record(&mut self.task, call_index, result);
            return Ok(NativeCall::Answered);
        }

        if self.handover_blocks(&ai_tool_call.name) {
            let reason = self.handover_report_due_reason();
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: ai_tool_call.name.clone(),
                reason: Some(reason.to_string()),
            })
            .await?;
            results.record(
                &mut self.task,
                call_index,
                ai::ToolResult::error_from_tool_call(ai_tool_call, reason),
            );
            return Ok(NativeCall::Answered);
        }

        let tool_arc = match self.tool_registry.get(&ai_tool_call.name) {
            Some(t) => t,
            None => {
                let result = self.unavailable_native_tool(ai_tool_call).await?;
                results.record(&mut self.task, call_index, result);
                return Ok(NativeCall::Answered);
            }
        };
        let tool_definition = tool_arc.definition();

        // The path parameter that forces an approval.
        let mut outside_workspace = None;
        let path_rejection =
            match self.path_verdict(&ai_tool_call.name, &tool_definition, &params_map) {
                PathVerdict::Allowed => None,
                PathVerdict::NeedsApproval(param_name) => {
                    outside_workspace = Some(param_name);
                    None
                }
                PathVerdict::Outside(param_name) => Some(format!(
                    "Path for parameter '{param_name}' is outside the workspace."
                )),
                PathVerdict::Invalid(reason) => Some(reason),
            };
        // A rejected call gets this one result and is not run.
        if let Some(reason) = path_rejection {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: ai_tool_call.name.clone(),
                reason: Some(reason.clone()),
            })
            .await?;
            results.record(
                &mut self.task,
                call_index,
                ai::ToolResult::error_from_tool_call(ai_tool_call, &reason),
            );
            return Ok(NativeCall::Answered);
        }

        let normal_approval_needed = !self
            .task
            .agent_config
            .should_auto_approve_tool(&ai_tool_call.name);

        if outside_workspace.is_some() || normal_approval_needed {
            let exclusive = !self.runs_in_parallel(&tool_definition);
            self.task
                .pending_native_tool_calls
                .insert(tool_call_id, ai_tool_call.clone());
            return Ok(NativeCall::NeedsApproval {
                pending: PendingToolCallInfo {
                    tool_call_id,
                    tool_name: ai_tool_call.name.clone(),
                    parameters: params_map,
                    tool_description: tool_definition.description.clone(),
                    approval_reason: ApprovalReason {
                        tool_not_auto_approved: normal_approval_needed,
                        path_outside_workspace: outside_workspace,
                    },
                },
                exclusive,
            });
        }
        let run = NativeRun {
            index: call_index,
            call: ai_tool_call.clone(),
            run: self.tool_run(tool_call_id, &ai_tool_call.name, tool_arc, params_map, None),
            message_to_agent: None,
        };
        Ok(NativeCall::Run(run))
    }

    /// Runs a native turn's planned calls. A turn a call stopped (a tool
    /// error that fails the task, `complete_task` or `ask_user`, a
    /// cancellation or a suspension) answers the calls awaiting approval
    /// and the held ones as not run. Otherwise `complete_task` awaiting
    /// approval after a failed call is answered, and with no approval
    /// left to wait for, the held calls run.
    async fn run_planned_calls(&mut self, plan: &mut NativePlan) -> Result<(), ArkoAgentsError> {
        let NativePlan {
            results,
            calls_needing_approval,
            runs,
            held,
        } = plan;
        let failed = results.failed(&self.task);
        self.run_native_calls(
            std::mem::take(runs),
            |state| matches!(state, TaskState::InProgress),
            failed,
            results,
        )
        .await?;
        // A turn a call stopped (a tool error that fails the task,
        // `complete_task` or `ask_user`, a cancellation or a
        // suspension) still answers every call once, including the
        // calls awaiting an approval that won't be asked.
        if !matches!(self.task.current_state, TaskState::InProgress) || self.stop_requested {
            let not_run = self.not_run_after_stop();
            for (index, pending) in calls_needing_approval.drain(..) {
                if let Some(call) = self
                    .task
                    .pending_native_tool_calls
                    .remove(&pending.tool_call_id)
                {
                    // `call` is the call `pending` describes (same name).
                    self.answer_not_run(results, index, &call, pending.tool_call_id, &not_run)
                        .await?;
                }
            }
            for native in held.drain(..) {
                self.answer_not_run(
                    results,
                    native.index,
                    &native.call,
                    native.run.tool_call_id,
                    &not_run,
                )
                .await?;
            }
        }
        if matches!(self.task.current_state, TaskState::InProgress) && !self.stop_requested {
            // complete_task awaiting approval after a call that was
            // rejected or failed is answered now, not asked about.
            let mut still_awaiting = Vec::new();
            for (index, pending) in calls_needing_approval.drain(..) {
                let after_failure = results.failed_before(&self.task, index);
                if pending.tool_name != COMPLETE_TASK_TOOL_NAME || !after_failure {
                    still_awaiting.push((index, pending));
                    continue;
                }
                if let Some(call) = self
                    .task
                    .pending_native_tool_calls
                    .remove(&pending.tool_call_id)
                {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: pending.tool_call_id,
                        tool_name: pending.tool_name,
                        reason: Some(COMPLETION_AFTER_FAILURE.to_string()),
                    })
                    .await?;
                    results.record(
                        &mut self.task,
                        index,
                        ai::ToolResult::from_tool_call(&call, COMPLETION_AFTER_FAILURE),
                    );
                }
            }
            *calls_needing_approval = still_awaiting;
            // With no approval left to wait for, held calls run now.
            if calls_needing_approval.is_empty() && !held.is_empty() {
                let failed = results.failed(&self.task);
                self.run_native_calls(
                    std::mem::take(held),
                    |state| matches!(state, TaskState::InProgress),
                    failed,
                    results,
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Answers a native call that doesn't run (`reason` says why): it is
    /// reported rejected, and its error result recorded at `index`.
    async fn answer_not_run(
        &mut self,
        results: &mut NativeResults,
        index: usize,
        call: &ai::ToolCall,
        tool_call_id: Uuid,
        reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        self.emit_event(AgentEvent::ToolRejected {
            tool_call_id,
            tool_name: call.name.clone(),
            reason: Some(reason.to_string()),
        })
        .await?;
        results.record(
            &mut self.task,
            index,
            ai::ToolResult::error_from_tool_call(call, reason),
        );
        Ok(())
    }

    /// The error result of a native call to a tool the task is not offered
    /// (or that isn't registered), reported with `UnavailableToolAttempted`.
    pub(super) async fn unavailable_native_tool(
        &self,
        ai_call: &ai::ToolCall,
    ) -> Result<ai::ToolResult, ArkoAgentsError> {
        self.emit_event(AgentEvent::UnavailableToolAttempted {
            tool_name: ai_call.name.clone(),
        })
        .await?;
        Ok(ai::ToolResult::error_from_tool_call(
            ai_call,
            &format!("Tool '{}' is not available.", ai_call.name),
        ))
    }

    /// The position of `call` in the turn it belongs to (the conversation's
    /// last assistant message), matched by its provider id, or without one
    /// by its name and arguments. A call not found sorts last.
    pub(super) fn turn_position(&self, call: &ai::ToolCall) -> usize {
        self.task
            .conversation
            .messages
            .iter()
            .rev()
            .find(|message| message.role == ai::Role::Assistant)
            .and_then(|turn| {
                turn.tool_calls.iter().position(|turn_call| match &call.id {
                    Some(id) => turn_call.id.as_ref() == Some(id),
                    // Without an id, the call with the same name and
                    // arguments.
                    None => {
                        turn_call.id.is_none()
                            && turn_call.name == call.name
                            && turn_call.arguments == call.arguments
                    }
                })
            })
            .unwrap_or(usize::MAX)
    }

    /// Once a turn's approvals are decided: runs the approved calls
    /// together with the calls held for them (see
    /// [`AgentToolConfig`](crate::agent_config::AgentToolConfig): auto-approved
    /// calls that aren't parallel-safe wait for an earlier call's approval),
    /// in call order under the parallel-safety rule, and adds every result,
    /// `decided` (rejections) included, in call order.
    pub(super) async fn run_decided_native_calls(
        &mut self,
        mut runs: Vec<NativeRun>,
        decided: Vec<(usize, ai::ToolResult)>,
    ) -> Result<(), ArkoAgentsError> {
        // The turn's calls that were rejected or failed before the
        // approvals were decided.
        let mut failed: std::collections::BTreeSet<usize> = self
            .task
            .pending_native_tool_results
            .iter()
            .filter(|result| result.is_error)
            .filter(|result| result.call_id.is_some())
            .map(|result| {
                self.turn_position(&ai::ToolCall {
                    id: result.call_id.clone(),
                    name: result.name.clone(),
                    arguments: Value::Null,
                    thought_signature: None,
                    provider_id: None,
                    invalid_arguments: None,
                })
            })
            .collect();
        let mut results = NativeResults::new(&self.task);
        for (index, result) in decided {
            results.record(&mut self.task, index, result);
        }
        let held: Vec<(Uuid, ai::ToolCall)> = self.task.pending_native_tool_calls.drain().collect();
        for (tool_call_id, call) in held {
            let index = self.turn_position(&call);
            let params = Self::tool_call_args_to_map(&call);
            match self.tool_registry.get(&call.name) {
                Some(tool) => runs.push(NativeRun {
                    index,
                    run: self.tool_run(tool_call_id, &call.name, tool, params, None),
                    call,
                    message_to_agent: None,
                }),
                None => {
                    let result = self.unavailable_native_tool(&call).await?;
                    results.record(&mut self.task, index, result);
                }
            }
        }
        runs.sort_by_key(|run| run.index);
        // And the rejections now.
        failed.extend(results.failed(&self.task));
        self.run_native_calls(runs, |state| !state.is_terminal(), failed, &mut results)
            .await?;
        results.finish(&mut self.task);
        Ok(())
    }

    /// The end of a wait's native decisions: the approved calls run with
    /// the held ones and every result, `decided` included, is added (see
    /// [`run_decided_native_calls`](Self::run_decided_native_calls)), the
    /// batch ends (pausing on its tool errors), and the turn's results go
    /// into the conversation if they are due.
    pub(super) async fn finish_native_decisions(
        &mut self,
        approved: Vec<NativeRun>,
        decided: Vec<(usize, ai::ToolResult)>,
    ) -> Result<(), ArkoAgentsError> {
        self.run_decided_native_calls(approved, decided).await?;
        // The decision put the task in progress, and a call that ran
        // doesn't wait for approval.
        debug_assert!(!matches!(
            self.task.current_state,
            TaskState::AwaitingToolApproval { .. } | TaskState::AwaitingBatchToolApproval { .. }
        ));
        self.end_tool_batch().await?;
        self.flush_native_results_if_due().await
    }

    /// An approved native call to run, or `Err` when this run doesn't offer
    /// its tool (a call saved awaiting approval), which is answered like any
    /// unavailable tool, even with caller output.
    #[allow(clippy::too_many_arguments, clippy::result_unit_err)]
    pub(super) fn approved_native_run(
        &self,
        index: usize,
        ai_call: &ai::ToolCall,
        internal_id: Uuid,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        custom_output: Option<String>,
        message_to_agent: Option<String>,
    ) -> Result<NativeRun, ()> {
        let tool = self.tool_registry.get(tool_name).ok_or(())?;
        Ok(NativeRun {
            index,
            call: ai_call.clone(),
            run: self.tool_run(internal_id, tool_name, tool, parameters, custom_output),
            message_to_agent,
        })
    }
}
