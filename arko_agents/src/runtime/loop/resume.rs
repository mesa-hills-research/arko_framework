//! A task that runs again: the calls it was running when it stopped, settled
//! by their tools' recovery modes, the turn it left open, and the calls a wait
//! left undecided.

use super::*;

impl AgentLoop {
    /// Answers the native calls of the last turn that have no result yet,
    /// alongside its other waiting results: a call still awaiting approval
    /// is rejected with `reason` (and `ToolRejected`), and a call with no
    /// result at all (a turn an older version left early) is answered as
    /// unreached.
    ///
    /// A call only held behind a call awaiting approval (not listed by the
    /// awaiting state) gets `held_reason` instead.
    pub(super) async fn answer_open_native_calls(
        &mut self,
        reason: &str,
        held_reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        let awaiting: Vec<Uuid> = match &self.task.current_state {
            TaskState::AwaitingToolApproval { tool_call_id, .. } => vec![*tool_call_id],
            TaskState::AwaitingBatchToolApproval { pending_calls, .. } => pending_calls
                .iter()
                .map(|pending| pending.tool_call_id)
                .collect(),
            _ => Vec::new(),
        };
        let approval_wait = matches!(
            self.task.current_state,
            TaskState::AwaitingToolApproval { .. } | TaskState::AwaitingBatchToolApproval { .. }
        );
        // The turn's results are not in the conversation yet while the
        // assistant turn is its last message.
        let turn_calls = match self.task.conversation.messages.last() {
            Some(message) if message.role == ai::Role::Assistant => message.tool_calls.clone(),
            _ => Vec::new(),
        };
        let mut undecided: Vec<_> = self.task.pending_native_tool_calls.drain().collect();
        // In the order the turn made them.
        undecided.sort_by_key(|(_, call)| {
            turn_calls
                .iter()
                .position(|turn_call| turn_call.id == call.id)
                .unwrap_or(usize::MAX)
        });
        let answered = |task: &Task, call: &ai::ToolCall| {
            task.pending_native_tool_results
                .iter()
                .any(|result| result.call_id == call.id)
        };
        for (tool_call_id, call) in undecided {
            if call.id.is_some() && answered(&self.task, &call) {
                continue;
            }
            let reason = if approval_wait && !awaiting.contains(&tool_call_id) {
                held_reason
            } else {
                reason
            };
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: call.name.clone(),
                reason: Some(reason.to_string()),
            })
            .await?;
            self.task
                .pending_native_tool_results
                .push(ai::ToolResult::error_from_tool_call(&call, reason));
        }
        for call in &turn_calls {
            if call.id.is_some() && !answered(&self.task, call) {
                self.task
                    .pending_native_tool_results
                    .push(ai::ToolResult::error_from_tool_call(call, UNREACHED_CALL));
            }
        }
        Ok(())
    }

    /// When a wait for a command ends without one (the command timeout or a
    /// closed command channel), answers the calls still awaiting approval
    /// with `reason` (and `ToolRejected`) and adds the turn's results to the
    /// conversation. The task returned `Interrupted` then pairs every native
    /// call of its last turn with a result, as providers require, the model
    /// learns why the calls did not run, and it can run again. A pause
    /// waiting for those decisions is dropped.
    ///
    /// Calls held behind the calls awaiting approval get `held_reason`.
    pub(super) async fn settle_interrupted_wait(
        &mut self,
        reason: &str,
        held_reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        self.task.deferred_tool_errors.clear();
        if self.is_native_mode() {
            self.answer_open_native_calls(reason, held_reason).await?;
            self.flush_native_results_to_conversation(None).await?;
            return Ok(());
        }
        let undecided = match &self.task.current_state {
            TaskState::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                ..
            } => vec![(*tool_call_id, tool_name.clone())],
            TaskState::AwaitingBatchToolApproval { pending_calls, .. } => pending_calls
                .iter()
                .map(|call| (call.tool_call_id, call.tool_name.clone()))
                .collect(),
            _ => Vec::new(),
        };
        let mut notes = Vec::new();
        for (tool_call_id, tool_name) in undecided {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: tool_name.clone(),
                reason: Some(reason.to_string()),
            })
            .await?;
            notes.push(format!("Tool {tool_name}: {reason}"));
        }
        let message = self
            .task
            .with_pending_text_tool_results(notes.join("\n\n---\n\n"));
        if !message.is_empty() {
            self.add_text_tool_results_to_conversation(&message);
            self.task.increment_user_message_count();
        }
        Ok(())
    }

    /// The tool and recovery mode of a call that was running when its task
    /// stopped: [`ToolRecovery::InDoubt`] for a tool this run doesn't offer,
    /// and [`ToolRecovery::Rerun`] for a call the host answered with its own
    /// output (only that output is given again).
    fn recovery_of(&self, call: &RunningToolCall) -> (Option<Arc<dyn Tool>>, ToolRecovery) {
        let Some(tool) = self.tool_registry.get(&call.tool_name) else {
            return (None, ToolRecovery::InDoubt);
        };
        let mode = if call.custom_output.is_some() {
            ToolRecovery::Rerun
        } else {
            tool.recovery_mode()
        };
        (Some(tool), mode)
    }

    /// Settles the calls a task was running when it stopped (saved at a
    /// safe point by a host that stopped; see [`Task::running_tool_calls`]),
    /// each by its tool's [`recovery_mode`](Tool::recovery_mode):
    /// [`InDoubt`](ToolRecovery::InDoubt) calls are answered with
    /// [`RESTARTED_CALL`], [`Rerun`](ToolRecovery::Rerun) and
    /// [`Resume`](ToolRecovery::Resume) calls run again (with
    /// [`TaskContext::recovering`](crate::tool::TaskContext::recovering)
    /// set), in call order under the parallel-safety rule. Their tool errors
    /// are handled as a batch's, by [`settle_open_turn`](Self::settle_open_turn)
    /// (or with the interrupted wait). Returns whether any call was running.
    pub(super) async fn recover_running_calls(&mut self) -> Result<bool, ArkoAgentsError> {
        let running = self.task.running_tool_calls.clone();
        if running.is_empty() {
            return Ok(false);
        }
        log::info!(
            "Task {}: settling {} tool call(s) that were running when the task stopped.",
            self.task.id,
            running.len()
        );
        self.begin_tool_batch();
        if self.is_native_mode() {
            self.recover_native_calls(running).await?;
        } else {
            self.recover_text_calls(running).await?;
        }
        Ok(true)
    }

    async fn recover_native_calls(
        &mut self,
        running: Vec<RunningToolCall>,
    ) -> Result<(), ArkoAgentsError> {
        let turn_calls = self
            .task
            .conversation
            .messages
            .iter()
            .rev()
            .find(|message| message.role == ai::Role::Assistant)
            .map(|turn| turn.tool_calls.clone())
            .unwrap_or_default();
        // The turn's calls that were rejected or failed before the stop.
        let mut failed: std::collections::BTreeSet<usize> = self
            .task
            .pending_native_tool_results
            .iter()
            .filter(|result| result.is_error && result.call_id.is_some())
            .filter_map(|result| turn_calls.iter().position(|call| call.id == result.call_id))
            .collect();
        let mut results = NativeResults::new(&self.task);
        let mut runs = Vec::new();
        for entry in running {
            let call = turn_calls
                .iter()
                .find(|call| match &entry.call_id {
                    Some(id) => call.id.as_ref() == Some(id),
                    None => {
                        call.id.is_none()
                            && call.name == entry.tool_name
                            && Self::tool_call_args_to_map(call) == entry.parameters
                    }
                })
                .cloned();
            let answered = |task: &Task, call: &ai::ToolCall| {
                call.id.is_some()
                    && task
                        .pending_native_tool_results
                        .iter()
                        .any(|result| result.call_id == call.id)
            };
            let Some(call) = call.filter(|call| !answered(&self.task, call)) else {
                // Answered already, or not a call of the last turn: nothing
                // to settle.
                self.call_settled(entry.tool_call_id);
                continue;
            };
            let index = self.turn_position(&call);
            match self.recovery_of(&entry) {
                (Some(tool), ToolRecovery::Rerun | ToolRecovery::Resume) => {
                    let mut run = self.tool_run(
                        entry.tool_call_id,
                        &entry.tool_name,
                        tool,
                        entry.parameters.clone(),
                        entry.custom_output.clone(),
                    );
                    run.recovering = true;
                    runs.push(NativeRun {
                        index,
                        call,
                        run,
                        message_to_agent: entry.message_to_agent.clone(),
                    });
                }
                _ => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: entry.tool_call_id,
                        tool_name: entry.tool_name.clone(),
                        reason: Some(RESTARTED_CALL.to_string()),
                    })
                    .await?;
                    results.record(
                        &mut self.task,
                        index,
                        ai::ToolResult::error_from_tool_call(&call, RESTARTED_CALL),
                    );
                    self.call_settled(entry.tool_call_id);
                }
            }
        }
        failed.extend(results.failed(&self.task));
        runs.sort_by_key(|run| run.index);
        self.run_native_calls(runs, |state| !state.is_terminal(), failed, &mut results)
            .await?;
        results.finish(&mut self.task);
        Ok(())
    }

    async fn recover_text_calls(
        &mut self,
        running: Vec<RunningToolCall>,
    ) -> Result<(), ArkoAgentsError> {
        for entry in running {
            if self.task.current_state.is_terminal() {
                // A call before it ended the task.
                self.call_settled(entry.tool_call_id);
                continue;
            }
            let result = match self.recovery_of(&entry) {
                (Some(_), ToolRecovery::Rerun | ToolRecovery::Resume) => {
                    self.execute_text_call(
                        &entry.tool_name,
                        entry.parameters.clone(),
                        entry.tool_call_id,
                        entry.custom_output.clone(),
                        entry.message_to_agent.clone(),
                        true,
                    )
                    .await?
                }
                _ => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: entry.tool_call_id,
                        tool_name: entry.tool_name.clone(),
                        reason: Some(RESTARTED_CALL.to_string()),
                    })
                    .await?;
                    self.call_settled(entry.tool_call_id);
                    PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
                        .format_tool_execution_error_as_user_message(
                            &entry.tool_name,
                            RESTARTED_CALL,
                        )
                }
            };
            self.record_text_result(result).await;
        }
        Ok(())
    }

    /// When a task runs again (not after an interruption, which
    /// [`settle_interrupted_wait`](Self::settle_interrupted_wait) settles):
    /// the last turn's calls that have no result (the task stopped before
    /// they ran, or a version that didn't record running calls saved it)
    /// are answered as unreached, a pause on the recovered calls' tool
    /// errors starts, and the turn's results go into the conversation, as
    /// at the end of a turn. In output-based mode the results waiting (the
    /// turn's, and the recovered calls') are sent with [`RESTARTED_TURN`].
    /// A wait for an approval goes on, with the results that wait with it.
    /// `recovered`: calls were running when the task stopped.
    pub(super) async fn settle_open_turn(
        &mut self,
        recovered: bool,
    ) -> Result<(), ArkoAgentsError> {
        let approval_wait = matches!(
            self.task.current_state,
            TaskState::AwaitingToolApproval { .. } | TaskState::AwaitingBatchToolApproval { .. }
        );
        let in_progress = matches!(self.task.current_state, TaskState::InProgress);
        if approval_wait || !(recovered || in_progress) {
            self.end_tool_batch().await?;
            return Ok(());
        }
        if self.is_native_mode() {
            let open_turn = self
                .task
                .conversation
                .messages
                .last()
                .is_some_and(|message| {
                    message.role == ai::Role::Assistant && !message.tool_calls.is_empty()
                });
            let waiting = !self.task.pending_native_tool_results.is_empty()
                || !self.task.pending_native_tool_calls.is_empty();
            if !(recovered || open_turn || waiting) {
                self.end_tool_batch().await?;
                return Ok(());
            }
            self.answer_open_native_calls(UNREACHED_CALL, UNREACHED_CALL)
                .await?;
            self.end_tool_batch().await?;
            if self.native_results_are_due() {
                self.flush_native_results_to_conversation(None).await?;
            }
            return Ok(());
        }
        if !recovered && self.task.pending_text_tool_results.is_empty() {
            self.end_tool_batch().await?;
            return Ok(());
        }
        self.task
            .pending_text_tool_results
            .push(RESTARTED_TURN.to_string());
        self.end_tool_batch().await?;
        if self.task.current_state.is_terminal() {
            // As at the end of a turn that ended the task.
            self.task.pending_text_tool_results.clear();
        } else if !self.task.current_state.is_awaiting_approval() {
            let results = self.task.with_pending_text_tool_results(String::new());
            let message =
                PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
                    .augment_user_message_with_env_details_and_reminder(
                        &self.task,
                        &results,
                        self.task.user_message_count + 1,
                    )?;
            self.add_text_tool_results_to_conversation(&message);
            self.mark_details_sent();
            self.task.increment_user_message_count();
        }
        Ok(())
    }
}
