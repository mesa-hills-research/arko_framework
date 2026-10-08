//! The host's commands: the wait for a command and its deadline, what follows
//! a decision, the end of a wait without one, and the commands taken at a turn
//! boundary or while recovery waits.

use super::*;

/// The current wait for a command, timed by the opt-in
/// `AgentConfig::user_command_timeout_seconds`. Its deadline is set once,
/// when the wait starts: context injected during the wait and commands the
/// wait ignores don't move it, and a decision ends the wait (the next one
/// gets a deadline of its own).
#[derive(Debug, Clone, Copy)]
pub(super) enum CommandWait {
    /// No wait has started since the last one ended.
    NotStarted,
    /// Waiting until this deadline (`None`: no timeout).
    Until(Option<tokio::time::Instant>),
    /// The host is deciding this wait itself ([`UserCommand::HostDeciding`],
    /// or the run's [`HostDecisions`](super::super::HostDecisions) when it
    /// started): no deadline.
    HostDeciding,
}

impl AgentLoop {
    /// Takes what every place that reads commands takes alike: context,
    /// kept for the next turn boundary, and a handover request, noted.
    /// Returns any other command.
    pub(super) fn absorb_command(&mut self, command: UserCommand) -> Option<UserCommand> {
        match command {
            UserCommand::InjectContext { content } => {
                self.queue_context_injection(content);
                None
            }
            UserCommand::RequestHandover => {
                self.note_handover_request();
                None
            }
            command => Some(command),
        }
    }

    /// Handle control commands at every interruptible model/recovery wait.
    pub(super) async fn handle_recovery_command(
        &mut self,
        command: Option<UserCommand>,
    ) -> Result<bool, ArkoAgentsError> {
        match command.map(|command| self.absorb_command(command)) {
            Some(Some(UserCommand::CancelTask)) => {
                self.task.set_state(TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.into(),
                });
                self.stop_requested = true;
            }
            Some(Some(UserCommand::SuspendTask)) => self.stop_requested = true,
            None => {
                self.task.set_state(TaskState::Interrupted {
                    reason: "Command channel disconnected during model recovery.".into(),
                });
                self.stop_requested = true;
            }
            Some(_) => {}
        }
        if self.stop_requested {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        Ok(self.stop_requested)
    }

    /// Takes every command queued at a turn boundary without waiting:
    /// context joins the next request, and `CancelTask`, `SuspendTask`, or a
    /// closed channel stops the loop at once (returns true, with
    /// `TaskStateChanged`). Context taken before a stop is not lost; see
    /// [`keep_taken_context`](Self::keep_taken_context).
    pub(super) async fn take_queued_commands(&mut self) -> Result<bool, ArkoAgentsError> {
        loop {
            match self
                .command_receiver
                .try_recv()
                .map(|command| self.absorb_command(command))
            {
                Ok(None) => {}
                Ok(Some(UserCommand::CancelTask)) => {
                    self.task.set_state(TaskState::Cancelled {
                        reason: CANCELLED_BY_USER.to_string(),
                    });
                    break;
                }
                Ok(Some(UserCommand::SuspendTask)) => break,
                Err(mpsc::error::TryRecvError::Empty) => return Ok(false),
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // No call waits for its result at a boundary, so the
                    // context taken goes into the conversation of the
                    // interrupted task (unless a refusal fallback holds it).
                    debug_assert!(
                        !self.results_pending(),
                        "a turn boundary with results pending"
                    );
                    let reason = "Command channel disconnected.".to_string();
                    log::error!("Task {}: {}", self.task.id, reason);
                    self.interrupt(reason).await?;
                    self.stop_requested = true;
                    return Ok(true);
                }
                Ok(Some(other_cmd)) => {
                    log::warn!(
                        "Task {}: Unexpected command {:?} received outside of an approval state. Ignoring.",
                        self.task.id,
                        other_cmd
                    );
                }
            }
        }
        // Cancelled, or suspended in its current state.
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.stop_requested = true;
        Ok(true)
    }

    /// The wait for a command, with its deadline (the opt-in command
    /// timeout, set when the wait starts; none while the host decides the
    /// run's waits), then the command, or the end of the wait without one.
    pub(super) async fn wait_for_command(&mut self) -> Result<Flow, ArkoAgentsError> {
        let deadline = match self.command_wait {
            CommandWait::Until(deadline) => deadline,
            CommandWait::HostDeciding => None,
            // A wait that starts while the host decides the run's
            // waits is not timed, even if that changes.
            CommandWait::NotStarted if self.host_decisions.get() => {
                self.command_wait = CommandWait::HostDeciding;
                None
            }
            CommandWait::NotStarted => {
                let deadline = self
                    .task
                    .agent_config
                    .user_command_timeout_seconds
                    .and_then(|seconds| {
                        tokio::time::Instant::now().checked_add(Duration::from_secs(seconds))
                    });
                self.command_wait = CommandWait::Until(deadline);
                deadline
            }
        };
        // Not timed while the host decides the run's waits.
        let deadline = deadline.filter(|_| !self.host_decisions.get());
        let command_future = self.command_receiver.recv();
        let command_result = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, command_future).await,
            None => Ok(command_future.await),
        };
        match command_result {
            Ok(Some(cmd)) => self.apply_command(cmd).await,
            Ok(None) => {
                let reason = "Command channel disconnected during approval state.".to_string();
                self.interrupt_wait(reason, HOST_STOPPED_CALL).await
            }
            // The host took over the run's decisions during the
            // wait: it isn't timed any more.
            Err(_) if self.host_decisions.get() => Ok(Flow::NextPass),
            Err(_timeout_err) => {
                let seconds = self
                    .task
                    .agent_config
                    .user_command_timeout_seconds
                    .unwrap_or(0);
                let reason = format!(
                    "Timeout ({}s) waiting for user command in state {:?}.",
                    seconds, self.task.current_state
                );
                self.interrupt_wait(reason, &approval_timed_out_call(seconds))
                    .await
            }
        }
    }

    /// A command that arrived during the wait. Context, `HostDeciding` and
    /// handover requests are taken without ending it; any other command
    /// is applied to the state the task waits in.
    async fn apply_command(&mut self, cmd: UserCommand) -> Result<Flow, ArkoAgentsError> {
        // None of these ends the wait or moves its deadline.
        let Some(cmd) = self.absorb_command(cmd) else {
            return Ok(Flow::NextPass);
        };
        if let UserCommand::HostDeciding = cmd {
            self.command_wait = CommandWait::HostDeciding;
            return Ok(Flow::NextPass);
        }
        let original_state_before_command_processing = self.task.current_state.clone();

        let decision = match self.task.current_state.clone() {
            TaskState::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                parameters,
                ..
            } => {
                self.decide_tool_approval(&cmd, tool_call_id, tool_name, parameters)
                    .await?
            }
            TaskState::AwaitingBatchToolApproval {
                batch_id,
                ref pending_calls,
            } => {
                self.decide_batch_approval(&cmd, batch_id, pending_calls)
                    .await?
            }
            TaskState::AwaitingUserResponse { ref questions } => {
                self.answer_question(&cmd, questions).await?
            }
            TaskState::AwaitingErrorRecoveryApproval {
                ref error_message,
                ref cause,
                error_kind,
            } => {
                self.decide_error_recovery(&cmd, error_message, cause, error_kind)
                    .await?
            }
            _ => Decision::Decided(FollowUp::default()),
        };
        let Decision::Decided(follow_up) = decision else {
            return Ok(Flow::NextPass);
        };
        self.after_decision(&cmd, original_state_before_command_processing, follow_up)
            .await
    }

    /// After a command the wait didn't ignore: the wait ends, a
    /// cancellation or suspension takes effect, a changed state is
    /// reported, and a task that goes on gets the command's message (or
    /// a request to continue) unless the decision needs none.
    async fn after_decision(
        &mut self,
        cmd: &UserCommand,
        original_state_before_command_processing: TaskState,
        follow_up: FollowUp,
    ) -> Result<Flow, ArkoAgentsError> {
        let FollowUp {
            mut next_user_message_from_command,
            mut next_message_is_runtime,
            native_handled,
        } = follow_up;
        // A command the wait didn't ignore (those continue
        // above) ends it; a later wait gets its own deadline.
        self.command_wait = CommandWait::NotStarted;

        if let UserCommand::CancelTask = cmd {
            log::info!("Task {}: Cancelled by user command.", self.task.id);
            // The calls it waited on are answered as not run.
            self.settle_interrupted_wait(TASK_CANCELLED_CALL, TASK_CANCELLED_CALL)
                .await?;
            self.task.set_state(TaskState::Cancelled {
                reason: CANCELLED_BY_USER.to_string(),
            });
        }

        if let UserCommand::SuspendTask = cmd {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
            return Ok(Flow::Exit);
        }

        if self.task.current_state != original_state_before_command_processing
            && !self.state_reported()
        {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }

        if self.task.current_state.is_terminal() {
            return Ok(Flow::Exit);
        }

        // Output-based results of a command that leaves the
        // task waiting (a pause on a tool error) wait with it;
        // the message that ends a wait carries them first.
        if !self.is_native_mode() && !native_handled {
            if self.task.current_state.is_awaiting_approval() {
                if !next_user_message_from_command.is_empty() {
                    self.task
                        .pending_text_tool_results
                        .push(std::mem::take(&mut next_user_message_from_command));
                }
            } else if !self.task.pending_text_tool_results.is_empty() {
                next_user_message_from_command = self
                    .task
                    .with_pending_text_tool_results(next_user_message_from_command);
                next_message_is_runtime = false;
            }
        }

        if matches!(self.task.current_state, TaskState::InProgress) && !native_handled {
            if !next_user_message_from_command.is_empty() {
                let pb =
                    PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
                let next_msg_num = self.task.user_message_count + 1;
                let augmented_msg = pb.augment_user_message_with_env_details_and_reminder(
                    &self.task,
                    &next_user_message_from_command,
                    next_msg_num,
                )?;
                self.add_text_tool_results_to_conversation(&augmented_msg);
                self.mark_details_sent();
                if next_message_is_runtime {
                    self.task.mark_last_message_runtime();
                }
                self.task.increment_user_message_count();
                self.reset_llm_recovery();
            } else {
                self.add_runtime_message(CONTINUE_AFTER_DECISION)?;
                self.reset_llm_recovery();
            }
        }

        if self.task.current_state.is_awaiting_approval() {
            return Ok(Flow::NextPass);
        }
        Ok(Flow::Proceed)
    }

    /// A wait that ended without a command (a closed command channel, or
    /// the command timeout): the calls it waited on are answered with
    /// `call_reason`, and the task is interrupted.
    async fn interrupt_wait(
        &mut self,
        reason: String,
        call_reason: &str,
    ) -> Result<Flow, ArkoAgentsError> {
        log::error!("Task {}: {}", self.task.id, reason);
        self.settle_interrupted_wait(call_reason, HELD_CALL_NOT_RUN)
            .await?;
        self.interrupt(reason).await?;
        Ok(Flow::Exit)
    }

    /// Interrupts the task for `reason` (its host stopped, or a wait for a
    /// command timed out): `TaskStateChanged`, then `ErrorEncountered`.
    pub(super) async fn interrupt(&mut self, reason: String) -> Result<(), ArkoAgentsError> {
        self.task.set_state(TaskState::Interrupted {
            reason: reason.clone(),
        });
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: reason,
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: None,
        })
        .await
    }
}
