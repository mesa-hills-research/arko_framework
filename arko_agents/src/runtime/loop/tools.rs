//! Running tool calls, in either tool calling mode: their runs and groups,
//! stops while they run, the execution middleware's decisions, tool errors
//! under the tool-error policy and the batch's pause, path checks, result
//! events, and the wait for calls' approval.

use super::*;

/// Why running tool calls were stopped. See [`AgentLoop::run_group`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolsStopped {
    Cancelled,
    Suspended,
}

/// How a group of calls ended. See [`AgentLoop::run_group`].
pub(super) struct GroupRun {
    /// Each call's outcome, in order: `None` for a call a stop ended before
    /// it finished.
    pub(super) outcomes: Vec<Option<Result<RunOutcome, crate::tool::ToolError>>>,
    /// A cancellation or suspension that arrived while the calls ran.
    pub(super) stop: Option<ToolsStopped>,
}

/// The workspace check of a call's `is_path` parameters.
pub(super) enum PathCheck {
    Inside,
    /// This parameter's path is outside the workspace.
    Outside(String),
    /// A path couldn't be judged. The call is rejected with this reason.
    Invalid(String),
}

/// A call's paths under the path-safety policy. See
/// [`AgentLoop::path_verdict`].
pub(super) enum PathVerdict {
    /// The paths don't stop the call.
    Allowed,
    /// This parameter's path is outside the workspace, and the policy asks
    /// for approval.
    NeedsApproval(String),
    /// This parameter's path is outside the workspace, and the policy
    /// rejects the call.
    Outside(String),
    /// A path couldn't be judged. The call is rejected with this reason.
    Invalid(String),
}

/// Tool names excluded from result interception because their results
/// carry semantic meaning used directly by the runtime (task state
/// transitions, user-facing questions, etc.).
const INTERCEPTION_EXCLUDED_TOOLS: &[&str] = &[
    COMPLETE_TASK_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    crate::tool::builtins::CLOSE_QUESTIONS_TOOL_NAME,
];

impl AgentLoop {
    /// What the task's tool calls see of it. See [`RunEnvironment`].
    fn run_environment(&self) -> Arc<RunEnvironment> {
        Arc::new(RunEnvironment {
            task_id: self.task.id,
            agent_name: self.task.agent_config.name.clone(),
            initial_task_prompt: self.task.initial_task_prompt.clone(),
            working_directory: self.task.current_working_directory.clone(),
            middleware: self.execution_middleware.clone(),
            stop: self.host_stop.or(&self.calls_stop.signal()),
            model: self.task.conversation.model.clone(),
            image_sink: self.image_sink.clone(),
            image_room: Some(Arc::new(crate::tool::ImageRoom::left(
                &self.task.conversation.model,
                &self.task.conversation.messages,
            ))),
            context: self.context_arrivals.clone(),
            #[cfg(any(unix, windows))]
            jobs: self.background_jobs.as_ref().map(Arc::downgrade),
            terminal_tools: RunEnvironment::offers_terminal_tools(&self.tool_registry),
        })
    }

    /// How a tool's results are intercepted: by the configured interceptor,
    /// unless the tool opts out or its results carry meaning the runtime
    /// uses (`complete_task`, `ask_user`). `load_skill` results only when
    /// the skills configuration opts in, with its own failure policy.
    pub(super) fn interception_for(
        &self,
        tool_name: &str,
        tool: &dyn Tool,
    ) -> Option<Interception> {
        let config = self.interceptor_config.as_ref()?;
        if !tool.allow_result_interception() || INTERCEPTION_EXCLUDED_TOOLS.contains(&tool_name) {
            return None;
        }
        let policy = if tool_name == crate::skill::LOAD_SKILL_TOOL_NAME {
            self.task
                .agent_config
                .skills_config
                .as_ref()?
                .load_skill_interception
                .clone()?
        } else {
            config.failure_policy.clone()
        };
        Some(Interception {
            interceptor: config.interceptor.clone(),
            policy,
        })
    }

    /// Whether calls to a tool may run beside other parallel-safe calls:
    /// the tool is parallel-safe and the agent doesn't ask for one call per
    /// turn.
    pub(super) fn runs_in_parallel(&self, definition: &ToolDefinition) -> bool {
        let config = &self.task.agent_config;
        // `complete_task` ends the turn, so it always runs alone, after the
        // calls before it (whatever an override says).
        definition.name != COMPLETE_TASK_TOOL_NAME
            && !config.asks_one_tool_call_per_turn()
            && config.is_parallel_safe(definition)
    }

    /// A call to `tool` to run, under the agent's time limit and
    /// parallel-safety settings.
    pub(super) fn tool_run(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        tool: Arc<dyn Tool>,
        mut params: HashMap<String, Value>,
        custom_output: Option<String>,
    ) -> ToolRun {
        let config = &self.task.agent_config;
        crate::tool::builtins::read_file::ignore_line_ranges(
            tool.as_ref(),
            &config.tool_config,
            &mut params,
        );
        ToolRun {
            tool_call_id,
            tool_name: tool_name.to_string(),
            timeout: config
                .tool_config
                .tool_timeout(tool_name, tool.uses_host_timeout()),
            parallel_safe: self.runs_in_parallel(&tool.definition()),
            interception: self.interception_for(tool_name, tool.as_ref()),
            interruptible: tool.interruptible(),
            background: tool.background_mode(),
            tool,
            params,
            custom_output,
            call_id: None,
            message_to_agent: None,
            recovering: false,
        }
    }

    /// The message a finished call to `tool_name` sends the user, given its
    /// `message` argument: only a call of the built-in `message_user` sends
    /// one (see [`message_user::message_to_user`]).
    ///
    /// [`message_user::message_to_user`]: crate::tool::builtins::message_user::message_to_user
    pub(super) fn message_to_user(
        &self,
        tool_name: &str,
        message: Option<&Value>,
    ) -> Option<String> {
        let tool = self.tool_registry.get(tool_name)?;
        crate::tool::builtins::message_user::message_to_user(tool_name, tool.as_ref(), message)
    }

    /// Runs `runs` at the same time while watching the command channel:
    /// context sent meanwhile is kept for the next boundary, and
    /// `CancelTask` or `SuspendTask` stops the calls (dropping them, which
    /// kills a command they started) and is returned. Each call's outcome is
    /// returned in order otherwise. An error is a middleware's own failure.
    /// A group with a call that isn't [interruptible](crate::tool::Tool::interruptible)
    /// runs to its end, and a stop that arrived meanwhile is returned with
    /// the outcomes. Either way the calls' stop signal fires
    /// ([`TaskContext::stop`](crate::tool::TaskContext::stop)), so such a
    /// call can wind down. A closed channel doesn't stop the calls: the
    /// boundary after them handles it.
    ///
    /// Before any call starts, the calls are recorded as running
    /// ([`Task::running_tool_calls`]) and the task is reported to the
    /// snapshot sink, which saves it before this returns (the write-ahead
    /// point): a task that stops while its calls run knows which ones may
    /// have run. A group of calls that recovery simply runs again
    /// ([`ToolRecovery::Rerun`], without the host's own output) doesn't wait
    /// for the save: if the task stops before it is saved, those calls are
    /// run again, or asked for again, which is harmless.
    pub(super) async fn run_group(&mut self, runs: Vec<ToolRun>) -> GroupRun {
        use futures::StreamExt;
        for run in &runs {
            // A recovered call is recorded already.
            if !self
                .task
                .running_tool_calls
                .iter()
                .any(|running| running.tool_call_id == run.tool_call_id)
            {
                self.task.running_tool_calls.push(run.running());
            }
        }
        let durable = runs.iter().any(|run| {
            run.custom_output.is_some() || run.tool.recovery_mode() != ToolRecovery::Rerun
        });
        self.report_task(durable).await;
        let interruptible = runs.iter().all(|run| run.interruptible);
        let environment = self.run_environment();
        let mut outcomes: Vec<Option<_>> = runs.iter().map(|_| None).collect();
        let mut calls: futures::stream::FuturesUnordered<_> = runs
            .into_iter()
            .enumerate()
            .map(|(index, run)| {
                let environment = environment.clone();
                async move { (index, tool_run::run_tool(environment, run).await) }
            })
            .collect();
        let mut listening = true;
        let mut stop = None;
        loop {
            tokio::select! {
                biased;
                finished = calls.next() => match finished {
                    Some((index, outcome)) => outcomes[index] = Some(outcome),
                    None => return GroupRun { outcomes, stop },
                },
                command = self.command_receiver.recv(), if listening => {
                    let stopped = match command.map(|command| self.absorb_command(command)) {
                        Some(None) => None,
                        Some(Some(UserCommand::CancelTask)) => {
                            self.calls_stop.stop("Task cancelled.");
                            Some(ToolsStopped::Cancelled)
                        }
                        Some(Some(UserCommand::SuspendTask)) => {
                            self.calls_stop.stop("Task suspended.");
                            Some(ToolsStopped::Suspended)
                        }
                        Some(Some(other)) => {
                            log::warn!(
                                "Task {}: Unexpected command {:?} received while tool calls ran. Ignoring.",
                                self.task.id,
                                other
                            );
                            None
                        }
                        None => {
                            listening = false;
                            None
                        }
                    };
                    if let Some(stopped) = stopped {
                        if interruptible {
                            // The calls still running are dropped; those
                            // that finished keep their outcomes.
                            return GroupRun { outcomes, stop: Some(stopped) };
                        }
                        // Taken after the calls return; later commands
                        // wait in the channel.
                        stop = Some(stopped);
                        listening = false;
                    }
                }
            }
        }
    }

    /// Drops a call from the running calls once its result is in the task.
    pub(super) fn call_settled(&mut self, tool_call_id: Uuid) {
        self.task
            .running_tool_calls
            .retain(|running| running.tool_call_id != tool_call_id);
    }

    /// Applies a command that stopped running tool calls, once the calls
    /// that finished are handled: a cancellation ends the task (unless one
    /// of the calls already ended it), and a suspension stops the loop after
    /// the turn (the task keeps its state, reported as it stops).
    pub(super) async fn stop_after_calls(
        &mut self,
        stop: ToolsStopped,
    ) -> Result<(), ArkoAgentsError> {
        match stop {
            ToolsStopped::Cancelled => {
                if self.task.current_state.is_terminal() {
                    return Ok(());
                }
                log::info!("Task {}: Cancelled while tool calls ran.", self.task.id);
                self.task.set_state(TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.to_string(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await
            }
            ToolsStopped::Suspended => {
                self.stop_requested = true;
                self.suspended_tools = true;
                Ok(())
            }
        }
    }

    /// A middleware stopped the task at a call: it fails, or with `cancel`
    /// is cancelled, unless it already ended (another call of the same
    /// group stopped it first). The call is reported rejected either way.
    pub(super) async fn stop_by_middleware(
        &mut self,
        tool_call_id: Uuid,
        tool_name: &str,
        reason: &str,
        cancel: bool,
    ) -> Result<(), ArkoAgentsError> {
        if self.task.current_state.is_terminal() {
            return self
                .emit_event(AgentEvent::ToolRejected {
                    tool_call_id,
                    tool_name: tool_name.to_string(),
                    reason: Some(reason.to_string()),
                })
                .await;
        }
        let reason_text = reason.to_string();
        let state = if cancel {
            TaskState::Cancelled {
                reason: reason_text,
            }
        } else {
            TaskState::Failed {
                reason: reason_text,
                cause: None,
                error_kind: None,
            }
        };
        self.stop_at_tool_call(tool_call_id, tool_name, reason, state)
            .await
    }

    /// A middleware paused the task at a call, which it answered with
    /// `message`: the call is reported rejected, and the task awaits
    /// error-recovery approval with `reason` once the other calls of its
    /// batch have run (at once outside a batch), unless it already ended.
    pub(super) async fn pause_by_middleware(
        &mut self,
        tool_call_id: Uuid,
        tool_name: &str,
        message: &str,
        reason: String,
    ) -> Result<(), ArkoAgentsError> {
        self.emit_event(AgentEvent::ToolRejected {
            tool_call_id,
            tool_name: tool_name.to_string(),
            reason: Some(message.to_string()),
        })
        .await?;
        self.pause_for_middleware_error(tool_call_id, tool_name, reason)
            .await
    }

    /// Pauses for a middleware hook failure after a call that ran. Its result
    /// has already been emitted and is kept in the conversation, so the host
    /// can continue without executing the call again.
    pub(super) async fn pause_for_middleware_error(
        &mut self,
        tool_call_id: Uuid,
        tool_name: &str,
        reason: String,
    ) -> Result<(), ArkoAgentsError> {
        if self.task.current_state.is_terminal() {
            return Ok(());
        }
        log::info!(
            "Task {}: execution middleware failure paused the task at '{tool_name}': {reason}",
            self.task.id
        );
        if self.deferring_tool_errors {
            self.task.deferred_tool_errors.push(DeferredToolError {
                message: reason,
                tool_call_id: Some(tool_call_id),
                middleware_pause: true,
            });
            return Ok(());
        }
        self.task
            .set_state(TaskState::AwaitingErrorRecoveryApproval {
                error_message: reason.clone(),
                cause: ErrorRecoveryCause::Middleware,
                error_kind: None,
            });
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: reason,
            requires_approval_to_retry: true,
            tool_call_id: Some(tool_call_id),
            error_kind: None,
        })
        .await
    }

    /// Checks a call's `is_path` parameters against the workspace, as
    /// [`crate::utils::is_path_in_workspace`] resolves them. A missing,
    /// null or empty value is not checked. Any other non-string value is
    /// invalid, since a tool could read it as a path the check never saw.
    pub(super) fn check_path_parameters(
        &self,
        definition: &ToolDefinition,
        params: &HashMap<String, Value>,
    ) -> PathCheck {
        for param_def in definition.parameters.iter().filter(|param| param.is_path) {
            let path = match params.get(&param_def.name) {
                None | Some(Value::Null) => continue,
                Some(Value::String(path)) if path.is_empty() => continue,
                Some(Value::String(path)) => path,
                Some(other) => {
                    return PathCheck::Invalid(format!(
                        "Path validation failed for parameter '{}'. A path must be a string, not {other}.",
                        param_def.name
                    ));
                }
            };
            match crate::utils::is_path_in_workspace(path, &self.task.current_working_directory) {
                Ok(true) => {}
                Ok(false) => return PathCheck::Outside(param_def.name.clone()),
                Err(error) => {
                    return PathCheck::Invalid(format!(
                        "Path validation failed for parameter '{}': {error}",
                        param_def.name
                    ));
                }
            }
        }
        PathCheck::Inside
    }

    /// A call's `is_path` parameters checked against the workspace (see
    /// [`check_path_parameters`](Self::check_path_parameters)), and a path
    /// outside it judged under the tool's path-safety policy. Each tool
    /// calling mode answers a rejection its own way.
    pub(super) fn path_verdict(
        &self,
        tool_name: &str,
        definition: &ToolDefinition,
        params: &HashMap<String, Value>,
    ) -> PathVerdict {
        match self.check_path_parameters(definition, params) {
            PathCheck::Inside => PathVerdict::Allowed,
            PathCheck::Invalid(reason) => PathVerdict::Invalid(reason),
            PathCheck::Outside(param_name) => {
                match self.task.agent_config.get_path_safety_policy(tool_name) {
                    PathSafetyPolicy::AutoReject => PathVerdict::Outside(param_name),
                    PathSafetyPolicy::RequireApproval => PathVerdict::NeedsApproval(param_name),
                    PathSafetyPolicy::AutoApprove => PathVerdict::Allowed,
                }
            }
        }
    }

    /// Emits `ToolResult` with the original output and, when interception
    /// modified the result, a follow-up `ToolResultIntercepted` event.
    pub(super) async fn emit_tool_result_events(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        original_output: &str,
        effective_output: &str,
        was_intercepted: bool,
    ) -> Result<(), ArkoAgentsError> {
        self.emit_event(AgentEvent::ToolResult {
            tool_call_id,
            tool_name: tool_name.to_string(),
            output: original_output.to_string(),
        })
        .await?;

        if was_intercepted {
            self.emit_event(AgentEvent::ToolResultIntercepted {
                tool_call_id,
                tool_name: tool_name.to_string(),
                original_output: original_output.to_string(),
                intercepted_output: effective_output.to_string(),
            })
            .await?;
        }

        Ok(())
    }

    /// Waits for the approval of a turn's calls: one call alone, or the
    /// calls of a turn of several as a batch (`TaskStateChanged`, then
    /// `ToolCallAwaitingApproval` or `BatchToolCallAwaitingApproval`).
    pub(super) async fn await_approvals(
        &mut self,
        mut calls_needing_approval: Vec<PendingToolCallInfo>,
        overall_batch_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        if calls_needing_approval.len() == 1 && overall_batch_id.is_none() {
            let single_pending = calls_needing_approval.remove(0);
            self.task.set_state(TaskState::AwaitingToolApproval {
                tool_call_id: single_pending.tool_call_id,
                tool_name: single_pending.tool_name.clone(),
                parameters: single_pending.parameters.clone(),
                tool_description: single_pending.tool_description.clone(),
                approval_reason: single_pending.approval_reason.clone(),
            });
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
            self.emit_event(AgentEvent::ToolCallAwaitingApproval {
                tool_call_id: single_pending.tool_call_id,
                tool_name: single_pending.tool_name,
                parameters: single_pending.parameters,
                tool_description: single_pending.tool_description,
                approval_reason: single_pending.approval_reason,
            })
            .await?;
        } else {
            let batch_id_for_state = overall_batch_id.unwrap_or_else(Uuid::new_v4);
            self.task.set_state(TaskState::AwaitingBatchToolApproval {
                batch_id: batch_id_for_state,
                pending_calls: calls_needing_approval,
            });
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
            self.emit_event(AgentEvent::BatchToolCallAwaitingApproval {
                batch_id: batch_id_for_state,
                pending_calls: match &self.task.current_state {
                    TaskState::AwaitingBatchToolApproval { pending_calls, .. } => {
                        pending_calls.clone()
                    }
                    _ => vec![],
                },
            })
            .await?;
        }
        Ok(())
    }

    /// Rejects a call a middleware stopped the task at, and moves the task to
    /// `state` (failed or cancelled).
    async fn stop_at_tool_call(
        &mut self,
        tool_call_id: Uuid,
        tool_name: &str,
        reason: &str,
        state: TaskState,
    ) -> Result<(), ArkoAgentsError> {
        self.emit_event(AgentEvent::ToolRejected {
            tool_call_id,
            tool_name: tool_name.to_string(),
            reason: Some(reason.to_string()),
        })
        .await?;
        self.task.set_state(state);
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await
    }

    /// Why a call of the turn was not run once the task stopped at an
    /// earlier call: it failed on a tool error, a middleware failed or
    /// cancelled it (a workflow's cancellation gives "This call was not run:
    /// Workflow cancelled."), it was cancelled or suspended while calls ran,
    /// or the turn otherwise ended.
    pub(super) fn not_run_after_stop(&self) -> String {
        if self.stop_requested && !self.task.current_state.is_terminal() {
            return TASK_SUSPENDED_CALL.to_string();
        }
        match &self.task.current_state {
            TaskState::Failed { .. } if self.failed_on_tool_error => TASK_FAILED_CALL.to_string(),
            TaskState::Cancelled { reason } if reason == CANCELLED_BY_USER => {
                TASK_CANCELLED_CALL.to_string()
            }
            TaskState::Failed { reason, .. } | TaskState::Cancelled { reason } => {
                format!("This call was not run: {}.", reason.trim_end_matches('.'))
            }
            _ => UNREACHED_CALL.to_string(),
        }
    }

    /// Starts running a batch of tool calls (a turn's, or an approval's):
    /// under `ToolErrorPolicy::PauseAndAwaitApproval` a failed call no longer
    /// pauses the task at once, so the batch's other calls run first, and
    /// [`end_tool_batch`](Self::end_tool_batch) pauses once.
    pub(super) fn begin_tool_batch(&mut self) {
        self.deferring_tool_errors = true;
    }

    /// Ends a batch begun with [`begin_tool_batch`](Self::begin_tool_batch):
    /// pauses once on its tool errors, with any a batch of the same turn
    /// left while calls awaited approval. While calls of the turn still
    /// await approval the pause waits for their decisions. A task that
    /// stopped doesn't pause.
    pub(super) async fn end_tool_batch(&mut self) -> Result<(), ArkoAgentsError> {
        self.deferring_tool_errors = false;
        if self.task.current_state.is_terminal() {
            self.task.deferred_tool_errors.clear();
        }
        if self.task.deferred_tool_errors.is_empty()
            || !matches!(self.task.current_state, TaskState::InProgress)
        {
            return Ok(());
        }
        let errors = std::mem::take(&mut self.task.deferred_tool_errors);
        let error_message = errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        // A decision the middleware left to the host is the host's, whatever
        // tool errors come with it.
        let cause = if errors.iter().any(|error| error.middleware_pause) {
            ErrorRecoveryCause::Middleware
        } else {
            ErrorRecoveryCause::ToolError
        };
        self.task
            .set_state(TaskState::AwaitingErrorRecoveryApproval {
                error_message: error_message.clone(),
                cause,
                error_kind: None,
            });
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        // The message lists every failure; one call is named only when it
        // is the only one.
        let tool_call_id = match errors.as_slice() {
            [error] => error.tool_call_id,
            _ => None,
        };
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message,
            requires_approval_to_retry: true,
            tool_call_id,
            error_kind: None,
        })
        .await
    }

    pub(super) async fn handle_tool_exec_error(
        &mut self,
        tool_name: &str,
        error: crate::tool::ToolError,
        tool_call_id: Option<Uuid>,
    ) -> Result<String, ArkoAgentsError> {
        let error_message_for_event = format!("Tool '{}' execution failed: {}", tool_name, error);
        log::warn!("Task {}: {}", self.task.id, error_message_for_event);

        let prompt_builder_for_error_msg = self.prompts();
        let error_message_for_llm = prompt_builder_for_error_msg
            .format_tool_execution_error_as_user_message(tool_name, &error.to_string());

        let policy = &self
            .task
            .agent_config
            .error_recovery_config
            .on_tool_execution_error;
        match policy {
            // The call's own result reports the error, and the batch goes on.
            ToolErrorPolicy::InformAgentAndContinue => {
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event.clone(),
                    requires_approval_to_retry: false,
                    tool_call_id,
                    error_kind: None,
                })
                .await?;
                Ok(error_message_for_llm)
            }
            // The batch's other calls run first; `end_tool_batch` pauses.
            // The call's own result reports the error in place (output-based
            // mode; a native call's error result carries it already).
            ToolErrorPolicy::PauseAndAwaitApproval if self.deferring_tool_errors => {
                self.task.deferred_tool_errors.push(DeferredToolError {
                    message: error_message_for_event,
                    tool_call_id,
                    middleware_pause: false,
                });
                Ok(error_message_for_llm)
            }
            ToolErrorPolicy::PauseAndAwaitApproval => {
                self.task
                    .set_state(TaskState::AwaitingErrorRecoveryApproval {
                        error_message: error_message_for_event.clone(),
                        cause: ErrorRecoveryCause::ToolError,
                        error_kind: None,
                    });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event.clone(),
                    requires_approval_to_retry: true,
                    tool_call_id,
                    error_kind: None,
                })
                .await?;
                Ok(error_message_for_llm)
            }
            ToolErrorPolicy::FailTask => {
                // A call running beside the one that ended the task (in a
                // group of parallel-safe calls) doesn't end it again.
                if !self.task.current_state.is_terminal() {
                    self.failed_on_tool_error = true;
                    self.task.set_state(TaskState::Failed {
                        reason: error_message_for_event.clone(),
                        cause: None,
                        error_kind: None,
                    });
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                }
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event,
                    requires_approval_to_retry: false,
                    tool_call_id,
                    error_kind: None,
                })
                .await?;
                Ok(String::new())
            }
        }
    }
}
