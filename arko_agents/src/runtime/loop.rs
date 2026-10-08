use super::command::{ToolApprovalDecision, UserCommand};
use super::event::{AgentEvent, ShortenReason};
use super::result_budget;
use super::tool_run::{self, Interception, RunEnvironment, RunOutcome, ToolRun};
use crate::agent_config::{
    LLMErrorPolicy, PathSafetyPolicy, PlanUsageAction, ToolCallingMode, ToolErrorPolicy,
    UsageLimitPolicy,
};
use crate::error::ArkoAgentsError;
use crate::handover::{HandoverHook, HandoverRecord, HandoverRequest, HandoverSeed};
use crate::parsing::{
    ParsedToolCall, PromptBuilder, RawParserEvent, ToolCallGroup, ToolCallParser,
    ToolParsingConfig, assemble_tool_calls,
};
use crate::skill::SkillPromptContext;
use crate::task::{
    ApprovalReason, CANCELLED_BY_USER, DeferredToolError, ErrorRecoveryCause, PendingToolCallInfo,
    PlanUsageWindowMark, RunningToolCall, Task, TaskState,
};
// Names the test modules (`use super::*`) rely on.
#[cfg(test)]
use crate::agent_config::AgentConfig;
#[cfg(test)]
use crate::tool::{
    InterceptedResult, TaskContext, ToolCallContext, ToolExecutionContext, ToolExecutionDecision,
    ToolExecutionOutcome, ToolOutput,
};
use crate::tool::{
    InterceptorConfig, Tool, ToolDefinition, ToolExecutionMiddleware, ToolImage, ToolRecovery,
    ToolRegistry,
    builtins::{ASK_USER_TOOL_NAME, COMPLETE_TASK_TOOL_NAME},
};
use arko_ai::{self as ai};
use futures::TryStreamExt;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
#[cfg(test)]
use tokio::time::timeout;
use tokio::time::{Duration, sleep as tokio_sleep};
use uuid::Uuid;

// The loop's parts.
mod texts;
use texts::*;
mod overflow;
use overflow::*;
mod handover;
mod monitors;
mod refusal;
use refusal::*;
mod recovery;
use recovery::*;
mod request;
#[cfg(test)]
use request::*;
mod stream;
use stream::*;
mod tools;
use tools::*;
mod native;
use native::*;
mod text;

/// What a decision on a wait leaves for after it; see
/// [`AgentLoop::after_decision`].
#[derive(Default)]
struct FollowUp {
    /// The message the decision adds (output-based results, an answer, a
    /// note), empty for none.
    next_user_message_from_command: String,
    /// That message is the runtime's.
    next_message_is_runtime: bool,
    /// The decision needs no follow-up message (it added what it had to).
    native_handled: bool,
}

/// How a command applied to the state a task waits in.
enum Decision {
    /// It doesn't fit the wait: the pass ends, and the wait goes on.
    Ignored,
    /// It decided the wait.
    Decided(FollowUp),
}

/// The current wait for a command, timed by the opt-in
/// `AgentConfig::user_command_timeout_seconds`. Its deadline is set once,
/// when the wait starts: context injected during the wait and commands the
/// wait ignores don't move it, and a decision ends the wait (the next one
/// gets a deadline of its own).
#[derive(Debug, Clone, Copy)]
enum CommandWait {
    /// No wait has started since the last one ended.
    NotStarted,
    /// Waiting until this deadline (`None`: no timeout).
    Until(Option<tokio::time::Instant>),
    /// The host is deciding this wait itself ([`UserCommand::HostDeciding`],
    /// or the run's [`HostDecisions`](super::HostDecisions) when it
    /// started): no deadline.
    HostDeciding,
}

/// Where the loop of [`AgentLoop::run`] goes after a phase of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// On to the pass's next phase.
    Proceed,
    /// To the next pass (`continue`), which reports the task, then checks
    /// for a stop or an end.
    NextPass,
    /// Out of the loop, to its finish (`break`).
    Exit,
}

/// Applies a phase's [`Flow`] in the loop of [`AgentLoop::run`], at the top
/// level of the loop's body (where `continue` and `break` are the loop's).
macro_rules! step {
    ($flow:expr) => {
        match $flow {
            Flow::Proceed => {}
            Flow::NextPass => continue,
            Flow::Exit => break,
        }
    };
}

/// The main agent execution loop.
pub struct AgentLoop {
    task: Task,
    llm_client: Arc<ai::LLMClient>,
    tool_registry: Arc<ToolRegistry>,
    event_sender: mpsc::Sender<AgentEvent>,
    command_receiver: mpsc::Receiver<UserCommand>,
    parsing_config: ToolParsingConfig,
    event_throttle_duration: Duration,
    simplified_tool_events: bool,

    text_buffer: String,
    reasoning_buffer: String,
    last_text_emit_time: Instant,
    last_reasoning_emit_time: Instant,

    llm_interaction_attempts: u32,
    recovery_started: Option<tokio::time::Instant>,
    capacity_recovery_started: Option<tokio::time::Instant>,
    capacity_retry_attempts: u32,
    /// Requests in this recovery episode that ended in a wait that doesn't
    /// count against the ordinary limits (capacity, rate limits).
    indefinite_interactions: u32,
    /// Ordinary recovery time used before its clock was paused for such a
    /// wait.
    paused_recovery_elapsed: Option<Duration>,
    /// The request in flight (or last made) went to the fallback model
    /// ([`Task::fallback_for`]).
    current_request_is_fallback: bool,
    /// Time spent in this recovery episode waiting out rate limits.
    rate_limit_waited: Duration,
    /// Cache monitor state: a low-efficiency warning is out, a pause
    /// episode is active, and the pause to apply before the next request.
    cache_warned: bool,
    cache_pause_episode: bool,
    cache_pause_pending: Option<String>,
    stop_requested: bool,
    /// A batch of tool calls is running; see [`begin_tool_batch`](Self::begin_tool_batch).
    deferring_tool_errors: bool,
    /// The task failed on a tool error under `ToolErrorPolicy::FailTask`
    /// (not on a middleware's decision); see
    /// [`not_run_after_stop`](Self::not_run_after_stop).
    failed_on_tool_error: bool,
    /// `SuspendTask` stopped running tool calls; the loop reports the
    /// task's state as it stops.
    suspended_tools: bool,
    /// The index of the message that carried this turn's environment
    /// details and reminder, when the loop added it since the last request;
    /// see [`add_taken_context`](Self::add_taken_context).
    details_sent_with: Option<usize>,
    /// The state the last `TaskStateChanged` reported, so a state reported
    /// by one path isn't reported again by another.
    last_reported_state: std::sync::Mutex<Option<TaskState>>,
    /// The wait for a command the task is in, with its deadline.
    command_wait: CommandWait,
    /// While set, the host decides every wait of the run and none is timed.
    pub(super) host_decisions: super::HostDecisions,
    #[cfg(test)]
    scripted_responses: std::sync::Mutex<VecDeque<ScriptedStreamResult>>,
    /// The model of every request made, in order.
    #[cfg(test)]
    requested_models: Arc<std::sync::Mutex<Vec<String>>>,
    /// The test's trace of the loop's events and snapshots; see
    /// [`loop_trace`].
    #[cfg(test)]
    trace: loop_trace::LoopTrace,

    skill_prompt_context: Option<SkillPromptContext>,

    interceptor_config: Option<InterceptorConfig>,
    execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    /// Asked for context at each safe turn boundary.
    pub(super) turn_context: Option<Arc<dyn super::TurnContextSource>>,
    /// Called when the conversation is handed over.
    pub(super) handover_hook: Option<Arc<dyn HandoverHook>>,
    /// Receives the task at its safe points; see [`report_task`](Self::report_task).
    pub(super) snapshots: crate::snapshot::TaskSnapshots,
    /// The host's stop signal for the task's calls; see
    /// [`AgentRuntime::with_stop_signal`](super::AgentRuntime::with_stop_signal).
    pub(super) host_stop: crate::stop::StopSignal,
    /// Fired when `CancelTask` or `SuspendTask` arrives while calls run, so
    /// that a call the runtime waits for winds down; see
    /// [`TaskContext::stop`](crate::tool::TaskContext::stop).
    calls_stop: crate::stop::StopSource,
    /// The plan-usage SlowDown's delay was checked against the prompt-cache
    /// lifetime (once per run).
    slow_down_cache_checked: bool,
    /// A threshold handover was skipped and logged; see
    /// `start_handover_if_due`.
    handover_skip_warned: bool,
    tool_context_cursor: usize,
    file_tree_baseline: Option<crate::file_tree::FileTreeSnapshot>,
    file_tree_error: Option<String>,
    /// Bytes per token of this task's requests, calibrated from their
    /// reported usage; see [`result_budget`].
    bytes_per_token: f64,
    /// The request in flight, or last sent.
    sent_request: Option<RequestMark>,
    /// The latest request that was answered: the task's latest reported
    /// usage is its.
    answered_request: Option<RequestMark>,
    /// Requests sent again in this episode after shortening a tool result
    /// the provider found over the model's input limit.
    overflow_recoveries: u32,
}

impl AgentLoop {
    /// Creates a new AgentLoop.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: Task,
        llm_client: Arc<ai::LLMClient>,
        tool_registry: Arc<ToolRegistry>,
        event_sender: mpsc::Sender<AgentEvent>,
        command_receiver: mpsc::Receiver<UserCommand>,
        event_throttle_ms: Option<u64>,
        simplified_tool_events: bool,
        skill_prompt_context: Option<SkillPromptContext>,
        interceptor_config: Option<InterceptorConfig>,
        execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    ) -> Self {
        let parsing_config = task.agent_config.get_tool_parsing_config();
        let throttle_ms = event_throttle_ms.unwrap_or(0);
        // The opt-in stream idle timeout applies to this agent's requests
        // only; clones of a client share its credentials.
        let llm_client = match task.agent_config.llm_retry_config.stream_idle_timeout_ms {
            Some(timeout) if timeout > 0 => Arc::new(
                (*llm_client)
                    .clone()
                    .with_stream_idle_timeout(Duration::from_millis(timeout)),
            ),
            _ => llm_client,
        };
        let now = Instant::now();
        // The tools offered are the only tools that run: a call to any other
        // one is answered as unavailable, in either tool calling mode.
        let offered = task.agent_config.tool_config.offered_tool_names();
        let tool_registry =
            Arc::new(tool_registry.restricted_to(offered.iter().map(String::as_str)));

        for name in tool_registry.list_tool_names() {
            if let Some(tool) = tool_registry.get(&name) {
                tool.reset_context(task.id);
            }
        }

        Self {
            task,
            llm_client,
            tool_registry,
            event_sender,
            command_receiver,
            parsing_config,
            event_throttle_duration: Duration::from_millis(throttle_ms),
            simplified_tool_events,
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            last_text_emit_time: now,
            last_reasoning_emit_time: now,
            llm_interaction_attempts: 0,
            recovery_started: None,
            capacity_recovery_started: None,
            capacity_retry_attempts: 0,
            indefinite_interactions: 0,
            paused_recovery_elapsed: None,
            current_request_is_fallback: false,
            rate_limit_waited: Duration::ZERO,
            cache_warned: false,
            cache_pause_episode: false,
            cache_pause_pending: None,
            stop_requested: false,
            deferring_tool_errors: false,
            failed_on_tool_error: false,
            suspended_tools: false,
            details_sent_with: None,
            last_reported_state: std::sync::Mutex::new(None),
            command_wait: CommandWait::NotStarted,
            host_decisions: super::HostDecisions::default(),
            #[cfg(test)]
            scripted_responses: std::sync::Mutex::new(VecDeque::new()),
            #[cfg(test)]
            requested_models: Arc::default(),
            #[cfg(test)]
            trace: loop_trace::LoopTrace::for_current_test(),
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
            turn_context: None,
            handover_hook: None,
            snapshots: Default::default(),
            host_stop: crate::stop::StopSignal::never(),
            calls_stop: crate::stop::StopSource::new(),
            slow_down_cache_checked: false,
            handover_skip_warned: false,
            tool_context_cursor: 0,
            file_tree_baseline: None,
            file_tree_error: None,
            bytes_per_token: result_budget::DEFAULT_BYTES_PER_TOKEN,
            sent_request: None,
            answered_request: None,
            overflow_recoveries: 0,
        }
    }

    /// Binds the task's tools to its command channel
    /// ([`Tool::bind_runtime`](crate::tool::Tool::bind_runtime)).
    pub(super) fn bind_tools(&self, sender: &mpsc::Sender<UserCommand>) {
        for name in self.tool_registry.list_tool_names() {
            if let Some(tool) = self.tool_registry.get(&name) {
                tool.bind_runtime(self.task.id, sender.downgrade());
            }
        }
    }

    /// Keeps context taken from the command channel with the task
    /// ([`Task::pending_context_injections`]) until it is added to the
    /// conversation, so a task that stops first never loses it.
    fn queue_context_injection(&mut self, content: String) {
        if !content.trim().is_empty() {
            self.task.pending_context_injections.push(content);
        }
    }

    fn observe_committed_tool_context(&mut self) {
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

    fn refresh_file_tree_context(&mut self) {
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
    fn flush_context_injections(&mut self) -> Result<(), ArkoAgentsError> {
        if self.handover_report_due() {
            return Ok(());
        }
        if let Some(content) = self
            .turn_context
            .as_ref()
            .and_then(|source| source.turn_context(self.task.id))
        {
            self.queue_context_injection(content);
        }
        self.add_taken_context()
    }

    /// Adds the context taken from the command channel
    /// ([`Task::pending_context_injections`]) to the conversation as one
    /// user message headed `# Workflow context update`, in the order it was
    /// sent. Only where a user message can go now; see
    /// [`context_can_be_added_now`](Self::context_can_be_added_now).
    fn add_taken_context(&mut self) -> Result<(), ArkoAgentsError> {
        // While the handover report is due, context waits for the fresh
        // conversation: added now, it would be read only through the report
        // and could be lost from it (a message that needs a reply, say).
        if self.task.pending_context_injections.is_empty() || self.handover_report_due() {
            return Ok(());
        }
        let content = self.task.pending_context_injections.join("\n\n");
        let prompt_builder =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
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
        self.task.conversation.add_user_message(&message);
        self.task.increment_user_message_count();
        self.task.pending_context_injections.clear();
        Ok(())
    }

    /// Whether continuing after an error-recovery pause sends the pending
    /// refusal fallback's request again, unchanged: a fallback is pending
    /// and no output-based results wait to be sent first.
    fn fallback_resend(&self) -> bool {
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
    fn context_can_be_added_now(&self) -> bool {
        !self.results_pending()
            && self.task.fallback_for.is_none()
            && self.task.refused_input.is_none()
    }

    /// Whether the last turn's results are not all in the conversation yet:
    /// the task awaits a command (an approval, an answer, or a recovery
    /// decision), results wait with it, or the conversation ends on a native
    /// turn whose calls have none.
    fn results_pending(&self) -> bool {
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
    fn keep_taken_context(&mut self) -> Result<(), ArkoAgentsError> {
        if matches!(self.task.current_state, TaskState::Completed { .. })
            || !self.context_can_be_added_now()
        {
            return Ok(());
        }
        self.add_taken_context()
    }

    /// Reports the task to the snapshot sink, if there is one: a safe point
    /// (see [`TaskSnapshotSink`](crate::snapshot::TaskSnapshotSink)). With
    /// `durable`, returns once the sink has saved it.
    async fn report_task(&mut self, durable: bool) {
        #[cfg(test)]
        self.trace.snapshot(&self.task, durable);
        self.snapshots.report(&self.task, durable).await;
    }

    fn is_native_mode(&self) -> bool {
        self.task.agent_config.tool_config.tool_calling_mode == ToolCallingMode::Native
    }

    /// Handle control commands at every interruptible model/recovery wait.
    async fn handle_recovery_command(
        &mut self,
        command: Option<UserCommand>,
    ) -> Result<bool, ArkoAgentsError> {
        match command {
            Some(UserCommand::InjectContext { content }) => self.queue_context_injection(content),
            Some(UserCommand::RequestHandover) => self.note_handover_request(),
            Some(UserCommand::CancelTask) => {
                self.task.set_state(TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.into(),
                });
                self.stop_requested = true;
            }
            Some(UserCommand::SuspendTask) => self.stop_requested = true,
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
    async fn take_queued_commands(&mut self) -> Result<bool, ArkoAgentsError> {
        loop {
            match self.command_receiver.try_recv() {
                Ok(UserCommand::InjectContext { content }) => {
                    self.queue_context_injection(content);
                }
                Ok(UserCommand::RequestHandover) => self.note_handover_request(),
                Ok(UserCommand::CancelTask) => {
                    self.task.set_state(TaskState::Cancelled {
                        reason: CANCELLED_BY_USER.to_string(),
                    });
                    break;
                }
                Ok(UserCommand::SuspendTask) => break,
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
                    .await?;
                    self.stop_requested = true;
                    return Ok(true);
                }
                Ok(other_cmd) => {
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

    /// Records that the conversation's last message carries this turn's
    /// environment details and reminder.
    fn mark_details_sent(&mut self) {
        self.details_sent_with = self.task.conversation.messages.len().checked_sub(1);
    }

    /// Checks the configuration before anything is reported (the
    /// developer environment details the run can show need content), and
    /// estimates the saved conversation's latest request.
    fn check_config(&mut self) -> Result<(), ArkoAgentsError> {
        self.task.agent_config.validate()?;
        self.answered_request = self.saved_request_mark();
        if self.is_native_mode()
            && self
                .task
                .agent_config
                .llm_config
                .generation_config
                .tool_config
                .as_ref()
                .is_some_and(|tools| !tools.tools.is_empty())
        {
            log::warn!(
                "Task {}: the model's tool_config lists tool definitions, which native tool calling doesn't send: the agent's tools come from its tool registry. Its tool_choice and parallel_tool_calls apply.",
                self.task.id
            );
        }
        // Every developer environment detail the run can show needs content:
        // details placed in every user message on every turn, and the others
        // while the system prompt and task prompt are still to be written.
        let writing_prompts = self.task.conversation.system_message.is_none();
        if let Some(error) = self
            .task
            .developer_env_details
            .iter()
            .flatten()
            .filter(|detail| {
                writing_prompts
                    || detail.placement
                        == crate::agent_config::EnvironmentDetailPlacement::EachUserMessage
            })
            .find_map(|detail| detail.missing_content_error())
        {
            return Err(ArkoAgentsError::Configuration(error));
        }
        log::info!(
            "Agent loop starting for task ID: {} (mode: {:?})",
            self.task.id,
            self.task.agent_config.tool_config.tool_calling_mode
        );
        Ok(())
    }

    /// The run's start: the task in progress (reported), what a stopped
    /// run left settled, and the prompts written.
    async fn start(&mut self) -> Result<(), ArkoAgentsError> {
        let interrupted = self.start_state().await?;
        // A task saved while its tool calls ran (by a host that stopped)
        // settles them before anything else.
        let recovered = self.recover_running_calls().await?;
        if interrupted {
            // Older versions left a native turn without results when an
            // approval wait ended this way.
            self.settle_interrupted_wait(UNDECIDED_CALL, UNDECIDED_CALL)
                .await?;
            self.end_tool_batch().await?;
        } else {
            self.settle_open_turn(recovered).await?;
        }
        self.write_prompts()
    }

    /// Puts a task that is pending or was interrupted in progress, and one
    /// saved while approved calls ran out of its approval wait, and reports
    /// a task in progress. Returns whether the task was interrupted.
    async fn start_state(&mut self) -> Result<bool, ArkoAgentsError> {
        let interrupted = matches!(self.task.current_state, TaskState::Interrupted { .. });
        if matches!(self.task.current_state, TaskState::Pending)
            || matches!(self.task.current_state, TaskState::Interrupted { .. })
        {
            self.task.set_state(TaskState::InProgress);
            self.task.monitor_pause = false;
        }
        // Calls run only once their approval is decided: a task saved while
        // they ran no longer waits for it.
        if !self.task.running_tool_calls.is_empty()
            && matches!(
                self.task.current_state,
                TaskState::AwaitingToolApproval { .. }
                    | TaskState::AwaitingBatchToolApproval { .. }
            )
        {
            self.task.set_state(TaskState::InProgress);
        }

        if matches!(self.task.current_state, TaskState::InProgress) {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        Ok(interrupted)
    }

    /// Writes the system prompt and the task prompt, unless the
    /// conversation has them already.
    fn write_prompts(&mut self) -> Result<(), ArkoAgentsError> {
        if self.task.conversation.system_message.is_none() {
            let mut tool_defs_for_prompt: Vec<ToolDefinition> = Vec::new();
            for name in &self.task.agent_config.tool_config.available_tool_names {
                if let Some(tool) = self.tool_registry.get(name) {
                    tool_defs_for_prompt.push(tool.definition());
                } else {
                    log::warn!(
                        "Agent {} configured with tool '{}' but it was not found in the registry.",
                        self.task.agent_config.name,
                        name
                    );
                }
            }
            if !tool_defs_for_prompt
                .iter()
                .any(|def| def.name == COMPLETE_TASK_TOOL_NAME)
                && let Some(ct_tool) = self.tool_registry.get(COMPLETE_TASK_TOOL_NAME)
            {
                tool_defs_for_prompt.push(ct_tool.definition());
            }

            let prompt_builder = PromptBuilder::new(
                &self.task.agent_config,
                tool_defs_for_prompt.iter().collect(),
                &self.parsing_config,
                self.skill_prompt_context.as_ref(),
            );
            let system_prompt = prompt_builder.build_system_prompt(&self.task)?;
            self.task.conversation.system_message = Some(system_prompt);

            if self.task.conversation.messages.is_empty()
                && !self.task.initial_task_prompt.is_empty()
            {
                let initial_user_prompt = prompt_builder
                    .format_initial_task_prompt(&self.task, &self.task.initial_task_prompt)?;
                self.task
                    .conversation
                    .add_user_message(&initial_user_prompt);
                self.task.increment_user_message_count();
                self.reset_llm_recovery();
            }
        }
        Ok(())
    }

    /// The start of a pass, after its snapshot: a stop requested since the
    /// last pass, or a task that ended, leaves the loop.
    async fn pass_start(&mut self) -> Result<Flow, ArkoAgentsError> {
        if self.stop_requested {
            // Suspended while tool calls ran, in the state the turn left
            // it in, which is reported as a suspension always is, unless
            // a call already ended the task and that was reported.
            let ended_and_reported = self.task.current_state.is_terminal() && self.state_reported();
            if std::mem::take(&mut self.suspended_tools) && !ended_and_reported {
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
            }
            return Ok(Flow::Exit);
        }
        if self.task.current_state.is_terminal() {
            log::info!(
                "Task {} reached terminal state: {:?}",
                self.task.id,
                self.task.current_state
            );
            return Ok(Flow::Exit);
        }
        Ok(Flow::Proceed)
    }

    /// The turn boundary before a model request: a stop or the end of the
    /// task, the commands queued meanwhile, then the handover, context and
    /// file tree, a user turn to answer, and the monitors and the host's
    /// hold, any of which can end the pass instead of the request.
    async fn turn_boundary(&mut self) -> Result<Flow, ArkoAgentsError> {
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
            self.refresh_file_tree_context();
        }
        self.ensure_user_turn_before_request()?;
        if !self.apply_plan_usage_threshold().await? {
            return Ok(Flow::NextPass);
        }
        if let Some(message) = self.cache_pause_pending.take() {
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

    /// The wait for a command, with its deadline (the opt-in command
    /// timeout, set when the wait starts; none while the host decides the
    /// run's waits), then the command, or the end of the wait without one.
    async fn wait_for_command(&mut self) -> Result<Flow, ArkoAgentsError> {
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
        // Neither ends the wait or moves its deadline.
        if let UserCommand::InjectContext { ref content } = cmd {
            self.queue_context_injection(content.clone());
            return Ok(Flow::NextPass);
        }
        if let UserCommand::HostDeciding = cmd {
            self.command_wait = CommandWait::HostDeciding;
            return Ok(Flow::NextPass);
        }
        if let UserCommand::RequestHandover = cmd {
            self.note_handover_request();
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

    /// `ApproveToolUse` or `RejectToolUse` for the call awaiting approval:
    /// an approved call runs (with the calls held for it, in native
    /// mode), a rejected one is answered. A decision for another call is
    /// ignored (a rejection reports the state again), as is any other
    /// command but `CancelTask` and `SuspendTask`.
    async fn decide_tool_approval(
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
    async fn decide_batch_approval(
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
    async fn answer_question(
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
    async fn decide_error_recovery(
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
                let pb =
                    PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
                let generic_prompt = "Please continue with the task based on the previous actions.";
                let next_msg_num = self.task.user_message_count + 1;
                let augmented_generic_prompt = pb
                    .augment_user_message_with_env_details_and_reminder(
                        &self.task,
                        generic_prompt,
                        next_msg_num,
                    )?;
                self.task
                    .conversation
                    .add_user_message(&augmented_generic_prompt);
                self.task.mark_last_message_runtime();
                self.task.increment_user_message_count();
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
        .await?;
        Ok(Flow::Exit)
    }

    /// Runs the agent loop to completion.
    pub async fn run(mut self) -> Result<Task, ArkoAgentsError> {
        self.check_config()?;
        self.start().await?;

        loop {
            // Every turn boundary, pause and wait passes here.
            self.report_task(false).await;
            step!(self.pass_start().await?);

            if self.task.current_state.is_awaiting_approval() {
                step!(self.wait_for_command().await?);
            }
            step!(self.turn_boundary().await?);

            let turn = match self.model_turn().await? {
                ModelTurn::Streamed(turn) => turn,
                ModelTurn::Ended(Flow::Exit) => break,
                ModelTurn::Ended(_) => continue,
            };
            // The model's turn is in the conversation; its calls haven't run.
            self.report_task(false).await;

            if self.is_native_mode() {
                step!(self.native_turn(turn).await?);
            } else {
                step!(self.text_turn(turn).await?);
            }

            if self.task.current_state.is_terminal() {
                break;
            }
        }

        self.finish().await
    }

    /// The end of the run, however the loop stopped: context taken is
    /// kept, the task reported, and `TaskStreamEnd` sent.
    async fn finish(mut self) -> Result<Task, ArkoAgentsError> {
        // Whatever stopped the loop, context it took is not lost.
        self.keep_taken_context()?;
        self.report_task(false).await;
        self.flush_throttled_text().await?;
        self.flush_throttled_reasoning().await?;
        self.emit_event(AgentEvent::TaskStreamEnd)
            .await
            .map_err(|e| {
                log::error!(
                    "Task {}: Failed to send TaskStreamEnd event: {}",
                    self.task.id,
                    e
                );
                e
            })?;

        log::info!(
            "Agent loop finished for task ID: {}. Final state: {:?}",
            self.task.id,
            self.task.current_state
        );
        Ok(self.task)
    }

    /// Answers the native calls of the last turn that have no result yet,
    /// alongside its other waiting results: a call still awaiting approval
    /// is rejected with `reason` (and `ToolRejected`), and a call with no
    /// result at all (a turn an older version left early) is answered as
    /// unreached.
    ///
    /// A call only held behind a call awaiting approval (not listed by the
    /// awaiting state) gets `held_reason` instead.
    async fn answer_open_native_calls(
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
    async fn settle_interrupted_wait(
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
    async fn recover_running_calls(&mut self) -> Result<bool, ArkoAgentsError> {
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
    async fn settle_open_turn(&mut self, recovered: bool) -> Result<(), ArkoAgentsError> {
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

    /// Whether the last `TaskStateChanged` reported the task's current
    /// state.
    fn state_reported(&self) -> bool {
        self.last_reported_state.lock().unwrap().as_ref() == Some(&self.task.current_state)
    }

    async fn emit_event(&self, event: AgentEvent) -> Result<(), ArkoAgentsError> {
        #[cfg(test)]
        self.trace.event(&event);
        if let AgentEvent::TaskStateChanged(state) = &event {
            *self.last_reported_state.lock().unwrap() = Some(state.clone());
        }
        if self.event_sender.send(event.clone()).await.is_err() {
            let err_msg = format!("Failed to send AgentEvent: {:?}", event);
            log::error!("{}", err_msg);
            return Err(ArkoAgentsError::ChannelSend(err_msg));
        }
        Ok(())
    }

    /// Runs one auto-approved native call (see
    /// [`run_native_calls`](Self::run_native_calls)) and returns its result.
    #[cfg(test)]
    async fn execute_native_tool(
        &mut self,
        ai_tool_call: &ai::ToolCall,
        internal_id: Uuid,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        let tool = self
            .tool_registry
            .get(&ai_tool_call.name)
            .ok_or_else(|| ArkoAgentsError::ToolNotFound(ai_tool_call.name.clone()))?;
        let params = Self::tool_call_args_to_map(ai_tool_call);
        let run = NativeRun {
            index: 0,
            call: ai_tool_call.clone(),
            run: self.tool_run(internal_id, &ai_tool_call.name, tool, params, None),
            message_to_agent: None,
        };
        self.run_native_calls_for_result(run, |_| true).await
    }

    /// Runs one native call and takes its result out of the task.
    #[cfg(test)]
    async fn run_native_calls_for_result(
        &mut self,
        run: NativeRun,
        keep_going: fn(&TaskState) -> bool,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        let mut results = NativeResults::new(&self.task);
        self.run_native_calls(vec![run], keep_going, Default::default(), &mut results)
            .await?;
        let recorded = results.positions.len();
        results.finish(&mut self.task);
        Ok(if recorded > 0 {
            self.task.pending_native_tool_results.pop()
        } else {
            None
        })
    }

    /// Runs one approved native call (see
    /// [`run_native_calls`](Self::run_native_calls)) and returns its result.
    #[cfg(test)]
    async fn execute_native_tool_for_approval(
        &mut self,
        ai_call: &ai::ToolCall,
        internal_id: Uuid,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        custom_output: Option<String>,
        message_to_agent: Option<String>,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        match self.approved_native_run(
            0,
            ai_call,
            internal_id,
            tool_name,
            parameters,
            custom_output,
            message_to_agent,
        ) {
            Ok(run) => {
                self.run_native_calls_for_result(run, |state| !state.is_terminal())
                    .await
            }
            Err(()) => self.unavailable_native_tool(ai_call).await.map(Some),
        }
    }

    fn append_error_recovery_user_message(&mut self, message: &str) -> Result<(), ArkoAgentsError> {
        let pb = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
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
            self.append_error_recovery_user_message(
                "Continue the task from where your previous response ended.",
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentLLMConfig, AgentToolConfig};
    use arko_ai::{GenerationConfig, Model};

    pub(super) fn test_config() -> AgentConfig {
        AgentConfig::builder(
            "batch-test".to_string(),
            "Batch Test".to_string(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_luna(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test".to_string())
        .instructions("test".to_string())
        .tool_config(AgentToolConfig::default())
        .build()
        .unwrap()
    }

    /// Tells the task about something once, at its first turn boundary.
    struct TellOnce(std::sync::atomic::AtomicBool);

    impl super::super::TurnContextSource for TellOnce {
        fn turn_context(&self, _task_id: Uuid) -> Option<String> {
            (!self.0.swap(true, std::sync::atomic::Ordering::SeqCst))
                .then(|| "Agent `b` is now running.".to_string())
        }
    }

    /// A turn context source's content reaches the conversation before the
    /// next model request, as a workflow context update, once.
    #[tokio::test]
    async fn turn_context_is_added_before_the_next_request() {
        use futures::StreamExt;
        let (mut agent, _events, _commands) = super::retry_tests::harness();
        let tool_config = &mut agent.task.agent_config.tool_config;
        tool_config.tool_calling_mode = ToolCallingMode::Native;
        tool_config.default_auto_approve_tool_use = true;
        agent.turn_context = Some(Arc::new(TellOnce(Default::default())));
        let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(turn.boxed()));
        let task = timeout(Duration::from_secs(10), agent.run())
            .await
            .unwrap()
            .unwrap();
        let updates = task
            .conversation
            .messages
            .iter()
            .filter(|message| message.role == ai::Role::User)
            .filter_map(|message| message.content.iter().find_map(ai::Content::as_text))
            .filter(|text| {
                text.starts_with("# Workflow context update\n\nAgent `b` is now running.")
            })
            .count();
        assert_eq!(updates, 1);
    }

    /// Holds every request while its flag is set.
    struct HoldWhile(Arc<std::sync::atomic::AtomicBool>);

    impl super::super::TurnContextSource for HoldWhile {
        fn turn_context(&self, _task_id: Uuid) -> Option<String> {
            None
        }

        fn hold_request(&self, _task_id: Uuid) -> Option<String> {
            self.0
                .load(std::sync::atomic::Ordering::SeqCst)
                .then(|| "Budget exceeded.".to_string())
        }
    }

    /// A held request isn't made: the task pauses with the
    /// `BudgetExceeded` cause, whatever its LLM error policy, and
    /// continuing asks again; once released, the request is made without
    /// an error note.
    #[tokio::test]
    async fn a_held_request_pauses_the_task_until_it_is_released() {
        use futures::StreamExt;
        let (mut agent, mut events, commands) = super::retry_tests::harness();
        agent.task.agent_config.llm_error_policy = LLMErrorPolicy::InformAgentAndContinue;
        let tool_config = &mut agent.task.agent_config.tool_config;
        tool_config.tool_calling_mode = ToolCallingMode::Native;
        tool_config.default_auto_approve_tool_use = true;
        let hold = Arc::new(std::sync::atomic::AtomicBool::new(true));
        agent.turn_context = Some(Arc::new(HoldWhile(hold.clone())));
        let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(turn.boxed()));
        let run = tokio::spawn(agent.run());
        let mut pauses = 0;
        while let Some(event) = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
        {
            match event {
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                    cause,
                    ..
                }) => assert_eq!(cause, ErrorRecoveryCause::BudgetExceeded),
                AgentEvent::ErrorEncountered {
                    requires_approval_to_retry: true,
                    ..
                } => {
                    pauses += 1;
                    if pauses == 2 {
                        hold.store(false, std::sync::atomic::Ordering::SeqCst);
                    }
                    commands
                        .send(UserCommand::ContinueAfterError { note: None })
                        .await
                        .unwrap();
                }
                AgentEvent::TaskStreamEnd => break,
                _ => {}
            }
        }
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(pauses, 2);
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert!(!task.monitor_pause);
        let notes = task
            .conversation
            .messages
            .iter()
            .filter_map(|message| message.content.iter().find_map(ai::Content::as_text))
            .filter(|text| text.contains("Budget exceeded") || text.contains("error"))
            .count();
        assert_eq!(notes, 0, "no note reaches the model");
    }

    /// Every context injection queued before a turn boundary reaches the
    /// next request, in one workflow context update, not one per request.
    #[tokio::test]
    async fn every_queued_injection_reaches_the_next_request() {
        use futures::StreamExt;
        let (mut agent, _events, commands) = super::retry_tests::harness();
        let tool_config = &mut agent.task.agent_config.tool_config;
        tool_config.tool_calling_mode = ToolCallingMode::Native;
        tool_config.default_auto_approve_tool_use = true;
        for content in ["First message.", "Second message."] {
            commands
                .try_send(UserCommand::InjectContext {
                    content: content.into(),
                })
                .unwrap();
        }
        let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(turn.boxed()));
        let task = timeout(Duration::from_secs(10), agent.run())
            .await
            .unwrap()
            .unwrap();
        let messages = &task.conversation.messages;
        let update = messages
            .iter()
            .position(|message| {
                message.role == ai::Role::User
                    && message
                        .content
                        .iter()
                        .filter_map(ai::Content::as_text)
                        .any(|text| {
                            text.starts_with(
                                "# Workflow context update\n\nFirst message.\n\nSecond message.",
                            )
                        })
            })
            .expect("one update carries both injections");
        let answer = messages
            .iter()
            .position(|message| message.role == ai::Role::Assistant)
            .unwrap();
        assert!(update < answer, "the update precedes the only request");
    }

    /// Context taken from the channel together with a suspension stays with
    /// the suspended task.
    #[tokio::test]
    async fn an_injection_taken_with_a_suspension_is_kept() {
        let (agent, _events, commands) = super::retry_tests::harness();
        commands
            .try_send(UserCommand::InjectContext {
                content: "Late note.".into(),
            })
            .unwrap();
        commands.try_send(UserCommand::SuspendTask).unwrap();
        let task = timeout(Duration::from_secs(10), agent.run())
            .await
            .unwrap()
            .unwrap();
        let kept = task.conversation.messages.iter().any(|message| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_text)
                .any(|text| text.starts_with("# Workflow context update\n\nLate note."))
        });
        assert!(kept, "{:?}", task.conversation.messages);
        assert!(
            task.conversation
                .messages
                .iter()
                .all(|message| message.role != ai::Role::Assistant),
            "no request was made"
        );
    }

    #[test]
    fn retry_events_saved_before_the_rate_limit_flag_still_load() {
        let event: AgentEvent = serde_json::from_value(serde_json::json!({
            "RetryScheduled": {"attempt": 2, "delay_ms": 1000, "elapsed_ms": 50}
        }))
        .unwrap();
        assert_eq!(
            event,
            AgentEvent::RetryScheduled {
                attempt: 2,
                delay_ms: 1000,
                elapsed_ms: 50,
                rate_limit: false,
            }
        );
    }

    /// Cancels the task at every call.
    struct CancelEveryCall;

    #[async_trait::async_trait]
    impl ToolExecutionMiddleware for CancelEveryCall {
        async fn before(
            &self,
            _context: &ToolExecutionContext<'_>,
            _call: &ToolCallContext<'_>,
        ) -> Result<ToolExecutionDecision, crate::tool::ToolError> {
            Ok(ToolExecutionDecision::CancelTask {
                reason: "Run cancelled.".into(),
            })
        }

        async fn after(
            &self,
            _context: &ToolExecutionContext<'_>,
            _call: &ToolCallContext<'_>,
            _outcome: ToolExecutionOutcome<'_>,
        ) -> Result<(), crate::tool::ToolError> {
            panic!("a cancelled call does not run");
        }
    }

    /// A middleware's cancel decision rejects the call and ends the task as
    /// cancelled, for an auto-approved call and for an approved one.
    #[tokio::test]
    async fn a_middleware_cancels_the_task_at_a_tool_call() {
        use futures::StreamExt;
        for approval in [false, true] {
            let (mut agent, mut events, commands) = super::retry_tests::harness();
            let tool_config = &mut agent.task.agent_config.tool_config;
            tool_config.tool_calling_mode = ToolCallingMode::Native;
            tool_config.default_auto_approve_tool_use = !approval;
            agent.execution_middleware = Some(Arc::new(CancelEveryCall));
            let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
            agent
                .scripted_responses
                .get_mut()
                .unwrap()
                .push_back(Ok(turn.boxed()));
            let runner = tokio::spawn(agent.run());
            let mut received = Vec::new();
            while let Some(event) = timeout(Duration::from_secs(10), events.recv())
                .await
                .unwrap()
            {
                if let AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } = &event {
                    commands
                        .send(UserCommand::ApproveToolUse {
                            tool_call_id: *tool_call_id,
                            custom_tool_output: None,
                            message_to_agent: None,
                        })
                        .await
                        .unwrap();
                }
                received.push(event);
            }
            let task = runner.await.unwrap().unwrap();
            assert_eq!(
                task.current_state,
                TaskState::Cancelled {
                    reason: "Run cancelled.".into()
                }
            );
            assert!(received.iter().any(|event| matches!(
                event,
                AgentEvent::ToolRejected { reason: Some(reason), .. } if reason == "Run cancelled."
            )));
            assert!(
                !received
                    .iter()
                    .any(|event| matches!(event, AgentEvent::ToolResult { .. }))
            );
        }
    }
}

#[cfg(test)]
#[path = "loop_trace.rs"]
mod loop_trace;

#[cfg(test)]
#[path = "golden_tests.rs"]
mod golden_tests;

#[cfg(test)]
#[path = "retry_tests.rs"]
mod retry_tests;

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod recovery_tests;

#[cfg(test)]
#[path = "long_run_tests.rs"]
mod long_run_tests;

#[cfg(test)]
#[path = "refusal_tests.rs"]
mod refusal_tests;

#[cfg(test)]
#[path = "monitor_tests.rs"]
mod monitor_tests;

#[cfg(test)]
#[path = "image_tests.rs"]
mod image_tests;

#[cfg(test)]
#[path = "context_tests.rs"]
mod context_tests;

#[cfg(test)]
#[path = "tool_call_tests.rs"]
mod tool_call_tests;

#[cfg(test)]
#[path = "injection_tests.rs"]
mod injection_tests;

#[cfg(test)]
#[path = "network_tests.rs"]
mod network_tests;

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod parallel_tests;

#[cfg(test)]
#[path = "command_wait_tests.rs"]
mod command_wait_tests;

#[cfg(test)]
#[path = "handover_tests.rs"]
mod handover_tests;

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod snapshot_tests;

#[cfg(test)]
#[path = "result_limit_tests.rs"]
mod result_limit_tests;
