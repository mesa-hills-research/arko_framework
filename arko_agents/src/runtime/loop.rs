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
mod decisions;
mod resume;
mod text;
use decisions::*;
mod wait;
use wait::*;

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
