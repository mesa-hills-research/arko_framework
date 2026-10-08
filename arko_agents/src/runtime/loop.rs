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
mod boundary;
mod start;

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

    /// Reports the task to the snapshot sink, if there is one: a safe point
    /// (see [`TaskSnapshotSink`](crate::snapshot::TaskSnapshotSink)). With
    /// `durable`, returns once the sink has saved it.
    async fn report_task(&mut self, durable: bool) {
        #[cfg(test)]
        self.trace.snapshot(&self.task, durable);
        self.snapshots.report(&self.task, durable).await;
    }

    /// A prompt builder for the runtime's messages to the model (without
    /// the tool definitions, which only the system prompt lists).
    fn prompts(&self) -> PromptBuilder<'_> {
        PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
    }

    fn is_native_mode(&self) -> bool {
        self.task.agent_config.tool_config.tool_calling_mode == ToolCallingMode::Native
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

    /// A pass that ends before its turn's calls (the turn was answered
    /// with a note, or no usable turn came): `Exit` for a task that ended,
    /// `NextPass` otherwise.
    fn end_of_pass(&self) -> Flow {
        if self.task.current_state.is_terminal() {
            Flow::Exit
        } else {
            Flow::NextPass
        }
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
        // A failed send hands the event back, for the error.
        if let Err(error) = self.event_sender.send(event).await {
            let err_msg = format!("Failed to send AgentEvent: {:?}", error.0);
            log::error!("{}", err_msg);
            return Err(ArkoAgentsError::ChannelSend(err_msg));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "loop_tests.rs"]
mod tests;

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
