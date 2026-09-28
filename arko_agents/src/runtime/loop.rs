use super::command::{ToolApprovalDecision, UserCommand};
use super::event::AgentEvent;
use super::tool_run::{self, Interception, RunEnvironment, RunOutcome, ToolRun};
use crate::agent_config::{
    LLMErrorPolicy, PathSafetyPolicy, PlanUsageAction, ToolCallingMode, ToolErrorPolicy,
    UsageLimitPolicy,
};
use crate::error::ArkoAgentsError;
use crate::parsing::{
    ParsedToolCall, PromptBuilder, RawParserEvent, ToolCallGroup, ToolCallParser,
    ToolParsingConfig, assemble_tool_calls,
};
use crate::skill::SkillPromptContext;
use crate::task::{
    CANCELLED_BY_USER, DeferredToolError, PendingToolCallInfo, PlanUsageWindowMark, Task, TaskState,
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
    InterceptorConfig, Tool, ToolDefinition, ToolExecutionMiddleware, ToolImage, ToolRegistry,
    builtins::{ASK_USER_TOOL_NAME, COMPLETE_TASK_TOOL_NAME},
};
use arko_ai::{self as ai};
use futures::TryStreamExt;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep as tokio_sleep, timeout};
use uuid::Uuid;

const OAUTH_REFRESH_MARGIN_SECS: u64 = 5 * 60;

/// A usage-limit wait retries this long after the reported reset, plus up
/// to [`USAGE_LIMIT_RESET_JITTER`].
const USAGE_LIMIT_RESET_MARGIN: Duration = Duration::from_secs(1);
const USAGE_LIMIT_RESET_JITTER: Duration = Duration::from_secs(30);

/// How recovery from a failed model request waits.
enum RecoveryWait {
    /// Bounded by the error policy's attempts and the elapsed budget; a
    /// message about the error is added before the retry.
    Ordinary,
    /// Opt-in indefinite capacity recovery (`capacity_retry`).
    Capacity(crate::agent_config::LLMCapacityRetryConfig),
    /// Default recovery from short rate limits and capacity errors
    /// (`rate_limit_retry`).
    RateLimit(crate::agent_config::LLMRateLimitRetryConfig),
    /// A plan usage limit, handled under `usage_limit_policy`.
    UsageLimit,
}

fn unix_millis_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, duration_millis)
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// `3h 12m`, `12m 5s` or `40s`.
fn describe_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (days, hours, minutes, rest) = (
        seconds / 86_400,
        seconds % 86_400 / 3_600,
        seconds % 3_600 / 60,
        seconds % 60,
    );
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {rest}s")
    } else {
        format!("{rest}s")
    }
}

/// The runtime's notes on a response it can't use, added to the
/// conversation before it asks again; see
/// [`AgentLoop::handle_model_output_failure`].
const NO_NATIVE_TOOL_CALL: &str = "Your response did not call a tool, so the task is still open. Continue with your tools, or call complete_task with a summary if the task is finished.";
const NO_TEXT_TOOL_CALL: &str = "Your response did not contain a tool call. Use one of your tools in the format the system prompt describes, or complete_task if the task is finished.";

/// `ToolRejected` reasons for calls a stream announced but never finished.
const INTERRUPTED_CALL: &str =
    "The model's response was interrupted before this call finished; it was not run.";
const FAILED_STREAM_CALL: &str =
    "The model's response failed before this call finished; it was not run.";
const REFUSED_TURN_CALL: &str = "turn refused";
const UNDELIVERED_CALL: &str =
    "The model's final response did not include this call; it was not run.";
const UNREACHED_CALL: &str = "The turn ended before this call ran; it was not run.";
/// Why a call awaiting approval was not run, when its wait ended without a
/// decision; see [`AgentLoop::settle_interrupted_wait`]. For a task saved
/// that way by an older version.
const UNDECIDED_CALL: &str =
    "The approval wait ended without a decision, so this call was not run.";
/// When the host's command channel closed during the wait.
const HOST_STOPPED_CALL: &str =
    "The approval wait ended because the host stopped responding, so this call was not run.";
/// Results of calls a tool error under `ToolErrorPolicy::FailTask` stopped.
const TASK_FAILED_CALL: &str =
    "The task failed on an earlier tool error, so this call was not run.";

/// Results of calls a `CancelTask` command stopped. The command carries no
/// reason: it comes from the user or from a host such as a workflow
/// stopping its agents.
const TASK_CANCELLED_CALL: &str = "The task was cancelled, so this call was not run.";
/// Results of calls a `SuspendTask` command left unrun.
const TASK_SUSPENDED_CALL: &str = "The task was suspended before this call ran, so it was not run. Call it again if it is still needed.";
/// Results of calls that were running when `CancelTask` or `SuspendTask`
/// arrived; see [`AgentLoop::run_group`].
const STOPPED_BY_CANCEL: &str = "The task was cancelled while this call was running, so the call was stopped before it finished.";
const STOPPED_BY_SUSPEND: &str = "The task was suspended while this call was running, so the call was stopped before it finished. Call it again if it is still needed.";

/// Why running tool calls were stopped; see [`AgentLoop::run_group`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolsStopped {
    Cancelled,
    Suspended,
}

/// How a group of calls ended; see [`AgentLoop::run_group`].
struct GroupRun {
    /// Each call's outcome, in order; `None` when a stop ended the calls.
    outcomes: Option<Vec<Result<RunOutcome, crate::tool::ToolError>>>,
    /// A cancellation or suspension that arrived while the calls ran.
    stop: Option<ToolsStopped>,
}

/// A native call to run, with where its result goes; see
/// [`AgentLoop::run_native_calls`].
struct NativeRun {
    /// Its position in the turn (or approval batch).
    index: usize,
    call: ai::ToolCall,
    run: ToolRun,
    /// The host's message with its approval, added to its result.
    message_to_agent: Option<String>,
}

/// When `user_command_timeout_seconds` ran out during the wait.
fn approval_timed_out_call(seconds: u64) -> String {
    format!(
        "Human approval timed out after {seconds} seconds, so this call was not run. You can call it again if it is still needed."
    )
}

/// Whether OpenAI encrypted reasoning produced by `from` is replayed to `to`.
/// Codex replays history unchanged when it switches between models of one
/// compaction family and compacts it otherwise (codex-rs
/// `maybe_run_previous_model_inline_compact`; its bundled catalog,
/// models-manager/models.json, gives every GPT-5.6 and GPT-6 model the same
/// `comp_hash`). arko can't compact, so for other pairs the reasoning items
/// are dropped and the reply's messages and calls replayed without them.
fn openai_reasoning_replays(from: &ai::Model, to: &ai::Model) -> bool {
    let family = |model: &ai::Model| {
        model.provider == ai::Provider::OpenAI
            && (model.is_openai_gpt_5_6()
                || model.is_openai_gpt_6_astra()
                || model.is_openai_gpt_6_sol_or_luna())
    };
    from.id == to.id || (family(from) && family(to))
}

/// Drops an assistant message's OpenAI reasoning (for a model that can't read
/// it; see [`openai_reasoning_replays`]). The response's other items then
/// lose their ids too: the Responses API rejects an item with an id (such as
/// `fc_…`) sent without the reasoning item that preceded it ("provided
/// without its required 'reasoning' item"), while ids are optional on input.
/// Codex never hits this, since it compacts rather than strip on such a
/// switch.
fn strip_openai_reasoning(message: &mut ai::Message) {
    let stripped = match &mut message.reasoning_artifact {
        Some(ai::ReasoningArtifact::OpenAIOutputItems { items }) => {
            let before = items.len();
            items.retain(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
            let stripped = items.len() != before;
            if stripped {
                for item in items.iter_mut() {
                    if let Some(item) = item.as_object_mut() {
                        item.remove("id");
                    }
                }
            }
            if items.is_empty() {
                message.reasoning_artifact = None;
            }
            stripped
        }
        Some(ai::ReasoningArtifact::OpenAIEncryptedReasoning { .. }) => {
            message.reasoning_artifact = None;
            true
        }
        _ => false,
    };
    if stripped {
        // Calls and text rebuilt without items carry these ids otherwise.
        message.id = None;
        for call in &mut message.tool_calls {
            call.provider_id = None;
        }
    }
}

/// The text of an output-based turn without its tool-call markup.
fn parsed_plain_text(events: &[RawParserEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            RawParserEvent::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

async fn wait_for_recovery_deadline(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending::<()>().await,
    }
}

/// The workspace check of a call's `is_path` parameters.
enum PathCheck {
    Inside,
    /// This parameter's path is outside the workspace.
    Outside(String),
    /// A path couldn't be judged; the call is rejected with this reason.
    Invalid(String),
}

/// Tool names excluded from result interception because their results
/// carry semantic meaning used directly by the runtime (task state
/// transitions, user-facing questions, etc.).
const INTERCEPTION_EXCLUDED_TOOLS: &[&str] = &[COMPLETE_TASK_TOOL_NAME, ASK_USER_TOOL_NAME];

#[cfg(test)]
type ScriptedStreamResult =
    Result<futures::stream::BoxStream<'static, Result<ai::StreamEvent, ai::Error>>, ai::Error>;

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
    #[cfg(test)]
    scripted_responses: std::sync::Mutex<VecDeque<ScriptedStreamResult>>,
    /// The model of every request made, in order.
    #[cfg(test)]
    requested_models: Arc<std::sync::Mutex<Vec<String>>>,

    skill_prompt_context: Option<SkillPromptContext>,

    interceptor_config: Option<InterceptorConfig>,
    execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    /// Asked for context at each safe turn boundary.
    pub(super) turn_context: Option<Arc<dyn super::TurnContextSource>>,
    tool_context_cursor: usize,
    file_tree_baseline: Option<crate::file_tree::FileTreeSnapshot>,
    file_tree_error: Option<String>,
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
            #[cfg(test)]
            scripted_responses: std::sync::Mutex::new(VecDeque::new()),
            #[cfg(test)]
            requested_models: Arc::default(),
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
            turn_context: None,
            tool_context_cursor: 0,
            file_tree_baseline: None,
            file_tree_error: None,
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
        if self.task.pending_context_injections.is_empty() {
            return Ok(());
        }
        let content = self.task.pending_context_injections.join("\n\n");
        let prompt_builder =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
        let next_message_number = self.task.user_message_count + 1;
        let message = prompt_builder.augment_user_message_with_env_details_and_reminder(
            &self.task,
            &format!("# Workflow context update\n\n{content}"),
            next_message_number,
        )?;
        self.task.conversation.add_user_message(&message);
        self.task.increment_user_message_count();
        self.task.pending_context_injections.clear();
        Ok(())
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

    /// What the task's tool calls see of it; see [`RunEnvironment`].
    fn run_environment(&self) -> Arc<RunEnvironment> {
        Arc::new(RunEnvironment {
            task_id: self.task.id,
            agent_name: self.task.agent_config.name.clone(),
            initial_task_prompt: self.task.initial_task_prompt.clone(),
            working_directory: self.task.current_working_directory.clone(),
            middleware: self.execution_middleware.clone(),
        })
    }

    /// How a tool's results are intercepted: by the configured interceptor,
    /// unless the tool opts out or its results carry meaning the runtime
    /// uses (`complete_task`, `ask_user`). `load_skill` results only when
    /// the skills configuration opts in, with its own failure policy.
    fn interception_for(&self, tool_name: &str, tool: &dyn Tool) -> Option<Interception> {
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

    /// A call to `tool` to run, under the agent's time limit and
    /// parallel-safety settings.
    fn tool_run(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        tool: Arc<dyn Tool>,
        params: HashMap<String, Value>,
        custom_output: Option<String>,
    ) -> ToolRun {
        let config = &self.task.agent_config;
        ToolRun {
            tool_call_id,
            tool_name: tool_name.to_string(),
            timeout: config
                .tool_config
                .tool_timeout(tool_name, tool.uses_host_timeout()),
            parallel_safe: !config.asks_one_tool_call_per_turn()
                && config.is_parallel_safe(&tool.definition()),
            interception: self.interception_for(tool_name, tool.as_ref()),
            interruptible: tool.interruptible(),
            tool,
            params,
            custom_output,
        }
    }

    /// Runs `runs` at the same time while watching the command channel:
    /// context sent meanwhile is kept for the next boundary, and
    /// `CancelTask` or `SuspendTask` stops the calls (dropping them, which
    /// kills a command they started) and is returned. Each call's outcome is
    /// returned in order otherwise; an error is a middleware's own failure.
    /// A group with a call that isn't [interruptible](crate::tool::Tool::interruptible)
    /// runs to its end, and a stop that arrived meanwhile is returned with
    /// the outcomes. A closed channel doesn't stop the calls: the boundary
    /// after them handles it.
    async fn run_group(&mut self, runs: Vec<ToolRun>) -> GroupRun {
        let interruptible = runs.iter().all(|run| run.interruptible);
        let environment = self.run_environment();
        let calls = futures::future::join_all(
            runs.into_iter()
                .map(|run| tool_run::run_tool(environment.clone(), run)),
        );
        tokio::pin!(calls);
        let mut listening = true;
        let mut stop = None;
        loop {
            tokio::select! {
                biased;
                outcomes = &mut calls => {
                    return GroupRun { outcomes: Some(outcomes), stop };
                }
                command = self.command_receiver.recv(), if listening => {
                    let stopped = match command {
                        Some(UserCommand::InjectContext { content }) => {
                            self.queue_context_injection(content);
                            None
                        }
                        Some(UserCommand::CancelTask) => Some(ToolsStopped::Cancelled),
                        Some(UserCommand::SuspendTask) => Some(ToolsStopped::Suspended),
                        Some(other) => {
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
                            return GroupRun { outcomes: None, stop: Some(stopped) };
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

    /// Stops the task for a command that stopped running tool calls: a
    /// cancellation ends it, and a suspension stops the loop after the turn
    /// (the task keeps its state, reported as it stops). Returns the result
    /// text of the stopped calls.
    async fn stop_running_tools(
        &mut self,
        stop: ToolsStopped,
    ) -> Result<&'static str, ArkoAgentsError> {
        match stop {
            ToolsStopped::Cancelled => {
                log::info!("Task {}: Cancelled while tool calls ran.", self.task.id);
                self.task.set_state(TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.to_string(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                Ok(STOPPED_BY_CANCEL)
            }
            ToolsStopped::Suspended => {
                self.stop_requested = true;
                self.suspended_tools = true;
                Ok(STOPPED_BY_SUSPEND)
            }
        }
    }

    /// Applies a stop that arrived while calls that aren't interruptible
    /// ran, once they have returned: a cancellation ends the task unless a
    /// call already ended it, and a suspension stops the loop.
    async fn stop_after_calls(&mut self, stop: ToolsStopped) -> Result<(), ArkoAgentsError> {
        if stop == ToolsStopped::Cancelled && self.task.current_state.is_terminal() {
            return Ok(());
        }
        self.stop_running_tools(stop).await.map(|_| ())
    }

    /// Runs native calls in order under the parallel-safety rule (see
    /// [`AgentToolConfig`](crate::agent_config::AgentToolConfig)): each group
    /// of [`tool_run::groups`] runs at the same time, after the one before.
    /// Once the task can no longer run calls (`keep_going` says so of its
    /// state, or it was suspended), the calls not yet run are answered as not
    /// run. Returns each call's result with its index.
    async fn run_native_calls(
        &mut self,
        runs: Vec<NativeRun>,
        keep_going: fn(&TaskState) -> bool,
    ) -> Result<Vec<(usize, ai::ToolResult)>, ArkoAgentsError> {
        let mut results = Vec::new();
        let mut unreached = Vec::new();
        for group in tool_run::groups(runs, |run| run.run.parallel_safe) {
            if !keep_going(&self.task.current_state) || self.stop_requested {
                unreached.extend(group);
                continue;
            }
            let (calls, group_runs): (Vec<_>, Vec<_>) = group
                .into_iter()
                .map(|native| {
                    let id = native.run.tool_call_id;
                    (
                        (native.index, native.call, id, native.message_to_agent),
                        native.run,
                    )
                })
                .unzip();
            let group = self.run_group(group_runs).await;
            match group.outcomes {
                Some(outcomes) => {
                    for ((index, call, id, message), outcome) in calls.into_iter().zip(outcomes) {
                        let result = self
                            .finish_native_call(&call, id, outcome?, message)
                            .await?;
                        results.push((index, result));
                    }
                    if let Some(stop) = group.stop {
                        self.stop_after_calls(stop).await?;
                    }
                }
                None => {
                    let stop = group.stop.expect("a group without outcomes was stopped");
                    let reason = self.stop_running_tools(stop).await?;
                    for (index, call, id, _) in calls {
                        self.emit_event(AgentEvent::ToolRejected {
                            tool_call_id: id,
                            tool_name: call.name.clone(),
                            reason: Some(reason.to_string()),
                        })
                        .await?;
                        results.push((index, ai::ToolResult::error_from_tool_call(&call, reason)));
                    }
                }
            }
        }
        if !unreached.is_empty() {
            let not_run = self.not_run_after_stop();
            for native in unreached {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: native.run.tool_call_id,
                    tool_name: native.call.name.clone(),
                    reason: Some(not_run.clone()),
                })
                .await?;
                results.push((
                    native.index,
                    ai::ToolResult::error_from_tool_call(&native.call, &not_run),
                ));
            }
        }
        Ok(results)
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
            RunOutcome::Stop { reason, cancel } => {
                self.stop_by_middleware(tool_call_id, tool_name, &reason, cancel)
                    .await?;
                Ok(ai::ToolResult::error_from_tool_call(call, &reason))
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
                let images = output
                    .images
                    .into_iter()
                    .map(|image| ai::ToolResultImage::new(image.description, image.image))
                    .collect();
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
                }
                let text = match message_to_agent {
                    Some(message) => {
                        let mut text = effective;
                        if !text.is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str("Message from user on approval:\n");
                        text.push_str(&message);
                        self.emit_tool_result_events(tool_call_id, tool_name, &text, &text, false)
                            .await?;
                        text
                    }
                    None => {
                        self.emit_tool_result_events(
                            tool_call_id,
                            tool_name,
                            &raw,
                            &effective,
                            intercepted,
                        )
                        .await?;
                        effective
                    }
                };
                if tool_name == COMPLETE_TASK_TOOL_NAME && open {
                    self.task.set_state(TaskState::Completed { summary: raw });
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                }
                Ok(ai::ToolResult::from_tool_call(call, &text).with_images(images))
            }
        }
    }

    /// A middleware stopped the task at a call: it fails, or with `cancel`
    /// is cancelled, unless it already ended (another call of the same
    /// group stopped it first); the call is reported rejected either way.
    async fn stop_by_middleware(
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
            }
        };
        self.stop_at_tool_call(tool_call_id, tool_name, reason, state)
            .await
    }

    fn is_native_mode(&self) -> bool {
        self.task.agent_config.tool_config.tool_calling_mode == ToolCallingMode::Native
    }

    /// Checks a call's `is_path` parameters against the workspace, as
    /// [`crate::utils::is_path_in_workspace`] resolves them. A missing,
    /// null or empty value is not checked; any other non-string value is
    /// invalid, since a tool could read it as a path the check never saw.
    fn check_path_parameters(
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
                        "Path validation failed for parameter '{}': a path must be a string, not {other}.",
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

    /// The tools of a native request: the agent's offered tools, with the
    /// configured tool choice (none, the provider's `auto`, by default: tool
    /// use is optional) and parallel-call setting. One call per turn
    /// ([`AgentConfig::asks_one_tool_call_per_turn`]) disables parallel calls
    /// where the provider can.
    fn build_native_tool_config(
        &self,
        model: &ai::Model,
        configured: Option<&ai::ToolConfig>,
    ) -> ai::ToolConfig {
        let agent_config = &self.task.agent_config;
        let offered = |tool: Arc<dyn crate::tool::Tool>| {
            agent_config
                .offered_tool_definition(tool.definition())
                .to_arko_ai_tool_definition()
        };
        let mut ai_tools = Vec::new();
        for name in &agent_config.tool_config.available_tool_names {
            if let Some(tool) = self.tool_registry.get(name) {
                ai_tools.push(offered(tool));
            }
        }
        if !agent_config
            .tool_config
            .available_tool_names
            .iter()
            .any(|n| n == COMPLETE_TASK_TOOL_NAME)
            && let Some(ct) = self.tool_registry.get(COMPLETE_TASK_TOOL_NAME)
        {
            ai_tools.push(offered(ct));
        }

        let mut config = ai::ToolConfig::new(ai_tools);
        if let Some(configured) = configured {
            config.tool_choice = configured.tool_choice.clone();
            config.parallel_tool_calls = configured.parallel_tool_calls;
        }
        if agent_config.asks_one_tool_call_per_turn() {
            // Gemini can't turn parallel calls off; its calls run one at a
            // time instead.
            config.parallel_tool_calls = model
                .supports_disabling_parallel_tool_calls()
                .then_some(false);
        }
        config
    }

    fn build_generation_config(&self) -> ai::GenerationConfig {
        self.generation_config_for(
            &self.task.conversation.model,
            &self.task.agent_config.llm_config.generation_config,
        )
    }

    fn generation_config_for(
        &self,
        model: &ai::Model,
        generation_config: &ai::GenerationConfig,
    ) -> ai::GenerationConfig {
        let mut config = generation_config.clone();
        if self.is_native_mode() {
            config = config.with_tool_config(
                self.build_native_tool_config(model, generation_config.tool_config.as_ref()),
            );
        }
        config
    }

    /// The conversation and settings of the next request: the task's, or
    /// after a refusal with [`AgentConfig::refusal_fallback`] set, the same
    /// conversation sent to the fallback model.
    fn next_request(&mut self) -> (ai::Conversation, ai::GenerationConfig) {
        let fallback = self
            .task
            .agent_config
            .refusal_fallback
            .clone()
            .filter(|_| self.task.fallback_for.is_some());
        self.current_request_is_fallback = fallback.is_some();
        let mut conversation = self.task.conversation.clone();
        #[cfg(test)]
        self.requested_models.lock().unwrap().push(
            fallback
                .as_ref()
                .map_or(&conversation.model.id, |fallback| &fallback.model.id)
                .clone(),
        );
        match fallback {
            Some(fallback) => {
                let config =
                    self.generation_config_for(&fallback.model, &fallback.generation_config);
                // The configured model's reasoning replays to the fallback
                // under the same rule as the other way around.
                if !openai_reasoning_replays(&conversation.model, &fallback.model) {
                    for message in &mut conversation.messages {
                        if message.role == ai::Role::Assistant {
                            strip_openai_reasoning(message);
                        }
                    }
                }
                conversation.model = fallback.model;
                // The ChatGPT backend's turn-state token routes the configured
                // model's turn; the fallback request starts its own.
                conversation.openai_turn_state = None;
                (conversation, config)
            }
            None => (conversation, self.build_generation_config()),
        }
    }

    fn tool_call_args_to_map(tc: &ai::ToolCall) -> HashMap<String, Value> {
        if let Value::Object(map) = &tc.arguments {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            HashMap::new()
        }
    }

    fn reset_llm_recovery(&mut self) {
        self.llm_interaction_attempts = 0;
        self.recovery_started = None;
        self.capacity_recovery_started = None;
        self.capacity_retry_attempts = 0;
        self.indefinite_interactions = 0;
        self.paused_recovery_elapsed = None;
        self.rate_limit_waited = Duration::ZERO;
    }

    /// Ordinary recovery attempts in this episode: requests made, less those
    /// that ended in a wait that doesn't count against the ordinary limits.
    fn ordinary_attempts(&self) -> u32 {
        self.llm_interaction_attempts
            .saturating_sub(self.indefinite_interactions)
            .max(1)
    }

    /// Stops the ordinary elapsed-recovery clock (and its deadline) while an
    /// indefinite wait lasts; see [`resume_ordinary_recovery_clock`](Self::resume_ordinary_recovery_clock).
    fn pause_ordinary_recovery_clock(&mut self) {
        if let Some(started) = self.recovery_started.take() {
            self.paused_recovery_elapsed = Some(
                self.paused_recovery_elapsed
                    .unwrap_or_default()
                    .saturating_add(started.elapsed()),
            );
        }
    }

    /// Restarts a paused ordinary recovery clock with the time it had
    /// already used.
    fn resume_ordinary_recovery_clock(&mut self) {
        if self.recovery_started.is_none()
            && let Some(used) = self.paused_recovery_elapsed.take()
        {
            let now = tokio::time::Instant::now();
            self.recovery_started = Some(now.checked_sub(used).unwrap_or(now));
        }
    }

    fn recovery_deadline(&self) -> Option<tokio::time::Instant> {
        self.recovery_started.map(|start| {
            start + Duration::from_millis(self.task.agent_config.llm_retry_config.max_elapsed_ms)
        })
    }

    /// Handle control commands at every interruptible model/recovery wait.
    async fn handle_recovery_command(
        &mut self,
        command: Option<UserCommand>,
    ) -> Result<bool, ArkoAgentsError> {
        match command {
            Some(UserCommand::InjectContext { content }) => self.queue_context_injection(content),
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

    /// Refresh and model stream initialization share one attempt and deadline.
    async fn run_llm_attempt<T>(
        &mut self,
        operation: impl std::future::Future<Output = Result<T, ai::Error>>,
    ) -> Result<Option<T>, ArkoAgentsError> {
        self.llm_interaction_attempts = self.llm_interaction_attempts.saturating_add(1);
        let deadline = self.recovery_deadline();
        tokio::pin!(operation);
        loop {
            tokio::select! {
                biased;
                _ = wait_for_recovery_deadline(deadline) => {
                    self.finish_llm_recovery("Elapsed model recovery budget exhausted during request.".into(), None).await?;
                    return Ok(None);
                }
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(None); }
                }
                result = &mut operation => {
                    return match result {
                        Ok(value) => Ok(Some(value)),
                        Err(ai::Error::Refusal(refusal)) => {
                            self.handle_refusal(*refusal, ai::Message::new_assistant(""), None).await?;
                            Ok(None)
                        }
                        Err(error) => { self.handle_llm_error(error, None).await?; Ok(None) }
                    };
                }
            }
        }
    }

    async fn emit_throttled_text(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.text_buffer.is_empty()
            && (self.event_throttle_duration.is_zero()
                || self.last_text_emit_time.elapsed() >= self.event_throttle_duration)
        {
            self.emit_event(AgentEvent::AssistantResponseChunk(self.text_buffer.clone()))
                .await?;
            self.text_buffer.clear();
            self.last_text_emit_time = Instant::now();
        }
        Ok(())
    }

    async fn flush_throttled_text(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.text_buffer.is_empty() {
            self.emit_event(AgentEvent::AssistantResponseChunk(self.text_buffer.clone()))
                .await?;
            self.text_buffer.clear();
        }
        Ok(())
    }

    async fn emit_throttled_reasoning(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.reasoning_buffer.is_empty()
            && (self.event_throttle_duration.is_zero()
                || self.last_reasoning_emit_time.elapsed() >= self.event_throttle_duration)
        {
            self.emit_event(AgentEvent::ReasoningChunk(self.reasoning_buffer.clone()))
                .await?;
            self.reasoning_buffer.clear();
            self.last_reasoning_emit_time = Instant::now();
        }
        Ok(())
    }

    async fn flush_throttled_reasoning(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.reasoning_buffer.is_empty() {
            self.emit_event(AgentEvent::ReasoningChunk(self.reasoning_buffer.clone()))
                .await?;
            self.reasoning_buffer.clear();
        }
        Ok(())
    }

    /// Adds the turn's native results to the conversation as one user
    /// message: the results, then `extra_text` (an answer or a note) and
    /// the environment details and reminder due with this user message, as
    /// text after the results in the same turn.
    fn flush_native_results_to_conversation(
        &mut self,
        extra_text: Option<&str>,
    ) -> Result<(), ArkoAgentsError> {
        if self.task.pending_native_tool_results.is_empty()
            && extra_text.is_none()
            && self.task.pending_tool_images.is_empty()
        {
            return Ok(());
        }
        let text = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
            .text_beside_tool_results(&self.task, extra_text, self.task.user_message_count + 1)?;
        self.task.add_pending_tool_results(text.as_deref());
        Ok(())
    }

    /// Whether an approved native turn's results go into the conversation
    /// now: the task no longer waits, or it waits for the answer to
    /// `ask_user`, which follows them (as when `ask_user` is auto-approved),
    /// so the answer can also come from
    /// [`Task::prepare_for_resumption_with_user_response`].
    fn native_results_are_due(&self) -> bool {
        !self.task.current_state.is_awaiting_approval()
            || matches!(
                self.task.current_state,
                TaskState::AwaitingUserResponse { .. }
            )
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

    fn add_text_tool_results_to_conversation(&mut self, text: &str) {
        let mut content = vec![ai::Content::Text(text.to_string())];
        content.extend(self.take_tool_image_content());
        self.task
            .conversation
            .add_user_message_with_contents(content);
    }

    /// Records model call usage on the task (computing the input delta)
    /// and emits the `ModelCallUsageReported` event.
    async fn record_and_emit_model_call_usage(
        &mut self,
        raw_usage: ai::ModelCallUsage,
    ) -> Result<(), ArkoAgentsError> {
        let usage_with_delta = self.task.record_model_call_usage(raw_usage);
        let usage = usage_with_delta.clone();
        self.emit_event(AgentEvent::ModelCallUsageReported(usage_with_delta))
            .await?;
        if let Some(plan_usage) = usage.plan_usage.clone() {
            self.emit_event(AgentEvent::PlanUsageReported(plan_usage))
                .await?;
        }
        if usage.served_model_changed() {
            let requested_model = usage.requested_model.clone().unwrap_or_else(|| {
                match self.task.agent_config.refusal_fallback.as_ref() {
                    Some(fallback) if self.current_request_is_fallback => fallback.model.id.clone(),
                    _ => self.task.conversation.model.id.clone(),
                }
            });
            self.emit_event(AgentEvent::ModelRerouted {
                provider: usage.provider.clone(),
                requested_model,
                served_model: usage.served_model.clone().unwrap_or(usage.model),
            })
            .await?;
        }
        self.monitor_cache().await
    }

    /// The opt-in cache monitor ([`MonitoringConfig::cache`](crate::agent_config::MonitoringConfig::cache)),
    /// after a call's usage is recorded.
    async fn monitor_cache(&mut self) -> Result<(), ArkoAgentsError> {
        let Some(config) = self.task.agent_config.monitoring.cache.clone() else {
            return Ok(());
        };
        // Calls to the refusal fallback model use another cache: they are
        // not judged, and later calls compare with the configured model's.
        if self.task.fallback_calls.last()
            == Some(&(self.task.usage_history.len().saturating_sub(1)))
        {
            return Ok(());
        }
        let configured: Vec<ai::UsageInfo> = self
            .task
            .usage_history
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.task.fallback_calls.contains(index))
            .map(|(_, usage)| usage.clone())
            .collect();
        let Some(efficiency) = super::monitor::cache_efficiency(&configured, &config) else {
            return Ok(());
        };
        self.emit_event(AgentEvent::CacheEfficiencyReported {
            call_efficiency: efficiency.call,
            rolling_efficiency: efficiency.rolling,
            calls: efficiency.calls,
        })
        .await?;
        if efficiency.calls < config.window_calls {
            return Ok(());
        }
        let low =
            |threshold: Option<f64>| threshold.filter(|threshold| efficiency.rolling < *threshold);
        match low(config.warn_below) {
            Some(threshold) if !self.cache_warned => {
                self.cache_warned = true;
                self.emit_event(AgentEvent::CacheEfficiencyLow {
                    rolling_efficiency: efficiency.rolling,
                    threshold,
                    pausing: false,
                })
                .await?;
            }
            Some(_) => {}
            None => self.cache_warned = false,
        }
        match low(config.pause_below) {
            Some(threshold) if !self.cache_pause_episode => {
                self.cache_pause_episode = true;
                self.cache_pause_pending = Some(format!(
                    "Prompt cache efficiency fell to {:.1}% over the last {} calls (threshold {:.1}%).",
                    efficiency.rolling * 100.0,
                    efficiency.calls,
                    threshold * 100.0
                ));
                self.emit_event(AgentEvent::CacheEfficiencyLow {
                    rolling_efficiency: efficiency.rolling,
                    threshold,
                    pausing: true,
                })
                .await?;
            }
            Some(_) => {}
            None => self.cache_pause_episode = false,
        }
        Ok(())
    }

    /// Emits `ToolResult` with the original output and, when interception
    /// modified the result, a follow-up `ToolResultIntercepted` event.
    async fn emit_tool_result_events(
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

    /// Runs the agent loop to completion.
    pub async fn run(mut self) -> Result<Task, ArkoAgentsError> {
        self.task.agent_config.validate()?;
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

        let interrupted = matches!(self.task.current_state, TaskState::Interrupted { .. });
        if matches!(self.task.current_state, TaskState::Pending)
            || matches!(self.task.current_state, TaskState::Interrupted { .. })
        {
            self.task.set_state(TaskState::InProgress);
            self.task.monitor_pause = false;
        }

        if matches!(self.task.current_state, TaskState::InProgress) {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        if interrupted {
            // Older versions left a native turn without results when an
            // approval wait ended this way.
            self.settle_interrupted_wait(UNDECIDED_CALL).await?;
        }

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

        'main_loop: loop {
            if self.stop_requested {
                if std::mem::take(&mut self.suspended_tools) {
                    // Suspended while tool calls ran, in the state the turn
                    // left it in.
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                }
                break 'main_loop;
            }
            if self.task.current_state.is_terminal() {
                log::info!(
                    "Task {} reached terminal state: {:?}",
                    self.task.id,
                    self.task.current_state
                );
                break 'main_loop;
            }

            if self.task.current_state.is_awaiting_approval() {
                let command_future = self.command_receiver.recv();
                let command_result =
                    if let Some(seconds) = self.task.agent_config.user_command_timeout_seconds {
                        timeout(Duration::from_secs(seconds), command_future).await
                    } else {
                        Ok(command_future.await)
                    };

                match command_result {
                    Ok(Some(cmd)) => {
                        if let UserCommand::InjectContext { ref content } = cmd {
                            self.queue_context_injection(content.clone());
                            continue 'main_loop;
                        }
                        let mut next_user_message_from_command = String::new();
                        let mut next_message_is_runtime = false;
                        let mut native_handled = false;
                        let original_state_before_command_processing =
                            self.task.current_state.clone();

                        match self.task.current_state.clone() {
                            TaskState::AwaitingToolApproval {
                                tool_call_id,
                                tool_name,
                                parameters,
                                tool_description: _,
                            } => match cmd {
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
                                        continue 'main_loop;
                                    }

                                    self.begin_tool_batch();
                                    if self.is_native_mode() {
                                        if let Some(ai_call) = self
                                            .task
                                            .pending_native_tool_calls
                                            .remove(&tool_call_id)
                                        {
                                            let result = self
                                                .execute_native_tool_for_approval(
                                                    &ai_call,
                                                    tool_call_id,
                                                    &tool_name,
                                                    parameters,
                                                    custom_tool_output.clone(),
                                                    message_to_agent.clone(),
                                                )
                                                .await?;
                                            if let Some(r) = result {
                                                self.task.pending_native_tool_results.push(r);
                                            }
                                        }
                                        if matches!(
                                            self.task.current_state,
                                            TaskState::AwaitingToolApproval { .. }
                                        ) {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                        self.end_tool_batch().await?;
                                        if self.native_results_are_due() {
                                            self.flush_native_results_to_conversation(None)?;
                                        }
                                        native_handled = true;
                                    } else {
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
                                            Ok(msg_part) => {
                                                next_user_message_from_command = msg_part
                                            }
                                            Err(e) => {
                                                log::error!(
                                                    "Task {}: Error during approved tool execution: {}",
                                                    self.task.id,
                                                    e
                                                );
                                            }
                                        }
                                        if matches!(self.task.current_state, TaskState::AwaitingToolApproval { tool_call_id: current_id, .. } if current_id == tool_call_id)
                                        {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                        self.end_tool_batch().await?;
                                    }
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
                                        continue 'main_loop;
                                    }
                                    self.emit_event(AgentEvent::ToolRejected {
                                        tool_call_id,
                                        tool_name: tool_name.clone(),
                                        reason: (*reason).clone(),
                                    })
                                    .await?;

                                    if self.is_native_mode() {
                                        if let Some(ai_call) = self
                                            .task
                                            .pending_native_tool_calls
                                            .remove(&tool_call_id)
                                        {
                                            let reason_str =
                                                reason.as_deref().unwrap_or("Rejected by user");
                                            self.task.pending_native_tool_results.push(
                                                ai::ToolResult::error_from_tool_call(
                                                    &ai_call, reason_str,
                                                ),
                                            );
                                        }
                                        self.task.set_state(TaskState::InProgress);
                                        self.flush_native_results_to_conversation(None)?;
                                        native_handled = true;
                                    } else {
                                        let pb = PromptBuilder::new(
                                            &self.task.agent_config,
                                            vec![],
                                            &self.parsing_config,
                                            None,
                                        );
                                        next_user_message_from_command = pb
                                            .format_tool_rejection_as_user_message(
                                                &tool_name,
                                                (*reason).as_deref(),
                                            );
                                        self.task.set_state(TaskState::InProgress);
                                    }
                                }
                                UserCommand::CancelTask => {}
                                UserCommand::SuspendTask => {}
                                _ => {
                                    log::warn!(
                                        "Task {}: Received inappropriate command while AwaitingToolApproval. Ignoring.",
                                        self.task.id
                                    );
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    continue 'main_loop;
                                }
                            },
                            TaskState::AwaitingBatchToolApproval {
                                batch_id,
                                ref pending_calls,
                            } => {
                                if let UserCommand::SubmitBatchToolApprovals {
                                    batch_id: cmd_batch_id,
                                    ref decisions,
                                } = cmd
                                {
                                    if cmd_batch_id != batch_id {
                                        log::warn!(
                                            "Task {}: Received SubmitBatchToolApprovals for unexpected batch_id. Ignoring.",
                                            self.task.id
                                        );
                                        continue 'main_loop;
                                    }

                                    self.begin_tool_batch();
                                    if self.is_native_mode() {
                                        // Rejections are answered at once;
                                        // the approved calls then run like a
                                        // turn's (parallel-safe ones at the
                                        // same time), in the batch's order.
                                        let mut results = Vec::new();
                                        let mut approved = Vec::new();
                                        for (index, pending_call_info) in
                                            pending_calls.iter().enumerate()
                                        {
                                            let Some(ai_call) = self
                                                .task
                                                .pending_native_tool_calls
                                                .remove(&pending_call_info.tool_call_id)
                                            else {
                                                continue;
                                            };
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
                                                    Err(()) => results.push((
                                                        index,
                                                        self.unavailable_native_tool(&ai_call)
                                                            .await?,
                                                    )),
                                                },
                                                Some(ToolApprovalDecision::Reject { reason }) => {
                                                    self.emit_event(AgentEvent::ToolRejected {
                                                        tool_call_id: pending_call_info
                                                            .tool_call_id,
                                                        tool_name: pending_call_info
                                                            .tool_name
                                                            .clone(),
                                                        reason: (*reason).clone(),
                                                    })
                                                    .await?;
                                                    let reason_str = reason
                                                        .as_deref()
                                                        .unwrap_or("Rejected by user");
                                                    results.push((
                                                        index,
                                                        ai::ToolResult::error_from_tool_call(
                                                            &ai_call, reason_str,
                                                        ),
                                                    ));
                                                }
                                                None => results.push((
                                                    index,
                                                    ai::ToolResult::error_from_tool_call(
                                                        &ai_call,
                                                        "No decision provided, implicitly rejected",
                                                    ),
                                                )),
                                            }
                                        }
                                        results.extend(
                                            self.run_native_calls(approved, |state| {
                                                !state.is_terminal()
                                            })
                                            .await?,
                                        );
                                        results.sort_by_key(|(index, _)| *index);
                                        self.task
                                            .pending_native_tool_results
                                            .extend(results.into_iter().map(|(_, result)| result));
                                        if matches!(
                                            self.task.current_state,
                                            TaskState::AwaitingBatchToolApproval { .. }
                                        ) {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                        self.end_tool_batch().await?;
                                        if self.native_results_are_due() {
                                            self.flush_native_results_to_conversation(None)?;
                                        }
                                        native_handled = true;
                                    } else {
                                        let mut combined_results_for_llm = Vec::new();
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
                                                        Ok(msg_part) => {
                                                            combined_results_for_llm.push(msg_part)
                                                        }
                                                        Err(e) => log::error!(
                                                            "Task {}: Error during approved batched tool execution: {}",
                                                            self.task.id,
                                                            e
                                                        ),
                                                    }
                                                }
                                                Some(ToolApprovalDecision::Reject { reason }) => {
                                                    self.emit_event(AgentEvent::ToolRejected {
                                                        tool_call_id: pending_call_info
                                                            .tool_call_id,
                                                        tool_name: pending_call_info
                                                            .tool_name
                                                            .clone(),
                                                        reason: (*reason).clone(),
                                                    })
                                                    .await?;
                                                    let pb = PromptBuilder::new(
                                                        &self.task.agent_config,
                                                        vec![],
                                                        &self.parsing_config,
                                                        None,
                                                    );
                                                    combined_results_for_llm.push(
                                                        pb.format_tool_rejection_as_user_message(
                                                            &pending_call_info.tool_name,
                                                            (*reason).as_deref(),
                                                        ),
                                                    );
                                                }
                                                None => {
                                                    let pb = PromptBuilder::new(
                                                        &self.task.agent_config,
                                                        vec![],
                                                        &self.parsing_config,
                                                        None,
                                                    );
                                                    combined_results_for_llm.push(pb.format_tool_rejection_as_user_message(&pending_call_info.tool_name, Some("Implicitly rejected: No decision provided.")));
                                                }
                                            }
                                            if !matches!(
                                                self.task.current_state,
                                                TaskState::AwaitingBatchToolApproval { .. }
                                            ) || self.stop_requested
                                            {
                                                break;
                                            }
                                        }
                                        next_user_message_from_command = combined_results_for_llm
                                            .into_iter()
                                            .filter(|result| !result.is_empty())
                                            .collect::<Vec<_>>()
                                            .join("\n\n---\n\n");
                                        if matches!(self.task.current_state, TaskState::AwaitingBatchToolApproval { batch_id: current_state_batch_id, .. } if current_state_batch_id == cmd_batch_id)
                                        {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                        self.end_tool_batch().await?;
                                    }
                                } else if matches!(
                                    cmd,
                                    UserCommand::CancelTask | UserCommand::SuspendTask
                                ) {
                                } else {
                                    log::warn!(
                                        "Task {}: Received inappropriate command while AwaitingBatchToolApproval. Ignoring.",
                                        self.task.id
                                    );
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    continue 'main_loop;
                                }
                            }
                            TaskState::AwaitingUserResponse { ref questions } => match cmd {
                                UserCommand::SubmitUserResponse { ref response } => {
                                    log::info!("Task {}: User response received.", self.task.id);
                                    let pb = PromptBuilder::new(
                                        &self.task.agent_config,
                                        vec![],
                                        &self.parsing_config,
                                        None,
                                    );
                                    let formatted = pb
                                        .format_user_response_as_user_message(questions, response);
                                    self.task.set_state(TaskState::InProgress);

                                    if self.is_native_mode() {
                                        self.flush_native_results_to_conversation(Some(
                                            &formatted,
                                        ))?;
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
                                    continue 'main_loop;
                                }
                            },
                            TaskState::AwaitingErrorRecoveryApproval { ref error_message } => {
                                match cmd {
                                    UserCommand::ApproveErrorRecovery {
                                        message: ref user_msg_on_recovery,
                                    } => {
                                        log::info!(
                                            "Task {}: Error recovery approved by user.",
                                            self.task.id
                                        );
                                        if std::mem::take(&mut self.task.monitor_pause) {
                                            // A monitor paused the task, not a
                                            // model error: nothing to report to
                                            // the model beyond the host's note.
                                            match user_msg_on_recovery
                                                .as_deref()
                                                .filter(|text| !text.is_empty())
                                            {
                                                Some(text) => {
                                                    next_user_message_from_command =
                                                        text.to_string()
                                                }
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
                                                self.answer_open_native_calls(UNREACHED_CALL)
                                                    .await?;
                                            }
                                            if self.is_native_mode()
                                                && !self.task.pending_native_tool_results.is_empty()
                                            {
                                                self.flush_native_results_to_conversation(Some(
                                                    &recovery_prompt,
                                                ))?;
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
                                    }
                                    UserCommand::ResolveRefusal { ref changes } => {
                                        match self.task.resolve_refusal(changes.clone()) {
                                            Ok(()) => {
                                                log::info!(
                                                    "Task {}: Refused input changed; continuing.",
                                                    self.task.id
                                                );
                                                self.task.monitor_pause = false;
                                                self.task.fallback_for = None;
                                                self.task.set_state(TaskState::InProgress);
                                                self.reset_llm_recovery();
                                                // Tools observed the old input:
                                                // they see the changed history
                                                // from the start again.
                                                self.tool_context_cursor = usize::MAX;
                                                // The changed input is sent as it is.
                                                native_handled = true;
                                            }
                                            Err(error) => {
                                                self.emit_event(AgentEvent::ErrorEncountered {
                                                    error_message: format!(
                                                        "Could not resolve the refusal: {error}"
                                                    ),
                                                    requires_approval_to_retry: true,
                                                    tool_call_id: None,
                                                })
                                                .await?;
                                                continue 'main_loop;
                                            }
                                        }
                                    }
                                    UserCommand::RejectErrorRecovery { message: _ } => {
                                        log::info!(
                                            "Task {}: Error recovery rejected by user.",
                                            self.task.id
                                        );
                                        // The failed task still records the
                                        // results its last turn holds.
                                        self.settle_interrupted_wait(UNREACHED_CALL).await?;
                                        self.task.set_state(TaskState::Failed {
                                            reason: format!(
                                                "Error recovery rejected by user for error: {}",
                                                error_message
                                            ),
                                        });
                                    }
                                    UserCommand::CancelTask => {}
                                    UserCommand::SuspendTask => {}
                                    _ => {
                                        log::warn!(
                                            "Task {}: Received inappropriate command while AwaitingErrorRecoveryApproval. Ignoring.",
                                            self.task.id
                                        );
                                        self.emit_event(AgentEvent::TaskStateChanged(
                                            self.task.current_state.clone(),
                                        ))
                                        .await?;
                                        continue 'main_loop;
                                    }
                                }
                            }
                            _ => {}
                        }

                        if let UserCommand::CancelTask = cmd {
                            log::info!("Task {}: Cancelled by user command.", self.task.id);
                            // The calls it waited on are answered as not run.
                            self.settle_interrupted_wait(TASK_CANCELLED_CALL).await?;
                            self.task.set_state(TaskState::Cancelled {
                                reason: CANCELLED_BY_USER.to_string(),
                            });
                        }

                        if let UserCommand::SuspendTask = cmd {
                            self.emit_event(AgentEvent::TaskStateChanged(
                                self.task.current_state.clone(),
                            ))
                            .await?;
                            break 'main_loop;
                        }

                        if self.task.current_state != original_state_before_command_processing {
                            self.emit_event(AgentEvent::TaskStateChanged(
                                self.task.current_state.clone(),
                            ))
                            .await?;
                        }

                        if self.task.current_state.is_terminal() {
                            break 'main_loop;
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

                        if matches!(self.task.current_state, TaskState::InProgress)
                            && !native_handled
                        {
                            if !next_user_message_from_command.is_empty() {
                                let pb = PromptBuilder::new(
                                    &self.task.agent_config,
                                    vec![],
                                    &self.parsing_config,
                                    None,
                                );
                                let next_msg_num = self.task.user_message_count + 1;
                                let augmented_msg = pb
                                    .augment_user_message_with_env_details_and_reminder(
                                        &self.task,
                                        &next_user_message_from_command,
                                        next_msg_num,
                                    )?;
                                self.add_text_tool_results_to_conversation(&augmented_msg);
                                if next_message_is_runtime {
                                    self.task.mark_last_message_runtime();
                                }
                                self.task.increment_user_message_count();
                                self.reset_llm_recovery();
                            } else {
                                let pb = PromptBuilder::new(
                                    &self.task.agent_config,
                                    vec![],
                                    &self.parsing_config,
                                    None,
                                );
                                let generic_prompt =
                                    "Please continue with the task based on the previous actions.";
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
                            continue 'main_loop;
                        }
                    }
                    Ok(None) => {
                        let reason =
                            "Command channel disconnected during approval state.".to_string();
                        log::error!("Task {}: {}", self.task.id, reason);
                        self.settle_interrupted_wait(HOST_STOPPED_CALL).await?;
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
                        })
                        .await?;
                        break 'main_loop;
                    }
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
                        log::error!("Task {}: {}", self.task.id, reason);
                        self.settle_interrupted_wait(&approval_timed_out_call(seconds))
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
                        })
                        .await?;
                        break 'main_loop;
                    }
                }
            }
            // A suspension while approved calls ran stops the loop here.
            if self.stop_requested {
                continue 'main_loop;
            }
            // Every command queued by now, also after an approved call ran,
            // so all context sent before this boundary (for example while a
            // tool call ran) reaches the next request together. The task is
            // not terminal here: a command or call that ended it broke out
            // of the loop above.
            if self.take_queued_commands().await? {
                break 'main_loop;
            }

            if !matches!(self.task.current_state, TaskState::InProgress) {
                if self.task.current_state.is_terminal() {
                    break 'main_loop;
                }
                tokio_sleep(Duration::from_millis(100)).await;
                continue 'main_loop;
            }

            // The fallback answers exactly the refused input: context updates
            // and file-tree changes wait for the next request.
            if self.task.fallback_for.is_none() {
                self.flush_context_injections()?;
                self.refresh_file_tree_context();
            }
            self.ensure_user_turn_before_request()?;
            if !self.apply_plan_usage_threshold().await? {
                continue 'main_loop;
            }
            if let Some(message) = self.cache_pause_pending.take() {
                self.finish_monitor_pause(message).await?;
                continue 'main_loop;
            }
            self.observe_committed_tool_context();

            let llm_client = Arc::clone(&self.llm_client);
            let (conversation, gen_config) = self.next_request();
            #[cfg(test)]
            let scripted_response = self.scripted_responses.get_mut().unwrap().pop_front();
            let request = async move {
                #[cfg(test)]
                if let Some(response) = scripted_response {
                    return response;
                }
                if conversation.model.provider == ai::Provider::OpenAI {
                    // Decided under the credential lock, so tasks sharing a
                    // client (or an auth.json) don't rotate twice.
                    llm_client
                        .refresh_openai_oauth_token_if_expiring(Duration::from_secs(
                            OAUTH_REFRESH_MARGIN_SECS,
                        ))
                        .await?;
                }
                llm_client.generate_stream(&conversation, gen_config).await
            };
            let Some(mut response_stream) = self.run_llm_attempt(request).await? else {
                continue 'main_loop;
            };

            let mut raw_events_for_turn: Vec<RawParserEvent> = Vec::new();
            let mut parser = if !self.is_native_mode() {
                Some(ToolCallParser::new(
                    self.task.agent_config.tool_config.offered_tool_names(),
                    self.parsing_config.clone(),
                ))
            } else {
                None
            };

            let mut had_stream_error = false;
            let mut final_generation_response: Option<ai::GenerationResponse> = None;
            let mut current_assistant_response_text_for_conv = String::new();
            let mut native_stream_call_ids: HashMap<usize, (Uuid, String)> = HashMap::new();
            let mut streamed_refusal: Option<ai::Refusal> = None;

            let stream_recovery_deadline = self.recovery_deadline();
            loop {
                tokio::select! {
                    biased;
                    _ = wait_for_recovery_deadline(stream_recovery_deadline) => {
                        if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                            self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                        }
                        self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                        self.finish_llm_recovery("Elapsed model recovery budget exhausted during stream.".into(), None).await?;
                        had_stream_error = true;
                        break;
                    }
                    maybe_cmd = self.command_receiver.recv() => {
                        match maybe_cmd {
                            Some(UserCommand::CancelTask) => {
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                                self.task.set_state(TaskState::Cancelled { reason: CANCELLED_BY_USER.to_string() });
                                self.emit_event(AgentEvent::TaskStateChanged(self.task.current_state.clone())).await?;
                                had_stream_error = true;
                                break;
                            }
                            Some(UserCommand::SuspendTask) => {
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                                self.emit_event(AgentEvent::TaskStateChanged(self.task.current_state.clone())).await?;
                                self.stop_requested = true;
                                had_stream_error = true;
                                break;
                            }
                            Some(UserCommand::InjectContext { content }) => {
                                self.queue_context_injection(content);
                            }
                            Some(_) => {}
                            None => {
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                                self.stop_requested = true;
                                let reason = "Command channel disconnected during LLM streaming.".to_string();
                                self.task.set_state(TaskState::Interrupted { reason: reason.clone() });
                                self.emit_event(AgentEvent::TaskStateChanged(self.task.current_state.clone())).await?;
                                self.emit_event(AgentEvent::ErrorEncountered { error_message: reason, requires_approval_to_retry: false, tool_call_id: None }).await?;
                                had_stream_error = true;
                                break;
                            }
                        }
                    }

                    stream_item = response_stream.try_next() => {
                        match stream_item {
                            Ok(Some(ai::StreamEvent::TextChunk(text_chunk))) => {
                                current_assistant_response_text_for_conv.push_str(&text_chunk);
                                if self.is_native_mode() {
                                    self.text_buffer.push_str(&text_chunk);
                                    self.emit_throttled_text().await?;
                                } else {
                                    raw_events_for_turn.extend(parser.as_mut().unwrap().ingest_chunk(&text_chunk));
                                }
                            }
                            Ok(Some(ai::StreamEvent::ReasoningChunk(reasoning))) => {
                                self.reasoning_buffer.push_str(&reasoning);
                                self.emit_throttled_reasoning().await?;
                            }
                            Ok(Some(ai::StreamEvent::ToolCallStart { index, name })) => {
                                if self.is_native_mode() {
                                    self.flush_throttled_text().await?;
                                    let uuid = Uuid::new_v4();
                                    native_stream_call_ids.insert(index, (uuid, name.clone()));
                                    self.emit_event(AgentEvent::ToolCallInitiated {
                                        tool_call_id: uuid,
                                        tool_name: name,
                                    }).await?;
                                }
                            }
                            Ok(Some(ai::StreamEvent::ToolCallArgumentsDelta { .. })) => {}
                            // Providers repeat the refusal on `End`; it is
                            // kept only for a stream that ends without one.
                            Ok(Some(ai::StreamEvent::Refusal(refusal))) => {
                                streamed_refusal = Some(refusal);
                            }
                            Ok(Some(ai::StreamEvent::End(boxed_generation_response))) => {
                                let mut generation_response = *boxed_generation_response;
                                if self.current_request_is_fallback {
                                    self.prepare_fallback_response(&mut generation_response);
                                }
                                self.task.usage_history.push(generation_response.usage.clone());
                                self.emit_event(AgentEvent::UsageReported(generation_response.usage.clone())).await?;
                                self.record_and_emit_model_call_usage(generation_response.model_call_usage.clone()).await?;
                                if generation_response.stop_reason == ai::StopReason::MaxTokens {
                                    let usage = &generation_response.model_call_usage;
                                    log::warn!("Task {}: the response of {} ended at its output-token limit.", self.task.id, usage.model);
                                    self.emit_event(AgentEvent::OutputTokenLimitReached {
                                        model: usage.served_model.clone().unwrap_or_else(|| usage.model.clone()),
                                        output_tokens: generation_response.usage.output_tokens,
                                    }).await?;
                                }
                                // A refused turn is reported, never retried
                                // unchanged, and never treated as a turn that
                                // failed to call a tool.
                                if let Some(refusal) = generation_response.refusal.clone() {
                                    // The call's routing token still belongs
                                    // to this turn, as for any other response.
                                    if let Some(token) = &generation_response.openai_turn_state {
                                        self.task.conversation.record_openai_turn_state(token);
                                    }
                                    if let Some(ref mut p) = parser {
                                        raw_events_for_turn.extend(p.finalize());
                                    }
                                    let plain_text = parser.is_some().then(|| parsed_plain_text(&raw_events_for_turn));
                                    self.reject_announced_calls(&mut native_stream_call_ids, REFUSED_TURN_CALL).await?;
                                    self.handle_refusal(refusal, generation_response.message, plain_text).await?;
                                    had_stream_error = true;
                                    break;
                                }
                                // A completed turn ends any fallback after a
                                // refusal and answers any refused input.
                                self.task.fallback_for = None;
                                self.task.refused_input = None;
                                self.task.conversation.add_assistant_message_from_response(generation_response.clone());
                                final_generation_response = Some(generation_response);
                                if let Some(ref mut p) = parser {
                                    raw_events_for_turn.extend(p.finalize());
                                }
                                break;
                            }
                            Ok(None) => {
                                if let Some(ref mut p) = parser {
                                    raw_events_for_turn.extend(p.finalize());
                                }
                                if let Some(refusal) = streamed_refusal.take() {
                                    let partial = ai::Message::new_assistant(&current_assistant_response_text_for_conv);
                                    let plain_text = parser.is_some().then(|| parsed_plain_text(&raw_events_for_turn));
                                    self.reject_announced_calls(&mut native_stream_call_ids, REFUSED_TURN_CALL).await?;
                                    self.handle_refusal(refusal, partial, plain_text).await?;
                                    had_stream_error = true;
                                    break;
                                }
                                // Without a final response no announced call runs.
                                self.reject_announced_calls(&mut native_stream_call_ids, FAILED_STREAM_CALL).await?;
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                if current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.handle_llm_error(ai::Error::StreamTruncated("the response stream ended with no output".to_string()), None).await?;
                                    had_stream_error = true;
                                }
                                break;
                            }
                            // A refusal reported as an error (Gemini
                            // generation-blocked codes): its partial text is
                            // not kept.
                            Err(ai::Error::Refusal(refusal)) => {
                                self.reject_announced_calls(&mut native_stream_call_ids, REFUSED_TURN_CALL).await?;
                                self.handle_refusal(*refusal, ai::Message::new_assistant(""), None).await?;
                                had_stream_error = true;
                                break;
                            }
                            // The request is sent again unchanged, so what the
                            // failed stream delivered is not kept: it would end
                            // the retried request on an assistant turn (a
                            // prefill, which Claude 4.6 and later reject).
                            Err(e) => {
                                self.reject_announced_calls(&mut native_stream_call_ids, FAILED_STREAM_CALL).await?;
                                self.handle_llm_error(e, None).await?;
                                had_stream_error = true;
                                break;
                            }
                        }
                    }
                }
            }
            self.flush_throttled_reasoning().await?;
            self.flush_throttled_text().await?;

            if had_stream_error
                || self.task.current_state.is_terminal()
                || !matches!(self.task.current_state, TaskState::InProgress)
            {
                if self.task.current_state.is_terminal() {
                    break 'main_loop;
                }
                continue 'main_loop;
            }

            if self.is_native_mode() {
                let tool_calls = final_generation_response
                    .as_ref()
                    .map(|r| r.message.tool_calls.clone())
                    .unwrap_or_default();

                if tool_calls.is_empty() {
                    log::warn!(
                        "LLM response for task {} did not contain a native tool call.",
                        self.task.id
                    );
                    self.reject_announced_calls(&mut native_stream_call_ids, UNDELIVERED_CALL)
                        .await?;
                    self.handle_model_output_failure(NO_NATIVE_TOOL_CALL.to_string())
                        .await?;
                    if self.task.current_state.is_terminal() {
                        break 'main_loop;
                    }
                    continue 'main_loop;
                }

                self.reset_llm_recovery();
                let overall_batch_id = (tool_calls.len() > 1).then(Uuid::new_v4);
                self.begin_tool_batch();
                // Every call gets exactly one result, in the order the model
                // made them: checks answer some at once, the auto-approved
                // calls run, and the calls that need approval wait for it.
                let mut results: Vec<Option<ai::ToolResult>> = vec![None; tool_calls.len()];
                let mut calls_needing_approval: Vec<(usize, PendingToolCallInfo)> = Vec::new();
                let mut runs: Vec<NativeRun> = Vec::new();
                for (call_index, ai_tool_call) in tool_calls.iter().enumerate() {
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
                        results[call_index] =
                            Some(ai::ToolResult::error_from_tool_call(ai_tool_call, &reason));
                        continue;
                    }

                    let tool_arc = match self.tool_registry.get(&ai_tool_call.name) {
                        Some(t) => t,
                        None => {
                            results[call_index] =
                                Some(self.unavailable_native_tool(ai_tool_call).await?);
                            continue;
                        }
                    };
                    let tool_definition = tool_arc.definition();

                    let mut force_approval = false;
                    let path_rejection = match self
                        .check_path_parameters(&tool_definition, &params_map)
                    {
                        PathCheck::Inside => None,
                        PathCheck::Invalid(reason) => Some(reason),
                        PathCheck::Outside(param_name) => {
                            match self
                                .task
                                .agent_config
                                .get_path_safety_policy(&ai_tool_call.name)
                            {
                                PathSafetyPolicy::AutoReject => Some(format!(
                                    "Path for parameter '{param_name}' is outside the workspace."
                                )),
                                PathSafetyPolicy::RequireApproval => {
                                    force_approval = true;
                                    None
                                }
                                PathSafetyPolicy::AutoApprove => None,
                            }
                        }
                    };
                    // A rejected call gets this one result and is not run.
                    if let Some(reason) = path_rejection {
                        self.emit_event(AgentEvent::ToolRejected {
                            tool_call_id,
                            tool_name: ai_tool_call.name.clone(),
                            reason: Some(reason.clone()),
                        })
                        .await?;
                        results[call_index] =
                            Some(ai::ToolResult::error_from_tool_call(ai_tool_call, &reason));
                        continue;
                    }

                    let normal_approval_needed = !self
                        .task
                        .agent_config
                        .should_auto_approve_tool(&ai_tool_call.name);

                    if force_approval || normal_approval_needed {
                        self.task
                            .pending_native_tool_calls
                            .insert(tool_call_id, ai_tool_call.clone());
                        calls_needing_approval.push((
                            call_index,
                            PendingToolCallInfo {
                                tool_call_id,
                                tool_name: ai_tool_call.name.clone(),
                                parameters: params_map,
                                tool_description: tool_definition.description.clone(),
                            },
                        ));
                    } else {
                        runs.push(NativeRun {
                            index: call_index,
                            call: ai_tool_call.clone(),
                            run: self.tool_run(
                                tool_call_id,
                                &ai_tool_call.name,
                                tool_arc,
                                params_map,
                                None,
                            ),
                            message_to_agent: None,
                        });
                    }
                }
                for (index, result) in self
                    .run_native_calls(runs, |state| matches!(state, TaskState::InProgress))
                    .await?
                {
                    results[index] = Some(result);
                }
                // A turn a call stopped (a tool error that fails the task,
                // `complete_task` or `ask_user`, a cancellation or a
                // suspension) still answers every call once, including the
                // calls awaiting an approval that won't be asked.
                if !matches!(self.task.current_state, TaskState::InProgress) || self.stop_requested
                {
                    let not_run = self.not_run_after_stop();
                    for (index, pending) in calls_needing_approval.drain(..) {
                        if let Some(call) = self
                            .task
                            .pending_native_tool_calls
                            .remove(&pending.tool_call_id)
                        {
                            self.emit_event(AgentEvent::ToolRejected {
                                tool_call_id: pending.tool_call_id,
                                tool_name: pending.tool_name,
                                reason: Some(not_run.clone()),
                            })
                            .await?;
                            results[index] =
                                Some(ai::ToolResult::error_from_tool_call(&call, &not_run));
                        }
                    }
                }
                self.task
                    .pending_native_tool_results
                    .extend(results.into_iter().flatten());
                // Calls announced while streaming that the response doesn't
                // hold (for example cut off at the token limit) are not run.
                self.reject_announced_calls(&mut native_stream_call_ids, UNDELIVERED_CALL)
                    .await?;

                if !calls_needing_approval.is_empty() {
                    let mut calls_needing_approval: Vec<PendingToolCallInfo> =
                        calls_needing_approval
                            .into_iter()
                            .map(|(_, pending)| pending)
                            .collect();
                    if calls_needing_approval.len() == 1 && overall_batch_id.is_none() {
                        let single_pending = calls_needing_approval.remove(0);
                        self.task.set_state(TaskState::AwaitingToolApproval {
                            tool_call_id: single_pending.tool_call_id,
                            tool_name: single_pending.tool_name.clone(),
                            parameters: single_pending.parameters.clone(),
                            tool_description: single_pending.tool_description.clone(),
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
                }
                // Pauses on the turn's tool errors now, or after the calls
                // still awaiting approval are decided.
                self.end_tool_batch().await?;
                if self.native_results_are_due() {
                    self.flush_native_results_to_conversation(None)?;
                }
            } else {
                let mut assistant_text_from_raw_events_for_pass1 = String::new();
                let mut parsed_call_ids = VecDeque::new();

                if !self.simplified_tool_events {
                    let mut current_tool_call_id_granular: Option<Uuid> = None;
                    let mut current_param_name_granular: Option<String> = None;

                    for raw_event in &raw_events_for_turn {
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
                            RawParserEvent::BatchedToolCallStart
                            | RawParserEvent::BatchedToolCallEnd => {}
                        }
                    }
                    if !assistant_text_from_raw_events_for_pass1.is_empty() {
                        self.text_buffer
                            .push_str(&assistant_text_from_raw_events_for_pass1);
                        self.emit_throttled_text().await?;
                    }
                }
                self.flush_throttled_text().await?;

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
                            if !self.task.current_state.is_terminal() {
                                self.handle_model_output_failure(NO_TEXT_TOOL_CALL.to_string())
                                    .await?;
                                if self.task.current_state.is_terminal() {
                                    break 'main_loop;
                                }
                                continue 'main_loop;
                            }
                        } else {
                            let current_group = tool_call_groups.remove(0);
                            let calls_to_process_with_ids: Vec<(ParsedToolCall, Uuid)> =
                                match current_group {
                                    ToolCallGroup::Single(pc) => vec![(
                                        pc,
                                        parsed_call_ids.pop_front().unwrap_or_else(Uuid::new_v4),
                                    )],
                                    ToolCallGroup::Batch(pcs) => {
                                        if let Some(first_tool_name_in_batch) =
                                            pcs.first().map(|call| &call.name)
                                        {
                                            if let Some(tool_def_arc) =
                                                self.tool_registry.get(first_tool_name_in_batch)
                                            {
                                                if !self
                                                    .task
                                                    .agent_config
                                                    .allows_batched_calls_for_tool(
                                                        &tool_def_arc.definition(),
                                                    )
                                                {
                                                    let err_msg = format!(
                                                        "Error: Tool '{}' cannot be used in a batch.",
                                                        first_tool_name_in_batch
                                                    );
                                                    if !next_user_message_str_accumulated.is_empty()
                                                    {
                                                        next_user_message_str_accumulated
                                                            .push_str("\n\n");
                                                    }
                                                    next_user_message_str_accumulated
                                                        .push_str(&err_msg);
                                                    Vec::new()
                                                } else {
                                                    pcs.into_iter()
                                                        .map(|pc| {
                                                            let id = parsed_call_ids
                                                                .pop_front()
                                                                .unwrap_or_else(Uuid::new_v4);
                                                            (pc, id)
                                                        })
                                                        .collect()
                                                }
                                            } else {
                                                pcs.into_iter()
                                                    .map(|pc| {
                                                        let id = parsed_call_ids
                                                            .pop_front()
                                                            .unwrap_or_else(Uuid::new_v4);
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

                            let mut calls_needing_approval = Vec::new();
                            let mut auto_approved_call_results = Vec::new();
                            let overall_batch_id = if calls_to_process_with_ids.len() > 1 {
                                Some(Uuid::new_v4())
                            } else {
                                None
                            };
                            self.begin_tool_batch();

                            for (parsed_tool_call_ref, tool_call_id_ref) in
                                &calls_to_process_with_ids
                            {
                                let tool_call_id = *tool_call_id_ref;

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

                                let tool_arc =
                                    match self.tool_registry.get(&parsed_tool_call_ref.name) {
                                        Some(t) => t,
                                        None => {
                                            self.emit_event(AgentEvent::UnavailableToolAttempted {
                                                tool_name: parsed_tool_call_ref.name.clone(),
                                            })
                                            .await?;
                                            auto_approved_call_results.push(format!(
                                                "Error: Tool '{}' is not available.",
                                                parsed_tool_call_ref.name
                                            ));
                                            continue;
                                        }
                                    };
                                let tool_definition = tool_arc.definition();

                                let mut force_approval = false;
                                let path_rejection = match self.check_path_parameters(
                                    &tool_definition,
                                    &parsed_tool_call_ref.parameters,
                                ) {
                                    PathCheck::Inside => None,
                                    PathCheck::Invalid(reason) => {
                                        Some(crate::tool::ToolError::ExecutionFailed(reason))
                                    }
                                    PathCheck::Outside(param_name) => match self
                                        .task
                                        .agent_config
                                        .get_path_safety_policy(&parsed_tool_call_ref.name)
                                    {
                                        PathSafetyPolicy::AutoReject => {
                                            Some(crate::tool::ToolError::PathAccessRejected {
                                                param_name,
                                                workspace_path: self
                                                    .task
                                                    .current_working_directory
                                                    .to_string_lossy()
                                                    .to_string(),
                                            })
                                        }
                                        PathSafetyPolicy::RequireApproval => {
                                            force_approval = true;
                                            None
                                        }
                                        PathSafetyPolicy::AutoApprove => None,
                                    },
                                };
                                if let Some(tool_error) = path_rejection {
                                    let error_message_for_llm = self
                                        .handle_tool_exec_error(
                                            &parsed_tool_call_ref.name,
                                            tool_error,
                                            Some(tool_call_id),
                                        )
                                        .await?;
                                    auto_approved_call_results.push(error_message_for_llm);
                                    // Under `FailTask` the rest of the batch stops.
                                    if self.task.current_state.is_terminal() {
                                        break;
                                    }
                                    continue;
                                }

                                let normal_approval_needed = !self
                                    .task
                                    .agent_config
                                    .should_auto_approve_tool(&parsed_tool_call_ref.name);

                                if force_approval || normal_approval_needed {
                                    calls_needing_approval.push(PendingToolCallInfo {
                                        tool_call_id,
                                        tool_name: parsed_tool_call_ref.name.clone(),
                                        parameters: parsed_tool_call_ref.parameters.clone(),
                                        tool_description: tool_definition.description.clone(),
                                    });
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
                                        Ok(msg_part) => auto_approved_call_results.push(msg_part),
                                        Err(e) => log::error!(
                                            "Task {}: Error during auto-approved tool execution: {}",
                                            self.task.id,
                                            e
                                        ),
                                    }
                                    if !matches!(self.task.current_state, TaskState::InProgress)
                                        || self.stop_requested
                                    {
                                        break;
                                    }
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
                                    auto_approved_call_results
                                        .push(format!("Tool {}: {not_run}", pending.tool_name));
                                }
                            }
                            if !calls_needing_approval.is_empty()
                                && matches!(self.task.current_state, TaskState::InProgress)
                            {
                                if calls_needing_approval.len() == 1 && overall_batch_id.is_none() {
                                    let single_pending = calls_needing_approval.remove(0);
                                    self.task.set_state(TaskState::AwaitingToolApproval {
                                        tool_call_id: single_pending.tool_call_id,
                                        tool_name: single_pending.tool_name.clone(),
                                        parameters: single_pending.parameters.clone(),
                                        tool_description: single_pending.tool_description.clone(),
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
                                    })
                                    .await?;
                                } else {
                                    let batch_id_for_state =
                                        overall_batch_id.unwrap_or_else(Uuid::new_v4);
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
                                            TaskState::AwaitingBatchToolApproval {
                                                pending_calls,
                                                ..
                                            } => pending_calls.clone(),
                                            _ => vec![],
                                        },
                                    })
                                    .await?;
                                }
                            }
                            // Pauses on the batch's tool errors now, or after
                            // the calls still awaiting approval are decided.
                            self.end_tool_batch().await?;
                            // The results of the calls that ran, which wait
                            // with the task below if it now waits.
                            next_user_message_str_accumulated = auto_approved_call_results
                                .into_iter()
                                .filter(|result| !result.is_empty())
                                .collect::<Vec<_>>()
                                .join("\n\n---\n\n");
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
                        if self.task.current_state.is_terminal() {
                            break 'main_loop;
                        }
                        continue 'main_loop;
                    }
                }

                if final_generation_response.is_some()
                    && !had_stream_error
                    && !self.task.current_state.is_awaiting_approval()
                {
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
                    let pb = PromptBuilder::new(
                        &self.task.agent_config,
                        vec![],
                        &self.parsing_config,
                        None,
                    );
                    let next_msg_num = self.task.user_message_count + 1;
                    let augmented_user_message = pb
                        .augment_user_message_with_env_details_and_reminder(
                            &self.task,
                            &next_user_message_str_accumulated,
                            next_msg_num,
                        )?;
                    self.add_text_tool_results_to_conversation(&augmented_user_message);
                    self.task.increment_user_message_count();
                } else if next_user_message_str_accumulated.is_empty()
                    && !self.task.current_state.is_terminal()
                    && !self.task.current_state.is_awaiting_approval()
                    && final_generation_response.is_some()
                {
                    let pb = PromptBuilder::new(
                        &self.task.agent_config,
                        vec![],
                        &self.parsing_config,
                        None,
                    );
                    let generic_retry_prompt = "Please continue with the task or use the 'complete_task' tool if finished. Ensure you use a tool.";
                    let next_msg_num = self.task.user_message_count + 1;
                    let augmented_generic_prompt = pb
                        .augment_user_message_with_env_details_and_reminder(
                            &self.task,
                            generic_retry_prompt,
                            next_msg_num,
                        )?;
                    self.task
                        .conversation
                        .add_user_message(&augmented_generic_prompt);
                    self.task.mark_last_message_runtime();
                    self.task.increment_user_message_count();
                }
            }

            if self.task.current_state.is_terminal() {
                break 'main_loop;
            }
        }

        // Whatever stopped the loop, context it took is not lost.
        self.keep_taken_context()?;
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

    /// Reports calls a stream announced (`ToolCallInitiated`) that will not
    /// run because the turn ended without them, so every announced call gets
    /// an outcome.
    async fn reject_announced_calls(
        &self,
        announced: &mut HashMap<usize, (Uuid, String)>,
        reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        let mut calls: Vec<_> = announced.drain().collect();
        calls.sort_by_key(|(index, _)| *index);
        for (_, (tool_call_id, tool_name)) in calls {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name,
                reason: Some(reason.to_string()),
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
    fn not_run_after_stop(&self) -> String {
        if self.stop_requested && !self.task.current_state.is_terminal() {
            return TASK_SUSPENDED_CALL.to_string();
        }
        match &self.task.current_state {
            TaskState::Failed { .. } if self.failed_on_tool_error => TASK_FAILED_CALL.to_string(),
            TaskState::Cancelled { reason } if reason == CANCELLED_BY_USER => {
                TASK_CANCELLED_CALL.to_string()
            }
            TaskState::Failed { reason } | TaskState::Cancelled { reason } => {
                format!("This call was not run: {}.", reason.trim_end_matches('.'))
            }
            _ => UNREACHED_CALL.to_string(),
        }
    }

    /// Starts running a batch of tool calls (a turn's, or an approval's):
    /// under `ToolErrorPolicy::PauseAndAwaitApproval` a failed call no longer
    /// pauses the task at once, so the batch's other calls run first, and
    /// [`end_tool_batch`](Self::end_tool_batch) pauses once.
    fn begin_tool_batch(&mut self) {
        self.deferring_tool_errors = true;
    }

    /// Ends a batch begun with [`begin_tool_batch`](Self::begin_tool_batch):
    /// pauses once on its tool errors, with any a batch of the same turn
    /// left while calls awaited approval. While calls of the turn still
    /// await approval the pause waits for their decisions; a task that
    /// stopped doesn't pause.
    async fn end_tool_batch(&mut self) -> Result<(), ArkoAgentsError> {
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
        self.task
            .set_state(TaskState::AwaitingErrorRecoveryApproval {
                error_message: error_message.clone(),
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
        })
        .await
    }

    /// Answers the native calls of the last turn that have no result yet,
    /// alongside its other waiting results: a call still awaiting approval
    /// is rejected with `reason` (and `ToolRejected`), and a call with no
    /// result at all (a turn an older version left early) is answered as
    /// unreached.
    async fn answer_open_native_calls(&mut self, reason: &str) -> Result<(), ArkoAgentsError> {
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
    async fn settle_interrupted_wait(&mut self, reason: &str) -> Result<(), ArkoAgentsError> {
        self.task.deferred_tool_errors.clear();
        if self.is_native_mode() {
            self.answer_open_native_calls(reason).await?;
            self.flush_native_results_to_conversation(None)?;
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

    async fn emit_event(&self, event: AgentEvent) -> Result<(), ArkoAgentsError> {
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
        Ok(self
            .run_native_calls(vec![run], |_| true)
            .await?
            .pop()
            .map(|(_, result)| result))
    }

    /// The error result of a native call to a tool the task is not offered
    /// (or that isn't registered), reported with `UnavailableToolAttempted`.
    async fn unavailable_native_tool(
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

    /// Runs one approved native call (see
    /// [`run_native_calls`](Self::run_native_calls)) and returns its result.
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
            Ok(run) => Ok(self
                .run_native_calls(vec![run], |state| !state.is_terminal())
                .await?
                .pop()
                .map(|(_, result)| result)),
            Err(()) => self.unavailable_native_tool(ai_call).await.map(Some),
        }
    }

    /// An approved native call to run, or `Err` when this run doesn't offer
    /// its tool (a call saved awaiting approval), which is answered like any
    /// unavailable tool, even with caller output.
    #[allow(clippy::too_many_arguments, clippy::result_unit_err)]
    fn approved_native_run(
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

    async fn finish_llm_recovery(
        &mut self,
        message: String,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        // A fallback request that ends here doesn't carry over: the refusal
        // it was to answer stands, and approving recovery resends to the
        // configured model.
        let message = match self.task.fallback_for.take() {
            Some(refusal) if self.current_request_is_fallback => format!(
                "{}. The fallback model could not answer instead: {message}",
                ai::Error::Refusal(Box::new(refusal))
            ),
            _ => message,
        };
        self.task.monitor_pause = false;
        let pause = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::PauseAndAwaitApproval
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if pause {
            self.task
                .set_state(TaskState::AwaitingErrorRecoveryApproval {
                    error_message: message.clone(),
                });
        } else {
            // Exhaustion must return control to the host rather than starting
            // an unbounded second retry cycle under RetryThenInformAgent.
            self.task.set_state(TaskState::Interrupted {
                reason: message.clone(),
            });
            self.stop_requested = true;
        }
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: pause,
            tool_call_id,
        })
        .await
    }

    /// Reports a refusal and stops recovery: the same input is expected to be
    /// refused again, so it is never retried unchanged. With the opt-in
    /// [`AgentConfig::refusal_fallback`], a refusal from the configured model
    /// instead sends the same input to the fallback model once; if that
    /// request fails, the refusal stands (see
    /// [`handle_llm_error`](Self::handle_llm_error)).
    ///
    /// A model's own refusal ([`ai::RefusalSource::keeps_turn`]) stays in
    /// history so the model can see what it declined, with the refusal
    /// explanation as its text when the turn has none (not when the fallback
    /// answers instead). Its tool calls are not run, so they are not kept
    /// either (a call without a result would invalidate the history; OpenAI
    /// output items stored with the turn replay only the calls it still
    /// holds), and a turn with no text at all is not kept. In output-based
    /// tool mode the turn's text holds tool-call markup that was never run:
    /// `plain_text` is the text without it, and only the refusal explanation
    /// (or else that plain text) is kept. Classifier and content-filter
    /// refusals are removed from history.
    ///
    /// The refused input is recorded in [`Task::refused_input`] and reported
    /// with `AgentEvent::RefusedInput`, for `UserCommand::ResolveRefusal`.
    async fn handle_refusal(
        &mut self,
        refusal: ai::Refusal,
        mut message: ai::Message,
        plain_text: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        self.flush_throttled_reasoning().await?;
        self.flush_throttled_text().await?;
        let from_fallback = std::mem::take(&mut self.current_request_is_fallback);
        self.task.fallback_for = None;
        let fallback = self
            .task
            .agent_config
            .refusal_fallback
            .clone()
            .filter(|_| !from_fallback);
        let input = self.task.current_input();
        if refusal.source.keeps_turn() && fallback.is_none() {
            message.tool_calls.clear();
            let has_text = |message: &ai::Message| {
                message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .any(|text| !text.trim().is_empty())
            };
            if let Some(plain_text) = plain_text {
                let text = refusal
                    .explanation
                    .clone()
                    .filter(|explanation| !explanation.trim().is_empty())
                    .unwrap_or(plain_text);
                message
                    .content
                    .retain(|content| content.as_text().is_none());
                if !text.trim().is_empty() {
                    message.content.insert(0, ai::Content::Text(text));
                }
                // OpenAI message items would replay the original text,
                // markup included; the message's own text replaces them.
                if let Some(ai::ReasoningArtifact::OpenAIOutputItems { items }) =
                    &mut message.reasoning_artifact
                {
                    items
                        .retain(|item| item.get("type").and_then(Value::as_str) != Some("message"));
                }
            } else if !has_text(&message)
                && let Some(explanation) = refusal.explanation.as_deref()
            {
                message
                    .content
                    .retain(|content| content.as_text().is_none());
                message
                    .content
                    .insert(0, ai::Content::Text(explanation.to_string()));
            }
            if has_text(&message) {
                self.task.conversation.messages.push(message);
            }
        }
        // Recorded before any fallback, so a failed fallback leaves the
        // refusal resolvable.
        self.task.refused_input = input.clone();
        self.emit_event(AgentEvent::RefusalReceived(refusal.clone()))
            .await?;
        if let Some(input) = input {
            self.emit_event(AgentEvent::RefusedInput(input)).await?;
        }
        if let Some(fallback) = fallback {
            self.emit_event(AgentEvent::RefusalFallback {
                model_id: fallback.model.id,
            })
            .await?;
            self.task.fallback_for = Some(refusal);
            return Ok(());
        }
        self.handle_llm_error(ai::Error::Refusal(Box::new(refusal)), None)
            .await
    }

    /// A fallback model's response joins the configured model's
    /// conversation: its ChatGPT turn-state token is not kept (it routed the
    /// fallback's request, not the configured model's turn), OpenAI
    /// reasoning items are dropped unless the configured model can read them
    /// (see [`openai_reasoning_replays`]), and the cache monitor skips the
    /// call.
    fn prepare_fallback_response(&mut self, response: &mut ai::GenerationResponse) {
        response.openai_turn_state = None;
        self.task.fallback_calls.push(self.task.usage_history.len());
        let Some(fallback) = &self.task.agent_config.refusal_fallback else {
            return;
        };
        if !openai_reasoning_replays(&fallback.model, &self.task.conversation.model) {
            strip_openai_reasoning(&mut response.message);
        }
    }

    /// How recovery from `error` waits.
    fn recovery_wait(&self, error: &ai::Error) -> RecoveryWait {
        if error.is_usage_limit() {
            return RecoveryWait::UsageLimit;
        }
        let retrying = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::RetryThenInformAgent { .. }
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if !retrying {
            return RecoveryWait::Ordinary;
        }
        let config = &self.task.agent_config.llm_retry_config;
        if error.is_capacity_error()
            && let Some(capacity) = &config.capacity_retry
        {
            return RecoveryWait::Capacity(capacity.clone());
        }
        if let Some(rate_limit) = &config.rate_limit_retry
            && (error.is_rate_limit() || error.is_capacity_error())
        {
            let max_wait = Duration::from_millis(rate_limit.max_wait_ms);
            // A rate limit asking for a longer wait, or rate limits that have
            // kept the task waiting this long in all (with or without
            // delays), are treated as a usage limit.
            let too_long = error.retry_after().is_some_and(|hint| hint > max_wait)
                || (error.is_rate_limit() && self.rate_limit_waited >= max_wait);
            if !too_long {
                return RecoveryWait::RateLimit(rate_limit.clone());
            }
            // A rate limit that resets this far off is a usage limit in all
            // but name.
            if error.is_rate_limit() {
                return RecoveryWait::UsageLimit;
            }
        }
        RecoveryWait::Ordinary
    }

    /// When a usage limit is waited out under
    /// [`UsageLimitPolicy::WaitUntilReset`], the delay before the retry.
    fn usage_limit_wait(&self, error: &ai::Error) -> Option<Duration> {
        let UsageLimitPolicy::WaitUntilReset { max_wait_ms } =
            self.task.agent_config.llm_retry_config.usage_limit_policy
        else {
            return None;
        };
        let reset_in = error.retry_after()?;
        // The whole wait, margin and jitter included, must fit max_wait.
        let longest = reset_in
            .saturating_add(USAGE_LIMIT_RESET_MARGIN)
            .saturating_add(USAGE_LIMIT_RESET_JITTER);
        if longest > Duration::from_millis(max_wait_ms) {
            return None;
        }
        // A little past the reset, spread so that agents sharing a plan do
        // not all retry at once.
        let jitter = if self.task.agent_config.llm_retry_config.jitter {
            ai::retry::random_sample() % (USAGE_LIMIT_RESET_JITTER.as_millis() as u64 + 1)
        } else {
            0
        };
        Some(
            reset_in
                .saturating_add(USAGE_LIMIT_RESET_MARGIN)
                .saturating_add(Duration::from_millis(jitter)),
        )
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

    /// Recovers from a failed model request (a connection failure, a server
    /// error, a truncated stream, a refusal, ...): under the error policy, the
    /// request is sent again unchanged, without a note in the conversation,
    /// since the model had no part in the failure, or recovery ends.
    async fn handle_llm_error(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        self.recover(error, tool_call_id, None).await
    }

    /// Recovers from a response the runtime can't use (no tool call, or an
    /// output-based call it can't read) under the error policy, like a
    /// failed request, but with `note` added to the conversation before the
    /// next request, so the model can do better. It isn't an outage: the
    /// request is made again at once, without backoff, and the elapsed
    /// recovery budget neither starts nor applies; the attempt limit bounds
    /// it.
    async fn handle_model_output_failure(&mut self, note: String) -> Result<(), ArkoAgentsError> {
        self.recover(ai::Error::ApiError(note.clone()), None, Some(note))
            .await
    }

    async fn recover(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
        note: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        if self.task.fallback_for.is_some() && self.current_request_is_fallback {
            // The fallback model couldn't answer the refused turn: the
            // refusal stands (finish_llm_recovery names it), and the
            // fallback's attempts don't count against the configured
            // model's recovery.
            self.reset_llm_recovery();
            return self
                .finish_llm_recovery(error.to_string(), tool_call_id)
                .await;
        }
        let mut config = self.task.agent_config.llm_retry_config.clone();
        let policy = self.task.agent_config.llm_error_policy.clone();
        let wait = self.recovery_wait(&error);
        if let RecoveryWait::UsageLimit = wait {
            return self.handle_usage_limit(error, tool_call_id).await;
        }
        let indefinite = !matches!(wait, RecoveryWait::Ordinary);
        let model_output = note.is_some();
        let (attempt, started) = if indefinite {
            // These waits neither consume nor reset ordinary recovery: its
            // elapsed clock stops while they last, and the requests they
            // make are not counted as its attempts.
            self.pause_ordinary_recovery_clock();
            self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
            let started = *self
                .capacity_recovery_started
                .get_or_insert_with(tokio::time::Instant::now);
            self.capacity_retry_attempts = self.capacity_retry_attempts.saturating_add(1);
            (self.capacity_retry_attempts, started)
        } else {
            self.resume_ordinary_recovery_clock();
            let started = if model_output {
                self.recovery_started
                    .unwrap_or_else(tokio::time::Instant::now)
            } else {
                *self
                    .recovery_started
                    .get_or_insert_with(tokio::time::Instant::now)
            };
            (self.ordinary_attempts(), started)
        };
        match &wait {
            RecoveryWait::Capacity(capacity) => {
                config.initial_delay_ms = capacity.initial_delay_ms;
                config.max_delay_ms = capacity.max_delay_ms;
            }
            RecoveryWait::RateLimit(rate_limit) => {
                config.initial_delay_ms = rate_limit.initial_delay_ms;
                config.max_delay_ms = rate_limit.max_delay_ms;
            }
            RecoveryWait::Ordinary | RecoveryWait::UsageLimit => {}
        }
        let elapsed = started.elapsed();
        let max_attempts = match policy {
            LLMErrorPolicy::RetryThenInformAgent { max_attempts }
            | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts } => {
                max_attempts.max(1)
            }
            LLMErrorPolicy::InformAgentAndContinue => u32::MAX,
            LLMErrorPolicy::PauseAndAwaitApproval => 1,
        };
        let message = match &wait {
            RecoveryWait::Capacity(_) => format!(
                "LLM capacity error (attempt {attempt}; waiting until service recovers): {error}"
            ),
            RecoveryWait::RateLimit(_) => {
                let kind = if error.is_capacity_error() {
                    "capacity error"
                } else {
                    "rate limit"
                };
                format!(
                    "LLM {kind} (attempt {attempt}; retrying when the provider allows): {error}"
                )
            }
            RecoveryWait::Ordinary | RecoveryWait::UsageLimit => match &note {
                Some(note) => {
                    format!("Model response not usable (attempt {attempt}/{max_attempts}): {note}")
                }
                None => format!("LLM error (attempt {attempt}/{max_attempts}): {error}"),
            },
        };
        let stop_reason = if !error.is_retryable() {
            Some("Failure requires a request, credential, or configuration change")
        } else if !indefinite && attempt >= max_attempts {
            Some("Retry attempt limit exhausted")
        } else {
            None
        };
        if let Some(reason) = stop_reason {
            return self
                .finish_llm_recovery(format!("{reason}. {message}"), tool_call_id)
                .await;
        }
        // A rate limit's own delay is followed as given (plus jitter) rather
        // than grown by backoff; backoff applies when there is none.
        let backoff_attempt = match wait {
            RecoveryWait::RateLimit(_) if error.retry_after().is_some() => 1,
            _ => attempt,
        };
        let delay = if model_output {
            Duration::ZERO
        } else {
            super::retry::retry_delay(
                &config,
                backoff_attempt,
                error.retry_after(),
                ai::retry::random_sample(),
            )
        };
        if !indefinite
            && !model_output
            && elapsed.saturating_add(delay) >= Duration::from_millis(config.max_elapsed_ms)
        {
            return self
                .finish_llm_recovery(
                    format!("Next retry cannot fit the elapsed recovery budget. {message}"),
                    tool_call_id,
                )
                .await;
        }
        let Some(retry_at) = tokio::time::Instant::now().checked_add(delay) else {
            return self
                .finish_llm_recovery(
                    format!("Server retry delay cannot be represented safely. {message}"),
                    tool_call_id,
                )
                .await;
        };
        let counts_toward_budget =
            matches!(wait, RecoveryWait::RateLimit(_)) && error.is_rate_limit();
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: false,
            tool_call_id,
        })
        .await?;
        self.emit_event(AgentEvent::RetryScheduled {
            attempt,
            delay_ms: delay.as_millis() as u64,
            elapsed_ms: elapsed.as_millis() as u64,
            rate_limit: error.is_rate_limit(),
        })
        .await?;
        if let Some(note) = &note {
            self.append_error_recovery_user_message(note)?;
        }
        let waiting_since = tokio::time::Instant::now();
        let waited = self.wait_before_retry(retry_at, tool_call_id).await;
        if counts_toward_budget {
            // The time actually waited: a command may have cut it short.
            self.rate_limit_waited = self
                .rate_limit_waited
                .saturating_add(waiting_since.elapsed());
        }
        waited
    }

    /// Handles a plan usage limit under
    /// [`LLMRetryConfig::usage_limit_policy`](crate::agent_config::LLMRetryConfig::usage_limit_policy):
    /// waits for the reset and retries unchanged (opt-in), or stops
    /// retrying.
    async fn handle_usage_limit(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let now_ms = unix_millis_now();
        let reset_in = error.retry_after();
        let resets_at_unix_ms = reset_in.map(|delay| now_ms.saturating_add(duration_millis(delay)));
        let when = match reset_in {
            Some(delay) => format!("resets in about {}", describe_duration(delay)),
            None => "reset time not reported".to_string(),
        };
        let retry_at = self.usage_limit_wait(&error).and_then(|delay| {
            tokio::time::Instant::now()
                .checked_add(delay)
                .map(|at| (delay, at))
        });
        let Some((delay, retry_at)) = retry_at else {
            self.emit_event(AgentEvent::UsageLimitReached {
                resets_at_unix_ms,
                waiting_until_unix_ms: None,
                error_message: error.to_string(),
            })
            .await?;
            return self
                .finish_llm_recovery(
                    format!("Usage limit reached ({when}). {error}"),
                    tool_call_id,
                )
                .await;
        };
        // Like the other unchanged retries, the wait doesn't count against
        // ordinary recovery.
        self.pause_ordinary_recovery_clock();
        self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
        self.emit_event(AgentEvent::UsageLimitReached {
            resets_at_unix_ms,
            waiting_until_unix_ms: Some(now_ms.saturating_add(duration_millis(delay))),
            error_message: error.to_string(),
        })
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: format!(
                "Usage limit reached ({when}); waiting {} before retrying. {error}",
                describe_duration(delay)
            ),
            requires_approval_to_retry: false,
            tool_call_id,
        })
        .await?;
        self.wait_before_retry(retry_at, tool_call_id).await
    }

    /// Waits until `retry_at` before a retry, handling commands; the task is
    /// then in progress again. An ordinary recovery deadline that passes
    /// first ends recovery.
    async fn wait_before_retry(
        &mut self,
        retry_at: tokio::time::Instant,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let deadline = self.recovery_deadline();
        let sleep = tokio::time::sleep_until(retry_at);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                _ = wait_for_recovery_deadline(deadline) => {
                    return self.finish_llm_recovery("Elapsed model recovery budget exhausted during backoff.".into(), tool_call_id).await;
                }
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(()); }
                }
                _ = &mut sleep => {
                    self.task.set_state(TaskState::InProgress);
                    return Ok(());
                }
            }
        }
    }

    /// Pauses for a monitor (like exhausted recovery), marking the pause so
    /// that approving it adds no error note to the conversation.
    async fn finish_monitor_pause(&mut self, message: String) -> Result<(), ArkoAgentsError> {
        // A fallback still to be made survives the pause.
        let fallback_for = self.task.fallback_for.clone();
        self.finish_llm_recovery(message, None).await?;
        self.task.fallback_for = fallback_for;
        self.task.monitor_pause = matches!(
            self.task.current_state,
            TaskState::AwaitingErrorRecoveryApproval { .. }
        );
        Ok(())
    }

    /// Before a model request: applies the opt-in plan usage threshold
    /// ([`MonitoringConfig::plan_usage`](crate::agent_config::MonitoringConfig::plan_usage))
    /// to the most recent report. Returns whether to make the request.
    async fn apply_plan_usage_threshold(&mut self) -> Result<bool, ArkoAgentsError> {
        let Some(threshold) = self.task.agent_config.monitoring.plan_usage.clone() else {
            return Ok(true);
        };
        let Some(usage) = self
            .task
            .model_call_usage_history
            .last()
            .and_then(|call| call.plan_usage.clone())
        else {
            return Ok(true);
        };
        let Some((window, reached)) = usage
            .most_used_window()
            .filter(|(_, window)| window.used_percent >= threshold.percent)
        else {
            return Ok(true);
        };
        // A report whose window has reset since (a task resumed later) is
        // no longer current.
        let now_seconds = unix_millis_now() / 1000;
        if reached
            .resets_at
            .is_some_and(|resets_at| resets_at <= now_seconds)
        {
            return Ok(true);
        }
        let mark = PlanUsageWindowMark {
            window: window.to_string(),
            resets_at: reached.resets_at,
        };
        let first = self.task.plan_usage_threshold_reported.as_ref() != Some(&mark);
        if first {
            self.task.plan_usage_threshold_reported = Some(mark);
            self.emit_event(AgentEvent::PlanUsageThresholdReached {
                window: window.to_string(),
                used_percent: reached.used_percent,
                threshold_percent: threshold.percent,
                resets_at_unix_ms: reached
                    .resets_at
                    .map(|seconds| seconds.saturating_mul(1000)),
                action: threshold.action.clone(),
            })
            .await?;
        }
        match threshold.action {
            PlanUsageAction::Pause if first => {
                let message = format!(
                    "Plan usage reached {:.1}% of the {window} usage-limit window (threshold {:.1}%).",
                    reached.used_percent, threshold.percent
                );
                self.finish_monitor_pause(message).await?;
                Ok(false)
            }
            PlanUsageAction::Pause => Ok(true),
            PlanUsageAction::SlowDown { delay_ms } => {
                let Some(until) =
                    tokio::time::Instant::now().checked_add(Duration::from_millis(delay_ms))
                else {
                    return Ok(true);
                };
                let sleep = tokio::time::sleep_until(until);
                tokio::pin!(sleep);
                loop {
                    tokio::select! {
                        biased;
                        command = self.command_receiver.recv() => {
                            if self.handle_recovery_command(command).await? { return Ok(false); }
                        }
                        _ = &mut sleep => break,
                    }
                }
                // The wait came after the boundary: commands sent meanwhile
                // are taken as there, and context sent meanwhile joins this
                // request rather than the next, under the boundary's rule
                // (only a pending fallback holds it back).
                if self.take_queued_commands().await? {
                    return Ok(false);
                }
                if self.task.fallback_for.is_none() {
                    self.add_taken_context()?;
                }
                Ok(true)
            }
        }
    }

    async fn handle_tool_exec_error(
        &mut self,
        tool_name: &str,
        error: crate::tool::ToolError,
        tool_call_id: Option<Uuid>,
    ) -> Result<String, ArkoAgentsError> {
        let error_message_for_event = format!("Tool '{}' execution failed: {}", tool_name, error);
        log::warn!("Task {}: {}", self.task.id, error_message_for_event);

        let prompt_builder_for_error_msg =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
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
                });
                Ok(error_message_for_llm)
            }
            ToolErrorPolicy::PauseAndAwaitApproval => {
                self.task
                    .set_state(TaskState::AwaitingErrorRecoveryApproval {
                        error_message: error_message_for_event.clone(),
                    });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event.clone(),
                    requires_approval_to_retry: true,
                    tool_call_id,
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
                })
                .await?;
                Ok(String::new())
            }
        }
    }

    /// Runs an output-based call (auto-approved or approved) and returns
    /// its result as text for the next user message. A cancellation or
    /// suspension while it runs stops it.
    async fn execute_tool_and_handle_result(
        &mut self,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        tool_call_id: Uuid,
        custom_tool_output: Option<String>,
        message_to_agent_on_approval: Option<String>,
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
        let run = self.tool_run(
            tool_call_id,
            tool_name,
            tool_arc.clone(),
            parameters,
            custom_tool_output,
        );
        let group = self.run_group(vec![run]).await;
        let outcome = match group.outcomes {
            Some(mut outcomes) => outcomes.pop().expect("one call, one outcome")?,
            None => {
                let stop = group.stop.expect("a group without outcomes was stopped");
                let reason = self.stop_running_tools(stop).await?;
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
            RunOutcome::Stop { reason, cancel } => {
                let formatted = format_error(self, &reason);
                self.stop_by_middleware(tool_call_id, tool_name, &reason, cancel)
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
                self.emit_tool_result_events(
                    tool_call_id,
                    tool_name,
                    &raw,
                    &effective,
                    intercepted,
                )
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
