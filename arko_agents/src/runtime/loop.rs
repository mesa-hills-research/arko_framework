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

/// The attempts recovery from a failed request makes under `policy`.
fn recovery_max_attempts(policy: LLMErrorPolicy, model_output: bool) -> u32 {
    match policy {
        LLMErrorPolicy::RetryThenInformAgent { max_attempts }
        | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts } => max_attempts.max(1),
        // Responses the runtime can't use are asked again at once and
        // outside the elapsed budget, so they need an attempt limit.
        LLMErrorPolicy::InformAgentAndContinue if model_output => {
            MODEL_OUTPUT_ATTEMPTS_WITHOUT_LIMIT
        }
        LLMErrorPolicy::InformAgentAndContinue => u32::MAX,
        LLMErrorPolicy::PauseAndAwaitApproval => 1,
    }
}

/// What a failed request's recovery reports: the error, the attempt,
/// and how it waits.
fn recovery_message(
    wait: &RecoveryWait,
    attempt: u32,
    max_attempts: u32,
    error: &ai::Error,
    note: &Option<String>,
    fallback: bool,
) -> String {
    let message = match wait {
        RecoveryWait::Capacity(_) => format!(
            "LLM capacity error (attempt {attempt}; waiting until service recovers): {error}"
        ),
        RecoveryWait::RateLimit(_) => {
            let kind = if error.is_capacity_error() {
                "capacity error"
            } else {
                "rate limit"
            };
            format!("LLM {kind} (attempt {attempt}; retrying when the provider allows): {error}")
        }
        RecoveryWait::Ordinary | RecoveryWait::UsageLimit => match note {
            Some(note) => {
                format!("Model response not usable (attempt {attempt}/{max_attempts}): {note}")
            }
            None => format!("LLM error (attempt {attempt}/{max_attempts}): {error}"),
        },
    };
    if fallback {
        format!(
            "The refusal fallback model's request failed; it still has the refused input to answer. {message}"
        )
    } else {
        message
    }
}

/// Why recovery stops at this attempt, with the pause's cause: an error
/// that retrying can't fix, or the attempts used up (not for a wait that
/// doesn't count them).
fn recovery_stop(
    error: &ai::Error,
    indefinite: bool,
    attempt: u32,
    max_attempts: u32,
    model_output: bool,
) -> Option<(&'static str, ErrorRecoveryCause)> {
    if !error.is_retryable() {
        Some((
            "Failure requires a request, credential, or configuration change",
            if matches!(error, ai::Error::Refusal(_)) {
                ErrorRecoveryCause::Refusal
            } else {
                ErrorRecoveryCause::NonRetryable
            },
        ))
    } else if !indefinite && attempt >= max_attempts {
        Some((
            "Retry attempt limit exhausted",
            if model_output {
                ErrorRecoveryCause::UnusableResponses
            } else {
                ErrorRecoveryCause::RetriesExhausted {
                    retry_after_ms: error.retry_after().map(duration_millis),
                }
            },
        ))
    } else {
        None
    }
}

/// The delay before the next attempt: none for a response the runtime
/// can't use, otherwise the backoff (from the provider's delay when it
/// gives one).
fn retry_delay_for(
    wait: &RecoveryWait,
    config: &crate::agent_config::LLMRetryConfig,
    attempt: u32,
    error: &ai::Error,
    model_output: bool,
) -> Duration {
    // A rate limit's own delay is followed as given (plus jitter) rather
    // than grown by backoff; backoff applies when there is none.
    let backoff_attempt = match wait {
        RecoveryWait::RateLimit(_) if error.retry_after().is_some() => 1,
        _ => attempt,
    };
    if model_output {
        Duration::ZERO
    } else {
        super::retry::retry_delay(
            config,
            backoff_attempt,
            error.retry_after(),
            ai::retry::random_sample(),
        )
    }
}

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

/// The facts the notice of a shortened `result` gives: `text` is its text
/// as it stands (without a notice it already ends with), at `tokens`
/// tokens, under its opt-in `cap`.
fn notice_facts<'a>(
    result: &'a ai::ToolResult,
    paths: &'a HashMap<String, String>,
    reason: ShortenReason,
    text: &str,
    tokens: usize,
    cap: Option<usize>,
    tool_config: &crate::agent_config::AgentToolConfig,
) -> result_budget::NoticeFacts<'a> {
    result_budget::NoticeFacts {
        tool_name: &result.name,
        path: result
            .call_id
            .as_ref()
            .and_then(|id| paths.get(id))
            .map(String::as_str),
        reason,
        tokens,
        bytes: text.len(),
        lines: text.lines().count(),
        head_lines: 0,
        tail_lines: 0,
        cap,
        terminal: tool_config.offers_terminal(),
        line_ranges: tool_config.read_file_line_ranges,
    }
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
/// The result of a tool call made while a handover report is due.
const HANDOVER_REPORT_DUE: &str = "This call was not run: your handover report is due. Reply with the report as plain text, without calling a tool (call complete_task instead if the task is done).";

/// The note after a reply to a report request that had no text.
const NO_HANDOVER_REPORT: &str = "Your reply had no text, so no handover report was received. Reply with the handover report as plain text, without calling a tool.";

const NO_NATIVE_TOOL_CALL: &str = "Your response did not call a tool, so the task is still open. Continue with your tools, or call complete_task with a summary if the task is finished.";
const NO_TEXT_TOOL_CALL: &str = "Your response did not contain a tool call. Use one of your tools in the format the system prompt describes, or complete_task if the task is finished.";

/// The attempts a response the runtime can't use gets under
/// `LLMErrorPolicy::InformAgentAndContinue`, which has no attempt limit.
const MODEL_OUTPUT_ATTEMPTS_WITHOUT_LIMIT: u32 = 10;

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
/// The result of a call that was running when the task stopped (its host
/// crashed, say), settled as [`ToolRecovery::InDoubt`] when the task runs
/// again.
const RESTARTED_CALL: &str = "The task was restarted while this call was running (its host stopped), so it may have run partly or fully. Check its effects before calling it again.";
/// Added to the output-based results of a turn the task stopped in the
/// middle of, once they are settled when it runs again.
const RESTARTED_TURN: &str = "The task was restarted while this turn's tool calls ran (its host stopped). A call of the turn without a result here was not run; call it again if it is still needed.";
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
    /// Each call's outcome, in order: `None` for a call a stop ended before
    /// it finished.
    outcomes: Vec<Option<Result<RunOutcome, crate::tool::ToolError>>>,
    /// A cancellation or suspension that arrived while the calls ran.
    stop: Option<ToolsStopped>,
}

/// Results of calls held behind a call awaiting approval (they never needed
/// approval themselves) when the wait ended without a decision.
const HELD_CALL_NOT_RUN: &str = "An earlier call in this turn was still waiting for approval when the wait ended, so this call was not run. Call it again if it is still needed.";

/// The result of `complete_task` called after an earlier call of its turn
/// was rejected or failed: the model chose to finish before it saw those
/// results.
const COMPLETION_AFTER_FAILURE: &str = "The task was not completed: an earlier call in this turn was rejected or failed (see its result). Review the results, and call complete_task again if the task is done.";

/// Results of calls that were running when the task ended another way (a
/// call beside them failed or completed it) as a stop arrived.
const STOPPED_TASK_ENDED: &str =
    "The task ended while this call was running, so the call was stopped before it finished.";

/// The result text of a call a stop ended while it ran, given the task's
/// state once the stop is applied: a cancellation it made, a suspension, or
/// an end another call of the group had already brought about.
fn stopped_call_reason(stop: ToolsStopped, state: &TaskState) -> &'static str {
    match (stop, state) {
        (_, TaskState::Cancelled { reason }) if reason == CANCELLED_BY_USER => STOPPED_BY_CANCEL,
        (_, state) if state.is_terminal() => STOPPED_TASK_ENDED,
        (ToolsStopped::Suspended, _) => STOPPED_BY_SUSPEND,
        (ToolsStopped::Cancelled, _) => STOPPED_BY_CANCEL,
    }
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

/// The results of a turn's (or an approval's) native calls, recorded in the
/// task ([`Task::pending_native_tool_results`]) as they are produced, so
/// that a snapshot taken meanwhile holds them; [`finish`](Self::finish)
/// puts them in call order.
struct NativeResults {
    /// Where the recorded results start in the task's pending results.
    start: usize,
    /// Each recorded result's position in its turn, in the order recorded.
    positions: Vec<usize>,
}

impl NativeResults {
    fn new(task: &Task) -> Self {
        Self {
            start: task.pending_native_tool_results.len(),
            positions: Vec::new(),
        }
    }

    fn record(&mut self, task: &mut Task, position: usize, result: ai::ToolResult) {
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
    fn failed(&self, task: &Task) -> std::collections::BTreeSet<usize> {
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
    fn finish(self, task: &mut Task) {
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

/// A native turn's calls once checked; see [`AgentLoop::native_turn`].
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

/// What a native call's checks decided; see
/// [`AgentLoop::classify_native_call`].
#[allow(
    clippy::large_enum_variant,
    reason = "returned once per call and taken apart at once: a box would only add an allocation"
)]
enum NativeCall {
    /// Answered at once: its result is recorded.
    Answered,
    /// It waits for approval; `exclusive`: it doesn't run beside other
    /// calls.
    NeedsApproval {
        pending: PendingToolCallInfo,
        exclusive: bool,
    },
    /// It runs without approval.
    Run(NativeRun),
}

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

/// A request as it was sent: what its reported usage is compared with to
/// estimate the next request (see [`result_budget`]).
#[derive(Debug, Clone, Copy)]
struct RequestMark {
    /// The conversation's message count.
    messages: usize,
    /// Its size in bytes ([`result_budget::request_bytes`]).
    bytes: usize,
    /// The conversation it belongs to ([`Task::conversation_usage_start`]).
    conversation: usize,
}

/// How many times a request the provider found over the model's input
/// limit is sent again after shortening a tool result; see
/// [`AgentLoop::recover_from_overflow`].
const MAX_OVERFLOW_RECOVERIES: u32 = 2;

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

/// The text of a message.
fn message_text(message: &ai::Message) -> String {
    message
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect()
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

/// A model's turn, streamed to its end, for the turn phase.
struct StreamedTurn {
    /// The final response, which the conversation holds; `None` for a
    /// stream that ended without one, its partial text kept.
    response: Option<ai::GenerationResponse>,
    /// The output-based parser's events (none in native mode).
    parser_events: Vec<RawParserEvent>,
    /// The calls the stream announced (`ToolCallInitiated`), by index, with
    /// their runtime ids and names: the turn runs those its response holds
    /// and rejects the others.
    announced: HashMap<usize, (Uuid, String)>,
}

/// How a pass's model request and its stream ended.
#[allow(
    clippy::large_enum_variant,
    reason = "returned once per turn and taken apart at once: a box would only add an allocation"
)]
enum ModelTurn {
    /// With a turn whose calls the turn phase handles.
    Streamed(StreamedTurn),
    /// Otherwise: the pass ends (`NextPass`, or `Exit` for a task that
    /// ended).
    Ended(Flow),
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

    /// Notes a host's [`UserCommand::RequestHandover`]: the report is
    /// requested at the next boundary where a user message can go. Ignored
    /// while a handover is under way.
    fn note_handover_request(&mut self) {
        if self.task.handover_requested.is_none() {
            self.task.handover_requested = Some(HandoverRequest {
                requested_at: chrono::Utc::now(),
                input_tokens: self.task.current_conversation_context_size().unwrap_or(0),
                report_requested: false,
            });
        }
    }

    /// The agent was asked for its handover report and hasn't written it.
    fn handover_report_due(&self) -> bool {
        self.task
            .handover_requested
            .as_ref()
            .is_some_and(|request| request.report_requested)
    }

    /// Whether a call to `tool_name` is rejected because the handover report
    /// is due: every call except `complete_task`.
    fn handover_blocks(&self, tool_name: &str) -> bool {
        self.handover_report_due() && tool_name != COMPLETE_TASK_TOOL_NAME
    }

    /// Before a model request (not a refusal fallback's): asks the agent for
    /// its handover report when a host asked for a handover or the current
    /// conversation reached the configured threshold, and a user message can
    /// go into the conversation.
    async fn start_handover_if_due(&mut self) -> Result<(), ArkoAgentsError> {
        if self.handover_report_due() || !self.context_can_be_added_now() {
            return Ok(());
        }
        let context_size = self.task.current_conversation_context_size();
        let config = self.task.agent_config.handover.clone();
        if self.task.handover_requested.is_none() {
            let threshold = config.as_ref().and_then(|config| {
                config.threshold(self.task.agent_config.llm_config.model.max_input_tokens)
            });
            let Some((threshold, size)) = threshold.zip(context_size) else {
                return Ok(());
            };
            // The request about to be sent, not the one before it: what
            // the turn added since (its tool results, above all) counts.
            let coming = self.estimated_request_tokens().max(size);
            if coming < threshold {
                return Ok(());
            }
            // A conversation whose first request was already over the
            // threshold would start over the threshold again: handing over
            // can't make it smaller.
            let first = self
                .task
                .usage_history
                .get(self.task.conversation_usage_start)
                .map_or(0, |usage| usage.input_tokens);
            if first >= threshold {
                if !std::mem::replace(&mut self.handover_skip_warned, true) {
                    log::warn!(
                        "Task {}: No handover: the conversation's first request ({} input tokens) was already over the handover threshold ({}).",
                        self.task.id,
                        first,
                        threshold
                    );
                }
                return Ok(());
            }
        }
        let request = self
            .task
            .handover_requested
            .get_or_insert_with(|| HandoverRequest {
                requested_at: chrono::Utc::now(),
                input_tokens: context_size.unwrap_or(0),
                report_requested: false,
            });
        request.report_requested = true;
        let input_tokens = request.input_tokens;
        let report_request = config
            .as_ref()
            .map_or(crate::handover::DEFAULT_REPORT_REQUEST, |config| {
                config.report_request()
            });
        self.task.conversation.add_user_message(report_request);
        self.task.mark_last_message_runtime();
        self.task.increment_user_message_count();
        log::info!(
            "Task {}: Handover requested at {} input tokens.",
            self.task.id,
            input_tokens
        );
        self.emit_event(AgentEvent::HandoverRequested { input_tokens })
            .await
    }

    /// A reply without tool calls while the handover report is due: its
    /// text is the report, and the conversation is handed over; a reply
    /// without text is answered with a note, like a turn without a tool
    /// call.
    async fn take_handover_report(&mut self, report: String) -> Result<(), ArkoAgentsError> {
        if report.trim().is_empty() {
            return self
                .handle_model_output_failure(NO_HANDOVER_REPORT.to_string())
                .await;
        }
        self.complete_handover(report).await
    }

    /// Replaces the conversation with a fresh one that starts from the
    /// seed (the [`HandoverHook`]'s, or the task's initial prompt) and the
    /// report, and resets what the loop keeps about the old one.
    async fn complete_handover(&mut self, report: String) -> Result<(), ArkoAgentsError> {
        let report = report.trim().to_string();
        let seed = match &self.handover_hook {
            Some(hook) => hook.on_handover(&self.task, &report),
            None => HandoverSeed::for_task(&self.task),
        };
        let input_tokens = self
            .task
            .handover_requested
            .take()
            .map_or(0, |request| request.input_tokens);
        let replaced_messages = self.task.start_fresh_conversation();
        self.task.handovers.push(HandoverRecord {
            at: chrono::Utc::now(),
            input_tokens,
            replaced_messages,
            report: report.clone(),
        });
        let prompt_builder =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
        let first_message =
            prompt_builder.format_initial_task_prompt(&self.task, &seed.message(&report))?;
        self.task.conversation.add_user_message(&first_message);
        self.task.increment_user_message_count();

        // What the loop keeps about the replaced conversation: tools that
        // track what the conversation holds start over (see
        // `observe_committed_tool_context`), the file tree is sent in full
        // again, and the cache monitor judges the fresh conversation alone.
        self.tool_context_cursor = usize::MAX;
        self.file_tree_baseline = None;
        self.file_tree_error = None;
        self.details_sent_with = None;
        self.cache_warned = false;
        self.cache_pause_episode = false;
        self.cache_pause_pending = None;
        self.handover_skip_warned = false;
        self.reset_llm_recovery();

        log::info!(
            "Task {}: Conversation handed over ({} messages replaced).",
            self.task.id,
            replaced_messages
        );
        let handovers = self.task.handovers.len();
        // The fresh conversation replaces the saved one.
        self.report_task(false).await;
        self.emit_event(AgentEvent::ConversationHandedOver {
            report,
            input_tokens,
            handovers,
        })
        .await
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

    /// What the task's tool calls see of it; see [`RunEnvironment`].
    fn run_environment(&self) -> Arc<RunEnvironment> {
        Arc::new(RunEnvironment {
            task_id: self.task.id,
            agent_name: self.task.agent_config.name.clone(),
            initial_task_prompt: self.task.initial_task_prompt.clone(),
            working_directory: self.task.current_working_directory.clone(),
            middleware: self.execution_middleware.clone(),
            stop: self.host_stop.or(&self.calls_stop.signal()),
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

    /// Whether calls to a tool may run beside other parallel-safe calls:
    /// the tool is parallel-safe and the agent doesn't ask for one call per
    /// turn.
    fn runs_in_parallel(&self, definition: &ToolDefinition) -> bool {
        let config = &self.task.agent_config;
        // `complete_task` ends the turn, so it always runs alone, after the
        // calls before it (whatever an override says).
        definition.name != COMPLETE_TASK_TOOL_NAME
            && !config.asks_one_tool_call_per_turn()
            && config.is_parallel_safe(definition)
    }

    /// A call to `tool` to run, under the agent's time limit and
    /// parallel-safety settings.
    fn tool_run(
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
            tool,
            params,
            custom_output,
            call_id: None,
            message_to_agent: None,
            recovering: false,
        }
    }

    /// Runs `runs` at the same time while watching the command channel:
    /// context sent meanwhile is kept for the next boundary, and
    /// `CancelTask` or `SuspendTask` stops the calls (dropping them, which
    /// kills a command they started) and is returned. Each call's outcome is
    /// returned in order otherwise; an error is a middleware's own failure.
    /// A group with a call that isn't [interruptible](crate::tool::Tool::interruptible)
    /// runs to its end, and a stop that arrived meanwhile is returned with
    /// the outcomes; either way the calls' stop signal fires
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
    async fn run_group(&mut self, runs: Vec<ToolRun>) -> GroupRun {
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
                    let stopped = match command {
                        Some(UserCommand::InjectContext { content }) => {
                            self.queue_context_injection(content);
                            None
                        }
                        Some(UserCommand::RequestHandover) => {
                            self.note_handover_request();
                            None
                        }
                        Some(UserCommand::CancelTask) => {
                            self.calls_stop.stop("Task cancelled.");
                            Some(ToolsStopped::Cancelled)
                        }
                        Some(UserCommand::SuspendTask) => {
                            self.calls_stop.stop("Task suspended.");
                            Some(ToolsStopped::Suspended)
                        }
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

    /// Reports the task to the snapshot sink, if there is one: a safe point
    /// (see [`TaskSnapshotSink`](crate::snapshot::TaskSnapshotSink)). With
    /// `durable`, returns once the sink has saved it.
    async fn report_task(&mut self, durable: bool) {
        #[cfg(test)]
        self.trace.snapshot(&self.task, durable);
        self.snapshots.report(&self.task, durable).await;
    }

    /// Records an output-based call's result (nothing for an empty one) in
    /// the task as it is produced ([`Task::pending_text_tool_results`]), so
    /// that a snapshot taken meanwhile holds it, and reports the task.
    async fn record_text_result(&mut self, result: String) {
        if !result.is_empty() {
            self.task.pending_text_tool_results.push(result);
        }
        self.report_task(false).await;
    }

    /// Takes the output-based results recorded since `start` back out of
    /// the task, in order.
    fn take_text_results(&mut self, start: usize) -> Vec<String> {
        let results = &mut self.task.pending_text_tool_results;
        results.split_off(start.min(results.len()))
    }

    /// Drops a call from the running calls once its result is in the task.
    fn call_settled(&mut self, tool_call_id: Uuid) {
        self.task
            .running_tool_calls
            .retain(|running| running.tool_call_id != tool_call_id);
    }

    /// Applies a command that stopped running tool calls, once the calls
    /// that finished are handled: a cancellation ends the task (unless one
    /// of the calls already ended it), and a suspension stops the loop after
    /// the turn (the task keeps its state, reported as it stops).
    async fn stop_after_calls(&mut self, stop: ToolsStopped) -> Result<(), ArkoAgentsError> {
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
    async fn run_native_calls(
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
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: native.run.tool_call_id,
                    tool_name: native.call.name.clone(),
                    reason: Some(not_run.clone()),
                })
                .await?;
                results.record(
                    &mut self.task,
                    native.index,
                    ai::ToolResult::error_from_tool_call(&native.call, &not_run),
                );
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
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: id,
                    tool_name: call.name.clone(),
                    reason: Some(reason.to_string()),
                })
                .await?;
                results.record(
                    &mut self.task,
                    index,
                    ai::ToolResult::error_from_tool_call(&call, reason),
                );
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
        }
    }

    /// A native call that finished: `ask_user` awaits the answer and
    /// `complete_task` completes the task (unless another call of its
    /// group already ended it), the result events are sent, and the result
    /// carries the host's message and the tool's images.
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
    async fn pause_by_middleware(
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
    async fn pause_for_middleware_error(
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

    /// Refresh and model stream initialization share one attempt. A request
    /// once sent runs until it completes or fails, or the host cancels or
    /// suspends the task: the elapsed recovery budget never cuts it off.
    async fn run_llm_attempt<T>(
        &mut self,
        operation: impl std::future::Future<Output = Result<T, ai::Error>>,
    ) -> Result<Option<T>, ArkoAgentsError> {
        self.llm_interaction_attempts = self.llm_interaction_attempts.saturating_add(1);
        tokio::pin!(operation);
        loop {
            tokio::select! {
                biased;
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
    async fn flush_native_results_to_conversation(
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
        let text = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None)
            .text_beside_tool_results(&self.task, extra_text, self.task.user_message_count + 1)?;
        self.task.add_pending_tool_results(text.as_deref());
        self.mark_details_sent();
        Ok(())
    }

    /// The estimated input of the current conversation's next request, in
    /// tokens: the latest answered request's reported input, plus the bytes
    /// added since at the task's bytes-per-token ratio (see
    /// [`result_budget`]). Without such a request in this run (a task run
    /// again, or a fresh conversation), the whole request is estimated from
    /// its bytes, and the latest reported request with its response is the
    /// least it can be. What the handover check compares, and what the
    /// overflow guard sizes shortened results by.
    fn estimated_request_tokens(&self) -> usize {
        self.request_estimate(false)
    }

    /// The fewest tokens the current conversation's next request can take:
    /// as [`estimated_request_tokens`](Self::estimated_request_tokens), but
    /// with every byte added since the latest answered request at
    /// [`result_budget::LEAST_BYTES_PER_TOKEN`] and the reply at its bytes
    /// alone. What the overflow guard decides by: it acts only when even
    /// this is over the model's input limit.
    fn least_request_tokens(&self) -> usize {
        self.request_estimate(true)
    }

    fn request_estimate(&self, least: bool) -> usize {
        let ratio = self.bytes_per_token;
        let messages = &self.task.conversation.messages;
        if let (Some(mark), Some(input)) = (
            self.answered_request,
            self.task.current_conversation_context_size(),
        ) && mark.conversation == self.task.conversation_usage_start
            && mark.messages <= messages.len()
        {
            let added = &messages[mark.messages..];
            let bytes = |assistant: bool| {
                added
                    .iter()
                    .filter(|message| (message.role == ai::Role::Assistant) == assistant)
                    .map(result_budget::message_bytes)
                    .sum::<usize>()
            };
            if least {
                return input + result_budget::least_tokens(bytes(true) + bytes(false));
            }
            // The model's reply counts at least its reported output, which
            // includes reasoning the next request may replay.
            let output = self
                .task
                .usage_history
                .last()
                .map_or(0, |usage| usage.output_tokens);
            let reply = output.max(result_budget::tokens(bytes(true), ratio));
            return input + reply + result_budget::tokens(bytes(false), ratio);
        }
        let whole =
            result_budget::request_bytes(&self.task.conversation, &self.build_generation_config());
        let latest = self
            .task
            .usage_history
            .get(self.task.conversation_usage_start..)
            .and_then(<[ai::UsageInfo]>::last);
        if least {
            return result_budget::least_tokens(whole)
                .max(latest.map_or(0, |usage| usage.input_tokens));
        }
        result_budget::tokens(whole, ratio)
            .max(latest.map_or(0, |usage| usage.input_tokens + usage.output_tokens))
    }

    /// For a task that runs again with requests in its current
    /// conversation: its latest request, as the conversation shows it (the
    /// messages before the model's latest reply), with the task's ratio
    /// calibrated from its reported input, so the next request is estimated
    /// as in the run that made it. `None` without such a request.
    fn saved_request_mark(&mut self) -> Option<RequestMark> {
        let input = self.task.current_conversation_context_size()?;
        let conversation = &self.task.conversation;
        let reply = conversation
            .messages
            .iter()
            .rposition(|message| message.role == ai::Role::Assistant)?;
        let bytes = result_budget::request_bytes_of(
            conversation,
            &conversation.messages[..reply],
            &self.build_generation_config(),
        );
        if let Some(ratio) = result_budget::calibrated_ratio(bytes, input) {
            self.bytes_per_token = ratio;
        }
        Some(RequestMark {
            messages: reply,
            bytes,
            conversation: self.task.conversation_usage_start,
        })
    }

    /// The paths the calls of the conversation's latest model turn passed
    /// to `read_file`, by call id: what a shortened result's notice names.
    fn read_file_paths(&self) -> HashMap<String, String> {
        self.task
            .conversation
            .messages
            .iter()
            .rev()
            .find(|message| message.role == ai::Role::Assistant && !message.tool_calls.is_empty())
            .into_iter()
            .flat_map(|message| &message.tool_calls)
            .filter(|call| call.name == crate::tool::builtins::READ_FILE_TOOL_NAME)
            .filter_map(|call| {
                let path = call.arguments.get("path")?.as_str()?;
                Some((call.id.clone()?, path.to_string()))
            })
            .collect()
    }

    /// The overflow guard and the opt-in caps
    /// ([`ToolResultLimits`](crate::agent_config::ToolResultLimits)) over
    /// the turn's native results, as they go into the conversation: after
    /// result interceptors and distillers, before the next request. A
    /// result over its cap, or among the largest when the turn's results
    /// would certainly put the next request over the model's input limit
    /// (see [`result_budget::plan`]), is shortened to its head and tail
    /// with a notice, and `ToolResultShortened` reports it; any other result
    /// is left alone.
    async fn limit_tool_results(&mut self) -> Result<(), ArkoAgentsError> {
        if self.task.pending_native_tool_results.is_empty() {
            return Ok(());
        }
        let tool_config = self.task.agent_config.tool_config.clone();
        let caps: Vec<Option<usize>> = self
            .task
            .pending_native_tool_results
            .iter()
            .map(|result| tool_config.max_result_tokens(&result.name))
            .collect();
        let guard = tool_config.result_limits.overflow_guard;
        if !guard && caps.iter().all(Option::is_none) {
            return Ok(());
        }
        let ratio = self.bytes_per_token;
        let limit = match guard {
            true => {
                self.llm_client
                    .max_input_tokens(&self.task.conversation.model)
                    .await
            }
            false => None,
        };
        let room = limit.map(|limit| {
            let margin = tool_config
                .result_limits
                .margin_tokens
                .unwrap_or_else(|| result_budget::default_margin(limit));
            result_budget::Room {
                available: limit
                    .saturating_sub(margin)
                    .saturating_sub(self.estimated_request_tokens()),
                certain: limit.saturating_sub(self.least_request_tokens()),
                least_share: ratio / result_budget::LEAST_BYTES_PER_TOKEN,
                preview: result_budget::preview_tokens(limit),
            }
        });
        let sizes: Vec<usize> = self
            .task
            .pending_native_tool_results
            .iter()
            .map(|result| result_budget::tokens(result.output.len(), ratio))
            .collect();
        let paths = self.read_file_paths();
        let facts = |index: usize, reason: ShortenReason| {
            let result = &self.task.pending_native_tool_results[index];
            notice_facts(
                result,
                &paths,
                reason,
                &result.output,
                sizes[index],
                caps[index],
                &tool_config,
            )
        };
        let notices: Vec<usize> = (0..sizes.len())
            .map(|index| {
                result_budget::notice_tokens(&facts(index, ShortenReason::Overflow), ratio)
            })
            .collect();
        let cuts = result_budget::plan(&sizes, &notices, &caps, room);
        if cuts.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut shortened = Vec::new();
        for (index, cut) in cuts.into_iter().enumerate() {
            let Some(cut) = cut else {
                continue;
            };
            let result = &self.task.pending_native_tool_results[index];
            let (text, kept) =
                result_budget::shorten(&result.output, cut.keep, ratio, facts(index, cut.reason));
            log::warn!(
                "Task {}: the result of {} (about {} tokens) was shortened to about {kept} tokens ({:?}).",
                self.task.id,
                result.name,
                sizes[index],
                cut.reason
            );
            shortened.push((
                index,
                text,
                AgentEvent::ToolResultShortened {
                    call_id: result.call_id.clone(),
                    tool_name: result.name.clone(),
                    original_tokens: sizes[index],
                    kept_tokens: kept,
                    reason: cut.reason,
                },
            ));
        }
        for (index, text, event) in shortened {
            self.task.pending_native_tool_results[index].output = text;
            self.emit_event(event).await?;
        }
        Ok(())
    }

    /// The largest tool result added to the conversation since the
    /// latest answered request (or since the last message with results):
    /// its message, its place in it, and its size in bytes.
    fn largest_new_result(&self) -> Option<(usize, usize, usize)> {
        let messages = &self.task.conversation.messages;
        // The results added since the latest answered request: one of them
        // made this request too large.
        let start = self
            .answered_request
            .filter(|mark| {
                mark.conversation == self.task.conversation_usage_start
                    && mark.messages <= messages.len()
            })
            .map(|mark| mark.messages)
            .or_else(|| {
                messages.iter().rposition(|message| {
                    message
                        .content
                        .iter()
                        .any(|content| content.as_tool_result().is_some())
                })
            })
            .unwrap_or(messages.len());
        messages
            .iter()
            .enumerate()
            .skip(start)
            .flat_map(|(message, content)| {
                content
                    .content
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, content)| {
                        content
                            .as_tool_result()
                            .map(|result| (message, index, result.output.len()))
                    })
            })
            .max_by_key(|(_, _, bytes)| *bytes)
    }

    /// A request the provider found over the model's input limit
    /// ([`ai::ErrorKind::ContextOverflow`]): with the overflow guard on, in
    /// native tool calling, the largest tool result added since the latest
    /// answered request is shortened (to a preview, or one already that
    /// small to its notice alone) and the request is sent again, at most
    /// [`MAX_OVERFLOW_RECOVERIES`] times in a row. Returns whether it is
    /// sent again; otherwise recovery goes on as for any error that
    /// retrying can't fix, and the task stops with the error's kind. A
    /// handover is no recovery: its report request would be over the limit
    /// too.
    async fn recover_from_overflow(&mut self, error: &ai::Error) -> Result<bool, ArkoAgentsError> {
        if error.kind() != ai::ErrorKind::ContextOverflow
            || !self
                .task
                .agent_config
                .tool_config
                .result_limits
                .overflow_guard
            || !self.is_native_mode()
            || self.overflow_recoveries >= MAX_OVERFLOW_RECOVERIES
        {
            return Ok(false);
        }
        let Some((message, index, _)) = self.largest_new_result() else {
            return Ok(false);
        };
        let ratio = self.bytes_per_token;
        let preview = self
            .llm_client
            .max_input_tokens(&self.task.conversation.model)
            .await
            .map_or(
                result_budget::preview_tokens(0),
                result_budget::preview_tokens,
            );
        let paths = self.read_file_paths();
        let tool_config = self.task.agent_config.tool_config.clone();
        let ai::Content::ToolResult(result) =
            &self.task.conversation.messages[message].content[index]
        else {
            return Ok(false);
        };
        // A notice the result already ends with is replaced, not kept.
        let (output, shortened_before) = match result.output.rfind("\n\n[Shortened: ") {
            Some(notice) => (result.output[..notice].to_string(), true),
            None if result.output.starts_with("[Shortened: ") => (String::new(), true),
            None => (result.output.clone(), false),
        };
        let tokens = result_budget::tokens(output.len(), ratio);
        // Too small to be what made the request too large.
        if tokens < result_budget::MIN_KEPT_TOKENS {
            return Ok(false);
        }
        // A preview first; one shortened before (by the guard, a cap or the
        // previous attempt) keeps its notice alone.
        let keep = if tokens > preview && !shortened_before {
            preview
        } else {
            0
        };
        let facts = notice_facts(
            result,
            &paths,
            ShortenReason::OverflowRecovery,
            &output,
            tokens,
            None,
            &tool_config,
        );
        let (text, kept) = result_budget::shorten(&output, keep, ratio, facts);
        let shortened = AgentEvent::ToolResultShortened {
            call_id: result.call_id.clone(),
            tool_name: result.name.clone(),
            original_tokens: tokens,
            kept_tokens: kept,
            reason: ShortenReason::OverflowRecovery,
        };
        let tool_name = result.name.clone();
        if let ai::Content::ToolResult(result) =
            &mut self.task.conversation.messages[message].content[index]
        {
            result.output = text;
        }
        self.overflow_recoveries += 1;
        // Sent again at once; not an ordinary recovery attempt.
        self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
        log::warn!(
            "Task {}: the request was over the model's input limit; the result of {tool_name} was shortened and the request is sent again ({}/{MAX_OVERFLOW_RECOVERIES}).",
            self.task.id,
            self.overflow_recoveries
        );
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: format!(
                "The request was over the model's input limit; the result of {tool_name} was shortened and the request is sent again ({}/{MAX_OVERFLOW_RECOVERIES}): {error}",
                self.overflow_recoveries
            ),
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: Some(ai::ErrorKind::ContextOverflow),
        })
        .await?;
        self.emit_event(shortened).await?;
        Ok(true)
    }

    /// Records that the conversation's last message carries this turn's
    /// environment details and reminder.
    fn mark_details_sent(&mut self) {
        self.details_sent_with = self.task.conversation.messages.len().checked_sub(1);
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
            .skip(self.task.conversation_usage_start)
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

    /// The pass's model request (to the refusal fallback's model while one
    /// is pending), made under the error policy's recovery, and its stream.
    async fn model_turn(&mut self) -> Result<ModelTurn, ArkoAgentsError> {
        let llm_client = Arc::clone(&self.llm_client);
        let (conversation, gen_config) = self.next_request();
        #[cfg(test)]
        self.trace.request(&conversation, &gen_config);
        // What this request's usage is compared with, to estimate the
        // next one (not for the refusal fallback's model).
        self.sent_request = (!self.current_request_is_fallback).then(|| RequestMark {
            messages: conversation.messages.len(),
            bytes: result_budget::request_bytes(&conversation, &gen_config),
            conversation: self.task.conversation_usage_start,
        });
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
        let Some(response_stream) = self.run_llm_attempt(request).await? else {
            return Ok(ModelTurn::Ended(Flow::NextPass));
        };
        // The provider accepted the request: the elapsed budget stops
        // counting while the response runs, however long that is, and
        // resumes if it fails (the waits between attempts still count,
        // so failures that keep coming stay bounded).
        self.pause_ordinary_recovery_clock();
        self.stream_response(response_stream).await
    }

    /// Streams a response to its end, or until the host stops the task:
    /// its text and reasoning are sent as they come, the calls it
    /// announces are reported, and its end is accounted for (usage, the
    /// request estimate, refusals). Every outcome but a usable turn ends
    /// the pass.
    async fn stream_response(
        &mut self,
        mut response_stream: futures::stream::BoxStream<
            'static,
            Result<ai::StreamEvent, ai::Error>,
        >,
    ) -> Result<ModelTurn, ArkoAgentsError> {
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

        // The stream runs until it ends or fails, or the host stops the
        // task; no recovery budget cuts it off (only arko_ai's opt-in
        // stream idle timeout can).
        loop {
            tokio::select! {
                biased;
                maybe_cmd = self.command_receiver.recv() => {
                    if self.on_stream_command(maybe_cmd, &current_assistant_response_text_for_conv, &final_generation_response, &mut native_stream_call_ids).await? {
                        had_stream_error = true;
                        break;
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
                            if self.on_response_end(*boxed_generation_response, &mut parser, &mut raw_events_for_turn, &mut native_stream_call_ids, &mut final_generation_response).await? {
                                had_stream_error = true;
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
                return Ok(ModelTurn::Ended(Flow::Exit));
            }
            return Ok(ModelTurn::Ended(Flow::NextPass));
        }
        Ok(ModelTurn::Streamed(StreamedTurn {
            response: final_generation_response,
            parser_events: raw_events_for_turn,
            announced: native_stream_call_ids,
        }))
    }

    /// A command that arrived while a response streamed. `CancelTask`,
    /// `SuspendTask` or a closed channel stops the stream: the partial
    /// text is kept, the calls it announced are rejected, and `true` says
    /// the pass ends. Context and handover requests are taken; other
    /// commands are ignored.
    async fn on_stream_command(
        &mut self,
        maybe_cmd: Option<UserCommand>,
        current_assistant_response_text_for_conv: &str,
        final_generation_response: &Option<ai::GenerationResponse>,
        native_stream_call_ids: &mut HashMap<usize, (Uuid, String)>,
    ) -> Result<bool, ArkoAgentsError> {
        match maybe_cmd {
            Some(UserCommand::CancelTask) => {
                if !current_assistant_response_text_for_conv.is_empty()
                    && final_generation_response.is_none()
                {
                    self.task
                        .conversation
                        .add_assistant_message(current_assistant_response_text_for_conv);
                }
                self.reject_announced_calls(native_stream_call_ids, INTERRUPTED_CALL)
                    .await?;
                self.task.set_state(TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.to_string(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                return Ok(true);
            }
            Some(UserCommand::SuspendTask) => {
                if !current_assistant_response_text_for_conv.is_empty()
                    && final_generation_response.is_none()
                {
                    self.task
                        .conversation
                        .add_assistant_message(current_assistant_response_text_for_conv);
                }
                self.reject_announced_calls(native_stream_call_ids, INTERRUPTED_CALL)
                    .await?;
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                self.stop_requested = true;
                return Ok(true);
            }
            Some(UserCommand::InjectContext { content }) => {
                self.queue_context_injection(content);
            }
            Some(UserCommand::RequestHandover) => self.note_handover_request(),
            Some(_) => {}
            None => {
                if !current_assistant_response_text_for_conv.is_empty()
                    && final_generation_response.is_none()
                {
                    self.task
                        .conversation
                        .add_assistant_message(current_assistant_response_text_for_conv);
                }
                self.reject_announced_calls(native_stream_call_ids, INTERRUPTED_CALL)
                    .await?;
                self.stop_requested = true;
                let reason = "Command channel disconnected during LLM streaming.".to_string();
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
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// A response's `End`: its usage and the request it answered are
    /// recorded and reported, then a refused turn is handled as a refusal
    /// (`true`: the pass ends), and any other turn joins the conversation
    /// as the final response.
    async fn on_response_end(
        &mut self,
        mut generation_response: ai::GenerationResponse,
        parser: &mut Option<ToolCallParser>,
        raw_events_for_turn: &mut Vec<RawParserEvent>,
        native_stream_call_ids: &mut HashMap<usize, (Uuid, String)>,
        final_generation_response: &mut Option<ai::GenerationResponse>,
    ) -> Result<bool, ArkoAgentsError> {
        if self.current_request_is_fallback {
            self.prepare_fallback_response(&mut generation_response);
        }
        self.task
            .usage_history
            .push(generation_response.usage.clone());
        // The request was answered: its usage calibrates
        // the estimate of the next one.
        self.answered_request = self.sent_request.take();
        if let Some(ratio) = self.answered_request.and_then(|mark| {
            result_budget::calibrated_ratio(mark.bytes, generation_response.usage.input_tokens)
        }) {
            self.bytes_per_token = ratio;
        }
        self.overflow_recoveries = 0;
        self.emit_event(AgentEvent::UsageReported(generation_response.usage.clone()))
            .await?;
        self.record_and_emit_model_call_usage(generation_response.model_call_usage.clone())
            .await?;
        if generation_response.stop_reason == ai::StopReason::MaxTokens {
            let usage = &generation_response.model_call_usage;
            log::warn!(
                "Task {}: the response of {} ended at its output-token limit.",
                self.task.id,
                usage.model
            );
            self.emit_event(AgentEvent::OutputTokenLimitReached {
                model: usage
                    .served_model
                    .clone()
                    .unwrap_or_else(|| usage.model.clone()),
                output_tokens: generation_response.usage.output_tokens,
            })
            .await?;
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
            if let Some(ref mut p) = *parser {
                raw_events_for_turn.extend(p.finalize());
            }
            let plain_text = parser
                .is_some()
                .then(|| parsed_plain_text(raw_events_for_turn));
            self.reject_announced_calls(native_stream_call_ids, REFUSED_TURN_CALL)
                .await?;
            self.handle_refusal(refusal, generation_response.message, plain_text)
                .await?;
            return Ok(true);
        }
        // A completed turn ends any fallback after a
        // refusal and answers any refused input.
        self.task.fallback_for = None;
        self.task.refused_input = None;
        // The provider answered: an outage is over, so
        // its elapsed budget no longer runs (a response
        // the runtime can't use still counts as an
        // attempt).
        self.recovery_started = None;
        self.paused_recovery_elapsed = None;
        self.task
            .conversation
            .add_assistant_message_from_response(generation_response.clone());
        *final_generation_response = Some(generation_response);
        if let Some(ref mut p) = *parser {
            raw_events_for_turn.extend(p.finalize());
        }
        Ok(false)
    }

    /// A native turn. Its calls are checked, and in call order those that
    /// need no approval run (or are held behind a call awaiting approval)
    /// and those that need it wait for it; every call gets one result. A
    /// turn without calls is the handover report, or a response without a
    /// tool call.
    async fn native_turn(&mut self, turn: StreamedTurn) -> Result<Flow, ArkoAgentsError> {
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
        if self.native_results_are_due() {
            self.flush_native_results_to_conversation(None).await?;
        }
        Ok(Flow::Proceed)
    }

    /// A native turn without calls: the handover report, when it is due,
    /// or a response without a tool call, which gets a note and is asked
    /// again under the error policy. Calls the stream announced are
    /// rejected.
    async fn native_turn_without_calls(
        &mut self,
        final_generation_response: Option<ai::GenerationResponse>,
        mut native_stream_call_ids: HashMap<usize, (Uuid, String)>,
    ) -> Result<Flow, ArkoAgentsError> {
        self.reject_announced_calls(&mut native_stream_call_ids, UNDELIVERED_CALL)
            .await?;
        if self.handover_report_due() {
            let report = final_generation_response
                .as_ref()
                .map(|response| message_text(&response.message))
                .unwrap_or_default();
            self.take_handover_report(report).await?;
            if self.task.current_state.is_terminal() {
                return Ok(Flow::Exit);
            }
            return Ok(Flow::NextPass);
        }
        log::warn!(
            "LLM response for task {} did not contain a native tool call.",
            self.task.id
        );
        self.handle_model_output_failure(NO_NATIVE_TOOL_CALL.to_string())
            .await?;
        if self.task.current_state.is_terminal() {
            return Ok(Flow::Exit);
        }
        Ok(Flow::NextPass)
    }

    /// A native call's checks, in order: arguments that aren't JSON, a
    /// handover report that is due, a tool the run doesn't offer, and its
    /// paths. A call they reject is answered (`ToolRejected` or
    /// `UnavailableToolAttempted`, and its result recorded); otherwise it
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

        if self.handover_blocks(&ai_tool_call.name) {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name: ai_tool_call.name.clone(),
                reason: Some(HANDOVER_REPORT_DUE.to_string()),
            })
            .await?;
            results.record(
                &mut self.task,
                call_index,
                ai::ToolResult::error_from_tool_call(ai_tool_call, HANDOVER_REPORT_DUE),
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
        let path_rejection = match self.check_path_parameters(&tool_definition, &params_map) {
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
                        outside_workspace = Some(param_name);
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
    /// and the held ones as not run; otherwise `complete_task` awaiting
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
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: pending.tool_call_id,
                        tool_name: pending.tool_name,
                        reason: Some(not_run.clone()),
                    })
                    .await?;
                    results.record(
                        &mut self.task,
                        index,
                        ai::ToolResult::error_from_tool_call(&call, &not_run),
                    );
                }
            }
            for native in held.drain(..) {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: native.run.tool_call_id,
                    tool_name: native.call.name.clone(),
                    reason: Some(not_run.clone()),
                })
                .await?;
                results.record(
                    &mut self.task,
                    native.index,
                    ai::ToolResult::error_from_tool_call(&native.call, &not_run),
                );
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

    /// Waits for the approval of a turn's calls: one call alone, or the
    /// calls of a turn of several as a batch (`TaskStateChanged`, then
    /// `ToolCallAwaitingApproval` or `BatchToolCallAwaitingApproval`).
    async fn await_approvals(
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

    /// An output-based turn. The parser's events are reported, then the
    /// turn's one call, or batch of calls of one tool, is checked, and in
    /// order its calls run or wait for approval; their results go into the
    /// conversation as one message (or wait with the task). A turn without
    /// a call is the handover report, or a response without a tool call.
    async fn text_turn(&mut self, turn: StreamedTurn) -> Result<Flow, ArkoAgentsError> {
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
                        if self.task.current_state.is_terminal() {
                            return Ok(Flow::Exit);
                        }
                        return Ok(Flow::NextPass);
                    }
                    if !self.task.current_state.is_terminal() {
                        self.handle_model_output_failure(NO_TEXT_TOOL_CALL.to_string())
                            .await?;
                        if self.task.current_state.is_terminal() {
                            return Ok(Flow::Exit);
                        }
                        return Ok(Flow::NextPass);
                    }
                } else {
                    let current_group = tool_call_groups.remove(0);
                    let calls_to_process_with_ids = self.calls_of_group(
                        current_group,
                        &mut parsed_call_ids,
                        &mut next_user_message_str_accumulated,
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
                            TextCall::NeedsApproval(pending) => {
                                calls_needing_approval.push(pending)
                            }
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
                            self.record_text_result(format!(
                                "Tool {}: {not_run}",
                                pending.tool_name
                            ))
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
                    next_user_message_str_accumulated =
                        self.take_text_results(start).join("\n\n---\n\n");
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
                    return Ok(Flow::Exit);
                }
                return Ok(Flow::NextPass);
            }
        }

        self.text_turn_message(
            next_user_message_str_accumulated,
            &final_generation_response,
        )?;
        Ok(Flow::Proceed)
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
            let pb =
                PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
            let generic_retry_prompt = "Please continue with the task or use the 'complete_task' tool if finished. Ensure you use a tool.";
            let next_msg_num = self.task.user_message_count + 1;
            let augmented_generic_prompt = pb.augment_user_message_with_env_details_and_reminder(
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
        Ok(())
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

    /// The position of `call` in the turn it belongs to (the conversation's
    /// last assistant message), matched by its provider id, or without one
    /// by its name and arguments; a call not found sorts last.
    fn turn_position(&self, call: &ai::ToolCall) -> usize {
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
    async fn run_decided_native_calls(
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

    /// Ends recovery: the task pauses for error-recovery approval, with
    /// `cause`, under the pausing error policies, or is interrupted.
    async fn finish_llm_recovery(
        &mut self,
        message: String,
        tool_call_id: Option<Uuid>,
        cause: ErrorRecoveryCause,
        error_kind: Option<ai::ErrorKind>,
    ) -> Result<(), ArkoAgentsError> {
        // A pending fallback stays pending: continuing sends its request
        // again, and resolving the refusal drops it.
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
                    cause,
                    error_kind,
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
            error_kind,
        })
        .await
    }

    /// Reports a refusal and stops recovery: the same input is expected to be
    /// refused again, so it is never retried unchanged. With the opt-in
    /// [`AgentConfig::refusal_fallback`], a refusal from the configured model
    /// instead sends the same input to the fallback model once; a failure
    /// of that request is recovered from by its own error, the fallback
    /// staying pending, and only the fallback's own refusal makes the
    /// refusal stand (see [`recover`](Self::recover)).
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
            // The refusal was a response: the fallback's request starts a
            // recovery of its own.
            self.reset_llm_recovery();
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

    /// The attempt a failed request is in its recovery, and when that
    /// recovery started. A wait that doesn't count against ordinary
    /// recovery (capacity, rate limits) stops its elapsed clock and counts
    /// on its own; otherwise the clock resumes (not for a response the
    /// runtime can't use, which isn't an outage).
    fn recovery_attempt(
        &mut self,
        indefinite: bool,
        model_output: bool,
    ) -> (u32, tokio::time::Instant) {
        if indefinite {
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
            // A response the runtime can't use isn't an outage: the elapsed
            // budget stays as it was (paused while the response ran).
            if !model_output {
                self.resume_ordinary_recovery_clock();
            }
            let started = if model_output {
                self.recovery_started
                    .unwrap_or_else(tokio::time::Instant::now)
            } else {
                *self
                    .recovery_started
                    .get_or_insert_with(tokio::time::Instant::now)
            };
            (self.ordinary_attempts(), started)
        }
    }

    async fn recover(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
        note: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        // A request over the model's input limit is made smaller, not sent
        // again unchanged.
        if note.is_none() && self.recover_from_overflow(&error).await? {
            return Ok(());
        }
        // Recorded with the pause or error that ends recovery.
        let error_kind = Some(error.kind()).filter(|kind| *kind != ai::ErrorKind::Unknown);
        // A failed fallback request (the refusal fallback answering a refused
        // turn) is recovered from like any request, by its own error: a
        // transient failure is retried, to the fallback model, and a pause
        // keeps the fallback pending, so continuing sends it again. Only the
        // fallback's own refusal (see `handle_refusal`) makes the refusal
        // stand.
        let fallback = self.task.fallback_for.is_some() && self.current_request_is_fallback;
        let mut config = self.task.agent_config.llm_retry_config.clone();
        let policy = self.task.agent_config.llm_error_policy.clone();
        let wait = self.recovery_wait(&error);
        if let RecoveryWait::UsageLimit = wait {
            return self.handle_usage_limit(error, tool_call_id).await;
        }
        let indefinite = !matches!(wait, RecoveryWait::Ordinary);
        let model_output = note.is_some();
        let (attempt, started) = self.recovery_attempt(indefinite, model_output);
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
        let max_attempts = recovery_max_attempts(policy, model_output);
        let message = recovery_message(&wait, attempt, max_attempts, &error, &note, fallback);
        let stop = recovery_stop(&error, indefinite, attempt, max_attempts, model_output);
        if let Some((reason, cause)) = stop {
            return self
                .finish_llm_recovery(
                    format!("{reason}. {message}"),
                    tool_call_id,
                    cause,
                    error_kind,
                )
                .await;
        }
        let delay = retry_delay_for(&wait, &config, attempt, &error, model_output);
        if !indefinite
            && !model_output
            && elapsed.saturating_add(delay) >= Duration::from_millis(config.max_elapsed_ms)
        {
            return self
                .finish_llm_recovery(
                    format!("Next retry cannot fit the elapsed recovery budget. {message}"),
                    tool_call_id,
                    ErrorRecoveryCause::RetriesExhausted {
                        retry_after_ms: Some(duration_millis(delay)),
                    },
                    error_kind,
                )
                .await;
        }
        let Some(retry_at) = tokio::time::Instant::now().checked_add(delay) else {
            return self
                .finish_llm_recovery(
                    format!("Server retry delay cannot be represented safely. {message}"),
                    tool_call_id,
                    ErrorRecoveryCause::RetriesExhausted {
                        retry_after_ms: Some(duration_millis(delay)),
                    },
                    error_kind,
                )
                .await;
        };
        let counts_toward_budget =
            matches!(wait, RecoveryWait::RateLimit(_)) && error.is_rate_limit();
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: false,
            tool_call_id,
            error_kind,
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
            // Asked again at once, with no budget deadline to meet.
            return Ok(());
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
                    ErrorRecoveryCause::UsageLimit { resets_at_unix_ms },
                    None,
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
            error_kind: None,
        })
        .await?;
        self.wait_before_retry(retry_at, tool_call_id).await
    }

    /// Waits until `retry_at` before a retry, handling commands; the task is
    /// then in progress again. An ordinary recovery deadline that passes
    /// first ends recovery: the elapsed budget limits the waits between
    /// attempts and the decision to make another, never a request in flight.
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
                    let retry_after_ms = Some(duration_millis(retry_at.saturating_duration_since(tokio::time::Instant::now())));
                    return self.finish_llm_recovery("Elapsed model recovery budget exhausted during backoff.".into(), tool_call_id, ErrorRecoveryCause::RetriesExhausted { retry_after_ms }, None).await;
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
    /// that approving it adds no error note to the conversation. A fallback
    /// still to be made survives the pause.
    async fn finish_monitor_pause(&mut self, message: String) -> Result<(), ArkoAgentsError> {
        self.finish_llm_recovery(message, None, ErrorRecoveryCause::Monitor, None)
            .await?;
        self.task.monitor_pause = matches!(
            self.task.current_state,
            TaskState::AwaitingErrorRecoveryApproval { .. }
        );
        Ok(())
    }

    /// Holds the next model request for the host
    /// ([`TurnContextSource::hold_request`](super::TurnContextSource::hold_request)):
    /// the task pauses for error-recovery approval, whatever the error
    /// policy, marked as a monitor pause is, so that approving it adds no
    /// error note. A fallback still to be made survives the pause.
    async fn hold_request(&mut self, reason: String) -> Result<(), ArkoAgentsError> {
        log::info!(
            "Task {}: the host holds the next request: {reason}",
            self.task.id
        );
        self.task
            .set_state(TaskState::AwaitingErrorRecoveryApproval {
                error_message: reason.clone(),
                cause: ErrorRecoveryCause::BudgetExceeded,
                error_kind: None,
            });
        self.task.monitor_pause = true;
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: reason,
            requires_approval_to_retry: true,
            tool_call_id: None,
            error_kind: None,
        })
        .await
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
            PlanUsageAction::SlowDown { delay_ms } => self.slow_down(delay_ms).await,
        }
    }

    /// The plan-usage slow-down: waits `delay_ms` before the request,
    /// taking commands (a stop ends the wait and the request), then takes
    /// the commands queued meanwhile as a turn boundary does, and the
    /// context sent meanwhile joins this request. Returns whether to make
    /// it.
    async fn slow_down(&mut self, delay_ms: u64) -> Result<bool, ArkoAgentsError> {
        if !std::mem::replace(&mut self.slow_down_cache_checked, true) {
            let lifetime = self
                .llm_client
                .prompt_cache_lifetime(
                    &self.task.agent_config.llm_config.model,
                    &self.task.agent_config.llm_config.generation_config,
                )
                .await;
            super::monitor::warn_if_slow_down_outlasts_cache(self.task.id, delay_ms, lifetime);
        }
        let Some(until) = tokio::time::Instant::now().checked_add(Duration::from_millis(delay_ms))
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
    async fn execute_text_call(
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
