use crate::agent_config::{AgentConfig, DeveloperDefinableEnvironmentDetail};
use crate::error::ArkoAgentsError;
use crate::handover::{HandoverRecord, HandoverRequest};
use arko_ai::{ModelCallUsage, UsageInfo};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use uuid::Uuid;

mod cut_back;
mod refusal;
pub use cut_back::{UserTurn, UserTurnKind};
pub use refusal::{RefusalResolution, RefusedInput, RefusedMessage, RefusedToolResult};

/// Information about a single tool call pending approval, typically part of a batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingToolCallInfo {
    /// Individual ID for this specific call.
    pub tool_call_id: Uuid,
    pub tool_name: String,
    pub parameters: HashMap<String, Value>,
    pub tool_description: String,
    /// Why the call awaits approval (not recorded for a call saved by an
    /// older version).
    #[serde(default, skip_serializing_if = "ApprovalReason::is_unrecorded")]
    pub approval_reason: ApprovalReason,
}

/// Why a tool call awaits approval: the agent's configuration doesn't
/// auto-approve the tool, a path it names is outside the workspace under
/// [`PathSafetyPolicy::RequireApproval`](crate::agent_config::PathSafetyPolicy::RequireApproval),
/// or both. Both unset: not recorded (a call saved by an older version).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalReason {
    /// The agent's tool configuration doesn't auto-approve the tool
    /// (`default_auto_approve_tool_use` and the tool's
    /// `auto_approve_tool_use` override).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tool_not_auto_approved: bool,
    /// The path parameter whose value is outside the workspace, when the
    /// tool's path-safety policy requires approval for that.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_outside_workspace: Option<String>,
}

impl ApprovalReason {
    /// Neither reason is recorded.
    pub fn is_unrecorded(&self) -> bool {
        !self.tool_not_auto_approved && self.path_outside_workspace.is_none()
    }

    /// A short description, such as "the agent's configuration requires
    /// approval for this tool" or "parameter 'path' is outside the
    /// workspace".
    pub fn describe(&self) -> String {
        let mut reasons = Vec::new();
        if self.tool_not_auto_approved {
            reasons.push("the agent's configuration requires approval for this tool".to_string());
        }
        if let Some(parameter) = &self.path_outside_workspace {
            reasons.push(format!("parameter '{parameter}' is outside the workspace"));
        }
        if reasons.is_empty() {
            return "approval required (reason not recorded)".to_string();
        }
        reasons.join("; ")
    }
}

/// Why a task awaits error-recovery approval
/// ([`TaskState::AwaitingErrorRecoveryApproval`]), so a host can decide
/// without reading the error message.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorRecoveryCause {
    /// Not recorded: a task saved by an older version.
    #[default]
    Unknown,
    /// A request that may succeed later (a network, server, capacity or
    /// rate-limit error) kept failing until recovery ended: its attempts or
    /// its elapsed budget ran out. Approving starts a new recovery.
    /// `retry_after_ms` is how long the provider asked to wait before the
    /// next attempt (or the wait the runtime had scheduled), when known.
    RetriesExhausted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
    /// A failure retrying unchanged can't fix: the request, a credential or
    /// the configuration must change (for example an authentication or
    /// invalid-request error).
    NonRetryable,
    /// Responses the runtime couldn't use (a native-mode turn without a
    /// tool call, or output-based tool calls it couldn't read) up to the
    /// attempt limit.
    UnusableResponses,
    /// A plan usage limit was reached; when it resets, in Unix
    /// milliseconds, when the provider said.
    UsageLimit { resets_at_unix_ms: Option<u64> },
    /// The model or a filter around it refused the turn, after the opt-in
    /// refusal fallback, if configured, also refused. The input is in
    /// [`Task::refused_input`].
    Refusal,
    /// Tool calls failed under `ToolErrorPolicy::PauseAndAwaitApproval`;
    /// their results report the errors.
    ToolError,
    /// An opt-in monitor paused the task before a request: the plan usage
    /// threshold or the cache-efficiency monitor.
    Monitor,
    /// The tool execution middleware paused the task at a call
    /// ([`ToolExecutionDecision::PauseTask`](crate::tool::ToolExecutionDecision::PauseTask)),
    /// or one of its hooks returned an error. This leaves the decision to the
    /// host (for example, a workflow's completion verifier could not review a
    /// completion).
    Middleware,
    /// The host held the task's next model request because a spending
    /// budget is used up
    /// ([`TurnContextSource::hold_request`](crate::runtime::TurnContextSource::hold_request)).
    /// No request was made; continuing sends it (or the host holds it
    /// again), and approving adds no error note.
    BudgetExceeded,
}

/// The reason a task stopped by [`UserCommand::CancelTask`](crate::runtime::command::UserCommand::CancelTask) records.
pub const CANCELLED_BY_USER: &str = "Task cancelled by user.";

/// Represents the current state of a task.
///
/// Tasks saved before [`TaskState::Cancelled`] existed recorded a cancelled
/// task as `Failed` with the reason [`CANCELLED_BY_USER`]; that state loads
/// as `Cancelled`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(from = "TaskStateRepr")]
pub enum TaskState {
    Pending,
    InProgress,
    /// Awaiting approval for a single tool call.
    AwaitingToolApproval {
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
        tool_description: String,
        /// Why the call awaits approval (not recorded in a task saved by an
        /// older version).
        #[serde(default, skip_serializing_if = "ApprovalReason::is_unrecorded")]
        approval_reason: ApprovalReason,
    },
    /// Awaiting approval for a batch where one or more calls require approval.
    AwaitingBatchToolApproval {
        /// An ID for the entire batch operation.
        batch_id: Uuid,
        /// Details of each call needing approval.
        pending_calls: Vec<PendingToolCallInfo>,
    },
    /// Awaiting user response to questions asked by the agent via `ask_user` tool.
    AwaitingUserResponse {
        /// The questions asked by the agent.
        questions: String,
    },
    AwaitingErrorRecoveryApproval {
        error_message: String,
        /// What caused the pause (`Unknown` in a task saved by an older
        /// version).
        #[serde(default)]
        cause: ErrorRecoveryCause,
        /// The kind of the model-request error that ended recovery, when it
        /// was classified ([`arko_ai::Error::kind`]): a lost login
        /// ([`ErrorKind::Authentication`](arko_ai::ErrorKind::Authentication)),
        /// a request over the model's input limit
        /// ([`ErrorKind::ContextOverflow`](arko_ai::ErrorKind::ContextOverflow)),
        /// and so on. `None` for an unclassified error, a pause of another
        /// cause, and in a task saved by an older version.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_kind: Option<arko_ai::ErrorKind>,
    },
    Interrupted {
        reason: String,
    },
    Completed {
        summary: String,
    },
    Failed {
        reason: String,
        /// Why, when the task failed at an error-recovery pause the host
        /// rejected (a workflow's error recovery policy fails a pause whose
        /// cause retrying can't fix): the pause's cause, such as
        /// [`ErrorRecoveryCause::NonRetryable`] for a request that needs a
        /// new credential or a configuration change. `None` for other
        /// failures, and in a task saved by an older version.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cause: Option<ErrorRecoveryCause>,
        /// The error's kind, when the task failed at an error-recovery
        /// pause that recorded one (see
        /// [`TaskState::AwaitingErrorRecoveryApproval`]). `None` otherwise,
        /// and in a task saved by an older version.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_kind: Option<arko_ai::ErrorKind>,
    },
    /// Stopped on request (a [`CancelTask`](crate::runtime::command::UserCommand::CancelTask)
    /// command or a middleware's
    /// [`ToolExecutionDecision::CancelTask`](crate::tool::ToolExecutionDecision::CancelTask))
    /// rather than by a failure.
    Cancelled {
        reason: String,
    },
}

/// The serialized form of [`TaskState`], read before the legacy mapping.
#[derive(Deserialize)]
enum TaskStateRepr {
    Pending,
    InProgress,
    AwaitingToolApproval {
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
        tool_description: String,
        #[serde(default)]
        approval_reason: ApprovalReason,
    },
    AwaitingBatchToolApproval {
        batch_id: Uuid,
        pending_calls: Vec<PendingToolCallInfo>,
    },
    AwaitingUserResponse {
        questions: String,
    },
    AwaitingErrorRecoveryApproval {
        error_message: String,
        #[serde(default)]
        cause: ErrorRecoveryCause,
        #[serde(default)]
        error_kind: Option<arko_ai::ErrorKind>,
    },
    Interrupted {
        reason: String,
    },
    Completed {
        summary: String,
    },
    Failed {
        reason: String,
        #[serde(default)]
        cause: Option<ErrorRecoveryCause>,
        #[serde(default)]
        error_kind: Option<arko_ai::ErrorKind>,
    },
    Cancelled {
        reason: String,
    },
}

impl From<TaskStateRepr> for TaskState {
    fn from(state: TaskStateRepr) -> Self {
        match state {
            TaskStateRepr::Pending => TaskState::Pending,
            TaskStateRepr::InProgress => TaskState::InProgress,
            TaskStateRepr::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
                approval_reason,
            } => TaskState::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
                approval_reason,
            },
            TaskStateRepr::AwaitingBatchToolApproval {
                batch_id,
                pending_calls,
            } => TaskState::AwaitingBatchToolApproval {
                batch_id,
                pending_calls,
            },
            TaskStateRepr::AwaitingUserResponse { questions } => {
                TaskState::AwaitingUserResponse { questions }
            }
            TaskStateRepr::AwaitingErrorRecoveryApproval {
                error_message,
                cause,
                error_kind,
            } => TaskState::AwaitingErrorRecoveryApproval {
                error_message,
                cause,
                error_kind,
            },
            TaskStateRepr::Interrupted { reason } => TaskState::Interrupted { reason },
            TaskStateRepr::Completed { summary } => TaskState::Completed { summary },
            // Saved before cancellation had its own state.
            TaskStateRepr::Failed { reason, .. } if reason == CANCELLED_BY_USER => {
                TaskState::Cancelled { reason }
            }
            TaskStateRepr::Failed {
                reason,
                cause,
                error_kind,
            } => TaskState::Failed {
                reason,
                cause,
                error_kind,
            },
            TaskStateRepr::Cancelled { reason } => TaskState::Cancelled { reason },
        }
    }
}

impl TaskState {
    /// Returns true if the task is in a terminal state (Completed, Failed or
    /// Cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed { .. } | TaskState::Failed { .. } | TaskState::Cancelled { .. }
        )
    }

    /// Returns true if the task is awaiting some form of user approval or response.
    pub fn is_awaiting_approval(&self) -> bool {
        matches!(
            self,
            TaskState::AwaitingToolApproval { .. }
                | TaskState::AwaitingBatchToolApproval { .. }
                | TaskState::AwaitingUserResponse { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. }
        )
    }
}

/// Represents a task being executed by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub agent_config_id: String,
    /// The config this task was started with. It is saved with its tool
    /// calling mode; a task saved without one ran output-based and keeps
    /// that mode (see [`ToolCallingMode`](crate::agent_config::ToolCallingMode)).
    #[serde(deserialize_with = "crate::agent_config::deserialize_task_agent_config")]
    pub agent_config: AgentConfig,
    pub initial_task_prompt: String,
    /// Images that go with the initial prompt. Set before the task first
    /// runs: the runtime moves them into the task's first message, after
    /// its text, so a task holds them once. [`Task::initial_images`] gives
    /// them either way.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub initial_task_images: Vec<arko_ai::Image>,
    pub conversation: arko_ai::Conversation,
    pub current_state: TaskState,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub start_time: DateTime<Utc>,
    #[serde(with = "chrono::serde::ts_seconds_option", default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<DateTime<Utc>>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub developer_env_details: Option<Vec<DeveloperDefinableEnvironmentDetail>>,

    #[serde(default)]
    pub usage_history: Vec<UsageInfo>,

    /// Per-turn model call usage metrics with computed input deltas.
    ///
    /// Each entry corresponds to one LLM call. The `input_delta_from_previous_call`
    /// field is populated by the agent runtime when the usage is recorded.
    #[serde(default)]
    pub model_call_usage_history: Vec<ModelCallUsage>,

    /// Task-specific current working directory.
    pub current_working_directory: PathBuf,

    /// Counter for the number of user messages sent to the LLM.
    /// Used for determining when to inject user reminders.
    #[serde(default)]
    pub user_message_count: usize,

    /// Tool outputs waiting for the remainder of a native batch or approval.
    /// Persist them together so image attachments retain their call correlation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_native_tool_results: Vec<arko_ai::ToolResult>,
    /// Native calls of the last turn that haven't run yet, by their
    /// runtime call id: the calls awaiting approval (those the awaiting
    /// state lists) and the auto-approved calls held until an earlier call's
    /// approval is decided (see
    /// [`AgentToolConfig`](crate::agent_config::AgentToolConfig)).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub pending_native_tool_calls: HashMap<Uuid, arko_ai::ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_tool_images: Vec<crate::tool::ToolImage>,
    /// Output-based tool results of a turn or approval that ended waiting
    /// for a command (an approval, or a pause on a tool error): they are
    /// added before the message that ends the wait. The output-based
    /// counterpart of `pending_native_tool_results`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_text_tool_results: Vec<String>,
    /// Tool errors the task pauses on under
    /// `ToolErrorPolicy::PauseAndAwaitApproval` once the other calls of
    /// their turn are done, including calls still awaiting approval, and
    /// the pauses the execution middleware asked for
    /// ([`ToolExecutionDecision::PauseTask`](crate::tool::ToolExecutionDecision::PauseTask)).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred_tool_errors: Vec<DeferredToolError>,

    /// The input of the latest refused turn, until the refusal is resolved,
    /// recovery is approved without a change, or a later turn completes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused_input: Option<RefusedInput>,
    /// Indices of conversation messages the runtime added (file-tree
    /// updates, error and recovery notes, requests to continue), which
    /// [`RefusalResolution`] leaves alone.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub runtime_messages: BTreeSet<usize>,
    /// Indices of tool-results messages whose text beside the results the
    /// runtime added (a recovery note), which [`RefusalResolution`] doesn't
    /// rephrase.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub runtime_text_messages: BTreeSet<usize>,
    /// Set after a refusal when the opt-in refusal fallback will answer the
    /// refused input: the refusal the fallback request answers. The next
    /// model request goes to the fallback model; saved so a task suspended
    /// in between still makes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_for: Option<arko_ai::Refusal>,
    /// Indices in `usage_history` of calls made to the refusal fallback
    /// model, which the cache monitor skips.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_calls: Vec<usize>,
    /// The plan-usage window last reported over the opt-in threshold, so a
    /// resumed task doesn't report (or pause for) it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_usage_threshold_reported: Option<PlanUsageWindowMark>,
    /// The task is paused by a monitor (plan usage or cache threshold)
    /// rather than an error; approving it adds no error note.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub monitor_pause: bool,
    /// Context the runtime has taken from its command channel
    /// ([`UserCommand::InjectContext`](crate::runtime::command::UserCommand::InjectContext))
    /// and not yet added to the conversation, in the order it was sent. It
    /// is added, as one workflow context update, at the next safe turn
    /// boundary, before any context sent later. A task that stops keeps here
    /// what could not be added before it stopped: in an awaiting state (a
    /// user message must not come between a call and its result), while a
    /// refusal is unresolved (a pending fallback answers exactly the refused
    /// input, and [`Task::resolve_refusal`] needs it to be the last input),
    /// and once completed (the model has not read it). Saved so a task that
    /// runs again still gets it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_context_injections: Vec<String>,
    /// The images of the context in `pending_context_injections` (see
    /// [`TurnContext::images`](crate::runtime::TurnContext::images)), added
    /// with it, after its text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_context_images: Vec<crate::runtime::ContextImage>,
    /// Set while the runtime waits for the agent's handover report; see
    /// [`crate::handover`]. Saved so a task stopped in between still hands
    /// over when it runs again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handover_requested: Option<HandoverRequest>,
    /// The handovers this task has made, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub handovers: Vec<HandoverRecord>,
    /// Index in `usage_history` of the current conversation's first model
    /// call: 0 until a handover. Calls before it belong to replaced
    /// conversations.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub conversation_usage_start: usize,
    /// The same index in `model_call_usage_history`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub conversation_model_call_start: usize,
    /// The conversation was cut back ([`Task::cut_back`]) since its latest
    /// model call, so that call's size is no longer the conversation's:
    /// [`current_conversation_context_size`](Self::current_conversation_context_size)
    /// is `None` until the next call. The conversation's first call stays
    /// where it was.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub context_size_unknown: bool,
    /// Index in `usage_history` of the first call the cache monitor judges:
    /// the conversation's first, or the first after a change of the prompt
    /// prefix ([`Task::reconfigure`]), whose full cache write is expected.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_window_start: usize,
    /// The Claude Code session holding the conversation of a task whose
    /// model runs in Claude Code, resumed when the task runs again. A
    /// handover starts a new one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code_session: Option<String>,
    /// The directory the Claude Code CLI runs in for the task's session,
    /// when it differs from the task's working directory: the one the
    /// session started in, which Claude Code finds it from. Set when the
    /// task moves ([`Task::move_to`]). arko's tools and checks work in
    /// [`current_working_directory`](Self::current_working_directory), and
    /// the CLI is given that directory too. A new session starts in the
    /// working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code_directory: Option<PathBuf>,
    /// How many messages of the conversation the Claude Code session has
    /// received or produced. User messages after them (a follow-up added to
    /// a completed task, say) are sent when the task runs again.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub claude_code_synced: usize,
    /// The system prompt changed after the Claude Code session recorded
    /// its first one ([`Task::reconfigure`]): the session's launches pass
    /// `--system-prompt-snapshot off`, so that it gets the current prompt.
    /// A handover's fresh session records the current one.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub claude_code_prompt_changed: bool,
    /// For each message of the conversation whose entry in the Claude Code
    /// session's transcript the runtime saw, by its index, the UUID of its
    /// last entry there: a model response, a turn's tool results, or the
    /// input the task sent. [`Task::cut_back`] resumes the session at one.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub claude_code_entries: BTreeMap<usize, String>,
    /// The entry of the Claude Code session that its next launch resumes
    /// at, after [`Task::cut_back`]: the transcript after it is left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code_resume_at: Option<String>,
    /// The Claude Code session that the next launch copies into the task's
    /// own session ([`claude_code_session`](Self::claude_code_session)), for
    /// a task made by [`Task::branch`]. The original is left unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code_fork_of: Option<String>,
    /// The task runs with a configuration its host gave it after it started
    /// ([`Task::reconfigure`]), not the one it was started with. A workflow
    /// resumed from a checkpoint keeps it, without comparing it with the
    /// agent's registered configuration.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reconfigured: bool,
    /// The tool calls that started running and have no recorded result yet,
    /// in the order they started. The runtime records a call here before it
    /// runs (and reports the task to its
    /// [`TaskSnapshotSink`](crate::snapshot::TaskSnapshotSink) before the
    /// call starts), and drops it once the call's result is in the task. A
    /// task saved while calls ran (by a host that crashed) still lists them:
    /// when the task runs again, each is settled by its tool's
    /// [`recovery_mode`](crate::tool::Tool::recovery_mode) before anything
    /// else happens.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub running_tool_calls: Vec<RunningToolCall>,
    /// The task and tool call this task runs inside, set by a host that
    /// runs a task within another task's tool call (a workflow's delegated
    /// agent or completion verifier), so that after a crash the host finds
    /// the task again when it recovers that call. The runtime doesn't use
    /// it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<TaskParent>,
}

/// The task and tool call a task runs inside; see [`Task::parent`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TaskParent {
    /// The ID of the task whose tool call runs this task.
    pub task_id: Uuid,
    /// The runtime's ID of that call
    /// ([`TaskContext::tool_call_id`](crate::tool::TaskContext::tool_call_id)),
    /// which a recovered call keeps.
    pub tool_call_id: Uuid,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

/// A user message's content: `text`, then `images`.
impl Task {
    /// The images that go with the initial prompt: `initial_task_images`
    /// before the first message is written, then the images right after the
    /// text of the conversation's first message (a fresh conversation's
    /// first message after a handover restates the task with them).
    pub fn initial_images(&self) -> Vec<arko_ai::Image> {
        if !self.initial_task_images.is_empty() {
            return self.initial_task_images.clone();
        }
        let Some(first) = self.conversation.messages.first() else {
            return Vec::new();
        };
        first
            .content
            .iter()
            .skip_while(|content| content.as_text().is_some())
            .map_while(|content| content.as_image().cloned())
            .collect()
    }
}

pub(crate) fn with_images(
    text: impl Into<String>,
    images: impl IntoIterator<Item = arko_ai::Image>,
) -> Vec<arko_ai::Content> {
    std::iter::once(arko_ai::Content::Text(text.into()))
        .chain(images.into_iter().map(arko_ai::Content::Image))
        .collect()
}

/// A context update's content: `text`, then each image after its label.
pub(crate) fn with_context_images(
    text: impl Into<String>,
    images: impl IntoIterator<Item = crate::runtime::ContextImage>,
) -> Vec<arko_ai::Content> {
    std::iter::once(arko_ai::Content::Text(text.into()))
        .chain(context_image_content(images))
        .collect()
}

/// Each image after its label, as message content.
pub(crate) fn context_image_content(
    images: impl IntoIterator<Item = crate::runtime::ContextImage>,
) -> impl Iterator<Item = arko_ai::Content> {
    images.into_iter().flat_map(|image| {
        [
            arko_ai::Content::Text(image.label),
            arko_ai::Content::Image(image.image),
        ]
    })
}

/// A tool call that started running and has no recorded result yet; see
/// [`Task::running_tool_calls`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunningToolCall {
    /// The runtime's id of the call, as the tool
    /// ([`TaskContext::tool_call_id`](crate::tool::TaskContext::tool_call_id)),
    /// the execution middleware and the call's events see it. A recovered
    /// call keeps it.
    pub tool_call_id: Uuid,
    /// The model's id of the call (the provider's tool-call id, or Claude
    /// Code's `tool_use_id`), which its result answers. Output-based calls
    /// have none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// The tool's name (the arko tool's, or a Claude Code built-in's).
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters: HashMap<String, Value>,
    /// Output the host gave with the call's approval instead of running the
    /// tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_output: Option<String>,
    /// The host's message given with the call's approval, added to its
    /// result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_to_agent: Option<String>,
    /// A Claude Code built-in (Claude Code runs it, not arko).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub built_in: bool,
    /// Claude Code: the arko call's result once it finished, until the
    /// session reports the result (and the conversation records it). A
    /// call recovered with its output completed before the host stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Images belonging to the completed Claude Code result in `output`.
    /// Kept until its transcript has the result, and replayed on recovery.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_images: Vec<crate::tool::ToolImage>,
}

/// A failed tool call the task will pause on; see
/// [`Task::deferred_tool_errors`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeferredToolError {
    /// The error, as `AgentEvent::ErrorEncountered` reports it.
    pub message: String,
    pub tool_call_id: Option<Uuid>,
    /// A pause the execution middleware asked for or caused with a hook error
    /// ([`ErrorRecoveryCause::Middleware`]) rather than a tool error. The
    /// pause has that cause when any of its entries is one.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub middleware_pause: bool,
}

/// A plan-usage window and its reset; see
/// [`Task::plan_usage_threshold_reported`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanUsageWindowMark {
    /// `"primary"` or `"secondary"`.
    pub window: String,
    /// The window's reset, in Unix seconds, when reported.
    pub resets_at: Option<u64>,
}

impl Task {
    /// Creates a new task with the given configuration.
    pub fn new(
        agent_config: AgentConfig,
        initial_task_prompt: String,
        initial_cwd: PathBuf,
        developer_env_details: Option<Vec<DeveloperDefinableEnvironmentDetail>>,
    ) -> Self {
        let task_id = Uuid::new_v4();
        let agent_config_id = agent_config.id.clone();
        let conversation = arko_ai::Conversation::new(agent_config.llm_config.model.clone());

        Self {
            id: task_id,
            agent_config_id,
            agent_config,
            initial_task_prompt,
            initial_task_images: Vec::new(),
            conversation,
            current_state: TaskState::Pending,
            start_time: Utc::now(),
            end_time: None,
            developer_env_details,
            usage_history: Vec::new(),
            model_call_usage_history: Vec::new(),
            current_working_directory: initial_cwd,
            user_message_count: 0,
            pending_native_tool_results: Vec::new(),
            pending_native_tool_calls: HashMap::new(),
            pending_tool_images: Vec::new(),
            pending_text_tool_results: Vec::new(),
            deferred_tool_errors: Vec::new(),
            refused_input: None,
            runtime_messages: BTreeSet::new(),
            runtime_text_messages: BTreeSet::new(),
            fallback_for: None,
            fallback_calls: Vec::new(),
            plan_usage_threshold_reported: None,
            monitor_pause: false,
            pending_context_injections: Vec::new(),
            pending_context_images: Vec::new(),
            handover_requested: None,
            handovers: Vec::new(),
            conversation_usage_start: 0,
            conversation_model_call_start: 0,
            context_size_unknown: false,
            cache_window_start: 0,
            claude_code_session: None,
            claude_code_directory: None,
            claude_code_synced: 0,
            claude_code_prompt_changed: false,
            claude_code_entries: BTreeMap::new(),
            claude_code_resume_at: None,
            claude_code_fork_of: None,
            reconfigured: false,
            running_tool_calls: Vec::new(),
            parent: None,
        }
    }

    /// Records a model call's usage metrics, computing the input token delta
    /// from the previous call in this task's history.
    ///
    /// Returns the usage with `input_delta_from_previous_call` populated.
    pub fn record_model_call_usage(&mut self, mut usage: ModelCallUsage) -> ModelCallUsage {
        // The first call of a conversation that replaced another has no
        // previous call to compare with. A subagent's call (Claude Code's
        // `Agent` tool) is compared with that subagent's, and the
        // conversation's own calls with each other.
        let previous = self
            .model_call_usage_history
            .get(self.conversation_model_call_start..)
            .and_then(|calls| {
                calls
                    .iter()
                    .rev()
                    .find(|call| call.subagent == usage.subagent)
            });
        if let Some(previous) = previous {
            let delta = usage.input_tokens as isize - previous.input_tokens as isize;
            usage = usage.with_input_delta(delta);
        }
        self.model_call_usage_history.push(usage.clone());
        usage
    }

    /// Loads a task from a JSON file.
    pub fn load_from_path(path: &Path) -> Result<Self, ArkoAgentsError> {
        let file_content =
            std::fs::read_to_string(path).map_err(|e| ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            })?;
        serde_json::from_str(&file_content).map_err(ArkoAgentsError::Json)
    }

    /// Saves the task to a JSON file, replacing it in one step (see
    /// [`write_file_atomically`](crate::utils::write_file_atomically)), so
    /// that a crash during the save leaves the previous file whole.
    pub fn save_to_path(&self, path: &Path) -> Result<(), ArkoAgentsError> {
        let json_string = serde_json::to_string_pretty(self)?;
        crate::utils::write_file_atomically(path, json_string.as_bytes()).map_err(|e| {
            ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            }
        })
    }

    /// Gets the runtime duration of the task, if it has ended.
    pub fn get_run_time(&self) -> Option<chrono::Duration> {
        self.end_time.map(|et| et - self.start_time)
    }

    /// Sets the task state, handling terminal state transitions appropriately.
    pub fn set_state(&mut self, new_state: TaskState) {
        let is_already_terminal = self.current_state.is_terminal() && self.end_time.is_some();
        let new_state_is_terminal = new_state.is_terminal();

        self.current_state = new_state;

        if new_state_is_terminal {
            if !is_already_terminal {
                self.end_time = Some(Utc::now());
            }
        } else {
            self.end_time = None;
        }
    }

    /// Continues the task with `config` in place of its configuration: its
    /// next run offers `config`'s tools and writes a system message from
    /// `config` in place of the current one. The conversation, with its
    /// prompt cache key, and the task's usage stay. Between turns only (a
    /// task pending, completed, or awaiting the user's answer), and with the
    /// same model (provider and ID): `InvalidTaskState` otherwise, and
    /// `config`'s own validation errors. A Claude Code task's session gets
    /// the new prompt and flags when it resumes
    /// ([`claude_code_prompt_changed`](Self::claude_code_prompt_changed)).
    /// The new tools and prompt are a different prefix, so the next request
    /// writes the provider's prompt cache afresh.
    pub fn reconfigure(&mut self, config: AgentConfig) -> Result<(), ArkoAgentsError> {
        if !matches!(
            self.current_state,
            TaskState::Pending
                | TaskState::Completed { .. }
                | TaskState::AwaitingUserResponse { .. }
        ) {
            return Err(ArkoAgentsError::InvalidTaskState(format!(
                "a task is reconfigured only between turns (pending, completed, or awaiting the user's answer), not {:?}",
                self.current_state
            )));
        }
        let (current, next) = (
            &self.agent_config.llm_config.model,
            &config.llm_config.model,
        );
        if current.provider != next.provider || current.id != next.id {
            return Err(ArkoAgentsError::InvalidTaskState(format!(
                "a reconfigured task keeps its model ({:?} '{}'), not '{}' of {:?}",
                current.provider, current.id, next.id, next.provider
            )));
        }
        config.validate()?;
        self.agent_config_id = config.id.clone();
        self.agent_config = config;
        self.conversation.system_message = None;
        self.reconfigured = true;
        // The new prompt prefix is written to the cache afresh: the cache
        // monitor judges the calls from here.
        self.cache_window_start = self.usage_history.len();
        if self.claude_code_session.is_some() {
            self.claude_code_prompt_changed = true;
        }
        Ok(())
    }

    /// Moves the task to working directory `directory`: its tools work
    /// there from its next run, and its system prompt (which may name the
    /// directory) is written again. A Claude Code task with a session keeps
    /// running the CLI in the session's directory, where Claude Code finds
    /// the session ([`claude_code_directory`](Self::claude_code_directory)),
    /// and the session gets the new prompt.
    pub fn move_to(&mut self, directory: PathBuf) {
        if directory == self.current_working_directory {
            return;
        }
        if self.claude_code_session.is_some() {
            self.claude_code_directory
                .get_or_insert_with(|| self.current_working_directory.clone());
            self.claude_code_prompt_changed = true;
        }
        if self.claude_code_directory.as_ref() == Some(&directory) {
            self.claude_code_directory = None;
        }
        self.current_working_directory = directory;
        self.conversation.system_message = None;
    }

    /// Prepares a task that was 'Completed' for resumption.
    /// Sets its state to 'InProgress' and adds the continuation prompt to the conversation.
    pub fn prepare_for_resumption_from_completed(
        &mut self,
        continuation_prompt: String,
    ) -> Result<(), ArkoAgentsError> {
        self.prepare_for_resumption_from_completed_with_images(continuation_prompt, Vec::new())
    }

    /// [`prepare_for_resumption_from_completed`](Self::prepare_for_resumption_from_completed)
    /// with `images` after the prompt, in the same message.
    pub fn prepare_for_resumption_from_completed_with_images(
        &mut self,
        continuation_prompt: String,
        images: Vec<arko_ai::Image>,
    ) -> Result<(), ArkoAgentsError> {
        if !matches!(self.current_state, TaskState::Completed { .. }) {
            return Err(ArkoAgentsError::InvalidTaskState(
                "Task must be in a 'Completed' state to be resumed this way.".to_string(),
            ));
        }
        self.set_state(TaskState::InProgress);
        // A report requested before the task completed is no longer due.
        self.handover_requested = None;
        // A continuation is a user message just like an answer to a paused
        // task. Keep the per-message environment details and reminder rules
        // consistent across both paths.
        let parsing = self.agent_config.get_tool_parsing_config();
        let continuation =
            crate::parsing::PromptBuilder::new(&self.agent_config, vec![], &parsing, None)
                .augment_user_message_with_env_details_and_reminder(
                    self,
                    &continuation_prompt,
                    self.user_message_count + 1,
                )?;
        self.conversation
            .add_user_message_with_contents(with_images(continuation, images));
        self.user_message_count += 1;
        Ok(())
    }

    /// Resume a task suspended by `ask_user` with the application's answer.
    /// Tool results of the question's turn that are still waiting (a task
    /// saved by an older version can hold an approved `ask_user`'s result)
    /// are added with the answer, so every call keeps its result.
    pub fn prepare_for_resumption_with_user_response(
        &mut self,
        response: String,
    ) -> Result<(), ArkoAgentsError> {
        self.prepare_for_resumption_with_user_response_and_images(response, Vec::new())
    }

    /// [`prepare_for_resumption_with_user_response`](Self::prepare_for_resumption_with_user_response)
    /// with `images` after the answer, in the same message.
    pub fn prepare_for_resumption_with_user_response_and_images(
        &mut self,
        response: String,
        images: Vec<arko_ai::Image>,
    ) -> Result<(), ArkoAgentsError> {
        let questions = match &self.current_state {
            TaskState::AwaitingUserResponse { questions } => questions.clone(),
            _ => {
                return Err(ArkoAgentsError::InvalidTaskState(
                    "Task must be awaiting user input to resume with a response".to_string(),
                ));
            }
        };
        self.set_state(TaskState::InProgress);
        let answer = self.with_pending_text_tool_results(format!(
            "Response to the agent's questions:\nQuestions: {questions}\nResponse: {response}"
        ));
        // With the environment details and reminder due, as the runtime
        // sends any user message.
        let parsing = self.agent_config.get_tool_parsing_config();
        let answer = crate::parsing::PromptBuilder::new(&self.agent_config, vec![], &parsing, None)
            .text_beside_tool_results(self, Some(&answer), self.user_message_count + 1)?
            .unwrap_or(answer);
        self.add_pending_tool_results_with_images(Some(&answer), images);
        Ok(())
    }

    /// `message` after the output-based tool results still waiting (see
    /// [`Task::pending_text_tool_results`]), which it takes.
    pub(crate) fn with_pending_text_tool_results(&mut self, message: String) -> String {
        let mut parts = std::mem::take(&mut self.pending_text_tool_results);
        if parts.is_empty() {
            return message;
        }
        if !message.is_empty() {
            parts.push(message);
        }
        parts.join("\n\n---\n\n")
    }

    /// Adds the tool results waiting for this turn to the conversation as
    /// one user message: the native results, then `text`, then the tool
    /// images, each after its description. Adds nothing when there is
    /// nothing to add; returns whether a message was added.
    pub(crate) fn add_pending_tool_results(&mut self, text: Option<&str>) -> bool {
        self.add_pending_tool_results_with_images(text, Vec::new())
    }

    /// [`add_pending_tool_results`](Self::add_pending_tool_results) with
    /// the user's `images` right after `text`, before the tool images.
    fn add_pending_tool_results_with_images(
        &mut self,
        text: Option<&str>,
        images: Vec<arko_ai::Image>,
    ) -> bool {
        let results = std::mem::take(&mut self.pending_native_tool_results);
        if results.is_empty()
            && text.is_none()
            && images.is_empty()
            && self.pending_tool_images.is_empty()
        {
            return false;
        }
        let mut content: Vec<_> = results
            .into_iter()
            .map(arko_ai::Content::ToolResult)
            .collect();
        if let Some(text) = text {
            content.push(arko_ai::Content::Text(text.to_string()));
        }
        content.extend(images.into_iter().map(arko_ai::Content::Image));
        content.extend(self.take_tool_image_content());
        self.conversation.add_user_message_with_contents(content);
        self.increment_user_message_count();
        true
    }

    /// The waiting tool images, each after its description, as message
    /// content.
    pub(crate) fn take_tool_image_content(&mut self) -> Vec<arko_ai::Content> {
        std::mem::take(&mut self.pending_tool_images)
            .into_iter()
            .flat_map(|attachment| {
                [
                    arko_ai::Content::Text(attachment.description),
                    arko_ai::Content::Image(attachment.image),
                ]
            })
            .collect()
    }

    /// Returns the summed usage of every LLM call in this task.
    pub fn total_usage(&self) -> UsageInfo {
        self.usage_history.iter().sum()
    }

    /// Returns the amount billed across all LLM usage for this task.
    /// Subscription (ChatGPT OAuth) calls contribute nothing; see
    /// [`UsageInfo::billed`].
    pub fn total_usage_cost(&self) -> f64 {
        self.total_usage().billed()
    }

    /// Returns the total input tokens (including cached input) across all LLM
    /// usage for this task.
    pub fn total_input_tokens(&self) -> usize {
        self.total_usage().input_tokens
    }

    /// Returns the total output tokens across all LLM usage for this task.
    pub fn total_output_tokens(&self) -> usize {
        self.total_usage().output_tokens
    }

    /// Returns the context size of the most recent LLM call, if available:
    /// its total input tokens, including cached input.
    pub fn current_context_size(&self) -> Option<usize> {
        self.usage_history.last().map(|u| u.input_tokens)
    }

    /// The context size of the current conversation's latest model call:
    /// [`current_context_size`](Self::current_context_size), but `None`
    /// after a handover until the fresh conversation's first call.
    pub fn current_conversation_context_size(&self) -> Option<usize> {
        self.latest_conversation_usage()
            .map(|usage| usage.input_tokens)
    }

    /// The usage of the current conversation's latest model call: `None`
    /// before its first, and after a cut back until the next call.
    pub fn latest_conversation_usage(&self) -> Option<&UsageInfo> {
        if self.context_size_unknown {
            return None;
        }
        self.usage_history
            .get(self.conversation_usage_start..)
            .and_then(<[UsageInfo]>::last)
    }

    /// The first call the cache monitor judges: the conversation's first,
    /// or the first after a reconfiguration.
    pub fn cache_monitor_start(&self) -> usize {
        self.conversation_usage_start.max(self.cache_window_start)
    }

    /// Records a model call's usage: the conversation's size is known again.
    pub fn record_usage(&mut self, usage: UsageInfo) {
        self.usage_history.push(usage);
        self.context_size_unknown = false;
    }

    /// Replaces the conversation with an empty one that keeps the system
    /// message, cache settings and model, and returns how many messages were
    /// replaced. Clears what refers to the old messages (the runtime-message
    /// indices, a refused input, the ChatGPT turn state) and the user
    /// message count, and marks where the fresh conversation's usage starts.
    /// Pending context injections are kept, to be added to the fresh
    /// conversation. The runtime calls this during a handover; the caller
    /// adds the fresh conversation's first message.
    pub fn start_fresh_conversation(&mut self) -> usize {
        let replaced = self.conversation.messages.len();
        self.conversation.messages.clear();
        self.conversation.openai_turn_state = None;
        self.runtime_messages.clear();
        self.runtime_text_messages.clear();
        self.refused_input = None;
        self.user_message_count = 0;
        self.conversation_usage_start = self.usage_history.len();
        self.conversation_model_call_start = self.model_call_usage_history.len();
        replaced
    }

    /// Increments the user message count and returns the new count.
    pub fn increment_user_message_count(&mut self) -> usize {
        self.user_message_count += 1;
        self.user_message_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{
        AgentLLMConfig, DeveloperDefinableEnvironmentDetail, EnvironmentDetailPlacement,
        UserReminder,
    };
    use crate::runtime::event::AgentEvent;

    fn state(json: Value) -> TaskState {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn pauses_and_failures_saved_without_an_error_kind_still_load() {
        use arko_ai::ErrorKind;
        // Saved before error kinds existed.
        assert_eq!(
            state(
                serde_json::json!({"AwaitingErrorRecoveryApproval": {"error_message": "401", "cause": "non_retryable"}})
            ),
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "401".into(),
                cause: ErrorRecoveryCause::NonRetryable,
                error_kind: None,
            }
        );
        assert_eq!(
            state(serde_json::json!({"Failed": {"reason": "401", "cause": "non_retryable"}})),
            TaskState::Failed {
                reason: "401".into(),
                cause: Some(ErrorRecoveryCause::NonRetryable),
                error_kind: None,
            }
        );
        // Without a kind, nothing is written for it, so older versions read
        // the state as before; with one, it is kept.
        let paused = TaskState::AwaitingErrorRecoveryApproval {
            error_message: "too long".into(),
            cause: ErrorRecoveryCause::NonRetryable,
            error_kind: Some(ErrorKind::ContextOverflow),
        };
        let saved = serde_json::to_value(&paused).unwrap();
        assert_eq!(
            saved["AwaitingErrorRecoveryApproval"]["error_kind"],
            "context_overflow"
        );
        assert_eq!(state(saved), paused);
        let failed = serde_json::to_value(TaskState::Failed {
            reason: "x".into(),
            cause: Some(ErrorRecoveryCause::NonRetryable),
            error_kind: None,
        })
        .unwrap();
        assert!(failed["Failed"].get("error_kind").is_none(), "{failed}");
        // A kind a later version adds loads as Unknown.
        assert_eq!(
            state(
                serde_json::json!({"Failed": {"reason": "x", "cause": "non_retryable", "error_kind": "from_the_future"}})
            ),
            TaskState::Failed {
                reason: "x".into(),
                cause: Some(ErrorRecoveryCause::NonRetryable),
                error_kind: Some(ErrorKind::Unknown),
            }
        );
    }

    #[test]
    fn a_cancellation_saved_as_a_failure_loads_as_cancelled() {
        assert_eq!(
            state(serde_json::json!({"Failed": {"reason": "Task cancelled by user."}})),
            TaskState::Cancelled {
                reason: CANCELLED_BY_USER.to_string()
            }
        );
        // A failure without a cause is saved as before, so older versions
        // read it; one with a cause keeps it.
        let failed = |cause| TaskState::Failed {
            reason: "x".into(),
            cause,
            error_kind: None,
        };
        assert_eq!(
            serde_json::to_value(failed(None)).unwrap(),
            serde_json::json!({"Failed": {"reason": "x"}})
        );
        assert_eq!(
            state(serde_json::json!({"Failed": {"reason": "x", "cause": "non_retryable"}})),
            failed(Some(ErrorRecoveryCause::NonRetryable))
        );
        // Only the exact reason the runtime recorded is a cancellation.
        for reason in [
            "Task cancelled by user",
            "Tool call cancelled by the server.",
        ] {
            assert_eq!(
                state(serde_json::json!({"Failed": {"reason": reason}})),
                TaskState::Failed {
                    reason: reason.to_string(),
                    cause: None,
                    error_kind: None,
                }
            );
        }
    }

    #[test]
    fn every_state_round_trips() {
        let states = [
            TaskState::Pending,
            TaskState::InProgress,
            TaskState::AwaitingToolApproval {
                tool_call_id: Uuid::new_v4(),
                tool_name: "write_file".into(),
                parameters: HashMap::from([("path".into(), serde_json::json!("a.txt"))]),
                tool_description: "Writes".into(),
                approval_reason: ApprovalReason {
                    tool_not_auto_approved: true,
                    path_outside_workspace: Some("path".into()),
                },
            },
            TaskState::AwaitingBatchToolApproval {
                batch_id: Uuid::new_v4(),
                pending_calls: vec![PendingToolCallInfo {
                    tool_call_id: Uuid::new_v4(),
                    tool_name: "read_file".into(),
                    parameters: HashMap::new(),
                    tool_description: "Reads".into(),
                    approval_reason: ApprovalReason {
                        tool_not_auto_approved: true,
                        path_outside_workspace: None,
                    },
                }],
            },
            TaskState::AwaitingUserResponse {
                questions: "Which?".into(),
            },
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "boom".into(),
                cause: ErrorRecoveryCause::UsageLimit {
                    resets_at_unix_ms: Some(1_790_000_000_000),
                },
                error_kind: None,
            },
            TaskState::Interrupted {
                reason: "network".into(),
            },
            TaskState::Completed {
                summary: "done".into(),
            },
            TaskState::Failed {
                reason: "verifier rejected".into(),
                cause: None,
                error_kind: None,
            },
            TaskState::Failed {
                reason: "a new credential is needed".into(),
                cause: Some(ErrorRecoveryCause::NonRetryable),
                error_kind: Some(arko_ai::ErrorKind::Authentication),
            },
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "prompt is too long".into(),
                cause: ErrorRecoveryCause::NonRetryable,
                error_kind: Some(arko_ai::ErrorKind::ContextOverflow),
            },
            TaskState::Cancelled {
                reason: "Workflow cancelled.".into(),
            },
        ];
        for original in states {
            // Naming every variant here makes a new one fail to compile until
            // it is added to this list (and to `TaskStateRepr`).
            match &original {
                TaskState::Pending
                | TaskState::InProgress
                | TaskState::AwaitingToolApproval { .. }
                | TaskState::AwaitingBatchToolApproval { .. }
                | TaskState::AwaitingUserResponse { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. }
                | TaskState::Interrupted { .. }
                | TaskState::Completed { .. }
                | TaskState::Failed { .. }
                | TaskState::Cancelled { .. } => {}
            }
            let json = serde_json::to_value(&original).unwrap();
            assert_eq!(state(json), original);
        }
        assert_eq!(
            serde_json::to_value(TaskState::Cancelled {
                reason: CANCELLED_BY_USER.into()
            })
            .unwrap(),
            serde_json::json!({"Cancelled": {"reason": CANCELLED_BY_USER}})
        );
        assert!(
            TaskState::Cancelled {
                reason: String::new()
            }
            .is_terminal()
        );
    }

    /// Approval and error-recovery states saved before their reasons and
    /// causes were recorded load with them unrecorded.
    #[test]
    fn awaiting_states_saved_without_reasons_load() {
        let call = Uuid::new_v4();
        assert_eq!(
            state(serde_json::json!({"AwaitingToolApproval": {
                "tool_call_id": call, "tool_name": "write_file",
                "parameters": {}, "tool_description": "Writes"
            }})),
            TaskState::AwaitingToolApproval {
                tool_call_id: call,
                tool_name: "write_file".into(),
                parameters: HashMap::new(),
                tool_description: "Writes".into(),
                approval_reason: ApprovalReason::default(),
            }
        );
        let TaskState::AwaitingBatchToolApproval { pending_calls, .. } =
            state(serde_json::json!({"AwaitingBatchToolApproval": {
                "batch_id": Uuid::new_v4(),
                "pending_calls": [{"tool_call_id": call, "tool_name": "read_file",
                    "parameters": {}, "tool_description": "Reads"}]
            }}))
        else {
            panic!("a batch approval state");
        };
        assert!(pending_calls[0].approval_reason.is_unrecorded());
        assert_eq!(
            state(serde_json::json!({"AwaitingErrorRecoveryApproval": {"error_message": "boom"}})),
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "boom".into(),
                cause: ErrorRecoveryCause::Unknown,
                error_kind: None,
            }
        );
        // An unrecorded reason is left out when saved.
        let saved = serde_json::to_value(&pending_calls[0]).unwrap();
        assert!(saved.get("approval_reason").is_none());
    }

    /// Context taken but not yet added is saved only when there is some, in
    /// order, and a task saved before the field existed loads without it.
    #[test]
    fn context_taken_but_not_added_is_saved_with_the_task() {
        let config = AgentConfig::builder(
            "saved".into(),
            "Saved".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let task = Task::new(config, "prompt".into(), PathBuf::from("/workspace"), None);
        let json = serde_json::to_value(&task).unwrap();
        assert!(json.get("pending_context_injections").is_none());
        let mut loaded: Task = serde_json::from_value(json).unwrap();
        assert!(loaded.pending_context_injections.is_empty());

        loaded.pending_context_injections = vec!["First.".into(), "Second.".into()];
        let json = serde_json::to_value(&loaded).unwrap();
        assert_eq!(
            json["pending_context_injections"],
            serde_json::json!(["First.", "Second."])
        );
        let loaded: Task = serde_json::from_str(&json.to_string()).unwrap();
        assert_eq!(loaded.pending_context_injections, ["First.", "Second."]);
    }

    #[test]
    fn completed_continuations_apply_environment_details_and_reminders() {
        let config = AgentConfig::builder(
            "continued".into(),
            "Continued".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let detail = DeveloperDefinableEnvironmentDetail::new_static(
            "Build".into(),
            "green".into(),
            EnvironmentDetailPlacement::EachUserMessage,
        );
        let mut task = Task::new(
            config,
            "prompt".into(),
            PathBuf::from("/workspace"),
            Some(vec![detail]),
        );
        task.agent_config.user_reminder = Some(UserReminder::every_message("Stay focused.".into()));
        task.conversation.add_user_message("Initial");
        task.user_message_count = 1;
        task.set_state(TaskState::Completed {
            summary: "done".into(),
        });
        task.prepare_for_resumption_from_completed("Continue".into())
            .unwrap();
        let message = task.conversation.messages.last().unwrap();
        let text = message
            .content
            .iter()
            .find_map(arko_ai::Content::as_text)
            .unwrap();
        assert!(
            text.contains("# Current Environment Information (updated this turn)\nBuild:\ngreen"),
            "{text}"
        );
        assert!(text.contains("# Reminder\nStay focused."));
        assert!(task.pending_context_injections.is_empty());
    }

    /// A new configuration without a tool calling mode gets native; a saved
    /// task records its mode, and one saved without it (by a version whose
    /// default was output-based) keeps output-based.
    #[test]
    fn saved_tasks_keep_their_tool_calling_mode_and_new_configs_default_to_native() {
        use crate::agent_config::ToolCallingMode;
        let config = serde_json::json!({
            "id": "a", "name": "A", "role_description": "r", "instructions": "i",
            "llm_config": {"provider": "OpenAI", "model_id": "gpt-5.6-luna"},
            "tool_config": {"available_tool_names": ["read_file"]}
        });
        let loaded: AgentConfig = serde_json::from_value(config.clone()).unwrap();
        assert_eq!(
            loaded.tool_config.tool_calling_mode,
            ToolCallingMode::Native
        );
        let mut bare = config.clone();
        bare.as_object_mut().unwrap().remove("tool_config");
        let loaded: AgentConfig = serde_json::from_value(bare.clone()).unwrap();
        assert_eq!(
            loaded.tool_config.tool_calling_mode,
            ToolCallingMode::Native
        );
        let yaml = "id: a\nname: A\nrole_description: r\ninstructions: i\nllm_config:\n  provider: OpenAI\n  model_id: gpt-5.6-luna\n";
        let loaded: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            loaded.tool_config.tool_calling_mode,
            ToolCallingMode::Native
        );

        let task = Task::new(loaded, "prompt".into(), PathBuf::from("/workspace"), None);
        let mut saved = serde_json::to_value(&task).unwrap();
        assert_eq!(
            saved["agent_config"]["tool_config"]["tool_calling_mode"],
            "Native"
        );
        assert_eq!(
            serde_json::from_value::<Task>(saved.clone())
                .unwrap()
                .agent_config
                .tool_config
                .tool_calling_mode,
            ToolCallingMode::Native
        );
        for old_config in [config, bare] {
            saved["agent_config"] = old_config;
            let old: Task = serde_json::from_value(saved.clone()).unwrap();
            assert_eq!(
                old.agent_config.tool_config.tool_calling_mode,
                ToolCallingMode::OutputBased
            );
        }
        // Loading a task doesn't change what a configuration loaded next gets.
        let after: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(after.tool_config.tool_calling_mode, ToolCallingMode::Native);
        assert!(serde_json::from_value::<Task>(serde_json::json!({"agent_config": {}})).is_err());
        let after: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(after.tool_config.tool_calling_mode, ToolCallingMode::Native);
    }

    #[test]
    fn saved_tasks_and_events_from_before_the_cancelled_state_still_load() {
        let config = AgentConfig::builder(
            "saved".into(),
            "Saved".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let mut task = Task::new(config, "prompt".into(), PathBuf::from("/workspace"), None);
        task.set_state(TaskState::Completed {
            summary: String::new(),
        });
        let mut json = serde_json::to_value(&task).unwrap();
        json["current_state"] =
            serde_json::json!({"Failed": {"reason": "Task cancelled by user."}});
        let loaded: Task = serde_json::from_str(&json.to_string()).unwrap();
        assert_eq!(
            loaded.current_state,
            TaskState::Cancelled {
                reason: CANCELLED_BY_USER.to_string()
            }
        );
        // Saved in whole seconds.
        assert_eq!(
            loaded.end_time.map(|time| time.timestamp()),
            task.end_time.map(|time| time.timestamp())
        );

        let event: AgentEvent = serde_json::from_value(serde_json::json!({
            "TaskStateChanged": {"Failed": {"reason": "Task cancelled by user."}}
        }))
        .unwrap();
        assert!(matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::Cancelled { .. })
        ));
    }
}
