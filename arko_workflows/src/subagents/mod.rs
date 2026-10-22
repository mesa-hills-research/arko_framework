//! Background subagents are agent runs that an agent starts and that run while
//! it goes on with its own work.
//!
//! With [`WorkflowBuilder::subagents`](crate::WorkflowBuilder::subagents),
//! the agents [`WorkflowServicesConfig`](crate::services::WorkflowServicesConfig)
//! names as callers get tools to start runs of its targets in the background
//! (`start_agent`, which returns a run id such as `r1` at once), to message
//! them (`message_agent`), to wait for their reports and messages
//! (`wait_for_agents`), to list them (`list_agents`) and to stop them
//! (`stop_agent`). A subagent gets `message_agent` and `wait_for_agents`
//! to talk with the run that started it (its lead) and, when the lead
//! allows it, with the lead's other runs (its peers).
//!
//! A subagent is an ordinary run of the same workflow. Its costs go into the
//! workflow's ledger, its result is checkpointed with the others, and the
//! workflow's budget, policies, hooks and notes apply to it. It is not
//! inside a tool call of its lead, so the lead's run can end, or be stopped,
//! while it runs on. Stopping or cancelling the whole workflow stops it too.
//!
//! A subagent may start subagents of its own when
//! [`SubagentsConfig::max_depth`] allows it. Stopping a subagent, with
//! [`Workflow::stop_subagent`](crate::Workflow::stop_subagent) or its lead's
//! `stop_agent`, also stops the subagents it started that still run or wait
//! for a place, and theirs in turn, with the same wrap-up. A subagent that
//! ends in any other way stops them too, since they would report to a run
//! that is gone. Each of them is told which run's stop or end reached it.
//! A subagent isn't done while its own subagents run or wait for a place.
//! Its `complete_task` is answered with them and what to do, and doesn't
//! complete, unless the run is being stopped, which reports at once.
//!
//! Reports, failures, stops and messages go through a mailbox kept per run.
//! An item reaches its recipient in the first way that applies:
//! - a `wait_for_agents` call that waits for it returns it, whole.
//! - while the recipient runs, at its next turn boundary, in its
//!   `# Workflow context update` (a long item as an excerpt, with the file
//!   that holds all of it).
//! - otherwise it waits in the mailbox. A subagent that had finished is
//!   continued with it. For a run the host drives, the workflow sends
//!   [`SubagentEventKind::NewsWaiting`], and the host continues the run (with
//!   [`Workflow::continue_agent`](crate::Workflow::continue_agent), say),
//!   and the news arrives at the continuation's first turn boundary, or the
//!   host takes it with [`Workflow::take_news`](crate::Workflow::take_news).
//!
//! The runs, their mailboxes and their workspaces are saved in checkpoints.
//! After a resume, the runs that were running are `interrupted`.
//! [`Workflow::resume_subagents`](crate::Workflow::resume_subagents)
//! relaunches them, and a message to one restarts it.

mod board;
pub(crate) use board::CONTEXT_ARRIVED;
#[cfg(test)]
mod nesting_tests;
#[cfg(test)]
mod tests;
mod tools;
pub mod workspace;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use arko_agents::{AgentConfig, ToolParameterDefinition};

pub(crate) use board::Board;
pub use tools::{
    LINK_AGENTS_TOOL_NAME, LIST_AGENTS_TOOL_NAME, MESSAGE_AGENT_TOOL_NAME, START_AGENT_TOOL_NAME,
    STOP_AGENT_TOOL_NAME, SUBAGENT_TOOL_NAMES, WAIT_FOR_AGENTS_TOOL_NAME,
};
pub(crate) use tools::{
    LinkAgentsTool, ListAgentsTool, MessageAgentTool, StartAgentTool, StopAgentTool,
    WaitForAgentsTool,
};
pub use workspace::{GitWorktreeProvider, IsolatedWorkspace, WorkspaceProvider, WorkspaceRequest};

/// The default [`SubagentsConfig::max_running`] of 4.
pub const DEFAULT_MAX_RUNNING: usize = 4;

/// The default [`SubagentsConfig::max_peer_messages`] of 20.
pub const DEFAULT_MAX_PEER_MESSAGES: u32 = 20;

/// The longest mailbox item a context update shows whole, 16 KiB in bytes.
/// A longer one is shown as an excerpt, with the file that holds all of it
/// (`wait_for_agents` returns it whole).
pub const MAX_MAIL_BYTES: usize = 16 * 1024;

/// About this many bytes of mail go into one context update. The rest come
/// at the next turn boundary. The first item always goes.
pub const UPDATE_MAX_MAIL_BYTES: usize = 48 * 1024;

/// How background subagents work in a workflow. See the [module](self).
///
/// Who may start which agent is
/// [`WorkflowServicesConfig`](crate::services::WorkflowServicesConfig)'s
/// callers and targets.
#[derive(Clone)]
pub struct SubagentsConfig {
    /// At most this many subagents run at once, workflow-wide. A start (or a
    /// restart) over it waits in a queue, first come first served. `None` allows
    /// any number. A subagent that waits in `wait_for_agents` for subagents of
    /// its own holds no place while it waits, or the runs it waits for could
    /// never start. The number running may pass the limit by the waiters that
    /// woke since. Default [`DEFAULT_MAX_RUNNING`].
    pub max_running: Option<NonZeroUsize>,
    /// How deep subagents may nest. A run may start subagents while its own
    /// depth is less than this (a run the host starts has depth 0, its
    /// subagents depth 1). With the default of 1, subagents start no subagents and are
    /// not offered `run_workflow_agent`.
    pub max_depth: u32,
    /// The lead may let its subagents message each other (`start_agent`'s
    /// `peers` and `link_agents`). Default on.
    pub peer_messages: bool,
    /// How many messages two subagents may send each other between two of
    /// their lead's messages to either of them. Default
    /// [`DEFAULT_MAX_PEER_MESSAGES`].
    pub max_peer_messages: u32,
    /// The lead gets a copy of every message its subagents send each other.
    /// A copy doesn't end its waits or start a turn of it. Default off.
    pub copy_peer_messages_to_lead: bool,
    /// `start_agent`'s `workspace` may name any directory. By default it must
    /// be inside the caller's own working directory.
    pub workspaces_anywhere: bool,
    /// Callers are offered `run_workflow_agent` too (it runs an agent to
    /// completion inside the call). Default on.
    pub offer_blocking_tool: bool,
    /// A subagent's `ask_user` question goes to its lead as a message, and
    /// the lead's next message to it is the answer. On by default. When off, the
    /// question goes to the workflow's
    /// [`UserInputHandler`](crate::user_input::UserInputHandler), as other
    /// runs' questions do.
    pub questions_to_lead: bool,
    /// Creates and removes isolated workspaces (`start_agent`'s `isolated`).
    pub workspace_provider: Option<Arc<dyn WorkspaceProvider>>,
    /// Adjusts a subagent's configuration as it starts.
    pub hook: Option<Arc<dyn SubagentHook>>,
    /// Context for each run's first message, such as the project's
    /// instructions. See [`StartContext`].
    pub start_context: Option<StartContext>,
    /// The host's own `start_agent` parameters, whose values the
    /// [hook](Self::hook) receives.
    pub start_parameters: Vec<ToolParameterDefinition>,
}

impl Default for SubagentsConfig {
    fn default() -> Self {
        Self {
            max_running: NonZeroUsize::new(DEFAULT_MAX_RUNNING),
            max_depth: 1,
            peer_messages: true,
            max_peer_messages: DEFAULT_MAX_PEER_MESSAGES,
            copy_peer_messages_to_lead: false,
            workspaces_anywhere: false,
            offer_blocking_tool: true,
            questions_to_lead: true,
            workspace_provider: None,
            hook: None,
            start_context: None,
            start_parameters: Vec::new(),
        }
    }
}

impl std::fmt::Debug for SubagentsConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubagentsConfig")
            .field("max_running", &self.max_running)
            .field("max_depth", &self.max_depth)
            .field("peer_messages", &self.peer_messages)
            .field("max_peer_messages", &self.max_peer_messages)
            .field(
                "copy_peer_messages_to_lead",
                &self.copy_peer_messages_to_lead,
            )
            .field("workspaces_anywhere", &self.workspaces_anywhere)
            .field("offer_blocking_tool", &self.offer_blocking_tool)
            .field("questions_to_lead", &self.questions_to_lead)
            .field("workspace_provider", &self.workspace_provider.is_some())
            .field("hook", &self.hook.is_some())
            .field("start_context", &self.start_context.is_some())
            .field(
                "start_parameters",
                &self
                    .start_parameters
                    .iter()
                    .map(|parameter| &parameter.name)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SubagentsConfig {
    /// At most `max` subagents run at once (at least 1). See
    /// [`max_running`](Self::max_running).
    pub fn max_running(mut self, max: usize) -> Self {
        self.max_running = NonZeroUsize::new(max.max(1));
        self
    }

    /// Any number of subagents may run at once.
    pub fn unlimited(mut self) -> Self {
        self.max_running = None;
        self
    }

    /// See [`max_depth`](Self::max_depth).
    pub fn max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// See [`peer_messages`](Self::peer_messages).
    pub fn peer_messages(mut self, enabled: bool) -> Self {
        self.peer_messages = enabled;
        self
    }

    /// See [`max_peer_messages`](Self::max_peer_messages).
    pub fn max_peer_messages(mut self, max: u32) -> Self {
        self.max_peer_messages = max;
        self
    }

    /// See [`copy_peer_messages_to_lead`](Self::copy_peer_messages_to_lead).
    pub fn copy_peer_messages_to_lead(mut self, enabled: bool) -> Self {
        self.copy_peer_messages_to_lead = enabled;
        self
    }

    /// See [`workspaces_anywhere`](Self::workspaces_anywhere).
    pub fn workspaces_anywhere(mut self) -> Self {
        self.workspaces_anywhere = true;
        self
    }

    /// See [`offer_blocking_tool`](Self::offer_blocking_tool).
    pub fn offer_blocking_tool(mut self, enabled: bool) -> Self {
        self.offer_blocking_tool = enabled;
        self
    }

    /// See [`questions_to_lead`](Self::questions_to_lead).
    pub fn questions_to_lead(mut self, enabled: bool) -> Self {
        self.questions_to_lead = enabled;
        self
    }

    /// Offers `start_agent`'s `isolated`, backed by `provider`.
    pub fn workspace_provider(mut self, provider: Arc<dyn WorkspaceProvider>) -> Self {
        self.workspace_provider = Some(provider);
        self
    }

    /// Adjusts each subagent's configuration as it starts. See
    /// [`SubagentHook`].
    pub fn hook(mut self, hook: Arc<dyn SubagentHook>) -> Self {
        self.hook = Some(hook);
        self
    }

    /// Gives each run context in its first message. See [`StartContext`].
    pub fn start_context(mut self, context: StartContext) -> Self {
        self.start_context = Some(context);
        self
    }

    /// Adds a parameter of the host's own to `start_agent`. Its value, as
    /// the caller gave it, reaches the [hook](Self::hook) in
    /// [`StartRequest::parameters`].
    pub fn start_parameter(mut self, parameter: ToolParameterDefinition) -> Self {
        self.start_parameters.push(parameter);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), crate::WorkflowError> {
        let reserved = tools::START_PARAMETERS;
        let mut seen = std::collections::HashSet::new();
        for parameter in &self.start_parameters {
            if reserved.contains(&parameter.name.as_str()) {
                return Err(crate::WorkflowError::BuilderInvalid(format!(
                    "start_agent already has a parameter '{}'",
                    parameter.name
                )));
            }
            if !seen.insert(parameter.name.as_str()) {
                return Err(crate::WorkflowError::BuilderInvalid(format!(
                    "start_agent parameter '{}' is added twice",
                    parameter.name
                )));
            }
        }
        Ok(())
    }
}

/// What a subagent is started with, for [`SubagentHook::prepare`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartRequest {
    pub workflow_run_id: Uuid,
    /// The subagent's run id (its task id).
    pub run_id: Uuid,
    /// Its short id, such as `r1`.
    pub handle: String,
    /// The agent it runs.
    pub agent: String,
    /// The run that started it, and its agent.
    pub caller_run_id: Uuid,
    pub caller_agent: String,
    /// The task as the caller gave it.
    pub task: String,
    /// The values the caller gave the host's own `start_agent` parameters
    /// ([`SubagentsConfig::start_parameter`]), by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Value>,
}

/// Adjusts a subagent's configuration as it starts, such as its prompt
/// cache from a host parameter of `start_agent`. Set with
/// [`SubagentsConfig::hook`]. A closure
/// `Fn(&StartRequest, &mut AgentConfig) -> Result<(), String>` is one.
pub trait SubagentHook: Send + Sync {
    /// Changes `config`, the registered configuration of `request.agent`,
    /// for the run `request` starts. An error refuses the start, and the
    /// caller is told why.
    ///
    /// When a workflow resumes from a checkpoint, the hook is called again
    /// for each saved subagent run, so that its saved configuration can be
    /// checked against the registered one with the same changes. It should
    /// make the same changes for the same request.
    fn prepare(&self, request: &StartRequest, config: &mut AgentConfig) -> Result<(), String>;
}

impl<F> SubagentHook for F
where
    F: Fn(&StartRequest, &mut AgentConfig) -> Result<(), String> + Send + Sync,
{
    fn prepare(&self, request: &StartRequest, config: &mut AgentConfig) -> Result<(), String> {
        self(request, config)
    }
}

/// Gives a subagent run context in its first message as it starts, such as
/// the project's instructions. The message starts with the text it returns
/// (under the host's own headings), and the task follows under `# Your
/// task`. `None`, or blank text, leaves the message as it is without it.
/// Set with [`SubagentsConfig::start_context`].
///
/// It is called once per run, as the run starts (or, when it was stopped
/// before it started, as it starts later), so a change in the host's
/// context reaches the runs started after it. The text is part of the
/// run's task prompt. A handover restates it, and the run keeps it when it
/// is continued or resumed.
pub type StartContext = Arc<dyn Fn(&StartRequest) -> Option<String> + Send + Sync>;

/// Where a subagent is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentState {
    /// Waiting for a place under [`SubagentsConfig::max_running`].
    Queued,
    Running,
    /// It completed. Its report went to its lead. A message continues it.
    Finished,
    /// It ended without finishing (an error, or a pause no one can answer).
    /// A message continues it if its run can go on.
    Failed,
    /// Its lead, the host or a stop of the workflow stopped it. A message
    /// restarts it where it stopped.
    Stopped,
    /// The workflow ended while it ran (it resumed from a checkpoint since).
    /// [`Workflow::resume_subagents`](crate::Workflow::resume_subagents) or a
    /// message restarts it.
    Interrupted,
    /// Its isolated workspace was removed. It can't run again.
    Closed,
}

impl SubagentState {
    /// Running or waiting to run.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    pub(crate) fn describe(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Finished => "finished",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
            Self::Interrupted => "interrupted (the workflow stopped while it ran)",
            Self::Closed => "closed (its workspace was removed)",
        }
    }
}

/// How a queued subagent starts once it has a place.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingLaunch {
    /// A new run, with this initial prompt.
    Start { prompt: String },
    /// Its run goes on (continued, restarted or recovered), with this
    /// message (may be empty).
    Resume { message: String },
}

/// A background subagent, as the workflow records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubagentRecord {
    /// Its run id (task id).
    pub run_id: Uuid,
    /// Its short id, such as `r1`.
    pub handle: String,
    pub agent: String,
    /// The run that started it (its lead), and its agent.
    pub caller_run_id: Uuid,
    pub caller_agent: String,
    /// Its depth, one more than its lead's.
    pub depth: u32,
    /// The task as its lead gave it.
    pub task: String,
    /// Its working directory.
    pub workspace: PathBuf,
    /// Its isolated workspace, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolated: Option<IsolatedWorkspace>,
    /// The values of the host's own `start_agent` parameters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Value>,
    /// The other runs of its lead it may message (and that may message it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<Uuid>,
    /// It may message every other run of its lead, and they it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub all_peers: bool,
    pub state: SubagentState,
    pub created: DateTime<Utc>,
    /// When its state last changed.
    pub since: DateTime<Utc>,
    /// Its latest report (its `complete_task` summary).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<String>,
    /// Why it failed or was stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// How it starts once it has a place (while queued).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<PendingLaunch>,
    /// Its lead stopped it, with this reason (until it runs again).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_by_caller: Option<String>,
    /// A stop accepted for the current launch, kept until it ends so that
    /// a checkpoint taken during startup preserves the stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_requested: Option<crate::StopRecord>,
    /// It waits for its lead's answer to a question.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub asking: bool,
    /// Its lead's conversation was cut back to before the call that started
    /// it (`ContinueOptions::cut_back_to`), so that conversation never
    /// started it, so its mail stays here, withheld from its lead
    /// ([`MailItem::withheld`]). If it still ran, it was stopped then, with
    /// its own subagents.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub detached: bool,
}

/// Who sent a mailbox item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MailSender {
    /// An agent run.
    Run(Uuid),
    /// The workflow, on behalf of its host.
    Host,
}

/// What a mailbox item is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MailKind {
    /// A subagent finished. The body is its report.
    Report,
    /// A subagent ended without finishing. The body says why.
    Failed,
    /// A subagent was stopped. The body says why, with its progress report.
    Stopped,
    /// A message.
    Message,
    /// A subagent's question for its lead, which waits for the answer.
    Question,
    /// A copy, for a lead, of a message between two of its subagents.
    Copy,
}

impl MailKind {
    /// Whether an item of this kind ends a wait and starts a turn of a
    /// recipient that isn't running, for every kind but a copy.
    pub fn wakes(self) -> bool {
        self != Self::Copy
    }
}

/// An item in a run's mailbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MailItem {
    /// Its number in the workflow, from 1.
    pub id: u64,
    /// The run it is for.
    pub to: Uuid,
    pub from: MailSender,
    pub kind: MailKind,
    /// What the item is, as its recipient is told, such as "r1 (agent `coder`)
    /// finished. Its report:".
    pub heading: String,
    /// The report, message or reason, whole.
    pub body: String,
    pub at: DateTime<Utc>,
    /// When it reached its recipient in a context update, a wait's result
    /// or a continuation's prompt, or taken by the host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivered_at: Option<DateTime<Utc>>,
    /// The file holding the whole body, for a body longer than
    /// [`MAX_MAIL_BYTES`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    /// Not given to its recipient. It is from a subagent the recipient's
    /// conversation no longer started ([`SubagentRecord::detached`]). It
    /// stays in the records for the host to show, and `delivered_at` is when
    /// it was set aside.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub withheld: bool,
}

impl MailItem {
    /// The marker the item starts with, by which a conversation is searched
    /// for it.
    pub(crate) fn marker(&self) -> String {
        format!("[mail {}]", self.id)
    }

    /// The whole item with its marker, heading and body.
    pub fn text(&self) -> String {
        format!("{} {}\n\n{}", self.marker(), self.heading, self.body.trim())
    }

    /// The item as a context update shows it: whole, or for a long body an
    /// excerpt with the file holding all of it. Every line after the first
    /// indented.
    pub(crate) fn excerpt(&self) -> String {
        let body = self.body.trim();
        let body = if body.len() <= MAX_MAIL_BYTES {
            body.to_string()
        } else {
            let mut end = MAX_MAIL_BYTES - 1024;
            while !body.is_char_boundary(end) {
                end -= 1;
            }
            let rest = match &self.file {
                Some(file) => format!(
                    "the whole text is in {} (read it with read_file)",
                    file.display()
                ),
                None => "wait_for_agents returns the whole text".to_string(),
            };
            format!("{}\n\n[... cut here: {rest} ...]", &body[..end])
        };
        format!("{} {}\n{}", self.marker(), self.heading, body).replace('\n', "\n  ")
    }
}

/// The subagents as a checkpoint saves them in
/// [`WorkflowState::subagents`](crate::persistence::state::WorkflowState::subagents).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SubagentsSnapshot {
    #[serde(default)]
    pub runs: Vec<SubagentRecord>,
    /// Every item not delivered yet, and the delivered ones whose recipient
    /// may not hold them yet (its run hasn't ended since).
    #[serde(default)]
    pub mail: Vec<MailItem>,
    /// The number of the last run id handed out (`r<n>`).
    #[serde(default)]
    pub last_handle: u32,
    /// The number of the last mailbox item.
    #[serde(default)]
    pub last_mail: u64,
    /// Messages two subagents sent each other since their lead last
    /// messaged either, as `(run, run, count)`, the smaller id first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peer_counts: Vec<(Uuid, Uuid, u32)>,
}

/// Something that happened to a background subagent, from
/// [`Workflow::subscribe_subagents`](crate::Workflow::subscribe_subagents).
/// The events are also recorded in the run's event log. Their order relative
/// to the [`WorkflowEvent`](crate::WorkflowEvent)s of the same runs is not
/// guaranteed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubagentEvent {
    pub timestamp: DateTime<Utc>,
    pub workflow_run_id: Uuid,
    /// The subagent's run id and short id, and its agent.
    pub run_id: Uuid,
    pub handle: String,
    pub agent_name: String,
    /// The run that started it, and its agent.
    pub caller_run_id: Uuid,
    pub caller_agent: String,
    pub kind: SubagentEventKind,
}

/// What a [`SubagentEvent`] reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubagentEventKind {
    /// It waits for a place under [`SubagentsConfig::max_running`].
    Queued,
    /// Its run started.
    Started,
    /// Its run goes on again, continued with a message, restarted or
    /// recovered.
    Resumed,
    /// It completed with this report.
    Finished { summary: String },
    /// It ended without finishing.
    Failed { error: String },
    /// It was stopped by its lead (`by_caller`), else by the host or with
    /// the workflow.
    Stopped { reason: String, by_caller: bool },
    /// A message went from run `from` to run `to` (mailbox item `mail_id`).
    Message { mail_id: u64, from: Uuid, to: Uuid },
    /// Mailbox items reached run `to`.
    Delivered { to: Uuid, mail_ids: Vec<u64> },
    /// The lead (`caller_run_id`) isn't running and has news waiting, and
    /// mailbox items `mail_ids` are for it. Start a turn of it (continue
    /// or restart its run), and the news arrives at its first turn
    /// boundary. It may come more than once for the same news. Check
    /// [`Workflow::pending_news`](crate::Workflow::pending_news).
    NewsWaiting { mail_ids: Vec<u64> },
    /// Its isolated workspace was removed.
    WorkspaceRemoved,
}
