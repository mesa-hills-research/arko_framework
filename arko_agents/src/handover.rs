//! Conversation handover lets an agent whose context grows large write a
//! report of its progress and continue from that report in a fresh
//! conversation, in the same task.
//!
//! Performance and cost both suffer as a conversation grows. A handover keeps
//! the task (its ID, state, budget and usage history) and replaces only the
//! conversation. The runtime asks the agent for a report, then starts a new
//! conversation whose first message is the task prompt followed by the
//! report. Nothing outside the task sees the agent stop, so hosts that
//! address the agent by name or task ID are unaffected.
//!
//! [`AgentConfig::handover`](crate::agent_config::AgentConfig::handover) turns
//! it on at a context size. A host can also ask for one at any time with
//! [`UserCommand::RequestHandover`](crate::runtime::command::UserCommand::RequestHandover),
//! and change the size while the task runs with a [`HandoverThreshold`].
//! A [`HandoverHook`] lets the host adjust the fresh conversation's first
//! message and update its own records at the moment the conversation is
//! replaced.
//!
//! How the agent gives the report depends on how it calls tools. With native
//! tool calling and `handover` set, arko's runtime offers it the
//! [`HANDOVER_REPORT_TOOL_NAME`] tool from the first request (so the tool
//! list, and the prompt cache, stay the same when the report is asked for),
//! and the report is that tool's `report` argument. Agents told that every
//! response must call a tool can then follow that rule while they write it.
//! Otherwise (output-based tool calling, Claude Code, or a host's request to
//! an agent without `handover`) the report is a reply without a tool call.
//!
//! A request still over the model's input limit after the overflow guard
//! shortened its results hands over too, for an agent that hands over. The
//! step that request added (the model's reply and its results) is left
//! out, and the report is asked for from the conversation as it was at the
//! last request the provider answered, with [`STEP_LEFT_OUT`] before the
//! request. An agent whose model runs in Claude Code does the same for a
//! turn that goes over the limit. The Claude Code session resumes at the
//! message that ended the last request Claude Code answered.
//!
//! An agent whose model runs in Claude Code can compact instead
//! ([`HandoverConfig::compact_in_claude_code`]). At the threshold, or at a
//! host's request, the runtime sends Claude Code's own `/compact` into the
//! session, and the agent goes on in the same session from Claude Code's
//! summary of the conversation. See [`crate::claude_code`].
//!
//! A host can also ask for the report alone, with [`Task::request_report`],
//! such as before it goes on with the work in another conversation, on
//! another model. The agent writes it as it writes a handover report, with
//! every other tool call rejected until it does, and the task then completes
//! with the report as its summary. The conversation is not replaced, and an
//! agent that compacts in Claude Code writes the report all the same.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::agent_config::AgentConfig;
use crate::error::ArkoAgentsError;
use crate::task::Task;
use crate::tool::{ToolDefinition, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};

/// The tool an agent gives its handover report with. See the
/// [module docs](self) for when it is offered.
pub const HANDOVER_REPORT_TOOL_NAME: &str = "handover_report";

/// The [`HANDOVER_REPORT_TOOL_NAME`] tool's parameter that holds the report.
pub const HANDOVER_REPORT_PARAM_NAME: &str = "report";

/// What the report request asks the report to cover, after it says how to
/// give the report.
macro_rules! report_contents {
    () => {
        "Cover what the task is, what you have done and the evidence for it (files, commands, results), decisions you made and why, what you were doing when you stopped, and the next steps. List anything you are still waiting for, such as a reply to a question you asked, and from whom, and every message you received that you haven't fully answered yet: who sent it, what it asked, and what you meant to reply. Be complete: the fresh conversation will have only the task and this report. Messages that arrive while you write it are delivered after it, in the fresh conversation."
    };
}

/// What the runtime asks for, as a user message, when a handover starts
/// and the agent is offered the [`HANDOVER_REPORT_TOOL_NAME`] tool. Other
/// tool calls are rejected until the agent calls it (it may still call
/// `complete_task` if the task is done).
pub const DEFAULT_REPORT_REQUEST: &str = concat!(
    "Your conversation is getting long, so it will be replaced with a fresh one that starts from your task and a report you write now. Call `handover_report` with that report; your other tool calls are not run until you do (call `complete_task` instead if the task is done). ",
    report_contents!()
);

/// [`DEFAULT_REPORT_REQUEST`] for an agent not offered `complete_task`
/// (its reply in text ends the task through
/// [`AgentToolConfig::text_reply_ends_task`](crate::agent_config::AgentToolConfig::text_reply_ends_task)).
pub const DEFAULT_REPORT_REQUEST_NO_COMPLETION: &str = concat!(
    "Your conversation is getting long, so it will be replaced with a fresh one that starts from your task and a report you write now. Call `handover_report` with that report. Your other tool calls are not run until you do. ",
    report_contents!()
);

/// What the runtime asks for when the agent gives the report as a reply
/// without a tool call (see the [module docs](self)). Other tool calls are
/// rejected until it does (it may still call `complete_task` if the task is
/// done).
pub const DEFAULT_TEXT_REPORT_REQUEST: &str = concat!(
    "Your conversation is getting long, so it will be replaced with a fresh one that starts from your task and a report you write now. Reply with that report as plain text, without calling a tool. ",
    report_contents!()
);

/// What the runtime asks a chat agent, whose reply in text ends its task,
/// for when its conversation is handed over.
/// The fresh conversation starts from the user's latest message and this
/// summary (see [`HandoverSeed::for_chat`]).
pub const DEFAULT_CHAT_REPORT_REQUEST: &str = "This conversation is getting long, so it will continue in a fresh one that starts from a summary you write now. Reply with that summary as plain text. Cover what the user wants and asked, what you answered, the facts, preferences and decisions to keep, and anything still open. The user's latest message comes before this request, and you haven't answered it yet: don't answer it here. The fresh conversation answers it from your summary, and will have only that message and the summary, so be complete.";

/// What a report asked for alone ([`Task::request_report`]) should cover,
/// after the request says how to give it.
macro_rules! report_only_contents {
    () => {
        "Cover what the task is, what you have done and the evidence for it (files, commands, results), decisions you made and why, what you were doing and what you meant to do next, and what you are unsure of. List anything you are still waiting for, such as a reply to a question you asked, and from whom, and every message you received that you haven't fully answered yet, with who sent it, what it asked, and what you meant to reply. Be complete, since the work may go on from this report alone."
    };
}

/// What the runtime asks for, as a user message, for a report alone
/// ([`Task::request_report`]) from an agent offered the
/// [`HANDOVER_REPORT_TOOL_NAME`] tool. Every other tool call is rejected
/// until the agent calls it.
pub const DEFAULT_REPORT_ONLY_REQUEST: &str = concat!(
    "Write a report of your work so far now. Call `handover_report` with it. Your other tool calls are not run until you do. ",
    report_only_contents!()
);

/// [`DEFAULT_REPORT_ONLY_REQUEST`] for an agent that gives the report as a
/// reply without a tool call (see the [module docs](self)).
pub const DEFAULT_TEXT_REPORT_ONLY_REQUEST: &str = concat!(
    "Write a report of your work so far now. Reply with it as plain text, without calling a tool. ",
    report_only_contents!()
);

/// Said before the report request when the step that took a request over
/// the model's input limit was left out of the conversation (see the
/// [module docs](self)).
pub const STEP_LEFT_OUT: &str = "The results of your latest step were too large for your context, so that step (your last reply and its tool results) was left out of this conversation. Say in your report what that step was for, so that the fresh conversation can do it again in smaller parts: narrower reads or searches, or output limited to what you need.";

/// The heading of the report in the fresh conversation's first message.
pub const REPORT_HEADING: &str = "# Handover report";

/// Said before the report in the fresh conversation's first message.
pub const REPORT_INTRODUCTION: &str = "Your earlier conversation on this task was replaced with this one to keep your context small. Before it was replaced, you wrote the report below. Continue the task from it.";

/// When an agent hands over. See the [module docs](self). At least one
/// threshold must be set. In arko's own runtime it is checked before each
/// model request against the estimated input of that request, using the latest
/// request's reported input, plus what the conversation gained since (the
/// reply, at least its reported output tokens, and the turn's tool
/// results) at the bytes-per-token ratio calibrated from the task's own
/// requests. This is the overflow guard's calibrated estimate, not the
/// 10-bytes-per-token lower bound it decides by (see
/// [`ToolResultLimits`](crate::agent_config::ToolResultLimits)). So a turn
/// whose results take the conversation over the threshold hands over
/// before the request that would carry them. The Claude Code runtime
/// checks the latest request's input, as Claude Code reports it.
/// [`HandoverRequest::input_tokens`] and [`HandoverRecord::input_tokens`]
/// record the latest request's reported input.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HandoverConfig {
    /// Hand over once a request's input reaches this many tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_input_tokens: Option<usize>,
    /// Hand over once a request's input reaches this share of the model's
    /// input limit, above 0 and at most 1. Ignored for a model whose limit
    /// is unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_context_fraction: Option<f64>,
    /// Replaces [`DEFAULT_REPORT_REQUEST`] (or [`DEFAULT_TEXT_REPORT_REQUEST`]
    /// where the report is a reply without a tool call). It should say how
    /// to give the report, using the [`HANDOVER_REPORT_TOOL_NAME`] tool where
    /// it is offered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_request: Option<String>,
    /// On a Claude Code model, compact instead of handing over. At the
    /// threshold, or when a host asks with
    /// [`UserCommand::RequestHandover`](crate::runtime::command::UserCommand::RequestHandover),
    /// the runtime has Claude Code compact the conversation with its own
    /// `/compact`, and the agent goes on in the same session from Claude
    /// Code's summary. A compaction that doesn't take place falls back to a
    /// handover. See [`crate::claude_code`]. Other models hand over as
    /// usual.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub compact_in_claude_code: bool,
}

impl HandoverConfig {
    /// A handover once a request's input reaches `tokens` tokens.
    pub fn at_input_tokens(tokens: usize) -> Self {
        Self {
            at_input_tokens: Some(tokens),
            ..Self::default()
        }
    }

    /// A handover once a request's input reaches `fraction` of the model's
    /// input limit.
    pub fn at_context_fraction(fraction: f64) -> Self {
        Self {
            at_context_fraction: Some(fraction),
            ..Self::default()
        }
    }

    /// The input size, in tokens, at which to hand over for a model that
    /// accepts `max_input_tokens` (0 when unknown), given by the lower of the two
    /// thresholds that apply.
    pub fn threshold(&self, max_input_tokens: usize) -> Option<usize> {
        let fraction = self
            .at_context_fraction
            .filter(|_| max_input_tokens > 0)
            .map(|fraction| (max_input_tokens as f64 * fraction) as usize);
        match (self.at_input_tokens, fraction) {
            (Some(tokens), Some(fraction)) => Some(tokens.min(fraction)),
            (tokens, fraction) => tokens.or(fraction),
        }
    }

    /// The request the agent receives when the handover starts, taken from
    /// `report_request`, or by default [`DEFAULT_REPORT_REQUEST`] when the
    /// agent gives the report with the [`HANDOVER_REPORT_TOOL_NAME`] tool
    /// (`with_tool`) and [`DEFAULT_TEXT_REPORT_REQUEST`] otherwise.
    pub fn report_request(&self, with_tool: bool) -> &str {
        self.report_request_for(with_tool, true)
    }

    /// [`report_request`](Self::report_request) for an agent offered
    /// `complete_task` or not (`completes`). The default request mentions
    /// it only where it is offered.
    pub fn report_request_for(&self, with_tool: bool, completes: bool) -> &str {
        self.report_request
            .as_deref()
            .unwrap_or(default_report_request_for(with_tool, completes))
    }

    /// The request an agent offered no tools (a chat) receives, taken from
    /// `report_request`, or by default [`DEFAULT_CHAT_REPORT_REQUEST`].
    pub fn chat_report_request(&self) -> &str {
        self.report_request
            .as_deref()
            .unwrap_or(DEFAULT_CHAT_REPORT_REQUEST)
    }

    pub fn validate(&self) -> Result<(), ArkoAgentsError> {
        let error = |message: &str| ArkoAgentsError::BuilderError {
            struct_name: "HandoverConfig".into(),
            message: message.into(),
        };
        if self.at_input_tokens.is_none() && self.at_context_fraction.is_none() {
            return Err(error(
                "set at_input_tokens or at_context_fraction. A host can also ask for a handover with UserCommand::RequestHandover.",
            ));
        }
        if self.at_input_tokens == Some(0) {
            return Err(error("at_input_tokens must be positive."));
        }
        if self
            .at_context_fraction
            .is_some_and(|fraction| !(fraction > 0.0 && fraction <= 1.0))
        {
            return Err(error("at_context_fraction must be above 0 and at most 1."));
        }
        if self
            .report_request
            .as_deref()
            .is_some_and(|request| request.trim().is_empty())
        {
            return Err(error("report_request must not be empty."));
        }
        Ok(())
    }
}

/// The default report request, using [`DEFAULT_REPORT_REQUEST`] when the agent
/// gives the report with the [`HANDOVER_REPORT_TOOL_NAME`] tool
/// (`with_tool`), [`DEFAULT_TEXT_REPORT_REQUEST`] otherwise.
pub fn default_report_request(with_tool: bool) -> &'static str {
    default_report_request_for(with_tool, true)
}

/// [`default_report_request`] for an agent offered `complete_task` or not
/// (`completes`), using [`DEFAULT_REPORT_REQUEST_NO_COMPLETION`] for one given
/// the tool and not offered `complete_task`.
pub fn default_report_request_for(with_tool: bool, completes: bool) -> &'static str {
    match (with_tool, completes) {
        (true, true) => DEFAULT_REPORT_REQUEST,
        (true, false) => DEFAULT_REPORT_REQUEST_NO_COMPLETION,
        (false, _) => DEFAULT_TEXT_REPORT_REQUEST,
    }
}

/// A handover threshold a host sets for the runs it starts, in place of
/// their configuration's ([`HandoverConfig::threshold`]). A run hands over
/// (or compacts, see [`HandoverConfig::compact_in_claude_code`]) once a
/// request's input reaches it, as at
/// [`at_input_tokens`](HandoverConfig::at_input_tokens), also for an agent
/// without [`AgentConfig::handover`](crate::agent_config::AgentConfig::handover).
/// Unset, the configuration's threshold applies. Set with
/// [`AgentRuntime::with_handover_threshold`](crate::runtime::AgentRuntime::with_handover_threshold).
///
/// The host may change it while the runs run, larger or smaller. Each
/// check reads it, so the next one goes by the new threshold. Like
/// [`HostDecisions`](crate::runtime::HostDecisions), this run-level setting is
/// not part of the task or its configuration and is not saved, so the
/// tools, the instructions and the conversation (and their prompt cache)
/// stay as they are, and a task run later without it hands over as
/// configured. Clones share the setting. The runtime doesn't cap it at the
/// model's input limit, so the host keeps it under that limit, with room for
/// the report, or gives each run its cap with [`with_most`](Self::with_most)
/// when one threshold serves runs on models of different sizes.
#[derive(Clone, Default)]
pub struct HandoverThreshold {
    tokens: Arc<AtomicUsize>,
    most: Option<HandoverCap>,
}

/// The largest threshold a run of a configuration goes by, if any: see
/// [`HandoverThreshold::with_most`].
type HandoverCap = Arc<dyn Fn(&AgentConfig) -> Option<usize> + Send + Sync>;

impl std::fmt::Debug for HandoverThreshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandoverThreshold")
            .field("tokens", &self.get())
            .field("capped", &self.most.is_some())
            .finish()
    }
}

impl HandoverThreshold {
    /// A threshold of `input_tokens`, or unset (`None`).
    pub fn new(input_tokens: Option<usize>) -> Self {
        let threshold = Self::default();
        threshold.set(input_tokens);
        threshold
    }

    /// This threshold, capped for each run at what `most` gives for the
    /// run's configuration (`None` sets no cap), whichever threshold the run
    /// goes by, the host's or its configuration's. One threshold then
    /// serves runs on models of different sizes, each kept under its own
    /// model's limit. `most` is asked at every check. Clones made from it
    /// share the cap.
    pub fn with_most(
        mut self,
        most: impl Fn(&AgentConfig) -> Option<usize> + Send + Sync + 'static,
    ) -> Self {
        self.most = Some(Arc::new(most));
        self
    }

    /// Changes the threshold for every clone. `Some(tokens)` hands over at
    /// `tokens` of input, and `None` (or `Some(0)`) goes back to the
    /// configuration's.
    pub fn set(&self, input_tokens: Option<usize>) {
        self.tokens
            .store(input_tokens.unwrap_or(0), Ordering::Release);
    }

    /// The threshold the host set, if any.
    pub fn get(&self) -> Option<usize> {
        Some(self.tokens.load(Ordering::Acquire)).filter(|tokens| *tokens > 0)
    }

    /// The threshold a run of `config` goes by: the host's, else the
    /// configuration's, up to the cap for `config` (see
    /// [`with_most`](Self::with_most)).
    pub(crate) fn of(&self, config: &AgentConfig) -> Option<usize> {
        let threshold = self.get().or_else(|| {
            config
                .handover
                .as_ref()?
                .threshold(config.llm_config.model.max_input_tokens)
        })?;
        let most = self.most.as_ref().and_then(|most| most(config));
        Some(most.map_or(threshold, |most| threshold.min(most)))
    }
}

/// The definition of the [`HANDOVER_REPORT_TOOL_NAME`] tool. The runtime
/// answers its calls itself rather than through a [`ToolRegistry`](crate::tool::ToolRegistry).
pub fn report_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: HANDOVER_REPORT_TOOL_NAME.to_string(),
        description: "Gives the handover report the runtime asks for when your conversation gets long: your conversation is then replaced with a fresh one that starts from your task and this report. Call it only after that request, which says what the report should cover.".to_string(),
        output_based_example: None,
        parameters: vec![ToolParameterDefinition {
            name: HANDOVER_REPORT_PARAM_NAME.to_string(),
            description: "The full report, in Markdown. The fresh conversation has only your task and this report.".to_string(),
            param_type: ToolParameterType::String,
            required: true,
            is_path: false,
            flag: None,
            default_value: None,
        }],
        parallel_safe: false,
        tool_type_details: ToolTypeDetails::BuiltIn,
        raw_json_schema_override: None,
    }
}

/// A handover the runtime has asked the agent for and is waiting on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HandoverRequest {
    pub requested_at: DateTime<Utc>,
    /// The input size of the latest request when the handover started. 0
    /// when a host asked for it before any request.
    pub input_tokens: usize,
    /// The report request is in the conversation. A host's request is noted
    /// at once and the report requested at the next turn boundary where a
    /// user message can go.
    #[serde(default)]
    pub report_requested: bool,
    /// The report the agent gave with the [`HANDOVER_REPORT_TOOL_NAME`]
    /// tool. The runtime replaces the conversation with it at the next turn
    /// boundary, once the turn's results are in, so a task stopped in
    /// between still hands over with it when it runs again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<String>,
    /// The report alone is asked for ([`Task::request_report`]). Once it
    /// comes, the task completes with it as its summary, and the
    /// conversation stays as it is.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub report_only: bool,
    /// The host's request for a report alone, in place of the default (see
    /// [`report_only_request`](Self::report_only_request)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
}

impl HandoverRequest {
    /// A request made now, at `input_tokens`, whose report isn't asked for
    /// yet.
    pub fn new(input_tokens: usize) -> Self {
        Self {
            requested_at: Utc::now(),
            input_tokens,
            report_requested: false,
            report: None,
            report_only: false,
            request: None,
        }
    }

    /// The request an agent receives for a report alone, which is the host's
    /// [`request`](Self::request), or by default
    /// [`DEFAULT_REPORT_ONLY_REQUEST`] when the agent gives the report with
    /// the [`HANDOVER_REPORT_TOOL_NAME`] tool (`with_tool`) and
    /// [`DEFAULT_TEXT_REPORT_ONLY_REQUEST`] otherwise.
    pub fn report_only_request(&self, with_tool: bool) -> &str {
        self.request.as_deref().unwrap_or(if with_tool {
            DEFAULT_REPORT_ONLY_REQUEST
        } else {
            DEFAULT_TEXT_REPORT_ONLY_REQUEST
        })
    }
}

/// A handover that has taken place.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HandoverRecord {
    pub at: DateTime<Utc>,
    /// The input size of the latest request when the handover started.
    pub input_tokens: usize,
    /// How many messages the replaced conversation had.
    pub replaced_messages: usize,
    /// The report the agent wrote.
    pub report: String,
}

/// The fresh conversation's first message, before the runtime adds the
/// environment details and reminder it adds to every task prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct HandoverSeed {
    /// The task, restated. Defaults to the task's initial prompt.
    pub task_prompt: String,
    /// Context the host adds after the report, such as the agents the task
    /// can message.
    pub context: Option<String>,
}

impl HandoverSeed {
    /// The default seed, with the task's initial prompt and no further context.
    pub fn for_task(task: &Task) -> Self {
        Self {
            task_prompt: task.initial_task_prompt.clone(),
            context: None,
        }
    }

    /// The default for an agent offered no tools (a chat), using the user's
    /// messages the conversation hasn't answered yet (see
    /// [`unanswered_user_messages`]), which the fresh conversation answers,
    /// in place of the task's initial prompt. Falls back to
    /// [`for_task`](Self::for_task) when there are none. Their images go
    /// into the fresh conversation's first message too.
    pub fn for_chat(task: &Task) -> Self {
        // Each message as the host sent it, without the environment details
        // and reminder the runtime added (the fresh conversation's first
        // message gets its own), and a message of images alone says so.
        let text = unanswered_user_messages(task)
            .into_iter()
            .map(|message| {
                let text = task.host_text(message);
                match (text.trim().is_empty(), message.image_count()) {
                    (false, _) => text,
                    (true, 0) => String::new(),
                    (true, 1) => "(The user sent the image below.)".to_string(),
                    (true, _) => "(The user sent the images below.)".to_string(),
                }
            })
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if text.is_empty() {
            return Self::for_task(task);
        }
        Self {
            task_prompt: text,
            context: None,
        }
    }

    /// The message text, with the task prompt, the report under
    /// [`REPORT_HEADING`], then the context.
    pub fn message(&self, report: &str) -> String {
        let mut message = format!(
            "{}\n\n{REPORT_HEADING}\n\n{REPORT_INTRODUCTION}\n\n{}",
            self.task_prompt.trim_end(),
            report.trim()
        );
        if let Some(context) = self.context.as_deref().filter(|c| !c.trim().is_empty()) {
            message.push_str("\n\n");
            message.push_str(context.trim());
        }
        message
    }
}

/// The user's messages since the model's last reply before the handover's
/// report request, oldest first. This is the input the conversation hasn't answered
/// yet, which a chat's fresh conversation answers. Messages the runtime
/// added (the report request, notes on errors) and tool results don't
/// count, while context updates (notes from the user, mail) do.
pub fn unanswered_user_messages(task: &Task) -> Vec<&arko_ai::Message> {
    let messages = &task.conversation.messages;
    // The report request, and the report after it, if it came.
    let mut end = messages.len();
    while end > 0
        && (messages[end - 1].role == arko_ai::Role::Assistant
            || task.runtime_messages.contains(&(end - 1)))
    {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && messages[start - 1].role == arko_ai::Role::User {
        start -= 1;
    }
    (start..end)
        .filter(|index| {
            !task.runtime_messages.contains(index)
                && messages[*index]
                    .content
                    .iter()
                    .all(|content| content.as_tool_result().is_none())
        })
        .map(|index| &messages[index])
        .collect()
}

/// The images of the user's messages a chat's fresh conversation answers
/// ([`unanswered_user_messages`]), for its first message.
pub(crate) fn unanswered_images(task: &Task) -> Vec<arko_ai::Content> {
    unanswered_user_messages(task)
        .into_iter()
        .flat_map(|message| &message.content)
        .filter(|content| content.as_image().is_some())
        .cloned()
        .collect()
}

/// The images of a fresh conversation's first message: a chat's
/// [`unanswered_images`], or the task's initial images, which go with the
/// task it restates.
pub(crate) fn seed_images(task: &Task, chat: bool) -> Vec<arko_ai::Content> {
    if chat {
        unanswered_images(task)
    } else {
        task.initial_images()
            .into_iter()
            .map(arko_ai::Content::Image)
            .collect()
    }
}

/// Called by the runtime at the moment a task's conversation is replaced,
/// or compacted in Claude Code. Set with
/// [`AgentRuntime::with_handover_hook`](crate::runtime::AgentRuntime::with_handover_hook).
pub trait HandoverHook: Send + Sync {
    /// `task` still holds the conversation being replaced. Returns the fresh
    /// conversation's first message. The default restates the task's
    /// initial prompt. A host that tracks what the conversation received
    /// (delivered messages, files read) updates its records here, before the
    /// conversation is gone.
    fn on_handover(&self, task: &Task, report: &str) -> HandoverSeed {
        let _ = report;
        HandoverSeed::for_task(task)
    }

    /// The configuration `task`'s fresh conversation runs with, in place of
    /// the task's, or `None` (the default) to keep the task's. `task` still
    /// holds the conversation being replaced. A host whose agent now runs
    /// with another configuration than the one a saved task keeps (other
    /// tools, say) gives that one here, since the fresh conversation writes
    /// the prompt cache afresh anyway. Its tools and the system prompt
    /// written from it apply from the fresh conversation's first request,
    /// and the task keeps it. Its skills are loaded and set up again when
    /// they differ from the task's (see [`same_skills`]), with a new
    /// `load_skill`. It must have the task's model (provider and ID), and the
    /// other tools it offers must be in the runtime's tool registry. One with
    /// another model, one that isn't valid, or one whose skills don't load
    /// is left out, and the task keeps its own (logged as a warning).
    fn handover_config(&self, task: &Task) -> Option<AgentConfig> {
        let _ = task;
        None
    }

    /// Claude Code compacted `task`'s conversation. Its summary replaced the
    /// earlier messages in the session (`task` keeps them all). Returns
    /// context to give the agent again with its next input, such as
    /// standing instructions a summary may leave out, or `None` (the
    /// default). It reaches the agent as a context update.
    fn on_compaction(&self, task: &Task) -> Option<String> {
        let _ = task;
        None
    }

    /// The questions `task`'s agent asked that are still open, as the host
    /// keeps them, or `None` (the default) to go by the task's own log
    /// ([`Task::questions`]), which knows of the answers given while the
    /// agent waited and of the questions it closed. The fresh conversation
    /// of a handover (after its report and context) and a compacted one
    /// (with its next input) restate them, so the agent doesn't lose track.
    /// An answer to questions the agent waited for leaves those the host
    /// still has open, so a host can answer some of them. A host that
    /// delivers answers later as context returns the questions it still has
    /// open. Claude Code waits for an answer only when the turn that asked
    /// ends, and doesn't wait for the questions this leaves out by then,
    /// such as those the host answered while the turn went on.
    fn open_questions(&self, task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        let _ = task;
        None
    }

    /// The number of the first of `count` questions `task`'s agent asks now
    /// (they are `q{n}`, `q{n + 1}`, ...), which a host that shows the
    /// questions of several tasks gives out, one ask at a time, so that no
    /// two of them share an id, even in tasks that run at the same time.
    /// `None` (the default) numbers them on from the task's own count. A
    /// number below the task's own count is raised to it.
    fn question_numbers(&self, task: &Task, count: u32) -> Option<u32> {
        let _ = (task, count);
        None
    }

    /// `task`'s agent asked `asked` (`wait`: it waits for the answer). The
    /// runtime calls it as it records them, before the agent goes on, so a
    /// host's [`open_questions`](Self::open_questions) has them for a
    /// handover or compaction that comes right after.
    fn questions_asked(&self, task: &Task, asked: &[crate::questions::AskedQuestion], wait: bool) {
        let _ = (task, asked, wait);
    }
}

/// Whether two skills configurations are the same, embedded skills
/// included (a saved configuration leaves those out).
pub fn same_skills(
    a: Option<&crate::skill::SkillsConfig>,
    b: Option<&crate::skill::SkillsConfig>,
) -> bool {
    let saved = |skills: Option<&crate::skill::SkillsConfig>| {
        skills.map(|skills| (serde_json::to_value(skills).ok(), skills.embedded.clone()))
    };
    saved(a) == saved(b)
}

/// Whether a task whose configuration is `current` would run the same with
/// `next` from a [`HandoverHook::handover_config`]. The runtime lists
/// `load_skill` by the skills, and gives it the per-tool entry
/// [`SkillManager::setup`](crate::skill::SkillManager::setup) does where
/// there is none, so neither counts. An explicit entry for `load_skill`
/// counts, and so do the skills, embedded ones included ([`same_skills`]).
pub fn same_for_handover(current: &AgentConfig, next: &AgentConfig) -> bool {
    let normalized = |config: &AgentConfig| {
        let mut config = config.clone();
        let tools = &mut config.tool_config;
        tools
            .available_tool_names
            .retain(|name| name != crate::skill::LOAD_SKILL_TOOL_NAME);
        let generated = serde_json::to_value(crate::skill::load_skill_default_override()).ok();
        if tools
            .tool_specific_configs
            .get(crate::skill::LOAD_SKILL_TOOL_NAME)
            .is_some_and(|entry| serde_json::to_value(entry).ok() == generated)
        {
            tools
                .tool_specific_configs
                .remove(crate::skill::LOAD_SKILL_TOOL_NAME);
        }
        serde_json::to_value(config).ok()
    };
    normalized(current) == normalized(next)
        && same_skills(current.skills_config.as_ref(), next.skills_config.as_ref())
}

/// The tools and skill prompt a runtime goes on with once its task takes
/// `config` from a [`HandoverHook::handover_config`], in place of
/// `current`. `None` keeps what the runtime has, when the skills stay the
/// same, and `config` then lists `load_skill` as `current` does. Otherwise
/// the skills load from `config` in `workspace`, its `load_skill` replaces
/// the runtime's (`config` lists it when it has skills to load), and the
/// prompt lists its skills. Without skills, there is neither. An explicit
/// per-tool entry for `load_skill` in `config` (its approval, time limit or
/// result cap) stays. Without one, a listed `load_skill` gets the entry
/// [`SkillManager::setup`](crate::skill::SkillManager::setup) gives.
pub(crate) fn handover_skills(
    all_tools: &crate::tool::ToolRegistry,
    current: &AgentConfig,
    config: &mut AgentConfig,
    workspace: &std::path::Path,
) -> Result<
    Option<(
        crate::tool::ToolRegistry,
        Option<crate::skill::SkillPromptContext>,
    )>,
    ArkoAgentsError,
> {
    use crate::skill::LOAD_SKILL_TOOL_NAME;
    let tools = &mut config.tool_config;
    tools
        .available_tool_names
        .retain(|name| name != LOAD_SKILL_TOOL_NAME);
    if same_skills(
        current.skills_config.as_ref(),
        config.skills_config.as_ref(),
    ) {
        if current
            .tool_config
            .available_tool_names
            .iter()
            .any(|name| name == LOAD_SKILL_TOOL_NAME)
        {
            tools
                .available_tool_names
                .push(LOAD_SKILL_TOOL_NAME.to_string());
            tools
                .tool_specific_configs
                .entry(LOAD_SKILL_TOOL_NAME.to_string())
                .or_insert_with(crate::skill::load_skill_default_override);
        }
        return Ok(None);
    }
    let names = all_tools
        .list_tool_names()
        .into_iter()
        .filter(|name| name != LOAD_SKILL_TOOL_NAME)
        .collect::<Vec<_>>();
    let mut registry = all_tools.restricted_to(names.iter().map(String::as_str));
    let prompt = match config.skills_config.clone() {
        Some(skills) => {
            let manager = crate::skill::SkillManager::load(&skills, workspace)?;
            Some(manager.setup(&mut registry, config))
        }
        None => None,
    };
    Ok(Some((registry, prompt)))
}

/// The highest question number already taken before `count` questions
/// `task`'s agent asks now (see [`HandoverHook::question_numbers`]), with
/// `taken` numbers the runtime holds for calls not recorded yet.
pub(crate) fn last_question_number(
    hook: Option<&dyn HandoverHook>,
    task: &Task,
    count: u32,
    taken: u32,
) -> u32 {
    let own = task.questions.asked.max(taken);
    match hook.and_then(|hook| hook.question_numbers(task, count)) {
        Some(first) => first.saturating_sub(1).max(own),
        None => own,
    }
}

/// The answer to the questions `task`'s agent waited for came: they are
/// settled, except those `hook` says are still open.
pub(crate) fn settle_waited(hook: Option<&dyn HandoverHook>, task: &mut Task) {
    let open = hook
        .and_then(|hook| hook.open_questions(task))
        .unwrap_or_default();
    let answered = std::mem::take(&mut task.questions.waiting)
        .into_iter()
        .filter(|id| !open.iter().any(|question| &question.id == id))
        .collect::<Vec<_>>();
    task.questions.settle(&answered);
}

/// Of `asked`, questions `task`'s agent waits for, those still open: in the
/// task's log (the agent didn't close them) and as `hook` says. The others
/// were settled since they were asked (answered, dismissed or closed), and
/// the log settles them too.
pub(crate) fn still_open(
    hook: Option<&dyn HandoverHook>,
    task: &mut Task,
    asked: Vec<crate::questions::AskedQuestion>,
) -> Vec<crate::questions::AskedQuestion> {
    let open = open_questions(hook, task);
    let (open, settled): (Vec<_>, Vec<_>) = asked.into_iter().partition(|question| {
        task.questions.get(&question.id).is_some()
            && open.iter().any(|known| known.id == question.id)
    });
    let settled = settled
        .into_iter()
        .map(|question| question.id)
        .collect::<Vec<_>>();
    task.questions.settle(&settled);
    open
}

/// `seed` with the section that restates the open questions `hook` (or the
/// task's log, without a hook) gives, after its context.
pub(crate) fn with_open_questions(
    mut seed: HandoverSeed,
    hook: Option<&dyn HandoverHook>,
    task: &Task,
) -> HandoverSeed {
    if let Some(section) = crate::questions::open_section(&open_questions(hook, task)) {
        seed.context = Some(match seed.context.take().filter(|c| !c.trim().is_empty()) {
            Some(context) => format!("{}\n\n{section}", context.trim_end()),
            None => section,
        });
    }
    seed
}

/// The questions still open, as `hook` (or, without one, the task's log)
/// says.
pub(crate) fn open_questions(
    hook: Option<&dyn HandoverHook>,
    task: &Task,
) -> Vec<crate::questions::AskedQuestion> {
    hook.and_then(|hook| hook.open_questions(task))
        .unwrap_or_else(|| task.questions.open.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_take_the_lower_limit() {
        let both = HandoverConfig {
            at_input_tokens: Some(150_000),
            at_context_fraction: Some(0.25),
            ..HandoverConfig::default()
        };
        assert_eq!(both.threshold(1_000_000), Some(150_000));
        assert_eq!(both.threshold(200_000), Some(50_000));
        assert_eq!(both.threshold(0), Some(150_000));
        assert_eq!(HandoverConfig::at_context_fraction(0.3).threshold(0), None);
        assert_eq!(
            HandoverConfig::at_context_fraction(0.3).threshold(1_000_000),
            Some(300_000)
        );
    }

    #[test]
    fn validation_needs_a_sensible_threshold() {
        assert!(HandoverConfig::default().validate().is_err());
        assert!(HandoverConfig::at_input_tokens(0).validate().is_err());
        assert!(HandoverConfig::at_context_fraction(1.5).validate().is_err());
        assert!(HandoverConfig::at_context_fraction(0.0).validate().is_err());
        assert!(HandoverConfig::at_context_fraction(0.25).validate().is_ok());
        let blank = HandoverConfig {
            report_request: Some("  ".into()),
            ..HandoverConfig::at_input_tokens(10)
        };
        assert!(blank.validate().is_err());
    }

    #[test]
    fn compaction_in_claude_code_is_saved_only_when_set() {
        let config = HandoverConfig::at_input_tokens(10);
        assert!(!config.compact_in_claude_code);
        let saved = serde_json::to_value(&config).unwrap();
        assert_eq!(saved, serde_json::json!({"at_input_tokens": 10}));
        let compacting = HandoverConfig {
            compact_in_claude_code: true,
            ..config
        };
        let saved = serde_json::to_value(&compacting).unwrap();
        assert_eq!(saved["compact_in_claude_code"], true);
        let loaded: HandoverConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(loaded, compacting);
    }

    #[test]
    fn the_report_request_says_how_to_give_the_report() {
        let config = HandoverConfig::at_input_tokens(10);
        assert_eq!(config.report_request(true), DEFAULT_REPORT_REQUEST);
        assert_eq!(config.report_request(false), DEFAULT_TEXT_REPORT_REQUEST);
        assert!(DEFAULT_REPORT_REQUEST.contains("Call `handover_report`"));
        assert!(DEFAULT_TEXT_REPORT_REQUEST.contains("without calling a tool"));
        // Both ask for the same contents.
        assert!(DEFAULT_REPORT_REQUEST.ends_with(report_contents!()));
        assert!(DEFAULT_TEXT_REPORT_REQUEST.ends_with(report_contents!()));
        let custom = HandoverConfig {
            report_request: Some("Report.".into()),
            ..config
        };
        assert_eq!(custom.report_request(true), "Report.");
        assert_eq!(custom.report_request(false), "Report.");
    }

    #[test]
    fn the_report_tool_takes_the_report() {
        let definition = report_tool_definition();
        assert_eq!(definition.name, HANDOVER_REPORT_TOOL_NAME);
        let [parameter] = definition.parameters.as_slice() else {
            panic!("{definition:?}");
        };
        assert_eq!(parameter.name, HANDOVER_REPORT_PARAM_NAME);
        assert!(parameter.required);
    }

    #[test]
    fn requests_saved_without_a_report_load() {
        let request = HandoverRequest::new(1_500);
        let saved = serde_json::to_value(&request).unwrap();
        assert!(saved.get("report").is_none(), "{saved}");
        let loaded: HandoverRequest = serde_json::from_value(serde_json::json!({
            "requested_at": request.requested_at,
            "input_tokens": 1_500,
            "report_requested": true,
        }))
        .unwrap();
        assert_eq!(loaded.report, None);
        assert!(loaded.report_requested);
    }

    /// A chat's seed restates the user's messages since the model's last
    /// reply (context updates included, the runtime's report request and
    /// the report itself left out), with their images. Without any, it
    /// restates the task's initial prompt.
    #[test]
    fn a_chat_seed_restates_the_unanswered_messages() {
        let config = crate::agent_config::AgentConfig::builder(
            "chat".into(),
            "Chat".into(),
            crate::agent_config::AgentLLMConfig {
                model: arko_ai::Model::anthropic_claude_opus_5_5(),
                generation_config: Default::default(),
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .build()
        .unwrap();
        let mut task = Task::new(config, "Hello.".into(), "/w".into(), None);
        assert_eq!(HandoverSeed::for_chat(&task).task_prompt, "Hello.");
        let conversation = &mut task.conversation;
        conversation.add_user_message("Hello.");
        conversation.add_assistant_message("Hi.");
        conversation.add_user_message_with_image(
            "What is this?",
            arko_ai::Image::new_base64("cGl4", "image/png"),
        );
        conversation.add_user_message("# Workflow context update\n\nAlso this.");
        conversation.add_user_message(DEFAULT_CHAT_REPORT_REQUEST);
        task.runtime_messages.insert(4);
        task.conversation.add_assistant_message("The summary.");
        let seed = HandoverSeed::for_chat(&task);
        assert_eq!(
            seed.task_prompt,
            "What is this?\n\n# Workflow context update\n\nAlso this."
        );
        assert_eq!(unanswered_user_messages(&task).len(), 2);
        assert_eq!(unanswered_images(&task).len(), 1);
        // A chat's fresh conversation gets the unanswered images, and a
        // task's gets the task's own, which go with the prompt it restates.
        let chart = arko_ai::Image::new_base64("Y2hhcnQ=", "image/png");
        task.initial_task_images = vec![chart.clone()];
        assert_eq!(seed_images(&task, true).len(), 1);
        assert!(seed_images(&task, true)[0].as_image() != Some(&chart));
        let task_images = seed_images(&task, false);
        assert_eq!(task_images.len(), 1);
        assert_eq!(task_images[0].as_image(), Some(&chart));

        // The runtime's environment details aren't restated (the fresh
        // first message gets its own), and a message of an image alone
        // says so instead of answering the task's first prompt.
        let mut task = Task::new(
            task.agent_config.clone(),
            "Hello.".into(),
            "/w".into(),
            None,
        );
        let conversation = &mut task.conversation;
        conversation.add_user_message("Hello.");
        conversation.add_assistant_message("Hi.");
        conversation.add_user_message(
            "Look.\n\n# Current Environment Information (updated this turn)\ncwd: /w",
        );
        conversation.add_user_message_with_contents(vec![arko_ai::Content::Image(chart.clone())]);
        assert_eq!(
            HandoverSeed::for_chat(&task).task_prompt,
            "Look.\n\n(The user sent the image below.)"
        );
        assert_eq!(seed_images(&task, true).len(), 1);
    }

    #[test]
    fn the_seed_message_restates_the_task_then_the_report() {
        let seed = HandoverSeed {
            task_prompt: "Fix the parser.\n".into(),
            context: Some("# Agents you can message\n- `lead`".into()),
        };
        let message = seed.message("  Done: lexer. Next: parser.  ");
        assert!(message.starts_with("Fix the parser.\n\n# Handover report\n\n"));
        assert!(message.contains("Done: lexer. Next: parser.\n\n# Agents you can message"));
    }
}
