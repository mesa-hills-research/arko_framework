//! The runtime of agents whose model runs in Claude Code. See
//! [`crate::claude_code`].
//!
//! Claude Code runs the agent loop in a session of the installed CLI. This
//! runtime starts the session, sends it the task's user messages, answers
//! its hooks and permission requests (the gate every tool call passes),
//! serves the agent's arko tools in-process, and turns what the session
//! reports into the task's conversation, usage and events. The conversation
//! the task keeps mirrors the session's: it is a record for hosts (a
//! workflow reads which messages it holds), not what the model is sent.
//!
//! The session lives while the task is in progress. A wait (for an answer,
//! or an error-recovery decision) ends it, and the task resumes it
//! ([`Task::claude_code_session`]) when it continues. A handover starts a
//! new session, after the report. A turn over the model's input limit has
//! the report written in the session resumed at an earlier message
//! (`ClaudeCodeRun::hand_over_without_the_step`). A compaction ([`HandoverConfig::compact_in_claude_code`](crate::handover::HandoverConfig::compact_in_claude_code))
//! keeps it: the runtime sends Claude Code's `/compact` between turns, and
//! the agent goes on from Claude Code's summary.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::ContextImage;
use super::command::UserCommand;
use super::event::AgentEvent;
use super::tool_run::{self, Interception, RunEnvironment, RunOutcome, ToolRun};
use crate::agent_config::{AgentConfig, LLMErrorPolicy, PathSafetyPolicy, ToolErrorPolicy};
use crate::claude_code::{HELD_FOR_COMPACTION, built_in_path_parameters};
use crate::error::ArkoAgentsError;
use crate::handover::{
    DEFAULT_TEXT_REPORT_REQUEST, HandoverHook, HandoverRecord, HandoverRequest, HandoverSeed,
};
use crate::parsing::{PromptBuilder, ToolParsingConfig};
use crate::skill::SkillPromptContext;
use crate::snapshot::TaskSnapshots;
use crate::task::{
    ApprovalReason, CONTEXT_UPDATE, ErrorRecoveryCause, RunningToolCall, Task, TaskState,
};
use crate::tool::builtins::{ASK_USER_TOOL_NAME, COMPLETE_TASK_TOOL_NAME, message_user};
use crate::tool::{
    InterceptorConfig, ToolCallContext, ToolExecutionContext, ToolExecutionDecision,
    ToolExecutionMiddleware, ToolExecutionOutcome, ToolRecovery, ToolRegistry,
};
use arko_ai::claude_code as cc;
use arko_ai::{self as ai};

/// The in-process MCP server serving the agent's arko tools.
pub const ARKO_TOOL_SERVER: &str = "arko";

/// How long Claude Code waits for the runtime's answer to a hook: the gate
/// decides at once, except for a built-in call's middleware `before`.
const HOOK_TIMEOUT_SECONDS: f64 = 600.0;

/// Past a plan limit's reset before continuing, and the most added to
/// spread agents sharing a plan (as for other agents).
const USAGE_LIMIT_RESET_MARGIN: Duration = Duration::from_secs(1);
const USAGE_LIMIT_RESET_JITTER: Duration = Duration::from_secs(30);

/// Turns in a row that end without `complete_task` before the task is
/// interrupted (the native runtime's limit on unusable responses).
const MAX_OPEN_TURNS: u32 = 10;

const OPEN_TURN_NOTE: &str = "Your turn ended without calling complete_task, so the task is still open. Continue with your tools, or call mcp__arko__complete_task with a summary if the task is finished.";

/// The note after a turn that ended without text, for an agent whose reply
/// in text ends the task ([`AgentToolConfig::text_reply_ends_task`](crate::agent_config::AgentToolConfig::text_reply_ends_task)).
const OPEN_TURN_NO_REPLY_NOTE: &str = "Your turn ended without a reply, so the task is still open. Continue with your tools, or reply in plain text, without a tool call, if the task is finished: that reply ends the task.";

/// The note after an open turn whose final text was sent to the user as a
/// message ([`AgentToolConfig::text_reply_as_message`](crate::agent_config::AgentToolConfig::text_reply_as_message)).
const OPEN_TURN_MESSAGE_NOTE: &str = "Your reply was sent to the user as a message. Your turn ended without calling complete_task, so the task is still open: continue with your tools, or call mcp__arko__complete_task with a summary if the task is finished.";

const NO_HANDOVER_REPORT: &str = "Your reply had no text, so no handover report was received. Reply with the handover report as plain text, without calling a tool.";

const TASK_COMPLETE: &str = "The task is complete; end your turn without calling tools.";

const ASKED: &str = "Your questions were sent to the user. End your turn now without calling tools; the answer arrives as the next message.";

const CONTINUE: &str = "Continue the task from where you left off.";

/// The most characters of context a hook carries. Claude Code replaces a
/// hook's text of over 10,000 characters with a preview of 2 KB and the name
/// of a file, so what doesn't fit waits for the next hook (or the next
/// input).
const HOOK_MAX_CHARS: usize = 9_500;

/// How long a stopped session waits for the CLI to say what became of a note
/// sent just before.
const NOTE_VERDICT_WAIT: Duration = Duration::from_secs(1);
/// How long what Claude Code's usage data says about extra usage stands
/// before an event asks for it again.
const USAGE_READ_EVERY: Duration = Duration::from_secs(300);
/// How long the end of a turn waits for Claude Code's answer about its usage.
const USAGE_END_WAIT: Duration = Duration::from_secs(1);

/// What the runtime sends to have Claude Code compact the session's
/// conversation: Claude Code's own command, as a user message.
const COMPACT_COMMAND: &str = "/compact";

/// The model's next input after a compaction the runtime asked for, when no
/// other input (a user's message) continues the task.
const COMPACTED: &str = "Claude Code compacted your conversation: the summary above replaces the earlier messages. Continue the task from where you left off.";

/// Heads the list of the calls held while a compaction was due.
const HELD_CALLS: &str = "For the compaction, arko held these tool calls of yours and ended your turn, so they did not run. Where Claude Code said the user rejected them, that was this hold, not the user. Make them again if they are still needed:";

/// A call's result as Claude Code is served it: `text`, then each image's
/// label on a line of its own, and the images as MCP image content.
fn served_output(text: String, images: Vec<crate::tool::ToolImage>) -> cc::ToolOutput {
    if images.is_empty() {
        return cc::ToolOutput::text(text);
    }
    let mut text = text;
    for image in images.iter().filter(|image| !image.description.is_empty()) {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&format!("Image: {}", image.description));
    }
    cc::ToolOutput::text(text).with_images(images.into_iter().map(|image| image.image).collect())
}

/// The error a refusal resolution gets: Claude Code sends its own
/// conversation.
const CANT_RESOLVE_REFUSAL: &str = "Could not resolve the refusal. A Claude Code agent's conversation is kept by Claude Code, so the refused input can't be changed. Continue or fail the task instead.";

/// Heads the note on the calls that were open when the task stopped. See
/// [`ClaudeCodeRun::recover_open_calls`].
const RESTARTED_NOTE: &str = "The task stopped while tool calls were open (arko's host or Claude Code stopped). As arko recorded them, and whatever Claude Code reports of them (it may call them interrupted):";

/// The arko tool a Claude Code tool name serves, if it is one.
fn arko_name(served: &str) -> Option<&str> {
    served
        .strip_prefix("mcp__")
        .and_then(|rest| rest.strip_prefix(ARKO_TOOL_SERVER))
        .and_then(|rest| rest.strip_prefix("__"))
}

/// A canonical key of a call's input, to match a gated arko call with the
/// MCP call that runs it (serde_json objects are sorted by key).
fn input_key(name: &str, input: &Value) -> (String, String) {
    (name.to_string(), input.to_string())
}

fn params_of(input: &Value) -> HashMap<String, Value> {
    input
        .as_object()
        .map(|object| object.clone().into_iter().collect())
        .unwrap_or_default()
}

/// What Claude Code's hooks and permission requests ask the runtime.
enum HostRequest {
    PreToolUse {
        tool_use_id: Option<String>,
        tool_name: String,
        input: Value,
        reply: oneshot::Sender<Value>,
    },
    Permission {
        request: cc::PermissionRequest,
        reply: oneshot::Sender<cc::PermissionDecision>,
    },
    PostToolUse {
        tool_use_id: Option<String>,
        tool_name: String,
        response: Value,
        reply: oneshot::Sender<Value>,
    },
}

struct HookForwarder {
    event: cc::HookEvent,
    requests: mpsc::UnboundedSender<HostRequest>,
}

#[async_trait]
impl cc::HookCallback for HookForwarder {
    async fn call(&self, input: Value, tool_use_id: Option<String>) -> Value {
        let tool_use_id = tool_use_id.or_else(|| {
            input
                .get("tool_use_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        let tool_name = input["tool_name"].as_str().unwrap_or_default().to_string();
        let (reply, answer) = oneshot::channel();
        let request = match self.event {
            cc::HookEvent::PreToolUse => HostRequest::PreToolUse {
                tool_use_id,
                tool_name,
                input: input.get("tool_input").cloned().unwrap_or(Value::Null),
                reply,
            },
            _ => HostRequest::PostToolUse {
                tool_use_id,
                tool_name,
                response: input.get("tool_response").cloned().unwrap_or(Value::Null),
                reply,
            },
        };
        if self.requests.send(request).is_err() {
            return stopping_answer(&self.event);
        }
        answer
            .await
            .unwrap_or_else(|_| stopping_answer(&self.event))
    }
}

fn stopping_answer(event: &cc::HookEvent) -> Value {
    match event {
        cc::HookEvent::PreToolUse => {
            cc::Hook::deny_tool("arko did not run this call because the agent is stopping.")
        }
        _ => cc::Hook::proceed(),
    }
}

/// What the model reads for a call the runtime refused. Claude Code shows
/// a hook's refusal as `PreToolUse:<tool> hook error: <text>`, so the text
/// says who refused the call and why.
fn refused(tool: &str, reason: &str) -> Value {
    cc::Hook::deny_tool(format!("arko did not run this {tool} call: {reason}"))
}

/// A call held while the handover report is due: not an error.
fn held_for_report(tool: &str, request: &str) -> Value {
    cc::Hook::deny_tool(format!(
        "Not an error: arko held this {tool} call so you can write your handover report now. {request}"
    ))
}

/// A call held while a compaction is due: not an error.
fn held_for_compaction(tool: &str) -> Value {
    cc::Hook::deny_tool(format!(
        "Not an error: arko held this {tool} call because Claude Code compacts your conversation now. End your turn without calling a tool; you continue after the compaction, and can make this call again then."
    ))
}

fn allow_tool() -> Value {
    json!({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}})
}

/// Sends the call to the permission request, where it waits for approval.
fn ask_tool() -> Value {
    json!({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask"}})
}

/// The input a built-in call runs with so that Claude Code runs it in
/// `directory`, the task's working directory, where the CLI doesn't run after
/// the task moved (see [`Task::claude_code_directory`]). The command guard and
/// the path policy judge the call from `directory`. `None` when the call
/// needs no change.
///
/// - `Bash` and `PowerShell` commands start with a change into `directory`.
///   A failed change ends the command before it runs.
/// - A path that names a file or a directory is made absolute from
///   `directory`, the way the path policy resolves it. A `Glob` or `Grep`
///   without a `path` searches `directory`.
fn in_directory(name: &str, input: &Value, directory: &Path) -> Option<Value> {
    let mut moved = input.as_object()?.clone();
    match name {
        "Bash" | "PowerShell" => {
            let command = moved.get("command")?.as_str()?;
            let text = directory.to_string_lossy();
            let change = if name == "Bash" {
                // Git Bash on Windows reads `C:/dir` and not `C:\dir`.
                let text = if cfg!(windows) {
                    text.replace('\\', "/")
                } else {
                    text.into_owned()
                };
                format!("cd -- '{}' || exit 1", text.replace('\'', r"'\''"))
            } else {
                // PowerShell reads the curly quotes as straight ones.
                let quoted: String = text
                    .chars()
                    .flat_map(|c| {
                        let doubled = matches!(c, '\'' | '‘' | '’' | '‚' | '‛');
                        std::iter::once(c).chain(doubled.then_some(c))
                    })
                    .collect();
                format!("Set-Location -LiteralPath '{quoted}' -ErrorAction Stop")
            };
            let command = format!("{change}\n{command}");
            moved.insert("command".into(), Value::String(command));
        }
        _ => {
            let parameters = built_in_path_parameters(name);
            if parameters.is_empty() {
                return None;
            }
            for parameter in parameters {
                let given = match moved.get(*parameter) {
                    Some(Value::String(path)) if !path.trim().is_empty() => Some(path.as_str()),
                    _ => None,
                };
                let path = match given {
                    // An absolute path, or one from the home directory,
                    // doesn't depend on where Claude Code runs.
                    Some(path) if path.starts_with('~') || Path::new(path).is_absolute() => {
                        continue;
                    }
                    Some(path) => path,
                    // Without a path, the search tools use the CLI's directory.
                    None if matches!(name, "Glob" | "Grep") => ".",
                    None => continue,
                };
                let absolute = crate::utils::resolve_path(path, directory).ok()?;
                moved.insert(
                    parameter.to_string(),
                    Value::String(absolute.to_string_lossy().into_owned()),
                );
            }
        }
    }
    let moved = Value::Object(moved);
    (moved != *input).then_some(moved)
}

/// `input` with its file path moved to the task's directory, when a file
/// tool names one in `from`, the directory the CLI still runs in after the
/// task moved. Claude Code makes a relative `file_path` (`notebook_path`)
/// absolute from `from` before the hooks see it, so a path under `from` that
/// is outside `directory`, the task's working directory, is the same path
/// under `directory`. A path inside `directory` stays as it is. `None` when
/// the call needs no change.
fn from_session_directory(
    name: &str,
    input: &Value,
    from: &Path,
    directory: &Path,
) -> Option<Value> {
    let parameter = match name {
        "Read" | "Write" | "Edit" | "MultiEdit" => "file_path",
        "NotebookEdit" => "notebook_path",
        _ => return None,
    };
    let path = input.get(parameter)?.as_str()?;
    if crate::utils::is_path_in_workspace(path, directory).ok()? {
        return None;
    }
    // Below `from` by the rule the path policy uses (case and `\\?\` on Windows).
    let relative = crate::utils::path_in_workspace(path, from).ok()??;
    let moved = crate::utils::resolve_path(&relative.to_string_lossy(), directory).ok()?;
    let mut mapped = input.clone();
    mapped[parameter] = Value::String(moved.to_string_lossy().into_owned());
    Some(mapped)
}

fn add_context(text: &str) -> Value {
    json!({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text}})
}

/// User message content as Claude Code's input blocks: text and images (an
/// image given by URL as a line of text).
fn input_blocks<'a>(contents: impl Iterator<Item = &'a ai::Content>) -> Value {
    Value::Array(
        contents
            .filter_map(|content| match content {
                ai::Content::Text(text) => Some(json!({"type": "text", "text": text})),
                ai::Content::Image(ai::Image::Base64 { data, mime_type }) => Some(json!({
                    "type": "image",
                    "source": {"type": "base64", "data": &**data, "media_type": mime_type},
                })),
                ai::Content::Image(ai::Image::Url(url)) => {
                    Some(json!({"type": "text", "text": format!("Image: {url}")}))
                }
                _ => None,
            })
            .collect(),
    )
}

struct PermissionForwarder {
    requests: mpsc::UnboundedSender<HostRequest>,
}

#[async_trait]
impl cc::PermissionHandler for PermissionForwarder {
    async fn decide(&self, request: cc::PermissionRequest) -> cc::PermissionDecision {
        let (reply, answer) = oneshot::channel();
        if self
            .requests
            .send(HostRequest::Permission { request, reply })
            .is_err()
        {
            return cc::PermissionDecision::deny("The agent is stopping.");
        }
        answer
            .await
            .unwrap_or_else(|_| cc::PermissionDecision::deny("The agent is stopping."))
    }
}

/// How an arko tool call ended, for the runtime to report.
#[derive(Debug, Clone)]
enum CallOutcome {
    Finished {
        raw: String,
        effective: String,
        intercepted: bool,
    },
    Rejected(String),
    Failed(String),
    Stop {
        reason: String,
        cancel: bool,
    },
    Pause {
        message: String,
        reason: String,
    },
}

/// A gated arko call, waiting for its MCP call or run again after a stop.
struct Admitted {
    id: Uuid,
    /// Claude Code's ID of the call (its `tool_use_id`).
    call_id: Option<String>,
    custom_output: Option<String>,
    /// What Claude Code reads of the call in place of its result: an
    /// `ask_user` or `close_questions` call's, which the runtime words as
    /// it admits the call.
    reply: Option<String>,
}

/// An `ask_user` or `close_questions` call the runtime took as it admitted
/// it, for when its result arrives.
enum QuestionCall {
    Asked {
        asked: Vec<crate::questions::AskedQuestion>,
        wait: bool,
        result: String,
    },
    Closed {
        ids: Vec<String>,
        why: Option<String>,
        result: String,
    },
}

/// Runs the agent's arko tools for Claude Code's MCP calls.
struct ToolHost {
    environment: Arc<RunEnvironment>,
    config: AgentConfig,
    registry: Arc<ToolRegistry>,
    interceptor: Option<InterceptorConfig>,
    admitted: Mutex<HashMap<(String, String), VecDeque<Admitted>>>,
    outcomes: Mutex<HashMap<Uuid, (CallOutcome, Vec<crate::tool::ToolImage>)>>,
}

impl ToolHost {
    fn interception_for(&self, name: &str, tool: &dyn crate::tool::Tool) -> Option<Interception> {
        let config = self.interceptor.as_ref()?;
        if !tool.allow_result_interception()
            || [
                COMPLETE_TASK_TOOL_NAME,
                ASK_USER_TOOL_NAME,
                crate::tool::builtins::CLOSE_QUESTIONS_TOOL_NAME,
            ]
            .contains(&name)
        {
            return None;
        }
        let policy = if name == crate::skill::LOAD_SKILL_TOOL_NAME {
            self.config
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

    async fn call(&self, name: &str, arguments: Value) -> cc::ToolOutput {
        let Some(tool) = self.registry.get(name) else {
            return cc::ToolOutput::error(format!("Tool '{name}' is not available."));
        };
        let admitted = self
            .admitted
            .lock()
            .unwrap()
            .get_mut(&input_key(name, &arguments))
            .and_then(VecDeque::pop_front);
        let Some(admitted) = admitted else {
            // Every call passes the gate first; one that didn't isn't run.
            return cc::ToolOutput::error(format!(
                "Tool '{name}' was called without passing the host's checks, so it was not run."
            ));
        };
        let id = admitted.id;
        let (outcome, output) = self.run(name, tool, &arguments, admitted, false).await;
        // The CLI may report only a call's text in its hook. Keep the
        // actual images until the runtime records that completed result.
        let images = output
            .images
            .iter()
            .cloned()
            .map(|image| crate::tool::ToolImage {
                description: String::new(),
                image,
            })
            .collect();
        self.outcomes.lock().unwrap().insert(id, (outcome, images));
        output
    }

    /// Runs the admitted call `call` to `tool` (`recovering`: again, after
    /// the task stopped while it ran), and returns its outcome and what
    /// Claude Code reads.
    async fn run(
        &self,
        name: &str,
        tool: Arc<dyn crate::tool::Tool>,
        arguments: &Value,
        call: Admitted,
        recovering: bool,
    ) -> (CallOutcome, cc::ToolOutput) {
        let Admitted {
            id,
            call_id,
            custom_output,
            reply,
        } = call;
        let mut params = params_of(arguments);
        crate::tool::builtins::read_file::ignore_line_ranges(
            tool.as_ref(),
            &self.config.tool_config,
            &mut params,
        );
        let run = ToolRun {
            tool_call_id: id,
            tool_name: name.to_string(),
            timeout: self
                .config
                .tool_config
                .tool_timeout(name, tool.uses_host_timeout()),
            parallel_safe: false,
            interception: self.interception_for(name, tool.as_ref()),
            interruptible: tool.interruptible(),
            background: tool.background_mode(),
            params,
            custom_output,
            tool,
            call_id,
            message_to_agent: None,
            recovering,
        };
        let outcome = match tool_run::run_tool(self.environment.clone(), run).await {
            Ok(outcome) => outcome,
            Err(error) => RunOutcome::Failed(error),
        };
        match outcome {
            RunOutcome::Finished {
                output,
                effective,
                intercepted,
            } => {
                let text = match (reply, name) {
                    (Some(reply), _) => reply,
                    (None, ASK_USER_TOOL_NAME) => ASKED.to_string(),
                    (None, _) => effective.clone(),
                };
                let reply = served_output(text, output.images);
                (
                    CallOutcome::Finished {
                        raw: output.text,
                        effective,
                        intercepted,
                    },
                    reply,
                )
            }
            RunOutcome::MiddlewareErrorAfter {
                output,
                effective,
                intercepted,
                reason,
                is_error,
            } => {
                let _ = (intercepted, is_error);
                (
                    CallOutcome::Pause {
                        message: effective.clone(),
                        reason,
                    },
                    served_output(effective, output.images),
                )
            }
            RunOutcome::Rejected(message) => (
                CallOutcome::Rejected(message.clone()),
                cc::ToolOutput::error(message),
            ),
            RunOutcome::MiddlewareErrorBefore { reason } => (
                CallOutcome::Pause {
                    message: reason.clone(),
                    reason: reason.clone(),
                },
                cc::ToolOutput::error(reason),
            ),
            RunOutcome::Stop { reason, cancel } => (
                CallOutcome::Stop {
                    reason: reason.clone(),
                    cancel,
                },
                cc::ToolOutput::error(reason),
            ),
            RunOutcome::Pause { message, reason } => (
                CallOutcome::Pause {
                    message: message.clone(),
                    reason,
                },
                cc::ToolOutput::error(message),
            ),
            RunOutcome::Failed(error) => {
                let text = format!("Tool '{name}' execution failed: {error}");
                (
                    CallOutcome::Failed(text.clone()),
                    cc::ToolOutput::error(text),
                )
            }
            // Claude Code gets the job's ID now, and the result comes as a
            // note when the job ends.
            RunOutcome::Moved { text } => (
                CallOutcome::Finished {
                    raw: text.clone(),
                    effective: text.clone(),
                    intercepted: false,
                },
                served_output(text, Vec::new()),
            ),
        }
    }
}

/// One arko tool, served to Claude Code.
struct ServedTool {
    name: String,
    description: String,
    schema: Value,
    host: Arc<ToolHost>,
}

#[async_trait]
impl cc::ClaudeTool for ServedTool {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn input_schema(&self) -> Value {
        self.schema.clone()
    }
    async fn call(&self, arguments: Value) -> cc::ToolOutput {
        self.host.call(&self.name, arguments).await
    }
}

/// A response's usage, until the response is complete.
struct PendingUsage {
    key: String,
    usage: Value,
    model: Option<String>,
    /// Why the response stopped, and the details (a refusal's category
    /// and explanation), from its `message_delta`.
    stop_reason: Option<String>,
    stop_details: Option<Value>,
}

impl PendingUsage {
    /// A request's whole input: uncached, cache reads and cache writes.
    fn input(usage: &Value) -> usize {
        [
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ]
        .iter()
        .map(|field| usage.get(field).and_then(Value::as_u64).unwrap_or(0) as usize)
        .sum()
    }
}

/// A tool call of the session, by its Claude Code `tool_use_id`.
struct Call {
    id: Uuid,
    /// The arko tool's name, or the built-in's.
    name: String,
    built_in: bool,
    record: ai::ToolCall,
    input: Value,
}

/// A call waiting for approval in a permission request.
struct Approval {
    tool_use_id: Option<String>,
    id: Uuid,
    name: String,
    input: Value,
    reply: oneshot::Sender<cc::PermissionDecision>,
    deadline: Option<tokio::time::Instant>,
}

/// A compaction of the session's conversation with Claude Code's
/// `/compact` (see [`HandoverConfig::compact_in_claude_code`](crate::handover::HandoverConfig::compact_in_claude_code)).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Compaction {
    /// Due: the model's next tool call is held and its turn interrupted
    /// (its other calls are held too), and `/compact` is sent when the turn
    /// ends (or first, when a session starts).
    Due,
    /// `/compact` was sent, and `compacted` once Claude Code reported the
    /// compaction.
    Sent { compacted: bool },
}

/// A message of the session's conversation, by the UUID that names it in
/// Claude Code's transcript, with the length of the task's conversation
/// once it was recorded there.
#[derive(Debug, Clone)]
struct Entry {
    uuid: String,
    messages: usize,
}

/// How a session ended.
enum SessionEnd {
    /// The task stopped (it ended, was suspended, or its host is gone).
    Stop,
    /// The task waits for a command. The session is closed meanwhile.
    Wait,
    /// A handover (the next session starts a fresh conversation), or a
    /// step left out (it resumes the session at an earlier message).
    Restart,
}

/// A stop the host asked for while the session ran.
#[derive(Clone, Copy)]
enum Stopping {
    Cancel,
    Suspend,
    Closed,
}

pub(super) struct ClaudeCodeRun {
    task: Task,
    registry: Arc<ToolRegistry>,
    events: mpsc::Sender<AgentEvent>,
    commands: mpsc::Receiver<UserCommand>,
    pub(super) turn_context: Option<Arc<dyn super::TurnContextSource>>,
    /// `ContextInjected` events for context taken and not yet reported.
    context_receipts: Vec<AgentEvent>,
    /// The identifiers of context taken that waits with images for the
    /// next input (see [`take_context`](Self::take_context)).
    held_context_ids: Vec<String>,
    pub(super) host_decisions: super::HostDecisions,
    pub(super) handover_hook: Option<Arc<dyn HandoverHook>>,
    /// The host's handover threshold, in place of the configuration's.
    pub(super) handover_threshold: crate::handover::HandoverThreshold,
    middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    interceptor: Option<InterceptorConfig>,
    skill_prompt_context: Option<SkillPromptContext>,
    parsing: ToolParsingConfig,

    calls: HashMap<String, Call>,
    /// Each thread's (the main conversation's, or a subagent's) latest
    /// request's usage, recorded once the request is complete. See
    /// [`observe_usage`](Self::observe_usage).
    pending_usage: HashMap<Option<String>, PendingUsage>,
    /// Requests whose usage the stream events report.
    streamed: std::collections::HashSet<String>,
    /// Indices in `usage_history` of subagents' requests, which the cache
    /// monitor leaves out.
    subagent_calls: Vec<usize>,
    /// The latest plan usage Claude Code reported (`rate_limit_event`), and
    /// the reset of a limit it reported reached.
    plan_usage: Option<ai::PlanUsage>,
    limit_reached: Option<Option<u64>>,
    /// The answer awaited to the request for Claude Code's usage data, when
    /// it was asked, and the extra usage it last showed.
    usage_reply: Option<oneshot::Receiver<Result<Value, String>>>,
    usage_asked: Option<std::time::Instant>,
    usage_read: Option<ai::ExtraUsage>,
    /// The main conversation's prompt-cache lifetime, as Claude Code's cache
    /// writes show it (1 hour or 5 minutes).
    cache_lifetime: Option<Duration>,
    /// The tools served to Claude Code as arko's MCP server, for the
    /// request sizes `ModelRequestStarted` reports.
    served_tools: Option<ai::ToolConfig>,
    /// The bytes per token of the conversation's requests so far, as the
    /// sizes reported for them and their input show it.
    bytes_per_token: f64,
    /// The plan-usage threshold's `SlowDown` delay, while usage is over it.
    slow_down: Option<Duration>,
    /// Allowed calls the `SlowDown` holds until their deadline: the request
    /// after each call waits with it. Commands are handled meanwhile.
    slowed: VecDeque<(tokio::time::Instant, oneshot::Sender<Value>, Value)>,
    cache_warned: bool,
    cache_pause_episode: bool,
    /// The cache monitor asked for a pause. The turn is interrupted.
    cache_interrupt: bool,
    /// A refusal Claude Code reported standing in this turn (no retry ran).
    refused: Option<ai::Refusal>,
    /// The `stop_details` of the last response the API refused. Claude
    /// Code's refusal message can leave the explanation out. These have it.
    refusal_details: Option<Value>,
    /// The refusal fallback model answers this turn. The configured model
    /// takes over again when it ends.
    fallback_turn: bool,
    /// The input of the current conversation's latest and first requests
    /// (the first also when an earlier run made it). A conversation is
    /// current from its start, its handover or its latest compaction.
    context_size: Option<usize>,
    first_context_size: Option<usize>,
    open_turns: u32,
    /// Questions `ask_user` sent in this turn that the agent waits for:
    /// their text and the questions.
    questions: Option<(String, Vec<crate::questions::AskedQuestion>)>,
    /// The `ask_user` and `close_questions` calls admitted whose results
    /// haven't arrived, by call ID.
    question_calls: HashMap<Uuid, QuestionCall>,
    /// The highest question number admitted calls took, recorded or not.
    questions_numbered: u32,
    /// The report request reached the model in this turn (a denied call's
    /// result, or a user message).
    report_delivered: bool,
    approval: Option<Approval>,
    /// Text to add after a call's result: a message given with its approval.
    after_result: HashMap<String, String>,
    /// Context delivered with the latest results, mirrored after them.
    delivered_context: Vec<String>,
    /// The marks of the context taken out of the task to be sent
    /// ([`TurnContext::marks`]), kept by the task once it holds that
    /// context: in a message it records, or pending again.
    taken_marks: BTreeMap<String, u64>,
    /// The marks of `delivered_context`, kept by the task once it is
    /// mirrored after the results.
    delivered_marks: BTreeMap<String, u64>,
    /// The notes the user typed that were sent as messages, by the UUID of the
    /// message, until the CLI reports what became of them, each with the
    /// marks of its update when the note was all of it.
    sent_notes: HashMap<String, (String, BTreeMap<String, u64>)>,
    /// A pause the turn's calls asked for, applied when the turn ends.
    pause_after_turn: Option<(String, ErrorRecoveryCause)>,
    /// The turn's tool errors under `ToolErrorPolicy::PauseAndAwaitApproval`
    /// (each with its call), reported with the tool-error pause at the
    /// turn's end, as arko's own runtime reports them: once.
    tool_errors: Vec<(String, Uuid)>,
    /// The kind of API error Claude Code reported in the turn's latest
    /// message (its `error` field: `authentication_failed`, `rate_limit`,
    /// …), if it reported one.
    api_error: Option<String>,
    /// The kind of the failed turn's error, recorded with the pause (or the
    /// error event) that reports it.
    error_kind: Option<arko_ai::ErrorKind>,
    stopping: Option<Stopping>,
    handover_skip_warned: bool,
    /// A compaction under way.
    compaction: Option<Compaction>,
    /// The calls held while it was due, as the model is told of them after.
    held_calls: Vec<String>,
    /// The latest message the session received whose UUID arko knows: a
    /// user message it sent, or a tool result.
    last_entry: Option<Entry>,
    /// The message before the model's latest response: the end of the
    /// conversation as the last request Claude Code answered had it, and
    /// whether tool results followed that response. A turn over the model's
    /// input limit hands over from it. See
    /// [`hand_over_without_the_step`](Self::hand_over_without_the_step).
    answered: Option<Entry>,
    step_results: bool,
    /// The message ID of the model's latest response.
    response: Option<String>,
    /// The main conversation's response streaming now (its message ID).
    streaming: Option<String>,
    /// The responses whose text or thinking streamed as it came, which their
    /// complete messages don't report again.
    streamed_text: HashSet<String>,
    /// When Claude Code started retrying the current request, while it does.
    retrying_since: Option<std::time::Instant>,
    /// The bytes of text and thinking the response streaming now has sent
    /// the host, until it finishes (its final `message_delta`).
    streamed_bytes: (usize, usize),
    /// What the session will have received once Claude Code has the input
    /// sent after a cut back or a branch (see `record_sent`).
    synced_on_send: Option<usize>,
    /// A step was left out for the handover under way (once per handover).
    step_left_out: bool,
    /// The message the next session resumes at.
    resume_at: Option<String>,
    host: Option<Arc<ToolHost>>,
    /// Receives the task at its safe points.
    pub(super) snapshots: TaskSnapshots,
    /// The host's stop signal for the task's arko tool calls. See
    /// [`AgentRuntime::with_stop_signal`](super::AgentRuntime::with_stop_signal).
    pub(super) host_stop: crate::stop::StopSignal,
    /// Receives the images the task's calls return. See
    /// [`AgentRuntime::with_tool_image_sink`](super::AgentRuntime::with_tool_image_sink).
    pub(super) image_sink: Option<Arc<dyn crate::tool::ToolImageSink>>,
    /// Fired when the host stops the session, so that a running arko tool
    /// call winds down. See [`TaskContext::stop`](crate::tool::TaskContext::stop).
    calls_stop: crate::stop::StopSource,
    /// Notified as context arrives, for the arko tool calls running. See
    /// [`TaskContext::context_arrived`](crate::tool::TaskContext::context_arrived).
    context_arrivals: crate::tool::ContextSource,
    /// The terminal manager arko tool calls move to the background with.
    /// See [`AgentRuntime::with_background_jobs`](super::AgentRuntime::with_background_jobs).
    #[cfg(any(unix, windows))]
    pub(super) background_jobs: Option<Arc<crate::terminal::TerminalManager>>,
}

impl ClaudeCodeRun {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        task: Task,
        registry: Arc<ToolRegistry>,
        events: mpsc::Sender<AgentEvent>,
        commands: mpsc::Receiver<UserCommand>,
        skill_prompt_context: Option<SkillPromptContext>,
        interceptor: Option<InterceptorConfig>,
        middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    ) -> Self {
        let offered = task.agent_config.tool_config.offered_tool_names();
        let registry = Arc::new(registry.restricted_to(offered.iter().map(String::as_str)));
        for name in registry.list_tool_names() {
            if let Some(tool) = registry.get(&name) {
                tool.reset_context(task.id);
            }
        }
        let parsing = task.agent_config.get_tool_parsing_config();
        let first_context_size = task
            .usage_history
            .get(task.conversation_usage_start)
            .map(|usage| usage.input_tokens);
        Self {
            task,
            registry,
            events,
            commands,
            turn_context: None,
            context_receipts: Vec::new(),
            held_context_ids: Vec::new(),
            host_decisions: super::HostDecisions::default(),
            handover_hook: None,
            handover_threshold: Default::default(),
            middleware,
            interceptor,
            skill_prompt_context,
            parsing,
            calls: HashMap::new(),
            pending_usage: HashMap::new(),
            streamed: Default::default(),
            subagent_calls: Vec::new(),
            plan_usage: None,
            limit_reached: None,
            usage_reply: None,
            usage_asked: None,
            usage_read: None,
            cache_lifetime: None,
            served_tools: None,
            bytes_per_token: super::result_budget::DEFAULT_BYTES_PER_TOKEN,
            slow_down: None,
            slowed: VecDeque::new(),
            cache_warned: false,
            cache_pause_episode: false,
            cache_interrupt: false,
            refused: None,
            refusal_details: None,
            fallback_turn: false,
            context_size: None,
            first_context_size,
            open_turns: 0,
            questions: None,
            question_calls: HashMap::new(),
            questions_numbered: 0,
            report_delivered: false,
            approval: None,
            after_result: HashMap::new(),
            delivered_context: Vec::new(),
            taken_marks: BTreeMap::new(),
            delivered_marks: BTreeMap::new(),
            sent_notes: HashMap::new(),
            pause_after_turn: None,
            tool_errors: Vec::new(),
            api_error: None,
            error_kind: None,
            stopping: None,
            handover_skip_warned: false,
            compaction: None,
            held_calls: Vec::new(),
            last_entry: None,
            answered: None,
            step_results: false,
            response: None,
            step_left_out: false,
            resume_at: None,
            streaming: None,
            streamed_text: HashSet::new(),
            retrying_since: None,
            streamed_bytes: (0, 0),
            synced_on_send: None,
            host: None,
            snapshots: TaskSnapshots::default(),
            host_stop: crate::stop::StopSignal::never(),
            image_sink: None,
            calls_stop: crate::stop::StopSource::new(),
            context_arrivals: crate::tool::ContextSource::new(),
            #[cfg(any(unix, windows))]
            background_jobs: None,
        }
    }

    /// Binds the task's tools to its command channel.
    pub(super) fn bind_tools(&self, sender: &mpsc::Sender<UserCommand>) {
        for name in self.registry.list_tool_names() {
            if let Some(tool) = self.registry.get(&name) {
                tool.bind_runtime(self.task.id, sender.downgrade());
            }
        }
        // Its background jobs' notes go there too.
        #[cfg(any(unix, windows))]
        if let Some(jobs) = &self.background_jobs {
            jobs.bind_runtime(self.task.id, sender.downgrade());
        }
    }

    async fn emit(&self, event: AgentEvent) -> Result<(), ArkoAgentsError> {
        self.events
            .send(event)
            .await
            .map_err(|error| ArkoAgentsError::ChannelSend(format!("{error:?}")))
    }

    async fn set_state(&mut self, state: TaskState) -> Result<(), ArkoAgentsError> {
        self.task.set_state(state);
        self.emit(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.report(false).await;
        Ok(())
    }

    /// Reports the task to the snapshot sink, if there is one. With
    /// `durable`, returns once the sink has saved it.
    async fn report(&mut self, durable: bool) {
        self.snapshots.report(&self.task, durable).await;
    }

    /// Records the call Claude Code's `tool_use_id` names as running before
    /// it is allowed to run, and waits until the task is saved with it (the
    /// write-ahead point): a task that stops while it runs knows it may have
    /// run. A read-only call (a [read-only built-in](crate::claude_code::READ_ONLY_BUILT_IN_TOOLS),
    /// or an arko tool whose calls recovery runs again) doesn't wait: if
    /// the task stops before it is saved, the model reads again.
    async fn record_running(
        &mut self,
        tool_use_id: Option<&String>,
        custom_output: Option<String>,
    ) {
        let Some(call) = tool_use_id.and_then(|tool_use_id| self.calls.get(tool_use_id)) else {
            return;
        };
        let running = RunningToolCall {
            tool_call_id: call.id,
            call_id: tool_use_id.cloned(),
            tool_name: call.name.clone(),
            parameters: params_of(&call.input),
            custom_output,
            message_to_agent: None,
            built_in: call.built_in,
            output: None,
            output_images: Vec::new(),
        };
        let rerunnable = running.custom_output.is_none()
            && if running.built_in {
                crate::claude_code::READ_ONLY_BUILT_IN_TOOLS.contains(&running.tool_name.as_str())
            } else {
                self.registry
                    .get(&running.tool_name)
                    .is_some_and(|tool| tool.recovery_mode() == ToolRecovery::Rerun)
            };
        self.task
            .running_tool_calls
            .retain(|recorded| recorded.call_id != running.call_id);
        self.task.running_tool_calls.push(running);
        self.report(!rerunnable).await;
    }

    /// A call's result, as Claude Code reported it: kept with the running
    /// call until the model's next message shows Claude Code has it (its
    /// transcript is saved a little after the result is reported).
    fn record_output(&mut self, tool_use_id: Option<&String>, content: &Value) {
        if let Some(running) =
            self.task.running_tool_calls.iter_mut().find(|running| {
                running.call_id.is_some() && running.call_id.as_ref() == tool_use_id
            })
        {
            running.output = Some(result_text(content));
            let images = self.host.as_ref().and_then(|host| {
                host.outcomes
                    .lock()
                    .unwrap()
                    .get(&running.tool_call_id)
                    .map(|(_, images)| images.clone())
            });
            if let Some(images) = images {
                running.output_images = images;
            } else {
                let images = result_images(content);
                if !images.is_empty() {
                    running.output_images = images;
                }
            }
        }
    }

    pub(super) async fn run(mut self) -> Result<Task, ArkoAgentsError> {
        self.task.agent_config.validate()?;
        if self.task.conversation.system_message.is_none() {
            let prompt = self.system_prompt()?;
            self.task.conversation.system_message = Some(prompt);
        }
        if matches!(self.task.current_state, TaskState::Pending) {
            self.set_state(TaskState::InProgress).await?;
        }
        // Images found gone become lines saying so, and are reported.
        self.task.settle_unavailable_images();
        if !self.task.unavailable_images.is_empty() {
            let reasons = std::mem::take(&mut self.task.unavailable_images);
            self.emit(AgentEvent::ImagesUnavailable { reasons }).await?;
        }
        if self.task.conversation.messages.is_empty() {
            let builder = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing, None);
            let first =
                builder.format_initial_task_prompt(&self.task, &self.task.initial_task_prompt)?;
            // Moved: the first message holds them from now on.
            let content =
                crate::task::with_images(first, std::mem::take(&mut self.task.initial_task_images));
            self.task
                .conversation
                .add_user_message_with_contents(content);
            self.task.increment_user_message_count();
        }
        // A call's approval can't outlive its session: a task saved while
        // one waited continues without it.
        if matches!(
            self.task.current_state,
            TaskState::AwaitingToolApproval { .. } | TaskState::AwaitingBatchToolApproval { .. }
        ) {
            self.add_user_message(
                "The run stopped while a tool call waited for approval, so the call was not run. Call it again if it is still needed.",
            )?;
            self.set_state(TaskState::InProgress).await?;
        }
        let result = self.drive().await;
        self.take_last_context().await;
        self.report(false).await;
        let _ = self.emit(AgentEvent::TaskStreamEnd).await;
        result?;
        log::info!(
            "Claude Code agent run finished for task {}: {:?}",
            self.task.id,
            self.task.current_state
        );
        Ok(self.task)
    }

    fn system_prompt(&self) -> Result<String, ArkoAgentsError> {
        // Offered nothing, the agent is told nothing about tools.
        let offers_tools = !self.offers_no_tools();
        let builder = PromptBuilder::new(
            &self.task.agent_config,
            vec![],
            &self.parsing,
            self.skill_prompt_context.as_ref(),
        )
        .tools_offered(offers_tools);
        let mut prompt = builder.build_system_prompt(&self.task)?;
        if offers_tools && self.task.agent_config.tool_config.ends_task_with_text() {
            prompt.push_str(&format!(
                "## Tool names\nThe tools of this task are served by the MCP server `{ARKO_TOOL_SERVER}`, so you see each as `mcp__{ARKO_TOOL_SERVER}__<name>`.\n"
            ));
        } else if offers_tools {
            prompt.push_str(&format!(
                "## Tool names\nThe tools of this task are served by the MCP server `{ARKO_TOOL_SERVER}`, so you see each as `mcp__{ARKO_TOOL_SERVER}__<name>`: `complete_task` is `mcp__{ARKO_TOOL_SERVER}__complete_task`.\n"
            ));
        }
        Ok(prompt)
    }

    /// Adds a user message to the conversation, with the environment details
    /// and reminder of every user message. It is sent with the next input.
    fn add_user_message(&mut self, text: &str) -> Result<(), ArkoAgentsError> {
        let builder = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing, None);
        let number = self.task.user_message_count + 1;
        let message = builder.augment_user_message(&self.task, text, number, true)?;
        self.task.conversation.add_user_message(&message);
        self.task.mark_last_message_runtime();
        self.task.increment_user_message_count();
        Ok(())
    }

    /// Before a session resumes after the task stopped while tool calls ran
    /// (its host stopped after saving it, or Claude Code exited): Claude
    /// Code settles the calls its transcript holds as interrupted, without
    /// running them again, and its transcript can lack calls arko already
    /// ran. The task's record of them is what counts: the resumed session's
    /// input tells the model each call's outcome. A call whose result arko
    /// had is reported completed, with its result. An arko call whose tool
    /// is [`Rerun`](ToolRecovery::Rerun) or [`Resume`](ToolRecovery::Resume)
    /// runs again now (with
    /// [`TaskContext::recovering`](crate::tool::TaskContext::recovering)
    /// set), and its result is reported. Any other call's outcome is
    /// unknown.
    async fn recover_open_calls(&mut self) -> Result<(), ArkoAgentsError> {
        if self.task.claude_code_session.is_none() || self.task.current_state.is_terminal() {
            // No session to tell (a new one starts from the conversation).
            self.task.running_tool_calls.clear();
            return Ok(());
        }
        // Kept until they are settled: a call run again stays recorded
        // while it runs.
        let open = self.task.running_tool_calls.clone();
        if open.is_empty() {
            return Ok(());
        }
        log::info!(
            "Task {}: {} tool call(s) were open when the task stopped.",
            self.task.id,
            open.len()
        );
        let mut lines = Vec::new();
        let mut images = Vec::new();
        // The pauses the calls run again ask for: the task pauses once.
        let mut pauses = Vec::new();
        for call in open {
            let served = if call.built_in {
                call.tool_name.clone()
            } else {
                format!("mcp__{ARKO_TOOL_SERVER}__{}", call.tool_name)
            };
            let named = match &call.call_id {
                Some(id) => format!("`{served}` ({id})"),
                None => format!("`{served}`"),
            };
            if let Some(output) = &call.output {
                lines.push(format!(
                    "- {named} had completed. Its result was: {}",
                    bounded(output)
                ));
                if !call.output_images.is_empty() {
                    images.push(ai::Content::Text(format!(
                        "Images from recovered tool {named}:"
                    )));
                    images.extend(
                        call.output_images
                            .iter()
                            .map(|image| ai::Content::Image(image.image.clone())),
                    );
                }
                continue;
            }
            let tool = (!call.built_in)
                .then(|| self.registry.get(&call.tool_name))
                .flatten()
                .filter(|_| {
                    // Their outcomes change the task; the model calls them
                    // again.
                    ![
                        COMPLETE_TASK_TOOL_NAME,
                        ASK_USER_TOOL_NAME,
                        crate::tool::builtins::CLOSE_QUESTIONS_TOOL_NAME,
                    ]
                    .contains(&call.tool_name.as_str())
                });
            let mode = match (&tool, &call.custom_output) {
                (Some(_), Some(_)) => ToolRecovery::Rerun,
                (Some(tool), None) => tool.recovery_mode(),
                (None, _) => ToolRecovery::InDoubt,
            };
            let Some(tool) = tool.filter(|_| mode != ToolRecovery::InDoubt) else {
                let mut line = format!(
                    "- {named} was running, so its outcome is unknown: it may have run partly or fully. Check its effects before calling it again."
                );
                if call.built_in {
                    line.push_str(" Commands it started may still be running.");
                }
                lines.push(line);
                continue;
            };
            let arguments = Value::Object(call.parameters.clone().into_iter().collect());
            let (outcome, output) = self
                .tool_host()
                .run(
                    &call.tool_name,
                    tool,
                    &arguments,
                    Admitted {
                        id: call.tool_call_id,
                        call_id: call.call_id.clone(),
                        custom_output: call.custom_output.clone(),
                        reply: None,
                    },
                    true,
                )
                .await;
            match outcome {
                CallOutcome::Finished { raw, .. } => {
                    self.emit(AgentEvent::ToolResult {
                        tool_call_id: call.tool_call_id,
                        tool_name: call.tool_name.clone(),
                        output: raw,
                    })
                    .await?;
                }
                CallOutcome::Rejected(reason) | CallOutcome::Failed(reason) => {
                    self.emit(AgentEvent::ToolRejected {
                        tool_call_id: call.tool_call_id,
                        tool_name: call.tool_name.clone(),
                        reason: Some(reason),
                    })
                    .await?;
                }
                CallOutcome::Stop { reason, cancel } => {
                    self.set_state(if cancel {
                        TaskState::Cancelled { reason }
                    } else {
                        TaskState::Failed {
                            reason,
                            cause: None,
                            error_kind: None,
                        }
                    })
                    .await?;
                }
                CallOutcome::Pause { reason, .. } => pauses.push(reason),
            }
            lines.push(format!(
                "- {named} was running, so it was run again. Its result: {}",
                bounded(&output.text)
            ));
            if !output.images.is_empty() {
                images.push(ai::Content::Text(format!(
                    "Images from recovered tool {named}:"
                )));
                images.extend(output.images.into_iter().map(ai::Content::Image));
            }
        }
        self.task.running_tool_calls.clear();
        if !pauses.is_empty() && !self.task.current_state.is_terminal() {
            self.pause(pauses.join("\n"), ErrorRecoveryCause::Middleware)
                .await?;
        }
        if self.task.current_state.is_terminal() {
            self.report(false).await;
            return Ok(());
        }
        let synced = self
            .task
            .claude_code_synced
            .min(self.task.conversation.messages.len());
        let unsent = self.task.conversation.messages[synced..]
            .iter()
            .any(|message| message.role == ai::Role::User);
        let mut note = format!("{RESTARTED_NOTE}\n{}", lines.join("\n"));
        if !unsent {
            note.push_str(&format!("\n\n{CONTINUE}"));
        }
        self.add_user_message(&note)?;
        // Keep the pixels with the recovery note so another stop before
        // sending it does not lose them or rerun the tool a second time.
        self.task
            .conversation
            .messages
            .last_mut()
            .unwrap()
            .content
            .extend(images);
        self.report(false).await;
        Ok(())
    }

    async fn drive(&mut self) -> Result<(), ArkoAgentsError> {
        loop {
            match self.task.current_state.clone() {
                state if state.is_terminal() => return Ok(()),
                TaskState::AwaitingUserResponse { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. }
                | TaskState::Interrupted { .. } => {
                    if matches!(self.task.current_state, TaskState::Interrupted { .. }) {
                        // Running an interrupted task again continues it.
                        self.set_state(TaskState::InProgress).await?;
                        continue;
                    }
                    if !self.wait_for_decision().await? {
                        return Ok(());
                    }
                    continue;
                }
                _ => {}
            }
            if !self.task.running_tool_calls.is_empty() {
                // Claude Code (or the host) stopped while calls ran.
                self.recover_open_calls().await?;
                continue;
            }
            if let Some(reason) = self.held_request() {
                self.hold(reason).await?;
                continue;
            }
            if !self.take_waiting_commands().await? {
                return Ok(());
            }
            let session = match self.start_session().await {
                Ok(session) => session,
                Err(error) => {
                    self.pause_on_error(
                        format!("Claude Code could not start: {error}"),
                        ErrorRecoveryCause::NonRetryable,
                    )
                    .await?;
                    continue;
                }
            };
            let end = self.drive_session(session).await?;
            match end {
                SessionEnd::Stop => return Ok(()),
                SessionEnd::Wait | SessionEnd::Restart => continue,
            }
        }
    }

    /// Waits for the command that ends an answer or error-recovery wait.
    /// Returns false when the task stops instead.
    async fn wait_for_decision(&mut self) -> Result<bool, ArkoAgentsError> {
        loop {
            let deadline = self
                .task
                .agent_config
                .user_command_timeout_seconds
                .filter(|_| !self.host_decisions.get())
                .map(|seconds| tokio::time::Instant::now() + Duration::from_secs(seconds));
            let command = match deadline {
                Some(deadline) => {
                    match tokio::time::timeout_at(deadline, self.commands.recv()).await {
                        Ok(command) => command,
                        Err(_) => {
                            self.set_state(TaskState::Interrupted {
                                reason: "No decision arrived before the command timeout.".into(),
                            })
                            .await?;
                            return Ok(false);
                        }
                    }
                }
                None => self.commands.recv().await,
            };
            let Some(command) = command else {
                self.set_state(TaskState::Interrupted {
                    reason: "Command channel disconnected.".into(),
                })
                .await?;
                return Ok(false);
            };
            match (self.task.current_state.clone(), command) {
                (_, UserCommand::InjectContext { content }) => self.queue_context(content, true),
                (_, UserCommand::HostDeciding) => {}
                (_, UserCommand::RequestHandover) => self.note_handover_request().await?,
                (_, UserCommand::CancelTask) => {
                    self.set_state(TaskState::Cancelled {
                        reason: crate::task::CANCELLED_BY_USER.into(),
                    })
                    .await?;
                    return Ok(false);
                }
                (_, UserCommand::SuspendTask) => return Ok(false),
                (
                    TaskState::AwaitingUserResponse { .. },
                    UserCommand::SubmitUserResponse { response },
                ) => {
                    // (Those the host still has open stay open.)
                    crate::handover::settle_waited(self.handover_hook.as_deref(), &mut self.task);
                    self.task
                        .prepare_for_resumption_with_user_response(response)?;
                    self.emit(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                    return Ok(true);
                }
                (
                    TaskState::AwaitingErrorRecoveryApproval {
                        error_message,
                        cause,
                        ..
                    },
                    UserCommand::ApproveErrorRecovery { message },
                ) => {
                    let mut note = format!(
                        "The previous attempt failed: {error_message}\nThe user approved continuing."
                    );
                    if matches!(
                        cause,
                        ErrorRecoveryCause::Monitor | ErrorRecoveryCause::BudgetExceeded
                    ) {
                        note = CONTINUE.to_string();
                    }
                    if let Some(message) = message {
                        note.push_str(&format!("\n\n{message}"));
                    }
                    self.add_user_message(&note)?;
                    self.set_state(TaskState::InProgress).await?;
                    return Ok(true);
                }
                (
                    TaskState::AwaitingErrorRecoveryApproval { .. },
                    UserCommand::ContinueAfterError { note },
                ) => {
                    self.add_user_message(note.as_deref().unwrap_or(CONTINUE))?;
                    self.set_state(TaskState::InProgress).await?;
                    return Ok(true);
                }
                (
                    TaskState::AwaitingErrorRecoveryApproval {
                        error_message,
                        cause,
                        error_kind,
                    },
                    UserCommand::RejectErrorRecovery { message },
                ) => {
                    // The pause's cause and error kind stay with the failure.
                    self.set_state(TaskState::Failed {
                        reason: message.unwrap_or_else(|| {
                            format!("Error recovery rejected by user for error: {error_message}")
                        }),
                        cause: Some(cause),
                        error_kind,
                    })
                    .await?;
                    return Ok(false);
                }
                // Claude Code keeps the conversation it sends, so the input
                // it was refused can't be changed: reported as arko's own
                // runtime reports a change it can't apply, an error the
                // host answers, the pause standing.
                (
                    TaskState::AwaitingErrorRecoveryApproval { .. },
                    UserCommand::ResolveRefusal { .. },
                ) => {
                    self.emit(AgentEvent::ErrorEncountered {
                        error_message: CANT_RESOLVE_REFUSAL.to_string(),
                        requires_approval_to_retry: true,
                        tool_call_id: None,
                        error_kind: None,
                    })
                    .await?;
                }
                (_, other) => log::warn!(
                    "Task {}: command {:?} doesn't apply while the task is {:?}. Ignored.",
                    self.task.id,
                    other,
                    self.task.current_state
                ),
            }
        }
    }

    /// Handles the commands that arrived while no session ran, before the
    /// next one starts, so that a request for a handover or a compaction
    /// applies to its first input. Returns false when the task stops
    /// instead.
    async fn take_waiting_commands(&mut self) -> Result<bool, ArkoAgentsError> {
        while let Ok(command) = self.commands.try_recv() {
            match command {
                UserCommand::InjectContext { content } => self.queue_context(content, true),
                UserCommand::RequestHandover => self.note_handover_request().await?,
                UserCommand::HostDeciding => {}
                UserCommand::CancelTask => {
                    self.set_state(TaskState::Cancelled {
                        reason: crate::task::CANCELLED_BY_USER.into(),
                    })
                    .await?;
                    return Ok(false);
                }
                UserCommand::SuspendTask => return Ok(false),
                other => log::warn!(
                    "Task {}: command {:?} doesn't apply before Claude Code runs. Ignored.",
                    self.task.id,
                    other
                ),
            }
        }
        Ok(true)
    }

    /// Takes the context sent after the run's last read of its channel (a
    /// moved call's result in a note, say), which dropping the channel
    /// would lose though its sender counts it delivered. It waits with the
    /// task for its next run. The channel is closed first, so anything sent
    /// later fails to send, and the sender keeps it.
    async fn take_last_context(&mut self) {
        self.commands.close();
        // `None` once the channel is empty and no sender holds a permit.
        while let Some(command) = self.commands.recv().await {
            match command {
                UserCommand::InjectContext { content } => self.queue_context(content, true),
                other => log::debug!(
                    "Task {}: command {other:?} came after the run ended. Ignored.",
                    self.task.id
                ),
            }
        }
    }

    /// Keeps `content` for the agent until it is handed over. Context the
    /// host sent (`from_host`) fires the arko calls' signal only when its
    /// turn context source says so.
    fn queue_context(&mut self, content: String, from_host: bool) {
        if !content.trim().is_empty() {
            let wakes = !from_host
                || self
                    .turn_context
                    .as_ref()
                    .is_none_or(|source| source.wakes_calls(&content));
            self.task.pending_context_injections.push(content);
            // An arko call that waits can end, so the agent reads it with
            // that call's result.
            if wakes {
                self.context_arrivals.notify();
            }
        }
    }

    async fn note_handover_request(&mut self) -> Result<(), ArkoAgentsError> {
        if self.task.handover_requested.is_some() || self.compaction.is_some() {
            return Ok(());
        }
        if self.compacts() {
            log::info!("Task {}: Compaction requested by the host.", self.task.id);
            self.compaction = Some(Compaction::Due);
            return Ok(());
        }
        let input_tokens = self.latest_size();
        self.task.handover_requested = Some(HandoverRequest {
            report_requested: true,
            ..HandoverRequest::new(input_tokens)
        });
        self.emit(AgentEvent::HandoverRequested { input_tokens })
            .await
    }

    fn handover_due(&self) -> bool {
        self.task.handover_requested.is_some()
    }

    /// Whether the agent compacts its conversation in Claude Code instead of
    /// handing it over.
    fn compacts(&self) -> bool {
        self.task
            .agent_config
            .handover
            .as_ref()
            .is_some_and(|handover| handover.compact_in_claude_code)
    }

    /// The input of the conversation's latest request: this run's, else as
    /// the task recorded it.
    fn latest_size(&self) -> usize {
        self.context_size
            .or_else(|| self.task.current_conversation_context_size())
            .unwrap_or(0)
    }

    fn report_request(&self) -> String {
        let handover = self.task.agent_config.handover.as_ref();
        if self.is_chat() {
            return handover
                .map_or(crate::handover::DEFAULT_CHAT_REPORT_REQUEST, |config| {
                    config.chat_report_request()
                })
                .to_string();
        }
        handover
            .map_or(DEFAULT_TEXT_REPORT_REQUEST, |config| {
                config.report_request(false)
            })
            .to_string()
    }

    /// The agent is a chat (`AgentToolConfig::chat`).
    fn is_chat(&self) -> bool {
        self.task.agent_config.tool_config.chat
    }

    /// The agent is offered nothing: neither arko's tools nor Claude Code's
    /// built-ins.
    fn offers_no_tools(&self) -> bool {
        self.task
            .agent_config
            .tool_config
            .offered_tool_names()
            .is_empty()
    }

    /// The context waiting for the model (taken from the channel, and the
    /// turn context source's), as one context update, with its images.
    /// `None` while a handover report is due, when it waits for the fresh
    /// conversation, and during a compaction, when it waits for the agent's
    /// next input. Where images can't go (`with_images` false: a hook's
    /// context is text), the text goes now, saying that the update's images
    /// come with the next input, and only the images wait. There the part
    /// of the update the user typed ([`TurnContext::user_content`]) comes
    /// apart from the rest, as the second text, to go as their message
    /// (the update's text is empty when only that part is waiting). With
    /// images, it is part of the update.
    fn take_context(&mut self, with_images: bool) -> Option<(String, String, Vec<ContextImage>)> {
        if self.handover_due() || self.compaction.is_some() {
            // It waits for a later input, and arko calls that start meanwhile
            // count newer context only.
            self.context_arrivals.caught_up();
            return None;
        }
        let mut update_ids = Vec::new();
        let mut arrived = 0;
        let mut typed = String::new();
        // A hook's text is limited. The injections that don't fit, and the
        // source's update when little room is left, wait for the next one.
        let mut deferred = Vec::new();
        let mut room = HOOK_MAX_CHARS;
        if !with_images {
            let (mut used, mut fit) = (0, 0);
            for content in &self.task.pending_context_injections {
                let chars = content.chars().count() + 2;
                if fit > 0 && used + chars > HOOK_MAX_CHARS {
                    break;
                }
                used += chars;
                fit += 1;
            }
            deferred = self.task.pending_context_injections.split_off(fit);
            room = HOOK_MAX_CHARS.saturating_sub(used);
        }
        let update = self.turn_context.as_ref().and_then(|source| {
            if with_images {
                source.context_update(self.task.id)
            } else if deferred.is_empty() && room >= HOOK_MAX_CHARS / 2 {
                source.context_update_within(self.task.id, room)
            } else {
                None
            }
        });
        if let Some(update) = update {
            if !update.content.trim().is_empty()
                || !update.user_content.trim().is_empty()
                || !update.images.is_empty()
            {
                update_ids = update.ids;
            }
            raise(&mut self.taken_marks, &update.marks);
            arrived = update.images.len();
            self.queue_context(update.content, false);
            if with_images {
                self.queue_context(update.user_content, false);
            } else if !update.user_content.trim().is_empty() {
                typed = update.user_content;
            }
            self.task.pending_context_images.extend(update.images);
        }
        if self.task.pending_context_injections.is_empty()
            && self.task.pending_context_images.is_empty()
            && typed.is_empty()
        {
            self.keep_taken_marks();
            return None;
        }
        let hold_images = !with_images && !self.task.pending_context_images.is_empty();
        if hold_images && self.task.pending_context_injections.is_empty() && typed.is_empty() {
            // Only images wait, said already: nothing goes until they can.
            self.held_context_ids.append(&mut update_ids);
            // (They wait in the task.)
            self.keep_taken_marks();
            return None;
        }
        let mut content = self.task.pending_context_injections.join("\n\n");
        let mut ids = self
            .turn_context
            .as_ref()
            .map(|source| {
                self.task
                    .pending_context_injections
                    .iter()
                    .filter_map(|content| source.identify(content))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        ids.append(&mut self.held_context_ids);
        ids.append(&mut update_ids);
        self.task.pending_context_injections = deferred;
        // Claude Code has it, so arko calls that start from now on count
        // newer context.
        self.context_arrivals.caught_up();
        let images = if hold_images {
            if arrived > 0 {
                // With the user's notes, which the images belong to.
                let text = if typed.is_empty() {
                    &mut content
                } else {
                    &mut typed
                };
                text.push_str(&match arrived {
                    1 => "\n\n(The image of this update can't come during a turn. It comes with your next input.)".to_string(),
                    count => format!("\n\n(The {count} images of this update can't come during a turn. They come with your next input.)"),
                });
            }
            Vec::new()
        } else {
            if content.is_empty() && typed.is_empty() {
                content = "The images an earlier update said would follow.".to_string();
            }
            std::mem::take(&mut self.task.pending_context_images)
        };
        self.context_receipts.push(AgentEvent::ContextInjected {
            ids,
            bytes: content.len() + typed.len(),
        });
        let content = if content.is_empty() {
            content
        } else {
            format!("{CONTEXT_UPDATE}\n\n{content}")
        };
        Some((content, typed, images))
    }

    /// The task holds the context taken since the last call, so it keeps its
    /// marks.
    fn keep_taken_marks(&mut self) {
        let marks = std::mem::take(&mut self.taken_marks);
        self.task.raise_context_marks(&marks);
    }

    /// Reports the context updates taken (and now sent) since the last call.
    async fn emit_context_receipts(&mut self) -> Result<(), ArkoAgentsError> {
        for event in std::mem::take(&mut self.context_receipts) {
            self.emit(event).await?;
        }
        Ok(())
    }

    fn session_options(
        &mut self,
        requests: &mpsc::UnboundedSender<HostRequest>,
    ) -> cc::ClaudeOptions {
        let config = self.task.agent_config.clone();
        let settings = config.claude_code.clone().unwrap_or_default();
        let generation = &config.llm_config.generation_config;
        let effort = match &generation.reasoning {
            Some(ai::ReasoningConfig::Anthropic { effort, .. })
            | Some(ai::ReasoningConfig::AnthropicAdaptive { effort }) => {
                effort.map(|effort| match effort {
                    ai::AnthropicReasoningEffort::Low => cc::Effort::Low,
                    ai::AnthropicReasoningEffort::Medium => cc::Effort::Medium,
                    ai::AnthropicReasoningEffort::High => cc::Effort::High,
                    ai::AnthropicReasoningEffort::XHigh => cc::Effort::XHigh,
                    ai::AnthropicReasoningEffort::Max => cc::Effort::Max,
                })
            }
            _ => None,
        };
        // `complete_task` is arko's, also where it isn't offered.
        let built_ins: Vec<String> = config
            .tool_config
            .available_tool_names
            .iter()
            .filter(|name| self.registry.get(name).is_none() && *name != COMPLETE_TASK_TOOL_NAME)
            .cloned()
            .collect();
        let mut cc_settings = serde_json::Map::new();
        if config.handover.is_some() {
            // The handover keeps the context small; Claude Code's own
            // compaction would replace the conversation first.
            cc_settings.insert("autoCompactEnabled".into(), json!(false));
        }
        if generation.anthropic_fast_mode == Some(true) {
            cc_settings.insert("fastMode".into(), json!(true));
        }
        if let Some(ttl) = generation.anthropic_cache_ttl {
            // The main conversation's cache TTL; unset, Claude Code uses 1
            // hour on a subscription within its limits, 5 minutes otherwise.
            let ttl = match ttl {
                ai::AnthropicCacheTtl::FiveMinutes => "5m",
                ai::AnthropicCacheTtl::OneHour => "1h",
            };
            cc_settings.insert("promptCacheTtl".into(), json!(ttl));
        }
        let system = self
            .task
            .conversation
            .system_message
            .clone()
            .unwrap_or_default();
        let host = self.tool_host();
        let mut server = cc::ToolServer::new(ARKO_TOOL_SERVER);
        let mut names = self.registry.list_tool_names();
        names.sort();
        let mut served = Vec::new();
        for name in names {
            let Some(tool) = self.registry.get(&name) else {
                continue;
            };
            let definition = config
                .offered_tool_definition(tool.definition())
                .to_arko_ai_tool_definition();
            served.push(definition.clone());
            server = server.tool(Arc::new(ServedTool {
                name: definition.name,
                description: definition.description,
                schema: definition.parameters,
                host: host.clone(),
            }));
        }
        self.served_tools = Some(ai::ToolConfig::new(served));
        // A cut back resumes at the entry the task recorded, and a branch
        // first copies the session it was branched from into its own. Both hold
        // until the session has the task's first new message (see
        // `record_sent`): Claude Code saves a cut, and creates a fork, only
        // with that message.
        let recorded_at = self.task.claude_code_resume_at.clone();
        let session = match (
            &self.task.claude_code_session,
            &self.task.claude_code_fork_of,
        ) {
            (Some(_), Some(parent)) => {
                // Each attempt forks under an ID of its own: one an attempt
                // that failed may have created can't be used again.
                let id = Uuid::new_v4().to_string();
                self.task.claude_code_session = Some(id.clone());
                cc::SessionStart::Fork {
                    session: parent.clone(),
                    at: recorded_at,
                    new_session: id,
                }
            }
            (Some(session), None) => cc::SessionStart::Resume {
                session: session.clone(),
                fork: false,
                at: self.resume_at.take().or(recorded_at),
            },
            (None, _) => {
                let id = Uuid::new_v4().to_string();
                self.task.claude_code_session = Some(id.clone());
                cc::SessionStart::WithId(id)
            }
        };
        let hook = |event: cc::HookEvent| {
            let mut hook = cc::Hook::new(
                event.clone(),
                None,
                Arc::new(HookForwarder {
                    event,
                    requests: requests.clone(),
                }),
            );
            hook.timeout_seconds = Some(HOOK_TIMEOUT_SECONDS);
            hook
        };
        cc::ClaudeOptions {
            cli_path: settings.cli_path.clone(),
            cli_prefix_args: settings.cli_args.clone(),
            // Claude Code finds a session only from the directory it started
            // in, so the CLI runs there after the task moved.
            working_directory: Some(
                self.task
                    .claude_code_directory
                    .clone()
                    .unwrap_or_else(|| self.task.current_working_directory.clone()),
            ),
            model: Some(config.llm_config.model.id.clone()),
            effort,
            system_prompt: if settings.replace_system_prompt {
                cc::SystemPrompt::Replace(system)
            } else {
                cc::SystemPrompt::Append(system)
            },
            // The session recorded an earlier prompt: this one replaces it.
            system_prompt_snapshot: self.task.claude_code_prompt_changed.then_some(false),
            built_in_tools: if built_ins.is_empty() {
                cc::BuiltInTools::None
            } else {
                cc::BuiltInTools::Only(built_ins)
            },
            permission_mode: Some(cc::PermissionMode::Default),
            session,
            settings: (!cc_settings.is_empty()).then_some(Value::Object(cc_settings)),
            setting_sources: Some(settings.setting_sources.clone()),
            env: settings.env.clone().into_iter().collect(),
            extra_args: if settings.setting_sources.is_empty() {
                // Only the agent's MCP servers, not the user's.
                vec!["--strict-mcp-config".into()]
            } else {
                Vec::new()
            },
            permission_handler: Some(Arc::new(PermissionForwarder {
                requests: requests.clone(),
            })),
            hooks: vec![
                hook(cc::HookEvent::PreToolUse),
                hook(cc::HookEvent::PostToolUse),
            ],
            tool_servers: vec![Arc::new(server)],
            // The task's own directory, where the CLI doesn't run after a
            // move, is open to Claude Code's tools too.
            add_directories: self
                .task
                .claude_code_directory
                .is_some()
                .then(|| self.task.current_working_directory.clone())
                .into_iter()
                .collect(),
            // Each request's final usage comes in its `message_delta`.
            include_partial_messages: true,
            ..cc::ClaudeOptions::default()
        }
    }

    fn tool_host(&mut self) -> Arc<ToolHost> {
        if let Some(host) = &self.host {
            return host.clone();
        }
        let host = Arc::new(ToolHost {
            environment: Arc::new(RunEnvironment {
                task_id: self.task.id,
                agent_name: self.task.agent_config.name.clone(),
                initial_task_prompt: self.task.initial_task_prompt.clone(),
                working_directory: self.task.current_working_directory.clone(),
                middleware: self.middleware.clone(),
                stop: self.host_stop.or(&self.calls_stop.signal()),
                model: self.task.agent_config.llm_config.model.clone(),
                image_sink: self.image_sink.clone(),
                // Claude Code holds the conversation and keeps its own count.
                image_room: None,
                context: self.context_arrivals.clone(),
                #[cfg(any(unix, windows))]
                jobs: self.background_jobs.as_ref().map(Arc::downgrade),
                terminal_tools: RunEnvironment::offers_terminal_tools(&self.registry),
            }),
            config: self.task.agent_config.clone(),
            registry: self.registry.clone(),
            interceptor: self.interceptor.take(),
            admitted: Mutex::default(),
            outcomes: Mutex::default(),
        });
        self.host = Some(host.clone());
        host
    }

    async fn start_session(
        &mut self,
    ) -> Result<(cc::ClaudeSession, mpsc::UnboundedReceiver<HostRequest>), cc::ClaudeError> {
        let (requests, receiver) = mpsc::unbounded_channel();
        let options = self.session_options(&requests);
        let session = cc::ClaudeSession::start(options).await?;
        if let Some(tools) = session.initialization().get("tools") {
            log::debug!("Task {}: Claude Code tools: {tools}", self.task.id);
        }
        Ok((session, receiver))
    }

    /// The task's input the session hasn't received: user messages added
    /// since it last ran (the task prompt, a follow-up, an answer, a note),
    /// and context waiting for the model. A resumed session with nothing new
    /// is asked to continue.
    fn unsent_input(&mut self) -> Result<Value, ArkoAgentsError> {
        let synced = self
            .task
            .claude_code_synced
            .min(self.task.conversation.messages.len());
        let mut parts: Vec<String> = self.task.conversation.messages[synced..]
            .iter()
            .filter(|message| message.role == ai::Role::User)
            .map(|message| {
                message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|text| !text.trim().is_empty())
            .collect();
        if parts.is_empty() {
            self.add_user_message(CONTINUE)?;
            parts.push(
                self.task
                    .conversation
                    .messages
                    .last()
                    .map(|message| {
                        message
                            .content
                            .iter()
                            .filter_map(ai::Content::as_text)
                            .collect::<String>()
                    })
                    .unwrap_or_default(),
            );
        }
        if let Some((context, _, images)) = self.take_context(true) {
            self.task.conversation.add_user_message_with_contents(
                crate::task::with_context_images(context.clone(), images),
            );
            self.task.increment_user_message_count();
            self.keep_taken_marks();
            parts.push(context);
        }
        let messages = &self.task.conversation.messages[synced..];
        let contents = || {
            messages
                .iter()
                .filter(|message| message.role == ai::Role::User)
                .flat_map(|message| &message.content)
        };
        let input = if contents().any(|content| content.as_image().is_some()) {
            input_blocks(contents())
        } else {
            Value::String(parts.join("\n\n"))
        };
        self.task.claude_code_synced = self.task.conversation.messages.len();
        Ok(input)
    }

    /// Sends a user message to the running session and records it.
    async fn send(
        &mut self,
        session: &cc::ClaudeSession,
        text: &str,
    ) -> Result<(), cc::ClaudeError> {
        self.send_with_images(session, text, Vec::new()).await
    }

    /// [`send`](Self::send), with `images` after the text, each after its
    /// label.
    async fn send_with_images(
        &mut self,
        session: &cc::ClaudeSession,
        text: &str,
        images: Vec<ContextImage>,
    ) -> Result<(), cc::ClaudeError> {
        self.add_user_message(text).ok();
        self.task.claude_code_synced = self.task.conversation.messages.len();
        let Some(message) = self.task.conversation.messages.last_mut() else {
            return Ok(());
        };
        let sent = if images.is_empty() {
            Value::String(
                message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .collect::<String>(),
            )
        } else {
            message
                .content
                .extend(crate::task::context_image_content(images));
            input_blocks(message.content.iter())
        };
        let uuid = session.send_content(sent).await?;
        self.record_sent(uuid);
        Ok(())
    }

    /// The session received a user message, the last of the task's
    /// conversation so far.
    fn record_sent(&mut self, uuid: String) {
        self.record_entry(Some(&uuid));
        self.last_entry = Some(Entry {
            uuid,
            messages: self.task.conversation.messages.len(),
        });
    }

    async fn drive_session(
        &mut self,
        (mut session, mut requests): (cc::ClaudeSession, mpsc::UnboundedReceiver<HostRequest>),
    ) -> Result<SessionEnd, ArkoAgentsError> {
        self.retrying_since = None;
        self.streamed_bytes = (0, 0);
        // Message IDs are a launch's own: what an earlier launch streamed
        // says nothing about this one's messages.
        self.streaming = None;
        self.response = None;
        self.streamed.clear();
        self.streamed_text.clear();
        // A compaction due as the session starts (asked for while the task
        // waited, or the conversation's last turn ended over the threshold)
        // comes first, and the task's input follows Claude Code's summary.
        // A session that has received nothing has nothing to compact.
        if self.compaction.is_none()
            && self.compacts()
            && !self.handover_due()
            && let Some(size) = self.over_threshold(self.latest_size())
        {
            log::info!(
                "Task {}: Compaction due at {size} input tokens.",
                self.task.id
            );
            self.compaction = Some(Compaction::Due);
        }
        if self.task.claude_code_synced == 0 && self.compaction == Some(Compaction::Due) {
            self.compaction = None;
        }
        // A chat over the handover size (or asked to hand over) writes its
        // summary first. Its turns end with a reply and no tool call, so the
        // report request can't reach it during a turn: it goes alone at the
        // start of the next one, and the host's messages since the last reply
        // wait for the fresh conversation, which answers them
        // (`HandoverSeed::for_chat`).
        let chat_reports = self.is_chat()
            && self.compaction.is_none()
            && !self.compacts()
            && self.task.claude_code_synced > 0
            && !self.report_delivered;
        if chat_reports
            && !self.handover_due()
            && let Some(size) = self.over_threshold(self.latest_size())
        {
            log::info!(
                "Task {}: Handover due at {size} input tokens.",
                self.task.id
            );
            self.note_handover_request().await?;
        }
        // After a cut back or a branch, the session has the task's input only
        // once Claude Code saved it: until then the task keeps the cut's
        // point as what the session has, so a task saved in between (and
        // run again after a crash) sends it again from the same entry.
        let synced_before = self.task.claude_code_synced;
        let resuming_a_cut =
            self.task.claude_code_resume_at.is_some() || self.task.claude_code_fork_of.is_some();
        let input = if self.compaction == Some(Compaction::Due) {
            self.compaction = Some(Compaction::Sent { compacted: false });
            Value::String(COMPACT_COMMAND.to_string())
        } else if chat_reports && self.handover_due() {
            self.report_delivered = true;
            let request = self.report_request();
            self.add_user_message(&request)?;
            self.task.claude_code_synced = self.task.conversation.messages.len();
            Value::String(
                self.task
                    .conversation
                    .messages
                    .last()
                    .map(|message| {
                        message
                            .content
                            .iter()
                            .filter_map(ai::Content::as_text)
                            .collect::<String>()
                    })
                    .unwrap_or_default(),
            )
        } else {
            self.unsent_input()?
        };
        if resuming_a_cut {
            self.synced_on_send = Some(self.task.claude_code_synced);
            self.task.claude_code_synced = synced_before;
        }
        self.api_error = None;
        self.error_kind = None;
        // A session that ended before its turn did: reported now.
        self.report_tool_errors().await?;
        // The session (and what it was sent) is saved before it runs.
        self.report(true).await;
        let sent = match input {
            Value::String(text) => session.send(text).await,
            content => session.send_content(content).await,
        };
        match sent {
            Ok(uuid) => self.record_sent(uuid),
            Err(error) => {
                let end = self.session_failed(error.to_string()).await?;
                let _ = session.close().await;
                return Ok(end);
            }
        }
        // The answer comes while the model works, before the first event.
        self.ask_usage_if_due(&session).await;
        self.emit_context_receipts().await?;
        self.questions = None;
        // Claude Code exited on its own: it may not have saved every call.
        let mut cli_exited = false;
        let end = loop {
            let approval_deadline = self
                .approval
                .as_ref()
                .and_then(|approval| approval.deadline);
            tokio::select! {
                biased;
                command = self.commands.recv() => {
                    self.handle_command(command, &session).await?;
                    if self.stopping.is_some() || self.task.current_state.is_terminal() {
                        // Calls held by the SlowDown don't run.
                        for (_, reply, _) in self.slowed.drain(..) {
                            let _ = reply.send(cc::Hook::deny_tool(
                                "arko did not run this call because the agent is stopping.",
                            ));
                        }
                    }
                }
                Some(request) = requests.recv() => {
                    self.handle_request(request, &session).await?;
                }
                _ = sleep_until(approval_deadline), if approval_deadline.is_some() => {
                    self.approval_timed_out().await?;
                }
                _ = sleep_until(self.slowed.front().map(|(deadline, ..)| *deadline)), if !self.slowed.is_empty() => {
                    let now = tokio::time::Instant::now();
                    while self.slowed.front().is_some_and(|(deadline, ..)| *deadline <= now) {
                        let (_, reply, answer) = self.slowed.pop_front().expect("checked");
                        let _ = reply.send(answer);
                    }
                }
                answer = Self::usage_answer(&mut self.usage_reply), if self.usage_reply.is_some() => {
                    self.usage_reply = None;
                    self.extra_usage_read(answer).await?;
                }
                message = session.next_message() => match message {
                    None => {
                        cli_exited = true;
                        if self.task.current_state.is_terminal() || self.stopping.is_some() {
                            break SessionEnd::Stop;
                        }
                        break self.session_failed("Claude Code exited during the turn.".into()).await?;
                    }
                    Some(cc::Message::Result(result)) => {
                        if let Some(end) = self.turn_ended(&result, &session).await? {
                            break end;
                        }
                    }
                    Some(message) => self.handle_message(message, &session).await?,
                },
            }
        };
        // Late hook requests get their stopping answer.
        drop(requests);
        self.slowed.clear();
        // A `/compact` whose turn didn't end: whether it compacted is
        // unknown, and the threshold decides again.
        if matches!(self.compaction, Some(Compaction::Sent { .. })) {
            self.compaction = None;
        }
        self.flush_usage().await?;
        if let Some(approval) = self.approval.take() {
            let _ = approval
                .reply
                .send(cc::PermissionDecision::deny("The agent is stopping."));
        }
        // A note written just before the turn stopped is cancelled after the
        // turn's result.
        while !self.sent_notes.is_empty()
            && let Ok(Some(message)) =
                tokio::time::timeout(NOTE_VERDICT_WAIT, session.next_message()).await
        {
            if let cc::Message::Other(raw) = message {
                self.note_settled(&raw);
            }
        }
        self.sent_notes.clear();
        match session.close().await {
            // Claude Code saved the session as it closed: its transcript
            // holds every call's outcome.
            Ok(_) if !cli_exited => self.task.running_tool_calls.clear(),
            Ok(_) => {}
            Err(error) => log::warn!("Task {}: closing Claude Code: {error}", self.task.id),
        }
        self.report(false).await;
        Ok(end)
    }

    async fn handle_command(
        &mut self,
        command: Option<UserCommand>,
        session: &cc::ClaudeSession,
    ) -> Result<(), ArkoAgentsError> {
        let stop = |this: &mut Self, stopping| {
            if this.stopping.is_none() {
                this.stopping = Some(stopping);
            }
            match stopping {
                Stopping::Cancel => this.calls_stop.stop("Task cancelled."),
                Stopping::Suspend => this.calls_stop.stop("Task suspended."),
                Stopping::Closed => false,
            };
        };
        match command {
            Some(UserCommand::InjectContext { content }) => self.queue_context(content, true),
            Some(UserCommand::RequestHandover) => self.note_handover_request().await?,
            Some(UserCommand::HostDeciding) => {
                if let Some(approval) = &mut self.approval {
                    approval.deadline = None;
                }
            }
            Some(UserCommand::ApproveToolUse {
                tool_call_id,
                custom_tool_output,
                message_to_agent,
            }) => {
                self.decide_approval(
                    tool_call_id,
                    true,
                    custom_tool_output,
                    message_to_agent,
                    None,
                )
                .await?
            }
            Some(UserCommand::RejectToolUse {
                tool_call_id,
                reason,
            }) => {
                self.decide_approval(tool_call_id, false, None, None, reason)
                    .await?
            }
            Some(UserCommand::SubmitBatchToolApprovals { decisions, .. }) => {
                if let Some(id) = self.approval.as_ref().map(|approval| approval.id)
                    && let Some(decision) = decisions.get(&id).cloned()
                {
                    match decision {
                        super::command::ToolApprovalDecision::Approve {
                            custom_tool_output,
                            message_to_agent,
                        } => {
                            self.decide_approval(
                                id,
                                true,
                                custom_tool_output,
                                message_to_agent,
                                None,
                            )
                            .await?
                        }
                        super::command::ToolApprovalDecision::Reject { reason } => {
                            self.decide_approval(id, false, None, None, reason).await?
                        }
                    }
                }
            }
            Some(UserCommand::CancelTask) => {
                stop(self, Stopping::Cancel);
                self.set_state(TaskState::Cancelled {
                    reason: crate::task::CANCELLED_BY_USER.into(),
                })
                .await?;
                let _ = session.interrupt().await;
            }
            Some(UserCommand::SuspendTask) => {
                stop(self, Stopping::Suspend);
                let _ = session.interrupt().await;
            }
            None => {
                stop(self, Stopping::Closed);
                self.set_state(TaskState::Interrupted {
                    reason: "Command channel disconnected.".into(),
                })
                .await?;
                let _ = session.interrupt().await;
            }
            Some(other) => log::warn!(
                "Task {}: command {:?} doesn't apply while Claude Code runs. Ignored.",
                self.task.id,
                other
            ),
        }
        Ok(())
    }

    async fn handle_request(
        &mut self,
        request: HostRequest,
        session: &cc::ClaudeSession,
    ) -> Result<(), ArkoAgentsError> {
        match request {
            HostRequest::PreToolUse {
                tool_use_id,
                tool_name,
                input,
                reply,
            } => {
                let judged = self
                    .input_from_session_directory(&tool_name, &input)
                    .unwrap_or_else(|| input.clone());
                let mut answer = self
                    .gate(tool_use_id.clone(), &tool_name, judged, session)
                    .await?;
                let allowed = answer["hookSpecificOutput"]["permissionDecision"] == "allow";
                if allowed {
                    if let Some(moved) = self.input_in_task_directory(&tool_name, &input) {
                        answer["hookSpecificOutput"]["updatedInput"] = moved;
                    }
                    self.record_running(tool_use_id.as_ref(), None).await;
                }
                match self.slow_down.filter(|_| allowed) {
                    Some(delay) => {
                        self.slowed
                            .push_back((tokio::time::Instant::now() + delay, reply, answer))
                    }
                    None => {
                        let _ = reply.send(answer);
                    }
                }
            }
            HostRequest::Permission { request, reply } => {
                self.permission(request, reply).await?;
            }
            HostRequest::PostToolUse {
                tool_use_id,
                tool_name,
                response,
                reply,
            } => {
                self.record_output(tool_use_id.as_ref(), &response);
                self.report(false).await;
                let answer = self
                    .after_tool(tool_use_id, &tool_name, response, session)
                    .await?;
                let _ = reply.send(answer);
            }
        }
        Ok(())
    }

    /// Every tool call's first check: the gate allows it, rejects it (the
    /// reason is its result), or sends it to a permission request that
    /// waits for approval.
    async fn gate(
        &mut self,
        tool_use_id: Option<String>,
        served: &str,
        input: Value,
        session: &cc::ClaudeSession,
    ) -> Result<Value, ArkoAgentsError> {
        let (name, built_in) = match arko_name(served) {
            Some(name) => (name.to_string(), false),
            None => (served.to_string(), true),
        };
        let id = Uuid::new_v4();
        let params = params_of(&input);
        let record = ai::ToolCall {
            id: tool_use_id.clone(),
            name: name.clone(),
            arguments: input.clone(),
            thought_signature: None,
            provider_id: None,
            invalid_arguments: None,
        };
        if let Some(tool_use_id) = &tool_use_id {
            self.calls.insert(
                tool_use_id.clone(),
                Call {
                    id,
                    name: name.clone(),
                    built_in,
                    record,
                    input: input.clone(),
                },
            );
        }
        self.emit(AgentEvent::ToolCallFormatted {
            tool_call_id: id,
            tool_name: name.clone(),
            parameters: params.clone(),
        })
        .await?;
        if matches!(self.task.current_state, TaskState::Completed { .. }) || self.stopping.is_some()
        {
            return self.reject_call(id, &name, TASK_COMPLETE.to_string()).await;
        }
        if self.handover_due() && name != COMPLETE_TASK_TOOL_NAME {
            self.report_delivered = true;
            let request = self.report_request();
            self.emit(AgentEvent::ToolRejected {
                tool_call_id: id,
                tool_name: name.clone(),
                reason: Some(request.clone()),
            })
            .await?;
            return Ok(held_for_report(&name, &request));
        }
        if self.compaction.is_some() && name != COMPLETE_TASK_TOOL_NAME {
            // The first call held ends the turn, so that the compaction
            // doesn't wait for the model.
            if self.compaction == Some(Compaction::Due) && self.held_calls.is_empty() {
                let _ = session.interrupt().await;
            }
            self.held_calls
                .push(format!("- `{served}` with {}", bounded(&input.to_string())));
            self.emit(AgentEvent::ToolRejected {
                tool_call_id: id,
                tool_name: name.clone(),
                reason: Some(HELD_FOR_COMPACTION.to_string()),
            })
            .await?;
            return Ok(held_for_compaction(&name));
        }
        let offered = if built_in {
            self.task
                .agent_config
                .tool_config
                .available_tool_names
                .contains(&name)
        } else {
            self.registry.get(&name).is_some()
        };
        if !offered {
            return self
                .reject_call(id, &name, format!("Tool '{name}' is not available."))
                .await;
        }
        // Claude Code's shell tools pass arko's command guard.
        #[cfg(any(unix, windows))]
        if built_in
            && let Some(dialect) = match name.as_str() {
                "Bash" => Some(crate::terminal::ShellDialect::Posix),
                "PowerShell" => Some(crate::terminal::ShellDialect::PowerShell),
                _ => None,
            }
            && let Some(command) = params.get("command").and_then(Value::as_str)
        {
            let guard = self
                .task
                .agent_config
                .claude_code
                .clone()
                .unwrap_or_default()
                .command_guard();
            if let Err(refusal) =
                guard.check(command, dialect, &self.task.current_working_directory)
            {
                return self.reject_call(id, &name, refusal.message).await;
            }
        }
        // The workspace path policy.
        let path_params: Vec<String> = if built_in {
            built_in_path_parameters(&name)
                .iter()
                .map(|param| param.to_string())
                .collect()
        } else {
            self.registry
                .get(&name)
                .map(|tool| {
                    tool.definition()
                        .parameters
                        .into_iter()
                        .filter(|param| param.is_path)
                        .map(|param| param.name)
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut outside = None;
        for param in path_params {
            let Some(path) = params.get(&param).and_then(Value::as_str) else {
                continue;
            };
            if path.is_empty() {
                continue;
            }
            match crate::utils::is_path_in_workspace(path, &self.task.current_working_directory) {
                Ok(true) => {}
                Ok(false) => match self.task.agent_config.get_path_safety_policy(&name) {
                    PathSafetyPolicy::AutoReject => {
                        let reason =
                            format!("Path for parameter '{param}' is outside the workspace.");
                        return self.reject_call(id, &name, reason).await;
                    }
                    PathSafetyPolicy::RequireApproval => {
                        outside = Some(param);
                        break;
                    }
                    PathSafetyPolicy::AutoApprove => {}
                },
                Err(error) => {
                    let reason = format!("Path validation failed for parameter '{param}': {error}");
                    return self.reject_call(id, &name, reason).await;
                }
            }
        }
        let needs_approval = !self.task.agent_config.should_auto_approve_tool(&name);
        if needs_approval || outside.is_some() {
            // The permission request that follows waits for approval.
            return Ok(ask_tool());
        }
        let call = Admitted {
            id,
            call_id: tool_use_id,
            custom_output: None,
            reply: None,
        };
        self.admit(&name, built_in, call, &input, session).await
    }

    /// The input an allowed call of Claude Code's own tools runs with, when
    /// the CLI doesn't run in the task's working directory (the task moved
    /// and the session stayed where it started) and the call depends on it.
    fn input_in_task_directory(&self, served: &str, input: &Value) -> Option<Value> {
        if self.task.claude_code_directory.is_none() || arko_name(served).is_some() {
            return None;
        }
        let mapped = self.input_from_session_directory(served, input);
        in_directory(
            served,
            mapped.as_ref().unwrap_or(input),
            &self.task.current_working_directory,
        )
        .or(mapped)
    }

    /// The input of a call of Claude Code's own tools with a file path that
    /// the CLI made absolute from the session's directory made the task's
    /// (see [`from_session_directory`]). The path policy judges and the host
    /// shows the call with it, since that is the file the call runs on.
    fn input_from_session_directory(&self, served: &str, input: &Value) -> Option<Value> {
        let from = self.task.claude_code_directory.as_deref()?;
        from_session_directory(served, input, from, &self.task.current_working_directory)
    }

    /// An `ask_user` or `close_questions` call being admitted: its
    /// questions are numbered now, or the ids to close are checked, so that
    /// what Claude Code reads of it names them (`call.reply`). The task's
    /// log records them, and the host hears of them, once the call ran
    /// (a middleware may still refuse it). A call whose input can't be read
    /// runs as it is, and its tool says why.
    fn take_question_call(&mut self, name: &str, input: &Value, call: &mut Admitted) {
        let taken = if name == ASK_USER_TOOL_NAME {
            let Some(Ok(questions)) = input
                .get(crate::tool::builtins::ask_user::ASK_USER_QUESTIONS_PARAM_NAME)
                .map(crate::questions::parse_questions)
            else {
                return;
            };
            let last = crate::handover::last_question_number(
                self.handover_hook.as_deref(),
                &self.task,
                questions.len() as u32,
                self.questions_numbered,
            );
            let wait = crate::questions::parse_wait(
                input.get(crate::tool::builtins::ask_user::ASK_USER_WAIT_PARAM_NAME),
            );
            let (asked, wait, result) = crate::questions::plan_ask(
                last,
                questions,
                wait,
                self.task.agent_config.tool_config.async_questions,
            );
            self.questions_numbered = last + asked.len() as u32;
            // A question the agent waits for ends its turn.
            call.reply = Some(if wait {
                let ids = asked.iter().map(|question| question.id.as_str());
                format!(
                    "Asked {}. End your turn now without calling tools, and the answer arrives as the next message.",
                    crate::questions::id_list(&ids.collect::<Vec<_>>())
                )
            } else {
                result.clone()
            });
            QuestionCall::Asked {
                asked,
                wait,
                result,
            }
        } else if name == crate::tool::builtins::CLOSE_QUESTIONS_TOOL_NAME {
            use crate::tool::builtins::close_questions as close;
            let Some(Ok(ids)) = input
                .get(close::CLOSE_QUESTIONS_IDS_PARAM_NAME)
                .map(crate::questions::parse_ids)
            else {
                return;
            };
            let why = input
                .get(close::CLOSE_QUESTIONS_WHY_PARAM_NAME)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|why| !why.is_empty())
                .map(str::to_string);
            let last = self.task.questions.asked.max(self.questions_numbered);
            let (ids, result) = crate::questions::plan_close(last, ids);
            call.reply = Some(result.clone());
            QuestionCall::Closed { ids, why, result }
        } else {
            return;
        };
        self.question_calls.insert(call.id, taken);
    }

    async fn reject_call(
        &self,
        id: Uuid,
        name: &str,
        reason: String,
    ) -> Result<Value, ArkoAgentsError> {
        self.emit(AgentEvent::ToolRejected {
            tool_call_id: id,
            tool_name: name.to_string(),
            reason: Some(reason.clone()),
        })
        .await?;
        Ok(refused(name, &reason))
    }

    /// An approved (or auto-approved) call: a built-in passes the
    /// middleware's `before`. An arko call is admitted for its MCP call,
    /// where the middleware runs.
    async fn admit(
        &mut self,
        name: &str,
        built_in: bool,
        mut call: Admitted,
        input: &Value,
        session: &cc::ClaudeSession,
    ) -> Result<Value, ArkoAgentsError> {
        if !built_in {
            self.take_question_call(name, input, &mut call);
            self.tool_host()
                .admitted
                .lock()
                .unwrap()
                .entry(input_key(name, input))
                .or_default()
                .push_back(call);
            return Ok(allow_tool());
        }
        let Admitted {
            id, custom_output, ..
        } = call;
        if let Some(output) = custom_output {
            // Claude Code runs a built-in itself: output the host supplies
            // is its result instead of running it.
            return Ok(cc::Hook::deny_tool(format!(
                "Not an error: the host supplied this {name} call's result instead of running it. Result: {output}"
            )));
        }
        let Some(middleware) = self.middleware.clone() else {
            return Ok(allow_tool());
        };
        let params = params_of(input);
        let execution = ToolExecutionContext {
            task_id: self.task.id,
            agent_name: &self.task.agent_config.name,
            initial_task_prompt: &self.task.initial_task_prompt,
            current_working_directory: &self.task.current_working_directory,
        };
        let call = ToolCallContext {
            tool_call_id: id,
            tool_name: name,
            parameters: &params,
        };
        let decision = middleware.before(&execution, &call).await;
        let (answer, event_reason) = match decision {
            Ok(ToolExecutionDecision::Continue) => return Ok(allow_tool()),
            Ok(ToolExecutionDecision::Reject { message_to_agent }) => {
                (refused(name, &message_to_agent), message_to_agent)
            }
            Ok(ToolExecutionDecision::FailTask { reason }) => {
                self.set_state(TaskState::Failed {
                    reason: reason.clone(),
                    cause: None,
                    error_kind: None,
                })
                .await?;
                let _ = session.interrupt().await;
                (refused(name, &reason), reason)
            }
            Ok(ToolExecutionDecision::CancelTask { reason }) => {
                self.set_state(TaskState::Cancelled {
                    reason: reason.clone(),
                })
                .await?;
                let _ = session.interrupt().await;
                (refused(name, &reason), reason)
            }
            Ok(ToolExecutionDecision::PauseTask {
                message_to_agent,
                reason,
            }) => {
                self.pause_after_turn = Some((reason, ErrorRecoveryCause::Middleware));
                let _ = session.interrupt().await;
                (refused(name, &message_to_agent), message_to_agent)
            }
            Err(error) => {
                let reason = format!("The execution middleware failed before '{name}': {error}");
                self.pause_after_turn = Some((reason.clone(), ErrorRecoveryCause::Middleware));
                let _ = session.interrupt().await;
                (refused(name, &reason), reason)
            }
        };
        self.emit(AgentEvent::ToolRejected {
            tool_call_id: id,
            tool_name: name.to_string(),
            reason: Some(event_reason),
        })
        .await?;
        Ok(answer)
    }

    /// A permission request: the gate asked for approval (or Claude Code
    /// asks about a call the gate didn't see). The call waits for the host's
    /// decision.
    async fn permission(
        &mut self,
        request: cc::PermissionRequest,
        reply: oneshot::Sender<cc::PermissionDecision>,
    ) -> Result<(), ArkoAgentsError> {
        let known = request
            .tool_use_id
            .as_ref()
            .and_then(|tool_use_id| self.calls.get(tool_use_id))
            .map(|call| (call.id, call.name.clone(), call.built_in));
        let Some((id, name, built_in)) = known else {
            // Not gated (an unexpected request): refused.
            let _ = reply.send(cc::PermissionDecision::deny(
                "This call did not pass the host's checks, so it was not run.",
            ));
            return Ok(());
        };
        let judged = built_in
            .then(|| self.input_from_session_directory(&name, &request.input))
            .flatten();
        let config = &self.task.agent_config;
        let params = params_of(judged.as_ref().unwrap_or(&request.input));
        let description = if built_in {
            format!("Claude Code's {name} tool.")
        } else {
            self.registry
                .get(&name)
                .map(|tool| tool.definition().description)
                .unwrap_or_default()
        };
        let outside = if built_in {
            built_in_path_parameters(&name)
                .iter()
                .find(|param| {
                    params
                        .get(**param)
                        .and_then(Value::as_str)
                        .is_some_and(|path| {
                            matches!(
                                crate::utils::is_path_in_workspace(
                                    path,
                                    &self.task.current_working_directory,
                                ),
                                Ok(false)
                            )
                        })
                })
                .map(|param| param.to_string())
        } else {
            None
        };
        let reason = ApprovalReason {
            tool_not_auto_approved: !config.should_auto_approve_tool(&name),
            path_outside_workspace: outside,
        };
        let deadline = config
            .user_command_timeout_seconds
            .filter(|_| !self.host_decisions.get())
            .map(|seconds| tokio::time::Instant::now() + Duration::from_secs(seconds));
        self.approval = Some(Approval {
            tool_use_id: request.tool_use_id.clone(),
            id,
            name: name.clone(),
            input: request.input.clone(),
            reply,
            deadline,
        });
        self.emit(AgentEvent::ToolCallAwaitingApproval {
            tool_call_id: id,
            tool_name: name.clone(),
            parameters: params.clone(),
            tool_description: description.clone(),
            approval_reason: reason.clone(),
        })
        .await?;
        self.set_state(TaskState::AwaitingToolApproval {
            tool_call_id: id,
            tool_name: name,
            parameters: params,
            tool_description: description,
            approval_reason: reason,
        })
        .await
    }

    async fn decide_approval(
        &mut self,
        id: Uuid,
        approve: bool,
        custom_output: Option<String>,
        message_to_agent: Option<String>,
        reason: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        if self
            .approval
            .as_ref()
            .is_none_or(|approval| approval.id != id)
        {
            log::warn!(
                "Task {}: no call {id} awaits approval. Ignored.",
                self.task.id
            );
            return Ok(());
        }
        let approval = self.approval.take().expect("checked");
        self.set_state(TaskState::InProgress).await?;
        if !approve {
            let reason = reason.unwrap_or_else(|| "The user rejected this call.".into());
            self.emit(AgentEvent::ToolRejected {
                tool_call_id: id,
                tool_name: approval.name.clone(),
                reason: Some(reason.clone()),
            })
            .await?;
            let _ = approval.reply.send(cc::PermissionDecision::deny(reason));
            return Ok(());
        }
        if let (Some(tool_use_id), Some(message)) = (&approval.tool_use_id, message_to_agent) {
            self.after_result.insert(
                tool_use_id.clone(),
                format!("Message from user on approval:\n{message}"),
            );
        }
        let built_in = self
            .calls
            .values()
            .find(|call| call.id == id)
            .is_some_and(|call| call.built_in);
        let decision = if built_in {
            match custom_output {
                Some(output) => cc::PermissionDecision::deny(output),
                None => {
                    self.record_running(approval.tool_use_id.as_ref(), None)
                        .await;
                    cc::PermissionDecision::Allow {
                        updated_input: self
                            .input_in_task_directory(&approval.name, &approval.input),
                        updated_permissions: Vec::new(),
                    }
                }
            }
        } else {
            self.record_running(approval.tool_use_id.as_ref(), custom_output.clone())
                .await;
            let mut call = Admitted {
                id,
                call_id: approval.tool_use_id.clone(),
                custom_output,
                reply: None,
            };
            self.take_question_call(&approval.name, &approval.input, &mut call);
            self.tool_host()
                .admitted
                .lock()
                .unwrap()
                .entry(input_key(&approval.name, &approval.input))
                .or_default()
                .push_back(call);
            cc::PermissionDecision::allow()
        };
        let _ = approval.reply.send(decision);
        Ok(())
    }

    async fn approval_timed_out(&mut self) -> Result<(), ArkoAgentsError> {
        let Some(approval) = self.approval.take() else {
            return Ok(());
        };
        let seconds = self
            .task
            .agent_config
            .user_command_timeout_seconds
            .unwrap_or_default();
        let reason = format!(
            "Human approval timed out after {seconds} seconds, so this call was not run. You can call it again if it is still needed."
        );
        self.emit(AgentEvent::ToolRejected {
            tool_call_id: approval.id,
            tool_name: approval.name.clone(),
            reason: Some(reason.clone()),
        })
        .await?;
        let _ = approval.reply.send(cc::PermissionDecision::deny(reason));
        self.set_state(TaskState::InProgress).await
    }

    /// After a call ran: the middleware's `after` for a built-in, and
    /// context for the model with the result.
    async fn after_tool(
        &mut self,
        tool_use_id: Option<String>,
        _served: &str,
        response: Value,
        session: &cc::ClaudeSession,
    ) -> Result<Value, ArkoAgentsError> {
        let call = tool_use_id.as_ref().and_then(|id| self.calls.get(id));
        if let (Some(call), Some(middleware)) = (call, self.middleware.clone())
            && call.built_in
        {
            let params = params_of(&call.input);
            let output = result_text(&response);
            let execution = ToolExecutionContext {
                task_id: self.task.id,
                agent_name: &self.task.agent_config.name,
                initial_task_prompt: &self.task.initial_task_prompt,
                current_working_directory: &self.task.current_working_directory,
            };
            let context = ToolCallContext {
                tool_call_id: call.id,
                tool_name: &call.name,
                parameters: &params,
            };
            if let Err(error) = middleware
                .after(&execution, &context, ToolExecutionOutcome::Success(&output))
                .await
            {
                self.pause_after_turn = Some((
                    format!(
                        "The execution middleware failed after '{}': {error}",
                        call.name
                    ),
                    ErrorRecoveryCause::Middleware,
                ));
            }
        }
        let mut texts = Vec::new();
        if let Some(message) = tool_use_id.and_then(|id| self.after_result.remove(&id)) {
            texts.push(message);
        }
        // A hook's context is text: its images wait for the next input. What
        // the user typed goes as their own message instead.
        let (context, typed, _) = self.take_context(false).unwrap_or_default();
        // The task holds it once the CLI reports the results it went with
        // (or it is pending again, cancelled), not now: a CLI that stops
        // first leaves the task without it.
        let marks = std::mem::take(&mut self.taken_marks);
        raise(&mut self.delivered_marks, &marks);
        let headed = !context.is_empty();
        if headed {
            texts.push(context);
        }
        if texts.is_empty() && typed.is_empty() {
            return Ok(cc::Hook::proceed());
        }
        let mut recorded = texts.clone();
        if !typed.is_empty() {
            // A note alone has the update's heading in the record and the hook.
            let shown = if headed {
                typed.clone()
            } else {
                format!("{CONTEXT_UPDATE}\n\n{typed}")
            };
            // It counts as delivered once it is written, and goes with the
            // next input if the CLI cancels it (see `note_settled`). Where
            // the write fails, the hook has it.
            match session.send_human(&typed).await {
                Ok(uuid) => {
                    // A note cancelled goes back to the task, which then
                    // holds the update when the note was all of it.
                    let marks = if headed {
                        BTreeMap::new()
                    } else {
                        marks.clone()
                    };
                    self.sent_notes.insert(uuid, (typed, marks));
                }
                Err(error) => {
                    log::warn!(
                        "Task {}: could not send the user's notes as a message ({error}). They go in the hook.",
                        self.task.id
                    );
                    texts.push(shown.clone());
                }
            }
            recorded.push(shown);
        }
        self.delivered_context.push(recorded.join("\n\n"));
        self.emit_context_receipts().await?;
        if texts.is_empty() {
            return Ok(cc::Hook::proceed());
        }
        Ok(add_context(&texts.join("\n\n")))
    }

    /// The CLI reports each message it is sent by its UUID, from `queued` to
    /// `completed`, or `cancelled` when the turn stopped before the model's
    /// next request. A note it cancelled never reached the model, so it goes
    /// with the next input.
    fn note_settled(&mut self, raw: &Value) {
        let state = raw["state"].as_str();
        if raw["type"] == "command_lifecycle"
            && matches!(state, Some("completed" | "cancelled"))
            && let Some((text, marks)) = raw["command_uuid"]
                .as_str()
                .and_then(|uuid| self.sent_notes.remove(uuid))
            && state == Some("cancelled")
        {
            self.queue_context(text, false);
            self.task.raise_context_marks(&marks);
        }
    }

    /// A message the session reported.
    async fn handle_message(
        &mut self,
        message: cc::Message,
        session: &cc::ClaudeSession,
    ) -> Result<(), ArkoAgentsError> {
        let result = self.handle_message_inner(message, session).await;
        if std::mem::take(&mut self.cache_interrupt) {
            let _ = session.interrupt().await;
        }
        result
    }

    /// Claude Code answered the input sent after a cut back or a branch: it
    /// has saved the input, so the cut (or the fork) is saved too, and the
    /// messages sent with it count as sent.
    fn confirm_resumed(&mut self) {
        if let Some(synced) = self.synced_on_send.take() {
            self.task.claude_code_synced = synced;
            self.task.claude_code_resume_at = None;
            self.task.claude_code_fork_of = None;
        }
    }

    async fn handle_message_inner(
        &mut self,
        message: cc::Message,
        session: &cc::ClaudeSession,
    ) -> Result<(), ArkoAgentsError> {
        // Any message of the turn but Claude Code's status: the turn runs,
        // so Claude Code has the input.
        if !matches!(message, cc::Message::System(_)) {
            self.confirm_resumed();
        }
        match message {
            cc::Message::Assistant(assistant) => {
                // A failed request's own message (see `api_error` below)
                // reports no request: its usage is all zeros.
                if let Some(usage) = &assistant.usage
                    && assistant.raw.get("error").and_then(Value::as_str).is_none()
                {
                    let key = assistant
                        .raw
                        .pointer("/message/id")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| Uuid::new_v4().to_string());
                    // The stream events report this request's usage, its
                    // final output included; this message's is a snapshot.
                    if !self.streamed.contains(&key) {
                        self.observe_usage(
                            key,
                            usage.clone(),
                            assistant.model.clone(),
                            assistant.parent_tool_use_id.clone(),
                        )
                        .await?;
                    }
                }
                // Over the host's budget: the turn ends here, held.
                if self.pause_after_turn.is_none()
                    && let Some(reason) = self.held_request()
                {
                    self.pause_after_turn = Some((reason, ErrorRecoveryCause::BudgetExceeded));
                    let _ = session.interrupt().await;
                }
                // A subagent's messages are its own conversation.
                if assistant.parent_tool_use_id.is_some() {
                    return Ok(());
                }
                // A request that failed: Claude Code reports it as a message
                // of its own, its error's kind in `error`.
                self.api_error = assistant
                    .raw
                    .get("error")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                // The first message of a response (a failed request's isn't
                // one) answers the request that ended with the latest entry.
                // A response's messages (its text, each of its calls) share
                // its ID, and the results of its calls come between them as
                // the calls finish.
                let id = assistant.raw.pointer("/message/id").and_then(Value::as_str);
                let first = match id {
                    Some(id) => self.response.as_deref() != Some(id),
                    None => self.last_entry.is_some(),
                };
                if self.api_error.is_none() && first {
                    self.response = id.map(str::to_string);
                    self.answered = self.last_entry.take();
                    self.step_results = false;
                }
                // The response's text and thinking streamed as they came
                // (`content_block_delta`): not again.
                let streamed = id.is_some_and(|id| self.streamed_text.contains(id));
                let mut text = String::new();
                let mut calls = Vec::new();
                for block in &assistant.content {
                    match block {
                        // Claude Code's note about a failed request is a
                        // warning for the host, and not part of the reply or
                        // the conversation.
                        cc::ContentBlock::Text(note) if self.api_error.is_some() => {
                            self.emit(AgentEvent::ServiceWarning { text: note.clone() })
                                .await?;
                        }
                        cc::ContentBlock::Text(chunk) => {
                            if !streamed {
                                self.emit(AgentEvent::AssistantResponseChunk(chunk.clone()))
                                    .await?;
                            }
                            text.push_str(chunk);
                        }
                        cc::ContentBlock::Thinking(thinking)
                            if !thinking.is_empty() && !streamed =>
                        {
                            self.emit(AgentEvent::ReasoningChunk(thinking.clone()))
                                .await?;
                        }
                        cc::ContentBlock::ToolUse { id, name, input } => {
                            let name = arko_name(name).unwrap_or(name).to_string();
                            calls.push(ai::ToolCall {
                                id: Some(id.clone()),
                                name,
                                arguments: input.clone(),
                                thought_signature: None,
                                provider_id: None,
                                invalid_arguments: None,
                            });
                        }
                        _ => {}
                    }
                }
                if !text.is_empty() || !calls.is_empty() {
                    let mut message = ai::Message::new_assistant(&text);
                    message.tool_calls = calls;
                    self.commit(message);
                    self.record_entry(assistant.raw.get("uuid").and_then(Value::as_str));
                    // The model read the results of the calls that finished,
                    // so Claude Code has saved them.
                    self.task
                        .running_tool_calls
                        .retain(|running| running.output.is_none());
                    self.report(false).await;
                }
                self.check_handover_threshold().await?;
            }
            cc::Message::User(user) if user.parent_tool_use_id.is_none() => {
                self.flush_usage().await?;
                let mut results = Vec::new();
                for block in &user.content {
                    let cc::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    else {
                        continue;
                    };
                    let text = result_text(content);
                    self.record_output(Some(tool_use_id), content);
                    let Some(call) = self.calls.get(tool_use_id) else {
                        continue;
                    };
                    // A built-in's images (Claude Code's `Read`): an arko
                    // tool's reached the sink as it ran.
                    if call.built_in
                        && let Some(sink) = &self.image_sink
                    {
                        let mut images = result_images(content);
                        // Captioned as `read_file` captions its images: the
                        // file's path and size.
                        if let Some(path) = call.record.arguments["file_path"].as_str() {
                            for image in &mut images {
                                image.description = match image.decoded_len() {
                                    Some(bytes) => format!("{path}, {}", file_size(bytes)),
                                    None => path.to_string(),
                                };
                            }
                        }
                        if !images.is_empty() {
                            sink.tool_images_with_call_id(
                                self.task.id,
                                call.id,
                                Some(tool_use_id),
                                &call.name,
                                &images,
                            );
                        }
                    }
                    let (id, name, record) = (call.id, call.name.clone(), call.record.clone());
                    results.push(if *is_error {
                        ai::ToolResult::error_from_tool_call(&record, &text)
                    } else {
                        ai::ToolResult::from_tool_call(&record, &text)
                    });
                    self.call_finished(id, &name, &record.arguments, &text, *is_error, session)
                        .await?;
                }
                let mut recorded = self.task.conversation.messages.len();
                let with_results = !results.is_empty();
                if with_results {
                    self.task.conversation.add_tool_results(results);
                    self.observe_last();
                    recorded += 1;
                    for context in std::mem::take(&mut self.delivered_context) {
                        self.task.conversation.add_user_message(&context);
                        self.task.increment_user_message_count();
                        self.observe_last();
                    }
                    let marks = std::mem::take(&mut self.delivered_marks);
                    self.task.raise_context_marks(&marks);
                    self.task.claude_code_synced = self.task.conversation.messages.len();
                    self.report(false).await;
                }
                // Claude Code adds the context delivered with the results after
                // them, so a session cut back to them leaves it out, and it is
                // sent again.
                if let Some(uuid) = user.raw.get("uuid").and_then(Value::as_str)
                    && user
                        .content
                        .iter()
                        .any(|block| matches!(block, cc::ContentBlock::ToolResult { .. }))
                {
                    self.last_entry = Some(Entry {
                        uuid: uuid.to_string(),
                        messages: recorded,
                    });
                    if with_results {
                        self.task
                            .claude_code_entries
                            .insert(recorded - 1, uuid.to_string());
                    }
                    self.step_results = true;
                }
            }
            cc::Message::StreamEvent {
                event,
                parent_tool_use_id,
                ..
            } => match event["type"].as_str() {
                Some("message_start") => {
                    let message = &event["message"];
                    if parent_tool_use_id.is_none() {
                        // A response: any retry before it is over, and a
                        // response that streamed without finishing failed.
                        self.discard_unfinished_stream().await?;
                        self.streaming = message["id"].as_str().map(str::to_string);
                        self.retrying_since = None;
                    }
                    self.request_started(message, &parent_tool_use_id).await?;
                    if let (Some(key), Some(usage)) = (message["id"].as_str(), message.get("usage"))
                    {
                        self.streamed.insert(key.to_string());
                        self.observe_usage(
                            key.to_string(),
                            usage.clone(),
                            message["model"].as_str().map(str::to_string),
                            parent_tool_use_id,
                        )
                        .await?;
                        self.check_handover_threshold().await?;
                    }
                }
                Some("message_delta") => {
                    if parent_tool_use_id.is_none() {
                        // The response finished: what it streamed stands.
                        self.streamed_bytes = (0, 0);
                    }
                    // The request's final usage.
                    if let (Some(pending), Some(Value::Object(usage))) = (
                        self.pending_usage.get_mut(&parent_tool_use_id),
                        event.get("usage"),
                    ) {
                        if let Value::Object(fields) = &mut pending.usage {
                            for (field, value) in usage {
                                fields.insert(field.clone(), value.clone());
                            }
                        }
                        let delta = &event["delta"];
                        pending.stop_reason = delta["stop_reason"].as_str().map(str::to_string);
                        pending.stop_details = Some(delta["stop_details"].clone())
                            .filter(|details| !details.is_null());
                        if pending.stop_reason.as_deref() == Some("refusal") {
                            self.refusal_details = pending.stop_details.clone();
                        }
                        self.flush_thread(&parent_tool_use_id).await?;
                    }
                }
                // A response that stops without its `message_delta` was
                // abandoned. Its stream failed, and Claude Code asks for it
                // again. Asked without streaming, the new response comes
                // whole, with no `message_start`. What the abandoned one
                // streamed goes now, so the host drops it before the new
                // response shows, and keeps the new one.
                Some("message_stop") if parent_tool_use_id.is_none() => {
                    self.discard_unfinished_stream().await?;
                }
                // The main conversation's text and thinking as they come. A
                // subagent's are its own.
                Some("content_block_delta") if parent_tool_use_id.is_none() => {
                    let delta = &event["delta"];
                    let chunk = match delta["type"].as_str() {
                        Some("text_delta") => delta["text"]
                            .as_str()
                            .map(|text| AgentEvent::AssistantResponseChunk(text.to_string())),
                        Some("thinking_delta") => delta["thinking"]
                            .as_str()
                            .map(|thinking| AgentEvent::ReasoningChunk(thinking.to_string())),
                        _ => None,
                    };
                    if let Some(chunk) = chunk.filter(|chunk| match chunk {
                        AgentEvent::AssistantResponseChunk(text)
                        | AgentEvent::ReasoningChunk(text) => !text.is_empty(),
                        _ => false,
                    }) {
                        if let Some(id) = &self.streaming {
                            self.streamed_text.insert(id.clone());
                        }
                        match &chunk {
                            AgentEvent::AssistantResponseChunk(text) => {
                                self.streamed_bytes.0 += text.len();
                            }
                            AgentEvent::ReasoningChunk(thinking) => {
                                self.streamed_bytes.1 += thinking.len();
                            }
                            _ => {}
                        }
                        self.emit(chunk).await?;
                    }
                }
                _ => {}
            },
            cc::Message::RateLimit { info, .. } => self.plan_usage_reported(&info, session).await?,
            cc::Message::Other(raw) => self.note_settled(&raw),
            cc::Message::System(system) if system.subtype == "model_refusal_fallback" => {
                // Claude Code retried the refused request on its fallback
                // model.
                let refusal = refusal_of(&system.raw, self.refusal_details.take());
                self.emit(AgentEvent::RefusalReceived(refusal)).await?;
                let model_id = system.raw["fallback_model"].as_str().unwrap_or_default();
                self.emit(AgentEvent::RefusalFallback {
                    model_id: model_id.to_string(),
                })
                .await?;
            }
            cc::Message::System(system) if system.subtype == "model_refusal" => {
                // The refusal stands: no retry ran.
                let refusal = refusal_of(&system.raw, self.refusal_details.take());
                self.emit(AgentEvent::RefusalReceived(refusal.clone()))
                    .await?;
                self.refused = Some(refusal);
            }
            cc::Message::System(system) if system.subtype == "model_fallback" => {
                self.emit(AgentEvent::ModelRerouted {
                    provider: ai::Provider::ClaudeCode.as_telemetry_str().to_string(),
                    requested_model: system.raw["original_model"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    served_model: system.raw["fallback_model"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                })
                .await?;
            }
            cc::Message::System(system) if system.subtype == "compact_boundary" => {
                self.compacted(&system.raw).await?;
            }
            cc::Message::System(system) if system.subtype == "api_retry" => {
                self.retry_scheduled(&system.raw).await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// A response that streamed and didn't finish failed: the host is told
    /// to drop what it streamed ([`AgentEvent::StreamedOutputDiscarded`]).
    async fn discard_unfinished_stream(&mut self) -> Result<(), ArkoAgentsError> {
        let (text_bytes, reasoning_bytes) = std::mem::take(&mut self.streamed_bytes);
        if text_bytes > 0 || reasoning_bytes > 0 {
            self.emit(AgentEvent::StreamedOutputDiscarded {
                text_bytes,
                reasoning_bytes,
            })
            .await?;
        }
        Ok(())
    }

    /// Claude Code will retry a failed request (its `api_retry` message):
    /// reported as the runtime reports its own retries, an error that needs
    /// no decision and the retry with its delay.
    async fn retry_scheduled(&mut self, raw: &Value) -> Result<(), ArkoAgentsError> {
        self.discard_unfinished_stream().await?;
        let field = |name: &str| raw.get(name).and_then(Value::as_u64);
        let attempt = field("attempt").unwrap_or(1);
        let error = raw.get("error").and_then(Value::as_str);
        let status = field("error_status").and_then(|status| u16::try_from(status).ok());
        let since = *self
            .retrying_since
            .get_or_insert_with(std::time::Instant::now);
        let what = match (error, status) {
            (Some(error), Some(status)) => format!("{error}, status {status}"),
            (Some(error), None) => error.to_string(),
            (None, Some(status)) => format!("status {status}"),
            (None, None) => "an error".to_string(),
        };
        let attempts = field("max_retries").map_or(String::new(), |max| format!(" of {max}"));
        log::info!(
            "Task {}: Claude Code is retrying a request ({what}), attempt {attempt}{attempts}.",
            self.task.id
        );
        let kind = ai::ErrorKind::from_claude_code(error, status, None, "");
        self.emit(AgentEvent::ErrorEncountered {
            error_message: format!(
                "Claude Code's request failed ({what}). It retries (attempt {attempt}{attempts})."
            ),
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: Some(kind).filter(|kind| *kind != ai::ErrorKind::Unknown),
        })
        .await?;
        self.emit(AgentEvent::RetryScheduled {
            attempt: u32::try_from(attempt).unwrap_or(u32::MAX),
            delay_ms: field("retry_delay_ms").unwrap_or(0),
            elapsed_ms: since.elapsed().as_millis() as u64,
            rate_limit: status == Some(429) || error == Some("rate_limit"),
        })
        .await
    }

    /// Records `uuid` as the transcript entry of the conversation's last
    /// message (see [`Task::claude_code_entries`]).
    fn record_entry(&mut self, uuid: Option<&str>) {
        if let (Some(uuid), Some(index)) =
            (uuid, self.task.conversation.messages.len().checked_sub(1))
        {
            self.task
                .claude_code_entries
                .insert(index, uuid.to_string());
        }
    }

    fn commit(&mut self, message: ai::Message) {
        self.task.conversation.messages.push(message);
        self.observe_last();
        self.task.claude_code_synced = self.task.conversation.messages.len();
    }

    /// Tools that track what the conversation holds observe each message.
    fn observe_last(&self) {
        let Some(message) = self.task.conversation.messages.last() else {
            return;
        };
        for name in self.registry.list_tool_names() {
            if let Some(tool) = self.registry.get(&name) {
                tool.observe_context(self.task.id, message);
            }
        }
    }

    /// A call's result arrived: its events, and what its outcome does to the
    /// task. `arguments` are the call's.
    async fn call_finished(
        &mut self,
        id: Uuid,
        name: &str,
        arguments: &Value,
        text: &str,
        is_error: bool,
        session: &cc::ClaudeSession,
    ) -> Result<(), ArkoAgentsError> {
        let outcome = self
            .host
            .as_ref()
            .and_then(|host| host.outcomes.lock().unwrap().remove(&id))
            .map(|(outcome, _)| outcome);
        // What its admission took of the questions counts only once it ran.
        let question_call = self.question_calls.remove(&id);
        match outcome {
            // A built-in, or a call the gate refused.
            None => {
                self.emit(AgentEvent::ToolResult {
                    tool_call_id: id,
                    tool_name: name.to_string(),
                    output: text.to_string(),
                })
                .await?;
                let _ = is_error;
            }
            Some(CallOutcome::Finished {
                raw,
                effective,
                intercepted,
            }) => {
                match &question_call {
                    Some(QuestionCall::Asked { asked, wait, .. }) => {
                        self.task.questions.record(asked, *wait);
                        if let Some(hook) = &self.handover_hook {
                            hook.questions_asked(&self.task, asked, *wait);
                        }
                    }
                    Some(QuestionCall::Closed { ids, .. }) => {
                        self.task.questions.settle(ids);
                    }
                    None => {}
                }
                let (raw, effective) = match &question_call {
                    Some(
                        QuestionCall::Asked { result, .. } | QuestionCall::Closed { result, .. },
                    ) => (result.clone(), result.clone()),
                    None => (raw, effective),
                };
                if let Some(message) = self.registry.get(name).and_then(|tool| {
                    message_user::message_to_user(
                        name,
                        tool.as_ref(),
                        arguments.get(message_user::MESSAGE_USER_MESSAGE_PARAM_NAME),
                    )
                }) {
                    self.emit(AgentEvent::MessageToUser {
                        message,
                        tool_call_id: Some(id),
                    })
                    .await?;
                }
                self.emit(AgentEvent::ToolResult {
                    tool_call_id: id,
                    tool_name: name.to_string(),
                    output: raw.clone(),
                })
                .await?;
                if intercepted && raw != effective {
                    self.emit(AgentEvent::ToolResultIntercepted {
                        tool_call_id: id,
                        tool_name: name.to_string(),
                        original_output: raw.clone(),
                        intercepted_output: effective,
                    })
                    .await?;
                }
                match question_call {
                    Some(QuestionCall::Asked {
                        asked, wait: true, ..
                    }) => {
                        self.questions = Some((crate::questions::questions_text(&asked), asked));
                    }
                    Some(QuestionCall::Asked { asked, .. }) => {
                        self.emit(AgentEvent::UserQuestionsAsked {
                            questions: crate::questions::questions_text(&asked),
                            asked,
                            wait: false,
                        })
                        .await?;
                    }
                    Some(QuestionCall::Closed { ids, why, .. }) if !ids.is_empty() => {
                        self.emit(AgentEvent::QuestionsClosed { ids, why }).await?;
                    }
                    // A host's own ask_user, whose input the runtime
                    // couldn't read, asks its result, numbered and recorded
                    // as any ask is. It waits, as such a call always did.
                    None if name == ASK_USER_TOOL_NAME => {
                        let (questions, wait) = crate::tool::builtins::ask_user::read_call(&raw);
                        let last = crate::handover::last_question_number(
                            self.handover_hook.as_deref(),
                            &self.task,
                            questions.len() as u32,
                            self.questions_numbered,
                        );
                        let (asked, wait, _) = crate::questions::take_ask(
                            &mut self.task.questions,
                            last,
                            questions,
                            wait,
                            false,
                        );
                        if let Some(hook) = &self.handover_hook {
                            hook.questions_asked(&self.task, &asked, wait);
                        }
                        self.questions = Some((crate::questions::questions_text(&asked), asked));
                    }
                    _ => {}
                }
                if name == COMPLETE_TASK_TOOL_NAME && !self.task.current_state.is_terminal() {
                    self.set_state(TaskState::Completed { summary: raw })
                        .await?;
                    // Its turn ends here.
                    let _ = session.interrupt().await;
                }
            }
            Some(CallOutcome::Rejected(reason)) => {
                self.emit(AgentEvent::ToolRejected {
                    tool_call_id: id,
                    tool_name: name.to_string(),
                    reason: Some(reason),
                })
                .await?;
            }
            Some(CallOutcome::Failed(error)) => {
                let policy = self
                    .task
                    .agent_config
                    .error_recovery_config
                    .on_tool_execution_error
                    .clone();
                if policy != ToolErrorPolicy::PauseAndAwaitApproval {
                    self.emit(AgentEvent::ErrorEncountered {
                        error_message: error.clone(),
                        requires_approval_to_retry: false,
                        tool_call_id: Some(id),
                        error_kind: None,
                    })
                    .await?;
                }
                match policy {
                    ToolErrorPolicy::InformAgentAndContinue => {}
                    // Reported by the pause at the turn's end. A pause the
                    // turn already asked for (a middleware's, a monitor's)
                    // stands, and the error is reported on its own then.
                    ToolErrorPolicy::PauseAndAwaitApproval => {
                        self.tool_errors.push((error.clone(), id));
                        if self.pause_after_turn.is_none() {
                            self.pause_after_turn = Some((error, ErrorRecoveryCause::ToolError));
                        }
                        let _ = session.interrupt().await;
                    }
                    ToolErrorPolicy::FailTask => {
                        self.set_state(TaskState::Failed {
                            reason: error,
                            cause: None,
                            error_kind: None,
                        })
                        .await?;
                        let _ = session.interrupt().await;
                    }
                }
            }
            Some(CallOutcome::Stop { reason, cancel }) => {
                self.emit(AgentEvent::ToolRejected {
                    tool_call_id: id,
                    tool_name: name.to_string(),
                    reason: Some(reason.clone()),
                })
                .await?;
                if !self.task.current_state.is_terminal() {
                    self.set_state(if cancel {
                        TaskState::Cancelled { reason }
                    } else {
                        TaskState::Failed {
                            reason,
                            cause: None,
                            error_kind: None,
                        }
                    })
                    .await?;
                }
                let _ = session.interrupt().await;
            }
            Some(CallOutcome::Pause { message, reason }) => {
                self.emit(AgentEvent::ToolRejected {
                    tool_call_id: id,
                    tool_name: name.to_string(),
                    reason: Some(message),
                })
                .await?;
                self.pause_after_turn = Some((reason, ErrorRecoveryCause::Middleware));
                let _ = session.interrupt().await;
            }
        }
        Ok(())
    }

    /// Notes a request's usage as it starts: the main conversation's
    /// requests set the context size a handover checks at once (a request's
    /// input doesn't change as it runs). The usage is recorded once the
    /// request is complete: its final `message_delta`, or else when the next
    /// request of its thread starts or the turn ends. Claude Code's
    /// `assistant` messages carry the output count as it stood when the
    /// stream started, and its turn result sums the turn's requests, so
    /// neither gives a request's output.
    async fn observe_usage(
        &mut self,
        key: String,
        usage: Value,
        model: Option<String>,
        parent: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        if parent.is_none() {
            // Anthropic's input counts are exclusive: the uncached input,
            // cache reads and cache writes add up to the whole input.
            let input = PendingUsage::input(&usage);
            self.context_size = Some(input);
            self.first_context_size.get_or_insert(input);
        }
        if self
            .pending_usage
            .get(&parent)
            .is_some_and(|pending| pending.key != key)
        {
            self.flush_thread(&parent).await?;
        }
        match self.pending_usage.get_mut(&parent) {
            Some(pending) => pending.usage = usage,
            None => {
                self.pending_usage.insert(
                    parent,
                    PendingUsage {
                        key,
                        usage,
                        model,
                        stop_reason: None,
                        stop_details: None,
                    },
                );
            }
        }
        Ok(())
    }

    /// A request's response started (`message_start`, as early as Claude
    /// Code shows a request): reported with the conversation's size for its
    /// own requests, which, with the input the response reports, calibrates
    /// the bytes per token reported with the next.
    async fn request_started(
        &mut self,
        message: &Value,
        subagent: &Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        let sizes = subagent
            .is_none()
            .then(|| ai::RequestSizes::of(&self.task.conversation, self.served_tools.as_ref()));
        let model = message["model"].as_str().map_or_else(
            || self.task.agent_config.llm_config.model.id.clone(),
            str::to_string,
        );
        self.emit(AgentEvent::ModelRequestStarted {
            model,
            subagent: subagent.clone(),
            sizes,
            bytes_per_token: self.bytes_per_token,
        })
        .await?;
        if let (Some(sizes), Some(usage)) = (sizes, message.get("usage"))
            && let Some(ratio) = super::result_budget::calibrated_bytes_per_token(
                sizes.text_bytes(),
                PendingUsage::input(usage),
            )
        {
            self.bytes_per_token = ratio;
        }
        Ok(())
    }

    /// Records every thread's pending usage.
    async fn flush_usage(&mut self) -> Result<(), ArkoAgentsError> {
        let threads: Vec<Option<String>> = self.pending_usage.keys().cloned().collect();
        for thread in threads {
            self.flush_thread(&thread).await?;
        }
        Ok(())
    }

    /// A request's usage, priced for `model`, as Claude Code reports it
    /// (Anthropic's exclusive input counts).
    fn usage_info(&self, usage: &Value, model: &ai::Model) -> ai::UsageInfo {
        let count = |field: &str| usage.get(field).and_then(Value::as_u64).unwrap_or(0) as usize;
        let mut info = ai::UsageInfo::new(PendingUsage::input(usage), count("output_tokens"))
            .with_cached_tokens(count("cache_read_input_tokens"))
            .with_cache_write_tokens(count("cache_creation_input_tokens"));
        // The list-price estimate counts 1-hour writes at their own rate.
        info.cache_write_1h_tokens = usage
            .pointer("/cache_creation/ephemeral_1h_input_tokens")
            .and_then(Value::as_u64)
            .map(|tokens| tokens as usize);
        info.calculate_cost(
            model,
            Some(&self.task.agent_config.llm_config.generation_config),
        );
        let metered = self
            .task
            .agent_config
            .claude_code
            .as_ref()
            .is_some_and(|settings| settings.metered_billing);
        info.with_billing(if metered {
            ai::BillingKind::Metered
        } else {
            ai::BillingKind::Subscription
        })
    }

    /// Keeps the session's prompt cache warm between turns: a ping in a
    /// fork of the session that Claude Code doesn't save (`--fork-session
    /// --no-session-persistence`), launched with the session's own flags,
    /// so its request reads the conversation's cache. The reply is
    /// discarded, and the task and its session are unchanged. Returns the
    /// ping's usage. Checked against Claude Code 2.1.299: such a fork read
    /// the conversation's cache and left no transcript.
    pub(super) async fn refresh_prompt_cache(mut self) -> Result<ai::UsageInfo, ArkoAgentsError> {
        let session = self
            .task
            .claude_code_session
            .clone()
            .filter(|_| {
                self.task.claude_code_fork_of.is_none()
                    && !self.task.conversation.messages.is_empty()
            })
            .ok_or_else(|| {
                ArkoAgentsError::LLM(ai::Error::UnsupportedFeature(
                    "The prompt cache can't be refreshed. The task has no Claude Code session yet."
                        .to_string(),
                ))
            })?;
        if self.task.conversation.system_message.is_none() {
            let prompt = self.system_prompt()?;
            self.task.conversation.system_message = Some(prompt);
        }
        let (requests, mut host_requests) = mpsc::unbounded_channel();
        let mut options = self.session_options(&requests);
        options.session = cc::SessionStart::Resume {
            session,
            fork: true,
            at: self.task.claude_code_resume_at.clone(),
        };
        options.max_turns = Some(1);
        options.extra_args.push("--no-session-persistence".into());
        let failed = |error: cc::ClaudeError| ArkoAgentsError::Runtime(error.to_string());
        let mut claude = cc::ClaudeSession::start(options).await.map_err(failed)?;
        claude
            .send(crate::runtime::CACHE_PING)
            .await
            .map_err(failed)?;
        // The ping runs no tool: a call it makes is refused.
        let result = loop {
            tokio::select! {
                message = claude.next_message() => match message {
                    Some(cc::Message::Result(result)) => break result,
                    Some(_) => {}
                    None => {
                        return Err(ArkoAgentsError::Runtime(
                            "Claude Code exited before it answered the cache refresh.".to_string(),
                        ));
                    }
                },
                Some(request) = host_requests.recv() => match request {
                    HostRequest::PreToolUse { tool_name, reply, .. } => {
                        let _ = reply.send(refused(&tool_name, "it refreshes the cache only."));
                    }
                    HostRequest::Permission { reply, .. } => {
                        let _ = reply.send(cc::PermissionDecision::deny(
                            "arko refreshes the cache only.",
                        ));
                    }
                    HostRequest::PostToolUse { reply, .. } => {
                        let _ = reply.send(cc::Hook::proceed());
                    }
                },
            }
        };
        let _ = claude.close().await;
        let model = self.task.agent_config.llm_config.model.clone();
        Ok(self.usage_info(&result.raw["usage"], &model))
    }

    /// Records a thread's pending usage.
    async fn flush_thread(&mut self, thread: &Option<String>) -> Result<(), ArkoAgentsError> {
        let Some(pending) = self.pending_usage.remove(thread) else {
            return Ok(());
        };
        let count = |field: &str| {
            pending
                .usage
                .get(field)
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize
        };
        let input = PendingUsage::input(&pending.usage);
        let output = count("output_tokens");
        // Claude Code may serve another model: its refusal fallback keeps
        // the rest of the session on the fallback model.
        let model = pending
            .model
            .as_deref()
            .and_then(|id| ai::Model::get(ai::Provider::Anthropic, id).ok())
            .map(ai::Model::claude_code)
            .unwrap_or_else(|| self.task.agent_config.llm_config.model.clone());
        let written_1h = pending
            .usage
            .pointer("/cache_creation/ephemeral_1h_input_tokens")
            .and_then(Value::as_u64)
            .map(|tokens| tokens as usize);
        let written_5m = pending
            .usage
            .pointer("/cache_creation/ephemeral_5m_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if thread.is_none() {
            // The cache lifetime Claude Code chose, as its writes show.
            if written_1h.is_some_and(|tokens| tokens > 0) {
                self.cache_lifetime =
                    Some(ai::prompt_cache::ANTHROPIC_EXTENDED_PROMPT_CACHE_LIFETIME);
            } else if written_5m > 0 {
                self.cache_lifetime = Some(ai::prompt_cache::ANTHROPIC_PROMPT_CACHE_LIFETIME);
            }
        }
        let info = self.usage_info(&pending.usage, &model);
        if thread.is_some() {
            self.subagent_calls.push(self.task.usage_history.len());
        }
        self.task.record_usage(info.clone());
        self.emit(AgentEvent::UsageReported(info)).await?;
        let served = pending.model.unwrap_or_else(|| model.id.clone());
        let mut call = ai::ModelCallUsage::new(
            pending.key,
            ai::Provider::ClaudeCode.as_telemetry_str().to_string(),
            served,
            input,
            output,
            model.max_input_tokens,
        )
        .with_plan_usage(self.plan_usage.clone());
        call.stop_reason = pending.stop_reason;
        call.stop_details = pending.stop_details;
        call.subagent = thread.clone();
        let call = self.task.record_model_call_usage(call);
        self.emit(AgentEvent::ModelCallUsageReported(call)).await?;
        if thread.is_none() {
            self.monitor_cache().await?;
        }
        Ok(())
    }

    /// Adds what Claude Code's usage data last showed about extra usage to
    /// `usage`, which has the event's flags.
    fn show_extra_usage(&self, usage: &mut ai::PlanUsage) {
        let Some(read) = &self.usage_read else {
            return;
        };
        let extra = usage.extra_usage.get_or_insert_with(Default::default);
        extra.enabled = read.enabled.or(extra.enabled);
        extra.disabled_reason = extra
            .disabled_reason
            .take()
            .or(read.disabled_reason.clone());
        extra.monthly_limit = read.monthly_limit;
        extra.used_credits = read.used_credits;
        extra.utilization = extra.utilization.or(read.utilization);
        extra.currency = read.currency.clone();
        extra.decimal_places = read.decimal_places;
    }

    /// Asks Claude Code for its usage data, which holds the amounts of extra
    /// usage, when this run last asked more than [`USAGE_READ_EVERY`] ago (a
    /// run asks when its turn starts, and an event asks again when due). The
    /// answer comes in the turn's messages, so nothing waits for it.
    async fn ask_usage_if_due(&mut self, session: &cc::ClaudeSession) {
        if self
            .usage_asked
            .is_some_and(|asked| asked.elapsed() < USAGE_READ_EVERY)
        {
            return;
        }
        self.ask_usage(session).await;
    }

    async fn ask_usage(&mut self, session: &cc::ClaudeSession) {
        self.usage_asked = Some(std::time::Instant::now());
        self.usage_reply = match session.request_usage().await {
            Ok(reply) => Some(reply),
            Err(error) => {
                log::debug!(
                    "Task {}: asking Claude Code for usage: {error}",
                    self.task.id
                );
                None
            }
        };
    }

    /// Waits for the answer to the latest ask, up to [`USAGE_END_WAIT`], and
    /// reports what it shows. A session closes right after its last turn,
    /// so the answer to the ask as a turn ends is taken here.
    async fn take_usage_answer(&mut self) -> Result<(), ArkoAgentsError> {
        let Some(reply) = self.usage_reply.take() else {
            return Ok(());
        };
        let answer = tokio::time::timeout(USAGE_END_WAIT, reply)
            .await
            .ok()
            .and_then(Result::ok);
        self.extra_usage_read(answer).await
    }

    async fn usage_answer(
        reply: &mut Option<oneshot::Receiver<Result<Value, String>>>,
    ) -> Option<Result<Value, String>> {
        match reply {
            Some(reply) => reply.await.ok(),
            None => std::future::pending().await,
        }
    }

    /// Claude Code's answer about its usage. One that gives no extra usage,
    /// or an error from a CLI that doesn't know the request, leaves the
    /// event's flags alone and drops what an earlier answer showed. No answer
    /// at all (the session ended) keeps it.
    async fn extra_usage_read(
        &mut self,
        answer: Option<Result<Value, String>>,
    ) -> Result<(), ArkoAgentsError> {
        let Some(answer) = answer else {
            return Ok(());
        };
        self.usage_read = answer.ok().as_ref().and_then(extra_usage_read);
        if self.usage_read.is_none() {
            return Ok(());
        }
        if let Some(mut usage) = self.plan_usage.clone() {
            self.show_extra_usage(&mut usage);
            self.plan_usage = Some(usage.clone());
            self.emit(AgentEvent::PlanUsageReported(usage)).await?;
        }
        Ok(())
    }

    /// Claude Code's plan usage (`rate_limit_event`: its five-hour window as
    /// the primary one, its seven-day window as the secondary), reported
    /// and checked against the opt-in threshold.
    async fn plan_usage_reported(
        &mut self,
        info: &Value,
        session: &cc::ClaudeSession,
    ) -> Result<(), ArkoAgentsError> {
        let mut usage = plan_usage_of(info);
        self.show_extra_usage(&mut usage);
        self.ask_usage_if_due(session).await;
        self.limit_reached = (info["status"] == "rejected" && info["isUsingOverage"] != true)
            .then(|| info["resetsAt"].as_u64());
        self.emit(AgentEvent::PlanUsageReported(usage.clone()))
            .await?;
        self.plan_usage = Some(usage.clone());
        let Some(threshold) = self.task.agent_config.monitoring.plan_usage.clone() else {
            return Ok(());
        };
        let now = chrono::Utc::now().timestamp().max(0) as u64;
        let reached = usage
            .most_used_window()
            .filter(|(_, window)| window.used_percent >= threshold.percent)
            .filter(|(_, window)| window.resets_at.is_none_or(|resets_at| resets_at > now));
        let Some((window, reached)) = reached else {
            self.slow_down = None;
            return Ok(());
        };
        let mark = crate::task::PlanUsageWindowMark {
            window: window.to_string(),
            resets_at: reached.resets_at,
        };
        let first = self.task.plan_usage_threshold_reported.as_ref() != Some(&mark);
        if first {
            self.task.plan_usage_threshold_reported = Some(mark);
            self.emit(AgentEvent::PlanUsageThresholdReached {
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
            crate::agent_config::PlanUsageAction::Pause => {
                if first {
                    self.pause_after_turn = Some((
                        format!(
                            "Plan usage reached {:.1}% of the {window} usage-limit window (threshold {:.1}%).",
                            reached.used_percent, threshold.percent
                        ),
                        ErrorRecoveryCause::Monitor,
                    ));
                    let _ = session.interrupt().await;
                }
            }
            crate::agent_config::PlanUsageAction::SlowDown { delay_ms } => {
                if self.slow_down.is_none() {
                    // The configured TTL, else the one Claude Code's writes
                    // show, else Anthropic's shorter one; Claude Code's
                    // subagents cache for 5 minutes whatever the main TTL.
                    let config = &self.task.agent_config;
                    let subagents = config
                        .tool_config
                        .available_tool_names
                        .iter()
                        .any(|tool| tool == "Agent" || tool == "Task");
                    let lifetime = if subagents {
                        ai::prompt_cache::ANTHROPIC_PROMPT_CACHE_LIFETIME
                    } else {
                        config
                            .llm_config
                            .generation_config
                            .anthropic_cache_ttl
                            .map(ai::AnthropicCacheTtl::duration)
                            .or(self.cache_lifetime)
                            .unwrap_or(ai::prompt_cache::ANTHROPIC_PROMPT_CACHE_LIFETIME)
                    };
                    super::monitor::warn_if_slow_down_outlasts_cache(
                        self.task.id,
                        delay_ms,
                        ai::PromptCacheLifetime::Known(lifetime),
                    );
                }
                self.slow_down = Some(Duration::from_millis(delay_ms));
            }
        }
        Ok(())
    }

    /// The opt-in cache monitor, after a request of the main conversation:
    /// as for other agents, over the current conversation's requests.
    async fn monitor_cache(&mut self) -> Result<(), ArkoAgentsError> {
        let Some(config) = self.task.agent_config.monitoring.cache.clone() else {
            return Ok(());
        };
        let calls: Vec<ai::UsageInfo> = self
            .task
            .usage_history
            .iter()
            .enumerate()
            .skip(self.task.cache_monitor_start())
            .filter(|(index, _)| !self.subagent_calls.contains(index))
            .map(|(_, usage)| usage.clone())
            .collect();
        let Some(efficiency) = super::monitor::cache_efficiency(&calls, &config) else {
            return Ok(());
        };
        self.emit(AgentEvent::CacheEfficiencyReported {
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
                self.emit(AgentEvent::CacheEfficiencyLow {
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
                self.pause_after_turn = Some((
                    format!(
                        "Prompt cache efficiency fell to {:.1}% over the last {} calls (threshold {:.1}%).",
                        efficiency.rolling * 100.0,
                        efficiency.calls,
                        threshold * 100.0
                    ),
                    ErrorRecoveryCause::Monitor,
                ));
                self.cache_interrupt = true;
                self.emit(AgentEvent::CacheEfficiencyLow {
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

    /// `size` (a request's input), when it reached the handover threshold
    /// of a conversation whose first request didn't: a conversation that
    /// starts over the threshold would start over it again after another
    /// handover or compaction.
    fn over_threshold(&mut self, size: usize) -> Option<usize> {
        let threshold = self.handover_threshold.of(&self.task.agent_config)?;
        if size < threshold {
            return None;
        }
        if self
            .first_context_size
            .is_some_and(|first| first >= threshold)
        {
            if !std::mem::replace(&mut self.handover_skip_warned, true) {
                let what = if self.compacts() {
                    "compaction"
                } else {
                    "handover"
                };
                log::warn!(
                    "Task {}: No {what}. The conversation's first request was already over the handover threshold ({threshold}).",
                    self.task.id
                );
            }
            return None;
        }
        Some(size)
    }

    /// After a request of the main conversation: a handover starts once its
    /// input reaches the threshold. The report request reaches the model as
    /// the result of its next tool call, or as a message when its turn ends.
    /// An agent that compacts holds its next call instead and ends its turn,
    /// and Claude Code compacts the conversation then.
    async fn check_handover_threshold(&mut self) -> Result<(), ArkoAgentsError> {
        if self.handover_due() || self.compaction.is_some() {
            return Ok(());
        }
        let Some(size) = self.context_size.and_then(|size| self.over_threshold(size)) else {
            return Ok(());
        };
        if self.compacts() {
            log::info!(
                "Task {}: Compaction due at {size} input tokens.",
                self.task.id
            );
            self.compaction = Some(Compaction::Due);
            return Ok(());
        }
        self.task.handover_requested = Some(HandoverRequest {
            report_requested: true,
            ..HandoverRequest::new(size)
        });
        log::info!(
            "Task {}: Handover requested at {size} input tokens.",
            self.task.id
        );
        self.emit(AgentEvent::HandoverRequested { input_tokens: size })
            .await
    }

    /// A turn ended. Returns how the session ends, or `None` to go on.
    async fn turn_ended(
        &mut self,
        result: &cc::ResultMessage,
        session: &cc::ClaudeSession,
    ) -> Result<Option<SessionEnd>, ArkoAgentsError> {
        self.flush_usage().await?;
        // The amounts of extra usage after this turn are asked for as it
        // ends, so the plan usage is current once the turn is over. A task
        // that already ended has no reader for a report, and without a plan
        // usage there is nothing to add the amounts to.
        if self.stopping.is_none()
            && !self.task.current_state.is_terminal()
            && self.plan_usage.is_some()
        {
            self.ask_usage(session).await;
            self.take_usage_answer().await?;
        }
        // A retry series ends with its turn, also one that failed, and so
        // does a response that streamed without finishing.
        self.retrying_since = None;
        self.discard_unfinished_stream().await?;
        let api_error = self.api_error.take();
        let ending = self.stopping.is_some() || self.task.current_state.is_terminal();
        if ending
            || !matches!(
                self.pause_after_turn,
                Some((_, ErrorRecoveryCause::ToolError))
            )
        {
            // No tool-error pause reports them: each is reported on its own.
            self.report_tool_errors().await?;
        }
        let tool_error_pause = std::mem::take(&mut self.tool_errors);
        if let Some(stopping) = self.stopping {
            let _ = stopping;
            return Ok(Some(SessionEnd::Stop));
        }
        if self.task.current_state.is_terminal() {
            return Ok(Some(SessionEnd::Stop));
        }
        if let Some(Compaction::Sent { compacted }) = self.compaction {
            // The `/compact` turn ended.
            self.compaction = None;
            if compacted {
                self.add_compaction_note()?;
            } else {
                self.held_calls.clear();
                if self.pause_after_turn.is_none() {
                    self.compaction_failed(result).await?;
                }
            }
            // Its note (or the report request) goes out with the task's
            // input, after the pause the turn asked for if it did.
            if self.pause_after_turn.is_none() {
                return self.send_unsent(session).await;
            }
        }
        if let Some((reason, cause)) = self.pause_after_turn.take() {
            if matches!(cause, ErrorRecoveryCause::BudgetExceeded) {
                self.hold(reason).await?;
                return Ok(Some(SessionEnd::Wait));
            }
            if matches!(cause, ErrorRecoveryCause::Monitor) {
                // As for other agents: a pause under the pausing error
                // policies, the task returned interrupted under the others.
                self.pause_on_error(reason, cause).await?;
                self.task.monitor_pause = matches!(
                    self.task.current_state,
                    TaskState::AwaitingErrorRecoveryApproval { .. }
                );
                return Ok(Some(
                    if matches!(self.task.current_state, TaskState::Interrupted { .. }) {
                        SessionEnd::Stop
                    } else {
                        SessionEnd::Wait
                    },
                ));
            }
            if cause == ErrorRecoveryCause::ToolError && !tool_error_pause.is_empty() {
                // Every failure, the call named when it is the only one.
                let tool_call_id = match tool_error_pause.as_slice() {
                    [(_, id)] => Some(*id),
                    _ => None,
                };
                let message = tool_error_pause
                    .into_iter()
                    .map(|(message, _)| message)
                    .collect::<Vec<_>>()
                    .join("\n");
                self.pause_for_call(message, cause, tool_call_id).await?;
            } else {
                self.pause(reason, cause).await?;
            }
            return Ok(Some(SessionEnd::Wait));
        }
        let fallback_answered = std::mem::take(&mut self.fallback_turn);
        if fallback_answered {
            // The fallback answered one turn; the configured model is back.
            let model = self.task.agent_config.llm_config.model.id.clone();
            if let Err(error) = session.set_model(Some(&model)).await {
                return self.session_failed(error.to_string()).await.map(Some);
            }
        }
        if let Some(refusal) = self.refused.take() {
            match self.task.agent_config.refusal_fallback.clone() {
                // Once per refusal: a fallback that refuses too, stands.
                Some(fallback) if !fallback_answered => {
                    if let Err(error) = session.set_model(Some(&fallback.model.id)).await {
                        return self.session_failed(error.to_string()).await.map(Some);
                    }
                    self.fallback_turn = true;
                    self.emit(AgentEvent::RefusalFallback {
                        model_id: fallback.model.id.clone(),
                    })
                    .await?;
                    if let Err(error) = self.send(session, CONTINUE).await {
                        return self.session_failed(error.to_string()).await.map(Some);
                    }
                    return Ok(None);
                }
                _ => {
                    let message = refusal
                        .explanation
                        .clone()
                        .unwrap_or_else(|| "The model refused the request.".to_string());
                    self.pause_on_error(message, ErrorRecoveryCause::Refusal)
                        .await?;
                    return Ok(Some(
                        if matches!(self.task.current_state, TaskState::Interrupted { .. }) {
                            SessionEnd::Stop
                        } else {
                            SessionEnd::Wait
                        },
                    ));
                }
            }
        }
        if let Some((questions, asked)) = self.questions.take() {
            self.set_state(TaskState::AwaitingUserResponse {
                questions: questions.clone(),
            })
            .await?;
            self.emit(AgentEvent::UserQuestionsAsked {
                questions,
                asked,
                wait: true,
            })
            .await?;
            return Ok(Some(SessionEnd::Wait));
        }
        if self.compaction == Some(Compaction::Due) && !self.held_calls.is_empty() {
            // The turn was interrupted for the compaction.
            return self.compact_now(session).await;
        }
        if result.is_error {
            let message = result
                .result
                .clone()
                .unwrap_or_else(|| format!("Claude Code ended the turn with {}", result.subtype));
            let api_error_status = result
                .raw
                .get("api_error_status")
                .and_then(Value::as_u64)
                .and_then(|status| u16::try_from(status).ok());
            let kind = arko_ai::ErrorKind::from_claude_code(
                api_error.as_deref(),
                api_error_status,
                result.raw.get("terminal_reason").and_then(Value::as_str),
                &message,
            );
            self.error_kind = Some(kind).filter(|kind| *kind != arko_ai::ErrorKind::Unknown);
            // Failures retrying doesn't fix: Claude Code isn't signed in, or
            // its login expired or was revoked (a new login is needed); its
            // account can't be billed (a credit or spend wall); or its
            // organization or plan may not use what was asked.
            let lasting = match kind {
                arko_ai::ErrorKind::Authentication => Some("could not authenticate"),
                arko_ai::ErrorKind::Billing => Some("can't bill its account"),
                arko_ai::ErrorKind::Permission => Some("isn't allowed to make the request"),
                _ => None,
            };
            if let Some(what) = lasting {
                self.pause_on_error(
                    format!("Claude Code {what}: {message}"),
                    ErrorRecoveryCause::NonRetryable,
                )
                .await?;
                return Ok(Some(
                    if matches!(self.task.current_state, TaskState::Interrupted { .. }) {
                        SessionEnd::Stop
                    } else {
                        SessionEnd::Wait
                    },
                ));
            }
            if kind == arko_ai::ErrorKind::ContextOverflow
                && self.compacts()
                && !self.handover_due()
            {
                // Over the model's input limit before the compaction: Claude
                // Code's `/compact` tries now (and a handover follows if it
                // can't).
                log::warn!(
                    "Task {}: The conversation went over the model's input limit, so it is compacted now.",
                    self.task.id
                );
                self.error_kind = None;
                return self.compact_now(session).await;
            }
            if kind == arko_ai::ErrorKind::ContextOverflow
                && self.hand_over_without_the_step(&message).await?
            {
                return Ok(Some(SessionEnd::Restart));
            }
            return self.session_failed(message).await.map(Some);
        }
        // The final reply of an agent whose reply in text ends the task: a
        // turn that ends with text, other than those above (Claude Code was
        // told to end them, or arko ended them) and the handover report. A
        // compaction or a report request the model hasn't seen yet waits,
        // as it waits for a `complete_task` call.
        if self.task.agent_config.tool_config.ends_task_with_text()
            && !(self.handover_due() && self.report_delivered)
            && self.ended_on_text()
            && let Some(reply) = result
                .result
                .as_ref()
                .filter(|text| !text.trim().is_empty())
        {
            self.set_state(TaskState::Completed {
                summary: reply.trim().to_string(),
            })
            .await?;
            return Ok(Some(SessionEnd::Stop));
        }
        if self.compaction == Some(Compaction::Due) {
            return self.compact_now(session).await;
        }
        if self.handover_due() {
            if !self.report_delivered {
                self.report_delivered = true;
                let request = self.report_request();
                if let Err(error) = self.send(session, &request).await {
                    return self.session_failed(error.to_string()).await.map(Some);
                }
                return Ok(None);
            }
            let report = result.result.clone().unwrap_or_default();
            if report.trim().is_empty() {
                if let Err(error) = self.send(session, NO_HANDOVER_REPORT).await {
                    return self.session_failed(error.to_string()).await.map(Some);
                }
                return Ok(None);
            }
            self.complete_handover(report).await?;
            return Ok(Some(SessionEnd::Restart));
        }
        let ends_with_text = self.task.agent_config.tool_config.ends_task_with_text();
        self.open_turns += 1;
        if self.open_turns > MAX_OPEN_TURNS {
            let without = if ends_with_text {
                "a reply"
            } else {
                "calling complete_task"
            };
            self.set_state(TaskState::Interrupted {
                reason: format!("{MAX_OPEN_TURNS} turns in a row ended without {without}."),
            })
            .await?;
            return Ok(Some(SessionEnd::Stop));
        }
        let mut note = if ends_with_text {
            OPEN_TURN_NO_REPLY_NOTE
        } else {
            OPEN_TURN_NOTE
        }
        .to_string();
        if self.task.agent_config.tool_config.text_reply_as_message
            && let Some(text) = result
                .result
                .as_ref()
                .filter(|text| !text.trim().is_empty())
        {
            self.emit(AgentEvent::MessageToUser {
                message: text.clone(),
                tool_call_id: None,
            })
            .await?;
            note = OPEN_TURN_MESSAGE_NOTE.to_string();
        }
        let mut images = Vec::new();
        if let Some((context, _, context_images)) = self.take_context(true) {
            note.push_str("\n\n");
            note.push_str(&context);
            images = context_images;
        }
        // The task records the message as it is sent.
        let sent = self.send_with_images(session, &note, images).await;
        self.keep_taken_marks();
        if let Err(error) = sent {
            return self.session_failed(error.to_string()).await.map(Some);
        }
        self.emit_context_receipts().await?;
        Ok(None)
    }

    /// Whether the turn's last response called no tool: the conversation
    /// ends on the model's text (Claude Code's turn result is its text).
    fn ended_on_text(&self) -> bool {
        self.task
            .conversation
            .messages
            .last()
            .is_some_and(|message| {
                message.role == ai::Role::Assistant && message.tool_calls.is_empty()
            })
    }

    /// The session failed (it exited, or a turn ended in an error Claude
    /// Code's own retries didn't recover from): the error policy decides.
    async fn session_failed(&mut self, message: String) -> Result<SessionEnd, ArkoAgentsError> {
        if self.stopping.is_some() || self.task.current_state.is_terminal() {
            return Ok(SessionEnd::Stop);
        }
        // A plan limit Claude Code reported reached: the pause waits for its
        // reset (a workflow's automatic recovery waits until then).
        if let Some(Some(resets_at)) = self.limit_reached
            && let Some(wait) = self.usage_limit_wait(resets_at)
        {
            self.error_kind = None;
            return self.wait_for_reset(message, resets_at, wait).await;
        }
        let cause = match self.limit_reached {
            Some(resets_at) => {
                let resets_at_unix_ms = resets_at.map(|seconds| seconds.saturating_mul(1000));
                self.emit(AgentEvent::UsageLimitReached {
                    resets_at_unix_ms,
                    waiting_until_unix_ms: None,
                    error_message: message.clone(),
                })
                .await?;
                ErrorRecoveryCause::UsageLimit { resets_at_unix_ms }
            }
            None => ErrorRecoveryCause::RetriesExhausted {
                retry_after_ms: None,
            },
        };
        self.pause_on_error(message, cause).await?;
        Ok(
            if matches!(self.task.current_state, TaskState::Interrupted { .. }) {
                SessionEnd::Stop
            } else {
                SessionEnd::Wait
            },
        )
    }

    /// Under [`UsageLimitPolicy::WaitUntilReset`](crate::agent_config::UsageLimitPolicy::WaitUntilReset),
    /// the wait for a limit that resets at `resets_at` (Unix seconds): until a
    /// little past the reset, when the whole wait fits `max_wait_ms`.
    fn usage_limit_wait(&self, resets_at: u64) -> Option<Duration> {
        let retry = &self.task.agent_config.llm_retry_config;
        let crate::agent_config::UsageLimitPolicy::WaitUntilReset { max_wait_ms } =
            retry.usage_limit_policy
        else {
            return None;
        };
        let now = chrono::Utc::now().timestamp().max(0) as u64;
        let reset_in = Duration::from_secs(resets_at.saturating_sub(now));
        let longest = reset_in + USAGE_LIMIT_RESET_MARGIN + USAGE_LIMIT_RESET_JITTER;
        if longest > Duration::from_millis(max_wait_ms) {
            return None;
        }
        // A little past the reset, spread so that agents sharing a plan
        // don't all continue at once.
        let jitter = if retry.jitter {
            ai::retry::random_sample() % (USAGE_LIMIT_RESET_JITTER.as_millis() as u64 + 1)
        } else {
            0
        };
        Some(reset_in + USAGE_LIMIT_RESET_MARGIN + Duration::from_millis(jitter))
    }

    /// Waits out a reached plan limit, handling commands meanwhile (a
    /// cancellation or suspension ends the wait), then continues the task in
    /// its session.
    async fn wait_for_reset(
        &mut self,
        message: String,
        resets_at: u64,
        wait: Duration,
    ) -> Result<SessionEnd, ArkoAgentsError> {
        let until = tokio::time::Instant::now() + wait;
        let waiting_until_unix_ms = (chrono::Utc::now().timestamp_millis().max(0) as u64)
            .saturating_add(wait.as_millis() as u64);
        self.emit(AgentEvent::UsageLimitReached {
            resets_at_unix_ms: Some(resets_at.saturating_mul(1000)),
            waiting_until_unix_ms: Some(waiting_until_unix_ms),
            error_message: message,
        })
        .await?;
        let sleep = tokio::time::sleep_until(until);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                command = self.commands.recv() => match command {
                    Some(UserCommand::InjectContext { content }) => self.queue_context(content, true),
                    Some(UserCommand::RequestHandover) => self.note_handover_request().await?,
                    Some(UserCommand::CancelTask) => {
                        self.set_state(TaskState::Cancelled {
                            reason: crate::task::CANCELLED_BY_USER.into(),
                        })
                        .await?;
                        return Ok(SessionEnd::Stop);
                    }
                    Some(UserCommand::SuspendTask) => return Ok(SessionEnd::Stop),
                    None => {
                        self.set_state(TaskState::Interrupted {
                            reason: "Command channel disconnected.".into(),
                        })
                        .await?;
                        return Ok(SessionEnd::Stop);
                    }
                    Some(_) => {}
                },
                _ = &mut sleep => break,
            }
        }
        self.limit_reached = None;
        // The session resumes, and the model is asked to continue.
        Ok(SessionEnd::Wait)
    }

    /// Pauses for an error-recovery decision under a pausing error policy,
    /// or returns the task interrupted under the others.
    async fn pause_on_error(
        &mut self,
        message: String,
        cause: ErrorRecoveryCause,
    ) -> Result<(), ArkoAgentsError> {
        log::warn!("Task {}: {message}", self.task.id);
        let pausing = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::PauseAndAwaitApproval
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if pausing {
            self.pause(message, cause).await
        } else {
            // Reported before the task ends interrupted: a workflow stops
            // reading its events at that state.
            let error_kind = self.error_kind.take();
            self.emit(AgentEvent::ErrorEncountered {
                error_message: message.clone(),
                requires_approval_to_retry: false,
                tool_call_id: None,
                error_kind,
            })
            .await?;
            self.set_state(TaskState::Interrupted { reason: message })
                .await
        }
    }

    /// Why the host holds the task's next request, if it does (see
    /// [`TurnContextSource::hold_request`](super::TurnContextSource::hold_request)).
    fn held_request(&self) -> Option<String> {
        self.turn_context
            .as_ref()
            .and_then(|source| source.hold_request(self.task.id))
    }

    /// Pauses the task for the host, which holds its next request: an
    /// error-recovery pause whatever the error policy, whose approval adds
    /// no error note.
    async fn hold(&mut self, reason: String) -> Result<(), ArkoAgentsError> {
        log::info!(
            "Task {}: the host holds the next request: {reason}",
            self.task.id
        );
        self.task.monitor_pause = true;
        self.pause(reason, ErrorRecoveryCause::BudgetExceeded).await
    }

    /// Pauses for an error-recovery decision, reported as arko's own
    /// runtime reports one: the state change that gives the pause's cause,
    /// then the error a host answers (a workflow decides the pause once it
    /// has both).
    async fn pause(
        &mut self,
        message: String,
        cause: ErrorRecoveryCause,
    ) -> Result<(), ArkoAgentsError> {
        self.pause_for_call(message, cause, None).await
    }

    /// [`pause`](Self::pause), naming the tool call the error is about. The
    /// failed turn's error kind, if one was recorded, goes with the pause.
    async fn pause_for_call(
        &mut self,
        message: String,
        cause: ErrorRecoveryCause,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let error_kind = self.error_kind.take();
        self.set_state(TaskState::AwaitingErrorRecoveryApproval {
            error_message: message.clone(),
            cause,
            error_kind,
        })
        .await?;
        self.emit(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: true,
            tool_call_id,
            error_kind,
        })
        .await
    }

    /// Reports the tool errors no tool-error pause reports, each as the
    /// error of its call.
    async fn report_tool_errors(&mut self) -> Result<(), ArkoAgentsError> {
        for (message, id) in std::mem::take(&mut self.tool_errors) {
            self.emit(AgentEvent::ErrorEncountered {
                error_message: message,
                requires_approval_to_retry: false,
                tool_call_id: Some(id),
                error_kind: None,
            })
            .await?;
        }
        Ok(())
    }

    /// Sends `/compact` into the session between turns.
    async fn compact_now(
        &mut self,
        session: &cc::ClaudeSession,
    ) -> Result<Option<SessionEnd>, ArkoAgentsError> {
        self.compaction = Some(Compaction::Sent { compacted: false });
        if let Err(error) = session.send(COMPACT_COMMAND).await {
            return self.session_failed(error.to_string()).await.map(Some);
        }
        Ok(None)
    }

    /// Sends the session the task's input it hasn't received, as a turn.
    async fn send_unsent(
        &mut self,
        session: &cc::ClaudeSession,
    ) -> Result<Option<SessionEnd>, ArkoAgentsError> {
        let sent = match self.unsent_input()? {
            Value::String(text) => session.send(text).await,
            content => session.send_content(content).await,
        };
        match sent {
            Ok(uuid) => self.record_sent(uuid),
            Err(error) => return self.session_failed(error.to_string()).await.map(Some),
        }
        self.emit_context_receipts().await?;
        Ok(None)
    }

    /// Claude Code compacted the session's conversation (`compact_boundary`),
    /// at the runtime's request or on its own: its summary replaced the
    /// earlier messages, which the task's conversation keeps. What the
    /// runtime keeps about the conversation starts over (as after a
    /// handover), and the handover hook's context waits for the agent's next
    /// input.
    async fn compacted(&mut self, raw: &Value) -> Result<(), ArkoAgentsError> {
        let metadata = &raw["compact_metadata"];
        let tokens = |field: &str| metadata[field].as_u64().map(|tokens| tokens as usize);
        let input_tokens = tokens("pre_tokens").unwrap_or_else(|| self.latest_size());
        let compacted_tokens = tokens("post_tokens");
        if let Some(Compaction::Sent { compacted }) = &mut self.compaction {
            *compacted = true;
        }
        // The requests so far belong to the conversation before.
        self.flush_usage().await?;
        self.task.conversation_usage_start = self.task.usage_history.len();
        self.task.conversation_model_call_start = self.task.model_call_usage_history.len();
        self.context_size = None;
        self.first_context_size = None;
        self.cache_warned = false;
        self.cache_pause_episode = false;
        self.handover_skip_warned = false;
        // The messages before the summary aren't the conversation's now:
        // the session can't be cut back to them.
        self.last_entry = None;
        self.task.claude_code_entries.clear();
        self.answered = None;
        self.step_results = false;
        for name in self.registry.list_tool_names() {
            if let Some(tool) = self.registry.get(&name) {
                tool.reset_context(self.task.id);
            }
        }
        let context = self
            .handover_hook
            .as_ref()
            .and_then(|hook| hook.on_compaction(&self.task));
        let questions = crate::questions::open_section(&crate::handover::open_questions(
            self.handover_hook.as_deref(),
            &self.task,
        ));
        let context = match (context, questions) {
            (Some(context), Some(questions)) => Some(format!("{context}\n\n{questions}")),
            (context, questions) => context.or(questions),
        };
        if let Some(context) = context {
            self.queue_context(context, false);
        }
        log::info!(
            "Task {}: Claude Code compacted the conversation ({input_tokens} input tokens before, {} after). The task's record keeps the full conversation.",
            self.task.id,
            compacted_tokens.map_or("unknown".to_string(), |tokens| tokens.to_string())
        );
        self.report(false).await;
        self.emit(AgentEvent::ConversationCompacted {
            input_tokens,
            compacted_tokens,
        })
        .await
    }

    /// After a compaction the runtime asked for: tells the model of the calls
    /// it held, and lets it continue when no other input (a user's message)
    /// does.
    fn add_compaction_note(&mut self) -> Result<(), ArkoAgentsError> {
        let mut parts = Vec::new();
        if !self.held_calls.is_empty() {
            parts.push(format!("{HELD_CALLS}\n{}", self.held_calls.join("\n")));
            self.held_calls.clear();
        }
        let synced = self
            .task
            .claude_code_synced
            .min(self.task.conversation.messages.len());
        let unsent = self.task.conversation.messages[synced..]
            .iter()
            .any(|message| message.role == ai::Role::User);
        if !unsent {
            parts.push(COMPACTED.to_string());
        }
        if parts.is_empty() {
            return Ok(());
        }
        self.add_user_message(&parts.join("\n\n"))
    }

    /// Claude Code didn't compact the conversation (`/compact` failed, or
    /// isn't a command of this Claude Code): the agent hands over instead.
    /// The report request goes out with the task's input that waited for
    /// the compaction, so the report covers it. In a conversation already
    /// over the model's input limit it fails as well, and the step that
    /// took it there is left out ([`hand_over_without_the_step`](Self::hand_over_without_the_step)).
    async fn compaction_failed(
        &mut self,
        result: &cc::ResultMessage,
    ) -> Result<(), ArkoAgentsError> {
        log::warn!(
            "Task {}: Claude Code did not compact the conversation ({}: {}), so the agent hands over instead.",
            self.task.id,
            result.subtype,
            result.result.as_deref().unwrap_or_default()
        );
        let input_tokens = self.latest_size();
        self.task.handover_requested = Some(HandoverRequest {
            report_requested: true,
            ..HandoverRequest::new(input_tokens)
        });
        self.report_delivered = true;
        let request = self.report_request();
        self.add_user_message(&request)?;
        self.emit(AgentEvent::HandoverRequested { input_tokens })
            .await
    }

    /// A turn went over the model's input limit, for an agent that hands
    /// over (it has `handover`, or the host set a threshold), as the native
    /// runtime's `hand_over_without_the_step` does: the step that took the
    /// conversation there (the model's reply, its results, and the runtime's
    /// messages after them, such as a report request) is left out, and the
    /// report is asked for from the conversation as the last request Claude
    /// Code answered had it, with [`STEP_LEFT_OUT`](crate::handover::STEP_LEFT_OUT)
    /// before the request. The next session resumes the session at the
    /// message that ended that request (Claude Code's `--resume-session-at`),
    /// and the context updates delivered since go again with the request.
    /// Returns whether it does: not without a handover, without tool results
    /// since the last answered request (a message took it over, such as the
    /// report request itself), or once a step was left out for this
    /// handover, so a report request that fails as well ends the run.
    async fn hand_over_without_the_step(&mut self, error: &str) -> Result<bool, ArkoAgentsError> {
        let hands_over =
            self.task.agent_config.handover.is_some() || self.handover_threshold.get().is_some();
        let Some(answered) = self.answered.take().filter(|answered| {
            hands_over
                && self.step_results
                && !self.step_left_out
                && answered.messages <= self.task.conversation.messages.len()
        }) else {
            return Ok(false);
        };
        let start = answered.messages;
        let in_step = |message: &ai::Message| {
            message.role == ai::Role::Assistant
                || message
                    .content
                    .iter()
                    .any(|content| content.as_tool_result().is_some())
        };
        let added = self.task.conversation.messages.split_off(start);
        let runtime = &self.task.runtime_messages;
        let kept: Vec<ai::Message> = added
            .into_iter()
            .enumerate()
            .filter(|(offset, message)| !in_step(message) && !runtime.contains(&(start + offset)))
            .map(|(_, message)| message)
            .collect();
        self.task.conversation.messages.extend(kept);
        // The messages kept after `start` are the host's, not the runtime's.
        self.task.runtime_messages.retain(|index| *index < start);
        self.task
            .runtime_text_messages
            .retain(|index| *index < start);
        // The resumed session has what came before `start`; the rest goes
        // with the report request.
        self.task.claude_code_synced = start;
        self.task
            .claude_code_entries
            .retain(|kept, _| *kept < start);
        self.resume_at = Some(answered.uuid);
        self.step_left_out = true;
        self.step_results = false;
        self.last_entry = None;
        self.error_kind = None;
        log::warn!(
            "Task {}: the conversation went over the model's input limit, so its latest step was left out and the agent hands over.",
            self.task.id
        );
        self.emit(AgentEvent::ErrorEncountered {
            error_message: format!(
                "The conversation went over the model's input limit, so its latest step was left out and the agent hands over from the conversation before it: {error}"
            ),
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: Some(ai::ErrorKind::ContextOverflow),
        })
        .await?;
        self.report_delivered = true;
        let request = format!(
            "{}\n\n{}",
            crate::handover::STEP_LEFT_OUT,
            self.report_request()
        );
        self.add_user_message(&request)?;
        if self.task.handover_requested.is_none() {
            let input_tokens = self.latest_size();
            self.task.handover_requested = Some(HandoverRequest {
                report_requested: true,
                ..HandoverRequest::new(input_tokens)
            });
            self.emit(AgentEvent::HandoverRequested { input_tokens })
                .await?;
        }
        Ok(true)
    }

    /// The report arrived: a fresh session starts from the seed and the
    /// report, in the same task.
    async fn complete_handover(&mut self, report: String) -> Result<(), ArkoAgentsError> {
        let report = report.trim().to_string();
        let chat = self.is_chat();
        let seed = match &self.handover_hook {
            Some(hook) => hook.on_handover(&self.task, &report),
            None if chat => HandoverSeed::for_chat(&self.task),
            None => HandoverSeed::for_task(&self.task),
        };
        let seed =
            crate::handover::with_open_questions(seed, self.handover_hook.as_deref(), &self.task);
        // A chat's fresh conversation answers the user's latest input, with
        // the images it held, and a task's restates the task, with its images.
        let images = crate::handover::seed_images(&self.task, chat);
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
        let builder = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing, None);
        let first = builder.format_initial_task_prompt(&self.task, &seed.message(&report))?;
        let mut content = vec![ai::Content::Text(first)];
        content.extend(images);
        self.task
            .conversation
            .add_user_message_with_contents(content);
        self.task.increment_user_message_count();
        // The next session is a new Claude Code conversation, which records
        // the current prompt.
        self.task.claude_code_session = None;
        self.task.claude_code_directory = None;
        self.task.claude_code_synced = 0;
        self.task.claude_code_prompt_changed = false;
        self.task.claude_code_entries.clear();
        self.task.claude_code_resume_at = None;
        self.task.claude_code_fork_of = None;
        for name in self.registry.list_tool_names() {
            if let Some(tool) = self.registry.get(&name) {
                tool.reset_context(self.task.id);
            }
        }
        self.calls.clear();
        // The session's calls ended with it.
        self.task.running_tool_calls.clear();
        self.pending_usage.clear();
        self.streamed.clear();
        self.streamed_text.clear();
        self.cache_warned = false;
        self.cache_pause_episode = false;
        self.context_size = None;
        self.first_context_size = None;
        self.report_delivered = false;
        self.open_turns = 0;
        self.handover_skip_warned = false;
        self.last_entry = None;
        self.answered = None;
        self.step_results = false;
        self.step_left_out = false;
        self.resume_at = None;
        log::info!(
            "Task {}: Conversation handed over ({replaced_messages} messages replaced).",
            self.task.id
        );
        let handovers = self.task.handovers.len();
        self.report(false).await;
        self.emit(AgentEvent::ConversationHandedOver {
            report,
            input_tokens,
            handovers,
        })
        .await
    }
}

/// Claude Code's `rate_limit_info` as plan usage of
/// [`SubscriptionPlan::ClaudeCode`](ai::SubscriptionPlan::ClaudeCode): the
/// five-hour window is the primary one and the seven-day window the
/// secondary, as in its `unifiedWindows`. An event without them reports its
/// one window.
fn plan_usage_of(info: &Value) -> ai::PlanUsage {
    let window = |value: &Value, minutes: u64| {
        value
            .get("utilization")
            .and_then(Value::as_f64)
            .map(|utilization| ai::PlanUsageWindow {
                used_percent: utilization * 100.0,
                window_minutes: Some(minutes),
                resets_at: value.get("resetsAt").and_then(Value::as_u64),
                reset_after_seconds: None,
            })
    };
    let limit_type = info["rateLimitType"].as_str().map(str::to_string);
    let (primary, secondary) = match info.get("unifiedWindows") {
        Some(windows) => (
            window(&windows["five_hour"], 300),
            window(&windows["seven_day"], 10_080),
        ),
        None => match limit_type.as_deref() {
            Some("five_hour") => (window(info, 300), None),
            // The usage credits' limit, not a window of the plan.
            Some("overage") | None => (None, None),
            Some(_) => (None, window(info, 10_080)),
        },
    };
    ai::PlanUsage {
        plan: Some(ai::SubscriptionPlan::ClaudeCode),
        plan_type: None,
        active_limit: limit_type,
        primary,
        secondary,
        credits: None,
        extra_usage: extra_usage_of(info),
    }
}

/// What a `rate_limit_info` says about extra usage (`overage*` fields).
/// `None` when it says nothing. Its `utilization` is the credits' limit used
/// when the limit it reports is that of the credits (`rateLimitType`
/// `overage`).
fn extra_usage_of(info: &Value) -> Option<ai::ExtraUsage> {
    let flag = |name: &str| info.get(name).and_then(Value::as_bool);
    let text = |name: &str| info.get(name).and_then(Value::as_str).map(str::to_string);
    // For the credits' own limit, the event's reset and utilization are theirs.
    let credits_limit = info["rateLimitType"] == "overage";
    let extra = ai::ExtraUsage {
        enabled: flag("overageEnabled"),
        in_use: flag("isUsingOverage") == Some(true) || flag("overageInUse") == Some(true),
        status: text("overageStatus"),
        disabled_reason: text("overageDisabledReason"),
        resets_at: info
            .get("overageResetsAt")
            .and_then(Value::as_u64)
            .or_else(|| credits_limit.then(|| info["resetsAt"].as_u64()).flatten()),
        utilization: credits_limit
            .then(|| {
                info["utilization"]
                    .as_f64()
                    .map(|fraction| fraction * 100.0)
            })
            .flatten(),
        ..Default::default()
    };
    (extra != ai::ExtraUsage::default()).then_some(extra)
}

/// The extra usage in Claude Code's answer to `get_usage`
/// (`rate_limits.extra_usage`), whose amounts are in minor units of its
/// currency (cents for USD). `None` when the answer has none.
fn extra_usage_read(answer: &Value) -> Option<ai::ExtraUsage> {
    let extra = answer["rate_limits"]["extra_usage"].as_object()?;
    let currency = extra
        .get("currency")
        .and_then(Value::as_str)
        .map(str::to_string);
    // Else as Claude Code shows amounts, in whole units for these currencies.
    let decimal_places = extra
        .get("decimal_places")
        .and_then(Value::as_u64)
        .and_then(|places| u32::try_from(places).ok())
        .or_else(|| {
            let currency = currency.as_deref()?.to_uppercase();
            Some(if ["JPY", "KRW", "VND"].contains(&currency.as_str()) {
                0
            } else {
                2
            })
        });
    Some(ai::ExtraUsage {
        decimal_places,
        enabled: extra.get("is_enabled").and_then(Value::as_bool),
        disabled_reason: extra
            .get("disabled_reason")
            .and_then(Value::as_str)
            .map(str::to_string),
        monthly_limit: extra.get("monthly_limit").and_then(Value::as_f64),
        used_credits: extra.get("used_credits").and_then(Value::as_f64),
        utilization: extra.get("utilization").and_then(Value::as_f64),
        currency,
        ..Default::default()
    })
}

/// A refusal Claude Code reported (`model_refusal`,
/// `model_refusal_fallback`): the API's refusal category and explanation.
/// `details` are the refused response's `stop_details`, for what the message
/// leaves out.
fn refusal_of(raw: &Value, details: Option<Value>) -> ai::Refusal {
    let details = details.unwrap_or_default();
    let field = |name: &str, detail: &str| {
        raw[name]
            .as_str()
            .or_else(|| details[detail].as_str())
            .map(str::to_string)
    };
    let mut refusal = ai::Refusal::new(ai::RefusalSource::SafetyClassifier);
    refusal.category = field("api_refusal_category", "category");
    refusal.explanation = field("api_refusal_explanation", "explanation");
    refusal.provider_details = raw.clone();
    refusal
}

/// How much of a recovered call's result the recovery note quotes.
const RECOVERED_RESULT_CHARS: usize = 2000;

/// `text`, cut to [`RECOVERED_RESULT_CHARS`].
fn bounded(text: &str) -> String {
    match text.char_indices().nth(RECOVERED_RESULT_CHARS) {
        Some((end, _)) => format!("{}… [cut]", &text[..end]),
        None => text.to_string(),
    }
}

/// A tool result's text, from its content (a string or text blocks).
fn result_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        Value::Object(object) if object.contains_key("content") => result_text(&object["content"]),
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("image") => {
            String::new()
        }
        other => other.to_string(),
    }
}

/// A file's size as `read_file` gives it: KB under a megabyte, else MB.
fn file_size(bytes: usize) -> String {
    if bytes < 1024 * 1024 {
        format!("{} KB", bytes.div_ceil(1024))
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Images in a reported result, either MCP content or Claude's image
/// source blocks. Arko tools retain their originals in `ToolHost` too.
fn result_images(content: &Value) -> Vec<crate::tool::ToolImage> {
    let blocks = match content {
        Value::Array(blocks) => blocks,
        Value::Object(object) if object.contains_key("content") => {
            return result_images(&object["content"]);
        }
        _ => return Vec::new(),
    };
    blocks
        .iter()
        .filter(|block| block["type"] == "image")
        .filter_map(|block| {
            let source = block.get("source").unwrap_or(block);
            let image = if source["type"] == "url" {
                ai::Image::new_url(source["url"].as_str()?)
            } else {
                ai::Image::new_base64(
                    source["data"].as_str()?,
                    source["media_type"]
                        .as_str()
                        .or_else(|| source["mimeType"].as_str())?,
                )
            };
            Some(crate::tool::ToolImage {
                description: String::new(),
                image,
            })
        })
        .collect()
}

async fn sleep_until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending::<()>().await,
    }
}

#[cfg(test)]
#[path = "claude_code_tests.rs"]
mod tests;

/// Raises `marks` to `taken`'s where they are higher.
fn raise(marks: &mut BTreeMap<String, u64>, taken: &BTreeMap<String, u64>) {
    for (key, value) in taken {
        let mark = marks.entry(key.clone()).or_insert(0);
        *mark = (*mark).max(*value);
    }
}
