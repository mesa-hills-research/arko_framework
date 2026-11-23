//! The connection to `agy acp`, which speaks the Agent Client Protocol (ACP)
//! version 1, JSON-RPC 2.0 with one JSON object per line on stdio.
//!
//! [`AcpClient::start`] runs the CLI in a process tree of its own and
//! completes `initialize`. Requests wait for their replies. Everything else
//! agy sends (session updates, the results of prompts, other notifications,
//! and agy's permission requests with the answers they got) arrives in order
//! through [`AcpClient::next_event`]. A prompt's result is an event too, so
//! it comes after the updates of its turn. A prompt sent while a turn runs
//! joins that turn, and the earlier prompt's result then comes at once.
//!
//! agy asks before each MCP tool call. A [`PermissionPolicy`] decides, and
//! the call is either allowed or its turn is cancelled. Requests for files
//! and terminals get `-32601`, since the client offers neither.
//!
//! Every launch sets [`NO_AUTO_UPDATE`], because the updater replaces the
//! shared binary while sessions run.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::process_tree::ProcessTree;

/// Turns agy's self-updates off. Only `true` works (`1` is ignored).
pub const NO_AUTO_UPDATE: (&str, &str) = ("AGY_CLI_DISABLE_AUTO_UPDATE", "true");

/// The largest line agy may send. A longer one ends the connection.
pub const MAX_FRAME: usize = 16 * 1024 * 1024;

/// The log target of the raw ACP traffic, one line per message (`->` to agy,
/// `<-` from it), at trace level. Enable it with
/// `RUST_LOG=arko_ai::antigravity::wire=trace`.
pub const WIRE_LOG: &str = "arko_ai::antigravity::wire";

/// The prefix of the tools an MCP server named `arko` offers, as agy names
/// them to the model.
pub const ARKO_TOOL_PREFIX: &str = "mcp_arko_";

const PROTOCOL_VERSION: u64 = 1;
const INITIALIZE_TIMEOUT: Duration = Duration::from_secs(60);
/// `session/new` starts the profile's MCP servers first.
const SESSION_TIMEOUT: Duration = Duration::from_secs(90);
const CONFIG_TIMEOUT: Duration = Duration::from_secs(30);
/// How long `close` waits for agy to exit after its input closes.
const EXIT_GRACE: Duration = Duration::from_secs(10);
/// How much of agy's stderr to keep for error messages.
const STDERR_TAIL: usize = 16 * 1024;
const METHOD_NOT_FOUND: i64 = -32601;

/// How to run `agy acp`.
#[derive(Clone)]
pub struct AcpOptions {
    /// The `agy` executable.
    pub program: PathBuf,
    /// Arguments before agy's own, such as a script run by an interpreter.
    pub prefix_args: Vec<String>,
    /// agy's settings directory (`--gemini-dir`). A new one keeps the user's
    /// sign-in, plan and models.
    pub profile_dir: PathBuf,
    /// The custom agent the profile defines (`--agent`).
    pub agent: Option<String>,
    /// Where agy writes its log (`--log-file`).
    pub log_file: Option<PathBuf>,
    /// The directory agy runs in.
    pub working_directory: PathBuf,
    /// Variables set on top of the inherited environment. [`NO_AUTO_UPDATE`]
    /// is set after them.
    pub env: Vec<(String, String)>,
    /// The name and version `initialize` sends (`clientInfo`).
    pub client_info: ClientInfo,
}

/// The client's name and version, as `initialize` sends them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

impl AcpOptions {
    /// Options with no agent, no log file and no extra environment.
    pub fn new(
        program: impl Into<PathBuf>,
        profile_dir: impl Into<PathBuf>,
        working_directory: impl Into<PathBuf>,
    ) -> Self {
        Self {
            program: program.into(),
            prefix_args: Vec::new(),
            profile_dir: profile_dir.into(),
            agent: None,
            log_file: None,
            working_directory: working_directory.into(),
            env: Vec::new(),
            client_info: ClientInfo {
                name: "arko_ai".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        }
    }

    /// agy's arguments, after [`prefix_args`](Self::prefix_args).
    pub fn arguments(&self) -> Vec<OsString> {
        let mut arguments: Vec<OsString> = [
            "acp",
            "--stdio",
            "--include-default-customizations=false",
            "--disable-telemetry",
            "--gemini-dir",
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        arguments.push(self.profile_dir.clone().into_os_string());
        if let Some(agent) = &self.agent {
            arguments.push("--agent".into());
            arguments.push(agent.into());
        }
        if let Some(log_file) = &self.log_file {
            arguments.push("--log-file".into());
            arguments.push(log_file.clone().into_os_string());
        }
        arguments
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AcpError {
    #[error("Couldn't start `{program}`. {source}")]
    Spawn {
        program: String,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("Antigravity exited.{}", if .stderr.is_empty() { String::new() } else { format!(" {}", .stderr) })]
    Exited { stderr: String },
    #[error("Antigravity refused `{method}`. {error}")]
    Rpc { method: String, error: RpcError },
    #[error("Antigravity didn't answer `{0}` in time.")]
    Timeout(String),
    #[error("Antigravity sent a message over 16 MiB.")]
    FrameTooLarge,
    #[error("Antigravity sent an unexpected reply. {0}")]
    Protocol(String),
}

impl AcpError {
    /// Whether the CLI refused to open a session without a sign-in.
    pub fn requires_sign_in(&self) -> bool {
        matches!(self, Self::Rpc { method, error }
            if matches!(method.as_str(), "session/new" | "session/resume")
                && error.code == -32000 && error.message == "Authentication required")
    }

    /// Whether the CLI could not read this saved session from the profile.
    pub fn cannot_read_session(&self, id: &str) -> bool {
        matches!(self, Self::Rpc { method, error }
            if method == "session/resume" && error.code == -32603
                && error.message == format!("Failed to read session {id}; retry, or load it with session/load"))
    }
}

/// A JSON-RPC error.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    fn parse(error: &Value) -> Self {
        Self {
            code: error
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            message: string_at(error, "message").unwrap_or_default(),
            data: error.get("data").filter(|data| !data.is_null()).cloned(),
        }
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

/// What agy sends besides the replies to requests, in the order it sends it.
#[derive(Debug, Clone, PartialEq)]
pub enum AcpEvent {
    /// A `session/update` notification.
    Update {
        session_id: String,
        update: SessionUpdate,
    },
    /// The reply to the `session/prompt` that [`AcpClient::prompt`] sent as
    /// `request`.
    PromptFinished {
        request: u64,
        outcome: Result<PromptResult, RpcError>,
    },
    /// A permission request agy asked, and the answer it got. A
    /// [`PermissionDecision::Cancel`] has cancelled the turn.
    Permission {
        request: PermissionRequest,
        decision: PermissionDecision,
    },
    /// Any other notification.
    Notification { method: String, params: Value },
}

/// One `session/update`.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionUpdate {
    /// A chunk of the reply's text (`agent_message_chunk`).
    AgentMessage { text: String },
    /// A chunk of the model's reasoning (`agent_thought_chunk`).
    AgentThought { text: String },
    /// A tool call starts (`tool_call`).
    ToolCall(ToolCall),
    /// A tool call changes (`tool_call_update`).
    ToolCallUpdate(ToolCall),
    /// How much of the context window the conversation uses
    /// (`usage_update`), in tokens.
    Usage { used: u64, size: u64 },
    /// Any other update, or one of the above in a shape it doesn't have
    /// (a chunk that isn't text).
    Other { kind: String, update: Value },
}

impl SessionUpdate {
    fn parse(update: Value) -> Self {
        let kind = string_at(&update, "sessionUpdate").unwrap_or_default();
        let parsed = match kind.as_str() {
            "agent_message_chunk" => chunk_text(&update).map(|text| Self::AgentMessage { text }),
            "agent_thought_chunk" => chunk_text(&update).map(|text| Self::AgentThought { text }),
            "tool_call" => ToolCall::parse(&update).map(Self::ToolCall),
            "tool_call_update" => ToolCall::parse(&update).map(Self::ToolCallUpdate),
            "usage_update" => match (
                update.get("used").and_then(Value::as_u64),
                update.get("size").and_then(Value::as_u64),
            ) {
                (Some(used), Some(size)) => Some(Self::Usage { used, size }),
                _ => None,
            },
            _ => None,
        };
        parsed.unwrap_or(Self::Other { kind, update })
    }
}

fn chunk_text(update: &Value) -> Option<String> {
    let content = update.get("content")?;
    if content.get("type").and_then(Value::as_str) != Some("text") {
        return None;
    }
    string_at(content, "text")
}

/// A tool call as a `tool_call` or `tool_call_update` describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// `toolCallId`.
    pub id: String,
    /// The tool's name as the model sees it, such as `mcp_arko_read_file`.
    pub name: Option<String>,
    /// `pending`, `in_progress`, `completed` or `failed`.
    pub status: Option<String>,
    /// The arguments (`rawInput`).
    pub raw_input: Option<Value>,
    /// The whole update.
    pub raw: Value,
}

impl ToolCall {
    fn parse(update: &Value) -> Option<Self> {
        Some(Self {
            id: string_at(update, "toolCallId")?,
            name: string_at(update, "name"),
            status: string_at(update, "status"),
            raw_input: update.get("rawInput").cloned(),
            raw: update.clone(),
        })
    }
}

/// The reply to a `session/prompt`.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptResult {
    /// `end_turn`, `cancelled`, or another ACP stop reason.
    pub stop_reason: String,
    /// The turn's tokens. agy reports them for every turn that reached the
    /// model, and not for a failed one.
    pub usage: Option<PromptUsage>,
}

impl PromptResult {
    fn parse(result: &Value) -> Self {
        Self {
            stop_reason: string_at(result, "stopReason").unwrap_or_default(),
            usage: result
                .get("usage")
                .filter(|usage| usage.is_object())
                .and_then(|usage| serde_json::from_value(usage.clone()).ok()),
        }
    }

    /// Whether the turn ended because it was cancelled.
    pub fn cancelled(&self) -> bool {
        self.stop_reason == "cancelled"
    }
}

/// The tokens of one turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PromptUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_read_tokens: u64,
    pub cached_write_tokens: u64,
    pub thought_tokens: u64,
    pub total_tokens: u64,
}

/// Whether a turn failed upstream, and why. agy reports a failure as a text
/// chunk starting with `⚠` (such as `⚠ INVALID_ARGUMENT (code 400): ...`) and
/// ends the turn with `end_turn` and no usage. `text` is the reply text of
/// the turn. The session then fails the same way on every prompt, so a fresh
/// session is needed.
pub fn turn_failure(text: &str, result: &PromptResult) -> Option<String> {
    if result.usage.is_some() || result.stop_reason != "end_turn" {
        return None;
    }
    text.rsplit_once('⚠')
        .map(|(_, reason)| reason.trim())
        .filter(|reason| !reason.is_empty())
        .map(str::to_string)
}

/// Whether agy's log says it couldn't load `agent` and ran its default agent
/// instead, with its own tools, prompt and the project's customizations.
/// agy writes it while it handles `session/new`, and it is the only sign.
pub fn agent_fell_back(log: &str, agent: &str) -> bool {
    log.contains(&format!(
        "Agent \"{agent}\" not found, falling back to default"
    ))
}

/// Reads the log file and checks it with [`agent_fell_back`].
pub async fn log_shows_fallback(log_file: &Path, agent: &str) -> io::Result<bool> {
    let log = tokio::fs::read(log_file).await?;
    Ok(agent_fell_back(&String::from_utf8_lossy(&log), agent))
}

/// A text content block for [`AcpClient::prompt`].
pub fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// An image content block for [`AcpClient::prompt`], with the image's bytes
/// in base64.
pub fn image_block(mime_type: &str, base64: &str) -> Value {
    json!({"type": "image", "mimeType": mime_type, "data": base64})
}

/// A `session/request_permission` from agy.
#[derive(Debug, Clone, PartialEq)]
pub struct PermissionRequest {
    pub session_id: String,
    pub tool_call_id: Option<String>,
    /// The tool's name as the model sees it.
    pub tool_name: Option<String>,
    pub options: Vec<PermissionOption>,
    /// The whole request.
    pub params: Value,
}

/// One answer a permission request offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionOption {
    pub option_id: String,
    pub name: String,
    /// `allow_once`, `allow_always`, `reject_once` or `reject_always`.
    pub kind: String,
}

impl PermissionRequest {
    fn parse(params: Value) -> Self {
        let tool_call = params.get("toolCall");
        let options = params
            .get("options")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|option| {
                Some(PermissionOption {
                    option_id: string_at(option, "optionId")?,
                    name: string_at(option, "name").unwrap_or_default(),
                    kind: string_at(option, "kind").unwrap_or_default(),
                })
            })
            .collect();
        Self {
            session_id: string_at(&params, "sessionId").unwrap_or_default(),
            tool_call_id: tool_call.and_then(|call| string_at(call, "toolCallId")),
            tool_name: tool_call.and_then(|call| string_at(call, "name")),
            options,
            params,
        }
    }

    /// The option that allows the call once.
    fn allow_once(&self) -> Option<&str> {
        self.options
            .iter()
            .find(|option| option.kind == "allow_once")
            .map(|option| option.option_id.as_str())
    }
}

/// How a permission request is answered. Rejecting a call would end agy's
/// whole turn, so a call that shouldn't run cancels the turn instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Allow the call once.
    Allow,
    /// Cancel the turn (`session/cancel`) and answer `cancelled`.
    Cancel,
}

/// Decides agy's permission requests. It runs on the task that reads agy's
/// output, so it returns at once.
pub trait PermissionPolicy: Send + Sync {
    fn decide(&self, request: &PermissionRequest) -> PermissionDecision;
}

impl<F> PermissionPolicy for F
where
    F: Fn(&PermissionRequest) -> PermissionDecision + Send + Sync,
{
    fn decide(&self, request: &PermissionRequest) -> PermissionDecision {
        self(request)
    }
}

/// Allows the tools of the MCP server named `arko` ([`ARKO_TOOL_PREFIX`])
/// and cancels the turn for any other.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArkoToolsOnly;

impl PermissionPolicy for ArkoToolsOnly {
    fn decide(&self, request: &PermissionRequest) -> PermissionDecision {
        let arko_tool = request
            .tool_name
            .as_deref()
            .is_some_and(|name| name.starts_with(ARKO_TOOL_PREFIX));
        if arko_tool {
            PermissionDecision::Allow
        } else {
            PermissionDecision::Cancel
        }
    }
}

/// What `session/new` returned.
#[derive(Debug, Clone, PartialEq)]
pub struct NewSession {
    pub session_id: String,
    /// The session's settings, such as `model` and `thought_level`.
    pub config_options: Vec<ConfigOption>,
    /// The plan's name, such as "Google AI Pro".
    pub tier: Option<String>,
    /// The whole reply.
    pub raw: Value,
}

impl NewSession {
    fn parse(raw: Value) -> Result<Self, AcpError> {
        let session_id = string_at(&raw, "sessionId")
            .ok_or_else(|| AcpError::Protocol("`session/new` gave no session ID.".into()))?;
        Ok(Self {
            session_id,
            config_options: ConfigOption::parse_all(&raw),
            tier: raw
                .pointer("/_meta/antigravity/tier/name")
                .and_then(Value::as_str)
                .map(str::to_string),
            raw,
        })
    }
}

/// A session setting with the values it takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigOption {
    pub id: String,
    pub name: String,
    pub current_value: Option<String>,
    pub options: Vec<ConfigValue>,
}

/// One value a [`ConfigOption`] takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigValue {
    pub value: String,
    pub name: String,
}

impl ConfigOption {
    fn parse_all(reply: &Value) -> Vec<Self> {
        reply
            .get("configOptions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Self::parse)
            .collect()
    }

    fn parse(option: &Value) -> Option<Self> {
        let options = option
            .get("options")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                Some(ConfigValue {
                    value: string_at(entry, "value")?,
                    name: string_at(entry, "name").unwrap_or_default(),
                })
            })
            .collect();
        Some(Self {
            id: string_at(option, "id")?,
            name: string_at(option, "name").unwrap_or_default(),
            current_value: string_at(option, "currentValue"),
            options,
        })
    }
}

/// The values the setting `id` takes, empty when there is no such setting.
pub fn config_values<'a>(options: &'a [ConfigOption], id: &str) -> Vec<&'a str> {
    options
        .iter()
        .filter(|option| option.id == id)
        .flat_map(|option| option.options.iter().map(|value| value.value.as_str()))
        .collect()
}

fn string_at(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn lock<T>(mutex: &StdMutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Who waits for the reply to a request.
enum Waiter {
    Reply(oneshot::Sender<Result<Value, RpcError>>),
    /// A prompt, whose reply becomes an [`AcpEvent::PromptFinished`].
    Prompt,
}

#[derive(Default)]
struct Pending {
    /// Set once agy's output has ended, after which no reply can come.
    closed: bool,
    waiters: HashMap<u64, Waiter>,
}

/// What the client and its tasks share. Each lock is taken alone, for one
/// statement, with nothing called under it.
#[derive(Default)]
struct Shared {
    pending: StdMutex<Pending>,
    stderr: StdMutex<String>,
    frame_too_large: AtomicBool,
}

impl Shared {
    /// Registers a waiter, unless agy's output has ended.
    fn register(&self, id: u64, waiter: Waiter) -> bool {
        let mut pending = lock(&self.pending);
        if pending.closed {
            return false;
        }
        pending.waiters.insert(id, waiter);
        true
    }

    fn forget(&self, id: u64) {
        let waiter = lock(&self.pending).waiters.remove(&id);
        drop(waiter);
    }

    /// Ends every wait, as no reply can come any more.
    fn close(&self) {
        let waiters = {
            let mut pending = lock(&self.pending);
            pending.closed = true;
            std::mem::take(&mut pending.waiters)
        };
        drop(waiters);
    }

    fn exit_error(&self) -> AcpError {
        if self.frame_too_large.load(Ordering::Acquire) {
            return AcpError::FrameTooLarge;
        }
        let stderr = lock(&self.stderr).trim().to_string();
        AcpError::Exited { stderr }
    }
}

enum Outgoing {
    Line(String),
    /// Closes agy's input, after the lines before it.
    Close,
}

/// A running `agy acp`.
///
/// agy runs in a process group of its own (a job object on Windows) with the
/// programs it starts, such as its MCP servers. Closing or dropping the
/// client ends them all. On Windows agy starts without a console window.
pub struct AcpClient {
    outgoing: mpsc::UnboundedSender<Outgoing>,
    events: mpsc::UnboundedReceiver<AcpEvent>,
    shared: Arc<Shared>,
    next_id: AtomicU64,
    child: Child,
    tree: ProcessTree,
    reader: JoinHandle<()>,
    writer: JoinHandle<()>,
    initialization: Value,
}

impl AcpClient {
    /// Starts agy and completes `initialize`.
    pub async fn start(
        options: AcpOptions,
        permissions: Arc<dyn PermissionPolicy>,
    ) -> Result<Self, AcpError> {
        Self::start_with_limit(options, permissions, MAX_FRAME).await
    }

    async fn start_with_limit(
        options: AcpOptions,
        permissions: Arc<dyn PermissionPolicy>,
        max_frame: usize,
    ) -> Result<Self, AcpError> {
        let mut command = Command::new(&options.program);
        command
            .args(&options.prefix_args)
            .args(options.arguments())
            .current_dir(&options.working_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(windows)]
        command.creation_flags(
            windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP
                | crate::process::CREATE_NO_WINDOW,
        );
        for (name, value) in &options.env {
            command.env(name, value);
        }
        command.env(NO_AUTO_UPDATE.0, NO_AUTO_UPDATE.1);
        let (mut child, tree) =
            ProcessTree::spawn(&mut command).map_err(|source| AcpError::Spawn {
                program: options.program.display().to_string(),
                source,
            })?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");

        let shared = Arc::new(Shared::default());
        tokio::spawn(collect_stderr(stderr, shared.clone()));
        let (outgoing, lines) = mpsc::unbounded_channel();
        let writer = tokio::spawn(write_lines(stdin, lines));
        let (events_sender, events) = mpsc::unbounded_channel();
        let router = Router {
            shared: shared.clone(),
            outgoing: outgoing.clone(),
            events: events_sender,
            permissions,
        };
        let reader = tokio::spawn(read_lines(stdout, max_frame, router));

        let mut client = Self {
            outgoing,
            events,
            shared,
            next_id: AtomicU64::new(0),
            child,
            tree,
            reader,
            writer,
            initialization: Value::Null,
        };
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": options.client_info.name, "version": options.client_info.version},
            "clientCapabilities": {
                "fs": {"readTextFile": false, "writeTextFile": false},
                "terminal": false,
            },
        });
        let initialization = client
            .request("initialize", params, INITIALIZE_TIMEOUT)
            .await?;
        let version = initialization
            .get("protocolVersion")
            .and_then(Value::as_u64);
        if version != Some(PROTOCOL_VERSION) {
            return Err(AcpError::Protocol(format!(
                "`initialize` chose protocol version {}, and the client speaks version {PROTOCOL_VERSION}.",
                version.map_or_else(|| "none".to_string(), |version| version.to_string())
            )));
        }
        client.initialization = initialization;
        Ok(client)
    }

    /// agy's reply to `initialize`.
    pub fn initialization(&self) -> &Value {
        &self.initialization
    }

    /// agy's version as `initialize` reported it (`agentInfo.version`).
    pub fn version(&self) -> Option<&str> {
        self.initialization
            .pointer("/agentInfo/version")
            .and_then(Value::as_str)
    }

    /// Starts a session in `cwd` (`session/new`). A custom agent takes its MCP
    /// servers from the profile, so none are passed here. No model is called.
    pub async fn new_session(&self, cwd: &Path) -> Result<NewSession, AcpError> {
        let cwd = cwd.to_str().ok_or_else(|| {
            AcpError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The session's folder needs a UTF-8 path.",
            ))
        })?;
        let reply = self
            .request(
                "session/new",
                json!({"cwd": cwd, "mcpServers": []}),
                SESSION_TIMEOUT,
            )
            .await?;
        NewSession::parse(reply)
    }

    /// Opens the session `session_id` that an earlier process of the same
    /// profile made (`session/resume`), with what it held. agy sends no
    /// updates of the old turns. The session is back on the account's
    /// default model and thought level, so set them again. It keeps the
    /// agent and system prompt it was made with, and reads the profile's MCP
    /// tools again. agy refuses a session its profile doesn't hold.
    pub async fn resume_session(
        &self,
        session_id: &str,
        cwd: &Path,
    ) -> Result<NewSession, AcpError> {
        let cwd = cwd.to_str().ok_or_else(|| {
            AcpError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The session's folder needs a UTF-8 path.",
            ))
        })?;
        let mut reply = self
            .request(
                "session/resume",
                json!({"sessionId": session_id, "cwd": cwd, "mcpServers": []}),
                SESSION_TIMEOUT,
            )
            .await?;
        if let Some(fields) = reply.as_object_mut() {
            fields
                .entry("sessionId")
                .or_insert_with(|| Value::String(session_id.to_string()));
        }
        NewSession::parse(reply)
    }

    /// Changes a session setting (`session/set_config_option`), such as
    /// `model` and then `thought_level`, and returns the settings as they are
    /// now. The thought levels on offer depend on the model.
    pub async fn set_config_option(
        &self,
        session_id: &str,
        config_id: &str,
        value: &str,
    ) -> Result<Vec<ConfigOption>, AcpError> {
        let reply = self
            .request(
                "session/set_config_option",
                json!({"sessionId": session_id, "configId": config_id, "value": value}),
                CONFIG_TIMEOUT,
            )
            .await?;
        Ok(ConfigOption::parse_all(&reply))
    }

    /// Queues a prompt (`session/prompt`) and returns its request ID.
    /// [`AcpEvent::PromptFinished`] carries its result and ID. A prompt sent
    /// while a turn runs joins it. The earlier result comes at once and the
    /// new result ends the turn.
    pub async fn prompt(&self, session_id: &str, prompt: Vec<Value>) -> Result<u64, AcpError> {
        self.send_request(
            "session/prompt",
            json!({"sessionId": session_id, "prompt": prompt}),
            Waiter::Prompt,
        )
    }

    /// Cancels the session's running turn (`session/cancel`). The running
    /// MCP call is cancelled, and the prompt's result says `cancelled`.
    pub async fn cancel(&self, session_id: &str) -> Result<(), AcpError> {
        self.notify("session/cancel", json!({"sessionId": session_id}))
            .await
    }

    /// Sends a request and waits up to `timeout` for its reply.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, AcpError> {
        let (sender, receiver) = oneshot::channel();
        let id = self.send_request(method, params, Waiter::Reply(sender))?;
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(error))) => Err(AcpError::Rpc {
                method: method.to_string(),
                error,
            }),
            Ok(Err(_)) => Err(self.exit_error()),
            Err(_) => {
                self.shared.forget(id);
                Err(AcpError::Timeout(method.to_string()))
            }
        }
    }

    /// Sends a notification.
    pub async fn notify(&self, method: &str, params: Value) -> Result<(), AcpError> {
        let line = json!({"jsonrpc": "2.0", "method": method, "params": params}).to_string();
        self.outgoing
            .send(Outgoing::Line(line))
            .map_err(|_| self.exit_error())
    }

    /// The next event. `None` once agy's output has ended, because it exited
    /// or sent a line over [`MAX_FRAME`] ([`exit_error`](Self::exit_error)
    /// says which). Drop or close the client then.
    pub async fn next_event(&mut self) -> Option<AcpEvent> {
        self.events.recv().await
    }

    /// Why the connection ended, with the end of agy's stderr.
    pub fn exit_error(&self) -> AcpError {
        self.shared.exit_error()
    }

    /// Closes agy's input and waits for it to exit, killing it if it hasn't
    /// within a grace period. The programs it started are killed with it.
    pub async fn close(mut self) -> Result<ExitStatus, AcpError> {
        let _ = self.outgoing.send(Outgoing::Close);
        // The tree is killed before agy is collected, while its group's ID
        // can't have passed to another program.
        let status = match tokio::time::timeout(EXIT_GRACE, self.tree.wait(&mut self.child)).await {
            Ok(status) => status,
            Err(_) => {
                self.tree.kill();
                match self.child.kill().await {
                    Ok(()) => self.child.wait().await,
                    Err(error) => Err(error),
                }
            }
        };
        Ok(status?)
    }

    fn send_request(&self, method: &str, params: Value, waiter: Waiter) -> Result<u64, AcpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        if !self.shared.register(id, waiter) {
            return Err(self.exit_error());
        }
        let line =
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
        if self.outgoing.send(Outgoing::Line(line)).is_err() {
            self.shared.forget(id);
            return Err(self.exit_error());
        }
        Ok(id)
    }
}

impl Drop for AcpClient {
    /// Ends what the client started. The tree goes first, while agy is still
    /// uncollected.
    fn drop(&mut self) {
        self.tree.kill();
        self.reader.abort();
        self.writer.abort();
    }
}

/// Writes lines to agy in the order they were queued, so a line is never cut
/// by another, and nothing waits for agy to read while holding a lock.
async fn write_lines(mut stdin: ChildStdin, mut lines: mpsc::UnboundedReceiver<Outgoing>) {
    while let Some(Outgoing::Line(mut line)) = lines.recv().await {
        log::trace!(target: WIRE_LOG, "-> {line}");
        line.push('\n');
        if stdin.write_all(line.as_bytes()).await.is_err() || stdin.flush().await.is_err() {
            break;
        }
    }
}

async fn collect_stderr(mut pipe: ChildStderr, shared: Arc<Shared>) {
    let mut buffer = [0u8; 4096];
    while let Ok(read) = pipe.read(&mut buffer).await {
        if read == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buffer[..read]).into_owned();
        log::debug!("agy stderr: {}", chunk.trim_end());
        let mut tail = lock(&shared.stderr);
        tail.push_str(&chunk);
        if tail.len() > STDERR_TAIL {
            let mut cut = tail.len() - STDERR_TAIL;
            while !tail.is_char_boundary(cut) {
                cut += 1;
            }
            tail.drain(..cut);
        }
    }
}

/// Reads agy's output line by line, refusing a line over a size limit.
struct Frames<R> {
    reader: BufReader<R>,
    line: Vec<u8>,
    max: usize,
}

impl<R: AsyncRead + Unpin> Frames<R> {
    fn new(reader: R, max: usize) -> Self {
        Self {
            reader: BufReader::new(reader),
            line: Vec::new(),
            max,
        }
    }

    /// The next line without its newline. `Ok(None)` at the end of the output
    /// or when reading it fails, `Err` for a line over the limit.
    async fn next(&mut self) -> Result<Option<Vec<u8>>, Oversized> {
        loop {
            let available = match self.reader.fill_buf().await {
                Ok(available) => available,
                Err(error) => {
                    log::debug!("Reading agy's output failed: {error}");
                    return Ok(None);
                }
            };
            if available.is_empty() {
                let last = std::mem::take(&mut self.line);
                return Ok((!last.is_empty()).then_some(last));
            }
            let newline = available.iter().position(|&byte| byte == b'\n');
            let length = newline.unwrap_or(available.len());
            if self.line.len() + length > self.max {
                return Err(Oversized);
            }
            self.line.extend_from_slice(&available[..length]);
            self.reader.consume(length + usize::from(newline.is_some()));
            if newline.is_some() {
                return Ok(Some(std::mem::take(&mut self.line)));
            }
        }
    }
}

/// A line over the size limit.
struct Oversized;

async fn read_lines(stdout: impl AsyncRead + Unpin, max_frame: usize, router: Router) {
    let mut frames = Frames::new(stdout, max_frame);
    loop {
        match frames.next().await {
            Ok(Some(line)) => router.route(&line),
            Ok(None) => break,
            Err(Oversized) => {
                log::warn!("agy sent a line over {max_frame} bytes, so the connection ends");
                router.shared.frame_too_large.store(true, Ordering::Release);
                break;
            }
        }
    }
    // No reply can come any more. Dropping the router ends the events.
    router.shared.close();
}

/// Sends each message from agy where it goes.
struct Router {
    shared: Arc<Shared>,
    outgoing: mpsc::UnboundedSender<Outgoing>,
    events: mpsc::UnboundedSender<AcpEvent>,
    permissions: Arc<dyn PermissionPolicy>,
}

impl Router {
    fn route(&self, line: &[u8]) {
        let text = String::from_utf8_lossy(line);
        log::trace!(target: WIRE_LOG, "<- {text}");
        let Ok(mut message) = serde_json::from_slice::<Value>(line) else {
            if !text.trim().is_empty() {
                log::debug!("agy stdout (not JSON): {text}");
            }
            return;
        };
        let method = string_at(&message, "method");
        let id = message
            .get_mut("id")
            .map(Value::take)
            .filter(|id| !id.is_null());
        let params = message
            .get_mut("params")
            .map(Value::take)
            .unwrap_or(Value::Null);
        match (method, id) {
            (Some(method), Some(id)) => self.answer(id, &method, params),
            (Some(method), None) => self.notification(method, params),
            (None, Some(id)) => self.reply(&id, &message),
            (None, None) => log::debug!("agy sent a message with no method and no ID: {text}"),
        }
    }

    fn reply(&self, id: &Value, message: &Value) {
        let Some(id) = id.as_u64() else {
            log::debug!("agy answered a request the client didn't send ({id})");
            return;
        };
        let outcome = match message.get("error").filter(|error| !error.is_null()) {
            Some(error) => Err(RpcError::parse(error)),
            None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
        };
        let waiter = lock(&self.shared.pending).waiters.remove(&id);
        match waiter {
            Some(Waiter::Reply(sender)) => {
                let _ = sender.send(outcome);
            }
            Some(Waiter::Prompt) => {
                let outcome = outcome.map(|result| PromptResult::parse(&result));
                let _ = self.events.send(AcpEvent::PromptFinished {
                    request: id,
                    outcome,
                });
            }
            None => log::debug!("agy answered request {id}, which nothing waits for"),
        }
    }

    fn notification(&self, method: String, mut params: Value) {
        let update = if method == "session/update" {
            params.get_mut("update").map(Value::take)
        } else {
            None
        };
        let event = match update {
            Some(update) => AcpEvent::Update {
                session_id: string_at(&params, "sessionId").unwrap_or_default(),
                update: SessionUpdate::parse(update),
            },
            None => AcpEvent::Notification { method, params },
        };
        let _ = self.events.send(event);
    }

    /// Answers a request from agy. Only permission requests are served.
    fn answer(&self, id: Value, method: &str, params: Value) {
        if method != "session/request_permission" {
            self.send(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": METHOD_NOT_FOUND, "message": format!("The client doesn't offer `{method}`.")},
            }));
            return;
        }
        let request = PermissionRequest::parse(params);
        let allow = match self.permissions.decide(&request) {
            PermissionDecision::Allow => request.allow_once().map(str::to_string),
            PermissionDecision::Cancel => None,
        };
        let (outcome, decision) = match allow {
            Some(option) => (
                json!({"outcome": "selected", "optionId": option}),
                PermissionDecision::Allow,
            ),
            None => {
                // ACP's order: cancel the turn, then answer what it left
                // waiting with `cancelled`.
                self.send(json!({
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": request.session_id},
                }));
                (json!({"outcome": "cancelled"}), PermissionDecision::Cancel)
            }
        };
        self.send(json!({"jsonrpc": "2.0", "id": id, "result": {"outcome": outcome}}));
        let _ = self.events.send(AcpEvent::Permission { request, decision });
    }

    fn send(&self, message: Value) {
        let _ = self.outgoing.send(Outgoing::Line(message.to_string()));
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A stand-in for `agy`, run by Python. `--version` prints a version,
    /// `-p /usage` prints a file, and `acp` serves ACP, where the first word
    /// of a prompt picks what the turn does. It writes down its command line
    /// and every message it gets.
    pub(crate) const FAKE_AGY: &str = r##""""A scripted stand-in for the Antigravity CLI (`agy`).

`--version` prints FAKE_AGY_VERSION (default 1.5.14). `-p /usage` prints the
file FAKE_AGY_USAGE_FILE and exits with FAKE_AGY_USAGE_EXIT. `acp` answers ACP
requests on stdio, and the first word of a prompt picks what the turn does.
Every command line and every message received goes to FAKE_AGY_RECORD.
"""

import json
import os
import sys

args = sys.argv[1:]
record = os.environ.get('FAKE_AGY_RECORD')


def note(entry):
    if record:
        with open(record, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')


note({'args': args, 'cwd': os.getcwd(),
      'no_update': os.environ.get('AGY_CLI_DISABLE_AUTO_UPDATE'),
      'extra': os.environ.get('FAKE_AGY_EXTRA')})

if args[:1] == ['--version']:
    print(os.environ.get('FAKE_AGY_VERSION', '1.5.14'))
    sys.exit(0)
if args[:2] == ['-p', '/usage']:
    with open(os.environ['FAKE_AGY_USAGE_FILE'], encoding='utf-8') as f:
        sys.stdout.write(f.read())
    sys.exit(int(os.environ.get('FAKE_AGY_USAGE_EXIT', '0')))
if args[:1] != ['acp']:
    sys.stderr.write('unknown command\n')
    sys.exit(2)

FAMILIES = [
    ('gemini-4-neon', 'Gemini 4 Neon', ['low', 'medium', 'high', 'xhigh'], 'high'),
    ('gemini-3.8-flash', 'Gemini 3.8 Flash', ['low', 'medium', 'high'], 'high'),
    ('gpt-oss-120b-medium', 'GPT-OSS 120B (Medium)', [], None),
]
USAGE = {'inputTokens': 100, 'outputTokens': 5, 'cachedReadTokens': 40,
         'cachedWriteTokens': 0, 'thoughtTokens': 2, 'totalTokens': 105}
state = {'model': 'gemini-4-neon', 'level': 'high'}
running = None
waiting = {}
next_request = [0]


def send(message):
    sys.stdout.write(json.dumps(dict({'jsonrpc': '2.0'}, **message)) + '\n')
    sys.stdout.flush()


def update(body):
    send({'method': 'session/update', 'params': {'sessionId': 's1', 'update': body}})


def text(kind, words):
    update({'sessionUpdate': kind, 'content': {'type': 'text', 'text': words}})


def finish(prompt_id, stop='end_turn', usage=True):
    global running
    result = {'stopReason': stop}
    if usage:
        result['usage'] = USAGE
    send({'id': prompt_id, 'result': result})
    if running == prompt_id:
        running = None


def ask(method, params, then):
    next_request[0] += 1
    request_id = str(next_request[0])
    waiting[request_id] = then
    send({'id': request_id, 'method': method, 'params': params})


def config_options():
    family = next(f for f in FAMILIES if f[0] == state['model'])
    options = [{'id': 'model', 'name': 'Model', 'category': 'model', 'type': 'select',
                'currentValue': state['model'],
                'options': [{'value': f[0], 'name': f[1]} for f in FAMILIES]}]
    if family[2]:
        options.append({'id': 'thought_level', 'name': 'Thinking Level',
                        'category': 'thought_level', 'type': 'select',
                        'currentValue': state['level'],
                        'options': [{'value': v, 'name': v.title()} for v in family[2]]})
    options.append({'id': 'mode', 'name': 'Mode', 'category': 'mode', 'type': 'select',
                    'currentValue': 'default',
                    'options': [{'value': 'default', 'name': 'Default'}]})
    return options


def on_prompt(prompt_id, params):
    global running
    words = params['prompt'][0]['text'].split()
    if running is not None:
        # A prompt sent while a turn runs joins it, and the first prompt's
        # result comes at once.
        finish(running)
        running = prompt_id
        text('agent_message_chunk', 'joined: ' + ' '.join(words))
        finish(prompt_id)
        return
    running = prompt_id
    command = words[0]
    if command == 'reply':
        update({'sessionUpdate': 'usage_update', 'used': 0, 'size': 1048576})
        text('agent_thought_chunk', 'thinking')
        text('agent_message_chunk', 'Hello')
        text('agent_message_chunk', ' there')
        update({'sessionUpdate': 'usage_update', 'used': 105, 'size': 1048576})
        finish(prompt_id)
    elif command == 'fail':
        text('agent_message_chunk',
             '\n⚠ INVALID_ARGUMENT (code 400): Unable to process input image.\n')
        finish(prompt_id, usage=False)
    elif command == 'tool':
        name = words[1]
        call = {'toolCallId': 'call_1', 'name': name, 'args': {'text': 'hi'},
                'status': 'pending', 'kind': 'other', 'title': 'Run ' + name,
                'rawInput': {'text': 'hi'}}
        update(dict({'sessionUpdate': 'tool_call'}, **call))

        def answered(reply):
            outcome = reply.get('result', {}).get('outcome', {})
            if running != prompt_id:
                return
            text('agent_message_chunk', json.dumps(outcome, sort_keys=True))
            update(dict({'sessionUpdate': 'tool_call_update'}, **dict(call, status='completed')))
            finish(prompt_id)

        ask('session/request_permission',
            {'sessionId': 's1', 'toolCall': dict({'sessionUpdate': 'tool_call'}, **call),
             'options': [{'optionId': 'allow', 'name': 'Allow', 'kind': 'allow_once'},
                         {'optionId': 'deny', 'name': 'Deny', 'kind': 'reject_once'}]},
            answered)
    elif command == 'files':
        codes = []

        def answered(reply):
            codes.append(reply.get('error', {}).get('code'))
            if len(codes) == 2:
                text('agent_message_chunk', json.dumps(codes))
                finish(prompt_id)

        ask('fs/read_text_file', {'sessionId': 's1', 'path': '/home/ada/proj/a.txt'}, answered)
        ask('terminal/create', {'sessionId': 's1', 'command': 'ls'}, answered)
    elif command == 'wait':
        text('agent_message_chunk', 'working')
    elif command == 'big':
        text('agent_message_chunk', 'x' * int(words[1]))
        finish(prompt_id)
    elif command == 'notify':
        send({'method': '_antigravity/hello', 'params': {'n': 1}})
        finish(prompt_id)
    elif command == 'exit':
        sys.stderr.write('fake agy crashed\n')
        sys.stderr.flush()
        os._exit(3)
    else:
        text('agent_message_chunk', 'unknown prompt')
        finish(prompt_id)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    note({'in': message})
    method = message.get('method')
    params = message.get('params', {})
    if method is None:
        then = waiting.pop(str(message.get('id')), None)
        if then:
            then(message)
        continue
    message_id = message.get('id')
    if method == 'initialize':
        send({'id': message_id, 'result': {
            'protocolVersion': int(os.environ.get('FAKE_AGY_PROTOCOL', '1')),
            'agentCapabilities': {'promptCapabilities': {'image': True},
                                  '_meta': {'antigravity': {'tier': True}}},
            'agentInfo': {'name': 'Antigravity ACP Server', 'version': '1.5.14'}}})
    elif method == 'session/new':
        if os.environ.get('FAKE_AGY_SIGNED_OUT'):
            send({'id': message_id, 'error': {'code': -32000, 'message': 'Authentication required'}})
        elif os.environ.get('FAKE_AGY_BROKEN'):
            send({'id': message_id, 'error': {'code': -32603, 'message': 'Internal error'}})
        else:
            send({'id': message_id, 'result': {
                'sessionId': 's1', 'configOptions': config_options(),
                '_meta': {'antigravity': {'tier': {'name': 'Google AI Pro'}}}}})
    elif method == 'session/set_config_option':
        if params['configId'] == 'model':
            family = next((f for f in FAMILIES if f[0] == params['value']), None)
            if family is None:
                send({'id': message_id, 'error': {'code': -32602, 'message': 'unknown model'}})
                continue
            state['model'], state['level'] = family[0], family[3]
        elif params['configId'] == 'thought_level':
            state['level'] = params['value']
        send({'id': message_id, 'result': {'configOptions': config_options()}})
    elif method == 'session/prompt':
        on_prompt(message_id, params)
    elif method == 'session/cancel':
        if running is not None:
            finish(running, stop='cancelled')
    elif message_id is not None:
        send({'id': message_id, 'error': {'code': -32601, 'message': 'no ' + method}})
"##;

    /// The usage the fake reports for each turn.
    const FAKE_USAGE: PromptUsage = PromptUsage {
        input_tokens: 100,
        output_tokens: 5,
        cached_read_tokens: 40,
        cached_write_tokens: 0,
        thought_tokens: 2,
        total_tokens: 105,
    };

    /// The Python interpreter that runs the fake (`ARKO_TEST_PYTHON` picks
    /// one).
    pub(crate) fn python() -> PathBuf {
        std::env::var_os("ARKO_TEST_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }))
    }

    /// The fake agy in a folder of its own, beside a profile and a working
    /// folder.
    pub(crate) struct Fake {
        pub dir: tempfile::TempDir,
        pub script: PathBuf,
        pub record: PathBuf,
    }

    impl Fake {
        pub fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let script = dir.path().join("fake_agy.py");
            std::fs::write(&script, FAKE_AGY).unwrap();
            std::fs::create_dir(dir.path().join("profile")).unwrap();
            std::fs::create_dir(dir.path().join("work")).unwrap();
            let record = dir.path().join("record.jsonl");
            Self {
                dir,
                script,
                record,
            }
        }

        pub fn prefix_args(&self) -> Vec<String> {
            vec![self.script.display().to_string()]
        }

        pub fn env(&self) -> Vec<(String, String)> {
            vec![("FAKE_AGY_RECORD".into(), self.record.display().to_string())]
        }

        pub fn profile(&self) -> PathBuf {
            self.dir.path().join("profile")
        }

        pub fn work(&self) -> PathBuf {
            self.dir.path().join("work")
        }

        pub fn options(&self) -> AcpOptions {
            let mut options = AcpOptions::new(python(), self.profile(), self.work());
            options.prefix_args = self.prefix_args();
            options.env = self.env();
            options
        }

        /// What the fake wrote down, one value per launch and per message.
        pub fn records(&self) -> Vec<Value> {
            std::fs::read_to_string(&self.record)
                .unwrap_or_default()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }

        /// The messages the fake got.
        pub fn received(&self) -> Vec<Value> {
            self.records()
                .into_iter()
                .filter_map(|mut record| record.get_mut("in").map(Value::take))
                .collect()
        }
    }

    /// Fails the test instead of hanging.
    pub(crate) async fn timed<F: std::future::Future>(future: F) -> F::Output {
        tokio::time::timeout(Duration::from_secs(30), future)
            .await
            .expect("the fake agy answers in time")
    }

    async fn start_with(options: AcpOptions, permissions: Arc<dyn PermissionPolicy>) -> AcpClient {
        timed(AcpClient::start(options, permissions)).await.unwrap()
    }

    /// A client of the fake with a session `s1`.
    async fn session(fake: &Fake) -> AcpClient {
        let client = start_with(fake.options(), Arc::new(ArkoToolsOnly)).await;
        timed(client.new_session(&fake.work())).await.unwrap();
        client
    }

    async fn prompt(client: &AcpClient, text: &str) -> u64 {
        timed(client.prompt("s1", vec![text_block(text)]))
            .await
            .unwrap()
    }

    /// The events up to the result of prompt `request`.
    async fn turn(client: &mut AcpClient, request: u64) -> Vec<AcpEvent> {
        let mut events = Vec::new();
        loop {
            let event = timed(client.next_event())
                .await
                .expect("agy is still running");
            let done = matches!(&event, AcpEvent::PromptFinished { request: finished, .. } if *finished == request);
            events.push(event);
            if done {
                return events;
            }
        }
    }

    fn update(update: SessionUpdate) -> AcpEvent {
        AcpEvent::Update {
            session_id: "s1".into(),
            update,
        }
    }

    fn message_text(events: &[AcpEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                AcpEvent::Update {
                    update: SessionUpdate::AgentMessage { text },
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn finished(events: &[AcpEvent]) -> &PromptResult {
        match events.last() {
            Some(AcpEvent::PromptFinished {
                outcome: Ok(result),
                ..
            }) => result,
            other => panic!("the turn ended with {other:?}"),
        }
    }

    fn decisions(events: &[AcpEvent]) -> Vec<(Option<String>, PermissionDecision)> {
        events
            .iter()
            .filter_map(|event| match event {
                AcpEvent::Permission { request, decision } => {
                    Some((request.tool_name.clone(), *decision))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn options_become_agy_arguments() {
        let mut options = AcpOptions::new("agy", "/home/ada/.arko/agy/run-1", "/home/ada/proj");
        options.agent = Some("arko".into());
        options.log_file = Some(PathBuf::from("/home/ada/.arko/agy/run-1/acp.log"));
        let arguments: Vec<String> = options
            .arguments()
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            arguments,
            [
                "acp",
                "--stdio",
                "--include-default-customizations=false",
                "--disable-telemetry",
                "--gemini-dir",
                "/home/ada/.arko/agy/run-1",
                "--agent",
                "arko",
                "--log-file",
                "/home/ada/.arko/agy/run-1/acp.log",
            ]
        );
        let plain = AcpOptions::new("agy", "/home/ada/.arko/agy/run-2", "/home/ada/proj");
        assert_eq!(plain.arguments().len(), 6);
    }

    /// The shapes agy 1.5.14 sends.
    #[test]
    fn session_updates_are_read_as_agy_sends_them() {
        assert_eq!(
            SessionUpdate::parse(
                json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "OK"}})
            ),
            SessionUpdate::AgentMessage { text: "OK".into() }
        );
        assert_eq!(
            SessionUpdate::parse(
                json!({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "Plan"}})
            ),
            SessionUpdate::AgentThought {
                text: "Plan".into()
            }
        );
        assert_eq!(
            SessionUpdate::parse(
                json!({"sessionUpdate": "usage_update", "used": 1168, "size": 1048576})
            ),
            SessionUpdate::Usage {
                used: 1168,
                size: 1048576
            }
        );
        let call = json!({
            "sessionUpdate": "tool_call",
            "toolCallId": "call_1",
            "name": "mcp_arko_probe_echo",
            "args": {"text": "hi", "toolSummary": "Echo test"},
            "status": "in_progress",
            "kind": "other",
            "title": "Run mcp_arko_probe_echo",
            "rawInput": {"text": "hi", "toolSummary": "Echo test"},
        });
        let SessionUpdate::ToolCall(parsed) = SessionUpdate::parse(call.clone()) else {
            panic!("not a tool call");
        };
        assert_eq!(parsed.id, "call_1");
        assert_eq!(parsed.name.as_deref(), Some("mcp_arko_probe_echo"));
        assert_eq!(parsed.status.as_deref(), Some("in_progress"));
        assert_eq!(
            parsed.raw_input,
            Some(json!({"text": "hi", "toolSummary": "Echo test"}))
        );
        assert_eq!(parsed.raw, call);
        let mut completed = call.clone();
        completed["sessionUpdate"] = json!("tool_call_update");
        completed["status"] = json!("completed");
        assert!(matches!(
            SessionUpdate::parse(completed),
            SessionUpdate::ToolCallUpdate(ToolCall { status: Some(status), .. }) if status == "completed"
        ));
        let title = json!({"sessionUpdate": "session_info_update", "title": "Echo"});
        assert_eq!(
            SessionUpdate::parse(title.clone()),
            SessionUpdate::Other {
                kind: "session_info_update".into(),
                update: title
            }
        );
        let image = json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "image", "data": "aGk=", "mimeType": "image/png"}});
        assert!(matches!(
            SessionUpdate::parse(image),
            SessionUpdate::Other { kind, .. } if kind == "agent_message_chunk"
        ));
    }

    #[test]
    fn a_turn_without_usage_failed() {
        let done = PromptResult::parse(&json!({
            "stopReason": "end_turn",
            "usage": {"inputTokens": 3697, "outputTokens": 165, "cachedReadTokens": 0, "cachedWriteTokens": 0, "thoughtTokens": 99, "totalTokens": 3862},
        }));
        assert_eq!(
            done.usage,
            Some(PromptUsage {
                input_tokens: 3697,
                output_tokens: 165,
                cached_read_tokens: 0,
                cached_write_tokens: 0,
                thought_tokens: 99,
                total_tokens: 3862,
            })
        );
        assert_eq!(turn_failure("Echo result: echo: hi", &done), None);

        let cancelled = PromptResult::parse(&json!({
            "stopReason": "cancelled",
            "usage": {"inputTokens": 9491, "outputTokens": 28, "cachedReadTokens": 0, "cachedWriteTokens": 0, "thoughtTokens": 0, "totalTokens": 9519},
        }));
        assert!(cancelled.cancelled());
        assert_eq!(turn_failure("", &cancelled), None);

        let failed = PromptResult::parse(&json!({"stopReason": "end_turn"}));
        assert_eq!(failed.usage, None);
        assert_eq!(
            turn_failure(
                "\n⚠ INVALID_ARGUMENT (code 400): Unable to process input image. Please retry or report in https://developers.generativeai.google/guide/troubleshooting\n",
                &failed
            )
            .as_deref(),
            Some(
                "INVALID_ARGUMENT (code 400): Unable to process input image. Please retry or report in https://developers.generativeai.google/guide/troubleshooting"
            )
        );
        assert_eq!(
            turn_failure(
                "Reading the file.\n⚠ failed to construct executor: unknown component\n",
                &failed
            )
            .as_deref(),
            Some("failed to construct executor: unknown component")
        );
        assert_eq!(turn_failure("", &failed), None);
    }

    #[tokio::test]
    async fn the_fallback_line_names_the_agent() {
        let log = "I1121 14:24:59.754007      53 agents.go:88] Agent \"arko\" not found, falling back to default\n";
        assert!(agent_fell_back(log, "arko"));
        assert!(!agent_fell_back(log, "arko-worker"));
        assert!(!agent_fell_back(
            "I1121 14:24:59.754007      53 server.go:1225] Creating new cascade trajectory\n",
            "arko"
        ));
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("acp.log");
        std::fs::write(&file, log).unwrap();
        assert!(log_shows_fallback(&file, "arko").await.unwrap());
        assert!(
            log_shows_fallback(&dir.path().join("none.log"), "arko")
                .await
                .is_err()
        );
    }

    #[test]
    fn session_new_gives_the_plan_and_settings() {
        let session = NewSession::parse(json!({
            "sessionId": "s1",
            "configOptions": [
                {"id": "model", "name": "Model", "category": "model", "type": "select", "currentValue": "gemini-4-neon",
                 "options": [{"value": "gemini-4-neon", "name": "Gemini 4 Neon"}, {"value": "gemini-3.8-flash", "name": "Gemini 3.8 Flash"}]},
                {"id": "thought_level", "name": "Thinking Level", "category": "thought_level", "type": "select", "currentValue": "high",
                 "options": [{"value": "low", "name": "Low"}, {"value": "medium", "name": "Medium"}, {"value": "high", "name": "High"}, {"value": "xhigh", "name": "Extra"}]},
            ],
            "_meta": {"antigravity": {"tier": {"name": "Google AI Pro", "id": "g1-pro-tier"}}},
        }))
        .unwrap();
        assert_eq!(session.session_id, "s1");
        assert_eq!(session.tier.as_deref(), Some("Google AI Pro"));
        assert_eq!(
            session.config_options[0].current_value.as_deref(),
            Some("gemini-4-neon")
        );
        assert_eq!(
            session.config_options[0].options[1],
            ConfigValue {
                value: "gemini-3.8-flash".into(),
                name: "Gemini 3.8 Flash".into()
            }
        );
        assert_eq!(
            config_values(&session.config_options, "thought_level"),
            ["low", "medium", "high", "xhigh"]
        );
        assert!(config_values(&session.config_options, "mode").is_empty());
        assert!(matches!(
            NewSession::parse(json!({"configOptions": []})),
            Err(AcpError::Protocol(_))
        ));
    }

    #[test]
    fn only_arko_tools_are_allowed() {
        let request = PermissionRequest::parse(json!({
            "sessionId": "s1",
            "toolCall": {"sessionUpdate": "tool_call", "toolCallId": "call_1", "name": "mcp_arko_probe_echo", "status": "pending", "rawInput": {"text": "hi"}},
            "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"}, {"optionId": "deny", "name": "Deny", "kind": "reject_once"}],
        }));
        assert_eq!(request.session_id, "s1");
        assert_eq!(request.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(request.tool_name.as_deref(), Some("mcp_arko_probe_echo"));
        assert_eq!(request.allow_once(), Some("allow"));
        assert_eq!(ArkoToolsOnly.decide(&request), PermissionDecision::Allow);
        let built_in = PermissionRequest::parse(json!({
            "sessionId": "s1",
            "toolCall": {"toolCallId": "call_2", "name": "run_command"},
            "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"}],
        }));
        assert_eq!(ArkoToolsOnly.decide(&built_in), PermissionDecision::Cancel);
        let unnamed = PermissionRequest::parse(json!({"sessionId": "s1", "options": []}));
        assert_eq!(ArkoToolsOnly.decide(&unnamed), PermissionDecision::Cancel);
        assert_eq!(unnamed.allow_once(), None);
    }

    #[tokio::test]
    async fn a_session_runs_a_turn() {
        let fake = Fake::new();
        let mut options = fake.options();
        options.env.push(("FAKE_AGY_EXTRA".into(), "kept".into()));
        // The caller's value can't turn updates back on.
        options.env.push((NO_AUTO_UPDATE.0.into(), "false".into()));
        let mut client = start_with(options, Arc::new(ArkoToolsOnly)).await;
        assert_eq!(client.version(), Some("1.5.14"));

        let session = timed(client.new_session(&fake.work())).await.unwrap();
        assert_eq!(session.session_id, "s1");
        assert_eq!(session.tier.as_deref(), Some("Google AI Pro"));
        assert_eq!(
            config_values(&session.config_options, "thought_level"),
            ["low", "medium", "high", "xhigh"]
        );
        let settings = timed(client.set_config_option("s1", "model", "gemini-3.8-flash"))
            .await
            .unwrap();
        assert_eq!(
            config_values(&settings, "thought_level"),
            ["low", "medium", "high"]
        );

        let request = prompt(&client, "reply").await;
        let events = turn(&mut client, request).await;
        assert_eq!(
            events,
            [
                update(SessionUpdate::Usage {
                    used: 0,
                    size: 1048576
                }),
                update(SessionUpdate::AgentThought {
                    text: "thinking".into()
                }),
                update(SessionUpdate::AgentMessage {
                    text: "Hello".into()
                }),
                update(SessionUpdate::AgentMessage {
                    text: " there".into()
                }),
                update(SessionUpdate::Usage {
                    used: 105,
                    size: 1048576
                }),
                AcpEvent::PromptFinished {
                    request,
                    outcome: Ok(PromptResult {
                        stop_reason: "end_turn".into(),
                        usage: Some(FAKE_USAGE),
                    }),
                },
            ]
        );
        assert!(timed(client.close()).await.unwrap().success());

        let launch = &fake.records()[0];
        assert_eq!(
            launch["args"],
            json!([
                "acp",
                "--stdio",
                "--include-default-customizations=false",
                "--disable-telemetry",
                "--gemini-dir",
                fake.profile().display().to_string(),
            ])
        );
        assert_eq!(launch["no_update"], "true");
        assert_eq!(launch["extra"], "kept");
        assert_eq!(
            Path::new(launch["cwd"].as_str().unwrap())
                .canonicalize()
                .unwrap(),
            fake.work().canonicalize().unwrap()
        );
        let received = fake.received();
        let initialize = &received[0]["params"];
        assert_eq!(initialize["protocolVersion"], 1);
        assert_eq!(initialize["clientCapabilities"]["terminal"], false);
        assert_eq!(
            initialize["clientCapabilities"]["fs"]["readTextFile"],
            false
        );
        assert_eq!(initialize["clientInfo"]["name"], "arko_ai");
        assert_eq!(received[1]["params"]["mcpServers"], json!([]));
        assert_eq!(
            received[3]["params"]["prompt"],
            json!([{"type": "text", "text": "reply"}])
        );
    }

    #[tokio::test]
    async fn client_info_is_sent_when_set() {
        let fake = Fake::new();
        let mut options = fake.options();
        options.client_info = ClientInfo {
            name: "example-host".into(),
            version: "2.0.0".into(),
        };
        let client = start_with(options, Arc::new(ArkoToolsOnly)).await;
        timed(client.close()).await.unwrap();
        assert_eq!(
            fake.received()[0]["params"]["clientInfo"],
            json!({"name": "example-host", "version": "2.0.0"})
        );
    }

    #[tokio::test]
    async fn an_arko_tool_is_allowed_once() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        let request = prompt(&client, "tool mcp_arko_read_file").await;
        let events = turn(&mut client, request).await;
        assert_eq!(
            decisions(&events),
            [(
                Some("mcp_arko_read_file".to_string()),
                PermissionDecision::Allow
            )]
        );
        assert_eq!(
            message_text(&events),
            r#"{"optionId": "allow", "outcome": "selected"}"#
        );
        assert_eq!(finished(&events).stop_reason, "end_turn");
    }

    #[tokio::test]
    async fn another_tool_cancels_the_turn_without_a_reject() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        let request = prompt(&client, "tool run_command").await;
        let events = turn(&mut client, request).await;
        assert_eq!(
            decisions(&events),
            [(Some("run_command".to_string()), PermissionDecision::Cancel)]
        );
        assert!(finished(&events).cancelled());
        // Everything sent is read once agy has exited.
        timed(client.close()).await.unwrap();
        let received = fake.received();
        let cancel = received
            .iter()
            .position(|message| message["method"] == "session/cancel")
            .expect("the turn was cancelled");
        let answer = received
            .iter()
            .position(|message| message["id"] == "1")
            .expect("the request was answered");
        assert!(cancel < answer);
        assert_eq!(
            received[answer]["result"]["outcome"],
            json!({"outcome": "cancelled"})
        );
    }

    #[tokio::test]
    async fn file_and_terminal_requests_are_refused() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        let request = prompt(&client, "files").await;
        let events = turn(&mut client, request).await;
        assert_eq!(message_text(&events), "[-32601, -32601]");
    }

    #[tokio::test]
    async fn a_prompt_during_a_turn_joins_it() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        let first = prompt(&client, "wait").await;
        assert_eq!(
            timed(client.next_event()).await,
            Some(update(SessionUpdate::AgentMessage {
                text: "working".into()
            }))
        );
        let second = prompt(&client, "note for the model").await;
        let events = turn(&mut client, second).await;
        assert!(
            matches!(&events[0], AcpEvent::PromptFinished { request, outcome: Ok(result) } if *request == first && result.stop_reason == "end_turn"),
            "{events:?}"
        );
        assert_eq!(message_text(&events), "joined: note for the model");
        assert_eq!(finished(&events).stop_reason, "end_turn");
    }

    #[tokio::test]
    async fn cancel_ends_the_running_turn() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        let request = prompt(&client, "wait").await;
        timed(client.next_event()).await.unwrap();
        timed(client.cancel("s1")).await.unwrap();
        let events = turn(&mut client, request).await;
        assert!(finished(&events).cancelled());
        assert_eq!(finished(&events).usage, Some(FAKE_USAGE));
    }

    #[tokio::test]
    async fn other_notifications_pass_through() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        let request = prompt(&client, "notify").await;
        let events = turn(&mut client, request).await;
        assert_eq!(
            events[0],
            AcpEvent::Notification {
                method: "_antigravity/hello".into(),
                params: json!({"n": 1}),
            }
        );
    }

    #[tokio::test]
    async fn a_line_over_the_limit_ends_the_connection() {
        let fake = Fake::new();
        let mut client = timed(AcpClient::start_with_limit(
            fake.options(),
            Arc::new(ArkoToolsOnly),
            4096,
        ))
        .await
        .unwrap();
        timed(client.new_session(&fake.work())).await.unwrap();
        prompt(&client, "big 10000").await;
        assert_eq!(timed(client.next_event()).await, None);
        assert!(matches!(client.exit_error(), AcpError::FrameTooLarge));
        assert!(matches!(
            timed(client.new_session(&fake.work())).await,
            Err(AcpError::FrameTooLarge)
        ));
    }

    #[tokio::test]
    async fn an_exit_ends_the_events_and_fails_later_requests() {
        let fake = Fake::new();
        let mut client = session(&fake).await;
        prompt(&client, "exit").await;
        assert_eq!(timed(client.next_event()).await, None);
        // stderr is read on a task of its own, so it may lag behind.
        let mut stderr = String::new();
        for _ in 0..400 {
            if let AcpError::Exited { stderr: tail } = client.exit_error() {
                stderr = tail;
            }
            if stderr.contains("fake agy crashed") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(stderr.contains("fake agy crashed"), "{stderr}");
        assert!(matches!(
            timed(client.new_session(&fake.work())).await,
            Err(AcpError::Exited { .. })
        ));
        assert!(matches!(
            timed(client.prompt("s1", vec![text_block("reply")])).await,
            Err(AcpError::Exited { .. })
        ));
    }

    #[tokio::test]
    async fn another_protocol_version_is_refused() {
        let fake = Fake::new();
        let mut options = fake.options();
        options.env.push(("FAKE_AGY_PROTOCOL".into(), "2".into()));
        let error = timed(AcpClient::start(options, Arc::new(ArkoToolsOnly)))
            .await
            .err()
            .expect("version 2 is refused");
        assert!(matches!(error, AcpError::Protocol(_)), "{error}");
    }

    #[tokio::test]
    async fn an_error_reply_names_the_method() {
        let fake = Fake::new();
        let mut options = fake.options();
        options.env.push(("FAKE_AGY_BROKEN".into(), "1".into()));
        let client = start_with(options, Arc::new(ArkoToolsOnly)).await;
        let error = timed(client.new_session(&fake.work())).await.err().unwrap();
        assert!(
            matches!(&error, AcpError::Rpc { method, error } if method == "session/new" && error.code == -32603),
            "{error}"
        );
        assert_eq!(
            error.to_string(),
            "Antigravity refused `session/new`. Internal error (code -32603)"
        );
    }

    #[tokio::test]
    async fn a_missing_program_is_a_spawn_error() {
        let dir = tempfile::tempdir().unwrap();
        let options = AcpOptions::new(dir.path().join("no-agy"), dir.path(), dir.path());
        let error = AcpClient::start(options, Arc::new(ArkoToolsOnly))
            .await
            .err()
            .unwrap();
        assert!(
            matches!(&error, AcpError::Spawn { source, .. } if source.kind() == io::ErrorKind::NotFound),
            "{error}"
        );
    }

    /// Many requests in flight at once, with agy's replies read on another
    /// thread, each get their own reply.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_get_their_own_replies() {
        const LEVELS: [&str; 4] = ["low", "medium", "high", "xhigh"];
        let fake = Fake::new();
        let client = session(&fake).await;
        for _ in 0..5 {
            let requests = (0..100).map(|index| {
                let client = &client;
                async move {
                    let level = LEVELS[index % LEVELS.len()];
                    let settings = client
                        .set_config_option("s1", "thought_level", level)
                        .await
                        .unwrap();
                    let current = settings
                        .iter()
                        .find(|option| option.id == "thought_level")
                        .and_then(|option| option.current_value.clone());
                    assert_eq!(current.as_deref(), Some(level));
                }
            });
            timed(futures::future::join_all(requests)).await;
        }
    }
}
