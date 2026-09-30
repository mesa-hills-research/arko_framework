use std::collections::HashMap;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;

use super::options::{ClaudeOptions, HookCallback, PermissionMode};
use super::protocol::{Message, PermissionHandler, PermissionRequest, ResultMessage};
use super::tools::ToolServer;

/// How long the CLI gets to answer `initialize`; it waits for MCP servers to
/// start first.
const INITIALIZE_TIMEOUT: Duration = Duration::from_secs(90);
const CONTROL_TIMEOUT: Duration = Duration::from_secs(60);
/// How long `close` waits for the CLI to exit after stdin closes.
const EXIT_GRACE: Duration = Duration::from_secs(10);
/// How much of the CLI's stderr to keep for error messages.
const STDERR_TAIL: usize = 16 * 1024;

/// Variables a Claude Code session sets for the programs it runs. A CLI
/// started by such a program would otherwise treat itself as nested in that
/// session: adopt its session ID, report through its messaging socket, and
/// take its effort. Authentication variables are kept.
const PARENT_SESSION_VARIABLES: &[&str] = &[
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_REMOTE_SESSION_ID",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_TEE_SDK_STDOUT",
    "CLAUDE_CODE_SYNC_SESSION_REFS",
    "CLAUDE_CODE_DIAGNOSTICS_FILE",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
];

#[derive(Debug, thiserror::Error)]
pub enum ClaudeError {
    #[error("could not start `{program}`: {source}")]
    Spawn {
        program: String,
        source: std::io::Error,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("Claude Code exited{}", if .stderr.is_empty() { String::new() } else { format!(": {}", .stderr) })]
    Exited { stderr: String },
    #[error("Claude Code rejected `{subtype}`: {message}")]
    Control { subtype: String, message: String },
    #[error("Claude Code did not answer `{0}` in time")]
    Timeout(String),
}

/// The messages of one turn, ending with its result.
#[derive(Debug, Clone)]
pub struct Turn {
    pub messages: Vec<Message>,
    pub result: ResultMessage,
}

type Pending = Arc<StdMutex<HashMap<String, oneshot::Sender<Result<Value, String>>>>>;

/// What answers the CLI's own control requests.
struct Handlers {
    permission: Option<Arc<dyn PermissionHandler>>,
    hooks: HashMap<String, Arc<dyn HookCallback>>,
    servers: HashMap<String, Arc<ToolServer>>,
}

/// A running Claude Code process holding one conversation.
pub struct ClaudeSession {
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    child: Child,
    messages: mpsc::UnboundedReceiver<Message>,
    pending: Pending,
    counter: AtomicU64,
    session_id: Arc<StdMutex<Option<String>>>,
    stderr: Arc<StdMutex<String>>,
    initialization: Value,
    reader: JoinHandle<()>,
}

impl ClaudeSession {
    /// Starts the CLI and completes the `initialize` handshake.
    pub async fn start(options: ClaudeOptions) -> Result<Self, ClaudeError> {
        let launch = super::locate::launch(&options.program());
        let program = launch.program.clone();
        let mut command = Command::new(&program);
        command
            .args(&launch.prefix_args)
            .args(options.arguments())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for variable in PARENT_SESSION_VARIABLES {
            command.env_remove(variable);
        }
        command.envs(&options.env);
        if let Some(directory) = &options.working_directory {
            command.current_dir(directory);
        }
        let mut child = command.spawn().map_err(|source| ClaudeError::Spawn {
            program: program.display().to_string(),
            source,
        })?;
        let stdin = Arc::new(Mutex::new(child.stdin.take()));
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr_pipe = child.stderr.take().expect("stderr is piped");

        let stderr = Arc::new(StdMutex::new(String::new()));
        tokio::spawn(collect_stderr(stderr_pipe, stderr.clone()));

        let mut hooks = HashMap::new();
        let mut hook_config = serde_json::Map::new();
        for (index, hook) in options.hooks.iter().enumerate() {
            let callback_id = format!("hook_{index}");
            hooks.insert(callback_id.clone(), hook.callback.clone());
            let mut matcher = json!({
                "matcher": hook.matcher,
                "hookCallbackIds": [callback_id],
            });
            if let Some(timeout) = hook.timeout_seconds {
                matcher["timeout"] = json!(timeout);
            }
            hook_config
                .entry(hook.event.as_str().to_string())
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .expect("array")
                .push(matcher);
        }
        let handlers = Arc::new(Handlers {
            permission: options.permission_handler.clone(),
            hooks,
            servers: options
                .tool_servers
                .iter()
                .map(|server| (server.name().to_string(), server.clone()))
                .collect(),
        });

        let pending: Pending = Arc::default();
        let session_id = Arc::new(StdMutex::new(None));
        let (sender, messages) = mpsc::unbounded_channel();
        let reader = tokio::spawn(read_stdout(
            stdout,
            stdin.clone(),
            sender,
            pending.clone(),
            handlers,
            session_id.clone(),
        ));

        let mut session = Self {
            stdin,
            child,
            messages,
            pending,
            counter: AtomicU64::new(0),
            session_id,
            stderr,
            initialization: Value::Null,
            reader,
        };
        let hooks = if hook_config.is_empty() {
            Value::Null
        } else {
            Value::Object(hook_config)
        };
        session.initialization = session
            .control(
                json!({"subtype": "initialize", "hooks": hooks}),
                INITIALIZE_TIMEOUT,
            )
            .await?;
        Ok(session)
    }

    /// The CLI's answer to `initialize`: slash commands, models, account.
    pub fn initialization(&self) -> &Value {
        &self.initialization
    }

    /// The conversation's session ID, once a message has named it. Pass it
    /// to [`SessionStart::Resume`](super::SessionStart::Resume) to continue
    /// the conversation in a new process.
    pub fn session_id(&self) -> Option<String> {
        self.session_id.lock().unwrap().clone()
    }

    /// Sends a user message, starting a turn (or queueing it behind the
    /// running one).
    pub async fn send(&self, text: impl Into<String>) -> Result<(), ClaudeError> {
        self.send_content(Value::String(text.into())).await
    }

    /// Sends a user message with content blocks (text, images).
    pub async fn send_content(&self, content: Value) -> Result<(), ClaudeError> {
        let message = json!({
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": null,
            "session_id": self.session_id().unwrap_or_else(|| "default".into()),
        });
        self.write(&message).await
    }

    /// The next message; `None` once the CLI has exited.
    pub async fn next_message(&mut self) -> Option<Message> {
        self.messages.recv().await
    }

    /// Collects messages up to the end of the current turn.
    pub async fn receive_turn(&mut self) -> Result<Turn, ClaudeError> {
        let mut messages = Vec::new();
        while let Some(message) = self.messages.recv().await {
            if let Message::Result(result) = &message {
                let result = result.clone();
                messages.push(message);
                return Ok(Turn { messages, result });
            }
            messages.push(message);
        }
        Err(self.exited())
    }

    /// Stops the running turn; its result still arrives.
    pub async fn interrupt(&self) -> Result<(), ClaudeError> {
        self.control(json!({"subtype": "interrupt"}), CONTROL_TIMEOUT)
            .await
            .map(drop)
    }

    /// Switches the model for later turns; `None` restores the default.
    pub async fn set_model(&self, model: Option<&str>) -> Result<(), ClaudeError> {
        self.control(
            json!({"subtype": "set_model", "model": model}),
            CONTROL_TIMEOUT,
        )
        .await
        .map(drop)
    }

    pub async fn set_permission_mode(&self, mode: PermissionMode) -> Result<(), ClaudeError> {
        self.control(
            json!({"subtype": "set_permission_mode", "mode": mode.as_str()}),
            CONTROL_TIMEOUT,
        )
        .await
        .map(drop)
    }

    /// Ends the conversation: closes stdin so the CLI finishes and saves its
    /// transcript, and kills it if it has not exited within a grace period.
    /// Interrupt a running turn first, or its prompt may be missing from the
    /// transcript.
    pub async fn close(mut self) -> Result<ExitStatus, ClaudeError> {
        self.stdin.lock().await.take();
        let status = match tokio::time::timeout(EXIT_GRACE, self.child.wait()).await {
            Ok(status) => status?,
            Err(_) => {
                self.child.kill().await?;
                self.child.wait().await?
            }
        };
        self.reader.abort();
        Ok(status)
    }

    async fn write(&self, message: &Value) -> Result<(), ClaudeError> {
        write_line(&self.stdin, message)
            .await
            .map_err(|_| self.exited())
    }

    async fn control(&self, request: Value, timeout: Duration) -> Result<Value, ClaudeError> {
        let subtype = request["subtype"].as_str().unwrap_or_default().to_string();
        let id = format!("req_{}", self.counter.fetch_add(1, Ordering::Relaxed) + 1);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id.clone(), sender);
        let message = json!({"type": "control_request", "request_id": id, "request": request});
        if let Err(error) = self.write(&message).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(Ok(response))) => Ok(response),
            Ok(Ok(Err(message))) => Err(ClaudeError::Control { subtype, message }),
            Ok(Err(_)) => Err(self.exited()),
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                Err(ClaudeError::Timeout(subtype))
            }
        }
    }

    fn exited(&self) -> ClaudeError {
        ClaudeError::Exited {
            stderr: self.stderr.lock().unwrap().trim().to_string(),
        }
    }
}

async fn write_line(stdin: &Mutex<Option<ChildStdin>>, message: &Value) -> std::io::Result<()> {
    let mut guard = stdin.lock().await;
    let pipe = guard
        .as_mut()
        .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::BrokenPipe))?;
    let mut line = message.to_string();
    log::trace!(target: WIRE_LOG, "-> {line}");
    line.push('\n');
    pipe.write_all(line.as_bytes()).await?;
    pipe.flush().await
}

/// The log target of the raw stream-json traffic, one line per message
/// (`->` to the CLI, `<-` from it), at trace level:
/// `RUST_LOG=arko_ai::claude_code::wire=trace`.
pub const WIRE_LOG: &str = "arko_ai::claude_code::wire";

async fn collect_stderr(mut pipe: tokio::process::ChildStderr, tail: Arc<StdMutex<String>>) {
    let mut buffer = [0u8; 4096];
    while let Ok(read) = pipe.read(&mut buffer).await {
        if read == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buffer[..read]);
        log::debug!("claude stderr: {}", chunk.trim_end());
        let mut tail = tail.lock().unwrap();
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

/// Routes stdout: control responses to their waiting requests, control
/// requests to the handlers, everything else to the session's messages.
async fn read_stdout(
    stdout: tokio::process::ChildStdout,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    messages: mpsc::UnboundedSender<Message>,
    pending: Pending,
    handlers: Arc<Handlers>,
    session_id: Arc<StdMutex<Option<String>>>,
) {
    let mut lines = BufReader::new(stdout).lines();
    let in_flight: Arc<StdMutex<HashMap<String, JoinHandle<()>>>> = Arc::default();
    while let Ok(Some(line)) = lines.next_line().await {
        log::trace!(target: WIRE_LOG, "<- {line}");
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            if !line.trim().is_empty() {
                log::debug!("claude stdout (not JSON): {line}");
            }
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("control_response") => {
                let response = &value["response"];
                let Some(id) = response["request_id"].as_str() else {
                    continue;
                };
                if let Some(sender) = pending.lock().unwrap().remove(id) {
                    let outcome = if response["subtype"] == "error" {
                        Err(response["error"]
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string())
                    } else {
                        Ok(response.get("response").cloned().unwrap_or(Value::Null))
                    };
                    let _ = sender.send(outcome);
                }
            }
            Some("control_request") => {
                let id = value["request_id"].as_str().unwrap_or_default().to_string();
                let request = value["request"].clone();
                let (handlers, stdin, registry) =
                    (handlers.clone(), stdin.clone(), in_flight.clone());
                let task_id = id.clone();
                let task = tokio::spawn(async move {
                    let response = match answer(&handlers, request).await {
                        Ok(response) => json!({
                            "type": "control_response",
                            "response": {"subtype": "success", "request_id": task_id, "response": response},
                        }),
                        Err(error) => json!({
                            "type": "control_response",
                            "response": {"subtype": "error", "request_id": task_id, "error": error},
                        }),
                    };
                    let _ = write_line(&stdin, &response).await;
                    registry.lock().unwrap().remove(&task_id);
                });
                in_flight.lock().unwrap().insert(id, task);
            }
            Some("control_cancel_request") => {
                // The CLI gave up on the request; it expects no answer.
                if let Some(task) = value["request_id"]
                    .as_str()
                    .and_then(|id| in_flight.lock().unwrap().remove(id))
                {
                    task.abort();
                }
            }
            _ => {
                let message = Message::parse(value);
                if let Some(id) = message.session_id() {
                    *session_id.lock().unwrap() = Some(id.to_string());
                }
                if messages.send(message).is_err() {
                    break;
                }
            }
        }
    }
    // The CLI has exited: waiting control requests fail.
    pending.lock().unwrap().clear();
}

/// Answers one control request from the CLI.
async fn answer(handlers: &Handlers, request: Value) -> Result<Value, String> {
    match request["subtype"].as_str().unwrap_or_default() {
        "can_use_tool" => {
            let Some(handler) = &handlers.permission else {
                return Err("no permission handler is configured".into());
            };
            let request = PermissionRequest::parse(request);
            let input = request.input.clone();
            Ok(handler.decide(request).await.to_response(&input))
        }
        "hook_callback" => {
            let id = request["callback_id"].as_str().unwrap_or_default();
            let callback = handlers
                .hooks
                .get(id)
                .ok_or_else(|| format!("no hook callback {id}"))?;
            let tool_use_id = request["tool_use_id"].as_str().map(str::to_string);
            Ok(callback.call(request["input"].clone(), tool_use_id).await)
        }
        "mcp_message" => {
            let name = request["server_name"].as_str().unwrap_or_default();
            let server = handlers
                .servers
                .get(name)
                .ok_or_else(|| format!("no in-process MCP server {name}"))?;
            // Notifications get no JSON-RPC reply, but the control request
            // carrying one still expects an answer.
            let reply = server
                .handle(&request["message"])
                .await
                .unwrap_or_else(|| json!({"jsonrpc": "2.0", "result": {}}));
            Ok(json!({"mcp_response": reply}))
        }
        other => Err(format!("unsupported control request: {other}")),
    }
}
