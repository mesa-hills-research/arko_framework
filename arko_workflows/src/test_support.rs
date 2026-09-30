//! Test support: a local model server speaking the OpenAI Responses
//! streaming protocol, and helpers for workflows that talk to it.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arko_agents::{
    AgentConfig, AgentLLMConfig, AgentToolConfig, TaskContext, Tool, ToolCallingMode,
    ToolDefinition, ToolError, ToolTypeDetails,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A reply to a model request.
pub(crate) enum Reply {
    /// A 200 response streaming these server-sent events.
    Stream(String),
    /// An error status, with an optional `Retry-After` in seconds.
    Status {
        status: u16,
        retry_after: Option<u64>,
        body: String,
    },
}

impl Reply {
    fn http(&self) -> String {
        match self {
            Self::Stream(stream) => format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{stream}",
                stream.len()
            ),
            Self::Status {
                status,
                retry_after,
                body,
            } => format!(
                "HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                retry_after.map_or(String::new(), |seconds| format!(
                    "Retry-After: {seconds}\r\n"
                )),
                body.len()
            ),
        }
    }
}

/// Answers each request (its index and body), or `None` to keep it waiting
/// until the server is dropped.
pub(crate) type Respond = dyn Fn(usize, String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Reply>> + Send>>
    + Send
    + Sync;

pub(crate) struct ResponsesServer {
    url: String,
    requests: Arc<AtomicUsize>,
    bodies: Arc<parking_lot::Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<()>,
}

impl ResponsesServer {
    pub async fn start(
        respond: impl Fn(usize, &str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self::start_async(move |index, body| {
            let reply = respond(index, &body).map(Reply::Stream);
            Box::pin(async move { reply })
        })
        .await
    }

    /// Like [`start`](Self::start), with replies that may wait for something.
    pub async fn start_async(
        respond: impl Fn(
            usize,
            String,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Reply>> + Send>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let respond: Arc<Respond> = Arc::new(respond);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let bodies = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let counter = requests.clone();
        let recorded = bodies.clone();
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            while let Ok((mut socket, _)) = listener.accept().await {
                let counter = counter.clone();
                let recorded = recorded.clone();
                let respond = respond.clone();
                connections.spawn(async move {
                    let Some(body) = read_request(&mut socket).await else {
                        return;
                    };
                    let index = {
                        let mut bodies = recorded.lock();
                        bodies.push(body.clone());
                        counter.fetch_add(1, Ordering::SeqCst)
                    };
                    let Some(reply) = respond(index, body).await else {
                        // Hold the connection until the server is dropped.
                        return std::future::pending::<()>().await;
                    };
                    let _ = socket.write_all(reply.http().as_bytes()).await;
                });
            }
        });
        Self {
            url,
            requests,
            bodies,
            task,
        }
    }

    /// The bodies of the requests received so far, in order.
    pub fn bodies(&self) -> Vec<String> {
        self.bodies.lock().clone()
    }

    /// Requests received in full so far.
    pub fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    /// Waits until `count` requests were received.
    pub async fn wait_for_requests(&self, count: usize) {
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while self.requests() < count {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("expected {count} requests, got {}", self.requests()));
    }

    pub fn client(&self) -> arko_ai::LLMClient {
        arko_ai::LLMClient::new().with_openai_client(
            arko_ai::providers::openai::OpenAIClient::new("test-key").with_base_url(&self.url),
        )
    }
}

impl Drop for ResponsesServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<String> {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 8192];
        let n = socket.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..n]);
        let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
        let length: usize = head
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .map_or(0, |value| value.trim().parse().unwrap());
        if bytes.len() >= end + 4 + length {
            return Some(String::from_utf8_lossy(&bytes[end + 4..end + 4 + length]).into_owned());
        }
    }
}

/// A streamed model turn calling tool `name` with `arguments`.
pub(crate) fn tool_call_turn(call_id: &str, name: &str, arguments: Value) -> String {
    let item = json!({
        "type": "function_call", "id": format!("fc_{call_id}"), "call_id": format!("call_{call_id}"),
        "name": name, "arguments": arguments.to_string(), "status": "completed",
    });
    let completed = json!({"type": "response.completed", "response": {
        "id": format!("resp_{call_id}"), "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-5.6-luna", "output": [], "usage": {"input_tokens": 10, "output_tokens": 5},
        "metadata": {},
    }});
    [
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        completed,
    ]
    .iter()
    .map(|event| format!("data: {event}\n\n"))
    .collect()
}

/// A streamed model turn answering with `text` and no tool call.
pub(crate) fn text_turn(id: &str, text: &str) -> String {
    let item = json!({
        "type": "message", "id": format!("msg_{id}"), "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    });
    let completed = json!({"type": "response.completed", "response": {
        "id": format!("resp_{id}"), "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-5.6-luna", "output": [], "usage": {"input_tokens": 10, "output_tokens": 5},
        "metadata": {},
    }});
    [
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": text}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        completed,
    ]
    .iter()
    .map(|event| format!("data: {event}\n\n"))
    .collect()
}

/// A streamed model turn completing the task with `summary`.
pub(crate) fn complete_turn(call_id: &str, summary: &str) -> String {
    tool_call_turn(call_id, "complete_task", json!({ "summary": summary }))
}

/// A native-mode agent that runs every tool call without approval, with
/// display name `display_name`.
pub(crate) fn agent_config(display_name: &str) -> AgentConfig {
    AgentConfig::builder(
        display_name.to_lowercase().replace(' ', "-"),
        display_name.to_string(),
        AgentLLMConfig {
            model: arko_ai::Model::openai_gpt_5_6_luna(),
            generation_config: Default::default(),
        },
    )
    .role_description("test agent".to_string())
    .instructions("test instructions".to_string())
    .tool_config(AgentToolConfig {
        default_auto_approve_tool_use: true,
        tool_calling_mode: ToolCallingMode::Native,
        ..AgentToolConfig::default()
    })
    .build()
    .unwrap()
}

/// A fresh, empty workspace directory, removed on drop.
pub(crate) struct Workspace(pub std::path::PathBuf);

impl Workspace {
    pub fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("arko-workflow-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A tool that counts its calls and, when `block` is set, never returns.
#[derive(Debug, Default)]
pub(crate) struct ProbeTool {
    pub name: &'static str,
    pub calls: AtomicUsize,
    pub block: bool,
    pub started: tokio::sync::Notify,
    /// Blocked calls whose future was dropped (their task was aborted).
    pub dropped: AtomicUsize,
    /// How a call running when the workflow stopped is recovered.
    pub recovery: arko_agents::ToolRecovery,
}

impl ProbeTool {
    /// A blocking probe is stuck: a stop doesn't interrupt it
    /// ([`Tool::interruptible`]), so its agent must be aborted.
    pub fn new(name: &'static str, block: bool) -> Arc<Self> {
        Arc::new(Self {
            name,
            block,
            ..Self::default()
        })
    }

    /// A probe whose calls running when the workflow stopped are recovered
    /// by `recovery`.
    pub fn recovered_by(
        name: &'static str,
        block: bool,
        recovery: arko_agents::ToolRecovery,
    ) -> Arc<Self> {
        Arc::new(Self {
            name,
            block,
            recovery,
            ..Self::default()
        })
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl Tool for ProbeTool {
    fn interruptible(&self) -> bool {
        !self.block
    }

    fn recovery_mode(&self) -> arko_agents::ToolRecovery {
        self.recovery
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.to_string(),
            description: "Test probe.".to_string(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        if self.block {
            struct Dropped<'a>(&'a AtomicUsize);
            impl Drop for Dropped<'_> {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
            let _dropped = Dropped(&self.dropped);
            std::future::pending::<()>().await;
        }
        Ok("probed".to_string())
    }
}
