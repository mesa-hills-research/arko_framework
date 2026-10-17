use async_trait::async_trait;
use serde_json::{Value, json};

/// A message Claude Code writes to stdout, other than control traffic.
/// Every variant keeps the full JSON in `raw`, so fields this client does
/// not model are still available.
#[derive(Debug, Clone)]
pub enum Message {
    System(SystemMessage),
    Assistant(AssistantMessage),
    /// A user turn as Claude Code records it. Tool results arrive here.
    User(UserMessage),
    /// The end of a turn.
    Result(ResultMessage),
    /// A raw API streaming event, with `include_partial_messages`.
    StreamEvent {
        event: Value,
        parent_tool_use_id: Option<String>,
        raw: Value,
    },
    /// Plan usage-limit status (`rate_limit_info`).
    RateLimit {
        info: Value,
        raw: Value,
    },
    Other(Value),
}

#[derive(Debug, Clone)]
pub struct SystemMessage {
    /// `init` (model, tools, MCP servers), `api_retry`, `task_started`, …
    pub subtype: String,
    pub session_id: Option<String>,
    pub raw: Value,
}

#[derive(Debug, Clone)]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub model: Option<String>,
    /// Set when a subagent produced the message.
    pub parent_tool_use_id: Option<String>,
    pub stop_reason: Option<String>,
    pub usage: Option<Value>,
    pub raw: Value,
}

#[derive(Debug, Clone)]
pub struct UserMessage {
    pub content: Vec<ContentBlock>,
    pub parent_tool_use_id: Option<String>,
    pub raw: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text(String),
    /// Reasoning text. Empty when Claude Code omits it.
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Value,
        is_error: bool,
    },
    Other(Value),
}

#[derive(Debug, Clone)]
pub struct ResultMessage {
    /// `success`, `error_max_turns`, `error_during_execution`, …
    pub subtype: String,
    pub is_error: bool,
    pub num_turns: u64,
    pub session_id: String,
    pub duration_ms: u64,
    /// The turn's final text.
    pub result: Option<String>,
    pub total_cost_usd: Option<f64>,
    pub usage: Option<Value>,
    pub stop_reason: Option<String>,
    pub raw: Value,
}

fn text(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn blocks(content: Option<&Value>) -> Vec<ContentBlock> {
    match content {
        Some(Value::String(text)) => vec![ContentBlock::Text(text.clone())],
        Some(Value::Array(items)) => items.iter().map(ContentBlock::parse).collect(),
        _ => Vec::new(),
    }
}

impl ContentBlock {
    fn parse(block: &Value) -> Self {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => ContentBlock::Text(text(block, "text").unwrap_or_default()),
            Some("thinking") => ContentBlock::Thinking(text(block, "thinking").unwrap_or_default()),
            Some("tool_use") => ContentBlock::ToolUse {
                id: text(block, "id").unwrap_or_default(),
                name: text(block, "name").unwrap_or_default(),
                input: block.get("input").cloned().unwrap_or(Value::Null),
            },
            Some("tool_result") => ContentBlock::ToolResult {
                tool_use_id: text(block, "tool_use_id").unwrap_or_default(),
                content: block.get("content").cloned().unwrap_or(Value::Null),
                is_error: block
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            },
            _ => ContentBlock::Other(block.clone()),
        }
    }
}

impl Message {
    pub(super) fn parse(raw: Value) -> Self {
        let message = raw.get("message");
        match raw.get("type").and_then(Value::as_str) {
            Some("system") => Message::System(SystemMessage {
                subtype: text(&raw, "subtype").unwrap_or_default(),
                session_id: text(&raw, "session_id"),
                raw,
            }),
            Some("assistant") => Message::Assistant(AssistantMessage {
                content: blocks(message.and_then(|m| m.get("content"))),
                model: message.and_then(|m| text(m, "model")),
                parent_tool_use_id: text(&raw, "parent_tool_use_id"),
                stop_reason: message.and_then(|m| text(m, "stop_reason")),
                usage: message.and_then(|m| m.get("usage")).cloned(),
                raw,
            }),
            Some("user") => Message::User(UserMessage {
                content: blocks(message.and_then(|m| m.get("content"))),
                parent_tool_use_id: text(&raw, "parent_tool_use_id"),
                raw,
            }),
            Some("result") => Message::Result(ResultMessage {
                subtype: text(&raw, "subtype").unwrap_or_default(),
                is_error: raw
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                num_turns: raw.get("num_turns").and_then(Value::as_u64).unwrap_or(0),
                session_id: text(&raw, "session_id").unwrap_or_default(),
                duration_ms: raw.get("duration_ms").and_then(Value::as_u64).unwrap_or(0),
                result: text(&raw, "result"),
                total_cost_usd: raw.get("total_cost_usd").and_then(Value::as_f64),
                usage: raw.get("usage").cloned(),
                stop_reason: text(&raw, "stop_reason"),
                raw,
            }),
            Some("stream_event") => Message::StreamEvent {
                event: raw.get("event").cloned().unwrap_or(Value::Null),
                parent_tool_use_id: text(&raw, "parent_tool_use_id"),
                raw,
            },
            Some("rate_limit_event") => Message::RateLimit {
                info: raw.get("rate_limit_info").cloned().unwrap_or(Value::Null),
                raw,
            },
            _ => Message::Other(raw),
        }
    }

    /// The session this message belongs to, when it names one.
    pub fn session_id(&self) -> Option<&str> {
        let raw = match self {
            Message::System(m) => &m.raw,
            Message::Assistant(m) => &m.raw,
            Message::User(m) => &m.raw,
            Message::Result(m) => &m.raw,
            Message::StreamEvent { raw, .. } | Message::RateLimit { raw, .. } => raw,
            Message::Other(raw) => raw,
        };
        raw.get("session_id").and_then(Value::as_str)
    }
}

/// A tool call Claude Code asks the host to approve.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub tool_name: String,
    pub input: Value,
    pub tool_use_id: Option<String>,
    /// Set for calls from a subagent.
    pub agent_id: Option<String>,
    pub blocked_path: Option<String>,
    pub decision_reason: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    /// Permission rules Claude Code suggests adding ("always allow …").
    pub suggestions: Vec<Value>,
    pub raw: Value,
}

impl PermissionRequest {
    pub(super) fn parse(raw: Value) -> Self {
        Self {
            tool_name: text(&raw, "tool_name").unwrap_or_default(),
            input: raw.get("input").cloned().unwrap_or(Value::Null),
            tool_use_id: text(&raw, "tool_use_id"),
            agent_id: text(&raw, "agent_id"),
            blocked_path: text(&raw, "blocked_path"),
            decision_reason: text(&raw, "decision_reason"),
            title: text(&raw, "title"),
            description: text(&raw, "description"),
            suggestions: raw
                .get("permission_suggestions")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            raw,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    Allow {
        /// Replaces the tool input.
        updated_input: Option<Value>,
        /// Permission rules to add, such as one of the request's suggestions.
        updated_permissions: Vec<Value>,
    },
    Deny {
        /// Shown to the model.
        message: String,
        /// Also stop the turn.
        interrupt: bool,
    },
}

impl PermissionDecision {
    pub fn allow() -> Self {
        PermissionDecision::Allow {
            updated_input: None,
            updated_permissions: Vec::new(),
        }
    }

    pub fn deny(message: impl Into<String>) -> Self {
        PermissionDecision::Deny {
            message: message.into(),
            interrupt: false,
        }
    }

    pub(super) fn to_response(&self, original_input: &Value) -> Value {
        match self {
            PermissionDecision::Allow {
                updated_input,
                updated_permissions,
            } => {
                let mut response = json!({
                    "behavior": "allow",
                    "updatedInput": updated_input.clone().unwrap_or_else(|| original_input.clone()),
                });
                if !updated_permissions.is_empty() {
                    response["updatedPermissions"] = Value::from(updated_permissions.clone());
                }
                response
            }
            PermissionDecision::Deny { message, interrupt } => {
                let mut response = json!({"behavior": "deny", "message": message});
                if *interrupt {
                    response["interrupt"] = Value::Bool(true);
                }
                response
            }
        }
    }
}

/// Decides tool calls Claude Code would otherwise ask the user about.
#[async_trait]
pub trait PermissionHandler: Send + Sync {
    async fn decide(&self, request: PermissionRequest) -> PermissionDecision;
}
