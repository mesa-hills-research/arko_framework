//! Tools the host serves to Claude Code in-process. Each [`ToolServer`] is an
//! MCP server (`"type": "sdk"` in the MCP configuration) whose JSON-RPC
//! messages travel over the control protocol, so no separate process or port
//! is needed. The model sees its tools as `mcp__<server>__<tool>`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

/// A tool served to Claude Code.
#[async_trait]
pub trait ClaudeTool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    /// JSON Schema of the arguments.
    fn input_schema(&self) -> Value;
    async fn call(&self, arguments: Value) -> ToolOutput;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }
}

/// An in-process MCP server with a fixed set of tools.
pub struct ToolServer {
    name: String,
    version: String,
    tools: Vec<Arc<dyn ClaudeTool>>,
}

impl fmt::Debug for ToolServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolServer")
            .field("name", &self.name)
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ToolServer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "1.0.0".into(),
            tools: Vec::new(),
        }
    }

    pub fn tool(mut self, tool: Arc<dyn ClaudeTool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The name the model and permission rules use for `tool`.
    pub fn qualified_name(&self, tool: &str) -> String {
        format!("mcp__{}__{}", self.name, tool)
    }

    /// Answers one JSON-RPC message; `None` for a notification.
    pub(super) async fn handle(&self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned()?;
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": message
                    .pointer("/params/protocolVersion")
                    .cloned()
                    .unwrap_or_else(|| json!("2025-06-18")),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": self.version},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({
                "tools": self
                    .tools
                    .iter()
                    .map(|tool| json!({
                        "name": tool.name(),
                        "description": tool.description(),
                        "inputSchema": tool.input_schema(),
                    }))
                    .collect::<Vec<_>>(),
            })),
            "tools/call" => {
                let name = message.pointer("/params/name").and_then(Value::as_str);
                let arguments = message
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match self
                    .tools
                    .iter()
                    .find(|tool| Some(tool.name().as_str()) == name)
                {
                    Some(tool) => {
                        let output = tool.call(arguments).await;
                        Ok(json!({
                            "content": [{"type": "text", "text": output.text}],
                            "isError": output.is_error,
                        }))
                    }
                    None => Err((-32602, format!("unknown tool: {}", name.unwrap_or("")))),
                }
            }
            _ => Err((-32601, format!("{method} is not supported"))),
        };
        Some(match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, message)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
            }
        })
    }
}
