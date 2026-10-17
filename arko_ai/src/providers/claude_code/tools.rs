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
    /// Base64 images sent after the text, as MCP `image` content. An image
    /// given by URL is sent as a line of text with its URL.
    pub images: Vec<crate::Image>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
            images: Vec::new(),
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
            images: Vec::new(),
        }
    }

    /// The output with `images` after its text.
    pub fn with_images(mut self, images: Vec<crate::Image>) -> Self {
        self.images = images;
        self
    }

    /// The MCP `content` of the result: the text (left out when empty and
    /// there are images), then each image.
    fn content(&self) -> Vec<Value> {
        let mut content = Vec::new();
        if !self.text.is_empty() || self.images.is_empty() {
            content.push(json!({"type": "text", "text": self.text}));
        }
        for image in &self.images {
            content.push(match image {
                crate::Image::Base64 { data, mime_type } => {
                    json!({"type": "image", "data": data, "mimeType": mime_type})
                }
                crate::Image::Url(url) => json!({"type": "text", "text": format!("Image: {url}")}),
            });
        }
        content
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

    /// Answers one JSON-RPC message. `None` for a notification.
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
                            "content": output.content(),
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

#[cfg(test)]
mod tests {
    use super::*;

    struct ImageTool;

    #[async_trait]
    impl ClaudeTool for ImageTool {
        fn name(&self) -> String {
            "image".into()
        }

        fn description(&self) -> String {
            "Returns an image for testing.".into()
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(&self, arguments: Value) -> ToolOutput {
            let output = if arguments["fail"].as_bool() == Some(true) {
                ToolOutput::error("could not render")
            } else {
                ToolOutput::text("rendered")
            };
            output.with_images(vec![crate::Image::from_bytes(b"image bytes", "image/png")])
        }
    }

    #[tokio::test]
    async fn tool_server_emits_image_blocks_and_keeps_the_error_flag() {
        let server = ToolServer::new("test").tool(Arc::new(ImageTool));
        for fail in [false, true] {
            let response = server
                .handle(&json!({
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "image", "arguments": {"fail": fail}},
                }))
                .await
                .unwrap();
            assert_eq!(
                response,
                json!({
                    "jsonrpc": "2.0",
                    "id": 7,
                    "result": {
                        "content": [
                            {"type": "text", "text": if fail { "could not render" } else { "rendered" }},
                            {"type": "image", "mimeType": "image/png", "data": "aW1hZ2UgYnl0ZXM="},
                        ],
                        "isError": fail,
                    },
                })
            );
        }
    }

    #[test]
    fn image_only_results_omit_empty_text_and_url_images_remain_readable() {
        assert_eq!(
            ToolOutput::text("").content(),
            vec![json!({"type": "text", "text": ""})]
        );
        let output = ToolOutput::text("").with_images(vec![
            crate::Image::from_bytes(b"gif", "image/gif"),
            crate::Image::Url("https://example.com/image.png".into()),
        ]);
        assert_eq!(
            output.content(),
            vec![
                json!({"type": "image", "mimeType": "image/gif", "data": "Z2lm"}),
                json!({"type": "text", "text": "Image: https://example.com/image.png"}),
            ]
        );
    }
}
