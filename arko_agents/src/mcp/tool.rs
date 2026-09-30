use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RoleClient};

use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

/// An MCP tool exposed by a remote MCP server, wrapped to implement the
/// `arko_agents::Tool` trait.
///
/// # Limitations
///
/// Because MCP tools execute remotely in an opaque server process,
/// the following `arko_agents` features **cannot** be enforced:
///
/// - **Workspace path policy** – The framework cannot inspect whether
///   a given MCP tool parameter represents a file-system path, nor can it
///   intercept or validate paths before the remote server acts on them.
///   The `is_path` flag on parameter definitions is always `false` for
///   MCP tools.
///
/// - **Output-based tool calling mode** – MCP tools are only supported
///   with native (API-level) tool calling. When MCP tools are registered,
///   the agent configuration should use `ToolCallingMode::Native`.
#[derive(Debug)]
pub struct McpTool {
    server_name: String,
    remote_tool_name: String,
    definition: ToolDefinition,
    peer: Peer<RoleClient>,
}

impl McpTool {
    pub fn new(server_name: String, mcp_tool: &rmcp::model::Tool, peer: Peer<RoleClient>) -> Self {
        let remote_tool_name = mcp_tool.name.to_string();
        let qualified_name = format!("mcp__{}__{}", server_name, remote_tool_name);
        let description = mcp_tool.description.as_deref().unwrap_or("").to_string();

        let raw_schema = Value::Object(mcp_tool.input_schema.as_ref().clone());
        let parameters = extract_parameters_from_schema(&raw_schema);

        // As Codex does: only a tool the server marks read-only runs at the
        // same time as other calls ("Correctly implemented MCP servers should
        // tolerate parallel calls to tools that advertise themselves as
        // read-only"); any other runs alone, in call order.
        let read_only = mcp_tool
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.read_only_hint)
            == Some(true);
        let definition = ToolDefinition {
            name: qualified_name,
            description: format!("[MCP:{}] {}", server_name, description),
            output_based_example: None,
            parameters,
            parallel_safe: read_only,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: Some(raw_schema),
        };

        Self {
            server_name,
            remote_tool_name,
            definition,
            peer,
        }
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn remote_tool_name(&self) -> &str {
        &self.remote_tool_name
    }
}

fn extract_parameters_from_schema(schema: &Value) -> Vec<ToolParameterDefinition> {
    let mut params = Vec::new();

    let properties = match schema.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return params,
    };

    let required_set: std::collections::HashSet<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    for (name, prop_schema) in properties {
        let type_str = prop_schema
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("string");
        let prop_desc = prop_schema
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();

        let param_type = match type_str {
            "boolean" => ToolParameterType::Bool,
            _ => ToolParameterType::String,
        };

        params.push(ToolParameterDefinition {
            name: name.clone(),
            description: prop_desc,
            param_type,
            required: required_set.contains(name),
            is_path: false,
            flag: None,
            default_value: None,
        });
    }

    params
}

#[async_trait]
impl Tool for McpTool {
    /// A tool the server marks read-only runs again when its call was
    /// running as the task stopped; any other is in doubt.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        if self.definition.parallel_safe {
            crate::tool::ToolRecovery::Rerun
        } else {
            crate::tool::ToolRecovery::InDoubt
        }
    }

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let arguments: serde_json::Map<String, Value> = params.into_iter().collect();

        let call_params =
            CallToolRequestParams::new(self.remote_tool_name.clone()).with_arguments(arguments);

        let result = self.peer.call_tool(call_params).await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "MCP tool '{}' on server '{}' call failed: {}",
                self.remote_tool_name, self.server_name, e
            ))
        })?;

        let mut output = String::new();
        for content in &result.content {
            if let Some(text) = content.as_text() {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(&text.text);
            }
        }

        if result.is_error == Some(true) {
            return Err(ToolError::ExecutionFailed(if output.is_empty() {
                format!(
                    "MCP tool '{}' returned an error with no text content",
                    self.remote_tool_name
                )
            } else {
                output
            }));
        }

        if output.is_empty()
            && let Some(ref structured) = result.structured_content
        {
            output = serde_json::to_string_pretty(structured).unwrap_or_default();
        }

        Ok(output)
    }
}
