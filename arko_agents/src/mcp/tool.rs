use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use rmcp::service::{Peer, RoleClient};

use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolOutput, ToolParameterDefinition,
    ToolParameterType, ToolTypeDetails,
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
        let definition = mcp_definition(
            qualified_name,
            &server_name,
            &super::McpToolInfo::from_listed(mcp_tool),
        );

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

/// The definition of `tool` of `server` as a tool named `name`: its
/// description marked with the server's name, its raw input schema, and
/// parallel-safe when the server marks it read-only.
pub(crate) fn mcp_definition(
    name: String,
    server_name: &str,
    tool: &super::McpToolInfo,
) -> ToolDefinition {
    let parameters = extract_parameters_from_schema(&tool.input_schema);
    ToolDefinition {
        name,
        description: format!("[MCP:{}] {}", server_name, tool.description),
        output_based_example: None,
        parameters,
        // As Codex does: only a tool the server marks read-only runs at the
        // same time as other calls ("Correctly implemented MCP servers should
        // tolerate parallel calls to tools that advertise themselves as
        // read-only"); any other runs alone, in call order.
        parallel_safe: tool.read_only,
        tool_type_details: ToolTypeDetails::DeveloperDefined,
        raw_json_schema_override: Some(tool.input_schema.clone()),
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
    /// running as the task stopped. Any other is in doubt.
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

    /// A request to the server, whose response a job can go on awaiting
    /// when the call moves to the background.
    fn background_mode(&self) -> crate::tool::ToolBackground {
        crate::tool::ToolBackground::Detach
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.execute_with_output(context, params)
            .await
            .map(|output| output.text)
    }

    /// The call's text, and its images (MCP `image` content, and embedded
    /// image resources) as the result's images. Other content is described
    /// in the text. A resource's text is included, and audio, other binary
    /// resources and images of a type the providers don't take are named.
    async fn execute_with_output(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        let arguments: serde_json::Map<String, Value> = params.into_iter().collect();
        super::call::call_tool(
            &self.peer,
            &self.server_name,
            &self.remote_tool_name,
            arguments,
        )
        .await
        .map_err(|failure| failure.error)
    }
}
