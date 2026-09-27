use std::sync::Arc;

use rmcp::ServiceExt;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::child_process::TokioChildProcess;
use tokio::process::Command;

use super::config::{McpConfig, McpServerConfig};
use super::tool::McpTool;
use crate::error::ArkoAgentsError;
use crate::tool::{Tool, ToolRegistry};

struct McpServerHandle {
    name: String,
    service: RunningService<RoleClient, ()>,
}

/// Manages the lifecycle of one or more MCP server connections.
///
/// Created from an [`McpConfig`], the manager spawns each configured server
/// as a child process, performs the MCP handshake, discovers tools, and wraps
/// them as [`McpTool`] instances that implement `arko_agents::Tool`.
///
/// When `shutdown` is called (or the manager is dropped), every child process
/// is terminated. The `rmcp` transport layer uses platform-appropriate process
/// group handling (via `process-wrap`) to avoid orphaned server processes even
/// on abnormal exit.
pub struct McpManager {
    servers: Vec<McpServerHandle>,
    tools: Vec<Arc<McpTool>>,
}

impl McpManager {
    /// Connects to all servers described in `config`, discovers their tools,
    /// and returns a fully initialised manager.
    ///
    /// If any server fails to start or handshake, previously connected servers
    /// are shut down before the error is returned.
    pub async fn from_config(config: &McpConfig) -> Result<Self, ArkoAgentsError> {
        let mut servers = Vec::new();
        let mut tools = Vec::new();

        for server_config in &config.mcp_servers {
            match Self::connect_server(server_config).await {
                Ok((handle, server_tools)) => {
                    servers.push(handle);
                    tools.extend(server_tools);
                }
                Err(e) => {
                    for mut server in servers {
                        let _ = server.service.close().await;
                    }
                    return Err(e);
                }
            }
        }

        Ok(Self { servers, tools })
    }

    async fn connect_server(
        config: &McpServerConfig,
    ) -> Result<(McpServerHandle, Vec<Arc<McpTool>>), ArkoAgentsError> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args);
        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let transport = TokioChildProcess::new(cmd).map_err(|e| {
            ArkoAgentsError::Runtime(format!(
                "Failed to spawn MCP server '{}' (command: '{}'): {}",
                config.name, config.command, e
            ))
        })?;

        let service: RunningService<RoleClient, ()> = ().serve(transport).await.map_err(|e| {
            ArkoAgentsError::Runtime(format!(
                "Failed to initialise MCP client for server '{}': {}",
                config.name, e
            ))
        })?;

        let mcp_tools = service.peer().list_all_tools().await.map_err(|e| {
            ArkoAgentsError::Runtime(format!(
                "Failed to list tools from MCP server '{}': {}",
                config.name, e
            ))
        })?;

        let peer = service.peer().clone();
        let mut wrapped_tools = Vec::new();

        for mcp_tool in &mcp_tools {
            let tool = Arc::new(McpTool::new(config.name.clone(), mcp_tool, peer.clone()));
            wrapped_tools.push(tool);
        }

        log::info!(
            "Connected to MCP server '{}': {} tool(s) available",
            config.name,
            wrapped_tools.len()
        );
        for tool in &wrapped_tools {
            log::info!("  - {}", tool.definition().name);
        }

        let handle = McpServerHandle {
            name: config.name.clone(),
            service,
        };

        Ok((handle, wrapped_tools))
    }

    /// Returns a slice of all discovered MCP tools across every connected server.
    pub fn tools(&self) -> &[Arc<McpTool>] {
        &self.tools
    }

    /// Registers every discovered MCP tool into the given `ToolRegistry`.
    pub fn register_tools(&self, registry: &mut ToolRegistry) {
        for tool in &self.tools {
            registry.register(tool.clone());
        }
    }

    /// Returns the qualified tool names (e.g. `mcp__filesystem__read_file`)
    /// suitable for insertion into `AgentToolConfig::available_tool_names`.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.definition().name).collect()
    }

    /// Gracefully shuts down every MCP server connection, terminating child
    /// processes. This is called automatically on drop but calling it
    /// explicitly allows awaiting the result.
    pub async fn shutdown(mut self) {
        for server in &mut self.servers {
            log::info!("Shutting down MCP server '{}'", server.name);
            if let Err(e) = server.service.close().await {
                log::warn!("Error closing MCP server '{}': {:?}", server.name, e);
            }
        }
        self.servers.clear();
    }
}
