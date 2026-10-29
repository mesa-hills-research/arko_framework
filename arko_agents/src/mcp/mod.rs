//! MCP servers over stdio. [`McpHub`] keeps a session's servers, which
//! agents reach through one tool, [`McpBridgeTool`]. [`McpManager`] starts
//! a fixed set of servers and wraps each of their tools as a tool of its
//! own ([`McpTool`]).

mod bridge;
mod call;
mod config;
mod connection;
mod hub;
mod manager;
mod pool;
mod tool;

pub use bridge::{
    AddServer, AllServers, MCP_TOOL_NAME, McpAccess, McpBridgeTool, McpChanges, McpScope,
};
pub use config::{McpConfig, McpServerConfig};
pub use hub::{
    DEFAULT_START_TIMEOUT, McpHub, McpToolInfo, ServerFacts, ServerState, direct_tool_name,
};
pub use manager::McpManager;
pub use tool::McpTool;

#[cfg(test)]
mod tests;
