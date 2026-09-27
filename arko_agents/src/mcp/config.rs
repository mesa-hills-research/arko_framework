use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::ArkoAgentsError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Top-level MCP configuration containing one or more server definitions.
///
/// # Example JSON
///
/// ```json
/// {
///   "mcpServers": [
///     {
///       "name": "filesystem",
///       "command": "npx",
///       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
///       "env": {}
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpConfig {
    pub mcp_servers: Vec<McpServerConfig>,
}

impl McpConfig {
    pub fn load_from_path(path: &Path) -> Result<Self, ArkoAgentsError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            })?;
        serde_json::from_str(&content).map_err(ArkoAgentsError::Json)
    }
}
