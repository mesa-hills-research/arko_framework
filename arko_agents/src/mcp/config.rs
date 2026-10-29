use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::ArkoAgentsError;

/// One MCP server, run over stdio: the program, its arguments and its
/// environment, already expanded.
///
/// The struct is `#[non_exhaustive]`, so fields can be added without
/// breaking callers. Build one with [`McpServerConfig::new`] and set the
/// public fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Whether workers may reach the server through
    /// [`McpBridgeTool::worker`](super::McpBridgeTool::worker). `false` keeps
    /// it to the main agent. Defaults to `true`.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub workers: bool,
    /// Whether one process serves every [`McpHub`](super::McpHub) of the
    /// process that names the same command, arguments and environment, as
    /// sessions open at the same time do. The host's own environment and
    /// working directory count too, as they were when the server was first
    /// started. Only for a server that keeps nothing per client. Defaults to
    /// `false`, a process per hub.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub shared: bool,
    /// What the server is for, in a sentence (such as why an agent added
    /// it). The bridge's `servers` action shows it, or else the first
    /// sentence of the server's own instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The `${VAR}`s the host put into the command, arguments and
    /// environment, as variable names and values. A server can print them
    /// to its stderr, so the log shows `${VAR}` in place of each value, as
    /// it does for the values of `env`. Not saved.
    #[serde(skip)]
    pub expanded: HashMap<String, String>,
}

fn default_true() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

impl McpServerConfig {
    /// A server named `name` that runs `command`, with no arguments or
    /// environment, offered to workers and not shared.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: HashMap::new(),
            workers: true,
            shared: false,
            description: None,
            expanded: HashMap::new(),
        }
    }

    /// This server with `expanded` as the host's [expanded](Self::expanded)
    /// variables.
    pub fn with_expanded(
        mut self,
        expanded: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.expanded = expanded
            .into_iter()
            .map(|(name, value)| (name.into(), value.into()))
            .collect();
        self
    }

    /// This server with `args` as its arguments.
    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// This server with `env` as its environment.
    pub fn with_env(
        mut self,
        env: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.env = env
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_config_saved_before_workers_and_shared_loads_with_their_defaults() {
        let config: McpConfig = serde_json::from_str(
            r#"{"mcpServers": [{"name": "files", "command": "npx", "args": ["-y", "x"]}]}"#,
        )
        .unwrap();
        let server = &config.mcp_servers[0];
        assert!(server.workers);
        assert!(!server.shared);
        assert_eq!(server.description, None);
        assert_eq!(
            server,
            &McpServerConfig::new("files", "npx").with_args(["-y", "x"])
        );
        // The defaults aren't written back.
        let written = serde_json::to_value(server).unwrap();
        assert_eq!(written.get("workers"), None);
        assert_eq!(written.get("shared"), None);
        let mut kept = server.clone();
        kept.workers = false;
        kept.shared = true;
        let written = serde_json::to_value(&kept).unwrap();
        assert_eq!(written["workers"], false);
        assert_eq!(written["shared"], true);
    }
}
