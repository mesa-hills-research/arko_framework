use std::collections::VecDeque;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::service::{RoleClient, RunningService};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, Command};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use super::config::{McpConfig, McpServerConfig};
use super::tool::McpTool;
use crate::error::ArkoAgentsError;
use crate::tool::{Tool, ToolRegistry};

/// How many of a server's last stderr lines its exit warning shows.
const STDERR_LINES_KEPT: usize = 20;
/// How long a stopping server gets to exit on its own before it is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);
/// How long an exited server's remaining stderr is read before its exit is
/// reported.
const STDERR_DRAIN: Duration = Duration::from_millis(250);

struct McpServerHandle {
    name: String,
    service: RunningService<RoleClient, ()>,
    /// Sent to stop the server process gracefully. Dropped, it is killed.
    stop: Option<oneshot::Sender<()>>,
    watcher: Option<JoinHandle<Option<ServerExit>>>,
}

impl McpServerHandle {
    /// Stops the process (see [`watch_server`]) and closes the connection,
    /// then waits until the process has exited. The watcher is told first,
    /// so the exit that closing the connection causes isn't reported as an
    /// unexpected one.
    async fn stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Err(e) = self.service.close().await {
            log::warn!("Error closing MCP server '{}': {:?}", self.name, e);
        }
        if let Some(watcher) = self.watcher.take() {
            let _ = watcher.await;
        }
    }
}

/// A server process that exited on its own (not stopped by the manager):
/// its exit status and last stderr lines, reported in a warning.
#[derive(Debug)]
// Only tests read the report back; the warning carries it otherwise.
#[cfg_attr(not(test), allow(dead_code))]
struct ServerExit {
    status: std::io::Result<ExitStatus>,
    stderr_tail: Vec<String>,
}

/// Watches a server process: logs each stderr line at info level, tagged
/// with the server's name, and when the process exits on its own, logs one
/// warning with its exit status and last stderr lines (and returns them).
/// `stop` stops the process: sent, it gets [`SHUTDOWN_GRACE`] to exit before
/// it is killed. Dropped, it is killed at once. A stopped process's exit is
/// logged at info level only.
fn watch_server(
    name: String,
    mut child: Child,
    stderr: Option<ChildStderr>,
    mut stop: oneshot::Receiver<()>,
) -> JoinHandle<Option<ServerExit>> {
    let tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_LINES_KEPT)));
    let reader = stderr.map(|stderr| {
        let (name, tail) = (name.clone(), tail.clone());
        tokio::spawn(async move {
            // Read until the pipe closes, whatever the bytes: a reader that
            // stopped early would close the pipe, and the server's next
            // write to stderr would fail.
            let mut reader = BufReader::new(stderr);
            let mut bytes = Vec::new();
            loop {
                bytes.clear();
                match reader.read_until(b'\n', &mut bytes).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let line = String::from_utf8_lossy(&bytes);
                        let line = line.trim_end_matches(['\n', '\r']).to_string();
                        log::info!("MCP server '{name}' stderr: {line}");
                        let mut tail = tail.lock().unwrap();
                        if tail.len() == STDERR_LINES_KEPT {
                            tail.pop_front();
                        }
                        tail.push_back(line);
                    }
                    Err(error) => {
                        log::info!("MCP server '{name}' stderr could not be read: {error}");
                        break;
                    }
                }
            }
        })
    });
    tokio::spawn(async move {
        let exited = tokio::select! {
            biased;
            graceful = &mut stop => {
                let status = if graceful.is_ok() {
                    tokio::time::timeout(SHUTDOWN_GRACE, child.wait()).await.ok()
                } else {
                    None
                };
                let status = match status {
                    Some(status) => status,
                    None => match child.kill().await {
                        Ok(()) => child.wait().await,
                        Err(error) => Err(error),
                    },
                };
                match status {
                    Ok(status) => log::info!("MCP server '{name}' stopped ({status})."),
                    Err(error) => log::info!("MCP server '{name}' stopped: {error}"),
                }
                None
            }
            status = child.wait() => Some(status),
        };
        let status = exited?;
        if let Some(reader) = reader {
            let _ = tokio::time::timeout(STDERR_DRAIN, reader).await;
        }
        let stderr_tail: Vec<String> = tail.lock().unwrap().iter().cloned().collect();
        let described = match &status {
            Ok(status) => status.to_string(),
            Err(error) => format!("exit status unavailable: {error}"),
        };
        if stderr_tail.is_empty() {
            log::warn!("MCP server '{name}' exited ({described}). It wrote nothing to stderr.");
        } else {
            log::warn!(
                "MCP server '{name}' exited ({described}). Its last stderr lines:\n{}",
                stderr_tail.join("\n")
            );
        }
        Some(ServerExit {
            status,
            stderr_tail,
        })
    })
}

/// Manages the lifecycle of one or more MCP server connections.
///
/// Created from an [`McpConfig`], the manager spawns each configured server
/// as a child process, performs the MCP handshake, discovers tools, and wraps
/// them as [`McpTool`] instances that implement `arko_agents::Tool`.
///
/// Each server's stderr goes to the log at info level, a line at a time,
/// tagged with the server's name. A server that exits while the manager
/// still uses it is reported once, with a warning giving its exit status
/// and last stderr lines. Its tools then fail with "Transport closed".
///
/// [`shutdown`](Self::shutdown) closes each connection and gives the server
/// three seconds to exit before killing it. Dropping the manager kills the
/// servers at once.
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
                    for server in &mut servers {
                        server.stop().await;
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
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let spawn_error = |error: std::io::Error| {
            ArkoAgentsError::Runtime(format!(
                "Failed to spawn MCP server '{}' (command: '{}'): {}",
                config.name, config.command, error
            ))
        };
        let mut child = cmd.spawn().map_err(spawn_error)?;
        let (Some(stdout), Some(stdin)) = (child.stdout.take(), child.stdin.take()) else {
            return Err(spawn_error(std::io::Error::other(
                "the server's standard input or output is not available",
            )));
        };
        let stderr = child.stderr.take();
        let (stop, stopped) = oneshot::channel();
        let watcher = watch_server(config.name.clone(), child, stderr, stopped);

        // Should the handshake fail, dropping `stop` kills the server.
        let service: RunningService<RoleClient, ()> = ().serve((stdout, stdin)).await.map_err(|e| {
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
            stop: Some(stop),
            watcher: Some(watcher),
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

    /// Gracefully shuts down every MCP server connection. Each server gets
    /// three seconds to exit once its connection closes, and is then
    /// killed. Dropping the manager instead kills the servers at once.
    pub async fn shutdown(mut self) {
        for server in &mut self.servers {
            log::info!("Shutting down MCP server '{}'", server.name);
            server.stop().await;
        }
        self.servers.clear();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn server(script: &str) -> (Child, Option<ChildStderr>) {
        let mut child = Command::new("sh")
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let stderr = child.stderr.take();
        (child, stderr)
    }

    /// A server that exits on its own is reported with its exit status and
    /// last stderr lines.
    #[tokio::test]
    async fn a_server_that_exits_is_reported_with_its_status_and_stderr() {
        let (child, stderr) = server("echo starting >&2; echo 'fatal: no config' >&2; exit 3");
        let (_stop, stopped) = oneshot::channel();
        let exit = watch_server("files".into(), child, stderr, stopped)
            .await
            .unwrap()
            .expect("an exit on its own is reported");
        assert_eq!(exit.status.unwrap().code(), Some(3));
        assert_eq!(exit.stderr_tail, ["starting", "fatal: no config"]);
    }

    /// Bytes that aren't UTF-8 don't stop the reading: the server can keep
    /// writing to stderr, and its lines are logged with replacements.
    #[tokio::test]
    async fn stderr_that_is_not_utf8_is_still_read() {
        let (child, stderr) = server(
            "printf 'bad \\377 byte\\n' >&2; i=0; while [ $i -lt 2000 ]; do echo more output $i >&2; i=$((i+1)); done; exit 4",
        );
        let (_stop, stopped) = oneshot::channel();
        let exit = watch_server("files".into(), child, stderr, stopped)
            .await
            .unwrap()
            .unwrap();
        // The server kept writing after the bad byte, and exited on its own.
        assert_eq!(exit.status.unwrap().code(), Some(4));
        assert_eq!(
            exit.stderr_tail.last().map(String::as_str),
            Some("more output 1999")
        );
    }

    /// Only the last lines are kept.
    #[tokio::test]
    async fn only_the_last_stderr_lines_are_kept() {
        let (child, stderr) =
            server("i=0; while [ $i -lt 30 ]; do echo line $i >&2; i=$((i+1)); done");
        let (_stop, stopped) = oneshot::channel();
        let exit = watch_server("files".into(), child, stderr, stopped)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(exit.stderr_tail.len(), STDERR_LINES_KEPT);
        assert_eq!(exit.stderr_tail.last().map(String::as_str), Some("line 29"));
    }

    /// A server the manager stops, gracefully or by dropping it, is not
    /// reported as an exit.
    #[tokio::test]
    async fn a_stopped_server_is_not_reported() {
        for graceful in [true, false] {
            let (child, stderr) = server("echo ready >&2; exec sleep 30");
            let (stop, stopped) = oneshot::channel();
            let watcher = watch_server("files".into(), child, stderr, stopped);
            if graceful {
                stop.send(()).unwrap();
            } else {
                drop(stop);
            }
            let report = tokio::time::timeout(Duration::from_secs(10), watcher)
                .await
                .expect("a stopped server is killed after the grace period")
                .unwrap();
            assert!(report.is_none());
        }
    }
}
