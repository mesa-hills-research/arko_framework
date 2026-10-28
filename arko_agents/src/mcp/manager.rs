use std::collections::VecDeque;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arko_ai::process::NoConsoleWindow;
use arko_ai::process_tree::ProcessTree;
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
// Only the tests that run a shell script read the report back (Unix only).
// The warning carries it otherwise.
#[cfg_attr(not(all(test, unix)), allow(dead_code))]
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
///
/// The server's whole process tree goes with it. What the server started is
/// killed when the server is killed and when it exits, and also if the task
/// is dropped, as when its runtime ends.
fn watch_server(
    name: String,
    mut child: Child,
    mut tree: ProcessTree,
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
                    None => {
                        tree.kill();
                        match child.kill().await {
                            Ok(()) => child.wait().await,
                            Err(error) => Err(error),
                        }
                    }
                };
                match status {
                    Ok(status) => log::info!("MCP server '{name}' stopped ({status})."),
                    Err(error) => log::info!("MCP server '{name}' stopped: {error}"),
                }
                None
            }
            status = child.wait() => Some(status),
        };
        // What the server started must not outlive it, and one that keeps
        // the stderr pipe open would delay the report below.
        tree.kill();
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
/// A server runs with the programs it starts in a process group of its own
/// (a job object on Windows), so stopping it ends the whole tree. A launcher
/// such as `npx`, `uvx` or `sh -c` doesn't leave the real server running.
///
/// [`shutdown`](Self::shutdown) closes each connection and gives the server
/// three seconds to exit before killing it and what it started. Dropping the
/// manager kills them at once. A host that is killed outright, or ends with
/// `std::process::exit`, runs neither. On Windows the job object still ends
/// the servers. On Unix they end when they notice their input has closed, so
/// one that ignores that keeps running.
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
        // The server and what it starts are one process tree (see
        // `ProcessTree`). A Linux `PR_SET_PDEATHSIG` would also end a server
        // when the host is killed outright, but it fires when the thread that
        // started the server exits, not the host, and reaches only the
        // launcher, not the programs behind it.
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .no_console_window();
        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let spawn_error = |error: std::io::Error| {
            ArkoAgentsError::Runtime(format!(
                "Failed to spawn MCP server '{}' (command: '{}'): {}",
                config.name, config.command, error
            ))
        };
        let (mut child, tree) = ProcessTree::spawn(&mut cmd).map_err(spawn_error)?;
        let (Some(stdout), Some(stdin)) = (child.stdout.take(), child.stdin.take()) else {
            return Err(spawn_error(std::io::Error::other(
                "the server's standard input or output is not available",
            )));
        };
        let stderr = child.stderr.take();
        let (stop, stopped) = oneshot::channel();
        let watcher = watch_server(config.name.clone(), child, tree, stderr, stopped);

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
    /// killed with everything it started. Dropping the manager instead kills
    /// the servers at once.
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
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use super::*;

    fn server(script: &str) -> (Child, ProcessTree, Option<ChildStderr>) {
        let mut command = Command::new("sh");
        command
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let (mut child, tree) = ProcessTree::spawn(&mut command).unwrap();
        let stderr = child.stderr.take();
        (child, tree, stderr)
    }

    /// A server that exits on its own is reported with its exit status and
    /// last stderr lines.
    #[tokio::test]
    async fn a_server_that_exits_is_reported_with_its_status_and_stderr() {
        let (child, tree, stderr) =
            server("echo starting >&2; echo 'fatal: no config' >&2; exit 3");
        let (_stop, stopped) = oneshot::channel();
        let exit = watch_server("files".into(), child, tree, stderr, stopped)
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
        let (child, tree, stderr) = server(
            "printf 'bad \\377 byte\\n' >&2; i=0; while [ $i -lt 2000 ]; do echo more output $i >&2; i=$((i+1)); done; exit 4",
        );
        let (_stop, stopped) = oneshot::channel();
        let exit = watch_server("files".into(), child, tree, stderr, stopped)
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
        let (child, tree, stderr) =
            server("i=0; while [ $i -lt 30 ]; do echo line $i >&2; i=$((i+1)); done");
        let (_stop, stopped) = oneshot::channel();
        let exit = watch_server("files".into(), child, tree, stderr, stopped)
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
            let (child, tree, stderr) = server("echo ready >&2; exec sleep 30");
            let (stop, stopped) = oneshot::channel();
            let watcher = watch_server("files".into(), child, tree, stderr, stopped);
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

    const STUB: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mcp_stub_server.py"
    );

    /// A server behind a `sh -c` launcher, as `npx` or `uvx` would start it.
    /// The launcher starts a program that ignores the end of its input,
    /// records the program's process ID in the returned file, and runs the
    /// stub in its own place. With `ignore_eof` the stub keeps running too
    /// when its input ends.
    fn launched_stub(ignore_eof: bool) -> (McpConfig, PathBuf) {
        let python = std::env::var("ARKO_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
        let pid_file = std::env::temp_dir().join(format!("arko-mcp-{}.pid", uuid::Uuid::new_v4()));
        let env = if ignore_eof {
            HashMap::from([("MCP_STUB_IGNORE_EOF".to_string(), "1".to_string())])
        } else {
            HashMap::new()
        };
        let config = McpConfig {
            mcp_servers: vec![McpServerConfig {
                name: "launched".into(),
                command: "sh".into(),
                args: vec![
                    "-c".into(),
                    "sleep 300 & echo $! > \"$1\"; shift; exec \"$@\"".into(),
                    "sh".into(),
                    pid_file.display().to_string(),
                    python,
                    STUB.into(),
                ],
                env,
            }],
        };
        (config, pid_file)
    }

    /// Kills the program a test started if the test fails with it running.
    struct Program(i32);

    impl Program {
        fn read(pid_file: &Path) -> Self {
            let pid = std::fs::read_to_string(pid_file).unwrap();
            let _ = std::fs::remove_file(pid_file);
            Self(pid.trim().parse().unwrap())
        }

        /// Whether the program has ended. A killed one that its parent hasn't
        /// collected yet counts as ended.
        fn has_ended(&self) -> bool {
            let pid = nix::unistd::Pid::from_raw(self.0);
            if nix::sys::signal::kill(pid, None).is_err() {
                return true;
            }
            std::fs::read_to_string(format!("/proc/{}/stat", self.0))
                .ok()
                .and_then(|stat| Some(stat.rsplit_once(')')?.1.trim_start().starts_with('Z')))
                .unwrap_or(false)
        }

        async fn ends(&self) {
            tokio::time::timeout(Duration::from_secs(10), async {
                while !self.has_ended() {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("the program the server started is still running");
        }
    }

    impl Drop for Program {
        fn drop(&mut self) {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(self.0),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
    }

    /// Stopping a server ends the programs it started as well, whether the
    /// server exits when its input ends or is killed after the grace period.
    #[tokio::test]
    async fn shutdown_ends_the_programs_a_server_started() {
        for ignore_eof in [false, true] {
            let (config, pid_file) = launched_stub(ignore_eof);
            let manager = McpManager::from_config(&config).await.unwrap();
            let program = Program::read(&pid_file);
            assert!(!program.has_ended());
            manager.shutdown().await;
            program.ends().await;
        }
    }

    /// Dropping the manager ends them too.
    #[tokio::test]
    async fn dropping_the_manager_ends_the_programs_a_server_started() {
        for ignore_eof in [false, true] {
            let (config, pid_file) = launched_stub(ignore_eof);
            let manager = McpManager::from_config(&config).await.unwrap();
            let program = Program::read(&pid_file);
            assert!(!program.has_ended());
            drop(manager);
            program.ends().await;
        }
    }

    /// An open that fails after a server started stops that server and
    /// what it started.
    #[tokio::test]
    async fn a_failed_open_ends_the_programs_of_the_servers_already_started() {
        let (mut config, pid_file) = launched_stub(false);
        config.mcp_servers.push(McpServerConfig {
            name: "missing".into(),
            command: "arko-no-such-mcp-server".into(),
            args: Vec::new(),
            env: HashMap::new(),
        });
        assert!(McpManager::from_config(&config).await.is_err());
        Program::read(&pid_file).ends().await;
    }

    /// A server that exits by itself leaves nothing of what it started.
    #[tokio::test]
    async fn a_server_that_exits_leaves_no_program_it_started() {
        let pid_file = std::env::temp_dir().join(format!("arko-mcp-{}.pid", uuid::Uuid::new_v4()));
        let config = McpConfig {
            mcp_servers: vec![McpServerConfig {
                name: "leaving".into(),
                command: "sh".into(),
                args: vec![
                    "-c".into(),
                    "sleep 300 & echo $! > \"$1\"".into(),
                    "sh".into(),
                    pid_file.display().to_string(),
                ],
                env: HashMap::new(),
            }],
        };
        assert!(McpManager::from_config(&config).await.is_err());
        Program::read(&pid_file).ends().await;
    }

    /// So does a runtime that ends before the manager is dropped, which
    /// leaves no task to stop the server.
    #[test]
    fn an_ended_runtime_takes_the_programs_a_server_started_with_it() {
        let (config, pid_file) = launched_stub(true);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let manager = runtime.block_on(McpManager::from_config(&config)).unwrap();
        let program = Program::read(&pid_file);
        assert!(!program.has_ended());
        drop(runtime);
        let waiting = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        waiting.block_on(program.ends());
        drop(manager);
    }
}
