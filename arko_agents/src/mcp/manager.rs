use std::sync::Arc;

use super::config::{McpConfig, McpServerConfig};
use super::connection::{Connection, OpenError};
use super::tool::McpTool;
use crate::error::ArkoAgentsError;
use crate::tool::{Tool, ToolRegistry};

/// Manages the lifecycle of one or more MCP server connections.
///
/// Created from an [`McpConfig`], the manager spawns each configured server
/// as a child process, performs the MCP handshake, discovers tools, and wraps
/// them as [`McpTool`] instances that implement `arko_agents::Tool`.
/// [`McpHub`](super::McpHub) starts, stops and adds servers while it runs,
/// and asks them for their tools each time.
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
/// one that ignores that keeps running, unless a host about to call
/// `std::process::exit` calls [`kill_mcp_servers`](crate::kill_mcp_servers)
/// first.
pub struct McpManager {
    servers: Vec<Connection>,
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
    ) -> Result<(Connection, Vec<Arc<McpTool>>), ArkoAgentsError> {
        // The error leaves out the command and the server's own words, which
        // can hold the values of its environment. The log has them.
        let connection = Connection::open(config).await.map_err(|error| {
            ArkoAgentsError::Runtime(match error {
                OpenError::Spawn(_) => format!(
                    "Failed to spawn MCP server '{}': {}",
                    config.name,
                    error.reason()
                ),
                OpenError::Handshake { .. } => format!(
                    "Failed to initialise MCP client for server '{}': {}",
                    config.name,
                    error.reason()
                ),
            })
        })?;

        // Should listing fail, dropping the connection kills the server.
        let peer = connection.peer();
        let mcp_tools = peer.list_all_tools().await.map_err(|e| {
            ArkoAgentsError::Runtime(format!(
                "Failed to list tools from MCP server '{}': {}",
                config.name, e
            ))
        })?;

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

        Ok((connection, wrapped_tools))
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
            server.stop().await;
        }
        self.servers.clear();
    }
}

#[cfg(all(test, unix))]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use super::*;

    pub(crate) const STUB: &str = concat!(
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
            mcp_servers: vec![
                McpServerConfig::new("launched", "sh")
                    .with_args([
                        "-c".into(),
                        "sleep 300 & echo $! > \"$1\"; shift; exec \"$@\"".into(),
                        "sh".into(),
                        pid_file.display().to_string(),
                        python,
                        STUB.into(),
                    ])
                    .with_env(env),
            ],
        };
        (config, pid_file)
    }

    /// Whether the process `pid` has ended. A killed one that its parent
    /// hasn't collected yet counts as ended.
    pub(crate) fn has_ended(pid: i32) -> bool {
        if nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_err() {
            return true;
        }
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| Some(stat.rsplit_once(')')?.1.trim_start().starts_with('Z')))
            .unwrap_or(false)
    }

    /// Kills the program a test started if the test fails with it running.
    pub(crate) struct Program(pub(crate) i32);

    impl Program {
        pub(crate) fn read(pid_file: &Path) -> Self {
            let pid = std::fs::read_to_string(pid_file).unwrap();
            let _ = std::fs::remove_file(pid_file);
            Self(pid.trim().parse().unwrap())
        }

        /// Whether the program has ended. A killed one that its parent hasn't
        /// collected yet counts as ended.
        pub(crate) fn has_ended(&self) -> bool {
            has_ended(self.0)
        }

        pub(crate) async fn ends(&self) {
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
        config
            .mcp_servers
            .push(McpServerConfig::new("missing", "arko-no-such-mcp-server"));
        assert!(McpManager::from_config(&config).await.is_err());
        Program::read(&pid_file).ends().await;
    }

    /// A server that exits by itself leaves nothing of what it started.
    #[tokio::test]
    async fn a_server_that_exits_leaves_no_program_it_started() {
        let pid_file = std::env::temp_dir().join(format!("arko-mcp-{}.pid", uuid::Uuid::new_v4()));
        let config = McpConfig {
            mcp_servers: vec![McpServerConfig::new("leaving", "sh").with_args([
                "-c".into(),
                "sleep 300 & echo $! > \"$1\"".into(),
                "sh".into(),
                pid_file.display().to_string(),
            ])],
        };
        assert!(McpManager::from_config(&config).await.is_err());
        Program::read(&pid_file).ends().await;
    }

    /// Set in the child process of
    /// [`kill_mcp_servers_ends_the_servers_before_an_exit`]: the files its
    /// server writes its own and its program's process IDs to.
    const EXIT_PIDS_ENV: &str = "ARKO_TEST_MCP_EXIT_PIDS";
    /// Set to `1` there for the child to call `kill_mcp_servers` before it
    /// exits.
    const EXIT_KILLS_ENV: &str = "ARKO_TEST_MCP_EXIT_KILLS";

    /// `kill_mcp_servers` ends each server's whole process group, the server
    /// behind its launcher and the program it started, before an exit that
    /// runs no destructor. Without it both outlive the host, as the server
    /// ignores the end of its input. Run in a child process of this test
    /// binary, since it kills every server of its process.
    #[test]
    fn kill_mcp_servers_ends_the_servers_before_an_exit() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        for kills in [false, true] {
            let id = uuid::Uuid::new_v4();
            let server_pid = std::env::temp_dir().join(format!("arko-mcp-exit-{id}.server"));
            let program_pid = std::env::temp_dir().join(format!("arko-mcp-exit-{id}.program"));
            let helper = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "mcp::manager::tests::exits_with_mcp_servers_running",
                    "--exact",
                    "--test-threads=1",
                ])
                .env(
                    EXIT_PIDS_ENV,
                    std::env::join_paths([&server_pid, &program_pid]).unwrap(),
                )
                .env(EXIT_KILLS_ENV, if kills { "1" } else { "0" })
                .stdin(std::process::Stdio::null())
                .output()
                .unwrap();
            assert_eq!(
                helper.status.code(),
                Some(7),
                "the helper exited: {}{}",
                String::from_utf8_lossy(&helper.stdout),
                String::from_utf8_lossy(&helper.stderr)
            );
            let server = Program::read(&server_pid);
            let program = Program::read(&program_pid);
            if kills {
                runtime.block_on(async {
                    server.ends().await;
                    program.ends().await;
                });
            } else {
                std::thread::sleep(Duration::from_millis(500));
                assert!(
                    !server.has_ended() && !program.has_ended(),
                    "they outlive a host that doesn't kill them"
                );
                // Dropping them kills them.
            }
        }
    }

    /// The child process of
    /// [`kill_mcp_servers_ends_the_servers_before_an_exit`]: starts a server
    /// behind a `sh -c` launcher, which ignores the end of its input and
    /// leaves a program running, writes their process IDs, kills the servers
    /// if asked, and exits at once with status 7. Does nothing in a normal
    /// test run.
    #[tokio::test]
    async fn exits_with_mcp_servers_running() {
        let Some(paths) = std::env::var_os(EXIT_PIDS_ENV) else {
            return;
        };
        let paths: Vec<PathBuf> = std::env::split_paths(&paths).collect();
        let python = std::env::var("ARKO_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
        let config = McpConfig {
            mcp_servers: vec![
                McpServerConfig::new("lasting", "sh")
                    .with_args([
                        "-c".into(),
                        "sleep 300 & echo $! > \"$1\"; echo $$ > \"$2\"; shift 2; exec \"$@\""
                            .into(),
                        "sh".into(),
                        paths[1].display().to_string(),
                        paths[0].display().to_string(),
                        python,
                        STUB.into(),
                    ])
                    .with_env(HashMap::from([(
                        "MCP_STUB_IGNORE_EOF".to_string(),
                        "1".to_string(),
                    )])),
            ],
        };
        let _manager = McpManager::from_config(&config).await.unwrap();
        if std::env::var(EXIT_KILLS_ENV).as_deref() == Ok("1") {
            crate::kill_mcp_servers();
        }
        std::process::exit(7);
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
