//! One running MCP server: its process tree, its stderr, and the MCP
//! session with it. Shared by [`McpManager`](super::McpManager) and
//! [`McpHub`](super::McpHub).

use std::collections::VecDeque;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arko_ai::process::NoConsoleWindow;
use arko_ai::process_tree::ProcessTree;
use rmcp::ServiceExt;
use rmcp::service::{Peer, RoleClient, RunningService};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, Command};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use super::config::McpServerConfig;

/// How many of a server's last stderr lines its exit warning shows.
const STDERR_LINES_KEPT: usize = 20;
/// How long a stopping server gets to exit on its own before it is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);
/// How long an exited server's remaining stderr is read before its exit is
/// reported.
const STDERR_DRAIN: Duration = Duration::from_millis(250);
/// How long a failed handshake waits for the server's exit to be reported,
/// to give the exit as the reason.
const EXIT_WAIT: Duration = Duration::from_millis(600);
/// The longest stderr line an exit's description quotes.
const QUOTED_LINE: usize = 200;

/// Why a server couldn't be opened.
#[derive(Debug)]
pub(crate) enum OpenError {
    /// Its program couldn't be started.
    Spawn(std::io::Error),
    /// The MCP handshake failed. With the server's exit, when it exited.
    Handshake { error: String, exit: Option<String> },
}

impl OpenError {
    /// The reason in a few plain words, for an agent or a user: the
    /// server's exit when it exited.
    pub(crate) fn reason(&self, config: &McpServerConfig) -> String {
        match self {
            Self::Spawn(error) if error.kind() == std::io::ErrorKind::NotFound => {
                format!("{} wasn't found", program(config))
            }
            Self::Spawn(error) => format!("{} couldn't be started ({error})", program(config)),
            Self::Handshake {
                exit: Some(exit), ..
            } => exit.clone(),
            Self::Handshake { error, exit: None } => {
                format!("the MCP handshake failed ({error})")
            }
        }
    }
}

/// The program a server runs, by its file name (`npx`, `python3`).
pub(crate) fn program(config: &McpServerConfig) -> String {
    std::path::Path::new(&config.command)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| config.command.clone())
}

/// A server's process and the MCP session with it.
pub(crate) struct Connection {
    name: String,
    service: RunningService<RoleClient, ()>,
    /// Sent to stop the server process gracefully. Dropped, it is killed.
    stop: Option<oneshot::Sender<()>>,
    watcher: Option<JoinHandle<Option<ServerExit>>>,
    /// How the process ended, once it ended on its own.
    exit: Arc<Mutex<Option<String>>>,
}

impl Connection {
    /// Starts the server and does the MCP handshake. The server's tasks run
    /// on the current runtime, and end with it.
    ///
    /// The server and what it starts are one process tree (see
    /// `ProcessTree`). A Linux `PR_SET_PDEATHSIG` would also end a server
    /// when the host is killed outright, but it fires when the thread that
    /// started the server exits, not the host, and reaches only the
    /// launcher, not the programs behind it.
    pub(crate) async fn open(config: &McpServerConfig) -> Result<Self, OpenError> {
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

        let (mut child, tree) = ProcessTree::spawn(&mut cmd).map_err(OpenError::Spawn)?;
        let (Some(stdout), Some(stdin)) = (child.stdout.take(), child.stdin.take()) else {
            return Err(OpenError::Spawn(std::io::Error::other(
                "the server's standard input or output is not available",
            )));
        };
        let stderr = child.stderr.take();
        let (stop, stopped) = oneshot::channel();
        let exit = Arc::new(Mutex::new(None));
        let watcher = watch_server(
            config.name.clone(),
            program(config),
            child,
            tree,
            stderr,
            stopped,
            exit.clone(),
        );

        // Should the handshake fail, dropping `stop` kills the server.
        let service = match ().serve((stdout, stdin)).await {
            Ok(service) => service,
            Err(error) => {
                // A server that exited is reported by its exit.
                let deadline = tokio::time::Instant::now() + EXIT_WAIT;
                let exited = loop {
                    if let Some(exit) = exit.lock().unwrap().clone() {
                        break Some(exit);
                    }
                    if watcher.is_finished() || tokio::time::Instant::now() >= deadline {
                        break exit.lock().unwrap().clone();
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                };
                return Err(OpenError::Handshake {
                    error: error.to_string(),
                    exit: exited,
                });
            }
        };
        Ok(Self {
            name: config.name.clone(),
            service,
            stop: Some(stop),
            watcher: Some(watcher),
            exit,
        })
    }

    pub(crate) fn peer(&self) -> Peer<RoleClient> {
        self.service.peer().clone()
    }

    /// The server's own instructions, from its handshake.
    pub(crate) fn instructions(&self) -> Option<String> {
        self.service
            .peer_info()
            .and_then(|info| info.instructions.clone())
            .filter(|text| !text.trim().is_empty())
    }

    /// How the process ended, once it ended on its own (`npx exited with
    /// code 1`), or `None` while it runs.
    pub(crate) fn exited(&self) -> Option<String> {
        if let Some(exit) = self.exit.lock().unwrap().clone() {
            return Some(exit);
        }
        // Closed without an exit report: its connection ended.
        (self.service.is_closed() || self.service.peer().is_transport_closed())
            .then(|| "its connection closed".to_string())
    }

    /// Stops the process (see [`watch_server`]) and closes the connection,
    /// then waits until the process has exited. The watcher is told first,
    /// so the exit that closing the connection causes isn't reported as an
    /// unexpected one.
    pub(crate) async fn stop(&mut self) {
        log::info!("Shutting down MCP server '{}'", self.name);
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
pub(crate) struct ServerExit {
    status: std::io::Result<ExitStatus>,
    stderr_tail: Vec<String>,
}

/// The signal that ended a process.
#[cfg(unix)]
fn signal_of(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn signal_of(_status: &ExitStatus) -> Option<i32> {
    None
}

/// `program`'s exit as a short phrase: `npx exited with code 1 after
/// writing “Cannot find module”`.
fn describe_exit(program: &str, status: &std::io::Result<ExitStatus>, tail: &[String]) -> String {
    let how = match status {
        Ok(status) => match (status.code(), signal_of(status)) {
            (Some(code), _) => format!("{program} exited with code {code}"),
            (None, Some(signal)) => format!("{program} ended on signal {signal}"),
            (None, None) => format!("{program} exited"),
        },
        Err(_) => format!("{program} exited"),
    };
    match tail.iter().rev().find(|line| !line.trim().is_empty()) {
        Some(line) => {
            let line = line.trim();
            let quoted: String = line.chars().take(QUOTED_LINE).collect();
            let cut = if quoted.len() < line.len() { "…" } else { "" };
            format!("{how} after writing “{quoted}{cut}”")
        }
        None => how,
    }
}

/// Watches a server process: logs each stderr line at info level, tagged
/// with the server's name, and when the process exits on its own, logs one
/// warning with its exit status and last stderr lines, records how it ended
/// in `exit` (see [`describe_exit`]), and returns them.
/// `stop` stops the process: sent, it gets [`SHUTDOWN_GRACE`] to exit before
/// it is killed. Dropped, it is killed at once. A stopped process's exit is
/// logged at info level only.
///
/// The server's whole process tree goes with it. What the server started is
/// killed when the server is killed and when it exits, and also if the task
/// is dropped, as when its runtime ends.
fn watch_server(
    name: String,
    program: String,
    mut child: Child,
    mut tree: ProcessTree,
    stderr: Option<ChildStderr>,
    mut stop: oneshot::Receiver<()>,
    exit: Arc<Mutex<Option<String>>>,
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
        *exit.lock().unwrap() = Some(describe_exit(&program, &status, &stderr_tail));
        Some(ServerExit {
            status,
            stderr_tail,
        })
    })
}

#[cfg(all(test, unix))]
mod tests {
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

    fn watch(
        script: &str,
        stopped: oneshot::Receiver<()>,
    ) -> (JoinHandle<Option<ServerExit>>, Arc<Mutex<Option<String>>>) {
        let (child, tree, stderr) = server(script);
        let exit = Arc::new(Mutex::new(None));
        let watcher = watch_server(
            "files".into(),
            "sh".into(),
            child,
            tree,
            stderr,
            stopped,
            exit.clone(),
        );
        (watcher, exit)
    }

    /// A server that exits on its own is reported with its exit status and
    /// last stderr lines.
    #[tokio::test]
    async fn a_server_that_exits_is_reported_with_its_status_and_stderr() {
        let (_stop, stopped) = oneshot::channel();
        let (watcher, described) = watch(
            "echo starting >&2; echo 'fatal: no config' >&2; exit 3",
            stopped,
        );
        let exit = watcher
            .await
            .unwrap()
            .expect("an exit on its own is reported");
        assert_eq!(exit.status.unwrap().code(), Some(3));
        assert_eq!(exit.stderr_tail, ["starting", "fatal: no config"]);
        assert_eq!(
            described.lock().unwrap().as_deref(),
            Some("sh exited with code 3 after writing “fatal: no config”")
        );
    }

    /// Bytes that aren't UTF-8 don't stop the reading: the server can keep
    /// writing to stderr, and its lines are logged with replacements.
    #[tokio::test]
    async fn stderr_that_is_not_utf8_is_still_read() {
        let (_stop, stopped) = oneshot::channel();
        let (watcher, _) = watch(
            "printf 'bad \\377 byte\\n' >&2; i=0; while [ $i -lt 2000 ]; do echo more output $i >&2; i=$((i+1)); done; exit 4",
            stopped,
        );
        let exit = watcher.await.unwrap().unwrap();
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
        let (_stop, stopped) = oneshot::channel();
        let (watcher, _) = watch(
            "i=0; while [ $i -lt 30 ]; do echo line $i >&2; i=$((i+1)); done",
            stopped,
        );
        let exit = watcher.await.unwrap().unwrap();
        assert_eq!(exit.stderr_tail.len(), STDERR_LINES_KEPT);
        assert_eq!(exit.stderr_tail.last().map(String::as_str), Some("line 29"));
    }

    /// A server the manager stops, gracefully or by dropping it, is not
    /// reported as an exit.
    #[tokio::test]
    async fn a_stopped_server_is_not_reported() {
        for graceful in [true, false] {
            let (stop, stopped) = oneshot::channel();
            let (watcher, described) = watch("echo ready >&2; exec sleep 30", stopped);
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
            assert_eq!(*described.lock().unwrap(), None);
        }
    }

    #[test]
    fn an_exit_is_described_in_a_few_words() {
        use std::os::unix::process::ExitStatusExt;
        let code = |code: i32| Ok(ExitStatus::from_raw(code << 8));
        assert_eq!(
            describe_exit("npx", &code(1), &[]),
            "npx exited with code 1"
        );
        assert_eq!(
            describe_exit("npx", &Ok(ExitStatus::from_raw(9)), &[]),
            "npx ended on signal 9"
        );
        let long = "x".repeat(300);
        let described = describe_exit("npx", &code(2), &[long, String::new()]);
        let (said, quoted) = described.split_once('“').unwrap();
        assert_eq!(said, "npx exited with code 2 after writing ");
        assert_eq!(quoted, format!("{}…”", "x".repeat(QUOTED_LINE)));
    }
}
