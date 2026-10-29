//! One running MCP server: its process tree, its stderr, and the MCP
//! session with it. Shared by [`McpManager`](super::McpManager) and
//! [`McpHub`](super::McpHub).

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
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
/// The shortest value of a server's own environment that [`Redactions`]
/// keeps out of the log. Shorter ones (`1`, `on`) would blank out ordinary
/// words and numbers.
const SHORTEST_REDACTED: usize = 4;
/// The shortest value of an inherited variable with a secret's name that
/// [`Redactions`] keeps out of the log.
const SHORTEST_INHERITED_SECRET: usize = 8;
/// What the name of a variable holding a secret has in it, ignoring case.
const SECRET_NAME_PARTS: [&str; 6] = ["TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"];

/// Why a server couldn't be opened.
#[derive(Debug)]
pub(crate) enum OpenError {
    /// Its program couldn't be started.
    Spawn(std::io::Error),
    /// The MCP handshake failed. With the server's exit, when it exited.
    Handshake { error: String, exit: Option<String> },
}

impl OpenError {
    /// The reason in a few fixed words, for an agent or a user: the
    /// server's exit when it exited. It never holds the server's command or
    /// what it wrote, which can hold the values of its environment. Those
    /// go to the log.
    pub(crate) fn reason(&self) -> String {
        match self {
            Self::Spawn(error) => match error.kind() {
                std::io::ErrorKind::NotFound => "its program wasn't found".into(),
                std::io::ErrorKind::PermissionDenied => {
                    "its program couldn't be started (permission denied)".into()
                }
                _ => "its program couldn't be started".into(),
            },
            Self::Handshake {
                exit: Some(exit), ..
            } => exit.clone(),
            Self::Handshake { exit: None, .. } => "it didn't complete the MCP handshake".into(),
        }
    }

    /// What the log says of it: the program by its file name and the error
    /// in full, with `redactions` applied.
    fn diagnostic(&self, config: &McpServerConfig, redactions: &Redactions) -> String {
        let program = program(config);
        redactions.apply(&match self {
            Self::Spawn(error) => format!("{program} couldn't be started: {error}"),
            Self::Handshake { error, .. } => format!("the MCP handshake failed: {error}"),
        })
    }
}

/// Whether the variable `name` looks like it holds a secret: its name has
/// one of [`SECRET_NAME_PARTS`] in it, or ends in `PASS`, ignoring case.
fn secret_name(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.ends_with("PASS") || SECRET_NAME_PARTS.iter().any(|part| name.contains(part))
}

/// The program a server runs, by its file name (`npx`, `python3`), for the
/// log only, as an expanded command can hold a secret.
fn program(config: &McpServerConfig) -> String {
    std::path::Path::new(&config.command)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| config.command.clone())
}

/// The values a server's log lines must not show, each shown as `${NAME}`
/// instead. They are the values of its config's
/// [`env`](McpServerConfig::env) and of the host's
/// [`expanded`](McpServerConfig::expanded) variables, and those of the
/// variables it inherits from the host whose names look secret (see
/// [`secret_name`]). A server can print its own environment and arguments
/// to stderr, a token among them.
#[derive(Debug, Default)]
pub(crate) struct Redactions(Vec<(String, String)>);

impl Redactions {
    /// The redactions for `config`'s server, whose process also gets
    /// `inherited`, the host's environment (a shared server's snapshot).
    pub(crate) fn of(
        config: &McpServerConfig,
        inherited: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Self {
        let explicit = config
            .env
            .iter()
            .chain(&config.expanded)
            .filter(|(_, value)| value.chars().count() >= SHORTEST_REDACTED)
            .map(|(name, value)| (name.clone(), value.clone()));
        let inherited = inherited
            .into_iter()
            .filter_map(|(name, value)| Some((name.into_string().ok()?, value.into_string().ok()?)))
            .filter(|(name, value)| {
                secret_name(name) && value.chars().count() >= SHORTEST_INHERITED_SECRET
            });
        let mut values: Vec<(String, String)> = explicit
            .chain(inherited)
            .map(|(name, value)| (value, format!("${{{name}}}")))
            .collect();
        // The longest first, so a value holding another goes whole.
        values.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.cmp(b)));
        values.dedup_by(|a, b| a.0 == b.0);
        Self(values)
    }

    /// `text` with each value replaced by its name.
    pub(crate) fn apply(&self, text: &str) -> String {
        let mut text = text.to_string();
        for (value, name) in &self.0 {
            if text.contains(value.as_str()) {
                text = text.replace(value.as_str(), name);
            }
        }
        text
    }
}

/// What a server's process starts from, fixed when it is taken: the whole
/// environment and the working directory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Launch {
    /// Every variable the process gets, sorted by name.
    pub(crate) env: Vec<(OsString, OsString)>,
    pub(crate) dir: Option<PathBuf>,
}

impl Launch {
    /// `config`'s launch from the host's `env` and working directory `dir`:
    /// `env` with the config's `env` over it. On Windows, where names don't
    /// differ by case, a config's `Path` replaces the host's `PATH`.
    pub(crate) fn of(
        config: &McpServerConfig,
        env: impl IntoIterator<Item = (OsString, OsString)>,
        dir: Option<PathBuf>,
    ) -> Self {
        let mut env: Vec<(OsString, OsString)> = env.into_iter().collect();
        for (name, value) in &config.env {
            env.retain(|(known, _)| !same_name(known, OsStr::new(name)));
            env.push((name.into(), value.into()));
        }
        env.sort();
        Self { env, dir }
    }

    /// The value of the variable `name`.
    fn var(&self, name: &str) -> Option<&OsStr> {
        self.env
            .iter()
            .find(|(known, _)| same_name(known, OsStr::new(name)))
            .map(|(_, value)| value.as_os_str())
    }

    /// Where the process would find `command`: a path made absolute against
    /// the working directory, or the first program of that name on the
    /// launch's `PATH` (with `PATHEXT`'s extensions on Windows). `None` when
    /// there's none.
    pub(crate) fn program(&self, command: &str) -> Option<PathBuf> {
        let absolute = |path: &Path| match &self.dir {
            Some(dir) if path.is_relative() => dir.join(path),
            _ => path.to_path_buf(),
        };
        let path = Path::new(command);
        if path.is_absolute() || path.components().count() > 1 {
            return Some(absolute(path));
        }
        let extensions: Vec<OsString> = if cfg!(windows) && path.extension().is_none() {
            let listed = self
                .var("PATHEXT")
                .unwrap_or(OsStr::new(".COM;.EXE;.BAT;.CMD"));
            listed
                .to_string_lossy()
                .split(';')
                .filter(|extension| !extension.is_empty())
                .map(OsString::from)
                .collect()
        } else {
            Vec::new()
        };
        std::env::split_paths(self.var("PATH")?)
            .map(|dir| absolute(&dir).join(command))
            .flat_map(|found| {
                let mut candidates = vec![found.clone()];
                candidates.extend(extensions.iter().map(|extension| {
                    let mut name = found.clone().into_os_string();
                    name.push(extension);
                    PathBuf::from(name)
                }));
                candidates
            })
            .find(|candidate| runnable(candidate))
    }
}

/// Whether two variable names are the same. Windows ignores their case.
fn same_name(a: &OsStr, b: &OsStr) -> bool {
    if cfg!(windows) {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

/// Whether `path` is a file the process could run.
fn runnable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        metadata.is_file()
    }
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
    redactions: Arc<Redactions>,
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
    ///
    /// A failure is logged in full, with [`Redactions`] applied, and its
    /// [`reason`](OpenError::reason) says it in fixed words.
    ///
    /// The process inherits the host's environment, with the config's
    /// `env` over it, and its working directory.
    ///
    /// The process belongs to its stop handle from the moment it's spawned,
    /// so dropping the future before the handshake ends kills it, with what
    /// it started.
    pub(crate) async fn open(config: &McpServerConfig) -> Result<Self, OpenError> {
        Self::open_in(config, None).await
    }

    /// [`open`](Self::open), with `launch`'s environment and working
    /// directory, when given, in place of the host's.
    pub(crate) async fn open_in(
        config: &McpServerConfig,
        launch: Option<&Launch>,
    ) -> Result<Self, OpenError> {
        let redactions = Arc::new(match launch {
            Some(launch) => Redactions::of(config, launch.env.iter().cloned()),
            None => Redactions::of(config, std::env::vars_os()),
        });
        let opened = Self::open_with(config, launch, redactions.clone()).await;
        if let Err(error) = &opened {
            log::warn!(
                "MCP server '{}': {}",
                config.name,
                error.diagnostic(config, &redactions)
            );
        }
        opened
    }

    async fn open_with(
        config: &McpServerConfig,
        launch: Option<&Launch>,
        redactions: Arc<Redactions>,
    ) -> Result<Self, OpenError> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .no_console_window();
        match launch {
            Some(launch) => {
                cmd.env_clear().envs(launch.env.iter().cloned());
                if let Some(dir) = &launch.dir {
                    cmd.current_dir(dir);
                }
            }
            None => {
                for (key, value) in &config.env {
                    cmd.env(key, value);
                }
            }
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
            child,
            tree,
            stderr,
            redactions.clone(),
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
            redactions,
        })
    }

    pub(crate) fn peer(&self) -> Peer<RoleClient> {
        self.service.peer().clone()
    }

    /// What its log lines leave out.
    pub(crate) fn redactions(&self) -> Arc<Redactions> {
        self.redactions.clone()
    }

    /// The server's own instructions, from its handshake.
    pub(crate) fn instructions(&self) -> Option<String> {
        self.service
            .peer_info()
            .and_then(|info| info.instructions.clone())
            .filter(|text| !text.trim().is_empty())
    }

    /// How the process ended, once it ended on its own (`it exited with
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

/// A server's exit as a short phrase, `it exited with code 1`, with nothing
/// the server wrote (its stderr goes to the log only).
fn describe_exit(status: &std::io::Result<ExitStatus>) -> String {
    match status {
        Ok(status) => match (status.code(), signal_of(status)) {
            (Some(code), _) => format!("it exited with code {code}"),
            (None, Some(signal)) => format!("it ended on signal {signal}"),
            (None, None) => "it exited".into(),
        },
        Err(_) => "it exited".into(),
    }
}

/// Watches a server process: logs each stderr line at info level, tagged
/// with the server's name and with `redactions` applied, and when the
/// process exits on its own, logs one warning with its exit status and last
/// stderr lines, records how it ended in `exit` (see [`describe_exit`]),
/// and returns them.
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
    redactions: Arc<Redactions>,
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
                        let line = redactions.apply(line.trim_end_matches(['\n', '\r']));
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
        *exit.lock().unwrap() = Some(describe_exit(&status));
        Some(ServerExit {
            status,
            stderr_tail,
        })
    })
}

#[cfg(all(test, unix))]
mod tests {
    use std::collections::HashMap;

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
        watch_redacting(script, Redactions::default(), stopped)
    }

    fn watch_redacting(
        script: &str,
        redactions: Redactions,
        stopped: oneshot::Receiver<()>,
    ) -> (JoinHandle<Option<ServerExit>>, Arc<Mutex<Option<String>>>) {
        let (child, tree, stderr) = server(script);
        let exit = Arc::new(Mutex::new(None));
        let watcher = watch_server(
            "files".into(),
            child,
            tree,
            stderr,
            Arc::new(redactions),
            stopped,
            exit.clone(),
        );
        (watcher, exit)
    }

    /// A server that exits on its own is reported with its exit status and
    /// last stderr lines, and its exit is described without them.
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
            Some("it exited with code 3")
        );
    }

    /// The values of the server's environment, and those the host expanded,
    /// are replaced by their names in what it writes.
    #[tokio::test]
    async fn stderr_is_kept_with_the_environment_values_replaced() {
        let mut config = McpServerConfig::new("files", "sh")
            .with_env([("GITHUB_TOKEN", "ghp_fake_token_1234"), ("SHORT", "on")]);
        config.expanded = HashMap::from([("API_KEY".into(), "sk-fake-key-5678".into())]);
        let (_stop, stopped) = oneshot::channel();
        let (watcher, _) = watch_redacting(
            "echo 'token ghp_fake_token_1234, key sk-fake-key-5678, mode on' >&2; exit 1",
            Redactions::of(&config, []),
            stopped,
        );
        let exit = watcher.await.unwrap().unwrap();
        assert_eq!(
            exit.stderr_tail,
            ["token ${GITHUB_TOKEN}, key ${API_KEY}, mode on"]
        );
    }

    /// A launch finds its program on its own `PATH`, and a path against its
    /// own working directory.
    #[test]
    fn a_launch_finds_its_program_on_its_own_path() {
        let config = McpServerConfig::new("files", "sh");
        let launch = |path: &str| {
            Launch::of(
                &config,
                [("PATH".into(), path.into())],
                Some("/work".into()),
            )
        };
        let found = launch("/nonexistent:/bin:/usr/bin").program("sh").unwrap();
        assert!(found.is_absolute() && found.ends_with("sh"), "{found:?}");
        assert_eq!(launch("/nonexistent").program("sh"), None);
        assert_eq!(
            launch("/bin").program("./run"),
            Some(PathBuf::from("/work/./run"))
        );
        assert_eq!(
            launch("/bin").program("/opt/run"),
            Some(PathBuf::from("/opt/run"))
        );
        // The config's env goes over the host's.
        let over = McpServerConfig::new("files", "sh").with_env([("PATH", "/nonexistent")]);
        let launch = Launch::of(&over, [("PATH".into(), "/bin".into())], None);
        assert_eq!(launch.env, [("PATH".into(), "/nonexistent".into())]);
        assert_eq!(launch.program("sh"), None);
    }

    /// Of what the server inherits, the values of 8 characters or more of
    /// the variables whose names look secret are kept out too. Its own
    /// `env` goes from 4 characters, whatever the name.
    #[test]
    fn inherited_secrets_are_kept_out_by_their_names() {
        let config = McpServerConfig::new("files", "sh").with_env([("MODE", "fast")]);
        let inherited = [
            ("FAKE_API_TOKEN", "fake-inherited-token-1234"),
            ("db_password", "hunter2hunter2"),
            ("SMTP_PASS", "longpassvalue"),
            ("GH_AUTH", "authvalue123"),
            ("SHORT_TOKEN", "abc1234"),
            ("HOME", "/home/someone"),
            ("PASSAGE", "notasecretvalue"),
        ]
        .map(|(name, value)| (OsString::from(name), OsString::from(value)));
        let redactions = Redactions::of(&config, inherited);
        assert_eq!(
            redactions.apply(
                "fast fake-inherited-token-1234 hunter2hunter2 longpassvalue authvalue123 abc1234 /home/someone notasecretvalue"
            ),
            "${MODE} ${FAKE_API_TOKEN} ${db_password} ${SMTP_PASS} ${GH_AUTH} abc1234 /home/someone notasecretvalue"
        );
    }

    #[test]
    fn a_value_holding_another_is_replaced_whole() {
        let config =
            McpServerConfig::new("files", "sh").with_env([("A", "abcd"), ("AB", "abcdefgh")]);
        assert_eq!(
            Redactions::of(&config, []).apply("abcdefgh abcd"),
            "${AB} ${A}"
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
        assert_eq!(
            describe_exit(&Ok(ExitStatus::from_raw(1 << 8))),
            "it exited with code 1"
        );
        assert_eq!(
            describe_exit(&Ok(ExitStatus::from_raw(9))),
            "it ended on signal 9"
        );
        assert_eq!(
            describe_exit(&Err(std::io::Error::other("gone"))),
            "it exited"
        );
    }
}
