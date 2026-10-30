//! Cooperative stop for tools and the commands they run.
//!
//! When a task is being stopped, its tools are told through a
//! [`StopSignal`], which they find in
//! [`TaskContext::stop`](crate::tool::TaskContext::stop). The signal fires
//! when the runtime receives `CancelTask` or `SuspendTask` while a call
//! runs, or when the host fires the signal it gave the runtime
//! ([`AgentRuntime::with_stop_signal`](crate::runtime::AgentRuntime::with_stop_signal)).
//! A workflow fires it when a graceful stop begins. A tool that watches it
//! can save its state and return early instead of being dropped mid-work.
//!
//! [`StopAwareCommand`] does this for a child process. The child is told
//! through a stop file, then sent `SIGTERM`, then killed.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use arko_ai::process::NoConsoleWindow;
use arko_ai::process_tree::exited;
use tokio::io::AsyncReadExt;
use tokio::sync::watch;

/// The environment variable [`StopAwareCommand`] sets for its child to name
/// the stop file, which exists once the task is stopping. A script
/// checks it between units of work, saves its state and exits, for example
/// `os.path.exists(os.environ["ARKO_STOP_FILE"])` in Python.
pub const STOP_FILE_ENV: &str = "ARKO_STOP_FILE";

/// How long [`StopAwareCommand`] waits, by default, for its child to exit
/// after creating the stop file, before it sends `SIGTERM`. See
/// [`StopAwareCommand::grace`].
pub const DEFAULT_STOP_GRACE: Duration = Duration::from_secs(10);

/// How long [`StopAwareCommand`] waits, by default, after `SIGTERM` before it
/// kills the child. See [`StopAwareCommand::terminate_grace`].
pub const DEFAULT_TERMINATE_GRACE: Duration = Duration::from_secs(5);

/// Whether a task is being stopped, for a tool to observe. See the
/// [module documentation](self).
///
/// Clones observe the same stop. [`never`](Self::never) (also the
/// `Default`) never fires. A stop, once fired, stays fired.
#[derive(Clone, Default)]
pub struct StopSignal {
    /// The signal fires when any of these does.
    sources: Vec<watch::Receiver<Option<Arc<str>>>>,
}

impl StopSignal {
    /// A signal that never fires, for a [`TaskContext`](crate::tool::TaskContext)
    /// a host builds outside a runtime, and for tests.
    pub fn never() -> Self {
        Self::default()
    }

    /// Whether the task is being stopped.
    pub fn is_stopping(&self) -> bool {
        self.sources.iter().any(|source| source.borrow().is_some())
    }

    /// Why the task is being stopped, once it is (for example
    /// `Task cancelled.`, or a workflow's stop reason).
    pub fn reason(&self) -> Option<String> {
        self.sources
            .iter()
            .find_map(|source| source.borrow().as_deref().map(str::to_string))
    }

    /// Resolves once the task is being stopped (at once if it already is).
    /// A signal that can no longer fire, such as [`never`](Self::never),
    /// never resolves.
    pub async fn stopping(&self) {
        if self.sources.is_empty() {
            return std::future::pending().await;
        }
        let waits = self.sources.iter().map(|source| {
            let mut source = source.clone();
            Box::pin(async move {
                // Dropped without firing: this source never fires.
                if source.wait_for(Option::is_some).await.is_err() {
                    std::future::pending::<()>().await;
                }
            })
        });
        futures::future::select_all(waits).await;
    }

    /// A signal that fires when this one or `other` does.
    pub(crate) fn or(&self, other: &StopSignal) -> StopSignal {
        StopSignal {
            sources: self.sources.iter().chain(&other.sources).cloned().collect(),
        }
    }
}

impl fmt::Debug for StopSignal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StopSignal")
            .field("stopping", &self.is_stopping())
            .finish()
    }
}

/// Fires a [`StopSignal`] so a host can tell a runtime's tools that
/// their task is stopping. Clones fire the same signal.
#[derive(Clone)]
pub struct StopSource(Arc<watch::Sender<Option<Arc<str>>>>);

impl StopSource {
    /// A source that has not fired.
    pub fn new() -> Self {
        Self(Arc::new(watch::Sender::new(None)))
    }

    /// The signal this source fires.
    pub fn signal(&self) -> StopSignal {
        StopSignal {
            sources: vec![self.0.subscribe()],
        }
    }

    /// Fires the signal with `reason`. Returns `false`, changing nothing,
    /// when it had fired already, so the first reason stands.
    pub fn stop(&self, reason: impl Into<String>) -> bool {
        let reason: Arc<str> = reason.into().into();
        self.0.send_if_modified(|state| {
            if state.is_some() {
                return false;
            }
            *state = Some(reason);
            true
        })
    }

    /// Whether the signal has fired.
    pub fn is_stopped(&self) -> bool {
        self.0.borrow().is_some()
    }
}

impl Default for StopSource {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StopSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StopSource")
            .field("stopped", &self.is_stopped())
            .finish()
    }
}

/// How a [`StopAwareCommand`]'s child ended once its task was stopping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildStop {
    /// It exited on its own within the grace after the stop file was
    /// created, having seen the file or not.
    Exited,
    /// It exited after `SIGTERM` was sent to its process group (Unix only).
    Terminated,
    /// It was killed. Its process group was sent `SIGKILL` (on Unix), or
    /// the process was terminated (elsewhere).
    Killed,
}

/// What a [`StopAwareCommand`] child produced.
#[derive(Debug)]
pub struct StopAwareOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// How the child ended when its task was stopping while it ran. `None`
    /// when it ended without a stop.
    pub stopped: Option<ChildStop>,
}

/// A child process that stops cooperatively when its task stops.
///
/// Wraps a [`tokio::process::Command`]. The child gets the environment
/// variable [`STOP_FILE_ENV`] (`ARKO_STOP_FILE`) naming a file that doesn't
/// exist yet. When the [`StopSignal`] fires while the child runs (or has
/// fired before it starts), the stop sequence runs:
///
/// 1. The stop file is created, and the child has [`grace`](Self::grace)
///    (default [`DEFAULT_STOP_GRACE`], 10 seconds) to save its state and
///    exit.
/// 2. On Unix, its process group is then sent `SIGTERM`, with
///    [`terminate_grace`](Self::terminate_grace) (default
///    [`DEFAULT_TERMINATE_GRACE`], 5 seconds) to exit.
/// 3. Then it is killed with `SIGKILL` to its process group on Unix, and
///    `TerminateProcess` on the child alone elsewhere.
///
/// On Unix the child leads a new process group, so the signals reach the
/// processes it starts, unless one of them leaves the group (a `setsid`, a
/// daemon). After a stop, the group is sent `SIGKILL` once the child has
/// ended, so processes still in the group are killed too. Elsewhere only the
/// child itself is terminated, so processes it started may keep running. A
/// stop outside Unix has no `SIGTERM` step.
///
/// Being in its own group, the child is not in the terminal's foreground
/// group, so the terminal's Ctrl-C doesn't reach it (the host's stop does).
/// It can't read the terminal because a read stops it (`SIGTTIN`) until it is
/// killed. Give a child that may read its standard input a null one (or a
/// file) when it runs with [`status`](Self::status). [`output`](Self::output)
/// always does. A host that exits at once (`std::process::exit`, which runs
/// no destructors) calls [`kill_stop_aware_commands`] first, or the children
/// keep running.
///
/// Dropping the future (the runtime drops an interruptible call when its
/// task is cancelled, or when the call's time limit passes) kills its process
/// group at once on Unix, or the child alone elsewhere.
///
/// The grace only runs once a stop is under way, so a call that is not
/// stopped runs as long as the command does. A tool that wants its child to
/// get the grace when its task is cancelled outright (not only in a
/// workflow's wrap-up) declares itself not
/// [interruptible](crate::tool::Tool::interruptible). The runtime then waits
/// for the call, whose signal fires on the cancellation.
///
/// The stop file is a fresh path in the system's temporary directory unless
/// [`stop_file`](Self::stop_file) names one (for a child in a sandbox that
/// sees only some directories). An existing file at that path is removed
/// before the child starts. A temporary stop file is removed when the child
/// has ended. One the caller named is left in place.
#[derive(Debug)]
pub struct StopAwareCommand {
    command: tokio::process::Command,
    stop_file: Option<PathBuf>,
    grace: Duration,
    terminate_grace: Duration,
}

impl StopAwareCommand {
    /// Runs `command` with the stop sequence and the default graces.
    pub fn new(command: tokio::process::Command) -> Self {
        Self {
            command,
            stop_file: None,
            grace: DEFAULT_STOP_GRACE,
            terminate_grace: DEFAULT_TERMINATE_GRACE,
        }
    }

    /// The stop file's path, instead of a fresh temporary one.
    pub fn stop_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.stop_file = Some(path.into());
        self
    }

    /// How long the child has, once the stop file exists, before `SIGTERM`
    /// (or, outside Unix, before it is terminated).
    pub fn grace(mut self, grace: Duration) -> Self {
        self.grace = grace;
        self
    }

    /// How long the child has after `SIGTERM` before it is killed (Unix).
    pub fn terminate_grace(mut self, grace: Duration) -> Self {
        self.terminate_grace = grace;
        self
    }

    /// The command, to configure further.
    pub fn command_mut(&mut self) -> &mut tokio::process::Command {
        &mut self.command
    }

    /// Runs the child to its end, stopping it as described above when
    /// `stop` fires, and returns its exit status and output. Its standard
    /// output and error are captured, and its standard input is null, so it
    /// never reads the host's input. On Windows it starts without a console
    /// window ([`NoConsoleWindow`]), which replaces creation flags set on the
    /// command.
    ///
    /// Once the child has ended after a stop, output still held open by a
    /// process it started (outside Unix, or one that left its group) is read
    /// for at most the terminate grace. What was read by then is returned.
    pub async fn output(mut self, stop: &StopSignal) -> io::Result<StopAwareOutput> {
        self.command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .no_console_window();
        self.run(stop).await
    }

    /// Runs the child to its end with its standard streams as the command
    /// configures them, stopping it as described above when `stop` fires.
    /// Returns its exit status and how a stop ended it.
    ///
    /// On Unix a child that inherits a terminal as its standard input can't
    /// read it (see above). Give one that may read a null standard input
    /// (`command_mut().stdin(Stdio::null())`), or a file. A host without a
    /// console (a windowed Windows app) calls
    /// [`no_console_window`](NoConsoleWindow::no_console_window) on a
    /// command whose console the user shouldn't see.
    pub async fn status(self, stop: &StopSignal) -> io::Result<(ExitStatus, Option<ChildStop>)> {
        let output = self.run(stop).await?;
        Ok((output.status, output.stopped))
    }

    async fn run(mut self, stop: &StopSignal) -> io::Result<StopAwareOutput> {
        let (stop_file, temporary) = match self.stop_file.take() {
            Some(path) => (path, false),
            None => (
                std::env::temp_dir().join(format!("arko-stop-{}", uuid::Uuid::new_v4())),
                true,
            ),
        };
        // A file left by an earlier run would stop the child at once.
        match std::fs::remove_file(&stop_file) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        self.command.env(STOP_FILE_ENV, &stop_file);
        self.command.kill_on_drop(true);
        #[cfg(unix)]
        self.command.process_group(0);
        let mut child = self.command.spawn()?;
        let mut group = ProcessGroup::of(&child);
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        let mut stdout_bytes = Vec::new();
        let mut stderr_bytes = Vec::new();
        let ended = {
            let reads = async {
                let out = async {
                    if let Some(pipe) = stdout.as_mut() {
                        pipe.read_to_end(&mut stdout_bytes).await?;
                    }
                    Ok::<_, io::Error>(())
                };
                let err = async {
                    if let Some(pipe) = stderr.as_mut() {
                        pipe.read_to_end(&mut stderr_bytes).await?;
                    }
                    Ok::<_, io::Error>(())
                };
                tokio::try_join!(out, err).map(|_| ())
            };
            tokio::pin!(reads);
            let wait = wait_with_stop(
                &mut child,
                &mut group,
                stop,
                &stop_file,
                self.grace,
                self.terminate_grace,
            );
            tokio::pin!(wait);
            let mut read = None;
            let ended = loop {
                tokio::select! {
                    result = &mut reads, if read.is_none() => read = Some(result),
                    ended = &mut wait => break ended,
                }
            };
            if read.is_none() {
                read = match ended {
                    // After a stop, a process the child started may hold its
                    // output open: read what comes for a bounded time.
                    Ok((_, Some(_))) => tokio::time::timeout(self.terminate_grace, &mut reads)
                        .await
                        .ok(),
                    _ => Some(reads.await),
                };
            }
            if let Some(Err(error)) = read {
                return Err(error);
            }
            ended
        };
        if temporary {
            let _ = std::fs::remove_file(&stop_file);
        }
        let (status, stopped) = ended?;
        Ok(StopAwareOutput {
            status,
            stdout: stdout_bytes,
            stderr: stderr_bytes,
            stopped,
        })
    }
}

/// Waits for `child` to end, running the stop sequence when `stop` fires.
async fn wait_with_stop(
    child: &mut tokio::process::Child,
    group: &mut ProcessGroup,
    stop: &StopSignal,
    stop_file: &Path,
    grace: Duration,
    terminate_grace: Duration,
) -> io::Result<(ExitStatus, Option<ChildStop>)> {
    // The child is collected only once its group is disarmed (and killed,
    // after a stop), while the group's ID can't have passed to another
    // program.
    tokio::select! {
        biased;
        () = exited(child) => {
            // Ended without a stop: what it started is its own business, as
            // with a plain command.
            group.disarm();
            return Ok((child.wait().await?, None));
        }
        () = stop.stopping() => {}
    }
    if let Err(error) = std::fs::File::create(stop_file) {
        log::warn!(
            "Could not create the stop file {} for a stopping command: {error}",
            stop_file.display()
        );
    }
    let mut how = ChildStop::Exited;
    let ended = tokio::time::timeout(grace, exited(child)).await.is_ok();
    #[cfg(unix)]
    let ended = ended || {
        group.signal(nix::sys::signal::Signal::SIGTERM);
        how = ChildStop::Terminated;
        tokio::time::timeout(terminate_grace, exited(child))
            .await
            .is_ok()
    };
    #[cfg(not(unix))]
    let _ = terminate_grace;
    if !ended {
        how = ChildStop::Killed;
        group.kill();
        // Already gone: the wait below reports how it ended.
        let _ = child.start_kill();
    }
    // What the child started ends with it.
    group.kill();
    group.disarm();
    let status = child.wait().await?;
    Ok((status, Some(how)))
}

/// Kills the process group of every [`StopAwareCommand`] child running in
/// this process, at once (`SIGKILL` on Unix, no effect elsewhere).
/// For a host about to exit without unwinding. `std::process::exit` runs no
/// destructors, so the calls' own kill on drop never runs, and since each
/// child leads its own process group the terminal's Ctrl-C doesn't reach it
/// either. Without this the children keep running after the host has gone.
/// arko_workflows' `stop_on_signals` calls it before it exits on a third
/// signal. The calls themselves then end as their children were killed.
pub fn kill_stop_aware_commands() {
    #[cfg(unix)]
    for id in armed_groups().iter() {
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(*id),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
}

/// The armed [`ProcessGroup`]s of this process, for
/// [`kill_stop_aware_commands`].
#[cfg(unix)]
static ARMED_GROUPS: std::sync::Mutex<std::collections::BTreeSet<i32>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

#[cfg(unix)]
fn armed_groups() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<i32>> {
    ARMED_GROUPS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A child's process group (Unix), killed when dropped while armed: a call
/// dropped mid-run leaves nothing running. Armed groups are recorded
/// process-wide, for [`kill_stop_aware_commands`].
struct ProcessGroup {
    #[cfg(unix)]
    id: Option<nix::unistd::Pid>,
}

impl ProcessGroup {
    fn of(child: &tokio::process::Child) -> Self {
        #[cfg(unix)]
        {
            let id = child.id().and_then(|id| i32::try_from(id).ok());
            if let Some(id) = id {
                armed_groups().insert(id);
            }
            Self {
                id: id.map(nix::unistd::Pid::from_raw),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = child;
            Self {}
        }
    }

    #[cfg(unix)]
    fn signal(&self, signal: nix::sys::signal::Signal) {
        if let Some(id) = self.id {
            let _ = nix::sys::signal::killpg(id, signal);
        }
    }

    fn kill(&self) {
        #[cfg(unix)]
        self.signal(nix::sys::signal::Signal::SIGKILL);
    }

    fn disarm(&mut self) {
        #[cfg(unix)]
        if let Some(id) = self.id.take() {
            armed_groups().remove(&id.as_raw());
        }
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.kill();
        self.disarm();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_signal_fires_once_and_any_source_fires_a_joined_signal() {
        let first = StopSource::new();
        let second = StopSource::new();
        let joined = first.signal().or(&second.signal());
        assert!(!joined.is_stopping());
        assert_eq!(joined.reason(), None);
        let waiting = tokio::spawn({
            let joined = joined.clone();
            async move { joined.stopping().await }
        });
        assert!(second.stop("Task cancelled."));
        assert!(!second.stop("later"), "the first reason stands");
        tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
        assert!(joined.is_stopping());
        assert_eq!(joined.reason().as_deref(), Some("Task cancelled."));
        assert!(!first.is_stopped());
        // Already fired: resolves at once.
        tokio::time::timeout(Duration::from_secs(5), joined.stopping())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_never_signal_and_a_dropped_source_never_fire() {
        let never = StopSignal::never();
        assert!(!never.is_stopping());
        assert!(
            tokio::time::timeout(Duration::from_millis(50), never.stopping())
                .await
                .is_err()
        );
        let source = StopSource::new();
        let signal = source.signal();
        drop(source);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), signal.stopping())
                .await
                .is_err()
        );
    }

    #[cfg(unix)]
    fn shell(script: &str) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("sh");
        command.arg("-c").arg(script);
        command
    }

    /// On Windows a child run for its output gets a hidden console of its
    /// own, so a host with no console doesn't open a console window for it.
    #[cfg(windows)]
    #[tokio::test]
    async fn windows_output_children_get_a_hidden_console_of_their_own() {
        let mut command = tokio::process::Command::new(crate::console_probe::python());
        command.arg("-c").arg(crate::console_probe::code());
        let output = StopAwareCommand::new(command)
            .output(&StopSignal::never())
            .await
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "CONSOLE=own hidden"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_command_that_is_not_stopped_runs_to_its_end() {
        let output = StopAwareCommand::new(shell(
            "test -f \"$ARKO_STOP_FILE\" && echo stopped || echo running",
        ))
        .output(&StopSignal::never())
        .await
        .unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "running\n");
        assert_eq!(output.stopped, None);
    }

    /// The child sees the stop file, saves its state and exits within the
    /// grace.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_that_watches_the_stop_file_exits_within_the_grace() {
        let source = StopSource::new();
        let started = std::env::temp_dir().join(format!("arko-started-{}", uuid::Uuid::new_v4()));
        let command = StopAwareCommand::new(shell(&format!(
            "touch '{}'; while [ ! -f \"$ARKO_STOP_FILE\" ]; do sleep 0.02; done; echo saved; exit 3",
            started.display()
        )))
        .grace(Duration::from_secs(10));
        let run = tokio::spawn({
            let signal = source.signal();
            async move { command.output(&signal).await }
        });
        tokio::time::timeout(Duration::from_secs(10), async {
            while !started.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let stopped_at = std::time::Instant::now();
        source.stop("stopping");
        let output = tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(stopped_at.elapsed() < Duration::from_secs(5));
        assert_eq!(output.stopped, Some(ChildStop::Exited));
        assert_eq!(output.status.code(), Some(3));
        assert_eq!(String::from_utf8_lossy(&output.stdout), "saved\n");
        let _ = std::fs::remove_file(started);
    }

    /// A child that ignores both the stop file and `SIGTERM` is killed, with
    /// what it started.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_that_ignores_the_stop_is_killed_with_its_group() {
        use std::os::unix::process::ExitStatusExt;
        let pid_file = std::env::temp_dir().join(format!("arko-pid-{}", uuid::Uuid::new_v4()));
        let source = StopSource::new();
        let command = StopAwareCommand::new(shell(&format!(
            "trap '' TERM; echo started; sleep 30 & echo $! > '{}'; while true; do wait; done",
            pid_file.display()
        )))
        .grace(Duration::from_millis(200))
        .terminate_grace(Duration::from_millis(200));
        let run = tokio::spawn({
            let signal = source.signal();
            async move { command.output(&signal).await }
        });
        let grandchild = read_pid(&pid_file).await;
        source.stop("stopping");
        let output = tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(output.stopped, Some(ChildStop::Killed));
        assert_eq!(output.status.signal(), Some(9));
        assert_eq!(String::from_utf8_lossy(&output.stdout), "started\n");
        assert_gone(grandchild).await;
        let _ = std::fs::remove_file(pid_file);
    }

    /// The process ID a test's child wrote to `path`, once it has.
    #[cfg(unix)]
    async fn read_pid(path: &Path) -> i32 {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(pid) = std::fs::read_to_string(path)
                    .ok()
                    .and_then(|text| text.trim().parse::<i32>().ok())
                {
                    return pid;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the child wrote its process ID")
    }

    /// A child that ignores the stop file but not `SIGTERM` ends at
    /// `SIGTERM`.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_that_ignores_the_stop_file_is_terminated() {
        let source = StopSource::new();
        source.stop("stopping");
        let output = StopAwareCommand::new(shell("while true; do sleep 0.02; done"))
            .grace(Duration::from_millis(200))
            .terminate_grace(Duration::from_secs(10))
            .output(&source.signal())
            .await
            .unwrap();
        assert_eq!(output.stopped, Some(ChildStop::Terminated));
        assert!(!output.status.success());
    }

    /// Dropping the call (an interruptible call cancelled, or past its time
    /// limit) kills the child's whole process group.
    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_the_call_kills_the_process_group() {
        let pid_file = std::env::temp_dir().join(format!("arko-pid-{}", uuid::Uuid::new_v4()));
        let command = StopAwareCommand::new(shell(&format!(
            "sleep 30 & echo $! > '{}'; wait",
            pid_file.display()
        )));
        let run = tokio::spawn(async move { command.output(&StopSignal::never()).await });
        let grandchild = read_pid(&pid_file).await;
        run.abort();
        let _ = run.await;
        assert_gone(grandchild).await;
        let _ = std::fs::remove_file(pid_file);
    }

    /// A child that exits within the grace leaves a process it started in
    /// its group: that process is killed once the child has ended.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_process_left_by_a_child_that_exits_in_the_grace_is_killed() {
        let pid_file = std::env::temp_dir().join(format!("arko-pid-{}", uuid::Uuid::new_v4()));
        let source = StopSource::new();
        let command = StopAwareCommand::new(shell(&format!(
            "sleep 30 >/dev/null 2>&1 & echo $! > '{}'; while [ ! -f \"$ARKO_STOP_FILE\" ]; do sleep 0.02; done; exit 0",
            pid_file.display()
        )))
        .grace(Duration::from_secs(10));
        let run = tokio::spawn({
            let signal = source.signal();
            async move { command.output(&signal).await }
        });
        let grandchild = read_pid(&pid_file).await;
        source.stop("stopping");
        let output = tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(output.stopped, Some(ChildStop::Exited));
        assert!(output.status.success());
        assert_gone(grandchild).await;
        let _ = std::fs::remove_file(pid_file);
    }

    /// Set for [`exits_after_killing_stop_aware_commands`]: the file its
    /// child's child writes its process ID to.
    #[cfg(unix)]
    const KILL_PID_FILE_ENV: &str = "ARKO_TEST_KILL_PID_FILE";

    /// `kill_stop_aware_commands` kills what the running children started,
    /// as a host does before it exits without unwinding. Run in a child
    /// process of this test binary, since it kills every such child of its
    /// process.
    #[cfg(unix)]
    #[test]
    fn kill_stop_aware_commands_kills_the_children_before_an_exit() {
        let pid_file = std::env::temp_dir().join(format!("arko-pid-{}", uuid::Uuid::new_v4()));
        let helper = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "stop::tests::exits_after_killing_stop_aware_commands",
                "--exact",
                "--test-threads=1",
            ])
            .env(KILL_PID_FILE_ENV, &pid_file)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert_eq!(
            helper.status.code(),
            Some(7),
            "the helper exited: {}{}",
            String::from_utf8_lossy(&helper.stdout),
            String::from_utf8_lossy(&helper.stderr)
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let grandchild = read_pid(&pid_file).await;
            assert_gone(grandchild).await;
        });
        let _ = std::fs::remove_file(pid_file);
    }

    /// The child process of
    /// [`kill_stop_aware_commands_kills_the_children_before_an_exit`]: runs
    /// a command that starts a long process, kills the commands and exits
    /// at once (with status 7). Does nothing in a normal test run.
    #[cfg(unix)]
    #[tokio::test]
    async fn exits_after_killing_stop_aware_commands() {
        let Some(pid_file) = std::env::var_os(KILL_PID_FILE_ENV) else {
            return;
        };
        let pid_file = std::path::PathBuf::from(pid_file);
        let command = StopAwareCommand::new(shell(&format!(
            "sleep 30 & echo $! > '{}'; wait",
            pid_file.display()
        )));
        let _run = tokio::spawn(async move { command.output(&StopSignal::never()).await });
        read_pid(&pid_file).await;
        kill_stop_aware_commands();
        std::process::exit(7);
    }

    /// Waits until process `pid` no longer exists (it was reaped).
    #[cfg(unix)]
    async fn assert_gone(pid: i32) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("process {pid} is still running"));
    }
}
