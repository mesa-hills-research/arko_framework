use super::{ChildProcess, ControlPaths, Reader, Writer};
use crate::terminal::TerminalConfig;
use std::collections::BTreeSet;
use std::io;
use std::os::fd::OwnedFd;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, ready};

use nix::fcntl::{FcntlArg, FdFlag, OFlag, fcntl};
use nix::pty::{Winsize, openpty};
use nix::sys::termios::{LocalFlags, SetArg, tcgetattr, tcsetattr};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf, unix::AsyncFd};
use tokio::process::Command;

pub(super) type Child = tokio::process::Child;

/// Armed terminal process groups in this process, for [`kill_all`].
static ARMED_GROUPS: Mutex<BTreeSet<i32>> = Mutex::new(BTreeSet::new());

fn armed_groups() -> MutexGuard<'static, BTreeSet<i32>> {
    ARMED_GROUPS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Kills the process group of every active terminal job and monitor check in
/// this process. Clones the registered group IDs before signaling so no lock
/// is held during `killpg`.
pub(crate) fn kill_all() {
    let pgids: Vec<i32> = {
        let groups = armed_groups();
        groups.iter().copied().collect()
    };
    for pgid in pgids {
        if pgid > 0 {
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(pgid),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
    }
}

#[cfg(test)]
pub(crate) fn armed_groups_snapshot() -> Vec<i32> {
    let groups = armed_groups();
    groups.iter().copied().collect()
}

/// Nonblocking descriptor. Dropping an async read never leaves a blocking worker.
struct Descriptor(AsyncFd<OwnedFd>);
impl Descriptor {
    fn new(fd: OwnedFd) -> io::Result<Self> {
        let flags = OFlag::from_bits_truncate(fcntl(&fd, FcntlArg::F_GETFL)?);
        fcntl(&fd, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
        // AsyncFd owns this OwnedFd for its whole life, satisfying I/O safety until
        // the minimum Tokio version supports register.
        #[allow(deprecated)]
        let descriptor = AsyncFd::new(fd)?;
        Ok(Self(descriptor))
    }
}
impl AsyncRead for Descriptor {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        loop {
            let mut guard = ready!(self.0.poll_read_ready(cx))?;
            match guard.try_io(|inner| {
                nix::unistd::read(inner.get_ref(), buf.initialize_unfilled())
                    .map_err(io::Error::from)
            }) {
                Ok(Ok(n)) => {
                    buf.advance(n);
                    return Poll::Ready(Ok(()));
                }
                // Linux PTYs report EIO after the last slave closes.
                Ok(Err(error)) if error.raw_os_error() == Some(nix::libc::EIO) => {
                    return Poll::Ready(Ok(()));
                }
                Ok(Err(error)) => return Poll::Ready(Err(error)),
                Err(_) => continue,
            }
        }
    }
}
impl AsyncWrite for Descriptor {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        loop {
            let mut guard = ready!(self.0.poll_write_ready(cx))?;
            match guard
                .try_io(|inner| nix::unistd::write(inner.get_ref(), buf).map_err(io::Error::from))
            {
                Ok(result) => return Poll::Ready(result),
                Err(_) => continue,
            }
        }
    }
    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Sessions own their process group, including children retaining output pipes.
pub(crate) struct ProcessGroup {
    pgid: i32,
    armed: bool,
}
impl ProcessGroup {
    fn arm(pgid: i32) -> Self {
        armed_groups().insert(pgid);
        Self { pgid, armed: true }
    }

    pub fn kill(&mut self) {
        if std::mem::take(&mut self.armed) {
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(self.pgid),
                nix::sys::signal::Signal::SIGKILL,
            );
            armed_groups().remove(&self.pgid);
        }
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Formats `score`, clamped to Linux's `-1000..=1000` range, into a stack
/// buffer so `pre_exec` can write `/proc/self/oom_score_adj` without
/// allocating.
#[cfg(target_os = "linux")]
fn format_oom_score_adj(score: i16) -> ([u8; 8], usize) {
    let clamped = score.clamp(-1000, 1000);
    let mut buf = [0u8; 8];
    let mut len = 0;
    let mut value = if clamped < 0 {
        buf[0] = b'-';
        len = 1;
        clamped.unsigned_abs()
    } else {
        clamped as u16
    };
    let start = len;
    if value == 0 {
        buf[len] = b'0';
        len += 1;
    } else {
        while value > 0 {
            buf[len] = b'0' + (value % 10) as u8;
            len += 1;
            value /= 10;
        }
        buf[start..len].reverse();
    }
    (buf, len)
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystemdScope {
    /// `systemd-run --user --scope` works and accepts `--expand-environment=no`.
    WithNoExpand,
    /// `systemd-run --user --scope` works and passes `$` verbatim without
    /// `--expand-environment=no` (older systemd).
    Plain,
    /// `systemd-run --user --scope` is absent or cannot create a user scope.
    Unavailable,
}

#[cfg(target_os = "linux")]
static SYSTEMD_SCOPE: std::sync::OnceLock<SystemdScope> = std::sync::OnceLock::new();

#[cfg(target_os = "linux")]
static SYSTEMD_PROBE_STARTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Starts the one-time `systemd-run --user --scope` probe on its own thread
/// when a memory cap is configured and the probe has not started yet.
#[cfg(target_os = "linux")]
pub(crate) fn start_memory_limit_probe(config: &TerminalConfig) {
    if config.memory_max_bytes.is_none() || SYSTEMD_SCOPE.get().is_some() {
        return;
    }
    if SYSTEMD_PROBE_STARTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("arko-systemd-scope-probe".into())
        .spawn(|| {
            let _ = systemd_scope_mode();
        });
}

/// Ensures the one-time `systemd-run --user --scope` probe has completed off
/// the Tokio worker threads.
#[cfg(target_os = "linux")]
pub(crate) async fn prepare_memory_limit(config: &TerminalConfig) {
    if config.memory_max_bytes.is_none() || SYSTEMD_SCOPE.get().is_some() {
        return;
    }
    let _ = tokio::task::spawn_blocking(systemd_scope_mode).await;
}

/// Ensures the one-time `systemd-run --user --scope` probe has completed
/// before a caller takes any manager lock.
#[cfg(target_os = "linux")]
pub(crate) fn prepare_memory_limit_sync(config: &TerminalConfig) {
    if config.memory_max_bytes.is_some() && SYSTEMD_SCOPE.get().is_none() {
        let _ = systemd_scope_mode();
    }
}

#[cfg(target_os = "linux")]
fn systemd_scope_mode() -> SystemdScope {
    *SYSTEMD_SCOPE.get_or_init(probe_systemd_scope)
}

#[cfg(target_os = "linux")]
enum ProbeOutcome {
    Exited(std::process::ExitStatus, Vec<u8>),
    NotFound,
    Failed,
    TimedOut,
}

#[cfg(target_os = "linux")]
fn run_systemd_probe(
    args: &[&str],
    capture_stdout: bool,
    timeout: std::time::Duration,
) -> ProbeOutcome {
    use std::io::Read;

    let mut command = std::process::Command::new("systemd-run");
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(if capture_stdout {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stderr(Stdio::null());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return ProbeOutcome::NotFound,
        Err(_) => return ProbeOutcome::Failed,
    };
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                if capture_stdout && let Some(pipe) = child.stdout.take() {
                    let _ = pipe.take(64).read_to_end(&mut stdout);
                }
                return ProbeOutcome::Exited(status, stdout);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return ProbeOutcome::TimedOut;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return ProbeOutcome::Failed;
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn probe_systemd_scope() -> SystemdScope {
    let timeout = std::time::Duration::from_secs(2);
    // Probe with the exact scope properties real jobs use. systemd 254+ expands
    // `$VAR` and `$$` in `--scope` arguments unless `--expand-environment=no`
    // is passed.
    let with_no_expand = [
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--expand-environment=no",
        "-p",
        "MemoryMax=67108864",
        "-p",
        "MemorySwapMax=0",
        "--",
        "true",
    ];
    match run_systemd_probe(&with_no_expand, false, timeout) {
        ProbeOutcome::Exited(status, _) if status.success() => return SystemdScope::WithNoExpand,
        ProbeOutcome::NotFound | ProbeOutcome::TimedOut => return SystemdScope::Unavailable,
        ProbeOutcome::Exited(_, _) | ProbeOutcome::Failed => {}
    }

    // Older systemd versions reject `--expand-environment=no`. Accept plain
    // `--scope` only if it passes `$$` through untouched.
    let plain = [
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        "MemoryMax=67108864",
        "-p",
        "MemorySwapMax=0",
        "--",
        "/bin/sh",
        "-c",
        "printf %s \"$1\"",
        "_",
        "$$",
    ];
    match run_systemd_probe(&plain, true, timeout) {
        ProbeOutcome::Exited(status, stdout) if status.success() && stdout == b"$$" => {
            SystemdScope::Plain
        }
        _ => SystemdScope::Unavailable,
    }
}

#[cfg(target_os = "linux")]
fn build_command_argv(
    config: &TerminalConfig,
    command: &str,
    scope: SystemdScope,
) -> (std::ffi::OsString, Vec<std::ffi::OsString>) {
    if let Some(bytes) = config.memory_max_bytes
        && scope != SystemdScope::Unavailable
    {
        let mut args = vec![
            std::ffi::OsString::from("--user"),
            std::ffi::OsString::from("--scope"),
            std::ffi::OsString::from("--quiet"),
            std::ffi::OsString::from("--collect"),
        ];
        if scope == SystemdScope::WithNoExpand {
            args.push(std::ffi::OsString::from("--expand-environment=no"));
        }
        args.push(std::ffi::OsString::from("-p"));
        args.push(std::ffi::OsString::from(format!("MemoryMax={bytes}")));
        args.push(std::ffi::OsString::from("-p"));
        args.push(std::ffi::OsString::from("MemorySwapMax=0"));
        args.push(std::ffi::OsString::from("--"));
        args.push(config.shell.as_os_str().to_os_string());
        args.extend(config.shell_args.iter().map(std::ffi::OsString::from));
        args.push(std::ffi::OsString::from(command));
        return (std::ffi::OsString::from("systemd-run"), args);
    }
    let mut args: Vec<std::ffi::OsString> = config
        .shell_args
        .iter()
        .map(std::ffi::OsString::from)
        .collect();
    args.push(std::ffi::OsString::from(command));
    (config.shell.as_os_str().to_os_string(), args)
}

pub(super) fn spawn(
    config: &TerminalConfig,
    command: &str,
    cwd: &std::path::Path,
    terminal: bool,
    paths: &ControlPaths,
) -> io::Result<ChildProcess> {
    #[cfg(target_os = "linux")]
    let mut process = {
        let scope = if config.memory_max_bytes.is_some() {
            systemd_scope_mode()
        } else {
            SystemdScope::Unavailable
        };
        let (program, args) = build_command_argv(config, command, scope);
        let mut process = Command::new(program);
        process.args(args);
        process
    };
    #[cfg(not(target_os = "linux"))]
    let mut process = {
        // Non-Linux Unix systems (including macOS) leave `memory_max_bytes`
        // unenforced because they offer no unprivileged hard physical-memory
        // cap for arbitrary child runtimes. `RLIMIT_AS` counts virtual
        // reservations such as the dyld shared cache and language runtime
        // arenas, `RLIMIT_DATA` does not cap `mmap`, and `RLIMIT_RSS` is
        // advisory.
        let mut process = Command::new(&config.shell);
        process.args(&config.shell_args).arg(command);
        process
    };
    process
        .current_dir(cwd)
        .kill_on_drop(true)
        .env("ARKO_INPUT_REQUEST_PATH", &paths.requests)
        .env("ARKO_INPUT_REPLY_PATH", &paths.replies)
        .envs(config.job_environment(terminal).iter().copied());
    let mut pty_output = None;
    let mut pty_input = None;
    if terminal {
        let pair = openpty(
            Some(&Winsize {
                ws_row: 24,
                ws_col: 80,
                ws_xpixel: 0,
                ws_ypixel: 0,
            }),
            None,
        )?;
        for fd in [&pair.master, &pair.slave] {
            fcntl(fd, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))?;
        }
        let mut settings = tcgetattr(&pair.slave)?;
        settings
            .local_flags
            .remove(LocalFlags::ECHO | LocalFlags::ECHONL);
        tcsetattr(&pair.slave, SetArg::TCSANOW, &settings)?;
        let input_master = nix::unistd::dup(&pair.master)?;
        fcntl(&input_master, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))?;
        pty_input = Some(Box::pin(Descriptor::new(input_master)?) as Writer);
        pty_output = Some(Box::pin(Descriptor::new(pair.master)?) as Reader);
        let slave = std::fs::File::from(pair.slave);
        process
            .stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?))
            .stderr(Stdio::from(slave));
    } else {
        process
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    }
    #[cfg(target_os = "linux")]
    let oom_score = config.oom_score_adj.map(format_oom_score_adj);
    // SAFETY: async-signal-safe syscalls only, with no allocation or lock
    // acquisition in the child. Each child gets a fresh session, PTY when
    // requested, and (on Linux) its configured oom_score_adj.
    unsafe {
        process.pre_exec(move || {
            if nix::libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            if terminal && nix::libc::ioctl(0, nix::libc::TIOCSCTTY as _, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            #[cfg(target_os = "linux")]
            if let Some((buf, len)) = oom_score {
                let fd = nix::libc::open(
                    c"/proc/self/oom_score_adj".as_ptr(),
                    nix::libc::O_WRONLY | nix::libc::O_CLOEXEC,
                );
                if fd >= 0 {
                    let _ = nix::libc::write(fd, buf.as_ptr().cast(), len);
                    let _ = nix::libc::close(fd);
                }
            }
            Ok(())
        });
    }
    let mut child = process.spawn()?;
    let pid = child
        .id()
        .ok_or_else(|| io::Error::other("child has no PID"))?;
    let pgid = i32::try_from(pid)
        .ok()
        .filter(|pgid| *pgid > 0)
        .ok_or_else(|| io::Error::other("child PID out of range"))?;
    let group = ProcessGroup::arm(pgid);
    let input = pty_input.unwrap_or_else(|| Box::pin(child.stdin.take().expect("piped stdin")));
    let output = match pty_output {
        Some(reader) => vec![reader],
        None => vec![
            Box::pin(child.stdout.take().expect("piped stdout")) as Reader,
            Box::pin(child.stderr.take().expect("piped stderr")) as Reader,
        ],
    };
    Ok(ChildProcess {
        group,
        child,
        input,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn format_oom_score_adj_clamps_and_formats_ascii() {
        for (input, expected) in [
            (i16::MIN, "-1000"),
            (-2000, "-1000"),
            (-1000, "-1000"),
            (-1, "-1"),
            (0, "0"),
            (500, "500"),
            (1000, "1000"),
            (2000, "1000"),
            (i16::MAX, "1000"),
        ] {
            let (buf, len) = format_oom_score_adj(input);
            assert_eq!(std::str::from_utf8(&buf[..len]).unwrap(), expected);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn build_command_argv_wraps_with_systemd_run_only_when_capped_and_available() {
        let base = TerminalConfig::new("/tmp/jobs");
        let (prog, args) = build_command_argv(&base, "echo hi", SystemdScope::WithNoExpand);
        assert_eq!(prog, "/bin/sh");
        assert_eq!(args, vec!["-c", "echo hi"]);

        let capped = base.clone().with_memory_max_bytes(Some(67_108_864));

        let (prog, args) = build_command_argv(&capped, "echo $$", SystemdScope::WithNoExpand);
        assert_eq!(prog, "systemd-run");
        assert_eq!(
            args,
            vec![
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "--expand-environment=no",
                "-p",
                "MemoryMax=67108864",
                "-p",
                "MemorySwapMax=0",
                "--",
                "/bin/sh",
                "-c",
                "echo $$",
            ]
        );

        let (prog, args) = build_command_argv(&capped, "echo $$", SystemdScope::Plain);
        assert_eq!(prog, "systemd-run");
        assert_eq!(
            args,
            vec![
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "-p",
                "MemoryMax=67108864",
                "-p",
                "MemorySwapMax=0",
                "--",
                "/bin/sh",
                "-c",
                "echo $$",
            ]
        );

        let (prog, args) = build_command_argv(&capped, "echo $$", SystemdScope::Unavailable);
        assert_eq!(prog, "/bin/sh");
        assert_eq!(args, vec!["-c", "echo $$"]);
    }

    #[test]
    fn armed_groups_handles_concurrent_inserts_removals_and_snapshots() {
        let mut handles = Vec::new();
        // Use negative sentinel IDs that cannot collide with real positive pgids.
        for worker in 0..4i32 {
            handles.push(std::thread::spawn(move || {
                for step in 0..2_000i32 {
                    let id = -((worker + 1) * 10_000 + step);
                    armed_groups().insert(id);
                    let _ = armed_groups_snapshot();
                    armed_groups().remove(&id);
                }
            }));
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        for handle in handles {
            handle.join().unwrap();
            assert!(
                std::time::Instant::now() < deadline,
                "concurrent ARMED_GROUPS access stalled"
            );
        }
        let remaining = armed_groups_snapshot();
        assert!(
            remaining.iter().all(|id| *id > 0),
            "sentinel IDs remained armed: {remaining:?}"
        );
    }
}
