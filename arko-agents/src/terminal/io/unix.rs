use super::{ChildProcess, ControlPaths, Reader, Writer};
use crate::terminal::TerminalConfig;
use std::io;
use std::os::fd::OwnedFd;
use std::pin::Pin;
use std::process::Stdio;
use std::task::{Context, Poll, ready};

use nix::fcntl::{FcntlArg, FdFlag, OFlag, fcntl};
use nix::pty::{Winsize, openpty};
use nix::sys::termios::{LocalFlags, SetArg, tcgetattr, tcsetattr};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf, unix::AsyncFd};
use tokio::process::Command;

pub(super) type Child = tokio::process::Child;

/// Nonblocking descriptor; dropping an async read never leaves a blocking worker.
struct Descriptor(AsyncFd<OwnedFd>);
impl Descriptor {
    fn new(fd: OwnedFd) -> io::Result<Self> {
        let flags = OFlag::from_bits_truncate(fcntl(&fd, FcntlArg::F_GETFL)?);
        fcntl(&fd, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
        Ok(Self(AsyncFd::new(fd)?))
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
    pid: u32,
    armed: bool,
}
impl ProcessGroup {
    pub fn kill(&mut self) {
        if std::mem::take(&mut self.armed) {
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(self.pid as i32),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.kill();
    }
}

pub(super) fn spawn(
    config: &TerminalConfig,
    command: &str,
    cwd: &std::path::Path,
    terminal: bool,
    paths: &ControlPaths,
) -> io::Result<ChildProcess> {
    let mut process = Command::new(&config.shell);
    process
        .args(&config.shell_args)
        .arg(command)
        .current_dir(cwd)
        .kill_on_drop(true)
        .env("ARKO_INPUT_REQUEST_PATH", &paths.requests)
        .env("ARKO_INPUT_REPLY_PATH", &paths.replies);
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
    // SAFETY: these are async-signal-safe syscalls only, with no allocation or
    // lock acquisition in the child. Each child gets a fresh session and PTY.
    unsafe {
        process.pre_exec(move || {
            if nix::libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            if terminal && nix::libc::ioctl(0, nix::libc::TIOCSCTTY as _, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = process.spawn()?;
    let group = ProcessGroup {
        pid: child
            .id()
            .ok_or_else(|| io::Error::other("child has no PID"))?,
        armed: true,
    };
    let input = pty_input.unwrap_or_else(|| Box::pin(child.stdin.take().expect("piped stdin")));
    let output = match pty_output {
        Some(reader) => vec![reader],
        None => vec![
            Box::pin(child.stdout.take().expect("piped stdout")) as Reader,
            Box::pin(child.stderr.take().expect("piped stderr")) as Reader,
        ],
    };
    Ok(ChildProcess {
        child,
        input,
        output,
        group,
    })
}
