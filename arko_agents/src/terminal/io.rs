//! Native terminal backends plus portable, append-only request/reply streams.
use super::TerminalConfig;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll, ready};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::time::{Sleep, sleep};

#[cfg(unix)]
#[path = "io/unix.rs"]
mod platform;
#[cfg(windows)]
#[path = "io/windows.rs"]
mod platform;

pub(super) type Reader = Pin<Box<dyn AsyncRead + Send>>;
pub(super) type Writer = Pin<Box<dyn AsyncWrite + Send>>;

struct ControlPaths {
    requests: PathBuf,
    replies: PathBuf,
}
pub(super) struct ChildProcess {
    pub child: platform::Child,
    pub input: Writer,
    pub output: Vec<Reader>,
    pub group: platform::ProcessGroup,
}
pub(super) struct Process {
    pub child: platform::Child,
    pub input: Writer,
    pub output: Vec<Reader>,
    pub requests: Reader,
    pub replies: Writer,
    pub group: platform::ProcessGroup,
}

// Both participants append complete JSON lines and flush. EOF means "no new
// bytes yet"; a local timer tails the file without waking an agent. A file
// transport avoids OS-specific inherited descriptors and socket dependencies.
const CONTROL_LIMIT: u64 = 8 * 1024 * 1024;
struct Requests {
    file: std::fs::File,
    cursor: u64,
    timer: Pin<Box<Sleep>>,
}
impl AsyncRead for Requests {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        loop {
            let length = self.file.metadata()?.len();
            if length > CONTROL_LIMIT || length < self.cursor {
                return Poll::Ready(Err(io::Error::other(
                    "request stream exceeded 8 MiB or was truncated",
                )));
            }
            let count = self.file.read(buf.initialize_unfilled())?;
            if count > 0 {
                self.cursor += count as u64;
                buf.advance(count);
                return Poll::Ready(Ok(()));
            }
            ready!(self.timer.as_mut().poll(cx));
            self.timer
                .as_mut()
                .reset(tokio::time::Instant::now() + Duration::from_millis(20));
        }
    }
}
struct Replies {
    file: std::fs::File,
    written: u64,
}
impl AsyncWrite for Replies {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.written + bytes.len() as u64 > CONTROL_LIMIT {
            return Poll::Ready(Err(io::Error::other("reply stream exceeded 8 MiB")));
        }
        let n = self.file.write(bytes)?;
        self.written += n as u64;
        Poll::Ready(Ok(n))
    }
    fn poll_flush(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.file.flush())
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

pub(super) fn spawn(
    config: &TerminalConfig,
    command: &str,
    cwd: &Path,
    terminal: bool,
    directory: &Path,
) -> io::Result<Process> {
    let paths = ControlPaths {
        requests: directory.join("requests.ndjson"),
        replies: directory.join("replies.ndjson"),
    };
    let requests = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&paths.requests)?;
    let replies = std::fs::OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(&paths.replies)?;
    let child = platform::spawn(config, command, cwd, terminal, &paths)?;
    Ok(Process {
        child: child.child,
        input: child.input,
        output: child.output,
        group: child.group,
        requests: Box::pin(Requests {
            file: requests,
            cursor: 0,
            timer: Box::pin(sleep(Duration::from_millis(20))),
        }),
        replies: Box::pin(Replies {
            file: replies,
            written: 0,
        }),
    })
}
