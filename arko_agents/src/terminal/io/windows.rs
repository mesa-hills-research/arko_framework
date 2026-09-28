//! Windows 10 1809+ / Server 2019+ ConPTY and pipe transport.
//! A job-list process attribute assigns the child before it can execute, so
//! cancellation includes descendants without a spawn/assignment race.
use super::{ChildProcess, ControlPaths, Reader, Writer};
use crate::terminal::TerminalConfig;
use std::ffi::{OsStr, OsString, c_void};
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::ExitStatusExt;
use std::path::Path;
use std::pin::Pin;
use std::ptr::{null, null_mut};
use std::task::{Context, Poll, ready};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Sleep, sleep};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::System::Console::*;
use windows_sys::Win32::System::JobObjects::*;
use windows_sys::Win32::System::Pipes::*;
use windows_sys::Win32::System::Threading::*;

fn checked(value: i32) -> io::Result<()> {
    if value == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}
fn wide(value: &OsStr) -> io::Result<Vec<u16>> {
    let mut bytes: Vec<_> = value.encode_wide().collect();
    if bytes.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "embedded NUL"));
    }
    bytes.push(0);
    Ok(bytes)
}

// Standard Windows argv quoting. cmd /C has its own command-string grammar;
// its final argument is enclosed verbatim instead of escaping shell syntax.
fn quote(value: &OsStr) -> OsString {
    use std::os::windows::ffi::OsStringExt;
    // Keep simple arguments bare, including cmd.exe's /D /S /C switches.
    if !value.is_empty()
        && !value
            .encode_wide()
            .any(|c| matches!(c, 9 | 10 | 13 | 32 | 34))
    {
        return value.to_owned();
    }
    let mut result = vec![b'"' as u16];
    let mut slashes = 0;
    for c in value.encode_wide() {
        if c == b'\\' as u16 {
            slashes += 1;
            continue;
        }
        result.extend(std::iter::repeat_n(
            b'\\' as u16,
            if c == b'"' as u16 {
                slashes * 2 + 1
            } else {
                slashes
            },
        ));
        slashes = 0;
        result.push(c);
    }
    result.extend(std::iter::repeat_n(b'\\' as u16, slashes * 2));
    result.push(b'"' as u16);
    OsString::from_wide(&result)
}
fn command_line(config: &TerminalConfig, command: &str) -> io::Result<Vec<u16>> {
    let mut line = quote(config.shell.as_os_str());
    for arg in &config.shell_args {
        line.push(" ");
        line.push(quote(OsStr::new(arg)));
    }
    line.push(" ");
    let cmd = config.shell.file_name().is_some_and(|name| {
        name.eq_ignore_ascii_case("cmd.exe") || name.eq_ignore_ascii_case("cmd")
    });
    if cmd {
        line.push("\"");
        line.push(command);
        line.push("\"");
    } else {
        line.push(quote(OsStr::new(command)));
    }
    wide(&line)
}
fn environment(paths: &ControlPaths, extra: &[(&str, &str)]) -> io::Result<Vec<u16>> {
    let mut vars: Vec<_> = std::env::vars_os()
        .filter(|(name, _)| {
            !name.eq_ignore_ascii_case("ARKO_INPUT_REQUEST_PATH")
                && !name.eq_ignore_ascii_case("ARKO_INPUT_REPLY_PATH")
                && !extra.iter().any(|(set, _)| name.eq_ignore_ascii_case(set))
        })
        .collect();
    vars.extend(
        extra
            .iter()
            .map(|(name, value)| (OsString::from(name), OsString::from(value))),
    );
    vars.push((
        "ARKO_INPUT_REQUEST_PATH".into(),
        paths.requests.as_os_str().to_owned(),
    ));
    vars.push((
        "ARKO_INPUT_REPLY_PATH".into(),
        paths.replies.as_os_str().to_owned(),
    ));
    vars.sort_by_key(|(name, _)| name.to_string_lossy().to_uppercase());
    let mut block = Vec::new();
    for (mut name, value) in vars {
        name.push("=");
        name.push(value);
        block.extend(wide(&name)?);
    }
    block.push(0);
    Ok(block)
}
fn pipe() -> io::Result<(File, File)> {
    let mut read = null_mut();
    let mut write = null_mut();
    let security = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: null_mut(),
        bInheritHandle: 1,
    };
    // SAFETY: valid out-pointers; returned handles are each owned exactly once.
    unsafe {
        checked(CreatePipe(&mut read, &mut write, &security, 65_536))?;
        Ok((File::from_raw_handle(read), File::from_raw_handle(write)))
    }
}
fn no_inherit(file: &File) -> io::Result<()> {
    // SAFETY: file owns this live handle.
    unsafe {
        checked(SetHandleInformation(
            file.as_raw_handle(),
            HANDLE_FLAG_INHERIT,
            0,
        ))
    }
}

struct Attributes(Vec<usize>);
impl Attributes {
    fn new(count: u32) -> io::Result<Self> {
        let mut size = 0;
        // SAFETY: first call queries required size; the Vec provides aligned storage.
        unsafe {
            InitializeProcThreadAttributeList(null_mut(), count, 0, &mut size);
            if size == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut storage = vec![0usize; size.div_ceil(size_of::<usize>())];
            checked(InitializeProcThreadAttributeList(
                storage.as_mut_ptr().cast(),
                count,
                0,
                &mut size,
            ))?;
            Ok(Self(storage))
        }
    }
    fn pointer(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.0.as_mut_ptr().cast()
    }
    // Caller keeps attribute values alive through CreateProcessW.
    unsafe fn set(&mut self, kind: u32, value: *const c_void, size: usize) -> io::Result<()> {
        unsafe {
            checked(UpdateProcThreadAttribute(
                self.pointer(),
                0,
                kind as usize,
                value,
                size,
                null_mut(),
                null(),
            ))
        }
    }
}
impl Drop for Attributes {
    fn drop(&mut self) {
        unsafe {
            DeleteProcThreadAttributeList(self.pointer());
        }
    }
}

struct Console(HPCON);
impl Drop for Console {
    fn drop(&mut self) {
        let console = self.0;
        // ConPTY shutdown can wait for its output to drain. Keep that work off
        // the async executor; the independent reader continues draining.
        std::thread::spawn(move || unsafe {
            ClosePseudoConsole(console);
        });
    }
}

pub(crate) struct ProcessGroup {
    job: OwnedHandle,
    console: Option<Console>,
    armed: bool,
}
impl ProcessGroup {
    pub fn kill(&mut self) {
        if std::mem::take(&mut self.armed) {
            unsafe {
                TerminateJobObject(self.job.as_raw_handle(), 1);
            }
            self.console.take();
        }
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.kill();
    }
}

pub(crate) struct Child {
    handle: OwnedHandle,
}
impl Child {
    pub async fn wait(&mut self) -> io::Result<std::process::ExitStatus> {
        loop {
            // SAFETY: the handle is owned for this call. Waiting first avoids
            // confusing a real exit code 259 with the STILL_ACTIVE sentinel.
            match unsafe { WaitForSingleObject(self.handle.as_raw_handle(), 0) } {
                WAIT_OBJECT_0 => {
                    let mut code = 0;
                    unsafe {
                        checked(GetExitCodeProcess(self.handle.as_raw_handle(), &mut code))?;
                    }
                    return Ok(std::process::ExitStatus::from_raw(code));
                }
                WAIT_TIMEOUT => sleep(Duration::from_millis(20)).await,
                _ => return Err(io::Error::last_os_error()),
            }
        }
    }
}

// Peek before reading an anonymous byte pipe: a cancelled read leaves no
// blocking worker behind, and only the available bytes are consumed.
struct PipeReader {
    file: File,
    timer: Pin<Box<Sleep>>,
}
impl AsyncRead for PipeReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        loop {
            let mut available = 0;
            let success = unsafe {
                PeekNamedPipe(
                    self.file.as_raw_handle(),
                    null_mut(),
                    0,
                    null_mut(),
                    &mut available,
                    null_mut(),
                )
            };
            if success == 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() == Some(ERROR_BROKEN_PIPE as i32) {
                    return Poll::Ready(Ok(()));
                }
                return Poll::Ready(Err(error));
            }
            if available > 0 {
                let take = buf.remaining().min(available as usize);
                let n = self.file.read(&mut buf.initialize_unfilled()[..take])?;
                buf.advance(n);
                return Poll::Ready(Ok(()));
            }
            ready!(self.timer.as_mut().poll(cx));
            self.timer
                .as_mut()
                .reset(tokio::time::Instant::now() + Duration::from_millis(10));
        }
    }
}
fn reader(file: File) -> Reader {
    Box::pin(PipeReader {
        file,
        timer: Box::pin(sleep(Duration::from_millis(10))),
    })
}

type WriteResult = oneshot::Receiver<io::Result<usize>>;
struct PipeWriter {
    sender: mpsc::Sender<(Vec<u8>, oneshot::Sender<io::Result<usize>>)>,
    pending: Option<WriteResult>,
}
impl AsyncWrite for PipeWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.pending.is_none() {
            let (response, receiver) = oneshot::channel();
            self.sender
                .try_send((bytes[..bytes.len().min(4096)].to_vec(), response))
                .map_err(|_| {
                    io::Error::new(io::ErrorKind::BrokenPipe, "terminal input worker closed")
                })?;
            self.pending = Some(receiver);
        }
        let result = ready!(Pin::new(self.pending.as_mut().unwrap()).poll(cx));
        self.pending = None;
        Poll::Ready(result.unwrap_or_else(|_| {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "terminal input worker closed",
            ))
        }))
    }
    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}
fn writer(mut file: File) -> io::Result<Writer> {
    let (sender, mut receiver) = mpsc::channel::<(Vec<u8>, oneshot::Sender<io::Result<usize>>)>(1);
    std::thread::Builder::new()
        .name("arko-terminal-input".into())
        .spawn(move || {
            while let Some((bytes, response)) = receiver.blocking_recv() {
                let result = file.write(&bytes);
                let failed = result.is_err();
                let _ = response.send(result);
                if failed {
                    break;
                }
            }
        })?;
    Ok(Box::pin(PipeWriter {
        sender,
        pending: None,
    }))
}

pub(super) fn spawn(
    config: &TerminalConfig,
    command: &str,
    cwd: &Path,
    terminal: bool,
    paths: &ControlPaths,
) -> io::Result<ChildProcess> {
    let mut command = command_line(config, command)?;
    let environment = environment(paths, config.job_environment(terminal))?;
    let cwd = wide(cwd.as_os_str())?;
    let (input_read, input_write) = pipe()?;
    let (output_read, output_write) = pipe()?;
    no_inherit(&input_write)?;
    no_inherit(&output_read)?;
    // SAFETY: all structures initialized, pointers backed by live storage,
    // handles are owned throughout attribute setup and process creation.
    unsafe {
        let job = CreateJobObjectW(null(), null());
        if job.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut group = ProcessGroup {
            job: OwnedHandle::from_raw_handle(job),
            console: None,
            armed: true,
        };
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        checked(SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            size_of_val(&limits) as u32,
        ))?;
        let jobs = [job];
        let mut attributes = Attributes::new(2)?;
        attributes.set(
            PROC_THREAD_ATTRIBUTE_JOB_LIST,
            jobs.as_ptr().cast(),
            size_of_val(&jobs),
        )?;
        let mut startup: STARTUPINFOEXW = zeroed();
        startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
        let mut output = vec![reader(output_read)];
        let error_pipe;
        let inherited;
        if terminal {
            let mut console = 0;
            let result = CreatePseudoConsole(
                COORD { X: 80, Y: 24 },
                input_read.as_raw_handle(),
                output_write.as_raw_handle(),
                0,
                &mut console,
            );
            if result < 0 {
                return Err(io::Error::other(format!(
                    "CreatePseudoConsole failed: HRESULT {result:#x}"
                )));
            }
            group.console = Some(Console(console));
            // Without this flag a child inherits this process's standard
            // handles when they are redirected (as under CI), bypassing the
            // pseudoconsole. Null handles select the pseudoconsole's own.
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            attributes.set(
                PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                console as *const c_void,
                size_of::<HPCON>(),
            )?;
            error_pipe = None;
        } else {
            let (error_read, error_write) = pipe()?;
            no_inherit(&error_read)?;
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = input_read.as_raw_handle();
            startup.StartupInfo.hStdOutput = output_write.as_raw_handle();
            startup.StartupInfo.hStdError = error_write.as_raw_handle();
            inherited = [
                input_read.as_raw_handle(),
                output_write.as_raw_handle(),
                error_write.as_raw_handle(),
            ];
            attributes.set(
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                inherited.as_ptr().cast(),
                size_of_val(&inherited),
            )?;
            output.push(reader(error_read));
            error_pipe = Some(error_write);
        }
        startup.lpAttributeList = attributes.pointer();
        let input = writer(input_write)?;
        let mut info: PROCESS_INFORMATION = zeroed();
        checked(CreateProcessW(
            null(),
            command.as_mut_ptr(),
            null(),
            null(),
            i32::from(!terminal),
            EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
            environment.as_ptr().cast(),
            cwd.as_ptr(),
            &startup.StartupInfo,
            &mut info,
        ))?;
        let thread = OwnedHandle::from_raw_handle(info.hThread);
        let child = Child {
            handle: OwnedHandle::from_raw_handle(info.hProcess),
        };
        drop(thread);
        drop(error_pipe);
        Ok(ChildProcess {
            child,
            input,
            output,
            group,
        })
    }
}
