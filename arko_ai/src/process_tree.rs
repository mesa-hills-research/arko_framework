//! A child process and the programs it starts, ended together.
//!
//! Ending only the child leaves the programs it started running. That
//! happens with a launcher such as `npx`, `uvx` or `sh -c`, which starts the
//! program you meant to run, and with a program the child left running in the
//! background. [`ProcessTree::spawn`] starts a command so that
//! [`kill`](ProcessTree::kill), or dropping the tree, ends all of them.
//!
//! On Unix the command leads a process group of its own, which `kill` sends
//! `SIGKILL`. On Windows the command is put in a job object that ends its
//! processes when it is terminated or its last handle closes.
//!
//! ```no_run
//! use arko_ai::process_tree::ProcessTree;
//!
//! # async fn run() -> std::io::Result<()> {
//! let mut command = tokio::process::Command::new("sh");
//! command.args(["-c", "sleep 300 & wait"]).kill_on_drop(true);
//! let (mut child, mut tree) = ProcessTree::spawn(&mut command)?;
//! // Ends the shell and the sleep.
//! tree.kill();
//! child.wait().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Kill the tree before collecting the child's exit status, as
//! [`ProcessTree::wait`] does. On Unix the group's ID is the child's process
//! ID, which is free for another program once the child is collected, and a
//! later kill could then reach that program's group. [`exited`] waits for the
//! child to exit without collecting it.
//!
//! # Limits
//!
//! A host that is killed outright (`SIGKILL`) or calls `std::process::exit`
//! runs no destructor. On Windows the system closes the job handle, so the
//! tree still ends. On Unix its programs keep running until they notice that
//! their input has closed.
//!
//! A program that leaves the group with `setsid` (Unix) is not reached. On
//! Windows the child joins the job right after it starts, so a program it
//! started before then stays out.

use std::io;
use std::process::ExitStatus;

use tokio::process::{Child, Command};

/// Waits until `child` has exited without collecting its exit status, so
/// its process ID stays taken until its `wait` collects it. On Unix that is
/// also the ID of a process group the child leads, as a [`ProcessTree`]'s
/// does, so the group can be killed safely until then. A child already
/// collected counts as exited. Elsewhere it waits for the child's exit, which
/// frees nothing there.
#[cfg(unix)]
pub async fn exited(child: &mut Child) {
    use tokio::signal::unix::{SignalKind, signal};
    let Some(pid) = child.id() else {
        return;
    };
    // Listening first, so an exit between the look and the wait isn't
    // missed. Without the signal, or should one be missed, it looks again
    // at intervals.
    let mut exits = signal(SignalKind::child()).ok();
    loop {
        if has_exited(pid) {
            return;
        }
        match exits.as_mut() {
            Some(exits) => {
                let _ = tokio::time::timeout(std::time::Duration::from_secs(1), exits.recv()).await;
            }
            None => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
        }
    }
}

/// Waits until `child` has exited, without collecting its exit status
/// where that frees its process ID (Unix).
#[cfg(not(unix))]
pub async fn exited(child: &mut Child) {
    let _ = child.wait().await;
}

/// Whether child `pid` has exited (or isn't a child any more), leaving it
/// uncollected.
#[cfg(unix)]
fn has_exited(pid: u32) -> bool {
    use nix::libc;
    let id = libc::id_t::from(pid);
    loop {
        // SAFETY: `info` is a zeroed `siginfo_t` that outlives the call,
        // which only writes to it. A zero `si_pid` after `WNOHANG` means no
        // change, as waitid(2) advises.
        let (result, info) = unsafe {
            let mut info: libc::siginfo_t = std::mem::zeroed();
            let result = libc::waitid(
                libc::P_PID,
                id,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            );
            (result, info)
        };
        if result == 0 {
            // SAFETY: the call filled in `info` for a child that changed.
            return unsafe { info.si_pid() } != 0;
        }
        if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
            // Not a child of this process, or collected already.
            return true;
        }
    }
}

/// A child's process group, killed by [`kill`](Self::kill) or when dropped.
#[cfg(unix)]
#[derive(Debug)]
pub struct ProcessTree {
    group: Option<nix::unistd::Pid>,
}

#[cfg(unix)]
impl ProcessTree {
    /// Starts `command` leading a process group of its own. Out of the
    /// terminal's foreground group, the child doesn't get the terminal's
    /// Ctrl-C.
    pub fn spawn(command: &mut Command) -> io::Result<(Child, Self)> {
        command.process_group(0);
        let child = command.spawn()?;
        let group = child
            .id()
            .and_then(|id| i32::try_from(id).ok())
            .map(nix::unistd::Pid::from_raw);
        Ok((child, Self { group }))
    }

    /// Kills every process of the tree at once. Later calls do nothing.
    /// Call it before the child's exit status is collected (see the
    /// module's notes).
    pub fn kill(&mut self) {
        if let Some(group) = self.group.take() {
            // The group is gone once all its processes have exited.
            let _ = nix::sys::signal::killpg(group, nix::sys::signal::Signal::SIGKILL);
        }
    }
}

/// A child's job object, killed by [`kill`](Self::kill) or when dropped.
#[cfg(windows)]
#[derive(Debug)]
pub struct ProcessTree {
    job: Option<std::os::windows::io::OwnedHandle>,
}

#[cfg(windows)]
impl ProcessTree {
    /// Starts `command` in a new job object. The command's creation flags
    /// stay as they are, so a host without a console of its own also calls
    /// [`no_console_window`](crate::process::NoConsoleWindow::no_console_window).
    pub fn spawn(command: &mut Command) -> io::Result<(Child, Self)> {
        let child = command.spawn()?;
        let tree = Self::adopt(&child);
        Ok((child, tree))
    }

    /// Puts the child in a new job object. A failure is logged and leaves the
    /// child outside a job.
    fn adopt(child: &Child) -> Self {
        use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };
        let Some(process) = child.raw_handle() else {
            return Self { job: None };
        };
        // SAFETY: the job handle is owned from creation on, the limits
        // structure is fully initialized (zeroed, then its flag set) and
        // outlives the call, and `process` is the live child's handle.
        let job = unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                log::warn!(
                    "A job object could not be created: {}",
                    io::Error::last_os_error()
                );
                return Self { job: None };
            }
            let job = OwnedHandle::from_raw_handle(job);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let limited = SetInformationJobObject(
                job.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) != 0;
            if !limited || AssignProcessToJobObject(job.as_raw_handle(), process) == 0 {
                log::warn!(
                    "A child process could not be put in a job object: {}",
                    io::Error::last_os_error()
                );
                return Self { job: None };
            }
            job
        };
        Self { job: Some(job) }
    }

    /// Kills every process of the tree at once. Later calls do nothing.
    pub fn kill(&mut self) {
        use std::os::windows::io::AsRawHandle;
        if let Some(job) = self.job.take() {
            // SAFETY: the job handle is owned and open.
            unsafe {
                windows_sys::Win32::System::JobObjects::TerminateJobObject(job.as_raw_handle(), 1);
            }
        }
    }
}

impl ProcessTree {
    /// Waits for `child`, the command the tree started, to exit, kills what
    /// it left running in the tree, and then collects its exit status. The
    /// tree is killed while the child is still uncollected, so its group
    /// can't have passed to another program.
    pub async fn wait(&mut self, child: &mut Child) -> io::Result<ExitStatus> {
        exited(child).await;
        self.kill();
        child.wait().await
    }
}

/// Other systems have no process tree to end, so this ends nothing.
#[cfg(not(any(unix, windows)))]
#[derive(Debug)]
pub struct ProcessTree;

#[cfg(not(any(unix, windows)))]
impl ProcessTree {
    /// Starts `command` as it is.
    pub fn spawn(command: &mut Command) -> io::Result<(Child, Self)> {
        Ok((command.spawn()?, Self))
    }

    /// Does nothing.
    pub fn kill(&mut self) {}
}

impl Drop for ProcessTree {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Stdio;
    use std::time::Duration;

    /// Starts `sh -c script` as a tree. The script writes the ID of the
    /// program it starts in the background to its standard output.
    async fn start(script: &str) -> (Child, ProcessTree, nix::unistd::Pid) {
        use tokio::io::AsyncBufReadExt;
        let mut command = Command::new("sh");
        command
            .args(["-c", script])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        let (mut child, tree) = ProcessTree::spawn(&mut command).unwrap();
        let mut lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let pid: i32 = lines.next_line().await.unwrap().unwrap().parse().unwrap();
        (child, tree, nix::unistd::Pid::from_raw(pid))
    }

    /// Whether the process has ended. A killed one that nobody has collected
    /// yet counts as ended.
    fn has_ended(pid: nix::unistd::Pid) -> bool {
        if nix::sys::signal::kill(pid, None).is_err() {
            return true;
        }
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| Some(stat.rsplit_once(')')?.1.trim_start().starts_with('Z')))
            .unwrap_or(false)
    }

    async fn ends(pid: nix::unistd::Pid) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !has_ended(pid) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the program is still running");
    }

    const BACKGROUND: &str = "sleep 300 & echo $!; wait";

    /// `exited` returns once the child has exited, and leaves it
    /// uncollected, so its process ID stays taken until `wait` collects it.
    #[tokio::test]
    async fn exited_leaves_the_child_uncollected() {
        let mut command = Command::new("sh");
        command
            .args(["-c", "sleep 0.2; exit 3"])
            .stdin(Stdio::null())
            .kill_on_drop(true);
        let mut child = command.spawn().unwrap();
        let pid = nix::unistd::Pid::from_raw(child.id().unwrap() as i32);
        tokio::time::timeout(Duration::from_secs(10), exited(&mut child))
            .await
            .expect("it exits");
        assert!(
            nix::sys::signal::kill(pid, None).is_ok(),
            "the exited child still holds its ID"
        );
        assert_eq!(child.wait().await.unwrap().code(), Some(3));
        assert!(nix::sys::signal::kill(pid, None).is_err());
        // Collected, it counts as exited.
        exited(&mut child).await;
    }

    /// `wait` kills what the child left running before it collects the
    /// child.
    #[tokio::test]
    async fn wait_kills_the_tree_before_collecting_the_child() {
        let (mut child, mut tree, program) = start("sleep 300 & echo $!; exit 4").await;
        let status = tokio::time::timeout(Duration::from_secs(10), tree.wait(&mut child))
            .await
            .expect("it exits")
            .unwrap();
        assert_eq!(status.code(), Some(4));
        ends(program).await;
    }

    #[tokio::test]
    async fn kill_ends_the_programs_the_child_started() {
        let (mut child, mut tree, program) = start(BACKGROUND).await;
        assert!(!has_ended(program));
        tree.kill();
        child.wait().await.unwrap();
        ends(program).await;
    }

    #[tokio::test]
    async fn dropping_the_tree_ends_the_programs_the_child_started() {
        let (_child, tree, program) = start(BACKGROUND).await;
        assert!(!has_ended(program));
        drop(tree);
        ends(program).await;
    }

    /// Killing the child alone leaves the program, which is the case the
    /// tree is for.
    #[tokio::test]
    async fn killing_only_the_child_leaves_the_programs_it_started() {
        let (mut child, tree, program) = start(BACKGROUND).await;
        child.kill().await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!has_ended(program));
        drop(tree);
        ends(program).await;
    }
}
