//! The processes around the CLI's background embedding: the owner it belongs to, which
//! it watches so that it ends when the owner ends, and starting it detached from the
//! CLI's terminal and descriptors.

use std::process::Command;

/// A process watched for its end. On Unix it is identified by its id and its start
/// time, so another process that later gets the same id is not taken for it (Linux
/// reads the start time from `/proc`, macOS from `proc_pidinfo`, and other Unix
/// systems check the id only). On Windows a handle to the process is held, which keeps
/// its id from being reused.
pub struct Owner {
    pid: u32,
    #[cfg(unix)]
    start: Option<u64>,
    #[cfg(windows)]
    handle: windows::Handle,
}

impl Owner {
    /// Watches the running process `pid`. `None` when there is none.
    pub fn watch(pid: u32) -> Option<Owner> {
        #[cfg(unix)]
        {
            let (running, start) = unix::state(pid)?;
            running.then_some(Owner { pid, start })
        }
        #[cfg(windows)]
        {
            Some(Owner {
                pid,
                handle: windows::open(pid)?,
            })
        }
    }

    /// An owner recorded with another start time than its process has: a process that
    /// reused the id of one that ended.
    #[cfg(all(test, unix))]
    pub fn with_start(pid: u32, start: u64) -> Owner {
        Owner {
            pid,
            start: Some(start),
        }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Whether the process is still running: the same process, not a zombie.
    pub fn alive(&self) -> bool {
        #[cfg(unix)]
        {
            unix::state(self.pid).is_some_and(|(running, start)| running && start == self.start)
        }
        #[cfg(windows)]
        {
            windows::alive(&self.handle)
        }
    }
}

/// The owner of a background run this process starts: the program that ran the CLI.
/// On Unix it is the leader of this process's session (the terminal's shell, or the
/// harness that started a session), which outlives the `sh -c` that tools and harnesses
/// often run a command through, or this process's parent when the session leader is
/// init or this process. On Windows it is the parent process.
pub fn default_owner() -> Option<u32> {
    #[cfg(unix)]
    {
        let own = std::process::id();
        let leader = unsafe { libc::getsid(0) };
        match u32::try_from(leader) {
            Ok(leader) if leader > 1 && leader != own => Some(leader),
            _ => Some(std::os::unix::process::parent_id()).filter(|&p| p > 1),
        }
    }
    #[cfg(windows)]
    {
        windows::parent()
    }
}

/// Prepares `command` to start a process that outlives this one without holding what
/// this one inherited. On Unix the process gets a session of its own (closing the
/// terminal does not end it), and every descriptor from 3 up is closed when it starts
/// (the standard three are the command's own). On Windows it is a detached process in
/// a new process group, and this process's standard handles are made non-inheritable
/// first, so a caller that reads this process's output sees it end. Other inheritable
/// handles this process received from its caller still pass to it on Windows: the
/// standard library has no way to limit the handles a child inherits.
pub fn detach(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let max = unix::max_fd();
        unsafe {
            command.pre_exec(move || {
                libc::setsid();
                unix::close_on_exec_from(3, max);
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
        windows::no_inherited_std_handles();
    }
}

#[cfg(unix)]
mod unix {
    /// Whether `pid` is running (not a zombie) and its start time, where the platform
    /// gives it. `None` when there is no such process.
    pub fn state(pid: u32) -> Option<(bool, Option<u64>)> {
        if pid == 0 {
            return None;
        }
        #[cfg(target_os = "linux")]
        {
            // /proc/PID/stat: "pid (comm) state ... starttime" (field 22), and comm may
            // hold spaces and parentheses.
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
            let fields: Vec<&str> = stat
                .get(stat.rfind(')')? + 1..)?
                .split_whitespace()
                .collect();
            let running = !matches!(fields.first().copied(), Some("Z" | "X" | "x"));
            let start = fields.get(19)?.parse().ok()?;
            Some((running, Some(start)))
        }
        #[cfg(target_os = "macos")]
        {
            let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
            let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
            let n = unsafe {
                libc::proc_pidinfo(
                    libc::pid_t::try_from(pid).ok()?,
                    libc::PROC_PIDTBSDINFO,
                    0,
                    (&mut info as *mut libc::proc_bsdinfo).cast(),
                    size,
                )
            };
            if n != size {
                return None;
            }
            let start = info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec;
            Some((info.pbi_status != libc::SZOMB, Some(start)))
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let exists = unsafe { libc::kill(libc::pid_t::try_from(pid).ok()?, 0) } == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
            exists.then_some((true, None))
        }
    }

    /// The highest descriptor number to mark close-on-exec when `close_range` can't:
    /// the soft limit on open files, at most 2^20.
    pub fn max_fd() -> libc::c_int {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
            return 1024;
        }
        limit.rlim_cur.min(1 << 20) as libc::c_int
    }

    /// Marks every descriptor from `first` on close-on-exec, in a child between fork and
    /// exec (async-signal-safe calls only).
    pub fn close_on_exec_from(first: libc::c_int, max: libc::c_int) {
        #[cfg(target_os = "linux")]
        {
            let done = unsafe {
                libc::syscall(
                    libc::SYS_close_range,
                    first as libc::c_uint,
                    libc::c_uint::MAX,
                    libc::CLOSE_RANGE_CLOEXEC,
                )
            } == 0;
            if done {
                return;
            }
        }
        for fd in first..max {
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
            if flags >= 0 && flags & libc::FD_CLOEXEC == 0 {
                unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) };
            }
        }
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    /// A process handle, closed on drop.
    pub struct Handle(usize);
    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.0 as *mut c_void) };
        }
    }

    const SYNCHRONIZE: u32 = 0x0010_0000;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const WAIT_TIMEOUT: u32 = 0x102;
    const HANDLE_FLAG_INHERIT: u32 = 1;

    #[repr(C)]
    #[allow(dead_code)]
    struct ProcessBasicInformation {
        exit_status: i32,
        peb: *mut c_void,
        affinity_mask: usize,
        base_priority: i32,
        process_id: usize,
        parent_process_id: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn WaitForSingleObject(handle: *mut c_void, millis: u32) -> u32;
        fn CloseHandle(handle: *mut c_void) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn SetHandleInformation(handle: *mut c_void, mask: u32, flags: u32) -> i32;
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQueryInformationProcess(
            process: *mut c_void,
            class: u32,
            info: *mut c_void,
            length: u32,
            returned: *mut u32,
        ) -> i32;
    }

    pub fn open(pid: u32) -> Option<Handle> {
        let h = unsafe { OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if h.is_null() {
            return None;
        }
        let handle = Handle(h as usize);
        alive(&handle).then_some(handle)
    }

    pub fn alive(handle: &Handle) -> bool {
        unsafe { WaitForSingleObject(handle.0 as *mut c_void, 0) == WAIT_TIMEOUT }
    }

    /// This process's parent (`ProcessBasicInformation`).
    pub fn parent() -> Option<u32> {
        let mut info = ProcessBasicInformation {
            exit_status: 0,
            peb: std::ptr::null_mut(),
            affinity_mask: 0,
            base_priority: 0,
            process_id: 0,
            parent_process_id: 0,
        };
        let status = unsafe {
            NtQueryInformationProcess(
                GetCurrentProcess(),
                0,
                (&mut info as *mut ProcessBasicInformation).cast(),
                std::mem::size_of::<ProcessBasicInformation>() as u32,
                std::ptr::null_mut(),
            )
        };
        (status == 0)
            .then(|| u32::try_from(info.parent_process_id).ok())
            .flatten()
            .filter(|&p| p != 0)
    }

    /// Makes this process's standard handles non-inheritable.
    pub fn no_inherited_std_handles() {
        use std::os::windows::io::AsRawHandle;
        for h in [
            std::io::stdin().as_raw_handle(),
            std::io::stdout().as_raw_handle(),
            std::io::stderr().as_raw_handle(),
        ] {
            if !h.is_null() {
                unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0) };
            }
        }
    }
}
