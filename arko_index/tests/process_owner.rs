//! The processes around the CLI's background embedding (`examples/support/process.rs`):
//! the owner a run watches, with its guard against a reused process id, the default
//! owner, and starting a process detached from the caller's descriptors.
#![cfg(unix)]
#[path = "../examples/support/process.rs"]
#[allow(dead_code)]
mod process;

use process::Owner;
use std::{
    process::Command,
    time::{Duration, Instant},
};

fn wait_until(done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn an_owner_is_alive_until_it_ends() {
    assert!(Owner::watch(std::process::id()).unwrap().alive());
    let mut child = Command::new("sleep").arg("60").spawn().unwrap();
    let owner = Owner::watch(child.id()).unwrap();
    assert!(owner.alive());
    child.kill().unwrap();
    // Ended, even before its parent reaps it.
    wait_until(|| !owner.alive());
    child.wait().unwrap();
    assert!(!owner.alive());
    assert!(Owner::watch(child.id()).is_none());
}

#[test]
fn a_reused_process_id_is_not_the_owner() {
    let pid = std::process::id();
    assert!(Owner::watch(pid).unwrap().alive());
    // The same id with another start time: a process that got the id of an owner that
    // ended.
    assert!(!Owner::with_start(pid, u64::MAX).alive());
}

#[test]
fn the_default_owner_is_a_running_ancestor() {
    let pid = process::default_owner().expect("a session leader or a parent");
    assert_ne!(pid, std::process::id());
    assert!(Owner::watch(pid).is_some_and(|o| o.alive()));
}

#[test]
fn a_detached_process_has_its_own_session_and_none_of_the_callers_descriptors() {
    // A descriptor this process holds without close-on-exec, as a wrapper's lock file
    // would be.
    let file = tempfile::tempfile().unwrap();
    let fd = 57;
    assert_eq!(
        unsafe { libc::dup2(std::os::fd::AsRawFd::as_raw_fd(&file), fd) },
        fd
    );
    let listing = |detached: bool| {
        let mut command = Command::new("sh");
        // macOS's ps has no session column. A new session also makes the shell the
        // leader of its own process group, which both systems' ps can show.
        command
            .arg("-c")
            .arg("ls /dev/fd; echo $$; ps -o pgid= -p $$");
        if detached {
            process::detach(&mut command);
        }
        String::from_utf8(command.output().unwrap().stdout).unwrap()
    };
    let inherited = listing(false);
    assert!(inherited.lines().any(|l| l.trim() == "57"), "{inherited}");
    let detached = listing(true);
    assert!(!detached.lines().any(|l| l.trim() == "57"), "{detached}");
    // (the shell's pid, its process group)
    let group = |out: &str| {
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        (
            lines[lines.len() - 2].to_owned(),
            lines[lines.len() - 1].to_owned(),
        )
    };
    let (pid, pgid) = group(&inherited);
    assert_ne!(pid, pgid, "{inherited}");
    let (pid, pgid) = group(&detached);
    assert_eq!(pid, pgid, "{detached}");
    unsafe { libc::close(fd) };
}
