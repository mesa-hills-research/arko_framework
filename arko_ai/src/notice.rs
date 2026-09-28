//! Warnings logged once per process, for settings that are dropped or can't
//! take effect on every request (an agent sends the same settings on every
//! turn, so warning each time would flood the log).

use std::collections::BTreeSet;
use std::sync::{Mutex, PoisonError};

static WARNED: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

#[cfg(test)]
thread_local! {
    /// Tests run in parallel threads of one process: each test thread gets a
    /// record of its own, as if it were a process of its own.
    static TEST_SCOPE: u64 = {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    };
}

#[cfg(not(test))]
fn scoped(key: &str) -> String {
    key.to_string()
}

#[cfg(test)]
fn scoped(key: &str) -> String {
    TEST_SCOPE.with(|scope| format!("{scope}:{key}"))
}

/// Logs `message()` at warn level the first time this process sees `key`,
/// and returns whether this was the first time. Messages must never hold
/// credentials or account ids.
pub(crate) fn warn_once(key: &str, message: impl FnOnce() -> String) -> bool {
    let first = WARNED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(scoped(key));
    if first {
        log::warn!("{}", message());
    }
    first
}

/// Whether [`warn_once`] was called with `key` (in this test thread).
#[cfg(test)]
pub(crate) fn warned(key: &str) -> bool {
    WARNED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains(&scoped(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_key_is_warned_about_once() {
        let first: Vec<bool> = (0..3)
            .map(|_| warn_once("notice-test", || "dropped".into()))
            .collect();
        assert_eq!(first, [true, false, false]);
        assert!(warned("notice-test"));
        assert!(!warned("notice-test-other"));
        // Another test thread has a record of its own.
        std::thread::spawn(|| assert!(!warned("notice-test")))
            .join()
            .unwrap();
    }
}
