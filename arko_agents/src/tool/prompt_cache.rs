use std::time::{Duration, SystemTime};

use arko_ai::PromptCacheLifetime;

/// A task's prompt cache as a call sees it
/// ([`TaskContext::prompt_cache`](super::TaskContext::prompt_cache)). A tool
/// that waits keeps the wait inside it, so the model's next request still
/// finds the prompt cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PromptCacheWindow {
    /// How long the provider keeps the cache after its last use.
    pub lifetime: PromptCacheLifetime,
    /// When the task's latest model request was sent, which is when it used
    /// the cache. `None` when the task has no request on record.
    pub last_used: Option<SystemTime>,
}

impl Default for PromptCacheWindow {
    /// An unknown lifetime, as for a call made outside a runtime.
    fn default() -> Self {
        Self::new(PromptCacheLifetime::Unknown, None)
    }
}

impl PromptCacheWindow {
    pub fn new(lifetime: PromptCacheLifetime, last_used: Option<SystemTime>) -> Self {
        Self {
            lifetime,
            last_used,
        }
    }

    /// The longest a call may wait from now and still end before the cache
    /// lapses, with the margin of [`PromptCacheLifetime::longest_wait`]. A
    /// known lifetime counts from [`last_used`](Self::last_used), so the time
    /// the model took to make the call is included. An unknown lifetime, or
    /// caching off, gives the 5-minute cache's wait from now.
    pub fn longest_wait(&self) -> Duration {
        self.longest_wait_at(SystemTime::now())
    }

    fn longest_wait_at(&self, now: SystemTime) -> Duration {
        let wait = self.lifetime.longest_wait();
        match (self.lifetime, self.last_used) {
            (PromptCacheLifetime::Known(_), Some(last_used)) => {
                wait.saturating_sub(now.duration_since(last_used).unwrap_or_default())
            }
            _ => wait,
        }
    }

    /// When the cache lapses if the task makes no request before. `None`
    /// unless the lifetime is known and the task has made a request.
    pub fn lapses_at(&self) -> Option<SystemTime> {
        match (self.lifetime, self.last_used) {
            (PromptCacheLifetime::Known(lifetime), Some(last_used)) => {
                last_used.checked_add(lifetime)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: Duration = Duration::from_secs(60);

    #[test]
    fn a_known_lifetime_counts_from_the_last_request() {
        let now = SystemTime::now();
        let used = |ago: Duration| {
            PromptCacheWindow::new(PromptCacheLifetime::Known(30 * MINUTE), Some(now - ago))
        };
        assert_eq!(used(Duration::ZERO).longest_wait_at(now), 27 * MINUTE);
        assert_eq!(used(2 * MINUTE).longest_wait_at(now), 25 * MINUTE);
        // Past the margin there is nothing left, and a clock that ran back
        // leaves the whole wait.
        assert_eq!(used(28 * MINUTE).longest_wait_at(now), Duration::ZERO);
        assert_eq!(
            used(Duration::ZERO).longest_wait_at(now - MINUTE),
            27 * MINUTE
        );
        assert_eq!(used(2 * MINUTE).lapses_at(), Some(now + 28 * MINUTE));
    }

    #[test]
    fn without_a_request_or_a_lifetime_the_wait_is_the_fixed_one() {
        let now = SystemTime::now();
        let window = |lifetime, last_used| PromptCacheWindow::new(lifetime, last_used);
        let unrecorded = window(PromptCacheLifetime::Known(5 * MINUTE), None);
        assert_eq!(unrecorded.longest_wait_at(now), Duration::from_secs(270));
        assert_eq!(unrecorded.lapses_at(), None);
        for lifetime in [PromptCacheLifetime::Unknown, PromptCacheLifetime::Disabled] {
            // The model's thinking time doesn't count where no cache lapses.
            let window = window(lifetime, Some(now - 10 * MINUTE));
            assert_eq!(window.longest_wait_at(now), Duration::from_secs(270));
            assert_eq!(window.lapses_at(), None);
        }
        assert_eq!(
            PromptCacheWindow::default().longest_wait(),
            Duration::from_secs(270)
        );
    }
}
