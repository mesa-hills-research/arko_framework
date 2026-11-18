//! The deadline a wait keeps to so the next request still finds the prompt
//! cached.
//!
//! # One deadline for a request
//!
//! A request has one cache deadline, `anchor + longest_wait`
//! ([`PromptCacheLifetime::longest_wait`], the lifetime less its margin). The
//! runtime works out the anchor once, when it first sees the response, and
//! puts it in the [`PromptCacheWindow`] of every call the response makes.
//! `terminal_control wait`, `wait_for_message` and `wait_for_agents` all take
//! the deadline from there ([`PromptCacheWindow::deadline`]). No wait works
//! out a request's age on its own, so no wait can give a request more time
//! than another.
//!
//! # The anchor is early
//!
//! The anchor is a moment no later than the one the provider could first have
//! used the cache at. If the anchor is wrong, the request looks older than it
//! is and never younger, so a wait ends before the cache lapses. The anchor is
//! not when the reply came. That can be minutes after the request, and for a
//! reply that isn't streamed it is the end of the reply. Both clocks are read
//! together at the anchor, so [`LatestRequest::sent_at`](crate::task::LatestRequest::sent_at) and
//! [`LatestRequest::sent_instant`](crate::task::LatestRequest::sent_instant) name the same moment.
//!
//! | Runtime | Anchor | Why it is no later than the cache's use |
//! |---|---|---|
//! | Arko's own | Just before the provider call that got the response | The provider can't read the request before it is sent. A retry after a failed attempt uses its own attempt's time. |
//! | Claude Code | The latest hand-off to the CLI that must come before its next main request (below), frozen when the response's first token or whole reply is first seen | The CLI sends the request after it gets what Arko handed it, so the hand-off is earlier. Anything Arko sees later, such as the first streamed token, isn't a bound. |
//!
//! A Claude Code hand-off counts only when it is known to come before the
//! next main request and not while one is in flight.
//!
//! - Input that starts or resumes a turn counts, stamped before it is sent.
//!   That is sent between turns, when no main request runs. A note the user
//!   typed is sent at a hook, possibly queued for a later boundary, so it
//!   never counts. Nor does a compaction.
//! - A hook request, a permission request or an arko tool's result counts
//!   when it is for a tool call the main conversation announced in its latest
//!   response. A call that isn't announced (a subagent's, an old one, or one
//!   whose hook came before its announcement) is ignored. Its time is the
//!   moment Arko received the request, or returns the result, so it is before
//!   the CLI's next request.
//! - When the response is first seen, the latest counted hand-off is frozen as
//!   its anchor and the announced calls are forgotten. Later hand-offs belong
//!   to the next response, and a response's window never changes.
//! - Where the order or the owner is uncertain, the older anchor stays.
//!
//! A request with no anchor can't say how old its cache is. That is a task
//! loaded from a save, a request to the refusal fallback's model (whose cache
//! the task's latest request doesn't record) and a Claude Code session that
//! counted no hand-off yet. With a known lifetime such a window leaves no
//! wait, and the next request gives it an anchor. With an unknown lifetime or
//! caching off, the wait is the 5-minute cache's from the start of the wait,
//! since nothing lapses. That is not the same as a deadline too far to
//! count, which sets no bound.
//!
//! # Waits only shorten
//!
//! A wait lasts the shorter of what it was asked for and what is left until
//! the cache deadline ([`PromptCacheWindow::cap`]). There is no minimum
//! wait, and nothing after the start of the wait moves the deadline. That
//! includes the wait for a message to reach the run's command channel, which
//! may take a second after a message arrives. It can outlast a configured
//! limit by that second and never the cache deadline, where it ends and
//! leaves the message to the next turn boundary. A configured limit
//! (`WaitForMessageTimeout::Fixed`, the model's `timeout_seconds`, `wait_ms`)
//! can shorten the wait and never lengthen it.
//!
//! Keep-warm requests, which refresh the cache, don't move the deadline. The
//! next request does.
//!
//! # Checked arithmetic
//!
//! A deadline is `Instant::checked_add`. `None` means a deadline too far to
//! count, which is no bound, and a deadline at or before the start of the wait
//! means no wait. An age is `saturating_duration_since`, on the monotonic
//! clock only. The wall clock is read for the text that says when the cache
//! lapses and for nothing else (`SystemTime::checked_add`, `None` when it
//! doesn't fit). A lifetime a host sets to any `u64` number of seconds must
//! give a value, never a panic.

use std::time::{Duration, SystemTime};

use arko_ai::PromptCacheLifetime;
use tokio::time::Instant;

/// A task's prompt cache as a call sees it
/// ([`TaskContext::prompt_cache`](super::TaskContext::prompt_cache)). A tool
/// that waits keeps the wait inside it, so the model's next request still
/// finds the prompt cached. The module's text says how.
///
/// The cache's age is measured on the monotonic clock, the one the wait's
/// timer uses, so a change of the wall clock can't lengthen a wait. A
/// [paused Tokio clock](tokio::time::pause) moves both together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PromptCacheWindow {
    /// How long the provider keeps the cache after its last use.
    pub lifetime: PromptCacheLifetime,
    /// The request's anchor, a moment no later than the one it used the
    /// cache at ([`LatestRequest::sent_instant`](crate::task::LatestRequest::sent_instant)).
    /// `None` when there is no anchor to trust, as for a task loaded from a
    /// save, so the cache's age isn't known.
    pub last_used: Option<Instant>,
}

impl Default for PromptCacheWindow {
    /// An unknown lifetime, as for a call made outside a runtime.
    fn default() -> Self {
        Self::new(PromptCacheLifetime::Unknown, None)
    }
}

impl PromptCacheWindow {
    pub fn new(lifetime: PromptCacheLifetime, last_used: Option<Instant>) -> Self {
        Self {
            lifetime,
            last_used,
        }
    }

    /// The latest moment a wait that starts at `start` may end, which is
    /// [`PromptCacheLifetime::longest_wait`] after the anchor. It is the same
    /// moment for every wait of a request.
    ///
    /// - `Some(moment)`, at or before `start`, leaves no wait. A known
    ///   lifetime with no anchor gives `start` itself.
    /// - `Some(moment)` after `start` is the end of the wait. With an unknown
    ///   lifetime, or caching off, nothing lapses, and it is the 5-minute
    ///   cache's wait from `start`.
    /// - `None` is a moment too far from the anchor to count, which sets
    ///   no bound.
    pub fn deadline(&self, start: Instant) -> Option<Instant> {
        let wait = self.lifetime.longest_wait();
        match (self.lifetime, self.last_used) {
            (PromptCacheLifetime::Known(_), Some(anchor)) => anchor.checked_add(wait),
            (PromptCacheLifetime::Known(_), None) => Some(start),
            _ => start.checked_add(wait),
        }
    }

    /// How long a wait that starts at `start` and was asked to last `wait`
    /// lasts, which is `wait` unless the [deadline](Self::deadline) comes
    /// first. It never gives more than `wait`.
    pub fn cap(&self, start: Instant, wait: Duration) -> Duration {
        match self.deadline(start) {
            Some(deadline) => wait.min(deadline.saturating_duration_since(start)),
            None => wait,
        }
    }

    /// When the cache lapses if the task makes no request before, on the
    /// wall clock, for telling the agent. `None` unless the lifetime is
    /// known, the anchor is, and the moment can be counted.
    pub fn lapses_at(&self) -> Option<SystemTime> {
        self.lapses_at_from(Instant::now(), SystemTime::now())
    }

    fn lapses_at_from(&self, now: Instant, wall: SystemTime) -> Option<SystemTime> {
        match (self.lifetime, self.last_used) {
            (PromptCacheLifetime::Known(lifetime), Some(anchor)) => {
                let left = anchor.checked_add(lifetime)?.saturating_duration_since(now);
                wall.checked_add(left)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: Duration = Duration::from_secs(60);
    const HOUR: Duration = Duration::from_secs(3600);

    fn known(lifetime: Duration, ago: Duration) -> (PromptCacheWindow, Instant) {
        let now = Instant::now();
        let window = PromptCacheWindow::new(
            PromptCacheLifetime::Known(lifetime),
            Some(now.checked_sub(ago).unwrap()),
        );
        (window, now)
    }

    #[test]
    fn a_known_lifetime_counts_from_the_anchor() {
        for (lifetime, ago, left) in [
            (30 * MINUTE, Duration::ZERO, 27 * MINUTE),
            (30 * MINUTE, 2 * MINUTE, 25 * MINUTE),
            // Past the margin there is nothing left.
            (30 * MINUTE, 28 * MINUTE, Duration::ZERO),
        ] {
            let (window, now) = known(lifetime, ago);
            assert_eq!(window.cap(now, Duration::MAX), left);
            assert_eq!(window.cap(now, MINUTE), MINUTE.min(left));
        }
    }

    /// Time passing between two calls of one request shrinks the second's
    /// wait by that time, as the timer that follows it counts. The paused
    /// clock stands for any clock the wait runs on.
    #[tokio::test(start_paused = true)]
    async fn the_wait_shrinks_as_the_clock_moves() {
        let window =
            PromptCacheWindow::new(PromptCacheLifetime::Known(5 * MINUTE), Some(Instant::now()));
        let wait = |window: &PromptCacheWindow| window.cap(Instant::now(), 4 * HOUR);
        assert_eq!(wait(&window), Duration::from_secs(270));
        tokio::time::advance(Duration::from_secs(100)).await;
        assert_eq!(wait(&window), Duration::from_secs(170));
        tokio::time::advance(Duration::from_secs(200)).await;
        assert_eq!(wait(&window), Duration::ZERO);
    }

    /// The wait has no wall-clock input, so a wall clock set back can't
    /// give any of it back. Only the text of when the cache lapses reads the
    /// wall clock, and it adds what is left on the monotonic one.
    #[test]
    fn the_wall_clock_gives_no_time_back() {
        // Used 4 minutes ago, with a minute left and a wait of 30 seconds.
        let (window, now) = known(5 * MINUTE, 4 * MINUTE);
        assert_eq!(window.cap(now, HOUR), Duration::from_secs(30));
        let wall = SystemTime::now();
        assert_eq!(window.lapses_at_from(now, wall), Some(wall + MINUTE));
        // An hour back on the wall clock changes the time told, not the
        // time left.
        let back = wall - HOUR;
        assert_eq!(window.lapses_at_from(now, back), Some(back + MINUTE));
        assert_eq!(window.cap(now, HOUR), Duration::from_secs(30));
    }

    /// A known lifetime whose anchor isn't known has no wait, and its
    /// deadline is the start itself.
    #[test]
    fn a_missing_anchor_leaves_no_wait() {
        let window = PromptCacheWindow::new(PromptCacheLifetime::Known(5 * MINUTE), None);
        let start = Instant::now();
        assert_eq!(window.deadline(start), Some(start));
        assert_eq!(window.cap(start, HOUR), Duration::ZERO);
        assert_eq!(window.lapses_at(), None);
    }

    /// An unknown lifetime and caching off are fixed from the start of the
    /// wait, however old the request is.
    #[test]
    fn without_a_lifetime_the_wait_is_the_fixed_one() {
        let now = Instant::now();
        for lifetime in [PromptCacheLifetime::Unknown, PromptCacheLifetime::Disabled] {
            let window = PromptCacheWindow::new(lifetime, now.checked_sub(10 * MINUTE));
            assert_eq!(
                window.deadline(now),
                Some(now + Duration::from_secs(270)),
                "{lifetime:?}"
            );
            assert_eq!(window.cap(now, HOUR), Duration::from_secs(270));
            assert_eq!(window.cap(now, MINUTE), MINUTE);
            assert_eq!(window.lapses_at(), None);
        }
        assert_eq!(
            PromptCacheWindow::default().cap(now, HOUR),
            Duration::from_secs(270)
        );
    }

    /// A deadline too far from the anchor to count sets no bound, which is
    /// the opposite of a missing anchor. Nothing panics, through the
    /// deadline, the cap and the time the cache lapses, for any lifetime.
    #[test]
    fn a_deadline_too_far_to_count_sets_no_bound() {
        let start = Instant::now();
        for lifetime in [
            Duration::from_secs(u64::MAX),
            Duration::MAX,
            Duration::from_secs(u64::MAX / 2),
            Duration::from_secs(i64::MAX as u64),
        ] {
            let window = PromptCacheWindow::new(PromptCacheLifetime::Known(lifetime), Some(start));
            // Too far to count, or counted and far past any wait.
            let deadline = window.deadline(start);
            assert!(
                deadline.is_none_or(|deadline| deadline > start + 24 * HOUR),
                "{lifetime:?}"
            );
            assert_eq!(window.cap(start, 4 * HOUR), 4 * HOUR);
            let _ = window.lapses_at();
        }
        let window = PromptCacheWindow::new(
            PromptCacheLifetime::Known(Duration::from_secs(u64::MAX)),
            Some(start),
        );
        // On Unix an instant counts its seconds in an i64, so this deadline
        // is too far to count. Windows counts them in a u64 and gives the
        // far deadline the loop above accepts.
        #[cfg(unix)]
        assert_eq!(window.deadline(start), None);
        assert_eq!(window.lapses_at(), None);
        // One that fits is counted, and an hour of it is the cache's.
        let window = PromptCacheWindow::new(PromptCacheLifetime::Known(HOUR), Some(start));
        assert_eq!(window.deadline(start), Some(start + 54 * MINUTE));
        let missing = PromptCacheWindow::new(PromptCacheLifetime::Known(Duration::MAX), None);
        assert_eq!(missing.deadline(start), Some(start));
    }
}
