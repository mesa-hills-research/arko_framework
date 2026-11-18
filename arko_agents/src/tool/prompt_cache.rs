//! The deadline a wait keeps to so the next request still finds the prompt
//! cached.
//!
//! # One deadline for a request
//!
//! A request has one cache deadline, `anchor + longest_wait`
//! ([`PromptCacheLifetime::longest_wait`], the lifetime less its margin). The
//! runtime works it out once, when it records the request, and puts it in
//! the [`PromptCacheWindow`] of every call the request's response makes.
//! `terminal_control wait`, `wait_for_message` and `wait_for_agents` all take
//! it from there (`PromptCacheWindow::deadline`). No wait works out a request's
//! age on its own, so no wait can give a request more time than another.
//!
//! # The anchor is early
//!
//! The anchor is a moment no later than the one the provider could first have
//! used the cache at. If the anchor is wrong, the request looks older than it
//! is and never younger, so a wait ends before the cache lapses. The anchor is
//! not when the reply came. That can be minutes after the request, and for a
//! reply that isn't streamed it is the end of the reply.
//!
//! | Runtime | Anchor | Why it is no later than the cache's use |
//! |---|---|---|
//! | Arko's own | Just before the provider call that got the response | The provider can't read the request before it is sent. A retry after a failed attempt uses its own attempt's time. |
//! | Claude Code | The latest time Arko handed the CLI something before the response's first token or whole reply came. That is the user's message sent, a tool result returned to it (from `ToolHost`), or a hook request received (the CLI sends its next request only after Arko's answer to one). | The CLI sends the request after it gets what Arko handed it, so the hand-over is earlier. Anything Arko sees later, such as the first streamed token, isn't a bound. |
//!
//! A request with no anchor can't say how old its cache is. That is a task loaded
//! from a save, a request to the refusal fallback's model (whose cache the
//! task's latest request doesn't record) and a Claude Code session that
//! handed over nothing yet. With a known lifetime such a window leaves no
//! wait, and the next request gives it an anchor. With an unknown lifetime or
//! caching off, the wait is the 5-minute cache's from the start of the wait,
//! since nothing lapses.
//!
//! Claude Code's anchor could be later than the request only if Arko handed
//! the CLI something while the main request was in flight, which a subagent's
//! hooks can cause. So a hook counts only for a tool call the main
//! conversation made, and a tool result only for such a call.
//!
//! # Waits only shorten
//!
//! A wait lasts the shorter of what it was asked for and what is left until
//! the cache deadline (`PromptCacheWindow::cap`). There is no minimum
//! wait, and nothing after the start of the wait moves the deadline. That
//! includes the wait for a message to reach the run's command channel, which
//! may take a second after a message arrives. It ends at the cache deadline
//! and leaves the message to the next turn boundary. A configured limit
//! (`WaitForMessageTimeout::Fixed`, the model's `timeout_seconds`, `wait_ms`)
//! can shorten the wait and never lengthen it.
//!
//! Keep-warm requests, which refresh the cache, don't move the deadline. The
//! next request does.
//!
//! `LatestRequest::sent_at` stays the wall-clock time a host shows, and it is
//! the first streamed token in Claude Code. `LatestRequest::sent_instant` is
//! the anchor in both runtimes.
//!
//! # Checked arithmetic
//!
//! A deadline is `Instant::checked_add`, and `None` means a deadline too far
//! to count, which is no bound. An age is `saturating_duration_since`, on the
//! monotonic clock only. The wall clock is read for the text that says when the
//! cache lapses and for nothing else (`SystemTime::checked_add`, `None` when
//! it doesn't fit). A lifetime a host sets to any `u64` number of seconds
//! must give a value, never a panic.

use std::time::{Duration, SystemTime};

use arko_ai::PromptCacheLifetime;
use tokio::time::Instant;

/// A task's prompt cache as a call sees it
/// ([`TaskContext::prompt_cache`](super::TaskContext::prompt_cache)). A tool
/// that waits keeps the wait inside it, so the model's next request still
/// finds the prompt cached.
///
/// The cache's age is measured on the monotonic clock, the one the wait's
/// timer uses, so a change of the wall clock can't lengthen a wait. A
/// [paused Tokio clock](tokio::time::pause) moves both together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PromptCacheWindow {
    /// How long the provider keeps the cache after its last use.
    pub lifetime: PromptCacheLifetime,
    /// When the task's latest model request was sent, which is when it used
    /// the cache. `None` when this process didn't see the request, as for a
    /// task loaded from a save, so the cache's age isn't known.
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

    /// The longest a call may wait from now and still end before the cache
    /// lapses, with the margin of [`PromptCacheLifetime::longest_wait`].
    ///
    /// A known lifetime counts from [`last_used`](Self::last_used), so the
    /// time the model took to make the call is included. That is zero when
    /// the margin is spent, and also when the cache's age isn't known, since
    /// the next request, which a call that returns at once leads to, gives
    /// it one. An unknown lifetime, or caching off, has nothing to lapse and
    /// gives the 5-minute cache's wait from now.
    pub fn longest_wait(&self) -> Duration {
        self.longest_wait_at(Instant::now())
    }

    fn longest_wait_at(&self, now: Instant) -> Duration {
        let wait = self.lifetime.longest_wait();
        match (self.lifetime, self.last_used) {
            (PromptCacheLifetime::Known(_), Some(used)) => {
                wait.saturating_sub(now.saturating_duration_since(used))
            }
            (PromptCacheLifetime::Known(_), None) => Duration::ZERO,
            _ => wait,
        }
    }

    /// When the cache lapses if the task makes no request before, on the
    /// wall clock, for telling the agent. `None` unless the lifetime is
    /// known and the task's latest request is.
    pub fn lapses_at(&self) -> Option<SystemTime> {
        self.lapses_at_from(Instant::now(), SystemTime::now())
    }

    fn lapses_at_from(&self, now: Instant, wall: SystemTime) -> Option<SystemTime> {
        match (self.lifetime, self.last_used) {
            (PromptCacheLifetime::Known(lifetime), Some(used)) => {
                let left = (used + lifetime).saturating_duration_since(now);
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

    fn known(lifetime: Duration, ago: Duration) -> (PromptCacheWindow, Instant) {
        let now = Instant::now();
        let window = PromptCacheWindow::new(
            PromptCacheLifetime::Known(lifetime),
            Some(now.checked_sub(ago).unwrap()),
        );
        (window, now)
    }

    #[test]
    fn a_known_lifetime_counts_from_the_last_request() {
        let (window, now) = known(30 * MINUTE, Duration::ZERO);
        assert_eq!(window.longest_wait_at(now), 27 * MINUTE);
        let (window, now) = known(30 * MINUTE, 2 * MINUTE);
        assert_eq!(window.longest_wait_at(now), 25 * MINUTE);
        // Past the margin there is nothing left.
        let (window, now) = known(30 * MINUTE, 28 * MINUTE);
        assert_eq!(window.longest_wait_at(now), Duration::ZERO);
    }

    /// Time passing between two calls of one request shrinks the second's
    /// wait by the time, as the timer that follows it counts. The paused
    /// clock stands for any clock the wait runs on.
    #[tokio::test(start_paused = true)]
    async fn the_wait_shrinks_as_the_clock_moves() {
        let window =
            PromptCacheWindow::new(PromptCacheLifetime::Known(5 * MINUTE), Some(Instant::now()));
        assert_eq!(window.longest_wait(), Duration::from_secs(270));
        tokio::time::advance(Duration::from_secs(100)).await;
        assert_eq!(window.longest_wait(), Duration::from_secs(170));
        tokio::time::advance(Duration::from_secs(200)).await;
        assert_eq!(window.longest_wait(), Duration::ZERO);
    }

    /// The wait has no wall-clock input, so a wall clock set back can't
    /// give any of it back. Only the text of when the cache lapses reads the
    /// wall clock, and it adds what is left on the monotonic one.
    #[test]
    fn the_wall_clock_gives_no_time_back() {
        // Used 4 minutes ago, with a minute left and a wait of 30 seconds.
        let (window, now) = known(5 * MINUTE, 4 * MINUTE);
        assert_eq!(window.longest_wait_at(now), Duration::from_secs(30));
        let wall = SystemTime::now();
        assert_eq!(window.lapses_at_from(now, wall), Some(wall + MINUTE));
        // An hour back on the wall clock changes the time told, not the
        // time left.
        let back = wall - 60 * MINUTE;
        assert_eq!(window.lapses_at_from(now, back), Some(back + MINUTE));
        assert_eq!(window.longest_wait_at(now), Duration::from_secs(30));
    }

    #[test]
    fn an_unknown_age_leaves_no_wait_on_a_known_cache() {
        let window = PromptCacheWindow::new(PromptCacheLifetime::Known(5 * MINUTE), None);
        assert_eq!(window.longest_wait(), Duration::ZERO);
        assert_eq!(window.lapses_at(), None);
    }

    #[test]
    fn without_a_lifetime_the_wait_is_the_fixed_one() {
        let now = Instant::now();
        for lifetime in [PromptCacheLifetime::Unknown, PromptCacheLifetime::Disabled] {
            // The model's thinking time doesn't count where no cache lapses.
            let window = PromptCacheWindow::new(lifetime, now.checked_sub(10 * MINUTE));
            assert_eq!(window.longest_wait_at(now), Duration::from_secs(270));
            assert_eq!(window.lapses_at(), None);
        }
        assert_eq!(
            PromptCacheWindow::default().longest_wait(),
            Duration::from_secs(270)
        );
    }
}
