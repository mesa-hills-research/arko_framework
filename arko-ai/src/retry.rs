//! Retry delays shared by the framework's recovery loops.
//!
//! arko-ai itself does not retry model calls; callers (arko-agents, arko-index,
//! and the OpenAI OAuth refresh loop) decide *whether* to retry using
//! [`Error::retry_class`](crate::Error::retry_class) and use
//! [`BackoffPolicy`] to decide *how long* to wait.

use std::time::Duration;

/// Exponential backoff with an optional equal jitter and a server-supplied
/// minimum.
///
/// For attempt `n` (starting at 1) the backoff cap is
/// `min(initial * 2^(n-1), max)`. With jitter the delay is drawn from
/// `[cap / 2, cap]`. A server hint (such as `Retry-After`) is a minimum: the
/// delay is never shorter than the hint, and with jitter up to `cap / 2` is
/// added above it so callers sharing a reset time do not retry in lockstep.
/// A server hint may exceed `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackoffPolicy {
    pub initial: Duration,
    pub max: Duration,
    pub jitter: bool,
}

impl BackoffPolicy {
    /// A jittered policy.
    pub const fn new(initial: Duration, max: Duration) -> Self {
        Self {
            initial,
            max,
            jitter: true,
        }
    }

    /// Delay before retry `attempt` (1 for the first retry). `sample` is any
    /// uniformly random value; it is a parameter so tests need no random
    /// source (see [`random_sample`]).
    pub fn delay(&self, attempt: u32, server_hint: Option<Duration>, sample: u64) -> Duration {
        let initial = millis(self.initial);
        let max = millis(self.max);
        let multiplier = 1_u64
            .checked_shl(attempt.saturating_sub(1))
            .unwrap_or(u64::MAX);
        let cap = initial.saturating_mul(multiplier).min(max);
        let backoff = if self.jitter {
            let floor = cap.div_ceil(2);
            floor + sample % (cap - floor + 1)
        } else {
            cap
        };
        // Add nonnegative jitter to a server minimum too, so a shared reset
        // does not synchronize every caller. Never retry before that minimum.
        let server_jitter = if self.jitter {
            sample % (cap / 2 + 1)
        } else {
            0
        };
        let minimum = server_hint
            .unwrap_or_default()
            .saturating_add(Duration::from_millis(server_jitter));
        Duration::from_millis(backoff).max(minimum)
    }

    /// [`BackoffPolicy::delay`] with a fresh [`random_sample`].
    pub fn next_delay(&self, attempt: u32, server_hint: Option<Duration>) -> Duration {
        self.delay(attempt, server_hint, random_sample())
    }
}

/// A uniformly random `u64` for [`BackoffPolicy::delay`].
pub fn random_sample() -> u64 {
    uuid::Uuid::new_v4().as_u128() as u64
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn exponential_delay_saturates_and_jitter_stays_in_bounds() {
        let mut policy = BackoffPolicy {
            initial: Duration::from_millis(250),
            max: Duration::from_secs(64),
            jitter: false,
        };
        assert_eq!(policy.delay(1, None, 0), Duration::from_millis(250));
        assert_eq!(policy.delay(2, None, 0), Duration::from_millis(500));
        assert_eq!(policy.delay(9, None, 0), Duration::from_secs(64));
        assert_eq!(policy.delay(u32::MAX, None, 0), Duration::from_secs(64));
        // Without jitter a server hint is an exact minimum and may exceed max.
        assert_eq!(
            policy.delay(1, Some(Duration::from_secs(90)), 7),
            Duration::from_secs(90)
        );

        policy.jitter = true;
        let delays: HashSet<_> = (0..1000).map(|n| policy.delay(4, None, n)).collect();
        assert!(delays.len() > 1);
        assert!(
            delays
                .iter()
                .all(|d| *d >= Duration::from_secs(1) && *d <= Duration::from_secs(2))
        );
        let hinted: HashSet<_> = (0..1000)
            .map(|n| policy.delay(4, Some(Duration::from_secs(60)), n))
            .collect();
        assert!(hinted.len() > 1);
        assert!(
            hinted
                .iter()
                .all(|d| *d >= Duration::from_secs(60) && *d <= Duration::from_secs(61))
        );
        let random = policy.next_delay(3, None);
        assert!(random >= Duration::from_millis(500) && random <= Duration::from_secs(1));
    }

    #[test]
    fn extreme_inputs_saturate_instead_of_overflowing() {
        let policy = BackoffPolicy::new(Duration::MAX, Duration::MAX);
        assert!(policy.delay(u32::MAX, Some(Duration::MAX), u64::MAX) >= Duration::from_secs(1));
        let zero = BackoffPolicy::new(Duration::ZERO, Duration::ZERO);
        assert_eq!(zero.delay(5, None, 12345), Duration::ZERO);
    }
}
