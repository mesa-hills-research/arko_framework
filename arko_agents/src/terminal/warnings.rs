//! A monitor's warning before it ends, and keeping it longer with a rearm.
//!
//! The warning comes [`warning_before`](MonitorLimits::warning_before) its
//! end, for monitors armed for at least
//! [`warned_from`](MonitorLimits::warned_from). A host can say when an
//! agent's prompt cache expires, and when its user last wrote. A warning
//! that would come after the owner's cache expires then comes before it
//! instead, so the turn it starts for an idle agent reads the conversation
//! from the cache, and a warning stops waking an idle agent once its user
//! hasn't written for a while.

use super::monitor::{Arming, MonitorLimits};
use super::*;
use std::time::SystemTime;
use tokio::time::Instant;

/// What decides when a monitor's warning comes and whether it wakes its
/// owner. The host's wall-clock times are kept as the instants they were
/// when it gave them.
#[derive(Debug)]
pub(super) struct WakeRules {
    /// When each owner's prompt cache expires, as its host last said.
    caches: HashMap<Uuid, Instant>,
    /// When the host's user last wrote, or the manager was made.
    user_active_at: Instant,
    /// How long after that warnings still wake an idle owner. `None` is
    /// always.
    warnings_wake_within: Option<Duration>,
}

impl Default for WakeRules {
    fn default() -> Self {
        Self {
            caches: HashMap::new(),
            user_active_at: Instant::now(),
            warnings_wake_within: Some(DEFAULT_WARNINGS_WAKE_WITHIN),
        }
    }
}

/// How long after its user last wrote a monitor's warning still wakes an
/// idle agent by default (2 hours).
pub const DEFAULT_WARNINGS_WAKE_WITHIN: Duration = Duration::from_secs(2 * 3600);

/// What [`warning_due`] needs to know of an owner.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct WakeInputs {
    /// When its prompt cache expires.
    pub(super) cache: Option<Instant>,
    /// When its user counts as away from, so warnings no longer wake it.
    pub(super) away_from: Option<Instant>,
    /// A live runtime is bound for it, so its agent works and reads its
    /// monitors' news as it does.
    pub(super) busy: bool,
}

/// When a monitor armed for `lasts`, until `expires`, warns its owner, or
/// `None` for a monitor armed for too short a time to warn. The warning
/// comes `warning_before` its end. If the owner's prompt cache is warm now
/// and expires before then, it comes `cache_margin` before the cache
/// expires instead, unless the user counts as away by then.
pub(super) fn warning_due(
    expires: Instant,
    lasts: Duration,
    limits: &MonitorLimits,
    cache_expires: Option<Instant>,
    away_from: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    if lasts < limits.warned_from {
        return None;
    }
    let normal = expires
        .checked_sub(limits.warning_before)
        .unwrap_or(expires);
    let early = cache_expires
        .filter(|cache| *cache > now)
        .and_then(|cache| cache.checked_sub(limits.cache_margin))
        .filter(|early| *early < normal && away_from.is_none_or(|away| *early < away));
    Some(early.unwrap_or(normal))
}

/// The tokio `Instant` of `at`, a wall-clock time, saturating far in the
/// future.
pub(super) fn instant(at: SystemTime) -> Instant {
    let now = Instant::now();
    match at.duration_since(SystemTime::now()) {
        Ok(ahead) => now
            .checked_add(ahead)
            .unwrap_or_else(|| now + Duration::from_secs(86_400 * 365 * 30)),
        Err(behind) => now.checked_sub(behind.duration()).unwrap_or(now),
    }
}

impl Arming {
    /// When its warning comes, if it hasn't yet.
    pub(super) fn warning(&self, limits: &MonitorLimits, inputs: WakeInputs) -> Option<Instant> {
        if self.warned {
            return None;
        }
        // A busy agent reads the warning only after its call, when the time
        // left before the cache expires says nothing, so it comes at its
        // usual time. The host's next word on the cache plans it again.
        warning_due(
            self.expires,
            self.lasts,
            limits,
            inputs.cache.filter(|_| !inputs.busy),
            inputs.away_from,
            Instant::now(),
        )
    }
}

impl TerminalManager {
    /// Tells the monitors when `owner`'s prompt cache expires, which is its
    /// latest model request (or the latest refresh of its cache) plus the
    /// cache's lifetime. `None` when the host doesn't know. A warning before a
    /// monitor ends that would come after the cache expires then comes
    /// [`cache_margin`](MonitorLimits::cache_margin) before it instead, and
    /// says how long the monitor has left, so a turn it starts for an idle
    /// agent reads the conversation from the cache. While a runtime bound
    /// for the owner is open (its agent works), warnings keep their usual
    /// time instead, since the agent reads them only after its call. A host
    /// calls this as the agent's requests start and as its turns end.
    /// Monitors already running plan their warnings again.
    pub fn set_prompt_cache_expiry(&self, owner: Uuid, expires_at: Option<SystemTime>) {
        {
            let mut rules = self.wake_rules.lock().unwrap();
            match expires_at {
                Some(at) => rules.caches.insert(owner, instant(at)),
                None => rules.caches.remove(&owner),
            };
        }
        self.monitor_timing_changed();
    }

    /// Tells the monitors when the host's user last wrote (sent a message).
    /// Once the user hasn't written for
    /// [`warning_wakes_within`](Self::warning_wakes_within), a monitor's
    /// warning no longer wakes an idle agent ([`subscribe_alerts`](Self::subscribe_alerts)
    /// doesn't report it), and goes with the agent's next turn instead.
    /// Until a host says, the user counts as having written when the
    /// manager was made.
    pub fn set_user_active_at(&self, at: SystemTime) {
        self.wake_rules.lock().unwrap().user_active_at = instant(at);
        self.monitor_timing_changed();
    }

    /// Sets how long after the user last wrote (see
    /// [`set_user_active_at`](Self::set_user_active_at)) a monitor's warning
    /// still wakes an idle agent, 2 hours by default
    /// ([`DEFAULT_WARNINGS_WAKE_WITHIN`]). `None` means always, and zero
    /// never. Without a limit, an idle agent that rearms its monitors at
    /// each warning would be woken for as long as it does.
    pub fn set_warning_wakes_within(&self, within: Option<Duration>) {
        self.wake_rules.lock().unwrap().warnings_wake_within = within;
        self.monitor_timing_changed();
    }

    /// How long after the user last wrote a monitor's warning still wakes
    /// an idle agent, as last set.
    pub fn warning_wakes_within(&self) -> Option<Duration> {
        self.wake_rules.lock().unwrap().warnings_wake_within
    }

    /// What decides `owner`'s warnings now.
    pub(super) fn wake_inputs(&self, owner: Uuid) -> WakeInputs {
        let busy = self
            .runtimes
            .lock()
            .unwrap()
            .get(&owner)
            .and_then(|binding| binding.sender.upgrade())
            .is_some_and(|sender| !sender.is_closed());
        let rules = self.wake_rules.lock().unwrap();
        WakeInputs {
            busy,
            cache: rules.caches.get(&owner).copied(),
            away_from: rules
                .warnings_wake_within
                .and_then(|within| rules.user_active_at.checked_add(within)),
        }
    }

    /// Whether the user counts as away, so warnings wake no one.
    pub(super) fn user_away(&self) -> bool {
        let rules = self.wake_rules.lock().unwrap();
        rules.warnings_wake_within.is_some_and(|within| {
            rules
                .user_active_at
                .checked_add(within)
                .is_some_and(|away| Instant::now() >= away)
        })
    }

    /// Tells the monitors' actors to plan their timed events again.
    pub(super) fn monitor_timing_changed(&self) {
        self.monitor_timing
            .send_modify(|count| *count = count.wrapping_add(1));
    }

    /// What changes when the monitors' timed events should be planned
    /// again.
    pub(super) fn watch_monitor_timing(&self) -> watch::Receiver<u64> {
        self.monitor_timing.subscribe()
    }

    /// Keeps `owner`'s monitor `id` for `lasts` from now, at most
    /// [`longest`](MonitorLimits::longest), and arms its warning again.
    /// Returns its new end. A monitor that ended can't be rearmed
    /// ([`TerminalError::Finished`]).
    pub fn rearm(&self, owner: Uuid, id: Uuid, lasts: Duration) -> Result<SystemTime> {
        let limits = self.monitor_limits();
        if lasts.is_zero() || lasts > limits.longest {
            return Err(TerminalError::Invalid(format!(
                "a monitor's time must be more than zero and at most {}",
                calls::describe(limits.longest)
            )));
        }
        let job = self.owned(owner, id)?;
        let expires_at = {
            let mut data = job.data.lock().unwrap();
            if data.record.monitor.is_none() {
                return Err(TerminalError::Invalid(format!("job {id} isn't a monitor")));
            }
            let running = data.record.state == JobState::Running;
            if data
                .monitor
                .as_ref()
                .is_none_or(|arming| !running || arming.ending)
            {
                return Err(TerminalError::Finished);
            }
            // Saved first, so a rearm that fails changes nothing.
            let expires_at = SystemTime::now() + lasts;
            let mut record = data.record.clone();
            if let Some(monitor) = record.monitor.as_mut() {
                monitor.expires_at = expires_at;
                monitor.warned = false;
            }
            persist(&self.config, &record)?;
            data.record = record;
            if let Some(arming) = data.monitor.as_mut() {
                arming.expires = Instant::now() + lasts;
                arming.lasts = lasts;
                arming.warned = false;
            }
            expires_at
        };
        // A warning that hasn't gone is out of date.
        self.drop_warning(owner, id);
        self.monitor_timing_changed();
        Ok(expires_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The warning comes before the end, none for a short monitor, and
    /// before a warm prompt cache expires when that's earlier, unless the
    /// user counts as away by then.
    #[test]
    fn the_warning_comes_before_the_end_or_before_the_cache_expires() {
        let limits = MonitorLimits::default();
        let now = Instant::now();
        let minutes = |count: u64| Duration::from_secs(60 * count);
        let expires = now + minutes(30);
        let due = |lasts: Duration, cache: Option<Instant>, away: Option<Instant>| {
            warning_due(expires, lasts, &limits, cache, away, now)
        };
        assert_eq!(due(minutes(30), None, None), Some(now + minutes(25)));
        assert_eq!(due(minutes(9), None, None), None, "too short to warn");
        assert_eq!(due(minutes(10), None, None), Some(now + minutes(25)));
        // A 5-minute cache warm now.
        assert_eq!(
            due(minutes(30), Some(now + minutes(5)), None),
            Some(now + minutes(4))
        );
        // A cache that outlasts the normal warning, or one already cold.
        assert_eq!(
            due(minutes(30), Some(now + minutes(60)), None),
            Some(now + minutes(25))
        );
        assert_eq!(
            due(minutes(30), Some(now - minutes(1)), None),
            Some(now + minutes(25))
        );
        // Within the margin it warns at once, while the cache is warm.
        assert_eq!(
            due(minutes(30), Some(now + Duration::from_secs(30)), None),
            Some(now - Duration::from_secs(30))
        );
        // Away by the early time, or before.
        assert_eq!(
            due(minutes(30), Some(now + minutes(5)), Some(now + minutes(3))),
            Some(now + minutes(25))
        );
        assert_eq!(
            due(minutes(30), Some(now + minutes(5)), Some(now + minutes(6))),
            Some(now + minutes(4))
        );
    }

    #[test]
    fn wall_clock_times_become_instants() {
        let now = Instant::now();
        let ahead = instant(SystemTime::now() + Duration::from_secs(300));
        assert!(ahead >= now + Duration::from_secs(299) && ahead <= now + Duration::from_secs(301));
        let behind = instant(SystemTime::now() - Duration::from_secs(300));
        assert!(behind <= now - Duration::from_secs(299));
        // About two thousand years ahead, which every platform's wall clock
        // holds (Windows' ends far sooner than Linux's).
        assert!(instant(SystemTime::UNIX_EPOCH + Duration::from_secs(1 << 36)) > now);
    }
}
