//! Plan budgets: limits on how much of a subscription plan's usage windows
//! a run uses, in percentage points of each window, measured from the
//! window's level when the run first saw it.
//!
//! Plan usage is account-wide: every agent's report
//! ([`AgentEvent::PlanUsageReported`](arko_agents::AgentEvent::PlanUsageReported))
//! updates the same windows, the latest report winning, and use of the same
//! plan outside the run while it runs counts too. The points are known
//! only once a request has completed and its report has arrived, so a plan
//! budget is a soft limit, as a dollar budget is: the requests in flight
//! when it is reached still complete.

use arko_ai::{PlanUsage, PlanUsageWindow};
use serde::{Deserialize, Serialize};

/// How far a window's reset time may move later without it counting as a
/// reset (reset times derived from "seconds until the reset" drift by a
/// few seconds between reports).
const RESET_TOLERANCE_SECS: u64 = 60;

/// Limits on a run's use of its subscription plan's usage windows, in
/// percentage points of each window (`Some(25.0)`: a quarter of the
/// window), counted from each window's level when the run started. `None`:
/// no limit for that window. Set with
/// [`WorkflowBuilder::plan_budget`](crate::WorkflowBuilder::plan_budget).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanBudget {
    /// Points of the five-hour window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub five_hour: Option<f64>,
    /// Points of the seven-day window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seven_day: Option<f64>,
}

impl PlanBudget {
    /// No limits.
    pub fn none() -> Self {
        Self::default()
    }

    /// Sets the five-hour window's limit, in points.
    pub fn five_hour(mut self, points: f64) -> Self {
        self.five_hour = Some(points);
        self
    }

    /// Sets the seven-day window's limit, in points.
    pub fn seven_day(mut self, points: f64) -> Self {
        self.seven_day = Some(points);
        self
    }

    /// The limit for `window`.
    pub fn limit(&self, window: PlanWindow) -> Option<f64> {
        match window {
            PlanWindow::FiveHour => self.five_hour,
            PlanWindow::SevenDay => self.seven_day,
        }
    }

    pub(crate) fn set(&mut self, window: PlanWindow, limit: Option<f64>) {
        match window {
            PlanWindow::FiveHour => self.five_hour = limit,
            PlanWindow::SevenDay => self.seven_day = limit,
        }
    }

    /// Whether any limit is set.
    pub fn is_set(&self) -> bool {
        self.five_hour.is_some() || self.seven_day.is_some()
    }
}

/// A plan usage window a [`PlanBudget`] limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanWindow {
    FiveHour,
    SevenDay,
}

impl PlanWindow {
    pub const ALL: [PlanWindow; 2] = [PlanWindow::FiveHour, PlanWindow::SevenDay];

    /// `"five_hour"` or `"seven_day"`.
    pub fn key(self) -> &'static str {
        match self {
            PlanWindow::FiveHour => "five_hour",
            PlanWindow::SevenDay => "seven_day",
        }
    }

    /// `"five-hour"` or `"seven-day"`.
    pub fn label(self) -> &'static str {
        match self {
            PlanWindow::FiveHour => "five-hour",
            PlanWindow::SevenDay => "seven-day",
        }
    }

    /// The window a reported window is: by its length when reported (300
    /// minutes: five-hour; 10080: seven-day; another length is neither),
    /// otherwise by position (primary: five-hour, secondary: seven-day).
    fn of(window: &PlanUsageWindow, primary: bool) -> Option<Self> {
        match window.window_minutes {
            Some(300) => Some(PlanWindow::FiveHour),
            Some(10_080) => Some(PlanWindow::SevenDay),
            Some(_) => None,
            None if primary => Some(PlanWindow::FiveHour),
            None => Some(PlanWindow::SevenDay),
        }
    }
}

/// The points of each window a run has used, as
/// [`Workflow::plan_usage_spent`](crate::Workflow::plan_usage_spent)
/// reports them; `None` for a window no report has shown yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanUsageSpent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub five_hour: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seven_day: Option<f64>,
}

impl PlanUsageSpent {
    /// The points used of `window`.
    pub fn get(&self, window: PlanWindow) -> Option<f64> {
        match window {
            PlanWindow::FiveHour => self.five_hour,
            PlanWindow::SevenDay => self.seven_day,
        }
    }
}

/// How much of one window a run has used.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanWindowUsage {
    /// The window's level (percent used) at the start of the current
    /// period: the first report the run saw, or 0 after a reset.
    pub baseline: f64,
    /// The latest level reported.
    pub latest: f64,
    /// Points used in earlier periods of the window (before its resets).
    #[serde(default)]
    pub accumulated: f64,
    /// When the current period resets, in Unix seconds, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resets_at: Option<u64>,
}

impl PlanWindowUsage {
    /// The points used: earlier periods and the current one.
    pub fn used(&self) -> f64 {
        (self.accumulated + self.latest - self.baseline).max(0.0)
    }

    fn start(level: f64, resets_at: Option<u64>) -> Self {
        Self {
            baseline: level,
            latest: level,
            accumulated: 0.0,
            resets_at,
        }
    }

    /// Takes a report of the window. A reset time moved later means the
    /// window reset: the points used before are kept and the new period
    /// counts from 0. Without reset times to compare, a lower level means
    /// the same. A lower level with the same reset time is an earlier
    /// report arriving late, and is ignored.
    fn observe(&mut self, level: f64, resets_at: Option<u64>) {
        let reset = match (self.resets_at, resets_at) {
            (Some(old), Some(new)) => {
                if new > old.saturating_add(RESET_TOLERANCE_SECS) {
                    true
                } else if new.saturating_add(RESET_TOLERANCE_SECS) < old || level < self.latest {
                    // From an earlier period, or late.
                    return;
                } else {
                    false
                }
            }
            _ => level < self.latest,
        };
        if reset {
            self.accumulated = self.used();
            self.baseline = 0.0;
        }
        self.latest = level;
        if resets_at.is_some() {
            self.resets_at = resets_at;
        }
    }
}

/// How much of each plan window a run has used, saved in checkpoints so a
/// resumed run keeps counting from where the run started.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanUsageTracking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub five_hour: Option<PlanWindowUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seven_day: Option<PlanWindowUsage>,
}

impl PlanUsageTracking {
    fn window_mut(&mut self, window: PlanWindow) -> &mut Option<PlanWindowUsage> {
        match window {
            PlanWindow::FiveHour => &mut self.five_hour,
            PlanWindow::SevenDay => &mut self.seven_day,
        }
    }

    /// The tracking of `window`, if a report showed it.
    pub fn window(&self, window: PlanWindow) -> Option<&PlanWindowUsage> {
        match window {
            PlanWindow::FiveHour => self.five_hour.as_ref(),
            PlanWindow::SevenDay => self.seven_day.as_ref(),
        }
    }

    /// Takes a plan usage report, at Unix time `now` (for windows reported
    /// with the seconds until their reset only).
    pub fn observe(&mut self, usage: &PlanUsage, now: u64) {
        for (reported, primary) in [(&usage.primary, true), (&usage.secondary, false)] {
            let Some(reported) = reported else {
                continue;
            };
            if !reported.used_percent.is_finite() {
                continue;
            }
            let Some(window) = PlanWindow::of(reported, primary) else {
                continue;
            };
            let resets_at = reported.resets_at.or_else(|| {
                reported
                    .reset_after_seconds
                    .map(|after| now.saturating_add(after))
            });
            let level = reported.used_percent.max(0.0);
            match self.window_mut(window) {
                Some(tracked) => tracked.observe(level, resets_at),
                slot @ None => *slot = Some(PlanWindowUsage::start(level, resets_at)),
            }
        }
    }

    /// The points used of each window.
    pub fn spent(&self) -> PlanUsageSpent {
        PlanUsageSpent {
            five_hour: self.five_hour.as_ref().map(PlanWindowUsage::used),
            seven_day: self.seven_day.as_ref().map(PlanWindowUsage::used),
        }
    }
}

/// A plan budget's limits and how much of each window the run has used,
/// as a checkpoint records them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanBudgetState {
    #[serde(default)]
    pub limits: PlanBudget,
    #[serde(default)]
    pub usage: PlanUsageTracking,
}

/// A window whose limit a report took the run over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanBudgetCrossing {
    pub window: PlanWindow,
    /// Points used.
    pub used: f64,
    /// The window's limit, in points.
    pub limit: f64,
}

impl PlanBudgetCrossing {
    /// "five-hour window 25.4 of 25 points".
    pub fn describe(&self) -> String {
        format!(
            "{} window {:.1} of {} points",
            self.window.label(),
            self.used,
            self.limit
        )
    }
}

/// The plan budget's state in a [`BudgetManager`](super::BudgetManager).
#[derive(Debug, Default)]
pub(crate) struct PlanBudgetTracker {
    pub(crate) limits: PlanBudget,
    pub(crate) usage: PlanUsageTracking,
    /// Windows reported over their limit since last within it.
    reported: [bool; 2],
}

fn index(window: PlanWindow) -> usize {
    match window {
        PlanWindow::FiveHour => 0,
        PlanWindow::SevenDay => 1,
    }
}

impl PlanBudgetTracker {
    /// The first window over its limit.
    pub(crate) fn exceeded(&self) -> Option<PlanBudgetCrossing> {
        PlanWindow::ALL.into_iter().find_map(|window| {
            let limit = self.limits.limit(window)?;
            let used = self.usage.window(window)?.used();
            (used > limit).then_some(PlanBudgetCrossing {
                window,
                used,
                limit,
            })
        })
    }

    fn over(&self) -> Vec<PlanBudgetCrossing> {
        PlanWindow::ALL
            .into_iter()
            .filter_map(|window| {
                let limit = self.limits.limit(window)?;
                let used = self.usage.window(window)?.used();
                (used > limit).then_some(PlanBudgetCrossing {
                    window,
                    used,
                    limit,
                })
            })
            .collect()
    }

    /// Takes a report; returns the windows it took over their limits (each
    /// once per crossing).
    pub(crate) fn observe(&mut self, usage: &PlanUsage, now: u64) -> Vec<PlanBudgetCrossing> {
        self.usage.observe(usage, now);
        let mut crossed = self.over();
        crossed.retain(|crossing| {
            !std::mem::replace(&mut self.reported[index(crossing.window)], true)
        });
        crossed
    }

    /// Sets the limits; a window now within its limit reports its next
    /// crossing again.
    pub(crate) fn set_limits(&mut self, limits: PlanBudget) {
        self.limits = limits;
        let over = self.over();
        for window in PlanWindow::ALL {
            if !over.iter().any(|crossing| crossing.window == window) {
                self.reported[index(window)] = false;
            }
        }
    }

    /// Restores the tracking a checkpoint recorded; windows already over
    /// their limits count as reported.
    pub(crate) fn restore(&mut self, usage: PlanUsageTracking) {
        self.usage = usage;
        self.reported = [false; 2];
        for crossing in self.over() {
            self.reported[index(crossing.window)] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(used: f64, minutes: Option<u64>, resets_at: Option<u64>) -> PlanUsageWindow {
        PlanUsageWindow {
            used_percent: used,
            window_minutes: minutes,
            resets_at,
            reset_after_seconds: None,
        }
    }

    fn report(primary: Option<PlanUsageWindow>, secondary: Option<PlanUsageWindow>) -> PlanUsage {
        PlanUsage {
            primary,
            secondary,
            ..PlanUsage::default()
        }
    }

    fn approx(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("a window");
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    #[test]
    fn points_count_from_the_first_report_in_both_windows() {
        let mut tracking = PlanUsageTracking::default();
        tracking.observe(
            &report(
                Some(window(40.0, Some(300), Some(10_000))),
                Some(window(10.0, Some(10_080), Some(500_000))),
            ),
            0,
        );
        assert_eq!(tracking.spent().five_hour, Some(0.0));
        tracking.observe(
            &report(
                Some(window(52.5, Some(300), Some(10_000))),
                Some(window(13.0, Some(10_080), Some(500_000))),
            ),
            0,
        );
        approx(tracking.spent().five_hour, 12.5);
        approx(tracking.spent().seven_day, 3.0);
        // A report of one window leaves the other as it was.
        tracking.observe(
            &report(Some(window(55.0, Some(300), Some(10_000))), None),
            0,
        );
        approx(tracking.spent().five_hour, 15.0);
        approx(tracking.spent().seven_day, 3.0);
        // A late report of an earlier level is ignored.
        tracking.observe(
            &report(Some(window(50.0, Some(300), Some(10_000))), None),
            0,
        );
        approx(tracking.spent().five_hour, 15.0);
    }

    #[test]
    fn a_reset_keeps_the_points_used_before_it() {
        let mut tracking = PlanUsageTracking::default();
        let five = |used, resets| report(Some(window(used, Some(300), resets)), None);
        tracking.observe(&five(70.0, Some(18_000)), 0);
        tracking.observe(&five(90.0, Some(18_000)), 0);
        // Jitter of the reset time is no reset.
        tracking.observe(&five(91.0, Some(18_010)), 0);
        approx(tracking.spent().five_hour, 21.0);
        // The window reset: the new period counts from 0.
        tracking.observe(&five(4.0, Some(36_000)), 0);
        approx(tracking.spent().five_hour, 25.0);
        tracking.observe(&five(10.0, Some(36_000)), 0);
        approx(tracking.spent().five_hour, 31.0);

        // Without reset times, a lower level is a reset.
        let mut tracking = PlanUsageTracking::default();
        tracking.observe(&five(30.0, None), 0);
        tracking.observe(&five(35.0, None), 0);
        tracking.observe(&five(2.0, None), 0);
        approx(tracking.spent().five_hour, 7.0);
    }

    #[test]
    fn windows_are_told_apart_by_their_length() {
        let mut tracking = PlanUsageTracking::default();
        // ChatGPT may report the seven-day window as its primary one.
        tracking.observe(
            &report(
                Some(window(20.0, Some(10_080), None)),
                Some(window(5.0, Some(300), None)),
            ),
            0,
        );
        assert_eq!(tracking.window(PlanWindow::SevenDay).unwrap().latest, 20.0);
        assert_eq!(tracking.window(PlanWindow::FiveHour).unwrap().latest, 5.0);
        // Without lengths: primary five-hour, secondary seven-day; another
        // length is neither.
        let mut tracking = PlanUsageTracking::default();
        tracking.observe(
            &report(
                Some(window(1.0, None, None)),
                Some(window(2.0, Some(60), None)),
            ),
            0,
        );
        assert_eq!(tracking.window(PlanWindow::FiveHour).unwrap().latest, 1.0);
        assert!(tracking.window(PlanWindow::SevenDay).is_none());
        // A reset time given as seconds from now.
        let mut tracking = PlanUsageTracking::default();
        let after = |used, after| {
            let mut window = window(used, None, None);
            window.reset_after_seconds = Some(after);
            report(Some(window), None)
        };
        tracking.observe(&after(10.0, 100), 1_000);
        tracking.observe(&after(12.0, 95), 1_007);
        approx(tracking.spent().five_hour, 2.0);
    }

    #[test]
    fn crossings_are_reported_once_and_restored_tracking_keeps_its_baseline() {
        let mut tracker = PlanBudgetTracker::default();
        tracker.set_limits(PlanBudget::none().five_hour(10.0));
        let five = |used| report(Some(window(used, Some(300), Some(18_000))), None);
        assert!(tracker.observe(&five(50.0), 0).is_empty());
        assert!(tracker.observe(&five(60.0), 0).is_empty(), "at, not over");
        let crossed = tracker.observe(&five(60.5), 0);
        assert_eq!(crossed.len(), 1);
        assert_eq!(crossed[0].describe(), "five-hour window 10.5 of 10 points");
        assert!(tracker.observe(&five(61.0), 0).is_empty(), "once");
        tracker.set_limits(PlanBudget::none().five_hour(20.0));
        assert!(tracker.exceeded().is_none());
        assert_eq!(tracker.observe(&five(71.0), 0).len(), 1, "a new crossing");

        // A checkpoint round trip keeps the baseline: a resumed run counts
        // from where the run started, not from the level at the resume.
        let state = PlanBudgetState {
            limits: tracker.limits,
            usage: tracker.usage.clone(),
        };
        let json = serde_json::to_string(&state).unwrap();
        let state: PlanBudgetState = serde_json::from_str(&json).unwrap();
        let mut resumed = PlanBudgetTracker::default();
        resumed.set_limits(PlanBudget::none().five_hour(30.0));
        resumed.restore(state.usage);
        approx(resumed.usage.spent().five_hour, 21.0);
        assert!(resumed.observe(&five(75.0), 0).is_empty());
        approx(resumed.usage.spent().five_hour, 25.0);
        assert_eq!(resumed.observe(&five(81.0), 0).len(), 1);
        let empty: PlanBudgetState = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, PlanBudgetState::default());
    }
}
