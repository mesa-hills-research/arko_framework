//! Plan budgets: limits on how much of subscription plans' usage windows a
//! run uses, in percentage points of each window, measured from the
//! window's level when the run first saw it.
//!
//! Each plan ([`SubscriptionPlan`]) is counted on its own. Plan usage is
//! account-wide: every agent's report of a plan
//! ([`AgentEvent::PlanUsageReported`](arko_agents::AgentEvent::PlanUsageReported))
//! updates the same windows, the latest report winning, and use of the same
//! plan outside the run while it runs counts too. The points are known
//! only once a request has completed and its report has arrived, so a plan
//! budget is a soft limit, as a dollar budget is: the requests in flight
//! when it is reached still complete.

use std::collections::{BTreeMap, HashSet};

use arko_ai::{PlanUsage, PlanUsageWindow, SubscriptionPlan};
use serde::{Deserialize, Serialize};

/// How far a window's reset time may move later without it counting as a
/// reset (reset times derived from "seconds until the reset" drift by a
/// few seconds between reports).
const RESET_TOLERANCE_SECS: u64 = 60;

/// Limits on a run's use of its subscription plans' usage windows, in
/// percentage points of each window (`Some(25.0)`: a quarter of the
/// window), counted from each window's level when the run started. `None`:
/// no limit for that window. Set with
/// [`WorkflowBuilder::plan_budget`](crate::WorkflowBuilder::plan_budget).
///
/// Each plan ([`SubscriptionPlan`]) has its own windows, counted
/// separately. `five_hour` and `seven_day` limit each plan's window of that
/// length. A plan's own limit (`chatgpt`, `claude_code`) replaces them for
/// that plan's window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanBudget {
    /// Points of each plan's five-hour window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub five_hour: Option<f64>,
    /// Points of each plan's seven-day window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seven_day: Option<f64>,
    /// The ChatGPT plan's own limits.
    #[serde(default, skip_serializing_if = "PlanLimits::is_empty")]
    pub chatgpt: PlanLimits,
    /// The Claude Code plan's own limits.
    #[serde(default, skip_serializing_if = "PlanLimits::is_empty")]
    pub claude_code: PlanLimits,
}

impl PlanBudget {
    /// No limits.
    pub fn none() -> Self {
        Self::default()
    }

    /// Sets each plan's five-hour window's limit, in points.
    pub fn five_hour(mut self, points: f64) -> Self {
        self.five_hour = Some(points);
        self
    }

    /// Sets each plan's seven-day window's limit, in points.
    pub fn seven_day(mut self, points: f64) -> Self {
        self.seven_day = Some(points);
        self
    }

    /// Sets the ChatGPT plan's own limits.
    pub fn chatgpt(mut self, limits: PlanLimits) -> Self {
        self.chatgpt = limits;
        self
    }

    /// Sets the Claude Code plan's own limits.
    pub fn claude_code(mut self, limits: PlanLimits) -> Self {
        self.claude_code = limits;
        self
    }

    /// The limit for each plan's `window` (`five_hour` or `seven_day`).
    pub fn limit(&self, window: PlanWindow) -> Option<f64> {
        match window {
            PlanWindow::FiveHour => self.five_hour,
            PlanWindow::SevenDay => self.seven_day,
        }
    }

    /// The limit that applies to `plan`'s `window`: the plan's own, else
    /// [`limit`](Self::limit).
    pub fn limit_for(&self, plan: SubscriptionPlan, window: PlanWindow) -> Option<f64> {
        self.own(Some(plan), window).or(self.limit(window))
    }

    /// The limit that applies to the tracking of `plan`'s `window`. Tracking
    /// without a plan (`None`) has the general limits.
    fn applies(&self, plan: Option<SubscriptionPlan>, window: PlanWindow) -> Option<f64> {
        match plan {
            Some(plan) => self.limit_for(plan, window),
            None => self.limit(window),
        }
    }

    /// Where each limit is set: the general ones (`None`), then each plan's
    /// own.
    pub(crate) fn slots() -> impl Iterator<Item = (Option<SubscriptionPlan>, PlanWindow)> {
        std::iter::once(None)
            .chain(SubscriptionPlan::ALL.map(Some))
            .flat_map(|plan| PlanWindow::ALL.map(move |window| (plan, window)))
    }

    /// The limit set for `window` of `plan` (`None`: the general one).
    pub(crate) fn own(&self, plan: Option<SubscriptionPlan>, window: PlanWindow) -> Option<f64> {
        match plan {
            None => self.limit(window),
            Some(SubscriptionPlan::ChatGpt) => self.chatgpt.limit(window),
            Some(SubscriptionPlan::ClaudeCode) => self.claude_code.limit(window),
        }
    }

    /// Sets the limit for `window` of `plan` (`None`: the general one).
    pub(crate) fn set(
        &mut self,
        plan: Option<SubscriptionPlan>,
        window: PlanWindow,
        limit: Option<f64>,
    ) {
        let slot = match plan {
            None => match window {
                PlanWindow::FiveHour => &mut self.five_hour,
                PlanWindow::SevenDay => &mut self.seven_day,
            },
            Some(SubscriptionPlan::ChatGpt) => self.chatgpt.slot(window),
            Some(SubscriptionPlan::ClaudeCode) => self.claude_code.slot(window),
        };
        *slot = limit;
    }

    /// Whether any limit is set.
    pub fn is_set(&self) -> bool {
        Self::slots().any(|(plan, window)| self.own(plan, window).is_some())
    }

    /// Checks that every limit is finite and non-negative. The error names
    /// the first that isn't.
    pub(crate) fn validate(&self) -> Result<(), String> {
        for (plan, window) in Self::slots() {
            if let Some(limit) = self.own(plan, window)
                && (!limit.is_finite() || limit < 0.0)
            {
                let plan = plan.map_or(String::new(), |plan| format!("{} ", plan.label()));
                return Err(format!(
                    "{plan}{} plan budget must be finite and non-negative",
                    window.label()
                ));
            }
        }
        Ok(())
    }
}

/// One plan's own limits in a [`PlanBudget`], in points of each of its
/// windows. `None`: the budget's limit for that window applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanLimits {
    /// Points of the plan's five-hour window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub five_hour: Option<f64>,
    /// Points of the plan's seven-day window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seven_day: Option<f64>,
}

impl PlanLimits {
    /// No limits of its own.
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

    fn slot(&mut self, window: PlanWindow) -> &mut Option<f64> {
        match window {
            PlanWindow::FiveHour => &mut self.five_hour,
            PlanWindow::SevenDay => &mut self.seven_day,
        }
    }

    /// Whether no limit is set.
    pub fn is_empty(&self) -> bool {
        self.five_hour.is_none() && self.seven_day.is_none()
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
    /// minutes: five-hour, 10080: seven-day, another length is neither),
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
/// [`Workflow::plan_usage_spent`](crate::Workflow::plan_usage_spent) (the
/// most any plan has used) and
/// [`Workflow::plan_usage_spent_of`](crate::Workflow::plan_usage_spent_of)
/// (one plan's) report them. `None` for a window no report has shown yet.
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

/// How much of one plan's windows a run has used, saved in checkpoints so a
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
    /// with the seconds until their reset only). The report is taken as
    /// this tracking's plan's, whatever plan it names.
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

/// A plan budget's limits and how much of each plan's windows the run has
/// used, as a checkpoint records them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanBudgetState {
    #[serde(default)]
    pub limits: PlanBudget,
    /// Tracking without a plan: what a checkpoint from before plans were
    /// counted apart recorded (the first plan to report after the resume
    /// takes it over), and reports that name no plan.
    #[serde(default)]
    pub usage: PlanUsageTracking,
    /// Each plan's tracking.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub plans: BTreeMap<SubscriptionPlan, PlanUsageTracking>,
}

/// A window whose limit a report took the run over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanBudgetCrossing {
    /// The plan whose window it is. `None` for tracking without a plan
    /// (see [`PlanBudgetState::usage`]).
    pub plan: Option<SubscriptionPlan>,
    pub window: PlanWindow,
    /// Points used.
    pub used: f64,
    /// The window's limit, in points.
    pub limit: f64,
}

impl PlanBudgetCrossing {
    /// "Claude Code five-hour window 25.4 of 25 points" (without a plan:
    /// "five-hour window 25.4 of 25 points").
    pub fn describe(&self) -> String {
        format!(
            "{}{} window {:.1} of {} points",
            self.plan
                .map_or(String::new(), |plan| format!("{} ", plan.label())),
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
    /// Each plan's tracking. `None`: tracking without a plan (see
    /// [`PlanBudgetState::usage`]).
    usage: BTreeMap<Option<SubscriptionPlan>, PlanUsageTracking>,
    /// Windows reported over their limit since last within it.
    reported: HashSet<(Option<SubscriptionPlan>, PlanWindow)>,
}

impl PlanBudgetTracker {
    /// The first window over its limit.
    pub(crate) fn exceeded(&self) -> Option<PlanBudgetCrossing> {
        self.over().into_iter().next()
    }

    fn over(&self) -> Vec<PlanBudgetCrossing> {
        self.usage
            .iter()
            .flat_map(|(&plan, tracking)| {
                PlanWindow::ALL.into_iter().filter_map(move |window| {
                    let limit = self.limits.applies(plan, window)?;
                    let used = tracking.window(window)?.used();
                    (used > limit).then_some(PlanBudgetCrossing {
                        plan,
                        window,
                        used,
                        limit,
                    })
                })
            })
            .collect()
    }

    /// Takes a report, counted with its plan's windows. Returns the windows
    /// it took over their limits (each once per crossing). The first plan to
    /// report takes over the tracking without a plan, if it has none of its
    /// own.
    pub(crate) fn observe(&mut self, usage: &PlanUsage, now: u64) -> Vec<PlanBudgetCrossing> {
        let plan = usage.plan;
        if plan.is_some()
            && !self.usage.contains_key(&plan)
            && let Some(unnamed) = self.usage.remove(&None)
        {
            self.usage.insert(plan, unnamed);
            for window in PlanWindow::ALL {
                if self.reported.remove(&(None, window)) {
                    self.reported.insert((plan, window));
                }
            }
            self.forget_within();
        }
        self.usage.entry(plan).or_default().observe(usage, now);
        let mut crossed = self.over();
        crossed.retain(|crossing| self.reported.insert((crossing.plan, crossing.window)));
        crossed
    }

    /// Windows now within their limits report their next crossing again.
    fn forget_within(&mut self) {
        let over: HashSet<_> = self
            .over()
            .into_iter()
            .map(|crossing| (crossing.plan, crossing.window))
            .collect();
        self.reported.retain(|window| over.contains(window));
    }

    /// Sets the limits. A window now within its limit reports its next
    /// crossing again.
    pub(crate) fn set_limits(&mut self, limits: PlanBudget) {
        self.limits = limits;
        self.forget_within();
    }

    /// The points used of each window: the most any plan has used.
    pub(crate) fn spent(&self) -> PlanUsageSpent {
        let most = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.usage.values().map(PlanUsageTracking::spent).fold(
            PlanUsageSpent::default(),
            |all, plan| PlanUsageSpent {
                five_hour: most(all.five_hour, plan.five_hour),
                seven_day: most(all.seven_day, plan.seven_day),
            },
        )
    }

    /// The points used of each of `plan`'s windows.
    pub(crate) fn spent_of(&self, plan: SubscriptionPlan) -> PlanUsageSpent {
        self.usage
            .get(&Some(plan))
            .map(PlanUsageTracking::spent)
            .unwrap_or_default()
    }

    /// The limits and the tracking, for a checkpoint.
    pub(crate) fn state(&self) -> PlanBudgetState {
        PlanBudgetState {
            limits: self.limits,
            usage: self.usage.get(&None).cloned().unwrap_or_default(),
            plans: self
                .usage
                .iter()
                .filter_map(|(plan, tracking)| Some(((*plan)?, tracking.clone())))
                .collect(),
        }
    }

    /// Restores the tracking a checkpoint recorded (`unnamed`: without a
    /// plan). Windows already over their limits count as reported.
    pub(crate) fn restore(
        &mut self,
        unnamed: PlanUsageTracking,
        plans: BTreeMap<SubscriptionPlan, PlanUsageTracking>,
    ) {
        self.usage = plans
            .into_iter()
            .map(|(plan, tracking)| (Some(plan), tracking))
            .collect();
        if unnamed != PlanUsageTracking::default() {
            self.usage.insert(None, unnamed);
        }
        self.reported = self
            .over()
            .into_iter()
            .map(|crossing| (crossing.plan, crossing.window))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATGPT: Option<SubscriptionPlan> = Some(SubscriptionPlan::ChatGpt);
    const CLAUDE_CODE: Option<SubscriptionPlan> = Some(SubscriptionPlan::ClaudeCode);

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

    /// A report of `plan`'s windows: five-hour at `five` (resetting at
    /// `resets_at`), seven-day at `seven`.
    fn of(plan: Option<SubscriptionPlan>, five: f64, resets_at: u64, seven: f64) -> PlanUsage {
        PlanUsage {
            plan,
            ..report(
                Some(window(five, Some(300), Some(resets_at))),
                Some(window(seven, Some(10_080), Some(9_000_000))),
            )
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
        let five = |used| PlanUsage {
            plan: CHATGPT,
            ..report(Some(window(used, Some(300), Some(18_000))), None)
        };
        assert!(tracker.observe(&five(50.0), 0).is_empty());
        assert!(tracker.observe(&five(60.0), 0).is_empty(), "at, not over");
        let crossed = tracker.observe(&five(60.5), 0);
        assert_eq!(crossed.len(), 1);
        assert_eq!(
            crossed[0].describe(),
            "ChatGPT five-hour window 10.5 of 10 points"
        );
        assert!(tracker.observe(&five(61.0), 0).is_empty(), "once");
        tracker.set_limits(PlanBudget::none().five_hour(20.0));
        assert!(tracker.exceeded().is_none());
        assert_eq!(tracker.observe(&five(71.0), 0).len(), 1, "a new crossing");

        // A checkpoint round trip keeps the baseline: a resumed run counts
        // from where the run started, not from the level at the resume.
        let json = serde_json::to_string(&tracker.state()).unwrap();
        let state: PlanBudgetState = serde_json::from_str(&json).unwrap();
        let mut resumed = PlanBudgetTracker::default();
        resumed.set_limits(PlanBudget::none().five_hour(30.0));
        resumed.restore(state.usage, state.plans);
        approx(resumed.spent().five_hour, 21.0);
        assert!(resumed.observe(&five(75.0), 0).is_empty());
        approx(resumed.spent_of(SubscriptionPlan::ChatGpt).five_hour, 25.0);
        assert_eq!(resumed.observe(&five(81.0), 0).len(), 1);
        let empty: PlanBudgetState = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, PlanBudgetState::default());
    }

    /// Two plans report interleaved, with their own levels and reset times:
    /// neither moves the other's counting, and the general limits apply to
    /// each plan's windows separately.
    #[test]
    fn plans_are_counted_apart_and_the_general_limits_apply_to_each() {
        let mut tracker = PlanBudgetTracker::default();
        tracker.set_limits(PlanBudget::none().five_hour(10.0).seven_day(5.0));
        let reports = [
            of(CHATGPT, 40.0, 18_000, 10.0),
            of(CLAUDE_CODE, 5.0, 12_000, 60.0),
            of(CHATGPT, 44.0, 18_000, 11.0),
            of(CLAUDE_CODE, 9.0, 12_000, 61.0),
            of(CHATGPT, 47.0, 18_010, 12.0),
            of(CLAUDE_CODE, 12.0, 12_005, 62.0),
        ];
        for report in &reports {
            assert!(tracker.observe(report, 0).is_empty(), "{report:?}");
        }
        let chatgpt = tracker.spent_of(SubscriptionPlan::ChatGpt);
        approx(chatgpt.five_hour, 7.0);
        approx(chatgpt.seven_day, 2.0);
        let claude_code = tracker.spent_of(SubscriptionPlan::ClaudeCode);
        approx(claude_code.five_hour, 7.0);
        approx(claude_code.seven_day, 2.0);
        // Without a plan's report, nothing of it.
        assert_eq!(
            PlanBudgetTracker::default().spent_of(SubscriptionPlan::ChatGpt),
            PlanUsageSpent::default()
        );

        // Claude Code's window resets; ChatGPT's goes on.
        assert!(
            tracker
                .observe(&of(CLAUDE_CODE, 0.0, 30_000, 62.0), 0)
                .is_empty()
        );
        assert!(
            tracker
                .observe(&of(CHATGPT, 48.0, 18_000, 12.0), 0)
                .is_empty()
        );
        approx(
            tracker.spent_of(SubscriptionPlan::ClaudeCode).five_hour,
            7.0,
        );
        approx(tracker.spent_of(SubscriptionPlan::ChatGpt).five_hour, 8.0);
        // The most any plan used of each window.
        approx(tracker.spent().five_hour, 8.0);

        // Each plan crosses the five-hour limit on its own points.
        let crossed = tracker.observe(&of(CLAUDE_CODE, 3.5, 30_000, 62.0), 0);
        assert_eq!(crossed.len(), 1);
        assert_eq!(
            crossed[0].describe(),
            "Claude Code five-hour window 10.5 of 10 points"
        );
        assert_eq!(tracker.exceeded().unwrap().plan, CLAUDE_CODE);
        let crossed = tracker.observe(&of(CHATGPT, 52.5, 18_000, 12.0), 0);
        assert_eq!((crossed.len(), crossed[0].plan), (1, CHATGPT));
    }

    /// A plan's own limit replaces the general one for that plan, and only
    /// that plan's reports take the run over it.
    #[test]
    fn a_plans_own_limit_is_crossed_by_its_own_reports_only() {
        let mut tracker = PlanBudgetTracker::default();
        tracker.set_limits(
            PlanBudget::none()
                .seven_day(5.0)
                .chatgpt(PlanLimits::none().seven_day(25.0))
                .claude_code(PlanLimits::none().five_hour(85.0)),
        );
        let limits = tracker.limits;
        assert_eq!(
            limits.limit_for(SubscriptionPlan::ChatGpt, PlanWindow::SevenDay),
            Some(25.0)
        );
        assert_eq!(
            limits.limit_for(SubscriptionPlan::ClaudeCode, PlanWindow::SevenDay),
            Some(5.0)
        );
        assert_eq!(
            limits.limit_for(SubscriptionPlan::ChatGpt, PlanWindow::FiveHour),
            None
        );
        assert!(
            tracker
                .observe(&of(CHATGPT, 10.0, 18_000, 10.0), 0)
                .is_empty()
        );
        assert!(
            tracker
                .observe(&of(CLAUDE_CODE, 0.0, 12_000, 1.0), 0)
                .is_empty()
        );
        // ChatGPT: 90 five-hour points (no limit), 20 seven-day (of 25).
        assert!(
            tracker
                .observe(&of(CHATGPT, 100.0, 18_000, 30.0), 0)
                .is_empty()
        );
        assert!(tracker.exceeded().is_none());
        // Claude Code: over its own five-hour limit, then over the general
        // seven-day one.
        let crossed = tracker.observe(&of(CLAUDE_CODE, 86.0, 12_000, 6.0), 0);
        assert_eq!(crossed.len(), 1);
        assert_eq!(
            crossed[0].describe(),
            "Claude Code five-hour window 86.0 of 85 points"
        );
        let crossed = tracker.observe(&of(CLAUDE_CODE, 86.0, 12_000, 6.5), 0);
        assert_eq!(
            crossed[0].describe(),
            "Claude Code seven-day window 5.5 of 5 points"
        );
        // Raising Claude Code's limits leaves ChatGPT's.
        tracker.set_limits(limits.claude_code(PlanLimits::none().five_hour(90.0).seven_day(10.0)));
        assert!(tracker.exceeded().is_none());

        // The JSON: the general limits as before, each plan's own apart.
        let json = serde_json::to_value(tracker.limits).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "seven_day": 5.0,
                "chatgpt": {"seven_day": 25.0},
                "claude_code": {"five_hour": 90.0, "seven_day": 10.0},
            })
        );
        let limits: PlanBudget = serde_json::from_value(json).unwrap();
        assert_eq!(limits, tracker.limits);
        assert_eq!(
            PlanBudget::none()
                .claude_code(PlanLimits::none().five_hour(f64::NAN))
                .validate(),
            Err("Claude Code five-hour plan budget must be finite and non-negative".into())
        );
    }

    /// A checkpoint from before plans were counted apart: its tracking and
    /// limits load, the counting goes on, and the first plan to report
    /// takes the tracking over. Another plan counts from its own first
    /// report.
    #[test]
    fn an_earlier_checkpoints_tracking_is_taken_over_by_the_first_plan_to_report() {
        let old = r#"{
            "limits": {"five_hour": 10.0},
            "usage": {
                "five_hour": {"baseline": 40.0, "latest": 48.0, "accumulated": 0.0, "resets_at": 18000},
                "seven_day": {"baseline": 10.0, "latest": 12.0}
            }
        }"#;
        let state: PlanBudgetState = serde_json::from_str(old).unwrap();
        assert_eq!(state.limits, PlanBudget::none().five_hour(10.0));
        assert!(state.plans.is_empty());
        let mut tracker = PlanBudgetTracker::default();
        tracker.set_limits(state.limits);
        tracker.restore(state.usage, state.plans);
        approx(tracker.spent().five_hour, 8.0);
        // Saved again as it was.
        assert_eq!(
            serde_json::to_value(tracker.state()).unwrap(),
            serde_json::to_value(serde_json::from_str::<PlanBudgetState>(old).unwrap()).unwrap()
        );

        let crossed = tracker.observe(&of(CHATGPT, 51.0, 18_000, 12.5), 0);
        assert_eq!(
            crossed[0].describe(),
            "ChatGPT five-hour window 11.0 of 10 points"
        );
        let chatgpt = tracker.spent_of(SubscriptionPlan::ChatGpt);
        approx(chatgpt.five_hour, 11.0);
        approx(chatgpt.seven_day, 2.5);
        assert_eq!(tracker.state().usage, PlanUsageTracking::default());
        assert!(
            tracker
                .observe(&of(CLAUDE_CODE, 30.0, 12_000, 70.0), 0)
                .is_empty()
        );
        approx(
            tracker.spent_of(SubscriptionPlan::ClaudeCode).five_hour,
            0.0,
        );

        // Tracking over its limit at the resume was reported before it: the
        // plan taking it over doesn't report it again, and reports its next
        // crossing once raised.
        let mut state: PlanBudgetState = serde_json::from_str(old).unwrap();
        state.usage.five_hour.as_mut().unwrap().latest = 55.0;
        let mut tracker = PlanBudgetTracker::default();
        tracker.set_limits(state.limits);
        tracker.restore(state.usage, state.plans);
        assert_eq!(tracker.exceeded().unwrap().plan, None);
        assert!(
            tracker
                .observe(&of(CLAUDE_CODE, 56.0, 18_000, 12.0), 0)
                .is_empty()
        );
        assert_eq!(tracker.exceeded().unwrap().plan, CLAUDE_CODE);
        tracker.set_limits(PlanBudget::none().claude_code(PlanLimits::none().five_hour(20.0)));
        assert!(tracker.exceeded().is_none());
        let crossed = tracker.observe(&of(CLAUDE_CODE, 61.0, 18_000, 12.0), 0);
        assert_eq!(crossed.len(), 1);
    }
}
