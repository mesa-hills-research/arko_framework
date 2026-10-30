//! Each owner's outbox, which holds the news of all its monitors. The news
//! goes to the owner's runtime as one note at most once each
//! [`alert_interval`](MonitorLimits::alert_interval). With no runtime bound
//! for the owner (its agent is idle), news that should wake it is announced
//! once with a [`JobAlert`], and the runtime it binds next gets the news.

use super::monitor::{
    AlertKind, CheckChange, EndReason, JobAlert, Line, MonitorLifetime, MonitorLimits,
    MonitorTrigger, clock, clock_seconds, minutes_left,
};
use super::*;
use std::time::SystemTime;
use tokio::time::{Instant, sleep_until};

/// Which monitor news is from, as its note names it.
#[derive(Debug, Clone)]
pub(super) struct Header {
    pub(super) job_id: Uuid,
    pub(super) command: String,
    pub(super) trigger: MonitorTrigger,
}

/// A check's result, as its news gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CheckResult {
    pub(super) success: bool,
    /// What failed, as a sentence says it after "It" (`exited with code
    /// 7`), for a failure.
    pub(super) failure: Option<String>,
    pub(super) output: String,
    pub(super) at: SystemTime,
}

/// A monitor's news.
#[derive(Debug, Clone)]
pub(super) enum News {
    /// Output lines matched.
    Lines {
        lines: Vec<Line>,
        /// More lines matched lately than the limits call noisy.
        noisy: bool,
        at: SystemTime,
    },
    /// A check's result changed.
    Check {
        result: CheckResult,
        /// The change is one the monitor alerts of.
        alerts: bool,
        /// How long the result before lasted, or `None` for the first
        /// result.
        previous_for: Option<Duration>,
    },
    /// The monitor ends soon, at `ends_at`, and a rearm keeps it for up
    /// to `longest` from then.
    EndsSoon {
        ends_at: SystemTime,
        longest: Duration,
    },
    /// The monitor ended.
    Ended {
        reason: EndReason,
        at: SystemTime,
        /// The last lines of its output, for a command that exited.
        last_lines: Vec<String>,
    },
}

/// The lines a monitor matched since its owner's last note.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Lines {
    shown: Vec<String>,
    shown_bytes: usize,
    /// How many matched in all.
    count: u64,
    /// When the first came.
    since: SystemTime,
    /// Where the first that isn't shown starts in the output.
    rest_from: Option<u64>,
}

/// The changes of a check since its owner's last note. The first is one
/// the monitor alerts of.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Check {
    changes: u32,
    latest: CheckResult,
    /// The lines of the latest result's output a note shows.
    shown: Vec<String>,
    /// It has more output than that.
    cut: bool,
    /// How long the result before the first change lasted, or `None` when
    /// the first is the check's first result.
    previous_for: Option<Duration>,
}

/// How a monitor ended.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Ended {
    reason: EndReason,
    at: SystemTime,
    last_lines: Vec<String>,
}

/// One monitor's news since its owner's last note.
#[derive(Debug, Clone)]
pub(super) struct Part {
    header: Header,
    lines: Option<Lines>,
    noisy: bool,
    check: Option<Check>,
    /// When it ends, and the longest a rearm keeps it, for its warning.
    ends_soon: Option<(SystemTime, Duration)>,
    ended: Option<Ended>,
    /// What of it should wake the owner, until it is announced.
    wakes: Option<AlertKind>,
    /// The monitor was stopped in a way that wakes no one (the user's stop,
    /// its owner's, or the end of its owner's run), so none of its news
    /// does.
    quiet: bool,
}

impl Part {
    /// Whether the part should wake an idle owner, told of or not. An alert
    /// does, and so do a warning while the user counts as active (not
    /// `away`) and a command that exited by itself.
    fn wakes_idle(&self, away: bool) -> bool {
        !self.quiet
            && (self.lines.is_some()
                || self.check.is_some()
                || (self.ends_soon.is_some() && !away)
                || self
                    .ended
                    .as_ref()
                    .is_some_and(|ended| matches!(ended.reason, EndReason::Exited { .. })))
    }

    /// Whether the part carries an alert, which its monitor's count of
    /// alerts counts.
    fn alerts(&self) -> bool {
        self.lines.is_some() || self.check.is_some()
    }
}

/// An owner's outbox.
#[derive(Debug, Default)]
pub(super) struct Outbox {
    parts: Vec<Part>,
    /// When news first came that the owner hasn't been sent or told of.
    pending_since: Option<Instant>,
    /// When the owner was last sent a note, or woken.
    last_went: Option<Instant>,
    /// A task drives the outbox.
    pub(super) driving: bool,
}

impl Outbox {
    pub(super) fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Adds `news` of `header`'s monitor.
    pub(super) fn add(&mut self, header: Header, news: News, limits: &MonitorLimits, now: Instant) {
        // A change the monitor doesn't alert of only updates an alert that
        // hasn't gone.
        if let News::Check { alerts: false, .. } = news
            && !self
                .parts
                .iter()
                .any(|part| part.header.job_id == header.job_id && part.check.is_some())
        {
            return;
        }
        let index = match self
            .parts
            .iter()
            .position(|part| part.header.job_id == header.job_id)
        {
            Some(index) => index,
            None => {
                self.parts.push(Part {
                    header,
                    lines: None,
                    noisy: false,
                    check: None,
                    ends_soon: None,
                    ended: None,
                    wakes: None,
                    quiet: false,
                });
                self.parts.len() - 1
            }
        };
        let part = &mut self.parts[index];
        match news {
            News::Lines { lines, noisy, at } => {
                let pending = part.lines.get_or_insert_with(|| Lines {
                    shown: Vec::new(),
                    shown_bytes: 0,
                    count: 0,
                    since: at,
                    rest_from: None,
                });
                for line in lines {
                    pending.count += 1;
                    if pending.rest_from.is_none()
                        && pending.shown.len() < limits.lines_per_note
                        && pending.shown_bytes + line.text.len() <= limits.bytes_per_note
                    {
                        pending.shown_bytes += line.text.len();
                        pending.shown.push(line.text);
                    } else {
                        pending.rest_from.get_or_insert(line.start);
                    }
                }
                part.noisy |= noisy;
                part.wakes = Some(AlertKind::Lines);
            }
            News::Check {
                result,
                alerts,
                previous_for,
            } => {
                let (shown, cut) = shown_lines(&result.output, limits);
                match &mut part.check {
                    Some(pending) => {
                        pending.changes += 1;
                        pending.latest = result;
                        pending.shown = shown;
                        pending.cut = cut;
                    }
                    None => {
                        part.check = Some(Check {
                            changes: 1,
                            latest: result,
                            shown,
                            cut,
                            previous_for,
                        });
                    }
                }
                if alerts {
                    part.wakes = Some(AlertKind::Check);
                }
            }
            News::EndsSoon { ends_at, longest } => {
                part.ends_soon = Some((ends_at, longest));
                part.wakes = part.wakes.or(Some(AlertKind::EndsSoon));
            }
            News::Ended {
                reason,
                at,
                last_lines,
            } => {
                // The warning is moot, and a stop that wakes no one wakes
                // no one through the news before it either.
                part.ends_soon = None;
                if part.wakes == Some(AlertKind::EndsSoon) {
                    part.wakes = None;
                }
                if matches!(
                    reason,
                    EndReason::StoppedByUser | EndReason::StoppedByOwner | EndReason::OwnerEnded
                ) {
                    part.quiet = true;
                    part.wakes = None;
                }
                part.ended = Some(Ended {
                    reason,
                    at,
                    last_lines,
                });
            }
        }
        self.pending_since.get_or_insert(now);
    }

    /// When the news should go, or `None` when there's none that hasn't
    /// been sent or told of. News that comes after a quiet interval waits
    /// briefly for more, and the owner gets at most one note (or alert)
    /// each interval.
    pub(super) fn due(&self, limits: &MonitorLimits) -> Option<Instant> {
        let since = self.pending_since?;
        let gathered = since + limits.gather;
        Some(match self.last_went {
            Some(went) => gathered.max(went + limits.alert_interval),
            None => gathered,
        })
    }

    /// When the interval since the owner was last sent news or woken ends.
    fn quiet_from(&self, limits: &MonitorLimits) -> Option<Instant> {
        self.last_went.map(|went| went + limits.alert_interval)
    }

    /// Forgets monitor `id`'s warning, for a monitor that was rearmed.
    fn forget_warning(&mut self, id: Uuid) {
        for part in &mut self.parts {
            if part.header.job_id == id {
                part.ends_soon = None;
                if part.wakes == Some(AlertKind::EndsSoon) {
                    part.wakes = None;
                }
            }
        }
        self.parts
            .retain(|part| part.lines.is_some() || part.check.is_some() || part.ended.is_some());
        if self.parts.is_empty() {
            self.pending_since = None;
        }
    }

    /// Forgets monitor `id`'s news.
    fn forget(&mut self, id: Uuid) {
        self.parts.retain(|part| part.header.job_id != id);
        if self.parts.is_empty() {
            self.pending_since = None;
        }
    }

    /// Takes all the news for a note sent now.
    pub(super) fn take(&mut self, now: Instant) -> Vec<Part> {
        self.pending_since = None;
        if !self.parts.is_empty() {
            self.last_went = Some(now);
        }
        std::mem::take(&mut self.parts)
    }

    /// No runtime takes the news. Returns what to wake the owner for, the
    /// monitor and its kind of news, if any news should wake it, and marks
    /// the news as told of. The news waits for the owner's next runtime.
    /// Alerts come before warnings, and a warning wakes no one while the
    /// user is `away`.
    pub(super) fn announce(&mut self, now: Instant, away: bool) -> Option<(Uuid, AlertKind)> {
        self.pending_since = None;
        let wakes = |kind: AlertKind| {
            self.parts
                .iter()
                .find_map(|part| (part.wakes == Some(kind)).then_some((part.header.job_id, kind)))
        };
        let wake = wakes(AlertKind::Lines)
            .or_else(|| wakes(AlertKind::Check))
            .or_else(|| wakes(AlertKind::EndsSoon).filter(|_| !away));
        for part in &mut self.parts {
            part.wakes = None;
        }
        if wake.is_some() {
            self.last_went = Some(now);
        }
        wake
    }
}

/// A monitor as its note describes it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Standing {
    /// When it ends, as last armed.
    pub(super) expires_at: SystemTime,
    /// Its count of alerts, this note's included.
    pub(super) alerts: u32,
    /// It ends after its first alert.
    pub(super) once: bool,
    /// It lasts while its owner's workers run, and one does.
    pub(super) kept: bool,
}

/// The note for `parts`. `monitors` gives each monitor as it stands.
pub(super) fn note(parts: &[Part], monitors: &HashMap<Uuid, Standing>, now: SystemTime) -> String {
    parts
        .iter()
        .map(|part| part_text(part, monitors.get(&part.header.job_id), now))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// What an alert says of when its monitor ends, unless the note says that
/// it ended or ends soon. A monitor that alerts once ends with its alert,
/// and one whose owner's workers keep it lasts while they run. One
/// forgotten before its note went ends now.
fn ends_line(standing: Option<&Standing>, now: SystemTime) -> Option<String> {
    let expires_at = match standing {
        Some(standing) if standing.once => return None,
        Some(standing) if standing.kept => {
            return Some("\nIt lasts while your workers run.".to_string());
        }
        Some(standing) => standing.expires_at,
        None => now,
    };
    Some(format!(
        "\nIt ends in {}.",
        minutes_left(expires_at.duration_since(now).unwrap_or_default())
    ))
}

/// The text of one monitor's part of a note. A job forgotten before its note
/// went has no `standing`.
fn part_text(part: &Part, standing: Option<&Standing>, now: SystemTime) -> String {
    let alerts = standing.map(|standing| standing.alerts);
    let header = &part.header;
    let id = header.job_id;
    let command = &header.command;
    let mut paragraphs = Vec::new();
    if let Some(lines) = &part.lines {
        let every = matches!(header.trigger, MonitorTrigger::Lines { pattern: None });
        let kind = if every { "" } else { "matching " };
        let mut text = format!(
            "Monitor alert. Job {id} (`{command}`, {}) printed {} {kind}{} since {}.",
            header.trigger.named(),
            lines.count,
            plural(lines.count, "line"),
            clock_seconds(lines.since)
        );
        for line in &lines.shown {
            text.push_str("\n  ");
            text.push_str(line);
        }
        if let Some(from) = lines.rest_from {
            let more = lines.count - lines.shown.len() as u64;
            text.push_str(&format!(
                "\n{more} more {kind}{} {} in its output. terminal_status with after_output {from} reads {}.",
                plural(more, "line"),
                if more == 1 { "is" } else { "are" },
                if more == 1 { "it" } else { "them" },
            ));
        }
        if part.noisy {
            text.push_str(if every {
                "\nA pattern saves turns."
            } else {
                "\nA narrower pattern saves turns."
            });
        }
        if part.ended.is_none() && part.ends_soon.is_none() {
            text.extend(ends_line(standing, now));
        }
        paragraphs.push(text);
    }
    if let Some(check) = &part.check {
        let mut text = check_text(header, check);
        if part.ended.is_none() && part.ends_soon.is_none() {
            text.extend(ends_line(standing, now));
        }
        paragraphs.push(text);
    }
    if let Some((ends_at, longest)) = part.ends_soon {
        paragraphs.push(format!(
            "Monitor {id} (`{command}`) ends in {}, at {}. monitor_control rearm keeps it going for up to {} from now. If you no longer need it, let it end.",
            minutes_left(ends_at.duration_since(now).unwrap_or_default()),
            clock(ends_at),
            calls::describe(longest)
        ));
    }
    if let Some(ended) = &part.ended {
        paragraphs.push(ended_text(header, ended, alerts));
    }
    paragraphs.join("\n\n")
}

/// The lines of a check's output a note shows, and whether there's more.
fn shown_lines(output: &str, limits: &MonitorLimits) -> (Vec<String>, bool) {
    let lines: Vec<&str> = output
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect();
    let mut shown = Vec::new();
    let mut bytes = 0;
    let mut cut = false;
    for line in &lines {
        if shown.len() == limits.lines_per_note
            || (bytes + line.len() > limits.bytes_per_note && !shown.is_empty())
        {
            cut = true;
            break;
        }
        let mut end = line.len().min(limits.bytes_per_note);
        while !line.is_char_boundary(end) {
            end -= 1;
        }
        cut |= end < line.len();
        bytes += end;
        shown.push(line[..end].to_string());
    }
    (shown, cut)
}

/// What a note says of a check's changes.
fn check_text(header: &Header, check: &Check) -> String {
    let id = header.job_id;
    let command = &header.command;
    let every = header.trigger.named();
    let latest = &check.latest;
    let at = clock_seconds(latest.at);
    let output = matches!(
        header.trigger,
        MonitorTrigger::Check {
            when: CheckChange::Output,
            ..
        }
    );
    let before = check.previous_for.map(calls::describe);
    let mut text = match (check.changes, output, latest.success, before) {
        // The first result, for a monitor that alerts once of the result
        // it waits for.
        (1, _, success, None) => format!(
            "Monitor alert. The first check of job {id} (`{command}`, {every}) {} at {at}.",
            if success { "succeeded" } else { "failed" }
        ),
        (1, true, _, Some(before)) => format!(
            "Monitor alert. The output of the check of job {id} (`{command}`, {every}) changed at {at}, after staying the same for {before}."
        ),
        (1, false, true, Some(before)) => format!(
            "Monitor alert. The check of job {id} (`{command}`, {every}) started succeeding at {at}, after failing for {before}."
        ),
        (1, false, false, Some(before)) => format!(
            "Monitor alert. The check of job {id} (`{command}`, {every}) started failing at {at}, after succeeding for {before}."
        ),
        (changes, true, _, _) => format!(
            "Monitor alert. The output of the check of job {id} (`{command}`, {every}) changed {changes} times since the last alert, most recently at {at}."
        ),
        (changes, false, success, _) => format!(
            "Monitor alert. The check of job {id} (`{command}`, {every}) changed {changes} times since the last alert, and now {} (since {at}).",
            if success { "succeeds" } else { "fails" }
        ),
    };
    if let Some(failure) = &latest.failure {
        text.push_str(&format!(" It {failure}."));
    }
    if check.shown.is_empty() {
        text.push_str(" It printed nothing.");
    } else {
        text.push_str(" Its output follows.");
        for line in &check.shown {
            text.push_str("\n  ");
            text.push_str(line);
        }
        if check.cut {
            text.push_str("\nterminal_status shows the rest of it.");
        }
    }
    text
}

/// ` It alerted 4 times.`, or nothing when the count isn't known (the job
/// was forgotten).
fn alerted(alerts: Option<u32>) -> String {
    match alerts {
        None => String::new(),
        Some(0) => " It didn't alert.".to_string(),
        Some(1) => " It alerted once.".to_string(),
        Some(count) => format!(" It alerted {count} times."),
    }
}

/// What a note says of a monitor that ended.
fn ended_text(header: &Header, ended: &Ended, alerts: Option<u32>) -> String {
    let id = header.job_id;
    let command = &header.command;
    let at = clock(ended.at);
    let checks = matches!(header.trigger, MonitorTrigger::Check { .. });
    match &ended.reason {
        EndReason::Expired if checks => format!(
            "Monitor {id} (`{command}`) ended at {at}, and its checks stopped.{}",
            alerted(alerts)
        ),
        EndReason::Expired => format!(
            "Monitor {id} (`{command}`) ended at {at} and its command was stopped.{}",
            alerted(alerts)
        ),
        EndReason::WorkersFinished if checks => format!(
            "Monitor {id} (`{command}`) ended at {at} because your workers finished, and its checks stopped.{}",
            alerted(alerts)
        ),
        EndReason::WorkersFinished => format!(
            "Monitor {id} (`{command}`) ended at {at} because your workers finished, and its command was stopped.{}",
            alerted(alerts)
        ),
        EndReason::Alerted if checks => {
            format!(
                "Monitor {id} (`{command}`) ended after its first alert, and its checks stopped."
            )
        }
        EndReason::Alerted => format!(
            "Monitor {id} (`{command}`) ended after its first alert, and its command was stopped."
        ),
        EndReason::Exited { code } => {
            let how = match code {
                Some(code) => format!("exited with code {code}"),
                None => "ended without an exit code".to_string(),
            };
            let mut text =
                format!("Monitor {id} (`{command}`) ended because its command {how} at {at}.");
            if ended.last_lines.is_empty() {
                text.push_str(" It printed nothing.");
            } else {
                text.push_str(" Its last lines follow.");
                for line in &ended.last_lines {
                    text.push_str("\n  ");
                    text.push_str(line);
                }
            }
            text
        }
        EndReason::StoppedByOwner => {
            format!(
                "Monitor {id} (`{command}`) stopped, as you asked.{}",
                alerted(alerts)
            )
        }
        EndReason::StoppedByUser => format!("The user stopped monitor {id} (`{command}`)."),
        EndReason::OwnerEnded => {
            format!("Monitor {id} (`{command}`) stopped because your run ended.")
        }
        EndReason::Failed { error } => {
            format!("Monitor {id} (`{command}`) failed at {at}, so it ended ({error}).")
        }
    }
}

/// How an alert's paragraph in a note begins.
const ALERT_START: &str = "Monitor alert. ";

/// How a check's alert begins, after [`ALERT_START`].
const CHECK_ALERT_START: &str = "Monitor alert. The ";

/// How a warning's paragraph in a note ends.
const WARNING_END: &str = "If you no longer need it, let it end.";

/// What a note of the terminal manager's (the content of the
/// `InjectContext` a runtime got) should wake its owner's idle agent for,
/// if anything. A monitor's alert gives [`AlertKind::Lines`] or
/// [`AlertKind::Check`], and its warning before it ends
/// [`AlertKind::EndsSoon`]. Other notes (a monitor's end, a job's update)
/// give `None`. A host that finds such a note still waiting in a task after
/// its turn ended (it came during the agent's last request) can start a
/// turn for it, as the manager would have woken the agent for it then. A
/// warning should wake the agent only while its user counts as active.
pub fn note_wakes(note: &str) -> Option<AlertKind> {
    let mut wakes = None;
    for line in note.lines() {
        if line.starts_with(CHECK_ALERT_START) {
            return Some(AlertKind::Check);
        }
        if line.starts_with(ALERT_START) {
            return Some(AlertKind::Lines);
        }
        if line.starts_with("Monitor ") && line.ends_with(WARNING_END) {
            wakes = Some(AlertKind::EndsSoon);
        }
    }
    wakes
}

/// `line` or `lines`.
fn plural(count: u64, word: &str) -> String {
    if count == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

impl TerminalManager {
    /// Adds `news` of `header`'s monitor to `owner`'s outbox, and drives
    /// the outbox until it has nothing left to send.
    pub(super) fn queue_news(self: &Arc<Self>, owner: Uuid, header: Header, news: News) {
        let limits = self.monitor_limits();
        let drive = {
            let mut outboxes = self.outboxes.lock().unwrap();
            let outbox = outboxes.entry(owner).or_default();
            outbox.add(header, news, &limits, Instant::now());
            !std::mem::replace(&mut outbox.driving, true)
        };
        if drive {
            tokio::spawn(drive_outbox(Arc::downgrade(self), owner));
        }
    }

    /// Forgets the news of `owner`'s monitor `id`, which its owner has
    /// been told some other way, or whose job was forgotten.
    pub(super) fn drop_news(&self, owner: Uuid, id: Uuid) {
        if let Some(outbox) = self.outboxes.lock().unwrap().get_mut(&owner) {
            outbox.forget(id);
        }
        self.drop_empty_outbox(owner);
    }

    /// Removes `owner`'s outbox when it holds nothing, no task drives it,
    /// and its interval has passed (else the next news wouldn't wait for
    /// it).
    fn drop_empty_outbox(&self, owner: Uuid) {
        let limits = self.monitor_limits();
        let mut outboxes = self.outboxes.lock().unwrap();
        if outboxes.get(&owner).is_some_and(|outbox| {
            outbox.is_empty()
                && !outbox.driving
                && outbox
                    .quiet_from(&limits)
                    .is_none_or(|quiet| quiet <= Instant::now())
        }) {
            outboxes.remove(&owner);
        }
    }

    /// Forgets the warning of `owner`'s monitor `id`, which was rearmed.
    pub(super) fn drop_warning(&self, owner: Uuid, id: Uuid) {
        if let Some(outbox) = self.outboxes.lock().unwrap().get_mut(&owner) {
            outbox.forget_warning(id);
        }
    }

    /// Whether `owner` has monitor news it hasn't been sent that should
    /// wake it (see [`TerminalManager::has_alerts`]).
    pub(super) fn has_monitor_alerts(&self, owner: Uuid) -> bool {
        let away = self.user_away();
        self.outboxes
            .lock()
            .unwrap()
            .get(&owner)
            .is_some_and(|outbox| outbox.parts.iter().any(|part| part.wakes_idle(away)))
    }

    /// Whether `owner` has monitor news it hasn't been sent.
    pub(super) fn has_monitor_news(&self, owner: Uuid) -> bool {
        self.outboxes
            .lock()
            .unwrap()
            .get(&owner)
            .is_some_and(|outbox| !outbox.is_empty())
    }

    /// Takes `owner`'s news as a note, counting the alerts it carries.
    fn take_note(&self, owner: Uuid) -> Option<String> {
        let parts = self
            .outboxes
            .lock()
            .unwrap()
            .get_mut(&owner)?
            .take(Instant::now());
        if parts.is_empty() {
            return None;
        }
        // Asked before any job's lock is taken.
        let workers = self.workers_running(owner);
        let mut monitors = HashMap::new();
        for part in &parts {
            let id = part.header.job_id;
            let Ok(job) = self.owned(owner, id) else {
                continue;
            };
            let mut data = job.data.lock().unwrap();
            let Some(monitor) = data.record.monitor.as_mut() else {
                continue;
            };
            if part.alerts() {
                monitor.alerts += 1;
            }
            monitors.insert(
                id,
                Standing {
                    expires_at: monitor.expires_at,
                    alerts: monitor.alerts,
                    once: monitor.once,
                    kept: workers && monitor.lifetime == MonitorLifetime::WhileWorkersRun,
                },
            );
            // An ended job's record is final, and may be forgotten.
            if part.alerts()
                && data.record.state == JobState::Running
                && let Err(error) = persist(&self.config, &data.record)
            {
                log::warn!("Terminal job {id}: could not save its count of alerts: {error}");
            }
        }
        Some(note(&parts, &monitors, SystemTime::now()))
    }

    /// No runtime took `owner`'s news. Wakes it if the news should.
    fn announce_news(&self, owner: Uuid) {
        let away = self.user_away();
        let wake = self
            .outboxes
            .lock()
            .unwrap()
            .get_mut(&owner)
            .and_then(|outbox| outbox.announce(Instant::now(), away));
        if let Some((job_id, kind)) = wake {
            let _ = self.alerts.send(JobAlert {
                owner,
                job_id,
                kind,
            });
        }
    }

    /// Sends `owner`'s news to the runtime bound now, if there is one, for
    /// a runtime that just bound. When its channel has room the news goes
    /// at once, so the runtime has it before its first request.
    pub(super) fn flush_news(self: &Arc<Self>, owner: Uuid) {
        if !self.has_monitor_news(owner) {
            return;
        }
        let sender = self
            .runtimes
            .lock()
            .unwrap()
            .get(&owner)
            .and_then(|binding| binding.sender.upgrade());
        if let Some(sender) = sender
            && let Ok(permit) = sender.try_reserve()
        {
            if let Some(content) = self.take_note(owner) {
                permit.send(UserCommand::InjectContext { content });
            }
            self.drop_empty_outbox(owner);
            return;
        }
        let manager = Arc::downgrade(self);
        tokio::spawn(async move {
            send_news(&manager, owner).await;
        });
    }
}

/// Sends `owner`'s news when it's due, until it has none left.
async fn drive_outbox(manager: Weak<TerminalManager>, owner: Uuid) {
    loop {
        // When to look again, or `None` to send the news now.
        let wait = {
            let Some(manager) = manager.upgrade() else {
                return;
            };
            let limits = manager.monitor_limits();
            let mut outboxes = manager.outboxes.lock().unwrap();
            let Some(outbox) = outboxes.get_mut(&owner) else {
                return;
            };
            let now = Instant::now();
            match outbox.due(&limits) {
                Some(due) if due > now => Some(due),
                Some(_) => None,
                // Emptied. It goes once its interval has passed, so owners
                // that are gone keep none, and news that comes before that
                // still waits for the interval.
                None if outbox.is_empty() => match outbox.quiet_from(&limits) {
                    Some(quiet) if quiet > now => Some(quiet),
                    _ => {
                        outboxes.remove(&owner);
                        return;
                    }
                },
                None => {
                    outbox.driving = false;
                    return;
                }
            }
        };
        if let Some(wait) = wait {
            sleep_until(wait).await;
            continue;
        }
        if !send_news(&manager, owner).await
            && let Some(manager) = manager.upgrade()
        {
            manager.announce_news(owner);
        }
    }
}

/// Sends `owner`'s news to its runtime as a note. Returns `false` when no
/// runtime is bound for it, or the one bound has ended.
async fn send_news(manager: &Weak<TerminalManager>, owner: Uuid) -> bool {
    let Some(binding) = manager
        .upgrade()
        .and_then(|manager| manager.runtimes.lock().unwrap().get(&owner).cloned())
    else {
        return false;
    };
    let Some(sender) = binding.sender.upgrade() else {
        return false;
    };
    // Reserved first, so a note waits for room without losing news that
    // comes meanwhile, which goes with it.
    let Ok(permit) = sender.reserve_owned().await else {
        return false;
    };
    let Some(manager) = manager.upgrade() else {
        return true;
    };
    let current = manager
        .runtimes
        .lock()
        .unwrap()
        .get(&owner)
        .is_some_and(|current| Arc::ptr_eq(&current.pending, &binding.pending));
    // A runtime that bound meanwhile was sent the news as it bound.
    if current && let Some(content) = manager.take_note(owner) {
        permit.send(UserCommand::InjectContext { content });
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(id: Uuid, pattern: Option<&str>) -> Header {
        Header {
            job_id: id,
            command: "npm run dev".into(),
            trigger: MonitorTrigger::Lines {
                pattern: pattern.map(str::to_string),
            },
        }
    }

    fn lines(texts: &[&str], from: u64) -> News {
        News::Lines {
            lines: texts
                .iter()
                .enumerate()
                .map(|(index, text)| Line {
                    text: text.to_string(),
                    start: from + 100 * index as u64,
                })
                .collect(),
            noisy: false,
            at: at(10 * 3600 + 41 * 60 + 5),
        }
    }

    fn at(seconds: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)
    }

    /// A timed monitor that ends at `expires_at`, with `alerts`.
    fn standing(expires_at: SystemTime, alerts: u32) -> Standing {
        Standing {
            expires_at,
            alerts,
            once: false,
            kept: false,
        }
    }

    /// News waits briefly for more, then goes. After that, the owner gets
    /// at most one note an interval, whatever number of monitors it has,
    /// counted from when a note went.
    #[test]
    fn news_goes_at_most_once_an_interval() {
        let limits = MonitorLimits::default();
        let start = Instant::now();
        let mut outbox = Outbox::default();
        assert_eq!(outbox.due(&limits), None);
        let (one, two) = (Uuid::new_v4(), Uuid::new_v4());
        outbox.add(
            header(one, Some("error")),
            lines(&["error 1"], 0),
            &limits,
            start,
        );
        assert_eq!(outbox.due(&limits), Some(start + limits.gather));
        // More news doesn't put it off.
        let soon = start + Duration::from_secs(1);
        outbox.add(header(two, None), lines(&["ready"], 0), &limits, soon);
        assert_eq!(outbox.due(&limits), Some(start + limits.gather));
        let sent = start + limits.gather;
        let parts = outbox.take(sent);
        assert_eq!(parts.len(), 2, "one note holds both monitors' news");
        assert_eq!(outbox.due(&limits), None);

        // News soon after waits for the interval, from when the note went.
        let next = sent + Duration::from_secs(5);
        outbox.add(
            header(one, Some("error")),
            lines(&["error 2"], 200),
            &limits,
            next,
        );
        assert_eq!(outbox.due(&limits), Some(sent + limits.alert_interval));
        outbox.take(sent + limits.alert_interval);
        // After a quiet interval, it gathers again.
        let quiet = sent + limits.alert_interval * 3;
        outbox.add(
            header(one, Some("error")),
            lines(&["error 3"], 300),
            &limits,
            quiet,
        );
        assert_eq!(outbox.due(&limits), Some(quiet + limits.gather));
    }

    /// With no runtime to take it, news that should wake the owner is
    /// announced once, and that counts as news going. It waits for the
    /// owner's next runtime. A monitor's end alone wakes no one.
    #[test]
    fn news_no_runtime_takes_is_announced_once() {
        let limits = MonitorLimits::default();
        let start = Instant::now();
        let mut outbox = Outbox::default();
        let id = Uuid::new_v4();
        outbox.add(
            header(id, Some("error")),
            lines(&["error"], 0),
            &limits,
            start,
        );
        assert_eq!(
            outbox.announce(start + limits.gather, false),
            Some((id, AlertKind::Lines))
        );
        assert!(!outbox.is_empty());
        assert_eq!(outbox.due(&limits), None, "told of, it waits");
        let ended = News::Ended {
            reason: EndReason::StoppedByUser,
            at: at(0),
            last_lines: Vec::new(),
        };
        let later = start + Duration::from_secs(5);
        outbox.add(header(id, Some("error")), ended, &limits, later);
        assert_eq!(
            outbox.due(&limits),
            Some(start + limits.gather + limits.alert_interval)
        );
        assert_eq!(outbox.announce(later, false), None, "an end wakes no one");
        // A runtime that binds gets everything.
        let parts = outbox.take(later);
        assert!(parts[0].lines.is_some() && parts[0].ended.is_some());
    }

    /// A note shows up to the limit of a monitor's lines, counts the rest,
    /// and says where its output has them.
    #[test]
    fn a_note_shows_lines_up_to_the_limit_and_counts_the_rest() {
        let limits = MonitorLimits {
            lines_per_note: 2,
            ..MonitorLimits::default()
        };
        let mut outbox = Outbox::default();
        let id = Uuid::new_v4();
        let now = Instant::now();
        let header = header(id, Some("error|warn"));
        outbox.add(
            header.clone(),
            lines(&["[ERROR] Failed to compile", "[ERROR] No module"], 0),
            &limits,
            now,
        );
        outbox.add(
            header.clone(),
            lines(&["warn: 1", "warn: 2", "warn: 3"], 18234),
            &limits,
            now,
        );
        let parts = outbox.take(now);
        let ends = at(11 * 3600 + 3 * 60 + 5);
        let monitors = HashMap::from([(id, standing(ends, 1))]);
        assert_eq!(
            note(&parts, &monitors, at(10 * 3600 + 41 * 60 + 5)),
            format!(
                "Monitor alert. Job {id} (`npm run dev`, lines that match `error|warn`) printed 5 matching lines since 10:41:05 UTC.\n  [ERROR] Failed to compile\n  [ERROR] No module\n3 more matching lines are in its output. terminal_status with after_output 18234 reads them.\nIt ends in 22 min."
            )
        );
        // Every line, one more, and a noisy run.
        let mut outbox = Outbox::default();
        let every = Header {
            trigger: MonitorTrigger::Lines { pattern: None },
            ..header
        };
        let mut news = lines(&["a", "b", "c"], 7);
        if let News::Lines { noisy, .. } = &mut news {
            *noisy = true;
        }
        outbox.add(every, news, &limits, now);
        let text = note(&outbox.take(now), &monitors, at(10 * 3600 + 41 * 60 + 5));
        assert!(
            text.contains("printed 3 lines since")
                && text.contains(
                    "\n1 more line is in its output. terminal_status with after_output 207 reads it.\nA pattern saves turns.\n"
                ),
            "{text}"
        );
    }

    fn check_header(id: Uuid, when: CheckChange) -> Header {
        Header {
            job_id: id,
            command: "curl -sf localhost:3000/health".into(),
            trigger: MonitorTrigger::Check {
                every_seconds: 15,
                when,
            },
        }
    }

    fn checked(success: bool, output: &str, seconds: u64, alerts: bool) -> News {
        News::Check {
            result: CheckResult {
                success,
                failure: (!success).then(|| "exited with code 7".to_string()),
                output: output.into(),
                at: at(seconds),
            },
            alerts,
            previous_for: Some(Duration::from_secs(180)),
        }
    }

    /// A check's note says how its result changed and gives its output. A
    /// check that changed several times since the last alert says how
    /// often, and how it is now. A change it doesn't alert of goes only with
    /// an alert.
    #[test]
    fn a_note_says_how_a_check_changed() {
        let limits = MonitorLimits::default();
        let id = Uuid::new_v4();
        let now = Instant::now();
        let monitors = HashMap::from([(id, standing(at(10 * 3600 + 70 * 60 + 30), 1))]);
        let header = check_header(id, CheckChange::Success);
        let mut outbox = Outbox::default();
        outbox.add(header.clone(), checked(false, "", 10, false), &limits, now);
        assert!(outbox.is_empty(), "a change it doesn't alert of, alone");
        let ok = 10 * 3600 + 44 * 60 + 30;
        outbox.add(
            header.clone(),
            checked(true, "{\"status\":\"ok\"}\n", ok, true),
            &limits,
            now,
        );
        assert_eq!(
            note(&outbox.take(now), &monitors, at(ok)),
            format!(
                "Monitor alert. The check of job {id} (`curl -sf localhost:3000/health`, every 15 s) started succeeding at 10:44:30 UTC, after failing for 3 min. Its output follows.\n  {{\"status\":\"ok\"}}\nIt ends in 26 min."
            )
        );

        // Flapping: the alert, then changes it doesn't alert of.
        outbox.add(header.clone(), checked(true, "", ok, true), &limits, now);
        outbox.add(
            header.clone(),
            checked(false, "", ok + 15, false),
            &limits,
            now,
        );
        outbox.add(
            header.clone(),
            checked(true, "", ok + 30, true),
            &limits,
            now,
        );
        outbox.add(
            header,
            checked(false, "refused", ok + 45, false),
            &limits,
            now,
        );
        assert_eq!(
            note(&outbox.take(now), &monitors, at(ok)),
            format!(
                "Monitor alert. The check of job {id} (`curl -sf localhost:3000/health`, every 15 s) changed 4 times since the last alert, and now fails (since 10:45:15 UTC). It exited with code 7. Its output follows.\n  refused\nIt ends in 26 min."
            )
        );

        // Its output, which runs over the limit.
        let limits = MonitorLimits {
            lines_per_note: 1,
            ..MonitorLimits::default()
        };
        outbox.add(
            check_header(id, CheckChange::Output),
            checked(true, "v2\nmore", ok, true),
            &limits,
            now,
        );
        assert_eq!(
            note(&outbox.take(now), &monitors, at(ok)),
            format!(
                "Monitor alert. The output of the check of job {id} (`curl -sf localhost:3000/health`, every 15 s) changed at 10:44:30 UTC, after staying the same for 3 min. Its output follows.\n  v2\nterminal_status shows the rest of it.\nIt ends in 26 min."
            )
        );
    }

    /// The warning before a monitor ends says how long it has left and how
    /// to keep it. It wakes an idle owner unless the user is away, after
    /// any alert. A rearm or the monitor's end makes it moot.
    #[test]
    fn a_warning_says_how_long_is_left() {
        let limits = MonitorLimits::default();
        let (id, other) = (Uuid::new_v4(), Uuid::new_v4());
        let now = Instant::now();
        let warning = || News::EndsSoon {
            ends_at: at(11 * 3600 + 42 * 60),
            longest: Duration::from_secs(1800),
        };
        let mut outbox = Outbox::default();
        outbox.add(header(id, None), warning(), &limits, now);
        assert_eq!(
            note(&outbox.take(now), &HashMap::new(), at(11 * 3600 + 20 * 60)),
            format!(
                "Monitor {id} (`npm run dev`) ends in 22 min, at 11:42 UTC. monitor_control rearm keeps it going for up to 30 min from now. If you no longer need it, let it end."
            )
        );
        outbox.add(header(id, None), warning(), &limits, now);
        assert_eq!(outbox.announce(now, true), None, "the user is away");
        outbox.add(header(other, None), lines(&["boom"], 0), &limits, now);
        outbox.add(header(id, None), warning(), &limits, now);
        assert_eq!(
            outbox.announce(now, false),
            Some((other, AlertKind::Lines)),
            "an alert first"
        );
        outbox.add(header(id, None), warning(), &limits, now);
        assert_eq!(outbox.announce(now, false), Some((id, AlertKind::EndsSoon)));
        outbox.take(now);
        // An alert with a warning gives the time left once.
        outbox.add(header(id, None), lines(&["boom"], 0), &limits, now);
        outbox.add(header(id, None), warning(), &limits, now);
        let text = note(&outbox.take(now), &HashMap::new(), at(11 * 3600 + 37 * 60));
        assert!(
            !text.contains("It ends in")
                && text.contains("\n  boom\n\nMonitor ")
                && text.contains(" ends in 5 min, at 11:42 UTC."),
            "{text}"
        );

        outbox.add(header(id, None), warning(), &limits, now);
        outbox.forget_warning(id);
        assert!(outbox.is_empty() && outbox.due(&limits).is_none());
        outbox.add(header(id, None), warning(), &limits, now);
        outbox.add(
            header(id, None),
            News::Ended {
                reason: EndReason::Expired,
                at: at(11 * 3600 + 42 * 60),
                last_lines: Vec::new(),
            },
            &limits,
            now,
        );
        let text = note(&outbox.take(now), &HashMap::new(), at(11 * 3600 + 42 * 60));
        assert!(
            !text.contains("rearm") && text.contains("ended at 11:42 UTC"),
            "{text}"
        );
        // Its job was forgotten, so its count of alerts isn't known.
        assert!(
            text.ends_with("ended at 11:42 UTC and its command was stopped."),
            "{text}"
        );
    }

    /// A note's alerts and warning are told from its other news, as a host
    /// reads them in a task.
    #[test]
    fn a_note_says_what_it_wakes_for() {
        let limits = MonitorLimits::default();
        let now = Instant::now();
        let id = Uuid::new_v4();
        let monitors = HashMap::from([(id, standing(at(11 * 3600), 1))]);
        let text = |news: Vec<(Header, News)>| {
            let mut outbox = Outbox::default();
            for (header, news) in news {
                outbox.add(header, news, &limits, now);
            }
            note(&outbox.take(now), &monitors, at(10 * 3600))
        };
        let ended = |reason| News::Ended {
            reason,
            at: at(0),
            last_lines: Vec::new(),
        };
        let warning = News::EndsSoon {
            ends_at: at(11 * 3600),
            longest: Duration::from_secs(1800),
        };
        let checked = News::Check {
            result: CheckResult {
                success: true,
                failure: None,
                output: String::new(),
                at: at(0),
            },
            alerts: true,
            previous_for: Some(Duration::from_secs(60)),
        };
        let lines_header = header(id, Some("error"));
        let check_header = Header {
            trigger: MonitorTrigger::Check {
                every_seconds: 15,
                when: CheckChange::Change,
            },
            ..header(id, None)
        };
        assert_eq!(
            note_wakes(&text(vec![(lines_header.clone(), lines(&["error"], 0))])),
            Some(AlertKind::Lines)
        );
        assert_eq!(
            note_wakes(&text(vec![(check_header, checked)])),
            Some(AlertKind::Check)
        );
        assert_eq!(
            note_wakes(&text(vec![(lines_header.clone(), warning)])),
            Some(AlertKind::EndsSoon)
        );
        // An alert beside another monitor's end.
        let other = header(Uuid::new_v4(), None);
        assert_eq!(
            note_wakes(&text(vec![
                (other.clone(), ended(EndReason::Expired)),
                (lines_header.clone(), lines(&["error"], 0)),
            ])),
            Some(AlertKind::Lines)
        );
        for reason in [
            EndReason::Expired,
            EndReason::StoppedByUser,
            EndReason::Exited { code: Some(1) },
        ] {
            assert_eq!(
                note_wakes(&text(vec![(other.clone(), ended(reason))])),
                None
            );
        }
        assert_eq!(note_wakes("Terminal update:\n{}"), None);
    }

    /// A stop that wakes no one (the user's, the owner's, or the end of the
    /// owner's run) takes the wake of the news that waited before it, and
    /// any end takes a warning's. An expiry keeps an alert's wake.
    #[test]
    fn a_stop_that_wakes_no_one_quiets_the_news_before_it() {
        let limits = MonitorLimits::default();
        let now = Instant::now();
        let id = Uuid::new_v4();
        let ended = |reason| News::Ended {
            reason,
            at: at(0),
            last_lines: Vec::new(),
        };
        let warning = || News::EndsSoon {
            ends_at: at(0),
            longest: Duration::from_secs(1800),
        };
        for reason in [
            EndReason::StoppedByUser,
            EndReason::StoppedByOwner,
            EndReason::OwnerEnded,
        ] {
            let mut outbox = Outbox::default();
            outbox.add(header(id, None), lines(&["boom"], 0), &limits, now);
            outbox.add(header(id, None), ended(reason.clone()), &limits, now);
            assert_eq!(outbox.announce(now, false), None, "{reason:?}");
            assert!(!outbox.is_empty(), "the news waits for the next turn");
        }
        let mut outbox = Outbox::default();
        outbox.add(header(id, None), warning(), &limits, now);
        outbox.add(header(id, None), ended(EndReason::Expired), &limits, now);
        assert_eq!(outbox.announce(now, false), None);
        let mut outbox = Outbox::default();
        outbox.add(header(id, None), lines(&["boom"], 0), &limits, now);
        outbox.add(header(id, None), ended(EndReason::Expired), &limits, now);
        assert_eq!(outbox.announce(now, false), Some((id, AlertKind::Lines)));
    }

    /// A monitor's end says how it ended, after the news that came before
    /// it, and how often it alerted.
    #[test]
    fn a_note_says_how_a_monitor_ended() {
        let limits = MonitorLimits::default();
        let id = Uuid::new_v4();
        let now = Instant::now();
        let ended = |reason: EndReason, last_lines: &[&str]| {
            let mut outbox = Outbox::default();
            outbox.add(
                header(id, None),
                News::Ended {
                    reason,
                    at: at(11 * 3600 + 42 * 60),
                    last_lines: last_lines.iter().map(|line| line.to_string()).collect(),
                },
                &limits,
                now,
            );
            note(
                &outbox.take(now),
                &HashMap::from([(id, standing(at(0), 4))]),
                at(11 * 3600 + 42 * 60),
            )
        };
        assert_eq!(
            ended(EndReason::Expired, &[]),
            format!(
                "Monitor {id} (`npm run dev`) ended at 11:42 UTC and its command was stopped. It alerted 4 times."
            )
        );
        assert_eq!(
            ended(
                EndReason::Exited { code: Some(1) },
                &["Error: listen EADDRINUSE"]
            ),
            format!(
                "Monitor {id} (`npm run dev`) ended because its command exited with code 1 at 11:42 UTC. Its last lines follow.\n  Error: listen EADDRINUSE"
            )
        );
        assert_eq!(
            ended(EndReason::StoppedByUser, &[]),
            format!("The user stopped monitor {id} (`npm run dev`).")
        );
        assert_eq!(
            ended(EndReason::StoppedByOwner, &[]),
            format!("Monitor {id} (`npm run dev`) stopped, as you asked. It alerted 4 times.")
        );
        // Lines that came before the end come first, without the time left.
        let mut outbox = Outbox::default();
        outbox.add(header(id, None), lines(&["boom"], 0), &limits, now);
        outbox.add(
            header(id, None),
            News::Ended {
                reason: EndReason::Exited { code: None },
                at: at(11 * 3600),
                last_lines: Vec::new(),
            },
            &limits,
            now,
        );
        let text = note(
            &outbox.take(now),
            &HashMap::from([(id, standing(at(0), 1))]),
            at(0),
        );
        assert!(
            text.ends_with(&format!(
                "\n  boom\n\nMonitor {id} (`npm run dev`) ended because its command ended without an exit code at 11:00 UTC. It printed nothing."
            )),
            "{text}"
        );
    }
}
