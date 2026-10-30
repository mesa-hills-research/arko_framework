//! Monitors, jobs that watch a command for their owner and alert it. A
//! monitor alerts of the output lines that match a pattern, or of a check,
//! run again and again, whose result changes. Its news goes to its owner's
//! outbox, which sends one note at most each
//! [`alert_interval`](MonitorLimits::alert_interval) for all of the owner's
//! monitors.

use super::outbox::{Header, News};
use super::*;
use std::time::SystemTime;
use tokio::time::Instant;

/// What a monitor watches. Made with [`MonitorOptions::lines`] or
/// [`MonitorOptions::check`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MonitorTrigger {
    /// The lines of the command's output that match `pattern`, a regular
    /// expression, or every line with `None`. The command runs until it
    /// exits or the monitor ends.
    #[non_exhaustive]
    Lines {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pattern: Option<String>,
    },
    /// The command, run as a check every `every_seconds`, and an alert when
    /// its result changes as `when` says.
    #[non_exhaustive]
    Check {
        every_seconds: u64,
        #[serde(default)]
        when: CheckChange,
    },
}

impl MonitorTrigger {
    /// What the monitor watches, for a list (`lines that match error|warn`,
    /// `every line`, `a check every 15 s`).
    pub fn watching(&self) -> String {
        match self {
            Self::Lines {
                pattern: Some(pattern),
            } => format!("lines that match {pattern}"),
            Self::Lines { pattern: None } => "every line".to_string(),
            Self::Check { every_seconds, .. } => format!("a check every {every_seconds} s"),
        }
    }

    /// What the monitor watches, as its notes name it, with the pattern as
    /// code.
    pub(super) fn named(&self) -> String {
        match self {
            Self::Lines {
                pattern: Some(pattern),
            } => format!("lines that match `{pattern}`"),
            Self::Lines { pattern: None } => "every line".to_string(),
            Self::Check { every_seconds, .. } => format!("every {every_seconds} s"),
        }
    }
}

/// Which change of a check's result alerts its owner. See
/// [`MonitorTrigger::Check`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CheckChange {
    /// It starts succeeding (its command exits with code 0).
    Success,
    /// It starts failing.
    Failure,
    /// It starts succeeding or failing, which is the default.
    #[default]
    Change,
    /// Its output changes.
    Output,
}

impl CheckChange {
    /// The change named as `monitor_start` takes it (`success`, `failure`,
    /// `change` or `output`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "success" => Some(Self::Success),
            "failure" => Some(Self::Failure),
            "change" => Some(Self::Change),
            "output" => Some(Self::Output),
            _ => None,
        }
    }

    /// `when it starts succeeding`, as the start's result says it.
    fn alerts_you(self) -> &'static str {
        match self {
            Self::Success => "when it starts succeeding",
            Self::Failure => "when it starts failing",
            Self::Change => "when it starts succeeding or failing",
            Self::Output => "when its output changes",
        }
    }
}

/// How long a monitor lasts, besides its time
/// ([`MonitorOptions::expires_in`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MonitorLifetime {
    /// Its time, from its start or latest rearm. The default.
    #[default]
    Timed,
    /// Its time, and longer while any of its owner's workers run, as the
    /// host's [`WorkerLiveness`] says. Once its time has passed it ends when
    /// the last worker finishes. While workers run it gives no warning
    /// before its end. With no worker running it ends at its time, as a
    /// timed monitor does. A host that adds no `WorkerLiveness` has no
    /// workers, so the monitor ends at its time.
    WhileWorkersRun,
}

impl MonitorLifetime {
    fn is_timed(&self) -> bool {
        *self == Self::Timed
    }
}

/// Says whether an agent's workers (subagents) run, for the monitors that
/// last while they do ([`MonitorLifetime::WhileWorkersRun`]). A host that
/// runs workers adds one with [`TerminalManager::add_worker_liveness`] and
/// keeps it while its workers can run. Each arko_workflows workflow with
/// subagents adds its own to the manager given to its `background_jobs`.
pub trait WorkerLiveness: Send + Sync {
    /// Whether any worker that `owner` (a task) started runs now, or waits
    /// to start. It is asked often, so it should answer quickly, and never
    /// call back into the terminal manager.
    fn has_running_workers(&self, owner: Uuid) -> bool;

    /// What changes whenever a worker starts or ends, so monitors don't
    /// wait to notice. A host that drops its `WorkerLiveness` changes it or
    /// closes it, so its monitors notice it no longer counts. Monitors also
    /// ask again each [`WORKERS_RECHECK`] while workers keep them.
    fn changes(&self) -> watch::Receiver<u64>;
}

/// How often a monitor kept by running workers asks again whether they run,
/// in case the [`WorkerLiveness`] misses a change (1 minute).
pub const WORKERS_RECHECK: Duration = Duration::from_secs(60);

/// What a monitor watches and for how long, for
/// [`TerminalManager::start_monitor`]. Made with [`lines`](Self::lines) or
/// [`check`](Self::check).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MonitorOptions {
    pub trigger: MonitorTrigger,
    /// How long the monitor lasts, at most [`MonitorLimits::longest`]. 30
    /// minutes by default.
    pub expires_in: Duration,
    /// What lists show instead of the command (default `None`, the
    /// command).
    pub label: Option<String>,
    /// The monitor ends after its first alert, stopping its command
    /// (default `false`). For an event that happens once, such as a CI run
    /// that finishes or a server that starts to answer. A check monitor
    /// whose first check already gives the result it waits for
    /// ([`CheckChange::Success`] or [`CheckChange::Failure`]) ends at once.
    pub once: bool,
    /// Whether it also lasts while its owner's workers run (default
    /// [`MonitorLifetime::Timed`]).
    pub lifetime: MonitorLifetime,
}

impl MonitorOptions {
    /// A monitor of the command's output lines that match `pattern`, a
    /// regular expression, or of every line with `None`.
    pub fn lines(pattern: Option<String>) -> Self {
        Self {
            trigger: MonitorTrigger::Lines { pattern },
            expires_in: MonitorLimits::default().longest,
            label: None,
            once: false,
            lifetime: MonitorLifetime::Timed,
        }
    }

    /// A monitor that runs the command as a check every `every_seconds`
    /// and alerts when its result changes as `when` says.
    pub fn check(every_seconds: u64, when: CheckChange) -> Self {
        Self {
            trigger: MonitorTrigger::Check {
                every_seconds,
                when,
            },
            ..Self::lines(None)
        }
    }

    /// These options with the monitor lasting `expires_in`.
    pub fn with_expiry(mut self, expires_in: Duration) -> Self {
        self.expires_in = expires_in;
        self
    }

    /// These options with `label` shown in lists instead of the command.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// These options with the monitor ending after its first alert. See
    /// [`once`](Self::once).
    pub fn once(mut self) -> Self {
        self.once = true;
        self
    }

    /// These options with the monitor lasting as `lifetime` says.
    pub fn with_lifetime(mut self, lifetime: MonitorLifetime) -> Self {
        self.lifetime = lifetime;
        self
    }
}

/// A monitor, as [`JobKind::Monitor`] describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MonitorInfo {
    pub trigger: MonitorTrigger,
    /// When it ends, as last armed. A monitor that lasts while workers run
    /// goes on past it while they do.
    pub expires_at: SystemTime,
    /// How many of its owner's notes carried its alerts.
    pub alerts: u32,
    /// It ends after its first alert ([`MonitorOptions::once`]).
    pub once: bool,
    /// Whether it also lasts while its owner's workers run.
    pub lifetime: MonitorLifetime,
}

/// News of a monitor that went to no runtime, because none was bound for
/// its owner. A host can start a turn of its idle agent for it, and the
/// runtime it binds gets the news. See
/// [`TerminalManager::subscribe_alerts`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct JobAlert {
    pub owner: Uuid,
    /// The monitor whose news it is. With several, the first.
    pub job_id: Uuid,
    pub kind: AlertKind,
}

/// What a [`JobAlert`] is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlertKind {
    /// Output lines matched.
    Lines,
    /// A check's result changed.
    Check,
    /// The monitor ends soon, and its owner may keep it with a rearm.
    EndsSoon,
}

/// The limits of monitors, set with
/// [`TerminalManager::set_monitor_limits`]. `Default` gives the values
/// below.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MonitorLimits {
    /// The running monitors one owner may have (5).
    pub per_owner: usize,
    /// The running monitors of all owners (20).
    pub in_all: usize,
    /// The longest a monitor lasts from its start or its latest rearm (30
    /// minutes).
    pub longest: Duration,
    /// How long before its end a monitor warns its owner (5 minutes).
    pub warning_before: Duration,
    /// A monitor armed for less than this ends without a warning (10
    /// minutes).
    pub warned_from: Duration,
    /// How long before an idle owner's prompt cache expires its early
    /// warning comes (1 minute). See
    /// [`TerminalManager::set_prompt_cache_expiry`].
    pub cache_margin: Duration,
    /// The least time between two of an owner's notes of monitor news, or
    /// two alerts to wake it (1 minute).
    pub alert_interval: Duration,
    /// How long news waits for more once the outbox has been quiet for
    /// `alert_interval` (2 seconds).
    pub gather: Duration,
    /// The lines of one monitor's news a note shows (20).
    pub lines_per_note: usize,
    /// The bytes of those lines (2,000).
    pub bytes_per_note: usize,
    /// The bytes of a line kept, the rest cut (500).
    pub line_bytes: usize,
    /// How long output can be quiet before a line without its end counts
    /// as a line (2 seconds).
    pub partial_line_after: Duration,
    /// How long a line monitor's start waits for a command that exits at
    /// once (1 second).
    pub start_wait: Duration,
    /// The shortest interval of a check (10 seconds).
    pub shortest_check: Duration,
    /// The longest interval of a check (600 seconds).
    pub longest_check: Duration,
    /// The longest one run of a check may take (60 seconds). A run also
    /// ends at its interval. One that runs over counts as a failure.
    pub check_run_limit: Duration,
    /// The bytes of a check's output compared and kept per run (4,096).
    pub check_output_bytes: usize,
    /// How long a check's start waits for its first result (10 seconds).
    pub first_check_wait: Duration,
    /// The longest pattern, in characters (1,000).
    pub pattern_chars: usize,
    /// The matching lines within `noisy_window` past which an alert
    /// suggests a narrower pattern (200).
    pub noisy_lines: usize,
    /// See `noisy_lines` (5 minutes).
    pub noisy_window: Duration,
}

impl Default for MonitorLimits {
    fn default() -> Self {
        Self {
            per_owner: 5,
            in_all: 20,
            longest: Duration::from_secs(30 * 60),
            warning_before: Duration::from_secs(5 * 60),
            warned_from: Duration::from_secs(10 * 60),
            cache_margin: Duration::from_secs(60),
            alert_interval: Duration::from_secs(60),
            gather: Duration::from_secs(2),
            lines_per_note: 20,
            bytes_per_note: 2_000,
            line_bytes: 500,
            partial_line_after: Duration::from_secs(2),
            start_wait: Duration::from_secs(1),
            shortest_check: Duration::from_secs(10),
            longest_check: Duration::from_secs(600),
            check_run_limit: Duration::from_secs(60),
            check_output_bytes: 4_096,
            first_check_wait: Duration::from_secs(10),
            pattern_chars: 1_000,
            noisy_lines: 200,
            noisy_window: Duration::from_secs(5 * 60),
        }
    }
}

impl MonitorLimits {
    fn check(&self) -> Result<()> {
        let durations = [
            self.longest,
            self.alert_interval,
            self.partial_line_after,
            self.shortest_check,
            self.longest_check,
            self.check_run_limit,
            self.noisy_window,
        ];
        if self.per_owner == 0
            || self.in_all == 0
            || durations.iter().any(|duration| {
                duration.is_zero() || std::time::Instant::now().checked_add(*duration).is_none()
            })
            || self.shortest_check > self.longest_check
            || self.lines_per_note == 0
            || self.bytes_per_note == 0
            || self.line_bytes == 0
            || self.check_output_bytes == 0
            || self.pattern_chars == 0
            || [
                self.warning_before,
                self.warned_from,
                self.cache_margin,
                self.gather,
                self.start_wait,
                self.first_check_wait,
            ]
            .iter()
            .any(|duration| std::time::Instant::now().checked_add(*duration).is_none())
        {
            return Err(TerminalError::Invalid("invalid monitor limits".into()));
        }
        Ok(())
    }
}

/// A monitor that started, as [`TerminalManager::start_monitor`] returns
/// it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MonitorStarted {
    pub job_id: Uuid,
    pub expires_at: SystemTime,
    /// What `monitor_start` returns. It says what the monitor watches until
    /// when, with its output so far, or the result of its first check.
    pub summary: String,
}

/// A monitor's saved state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MonitorRecord {
    pub(super) trigger: MonitorTrigger,
    pub(super) expires_at: SystemTime,
    #[serde(default)]
    pub(super) alerts: u32,
    /// Its warning for this arm went.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) warned: bool,
    /// It ends after its first alert.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) once: bool,
    #[serde(default, skip_serializing_if = "MonitorLifetime::is_timed")]
    pub(super) lifetime: MonitorLifetime,
}

/// A monitor's saved state, or `None` for one this version can't read, such
/// as a trigger kind a later version added, so that one record doesn't keep
/// the whole jobs directory from opening.
pub(super) fn readable_record<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<MonitorRecord>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| match serde_json::from_value(value) {
        Ok(record) => Some(record),
        Err(error) => {
            log::warn!("A monitor's record this version can't read is kept as a job's ({error})");
            None
        }
    }))
}

/// A running monitor's timing, kept with its job's data.
#[derive(Debug, Clone)]
pub(super) struct Arming {
    pub(super) expires: Instant,
    /// How long it was armed for, at its start or latest rearm.
    pub(super) lasts: Duration,
    /// Its warning for this arm went.
    pub(super) warned: bool,
    /// It is ending (it expired, or it was stopped), so it can't be
    /// rearmed.
    pub(super) ending: bool,
    /// Its time passed while its owner's workers ran, which kept it.
    pub(super) held: bool,
}

/// How a monitor ended by itself, other than by its command's exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Done {
    /// It lasted as long as it was armed for.
    Expired,
    /// Its time had passed, and the last of its owner's workers finished.
    WorkersFinished,
    /// It alerted once, which was all it was for.
    Alerted,
}

impl Done {
    /// The state its job ends in. A monitor that did what it was for
    /// succeeded.
    pub(super) fn state(self) -> JobState {
        match self {
            Self::Expired | Self::WorkersFinished => JobState::TimedOut,
            Self::Alerted => JobState::Succeeded,
        }
    }

    /// What its record says of the end, as its error.
    pub(super) fn error(self) -> Option<String> {
        match self {
            Self::Expired => Some("expired".into()),
            Self::WorkersFinished => Some("ended after its agent's workers finished".into()),
            Self::Alerted => None,
        }
    }
}

/// What a running monitor is, for its actor.
pub(super) struct Running {
    pub(super) header: Header,
    pub(super) owner: Uuid,
    pub(super) limits: MonitorLimits,
    /// It ends after its first alert.
    pub(super) once: bool,
    pub(super) lifetime: MonitorLifetime,
}

impl Running {
    /// When its next timed event is due, its warning or its end. While its
    /// owner's workers keep it, that is its time or the next look at the
    /// workers.
    pub(super) fn next_event(
        &self,
        manager: &Weak<TerminalManager>,
        data: &Mutex<JobData>,
    ) -> Option<Instant> {
        let inputs = self.wake_inputs(manager);
        // Asked before the job's lock is taken.
        let kept = self.kept_by_workers(manager);
        let data = data.lock().unwrap();
        let arming = data.monitor.as_ref()?;
        if kept {
            let recheck = Instant::now() + WORKERS_RECHECK;
            return Some(if arming.held {
                recheck
            } else {
                arming.expires.min(recheck)
            });
        }
        Some(
            arming
                .warning(&self.limits, inputs)
                .map_or(arming.expires, |warning| warning.min(arming.expires)),
        )
    }

    /// Whether its owner's workers keep it: it lasts while they run, and
    /// one does.
    fn kept_by_workers(&self, manager: &Weak<TerminalManager>) -> bool {
        self.lifetime == MonitorLifetime::WhileWorkersRun
            && manager
                .upgrade()
                .is_some_and(|manager| manager.workers_running(self.owner))
    }

    fn wake_inputs(&self, manager: &Weak<TerminalManager>) -> warnings::WakeInputs {
        manager
            .upgrade()
            .map(|manager| manager.wake_inputs(self.owner))
            .unwrap_or_default()
    }

    /// Runs its timed event if it's due, which warns its owner that it ends
    /// soon or ends it. Returns how the monitor ended, if it did, in which
    /// case it can no longer be rearmed. One whose owner's workers keep it
    /// neither warns nor ends.
    pub(super) fn timed_event(
        &self,
        manager: &Weak<TerminalManager>,
        data: &Mutex<JobData>,
        config: &TerminalConfig,
    ) -> Option<Done> {
        let inputs = self.wake_inputs(manager);
        let kept = self.kept_by_workers(manager);
        let now = Instant::now();
        let ends_at = {
            let mut data = data.lock().unwrap();
            let arming = data.monitor.as_mut()?;
            if kept {
                arming.held |= now >= arming.expires;
                return None;
            }
            if now >= arming.expires {
                arming.ending = true;
                return Some(if arming.held {
                    Done::WorkersFinished
                } else {
                    Done::Expired
                });
            }
            if arming
                .warning(&self.limits, inputs)
                .is_none_or(|warning| now < warning)
            {
                return None;
            }
            arming.warned = true;
            let ends_at = SystemTime::now() + (arming.expires - now);
            if let Some(monitor) = data.record.monitor.as_mut() {
                monitor.warned = true;
            }
            if let Err(error) = persist(config, &data.record) {
                log::warn!(
                    "Terminal job {}: could not save its warning: {error}",
                    self.header.job_id
                );
            }
            ends_at
        };
        self.tell(
            manager,
            News::EndsSoon {
                ends_at,
                longest: self.limits.longest,
            },
        );
        None
    }

    /// Marks the monitor as ending, so it can't be rearmed.
    pub(super) fn ending(&self, data: &Mutex<JobData>) {
        if let Some(arming) = data.lock().unwrap().monitor.as_mut() {
            arming.ending = true;
        }
    }

    /// Queues `news` for the owner.
    pub(super) fn tell(&self, manager: &Weak<TerminalManager>, news: News) {
        if let Some(manager) = manager.upgrade() {
            manager.queue_news(self.owner, self.header.clone(), news);
        }
    }
}

/// How a monitor ended, for its last news.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum EndReason {
    /// It lasted as long as it was armed for.
    Expired,
    /// Its time had passed, and the last of its owner's workers finished.
    WorkersFinished,
    /// It alerted once, as it was asked to.
    Alerted,
    /// It was to alert once, and its first check already gave the result it
    /// waits for (`success` or a failure), which its start didn't report.
    AtOnce { success: bool },
    /// Its command exited on its own.
    Exited { code: Option<i32> },
    /// Its owner stopped it.
    StoppedByOwner,
    /// The host's user stopped it.
    StoppedByUser,
    /// Its owner's run ended.
    OwnerEnded,
    /// It failed (its record couldn't be saved, its output couldn't be
    /// read).
    Failed { error: String },
}

/// How a monitor that ended in `state` ended, `done` by itself or
/// `stopped_by` whom, or `None` when its host's exit stopped it, which tells
/// no one.
pub(super) fn end_reason(
    state: JobState,
    done: Option<Done>,
    stopped_by: Option<calls::StopCause>,
    exit_code: Option<i32>,
    error: Option<&str>,
) -> Option<EndReason> {
    Some(match (state, done) {
        (JobState::TimedOut, Some(Done::Expired)) => EndReason::Expired,
        (JobState::TimedOut, Some(Done::WorkersFinished)) => EndReason::WorkersFinished,
        (JobState::Succeeded, Some(Done::Alerted)) => EndReason::Alerted,
        (JobState::Cancelled, _) => match stopped_by {
            Some(calls::StopCause::Owner) => EndReason::StoppedByOwner,
            Some(calls::StopCause::User) => EndReason::StoppedByUser,
            Some(calls::StopCause::OwnerEnded) => EndReason::OwnerEnded,
            None => return None,
        },
        (JobState::Succeeded, _) => EndReason::Exited { code: exit_code },
        (JobState::Failed, _) if error.is_none() => EndReason::Exited { code: exit_code },
        _ => EndReason::Failed {
            error: error.unwrap_or("it stopped unexpectedly").to_string(),
        },
    })
}

/// A line of a monitor's output, as matching reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Line {
    /// Its text without escape sequences or other control characters, cut
    /// to the limit with `…`.
    pub(super) text: String,
    /// Where it starts in the job's output.
    pub(super) start: u64,
}

/// Splits a job's output into lines. Lines end at `\n` or `\r`, so each
/// step of a progress bar is one. Blank lines are left out.
#[derive(Debug)]
pub(super) struct LineSplitter {
    line_bytes: usize,
    /// The raw bytes of the line so far, up to four times `line_bytes`, so
    /// escape sequences are whole before they are taken out.
    current: Vec<u8>,
    /// The line had more bytes than `current` keeps.
    overflow: bool,
    /// Where the line starts in the output.
    start: u64,
    /// Where the output read so far ends.
    end: u64,
    /// When the line so far last grew, while it has no end.
    last: Option<Instant>,
}

impl LineSplitter {
    pub(super) fn new(line_bytes: usize) -> Self {
        Self {
            line_bytes,
            current: Vec::new(),
            overflow: false,
            start: 0,
            end: 0,
            last: None,
        }
    }

    /// The lines that `bytes`, the output at `offset`, ends.
    pub(super) fn push(&mut self, bytes: &[u8], offset: u64, now: Instant) -> Vec<Line> {
        let mut lines = Vec::new();
        let mut from = 0;
        for (index, byte) in bytes.iter().enumerate() {
            if matches!(byte, b'\n' | b'\r') {
                self.keep(&bytes[from..index]);
                lines.extend(self.end_line());
                from = index + 1;
                self.start = offset + from as u64;
            }
        }
        self.keep(&bytes[from..]);
        self.end = offset + bytes.len() as u64;
        self.last = (!self.current.is_empty() || self.overflow).then_some(now);
        lines
    }

    fn keep(&mut self, bytes: &[u8]) {
        let room = (self.line_bytes * 4).saturating_sub(self.current.len());
        self.current
            .extend_from_slice(&bytes[..bytes.len().min(room)]);
        if bytes.len() > room {
            self.overflow = true;
        }
    }

    fn end_line(&mut self) -> Option<Line> {
        let raw = std::mem::take(&mut self.current);
        let overflow = std::mem::take(&mut self.overflow);
        self.last = None;
        line_text(&raw, overflow, self.line_bytes).map(|text| Line {
            text,
            start: self.start,
        })
    }

    /// When the line so far counts as a line without its end, once output
    /// has been quiet for `after`.
    pub(super) fn partial_due(&self, after: Duration) -> Option<Instant> {
        self.last.map(|last| last + after)
    }

    /// The line so far, as a line. What comes next starts a new one.
    pub(super) fn flush(&mut self) -> Option<Line> {
        let line = self.end_line();
        self.start = self.end;
        line
    }
}

/// The text of a line's raw bytes, or `None` for a blank line.
fn line_text(raw: &[u8], overflow: bool, line_bytes: usize) -> Option<String> {
    let text = detection::strip_escapes(&String::from_utf8_lossy(raw));
    let text: String = text
        .chars()
        .filter(|character| !character.is_control() || *character == '\t')
        .collect();
    let text = text.trim_end();
    if text.trim_start().is_empty() {
        return None;
    }
    let mut cut = text.len().min(line_bytes);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut line = text[..cut].to_string();
    if overflow || cut < text.len() {
        line.push('…');
    }
    Some(line)
}

/// Compiles a monitor's pattern, with a size limit so a pattern can't take
/// much memory. The `regex` crate matches in linear time.
pub(super) fn compile(pattern: &str) -> std::result::Result<regex::Regex, String> {
    regex::RegexBuilder::new(pattern)
        .size_limit(1 << 20)
        .dfa_size_limit(1 << 20)
        .build()
        .map_err(|error| error.to_string())
}

/// Which lines alert, and whether they come too often.
#[derive(Debug)]
pub(super) struct Matcher {
    regex: Option<regex::Regex>,
    /// When the latest matching lines came, at most `noisy_lines + 1`.
    recent: VecDeque<Instant>,
    noisy_lines: usize,
    noisy_window: Duration,
}

impl Matcher {
    pub(super) fn new(regex: Option<regex::Regex>, limits: &MonitorLimits) -> Self {
        Self {
            regex,
            recent: VecDeque::new(),
            noisy_lines: limits.noisy_lines,
            noisy_window: limits.noisy_window,
        }
    }

    /// Whether `line` alerts.
    pub(super) fn matches(&mut self, line: &str, now: Instant) -> bool {
        let matched = self.regex.as_ref().is_none_or(|regex| regex.is_match(line));
        if matched {
            self.recent.push_back(now);
            if self.recent.len() > self.noisy_lines.saturating_add(1) {
                self.recent.pop_front();
            }
        }
        matched
    }

    /// Whether more than `noisy_lines` matched within `noisy_window`.
    pub(super) fn is_noisy(&self, now: Instant) -> bool {
        self.recent.len() > self.noisy_lines
            && self
                .recent
                .front()
                .is_some_and(|first| now.saturating_duration_since(*first) <= self.noisy_window)
    }
}

/// Watches a line monitor's output, in its job's actor.
pub(super) struct LineWatch {
    pub(super) running: Running,
    splitter: LineSplitter,
    matcher: Matcher,
    /// A monitor that alerts once has alerted, and takes no more lines.
    alerted: bool,
}

impl LineWatch {
    pub(super) fn new(running: Running, regex: Option<regex::Regex>) -> Self {
        Self {
            splitter: LineSplitter::new(running.limits.line_bytes),
            matcher: Matcher::new(regex, &running.limits),
            running,
            alerted: false,
        }
    }

    /// Whether the monitor alerted once and is done.
    pub(super) fn is_done(&self) -> bool {
        self.alerted
    }

    /// Reads `bytes`, the output at `offset`, and tells the owner of the
    /// lines that match.
    pub(super) fn output(&mut self, manager: &Weak<TerminalManager>, bytes: &[u8], offset: u64) {
        let now = Instant::now();
        let lines = self.splitter.push(bytes, offset, now);
        self.matched(manager, lines, now);
    }

    /// When a line without its end counts as a line.
    pub(super) fn partial_due(&self) -> Option<Instant> {
        self.splitter
            .partial_due(self.running.limits.partial_line_after)
    }

    /// Counts the line so far as a line.
    pub(super) fn flush(&mut self, manager: &Weak<TerminalManager>) {
        let now = Instant::now();
        let lines = self.splitter.flush().into_iter().collect();
        self.matched(manager, lines, now);
    }

    fn matched(&mut self, manager: &Weak<TerminalManager>, lines: Vec<Line>, now: Instant) {
        if self.alerted {
            return;
        }
        let matched: Vec<Line> = lines
            .into_iter()
            .filter(|line| self.matcher.matches(&line.text, now))
            .collect();
        if !matched.is_empty() {
            let noisy = self.matcher.is_noisy(now);
            self.running.tell(
                manager,
                News::Lines {
                    lines: matched,
                    noisy,
                    at: SystemTime::now(),
                },
            );
            self.alerted = self.running.once;
        }
    }
}

/// `17:42 UTC`.
pub(super) fn clock(at: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(at)
        .format("%H:%M UTC")
        .to_string()
}

/// `17:41:05 UTC`.
pub(super) fn clock_seconds(at: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(at)
        .format("%H:%M:%S UTC")
        .to_string()
}

/// `22 min`, rounded to the nearest minute, or `less than a minute`.
pub(super) fn minutes_left(left: Duration) -> String {
    let seconds = left.as_secs();
    if seconds < 60 {
        "less than a minute".to_string()
    } else {
        format!("{} min", (seconds + 30) / 60)
    }
}

/// A line monitor about to start, for [`TerminalManager::start_job`].
pub(super) struct Watching {
    pub(super) record: MonitorRecord,
    pub(super) arming: Arming,
    regex: Option<regex::Regex>,
    pub(super) limits: MonitorLimits,
}

impl Watching {
    /// A monitor of `options`, armed from now, with its pattern compiled.
    pub(super) fn new(
        options: &MonitorOptions,
        regex: Option<regex::Regex>,
        limits: &MonitorLimits,
    ) -> Self {
        Self {
            record: MonitorRecord {
                trigger: options.trigger.clone(),
                expires_at: SystemTime::now() + options.expires_in,
                alerts: 0,
                warned: false,
                once: options.once,
                lifetime: options.lifetime,
            },
            arming: Arming {
                expires: Instant::now() + options.expires_in,
                lasts: options.expires_in,
                warned: false,
                ending: false,
                held: false,
            },
            regex,
            limits: limits.clone(),
        }
    }

    /// The monitor of job `id`, which `owner` started.
    pub(super) fn running(&self, id: Uuid, command: String, owner: Uuid) -> Running {
        Running {
            header: Header {
                job_id: id,
                command,
                trigger: self.record.trigger.clone(),
            },
            owner,
            limits: self.limits.clone(),
            once: self.record.once,
            lifetime: self.record.lifetime,
        }
    }

    /// What watches job `id`'s output, which `owner` started.
    pub(super) fn watch(self, id: Uuid, command: String, owner: Uuid) -> LineWatch {
        let running = self.running(id, command, owner);
        LineWatch::new(running, self.regex)
    }
}

/// Refuses another monitor when `owner`, or all owners, have as many
/// running as `limits` allow.
pub(super) fn check_room(
    jobs: &HashMap<Uuid, Arc<Job>>,
    owner: Uuid,
    limits: &MonitorLimits,
) -> Result<()> {
    let (mut owners, mut all) = (0, 0);
    for job in jobs.values() {
        let data = job.data.lock().unwrap();
        if data.record.monitor.is_some() && data.record.state == JobState::Running {
            all += 1;
            if data.record.owner == owner {
                owners += 1;
            }
        }
    }
    if owners >= limits.per_owner {
        return Err(TerminalError::Invalid(format!(
            "you have {owners} monitors running, the most an agent may have. Stop one with monitor_control stop, or let one end"
        )));
    }
    if all >= limits.in_all {
        return Err(TerminalError::Invalid(format!(
            "{all} monitors are running in this session, the most it may have. Stop one of yours with monitor_control stop, or let one end"
        )));
    }
    Ok(())
}

/// The last lines of a job's output, for the note of a command that
/// exited, at most 10 lines and 1,000 bytes.
pub(super) fn last_lines(output: &VecDeque<u8>, line_bytes: usize) -> Vec<String> {
    const LINES: usize = 10;
    const BYTES: usize = 1_000;
    let tail: Vec<u8> = output
        .iter()
        .skip(output.len().saturating_sub(BYTES * 4))
        .copied()
        .collect();
    let mut splitter = LineSplitter::new(line_bytes);
    let now = Instant::now();
    let mut lines: Vec<String> = splitter
        .push(&tail, 0, now)
        .into_iter()
        .chain(splitter.flush())
        .map(|line| line.text)
        .collect();
    let mut bytes = 0;
    let mut keep = 0;
    for line in lines.iter().rev().take(LINES) {
        if bytes + line.len() > BYTES && keep > 0 {
            break;
        }
        bytes += line.len();
        keep += 1;
    }
    lines.drain(..lines.len() - keep);
    lines
}

impl TerminalManager {
    /// Sets the limits of monitors, which apply to monitors that start from
    /// now on. `TerminalError::Invalid` for limits that can't work (none
    /// allowed, zero intervals, times too long to count).
    pub fn set_monitor_limits(&self, limits: MonitorLimits) -> Result<()> {
        limits.check()?;
        *self.monitor_limits.lock().unwrap() = limits;
        Ok(())
    }

    /// The limits of monitors, as last set.
    pub fn monitor_limits(&self) -> MonitorLimits {
        self.monitor_limits.lock().unwrap().clone()
    }

    /// Adds what says whether some agents' workers run, for the monitors
    /// that last while they do ([`MonitorLifetime::WhileWorkersRun`]). The
    /// manager holds it weakly, so it counts until its host drops it. Each
    /// workflow that shares the manager adds its own, and a monitor is kept
    /// while any of them says its owner has a worker running.
    /// `monitor_start` offers its `lasts` parameter while one counts. Running
    /// monitors plan their timed events again.
    pub fn add_worker_liveness(&self, liveness: Weak<dyn WorkerLiveness>) {
        {
            let mut all = self.worker_liveness.lock().unwrap();
            all.retain(|liveness| liveness.strong_count() > 0);
            all.push(liveness);
        }
        self.monitor_timing_changed();
    }

    /// The [`WorkerLiveness`]es that count, those whose hosts still hold
    /// them.
    fn live_worker_liveness(&self) -> Vec<Arc<dyn WorkerLiveness>> {
        let mut all = self.worker_liveness.lock().unwrap();
        all.retain(|liveness| liveness.strong_count() > 0);
        all.iter().filter_map(Weak::upgrade).collect()
    }

    /// Whether a [`WorkerLiveness`] counts, so monitors can last while
    /// workers run.
    pub(super) fn has_worker_liveness(&self) -> bool {
        !self.live_worker_liveness().is_empty()
    }

    /// Whether `owner` has workers running, as any [`WorkerLiveness`] says.
    /// Never asked with a job's lock held, as it may take the hosts' own
    /// locks.
    pub(super) fn workers_running(&self, owner: Uuid) -> bool {
        self.live_worker_liveness()
            .iter()
            .any(|liveness| liveness.has_running_workers(owner))
    }

    /// What changes as workers start and end, one for each
    /// [`WorkerLiveness`] that counts.
    pub(super) fn worker_changes(&self) -> Vec<watch::Receiver<u64>> {
        self.live_worker_liveness()
            .iter()
            .map(|liveness| liveness.changes())
            .collect()
    }

    /// The news of monitors that went to no runtime from now on, because
    /// none was bound for its owner, when it should wake the owner. A host
    /// can start a turn of the owner's idle agent for it, and the runtime it
    /// binds gets the news. At most one alert goes to an owner each
    /// [`alert_interval`](MonitorLimits::alert_interval), whatever number
    /// of monitors it has. A monitor that ends because its command exited
    /// is reported to [`subscribe_finished`](Self::subscribe_finished)
    /// instead. The other ends (it expired, its owner or the user stopped
    /// it) wake no one, and their news goes to the owner's next runtime.
    pub fn subscribe_alerts(&self) -> tokio::sync::broadcast::Receiver<JobAlert> {
        self.alerts.subscribe()
    }

    /// Starts a monitor of `command`, run in the context's working
    /// directory for the context's task, its owner. Its alerts reach the
    /// owner's runtime as notes, gathered in the owner's outbox, which sends
    /// one note at most each
    /// [`alert_interval`](MonitorLimits::alert_interval). The monitor is a
    /// job, so `terminal_status` reads its output (a check's runs, each
    /// with its time and how it ended), and it is listed with
    /// [`JobKind::Monitor`]. It ends when `options.expires_in` has passed,
    /// stopping its command, or when the command exits.
    ///
    /// A monitor of lines runs its command in a terminal (a PTY), so
    /// programs print each line as it comes. Its start waits up to
    /// [`start_wait`](MonitorLimits::start_wait) for a command that exits at
    /// once. A check runs its command with pipes and closed input, never
    /// two runs at once, each stopped (with the processes it started) after
    /// its interval or [`check_run_limit`](MonitorLimits::check_run_limit),
    /// whichever is shorter. Its start runs the first check and waits up to
    /// [`first_check_wait`](MonitorLimits::first_check_wait) for it, so the
    /// result says where changes count from. A first check that can't run
    /// at all is an error. The command guard checks the command once, and
    /// the [limits](Self::set_monitor_limits) bound the options and the
    /// number of monitors.
    pub async fn start_monitor(
        self: &Arc<Self>,
        context: &TaskContext<'_>,
        command: &str,
        options: MonitorOptions,
    ) -> Result<MonitorStarted> {
        let limits = self.monitor_limits();
        let regex = validate(
            command,
            &options,
            &limits,
            ShellDialect::from_shell(&self.config.shell),
        )?;
        let label = options.label.clone().unwrap_or_else(|| command.to_string());
        let watching = Watching::new(&options, regex, &limits);
        let expires_at = watching.record.expires_at;
        if let MonitorTrigger::Check { .. } = options.trigger {
            self.check_command(command, context.current_working_directory)?;
            let config = self.config.clone();
            let (command, cwd) = (
                command.to_string(),
                context.current_working_directory.to_path_buf(),
            );
            let keep = limits.check_output_bytes;
            let (id, first) = self
                .start_check(context, label.clone(), watching, move |directory| {
                    checks::command_checker(config, command, cwd, directory, keep)
                })
                .await?;
            if let (true, MonitorTrigger::Check { when, .. }, Some(run)) =
                (options.once, &options.trigger, &first)
                && checks::awaited(*when, run.success)
            {
                let already = if run.success { "succeeds" } else { "fails" };
                return Ok(MonitorStarted {
                    job_id: id,
                    expires_at,
                    summary: format!(
                        "Monitor {id} ended at once, because its check `{label}` already {already}. {}",
                        run.first_sentence(&limits)
                    ),
                });
            }
            let first = match first {
                Some(run) => run.first_sentence(&limits),
                // Stopped (by the user, or the host's exit) before its first
                // check ended.
                None if self
                    .status(context.task_id, id, None)
                    .is_ok_and(|snapshot| snapshot.state.is_terminal()) =>
                {
                    "It was stopped before its first check ended.".to_string()
                }
                None => {
                    "Its first check is still running, and changes are judged against its result."
                        .to_string()
                }
            };
            return Ok(MonitorStarted {
                job_id: id,
                expires_at,
                summary: format!(
                    "{} {first}",
                    started_sentence(id, &label, &options, expires_at)
                ),
            });
        }
        let started = self.start_job(
            context,
            command,
            StartOptions {
                terminal: true,
                detect_prompts: false,
                label: Some(label.clone()),
                ..StartOptions::default()
            },
            Some(watching),
        )?;
        let id = started.job_id;
        let done = self
            .jobs
            .lock()
            .unwrap()
            .get(&id)
            .map(|job| job.done.clone());
        if let Some(mut done) = done {
            let _ = tokio::time::timeout(limits.start_wait, done.wait_for(|done| *done)).await;
        }
        let snapshot = self.status(context.task_id, id, None)?;
        // Killed after its first alert, not exited.
        let alerted =
            options.once && snapshot.state == JobState::Succeeded && snapshot.exit_code.is_none();
        let summary = if alerted {
            // The start says it alerted and ended, so its note would only
            // repeat it.
            self.drop_news(context.task_id, id);
            format!(
                "Monitor {id} ended after its first alert, which came at once. {}",
                output_so_far(&snapshot, "Its output follows.", "It printed nothing.")
            )
        } else if snapshot.state.is_terminal() {
            // The start says it ended, so its note would only repeat it.
            self.drop_news(context.task_id, id);
            let how = match snapshot.exit_code {
                Some(code) => format!("exited with code {code}"),
                None => "ended without an exit code".to_string(),
            };
            format!(
                "Monitor {id} ended at once, because its command `{label}` {how}. {}",
                output_so_far(&snapshot, "Its output follows.", "It printed nothing.")
            )
        } else {
            format!(
                "{} {}",
                started_sentence(id, &label, &options, expires_at),
                output_so_far(
                    &snapshot,
                    "Its output so far follows.",
                    "It has printed nothing yet."
                )
            )
        };
        Ok(MonitorStarted {
            job_id: id,
            expires_at,
            summary,
        })
    }
}

/// `follows`, then a job's output, or `none` without any.
fn output_so_far(snapshot: &TerminalSnapshot, follows: &str, none: &str) -> String {
    let text = snapshot.output.text.trim_end();
    if text.is_empty() {
        none.to_string()
    } else {
        format!("{follows}\n{text}")
    }
}

/// Checks a monitor's command and options against `limits`, and compiles
/// its pattern.
pub(super) fn validate(
    command: &str,
    options: &MonitorOptions,
    limits: &MonitorLimits,
    dialect: ShellDialect,
) -> Result<Option<regex::Regex>> {
    if command.trim().is_empty() || command.len() > 65_536 {
        return Err(TerminalError::Invalid(
            "the command is empty or longer than 65,536 bytes".into(),
        ));
    }
    if dialect == ShellDialect::Cmd && command.contains(['\n', '\r']) {
        return Err(TerminalError::Invalid(
            "cmd.exe runs one line, so give the command on one line".into(),
        ));
    }
    if options.expires_in.is_zero() || options.expires_in > limits.longest {
        return Err(TerminalError::Invalid(format!(
            "a monitor's time must be more than zero and at most {}",
            calls::describe(limits.longest)
        )));
    }
    match &options.trigger {
        MonitorTrigger::Lines { pattern: None } => Ok(None),
        MonitorTrigger::Lines {
            pattern: Some(pattern),
        } => {
            let length = pattern.chars().count();
            if length == 0 || length > limits.pattern_chars {
                return Err(TerminalError::Invalid(format!(
                    "the pattern has {length} characters, and it may have 1 to {}",
                    limits.pattern_chars
                )));
            }
            compile(pattern).map(Some).map_err(|error| {
                TerminalError::Invalid(format!(
                    "the pattern isn't a regular expression this monitor can use ({error})"
                ))
            })
        }
        MonitorTrigger::Check { every_seconds, .. } => {
            let every = Duration::from_secs(*every_seconds);
            if every < limits.shortest_check || every > limits.longest_check {
                return Err(TerminalError::Invalid(format!(
                    "a check runs every {} to {} seconds",
                    limits.shortest_check.as_secs(),
                    limits.longest_check.as_secs()
                )));
            }
            Ok(None)
        }
    }
}

/// The first sentences of `monitor_start`'s result, up to what it says of
/// the output or first check.
pub(super) fn started_sentence(
    id: Uuid,
    command: &str,
    options: &MonitorOptions,
    expires_at: SystemTime,
) -> String {
    let watches = match &options.trigger {
        MonitorTrigger::Lines {
            pattern: Some(pattern),
        } => format!("and alerts you of lines that match `{pattern}`"),
        MonitorTrigger::Lines { pattern: None } => "and alerts you of every line".to_string(),
        MonitorTrigger::Check {
            every_seconds,
            when,
        } => format!(
            "every {every_seconds} s and alerts you {}",
            when.alerts_you()
        ),
    };
    let workers = match options.lifetime {
        MonitorLifetime::WhileWorkersRun => ", and longer while your workers run",
        MonitorLifetime::Timed => "",
    };
    let once = if options.once {
        " It ends after its first alert."
    } else {
        ""
    };
    format!(
        "Monitor {id} started. It runs `{command}` {watches}, until {} ({}){workers}.{once}",
        clock(expires_at),
        calls::describe(options.expires_in)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(lines: Vec<Line>) -> Vec<String> {
        lines.into_iter().map(|line| line.text).collect()
    }

    /// Lines end at `\n` or `\r`, so each step of a progress bar is a line,
    /// `\r\n` makes no blank line, and escape sequences and control
    /// characters are taken out before matching. Each line knows where it
    /// starts in the output.
    #[test]
    fn the_splitter_ends_lines_at_either_newline_and_strips_escapes() {
        let now = Instant::now();
        let mut splitter = LineSplitter::new(500);
        let lines = splitter.push(
            b"one\r\ntwo\n\x1b[31mred\x1b[0m\x07\n 10%\r 50%\rdone",
            0,
            now,
        );
        assert_eq!(
            lines
                .iter()
                .map(|line| (line.text.as_str(), line.start))
                .collect::<Vec<_>>(),
            [
                ("one", 0),
                ("two", 5),
                ("red", 9),
                (" 10%", 23),
                (" 50%", 28)
            ]
        );
        // The rest waits for its end, which may come in a later read.
        assert_eq!(
            texts(splitter.push(b"!\n\n", 37, now)),
            ["done!".to_string()]
        );
        assert_eq!(splitter.partial_due(Duration::from_secs(2)), None);
        assert_eq!(splitter.push(b"next", 40, now)[..], []);
        assert_eq!(splitter.start, 40);
    }

    /// A line without its end counts as a line once output has been quiet
    /// for a while, and what follows starts a new line.
    #[test]
    fn a_partial_line_counts_once_output_is_quiet() {
        let start = Instant::now();
        let mut splitter = LineSplitter::new(500);
        assert!(splitter.push(b"Password: ", 0, start).is_empty());
        let after = Duration::from_secs(2);
        assert_eq!(splitter.partial_due(after), Some(start + after));
        // More output puts it off.
        let later = start + Duration::from_secs(1);
        assert!(splitter.push(b"(again) ", 10, later).is_empty());
        assert_eq!(splitter.partial_due(after), Some(later + after));
        let line = splitter.flush().unwrap();
        assert_eq!((line.text.as_str(), line.start), ("Password: (again)", 0));
        assert_eq!(splitter.partial_due(after), None);
        let lines = splitter.push(b"ok\n", 18, later);
        assert_eq!(
            lines
                .iter()
                .map(|line| (line.text.as_str(), line.start))
                .collect::<Vec<_>>(),
            [("ok", 18)]
        );
        assert_eq!(splitter.flush(), None);
    }

    /// A long line is cut at the limit, on a character boundary, with `…`,
    /// and the splitter keeps only a few times the limit while it waits for
    /// the line's end.
    #[test]
    fn a_long_line_is_cut() {
        let now = Instant::now();
        let mut splitter = LineSplitter::new(10);
        let mut long = "é".repeat(30).into_bytes();
        long.push(b'\n');
        let lines = texts(splitter.push(&long, 0, now));
        assert_eq!(lines, [format!("{}…", "é".repeat(5))]);
        assert!(splitter.push(&[b'x'; 1000], 61, now).is_empty());
        assert_eq!(splitter.current.len(), 40);
        assert_eq!(
            texts(splitter.push(b"\n", 1061, now)),
            [format!("{}…", "x".repeat(10))]
        );
        assert_eq!(
            texts(splitter.push(b"0123456789\n", 1062, now)),
            ["0123456789"]
        );
    }

    /// Without a pattern every line matches. A model's pattern compiles
    /// with a size limit, and an invalid one says why.
    #[test]
    fn patterns_match_lines_and_count_noisy_runs() {
        let limits = MonitorLimits {
            noisy_lines: 3,
            noisy_window: Duration::from_secs(60),
            ..MonitorLimits::default()
        };
        let now = Instant::now();
        let mut every = Matcher::new(None, &limits);
        assert!(every.matches("anything", now));
        let mut errors = Matcher::new(Some(compile("(?i)error|warn").unwrap()), &limits);
        assert!(errors.matches("[ERROR] Failed to compile", now));
        assert!(!errors.matches("compiled in 2 s", now));
        assert!(!errors.is_noisy(now));
        for _ in 0..3 {
            errors.matches("warn: deprecated", now);
        }
        assert!(errors.is_noisy(now));
        assert!(!errors.is_noisy(now + Duration::from_secs(61)));
        // A host that turns the hint off.
        let mut never = Matcher::new(
            None,
            &MonitorLimits {
                noisy_lines: usize::MAX,
                ..MonitorLimits::default()
            },
        );
        assert!(never.matches("line", now) && !never.is_noisy(now));
        assert!(compile("(unclosed").unwrap_err().contains("unclosed"));
        assert!(compile("a{1000}{1000}").is_err());
    }

    /// The command, the expiry, the pattern and the check's interval are
    /// checked against the limits.
    #[test]
    fn options_are_checked_against_the_limits() {
        let limits = MonitorLimits::default();
        let check = |command: &str, options: MonitorOptions| {
            validate(command, &options, &limits, ShellDialect::Posix).map(|regex| regex.is_some())
        };
        assert!(!check("npm run dev", MonitorOptions::lines(None)).unwrap());
        assert!(check("npm run dev", MonitorOptions::lines(Some("error".into()))).unwrap());
        assert!(check(" ", MonitorOptions::lines(None)).is_err());
        assert!(check("x", MonitorOptions::lines(Some(String::new()))).is_err());
        assert!(check("x", MonitorOptions::lines(Some("a".repeat(1001)))).is_err());
        let invalid = check("x", MonitorOptions::lines(Some("(".into()))).unwrap_err();
        assert!(
            invalid
                .to_string()
                .starts_with("the pattern isn't a regular expression this monitor can use ("),
            "{invalid}"
        );
        assert!(check("x", MonitorOptions::check(10, CheckChange::Change)).is_ok());
        assert!(check("x", MonitorOptions::check(9, CheckChange::Change)).is_err());
        assert!(check("x", MonitorOptions::check(601, CheckChange::Change)).is_err());
        let lines = MonitorOptions::lines(None);
        assert!(check("x", lines.clone().with_expiry(Duration::ZERO)).is_err());
        assert!(check("x", lines.clone().with_expiry(Duration::from_secs(1801))).is_err());
        assert!(check("x", lines.clone().with_expiry(Duration::from_secs(60))).is_ok());
        // cmd.exe takes one line.
        assert!(
            validate("a\nb", &lines, &limits, ShellDialect::Cmd).is_err()
                && validate("a\nb", &lines, &limits, ShellDialect::Posix).is_ok()
        );
    }

    #[test]
    fn times_read_as_clocks_and_minutes() {
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(17 * 3600 + 41 * 60 + 5);
        assert_eq!(clock(at), "17:41 UTC");
        assert_eq!(clock_seconds(at), "17:41:05 UTC");
        assert_eq!(minutes_left(Duration::from_secs(59)), "less than a minute");
        assert_eq!(minutes_left(Duration::from_secs(299)), "5 min");
        assert_eq!(minutes_left(Duration::from_secs(22 * 60 + 10)), "22 min");
    }

    /// A monitor that alerts once or lasts while workers run is saved so.
    /// A record without them, as earlier versions saved it, reads as a timed
    /// monitor that alerts every time.
    #[test]
    fn once_and_the_lifetime_are_saved_when_set() {
        let record = |once, lifetime| MonitorRecord {
            trigger: MonitorTrigger::Lines { pattern: None },
            expires_at: SystemTime::UNIX_EPOCH,
            alerts: 0,
            warned: false,
            once,
            lifetime,
        };
        let saved = serde_json::to_value(record(false, MonitorLifetime::Timed)).unwrap();
        assert_eq!(saved.get("once"), None);
        assert_eq!(saved.get("lifetime"), None);
        let read: MonitorRecord = serde_json::from_value(saved).unwrap();
        assert!(!read.once && read.lifetime == MonitorLifetime::Timed);
        let saved = serde_json::to_value(record(true, MonitorLifetime::WhileWorkersRun)).unwrap();
        assert_eq!(
            (&saved["once"], &saved["lifetime"]),
            (
                &serde_json::json!(true),
                &serde_json::json!("while_workers_run")
            )
        );
        let read: MonitorRecord = serde_json::from_value(saved).unwrap();
        assert!(read.once && read.lifetime == MonitorLifetime::WhileWorkersRun);
    }

    /// The trigger is saved by kind, and records saved without its
    /// optional parts read as their defaults.
    #[test]
    fn triggers_are_saved_by_kind() {
        assert_eq!(
            serde_json::to_value(MonitorTrigger::Lines {
                pattern: Some("error".into())
            })
            .unwrap(),
            serde_json::json!({"kind": "lines", "pattern": "error"})
        );
        assert_eq!(
            serde_json::from_value::<MonitorTrigger>(
                serde_json::json!({"kind": "check", "every_seconds": 15})
            )
            .unwrap(),
            MonitorTrigger::Check {
                every_seconds: 15,
                when: CheckChange::Change
            }
        );
        assert_eq!(
            MonitorTrigger::Lines {
                pattern: Some("error|warn".into())
            }
            .watching(),
            "lines that match error|warn"
        );
    }
}
