//! Each owner's outbox, which holds the news of all its monitors. The news
//! goes to the owner's runtime as one note at most once each
//! [`alert_interval`](MonitorLimits::alert_interval). With no runtime bound
//! for the owner (its agent is idle), news that should wake it is announced
//! once with a [`JobAlert`], and the runtime it binds next gets the news.

use super::monitor::{
    AlertKind, EndReason, JobAlert, Line, MonitorLimits, MonitorTrigger, clock, clock_seconds,
    minutes_left,
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
    ended: Option<Ended>,
    /// What of it should wake the owner, until it is announced.
    wakes: Option<AlertKind>,
}

impl Part {
    /// Whether the part carries an alert, which its monitor's count of
    /// alerts counts.
    fn alerts(&self) -> bool {
        self.lines.is_some()
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
                    ended: None,
                    wakes: None,
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
            News::Ended {
                reason,
                at,
                last_lines,
            } => {
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
    pub(super) fn announce(&mut self, now: Instant) -> Option<(Uuid, AlertKind)> {
        self.pending_since = None;
        let wake = self
            .parts
            .iter()
            .find_map(|part| part.wakes.map(|kind| (part.header.job_id, kind)));
        for part in &mut self.parts {
            part.wakes = None;
        }
        if wake.is_some() {
            self.last_went = Some(now);
        }
        wake
    }
}

/// The note for `parts`. `monitors` gives each monitor's end, as last
/// armed, and its count of alerts, this note's included.
pub(super) fn note(
    parts: &[Part],
    monitors: &HashMap<Uuid, (SystemTime, u32)>,
    now: SystemTime,
) -> String {
    parts
        .iter()
        .map(|part| {
            let (expires_at, alerts) = monitors
                .get(&part.header.job_id)
                .copied()
                .unwrap_or((now, 0));
            part_text(part, expires_at, alerts, now)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The text of one monitor's part of a note.
fn part_text(part: &Part, expires_at: SystemTime, alerts: u32, now: SystemTime) -> String {
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
        if part.ended.is_none() {
            text.push_str(&format!(
                "\nIt ends in {}.",
                minutes_left(expires_at.duration_since(now).unwrap_or_default())
            ));
        }
        paragraphs.push(text);
    }
    if let Some(ended) = &part.ended {
        paragraphs.push(ended_text(header, ended, alerts));
    }
    paragraphs.join("\n\n")
}

/// `It alerted 4 times.`
fn alerted(alerts: u32) -> String {
    match alerts {
        0 => "It didn't alert.".to_string(),
        1 => "It alerted once.".to_string(),
        count => format!("It alerted {count} times."),
    }
}

/// What a note says of a monitor that ended.
fn ended_text(header: &Header, ended: &Ended, alerts: u32) -> String {
    let id = header.job_id;
    let command = &header.command;
    let at = clock(ended.at);
    match &ended.reason {
        EndReason::Expired => format!(
            "Monitor {id} (`{command}`) ended at {at} and its command was stopped. {}",
            alerted(alerts)
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
                "Monitor {id} (`{command}`) stopped, as you asked. {}",
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
    /// been told some other way.
    pub(super) fn drop_news(&self, owner: Uuid, id: Uuid) {
        if let Some(outbox) = self.outboxes.lock().unwrap().get_mut(&owner) {
            outbox.forget(id);
        }
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
            monitors.insert(id, (monitor.expires_at, monitor.alerts));
            if part.alerts()
                && let Err(error) = persist(&self.config, &data.record)
            {
                log::warn!("Terminal job {id}: could not save its count of alerts: {error}");
            }
        }
        Some(note(&parts, &monitors, SystemTime::now()))
    }

    /// No runtime took `owner`'s news. Wakes it if the news should.
    fn announce_news(&self, owner: Uuid) {
        let wake = self
            .outboxes
            .lock()
            .unwrap()
            .get_mut(&owner)
            .and_then(|outbox| outbox.announce(Instant::now()));
        if let Some((job_id, kind)) = wake {
            let _ = self.alerts.send(JobAlert {
                owner,
                job_id,
                kind,
            });
        }
    }

    /// Sends `owner`'s news to the runtime bound now, if there is one, for
    /// a runtime that just bound.
    pub(super) fn flush_news(self: &Arc<Self>, owner: Uuid) {
        if self.has_monitor_news(owner) {
            let manager = Arc::downgrade(self);
            tokio::spawn(async move {
                send_news(&manager, owner).await;
            });
        }
    }
}

/// Sends `owner`'s news when it's due, until it has none left.
async fn drive_outbox(manager: Weak<TerminalManager>, owner: Uuid) {
    loop {
        let due = {
            let Some(manager) = manager.upgrade() else {
                return;
            };
            let limits = manager.monitor_limits();
            let mut outboxes = manager.outboxes.lock().unwrap();
            let Some(outbox) = outboxes.get_mut(&owner) else {
                return;
            };
            match outbox.due(&limits) {
                Some(due) => due,
                None => {
                    outbox.driving = false;
                    return;
                }
            }
        };
        if due > Instant::now() {
            sleep_until(due).await;
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
            outbox.announce(start + limits.gather),
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
        assert_eq!(outbox.announce(later), None, "an end wakes no one");
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
        let monitors = HashMap::from([(id, (ends, 1))]);
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
                &HashMap::from([(id, (at(0), 4))]),
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
        let text = note(&outbox.take(now), &HashMap::from([(id, (at(0), 1))]), at(0));
        assert!(
            text.ends_with(&format!(
                "\n  boom\n\nMonitor {id} (`npm run dev`) ended because its command ended without an exit code at 11:00 UTC. It printed nothing."
            )),
            "{text}"
        );
    }
}
