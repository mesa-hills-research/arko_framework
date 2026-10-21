//! Monitors that run a check again and again, and alert their owner when
//! its result changes.

use super::monitor::{CheckChange, MonitorTrigger, Running, clock_seconds};
use super::outbox::{CheckResult, News};
use super::*;
use futures::future::BoxFuture;
use std::time::SystemTime;
use tokio::io::AsyncReadExt;
use tokio::time::{Instant, sleep_until};

/// One run of a check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CheckRun {
    /// It exited with code 0.
    pub(super) success: bool,
    pub(super) exit_code: Option<i32>,
    /// It ran longer than its limit and was stopped.
    pub(super) timed_out: Option<Duration>,
    /// Why it couldn't run.
    pub(super) error: Option<String>,
    /// The first bytes of its output, stdout and stderr in arrival order.
    pub(super) output: Vec<u8>,
    /// How many more bytes it printed.
    pub(super) more: u64,
    pub(super) took: Duration,
    pub(super) at: SystemTime,
}

impl CheckRun {
    /// A run that couldn't start.
    pub(super) fn failed_to_run(error: impl std::fmt::Display) -> Self {
        Self {
            success: false,
            exit_code: None,
            timed_out: None,
            error: Some(error.to_string()),
            output: Vec::new(),
            more: 0,
            took: Duration::ZERO,
            at: SystemTime::now(),
        }
    }

    /// What failed, as a note says it after "It" (`exited with code 7`),
    /// or `None` for a success.
    fn failure(&self) -> Option<String> {
        if self.success {
            return None;
        }
        Some(match (&self.error, self.timed_out, self.exit_code) {
            (Some(error), _, _) => format!("couldn't run ({error})"),
            (None, Some(limit), _) => format!("timed out after {}", calls::describe(limit)),
            (None, None, Some(code)) => format!("exited with code {code}"),
            (None, None, None) => "ended without an exit code".to_string(),
        })
    }

    /// Its output as text, without escape sequences.
    fn text(&self) -> String {
        detection::strip_escapes(&String::from_utf8_lossy(&self.output))
    }

    /// The run as its news gives it.
    pub(super) fn result(&self) -> CheckResult {
        CheckResult {
            success: self.success,
            failure: self.failure(),
            output: self.text(),
            at: self.at,
        }
    }

    /// The line and output the run adds to its job's output and log.
    fn history(&self) -> Vec<u8> {
        let at = clock_seconds(self.at);
        let took = format!("{:.1} s", self.took.as_secs_f64());
        let mut line = match (&self.error, self.timed_out, self.exit_code) {
            (Some(error), _, _) => format!("[{at}] couldn't run ({error})\n"),
            (None, Some(limit), _) => {
                format!("[{at}] timed out after {}\n", calls::describe(limit))
            }
            (None, None, Some(code)) => format!("[{at}] exit {code}, {took}\n"),
            (None, None, None) => format!("[{at}] ended without an exit code, {took}\n"),
        }
        .into_bytes();
        line.extend_from_slice(&self.output);
        if !self.output.is_empty() && !self.output.ends_with(b"\n") {
            line.push(b'\n');
        }
        if self.more > 0 {
            line.extend_from_slice(format!("({} more bytes not kept)\n", self.more).as_bytes());
        }
        line
    }

    /// What `monitor_start` says of the first run.
    pub(super) fn first_sentence(&self, limits: &MonitorLimits) -> String {
        let took = format!("{:.1} s", self.took.as_secs_f64());
        let how = match (&self.error, self.timed_out, self.exit_code) {
            (Some(error), _, _) => format!("Its first check couldn't run ({error})"),
            (None, Some(limit), _) => {
                format!("Its first check timed out after {}", calls::describe(limit))
            }
            _ if self.success => format!("Its first check succeeded in {took}"),
            (None, None, Some(code)) => {
                format!("Its first check failed with exit code {code} in {took}")
            }
            (None, None, None) => {
                format!("Its first check ended without an exit code in {took}")
            }
        };
        let text = self.text();
        let text = text.trim_end();
        if text.is_empty() {
            format!("{how}, with no output.")
        } else {
            let mut shown = String::new();
            for line in text.lines().take(limits.lines_per_note) {
                if shown.len() + line.len() > limits.bytes_per_note && !shown.is_empty() {
                    break;
                }
                shown.push_str("\n  ");
                shown.push_str(line);
            }
            format!("{how}. Its output follows.{shown}")
        }
    }
}

/// Runs one check of at most the given time.
pub(super) type Checker = Arc<dyn Fn(Duration) -> BoxFuture<'static, CheckRun> + Send + Sync>;

/// The checker that runs `command` in `cwd` with the configured shell and
/// pipes, its input closed, keeping `keep` bytes of its output. A run that
/// takes longer than its time is stopped with the processes it started,
/// and so is a run that ends while they go on.
pub(super) fn command_checker(
    config: TerminalConfig,
    command: String,
    cwd: PathBuf,
    directory: PathBuf,
    keep: usize,
) -> Checker {
    Arc::new(move |limit| {
        let (config, command, cwd, directory) = (
            config.clone(),
            command.clone(),
            cwd.clone(),
            directory.clone(),
        );
        Box::pin(async move { run_check(&config, &command, &cwd, &directory, limit, keep).await })
    })
}

async fn run_check(
    config: &TerminalConfig,
    command: &str,
    cwd: &Path,
    directory: &Path,
    limit: Duration,
    keep: usize,
) -> CheckRun {
    let started = Instant::now();
    let at = SystemTime::now();
    let process = match io::spawn_check(config, command, cwd, directory) {
        Ok(process) => process,
        Err(error) => return CheckRun::failed_to_run(error),
    };
    let io::ChildProcess {
        mut child,
        input,
        output,
        mut group,
    } = process;
    drop(input);
    let kept = Arc::new(Mutex::new((Vec::new(), 0u64)));
    let mut readers = tokio::task::JoinSet::new();
    for mut reader in output {
        let kept = kept.clone();
        readers.spawn(async move {
            let mut bytes = vec![0; 8192];
            while let Ok(count) = reader.read(&mut bytes).await {
                if count == 0 {
                    break;
                }
                let mut kept = kept.lock().unwrap();
                let room = keep.saturating_sub(kept.0.len()).min(count);
                kept.0.extend_from_slice(&bytes[..room]);
                kept.1 += (count - room) as u64;
            }
        });
    }
    let waited = tokio::time::timeout(limit, child.wait()).await;
    group.kill();
    let _ = child.wait().await;
    // What it printed before it ended. Its processes are gone, so the
    // pipes close.
    let _ = tokio::time::timeout(Duration::from_secs(1), async {
        while readers.join_next().await.is_some() {}
    })
    .await;
    readers.abort_all();
    let (output, more) = std::mem::take(&mut *kept.lock().unwrap());
    let mut run = CheckRun {
        success: false,
        exit_code: None,
        timed_out: None,
        error: None,
        output,
        more,
        took: started.elapsed(),
        at,
    };
    match waited {
        Ok(Ok(status)) => {
            run.success = status.success();
            run.exit_code = status.code();
        }
        Ok(Err(error)) => run.error = Some(error.to_string()),
        Err(_) => run.timed_out = Some(limit),
    }
    run
}

/// Tells a check's changes from its steady results.
#[derive(Debug)]
pub(super) struct CheckState {
    when: CheckChange,
    /// The latest result, and since when it has been the same (as `when`
    /// compares it).
    current: Option<(CheckResult, SystemTime)>,
}

impl CheckState {
    pub(super) fn new(when: CheckChange) -> Self {
        Self {
            when,
            current: None,
        }
    }

    /// The news of `result`, if it changed the check's result. The first
    /// result is where changes count from, and alerts no one. A change of
    /// the kind `when` names alerts the owner. Other changes only update an
    /// alert that hasn't gone yet.
    pub(super) fn observe(&mut self, result: CheckResult) -> Option<News> {
        let Some((previous, since)) = self.current.take() else {
            let at = result.at;
            self.current = Some((result, at));
            return None;
        };
        let changed = match self.when {
            CheckChange::Output => previous.output != result.output,
            _ => previous.success != result.success,
        };
        if !changed {
            self.current = Some((result, since));
            return None;
        }
        let alerts = match self.when {
            CheckChange::Success => result.success,
            CheckChange::Failure => !result.success,
            _ => true,
        };
        let previous_for = result.at.duration_since(since).unwrap_or_default();
        let at = result.at;
        self.current = Some((result.clone(), at));
        Some(News::Check {
            result,
            alerts,
            previous_for,
        })
    }
}

/// Runs a check monitor's job, its checks (never two at once, each at most
/// its interval or the limit) and its end.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run(
    manager: Weak<TerminalManager>,
    config: TerminalConfig,
    data: Arc<Mutex<JobData>>,
    mut log: std::fs::File,
    running: Running,
    checker: Checker,
    mut controls: mpsc::Receiver<Control>,
    mut cancellation: watch::Receiver<bool>,
    changed: watch::Sender<u64>,
    first: oneshot::Sender<CheckRun>,
) {
    let MonitorTrigger::Check {
        every_seconds,
        when,
    } = running.header.trigger.clone()
    else {
        unreachable!("a check monitor runs checks")
    };
    let every = Duration::from_secs(every_seconds);
    let limit = every.min(running.limits.check_run_limit);
    let mut state = CheckState::new(when);
    let mut first = Some(first);
    let mut logged = 0;
    let mut next = Instant::now();
    let mut current: Option<(BoxFuture<'static, CheckRun>, Instant)> = None;
    let mut expired = false;
    let mut failure: Option<String> = None;
    // The start reported a first check that couldn't run, and ends the
    // monitor.
    let mut told = false;
    let mut timing = manager
        .upgrade()
        .map(|manager| manager.watch_monitor_timing());
    let stop = loop {
        let event = runner::until(running.next_event(&manager, &data));
        let idle = current.is_none();
        let check = async {
            match current.as_mut() {
                // A checker that overstays its limit counts as timed out,
                // counted from the run's start.
                Some((run, started)) => {
                    tokio::time::timeout_at(*started + limit + Duration::from_secs(5), run)
                        .await
                        .unwrap_or_else(|_| CheckRun {
                            timed_out: Some(limit),
                            error: None,
                            took: started.elapsed(),
                            ..CheckRun::failed_to_run("")
                        })
                }
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            biased;
            _ = cancellation.changed() => break JobState::Cancelled,
            run = check => {
                current = None;
                let history = run.history();
                let result = {
                    let mut data = data.lock().unwrap();
                    runner::output(&mut data, &history, &mut log, &mut logged, &config)
                };
                changed.send_replace(data.lock().unwrap().record.revision);
                if let Err(error) = result {
                    failure = Some(error.to_string());
                    break JobState::Failed;
                }
                if let Some(first) = first.take() {
                    let couldnt_run = run.error.is_some();
                    let _ = first.send(run.clone());
                    if couldnt_run {
                        told = true;
                        failure = run.error.clone();
                        break JobState::Failed;
                    }
                }
                if let Some(news) = state.observe(run.result()) {
                    running.tell(&manager, news);
                }
            }
            _ = sleep_until(next), if idle => {
                let started = Instant::now();
                next = started + every;
                current = Some((checker(limit), started));
            }
            // Planned again.
            () = runner::retimed(&mut timing) => {}
            () = event => {
                if running.timed_event(&manager, &data, &config) {
                    expired = true;
                    break JobState::TimedOut;
                }
            }
            control = controls.recv() => match control {
                Some(Control::Input { response, .. }) => {
                    let _ = response.send(Err(TerminalError::Invalid(
                        "this job runs a monitor's check, which takes no input".into(),
                    )));
                }
                None => break JobState::Cancelled,
            },
        }
    };
    running.ending(&data);
    // Dropping a run stops its processes.
    drop(current);
    let (snapshot, ending) = {
        let mut data = data.lock().unwrap();
        let ending = monitor::end_reason(stop, expired, data.stopped_by, None, failure.as_deref());
        data.record.state = stop;
        data.record.error = failure.or(match data.stopped_by {
            _ if expired => Some("expired".into()),
            Some(calls::StopCause::User) => Some("stopped by the user".into()),
            Some(calls::StopCause::OwnerEnded) => {
                Some("stopped because its agent's run ended".into())
            }
            _ => None,
        });
        data.record.revision += 1;
        if let Err(error) = persist(&config, &data.record) {
            data.record.state = JobState::Failed;
            data.record.error = Some(format!("could not persist terminal result: {error}"));
        }
        let snapshot = data
            .snapshot(&config, None, config.excerpt_bytes)
            .expect("valid output window");
        (snapshot, ending)
    };
    if let Some(reason) = ending.filter(|_| !told) {
        running.tell(
            &manager,
            News::Ended {
                reason,
                at: SystemTime::now(),
                last_lines: Vec::new(),
            },
        );
    }
    changed.send_replace(snapshot.revision);
}

impl TerminalManager {
    /// Starts a check monitor that runs its checks with `checker`, and
    /// waits for its first result.
    pub(super) async fn start_check(
        self: &Arc<Self>,
        context: &TaskContext<'_>,
        label: String,
        watching: monitor::Watching,
        checker: impl FnOnce(PathBuf) -> Checker,
    ) -> Result<(Uuid, Option<CheckRun>)> {
        let limits = watching.limits.clone();
        let owner = context.task_id;
        let (data, directory, receiver, cancellation, changed, completed, mut done) = {
            let mut jobs = self.jobs.lock().unwrap();
            if self.closing.load(std::sync::atomic::Ordering::Acquire) {
                return Err(TerminalError::Invalid(
                    "terminal manager is shut down".into(),
                ));
            }
            monitor::check_room(&jobs, owner, &limits)?;
            let id = Uuid::new_v4();
            let directory = self.config.directory.join(id.to_string());
            std::fs::create_dir(&directory)?;
            let record = Record {
                job_id: id,
                owner,
                revision: 0,
                state: JobState::Running,
                input: None,
                exit_code: None,
                error: None,
                terminal: false,
                output_end: 0,
                log_truncated: false,
                held: false,
                tool_call_id: context.tool_call_id,
                label: Some(label.clone()),
                call: None,
                stopped_by_close: false,
                monitor: Some(watching.record.clone()),
            };
            persist(&self.config, &record)?;
            let mut data = JobData::new(record);
            data.monitor = Some(watching.arming.clone());
            let data = Arc::new(Mutex::new(data));
            let (controls, receiver) = mpsc::channel(4);
            let (cancel, cancellation) = watch::channel(false);
            let (changed, changes) = watch::channel(0);
            let (completed, done) = watch::channel(false);
            jobs.insert(
                id,
                Arc::new(Job {
                    data: data.clone(),
                    working_directory: Some(context.current_working_directory.to_path_buf()),
                    command: label.clone(),
                    started: std::time::Instant::now(),
                    controls,
                    cancel,
                    changes,
                    done: done.clone(),
                }),
            );
            (
                data,
                directory,
                receiver,
                cancellation,
                changed,
                completed,
                done,
            )
        };
        let id = data.lock().unwrap().record.job_id;
        let log = std::fs::File::create(directory.join("output.log"))?;
        let checker = checker(directory);
        let running = watching.running(id, label, owner);
        let (first, first_run) = oneshot::channel();
        let weak = Arc::downgrade(self);
        let config = self.config.clone();
        let directory_lock = self._directory_lock.clone();
        tokio::spawn(async move {
            run(
                weak,
                config,
                data,
                log,
                running,
                checker,
                receiver,
                cancellation,
                changed,
                first,
            )
            .await;
            drop(directory_lock);
            completed.send_replace(true);
        });
        let first = match tokio::time::timeout(limits.first_check_wait, first_run).await {
            Ok(Ok(run)) => Some(run),
            _ => None,
        };
        if let Some(run) = &first
            && let Some(error) = &run.error
        {
            let _ = done.wait_for(|done| *done).await;
            return Err(TerminalError::Invalid(format!(
                "could not run the check ({error})"
            )));
        }
        Ok((id, first))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(success: bool, output: &str, seconds: u64) -> CheckResult {
        CheckResult {
            success,
            failure: (!success).then(|| "exited with code 1".to_string()),
            output: output.into(),
            at: SystemTime::UNIX_EPOCH + Duration::from_secs(seconds),
        }
    }

    /// What `observe` gives for each result: `None`, or whether the news
    /// alerts and how long the earlier result lasted, in seconds.
    fn observed(when: CheckChange, results: &[(bool, &str)]) -> Vec<Option<(bool, u64)>> {
        let mut state = CheckState::new(when);
        results
            .iter()
            .enumerate()
            .map(|(index, (success, output))| {
                match state.observe(result(*success, output, 10 * index as u64)) {
                    Some(News::Check {
                        alerts,
                        previous_for,
                        ..
                    }) => Some((alerts, previous_for.as_secs())),
                    Some(other) => panic!("{other:?}"),
                    None => None,
                }
            })
            .collect()
    }

    /// A check alerts on a change of its result, never while it stays the
    /// same, and its first result alerts no one. Which change alerts
    /// follows `when`, and the other changes are news that only updates a
    /// pending alert.
    #[test]
    fn checks_alert_on_the_changes_they_watch() {
        let runs = [
            (false, ""),
            (false, ""),
            (true, "ok"),
            (true, "ok"),
            (false, "down"),
            (true, "ok 2"),
        ];
        assert_eq!(
            observed(CheckChange::Change, &runs),
            [
                None,
                None,
                Some((true, 20)),
                None,
                Some((true, 20)),
                Some((true, 10))
            ]
        );
        assert_eq!(
            observed(CheckChange::Success, &runs),
            [
                None,
                None,
                Some((true, 20)),
                None,
                Some((false, 20)),
                Some((true, 10))
            ]
        );
        assert_eq!(
            observed(CheckChange::Failure, &runs),
            [
                None,
                None,
                Some((false, 20)),
                None,
                Some((true, 20)),
                Some((false, 10))
            ]
        );
        // The output changes at each of the last four, success or not.
        assert_eq!(
            observed(CheckChange::Output, &runs),
            [
                None,
                None,
                Some((true, 20)),
                None,
                Some((true, 20)),
                Some((true, 10))
            ]
        );
        assert_eq!(
            observed(CheckChange::Output, &[(true, "a"), (false, "a")]),
            [None, None]
        );
    }

    /// A run's line in its job's output says when it ran and how it ended,
    /// and the start says what the first run gave.
    #[test]
    fn a_runs_history_and_first_sentence() {
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(10 * 3600 + 44 * 60 + 30);
        let run = CheckRun {
            success: false,
            exit_code: Some(7),
            timed_out: None,
            error: None,
            output: b"refused".to_vec(),
            more: 12,
            took: Duration::from_millis(120),
            at,
        };
        assert_eq!(
            String::from_utf8(run.history()).unwrap(),
            "[10:44:30 UTC] exit 7, 0.1 s\nrefused\n(12 more bytes not kept)\n"
        );
        let limits = MonitorLimits::default();
        assert_eq!(
            run.first_sentence(&limits),
            "Its first check failed with exit code 7 in 0.1 s. Its output follows.\n  refused"
        );
        let quiet = CheckRun {
            output: Vec::new(),
            ..run.clone()
        };
        assert_eq!(
            quiet.first_sentence(&limits),
            "Its first check failed with exit code 7 in 0.1 s, with no output."
        );
        let passed = CheckRun {
            success: true,
            exit_code: Some(0),
            output: b"{\"status\":\"ok\"}\n".to_vec(),
            ..run.clone()
        };
        assert_eq!(
            passed.first_sentence(&limits),
            "Its first check succeeded in 0.1 s. Its output follows.\n  {\"status\":\"ok\"}"
        );
        let slow = CheckRun {
            exit_code: None,
            timed_out: Some(Duration::from_secs(15)),
            output: Vec::new(),
            ..run
        };
        assert_eq!(
            String::from_utf8(slow.history()).unwrap(),
            "[10:44:30 UTC] timed out after 15 s\n(12 more bytes not kept)\n"
        );
        assert_eq!(
            slow.result().failure.as_deref(),
            Some("timed out after 15 s")
        );
    }
}
