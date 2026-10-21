//! Monitors: alerts of matching lines, the outbox's rate, how monitors end,
//! the limits, and restarts.

use super::monitor::{EndReason, Line};
use super::outbox::{Header, News};
use super::tests::{Fixture, note};
use super::*;
use serde_json::json;
use std::time::SystemTime;
use tokio::time::{Instant, timeout};

/// Limits that send news soon, for tests that run real commands.
fn quick() -> MonitorLimits {
    MonitorLimits {
        alert_interval: Duration::from_millis(400),
        gather: Duration::from_millis(100),
        start_wait: Duration::ZERO,
        ..MonitorLimits::default()
    }
}

fn fixture(limits: MonitorLimits) -> Fixture {
    let f = Fixture::new(|_| {});
    f.manager.set_monitor_limits(limits).unwrap();
    f
}

/// Starts a monitor of a Python fixture running `code`, shown as `label`.
async fn monitor(f: &Fixture, code: &str, options: MonitorOptions) -> MonitorStarted {
    f.manager
        .start_monitor(&f.context(), &f.command(code), options)
        .await
        .unwrap()
}

fn lines(pattern: &str) -> MonitorOptions {
    MonitorOptions::lines(Some(pattern.into())).with_label("dev server")
}

/// Waits for `owner`'s job `id` to end.
async fn ended(f: &Fixture, id: Uuid) -> TerminalSnapshot {
    timeout(Duration::from_secs(30), async {
        loop {
            let snapshot = f.manager.status(f.owner, id, None).unwrap();
            if snapshot.state.is_terminal() {
                return snapshot;
            }
            let _ = f
                .manager
                .wait(f.owner, id, snapshot.revision, Duration::from_millis(100))
                .await;
        }
    })
    .await
    .expect("the job ends")
}

fn header(id: Uuid) -> Header {
    Header {
        job_id: id,
        command: "npm run dev".into(),
        trigger: MonitorTrigger::Lines {
            pattern: Some("error".into()),
        },
    }
}

fn matched(text: &str) -> News {
    News::Lines {
        lines: vec![Line {
            text: text.into(),
            start: 0,
        }],
        noisy: false,
        at: SystemTime::now(),
    }
}

/// The note a runtime got within `within`, if one came.
async fn note_within(
    receiver: &mut mpsc::Receiver<UserCommand>,
    within: Duration,
) -> Option<String> {
    match timeout(within, receiver.recv()).await {
        Ok(Some(UserCommand::InjectContext { content })) => Some(content),
        Ok(other) => panic!("not a note: {other:?}"),
        Err(_) => None,
    }
}

/// An owner's news goes as one note, for all its monitors, after a short
/// wait for more. Later news waits until an interval has passed since that
/// note, however much comes.
#[tokio::test(start_paused = true)]
async fn an_owners_news_goes_at_most_once_an_interval() {
    let f = Fixture::new(|_| {});
    let limits = f.manager.monitor_limits();
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let (one, two) = (Uuid::new_v4(), Uuid::new_v4());
    let start = Instant::now();
    f.manager
        .queue_news(f.owner, header(one), matched("error one"));
    tokio::time::sleep(Duration::from_secs(1)).await;
    f.manager
        .queue_news(f.owner, header(two), matched("error two"));
    let first = note(&mut receiver).await;
    assert_eq!(start.elapsed(), limits.gather);
    assert!(
        first.contains(&format!("Job {one} ")) && first.contains(&format!("Job {two} ")),
        "{first}"
    );
    let sent = Instant::now();

    tokio::time::sleep(Duration::from_secs(5)).await;
    f.manager
        .queue_news(f.owner, header(one), matched("error three"));
    f.manager
        .queue_news(f.owner, header(two), matched("error four"));
    assert_eq!(
        note_within(&mut receiver, Duration::from_secs(50)).await,
        None
    );
    let second = note(&mut receiver).await;
    assert_eq!(sent.elapsed(), limits.alert_interval);
    assert!(
        second.contains("error three") && second.contains("error four"),
        "{second}"
    );
    assert!(!f.manager.has_news(f.owner));
}

/// News no runtime takes waits for the owner's next one, and wakes the
/// owner once with an alert, at most one an interval. A monitor's end alone
/// wakes no one.
#[tokio::test(start_paused = true)]
async fn news_no_runtime_takes_wakes_its_owner_once() {
    let f = Fixture::new(|_| {});
    let limits = f.manager.monitor_limits();
    let mut alerts = f.manager.subscribe_alerts();
    let id = Uuid::new_v4();
    let start = Instant::now();
    f.manager
        .queue_news(f.owner, header(id), matched("error one"));
    let alert = alerts.recv().await.unwrap();
    assert_eq!(start.elapsed(), limits.gather);
    assert_eq!(
        alert,
        JobAlert {
            owner: f.owner,
            job_id: id,
            kind: AlertKind::Lines
        }
    );
    assert!(f.manager.has_news(f.owner));
    let woke = Instant::now();

    f.manager
        .queue_news(f.owner, header(id), matched("error two"));
    alerts.recv().await.unwrap();
    assert_eq!(woke.elapsed(), limits.alert_interval);

    f.manager.queue_news(
        f.owner,
        header(id),
        News::Ended {
            reason: EndReason::StoppedByUser,
            at: SystemTime::now(),
            last_lines: Vec::new(),
        },
    );
    tokio::time::sleep(limits.alert_interval * 2).await;
    assert!(alerts.try_recv().is_err(), "an end wakes no one");

    // The runtime of its next turn gets all of it.
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let text = note(&mut receiver).await;
    assert!(
        text.contains("error one")
            && text.contains("error two")
            && text.contains(&format!("The user stopped monitor {id} (`npm run dev`).")),
        "{text}"
    );
    assert!(!f.manager.has_news(f.owner));
    assert!(alerts.try_recv().is_err());
}

/// A line monitor's matching lines reach its owner as one note, up to the
/// limit, with the count of the rest and where its output has them. The
/// monitor is listed as one, with its count of alerts.
#[tokio::test]
async fn a_line_monitor_alerts_of_matching_lines() {
    let f = fixture(MonitorLimits {
        lines_per_note: 2,
        ..quick()
    });
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let started = monitor(
        &f,
        "print('\\n'.join(['[ERROR] one', 'fine', '[ERROR] two', 'warn three', '[error] four']), flush=True)\ntime.sleep(60)",
        lines("(?i)error|warn"),
    )
    .await;
    let id = started.job_id;
    assert!(
        started.summary.starts_with(&format!(
            "Monitor {id} started. It runs `dev server` alerts you of lines that match `(?i)error|warn`, until "
        )) && started.summary.contains(" UTC (30 min). "),
        "{}",
        started.summary
    );
    // One write, read at once, so one note.
    let text = note(&mut receiver).await;
    assert!(
        text.starts_with(&format!(
            "Monitor alert. Job {id} (`dev server`, lines that match `(?i)error|warn`) printed 4 matching lines since "
        )) && text.contains(
            " UTC.\n  [ERROR] one\n  [ERROR] two\n2 more matching lines are in its output. terminal_status with after_output "
        ) && text.ends_with(" reads them.\nIt ends in 30 min."),
        "{text}"
    );
    let rest = text
        .split("terminal_status with after_output ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .expect("where the rest is");
    let rest = f
        .manager
        .status(f.owner, id, Some(rest.parse().unwrap()))
        .unwrap();
    assert!(
        rest.output.text.trim_start().starts_with("warn three"),
        "{:?}",
        rest.output.text
    );
    let jobs = f.manager.jobs();
    let JobKind::Monitor(info) = &jobs[0].kind else {
        panic!("not a monitor: {jobs:?}")
    };
    assert_eq!(
        info.trigger,
        MonitorTrigger::Lines {
            pattern: Some("(?i)error|warn".into())
        }
    );
    assert_eq!(info.alerts, 1);
    assert_eq!(info.expires_at, started.expires_at);
    assert_eq!(jobs[0].label, "dev server");

    // The owner's stop says so, and wakes no one.
    let mut finished = f.manager.subscribe_finished();
    f.manager.cancel(f.owner, id).unwrap();
    let stopped = note(&mut receiver).await;
    assert_eq!(
        stopped,
        format!("Monitor {id} (`dev server`) stopped, as you asked. It alerted once.")
    );
    assert_eq!(ended(&f, id).await.state, JobState::Cancelled);
    assert!(finished.try_recv().is_err());
}

/// A monitor ends when its time is up, which stops its command and wakes
/// no one. A command that exits by itself ends it too, and is reported as
/// finished, as other jobs are. One that exits at once says so in the
/// start's result.
#[tokio::test]
async fn a_monitor_ends_when_it_expires_or_its_command_exits() {
    let f = fixture(quick());
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let expiring = monitor(
        &f,
        "time.sleep(60)",
        lines("error").with_expiry(Duration::from_secs(1)),
    )
    .await;
    let done = ended(&f, expiring.job_id).await;
    assert_eq!(
        (done.state, done.error.as_deref()),
        (JobState::TimedOut, Some("expired"))
    );
    let text = note(&mut receiver).await;
    assert!(
        text.starts_with(&format!(
            "Monitor {} (`dev server`) ended at ",
            expiring.job_id
        )) && text.ends_with(" UTC and its command was stopped. It didn't alert."),
        "{text}"
    );
    assert!(finished.try_recv().is_err(), "an expiry wakes no one");

    let exiting = monitor(
        &f,
        "time.sleep(0.5)\nprint('bye', flush=True)\nsys.exit(3)",
        lines("error"),
    )
    .await;
    assert!(exiting.summary.ends_with("It has printed nothing yet."));
    let text = note(&mut receiver).await;
    assert!(
        text.starts_with(&format!(
            "Monitor {} (`dev server`) ended because its command exited with code 3 at ",
            exiting.job_id
        )) && text.ends_with(" UTC. Its last lines follow.\n  bye"),
        "{text}"
    );
    assert_eq!(
        timeout(Duration::from_secs(5), finished.recv())
            .await
            .unwrap()
            .unwrap(),
        JobFinished {
            owner: f.owner,
            job_id: exiting.job_id,
            state: JobState::Failed
        }
    );

    f.manager
        .set_monitor_limits(MonitorLimits {
            start_wait: Duration::from_secs(10),
            ..quick()
        })
        .unwrap();
    let at_once = monitor(&f, "print('nope', flush=True)\nsys.exit(2)", lines("error")).await;
    assert!(
        at_once.summary.starts_with(&format!(
            "Monitor {} ended at once, because its command `dev server` exited with code 2. Its output follows.\n",
            at_once.job_id
        )) && at_once.summary.ends_with("nope"),
        "{}",
        at_once.summary
    );
    assert_eq!(
        note_within(&mut receiver, Duration::from_secs(1)).await,
        None,
        "the start said it ended"
    );
}

/// The user's stop and the end of the owner's run stop a monitor without
/// waking anyone, and its note says why it stopped. The owner's commands
/// keep running.
#[tokio::test]
async fn stopping_a_monitor_for_the_user_or_its_owners_end_wakes_no_one() {
    let f = fixture(quick());
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let mut alerts = f.manager.subscribe_alerts();
    let first = monitor(&f, "time.sleep(60)", lines("error")).await.job_id;
    f.manager.stop_for_user(f.owner, first).unwrap();
    assert_eq!(
        note(&mut receiver).await,
        format!("The user stopped monitor {first} (`dev server`).")
    );
    assert_eq!(
        ended(&f, first).await.error.as_deref(),
        Some("stopped by the user")
    );

    let second = monitor(&f, "time.sleep(60)", lines("error")).await.job_id;
    let command = f
        .manager
        .start(
            &f.context(),
            &f.command("time.sleep(60)"),
            Default::default(),
        )
        .unwrap()
        .job_id;
    assert_eq!(f.manager.stop_owner_watchers(f.owner), 1);
    assert_eq!(
        note(&mut receiver).await,
        format!("Monitor {second} (`dev server`) stopped because your run ended.")
    );
    assert_eq!(
        ended(&f, second).await.error.as_deref(),
        Some("stopped because its agent's run ended")
    );
    assert_eq!(
        f.manager.status(f.owner, command, None).unwrap().state,
        JobState::Running
    );
    assert!(finished.try_recv().is_err() && alerts.try_recv().is_err());
    f.manager.cancel(f.owner, command).unwrap();
}

/// An owner may have so many monitors running, and a session so many. The
/// command guard checks a monitor's command, and an invalid pattern says
/// why.
#[tokio::test]
async fn monitors_are_limited_and_their_commands_checked() {
    let f = fixture(MonitorLimits {
        per_owner: 2,
        in_all: 3,
        ..quick()
    });
    for _ in 0..2 {
        monitor(&f, "time.sleep(60)", lines("error")).await;
    }
    let refused = f
        .manager
        .start_monitor(&f.context(), &f.command("time.sleep(60)"), lines("error"))
        .await
        .unwrap_err();
    assert!(
        refused
            .to_string()
            .starts_with("you have 2 monitors running, the most an agent may have."),
        "{refused}"
    );
    // Another owner has room until the session is full.
    let other = TaskContext::new(Uuid::new_v4(), "other", "test monitors", &f.root);
    f.manager
        .start_monitor(&other, &f.command("time.sleep(60)"), lines("error"))
        .await
        .unwrap();
    let full = f
        .manager
        .start_monitor(&other, &f.command("time.sleep(60)"), lines("error"))
        .await
        .unwrap_err();
    assert!(
        full.to_string()
            .starts_with("3 monitors are running in this session"),
        "{full}"
    );
    f.manager.shutdown().await;

    let f = fixture(quick());
    let guarded = f
        .manager
        .start_monitor(&f.context(), "rm -rf /", MonitorOptions::lines(None))
        .await;
    assert!(matches!(guarded, Err(TerminalError::Refused(_))));
    let invalid = f
        .manager
        .start_monitor(&f.context(), "echo hi", lines("(unclosed"))
        .await
        .unwrap_err();
    assert!(
        invalid
            .to_string()
            .starts_with("the pattern isn't a regular expression"),
        "{invalid}"
    );
    assert!(f.manager.jobs().is_empty());
    assert!(
        f.manager
            .set_monitor_limits(MonitorLimits {
                per_owner: 0,
                ..MonitorLimits::default()
            })
            .is_err()
    );
}

/// A monitor that ran when its host stopped is an interrupted record after
/// the restart, listed with what it watched, and it never runs again. One
/// a shutdown stopped is listed too.
#[tokio::test]
async fn monitors_stopped_by_a_restart_are_listed_and_never_run_again() {
    let f = Fixture::new(|_| {});
    let directory = f.root.join("restarted");
    let manager = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    manager.set_monitor_limits(quick()).unwrap();
    let stopped = manager
        .start_monitor(&f.context(), &f.command("time.sleep(60)"), lines("error"))
        .await
        .unwrap();
    manager.shutdown().await;
    drop(manager);
    // One saved as running, as a crash leaves it.
    let crashed = Uuid::new_v4();
    std::fs::create_dir(directory.join(crashed.to_string())).unwrap();
    std::fs::write(
        directory.join(crashed.to_string()).join("job.json"),
        json!({
            "job_id": crashed, "owner": f.owner, "revision": 1, "state": "running",
            "input": null, "exit_code": null, "error": null, "terminal": true,
            "output_end": 0, "log_truncated": false, "label": "npm run dev",
            "monitor": {
                "trigger": {"kind": "lines"},
                "expires_at": {"secs_since_epoch": 1_800_000_000u64, "nanos_since_epoch": 0},
                "alerts": 3
            }
        })
        .to_string(),
    )
    .unwrap();

    let manager = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    let mut interrupted = manager.interrupted_at_open();
    interrupted.sort_by_key(|job| job.job_id != crashed);
    assert_eq!(interrupted.len(), 2, "{interrupted:?}");
    assert_eq!(
        (interrupted[0].label.as_str(), &interrupted[0].kind),
        (
            "npm run dev",
            &JobKind::Monitor(MonitorInfo {
                trigger: MonitorTrigger::Lines { pattern: None },
                expires_at: SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000),
                alerts: 3,
            })
        )
    );
    assert_eq!(interrupted[1].job_id, stopped.job_id);
    assert!(matches!(
        &interrupted[1].kind,
        JobKind::Monitor(info) if info.expires_at == stopped.expires_at
    ));
    assert_eq!(
        manager.status(f.owner, crashed, None).unwrap().state,
        JobState::Interrupted
    );
    assert!(manager.jobs().is_empty());
}

/// A checker that gives `script`'s results in turn, each taking its time
/// (or the run's limit, as a time-out), then successes. It counts the most
/// runs at once.
fn scripted(
    script: Vec<(bool, &'static str, Duration)>,
) -> (checks::Checker, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let base = (SystemTime::now(), Instant::now());
    let queue = Arc::new(Mutex::new(VecDeque::from(script)));
    let (running, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let checker: checks::Checker = Arc::new({
        let most = most.clone();
        move |limit| {
            let (queue, running, most) = (queue.clone(), running.clone(), most.clone());
            Box::pin(async move {
                most.fetch_max(running.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                let at = base.0 + (Instant::now() - base.1);
                let (success, output, took) = queue.lock().unwrap().pop_front().unwrap_or((
                    true,
                    "ok",
                    Duration::from_millis(100),
                ));
                tokio::time::sleep(took.min(limit)).await;
                running.fetch_sub(1, Ordering::SeqCst);
                let timed_out = (took > limit).then_some(limit);
                checks::CheckRun {
                    success: success && timed_out.is_none(),
                    exit_code: timed_out.is_none().then_some(if success { 0 } else { 1 }),
                    timed_out,
                    error: None,
                    output: output.as_bytes().to_vec(),
                    more: 0,
                    took: took.min(limit),
                    at,
                }
            })
        }
    });
    (checker, most)
}

/// A check runs every interval, never two runs at once, a run that runs
/// over its interval is stopped and counts as a failure, and each run is in
/// the job's output. Its owner hears of a change, not of a steady result,
/// and of several changes within an interval as one alert with their count.
#[tokio::test(start_paused = true)]
async fn a_check_alerts_when_its_result_changes() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let quick = Duration::from_millis(100);
    let (checker, most) = scripted(vec![
        (false, "down", quick),
        (false, "down", quick),
        (true, "ok", quick),
        (true, "ok", quick),
        (false, "down", Duration::from_secs(15)),
        (true, "ok", quick),
        (false, "down", quick),
        (true, "ok", quick),
    ]);
    let options = MonitorOptions::check(10, CheckChange::Change).with_label("health");
    let watching = monitor::Watching::new(&options, None, &f.manager.monitor_limits());
    let start = Instant::now();
    let (id, first) = f
        .manager
        .start_check(&f.context(), "health".into(), watching, move |_| checker)
        .await
        .unwrap();
    let first = first.expect("the first result");
    assert_eq!(
        (first.success, first.output.as_slice()),
        (false, &b"down"[..])
    );

    let mut next = async || {
        note_within(&mut receiver, Duration::from_secs(3600))
            .await
            .expect("a note")
    };
    let text = next().await;
    assert_eq!(start.elapsed(), Duration::from_millis(22_100));
    assert!(
        text.starts_with(&format!(
            "Monitor alert. The check of job {id} (`health`, every 10 s) started succeeding at "
        )) && text.contains(" UTC, after failing for 20 s. Its output follows.\n  ok\n"),
        "{text}"
    );
    // The slow run at 40 s is the next change, and three more come before
    // the interval since the last note has passed.
    let text = next().await;
    assert_eq!(start.elapsed(), Duration::from_millis(82_100));
    assert!(
        text.contains(" changed 4 times since the last alert, and now succeeds (since "),
        "{text}"
    );
    assert_eq!(
        note_within(&mut receiver, Duration::from_secs(300)).await,
        None,
        "a steady result alerts no one"
    );
    assert_eq!(most.load(std::sync::atomic::Ordering::SeqCst), 1);
    let history = f.manager.status(f.owner, id, Some(0)).unwrap().output.text;
    assert!(
        history.contains(" UTC] exit 1, 0.1 s\ndown\n")
            && history.contains(" UTC] timed out after 10 s\ndown\n")
            && history.contains(" UTC] exit 0, 0.1 s\nok\n"),
        "{history}"
    );
}

/// A real check, run with the configured shell, gives its first result in
/// the start's result, and its change of result reaches the owner. A check
/// that runs over its interval is stopped. A check that can't run at all
/// is an error, and so is a command the guard refuses.
#[tokio::test]
async fn a_check_runs_its_command_again_and_again() {
    let f = fixture(MonitorLimits {
        shortest_check: Duration::from_secs(1),
        ..quick()
    });
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let flag = f.root.join("ready");
    let code = format!(
        "print('checked', flush=True)\nsys.exit(0 if os.path.exists({:?}) else 7)",
        flag.display().to_string()
    );
    let started = monitor(
        &f,
        &code,
        MonitorOptions::check(1, CheckChange::Success).with_label("ready check"),
    )
    .await;
    let id = started.job_id;
    assert!(
        started.summary.starts_with(&format!(
            "Monitor {id} started. It runs `ready check` every 1 s and alerts you when it starts succeeding, until "
        )) && started.summary.contains(" UTC (30 min). Its first check failed with exit code 7 in ")
            && started.summary.ends_with(" s. Its output follows.\n  checked"),
        "{}",
        started.summary
    );
    std::fs::write(&flag, "").unwrap();
    let text = note(&mut receiver).await;
    assert!(
        text.starts_with(&format!(
            "Monitor alert. The check of job {id} (`ready check`, every 1 s) started succeeding at "
        )) && text.contains(" Its output follows.\n  checked\n"),
        "{text}"
    );
    let snapshot = f.manager.status(f.owner, id, None).unwrap();
    assert!(!snapshot.terminal);
    f.manager.cancel(f.owner, id).unwrap();
    assert_eq!(ended(&f, id).await.state, JobState::Cancelled);

    let slow = monitor(
        &f,
        "time.sleep(30)",
        MonitorOptions::check(1, CheckChange::Change).with_label("slow"),
    )
    .await;
    assert!(
        slow.summary
            .ends_with("Its first check timed out after 1 s, with no output."),
        "{}",
        slow.summary
    );
    f.manager.cancel(f.owner, slow.job_id).unwrap();

    let refused = f
        .manager
        .start_monitor(
            &f.context(),
            "rm -rf /",
            MonitorOptions::check(1, CheckChange::Change),
        )
        .await;
    assert!(matches!(refused, Err(TerminalError::Refused(_))));

    let broken = Fixture::new(|config| config.shell = f.root.join("no-such-shell"));
    broken.manager.set_monitor_limits(quick()).unwrap();
    let error = broken
        .manager
        .start_monitor(
            &broken.context(),
            "true",
            MonitorOptions::check(10, CheckChange::Change),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().starts_with("could not run the check: "),
        "{error}"
    );
    assert!(broken.manager.jobs().is_empty());
}

/// Starts a check monitor of `owner` that lasts `lasts`, whose checks run
/// every 10 minutes and always succeed, with no process.
async fn steady_check(f: &Fixture, context: &TaskContext<'_>, lasts: Duration) -> Uuid {
    let options = MonitorOptions::check(600, CheckChange::Change)
        .with_label("health")
        .with_expiry(lasts);
    let watching = monitor::Watching::new(&options, None, &f.manager.monitor_limits());
    let (checker, _) = scripted(Vec::new());
    f.manager
        .start_check(context, "health".into(), watching, move |_| checker)
        .await
        .unwrap()
        .0
}

/// The note within an hour of paused time.
async fn next_note(receiver: &mut mpsc::Receiver<UserCommand>) -> String {
    note_within(receiver, Duration::from_secs(3600))
        .await
        .expect("a note")
}

/// A monitor warns its owner 5 minutes before it ends. A rearm keeps it
/// longer and arms the warning again. One armed for less than 10 minutes
/// ends without a warning, and an expiry wakes no one.
#[tokio::test(start_paused = true)]
async fn a_monitor_warns_before_it_ends_and_rearm_keeps_it() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let minutes = |count: u64| Duration::from_secs(60 * count);
    let start = Instant::now();
    let id = steady_check(&f, &f.context(), minutes(30)).await;
    let warning = next_note(&mut receiver).await;
    assert_eq!(start.elapsed(), minutes(25) + Duration::from_secs(2));
    assert!(
        warning.starts_with(&format!("Monitor {id} (`health`) ends in 5 min, at "))
            && warning.ends_with(
                " UTC. monitor_control rearm keeps it going for up to 30 min from now. If you no longer need it, let it end."
            ),
        "{warning}"
    );

    let rearmed = Instant::now();
    let ends = f.manager.rearm(f.owner, id, minutes(20)).unwrap();
    let JobKind::Monitor(info) = &f.manager.jobs()[0].kind else {
        panic!("not a monitor")
    };
    assert_eq!(info.expires_at, ends);
    let warning = next_note(&mut receiver).await;
    assert_eq!(rearmed.elapsed(), minutes(15) + Duration::from_secs(2));
    assert!(warning.contains(" ends in 5 min, at "), "{warning}");
    let ended = next_note(&mut receiver).await;
    assert_eq!(rearmed.elapsed(), minutes(20) + Duration::from_secs(2));
    assert!(
        ended.starts_with(&format!("Monitor {id} (`health`) ended at "))
            && ended.ends_with(" UTC, and its checks stopped. It didn't alert."),
        "{ended}"
    );
    let done = ended_paused(&f, id).await;
    assert_eq!(
        (done.state, done.error.as_deref()),
        (JobState::TimedOut, Some("expired"))
    );
    assert!(finished.try_recv().is_err());
    assert!(matches!(
        f.manager.rearm(f.owner, id, minutes(10)),
        Err(TerminalError::Finished)
    ));

    let started = Instant::now();
    let short = steady_check(&f, &f.context(), minutes(9)).await;
    assert!(
        f.manager.rearm(f.owner, short, minutes(31)).is_err(),
        "longer than a monitor may last"
    );
    let ended = next_note(&mut receiver).await;
    assert_eq!(started.elapsed(), minutes(9) + Duration::from_secs(2));
    assert!(ended.contains(" ended at "), "no warning first: {ended}");

    // Only a monitor can be rearmed.
    let call = f
        .manager
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "search",
            "search".into(),
            crate::stop::StopSource::new(),
            Box::pin(std::future::pending()),
        )
        .unwrap()
        .job_id;
    let refused = f.manager.rearm(f.owner, call, minutes(10)).unwrap_err();
    assert_eq!(refused.to_string(), format!("job {call} isn't a monitor"));
}

/// Waits in paused time for job `id` to end.
async fn ended_paused(f: &Fixture, id: Uuid) -> TerminalSnapshot {
    for _ in 0..100 {
        let snapshot = f.manager.status(f.owner, id, None).unwrap();
        if snapshot.state.is_terminal() {
            return snapshot;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("job {id} didn't end");
}

/// An idle agent whose prompt cache expires before its monitor's warning
/// is woken just before the cache expires, and the warning says how long
/// the monitor has left. A cache kept warm puts the warning off. While the
/// user is away the warning wakes no one, and comes at its usual time.
#[tokio::test(start_paused = true)]
async fn an_idle_agents_warning_comes_before_its_cache_expires() {
    let f = Fixture::new(|_| {});
    let mut alerts = f.manager.subscribe_alerts();
    let minutes = |count: u64| Duration::from_secs(60 * count);
    f.manager
        .set_prompt_cache_expiry(f.owner, Some(SystemTime::now() + minutes(5)));
    let start = Instant::now();
    let id = steady_check(&f, &f.context(), minutes(30)).await;
    let alert = alerts.recv().await.unwrap();
    assert_eq!(start.elapsed(), minutes(4) + Duration::from_secs(2));
    assert_eq!(
        alert,
        JobAlert {
            owner: f.owner,
            job_id: id,
            kind: AlertKind::EndsSoon
        }
    );
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let warning = next_note(&mut receiver).await;
    assert!(
        warning.starts_with(&format!("Monitor {id} (`health`) ends in 26 min, at ")),
        "{warning}"
    );
    drop((sender, receiver));
    f.manager.cancel(f.owner, id).unwrap();

    // The cache refreshed after 3 minutes, for 5 more.
    let other = TaskContext::new(Uuid::new_v4(), "other", "test monitors", &f.root);
    f.manager
        .set_prompt_cache_expiry(other.task_id, Some(SystemTime::now() + minutes(5)));
    let refreshed = steady_check(&f, &other, minutes(30)).await;
    let start = Instant::now();
    tokio::time::sleep(minutes(3)).await;
    f.manager
        .set_prompt_cache_expiry(other.task_id, Some(SystemTime::now() + minutes(5)));
    let alert = alerts.recv().await.unwrap();
    assert_eq!(alert.job_id, refreshed);
    assert_eq!(start.elapsed(), minutes(7) + Duration::from_secs(2));
    f.manager.cancel(other.task_id, refreshed).unwrap();

    // The user hasn't written for longer than warnings wake for.
    f.manager.set_warning_wakes_within(Some(minutes(1)));
    f.manager.set_user_active_at(SystemTime::now() - minutes(2));
    let away = TaskContext::new(Uuid::new_v4(), "away", "test monitors", &f.root);
    f.manager
        .set_prompt_cache_expiry(away.task_id, Some(SystemTime::now() + minutes(5)));
    let unattended = steady_check(&f, &away, minutes(30)).await;
    // Not before the cache expires, but 5 minutes before the end.
    tokio::time::sleep(minutes(24)).await;
    assert!(!f.manager.has_news(away.task_id));
    tokio::time::sleep(minutes(3)).await;
    assert!(alerts.try_recv().is_err(), "a warning wakes no one");
    assert!(f.manager.has_news(away.task_id));
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(away.task_id, sender.downgrade());
    let warning = next_note(&mut receiver).await;
    assert!(
        warning.starts_with(&format!("Monitor {unattended} (`health`) ends in ")),
        "{warning}"
    );
    // With no limit, warnings always wake.
    f.manager.set_warning_wakes_within(None);
    assert_eq!(f.manager.warning_wakes_within(), None);
    drop((sender, receiver));
    let later = steady_check(&f, &away, minutes(10)).await;
    assert_eq!(alerts.recv().await.unwrap().job_id, later);
}

/// A line monitor's command warns, plans its warning again when its
/// owner's prompt cache changes, takes a rearm, and is stopped at its end.
#[tokio::test]
async fn a_line_monitor_warns_and_takes_a_rearm() {
    let f = fixture(MonitorLimits {
        warned_from: Duration::from_secs(1),
        warning_before: Duration::from_secs(1),
        cache_margin: Duration::from_millis(500),
        ..quick()
    });
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let start = std::time::Instant::now();
    let id = monitor(
        &f,
        "time.sleep(60)",
        lines("error").with_expiry(Duration::from_secs(4)),
    )
    .await
    .job_id;
    // A cache that expires before the warning brings it forward.
    f.manager.set_prompt_cache_expiry(
        f.owner,
        Some(SystemTime::now() + Duration::from_millis(1500)),
    );
    let warning = note(&mut receiver).await;
    assert!(
        start.elapsed() < Duration::from_millis(2800),
        "{:?}",
        start.elapsed()
    );
    assert!(
        warning.starts_with(&format!(
            "Monitor {id} (`dev server`) ends in less than a minute, at "
        )),
        "{warning}"
    );
    f.manager.set_prompt_cache_expiry(f.owner, None);
    f.manager
        .rearm(f.owner, id, Duration::from_secs(2))
        .unwrap();
    let rearmed = std::time::Instant::now();
    let again = note(&mut receiver).await;
    assert!(
        again.contains(" ends in less than a minute, at "),
        "{again}"
    );
    let end = note(&mut receiver).await;
    assert!(rearmed.elapsed() >= Duration::from_secs(2));
    assert!(
        end.starts_with(&format!("Monitor {id} (`dev server`) ended at ")),
        "{end}"
    );
    assert_eq!(ended(&f, id).await.state, JobState::TimedOut);
}

fn params(value: serde_json::Value) -> HashMap<String, serde_json::Value> {
    serde_json::from_value(value).unwrap()
}

/// `monitor_start` starts a monitor of lines or a check from the model's
/// parameters (numbers given as numbers or text), and refuses parameters
/// that don't go together. `monitor_control` lists the agent's monitors,
/// rearms one and stops one.
#[tokio::test]
async fn the_monitor_tools_start_list_rearm_and_stop_monitors() {
    use crate::tool::{Tool, ToolParameterType};
    let f = fixture(quick());
    let tools = MonitorTool::all(f.manager.clone());
    let (start, control) = (&tools[0], &tools[1]);
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect::<Vec<_>>(),
        MonitorTool::NAMES
    );
    let definition = start.definition();
    assert!(definition.description.contains("(30 at most)"));
    let kinds: HashMap<String, (bool, ToolParameterType)> = definition
        .parameters
        .iter()
        .map(|param| {
            (
                param.name.clone(),
                (param.required, param.param_type.clone()),
            )
        })
        .collect();
    assert_eq!(kinds["command"], (true, ToolParameterType::String));
    assert_eq!(kinds["every_seconds"], (false, ToolParameterType::Integer));
    assert_eq!(
        kinds["expires_minutes"],
        (false, ToolParameterType::Integer)
    );
    assert!(
        tools
            .iter()
            .all(|tool| !tool.allow_result_interception() && !tool.uses_host_timeout())
    );

    let started = start
        .execute(
            &f.context(),
            params(json!({
                "command": f.command("time.sleep(60)"),
                "pattern": "error",
                "expires_minutes": 20
            })),
        )
        .await
        .unwrap();
    let id: Uuid = started
        .strip_prefix("Monitor ")
        .and_then(|rest| rest.split(' ').next())
        .unwrap()
        .parse()
        .unwrap();
    assert!(started.contains(" UTC (20 min). "), "{started}");
    let check = start
        .execute(
            &f.context(),
            params(json!({
                "command": f.command("sys.exit(0)"),
                "every_seconds": "10",
                "alert_when": "success"
            })),
        )
        .await
        .unwrap();
    assert!(
        check.contains(" every 10 s and alerts you when it starts succeeding, until ")
            && check.contains(". Its first check succeeded in "),
        "{check}"
    );

    let listed: Vec<serde_json::Value> = serde_json::from_str(
        &control
            .execute(&f.context(), params(json!({"action": "list"})))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(listed.len(), 2);
    let first = listed
        .iter()
        .find(|monitor| monitor["job_id"] == id.to_string())
        .unwrap();
    assert_eq!(first["watching"], "lines that match error");
    assert_eq!(first["alerts"], 0);
    assert_eq!(first["ends_in_minutes"], 20);

    let rearmed = control
        .execute(
            &f.context(),
            params(json!({"action": "rearm", "job_id": id.to_string(), "minutes": 10})),
        )
        .await
        .unwrap();
    assert!(
        rearmed.starts_with(&format!("Monitor {id} now ends at "))
            && rearmed.ends_with(" UTC, in 10 min."),
        "{rearmed}"
    );
    assert_eq!(
        control
            .execute(
                &f.context(),
                params(json!({"action": "stop", "job_id": id.to_string()})),
            )
            .await
            .unwrap(),
        format!("Stopped monitor {id}.")
    );
    assert_eq!(ended(&f, id).await.state, JobState::Cancelled);
    let again = control
        .execute(
            &f.context(),
            params(json!({"action": "stop", "job_id": id.to_string()})),
        )
        .await
        .unwrap_err();
    assert!(
        again
            .to_string()
            .ends_with(&format!("monitor {id} has ended already"))
    );
    let after = control
        .execute(
            &f.context(),
            params(json!({"action": "rearm", "job_id": id.to_string()})),
        )
        .await
        .unwrap_err();
    assert_eq!(
        after.to_string(),
        format!("Execution failed: monitor {id} has ended. monitor_start starts a new one")
    );

    // Parameters that don't go together, or out of range.
    for (bad, says) in [
        (
            json!({"command": "x", "pattern": "e", "every_seconds": 10}),
            "Give one or the other",
        ),
        (
            json!({"command": "x", "alert_when": "success"}),
            "alert_when is for a check",
        ),
        (
            json!({"command": "x", "every_seconds": 10, "alert_when": "sometimes"}),
            "alert_when is success, failure, change or output",
        ),
        (
            json!({"command": "x", "expires_minutes": 31}),
            "expires_minutes is from 1 to 30",
        ),
        (
            json!({"command": "x", "expires_minutes": 0}),
            "expires_minutes is from 1 to 30",
        ),
        (
            json!({"command": "x", "every_seconds": 5}),
            "a check runs every 10 to 600 seconds",
        ),
    ] {
        let error = start
            .execute(&f.context(), params(bad))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(says), "{error}");
    }
    let job = f
        .manager
        .start(
            &f.context(),
            &f.command("time.sleep(60)"),
            Default::default(),
        )
        .unwrap()
        .job_id;
    let stop = control
        .execute(
            &f.context(),
            params(json!({"action": "stop", "job_id": job.to_string()})),
        )
        .await
        .unwrap_err();
    assert!(
        stop.to_string()
            .contains("isn't a monitor. terminal_control cancel stops other jobs")
    );
    assert!(
        control
            .execute(
                &f.context(),
                params(json!({"action": "pause", "job_id": job.to_string()}))
            )
            .await
            .is_err()
    );
    f.manager.cancel(f.owner, job).unwrap();
}

/// `register_tools` registers the monitor tools with the terminal tools,
/// and a runtime offered only the monitor tools gets its monitors' notes.
#[tokio::test]
async fn a_runtime_offered_the_monitor_tools_gets_their_notes() {
    let f = fixture(quick());
    let mut registry = ToolRegistry::new();
    f.manager.register_tools(&mut registry);
    for name in TerminalTool::NAMES.iter().chain(&MonitorTool::NAMES) {
        assert!(registry.get(name).is_some(), "{name}");
    }
    let config = crate::AgentConfig::builder(
        "monitor-test".into(),
        "Tester".into(),
        crate::AgentLLMConfig {
            model: arko_ai::Model::openai_gpt_5_6_luna(),
            generation_config: Default::default(),
        },
    )
    .role_description("test".into())
    .instructions("test".into())
    .tool_config(crate::AgentToolConfig {
        available_tool_names: MonitorTool::NAMES.map(String::from).to_vec(),
        ..Default::default()
    })
    .build()
    .unwrap();
    let mut task = crate::Task::new(config, "test".into(), f.root.clone(), None);
    task.set_state(crate::TaskState::AwaitingUserResponse {
        questions: "pending".into(),
    });
    let id = task.id;
    let (_events, commands, handle) = crate::AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .run_task(task, Arc::new(registry), None, false, None, None, None);
    let binding = f.manager.runtimes.lock().unwrap().get(&id).cloned();
    assert!(
        binding.is_some_and(|binding| binding.sender.upgrade().unwrap().same_channel(&commands))
    );
    commands.send(UserCommand::SuspendTask).await.unwrap();
    timeout(Duration::from_secs(1), handle)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

/// A monitor that ended and was forgotten keeps its record forgotten when
/// its news goes later.
#[tokio::test]
async fn news_of_a_forgotten_monitor_leaves_its_record_alone() {
    let f = fixture(quick());
    let id = monitor(
        &f,
        "time.sleep(0.3)\nprint('error at exit', flush=True)",
        lines("error"),
    )
    .await
    .job_id;
    let done = ended(&f, id).await;
    assert!(f.manager.has_news(f.owner));
    f.manager.forget(f.owner, id).unwrap();
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let text = note(&mut receiver).await;
    assert!(text.contains("\n  error at exit\n"), "{text}");
    let directory = done.log_path.parent().unwrap().to_path_buf();
    assert!(!directory.join("job.json").exists());
    assert!(directory.join("finished.json").exists());
}
