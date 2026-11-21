//! `terminal_control wait`, covering how long it lasts, the ceiling a longer
//! request is cut to, and what ends it sooner.
//!
//! Most tests pause the clock, so a wait of hours takes no time, and check
//! how long the wait lasted on that clock.

use super::monitor::clock_seconds;
use super::tests::Fixture;
use super::tools::describe;
use super::*;
use crate::tool::{ContextSource, PromptCacheWindow, Tool};
use arko_ai::PromptCacheLifetime::{self, Disabled, Known, Unknown};
use serde_json::{Value, json};
use tokio::time::Instant;

const MINUTE: Duration = Duration::from_secs(60);
const HOUR: Duration = Duration::from_secs(3600);

fn params(value: Value) -> HashMap<String, Value> {
    serde_json::from_value(value).unwrap()
}

/// A job that prints nothing for a minute, started by the fixture's owner.
fn silent(f: &Fixture) -> TerminalSnapshot {
    started(f, "time.sleep(60)")
}

fn started(f: &Fixture, code: &str) -> TerminalSnapshot {
    f.manager
        .start(
            &f.context(),
            &f.command(code),
            StartOptions {
                terminal: false,
                ..Default::default()
            },
        )
        .unwrap()
}

/// `terminal_control`, which waits.
fn control(f: &Fixture) -> TerminalTool {
    TerminalTool::all(f.manager.clone()).remove(3)
}

fn call(job: &TerminalSnapshot, wait_ms: Option<u64>) -> HashMap<String, Value> {
    let mut call = json!({
        "action": "wait",
        "job_id": job.job_id.to_string(),
        "revision": job.revision.to_string(),
    });
    if let Some(wait_ms) = wait_ms {
        call["wait_ms"] = wait_ms.to_string().into();
    }
    params(call)
}

/// The result of waiting for `job` for `wait_ms` in a context with `cache`,
/// and how long it took on the Tokio clock.
async fn wait(
    f: &Fixture,
    job: &TerminalSnapshot,
    wait_ms: Option<u64>,
    cache: PromptCacheWindow,
) -> (Value, Duration) {
    let mut context = f.context();
    context.prompt_cache = cache;
    let started = Instant::now();
    let result = control(f)
        .execute(&context, call(job, wait_ms))
        .await
        .unwrap();
    (serde_json::from_str(&result).unwrap(), started.elapsed())
}

fn used_now(lifetime: PromptCacheLifetime) -> PromptCacheWindow {
    PromptCacheWindow::new(lifetime, Some(Instant::now()))
}

fn used_ago(lifetime: PromptCacheLifetime, ago: Duration) -> PromptCacheWindow {
    PromptCacheWindow::new(lifetime, Instant::now().checked_sub(ago))
}

/// `elapsed` is `seconds`, give or take the clock's millisecond.
#[track_caller]
fn assert_lasted(elapsed: Duration, seconds: u64) {
    let seconds = Duration::from_secs(seconds);
    assert!(
        (seconds..seconds + Duration::from_millis(50)).contains(&elapsed),
        "lasted {elapsed:?}, not {seconds:?}"
    );
}

fn note(result: &Value) -> &str {
    result["note"].as_str().expect("a note")
}

/// A wait over the old 30 seconds, inside the cache, lasts as long as the
/// agent asks and says nothing of a cut. Without `wait_ms`, it defaults to the
/// prompt-cache cap (or 30 seconds if no cache deadline is set).
#[tokio::test(start_paused = true)]
async fn a_wait_inside_the_cache_lasts_as_long_as_asked() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    for seconds in [100, 1500] {
        let (result, elapsed) =
            wait(&f, &job, Some(seconds * 1000), used_now(Known(30 * MINUTE))).await;
        assert_lasted(elapsed, seconds);
        assert!(result.get("note").is_none(), "{result}");
        assert_eq!(result["revision"], job.revision);
        assert_eq!(result["state"], "running");
    }
    let (result, elapsed) = wait(&f, &job, None, used_now(Known(30 * MINUTE))).await;
    assert_lasted(elapsed, 27 * 60);
    assert!(result.get("note").is_none(), "{result}");
    let (result, elapsed) = wait(&f, &job, None, used_now(Unknown)).await;
    assert_lasted(elapsed, 270);
    assert!(result.get("note").is_none(), "{result}");
    #[cfg(unix)]
    {
        // A lifetime too far to count sets no deadline, falling back to 30 s.
        let (result, elapsed) = wait(
            &f,
            &job,
            None,
            used_now(Known(Duration::from_secs(u64::MAX))),
        )
        .await;
        assert_lasted(elapsed, 30);
        assert!(result.get("note").is_none(), "{result}");
    }
}

/// A request above the ceiling is cut to it for each kind of lifetime, and
/// the result says why.
#[tokio::test(start_paused = true)]
async fn a_wait_above_the_ceiling_is_cut_and_the_result_says_so() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let four_hours = 4 * 3600 * 1000;
    // A known lifetime counts from the latest request. The ceiling is its
    // lifetime less a tenth of it, or 30 seconds.
    for (lifetime, seconds, said) in [
        (Known(5 * MINUTE), 270, "4 min 30 s"),
        (Known(30 * MINUTE), 27 * 60, "27 min"),
        (Known(HOUR), 54 * 60, "54 min"),
    ] {
        let cache = used_now(lifetime);
        let (result, elapsed) = wait(&f, &job, Some(four_hours), cache).await;
        assert_lasted(elapsed, seconds);
        assert_eq!(
            note(&result),
            format!(
                "The wait ended after {said}, not the 4 h you asked for, because your prompt cache lapses at {}. Call terminal_control wait again to keep waiting.",
                clock_seconds(cache.lapses_at().unwrap())
            ),
            "{lifetime:?}"
        );
    }
    // With no lifetime to keep, the ceiling is the 5-minute cache's, from
    // the start of the wait however long ago the request was.
    for lifetime in [Unknown, Disabled] {
        let (result, elapsed) =
            wait(&f, &job, Some(four_hours), used_ago(lifetime, 10 * MINUTE)).await;
        assert_lasted(elapsed, 270);
        assert_eq!(
            note(&result),
            "The wait ended after 4 min 30 s, not the 4 h you asked for, the longest a wait lasts here. Call terminal_control wait again to keep waiting.",
            "{lifetime:?}"
        );
    }
}

/// The time the model took before it called the tool counts against a known
/// lifetime, and a cache already past its margin leaves no wait at all. A
/// request whose time isn't known (a task loaded from a save) leaves none
/// either, so it can't be given a fresh budget.
#[tokio::test(start_paused = true)]
async fn a_known_lifetime_counts_from_the_latest_request() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let (result, elapsed) = wait(
        &f,
        &job,
        Some(600_000),
        used_ago(Known(5 * MINUTE), 2 * MINUTE),
    )
    .await;
    assert_lasted(elapsed, 150);
    assert!(
        note(&result).starts_with("The wait ended after 2 min 30 s, not the 10 min you asked for,"),
        "{result}"
    );
    // Inside what is left, no cut.
    let (result, elapsed) = wait(
        &f,
        &job,
        Some(100_000),
        used_ago(Known(5 * MINUTE), 2 * MINUTE),
    )
    .await;
    assert_lasted(elapsed, 100);
    assert!(result.get("note").is_none(), "{result}");
    // Spent.
    let (result, elapsed) = wait(
        &f,
        &job,
        Some(600_000),
        used_ago(Known(5 * MINUTE), Duration::from_secs(280)),
    )
    .await;
    assert_eq!(elapsed, Duration::ZERO);
    assert!(
        note(&result).starts_with("The wait ended after less than a second, not the 10 min"),
        "{result}"
    );
    // Not known.
    let (result, elapsed) = wait(
        &f,
        &job,
        Some(600_000),
        PromptCacheWindow::new(Known(5 * MINUTE), None),
    )
    .await;
    assert_eq!(elapsed, Duration::ZERO);
    assert!(
        note(&result).ends_with(
            "because it can't tell when your prompt cache lapses. Call terminal_control wait again to keep waiting."
        ),
        "{result}"
    );
}

/// Calls made from one request share its cache. Time that passes during the
/// first call's wait is gone from the second's.
#[tokio::test(start_paused = true)]
async fn waits_of_one_request_share_its_cache() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let cache = used_now(Known(5 * MINUTE));
    let (result, elapsed) = wait(&f, &job, Some(100_000), cache).await;
    assert_lasted(elapsed, 100);
    assert!(result.get("note").is_none(), "{result}");
    // 170 of the 270 seconds are left, and no more.
    let (result, elapsed) = wait(&f, &job, Some(4 * 3600 * 1000), cache).await;
    assert_lasted(elapsed, 170);
    assert!(
        note(&result).starts_with("The wait ended after 2 min 50 s, not the 4 h you asked for,"),
        "{result}"
    );
    // Only the margin is left, and no wait.
    let (_, elapsed) = wait(&f, &job, Some(60_000), cache).await;
    assert_eq!(elapsed, Duration::ZERO);
}

/// New output ends a long wait at once, with no word about a cut, whatever
/// the ceiling.
#[tokio::test]
async fn new_output_ends_a_long_wait_early() {
    let f = Fixture::new(|_| {});
    let job = started(
        &f,
        "time.sleep(.3); print('ready', flush=True); time.sleep(60)",
    );
    let started = std::time::Instant::now();
    // 4 hours, over the 27 minutes of a 30-minute cache.
    let (result, _) = wait(
        &f,
        &job,
        Some(4 * 3600 * 1000),
        used_now(Known(30 * MINUTE)),
    )
    .await;
    assert!(started.elapsed() < Duration::from_secs(20));
    assert!(
        result["revision"].as_u64().unwrap() > job.revision,
        "{result}"
    );
    assert!(result.get("note").is_none(), "{result}");
}

/// An update that leaves the job as it was, such as the first part of an
/// input declaration, doesn't end the wait, and the note that says it ran
/// to the ceiling comes with the ceiling, not with the update.
#[tokio::test]
async fn an_update_that_changes_nothing_doesnt_end_a_wait() {
    let f = Fixture::new(|_| {});
    let job = started(
        &f,
        "fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nos.write(fd,b'{\"type\":\"input_re')\ntime.sleep(60)",
    );
    // Three seconds of a 5-minute cache's ceiling are left.
    let cache = used_ago(Known(5 * MINUTE), Duration::from_secs(267));
    let (result, elapsed) = wait(&f, &job, Some(60_000), cache).await;
    assert!(elapsed >= Duration::from_millis(2900), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(20), "{elapsed:?}");
    assert_eq!(result["revision"], job.revision);
    assert!(
        note(&result).starts_with("The wait ended after 3 s, not the 1 min you asked for,"),
        "{result}"
    );
}

/// The manager's wait reports whether its time ran out, and its time runs
/// from the start through updates that change nothing.
#[tokio::test(start_paused = true)]
async fn the_manager_says_whether_the_time_ran_out() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let started = Instant::now();
    let (snapshot, timed_out) = f
        .manager
        .wait_for_change(
            f.owner,
            job.job_id,
            job.revision,
            Some(started + Duration::from_secs(90)),
        )
        .await
        .unwrap();
    assert!(timed_out);
    assert_lasted(started.elapsed(), 90);
    assert_eq!(snapshot.revision, job.revision);
    // A revision other than the one the caller saw is a change at once.
    let (_, timed_out) = f
        .manager
        .wait_for_change(
            f.owner,
            job.job_id,
            job.revision + 1,
            Some(started + Duration::from_secs(900)),
        )
        .await
        .unwrap();
    assert!(!timed_out);
    assert_lasted(started.elapsed(), 90);
}

/// The wait ends at the deadline the caller fixed, so time that passes
/// between fixing it and the wait's own start comes out of the wait. That is
/// what looking up the job and locking its record take. A deadline that has
/// passed ends the wait at once, and none has no end.
#[tokio::test(start_paused = true)]
async fn the_manager_waits_until_a_deadline_fixed_earlier() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let wait = |deadline| {
        f.manager
            .wait_for_change(f.owner, job.job_id, job.revision, deadline)
    };
    let fixed = Instant::now();
    tokio::time::advance(Duration::from_secs(4)).await;
    let (_, timed_out) = wait(Some(fixed + Duration::from_secs(10))).await.unwrap();
    assert!(timed_out);
    assert_lasted(fixed.elapsed(), 10);
    // Passed already.
    let before = Instant::now();
    let (_, timed_out) = wait(Some(fixed)).await.unwrap();
    assert!(timed_out);
    assert_eq!(before.elapsed(), Duration::ZERO);
    // No deadline ends with a change.
    let waiting = tokio::spawn({
        let manager = f.manager.clone();
        let (owner, id, revision) = (f.owner, job.job_id, job.revision);
        async move { manager.wait_for_change(owner, id, revision, None).await }
    });
    tokio::time::sleep(Duration::from_secs(3600)).await;
    assert!(!waiting.is_finished());
    waiting.abort();
}

/// The tool ends its wait at its deadline when the job's record is locked
/// meanwhile, as it is while the runner writes. The lock isn't part of the
/// wait's time.
#[tokio::test(flavor = "multi_thread")]
async fn a_locked_record_doesnt_extend_a_wait() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let held = f.manager.owned(f.owner, job.job_id).unwrap();
    let (locked, is_locked) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let _data = held.data.lock().unwrap();
        locked.send(()).unwrap();
        // Long enough to see, and under the wait.
        std::thread::sleep(Duration::from_millis(400));
    });
    is_locked.recv().unwrap();
    let cache = PromptCacheWindow::new(
        Known(5 * MINUTE),
        Instant::now().checked_sub(Duration::from_millis(269_400)),
    );
    // A wait cut to 600 ms of the cache, started with the record locked.
    let started = std::time::Instant::now();
    let (result, _) = wait(&f, &job, Some(60_000), cache).await;
    let waited = started.elapsed();
    holder.join().unwrap();
    assert!(waited >= Duration::from_millis(550), "{waited:?}");
    assert!(waited < Duration::from_millis(900), "{waited:?}");
    assert!(
        note(&result).starts_with("The wait ended after"),
        "{result}"
    );
}

/// Context that arrives for the agent, before or during the wait, and a
/// stop end a long wait too.
#[tokio::test]
async fn context_and_a_stop_end_a_long_wait() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let wait_ms = 4 * 3600 * 1000;
    let tool = control(&f);
    let cache = used_now(Known(30 * MINUTE));
    let ends_soon = |started: std::time::Instant| {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{:?}",
            started.elapsed()
        );
    };

    // Context that arrived already.
    let source = ContextSource::new();
    let mut context = f.context();
    context.context_arrived = source.signal();
    context.prompt_cache = cache;
    source.notify();
    let started = std::time::Instant::now();
    let result = tool
        .execute(&context, call(&job, Some(wait_ms)))
        .await
        .unwrap();
    ends_soon(started);
    assert!(!result.contains("note"), "{result}");

    // Context that arrives while the wait goes on.
    let source = ContextSource::new();
    let mut context = f.context();
    context.context_arrived = source.signal();
    context.prompt_cache = cache;
    let notifier = source.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        notifier.notify();
    });
    let started = std::time::Instant::now();
    let result: Value = serde_json::from_str(
        &tool
            .execute(&context, call(&job, Some(wait_ms)))
            .await
            .unwrap(),
    )
    .unwrap();
    ends_soon(started);
    assert_eq!(result["revision"], job.revision);
    assert!(result.get("note").is_none(), "{result}");

    // A stop.
    let stop = crate::stop::StopSource::new();
    let mut context = f.context();
    context.stop = stop.signal();
    context.prompt_cache = cache;
    let stopper = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        stop.stop("Task suspended.");
    });
    let started = std::time::Instant::now();
    tool.execute(&context, call(&job, Some(wait_ms)))
        .await
        .unwrap();
    ends_soon(started);
    stopper.await.unwrap();
}

/// The description says waits are bounded by the prompt cache and emits enum
/// values for `action`.
#[test]
fn the_description_says_how_long_a_wait_may_be() {
    let f = Fixture::new(|_| {});
    let definition = control(&f).definition();
    let description = &definition.description;
    assert!(description.contains("prompt cache"), "{description}");
    assert!(!description.contains("30 seconds"), "{description}");
    let wait_ms = definition
        .parameters
        .iter()
        .find(|parameter| parameter.name == "wait_ms")
        .unwrap();
    assert!(!wait_ms.description.contains("30000"), "{wait_ms:?}");
    let schema = definition.to_json_schema();
    assert_eq!(
        schema["properties"]["action"]["enum"],
        json!(["list", "wait", "cancel", "forget"])
    );
}

#[test]
fn durations_read_as_seconds_minutes_or_hours() {
    let seconds = Duration::from_secs;
    for (duration, text) in [
        (Duration::from_millis(300), "less than a second"),
        (seconds(1), "1 s"),
        (Duration::from_millis(269_700), "4 min 30 s"),
        (seconds(270), "4 min 30 s"),
        (seconds(27 * 60), "27 min"),
        (seconds(4 * 3600), "4 h"),
        (seconds(10 * 3600 + 48 * 60 + 13), "10 h 48 min"),
    ] {
        assert_eq!(describe(duration), text);
    }
}
