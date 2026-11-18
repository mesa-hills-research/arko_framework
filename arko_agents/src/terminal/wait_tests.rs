//! `terminal_control wait`, covering how long it lasts, the ceiling a longer
//! request is cut to, and what ends it sooner.

use super::monitor::clock_seconds;
use super::tests::Fixture;
use super::tools::describe;
use super::*;
use crate::tool::{ContextSource, PromptCacheWindow, Tool};
use arko_ai::PromptCacheLifetime::{self, Disabled, Known, Unknown};
use serde_json::{Value, json};
use std::time::SystemTime;

const MINUTE: Duration = Duration::from_secs(60);

fn params(value: Value) -> HashMap<String, Value> {
    serde_json::from_value(value).unwrap()
}

/// A job that prints nothing for a minute, started by the fixture's owner.
fn silent(f: &Fixture) -> TerminalSnapshot {
    f.manager
        .start(
            &f.context(),
            &f.command("time.sleep(60)"),
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

/// The result of waiting for `job` for `wait_ms` in a context with `cache`.
async fn wait(
    f: &Fixture,
    job: &TerminalSnapshot,
    wait_ms: u64,
    cache: PromptCacheWindow,
) -> Value {
    let mut context = f.context();
    context.prompt_cache = cache;
    let result = control(f)
        .execute(
            &context,
            params(json!({
                "action": "wait",
                "job_id": job.job_id.to_string(),
                "revision": job.revision.to_string(),
                "wait_ms": wait_ms.to_string(),
            })),
        )
        .await
        .unwrap();
    serde_json::from_str(&result).unwrap()
}

fn used_now(lifetime: PromptCacheLifetime) -> PromptCacheWindow {
    PromptCacheWindow::new(lifetime, Some(SystemTime::now()))
}

/// A request above the ceiling is cut to it for each kind of lifetime, and
/// the result says why. The clock is paused, so the long waits take no time,
/// and the job prints nothing.
#[tokio::test(start_paused = true)]
async fn a_wait_above_the_ceiling_is_cut_and_the_result_says_so() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let four_hours = 4 * 3600 * 1000;
    // A known lifetime counts from the latest request. The ceiling is its
    // lifetime less a tenth of it, or 30 seconds.
    for (lifetime, waited) in [
        (Known(5 * MINUTE), "4 min 30 s"),
        (Known(30 * MINUTE), "27 min"),
        (Known(60 * MINUTE), "54 min"),
    ] {
        let cache = used_now(lifetime);
        let note = wait(&f, &job, four_hours, cache).await["note"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            note,
            format!(
                "The wait ended after {waited}, not the 4 h you asked for, because your prompt cache lapses at {}. Call terminal_control wait again to keep waiting.",
                clock_seconds(cache.lapses_at().unwrap())
            ),
            "{lifetime:?}"
        );
    }
    // With no lifetime to keep, the ceiling is the 5-minute cache's, fixed.
    for lifetime in [Unknown, Disabled] {
        let note = wait(&f, &job, four_hours, used_now(lifetime)).await["note"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            note,
            "The wait ended after 4 min 30 s, not the 4 h you asked for, the longest a wait lasts here. Call terminal_control wait again to keep waiting.",
            "{lifetime:?}"
        );
    }
}

/// Time the model took before calling the tool counts against a known
/// lifetime, so the ceiling shrinks, and a request inside it is not cut.
#[tokio::test(start_paused = true)]
async fn a_known_lifetime_counts_from_the_latest_request() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let cache = PromptCacheWindow::new(Known(5 * MINUTE), Some(SystemTime::now() - 2 * MINUTE));
    let result = wait(&f, &job, 10 * 60 * 1000, cache).await;
    let note = result["note"].as_str().unwrap();
    assert!(
        note.starts_with("The wait ended after 2 min 30 s, not the 10 min you asked for,"),
        "{note}"
    );
    // Inside the ceiling, and by default, the wait just ends with the job as it was.
    for asked in [60_000, 100_000] {
        let result = wait(&f, &job, asked, cache).await;
        assert!(result.get("note").is_none(), "{result}");
        assert_eq!(result["revision"], job.revision);
        assert_eq!(result["state"], "running");
    }
}

/// A wait that gets no `wait_ms` lasts a second, as before.
#[tokio::test(start_paused = true)]
async fn a_wait_without_a_time_lasts_one_second() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let started = tokio::time::Instant::now();
    let result = control(&f)
        .execute(
            &f.context(),
            params(json!({
                "action": "wait",
                "job_id": job.job_id.to_string(),
                "revision": job.revision.to_string(),
            })),
        )
        .await
        .unwrap();
    assert!(
        (Duration::from_secs(1)..Duration::from_secs(2)).contains(&started.elapsed()),
        "{:?}",
        started.elapsed()
    );
    assert!(!result.contains("note"), "{result}");
}

/// New output ends a long wait at once, with no word about a cut, whatever
/// the ceiling.
#[tokio::test]
async fn new_output_ends_a_long_wait_early() {
    let f = Fixture::new(|_| {});
    let job = f
        .manager
        .start(
            &f.context(),
            &f.command("time.sleep(.3); print('ready', flush=True); time.sleep(60)"),
            StartOptions {
                terminal: false,
                ..Default::default()
            },
        )
        .unwrap();
    let started = std::time::Instant::now();
    // 4 hours, over the 27 minutes of a 30-minute cache.
    let result = wait(&f, &job, 4 * 3600 * 1000, used_now(Known(30 * MINUTE))).await;
    assert!(started.elapsed() < Duration::from_secs(20));
    assert!(
        result["revision"].as_u64().unwrap() > job.revision,
        "{result}"
    );
    assert!(result.get("note").is_none(), "{result}");
}

/// Context that arrives for the agent and a stop end a long wait too.
#[tokio::test]
async fn context_and_a_stop_end_a_long_wait() {
    let f = Fixture::new(|_| {});
    let job = silent(&f);
    let wait_ms = 4 * 3600 * 1000;
    let call = || {
        params(json!({
            "action": "wait",
            "job_id": job.job_id.to_string(),
            "revision": job.revision.to_string(),
            "wait_ms": wait_ms.to_string(),
        }))
    };
    let tool = control(&f);

    let source = ContextSource::new();
    let mut context = f.context();
    context.context_arrived = source.signal();
    context.prompt_cache = used_now(Known(30 * MINUTE));
    let notifier = source.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        notifier.notify();
    });
    let started = std::time::Instant::now();
    let result: Value =
        serde_json::from_str(&tool.execute(&context, call()).await.unwrap()).unwrap();
    assert!(started.elapsed() < Duration::from_secs(20));
    assert_eq!(result["revision"], job.revision);
    assert!(result.get("note").is_none(), "{result}");

    let stop = crate::stop::StopSource::new();
    let mut context = f.context();
    context.stop = stop.signal();
    context.prompt_cache = used_now(Known(30 * MINUTE));
    let stopper = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        stop.stop("Task suspended.");
    });
    let started = std::time::Instant::now();
    tool.execute(&context, call()).await.unwrap();
    assert!(started.elapsed() < Duration::from_secs(20));
    stopper.await.unwrap();
}

/// The description says the agent may ask for a longer wait, and what bounds
/// it.
#[test]
fn the_description_says_how_long_a_wait_may_be() {
    let f = Fixture::new(|_| {});
    let definition = control(&f).definition();
    let description = &definition.description;
    for part in [
        "`wait_ms`",
        "Ask for longer",
        "prompt cache's lifetime less a margin (4 min 30 s for a 5-minute cache)",
        "cut to that",
        "when your cache lapses",
    ] {
        assert!(description.contains(part), "{part}: {description}");
    }
    assert!(!description.contains("30 seconds"), "{description}");
    let wait_ms = definition
        .parameters
        .iter()
        .find(|parameter| parameter.name == "wait_ms")
        .unwrap();
    assert!(!wait_ms.description.contains("30000"), "{wait_ms:?}");
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
