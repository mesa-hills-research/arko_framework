//! The task's latest request (`Task::latest_request`) in the Claude Code
//! runtime: the conversation's own requests, timed from their first
//! streamed token, with the cache lifetime Claude Code's writes show, and
//! none after a compaction until the next request.

use super::*;
use crate::task::RequestRecord;

/// The conversation's requests set it as each one's usage is recorded,
/// with their input and cache counts. The lifetime follows the writes: an
/// hour's write gives an hour, a request that only reads keeps the
/// conversation's, and one that wrote both lifetimes gives the shorter. A
/// subagent's request, flushed last here, leaves it. A request is timed
/// from its first token, so the reply after the pause is timed after it.
#[tokio::test]
async fn the_conversation_s_own_requests_are_its_latest() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 3000, "cached": 0, "written_1h": 2000},
        {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 3100, "cached": 3000},
        {"tool": "Grep", "input": {"pattern": "c"}, "input_tokens": 3300, "cached": 3000,
         "written_1h": 100, "written_5m": 200},
        {"sleep": 0.3},
        {"text": "All done.", "input_tokens": 3400, "cached": 3300},
        {"subagent": {"parent": "toolu_agent", "text": "Found it.", "input_tokens": 450}},
    ]]));
    let task = Task::new(
        replying(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let recorder = Arc::new(Recorder::default());
    let started = chrono::Utc::now();
    let (task, _) = start_with(task, recording(&recorder)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let lifetime = |minutes: u64| Some(Duration::from_secs(minutes * 60));
    let mut seen = Vec::new();
    for (snapshot, _) in recorder.snapshots() {
        if let Some(request) = snapshot.latest_request.latest() {
            let entry = (request.input_tokens, request.cache_lifetime());
            if seen.last() != Some(&entry) {
                seen.push(entry);
            }
        }
    }
    assert_eq!(
        seen,
        [
            (3000, lifetime(60)),
            (3100, lifetime(60)),
            (3300, lifetime(5)),
            (3400, lifetime(5)),
        ]
    );
    let last = task.latest_request.latest().unwrap();
    assert_eq!((last.cached_tokens, last.cache_write_tokens), (3300, 0));
    assert_eq!(task.usage_history.last().unwrap().input_tokens, 450);
    let grep_c = recorder.latest(|snapshot| {
        snapshot
            .latest_request
            .latest()
            .is_some_and(|request| request.input_tokens == 3300)
    });
    let grep_c = grep_c.latest_request.latest().unwrap().sent_at;
    assert!(started <= grep_c);
    assert!(
        last.sent_at - grep_c >= chrono::Duration::milliseconds(300),
        "{}",
        last.sent_at - grep_c
    );
}

/// A response Claude Code got without streaming has no first token to
/// time it, so it is timed when it comes, and is the latest all the same.
#[tokio::test]
async fn a_response_that_wasn_t_streamed_is_the_latest_too() {
    let fake = Fake::new(json!([[
        {"text": "ok", "input_tokens": 900, "cached": 850, "unstreamed": true},
    ]]));
    let task = Task::new(chat_config(&fake), "Say ok.".into(), fake.workspace(), None);
    let (task, _) = start(task).finish().await;
    let request = task.latest_request.latest().expect("a request");
    assert_eq!((request.input_tokens, request.cached_tokens), (900, 850));
}

/// A compaction replaces the conversation the cache held, so the task has
/// no request until the next one is answered. A task saved in between has
/// none.
#[tokio::test]
async fn a_compaction_leaves_no_request_until_the_next() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 900},
        {"system": {"subtype": "compact_boundary",
                    "compact_metadata": {"trigger": "auto", "pre_tokens": 950, "post_tokens": 120}}},
        complete("done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let recorder = Arc::new(Recorder::default());
    let (task, _) = start_with(task, recording(&recorder)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let compacted = recorder
        .snapshots()
        .into_iter()
        .map(|(snapshot, _)| snapshot)
        .find(|snapshot| snapshot.conversation_usage_start == 1)
        .expect("a snapshot after the compaction");
    assert_eq!(compacted.latest_request, RequestRecord::NoRequest);
    assert!(matches!(task.latest_request, RequestRecord::Latest(_)));
}
