//! A test-only trace of the agent loop: every event it emits and every task
//! it reports at a safe point, one line each, normalized so that two runs of
//! a test give the same lines. It pins the order of events and snapshots,
//! and the task at each snapshot, for changes to the loop that must keep its
//! behaviour.
//!
//! A loop is traced when its test is: a test that calls [`record`] (the
//! golden scenarios of `golden_tests.rs`), or every test when one of these
//! variables is set:
//!
//! - `ARKO_LOOP_TRACE_DIR=<dir>`: each test's trace goes to
//!   `<dir>/<test>.trace`, appended line by line (clear the folder first).
//! - `ARKO_LOOP_TRACE_STDOUT=1`: each line goes to standard output as
//!   `@@looptrace <test> <line>`, past libtest's capture, for a run whose
//!   files stay on another machine.
//!
//! `ARKO_LOOP_TRACE_FULL=1` keeps long strings whole and prints each
//! snapshot's task in full instead of its hash, to find what differs.
//!
//! A line is `E <event>` (the event as JSON) or `S <counts> state=<state>
//! task=<hash>`, where the hash covers the whole task except its agent
//! configuration (and its model, named by id). Normalizing sorts JSON keys,
//! replaces UUIDs with `id0`, `id1`, … in the order a test's trace first
//! shows them, masks wall-clock times, the time to a usage-limit reset and
//! the awaiting state a command timeout prints with `{:?}` (its maps print
//! in a different order each run), and shortens strings over
//! [`LONG_STRING`] characters to their head and a hash.
//!
//! The test is the thread the loop is created on: libtest runs each test on
//! a thread named after it, and the loop's tests use tokio's current-thread
//! runtime. The loops a test creates share its trace (and its UUID labels).

use crate::runtime::event::AgentEvent;
use crate::task::Task;
use regex::Regex;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, LazyLock, Mutex};

/// Strings longer than this many characters are shortened to their head and
/// a hash, unless `ARKO_LOOP_TRACE_FULL` is set.
const LONG_STRING: usize = 400;
/// The characters a shortened string keeps.
const LONG_STRING_HEAD: usize = 80;

/// The traces of the tests being traced, by test name.
static TESTS: LazyLock<Mutex<HashMap<String, Arc<TestTrace>>>> = LazyLock::new(Default::default);

static UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        .unwrap()
});
static DATETIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})").unwrap()
});
/// A usage-limit reset, given relative to the wall clock.
static RESETS_IN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"resets in about (\d+[dhms] ?)+").unwrap());
/// The awaiting state a command timeout prints with `{:?}`: its maps print
/// in a different order each run.
static TIMEOUT_STATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)(waiting for user command in state ).*").unwrap());

/// Object keys whose values are wall-clock times.
fn is_time_key(key: &str) -> bool {
    key.ends_with("_unix_ms") || key == "start_time" || key == "end_time"
}

fn env_set(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| !value.is_empty())
}

fn full() -> bool {
    env_set("ARKO_LOOP_TRACE_FULL")
}

/// The longest string a line keeps whole.
fn longest() -> Option<usize> {
    (!full()).then_some(LONG_STRING)
}

fn current_test() -> String {
    std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_string()
}

/// The trace of the test a loop runs in, if that test is traced.
#[derive(Default)]
pub(super) struct LoopTrace(Option<Arc<TestTrace>>);

impl LoopTrace {
    /// The trace of the test running on this thread: on when the test
    /// called [`record`] or a trace variable is set, off otherwise.
    pub(super) fn for_current_test() -> Self {
        let name = current_test();
        let mut tests = TESTS.lock().unwrap();
        if let Some(trace) = tests.get(&name) {
            return Self(Some(trace.clone()));
        }
        if !env_set("ARKO_LOOP_TRACE_DIR") && !env_set("ARKO_LOOP_TRACE_STDOUT") {
            return Self(None);
        }
        let trace = Arc::new(TestTrace::new(name.clone()));
        tests.insert(name, trace.clone());
        Self(Some(trace))
    }

    pub(super) fn event(&self, event: &AgentEvent) {
        if let Some(trace) = &self.0 {
            trace.event(event);
        }
    }

    pub(super) fn snapshot(&self, task: &Task, durable: bool) {
        if let Some(trace) = &self.0 {
            trace.snapshot(task, durable);
        }
    }
}

/// Traces the loops the current test creates from now on, whether or not a
/// trace variable is set, and returns its trace.
pub(super) fn record() -> Arc<TestTrace> {
    let name = current_test();
    TESTS
        .lock()
        .unwrap()
        .entry(name.clone())
        .or_insert_with(|| Arc::new(TestTrace::new(name)))
        .clone()
}

/// One test's trace.
pub(super) struct TestTrace {
    name: String,
    state: Mutex<Normalizer>,
}

impl TestTrace {
    fn new(name: String) -> Self {
        Self {
            name,
            state: Mutex::new(Normalizer::default()),
        }
    }

    /// Replaces `literal` with `replacement` in every string from now on
    /// (before UUIDs are labelled): a temporary path, say.
    pub(super) fn mask(&self, literal: &str, replacement: &str) {
        self.state
            .lock()
            .unwrap()
            .literals
            .push((literal.to_string(), replacement.to_string()));
    }

    /// The lines traced so far.
    pub(super) fn lines(&self) -> Vec<String> {
        self.state.lock().unwrap().lines.clone()
    }

    /// `value` normalized as the trace's lines are (UUID labels shared with
    /// them), with strings over `longest` characters shortened to their head
    /// and a hash.
    pub(super) fn normalized(&self, value: Value, longest: usize) -> Value {
        self.state.lock().unwrap().value(value, Some(longest))
    }

    fn event(&self, event: &AgentEvent) {
        let value = serde_json::to_value(event).expect("an event serializes");
        let mut state = self.state.lock().unwrap();
        let line = format!("E {}", state.value(value, longest()));
        self.push(&mut state, line);
    }

    fn snapshot(&self, task: &Task, durable: bool) {
        let mut state = self.state.lock().unwrap();
        let current_state = serde_json::to_value(&task.current_state).expect("a state serializes");
        let current_state = state.value(current_state, longest());
        let digest = state.value(task_digest(task), None).to_string();
        let digest = if full() {
            digest
        } else {
            format!("{:016x}", fnv1a(digest.as_bytes()))
        };
        let line = format!(
            "S durable={} msgs={} native_results={} native_calls={} text_results={} running={} injections={} handovers={} state={current_state} task={digest}",
            u8::from(durable),
            task.conversation.messages.len(),
            task.pending_native_tool_results.len(),
            task.pending_native_tool_calls.len(),
            task.pending_text_tool_results.len(),
            task.running_tool_calls.len(),
            task.pending_context_injections.len(),
            task.handovers.len(),
        );
        self.push(&mut state, line);
    }

    fn push(&self, state: &mut Normalizer, line: String) {
        if let Some(dir) = std::env::var_os("ARKO_LOOP_TRACE_DIR").filter(|dir| !dir.is_empty()) {
            let path = std::path::Path::new(&dir).join(format!("{}.trace", file_name(&self.name)));
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            writeln!(file, "{line}").unwrap();
        }
        if env_set("ARKO_LOOP_TRACE_STDOUT") {
            // Past libtest's capture, which only `print!` honours.
            let _ = writeln!(std::io::stdout().lock(), "@@looptrace {} {line}", self.name);
        }
        state.lines.push(line);
    }
}

/// A test name as a file name.
pub(super) fn file_name(test: &str) -> String {
    test.replace("::", ".")
}

/// What a snapshot's hash covers: the task without its agent configuration,
/// which a run doesn't change, and with its conversation's model named by
/// id (model data such as prices changes between versions).
fn task_digest(task: &Task) -> Value {
    let mut value = serde_json::to_value(task).expect("a task serializes");
    if let Some(task) = value.as_object_mut() {
        task.remove("agent_config");
        if let Some(conversation) = task.get_mut("conversation").and_then(Value::as_object_mut) {
            conversation.insert("model".into(), Value::String(task_model(conversation)));
        }
    }
    value
}

fn task_model(conversation: &Map<String, Value>) -> String {
    conversation
        .get("model")
        .and_then(|model| model.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `value` as JSON with every object's keys sorted (maps serialized from a
/// `HashMap` come in a different order each run).
fn canonical(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by_key(|(key, _)| *key);
                Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key.clone(), sorted(value)))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(value).to_string()
}

/// FNV-1a, 64 bits: a hash that is the same in every run and on every
/// platform.
pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A test's normalizing state, and its lines.
#[derive(Default)]
struct Normalizer {
    /// UUIDs (lowercase) by label, in the order first seen.
    labels: HashMap<String, String>,
    literals: Vec<(String, String)>,
    lines: Vec<String>,
}

impl Normalizer {
    /// `value` normalized, with strings over `longest` characters shortened.
    fn value(&mut self, value: Value, longest: Option<usize>) -> Value {
        match value {
            Value::String(text) => Value::String(self.string(&text, longest)),
            Value::Array(items) => Value::Array(
                items
                    .into_iter()
                    .map(|item| self.value(item, longest))
                    .collect(),
            ),
            Value::Object(map) => Value::Object(self.object(map, longest)),
            other => other,
        }
    }

    /// An object with its keys sorted. Its entries are normalized in an
    /// order that doesn't depend on UUIDs not labelled yet (a map keyed by
    /// call ids, say), so that labels are given in the same order each run.
    fn object(&mut self, map: Map<String, Value>, longest: Option<usize>) -> Map<String, Value> {
        let mut entries: Vec<((String, String), String, Value)> = map
            .into_iter()
            .map(|(key, value)| {
                let order = (self.shape(&key), self.shape(&canonical(&value)));
                (order, key, value)
            })
            .collect();
        entries.sort_by(|(a, ..), (b, ..)| a.cmp(b));
        let mut normalized = Vec::with_capacity(entries.len());
        for (_, key, value) in entries {
            let value = if is_time_key(&key) && !value.is_null() {
                Value::String("<time>".into())
            } else {
                self.value(value, longest)
            };
            normalized.push((self.string(&key, None), value));
        }
        normalized.sort_by(|(a, _), (b, _)| a.cmp(b));
        normalized.into_iter().collect()
    }

    /// `text` with the UUIDs already labelled replaced by their labels and
    /// the others by `<uuid>`, labelling none.
    fn shape(&self, text: &str) -> String {
        let text = self.masked(text);
        UUID.replace_all(&text, |found: &regex::Captures<'_>| {
            self.labels
                .get(&found[0].to_ascii_lowercase())
                .cloned()
                .unwrap_or_else(|| "<uuid>".into())
        })
        .into_owned()
    }

    fn masked(&self, text: &str) -> String {
        let mut text = text.to_string();
        for (literal, replacement) in &self.literals {
            if !literal.is_empty() {
                text = text.replace(literal.as_str(), replacement);
            }
        }
        text
    }

    fn string(&mut self, text: &str, longest: Option<usize>) -> String {
        let text = self.masked(text);
        let labels = &mut self.labels;
        let text = UUID.replace_all(&text, |found: &regex::Captures<'_>| {
            let next = labels.len();
            labels
                .entry(found[0].to_ascii_lowercase())
                .or_insert_with(|| format!("id{next}"))
                .clone()
        });
        let text = DATETIME.replace_all(&text, "<datetime>");
        let text = RESETS_IN.replace_all(&text, "resets in about <duration>");
        let text = TIMEOUT_STATE.replace_all(&text, "${1}<state>.");
        let characters = text.chars().count();
        if longest.is_none_or(|longest| characters <= longest) {
            return text.into_owned();
        }
        let head: String = text.chars().take(LONG_STRING_HEAD).collect();
        format!(
            "{head}…<{characters} chars, {:016x}>",
            fnv1a(text.as_bytes())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids_are_labelled_in_the_order_first_seen() {
        let mut normalizer = Normalizer::default();
        let a = "6f1c1a2e-1111-4aaa-8bbb-000000000001";
        let b = "6f1c1a2e-2222-4aaa-8bbb-000000000002";
        let value = serde_json::json!([format!("call {b}"), {"id": a, "other": b}]);
        assert_eq!(
            normalizer.value(value, Some(LONG_STRING)),
            serde_json::json!(["call id0", {"id": "id1", "other": "id0"}])
        );
    }

    /// A map keyed by UUIDs not seen yet is labelled by its entries'
    /// contents, not by the UUIDs, whatever its order.
    #[test]
    fn maps_keyed_by_new_uuids_are_labelled_by_content() {
        let a = "6f1c1a2e-1111-4aaa-8bbb-000000000001";
        let b = "0f1c1a2e-2222-4aaa-8bbb-000000000002";
        for (first, second) in [(a, b), (b, a)] {
            let mut normalizer = Normalizer::default();
            let mut map = Map::new();
            map.insert(first.into(), serde_json::json!({"name": first == a}));
            map.insert(second.into(), serde_json::json!({"name": second == a}));
            let value = normalizer.value(Value::Object(map), Some(LONG_STRING));
            assert_eq!(
                value,
                serde_json::json!({"id0": {"name": false}, "id1": {"name": true}})
            );
        }
    }

    #[test]
    fn wall_clock_times_and_timeout_states_are_masked() {
        let mut normalizer = Normalizer::default();
        let value = serde_json::json!({
            "resets_at_unix_ms": 1_789_000_000_000_u64,
            "waiting_until_unix_ms": null,
            "start_time": 1_789_000_000,
            "text": "at 2026-10-07T12:00:00.123456+00:00, resets in about 3h 12m. Usage",
            "reason": "Timeout (5s) waiting for user command in state AwaitingToolApproval { a: {\"x\": 1} }.",
        });
        assert_eq!(
            normalizer.value(value, Some(LONG_STRING)),
            serde_json::json!({
                "reason": "Timeout (5s) waiting for user command in state <state>.",
                "resets_at_unix_ms": "<time>",
                "start_time": "<time>",
                "text": "at <datetime>, resets in about <duration>. Usage",
                "waiting_until_unix_ms": null,
            })
        );
    }

    /// A test that records its trace gets every event and snapshot of its
    /// loops, in order, without a trace variable.
    #[tokio::test]
    async fn a_recorded_test_traces_its_loops_events_and_snapshots() {
        use futures::StreamExt;
        let trace = record();
        let (mut agent, _events, _commands) = super::super::retry_tests::harness();
        let workspace = agent.task.current_working_directory.clone();
        trace.mask(&workspace.to_string_lossy(), "<workspace>");
        let tool_config = &mut agent.task.agent_config.tool_config;
        tool_config.tool_calling_mode = crate::agent_config::ToolCallingMode::Native;
        tool_config.default_auto_approve_tool_use = true;
        let turn = futures::stream::iter([Ok(super::super::retry_tests::complete_response())]);
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(turn.boxed()));
        let task = agent.run().await.unwrap();
        assert!(matches!(
            task.current_state,
            crate::task::TaskState::Completed { .. }
        ));
        let lines = trace.lines();
        let kinds: String = lines.iter().map(|line| &line[..1]).collect();
        assert_eq!(kinds, "ESEESESEESSE", "{lines:#?}");
        assert_eq!(lines[0], r#"E {"TaskStateChanged":"InProgress"}"#);
        assert!(lines[1].starts_with("S durable=0 msgs=1 native_results=0 "));
        assert_eq!(lines.last().unwrap(), r#"E "TaskStreamEnd""#);
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with(r#"E {"ToolResult":{"output":"done""#)),
            "{lines:#?}"
        );
    }

    #[test]
    fn long_strings_keep_their_head_and_a_hash() {
        let mut normalizer = Normalizer::default();
        let long = "x".repeat(LONG_STRING + 1);
        let shortened = normalizer.string(&long, Some(LONG_STRING));
        assert!(shortened.starts_with(&"x".repeat(LONG_STRING_HEAD)));
        assert!(shortened.ends_with(&format!("<401 chars, {:016x}>", fnv1a(long.as_bytes()))));
        assert_eq!(normalizer.string(&long, None), long);
    }
}
