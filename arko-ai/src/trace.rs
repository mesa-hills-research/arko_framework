//! Opt-in, per-call request tracing for diagnosing prompt caching.
//!
//! When a [`RequestTrace`] is attached to a client, every model call writes
//! its exact request and raw terminal response to disk, and appends one
//! summary line to `calls.jsonl`, including prefix-parity metrics against
//! the previous request of the same conversation:
//!
//! ```text
//! <dir>/calls.jsonl
//! <dir>/<conversation prompt_cache_key, sanitized>/NNNN-request.json
//! <dir>/<conversation prompt_cache_key, sanitized>/NNNN-response.json
//! ```
//!
//! The previous request is the last one that didn't fail: a failed or
//! abandoned call is traced, but the call after it (typically its retry) is
//! compared with the call before it.
//!
//! Credentials are redacted. Traces still contain prompt and response
//! content, so keep them local. Tracing never fails a model call: I/O
//! errors are logged and ignored.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Request headers whose values are never written to a trace.
const REDACTED_HEADERS: &[&str] = &[
    "authorization",
    "chatgpt-account-id",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "proxy-authorization",
];

/// Response headers copied into the `calls.jsonl` summary when present.
const SUMMARY_HEADERS: &[&str] = &[
    "x-request-id",
    "x-oai-request-id",
    "x-codex-plan-type",
    "x-codex-primary-used-percent",
    "x-codex-secondary-used-percent",
    "openai-model",
    "x-openai-model",
    "x-reasoning-included",
    "x-codex-safety-buffering-enabled",
    "x-codex-safety-buffering-faster-model",
    "openai-processing-ms",
];

/// Writes each model call's request and response to a directory.
#[derive(Clone, Debug)]
pub struct RequestTrace {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    dir: PathBuf,
    conversations: Mutex<HashMap<String, ConversationState>>,
    calls: Mutex<()>,
}

#[derive(Debug, Default)]
struct ConversationState {
    next_seq: u64,
    previous: Option<PreviousRequest>,
}

#[derive(Debug, Clone)]
struct PreviousRequest {
    seq: u64,
    items: Vec<String>,
    input_json: String,
    instructions_hash: Option<String>,
    tools_hash: Option<String>,
    input_tokens: Option<u64>,
}

impl RequestTrace {
    /// Traces calls into `dir`, which is created on first use.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(Inner {
                dir: dir.into(),
                conversations: Mutex::new(HashMap::new()),
                calls: Mutex::new(()),
            }),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }

    /// Starts tracing one call. `body` is the serialized request body, used
    /// to compare `input`, `instructions` and `tools` with the previous
    /// request of the same conversation.
    pub(crate) fn begin(
        &self,
        conversation_key: &str,
        requested_model: &str,
        body: &Value,
        stream: bool,
    ) -> TraceCall {
        let items: Vec<String> = body
            .get("input")
            .and_then(Value::as_array)
            .map(|items| items.iter().map(Value::to_string).collect())
            .unwrap_or_default();
        let input_json = body.get("input").map(Value::to_string).unwrap_or_default();
        let instructions_hash = body.get("instructions").map(hash_value);
        let tools_hash = body.get("tools").map(hash_value);

        let (seq, previous) = {
            let mut conversations = lock(&self.inner.conversations);
            let state = conversations.entry(conversation_key.to_string()).or_default();
            state.next_seq += 1;
            let previous = state.previous.replace(PreviousRequest {
                seq: state.next_seq,
                items: items.clone(),
                input_json: input_json.clone(),
                instructions_hash: instructions_hash.clone(),
                tools_hash: tools_hash.clone(),
                input_tokens: None,
            });
            (state.next_seq, previous)
        };

        let prefix = prefix_metrics(
            previous.as_ref(),
            &items,
            &input_json,
            instructions_hash.as_deref(),
            tools_hash.as_deref(),
        );

        TraceCall {
            trace: self.clone(),
            conversation_key: conversation_key.to_string(),
            seq,
            requested_model: requested_model.to_string(),
            stream,
            started_unix_ms: unix_ms(SystemTime::now()),
            started: Instant::now(),
            headers_ms: None,
            first_event_ms: None,
            attempts: 0,
            url: None,
            http_status: None,
            response_headers: Map::new(),
            sent_turn_state: None,
            prefix,
            previous_input_tokens: previous.as_ref().and_then(|previous| previous.input_tokens),
            previous,
            created_model: None,
            streamed_items: Vec::new(),
            event_counts: BTreeMap::new(),
            finished: false,
        }
    }

    /// Makes `previous` the conversation's previous request again, if call
    /// `seq` (which failed) is still the latest.
    fn restore_previous(&self, conversation_key: &str, seq: u64, previous: Option<PreviousRequest>) {
        let mut conversations = lock(&self.inner.conversations);
        if let Some(state) = conversations.get_mut(conversation_key)
            && state.previous.as_ref().is_some_and(|latest| latest.seq == seq)
        {
            state.previous = previous;
        }
    }

    fn conversation_dir(&self, conversation_key: &str) -> PathBuf {
        self.inner.dir.join(sanitize(conversation_key))
    }

    fn remember_input_tokens(&self, conversation_key: &str, seq: u64, input_tokens: Option<u64>) {
        let mut conversations = lock(&self.inner.conversations);
        if let Some(previous) = conversations
            .get_mut(conversation_key)
            .and_then(|state| state.previous.as_mut())
            .filter(|previous| previous.seq == seq)
        {
            previous.input_tokens = input_tokens;
        }
    }

    fn append_summary(&self, line: &Value) {
        let _guard = lock(&self.inner.calls);
        let result = fs::create_dir_all(&self.inner.dir).and_then(|()| {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.inner.dir.join("calls.jsonl"))?;
            writeln!(file, "{line}")
        });
        if let Err(error) = result {
            log::warn!("arko_ai request trace: failed to append calls.jsonl: {error}");
        }
    }
}

/// One traced model call. Created by [`RequestTrace::begin`]; completed by
/// [`TraceCall::finish`] or [`TraceCall::fail`]. A call dropped before either
/// is recorded as abandoned (for example, a stream the consumer stopped
/// reading).
#[derive(Debug)]
pub(crate) struct TraceCall {
    trace: RequestTrace,
    conversation_key: String,
    seq: u64,
    requested_model: String,
    stream: bool,
    started_unix_ms: u128,
    started: Instant,
    headers_ms: Option<u128>,
    first_event_ms: Option<u128>,
    attempts: u32,
    url: Option<String>,
    http_status: Option<u16>,
    response_headers: Map<String, Value>,
    /// The `x-codex-turn-state` request header of the last attempt.
    sent_turn_state: Option<String>,
    prefix: Value,
    previous_input_tokens: Option<u64>,
    /// The request this call is compared with, restored if the call fails.
    previous: Option<PreviousRequest>,
    created_model: Option<String>,
    streamed_items: Vec<Value>,
    event_counts: BTreeMap<String, u64>,
    finished: bool,
}

impl TraceCall {
    /// Records the exact request about to be sent. Called for every attempt;
    /// a retried request overwrites the file and increments `attempts`.
    pub(crate) fn record_request(&mut self, request: &reqwest::Request) {
        self.attempts += 1;
        self.url = Some(request.url().to_string());
        self.sent_turn_state = request
            .headers()
            .get("x-codex-turn-state")
            .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
        let bytes = request.body().and_then(reqwest::Body::as_bytes).unwrap_or_default();
        let body: Value = serde_json::from_slice(bytes).unwrap_or(Value::Null);
        let document = json!({
            "conversation": self.conversation_key,
            "seq": self.seq,
            "attempt": self.attempts,
            "method": request.method().as_str(),
            "url": request.url().as_str(),
            "headers": redacted_headers(request.headers()),
            "body_bytes": bytes.len(),
            "body_sha256": hex_sha256(bytes),
            "body": body,
        });
        self.write_file("request", &document);
    }

    /// Records the response status line and headers.
    pub(crate) fn record_response_head(&mut self, response: &reqwest::Response) {
        self.headers_ms = Some(self.started.elapsed().as_millis());
        self.http_status = Some(response.status().as_u16());
        self.response_headers = redacted_headers(response.headers());
    }

    /// Records one streamed event. `response.created` supplies the served
    /// model; `response.output_item.done` items are kept so they can be
    /// compared with the terminal response's `output`.
    pub(crate) fn on_event(&mut self, event_type: &str, value: &Value) {
        if self.first_event_ms.is_none() {
            self.first_event_ms = Some(self.started.elapsed().as_millis());
        }
        *self.event_counts.entry(event_type.to_string()).or_default() += 1;
        match event_type {
            "response.created" => {
                self.created_model = value
                    .pointer("/response/model")
                    .and_then(Value::as_str)
                    .map(String::from);
            }
            "response.output_item.done" => {
                if let Some(item) = value.get("item") {
                    self.streamed_items.push(item.clone());
                }
            }
            _ => {}
        }
    }

    /// Completes the call with the raw terminal response object (the body of
    /// a non-streaming response, or `response` of the terminal stream event).
    pub(crate) fn finish(mut self, response: &Value) {
        self.finished = true;
        let status = response
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        self.complete(&status, Some(response), None);
    }

    /// Completes the call with an error (non-success status, stream error,
    /// truncation). `body` is the provider's error payload, if any.
    pub(crate) fn fail(mut self, error: &str, body: Option<&Value>) {
        self.finished = true;
        self.complete("error", body, Some(error));
    }

    fn complete(&mut self, status: &str, response: Option<&Value>, error: Option<&str>) {
        let completed_ms = self.started.elapsed().as_millis();
        let usage = response.and_then(|response| response.get("usage"));
        let int = |pointer: &str| usage.and_then(|usage| usage.pointer(pointer)).and_then(Value::as_u64);
        let input_tokens = int("/input_tokens");
        let cached_tokens = int("/input_tokens_details/cached_tokens");
        let completed_output: Vec<Value> = response
            .and_then(|response| response.get("output"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let response_id = response.and_then(|response| response.get("id")).and_then(Value::as_str);

        if error.is_some() {
            self.trace
                .restore_previous(&self.conversation_key, self.seq, self.previous.take());
        } else {
            self.trace
                .remember_input_tokens(&self.conversation_key, self.seq, input_tokens);
        }

        let document = json!({
            "conversation": self.conversation_key,
            "seq": self.seq,
            "status": status,
            "error": error,
            "http_status": self.http_status,
            "headers": self.response_headers,
            "timings_ms": {
                "headers": self.headers_ms,
                "first_event": self.first_event_ms,
                "completed": completed_ms,
            },
            "created_model": self.created_model,
            "event_counts": self.event_counts,
            "streamed_output_items": self.streamed_items,
            "response": response,
        });
        self.write_file("response", &document);

        let previous_found = match (self.previous_input_tokens, cached_tokens) {
            (Some(previous), Some(cached)) => Some(cached + 1024 >= previous),
            (Some(_), None) if error.is_none() => Some(false),
            _ => None,
        };
        let shortfall = match (self.previous_input_tokens, cached_tokens) {
            (Some(previous), cached) => Some(previous as i64 - cached.unwrap_or(0) as i64),
            _ => None,
        };
        let mut summary_headers = Map::new();
        for name in SUMMARY_HEADERS {
            if let Some(value) = self.response_headers.get(*name) {
                summary_headers.insert((*name).to_string(), value.clone());
            }
        }
        let turn_state = self.response_headers.get("x-codex-turn-state").and_then(Value::as_str);

        let line = json!({
            "completed_unix_ms": unix_ms(SystemTime::now()),
            "started_unix_ms": self.started_unix_ms,
            "conversation": self.conversation_key,
            "seq": self.seq,
            "stream": self.stream,
            "attempts": self.attempts,
            "url": self.url,
            "status": status,
            "error": error,
            "http_status": self.http_status,
            "requested_model": self.requested_model,
            "created_model": self.created_model,
            "completed_model": response.and_then(|response| response.get("model")),
            "response_id": response_id,
            "response_id_prefix": response_id.and_then(response_id_prefix),
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "cached_tokens_present": usage.is_some_and(|usage| usage.pointer("/input_tokens_details/cached_tokens").is_some()),
            "cache_write_tokens": int("/input_tokens_details/cache_write_tokens"),
            "output_tokens": int("/output_tokens"),
            "reasoning_tokens": int("/output_tokens_details/reasoning_tokens"),
            "previous_input_tokens": self.previous_input_tokens,
            "previous_found": previous_found,
            "shortfall_tokens": shortfall,
            "prefix": self.prefix,
            "completed_output_count": completed_output.len(),
            "completed_output_types": item_types(&completed_output),
            "streamed_output_count": self.streamed_items.len(),
            "streamed_output_types": item_types(&self.streamed_items),
            "streamed_reasoning_with_encrypted_content": self.streamed_items.iter().filter(|item| {
                item.get("type").and_then(Value::as_str) == Some("reasoning")
                    && item.get("encrypted_content").and_then(Value::as_str).is_some_and(|content| !content.is_empty())
            }).count(),
            "response_reasoning": response.and_then(|response| response.get("reasoning")),
            "response_service_tier": response.and_then(|response| response.get("service_tier")),
            "turn_state_present": turn_state.is_some(),
            "turn_state_sha256": turn_state.map(|value| hex_sha256(value.as_bytes())),
            "turn_state_sent": self.sent_turn_state.is_some(),
            "turn_state_sent_sha256": self.sent_turn_state.as_ref().map(|value| hex_sha256(value.as_bytes())),
            "headers": summary_headers,
            "timings_ms": {
                "headers": self.headers_ms,
                "first_event": self.first_event_ms,
                "completed": completed_ms,
            },
        });
        self.trace.append_summary(&line);
    }

    fn write_file(&self, kind: &str, document: &Value) {
        let dir = self.trace.conversation_dir(&self.conversation_key);
        let path = dir.join(format!("{:04}-{kind}.json", self.seq));
        let result = fs::create_dir_all(&dir).and_then(|()| {
            let text = serde_json::to_string_pretty(document).map_err(std::io::Error::other)?;
            fs::write(&path, text)
        });
        if let Err(error) = result {
            log::warn!("arko_ai request trace: failed to write {}: {error}", path.display());
        }
    }
}

impl Drop for TraceCall {
    fn drop(&mut self) {
        if !self.finished {
            self.finished = true;
            self.complete("abandoned", None, Some("call dropped before completion"));
        }
    }
}

/// Compares the current request with the previous one of the same
/// conversation. `previous_is_prefix` is true when every previous `input`
/// item appears unchanged, in order, at the start of the current `input`.
fn prefix_metrics(
    previous: Option<&PreviousRequest>,
    items: &[String],
    input_json: &str,
    instructions_hash: Option<&str>,
    tools_hash: Option<&str>,
) -> Value {
    let Some(previous) = previous else {
        return json!({
            "first_request": true,
            "items": items.len(),
            "input_bytes": input_json.len(),
            "instructions_sha256": instructions_hash,
            "tools_sha256": tools_hash,
        });
    };
    let common_items = previous
        .items
        .iter()
        .zip(items)
        .take_while(|(previous, current)| previous == current)
        .count();
    let common_bytes = previous
        .input_json
        .bytes()
        .zip(input_json.bytes())
        .take_while(|(previous, current)| previous == current)
        .count();
    json!({
        "first_request": false,
        "previous_is_prefix": common_items == previous.items.len() && items.len() >= previous.items.len(),
        "common_items": common_items,
        "previous_items": previous.items.len(),
        "items": items.len(),
        "common_input_bytes": common_bytes,
        "previous_input_bytes": previous.input_json.len(),
        "input_bytes": input_json.len(),
        "instructions_sha256": instructions_hash,
        "tools_sha256": tools_hash,
        "instructions_changed": previous.instructions_hash.as_deref() != instructions_hash,
        "tools_changed": previous.tools_hash.as_deref() != tools_hash,
    })
}

fn redacted_headers(headers: &reqwest::header::HeaderMap) -> Map<String, Value> {
    let mut map = Map::new();
    for (name, value) in headers {
        let name = name.as_str().to_ascii_lowercase();
        let value = if REDACTED_HEADERS.contains(&name.as_str()) {
            "[REDACTED]".to_string()
        } else {
            String::from_utf8_lossy(value.as_bytes()).into_owned()
        };
        match map.get_mut(&name) {
            Some(Value::String(existing)) => {
                existing.push_str(", ");
                existing.push_str(&value);
            }
            _ => {
                map.insert(name, Value::String(value));
            }
        }
    }
    map
}

fn item_types(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.get("type").and_then(Value::as_str).unwrap_or("?").to_string())
        .collect()
}

/// The 16 hex characters after `resp_`, which stay constant per server-side
/// conversation on the ChatGPT backend.
fn response_id_prefix(id: &str) -> Option<String> {
    id.strip_prefix("resp_")
        .map(|rest| rest.chars().take(16).collect())
}

fn sanitize(key: &str) -> String {
    let sanitized: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
        .collect();
    if sanitized.is_empty() { "_".to_string() } else { sanitized }
}

fn hash_value(value: &Value) -> String {
    hex_sha256(value.to_string().as_bytes())
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unix_ms(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    #[test]
    fn credentials_are_redacted_and_other_headers_kept() {
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_static("Bearer secret-token"));
        headers.insert("chatgpt-account-id", HeaderValue::from_static("acct-123"));
        headers.insert("x-api-key", HeaderValue::from_static("sk-secret"));
        headers.insert("OpenAI-Beta", HeaderValue::from_static("responses=experimental"));
        let map = redacted_headers(&headers);
        assert_eq!(map["authorization"], "[REDACTED]");
        assert_eq!(map["chatgpt-account-id"], "[REDACTED]");
        assert_eq!(map["x-api-key"], "[REDACTED]");
        assert_eq!(map["openai-beta"], "responses=experimental");
        let text = Value::Object(map).to_string();
        assert!(!text.contains("secret") && !text.contains("acct-123"));
    }

    fn body(items: &[Value], instructions: &str) -> Value {
        json!({"instructions": instructions, "tools": [{"name": "t"}], "input": items})
    }

    #[test]
    fn prefix_metrics_detect_append_only_growth_and_rewrites() {
        let trace = RequestTrace::new(std::env::temp_dir().join("arko-trace-unit-unused"));
        let a = json!({"role": "user", "content": "a"});
        let b = json!({"role": "assistant", "content": "b"});
        let c = json!({"role": "user", "content": "c"});

        let first = trace.begin("k", "m", &body(std::slice::from_ref(&a), "sys"), false);
        assert_eq!(first.prefix["first_request"], true);
        let seq1 = first.seq;
        std::mem::forget(first);
        trace.remember_input_tokens("k", seq1, Some(5000));

        let second = trace.begin("k", "m", &body(&[a.clone(), b.clone(), c.clone()], "sys"), false);
        assert_eq!(second.prefix["previous_is_prefix"], true);
        assert_eq!(second.prefix["common_items"], 1);
        assert_eq!(second.prefix["instructions_changed"], false);
        assert_eq!(second.previous_input_tokens, Some(5000));
        std::mem::forget(second);

        let rewritten = json!({"role": "assistant", "content": "B"});
        let third = trace.begin("k", "m", &body(&[a.clone(), rewritten, c], "sys2"), false);
        assert_eq!(third.prefix["previous_is_prefix"], false);
        assert_eq!(third.prefix["common_items"], 1);
        assert_eq!(third.prefix["instructions_changed"], true);
        assert!(third.prefix["common_input_bytes"].as_u64().unwrap() > 0);
        std::mem::forget(third);

        let other = trace.begin("other", "m", &body(&[a], "sys"), false);
        assert_eq!(other.prefix["first_request"], true);
        std::mem::forget(other);
    }

    #[test]
    fn a_finished_call_writes_files_and_a_summary_with_previous_found() {
        let dir = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let trace = RequestTrace::new(&dir);
        let item = json!({"role": "user", "content": "a"});

        let first = trace.begin("arko:conversation:x", "gpt-6-astra", &body(std::slice::from_ref(&item), "s"), true);
        first.finish(&json!({"id": "resp_0123456789abcdef99", "status": "completed", "model": "gpt-6-astra",
            "output": [], "usage": {"input_tokens": 4000, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 10}}));

        let mut second = trace.begin("arko:conversation:x", "gpt-6-astra", &body(&[item.clone(), item], "s"), true);
        second.on_event("response.created", &json!({"response": {"model": "gpt-5.6-luna"}}));
        second.on_event("response.output_item.done", &json!({"item": {"type": "reasoning", "encrypted_content": "e"}}));
        second.finish(&json!({"id": "resp_0123456789abcdef77", "status": "completed", "model": "gpt-5.6-luna",
            "output": [], "usage": {"input_tokens": 4500, "input_tokens_details": {"cached_tokens": 3968}, "output_tokens": 10}}));

        let lines: Vec<Value> = fs::read_to_string(dir.join("calls.jsonl")).unwrap()
            .lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["cached_tokens"], 0);
        assert_eq!(lines[0]["cached_tokens_present"], true);
        assert!(lines[0]["previous_found"].is_null());
        assert_eq!(lines[1]["previous_found"], true);
        assert_eq!(lines[1]["shortfall_tokens"], 32);
        assert_eq!(lines[1]["created_model"], "gpt-5.6-luna");
        assert_eq!(lines[1]["streamed_reasoning_with_encrypted_content"], 1);
        assert_eq!(lines[1]["completed_output_count"], 0);
        assert_eq!(lines[1]["response_id_prefix"], "0123456789abcdef");
        assert!(dir.join("arko_conversation_x").join("0002-response.json").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_retry_is_compared_with_the_call_before_the_failed_attempt() {
        let dir = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let trace = RequestTrace::new(&dir);
        let (a, b) = (json!({"role": "user", "content": "a"}), json!({"role": "user", "content": "b"}));
        let usage = |input: u64, cached: u64| json!({"status": "completed",
            "usage": {"input_tokens": input, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": 1}});

        trace.begin("k", "m", &body(std::slice::from_ref(&a), "s"), true).finish(&usage(4000, 0));
        trace.begin("k", "m", &body(&[a.clone(), b.clone()], "s"), true).fail("rate limited", None);
        let retry = trace.begin("k", "m", &body(&[a.clone(), b.clone()], "s"), true);
        assert_eq!(retry.seq, 3);
        assert_eq!(retry.prefix["previous_items"], 1);
        assert_eq!(retry.previous_input_tokens, Some(4000));
        retry.finish(&usage(4500, 3968));

        let lines: Vec<Value> = fs::read_to_string(dir.join("calls.jsonl")).unwrap()
            .lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(lines[1]["status"], "error");
        assert_eq!(lines[2]["previous_found"], true);
        assert_eq!(lines[2]["turn_state_sent"], false);
        fs::remove_dir_all(&dir).ok();
    }
}
