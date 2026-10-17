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
//! A key made only of lowercase ASCII letters, digits, `.`, `-` and `_` is
//! its own directory name (unless it ends in `-` and 16 hex digits, like a
//! hashed name). Other keys are lowercased with other characters replaced
//! by `_`, plus `-` and 16 hex digits of the key's SHA-256
//! (`arko:conversation:x` → `arko_conversation_x-ed168f36ad25bfa2`), so
//! different keys never share a directory, even on case-insensitive
//! filesystems. Each `calls.jsonl` line names its directory in
//! `conversation_dir`.
//!
//! The previous request is the last one that didn't fail: a failed or
//! abandoned call is traced, but the call after it (typically its retry) is
//! compared with the call before it. A response that doesn't parse counts as
//! a failed call.
//!
//! Per conversation the trace keeps a hash of each `input` item of the
//! previous request, and the serialized `input` itself only for the 32 most
//! recently traced conversations, so memory stays bounded in long-running
//! processes. For an older conversation's next
//! request, `common_input_bytes` is `null`. The other metrics stay exact.
//!
//! Credentials are redacted, and `x-codex-turn-state` routing tokens are
//! written as their SHA-256 (`sha256:<hex>`, the hash the summary line
//! reports). Conversation directory names never leave `<dir>`. Traces still
//! contain prompt and response content, so keep them local. Tracing never
//! fails a model call: I/O errors are logged and ignored.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Request and response headers whose values are never written to a trace.
const REDACTED_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "chatgpt-account-id",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "x-auth-token",
    "cookie",
    "set-cookie",
];

/// Headers written as `sha256:<hex>` of their value: routing tokens that
/// shouldn't be stored, but whose changes the trace should show.
const HASHED_HEADERS: &[&str] = &["x-codex-turn-state"];

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

/// How many conversations, most recently traced first, keep their previous
/// request's serialized `input` for the byte-level prefix comparison.
const RETAINED_INPUTS: usize = 32;

/// Writes each model call's request and response to a directory.
#[derive(Clone, Debug)]
pub struct RequestTrace {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    dir: PathBuf,
    conversations: Mutex<Conversations>,
    calls: Mutex<()>,
}

#[derive(Debug, Default)]
struct Conversations {
    states: HashMap<String, ConversationState>,
    /// Conversations whose previous request keeps its serialized `input`,
    /// least recently traced first. At most [`RETAINED_INPUTS`].
    retained: VecDeque<String>,
}

impl Conversations {
    /// Marks `key` as the most recently traced conversation and drops the
    /// serialized input of the least recent ones beyond the limit.
    fn retain_input(&mut self, key: &str) {
        if let Some(position) = self.retained.iter().position(|retained| retained == key) {
            self.retained.remove(position);
        }
        self.retained.push_back(key.to_string());
        while self.retained.len() > RETAINED_INPUTS {
            let Some(evicted) = self.retained.pop_front() else {
                break;
            };
            if let Some(previous) = self
                .states
                .get_mut(&evicted)
                .and_then(|state| state.previous.as_mut())
            {
                previous.input_json = None;
            }
        }
    }
}

#[derive(Debug, Default)]
struct ConversationState {
    next_seq: u64,
    previous: Option<PreviousRequest>,
}

#[derive(Debug, Clone)]
struct PreviousRequest {
    seq: u64,
    /// Hash of each serialized `input` item, for `common_items`.
    item_hashes: Vec<u128>,
    input_bytes: usize,
    /// The serialized `input`, for `common_input_bytes`. Dropped once the
    /// conversation is not among the [`RETAINED_INPUTS`] most recent.
    input_json: Option<Arc<str>>,
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
                conversations: Mutex::new(Conversations::default()),
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
        let item_hashes: Vec<u128> = body
            .get("input")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| hash128(item.to_string().as_bytes()))
                    .collect()
            })
            .unwrap_or_default();
        let input_json: Arc<str> = body
            .get("input")
            .map(Value::to_string)
            .unwrap_or_default()
            .into();
        let instructions_hash = body.get("instructions").map(hash_value);
        let tools_hash = body.get("tools").map(hash_value);

        let (seq, previous) = {
            let mut conversations = lock(&self.inner.conversations);
            let state = conversations
                .states
                .entry(conversation_key.to_string())
                .or_default();
            state.next_seq += 1;
            let seq = state.next_seq;
            let previous = state.previous.replace(PreviousRequest {
                seq,
                item_hashes: item_hashes.clone(),
                input_bytes: input_json.len(),
                input_json: Some(input_json.clone()),
                instructions_hash: instructions_hash.clone(),
                tools_hash: tools_hash.clone(),
                input_tokens: None,
            });
            conversations.retain_input(conversation_key);
            (seq, previous)
        };

        let prefix = prefix_metrics(
            previous.as_ref(),
            &item_hashes,
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
            sent_turn_state_sha256: None,
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
    fn restore_previous(
        &self,
        conversation_key: &str,
        seq: u64,
        previous: Option<PreviousRequest>,
    ) {
        let mut conversations = lock(&self.inner.conversations);
        let retained = conversations
            .retained
            .iter()
            .any(|key| key == conversation_key);
        if let Some(state) = conversations.states.get_mut(conversation_key)
            && state
                .previous
                .as_ref()
                .is_some_and(|latest| latest.seq == seq)
        {
            state.previous = previous.map(|mut previous| {
                // Evicted while the call ran: keep the bound.
                if !retained {
                    previous.input_json = None;
                }
                previous
            });
        }
    }

    fn conversation_dir(&self, conversation_key: &str) -> PathBuf {
        self.inner.dir.join(sanitize(conversation_key))
    }

    fn remember_input_tokens(&self, conversation_key: &str, seq: u64, input_tokens: Option<u64>) {
        let mut conversations = lock(&self.inner.conversations);
        if let Some(previous) = conversations
            .states
            .get_mut(conversation_key)
            .and_then(|state| state.previous.as_mut())
            .filter(|previous| previous.seq == seq)
        {
            previous.input_tokens = input_tokens;
        }
    }

    #[cfg(test)]
    fn retained_input_bytes(&self) -> usize {
        lock(&self.inner.conversations)
            .states
            .values()
            .filter_map(|state| state.previous.as_ref()?.input_json.as_ref())
            .map(|input| input.len())
            .sum()
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

/// One traced model call. Created by [`RequestTrace::begin`]. Completed by
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
    /// SHA-256 of the `x-codex-turn-state` request header of the last attempt.
    sent_turn_state_sha256: Option<String>,
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
    /// Records the exact request about to be sent. Called for every attempt.
    /// A retried request overwrites the file and increments `attempts`.
    pub(crate) fn record_request(&mut self, request: &reqwest::Request) {
        self.attempts += 1;
        self.url = Some(request.url().to_string());
        self.sent_turn_state_sha256 = request
            .headers()
            .get("x-codex-turn-state")
            .map(|value| hex_sha256(value.as_bytes()));
        let bytes = request
            .body()
            .and_then(reqwest::Body::as_bytes)
            .unwrap_or_default();
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
    /// model. `response.output_item.done` items are kept so they can be
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
        let int = |pointer: &str| {
            usage
                .and_then(|usage| usage.pointer(pointer))
                .and_then(Value::as_u64)
        };
        let input_tokens = int("/input_tokens");
        let cached_tokens = int("/input_tokens_details/cached_tokens");
        let completed_output: Vec<Value> = response
            .and_then(|response| response.get("output"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let response_id = response
            .and_then(|response| response.get("id"))
            .and_then(Value::as_str);

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

        // Usage numbers come from the server: saturate rather than overflow.
        let previous_found = match (self.previous_input_tokens, cached_tokens) {
            (Some(previous), Some(cached)) => Some(cached.saturating_add(1024) >= previous),
            (Some(_), None) if error.is_none() => Some(false),
            _ => None,
        };
        let shortfall = self.previous_input_tokens.map(|previous| {
            let difference = i128::from(previous) - i128::from(cached_tokens.unwrap_or(0));
            difference.clamp(i64::MIN.into(), i64::MAX.into()) as i64
        });
        let mut summary_headers = Map::new();
        for name in SUMMARY_HEADERS {
            if let Some(value) = self.response_headers.get(*name) {
                summary_headers.insert((*name).to_string(), value.clone());
            }
        }
        // Hashed by `redacted_headers`.
        let turn_state_sha256 = self
            .response_headers
            .get("x-codex-turn-state")
            .and_then(Value::as_str)
            .and_then(|value| value.strip_prefix("sha256:"));

        let mut line = json!({
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
            "turn_state_present": turn_state_sha256.is_some(),
            "turn_state_sha256": turn_state_sha256,
            "turn_state_sent": self.sent_turn_state_sha256.is_some(),
            "turn_state_sent_sha256": self.sent_turn_state_sha256,
            "headers": summary_headers,
            "timings_ms": {
                "headers": self.headers_ms,
                "first_event": self.first_event_ms,
                "completed": completed_ms,
            },
        });
        // Added outside `json!`, which is at its macro recursion limit.
        line["conversation_dir"] = Value::String(sanitize(&self.conversation_key));
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
            log::warn!(
                "arko_ai request trace: failed to write {}: {error}",
                path.display()
            );
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
/// Items are compared by hash. `common_input_bytes` is `null` when the
/// previous request's serialized input is no longer retained.
fn prefix_metrics(
    previous: Option<&PreviousRequest>,
    items: &[u128],
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
        .item_hashes
        .iter()
        .zip(items)
        .take_while(|(previous, current)| previous == current)
        .count();
    let common_bytes = previous.input_json.as_ref().map(|previous| {
        previous
            .bytes()
            .zip(input_json.bytes())
            .take_while(|(previous, current)| previous == current)
            .count()
    });
    let previous_items = previous.item_hashes.len();
    json!({
        "first_request": false,
        "previous_is_prefix": common_items == previous_items && items.len() >= previous_items,
        "common_items": common_items,
        "previous_items": previous_items,
        "items": items.len(),
        "common_input_bytes": common_bytes,
        "previous_input_bytes": previous.input_bytes,
        "input_bytes": input_json.len(),
        "instructions_sha256": instructions_hash,
        "tools_sha256": tools_hash,
        "instructions_changed": previous.instructions_hash.as_deref() != instructions_hash,
        "tools_changed": previous.tools_hash.as_deref() != tools_hash,
    })
}

/// Headers as written to a trace: credentials redacted, routing tokens
/// hashed, repeated headers joined with `, `.
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
    for name in HASHED_HEADERS {
        if let Some(Value::String(value)) = map.get_mut(*name) {
            *value = format!("sha256:{}", hex_sha256(value.as_bytes()));
        }
    }
    map
}

fn item_types(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .map(|item| {
            item.get("type")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

/// The 16 hex characters after `resp_`, which stay constant per server-side
/// conversation on the ChatGPT backend.
fn response_id_prefix(id: &str) -> Option<String> {
    id.strip_prefix("resp_")
        .map(|rest| rest.chars().take(16).collect())
}

/// A directory name for a conversation key that stays inside the trace
/// directory on every platform and differs between keys. Characters other
/// than ASCII letters, digits,
/// `.`, `-` and `_` become `_` and ASCII letters are lowercased, so the name
/// holds no path separators or drive prefixes. A key this leaves unchanged
/// is its own name, unless it ends like a hashed name (`-` and 16 lowercase
/// hex digits). Any other key gets `-<hash of the key>` appended, so keys
/// that differ only in replaced characters or in case (`team/a`, `team:a`,
/// `Team_a`) keep apart, also on case-insensitive filesystems, and no key
/// can take another's hashed name. A name that would still be unsafe
/// (empty, `.`/`..` or any other leading or trailing dot, a Windows device
/// name such as `CON`, or too long) becomes `_<readable part>-<hash of the
/// key>` instead.
fn sanitize(key: &str) -> String {
    const MAX_LEN: usize = 100;
    let mapped: String = key
        .chars()
        .map(|c| match c {
            'A'..='Z' => c.to_ascii_lowercase(),
            'a'..='z' | '0'..='9' | '.' | '-' | '_' => c,
            _ => '_',
        })
        .collect();
    let stem = mapped.split('.').next().unwrap_or_default();
    let safe = !mapped.is_empty()
        && !mapped.starts_with('.')
        && !mapped.ends_with('.')
        && mapped.len() <= MAX_LEN
        && !is_windows_device_name(stem);
    if safe && mapped == key && !ends_with_hash(key) {
        return mapped;
    }
    let hash = &hex_sha256(key.as_bytes())[..16];
    if safe {
        return format!("{mapped}-{hash}");
    }
    let readable: String = mapped.trim_matches('.').chars().take(64).collect();
    format!("_{readable}-{hash}")
}

/// Whether `name` ends as a hashed name does: `-` and 16 lowercase hex digits.
fn ends_with_hash(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() >= 17
        && bytes[bytes.len() - 17] == b'-'
        && bytes[bytes.len() - 16..]
            .iter()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Names Windows reserves for devices in every directory, with or without
/// an extension.
fn is_windows_device_name(stem: &str) -> bool {
    let stem = stem.to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.as_bytes()[3].is_ascii_digit())
}

fn hash_value(value: &Value) -> String {
    hex_sha256(value.to_string().as_bytes())
}

/// The first 128 bits of the SHA-256 of `bytes`.
fn hash128(bytes: &[u8]) -> u128 {
    let digest = Sha256::digest(bytes);
    let mut first = [0; 16];
    first.copy_from_slice(&digest[..16]);
    u128::from_be_bytes(first)
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn unix_ms(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    #[test]
    fn credentials_are_redacted_and_other_headers_kept() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_static("Bearer secret-token"),
        );
        headers.insert("chatgpt-account-id", HeaderValue::from_static("acct-123"));
        headers.insert("x-api-key", HeaderValue::from_static("sk-secret"));
        headers.insert(
            "OpenAI-Beta",
            HeaderValue::from_static("responses=experimental"),
        );
        let map = redacted_headers(&headers);
        assert_eq!(map["authorization"], "[REDACTED]");
        assert_eq!(map["chatgpt-account-id"], "[REDACTED]");
        assert_eq!(map["x-api-key"], "[REDACTED]");
        assert_eq!(map["openai-beta"], "responses=experimental");
        let text = Value::Object(map).to_string();
        assert!(!text.contains("secret") && !text.contains("acct-123"));
    }

    #[test]
    fn every_credential_header_is_redacted() {
        let mut headers = HeaderMap::new();
        for name in [
            "authorization",
            "proxy-authorization",
            "api-key",
            "x-api-key",
            "x-goog-api-key",
            "x-auth-token",
            "cookie",
            "set-cookie",
            "chatgpt-account-id",
        ] {
            headers.insert(name, HeaderValue::from_static("credential-value"));
        }
        let map = redacted_headers(&headers);
        assert_eq!(map.len(), 9);
        assert!(map.values().all(|value| value == "[REDACTED]"), "{map:?}");
    }

    #[test]
    fn turn_state_tokens_are_hashed_in_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-codex-turn-state", HeaderValue::from_static("turn-token"));
        let map = redacted_headers(&headers);
        assert_eq!(
            map["x-codex-turn-state"],
            format!("sha256:{}", hex_sha256(b"turn-token"))
        );
    }

    #[test]
    fn sanitized_names_stay_inside_the_trace_dir() {
        let dir = Path::new("traces");
        for key in [
            "",
            ".",
            "..",
            "...",
            "../..",
            "../../etc",
            "a/../..",
            "..\\..",
            "C:\\x",
            "/abs",
            ".hidden",
            "trailing.",
            "CON",
            "nul.txt",
            "Lpt1",
        ] {
            let name = sanitize(key);
            assert!(
                !name.is_empty() && name != "." && name != "..",
                "{key:?} -> {name:?}"
            );
            assert!(
                !name.starts_with('.') && !name.ends_with('.'),
                "{key:?} -> {name:?}"
            );
            assert!(!name.contains(['/', '\\', ':']), "{key:?} -> {name:?}");
            let joined = dir.join(&name);
            assert_eq!(joined.parent(), Some(dir), "{key:?} -> {name:?}");
            assert_eq!(joined.components().count(), 2, "{key:?} -> {name:?}");
        }
        // Ordinary keys keep their readable names; unusable ones stay distinct.
        assert_eq!(
            sanitize("arko:conversation:x"),
            "arko_conversation_x-ed168f36ad25bfa2"
        );
        assert_ne!(sanitize("."), sanitize(".."));
        assert_ne!(sanitize(""), sanitize("."));
        assert!(sanitize(&"k".repeat(1000)).len() < 128);
        assert_ne!(sanitize(&"k".repeat(1000)), sanitize(&"k".repeat(1001)));
    }

    #[test]
    fn keys_that_map_to_the_same_name_get_distinct_stable_dirs() {
        // A key that is already a lowercase safe name is its own dir.
        for key in ["team_a", "conv-1.2", "0123abcd"] {
            assert_eq!(sanitize(key), key);
        }
        // Any mapped key gets a stable hash of itself appended.
        assert_eq!(sanitize("team/a"), "team_a-65f838a2fbff37d8");
        assert_eq!(sanitize("team:a"), "team_a-5752fddeb3d6826d");
        assert_eq!(sanitize("Team_a"), "team_a-ce273f5f4372bcfa");
        assert_eq!(sanitize("TEAM_A"), "team_a-a44a58ae06e1266c");
        // A key that looks like a hashed name is hashed too, so it can't
        // take the directory of the key that hashes to it.
        assert_eq!(
            sanitize("team_a-65f838a2fbff37d8"),
            "team_a-65f838a2fbff37d8-288075f2d9c64d12"
        );
        assert_ne!(sanitize("team_a-65f838a2fbff37d8"), sanitize("team/a"));
        for key in [
            "team_a-65f838a2fbff37d",
            "team_a-65f838a2fbff37dg",
            "team_a_65f838a2fbff37d8",
        ] {
            assert_eq!(sanitize(key), key, "not a hashed name");
        }
        // Distinct even compared case-insensitively.
        let keys = [
            "team_a", "team/a", "team:a", "team\\a", "Team_a", "TEAM_A", "team a", "tëam_a", "CON",
            "con", ".", "..",
        ];
        let names: std::collections::HashSet<String> = keys
            .iter()
            .map(|key| sanitize(key).to_ascii_lowercase())
            .collect();
        assert_eq!(names.len(), keys.len(), "{names:?}");
    }

    #[test]
    fn absurd_usage_numbers_saturate_instead_of_overflowing() {
        let dir = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let trace = RequestTrace::new(&dir);
        let item = json!({"role": "user", "content": "a"});
        let usage = |input: u64, cached: u64| {
            json!({"status": "completed",
            "usage": {"input_tokens": input, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": 1}})
        };
        trace
            .begin("k", "m", &body(std::slice::from_ref(&item), "s"), false)
            .finish(&usage(i64::MAX as u64, 0));
        trace
            .begin("k", "m", &body(std::slice::from_ref(&item), "s"), false)
            .finish(&usage(u64::MAX, u64::MAX));
        trace
            .begin("k", "m", &body(std::slice::from_ref(&item), "s"), false)
            .finish(&usage(0, 0));
        let lines: Vec<Value> = fs::read_to_string(dir.join("calls.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[1]["previous_found"], true);
        assert_eq!(
            lines[1]["shortfall_tokens"],
            i64::MIN,
            "i64::MAX - u64::MAX"
        );
        assert_eq!(lines[2]["previous_found"], false);
        assert_eq!(
            lines[2]["shortfall_tokens"],
            i64::MAX,
            "u64::MAX - 0, clamped"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_dot_dot_key_writes_inside_the_trace_dir() {
        let root = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let dir = root.join("trace");
        let trace = RequestTrace::new(&dir);
        trace
            .begin(
                "..",
                "m",
                &body(&[json!({"role": "user", "content": "a"})], "s"),
                false,
            )
            .finish(&json!({"status": "completed"}));
        let outside: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            outside,
            ["trace"],
            "nothing is written next to the trace dir"
        );
        let conversation_dir = dir.join(sanitize(".."));
        assert!(conversation_dir.join("0001-response.json").exists());
        fs::remove_dir_all(&root).ok();
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

        let second = trace.begin(
            "k",
            "m",
            &body(&[a.clone(), b.clone(), c.clone()], "sys"),
            false,
        );
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
    fn prefix_metrics_are_exact() {
        let trace = RequestTrace::new(std::env::temp_dir().join("arko-trace-unit-unused"));
        let a = json!({"role": "user", "content": "a"});
        let b = json!({"role": "assistant", "content": "b"});
        let c = json!({"role": "user", "content": "c"});
        let prefix = |items: &[Value], instructions: &str| {
            let call = trace.begin("k", "m", &body(items, instructions), false);
            let prefix = call.prefix.clone();
            trace.remember_input_tokens("k", call.seq, Some(100));
            std::mem::forget(call);
            prefix
        };
        let (sys, sys2) = (hash_value(&json!("sys")), hash_value(&json!("sys2")));
        let tools = hash_value(&json!([{"name": "t"}]));

        assert_eq!(
            prefix(std::slice::from_ref(&a), "sys"),
            json!({
                "first_request": true, "items": 1, "input_bytes": 31,
                "instructions_sha256": sys, "tools_sha256": tools,
            })
        );
        assert_eq!(
            prefix(&[a.clone(), b.clone(), c.clone()], "sys"),
            json!({
                "first_request": false, "previous_is_prefix": true,
                "common_items": 1, "previous_items": 1, "items": 3,
                "common_input_bytes": 30, "previous_input_bytes": 31, "input_bytes": 96,
                "instructions_sha256": sys, "tools_sha256": tools,
                "instructions_changed": false, "tools_changed": false,
            })
        );
        let rewritten = json!({"role": "assistant", "content": "B"});
        assert_eq!(
            prefix(&[a.clone(), rewritten, c.clone()], "sys2"),
            json!({
                "first_request": false, "previous_is_prefix": false,
                "common_items": 1, "previous_items": 3, "items": 3,
                "common_input_bytes": 43, "previous_input_bytes": 96, "input_bytes": 96,
                "instructions_sha256": sys2, "tools_sha256": tools,
                "instructions_changed": true, "tools_changed": false,
            })
        );
        assert_eq!(
            prefix(std::slice::from_ref(&a), "sys2"),
            json!({
                "first_request": false, "previous_is_prefix": false,
                "common_items": 1, "previous_items": 3, "items": 1,
                "common_input_bytes": 30, "previous_input_bytes": 96, "input_bytes": 31,
                "instructions_sha256": sys2, "tools_sha256": tools,
                "instructions_changed": false, "tools_changed": false,
            })
        );
    }

    /// Only the most recently used conversations keep their serialized
    /// input. The others keep item hashes, so every metric but
    /// `common_input_bytes` stays exact for them.
    #[test]
    fn retained_input_is_bounded() {
        let dir = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let trace = RequestTrace::new(&dir);
        let item =
            |n: usize| json!({"role": "user", "content": format!("{n}:{}", "x".repeat(1000))});
        let first = |key: &str| {
            let call = trace.begin(key, "m", &body(&[item(0)], "s"), false);
            std::mem::forget(call);
        };
        for n in 0..10 * RETAINED_INPUTS {
            first(&format!("conversation-{n}"));
        }
        let one_input = json!([item(0)]).to_string().len();
        assert!(
            trace.retained_input_bytes() <= RETAINED_INPUTS * one_input,
            "{}",
            trace.retained_input_bytes()
        );

        // The oldest conversation lost its bytes but not its item hashes.
        let next = trace.begin(
            "conversation-0",
            "m",
            &body(&[item(0), item(1)], "s"),
            false,
        );
        assert_eq!(next.prefix["previous_is_prefix"], true);
        assert_eq!(next.prefix["common_items"], 1);
        assert_eq!(next.prefix["previous_input_bytes"], one_input);
        assert!(next.prefix["common_input_bytes"].is_null());
        std::mem::forget(next);
        // A recent one compares bytes as before.
        let recent = format!("conversation-{}", 10 * RETAINED_INPUTS - 1);
        let next = trace.begin(&recent, "m", &body(&[item(0), item(1)], "s"), false);
        assert_eq!(next.prefix["common_input_bytes"], one_input - 1);
        std::mem::forget(next);

        // A call that fails after its conversation was evicted restores the
        // previous request without its bytes.
        first("restored");
        let failing = trace.begin("restored", "m", &body(&[item(0), item(1)], "s"), false);
        for n in 0..RETAINED_INPUTS {
            first(&format!("other-{n}"));
        }
        failing.fail("rate limited", None);
        assert!(
            trace.retained_input_bytes() <= RETAINED_INPUTS * one_input,
            "{}",
            trace.retained_input_bytes()
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_finished_call_writes_files_and_a_summary_with_previous_found() {
        let dir = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let trace = RequestTrace::new(&dir);
        let item = json!({"role": "user", "content": "a"});

        let first = trace.begin(
            "arko:conversation:x",
            "gpt-6-astra",
            &body(std::slice::from_ref(&item), "s"),
            true,
        );
        first.finish(&json!({"id": "resp_0123456789abcdef99", "status": "completed", "model": "gpt-6-astra",
            "output": [], "usage": {"input_tokens": 4000, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 10}}));

        let mut second = trace.begin(
            "arko:conversation:x",
            "gpt-6-astra",
            &body(&[item.clone(), item], "s"),
            true,
        );
        second.on_event(
            "response.created",
            &json!({"response": {"model": "gpt-5.6-luna"}}),
        );
        second.on_event(
            "response.output_item.done",
            &json!({"item": {"type": "reasoning", "encrypted_content": "e"}}),
        );
        second.finish(&json!({"id": "resp_0123456789abcdef77", "status": "completed", "model": "gpt-5.6-luna",
            "output": [], "usage": {"input_tokens": 4500, "input_tokens_details": {"cached_tokens": 3968}, "output_tokens": 10}}));

        let lines: Vec<Value> = fs::read_to_string(dir.join("calls.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
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
        assert_eq!(
            lines[1]["conversation_dir"],
            "arko_conversation_x-ed168f36ad25bfa2"
        );
        let conversation_dir = lines[1]["conversation_dir"].as_str().unwrap();
        assert!(
            dir.join(conversation_dir)
                .join("0002-response.json")
                .exists()
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_retry_is_compared_with_the_call_before_the_failed_attempt() {
        let dir = std::env::temp_dir().join(format!("arko-trace-unit-{}", uuid::Uuid::new_v4()));
        let trace = RequestTrace::new(&dir);
        let (a, b) = (
            json!({"role": "user", "content": "a"}),
            json!({"role": "user", "content": "b"}),
        );
        let usage = |input: u64, cached: u64| {
            json!({"status": "completed",
            "usage": {"input_tokens": input, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": 1}})
        };

        trace
            .begin("k", "m", &body(std::slice::from_ref(&a), "s"), true)
            .finish(&usage(4000, 0));
        trace
            .begin("k", "m", &body(&[a.clone(), b.clone()], "s"), true)
            .fail("rate limited", None);
        let retry = trace.begin("k", "m", &body(&[a.clone(), b.clone()], "s"), true);
        assert_eq!(retry.seq, 3);
        assert_eq!(retry.prefix["previous_items"], 1);
        assert_eq!(retry.previous_input_tokens, Some(4000));
        retry.finish(&usage(4500, 3968));

        let lines: Vec<Value> = fs::read_to_string(dir.join("calls.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[1]["status"], "error");
        assert_eq!(lines[2]["previous_found"], true);
        assert_eq!(lines[2]["turn_state_sent"], false);
        fs::remove_dir_all(&dir).ok();
    }
}
