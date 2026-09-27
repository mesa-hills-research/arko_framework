//! Measures prompt-cache behavior on the ChatGPT OAuth (Codex backend) path.
//!
//! Runs several agent-like tool-loop conversations in parallel, traces every
//! call with [`RequestTrace`], and summarizes cache hits per conversation.
//! The key metric is `prev_found`: the share of calls whose cached prefix
//! covers the previous request of the same conversation (within 1,024
//! tokens), which reflects whether requests reach the server holding that
//! conversation's cache. Raw hit rates depend on run length (every call adds
//! uncacheable tokens), so the summary also reports the `ceiling` (the share
//! of input the previous call of the conversation could have cached),
//! `efficiency` (hit rate / ceiling) and `prev_input_cached` (how much of the
//! previous call's input was cached).
//!
//! Credentials come from `~/.codex/auth.json` (`$CODEX_HOME/auth.json`, or
//! `--auth-file`), kept in step with the file by arko_ai's
//! [`CodexAuthStore`]. The access token is used as is while valid. Before it
//! expires, the file is re-read, and tokens Codex refreshed in the meantime
//! are adopted; otherwise it is refreshed. Every refresh (this one, or the
//! provider's own after a 401) is written back at once, before any
//! rate-limit wait: the previous contents go to `auth.json.bak` (one backup,
//! replaced each time), the file is replaced atomically, and newer tokens
//! written by Codex are never overwritten. Refreshed tokens that can't be
//! written go to an owner-only `auth.json.recovered-<unix seconds>` next to
//! it, and the run stops. Tokens and the account id are never printed.
//!
//! By default requests go out exactly as the provider builds them (Codex
//! parity: identity headers, turn state, verbatim output-item replay,
//! `reasoning.context`). `--legacy` reproduces the provider as it was before
//! that fix; the other experiment flags re-enable one piece each on top of
//! `--legacy` (they are the default otherwise). Adjustments go through an
//! [`OpenAIRequestHook`]:
//!
//! ```text
//! cargo run -p arko_ai --example oauth_cache_probe -- --out <dir>              # provider defaults
//! cargo run -p arko_ai --example oauth_cache_probe -- --out <dir> --legacy     # before the fix
//! cargo run -p arko_ai --example oauth_cache_probe -- --out <dir> --legacy --session-id-header
//! cargo run -p arko_ai --example oauth_cache_probe -- --summarize-only --out <dir>
//! ```
//!
//! A short-term rate limit (HTTP 429 that isn't a plan or quota limit) is
//! retried after its `Retry-After` delay, up to three times per call; every
//! attempt is traced. Plan limits, 401 and 403 stop every conversation.

use arko_ai::{
    ChatGptOAuthCredentials, ChatGptOAuthStore, CodexAuthFile, CodexAuthStore, Content, Error,
    GenerationConfig, GenerationResponse, LLMClient, Model, OpenAIEffort, OpenAIRequestHook,
    Provider, ReasoningConfig, RequestTrace, StreamEvent, ToolConfig, ToolDefinition, ToolResult,
    providers::openai::OpenAIClient,
};
use clap::Parser;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Parser, Debug, Clone)]
#[command(about = "Measure ChatGPT OAuth prompt caching with a traced tool-loop workload")]
struct Args {
    /// Trace and summary directory; must be new or empty unless --summarize-only.
    #[arg(long)]
    out: PathBuf,
    /// Codex credentials file (default: ~/.codex/auth.json).
    #[arg(long)]
    auth_file: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    conversations: usize,
    #[arg(long, default_value_t = 15)]
    turns: usize,
    #[arg(long, default_value = "gpt-6-astra")]
    model: String,
    /// none, minimal, low, medium, high, xhigh or max.
    #[arg(long, default_value = "xhigh")]
    effort: String,
    /// Label mixed into the task text so separate runs don't share cached prefixes
    /// beyond the instructions and tools.
    #[arg(long, default_value = "baseline")]
    label: String,
    /// Only recompute the summary from an existing trace in --out.
    #[arg(long)]
    summarize_only: bool,

    /// Reproduce the provider as it was before the Codex-parity fix: no
    /// identity headers, `OpenAI-Beta: responses=experimental`, no turn state,
    /// no `Accept` header, no `reasoning.context`, and the old replay (the
    /// assistant output came from `response.completed`, which the ChatGPT
    /// backend leaves empty: no text, reasoning or output items; tool calls
    /// from the stream, each followed by an empty assistant message). The
    /// flags below re-enable single pieces on top of it.
    #[arg(long)]
    legacy: bool,
    /// With --legacy: send `session-id: <prompt_cache_key>`.
    #[arg(long)]
    session_id_header: bool,
    /// With --legacy: send `thread-id: <prompt_cache_key>`.
    #[arg(long)]
    thread_id_header: bool,
    /// With --legacy: send `x-client-request-id: <prompt_cache_key>`.
    #[arg(long)]
    client_request_id_header: bool,
    /// With --legacy: echo the turn's `x-codex-turn-state`.
    #[arg(long)]
    echo_turn_state: bool,
    /// With --legacy: omit `OpenAI-Beta: responses=experimental`.
    #[arg(long)]
    no_legacy_beta_header: bool,
    /// With --legacy: replay the output items assembled from
    /// `response.output_item.done`, verbatim and in order.
    #[arg(long)]
    replay_output_items: bool,
    /// Override `reasoning.context` (`auto`, `current_turn`, `all_turns`), or
    /// `omit` to leave it out. Default: what the provider sends (`all_turns`
    /// for GPT-5.6 and GPT-6), or nothing with --legacy.
    #[arg(long)]
    reasoning_context: Option<String>,
}

/// What requests look like in a run, after --legacy and the flags.
#[derive(Clone, Debug, PartialEq)]
struct Settings {
    legacy: bool,
    session_id_header: bool,
    thread_id_header: bool,
    client_request_id_header: bool,
    turn_state: bool,
    legacy_beta_header: bool,
    /// The provider sends `Accept: text/event-stream`; the old one didn't.
    accept_header: bool,
    replay_output_items: bool,
    /// `None` keeps what the provider sends; `Some("omit")` removes it.
    reasoning_context: Option<String>,
}

impl Settings {
    fn new(args: &Args) -> Self {
        let legacy = args.legacy;
        Self {
            legacy,
            session_id_header: !legacy || args.session_id_header,
            thread_id_header: !legacy || args.thread_id_header,
            client_request_id_header: !legacy || args.client_request_id_header,
            turn_state: !legacy || args.echo_turn_state,
            legacy_beta_header: legacy && !args.no_legacy_beta_header,
            accept_header: !legacy,
            replay_output_items: !legacy || args.replay_output_items,
            reasoning_context: args
                .reasoning_context
                .clone()
                .or_else(|| legacy.then(|| "omit".to_string())),
        }
    }

    fn manifest(&self) -> Value {
        json!({
            "legacy": self.legacy,
            "session_id_header": self.session_id_header,
            "thread_id_header": self.thread_id_header,
            "client_request_id_header": self.client_request_id_header,
            "turn_state": self.turn_state,
            "legacy_beta_header": self.legacy_beta_header,
            "accept_header": self.accept_header,
            "replay_output_items": self.replay_output_items,
            "reasoning_context": self.reasoning_context.as_deref().unwrap_or("provider default"),
        })
    }
}

/// Probe-local request changes for A/B arms. With the default settings,
/// requests are sent exactly as the provider builds them.
struct ProbeHook {
    settings: Settings,
}

impl OpenAIRequestHook for ProbeHook {
    fn prepare(&self, _conversation_key: &str, headers: &mut HeaderMap, body: &mut Value) {
        let settings = &self.settings;
        for (name, keep) in [
            ("session-id", settings.session_id_header),
            ("thread-id", settings.thread_id_header),
            ("x-client-request-id", settings.client_request_id_header),
            ("x-codex-turn-state", settings.turn_state),
            ("accept", settings.accept_header),
        ] {
            if !keep {
                headers.remove(name);
            }
        }
        if settings.legacy_beta_header {
            headers.insert(
                HeaderName::from_static("openai-beta"),
                HeaderValue::from_static("responses=experimental"),
            );
        }
        if let Some(context) = &settings.reasoning_context
            && let Some(reasoning) = body.get_mut("reasoning").and_then(Value::as_object_mut)
        {
            if context == "omit" {
                reasoning.remove("context");
            } else {
                reasoning.insert("context".into(), Value::String(context.clone()));
            }
        }
        if !settings.replay_output_items
            && let Some(input) = body.get_mut("input").and_then(Value::as_array_mut)
        {
            insert_legacy_empty_messages(input);
        }
    }
}

/// The pre-fix provider kept only what `response.completed.output` held,
/// which the ChatGPT backend leaves empty: the message had no text,
/// reasoning or id, and its tool calls came from the stream.
fn legacy_response(mut response: GenerationResponse) -> GenerationResponse {
    response.message.content = vec![Content::Text(String::new())];
    response.message.reasoning_artifact = None;
    response.message.id = None;
    response
}

/// The pre-fix provider replayed an empty assistant message after each
/// turn's function calls; the provider no longer does.
fn insert_legacy_empty_messages(input: &mut Vec<Value>) {
    let empty = json!({"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": ""}]});
    let mut index = 0;
    while index < input.len() {
        let ends_calls = input[index]["type"] == "function_call"
            && input.get(index + 1).is_none_or(|next| next["type"] != "function_call");
        let followed_by_assistant = input
            .get(index + 1)
            .is_some_and(|next| next["type"] == "message" && next["role"] == "assistant");
        if ends_calls && !followed_by_assistant {
            input.insert(index + 1, empty.clone());
            index += 1;
        }
        index += 1;
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    if let Err(message) = run(args).await {
        eprintln!("oauth_cache_probe: {message}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<(), String> {
    if args.summarize_only {
        let summary = summarize(&args.out)?;
        write_summary(&args.out, &summary)?;
        return Ok(());
    }
    let settings = Settings::new(&args);
    if !args.legacy
        && (args.session_id_header
            || args.thread_id_header
            || args.client_request_id_header
            || args.echo_turn_state
            || args.no_legacy_beta_header
            || args.replay_output_items)
    {
        eprintln!("oauth_cache_probe: without --legacy, the header and replay flags are already the default");
    }
    if args.out.exists() && fs::read_dir(&args.out).map_err(|e| e.to_string())?.next().is_some() {
        return Err(format!("{} is not empty; choose a new --out directory", args.out.display()));
    }
    let effort = parse_effort(&args.effort)?;
    let model = Model::get(Provider::OpenAI, &args.model).map_err(|e| e.to_string())?;
    let auth_path = args
        .auth_file
        .clone()
        .or_else(CodexAuthFile::default_path)
        .ok_or("no home directory to find ~/.codex/auth.json in; pass --auth-file")?;
    let tokens = TokenKeeper::load(&auth_path)?;
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;

    let hook = Arc::new(ProbeHook { settings: settings.clone() });
    let client = tokens
        .client()?
        .with_request_trace(RequestTrace::new(&args.out))
        .with_openai_request_hook(hook);
    tokens.ensure_fresh(&client).await?;

    let started = SystemTime::now();
    let manifest = json!({
        "started_unix_ms": unix_ms(started),
        "model": args.model,
        "effort": args.effort,
        "conversations": args.conversations,
        "turns": args.turns,
        "label": args.label,
        "experiment": settings.manifest(),
        "endpoint": "chatgpt.com/backend-api/codex/responses (ChatGPT OAuth, HTTP SSE)",
    });
    fs::write(args.out.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap())
        .map_err(|e| e.to_string())?;
    println!("oauth_cache_probe: {} conversations x {} turns on {} ({}){} -> {}",
        args.conversations, args.turns, args.model, args.effort,
        if settings.legacy { ", legacy" } else { "" }, args.out.display());

    let stop = Arc::new(AtomicBool::new(false));
    let mut tasks = Vec::new();
    for index in 0..args.conversations {
        let (client, model, args, stop, tokens) =
            (client.clone(), model.clone(), args.clone(), stop.clone(), tokens.clone());
        let legacy_replay = !settings.replay_output_items;
        tasks.push(tokio::spawn(async move {
            run_conversation(index, &client, model, effort, &args, legacy_replay, &stop, &tokens).await
        }));
    }
    let mut outcomes = Vec::new();
    for task in tasks {
        outcomes.push(task.await.unwrap_or_else(|e| format!("task panicked: {e}")));
    }
    for (index, outcome) in outcomes.iter().enumerate() {
        println!("conversation {index}: {outcome}");
    }
    let mut summary = summarize(&args.out)?;
    summary["outcomes"] = json!(outcomes);
    summary["token_refreshes"] = json!(tokens.refreshes());
    write_summary(&args.out, &summary)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_conversation(
    index: usize,
    client: &LLMClient,
    model: Model,
    effort: OpenAIEffort,
    args: &Args,
    legacy_replay: bool,
    stop: &AtomicBool,
    tokens: &TokenKeeper,
) -> String {
    let mut conversation = client.new_conversation(model).with_system_message(&instructions());
    conversation.add_user_message(&task_prompt(index, &args.label));
    let config = GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openai(effort, false))
        .with_tool_config(ToolConfig::new(tools()).with_parallel_tool_calls(false));

    let mut consecutive_errors = 0;
    let mut turn = 0;
    while turn < args.turns {
        if stop.load(Ordering::SeqCst) {
            return format!("stopped after {turn} turns (another conversation hit a stop condition)");
        }
        if let Err(message) = tokens.ensure_fresh(client).await {
            stop.store(true, Ordering::SeqCst);
            return format!("stopped: {message}");
        }
        match generate(index, client, &conversation, &config, tokens).await {
            Ok(response) => {
                consecutive_errors = 0;
                let calls = response.message.tool_calls.clone();
                let response = if legacy_replay { legacy_response(response) } else { response };
                conversation.add_assistant_message_from_response(response);
                if calls.is_empty() {
                    conversation.add_user_message(
                        "Continue the audit: call exactly one tool now. Do not answer in prose.",
                    );
                } else {
                    let results = calls
                        .iter()
                        .map(|call| ToolResult::from_tool_call(call, &tool_output(index, turn, &call.name)))
                        .collect();
                    conversation.add_tool_results(results);
                }
                turn += 1;
                println!("conversation {index}: turn {turn}/{}", args.turns);
            }
            Err(error) => {
                let text = error.to_string();
                // A short-term rate limit that outlasted its retries also
                // stops the run: the account is being throttled.
                if matches!(classify(&error), Failure::StopAll | Failure::RateLimited(_)) {
                    stop.store(true, Ordering::SeqCst);
                    return format!("stopped all conversations after {turn} turns: {}", error_text(&error, 300));
                }
                consecutive_errors += 1;
                eprintln!("conversation {index}: error on turn {}: {}", turn + 1, truncate(&text, 300));
                if consecutive_errors >= 2 {
                    return format!("gave up after {turn} turns: {}", truncate(&text, 300));
                }
            }
        }
    }
    format!("completed {turn} turns")
}

/// One model call. A short-term rate limit is retried after its delay, up
/// to [`RATE_LIMIT_RETRIES`] times; each attempt is traced as its own call.
async fn generate(
    index: usize,
    client: &LLMClient,
    conversation: &arko_ai::Conversation,
    config: &GenerationConfig,
    tokens: &TokenKeeper,
) -> Result<GenerationResponse, Error> {
    retry_rate_limits(
        index,
        client,
        tokens,
        || generate_once(client, conversation, config),
        tokio::time::sleep,
    )
    .await
}

/// Runs `attempt`, retrying short-term rate limits after `sleep`ing for
/// their delay. Tokens the provider refreshes during an attempt are saved
/// as it refreshes, before any wait; a failure to save them stops the run.
async fn retry_rate_limits<T, Attempt, Sleep>(
    index: usize,
    client: &LLMClient,
    tokens: &TokenKeeper,
    mut attempt: impl FnMut() -> Attempt,
    mut sleep: impl FnMut(Duration) -> Sleep,
) -> Result<T, Error>
where
    Attempt: std::future::Future<Output = Result<T, Error>>,
    Sleep: std::future::Future<Output = ()>,
{
    let mut retries = 0;
    loop {
        let result = attempt().await;
        // Tokens refreshed (by any conversation) that couldn't be saved stop
        // the run.
        tokens.check().map_err(Error::AuthenticationFailed)?;
        let error = match result {
            Ok(response) => return Ok(response),
            Err(error) => error,
        };
        let Failure::RateLimited(hint) = classify(&error) else {
            return Err(error);
        };
        if retries == RATE_LIMIT_RETRIES {
            return Err(error);
        }
        retries += 1;
        let delay = rate_limit_delay(hint, arko_ai::retry::random_sample());
        eprintln!(
            "conversation {index}: rate limited ({}); retry {retries}/{RATE_LIMIT_RETRIES} in {:.1}s",
            truncate(&error.to_string(), 160),
            delay.as_secs_f64(),
        );
        sleep(delay).await;
        tokens.ensure_fresh(client).await.map_err(Error::AuthenticationFailed)?;
    }
}

async fn generate_once(
    client: &LLMClient,
    conversation: &arko_ai::Conversation,
    config: &GenerationConfig,
) -> Result<GenerationResponse, Error> {
    let mut stream = client.generate_stream(conversation, config.clone()).await?;
    while let Some(event) = stream.next().await {
        if let StreamEvent::End(response) = event? {
            return Ok(*response);
        }
    }
    Err(Error::StreamTruncated("stream ended without a response".into()))
}

const RATE_LIMIT_RETRIES: u32 = 3;
/// Wait when a rate limit comes without a delay hint.
const DEFAULT_RATE_LIMIT_WAIT: Duration = Duration::from_secs(30);
const MAX_RATE_LIMIT_WAIT: Duration = Duration::from_secs(120);

/// Error codes of plan, quota and spend limits, which waiting a minute
/// doesn't lift (the codes Codex maps to usage-limit and quota errors).
const PLAN_LIMIT_CODES: &[&str] = &[
    "usage_limit_reached",
    "usage_not_included",
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
];

#[derive(Debug, PartialEq)]
enum Failure {
    /// A plan limit or an authentication failure: stop every conversation.
    StopAll,
    /// A short-term rate limit, with the server's delay hint.
    RateLimited(Option<Duration>),
    Other,
}

fn classify(error: &Error) -> Failure {
    let text = error.to_string();
    if PLAN_LIMIT_CODES.iter().any(|code| text.contains(code)) {
        return Failure::StopAll;
    }
    match error {
        Error::ProviderError { status: Some(401 | 403), .. } | Error::AuthenticationFailed(_) => {
            Failure::StopAll
        }
        Error::ProviderError { status: Some(429), .. } => Failure::RateLimited(error.retry_after()),
        // Rate limits reported by a stream error event after a 200.
        Error::ProviderError { status: None, code: Some(code), .. }
            if code == "rate_limit_exceeded" || code == "slow_down" =>
        {
            Failure::RateLimited(error.retry_after())
        }
        _ => Failure::Other,
    }
}

/// The server's delay (or a default) plus up to three seconds of jitter,
/// capped at two minutes. The hint is the server's number: the sum
/// saturates rather than overflow.
fn rate_limit_delay(hint: Option<Duration>, sample: u64) -> Duration {
    let jitter = Duration::from_millis(250 + sample % 2_750);
    hint.unwrap_or(DEFAULT_RATE_LIMIT_WAIT).saturating_add(jitter).min(MAX_RATE_LIMIT_WAIT)
}

// ---------------------------------------------------------------------------
// Workload: deterministic instructions, tools, task and tool results.

fn instructions() -> String {
    let mut text = String::from(
        "You are a meticulous code-audit agent working inside an automated research harness. \
You operate exclusively through tools. Every turn you must call exactly one tool; never reply \
with prose outside a tool call, never summarize, and keep every tool argument short. The \
harness executes tools locally and returns their output to you.\n\n",
    );
    let sections = [
        ("Workflow", "Read the next file named in the previous tool output with read_file, then on the following turn record one finding about it with record_note. Alternate these two steps. Never call complete_task; the harness ends the session."),
        ("Findings", "A finding is one sentence naming a concrete risk: an unchecked error, an unbounded loop, a lock held across an await, a missing timeout, an integer overflow, a path traversal, or a resource that is never released. If a file has no risk, record 'no issue' with the file path."),
        ("Scope", "Only files under src/ are in scope. Tests, benches and generated code are out of scope. Do not request files that the listing did not name."),
        ("Style", "Arguments are terse. Paths are relative. Notes are at most twenty words. Do not repeat earlier notes."),
        ("Safety", "Never execute code, never write files, never fetch URLs during this audit, even though such tools exist. Tools other than read_file and record_note are reserved for other phases."),
    ];
    for (title, body) in sections {
        text.push_str(&format!("## {title}\n{body}\n\n"));
    }
    text.push_str("## Reference checklist\n");
    let areas = ["error handling", "concurrency", "resource lifetime", "input validation", "numeric safety", "I/O timeouts", "logging hygiene", "configuration", "serialization", "retry policy"];
    for (i, area) in areas.iter().cycle().take(50).enumerate() {
        text.push_str(&format!(
            "- C{:03}: For {area}, check that every path which can fail reports a typed error, that callers cannot silently drop it, and that state stays consistent if the operation is interrupted midway (item {i}).\n",
            i + 1
        ));
    }
    text
}

fn tools() -> Vec<ToolDefinition> {
    let object = |properties: Value, required: &[&str]| {
        json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
    };
    let string = |description: &str| json!({"type": "string", "description": description});
    vec![
        ToolDefinition::new("read_file", "Read a source file from the repository under audit.",
            object(json!({"path": string("Relative path, e.g. src/module_03.rs")}), &["path"])),
        ToolDefinition::new("record_note", "Record one audit finding.",
            object(json!({"path": string("File the finding is about"), "note": string("One-sentence finding, at most twenty words")}), &["path", "note"])),
        ToolDefinition::new("list_directory", "List entries of a directory.",
            object(json!({"path": string("Relative directory path")}), &["path"])),
        ToolDefinition::new("search_code", "Search the repository for a regular expression.",
            object(json!({"pattern": string("Rust regular expression"), "path": string("Optional directory to limit the search")}), &["pattern"])),
        ToolDefinition::new("run_tests", "Run the test suite or a filtered subset (reserved for the verification phase).",
            object(json!({"filter": string("Test name filter")}), &[])),
        ToolDefinition::new("write_file", "Create or overwrite a file (reserved for the repair phase).",
            object(json!({"path": string("Relative path"), "content": string("Full file content")}), &["path", "content"])),
        ToolDefinition::new("apply_patch", "Apply a unified diff (reserved for the repair phase).",
            object(json!({"patch": string("Unified diff")}), &["patch"])),
        ToolDefinition::new("query_database", "Run a read-only SQL query against the audit database.",
            object(json!({"sql": string("SQL statement")}), &["sql"])),
        ToolDefinition::new("fetch_url", "Fetch a URL (reserved for the research phase).",
            object(json!({"url": string("Absolute URL")}), &["url"])),
        ToolDefinition::new("complete_task", "Finish the session with a final report (the harness decides when; do not call).",
            object(json!({"report": string("Final report")}), &["report"])),
    ]
}

fn task_prompt(index: usize, label: &str) -> String {
    format!(
        "Audit session {label}-{index}. Repository: service-{index} (a Rust network service). \
Listing: src/module_01.rs through src/module_40.rs. Start by calling read_file on src/module_01.rs."
    )
}

/// Deterministic, realistic tool output of about 2.5k tokens per call.
fn tool_output(conversation: usize, turn: usize, tool: &str) -> String {
    let module = turn / 2 + 1;
    if tool == "record_note" {
        return format!(
            "Note recorded.\n\nNext: call read_file on src/module_{:02}.rs.\n\n{}",
            module + 1,
            synthetic_source(conversation, turn, 170)
        );
    }
    format!(
        "// src/module_{module:02}.rs (service-{conversation})\n{}\n\nNext: call record_note with one finding about src/module_{module:02}.rs.",
        synthetic_source(conversation, turn, 190)
    )
}

fn synthetic_source(conversation: usize, turn: usize, lines: usize) -> String {
    let mut state = (conversation as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (turn as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let idents = ["buffer", "handle", "config", "request", "session", "cursor", "payload", "limit", "retry", "offset", "window", "ledger"];
    let types = ["u32", "usize", "String", "Vec<u8>", "Option<Duration>", "Arc<Mutex<State>>", "HashMap<String, u64>"];
    let mut text = String::new();
    for line in 0..lines {
        let a = idents[(next() % idents.len() as u64) as usize];
        let b = idents[(next() % idents.len() as u64) as usize];
        let t = types[(next() % types.len() as u64) as usize];
        let n = next() % 4096;
        let code = match line % 7 {
            0 => format!("pub fn {a}_{b}_{n}(input: &{t}) -> Result<{t}, Error> {{"),
            1 => format!("    let {a} = self.{b}.get({n}).ok_or(Error::Missing(\"{b}\"))?;"),
            2 => format!("    if {a}.len() > {n} {{ return Err(Error::Limit({n})); }}"),
            3 => format!("    let guard = self.{b}.lock().await; // held while calling {a}"),
            4 => format!("    for item in {a}.iter().skip({n}) {{ total += item.{b}; }}"),
            5 => format!("    log::debug!(\"{a} {b} {{}}\", {n}); Ok({a}.clone())"),
            _ => "}".to_string(),
        };
        text.push_str(&code);
        text.push('\n');
    }
    text
}

fn parse_effort(value: &str) -> Result<OpenAIEffort, String> {
    Ok(match value {
        "none" => OpenAIEffort::None,
        "minimal" => OpenAIEffort::Minimal,
        "low" => OpenAIEffort::Low,
        "medium" => OpenAIEffort::Medium,
        "high" => OpenAIEffort::High,
        "xhigh" => OpenAIEffort::XHigh,
        "max" => OpenAIEffort::Max,
        other => return Err(format!("unknown effort {other}")),
    })
}

// ---------------------------------------------------------------------------
// Credentials: ~/.codex/auth.json through arko_ai's `CodexAuthStore`.
//
// ChatGPT refresh tokens rotate: once a refresh succeeds, the previous refresh
// token no longer works, and Codex, which shares the file, is signed out
// unless the new tokens reach it. The client's credential store re-reads the
// file before every refresh (the probe's own, or the provider's after a 401),
// adopting tokens Codex refreshed in the meantime, and saves rotated tokens
// as soon as the refresh succeeds, before any wait. Rotated tokens it can't
// save go to a private recovery file next to auth.json, and the run stops.

/// Refresh when the access token expires within this margin.
const REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// The client's credential store: auth.json's [`CodexAuthStore`], noting
/// each refresh for the summary and stopping the run after a failed save.
struct TokenKeeper {
    store: CodexAuthStore,
    /// auth.json as loaded, for the client's first credentials.
    file: CodexAuthFile,
    refreshes: Mutex<Vec<String>>,
    /// Set when rotated tokens could not be saved to the file: the run
    /// stops. Names the recovery file; never holds a token.
    unsaved: Mutex<Option<String>>,
}

impl TokenKeeper {
    fn load(path: &Path) -> Result<Arc<Self>, String> {
        let file = CodexAuthFile::load(path)
            .map_err(|e| format!("cannot use {}: {e}; sign in with Codex first", path.display()))?;
        Ok(Arc::new(Self {
            store: CodexAuthStore::new(file.clone()),
            file,
            refreshes: Mutex::new(Vec::new()),
            unsaved: Mutex::new(None),
        }))
    }

    /// An OpenAI client using the file's credentials, kept in step with the
    /// file through this store.
    fn openai_client(self: &Arc<Self>) -> Result<OpenAIClient, String> {
        let credentials = self.file.credentials().map_err(|e| e.to_string())?;
        let store: Arc<dyn ChatGptOAuthStore> = self.clone();
        Ok(OpenAIClient::new_chatgpt_oauth(
            &credentials.access_token,
            credentials.refresh_token.as_deref(),
            &credentials.account_id,
        )
        .with_credential_store(Some(store)))
    }

    fn client(self: &Arc<Self>) -> Result<LLMClient, String> {
        Ok(LLMClient::new().with_openai_client(self.openai_client()?))
    }

    fn refreshes(&self) -> Vec<String> {
        self.refreshes.lock().unwrap().clone()
    }

    fn note(&self, outcome: String) {
        eprintln!("oauth_cache_probe: {outcome}");
        self.refreshes.lock().unwrap().push(outcome);
    }

    /// Fails, naming the recovery file, once rotated tokens could not be
    /// saved.
    fn check(&self) -> Result<(), String> {
        match &*self.unsaved.lock().unwrap() {
            Some(message) => Err(message.clone()),
            None => Ok(()),
        }
    }

    /// Refreshes when the access token expires within [`REFRESH_MARGIN`],
    /// unless auth.json holds newer tokens (written by Codex), which are
    /// adopted instead. Refreshed tokens are saved before this returns.
    async fn ensure_fresh(&self, client: &LLMClient) -> Result<(), String> {
        self.check()?;
        let refreshed = client.refresh_openai_oauth_token_if_expiring(REFRESH_MARGIN).await;
        // Also when this call only waited for another conversation's
        // refresh: if that one's tokens couldn't be saved, stop here.
        self.check()?;
        refreshed.map_err(|error| format!("token refresh failed: {}", error_text(&error, 200)))?;
        Ok(())
    }
}

impl ChatGptOAuthStore for TokenKeeper {
    fn before_refresh(
        &self,
        current: &ChatGptOAuthCredentials,
    ) -> Result<Option<ChatGptOAuthCredentials>, Error> {
        let path = self.file.path().display();
        match self.store.before_refresh(current) {
            Ok(newer) => {
                if newer.is_some() {
                    self.note(format!("adopted newer tokens another process wrote to {path}"));
                }
                Ok(newer)
            }
            Err(error) => {
                self.note(format!("cannot use {path} for a refresh: {}", error_message(&error)));
                Err(error)
            }
        }
    }

    fn save_refreshed(&self, refreshed: &ChatGptOAuthCredentials, id_token: Option<&str>) -> Result<(), Error> {
        match self.store.save_refreshed(refreshed, id_token) {
            Ok(()) => {
                self.note(format!(
                    "access token refreshed; new tokens written to {} (previous contents in its .bak)",
                    self.file.path().display()
                ));
                Ok(())
            }
            Err(error) => {
                // The error names the recovery file (never a token).
                let message = format!("{}; stopping", error_message(&error));
                self.note(message.clone());
                *self.unsaved.lock().unwrap() = Some(message);
                Err(error)
            }
        }
    }

    fn refresh_lock(&self) -> Option<Arc<tokio::sync::Mutex<()>>> {
        self.store.refresh_lock()
    }
}

// ---------------------------------------------------------------------------
// Summary from <out>/calls.jsonl.

fn summarize(out: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(out.join("calls.jsonl"))
        .map_err(|e| format!("cannot read {}/calls.jsonl: {e}", out.display()))?;
    let calls: Vec<Value> = text.lines().filter_map(|line| serde_json::from_str(line).ok()).collect();
    let mut by_conversation: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for call in &calls {
        by_conversation
            .entry(call["conversation"].as_str().unwrap_or("?").to_string())
            .or_default()
            .push(call.clone());
    }
    let mut conversations = Vec::new();
    for (key, calls) in &by_conversation {
        let mut stats = stats(calls);
        stats["conversation"] = json!(key);
        let prefixes: std::collections::BTreeSet<&str> =
            calls.iter().filter_map(|c| c["response_id_prefix"].as_str()).collect();
        stats["distinct_response_id_prefixes"] = json!(prefixes.len());
        conversations.push(stats);
    }
    let mut overall = stats(&calls);
    overall["conversations"] = json!(by_conversation.len());
    let manifest = fs::read_to_string(out.join("manifest.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    Ok(json!({"manifest": manifest, "overall": overall, "per_conversation": conversations}))
}

fn stats(calls: &[Value]) -> Value {
    let ok: Vec<&Value> = calls.iter().filter(|c| c["status"] != "error" && c["status"] != "abandoned").collect();
    let sum = |field: &str| ok.iter().filter_map(|c| c[field].as_u64()).sum::<u64>();
    let (input, cached, output, reasoning) =
        (sum("input_tokens"), sum("cached_tokens"), sum("output_tokens"), sum("reasoning_tokens"));
    let warm: Vec<&&Value> = ok.iter().filter(|c| c["seq"].as_u64().unwrap_or(0) >= 3).collect();
    let warm_input: u64 = warm.iter().filter_map(|c| c["input_tokens"].as_u64()).sum();
    let warm_cached: u64 = warm.iter().filter_map(|c| c["cached_tokens"].as_u64()).sum();
    let later: Vec<&&Value> = ok.iter().filter(|c| c["seq"].as_u64().unwrap_or(0) >= 2).collect();
    let zero = later.iter().filter(|c| c["cached_tokens"].as_u64().unwrap_or(0) == 0).count();
    let prev: Vec<bool> = later.iter().filter_map(|c| c["previous_found"].as_bool()).collect();
    let mut shortfalls: Vec<i64> = later
        .iter()
        .filter(|c| c["cached_tokens"].as_u64().unwrap_or(0) > 0)
        .filter_map(|c| c["shortfall_tokens"].as_i64())
        .collect();
    shortfalls.sort_unstable();
    let mut served: BTreeMap<String, u64> = BTreeMap::new();
    let mut reroutes = 0;
    for call in &ok {
        let model = call["completed_model"].as_str().or(call["created_model"].as_str()).unwrap_or("?");
        *served.entry(model.to_string()).or_default() += 1;
        if call["requested_model"].as_str() != Some(model) {
            reroutes += 1;
        }
    }
    let mut completed_ms: Vec<u64> = ok.iter().filter_map(|c| c["timings_ms"]["completed"].as_u64()).collect();
    let mut first_ms: Vec<u64> = ok.iter().filter_map(|c| c["timings_ms"]["first_event"].as_u64()).collect();
    completed_ms.sort_unstable();
    first_ms.sort_unstable();
    let empty_completed_output = ok.iter().filter(|c| c["completed_output_count"].as_u64() == Some(0)).count();
    let streamed_reasoning: u64 =
        ok.iter().filter_map(|c| c["streamed_reasoning_with_encrypted_content"].as_u64()).sum();
    let not_prefix = ok.iter().filter(|c| c["prefix"]["previous_is_prefix"] == false).count();
    let reuse = PrefixReuse::of(&ok);
    let ratio = |a: u64, b: u64| if b == 0 { Value::Null } else { json!(round(a as f64 / b as f64)) };
    json!({
        "calls": calls.len(),
        "successful_calls": ok.len(),
        "failed_calls": calls.len() - ok.len(),
        "input_tokens": input,
        "cached_tokens": cached,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "hit_rate": ratio(cached, input),
        "ceiling": ratio(reuse.reusable, input),
        "efficiency": ratio(cached, reuse.reusable),
        "prev_input_cached": ratio(reuse.previous_cached, reuse.previous_input),
        "warm_hit_rate_seq_ge_3": ratio(warm_cached, warm_input),
        "zero_cache_share_seq_ge_2": ratio(zero as u64, later.len() as u64),
        "prev_found_share": ratio(prev.iter().filter(|p| **p).count() as u64, prev.len() as u64),
        "median_shortfall_on_hits": percentile_i64(&shortfalls, 50),
        "served_models": served,
        "rerouted_calls": reroutes,
        "calls_with_empty_completed_output": empty_completed_output,
        "streamed_reasoning_items_with_encrypted_content": streamed_reasoning,
        "requests_not_extending_previous_input": not_prefix,
        "latency_ms": {
            "first_event_p50": percentile(&first_ms, 50), "first_event_p90": percentile(&first_ms, 90),
            "completed_p50": percentile(&completed_ms, 50), "completed_p90": percentile(&completed_ms, 90),
        },
    })
}

/// How much of each call's input the previous successful call of the same
/// conversation could have left in the cache, summed over calls.
#[derive(Debug, Default, PartialEq)]
struct PrefixReuse {
    /// Σ min(previous input, input): the input a perfect cache would serve.
    reusable: u64,
    /// Σ previous input.
    previous_input: u64,
    /// Σ min(cached, previous input).
    previous_cached: u64,
}

impl PrefixReuse {
    fn of(calls: &[&Value]) -> Self {
        let mut by_conversation: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
        for call in calls {
            by_conversation.entry(call["conversation"].as_str().unwrap_or("?")).or_default().push(call);
        }
        let mut reuse = Self::default();
        for calls in by_conversation.values_mut() {
            calls.sort_by_key(|call| call["seq"].as_u64().unwrap_or(0));
            for pair in calls.windows(2) {
                let previous = pair[0]["input_tokens"].as_u64().unwrap_or(0);
                let input = pair[1]["input_tokens"].as_u64().unwrap_or(0);
                let cached = pair[1]["cached_tokens"].as_u64().unwrap_or(0);
                reuse.reusable += previous.min(input);
                reuse.previous_input += previous;
                reuse.previous_cached += cached.min(previous);
            }
        }
        reuse
    }
}

fn write_summary(out: &Path, summary: &Value) -> Result<(), String> {
    fs::write(out.join("summary.json"), serde_json::to_string_pretty(summary).unwrap()).map_err(|e| e.to_string())?;
    let mut text = String::new();
    let line = |label: &str, s: &Value| {
        format!(
            "{label:<28} calls {:>3}  hit {:>6}  ceiling {:>6}  efficiency {:>6}  prev_input_cached {:>6}  prev_found {:>6}  warm {:>6}  zero {:>6}  shortfall_p50 {:>7}  input {:>8}  cached {:>8}  output {:>6}  served {}\n",
            s["successful_calls"], fmt(&s["hit_rate"]), fmt(&s["ceiling"]), fmt(&s["efficiency"]),
            fmt(&s["prev_input_cached"]), fmt(&s["prev_found_share"]), fmt(&s["warm_hit_rate_seq_ge_3"]),
            fmt(&s["zero_cache_share_seq_ge_2"]),
            s["median_shortfall_on_hits"], s["input_tokens"], s["cached_tokens"], s["output_tokens"],
            s["served_models"]
        )
    };
    text.push_str(&line("overall", &summary["overall"]));
    for conversation in summary["per_conversation"].as_array().into_iter().flatten() {
        let key = conversation["conversation"].as_str().unwrap_or("?");
        text.push_str(&line(&key.chars().rev().take(12).collect::<String>().chars().rev().collect::<String>(), conversation));
    }
    let overall = &summary["overall"];
    text.push_str(&format!(
        "empty response.completed output: {} of {} calls; streamed reasoning items with encrypted_content: {}; rerouted calls: {}; latency {}\n",
        overall["calls_with_empty_completed_output"], overall["successful_calls"],
        overall["streamed_reasoning_items_with_encrypted_content"], overall["rerouted_calls"], overall["latency_ms"]
    ));
    print!("{text}");
    fs::write(out.join("summary.txt"), text).map_err(|e| e.to_string())
}

fn fmt(value: &Value) -> String {
    value.as_f64().map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "-".into())
}

fn round(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn percentile(values: &[u64], p: usize) -> Value {
    if values.is_empty() { Value::Null } else { json!(values[(values.len() - 1) * p / 100]) }
}

fn percentile_i64(values: &[i64], p: usize) -> Value {
    if values.is_empty() { Value::Null } else { json!(values[(values.len() - 1) * p / 100]) }
}

/// An error's own text: an authentication failure without its
/// "Authentication failed: " prefix.
fn error_message(error: &Error) -> String {
    match error {
        Error::AuthenticationFailed(message) => message.clone(),
        other => other.to_string(),
    }
}

/// An error's text, shortened to `max` characters unless it is an
/// authentication failure: those are arko_ai's own messages, which name the
/// recovery file of tokens that could not be saved.
fn error_text(error: &Error, max: usize) -> String {
    match error {
        Error::AuthenticationFailed(_) => error.to_string(),
        _ => truncate(&error.to_string(), max),
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max { text.to_string() } else { text.chars().take(max).collect::<String>() + "…" }
}

fn unix_ms(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_error(status: Option<u16>, body: &str, retry_after: Option<Duration>) -> Error {
        let value: Value = serde_json::from_str(body).unwrap_or_default();
        let error = value.get("error").unwrap_or(&value);
        let code = error.get("code").or_else(|| error.get("type")).and_then(Value::as_str);
        Error::ProviderError {
            message: format!("OpenAI request failed: {body}"),
            status,
            code: code.map(String::from),
            retry_after,
        }
    }

    #[test]
    fn short_term_rate_limits_are_retried_and_plan_limits_stop_the_run() {
        let seconds = Duration::from_secs;
        let rate_limited = provider_error(Some(429), r#"{"detail":"Rate limit exceeded"}"#, Some(seconds(36)));
        assert_eq!(classify(&rate_limited), Failure::RateLimited(Some(seconds(36))));
        let unhinted = provider_error(Some(429), r#"{"error":{"code":"rate_limit_exceeded"}}"#, None);
        assert_eq!(classify(&unhinted), Failure::RateLimited(None));
        let streamed = provider_error(None, r#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#, None);
        assert_eq!(classify(&streamed), Failure::RateLimited(None));

        for body in [
            r#"{"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","resets_in_seconds":3600}}"#,
            r#"{"error":{"type":"usage_not_included"}}"#,
            r#"{"error":{"code":"insufficient_quota"}}"#,
        ] {
            assert_eq!(classify(&provider_error(Some(429), body, Some(seconds(3600)))), Failure::StopAll, "{body}");
        }
        for status in [401, 403] {
            assert_eq!(classify(&provider_error(Some(status), "{}", None)), Failure::StopAll);
        }
        assert_eq!(classify(&Error::AuthenticationFailed("refresh failed".into())), Failure::StopAll);
        assert_eq!(classify(&provider_error(Some(500), r#"{"error":{"code":"server_error"}}"#, None)), Failure::Other);
        assert_eq!(classify(&Error::StreamTruncated("eof".into())), Failure::Other);

        for sample in [0, 1, 2_749, u64::MAX] {
            let delay = rate_limit_delay(Some(seconds(36)), sample);
            assert!(delay >= seconds(36) && delay <= Duration::from_millis(39_000), "{delay:?}");
            assert!(rate_limit_delay(None, sample) >= DEFAULT_RATE_LIMIT_WAIT);
            assert_eq!(rate_limit_delay(Some(seconds(600)), sample), MAX_RATE_LIMIT_WAIT);
        }
    }

    #[test]
    fn ceiling_and_efficiency_use_the_previous_successful_call() {
        let call = |conversation: &str, seq: u64, input: u64, cached: u64, status: &str| json!({
            "conversation": conversation, "seq": seq, "status": status,
            "input_tokens": input, "cached_tokens": cached,
        });
        let calls = [
            call("a", 1, 4_000, 0, "completed"),
            call("a", 2, 0, 0, "error"),
            call("a", 3, 5_000, 3_000, "completed"),
            call("a", 4, 4_500, 4_500, "completed"),
            call("b", 1, 2_000, 1_000, "completed"),
        ];
        let summary = stats(&calls);
        // Pairs: (4000 -> 5000) and (5000 -> 4500); the failed call and each
        // conversation's first call have no previous.
        assert_eq!(summary["ceiling"], json!(round(8_500.0 / 15_500.0)));
        assert_eq!(summary["efficiency"], json!(round(8_500.0 / 8_500.0)));
        assert_eq!(summary["prev_input_cached"], json!(round(7_500.0 / 9_000.0)));
        assert_eq!(summary["hit_rate"], json!(round(8_500.0 / 15_500.0)));
    }

    #[test]
    fn legacy_replay_adds_the_empty_message_after_each_turns_calls() {
        let user = json!({"role": "user", "type": "message", "content": "go"});
        let call = |id: &str| json!({"type": "function_call", "call_id": id});
        let output = |id: &str| json!({"type": "function_call_output", "call_id": id});
        let empty = json!({"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": ""}]});
        let mut input = vec![user.clone(), call("a"), output("a"), call("b"), call("c"), output("b"), output("c"), empty.clone()];
        insert_legacy_empty_messages(&mut input);
        assert_eq!(input, [
            user, call("a"), empty.clone(), output("a"), call("b"), call("c"), empty.clone(),
            output("b"), output("c"), empty,
        ]);
    }

    #[test]
    fn legacy_turns_off_every_fix_and_flags_restore_them_one_by_one() {
        let parse = |flags: &[&str]| {
            let mut argv = vec!["probe", "--out", "x"];
            argv.extend_from_slice(flags);
            Settings::new(&Args::parse_from(argv))
        };
        let current = parse(&[]);
        assert!(current.session_id_header && current.turn_state && current.replay_output_items && current.accept_header);
        assert!(!current.legacy_beta_header);
        assert_eq!(current.reasoning_context, None);

        let legacy = parse(&["--legacy"]);
        assert!(!legacy.session_id_header && !legacy.thread_id_header && !legacy.client_request_id_header);
        assert!(!legacy.turn_state && !legacy.replay_output_items && !legacy.accept_header);
        assert!(legacy.legacy_beta_header);
        assert_eq!(legacy.reasoning_context.as_deref(), Some("omit"));

        let arm = parse(&["--legacy", "--session-id-header", "--replay-output-items"]);
        assert!(arm.session_id_header && arm.replay_output_items && !arm.thread_id_header);
        assert_eq!(parse(&["--replay-output-items"]), current);
        assert_eq!(parse(&["--reasoning-context", "current_turn"]).reasoning_context.as_deref(), Some("current_turn"));
    }

    #[test]
    fn huge_rate_limit_hints_are_capped_without_overflow() {
        for sample in [0, 2_749, u64::MAX] {
            assert_eq!(rate_limit_delay(Some(Duration::from_secs(u64::MAX)), sample), MAX_RATE_LIMIT_WAIT);
            assert_eq!(rate_limit_delay(Some(Duration::MAX), sample), MAX_RATE_LIMIT_WAIT);
        }
    }

    // Credential handling, on temporary copies of a Codex auth.json with
    // fake tokens. Refreshes go to a local server standing in for the token
    // endpoint and the Responses API (the client uses the system proxy
    // settings, so a proxy must exempt 127.0.0.1). Refresh tokens are unique
    // per test: the process's record of spent tokens is shared by every test.

    fn now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    }

    /// A JWT-shaped fake access token expiring at `exp`.
    fn jwt(tag: &str, exp: u64) -> String {
        use base64::Engine;
        let claims = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(json!({"exp": exp, "tag": tag}).to_string());
        format!("e30.{claims}.sig")
    }

    fn auth_json(access: &str, refresh: &str) -> String {
        serde_json::to_string_pretty(&json!({
            "OPENAI_API_KEY": null,
            "auth_mode": "chatgpt",
            "last_refresh": "2026-09-21T06:33:44.750634434Z",
            "tokens": {"id_token": "id.old.sig", "access_token": access, "refresh_token": refresh, "account_id": "acct"},
        }))
        .unwrap()
    }

    fn auth_file(access: &str, refresh: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        fs::write(&path, auth_json(access, refresh)).unwrap();
        (dir, path)
    }

    /// (access token, refresh token) in a credentials file.
    fn on_disk(path: &Path) -> (String, String) {
        let value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let token = |name: &str| value["tokens"][name].as_str().unwrap().to_string();
        (token("access_token"), token("refresh_token"))
    }

    fn file_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> =
            fs::read_dir(dir).unwrap().map(|entry| entry.unwrap().file_name().into_string().unwrap()).collect();
        names.sort();
        names
    }

    fn counter() -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::new(std::sync::atomic::AtomicUsize::new(0))
    }

    /// The token endpoint's answer to a refresh.
    fn rotated(access: &str, refresh: &str) -> (u16, String) {
        (200, json!({"access_token": access, "refresh_token": refresh, "id_token": "id.new.sig"}).to_string())
    }

    /// A local HTTP server answering each request with `answer(path)`
    /// (status and JSON body), one connection at a time.
    struct Server {
        url: String,
        paths: Arc<Mutex<Vec<String>>>,
    }

    impl Server {
        async fn start(answer: impl Fn(&str) -> (u16, String) + Send + 'static) -> Self {
            use tokio::io::AsyncWriteExt;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let paths = Arc::new(Mutex::new(Vec::new()));
            let seen = paths.clone();
            tokio::spawn(async move {
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let path = read_request(&mut socket).await;
                    seen.lock().unwrap().push(path.clone());
                    let (status, body) = answer(&path);
                    let reply = format!(
                        "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    socket.write_all(reply.as_bytes()).await.unwrap();
                }
            });
            Self { url, paths }
        }

        /// How many requests went to `path`.
        fn requests(&self, path: &str) -> usize {
            self.paths.lock().unwrap().iter().filter(|seen| *seen == path).count()
        }

        /// The probe's client, sending its requests and refreshes here.
        fn client(&self, tokens: &Arc<TokenKeeper>) -> LLMClient {
            let openai = tokens
                .openai_client()
                .unwrap()
                .with_base_url(&self.url)
                .with_oauth_token_url(format!("{}/token", self.url));
            LLMClient::new().with_openai_client(openai)
        }
    }

    /// Reads one request, body included; returns its path.
    async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        loop {
            let read = socket.read(&mut chunk).await.unwrap();
            assert!(read > 0, "the connection closed mid-request");
            bytes.extend_from_slice(&chunk[..read]);
            let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&bytes[..end]).into_owned();
            let length: usize = head
                .lines()
                .find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap()))
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return head.split_whitespace().nth(1).unwrap_or_default().to_string();
            }
        }
    }

    #[tokio::test]
    async fn a_refresh_during_a_request_is_saved_before_the_rate_limit_wait() {
        let (_dir, path) = auth_file(&jwt("a1", now() + 3_600), "rt_request_1");
        let rotated_access = jwt("a2", now() + 3_600);
        // The server rejects the first request's token as expired; the
        // provider refreshes and resends it, and that is rate limited. The
        // retry after the wait fails for good.
        let server = {
            let (access, responses) = (rotated_access.clone(), counter());
            Server::start(move |path| match path {
                "/token" => rotated(&access, "rt_request_2"),
                _ => match responses.fetch_add(1, Ordering::SeqCst) {
                    0 => (401, r#"{"error":{"code":"token_expired"}}"#.into()),
                    1 => (429, r#"{"detail":"Rate limit exceeded"}"#.into()),
                    _ => (500, r#"{"error":{"code":"server_error"}}"#.into()),
                },
            })
            .await
        };
        let tokens = TokenKeeper::load(&path).unwrap();
        let client = server.client(&tokens);
        let mut conversation = client.new_conversation(Model::get(Provider::OpenAI, "gpt-6-astra").unwrap());
        conversation.add_user_message("Start the audit.");
        let config = GenerationConfig::default();
        let mut waits = Vec::new();
        let result = retry_rate_limits(
            0,
            &client,
            &tokens,
            || generate_once(&client, &conversation, &config),
            |_delay| {
                waits.push(on_disk(&path));
                async {}
            },
        )
        .await;
        assert!(matches!(result, Err(Error::ProviderError { status: Some(500), .. })), "{:?}", result.err());
        assert_eq!(waits, [(rotated_access, "rt_request_2".to_string())], "the rotated tokens were on disk during the wait");
        assert_eq!((server.requests("/token"), server.requests("/responses")), (1, 3));
        assert_eq!(tokens.refreshes().len(), 1);
    }

    #[tokio::test]
    async fn newer_tokens_written_by_codex_are_adopted_instead_of_refreshing() {
        let (_dir, path) = auth_file(&jwt("a1", now() - 60), "rt_adopt_1");
        let server = Server::start(|_| rotated(&jwt("a2", now() + 3_600), "rt_adopt_2")).await;
        let tokens = TokenKeeper::load(&path).unwrap();
        let client = server.client(&tokens);
        // Codex refreshed while the run was going.
        let newer = jwt("a3", now() + 3_600);
        let codex = auth_json(&newer, "rt_adopt_3");
        fs::write(&path, &codex).unwrap();

        tokens.ensure_fresh(&client).await.unwrap();
        assert_eq!(server.requests("/token"), 0);
        let live = client.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!((live.access_token, live.refresh_token), (newer, Some("rt_adopt_3".into())));
        assert_eq!(fs::read_to_string(&path).unwrap(), codex);
        assert_eq!(tokens.refreshes().len(), 1);
    }

    #[tokio::test]
    async fn refreshed_tokens_that_cannot_be_saved_go_to_a_private_recovery_file_and_stop_the_run() {
        let (dir, path) = auth_file(&jwt("a1", now() - 60), "rt_unsaved_1");
        // Codex also refreshes, and writes its tokens while this process
        // refreshes: ours must not overwrite them, so they can't be saved.
        let codex = auth_json(&jwt("a3", now() + 3_600), "rt_unsaved_3");
        let ours = jwt("a2", now() + 3_600);
        let server = {
            let (codex_path, codex_text, ours) = (path.clone(), codex.clone(), ours.clone());
            Server::start(move |_| {
                fs::write(&codex_path, &codex_text).unwrap();
                rotated(&ours, "rt_unsaved_2")
            })
            .await
        };
        let tokens = TokenKeeper::load(&path).unwrap();
        let client = server.client(&tokens);

        // Two conversations check at once: the second waits for the first
        // one's refresh, and stops too instead of sending another request.
        let (first, second) = tokio::join!(tokens.ensure_fresh(&client), tokens.ensure_fresh(&client));
        let error = first.unwrap_err();
        assert_eq!(second, Err(error.clone()));
        assert_eq!(server.requests("/token"), 1);
        assert_eq!(fs::read_to_string(&path).unwrap(), codex, "Codex's tokens are kept");
        let recovered: Vec<String> =
            file_names(dir.path()).into_iter().filter(|name| name.starts_with("auth.json.recovered-")).collect();
        assert_eq!(recovered.len(), 1, "{:?}", file_names(dir.path()));
        let recovery = dir.path().join(&recovered[0]);
        assert_eq!(on_disk(&recovery), (ours.clone(), "rt_unsaved_2".to_string()));
        assert!(error.contains(&recovery.display().to_string()), "{error}");
        assert!(error.ends_with("; stopping"), "{error}");
        assert!(!error.contains("rt_unsaved_2") && !error.contains(&ours), "no tokens in messages: {error}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&recovery).unwrap().permissions().mode() & 0o777, 0o600);
        }

        // The run is over: later checks fail the same way and write nothing.
        assert_eq!(tokens.check(), Err(error.clone()));
        assert_eq!(tokens.ensure_fresh(&client).await, Err(error.clone()));
        // A run stopped this way ends like any other authentication failure.
        let stopped = retry_rate_limits(0, &client, &tokens, || async { Ok(()) }, |_| async {}).await.unwrap_err();
        assert_eq!(classify(&stopped), Failure::StopAll);
        // Its outcome line names the recovery file in full.
        let outcome = error_text(&stopped, 300);
        assert_eq!(outcome, format!("Authentication failed: {error}"));
        // A request in flight at the stop that refreshes after a 401 sends
        // no refresh token again: the store hands it Codex's tokens.
        let _ = client.refresh_openai_oauth_token_once().await;
        assert_eq!(server.requests("/token"), 1);
        assert_eq!(file_names(dir.path()).len(), 2, "{:?}", file_names(dir.path()));
        assert_eq!(fs::read_to_string(&path).unwrap(), codex);
    }

    #[tokio::test]
    async fn no_refresh_without_rereading_the_file_first() {
        let (dir, path) = auth_file(&jwt("a1", now() - 60), "rt_reread_1");
        let server = Server::start(|_| rotated(&jwt("a2", now() + 3_600), "rt_reread_2")).await;
        let tokens = TokenKeeper::load(&path).unwrap();
        let client = server.client(&tokens);
        fs::remove_file(&path).unwrap();

        // Nothing rotated, so nothing is at risk: requests go on.
        tokens.check().unwrap();
        // But refreshing now could rotate away tokens Codex wrote.
        let error = tokens.ensure_fresh(&client).await.unwrap_err();
        assert!(error.starts_with("token refresh failed"), "{error}");
        assert!(tokens.refreshes()[0].starts_with("cannot use"), "{:?}", tokens.refreshes());
        assert_eq!(server.requests("/token"), 0);
        assert!(file_names(dir.path()).is_empty());
    }

    #[tokio::test]
    async fn repeated_refreshes_keep_one_backup() {
        let (dir, path) = auth_file(&jwt("a1", now() - 60), "rt_backup_1");
        // Each refresh returns an access token expiring within the margin,
        // so the next check refreshes again.
        let server = {
            let issued = counter();
            Server::start(move |_| {
                let n = issued.fetch_add(1, Ordering::SeqCst) + 2;
                rotated(&jwt(&format!("a{n}"), now() + 60), &format!("rt_backup_{n}"))
            })
            .await
        };
        let tokens = TokenKeeper::load(&path).unwrap();
        let client = server.client(&tokens);
        for n in 2..=3 {
            tokens.ensure_fresh(&client).await.unwrap();
            assert_eq!(on_disk(&path).1, format!("rt_backup_{n}"));
        }
        assert_eq!(file_names(dir.path()), ["auth.json", "auth.json.bak"]);
        assert_eq!(on_disk(&dir.path().join("auth.json.bak")).1, "rt_backup_2");
        assert_eq!(tokens.refreshes().len(), 2);
    }

    #[test]
    fn workload_is_deterministic_and_sized() {
        assert_eq!(tool_output(1, 4, "read_file"), tool_output(1, 4, "read_file"));
        assert_ne!(tool_output(0, 4, "read_file"), tool_output(1, 4, "read_file"));
        let chars = tool_output(0, 3, "read_file").len();
        assert!((7_000..14_000).contains(&chars), "{chars}");
        let instructions = instructions().len();
        assert!((10_000..18_000).contains(&instructions), "{instructions}");
    }
}
