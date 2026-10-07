//! The dry run's model: a local server speaking the OpenAI Responses
//! streaming protocol that plays each role with a fixed script. Its tool
//! calls run for real (the shell, `submit_question`), so a dry run checks
//! the whole pipeline (checkouts, indexes, the arms' tools, grading,
//! resume, the report) and measures prompt and tool-output sizes, without a
//! model call.
//!
//! Token counts are estimates: input is the request's text at 4 bytes per
//! token, with the previous request of the same conversation counted as
//! cached; output is the reply's text at 4 bytes per token plus a fixed
//! number of reasoning tokens per call.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::BenchResult;

#[derive(Debug, Clone, Copy)]
pub struct MockSettings {
    /// Reasoning tokens reported for every call.
    pub reasoning_tokens: usize,
    /// Shell commands a scripted solver (and verifier) runs before it
    /// answers: searches and reads in turn.
    pub solver_steps: usize,
    /// How long each reply takes, to rehearse stops and resumes.
    pub delay: std::time::Duration,
}

pub struct MockServer {
    url: String,
    task: tokio::task::JoinHandle<()>,
}

struct MockState {
    settings: MockSettings,
    /// Input tokens of each conversation's previous request.
    previous_input: parking_lot::Mutex<HashMap<String, usize>>,
    counter: std::sync::atomic::AtomicUsize,
}

impl MockServer {
    pub async fn start(settings: MockSettings) -> BenchResult<Self> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let url = format!("http://{}", listener.local_addr()?);
        let state = Arc::new(MockState {
            settings,
            previous_input: parking_lot::Mutex::new(HashMap::new()),
            counter: std::sync::atomic::AtomicUsize::new(0),
        });
        let task = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let state = state.clone();
                tokio::spawn(async move {
                    let Some(body) = read_request(&mut socket).await else {
                        return;
                    };
                    let reply = match serde_json::from_str::<Value>(&body) {
                        Ok(request) => respond(&state, &request),
                        Err(error) => error_reply(&format!("bad request body: {error}")),
                    };
                    if !state.settings.delay.is_zero() {
                        tokio::time::sleep(state.settings.delay).await;
                    }
                    let _ = socket.write_all(reply.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        eprintln!("mock model listening at {url} (dry run: no model calls)");
        Ok(Self { url, task })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn shutdown(self) {
        self.task.abort();
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<String> {
    let mut bytes = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let n = socket.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..n]);
        let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
        let length: usize = head
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0);
        if bytes.len() >= end + 4 + length {
            return Some(String::from_utf8_lossy(&bytes[end + 4..end + 4 + length]).into_owned());
        }
    }
}

fn error_reply(message: &str) -> String {
    let body = json!({"error": {"message": message, "type": "invalid_request_error"}}).to_string();
    format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// The parts of a request the scripts read.
#[derive(Debug, Default)]
struct View {
    instructions: String,
    user_texts: Vec<String>,
    calls: Vec<(String, Value)>,
    outputs: Vec<String>,
    cache_key: String,
    text_bytes: usize,
}

fn view_of(request: &Value) -> View {
    let mut view = View {
        instructions: request["instructions"].as_str().unwrap_or("").to_string(),
        cache_key: request["prompt_cache_key"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        ..View::default()
    };
    view.text_bytes = view.instructions.len() + request["tools"].to_string().len();
    for item in request["input"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("function_call") => {
                let arguments = item["arguments"].as_str().unwrap_or("{}");
                view.text_bytes += arguments.len();
                view.calls.push((
                    item["name"].as_str().unwrap_or("").to_string(),
                    serde_json::from_str(arguments).unwrap_or(Value::Null),
                ));
            }
            Some("function_call_output") => {
                let output = match &item["output"] {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                view.text_bytes += output.len();
                view.outputs.push(output);
            }
            _ => {
                let text = message_text(item);
                view.text_bytes += text.len();
                if item["role"] == "user" {
                    view.user_texts.push(text);
                }
            }
        }
    }
    view
}

fn message_text(item: &Value) -> String {
    match &item["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn hash_of(parts: &[&str]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut hasher);
    hasher.finish()
}

/// A shell argument in single quotes.
fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// A reply: a tool call, or text.
enum Action {
    Call(&'static str, Value),
}

fn respond(state: &MockState, request: &Value) -> String {
    let view = view_of(request);
    let steps = state.settings.solver_steps;
    let action = if view.instructions.contains("[index-benchmark role: solver]") {
        solver(&view, steps)
    } else if view.instructions.contains("[index-benchmark role: grader]") {
        grader(&view)
    } else if view.instructions.contains("[index-benchmark role: writer]") {
        writer(&view)
    } else if view
        .instructions
        .contains("[index-benchmark role: verifier]")
    {
        verifier(&view, steps)
    } else {
        Action::Call("complete_task", json!({"summary": "mock: unknown role"}))
    };
    // The text, plus the reasoning of the earlier calls, which a stateless
    // request carries back (encrypted) in its input.
    let input_tokens =
        view.text_bytes.div_ceil(4) + state.settings.reasoning_tokens * view.calls.len();
    let cached = {
        let mut previous = state.previous_input.lock();
        let cached = previous
            .get(&view.cache_key)
            .map_or(0, |tokens| tokens.min(&input_tokens) / 128 * 128);
        previous.insert(view.cache_key.clone(), input_tokens);
        cached
    };
    let n = state
        .counter
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let Action::Call(name, arguments) = action;
    let arguments = arguments.to_string();
    let output_tokens = state.settings.reasoning_tokens + arguments.len().div_ceil(4);
    let item = json!({
        "type": "function_call", "id": format!("fc_mock{n}"), "call_id": format!("call_mock{n}"),
        "name": name, "arguments": arguments, "status": "completed",
    });
    let completed = json!({"type": "response.completed", "response": {
        "id": format!("resp_mock{n}"), "object": "response", "created_at": 0, "status": "completed",
        "model": request["model"].as_str().unwrap_or("mock"), "output": [],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": state.settings.reasoning_tokens},
        },
        "metadata": {},
    }});
    let stream: String = [
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        completed,
    ]
    .iter()
    .map(|event| format!("data: {event}\n\n"))
    .collect();
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{stream}",
        stream.len()
    )
}

fn shell(command: String) -> Action {
    Action::Call(crate::shell::SHELL_TOOL, json!({ "command": command }))
}

fn shell_calls(view: &View) -> usize {
    view.calls
        .iter()
        .filter(|(name, _)| name == crate::shell::SHELL_TOOL)
        .count()
}

/// The question in a solver's or verifier's first message.
fn question_of(view: &View) -> String {
    let first = view.user_texts.first().map(String::as_str).unwrap_or("");
    first
        .lines()
        .find_map(|line| {
            line.strip_prefix("Question (")
                .and_then(|rest| rest.split_once("): "))
                .map(|(_, question)| question.to_string())
        })
        .unwrap_or_default()
}

/// The words of `text` most worth searching for: identifiers in backticks
/// first, then the longest other words.
fn keywords(text: &str) -> Vec<String> {
    let mut words: Vec<String> = text
        .split('`')
        .skip(1)
        .step_by(2)
        .map(|word| word.trim().to_string())
        .filter(|word| !word.is_empty() && !word.contains(' '))
        .collect();
    let mut others: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| word.len() > 3)
        .map(str::to_string)
        .collect();
    others.sort_by_key(|word| std::cmp::Reverse(word.len()));
    for word in others {
        if !words.contains(&word) {
            words.push(word);
        }
    }
    words.truncate(6);
    words
}

/// The `round`th search a scripted agent runs, in its arm: arms B and C
/// search the question, then its keywords one by one (arm C alternates
/// dense and lexical search); arm A greps the keywords one by one.
fn search_command(instructions: &str, question: &str, round: usize) -> String {
    let words = keywords(question);
    let query = if round == 0 || words.is_empty() {
        question
            .split_whitespace()
            .take(12)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        words[(round - 1) % words.len()].clone()
    };
    if instructions.contains("arko_index_dense") && round.is_multiple_of(2) {
        format!("arko_index_dense {}", quote(&query))
    } else if instructions.contains("arko_index search2") {
        format!("arko_index search2 \"$ARKO_INDEX_STATE\" {}", quote(&query))
    } else if words.is_empty() {
        "ls".to_string()
    } else {
        format!(
            "rg -n -i --max-count 3 -e {} . | head -60",
            quote(&words[round % words.len()])
        )
    }
}

/// The first `path:line` a search printed: an `rg` line or a search2 JSON
/// line.
fn first_hit(output: &str) -> Option<(String, u32)> {
    for line in output.lines() {
        if let Ok(hit) = serde_json::from_str::<Value>(line)
            && let (Some(path), Some(line)) = (hit["path"].as_str(), hit["start_line"].as_u64())
        {
            return Some((path.to_string(), line as u32));
        }
        let mut parts = line.splitn(3, ':');
        if let (Some(path), Some(number)) = (parts.next(), parts.next())
            && let Ok(number) = number.parse::<u32>()
            && path.contains('.')
        {
            return Some((path.trim_start_matches("./").to_string(), number));
        }
    }
    None
}

/// A solver that searches and reads in turn, `steps` shell commands in all,
/// then answers with the first hit it found.
fn solver(view: &View, steps: usize) -> Action {
    let question = question_of(view);
    let done = shell_calls(view);
    if done < steps.max(1) {
        if done.is_multiple_of(2) {
            return shell(search_command(&view.instructions, &question, done / 2));
        }
        return match view.outputs.last().and_then(|output| first_hit(output)) {
            Some((path, line)) => shell(format!(
                "sed -n '{},{}p' {}",
                line.saturating_sub(20).max(1),
                line + 40,
                quote(&path)
            )),
            None => shell("ls".to_string()),
        };
    }
    let hit = view
        .outputs
        .iter()
        .find_map(|output| first_hit(output))
        .map(|(path, line)| format!("{path}:{line}"))
        .unwrap_or_default();
    Action::Call(
        "complete_task",
        json!({ "summary": format!("ANSWER:\n(mock) The answer is the code at {hit}.\nLOCATIONS:\n{hit}\n") }),
    )
}

fn grader(view: &View) -> Action {
    let prompt = view.user_texts.first().map(String::as_str).unwrap_or("");
    let ids: Vec<String> = prompt
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix('[')?;
            let (id, _) = rest.split_once("] ")?;
            (!id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric()))
                .then(|| id.to_string())
        })
        .collect();
    let items: Vec<Value> = ids
        .iter()
        .map(|id| {
            let met = hash_of(&[&view.cache_key, id]) % 10 < 6;
            json!({"id": id, "met": met, "note": "mock"})
        })
        .collect();
    let met = items.iter().filter(|item| item["met"] == true).count();
    let verdict = if met == items.len() {
        "correct"
    } else if met > 0 {
        "partially_correct"
    } else {
        "incorrect"
    };
    Action::Call(
        "complete_task",
        json!({ "summary": json!({"items": items, "wrong_claims": [], "verdict": verdict, "notes": "mock grade"}).to_string() }),
    )
}

fn extensions(language: &str) -> &'static [&'static str] {
    match language {
        "rust" => &["rs"],
        "c" => &["c"],
        "cpp" => &["cc", "cpp"],
        "go" => &["go"],
        "java" => &["java"],
        "python" => &["py"],
        "typescript" => &["ts"],
        "javascript" => &["js"],
        _ => &["rs", "c", "go", "py", "java", "ts", "cc"],
    }
}

fn writer(view: &View) -> Action {
    let prompt = view.user_texts.first().map(String::as_str).unwrap_or("");
    let language = prompt
        .split("mainly ")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .unwrap_or("");
    let kind = prompt
        .split("kind `")
        .nth(1)
        .and_then(|rest| rest.split('`').next())
        .unwrap_or("definition")
        .to_string();
    let count: usize = prompt
        .split("Write ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|number| number.parse().ok())
        .unwrap_or(1);
    let shells = shell_calls(view);
    if shells == 0 {
        let names: Vec<String> = extensions(language)
            .iter()
            .map(|extension| format!("-name '*.{extension}'"))
            .collect();
        return shell(format!(
            "find . -type f \\( {} \\) -not -path './.git/*' -not -path '*/test*' | sort | head -60",
            names.join(" -o ")
        ));
    }
    let files: Vec<String> = view
        .outputs
        .first()
        .map(|output| {
            output
                .lines()
                .filter(|line| line.starts_with("./"))
                .map(|line| line.trim_start_matches("./").to_string())
                .collect()
        })
        .unwrap_or_default();
    if files.is_empty() {
        return Action::Call("complete_task", json!({"summary": "mock: no files"}));
    }
    let file = &files[(hash_of(&[&view.cache_key]) as usize) % files.len()];
    if shells == 1 {
        return shell(format!("arko_index outline {}", quote(file)));
    }
    let submitted = view
        .calls
        .iter()
        .filter(|(name, _)| name == "submit_question")
        .count();
    let entries: Vec<(u32, u32, String)> = view
        .outputs
        .get(1)
        .map(|output| {
            output
                .lines()
                .filter_map(|line| {
                    let mut fields = line.split('\t');
                    let range = fields.next()?.rsplit(':').next()?.to_string();
                    let kind = fields.next()?;
                    let symbol = fields.next()?.trim();
                    let (start, end) = range.split_once('-')?;
                    let wanted = !symbol.is_empty()
                        && [
                            "function",
                            "method",
                            "class",
                            "struct",
                            "type",
                            "enum",
                            "interface",
                        ]
                        .iter()
                        .any(|wanted| kind.contains(wanted));
                    if !wanted {
                        return None;
                    }
                    Some((start.parse().ok()?, end.parse().ok()?, symbol.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    if submitted >= count || entries.is_empty() {
        return Action::Call(
            "complete_task",
            json!({"summary": format!("mock: submitted {submitted} questions")}),
        );
    }
    let (start, end, symbol) = &entries[submitted % entries.len()];
    let location = format!("{file}:{start}-{end} | the definition of {symbol}");
    let (question, facts) = match kind.as_str() {
        "callers" => (
            format!("(mock) Which functions call `{symbol}`?"),
            String::new(),
        ),
        "behaviour" => (
            format!("(mock) How does `{symbol}` work?"),
            format!(
                "{symbol} is defined in {file}\n{symbol} starts at line {start}\n{symbol} ends at line {end}"
            ),
        ),
        "impact" => (
            format!("(mock) What would a change to `{symbol}` affect?"),
            format!("{symbol} is defined in {file}"),
        ),
        _ => (
            format!("(mock) Where is `{symbol}` defined?"),
            String::new(),
        ),
    };
    Action::Call(
        "submit_question",
        json!({
            "kind": kind,
            "question": question,
            "answer": format!("(mock) `{symbol}` is defined at {file}:{start}."),
            "locations": location,
            "facts": facts,
            "difficulty": "easy",
            "evidence": "mock: arko_index outline",
        }),
    )
}

fn verifier(view: &View, steps: usize) -> Action {
    let in_key_phase = view
        .user_texts
        .iter()
        .any(|text| text.contains("ANSWER KEY to check"));
    if in_key_phase {
        let question = question_of(view);
        let verdict = if hash_of(&[&question]).is_multiple_of(5) {
            "disagree"
        } else {
            "agree"
        };
        return Action::Call(
            "complete_task",
            json!({ "summary": json!({"verdict": verdict, "problems": [], "notes": "mock verdict"}).to_string() }),
        );
    }
    solver(view, steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_hits_in_rg_and_search2_output() {
        assert_eq!(
            first_hit("exit code: 0\n./src/a.rs:12:fn a() {}\n"),
            Some(("src/a.rs".to_string(), 12))
        );
        assert_eq!(
            first_hit("exit code: 0\n{\"path\":\"src/b.go\",\"start_line\":40,\"end_line\":60}\n"),
            Some(("src/b.go".to_string(), 40))
        );
        assert_eq!(first_hit("exit code: 1\n"), None);
    }

    #[test]
    fn scripts_follow_their_arm() {
        let question = "Where is the function that retries `fetch_page` defined?";
        let index = "... arko_index search2 \"$ARKO_INDEX_STATE\" ...";
        assert!(search_command(index, question, 0).starts_with("arko_index search2"));
        assert!(search_command(index, question, 1).ends_with("'fetch_page'"));
        let dense = "... arko_index search2 ... arko_index_dense ...";
        assert!(search_command(dense, question, 0).starts_with("arko_index_dense"));
        assert!(search_command(dense, question, 1).starts_with("arko_index search2"));
        let terminal = search_command("terminal only", question, 0);
        assert!(terminal.starts_with("rg -n"), "{terminal}");
        assert!(terminal.contains("'fetch_page'"));
        assert!(!search_command("terminal only", question, 1).contains("'fetch_page'"));
        assert_eq!(keywords(question)[0], "fetch_page");
    }

    #[test]
    fn quotes_shell_arguments() {
        assert_eq!(quote("it's"), "'it'\\''s'");
    }
}
