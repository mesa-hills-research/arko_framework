//! Failover between ChatGPT accounts against a mock ChatGPT backend that
//! answers each account (by `chatgpt-account-id`) from its own script.

use super::*;
use crate::{client::LLMClient, model::Model, tool::ToolResult};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::json;
use std::{collections::VecDeque, time::UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// One scripted answer: to the account `account`, at `path`.
struct Answer {
    account: &'static str,
    path: &'static str,
    status: u16,
    body: String,
    headers: Vec<(&'static str, String)>,
}

fn answer(account: &'static str, path: &'static str, status: u16, body: &str) -> Answer {
    Answer {
        account,
        path,
        status,
        body: body.into(),
        headers: Vec::new(),
    }
}

impl Answer {
    fn header(mut self, name: &'static str, value: &str) -> Self {
        self.headers.push((name, value.into()));
        self
    }
}

/// A request the mock received.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    account: String,
    head: String,
    body: Value,
}

impl Seen {
    fn header(&self, name: &str) -> Option<String> {
        self.head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
    }
}

struct Mock {
    url: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    script: Arc<Mutex<VecDeque<Answer>>>,
}

impl Mock {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let script: Arc<Mutex<VecDeque<Answer>>> = Arc::new(Mutex::new(VecDeque::new()));
        let (recorded, answers) = (seen.clone(), script.clone());
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let (recorded, answers) = (recorded.clone(), answers.clone());
                tokio::spawn(async move {
                    let request = read(&mut socket).await;
                    let scripted = {
                        let mut answers = answers.lock().unwrap();
                        let position = answers.iter().position(|scripted| {
                            scripted.account == request.account
                                && request.path.ends_with(scripted.path)
                        });
                        position.and_then(|position| answers.remove(position))
                    };
                    recorded.lock().unwrap().push(request);
                    let reply = scripted.unwrap_or_else(|| {
                        answer("", "", 500, r#"{"error":{"message":"unscripted request"}}"#)
                    });
                    write(&mut socket, &reply).await;
                });
            }
        });
        Self { url, seen, script }
    }

    fn expect(&self, answers: impl IntoIterator<Item = Answer>) {
        self.script.lock().unwrap().extend(answers);
    }

    /// The requests received since the last call.
    fn take(&self) -> Vec<Seen> {
        std::mem::take(&mut *self.seen.lock().unwrap())
    }

    fn unanswered(&self) -> usize {
        self.script.lock().unwrap().len()
    }
}

async fn read(socket: &mut tokio::net::TcpStream) -> Seen {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 8192];
        let n = socket.read(&mut chunk).await.unwrap();
        assert!(n > 0, "connection closed mid-request");
        bytes.extend_from_slice(&chunk[..n]);
        let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
        let length: usize = head
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|length| length.trim().parse().unwrap())
            })
            .unwrap_or(0);
        if bytes.len() < end + 4 + length {
            continue;
        }
        let body = if length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap()
        };
        let mut request_line = head.lines().next().unwrap().split(' ');
        let method = request_line.next().unwrap().to_string();
        let path = request_line.next().unwrap().to_string();
        let mut seen = Seen {
            method,
            path,
            account: String::new(),
            head,
            body,
        };
        seen.account = seen.header("chatgpt-account-id").unwrap_or_default();
        return seen;
    }
}

async fn write(socket: &mut tokio::net::TcpStream, answer: &Answer) {
    let content_type = if answer.body.starts_with("data:") {
        "text/event-stream"
    } else {
        "application/json"
    };
    let extra: String = answer
        .headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    let response = format!(
        "HTTP/1.1 {} Test\r\nContent-Type: {content_type}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        answer.status,
        answer.body.len(),
        answer.body,
    );
    socket.write_all(response.as_bytes()).await.unwrap();
}

/// Expires in 2100.
fn fresh_jwt() -> String {
    let claims = URL_SAFE_NO_PAD.encode(r#"{"exp":4102444800}"#);
    format!("header.{claims}.signature")
}

const PRIMARY: &str = "acct-primary";
const SECONDARY: &str = "acct-secondary";

fn account(label: &str, account_id: &str, url: &str) -> ChatGptAccount {
    let mut client =
        OpenAIClient::new_chatgpt_oauth(&fresh_jwt(), Some("synthetic-refresh"), account_id)
            .with_base_url(url)
            .with_oauth_token_url(format!("{url}/token"));
    client.http_client = crate::providers::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    ChatGptAccount::new(label, client)
}

type Events = Arc<Mutex<Vec<ChatGptAccountEvent>>>;

/// `primary` and `secondary` on `mock`, checked on every request (no
/// interval), with their events collected.
fn pool(mock: &Mock) -> (ChatGptAccounts, Events) {
    let events: Events = Arc::default();
    let recorded = events.clone();
    let accounts = ChatGptAccounts::new([
        account("primary", PRIMARY, &mock.url),
        account("secondary", SECONDARY, &mock.url),
    ])
    .unwrap()
    .with_status_check_interval(Duration::ZERO)
    .with_usage_url(format!("{}/wham/usage", mock.url))
    .with_observer(move |event| recorded.lock().unwrap().push(event.clone()));
    (accounts, events)
}

fn take(events: &Events) -> Vec<ChatGptAccountEvent> {
    std::mem::take(&mut *events.lock().unwrap())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn at(seconds: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(seconds)
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

/// A turn answering `text` after reasoning whose encrypted content is
/// `encrypted`.
fn turn(encrypted: &str, text: &str) -> String {
    sse(&[
        json!({"type": "response.created", "response": {"model": "gpt-6-astra"}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item":
            {"type": "reasoning", "id": format!("rs_{encrypted}"), "summary": [], "encrypted_content": encrypted}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": text}),
        json!({"type": "response.output_item.done", "output_index": 1, "item":
            {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
             "content": [{"type": "output_text", "text": text, "annotations": []}]}}),
        json!({"type": "response.completed", "response": {
            "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
            "model": "gpt-6-astra", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}),
    ])
}

fn usage_limit(resets_at: u64) -> String {
    json!({"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
        "plan_type": "pro", "resets_at": resets_at}})
    .to_string()
}

fn usage_status(allowed: bool, has_credits: bool) -> String {
    json!({
        "plan_type": "pro",
        "rate_limit": {"allowed": allowed, "limit_reached": !allowed,
            "primary_window": {"used_percent": if allowed { 20 } else { 100 },
                "limit_window_seconds": 18000, "reset_after_seconds": 3600, "reset_at": unix_now() + 3600},
            "secondary_window": null},
        "credits": {"has_credits": has_credits, "unlimited": false, "balance": "0"},
        "spend_control": {"reached": false},
        "rate_limit_reached_type": null,
        "rate_limit_reset_credits": {"available_count": 2},
    })
    .to_string()
}

fn conversation() -> Conversation {
    let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
    conversation.add_user_message("Audit src/lib.rs.");
    conversation
}

fn responses(seen: &[Seen]) -> Vec<&Seen> {
    seen.iter().filter(|seen| seen.method == "POST").collect()
}

fn text(response: &GenerationResponse) -> String {
    response
        .message
        .content
        .iter()
        .filter_map(crate::message::Content::as_text)
        .collect()
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("the test finished in time")
}

#[tokio::test]
async fn a_usage_limit_moves_the_request_to_the_next_account_at_once() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        let client = LLMClient::new().with_chatgpt_accounts(accounts.clone());
        let resets_at = unix_now() + 3600;
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(resets_at)),
            answer(
                SECONDARY,
                "/responses",
                200,
                &turn("enc_b", "from secondary"),
            ),
            answer(SECONDARY, "/responses", 200, &turn("enc_b2", "again")),
        ]);

        let config = GenerationConfig::default();
        let first = client
            .generate_text(&conversation(), config.clone())
            .await
            .unwrap();
        assert_eq!(text(&first), "from secondary");
        assert_eq!(first.model_call_usage.account.as_deref(), Some("secondary"));
        let seen = mock.take();
        let accounts_tried: Vec<&str> = seen.iter().map(|seen| seen.account.as_str()).collect();
        assert_eq!(accounts_tried, [PRIMARY, SECONDARY]);
        // The same request, unchanged, went to the secondary.
        assert_eq!(seen[0].body, seen[1].body);

        let events = take(&events);
        assert_eq!(
            events,
            [
                ChatGptAccountEvent::Exhausted {
                    account: "primary".into(),
                    resets_at: events.first().and_then(|event| match event {
                        ChatGptAccountEvent::Exhausted { resets_at, .. } => *resets_at,
                        _ => None,
                    }),
                    code: Some("usage_limit_reached".into()),
                },
                ChatGptAccountEvent::Switched {
                    from: "primary".into(),
                    to: "secondary".into(),
                    reason: SwitchReason::Exhausted,
                },
            ]
        );
        let ChatGptAccountEvent::Exhausted {
            resets_at: Some(reported),
            ..
        } = &events[0]
        else {
            panic!("no reset time")
        };
        let reported = reported.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert!(
            reported.abs_diff(resets_at) <= 2,
            "{reported} vs {resets_at}"
        );
        assert!(events[1].to_string().contains("`secondary`"));

        let statuses = accounts.statuses();
        assert!(statuses[0].exhausted && !statuses[0].active);
        assert_eq!(
            statuses[0].error_code.as_deref(),
            Some("usage_limit_reached")
        );
        assert!(!statuses[1].exhausted && statuses[1].active);
        assert_eq!(accounts.active_account().as_deref(), Some("secondary"));

        // Streaming, the next request goes straight to the secondary after
        // a check finds the primary still used up.
        mock.expect([answer(
            PRIMARY,
            "/wham/usage",
            200,
            &usage_status(false, false),
        )]);
        let mut stream = client
            .generate_stream(&conversation(), config.clone())
            .await
            .unwrap();
        let mut end = None;
        while let Some(event) = stream.next().await {
            if let StreamEvent::End(response) = event.unwrap() {
                end = Some(response);
            }
        }
        assert_eq!(
            end.unwrap().model_call_usage.account.as_deref(),
            Some("secondary")
        );
        let seen = mock.take();
        assert_eq!(seen.len(), 2);
        assert_eq!(
            (seen[0].method.as_str(), seen[0].account.as_str()),
            ("GET", PRIMARY)
        );
        assert_eq!(responses(&seen)[0].account, SECONDARY);
        assert_eq!(mock.unanswered(), 0);
    })
    .await;
}

#[tokio::test]
async fn every_account_exhausted_fails_with_the_earliest_reset() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        let accounts = accounts.with_status_check_interval(Duration::from_secs(3600));
        let now = unix_now();
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(now + 7200)),
            answer(SECONDARY, "/responses", 429, &usage_limit(now + 3600)),
        ]);
        let error = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(error.is_usage_limit(), "{error}");
        let wait = error.retry_after().unwrap().as_secs();
        assert!((3590..=3600).contains(&wait), "{wait}");
        assert_eq!(responses(&mock.take()).len(), 2);
        let reported = take(&events);
        assert!(matches!(
            reported.last(),
            Some(ChatGptAccountEvent::AllExhausted { resets_at: Some(_) })
        ));
        assert_eq!(
            reported
                .iter()
                .filter(|event| matches!(event, ChatGptAccountEvent::Exhausted { .. }))
                .count(),
            2
        );

        // Until a reset or a check, requests fail at once, without being sent
        // and without another notification.
        let again = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(again.is_usage_limit());
        assert!(again.retry_after().unwrap().as_secs() <= 3600);
        assert!(mock.take().is_empty());
        assert!(take(&events).is_empty());
    })
    .await;
}

#[tokio::test]
async fn requests_move_back_once_the_usage_endpoint_allows_the_primary() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "secondary")),
        ]);
        accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        mock.take();
        take(&events);

        // Still used up: the secondary goes on serving.
        mock.expect([
            answer(PRIMARY, "/wham/usage", 200, &usage_status(false, false)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b2", "secondary")),
        ]);
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("secondary")
        );
        let seen = mock.take();
        let check = &seen[0];
        assert_eq!(
            (check.method.as_str(), check.account.as_str()),
            ("GET", PRIMARY)
        );
        assert!(
            check
                .header("authorization")
                .unwrap()
                .starts_with("Bearer ")
        );
        assert!(take(&events).is_empty());
        let status = accounts.statuses()[0].usage.clone().unwrap();
        assert_eq!(status.allowed, Some(false));
        assert_eq!(status.reset_credits_available, Some(2));

        // Reading an account's usage changes nothing else.
        mock.expect([answer(
            SECONDARY,
            "/wham/usage",
            200,
            &usage_status(true, true),
        )]);
        let read = accounts.read_usage("secondary").await.unwrap();
        assert_eq!(read.usable(), Some(true));
        assert!(accounts.read_usage("nobody").await.is_err());
        let statuses = accounts.statuses();
        assert_eq!(statuses[1].usage, Some(read));
        assert!(statuses[0].exhausted && !statuses[1].exhausted);
        assert_eq!(mock.take()[0].account, SECONDARY);
        assert!(take(&events).is_empty());

        // The operator pressed a usage reset: the next check allows it.
        mock.expect([
            answer(PRIMARY, "/wham/usage", 200, &usage_status(true, false)),
            answer(PRIMARY, "/responses", 200, &turn("enc_a", "primary")),
        ]);
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("primary")
        );
        assert_eq!(
            take(&events),
            [
                ChatGptAccountEvent::Usable {
                    account: "primary".into(),
                    reason: UsableReason::StatusCheck,
                },
                ChatGptAccountEvent::Switched {
                    from: "secondary".into(),
                    to: "primary".into(),
                    reason: SwitchReason::Restored,
                },
            ]
        );
        assert_eq!(mock.unanswered(), 0);
    })
    .await;
}

#[tokio::test]
async fn credits_keep_an_account_usable() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "secondary")),
        ]);
        accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        take(&events);

        // Out of plan usage, but with credits: usable, as Codex decides.
        mock.expect([
            answer(PRIMARY, "/wham/usage", 200, &usage_status(false, true)),
            answer(PRIMARY, "/responses", 200, &turn("enc_a", "primary"))
                .header("x-codex-primary-used-percent", "100")
                .header("x-codex-primary-window-minutes", "300")
                .header("x-codex-credits-has-credits", "true")
                .header("x-codex-credits-unlimited", "false")
                .header("x-codex-credits-balance", "33950"),
        ]);
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("primary")
        );
        let events = take(&events);
        assert!(events.contains(&ChatGptAccountEvent::UsingCredits {
            account: "primary".into(),
            balance: Some("33950".into()),
        }));
        assert!(accounts.statuses()[0].using_credits());
    })
    .await;
}

#[tokio::test]
async fn the_reset_time_and_mark_usable_bring_an_account_back_without_a_check() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        let mut accounts = accounts.with_status_check_interval(Duration::from_secs(3600));
        let clock = Arc::new(Mutex::new(SystemTime::now()));
        accounts.clock = Some(clock.clone());
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "secondary")),
        ]);
        accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        take(&events);

        // Eleven minutes later the reset time has passed.
        *clock.lock().unwrap() += Duration::from_secs(660);
        mock.expect([answer(
            PRIMARY,
            "/responses",
            200,
            &turn("enc_a", "primary"),
        )]);
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("primary")
        );
        assert_eq!(
            take(&events)[0],
            ChatGptAccountEvent::Usable {
                account: "primary".into(),
                reason: UsableReason::ResetTimePassed,
            }
        );
        assert!(mock.take().iter().all(|seen| seen.method == "POST"));

        // Exhausted again; the operator says it was reset.
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b2", "secondary")),
        ]);
        accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        take(&events);
        accounts.mark_usable("primary").unwrap();
        assert!(accounts.mark_usable("nobody").is_err());
        assert_eq!(
            take(&events),
            [ChatGptAccountEvent::Usable {
                account: "primary".into(),
                reason: UsableReason::MarkedUsable,
            }]
        );
        mock.expect([answer(
            PRIMARY,
            "/responses",
            200,
            &turn("enc_a2", "primary"),
        )]);
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("primary")
        );
        assert_eq!(mock.unanswered(), 0);
    })
    .await;
}

#[tokio::test]
async fn check_now_reads_every_exhausted_account_and_a_failed_check_allows_a_trial() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        let accounts = accounts.with_status_check_interval(Duration::from_secs(3600));
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "secondary")),
        ]);
        accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        take(&events);
        mock.take();

        // The usage endpoint refuses the check: nothing is decided, and the
        // next request tries the account.
        mock.expect([answer(
            PRIMARY,
            "/wham/usage",
            403,
            r#"{"detail":"forbidden"}"#,
        )]);
        let statuses = accounts.check_now().await;
        assert!(statuses[0].exhausted && statuses[0].usage.is_none());
        assert!(take(&events).is_empty());
        assert_eq!(mock.take()[0].method, "GET");

        // The trial is refused again: set aside quietly, served by the
        // secondary, and not tried again before the next check.
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b2", "secondary")),
            answer(SECONDARY, "/responses", 200, &turn("enc_b3", "secondary")),
        ]);
        for _ in 0..2 {
            let response = accounts
                .generate_text(&conversation(), &GenerationConfig::default())
                .await
                .unwrap();
            assert_eq!(
                response.model_call_usage.account.as_deref(),
                Some("secondary")
            );
        }
        let tried: Vec<String> = mock.take().into_iter().map(|seen| seen.account).collect();
        assert_eq!(tried, [PRIMARY, SECONDARY, SECONDARY]);
        assert!(take(&events).is_empty());

        // A trial failing for another reason falls through as well.
        mock.expect([
            answer(PRIMARY, "/wham/usage", 503, "{}"),
            answer(
                PRIMARY,
                "/responses",
                500,
                r#"{"error":{"message":"oops"}}"#,
            ),
            answer(SECONDARY, "/responses", 200, &turn("enc_b4", "secondary")),
        ]);
        accounts.check_now().await;
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("secondary")
        );
        assert!(accounts.statuses()[0].exhausted);
        assert!(take(&events).is_empty());
        assert_eq!(mock.unanswered(), 0);

        // A check that allows it while the backend still refuses it (a
        // limit the check doesn't show): tried once, nobody notified.
        mock.expect([
            answer(PRIMARY, "/wham/usage", 200, &usage_status(true, false)),
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b5", "secondary")),
        ]);
        let statuses = accounts.check_now().await;
        assert!(statuses[0].exhausted);
        assert_eq!(statuses[0].retry, Some(UsableReason::StatusCheck));
        assert_eq!(statuses[0].usage.as_ref().unwrap().allowed, Some(true));
        assert!(statuses[0].checked_at.is_some());
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("secondary")
        );
        assert!(take(&events).is_empty());
        assert_eq!(accounts.statuses()[0].retry, None);

        // Allowed, and accepted: taken back.
        mock.expect([
            answer(PRIMARY, "/wham/usage", 200, &usage_status(true, false)),
            answer(PRIMARY, "/responses", 200, &turn("enc_a", "primary")),
        ]);
        accounts.check_now().await;
        let response = accounts
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(
            response.model_call_usage.account.as_deref(),
            Some("primary")
        );
        assert_eq!(
            take(&events),
            [
                ChatGptAccountEvent::Usable {
                    account: "primary".into(),
                    reason: UsableReason::StatusCheck,
                },
                ChatGptAccountEvent::Switched {
                    from: "secondary".into(),
                    to: "primary".into(),
                    reason: SwitchReason::Restored,
                },
            ]
        );
        assert!(!accounts.statuses()[0].exhausted);
        assert_eq!(mock.unanswered(), 0);
    })
    .await;
}

#[tokio::test]
async fn a_refusal_streamed_before_any_output_moves_the_request() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, events) = pool(&mock);
        let failed = sse(&[
            json!({"type": "response.created", "response": {"model": "gpt-6-astra"}}),
            json!({"type": "response.failed", "response": {"status": "failed",
                "error": {"code": "usage_limit_reached", "message": "The usage limit has been reached"}}}),
        ]);
        mock.expect([
            answer(PRIMARY, "/responses", 200, &failed),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "secondary")),
        ]);
        let mut stream = accounts
            .generate_stream(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        let mut end = None;
        while let Some(event) = stream.next().await {
            if let StreamEvent::End(response) = event.unwrap() {
                end = Some(response);
            }
        }
        assert_eq!(
            end.unwrap().model_call_usage.account.as_deref(),
            Some("secondary")
        );
        assert!(matches!(
            take(&events)[..],
            [
                ChatGptAccountEvent::Exhausted { .. },
                ChatGptAccountEvent::Switched { .. }
            ]
        ));
    })
    .await;
}

#[tokio::test]
async fn a_refusal_after_output_fails_the_stream_with_a_transient_error() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, _events) = pool(&mock);
        let failed = sse(&[
            json!({"type": "response.output_text.delta", "output_index": 0, "delta": "partial"}),
            json!({"type": "response.failed", "response": {"status": "failed",
                "error": {"code": "usage_limit_reached", "message": "The usage limit has been reached"}}}),
        ]);
        mock.expect([answer(PRIMARY, "/responses", 200, &failed)]);
        let mut stream = accounts
            .generate_stream(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::TextChunk(_)))
        ));
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, Error::StreamTruncated(_)), "{error}");
        assert!(error.is_retryable() && !error.is_usage_limit());
        assert!(stream.next().await.is_none());
        assert!(accounts.statuses()[0].exhausted);
    })
    .await;
}

/// The request's reasoning items' encrypted contents, and its turn-state
/// header.
fn replayed(seen: &Seen) -> (Vec<String>, Option<String>) {
    let reasoning = seen.body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .map(|item| item["encrypted_content"].as_str().unwrap().to_string())
        .collect();
    (reasoning, seen.header("x-codex-turn-state"))
}

#[tokio::test]
async fn each_account_is_sent_only_its_own_reasoning_and_turn_state() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, _events) = pool(&mock);
        let accounts = accounts.with_status_check_interval(Duration::from_secs(3600));
        let config = GenerationConfig::default();
        let mut conversation = conversation();
        // Reasoning from before this process counts as the primary's.
        conversation.add_assistant_message_with_reasoning(
            "Earlier.",
            ReasoningArtifact::OpenAIEncryptedReasoning {
                items: vec![crate::reasoning::OpenAIReasoningItem {
                    encrypted_content: "enc_old".into(),
                    summaries: None,
                    id: None,
                }],
            },
        );
        conversation.add_user_message("Go on.");

        // The primary serves a tool turn and issues a turn-state token.
        let tool_turn = sse(&[
            json!({"type": "response.output_item.done", "output_index": 0, "item":
                {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "enc_a"}}),
            json!({"type": "response.output_item.done", "output_index": 1, "item":
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file",
                 "arguments": "{}", "status": "completed"}}),
            json!({"type": "response.completed", "response": {
                "id": "resp_a", "object": "response", "created_at": 0, "status": "completed",
                "model": "gpt-6-astra", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        ]);
        mock.expect([answer(PRIMARY, "/responses", 200, &tool_turn)
            .header("x-codex-turn-state", "ts_primary")]);
        let response = accounts.generate_text(&conversation, &config).await.unwrap();
        let results = response
            .message
            .tool_calls
            .iter()
            .map(|call| ToolResult::from_tool_call(call, "contents"))
            .collect();
        conversation.add_assistant_message_from_response(response);
        conversation.add_tool_results(results);
        assert_eq!(conversation.current_openai_turn_state(), Some("ts_primary"));
        mock.take();

        // Within the same turn the primary is exhausted: the secondary gets
        // neither the primary's reasoning nor its turn-state token, but the
        // rest of the conversation unchanged.
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "done"))
                .header("x-codex-turn-state", "ts_secondary"),
        ]);
        let response = accounts.generate_text(&conversation, &config).await.unwrap();
        conversation.add_assistant_message_from_response(response);
        let seen = mock.take();
        assert_eq!(
            replayed(&seen[0]),
            (
                vec!["enc_old".to_string(), "enc_a".to_string()],
                Some("ts_primary".to_string())
            )
        );
        assert_eq!(replayed(&seen[1]), (vec![], None));
        let call_items = |seen: &Seen| {
            seen.body["input"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] != "reasoning")
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(call_items(&seen[0]), call_items(&seen[1]));
        // The conversation itself keeps everything.
        assert_eq!(conversation.current_openai_turn_state(), Some("ts_primary"));

        // The secondary goes on with its own reasoning only.
        conversation.add_user_message("And then?");
        mock.expect([answer(SECONDARY, "/responses", 200, &turn("enc_b2", "more"))]);
        let response = accounts.generate_text(&conversation, &config).await.unwrap();
        conversation.add_assistant_message_from_response(response);
        assert_eq!(replayed(&mock.take()[0]), (vec!["enc_b".to_string()], None));

        // Back on the primary: its own reasoning, not the secondary's.
        accounts.mark_usable("primary").unwrap();
        conversation.add_user_message("Finish.");
        mock.expect([answer(PRIMARY, "/responses", 200, &turn("enc_a2", "end"))]);
        accounts.generate_text(&conversation, &config).await.unwrap();
        assert_eq!(
            replayed(&mock.take()[0]),
            (vec!["enc_old".to_string(), "enc_a".to_string()], None)
        );
    })
    .await;
}

#[tokio::test]
async fn the_llm_client_applies_its_openai_settings_to_every_account() {
    within(async {
        let mock = Mock::start().await;
        let (accounts, _events) = pool(&mock);
        let client = LLMClient::new()
            .with_chatgpt_accounts(accounts)
            .with_openai_originator("arko-test")
            .with_openai_user_agent("arko-test/1");
        assert!(client.is_openai_chatgpt_oauth().await);
        assert!(client.chatgpt_accounts().is_some());
        assert_eq!(
            client
                .openai_chatgpt_oauth_credentials()
                .await
                .unwrap()
                .account_id,
            PRIMARY
        );
        assert!(
            client
                .update_openai_auth(OpenAIAuth::ApiKey("key".into()))
                .await
                .is_err()
        );
        mock.expect([
            answer(PRIMARY, "/responses", 429, &usage_limit(unix_now() + 3600)),
            answer(SECONDARY, "/responses", 200, &turn("enc_b", "ok")),
            answer(PRIMARY, "/wham/usage", 200, &usage_status(false, false)),
        ]);
        client
            .generate_text(&conversation(), GenerationConfig::default())
            .await
            .unwrap();
        // The credentials of the account the next request goes to.
        assert_eq!(
            client
                .openai_chatgpt_oauth_credentials()
                .await
                .unwrap()
                .account_id,
            SECONDARY
        );
        client.chatgpt_accounts().unwrap().check_now().await;
        for seen in mock.take() {
            if seen.method == "POST" {
                assert_eq!(seen.header("originator").as_deref(), Some("arko-test"));
            }
            assert_eq!(seen.header("user-agent").as_deref(), Some("arko-test/1"));
        }

        // A single account set afterwards replaces the list.
        let single = client.with_openai_client(account("solo", PRIMARY, &mock.url).client);
        assert!(single.chatgpt_accounts().is_none());
        mock.expect([answer(PRIMARY, "/responses", 200, &turn("enc_s", "solo"))]);
        let response = single
            .generate_text(&conversation(), GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(response.model_call_usage.account, None);
    })
    .await;
}

#[test]
fn account_lists_are_validated() {
    let url = "http://127.0.0.1:9";
    assert!(ChatGptAccounts::new([]).is_err());
    assert!(ChatGptAccounts::new([account("a", "one", url), account("a", "two", url)]).is_err());
    assert!(ChatGptAccounts::new([account(" ", "one", url)]).is_err());
    assert!(ChatGptAccounts::new([account("a", "one", url), account("b", "one", url)]).is_err());
    let api_key = ChatGptAccount::new("key", OpenAIClient::new("sk-synthetic"));
    assert!(ChatGptAccounts::new([account("a", "one", url), api_key]).is_err());
    let accounts =
        ChatGptAccounts::new([account("a", "one", url), account("b", "two", url)]).unwrap();
    assert_eq!(accounts.accounts().len(), 2);
    assert_eq!(accounts.active_account(), None);
    // No secret in Debug output.
    let debug = format!("{accounts:?}");
    assert!(debug.contains("\"a\"") && !debug.contains("one") && !debug.contains("refresh"));
}

#[test]
fn usage_endpoint_reports_are_read_as_codex_reads_them() {
    let status = ChatGptUsageStatus::from_json(&json!({
        "plan_type": "pro",
        "rate_limit": {"allowed": false, "limit_reached": true,
            "primary_window": {"used_percent": 100, "limit_window_seconds": 18000,
                "reset_after_seconds": 120, "reset_at": 1_900_000_000u64},
            "secondary_window": {"used_percent": 40, "limit_window_seconds": 604800,
                "reset_after_seconds": 9000, "reset_at": 1_900_500_000u64}},
        "credits": {"has_credits": false, "unlimited": false, "balance": "0"},
        "spend_control": {"reached": false},
        "rate_limit_reached_type": {"type": "rate_limit_reached"},
        "rate_limit_reset_credits": {"available_count": 2},
    }));
    assert_eq!(status.plan_type.as_deref(), Some("pro"));
    assert_eq!(status.allowed, Some(false));
    assert_eq!(status.limit_reached, Some(true));
    assert_eq!(
        status.primary,
        Some(PlanUsageWindow {
            used_percent: 100.0,
            window_minutes: Some(300),
            resets_at: Some(1_900_000_000),
            reset_after_seconds: Some(120),
        })
    );
    assert_eq!(
        status.secondary.as_ref().unwrap().window_minutes,
        Some(10_080)
    );
    assert_eq!(
        status.limit_reached_type.as_deref(),
        Some("rate_limit_reached")
    );
    assert_eq!(status.reset_credits_available, Some(2));
    assert_eq!(status.usable(), Some(false));
    assert_eq!(status.exhausted_until(), Some(at(1_900_000_000)));

    let allowed = |allowed: bool, has_credits: bool, reached: Option<&str>| ChatGptUsageStatus {
        allowed: Some(allowed),
        credits: Some(PlanCredits {
            has_credits,
            ..PlanCredits::default()
        }),
        limit_reached_type: reached.map(String::from),
        ..ChatGptUsageStatus::default()
    };
    assert_eq!(allowed(true, false, None).usable(), Some(true));
    // Credits make an exhausted plan usable, as in Codex.
    assert_eq!(allowed(false, true, None).usable(), Some(true));
    // Unless a limit type (such as depleted credits) is reported.
    assert_eq!(
        allowed(false, true, Some("workspace_owner_credits_depleted")).usable(),
        Some(false)
    );
    let mut spent = allowed(true, false, None);
    spent.spend_control_reached = Some(true);
    assert_eq!(spent.usable(), Some(false));
    // Without `rate_limit.allowed` a check can't tell.
    assert_eq!(ChatGptUsageStatus::from_json(&json!({})).usable(), None);
    assert_eq!(
        ChatGptUsageStatus::from_json(&json!({"rate_limit_reached_type": null})).limit_reached_type,
        None
    );
}

#[test]
fn credit_and_quota_walls_exhaust_an_account_but_other_errors_do_not() {
    let error = |status: u16, body: &str| Error::provider_error("test", Some(status), None, body);
    for body in [
        r#"{"error":{"type":"usage_limit_reached","resets_at":1900000000}}"#,
        r#"{"error":{"type":"usage_not_included"}}"#,
        r#"{"error":{"code":"credit_balance_exhausted"}}"#,
        r#"{"error":{"code":"insufficient_quota"}}"#,
    ] {
        assert!(exhausts_account(&error(429, body)), "{body}");
    }
    for (status, body) in [
        (429, r#"{"error":{"code":"rate_limit_exceeded"}}"#),
        (401, r#"{"error":{"code":"token_expired"}}"#),
        (500, r#"{"error":{"message":"oops"}}"#),
        (400, r#"{"error":{"code":"context_length_exceeded"}}"#),
    ] {
        assert!(!exhausts_account(&error(status, body)), "{body}");
    }
}
