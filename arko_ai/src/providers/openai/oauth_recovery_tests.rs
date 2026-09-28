use super::*;
use crate::model::Model;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub(super) struct Reply {
    path: &'static str,
    status: u16,
    body: String,
    headers: Vec<(&'static str, String)>,
}

pub(super) fn reply(path: &'static str, status: u16, body: &str) -> Reply {
    Reply {
        path,
        status,
        body: body.into(),
        headers: Vec::new(),
    }
}

impl Reply {
    pub(super) fn with_header(mut self, name: &'static str, value: &str) -> Self {
        self.headers.push((name, value.into()));
        self
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> (String, Value) {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 4096];
        let n = socket.read(&mut chunk).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|s| s.trim().parse().unwrap())
                })
                .unwrap();
            if bytes.len() >= end + 4 + length {
                let body = serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                return (headers, body);
            }
        }
    }
}

/// Writes `reply`, with `Retry-After: 60` unless the reply sets its own.
async fn write_reply(socket: &mut tokio::net::TcpStream, reply: &Reply) {
    let content_type = if reply.body.starts_with("data:") {
        "text/event-stream"
    } else {
        "application/json"
    };
    let mut extra: String = reply
        .headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    if !reply
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
    {
        extra.push_str("Retry-After: 60\r\n");
    }
    let response = format!(
        "HTTP/1.1 {} Test\r\nContent-Type: {}\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        content_type,
        extra,
        reply.body.len(),
        reply.body,
    );
    socket.write_all(response.as_bytes()).await.unwrap();
}

pub(super) async fn server(
    replies: Vec<Reply>,
) -> (String, tokio::task::JoinHandle<Vec<(String, Value)>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let (headers, body) = read_request(&mut socket).await;
            assert!(headers.starts_with(&format!("POST {} ", reply.path)));
            requests.push((headers, body));
            write_reply(&mut socket, &reply).await;
        }
        requests
    });
    (url, task)
}

/// A server answering requests concurrently with `replies` in the order the
/// requests arrive, for as long as the test runs; a request beyond them gets
/// no answer. Every request is recorded, and `arrived` gets its body as it
/// comes in. With `gated`, each answer first waits for a permit on `gate`.
struct TestServer {
    url: String,
    requests: Arc<std::sync::Mutex<Vec<(String, Value)>>>,
    arrived: tokio::sync::mpsc::UnboundedReceiver<Value>,
    gate: Arc<tokio::sync::Semaphore>,
}

impl TestServer {
    async fn start(replies: Vec<Reply>, gated: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (arrivals, arrived) = tokio::sync::mpsc::unbounded_channel();
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let replies = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
            replies,
        )));
        let recorded = requests.clone();
        let permits = gate.clone();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let (recorded, arrivals, permits, replies) = (
                    recorded.clone(),
                    arrivals.clone(),
                    permits.clone(),
                    replies.clone(),
                );
                tokio::spawn(async move {
                    let (headers, body) = read_request(&mut socket).await;
                    let reply = replies.lock().unwrap().pop_front();
                    recorded.lock().unwrap().push((headers, body.clone()));
                    let _ = arrivals.send(body);
                    if gated {
                        permits.acquire().await.unwrap().forget();
                    }
                    match reply {
                        Some(reply) => write_reply(&mut socket, &reply).await,
                        // Hold the connection open: the request never completes.
                        None => std::future::pending().await,
                    }
                });
            }
        });
        Self {
            url,
            requests,
            arrived,
            gate,
        }
    }

    fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

/// Points `client` at the test server `url` (Responses and token requests).
fn served_by(mut client: OpenAIClient, url: &str) -> OpenAIClient {
    client = client
        .with_base_url(url)
        .with_oauth_token_url(format!("{url}/token"));
    client.http_client = crate::providers::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    client
}

/// A synthetic JWT expiring at `exp` (Unix seconds).
fn jwt(exp: u64) -> String {
    let claims = URL_SAFE_NO_PAD.encode(format!(r#"{{"exp":{exp}}}"#));
    format!("header.{claims}.signature")
}

/// Expires in 2100.
fn fresh_jwt() -> String {
    jwt(4_102_444_800)
}

pub(super) fn client(url: &str) -> OpenAIClient {
    served_by(
        OpenAIClient::new_chatgpt_oauth(
            &fresh_jwt(),
            Some("synthetic-refresh"),
            "synthetic-account",
        ),
        url,
    )
}

/// An API-key client sending to the test server.
pub(super) fn api_key_client(url: &str) -> OpenAIClient {
    let mut client = client(url);
    client.auth = Arc::new(RwLock::new(OpenAIAuth::ApiKey("synthetic-key".into())));
    client
}

fn expired() -> Reply {
    reply("/responses", 401, r#"{"error":{"code":"token_expired"}}"#)
}

fn refreshed() -> Reply {
    reply(
        "/token",
        200,
        r#"{"access_token":"synthetic-new-access","refresh_token":"synthetic-new-refresh"}"#,
    )
}

pub(super) fn completed_response() -> Value {
    serde_json::json!({
        "id":"resp_test", "object":"response", "created_at":0, "status":"completed",
        "model":"gpt-6-astra", "output":[{"type":"message", "id":"msg_test", "status":"completed", "role":"assistant",
        "content":[{"type":"output_text", "text":"ok"}]}],
        "usage":{"input_tokens":1, "output_tokens":1}, "metadata":{},
    })
}

#[tokio::test]
async fn oauth_recovery_handles_server_expiry_for_text_and_stream_without_an_output_cap() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for streaming in [false, true] {
            // ChatGPT OAuth text calls stream too.
            let body = format!(
                "data: {}\n\n",
                serde_json::json!({"type":"response.completed", "response":completed_response()})
            );
            let (url, server) = server(vec![
                expired(),
                refreshed(),
                reply("/responses", 200, &body),
            ])
            .await;
            // One refresh token per round: the first round's is spent.
            let refresh_token = format!("server-expiry-{streaming}");
            let client = served_by(
                OpenAIClient::new_chatgpt_oauth(
                    &fresh_jwt(),
                    Some(&refresh_token),
                    "synthetic-account",
                ),
                &url,
            );
            assert!(!client.is_oauth_token_expired(Duration::from_secs(60)).await);
            let conversation = Conversation::new(Model::openai_gpt_6_astra());
            if streaming {
                let mut stream = client
                    .generate_stream(&conversation, &GenerationConfig::default())
                    .await
                    .unwrap();
                assert!(matches!(
                    stream.next().await.unwrap().unwrap(),
                    StreamEvent::End(_)
                ));
                assert!(stream.next().await.is_none());
            } else {
                let response = client
                    .generate_text(&conversation, &GenerationConfig::default())
                    .await
                    .unwrap();
                assert_eq!(response.model_call_usage.call_id, "resp_test");
            }
            let calls = server.await.unwrap();
            assert_eq!(calls.len(), 3);
            assert_eq!(
                calls[0].1, calls[2].1,
                "retry must preserve the model request"
            );
            assert!(calls[0].1.get("max_output_tokens").is_none());
            assert_eq!(calls[0].1["stream"], true);
            assert_eq!(calls[1].1["refresh_token"], refresh_token.as_str());
            assert!(
                calls[2]
                    .0
                    .to_lowercase()
                    .contains("authorization: bearer synthetic-new-access")
            );
            let credentials = client.chatgpt_oauth_credentials().await.unwrap();
            assert_eq!(
                credentials.refresh_token.as_deref(),
                Some("synthetic-new-refresh")
            );
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn oauth_recovery_returns_a_second_rejection_without_a_refresh_loop() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![expired(), refreshed(), expired()]).await;
        let client = client(&url);
        let error = client.generate_text(&Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default()).await.unwrap_err();
        assert!(matches!(error, Error::ProviderError { status: Some(401), ref code, .. } if code.as_deref() == Some("token_expired")));
        assert!(!error.is_retryable());
        assert_eq!(server.await.unwrap().len(), 3);
    }).await.unwrap();
}

#[tokio::test]
async fn oauth_recovery_does_not_refresh_api_keys_other_401s_or_missing_refresh_tokens() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for mode in 0..3 {
            let failure = if mode == 1 {
                reply(
                    "/responses",
                    401,
                    r#"{"error":{"code":"permission_denied"}}"#,
                )
            } else {
                expired()
            };
            let (url, server) = server(vec![failure]).await;
            let client = client(&url);
            if mode == 0 {
                client
                    .update_auth(OpenAIAuth::ApiKey("synthetic-key".into()))
                    .await;
            } else if mode == 2 {
                let mut credentials = client.chatgpt_oauth_credentials().await.unwrap();
                credentials.refresh_token = None;
                client
                    .update_auth(OpenAIAuth::ChatGptOAuth(credentials))
                    .await;
            }
            let error = client
                .generate_text(
                    &Conversation::new(Model::openai_gpt_6_astra()),
                    &GenerationConfig::default(),
                )
                .await
                .unwrap_err();
            assert!(!error.is_retryable());
            if mode == 2 {
                assert!(matches!(error, Error::AuthenticationFailed(_)));
            } else {
                assert!(matches!(
                    error,
                    Error::ProviderError {
                        status: Some(401),
                        ..
                    }
                ));
            }
            assert_eq!(server.await.unwrap().len(), 1);
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn oauth_recovery_refresh_failures_return_to_bounded_agent_recovery() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for (status, body, retryable) in [
            (429, r#"{"error":"temporarily_unavailable"}"#, true),
            (400, r#"{"error":"invalid_grant"}"#, false),
        ] {
            let (url, server) = server(vec![expired(), reply("/token", status, body)]).await;
            // Reactive recovery must remain bounded even with legacy unlimited refresh enabled.
            let client = client(&url).with_oauth_rate_limit_retries(true);
            let error = client
                .generate_text(
                    &Conversation::new(Model::openai_gpt_6_astra()),
                    &GenerationConfig::default(),
                )
                .await
                .unwrap_err();
            assert_eq!(error.is_retryable(), retryable);
            assert_eq!(error.retry_after(), Some(Duration::from_secs(60)));
            assert_eq!(server.await.unwrap().len(), 2);
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn oauth_recovery_concurrent_rejections_rotate_once_and_share_the_new_credentials() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![refreshed()]).await;
        let client = client(&url);
        let peer = client.clone();
        let rejected = client.snapshot_auth().await;
        let (a, b) = tokio::join!(
            client.refresh_oauth_token_for_auth(&rejected, false),
            peer.refresh_oauth_token_for_auth(&rejected, false),
        );
        assert_eq!(a.unwrap().access_token, "synthetic-new-access");
        assert_eq!(b.unwrap().access_token, "synthetic-new-access");
        assert_eq!(server.await.unwrap().len(), 1);
        assert_eq!(
            peer.chatgpt_oauth_credentials()
                .await
                .unwrap()
                .refresh_token
                .as_deref(),
            Some("synthetic-new-refresh")
        );
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn public_http_client_refreshes_without_a_process_wide_crypto_provider() {
    // Building the public client must not need an installed crypto provider.
    let _ = crate::http_client();
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![refreshed()]).await;
        let http_client = crate::http_client_builder().no_proxy().build().unwrap();
        let result = crate::refresh_chatgpt_oauth_token(
            &http_client,
            "synthetic-refresh",
            Some("test-client"),
            Some(&format!("{url}/token")),
        )
        .await
        .unwrap();
        assert_eq!(result.access_token, "synthetic-new-access");
        let requests = server.await.unwrap();
        assert_eq!(requests[0].1["refresh_token"], "synthetic-refresh");
        assert_eq!(requests[0].1["client_id"], "test-client");
    })
    .await
    .unwrap();
}

#[test]
fn credential_debug_output_redacts_tokens() {
    let credentials = ChatGptOAuthCredentials {
        access_token: "synthetic-access".into(),
        refresh_token: Some("synthetic-refresh".into()),
        account_id: "synthetic-account".into(),
    };
    let debug = format!("{credentials:?}");
    assert!(!debug.contains("synthetic-account"), "{debug}");
    assert!(debug.contains("account_id: \"<redacted>\""), "{debug}");
    let refreshed = OAuthRefreshResult {
        access_token: "synthetic-access".into(),
        refresh_token: Some("synthetic-refresh".into()),
        id_token: Some("synthetic-id".into()),
    };
    let all = format!("{debug} {refreshed:?} {refreshed:#?}");
    for secret in ["synthetic-access", "synthetic-refresh", "synthetic-id"] {
        assert!(!all.contains(secret), "{all}");
    }
    let without_refresh = ChatGptOAuthCredentials {
        refresh_token: None,
        ..credentials
    };
    assert!(format!("{without_refresh:?}").contains("refresh_token: None"));
}

/// A scripted [`ChatGptOAuthStore`]: `before_refresh` answers from `reads`
/// in order (then `None`), and saved credentials are recorded.
#[derive(Default)]
struct ScriptedStore {
    reads: std::sync::Mutex<
        std::collections::VecDeque<Result<Option<ChatGptOAuthCredentials>, Error>>,
    >,
    seen: std::sync::Mutex<Vec<ChatGptOAuthCredentials>>,
    saved: std::sync::Mutex<Vec<(ChatGptOAuthCredentials, Option<String>)>>,
    fail_saves: bool,
}

impl ChatGptOAuthStore for ScriptedStore {
    fn before_refresh(
        &self,
        current: &ChatGptOAuthCredentials,
    ) -> Result<Option<ChatGptOAuthCredentials>, Error> {
        self.seen.lock().unwrap().push(current.clone());
        self.reads.lock().unwrap().pop_front().unwrap_or(Ok(None))
    }

    fn save_refreshed(
        &self,
        refreshed: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<(), Error> {
        self.saved
            .lock()
            .unwrap()
            .push((refreshed.clone(), id_token.map(String::from)));
        if self.fail_saves {
            return Err(Error::AuthenticationFailed("could not save".into()));
        }
        Ok(())
    }
}

fn stored(access: &str, refresh: &str) -> ChatGptOAuthCredentials {
    ChatGptOAuthCredentials {
        access_token: access.into(),
        refresh_token: Some(refresh.into()),
        account_id: "synthetic-account".into(),
    }
}

fn with_store(client: OpenAIClient, store: &Arc<ScriptedStore>) -> OpenAIClient {
    client.with_credential_store(Some(store.clone() as Arc<dyn ChatGptOAuthStore>))
}

fn completed_stream() -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"type":"response.completed", "response":completed_response()})
    )
}

#[tokio::test]
async fn a_store_is_read_before_each_refresh_and_receives_the_rotated_tokens() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![
            expired(),
            reply(
                "/token",
                200,
                r#"{"access_token":"synthetic-new-access","refresh_token":"synthetic-new-refresh","id_token":"synthetic-id"}"#,
            ),
            reply("/responses", 200, &completed_stream()),
        ])
        .await;
        let store = Arc::new(ScriptedStore::default());
        let client = with_store(client(&url), &store);
        client
            .generate_text(&Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(server.await.unwrap().len(), 3);
        let seen = store.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1, "read once, before the refresh");
        assert_eq!(seen[0].refresh_token.as_deref(), Some("synthetic-refresh"));
        let saved = store.saved.lock().unwrap().clone();
        assert_eq!(
            saved,
            [(stored("synthetic-new-access", "synthetic-new-refresh"), Some("synthetic-id".into()))]
        );
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn tokens_another_program_refreshed_are_used_without_a_refresh_request() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // No /token reply: a refresh request would fail the test.
        let (url, server) = server(vec![
            expired(),
            reply("/responses", 200, &completed_stream()),
        ])
        .await;
        let store = Arc::new(ScriptedStore::default());
        store
            .reads
            .lock()
            .unwrap()
            .push_back(Ok(Some(stored("access-from-codex", "refresh-from-codex"))));
        let client = with_store(client(&url), &store);
        client
            .generate_text(
                &Conversation::new(Model::openai_gpt_6_astra()),
                &GenerationConfig::default(),
            )
            .await
            .unwrap();
        let calls = server.await.unwrap();
        assert_eq!(calls.len(), 2);
        assert!(
            calls[1]
                .0
                .to_lowercase()
                .contains("authorization: bearer access-from-codex")
        );
        assert_eq!(
            client.chatgpt_oauth_credentials().await.unwrap(),
            stored("access-from-codex", "refresh-from-codex")
        );
        assert!(
            store.saved.lock().unwrap().is_empty(),
            "nothing was refreshed"
        );
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn an_unreadable_store_or_another_account_stops_the_refresh() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for read in [
            Err(Error::AuthenticationFailed(
                "the store holds another account".into(),
            )),
            Ok(Some(ChatGptOAuthCredentials {
                account_id: "another-account".into(),
                ..stored("a", "r")
            })),
        ] {
            let (url, server) = server(vec![expired()]).await;
            let store = Arc::new(ScriptedStore::default());
            store.reads.lock().unwrap().push_back(read);
            let client = with_store(client(&url), &store);
            let error = client
                .generate_text(
                    &Conversation::new(Model::openai_gpt_6_astra()),
                    &GenerationConfig::default(),
                )
                .await
                .unwrap_err();
            assert!(matches!(error, Error::AuthenticationFailed(_)), "{error}");
            assert!(!error.is_retryable());
            assert_eq!(server.await.unwrap().len(), 1, "no refresh request");
            let credentials = client.chatgpt_oauth_credentials().await.unwrap();
            assert_eq!(credentials.account_id, "synthetic-account");
            assert_eq!(
                credentials.refresh_token.as_deref(),
                Some("synthetic-refresh")
            );
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_refresh_rejected_after_another_program_rotated_adopts_its_tokens() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![reply(
            "/token",
            400,
            r#"{"error":{"code":"refresh_token_reused"}}"#,
        )])
        .await;
        let store = Arc::new(ScriptedStore::default());
        {
            let mut reads = store.reads.lock().unwrap();
            reads.push_back(Ok(None));
            reads.push_back(Ok(Some(stored("access-from-codex", "refresh-from-codex"))));
        }
        let client = with_store(client(&url), &store);
        let result = client.refresh_oauth_token_once().await.unwrap();
        assert_eq!(result.access_token, "access-from-codex");
        assert_eq!(server.await.unwrap().len(), 1);
        assert_eq!(store.seen.lock().unwrap().len(), 2);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_failed_save_is_reported_but_the_client_keeps_the_new_tokens() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![refreshed()]).await;
        let store = Arc::new(ScriptedStore {
            fail_saves: true,
            ..Default::default()
        });
        let client = with_store(client(&url), &store);
        let error = client.refresh_oauth_token_once().await.unwrap_err();
        assert!(matches!(error, Error::AuthenticationFailed(_)), "{error}");
        assert_eq!(server.await.unwrap().len(), 1);
        assert_eq!(
            client.chatgpt_oauth_credentials().await.unwrap(),
            stored("synthetic-new-access", "synthetic-new-refresh")
        );
    })
    .await
    .unwrap();
}

/// Rate-limit retry timing in milliseconds, so loop tests run in real time
/// (a paused clock jumps ahead while their loopback I/O is under way). One
/// wait honors at most `max_server_hint` of a hint.
fn quick_retries(mut client: OpenAIClient, max_server_hint: Duration) -> OpenAIClient {
    client.oauth_refresh_retry = OAuthRefreshRetry {
        backoff: crate::retry::BackoffPolicy {
            initial: Duration::from_millis(5),
            max: Duration::from_millis(20),
            jitter: false,
        },
        max_server_hint,
    };
    client
}

/// Each rate-limited retry re-reads the store (Codex may refresh while this
/// client waits), and the retries stop at the cap. (The 900 s cap on one
/// wait's server hint is unit-tested with the delay computation.)
#[tokio::test]
async fn rate_limited_refreshes_retry_a_bounded_number_of_times() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let limited = || reply("/token", 429, r#"{"error":{"code":"rate_limit_exceeded"}}"#);
        let (url, server) = server(
            (0..=OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES)
                .map(|_| limited())
                .collect(),
        )
        .await;
        let store = Arc::new(ScriptedStore::default());
        // The server's Retry-After: 60 is honored up to 20 ms here.
        let client = quick_retries(client(&url), Duration::from_millis(20));
        let client = with_store(client, &store).with_oauth_rate_limit_retries(true);
        let started = std::time::Instant::now();
        let error = client.refresh_oauth_token().await.unwrap_err();
        assert!(
            matches!(
                error,
                Error::ProviderError {
                    status: Some(429),
                    ..
                }
            ),
            "{error}"
        );
        let requests = server.await.unwrap().len() as u32;
        assert_eq!(requests, OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES + 1);
        // One read before each request, plus one after the last failure.
        assert_eq!(store.seen.lock().unwrap().len() as u32, requests + 1);
        // Every wait honored the (capped) server hint.
        assert!(
            started.elapsed() >= Duration::from_millis(20) * OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES
        );
    })
    .await
    .unwrap();
}

fn permission_denied() -> Reply {
    reply(
        "/responses",
        401,
        r#"{"error":{"code":"permission_denied"}}"#,
    )
}

/// Like Codex's unauthorized recovery: any 401 first re-reads the store, and
/// newer tokens there are used for one more try; only `token_expired`
/// refreshes.
#[tokio::test]
async fn any_401_adopts_newer_stored_tokens_and_sends_again_once() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for second_reply_ok in [true, false] {
            // No /token reply: a refresh request would fail the test.
            let second = if second_reply_ok {
                reply("/responses", 200, &completed_stream())
            } else {
                permission_denied()
            };
            let (url, server) = server(vec![permission_denied(), second]).await;
            let store = Arc::new(ScriptedStore::default());
            store
                .reads
                .lock()
                .unwrap()
                .push_back(Ok(Some(stored("access-from-codex", "refresh-from-codex"))));
            let client = with_store(client(&url), &store);
            let result = client
                .generate_text(
                    &Conversation::new(Model::openai_gpt_6_astra()),
                    &GenerationConfig::default(),
                )
                .await;
            let calls = server.await.unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].1, calls[1].1, "the same request is sent again");
            assert!(
                calls[1]
                    .0
                    .to_lowercase()
                    .contains("authorization: bearer access-from-codex")
            );
            if second_reply_ok {
                result.unwrap();
            } else {
                // A second rejection is returned; nothing loops.
                let error = result.unwrap_err();
                assert!(
                    matches!(
                        error,
                        Error::ProviderError {
                            status: Some(401),
                            ..
                        }
                    ),
                    "{error}"
                );
            }
            assert_eq!(store.seen.lock().unwrap().len(), 1, "read once");
            assert!(
                store.saved.lock().unwrap().is_empty(),
                "nothing was refreshed"
            );
            assert_eq!(
                client.chatgpt_oauth_credentials().await.unwrap(),
                stored("access-from-codex", "refresh-from-codex")
            );
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_401_without_newer_stored_tokens_is_returned_without_a_refresh() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for read in [
            Ok(None),
            Err(Error::AuthenticationFailed("the store cannot be read".into())),
            Ok(Some(ChatGptOAuthCredentials {
                account_id: "another-account".into(),
                ..stored("a", "r")
            })),
        ] {
            let (url, server) = server(vec![permission_denied()]).await;
            let store = Arc::new(ScriptedStore::default());
            store.reads.lock().unwrap().push_back(read);
            let client = with_store(client(&url), &store);
            let before = client.chatgpt_oauth_credentials().await.unwrap();
            let error = client
                .generate_text(&Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default())
                .await
                .unwrap_err();
            // The 401 itself, not the store's error.
            assert!(
                matches!(&error, Error::ProviderError { status: Some(401), code: Some(code), .. } if code == "permission_denied"),
                "{error}"
            );
            assert_eq!(server.await.unwrap().len(), 1, "no refresh request, no resend");
            assert_eq!(store.seen.lock().unwrap().len(), 1);
            assert!(store.saved.lock().unwrap().is_empty());
            assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), before);
        }
    })
    .await
    .unwrap();
}

/// An `auth.json` in a temporary directory holding `client()`'s refresh
/// token and account.
fn auth_file(access_token: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auth.json");
    let contents = serde_json::json!({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": access_token,
            "refresh_token": "synthetic-refresh",
            "account_id": "synthetic-account",
            "id_token": "synthetic-id",
        },
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&contents).unwrap()).unwrap();
    (dir, path)
}

fn file_credentials(path: &std::path::Path) -> ChatGptOAuthCredentials {
    crate::codex_auth::CodexAuthFile::load(path)
        .unwrap()
        .credentials()
        .unwrap()
}

fn second_refresh() -> Reply {
    reply(
        "/token",
        200,
        r#"{"access_token":"second-access","refresh_token":"second-refresh"}"#,
    )
}

#[tokio::test]
async fn clients_made_separately_from_one_auth_file_refresh_it_once() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // A second refresh request would be answered too.
        let server = TestServer::start(vec![refreshed(), second_refresh()], false).await;
        let (_dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let first = served_by(file.openai_client().unwrap(), &server.url);
        let second = served_by(file.openai_client().unwrap(), &server.url);
        let (a, b) = tokio::join!(first.refresh_oauth_token(), second.refresh_oauth_token());
        assert_eq!(a.unwrap().access_token, "synthetic-new-access");
        assert_eq!(b.unwrap().access_token, "synthetic-new-access");
        assert_eq!(server.request_count(), 1);
        let new = stored("synthetic-new-access", "synthetic-new-refresh");
        for client in [&first, &second] {
            assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), new);
        }
        assert_eq!(file_credentials(&path), new);
    })
    .await
    .unwrap();
}

/// The first client's rotated tokens can't be saved, so the file keeps the
/// spent refresh token: the other clients of the file, one made from it
/// afterwards, and the first client too (it doesn't rotate its tokens again
/// before they are saved) stop with an error naming the recovery file; no
/// one is handed the unsaved tokens. Once the file can be written, the first
/// client's next refresh saves them first (with their id_token) and
/// refreshes them, and the others take the newest tokens from the file.
#[tokio::test]
async fn a_rotation_another_client_could_not_save_stops_every_refresh_until_it_is_saved() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let with_id_token = reply(
            "/token",
            200,
            r#"{"access_token":"synthetic-new-access","refresh_token":"synthetic-new-refresh","id_token":"synthetic-new-id"}"#,
        );
        // A refresh with the spent token would be answered too.
        let server = TestServer::start(vec![with_id_token, second_refresh(), refreshed()], false).await;
        let (dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let first = served_by(file.openai_client().unwrap(), &server.url);
        let second = served_by(file.openai_client().unwrap(), &server.url);
        // A directory where the backup goes makes saving fail.
        let backup = dir.path().join("auth.json.bak");
        std::fs::create_dir(&backup).unwrap();
        let error = first.refresh_oauth_token_once().await.unwrap_err();
        assert!(error.to_string().contains(".recovered-"), "{error}");
        let rotated = stored("synthetic-new-access", "synthetic-new-refresh");
        assert_eq!(first.chatgpt_oauth_credentials().await.unwrap(), rotated);
        let spent = file_credentials(&path);
        assert_eq!(spent.refresh_token.as_deref(), Some("synthetic-refresh"));

        let third = served_by(file.openai_client().unwrap(), &server.url);
        for client in [&second, &third, &first] {
            let error = client.refresh_oauth_token_once().await.unwrap_err();
            assert!(error.to_string().contains(".recovered-"), "{error}");
        }
        assert_eq!(second.chatgpt_oauth_credentials().await.unwrap(), spent);
        assert_eq!(first.chatgpt_oauth_credentials().await.unwrap(), rotated);
        assert_eq!(server.request_count(), 1);
        assert_eq!(file_credentials(&path), spent);

        std::fs::remove_dir(&backup).unwrap();
        let result = first.refresh_oauth_token_once().await.unwrap();
        assert_eq!(result.access_token, "second-access");
        let newest = stored("second-access", "second-refresh");
        assert_eq!(file_credentials(&path), newest);
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["id_token"], "synthetic-new-id");
        for client in [&second, &third] {
            assert_eq!(client.refresh_oauth_token_once().await.unwrap().access_token, "second-access");
            assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), newest);
        }
        assert_eq!(sent_refresh_tokens(&server), ["synthetic-refresh", "synthetic-new-refresh"]);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_refresh_dropped_after_its_request_was_sent_still_installs_and_saves_the_tokens() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut server = TestServer::start(vec![refreshed()], true).await;
        let (_dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let client = served_by(file.openai_client().unwrap(), &server.url);
        let caller = tokio::spawn({
            let client = client.clone();
            async move { client.refresh_oauth_token_once().await }
        });
        let request = server.arrived.recv().await.unwrap();
        assert_eq!(request["refresh_token"], "synthetic-refresh");
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        // The refresh holds the credentials until the new tokens are
        // installed and saved: a host about to shut its runtime down awaits
        // them, and so waits for the refresh.
        let host = crate::LLMClient::new().with_openai_client(client.clone());
        let settled = host.openai_chatgpt_oauth_credentials();
        tokio::pin!(settled);
        assert!(futures::poll!(&mut settled).is_pending());
        // The refresh token rotates once the server answers, whether or not
        // the caller is still there.
        server.gate.add_permits(1);
        let new = stored("synthetic-new-access", "synthetic-new-refresh");
        assert_eq!(settled.await.unwrap(), new);
        assert_eq!(file_credentials(&path), new);
        assert_eq!(server.request_count(), 1);
    })
    .await
    .unwrap();
}

/// While one client's refresh waits on a stalled token endpoint (holding the
/// file's refresh lock), another client of the same file keeps serving
/// requests with its still valid token, also while it waits for that lock to
/// refresh; it then finds the first client's tokens in the file.
#[tokio::test]
async fn a_client_waiting_for_another_clients_refresh_keeps_serving_requests() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut tokens = TestServer::start(vec![refreshed()], true).await;
        let answered = || reply("/responses", 200, &completed_stream());
        let responses = TestServer::start(vec![answered(), answered()], false).await;
        let (_dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let client = || {
            served_by(file.openai_client().unwrap(), &responses.url)
                .with_oauth_token_url(format!("{}/token", tokens.url))
        };
        let (first, second) = (client(), client());
        let refreshing = tokio::spawn({
            let first = first.clone();
            async move { first.refresh_oauth_token_once().await }
        });
        // The refresh request is out: the first client holds the file's
        // refresh lock until the token endpoint answers.
        tokens.arrived.recv().await.unwrap();
        let conversation = Conversation::new(Model::openai_gpt_6_astra());
        second
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap();
        // The second client's refresh waits for the file's lock without
        // holding the second client's credentials.
        let waiting = second.refresh_oauth_token_once();
        tokio::pin!(waiting);
        assert!(futures::poll!(&mut waiting).is_pending());
        let held = second.chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(held.refresh_token.as_deref(), Some("synthetic-refresh"));
        second
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap();
        assert!(futures::poll!(&mut waiting).is_pending());
        assert_eq!(responses.request_count(), 2);

        tokens.gate.add_permits(1);
        assert_eq!(
            refreshing.await.unwrap().unwrap().access_token,
            "synthetic-new-access"
        );
        assert_eq!(waiting.await.unwrap().access_token, "synthetic-new-access");
        assert_eq!(
            tokens.request_count(),
            1,
            "the second client adopted, not refreshed"
        );
        let new = stored("synthetic-new-access", "synthetic-new-refresh");
        for client in [&first, &second] {
            assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), new);
        }
        assert_eq!(file_credentials(&path), new);
    })
    .await
    .unwrap();
}

/// A token endpoint that takes the request but never answers: the refresh
/// times out with a transient error (back to the agent's bounded recovery)
/// and lets go of the credentials.
#[tokio::test]
async fn a_stalled_token_endpoint_times_out_with_a_transient_error() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // The 401 is answered; the refresh request after it never is.
        let server = TestServer::start(vec![expired()], false).await;
        let mut client = client(&server.url);
        assert_eq!(client.oauth_refresh_timeout, Some(OAUTH_REFRESH_TIMEOUT));
        assert_eq!(OAUTH_REFRESH_TIMEOUT, Duration::from_secs(60));
        client.oauth_refresh_timeout = Some(Duration::from_millis(300));
        let before = client.chatgpt_oauth_credentials().await.unwrap();
        let error = client
            .generate_text(
                &Conversation::new(Model::openai_gpt_6_astra()),
                &GenerationConfig::default(),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::HttpError(error) if error.is_timeout()),
            "{error}"
        );
        assert_eq!(error.retry_class(), crate::RetryClass::Transient);
        assert_eq!(server.request_count(), 2, "the refresh request was sent");
        assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), before);

        let error = client.refresh_oauth_token_once().await.unwrap_err();
        assert!(
            matches!(&error, Error::HttpError(error) if error.is_timeout()),
            "{error}"
        );
        assert!(error.is_retryable());
        assert_eq!(server.request_count(), 3);
        assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), before);
    })
    .await
    .unwrap();
}

/// A rate-limited refresh waits with no lock held: another caller refreshes
/// meanwhile, and the waiting one then takes its tokens instead of sending a
/// third request. Real time, with a one-second Retry-After. (That one wait
/// honors at most 15 minutes of a longer hint is unit-tested with the delay
/// computation.)
#[tokio::test]
async fn a_rate_limited_refresh_waits_without_holding_the_credentials() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let limited = reply("/token", 429, r#"{"error":{"code":"rate_limit_exceeded"}}"#)
            .with_header("Retry-After", "1");
        let mut server = TestServer::start(vec![limited, refreshed()], false).await;
        let client = quick_retries(client(&server.url), OAUTH_REFRESH_RETRY.max_server_hint)
            .with_oauth_rate_limit_retries(true);
        let waiting = tokio::spawn({
            let client = client.clone();
            async move {
                let started = std::time::Instant::now();
                let result = client.refresh_oauth_token().await;
                (result, started.elapsed())
            }
        });
        server.arrived.recv().await.unwrap();
        // While the first caller waits out the 429, another one refreshes.
        let other = client.refresh_oauth_token_once().await.unwrap();
        assert_eq!(other.access_token, "synthetic-new-access");
        // The waiting caller then finds the new tokens instead of refreshing.
        let (result, waited) = waiting.await.unwrap();
        assert_eq!(result.unwrap().access_token, "synthetic-new-access");
        assert!(waited >= Duration::from_secs(1), "{waited:?}");
        assert_eq!(server.request_count(), 2);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn an_explicit_refresh_that_waits_for_another_returns_its_tokens() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut server = TestServer::start(vec![refreshed(), second_refresh()], true).await;
        let client = client(&server.url);
        let first = tokio::spawn({
            let client = client.clone();
            async move { client.refresh_oauth_token().await }
        });
        server.arrived.recv().await.unwrap();
        // The first refresh holds the credentials; this one waits for them.
        let second = client.refresh_oauth_token();
        tokio::pin!(second);
        assert!(futures::poll!(&mut second).is_pending());
        server.gate.add_permits(2);
        assert_eq!(
            first.await.unwrap().unwrap().access_token,
            "synthetic-new-access"
        );
        assert_eq!(second.await.unwrap().access_token, "synthetic-new-access");
        assert_eq!(server.request_count(), 1);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn concurrent_expiry_checks_refresh_once() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let margin = Duration::from_secs(60);
        let answer = || {
            let body = serde_json::json!({"access_token": fresh_jwt(), "refresh_token": "synthetic-new-refresh"});
            reply("/token", 200, &body.to_string())
        };
        for clones in [false, true] {
            let server = TestServer::start(vec![answer(), answer()], false).await;
            // One refresh token per round: the first round's is spent.
            let refresh_token = format!("expiry-check-{clones}");
            let expired = OpenAIClient::new_chatgpt_oauth(&jwt(1), Some(&refresh_token), "synthetic-account");
            let client = served_by(expired, &server.url);
            let (a, b) = if clones {
                let first = crate::LLMClient::new().with_openai_client(client.clone());
                let second = first.clone();
                tokio::join!(
                    first.refresh_openai_oauth_token_if_expiring(margin),
                    second.refresh_openai_oauth_token_if_expiring(margin),
                )
            } else {
                tokio::join!(
                    client.refresh_oauth_token_if_expiring(margin),
                    client.refresh_oauth_token_if_expiring(margin),
                )
            };
            // One call refreshed; the other found the new token.
            let (a, b) = (a.unwrap(), b.unwrap());
            assert_eq!(usize::from(a.is_some()) + usize::from(b.is_some()), 1, "{clones}");
            assert_eq!(a.or(b).unwrap().access_token, fresh_jwt());
            assert_eq!(server.request_count(), 1, "{clones}");
            assert!(client.refresh_oauth_token_if_expiring(margin).await.unwrap().is_none());
            assert_eq!(server.request_count(), 1);
            let credentials = client.chatgpt_oauth_credentials().await.unwrap();
            assert_eq!(credentials.refresh_token.as_deref(), Some("synthetic-new-refresh"));
        }
        // API-key and unconfigured clients have nothing to refresh.
        let server = TestServer::start(Vec::new(), false).await;
        let api_key = api_key_client(&server.url);
        assert!(api_key.refresh_oauth_token_if_expiring(margin).await.unwrap().is_none());
        let unconfigured = crate::LLMClient::new();
        assert!(unconfigured.refresh_openai_oauth_token_if_expiring(margin).await.unwrap().is_none());
        assert_eq!(server.request_count(), 0);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn an_expiry_check_adopts_stored_tokens_and_refreshes_them_only_if_they_expire_too() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let margin = Duration::from_secs(60);
        let server = TestServer::start(vec![refreshed()], false).await;
        for (stored_access, refreshes) in [(fresh_jwt(), false), (jwt(1), true)] {
            let store = Arc::new(ScriptedStore::default());
            let from_codex = stored(&stored_access, "refresh-from-codex");
            store
                .reads
                .lock()
                .unwrap()
                .push_back(Ok(Some(from_codex.clone())));
            let expired = OpenAIClient::new_chatgpt_oauth(
                &jwt(1),
                Some("synthetic-refresh"),
                "synthetic-account",
            );
            let client = with_store(served_by(expired, &server.url), &store);
            let result = client
                .refresh_oauth_token_if_expiring(margin)
                .await
                .unwrap()
                .unwrap();
            let saved = store.saved.lock().unwrap().clone();
            if refreshes {
                assert_eq!(result.access_token, "synthetic-new-access");
                assert_eq!(server.request_count(), 1);
                // The stored refresh token was used, not the client's old one.
                assert_eq!(
                    server.requests.lock().unwrap()[0].1["refresh_token"],
                    "refresh-from-codex"
                );
                assert_eq!(
                    saved,
                    [(
                        stored("synthetic-new-access", "synthetic-new-refresh"),
                        None
                    )]
                );
            } else {
                assert_eq!(result.access_token, stored_access);
                assert_eq!(server.request_count(), 0);
                assert!(saved.is_empty());
                assert_eq!(
                    client.chatgpt_oauth_credentials().await.unwrap(),
                    from_codex
                );
            }
        }
    })
    .await
    .unwrap();
}

/// After a rotation that couldn't be saved, the user signs in again (as the
/// recovery message suggests): every client takes the new credentials, the
/// one holding the forgotten rotation too, and none refreshes with the
/// rotated or the spent token.
#[tokio::test]
async fn clients_take_the_credentials_of_a_new_sign_in_after_an_unsaved_rotation() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // Any further refresh request would be answered too.
        let server = TestServer::start(vec![refreshed(), second_refresh()], false).await;
        let (dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let first = served_by(file.openai_client().unwrap(), &server.url);
        let second = served_by(file.openai_client().unwrap(), &server.url);
        let backup = dir.path().join("auth.json.bak");
        std::fs::create_dir(&backup).unwrap();
        assert!(first.refresh_oauth_token_once().await.is_err());
        std::fs::remove_dir(&backup).unwrap();

        let signed_in = serde_json::json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": fresh_jwt(),
                "refresh_token": "signed-in-refresh",
                "account_id": "synthetic-account",
                "id_token": "signed-in-id",
            },
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&signed_in).unwrap()).unwrap();
        for client in [&first, &second] {
            client.refresh_oauth_token_once().await.unwrap();
            let credentials = client.chatgpt_oauth_credentials().await.unwrap();
            assert_eq!(
                credentials.refresh_token.as_deref(),
                Some("signed-in-refresh")
            );
        }
        assert_eq!(
            server.request_count(),
            1,
            "only the first, failed-to-save refresh"
        );
    })
    .await
    .unwrap();
}

/// A host that builds a client per run and drops it: the unsaved rotation
/// outlives the clients, so the next client made from the file doesn't
/// refresh the spent token, and takes the rotation once the file can be
/// written.
#[tokio::test]
async fn an_unsaved_rotation_outlives_the_clients_of_the_file() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let server = TestServer::start(vec![refreshed(), second_refresh()], false).await;
        let (dir, path) = auth_file(&fresh_jwt());
        let backup = dir.path().join("auth.json.bak");
        std::fs::create_dir(&backup).unwrap();
        {
            let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
            let run = served_by(file.openai_client().unwrap(), &server.url);
            assert!(run.refresh_oauth_token_once().await.is_err());
        }
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let next_run = served_by(file.openai_client().unwrap(), &server.url);
        let error = next_run.refresh_oauth_token_once().await.unwrap_err();
        assert!(error.to_string().contains(".recovered-"), "{error}");
        std::fs::remove_dir(&backup).unwrap();
        let result = next_run.refresh_oauth_token_once().await.unwrap();
        assert_eq!(result.access_token, "synthetic-new-access");
        assert_eq!(sent_refresh_tokens(&server), ["synthetic-refresh"]);
    })
    .await
    .unwrap();
}

/// Writes `auth.json` at `path` as Codex does, holding these tokens.
fn write_auth(path: &std::path::Path, access_token: &str, refresh_token: &str, account_id: &str) {
    let contents = serde_json::json!({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "account_id": account_id,
            "id_token": "synthetic-id",
        },
    });
    std::fs::write(path, serde_json::to_vec_pretty(&contents).unwrap()).unwrap();
}

/// A token endpoint answer rotating to these tokens.
fn rotates_to(access_token: &str, refresh_token: &str) -> Reply {
    let body = serde_json::json!({"access_token": access_token, "refresh_token": refresh_token});
    reply("/token", 200, &body.to_string())
}

/// The refresh tokens `server`'s token endpoint received, in order.
fn sent_refresh_tokens(server: &TestServer) -> Vec<String> {
    server
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(headers, _)| headers.starts_with("POST /token "))
        .map(|(_, body)| body["refresh_token"].as_str().unwrap().to_string())
        .collect()
}

/// B1: two clients of one `auth.json`. The first refreshes twice, and the
/// second rotation can't be saved. The idle second client, whose tokens and
/// the file's were both spent by those refreshes, gets an error naming the
/// recovery file from its proactive refresh instead of taking the file's
/// spent tokens and refreshing them; once the file can be written, it takes
/// the rotation. No refresh token reaches the server twice.
#[tokio::test]
async fn an_idle_client_never_refreshes_tokens_another_client_spent() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // Further refresh requests would be answered too.
        let server = TestServer::start(
            vec![
                rotates_to(&jwt(1), "b1-refresh-1"),
                rotates_to(&fresh_jwt(), "b1-refresh-2"),
                rotates_to(&fresh_jwt(), "b1-refresh-3"),
                rotates_to(&fresh_jwt(), "b1-refresh-4"),
            ],
            false,
        )
        .await;
        let (dir, path) = auth_file(&jwt(1));
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let first = served_by(file.openai_client().unwrap(), &server.url);
        let idle = crate::LLMClient::new()
            .with_openai_client(served_by(file.openai_client().unwrap(), &server.url));
        let held = idle.openai_chatgpt_oauth_credentials().await.unwrap();
        first.refresh_oauth_token_once().await.unwrap();
        // A directory in place of that save's backup fails the next save.
        let backup = dir.path().join("auth.json.bak");
        std::fs::remove_file(&backup).unwrap();
        std::fs::create_dir(&backup).unwrap();
        assert!(first.refresh_oauth_token_once().await.is_err());
        assert_eq!(
            file_credentials(&path).refresh_token.as_deref(),
            Some("b1-refresh-1")
        );

        // The agent loop's proactive refresh.
        let margin = Duration::from_secs(60);
        let error = idle
            .refresh_openai_oauth_token_if_expiring(margin)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(".recovered-"), "{error}");
        assert!(!error.is_retryable());
        assert_eq!(idle.openai_chatgpt_oauth_credentials().await.unwrap(), held);
        assert_eq!(
            sent_refresh_tokens(&server),
            ["synthetic-refresh", "b1-refresh-1"]
        );

        std::fs::remove_dir(&backup).unwrap();
        let adopted = idle
            .refresh_openai_oauth_token_if_expiring(margin)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(adopted.refresh_token.as_deref(), Some("b1-refresh-2"));
        assert_eq!(
            file_credentials(&path).refresh_token.as_deref(),
            Some("b1-refresh-2")
        );
        assert_eq!(
            sent_refresh_tokens(&server),
            ["synthetic-refresh", "b1-refresh-1"]
        );
    })
    .await
    .unwrap();
}

/// M1a: two unsaved rotations in a row. The first client's rotation can't be
/// saved; the user signs in again; the second client takes the new sign-in,
/// refreshes it, and can't save that either. The first client then neither
/// takes the spent sign-in tokens nor refreshes them; once the file can be
/// written, it takes the second rotation.
#[tokio::test]
async fn a_client_never_takes_a_sign_in_another_client_spent() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let server = TestServer::start(
            vec![
                rotates_to(&fresh_jwt(), "m1a-refresh-1"),
                rotates_to(&fresh_jwt(), "m1a-refresh-2"),
                rotates_to(&fresh_jwt(), "m1a-refresh-3"),
                rotates_to(&fresh_jwt(), "m1a-refresh-4"),
            ],
            false,
        )
        .await;
        let (dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let first = served_by(file.openai_client().unwrap(), &server.url);
        let second = served_by(file.openai_client().unwrap(), &server.url);
        let backup = dir.path().join("auth.json.bak");
        std::fs::create_dir(&backup).unwrap();
        assert!(first.refresh_oauth_token_once().await.is_err());

        write_auth(&path, &fresh_jwt(), "m1a-signed-in", "synthetic-account");
        let adopted = second.refresh_oauth_token_once().await.unwrap();
        assert_eq!(adopted.refresh_token.as_deref(), Some("m1a-signed-in"));
        assert!(second.refresh_oauth_token_once().await.is_err());
        assert_eq!(
            file_credentials(&path).refresh_token.as_deref(),
            Some("m1a-signed-in")
        );

        let error = first.refresh_oauth_token_once().await.unwrap_err();
        assert!(error.to_string().contains(".recovered-"), "{error}");
        let held = first.chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(held.refresh_token.as_deref(), Some("m1a-refresh-1"));
        assert_eq!(
            sent_refresh_tokens(&server),
            ["synthetic-refresh", "m1a-signed-in"]
        );

        std::fs::remove_dir(&backup).unwrap();
        let taken = first.refresh_oauth_token_once().await.unwrap();
        assert_eq!(taken.refresh_token.as_deref(), Some("m1a-refresh-2"));
        assert_eq!(
            file_credentials(&path).refresh_token.as_deref(),
            Some("m1a-refresh-2")
        );
        assert_eq!(
            sent_refresh_tokens(&server),
            ["synthetic-refresh", "m1a-signed-in"]
        );
    })
    .await
    .unwrap();
}

/// M1b: while the file holds another account, the unsaved rotation is kept
/// (also once its client is dropped). When the file is put back to the
/// spent tokens, a client made from it doesn't refresh them: it gets an
/// error naming the recovery file, then the rotation once it can be saved.
#[tokio::test]
async fn an_unsaved_rotation_is_kept_while_the_file_holds_another_account() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let server = TestServer::start(
            vec![
                rotates_to(&fresh_jwt(), "m1b-refresh-1"),
                rotates_to(&fresh_jwt(), "m1b-refresh-2"),
                rotates_to(&fresh_jwt(), "m1b-refresh-3"),
            ],
            false,
        )
        .await;
        let (dir, path) = auth_file(&fresh_jwt());
        let original = std::fs::read(&path).unwrap();
        let backup = dir.path().join("auth.json.bak");
        std::fs::create_dir(&backup).unwrap();
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let holder = served_by(file.openai_client().unwrap(), &server.url);
        assert!(holder.refresh_oauth_token_once().await.is_err());

        write_auth(&path, &fresh_jwt(), "m1b-other-refresh", "other-account");
        let error = holder
            .refresh_oauth_token_once()
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("different ChatGPT account") && error.contains(".recovered-"),
            "{error}"
        );
        drop(holder);

        std::fs::write(&path, &original).unwrap();
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let later = served_by(file.openai_client().unwrap(), &server.url);
        let error = later.refresh_oauth_token_once().await.unwrap_err();
        assert!(error.to_string().contains(".recovered-"), "{error}");
        assert_eq!(sent_refresh_tokens(&server), ["synthetic-refresh"]);

        std::fs::remove_dir(&backup).unwrap();
        let taken = later.refresh_oauth_token_once().await.unwrap();
        assert_eq!(taken.refresh_token.as_deref(), Some("m1b-refresh-1"));
        assert_eq!(
            file_credentials(&path).refresh_token.as_deref(),
            Some("m1b-refresh-1")
        );
        assert_eq!(sent_refresh_tokens(&server), ["synthetic-refresh"]);
    })
    .await
    .unwrap();
}

fn refused_as_spent(error: &Error) -> bool {
    matches!(error, Error::AuthenticationFailed(message) if message.contains("can't be used again"))
        && !error.is_retryable()
}

/// Once a refresh in this process exchanged a refresh token for a new one,
/// no entry point sends it again or adopts stored credentials holding it:
/// `refresh_chatgpt_oauth_token`, explicit refreshes (with and without
/// rate-limit retries), the proactive refresh, the refresh after a
/// `token_expired` 401, and the store reads before a refresh and after any
/// other 401.
#[tokio::test]
async fn a_spent_refresh_token_is_never_sent_or_adopted_again() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // Anything beyond these would get no answer (and time the test out).
        let server =
            TestServer::start(vec![refreshed(), expired(), permission_denied()], false).await;
        let token_url = format!("{}/token", server.url);
        let http_client = crate::http_client_builder().no_proxy().build().unwrap();
        let exchanged = crate::refresh_chatgpt_oauth_token(
            &http_client,
            "synthetic-refresh",
            None,
            Some(&token_url),
        )
        .await
        .unwrap();
        assert_eq!(
            exchanged.refresh_token.as_deref(),
            Some("synthetic-new-refresh")
        );

        let error = crate::refresh_chatgpt_oauth_token(
            &http_client,
            "synthetic-refresh",
            None,
            Some(&token_url),
        )
        .await
        .unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        // Refresh tokens are unique: not sent to another token URL either.
        let elsewhere = TestServer::start(Vec::new(), false).await;
        let error = client(&elsewhere.url)
            .refresh_oauth_token_once()
            .await
            .unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        assert_eq!(elsewhere.request_count(), 0);

        let holding_spent = client(&server.url);
        let before = holding_spent.chatgpt_oauth_credentials().await.unwrap();
        for error in [
            holding_spent.refresh_oauth_token().await.unwrap_err(),
            holding_spent.refresh_oauth_token_once().await.unwrap_err(),
            holding_spent
                .clone()
                .with_oauth_rate_limit_retries(true)
                .refresh_oauth_token()
                .await
                .unwrap_err(),
        ] {
            assert!(refused_as_spent(&error), "{error}");
        }
        let expiring = served_by(
            OpenAIClient::new_chatgpt_oauth(
                &jwt(1),
                Some("synthetic-refresh"),
                "synthetic-account",
            ),
            &server.url,
        );
        let error = expiring
            .refresh_oauth_token_if_expiring(Duration::from_secs(60))
            .await
            .unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        // A request rejected with `token_expired`.
        let conversation = Conversation::new(Model::openai_gpt_6_astra());
        let error = holding_spent
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        assert_eq!(
            holding_spent.chatgpt_oauth_credentials().await.unwrap(),
            before
        );

        // A store offering credentials holding the spent token.
        let offered = stored(&fresh_jwt(), "synthetic-refresh");
        let store = Arc::new(ScriptedStore::default());
        store
            .reads
            .lock()
            .unwrap()
            .push_back(Ok(Some(offered.clone())));
        let other = stored("other-access", "other-refresh");
        let with_store_client = with_store(
            served_by(
                OpenAIClient::new_chatgpt_oauth(
                    "other-access",
                    Some("other-refresh"),
                    "synthetic-account",
                ),
                &server.url,
            ),
            &store,
        );
        let error = with_store_client
            .refresh_oauth_token_once()
            .await
            .unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        assert_eq!(
            with_store_client.chatgpt_oauth_credentials().await.unwrap(),
            other
        );
        // After another 401, the store's credentials aren't taken either:
        // the 401 stands and the request isn't sent again.
        store.reads.lock().unwrap().push_back(Ok(Some(offered)));
        let error = with_store_client
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                Error::ProviderError {
                    status: Some(401),
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(
            with_store_client.chatgpt_oauth_credentials().await.unwrap(),
            other
        );

        assert_eq!(sent_refresh_tokens(&server), ["synthetic-refresh"]);
        assert_eq!(
            server.request_count(),
            3,
            "one refresh and two Responses requests"
        );
    })
    .await
    .unwrap();
}

/// A refresh waiting out an HTTP 429 finds, on its next attempt, that
/// another client (sharing neither its credentials nor a store) exchanged
/// the same refresh token meanwhile: it isn't sent again. Real time, with a
/// one-second Retry-After.
#[tokio::test]
async fn a_rate_limited_refresh_does_not_resend_a_token_spent_while_it_waited() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let limited = reply("/token", 429, r#"{"error":{"code":"rate_limit_exceeded"}}"#)
            .with_header("Retry-After", "1");
        let mut server = TestServer::start(vec![limited, refreshed()], false).await;
        let waiting = quick_retries(client(&server.url), Duration::from_secs(1))
            .with_oauth_rate_limit_retries(true);
        let waiting = tokio::spawn(async move { waiting.refresh_oauth_token().await });
        server.arrived.recv().await.unwrap();
        client(&server.url)
            .refresh_oauth_token_once()
            .await
            .unwrap();
        let error = waiting.await.unwrap().unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        assert_eq!(
            sent_refresh_tokens(&server),
            ["synthetic-refresh", "synthetic-refresh"]
        );
        assert_eq!(server.request_count(), 2);
    })
    .await
    .unwrap();
}

/// A host refreshed the file's token itself (with
/// `refresh_chatgpt_oauth_token`) and installed the result in a client of
/// the file: the file now holds spent credentials, so the client's are saved
/// over them and refreshed. The spent token is never sent again.
#[tokio::test]
async fn credentials_a_host_refreshed_itself_replace_the_spent_ones_in_the_file() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let server =
            TestServer::start(vec![refreshed(), second_refresh(), refreshed()], false).await;
        let (_dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let client = served_by(file.openai_client().unwrap(), &server.url);
        let http_client = crate::http_client_builder().no_proxy().build().unwrap();
        let token_url = format!("{}/token", server.url);
        let result = crate::refresh_chatgpt_oauth_token(
            &http_client,
            "synthetic-refresh",
            None,
            Some(&token_url),
        )
        .await
        .unwrap();
        let installed = ChatGptOAuthCredentials {
            access_token: result.access_token,
            refresh_token: result.refresh_token,
            account_id: "synthetic-account".into(),
        };
        client
            .update_auth(OpenAIAuth::ChatGptOAuth(installed))
            .await;
        let refreshed_now = client.refresh_oauth_token_once().await.unwrap();
        assert_eq!(refreshed_now.access_token, "second-access");
        assert_eq!(
            file_credentials(&path),
            stored("second-access", "second-refresh")
        );
        assert_eq!(
            sent_refresh_tokens(&server),
            ["synthetic-refresh", "synthetic-new-refresh"]
        );
    })
    .await
    .unwrap();
}

/// A store client refreshes the file's token at one token URL; a client
/// without a store, holding the old credentials and sending to another URL,
/// doesn't send the spent token there.
#[tokio::test]
async fn a_token_spent_at_one_token_url_is_not_sent_to_another() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let first = TestServer::start(vec![refreshed()], false).await;
        let second = TestServer::start(vec![refreshed()], false).await;
        let (_dir, path) = auth_file(&fresh_jwt());
        let file = crate::codex_auth::CodexAuthFile::load(&path).unwrap();
        let with_store = served_by(file.openai_client().unwrap(), &first.url);
        with_store.refresh_oauth_token_once().await.unwrap();
        let old = file.credentials().unwrap();
        let storeless = served_by(
            OpenAIClient::new_chatgpt_oauth(
                &old.access_token,
                old.refresh_token.as_deref(),
                &old.account_id,
            ),
            &second.url,
        );
        let error = storeless.refresh_oauth_token_once().await.unwrap_err();
        assert!(refused_as_spent(&error), "{error}");
        assert_eq!(sent_refresh_tokens(&first), ["synthetic-refresh"]);
        assert_eq!(second.request_count(), 0);
    })
    .await
    .unwrap();
}

/// A refresh the server refuses because the token was reused, expired or
/// revoked records the token: later refreshes fail at once instead of
/// sending it again. Other refusals don't.
#[tokio::test]
async fn a_token_the_server_refused_for_good_is_not_sent_again() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let margin = Duration::from_secs(60);
        for code in [
            "refresh_token_reused",
            "refresh_token_expired",
            "refresh_token_invalidated",
        ] {
            let refusal = reply(
                "/token",
                401,
                &serde_json::json!({"error": {"code": code}}).to_string(),
            );
            let server = TestServer::start(vec![refusal, refreshed()], false).await;
            let refresh_token = format!("refused-{code}");
            let expiring = served_by(
                OpenAIClient::new_chatgpt_oauth(&jwt(1), Some(&refresh_token), "synthetic-account"),
                &server.url,
            );
            let error = expiring
                .refresh_oauth_token_if_expiring(margin)
                .await
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::ProviderError {
                        status: Some(401),
                        ..
                    }
                ),
                "{error}"
            );
            for _ in 0..2 {
                let error = expiring
                    .refresh_oauth_token_if_expiring(margin)
                    .await
                    .unwrap_err();
                assert!(refused_as_spent(&error), "{error}");
            }
            assert_eq!(server.request_count(), 1, "{code}");
        }
        // A refusal that says nothing of the token's fate: it is sent again.
        let busy = reply("/token", 500, r#"{"error":{"code":"server_error"}}"#);
        let server = TestServer::start(vec![busy, refreshed()], false).await;
        let expiring = served_by(
            OpenAIClient::new_chatgpt_oauth(
                &jwt(1),
                Some("refused-server-error"),
                "synthetic-account",
            ),
            &server.url,
        );
        assert!(
            expiring
                .refresh_oauth_token_if_expiring(margin)
                .await
                .is_err()
        );
        expiring
            .refresh_oauth_token_if_expiring(margin)
            .await
            .unwrap();
        assert_eq!(server.request_count(), 2);
    })
    .await
    .unwrap();
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Credentials without a refresh token are used until the server rejects
/// them, instead of failing from the refresh margin before their expiry.
#[tokio::test]
async fn an_access_token_without_a_refresh_token_is_used_until_the_server_rejects_it() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let margin = Duration::from_secs(300);
        let (url, server) = server(vec![expired()]).await;
        // Expires within the refresh margin, but hasn't expired yet.
        let expiring = served_by(
            OpenAIClient::new_chatgpt_oauth(&jwt(now_secs() + 60), None, "synthetic-account"),
            &url,
        );
        assert!(expiring.is_oauth_token_expired(margin).await);
        assert!(
            expiring
                .refresh_oauth_token_if_expiring(margin)
                .await
                .unwrap()
                .is_none()
        );
        // An unreadable expiry is left to the server too.
        let unreadable = served_by(
            OpenAIClient::new_chatgpt_oauth("not-a-jwt", None, "synthetic-account"),
            &url,
        );
        assert!(
            unreadable
                .refresh_oauth_token_if_expiring(margin)
                .await
                .unwrap()
                .is_none()
        );

        // The server's rejection is the clear error.
        let error = expiring
            .generate_text(
                &Conversation::new(Model::openai_gpt_6_astra()),
                &GenerationConfig::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::AuthenticationFailed(_)), "{error}");
        assert!(!error.is_retryable());
        assert!(error.to_string().contains("no refresh token"), "{error}");
        assert_eq!(server.await.unwrap().len(), 1, "no refresh request");

        // So is a token whose expiry has passed: only the server's
        // rejection counts.
        let gone = served_by(
            OpenAIClient::new_chatgpt_oauth(&jwt(now_secs() - 60), None, "synthetic-account"),
            &url,
        );
        assert!(
            gone.refresh_oauth_token_if_expiring(margin)
                .await
                .unwrap()
                .is_none()
        );
    })
    .await
    .unwrap();
}

/// Credentials taken from the environment can't be updated there: the first
/// rotation says so, once.
#[tokio::test]
async fn a_rotation_no_store_keeps_warns_only_when_asked() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![
            rotates_to("access-a", "refresh-a2"),
            rotates_to("access-b", "refresh-b2"),
        ])
        .await;
        let host = served_by(
            OpenAIClient::new_chatgpt_oauth(&fresh_jwt(), Some("refresh-a"), "synthetic-account"),
            &url,
        );
        host.refresh_oauth_token().await.unwrap();
        assert!(!crate::notice::warned("openai-oauth:unsaved-rotation"));
        let from_env = served_by(
            OpenAIClient::new_chatgpt_oauth(&fresh_jwt(), Some("refresh-b"), "synthetic-account")
                .with_unsaved_rotation_warning(),
            &url,
        );
        from_env.refresh_oauth_token().await.unwrap();
        assert!(crate::notice::warned("openai-oauth:unsaved-rotation"));
        assert_eq!(server.await.unwrap().len(), 2);
    })
    .await
    .unwrap();
}
