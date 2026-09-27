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
    if !reply.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("retry-after")) {
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

pub(super) async fn server(replies: Vec<Reply>) -> (String, tokio::task::JoinHandle<Vec<(String, Value)>>) {
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
        let replies = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(replies)));
        let recorded = requests.clone();
        let permits = gate.clone();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let (recorded, arrivals, permits, replies) =
                    (recorded.clone(), arrivals.clone(), permits.clone(), replies.clone());
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
    client.http_client = crate::providers::http_client_builder().no_proxy().build().unwrap();
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
        OpenAIClient::new_chatgpt_oauth(&fresh_jwt(), Some("synthetic-refresh"), "synthetic-account"),
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
            let client = client(&url);
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
            assert_eq!(calls[1].1["refresh_token"], "synthetic-refresh");
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
    reads: std::sync::Mutex<std::collections::VecDeque<Result<Option<ChatGptOAuthCredentials>, Error>>>,
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
        let (url, server) = server(vec![expired(), reply("/responses", 200, &completed_stream())]).await;
        let store = Arc::new(ScriptedStore::default());
        store
            .reads
            .lock()
            .unwrap()
            .push_back(Ok(Some(stored("access-from-codex", "refresh-from-codex"))));
        let client = with_store(client(&url), &store);
        client
            .generate_text(&Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default())
            .await
            .unwrap();
        let calls = server.await.unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls[1].0.to_lowercase().contains("authorization: bearer access-from-codex"));
        assert_eq!(
            client.chatgpt_oauth_credentials().await.unwrap(),
            stored("access-from-codex", "refresh-from-codex")
        );
        assert!(store.saved.lock().unwrap().is_empty(), "nothing was refreshed");
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn an_unreadable_store_or_another_account_stops_the_refresh() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for read in [
            Err(Error::AuthenticationFailed("the store holds another account".into())),
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
                .generate_text(&Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::AuthenticationFailed(_)), "{error}");
            assert!(!error.is_retryable());
            assert_eq!(server.await.unwrap().len(), 1, "no refresh request");
            let credentials = client.chatgpt_oauth_credentials().await.unwrap();
            assert_eq!(credentials.account_id, "synthetic-account");
            assert_eq!(credentials.refresh_token.as_deref(), Some("synthetic-refresh"));
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

/// Each rate-limited retry re-reads the store (Codex may refresh while this
/// client waits), and the retries stop at the cap.
#[tokio::test(start_paused = true)]
async fn rate_limited_refreshes_retry_a_bounded_number_of_times() {
    let limited = || reply("/token", 429, r#"{"error":{"code":"rate_limit_exceeded"}}"#);
    let (url, server) =
        server((0..=OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES).map(|_| limited()).collect()).await;
    let store = Arc::new(ScriptedStore::default());
    let client = with_store(client(&url), &store).with_oauth_rate_limit_retries(true);
    let started = tokio::time::Instant::now();
    let error = client.refresh_oauth_token().await.unwrap_err();
    assert!(matches!(error, Error::ProviderError { status: Some(429), .. }), "{error}");
    let requests = server.await.unwrap().len() as u32;
    assert_eq!(requests, OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES + 1);
    // One read before each request, plus one after the last failure.
    assert_eq!(store.seen.lock().unwrap().len() as u32, requests + 1);
    // Every wait honored the server's Retry-After: 60.
    assert!(started.elapsed() >= Duration::from_secs(60) * OAUTH_REFRESH_MAX_RATE_LIMIT_RETRIES);
}

fn permission_denied() -> Reply {
    reply("/responses", 401, r#"{"error":{"code":"permission_denied"}}"#)
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
                .generate_text(&Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default())
                .await;
            let calls = server.await.unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].1, calls[1].1, "the same request is sent again");
            assert!(calls[1].0.to_lowercase().contains("authorization: bearer access-from-codex"));
            if second_reply_ok {
                result.unwrap();
            } else {
                // A second rejection is returned; nothing loops.
                let error = result.unwrap_err();
                assert!(matches!(error, Error::ProviderError { status: Some(401), .. }), "{error}");
            }
            assert_eq!(store.seen.lock().unwrap().len(), 1, "read once");
            assert!(store.saved.lock().unwrap().is_empty(), "nothing was refreshed");
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
    crate::codex_auth::CodexAuthFile::load(path).unwrap().credentials().unwrap()
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
        // The refresh token rotates once the server answers, whether or not
        // the caller is still there.
        server.gate.add_permits(1);
        // The refresh holds the credentials until the new tokens are
        // installed and saved.
        let new = stored("synthetic-new-access", "synthetic-new-refresh");
        assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), new);
        assert_eq!(file_credentials(&path), new);
        assert_eq!(server.request_count(), 1);
    })
    .await
    .unwrap();
}

/// The 429 asks for 30 days; one wait honors 15 minutes of that, with no
/// lock held.
#[tokio::test(start_paused = true)]
async fn a_rate_limited_refresh_waits_without_holding_the_credentials() {
    let asked = crate::retry::MAX_SERVER_DELAY_HINT;
    // The paused clock also jumps ahead while the test server's replies are
    // on their way (to the HTTP client's idle-pool timers), so the bounds on
    // virtual time are loose; they still tell 15 minutes from 30 days.
    tokio::time::timeout(asked * 2 / 3, async {
        let limited = reply("/token", 429, r#"{"error":{"code":"rate_limit_exceeded"}}"#)
            .with_header("Retry-After", &asked.as_secs().to_string());
        let mut server = TestServer::start(vec![limited, refreshed()], false).await;
        let client = client(&server.url).with_oauth_rate_limit_retries(true);
        let waiting = tokio::spawn({
            let client = client.clone();
            async move {
                let started = tokio::time::Instant::now();
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
        assert!(waited >= Duration::from_secs(900) && waited < asked / 2, "{waited:?}");
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
        assert_eq!(first.await.unwrap().unwrap().access_token, "synthetic-new-access");
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
            let expired = OpenAIClient::new_chatgpt_oauth(&jwt(1), Some("synthetic-refresh"), "synthetic-account");
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
            store.reads.lock().unwrap().push_back(Ok(Some(from_codex.clone())));
            let expired = OpenAIClient::new_chatgpt_oauth(&jwt(1), Some("synthetic-refresh"), "synthetic-account");
            let client = with_store(served_by(expired, &server.url), &store);
            let result = client.refresh_oauth_token_if_expiring(margin).await.unwrap().unwrap();
            let saved = store.saved.lock().unwrap().clone();
            if refreshes {
                assert_eq!(result.access_token, "synthetic-new-access");
                assert_eq!(server.request_count(), 1);
                // The stored refresh token was used, not the client's old one.
                assert_eq!(server.requests.lock().unwrap()[0].1["refresh_token"], "refresh-from-codex");
                assert_eq!(saved, [(stored("synthetic-new-access", "synthetic-new-refresh"), None)]);
            } else {
                assert_eq!(result.access_token, stored_access);
                assert_eq!(server.request_count(), 0);
                assert!(saved.is_empty());
                assert_eq!(client.chatgpt_oauth_credentials().await.unwrap(), from_codex);
            }
        }
    })
    .await
    .unwrap();
}
