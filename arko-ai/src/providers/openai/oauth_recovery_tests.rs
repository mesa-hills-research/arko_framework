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

pub(super) async fn server(replies: Vec<Reply>) -> (String, tokio::task::JoinHandle<Vec<(String, Value)>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (headers, body) = loop {
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
                        let body =
                            serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                        break (headers, body);
                    }
                }
            };
            assert!(headers.starts_with(&format!("POST {} ", reply.path)));
            requests.push((headers, body));
            let content_type = if reply.body.starts_with("data:") {
                "text/event-stream"
            } else {
                "application/json"
            };
            let extra: String = reply
                .headers
                .iter()
                .map(|(name, value)| format!("{name}: {value}\r\n"))
                .collect();
            let response = format!(
                "HTTP/1.1 {} Test\r\nContent-Type: {}\r\nRetry-After: 60\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.status,
                content_type,
                extra,
                reply.body.len(),
                reply.body,
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
        requests
    });
    (url, task)
}

pub(super) fn client(url: &str) -> OpenAIClient {
    let claims = URL_SAFE_NO_PAD.encode(br#"{"exp":4102444800}"#);
    let mut client = OpenAIClient::new_chatgpt_oauth(
        &format!("header.{claims}.signature"),
        Some("synthetic-refresh"),
        "synthetic-account",
    )
    .with_base_url(url)
    .with_oauth_token_url(format!("{url}/token"));
    client.http_client = crate::providers::http_client_builder().no_proxy().build().unwrap();
    client
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
