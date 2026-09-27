use super::*;
use crate::model::Model;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub(super) struct Reply {
    path: &'static str,
    status: u16,
    body: String,
}

pub(super) fn reply(path: &'static str, status: u16, body: &str) -> Reply {
    Reply {
        path,
        status,
        body: body.into(),
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
            let response = format!(
                "HTTP/1.1 {} Test\r\nContent-Type: {}\r\nRetry-After: 60\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.status,
                content_type,
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
            let completed = completed_response();
            let body = if streaming {
                format!(
                    "data: {}\n\n",
                    serde_json::json!({"type":"response.completed", "response":completed})
                )
            } else {
                completed.to_string()
            };
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
    assert!(debug.contains("synthetic-account"), "{debug}");
    assert!(debug.contains("<redacted>"), "{debug}");
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
