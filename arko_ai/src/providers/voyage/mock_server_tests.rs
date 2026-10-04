//! Voyage error handling end to end, against a local raw-TCP server (the
//! pattern of `providers/openrouter/mock_server_tests.rs`), through
//! `with_base_url`. Statuses and bodies follow Voyage's documentation
//! (https://docs.voyageai.com/docs/error-codes and the 4XX `detail` body of
//! https://docs.voyageai.com/reference/embeddings-api).

use super::*;
use crate::{
    embedding_provider_identifier::EmbeddingProviderIdentifier, error::ErrorKind, error::RetryClass,
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// What the server does with the one request it receives.
enum Reply {
    /// A complete HTTP response.
    Http {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: String,
    },
    /// Closes the connection without answering.
    Drop,
    /// A 200 whose `Content-Length` promises more than is sent before the
    /// connection closes.
    CutOff { body: String, declared: usize },
}

fn reply(status: u16, body: impl Into<String>) -> Reply {
    Reply::Http {
        status,
        headers: Vec::new(),
        body: body.into(),
    }
}

fn with_header(reply: Reply, name: &'static str, value: &str) -> Reply {
    match reply {
        Reply::Http {
            status,
            mut headers,
            body,
        } => {
            headers.push((name, value.to_string()));
            Reply::Http {
                status,
                headers,
                body,
            }
        }
        other => other,
    }
}

/// Voyage's documented 4XX body: `{"detail": "<the error message>"}`.
fn detail(message: &str) -> String {
    serde_json::json!({ "detail": message }).to_string()
}

/// A successful response with `vectors` at their indexes, as documented.
fn embeddings(vectors: &[(usize, Vec<f32>)]) -> String {
    let data: Vec<_> = vectors
        .iter()
        .map(|(index, embedding)| {
            serde_json::json!({"object": "embedding", "embedding": embedding, "index": index})
        })
        .collect();
    serde_json::json!({
        "object": "list", "data": data, "model": "voyage-code-4",
        "usage": {"total_tokens": 8}
    })
    .to_string()
}

/// Serves `reply` to one request, after reading the whole request.
async fn serve(reply: Reply) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        loop {
            let mut chunk = [0; 8192];
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0, "connection closed before the request was complete");
            bytes.extend_from_slice(&chunk[..n]);
            let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| {
                    line.strip_prefix("content-length:")
                        .map(|s| s.trim().parse().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                break;
            }
        }
        let response = match reply {
            Reply::Drop => return,
            Reply::Http {
                status,
                headers,
                body,
            } => {
                let mut response =
                    format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n");
                for (name, value) in headers {
                    response.push_str(&format!("{name}: {value}\r\n"));
                }
                response.push_str(&format!(
                    "Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                ));
                response
            }
            Reply::CutOff { body, declared } => format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n{body}"
            ),
        };
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    (url, task)
}

fn client(url: &str) -> VoyageClient {
    let mut client = VoyageClient::new("pa-test").with_base_url(url);
    client.http_client = crate::providers::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    client
}

/// Embeds two texts as two-dimensional voyage-code-4 vectors against a server
/// that answers with `reply`.
async fn embed(reply: Reply) -> Result<ArkoEmbeddingResponse, Error> {
    let (url, server) = serve(reply).await;
    let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-code-4")
        .with_output_dimensionality(2);
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        client(&url).embed_texts(
            &["fn a() {}".into(), "fn b() {}".into()],
            &config,
            &EmbeddingModelDetails::voyage_code_4(),
        ),
    )
    .await
    .expect("test timed out");
    server.await.unwrap();
    result
}

async fn failure(reply: Reply) -> Error {
    embed(reply).await.unwrap_err()
}

fn status(error: &Error) -> Option<u16> {
    match error {
        Error::ProviderError { status, .. } => *status,
        _ => None,
    }
}

#[tokio::test]
async fn a_documented_response_comes_back_in_input_order_with_its_usage() {
    let response = embed(reply(
        200,
        embeddings(&[(1, vec![0.5, 0.5]), (0, vec![1.0, 0.0])]),
    ))
    .await
    .unwrap();
    assert_eq!(response.embeddings, vec![vec![1.0, 0.0], vec![0.5, 0.5]]);
    assert_eq!(response.usage.input_tokens, 8);
}

/// 400 Invalid Request, 401 Unauthorized and 403 Forbidden: the request,
/// key or IP address must change. The status and Voyage's `detail` are kept.
#[tokio::test]
async fn documented_client_errors_keep_their_status_and_detail_and_are_permanent() {
    for (code, message, kind) in [
        (400, "A parameter is invalid.", ErrorKind::InvalidRequest),
        (
            401,
            "Provided API key is invalid.",
            ErrorKind::Authentication,
        ),
        (403, "Forbidden.", ErrorKind::Permission),
    ] {
        let error = failure(reply(code, detail(message))).await;
        assert_eq!(status(&error), Some(code), "{error}");
        assert_eq!(error.kind(), kind, "{error}");
        assert_eq!(error.retry_class(), RetryClass::Permanent, "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }
}

/// The 400s that name a limit, with the details Voyage answered with live
/// (2026-10-04), take the kinds arko's own limit checks give them; other
/// 400 details stay invalid requests.
#[tokio::test]
async fn limit_rejections_take_the_kind_of_the_limit() {
    for (message, kind) in [
        (
            "Request to model 'voyage-code-4' failed. The example at index 0 in your batch has too many tokens and does not fit into the model's context window of 32000 tokens. Please lower the number of tokens in the listed example(s) or use truncation.",
            ErrorKind::ContextOverflow,
        ),
        (
            "Request to model 'voyage-code-4' failed. The batch size limit is 1000. Your batch size is 1001. Please resubmit with a smaller batch size.",
            ErrorKind::RequestTooLarge,
        ),
        (
            "Value '100' supplied for argument 'output_dimension' is not valid -- accepted values for 'voyage-code-4' are [256, 512, 1024, 2048].",
            ErrorKind::InvalidRequest,
        ),
    ] {
        let error = failure(reply(400, detail(message))).await;
        assert_eq!(status(&error), Some(400), "{error}");
        assert_eq!(error.kind(), kind, "{error}");
        assert_eq!(error.retry_class(), RetryClass::Permanent, "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }
}

/// A request over the model's total tokens per request: Voyage's
/// `error_code` (undocumented, answered live 2026-10-04) is kept as the code.
#[tokio::test]
async fn a_request_over_the_total_token_limit_keeps_voyages_error_code() {
    let body = r#"{"detail":"Request to model 'voyage-code-4' failed. The max allowed tokens per submitted batch is 320000. Your batch has 330015 tokens after truncation. Please lower the number of tokens in the batch.","error_code":"TOO_MANY_TOKENS_IN_BATCH"}"#;
    let error = failure(reply(400, body)).await;
    let Error::ProviderError { status, code, .. } = &error else {
        panic!("{error:?}")
    };
    assert_eq!(*status, Some(400));
    assert_eq!(code.as_deref(), Some("TOO_MANY_TOKENS_IN_BATCH"));
    assert_eq!(error.kind(), ErrorKind::ContextOverflow, "{error}");
    assert_eq!(error.retry_class(), RetryClass::Permanent, "{error}");
}

/// 429 Rate Limit Exceeded: "Please pace your requests." A `Retry-After`
/// (which Voyage doesn't document) is honoured when one is sent.
#[tokio::test]
async fn rate_limits_are_transient_and_keep_a_retry_after() {
    for (retry_after, expected) in [(None, None), (Some("7"), Some(Duration::from_secs(7)))] {
        let mut answer = reply(429, detail("Rate limit exceeded."));
        if let Some(seconds) = retry_after {
            answer = with_header(answer, "Retry-After", seconds);
        }
        let error = failure(answer).await;
        assert_eq!(status(&error), Some(429), "{error}");
        assert_eq!(error.retry_class(), RetryClass::Transient, "{error}");
        assert!(error.is_rate_limit(), "{error}");
        assert_eq!(error.retry_after(), expected, "{error}");
        assert_eq!(error.kind(), ErrorKind::Unknown, "{error}");
    }
}

/// 500 Server Error: "Retry your request after a brief wait."
#[tokio::test]
async fn a_server_error_is_transient() {
    let error = failure(reply(500, detail("Internal Server Error"))).await;
    assert_eq!(status(&error), Some(500), "{error}");
    assert_eq!(error.retry_class(), RetryClass::Transient, "{error}");
    assert!(
        !error.is_capacity_error() && !error.is_rate_limit(),
        "{error}"
    );
    assert_eq!(error.kind(), ErrorKind::Unknown, "{error}");
}

/// 502, 503 and 504 Service Unavailable: "Our servers are currently
/// experiencing high traffic", an explicit capacity signal, whatever the
/// body (Voyage documents none for 5XX). The status, the body and a delay
/// hint are kept.
#[tokio::test]
async fn service_unavailable_statuses_are_capacity_signals() {
    for (code, body) in [
        (502, "<html><body>502 Bad Gateway</body></html>".to_string()),
        (503, detail("Service Unavailable")),
        (504, String::new()),
    ] {
        let error = failure(reply(code, body.clone())).await;
        assert_eq!(status(&error), Some(code), "{error}");
        assert_eq!(error.retry_class(), RetryClass::Capacity, "{error}");
        assert!(
            error.is_capacity_error() && !error.is_rate_limit(),
            "{error}"
        );
        assert_eq!(error.kind(), ErrorKind::Unknown, "{error}");
        assert!(error.to_string().contains(&body), "{error}");
    }
    let error = failure(with_header(reply(503, ""), "Retry-After", "3")).await;
    assert_eq!(error.retry_class(), RetryClass::Capacity, "{error}");
    assert_eq!(error.retry_after(), Some(Duration::from_secs(3)));
    // Only Voyage's documented meaning makes a bare 503 a capacity signal.
    let generic = Error::provider_error("Other", Some(503), None, "");
    assert_eq!(generic.retry_class(), RetryClass::Transient);
}

/// A connection closed before any response: no status, and whether Voyage
/// processed the request is unknown.
#[tokio::test]
async fn a_dropped_connection_is_a_transport_error_without_a_status() {
    let error = failure(Reply::Drop).await;
    assert!(
        matches!(&error, Error::HttpError(error) if error.status().is_none()),
        "{error:?}"
    );
    assert_eq!(error.retry_class(), RetryClass::Transient, "{error}");
    assert_eq!(error.kind(), ErrorKind::Unknown, "{error}");
}

/// A 200 whose body was cut off in transfer: Voyage answered, so the
/// request was processed (and may have been billed). It stays a transport
/// error, and permanent, so no caller resends it on its own.
#[tokio::test]
async fn a_success_cut_off_in_transfer_is_a_permanent_transport_error() {
    let body = embeddings(&[(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])]);
    let error = failure(Reply::CutOff {
        body: body[..40].to_string(),
        declared: body.len(),
    })
    .await;
    assert!(matches!(error, Error::HttpError(_)), "{error:?}");
    assert_eq!(error.retry_class(), RetryClass::Permanent, "{error}");
    assert_eq!(error.kind(), ErrorKind::Unknown, "{error}");
}

/// A 200 that isn't the documented embeddings object (it doesn't parse, or
/// its `index`es or vector lengths don't match the request) is Voyage's
/// fault, not the caller's invalid request. It is permanent: the request
/// was answered and may have been billed.
#[tokio::test]
async fn a_malformed_success_is_not_an_invalid_request() {
    for (body, says) in [
        ("<html>oops</html>".to_string(), "expected value"),
        (
            r#"{"object":"list","data":[]}"#.to_string(),
            "missing field `usage`",
        ),
        (embeddings(&[(0, vec![1.0, 0.0])]), "index"),
        (
            embeddings(&[(0, vec![1.0, 0.0]), (0, vec![1.0, 0.0])]),
            "index",
        ),
        (embeddings(&[(0, vec![1.0, 0.0]), (1, vec![1.0])]), "vector"),
    ] {
        let error = failure(reply(200, body)).await;
        assert!(matches!(error, Error::JsonError(_)), "{error:?}");
        assert_eq!(error.kind(), ErrorKind::Unknown, "{error}");
        assert_eq!(error.retry_class(), RetryClass::Permanent, "{error}");
        let message = error.to_string();
        assert!(
            message.contains("Voyage") && message.contains(says),
            "{message}"
        );
    }
}
