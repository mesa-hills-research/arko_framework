//! End-to-end tests against a local raw-TCP server (the pattern of
//! `providers/openai/oauth_recovery_tests.rs`), through `with_base_url`.

use super::*;
use crate::{client::LLMClient, error::RetryClass, stop::RefusalSource, tool::ToolDefinition};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: String,
}

fn reply(status: u16, body: impl Into<String>) -> Reply {
    Reply {
        status,
        headers: Vec::new(),
        body: body.into(),
    }
}

fn with_retry_after(mut reply: Reply, seconds: u64) -> Reply {
    reply.headers.push(("Retry-After", seconds.to_string()));
    reply
}

/// One received request: the request line and headers (lowercased), and the
/// JSON body (`Null` for none).
#[derive(Debug)]
struct Received {
    head: String,
    body: Value,
}

impl Received {
    fn header(&self, name: &str) -> Option<&str> {
        let prefix = format!("{}:", name.to_ascii_lowercase());
        self.head
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .map(str::trim)
    }
}

async fn serve(replies: Vec<Reply>) -> (String, tokio::task::JoinHandle<Vec<Received>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/api/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut received = Vec::new();
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (head, body) = loop {
                let mut chunk = [0; 8192];
                let n = socket.read(&mut chunk).await.unwrap();
                assert!(n > 0, "connection closed before the request was complete");
                bytes.extend_from_slice(&chunk[..n]);
                let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let head = String::from_utf8(bytes[..end].to_vec())
                    .unwrap()
                    .to_ascii_lowercase();
                let length: usize = head
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length:")
                            .map(|s| s.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                if bytes.len() >= end + 4 + length {
                    let body = if length == 0 {
                        Value::Null
                    } else {
                        serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap()
                    };
                    break (head, body);
                }
            };
            received.push(Received { head, body });
            let content_type = if reply.body.starts_with("data:") || reply.body.starts_with(':') {
                "text/event-stream"
            } else {
                "application/json"
            };
            let mut response = format!(
                "HTTP/1.1 {} Test\r\nContent-Type: {content_type}\r\n",
                reply.status
            );
            for (name, value) in &reply.headers {
                response.push_str(&format!("{name}: {value}\r\n"));
            }
            response.push_str(&format!(
                "Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.body.len(),
                reply.body
            ));
            socket.write_all(response.as_bytes()).await.unwrap();
        }
        received
    });
    (url, task)
}

fn client(url: &str) -> OpenRouterClient {
    let mut client = OpenRouterClient::new("sk-or-test").with_base_url(url);
    client.http_client = crate::providers::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    client
}

fn conversation(slug: &str) -> Conversation {
    let mut conversation = Conversation::new(Model::openrouter(slug));
    conversation.add_user_message("hi");
    conversation
}

fn completion(text: &str) -> String {
    json!({
        "id": "gen-1", "object": "chat.completion", "created": 1, "model": "anthropic/claude-sonnet-5", "provider": "Anthropic",
        "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "end_turn",
            "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6, "cost": 0.00002}
    })
    .to_string()
}

async fn timed<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("test timed out")
}

#[tokio::test]
async fn requests_carry_only_the_key_and_no_attribution_by_default() {
    timed(async {
        let (url, server) = serve(vec![reply(200, completion("ok"))]).await;
        let client = client(&url);
        let response = client
            .generate_text(
                &conversation("anthropic/claude-sonnet-5"),
                &GenerationConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(response.text(), "ok");
        assert_eq!(response.usage.reported_cost, Some(0.00002));
        let requests = server.await.unwrap();
        let request = &requests[0];
        assert!(
            request.head.starts_with("post /api/v1/chat/completions "),
            "{}",
            request.head
        );
        assert_eq!(request.header("authorization"), Some("bearer sk-or-test"));
        for header in [
            "http-referer",
            "x-openrouter-title",
            "x-title",
            "x-openrouter-categories",
            "x-openrouter-metadata",
            "x-anthropic-beta",
        ] {
            assert!(
                request.header(header).is_none(),
                "{header} must not be sent by default"
            );
        }
        assert_eq!(request.body["model"], "anthropic/claude-sonnet-5");
        assert!(
            request.body["session_id"]
                .as_str()
                .and_then(|key| uuid::Uuid::parse_str(key).ok())
                .is_some()
        );
        assert!(request.body.get("stream").is_none());
    })
    .await;
}

/// A chat request goes to the base URL chosen for its model, and to the
/// client's own base URL when none is chosen.
#[tokio::test]
async fn chat_requests_go_to_the_base_url_chosen_for_their_model() {
    timed(async {
        let (url, server) = serve(vec![
            reply(200, completion("eu")),
            reply(200, completion("us")),
        ])
        .await;
        let eu = format!("{url}/eu/");
        let client = client(&url)
            .with_base_url_for(move |slug| (slug == "mistralai/mistral-large").then(|| eu.clone()));
        for slug in ["mistralai/mistral-large", "anthropic/claude-sonnet-5"] {
            client
                .generate_text(&conversation(slug), &GenerationConfig::default())
                .await
                .unwrap();
        }
        let requests = server.await.unwrap();
        assert!(
            requests[0]
                .head
                .starts_with("post /api/v1/eu/chat/completions "),
            "{}",
            requests[0].head
        );
        assert!(
            requests[1]
                .head
                .starts_with("post /api/v1/chat/completions "),
            "{}",
            requests[1].head
        );
    })
    .await;
}

#[tokio::test]
async fn attribution_and_beta_headers_are_sent_when_configured() {
    timed(async {
        let (url, server) = serve(vec![reply(200, completion("ok"))]).await;
        let client = client(&url).with_app_attribution(
            AppAttribution::new("https://example.app")
                .with_title("Example App")
                .with_categories(["cli-agent", "programming-app"])
                .hidden(),
        );
        let tools = crate::tool::ToolConfig::new(vec![
            ToolDefinition::new("lookup", "", json!({"type": "object"})).with_strict(true),
        ]);
        client
            .generate_text(
                &conversation("anthropic/claude-sonnet-5"),
                &GenerationConfig::default().with_tool_config(tools),
            )
            .await
            .unwrap();
        let request = &server.await.unwrap()[0];
        assert_eq!(request.header("http-referer"), Some("https://example.app"));
        assert_eq!(request.header("x-openrouter-title"), Some("example app"));
        assert_eq!(
            request.header("x-openrouter-categories"),
            Some("cli-agent,programming-app")
        );
        assert_eq!(
            request.header("x-openrouter-app-visibility"),
            Some("hidden")
        );
        assert_eq!(
            request.header("x-anthropic-beta"),
            Some("structured-outputs-2025-11-13")
        );
        assert_eq!(
            request.body["provider"],
            json!({"sort": "price"}),
            "Auto Exacto policy on a Claude tool request"
        );
    })
    .await;
}

#[tokio::test]
async fn streams_through_the_llm_client_end_once() {
    timed(async {
        let body = format!(
            ": OPENROUTER PROCESSING\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"id": "gen-2", "model": "openai/gpt-6-sol", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": null}]}),
            json!({"id": "gen-2", "model": "openai/gpt-6-sol", "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop", "native_finish_reason": "completed"}]}),
            json!({"id": "gen-2", "model": "openai/gpt-6-sol", "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop", "native_finish_reason": "completed"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "cost": 0.00001}}),
        );
        let (url, server) = serve(vec![reply(200, body)]).await;
        let llm = LLMClient::new().with_openrouter_client(client(&url));
        let mut stream = llm
            .generate_stream(&conversation("openai/gpt-6-sol"), GenerationConfig::default())
            .await
            .unwrap();
        let mut ends = Vec::new();
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::TextChunk(chunk) => text.push_str(&chunk),
                StreamEvent::End(response) => ends.push(response),
                _ => {}
            }
        }
        assert_eq!(text, "Hi");
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].model_call_usage.stop_reason.as_deref(), Some("stop"));
        let request = &server.await.unwrap()[0];
        assert_eq!(request.body["stream"], true);
        for field in ["stream_options", "usage", "transforms"] {
            assert!(request.body.get(field).is_none(), "{field}");
        }
    })
    .await;
}

#[tokio::test]
async fn http_errors_keep_status_delays_and_classification() {
    timed(async {
        let in_flight = json!({"error": {"code": 402, "message": "in-flight budget",
            "metadata": {"reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget"}}});
        let credits = json!({"error": {"code": 402, "message": "Insufficient credits",
            "metadata": {"limit_source": "openrouter_credits"}}});
        let refusal = json!({"error": {"code": 403, "message": "This content was flagged for possible cybersecurity risk.",
            "metadata": {"error_type": "refusal", "provider_code": "cyber_policy"}}});
        let limited = json!({"error": {"code": 429, "message": "Rate limit exceeded",
            "metadata": {"error_type": "rate_limit_exceeded"}}});
        let unroutable = json!({"error": {"code": 503, "message": "No available model provider that meets your routing requirements"}});
        let overloaded = json!({"error": {"code": 503, "message": "Overloaded", "metadata": {"error_type": "provider_overloaded"}}});
        let cases = [
            (with_retry_after(reply(402, in_flight.to_string()), 60), RetryClass::Transient, Some(60)),
            (with_retry_after(reply(402, credits.to_string()), 60), RetryClass::Permanent, None),
            (with_retry_after(reply(429, limited.to_string()), 60), RetryClass::Transient, Some(60)),
            // Unroutable is permanent unless every provider asked to wait.
            (reply(503, unroutable.to_string()), RetryClass::Permanent, None),
            (with_retry_after(reply(503, unroutable.to_string()), 60), RetryClass::Transient, Some(60)),
            (with_retry_after(reply(503, overloaded.to_string()), 30), RetryClass::Capacity, Some(30)),
            (reply(403, refusal.to_string()), RetryClass::Permanent, None),
        ];
        for (reply, class, delay) in cases {
            let (url, server) = serve(vec![reply]).await;
            let error = client(&url)
                .generate_text(&conversation("openai/gpt-6-sol"), &GenerationConfig::default())
                .await
                .unwrap_err();
            assert_eq!(error.retry_class(), class, "{error}");
            assert_eq!(error.retry_after(), delay.map(Duration::from_secs), "{error}");
            server.await.unwrap();
            if let Error::Refusal(refusal) = &error {
                assert_eq!(refusal.source, RefusalSource::SafetyClassifier);
                assert_eq!(refusal.category.as_deref(), Some("cyber_policy"));
            }
        }
    })
    .await;
}

#[tokio::test]
async fn a_200_response_with_an_error_body_is_an_error_for_text_and_stream_calls() {
    timed(async {
        let error_body = json!({"id": "gen-3", "error": {"code": 502, "message": "Provider returned error",
            "metadata": {"error_type": "provider_unavailable", "provider_name": "Anthropic"}}});
        let (url, server) = serve(vec![
            reply(200, error_body.to_string()),
            reply(200, format!("data: {}\n\n", error_body)),
        ])
        .await;
        let client = client(&url);
        let error = client
            .generate_text(&conversation("anthropic/claude-sonnet-5"), &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::ProviderError { status: Some(502), .. }), "{error}");
        assert!(error.is_retryable());
        let mut stream = client
            .generate_stream(&conversation("anthropic/claude-sonnet-5"), &GenerationConfig::default())
            .await
            .unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, Error::ProviderError { status: Some(502), .. }), "{error}");
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    })
    .await;
}

#[tokio::test]
async fn catalogs_are_fetched_only_on_request_and_validate_later_calls() {
    timed(async {
        let models = json!({"data": [{
            "id": "openai/gpt-6-sol", "canonical_slug": "openai/gpt-6-sol-20260903", "context_length": 1050000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000005", "completion": "0.00003", "input_cache_read": "0.0000005", "input_cache_write": "0.00000625"},
            "top_provider": {"context_length": 1050000, "max_completion_tokens": 128000},
            "supported_parameters": ["max_tokens", "reasoning", "tools", "tool_choice"],
            "reasoning": {"mandatory": false, "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"], "default_effort": "medium"}
        }]});
        let (url, server) = serve(vec![reply(200, models.to_string()), reply(200, models.to_string())]).await;
        let client = client(&url);
        assert!(client.catalog().is_none(), "nothing is fetched implicitly");
        let catalog = client.fetch_models().await.unwrap();
        assert_eq!(catalog.source(), CatalogSource::Public);
        assert_eq!(client.model("openai/gpt-6-sol").unwrap().max_output_tokens, 128_000);
        let catalog = client.clone().fetch_user_models().await.unwrap();
        assert_eq!(catalog.source(), CatalogSource::User);
        assert_eq!(client.catalog().unwrap().source(), CatalogSource::User, "clones share the catalog");
        let requests = server.await.unwrap();
        assert!(requests[0].head.starts_with("get /api/v1/models "), "{}", requests[0].head);
        assert!(requests[0].header("authorization").is_none(), "the public list needs no key");
        assert!(requests[1].head.starts_with("get /api/v1/models/user "), "{}", requests[1].head);
        assert_eq!(requests[1].header("authorization"), Some("bearer sk-or-test"));

        // Validation now uses the catalog: no request is sent for these.
        let unknown = client
            .generate_text(&conversation("openai/gpt-7"), &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(matches!(unknown, Error::ModelNotSupported(_)), "{unknown}");
        let temperature = client
            .generate_text(&conversation("openai/gpt-6-sol"), &GenerationConfig::default().with_temperature(0.5))
            .await
            .unwrap_err();
        assert!(temperature.to_string().contains("temperature"), "{temperature}");

        let (url, server) = serve(vec![reply(401, r#"{"error":{"code":401,"message":"No auth credentials found"}}"#)]).await;
        let error = self::client(&url).fetch_user_models().await.unwrap_err();
        assert!(matches!(error, Error::ProviderError { status: Some(401), .. }));
        server.await.unwrap();
    })
    .await;
}
