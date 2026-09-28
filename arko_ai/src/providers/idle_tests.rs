//! The opt-in stream idle timeout against a local server that streams on a
//! real clock (a paused clock doesn't mix well with real sockets).

use crate::{
    client::LLMClient,
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    model::Model,
    providers::{
        StreamEvent, anthropic::AnthropicClient, google::GoogleClient, openai::OpenAIClient,
        openrouter::OpenRouterClient,
    },
};
use futures::StreamExt;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// What the server does after the response head.
enum Step {
    Send(String),
    SendBytes(Vec<u8>),
    Pause(Duration),
    /// Waits for the test's signal.
    Hold(tokio::sync::oneshot::Receiver<()>),
}

/// Serves one request: a 200 response head for an event stream with no
/// length (the body ends when the connection closes), then `steps`. The
/// connection is closed only when the steps are done.
async fn serve(steps: Vec<Step>, content_type: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        loop {
            let mut chunk = [0; 8192];
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0, "connection closed before the request was complete");
            request.extend_from_slice(&chunk[..n]);
            let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map_or(0, |value| value.trim().parse().unwrap());
            if request.len() >= end + 4 + length {
                break;
            }
        }
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        for step in steps {
            match step {
                Step::Send(data) => {
                    if socket.write_all(data.as_bytes()).await.is_err() {
                        return;
                    }
                }
                Step::SendBytes(data) => {
                    if socket.write_all(&data).await.is_err() {
                        return;
                    }
                }
                Step::Pause(duration) => tokio::time::sleep(duration).await,
                Step::Hold(signal) => {
                    let _ = signal.await;
                }
            }
        }
    });
    url
}

fn completed() -> String {
    let response = serde_json::json!({
        "id": "resp_idle", "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-6-astra", "output": [{"type": "message", "id": "msg_idle", "status": "completed",
        "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    });
    format!(
        "data: {}\n\n",
        serde_json::json!({"type": "response.completed", "response": response})
    )
}

fn conversation(model: Model) -> Conversation {
    let mut conversation = Conversation::new(model);
    conversation.add_user_message("Hi.");
    conversation
}

fn openai(url: &str) -> LLMClient {
    LLMClient::new().with_openai_client(OpenAIClient::new("test").with_base_url(url))
}

async fn last_event(client: &LLMClient, model: Model) -> Result<StreamEvent, Error> {
    let mut stream = client
        .generate_stream(&conversation(model), GenerationConfig::default())
        .await?;
    let mut last = None;
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => last = Some(event),
            Err(error) => return Err(error),
        }
    }
    Ok(last.expect("the stream ended without an event"))
}

/// Off by default: a stream that goes quiet is waited for.
#[tokio::test]
async fn without_a_timeout_a_quiet_stream_is_waited_for() {
    let (release, released) = tokio::sync::oneshot::channel();
    let url = serve(
        vec![
            Step::Send(": still thinking\n\n".into()),
            Step::Hold(released),
            Step::Send(completed()),
        ],
        "text/event-stream",
    )
    .await;
    let client = openai(&url);
    let mut stream = client
        .generate_stream(
            &conversation(Model::openai_gpt_6_astra()),
            GenerationConfig::default(),
        )
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(700), stream.next())
            .await
            .is_err(),
        "still waiting"
    );
    release.send(()).unwrap();
    let event = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(event, StreamEvent::End(_)), "{event:?}");
}

/// Set on the `LLMClient`, the timeout reaches every provider: a stream
/// quiet for longer ends with a transient error naming the timeout.
#[tokio::test]
async fn a_stream_quiet_past_the_timeout_fails_with_a_transient_error_on_every_provider() {
    let timeout = Duration::from_millis(300);
    let mut holds = Vec::new();
    let mut quiet = async || {
        let (hold, held) = tokio::sync::oneshot::channel::<()>();
        holds.push(hold);
        serve(
            vec![Step::Send(": ping\n\n".into()), Step::Hold(held)],
            "text/event-stream",
        )
        .await
    };
    let providers = [
        (
            LLMClient::new()
                .with_openai_client(OpenAIClient::new("test").with_base_url(quiet().await)),
            Model::openai_gpt_6_astra(),
        ),
        (
            LLMClient::new()
                .with_anthropic_client(AnthropicClient::new("test").with_base_url(quiet().await)),
            Model::anthropic_claude_opus_5_5(),
        ),
        (
            LLMClient::new()
                .with_google_client(GoogleClient::new("test").with_base_url(quiet().await)),
            Model::google_gemini_3_5_flash(),
        ),
        (
            LLMClient::new()
                .with_openrouter_client(OpenRouterClient::new("test").with_base_url(quiet().await)),
            Model::openrouter("anthropic/claude-opus-5.5"),
        ),
    ];
    for (client, model) in providers {
        // Set after the provider was configured.
        let client = client.with_stream_idle_timeout(timeout);
        let started = Instant::now();
        let error =
            tokio::time::timeout(Duration::from_secs(5), last_event(&client, model.clone()))
                .await
                .unwrap()
                .unwrap_err();
        assert!(
            matches!(error, Error::StreamTruncated(_)),
            "{}: {error}",
            model.id
        );
        assert!(error.is_retryable(), "{}", model.id);
        assert!(error.to_string().contains("idle timeout"), "{error}");
        assert!(started.elapsed() >= timeout, "{}", model.id);
    }
    drop(holds);
}

/// The timeout is the longest wait between data: a response that keeps
/// sending (here keep-alive comments) runs longer than the timeout.
#[tokio::test]
async fn data_arriving_in_time_restarts_the_timeout() {
    let mut steps = Vec::new();
    for _ in 0..10 {
        steps.push(Step::Pause(Duration::from_millis(100)));
        steps.push(Step::Send(": keep-alive\n\n".into()));
    }
    steps.push(Step::Send(completed()));
    let url = serve(steps, "text/event-stream").await;
    let started = Instant::now();
    let client = openai(&url).with_stream_idle_timeout(Duration::from_millis(400));
    let event = last_event(&client, Model::openai_gpt_6_astra())
        .await
        .unwrap();
    assert!(matches!(event, StreamEvent::End(_)), "{event:?}");
    assert!(started.elapsed() >= Duration::from_millis(900));
}

/// A response that doesn't stream is unaffected, however long its body takes.
#[tokio::test]
async fn responses_that_do_not_stream_are_unaffected() {
    let body = completed()
        .strip_prefix("data: ")
        .map(|event| {
            let event: serde_json::Value = serde_json::from_str(event.trim()).unwrap();
            event["response"].to_string()
        })
        .unwrap();
    let url = serve(
        vec![Step::Pause(Duration::from_millis(700)), Step::Send(body)],
        "application/json",
    )
    .await;
    let client = openai(&url).with_stream_idle_timeout(Duration::from_millis(200));
    let response = client
        .generate_text(
            &conversation(Model::openai_gpt_6_astra()),
            GenerationConfig::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "ok");
}

/// A cut that splits an event (here inside a multi-byte character) is the
/// same transient idle error, not a malformed-stream error.
#[tokio::test]
async fn a_cut_inside_an_event_is_the_idle_error_too() {
    let (hold, held) = tokio::sync::oneshot::channel::<()>();
    let url = serve(
        vec![
            Step::SendBytes(
                b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"\xe2\x82".to_vec(),
            ),
            Step::Hold(held),
        ],
        "text/event-stream",
    )
    .await;
    let client = openai(&url).with_stream_idle_timeout(Duration::from_millis(300));
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        last_event(&client, Model::openai_gpt_6_astra()),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(matches!(error, Error::StreamTruncated(_)), "{error}");
    assert!(error.to_string().contains("idle timeout"), "{error}");
    drop(hold);
}
