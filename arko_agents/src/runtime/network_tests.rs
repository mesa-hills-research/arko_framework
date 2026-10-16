//! The runtime against a local OpenAI-compatible server on a real clock (a
//! paused clock doesn't mix well with real sockets): stream idle timeouts
//! and failed requests.

use super::*;
use crate::agent_config::{AgentLLMConfig, LLMRetryConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// How the server answers one request.
enum Answer {
    /// A response head and a comment, then nothing while the connection
    /// stays open.
    Stall,
    /// A response that calls `complete_task`.
    Complete,
    /// A whole (not streamed) response of text, with cache reads.
    Text,
    /// A 400 refusing the request's output limit.
    RefuseCap,
}

/// Serves `answers` to consecutive requests; returns the base URL and the
/// request bodies received.
async fn serve(answers: Vec<Answer>) -> (String, Arc<std::sync::Mutex<Vec<Value>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
    let received = bodies.clone();
    tokio::spawn(async move {
        let mut stalled = Vec::new();
        for answer in answers {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let body = loop {
                let mut chunk = [0; 16384];
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
                    break request[end + 4..end + 4 + length].to_vec();
                }
            };
            received
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&body).unwrap());
            if let Answer::RefuseCap = answer {
                let error = serde_json::json!({"error": {
                    "message": "Invalid 'max_output_tokens': integer below minimum value.",
                    "type": "invalid_request_error", "param": "max_output_tokens",
                    "code": "integer_below_min_value",
                }})
                .to_string();
                let reply = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{error}",
                    error.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
                continue;
            }
            if let Answer::Text = answer {
                let message = serde_json::json!({
                    "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                });
                let response = serde_json::json!({
                    "id": "resp_1", "object": "response", "created_at": 0,
                    "status": "completed", "model": "gpt-5.6-luna", "output": [message],
                    "usage": {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1100},
                              "output_tokens": 1},
                })
                .to_string();
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",
                    response.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
                continue;
            }
            let head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
            socket.write_all(head.as_bytes()).await.unwrap();
            match answer {
                Answer::Stall => {
                    socket.write_all(b": thinking\n\n").await.unwrap();
                    stalled.push(socket);
                }
                Answer::Text | Answer::RefuseCap => unreachable!("answered above"),
                Answer::Complete => {
                    let call = serde_json::json!({
                        "type": "function_call", "id": "fc_1", "call_id": "call_1",
                        "name": "complete_task", "arguments": "{\"summary\":\"done\"}",
                        "status": "completed"
                    });
                    let response = serde_json::json!({
                        "id": "resp_1", "object": "response", "created_at": 0,
                        "status": "completed", "model": "gpt-5.6-luna", "output": [call],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    });
                    let events = format!(
                        "data: {}\n\ndata: {}\n\n",
                        serde_json::json!({"type": "response.output_item.done", "output_index": 0, "item": call}),
                        serde_json::json!({"type": "response.completed", "response": response})
                    );
                    socket.write_all(events.as_bytes()).await.unwrap();
                }
            }
        }
        // Stalled connections stay open until the test ends.
        std::future::pending::<()>().await;
        drop(stalled);
    });
    (url, bodies)
}

fn agent(
    url: &str,
    retry: LLMRetryConfig,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let mut config = AgentConfig::builder(
        "network".into(),
        "Network".into(),
        AgentLLMConfig {
            model: ai::Model::openai_gpt_5_6_luna(),
            generation_config: Default::default(),
        },
    )
    .role_description("test".into())
    .instructions("test".into())
    .llm_retry_config(retry)
    .build()
    .unwrap();
    config.tool_config.default_auto_approve_tool_use = true;
    let task = Task::new(config, "Finish.".into(), std::env::temp_dir(), None);
    let client = ai::LLMClient::new()
        .with_openai_client(ai::providers::openai::OpenAIClient::new("test").with_base_url(url));
    let (events, receiver) = mpsc::channel(200);
    let (sender, commands) = mpsc::channel(10);
    let agent = AgentLoop::new(
        task,
        Arc::new(client),
        Arc::new(ToolRegistry::new()),
        events,
        commands,
        None,
        true,
        None,
        None,
        None,
    );
    (agent, receiver, sender)
}

/// The opt-in stream idle timeout reaches the agent's requests: a stream
/// that goes quiet is cut off and sent again, instead of waiting forever.
#[tokio::test]
async fn a_stream_idle_timeout_cuts_a_quiet_stream_off_and_it_is_sent_again() {
    let (url, bodies) = serve(vec![Answer::Stall, Answer::Complete]).await;
    let (agent, mut events, _commands) = agent(
        &url,
        LLMRetryConfig {
            initial_delay_ms: 1,
            jitter: false,
            stream_idle_timeout_ms: Some(300),
            ..LLMRetryConfig::default()
        },
    );
    let task = timeout(Duration::from_secs(20), agent.run())
        .await
        .expect("a quiet stream is cut off")
        .unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { ref summary } if summary == "done"),
        "{:?}",
        task.current_state
    );
    // The request was sent again unchanged, without a note about the cut.
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["input"], bodies[1]["input"]);
    assert!(task.runtime_messages.is_empty());
    let mut truncated = false;
    while let Ok(event) = events.try_recv() {
        if let AgentEvent::ErrorEncountered { error_message, .. } = event {
            truncated |= error_message.contains("stream idle timeout");
        }
    }
    assert!(truncated, "the cut is reported as a truncated stream");
}

/// Without a timeout (the default) a quiet stream is waited for.
#[tokio::test]
async fn without_a_stream_idle_timeout_a_quiet_stream_is_waited_for() {
    let (url, _bodies) = serve(vec![Answer::Stall]).await;
    let (agent, _events, _commands) = agent(&url, LLMRetryConfig::default());
    assert!(
        timeout(Duration::from_secs(2), agent.run()).await.is_err(),
        "the request is still waiting"
    );
}

/// A cache refresh sends the task's next request (its system prompt, tools,
/// conversation and cache key) with a short ping after it and OpenAI's
/// smallest output limit, and returns the request's usage.
#[tokio::test]
async fn a_cache_refresh_sends_the_next_request_with_a_ping() {
    let (url, bodies) = serve(vec![Answer::Text]).await;
    let (mut agent, _events, _commands) = agent(&url, LLMRetryConfig::default());
    agent.task.conversation.system_message = Some("You are a test agent.".into());
    agent.task.conversation.add_user_message("Finish.");
    agent.task.conversation.add_assistant_message("Done.");
    agent.task.set_state(TaskState::Completed {
        summary: "Done.".into(),
    });
    let key = agent.task.conversation.prompt_cache_key.clone();
    let refresh = agent.refresh_prompt_cache().await.unwrap();
    assert!(!refresh.resent_last_request);
    assert_eq!(refresh.output_cap, Some(16));
    let usage = refresh.usage;
    assert_eq!(usage.input_tokens, 1200);
    assert_eq!(usage.cached_tokens, Some(1100));
    let body = bodies.lock().unwrap()[0].clone();
    assert_eq!(body["instructions"], "You are a test agent.");
    assert_eq!(body["max_output_tokens"], 16);
    assert_eq!(body["prompt_cache_key"].as_str(), Some(key.as_str()));
    assert_eq!(body["tools"][0]["name"], "complete_task");
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 3);
    assert_eq!(input[2]["content"], crate::runtime::CACHE_PING);
}

/// Where a request can't keep the cache warm, nothing is sent.
#[tokio::test]
async fn a_cache_refresh_is_refused_where_it_can_t_help() {
    let unsupported = |agent: AgentLoop| async move {
        match agent.refresh_prompt_cache().await {
            Err(ArkoAgentsError::LLM(ai::Error::UnsupportedFeature(reason))) => reason,
            other => panic!("{other:?}"),
        }
    };
    let (fresh, _events, _commands) = agent("http://127.0.0.1:9", LLMRetryConfig::default());
    assert!(unsupported(fresh).await.contains("no request"));

    let (mut gemini, _events, _commands) = agent("http://127.0.0.1:9", LLMRetryConfig::default());
    gemini.task.conversation.system_message = Some("prompt".into());
    gemini.task.conversation.add_user_message("Finish.");
    gemini.task.conversation.model =
        ai::Model::get(ai::Provider::Google, "gemini-3.5-flash").unwrap();
    gemini.task.agent_config.llm_config.model = gemini.task.conversation.model.clone();
    assert!(unsupported(gemini).await.contains("can't be kept warm"));

    // OpenRouter derives Claude's thinking budget from the output limit.
    let (mut openrouter, _events, _commands) =
        agent("http://127.0.0.1:9", LLMRetryConfig::default());
    openrouter.task.conversation.system_message = Some("prompt".into());
    openrouter.task.conversation.add_user_message("Finish.");
    openrouter.task.conversation.model = ai::Model::openrouter("anthropic/claude-sonnet-5");
    let llm = &mut openrouter.task.agent_config.llm_config;
    llm.model = openrouter.task.conversation.model.clone();
    llm.generation_config.reasoning = Some(ai::ReasoningConfig::openrouter_effort(
        ai::ReasoningEffort::High,
    ));
    assert!(
        unsupported(openrouter)
            .await
            .contains("change the thinking parameters")
    );
}

/// A provider that refuses the output limit gets the request once more
/// with a larger one, and the refresh says so.
#[tokio::test]
async fn a_refused_output_limit_is_raised_once_and_reported() {
    let (url, bodies) = serve(vec![Answer::RefuseCap, Answer::Text]).await;
    let (mut agent, _events, _commands) = agent(&url, LLMRetryConfig::default());
    agent.task.conversation.system_message = Some("You are a test agent.".into());
    agent.task.conversation.add_user_message("Finish.");
    agent.task.conversation.add_assistant_message("Done.");
    let refresh = agent.refresh_prompt_cache().await.unwrap();
    assert_eq!(refresh.output_cap, Some(1_024));
    let refused = refresh.refused_cap.unwrap();
    assert!(refused.contains("max_output_tokens"), "{refused}");
    assert_eq!(refresh.usage.cached_tokens, Some(1100));
    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies[0]["max_output_tokens"], 16);
    assert_eq!(bodies[1]["max_output_tokens"], 1_024);
    assert_eq!(bodies[1]["input"], bodies[0]["input"]);
}
