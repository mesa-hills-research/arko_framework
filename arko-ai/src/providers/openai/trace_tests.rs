use super::oauth_recovery_tests::{client, completed_response, reply, server};
use super::*;
use crate::{model::Model, trace::RequestTrace};
use reqwest::header::{HeaderMap, HeaderValue};

struct SessionHeader;

impl OpenAIRequestHook for SessionHeader {
    fn prepare(&self, conversation_key: &str, headers: &mut HeaderMap, _body: &mut Value) {
        headers.insert("session-id", HeaderValue::from_str(conversation_key).unwrap());
    }
}

#[tokio::test]
async fn traced_stream_records_request_events_and_empty_completed_output() {
    let mut completed = completed_response();
    completed["output"] = serde_json::json!([]);
    let body = format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\n",
        serde_json::json!({"type": "response.created", "response": {"model": "gpt-6-astra"}}),
        serde_json::json!({"type": "response.output_item.done", "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []}}),
        serde_json::json!({"type": "response.completed", "response": completed}),
    );
    let (url, server) = server(vec![reply("/responses", 200, &body)]).await;
    let dir = std::env::temp_dir().join(format!("arko-trace-stream-{}", uuid::Uuid::new_v4()));
    let client = client(&url)
        .with_request_trace(Some(RequestTrace::new(&dir)))
        .with_request_hook(Some(Arc::new(SessionHeader)));
    let mut conversation = Conversation::new(Model::get(ProviderType::OpenAI, "gpt-6-astra").unwrap());
    conversation.add_user_message("hi");

    let mut stream = client
        .generate_stream(&conversation, &GenerationConfig::default())
        .await
        .unwrap();
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    drop(stream);

    let requests = server.await.unwrap();
    let key = conversation.prompt_cache_key.to_ascii_lowercase();
    assert!(requests[0].0.to_ascii_lowercase().contains(&format!("session-id: {key}")));

    let line: Value =
        serde_json::from_str(std::fs::read_to_string(dir.join("calls.jsonl")).unwrap().trim()).unwrap();
    assert_eq!(line["status"], "completed");
    assert_eq!(line["created_model"], "gpt-6-astra");
    assert_eq!(line["completed_output_count"], 0);
    assert_eq!(line["streamed_output_count"], 1);
    assert_eq!(line["streamed_reasoning_with_encrypted_content"], 1);
    assert_eq!(line["prefix"]["first_request"], true);

    let conversation_dir = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    let request: Value = serde_json::from_str(
        &std::fs::read_to_string(conversation_dir.join("0001-request.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(request["headers"]["authorization"], "[REDACTED]");
    assert_eq!(request["headers"]["chatgpt-account-id"], "[REDACTED]");
    assert_eq!(request["headers"]["session-id"], conversation.prompt_cache_key);
    assert_eq!(request["body"]["model"], "gpt-6-astra");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn a_failed_request_is_traced_as_an_error() {
    let (url, server) = server(vec![reply("/responses", 400, r#"{"error":{"code":"invalid_request_error","message":"bad"}}"#)]).await;
    let dir = std::env::temp_dir().join(format!("arko-trace-error-{}", uuid::Uuid::new_v4()));
    let client = client(&url).with_request_trace(Some(RequestTrace::new(&dir)));
    let mut conversation = Conversation::new(Model::get(ProviderType::OpenAI, "gpt-6-astra").unwrap());
    conversation.add_user_message("hi");

    assert!(client.generate_text(&conversation, &GenerationConfig::default()).await.is_err());
    server.await.unwrap();

    let line: Value =
        serde_json::from_str(std::fs::read_to_string(dir.join("calls.jsonl")).unwrap().trim()).unwrap();
    assert_eq!(line["status"], "error");
    assert_eq!(line["http_status"], 400);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn llm_client_settings_apply_to_an_installed_openai_client() {
    let bad = r#"{"error":{"code":"invalid_request_error","message":"bad"}}"#;
    let (url, server) =
        server(vec![reply("/responses", 400, bad), reply("/responses", 400, bad)]).await;
    let mut conversation = Conversation::new(Model::get(ProviderType::OpenAI, "gpt-6-astra").unwrap());
    conversation.add_user_message("hi");
    let calls = |dir: &std::path::Path| {
        std::fs::read_to_string(dir.join("calls.jsonl")).unwrap_or_default().lines().count()
    };

    // A trace set before the client is installed still applies to it.
    let dir = std::env::temp_dir().join(format!("arko-trace-installed-{}", uuid::Uuid::new_v4()));
    let llm = crate::LLMClient::new()
        .with_request_trace(RequestTrace::new(&dir))
        .with_openai_client(client(&url));
    assert!(llm.generate_text(&conversation, GenerationConfig::default()).await.is_err());
    assert_eq!(calls(&dir), 1);

    // Settings the LLMClient leaves unset keep the installed client's own.
    let own = std::env::temp_dir().join(format!("arko-trace-own-{}", uuid::Uuid::new_v4()));
    let llm = crate::LLMClient::new()
        .with_openai_request_hook(Arc::new(SessionHeader))
        .with_openai_client(client(&url).with_request_trace(Some(RequestTrace::new(&own))));
    assert!(llm.generate_text(&conversation, GenerationConfig::default()).await.is_err());
    assert_eq!(calls(&own), 1);

    let requests = server.await.unwrap();
    let key = conversation.prompt_cache_key.to_ascii_lowercase();
    assert!(requests[1].0.to_ascii_lowercase().contains(&format!("session-id: {key}")));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&own).ok();
}
