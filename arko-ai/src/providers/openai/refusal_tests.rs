use super::*;
use super::oauth_recovery_tests::{client, completed_response, reply, server};
use crate::model::Model;

fn event(value: Value) -> String {
    format!("data: {value}\n\n")
}

async fn streamed(body: String) -> GenerationResponse {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![reply("/responses", 200, &body)]).await;
        let client = client(&url);
        let conversation = Conversation::new(Model::openai_gpt_6_astra());
        let mut stream = client.generate_stream(&conversation, &GenerationConfig::default()).await.unwrap();
        let mut result = None;
        while let Some(item) = stream.next().await {
            if let StreamEvent::End(response) = item.unwrap() { result = Some(*response); }
        }
        server.await.unwrap();
        result.expect("terminal response and usage must be emitted")
    }).await.unwrap()
}

fn assert_refusal(response: &GenerationResponse) {
    assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("refusal"));
    assert!(matches!(&response.message.content[0], Content::Text(text) if text == "Fixture refusal."));
    let saved = serde_json::to_value(&response.model_call_usage).unwrap();
    assert_eq!(saved["stop_details"]["refusals"][0]["refusal"], "Fixture refusal.");
}

#[tokio::test]
async fn refusal_content_without_text_is_preserved_for_text_and_stream_calls() {
    let mut response = completed_response();
    response["output"][0]["content"] = serde_json::json!([
        {"type": "refusal", "refusal": "Fixture refusal."}
    ]);
    let streamed_response = streamed(event(serde_json::json!({
        "type": "response.completed", "response": response
    }))).await;
    assert_refusal(&streamed_response);
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![reply("/responses", 200, &response.to_string())]).await;
        let response = client(&url).generate_text(
            &Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default(),
        ).await.unwrap();
        assert_refusal(&response);
        server.await.unwrap();
    }).await.unwrap();
}

#[tokio::test]
async fn streaming_refusal_done_replaces_deltas_and_does_not_recover_partial_tools() {
    let mut response = completed_response();
    response["output"] = serde_json::json!([]);
    let body = [
        serde_json::json!({"type":"response.output_item.added", "item":{"type":"function_call", "id":"fc_test", "call_id":"call_test", "name":"fixture_tool"}}),
        serde_json::json!({"type":"response.function_call_arguments.delta", "delta":"{\"partial\":"}),
        serde_json::json!({"type":"response.refusal.delta", "item_id":"msg_test", "content_index":0, "delta":"Fixture "}),
        serde_json::json!({"type":"response.refusal.delta", "item_id":"msg_test", "content_index":0, "delta":"refusal."}),
        serde_json::json!({"type":"response.refusal.done", "item_id":"msg_test", "content_index":0, "refusal":"Fixture refusal."}),
        serde_json::json!({"type":"response.completed", "response":response}),
    ].into_iter().map(event).collect::<String>();
    let response = streamed(body).await;
    assert_refusal(&response);
    assert!(response.message.tool_calls.is_empty());
    assert_eq!(response.model_call_usage.stop_details.unwrap()["refusals"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn content_filter_incomplete_event_preserves_reason_and_usage() {
    let mut response = completed_response();
    response["status"] = "incomplete".into();
    response["incomplete_details"] = serde_json::json!({"reason":"content_filter"});
    let response = streamed(event(serde_json::json!({
        "type":"response.incomplete", "response":response,
    }))).await;
    assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("content_filter"));
    assert_eq!(response.model_call_usage.stop_details.unwrap()["incomplete_details"]["reason"], "content_filter");
    assert_eq!(response.usage.output_tokens, 1);
}

#[test]
fn ordinary_text_mentioning_refusal_is_not_classified_as_refusal() {
    let mut value = completed_response();
    value["output"][0]["content"][0]["text"] = "The word refusal is just fixture text.".into();
    let response: OpenAIResponse = serde_json::from_value(value).unwrap();
    let usage = ModelCallUsage::new("test".into(), "openai".into(), "test".into(), 1, 1, 100);
    let usage = OpenAIClient::with_response_stop_metadata(usage, &response, &StreamingRefusals::new());
    assert_eq!(usage.stop_reason.as_deref(), Some("completed"));
    assert!(usage.stop_details.is_none());
}
