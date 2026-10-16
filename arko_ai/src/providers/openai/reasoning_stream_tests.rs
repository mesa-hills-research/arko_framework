//! Reasoning summaries in a Responses stream: each summary delta is a
//! `StreamEvent::ReasoningChunk` as it comes, apart from the reply's text,
//! with an API key and over ChatGPT OAuth.

use super::*;
use crate::model::Model;

fn event(value: Value) -> String {
    format!("data: {value}\n\n")
}

fn body() -> String {
    let reasoning = serde_json::json!({
        "type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
        "summary": [{"type": "summary_text", "text": "Adding two and two."}],
    });
    let message = serde_json::json!({
        "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": "Four.", "annotations": []}],
    });
    [
        serde_json::json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1",
                           "output_index": 0, "summary_index": 0, "delta": "Adding two "}),
        serde_json::json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1",
                           "output_index": 0, "summary_index": 0, "delta": "and two."}),
        serde_json::json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning}),
        serde_json::json!({"type": "response.output_text.delta", "item_id": "msg_1",
                           "output_index": 1, "delta": "Four."}),
        serde_json::json!({"type": "response.output_item.done", "output_index": 1, "item": message}),
        serde_json::json!({"type": "response.completed", "response": {
            "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
            "model": "gpt-6-astra", "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 20,
                      "output_tokens_details": {"reasoning_tokens": 12}},
        }}),
    ]
    .into_iter()
    .map(event)
    .collect()
}

#[tokio::test]
async fn summary_deltas_stream_as_reasoning_chunks() {
    for oauth in [false, true] {
        let bytes = futures::stream::iter([Ok::<_, std::io::Error>(body())]);
        let config = GenerationConfig::default().with_reasoning(crate::ReasoningConfig::openai(
            crate::OpenAIEffort::Medium,
            true,
        ));
        let mut stream =
            OpenAIClient::stream_events(bytes, Model::openai_gpt_6_astra(), config, oauth);
        let mut reasoning = Vec::new();
        let mut text = Vec::new();
        let mut end = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::ReasoningChunk(chunk) => reasoning.push(chunk),
                StreamEvent::TextChunk(chunk) => text.push(chunk),
                StreamEvent::End(response) => end = Some(response),
                _ => {}
            }
        }
        assert_eq!(reasoning, ["Adding two ", "and two."], "oauth: {oauth}");
        assert_eq!(text, ["Four."], "oauth: {oauth}");
        assert_eq!(end.unwrap().text(), "Four.");
    }
}
