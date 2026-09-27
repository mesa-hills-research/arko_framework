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

/// Every event the parser yields for a fixture SSE body, stopping at the first error.
async fn events(body: String) -> (Vec<StreamEvent>, Option<Error>) {
    let bytes = futures::stream::iter([Ok::<_, std::io::Error>(body)]);
    let mut stream = OpenAIClient::stream_events(
        bytes, Model::openai_gpt_6_astra(), GenerationConfig::default(), false,
    );
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(error) => return (events, Some(error)),
        }
    }
    (events, None)
}

fn end(events: &[StreamEvent]) -> &GenerationResponse {
    match events.last() {
        Some(StreamEvent::End(response)) => response,
        other => panic!("expected End last, got {other:?}"),
    }
}

fn assert_model_refusal(response: &GenerationResponse) {
    assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("refusal"));
    assert_eq!(response.stop_reason, StopReason::Refusal);
    let refusal = response.refusal.as_ref().unwrap();
    assert_eq!(refusal.source, RefusalSource::Model);
    assert_eq!(refusal.explanation.as_deref(), Some("Fixture refusal."));
    assert_eq!(response.text(), "", "refusal text must not be part of the message text");
    let saved = serde_json::to_value(&response.model_call_usage).unwrap();
    assert_eq!(saved["stop_details"]["refusals"][0]["refusal"], "Fixture refusal.");
}

#[tokio::test]
async fn refusal_content_is_reported_as_a_refusal_for_text_and_stream_calls() {
    let mut response = completed_response();
    response["output"][0]["content"] = serde_json::json!([
        {"type": "refusal", "refusal": "Fixture refusal."}
    ]);
    let streamed_response = streamed(event(serde_json::json!({
        "type": "response.completed", "response": response
    }))).await;
    assert_model_refusal(&streamed_response);
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![reply("/responses", 200, &response.to_string())]).await;
        let response = client(&url).generate_text(
            &Conversation::new(Model::openai_gpt_6_astra()), &GenerationConfig::default(),
        ).await.unwrap();
        assert_model_refusal(&response);
        server.await.unwrap();
    }).await.unwrap();
}

#[tokio::test]
async fn streamed_refusal_is_announced_once_and_never_streamed_as_text() {
    let mut response = completed_response();
    response["output"] = serde_json::json!([]);
    let body = [
        serde_json::json!({"type":"response.output_item.added", "output_index":0, "item":{"type":"function_call", "id":"fc_test", "call_id":"call_test", "name":"fixture_tool"}}),
        serde_json::json!({"type":"response.function_call_arguments.delta", "item_id":"fc_test", "output_index":0, "delta":"{\"partial\":"}),
        serde_json::json!({"type":"response.refusal.delta", "item_id":"msg_test", "content_index":0, "delta":"Fixture "}),
        serde_json::json!({"type":"response.refusal.delta", "item_id":"msg_test", "content_index":0, "delta":"refusal."}),
        serde_json::json!({"type":"response.refusal.done", "item_id":"msg_test", "content_index":0, "refusal":"Fixture refusal."}),
        serde_json::json!({"type":"response.completed", "response":response}),
    ].into_iter().map(event).collect::<String>();
    let (events, error) = events(body).await;
    assert!(error.is_none());
    assert!(!events.iter().any(|event| matches!(event, StreamEvent::TextChunk(_))));
    let refusals: Vec<_> = events.iter().filter(|event| matches!(event, StreamEvent::Refusal(_))).collect();
    assert_eq!(refusals.len(), 1);
    assert!(matches!(events[events.len() - 2], StreamEvent::Refusal(_)), "Refusal precedes End");
    let response = end(&events);
    assert_model_refusal(response);
    assert!(response.message.tool_calls.is_empty(), "an unfinished call is never recovered");
    assert_eq!(response.model_call_usage.stop_details.clone().unwrap()["refusals"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn content_filter_moves_partial_output_out_of_the_message() {
    let mut response = completed_response();
    response["status"] = "incomplete".into();
    response["incomplete_details"] = serde_json::json!({"reason":"content_filter"});
    let response = streamed(event(serde_json::json!({
        "type":"response.incomplete", "response":response,
    }))).await;
    assert_eq!(response.model_call_usage.stop_reason.as_deref(), Some("content_filter"));
    assert_eq!(response.model_call_usage.stop_details.clone().unwrap()["incomplete_details"]["reason"], "content_filter");
    assert_eq!(response.usage.output_tokens, 1);
    assert_eq!(response.stop_reason, StopReason::Refusal);
    let refusal = response.refusal.as_ref().unwrap();
    assert_eq!(refusal.source, RefusalSource::ContentFilter);
    assert_eq!(refusal.partial_output.as_deref(), Some("ok"));
    assert_eq!(response.text(), "");
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
    let response = OpenAIClient::generation_response(
        &response, &Model::openai_gpt_6_astra(), &GenerationConfig::default(), false,
        &StreamingRefusals::new(), Vec::new(),
    );
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(response.refusal.is_none());
}

fn function_call(id: &str, arguments: &str, status: &str) -> Value {
    serde_json::json!({"type":"function_call", "id":format!("fc_{id}"), "call_id":format!("call_{id}"),
        "name":"fixture_tool", "arguments":arguments, "status":status})
}

#[test]
fn max_output_tokens_keeps_finished_calls_and_drops_cut_off_ones() {
    let mut value = completed_response();
    value["status"] = "incomplete".into();
    value["incomplete_details"] = serde_json::json!({"reason":"max_output_tokens"});
    value["output"] = serde_json::json!([
        function_call("a", "{\"n\":1}", "completed"),
        function_call("b", "{\"n\":", "incomplete"),
    ]);
    let response: OpenAIResponse = serde_json::from_value(value).unwrap();
    let response = OpenAIClient::generation_response(
        &response, &Model::openai_gpt_6_astra(), &GenerationConfig::default(), false,
        &StreamingRefusals::new(), Vec::new(),
    );
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].arguments, serde_json::json!({"n":1}));
    assert!(matches!(
        response.single_tool_call("other"),
        Err(Error::UnexpectedToolCalls { stop_reason: StopReason::MaxTokens, .. })
    ));
}

#[tokio::test]
async fn interleaved_parallel_call_deltas_stay_with_their_calls() {
    let mut response = completed_response();
    response["output"] = serde_json::json!([]);
    let added = |id: &str, index: u64| serde_json::json!({"type":"response.output_item.added", "output_index":index,
        "item":{"type":"function_call", "id":format!("fc_{id}"), "call_id":format!("call_{id}"), "name":format!("tool_{id}")}});
    let delta = |id: &str, index: u64, delta: &str| serde_json::json!({"type":"response.function_call_arguments.delta",
        "item_id":format!("fc_{id}"), "output_index":index, "delta":delta});
    let done = |id: &str, index: u64, arguments: &str| serde_json::json!({"type":"response.function_call_arguments.done",
        "item_id":format!("fc_{id}"), "output_index":index, "arguments":arguments});
    let body = [
        added("a", 0), added("b", 1),
        delta("a", 0, "{\"x\":"), delta("b", 1, "{\"y\":"), delta("b", 1, "2}"), delta("a", 0, "1}"),
        done("a", 0, "{\"x\":1}"), done("b", 1, "{\"y\":2}"),
        serde_json::json!({"type":"response.completed", "response":response}),
    ].into_iter().map(event).collect::<String>();
    let (events, error) = events(body).await;
    assert!(error.is_none());
    let deltas: Vec<(usize, String)> = events.iter().filter_map(|event| match event {
        StreamEvent::ToolCallArgumentsDelta { index, delta } => Some((*index, delta.clone())),
        _ => None,
    }).collect();
    assert_eq!(deltas, [(0, "{\"x\":".into()), (1, "{\"y\":".into()), (1, "2}".into()), (0, "1}".into())]);
    let response = end(&events);
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let calls: Vec<_> = response.message.tool_calls.iter()
        .map(|call| (call.name.as_str(), call.arguments.clone(), call.provider_id.as_deref()))
        .collect();
    assert_eq!(calls, [
        ("tool_a", serde_json::json!({"x":1}), Some("fc_a")),
        ("tool_b", serde_json::json!({"y":2}), Some("fc_b")),
    ]);
}

#[tokio::test]
async fn a_stream_without_a_terminal_event_is_truncated() {
    let body = [
        serde_json::json!({"type":"response.output_text.delta", "delta":"partial"}),
    ].into_iter().map(event).collect::<String>();
    let (events, error) = events(body).await;
    assert!(matches!(events.as_slice(), [StreamEvent::TextChunk(text)] if text == "partial"));
    let error = error.expect("EOF before completion must be an error");
    assert!(matches!(error, Error::StreamTruncated(_)));
    assert!(error.is_retryable());
}

#[tokio::test]
async fn stream_error_events_without_a_code_stay_retryable() {
    // Flat `error` event with a null code: `type` is the event name.
    let body = event(serde_json::json!({"type":"error","code":null,
        "message":"An error occurred while processing your request.","param":null,"sequence_number":3}));
    let (_, error) = events(body).await;
    let error = error.unwrap();
    assert!(matches!(&error, Error::ProviderError { code: None, .. }), "{error:?}");
    assert_eq!(error.retry_class(), crate::RetryClass::Transient);
    // `response.failed` is classified by the error on its response.
    for (code, class) in [
        ("server_error", crate::RetryClass::Transient),
        ("invalid_prompt", crate::RetryClass::Permanent),
    ] {
        let body = event(serde_json::json!({"type":"response.failed","sequence_number":5,
            "response":{"id":"resp_1","status":"failed","error":{"code":code,"message":"Fixture failure."}}}));
        let (_, error) = events(body).await;
        let error = error.unwrap();
        assert!(error.to_string().contains("OpenAI stream error: Fixture failure."), "{error}");
        assert_eq!(error.retry_class(), class, "{code}");
    }
}

#[test]
fn invalid_argument_json_is_kept_raw_and_replayed_verbatim() {
    let mut value = completed_response();
    value["output"] = serde_json::json!([function_call("a", "{not json", "completed")]);
    let parsed: OpenAIResponse = serde_json::from_value(value).unwrap();
    let response = OpenAIClient::generation_response(
        &parsed, &Model::openai_gpt_6_astra(), &GenerationConfig::default(), false,
        &StreamingRefusals::new(), Vec::new(),
    );
    let call = &response.message.tool_calls[0];
    assert_eq!(call.invalid_arguments.as_deref(), Some("{not json"));
    assert!(call.arguments.is_null());
    assert!(matches!(response.single_tool_call("fixture_tool"), Err(Error::InvalidToolArguments { .. })));

    let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
    conversation.add_user_message("go");
    conversation.add_assistant_message_from_response(response);
    let request = OpenAIClient::new("test")
        .build_request(&conversation, &GenerationConfig::default(), false)
        .unwrap();
    let value = serde_json::to_value(request).unwrap();
    let replayed = value["input"].as_array().unwrap().iter()
        .find(|item| item["type"] == "function_call").unwrap();
    assert_eq!(replayed["arguments"], "{not json");
}

#[test]
fn unknown_output_items_and_missing_optional_fields_are_tolerated() {
    let value = serde_json::json!({
        "id":"resp_test", "status":"completed", "model":"gpt-6-astra",
        "output":[
            {"type":"web_search_call", "id":"ws_1", "status":"completed"},
            {"type":"reasoning", "id":"rs_no_content"},
            {"type":"message", "id":"msg_test", "content":[{"type":"output_text", "text":"ok"}]},
        ],
        "usage":{"input_tokens":1, "output_tokens":1},
    });
    let parsed: OpenAIResponse = serde_json::from_value(value).unwrap();
    let response = OpenAIClient::generation_response(
        &parsed, &Model::openai_gpt_6_astra(), &GenerationConfig::default(), false,
        &StreamingRefusals::new(), Vec::new(),
    );
    assert_eq!(response.text(), "ok");
    assert!(response.message.reasoning_artifact.is_none(), "reasoning without encrypted content can't be replayed");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

#[test]
fn every_reasoning_item_is_kept_and_replayed_in_order() {
    let mut value = completed_response();
    value["output"] = serde_json::json!([
        {"type":"reasoning", "id":"rs_1", "encrypted_content":"enc_1", "summary":[{"type":"summary_text", "text":"first"}]},
        {"type":"reasoning", "id":"rs_2", "encrypted_content":"enc_2"},
        function_call("a", "{}", "completed"),
    ]);
    let parsed: OpenAIResponse = serde_json::from_value(value).unwrap();
    let response = OpenAIClient::generation_response(
        &parsed, &Model::openai_gpt_6_astra(), &GenerationConfig::default(), false,
        &StreamingRefusals::new(), Vec::new(),
    );
    let Some(ReasoningArtifact::OpenAIEncryptedReasoning { items }) = &response.message.reasoning_artifact else {
        panic!("expected OpenAI reasoning");
    };
    assert_eq!(items.iter().map(|item| item.id.as_deref().unwrap()).collect::<Vec<_>>(), ["rs_1", "rs_2"]);

    let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
    conversation.add_user_message("go");
    conversation.add_assistant_message_from_response(response);
    let request = OpenAIClient::new("test")
        .build_request(&conversation, &GenerationConfig::default(), false)
        .unwrap();
    let value = serde_json::to_value(request).unwrap();
    let kinds: Vec<(String, String)> = value["input"].as_array().unwrap().iter()
        .filter(|item| item["type"] != "message")
        .map(|item| (item["type"].as_str().unwrap().into(), item["id"].as_str().unwrap_or("").into()))
        .collect();
    assert_eq!(kinds, [
        ("reasoning".into(), "rs_1".into()),
        ("reasoning".into(), "rs_2".into()),
        ("function_call".into(), "fc_a".into()),
    ]);
}

#[test]
fn a_single_image_message_is_sent_as_an_image_part() {
    let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
    conversation.messages.push(crate::message::Message::new_user_with_contents(vec![
        Content::Image(Image::new_url("https://example.com/fixture.png")),
    ]));
    let request = OpenAIClient::new("test")
        .build_request(&conversation, &GenerationConfig::default(), false)
        .unwrap();
    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["input"][0]["content"], serde_json::json!([
        {"type":"input_image", "image_url":"https://example.com/fixture.png"}
    ]));
}
