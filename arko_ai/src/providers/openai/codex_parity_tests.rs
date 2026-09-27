//! ChatGPT OAuth parity with Codex over HTTP: identity headers, turn state,
//! output assembly from `response.output_item.done`, verbatim replay, and
//! reroute detection.

use super::oauth_recovery_tests::{api_key_client, client, completed_response, reply, server};
use super::*;
use crate::{
    model::Model,
    tool::{ToolConfig, ToolDefinition, ToolResult},
};
use serde_json::json;

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

/// The value of header `name` in a raw HTTP request head.
fn header(raw: &str, name: &str) -> Option<String> {
    raw.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

fn completed(output: Value) -> Value {
    let mut response = completed_response();
    response["output"] = output;
    response
}

fn reasoning_item() -> Value {
    json!({"type": "reasoning", "id": "rs_1", "summary": [], "content": [],
        "encrypted_content": "enc_1", "status": "completed"})
}

fn commentary_item() -> Value {
    json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "Reading the file.", "annotations": [], "logprobs": []}]})
}

fn call_item(id: &str) -> Value {
    json!({"type": "function_call", "id": format!("fc_{id}"), "call_id": format!("call_{id}"),
        "name": "read_file", "arguments": "{\"path\":\"src/lib.rs\"}", "status": "completed"})
}

fn without_status(item: &Value) -> Value {
    let mut item = item.clone();
    item.as_object_mut().unwrap().remove("status");
    item
}

/// A Codex-backend stream: every item arrives as `output_item.done`, and
/// `response.completed` carries no output.
fn tool_turn_stream(call: &str) -> String {
    sse(&[
        json!({"type": "response.created", "response": {"model": "gpt-6-astra"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning_item()}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1"}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 1, "delta": "Reading the file."}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": commentary_item()}),
        json!({"type": "response.output_item.added", "output_index": 2,
            "item": {"type": "function_call", "id": format!("fc_{call}"), "call_id": format!("call_{call}"), "name": "read_file"}}),
        json!({"type": "response.function_call_arguments.delta", "item_id": format!("fc_{call}"), "output_index": 2,
            "delta": "{\"path\":\"src/lib.rs\"}"}),
        json!({"type": "response.function_call_arguments.done", "item_id": format!("fc_{call}"), "output_index": 2,
            "arguments": "{\"path\":\"src/lib.rs\"}"}),
        json!({"type": "response.output_item.done", "output_index": 2, "item": call_item(call)}),
        json!({"type": "response.completed", "response": completed(json!([]))}),
    ])
}

fn text_turn_stream(text: &str) -> String {
    let message = json!({"type": "message", "id": "msg_final", "status": "completed", "role": "assistant",
        "phase": "final_answer", "content": [{"type": "output_text", "text": text, "annotations": []}]});
    sse(&[
        json!({"type": "response.output_text.delta", "item_id": "msg_final", "output_index": 0, "delta": text}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": message}),
        json!({"type": "response.completed", "response": completed(json!([]))}),
    ])
}

fn tool_config() -> GenerationConfig {
    GenerationConfig::default().with_tool_config(ToolConfig::new(vec![ToolDefinition::new(
        "read_file",
        "Read a file.",
        json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    )]))
}

fn conversation() -> Conversation {
    let mut conversation = Conversation::new(Model::openai_gpt_6_astra());
    conversation.add_user_message("Audit src/lib.rs.");
    conversation
}

fn answer_tools(conversation: &mut Conversation, response: &GenerationResponse) {
    let results = response
        .message
        .tool_calls
        .iter()
        .map(|call| ToolResult::from_tool_call(call, "fn main() {}"))
        .collect();
    conversation.add_tool_results(results);
}

#[tokio::test]
async fn oauth_requests_carry_the_conversation_identity_and_api_key_requests_do_not() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let conversation = conversation();
        let key = conversation.prompt_cache_key.clone();
        let stream =
            sse(&[json!({"type": "response.completed", "response": completed_response()})]);
        let (url, server) = server(vec![
            reply("/responses", 200, &stream),
            reply("/responses", 200, &completed_response().to_string()),
            reply("/responses", 200, &stream),
            reply("/responses", 200, &completed_response().to_string()),
        ])
        .await;
        client(&url)
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap();
        api_key_client(&url)
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap();
        let identity = |client: OpenAIClient| {
            client
                .with_originator(Some("fixture-app".into()))
                .with_user_agent(Some("fixture-app/1.0".into()))
        };
        identity(client(&url))
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap();
        identity(api_key_client(&url))
            .generate_text(&conversation, &GenerationConfig::default())
            .await
            .unwrap();
        let requests = server.await.unwrap();

        let (oauth, oauth_body) = &requests[0];
        for name in ["session-id", "thread-id", "x-client-request-id"] {
            assert_eq!(header(oauth, name).as_deref(), Some(key.as_str()), "{name}");
        }
        assert_eq!(
            header(oauth, "accept").as_deref(),
            Some("text/event-stream")
        );
        assert_eq!(
            header(oauth, "chatgpt-account-id").as_deref(),
            Some("synthetic-account")
        );
        for absent in [
            "openai-beta",
            "originator",
            "user-agent",
            "x-codex-turn-state",
        ] {
            assert_eq!(header(oauth, absent), None, "{absent}");
        }
        assert_eq!(oauth_body["stream"], true);
        assert_eq!(oauth_body["store"], false);
        assert_eq!(oauth_body["prompt_cache_key"], key);
        assert_eq!(oauth_body["reasoning"]["context"], "all_turns");
        assert_eq!(
            oauth_body["reasoning"]["effort"], "medium",
            "OAuth: Codex always sends an effort"
        );
        assert_eq!(
            oauth_body["include"],
            json!(["reasoning.encrypted_content"])
        );
        assert!(oauth_body.get("prompt_cache_options").is_none());

        let (api, api_body) = &requests[1];
        for absent in [
            "session-id",
            "thread-id",
            "x-client-request-id",
            "openai-beta",
            "originator",
            "user-agent",
            "x-codex-turn-state",
            "chatgpt-account-id",
        ] {
            assert_eq!(header(api, absent), None, "{absent}");
        }
        assert_eq!(
            header(api, "authorization").as_deref(),
            Some("Bearer synthetic-key")
        );
        assert_eq!(api_body["stream"], false);
        assert_eq!(api_body["prompt_cache_key"], key);
        // `all_turns` is documented for API keys on GPT-5.6 only.
        assert!(
            api_body.get("reasoning").is_none(),
            "API key: model default effort, no context"
        );
        assert_eq!(api_body["prompt_cache_options"]["mode"], "implicit");

        for (raw, _) in &requests[2..] {
            assert_eq!(header(raw, "originator").as_deref(), Some("fixture-app"));
            assert_eq!(
                header(raw, "user-agent").as_deref(),
                Some("fixture-app/1.0")
            );
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn oauth_text_calls_assemble_output_from_output_item_done_and_replay_it_verbatim() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![
            reply("/responses", 200, &tool_turn_stream("a")),
            reply("/responses", 200, &text_turn_stream("No issues found.")),
        ])
        .await;
        let client = client(&url);
        let config = tool_config();
        let mut conversation = conversation();

        let response = client.generate_text(&conversation, &config).await.unwrap();
        assert_eq!(response.text(), "Reading the file.");
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        let call = &response.message.tool_calls[0];
        assert_eq!(
            (call.id.as_deref(), call.provider_id.as_deref()),
            (Some("call_a"), Some("fc_a"))
        );
        assert_eq!(call.arguments, json!({"path": "src/lib.rs"}));
        let Some(ReasoningArtifact::OpenAIOutputItems { items }) =
            &response.message.reasoning_artifact
        else {
            panic!("expected output items");
        };
        let expected =
            [reasoning_item(), commentary_item(), call_item("a")].map(|item| without_status(&item));
        assert_eq!(items.as_slice(), expected.as_slice());

        conversation.add_assistant_message_from_response(response.clone());
        answer_tools(&mut conversation, &response);
        let final_response = client.generate_text(&conversation, &config).await.unwrap();
        assert_eq!(final_response.text(), "No issues found.");
        assert_eq!(final_response.stop_reason, StopReason::EndTurn);

        let requests = server.await.unwrap();
        let input = requests[1].1["input"].as_array().unwrap();
        // Items replay in output order, exactly as received minus `status`:
        // no reconstruction and no empty assistant message.
        assert_eq!(input.len(), 5, "{input:#?}");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1..4], expected);
        assert_eq!(
            input[4],
            json!({"type": "function_call_output", "call_id": "call_a", "output": "fn main() {}"})
        );

        // The final answer replays after the tool output.
        conversation.add_assistant_message_from_response(final_response);
        conversation.add_user_message("Thanks.");
        let request = client.build_request(&conversation, &config, true).unwrap();
        let input = serde_json::to_value(request).unwrap()["input"].clone();
        let kinds: Vec<_> = input
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                format!(
                    "{}:{}",
                    item["type"].as_str().unwrap_or("message"),
                    item["id"].as_str().unwrap_or("")
                )
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "message:",
                "reasoning:rs_1",
                "message:msg_1",
                "function_call:fc_a",
                "function_call_output:",
                "message:msg_final",
                "message:",
            ]
        );
        assert_eq!(input[5]["phase"], "final_answer");
    })
    .await
    .unwrap();
}

#[test]
fn completed_output_is_merged_only_for_items_the_stream_did_not_deliver() {
    let streamed = vec![(Some(0), reasoning_item()), (Some(2), call_item("a"))];
    let completed = [reasoning_item(), commentary_item(), call_item("a")];
    let output = OpenAIClient::assemble_output(&streamed, &completed);
    assert_eq!(output, completed);
    assert_eq!(OpenAIClient::assemble_output(&[], &completed), completed);
    // Stream versions win over the terminal event's copies.
    let mut streamed_call = call_item("a");
    streamed_call["arguments"] = "{\"path\":\"streamed\"}".into();
    let output =
        OpenAIClient::assemble_output(&[(Some(0), streamed_call.clone())], &[call_item("a")]);
    assert_eq!(output, [streamed_call]);
}

#[tokio::test]
async fn streamed_deltas_without_output_items_are_kept_and_replayed() {
    let body = sse(&[
        json!({"type": "response.output_text.delta", "delta": "Checking."}),
        json!({"type": "response.output_item.added", "output_index": 1,
            "item": {"type": "function_call", "id": "fc_x", "call_id": "call_x", "name": "read_file"}}),
        json!({"type": "response.function_call_arguments.done", "item_id": "fc_x", "output_index": 1, "arguments": "{\"path\":\"a\"}"}),
        json!({"type": "response.completed", "response": completed(json!([]))}),
    ]);
    let bytes = futures::stream::iter([Ok::<_, std::io::Error>(body)]);
    let mut stream = OpenAIClient::stream_events(
        bytes,
        Model::openai_gpt_6_astra(),
        GenerationConfig::default(),
        true,
    );
    let mut response = None;
    while let Some(event) = stream.next().await {
        if let StreamEvent::End(end) = event.unwrap() {
            response = Some(*end);
        }
    }
    let response = response.unwrap();
    assert_eq!(response.text(), "Checking.");
    assert_eq!(response.message.tool_calls[0].id.as_deref(), Some("call_x"));
    let Some(ReasoningArtifact::OpenAIOutputItems { items }) = &response.message.reasoning_artifact
    else {
        panic!("expected output items");
    };
    assert_eq!(
        items.as_slice(),
        [
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]}),
            json!({"type": "function_call", "id": "fc_x", "call_id": "call_x", "name": "read_file", "arguments": "{\"path\":\"a\"}"}),
        ]
    );
}

#[test]
fn a_replayed_call_needs_the_message_to_still_hold_it() {
    let items = vec![
        without_status(&reasoning_item()),
        without_status(&call_item("a")),
        without_status(&call_item("b")),
    ];
    let mut message = Message::new_assistant("");
    message.reasoning_artifact = Some(ReasoningArtifact::OpenAIOutputItems { items });
    // The application kept only call b and added call c.
    message.tool_calls.push(ToolCall::from_argument_text(
        Some("call_b".into()),
        "read_file".into(),
        "{}",
    ));
    message.tool_calls.push(ToolCall::from_argument_text(
        Some("call_c".into()),
        "read_file".into(),
        "{}",
    ));
    let mut conversation = conversation();
    conversation.messages.push(message);
    let request = OpenAIClient::new("test")
        .build_request(&conversation, &GenerationConfig::default(), false)
        .unwrap();
    let input = serde_json::to_value(request).unwrap()["input"].clone();
    let calls: Vec<_> = input
        .as_array()
        .unwrap()
        .iter()
        .skip(1)
        .map(|item| {
            format!(
                "{}:{}",
                item["type"].as_str().unwrap(),
                item["call_id"].as_str().unwrap_or("")
            )
        })
        .collect();
    assert_eq!(
        calls,
        ["reasoning:", "function_call:call_b", "function_call:call_c"]
    );
}

/// The Responses API rejects a replayed reasoning item that nothing follows
/// ("Item 'rs_…' of type 'reasoning' was provided without its required
/// following item", openai/codex#1498), as a response cut off after its
/// reasoning leaves.
#[tokio::test]
async fn trailing_reasoning_without_a_following_item_is_not_replayed() {
    let mut cut_off = completed(json!([]));
    cut_off["status"] = "incomplete".into();
    cut_off["incomplete_details"] = json!({"reason": "max_output_tokens"});
    let body = sse(&[
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning_item()}),
        json!({"type": "response.incomplete", "response": cut_off}),
    ]);
    let bytes = futures::stream::iter([Ok::<_, std::io::Error>(body)]);
    let mut stream = OpenAIClient::stream_events(
        bytes,
        Model::openai_gpt_6_astra(),
        GenerationConfig::default(),
        true,
    );
    let mut response = None;
    while let Some(event) = stream.next().await {
        if let StreamEvent::End(end) = event.unwrap() {
            response = Some(*end);
        }
    }
    let response = response.unwrap();
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
    let mut history = conversation();
    history.add_assistant_message_from_response(response);
    history.add_user_message("Continue.");
    let request = OpenAIClient::new("test")
        .build_request(&history, &GenerationConfig::default(), true)
        .unwrap();
    let input = serde_json::to_value(request).unwrap()["input"].clone();
    // No dangling reasoning and no synthesized empty assistant message.
    assert_eq!(
        input,
        json!([
            {"role": "user", "type": "message", "content": "Audit src/lib.rs."},
            {"role": "user", "type": "message", "content": "Continue."},
        ])
    );

    // Only the trailing run of reasoning goes; reasoning that something
    // follows (an item, a call the message holds, or the message's own
    // text) is replayed.
    let second_reasoning = || {
        let mut item = without_status(&reasoning_item());
        item["id"] = "rs_2".into();
        item
    };
    let replay = |items: Vec<Value>, text: &str, calls: &[&str]| {
        let mut message = Message::new_assistant(text);
        message.reasoning_artifact = Some(ReasoningArtifact::OpenAIOutputItems { items });
        for call in calls {
            message.tool_calls.push(ToolCall::from_argument_text(
                Some(format!("call_{call}")),
                "read_file".into(),
                "{}",
            ));
        }
        let mut conversation = conversation();
        conversation.messages.push(message);
        conversation.add_user_message("Next.");
        let request = OpenAIClient::new("test")
            .build_request(&conversation, &GenerationConfig::default(), true)
            .unwrap();
        let input = serde_json::to_value(request).unwrap()["input"].clone();
        input
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                format!(
                    "{}:{}",
                    item["type"].as_str().unwrap(),
                    item["id"].as_str().or(item["role"].as_str()).unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
    };
    let reasoning = || without_status(&reasoning_item());
    let commentary = || without_status(&commentary_item());
    assert_eq!(
        replay(vec![reasoning(), second_reasoning()], "", &[]),
        ["message:user", "message:user"]
    );
    assert_eq!(
        replay(vec![reasoning(), commentary(), second_reasoning()], "", &[]),
        [
            "message:user",
            "reasoning:rs_1",
            "message:msg_1",
            "message:user"
        ],
    );
    // The call was dropped by the application: nothing follows the reasoning.
    assert_eq!(
        replay(vec![reasoning(), without_status(&call_item("a"))], "", &[]),
        ["message:user", "message:user"]
    );
    assert_eq!(
        replay(vec![reasoning()], "", &["b"]),
        [
            "message:user",
            "reasoning:rs_1",
            "function_call:",
            "message:user"
        ],
    );
    assert_eq!(
        replay(vec![reasoning()], "Declined.", &[]),
        [
            "message:user",
            "reasoning:rs_1",
            "message:assistant",
            "message:user"
        ],
    );
}

/// A refused turn an agent keeps (arko_agents' `handle_refusal`) loses its
/// tool calls and gets the refusal explanation as its text when it has
/// none. Its replay must not carry the dropped calls, and must still show
/// what the model declined.
#[tokio::test]
async fn a_kept_refusal_replays_without_the_calls_it_dropped() {
    let refusal_message = json!({"type": "message", "id": "msg_r", "status": "completed", "role": "assistant",
        "content": [{"type": "refusal", "refusal": "Fixture refusal."}]});
    let refusal_deltas = [
        json!({"type": "response.refusal.delta", "item_id": "msg_r", "content_index": 0, "delta": "Fixture refusal."}),
        json!({"type": "response.refusal.done", "item_id": "msg_r", "content_index": 0, "refusal": "Fixture refusal."}),
    ];
    let done = |index: u64, item: Value| json!({"type": "response.output_item.done", "output_index": index, "item": item});
    let completed_event = json!({"type": "response.completed", "response": completed(json!([]))});
    // The backend's usual stream, and one whose message item never arrived.
    let with_message = [
        vec![done(0, reasoning_item())],
        refusal_deltas.to_vec(),
        vec![
            done(1, refusal_message.clone()),
            done(2, call_item("a")),
            completed_event.clone(),
        ],
    ]
    .concat();
    let without_message = [
        vec![done(0, reasoning_item())],
        refusal_deltas.to_vec(),
        vec![done(1, call_item("a")), completed_event],
    ]
    .concat();

    for (events, expected_message) in [
        (with_message, without_status(&refusal_message)),
        (
            without_message,
            json!({"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Fixture refusal."}]}),
        ),
    ] {
        let bytes = futures::stream::iter([Ok::<_, std::io::Error>(sse(&events))]);
        let mut stream = OpenAIClient::stream_events(
            bytes,
            Model::openai_gpt_6_astra(),
            GenerationConfig::default(),
            true,
        );
        let mut response = None;
        while let Some(event) = stream.next().await {
            if let StreamEvent::End(end) = event.unwrap() {
                response = Some(*end);
            }
        }
        let response = response.unwrap();
        let refusal = response.refusal.clone().unwrap();
        assert_eq!(refusal.source, RefusalSource::Model);
        assert_eq!(response.message.tool_calls[0].id.as_deref(), Some("call_a"));
        let Some(ReasoningArtifact::OpenAIOutputItems { items }) =
            &response.message.reasoning_artifact
        else {
            panic!("expected output items");
        };
        assert!(items.iter().any(|item| item["type"] == "function_call"));

        // What the agent keeps: no calls, and the explanation as the text.
        let mut kept = response.message;
        kept.tool_calls.clear();
        kept.content = vec![Content::Text(refusal.explanation.unwrap())];
        let mut conversation = conversation();
        conversation.messages.push(kept);
        conversation.add_user_message("Something else, then.");

        let request = OpenAIClient::new("test")
            .build_request(&conversation, &GenerationConfig::default(), false)
            .unwrap();
        let input = serde_json::to_value(request).unwrap()["input"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            !input
                .iter()
                .any(|item| item["type"] == "function_call"
                    || item["type"] == "function_call_output"),
            "a dropped call must not be replayed: {input:?}"
        );
        assert_eq!(
            input[1..3],
            [without_status(&reasoning_item()), expected_message]
        );
        assert_eq!(input[3]["role"], "user");
        assert_eq!(input.len(), 4);
    }
}

#[tokio::test]
async fn turn_state_is_echoed_within_a_turn_and_dropped_at_the_next_user_message() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![
            reply("/responses", 200, &tool_turn_stream("a"))
                .with_header("x-codex-turn-state", "turn-1"),
            reply("/responses", 200, &tool_turn_stream("b"))
                .with_header("x-codex-turn-state", "turn-1b"),
            reply("/responses", 200, &text_turn_stream("Done."))
                .with_header("x-codex-turn-state", "turn-1c"),
            reply("/responses", 200, &tool_turn_stream("c"))
                .with_header("x-codex-turn-state", "turn-2"),
            reply("/responses", 200, &text_turn_stream("Done again.")),
            reply("/responses", 200, &completed_response().to_string()),
        ])
        .await;
        let client = client(&url);
        let config = tool_config();
        let mut conversation = conversation();

        // Turn 1: a fresh user message, two tool follow-ups.
        for _ in 0..2 {
            let response = client.generate_text(&conversation, &config).await.unwrap();
            conversation.add_assistant_message_from_response(response.clone());
            answer_tools(&mut conversation, &response);
        }
        let response = client.generate_text(&conversation, &config).await.unwrap();
        assert_eq!(response.openai_turn_state.as_deref(), Some("turn-1c"));
        conversation.add_assistant_message_from_response(response);

        // Turn 2.
        conversation.add_user_message("Now src/main.rs.");
        let response = client.generate_text(&conversation, &config).await.unwrap();
        conversation.add_assistant_message_from_response(response.clone());
        answer_tools(&mut conversation, &response);
        client.generate_text(&conversation, &config).await.unwrap();

        // API-key requests never carry it.
        assert_eq!(conversation.current_openai_turn_state(), Some("turn-2"));
        api_key_client(&url)
            .generate_text(&conversation, &config)
            .await
            .unwrap();

        let sent: Vec<_> = server
            .await
            .unwrap()
            .iter()
            .map(|(raw, _)| header(raw, "x-codex-turn-state"))
            .collect();
        let expected = [
            None,
            Some("turn-1"),
            Some("turn-1"),
            None,
            Some("turn-2"),
            None,
        ];
        assert_eq!(sent, expected.map(|value| value.map(String::from)));
    })
    .await
    .unwrap();
}

#[test]
fn saved_conversations_with_reasoning_items_replay_through_reconstruction() {
    // Conversations saved before output items were kept: one reasoning item
    // stored directly (oldest form) and a list of items; no turn state.
    let key = "arko:conversation:saved";
    let saved = json!({
        "model": serde_json::to_value(Model::openai_gpt_6_astra()).unwrap(),
        "system_message": "sys", "enable_cache": true, "prompt_cache_key": key,
        "messages": [
            {"role": "User", "content": [{"Text": "Audit."}], "reasoning_artifact": null, "id": null},
            {"role": "Assistant", "content": [{"Text": ""}], "id": null,
                "reasoning_artifact": {"OpenAIEncryptedReasoning": {"encrypted_content": "enc_old", "summaries": null, "id": "rs_old"}},
                "tool_calls": [{"id": "call_1", "name": "read_file", "arguments": {"path": "a"}, "provider_id": "fc_1"}]},
            {"role": "User", "id": null, "reasoning_artifact": null,
                "content": [{"ToolResult": {"call_id": "call_1", "name": "read_file", "output": "ok", "is_error": false}}]},
            {"role": "Assistant", "content": [{"Text": "Looks fine."}], "id": "msg_2",
                "reasoning_artifact": {"OpenAIEncryptedReasoning": {"items": [
                    {"encrypted_content": "enc_2", "summaries": ["s"], "id": "rs_2"}]}}},
        ],
    });
    let conversation: Conversation = serde_json::from_value(saved).unwrap();
    assert_eq!(conversation.openai_turn_state, None);
    let request = OpenAIClient::new("test")
        .build_request(&conversation, &GenerationConfig::default(), true)
        .unwrap();
    let input = serde_json::to_value(request).unwrap()["input"].clone();
    assert_eq!(
        input,
        json!([
            {"role": "user", "type": "message", "content": "Audit."},
            {"type": "reasoning", "encrypted_content": "enc_old", "id": "rs_old"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"a\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            {"type": "reasoning", "encrypted_content": "enc_2", "summary": [{"type": "summary_text", "text": "s"}], "id": "rs_2"},
            {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "Looks fine."}], "id": "msg_2"},
        ])
    );
}

#[tokio::test]
async fn plan_usage_headers_are_reported_with_the_call() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![
            reply("/responses", 200, &text_turn_stream("ok"))
                .with_header("x-codex-plan-type", "pro")
                .with_header("x-codex-primary-used-percent", "92.5")
                .with_header("x-codex-primary-window-minutes", "300")
                .with_header("x-codex-primary-reset-at", "1791108021")
                .with_header("x-codex-secondary-used-percent", "40")
                .with_header("x-codex-secondary-window-minutes", "10080"),
            reply("/responses", 200, &text_turn_stream("ok")),
        ])
        .await;
        let client = client(&url);
        let response = client
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        let usage = response.model_call_usage.plan_usage.unwrap();
        assert_eq!(usage.plan_type.as_deref(), Some("pro"));
        let (window, most_used) = usage.most_used_window().unwrap();
        assert_eq!((window, most_used.used_percent), ("primary", 92.5));
        assert_eq!(most_used.resets_at, Some(1_791_108_021));
        assert_eq!(usage.secondary.unwrap().window_minutes, Some(10080));
        let without = client
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(without.model_call_usage.plan_usage, None);
        server.await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_served_model_other_than_the_requested_one_is_a_reroute() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (url, server) = server(vec![
            reply("/responses", 200, &text_turn_stream("ok"))
                .with_header("openai-model", "gpt-5.2"),
            reply("/responses", 200, &text_turn_stream("ok")),
        ])
        .await;
        let client = client(&url);
        let rerouted = client
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(rerouted.served_model(), Some("gpt-5.2"));
        assert!(rerouted.rerouted());
        let saved = serde_json::to_value(&rerouted.model_call_usage).unwrap();
        assert_eq!(
            (saved["served_model"].as_str(), saved["rerouted"].as_bool()),
            (Some("gpt-5.2"), Some(true))
        );

        let served = client
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap();
        assert_eq!(served.served_model(), Some("gpt-6-astra"));
        assert!(!served.rerouted());
        assert!(
            serde_json::to_value(&served.model_call_usage)
                .unwrap()
                .get("rerouted")
                .is_none()
        );
        server.await.unwrap();
    })
    .await
    .unwrap();

    let response = |model: &str| -> OpenAIResponse {
        serde_json::from_value(json!({"id": "r", "status": "completed", "model": model,
            "usage": {"input_tokens": 1, "output_tokens": 1}}))
        .unwrap()
    };
    let head = |requested: &str, header: Option<&str>| ResponseHead {
        requested_model: requested.into(),
        served_model: header.map(String::from),
        turn_state: None,
        plan_usage: None,
    };
    let served = |head: &ResponseHead, model: &str, created: Option<&str>| {
        OpenAIClient::served_model(head, &response(model), created)
    };
    // Snapshot suffixes and case don't count as a reroute.
    assert_eq!(
        served(&head("gpt-5.5", None), "gpt-5.5-2026-04-23", None),
        (Some("gpt-5.5-2026-04-23".into()), false)
    );
    assert!(
        !served(
            &head("gpt-6-astra", None),
            "GPT-6-Astra",
            Some("gpt-6-astra")
        )
        .1
    );
    // Any reported model that differs does; the header is the served model.
    assert_eq!(
        served(&head("gpt-6-astra", None), "gpt-6-astra", Some("gpt-5.2")),
        (Some("gpt-6-astra".into()), true)
    );
    assert_eq!(
        served(&head("gpt-6-astra", Some("gpt-5.2")), "gpt-6-astra", None),
        (Some("gpt-5.2".into()), true)
    );
    assert_eq!(served(&head("gpt-6-astra", None), "", None), (None, false));
    assert!(served(&head("gpt-6-astra", None), "gpt-6-astra-é", None).1);
}

/// `reasoning.context: "all_turns"` is documented for API-key requests only
/// on GPT-5.6 ("GPT-5.6 models support this value"); over ChatGPT OAuth it
/// is sent for GPT-5.6 and GPT-6, as Codex does (verified live).
#[test]
fn all_turns_reasoning_context_goes_where_documented_or_verified() {
    let context = |model: Model, oauth: bool, config: &GenerationConfig| {
        let request = OpenAIClient::new("test")
            .build_request(&Conversation::new(model), config, oauth)
            .unwrap();
        serde_json::to_value(request).unwrap()["reasoning"]["context"]
            .as_str()
            .map(String::from)
    };
    let effort = GenerationConfig::default().with_reasoning(
        crate::reasoning::ReasoningConfig::openai(OpenAIEffort::High, false),
    );
    for config in [GenerationConfig::default(), effort] {
        for model in [
            Model::openai_gpt_5_6_sol(),
            Model::openai_gpt_5_6_luna(),
            Model::openai_gpt_5_6(),
        ] {
            for oauth in [false, true] {
                assert_eq!(
                    context(model.clone(), oauth, &config).as_deref(),
                    Some("all_turns"),
                    "{} oauth={oauth}",
                    model.id
                );
            }
        }
        for model in [
            Model::openai_gpt_6_astra(),
            Model::openai_gpt_6_sol(),
            Model::openai_gpt_6_luna(),
        ] {
            assert_eq!(
                context(model.clone(), true, &config).as_deref(),
                Some("all_turns"),
                "{} OAuth",
                model.id
            );
            assert_eq!(
                context(model.clone(), false, &config),
                None,
                "{} API key",
                model.id
            );
        }
        for model in [Model::openai_gpt_5_5(), Model::openai_gpt_5_4()] {
            for oauth in [false, true] {
                assert_eq!(
                    context(model.clone(), oauth, &config),
                    None,
                    "{} oauth={oauth}",
                    model.id
                );
            }
        }
    }
    // GPT-6 over an API key without an effort or mode sends no reasoning object.
    let request = OpenAIClient::new("test")
        .build_request(
            &Conversation::new(Model::openai_gpt_6_astra()),
            &GenerationConfig::default(),
            false,
        )
        .unwrap();
    assert!(
        serde_json::to_value(request)
            .unwrap()
            .get("reasoning")
            .is_none()
    );
}
