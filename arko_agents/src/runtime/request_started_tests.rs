//! `ModelRequestStarted`: each model request is reported as it goes out,
//! with its size by section and the bytes per token its predecessors
//! calibrated.

use super::tool_call_tests::{agent_for, config, drain, native_call};
use super::*;

/// A response holding `message`, whose request the provider counted as
/// `input_tokens` input tokens.
fn answered(message: ai::Message, input_tokens: usize) -> ScriptedStreamResult {
    let stop_reason = if message.has_tool_calls() {
        ai::StopReason::ToolUse
    } else {
        ai::StopReason::EndTurn
    };
    let response = ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(input_tokens, 5),
        model_call_usage: ai::ModelCallUsage::new(
            "call".into(),
            "test".into(),
            "test".into(),
            input_tokens,
            5,
            1_000_000,
        ),
        stop_reason,
        refusal: None,
        openai_turn_state: None,
    };
    Ok(Box::pin(futures::stream::iter([Ok(ai::StreamEvent::End(
        Box::new(response),
    ))])))
}

fn with_calls(calls: Vec<ai::ToolCall>) -> ai::Message {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls = calls;
    message
}

/// Reported before each request, in order with the responses: the first
/// at the default ratio, the next at the ratio the first one's reported
/// input calibrated. Sizes count what each request holds.
#[tokio::test]
async fn each_request_is_reported_as_it_goes_out() {
    let task = Task::new(
        config(ToolCallingMode::Native, &["plain_text"]),
        "Do it.".into(),
        std::env::temp_dir(),
        None,
    );
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(super::image_tests::TextTool));
    let (mut agent, mut events, _commands) =
        agent_for(task, registry, Arc::new(ai::LLMClient::new()));
    agent.scripted_responses.get_mut().unwrap().extend([
        answered(
            with_calls(vec![native_call("c1", "plain_text", serde_json::json!({}))]),
            1_000,
        ),
        answered(
            with_calls(vec![native_call(
                "c2",
                "complete_task",
                serde_json::json!({"summary": "done"}),
            )]),
            1_500,
        ),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let events = drain(&mut events);

    let started: Vec<(usize, &AgentEvent)> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, AgentEvent::ModelRequestStarted { .. }))
        .collect();
    let usages: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, AgentEvent::UsageReported(_)))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(started.len(), 2, "{events:#?}");
    assert!(started[0].0 < usages[0] && usages[0] < started[1].0 && started[1].0 < usages[1]);

    let request = |index: usize| match started[index].1 {
        AgentEvent::ModelRequestStarted {
            model,
            subagent,
            sizes,
            bytes_per_token,
            ..
        } => (
            model.clone(),
            subagent.clone(),
            sizes.unwrap(),
            *bytes_per_token,
        ),
        _ => unreachable!(),
    };
    let (model, subagent, first, ratio) = request(0);
    assert_eq!(model, task.conversation.model.id);
    assert_eq!(subagent, None);
    assert_eq!(ratio, result_budget::DEFAULT_BYTES_PER_TOKEN);
    let system = task.conversation.system_message.as_ref().unwrap();
    assert_eq!(first.system_prompt, system.len());
    assert!(first.tool_definitions > 0);
    assert!(first.user_text > 0);
    assert_eq!((first.tool_calls, first.tool_results), (0, 0));

    let (_, _, second, ratio) = request(1);
    assert_eq!(
        Some(ratio),
        result_budget::calibrated_bytes_per_token(first.text_bytes(), 1_000)
    );
    assert_eq!(second.system_prompt, first.system_prompt);
    assert!(second.tool_calls > 0 && second.tool_results > 0);
    assert!(second.text_bytes() > first.text_bytes());
}

/// The provider counts a request's images in its input tokens, and the
/// text bytes don't hold them: the ratio is calibrated from what the
/// images' estimated tokens leave of the input.
#[tokio::test]
async fn the_ratio_leaves_the_images_out() {
    let mut task = Task::new(
        config(ToolCallingMode::Native, &["plain_text"]),
        "Look.".into(),
        std::env::temp_dir(),
        None,
    );
    // A PNG header of 1,500 × 750 pixels: 47 × 24 patches of 32 pixels,
    // times 1.2, for the test's OpenAI model.
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
    png.extend(1_500u32.to_be_bytes());
    png.extend(750u32.to_be_bytes());
    task.initial_task_images = vec![ai::Image::from_bytes(&png, "image/png")];
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(super::image_tests::TextTool));
    let (mut agent, mut events, _commands) =
        agent_for(task, registry, Arc::new(ai::LLMClient::new()));
    agent.scripted_responses.get_mut().unwrap().extend([
        answered(
            with_calls(vec![native_call("c1", "plain_text", serde_json::json!({}))]),
            2_500,
        ),
        answered(
            with_calls(vec![native_call(
                "c2",
                "complete_task",
                serde_json::json!({"summary": "done"}),
            )]),
            3_000,
        ),
    ]);
    timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    let started: Vec<_> = drain(&mut events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted {
                sizes,
                bytes_per_token,
                ..
            } => Some((sizes.unwrap(), bytes_per_token)),
            _ => None,
        })
        .collect();
    let (first, _) = started[0];
    assert_eq!(first.image_tokens, 1_354);
    assert_eq!(
        Some(started[1].1),
        result_budget::calibrated_bytes_per_token(first.text_bytes(), 2_500 - 1_354)
    );
}
