//! Refusal recovery: the refused input is reported, the host changes it and
//! the task continues; the opt-in fallback model; announced calls and
//! output-based markup of refused turns.

use super::retry_tests::{
    complete_response, harness, model_refusal, refused_response, run_refused_turn,
};
use super::*;
use crate::agent_config::AgentLLMConfig;
use crate::task::{RefusalResolution, RefusedInput, RefusedToolResult};
use futures::StreamExt;

fn native(agent: &mut AgentLoop) {
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent.task.agent_config.tool_config.default_auto_approve_tool_use = true;
}

fn push(agent: &mut AgentLoop, turn: ai::StreamEvent) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(turn)]).boxed()));
}

fn refused() -> ai::StreamEvent {
    refused_response(ai::Message::new_assistant(""), model_refusal())
}

/// A native turn calling a tool the registry doesn't have: its error result
/// is the next input.
fn missing_tool_turn() -> ai::StreamEvent {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(ai::ToolCall {
        id: Some("call-missing".into()),
        name: "missing_tool".into(),
        arguments: serde_json::json!({}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    });
    ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new("t".into(), "openai".into(), "m".into(), 0, 0, 1000),
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }))
}

/// Runs the agent, answering the first refusal pause with `resolutions` in
/// order (a failed one leaves the task waiting for the next), and returns
/// the task and every event.
async fn run_resolving(
    agent: AgentLoop,
    mut events: mpsc::Receiver<AgentEvent>,
    commands: mpsc::Sender<UserCommand>,
    resolutions: Vec<RefusalResolution>,
) -> (Task, Vec<AgentEvent>) {
    let runner = tokio::spawn(agent.run());
    let mut resolutions = resolutions.into_iter();
    let mut received = Vec::new();
    loop {
        let event = timeout(Duration::from_secs(60), events.recv())
            .await
            .unwrap()
            .unwrap();
        received.push(event.clone());
        match event {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            | AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } if received
                .iter()
                .any(|event| matches!(event, AgentEvent::RefusedInput(_))) =>
            {
                if let Some(resolution) = resolutions.next() {
                    commands
                        .send(UserCommand::ResolveRefusal { resolution })
                        .await
                        .unwrap();
                }
            }
            AgentEvent::TaskStreamEnd => break,
            _ => {}
        }
    }
    (runner.await.unwrap().unwrap(), received)
}

fn refused_inputs(events: &[AgentEvent]) -> Vec<RefusedInput> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RefusedInput(input) => Some(input.clone()),
            _ => None,
        })
        .collect()
}

fn texts(message: &ai::Message) -> String {
    message.content.iter().filter_map(ai::Content::as_text).collect()
}

#[tokio::test(start_paused = true)]
async fn a_refused_user_message_is_reported_and_can_be_rephrased() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(
        agent,
        events,
        commands,
        vec![RefusalResolution::RephraseUserMessage {
            text: "A rephrased task.".into(),
        }],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }), "{:?}", task.current_state);
    assert_eq!(refused_inputs(&events), [RefusedInput::UserMessage { message_index: 0 }]);
    // The refused turn is gone and the rephrased prompt was answered.
    let messages = &task.conversation.messages;
    assert_eq!(texts(&messages[0]), "A rephrased task.");
    assert_eq!(texts(&messages[1]), "Finished after recovery.");
    assert!(!messages.iter().any(|message| texts(message).contains("can't help")));
    assert!(task.refused_input.is_none());
    assert!(!events.iter().any(|event| matches!(event, AgentEvent::RetryScheduled { .. })));
}

#[tokio::test(start_paused = true)]
async fn refused_tool_results_can_be_redacted_and_an_invalid_resolution_keeps_waiting() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    push(&mut agent, missing_tool_turn());
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(
        agent,
        events,
        commands,
        vec![
            // Tool results can't be removed like a user message.
            RefusalResolution::RemoveUserMessage,
            RefusalResolution::RedactToolResults {
                call_ids: vec!["call-missing".into()],
                replacement: "[output removed]".into(),
            },
        ],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }), "{:?}", task.current_state);
    assert_eq!(
        refused_inputs(&events),
        [RefusedInput::ToolResults {
            message_index: 2,
            results: vec![RefusedToolResult {
                call_id: Some("call-missing".into()),
                tool_name: "missing_tool".into(),
            }],
        }]
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { error_message, .. }
            if error_message.contains("Could not resolve the refusal") && error_message.contains("redact")
    )));
    let result = task.conversation.messages[2].content[0].as_tool_result().unwrap();
    assert_eq!(
        (result.call_id.as_deref(), result.output.as_str()),
        (Some("call-missing"), "[output removed]")
    );
    assert_eq!(texts(&task.conversation.messages[3]), "Finished after recovery.");
}

/// Under an informing error policy the refused task is returned to the
/// host, which resolves the refusal on the task and runs it again.
#[tokio::test(start_paused = true)]
async fn a_returned_task_can_be_resolved_and_run_again() {
    let (task, _, events) = run_refused_turn(ToolCallingMode::Native, vec![refused()]).await;
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert_eq!(refused_inputs(&events), [RefusedInput::UserMessage { message_index: 0 }]);
    assert_eq!(task.refused_input, Some(RefusedInput::UserMessage { message_index: 0 }));
    // Saved and loaded, the refused input survives.
    let mut task: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
    // The prompt is the only user message: it can be rephrased, not removed.
    assert!(task.resolve_refusal(RefusalResolution::RemoveUserMessage).is_err());
    assert!(
        task.resolve_refusal(RefusalResolution::RedactToolResults {
            call_ids: vec!["x".into()],
            replacement: String::new(),
        })
        .is_err()
    );
    task.resolve_refusal(RefusalResolution::RephraseUserMessage {
        text: "Try this instead.".into(),
    })
    .unwrap();
    assert!(task.refused_input.is_none());
    assert!(task.resolve_refusal(RefusalResolution::RemoveUserMessage).is_err(), "resolved once");

    let (mut agent, _events, _commands) = harness();
    agent.task = task;
    native(&mut agent);
    push(&mut agent, complete_response());
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(texts(&task.conversation.messages[0]), "Try this instead.");

    // Tasks saved before refused inputs existed load without one.
    let mut saved = serde_json::to_value(&task).unwrap();
    saved.as_object_mut().unwrap().remove("refused_input");
    let loaded: Task = serde_json::from_value(saved).unwrap();
    assert!(loaded.refused_input.is_none());
}

#[tokio::test(start_paused = true)]
async fn a_refused_user_message_after_earlier_turns_can_be_removed() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    agent.task.conversation.system_message = Some("resumed".into());
    agent.task.conversation.add_user_message("first task");
    agent.task.conversation.add_assistant_message("done");
    agent.task.conversation.add_user_message("objectionable request");
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(agent, events, commands, vec![RefusalResolution::RemoveUserMessage]).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(refused_inputs(&events), [RefusedInput::UserMessage { message_index: 2 }]);
    let texts: Vec<String> = task.conversation.messages.iter().map(texts).collect();
    assert!(!texts.iter().any(|text| text.contains("objectionable")), "{texts:?}");
    // The conversation ended on an assistant turn, so the model was asked
    // to continue.
    assert!(texts[2].contains("Continue the task"), "{texts:?}");
}

#[tokio::test(start_paused = true)]
async fn an_opt_in_fallback_model_answers_a_refused_turn_once() {
    let (mut agent, mut events, _commands) = harness();
    native(&mut agent);
    agent.task.agent_config.refusal_fallback = Some(AgentLLMConfig {
        model: ai::Model::openai_gpt_6_sol(),
        generation_config: Default::default(),
    });
    let requested = Arc::clone(&agent.requested_models);
    push(&mut agent, refused());
    push(&mut agent, missing_tool_turn());
    push(&mut agent, complete_response());
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // The refused turn went to the fallback once; later turns use the
    // configured model again.
    assert_eq!(
        *requested.lock().unwrap(),
        ["gpt-5.6-luna", "gpt-6-sol", "gpt-5.6-luna"]
    );
    assert!(!task.conversation.messages.iter().any(|message| texts(message).contains("can't help")));
    assert!(task.refused_input.is_none());
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    assert!(received.iter().any(|event| matches!(
        event,
        AgentEvent::RefusalFallback { model_id } if model_id == "gpt-6-sol"
    )));
    assert!(refused_inputs(&received).is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_refusal_from_the_fallback_model_stops_the_task() {
    let (mut agent, mut events, _commands) = harness();
    native(&mut agent);
    agent.task.agent_config.llm_error_policy = LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    agent.task.agent_config.refusal_fallback = Some(AgentLLMConfig {
        model: ai::Model::openai_gpt_6_sol(),
        generation_config: Default::default(),
    });
    let requested = Arc::clone(&agent.requested_models);
    push(&mut agent, refused());
    push(&mut agent, refused());
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert_eq!(*requested.lock().unwrap(), ["gpt-5.6-luna", "gpt-6-sol"]);
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    assert_eq!(refused_inputs(&received), [RefusedInput::UserMessage { message_index: 0 }]);
}

/// Low (a): calls a native stream announced before the refusal get
/// `ToolRejected("turn refused")`.
#[tokio::test(start_paused = true)]
async fn calls_announced_before_a_native_refusal_are_rejected() {
    let (task, _, events) = run_refused_turn(
        ToolCallingMode::Native,
        vec![
            ai::StreamEvent::ToolCallStart {
                index: 0,
                name: "complete_task".into(),
            },
            ai::StreamEvent::Refusal(model_refusal()),
            refused(),
        ],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    let initiated: Vec<Uuid> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallInitiated { tool_call_id, .. } => Some(*tool_call_id),
            _ => None,
        })
        .collect();
    let rejected: Vec<(Uuid, Option<String>)> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolRejected {
                tool_call_id,
                reason,
                ..
            } => Some((*tool_call_id, reason.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(initiated.len(), 1);
    assert_eq!(rejected, [(initiated[0], Some("turn refused".to_string()))]);
}

/// Low (b): an output-based refused turn keeps only the refusal
/// explanation, or without one its text minus the tool-call markup that
/// was never run; OpenAI message items holding that markup are not replayed.
#[tokio::test(start_paused = true)]
async fn output_based_refusals_keep_no_tool_call_markup() {
    let markup = "Sure. <complete_task><summary>done</summary></complete_task>";
    for explanation in [Some("I can't help with that."), None] {
        let mut refusal = model_refusal();
        refusal.explanation = explanation.map(String::from);
        let mut message = ai::Message::new_assistant(markup);
        message.reasoning_artifact = Some(ai::ReasoningArtifact::OpenAIOutputItems {
            items: vec![
                serde_json::json!({"type":"reasoning", "id":"rs_1", "summary":[], "encrypted_content":"enc"}),
                serde_json::json!({"type":"message", "id":"msg_1", "role":"assistant",
                    "content":[{"type":"output_text", "text": markup}]}),
            ],
        });
        let (task, messages_before, events) = run_refused_turn(
            ToolCallingMode::OutputBased,
            vec![
                ai::StreamEvent::TextChunk(markup.into()),
                refused_response(message, refusal),
            ],
        )
        .await;
        assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
        let kept: Vec<&ai::Message> = task.conversation.messages[messages_before..]
            .iter()
            .filter(|message| message.role == ai::Role::Assistant)
            .collect();
        assert_eq!(kept.len(), 1);
        let expected = explanation.unwrap_or("Sure. ");
        assert_eq!(texts(kept[0]), expected);
        let Some(ai::ReasoningArtifact::OpenAIOutputItems { items }) = &kept[0].reasoning_artifact else {
            panic!("the reasoning item is kept");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "reasoning");
        assert!(!events.iter().any(|event| matches!(event, AgentEvent::ToolResult { .. })));
    }
}
