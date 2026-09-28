//! Recovery follow-ups: attempt accounting across capacity waits, partial
//! output of failed streams, and conversations that end on an assistant turn.

use super::retry_tests::{capacity_failure, complete_response, harness, provider_failure};
use super::*;
use futures::StreamExt;

fn events_of(events: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    received
}

fn assistant_texts(task: &Task) -> Vec<String> {
    task.conversation
        .messages
        .iter()
        .filter(|message| message.role == ai::Role::Assistant)
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_text)
                .collect()
        })
        .collect()
}

/// Capacity failures between ordinary ones neither reset nor consume the
/// ordinary limits: the attempt limit still ends recovery, and hours of
/// capacity waiting don't use up the ordinary elapsed budget.
#[tokio::test(start_paused = true)]
async fn alternating_capacity_and_ordinary_failures_still_exhaust_the_attempt_limit() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
    agent.task.agent_config.llm_retry_config.max_elapsed_ms = 10_000;
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 3 };
    for round in 1..=3_u32 {
        agent
            .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
            .await
            .unwrap();
        assert_eq!(agent.ordinary_attempts(), round);
        if round == 3 {
            break;
        }
        assert!(matches!(agent.task.current_state, TaskState::InProgress));
        agent
            .run_llm_attempt::<()>(async { Err(capacity_failure(Some(Duration::from_secs(3600)))) })
            .await
            .unwrap();
        assert!(matches!(agent.task.current_state, TaskState::InProgress));
    }
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { ref error_message }
            if error_message.contains("attempt limit exhausted")
    ));
    let delays: Vec<u64> = events_of(&mut events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::RetryScheduled { delay_ms, .. } => Some(delay_ms),
            _ => None,
        })
        .collect();
    // Ordinary backoff continues from attempt to attempt (250, 500 ms);
    // capacity waits honor the hour-long hint in between.
    assert_eq!(delays, [250, 3_600_000, 500, 3_600_000]);
}

/// A stream that fails with a capacity error under opt-in capacity recovery
/// is retried unchanged: its partial output is not kept (it would end the
/// retried request on an assistant turn), and no message is added.
#[tokio::test(start_paused = true)]
async fn a_capacity_retry_after_partial_output_resends_the_request_unchanged() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(Ok(futures::stream::iter([
        Ok(ai::StreamEvent::TextChunk("partial answer".into())),
        Err(capacity_failure(None)),
    ])
    .boxed()));
    scripted.push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
    let task = timeout(Duration::from_secs(3600), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(assistant_texts(&task), ["Finished after recovery."]);
    // The task prompt, the completed turn and its tool result.
    let roles: Vec<_> = task
        .conversation
        .messages
        .iter()
        .map(|m| m.role.clone())
        .collect();
    assert_eq!(roles, [ai::Role::User, ai::Role::Assistant, ai::Role::User]);
}

/// A failed request (here a 503 after some output, then a truncated
/// stream) is sent again unchanged: the failed stream's output is not kept
/// and no note about the failure is added, since the model had no part in
/// it.
#[tokio::test(start_paused = true)]
async fn a_failed_request_is_sent_again_unchanged_without_a_note() {
    for failure in [
        provider_failure(503, None),
        ai::Error::StreamTruncated("no data arrived for 1s".into()),
    ] {
        let (mut agent, mut events, _commands) = harness();
        agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
        agent
            .task
            .agent_config
            .tool_config
            .default_auto_approve_tool_use = true;
        let scripted = agent.scripted_responses.get_mut().unwrap();
        scripted.push_back(Ok(futures::stream::iter([
            Ok(ai::StreamEvent::TextChunk("partial answer".into())),
            Err(failure),
        ])
        .boxed()));
        scripted.push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
        let task = timeout(Duration::from_secs(60), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        // The task prompt, the completed turn and its tool result.
        let roles: Vec<_> = task
            .conversation
            .messages
            .iter()
            .map(|m| m.role.clone())
            .collect();
        assert_eq!(roles, [ai::Role::User, ai::Role::Assistant, ai::Role::User]);
        assert!(!assistant_texts(&task).contains(&"partial answer".to_string()));
        assert!(task.runtime_messages.is_empty());
        // The failure is still reported to the host.
        let events = events_of(&mut events);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::RetryScheduled { attempt: 1, .. }))
        );
    }
}

/// A response the runtime can't use gets a note before the next request:
/// a native turn without a tool call is told how to go on. It is asked
/// again at once, without backoff, and without starting the elapsed
/// recovery budget (the model's work is not an outage): an hour-long
/// response afterwards is not cut off.
#[tokio::test(start_paused = true)]
async fn a_response_without_a_tool_call_gets_a_note() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(super::image_tests::response(ai::Message::new_assistant(
        "I think the task is done.",
    )));
    // The next response takes an hour.
    scripted.push_back(Ok(futures::stream::once(async {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok(complete_response())
    })
    .boxed()));
    let started = tokio::time::Instant::now();
    let task = timeout(Duration::from_secs(7200), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert!(started.elapsed() >= Duration::from_secs(3600));
    let delays: Vec<u64> = events_of(&mut events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::RetryScheduled { delay_ms, .. } => Some(delay_ms),
            _ => None,
        })
        .collect();
    assert_eq!(delays, [0]);
    let note = &task.conversation.messages[2];
    assert_eq!(note.role, ai::Role::User);
    assert_eq!(note.content[0].as_text(), Some(NO_NATIVE_TOOL_CALL));
    assert!(task.runtime_messages.contains(&2));
}

/// A task resumed while its conversation ends on assistant text (output
/// kept from an interrupted stream) asks the model to continue rather than
/// sending a request that ends on an assistant turn.
#[tokio::test(start_paused = true)]
async fn a_resumed_conversation_ending_on_assistant_text_gets_a_user_turn() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    agent.task.conversation.system_message = Some("resumed".into());
    agent.task.conversation.add_user_message("task");
    agent
        .task
        .conversation
        .add_assistant_message("interrupted output");
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let continuation = &task.conversation.messages[2];
    assert_eq!(continuation.role, ai::Role::User);
    let text: String = continuation
        .content
        .iter()
        .filter_map(ai::Content::as_text)
        .collect();
    assert!(text.contains("Continue the task"), "{text}");
}

/// Calls a native stream announced before it failed are reported as not
/// run, with the ids their `ToolCallInitiated` events carried.
#[tokio::test(start_paused = true)]
async fn calls_announced_by_a_failed_stream_are_reported_as_rejected() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(Ok(futures::stream::iter([
        Ok(ai::StreamEvent::ToolCallStart {
            index: 0,
            name: "complete_task".into(),
        }),
        Err(provider_failure(503, None)),
    ])
    .boxed()));
    scripted.push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let events = events_of(&mut events);
    let initiated: Vec<Uuid> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallInitiated { tool_call_id, .. } => Some(*tool_call_id),
            _ => None,
        })
        .collect();
    let rejected: Vec<Uuid> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolRejected {
                tool_call_id,
                reason: Some(reason),
                ..
            } if reason == FAILED_STREAM_CALL => Some(*tool_call_id),
            _ => None,
        })
        .collect();
    // The scripted completion carries its call without announcing it.
    assert_eq!(initiated.len(), 1);
    assert_eq!(rejected, initiated);
}

/// Resumed tasks price calls with the registry's current data for their
/// model, not the copy saved with the task.
#[test]
fn a_resumed_task_takes_current_model_data_from_the_registry() {
    let (agent, _events, _commands) = harness();
    let mut task = agent.task;
    task.conversation.model.pricing.input_cost_per_million = 999.0;
    task.conversation.model.max_input_tokens = 1;
    let restored: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
    let current = ai::Model::openai_gpt_5_6_luna();
    assert_eq!(
        restored.conversation.model.pricing.input_cost_per_million,
        current.pricing.input_cost_per_million
    );
    assert_eq!(
        restored.conversation.model.max_input_tokens,
        current.max_input_tokens
    );
}

/// A response that ends at the output-token limit is reported, so a host
/// can tell the user; the turn is then handled as usual.
#[tokio::test(start_paused = true)]
async fn a_response_cut_at_the_output_limit_is_reported() {
    let (mut agent, mut events, _commands) = harness();
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let cut = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message: ai::Message::new_assistant("Here is the first half of the fi"),
        usage: ai::UsageInfo::new(0, 128_000),
        model_call_usage: ai::ModelCallUsage::new(
            "cut".into(),
            "test".into(),
            "test".into(),
            0,
            128_000,
            1000,
        ),
        stop_reason: ai::StopReason::MaxTokens,
        refusal: None,
        openai_turn_state: None,
    }));
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(Ok(futures::stream::iter([Ok(cut)]).boxed()));
    scripted.push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let reported: Vec<_> = events_of(&mut events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::OutputTokenLimitReached { output_tokens, .. } => Some(output_tokens),
            _ => None,
        })
        .collect();
    assert_eq!(reported, [128_000]);
}

/// Under `InformAgentAndContinue` (no attempt limit), responses without a
/// tool call are asked again at once but not forever: they get
/// `MODEL_OUTPUT_ATTEMPTS_WITHOUT_LIMIT` attempts, then the task is
/// returned interrupted.
#[tokio::test(start_paused = true)]
async fn responses_without_a_tool_call_are_bounded_without_an_attempt_limit() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent.task.agent_config.llm_error_policy = LLMErrorPolicy::InformAgentAndContinue;
    let scripted = agent.scripted_responses.get_mut().unwrap();
    for _ in 0..50 {
        scripted.push_back(super::image_tests::response(ai::Message::new_assistant(
            "Just chatting.",
        )));
    }
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(task.current_state, TaskState::Interrupted { ref reason } if reason.contains("attempt limit")),
        "{:?}",
        task.current_state
    );
    assert_eq!(
        task.usage_history.len(),
        MODEL_OUTPUT_ATTEMPTS_WITHOUT_LIMIT as usize
    );
}

/// A completed response ends an outage's elapsed budget: after a failed
/// request, a response without a tool call and a slow next response are
/// not cut off by the budget the failure started.
#[tokio::test(start_paused = true)]
async fn a_completed_response_ends_the_elapsed_budget_of_a_failure() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(Err(provider_failure(503, None)));
    scripted.push_back(super::image_tests::response(ai::Message::new_assistant(
        "Thinking out loud.",
    )));
    scripted.push_back(Ok(futures::stream::once(async {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok(complete_response())
    })
    .boxed()));
    let task = timeout(Duration::from_secs(7200), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
}
