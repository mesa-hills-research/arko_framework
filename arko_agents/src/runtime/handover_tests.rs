//! In-place conversation handover: the report request, the report turn,
//! and the fresh conversation.

use super::image_tests::{call, harness};
use super::*;
use crate::handover::{DEFAULT_REPORT_REQUEST, HandoverConfig, REPORT_HEADING};
use futures::StreamExt;

/// A scripted response with `input_tokens` of input. Output-based text is
/// streamed too, since the parser reads the stream.
fn scripted(message: ai::Message, input_tokens: usize) -> ScriptedStreamResult {
    let stop_reason = if message.has_tool_calls() {
        ai::StopReason::ToolUse
    } else {
        ai::StopReason::EndTurn
    };
    let text = message_text(&message);
    let end = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(input_tokens, 10),
        model_call_usage: ai::ModelCallUsage::new(
            "handover".into(),
            "test".into(),
            "test".into(),
            input_tokens,
            10,
            1000,
        ),
        stop_reason,
        refusal: None,
        openai_turn_state: None,
    }));
    let mut events = Vec::new();
    if !text.is_empty() {
        events.push(Ok(ai::StreamEvent::TextChunk(text)));
    }
    events.push(Ok(end));
    Ok(futures::stream::iter(events).boxed())
}

fn native_call(name: &str, input_tokens: usize) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(call(name, name));
    scripted(message, input_tokens)
}

fn native_complete() -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    let mut complete = call("complete", "complete_task");
    complete.arguments = serde_json::json!({"summary": "done"});
    message.tool_calls.push(complete);
    scripted(message, 0)
}

fn reply(text: &str, input_tokens: usize) -> ScriptedStreamResult {
    scripted(ai::Message::new_assistant(text), input_tokens)
}

fn push(agent: &mut AgentLoop, responses: Vec<ScriptedStreamResult>) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend(responses);
}

async fn run(agent: AgentLoop, events: &mut mpsc::Receiver<AgentEvent>) -> (Task, Vec<AgentEvent>) {
    let outcome = timeout(Duration::from_secs(5), agent.run()).await;
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    let task = outcome
        .unwrap_or_else(|error| panic!("{error}: {received:#?}"))
        .unwrap();
    (task, received)
}

fn texts(task: &Task) -> Vec<String> {
    task.conversation
        .messages
        .iter()
        .map(message_text)
        .collect()
}

#[tokio::test]
async fn a_conversation_over_the_threshold_is_handed_over_in_place() {
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::Native);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    let task_id = agent.task.id;
    push(
        &mut agent,
        vec![
            // Under the threshold: no handover.
            native_call("plain_text", 500),
            // Over it: the next request asks for the report.
            native_call("plain_text", 1_500),
            reply("Done: read the fixture. Next: finish.", 1_600),
            // The fresh conversation, small again.
            native_call("plain_text", 200),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;

    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.id, task_id);
    assert_eq!(task.handovers.len(), 1);
    let record = &task.handovers[0];
    assert_eq!(record.input_tokens, 1_500);
    assert_eq!(record.report, "Done: read the fixture. Next: finish.");
    assert!(record.replaced_messages >= 6, "{record:?}");
    assert!(task.handover_requested.is_none());
    // Three calls before the handover, two after it.
    assert_eq!(task.conversation_usage_start, 3);
    assert_eq!(task.usage_history.len(), 5);

    let texts = texts(&task);
    let first = &texts[0];
    assert!(first.starts_with("Inspect fixture image\n\n"), "{first}");
    assert!(first.contains(REPORT_HEADING));
    assert!(first.contains("Done: read the fixture. Next: finish."));
    assert!(
        !texts
            .iter()
            .any(|text| text.contains(DEFAULT_REPORT_REQUEST)),
        "the report request went with the old conversation"
    );

    let requested: Vec<usize> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::HandoverRequested { input_tokens } => Some(*input_tokens),
            _ => None,
        })
        .collect();
    assert_eq!(requested, [1_500]);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationHandedOver { input_tokens: 1_500, handovers: 1, report }
            if report == "Done: read the fixture. Next: finish."
    )));
    // The usage delta doesn't compare the fresh conversation with the old.
    let first_fresh = &task.model_call_usage_history[task.conversation_model_call_start];
    assert_eq!(first_fresh.input_delta_from_previous_call, None);
}

#[tokio::test]
async fn tool_calls_in_the_report_turn_are_rejected_except_complete_task() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    // No threshold: the host asks.
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            native_call("plain_text", 100),
            reply("Nothing done yet.", 120),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;

    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].input_tokens, 0);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolRejected { tool_name, reason: Some(reason), .. }
            if tool_name == "plain_text" && reason.contains("handover report is due")
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolResult { tool_name, .. } if tool_name == "plain_text"))
    );
}

#[tokio::test]
async fn complete_task_during_the_report_turn_completes_the_task() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(&mut agent, vec![native_complete()]);
    let (task, _events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.handovers.is_empty());
}

#[tokio::test]
async fn a_reply_without_text_gets_a_note_and_the_report_is_asked_again() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![reply("", 10), reply("The report.", 20), native_complete()],
    );
    let (task, _events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "The report.");
}

#[tokio::test]
async fn output_based_agents_hand_over_from_their_plain_text() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::OutputBased);
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            reply("<plain_text></plain_text>", 10),
            reply("Report in text mode.", 20),
            reply("<complete_task><summary>done</summary></complete_task>", 5),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "Report in text mode.");
    assert!(texts(&task)[0].contains("Report in text mode."));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolRejected { tool_name, .. } if tool_name == "plain_text"
    )));
}

struct RosterHook {
    seen_messages: std::sync::Mutex<Option<usize>>,
}

impl HandoverHook for RosterHook {
    fn on_handover(&self, task: &Task, report: &str) -> HandoverSeed {
        assert_eq!(report, "Hooked report.");
        *self.seen_messages.lock().unwrap() = Some(task.conversation.messages.len());
        HandoverSeed {
            task_prompt: "Restated task".into(),
            context: Some("# Agents you can message\n- `lead`".into()),
        }
    }
}

#[tokio::test]
async fn the_hook_sees_the_old_conversation_and_sets_the_seed() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    let hook = Arc::new(RosterHook {
        seen_messages: Default::default(),
    });
    agent.handover_hook = Some(hook.clone());
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![reply("Hooked report.", 10), native_complete()],
    );
    let (task, _events) = run(agent, &mut events).await;
    // The task prompt, the report request and the report.
    assert_eq!(*hook.seen_messages.lock().unwrap(), Some(3));
    let first = &texts(&task)[0];
    assert!(
        first.starts_with("Restated task\n\n# Handover report"),
        "{first}"
    );
    assert!(first.contains("Hooked report.\n\n# Agents you can message\n- `lead`"));
    // The runtime still frames it as a task prompt; the saved initial
    // prompt is unchanged.
    assert_eq!(task.initial_task_prompt, "Inspect fixture image");
}

#[test]
fn tasks_saved_before_handovers_load() {
    let (agent, _events, _commands) = harness(ToolCallingMode::Native);
    let mut saved = serde_json::to_value(&agent.task).unwrap();
    let object = saved.as_object_mut().unwrap();
    for field in [
        "handover_requested",
        "handovers",
        "conversation_usage_start",
        "conversation_model_call_start",
    ] {
        assert!(!object.contains_key(field), "{field} is written by default");
    }
    let task: Task = serde_json::from_value(saved).unwrap();
    assert!(task.handovers.is_empty());
    assert_eq!(task.conversation_usage_start, 0);
    assert!(task.agent_config.handover.is_none());
}
