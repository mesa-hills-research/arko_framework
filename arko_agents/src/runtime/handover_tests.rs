//! In-place conversation handover: the report request, the report turn,
//! and the fresh conversation.

use super::image_tests::{call, harness};
use super::*;
use crate::handover::{
    DEFAULT_REPORT_REQUEST, DEFAULT_TEXT_REPORT_REQUEST, HandoverConfig, HandoverThreshold,
    REPORT_HEADING,
};
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

/// A `handover_report` call with `report`, after the calls named in
/// `before` (each called with its name as its id).
fn native_report(before: &[&str], report: &str, input_tokens: usize) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    for name in before {
        message.tool_calls.push(call(name, name));
    }
    let mut report_call = call("report", HANDOVER_REPORT_TOOL_NAME);
    report_call.arguments = serde_json::json!({ "report": report });
    message.tool_calls.push(report_call);
    scripted(message, input_tokens)
}

/// A native agent with `handover` set, which gives its report with the
/// `handover_report` tool. The threshold is out of reach: the tests ask for
/// the handover with a command.
fn tool_harness() -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let (mut agent, events, commands) = harness(ToolCallingMode::Native);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000_000));
    (agent, events, commands)
}

/// Why the calls to `tool` were rejected, from the events (the replaced
/// conversation that held their results is gone).
fn rejections(events: &[AgentEvent], tool: &str) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolRejected {
                tool_name,
                reason: Some(reason),
                ..
            } if tool_name == tool => Some(reason.clone()),
            _ => None,
        })
        .collect()
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
    let chart = ai::Image::new_base64("Y2hhcnQ=", "image/png");
    agent.task.initial_task_images = vec![chart.clone()];
    let task_id = agent.task.id;
    push(
        &mut agent,
        vec![
            // Under the threshold: no handover.
            native_call("plain_text", 500),
            // Over it: the next request asks for the report.
            native_call("plain_text", 1_500),
            native_report(&[], "Done: read the fixture. Next: finish.", 1_600),
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

    // The task's image restated with it, held once.
    assert_eq!(
        task.conversation.messages[0].content[1].as_image(),
        Some(&chart)
    );
    assert!(task.initial_task_images.is_empty());
    assert_eq!(task.initial_images(), [chart]);
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

/// With `handover` set, native requests offer `handover_report` from the
/// first one, so the tool list is the same when the report is asked for.
/// Without it, or for output-based agents, the tool isn't offered.
#[tokio::test]
async fn the_report_tool_is_offered_with_handover_set() {
    let offers = |agent: &AgentLoop| {
        agent
            .build_native_tool_config(&agent.task.conversation.model, None)
            .tools
            .iter()
            .any(|tool| tool.name == HANDOVER_REPORT_TOOL_NAME)
    };
    let (agent, _events, _commands) = harness(ToolCallingMode::Native);
    assert!(!agent.offers_report_tool());
    assert!(!offers(&agent));
    let (agent, _events, _commands) = tool_harness();
    assert!(agent.offers_report_tool());
    assert!(offers(&agent));
    let (mut agent, _events, _commands) = harness(ToolCallingMode::OutputBased);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    assert!(!agent.offers_report_tool());
}

/// The report request asks for `handover_report`, the turn's other calls
/// are rejected with a result that asks for it too, and the report call's
/// argument is the report the fresh conversation starts from.
#[tokio::test]
async fn the_report_comes_from_the_report_tool() {
    let (mut agent, mut events, commands) = tool_harness();
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            native_report(&["plain_text"], "  Tool report: nothing done yet.  ", 100),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;

    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    let record = &task.handovers[0];
    assert_eq!(record.report, "Tool report: nothing done yet.");
    // The task prompt, the request, the report turn and its results.
    assert_eq!(record.replaced_messages, 4);
    assert!(task.handover_requested.is_none());
    let first = &texts(&task)[0];
    assert!(first.contains(REPORT_HEADING));
    assert!(first.contains("Tool report: nothing done yet."));

    assert_eq!(
        rejections(&events, "plain_text"),
        [HANDOVER_REPORT_TOOL_DUE]
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolResult { tool_name, output, .. }
            if tool_name == HANDOVER_REPORT_TOOL_NAME && output == REPORT_RECEIVED
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationHandedOver { report, handovers: 1, .. }
            if report == "Tool report: nothing done yet."
    )));
}

/// What the old conversation held when it was replaced: the request for
/// the tool, and the report call answered.
#[tokio::test]
async fn the_replaced_conversation_asks_for_the_tool_and_holds_its_result() {
    struct Seen(std::sync::Mutex<Option<Vec<ai::Message>>>);
    impl HandoverHook for Seen {
        fn on_handover(&self, task: &Task, _report: &str) -> HandoverSeed {
            *self.0.lock().unwrap() = Some(task.conversation.messages.clone());
            HandoverSeed::for_task(task)
        }
    }
    let (mut agent, mut events, commands) = tool_harness();
    let hook = Arc::new(Seen(Default::default()));
    agent.handover_hook = Some(hook.clone());
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![native_report(&[], "The report.", 100), native_complete()],
    );
    let (task, _events) = run(agent, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    let old = hook.0.lock().unwrap().clone().unwrap();
    assert!(message_text(&old[1]).contains(DEFAULT_REPORT_REQUEST));
    let results: Vec<&ai::ToolResult> = old
        .last()
        .unwrap()
        .content
        .iter()
        .filter_map(ai::Content::as_tool_result)
        .collect();
    assert_eq!(results.len(), 1, "{old:#?}");
    assert!(!results[0].is_error);
    assert_eq!(results[0].name, HANDOVER_REPORT_TOOL_NAME);
}

/// An agent offered `handover_report` that replies without a tool call
/// gets a note asking for the tool: its text isn't taken as the report.
#[tokio::test]
async fn a_reply_without_the_report_tool_gets_a_note() {
    let (mut agent, mut events, commands) = tool_harness();
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            reply("I'll write the report now.", 10),
            native_report(&[], "The real report.", 20),
            native_complete(),
        ],
    );
    let (task, _events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "The real report.");
}

/// A report call without a report is rejected and the report asked again;
/// one made when no report is due is rejected and the task goes on.
#[tokio::test]
async fn report_calls_without_a_report_or_a_request_are_rejected() {
    let (mut agent, mut events, commands) = tool_harness();
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            native_report(&[], "   ", 10),
            native_report(&[], "Second try.", 20),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "Second try.");
    assert_eq!(
        rejections(&events, HANDOVER_REPORT_TOOL_NAME),
        [EMPTY_REPORT_CALL]
    );

    let (mut agent, mut events, _commands) = tool_harness();
    push(
        &mut agent,
        vec![native_report(&[], "Unasked.", 10), native_complete()],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.handovers.is_empty());
    assert!(task.handover_requested.is_none());
    assert_eq!(
        rejections(&events, HANDOVER_REPORT_TOOL_NAME),
        [REPORT_NOT_REQUESTED]
    );
}

/// `complete_task` beside the report completes the task: no handover.
#[tokio::test]
async fn complete_task_beside_the_report_completes_the_task() {
    let (mut agent, mut events, commands) = tool_harness();
    commands.try_send(UserCommand::RequestHandover).unwrap();
    let mut message = ai::Message::new_assistant("");
    let mut report = call("report", HANDOVER_REPORT_TOOL_NAME);
    report.arguments = serde_json::json!({"report": "Done anyway."});
    let mut complete = call("complete", "complete_task");
    complete.arguments = serde_json::json!({"summary": "done"});
    message.tool_calls.extend([report, complete]);
    push(&mut agent, vec![scripted(message, 10)]);
    let (task, _events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.handovers.is_empty());
}

/// A task saved with the report received, before its conversation was
/// replaced, hands over with that report when it runs again, before its
/// next request.
#[tokio::test]
async fn a_task_saved_with_its_report_hands_over_when_it_runs_again() {
    let (mut agent, mut events, _commands) = tool_harness();
    agent.task.handover_requested = Some(HandoverRequest {
        report_requested: true,
        report: Some("Saved report.".into()),
        ..HandoverRequest::new(1_500)
    });
    push(&mut agent, vec![native_complete()]);
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "Saved report.");
    assert_eq!(task.handovers[0].input_tokens, 1_500);
    assert!(texts(&task)[0].contains("Saved report."));
    // The one request went to the fresh conversation.
    assert_eq!(task.usage_history.len(), 1);
    assert_eq!(task.conversation_usage_start, 0);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { .. }))
    );
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

/// Output-based agents give the report as plain text, also with `handover`
/// set: the tool is for native tool calling.
#[tokio::test]
async fn output_based_agents_hand_over_from_their_plain_text() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::OutputBased);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000_000));
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
        // Without `handover` set, the report is asked for as plain text.
        assert!(message_text(&task.conversation.messages[1]).contains(DEFAULT_TEXT_REPORT_REQUEST));
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

/// Gives the fresh conversation its configuration.
struct ConfigHook(crate::agent_config::AgentConfig);

impl HandoverHook for ConfigHook {
    fn handover_config(&self, _task: &Task) -> Option<crate::agent_config::AgentConfig> {
        Some(self.0.clone())
    }
}

/// The ToolResult events of `tool`'s calls.
fn results(events: &[AgentEvent], tool: &str) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolResult {
                tool_name, output, ..
            } if tool_name == tool => Some(output.clone()),
            _ => None,
        })
        .collect()
}

/// The hook's configuration runs the fresh conversation: its system prompt
/// is written from it, and a tool it offers that the task didn't offer
/// runs. The task keeps it, without being marked reconfigured.
#[tokio::test]
async fn the_hook_s_configuration_runs_the_fresh_conversation() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    let mut current = agent.task.agent_config.clone();
    current.instructions = "Answer in plain text.".into();
    // The task offers the image tool alone of the two, as saved.
    agent.task.agent_config.tool_config.available_tool_names =
        vec!["view_image".into(), "complete_task".into()];
    agent.tool_registry = Arc::new(agent.all_tools.restricted_to(["view_image"]));
    agent.handover_hook = Some(Arc::new(ConfigHook(current)));
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            reply("The report.", 10),
            native_call("plain_text", 10),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    assert!(
        task.agent_config
            .tool_config
            .available_tool_names
            .contains(&"plain_text".to_string())
    );
    assert!(!task.reconfigured);
    let prompt = task.conversation.system_message.as_deref().unwrap();
    assert!(prompt.contains("Answer in plain text."), "{prompt}");
    assert_eq!(results(&events, "plain_text"), ["Legacy text output"]);
}

/// A configuration with another model than the task's is left out: the
/// fresh conversation keeps the task's.
#[tokio::test]
async fn a_hook_s_configuration_with_another_model_is_left_out() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    let mut other = agent.task.agent_config.clone();
    other.instructions = "Answer in plain text.".into();
    other.llm_config.model = ai::Model::openai_gpt_6_1_sol();
    agent.handover_hook = Some(Arc::new(ConfigHook(other)));
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![reply("The report.", 10), native_complete()],
    );
    let (task, _) = run(agent, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.agent_config.instructions, "test");
    assert_eq!(
        task.agent_config.llm_config.model.id,
        ai::Model::openai_gpt_5_6_luna().id
    );
}

/// A loop for `task`, with the harness's tools.
fn loop_for(
    task: Task,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let (agent, _, _) = harness(ToolCallingMode::Native);
    let (events, receiver) = mpsc::channel(200);
    let (sender, commands) = mpsc::channel(10);
    let agent = AgentLoop::new(
        task,
        Arc::new(ai::LLMClient::new()),
        agent.all_tools.clone(),
        events,
        commands,
        None,
        false,
        None,
        None,
        None,
    );
    (agent, receiver, sender)
}

/// The harness's task, completed with `complete_task`.
async fn completed(mut agent: AgentLoop, events: &mut mpsc::Receiver<AgentEvent>) -> Task {
    push(&mut agent, vec![native_complete()]);
    let (task, _) = run(agent, events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    task
}

/// A report asked for alone: one more turn of the same conversation, whose
/// `handover_report` call gives the report. The task completes with it as
/// its summary, without a handover, and the conversation keeps the request
/// and the report.
#[tokio::test]
async fn a_report_asked_for_alone_completes_the_task_in_the_same_conversation() {
    let (agent, mut events, _commands) = tool_harness();
    let mut task = completed(agent, &mut events).await;
    let before = task.conversation.messages.len();
    task.request_report(None).unwrap();
    let (mut agent, mut events, _commands) = loop_for(task);
    push(&mut agent, vec![native_report(&[], "REPORT: done.", 10)]);
    let (task, events) = run(agent, &mut events).await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "REPORT: done."),
        "{:?}",
        task.current_state
    );
    assert!(task.handovers.is_empty());
    assert!(task.handover_requested.is_none());
    let texts = texts(&task);
    assert_eq!(texts.len(), before + 3, "{texts:#?}");
    assert_eq!(texts[before], crate::handover::DEFAULT_REPORT_ONLY_REQUEST);
    assert_eq!(
        results(&events, HANDOVER_REPORT_TOOL_NAME),
        ["Report received."]
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { .. }))
    );
}

/// Until the report comes, every other call is rejected, `complete_task`
/// too, and the host's request replaces the default.
#[tokio::test]
async fn a_report_asked_for_alone_holds_back_complete_task_too() {
    let (agent, mut events, _commands) = tool_harness();
    let mut task = completed(agent, &mut events).await;
    task.request_report(Some("Report for the next model.".into()))
        .unwrap();
    let (mut agent, mut events, _commands) = loop_for(task);
    push(
        &mut agent,
        vec![
            native_call(COMPLETE_TASK_TOOL_NAME, 10),
            native_report(&[], "REPORT: done.", 10),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "REPORT: done.")
    );
    assert_eq!(
        rejections(&events, COMPLETE_TASK_TOOL_NAME),
        [HANDOVER_REPORT_TOOL_DUE_NO_COMPLETION]
    );
    assert!(texts(&task).contains(&"Report for the next model.".to_string()));
}

/// An agent not offered `handover_report` gives the report as a reply.
#[tokio::test]
async fn a_report_asked_for_alone_comes_as_a_reply_without_the_report_tool() {
    let (agent, mut events, _commands) = harness(ToolCallingMode::Native);
    let mut task = completed(agent, &mut events).await;
    task.request_report(None).unwrap();
    let (mut agent, mut events, _commands) = loop_for(task);
    push(&mut agent, vec![reply("REPORT: done.", 10)]);
    let (task, _) = run(agent, &mut events).await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "REPORT: done.")
    );
    assert!(texts(&task).contains(&crate::handover::DEFAULT_TEXT_REPORT_ONLY_REQUEST.to_string()));
}

/// A report is asked for between turns only, with a request that says
/// something.
#[test]
fn a_report_is_asked_for_between_turns_only() {
    let (agent, _, _) = harness(ToolCallingMode::Native);
    let mut task = agent.task;
    assert!(task.request_report(None).is_err());
    task.set_state(TaskState::Completed {
        summary: "Done.".into(),
    });
    assert!(task.request_report(Some(" ".into())).is_err());
    task.request_report(None).unwrap();
    assert_eq!(task.current_state, TaskState::InProgress);
}

/// The skills configuration of one embedded skill, `name`.
fn skills_of(name: &str) -> crate::skill::SkillsConfig {
    crate::skill::SkillsConfig {
        embedded: vec![crate::skill::EmbeddedSkill::new(
            name,
            format!("---\nname: {name}\ndescription: The {name} skill.\n---\nThe {name} body.\n"),
        )],
        ..Default::default()
    }
}

/// A `load_skill` call for skill `name`.
fn load_skill_call(name: &str) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    let mut load = call("load", crate::skill::LOAD_SKILL_TOOL_NAME);
    load.arguments = serde_json::json!({ "name": name });
    message.tool_calls.push(load);
    scripted(message, 10)
}

/// A loop like the harness's whose task has `skills`, set up as a host sets
/// them up.
fn loop_with_skills(
    skills: Option<crate::skill::SkillsConfig>,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let (agent, _, _) = harness(ToolCallingMode::Native);
    let mut task = agent.task.clone();
    let names = agent.all_tools.list_tool_names();
    let mut registry = agent
        .all_tools
        .restricted_to(names.iter().map(String::as_str));
    task.agent_config.skills_config = skills.clone();
    let prompt = skills.map(|skills| {
        crate::skill::SkillManager::load(&skills, &task.current_working_directory)
            .unwrap()
            .setup(&mut registry, &mut task.agent_config)
    });
    let (events, receiver) = mpsc::channel(200);
    let (sender, commands) = mpsc::channel(10);
    let agent = AgentLoop::new(
        task,
        Arc::new(ai::LLMClient::new()),
        Arc::new(registry),
        events,
        commands,
        None,
        false,
        prompt,
        None,
        None,
    );
    (agent, receiver, sender)
}

/// The hook's configuration brings its skills to the fresh conversation:
/// its system prompt lists them and `load_skill` loads them, from one skill
/// to another, from none to some, and from some to none (`load_skill` is no
/// longer offered).
#[tokio::test]
async fn a_handover_s_configuration_brings_its_skills() {
    for (before, after) in [
        (Some("alpha"), Some("beta")),
        (None, Some("beta")),
        (Some("alpha"), None),
    ] {
        let (mut agent, mut events, commands) = loop_with_skills(before.map(skills_of));
        // As a host gives it, before the runtime sets its skills up.
        let mut current = agent.task.agent_config.clone();
        current.skills_config = after.map(skills_of);
        current
            .tool_config
            .available_tool_names
            .retain(|name| name != crate::skill::LOAD_SKILL_TOOL_NAME);
        agent.handover_hook = Some(Arc::new(ConfigHook(current)));
        commands.try_send(UserCommand::RequestHandover).unwrap();
        push(
            &mut agent,
            vec![
                reply("The report.", 10),
                load_skill_call(after.or(before).unwrap()),
                native_complete(),
            ],
        );
        let (task, events) = run(agent, &mut events).await;
        assert_eq!(task.handovers.len(), 1);
        let prompt = task.conversation.system_message.clone().unwrap();
        let loaded = results(&events, crate::skill::LOAD_SKILL_TOOL_NAME);
        match after {
            Some(name) => {
                assert!(prompt.contains(&format!("The {name} skill.")), "{prompt}");
                assert_eq!(loaded.len(), 1, "{before:?} to {after:?}: {loaded:?}");
                assert!(
                    loaded[0].contains(&format!("The {name} body.")),
                    "{loaded:?}"
                );
            }
            None => {
                assert!(!prompt.contains("Available Skills"), "{prompt}");
                assert!(
                    loaded.iter().all(|output| !output.contains("body.")),
                    "{loaded:?}"
                );
            }
        }
        if let Some(name) = before.filter(|_| before != after) {
            assert!(!prompt.contains(&format!("The {name} skill.")), "{prompt}");
        }
    }
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

/// Context that arrives as a handover starts (a message that needs a reply,
/// say) is not added to the conversation being replaced, where only the
/// report could carry it: it is added to the fresh one.
#[tokio::test]
async fn context_arriving_during_the_report_waits_for_the_fresh_conversation() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    commands.try_send(UserCommand::RequestHandover).unwrap();
    commands
        .try_send(UserCommand::InjectContext {
            content: "Message from agent 'lead': what did you find?".into(),
        })
        .unwrap();
    struct Seen(std::sync::Mutex<Vec<String>>);
    impl HandoverHook for Seen {
        fn on_handover(&self, task: &Task, _report: &str) -> HandoverSeed {
            *self.0.lock().unwrap() = texts(task);
            HandoverSeed::for_task(task)
        }
    }
    let hook = Arc::new(Seen(Default::default()));
    agent.handover_hook = Some(hook.clone());
    push(
        &mut agent,
        vec![reply("Nothing yet.", 10), native_complete()],
    );
    let (task, _events) = run(agent, &mut events).await;

    assert_eq!(task.handovers.len(), 1);
    let old = hook.0.lock().unwrap().clone();
    assert!(
        !old.iter().any(|text| text.contains("what did you find?")),
        "{old:#?}"
    );
    let fresh = texts(&task);
    assert!(fresh[0].contains(REPORT_HEADING));
    assert!(
        fresh
            .iter()
            .skip(1)
            .any(|text| text.contains("# Workflow context update")
                && text.contains("what did you find?")),
        "{fresh:#?}"
    );
    assert!(task.pending_context_injections.is_empty());
}

/// A conversation whose first request was already over the threshold
/// doesn't hand over: the fresh one would start over it too.
#[tokio::test]
async fn no_handover_loop_when_the_first_request_is_over_the_threshold() {
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::Native);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    push(
        &mut agent,
        vec![
            native_call("plain_text", 2_000),
            native_call("plain_text", 2_500),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.handovers.is_empty());
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { .. }))
    );
}

/// The host's threshold replaces the configured one, larger or smaller,
/// and unset, the configured one applies again.
#[tokio::test]
async fn the_hosts_threshold_moves_the_handover_both_ways() {
    // Raised over the configured 1,000: no handover at 2,500.
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::Native);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    let threshold = HandoverThreshold::new(Some(5_000));
    agent.handover_threshold = threshold.clone();
    push(
        &mut agent,
        vec![
            native_call("plain_text", 500),
            native_call("plain_text", 2_500),
            native_complete(),
        ],
    );
    let (task, _) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.handovers.is_empty());

    // Lowered under the configured 1,000,000: a handover at 1,500.
    let (mut agent, mut events, _commands) = tool_harness();
    threshold.set(Some(1_000));
    agent.handover_threshold = threshold.clone();
    push(
        &mut agent,
        vec![
            native_call("plain_text", 500),
            native_call("plain_text", 1_500),
            native_report(&[], "Done: read the fixture.", 1_600),
            native_call("plain_text", 200),
            native_complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].input_tokens, 1_500);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::HandoverRequested {
            input_tokens: 1_500
        }
    )));

    // Unset: the configured 1,000,000 again, so no handover at 1,500.
    let (mut agent, mut events, _commands) = tool_harness();
    threshold.set(None);
    agent.handover_threshold = threshold;
    push(
        &mut agent,
        vec![
            native_call("plain_text", 500),
            native_call("plain_text", 1_500),
            native_complete(),
        ],
    );
    let (task, _) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.handovers.is_empty());
}

/// A threshold with a cap per run: runs that share the host's threshold
/// each hand over at no more than the cap their own configuration gets
/// (as runs on models of different sizes), whichever threshold they go by.
#[tokio::test]
async fn a_capped_threshold_holds_each_run_under_its_own_cap() {
    // The cap for the configurations named "small"; none for the others.
    let threshold = HandoverThreshold::new(Some(5_000))
        .with_most(|config| (config.name == "small").then_some(1_000));
    let handed_over = |responses: Vec<ScriptedStreamResult>, name: &str, threshold| {
        let (mut agent, mut events, commands) = tool_harness();
        agent.task.agent_config.name = name.into();
        agent.handover_threshold = threshold;
        push(&mut agent, responses);
        async move {
            let (task, events) = run(agent, &mut events).await;
            drop(commands);
            assert!(
                matches!(task.current_state, TaskState::Completed { .. }),
                "{:?}: {events:#?}",
                task.current_state
            );
            task.handovers
                .iter()
                .map(|handover| handover.input_tokens)
                .collect::<Vec<_>>()
        }
    };
    let capped = || {
        vec![
            native_call("plain_text", 500),
            native_call("plain_text", 1_500),
            native_report(&[], "Done: read the fixture.", 1_600),
            native_call("plain_text", 200),
            native_complete(),
        ]
    };
    let uncapped = || {
        vec![
            native_call("plain_text", 500),
            native_call("plain_text", 2_500),
            native_complete(),
        ]
    };
    // The host's 5,000, held at the small run's 1,000.
    assert_eq!(
        handed_over(capped(), "small", threshold.clone()).await,
        [1_500]
    );
    // Another run of the same threshold goes by the host's 5,000.
    assert!(
        handed_over(uncapped(), "large", threshold.clone())
            .await
            .is_empty()
    );
    // Unset, the configured 1,000,000 is held at the cap too.
    threshold.set(None);
    assert_eq!(threshold.get(), None);
    assert_eq!(
        handed_over(capped(), "small", threshold.clone()).await,
        [1_500]
    );
    assert!(handed_over(uncapped(), "large", threshold).await.is_empty());
}
