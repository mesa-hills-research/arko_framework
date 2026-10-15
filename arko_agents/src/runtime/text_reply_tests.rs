//! `text_reply_ends_task`: a reply in plain text, without a tool call, is
//! the agent's final reply and completes the task, as `complete_task` would.

use super::image_tests::{call, harness, response};
use super::tool_call_tests::{agent_for, drain, native_call, turn};
use super::*;
use crate::handover::{DEFAULT_REPORT_REQUEST_NO_COMPLETION, HandoverConfig};
use futures::StreamExt;

/// A native agent offered `plain_text`, `view_image` and (listed, but not
/// offered once the setting is on) `complete_task`, whose reply in text
/// ends the task.
fn replying_harness() -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let (agent, _, _) = harness(ToolCallingMode::Native);
    let mut task = agent.task;
    task.agent_config.tool_config.text_reply_ends_task = true;
    agent_for(task, registry(), Arc::new(ai::LLMClient::new()))
}

fn push(agent: &mut AgentLoop, responses: Vec<ScriptedStreamResult>) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend(responses);
}

/// A response with `text` and `calls`, its text streamed first.
fn streamed(text: &str, calls: Vec<ai::ToolCall>) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant(text);
    message.tool_calls = calls;
    let Ok(end) = response(message) else {
        unreachable!()
    };
    let chunk = (!text.is_empty()).then(|| Ok(ai::StreamEvent::TextChunk(text.into())));
    Ok(futures::stream::iter(chunk).chain(end).boxed())
}

async fn run(agent: AgentLoop, events: &mut mpsc::Receiver<AgentEvent>) -> (Task, Vec<AgentEvent>) {
    let outcome = timeout(Duration::from_secs(5), agent.run()).await;
    let received = drain(events);
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

fn completions(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TaskStateChanged(TaskState::Completed { summary }) => Some(summary.clone()),
            _ => None,
        })
        .collect()
}

fn retries(events: &[AgentEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, AgentEvent::RetryScheduled { .. }))
        .count()
}

/// A reply in text completes the task with the reply (trimmed) as its
/// summary, reported as a `complete_task` completion is: one
/// `TaskStateChanged(Completed)`, after the reply streamed. The agent isn't
/// offered `complete_task`, although its configuration lists it, and the
/// system prompt says how the task ends.
#[tokio::test]
async fn a_reply_in_text_completes_the_task_with_it() {
    let (mut agent, mut events, _commands) = replying_harness();
    let offered: Vec<String> = agent
        .build_native_tool_config(&agent.task.conversation.model, None)
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    assert!(
        !offered.contains(&"complete_task".to_string()),
        "{offered:?}"
    );
    assert!(offered.contains(&"plain_text".to_string()), "{offered:?}");
    push(&mut agent, vec![streamed("\nThe answer is 391.\n", vec![])]);
    let (task, events) = run(agent, &mut events).await;

    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "The answer is 391.".into()
        }
    );
    assert_eq!(completions(&events), ["The answer is 391."]);
    let streamed_at = events
        .iter()
        .position(|event| matches!(event, AgentEvent::AssistantResponseChunk(_)))
        .unwrap();
    let completed_at = events
        .iter()
        .position(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::Completed { .. })
            )
        })
        .unwrap();
    assert!(streamed_at < completed_at, "{events:#?}");
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
    );
    assert!(matches!(events.last(), Some(AgentEvent::TaskStreamEnd)));
    // The reply is the conversation's last message.
    let last = task.conversation.messages.last().unwrap();
    assert_eq!(last.role, ai::Role::Assistant);
    assert_eq!(message_text(last), "\nThe answer is 391.\n");
    let system = task.conversation.system_message.clone().unwrap();
    assert!(
        system.contains("A response without a tool call is your final reply: it ends the task"),
        "{system}"
    );
    assert!(!system.contains("complete_task"), "{system}");
}

/// Text beside tool calls doesn't end the task: the calls run and the
/// task goes on until a reply without a call.
#[tokio::test]
async fn text_beside_calls_does_not_end_the_task() {
    let (mut agent, mut events, _commands) = replying_harness();
    push(
        &mut agent,
        vec![
            streamed("Looking first.", vec![call("look", "plain_text")]),
            streamed("Done looking.", vec![]),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(completions(&events), ["Done looking."]);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolResult { tool_name, .. } if tool_name == "plain_text"
    )));
    assert_eq!(task.usage_history.len(), 2);
}

/// A response with neither text nor a tool call is still one the runtime
/// can't use: a note (without `complete_task`) asks again, under the error
/// policy, and the next reply completes the task.
#[tokio::test]
async fn an_empty_response_still_gets_a_note() {
    let (mut agent, mut events, _commands) = replying_harness();
    push(
        &mut agent,
        vec![streamed("", vec![]), streamed("Now a reply.", vec![])],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(completions(&events), ["Now a reply."]);
    assert_eq!(retries(&events), 1);
    let texts = texts(&task);
    assert!(
        texts
            .iter()
            .any(|text| text == NO_NATIVE_TOOL_CALL_OR_REPLY),
        "{texts:#?}"
    );
    assert!(!NO_NATIVE_TOOL_CALL_OR_REPLY.contains("complete_task"));
}

/// A call to `complete_task` from an agent that isn't offered it is a call
/// to an unavailable tool: it completes nothing.
#[tokio::test]
async fn complete_task_is_not_available() {
    let (mut agent, mut events, _commands) = replying_harness();
    push(
        &mut agent,
        vec![
            turn(vec![native_call(
                "done",
                "complete_task",
                serde_json::json!({"summary": "Through the tool."}),
            )]),
            streamed("Through text.", vec![]),
        ],
    );
    let (_task, events) = run(agent, &mut events).await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::UnavailableToolAttempted { tool_name, .. } if tool_name == "complete_task"
    )));
    assert_eq!(completions(&events), ["Through text."]);
}

/// A handover report that is due comes first: a reply then is the report
/// (for an agent that gives it as a reply), and the task ends with a reply
/// in the fresh conversation. Calls rejected meanwhile don't mention
/// `complete_task`.
#[tokio::test]
async fn a_due_handover_report_comes_before_the_final_reply() {
    let (mut agent, mut events, commands) = replying_harness();
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        vec![
            streamed("", vec![call("look", "plain_text")]),
            streamed("Report: nothing done yet.", vec![]),
            streamed("The final reply.", vec![]),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "Report: nothing done yet.");
    assert_eq!(completions(&events), ["The final reply."]);
    let rejected = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolRejected {
                tool_name,
                reason: Some(reason),
                ..
            } if tool_name == "plain_text" => Some(reason.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(rejected, HANDOVER_REPORT_DUE_NO_COMPLETION);
}

/// For an agent given `handover_report`, a reply while the report is due is
/// answered with a note asking for that call (it doesn't end the task), and
/// the report request doesn't offer `complete_task`.
#[tokio::test]
async fn with_the_report_tool_a_reply_waits_for_the_report() {
    let (mut agent, mut events, commands) = replying_harness();
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000_000));
    commands.try_send(UserCommand::RequestHandover).unwrap();
    let mut report = call("report", crate::handover::HANDOVER_REPORT_TOOL_NAME);
    report.arguments = serde_json::json!({"report": "The report."});
    push(
        &mut agent,
        vec![
            streamed("I'm done, here's my answer.", vec![]),
            streamed("", vec![report]),
            streamed("The final reply.", vec![]),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].report, "The report.");
    assert_eq!(completions(&events), ["The final reply."]);
    assert_eq!(retries(&events), 1);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { error_message, .. }
            if error_message.contains("Call handover_report with the report first")
    )));
    assert!(!DEFAULT_REPORT_REQUEST_NO_COMPLETION.contains("complete_task"));
}

/// Context taken while the final reply streams is not added to the
/// completed task's conversation, as with a final `complete_task`: the task
/// keeps it, and a continuation sends it once, after its message.
#[tokio::test(start_paused = true)]
async fn context_taken_during_the_final_reply_waits_for_a_continuation() {
    let (mut agent, _events, commands) = replying_harness();
    let Ok(end) = response(ai::Message::new_assistant("All done.")) else {
        unreachable!()
    };
    let final_reply = futures::stream::once(async {
        tokio_sleep(Duration::from_secs(1)).await;
    })
    .filter_map(|()| async { None })
    .chain(end)
    .boxed();
    push(&mut agent, vec![Ok(final_reply)]);
    let run = tokio::spawn(agent.run());
    tokio_sleep(Duration::from_millis(500)).await;
    commands
        .try_send(UserCommand::InjectContext {
            content: "Too late.".into(),
        })
        .unwrap();
    let mut task = run.await.unwrap().unwrap();
    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "All done.".into()
        }
    );
    assert_eq!(task.pending_context_injections, ["Too late."]);
    assert!(
        !texts(&task)
            .iter()
            .any(|text| text.contains("Workflow context update"))
    );

    task.prepare_for_resumption_from_completed("Carry on.".into())
        .unwrap();
    let (mut agent, _events, _commands) =
        agent_for(task, registry(), Arc::new(ai::LLMClient::new()));
    push(&mut agent, vec![streamed("Carried on.", vec![])]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "Carried on.".into()
        }
    );
    assert!(task.pending_context_injections.is_empty());
    let texts = texts(&task);
    let carry_on = texts.iter().position(|text| text == "Carry on.").unwrap();
    assert!(
        texts[carry_on + 1].starts_with("# Workflow context update\n\nToo late."),
        "{texts:#?}"
    );
}

/// The tools of [`harness`].
fn registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(super::image_tests::ImageTool));
    registry.register(Arc::new(super::image_tests::TextTool));
    registry
}

/// Output-based mode ignores the setting: `complete_task` is offered and a
/// reply in text without a tool call is not an ending.
#[tokio::test]
async fn output_based_agents_ignore_the_setting() {
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::OutputBased);
    agent.task.agent_config.tool_config.text_reply_ends_task = true;
    assert!(
        agent
            .task
            .agent_config
            .tool_config
            .offered_tool_names()
            .contains("complete_task")
    );
    push(
        &mut agent,
        vec![
            streamed("Just text.", vec![]),
            streamed(
                "<complete_task>\n<summary>done</summary>\n</complete_task>",
                vec![],
            ),
        ],
    );
    let (_task, events) = run(agent, &mut events).await;
    assert_eq!(completions(&events), ["done"]);
    assert_eq!(retries(&events), 1);
}

/// The setting is refused together with `text_reply_as_message`.
#[tokio::test]
async fn the_setting_is_refused_with_text_reply_as_message() {
    let (mut agent, _events, _commands) = replying_harness();
    agent.task.agent_config.tool_config.text_reply_as_message = true;
    let error = agent.run().await.unwrap_err().to_string();
    assert!(
        error.contains("text_reply_ends_task and text_reply_as_message"),
        "{error}"
    );
}
