//! Agents offered no tools: no tools, a reply in text ends the task. Their
//! requests carry no tool configuration and their system prompt says
//! nothing about tools.

use super::image_tests::response;
use super::tool_call_tests::{agent_for, drain};
use super::*;
use crate::handover::{DEFAULT_CHAT_REPORT_REQUEST, HandoverConfig};
use futures::StreamExt;

/// A native agent offered `tools` (from the built-ins and `plain_text`),
/// whose reply in text ends the task.
pub(super) fn replying_agent(
    tools: &[&str],
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let mut config = super::tests::test_config();
    config.tool_config.available_tool_names = tools.iter().map(|tool| tool.to_string()).collect();
    config.tool_config.text_reply_ends_task = true;
    let task = Task::new(config, "Hello.".into(), std::env::temp_dir(), None);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(super::image_tests::TextTool));
    agent_for(task, registry, Arc::new(ai::LLMClient::new()))
}

/// A native agent offered nothing: no tools, and its reply in text ends
/// the task.
pub(super) fn chat_agent() -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    replying_agent(&[])
}

pub(super) fn push(agent: &mut AgentLoop, replies: &[&str]) {
    for reply in replies {
        let Ok(end) = response(ai::Message::new_assistant(reply)) else {
            unreachable!()
        };
        let chunk = Ok(ai::StreamEvent::TextChunk((*reply).into()));
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(futures::stream::iter([chunk]).chain(end).boxed()));
    }
}

pub(super) async fn run(
    agent: AgentLoop,
    events: &mut mpsc::Receiver<AgentEvent>,
) -> (Task, Vec<AgentEvent>) {
    let outcome = timeout(Duration::from_secs(5), agent.run()).await;
    let received = drain(events);
    let task = outcome
        .unwrap_or_else(|error| panic!("{error}: {received:#?}"))
        .unwrap();
    (task, received)
}

/// Its requests carry no tool configuration (so no provider gets a tool
/// list or tool choice), its system prompt has no section about tools, and
/// its reply ends the task.
#[tokio::test]
async fn an_agent_offered_no_tools_makes_requests_without_tools() {
    let (mut agent, mut events, _commands) = chat_agent();
    assert!(!agent.offers_tools());
    assert!(agent.build_generation_config().tool_config.is_none());
    push(&mut agent, &["Hi there."]);
    let (task, _) = run(agent, &mut events).await;

    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "Hi there.".into()
        }
    );
    let system = task.conversation.system_message.unwrap();
    assert!(!system.contains("## Tool"), "{system}");
    assert!(!system.contains("tool"), "{system}");
}

/// A chat whose conversation holds two questions, the second with an
/// image, unanswered; a handover is set.
fn chat_with_history() -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let (mut agent, events, commands) = chat_agent();
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000_000));
    let conversation = &mut agent.task.conversation;
    conversation.add_user_message("First question.");
    conversation.add_assistant_message("First answer.");
    conversation.add_user_message_with_image(
        "Second question, about this picture.",
        ai::Image::new_base64("cGl4ZWxz", "image/png"),
    );
    (agent, events, commands)
}

/// A chat with a handover gives its report as a reply in text: it isn't
/// offered `handover_report`, its requests stay without tools, and the
/// report request asks for a summary that leaves the latest message to the
/// fresh conversation.
#[tokio::test]
async fn a_chat_is_asked_for_a_summary_in_text() {
    let (mut agent, _events, _commands) = chat_with_history();
    assert!(!agent.offers_report_tool());
    assert!(agent.build_generation_config().tool_config.is_none());
    agent.note_handover_request();
    agent.start_handover_if_due().await.unwrap();
    let request = agent.task.conversation.messages.last().unwrap();
    assert_eq!(message_text(request), DEFAULT_CHAT_REPORT_REQUEST);
    assert!(
        agent
            .task
            .runtime_messages
            .contains(&(agent.task.conversation.messages.len() - 1))
    );
    assert!(agent.build_generation_config().tool_config.is_none());

    // A report request of the host's own replaces it.
    let (mut agent, _events, _commands) = chat_with_history();
    agent.task.agent_config.handover = Some(HandoverConfig {
        report_request: Some("Summarize.".into()),
        ..HandoverConfig::at_input_tokens(1_000_000)
    });
    agent.note_handover_request();
    agent.start_handover_if_due().await.unwrap();
    let request = agent.task.conversation.messages.last().unwrap();
    assert_eq!(message_text(request), "Summarize.");
}

/// The summary replaces the conversation: the fresh one starts from the
/// user's unanswered message (with its image) and the summary, and the
/// reply there ends the task.
#[tokio::test]
async fn a_chat_goes_on_from_the_latest_message_and_the_summary() {
    let (mut agent, mut events, commands) = chat_with_history();
    commands.try_send(UserCommand::RequestHandover).unwrap();
    push(
        &mut agent,
        &[
            "The user asked a first question, answered.",
            "Second answer.",
        ],
    );
    let (task, events) = run(agent, &mut events).await;

    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "Second answer.".into()
        }
    );
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(
        task.handovers[0].report,
        "The user asked a first question, answered."
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationHandedOver { handovers: 1, .. }
    )));
    let [first, reply] = task.conversation.messages.as_slice() else {
        panic!("{:#?}", task.conversation.messages);
    };
    let text = message_text(first);
    assert!(
        text.starts_with("Second question, about this picture.\n\n# Handover report"),
        "{text}"
    );
    assert!(text.contains("The user asked a first question, answered."));
    assert!(!text.contains("First question."), "{text}");
    assert_eq!(
        first
            .content
            .iter()
            .filter(|content| content.as_image().is_some())
            .count(),
        1
    );
    assert_eq!(message_text(reply), "Second answer.");
}

/// An agent with tools keeps its tool configuration and its section.
#[tokio::test]
async fn an_agent_offered_a_tool_still_says_how_to_use_it() {
    let (mut agent, mut events, _commands) = replying_agent(&["plain_text"]);
    assert!(agent.offers_tools());
    let config = agent.build_generation_config();
    assert_eq!(config.tool_config.unwrap().tools.len(), 1);
    push(&mut agent, &["Done."]);
    let (task, _) = run(agent, &mut events).await;
    let system = task.conversation.system_message.unwrap();
    assert!(system.contains("## Tool Usage"), "{system}");
}

/// Reasoning the provider streams (summaries, thinking) reaches the host as
/// `ReasoningChunk` events, in order and before the reply's text.
#[tokio::test]
async fn streamed_reasoning_reaches_the_host() {
    let (mut agent, mut events, _commands) = chat_agent();
    let Ok(end) = response(ai::Message::new_assistant("Four.")) else {
        unreachable!()
    };
    let chunks = [
        Ok(ai::StreamEvent::ReasoningChunk("Adding two ".into())),
        Ok(ai::StreamEvent::ReasoningChunk("and two.".into())),
        Ok(ai::StreamEvent::TextChunk("Four.".into())),
    ];
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter(chunks).chain(end).boxed()));
    let (_, received) = run(agent, &mut events).await;
    let streamed: Vec<_> = received
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ReasoningChunk(chunk) => Some(("reasoning", chunk.as_str())),
            AgentEvent::AssistantResponseChunk(chunk) => Some(("text", chunk.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        streamed,
        [
            ("reasoning", "Adding two "),
            ("reasoning", "and two."),
            ("text", "Four.")
        ]
    );
}
