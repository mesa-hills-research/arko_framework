//! Agents offered no tools: no tools, a reply in text ends the task. Their
//! requests carry no tool configuration and their system prompt says
//! nothing about tools.

use super::image_tests::response;
use super::tool_call_tests::{agent_for, drain};
use super::*;
use futures::StreamExt;

/// A native agent offered `tools` (from the built-ins and `plain_text`),
/// whose reply in text ends the task.
fn replying_agent(
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
fn chat_agent() -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    replying_agent(&[])
}

fn push(agent: &mut AgentLoop, replies: &[&str]) {
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

async fn run(agent: AgentLoop, events: &mut mpsc::Receiver<AgentEvent>) -> (Task, Vec<AgentEvent>) {
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
