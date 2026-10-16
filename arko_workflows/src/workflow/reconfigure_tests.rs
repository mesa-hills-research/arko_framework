//! A run continued with a new configuration (`ContinueOptions::reconfigure`):
//! the switch between a chat and an agent, in the same conversation.

use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::persistence::PersistenceConfig;
use crate::test_support::{ProbeTool, ResponsesServer, Workspace, agent_config, text_turn};

/// The agent as a chat: offered no tools, its reply in text ends its task.
fn chat() -> AgentConfig {
    let mut config = agent_config("Main");
    config.tool_config.text_reply_ends_task = true;
    config.instructions = "Chat with the user.".into();
    config
}

/// The agent as an agent: offered the probe.
fn agent() -> AgentConfig {
    let mut config = chat();
    config.tool_config.available_tool_names = vec!["probe".into()];
    config.instructions = "Work on the user's task.".into();
    config
}

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", chat())
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| tool["name"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// A chat continued as an agent: the same run and conversation, with its
/// prompt cache key, now offered its tools (the workflow's added, as for a
/// new run) under a system prompt from the new configuration. As a chat it
/// had none of them. A resume keeps the new configuration, though the
/// agent is registered as a chat.
#[tokio::test]
async fn a_chat_continued_as_an_agent_keeps_its_conversation_and_gets_tools() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let second = workflow
        .continue_agent_with(
            &first,
            "Now work on it.",
            ContinueOptions::new().reconfigure(agent()),
        )
        .await
        .unwrap();
    assert_eq!(second.agent_run_id, first.agent_run_id);
    assert!(second.task.reconfigured);

    let bodies: Vec<Value> = server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    assert!(tool_names(&bodies[0]).is_empty(), "{}", bodies[0]);
    let chat_prompt = bodies[0]["instructions"].as_str().unwrap();
    assert!(chat_prompt.contains("Chat with the user."));
    assert!(!chat_prompt.contains("## Tool Usage"));
    assert_eq!(tool_names(&bodies[1]), ["probe"]);
    let agent_prompt = bodies[1]["instructions"].as_str().unwrap();
    assert!(agent_prompt.contains("Work on the user's task."));
    assert!(agent_prompt.contains("## Tool Usage"));
    assert_eq!(bodies[0]["prompt_cache_key"], bodies[1]["prompt_cache_key"]);
    let sent = bodies[1]["input"].to_string();
    assert!(sent.contains("Hello.") && sent.contains("Sure.") && sent.contains("Now work on it."));

    // Resumed with the agent registered as a chat, the run keeps the
    // configuration it was given, and continues with it.
    workflow.checkpoint().await.unwrap();
    let run = workspace.0.join(".run");
    let latest = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    let resumed = builder(&server, &workspace)
        .resume_from(run.join(latest.trim()))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    assert!(restored.task.reconfigured);
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    let last: Value = serde_json::from_str(server.bodies().last().unwrap()).unwrap();
    assert_eq!(tool_names(&last), ["probe"]);
}

/// The model stays: another one is refused before anything runs.
#[tokio::test]
async fn a_run_is_reconfigured_with_its_model() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let mut other = agent();
    other.llm_config.model = arko_ai::Model::anthropic_claude_opus_5_5();
    let error = workflow
        .continue_agent_with(&first, "Go on.", ContinueOptions::new().reconfigure(other))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("keeps its model")),
        "{error}"
    );
    assert_eq!(server.requests(), 1);
}
