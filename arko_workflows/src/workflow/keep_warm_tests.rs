//! Keeping an idle run's prompt cache warm (`Workflow::refresh_prompt_cache`).

use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::test_support::{
    ProbeTool, Reply, ResponsesServer, Workspace, agent_config, text_turn, tool_call_turn,
};

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    let mut chat = agent_config("Main");
    chat.tool_config.text_reply_ends_task = true;
    Workflow::builder()
        .name("keep-warm-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", chat)
}

/// A run waiting for the user's next message keeps its prompt cache warm:
/// its next request with a ping after it goes out, its cost is charged to
/// the agent, and the run's conversation stays as it was.
#[tokio::test]
async fn an_idle_run_s_prompt_cache_is_kept_warm() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start_async(|index, _| {
        let reply = match index {
            0 => Reply::Stream(text_turn("t0", "Sure.")),
            _ => Reply::Status {
                status: 200,
                retry_after: None,
                body: serde_json::json!({
                    "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
                    "model": "gpt-5.6-luna",
                    "output": [{"type": "message", "id": "m1", "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
                    "usage": {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1100},
                              "output_tokens": 1},
                })
                .to_string(),
            },
        };
        Box::pin(async move { Some(reply) })
    })
    .await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let ledger_before = workflow.cost_ledger().entries.len();
    let refresh = workflow.refresh_prompt_cache(&first).await.unwrap();
    assert_eq!(refresh.usage.cached_tokens, Some(1100));
    let bodies: Vec<Value> = server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    assert_eq!(bodies[1]["prompt_cache_key"], bodies[0]["prompt_cache_key"]);
    assert_eq!(bodies[1]["instructions"], bodies[0]["instructions"]);
    assert_eq!(bodies[1]["max_output_tokens"], 16);
    let input = bodies[1]["input"].as_array().unwrap();
    assert!(
        input.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("single word ok")
    );
    assert_eq!(workflow.cost_ledger().entries.len(), ledger_before + 1);
    let unchanged = workflow.agent_result(first.agent_run_id).unwrap();
    assert_eq!(
        unchanged.task.conversation.messages.len(),
        first.task.conversation.messages.len()
    );
}

/// A non-streamed reply, as a refresh asks for: one word, its input mostly
/// read from the cache.
fn refresh_reply() -> Reply {
    Reply::Status {
        status: 200,
        retry_after: None,
        body: serde_json::json!({
            "id": "resp_r", "object": "response", "created_at": 0, "status": "completed",
            "model": "gpt-5.6-luna",
            "output": [{"type": "message", "id": "m1", "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1100},
                      "output_tokens": 1},
        })
        .to_string(),
    }
}

/// A server whose first reply is `first` (streamed), and whose refreshes
/// get [`refresh_reply`].
async fn server_with(first: String) -> ResponsesServer {
    ResponsesServer::start_async(move |index, _| {
        let reply = match index {
            0 => Reply::Stream(first.clone()),
            _ => refresh_reply(),
        };
        Box::pin(async move { Some(reply) })
    })
    .await
}

fn bodies(server: &ResponsesServer) -> Vec<Value> {
    server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect()
}

/// A run paused for a tool approval waits with a call that has no output:
/// a message can't follow it, so its last request goes again unchanged,
/// which ends where the cache entry its next request reads ends.
#[tokio::test]
async fn a_run_paused_for_an_approval_sends_its_last_request_again() {
    let workspace = Workspace::new();
    let server = server_with(tool_call_turn("p1", "probe", serde_json::json!({}))).await;
    let mut config = agent_config("Main");
    config.tool_config.available_tool_names = vec!["probe".into()];
    config.tool_config.default_auto_approve_tool_use = false;
    let workflow = Workflow::builder()
        .name("keep-warm-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", config)
        .agent_tool("main", ProbeTool::new("probe", false))
        .approval_policy(crate::policy::ApprovalPolicy::RespectAgentConfig)
        .build()
        .unwrap();
    let error = workflow.run_agent("main", "Probe it.").await.unwrap_err();
    let WorkflowError::AgentPaused { task_id, .. } = error else {
        panic!("{error:?}");
    };
    let paused = workflow.agent_result(task_id).unwrap();
    let refresh = workflow.refresh_prompt_cache(&paused).await.unwrap();
    assert!(refresh.resent_last_request);
    assert_eq!(refresh.output_cap, Some(16));
    let bodies = bodies(&server);
    assert_eq!(bodies.len(), 2);
    for key in ["input", "tools", "instructions", "prompt_cache_key"] {
        assert_eq!(bodies[1][key], bodies[0][key], "{key}");
    }
    assert_eq!(bodies[1]["max_output_tokens"], 16);
}

/// A refresh is charged as a model call is, and saved at once: the live
/// checkpoint holds its cost. With a budget exceeded, nothing is sent.
#[tokio::test]
async fn a_refresh_is_charged_saved_and_held_to_the_budget() {
    let workspace = Workspace::new();
    let server = server_with(text_turn("t0", "Sure.")).await;
    let run = workspace.0.join(".run");
    let workflow = builder(&server, &workspace)
        .persistence(crate::persistence::PersistenceConfig::new(&run))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.refresh_prompt_cache(&first).await.unwrap();
    let entries = workflow.cost_ledger().entries.len();
    let latest = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    assert_eq!(latest.trim(), crate::persistence::LIVE_CHECKPOINT_FILE);
    let saved = crate::persistence::PersistenceManager::load_checkpoint(&run.join(latest.trim()))
        .await
        .unwrap();
    assert_eq!(saved.cost_ledger.entries.len(), entries);
    assert_eq!(
        serde_json::to_value(&saved.cost_ledger).unwrap(),
        serde_json::to_value(workflow.cost_ledger()).unwrap()
    );

    workflow.set_agent_budget("main", Some(0.0)).unwrap();
    let error = workflow.refresh_prompt_cache(&first).await.unwrap_err();
    assert!(
        matches!(&error, WorkflowError::AgentBudgetExceeded { agent, .. } if agent == "main"),
        "{error:?}"
    );
    assert_eq!(server.bodies().len(), 2);
    assert_eq!(workflow.cost_ledger().entries.len(), entries);
}
