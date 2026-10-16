//! Keeping an idle run's prompt cache warm (`Workflow::refresh_prompt_cache`).

use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::test_support::{ResponsesServer, Workspace, agent_config, text_turn};

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
    use crate::test_support::Reply;
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
