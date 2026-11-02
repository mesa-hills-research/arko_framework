//! A run whose reply is optional (`RunOptions::reply_optional`,
//! `ContinueOptions::reply_optional`) ends without one when its agent has
//! nothing to say.

use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::test_support::{ResponsesServer, Workspace, agent_config, text_turn};
use crate::workflow::stop::{RunStopIntent, StopRequest};

/// An agent whose reply in text ends its task.
fn replying() -> AgentConfig {
    let mut config = agent_config("Main");
    config.tool_config.text_reply_ends_task = true;
    config
}

fn workflow(server: &ResponsesServer, workspace: &Workspace) -> Workflow {
    Workflow::builder()
        .name("reply-optional-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", replying())
        .build()
        .unwrap()
}

/// A run started with the option ends without a reply when its response
/// has neither text nor a call. A continuation without the option is asked
/// for a reply again (a note, then the reply), and one with it ends without
/// one again.
#[tokio::test]
async fn each_run_says_whether_it_may_end_without_a_reply() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        let text = if index == 2 { "Here it is." } else { "" };
        Some(text_turn(&format!("t{index}"), text))
    })
    .await;
    let workflow = workflow(&server, &workspace);

    let news = workflow
        .run_agent_with("main", "News.", RunOptions::new().reply_optional(true))
        .await
        .unwrap();
    assert_eq!(news.status, AgentRunStatus::Completed);
    assert_eq!(news.summary, "");
    assert!(news.task().unwrap().reply_optional);
    assert_eq!(server.requests(), 1);

    let asked = workflow.continue_agent(&news, "A question.").await.unwrap();
    assert_eq!(asked.summary, "Here it is.");
    assert!(!asked.task().unwrap().reply_optional);
    assert_eq!(server.requests(), 3);

    let more = workflow
        .continue_agent_with(
            &asked,
            "More news.",
            ContinueOptions::new().reply_optional(true),
        )
        .await
        .unwrap();
    assert_eq!(more.status, AgentRunStatus::Completed);
    assert_eq!(more.summary, "");
    assert_eq!(server.requests(), 4);
}

/// A reopened run takes the setting from its options, as a continued one
/// does.
#[tokio::test]
async fn a_reopened_run_may_end_without_a_reply() {
    let workspace = Workspace::new();
    // The first request waits until the run is stopped.
    let server =
        ResponsesServer::start(|index, _| (index > 0).then(|| text_turn(&format!("t{index}"), "")))
            .await;
    let workflow = Arc::new(workflow(&server, &workspace));
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("main", "A question.").await }
    });
    server.wait_for_requests(1).await;
    let stopped = workflow
        .stop_agent(
            "main",
            StopRequest::operator("Esc"),
            RunStopIntent::Relaunch,
        )
        .unwrap();
    let _ = tokio::time::timeout(Duration::from_secs(20), run)
        .await
        .unwrap()
        .unwrap();
    let cancelled = workflow.agent_result(stopped[0]).unwrap();
    assert_eq!(cancelled.status, AgentRunStatus::Cancelled);

    let reopened = workflow
        .reopen_agent_with(
            &cancelled,
            "News.",
            ContinueOptions::new().reply_optional(true),
        )
        .await
        .unwrap();
    assert_eq!(reopened.status, AgentRunStatus::Completed);
    assert_eq!(reopened.summary, "");
    assert!(reopened.task().unwrap().reply_optional);
}
