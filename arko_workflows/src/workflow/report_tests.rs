//! A report asked of a run alone (`Workflow::request_report`), without a
//! handover.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::persistence::PersistenceConfig;
use crate::policy::AgentPause;
use crate::test_support::{
    ProbeTool, Reply, ResponsesServer, Workspace, agent_config, complete_turn, text_turn,
    tool_call_turn,
};
use crate::workflow::stop::{RunStopIntent, StopRequest};

/// How the default request for a report alone begins.
const REQUESTED: &str = "Write a report of your work so far now.";

/// An agent offered the probe, whose reply in text ends its task, and which
/// gives a report with `handover_report`.
fn agent() -> AgentConfig {
    let mut config = agent_config("Main");
    config.tool_config.available_tool_names = vec!["probe".into()];
    config.tool_config.text_reply_ends_task = true;
    config.handover = Some(arko_agents::HandoverConfig::at_input_tokens(1_000_000));
    config
}

fn builder(
    server: &ResponsesServer,
    workspace: &Workspace,
    config: AgentConfig,
) -> WorkflowBuilder {
    Workflow::builder()
        .name("report-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", config)
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
}

fn bodies(server: &ResponsesServer) -> Vec<Value> {
    server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect()
}

/// The report call that answers a request for a report.
fn report_turn() -> String {
    tool_call_turn(
        "report",
        "handover_report",
        json!({"report": "REPORT: said hello."}),
    )
}

/// The run writes the report in one more turn of its conversation, with
/// the same prompt cache key, and completes with it as its summary. No
/// fresh conversation starts, and the conversation goes on after it.
#[tokio::test]
async fn a_report_is_written_on_request_in_the_same_conversation() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, body| {
        Some(if body.contains(REQUESTED) && !body.contains("Go on.") {
            report_turn()
        } else {
            text_turn(&format!("t{index}"), "Sure.")
        })
    })
    .await;
    let workflow = builder(&server, &workspace, agent())
        .agent_tool("main", ProbeTool::new("probe", false))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let result = workflow.request_report(&first, None).await.unwrap();
    assert_eq!(result.agent_run_id, first.agent_run_id);
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.summary, "REPORT: said hello.");
    let task = result.task().unwrap();
    assert!(task.handovers.is_empty());
    assert!(task.handover_requested.is_none());
    let sent = bodies(&server);
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0]["prompt_cache_key"], sent[1]["prompt_cache_key"]);
    let input = sent[1]["input"].to_string();
    assert!(input.contains("Hello.") && input.contains("Sure.") && input.contains(REQUESTED));
    assert_eq!(
        workflow.agent_result(first.agent_run_id).unwrap().summary,
        result.summary
    );

    workflow.continue_agent(&result, "Go on.").await.unwrap();
    let last = bodies(&server).pop().unwrap()["input"].to_string();
    assert!(last.contains("REPORT: said hello.") && last.contains("Go on."));
}

/// The host's request replaces the default, and the run's other calls,
/// `complete_task` included, are rejected until the report comes.
#[tokio::test]
async fn a_report_s_request_is_the_host_s_and_other_calls_wait() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if !body.contains("Summarize it for the next model.") {
            complete_turn("c0", "Started.")
        } else if !body.contains("call_p1") {
            tool_call_turn("p1", "probe", json!({}))
        } else if !body.contains("call_c1") {
            complete_turn("c1", "Done.")
        } else {
            report_turn()
        })
    })
    .await;
    let mut config = agent();
    config.tool_config.text_reply_ends_task = false;
    let probe = ProbeTool::new("probe", false);
    let workflow = builder(&server, &workspace, config)
        .agent_tool("main", probe.clone())
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    assert_eq!(first.summary, "Started.");
    let result = workflow
        .request_report(&first, Some("Summarize it for the next model."))
        .await
        .unwrap();
    assert_eq!(result.summary, "REPORT: said hello.");
    assert_eq!(probe.calls(), 0);
    let input = bodies(&server).pop().unwrap()["input"].to_string();
    assert!(!input.contains(REQUESTED), "{input}");
    let rejected =
        "This call was not run: your handover report is due. Call handover_report with the report.";
    assert_eq!(input.matches(rejected).count(), 2, "{input}");
}

/// An error at the report's request: the error policy's automatic recovery
/// doesn't apply. The caller gets the pause with its cause and kind, and the
/// run is as it was before the request, its model calls still counted.
async fn a_failed_report(
    reply: Reply,
    config: AgentConfig,
) -> (
    WorkflowError,
    AgentResult,
    AgentResult,
    Workflow,
    ResponsesServer,
    Workspace,
) {
    let workspace = Workspace::new();
    let reply = Arc::new(parking_lot::Mutex::new(Some(reply)));
    let server = ResponsesServer::start_async(move |index, body| {
        let reply = if body.contains(REQUESTED) {
            reply.lock().take()
        } else {
            None
        };
        let reply =
            reply.unwrap_or_else(|| Reply::Stream(text_turn(&format!("t{index}"), "Sure.")));
        Box::pin(async move { Some(reply) })
    })
    .await;
    let workflow = builder(&server, &workspace, config)
        .agent_tool("main", ProbeTool::new("probe", false))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(20),
        workflow.request_report(&first, None),
    )
    .await
    .expect("the report doesn't wait")
    .unwrap_err();
    let restored = workflow.agent_result(first.agent_run_id).unwrap();
    (error, first, restored, workflow, server, workspace)
}

/// A credential the provider refuses.
#[tokio::test]
async fn a_report_refused_for_its_credential_says_so_and_leaves_the_run_as_it_was() {
    let (error, first, restored, workflow, server, _workspace) = a_failed_report(
        Reply::Status {
            status: 401,
            retry_after: None,
            body: json!({"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}).to_string(),
        },
        agent(),
    )
    .await;
    assert!(
        matches!(
            &error,
            WorkflowError::AgentPaused {
                pause: AgentPause::ErrorRecovery {
                    error_kind: Some(arko_ai::ErrorKind::Authentication),
                    ..
                },
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(restored.status, AgentRunStatus::Completed);
    assert_eq!(restored.summary, first.summary);
    let (before, after) = (first.task().unwrap(), restored.task().unwrap());
    assert_eq!(
        after.conversation.messages.len(),
        before.conversation.messages.len()
    );
    assert!(after.handover_requested.is_none());
    assert!(after.usage_history.len() >= before.usage_history.len());
    // The run goes on from where it was.
    workflow.continue_agent(&restored, "Go on.").await.unwrap();
    let last = bodies(&server).pop().unwrap()["input"].to_string();
    assert!(!last.contains(REQUESTED), "{last}");
}

/// A plan's usage limit isn't waited out, also by an agent set to wait
/// until it resets.
#[tokio::test]
async fn a_report_at_a_plan_limit_doesn_t_wait_for_the_reset() {
    let mut config = agent();
    config.llm_retry_config.usage_limit_policy = arko_agents::UsageLimitPolicy::WaitUntilReset {
        max_wait_ms: 3_600_000,
    };
    let resets_at = chrono::Utc::now().timestamp() + 60;
    let (error, _, restored, _, _, _workspace) = a_failed_report(
        Reply::Status {
            status: 429,
            retry_after: Some(60),
            body: json!({"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached", "plan_type": "pro", "resets_at": resets_at}}).to_string(),
        },
        config,
    )
    .await;
    assert!(
        matches!(
            &error,
            WorkflowError::AgentPaused {
                pause: AgentPause::ErrorRecovery {
                    cause: arko_agents::ErrorRecoveryCause::UsageLimit { .. },
                    ..
                },
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(restored.status, AgentRunStatus::Completed);
}

/// Stopping the run cancels the report: the caller gets the stop, and the
/// run is as it was before the request.
#[tokio::test]
async fn a_report_is_cancelled_by_stopping_the_run() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, body| {
        // The report's request waits until the run is stopped.
        (!body.contains(REQUESTED) || body.contains("Go on."))
            .then(|| text_turn(&format!("t{index}"), "Sure."))
    })
    .await;
    let workflow = Arc::new(
        builder(&server, &workspace, agent())
            .agent_tool("main", ProbeTool::new("probe", false))
            .build()
            .unwrap(),
    );
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let report = tokio::spawn({
        let (workflow, first) = (workflow.clone(), first.clone());
        async move { workflow.request_report(&first, None).await }
    });
    server.wait_for_requests(2).await;
    assert!(workflow.stop_run(
        first.agent_run_id,
        StopRequest::new(crate::workflow::stop::StopKind::Operator, "Ctrl-C"),
        RunStopIntent::Relaunch,
    ));
    let error = tokio::time::timeout(Duration::from_secs(20), report)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::AgentStopped { .. }),
        "{error}"
    );
    let restored = workflow.agent_result(first.agent_run_id).unwrap();
    assert_eq!(restored.status, AgentRunStatus::Completed);
    assert!(restored.stopped_by.is_none());
    assert_eq!(
        restored.task().unwrap().conversation.messages.len(),
        first.task().unwrap().conversation.messages.len()
    );
    workflow.continue_agent(&restored, "Go on.").await.unwrap();
}

/// A Claude Code agent writes the report in its session, as a reply, also
/// when it compacts in Claude Code, and no fresh session starts.
#[tokio::test]
async fn a_claude_code_agent_writes_a_report_on_request_in_its_session() {
    use super::recovery_tests::FakeClaude;
    let fake = FakeClaude::new(json!([
        [{"tool": "mcp__arko__complete_task", "input": {"summary": "Started."}}],
        [{"text": "REPORT: started the work."}],
    ]));
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let mut claude = fake.agent();
    claude.handover = Some(arko_agents::HandoverConfig {
        compact_in_claude_code: true,
        ..arko_agents::HandoverConfig::at_input_tokens(1)
    });
    let workflow = builder(&server, &workspace, claude).build().unwrap();
    let first = workflow.run_agent("main", "Start.").await.unwrap();
    let session = first.task().unwrap().claude_code_session.clone().unwrap();
    let result = workflow.request_report(&first, None).await.unwrap();
    assert_eq!(result.summary, "REPORT: started the work.");
    let task = result.task().unwrap();
    assert!(task.handovers.is_empty());
    assert_eq!(task.claude_code_session.as_deref(), Some(session.as_str()));
    let users = fake.logged("user");
    assert_eq!(users.len(), 2, "{users:?}");
    let input = users[1]["text"].as_str().unwrap();
    assert!(input.contains("Reply with it as plain text"), "{input}");
    assert!(!input.starts_with("/compact"), "{input}");
}
