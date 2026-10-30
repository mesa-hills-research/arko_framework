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

/// A Claude Code agent writes the report in a copy of its session, as a
/// reply, also when it compacts in Claude Code, and no fresh session starts.
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
    // Written in a copy of the session, which the run goes on in.
    assert_ne!(task.claude_code_session.as_deref(), Some(session.as_str()));
    let users = fake.logged("user");
    assert_eq!(users.len(), 2, "{users:?}");
    let input = users[1]["text"].as_str().unwrap();
    assert!(input.contains("Reply with it as plain text"), "{input}");
    assert!(!input.starts_with("/compact"), "{input}");
}

/// How a report ends that doesn't come.
#[derive(Clone, Copy, Debug)]
enum NoReport {
    /// Its request fails.
    Failed,
    /// The host stops the run.
    Stopped,
}

/// A peer's message that reaches the run while it writes a report that
/// doesn't come is held back, as during any report, and is given once when
/// the run goes on: in the same workflow, or after a resume from a
/// checkpoint.
#[tokio::test]
async fn a_message_during_a_report_that_doesn_t_come_is_given_once() {
    for (end, reload) in [
        (NoReport::Failed, false),
        (NoReport::Failed, true),
        (NoReport::Stopped, false),
        (NoReport::Stopped, true),
    ] {
        let workspace = Workspace::new();
        let release = Arc::new(tokio::sync::Notify::new());
        let server = {
            let release = release.clone();
            ResponsesServer::start_async(move |index, body| {
                let release = release.clone();
                Box::pin(async move {
                    if body.contains(REQUESTED) && !body.contains("Go on.") {
                        // Held until the message is in, then failed (or
                        // held for good, for a stop).
                        release.notified().await;
                        return matches!(end, NoReport::Failed).then(|| Reply::Status {
                            status: 401,
                            retry_after: None,
                            body: json!({"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}).to_string(),
                        });
                    }
                    Some(Reply::Stream(text_turn(&format!("t{index}"), "Sure.")))
                })
            })
            .await
        };
        let build = || {
            builder(&server, &workspace, agent())
                .agent("peer", agent_config("Peer"))
                .agent_tool("main", ProbeTool::new("probe", false))
        };
        let workflow = Arc::new(build().build().unwrap());
        let first = workflow.run_agent("main", "Hello.").await.unwrap();
        let report = tokio::spawn({
            let (workflow, first) = (workflow.clone(), first.clone());
            async move { workflow.request_report(&first, None).await }
        });
        server.wait_for_requests(2).await;
        workflow
            .context()
            .messages
            .send("peer", None, "main", "Use British spelling.".into(), vec![])
            .await
            .unwrap();
        // Taken by the run, which holds it back while the report is due.
        tokio::time::sleep(Duration::from_millis(300)).await;
        match end {
            NoReport::Failed => release.notify_one(),
            NoReport::Stopped => {
                assert!(workflow.stop_run(
                    first.agent_run_id,
                    StopRequest::new(crate::workflow::stop::StopKind::Operator, "Ctrl-C"),
                    RunStopIntent::Relaunch,
                ));
            }
        }
        tokio::time::timeout(Duration::from_secs(20), report)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        let workflow = if reload {
            workflow.checkpoint().await.unwrap();
            drop(workflow);
            Arc::new(
                build()
                    .resume_from(latest_checkpoint(&workspace))
                    .build()
                    .unwrap(),
            )
        } else {
            workflow
        };
        let restored = workflow.agent_result(first.agent_run_id).unwrap();
        workflow.continue_agent(&restored, "Go on.").await.unwrap();
        let input = bodies(&server).pop().unwrap()["input"].to_string();
        assert_eq!(
            input.matches("Use British spelling.").count(),
            1,
            "{end:?}, reload {reload}: {input}"
        );
    }
}

/// The latest checkpoint of the run directory.
fn latest_checkpoint(workspace: &Workspace) -> std::path::PathBuf {
    let run = workspace.0.join(".run");
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    run.join(name.trim())
}

/// Under the error policies that inform the agent too, a report's plan
/// limit and refused credential come back as the pause that says why, with
/// the agent set to wait for a plan's reset.
#[tokio::test]
async fn a_report_s_failure_says_why_under_every_error_policy() {
    for policy in [
        arko_agents::LLMErrorPolicy::InformAgentAndContinue,
        arko_agents::LLMErrorPolicy::RetryThenInformAgent { max_attempts: 2 },
    ] {
        let mut config = agent();
        config.llm_error_policy = policy.clone();
        config.llm_retry_config.usage_limit_policy =
            arko_agents::UsageLimitPolicy::WaitUntilReset {
                max_wait_ms: 3_600_000,
            };
        let (error, ..) = a_failed_report(
            Reply::Status {
                status: 401,
                retry_after: None,
                body: json!({"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}).to_string(),
            },
            config.clone(),
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
            "{policy:?}: {error}"
        );
        let resets_at = chrono::Utc::now().timestamp() + 60;
        let (error, ..) = a_failed_report(
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
            "{policy:?}: {error}"
        );
    }
}

/// The same for a Claude Code agent: a login Claude Code lacks, and a plan
/// limit it reports reached, under each error policy.
#[tokio::test]
async fn a_claude_code_report_s_failure_says_why_under_every_error_policy() {
    use super::recovery_tests::FakeClaude;
    let resets_at = chrono::Utc::now().timestamp() as u64 + 60;
    let failures = [
        json!([{"api_error": "authentication_failed", "text": "Not logged in · Please run /login", "status": null}]),
        json!([
            {"rate_limit": {"status": "rejected", "rateLimitType": "five_hour", "resetsAt": resets_at}},
            {"fail": "Claude usage limit reached."},
        ]),
    ];
    for policy in [
        arko_agents::LLMErrorPolicy::InformAgentAndContinue,
        arko_agents::LLMErrorPolicy::RetryThenInformAgent { max_attempts: 2 },
    ] {
        for (index, failure) in failures.iter().enumerate() {
            let fake = FakeClaude::new(json!([
                [{"tool": "mcp__arko__complete_task", "input": {"summary": "Started."}}],
                failure,
            ]));
            let workspace = Workspace::new();
            let server =
                ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure.")))
                    .await;
            let mut claude = fake.agent();
            claude.llm_error_policy = policy.clone();
            claude.llm_retry_config.usage_limit_policy =
                arko_agents::UsageLimitPolicy::WaitUntilReset {
                    max_wait_ms: 3_600_000,
                };
            let workflow = builder(&server, &workspace, claude).build().unwrap();
            let first = workflow.run_agent("main", "Start.").await.unwrap();
            let error = tokio::time::timeout(
                Duration::from_secs(20),
                workflow.request_report(&first, None),
            )
            .await
            .expect("the report doesn't wait")
            .unwrap_err();
            let WorkflowError::AgentPaused {
                pause:
                    AgentPause::ErrorRecovery {
                        cause, error_kind, ..
                    },
                ..
            } = &error
            else {
                panic!("{policy:?}: {error}");
            };
            if index == 0 {
                assert_eq!(*error_kind, Some(arko_ai::ErrorKind::Authentication));
            } else {
                assert!(
                    matches!(cause, arko_agents::ErrorRecoveryCause::UsageLimit { .. }),
                    "{cause:?}"
                );
            }
        }
    }
}

/// A Claude Code report that doesn't come leaves the session as it was,
/// since it was written in a copy: the run's next input goes on in the
/// session it had, which holds neither the request nor what the model wrote
/// of it. That holds for a session the runtime recorded no transcript entries
/// of too (saved by an older version, or compacted), which can't be cut back.
#[tokio::test]
async fn a_claude_code_report_that_doesn_t_come_leaves_the_session_as_it_was() {
    use super::recovery_tests::FakeClaude;
    for entries in [true, false] {
        let fake = FakeClaude::new(json!([
            [{"tool": "mcp__arko__complete_task", "input": {"summary": "Started."}}],
            [{"api_error": "authentication_failed", "text": "Not logged in · Please run /login", "status": null}],
            [{"tool": "mcp__arko__complete_task", "input": {"summary": "Went on."}}],
        ]));
        let workspace = Workspace::new();
        let server =
            ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
        let workflow = builder(&server, &workspace, fake.agent()).build().unwrap();
        let mut first = workflow.run_agent("main", "Start.").await.unwrap();
        let session = first.task().unwrap().claude_code_session.clone().unwrap();
        if !entries {
            let mut task = first.task().unwrap().clone();
            task.claude_code_entries.clear();
            first = AgentResult::of_task(first.agent_name.clone(), task);
        }
        workflow.request_report(&first, None).await.unwrap_err();
        let restored = workflow.agent_result(first.agent_run_id).unwrap();
        assert_eq!(
            restored.task().unwrap().claude_code_session.as_deref(),
            Some(session.as_str())
        );
        let result = workflow.continue_agent(&restored, "Go on.").await.unwrap();
        assert_eq!(result.summary, "Went on.");
        let users = fake.logged("user");
        assert_eq!(users.len(), 3, "{users:?}");
        let history = users[2]["history"].to_string();
        assert!(!history.contains("Write a report"), "{history}");
        assert!(!history.contains("Not logged in"), "{history}");
        assert!(history.contains("Start."), "{history}");
        let argv = fake.logged("argv")[2]["argv"].to_string();
        assert!(argv.contains(&format!("--resume={session}")), "{argv}");
        assert!(!argv.contains("--fork-session"), "{argv}");
    }
}

/// The agent of `agent()`, with the probe.
fn report_builder(server: &ResponsesServer, run_dir: &std::path::Path) -> WorkflowBuilder {
    Workflow::builder()
        .name("report-test")
        .workspace(run_dir.parent().unwrap())
        .llm_client(Arc::new(server.client()))
        .agent("main", agent())
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(run_dir))
}

/// A server that answers every request with a plan's usage limit.
async fn limited_server() -> ResponsesServer {
    ResponsesServer::start_async(|_, _| {
        Box::pin(async move {
            Some(Reply::Status {
                status: 429,
                retry_after: Some(3600),
                body: json!({"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached", "plan_type": "pro", "resets_at": chrono::Utc::now().timestamp() + 3600}}).to_string(),
            })
        })
    })
    .await
}

/// Runs the agent once in `workspace` (run directory `.run`), then asks it
/// for a report, with a note waiting for its next turn, which makes one call
/// (rejected) and then waits for the model until the process ends, after a
/// checkpoint. Returns the run as it was before the report, and the
/// checkpoint.
async fn a_report_under_way(workspace: &Workspace) -> (AgentResult, std::path::PathBuf) {
    let server = ResponsesServer::start(|index, body| {
        if !body.contains(REQUESTED) {
            return Some(text_turn(&format!("t{index}"), "Sure."));
        }
        (!body.contains("function_call_output")).then(|| tool_call_turn("p1", "probe", json!({})))
    })
    .await;
    let run_dir = workspace.0.join(".run");
    let workflow = Arc::new(report_builder(&server, &run_dir).build().unwrap());
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let mut waiting = first.task().unwrap().clone();
    waiting.pending_context_injections = vec!["A held note: use tabs.".into()];
    let before = AgentResult::of_task(first.agent_name.clone(), waiting);
    let report = tokio::spawn({
        let (workflow, before) = (workflow.clone(), before.clone());
        async move { workflow.request_report(&before, None).await }
    });
    server.wait_for_requests(3).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    workflow.checkpoint().await.unwrap();
    let checkpoint = latest_checkpoint(workspace);
    // The process ends with the report under way.
    report.abort();
    let _ = report.await;
    drop(workflow);
    (before, checkpoint)
}

/// A report under way when the workflow stops is never resumed. The
/// checkpoint keeps the run as it was before the request, and a resume puts
/// it back so, with the model calls and the context the report run took.
/// Nothing waits, and no model is asked: the resumed server answers every
/// request with a plan limit, which a recovery or a new report gets at once.
#[tokio::test]
async fn a_report_under_way_at_a_checkpoint_is_put_back_on_resume() {
    let workspace = Workspace::new();
    let run_dir = workspace.0.join(".run");
    let (before, checkpoint) = a_report_under_way(&workspace).await;
    let saved = crate::persistence::PersistenceManager::load_checkpoint(&checkpoint)
        .await
        .unwrap();
    assert_eq!(saved.report_baselines.len(), 1);
    let interrupted = saved
        .agent_results_in(&run_dir)
        .unwrap()
        .into_iter()
        .find(|result| result.agent_run_id == before.agent_run_id)
        .unwrap();
    assert_eq!(interrupted.status, AgentRunStatus::Interrupted);

    let limited = limited_server().await;
    let resumed = report_builder(&limited, &run_dir)
        .resume_from(&checkpoint)
        .build()
        .unwrap();
    assert!(resumed.interrupted_runs().is_empty());
    let restored = resumed.agent_result(before.agent_run_id).unwrap();
    assert_eq!(restored.status, AgentRunStatus::Completed);
    assert_eq!(restored.summary, before.summary);
    let (was, now) = (before.task().unwrap(), restored.task().unwrap());
    assert_eq!(
        now.conversation.messages.len(),
        was.conversation.messages.len()
    );
    assert!(now.handover_requested.is_none());
    assert_eq!(now.pending_context_injections, ["A held note: use tabs."]);
    assert_eq!(now.usage_history.len(), was.usage_history.len() + 1);

    // Recovering the result read before the resume returns the run as it
    // was, at once.
    let recovered =
        tokio::time::timeout(Duration::from_secs(5), resumed.recover_agent(&interrupted))
            .await
            .expect("nothing waits")
            .unwrap();
    assert_eq!(recovered.status, AgentRunStatus::Completed);
    assert_eq!(limited.requests(), 0);
    // A new report gets the plan limit at once.
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        resumed.request_report(&restored, None),
    )
    .await
    .expect("nothing waits")
    .unwrap_err();
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
    // The checkpoint after it saves the run as it was put back.
    resumed.checkpoint().await.unwrap();
    let latest =
        crate::persistence::PersistenceManager::load_checkpoint(&latest_checkpoint(&workspace))
            .await
            .unwrap();
    assert!(latest.report_baselines.is_empty());
    let saved_run = latest
        .agent_results_in(&run_dir)
        .unwrap()
        .into_iter()
        .find(|result| result.agent_run_id == before.agent_run_id)
        .unwrap();
    assert_eq!(saved_run.status, AgentRunStatus::Completed);
    assert_eq!(
        saved_run.task().unwrap().pending_context_injections,
        ["A held note: use tabs."]
    );
}

/// A run saved writing a report without the run as it was before (a
/// checkpoint that doesn't keep it) is refused by a recovery, which never
/// resumes a report, and no model is asked.
#[tokio::test]
async fn a_report_saved_without_the_run_before_it_isnt_recovered() {
    let workspace = Workspace::new();
    let run_dir = workspace.0.join(".run");
    let (before, checkpoint) = a_report_under_way(&workspace).await;
    let mut state: Value =
        serde_json::from_str(&std::fs::read_to_string(&checkpoint).unwrap()).unwrap();
    state.as_object_mut().unwrap().remove("report_baselines");
    let without = run_dir.join("without-baseline.json");
    std::fs::write(&without, state.to_string()).unwrap();
    let limited = limited_server().await;
    let resumed = report_builder(&limited, &run_dir)
        .resume_from(&without)
        .build()
        .unwrap();
    let interrupted = resumed.interrupted_runs();
    assert_eq!(interrupted.len(), 1);
    assert_eq!(interrupted[0].agent_run_id, before.agent_run_id);
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        resumed.recover_agent(&interrupted[0]),
    )
    .await
    .expect("nothing waits")
    .unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("writing a report")),
        "{error}"
    );
    assert_eq!(limited.requests(), 0);
}
