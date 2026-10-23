//! Graceful stops: the wrap-up, blameless results, restarting what a stop
//! ended after a resume, the signal helper, and saved stops.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arko_agents::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::*;
use crate::hooks::{HookPoint, HookResult};
use crate::persistence::PersistenceConfig;
use crate::persistence::state::WorkflowState;
use crate::test_support::{
    Reply, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::cancel::CancelCause;
use crate::workflow::result::AgentRunStatus;
use crate::workflow::stop::{StopKind, StopRecord, wrap_up_notice};

/// What the agents are told when a wrap-up begins.
const NOTICE: &str = "The workflow is stopping";
/// What a run continued after a stop's wrap-up is told.
const RESTARTED: &str = "has now restarted";
/// What a reopened run is told.
const REOPENED: &str = "has now been reopened";

/// A tool that works until its task's stop signal fires, then saves and
/// returns what it was told.
#[derive(Debug, Default)]
struct Saver {
    started: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl Tool for Saver {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "saver".to_string(),
            description: "Works until stopped.".to_string(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.started.notify_one();
        context.stop.stopping().await;
        Ok(format!(
            "saved ({})",
            context.stop.reason().unwrap_or_default()
        ))
    }
}

/// The `OnWorkflowCancelled` hooks fired, with their metadata.
#[derive(Clone, Default)]
struct Cancelled(Arc<parking_lot::Mutex<Vec<HashMap<String, Value>>>>);

impl Cancelled {
    fn register(&self, builder: WorkflowBuilder) -> WorkflowBuilder {
        let fired = self.0.clone();
        builder.hook(
            HookPoint::OnWorkflowCancelled,
            Arc::new(move |hook| {
                fired.lock().push(hook.metadata.clone());
                HookResult::Continue
            }),
        )
    }

    fn fired(&self) -> Vec<HashMap<String, Value>> {
        self.0.lock().clone()
    }
}

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("stop-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent("reviewer", agent_config("Friendly Reviewer"))
}

fn persistence(workspace: &Workspace) -> PersistenceConfig {
    PersistenceConfig::new(workspace.0.join(".run"))
}

/// The checkpoint `latest-checkpoint` points to, as saved and as loaded.
fn latest_checkpoint(workspace: &Workspace) -> (std::path::PathBuf, Value, WorkflowState) {
    let run = workspace.0.join(".run");
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    let path = run.join(name.trim());
    let text = std::fs::read_to_string(&path).unwrap();
    (
        path,
        serde_json::from_str(&text).unwrap(),
        serde_json::from_str(&text).unwrap(),
    )
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("the workflow call returned")
}

/// A server for the coder (`CODER-TASK`: it calls `saver`, and completes
/// once told the run is stopping), a reviewer that never gets an answer
/// (`REVIEWER-TASK`), and runs restarted after a stop, which complete.
async fn server() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        if body.contains(RESTARTED) || body.contains(REOPENED) {
            return Some(complete_turn("again", "Finished."));
        }
        if body.contains("EARLY-TASK") {
            return Some(complete_turn("early", "Done early."));
        }
        if body.contains("REVIEWER-TASK") {
            return None;
        }
        Some(if body.contains(NOTICE) {
            complete_turn("done", "Stopped at step 3.")
        } else {
            tool_call_turn("save", "saver", json!({}))
        })
    })
    .await
}

fn drain_record(result: &AgentResult) -> &StopRecord {
    let record = result.stopped_by().expect("the stop ended the run");
    assert_eq!(record.kind, StopKind::Drain);
    assert_eq!(record.reason, "deploying a fix");
    record
}

/// With a wrap-up, no new run starts, running tools see their stop signal,
/// and agents are told to finish: one completes. Cancelling then ends the
/// wrap-up at once with the stop's cause; the agent still running is
/// cancelled, the stop is checkpointed (in the form older versions read
/// as a cancellation), and both results are marked as ended by the stop.
#[tokio::test]
async fn a_wrap_up_lets_agents_finish_then_the_run_is_cancelled_and_checkpointed() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let hooks = Cancelled::default();
    let workflow = Arc::new(
        hooks
            .register(builder(&server, &workspace))
            .global_tool(saver.clone())
            .persistence(persistence(&workspace))
            .build()
            .unwrap(),
    );
    let early = within(workflow.run_agent("coder", "EARLY-TASK"))
        .await
        .unwrap();
    assert_eq!(early.stopped_by(), None, "it ended before the stop");
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    within(saver.started.notified()).await;
    server.wait_for_requests(3).await;

    let request = StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(600));
    assert!(workflow.stop(request));
    assert!(workflow.is_stopping());
    assert!(!workflow.is_cancelled());
    assert!(
        !workflow.stop(StopRequest::operator("again").wrap_up(Duration::from_secs(1))),
        "the first stop stands"
    );
    // No run starts during the wrap-up.
    assert!(matches!(
        workflow.run_agent("coder", "EARLY-TASK").await,
        Err(WorkflowError::Cancelled)
    ));

    // The coder's tool saw its stop signal, and the coder was told to
    // finish; it completed, blamelessly.
    let coded = within(coder).await.unwrap().unwrap();
    assert_eq!(coded.status, AgentRunStatus::Completed);
    assert_eq!(coded.summary, "Stopped at step 3.");
    drain_record(&coded);
    let saved = coded
        .task()
        .unwrap()
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(arko_ai::Content::as_tool_result)
        .find(|result| result.call_id.as_deref() == Some("call_save"))
        .unwrap();
    assert_eq!(
        saved.output,
        "saved (Workflow stopped (drain): deploying a fix.)"
    );
    let notice = wrap_up_notice("deploying a fix", Duration::from_secs(600));
    let told = server
        .bodies()
        .iter()
        .filter(|body| body.contains(&notice))
        .count();
    assert_eq!(told, 1, "the coder is told once, the reviewer never asked");
    assert!(!workflow.is_cancelled(), "the reviewer still runs");

    // Cancelling ends the wrap-up; the stop's cause stands.
    assert!(workflow.cancel());
    assert!(matches!(
        within(reviewer).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    assert!(!workflow.is_stopping());
    let stopped = CancelCause::Stopped {
        kind: StopKind::Drain,
        reason: "deploying a fix".to_string(),
    };
    assert_eq!(workflow.cancellation().unwrap().cause, stopped);
    let results = workflow.result().agent_results;
    let reviewed = results
        .iter()
        .find(|result| result.agent_name == "reviewer")
        .unwrap();
    assert_eq!(reviewed.status, AgentRunStatus::Cancelled);
    assert_eq!(
        reviewed.error.as_deref(),
        Some("Workflow stopped (drain): deploying a fix.")
    );
    drain_record(reviewed);
    let fired = hooks.fired();
    assert_eq!(fired.len(), 1);
    assert_eq!(fired[0]["stop_kind"], "drain");
    assert_eq!(fired[0]["reason"], "deploying a fix");

    // The checkpoint records the stop and the marks; the stop is saved as
    // a cancellation with the stop beside it.
    let (_, saved, state) = latest_checkpoint(&workspace);
    assert_eq!(saved["cancellation"]["cause"], "Requested");
    assert_eq!(saved["cancellation"]["stop"]["kind"], "drain");
    assert_eq!(state.cancellation.unwrap().cause, stopped);
    let marked = state
        .agent_runs
        .iter()
        .map(|record| (record.summary.as_str(), record.stopped_by.is_some()))
        .collect::<Vec<_>>();
    assert_eq!(
        marked,
        [
            ("Done early.", false),
            ("Stopped at step 3.", true),
            ("", true)
        ]
    );
}

/// The wrap-up ends once no agent runs any more, long before its time:
/// the run is stopped then, and the last run's call returns the stop.
#[tokio::test]
async fn a_wrap_up_ends_when_the_last_agent_finishes() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(saver.clone())
            .build()
            .unwrap(),
    );
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    within(saver.started.notified()).await;
    assert!(
        workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(3_600)))
    );
    assert!(matches!(
        within(coder).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    assert!(workflow.is_cancelled());
    let result = &workflow.result().agent_results[0];
    assert_eq!(result.status, AgentRunStatus::Completed);
    drain_record(result);
    assert!(matches!(
        within(workflow.finish()).await,
        Err(WorkflowError::Cancelled)
    ));
}

/// An agent that doesn't finish is cancelled when the wrap-up's time is up.
#[tokio::test]
async fn a_wrap_up_ends_when_its_time_is_up() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = Arc::new(builder(&server, &workspace).build().unwrap());
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    server.wait_for_requests(1).await;
    let started = Instant::now();
    assert!(workflow.stop(StopRequest::limit("out of time").wrap_up(Duration::from_millis(300))));
    assert!(matches!(
        within(reviewer).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    assert!(started.elapsed() >= Duration::from_millis(300));
    let result = &workflow.result().agent_results[0];
    assert_eq!(result.status, AgentRunStatus::Cancelled);
    assert_eq!(
        result.stopped_by().map(|stop| stop.kind),
        Some(StopKind::Limit)
    );
}

/// Without a wrap-up a stop cancels at once, like `cancel`, with its own
/// cause; with nothing running a wrap-up stops at once.
#[tokio::test]
async fn a_stop_without_a_wrap_up_or_with_nothing_running_stops_at_once() {
    let workspace = Workspace::new();
    let server = server().await;
    let hooks = Cancelled::default();
    let workflow = Arc::new(
        hooks
            .register(builder(&server, &workspace))
            .build()
            .unwrap(),
    );
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    server.wait_for_requests(1).await;
    assert!(workflow.stop(StopRequest::operator("Ctrl-C")));
    assert!(workflow.is_cancelled());
    assert!(!workflow.stop(StopRequest::operator("again")));
    assert!(matches!(
        within(reviewer).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let result = &workflow.result().agent_results[0];
    assert_eq!(
        result
            .stopped_by()
            .map(|stop| (stop.kind, stop.reason.as_str())),
        Some((StopKind::Operator, "Ctrl-C"))
    );
    assert_eq!(
        result.error.as_deref(),
        Some("Workflow stopped (operator): Ctrl-C.")
    );
    assert_eq!(hooks.fired().len(), 1);
    assert_eq!(hooks.fired()[0]["stop_kind"], "operator");

    let idle = builder(&server, &workspace).build().unwrap();
    assert!(idle.stop(StopRequest::failure("lost the login").wrap_up(Duration::from_secs(60))));
    assert!(idle.is_cancelled());
    assert!(!idle.is_stopping());
    assert_eq!(
        idle.cancellation().unwrap().cause,
        CancelCause::Stopped {
            kind: StopKind::Failure,
            reason: "lost the login".to_string()
        }
    );
}

/// After resuming from a stopped run's checkpoint, `stopped_runs` lists the
/// runs the stop ended (not the one that finished before it), and
/// `restart` relaunches each: the completed one is continued, the cancelled
/// one reopened with the note. Restarted, they are no longer listed.
#[tokio::test]
async fn restart_after_resume_relaunches_the_runs_a_stop_ended() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(saver.clone())
            .persistence(persistence(&workspace))
            .build()
            .unwrap(),
    );
    within(workflow.run_agent("coder", "EARLY-TASK"))
        .await
        .unwrap();
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    within(saver.started.notified()).await;
    server.wait_for_requests(3).await;
    assert!(workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(600))));
    within(coder).await.unwrap().unwrap();
    // A second stop without a wrap-up ends the wrap-up at once.
    assert!(workflow.stop(StopRequest::drain("now")));
    assert!(within(reviewer).await.unwrap().is_err());
    let (path, _, _) = latest_checkpoint(&workspace);
    drop(workflow);

    let resumed = builder(&server, &workspace)
        .global_tool(Arc::new(Saver::default()))
        .persistence(persistence(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let stopped = resumed.stopped_runs();
    let listed = stopped
        .iter()
        .map(|run| (run.result.agent_name.as_str(), run.restart))
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [("coder", Restart::Continue), ("reviewer", Restart::Reopen)]
    );
    for run in &stopped {
        drain_record(&run.result);
        let result = within(resumed.restart(run, "Carry on.")).await.unwrap();
        assert_eq!(result.status, AgentRunStatus::Completed);
        assert_eq!(result.agent_run_id, run.result.agent_run_id);
        assert_eq!(result.stopped_by(), None);
    }
    assert!(resumed.stopped_runs().is_empty());
    let bodies = server.bodies();
    let continued = bodies.iter().find(|body| body.contains(RESTARTED)).unwrap();
    assert!(continued.contains("The workflow was stopped (deploying a fix) while you worked"));
    assert!(continued.contains("Carry on."));
    let reopened = bodies.iter().find(|body| body.contains(REOPENED)).unwrap();
    assert!(reopened.contains("Carry on."));
}

/// `stop_on_signals` stops the run with its request on `SIGTERM`.
#[cfg(unix)]
#[tokio::test]
async fn stop_on_signals_stops_the_run_on_sigterm() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = Arc::new(builder(&server, &workspace).build().unwrap());
    let listener = crate::workflow::stop::stop_on_signals(
        workflow.cancel_handle(),
        StopRequest::operator("signal").wrap_up(Duration::from_secs(60)),
    )
    .unwrap();
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    server.wait_for_requests(1).await;
    let status = std::process::Command::new("kill")
        .arg("-TERM")
        .arg(std::process::id().to_string())
        .status()
        .unwrap();
    assert!(status.success());
    within(async {
        while !workflow.is_stopping() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    // The reviewer doesn't finish: end the wrap-up.
    assert!(workflow.cancel());
    assert!(within(reviewer).await.unwrap().is_err());
    assert_eq!(
        workflow.cancellation().unwrap().cause,
        CancelCause::Stopped {
            kind: StopKind::Operator,
            reason: "signal".to_string()
        }
    );
    listener.abort();
}

/// Saved data stays loadable both ways: a checkpoint from before graceful
/// stops loads (a plain cancellation, results without stop marks), and a
/// stop this version saves reads, in the form older versions know, as a
/// plain cancellation.
#[test]
fn checkpoints_from_before_stops_load_and_older_versions_read_a_stop() {
    let old = json!({
        "workflow_name": "stop-test",
        "checkpoint_id": uuid::Uuid::new_v4(),
        "timestamp": "2026-09-01T12:00:00Z",
        "memory_snapshot": [],
        "message_queues": {},
        "budget_spent": 0.0,
        "agent_runs": [{
            "agent_run_id": uuid::Uuid::new_v4(),
            "agent_name": "coder",
            "status": "Cancelled",
            "summary": "",
            "error": "Workflow cancelled.",
            "task_file": "agents/x/y.json"
        }],
        "cancellation": {"cause": "Requested", "timestamp": "2026-09-01T12:00:00Z"}
    });
    let state: WorkflowState = serde_json::from_value(old).unwrap();
    assert_eq!(state.cancellation.unwrap().cause, CancelCause::Requested);
    assert_eq!(state.agent_runs[0].stopped_by, None);

    // A stop as this version saves it, read by a version that knows only
    // these causes.
    #[derive(serde::Deserialize, Debug, PartialEq)]
    #[allow(dead_code)]
    enum OldCause {
        Requested,
        AbortedByHook { hook_point: String, reason: String },
    }
    #[derive(serde::Deserialize)]
    struct OldCancellation {
        cause: OldCause,
        #[allow(dead_code)]
        timestamp: chrono::DateTime<chrono::Utc>,
    }
    let stop = crate::workflow::cancel::Cancellation {
        cause: CancelCause::Stopped {
            kind: StopKind::Limit,
            reason: "deadline".to_string(),
        },
        timestamp: chrono::Utc::now(),
    };
    let saved = serde_json::to_value(&stop).unwrap();
    let old: OldCancellation = serde_json::from_value(saved.clone()).unwrap();
    assert_eq!(old.cause, OldCause::Requested);
    let read: crate::workflow::cancel::Cancellation = serde_json::from_value(saved).unwrap();
    assert_eq!(read, stop);
    // Other causes are saved as before.
    let hook = crate::workflow::cancel::Cancellation {
        cause: CancelCause::AbortedByHook {
            hook_point: "PreToolExecution".to_string(),
            reason: "no".to_string(),
        },
        timestamp: chrono::Utc::now(),
    };
    let saved = serde_json::to_value(&hook).unwrap();
    assert!(saved.get("stop").is_none());
    let old: OldCancellation = serde_json::from_value(saved).unwrap();
    assert!(matches!(old.cause, OldCause::AbortedByHook { .. }));
}

/// A completion verifier's request.
const VERIFIER: &str = "You are a completion verifier";

/// The output of call `call_id` in `result`'s conversation.
fn tool_output(result: &AgentResult, call_id: &str) -> Option<String> {
    result
        .task()
        .unwrap()
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(arko_ai::Content::as_tool_result)
        .find(|result| result.call_id.as_deref() == Some(call_id))
        .map(|result| result.output.clone())
}

fn checks_coder(builder: WorkflowBuilder) -> WorkflowBuilder {
    builder.completion_verifier(
        crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
            .target("coder"),
    )
}

/// A completion made after the wrap-up notice reports progress: its
/// completion verifier doesn't review it (it would reject it as
/// incomplete, failing the run), the run is marked as ended by the stop,
/// and after a resume it is continued, and its next completion reviewed.
#[tokio::test]
async fn a_completion_after_the_notice_is_reviewed_only_once_the_run_is_continued() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        if body.contains(VERIFIER) {
            return Some(complete_turn(
                "verdict",
                if body.contains("steps 4 and 5 are left") {
                    "REJECTED_MAJOR\nSteps 4 and 5 of the task are not done."
                } else {
                    "ACCEPTED"
                },
            ));
        }
        if body.contains(RESTARTED) {
            return Some(complete_turn("again", "All five steps done."));
        }
        Some(if body.contains(NOTICE) {
            complete_turn("done", "Stopped at step 3; steps 4 and 5 are left.")
        } else {
            tool_call_turn("save", "saver", json!({}))
        })
    })
    .await;
    let saver = Arc::new(Saver::default());
    let build = |resume: Option<&std::path::Path>| {
        let mut builder = checks_coder(builder(&server, &workspace))
            .global_tool(saver.clone())
            .persistence(persistence(&workspace));
        if let Some(path) = resume {
            builder = builder.resume_from(path);
        }
        builder.build().unwrap()
    };
    let workflow = Arc::new(build(None));
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    within(saver.started.notified()).await;
    assert!(workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(600))));
    // The last agent to end ends the wrap-up: its call returns the stop.
    assert!(matches!(
        within(coder).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let results = workflow.result().agent_results;
    assert_eq!(results.len(), 1, "no review ran");
    assert_eq!(results[0].status, AgentRunStatus::Completed);
    assert_eq!(
        results[0].summary,
        "Stopped at step 3; steps 4 and 5 are left."
    );
    drain_record(&results[0]);
    assert!(!server.bodies().iter().any(|body| body.contains(VERIFIER)));
    let (path, _, _) = latest_checkpoint(&workspace);
    drop(workflow);

    let resumed = build(Some(&path));
    let stopped = resumed.stopped_runs();
    let listed = stopped
        .iter()
        .map(|run| (run.result.agent_name.as_str(), run.restart))
        .collect::<Vec<_>>();
    assert_eq!(listed, [("coder", Restart::Continue)]);
    let result = within(resumed.restart(&stopped[0], "")).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.summary, "All five steps done.");
    assert_eq!(result.stopped_by(), None);
    let reviews = server
        .bodies()
        .iter()
        .filter(|body| body.contains(VERIFIER))
        .count();
    assert_eq!(reviews, 1, "the continued run's completion is reviewed");
}

/// A delegated run that completes after the wrap-up notice reports
/// progress, unreviewed: `stopped_runs` leaves it to its caller, so the
/// caller's `run_workflow_agent` result names it after the progress
/// report. The restarted caller resumes it with `resume_run`, and its next
/// completion is reviewed.
#[tokio::test]
async fn a_delegated_run_told_to_wrap_up_is_named_to_its_caller_and_reviewed_once_resumed() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        if body.contains(VERIFIER) {
            return Some(complete_turn(
                "verdict",
                if body.contains("Coder finished all five.") {
                    "ACCEPTED"
                } else {
                    "REJECTED_MAJOR\nSteps 4 and 5 of the task are not done."
                },
            ));
        }
        if body.contains("LEAD-TASK") {
            if body.contains("Coder finished all five.") {
                return Some(complete_turn("lead-again", "Lead finished."));
            }
            if body.contains(RESTARTED) {
                // The run its delegation result named.
                let end = body
                    .find(" of agent 'coder' stopped with the workflow")
                    .expect("the delegation result names the coder's run");
                let run_id = &body[end - 36..end];
                return Some(tool_call_turn(
                    "resume",
                    crate::services::RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "coder", "resume_run": run_id, "prompt": "Carry on."}),
                ));
            }
            if body.contains(NOTICE) {
                return Some(complete_turn("lead-done", "Lead stopped."));
            }
            return Some(tool_call_turn(
                "delegate",
                crate::services::RUN_WORKFLOW_AGENT_TOOL_NAME,
                json!({"agent": "coder", "prompt": "CODER-TASK"}),
            ));
        }
        Some(if body.contains("Carry on.") {
            complete_turn("again", "Coder finished all five.")
        } else if body.contains(NOTICE) {
            complete_turn("done", "Stopped at step 3; steps 4 and 5 are left.")
        } else {
            tool_call_turn("save", "saver", json!({}))
        })
    })
    .await;
    let saver = Arc::new(Saver::default());
    let build = |resume: Option<&std::path::Path>| {
        let mut builder = checks_coder(builder(&server, &workspace))
            .agent("lead", agent_config("Lead"))
            .workflow_services(
                crate::services::WorkflowServicesConfig::default()
                    .allow_caller("lead")
                    .allow_target("coder"),
            )
            .global_tool(saver.clone())
            .persistence(persistence(&workspace));
        if let Some(path) = resume {
            builder = builder.resume_from(path);
        }
        builder.build().unwrap()
    };
    let workflow = Arc::new(build(None));
    let lead = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "LEAD-TASK").await }
    });
    within(saver.started.notified()).await;
    assert!(workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(600))));
    let _ = within(lead).await.unwrap();
    let results = workflow.result().agent_results;
    let find = |agent: &str| {
        results
            .iter()
            .find(|result| result.agent_name == agent)
            .unwrap()
            .clone()
    };
    let (coder, lead) = (find("coder"), find("lead"));
    assert_eq!(coder.status, AgentRunStatus::Completed);
    assert_eq!(coder.summary, "Stopped at step 3; steps 4 and 5 are left.");
    drain_record(&coder);
    assert!(!server.bodies().iter().any(|body| body.contains(VERIFIER)));
    // The caller gets the progress report, then the note naming the run.
    let id = coder.agent_run_id;
    assert_eq!(
        tool_output(&lead, "call_delegate").unwrap(),
        format!(
            "Stopped at step 3; steps 4 and 5 are left.\n\nAgent run {id} of agent 'coder' stopped with the workflow before it finished: its summary is a progress report that no completion verifier reviewed. You can resume it instead of starting over: call run_workflow_agent with agent \"coder\", resume_run \"{id}\" and a prompt telling it what happened and what to do next; it continues its own task."
        )
    );
    let (path, _, _) = latest_checkpoint(&workspace);
    drop(workflow);

    let resumed = build(Some(&path));
    let stopped = resumed.stopped_runs();
    let listed = stopped
        .iter()
        .map(|run| (run.result.agent_name.as_str(), run.restart))
        .collect::<Vec<_>>();
    assert_eq!(listed, [("lead", Restart::Continue)], "left to its caller");
    let restarted = within(resumed.restart(&stopped[0], "")).await.unwrap();
    assert_eq!(restarted.status, AgentRunStatus::Completed);
    assert_eq!(restarted.summary, "Lead finished.");
    assert_eq!(
        tool_output(&restarted, "call_resume").unwrap(),
        "Coder finished all five.",
        "a reviewed completion carries no note"
    );
    let continued = resumed
        .result()
        .agent_results
        .into_iter()
        .find(|result| result.agent_run_id == id)
        .unwrap();
    assert_eq!(continued.status, AgentRunStatus::Completed);
    assert_eq!(continued.summary, "Coder finished all five.");
    assert_eq!(continued.stopped_by(), None);
    let reviews = server
        .bodies()
        .into_iter()
        .filter(|body| body.contains(VERIFIER))
        .collect::<Vec<_>>();
    assert_eq!(
        reviews.len(),
        1,
        "the continued run's completion is reviewed"
    );
    assert!(reviews[0].contains("Coder finished all five."));
}

/// A completion the agent made in the wrap-up before it was told (its
/// model answered after the stop began) is its own: the completion
/// verifier's review may start then, isn't told to wrap up, and decides
/// it.
#[tokio::test]
async fn a_completion_made_before_the_notice_is_reviewed_in_the_wrap_up() {
    let workspace = Workspace::new();
    let release = Arc::new(tokio::sync::Notify::new());
    let server = ResponsesServer::start_async({
        let release = release.clone();
        move |_, body| {
            let release = release.clone();
            Box::pin(async move {
                if body.contains(VERIFIER) {
                    return Some(Reply::Stream(complete_turn("verdict", "ACCEPTED")));
                }
                // The coder's answer comes once the wrap-up has begun.
                release.notified().await;
                Some(Reply::Stream(complete_turn("done", "All done.")))
            })
        }
    })
    .await;
    let workflow = Arc::new(checks_coder(builder(&server, &workspace)).build().unwrap());
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    server.wait_for_requests(1).await;
    assert!(workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(600))));
    release.notify_one();
    assert!(matches!(
        within(coder).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let bodies = server.bodies();
    let reviews = bodies
        .iter()
        .filter(|body| body.contains(VERIFIER))
        .collect::<Vec<_>>();
    assert_eq!(reviews.len(), 1, "the review started in the wrap-up");
    assert!(
        !reviews[0].contains(NOTICE),
        "a review isn't told to wrap up"
    );
    let results = workflow.result().agent_results;
    let coded = results
        .iter()
        .find(|result| result.agent_name == "coder")
        .unwrap();
    assert_eq!(coded.status, AgentRunStatus::Completed);
    assert_eq!(coded.summary, "All done.");
    assert_eq!(coded.stopped_by(), None, "a reviewed completion is its own");
    assert!(workflow.is_cancelled());
}

/// During a wrap-up an agent's `run_workflow_agent` call starts nothing
/// and fails with `Workflow cancelled`, also when its target is a
/// completion verifier's agent (only a review may start).
async fn delegating_in_a_wrap_up_is_refused(target_verifies: bool) {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        if !body.contains("CODER-TASK") {
            return body
                .contains("LATE-TASK")
                .then(|| complete_turn("late", "Done late."));
        }
        // The call and its result.
        if body.matches("call_late").count() >= 2 {
            return Some(complete_turn("done", "Stopped."));
        }
        Some(if body.contains(NOTICE) {
            tool_call_turn(
                "late",
                crate::services::RUN_WORKFLOW_AGENT_TOOL_NAME,
                json!({"agent": "reviewer", "prompt": "LATE-TASK"}),
            )
        } else {
            tool_call_turn("save", "saver", json!({}))
        })
    })
    .await;
    let saver = Arc::new(Saver::default());
    let mut builder = builder(&server, &workspace)
        .agent("writer", agent_config("Friendly Writer"))
        .global_tool(saver.clone())
        .workflow_services(
            crate::services::WorkflowServicesConfig::default()
                .allow_caller("coder")
                .allow_target("reviewer"),
        );
    if target_verifies {
        builder = builder.completion_verifier(
            crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
                .target("writer"),
        );
    }
    let workflow = Arc::new(builder.build().unwrap());
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    within(saver.started.notified()).await;
    assert!(workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(600))));
    let _ = within(coder).await.unwrap();
    let results = workflow.result().agent_results;
    assert!(results.iter().all(|result| result.agent_name != "reviewer"));
    assert!(
        !server
            .bodies()
            .iter()
            .any(|body| body.contains("LATE-TASK") && !body.contains("CODER-TASK")),
        "no run started"
    );
    let coded = results
        .iter()
        .find(|result| result.agent_name == "coder")
        .unwrap();
    let late = tool_output(coded, "call_late").unwrap();
    assert!(late.contains("Workflow cancelled"), "{late}");
}

#[tokio::test]
async fn delegating_in_a_wrap_up_is_refused_for_an_ordinary_target() {
    delegating_in_a_wrap_up_is_refused(false).await;
}

#[tokio::test]
async fn delegating_in_a_wrap_up_is_refused_for_a_verifiers_agent() {
    delegating_in_a_wrap_up_is_refused(true).await;
}

/// `finish` during a wrap-up waits for the run to stop, and returns the
/// stop.
#[tokio::test]
async fn finish_during_a_wrap_up_waits_for_the_stop() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = Arc::new(builder(&server, &workspace).build().unwrap());
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    server.wait_for_requests(1).await;
    let started = Instant::now();
    assert!(workflow.stop(StopRequest::limit("out of time").wrap_up(Duration::from_millis(500))));
    assert!(matches!(
        within(workflow.finish()).await,
        Err(WorkflowError::Cancelled)
    ));
    assert!(
        started.elapsed() >= Duration::from_millis(500),
        "finish waited for the wrap-up"
    );
    assert!(within(reviewer).await.unwrap().is_err());
}

/// A delegation in flight when the stop cancels it: the caller's call names
/// the delegated run, and after the resume only the caller is listed;
/// restarted, it resumes that run (`resume_run`) instead of starting the
/// work again.
#[tokio::test]
async fn a_delegation_the_stop_cancelled_is_resumed_by_its_restarted_caller() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        if !body.contains("CODER-TASK") {
            // The delegated run: answered once reopened.
            return (body.contains("CHILD-TASK") && body.contains(REOPENED))
                .then(|| complete_turn("child", "Child finished."));
        }
        if body.contains("Child finished.") {
            return Some(complete_turn("again", "Coder finished."));
        }
        if body.contains(REOPENED) {
            // Resumes the run its failed call named.
            let end = body.find(" of agent 'reviewer' was stopped")?;
            let run_id = &body[end - 36..end];
            return Some(tool_call_turn(
                "resume",
                crate::services::RUN_WORKFLOW_AGENT_TOOL_NAME,
                json!({"agent": "reviewer", "resume_run": run_id, "prompt": "Carry on."}),
            ));
        }
        Some(tool_call_turn(
            "delegate",
            crate::services::RUN_WORKFLOW_AGENT_TOOL_NAME,
            json!({"agent": "reviewer", "prompt": "CHILD-TASK"}),
        ))
    })
    .await;
    let child_requests = |server: &ResponsesServer| {
        server
            .bodies()
            .into_iter()
            .filter(|body| body.contains("CHILD-TASK") && !body.contains("CODER-TASK"))
            .collect::<Vec<_>>()
    };
    let build = |resume: Option<&std::path::Path>| {
        let mut builder = builder(&server, &workspace)
            .workflow_services(
                crate::services::WorkflowServicesConfig::default()
                    .allow_caller("coder")
                    .allow_target("reviewer"),
            )
            .persistence(persistence(&workspace));
        if let Some(path) = resume {
            builder = builder.resume_from(path);
        }
        builder.build().unwrap()
    };
    let workflow = Arc::new(build(None));
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    within(async {
        while child_requests(&server).is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_millis(300)))
    );
    assert!(matches!(
        within(coder).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let results = workflow.result().agent_results;
    let child = results
        .iter()
        .find(|result| result.agent_name == "reviewer")
        .unwrap()
        .clone();
    assert_eq!(child.status, AgentRunStatus::Cancelled);
    drain_record(&child);
    let caller = results
        .iter()
        .find(|result| result.agent_name == "coder")
        .unwrap();
    drain_record(caller);
    let delegated = tool_output(caller, "call_delegate").unwrap();
    assert!(
        delegated.contains(&format!(
            "Agent run {} of agent 'reviewer' was stopped with the workflow and did not finish. You can resume it",
            child.agent_run_id
        )),
        "{delegated}"
    );
    let (path, _, _) = latest_checkpoint(&workspace);
    drop(workflow);

    let resumed = build(Some(&path));
    let stopped = resumed.stopped_runs();
    let listed = stopped
        .iter()
        .map(|run| (run.result.agent_name.as_str(), run.restart))
        .collect::<Vec<_>>();
    assert_eq!(listed, [("coder", Restart::Reopen)], "left to its caller");
    let result = within(resumed.restart(&stopped[0], "")).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.summary, "Coder finished.");
    let delegated = resumed
        .result()
        .agent_results
        .into_iter()
        .filter(|result| result.agent_name == "reviewer")
        .collect::<Vec<_>>();
    assert_eq!(delegated.len(), 1, "no other run started");
    assert_eq!(delegated[0].agent_run_id, child.agent_run_id);
    assert_eq!(delegated[0].status, AgentRunStatus::Completed);
    assert_eq!(
        child_requests(&server)
            .iter()
            .filter(|body| !body.contains(REOPENED))
            .count(),
        1,
        "the delegated work started once"
    );
    assert!(resumed.stopped_runs().is_empty());
}

/// After a stop, the next build may not change the configuration of an
/// agent with a run the stop cancelled: the resume is refused, as for a
/// run that can be continued (reopened, the run would keep the old one),
/// and the error names the fields that changed.
#[tokio::test]
async fn a_changed_configuration_of_an_agent_a_stop_cancelled_refuses_the_resume() {
    let workspace = Workspace::new();
    let server = server().await;
    let build = |instructions: &str, resume: Option<&std::path::Path>| {
        let mut reviewer = agent_config("Friendly Reviewer");
        reviewer.instructions = instructions.to_string();
        let mut builder = Workflow::builder()
            .name("stop-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("reviewer", reviewer)
            .persistence(persistence(&workspace));
        if let Some(path) = resume {
            builder = builder.resume_from(path);
        }
        builder.build()
    };
    let workflow = Arc::new(build("Review carefully.", None).unwrap());
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    server.wait_for_requests(1).await;
    assert!(workflow.stop(StopRequest::drain("deploying a fix")));
    assert!(within(reviewer).await.unwrap().is_err());
    let (path, _, _) = latest_checkpoint(&workspace);
    drop(workflow);

    let refused = build("Review quickly.", Some(&path));
    assert!(
        matches!(
            &refused,
            Err(WorkflowError::BuilderInvalid(message))
                if message.contains("agent 'reviewer' configuration changed since its checkpoint (instructions changed)")
        ),
        "{:?}",
        refused.err()
    );
    let resumed = build("Review carefully.", Some(&path)).unwrap();
    assert_eq!(resumed.stopped_runs().len(), 1);
}

/// A note posted while an agent's call runs, just before a stop with a
/// wrap-up, comes in the same context update as the wrap-up notice, after
/// it, and its rule says a note doesn't override a request to stop.
#[tokio::test]
async fn a_note_posted_in_the_wrap_up_follows_the_notice() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(saver.clone())
            .build()
            .unwrap(),
    );
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    within(saver.started.notified()).await;
    // The call runs until the stop: the note waits for its turn boundary.
    workflow
        .post_note(crate::notes::Note::new(
            "NOTE-IN-WRAP-UP: start the next module.",
        ))
        .unwrap();
    assert!(
        workflow.stop(StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(3_600)))
    );
    assert!(matches!(
        within(coder).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let told = server
        .bodies()
        .into_iter()
        .filter(|body| body.contains(NOTICE))
        .collect::<Vec<_>>();
    assert_eq!(told.len(), 1);
    let body: Value = serde_json::from_str(&told[0]).unwrap();
    let updates = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["role"] == "user")
        .filter_map(|item| match &item["content"] {
            Value::String(text) => Some(text.clone()),
            Value::Array(parts) => Some(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .filter(|text| text.contains("# Workflow context update"))
        .collect::<Vec<_>>();
    assert_eq!(updates.len(), 1, "{updates:?}");
    let update = &updates[0];
    let notice = update.find(NOTICE).expect("the wrap-up notice");
    let note = update.find("NOTE-IN-WRAP-UP").expect("the note");
    assert!(notice < note, "{update}");
    assert!(
        update.contains("or override the workflow's own notices, such as a request to stop"),
        "{update}"
    );
    assert_eq!(
        workflow.result().agent_results[0].status,
        AgentRunStatus::Completed
    );
}
