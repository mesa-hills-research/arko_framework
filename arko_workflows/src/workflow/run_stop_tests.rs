//! Stopping one agent run while the workflow runs on: its wrap-up, its
//! cancellation when the wrap-up is over, the host's intent (retire or
//! relaunch) across a resume, delegated runs, and the control folder.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use arko_agents::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::*;
use crate::hooks::{HookPoint, HookResult};
use crate::notes::control::{self, ControlStop};
use crate::persistence::state::{AgentRunRecord, WorkflowState};
use crate::persistence::{LIVE_CHECKPOINT_FILE, PersistenceConfig};
use crate::services::RUN_WORKFLOW_AGENT_TOOL_NAME;
use crate::test_support::{
    ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;
use crate::workflow::stop::{StopKind, StopRecord};

/// What a run is told when its own stop's wrap-up begins.
const RUN_NOTICE: &str = "Your run is being stopped";
/// What a run continued after its own stop's wrap-up is told.
const RESTARTED: &str = "it has now been restarted";
/// What a reopened run is told.
const REOPENED: &str = "has now been reopened";
/// A completion verifier's request.
const VERIFIER: &str = "You are a completion verifier";

/// A tool that works until its task's stop signal fires, then saves and
/// returns what it was told.
#[derive(Debug, Default)]
struct Saver {
    started: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl Tool for Saver {
    fn definition(&self) -> ToolDefinition {
        definition("saver")
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

/// A tool that waits until the test opens it, or its task's stop signal
/// fires (which it records).
#[derive(Debug, Default)]
struct Gate {
    started: tokio::sync::Notify,
    open: tokio::sync::Notify,
    stopped: AtomicBool,
}

#[async_trait::async_trait]
impl Tool for Gate {
    fn definition(&self) -> ToolDefinition {
        definition("gate")
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.started.notify_one();
        tokio::select! {
            _ = self.open.notified() => Ok("released".to_string()),
            _ = context.stop.stopping() => {
                self.stopped.store(true, Ordering::SeqCst);
                Ok("stopped".to_string())
            }
        }
    }
}

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: "Works until told.".to_string(),
        output_based_example: None,
        parameters: vec![],
        parallel_safe: false,
        tool_type_details: ToolTypeDetails::DeveloperDefined,
        raw_json_schema_override: None,
    }
}

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("run-stop-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent("reviewer", agent_config("Friendly Reviewer"))
        .agent("other", agent_config("Other Agent"))
}

fn run_dir(workspace: &Workspace) -> std::path::PathBuf {
    workspace.0.join(".run")
}

fn persistence(workspace: &Workspace) -> PersistenceConfig {
    PersistenceConfig::new(run_dir(workspace))
}

/// The checkpoint `latest-checkpoint` points to, and its state.
fn latest_checkpoint(workspace: &Workspace) -> (std::path::PathBuf, WorkflowState) {
    let run = run_dir(workspace);
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    let path = run.join(name.trim());
    let state = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    (path, state)
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("the workflow call returned")
}

/// Waits, up to a generous limit, until `done` holds.
async fn until(mut done: impl FnMut() -> bool) {
    within(async {
        while !done() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
}

/// The running run of `agent`.
fn running(workflow: &Workflow, agent: &str) -> uuid::Uuid {
    let runs = workflow.context().registry.runs_of(agent);
    assert_eq!(runs.len(), 1, "one run of {agent} runs");
    runs[0]
}

/// A server for the coder (`CODER-TASK`: it calls `saver`, and completes
/// once told its run is stopping), a reviewer that never gets an answer
/// (`REVIEWER-TASK`), another agent that waits at its gate (`OTHER-TASK`),
/// a run that completes at once (`EARLY-TASK`), runs restarted or
/// reopened, which complete, and a completion verifier that rejects a
/// progress report.
async fn server() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        if body.contains(VERIFIER) {
            return Some(complete_turn(
                "verdict",
                if body.contains("Stopped at step 3.") {
                    "REJECTED_MAJOR\nSteps 4 and 5 are not done."
                } else {
                    "ACCEPTED"
                },
            ));
        }
        if body.contains(RESTARTED) {
            return Some(complete_turn("again", "Finished."));
        }
        if body.contains(REOPENED) {
            return Some(complete_turn("again", "Finished again."));
        }
        if body.contains("EARLY-TASK") {
            return Some(complete_turn("early", "Done early."));
        }
        if body.contains("REVIEWER-TASK") {
            return None;
        }
        if body.contains("OTHER-TASK") {
            return Some(if body.contains("released") {
                complete_turn("other", "Other done.")
            } else {
                tool_call_turn("gate", "gate", json!({}))
            });
        }
        Some(if body.contains(RUN_NOTICE) {
            complete_turn("done", "Stopped at step 3.")
        } else {
            tool_call_turn("save", "saver", json!({}))
        })
    })
    .await
}

fn checks_coder(builder: WorkflowBuilder) -> WorkflowBuilder {
    builder.completion_verifier(
        crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
            .target("coder"),
    )
}

/// The output of call `call_id` in `result`'s conversation.
fn tool_output(result: &AgentResult, call_id: &str) -> Option<String> {
    result
        .task
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(arko_ai::Content::as_tool_result)
        .find(|result| result.call_id.as_deref() == Some(call_id))
        .map(|result| result.output.clone())
}

/// Counts the hooks fired at `point`.
fn count(builder: WorkflowBuilder, point: HookPoint) -> (WorkflowBuilder, Arc<AtomicUsizeCounter>) {
    let counter = Arc::new(AtomicUsizeCounter::default());
    let fired = counter.clone();
    let builder = builder.hook(
        point,
        Arc::new(move |_| {
            fired.0.fetch_add(1, Ordering::SeqCst);
            HookResult::Continue
        }),
    );
    (builder, counter)
}

#[derive(Default)]
struct AtomicUsizeCounter(std::sync::atomic::AtomicUsize);

impl AtomicUsizeCounter {
    fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// A stop of one run with a wrap-up: its tools see their stop signal, it
/// is told to finish, and its completion is a progress report that its
/// completion verifier doesn't review; the run call returns it, marked.
/// Meanwhile new runs start, and the other running agent's tools see no
/// stop.
#[tokio::test]
async fn a_run_stopped_alone_wraps_up_while_the_others_run_on() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let gate = Arc::new(Gate::default());
    let workflow = Arc::new(
        checks_coder(builder(&server, &workspace))
            .global_tool(saver.clone())
            .global_tool(gate.clone())
            .build()
            .unwrap(),
    );
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    let other = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("other", "OTHER-TASK").await }
    });
    within(saver.started.notified()).await;
    within(gate.started.notified()).await;

    let request =
        StopRequest::operator("moving it to another model").wrap_up(Duration::from_secs(600));
    let stopped = workflow
        .stop_agent("coder", request.clone(), RunStopIntent::Retire)
        .unwrap();
    assert_eq!(stopped.len(), 1);
    assert!(
        !workflow.stop_run(stopped[0], request, RunStopIntent::Relaunch),
        "the first stop stands"
    );
    assert!(!workflow.is_stopping());
    assert!(!workflow.is_cancelled());
    // New runs start as usual, also of the agent being stopped.
    let early = within(workflow.run_agent("coder", "EARLY-TASK"))
        .await
        .unwrap();
    assert_eq!(early.summary, "Done early.");
    assert_eq!(early.stopped_by(), None);

    let coded = within(coder).await.unwrap().unwrap();
    assert_eq!(coded.agent_run_id, stopped[0]);
    assert_eq!(coded.status, AgentRunStatus::Completed);
    assert_eq!(coded.summary, "Stopped at step 3.");
    let record = coded.stopped_by().expect("its stop ended it");
    assert_eq!(
        (record.kind, record.reason.as_str(), record.intent),
        (
            StopKind::Operator,
            "moving it to another model",
            Some(RunStopIntent::Retire)
        )
    );
    assert!(record.is_retired());
    assert_eq!(
        tool_output(&coded, "call_save").unwrap(),
        "saved (Run stopped (operator): moving it to another model.)"
    );
    let bodies = server.bodies();
    assert!(bodies.iter().any(|body| body.contains(
        "Your run is being stopped: moving it to another model. You have about 600 seconds before you are stopped."
    )));
    assert!(
        !bodies
            .iter()
            .any(|body| body.contains(VERIFIER) && body.contains("Stopped at step 3.")),
        "the progress report isn't reviewed"
    );

    // The other agent ran on, untouched.
    assert!(!gate.stopped.load(Ordering::SeqCst));
    assert_eq!(workflow.agent_status("other"), Some(AgentStatus::Running));
    gate.open.notify_one();
    let other = within(other).await.unwrap().unwrap();
    assert_eq!(other.summary, "Other done.");
    assert_eq!(other.stopped_by(), None);

    assert!(matches!(
        workflow.stop_agent("ghost", StopRequest::operator("x"), RunStopIntent::Retire),
        Err(WorkflowError::AgentNotFound(_))
    ));
    assert_eq!(
        workflow
            .stop_agent("coder", StopRequest::operator("x"), RunStopIntent::Retire)
            .unwrap(),
        Vec::<uuid::Uuid>::new(),
        "none runs"
    );
    assert!(!workflow.stop_run(
        stopped[0],
        StopRequest::operator("x"),
        RunStopIntent::Retire
    ));
}

/// A run that doesn't finish within its wrap-up is cancelled (within the
/// grace period): its call returns `AgentStopped`, which isn't a failure,
/// and the run, stopped to be relaunched, is listed by `stopped_runs` at
/// once and restarted while the workflow runs.
#[tokio::test]
async fn a_run_that_does_not_wrap_up_in_time_is_cancelled_then_relaunched() {
    let workspace = Workspace::new();
    let server = server().await;
    let (builder, failed) = count(builder(&server, &workspace), HookPoint::OnWorkflowFailed);
    let (builder, cancelled) = count(builder, HookPoint::PostAgentCancellation);
    let workflow = Arc::new(builder.build().unwrap());
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    server.wait_for_requests(1).await;
    let id = running(&workflow, "reviewer");
    let started = Instant::now();
    assert!(workflow.stop_run(
        id,
        StopRequest::limit("out of time").wrap_up(Duration::from_millis(300)),
        RunStopIntent::Relaunch
    ));
    let error = within(reviewer).await.unwrap().unwrap_err();
    assert!(started.elapsed() >= Duration::from_millis(300));
    let WorkflowError::AgentStopped {
        agent,
        task_id,
        reason,
    } = error
    else {
        panic!("{error}");
    };
    assert_eq!(
        (agent.as_str(), task_id, reason.as_str()),
        ("reviewer", id, "Run stopped (limit): out of time.")
    );
    assert!(!workflow.is_cancelled());
    let result = workflow.agent_result(id).unwrap();
    assert_eq!(result.status, AgentRunStatus::Cancelled);
    assert_eq!(
        result.error.as_deref(),
        Some("Run stopped (limit): out of time.")
    );
    let record = result.stopped_by().unwrap();
    assert_eq!(
        (record.kind, record.intent),
        (StopKind::Limit, Some(RunStopIntent::Relaunch))
    );
    assert_eq!(failed.get(), 0, "not a failure");
    assert_eq!(cancelled.get(), 1);

    let stopped = workflow.stopped_runs();
    let listed = stopped
        .iter()
        .map(|run| (run.result.agent_run_id, run.restart))
        .collect::<Vec<_>>();
    assert_eq!(listed, [(id, Restart::Reopen)]);
    let again = within(workflow.restart(&stopped[0], "Carry on."))
        .await
        .unwrap();
    assert_eq!(again.agent_run_id, id);
    assert_eq!(again.summary, "Finished again.");
    assert_eq!(again.stopped_by(), None);
    assert!(workflow.stopped_runs().is_empty());
}

/// Without a wrap-up a run is cancelled at once; in a parallel batch it
/// comes back in its place, and doesn't stop the others under fail-fast.
#[tokio::test]
async fn a_run_stopped_without_a_wrap_up_is_cancelled_at_once_and_its_batch_runs_on() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = Arc::new(Gate::default());
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(gate.clone())
            .fail_fast(true)
            .build()
            .unwrap(),
    );
    let batch = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("reviewer", "REVIEWER-TASK".to_string()),
                    ("other", "OTHER-TASK".to_string()),
                ])
                .await
        }
    });
    within(gate.started.notified()).await;
    server.wait_for_requests(2).await;
    let stopped = workflow
        .stop_agent(
            "reviewer",
            StopRequest::operator("not needed"),
            RunStopIntent::Retire,
        )
        .unwrap();
    assert_eq!(stopped.len(), 1);
    until(|| workflow.agent_result(stopped[0]).is_some()).await;
    assert!(!gate.stopped.load(Ordering::SeqCst));
    gate.open.notify_one();
    let results = within(batch).await.unwrap().unwrap();
    assert_eq!(results[0].agent_run_id, stopped[0]);
    assert_eq!(results[0].status, AgentRunStatus::Cancelled);
    assert_eq!(
        results[0].error.as_deref(),
        Some("Run stopped (operator): not needed.")
    );
    assert!(results[0].stopped_by().unwrap().is_retired());
    assert_eq!(results[1].summary, "Other done.");
    assert!(workflow.stopped_runs().is_empty(), "a retired run");
}

/// The host's intent is saved with the results: after a resume, a run
/// stopped to be relaunched is listed and continued, and a retired one is
/// not listed and `restart` refuses it, also when a workflow stop ended it
/// in its wrap-up (its own stop stands).
#[tokio::test]
async fn retire_and_relaunch_are_kept_across_a_resume() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let build = |resume: Option<&std::path::Path>| {
        let mut builder = builder(&server, &workspace)
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
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    within(saver.started.notified()).await;
    server.wait_for_requests(2).await;
    let (coder_id, reviewer_id) = (running(&workflow, "coder"), running(&workflow, "reviewer"));
    assert!(workflow.stop_run(
        coder_id,
        StopRequest::operator("pausing it").wrap_up(Duration::from_secs(600)),
        RunStopIntent::Relaunch
    ));
    let coded = within(coder).await.unwrap().unwrap();
    assert_eq!(
        coded.stopped_by().unwrap().intent,
        Some(RunStopIntent::Relaunch)
    );
    assert!(workflow.stop_run(
        reviewer_id,
        StopRequest::operator("replacing it").wrap_up(Duration::from_secs(3_600)),
        RunStopIntent::Retire
    ));
    // A drain ends the run's wrap-up: its own stop stands on its result.
    assert!(workflow.stop(StopRequest::drain("deploying a fix")));
    assert!(matches!(
        within(reviewer).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let reviewed = workflow.agent_result(reviewer_id).unwrap();
    assert_eq!(reviewed.status, AgentRunStatus::Cancelled);
    let record = reviewed.stopped_by().unwrap();
    assert_eq!(
        (record.reason.as_str(), record.intent),
        ("replacing it", Some(RunStopIntent::Retire))
    );
    let (path, state) = latest_checkpoint(&workspace);
    let saved = |id: uuid::Uuid| {
        state
            .agent_runs
            .iter()
            .find(|record| record.agent_run_id == id)
            .and_then(|record| record.stopped_by.clone())
            .and_then(|stop| stop.intent)
    };
    assert_eq!(saved(coder_id), Some(RunStopIntent::Relaunch));
    assert_eq!(saved(reviewer_id), Some(RunStopIntent::Retire));
    drop(workflow);

    let resumed = build(Some(&path));
    assert!(resumed.interrupted_runs().is_empty());
    let stopped = resumed.stopped_runs();
    let listed = stopped
        .iter()
        .map(|run| (run.result.agent_run_id, run.restart))
        .collect::<Vec<_>>();
    assert_eq!(listed, [(coder_id, Restart::Continue)]);
    let retired = resumed.agent_result(reviewer_id).unwrap();
    assert!(retired.stopped_by().unwrap().is_retired());
    let refused = resumed
        .restart(
            &StoppedRun {
                result: retired,
                restart: Restart::Reopen,
            },
            "",
        )
        .await;
    assert!(
        matches!(&refused, Err(WorkflowError::BuilderInvalid(message)) if message.contains("retired")),
        "{refused:?}"
    );
    let continued = within(resumed.restart(&stopped[0], "Carry on."))
        .await
        .unwrap();
    assert_eq!(continued.summary, "Finished.");
    assert_eq!(continued.stopped_by(), None);
    assert!(server.bodies().iter().any(|body| body.contains(
        "Your run was stopped (pausing it) while you worked, so you finished early, and it has now been restarted."
    )));
    assert!(resumed.stopped_runs().is_empty());
}

/// A stop under way is saved with the live checkpoint: when the process
/// ends in a run's wrap-up, the resume keeps the host's intent, so a
/// retired run is not listed for recovery, and one to relaunch is.
#[tokio::test]
async fn a_run_the_process_ended_in_its_wrap_up_keeps_its_intent() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        if body.contains(RUN_NOTICE) || body.contains("REVIEWER-TASK") {
            return None;
        }
        Some(tool_call_turn("save", "saver", json!({})))
    })
    .await;
    let saver = Arc::new(Saver::default());
    let build = |resume: Option<&std::path::Path>| {
        let mut builder = builder(&server, &workspace)
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
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    within(saver.started.notified()).await;
    server.wait_for_requests(2).await;
    let (coder_id, reviewer_id) = (running(&workflow, "coder"), running(&workflow, "reviewer"));
    let wrap_up = StopRequest::operator("replacing it").wrap_up(Duration::from_secs(3_600));
    assert!(workflow.stop_run(coder_id, wrap_up.clone(), RunStopIntent::Retire));
    assert!(workflow.stop_run(reviewer_id, wrap_up, RunStopIntent::Relaunch));
    // The coder is told, and waits on its model; the stops are saved.
    until(|| server.bodies().iter().any(|body| body.contains(RUN_NOTICE))).await;
    let run = run_dir(&workspace);
    until(|| {
        std::fs::read_to_string(run.join(LIVE_CHECKPOINT_FILE))
            .ok()
            .and_then(|json| serde_json::from_str::<WorkflowState>(&json).ok())
            .is_some_and(|state| {
                state.agent_runs.len() == 2
                    && state
                        .agent_runs
                        .iter()
                        .all(|record| record.stopped_by.is_some())
            })
    })
    .await;
    let path = run.join(LIVE_CHECKPOINT_FILE);
    workflow.engine.persistence().unwrap().simulate_crash();
    for task in [coder, reviewer] {
        task.abort();
        let _ = task.await;
    }
    drop(workflow);

    let resumed = build(Some(&path));
    let coded = resumed.agent_result(coder_id).unwrap();
    assert_eq!(coded.status, AgentRunStatus::Interrupted);
    assert!(coded.stopped_by().unwrap().is_retired());
    let interrupted = resumed
        .interrupted_runs()
        .into_iter()
        .map(|run| run.agent_run_id)
        .collect::<Vec<_>>();
    assert_eq!(interrupted, [reviewer_id]);
    let listed = resumed
        .stopped_runs()
        .into_iter()
        .map(|run| (run.result.agent_run_id, run.restart))
        .collect::<Vec<_>>();
    assert_eq!(listed, [(reviewer_id, Restart::Recover)]);
}

/// A server for a lead that delegates `CODER-TASK` to the coder and, given
/// a note that it can resume the coder's run, resumes it; a lead told to
/// stop finishes.
async fn delegation_server() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        if body.contains("LEAD-TASK") {
            if body.contains(RUN_NOTICE) {
                return Some(complete_turn("lead-done", "Lead stopped."));
            }
            if body.contains("Coder finished.") || body.contains("can't be resumed") {
                return Some(complete_turn("lead-again", "Lead finished."));
            }
            if let Some(end) = body.find(" of agent 'coder' was stopped by the workflow's host") {
                let run_id = &body[end - 36..end];
                return Some(tool_call_turn(
                    "resume",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "coder", "resume_run": run_id, "prompt": "Carry on."}),
                ));
            }
            return Some(tool_call_turn(
                "delegate",
                RUN_WORKFLOW_AGENT_TOOL_NAME,
                json!({"agent": "coder", "prompt": "CODER-TASK"}),
            ));
        }
        Some(if body.contains("Carry on.") {
            complete_turn("again", "Coder finished.")
        } else if body.contains(RUN_NOTICE) {
            complete_turn("done", "Stopped at step 3.")
        } else {
            tool_call_turn("save", "saver", json!({}))
        })
    })
    .await
}

fn delegating(server: &ResponsesServer, workspace: &Workspace, saver: Arc<Saver>) -> Workflow {
    builder(server, workspace)
        .agent("lead", agent_config("Lead"))
        .workflow_services(
            crate::services::WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("coder"),
        )
        .global_tool(saver)
        .build()
        .unwrap()
}

/// A delegated run stopped alone to be relaunched returns its progress
/// report to its caller with a note naming it, and the caller, which runs
/// on, can resume it.
#[tokio::test]
async fn a_delegated_run_stopped_alone_is_named_to_its_caller_which_can_resume_it() {
    let workspace = Workspace::new();
    let server = delegation_server().await;
    let saver = Arc::new(Saver::default());
    let workflow = Arc::new(delegating(&server, &workspace, saver.clone()));
    let lead = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "LEAD-TASK").await }
    });
    within(saver.started.notified()).await;
    let coder_id = running(&workflow, "coder");
    assert!(workflow.stop_run(
        coder_id,
        StopRequest::operator("pausing the coder").wrap_up(Duration::from_secs(600)),
        RunStopIntent::Relaunch
    ));
    let lead = within(lead).await.unwrap().unwrap();
    assert_eq!(lead.summary, "Lead finished.");
    assert_eq!(lead.stopped_by(), None, "the caller wasn't stopped");
    assert_eq!(
        tool_output(&lead, "call_delegate").unwrap(),
        format!(
            "Stopped at step 3.\n\nAgent run {coder_id} of agent 'coder' was stopped by the workflow's host (pausing the coder) before it finished: its summary is a progress report that no completion verifier reviewed. You can resume it instead of starting over: call run_workflow_agent with agent \"coder\", resume_run \"{coder_id}\" and a prompt telling it what happened and what to do next; it continues its own task."
        )
    );
    assert_eq!(
        tool_output(&lead, "call_resume").unwrap(),
        "Coder finished."
    );
    let coder = workflow.agent_result(coder_id).unwrap();
    assert_eq!(coder.summary, "Coder finished.");
    assert_eq!(coder.stopped_by(), None);
    assert!(!workflow.is_cancelled());
}

/// A delegated run retired without a wrap-up is cancelled at once: its
/// caller's call fails, naming the run as retired, and resuming it is
/// refused.
#[tokio::test]
async fn a_retired_delegated_run_cant_be_resumed_by_its_caller() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        if body.contains("LEAD-TASK") {
            if body.contains("can't be resumed: the workflow's host retired it") {
                return Some(complete_turn("lead-again", "Lead finished."));
            }
            if let Some(end) = body.find(" of agent 'coder' was stopped by the workflow's host") {
                let run_id = &body[end - 36..end];
                return Some(tool_call_turn(
                    "resume",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "coder", "resume_run": run_id, "prompt": "Carry on."}),
                ));
            }
            return Some(tool_call_turn(
                "delegate",
                RUN_WORKFLOW_AGENT_TOOL_NAME,
                json!({"agent": "coder", "prompt": "CODER-TASK"}),
            ));
        }
        Some(tool_call_turn("save", "saver", json!({})))
    })
    .await;
    let saver = Arc::new(Saver::default());
    let workflow = Arc::new(delegating(&server, &workspace, saver.clone()));
    let lead = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "LEAD-TASK").await }
    });
    within(saver.started.notified()).await;
    let coder_id = running(&workflow, "coder");
    assert!(workflow.stop_run(
        coder_id,
        StopRequest::operator("replaced"),
        RunStopIntent::Retire
    ));
    let lead = within(lead).await.unwrap().unwrap();
    assert_eq!(lead.summary, "Lead finished.");
    let delegated = tool_output(&lead, "call_delegate").unwrap();
    assert!(
        delegated.ends_with(&format!(
            "Agent run {coder_id} of agent 'coder' was stopped by the workflow's host (replaced) and did not finish. The host retired it, so it can't be resumed."
        )),
        "{delegated}"
    );
    let resumed = tool_output(&lead, "call_resume").unwrap();
    assert!(
        resumed.contains("can't be resumed: the workflow's host retired it"),
        "{resumed}"
    );
    let coder = workflow.agent_result(coder_id).unwrap();
    assert_eq!(coder.status, AgentRunStatus::Cancelled);
    assert!(coder.stopped_by().unwrap().is_retired());
}

/// Stopping a run stops the runs inside its tool calls too: the delegated
/// run wraps up and reports to its caller, which then wraps up itself; both
/// are marked with the stop.
#[tokio::test]
async fn stopping_a_run_stops_the_runs_it_delegated_to() {
    let workspace = Workspace::new();
    let server = delegation_server().await;
    let saver = Arc::new(Saver::default());
    let workflow = Arc::new(delegating(&server, &workspace, saver.clone()));
    let lead = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "LEAD-TASK").await }
    });
    within(saver.started.notified()).await;
    let (lead_id, coder_id) = (running(&workflow, "lead"), running(&workflow, "coder"));
    assert_eq!(
        workflow
            .stop_agent(
                "lead",
                StopRequest::operator("retiring the lead").wrap_up(Duration::from_secs(600)),
                RunStopIntent::Retire,
            )
            .unwrap(),
        [lead_id]
    );
    let lead = within(lead).await.unwrap().unwrap();
    assert_eq!(lead.summary, "Lead stopped.");
    assert!(lead.stopped_by().unwrap().is_retired());
    let coder = workflow.agent_result(coder_id).unwrap();
    assert_eq!(coder.summary, "Stopped at step 3.");
    assert!(coder.stopped_by().unwrap().is_retired());
    assert!(tool_output(&lead, "call_delegate").unwrap().ends_with(
        "before it finished: its summary is a progress report that no completion verifier reviewed. The host retired it, so it can't be resumed."
    ));
    assert!(workflow.stopped_runs().is_empty());
}

/// Another process stops a run through the control folder: the running
/// workflow reads the request within a second. A request written before
/// the workflow was built is skipped.
#[tokio::test]
async fn a_control_folder_request_stops_a_running_run() {
    let workspace = Workspace::new();
    let server = server().await;
    let saver = Arc::new(Saver::default());
    let run = run_dir(&workspace);
    std::fs::create_dir_all(control::folder(&run)).unwrap();
    control::request_stop(&run, &ControlStop::agent("reviewer", "stale")).unwrap();
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(saver.clone())
            .persistence(persistence(&workspace))
            .control_folder()
            .build()
            .unwrap(),
    );
    let coder = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "CODER-TASK").await }
    });
    let reviewer = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("reviewer", "REVIEWER-TASK").await }
    });
    within(saver.started.notified()).await;
    server.wait_for_requests(2).await;
    let written = control::request_stop(
        &run,
        &ControlStop::agent("coder", "from runctl").wrap_up(Duration::from_secs(600)),
    )
    .unwrap();
    assert!(written.id.is_some());
    let coded = within(coder).await.unwrap().unwrap();
    let record = coded.stopped_by().unwrap();
    assert_eq!(
        (record.kind, record.reason.as_str(), record.intent),
        (
            StopKind::Operator,
            "from runctl",
            Some(RunStopIntent::Retire)
        )
    );
    assert_eq!(
        workflow.agent_status("reviewer"),
        Some(AgentStatus::Running),
        "the stale request was skipped"
    );
    reviewer.abort();
    let requests = control::read_stops(&run).unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1], written);

    // Requests name a run or an agent; the folder must exist.
    let both = ControlStop {
        agent: Some("coder".to_string()),
        ..ControlStop::run(uuid::Uuid::new_v4(), "x")
    };
    let invalid = control::request_stop(&run, &both).unwrap_err();
    assert_eq!(invalid.kind(), std::io::ErrorKind::InvalidInput);
    let missing = control::request_stop(
        &workspace.0.join("nowhere"),
        &ControlStop::agent("coder", "x"),
    )
    .unwrap_err();
    assert_eq!(missing.kind(), std::io::ErrorKind::NotFound);
}

/// Records saved before runs could be stopped alone load without an
/// intent, and a workflow stop is still saved without one; a hand-written
/// stop request needs only its target.
#[test]
fn records_without_an_intent_load_and_requests_default() {
    let record: StopRecord = serde_json::from_value(json!({
        "kind": "drain",
        "reason": "deploying a fix",
        "timestamp": "2026-10-01T12:00:00Z"
    }))
    .unwrap();
    assert_eq!(record.intent, None);
    assert!(!record.is_retired());
    assert!(
        serde_json::to_value(&record)
            .unwrap()
            .get("intent")
            .is_none()
    );
    let run: AgentRunRecord = serde_json::from_value(json!({
        "agent_run_id": uuid::Uuid::new_v4(),
        "agent_name": "coder",
        "status": "Cancelled",
        "summary": "",
        "task_file": "agents/x/y.json",
        "stopped_by": {"kind": "operator", "reason": "Ctrl-C", "timestamp": "2026-10-01T12:00:00Z"}
    }))
    .unwrap();
    assert_eq!(run.stopped_by.unwrap().intent, None);
    let retired = StopRecord {
        intent: Some(RunStopIntent::Retire),
        ..record
    };
    assert_eq!(
        serde_json::to_value(&retired).unwrap()["intent"],
        json!("retire")
    );

    let stop: ControlStop = serde_json::from_value(json!({"agent": "worker-12"})).unwrap();
    assert_eq!(stop, ControlStop::agent("worker-12", ""));
    assert_eq!(stop.kind, StopKind::Operator);
    assert_eq!(stop.intent, RunStopIntent::Retire);
    assert_eq!(stop.request().wrap_up, None);
    let stop: ControlStop = serde_json::from_value(json!({
        "run": uuid::Uuid::nil(),
        "kind": "limit",
        "reason": "quota",
        "wrap_up_secs": 90,
        "intent": "relaunch"
    }))
    .unwrap();
    let request = stop.request();
    assert_eq!(
        (request.kind, request.reason.as_str()),
        (StopKind::Limit, "quota")
    );
    assert_eq!(
        request.wrap_up.map(|wrap_up| wrap_up.within),
        Some(Duration::from_secs(90))
    );
    assert_eq!(stop.intent, RunStopIntent::Relaunch);
}
