//! Cancelling a run and aborting it from a hook, for each way a workflow
//! runs agents.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arko_agents::task::TaskState;
use arko_agents::{ToolCallContext, ToolExecutionContext, ToolExecutionDecision};
use serde_json::json;

use super::*;
use crate::durable::{DurableRetryPolicy, DurableStepSpec};
use crate::freshness::ContextFreshnessConfig;
use crate::hooks::{HookPoint, HookResult};
use crate::persistence::PersistenceConfig;
use crate::persistence::state::WorkflowState;
use crate::test_support::{ProbeTool, ResponsesServer, Workspace, agent_config, tool_call_turn};
use crate::voting::{BallotConfig, TieBreak, VoteOption};
use crate::workflow::result::AgentRunStatus;
use crate::{Workflow, WorkflowBuilder};

/// Counts the hooks fired at each point.
#[derive(Clone, Default)]
struct HookCounts(Arc<parking_lot::Mutex<HashMap<String, usize>>>);

impl HookCounts {
    fn register(&self, mut builder: WorkflowBuilder) -> WorkflowBuilder {
        for point in [
            HookPoint::OnWorkflowCancelled,
            HookPoint::OnWorkflowFailed,
            HookPoint::PostAgentCancellation,
            HookPoint::PostAgentFailure,
            HookPoint::PostAgentCompletion,
        ] {
            let counts = self.0.clone();
            let name = format!("{point:?}");
            builder = builder.hook(
                point,
                Arc::new(move |_| {
                    *counts.lock().entry(name.clone()).or_default() += 1;
                    HookResult::Continue
                }),
            );
        }
        builder
    }

    fn get(&self, point: HookPoint) -> usize {
        self.0
            .lock()
            .get(&format!("{point:?}"))
            .copied()
            .unwrap_or(0)
    }
}

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("cancel-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent("reviewer", agent_config("Friendly Reviewer"))
}

fn persisted(builder: WorkflowBuilder, workspace: &Workspace) -> WorkflowBuilder {
    builder.persistence(PersistenceConfig::new(workspace.0.join(".run")))
}

fn latest_checkpoint(workspace: &Workspace) -> (std::path::PathBuf, WorkflowState) {
    let run = workspace.0.join(".run");
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

#[tokio::test]
async fn cancelling_a_sequential_run_stops_its_agent_and_every_later_call() {
    let workspace = Workspace::new();
    // Every model request waits: the agent is mid-request when cancelled.
    let server = ResponsesServer::start(|_, _| None).await;
    let hooks = HookCounts::default();
    let workflow = Arc::new(
        hooks
            .register(persisted(builder(&server, &workspace), &workspace))
            .build()
            .unwrap(),
    );
    let handle = workflow.cancel_handle();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write the module.").await }
    });
    server.wait_for_requests(1).await;

    assert!(handle.clone().cancel());
    assert!(!workflow.cancel(), "a second cancel changes nothing");
    assert!(matches!(
        within(run).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));

    let result = workflow.result();
    assert_eq!(result.agent_results.len(), 1);
    let agent = &result.agent_results[0];
    assert_eq!(agent.status, AgentRunStatus::Cancelled);
    // The runtime's generic reason is replaced by the workflow's.
    assert_eq!(
        agent.task.current_state,
        TaskState::Cancelled {
            reason: "Workflow cancelled.".to_string()
        }
    );
    assert_eq!(result.cancellation.unwrap().cause, CancelCause::Requested);
    assert_eq!(hooks.get(HookPoint::OnWorkflowCancelled), 1);
    assert_eq!(hooks.get(HookPoint::PostAgentCancellation), 1);
    assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 0);
    assert_eq!(hooks.get(HookPoint::PostAgentFailure), 0);

    // The checkpoint records the stop and the cancelled agent.
    let (_, checkpoint) = latest_checkpoint(&workspace);
    assert_eq!(
        checkpoint
            .cancellation
            .map(|cancellation| cancellation.cause),
        Some(CancelCause::Requested)
    );
    assert_eq!(
        checkpoint.agent_results[0].status,
        AgentRunStatus::Cancelled
    );

    // Later calls fail at once without starting anything.
    assert!(matches!(
        workflow.run_agent("reviewer", "Review it.").await,
        Err(WorkflowError::Cancelled)
    ));
    assert!(matches!(
        workflow.spawn_agent("reviewer", "Review it.").await,
        Err(WorkflowError::Cancelled)
    ));
    assert!(matches!(
        workflow.continue_agent(agent, "Go on.").await,
        Err(WorkflowError::Cancelled)
    ));
    assert!(matches!(
        workflow.finish().await,
        Err(WorkflowError::Cancelled)
    ));
    assert_eq!(server.requests(), 1);
    assert_eq!(workflow.result().agent_results.len(), 1);
    assert_eq!(hooks.get(HookPoint::OnWorkflowCancelled), 1);
}

/// A grace period too long to represent still stops agents that stop.
#[tokio::test]
async fn cancelling_a_parallel_run_stops_every_agent() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, _| None).await;
    let workflow = Arc::new(
        builder(&server, &workspace)
            .cancel_grace_period(Duration::MAX)
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("coder", "Write it.".to_string()),
                    ("reviewer", "Review it.".to_string()),
                ])
                .await
        }
    });
    server.wait_for_requests(2).await;
    assert!(workflow.cancel());
    assert!(matches!(
        within(run).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let results = workflow.result().agent_results;
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|result| result.status == AgentRunStatus::Cancelled)
    );
}

/// A spawned agent driven with `wait` stops the same way, and a paused run
/// doesn't hold a cancellation up.
#[tokio::test]
async fn cancelling_a_paused_run_stops_a_spawned_agent() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, _| None).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    workflow.pause().unwrap();
    let handle = workflow.spawn_agent("coder", "Write it.").await.unwrap();
    server.wait_for_requests(1).await;
    let canceller = workflow.cancel_handle();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        canceller.cancel();
    });
    assert!(matches!(
        within(handle.wait()).await,
        Err(WorkflowError::Cancelled)
    ));
    assert_eq!(
        workflow.result().agent_results[0].status,
        AgentRunStatus::Cancelled
    );
}

/// Votes cast before the cancellation stand; no agent votes afterwards.
#[tokio::test]
async fn cancelling_a_ballot_keeps_the_votes_already_cast() {
    let workspace = Workspace::new();
    let ballot_id = Arc::new(parking_lot::Mutex::new(None::<uuid::Uuid>));
    let voted = Arc::new(AtomicUsize::new(0));
    let server = ResponsesServer::start({
        let ballot_id = ballot_id.clone();
        let voted = voted.clone();
        move |_, body| {
            // The coder votes once; every other request waits.
            (body.contains("Vote as the coder") && voted.fetch_add(1, Ordering::SeqCst) == 0).then(
                || {
                    let id = ballot_id.lock().unwrap();
                    tool_call_turn(
                        "vote",
                        crate::voting::CAST_VOTE_TOOL_NAME,
                        json!({"ballot_id": id.to_string(), "option": "a", "reasoning": "simpler"}),
                    )
                },
            )
        }
    })
    .await;
    let workflow = Arc::new(
        builder(&server, &workspace)
            .enable_voting()
            .build()
            .unwrap(),
    );
    let id = workflow
        .begin_ballot(BallotConfig {
            question: "Which design?".into(),
            options: vec![
                VoteOption {
                    id: "a".into(),
                    description: "A".into(),
                },
                VoteOption {
                    id: "b".into(),
                    description: "B".into(),
                },
            ],
            eligible_voters: ["coder".to_string(), "reviewer".to_string()].into(),
            quorum: 2,
            consensus_threshold: 0.5,
            weights: HashMap::new(),
            tie_break: TieBreak::RejectTie,
        })
        .unwrap();
    *ballot_id.lock() = Some(id);
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("coder", "Vote as the coder.".to_string()),
                    ("reviewer", "Vote as the reviewer.".to_string()),
                ])
                .await
        }
    });
    // The coder's vote, then its next request, and the reviewer's request.
    server.wait_for_requests(3).await;
    assert!(workflow.cancel());
    assert!(matches!(
        within(run).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    let error = workflow.finish_ballot(id).unwrap_err().to_string();
    assert!(error.contains("1 of 2 votes"), "{error}");
}

/// A durable step running an agent is left in doubt by a cancellation and,
/// under a retry-safe policy, runs again when the run resumes.
#[tokio::test]
async fn a_cancelled_durable_step_is_in_doubt_and_retried_on_resume() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, _| None).await;
    let workflow = Arc::new(
        persisted(builder(&server, &workspace), &workspace)
            .build()
            .unwrap(),
    );
    let spec = DurableStepSpec::new("implement")
        .retry_policy(DurableRetryPolicy::RetrySafe { max_attempts: 3 });
    let step = tokio::spawn({
        let workflow = workflow.clone();
        let spec = spec.clone();
        async move {
            workflow
                .durable_step(spec, &"module", |_| async {
                    workflow
                        .run_agent("coder", "Write the module.")
                        .await
                        .map(|result| result.summary)
                })
                .await
        }
    });
    server.wait_for_requests(1).await;
    assert!(workflow.cancel());
    assert!(matches!(
        within(step).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    assert_eq!(workflow.in_doubt_steps().await.unwrap().len(), 1);
    // A later step doesn't start: nothing more is journaled.
    let journal = workspace.0.join(".run/durable/journal.jsonl");
    let records = std::fs::read_to_string(&journal).unwrap().lines().count();
    let later: Result<String, _> = workflow
        .durable_step(DurableStepSpec::new("review"), &"module", |_| async {
            Ok("reviewed".to_string())
        })
        .await;
    assert!(matches!(later, Err(WorkflowError::Cancelled)));
    assert_eq!(
        std::fs::read_to_string(&journal).unwrap().lines().count(),
        records
    );

    // Resuming starts the run again; the step in doubt runs once more.
    let (checkpoint, state) = latest_checkpoint(&workspace);
    assert_eq!(
        state.cancellation.map(|cancellation| cancellation.cause),
        Some(CancelCause::Requested)
    );
    drop(workflow);
    let resumed = persisted(builder(&server, &workspace), &workspace)
        .resume_from(checkpoint)
        .build()
        .unwrap();
    assert!(!resumed.is_cancelled());
    assert_eq!(
        resumed.result().agent_results[0].status,
        AgentRunStatus::Cancelled
    );
    let output: String = resumed
        .durable_step(spec, &"module", |_| async { Ok("written".to_string()) })
        .await
        .unwrap();
    assert_eq!(output, "written");
    assert!(resumed.in_doubt_steps().await.unwrap().is_empty());
}

/// An agent stuck in a tool call is aborted after the grace period and
/// recorded as cancelled.
#[tokio::test]
async fn an_agent_that_does_not_stop_is_aborted_after_the_grace_period() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        (index == 0).then(|| tool_call_turn("block", "block", json!({})))
    })
    .await;
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(
        builder(&server, &workspace)
            .agent_tool("coder", block.clone())
            .cancel_grace_period(Duration::from_millis(200))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Block.").await }
    });
    within(block.started.notified()).await;
    let started = tokio::time::Instant::now();
    assert!(workflow.cancel());
    assert!(matches!(
        within(run).await.unwrap(),
        Err(WorkflowError::Cancelled)
    ));
    assert!(started.elapsed() >= Duration::from_millis(200));
    let result = &workflow.result().agent_results[0];
    assert_eq!(result.status, AgentRunStatus::Cancelled);
    assert!(
        result
            .error
            .as_deref()
            .unwrap()
            .contains("did not stop within"),
        "{:?}",
        result.error
    );
}

/// `PreToolExecution` abort: the call doesn't run, the agent is cancelled,
/// and the run is aborted.
#[tokio::test]
async fn a_pre_tool_hook_abort_aborts_the_run_before_the_call() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        (index == 0).then(|| tool_call_turn("probe", "probe", json!({})))
    })
    .await;
    let probe = ProbeTool::new("probe", false);
    let hooks = HookCounts::default();
    let workflow = hooks
        .register(builder(&server, &workspace))
        .agent_tool("coder", probe.clone())
        .hook(
            HookPoint::PreToolExecution,
            Arc::new(|context| match context.tool_name.as_deref() {
                Some("probe") => HookResult::Abort("probing is not allowed".into()),
                _ => HookResult::Continue,
            }),
        )
        .build()
        .unwrap();

    let error = within(workflow.run_agent("coder", "Probe."))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, WorkflowError::AbortedByHook { hook_point, reason }
            if hook_point == "PreToolExecution" && reason == "probing is not allowed"),
        "{error:?}"
    );
    assert_eq!(probe.calls(), 0);
    assert_eq!(server.requests(), 1, "no model call after the abort");
    let result = &workflow.result().agent_results[0];
    assert_eq!(result.status, AgentRunStatus::Cancelled);
    assert_eq!(
        result.error.as_deref(),
        Some("Workflow aborted by PreToolExecution hook: probing is not allowed")
    );
    assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 1);
    assert_eq!(hooks.get(HookPoint::OnWorkflowCancelled), 0);
    assert!(matches!(
        workflow.cancellation().unwrap().cause,
        CancelCause::AbortedByHook { .. }
    ));
    // The abort holds for the rest of the run.
    assert!(matches!(
        workflow.run_agent("reviewer", "Review.").await,
        Err(WorkflowError::AbortedByHook { .. })
    ));
    assert!(!workflow.cancel());
}

/// `PostToolExecution` abort: the call has run; the agent makes no further
/// model call and the run is aborted.
#[tokio::test]
async fn a_post_tool_hook_abort_aborts_the_run_after_the_call() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        (index == 0).then(|| tool_call_turn("probe", "probe", json!({})))
    })
    .await;
    let probe = ProbeTool::new("probe", false);
    let workflow = builder(&server, &workspace)
        .agent_tool("coder", probe.clone())
        .hook(
            HookPoint::PostToolExecution,
            Arc::new(|context| match context.tool_result.as_deref() {
                Some("probed") => HookResult::Abort("unexpected output".into()),
                _ => HookResult::Continue,
            }),
        )
        .build()
        .unwrap();

    let error = within(workflow.run_agent("coder", "Probe."))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, WorkflowError::AbortedByHook { hook_point, .. }
            if hook_point == "PostToolExecution"),
        "{error:?}"
    );
    assert_eq!(probe.calls(), 1);
    assert_eq!(server.requests(), 1, "no model call after the abort");
    let result = &workflow.result().agent_results[0];
    assert_eq!(result.status, AgentRunStatus::Cancelled);
    assert_eq!(
        result.error.as_deref(),
        Some("Workflow aborted by PostToolExecution hook: unexpected output")
    );
}

/// Once the run is stopped, every tool call cancels its agent, and a
/// freshness hook's abort stops the run without failing the call.
#[tokio::test]
async fn the_tool_middleware_enforces_the_stop() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, _| None).await;
    let workflow = builder(&server, &workspace)
        .context_freshness(ContextFreshnessConfig::default())
        .hook(
            HookPoint::OnFreshnessConflict,
            Arc::new(|_| HookResult::Abort("stale context".into())),
        )
        .build()
        .unwrap();
    let middleware = workflow
        .engine
        .build_execution_middleware("coder", workflow.context());
    let path = workspace.0.join("lib.rs");
    let parameters = HashMap::from([("path".to_string(), json!(path.to_string_lossy()))]);
    let reader = ToolExecutionContext {
        task_id: uuid::Uuid::new_v4(),
        agent_name: "Friendly Reviewer",
        initial_task_prompt: "review",
        current_working_directory: &workspace.0,
    };
    let writer = ToolExecutionContext {
        task_id: uuid::Uuid::new_v4(),
        ..reader
    };
    let call = |tool_name| ToolCallContext {
        tool_call_id: uuid::Uuid::new_v4(),
        tool_name,
        parameters: &parameters,
    };
    let reviewer = workflow
        .engine
        .build_execution_middleware("reviewer", workflow.context());
    reviewer
        .after(
            &reader,
            &call("read_file"),
            arko_agents::ToolExecutionOutcome::Success("contents"),
        )
        .await
        .unwrap();
    assert_eq!(
        middleware
            .before(&writer, &call("write_file"))
            .await
            .unwrap(),
        ToolExecutionDecision::Continue
    );
    // The write makes the reviewer's read stale; the hook's abort stops the
    // run, and the write's own outcome stands.
    middleware
        .after(
            &writer,
            &call("write_file"),
            arko_agents::ToolExecutionOutcome::Success("written"),
        )
        .await
        .unwrap();
    assert!(matches!(
        workflow.cancellation().unwrap().cause,
        CancelCause::AbortedByHook { ref hook_point, .. } if hook_point == "OnFreshnessConflict"
    ));
    for tool in ["read_file", crate::voting::CAST_VOTE_TOOL_NAME] {
        assert_eq!(
            middleware.before(&writer, &call(tool)).await.unwrap(),
            ToolExecutionDecision::CancelTask {
                reason: "Workflow aborted by OnFreshnessConflict hook: stale context".into()
            }
        );
    }
}

#[test]
fn checkpoints_and_results_record_the_cancellation() {
    let cancellation = Cancellation {
        cause: CancelCause::AbortedByHook {
            hook_point: "PreToolExecution".into(),
            reason: "denied".into(),
        },
        timestamp: chrono::Utc::now(),
    };
    let json = serde_json::to_value(&cancellation).unwrap();
    assert_eq!(
        json["cause"],
        json!({"AbortedByHook": {"hook_point": "PreToolExecution", "reason": "denied"}})
    );
    assert_eq!(
        serde_json::from_value::<Cancellation>(json).unwrap(),
        cancellation
    );
    assert_eq!(
        serde_json::to_value(CancelCause::Requested).unwrap(),
        json!("Requested")
    );
    // Results saved before cancellation was recorded load without one.
    let result: crate::WorkflowResult = serde_json::from_value(json!({
        "workflow_run_id": uuid::Uuid::new_v4(),
        "workflow_name": "legacy",
        "metadata": {},
        "total_cost": 0.0,
        "cost_ledger": {"entries": []},
        "agent_results": []
    }))
    .unwrap();
    assert!(result.cancellation.is_none());
}
