//! Budgets: agents held over the budget (`BudgetExceededPolicy::Pause`, the
//! default) and released by a raise, live or on resume; runs cancelled by
//! the budget (`Cancel`) reopened; limits saved with checkpoints.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arko_agents::tool::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::*;
use crate::budget::BudgetExceededPolicy;
use crate::context::SharedContext;
use crate::hooks::{HookPoint, HookResult};
use crate::messaging::registry::{AgentStatus, WaitReason};
use crate::persistence::PersistenceConfig;
use crate::persistence::state::WorkflowState;
use crate::policy::{AgentPause, AgentResume};
use crate::services::{RUN_WORKFLOW_AGENT_TOOL_NAME, WorkflowServicesConfig};
use crate::test_support::{
    ProbeTool, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::engine::REOPENED_NOTE;
use crate::workflow::result::AgentRunStatus;

/// A model call of the test server costs this much; a budget of
/// `LOW` is exceeded by the first call.
const LOW: f64 = 1e-6;

/// What approving an ordinary error-recovery pause tells the model.
const ERROR_NOTE: &str = "resulted in an error";

/// Answers "Delegate." with a `run_workflow_agent` call running `reviewer`
/// ("Review."), and completes once it has a result; any other task calls
/// `gate`, then completes.
async fn server() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        let answered = body.contains("function_call_output");
        Some(if body.contains("Delegate.") {
            if answered {
                complete_turn("coder-done", "coder done")
            } else {
                tool_call_turn(
                    "delegate",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "reviewer", "prompt": "Review."}),
                )
            }
        } else if answered {
            complete_turn("done", "done")
        } else {
            tool_call_turn("gate", "gate", json!({}))
        })
    })
    .await
}

/// A tool that returns once its agent's first model call is charged (so
/// the test knows the budget is exceeded before the agent's next request),
/// counting its calls.
struct GateTool {
    context: Arc<OnceLock<Arc<SharedContext>>>,
    calls: AtomicUsize,
}

impl GateTool {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            context: Arc::default(),
            calls: AtomicUsize::new(0),
        })
    }

    fn attach(&self, workflow: &Workflow) {
        let _ = self.context.set(workflow.context().clone());
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for GateTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GateTool")
    }
}

#[async_trait::async_trait]
impl Tool for GateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "gate".to_string(),
            description: "Test gate.".to_string(),
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        let workflow = self.context.get().expect("attached").clone();
        let agent = workflow
            .agent_configs
            .iter()
            .find(|(_, config)| config.name == context.agent_name)
            .map(|(name, _)| name.clone())
            .unwrap_or_default();
        let charged = || workflow.budget.agent_spent(&agent) > 0.0;
        tokio::time::timeout(Duration::from_secs(10), async {
            while !charged() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the call was charged");
        Ok("gate passed".to_string())
    }
}

fn builder(
    server: &ResponsesServer,
    workspace: &Workspace,
    gate: &Arc<GateTool>,
) -> WorkflowBuilder {
    Workflow::builder()
        .name("budget-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent("reviewer", agent_config("Friendly Reviewer"))
        .global_tool(gate.clone())
}

/// `builder`, with `coder` allowed to run `reviewer`.
fn delegating(
    server: &ResponsesServer,
    workspace: &Workspace,
    gate: &Arc<GateTool>,
) -> WorkflowBuilder {
    builder(server, workspace, gate).workflow_services(
        WorkflowServicesConfig::default()
            .allow_caller("coder")
            .allow_target("reviewer"),
    )
}

fn run_dir(workspace: &Workspace) -> std::path::PathBuf {
    workspace.0.join(".run")
}

fn persisted(builder: WorkflowBuilder, workspace: &Workspace) -> WorkflowBuilder {
    builder.persistence(PersistenceConfig::new(run_dir(workspace)))
}

fn build(builder: WorkflowBuilder, gate: &Arc<GateTool>) -> Workflow {
    let workflow = builder.build().unwrap();
    gate.attach(&workflow);
    workflow
}

/// The checkpoint `latest-checkpoint` points to.
fn latest(workspace: &Workspace) -> std::path::PathBuf {
    let run = run_dir(workspace);
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    run.join(name.trim())
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(20), future)
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

/// Counts `OnBudgetExceeded`.
fn count_exceeded(builder: WorkflowBuilder, count: &Arc<AtomicUsize>) -> WorkflowBuilder {
    let count = count.clone();
    builder.hook(
        HookPoint::OnBudgetExceeded,
        Arc::new(move |_| {
            count.fetch_add(1, Ordering::SeqCst);
            HookResult::Continue
        }),
    )
}

fn assert_held(error: &WorkflowError) -> uuid::Uuid {
    match error {
        WorkflowError::AgentPaused {
            task_id,
            pause:
                AgentPause::ErrorRecovery {
                    cause: arko_agents::ErrorRecoveryCause::BudgetExceeded,
                    error,
                    ..
                },
            ..
        } => {
            assert!(error.contains("exceeded"), "{error}");
            *task_id
        }
        other => panic!("expected a budget pause, got {other:?}"),
    }
}

fn continue_after_error() -> AgentResume {
    AgentResume::ContinueAfterError { note: None }
}

/// Over the workflow's budget, an agent run the host started is held
/// before its next request and returned paused: its tool call ran, nothing
/// was cancelled, and `OnBudgetExceeded` fired once. Resumed without a
/// raise it is held again without a request; after `set_budget` raises the
/// limit, it continues where it was, without an error note.
#[tokio::test]
async fn an_agent_over_the_workflow_budget_pauses_until_it_is_raised() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let exceeded = Arc::new(AtomicUsize::new(0));
    let workflow = build(
        count_exceeded(builder(&server, &workspace, &gate).budget(LOW), &exceeded),
        &gate,
    );

    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    let id = assert_held(&error);
    assert_eq!(server.requests(), 1, "the next request is held");
    assert_eq!(gate.calls(), 1, "the call ran");
    assert_eq!(exceeded.load(Ordering::SeqCst), 1);
    let paused = workflow.agent_result(id).unwrap();
    assert_eq!(paused.status, AgentRunStatus::Paused);
    assert!(workflow.cancellation().is_none());

    // Still over the budget: held again, nothing sent.
    let error = within(workflow.resume_agent(&paused, continue_after_error()))
        .await
        .unwrap_err();
    assert_eq!(assert_held(&error), id);
    assert_eq!(server.requests(), 1);
    assert_eq!(exceeded.load(Ordering::SeqCst), 1, "once per crossing");

    workflow.set_budget(Some(1.0)).unwrap();
    assert_eq!(workflow.budget(), Some(1.0));
    let paused = workflow.agent_result(id).unwrap();
    let result = within(workflow.resume_agent(&paused, continue_after_error()))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.agent_run_id, id);
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2);
    assert!(bodies[1].contains("gate passed"), "{}", bodies[1]);
    assert!(!bodies[1].contains(ERROR_NOTE), "{}", bodies[1]);
    assert_eq!(gate.calls(), 1);
}

/// An agent's own budget holds only that agent's runs; raising it
/// releases them.
#[tokio::test]
async fn an_agent_over_its_own_budget_pauses_and_the_others_run() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let workflow = build(
        builder(&server, &workspace, &gate).agent_budget("coder", LOW),
        &gate,
    );

    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    let id = assert_held(&error);
    let reviewer = within(workflow.run_agent("reviewer", "Write."))
        .await
        .unwrap();
    assert_eq!(reviewer.status, AgentRunStatus::Completed);

    // A new run of the held agent starts and is held at its next request.
    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    assert_ne!(assert_held(&error), id);

    workflow.set_agent_budget("coder", Some(1.0)).unwrap();
    let paused = workflow.agent_result(id).unwrap();
    let result = within(workflow.resume_agent(&paused, continue_after_error()))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert!(matches!(
        workflow.set_agent_budget("nobody", None),
        Err(WorkflowError::AgentNotFound(_))
    ));
    assert!(workflow.set_budget(Some(f64::NAN)).is_err());
}

/// With `wait_for_budget_raise`, a run the host started waits in the call
/// while it is held; a live raise releases it.
#[tokio::test]
async fn a_live_raise_releases_a_waiting_run() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let workflow = Arc::new(build(
        builder(&server, &workspace, &gate)
            .budget(LOW)
            .wait_for_budget_raise(true),
        &gate,
    ));
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write.").await }
    });
    until(|| {
        workflow.agent_status("coder") == Some(AgentStatus::Waiting(WaitReason::BudgetExceeded))
    })
    .await;
    assert!(workflow.is_budget_blocked());
    assert_eq!(server.requests(), 1);

    workflow.set_budget(Some(1.0)).unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert!(!server.bodies()[1].contains(ERROR_NOTE));
}

/// A run inside another agent's tool call waits for a raise (its parent
/// waits on the call): the budget blocks the whole run until
/// `set_agent_budget` releases it.
#[tokio::test]
async fn a_live_raise_releases_a_delegated_run() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let workflow = Arc::new(build(
        delegating(&server, &workspace, &gate).agent_budget("reviewer", LOW),
        &gate,
    ));
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Delegate.").await }
    });
    within(workflow.budget_blocked()).await;
    assert_eq!(
        workflow.agent_status("reviewer"),
        Some(AgentStatus::Waiting(WaitReason::BudgetExceeded))
    );
    assert_eq!(server.requests(), 2);

    workflow.set_agent_budget("reviewer", Some(1.0)).unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "coder done");
    assert_eq!(server.requests(), 4);
    assert!(!workflow.is_budget_blocked());
}

/// A run paused by the budget, resumed with a higher budget, continues:
/// `resume_agent` sends its held request. A limit raised while it was live
/// is kept by a resume that passes the original limit.
#[tokio::test]
async fn resuming_with_a_higher_budget_continues_a_paused_run() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let workflow = build(
        persisted(builder(&server, &workspace, &gate).budget(LOW), &workspace),
        &gate,
    );
    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    let id = assert_held(&error);
    let path = latest(&workspace);
    drop(workflow);

    let gate = GateTool::new();
    let resumed = build(
        persisted(builder(&server, &workspace, &gate).budget(1.0), &workspace).resume_from(&path),
        &gate,
    );
    let paused = resumed.agent_result(id).unwrap();
    assert_eq!(paused.status, AgentRunStatus::Paused);
    let result = within(resumed.resume_agent(&paused, continue_after_error()))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(server.requests(), 2);

    // A live change is saved, and kept by a resume with the original limit.
    resumed.set_budget(Some(0.5)).unwrap();
    resumed.checkpoint().await.unwrap();
    let path = latest(&workspace);
    let state: WorkflowState =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(state.budget_limits.unwrap().limit, Some(0.5));
    drop(resumed);
    let limit = |builder: WorkflowBuilder| builder.resume_from(&path).build().unwrap().budget();
    let fresh = || persisted(builder(&server, &workspace, &gate), &workspace);
    assert_eq!(limit(fresh().budget(LOW)), Some(0.5), "the higher one");
    assert_eq!(limit(fresh().budget(2.0)), Some(2.0), "the builder's");
    assert_eq!(limit(fresh()), None, "none set");

    // A checkpoint from before limits were saved: the builder's.
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("budget_limits");
    std::fs::write(&path, json.to_string()).unwrap();
    assert_eq!(limit(fresh().budget(LOW)), Some(LOW));
}

/// The budget holds a delegated run and blocks the whole run; the host
/// stops there (the process ends). Resumed with a higher budget,
/// recovering the delegating run recovers the held run, which continues
/// from its pause: nothing runs twice.
#[tokio::test]
async fn a_held_delegation_continues_after_a_resume_with_a_higher_budget() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let workflow = Arc::new(build(
        persisted(
            delegating(&server, &workspace, &gate).agent_budget("reviewer", LOW),
            &workspace,
        ),
        &gate,
    ));
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Delegate.").await }
    });
    within(workflow.budget_blocked()).await;
    // Saved in its pause.
    let reviewer_saved = || {
        let path = latest(&workspace);
        let Ok(json) = std::fs::read_to_string(&path) else {
            return false;
        };
        let Ok(state) = serde_json::from_str::<WorkflowState>(&json) else {
            return false;
        };
        state.agent_runs.iter().any(|record| {
            record.agent_name == "reviewer"
                && std::fs::read_to_string(run_dir(&workspace).join(&record.task_file))
                    .is_ok_and(|task| task.contains("budget_exceeded"))
        })
    };
    until(reviewer_saved).await;
    let path = latest(&workspace);
    // The process ends.
    workflow.engine.persistence().unwrap().simulate_crash();
    run.abort();
    let _ = run.await;
    drop(workflow);
    assert_eq!(server.requests(), 2);

    let gate = GateTool::new();
    let resumed = build(
        persisted(
            delegating(&server, &workspace, &gate).agent_budget("reviewer", 1.0),
            &workspace,
        )
        .resume_from(&path),
        &gate,
    );
    let interrupted = resumed.interrupted_runs();
    assert_eq!(interrupted.len(), 1);
    assert_eq!(interrupted[0].agent_name, "coder");
    let result = within(resumed.recover_agent(&interrupted[0]))
        .await
        .unwrap();
    assert_eq!(result.summary, "coder done");
    assert_eq!(gate.calls(), 0, "the reviewer's call had run");
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 4, "{bodies:#?}");
    assert!(bodies[2].contains("gate passed"), "{}", bodies[2]);
}

/// Under `Cancel` the agent over the budget is cancelled; raised, the
/// run is reopened where it stopped, with its identity.
#[tokio::test]
async fn a_run_cancelled_by_the_budget_is_reopened() {
    let workspace = Workspace::new();
    let server = server().await;
    let gate = GateTool::new();
    let workflow = build(
        persisted(
            builder(&server, &workspace, &gate)
                .budget(LOW)
                .budget_exceeded_policy(BudgetExceededPolicy::Cancel),
            &workspace,
        ),
        &gate,
    );
    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::BudgetExceeded { .. }),
        "{error:?}"
    );
    let cancelled = workflow.result().agent_results[0].clone();
    assert_eq!(cancelled.status, AgentRunStatus::Cancelled);
    // Refused while the budget is exceeded.
    let error = within(workflow.reopen_agent(&cancelled, "Go on."))
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::BudgetExceeded { .. }),
        "{error:?}"
    );
    let path = latest(&workspace);
    drop(workflow);

    // Resumed from a checkpoint as an earlier version wrote it.
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("budget_limits");
    std::fs::write(&path, json.to_string()).unwrap();
    let gate = GateTool::new();
    let resumed = build(
        persisted(
            builder(&server, &workspace, &gate)
                .budget(1.0)
                .budget_exceeded_policy(BudgetExceededPolicy::Cancel),
            &workspace,
        )
        .resume_from(&path),
        &gate,
    );
    let previous = resumed.agent_result(cancelled.agent_run_id).unwrap();
    assert_eq!(previous.status, AgentRunStatus::Cancelled);
    let result = within(resumed.reopen_agent(&previous, "The budget was raised."))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.agent_run_id, cancelled.agent_run_id);
    let body = server.bodies().pop().unwrap();
    assert!(body.contains(REOPENED_NOTE), "{body}");
    assert!(body.contains("The budget was raised."), "{body}");
    // Only a cancelled run is reopened.
    let error = resumed.reopen_agent(&result, "Again.").await.unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("only a cancelled")),
        "{error}"
    );
}

/// An agent that doesn't stop within the grace period is recorded as the
/// task journal last saved it (its call running), not as it was launched,
/// so reopening it continues from there.
#[tokio::test]
async fn an_aborted_agent_keeps_its_latest_saved_task() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains("function_call_output") {
            complete_turn("done", "done")
        } else {
            tool_call_turn("block", "block", json!({}))
        })
    })
    .await;
    let block = ProbeTool::new("block", true);
    let workflow = Workflow::builder()
        .name("budget-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent_tool("coder", block.clone())
        .budget(0.0)
        .budget_exceeded_policy(BudgetExceededPolicy::Cancel)
        // Long enough for the call's write-ahead save under load.
        .cancel_grace_period(Duration::from_millis(500))
        .persistence(PersistenceConfig::new(run_dir(&workspace)))
        .build()
        .unwrap();
    let error = within(workflow.run_agent("coder", "Block."))
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::BudgetExceeded { .. }),
        "{error:?}"
    );
    let aborted = workflow.result().agent_results[0].clone();
    assert!(
        aborted.error.as_deref().unwrap().contains("was aborted"),
        "{:?}",
        aborted.error
    );
    assert_eq!(aborted.task.running_tool_calls.len(), 1, "saved mid-call");
    assert!(aborted.task.conversation.messages.len() >= 2);

    // The call may have been aborted before the tool itself started: its
    // write-ahead save comes first.
    let calls = block.calls();
    assert!(calls <= 1);
    workflow.set_budget(Some(1.0)).unwrap();
    let result = within(workflow.reopen_agent(&aborted, "")).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(block.calls(), calls, "in doubt, not run again");
}

/// Saved data from before budget holds loads: tasks and checkpoints
/// without the new fields; the new cause and wait reason round-trip.
#[test]
fn saved_data_with_and_without_budget_holds_loads() {
    let cause: arko_agents::ErrorRecoveryCause =
        serde_json::from_value(json!("budget_exceeded")).unwrap();
    assert_eq!(cause, arko_agents::ErrorRecoveryCause::BudgetExceeded);
    assert_eq!(
        serde_json::to_value(&cause).unwrap(),
        json!("budget_exceeded")
    );
    let reason: WaitReason = serde_json::from_value(json!("budget_exceeded")).unwrap();
    assert_eq!(reason, WaitReason::BudgetExceeded);
    let state: WorkflowState = serde_json::from_value(json!({
        "workflow_name": "legacy",
        "checkpoint_id": uuid::Uuid::new_v4(),
        "timestamp": chrono::Utc::now(),
        "memory_snapshot": [],
        "message_queues": {},
        "budget_spent": 1.5,
    }))
    .unwrap();
    assert!(state.budget_limits.is_none());
    let policy: BudgetExceededPolicy = serde_json::from_value(json!("cancel")).unwrap();
    assert_eq!(policy, BudgetExceededPolicy::Cancel);
    assert_eq!(BudgetExceededPolicy::default(), BudgetExceededPolicy::Pause);
}
