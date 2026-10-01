//! Plan budgets: agents held once a plan usage window's points are used
//! (as reported with each response), released by a raise; each plan
//! counted and limited on its own; the counting saved with checkpoints and
//! kept by a resume.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arko_agents::tool::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use arko_ai::SubscriptionPlan;
use parking_lot::Mutex;
use serde_json::{Value, json};

use super::*;
use crate::budget::{BudgetExceededPolicy, PlanBudget, PlanLimits, PlanUsageSpent, PlanWindow};
use crate::context::SharedContext;
use crate::hooks::{HookPoint, HookResult};
use crate::persistence::PersistenceConfig;
use crate::persistence::state::WorkflowState;
use crate::policy::{AgentPause, AgentResume};
use crate::test_support::{
    Reply, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;

/// The five-hour window's reset time the server reports.
const RESETS_AT: u64 = 4_000_000_000;

/// The five-hour window's level the server reports with response `index`
/// (40, then 60, then 61), and the seven-day window at 5 + index.
fn level(index: usize) -> f64 {
    match index {
        0 => 40.0,
        1 => 60.0,
        _ => 61.0,
    }
}

/// Answers the first two requests of each task with a `gate` call (the
/// second waiting for 20 points to be counted), then completes; each
/// response reports the plan usage of [`level`].
async fn server() -> ResponsesServer {
    let index = Arc::new(AtomicUsize::new(0));
    ResponsesServer::start_async(move |_, body| {
        let calls = body.matches("function_call_output").count();
        let stream = match calls {
            0 => tool_call_turn("gate0", "gate", json!({"points": 0.0})),
            1 => tool_call_turn("gate1", "gate", json!({"points": 20.0})),
            _ => complete_turn("done", "done"),
        };
        let index = index.fetch_add(1, Ordering::SeqCst);
        let headers = vec![
            format!("x-codex-primary-used-percent: {}", level(index)),
            "x-codex-primary-window-minutes: 300".to_string(),
            format!("x-codex-primary-reset-at: {RESETS_AT}"),
            format!("x-codex-secondary-used-percent: {}", 5 + index),
            "x-codex-secondary-window-minutes: 10080".to_string(),
        ];
        Box::pin(async move { Some(Reply::StreamWithHeaders { stream, headers }) })
    })
    .await
}

/// A tool that returns once the run has counted `points` of the five-hour
/// window (so the test knows the report was taken before the agent's next
/// request).
#[derive(Default)]
struct GateTool {
    context: OnceLock<Arc<SharedContext>>,
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
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let points = params.get("points").and_then(Value::as_f64).unwrap_or(0.0);
        let workflow = self.context.get().expect("attached").clone();
        let counted = || {
            workflow
                .budget
                .plan_spent()
                .five_hour
                .is_some_and(|spent| spent >= points)
        };
        tokio::time::timeout(Duration::from_secs(10), async {
            while !counted() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the report was taken");
        Ok("gate passed".to_string())
    }
}

/// The metadata of each `OnBudgetExceeded`.
type Crossings = Arc<Mutex<Vec<HashMap<String, Value>>>>;

fn build(
    server: &ResponsesServer,
    workspace: &Workspace,
    configure: impl FnOnce(WorkflowBuilder) -> WorkflowBuilder,
    crossings: &Crossings,
) -> Workflow {
    let gate = Arc::new(GateTool::default());
    let recorded = crossings.clone();
    let builder = Workflow::builder()
        .name("plan-budget-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .global_tool(gate.clone())
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .hook(
            HookPoint::OnBudgetExceeded,
            Arc::new(move |hook| {
                recorded.lock().push(hook.metadata.clone());
                HookResult::Continue
            }),
        );
    let workflow = configure(builder).build().unwrap();
    let _ = gate.context.set(workflow.context().clone());
    workflow
}

fn latest(workspace: &Workspace) -> std::path::PathBuf {
    let run = workspace.0.join(".run");
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    run.join(name.trim())
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(20), future)
        .await
        .expect("the workflow call returned")
}

fn assert_close(actual: Option<f64>, expected: f64) {
    let actual = actual.expect("a window");
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

/// Past the five-hour limit, the agent is held before its next request,
/// as for a dollar budget, and `OnBudgetExceeded` names the window once.
/// Raising the limit lets it go on. The counting is saved with the
/// checkpoint; a resume keeps the run's baseline and the higher limit.
#[tokio::test]
async fn a_plan_window_over_its_limit_holds_the_agent_until_it_is_raised() {
    let workspace = Workspace::new();
    let server = server().await;
    let crossings = Crossings::default();
    let workflow = build(
        &server,
        &workspace,
        |builder| builder.plan_budget(PlanBudget::none().five_hour(10.0)),
        &crossings,
    );

    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    let WorkflowError::AgentPaused {
        task_id,
        pause:
            AgentPause::ErrorRecovery {
                cause: arko_agents::ErrorRecoveryCause::BudgetExceeded,
                error,
            },
        ..
    } = &error
    else {
        panic!("expected a budget pause, got {error:?}");
    };
    assert!(
        error.contains("ChatGPT five-hour window 20.0 of 10 points"),
        "{error}"
    );
    assert_eq!(server.requests(), 2, "the next request is held");
    let spent = workflow.plan_usage_spent();
    assert_close(spent.five_hour, 20.0);
    assert_close(spent.seven_day, 1.0);
    {
        let crossings = crossings.lock();
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0]["plan"], "chatgpt");
        assert_eq!(crossings[0]["plan_window"], "five_hour");
        assert_eq!(crossings[0]["plan_points_limit"], 10.0);
    }

    // Still over: held again without a request.
    let paused = workflow.agent_result(*task_id).unwrap();
    let resume = || AgentResume::ContinueAfterError { note: None };
    let error = within(workflow.resume_agent(&paused, resume()))
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::AgentPaused { .. }),
        "{error}"
    );
    assert_eq!(server.requests(), 2);

    workflow
        .set_plan_budget(PlanBudget::none().five_hour(30.0))
        .unwrap();
    assert_eq!(workflow.plan_budget().five_hour, Some(30.0));
    let paused = workflow.agent_result(*task_id).unwrap();
    let result = within(workflow.resume_agent(&paused, resume()))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_close(workflow.plan_usage_spent().five_hour, 21.0);
    assert_eq!(crossings.lock().len(), 1);

    workflow.checkpoint().await.unwrap();
    let path = latest(&workspace);
    let state: WorkflowState =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let plan = state.plan_budget.unwrap();
    assert_eq!(plan.limits.five_hour, Some(30.0));
    assert_eq!(
        plan.plans[&SubscriptionPlan::ChatGpt]
            .window(PlanWindow::FiveHour)
            .unwrap()
            .baseline,
        40.0
    );
    assert_eq!(plan.usage, Default::default());
    drop(workflow);

    // Resumed: the higher limit applies, and the counting goes on from the
    // run's baseline (a new run's report at 61 adds nothing).
    let resumed = build(
        &server,
        &workspace,
        |builder| {
            builder
                .plan_budget(PlanBudget::none().five_hour(10.0).seven_day(50.0))
                .resume_from(&path)
        },
        &crossings,
    );
    assert_eq!(
        resumed.plan_budget(),
        PlanBudget::none().five_hour(30.0).seven_day(50.0)
    );
    assert_close(resumed.plan_usage_spent().five_hour, 21.0);
    let result = within(resumed.run_agent("coder", "Write again."))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_close(resumed.plan_usage_spent().five_hour, 21.0);
    drop(resumed);

    // No plan budget on the builder: none, the counting kept.
    let path = latest(&workspace);
    let resumed = build(
        &server,
        &workspace,
        |builder| builder.resume_from(&path),
        &crossings,
    );
    assert_eq!(resumed.plan_budget(), PlanBudget::none());
    assert_close(resumed.plan_usage_spent().five_hour, 21.0);
    drop(resumed);

    // A checkpoint from before plans were counted apart: its counting goes
    // on, taken over by the first plan to report.
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let plan = json["plan_budget"].as_object_mut().unwrap();
    let chatgpt = plan.remove("plans").unwrap()["chatgpt"].take();
    plan.insert("usage".to_string(), chatgpt);
    std::fs::write(&path, json.to_string()).unwrap();
    let resumed = build(
        &server,
        &workspace,
        |builder| {
            builder
                .plan_budget(PlanBudget::none().five_hour(22.0))
                .resume_from(&path)
        },
        &crossings,
    );
    assert_close(resumed.plan_usage_spent().five_hour, 21.0);
    let chatgpt = || resumed.plan_usage_spent_of(SubscriptionPlan::ChatGpt);
    assert_eq!(chatgpt(), PlanUsageSpent::default());
    let result = within(resumed.run_agent("coder", "Write again."))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_close(chatgpt().five_hour, 21.0);
    assert_close(resumed.plan_usage_spent().five_hour, 21.0);
    resumed.checkpoint().await.unwrap();
    drop(resumed);
    let path = latest(&workspace);

    // A checkpoint from before plan budgets: the builder's, counting afresh.
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("plan_budget");
    std::fs::write(&path, json.to_string()).unwrap();
    let resumed = build(
        &server,
        &workspace,
        |builder| {
            builder
                .plan_budget(PlanBudget::none().five_hour(5.0))
                .resume_from(&path)
        },
        &crossings,
    );
    assert_eq!(resumed.plan_budget(), PlanBudget::none().five_hour(5.0));
    assert_eq!(resumed.plan_usage_spent().five_hour, None);
}

/// Under `BudgetExceededPolicy::Cancel`, the agent is cancelled once a
/// window is over its limit, and new runs are refused.
#[tokio::test]
async fn under_cancel_a_plan_window_over_its_limit_cancels_the_agent() {
    let workspace = Workspace::new();
    let server = server().await;
    let crossings = Crossings::default();
    let workflow = build(
        &server,
        &workspace,
        |builder| {
            builder
                .plan_budget(PlanBudget::none().seven_day(0.5))
                .budget_exceeded_policy(BudgetExceededPolicy::Cancel)
        },
        &crossings,
    );
    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    assert!(
        matches!(
            &error,
            WorkflowError::PlanBudgetExceeded { plan: Some(plan), window, .. }
                if plan == "ChatGPT" && window == "seven-day"
        ),
        "{error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("ChatGPT seven-day window used 1.0 of 0.5 points"),
        "{error}"
    );
    assert_eq!(crossings.lock()[0]["plan_window"], "seven_day");
    let error = within(workflow.run_agent("coder", "Write."))
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::PlanBudgetExceeded { .. }),
        "{error:?}"
    );
}

/// A plan's own limit counts only that plan's reports: a Claude Code limit
/// leaves the ChatGPT agent going, and a ChatGPT limit, replacing the
/// general one for ChatGPT, holds it.
#[tokio::test]
async fn a_plans_own_limit_holds_agents_on_its_own_reports_only() {
    let workspace = Workspace::new();
    let server = server().await;
    let crossings = Crossings::default();
    let workflow = build(
        &server,
        &workspace,
        |builder| {
            builder.plan_budget(PlanBudget::none().claude_code(PlanLimits::none().five_hour(10.0)))
        },
        &crossings,
    );
    let result = within(workflow.run_agent("coder", "Write.")).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_close(
        workflow
            .plan_usage_spent_of(SubscriptionPlan::ChatGpt)
            .five_hour,
        21.0,
    );
    assert_eq!(
        workflow.plan_usage_spent_of(SubscriptionPlan::ClaudeCode),
        PlanUsageSpent::default()
    );
    assert!(crossings.lock().is_empty());

    workflow
        .set_plan_budget(
            PlanBudget::none()
                .five_hour(50.0)
                .chatgpt(PlanLimits::none().five_hour(10.0)),
        )
        .unwrap();
    let error = within(workflow.run_agent("coder", "Write again."))
        .await
        .unwrap_err();
    let WorkflowError::AgentPaused {
        pause: AgentPause::ErrorRecovery { error, .. },
        ..
    } = &error
    else {
        panic!("expected a budget pause, got {error:?}");
    };
    assert!(
        error.contains("ChatGPT five-hour window 21.0 of 10 points"),
        "{error}"
    );
    assert_eq!(server.requests(), 3, "held before its first request");
    assert!(
        workflow
            .set_plan_budget(PlanBudget::none().chatgpt(PlanLimits::none().seven_day(-1.0)))
            .is_err()
    );
}
