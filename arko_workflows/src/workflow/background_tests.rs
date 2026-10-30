//! Agents' tool calls that move to the background, through
//! `WorkflowBuilder::background_jobs`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arko_agents::terminal::{JobKind, JobState, TerminalConfig, TerminalManager};
use arko_agents::{TaskContext, Tool, ToolBackground, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};
use tokio::sync::Notify;

use super::*;
use crate::test_support::{
    ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};

/// A tool whose calls run until the test releases them, or their stop
/// signal fires.
#[derive(Debug, Default)]
struct Slow {
    started: Notify,
    release: Notify,
    stopped: AtomicBool,
}

#[async_trait::async_trait]
impl Tool for Slow {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "slow".into(),
            description: "Runs until released.".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    fn background_mode(&self) -> ToolBackground {
        ToolBackground::Detach
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.started.notify_one();
        tokio::select! {
            () = self.release.notified() => Ok("slow done".into()),
            () = context.stop.stopping() => {
                self.stopped.store(true, Ordering::SeqCst);
                Ok("slow wound down".into())
            }
        }
    }
}

fn worker() -> arko_agents::AgentConfig {
    let mut worker = agent_config("Worker");
    worker.tool_config.available_tool_names = vec!["slow".into()];
    worker
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .expect("in time")
}

/// Moves the first call that can move, once there is one, and returns its
/// owner.
async fn move_first(jobs: &TerminalManager) -> uuid::Uuid {
    within(async {
        loop {
            if let Some(call) = jobs.foreground_calls().pop() {
                assert_eq!(jobs.move_to_background(call.owner, None), 1);
                return call.owner;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
}

/// An agent's call moves when the host moves it: the agent goes on with the
/// job's ID as the call's result, and the job keeps running after the run.
#[tokio::test]
async fn an_agents_call_moves_to_the_background() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn("c0", "slow", json!({})),
            _ => complete_turn("c1", "done"),
        })
    })
    .await;
    let jobs = TerminalManager::new(TerminalConfig::new(workspace.0.join(".jobs"))).unwrap();
    let slow = Arc::new(Slow::default());
    let workflow = Workflow::builder()
        .name("background-jobs-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("worker", worker())
        .global_tool(slow.clone())
        .background_jobs(jobs.clone())
        .build()
        .unwrap();
    let (result, owner) = tokio::join!(workflow.run_agent("worker", "Run it."), move_first(&jobs));
    let result = result.unwrap();
    assert_eq!(owner, result.agent_run_id);
    let output = server.bodies()[1].clone();
    assert!(
        output.contains("in the background as job "),
        "the next request has the moved result: {output}"
    );
    let running = jobs.jobs();
    assert_eq!(running.len(), 1);
    assert_eq!(
        running[0].kind,
        JobKind::Call {
            tool_name: "slow".into()
        }
    );
    slow.release.notify_one();
    let job = running[0].job_id;
    within(async {
        while !jobs.status(owner, job, None).unwrap().state.is_terminal() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert_eq!(
        jobs.status(owner, job, None).unwrap().state,
        JobState::Succeeded
    );
    // Its note waits for the agent's next run.
    assert!(jobs.has_news(owner));
}

/// When a subagent's run ends, its moved calls stop.
#[tokio::test]
async fn a_subagents_moved_calls_stop_when_its_run_ends() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        let turn = body.matches("\"function_call_output\"").count();
        Some(match turn {
            0 => tool_call_turn("w0", "slow", json!({})),
            _ => complete_turn("w1", "worker done"),
        })
    })
    .await;
    let jobs = TerminalManager::new(TerminalConfig::new(workspace.0.join(".jobs"))).unwrap();
    let slow = Arc::new(Slow::default());
    let workflow = Workflow::builder()
        .name("background-subagent-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Lead"))
        .agent("worker", worker())
        .global_tool(slow.clone())
        .workflow_services(
            crate::services::WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("worker"),
        )
        .subagents(crate::subagents::SubagentsConfig::default())
        .background_jobs(jobs.clone())
        .build()
        .unwrap();
    let ctx = workflow.context();
    ctx.subagents
        .start(
            ctx,
            uuid::Uuid::new_v4(),
            "lead",
            0,
            &workspace.0,
            "worker",
            "Run it.",
            None,
            false,
            None,
            None,
            std::collections::BTreeMap::new(),
        )
        .await
        .unwrap();
    let owner = move_first(&jobs).await;
    within(async {
        while !workflow
            .subagents()
            .iter()
            .all(|run| !run.state.is_active())
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    let job = job_named(&server.bodies().concat());
    let job = within(async {
        loop {
            let ended = jobs.status(owner, job, None).unwrap();
            if ended.state.is_terminal() {
                return ended;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert_eq!(job.state, JobState::Cancelled);
    assert_eq!(
        job.error.as_deref(),
        Some("stopped because its agent's run ended")
    );
    assert!(slow.stopped.load(Ordering::SeqCst));
}

/// The job a moved result in `text` names.
fn job_named(text: &str) -> uuid::Uuid {
    let (_, rest) = text
        .split_once(arko_agents::terminal::IN_THE_BACKGROUND)
        .expect("a moved result");
    rest[..36].parse().unwrap()
}

/// A server whose worker calls `slow`, then completes.
#[cfg(unix)]
async fn slow_worker_server() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        let turn = body.matches("\"function_call_output\"").count();
        Some(match turn {
            0 => tool_call_turn("w0", "slow", json!({})),
            _ => complete_turn("w1", "worker done"),
        })
    })
    .await
}

/// A workflow with a lead and a worker that calls `slow`, whose jobs are
/// `jobs`, with subagents when `subagents` is set.
#[cfg(unix)]
fn lead_and_worker(
    server: &ResponsesServer,
    workspace: &Workspace,
    slow: &Arc<Slow>,
    jobs: &Arc<TerminalManager>,
    subagents: bool,
) -> Workflow {
    let builder = Workflow::builder()
        .name("worker-lifetime-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Lead"))
        .agent("worker", worker())
        .global_tool(slow.clone())
        .background_jobs(jobs.clone());
    let builder = if subagents {
        builder
            .workflow_services(
                crate::services::WorkflowServicesConfig::default()
                    .allow_caller("lead")
                    .allow_target("worker"),
            )
            .subagents(crate::subagents::SubagentsConfig::default())
    } else {
        builder
    };
    builder.build().unwrap()
}

/// Starts a worker of `workflow` for a lead run `lead`, and waits until its
/// `slow` call runs.
#[cfg(unix)]
async fn start_worker(workflow: &Workflow, workspace: &Workspace, slow: &Slow, lead: uuid::Uuid) {
    let ctx = workflow.context();
    ctx.subagents
        .start(
            ctx,
            lead,
            "lead",
            0,
            &workspace.0,
            "worker",
            "Run it.",
            None,
            false,
            None,
            None,
            std::collections::BTreeMap::new(),
        )
        .await
        .unwrap();
    within(slow.started.notified()).await;
}

/// Starts a monitor of `lead` that lasts a second, and while its workers
/// run.
#[cfg(unix)]
async fn kept_monitor(
    jobs: &Arc<TerminalManager>,
    workspace: &Workspace,
    lead: uuid::Uuid,
) -> uuid::Uuid {
    use arko_agents::terminal::{MonitorLifetime, MonitorOptions};
    let context = TaskContext::new(lead, "lead", "Lead it.", &workspace.0);
    jobs.start_monitor(
        &context,
        "sleep 60",
        MonitorOptions::lines(None)
            .with_expiry(Duration::from_secs(1))
            .with_lifetime(MonitorLifetime::WhileWorkersRun),
    )
    .await
    .unwrap()
    .job_id
}

/// Waits for `lead`'s monitor `id` to end, and checks that it ended after
/// its workers.
#[cfg(unix)]
async fn ended_after_workers(jobs: &TerminalManager, lead: uuid::Uuid, id: uuid::Uuid) {
    let ended = within(async {
        loop {
            let ended = jobs.status(lead, id, None).unwrap();
            if ended.state.is_terminal() {
                return ended;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert_eq!(
        (ended.state, ended.error.as_deref()),
        (
            JobState::TimedOut,
            Some("ended after its agent's workers finished")
        )
    );
}

/// Whether `monitor_start` offers `lasts`.
#[cfg(unix)]
fn offers_lasts(jobs: &Arc<TerminalManager>) -> bool {
    arko_agents::terminal::MonitorTool::all(jobs.clone())[0]
        .definition()
        .parameters
        .iter()
        .any(|param| param.name == "lasts")
}

/// With subagents, the manager learns whether an agent's workers run. A
/// monitor of a lead that lasts while they do goes on past its time while
/// its worker runs, and ends when the worker finishes. `monitor_start`
/// offers that lifetime.
#[cfg(unix)]
#[tokio::test]
async fn a_leads_monitor_lasts_while_its_worker_runs() {
    let workspace = Workspace::new();
    let server = slow_worker_server().await;
    let jobs = TerminalManager::new(TerminalConfig::new(workspace.0.join(".jobs"))).unwrap();
    assert!(!offers_lasts(&jobs));
    let slow = Arc::new(Slow::default());
    let workflow = lead_and_worker(&server, &workspace, &slow, &jobs, true);
    assert!(offers_lasts(&jobs));
    let lead = uuid::Uuid::new_v4();
    start_worker(&workflow, &workspace, &slow, lead).await;
    let monitor = kept_monitor(&jobs, &workspace, lead).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        jobs.status(lead, monitor, None).unwrap().state,
        JobState::Running
    );
    slow.release.notify_one();
    ended_after_workers(&jobs, lead, monitor).await;
    assert!(
        workflow
            .subagents()
            .iter()
            .all(|run| !run.state.is_active())
    );
}

/// A workflow that is dropped has no workers running, even while its host
/// still holds its context, so the monitors its workers kept end.
#[cfg(unix)]
#[tokio::test]
async fn a_dropped_workflows_workers_keep_no_monitor() {
    let workspace = Workspace::new();
    let server = slow_worker_server().await;
    let jobs = TerminalManager::new(TerminalConfig::new(workspace.0.join(".jobs"))).unwrap();
    let slow = Arc::new(Slow::default());
    let workflow = lead_and_worker(&server, &workspace, &slow, &jobs, true);
    let lead = uuid::Uuid::new_v4();
    start_worker(&workflow, &workspace, &slow, lead).await;
    let monitor = kept_monitor(&jobs, &workspace, lead).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let context = workflow.context().clone();
    drop(workflow);
    ended_after_workers(&jobs, lead, monitor).await;
    assert!(!offers_lasts(&jobs));
    drop(context);
}

/// Workflows that share a manager each tell it of their own workers. A
/// lead's monitor is kept by its own workflow's worker whichever workflow
/// was built last, and a workflow without subagents takes nothing away.
#[cfg(unix)]
#[tokio::test]
async fn workflows_that_share_a_manager_keep_their_own_monitors() {
    let workspace = Workspace::new();
    let server = slow_worker_server().await;
    let jobs = TerminalManager::new(TerminalConfig::new(workspace.0.join(".jobs"))).unwrap();
    let first_slow = Arc::new(Slow::default());
    let first = lead_and_worker(&server, &workspace, &first_slow, &jobs, true);
    let lead = uuid::Uuid::new_v4();
    start_worker(&first, &workspace, &first_slow, lead).await;
    let monitor = kept_monitor(&jobs, &workspace, lead).await;
    let second_slow = Arc::new(Slow::default());
    let _second = lead_and_worker(&server, &workspace, &second_slow, &jobs, true);
    let _plain = lead_and_worker(
        &server,
        &workspace,
        &Arc::new(Slow::default()),
        &jobs,
        false,
    );
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        jobs.status(lead, monitor, None).unwrap().state,
        JobState::Running
    );
    assert!(offers_lasts(&jobs));
    first_slow.release.notify_one();
    ended_after_workers(&jobs, lead, monitor).await;
}
