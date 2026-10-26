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
