//! Subagents that start subagents. A stop reaches the whole tree below the
//! run it stops, and a run that ends stops its subagents.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use arko_agents::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::tests::{Gate, Step, bodies_of, result_of, role, start, turn, until, wait, within};
use super::*;
use crate::services::WorkflowServicesConfig;
use crate::test_support::{ResponsesServer, Workspace, agent_config, tool_call_turn};
use crate::{RunStopIntent, StopRequest, Workflow, WorkflowBuilder};

/// A server that answers each conversation from its script, turn by turn,
/// and completes it (`<role> done`) once the script is over. A run told its
/// run is being stopped saves its progress and completes at once.
pub(super) async fn serve_tree(scripts: Vec<(&'static str, Vec<Step>)>) -> ResponsesServer {
    let scripts: HashMap<String, Vec<Step>> = scripts
        .into_iter()
        .map(|(role, steps)| (role.to_string(), steps))
        .collect();
    ResponsesServer::start(move |_, body| {
        let role = role(body);
        let turn = turn(body);
        let id = format!("{role}_{turn}");
        if body.contains("Your run is being stopped") {
            return Some(tool_call_turn(
                &id,
                "complete_task",
                json!({ "summary": format!("{role} saved its progress") }),
            ));
        }
        Some(match scripts.get(&role).and_then(|steps| steps.get(turn)) {
            Some((tool, arguments)) => tool_call_turn(&id, tool, arguments.clone()),
            None => tool_call_turn(
                &id,
                "complete_task",
                json!({ "summary": format!("{role} done") }),
            ),
        })
    })
    .await
}

/// A workflow where `lead` and `worker` start `worker` runs.
pub(super) fn nested_builder(
    server: &ResponsesServer,
    workspace: &Workspace,
    config: SubagentsConfig,
) -> WorkflowBuilder {
    Workflow::builder()
        .name("nesting-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Lead"))
        .agent("worker", agent_config("Worker"))
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_caller("worker")
                .allow_target("worker"),
        )
        .subagents(config)
}

/// A tool that waits until the test opens it, or the task's stop signal
/// fires.
#[derive(Debug)]
pub(super) struct Barrier {
    name: &'static str,
    pub(super) open: tokio::sync::Notify,
}

impl Barrier {
    pub(super) fn new(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            open: tokio::sync::Notify::new(),
        })
    }
}

#[async_trait::async_trait]
impl Tool for Barrier {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.to_string(),
            description: "Waits.".to_string(),
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
        tokio::select! {
            _ = self.open.notified() => Ok("opened".to_string()),
            _ = context.stop.stopping() => Ok("stopped".to_string()),
        }
    }
}

/// A lead that starts `r1`, which starts `r2`, which starts `r3`. `r3` waits
/// at the `gate` tool, and `r1` and `r2` wait for their subagents.
struct Tree {
    server: ResponsesServer,
    workflow: Arc<Workflow>,
    gate: Arc<Gate>,
    hold: Arc<Barrier>,
    events: tokio::sync::broadcast::Receiver<SubagentEvent>,
    _workspace: Workspace,
}

impl Tree {
    /// The tree once `r3` is at its gate. `r1` does `r1_script`.
    async fn grow(r1_script: Vec<Step>) -> Self {
        let server = serve_tree(vec![
            ("lead", vec![start("Level one.")]),
            ("r1", r1_script),
            ("r2", vec![start("Level three."), wait()]),
            ("r3", vec![("gate", json!({}))]),
        ])
        .await;
        let workspace = Workspace::new();
        let gate = Arc::new(Gate::default());
        let hold = Barrier::new("hold");
        let workflow = Arc::new(
            nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(3))
                .agent_tool("worker", gate.clone())
                .agent_tool("worker", hold.clone())
                .build()
                .unwrap(),
        );
        let events = workflow.subscribe_subagents();
        within(workflow.run_agent("lead", "Delegate."))
            .await
            .unwrap();
        within(gate.started.notified()).await;
        let tree = Self {
            server,
            workflow,
            gate,
            hold,
            events,
            _workspace: workspace,
        };
        assert_eq!(tree.handles(), ["r1", "r2", "r3"]);
        assert_eq!(
            tree.depths(),
            [
                ("r1".to_string(), 1),
                ("r2".to_string(), 2),
                ("r3".to_string(), 3)
            ]
        );
        tree
    }

    /// The usual tree, where `r1` and `r2` wait for their subagents.
    async fn grow_waiting() -> Self {
        Self::grow(vec![start("Level two."), wait()]).await
    }

    fn handles(&self) -> Vec<String> {
        self.workflow
            .subagents()
            .into_iter()
            .map(|run| run.handle)
            .collect()
    }

    fn depths(&self) -> Vec<(String, u32)> {
        self.workflow
            .subagents()
            .into_iter()
            .map(|run| (run.handle, run.depth))
            .collect()
    }

    fn record(&self, handle: &str) -> SubagentRecord {
        self.workflow
            .subagents()
            .into_iter()
            .find(|run| run.handle == handle)
            .unwrap()
    }

    fn run_id(&self, handle: &str) -> Uuid {
        self.record(handle).run_id
    }

    async fn until_in(&self, handle: &str, state: SubagentState) {
        until(|| self.record(handle).state == state).await;
    }

    /// The stops reported so far, by run, each with the reason and whether
    /// the run's lead stopped it.
    fn stops(&mut self) -> BTreeMap<String, (String, bool)> {
        let mut stops = BTreeMap::new();
        while let Ok(event) = self.events.try_recv() {
            if let SubagentEventKind::Stopped { reason, by_caller } = event.kind {
                stops.insert(event.handle, (reason, by_caller));
            }
        }
        stops
    }
}

fn stopped(reason: &str, by_caller: bool) -> (String, bool) {
    (reason.to_string(), by_caller)
}

/// Stopping the run at the top of a tree stops the runs under it, with a
/// wrap-up or without one. Each is told which run's stop reached it.
#[tokio::test]
async fn a_stop_of_the_top_of_a_tree_stops_everything_below() {
    for wrap_up in [true, false] {
        let mut tree = Tree::grow_waiting().await;
        let mut request = StopRequest::operator("enough");
        if wrap_up {
            request = request.wrap_up(Duration::from_secs(30));
        }
        assert!(tree.workflow.stop_subagent(tree.run_id("r1"), request));
        for handle in ["r1", "r2", "r3"] {
            tree.until_in(handle, SubagentState::Stopped).await;
        }
        assert_eq!(
            tree.stops(),
            BTreeMap::from([
                ("r1".to_string(), stopped("enough", false)),
                ("r2".to_string(), stopped("its lead r1 was stopped", false)),
                ("r3".to_string(), stopped("its lead r2 was stopped", false)),
            ]),
            "wrap-up {wrap_up}"
        );
        // Each stopped run is told why in its record too.
        assert_eq!(
            tree.record("r3").error.as_deref(),
            Some("its lead r2 was stopped")
        );
        // The stop of a run that was stopped for its lead restarts nothing.
        assert!(tree.workflow.active_subagents().is_empty());
    }
}

/// Stopping a run in the middle stops its subagents, and the run above it
/// goes on, told that its subagent stopped.
#[tokio::test]
async fn a_stop_of_the_middle_of_a_tree_stops_what_is_below_it() {
    let mut tree = Tree::grow_waiting().await;
    let request = StopRequest::operator("enough").wrap_up(Duration::from_secs(30));
    assert!(tree.workflow.stop_subagent(tree.run_id("r2"), request));
    tree.until_in("r2", SubagentState::Stopped).await;
    tree.until_in("r3", SubagentState::Stopped).await;
    // r1 hears of r2's stop, and reports.
    tree.until_in("r1", SubagentState::Finished).await;
    assert_eq!(
        tree.stops(),
        BTreeMap::from([
            ("r2".to_string(), stopped("enough", false)),
            ("r3".to_string(), stopped("its lead r2 was stopped", false)),
        ])
    );
    let r1 = bodies_of(&tree.server, "r1");
    let told = result_of(&r1[2], "r1", 1);
    assert!(
        told.contains(
            "r2 (agent `worker`) stopped before it finished: the workflow's host stopped it (enough)."
        ),
        "{told}"
    );
    // r3's stop is news for r2, which had ended already.
    assert!(!told.contains("r3"), "{told}");
}

/// A run's `stop_agent` of its subagent stops that subagent's subagents too,
/// and says so.
#[tokio::test]
async fn a_stop_agent_of_a_subagent_stops_the_runs_below_it() {
    let mut tree = Tree::grow(vec![
        start("Level two."),
        ("hold", json!({})),
        ("stop_agent", json!({ "run": "r2", "reason": "not needed" })),
        wait(),
    ])
    .await;
    // r1 stops r2 now that r3 is at its gate.
    tree.hold.open.notify_one();
    tree.until_in("r1", SubagentState::Finished).await;
    tree.until_in("r2", SubagentState::Stopped).await;
    tree.until_in("r3", SubagentState::Stopped).await;
    let r1 = bodies_of(&tree.server, "r1");
    let stopping = result_of(&r1[3], "r1", 2);
    assert!(
        stopping.starts_with("r2 is stopping")
            && stopping.contains("Its own agent run r3 is stopping with it."),
        "{stopping}"
    );
    assert_eq!(
        tree.stops(),
        BTreeMap::from([
            ("r2".to_string(), stopped("not needed", true)),
            ("r3".to_string(), stopped("its lead r2 was stopped", false)),
        ])
    );
    assert_eq!(tree.gate.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// A run stopped on its own, not through the subagents, ends and takes its
/// subagents with it, since they would report to a run that is gone.
#[tokio::test]
async fn a_run_that_ends_stops_its_subagents() {
    let mut tree = Tree::grow_waiting().await;
    let request = StopRequest::operator("replacing it").wrap_up(Duration::from_secs(30));
    assert!(
        tree.workflow
            .stop_run(tree.run_id("r1"), request, RunStopIntent::Relaunch)
    );
    for handle in ["r1", "r2", "r3"] {
        tree.until_in(handle, SubagentState::Stopped).await;
    }
    let stops = tree.stops();
    assert_eq!(stops["r2"], stopped("its lead r1 ended", false));
    assert_eq!(stops["r3"], stopped("its lead r2 was stopped", false));
}

/// A subagent waiting for a place is stopped before it starts.
#[tokio::test]
async fn a_stop_reaches_the_subagents_still_waiting_for_a_place() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("First."), start("Second."), wait()]),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_depth(2).max_running(2),
    )
    .agent_tool("worker", gate.clone())
    .build()
    .unwrap();
    let mut events = workflow.subscribe_subagents();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(gate.started.notified()).await;
    until(|| workflow.subagents().len() == 3).await;
    let record = |handle: &str| {
        workflow
            .subagents()
            .into_iter()
            .find(|run| run.handle == handle)
            .unwrap()
    };
    assert_eq!(record("r3").state, SubagentState::Queued);

    let request = StopRequest::operator("enough").wrap_up(Duration::from_secs(30));
    assert!(workflow.stop_subagent(record("r1").run_id, request));
    let third = record("r3");
    assert_eq!(third.state, SubagentState::Stopped);
    assert_eq!(third.error.as_deref(), Some("its lead r1 was stopped"));
    until(|| workflow.active_subagents().is_empty()).await;
    // It never ran, and its lead is told why.
    assert!(bodies_of(&server, "r3").is_empty());
    let mut reasons = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let SubagentEventKind::Stopped { reason, by_caller } = event.kind {
            reasons.push((event.handle, reason, by_caller));
        }
    }
    assert!(
        reasons.contains(&("r3".into(), "its lead r1 was stopped".into(), false)),
        "{reasons:?}"
    );
}
