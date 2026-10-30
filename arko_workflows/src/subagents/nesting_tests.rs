//! Subagents that start subagents. A stop reaches the whole tree below the
//! run it stops, and a run that ends stops its subagents.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use arko_agents::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::tests::{
    Gate, Step, bodies_of, complete, offered, result_of, role, start, turn, until, wait, within,
};
use super::*;
use crate::persistence::PersistenceConfig;
use crate::services::WorkflowServicesConfig;
use crate::test_support::{ResponsesServer, Workspace, agent_config, tool_call_turn};
use crate::workflow::result::AgentRunStatus;
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
            read_only: false,
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

/// A subagent started by a subagent is told its lead by handle, and one
/// started by the host's run by that run's agent.
#[tokio::test]
async fn a_nested_subagent_is_told_its_lead_by_handle() {
    let tree = Tree::grow_waiting().await;
    for (handle, lead) in [("r1", "agent `lead`"), ("r2", "r1"), ("r3", "r2")] {
        let first = &bodies_of(&tree.server, handle)[0];
        assert!(
            first.contains(&format!("started by {lead} (your lead)")),
            "{handle}"
        );
    }
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
    // It names r3, which stopped with it.
    assert!(told.contains("Its worker r3 stopped with it."), "{told}");
}

/// The news of a run that failed names the subagents that stopped with it.
#[tokio::test]
async fn a_failed_run_s_news_names_the_subagents_that_stopped_with_it() {
    use crate::test_support::Reply;
    let server = ResponsesServer::start_async(|_, body| {
        let (role, turn) = (role(&body), turn(&body));
        let id = format!("{role}_{turn}");
        let reply = match (role.as_str(), turn) {
            ("lead", 0) => Reply::Stream(tool_call_turn(&id, "start_agent", start_args("One."))),
            ("lead", 1) => Reply::Stream(tool_call_turn(&id, "wait_for_agents", json!({}))),
            ("r1", 0) => Reply::Stream(tool_call_turn(&id, "start_agent", start_args("Two."))),
            ("r1", _) => Reply::Status {
                status: 401,
                retry_after: None,
                body: json!({"error": {"message": "synthetic 401", "type": "auth"}}).to_string(),
            },
            ("r2", _) => Reply::Stream(tool_call_turn(&id, "gate", json!({}))),
            _ => Reply::Stream(tool_call_turn(
                &id,
                "complete_task",
                json!({ "summary": format!("{role} done") }),
            )),
        };
        Box::pin(async move { Some(reply) })
    })
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(3))
        .agent_tool("worker", gate)
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let lead = bodies_of(&server, "lead");
    let told = result_of(&lead[2], "lead", 1);
    assert!(
        told.contains("r1 (agent `worker`) stopped with an error:")
            && told.contains("\n\nIts worker r2 stopped with it.\n\nIts run can't go on"),
        "{told}"
    );
}

fn start_args(task: &str) -> Value {
    json!({ "agent": "worker", "task": task })
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

/// The kind of the stop that ends a run is the kind of the stop of the
/// subagents it takes with it, and of theirs in turn.
#[tokio::test]
async fn a_run_that_ends_in_a_drain_stops_its_subagents_in_a_drain() {
    let tree = Tree::grow_waiting().await;
    let request = StopRequest::drain("moving").wrap_up(Duration::from_secs(30));
    assert!(
        tree.workflow
            .stop_run(tree.run_id("r1"), request, RunStopIntent::Relaunch)
    );
    for handle in ["r1", "r2", "r3"] {
        tree.until_in(handle, SubagentState::Stopped).await;
    }
    let stop_of = |handle: &str| {
        tree.workflow
            .agent_result(tree.run_id(handle))
            .unwrap()
            .stopped_by
            .unwrap()
    };
    for (handle, reason) in [
        ("r1", "moving"),
        ("r2", "its lead r1 ended"),
        ("r3", "its lead r2 was stopped"),
    ] {
        let stop = stop_of(handle);
        assert_eq!(
            (stop.kind, stop.reason.as_str()),
            (crate::StopKind::Drain, reason),
            "{handle}"
        );
    }
}

/// The lead of a subagent that stopped because its own lead did is told so,
/// in its mailbox, and not in the words of a stop by the host.
#[tokio::test]
async fn the_news_of_a_subagent_stopped_for_its_lead_says_so() {
    let tree = Tree::grow_waiting().await;
    let request = StopRequest::operator("replacing it").wrap_up(Duration::from_secs(30));
    assert!(
        tree.workflow
            .stop_run(tree.run_id("r1"), request, RunStopIntent::Relaunch)
    );
    for handle in ["r1", "r2", "r3"] {
        tree.until_in(handle, SubagentState::Stopped).await;
    }
    // Their leads ended, so the news waits in their mailboxes.
    let headings = |handle: &str| {
        tree.workflow
            .pending_news(tree.run_id(handle))
            .into_iter()
            .map(|item| item.heading)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        headings("r1"),
        ["r2 (agent `worker`) stopped before it finished: its lead r1 ended."]
    );
    assert_eq!(
        headings("r2"),
        ["r3 (agent `worker`) stopped before it finished: its lead r2 was stopped."]
    );
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

/// A lead cut back to before the message that started a subagent no longer
/// has that subagent. It is detached, nobody reads its report, so it is
/// stopped with the subagents it started, and the lead's conversation hears
/// nothing of them.
#[tokio::test]
async fn a_cut_back_stops_the_subagents_it_detaches_with_their_own() {
    let server = ResponsesServer::start(|_, body| {
        let role = role(body);
        let id = format!("{role}_{}", turn(body));
        let complete =
            |summary: &str| tool_call_turn(&id, "complete_task", json!({ "summary": summary }));
        Some(match role.as_str() {
            _ if body.contains("Your run is being stopped") => complete("saved its progress"),
            "lead" if body.contains("Something else.") => complete("lead done again"),
            "lead" if body.contains("Delegate.") && !body.contains("Started r1") => {
                let (tool, arguments) = start("Level two.");
                tool_call_turn(&id, tool, arguments)
            }
            "r1" if turn(body) == 0 => {
                let (tool, arguments) = start("Level three.");
                tool_call_turn(&id, tool, arguments)
            }
            "r1" => tool_call_turn(&id, "wait_for_agents", json!({})),
            "r2" => tool_call_turn(&id, "gate", json!({})),
            _ => complete("lead done"),
        })
    })
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(3))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let mut events = workflow.subscribe_subagents();
    let first = within(workflow.run_agent("lead", "First.")).await.unwrap();
    let delegated = within(workflow.continue_agent(&first, "Delegate."))
        .await
        .unwrap();
    within(gate.started.notified()).await;
    let index = delegated
        .task()
        .unwrap()
        .user_turns()
        .iter()
        .find(|turn| turn.text.starts_with("Delegate."))
        .unwrap()
        .index;
    let again = within(workflow.continue_agent_with(
        &delegated,
        "Something else.",
        crate::ContinueOptions::new().cut_back_to(index),
    ))
    .await
    .unwrap();
    assert_eq!(again.summary, "lead done again");

    let record = |handle: &str| {
        workflow
            .subagents()
            .into_iter()
            .find(|run| run.handle == handle)
            .unwrap()
    };
    until(|| workflow.active_subagents().is_empty()).await;
    let (r1, r2) = (record("r1"), record("r2"));
    assert!(r1.detached && !r2.detached);
    assert_eq!(r1.state, SubagentState::Stopped);
    assert_eq!(r2.state, SubagentState::Stopped);
    assert_eq!(
        r1.error.as_deref(),
        Some("its lead's conversation was cut back to before it started")
    );
    assert_eq!(r2.error.as_deref(), Some("its lead r1 was stopped"));
    // The lead's conversation didn't hear of r1, started or stopped.
    let last = bodies_of(&server, "lead").last().unwrap().clone();
    assert!(!last.contains("r1 (agent"), "{last}");
    assert!(!last.contains("saved its progress"), "{last}");
    let mut stopped = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let SubagentEventKind::Stopped { by_caller, .. } = event.kind {
            stopped.push((event.handle, by_caller));
        }
    }
    stopped.sort();
    assert_eq!(
        stopped,
        [("r1".to_string(), false), ("r2".to_string(), false)]
    );
}

/// A subagent that started subagents can't report while they run or wait for
/// a place. Its `complete_task` is answered with the runs, and it reports
/// once they are done.
#[tokio::test]
async fn a_subagent_cant_report_while_its_subagents_are_unfinished() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        (
            "r1",
            vec![
                start("First."),
                start("Second."),
                complete("Too early."),
                ("wait_for_agents", json!({ "all": true })),
            ],
        ),
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
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| bodies_of(&server, "r1").len() >= 4).await;
    let refused = result_of(&bodies_of(&server, "r1")[3], "r1", 2);
    assert_eq!(
        refused,
        "Your agent runs r2 (running) and r3 (queued) haven't finished. Wait for them with wait_for_agents, or stop the ones you don't need with stop_agent, then report again."
    );
    // The early report went nowhere.
    let r1 = workflow.subagents().remove(0);
    assert_eq!(r1.state, SubagentState::Running);
    assert_eq!(r1.report, None);
    assert!(workflow.pending_news(lead.agent_run_id).is_empty());

    gate.open.notify_one();
    until(|| workflow.subagents()[0].state == SubagentState::Finished).await;
    let r1 = workflow.subagents().remove(0);
    assert_eq!(r1.report.as_deref(), Some("r1 done"));
    let news = workflow.pending_news(lead.agent_run_id);
    let reports: Vec<&str> = news.iter().map(|item| item.body.as_str()).collect();
    assert_eq!(reports, ["r1 done"]);
}

/// One unfinished subagent is named in the singular.
#[tokio::test]
async fn one_unfinished_subagent_is_named_in_the_singular() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("First."), complete("Too early.")]),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| bodies_of(&server, "r1").len() >= 3).await;
    assert_eq!(
        result_of(&bodies_of(&server, "r1")[2], "r1", 1),
        "Your agent run r2 (running) hasn't finished. Wait for it with wait_for_agents, or stop it with stop_agent, then report again."
    );
    gate.open.notify_one();
}

/// A subagent that is told to wrap up reports its progress whatever its own
/// subagents are doing. Here one of them is stuck and won't end.
#[tokio::test]
async fn a_subagent_being_stopped_may_report_with_unfinished_subagents() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("Level two."), wait()]),
        ("r2", vec![("probe", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let probe = crate::test_support::ProbeTool::new("probe", true);
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", probe.clone())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(probe.started.notified()).await;
    let request = StopRequest::operator("enough").wrap_up(Duration::from_secs(60));
    assert!(workflow.stop_subagent(workflow.subagents()[0].run_id, request));
    until(|| workflow.subagents()[0].state == SubagentState::Stopped).await;
    let records = workflow.subagents();
    assert_eq!(records[0].report.as_deref(), Some("r1 saved its progress"));
    // Its subagent was told to stop too, and still hasn't ended.
    assert_eq!(records[1].state, SubagentState::Running);
    assert!(records[1].stop_requested.is_some());
}

/// Two subagents each start a subagent and wait for it while only two runs
/// may run at once. The waiting ones hold no place, so the queued ones start.
#[tokio::test]
async fn subagents_waiting_for_their_own_subagents_hold_no_place() {
    let server = serve_tree(vec![
        ("lead", vec![start("Parent one."), start("Parent two.")]),
        ("r1", vec![start("Child."), wait()]),
        ("r2", vec![start("Child."), wait()]),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_depth(2).max_running(2),
    )
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate twice."))
        .await
        .unwrap();
    until(|| {
        let runs = workflow.subagents();
        runs.len() == 4 && runs.iter().all(|run| run.state == SubagentState::Finished)
    })
    .await;
    let depths = workflow
        .subagents()
        .into_iter()
        .map(|run| run.depth)
        .collect::<Vec<_>>();
    assert_eq!(depths, [1, 1, 2, 2]);
}

/// A run's list says what a subagent of it waits for.
#[tokio::test]
async fn the_list_says_what_a_waiting_subagent_waits_for() {
    let server = serve_tree(vec![
        ("lead", vec![start("Parent.")]),
        ("r1", vec![start("Child."), wait()]),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(gate.started.notified()).await;
    let ctx = workflow.context();
    // r1 may still be about to start waiting.
    until(|| {
        ctx.subagents
            .list(ctx, lead.agent_run_id)
            .contains("running (waiting for r2) for ")
    })
    .await;
    let r1 = workflow.subagents()[0].run_id;
    let listed = ctx.subagents.list(ctx, r1);
    assert!(
        listed.starts_with("r2 · agent `worker` · running for "),
        "{listed}"
    );
    gate.open.notify_one();
    until(|| workflow.active_subagents().is_empty()).await;
    assert!(
        ctx.subagents
            .list(ctx, lead.agent_run_id)
            .contains("finished")
    );
}

/// A subagent that waits for queued subagents, and wakes up when they are
/// done, doesn't make a run start twice.
#[tokio::test]
async fn a_waiting_subagent_that_wakes_makes_no_start_twice() {
    let server = serve_tree(vec![
        ("lead", vec![start("Parent.")]),
        (
            "r1",
            vec![
                start("First child."),
                start("Second child."),
                ("wait_for_agents", json!({ "all": true })),
            ],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_depth(2).max_running(1),
    )
    .build()
    .unwrap();
    let mut events = workflow.subscribe_subagents();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| {
        let runs = workflow.subagents();
        runs.len() == 3 && runs.iter().all(|run| run.state == SubagentState::Finished)
    })
    .await;
    let mut counts: BTreeMap<(String, &str), usize> = BTreeMap::new();
    while let Ok(event) = events.try_recv() {
        let kind = match event.kind {
            SubagentEventKind::Queued => "queued",
            SubagentEventKind::Started => "started",
            SubagentEventKind::Resumed => "resumed",
            _ => continue,
        };
        *counts.entry((event.handle, kind)).or_default() += 1;
    }
    let once =
        |handle: &str, kind: &'static str| counts.get(&(handle.to_string(), kind)) == Some(&1);
    for handle in ["r1", "r2", "r3"] {
        assert!(once(handle, "started"), "{handle}: {counts:?}");
        assert!(
            !counts.contains_key(&(handle.to_string(), "resumed")),
            "{counts:?}"
        );
    }
    // The two children waited for a place, one time each.
    assert!(once("r2", "queued") && once("r3", "queued"), "{counts:?}");
}

/// After a resume, a subagent whose lead won't run again isn't relaunched,
/// since it would report to a run that is gone. It is stopped, and its lead
/// is told. One whose lead is relaunched too goes on.
#[tokio::test]
async fn a_resume_stops_a_subagent_whose_lead_does_not_run_again() {
    for lead_state in ["failed", "interrupted"] {
        let server = serve_tree(vec![
            ("lead", vec![start("Level one.")]),
            ("r1", vec![start("Level two."), wait()]),
            ("r2", vec![("gate", json!({}))]),
        ])
        .await;
        let workspace = Workspace::new();
        let run_dir = workspace.0.join(".run");
        let gate = Arc::new(Gate::default());
        let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
            .agent_tool("worker", gate.clone())
            .persistence(PersistenceConfig::new(&run_dir))
            .build()
            .unwrap();
        within(workflow.run_agent("lead", "Delegate."))
            .await
            .unwrap();
        within(gate.started.notified()).await;
        workflow.stop(StopRequest::operator("closing"));
        until(|| workflow.active_subagents().is_empty()).await;
        workflow.checkpoint().await.unwrap();
        drop(workflow);

        // As if r1 had failed (or the process had ended with r2 running).
        let latest = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
        let mut saved: Value =
            serde_json::from_str(&std::fs::read_to_string(run_dir.join(latest.trim())).unwrap())
                .unwrap();
        saved["subagents"]["runs"][0]["state"] = json!(lead_state);
        saved["subagents"]["runs"][1]["state"] = json!("interrupted");
        // In a file of its own, which nothing else writes.
        let path = run_dir.join("edited-checkpoint.json");
        std::fs::write(&path, saved.to_string()).unwrap();

        let before = bodies_of(&server, "r2").len();
        let resumed = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
            .agent_tool("worker", Arc::new(Gate::default()))
            .persistence(PersistenceConfig::new(&run_dir))
            .resume_from(&path)
            .build()
            .unwrap();
        let records = resumed.subagents();
        let lead = records[0].run_id;
        let relaunched = resumed.resume_subagents();
        let records = resumed.subagents();
        if lead_state == "interrupted" {
            assert_eq!(relaunched, ["r1", "r2"]);
            assert!(
                records.iter().all(|run| run.state.is_active()),
                "{records:?}"
            );
            continue;
        }
        assert!(relaunched.is_empty(), "{relaunched:?}");
        assert_eq!(records[0].state, SubagentState::Failed);
        assert_eq!(records[1].state, SubagentState::Stopped);
        assert_eq!(
            records[1].error.as_deref(),
            Some("its lead r1 isn't running")
        );
        assert_eq!(bodies_of(&server, "r2").len(), before);
        let news = resumed.pending_news(lead);
        assert!(
            news.iter().any(|item| item.kind == MailKind::Stopped
                && item.heading
                    == "r2 (agent `worker`) stopped before it was relaunched: its lead r1 isn't running."),
            "{news:?}"
        );
        // Its lead's message restarts it where it stopped.
        let lead_run = records[1].caller_run_id;
        let sent = resumed
            .context()
            .subagents
            .message(resumed.context(), lead_run, "r2", "Carry on.")
            .unwrap();
        assert!(sent.contains("restarts where it stopped"), "{sent}");
        until(|| bodies_of(&server, "r2").len() > before).await;
    }
}

/// A run that hands its conversation over is told again which subagents it
/// started still work, and it hears from them as before.
#[tokio::test]
async fn a_handover_restates_the_subagents_that_still_work() {
    const REPORT: &str = "HANDOVER-REPORT: waiting for the count.";
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server = ResponsesServer::start(move |_, body| {
        let id = format!(
            "{}_{}",
            role(body),
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );
        if role(body) == "r1" {
            return Some(tool_call_turn(&id, "gate", json!({})));
        }
        Some(
            if body.contains("Your conversation is getting long") && !body.contains(REPORT) {
                tool_call_turn(&id, "handover_report", json!({ "report": REPORT }))
            } else if body.contains(REPORT) {
                tool_call_turn(&id, "complete_task", json!({ "summary": "lead done" }))
            } else if turn(body) == 0 {
                let (tool, arguments) = start("Count the files.");
                tool_call_turn(&id, tool, arguments)
            } else {
                // The conversation grows over the handover's size.
                tool_call_turn(&id, "list_agents", json!({}))
                    .replace("\"input_tokens\":10", "\"input_tokens\":5000")
            },
        )
    })
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let mut lead = agent_config("Lead");
    lead.handover = Some(arko_agents::HandoverConfig::at_input_tokens(1_000));
    let workflow = Workflow::builder()
        .name("nesting-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", lead)
        .agent("worker", agent_config("Worker"))
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("worker"),
        )
        .subagents(SubagentsConfig::default())
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let result = within(workflow.run_agent("lead", "Delegate the count."))
        .await
        .unwrap();
    assert_eq!(result.task().unwrap().handovers.len(), 1);
    let after = bodies_of(&server, "lead")
        .into_iter()
        .find(|body| body.contains(REPORT))
        .unwrap();
    assert!(
        after.contains("# Your workers\\nThese agent runs of yours still work"),
        "{after}"
    );
    assert!(
        after.contains("- r1 (agent `worker`, running for ")
            && after.contains("): Count the files."),
        "{after}"
    );
    // The report stays where it was, ahead of the workers.
    assert!(after.find(REPORT).unwrap() < after.find("# Your workers").unwrap());
}

/// After Claude Code compacts a conversation, the host's hook gives the
/// run's standing notes and its subagents again.
#[tokio::test]
async fn a_compaction_restates_the_subagents_that_still_work() {
    let server = serve_tree(vec![("r1", vec![("gate", json!({}))])]).await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default())
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let ctx = workflow.context();
    let task = arko_agents::Task::new(
        agent_config("Lead"),
        "Delegate.".to_string(),
        workspace.0.clone(),
        None,
    );
    let hook = crate::workflow::engine::WorkflowHandover {
        context: ctx.clone(),
        agent_name: "lead".to_string(),
        prompt_message_ids: Vec::new(),
        review: false,
    };
    use arko_agents::HandoverHook;
    assert_eq!(hook.on_compaction(&task), None);
    ctx.subagents
        .start(
            ctx,
            task.id,
            "lead",
            0,
            &workspace.0,
            "worker",
            "Count the files.",
            None,
            false,
            None,
            None,
            BTreeMap::new(),
        )
        .await
        .unwrap();
    within(gate.started.notified()).await;
    let restated = hook.on_compaction(&task).unwrap();
    assert!(restated.starts_with("# Your workers\n"), "{restated}");
    assert!(
        restated.contains("- r1 (agent `worker`, running for "),
        "{restated}"
    );
    // The seed of a handover has them too, as its whole context when the run
    // has no notes and no list of agents to restate.
    let seed = hook.on_handover(&task, "Report.");
    assert!(seed.context.unwrap().starts_with("# Your workers\n"));
}

/// A host's hook says more about a run, and its lead reads that with the
/// start's result and in the list.
#[tokio::test]
async fn the_hosts_hook_describes_a_subagent_to_its_lead() {
    struct Describing;
    impl SubagentHook for Describing {
        fn prepare(&self, _: &StartRequest, _: &mut AgentConfig) -> Result<(), String> {
            Ok(())
        }

        fn describe(&self, record: &SubagentRecord) -> Option<String> {
            Some(format!(
                "{} runs on a model of the host's choosing.\nIt was started at depth {}.",
                record.handle, record.depth
            ))
        }
    }
    let server = serve_tree(vec![(
        "lead",
        vec![start("Look around."), ("list_agents", json!({}))],
    )])
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().hook(Arc::new(Describing)),
    )
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let lead = bodies_of(&server, "lead");
    let started = result_of(&lead[1], "lead", 0);
    assert!(
        started.starts_with("Started r1 (agent `worker`).")
            && started.ends_with(
                "\nr1 runs on a model of the host's choosing.\nIt was started at depth 1."
            ),
        "{started}"
    );
    let listed = result_of(&lead[2], "lead", 1);
    assert!(
        listed.contains(
            "\n  task: Look around.\n  r1 runs on a model of the host's choosing.\n  It was started at depth 1."
        ),
        "{listed}"
    );
    // A hook that doesn't describe leaves both as they were, and so does a
    // closure.
    let record = workflow.subagents().remove(0);
    let closure = |_: &StartRequest, _: &mut AgentConfig| Ok(());
    assert_eq!(SubagentHook::describe(&closure, &record), None);
}

/// A hook that withholds the start tools from the subagents it names by
/// handle, and keeps who it was asked about.
struct Withholding {
    withheld: Vec<&'static str>,
    asked: parking_lot::Mutex<Vec<(String, Uuid)>>,
}

impl Withholding {
    fn new(withheld: Vec<&'static str>) -> Arc<Self> {
        Arc::new(Self {
            withheld,
            asked: parking_lot::Mutex::new(Vec::new()),
        })
    }
}

impl SubagentHook for Withholding {
    fn prepare(&self, _: &StartRequest, config: &mut AgentConfig) -> Result<(), String> {
        // The host lists the tools of its workers itself.
        config.tool_config.available_tool_names = vec!["gate".to_string()];
        Ok(())
    }

    fn offers_start_tools(&self, record: &SubagentRecord) -> bool {
        self.asked
            .lock()
            .push((record.handle.clone(), record.run_id));
        !self.withheld.contains(&record.handle.as_str())
    }
}

const START_TOOLS: [&str; 5] = [
    "start_agent",
    "list_agents",
    "stop_agent",
    "link_agents",
    "run_workflow_agent",
];

/// The `to` text `message_agent` has in a request.
fn message_agent_to(body: &str) -> String {
    let body: Value = serde_json::from_str(body).unwrap();
    let tool = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "message_agent")
        .unwrap();
    tool["parameters"]["properties"]["to"]["description"]
        .as_str()
        .unwrap()
        .to_string()
}

/// A hook can keep the tools that start runs from one subagent and leave
/// them to another, whatever tool list it sets. The one without them still
/// talks with its lead, and its `message_agent` is worded for that. The
/// hook is asked about subagents only, never the run the host drives.
#[tokio::test]
async fn a_hook_can_withhold_the_start_tools_from_one_subagent() {
    let server = serve_tree(vec![
        ("lead", vec![start("First."), start("Second.")]),
        ("r1", vec![("gate", json!({}))]),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let hook = Withholding::new(vec!["r1"]);
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_depth(3).hook(hook.clone()),
    )
    .agent_tool("worker", gate.clone())
    .build()
    .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate twice."))
        .await
        .unwrap();
    until(|| gate.calls.load(std::sync::atomic::Ordering::SeqCst) == 2).await;

    let first = &bodies_of(&server, "r1")[0];
    let tools = offered(first);
    for tool in START_TOOLS {
        assert!(!tools.contains(&tool.to_string()), "{tool}: {tools:?}");
    }
    for tool in ["gate", "complete_task", "message_agent", "wait_for_agents"] {
        assert!(tools.contains(&tool.to_string()), "{tool}: {tools:?}");
    }
    assert_eq!(
        message_agent_to(first),
        "`lead` for the agent that started you, or the run id of another agent run your lead linked you with."
    );
    let second = &bodies_of(&server, "r2")[0];
    let tools = offered(second);
    for tool in START_TOOLS {
        assert!(tools.contains(&tool.to_string()), "{tool}: {tools:?}");
    }
    assert!(message_agent_to(second).starts_with("The run id of one of your agent runs"));
    // The run the host drives keeps them, and the hook was asked about
    // subagents only.
    let lead_tools = offered(&bodies_of(&server, "lead")[0]);
    assert!(
        lead_tools.contains(&"start_agent".to_string()),
        "{lead_tools:?}"
    );
    let records = workflow.subagents();
    for (handle, run_id) in hook.asked.lock().iter() {
        assert!(
            records
                .iter()
                .any(|run| run.handle == *handle && run.run_id == *run_id),
            "asked about {handle}"
        );
        assert_ne!(*run_id, lead.agent_run_id);
    }
    assert!(!hook.asked.lock().is_empty());
}

/// The answer of the hook is the same when the subagent is continued with a
/// message and when it resumes from a checkpoint.
#[tokio::test]
async fn the_hooks_answer_stays_for_a_continued_and_a_resumed_subagent() {
    // Continued: r1 finishes, and a message from the lead continues it.
    let server = serve_tree(vec![("lead", vec![start("First.")])]).await;
    let workspace = Workspace::new();
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default()
            .max_depth(3)
            .hook(Withholding::new(vec!["r1"])),
    )
    .build()
    .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| workflow.subagents()[0].state == SubagentState::Finished).await;
    let ctx = workflow.context();
    ctx.subagents
        .message(ctx, lead.agent_run_id, "r1", "Once more.")
        .unwrap();
    until(|| bodies_of(&server, "r1").len() >= 2).await;
    let requests = bodies_of(&server, "r1");
    assert!(requests[1].contains("Once more."));
    let tools = offered(&requests[0]);
    assert!(!tools.contains(&"start_agent".to_string()), "{tools:?}");
    assert_eq!(offered(&requests[1]), tools);

    // Resumed: r1 waits at its gate when the workflow stops, and the
    // resumed workflow relaunches it.
    let server = serve_tree(vec![
        ("lead", vec![start("First.")]),
        ("r1", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let run_dir = workspace.0.join(".run");
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default()
            .max_depth(3)
            .hook(Withholding::new(vec!["r1"])),
    )
    .agent_tool("worker", gate.clone())
    .persistence(PersistenceConfig::new(&run_dir))
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(gate.started.notified()).await;
    workflow.stop(StopRequest::operator("closing"));
    until(|| workflow.active_subagents().is_empty()).await;
    workflow.checkpoint().await.unwrap();
    drop(workflow);
    let before = bodies_of(&server, "r1").len();
    let latest = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
    let gate = Arc::new(Gate::default());
    let resumed = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default()
            .max_depth(3)
            .hook(Withholding::new(vec!["r1"])),
    )
    .agent_tool("worker", gate.clone())
    .persistence(PersistenceConfig::new(&run_dir))
    .resume_from(run_dir.join(latest.trim()))
    .build()
    .unwrap();
    assert_eq!(resumed.resume_subagents(), ["r1"]);
    until(|| bodies_of(&server, "r1").len() > before).await;
    let requests = bodies_of(&server, "r1");
    let tools = offered(&requests[0]);
    assert!(!tools.contains(&"start_agent".to_string()), "{tools:?}");
    assert_eq!(offered(requests.last().unwrap()), tools);
}

/// A start by a subagent the hook withholds the tools from is refused, in
/// words that name no setting of the host.
#[tokio::test]
async fn a_start_by_a_subagent_without_the_start_tools_is_refused() {
    let server = serve_tree(vec![
        ("lead", vec![start("First.")]),
        ("r1", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default()
            .max_depth(3)
            .hook(Withholding::new(vec!["r1"])),
    )
    .agent_tool("worker", gate.clone())
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(gate.started.notified()).await;
    let ctx = workflow.context();
    let r1 = workflow.subagents()[0].run_id;
    let refused = ctx
        .subagents
        .start(
            ctx,
            r1,
            "worker",
            1,
            &workspace.0,
            "worker",
            "Take this.",
            None,
            false,
            None,
            None,
            BTreeMap::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        refused,
        "You can't start agent runs. Do the work yourself, or tell your lead what you need with message_agent."
    );
    assert_eq!(workflow.subagents().len(), 1);
}

/// A model that answers every refusal with another `complete_task` is
/// refused five times. The sixth is accepted, its subagents are stopped with
/// the run, and its lead is told they were still working.
#[tokio::test]
async fn a_model_that_keeps_reporting_is_refused_five_times_and_then_its_subagents_stop() {
    let mut steps = vec![start("Level two.")];
    for number in 1..=8 {
        steps.push(complete(&format!("Report {number}.")));
    }
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", steps),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| workflow.subagents()[0].state == SubagentState::Finished).await;
    until(|| workflow.active_subagents().is_empty()).await;

    // The first request, the answer to the start, and five refusals.
    let requests = bodies_of(&server, "r1");
    assert_eq!(requests.len(), 7);
    let refusal = |number: usize| result_of(&requests[number + 1], "r1", number);
    for number in 1..=2 {
        assert!(
            refusal(number).ends_with("then report again."),
            "{}",
            refusal(number)
        );
    }
    for number in 3..=4 {
        assert!(
            refusal(number).ends_with(&format!(
                "This is refusal {number} of 5. After refusal 5, your next complete_task is accepted and the agent runs that still work are stopped."
            )),
            "{}",
            refusal(number)
        );
    }
    assert!(
        refusal(5).ends_with(
            "That was the last refusal. Your next complete_task is accepted, and the agent runs that still work are stopped."
        ),
        "{}",
        refusal(5)
    );
    let records = workflow.subagents();
    assert_eq!(records[0].report.as_deref(), Some("Report 6."));
    assert_eq!(records[1].state, SubagentState::Stopped);
    assert_eq!(records[1].error.as_deref(), Some("its lead r1 ended"));
    let news = workflow.pending_news(lead.agent_run_id);
    let reports: Vec<&str> = news.iter().map(|item| item.body.as_str()).collect();
    assert_eq!(
        reports,
        ["Report 6.\n\nIt finished while r2 was still working, so r2 was stopped."]
    );
}

/// Refusals are counted in a row. Another tool call in between starts the
/// count again.
#[tokio::test]
async fn another_tool_call_between_refusals_starts_the_count_again() {
    let mut steps = vec![start("Level two.")];
    for _ in 0..4 {
        steps.push(complete("Done?"));
    }
    steps.push(("list_agents", json!({})));
    for _ in 0..4 {
        steps.push(complete("Done?"));
    }
    steps.push(("wait_for_agents", json!({ "all": true })));
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", steps),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    // The ninth request answers r1's last refusal, and it waits.
    until(|| bodies_of(&server, "r1").len() >= 11).await;
    assert_eq!(workflow.subagents()[0].state, SubagentState::Running);
    gate.open.notify_one();
    until(|| workflow.active_subagents().is_empty()).await;
    let records = workflow.subagents();
    assert_eq!(records[0].report.as_deref(), Some("r1 done"));
    assert_eq!(records[1].state, SubagentState::Finished);
}

/// `wait_for_agents` named a run that is neither the caller's own nor
/// linked to it. It is refused at once, and not left to wait for a report
/// that would never come to it.
#[tokio::test]
async fn waiting_for_a_run_that_is_not_the_callers_is_refused_at_once() {
    let server = serve_tree(vec![
        ("lead", vec![start("First."), start("Second.")]),
        (
            "r1",
            vec![(
                "wait_for_agents",
                json!({ "runs": "r2", "timeout_seconds": 600 }),
            )],
        ),
        ("r2", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate twice."))
        .await
        .unwrap();
    until(|| bodies_of(&server, "r1").len() >= 2).await;
    let refused = result_of(&bodies_of(&server, "r1")[1], "r1", 0);
    assert!(
        refused.contains("there is no agent run `r2` of yours"),
        "{refused}"
    );
    gate.open.notify_one();
}

/// A peer the lead linked the caller with can be waited for: its message
/// ends the wait.
#[tokio::test]
async fn waiting_for_a_linked_peer_ends_with_its_message() {
    let server = serve_tree(vec![
        (
            "lead",
            vec![
                (
                    "start_agent",
                    json!({ "agent": "worker", "task": "First.", "peers": "all" }),
                ),
                start("Second."),
            ],
        ),
        ("r1", vec![("wait_for_agents", json!({ "runs": "r2" }))]),
        (
            "r2",
            vec![(
                "message_agent",
                json!({ "to": "r1", "message": "Here is what I found." }),
            )],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate twice."))
        .await
        .unwrap();
    until(|| bodies_of(&server, "r1").len() >= 2).await;
    let heard = result_of(&bodies_of(&server, "r1")[1], "r1", 0);
    assert!(heard.contains("Here is what I found."), "{heard}");
}

/// A hook whose `describe` panics describes nothing. The start it was asked
/// about still returns, and the run it started is still its lead's.
#[tokio::test]
async fn a_describe_that_panics_leaves_the_start_and_the_list_alone() {
    struct Panicking;
    impl SubagentHook for Panicking {
        fn prepare(&self, _: &StartRequest, _: &mut AgentConfig) -> Result<(), String> {
            Ok(())
        }

        fn describe(&self, _: &SubagentRecord) -> Option<String> {
            panic!("a bug in the host's describe");
        }
    }
    let server = serve_tree(vec![(
        "lead",
        vec![start("Look around."), ("list_agents", json!({}))],
    )])
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().hook(Arc::new(Panicking)),
    )
    .build()
    .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    assert_eq!(lead.status, AgentRunStatus::Completed);
    let requests = bodies_of(&server, "lead");
    let started = result_of(&requests[1], "lead", 0);
    assert!(
        started.starts_with("Started r1 (agent `worker`)."),
        "{started}"
    );
    assert!(!started.contains('\n'), "{started}");
    let listed = result_of(&requests[2], "lead", 1);
    assert!(listed.starts_with("r1 · agent `worker`"), "{listed}");
    until(|| workflow.active_subagents().is_empty()).await;
}

/// A workflow stop with a wrap-up, when a run ends first and its subagent is
/// stuck in a tool that ignores stops, leaves the subagent's stop the
/// workflow's own, not the one a run that ends makes for its subagents.
#[tokio::test]
async fn a_workflow_wrap_up_stays_the_stop_of_a_subagent_that_outlasts_its_lead() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("Level two."), wait()]),
        ("r2", vec![("probe", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let probe = crate::test_support::ProbeTool::new("probe", true);
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", probe.clone())
        .cancel_grace_period(Duration::from_millis(100))
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(probe.started.notified()).await;
    assert!(workflow.stop(StopRequest::operator("closing").wrap_up(Duration::from_secs(300))));
    // r1 finishes its wrap-up, and r2 can't.
    until(|| workflow.subagents()[0].state == SubagentState::Stopped).await;
    assert_eq!(workflow.subagents()[1].state, SubagentState::Running);
    // The wrap-up ends now, and r2 is aborted.
    assert!(workflow.stop(StopRequest::operator("closing for good")));
    until(|| workflow.active_subagents().is_empty()).await;
    let stop = workflow
        .agent_result(workflow.subagents()[1].run_id)
        .unwrap()
        .stopped_by
        .unwrap();
    assert_eq!((stop.reason.as_str(), stop.intent), ("closing", None));
    assert_eq!(workflow.subagents()[1].error.as_deref(), Some("closing"));
}

/// A worker whose text reply ends its task isn't offered `complete_task`, so
/// it can end with subagents that still work. Its subagents are stopped, and
/// the news of that doesn't start its finished run again.
#[tokio::test]
async fn the_news_of_subagents_stopped_for_a_finished_lead_does_not_restart_it() {
    // r2 waits for a place (one run at a time), or runs and waits at a gate.
    for max_running in [1, 2] {
        let server = ResponsesServer::start(move |_, body| {
            let role = role(body);
            let id = format!("{role}_{}", turn(body));
            if body.contains("Your run is being stopped") {
                return Some(crate::test_support::text_turn(
                    &id,
                    &format!("{role} saved"),
                ));
            }
            Some(match (role.as_str(), turn(body)) {
                ("lead", 0) => {
                    let (tool, arguments) = start("Level one.");
                    tool_call_turn(&id, tool, arguments)
                }
                ("r1", 0) => {
                    let (tool, arguments) = start("Level two.");
                    tool_call_turn(&id, tool, arguments)
                }
                ("r1", _) => crate::test_support::text_turn(&id, "r1 is finished"),
                ("r2", _) => tool_call_turn(&id, "gate", json!({})),
                _ => tool_call_turn(&id, "complete_task", json!({ "summary": "lead done" })),
            })
        })
        .await;
        let workspace = Workspace::new();
        let gate = Arc::new(Gate::default());
        let mut worker = agent_config("Worker");
        worker.tool_config.text_reply_ends_task = true;
        let workflow = Workflow::builder()
            .name("nesting-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("lead", agent_config("Lead"))
            .agent("worker", worker)
            .workflow_services(
                WorkflowServicesConfig::default()
                    .allow_caller("lead")
                    .allow_caller("worker")
                    .allow_target("worker"),
            )
            .subagents(
                SubagentsConfig::default()
                    .max_depth(2)
                    .max_running(max_running),
            )
            .agent_tool("worker", gate.clone())
            .build()
            .unwrap();
        let mut events = workflow.subscribe_subagents();
        within(workflow.run_agent("lead", "Delegate."))
            .await
            .unwrap();
        // r1 ends at once with r2 queued (one place), or after r2 started.
        until(|| {
            workflow
                .subagents()
                .get(1)
                .is_some_and(|run| run.state == SubagentState::Stopped)
        })
        .await;
        let records = workflow.subagents();
        assert_eq!(records[0].state, SubagentState::Finished, "{max_running}");
        assert_eq!(records[1].error.as_deref(), Some("its lead r1 ended"));
        // The stop ran to its end already (the runtime of the test is one
        // thread), and r1 wasn't relaunched to read it.
        while let Ok(event) = events.try_recv() {
            assert!(
                !matches!(event.kind, SubagentEventKind::Resumed),
                "{} resumed",
                event.handle
            );
        }
        assert_eq!(records[0].state, SubagentState::Finished);
    }
}

/// A subagent that woke from its wait holds a place again. With one place, a
/// start by its lead is queued behind it.
#[tokio::test]
async fn a_subagent_that_woke_from_its_wait_holds_a_place_again() {
    let server = serve_tree(vec![
        ("lead", vec![start("Parent.")]),
        ("r1", vec![start("Child."), wait(), ("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_depth(2).max_running(1),
    )
    .agent_tool("worker", gate.clone())
    .build()
    .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    // r1 woke with its child's report and works at the gate.
    within(gate.started.notified()).await;
    let ctx = workflow.context();
    let started = ctx
        .subagents
        .start(
            ctx,
            lead.agent_run_id,
            "lead",
            0,
            &workspace.0,
            "worker",
            "Another.",
            None,
            false,
            None,
            None,
            BTreeMap::new(),
        )
        .await
        .unwrap();
    assert!(
        started.starts_with("Queued r3 (agent `worker`)"),
        "{started}"
    );
    gate.open.notify_one();
}

/// Starts a `worker` for run `caller` at `depth` from the board, as
/// `start_agent` does, and returns its result.
async fn begin(
    ctx: &Arc<crate::context::SharedContext>,
    caller: Uuid,
    depth: u32,
    directory: &std::path::Path,
    task: &str,
) -> String {
    ctx.subagents
        .start(
            ctx,
            caller,
            "lead",
            depth,
            directory,
            "worker",
            task,
            None,
            false,
            None,
            None,
            BTreeMap::new(),
        )
        .await
        .unwrap()
}

/// A wait that is cancelled gives its place back to nobody: the run holds
/// its place again.
#[tokio::test]
async fn a_cancelled_wait_leaves_no_waiter_behind() {
    // r1 and r2 never get an answer, so they stay running. r3 finishes.
    let server = ResponsesServer::start(|_, body| match role(body).as_str() {
        "r1" | "r2" => None,
        other => Some(tool_call_turn(
            &format!("{other}_0"),
            "complete_task",
            json!({ "summary": format!("{other} done") }),
        )),
    })
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_depth(2).max_running(2),
    )
    .build()
    .unwrap();
    let ctx = workflow.context();
    let lead = Uuid::new_v4();
    begin(ctx, lead, 0, &workspace.0, "First.").await;
    let r1 = workflow.subagents()[0].run_id;
    begin(ctx, r1, 1, &workspace.0, "Child.").await;

    // r1 waits for its child, and holds no place meanwhile.
    let tool = WaitForAgentsTool::new(ctx.clone(), "worker".into(), Arc::new(server.client()));
    let waiting = tokio::spawn({
        let directory = workspace.0.clone();
        async move {
            let task_context = TaskContext::new(r1, "worker", "", &directory);
            tool.execute(&task_context, HashMap::new()).await
        }
    });
    until(|| {
        ctx.subagents
            .list(ctx, lead)
            .contains("running (waiting for r2)")
    })
    .await;
    let started = begin(ctx, lead, 0, &workspace.0, "Third.").await;
    assert!(started.starts_with("Started r3"), "{started}");
    until(|| workflow.subagents()[2].state == SubagentState::Finished).await;

    // The wait is cancelled, and r1 holds its place again, next to r2.
    waiting.abort();
    let _ = waiting.await;
    let started = begin(ctx, lead, 0, &workspace.0, "Fourth.").await;
    assert!(started.starts_with("Queued r4"), "{started}");
}

/// Stopping the run the host drives, as when its user interrupts it, leaves
/// the subagents it started working.
#[tokio::test]
async fn stopping_the_run_the_host_drives_leaves_its_subagents_working() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one."), wait()]),
        ("r1", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = Arc::new(
        nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
            .agent_tool("worker", gate.clone())
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Delegate.").await }
    });
    within(gate.started.notified()).await;
    let lead = workflow.subagents()[0].caller_run_id;
    assert!(workflow.stop_run(lead, StopRequest::operator("esc"), RunStopIntent::Relaunch));
    let _ = within(run).await.unwrap();
    // It is still working: it finishes when its gate opens.
    assert_eq!(workflow.subagents()[0].state, SubagentState::Running);
    gate.open.notify_one();
    until(|| workflow.subagents()[0].state == SubagentState::Finished).await;
    assert_eq!(workflow.subagents()[0].report.as_deref(), Some("r1 done"));
}

/// A server whose model reply to the second request of `held` waits until
/// the test lets it go, so that the test can stop a run while the run's reply
/// is on its way. The reply is a `complete_task` call.
async fn serve_holding_a_report(
    held: &'static str,
    reached: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
) -> ResponsesServer {
    use crate::test_support::Reply;
    ResponsesServer::start_async(move |_, body| {
        let (role, turn) = (role(&body), turn(&body));
        let id = format!("{role}_{turn}");
        let notice =
            body.contains("Your run is being stopped") || body.contains("The workflow is stopping");
        let (reached, release) = (reached.clone(), release.clone());
        Box::pin(async move {
            let stream = match (role.as_str(), turn) {
                ("lead", 0) => {
                    let (tool, arguments) = start("Level one.");
                    tool_call_turn(&id, tool, arguments)
                }
                (role, 0) if role == held => {
                    let (tool, arguments) = start("Level two.");
                    tool_call_turn(&id, tool, arguments)
                }
                (role, 1) if role == held && !notice => {
                    reached.notify_one();
                    release.notified().await;
                    tool_call_turn(&id, "complete_task", json!({ "summary": "Report." }))
                }
                ("r2", _) if !notice => tool_call_turn(&id, "gate", json!({})),
                _ => tool_call_turn(
                    &id,
                    "complete_task",
                    json!({ "summary": "saved its progress" }),
                ),
            };
            Some(Reply::Stream(stream))
        })
    })
    .await
}

/// A run that is stopped while its report is on its way is let through even
/// before it has been told, whatever its subagents do. The stop is the run's
/// own, or the workflow's wrap-up.
#[tokio::test]
async fn a_report_on_its_way_is_let_through_when_the_run_or_the_workflow_is_stopping() {
    for workflow_stop in [false, true] {
        let reached = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let server = serve_holding_a_report("r1", reached.clone(), release.clone()).await;
        let workspace = Workspace::new();
        let gate = Arc::new(Gate::default());
        let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
            .agent_tool("worker", gate.clone())
            .build()
            .unwrap();
        within(workflow.run_agent("lead", "Delegate."))
            .await
            .unwrap();
        within(gate.started.notified()).await;
        within(reached.notified()).await;
        let request = StopRequest::operator("enough").wrap_up(Duration::from_secs(300));
        if workflow_stop {
            assert!(workflow.stop(request));
        } else {
            assert!(workflow.stop_subagent(workflow.subagents()[0].run_id, request));
        }
        // r1's `complete_task` arrives, with r2 still working.
        release.notify_one();
        until(|| workflow.subagents()[0].state == SubagentState::Stopped).await;
        // It was not refused: no request followed the report.
        let requests = bodies_of(&server, "r1");
        assert_eq!(requests.len(), 2, "workflow stop {workflow_stop}");
        assert_eq!(
            workflow.subagents()[0].report.as_deref(),
            Some("Report."),
            "workflow stop {workflow_stop}"
        );
    }
}

/// Past twenty, the subagents a handover restates are counted, not listed.
#[tokio::test]
async fn a_handover_lists_twenty_subagents_and_counts_the_rest() {
    let server = ResponsesServer::start(|_, _| None).await;
    let workspace = Workspace::new();
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().unlimited())
        .build()
        .unwrap();
    let ctx = workflow.context();
    let lead = Uuid::new_v4();
    for number in 1..=25 {
        ctx.subagents
            .start(
                ctx,
                lead,
                "lead",
                0,
                &workspace.0,
                "worker",
                &format!("Task number {number}."),
                None,
                false,
                None,
                None,
                BTreeMap::new(),
            )
            .await
            .unwrap();
    }
    let section = ctx.subagents.workers_section(lead).unwrap();
    let listed = section
        .lines()
        .filter(|line| line.starts_with("- r"))
        .count();
    assert_eq!(listed, 20, "{section}");
    assert!(
        section.contains("- r20 (agent `worker`, running for "),
        "{section}"
    );
    assert!(!section.contains("- r21 "), "{section}");
    assert!(
        section.ends_with("- (5 more, list_agents shows all of them.)"),
        "{section}"
    );
}

/// `stop_agent` names every run that stops with the one it stops.
#[tokio::test]
async fn stop_agent_names_all_the_runs_that_stop_with_a_subagent() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("First."), start("Second."), wait()]),
        ("r2", vec![("gate", json!({}))]),
        ("r3", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| gate.calls.load(std::sync::atomic::Ordering::SeqCst) == 2).await;
    let ctx = workflow.context();
    let stopping = ctx
        .subagents
        .stop(ctx, lead.agent_run_id, "r1", "", false)
        .await
        .unwrap();
    assert!(
        stopping.contains("Its own agent runs r2 and r3 are stopping with it."),
        "{stopping}"
    );
    until(|| workflow.active_subagents().is_empty()).await;
}

/// A run's subagent that waits for the run's answer is named as such when
/// the run tries to report.
#[tokio::test]
async fn a_subagent_asking_a_question_is_named_when_its_lead_tries_to_report() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        (
            "r1",
            vec![start("Level two."), ("hold", json!({})), complete("Early.")],
        ),
        (
            "r2",
            vec![("ask_user", json!({ "questions": "May I go on?" }))],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let hold = Barrier::new("hold");
    let mut worker = agent_config("Worker");
    worker
        .tool_config
        .available_tool_names
        .push("ask_user".to_string());
    let workflow = Workflow::builder()
        .name("nesting-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Lead"))
        .agent("worker", worker)
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_caller("worker")
                .allow_target("worker"),
        )
        .subagents(SubagentsConfig::default().max_depth(2))
        .agent_tool("worker", hold.clone())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| workflow.subagents().get(1).is_some_and(|run| run.asking)).await;
    hold.open.notify_one();
    until(|| bodies_of(&server, "r1").len() >= 4).await;
    let refused = result_of(&bodies_of(&server, "r1")[3], "r1", 2);
    assert!(
        refused.starts_with("Your agent run r2 (asking you a question) hasn't finished."),
        "{refused}"
    );
}

/// A workflow stopped without a wrap-up, with a tree of runs working, is
/// relaunched whole by a resume.
#[tokio::test]
async fn a_resume_relaunches_a_whole_tree_that_a_workflow_stop_cut_off() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("Level two."), wait()]),
        ("r2", vec![start("Level three."), wait()]),
        ("r3", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let run_dir = workspace.0.join(".run");
    let gate = Arc::new(Gate::default());
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(3))
        .agent_tool("worker", gate.clone())
        .persistence(PersistenceConfig::new(&run_dir))
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    within(gate.started.notified()).await;
    workflow.stop(StopRequest::operator("closing"));
    until(|| workflow.active_subagents().is_empty()).await;
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let latest = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
    let resumed = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(3))
        .agent_tool("worker", Arc::new(Gate::default()))
        .persistence(PersistenceConfig::new(&run_dir))
        .resume_from(run_dir.join(latest.trim()))
        .build()
        .unwrap();
    assert_eq!(resumed.resume_subagents(), ["r1", "r2", "r3"]);
    assert!(
        resumed.subagents().iter().all(|run| run.state.is_active()),
        "{:?}",
        resumed.subagents()
    );
}

/// The host's message to a nested subagent puts a copy in the mailbox of the
/// run that started it, under the copy heading. An empty heading sends no
/// copy, and the subagent gets the message either way.
#[tokio::test]
async fn an_empty_copy_heading_sends_no_copy_of_the_hosts_message() {
    let server = serve_tree(vec![
        ("lead", vec![start("Level one.")]),
        ("r1", vec![start("Level two."), wait()]),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = nested_builder(&server, &workspace, SubagentsConfig::default().max_depth(2))
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let records = || workflow.subagents();
    until(|| {
        records().len() == 2
            && records()
                .iter()
                .all(|run| run.state == SubagentState::Finished)
    })
    .await;
    let starter = records()
        .iter()
        .find(|run| run.handle == "r1")
        .unwrap()
        .run_id;
    let copies = || {
        workflow
            .pending_news(starter)
            .into_iter()
            .filter(|item| item.kind == MailKind::Copy)
            .map(|item| item.heading)
            .collect::<Vec<_>>()
    };
    assert!(copies().is_empty());

    let sent = workflow
        .message_subagent("r2", "First.", "The user sent you a message:", "")
        .unwrap();
    assert!(sent.contains("continues its task"), "{sent}");
    assert!(copies().is_empty(), "{:?}", copies());
    until(|| bodies_of(&server, "r2").len() == 2).await;
    assert!(bodies_of(&server, "r2")[1].contains("First."));
    until(|| {
        records()
            .iter()
            .all(|run| run.state == SubagentState::Finished)
    })
    .await;

    workflow
        .message_subagent(
            "r2",
            "Second.",
            "The user sent you a message:",
            "The user messaged r2 (a copy for you):",
        )
        .unwrap();
    assert_eq!(copies(), ["The user messaged r2 (a copy for you):"]);
    until(|| bodies_of(&server, "r2").len() == 3).await;
    assert!(bodies_of(&server, "r2")[2].contains("Second."));
}
