//! Background subagents end to end, against the local model server: starts,
//! reports at turn boundaries and through waits, messages both ways, the
//! queue, stops, isolation, per-run workspaces, questions, receipts, and
//! checkpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arko_agents::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::*;
use crate::persistence::PersistenceConfig;
use crate::services::WorkflowServicesConfig;
use crate::test_support::{
    ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;
use crate::{Workflow, WorkflowBuilder};

/// A model turn: the tool it calls and its arguments.
pub(super) type Step = (&'static str, Value);

/// Which conversation a request belongs to: a subagent's (`r1`, `r2`, ...)
/// or the lead's.
pub(super) fn role(body: &str) -> String {
    for n in 1..10 {
        if body.contains(&format!("You are agent run r{n},")) {
            return format!("r{n}");
        }
    }
    "lead".to_string()
}

/// How many tool results a request sends: the turn its conversation is at.
pub(super) fn turn(body: &str) -> usize {
    body.matches("\"function_call_output\"").count()
}

/// A server that answers each conversation from its script, turn by turn,
/// and completes it (`<role> done`) once the script is over.
pub(super) async fn serve(scripts: Vec<(&'static str, Vec<Step>)>) -> ResponsesServer {
    let scripts: HashMap<String, Vec<Step>> = scripts
        .into_iter()
        .map(|(role, steps)| (role.to_string(), steps))
        .collect();
    ResponsesServer::start(move |_, body| {
        let role = role(body);
        let turn = turn(body);
        let id = format!("{role}_{turn}");
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

pub(super) fn builder(
    server: &ResponsesServer,
    workspace: &Workspace,
    config: SubagentsConfig,
) -> WorkflowBuilder {
    builder_with(server, workspace, config, agent_config("Worker"))
}

/// [`builder`] with agent `worker` configured as `worker`.
pub(super) fn builder_with(
    server: &ResponsesServer,
    workspace: &Workspace,
    config: SubagentsConfig,
    worker: AgentConfig,
) -> WorkflowBuilder {
    Workflow::builder()
        .name("subagents-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Lead"))
        .agent("worker", worker)
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("worker"),
        )
        .subagents(config)
}

pub(super) async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .expect("in time")
}

/// Waits until `done` holds.
pub(super) async fn until(mut done: impl FnMut() -> bool) {
    within(async {
        while !done() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
}

/// The requests of conversation `role`, in order.
pub(super) fn bodies_of(server: &ResponsesServer, of: &str) -> Vec<String> {
    server
        .bodies()
        .into_iter()
        .filter(|body| role(body) == of)
        .collect()
}

/// The tool result a conversation's request carries for its call at turn
/// `turn` (whose id is `<role>_<turn>`).
pub(super) fn result_of(body: &str, of: &str, turn: usize) -> String {
    let body: Value = serde_json::from_str(body).unwrap();
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| {
            item["type"] == "function_call_output" && item["call_id"] == format!("call_{of}_{turn}")
        })
        .and_then(|item| item["output"].as_str())
        .unwrap_or_default()
        .to_string()
}

/// The names of the tools a request offers.
pub(super) fn offered(body: &str) -> Vec<String> {
    let body: Value = serde_json::from_str(body).unwrap();
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// A tool that waits until the test opens it, or the task's stop signal
/// fires.
#[derive(Debug, Default)]
pub(super) struct Gate {
    pub(super) started: tokio::sync::Notify,
    pub(super) open: tokio::sync::Notify,
    pub(super) calls: AtomicUsize,
}

#[async_trait::async_trait]
impl Tool for Gate {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "gate".to_string(),
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        tokio::select! {
            _ = self.open.notified() => Ok("opened".to_string()),
            _ = context.stop.stopping() => Ok("stopped".to_string()),
        }
    }
}

/// A tool that answers with the directory it was called in.
#[derive(Debug)]
struct Where;

#[async_trait::async_trait]
impl Tool for Where {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "where".to_string(),
            description: "Says where.".to_string(),
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
        Ok(format!(
            "here: {}",
            context.current_working_directory.display()
        ))
    }
}

pub(super) fn start(task: &str) -> Step {
    ("start_agent", json!({ "agent": "worker", "task": task }))
}

pub(super) fn wait() -> Step {
    ("wait_for_agents", json!({}))
}

pub(super) fn complete(summary: &str) -> Step {
    ("complete_task", json!({ "summary": summary }))
}

/// `start_agent` returns a run id at once while the subagent works, and
/// `wait_for_agents` returns its report itself. The report's costs are in
/// the workflow's ledger, and the subagent isn't a host run to recover.
#[tokio::test]
async fn a_start_returns_at_once_and_a_wait_returns_the_report() {
    let server = serve(vec![
        ("lead", vec![start("Count the files."), wait()]),
        ("r1", vec![complete("There are 3 files.")]),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .build()
        .unwrap();
    let mut events = workflow.subscribe_subagents();
    let result = within(workflow.run_agent("lead", "Delegate the count."))
        .await
        .unwrap();
    assert_eq!(result.summary, "lead done");

    let lead = bodies_of(&server, "lead");
    let started = result_of(&lead[1], "lead", 0);
    assert!(
        started.starts_with("Started r1 (agent `worker`)."),
        "{started}"
    );
    let waited = result_of(&lead[2], "lead", 1);
    assert!(
        waited.contains("[mail 1] r1 (agent `worker`) finished. Its report:\n\nThere are 3 files."),
        "{waited}"
    );
    let worker = &bodies_of(&server, "r1")[0];
    assert!(worker.contains("Count the files."));
    assert!(worker.contains("started by agent `lead` (your lead)"));
    assert!(worker.contains("Keep it under about 16 KB, and put long detail in a file"));
    // The lead may start runs and use the blocking tool; the subagent
    // talks with its lead only.
    let lead_tools = offered(&lead[0]);
    for tool in [
        "start_agent",
        "message_agent",
        "wait_for_agents",
        "list_agents",
        "stop_agent",
        "link_agents",
        "run_workflow_agent",
    ] {
        assert!(
            lead_tools.contains(&tool.to_string()),
            "{tool}: {lead_tools:?}"
        );
    }
    let worker_tools = offered(worker);
    assert!(worker_tools.contains(&"message_agent".to_string()));
    assert!(worker_tools.contains(&"wait_for_agents".to_string()));
    for tool in ["start_agent", "run_workflow_agent", "stop_agent"] {
        assert!(
            !worker_tools.contains(&tool.to_string()),
            "{tool}: {worker_tools:?}"
        );
    }

    let records = workflow.subagents();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].state, SubagentState::Finished);
    assert_eq!(records[0].report.as_deref(), Some("There are 3 files."));
    assert_eq!(records[0].caller_run_id, result.agent_run_id);
    let ledger = workflow.cost_ledger();
    assert!(
        ledger.entries.iter().any(|entry| matches!(
            &entry.source,
            crate::budget::CostSource::Agent { agent_name, .. } if agent_name == "worker"
        )),
        "{:?}",
        ledger.entries
    );
    assert!(workflow.interrupted_runs().is_empty());
    let mut kinds = Vec::new();
    while let Ok(event) = events.try_recv() {
        kinds.push(event.kind);
    }
    assert!(matches!(kinds[0], SubagentEventKind::Started));
    assert!(kinds.iter().any(|kind| matches!(kind, SubagentEventKind::Finished { summary } if summary == "There are 3 files.")));
    assert!(kinds.iter().any(
        |kind| matches!(kind, SubagentEventKind::Delivered { mail_ids, .. } if mail_ids == &[1])
    ));
}

/// A report reaches a lead that is busy at its next turn boundary, in its
/// context update, which `ContextInjected` reports with the item's id.
#[tokio::test]
async fn a_report_reaches_a_running_lead_at_its_next_step() {
    let server = serve(vec![
        ("lead", vec![start("Look around."), ("gate", json!({}))]),
        ("r1", vec![complete("Looked.")]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = Arc::new(
        builder(&server, &workspace, SubagentsConfig::default())
            .agent_tool("lead", gate.clone())
            .build()
            .unwrap(),
    );
    let mut agent_events = workflow.subscribe_events();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Delegate.").await }
    });
    within(gate.started.notified()).await;
    until(|| {
        workflow
            .subagents()
            .first()
            .is_some_and(|run| run.state == SubagentState::Finished)
    })
    .await;
    gate.open.notify_one();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "lead done");
    let lead = bodies_of(&server, "lead");
    let last = lead.last().unwrap();
    assert!(
        last.contains("# Workflow context update")
            && last.contains("[mail 1] r1 (agent `worker`) finished. Its report:\\n  Looked."),
        "{last}"
    );
    let mut receipts = Vec::new();
    while let Ok(event) = agent_events.try_recv() {
        if let arko_agents::AgentEvent::ContextInjected { ids, .. } = event.event
            && event.agent_run_id == result.agent_run_id
        {
            receipts.push(ids);
        }
    }
    assert!(
        receipts
            .iter()
            .any(|ids| ids.contains(&"mail:1".to_string())),
        "{receipts:?}"
    );
    assert!(workflow.pending_news(result.agent_run_id).is_empty());
}

/// A lead that finished before its subagent did gets `NewsWaiting`, and a
/// continuation gets the report at its first turn boundary. The report
/// survives a checkpoint and resume while it waits.
#[tokio::test]
async fn news_for_an_idle_lead_waits_and_reaches_its_next_turn() {
    let server = serve(vec![
        (
            "lead",
            vec![start("Take your time."), complete("Started it.")],
        ),
        ("r1", vec![("gate", json!({})), complete("Took my time.")]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let run_dir = workspace.0.join(".run");
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .agent_tool("worker", gate.clone())
        .persistence(PersistenceConfig::new(&run_dir))
        .build()
        .unwrap();
    let mut events = workflow.subscribe_subagents();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    assert_eq!(lead.summary, "Started it.");
    within(gate.started.notified()).await;
    gate.open.notify_one();
    let news = within(async {
        loop {
            let event = events.recv().await.unwrap();
            if let SubagentEventKind::NewsWaiting { mail_ids } = event.kind {
                assert_eq!(event.caller_run_id, lead.agent_run_id);
                return mail_ids;
            }
        }
    })
    .await;
    assert_eq!(news, [1]);
    let pending = workflow.pending_news(lead.agent_run_id);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].kind, MailKind::Report);
    assert_eq!(pending[0].body, "Took my time.");

    // Saved, it waits across a resume.
    workflow.checkpoint().await.unwrap();
    drop(workflow);
    let latest = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
    let resumed = builder(&server, &workspace, SubagentsConfig::default())
        .agent_tool("worker", gate.clone())
        .persistence(PersistenceConfig::new(&run_dir))
        .resume_from(run_dir.join(latest.trim()))
        .build()
        .unwrap();
    assert_eq!(resumed.pending_news(lead.agent_run_id).len(), 1);
    assert_eq!(resumed.subagents()[0].state, SubagentState::Finished);
    let lead_result = resumed.agent_result(lead.agent_run_id).unwrap();
    let continued = within(resumed.continue_agent(&lead_result, "Your agents have news."))
        .await
        .unwrap();
    assert_eq!(continued.agent_run_id, lead.agent_run_id);
    let last = bodies_of(&server, "lead").pop().unwrap();
    assert!(last.contains("Your agents have news."));
    assert!(
        last.contains("[mail 1] r1 (agent `worker`) finished. Its report:\\n  Took my time."),
        "{last}"
    );
    assert!(resumed.pending_news(lead.agent_run_id).is_empty());
}

/// Messages go both ways while a subagent runs, and each wait returns the
/// text itself: the subagent asks, the lead answers, the subagent reports.
/// A message to the finished subagent continues its run, with its context.
#[tokio::test]
async fn messages_go_both_ways_and_a_finished_subagent_continues() {
    let server = serve(vec![
        (
            "lead",
            vec![
                start("Ask me first."),
                wait(),
                (
                    "message_agent",
                    json!({ "to": "r1", "message": "Use blue." }),
                ),
                wait(),
                (
                    "message_agent",
                    json!({ "to": "r1", "message": "Now make it red." }),
                ),
                wait(),
            ],
        ),
        (
            "r1",
            vec![
                (
                    "message_agent",
                    json!({ "to": "lead", "message": "Which colour?" }),
                ),
                wait(),
                complete("Painted it blue."),
                complete("Painted it red."),
            ],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Paint.")).await.unwrap();
    let lead = bodies_of(&server, "lead");
    let asked = result_of(&lead[2], "lead", 1);
    assert!(
        asked.contains(
            "r1 (agent `worker`) sent you a message (it goes on working):\n\nWhich colour?"
        ),
        "{asked}"
    );
    assert!(result_of(&lead[4], "lead", 3).contains("Painted it blue."));
    let sent = result_of(&lead[5], "lead", 4);
    assert!(
        sent.contains("continues its task with your message"),
        "{sent}"
    );
    assert!(result_of(&lead[6], "lead", 5).contains("Painted it red."));

    let worker = bodies_of(&server, "r1");
    let answered = result_of(&worker[2], "r1", 1);
    assert!(
        answered.contains("Your lead sent you a message:\n\nUse blue."),
        "{answered}"
    );
    // Continued: the same conversation, with the message after the report.
    let continued = worker.last().unwrap();
    assert!(continued.contains("Painted it blue."));
    assert!(continued.contains("Now make it red."));
    let records = workflow.subagents();
    assert_eq!(records.len(), 1, "continued, not started again");
    assert_eq!(records[0].report.as_deref(), Some("Painted it red."));
}

/// The host's message goes as the lead's would, under the host's heading:
/// a finished subagent continues with it. The lead gets a copy with its
/// news, after the report before it and before the report it brings.
#[tokio::test]
async fn the_hosts_message_continues_a_subagent_and_its_lead_gets_a_copy() {
    let server = serve(vec![
        ("lead", vec![start("Paint it."), complete("Started it.")]),
        (
            "r1",
            vec![complete("Painted it blue."), complete("Painted it red.")],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .build()
        .unwrap();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| workflow.subagents()[0].state == SubagentState::Finished).await;
    let send = |to: &str| {
        workflow.message_subagent(
            to,
            "Now make it red.",
            "The user sent you a message:",
            "The user messaged r1 (a copy for you):",
        )
    };
    assert!(send("r9").is_err());
    let sent = send("r1").unwrap();
    assert!(sent.contains("continues its task"), "{sent}");
    until(|| workflow.subagents()[0].report.as_deref() == Some("Painted it red.")).await;
    let continued = bodies_of(&server, "r1").pop().unwrap();
    assert!(
        continued.contains("The user sent you a message:\\n  Now make it red."),
        "{continued}"
    );
    let news = workflow.pending_news(lead.agent_run_id);
    let kinds: Vec<(MailKind, &str)> = news
        .iter()
        .map(|item| (item.kind, item.heading.as_str()))
        .collect();
    assert_eq!(
        kinds,
        [
            (
                MailKind::Report,
                "r1 (agent `worker`) finished. Its report:"
            ),
            (MailKind::Copy, "The user messaged r1 (a copy for you):"),
            (
                MailKind::Report,
                "r1 (agent `worker`) finished. Its report:"
            ),
        ]
    );
}

/// Over the cap a start is queued, and starts when a place frees up.
#[tokio::test]
async fn starts_over_the_cap_wait_in_a_queue() {
    let server = serve(vec![
        (
            "lead",
            vec![
                start("First."),
                start("Second."),
                ("list_agents", json!({})),
                ("wait_for_agents", json!({ "all": true })),
            ],
        ),
        ("r1", vec![complete("One.")]),
        ("r2", vec![complete("Two.")]),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_running(1),
    )
    .build()
    .unwrap();
    let mut events = workflow.subscribe_subagents();
    within(workflow.run_agent("lead", "Delegate twice."))
        .await
        .unwrap();
    let lead = bodies_of(&server, "lead");
    let queued = result_of(&lead[2], "lead", 1);
    assert!(queued.starts_with("Queued r2 (agent `worker`)"), "{queued}");
    // Each report reaches the lead once: at a turn boundary, or in the wait.
    let all = result_of(&lead[4], "lead", 3);
    assert!(all.contains("Two."), "{all}");
    assert!(lead[4].contains("One."));
    assert!(all.contains("None of your agent runs is running now."));
    let mut order = Vec::new();
    while let Ok(event) = events.try_recv() {
        match event.kind {
            SubagentEventKind::Queued
            | SubagentEventKind::Started
            | SubagentEventKind::Finished { .. } => {
                order.push(format!(
                    "{} {:?}",
                    event.handle,
                    std::mem::discriminant(&event.kind)
                ));
            }
            _ => {}
        }
    }
    let position = |handle: &str, kind: SubagentEventKind| {
        order
            .iter()
            .position(|line| *line == format!("{handle} {:?}", std::mem::discriminant(&kind)))
            .unwrap()
    };
    assert!(
        position("r2", SubagentEventKind::Queued)
            < position(
                "r1",
                SubagentEventKind::Finished {
                    summary: String::new()
                }
            )
    );
    assert!(
        position(
            "r1",
            SubagentEventKind::Finished {
                summary: String::new()
            }
        ) < position("r2", SubagentEventKind::Started)
    );
    assert!(
        workflow
            .subagents()
            .iter()
            .all(|run| run.state == SubagentState::Finished)
    );
}

/// A queued start the lead stops never runs, and the lead's message starts
/// it later, from its task, with the message.
#[tokio::test]
async fn a_queued_start_stopped_starts_with_a_later_message() {
    let server = serve(vec![
        (
            "lead",
            vec![
                start("First."),
                start("Second."),
                ("stop_agent", json!({ "run": "r2" })),
                (
                    "message_agent",
                    json!({ "to": "r2", "message": "Do it after all." }),
                ),
                ("message_agent", json!({ "to": "r1", "message": "Finish." })),
                ("wait_for_agents", json!({ "all": true })),
            ],
        ),
        ("r1", vec![wait(), complete("One.")]),
        ("r2", vec![complete("Two.")]),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_running(1),
    )
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate twice."))
        .await
        .unwrap();
    let lead = bodies_of(&server, "lead");
    let stopped = result_of(&lead[3], "lead", 2);
    assert!(
        stopped.contains("Stopped r2 before it started"),
        "{stopped}"
    );
    let all = result_of(&lead[6], "lead", 5);
    assert!(lead[6].contains("One.") && all.contains("Two."), "{all}");
    let second = bodies_of(&server, "r2");
    assert!(second[0].contains("Second."));
    assert!(
        second.iter().any(|body| body.contains("Do it after all.")),
        "the lead's message reaches it"
    );
    assert!(
        workflow
            .subagents()
            .iter()
            .all(|run| run.state == SubagentState::Finished)
    );
}

/// A lead's stop winds its subagent down: the subagent saves its progress
/// and reports, and the lead is told which run stopped and why. A message
/// restarts it where it stopped.
#[tokio::test]
async fn a_stopped_subagent_reports_and_restarts_with_a_message() {
    let server = ResponsesServer::start(|_, body| {
        let role = role(body);
        let turn = turn(body);
        let id = format!("{role}_{turn}");
        Some(match (role.as_str(), turn) {
            ("lead", 0) => tool_call_turn(
                &id,
                "start_agent",
                json!({ "agent": "worker", "task": "Work long." }),
            ),
            ("lead", 1) => tool_call_turn(
                &id,
                "stop_agent",
                json!({ "run": "r1", "reason": "plans changed" }),
            ),
            ("lead", 2) => tool_call_turn(&id, "wait_for_agents", json!({})),
            ("lead", 3) => tool_call_turn(
                &id,
                "message_agent",
                json!({ "to": "r1", "message": "Carry on after all." }),
            ),
            ("lead", 4) => tool_call_turn(&id, "wait_for_agents", json!({})),
            ("r1", _) if body.contains("Carry on after all.") => tool_call_turn(
                &id,
                "complete_task",
                json!({ "summary": "Finished after all." }),
            ),
            ("r1", _) if body.contains("Your run is being stopped") => {
                tool_call_turn(&id, "complete_task", json!({ "summary": "Half done." }))
            }
            ("r1", _) => tool_call_turn(&id, "gate", json!({})),
            (_, _) => tool_call_turn(&id, "complete_task", json!({ "summary": "lead done" })),
        })
    })
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let lead = bodies_of(&server, "lead");
    let stopping = result_of(&lead[2], "lead", 1);
    assert!(stopping.starts_with("r1 is stopping"), "{stopping}");
    let told = result_of(&lead[3], "lead", 2);
    assert!(
        told.contains(
            "r1 (agent `worker`) stopped before it finished: you stopped it: plans changed."
        ) && told.contains("Half done."),
        "{told}"
    );
    let restarted = result_of(&lead[4], "lead", 3);
    assert!(
        restarted.contains("restarts where it stopped"),
        "{restarted}"
    );
    assert!(result_of(&lead[5], "lead", 4).contains("Finished after all."));
    let record = &workflow.subagents()[0];
    assert_eq!(record.state, SubagentState::Finished);
}

/// A run stopped by the host tells its lead too.
#[tokio::test]
async fn a_host_stop_is_reported_to_the_lead() {
    let server = serve(vec![
        ("lead", vec![start("Work."), wait()]),
        ("r1", vec![("gate", json!({}))]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = Arc::new(
        builder(&server, &workspace, SubagentsConfig::default())
            .agent_tool("worker", gate.clone())
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Delegate.").await }
    });
    within(gate.started.notified()).await;
    let id = workflow.subagents()[0].run_id;
    assert!(workflow.stop_subagent(id, crate::StopRequest::operator("maintenance")));
    within(run).await.unwrap().unwrap();
    let lead = bodies_of(&server, "lead");
    let told = result_of(&lead[2], "lead", 1);
    assert!(
        told.contains("the workflow's host stopped it (maintenance)"),
        "{told}"
    );
    assert_eq!(workflow.subagents()[0].state, SubagentState::Stopped);
}

async fn start_worker(workflow: &Workflow, caller: Uuid, task: &str) -> Uuid {
    let ctx = workflow.context();
    ctx.subagents
        .start(
            ctx,
            caller,
            "lead",
            0,
            &ctx.workspace,
            "worker",
            task,
            None,
            false,
            None,
            Default::default(),
        )
        .await
        .unwrap();
    workflow.subagents().last().unwrap().run_id
}

#[tokio::test]
async fn a_host_stop_of_a_queued_subagent_saves_news_and_allows_restart() {
    let server = serve(vec![("r1", vec![("gate", json!({}))])]).await;
    let workspace = Workspace::new();
    let run_dir = workspace.0.join(".run");
    let gate = Arc::new(Gate::default());
    let workflow = builder(
        &server,
        &workspace,
        SubagentsConfig::default().max_running(1),
    )
    .agent_tool("worker", gate.clone())
    .persistence(PersistenceConfig::new(&run_dir))
    .build()
    .unwrap();
    let caller = Uuid::new_v4();
    start_worker(&workflow, caller, "First.").await;
    let queued = start_worker(&workflow, caller, "Second.").await;
    within(gate.started.notified()).await;
    assert_eq!(workflow.subagents()[1].state, SubagentState::Queued);
    assert!(workflow.stop_subagent(queued, crate::StopRequest::operator("plans changed")));
    assert!(!workflow.stop_subagent(queued, crate::StopRequest::operator("again")));
    assert!(!workflow.stop_subagent(Uuid::new_v4(), crate::StopRequest::operator("unknown")));
    assert!(
        workflow.pending_news(caller).iter().any(|item| {
            item.kind == MailKind::Stopped && item.heading.contains("plans changed")
        })
    );
    workflow.checkpoint().await.unwrap();
    let latest = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
    let saved =
        crate::persistence::PersistenceManager::load_checkpoint(&run_dir.join(latest.trim()))
            .await
            .unwrap()
            .subagents
            .unwrap();
    assert_eq!(saved.runs[1].state, SubagentState::Stopped);
    assert_eq!(saved.runs[1].pending, None);
    assert!(saved.mail.iter().any(|item| item.kind == MailKind::Stopped));
    gate.open.notify_one();
    until(|| workflow.active_subagents().is_empty()).await;
    assert!(bodies_of(&server, "r2").is_empty());
    workflow
        .context()
        .subagents
        .message(workflow.context(), caller, "r2", "Do it after all.")
        .unwrap();
    until(|| workflow.subagents()[1].state == SubagentState::Finished).await;
    assert!(bodies_of(&server, "r2")[0].contains("Second."));
    assert!(
        bodies_of(&server, "r2")
            .iter()
            .any(|body| body.contains("Do it after all."))
    );
}

#[tokio::test]
async fn a_host_stop_before_registration_is_applied_and_allows_restart() {
    for (wraps_up, escalate) in [(false, false), (true, false), (true, true)] {
        let server = serve(vec![]).await;
        let workspace = Workspace::new();
        let workflow = builder(&server, &workspace, SubagentsConfig::default())
            .build()
            .unwrap();
        let caller = Uuid::new_v4();
        // The current-thread runtime cannot poll the spawned driver before
        // this task yields, so this stop precedes RunControl registration.
        let id = start_worker(&workflow, caller, "Work.").await;
        assert!(workflow.context().run_control.run_cancel(id).is_none());
        let mut request = crate::StopRequest::operator("startup stop");
        if wraps_up {
            request = request.wrap_up(Duration::from_secs(30));
        }
        assert!(workflow.stop_subagent(id, request.clone()));
        assert!(!workflow.stop_subagent(id, request));
        if escalate {
            assert!(workflow.stop_subagent(id, crate::StopRequest::operator("stop now")));
        }
        // A checkpoint in this same registration window preserves the
        // stop even though the agent has no saved Task yet.
        let saved =
            serde_json::to_string(&workflow.context().subagents.snapshot().unwrap()).unwrap();
        let restored = super::board::Board::new(
            Some(SubagentsConfig::default()),
            Some(serde_json::from_str(&saved).unwrap()),
            &[],
            Uuid::new_v4(),
            None,
        );
        assert_eq!(restored.records()[0].state, SubagentState::Stopped);
        assert_eq!(restored.records()[0].error.as_deref(), Some("startup stop"));
        assert!(
            restored
                .pending(caller)
                .iter()
                .any(|item| item.kind == MailKind::Stopped)
        );
        until(|| workflow.subagents()[0].state == SubagentState::Stopped).await;
        let stopped = workflow.agent_result(id).unwrap().stopped_by.unwrap();
        assert_eq!(stopped.reason, "startup stop");
        assert_eq!(stopped.intent, Some(crate::RunStopIntent::Relaunch));
        if !wraps_up || escalate {
            assert!(bodies_of(&server, "r1").is_empty());
        }
        workflow
            .context()
            .subagents
            .message(workflow.context(), caller, "r1", "Carry on.")
            .unwrap();
        until(|| workflow.subagents()[0].state == SubagentState::Finished).await;
    }
}

/// A subagent's question goes to its lead, and the lead's message answers
/// it: the subagent gets the answer as its question's.
#[tokio::test]
async fn a_subagents_question_goes_to_its_lead() {
    let server = serve(vec![
        (
            "lead",
            vec![
                start("Ask before you act."),
                wait(),
                (
                    "message_agent",
                    json!({ "to": "r1", "message": "Yes, go ahead." }),
                ),
                wait(),
            ],
        ),
        (
            "r1",
            vec![
                (
                    "ask_user",
                    json!({ "questions": "May I delete the cache?" }),
                ),
                complete("Deleted it."),
            ],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let mut worker = agent_config("Worker");
    worker
        .tool_config
        .available_tool_names
        .push("ask_user".to_string());
    let workflow = builder_with(&server, &workspace, SubagentsConfig::default(), worker)
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let lead = bodies_of(&server, "lead");
    let asked = result_of(&lead[2], "lead", 1);
    assert!(
        asked.contains("r1 (agent `worker`) asks you") && asked.contains("May I delete the cache?"),
        "{asked}"
    );
    assert!(
        result_of(&lead[3], "lead", 2).contains("gets it as the answer"),
        "{}",
        result_of(&lead[3], "lead", 2)
    );
    assert!(result_of(&lead[4], "lead", 3).contains("Deleted it."));
    let worker = bodies_of(&server, "r1");
    assert!(worker.last().unwrap().contains("Yes, go ahead."));
}

/// Linked subagents message each other, up to the cap between two of
/// their lead's messages; the lead may get a copy.
#[tokio::test]
async fn linked_subagents_message_each_other_up_to_a_cap() {
    let server = serve(vec![
        (
            "lead",
            vec![
                (
                    "start_agent",
                    json!({ "agent": "worker", "task": "Wait for r2." }),
                ),
                (
                    "start_agent",
                    json!({ "agent": "worker", "task": "Tell r1.", "peers": "r1" }),
                ),
                ("wait_for_agents", json!({ "all": true })),
            ],
        ),
        ("r1", vec![wait()]),
        (
            "r2",
            vec![
                (
                    "message_agent",
                    json!({ "to": "r1", "message": "Found it." }),
                ),
                ("message_agent", json!({ "to": "r1", "message": "Again." })),
            ],
        ),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(
        &server,
        &workspace,
        SubagentsConfig::default()
            .max_peer_messages(1)
            .copy_peer_messages_to_lead(true),
    )
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let r1 = bodies_of(&server, "r1");
    let got = result_of(&r1[1], "r1", 0);
    assert!(
        got.contains(
            "r2 (agent `worker`), another agent run of your lead, sent you a message:\n\nFound it."
        ),
        "{got}"
    );
    let r2 = bodies_of(&server, "r2");
    let refused = result_of(&r2[2], "r2", 1);
    assert!(refused.contains("the most allowed"), "{refused}");
    let lead = bodies_of(&server, "lead");
    let all = lead.iter().map(|body| body.as_str()).collect::<String>();
    assert!(all.contains("sent r1 a message (a copy for you)"));
    // Unlinked, r1 can't message r2's peers it doesn't know: r2 linked them both ways.
    assert_eq!(
        workflow.subagents()[1].peers,
        vec![workflow.subagents()[0].run_id]
    );
}

/// A run works in the directory it is given: the host's
/// `run_agent_with`, and `start_agent`'s `workspace` (inside the caller's
/// directory only).
#[tokio::test]
async fn runs_work_in_their_own_directories() {
    let workspace = Workspace::new();
    std::fs::create_dir_all(workspace.0.join("a/b")).unwrap();
    let server = serve(vec![
        (
            "lead",
            vec![
                ("where", json!({})),
                (
                    "start_agent",
                    json!({ "agent": "worker", "task": "Where are you?", "workspace": "b" }),
                ),
                (
                    "start_agent",
                    json!({ "agent": "worker", "task": "Escape.", "workspace": "../.." }),
                ),
                wait(),
            ],
        ),
        ("r1", vec![("where", json!({}))]),
    ])
    .await;
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .global_tool(Arc::new(Where))
        .build()
        .unwrap();
    let result =
        within(workflow.run_agent_with("lead", "Look.", crate::RunOptions::new().workspace("a")))
            .await
            .unwrap();
    let a = workspace.0.join("a");
    assert_eq!(result.task.current_working_directory, a);
    let lead = bodies_of(&server, "lead");
    assert_eq!(
        result_of(&lead[1], "lead", 0),
        format!("here: {}", a.display())
    );
    let escaped = result_of(&lead[3], "lead", 2);
    assert!(
        escaped.contains("must be inside your working directory"),
        "{escaped}"
    );
    let worker = bodies_of(&server, "r1");
    assert_eq!(
        result_of(&worker[1], "r1", 0),
        format!("here: {}", a.join("b").display())
    );
    assert_eq!(workflow.subagents()[0].workspace, a.join("b"));
    // A directory that doesn't exist is refused.
    let error = workflow
        .run_agent_with(
            "lead",
            "Look.",
            crate::RunOptions::new().workspace("missing"),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("does not exist"), "{error}");
}

/// A provider that makes a directory per run and counts removals.
#[derive(Debug, Default)]
struct Folders {
    root: std::path::PathBuf,
    removed: AtomicUsize,
}

#[async_trait::async_trait]
impl WorkspaceProvider for Folders {
    async fn create(&self, request: &WorkspaceRequest) -> Result<IsolatedWorkspace, String> {
        let path = self.root.join(&request.handle);
        std::fs::create_dir_all(&path).map_err(|error| error.to_string())?;
        Ok(IsolatedWorkspace {
            branch: Some(format!("work/{}", request.handle)),
            description: Some(format!("You work in your own folder {}.", path.display())),
            ..IsolatedWorkspace::at(path)
        })
    }

    async fn remove(&self, workspace: &IsolatedWorkspace) -> Result<(), String> {
        self.removed.fetch_add(1, Ordering::SeqCst);
        std::fs::remove_dir_all(&workspace.path).map_err(|error| error.to_string())
    }
}

/// An isolated subagent works in the workspace the host's provider makes,
/// which is kept for its later runs and removed on request; a host
/// parameter of `start_agent` reaches the host's hook.
#[tokio::test]
async fn an_isolated_subagent_works_in_its_providers_workspace() {
    let workspace = Workspace::new();
    let server = serve(vec![
        (
            "lead",
            vec![
                ("start_agent", json!({ "agent": "worker", "task": "Build it.", "isolated": true, "flavour": "spicy" })),
                wait(),
                ("stop_agent", json!({ "run": "r1", "remove_workspace": true })),
                ("message_agent", json!({ "to": "r1", "message": "More." })),
            ],
        ),
        ("r1", vec![("where", json!({}))]),
    ])
    .await;
    let provider = Arc::new(Folders {
        root: workspace.0.join("isolated"),
        ..Folders::default()
    });
    let hooked = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let config = SubagentsConfig::default()
        .workspace_provider(provider.clone())
        .start_parameter(arko_agents::ToolParameterDefinition {
            name: "flavour".to_string(),
            description: "How it tastes.".to_string(),
            param_type: arko_agents::ToolParameterType::String,
            required: false,
            is_path: false,
            flag: None,
            default_value: None,
        })
        .hook(Arc::new({
            let hooked = hooked.clone();
            move |request: &StartRequest, config: &mut arko_agents::AgentConfig| {
                hooked
                    .lock()
                    .push(request.parameters.get("flavour").cloned());
                config.instructions.push_str(" Taste it.");
                Ok(())
            }
        }));
    let workflow = builder(&server, &workspace, config)
        .global_tool(Arc::new(Where))
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    let path = workspace.0.join("isolated").join("r1");
    let lead = bodies_of(&server, "lead");
    assert!(offered(&lead[0]).contains(&"start_agent".to_string()));
    assert!(
        lead[0].contains("\"flavour\""),
        "the host's parameter is offered"
    );
    let started = result_of(&lead[1], "lead", 0);
    assert!(
        started.contains("in an isolated workspace") && started.contains("work/r1"),
        "{started}"
    );
    let worker = bodies_of(&server, "r1");
    assert!(worker[0].contains("You work in your own folder"));
    assert!(
        worker[0].contains("Taste it."),
        "the hook changed its configuration"
    );
    assert_eq!(
        result_of(&worker[1], "r1", 0),
        format!("here: {}", path.display())
    );
    assert_eq!(*hooked.lock(), [Some(json!("spicy"))]);
    let removed = result_of(&lead[3], "lead", 2);
    assert!(removed.contains("Removed r1's workspace"), "{removed}");
    assert_eq!(provider.removed.load(Ordering::SeqCst), 1);
    let refused = result_of(&lead[4], "lead", 3);
    assert!(refused.contains("workspace was removed"), "{refused}");
    let record = &workflow.subagents()[0];
    assert_eq!(record.state, SubagentState::Closed);
    assert_eq!(record.isolated.as_ref().unwrap().path, path);
}

/// The user messages a request sends, joined.
fn user_text(body: &str) -> String {
    let body: Value = serde_json::from_str(body).unwrap();
    let mut text = Vec::new();
    for item in body["input"].as_array().unwrap() {
        if item["role"] != "user" {
            continue;
        }
        match &item["content"] {
            Value::String(content) => text.push(content.clone()),
            Value::Array(parts) => text.extend(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str().map(str::to_string)),
            ),
            _ => {}
        }
    }
    text.join("\n")
}

/// The host's start context comes first in a run's first message, with the
/// task under its own heading after it, and is asked for as each run
/// starts; a run it gives none (or blank text) starts as without it. A
/// handover restates it with the task.
#[tokio::test]
async fn the_hosts_start_context_opens_the_first_message() {
    let server = ResponsesServer::start(|_, body| {
        let role = role(body);
        let turn = turn(body);
        let id = format!("{role}_{turn}");
        let text = user_text(body);
        Some(match role.as_str() {
            "lead" => match turn {
                0 => tool_call_turn(
                    &id,
                    "start_agent",
                    json!({ "agent": "worker", "task": "Count the files." }),
                ),
                1 => tool_call_turn(
                    &id,
                    "start_agent",
                    json!({ "agent": "worker", "task": "Sort the files." }),
                ),
                2 | 3 => tool_call_turn(&id, "wait_for_agents", json!({})),
                _ => tool_call_turn(&id, "complete_task", json!({ "summary": "lead done" })),
            },
            "r1" if text.contains("# Handover report") => {
                tool_call_turn(&id, "complete_task", json!({ "summary": "Counted." }))
            }
            "r1" if text.contains("Your conversation is getting long") => tool_call_turn(
                &id,
                "handover_report",
                json!({ "report": "REPORT: counted half." }),
            ),
            // Under the handover size, then over it.
            "r1" => tool_call_turn(&id, "where", json!({})).replace(
                "\"input_tokens\":10",
                if turn == 0 {
                    "\"input_tokens\":5000"
                } else {
                    "\"input_tokens\":20000"
                },
            ),
            _ => tool_call_turn(&id, "complete_task", json!({ "summary": "Sorted." })),
        })
    })
    .await;
    let workspace = Workspace::new();
    let asked = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let config = SubagentsConfig::default().start_context(Arc::new({
        let asked = asked.clone();
        move |request: &StartRequest| {
            asked
                .lock()
                .push((request.handle.clone(), request.agent.clone()));
            Some(if request.handle == "r1" {
                "# Project rules\n\nUse tabs.\n".to_string()
            } else {
                "  \n".to_string()
            })
        }
    }));
    let mut worker = agent_config("Worker");
    worker.handover = Some(arko_agents::HandoverConfig::at_input_tokens(10_000));
    let workflow = builder_with(&server, &workspace, config, worker)
        .global_tool(Arc::new(Where))
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    until(|| {
        workflow
            .subagents()
            .iter()
            .all(|run| run.state == SubagentState::Finished)
    })
    .await;
    assert_eq!(
        *asked.lock(),
        [
            ("r1".to_string(), "worker".to_string()),
            ("r2".to_string(), "worker".to_string())
        ]
    );
    let first = bodies_of(&server, "r1");
    let opening = user_text(&first[0]);
    assert!(
        opening.contains(
            "# Project rules\n\nUse tabs.\n\n# Your task\n\nCount the files.\n\n# Working as an agent run\nYou are agent run r1,"
        ),
        "{opening}"
    );
    // The fresh conversation after the handover restates it, once.
    let fresh = user_text(first.last().unwrap());
    assert!(fresh.contains("REPORT: counted half."), "{fresh}");
    assert_eq!(fresh.matches("Use tabs.").count(), 1, "{fresh}");
    assert!(
        fresh.contains("# Project rules\n\nUse tabs.\n\n# Your task\n\nCount the files."),
        "{fresh}"
    );
    let second = user_text(&bodies_of(&server, "r2")[0]);
    assert!(
        second.contains("Sort the files.\n\n# Working as an agent run\nYou are agent run r2,"),
        "{second}"
    );
    assert!(!second.contains("# Your task"), "{second}");
    // The record keeps the task as the lead gave it.
    assert_eq!(workflow.subagents()[0].task, "Count the files.");
}

/// A subagent running when its workflow stops comes back from the
/// checkpoint stopped, with its lead's knowledge of it, and
/// `resume_subagents` relaunches it where it was. The news that it stopped,
/// which the lead hadn't received, is withdrawn, and its report then
/// reaches its lead alone.
#[tokio::test]
async fn subagents_survive_a_stop_and_a_resume() {
    let server = ResponsesServer::start(|_, body| {
        let role = role(body);
        let turn = turn(body);
        let id = format!("{role}_{turn}");
        Some(match (role.as_str(), turn) {
            ("lead", 0) => tool_call_turn(
                &id,
                "start_agent",
                json!({ "agent": "worker", "task": "Work on." }),
            ),
            ("lead", 1) => tool_call_turn(&id, "gate", json!({})),
            ("r1", _) if body.contains("has now been reopened") => {
                tool_call_turn(&id, "complete_task", json!({ "summary": "Done at last." }))
            }
            ("r1", _) => tool_call_turn(&id, "gate", json!({})),
            _ => tool_call_turn(&id, "complete_task", json!({ "summary": "lead done" })),
        })
    })
    .await;
    let workspace = Workspace::new();
    let run_dir = workspace.0.join(".run");
    let gate = Arc::new(Gate::default());
    let workflow = Arc::new(
        builder(&server, &workspace, SubagentsConfig::default())
            .global_tool(gate.clone())
            .persistence(PersistenceConfig::new(&run_dir))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Delegate.").await }
    });
    until(|| gate.calls.load(Ordering::SeqCst) == 2).await;
    workflow.stop(crate::StopRequest::operator("closing"));
    let _ = within(run).await.unwrap();
    until(|| workflow.active_subagents().is_empty()).await;
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let latest = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
    let gate = Arc::new(Gate::default());
    let resumed = builder(&server, &workspace, SubagentsConfig::default())
        .global_tool(gate.clone())
        .persistence(PersistenceConfig::new(&run_dir))
        .resume_from(run_dir.join(latest.trim()))
        .build()
        .unwrap();
    let records = resumed.subagents();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].state, SubagentState::Stopped);
    // Left to `resume_subagents`, not to the host's restart of its own runs.
    assert!(
        resumed
            .stopped_runs()
            .iter()
            .all(|run| run.result.agent_name == "lead")
    );
    let lead = resumed
        .stopped_runs()
        .into_iter()
        .find(|run| run.result.agent_name == "lead")
        .unwrap();
    let stopped = |news: &[MailItem]| news.iter().any(|item| item.kind == MailKind::Stopped);
    let news = resumed.pending_news(lead.result.agent_run_id);
    assert!(stopped(&news), "the stop's news: {news:?}");
    assert_eq!(resumed.resume_subagents(), ["r1"]);
    let news = resumed.pending_news(lead.result.agent_run_id);
    assert!(!stopped(&news), "withdrawn: {news:?}");
    until(|| resumed.subagents()[0].state == SubagentState::Finished).await;
    let news = resumed.pending_news(lead.result.agent_run_id);
    let bodies: Vec<&str> = news.iter().map(|item| item.body.as_str()).collect();
    assert_eq!(bodies, ["Done at last."]);
}

/// Background runs need workflow services, and host parameters may not
/// take `start_agent`'s own names; a host tool may not take a subagent
/// tool's name.
#[test]
fn the_builder_checks_the_subagents_configuration() {
    let build = |builder: WorkflowBuilder| builder.build().err().map(|error| error.to_string());
    let base = || {
        Workflow::builder()
            .name("subagents-test")
            .workspace(std::env::temp_dir())
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("lead", agent_config("Lead"))
            .agent("worker", agent_config("Worker"))
    };
    let error = build(base().subagents(SubagentsConfig::default())).unwrap();
    assert!(error.contains("need workflow_services"), "{error}");
    let services = || {
        WorkflowServicesConfig::default()
            .allow_caller("lead")
            .allow_target("worker")
    };
    let clash = arko_agents::ToolParameterDefinition {
        name: "task".to_string(),
        description: String::new(),
        param_type: arko_agents::ToolParameterType::String,
        required: false,
        is_path: false,
        flag: None,
        default_value: None,
    };
    let error = build(
        base()
            .workflow_services(services())
            .subagents(SubagentsConfig::default().start_parameter(clash)),
    )
    .unwrap();
    assert!(error.contains("already has a parameter 'task'"), "{error}");
    assert!(
        build(
            base()
                .workflow_services(services())
                .subagents(SubagentsConfig::default())
        )
        .is_none()
    );
}

/// Saved mail and records load from JSON with only their required fields.
#[test]
fn a_snapshot_loads_with_defaults() {
    let snapshot: SubagentsSnapshot = serde_json::from_value(json!({})).unwrap();
    assert_eq!(snapshot, SubagentsSnapshot::default());
    let item: MailItem = serde_json::from_value(json!({
        "id": 3, "to": uuid::Uuid::nil(), "from": "host", "kind": "message",
        "heading": "h", "body": "b", "at": "2026-01-01T00:00:00Z",
    }))
    .unwrap();
    assert_eq!(item.delivered_at, None);
    assert_eq!(item.text(), "[mail 3] h\n\nb\n\n[end of mail 3]");
    let long = MailItem {
        body: "x".repeat(MAX_MAIL_BYTES + 10),
        file: Some("/tmp/mail-3.md".into()),
        ..item
    };
    let excerpt = long.excerpt();
    assert!(excerpt.len() < MAX_MAIL_BYTES);
    assert!(excerpt.contains("the whole text is in /tmp/mail-3.md"));
    assert!(excerpt.find("[end of mail 3]") > excerpt.find("cut here"));
    let _ = AgentRunStatus::Completed;
}

/// A lead cut back to before the message whose turn started a subagent
/// (`ContinueOptions::cut_back_to`) abandoned that timeline: the subagent
/// is detached, and its report, waiting for the lead, isn't given to the
/// cut conversation. It stays in the subagent's record.
#[tokio::test]
async fn a_cut_back_lead_gets_no_mail_from_a_subagent_it_no_longer_started() {
    let server = ResponsesServer::start(|_, body| {
        let id = format!("{}_{}", role(body), turn(body));
        Some(match (role(body).as_str(), turn(body)) {
            ("lead", _) if body.contains("Something else.") => tool_call_turn(
                &id,
                "complete_task",
                json!({ "summary": "lead done again" }),
            ),
            ("lead", 1) => {
                let (tool, arguments) = start("Count the files.");
                tool_call_turn(&id, tool, arguments)
            }
            ("lead", _) => tool_call_turn(&id, "complete_task", json!({ "summary": "lead done" })),
            (_, _) => tool_call_turn(
                &id,
                "complete_task",
                json!({ "summary": "There are 3 files." }),
            ),
        })
    })
    .await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "First.")).await.unwrap();
    let delegated = within(workflow.continue_agent(&first, "Delegate the count."))
        .await
        .unwrap();
    until(|| {
        workflow
            .subagents()
            .first()
            .is_some_and(|record| record.state == SubagentState::Finished)
    })
    .await;
    let index = delegated
        .task
        .user_turns()
        .iter()
        .find(|turn| turn.text.starts_with("Delegate the count."))
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
    let last = bodies_of(&server, "lead").last().unwrap().clone();
    assert!(!last.contains("There are 3 files."), "{last}");
    assert!(!last.contains("Count the files."), "{last}");
    let records = workflow.subagents();
    assert!(records[0].detached);
    assert_eq!(records[0].report.as_deref(), Some("There are 3 files."));
}

/// A subagent's prompt cache is refreshed with the tools of its own
/// requests, which its depth decides: a subagent that may itself start
/// subagents is offered no `start_agent` at the deepest level, and neither
/// is its refresh.
#[tokio::test]
async fn a_subagent_s_refresh_offers_the_tools_of_its_depth() {
    use crate::test_support::Reply;
    let scripts = HashMap::from([
        ("lead".to_string(), vec![start("Count the files."), wait()]),
        ("r1".to_string(), vec![complete("There are 3 files.")]),
    ]);
    let server = ResponsesServer::start_async(move |_, body| {
        let request: Value = serde_json::from_str(&body).unwrap();
        let reply = if request["stream"] == json!(false) {
            Reply::Status {
                status: 200,
                retry_after: None,
                body: json!({
                    "id": "resp_r", "object": "response", "created_at": 0,
                    "status": "completed", "model": "gpt-5.6-luna",
                    "output": [{"type": "message", "id": "m1", "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok",
                                             "annotations": []}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                })
                .to_string(),
            }
        } else {
            let (role, turn) = (role(&body), turn(&body));
            let id = format!("{role}_{turn}");
            Reply::Stream(match scripts.get(&role).and_then(|steps| steps.get(turn)) {
                Some((tool, arguments)) => tool_call_turn(&id, tool, arguments.clone()),
                None => tool_call_turn(&id, "complete_task", json!({"summary": "done"})),
            })
        };
        Box::pin(async move { Some(reply) })
    })
    .await;
    let workspace = Workspace::new();
    let workflow = Workflow::builder()
        .name("subagents-test")
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
        .subagents(SubagentsConfig::default())
        .build()
        .unwrap();
    within(workflow.run_agent("lead", "Delegate the count."))
        .await
        .unwrap();
    let run = workflow.subagents()[0].run_id;
    let result = workflow.agent_result(run).unwrap();
    workflow.refresh_prompt_cache(&result).await.unwrap();
    let worker = bodies_of(&server, "r1");
    let (own, refresh) = (&worker[worker.len() - 2], &worker[worker.len() - 1]);
    assert!(refresh.contains("\"stream\":false"), "{refresh}");
    assert!(!offered(own).contains(&"start_agent".to_string()));
    assert_eq!(offered(refresh), offered(own));
}

/// A cut back detaches only the subagents the lead started in the messages
/// it removes. One started before a handover replaced the conversation
/// isn't in them: it stays attached, and its answer to the lead's next
/// message reaches the lead.
#[tokio::test]
async fn a_cut_after_a_handover_keeps_the_subagents_started_before_it() {
    const REPORT: &str = "HANDOVER-REPORT: r1 counts the files.";
    let calls = Arc::new(AtomicUsize::new(0));
    let after_cut = Arc::new(AtomicUsize::new(0));
    let server = ResponsesServer::start(move |_, body| {
        let id = format!("{}_{}", role(body), calls.fetch_add(1, Ordering::SeqCst));
        if role(body) == "r1" {
            let summary = if body.contains("Count again.") {
                "There are 4 files."
            } else {
                "There are 3 files."
            };
            return Some(complete_turn(&id, summary));
        }
        Some(
            if body.contains("Your conversation is getting long") && !body.contains(REPORT) {
                tool_call_turn(&id, "handover_report", json!({ "report": REPORT }))
            } else if body.contains("Something else.") {
                match after_cut.fetch_add(1, Ordering::SeqCst) {
                    0 => tool_call_turn(
                        &id,
                        "message_agent",
                        json!({ "to": "r1", "message": "Count again." }),
                    ),
                    1 => tool_call_turn(&id, "wait_for_agents", json!({ "runs": "r1" })),
                    _ => complete_turn(&id, "lead done again"),
                }
            } else if body.contains("Second.") {
                complete_turn(&id, "second done")
            } else if body.contains(REPORT) {
                complete_turn(&id, "lead done after the handover")
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
    let mut lead = agent_config("Lead");
    lead.handover = Some(arko_agents::HandoverConfig::at_input_tokens(1_000));
    let workflow = Workflow::builder()
        .name("subagents-test")
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
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "First.")).await.unwrap();
    assert_eq!(first.task.handovers.len(), 1);
    let second = within(workflow.continue_agent(&first, "Second."))
        .await
        .unwrap();
    until(|| {
        workflow
            .subagents()
            .first()
            .is_some_and(|record| record.state == SubagentState::Finished)
    })
    .await;
    let index = second
        .task
        .user_turns()
        .iter()
        .find(|turn| turn.text.starts_with("Second."))
        .unwrap()
        .index;
    let again = within(workflow.continue_agent_with(
        &second,
        "Something else.",
        crate::ContinueOptions::new().cut_back_to(index),
    ))
    .await
    .unwrap();
    assert_eq!(again.summary, "lead done again");
    assert!(!workflow.subagents()[0].detached);
    let last = bodies_of(&server, "lead").last().unwrap().clone();
    assert!(last.contains("There are 4 files."), "{last}");
}

/// Moving `wait_for_agents` to the background ends it, and so does context
/// that arrives for the run, each after the reports it collected.
#[tokio::test]
async fn a_wait_ends_when_moved_or_when_context_arrives() {
    // r1 works until the server is dropped, and r2 finishes at once.
    let server = ResponsesServer::start(|_, body| match role(body).as_str() {
        "r2" => Some(tool_call_turn(
            "r2_0",
            "complete_task",
            json!({ "summary": "r2 found it" }),
        )),
        _ => None,
    })
    .await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .build()
        .unwrap();
    let ctx = workflow.context();
    let lead = uuid::Uuid::new_v4();
    for task in ["Search slowly.", "Search quickly."] {
        ctx.subagents
            .start(
                ctx,
                lead,
                "lead",
                0,
                &workspace.0,
                "worker",
                task,
                None,
                false,
                None,
                std::collections::BTreeMap::new(),
            )
            .await
            .unwrap();
    }
    until(|| {
        workflow
            .subagents()
            .iter()
            .any(|run| run.handle == "r2" && run.state == SubagentState::Finished)
    })
    .await;
    let tool = WaitForAgentsTool::new(ctx.clone(), "lead".into(), Arc::new(server.client()));
    assert_eq!(tool.background_mode(), arko_agents::ToolBackground::Own);
    let all = HashMap::from([("all".to_string(), json!(true))]);

    let moves = arko_agents::MoveSource::new();
    let mut task_context = TaskContext::new(lead, "lead", "", &workspace.0);
    task_context.moved = moves.signal();
    let (result, ()) = tokio::join!(tool.execute(&task_context, all.clone()), async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        moves.fire();
    });
    let result = result.unwrap();
    assert!(result.contains("r2 found it"), "{result}");
    assert!(
        result.starts_with(&format!("{} Still running: r1.", board::WAIT_MOVED)),
        "{result}"
    );

    let arrivals = arko_agents::ContextSource::new();
    let mut task_context = TaskContext::new(lead, "lead", "", &workspace.0);
    task_context.context_arrived = arrivals.signal();
    let (result, ()) = within(async {
        tokio::join!(tool.execute(&task_context, all), async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            arrivals.notify();
        })
    })
    .await;
    assert_eq!(
        result.unwrap(),
        format!("{} Still running: r1.", board::CONTEXT_ARRIVED)
    );
}
