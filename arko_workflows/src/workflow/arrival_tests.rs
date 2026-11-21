//! What arrives for a running agent ends its `terminal_control wait`, in
//! arko's runtime and in Claude Code. That is a note posted through the
//! workflow's notes, before the call or while it waits, and a message from a
//! worker's lead.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use arko_agents::terminal::{StartOptions, TerminalConfig, TerminalManager, TerminalTool};
use arko_agents::{TaskContext, Tool, ToolDefinition, ToolError, ToolTypeDetails};
use serde_json::{Value, json};

use super::recovery_tests::FakeClaude;
use super::*;
use crate::notes::{Note, NotesHandle};
use crate::test_support::{
    ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};

/// How long each wait asks for. A wait that runs its time fails the test.
const WAIT_MS: u64 = 20_000;

/// A wait that took less ended for what arrived.
const ENDED_EARLY: Duration = Duration::from_secs(10);

/// How long a test lets a wait run before it posts what should end it.
const WAITING: Duration = Duration::from_millis(300);

/// A tool whose call starts a job that prints nothing for ten minutes and
/// waits on it with the real `terminal_control wait`, as an agent does. It
/// records how long each wait took. The jobs run on until the tool is
/// dropped, so no job's end reaches the agent meanwhile.
#[derive(Debug)]
struct WaitsOnJob {
    jobs: Arc<TerminalManager>,
    control: TerminalTool,
    /// How many waits started.
    started: tokio::sync::watch::Sender<usize>,
    waited: Mutex<Vec<Duration>>,
    /// Each job's owner and id.
    started_jobs: Mutex<Vec<(uuid::Uuid, uuid::Uuid)>>,
}

impl WaitsOnJob {
    fn new(jobs: &Arc<TerminalManager>) -> Arc<Self> {
        let control = TerminalTool::all(jobs.clone())
            .pop()
            .expect("terminal_control comes last");
        assert_eq!(control.name(), "terminal_control");
        Arc::new(Self {
            jobs: jobs.clone(),
            control,
            started: tokio::sync::watch::Sender::new(0),
            waited: Mutex::new(Vec::new()),
            started_jobs: Mutex::new(Vec::new()),
        })
    }

    /// Once wait `count` (from 1) has started.
    async fn waiting(&self, count: usize) {
        let mut started = self.started.subscribe();
        within(started.wait_for(|started| *started >= count))
            .await
            .unwrap();
    }

    fn waited(&self) -> Vec<Duration> {
        self.waited.lock().unwrap().clone()
    }
}

/// The jobs stop with the test, also one that failed.
impl Drop for WaitsOnJob {
    fn drop(&mut self) {
        let started = self
            .started_jobs
            .get_mut()
            .map(std::mem::take)
            .unwrap_or_default();
        for (owner, job) in started {
            let _ = self.jobs.cancel(owner, job);
        }
    }
}

#[async_trait::async_trait]
impl Tool for WaitsOnJob {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "waits_on_job".into(),
            description: "Waits on a quiet job.".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    fn uses_host_timeout(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let job = self
            .jobs
            .start(
                context,
                "sleep 600",
                StartOptions {
                    terminal: false,
                    detect_prompts: false,
                    ..Default::default()
                },
            )
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        self.started_jobs
            .lock()
            .unwrap()
            .push((context.task_id, job.job_id));
        let params = HashMap::from([
            ("action".to_string(), json!("wait")),
            ("job_id".to_string(), json!(job.job_id.to_string())),
            ("revision".to_string(), json!(job.revision)),
            ("wait_ms".to_string(), json!(WAIT_MS)),
        ]);
        self.started.send_modify(|started| *started += 1);
        let started = Instant::now();
        let result = self.control.execute(context, params).await;
        self.waited.lock().unwrap().push(started.elapsed());
        result
    }
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(90), future)
        .await
        .expect("in time")
}

fn jobs(workspace: &Workspace) -> Arc<TerminalManager> {
    TerminalManager::new(TerminalConfig::new(workspace.0.join(".jobs"))).unwrap()
}

/// The operator's note for `agent`, as arko posts what its user types.
fn typed(text: &str, agent: &str) -> Note {
    Note::new(text).to([agent]).once()
}

/// Wait `waited` ended for what arrived. With `while_it_ran`, that came
/// [`WAITING`] after it started, so it didn't end at once.
fn ended_early(waited: Duration, while_it_ran: bool) {
    assert!(waited < ENDED_EARLY, "the wait ran {waited:?}");
    if while_it_ran {
        assert!(waited >= WAITING - Duration::from_millis(50), "{waited:?}");
    }
}

/// An agent in arko's runtime waits on a job twice. A note posted after the
/// model's request, before the first call starts, ends that wait at once. A
/// note posted during the second wait ends it. Each request after a wait
/// holds its note.
#[tokio::test]
async fn a_note_ends_a_native_agents_terminal_wait() {
    let workspace = Workspace::new();
    let notes: Arc<OnceLock<NotesHandle>> = Arc::default();
    let server = ResponsesServer::start({
        let notes = notes.clone();
        move |index, _| {
            Some(match index {
                0 => {
                    // After this request, before its call.
                    notes
                        .get()
                        .unwrap()
                        .post(typed("Before the call.", "writer"))
                        .unwrap();
                    tool_call_turn("c0", "waits_on_job", json!({}))
                }
                1 => tool_call_turn("c1", "waits_on_job", json!({})),
                _ => complete_turn("c2", "done"),
            })
        }
    })
    .await;
    let jobs = jobs(&workspace);
    let tool = WaitsOnJob::new(&jobs);
    let mut writer = agent_config("Writer");
    writer.tool_config.available_tool_names = vec!["waits_on_job".into()];
    let workflow = Arc::new(
        Workflow::builder()
            .name("arrival-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("writer", writer)
            .global_tool(tool.clone())
            .background_jobs(jobs.clone())
            .build()
            .unwrap(),
    );
    notes.set(workflow.notes_handle()).ok().unwrap();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("writer", "Wait on it.").await }
    });
    tool.waiting(2).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("During the wait.", "writer"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "done");
    let waited = tool.waited();
    assert_eq!(waited.len(), 2, "{waited:?}");
    ended_early(waited[0], false);
    ended_early(waited[1], true);
    let bodies = server.bodies();
    assert!(bodies[1].contains("Before the call."), "{}", bodies[1]);
    assert!(bodies[2].contains("During the wait."), "{}", bodies[2]);
}

/// The same in Claude Code, where the notes the user types reach the model
/// as their message after the call that waited. The first note comes after
/// a search's hook, before the CLI makes the call that waits.
#[tokio::test]
async fn a_note_ends_a_claude_code_agents_terminal_wait() {
    let fake = FakeClaude::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "mcp__arko__waits_on_job", "input": {}, "sleep_before": 1.5},
        {"tool": "mcp__arko__waits_on_job", "input": {}},
        {"tool": "mcp__arko__complete_task", "input": {"summary": "done"}},
    ]]));
    let workspace = Workspace::new();
    let jobs = jobs(&workspace);
    let tool = WaitsOnJob::new(&jobs);
    let mut writer = fake.agent();
    writer
        .tool_config
        .available_tool_names
        .extend(["Grep".to_string(), "waits_on_job".to_string()]);
    let workflow = Arc::new(
        Workflow::builder()
            .name("arrival-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("writer", writer)
            .global_tool(tool.clone())
            .background_jobs(jobs.clone())
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("writer", "Wait on it.").await }
    });
    within(async {
        while fake.logged("post").is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert_eq!(*tool.started.borrow(), 0, "posted before the call");
    workflow
        .post_note(typed("Before the call.", "writer"))
        .unwrap();
    tool.waiting(2).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("During the wait.", "writer"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "done", "{:?}", result.error);
    let waited = tool.waited();
    assert_eq!(waited.len(), 2, "{waited:?}");
    ended_early(waited[0], false);
    ended_early(waited[1], true);
    let typed: Vec<String> = fake
        .logged("midturn")
        .iter()
        .map(|entry| entry["text"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(typed.len(), 2, "{typed:#?}");
    assert!(typed[0].contains("Before the call."), "{typed:#?}");
    assert!(typed[1].contains("During the wait."), "{typed:#?}");
}

/// A worker's wait ends when its lead's message comes, and the worker's next
/// request holds the message.
#[tokio::test]
async fn a_leads_message_ends_a_workers_terminal_wait() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        let turn = body.matches("\"function_call_output\"").count();
        Some(match turn {
            0 => tool_call_turn("w0", "waits_on_job", json!({})),
            _ => complete_turn("w1", "worker done"),
        })
    })
    .await;
    let jobs = jobs(&workspace);
    let tool = WaitsOnJob::new(&jobs);
    let mut worker = agent_config("Worker");
    worker.tool_config.available_tool_names = vec!["waits_on_job".into()];
    let workflow = Workflow::builder()
        .name("arrival-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Lead"))
        .agent("worker", worker)
        .global_tool(tool.clone())
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
    let lead = uuid::Uuid::new_v4();
    ctx.subagents
        .start(
            ctx,
            lead,
            "lead",
            0,
            &workspace.0,
            "worker",
            "Wait on it.",
            None,
            false,
            None,
            None,
            std::collections::BTreeMap::new(),
        )
        .await
        .unwrap();
    tool.waiting(1).await;
    tokio::time::sleep(WAITING).await;
    ctx.subagents
        .message(ctx, lead, "r1", "Stop waiting.")
        .unwrap();
    within(async {
        while workflow.subagents().iter().any(|run| run.state.is_active()) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    let waited = tool.waited();
    assert_eq!(waited.len(), 1, "{waited:?}");
    ended_early(waited[0], true);
    let last = server.bodies().pop().unwrap();
    assert!(last.contains("Stop waiting."), "{last}");
}

/// A custom async tool that waits for an external answer (like an approval
/// tray) and selects on `context.context_arrived.arrived()`.
#[derive(Debug)]
struct WaitsOnApproval {
    started: tokio::sync::watch::Sender<usize>,
    waited: Mutex<Vec<(Duration, bool)>>,
}

impl WaitsOnApproval {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            started: tokio::sync::watch::Sender::new(0),
            waited: Mutex::new(Vec::new()),
        })
    }

    async fn waiting(&self, count: usize) {
        let mut started = self.started.subscribe();
        within(started.wait_for(|started| *started >= count))
            .await
            .unwrap();
    }

    fn waited(&self) -> Vec<(Duration, bool)> {
        self.waited.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl Tool for WaitsOnApproval {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "waits_on_approval".into(),
            description: "Waits for an approval or a note.".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    fn uses_host_timeout(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let (_answer_tx, answer_rx) = tokio::sync::oneshot::channel::<String>();
        self.started.send_modify(|started| *started += 1);
        let started = Instant::now();
        let reply = tokio::select! {
            biased;
            answer = answer_rx => answer.unwrap_or_else(|_| "unanswered".into()),
            () = context.context_arrived.arrived() => {
                if context.context_arrived.note_waits() {
                    "note woke approval".into()
                } else {
                    "update woke approval".into()
                }
            }
            () = context.stop.stopping() => "stopped".into(),
            _ = tokio::time::sleep(Duration::from_millis(WAIT_MS)) => "timed out".into(),
        };
        self.waited
            .lock()
            .unwrap()
            .push((started.elapsed(), context.context_arrived.note_waits()));
        Ok(reply)
    }
}

/// A custom async tool waiting on a channel and `context.context_arrived`
/// wakes on a note posted before the call and on one posted during the wait.
#[tokio::test]
async fn a_note_ends_a_custom_async_wait_tool() {
    let workspace = Workspace::new();
    let notes: Arc<OnceLock<NotesHandle>> = Arc::default();
    let server = ResponsesServer::start({
        let notes = notes.clone();
        move |index, _| {
            Some(match index {
                0 => {
                    notes
                        .get()
                        .unwrap()
                        .post(typed("Before approval.", "writer"))
                        .unwrap();
                    tool_call_turn("a0", "waits_on_approval", json!({}))
                }
                1 => tool_call_turn("a1", "waits_on_approval", json!({})),
                _ => complete_turn("a2", "done"),
            })
        }
    })
    .await;
    let tool = WaitsOnApproval::new();
    let mut writer = agent_config("Writer");
    writer.tool_config.available_tool_names = vec!["waits_on_approval".into()];
    let workflow = Arc::new(
        Workflow::builder()
            .name("arrival-approval-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("writer", writer)
            .global_tool(tool.clone())
            .build()
            .unwrap(),
    );
    notes.set(workflow.notes_handle()).ok().unwrap();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("writer", "Ask approval.").await }
    });
    tool.waiting(2).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("During approval.", "writer"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "done");
    let waited = tool.waited();
    assert_eq!(waited.len(), 2, "{waited:?}");
    ended_early(waited[0].0, false);
    assert!(waited[0].1, "note_waits true before call");
    ended_early(waited[1].0, true);
    assert!(waited[1].1, "note_waits true during wait");
    let bodies = server.bodies();
    assert!(bodies[1].contains("note woke approval"), "{}", bodies[1]);
    assert!(bodies[1].contains("Before approval."), "{}", bodies[1]);
    assert!(bodies[2].contains("note woke approval"), "{}", bodies[2]);
    assert!(bodies[2].contains("During approval."), "{}", bodies[2]);
}

/// The same custom async wait tool in Claude Code wakes on a note posted
/// before the call and on one posted during the wait, and each note is
/// delivered right after its call returns.
#[tokio::test]
async fn a_note_ends_a_custom_async_wait_tool_in_claude_code() {
    let fake = FakeClaude::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "mcp__arko__waits_on_approval", "input": {}, "sleep_before": 1.5},
        {"tool": "mcp__arko__waits_on_approval", "input": {}},
        {"tool": "mcp__arko__complete_task", "input": {"summary": "done"}},
    ]]));
    let workspace = Workspace::new();
    let tool = WaitsOnApproval::new();
    let mut writer = fake.agent();
    writer
        .tool_config
        .available_tool_names
        .extend(["Grep".to_string(), "waits_on_approval".to_string()]);
    let workflow = Arc::new(
        Workflow::builder()
            .name("arrival-approval-cc-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("writer", writer)
            .global_tool(tool.clone())
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("writer", "Ask approval.").await }
    });
    within(async {
        while fake.logged("post").is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    workflow
        .post_note(typed("Before cc approval.", "writer"))
        .unwrap();
    tool.waiting(2).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("During cc approval.", "writer"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "done", "{:?}", result.error);
    let waited = tool.waited();
    assert_eq!(waited.len(), 2, "{waited:?}");
    ended_early(waited[0].0, false);
    assert!(waited[0].1);
    ended_early(waited[1].0, true);
    assert!(waited[1].1);
    let typed: Vec<String> = fake
        .logged("midturn")
        .iter()
        .map(|entry| entry["text"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(typed.len(), 2, "{typed:#?}");
    assert!(typed[0].contains("Before cc approval."), "{typed:#?}");
    assert!(typed[1].contains("During cc approval."), "{typed:#?}");
}

type ObservedStart = (Duration, bool, Option<String>, bool);

/// A `SubagentHook` whose `prepare_start` awaits approval (or wakes on a
/// note), returning `StartOutcome::NotStarted` when woken by a note and
/// `StartOutcome::Start` when approved.
struct ApprovalStartHook {
    started: tokio::sync::watch::Sender<usize>,
    approve: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    seen: Mutex<Vec<ObservedStart>>,
}

impl ApprovalStartHook {
    fn new(approve: tokio::sync::oneshot::Receiver<()>) -> Arc<Self> {
        Arc::new(Self {
            started: tokio::sync::watch::Sender::new(0),
            approve: Mutex::new(Some(approve)),
            seen: Mutex::new(Vec::new()),
        })
    }

    async fn waiting(&self, count: usize) {
        let mut started = self.started.subscribe();
        within(started.wait_for(|started| *started >= count))
            .await
            .unwrap();
    }

    fn seen(&self) -> Vec<ObservedStart> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl crate::subagents::SubagentHook for ApprovalStartHook {
    fn prepare(
        &self,
        _request: &crate::subagents::StartRequest,
        _config: &mut arko_agents::AgentConfig,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn prepare_start(
        &self,
        _request: &crate::subagents::StartRequest,
        _config: &mut arko_agents::AgentConfig,
        call: &crate::subagents::StartCall,
    ) -> Result<crate::subagents::StartOutcome, String> {
        let attempt = *self.started.borrow() + 1;
        self.started.send_replace(attempt);
        let started = Instant::now();
        let approve = if attempt == 3 {
            self.approve.lock().unwrap().take()
        } else {
            None
        };
        let outcome = tokio::select! {
            biased;
            () = call.context_arrived.arrived() => {
                crate::subagents::StartOutcome::NotStarted(
                    "A note arrived while waiting to approve the worker. Read it first.".into(),
                )
            }
            () = call.stop.stopping() => {
                crate::subagents::StartOutcome::NotStarted("Stopped while waiting.".into())
            }
            _ = async {
                match approve {
                    Some(rx) => {
                        let _ = rx.await;
                    }
                    None => std::future::pending::<()>().await,
                }
            } => crate::subagents::StartOutcome::Start,
            _ = tokio::time::sleep(Duration::from_millis(WAIT_MS)) => {
                return Err("timed out waiting for approval".into());
            }
        };
        self.seen.lock().unwrap().push((
            started.elapsed(),
            call.context_arrived.note_waits(),
            call.call_id.clone(),
            call.tool_call_id.is_some(),
        ));
        Ok(outcome)
    }
}

/// `SubagentHook::prepare_start` receives `StartCall` (with `call_id`,
/// `tool_call_id`, `stop` and `context_arrived`), wakes on notes posted
/// before or during the wait, returns `StartOutcome::NotStarted` without
/// launching the subagent or failing the call, and starts the subagent once
/// approved with its handle recycled.
#[tokio::test]
async fn a_note_ends_a_subagent_start_hook_wait() {
    let workspace = Workspace::new();
    let notes: Arc<OnceLock<NotesHandle>> = Arc::default();
    let (approve_tx, approve_rx) = tokio::sync::oneshot::channel();
    let hook = ApprovalStartHook::new(approve_rx);
    let server = ResponsesServer::start({
        let notes = notes.clone();
        move |_, body| {
            let is_worker = body.contains("You are agent run r");
            if is_worker {
                return Some(complete_turn("w0", "worker ran"));
            }
            let turn = body.matches("\"function_call_output\"").count();
            Some(match turn {
                0 => {
                    notes
                        .get()
                        .unwrap()
                        .post(typed("Before start.", "lead"))
                        .unwrap();
                    tool_call_turn(
                        "s0",
                        "start_agent",
                        json!({"agent": "worker", "task": "Check it."}),
                    )
                }
                1 => tool_call_turn(
                    "s1",
                    "start_agent",
                    json!({"agent": "worker", "task": "Check it again."}),
                ),
                2 => tool_call_turn(
                    "s2",
                    "start_agent",
                    json!({"agent": "worker", "task": "Run now."}),
                ),
                _ => complete_turn("done", "all good"),
            })
        }
    })
    .await;
    let workflow = Arc::new(
        Workflow::builder()
            .name("start-hook-wake-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("lead", agent_config("Lead"))
            .agent("worker", agent_config("Worker"))
            .workflow_services(
                crate::services::WorkflowServicesConfig::default()
                    .allow_caller("lead")
                    .allow_target("worker"),
            )
            .subagents(crate::subagents::SubagentsConfig::default().hook(hook.clone()))
            .build()
            .unwrap(),
    );
    notes.set(workflow.notes_handle()).ok().unwrap();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Start a worker.").await }
    });
    hook.waiting(2).await;
    tokio::time::sleep(WAITING).await;
    workflow.post_note(typed("During start.", "lead")).unwrap();
    hook.waiting(3).await;
    assert!(
        workflow.subagents().is_empty(),
        "deferred starts launch no subagent"
    );
    approve_tx.send(()).unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "all good");
    let seen = hook.seen();
    assert_eq!(seen.len(), 3, "{seen:?}");
    ended_early(seen[0].0, false);
    assert!(seen[0].1 && seen[0].2.as_deref() == Some("call_s0") && seen[0].3);
    ended_early(seen[1].0, true);
    assert!(seen[1].1 && seen[1].2.as_deref() == Some("call_s1") && seen[1].3);
    assert!(seen[2].2.as_deref() == Some("call_s2") && seen[2].3);
    let runs = workflow.subagents();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].handle, "r1", "deferred starts release the handle");
    let bodies: Vec<String> = server
        .bodies()
        .into_iter()
        .filter(|body| !body.contains("You are agent run r"))
        .collect();
    assert!(bodies[1].contains("Before start."), "{}", bodies[1]);
    assert!(bodies[2].contains("During start."), "{}", bodies[2]);
}

/// In Claude Code, a `SubagentHook::prepare_start` woken by a note returns
/// `StartOutcome::NotStarted` as a non-error tool result, so the PostToolUse
/// hook delivers the note right away.
#[tokio::test]
async fn a_note_ends_a_subagent_start_hook_wait_in_claude_code() {
    let fake = FakeClaude::new(json!([[
        {"tool": "mcp__arko__start_agent", "input": {"agent": "worker", "task": "Check it."}},
        {"tool": "mcp__arko__complete_task", "input": {"summary": "saw note"}},
    ]]));
    let workspace = Workspace::new();
    let (_approve_tx, approve_rx) = tokio::sync::oneshot::channel();
    let hook = ApprovalStartHook::new(approve_rx);
    let workflow = Arc::new(
        Workflow::builder()
            .name("start-hook-cc-wake-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("lead", fake.agent())
            .agent("worker", agent_config("Worker"))
            .workflow_services(
                crate::services::WorkflowServicesConfig::default()
                    .allow_caller("lead")
                    .allow_target("worker"),
            )
            .subagents(crate::subagents::SubagentsConfig::default().hook(hook.clone()))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Start a worker.").await }
    });
    hook.waiting(1).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("Hold the worker.", "lead"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "saw note", "{:?}", result.error);
    let seen = hook.seen();
    assert_eq!(seen.len(), 1, "{seen:?}");
    ended_early(seen[0].0, true);
    assert!(seen[0].1 && seen[0].3);
    assert!(workflow.subagents().is_empty());
    let posts = fake.logged("post");
    let start_post = posts
        .iter()
        .find(|entry| entry["tool"] == "mcp__arko__start_agent")
        .expect("start_agent post hook");
    assert_eq!(start_post["is_error"], json!(false), "{start_post}");
    let typed: Vec<String> = fake
        .logged("midturn")
        .iter()
        .map(|entry| entry["text"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(typed.len(), 1, "{typed:#?}");
    assert!(typed[0].contains("Hold the worker."), "{typed:#?}");
}

/// A `UserInputHandler` whose `respond` awaits `request.context_arrived`.
struct WakingInputHandler {
    started: tokio::sync::watch::Sender<usize>,
    waited: Mutex<Vec<(Duration, bool)>>,
}

impl WakingInputHandler {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            started: tokio::sync::watch::Sender::new(0),
            waited: Mutex::new(Vec::new()),
        })
    }

    async fn waiting(&self, count: usize) {
        let mut started = self.started.subscribe();
        within(started.wait_for(|started| *started >= count))
            .await
            .unwrap();
    }

    fn waited(&self) -> Vec<(Duration, bool)> {
        self.waited.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl crate::user_input::UserInputHandler for WakingInputHandler {
    async fn respond(
        &self,
        request: crate::user_input::UserInputRequest,
    ) -> Result<String, String> {
        self.started.send_modify(|started| *started += 1);
        let started = Instant::now();
        let (_tx, rx) = tokio::sync::oneshot::channel::<String>();
        let reply = tokio::select! {
            biased;
            ans = rx => ans.unwrap_or_else(|_| "unanswered".into()),
            () = request.context_arrived.arrived() => {
                "A note arrived while waiting for your answer.".into()
            }
            _ = tokio::time::sleep(Duration::from_millis(WAIT_MS)) => "timed out".into(),
        };
        self.waited
            .lock()
            .unwrap()
            .push((started.elapsed(), request.context_arrived.note_waits()));
        Ok(reply)
    }
}

/// `UserInputRequest::context_arrived` wakes a `UserInputHandler` on notes
/// posted before and during the question wait, and the agent reads each note
/// in its next request.
#[tokio::test]
async fn a_note_ends_a_user_input_handler_wait() {
    let workspace = Workspace::new();
    let notes: Arc<OnceLock<NotesHandle>> = Arc::default();
    let server = ResponsesServer::start({
        let notes = notes.clone();
        move |index, _| {
            Some(match index {
                0 => {
                    notes
                        .get()
                        .unwrap()
                        .post(typed("Before question.", "lead"))
                        .unwrap();
                    tool_call_turn("q0", "ask_user", json!({"questions": "Tabs or spaces?"}))
                }
                1 => tool_call_turn("q1", "ask_user", json!({"questions": "Push now?"})),
                _ => complete_turn("done", "read notes"),
            })
        }
    })
    .await;
    let handler = WakingInputHandler::new();
    let mut lead = agent_config("Lead");
    lead.tool_config
        .available_tool_names
        .push("ask_user".into());
    let workflow = Arc::new(
        Workflow::builder()
            .name("input-wake-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("lead", lead)
            .user_input_handler(handler.clone())
            .build()
            .unwrap(),
    );
    notes.set(workflow.notes_handle()).ok().unwrap();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Ask the user.").await }
    });
    handler.waiting(2).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("During question.", "lead"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "read notes");
    let waited = handler.waited();
    assert_eq!(waited.len(), 2, "{waited:?}");
    ended_early(waited[0].0, false);
    assert!(waited[0].1);
    ended_early(waited[1].0, true);
    assert!(waited[1].1);
    let bodies = server.bodies();
    assert!(bodies[1].contains("Before question."), "{}", bodies[1]);
    assert!(bodies[2].contains("During question."), "{}", bodies[2]);
}

/// The same in Claude Code: when a Claude Code agent parks on `ask_user`,
/// posting a note wakes `UserInputHandler::respond` via `context_arrived`,
/// and the note reaches Claude Code with its next turn.
#[tokio::test]
async fn a_note_ends_a_user_input_handler_wait_in_claude_code() {
    let fake = FakeClaude::new(json!([
        [
            {"tool": "mcp__arko__ask_user", "input": {"questions": ["Tabs or spaces?"]}},
            {"text": "Waiting for your answer."},
        ],
        [
            {"tool": "mcp__arko__complete_task", "input": {"summary": "got note"}},
        ],
    ]));
    let workspace = Workspace::new();
    let handler = WakingInputHandler::new();
    let mut lead = fake.agent();
    lead.tool_config
        .available_tool_names
        .push("ask_user".into());
    lead.tool_config.async_questions = true;
    let workflow = Arc::new(
        Workflow::builder()
            .name("input-cc-wake-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("lead", lead)
            .user_input_handler(handler.clone())
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("lead", "Ask the user.").await }
    });
    handler.waiting(1).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("Use two spaces.", "lead"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "got note", "{:?}", result.error);
    let waited = handler.waited();
    assert_eq!(waited.len(), 1, "{waited:?}");
    ended_early(waited[0].0, true);
    assert!(waited[0].1);
    let users: Vec<String> = fake
        .logged("user")
        .iter()
        .map(|entry| entry.to_string())
        .collect();
    assert!(
        users.iter().any(|entry| entry.contains("Use two spaces.")),
        "{users:#?}"
    );
}

/// A `ToolApprovalHandler` whose `decide` wakes on `request.context_arrived`.
struct WakingApprovalHandler {
    started: tokio::sync::watch::Sender<usize>,
    waited: Mutex<Vec<(Duration, bool)>>,
}

impl WakingApprovalHandler {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            started: tokio::sync::watch::Sender::new(0),
            waited: Mutex::new(Vec::new()),
        })
    }

    async fn waiting(&self, count: usize) {
        let mut started = self.started.subscribe();
        within(started.wait_for(|started| *started >= count))
            .await
            .unwrap();
    }

    fn waited(&self) -> Vec<(Duration, bool)> {
        self.waited.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl crate::policy::ToolApprovalHandler for WakingApprovalHandler {
    async fn decide(
        &self,
        request: crate::policy::ToolApprovalRequest,
    ) -> crate::policy::ToolApprovalOutcome {
        self.started.send_modify(|started| *started += 1);
        let started = Instant::now();
        let (_tx, rx) = tokio::sync::oneshot::channel::<bool>();
        let outcome = tokio::select! {
            biased;
            _ = rx => crate::policy::ToolApprovalOutcome::Approve,
            () = request.context_arrived.arrived() => {
                crate::policy::ToolApprovalOutcome::Reject {
                    reason: Some("A note arrived while waiting for approval.".into()),
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(WAIT_MS)) => {
                crate::policy::ToolApprovalOutcome::Reject {
                    reason: Some("timed out".into()),
                }
            }
        };
        self.waited
            .lock()
            .unwrap()
            .push((started.elapsed(), request.context_arrived.note_waits()));
        outcome
    }
}

/// `ToolApprovalRequest::context_arrived` wakes a `ToolApprovalHandler` when
/// a note arrives, and the agent reads the note in its next request.
#[tokio::test]
async fn a_note_ends_a_tool_approval_handler_wait() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn(
                "t0",
                "write_file",
                json!({"path": "hello.txt", "content": "hi"}),
            ),
            _ => complete_turn("done", "read note"),
        })
    })
    .await;
    let handler = WakingApprovalHandler::new();
    let mut writer = agent_config("Writer");
    writer.tool_config.available_tool_names = vec!["write_file".into()];
    writer.tool_config.default_auto_approve_tool_use = false;
    writer.tool_config.tool_specific_configs.insert(
        "complete_task".into(),
        arko_agents::PerToolConfigOverride {
            auto_approve_tool_use: Some(true),
            ..Default::default()
        },
    );
    let workflow = Arc::new(
        Workflow::builder()
            .name("tool-approval-wake-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("writer", writer)
            .approval_policy(crate::policy::ApprovalPolicy::Handler(handler.clone()))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("writer", "Write a file.").await }
    });
    handler.waiting(1).await;
    tokio::time::sleep(WAITING).await;
    workflow
        .post_note(typed("Skip writing the file.", "writer"))
        .unwrap();
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.summary, "read note");
    let waited = handler.waited();
    assert_eq!(waited.len(), 1, "{waited:?}");
    ended_early(waited[0].0, true);
    assert!(waited[0].1);
    let bodies = server.bodies();
    assert!(
        bodies[1].contains("Skip writing the file."),
        "{}",
        bodies[1]
    );
    assert!(!workspace.0.join("hello.txt").exists());
}
