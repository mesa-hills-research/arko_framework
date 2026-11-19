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
