//! Calls that wait while context arrives for their task, and calls that
//! move to the background.

use super::tool_call_tests::{
    agent, assert_calls_paired, complete, config, native_call, result_for, text_turn, turn,
    wait_for, workspace,
};
use super::*;
use crate::terminal::{IN_THE_BACKGROUND, JobKind, TerminalConfig, TerminalManager};
use crate::tool::{Tool, ToolBackground, ToolError, ToolTypeDetails};
use async_trait::async_trait;
use tokio::sync::Notify;

/// A tool that waits until context arrives for its task, or `limit`.
#[derive(Debug)]
struct WaitsForContext {
    limit: Duration,
}

#[async_trait]
impl Tool for WaitsForContext {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "waits_for_context".into(),
            description: "Waits for context".into(),
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
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        match timeout(self.limit, context.context_arrived.arrived()).await {
            Ok(()) => Ok("context arrived".into()),
            Err(_) => Ok("no context".into()),
        }
    }
}

fn waits_for_context(limit: Duration) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(WaitsForContext { limit }));
    registry
}

/// Context sent while a call waits ends the wait, and reaches the model with
/// the call's result.
#[tokio::test]
async fn context_sent_while_a_call_waits_ends_the_wait() {
    let workspace = workspace();
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &["waits_for_context"]),
        waits_for_context(Duration::from_secs(60)),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "wait",
            "waits_for_context",
            serde_json::json!({}),
        )]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "waits_for_context")
            .then_some(())
    })
    .await;
    commands
        .send(UserCommand::InjectContext {
            content: "The build finished.".into(),
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .expect("the wait ends when the context arrives")
        .unwrap()
        .unwrap();
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "wait")[0].output,
        "context arrived"
    );
    let results = task
        .conversation
        .messages
        .iter()
        .position(|message| result_for_message(message, "wait"))
        .unwrap();
    // The context follows the result, before the next request.
    assert!(
        task.conversation.messages[results..]
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(ai::Content::as_text)
            .any(|text| text.contains("The build finished."))
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

fn result_for_message(message: &ai::Message, call_id: &str) -> bool {
    message
        .content
        .iter()
        .filter_map(ai::Content::as_tool_result)
        .any(|result| result.call_id.as_deref() == Some(call_id))
}

/// Context taken before a call starts is in the conversation already, so it
/// doesn't end the call's wait.
#[tokio::test]
async fn context_taken_before_a_call_doesnt_end_its_wait() {
    let workspace = workspace();
    let (mut agent, _events, commands) = agent(
        config(ToolCallingMode::Native, &["waits_for_context"]),
        waits_for_context(Duration::from_millis(300)),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "wait",
            "waits_for_context",
            serde_json::json!({}),
        )]),
        complete(),
    ]);
    commands
        .send(UserCommand::InjectContext {
            content: "Sent before the call.".into(),
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result_for(&task.conversation, "wait")[0].output,
        "no context"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A tool that runs until the test releases it, or its stop signal fires.
/// An `Own` one also returns once it is moved.
#[derive(Debug)]
struct Slow {
    name: &'static str,
    mode: ToolBackground,
    parallel_safe: bool,
    started: Notify,
    release: Notify,
    /// Why its stop signal fired, once it did.
    stopped: std::sync::Mutex<Option<String>>,
}

impl Slow {
    fn new(name: &'static str, mode: ToolBackground) -> Arc<Self> {
        Arc::new(Self {
            name,
            mode,
            parallel_safe: false,
            started: Notify::new(),
            release: Notify::new(),
            stopped: Default::default(),
        })
    }
}

#[async_trait]
impl Tool for Slow {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.into(),
            description: "Runs until released".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: self.parallel_safe,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    fn background_mode(&self) -> ToolBackground {
        self.mode
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.started.notify_one();
        tokio::select! {
            () = self.release.notified() => Ok(format!("{} done", self.name)),
            () = context.moved.moved() => Ok(format!("{} moved itself", self.name)),
            () = context.stop.stopping() => {
                *self.stopped.lock().unwrap() = context.stop.reason();
                Ok(format!("{} wound down", self.name))
            }
        }
    }
}

/// A tool that says it started, then waits for context.
#[derive(Debug, Default)]
struct Listens {
    started: Notify,
}

#[async_trait]
impl Tool for Listens {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "listens".into(),
            ..WaitsForContext {
                limit: Duration::ZERO,
            }
            .definition()
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.started.notify_one();
        match timeout(Duration::from_secs(120), context.context_arrived.arrived()).await {
            Ok(()) => Ok("context arrived".into()),
            Err(_) => Ok("no context".into()),
        }
    }
}

/// A terminal manager in a fresh directory under `workspace`.
fn jobs(workspace: &std::path::Path) -> Arc<TerminalManager> {
    TerminalManager::new(TerminalConfig::new(workspace.join(".jobs"))).unwrap()
}

/// Waits until `jobs` lists a call that can move, and returns it.
async fn listed(jobs: &TerminalManager) -> crate::terminal::ForegroundCall {
    timeout(Duration::from_secs(10), async {
        loop {
            if let Some(call) = jobs.foreground_calls().pop() {
                return call;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("a call that can move")
}

/// The text of every message of `conversation`.
fn texts(conversation: &ai::Conversation) -> String {
    conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The job a moved result names.
fn job_of(result: &str) -> Uuid {
    result
        .strip_prefix(IN_THE_BACKGROUND)
        .unwrap_or_else(|| panic!("not a moved result: {result}"))
        .split(',')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

/// A `Detach` call the user moves gets a result naming its job at once, and
/// is settled. Its real result comes as a note when the job ends, which ends
/// a later wait.
#[tokio::test]
async fn a_moved_call_answers_at_once_and_its_result_comes_as_a_note() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let slow = Slow::new("slow", ToolBackground::Detach);
    let listens = Arc::new(Listens::default());
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    registry.register(listens.clone());
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &["slow", "listens"]),
        registry,
        &workspace,
    );
    agent.background_jobs = Some(jobs.clone());
    agent.bind_tools(&commands);
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("slow", "slow", serde_json::json!({}))]),
        turn(vec![native_call(
            "listen",
            "listens",
            serde_json::json!({}),
        )]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let call = listed(&jobs).await;
    assert_eq!((call.owner, call.tool_name.as_str()), (task_id, "slow"));
    assert_eq!(jobs.move_to_background(task_id, None), 1);
    timeout(Duration::from_secs(10), listens.started.notified())
        .await
        .unwrap();
    assert!(jobs.foreground_calls().is_empty());
    let running = jobs.jobs();
    assert_eq!(running.len(), 1);
    assert_eq!(
        (running[0].kind.clone(), running[0].tool_call_id),
        (
            JobKind::Call {
                tool_name: "slow".into()
            },
            Some(call.tool_call_id)
        )
    );
    slow.release.notify_one();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .expect("the note ends the wait")
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert!(task.running_tool_calls.is_empty());
    let moved = &result_for(&task.conversation, "slow")[0];
    assert!(!moved.is_error);
    let job = job_of(&moved.output);
    assert_eq!(job, running[0].job_id);
    assert!(
        moved.output.contains(
            "\nThe user moved this call to the background. It goes on running until it ends or reaches its time limit of 20 minutes. Its result reaches you as a note when it ends."
        ),
        "{}",
        moved.output
    );
    assert!(!moved.output.contains("terminal_status"), "not offered");
    assert_eq!(
        result_for(&task.conversation, "listen")[0].output,
        "context arrived"
    );
    assert!(texts(&task.conversation).contains(&format!(
        "Job {job} ended. It ran your slow call, which moved to the background 0 s ago. Its result follows.\n\nslow done"
    )));
    // The call's events have the moved result.
    let mut seen = Vec::new();
    let output = wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::ToolResult {
            tool_name, output, ..
        } if tool_name == "slow" => Some(output.clone()),
        _ => None,
    })
    .await;
    assert_eq!(output, moved.output);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Once a call moved, cancelling its task doesn't stop it. Stopping its job
/// does, through its stop signal.
#[tokio::test]
async fn a_moved_call_outlives_its_turn_and_stops_with_its_job() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let slow = Slow::new("slow", ToolBackground::Detach);
    let listens = Arc::new(Listens::default());
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    registry.register(listens.clone());
    let mut config = config(ToolCallingMode::Native, &["slow", "listens"]);
    config.tool_config.tool_timeout_seconds = None;
    let (mut agent, _events, commands) = agent(config, registry, &workspace);
    agent.background_jobs = Some(jobs.clone());
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("slow", "slow", serde_json::json!({}))]),
        turn(vec![native_call(
            "listen",
            "listens",
            serde_json::json!({}),
        )]),
    ]);
    let run = tokio::spawn(agent.run());
    listed(&jobs).await;
    jobs.move_to_background(task_id, None);
    timeout(Duration::from_secs(10), listens.started.notified())
        .await
        .unwrap();
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    let job = job_of(&result_for(&task.conversation, "slow")[0].output);
    assert!(
        slow.stopped.lock().unwrap().is_none(),
        "the task's stop doesn't reach it"
    );
    assert_eq!(jobs.jobs()[0].job_id, job);
    assert_eq!(
        jobs.status(task_id, job, None).unwrap().state,
        crate::terminal::JobState::Running
    );

    jobs.cancel(task_id, job).unwrap();
    let ended = timeout(Duration::from_secs(10), async {
        loop {
            let status = jobs.status(task_id, job, None).unwrap();
            if status.state.is_terminal() {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(ended.state, crate::terminal::JobState::Cancelled);
    assert_eq!(ended.output.text, "slow wound down");
    assert_eq!(
        slow.stopped.lock().unwrap().as_deref(),
        Some("The job was stopped.")
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A moved call doesn't keep its manager alive. Dropping the host's last
/// handle stops the call, as it stops the manager's other jobs, and frees
/// the manager's directory.
#[tokio::test]
async fn dropping_the_manager_stops_a_moved_call() {
    let workspace = workspace();
    let directory = workspace.join(".jobs");
    let jobs = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    let slow = Slow::new("slow", ToolBackground::Detach);
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    let mut config = config(ToolCallingMode::Native, &["slow"]);
    config.tool_config.tool_timeout_seconds = None;
    let (mut agent, _events, _commands) = agent(config, registry, &workspace);
    agent.background_jobs = Some(jobs.clone());
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("slow", "slow", serde_json::json!({}))]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    listed(&jobs).await;
    assert_eq!(jobs.move_to_background(task_id, None), 1);
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let job = job_of(&result_for(&task.conversation, "slow")[0].output);
    drop(task);
    let manager = Arc::downgrade(&jobs);
    drop(jobs);
    assert!(manager.upgrade().is_none(), "the moved call holds it");
    let state = || -> String {
        let record: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join(job.to_string()).join("job.json")).unwrap(),
        )
        .unwrap();
        record["state"].as_str().unwrap().to_string()
    };
    timeout(Duration::from_secs(10), async {
        while state() == "running" {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the call's job ends");
    assert_eq!(state(), "cancelled");
    assert_eq!(
        slow.stopped.lock().unwrap().as_deref(),
        Some("The job was stopped.")
    );
    TerminalManager::new(TerminalConfig::new(&directory)).expect("the directory is free");
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A `background_after` too long to count from a call's start means the
/// call never moves by itself.
#[tokio::test]
async fn a_huge_background_after_never_moves_a_call() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    jobs.set_background_after(Some(Duration::MAX));
    let slow = Slow::new("slow", ToolBackground::Detach);
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["slow"]),
        registry,
        &workspace,
    );
    agent.background_jobs = Some(jobs.clone());
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("slow", "slow", serde_json::json!({}))]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    listed(&jobs).await;
    slow.release.notify_one();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .expect("the run doesn't panic")
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(
        result_for(&task.conversation, "slow")[0].output,
        "slow done"
    );
    assert!(jobs.jobs().is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// An `Own` tool sees the move and returns its own result. Its call becomes
/// no job.
#[tokio::test]
async fn an_own_tool_sees_the_move_and_returns() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let mut registry = ToolRegistry::new();
    registry.register(Slow::new("slow", ToolBackground::Own));
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["slow"]),
        registry,
        &workspace,
    );
    agent.background_jobs = Some(jobs.clone());
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("slow", "slow", serde_json::json!({}))]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    listed(&jobs).await;
    assert_eq!(jobs.move_to_background(task_id, None), 1);
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        result_for(&task.conversation, "slow")[0].output,
        "slow moved itself"
    );
    assert!(jobs.jobs().is_empty() && jobs.foreground_calls().is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Lets the calls run once the test opens `gate`.
#[derive(Debug, Default)]
struct Gate {
    open: Notify,
    waiting: Notify,
}

#[async_trait]
impl crate::tool::ToolExecutionMiddleware for Gate {
    async fn before(
        &self,
        _: &crate::tool::ToolExecutionContext<'_>,
        _: &crate::tool::ToolCallContext<'_>,
    ) -> Result<crate::tool::ToolExecutionDecision, ToolError> {
        self.waiting.notify_one();
        self.open.notified().await;
        Ok(crate::tool::ToolExecutionDecision::Continue)
    }

    async fn after(
        &self,
        _: &crate::tool::ToolExecutionContext<'_>,
        _: &crate::tool::ToolCallContext<'_>,
        _: crate::tool::ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError> {
        Ok(())
    }
}

/// A call is listed once the middleware lets it run, and a call of a tool
/// that can't move isn't listed.
#[tokio::test]
async fn only_running_calls_that_can_move_are_listed() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let never = Slow::new("never", ToolBackground::Never);
    let detach = Slow::new("detach", ToolBackground::Detach);
    let mut registry = ToolRegistry::new();
    registry.register(never.clone());
    registry.register(detach.clone());
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["never", "detach"]),
        registry,
        &workspace,
    );
    let gate = Arc::new(Gate::default());
    agent.background_jobs = Some(jobs.clone());
    agent.execution_middleware = Some(gate.clone());
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("never", "never", serde_json::json!({}))]),
        turn(vec![native_call("detach", "detach", serde_json::json!({}))]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    timeout(Duration::from_secs(10), gate.waiting.notified())
        .await
        .unwrap();
    gate.open.notify_one();
    timeout(Duration::from_secs(10), never.started.notified())
        .await
        .unwrap();
    assert!(jobs.foreground_calls().is_empty(), "it can't move");
    never.release.notify_one();
    timeout(Duration::from_secs(10), gate.waiting.notified())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        jobs.foreground_calls().is_empty(),
        "the middleware hasn't let it run"
    );
    gate.open.notify_one();
    assert_eq!(listed(&jobs).await.tool_name, "detach");
    detach.release.notify_one();
    // complete_task.
    timeout(Duration::from_secs(10), gate.waiting.notified())
        .await
        .unwrap();
    gate.open.notify_one();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        result_for(&task.conversation, "detach")[0].output,
        "detach done"
    );
    assert!(jobs.foreground_calls().is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// One call of a group running at the same time moves, and the others keep
/// their results.
#[tokio::test]
async fn a_call_of_a_parallel_group_moves_alone() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let parallel = |name| {
        Arc::new(Slow {
            parallel_safe: true,
            ..Arc::into_inner(Slow::new(name, ToolBackground::Detach)).unwrap()
        })
    };
    let (first, second) = (parallel("first"), parallel("second"));
    let mut registry = ToolRegistry::new();
    registry.register(first.clone());
    registry.register(second.clone());
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["first", "second"]),
        registry,
        &workspace,
    );
    agent.background_jobs = Some(jobs.clone());
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![
            native_call("first", "first", serde_json::json!({})),
            native_call("second", "second", serde_json::json!({})),
        ]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    for slow in [&first, &second] {
        timeout(Duration::from_secs(10), slow.started.notified())
            .await
            .unwrap();
    }
    let calls = jobs.foreground_calls();
    assert_eq!(calls.len(), 2);
    let moving = calls
        .iter()
        .find(|call| call.tool_name == "first")
        .unwrap()
        .tool_call_id;
    assert_eq!(jobs.move_to_background(task_id, Some(moving)), 1);
    second.release.notify_one();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_calls_paired(&task.conversation);
    let job = job_of(&result_for(&task.conversation, "first")[0].output);
    assert_eq!(
        result_for(&task.conversation, "second")[0].output,
        "second done"
    );
    first.release.notify_one();
    timeout(Duration::from_secs(10), async {
        while !jobs.status(task_id, job, None).unwrap().state.is_terminal() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        jobs.status(task_id, job, None).unwrap().output.text,
        "first done"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A call moves by itself after the manager's `background_after`, and the
/// host's time limit still stops it, counted from its start.
#[tokio::test(start_paused = true)]
async fn a_long_call_moves_by_itself_and_keeps_its_time_limit() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    jobs.set_background_after(Some(Duration::from_secs(10)));
    let slow = Slow::new("slow", ToolBackground::Detach);
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    let mut config = config(ToolCallingMode::Native, &["slow"]);
    config.tool_config.tool_timeout_seconds = Some(60);
    let (mut agent, _events, commands) = agent(config, registry, &workspace);
    agent.background_jobs = Some(jobs.clone());
    agent.bind_tools(&commands);
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("slow", "slow", serde_json::json!({}))]),
        complete(),
    ]);
    let started = tokio::time::Instant::now();
    let task = timeout(Duration::from_secs(30), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(started.elapsed() >= Duration::from_secs(10));
    let moved = &result_for(&task.conversation, "slow")[0].output;
    assert!(
        moved.contains(
            "\nIt ran longer than 10 seconds, so it moved to the background. It goes on running until it ends or reaches its time limit of 1 minute."
        ),
        "{moved}"
    );
    let job = job_of(moved);
    let ended = timeout(Duration::from_secs(120), async {
        loop {
            let status = jobs.status(task_id, job, None).unwrap();
            if status.state.is_terminal() {
                return status;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    })
    .await
    .unwrap();
    assert!(started.elapsed() >= Duration::from_secs(60));
    assert!(
        started.elapsed() < Duration::from_secs(70),
        "from its start"
    );
    assert_eq!(ended.state, crate::terminal::JobState::Failed);
    assert!(
        ended
            .output
            .text
            .starts_with("Tool 'slow' execution failed"),
        "{}",
        ended.output.text
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// An output-based call moves the same way.
#[tokio::test]
async fn an_output_based_call_moves() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let slow = Slow::new("slow", ToolBackground::Detach);
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::OutputBased, &["slow"]),
        registry,
        &workspace,
    );
    agent.background_jobs = Some(jobs.clone());
    let task_id = agent.task.id;
    agent.scripted_responses.get_mut().unwrap().extend([
        text_turn("<slow>\n</slow>"),
        text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
    ]);
    let run = tokio::spawn(agent.run());
    listed(&jobs).await;
    assert_eq!(jobs.move_to_background(task_id, None), 1);
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(texts(&task.conversation).contains(IN_THE_BACKGROUND));
    assert_eq!(jobs.jobs().len(), 1);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A command template's calls can move, since its command goes on as a job.
/// The built-in tools' can't.
#[test]
fn command_template_tools_can_move_and_built_ins_cant() {
    let mut registry = ToolRegistry::new();
    registry
        .load_end_user_tool(ToolDefinition {
            name: "build".into(),
            description: "Builds".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::EndUserDefined {
                command_template: "make".into(),
            },
            raw_json_schema_override: None,
        })
        .unwrap();
    assert_eq!(
        registry.get("build").unwrap().background_mode(),
        ToolBackground::Detach
    );
    for name in registry.list_tool_names() {
        if name != "build" {
            assert_eq!(
                registry.get(&name).unwrap().background_mode(),
                ToolBackground::Never,
                "{name}"
            );
        }
    }
}

/// A tool that leaves work on another task, which watches the call's stop
/// signal, and waits.
#[derive(Debug, Default)]
struct LeavesWork {
    started: Notify,
    /// Why the left work was told to stop, once it was.
    stopped: Arc<std::sync::Mutex<Option<String>>>,
}

#[async_trait]
impl Tool for LeavesWork {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "leaves_work".into(),
            ..Slow::new("leaves_work", ToolBackground::Detach).definition()
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
        let (stop, stopped) = (context.stop.clone(), self.stopped.clone());
        tokio::spawn(async move {
            stop.stopping().await;
            *stopped.lock().unwrap() = stop.reason();
        });
        self.started.notify_one();
        std::future::pending().await
    }
}

/// Cancelling a task drops a movable call running in the foreground, and its
/// stop signal still fires, so work it left elsewhere stops too.
#[tokio::test]
async fn a_cancelled_movable_call_still_sees_its_stop() {
    let workspace = workspace();
    let jobs = jobs(&workspace);
    let tool = Arc::new(LeavesWork::default());
    let mut registry = ToolRegistry::new();
    registry.register(tool.clone());
    let (mut agent, _events, commands) = agent(
        config(ToolCallingMode::Native, &["leaves_work"]),
        registry,
        &workspace,
    );
    agent.background_jobs = Some(jobs.clone());
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![native_call(
            "work",
            "leaves_work",
            serde_json::json!({}),
        )]));
    let run = tokio::spawn(agent.run());
    timeout(Duration::from_secs(10), tool.started.notified())
        .await
        .unwrap();
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    timeout(Duration::from_secs(10), async {
        while tool.stopped.lock().unwrap().is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the left work hears of the stop");
    assert_eq!(
        tool.stopped.lock().unwrap().as_deref(),
        Some("Task cancelled.")
    );
    assert!(jobs.foreground_calls().is_empty() && jobs.jobs().is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}
