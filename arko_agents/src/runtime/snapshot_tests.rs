//! Task snapshots at the loop's safe points, and tasks saved while tool
//! calls ran that run again (recovery of open calls).

use super::image_tests::response;
use super::tool_call_tests::{
    agent, agent_for, assert_calls_paired, complete, config, drain, native_call, result_for,
    text_turn, turn, wait_for, workspace,
};
use super::*;
use crate::snapshot::{TaskSnapshotSink, TaskSnapshots};
use crate::task::RunningToolCall;
use crate::tool::{ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;
use std::sync::Mutex;

/// Keeps every snapshot it is given, with its `durable` flag.
#[derive(Default)]
struct Recorder {
    snapshots: Mutex<Vec<(Task, bool)>>,
    /// Refuses every snapshot.
    failing: bool,
}

impl Recorder {
    fn snapshots(&self) -> Vec<(Task, bool)> {
        self.snapshots.lock().unwrap().clone()
    }

    fn last(&self) -> Option<(Task, bool)> {
        self.snapshots.lock().unwrap().last().cloned()
    }
}

#[async_trait]
impl TaskSnapshotSink for Recorder {
    async fn task_changed(&self, task: &Task, durable: bool) -> Result<(), String> {
        self.snapshots.lock().unwrap().push((task.clone(), durable));
        if self.failing {
            return Err("disk full".into());
        }
        Ok(())
    }
}

/// What a probe saw when it ran.
#[derive(Debug, Clone)]
struct Seen {
    tool_call_id: Option<Uuid>,
    recovering: bool,
    /// The sink's latest snapshot as the call started.
    snapshot: Option<(Task, bool)>,
}

/// A tool that records each call it runs. `Resume` probes answer
/// "resumed" when recovering.
#[derive(Debug)]
struct Probe {
    name: &'static str,
    mode: ToolRecovery,
    parallel_safe: bool,
    seen: Arc<Mutex<Vec<Seen>>>,
    recorder: Option<Arc<Recorder>>,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Recorder")
    }
}

impl Probe {
    fn new(name: &'static str, mode: ToolRecovery) -> Self {
        Self {
            name,
            mode,
            parallel_safe: false,
            seen: Arc::default(),
            recorder: None,
        }
    }
}

#[async_trait]
impl Tool for Probe {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.into(),
            description: "Records its calls".into(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: "what".into(),
                description: "Anything".into(),
                param_type: ToolParameterType::String,
                required: false,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: self.parallel_safe,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    fn recovery_mode(&self) -> ToolRecovery {
        self.mode
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.seen.lock().unwrap().push(Seen {
            tool_call_id: context.tool_call_id,
            recovering: context.recovering,
            snapshot: self.recorder.as_ref().and_then(|recorder| recorder.last()),
        });
        let what = params
            .get("what")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Ok(if self.mode == ToolRecovery::Resume && context.recovering {
            format!("{} resumed {what}", self.name)
        } else {
            format!("{} ran {what}", self.name)
        })
    }
}

fn registry(probes: Vec<Arc<Probe>>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for probe in probes {
        registry.register(probe);
    }
    registry
}

fn with_sink(agent: &mut AgentLoop, recorder: &Arc<Recorder>) {
    agent.snapshots = TaskSnapshots::new(Some(recorder.clone() as Arc<dyn TaskSnapshotSink>));
}

fn running(id: Uuid, call_id: Option<&str>, name: &str, what: &str) -> RunningToolCall {
    RunningToolCall {
        tool_call_id: id,
        call_id: call_id.map(str::to_string),
        tool_name: name.into(),
        parameters: HashMap::from([("what".to_string(), Value::from(what))]),
        custom_output: None,
        message_to_agent: None,
        built_in: false,
        output: None,
    }
}

/// A task as saved while `calls` of its last native turn ran.
fn saved_native_task(agent: &mut AgentLoop, calls: Vec<ai::ToolCall>) {
    let mut assistant = ai::Message::new_assistant("");
    assistant.tool_calls = calls;
    let task = &mut agent.task;
    task.conversation.system_message = Some("system".into());
    task.conversation.add_user_message("Do the task");
    task.user_message_count = 1;
    task.conversation.messages.push(assistant);
}

/// The task as a host would load it: through its saved form.
fn reloaded(task: &Task) -> Task {
    serde_json::from_str(&serde_json::to_string(task).unwrap()).unwrap()
}

#[tokio::test]
async fn the_turn_is_reported_and_its_calls_saved_before_they_run() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let probe = Arc::new(Probe {
        recorder: Some(recorder.clone()),
        ..Probe::new("probe", ToolRecovery::InDoubt)
    });
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["probe"]),
        registry(vec![probe.clone()]),
        &workspace,
    );
    with_sink(&mut agent, &recorder);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "a",
            "probe",
            serde_json::json!({"what": "one"}),
        )]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    // The call saw its id, and the sink had saved it as running, durably.
    let seen = probe.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let id = seen[0].tool_call_id.expect("the call's id");
    assert!(!seen[0].recovering);
    let (saved, durable) = seen[0].snapshot.clone().unwrap();
    assert!(durable, "the write-ahead snapshot is durable");
    assert_eq!(saved.running_tool_calls.len(), 1);
    assert_eq!(saved.running_tool_calls[0].tool_call_id, id);
    assert_eq!(saved.running_tool_calls[0].call_id.as_deref(), Some("a"));
    assert_eq!(saved.running_tool_calls[0].tool_name, "probe");

    let snapshots = recorder.snapshots();
    let write_ahead = snapshots
        .iter()
        .position(|(task, durable)| *durable && !task.running_tool_calls.is_empty())
        .unwrap();
    // Before it: the turn, reported with no call running and no result.
    let (turn_saved, _) = snapshots[..write_ahead]
        .iter()
        .rev()
        .find(|(task, _)| {
            task.conversation.messages.last().is_some_and(|message| {
                message
                    .tool_calls
                    .iter()
                    .any(|call| call.id.as_deref() == Some("a"))
            })
        })
        .expect("the turn is reported before its calls run");
    assert!(turn_saved.running_tool_calls.is_empty());
    assert!(turn_saved.pending_native_tool_results.is_empty());
    // After it: the result, with nothing running.
    let (after, durable) = &snapshots[write_ahead + 1];
    assert!(!durable);
    assert!(after.running_tool_calls.is_empty());
    assert_eq!(after.pending_native_tool_results.len(), 1);
    assert_eq!(after.pending_native_tool_results[0].output, "probe ran one");
    // The end, and never the same snapshot twice in a row.
    let (last, _) = snapshots.last().unwrap();
    assert!(matches!(last.current_state, TaskState::Completed { .. }));
    assert!(last.running_tool_calls.is_empty());
    for pair in snapshots.windows(2) {
        let (first, second) = (&pair[0], &pair[1]);
        if !second.1 {
            assert_ne!(
                serde_json::to_string(&first.0).unwrap(),
                serde_json::to_string(&second.0).unwrap(),
                "an unchanged task is reported again"
            );
        }
    }
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn each_result_is_in_the_task_before_the_next_call_runs() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let probe = Arc::new(Probe {
        recorder: Some(recorder.clone()),
        ..Probe::new("probe", ToolRecovery::InDoubt)
    });
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["probe"]),
        registry(vec![probe.clone()]),
        &workspace,
    );
    with_sink(&mut agent, &recorder);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![
            native_call("a", "probe", serde_json::json!({"what": "one"})),
            native_call("b", "probe", serde_json::json!({"what": "two"})),
        ]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_calls_paired(&task.conversation);
    let seen = probe.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let (saved, durable) = seen[1].snapshot.clone().unwrap();
    assert!(durable);
    // The first call's result is saved; only the second runs.
    assert_eq!(saved.pending_native_tool_results.len(), 1);
    assert_eq!(
        saved.pending_native_tool_results[0].call_id.as_deref(),
        Some("a")
    );
    assert_eq!(saved.running_tool_calls.len(), 1);
    assert_eq!(saved.running_tool_calls[0].call_id.as_deref(), Some("b"));
    // The results reach the conversation in call order, as before.
    let results: Vec<_> = task
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_tool_result)
        .filter_map(|result| result.call_id.clone())
        .collect();
    assert_eq!(results, ["a", "b", "done"]);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn output_based_results_are_in_the_task_before_the_next_call_runs() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let probe = Arc::new(Probe {
        recorder: Some(recorder.clone()),
        parallel_safe: true,
        ..Probe::new("probe", ToolRecovery::InDoubt)
    });
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::OutputBased, &["probe"]),
        registry(vec![probe.clone()]),
        &workspace,
    );
    with_sink(&mut agent, &recorder);
    agent.scripted_responses.get_mut().unwrap().extend([
        text_turn(
            "<batched_tool_call>\n<probe>\n<what>one</what>\n</probe>\n<probe>\n<what>two</what>\n</probe>\n</batched_tool_call>",
        ),
        text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let seen = probe.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let (saved, durable) = seen[1].snapshot.clone().unwrap();
    assert!(durable);
    assert_eq!(saved.pending_text_tool_results.len(), 1);
    assert!(saved.pending_text_tool_results[0].contains("probe ran one"));
    assert_eq!(saved.running_tool_calls.len(), 1);
    assert_eq!(
        saved.running_tool_calls[0].tool_call_id,
        seen[1].tool_call_id.unwrap()
    );
    // Both results in one message, in order, and nothing left waiting.
    assert!(task.pending_text_tool_results.is_empty());
    let message = task
        .conversation
        .messages
        .iter()
        .map(message_text)
        .find(|text| text.contains("probe ran one"))
        .unwrap();
    let (one, two) = (
        message.find("probe ran one").unwrap(),
        message.find("probe ran two").unwrap(),
    );
    assert!(one < two, "{message}");
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn open_native_calls_are_settled_by_their_tools_recovery_modes() {
    let workspace = workspace();
    let writer = Arc::new(Probe::new("writer", ToolRecovery::InDoubt));
    let reader = Arc::new(Probe::new("reader", ToolRecovery::Rerun));
    let resumer = Arc::new(Probe::new("resumer", ToolRecovery::Resume));
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::Native, &["writer", "reader", "resumer"]),
        registry(vec![writer.clone(), reader.clone(), resumer.clone()]),
        &workspace,
    );
    let answered = native_call("answered", "reader", serde_json::json!({"what": "zero"}));
    let write = native_call("write", "writer", serde_json::json!({"what": "w"}));
    let read = native_call("read", "reader", serde_json::json!({"what": "r"}));
    let resume = native_call("resume", "resumer", serde_json::json!({"what": "child"}));
    let unreached = native_call("unreached", "writer", serde_json::json!({"what": "u"}));
    saved_native_task(
        &mut agent,
        vec![answered.clone(), write, read, resume, unreached],
    );
    let (write_id, read_id, resume_id) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let task = &mut agent.task;
    task.pending_native_tool_results = vec![ai::ToolResult::from_tool_call(&answered, "kept")];
    task.running_tool_calls = vec![
        running(write_id, Some("write"), "writer", "w"),
        running(read_id, Some("read"), "reader", "r"),
        running(resume_id, Some("resume"), "resumer", "child"),
    ];
    agent.task = reloaded(&agent.task);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.running_tool_calls.is_empty());
    assert_calls_paired(&task.conversation);
    let output = |call: &str| result_for(&task.conversation, call)[0].clone();
    assert_eq!(output("answered").output, "kept");
    assert_eq!(output("write").output, RESTARTED_CALL);
    assert!(output("write").is_error);
    assert_eq!(output("read").output, "reader ran r");
    assert_eq!(output("resume").output, "resumer resumed child");
    assert_eq!(output("unreached").output, UNREACHED_CALL);
    // Only the results the turn had, in one message, before the next turn.
    assert_eq!(result_for(&task.conversation, "read").len(), 1);

    assert!(
        writer.seen.lock().unwrap().is_empty(),
        "an in-doubt call doesn't run"
    );
    let read_seen = reader.seen.lock().unwrap().clone();
    assert_eq!(read_seen.len(), 1);
    assert_eq!(read_seen[0].tool_call_id, Some(read_id));
    assert!(read_seen[0].recovering);
    let resume_seen = resumer.seen.lock().unwrap().clone();
    assert_eq!(resume_seen.len(), 1);
    assert_eq!(resume_seen[0].tool_call_id, Some(resume_id));
    assert!(resume_seen[0].recovering);

    let events = drain(&mut events);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolRejected { tool_call_id, reason: Some(reason), .. }
            if *tool_call_id == write_id && reason == RESTARTED_CALL
    )));
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn a_recovered_call_is_saved_as_running_until_its_result_is_in() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let resumer = Arc::new(Probe {
        recorder: Some(recorder.clone()),
        ..Probe::new("resumer", ToolRecovery::Resume)
    });
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["resumer"]),
        registry(vec![resumer.clone()]),
        &workspace,
    );
    with_sink(&mut agent, &recorder);
    saved_native_task(
        &mut agent,
        vec![native_call(
            "child",
            "resumer",
            serde_json::json!({"what": "c"}),
        )],
    );
    let id = Uuid::new_v4();
    agent.task.running_tool_calls = vec![running(id, Some("child"), "resumer", "c")];
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let seen = resumer.seen.lock().unwrap().clone();
    let (saved, durable) = seen[0].snapshot.clone().unwrap();
    assert!(durable);
    assert_eq!(
        saved.running_tool_calls.len(),
        1,
        "recorded once, as before"
    );
    assert_eq!(saved.running_tool_calls[0].tool_call_id, id);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn open_output_based_calls_are_settled_with_the_turns_results() {
    let workspace = workspace();
    let writer = Arc::new(Probe::new("writer", ToolRecovery::InDoubt));
    let reader = Arc::new(Probe::new("reader", ToolRecovery::Rerun));
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::OutputBased, &["writer", "reader"]),
        registry(vec![writer.clone(), reader.clone()]),
        &workspace,
    );
    let task = &mut agent.task;
    task.conversation.system_message = Some("system".into());
    task.conversation.add_user_message("Do the task");
    task.user_message_count = 1;
    task.conversation.messages.push(ai::Message::new_assistant(
        "<batched_tool_call>\n<reader>\n<what>first</what>\n</reader>\n<reader>\n<what>r</what>\n</reader>\n<writer>\n<what>w</what>\n</writer>\n</batched_tool_call>",
    ));
    task.pending_text_tool_results = vec!["Result of the first call".into()];
    task.running_tool_calls = vec![
        running(Uuid::new_v4(), None, "reader", "r"),
        running(Uuid::new_v4(), None, "writer", "w"),
    ];
    agent.task = reloaded(&agent.task);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(text_turn(
            "<complete_task>\n<summary>done</summary>\n</complete_task>",
        ));
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.running_tool_calls.is_empty());
    assert!(task.pending_text_tool_results.is_empty());
    let texts: Vec<_> = task
        .conversation
        .messages
        .iter()
        .map(message_text)
        .collect();
    // The message after the turn holds its results: the one saved, the
    // rerun, the in-doubt call, and the note, in that order.
    let answer = &texts[2];
    let positions: Vec<usize> = [
        "Result of the first call",
        "reader ran r",
        RESTARTED_CALL,
        RESTARTED_TURN,
    ]
    .iter()
    .map(|part| {
        answer
            .find(part)
            .unwrap_or_else(|| panic!("{part}: {answer}"))
    })
    .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "{answer}"
    );
    assert!(
        !texts
            .iter()
            .any(|text| text.contains("Continue the task from where your previous response ended")),
        "{texts:#?}"
    );
    assert!(writer.seen.lock().unwrap().is_empty());
    assert!(reader.seen.lock().unwrap()[0].recovering);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn an_approved_call_running_at_the_stop_is_not_asked_about_again() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let writer = Arc::new(Probe {
        recorder: Some(recorder.clone()),
        ..Probe::new("writer", ToolRecovery::InDoubt)
    });
    let mut config = config(ToolCallingMode::Native, &["writer"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) =
        agent(config.clone(), registry(vec![writer.clone()]), &workspace);
    with_sink(&mut agent, &recorder);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "write",
            "writer",
            serde_json::json!({"what": "w"}),
        )]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let mut seen_events = Vec::new();
    let id = wait_for(&mut events, &mut seen_events, |event| match event {
        AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => Some(*tool_call_id),
        _ => None,
    })
    .await;
    commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    // While the approved call ran, the saved task no longer waited.
    let (saved, durable) = writer.seen.lock().unwrap()[0].snapshot.clone().unwrap();
    assert!(durable);
    assert_eq!(saved.current_state, TaskState::InProgress);
    assert_eq!(saved.running_tool_calls[0].tool_call_id, id);

    // That snapshot runs again (its host crashed): the call is in doubt,
    // and nobody is asked about it.
    let (mut agent, mut events, _commands) = agent_for(
        reloaded(&saved),
        registry(vec![writer.clone()]),
        Arc::new(ai::LLMClient::new()),
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "write")[0].output,
        RESTARTED_CALL
    );
    assert_eq!(writer.seen.lock().unwrap().len(), 1, "it ran once");
    assert!(
        !drain(&mut events)
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
    );

    // A snapshot of an older kind, still waiting while the call ran, is
    // treated the same.
    let mut waiting = reloaded(&saved);
    waiting.set_state(TaskState::AwaitingToolApproval {
        tool_call_id: id,
        tool_name: "writer".into(),
        parameters: HashMap::new(),
        tool_description: String::new(),
        approval_reason: Default::default(),
    });
    let (mut agent, _events, _commands) = agent_for(
        waiting,
        registry(vec![writer.clone()]),
        Arc::new(ai::LLMClient::new()),
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(
        result_for(&task.conversation, "write")[0].output,
        RESTARTED_CALL
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn an_approval_wait_saved_at_a_stop_waits_again() {
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let workspace = workspace();
        let recorder = Arc::new(Recorder::default());
        let writer = Arc::new(Probe::new("writer", ToolRecovery::InDoubt));
        let mut config = config(mode.clone(), &["writer"]);
        config.tool_config.default_auto_approve_tool_use = false;
        let (mut agent, mut events, _commands) =
            agent(config, registry(vec![writer.clone()]), &workspace);
        with_sink(&mut agent, &recorder);
        let (call, done) = if mode == ToolCallingMode::Native {
            (
                turn(vec![native_call(
                    "write",
                    "writer",
                    serde_json::json!({"what": "w"}),
                )]),
                complete(),
            )
        } else {
            (
                text_turn("<writer>\n<what>w</what>\n</writer>"),
                text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
            )
        };
        agent.scripted_responses.get_mut().unwrap().push_back(call);
        // The host crashes during the wait: the run is dropped.
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        wait_for(&mut events, &mut seen, |event| match event {
            AgentEvent::ToolCallAwaitingApproval { .. } => Some(()),
            _ => None,
        })
        .await;
        // The wait is reported at the loop's next boundary.
        let saved = loop {
            if let Some((task, _)) = recorder.last()
                && task.current_state.is_awaiting_approval()
            {
                break task;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        run.abort();

        let id = match &saved.current_state {
            TaskState::AwaitingToolApproval { tool_call_id, .. } => *tool_call_id,
            other => panic!("{other:?}"),
        };
        let (mut agent, mut events, commands) = agent_for(
            reloaded(&saved),
            registry(vec![writer.clone()]),
            Arc::new(ai::LLMClient::new()),
        );
        agent.scripted_responses.get_mut().unwrap().push_back(done);
        let run = tokio::spawn(agent.run());
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(writer.seen.lock().unwrap().is_empty(), "{mode:?}: it waits");
        commands
            .send(UserCommand::ApproveToolUse {
                tool_call_id: id,
                custom_tool_output: None,
                message_to_agent: None,
            })
            .await
            .unwrap();
        let task = timeout(Duration::from_secs(5), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{mode:?}: {:?}",
            task.current_state
        );
        assert_eq!(writer.seen.lock().unwrap().len(), 1, "{mode:?}");
        if mode == ToolCallingMode::Native {
            assert_calls_paired(&task.conversation);
        }
        drain(&mut events);
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn a_turn_saved_before_its_calls_ran_answers_them_before_any_context() {
    let workspace = workspace();
    let writer = Arc::new(Probe::new("writer", ToolRecovery::InDoubt));
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["writer"]),
        registry(vec![writer.clone()]),
        &workspace,
    );
    saved_native_task(
        &mut agent,
        vec![native_call(
            "write",
            "writer",
            serde_json::json!({"what": "w"}),
        )],
    );
    agent
        .task
        .pending_context_injections
        .push("A message arrived.".into());
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // The results directly follow the turn (the context comes after them).
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "write")[0].output,
        UNREACHED_CALL
    );
    assert!(writer.seen.lock().unwrap().is_empty());
    assert!(
        task.conversation
            .messages
            .iter()
            .map(message_text)
            .any(|text| text.contains("A message arrived."))
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn a_pause_on_a_tool_error_left_by_the_stop_still_happens() {
    let workspace = workspace();
    let reader = Arc::new(Probe::new("reader", ToolRecovery::Rerun));
    let mut config = config(ToolCallingMode::Native, &["reader"]);
    config.error_recovery_config.on_tool_execution_error = ToolErrorPolicy::PauseAndAwaitApproval;
    let (mut agent, mut events, commands) =
        agent(config, registry(vec![reader.clone()]), &workspace);
    let failed = native_call("failed", "reader", serde_json::json!({"what": "x"}));
    let read = native_call("read", "reader", serde_json::json!({"what": "r"}));
    saved_native_task(&mut agent, vec![failed.clone(), read]);
    let id = Uuid::new_v4();
    let task = &mut agent.task;
    task.pending_native_tool_results = vec![ai::ToolResult::error_from_tool_call(
        &failed,
        "Tool 'reader' execution failed: broken",
    )];
    task.deferred_tool_errors = vec![DeferredToolError {
        message: "Tool 'reader' execution failed: broken".into(),
        tool_call_id: Some(Uuid::new_v4()),
        middleware_pause: false,
    }];
    task.running_tool_calls = vec![running(id, Some("read"), "reader", "r")];
    agent.task = reloaded(&agent.task);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    let cause = wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
            cause, ..
        }) => Some(cause.clone()),
        _ => None,
    })
    .await;
    assert_eq!(cause, ErrorRecoveryCause::ToolError);
    commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "read")[0].output,
        "reader ran r"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn a_handover_reports_the_fresh_conversation() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let (mut agent, _events, commands) = agent(
        config(ToolCallingMode::Native, &[]),
        ToolRegistry::new(),
        &workspace,
    );
    with_sink(&mut agent, &recorder);
    commands.try_send(UserCommand::RequestHandover).unwrap();
    agent.scripted_responses.get_mut().unwrap().extend([
        response(ai::Message::new_assistant("My handover report.")),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(task.handovers.len(), 1);
    let (handed_over, _) = recorder
        .snapshots()
        .into_iter()
        .find(|(task, _)| !task.handovers.is_empty())
        .unwrap();
    // The first snapshot after the handover holds the fresh conversation
    // only.
    assert_eq!(handed_over.conversation.messages.len(), 1);
    assert!(message_text(&handed_over.conversation.messages[0]).contains("My handover report."));
    assert!(handed_over.handover_requested.is_none());
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn a_sink_that_fails_does_not_stop_the_task() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder {
        failing: true,
        ..Recorder::default()
    });
    let probe = Arc::new(Probe::new("probe", ToolRecovery::InDoubt));
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["probe"]),
        registry(vec![probe.clone()]),
        &workspace,
    );
    with_sink(&mut agent, &recorder);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("a", "probe", serde_json::json!({}))]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(probe.seen.lock().unwrap().len(), 1);
    // A failed report is not taken as saved: the next safe point reports
    // the same task again.
    let snapshots = recorder.snapshots();
    assert!(snapshots.len() >= 2);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn the_runtime_reports_to_the_sink_it_is_given() {
    let workspace = workspace();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(ToolCallingMode::Native, &[]),
        "Do the task".into(),
        workspace.clone(),
        None,
    );
    // No model is reachable: the run pauses on its first request.
    let runtime = super::super::AgentRuntime::new(Arc::new(ai::LLMClient::new()))
        .with_task_snapshots(recorder.clone());
    let (_events, commands, handle) = runtime.run_task(
        task,
        Arc::new(ToolRegistry::new()),
        None,
        false,
        None,
        None,
        None,
    );
    let paused = |snapshots: &[(Task, bool)]| {
        snapshots
            .iter()
            .any(|(task, _)| task.current_state.is_awaiting_approval())
    };
    timeout(Duration::from_secs(10), async {
        while !paused(&recorder.snapshots()) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the pause is reported");
    let _ = commands.send(UserCommand::CancelTask).await;
    let task = timeout(Duration::from_secs(10), handle)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let snapshots = recorder.snapshots();
    let (first, _) = &snapshots[0];
    assert_eq!(first.current_state, TaskState::InProgress);
    assert_eq!(first.conversation.messages.len(), 1, "the task prompt");
    let (last, _) = snapshots.last().unwrap();
    assert_eq!(last.current_state, task.current_state);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[test]
fn a_task_saved_without_running_calls_loads_and_running_calls_round_trip() {
    let task = Task::new(
        config(ToolCallingMode::Native, &[]),
        "Do the task".into(),
        std::env::temp_dir(),
        None,
    );
    let mut saved = serde_json::to_value(&task).unwrap();
    assert!(
        saved.get("running_tool_calls").is_none(),
        "not written when empty"
    );
    saved.as_object_mut().unwrap().remove("running_tool_calls");
    let loaded: Task = serde_json::from_value(saved).unwrap();
    assert!(loaded.running_tool_calls.is_empty());

    let mut task = loaded;
    let mut call = running(Uuid::new_v4(), Some("toolu_1"), "Bash", "ls");
    call.built_in = true;
    call.output = Some("a.txt".into());
    task.running_tool_calls.push(call.clone());
    task.running_tool_calls
        .push(running(Uuid::new_v4(), None, "reader", "r"));
    let loaded = reloaded(&task);
    assert_eq!(loaded.running_tool_calls, task.running_tool_calls);
    // A minimal record (as a later version might trim it) loads too.
    let minimal: RunningToolCall = serde_json::from_value(serde_json::json!({
        "tool_call_id": Uuid::new_v4(),
        "tool_name": "reader",
    }))
    .unwrap();
    assert!(minimal.parameters.is_empty() && !minimal.built_in && minimal.output.is_none());
    assert_eq!(
        serde_json::to_value(ToolRecovery::Resume).unwrap(),
        serde_json::json!("resume")
    );
}
