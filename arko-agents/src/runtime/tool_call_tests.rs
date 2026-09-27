//! Tool calls through the runtime loop.

use super::image_tests::{call, response};
use super::refusal_tests::{call_item, responses_server, stream_of};
use super::*;
use crate::agent_config::PerToolConfigOverride;
use crate::tool::{Tool, ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;
use futures::StreamExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A batch-safe tool with a path parameter that counts its runs.
#[derive(Debug)]
struct Counter(Arc<AtomicUsize>);

#[async_trait]
impl Tool for Counter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "counter".into(),
            description: "Counts its runs".into(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: "path".into(),
                description: "A path".into(),
                param_type: ToolParameterType::String,
                required: false,
                is_path: true,
                flag: None,
                default_value: None,
            }],
            supports_batched_calls: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("counted".into())
    }
}

fn workspace() -> PathBuf {
    let path = std::env::temp_dir().join(format!("arko-tool-calls-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// An auto-approving agent offered `tools` (and `complete_task`, which stays
/// auto-approved when the default changes).
fn config(mode: ToolCallingMode, tools: &[&str]) -> AgentConfig {
    let mut config = super::tests::test_config();
    config.tool_config.tool_calling_mode = mode;
    config.tool_config.default_auto_approve_tool_use = true;
    config.tool_config.available_tool_names = tools.iter().map(|name| name.to_string()).collect();
    config.tool_config.tool_specific_configs.insert(
        COMPLETE_TASK_TOOL_NAME.into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(true),
            ..Default::default()
        },
    );
    config
}

fn agent_for(
    mut task: Task,
    registry: ToolRegistry,
    client: Arc<ai::LLMClient>,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    if task.current_state == TaskState::Pending {
        task.set_state(TaskState::InProgress);
    }
    let (events, receiver) = mpsc::channel(500);
    let (sender, commands) = mpsc::channel(10);
    let agent = AgentLoop::new(
        task,
        client,
        Arc::new(registry),
        events,
        commands,
        None,
        true,
        None,
        None,
        None,
    );
    (agent, receiver, sender)
}

fn agent(
    config: AgentConfig,
    registry: ToolRegistry,
    workspace: &std::path::Path,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let task = Task::new(config, "Do the task".into(), workspace.to_path_buf(), None);
    agent_for(task, registry, Arc::new(ai::LLMClient::new()))
}

fn native_call(id: &str, name: &str, arguments: Value) -> ai::ToolCall {
    ai::ToolCall {
        arguments,
        ..call(id, name)
    }
}

fn turn(calls: Vec<ai::ToolCall>) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls = calls;
    response(message)
}

fn complete() -> ScriptedStreamResult {
    turn(vec![native_call(
        "done",
        "complete_task",
        serde_json::json!({"summary": "done"}),
    )])
}

/// An output-based turn: the response text streamed, then the response.
fn text_turn(text: &str) -> ScriptedStreamResult {
    let Ok(end) = response(ai::Message::new_assistant(text)) else {
        unreachable!()
    };
    Ok(
        futures::stream::iter([Ok(ai::StreamEvent::TextChunk(text.into()))])
            .chain(end)
            .boxed(),
    )
}

fn drain(events: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    received
}

/// Each assistant turn's calls are answered, one result per call, by the
/// message that follows it, as providers require of a request.
fn assert_calls_paired(conversation: &ai::Conversation) {
    for (index, message) in conversation.messages.iter().enumerate() {
        if message.role != ai::Role::Assistant || message.tool_calls.is_empty() {
            continue;
        }
        let mut calls: Vec<_> = message
            .tool_calls
            .iter()
            .map(|call| call.id.clone())
            .collect();
        let mut results: Vec<_> = conversation
            .messages
            .get(index + 1)
            .map(|next| {
                next.content
                    .iter()
                    .filter_map(ai::Content::as_tool_result)
                    .map(|result| result.call_id.clone())
                    .collect()
            })
            .unwrap_or_default();
        calls.sort();
        results.sort();
        assert_eq!(calls, results, "turn {index}: {:#?}", conversation.messages);
    }
}

fn result_for<'a>(conversation: &'a ai::Conversation, call_id: &str) -> Vec<&'a ai::ToolResult> {
    conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_tool_result)
        .filter(|result| result.call_id.as_deref() == Some(call_id))
        .collect()
}

// The workspace path check.

#[tokio::test]
async fn a_native_path_check_error_answers_the_call_once_without_running_it() {
    for auto_approve in [true, false] {
        let workspace = workspace();
        let runs = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Counter(Arc::clone(&runs))));
        let mut config = config(ToolCallingMode::Native, &["counter"]);
        config.tool_config.default_auto_approve_tool_use = auto_approve;
        let (mut agent, mut events, _commands) = agent(config, registry, &workspace);
        agent.scripted_responses.get_mut().unwrap().extend([
            turn(vec![native_call(
                "bad",
                "counter",
                serde_json::json!({"path": "a\u{0}b"}),
            )]),
            complete(),
        ]);
        let task = timeout(Duration::from_secs(5), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        let results = result_for(&task.conversation, "bad");
        assert_eq!(results.len(), 1, "{results:?}");
        assert!(results[0].is_error && results[0].output.contains("Path validation failed"));
        assert_calls_paired(&task.conversation);
        let events = drain(&mut events);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { tool_name, .. } if tool_name == "counter"
        )));
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn native_paths_outside_the_workspace_are_judged_as_the_tools_open_them() {
    // `~` is a directory name, not the home directory: `~/../../x` leaves
    // the workspace, and `~/notes.txt` stays in it.
    let root = workspace();
    let workspace = root.join("ws");
    std::fs::create_dir_all(workspace.join("~")).unwrap();
    std::fs::write(workspace.join("~/notes.txt"), "TILDE-NOTES").unwrap();
    std::fs::write(root.join("outside.txt"), "OUTSIDE-SECRET").unwrap();
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["read_file"]),
        ToolRegistry::new(),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "escape",
            "read_file",
            serde_json::json!({"path": "~/../../outside.txt"}),
        )]),
        turn(vec![native_call(
            "notes",
            "read_file",
            serde_json::json!({"path": " ~/notes.txt "}),
        )]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    let escape = result_for(&task.conversation, "escape");
    assert!(escape[0].is_error && escape[0].output.contains("outside the workspace"));
    assert!(
        !serde_json::to_string(&task.conversation)
            .unwrap()
            .contains("OUTSIDE-SECRET")
    );
    assert!(
        result_for(&task.conversation, "notes")[0]
            .output
            .contains("TILDE-NOTES")
    );
    std::fs::remove_dir_all(root).unwrap();
}

// Only the tools the agent is offered run.

#[tokio::test]
async fn native_calls_to_registered_tools_the_agent_is_not_offered_do_not_run() {
    let workspace = workspace();
    std::fs::write(workspace.join("secret.txt"), "SECRET-CONTENTS").unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Counter(Arc::clone(&runs))));
    // read_file, write_file and counter are registered, not offered.
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::Native, &["list_files"]),
        registry,
        &workspace,
    );
    let offered: Vec<_> = agent
        .build_native_tool_config(&agent.task.conversation.model)
        .tools
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    assert_eq!(offered, ["list_files", "complete_task"]);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "read",
            "read_file",
            serde_json::json!({"path": "secret.txt"}),
        )]),
        turn(vec![native_call(
            "write",
            "write_file",
            serde_json::json!({"path": "new.txt", "content": "x"}),
        )]),
        turn(vec![native_call("count", "counter", serde_json::json!({}))]),
        turn(vec![native_call(
            "list",
            "list_files",
            serde_json::json!({"directory": "."}),
        )]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert!(!workspace.join("new.txt").exists());
    for id in ["read", "write", "count"] {
        let results = result_for(&task.conversation, id);
        assert_eq!(results.len(), 1, "{id}");
        assert!(results[0].is_error && results[0].output.contains("is not available"));
    }
    assert!(!result_for(&task.conversation, "list")[0].is_error);
    assert!(
        !serde_json::to_string(&task.conversation)
            .unwrap()
            .contains("SECRET-CONTENTS")
    );
    assert_calls_paired(&task.conversation);
    let events = drain(&mut events);
    let unavailable: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::UnavailableToolAttempted { tool_name } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(unavailable, ["read_file", "write_file", "counter"]);
    let ran: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolResult { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ran, ["list_files", "complete_task"]);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn an_approved_call_to_a_tool_the_run_does_not_offer_is_answered_as_unavailable() {
    // A call saved awaiting approval (by an older version, or with another
    // configuration) whose tool this run does not offer.
    let workspace = workspace();
    let runs = Arc::new(AtomicUsize::new(0));
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Counter(Arc::clone(&runs))));
        let (mut agent, mut events, _commands) =
            agent(config(mode.clone(), &[]), registry, &workspace);
        let answer = if mode == ToolCallingMode::Native {
            let result = agent
                .execute_native_tool_for_approval(
                    &call("saved", "counter"),
                    Uuid::new_v4(),
                    "counter",
                    HashMap::new(),
                    None,
                    None,
                )
                .await
                .unwrap()
                .unwrap();
            assert!(result.is_error);
            result.output
        } else {
            agent
                .execute_tool_and_handle_result(
                    "counter",
                    HashMap::new(),
                    Uuid::new_v4(),
                    None,
                    None,
                )
                .await
                .unwrap()
        };
        assert!(answer.contains("is not available"), "{mode:?}: {answer}");
        assert!(drain(&mut events).iter().any(|event| matches!(
            event,
            AgentEvent::UnavailableToolAttempted { tool_name } if tool_name == "counter"
        )));
    }
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(workspace).unwrap();
}

// SuspendTask in every awaiting state.

#[tokio::test]
async fn suspend_returns_the_task_from_every_awaiting_state() {
    let workspace = workspace();
    let pending = |name: &str| PendingToolCallInfo {
        tool_call_id: Uuid::new_v4(),
        tool_name: name.into(),
        parameters: HashMap::new(),
        tool_description: String::new(),
    };
    let single = pending("read_file");
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        for state in [
            TaskState::AwaitingToolApproval {
                tool_call_id: single.tool_call_id,
                tool_name: single.tool_name.clone(),
                parameters: HashMap::new(),
                tool_description: String::new(),
            },
            TaskState::AwaitingBatchToolApproval {
                batch_id: Uuid::new_v4(),
                pending_calls: vec![pending("read_file"), pending("list_files")],
            },
            TaskState::AwaitingUserResponse {
                questions: "Which file?".into(),
            },
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "failed".into(),
            },
        ] {
            let mut config = config(mode.clone(), &["read_file", "list_files"]);
            config.tool_config.default_auto_approve_tool_use = false;
            let (mut agent, _events, commands) = agent(config, ToolRegistry::new(), &workspace);
            agent.task.set_state(state.clone());
            commands.send(UserCommand::SuspendTask).await.unwrap();
            let task = timeout(Duration::from_secs(5), agent.run())
                .await
                .unwrap_or_else(|_| panic!("{mode:?} {state:?} ignored SuspendTask"))
                .unwrap();
            assert_eq!(task.current_state, state);
        }
    }
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn a_native_batch_approval_survives_suspension_saving_and_resumption() {
    let workspace = workspace();
    std::fs::write(workspace.join("a.txt"), "CONTENTS-OF-A").unwrap();
    let mut config = config(ToolCallingMode::Native, &["read_file", "write_file"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![
            native_call("read", "read_file", serde_json::json!({"path": "a.txt"})),
            native_call(
                "write",
                "write_file",
                serde_json::json!({"path": "b.txt", "content": "x"}),
            ),
        ]));
    let run = tokio::spawn(agent.run());
    let pending_calls = loop {
        match timeout(Duration::from_secs(5), events.recv())
            .await
            .unwrap()
        {
            Some(AgentEvent::BatchToolCallAwaitingApproval { pending_calls, .. }) => {
                break pending_calls;
            }
            Some(_) => {}
            None => panic!("the run ended without asking for approval"),
        }
    };
    commands.send(UserCommand::SuspendTask).await.unwrap();
    let task = timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let TaskState::AwaitingBatchToolApproval { batch_id, .. } = task.current_state.clone() else {
        panic!("suspended in {:?}", task.current_state);
    };
    assert_eq!(task.pending_native_tool_calls.len(), 2);

    let path = workspace.join("task.json");
    task.save_to_path(&path).unwrap();
    let task = Task::load_from_path(&path).unwrap();
    let (mut agent, _events, commands) =
        agent_for(task, ToolRegistry::new(), Arc::new(ai::LLMClient::new()));
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let decision = |name: &str| match name {
        "read_file" => ToolApprovalDecision::Approve {
            custom_tool_output: None,
            message_to_agent: None,
        },
        _ => ToolApprovalDecision::Reject {
            reason: Some("Not now".into()),
        },
    };
    commands
        .send(UserCommand::SubmitBatchToolApprovals {
            batch_id,
            decisions: pending_calls
                .iter()
                .map(|call| (call.tool_call_id, decision(&call.tool_name)))
                .collect(),
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert!(
        result_for(&task.conversation, "read")[0]
            .output
            .contains("CONTENTS-OF-A")
    );
    assert!(result_for(&task.conversation, "write")[0].is_error);
    assert!(!workspace.join("b.txt").exists());
    std::fs::remove_dir_all(workspace).unwrap();
}

// Approval waits that end without a decision.

/// Every function call of the request's input has its output, as the
/// Responses API requires.
fn assert_request_pairs_calls(request: &Value) {
    let input = request["input"].as_array().unwrap();
    let ids = |kind: &str| {
        let mut ids: Vec<_> = input
            .iter()
            .filter(|item| item["type"] == kind)
            .map(|item| item["call_id"].as_str().unwrap().to_string())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(
        ids("function_call"),
        ids("function_call_output"),
        "{input:#?}"
    );
}

#[tokio::test]
async fn an_approval_wait_that_ends_without_a_decision_leaves_a_resumable_native_turn() {
    for timed_out in [true, false] {
        let workspace = workspace();
        std::fs::write(workspace.join("a.txt"), "CONTENTS-OF-A").unwrap();
        let model = ai::Model::openai_gpt_5_6_luna();
        let (url, captured) = responses_server(vec![
            (
                stream_of(
                    &model.id,
                    vec![
                        call_item("read", "read_file", r#"{"path":"a.txt"}"#),
                        call_item("write", "write_file", r#"{"path":"b.txt","content":"x"}"#),
                    ],
                ),
                None,
            ),
            (
                stream_of(
                    &model.id,
                    vec![call_item("done", "complete_task", r#"{"summary":"done"}"#)],
                ),
                None,
            ),
        ])
        .await;
        let client = Arc::new(ai::LLMClient::new().with_openai_client(
            ai::providers::openai::OpenAIClient::new("test-key").with_base_url(url),
        ));
        let mut config = config(ToolCallingMode::Native, &["read_file", "write_file"]);
        config.tool_config.default_auto_approve_tool_use = false;
        config.tool_config.tool_specific_configs.insert(
            "read_file".into(),
            PerToolConfigOverride {
                auto_approve_tool_use: Some(true),
                ..Default::default()
            },
        );
        config.user_command_timeout_seconds = timed_out.then_some(1);
        let task = Task::new(config, "Do the task".into(), workspace.clone(), None);
        let (agent, mut events, commands) =
            agent_for(task, ToolRegistry::new(), Arc::clone(&client));
        let run = tokio::spawn(agent.run());
        // read_file ran; write_file, in the same turn, waits.
        loop {
            match timeout(Duration::from_secs(30), events.recv())
                .await
                .unwrap()
            {
                Some(AgentEvent::BatchToolCallAwaitingApproval { pending_calls, .. }) => {
                    let names: Vec<_> = pending_calls
                        .iter()
                        .map(|call| call.tool_name.as_str())
                        .collect();
                    assert_eq!(names, ["write_file"]);
                    break;
                }
                Some(_) => {}
                None => panic!("the run ended without asking for approval"),
            }
        }
        let _kept_open = if timed_out {
            Some(commands)
        } else {
            drop(commands);
            None
        };
        let task = timeout(Duration::from_secs(30), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
        assert!(task.pending_native_tool_calls.is_empty());
        assert!(task.pending_native_tool_results.is_empty());
        assert_calls_paired(&task.conversation);
        let undecided = result_for(&task.conversation, "call_write");
        assert!(undecided[0].is_error && undecided[0].output == UNDECIDED_CALL);
        assert!(
            result_for(&task.conversation, "call_read")[0]
                .output
                .contains("CONTENTS-OF-A")
        );
        assert!(drain(&mut events).iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { tool_name, reason: Some(reason), .. }
                if tool_name == "write_file" && reason == UNDECIDED_CALL
        )));
        assert!(!workspace.join("b.txt").exists());

        // Run again, the request pairs every call with its output.
        let (agent, _events, _commands) = agent_for(task, ToolRegistry::new(), client);
        let task = timeout(Duration::from_secs(30), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{:?}",
            task.current_state
        );
        let requests = captured.lock().unwrap().clone();
        assert_eq!(requests.len(), 2);
        assert_request_pairs_calls(&requests[1].1);
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn a_native_turn_an_older_version_left_without_results_is_settled_when_run() {
    // Saved by a version that dropped the approval wait on a timeout: the
    // turn's auto-approved result and the undecided call are still pending,
    // and a call the turn never reached has nothing at all.
    let workspace = workspace();
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["read_file", "write_file"]),
        ToolRegistry::new(),
        &workspace,
    );
    let read = native_call("read", "read_file", serde_json::json!({"path": "a.txt"}));
    let write = native_call("write", "write_file", serde_json::json!({"path": "b.txt"}));
    let skipped = native_call("skipped", "read_file", serde_json::json!({"path": "c.txt"}));
    let mut assistant = ai::Message::new_assistant("");
    assistant.tool_calls = vec![read.clone(), write.clone(), skipped];
    let task = &mut agent.task;
    task.conversation.system_message = Some("system".into());
    task.conversation.add_user_message("Do the task");
    task.conversation.messages.push(assistant);
    task.pending_native_tool_results = vec![ai::ToolResult::from_tool_call(&read, "1 | a")];
    task.pending_native_tool_calls.insert(Uuid::new_v4(), write);
    task.set_state(TaskState::Interrupted {
        reason: "Timeout (60s) waiting for user command".into(),
    });
    let saved = serde_json::to_value(&agent.task).unwrap();
    agent.task = serde_json::from_value(saved).unwrap();
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
        UNDECIDED_CALL
    );
    assert_eq!(result_for(&task.conversation, "read")[0].output, "1 | a");
    assert_eq!(
        result_for(&task.conversation, "skipped")[0].output,
        UNREACHED_CALL
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Waits for the first event `pick` accepts.
async fn next_event<T>(
    events: &mut mpsc::Receiver<AgentEvent>,
    mut pick: impl FnMut(AgentEvent) -> Option<T>,
) -> T {
    loop {
        match timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
        {
            Some(event) => {
                if let Some(picked) = pick(event) {
                    return picked;
                }
            }
            None => panic!("the run ended before the expected event"),
        }
    }
}

#[tokio::test]
async fn a_native_turn_a_paused_tool_error_stops_still_answers_every_call() {
    // write_file waits for approval, the missing file's read_file pauses
    // the task, and the turn never reaches the last read_file.
    for timed_out in [false, true] {
        let workspace = workspace();
        std::fs::write(workspace.join("c.txt"), "C-CONTENTS").unwrap();
        let mut config = config(ToolCallingMode::Native, &["read_file", "write_file"]);
        config.tool_config.default_auto_approve_tool_use = false;
        config.tool_config.tool_specific_configs.insert(
            "read_file".into(),
            PerToolConfigOverride {
                auto_approve_tool_use: Some(true),
                ..Default::default()
            },
        );
        config.error_recovery_config.on_tool_execution_error =
            ToolErrorPolicy::PauseAndAwaitApproval;
        config.user_command_timeout_seconds = timed_out.then_some(1);
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        agent.scripted_responses.get_mut().unwrap().extend([
            turn(vec![
                native_call(
                    "write",
                    "write_file",
                    serde_json::json!({"path": "b.txt", "content": "x"}),
                ),
                native_call(
                    "missing",
                    "read_file",
                    serde_json::json!({"path": "missing.txt"}),
                ),
                native_call("c", "read_file", serde_json::json!({"path": "c.txt"})),
            ]),
            complete(),
        ]);
        let run = tokio::spawn(agent.run());
        next_event(&mut events, |event| match event {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. }) => {
                Some(())
            }
            _ => None,
        })
        .await;
        // The call left awaiting an approval that won't be asked is rejected.
        let rejected = next_event(&mut events, |event| match event {
            AgentEvent::ToolRejected {
                tool_name,
                reason: Some(reason),
                ..
            } if reason == UNREACHED_CALL => Some(tool_name),
            _ => None,
        })
        .await;
        assert_eq!(rejected, "write_file");
        let task = if timed_out {
            let task = timeout(Duration::from_secs(10), run)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
            assert_calls_paired(&task.conversation);
            let (mut agent, _events, _commands) =
                agent_for(task, ToolRegistry::new(), Arc::new(ai::LLMClient::new()));
            agent
                .scripted_responses
                .get_mut()
                .unwrap()
                .push_back(complete());
            timeout(Duration::from_secs(10), agent.run())
                .await
                .unwrap()
                .unwrap()
        } else {
            commands
                .send(UserCommand::ApproveErrorRecovery { message: None })
                .await
                .unwrap();
            timeout(Duration::from_secs(10), run)
                .await
                .unwrap()
                .unwrap()
                .unwrap()
        };
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert_calls_paired(&task.conversation);
        assert!(result_for(&task.conversation, "missing")[0].is_error);
        for id in ["write", "c"] {
            let results = result_for(&task.conversation, id);
            assert_eq!(results.len(), 1, "{id}");
            assert_eq!(results[0].output, UNREACHED_CALL, "{id}");
        }
        assert!(
            !serde_json::to_string(&task.conversation)
                .unwrap()
                .contains("C-CONTENTS")
        );
        assert!(!workspace.join("b.txt").exists());
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn an_approved_ask_user_answered_through_the_task_keeps_its_result() {
    // A host that suspends while the agent waits for an answer and gives
    // it with `prepare_for_resumption_with_user_response` (arko-workflows'
    // `continue_agent`).
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![native_call(
            "ask",
            "ask_user",
            serde_json::json!({"questions": "Which file?"}),
        )]));
    let run = tokio::spawn(agent.run());
    let tool_call_id = next_event(&mut events, |event| match event {
        AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => Some(tool_call_id),
        _ => None,
    })
    .await;
    commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await
        .unwrap();
    next_event(&mut events, |event| match event {
        AgentEvent::UserQuestionsAsked { .. } => Some(()),
        _ => None,
    })
    .await;
    commands.send(UserCommand::SuspendTask).await.unwrap();
    let mut task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingUserResponse { .. }
    ));
    task.prepare_for_resumption_with_user_response("README.md".into())
        .unwrap();
    let (mut agent, _events, _commands) =
        agent_for(task, ToolRegistry::new(), Arc::new(ai::LLMClient::new()));
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "ask")[0].output,
        "Which file?"
    );
    assert!(
        serde_json::to_string(&task.conversation)
            .unwrap()
            .contains("README.md")
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

// Boolean parameters of output-based calls.

#[cfg(unix)]
#[tokio::test]
async fn output_based_end_user_tools_receive_boolean_parameters() {
    let workspace = workspace();
    let parameter = |name: &str, param_type, flag: Option<&str>| ToolParameterDefinition {
        name: name.into(),
        description: name.into(),
        param_type,
        required: false,
        is_path: false,
        flag: flag.map(Into::into),
        default_value: None,
    };
    let mut registry = ToolRegistry::new();
    registry
        .load_end_user_tool(ToolDefinition {
            name: "say".into(),
            description: "Echoes its text".into(),
            output_based_example: None,
            parameters: vec![
                parameter("text", ToolParameterType::String, None),
                parameter("shout", ToolParameterType::Bool, Some("--shout")),
            ],
            supports_batched_calls: false,
            tool_type_details: ToolTypeDetails::EndUserDefined {
                command_template: "echo".into(),
            },
            raw_json_schema_override: None,
        })
        .unwrap();
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::OutputBased, &["say"]),
        registry,
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        text_turn("<say>\n<text>hi</text>\n<shout> TRUE </shout>\n</say>"),
        text_turn("<say>\n<text>hi</text>\n<shout>false</shout>\n</say>"),
        text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let events = drain(&mut events);
    let outputs: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolResult {
                tool_name, output, ..
            } if tool_name == "say" => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs, ["hi --shout\n", "hi\n"]);
    let shout: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallFormatted { parameters, .. } => parameters.get("shout").cloned(),
            _ => None,
        })
        .collect();
    assert_eq!(shout, [Value::Bool(true), Value::Bool(false)]);
    std::fs::remove_dir_all(workspace).unwrap();
}

// complete_task's summary description in both tool calling modes.

#[test]
fn the_summary_override_applies_in_both_tool_calling_modes() {
    let workspace = workspace();
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let mut config = config(mode.clone(), &[]);
        config.complete_task_summary_prompt_override = Some("List every file changed.".into());
        let (agent, _events, _commands) = agent(config, ToolRegistry::new(), &workspace);
        let described = if mode == ToolCallingMode::Native {
            let tools = agent
                .build_native_tool_config(&agent.task.conversation.model)
                .tools;
            tools[0].parameters["properties"]["summary"]["description"]
                .as_str()
                .unwrap()
                .to_string()
        } else {
            let complete_task = agent
                .tool_registry
                .get("complete_task")
                .unwrap()
                .definition();
            PromptBuilder::new(
                &agent.task.agent_config,
                vec![&complete_task],
                &agent.parsing_config,
                None,
            )
            .build_system_prompt(&agent.task)
            .unwrap()
        };
        assert!(
            described.contains("List every file changed."),
            "{mode:?}: {described}"
        );
    }
    std::fs::remove_dir_all(workspace).unwrap();
}
