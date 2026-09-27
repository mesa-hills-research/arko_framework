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

/// A new directory under the canonical temporary directory (on macOS,
/// `/private/var/...` rather than the `/var/...` link).
fn workspace() -> PathBuf {
    let path = std::fs::canonicalize(std::env::temp_dir())
        .unwrap()
        .join(format!("arko-tool-calls-{}", Uuid::new_v4()));
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
async fn a_non_string_value_for_a_path_parameter_is_rejected_not_run() {
    let workspace = workspace();
    let runs = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Counter(Arc::clone(&runs))));
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["counter"]),
        registry,
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "list",
            "counter",
            serde_json::json!({"path": ["../../etc/passwd"]}),
        )]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let result = &result_for(&task.conversation, "list")[0];
    assert!(result.is_error && result.output.contains("must be a string"));
    std::fs::remove_dir_all(workspace).unwrap();
}

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
    // configuration) whose tool this run does not offer, approved with or
    // without output the host provides.
    let workspace = workspace();
    let runs = Arc::new(AtomicUsize::new(0));
    let modes = [ToolCallingMode::Native, ToolCallingMode::OutputBased];
    for (mode, custom_output) in modes
        .iter()
        .flat_map(|mode| [(mode.clone(), None), (mode.clone(), Some("from the host"))])
    {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Counter(Arc::clone(&runs))));
        let (mut agent, mut events, _commands) =
            agent(config(mode.clone(), &[]), registry, &workspace);
        let custom_output = custom_output.map(String::from);
        let answer = if mode == ToolCallingMode::Native {
            let result = agent
                .execute_native_tool_for_approval(
                    &call("saved", "counter"),
                    Uuid::new_v4(),
                    "counter",
                    HashMap::new(),
                    custom_output,
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
                    custom_output,
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
        // The result says why the call didn't run.
        let why = if timed_out {
            "Human approval timed out after 1 seconds, so this call was not run. You can call it again if it is still needed."
        } else {
            "The approval wait ended because the host stopped responding, so this call was not run."
        };
        let undecided = result_for(&task.conversation, "call_write");
        assert!(undecided[0].is_error && undecided[0].output == why);
        assert!(
            result_for(&task.conversation, "call_read")[0]
                .output
                .contains("CONTENTS-OF-A")
        );
        assert!(drain(&mut events).iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { tool_name, reason: Some(reason), .. }
                if tool_name == "write_file" && reason == why
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

/// Waits for the first event `pick` accepts, keeping every event seen in
/// `seen`.
async fn wait_for<T>(
    events: &mut mpsc::Receiver<AgentEvent>,
    seen: &mut Vec<AgentEvent>,
    mut pick: impl FnMut(&AgentEvent) -> Option<T>,
) -> T {
    loop {
        match timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
        {
            Some(event) => {
                let picked = pick(&event);
                seen.push(event);
                if let Some(picked) = picked {
                    return picked;
                }
            }
            None => panic!("the run ended before the expected event: {seen:#?}"),
        }
    }
}

fn paused(events: &[AgentEvent]) -> usize {
    events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ErrorEncountered {
                    requires_approval_to_retry: true,
                    ..
                }
            )
        })
        .count()
}

/// A native batch where the middle `read_file` fails on a missing file.
fn failing_batch(first: ai::ToolCall) -> ScriptedStreamResult {
    turn(vec![
        first,
        native_call(
            "missing",
            "read_file",
            serde_json::json!({"path": "missing.txt"}),
        ),
        native_call("c", "read_file", serde_json::json!({"path": "c.txt"})),
    ])
}

fn read_a() -> ai::ToolCall {
    native_call("a", "read_file", serde_json::json!({"path": "a.txt"}))
}

fn batch_workspace() -> PathBuf {
    let workspace = workspace();
    std::fs::write(workspace.join("a.txt"), "A-CONTENTS").unwrap();
    std::fs::write(workspace.join("c.txt"), "C-CONTENTS").unwrap();
    workspace
}

#[tokio::test]
async fn a_failed_call_in_a_native_batch_gets_its_own_error_result_and_the_others_run() {
    // The default tool error policy.
    let workspace = batch_workspace();
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::Native, &["read_file"]),
        ToolRegistry::new(),
        &workspace,
    );
    assert_eq!(
        agent
            .task
            .agent_config
            .error_recovery_config
            .on_tool_execution_error,
        ToolErrorPolicy::InformAgentAndContinue
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([failing_batch(read_a()), complete()]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    // One message answers the batch with three results and nothing else.
    let turn = task
        .conversation
        .messages
        .iter()
        .position(|message| message.tool_calls.len() == 3)
        .unwrap();
    let answer = &task.conversation.messages[turn + 1];
    assert_eq!(answer.content.len(), 3, "{answer:#?}");
    let results: Vec<_> = answer
        .content
        .iter()
        .map(|content| content.as_tool_result().unwrap())
        .collect();
    assert!(!results[0].is_error && results[0].output.contains("A-CONTENTS"));
    assert!(results[1].is_error && results[1].output.contains("execution failed"));
    assert!(!results[2].is_error && results[2].output.contains("C-CONTENTS"));
    assert!(
        !serde_json::to_string(&task.conversation)
            .unwrap()
            .contains("failed during execution")
    );
    let events = drain(&mut events);
    assert_eq!(paused(&events), 0);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { requires_approval_to_retry: false, error_message, .. }
            if error_message.contains("read_file")
    )));
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn under_pause_a_native_batch_finishes_before_pausing_once() {
    // write_file waits for approval, read_file runs without it, and the
    // missing file's read fails.
    for timed_out in [false, true] {
        let workspace = batch_workspace();
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
        let write = native_call(
            "write",
            "write_file",
            serde_json::json!({"path": "b.txt", "content": "x"}),
        );
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .extend([failing_batch(write), complete()]);
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        // The rest of the batch ran; the pause waits for write_file's
        // approval.
        let (batch_id, pending) = wait_for(&mut events, &mut seen, |event| match event {
            AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } => Some((*batch_id, pending_calls.clone())),
            _ => None,
        })
        .await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].tool_name, "write_file");
        assert_eq!(paused(&seen), 0);
        let task = if timed_out {
            let task = timeout(Duration::from_secs(10), run)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            // The wait ended without a decision: no pause follows.
            assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
            assert!(task.deferred_tool_errors.is_empty());
            assert_calls_paired(&task.conversation);
            seen.extend(drain(&mut events));
            let (mut agent, mut events, _commands) =
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
            seen.extend(drain(&mut events));
            task
        } else {
            commands
                .send(UserCommand::SubmitBatchToolApprovals {
                    batch_id,
                    decisions: HashMap::from([(
                        pending[0].tool_call_id,
                        ToolApprovalDecision::Approve {
                            custom_tool_output: None,
                            message_to_agent: None,
                        },
                    )]),
                })
                .await
                .unwrap();
            let error = wait_for(&mut events, &mut seen, |event| match event {
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                    error_message,
                }) => Some(error_message.clone()),
                _ => None,
            })
            .await;
            assert!(error.contains("read_file"), "{error}");
            // write_file ran before the pause.
            assert!(workspace.join("b.txt").exists());
            commands
                .send(UserCommand::ApproveErrorRecovery { message: None })
                .await
                .unwrap();
            let task = timeout(Duration::from_secs(10), run)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            seen.extend(drain(&mut events));
            task
        };
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert_calls_paired(&task.conversation);
        assert_eq!(paused(&seen), usize::from(!timed_out), "{seen:#?}");
        assert!(result_for(&task.conversation, "missing")[0].is_error);
        assert!(
            result_for(&task.conversation, "c")[0]
                .output
                .contains("C-CONTENTS")
        );
        let write = &result_for(&task.conversation, "write")[0];
        assert_eq!(write.is_error, timed_out, "{write:?}");
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn under_fail_task_a_native_batch_stops_and_answers_the_calls_it_did_not_run() {
    for approvals in [false, true] {
        let workspace = batch_workspace();
        let mut config = config(ToolCallingMode::Native, &["read_file"]);
        config.tool_config.default_auto_approve_tool_use = !approvals;
        config.error_recovery_config.on_tool_execution_error = ToolErrorPolicy::FailTask;
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(failing_batch(read_a()));
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        if approvals {
            let (batch_id, pending) = wait_for(&mut events, &mut seen, |event| match event {
                AgentEvent::BatchToolCallAwaitingApproval {
                    batch_id,
                    pending_calls,
                } => Some((*batch_id, pending_calls.clone())),
                _ => None,
            })
            .await;
            let approve = || ToolApprovalDecision::Approve {
                custom_tool_output: None,
                message_to_agent: None,
            };
            commands
                .send(UserCommand::SubmitBatchToolApprovals {
                    batch_id,
                    decisions: pending
                        .iter()
                        .map(|call| (call.tool_call_id, approve()))
                        .collect(),
                })
                .await
                .unwrap();
        }
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        seen.extend(drain(&mut events));
        assert!(matches!(task.current_state, TaskState::Failed { .. }));
        assert_calls_paired(&task.conversation);
        assert!(
            result_for(&task.conversation, "a")[0]
                .output
                .contains("A-CONTENTS")
        );
        assert!(result_for(&task.conversation, "missing")[0].is_error);
        let not_run = &result_for(&task.conversation, "c")[0];
        assert!(not_run.is_error);
        assert_eq!(not_run.output, TASK_FAILED_CALL);
        assert!(
            !serde_json::to_string(&task.conversation)
                .unwrap()
                .contains("C-CONTENTS")
        );
        // A call that waited for approval is reported as not run.
        assert_eq!(
            seen.iter().any(|event| matches!(
                event,
                AgentEvent::ToolRejected { reason: Some(reason), .. } if reason == TASK_FAILED_CALL
            )),
            approvals
        );
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// Stops the task at a call to `b.txt`, as a workflow's cancellation (or a
/// middleware failing the task) does.
struct StopAtB {
    cancel: bool,
}

#[async_trait]
impl ToolExecutionMiddleware for StopAtB {
    async fn before(
        &self,
        _: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        if call.parameters.get("path").and_then(Value::as_str) != Some("b.txt") {
            return Ok(ToolExecutionDecision::Continue);
        }
        Ok(if self.cancel {
            ToolExecutionDecision::CancelTask {
                reason: "Workflow cancelled.".into(),
            }
        } else {
            ToolExecutionDecision::FailTask {
                reason: "Budget exhausted.".into(),
            }
        })
    }

    async fn after(
        &self,
        _: &ToolExecutionContext<'_>,
        _: &ToolCallContext<'_>,
        _: ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError> {
        Ok(())
    }
}

/// A native turn whose calls the stream announces first, as providers
/// stream them.
fn announced_turn(calls: Vec<ai::ToolCall>) -> ScriptedStreamResult {
    let starts: Vec<_> = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            Ok(ai::StreamEvent::ToolCallStart {
                index,
                name: call.name.clone(),
            })
        })
        .collect();
    let Ok(end) = turn(calls) else { unreachable!() };
    Ok(futures::stream::iter(starts).chain(end).boxed())
}

#[tokio::test]
async fn a_native_batch_a_middleware_stops_answers_the_calls_it_did_not_run() {
    for (cancel, approvals) in [(true, false), (true, true), (false, false), (false, true)] {
        let workspace = batch_workspace();
        let mut config = config(ToolCallingMode::Native, &["read_file"]);
        config.tool_config.default_auto_approve_tool_use = !approvals;
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        agent.execution_middleware = Some(Arc::new(StopAtB { cancel }));
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(announced_turn(vec![
                read_a(),
                native_call("b", "read_file", serde_json::json!({"path": "b.txt"})),
                native_call("c", "read_file", serde_json::json!({"path": "c.txt"})),
            ]));
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        if approvals {
            let (batch_id, pending) = wait_for(&mut events, &mut seen, |event| match event {
                AgentEvent::BatchToolCallAwaitingApproval {
                    batch_id,
                    pending_calls,
                } => Some((*batch_id, pending_calls.clone())),
                _ => None,
            })
            .await;
            let approve = || ToolApprovalDecision::Approve {
                custom_tool_output: None,
                message_to_agent: None,
            };
            commands
                .send(UserCommand::SubmitBatchToolApprovals {
                    batch_id,
                    decisions: pending
                        .iter()
                        .map(|call| (call.tool_call_id, approve()))
                        .collect(),
                })
                .await
                .unwrap();
        }
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        seen.extend(drain(&mut events));
        let reason = if cancel {
            "Workflow cancelled."
        } else {
            "Budget exhausted."
        };
        let expected = if cancel {
            TaskState::Cancelled {
                reason: reason.into(),
            }
        } else {
            TaskState::Failed {
                reason: reason.into(),
            }
        };
        assert_eq!(task.current_state, expected);
        assert_calls_paired(&task.conversation);
        assert!(
            result_for(&task.conversation, "a")[0]
                .output
                .contains("A-CONTENTS")
        );
        assert_eq!(result_for(&task.conversation, "b")[0].output, reason);
        // The call the stop left gets the stop's reason, in its result and
        // its `ToolRejected`.
        let not_run = format!("This call was not run: {reason}");
        assert_eq!(result_for(&task.conversation, "c")[0].output, not_run);
        let rejected: Vec<_> = seen
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolRejected {
                    reason: Some(rejected),
                    ..
                } if rejected.starts_with("This call was not run") => Some(rejected.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(rejected, [not_run.as_str()], "{cancel} {approvals}");
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn cancelling_during_an_approval_wait_answers_the_waiting_calls() {
    let workspace = batch_workspace();
    let mut config = config(ToolCallingMode::Native, &["read_file"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![
            read_a(),
            native_call("c", "read_file", serde_json::json!({"path": "c.txt"})),
        ]));
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::BatchToolCallAwaitingApproval { .. }).then_some(())
    })
    .await;
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    seen.extend(drain(&mut events));
    assert_eq!(
        task.current_state,
        TaskState::Cancelled {
            reason: crate::task::CANCELLED_BY_USER.into()
        }
    );
    assert_calls_paired(&task.conversation);
    // The command comes from the user or from a host such as a workflow.
    for id in ["a", "c"] {
        assert_eq!(
            result_for(&task.conversation, id)[0].output,
            TASK_CANCELLED_CALL,
            "{id}"
        );
    }
    let rejected = seen
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ToolRejected { reason: Some(reason), .. } if reason == TASK_CANCELLED_CALL
            )
        })
        .count();
    assert_eq!(rejected, 2);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn rejecting_recovery_records_the_turn_and_a_pause_names_every_failure() {
    // One failed call, then two.
    for failures in [1, 2] {
        let workspace = batch_workspace();
        let mut config = config(ToolCallingMode::Native, &["read_file"]);
        config.error_recovery_config.on_tool_execution_error =
            ToolErrorPolicy::PauseAndAwaitApproval;
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        let first = if failures == 1 {
            read_a()
        } else {
            native_call("a", "read_file", serde_json::json!({"path": "gone.txt"}))
        };
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(failing_batch(first));
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        let (message, tool_call_id) = wait_for(&mut events, &mut seen, |event| match event {
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                error_message,
                tool_call_id,
            } => Some((error_message.clone(), *tool_call_id)),
            _ => None,
        })
        .await;
        // The pause lists every failure and names a call only when one
        // failed.
        assert_eq!(message.lines().count(), failures);
        assert_eq!(tool_call_id.is_some(), failures == 1);
        commands
            .send(UserCommand::RejectErrorRecovery { message: None })
            .await
            .unwrap();
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Failed { .. }));
        // The failed task still records the turn's results.
        assert!(task.pending_native_tool_results.is_empty());
        assert_calls_paired(&task.conversation);
        assert!(
            result_for(&task.conversation, "c")[0]
                .output
                .contains("C-CONTENTS")
        );
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

#[tokio::test]
async fn an_output_based_error_under_pause_keeps_its_result_when_the_wait_ends() {
    // The missing file's read fails and pauses the task once the read
    // outside the workspace is decided; the approval wait times out first.
    let root = workspace();
    let workspace = root.join("ws");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(root.join("outside.txt"), "OUTSIDE-CONTENTS").unwrap();
    let mut config = config(ToolCallingMode::OutputBased, &["read_file"]);
    config.tool_config.default_out_of_workspace_path_policy = PathSafetyPolicy::RequireApproval;
    config.error_recovery_config.on_tool_execution_error = ToolErrorPolicy::PauseAndAwaitApproval;
    config.user_command_timeout_seconds = Some(1);
    let (mut agent, _events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    agent.scripted_responses.get_mut().unwrap().push_back(text_turn(
        "<batched_tool_call>\n<read_file>\n<path>missing.txt</path>\n</read_file>\n<read_file>\n<path>../outside.txt</path>\n</read_file>\n</batched_tool_call>",
    ));
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert!(task.deferred_tool_errors.is_empty());
    let last = task
        .conversation
        .messages
        .last()
        .and_then(|message| message.content.first()?.as_text())
        .unwrap();
    assert!(last.contains("failed during execution"), "{last}");
    assert!(
        last.contains("Tool read_file: Human approval timed out after 1 seconds"),
        "{last}"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn recovery_approval_answers_calls_an_older_version_stranded() {
    // Saved by a version that paused on a tool error mid-batch and left the
    // rest of the turn without results.
    let workspace = workspace();
    let (mut agent, _events, commands) = agent(
        config(ToolCallingMode::Native, &["read_file"]),
        ToolRegistry::new(),
        &workspace,
    );
    let failed = native_call("failed", "read_file", serde_json::json!({"path": "x"}));
    let stranded = native_call("stranded", "read_file", serde_json::json!({"path": "y"}));
    let mut assistant = ai::Message::new_assistant("");
    assistant.tool_calls = vec![failed.clone(), stranded.clone()];
    let task = &mut agent.task;
    task.conversation.system_message = Some("system".into());
    task.conversation.add_user_message("Do the task");
    task.conversation.messages.push(assistant);
    task.pending_native_tool_results =
        vec![ai::ToolResult::error_from_tool_call(&failed, "not found")];
    task.pending_native_tool_calls
        .insert(Uuid::new_v4(), stranded);
    task.set_state(TaskState::AwaitingErrorRecoveryApproval {
        error_message: "Tool 'read_file' execution failed: not found".into(),
    });
    let saved = serde_json::to_value(&agent.task).unwrap();
    agent.task = serde_json::from_value(saved).unwrap();
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    commands
        .send(UserCommand::ApproveErrorRecovery { message: None })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "stranded")[0].output,
        UNREACHED_CALL
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

#[test]
fn an_answer_given_through_the_task_carries_the_waiting_results_of_the_question() {
    // Saved by a version that kept an approved `ask_user`'s result waiting.
    let mut task = Task::new(
        config(ToolCallingMode::Native, &["ask_user"]),
        "Do the task".into(),
        std::env::temp_dir(),
        None,
    );
    let ask = native_call(
        "ask",
        "ask_user",
        serde_json::json!({"questions": "Which file?"}),
    );
    let mut assistant = ai::Message::new_assistant("");
    assistant.tool_calls = vec![ask.clone()];
    task.conversation.add_user_message("Do the task");
    task.conversation.messages.push(assistant);
    task.pending_native_tool_results = vec![ai::ToolResult::from_tool_call(&ask, "Which file?")];
    task.set_state(TaskState::AwaitingUserResponse {
        questions: "Which file?".into(),
    });
    task.prepare_for_resumption_with_user_response("README.md".into())
        .unwrap();
    assert_calls_paired(&task.conversation);
    assert!(task.pending_native_tool_results.is_empty());
    let answer = task.conversation.messages.last().unwrap();
    assert!(answer.content.iter().any(|content| {
        content
            .as_text()
            .is_some_and(|text| text.contains("README.md"))
    }));
}

// `ask_user` without an approval step.

#[test]
fn ask_user_is_auto_approved_unless_its_override_says_otherwise() {
    let mut config = config(ToolCallingMode::Native, &["ask_user", "read_file"]);
    config.tool_config.default_auto_approve_tool_use = false;
    config.tool_config.tool_specific_configs.clear();
    assert!(config.should_auto_approve_tool("ask_user"));
    // complete_task keeps its approval behaviour.
    assert!(!config.should_auto_approve_tool("complete_task"));
    assert!(!config.should_auto_approve_tool("read_file"));
    config.tool_config.tool_specific_configs.insert(
        "ask_user".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    assert!(!config.should_auto_approve_tool("ask_user"));
}

#[tokio::test]
async fn ask_user_asks_without_an_approval_step_by_default() {
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let workspace = workspace();
        let mut config = config(mode.clone(), &["ask_user"]);
        config.tool_config.default_auto_approve_tool_use = false;
        let native = mode == ToolCallingMode::Native;
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        let (question, done) = if native {
            (
                turn(vec![native_call(
                    "ask",
                    "ask_user",
                    serde_json::json!({"questions": "Which file?"}),
                )]),
                complete(),
            )
        } else {
            (
                text_turn("<ask_user>\n<questions>Which file?</questions>\n</ask_user>"),
                text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
            )
        };
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .extend([question, done]);
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        wait_for(&mut events, &mut seen, |event| {
            matches!(event, AgentEvent::UserQuestionsAsked { .. }).then_some(())
        })
        .await;
        assert!(
            !seen.iter().any(|event| matches!(
                event,
                AgentEvent::ToolCallAwaitingApproval { .. }
                    | AgentEvent::BatchToolCallAwaitingApproval { .. }
            )),
            "{mode:?}"
        );
        commands
            .send(UserCommand::SubmitUserResponse {
                response: "README.md".into(),
            })
            .await
            .unwrap();
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{mode:?}"
        );
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

// Output-based results of a turn that waits.

#[tokio::test]
async fn output_based_batch_results_wait_for_the_calls_awaiting_approval() {
    // The read inside the workspace runs at once; the one outside waits
    // for approval (`RequireApproval`).
    for ending in ["approve", "suspend", "timeout"] {
        let root = workspace();
        let workspace = root.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(workspace.join("inside.txt"), "INSIDE-CONTENTS").unwrap();
        std::fs::write(root.join("outside.txt"), "OUTSIDE-CONTENTS").unwrap();
        let mut config = config(ToolCallingMode::OutputBased, &["read_file"]);
        config.tool_config.default_out_of_workspace_path_policy = PathSafetyPolicy::RequireApproval;
        config.user_command_timeout_seconds = (ending == "timeout").then_some(1);
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        agent.scripted_responses.get_mut().unwrap().extend([
            text_turn(
                "<batched_tool_call>\n<read_file>\n<path>inside.txt</path>\n</read_file>\n<read_file>\n<path>../outside.txt</path>\n</read_file>\n</batched_tool_call>",
            ),
            text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
        ]);
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        let (batch_id, pending) = wait_for(&mut events, &mut seen, |event| match event {
            AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } => Some((*batch_id, pending_calls.clone())),
            _ => None,
        })
        .await;
        assert_eq!(pending.len(), 1);
        let approve = UserCommand::SubmitBatchToolApprovals {
            batch_id,
            decisions: HashMap::from([(
                pending[0].tool_call_id,
                ToolApprovalDecision::Approve {
                    custom_tool_output: None,
                    message_to_agent: None,
                },
            )]),
        };
        let task = match ending {
            "approve" => {
                commands.send(approve).await.unwrap();
                timeout(Duration::from_secs(10), run)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap()
            }
            "suspend" => {
                commands.send(UserCommand::SuspendTask).await.unwrap();
                let task = timeout(Duration::from_secs(10), run)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(task.pending_text_tool_results[0].contains("INSIDE-CONTENTS"));
                let path = root.join("task.json");
                task.save_to_path(&path).unwrap();
                let task = Task::load_from_path(&path).unwrap();
                let (mut agent, _events, commands) =
                    agent_for(task, ToolRegistry::new(), Arc::new(ai::LLMClient::new()));
                agent
                    .scripted_responses
                    .get_mut()
                    .unwrap()
                    .push_back(text_turn(
                        "<complete_task>\n<summary>done</summary>\n</complete_task>",
                    ));
                commands.send(approve).await.unwrap();
                timeout(Duration::from_secs(10), agent.run())
                    .await
                    .unwrap()
                    .unwrap()
            }
            _ => {
                let task = timeout(Duration::from_secs(10), run)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
                seen.extend(drain(&mut events));
                assert!(seen.iter().any(|event| matches!(
                    event,
                    AgentEvent::ToolRejected { reason: Some(reason), .. }
                        if reason.starts_with("Human approval timed out after 1 seconds")
                )));
                task
            }
        };
        assert!(task.pending_text_tool_results.is_empty(), "{ending}");
        // One message holds the result that ran at once and the outcome of
        // the call that waited.
        let answer = task
            .conversation
            .messages
            .iter()
            .filter_map(|message| message.content.first()?.as_text())
            .find(|text| text.contains("INSIDE-CONTENTS"))
            .unwrap_or_else(|| panic!("{ending}: {:#?}", task.conversation.messages));
        let outcome = if ending == "timeout" {
            "Tool read_file: Human approval timed out after 1 seconds"
        } else {
            "OUTSIDE-CONTENTS"
        };
        assert!(answer.contains(outcome), "{ending}: {answer}");
        if ending != "timeout" {
            assert!(
                matches!(task.current_state, TaskState::Completed { .. }),
                "{ending}"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[tokio::test]
async fn an_output_based_batch_stops_when_a_path_rejection_fails_the_task() {
    let workspace = workspace();
    let runs = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Counter(Arc::clone(&runs))));
    let mut config = config(ToolCallingMode::OutputBased, &["counter"]);
    config.error_recovery_config.on_tool_execution_error = ToolErrorPolicy::FailTask;
    let (mut agent, _events, _commands) = agent(config, registry, &workspace);
    agent.scripted_responses.get_mut().unwrap().push_back(text_turn(
        "<batched_tool_call>\n<counter>\n<path>../outside</path>\n</counter>\n<counter>\n<path>inside</path>\n</counter>\n</batched_tool_call>",
    ));
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Failed { .. }));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn an_approved_ask_user_answered_through_the_task_keeps_its_result() {
    // A host that suspends while the agent waits for an answer and gives
    // it with `prepare_for_resumption_with_user_response` (arko_workflows'
    // `continue_agent`). `ask_user`'s own override brings its approval back.
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.tool_specific_configs.insert(
        "ask_user".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
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
