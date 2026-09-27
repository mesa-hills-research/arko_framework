//! Tool calls through the runtime loop.

use super::image_tests::{call, response};
use super::*;
use crate::agent_config::PerToolConfigOverride;
use crate::tool::{Tool, ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;
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
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Counter(Arc::clone(&runs))));
    let (mut agent, _events, _commands) =
        agent(config(ToolCallingMode::Native, &[]), registry, &workspace);
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
    assert!(result.is_error && result.output.contains("is not available"));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(workspace).unwrap();
}
