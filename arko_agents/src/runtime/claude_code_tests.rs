//! The Claude Code runtime against a scripted fake CLI
//! (`tests/fixtures/fake_claude_agent.py`; `ARKO_TEST_PYTHON` selects the
//! interpreter).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::agent_config::{
    AgentConfig, AgentLLMConfig, AgentToolConfig, PathSafetyPolicy, PerToolConfigOverride,
};
use crate::claude_code::ClaudeCodeSettings;
use crate::handover::{HandoverConfig, REPORT_HEADING};
use crate::runtime::AgentRuntime;
use crate::runtime::command::UserCommand;
use crate::runtime::event::AgentEvent;
use crate::task::{Task, TaskState};
use crate::tool::ToolRegistry;

/// A fake CLI with its script, log and turn counter in a fresh directory.
struct Fake {
    dir: PathBuf,
}

impl Fake {
    fn new(script: Value) -> Self {
        let dir = std::env::temp_dir().join(format!("arko-fake-claude-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("workspace")).unwrap();
        std::fs::write(dir.join("script.json"), script.to_string()).unwrap();
        Self { dir }
    }

    fn workspace(&self) -> PathBuf {
        self.dir.join("workspace")
    }

    fn settings(&self) -> ClaudeCodeSettings {
        let python = std::env::var("ARKO_TEST_PYTHON")
            .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_claude_agent.py");
        let path = |name: &str| self.dir.join(name).display().to_string();
        ClaudeCodeSettings {
            cli_path: Some(python.into()),
            cli_args: vec![script.display().to_string()],
            env: BTreeMap::from([
                ("FAKE_CLAUDE_SCRIPT".to_string(), path("script.json")),
                ("FAKE_CLAUDE_LOG".to_string(), path("log.jsonl")),
                ("FAKE_CLAUDE_STATE".to_string(), path("state")),
            ]),
            ..ClaudeCodeSettings::default()
        }
    }

    /// The fake's log: one JSON object per event.
    fn log(&self) -> Vec<Value> {
        std::fs::read_to_string(self.dir.join("log.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn logged(&self, event: &str) -> Vec<Value> {
        self.log()
            .into_iter()
            .filter(|entry| entry["event"] == event)
            .collect()
    }
}

// KEEP_FAKE=1 keeps the directory (script, log) for debugging.
impl Drop for Fake {
    fn drop(&mut self) {
        if std::env::var("KEEP_FAKE").is_err() {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

fn config(fake: &Fake, tools: &[&str]) -> AgentConfig {
    AgentConfig::builder(
        "cc-agent".into(),
        "CC agent".into(),
        AgentLLMConfig {
            model: arko_ai::Model::claude_code(arko_ai::Model::anthropic_claude_opus_5_5()),
            generation_config: Default::default(),
        },
    )
    .role_description("Tests the Claude Code runtime.".into())
    .instructions("Follow the script.".into())
    .tool_config(AgentToolConfig {
        default_auto_approve_tool_use: true,
        available_tool_names: tools.iter().map(|tool| tool.to_string()).collect(),
        ..AgentToolConfig::default()
    })
    .claude_code(fake.settings())
    .build()
    .unwrap()
}

struct Run {
    events: mpsc::Receiver<AgentEvent>,
    commands: mpsc::Sender<UserCommand>,
    handle: tokio::task::JoinHandle<Result<Task, crate::ArkoAgentsError>>,
}

fn start(task: Task) -> Run {
    start_with(task, AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())))
}

fn start_with(task: Task, runtime: AgentRuntime) -> Run {
    let (events, commands, handle) = runtime.run_task(
        task,
        Arc::new(ToolRegistry::new()),
        None,
        false,
        None,
        None,
        None,
    );
    Run {
        events,
        commands,
        handle,
    }
}

impl Run {
    /// Events up to the one `stop` matches (inclusive), or the end.
    async fn events_until(&mut self, stop: impl Fn(&AgentEvent) -> bool) -> Vec<AgentEvent> {
        let mut seen = Vec::new();
        loop {
            let event = tokio::time::timeout(Duration::from_secs(20), self.events.recv())
                .await
                .unwrap_or_else(|_| panic!("no event in time; so far: {seen:#?}"));
            let Some(event) = event else {
                return seen;
            };
            let done = stop(&event);
            seen.push(event);
            if done {
                return seen;
            }
        }
    }

    async fn finish(mut self) -> (Task, Vec<AgentEvent>) {
        let events = self.events_until(|_| false).await;
        let task = tokio::time::timeout(Duration::from_secs(20), self.handle)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        (task, events)
    }
}

fn texts(task: &Task) -> Vec<String> {
    task.conversation
        .messages
        .iter()
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(arko_ai::Content::as_text)
                .collect::<String>()
        })
        .collect()
}

fn complete(summary: &str) -> Value {
    json!({"tool": "mcp__arko__complete_task", "input": {"summary": summary}})
}

#[tokio::test]
async fn an_agent_completes_through_its_arko_tool() {
    let fake = Fake::new(json!([[
        {"text": "Working on it.", "input_tokens": 1200},
        complete("All done"),
    ]]));
    let task = Task::new(
        config(&fake, &[]),
        "Do the thing.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;

    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "All done"),
        "{:?}",
        task.current_state
    );
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    let has = |flag: &str| argv.iter().any(|arg| arg == flag);
    assert!(has("--strict-mcp-config"), "{argv:?}");
    assert!(has("--setting-sources="), "{argv:?}");
    assert!(has("--append-system-prompt"), "{argv:?}");
    let tools = argv.iter().position(|arg| arg == "--tools").unwrap();
    assert_eq!(argv[tools + 1], "", "no built-in tools unless listed");
    let model = argv.iter().position(|arg| arg == "--model").unwrap();
    assert_eq!(argv[model + 1], "claude-opus-5-5");
    let session = task.claude_code_session.clone().unwrap();
    assert!(has(&format!("--session-id={session}")), "{argv:?}");

    // Its arko tools are served with their schemas; only offered ones.
    let served = &fake.logged("tools")[0]["tools"];
    let names: Vec<&str> = served
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["complete_task"]);
    assert_eq!(served[0]["inputSchema"]["required"], json!(["summary"]));
    // The task prompt went first, with the system prompt naming the tools.
    let users = fake.logged("user");
    assert!(
        users[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Do the thing.")
    );
    assert!(
        task.conversation
            .system_message
            .as_deref()
            .unwrap()
            .contains("mcp__arko__complete_task")
    );
    // Usage: the whole input, cached reads included, billed nothing.
    let usage = &task.usage_history[0];
    assert_eq!(usage.input_tokens, 1200);
    assert_eq!(usage.cached_tokens, Some(1198));
    assert_eq!(usage.billing, arko_ai::BillingKind::Subscription);
    assert_eq!(usage.billed(), 0.0);
    assert!(usage.total_cost > 0.0, "the estimate at list prices");
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "complete_task"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantResponseChunk(text) if text == "Working on it."
    )));
    assert!(matches!(events.last(), Some(AgentEvent::TaskStreamEnd)));
    // The conversation mirrors the session's: the call and its result.
    assert!(task.conversation.messages.iter().any(|message| {
        message
            .tool_calls
            .iter()
            .any(|call| call.name == "complete_task")
    }));
}

#[tokio::test]
async fn a_turn_without_complete_task_gets_a_note() {
    let fake = Fake::new(json!([
        [{"text": "I think I'm done."}],
        [complete("Now done")],
    ]));
    let task = Task::new(
        config(&fake, &[]),
        "Do the thing.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let users = fake.logged("user");
    assert_eq!(users.len(), 2);
    assert!(
        users[1]["text"]
            .as_str()
            .unwrap()
            .contains("without calling complete_task")
    );
}

/// A Claude Code agent's `message_user` calls reach the host as
/// `MessageToUser`, and with `text_reply_as_message` so does the text of a
/// turn that ends without `complete_task`; off, that text isn't sent.
#[tokio::test]
async fn messages_reach_the_user_through_the_tool_and_open_turns() {
    for opted_in in [false, true] {
        let fake = Fake::new(json!([
            [{"text": "Halfway there."}],
            [
                {"tool": "mcp__arko__message_user", "input": {"message": "Almost done."}},
                complete("Now done"),
            ],
        ]));
        let mut config = config(&fake, &["message_user"]);
        config.tool_config.text_reply_as_message = opted_in;
        let task = Task::new(config, "Do the thing.".into(), fake.workspace(), None);
        let (task, events) = start(task).finish().await;
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        let served = &fake.logged("tools")[0]["tools"];
        assert!(
            served
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "message_user"),
            "{served}"
        );
        let call = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::ToolCallFormatted {
                    tool_call_id,
                    tool_name,
                    ..
                } if tool_name == "message_user" => Some(*tool_call_id),
                _ => None,
            })
            .unwrap();
        let messages: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MessageToUser {
                    message,
                    tool_call_id,
                } => Some((message.as_str(), *tool_call_id)),
                _ => None,
            })
            .collect();
        let mut expected = vec![("Almost done.", Some(call))];
        if opted_in {
            expected.insert(0, ("Halfway there.", None));
        }
        assert_eq!(messages, expected, "opted in: {opted_in}");
        let users = fake.logged("user");
        let note = users[1]["text"].as_str().unwrap();
        assert!(note.contains("without calling complete_task"), "{note}");
        assert_eq!(
            note.contains("sent to the user as a message"),
            opted_in,
            "{note}"
        );
        assert!(
            fake.logged("post")[0]["output"]
                .as_str()
                .unwrap()
                .contains("message was sent to the user")
        );
    }
}

#[tokio::test]
async fn built_ins_are_offered_only_when_listed_and_pass_the_gate() {
    // An absolute path outside the workspace on every platform.
    let outside = std::env::temp_dir()
        .join(format!("arko-outside-{}.txt", uuid::Uuid::new_v4()))
        .display()
        .to_string();
    let fake = Fake::new(json!([[
        {"tool": "Read", "input": {"file_path": "notes.txt"}},
        {"tool": "Read", "input": {"file_path": outside}},
        {"tool": "Bash", "input": {"command": "ls"}},
        complete("Read it"),
    ]]));
    let mut config = config(&fake, &["Read", "Grep"]);
    config.tool_config.default_out_of_workspace_path_policy = PathSafetyPolicy::AutoReject;
    let task = Task::new(config, "Read the notes.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    let tools = argv.iter().position(|arg| arg == "--tools").unwrap();
    assert_eq!(argv[tools + 1], "Read,Grep");
    let posts = fake.logged("post");
    assert_eq!(posts[0]["output"], "builtin Read ran");
    assert_eq!(posts[1]["is_error"], true);
    assert!(
        posts[1]["output"]
            .as_str()
            .unwrap()
            .contains("outside the workspace")
    );
    // Not listed: refused, though Claude Code has it.
    assert_eq!(posts[2]["is_error"], true);
    assert!(
        posts[2]["output"]
            .as_str()
            .unwrap()
            .contains("not available")
    );
    let rejected = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolRejected { .. }))
        .count();
    assert_eq!(rejected, 2);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolResult { tool_name, output, .. } if tool_name == "Read" && output == "builtin Read ran"
    )));
}

#[tokio::test]
async fn a_call_that_needs_approval_waits_for_the_host() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "cargo test"}},
        {"tool": "Bash", "input": {"command": "rm -rf target"}},
        complete("Tested"),
    ]]));
    let mut config = config(&fake, &["Bash"]);
    config.tool_config.tool_specific_configs.insert(
        "Bash".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let task = Task::new(config, "Run the tests.".into(), fake.workspace(), None);
    let mut run = start(task);
    for approve in [true, false] {
        let events = run
            .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
            .await;
        let Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) = events.last() else {
            panic!("{events:#?}");
        };
        let command = if approve {
            UserCommand::ApproveToolUse {
                tool_call_id: *tool_call_id,
                custom_tool_output: None,
                message_to_agent: Some("Only the unit tests.".into()),
            }
        } else {
            UserCommand::RejectToolUse {
                tool_call_id: *tool_call_id,
                reason: Some("Not that.".into()),
            }
        };
        run.commands.send(command).await.unwrap();
    }
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let permissions = fake.logged("permission");
    assert_eq!(permissions[0]["answer"]["behavior"], "allow");
    assert_eq!(permissions[1]["answer"]["behavior"], "deny");
    assert_eq!(permissions[1]["answer"]["message"], "Not that.");
    // The approval's message comes with the call's result.
    let posts = fake.logged("post");
    assert!(
        posts[0]["answer"]["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap()
            .contains("Only the unit tests.")
    );
}

#[tokio::test]
async fn ask_user_waits_and_the_answer_resumes_the_session() {
    let fake = Fake::new(json!([
        [{"tool": "mcp__arko__ask_user", "input": {"questions": "Which file?"}}],
        [complete("Used the answer")],
    ]));
    let task = Task::new(
        config(&fake, &["ask_user"]),
        "Edit a file.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    run.events_until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "main.rs".into(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let argvs = fake.logged("argv");
    assert_eq!(argvs.len(), 2, "the wait closed the session");
    let session = task.claude_code_session.clone().unwrap();
    let second: Vec<String> = serde_json::from_value(argvs[1]["argv"].clone()).unwrap();
    assert!(
        second.contains(&format!("--resume={session}")),
        "{second:?}"
    );
    let users = fake.logged("user");
    assert!(users[1]["text"].as_str().unwrap().contains("main.rs"));
    assert!(
        fake.logged("post")[0]["output"]
            .as_str()
            .unwrap()
            .contains("questions were sent")
    );
}

#[tokio::test]
async fn context_arrives_with_the_next_result() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "todo"}},
        complete("Found them"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let run = start(task);
    run.commands
        .send(UserCommand::InjectContext {
            content: "Message from agent 'lead': skip vendor/.".into(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    let posts = fake.logged("post");
    let context = posts[0]["answer"]["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(
        context.starts_with("# Workflow context update"),
        "{context}"
    );
    assert!(context.contains("skip vendor/"));
    // Recorded after the result, where a workflow looks for it.
    let texts = texts(&task);
    assert!(
        texts
            .iter()
            .any(|text| text.starts_with("# Workflow context update")
                && text.contains("skip vendor/"))
    );
    assert!(task.pending_context_injections.is_empty());
}

#[tokio::test]
async fn a_handover_continues_in_a_fresh_session() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "REPORT: searched a and b; next, c."},
        ],
        [complete("Searched c")],
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(1000));
    let task = Task::new(config, "Search a, b and c.".into(), fake.workspace(), None);
    let first_session = {
        let (task, events) = start(task).finish().await;
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{:?}",
            task.current_state
        );
        assert_eq!(task.handovers.len(), 1);
        assert_eq!(
            task.handovers[0].report,
            "REPORT: searched a and b; next, c."
        );
        assert_eq!(task.handovers[0].input_tokens, 1500);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::HandoverRequested { input_tokens: 1500 }))
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ConversationHandedOver { handovers: 1, .. }
        )));
        // The fresh conversation starts from the task and the report.
        let first = &texts(&task)[0];
        assert!(first.starts_with("Search a, b and c."), "{first}");
        assert!(first.contains(REPORT_HEADING));
        task.claude_code_session.clone().unwrap()
    };
    // The second Grep call got the report request as its result.
    let posts = fake.logged("post");
    assert_eq!(posts[1]["is_error"], true);
    assert!(
        posts[1]["output"]
            .as_str()
            .unwrap()
            .contains("Reply with that report as plain text")
    );
    // Claude Code's own compaction was off, and the fresh session is new.
    let argvs = fake.logged("argv");
    assert_eq!(argvs.len(), 2);
    let first_argv: Vec<String> = serde_json::from_value(argvs[0]["argv"].clone()).unwrap();
    let settings = first_argv
        .iter()
        .position(|arg| arg == "--settings")
        .unwrap();
    assert!(first_argv[settings + 1].contains("\"autoCompactEnabled\":false"));
    let second_argv: Vec<String> = serde_json::from_value(argvs[1]["argv"].clone()).unwrap();
    assert!(second_argv.contains(&format!("--session-id={first_session}")));
    assert!(!first_argv.contains(&format!("--session-id={first_session}")));
    let users = fake.logged("user");
    assert!(users[1]["text"].as_str().unwrap().contains(REPORT_HEADING));
}

#[tokio::test]
async fn a_crash_pauses_and_continuing_resumes_the_session() {
    let fake = Fake::new(json!([
        [{"text": "Starting."}, {"exit": 3}],
        [complete("Recovered")],
    ]));
    let task = Task::new(
        config(&fake, &[]),
        "Do the thing.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    let seen = run
        .events_until(|event| {
            matches!(
                event,
                AgentEvent::ErrorEncountered {
                    requires_approval_to_retry: true,
                    ..
                }
            )
        })
        .await;
    // As arko's own runtime reports a pause: the state change that gives
    // its cause, then the error a host answers.
    assert!(
        seen.iter().any(|event| matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )),
        "{seen:#?}"
    );
    run.commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let argvs = fake.logged("argv");
    let second: Vec<String> = serde_json::from_value(argvs[1]["argv"].clone()).unwrap();
    assert!(second.iter().any(|arg| arg.starts_with("--resume=")));
    assert!(
        fake.logged("user")[1]["text"]
            .as_str()
            .unwrap()
            .contains("Continue the task")
    );
}

/// Holds every request after the first while its flag is set.
struct HoldAfterFirst {
    asked: std::sync::atomic::AtomicUsize,
    hold: Arc<std::sync::atomic::AtomicBool>,
}

impl crate::runtime::TurnContextSource for HoldAfterFirst {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        None
    }

    fn hold_request(&self, _task_id: uuid::Uuid) -> Option<String> {
        use std::sync::atomic::Ordering;
        let asked = self.asked.fetch_add(1, Ordering::SeqCst);
        (asked > 0 && self.hold.load(Ordering::SeqCst)).then(|| "Budget exceeded.".to_string())
    }
}

/// A turn the host holds is interrupted and the task pauses with the
/// `BudgetExceeded` cause; continuing while it is still held starts no
/// session, and once released the session resumes.
#[tokio::test]
async fn a_held_turn_pauses_until_the_host_releases_it() {
    let fake = Fake::new(json!([
        [{"text": "Working."}, {"tool": "Grep", "input": {"pattern": "a"}}],
        [complete("done")],
    ]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let hold = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(HoldAfterFirst {
            asked: Default::default(),
            hold: hold.clone(),
        }),
    );
    let mut run = start_with(task, runtime);
    let held = |event: &AgentEvent| {
        matches!(
            event,
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            }
        )
    };
    let events = run.events_until(held).await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
            cause: crate::task::ErrorRecoveryCause::BudgetExceeded,
            ..
        })
    )));
    run.commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    run.events_until(held).await;
    assert_eq!(fake.logged("argv").len(), 1, "no session while held");
    hold.store(false, std::sync::atomic::Ordering::SeqCst);
    run.commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let argvs = fake.logged("argv");
    assert_eq!(argvs.len(), 2);
    let second: Vec<String> = serde_json::from_value(argvs[1]["argv"].clone()).unwrap();
    assert!(second.iter().any(|arg| arg.starts_with("--resume=")));
}

#[test]
fn built_in_tools_are_only_for_claude_code_models() {
    let fake = Fake::new(json!([]));
    let mut config = config(&fake, &["Read"]);
    assert!(config.validate().is_ok());
    config.tool_config.available_tool_names = vec!["AskUserQuestion".into()];
    assert!(config.validate().is_err());
    config.tool_config.available_tool_names = vec!["Bash".into()];
    config.tool_config.tool_specific_configs.insert(
        "Bash".into(),
        PerToolConfigOverride {
            tool_timeout_seconds: Some(Some(30)),
            ..PerToolConfigOverride::default()
        },
    );
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("tool_timeout_seconds"), "{error}");
    config.tool_config.tool_specific_configs.clear();

    config.llm_config.model = arko_ai::Model::anthropic_claude_opus_5_5();
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("Claude Code built-in tool"), "{error}");
    config.tool_config.available_tool_names.clear();
    assert!(
        config.validate().is_err(),
        "claude_code settings on an API model"
    );
    config.claude_code = None;
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn a_response_sent_as_several_messages_is_counted_once_in_full() {
    let fake = Fake::new(json!([[
        {"text": "Let me finish.", "input_tokens": 900, "output_tokens": 3},
        {"tool": "mcp__arko__complete_task", "input": {"summary": "done"},
         "input_tokens": 900, "output_tokens": 40, "continues": true},
    ]]));
    let task = Task::new(
        config(&fake, &[]),
        "Do the thing.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.usage_history.len(), 1);
    assert_eq!(task.usage_history[0].output_tokens, 40);
    assert_eq!(task.usage_history[0].input_tokens, 900);
    let reported = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::ModelCallUsageReported(_)))
        .count();
    assert_eq!(reported, 1);
}

#[tokio::test]
async fn shell_commands_pass_the_command_guard() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "ls -la"}},
        {"tool": "Bash", "input": {"command": "rm -rf ~"}},
        complete("Listed"),
    ]]));
    let task = Task::new(
        config(&fake, &["Bash"]),
        "List files.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let posts = fake.logged("post");
    assert_eq!(posts[0]["output"], "builtin Bash ran");
    let refused = posts[1]["output"].as_str().unwrap();
    assert_eq!(posts[1]["is_error"], true);
    // Claude Code shows it as "PreToolUse:Bash hook error: <this>".
    assert!(
        refused.starts_with("arko did not run this Bash call: "),
        "{refused}"
    );

    // Turned off, the command runs.
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "rm -rf ~"}},
        complete("Ran"),
    ]]));
    let mut config = config(&fake, &["Bash"]);
    config
        .claude_code
        .as_mut()
        .unwrap()
        .dangerously_allow_all_commands = true;
    let task = Task::new(config, "Run it.".into(), fake.workspace(), None);
    let (task, _) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(fake.logged("post")[0]["output"], "builtin Bash ran");
}

#[cfg(any(unix, windows))]
#[tokio::test]
async fn command_rules_refuse_shell_commands() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "git status --short"}},
        {"tool": "Bash", "input": {"command": "cd sub && git push"}},
        complete("Checked"),
    ]]));
    let mut config = config(&fake, &["Bash"]);
    config.claude_code.as_mut().unwrap().command_rules =
        vec![crate::terminal::CommandRule::allow("git status *").because("Only status here.")];
    let task = Task::new(config, "Check.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let posts = fake.logged("post");
    assert_eq!(posts[0]["output"], "builtin Bash ran");
    assert_eq!(posts[1]["is_error"], true);
    let refused = posts[1]["output"].as_str().unwrap();
    assert!(
        refused.contains("Refused: `git push` isn't one of the `git` commands allowed here")
            && refused.contains("Only status here."),
        "{refused}"
    );
    // The host sees the refusal as a rejected call.
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolRejected { tool_name, reason: Some(reason), .. }
            if tool_name == "Bash" && reason.starts_with("Refused: `git push`")
    )));
}

#[tokio::test]
async fn a_call_held_for_the_report_says_it_is_not_an_error() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"text": "REPORT"},
    ], [complete("done")]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let run = start(task);
    run.commands
        .send(UserCommand::RequestHandover)
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert_eq!(task.handovers.len(), 1);
    let held = fake.logged("post")[0]["output"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        held.starts_with(
            "Not an error: arko held this Grep call so you can write your handover report now."
        ),
        "{held}"
    );
}

fn windows(five_hour: f64, seven_day: f64) -> Value {
    let later = chrono::Utc::now().timestamp() + 3600;
    json!({
        "five_hour": {"utilization": five_hour, "resetsAt": later},
        "seven_day": {"utilization": seven_day, "resetsAt": later + 86_400},
    })
}

#[tokio::test]
async fn plan_usage_is_reported_and_the_threshold_pauses() {
    let fake = Fake::new(json!([
        [
            {"rate_limit": {"status": "allowed_warning", "rateLimitType": "five_hour",
                            "unifiedWindows": windows(0.9, 0.2)}},
            {"tool": "Grep", "input": {"pattern": "a"}},
            {"tool": "Grep", "input": {"pattern": "b"}},
        ],
        [complete("done")],
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.monitoring.plan_usage = Some(crate::agent_config::PlanUsageThreshold {
        percent: 80.0,
        action: crate::agent_config::PlanUsageAction::Pause,
    });
    let task = Task::new(config, "Search.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run
        .events_until(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            )
        })
        .await;
    let Some(AgentEvent::PlanUsageReported(usage)) = events
        .iter()
        .find(|event| matches!(event, AgentEvent::PlanUsageReported(_)))
    else {
        panic!("{events:#?}");
    };
    assert_eq!(usage.plan, Some(arko_ai::SubscriptionPlan::ClaudeCode));
    assert_eq!(usage.primary.as_ref().unwrap().used_percent, 90.0);
    assert_eq!(usage.primary.as_ref().unwrap().window_minutes, Some(300));
    assert!((usage.secondary.as_ref().unwrap().used_percent - 20.0).abs() < 1e-9);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::PlanUsageThresholdReached { window, .. } if window == "primary"
    )));
    run.commands
        .send(UserCommand::ApproveErrorRecovery { message: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // The calls after the report carry it.
    assert!(
        task.model_call_usage_history
            .iter()
            .any(|call| call.plan_usage.is_some())
    );
    assert_eq!(
        fake.logged("post").len(),
        2,
        "the second call ran before the interrupt"
    );
}

#[tokio::test]
async fn a_failure_at_a_reached_plan_limit_waits_for_its_reset() {
    let resets_at = chrono::Utc::now().timestamp() as u64 + 7200;
    let fake = Fake::new(json!([[
        {"rate_limit": {"status": "rejected", "rateLimitType": "seven_day", "resetsAt": resets_at,
                        "unifiedWindows": windows(0.1, 1.0)}},
        {"fail": "Claude usage limit reached."},
    ]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run
        .events_until(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            )
        })
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::UsageLimitReached { resets_at_unix_ms: Some(at), .. } if *at == resets_at * 1000
    )));
    let Some(AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
        cause, ..
    })) = events.last()
    else {
        unreachable!()
    };
    assert_eq!(
        *cause,
        crate::task::ErrorRecoveryCause::UsageLimit {
            resets_at_unix_ms: Some(resets_at * 1000)
        }
    );
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    run.finish().await;
}

#[tokio::test]
async fn the_cache_monitor_judges_the_main_conversation() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 1000, "cached": 0},
        {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1100, "cached": 0},
        {"tool": "Grep", "input": {"pattern": "c"}, "input_tokens": 1200, "cached": 0},
        complete("done"),
    ]]));
    let mut config = config(&fake, &["Grep"]);
    config.monitoring.cache = Some(crate::agent_config::CacheMonitorConfig {
        warmup_calls: 1,
        window_calls: 2,
        warn_below: Some(0.5),
        pause_below: None,
    });
    let task = Task::new(config, "Search.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::CacheEfficiencyReported { .. }))
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::CacheEfficiencyLow { pausing: false, .. }))
            .count(),
        1
    );
}

fn refusal_message(subtype: &str) -> Value {
    json!({"system": {
        "subtype": subtype, "original_model": "claude-opus-5-5", "fallback_model": "claude-sonnet-5",
        "api_refusal_category": "cyber", "api_refusal_explanation": "The request was refused.",
    }})
}

#[tokio::test]
async fn claude_codes_own_refusal_retry_is_reported() {
    let fake = Fake::new(json!([[
        refusal_message("model_refusal_fallback"),
        complete("done")
    ]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::RefusalReceived(refusal) if refusal.category.as_deref() == Some("cyber")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::RefusalFallback { model_id } if model_id == "claude-sonnet-5"
    )));
}

#[tokio::test]
async fn arko_tools_get_long_mcp_limits() {
    // An inherited short limit (a cloud session sets 60 seconds) would cut
    // off a delegated agent run that takes minutes.
    let fake = Fake::new(json!([[complete("done")]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    start(task).finish().await;
    let env = &fake.logged("argv")[0]["env"];
    assert_eq!(env["MCP_TOOL_TIMEOUT"], "604800000");
    assert_eq!(env["CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT"], "0");
}

#[tokio::test]
async fn bash_calls_start_in_the_workspace() {
    // The command guard checks each Bash call from the workspace; by default
    // the CLI would keep a `cd` from the call before. The settings' `env`
    // can still change it.
    const VARIABLE: &str = "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR";
    let fake = Fake::new(json!([[complete("done")], [complete("done")]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    start(task).finish().await;
    let mut config = config(&fake, &[]);
    let settings = config.claude_code.as_mut().unwrap();
    settings.env.insert(VARIABLE.into(), "0".into());
    let task = Task::new(config, "Do it.".into(), fake.workspace(), None);
    start(task).finish().await;
    let runs = fake.logged("argv");
    assert_eq!(runs[0]["env"][VARIABLE], "1");
    assert_eq!(runs[1]["env"][VARIABLE], "0");
}

#[tokio::test]
async fn metered_billing_bills_the_estimate() {
    let fake = Fake::new(json!([[complete("done")]]));
    let mut config = config(&fake, &[]);
    config.claude_code.as_mut().unwrap().metered_billing = true;
    let task = Task::new(config, "Do it.".into(), fake.workspace(), None);
    let (task, _) = start(task).finish().await;
    let usage = &task.usage_history[0];
    assert_eq!(usage.billing, arko_ai::BillingKind::Metered);
    assert!(usage.billed() > 0.0);
    assert_eq!(usage.billed(), usage.total_cost);
}

#[tokio::test]
async fn claude_codes_retry_is_recorded_as_live() {
    // As Claude Code 2.1.285 sends it: the refused response's stop_details
    // carry the explanation the system message leaves out, and the rest of
    // the session runs on the fallback model.
    let fake = Fake::new(json!([[
        {"refused": {"category": "cyber", "explanation": "Blocked under the usage policy."}},
        {"system": {
            "subtype": "model_refusal_fallback", "trigger": "refusal", "direction": "retry",
            "scope": "session", "original_model": "claude-opus-5-5",
            "fallback_model": "claude-opus-4-8", "api_refusal_category": "cyber",
            "api_refusal_explanation": null,
        }},
        dict_with(complete("done"), "model", "claude-opus-4-8"),
    ]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::RefusalReceived(refusal)
            if refusal.category.as_deref() == Some("cyber")
                && refusal.explanation.as_deref() == Some("Blocked under the usage policy.")
    )));
    let calls: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ModelCallUsageReported(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].stop_reason.as_deref(), Some("refusal"));
    assert_eq!(calls[0].stop_details.as_ref().unwrap()["category"], "cyber");
    assert_eq!(calls[1].model, "claude-opus-4-8");
    // The fallback's call is priced as the fallback model.
    let mut expected = arko_ai::UsageInfo::new(10, 5).with_cached_tokens(8);
    expected.calculate_cost(&arko_ai::Model::anthropic_claude_opus_4_8(), None);
    assert_eq!(task.usage_history[1].total_cost, expected.total_cost);
}

fn dict_with(mut step: Value, field: &str, value: &str) -> Value {
    step[field] = json!(value);
    step
}

#[tokio::test]
async fn a_standing_refusal_goes_to_the_fallback_for_one_turn() {
    let fake = Fake::new(json!([
        [refusal_message("model_refusal"), {"text": "I can't help with that."}],
        [{"text": "Continuing on the fallback."}],
        [complete("done")],
    ]));
    let mut config = config(&fake, &[]);
    config.refusal_fallback = Some(AgentLLMConfig {
        model: arko_ai::Model::claude_code(arko_ai::Model::anthropic_claude_sonnet_5_5()),
        generation_config: Default::default(),
    });
    let task = Task::new(config, "Do it.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let switches: Vec<String> = fake
        .logged("set_model")
        .iter()
        .map(|entry| entry["model"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(switches, ["claude-sonnet-5-5", "claude-opus-5-5"]);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::RefusalFallback { model_id } if model_id == "claude-sonnet-5-5"
    )));
}

#[tokio::test]
async fn without_a_fallback_a_standing_refusal_pauses() {
    let fake = Fake::new(json!([[refusal_message("model_refusal"), {"text": "No."}]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run
        .events_until(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            )
        })
        .await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::TaskStateChanged(
            TaskState::AwaitingErrorRecoveryApproval {
                cause: crate::task::ErrorRecoveryCause::Refusal,
                ..
            }
        ))
    ));
    // Reported with its error, as every pause is.
    let reported = run
        .events_until(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
        .await;
    assert!(
        matches!(
            reported.as_slice(),
            [AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            }]
        ),
        "{reported:#?}"
    );
    // Claude Code keeps its conversation: a change to the refused input is
    // answered with an error to decide, the pause standing (the command
    // was once ignored, and a host waited for ever).
    run.commands
        .send(UserCommand::ResolveRefusal {
            changes: vec![crate::task::RefusalResolution::RemoveUserMessage { message_index: 0 }],
        })
        .await
        .unwrap();
    let answered = run
        .events_until(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
        .await;
    assert!(
        matches!(
            answered.last(),
            Some(AgentEvent::ErrorEncountered {
                error_message,
                requires_approval_to_retry: true,
                ..
            }) if error_message.starts_with("Could not resolve the refusal")
        ),
        "{answered:#?}"
    );
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
}

/// A request that fails because Claude Code can't authenticate (it isn't
/// signed in, or its login expired or was revoked), can't bill its account
/// or isn't allowed to make it pauses as one retrying can't fix, by the
/// error's kind or status Claude Code reports; another failed request pauses as retries exhausted, as
/// before. Claude Code reports these on the first turn, not at its start
/// (checked with Claude Code 2.1.286). The pause records the error's kind
/// when Claude Code's report names one.
#[tokio::test]
async fn a_failed_login_pauses_as_non_retryable() {
    use crate::task::ErrorRecoveryCause;
    use arko_ai::ErrorKind;
    for (kind, status, expected, error_kind) in [
        (
            json!("authentication_failed"),
            json!(null),
            ErrorRecoveryCause::NonRetryable,
            Some(ErrorKind::Authentication),
        ),
        (
            json!(null),
            json!(401),
            ErrorRecoveryCause::NonRetryable,
            Some(ErrorKind::Authentication),
        ),
        (
            json!("server_error"),
            json!(500),
            ErrorRecoveryCause::RetriesExhausted {
                retry_after_ms: None,
            },
            None,
        ),
        (
            json!("billing_error"),
            json!(400),
            ErrorRecoveryCause::NonRetryable,
            Some(ErrorKind::Billing),
        ),
        (
            json!("oauth_org_not_allowed"),
            json!(null),
            ErrorRecoveryCause::NonRetryable,
            Some(ErrorKind::Permission),
        ),
        (
            json!(null),
            json!(403),
            ErrorRecoveryCause::NonRetryable,
            Some(ErrorKind::Permission),
        ),
    ] {
        let fake = Fake::new(json!([[
            {"system": {"subtype": "api_retry", "attempt": 1, "error": kind, "error_status": status}},
            {"api_error": kind, "text": "Not logged in · Please run /login", "status": status},
        ]]));
        let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
        let mut run = start(task);
        let events = run
            .events_until(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
            .await;
        let pause = events.iter().find_map(|event| match event {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                error_message,
                cause,
                error_kind,
            }) => Some((error_message.clone(), cause.clone(), *error_kind)),
            _ => None,
        });
        let (message, cause, recorded) = pause.unwrap_or_else(|| panic!("{events:#?}"));
        assert_eq!(cause, expected, "{kind} {status}");
        assert_eq!(recorded, error_kind, "{kind} {status}");
        assert!(message.contains("Not logged in"), "{message}");
        assert!(matches!(
            events.last(),
            Some(AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                error_kind: reported,
                ..
            }) if *reported == error_kind
        ));
        run.commands.send(UserCommand::CancelTask).await.unwrap();
        run.finish().await;
    }
}

fn slow_down_config(fake: &Fake, delay_ms: u64) -> AgentConfig {
    let mut config = config(fake, &["Grep"]);
    config.monitoring.plan_usage = Some(crate::agent_config::PlanUsageThreshold {
        percent: 80.0,
        action: crate::agent_config::PlanUsageAction::SlowDown { delay_ms },
    });
    config
}

#[tokio::test]
async fn slow_down_holds_each_call_while_usage_is_over_the_threshold() {
    let fake = Fake::new(json!([[
        {"rate_limit": {"status": "allowed_warning", "unifiedWindows": windows(0.9, 0.1)}},
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "Grep", "input": {"pattern": "b"}},
        complete("done"),
    ]]));
    let task = Task::new(
        slow_down_config(&fake, 300),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let started = std::time::Instant::now();
    let (task, _) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // Three calls, each held 300 ms.
    assert!(
        started.elapsed() >= Duration::from_millis(900),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_cancellation_ends_a_slow_down_at_once() {
    let fake = Fake::new(json!([[
        {"rate_limit": {"status": "allowed_warning", "unifiedWindows": windows(0.9, 0.1)}},
        {"tool": "Grep", "input": {"pattern": "a"}},
        complete("done"),
    ]]));
    let task = Task::new(
        slow_down_config(&fake, 60_000),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    run.events_until(|event| matches!(event, AgentEvent::ToolCallFormatted { .. }))
        .await;
    let cancelled = std::time::Instant::now();
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert!(
        cancelled.elapsed() < Duration::from_secs(10),
        "{:?}",
        cancelled.elapsed()
    );
    let posts = fake.logged("post");
    assert!(
        posts[0]["output"]
            .as_str()
            .unwrap()
            .contains("the agent is stopping")
    );
}

#[tokio::test]
async fn wait_until_reset_waits_out_the_limit_and_continues() {
    let resets_at = chrono::Utc::now().timestamp() as u64 + 1;
    let fake = Fake::new(json!([
        [
            {"rate_limit": {"status": "rejected", "rateLimitType": "five_hour", "resetsAt": resets_at,
                            "unifiedWindows": windows(1.0, 0.2)}},
            {"fail": "Claude usage limit reached."},
        ],
        [complete("done after the reset")],
    ]));
    let mut config = config(&fake, &[]);
    config.llm_retry_config.jitter = false;
    config.llm_retry_config.usage_limit_policy =
        crate::agent_config::UsageLimitPolicy::WaitUntilReset {
            max_wait_ms: 60_000,
        };
    let task = Task::new(config, "Do it.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "done after the reset"),
        "{:?}",
        task.current_state
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::UsageLimitReached { waiting_until_unix_ms: Some(_), resets_at_unix_ms: Some(at), .. }
            if *at == resets_at * 1000
    )));
    // No pause: the task waited and resumed its session.
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
    )));
    let second: Vec<String> =
        serde_json::from_value(fake.logged("argv")[1]["argv"].clone()).unwrap();
    assert!(second.iter().any(|arg| arg.starts_with("--resume=")));
}

#[tokio::test]
async fn one_hour_cache_writes_are_recorded_and_the_ttl_is_passed_on() {
    let fake = Fake::new(json!([[
        {"text": "Starting.", "input_tokens": 3000, "cached": 0, "written_1h": 2000},
        complete("done"),
    ]]));
    let mut config = config(&fake, &[]);
    config.llm_config.generation_config.anthropic_cache_ttl =
        Some(arko_ai::AnthropicCacheTtl::OneHour);
    let task = Task::new(config, "Do it.".into(), fake.workspace(), None);
    let (task, _) = start(task).finish().await;
    let usage = &task.usage_history[0];
    assert_eq!(usage.cache_write_tokens, Some(2000));
    assert_eq!(usage.cache_write_1h_tokens, Some(2000));
    // Opus 5.5's 1-hour write price, $8 per million.
    let expected = 2000.0 * 8.00 / 1_000_000.0;
    assert!(
        (usage.cache_write_cost.unwrap() - expected).abs() < 1e-9,
        "{:?}",
        usage.cache_write_cost
    );
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    let settings = argv.iter().position(|arg| arg == "--settings").unwrap();
    assert!(
        argv[settings + 1].contains("\"promptCacheTtl\":\"1h\""),
        "{argv:?}"
    );
    assert_eq!(
        arko_ai::prompt_cache::prompt_cache_lifetime(
            &task.agent_config.llm_config.model,
            &task.agent_config.llm_config.generation_config,
            false
        ),
        arko_ai::PromptCacheLifetime::Known(Duration::from_secs(3600))
    );
}

/// Keeps every task snapshot the runtime reports.
#[derive(Default)]
struct Recorder(std::sync::Mutex<Vec<(Task, bool)>>);

#[async_trait::async_trait]
impl crate::snapshot::TaskSnapshotSink for Recorder {
    async fn task_changed(&self, task: &Task, durable: bool) -> Result<(), String> {
        self.0.lock().unwrap().push((task.clone(), durable));
        Ok(())
    }
}

impl Recorder {
    fn snapshots(&self) -> Vec<(Task, bool)> {
        self.0.lock().unwrap().clone()
    }

    /// The latest snapshot `pick` accepts.
    fn latest(&self, pick: impl Fn(&Task) -> bool) -> Task {
        self.snapshots()
            .into_iter()
            .rev()
            .find(|(task, _)| pick(task))
            .map(|(task, _)| task)
            .expect("a snapshot")
    }
}

fn recording(recorder: &Arc<Recorder>) -> AgentRuntime {
    AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_task_snapshots(recorder.clone())
}

/// Runs a task saved by the host that crashed (what the recorder saved)
/// again, and returns it with the fake's first message to the resumed
/// session.
async fn run_saved(fake: &Fake, saved: &Task) -> (Task, String) {
    let saved: Task = serde_json::from_str(&serde_json::to_string(saved).unwrap()).unwrap();
    let sent_before = fake.logged("user").len();
    let (task, _) = start(saved).finish().await;
    let resumed = fake.logged("user")[sent_before]["text"]
        .as_str()
        .unwrap()
        .to_string();
    let argvs = fake.logged("argv");
    let argv: Vec<String> = serde_json::from_value(argvs.last().unwrap()["argv"].clone()).unwrap();
    assert!(
        argv.iter().any(|arg| arg.starts_with("--resume=")),
        "{argv:?}"
    );
    (task, resumed)
}

#[tokio::test]
async fn the_session_is_saved_before_it_is_sent_anything() {
    let fake = Fake::new(json!([[complete("done")]]));
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let (task, _) = start_with(task, recording(&recorder)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let snapshots = recorder.snapshots();
    let (first_durable, _) = snapshots
        .iter()
        .find(|(_, durable)| *durable)
        .expect("a durable snapshot");
    assert_eq!(first_durable.claude_code_session, task.claude_code_session);
    assert!(first_durable.claude_code_session.is_some());
    // What the session is sent first (the task prompt) is saved with it.
    assert_eq!(first_durable.claude_code_synced, 1);
    let (last, _) = snapshots.last().unwrap();
    assert!(matches!(last.current_state, TaskState::Completed { .. }));
    assert!(last.running_tool_calls.is_empty());
}

/// A read-only built-in call is recorded running without waiting for the
/// save; any other call waits.
#[tokio::test]
async fn only_calls_with_effects_wait_for_the_save() {
    let fake = Fake::new(json!([[
        {"tool": "Read", "input": {"file_path": "a.txt"}, "id": "toolu_read"},
        {
            "tool": "mcp__arko__write_file",
            "input": {"path": "b.txt", "content": "x"},
            "id": "toolu_write",
        },
        complete("done"),
    ]]));
    std::fs::write(fake.workspace().join("a.txt"), "A").unwrap();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["Read", "write_file"]),
        "Read a, write b.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start_with(task, recording(&recorder)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let first_report = |call: &str| {
        recorder
            .snapshots()
            .into_iter()
            .find(|(task, _)| {
                task.running_tool_calls
                    .iter()
                    .any(|running| running.call_id.as_deref() == Some(call))
            })
            .map(|(_, durable)| durable)
            .expect("the call is recorded")
    };
    assert!(!first_report("toolu_read"));
    assert!(first_report("toolu_write"));
}

#[tokio::test]
async fn a_call_running_when_the_host_stopped_is_reported_when_the_session_resumes() {
    let fake = Fake::new(json!([
        [{
            "tool": "mcp__arko__write_file",
            "input": {"path": "a.txt", "content": "x"},
            "id": "toolu_write",
            "crash": "pre",
        }],
        [complete("Recovered")],
    ]));
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["write_file"]),
        "Write a.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start_with(task, recording(&recorder));
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    // The host crashed while the call ran: what it saved last then is what
    // it runs again.
    run.handle.abort();
    let saved = recorder.latest(|task| {
        !task.running_tool_calls.is_empty() && task.current_state == TaskState::InProgress
    });
    let open = &saved.running_tool_calls[0];
    assert_eq!(open.call_id.as_deref(), Some("toolu_write"));
    assert_eq!(open.tool_name, "write_file");
    assert!(!open.built_in && open.output.is_none());
    // The call was saved before it was allowed.
    let durable = recorder
        .snapshots()
        .into_iter()
        .find(|(task, _)| !task.running_tool_calls.is_empty())
        .unwrap();
    assert!(durable.1, "the first snapshot with the call is durable");

    let (task, resumed) = run_saved(&fake, &saved).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        resumed.contains("`mcp__arko__write_file` (toolu_write)"),
        "{resumed}"
    );
    assert!(resumed.contains("its outcome is unknown"), "{resumed}");
    assert!(resumed.contains("Continue the task"), "{resumed}");
    assert!(task.running_tool_calls.is_empty());
}

#[tokio::test]
async fn a_rerun_tool_open_at_the_stop_runs_again_for_the_resumed_session() {
    let fake = Fake::new(json!([
        [{
            "tool": "mcp__arko__read_file",
            "input": {"path": "a.txt"},
            "id": "toolu_read",
            "crash": "pre",
        }],
        [complete("Recovered")],
    ]));
    std::fs::write(fake.workspace().join("a.txt"), "CONTENTS-OF-A").unwrap();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["read_file"]),
        "Read a.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start_with(task, recording(&recorder));
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    run.handle.abort();
    let saved = recorder.latest(|task| {
        !task.running_tool_calls.is_empty() && task.current_state == TaskState::InProgress
    });
    let (task, resumed) = run_saved(&fake, &saved).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        resumed.contains("`mcp__arko__read_file` (toolu_read) was running, so it was run again."),
        "{resumed}"
    );
    assert!(resumed.contains("CONTENTS-OF-A"), "{resumed}");
}

#[tokio::test]
async fn recovered_tool_images_are_sent_and_survive_saving_before_delivery() {
    let fake = Fake::new(json!([
        [{
            "tool": "mcp__arko__read_file",
            "input": {"path": "a.png"},
            "id": "toolu_image",
            "crash": "pre",
        }],
        [complete("Recovered")],
        [complete("Recovered from saved note")],
    ]));
    let bytes = b"\x89PNG\r\n\x1a\nimage-fixture";
    std::fs::write(fake.workspace().join("a.png"), bytes).unwrap();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["read_file"]),
        "Read the image.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start_with(task, recording(&recorder));
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    run.handle.abort();
    let saved = recorder.latest(|task| {
        !task.running_tool_calls.is_empty() && task.current_state == TaskState::InProgress
    });
    let saved: Task = serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
    let recovered = Arc::new(Recorder::default());
    let (task, _) = start_with(saved, recording(&recovered)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // The fake logs array input as JSON inside its `text` field.
    let sent: Value =
        serde_json::from_str(fake.logged("user")[1]["text"].as_str().unwrap()).unwrap();
    let blocks = sent.as_array().expect("recovered image input uses blocks");
    let image = arko_ai::Image::from_bytes(bytes, "image/png");
    let arko_ai::Image::Base64 { data, mime_type } = &image else {
        unreachable!()
    };
    assert_eq!(
        blocks.last().unwrap(),
        &json!({
            "type": "image",
            "source": {"type": "base64", "data": data, "media_type": mime_type},
        })
    );
    assert!(
        blocks[blocks.len() - 2]["text"]
            .as_str()
            .unwrap()
            .contains("`mcp__arko__read_file` (toolu_image)"),
        "{sent}"
    );
    let pending = recovered.latest(|task| {
        task.running_tool_calls.is_empty()
            && task.claude_code_synced < task.conversation.messages.len()
            && task.conversation.messages.last().is_some_and(|message| {
                message
                    .content
                    .iter()
                    .any(|content| content.as_image() == Some(&image))
            })
    });
    // Resume the snapshot taken after recovery, before delivery. The image
    // must come from saved state even though the source file no longer exists.
    std::fs::remove_file(fake.workspace().join("a.png")).unwrap();
    let (task, resumed) = run_saved(&fake, &pending).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(serde_json::from_str::<Value>(&resumed).unwrap(), sent);
    assert!(task.running_tool_calls.is_empty());
}

#[tokio::test]
async fn completed_tool_images_survive_a_stop_before_claude_code_records_the_result() {
    let fake = Fake::new(json!([
        [{
            "tool": "mcp__arko__read_file",
            "input": {"path": "done.png"},
            "id": "toolu_image_done",
            "crash": "post",
        }],
        [complete("Recovered completed image")],
    ]));
    let path = fake.workspace().join("done.png");
    let bytes = b"\x89PNG\r\n\x1a\ncompleted-image-fixture";
    std::fs::write(&path, bytes).unwrap();
    let image = arko_ai::Image::from_bytes(bytes, "image/png");
    let arko_ai::Image::Base64 { data, mime_type } = &image else {
        unreachable!()
    };
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["read_file"]),
        "Inspect the image.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start_with(task, recording(&recorder));
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    run.handle.abort();
    let saved = recorder.latest(|task| {
        task.current_state == TaskState::InProgress
            && task
                .running_tool_calls
                .iter()
                .any(|call| call.output.is_some())
    });
    let call = &saved.running_tool_calls[0];
    assert_eq!(call.output_images.len(), 1);
    assert_eq!(call.output_images[0].image, image);
    assert!(!call.output.as_ref().unwrap().contains(data));
    assert!(
        !format!("{call:?}").contains(data),
        "debug must omit pixels"
    );
    // The CLI's hook in this fixture returns text only. The host must have
    // kept the image it served, and replay it without executing read_file.
    std::fs::remove_file(path).unwrap();
    let (task, resumed) = run_saved(&fake, &saved).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let sent: Value = serde_json::from_str(&resumed).unwrap();
    let blocks = sent.as_array().expect("saved images use content blocks");
    assert!(
        blocks[0]["text"]
            .as_str()
            .unwrap()
            .contains("had completed")
    );
    assert_eq!(
        blocks.last().unwrap(),
        &json!({
            "type": "image",
            "source": {"type": "base64", "data": data, "media_type": mime_type},
        })
    );
    assert!(task.running_tool_calls.is_empty());
}

#[test]
fn reported_image_results_keep_pixels_out_of_text() {
    let blocks = json!([
        {"type": "text", "text": "Two images"},
        {"type": "image", "data": "bWNw", "mimeType": "image/png"},
        {"type": "image", "source": {"type": "base64", "data": "Y2xhdWRl", "media_type": "image/jpeg"}},
    ]);
    for content in [blocks.clone(), json!({"content": blocks, "isError": false})] {
        assert_eq!(super::result_text(&content), "Two images");
        let images = super::result_images(&content);
        assert_eq!(images.len(), 2);
        assert_eq!(
            images[0].image,
            arko_ai::Image::new_base64("bWNw", "image/png")
        );
        assert_eq!(
            images[1].image,
            arko_ai::Image::new_base64("Y2xhdWRl", "image/jpeg")
        );
    }
    assert_eq!(
        super::result_text(&json!({"type": "image", "data": "bWNw", "mimeType": "image/png"})),
        ""
    );
}

#[tokio::test]
async fn a_call_that_finished_before_claude_code_exited_is_reported_completed() {
    let fake = Fake::new(json!([
        [{
            "tool": "mcp__arko__write_file",
            "input": {"path": "a.txt", "content": "x"},
            "id": "toolu_write",
            "crash": "post",
        }],
        [complete("Recovered")],
    ]));
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["write_file"]),
        "Write a.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start_with(task, recording(&recorder));
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    assert!(fake.workspace().join("a.txt").exists(), "the call ran");
    let saved = recorder.latest(|task| !task.running_tool_calls.is_empty());
    assert!(
        saved.running_tool_calls[0].output.is_some(),
        "its result is saved"
    );
    // Claude Code exited on its own; the host continues the task.
    run.commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let resumed = fake.logged("user")[1]["text"].as_str().unwrap().to_string();
    assert!(
        resumed.contains("`mcp__arko__write_file` (toolu_write) had completed. Its result was:"),
        "{resumed}"
    );
    // The note follows the host's continue note, and isn't a second one.
    assert_eq!(resumed.matches("Continue the task").count(), 1, "{resumed}");
    assert!(task.running_tool_calls.is_empty());
}

/// Line ranges Claude Code passes to arko's `read_file` while they are
/// off are ignored: the call reads the whole file. On, they read only those
/// lines.
#[tokio::test]
async fn read_file_line_ranges_passed_while_off_are_ignored() {
    for ranges in [false, true] {
        let fake = Fake::new(json!([[
            {"tool": "mcp__arko__read_file", "input": {"path": "a.txt", "start_line": 2, "end_line": 2}},
            complete("done"),
        ]]));
        std::fs::write(fake.workspace().join("a.txt"), "one\ntwo\nthree\n").unwrap();
        let mut config = config(&fake, &["read_file"]);
        config.tool_config.read_file_line_ranges = ranges;
        let task = Task::new(config, "Read it.".into(), fake.workspace(), None);
        let (task, events) = start(task).finish().await;
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        let output = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::ToolResult {
                    tool_name, output, ..
                } if tool_name == "mcp__arko__read_file" || tool_name == "read_file" => {
                    Some(output.clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("{events:#?}"));
        if ranges {
            assert!(output.starts_with("2 | two\n"), "{output}");
            assert!(!output.contains("1 | one"), "{output}");
        } else {
            assert!(
                output.starts_with("1 | one\n2 | two\n3 | three"),
                "{output}"
            );
        }
    }
}

#[tokio::test]
async fn calls_that_ended_normally_are_not_reported_again() {
    let fake = Fake::new(json!([
        [
            {"tool": "mcp__arko__read_file", "input": {"path": "a.txt"}},
            {"tool": "mcp__arko__ask_user", "input": {"questions": "Which one?"}},
        ],
        [complete("done")],
    ]));
    std::fs::write(fake.workspace().join("a.txt"), "A").unwrap();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["read_file", "ask_user"]),
        "Read and ask.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start_with(task, recording(&recorder));
    run.events_until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    // Until the session closes, the finished calls are kept with their
    // results; once it closed cleanly for the wait, nothing is open.
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            let waiting = recorder.latest(|task| {
                matches!(task.current_state, TaskState::AwaitingUserResponse { .. })
            });
            if waiting.running_tool_calls.is_empty() {
                break;
            }
            assert!(
                waiting
                    .running_tool_calls
                    .iter()
                    .all(|call| call.output.is_some())
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the closed session's calls are dropped");
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "The first.".into(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let resumed = fake.logged("user")[1]["text"].as_str().unwrap().to_string();
    assert!(!resumed.contains("tool calls were open"), "{resumed}");
}
