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
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
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
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    run.finish().await;
}
