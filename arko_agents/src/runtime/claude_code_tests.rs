//! The Claude Code runtime against a scripted fake CLI
//! (`tests/fixtures/fake_claude_agent.py`; `ARKO_TEST_PYTHON` selects the
//! interpreter).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
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

#[path = "claude_code_latest_request_tests.rs"]
mod latest_request;

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

    /// The file whose existence releases a gated `serve` step of the fake.
    fn gate(&self) -> PathBuf {
        self.dir.join("gate")
    }

    /// What the fake answers `get_usage` with. Without it the fake gives an
    /// error, as a CLI that doesn't know the request does.
    fn answer_usage(&self, answer: Value) {
        std::fs::write(self.dir.join("usage.json"), answer.to_string()).unwrap();
    }

    fn settings(&self) -> ClaudeCodeSettings {
        let python = std::env::var("ARKO_TEST_PYTHON")
            .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_claude_agent.py");
        let path = |name: &str| self.dir.join(name).display().to_string();
        let mut env = BTreeMap::from([
            ("FAKE_CLAUDE_SCRIPT".to_string(), path("script.json")),
            ("FAKE_CLAUDE_LOG".to_string(), path("log.jsonl")),
            ("FAKE_CLAUDE_STATE".to_string(), path("state")),
            ("FAKE_CLAUDE_GATE".to_string(), path("gate")),
        ]);
        if self.dir.join("usage.json").exists() {
            env.insert("FAKE_CLAUDE_USAGE".to_string(), path("usage.json"));
        }
        ClaudeCodeSettings {
            cli_path: Some(python.into()),
            cli_args: vec![script.display().to_string()],
            env,
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

/// Waits, for up to 20 seconds, until `done`.
async fn until(done: impl Fn() -> bool) {
    for _ in 0..400 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("not in time");
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

/// The tools that only read are served with the MCP `readOnlyHint`, so
/// Claude Code runs a response's calls to them together. The others have
/// no annotations.
#[tokio::test]
async fn read_only_tools_are_served_with_the_read_only_hint() {
    let fake = Fake::new(json!([[complete("Done")]]));
    let task = Task::new(
        config(&fake, &["read_file", "list_files", "write_file"]),
        "Look around.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let served = &fake.logged("tools")[0]["tools"];
    let hints: Vec<(&str, &Value)> = served
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| (tool["name"].as_str().unwrap(), &tool["annotations"]))
        .collect();
    let read_only = json!({"readOnlyHint": true});
    assert_eq!(
        hints,
        [
            ("complete_task", &Value::Null),
            ("list_files", &read_only),
            ("read_file", &read_only),
            ("write_file", &Value::Null),
        ]
    );
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

/// A thinking block reaches the host as reasoning, before the response's
/// text, and stays out of the agent's text in the conversation.
#[tokio::test]
async fn thinking_reaches_the_host_as_reasoning_before_the_text() {
    let fake = Fake::new(json!([[
        {"thinking": "The thing is small.\nDo it at once."},
        {"text": "Doing it.", "continues": true},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &[]),
        "Do the thing.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let streamed: Vec<(&str, &str)> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ReasoningChunk(text) => Some(("thinking", text.as_str())),
            AgentEvent::AssistantResponseChunk(text) => Some(("text", text.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        streamed,
        [
            ("thinking", "The thing is small.\nDo it at once."),
            ("text", "Doing it.")
        ]
    );
    let texts = texts(&task);
    assert!(texts.iter().any(|text| text == "Doing it."), "{texts:?}");
    assert!(
        !texts.iter().any(|text| text.contains("small")),
        "{texts:?}"
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
        {"tool": "Grep", "input": {"pattern": "a", "path": "~/"}},
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
    // A search of `~`, which Claude Code may expand to the home directory.
    assert_eq!(posts[2]["is_error"], true);
    assert!(
        posts[2]["output"]
            .as_str()
            .unwrap()
            .contains("outside the workspace")
    );
    // Not listed: refused, though Claude Code has it.
    assert_eq!(posts[3]["is_error"], true);
    assert!(
        posts[3]["output"]
            .as_str()
            .unwrap()
            .contains("not available")
    );
    let rejected = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolRejected { .. }))
        .count();
    assert_eq!(rejected, 3);
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
            ids: Vec::new(),
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
            .starts_with("Asked q1. End your turn now")
    );
}

/// With `async_questions` and `wait: false`, the MCP `ask_user` returns at
/// once, naming the questions' ids, and the turn goes on. `close_questions` closes one, and
/// the host hears of both.
#[tokio::test]
async fn an_ask_in_claude_code_returns_at_once_with_ids() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__ask_user", "input": {"questions": [
            {"question": "Tabs or spaces?", "choices": ["Tabs", "Spaces"]},
            {"question": "Push now?", "default": "not yet"}
        ], "wait": false}},
        {"tool": "mcp__arko__close_questions", "input": {"ids": ["q2"], "why": "Decided"}},
        complete("Went on"),
    ]]));
    let mut config = config(&fake, &["ask_user", "close_questions"]);
    config.tool_config.async_questions = true;
    let task = Task::new(config, "Format the code.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::UserQuestionsAsked { asked, wait: false, .. }
            if asked.len() == 2 && asked[1].id == "q2"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::QuestionsClosed { ids, .. } if ids == &["q2"]
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::TaskStateChanged(TaskState::AwaitingUserResponse { .. })
    )));
    let posts = fake.logged("post");
    assert_eq!(
        posts[0]["output"].as_str().unwrap(),
        "Asked q1 and q2. Their answers reach you as a note. Until then, do only work that no answer could change."
    );
    assert_eq!(posts[1]["output"].as_str().unwrap(), "Closed q2.");
    assert_eq!(fake.logged("argv").len(), 1, "the turn went on");
    assert_eq!(task.questions.open.len(), 1);
    assert_eq!(task.questions.open[0].id, "q1");
}

/// Refuses the `ask_user` call that asks "Refused?" and every
/// `close_questions` call.
struct RefusesQuestions;

#[async_trait::async_trait]
impl crate::tool::ToolExecutionMiddleware for RefusesQuestions {
    async fn before(
        &self,
        _context: &crate::tool::ToolExecutionContext<'_>,
        call: &crate::tool::ToolCallContext<'_>,
    ) -> Result<crate::tool::ToolExecutionDecision, crate::tool::ToolError> {
        let refused = call.tool_name == "close_questions"
            || call
                .parameters
                .get("questions")
                .is_some_and(|questions| questions.to_string().contains("Refused?"));
        Ok(if refused {
            crate::tool::ToolExecutionDecision::Reject {
                message_to_agent: "Not here.".into(),
            }
        } else {
            crate::tool::ToolExecutionDecision::Continue
        })
    }

    async fn after(
        &self,
        _context: &crate::tool::ToolExecutionContext<'_>,
        _call: &crate::tool::ToolCallContext<'_>,
        _outcome: crate::tool::ToolExecutionOutcome<'_>,
    ) -> Result<(), crate::tool::ToolError> {
        Ok(())
    }
}

/// A middleware refuses an `ask_user` and a `close_questions` call after
/// Claude Code's gate admitted them: the refused ask leaves no question
/// open and the refused close closes none, and the host hears of neither,
/// so the task's log and the host agree.
#[tokio::test]
async fn refused_question_calls_in_claude_code_change_no_questions() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__ask_user", "input": {"questions": ["Tabs or spaces?"], "wait": false}},
        {"tool": "mcp__arko__ask_user", "input": {"questions": ["Refused?"], "wait": false}},
        {"tool": "mcp__arko__close_questions", "input": {"ids": ["q1"]}},
        complete("Went on"),
    ]]));
    let mut config = config(&fake, &["ask_user", "close_questions"]);
    config.tool_config.async_questions = true;
    let task = Task::new(config, "Format the code.".into(), fake.workspace(), None);
    let (events, commands, handle) = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .run_task(
            task,
            Arc::new(ToolRegistry::new()),
            None,
            false,
            None,
            None,
            Some(Arc::new(RefusesQuestions)),
        );
    let (task, events) = Run {
        events,
        commands,
        handle,
    }
    .finish()
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let asked = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::UserQuestionsAsked { asked, .. } => Some(asked.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(asked.len(), 1, "{asked:?}");
    assert_eq!(asked[0][0].id, "q1");
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::QuestionsClosed { .. }))
    );
    assert_eq!(
        task.questions
            .open
            .iter()
            .map(|question| question.id.as_str())
            .collect::<Vec<_>>(),
        ["q1"]
    );
    assert_eq!(task.questions.asked, 1);
}

/// A launch's message IDs are its own: after an answer relaunches Claude
/// Code (whose IDs may start over), a reply whose ID an earlier launch
/// streamed is still sent to the host.
#[tokio::test]
async fn a_relaunch_s_replies_reach_the_host_whatever_their_ids() {
    let fake = Fake::new(json!([
        [
            {"text": "Let me ask.", "stream": true},
            {"tool": "mcp__arko__ask_user", "input": {"questions": "Blue or green?"}},
        ],
        [{"text": "Noted your colour."}, complete("Noted")],
    ]));
    let task = Task::new(
        config(&fake, &["ask_user"]),
        "Pick a colour.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    run.events_until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "Blue.".into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, events) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(fake.logged("argv").len(), 2);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantResponseChunk(text) if text == "Noted your colour."
    )));
}

/// Context for the model once it has been asked for context before.
struct SecondUpdate {
    asked: std::sync::atomic::AtomicUsize,
}

impl crate::runtime::TurnContextSource for SecondUpdate {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        let asked = self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (asked == 1).then(|| "Message from agent 'lead': skip vendor/.".to_string())
    }
}

/// Context that arrives during a turn goes with the next call's result.
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
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(SecondUpdate {
            asked: Default::default(),
        }),
    );
    let (task, _) = start_with(task, runtime).finish().await;
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

/// The notes the user typed, which a source gives apart from the rest of
/// its update, once it has been asked for context before.
struct TypedSecondUpdate {
    content: &'static str,
    asked: std::sync::atomic::AtomicUsize,
}

const TYPED_NOTE: &str = "A new note on your task.\n- Note 3 from the operator (the person running this workflow), 2026-10-23 08:00 UTC:\n  Use tabs.";

impl crate::runtime::TurnContextSource for TypedSecondUpdate {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        None
    }

    fn context_update(&self, _task_id: uuid::Uuid) -> Option<crate::runtime::TurnContext> {
        let asked = self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (asked == 1).then(|| crate::runtime::TurnContext {
            content: self.content.into(),
            user_content: TYPED_NOTE.into(),
            ids: vec!["message:7".into(), "note:3".into()],
            ..Default::default()
        })
    }
}

/// Runs a task whose first call gets a context update of `content` and the
/// typed note.
async fn run_with_typed_note(content: &'static str) -> (Fake, Task, Vec<AgentEvent>) {
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
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(TypedSecondUpdate {
            content,
            asked: Default::default(),
        }),
    );
    let (task, events) = start_with(task, runtime).finish().await;
    (fake, task, events)
}

/// What the user typed reaches Claude Code during a turn as a message from
/// the user (not in the hook), while the rest of the update (a worker's
/// mail) stays in the hook. Both count as delivered and are recorded after
/// the result, in one update.
#[tokio::test]
async fn a_note_the_user_typed_goes_as_their_message_and_the_mail_in_the_hook() {
    let (fake, task, events) =
        run_with_typed_note("Message from agent 'lead': skip vendor/.").await;
    let sent = fake.logged("midturn");
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0]["text"], TYPED_NOTE);
    assert_eq!(sent[0]["origin"], json!({"kind": "human"}));
    // The message is no input of the session: no entry of the transcript
    // stands for it, and the CLI completed it.
    assert_eq!(fake.logged("user").len(), 1);
    let uuid = sent[0]["uuid"].as_str().unwrap();
    assert!(
        !task.claude_code_entries.values().any(|entry| entry == uuid),
        "{:?}",
        task.claude_code_entries
    );
    assert!(task.pending_context_injections.is_empty());
    let posts = fake.logged("post");
    let hooked = posts[0]["answer"]["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(hooked.contains("skip vendor/"), "{hooked}");
    assert!(!hooked.contains("Use tabs."), "{hooked}");
    // The hook's record first. The note's follows once the CLI completed
    // its message too, which the fake does as the turn ends.
    let updates = texts(&task)
        .into_iter()
        .filter(|text| text.starts_with("# Workflow context update"))
        .collect::<Vec<_>>();
    assert!(updates[0].ends_with("skip vendor/."), "{updates:?}");
    assert_eq!(
        updates[1],
        format!("# Workflow context update\n\n{TYPED_NOTE}")
    );
    let injected: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ContextInjected { ids, .. } => Some(ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(injected[0], ["message:7", "note:3"]);
}

/// A note alone leaves the hook empty, and is recorded under the update's
/// heading.
#[tokio::test]
async fn a_note_the_user_typed_alone_leaves_the_hook_empty() {
    let (fake, task, _) = run_with_typed_note("").await;
    assert_eq!(fake.logged("midturn").len(), 1);
    let posts = fake.logged("post");
    assert!(posts[0]["answer"]["hookSpecificOutput"]["additionalContext"].is_null());
    assert!(texts(&task).contains(&format!("# Workflow context update\n\n{TYPED_NOTE}")));
}

/// A note the CLI cancels (the turn stopped before the model's next request)
/// never reached the model, so it goes with the task's next input.
#[tokio::test]
async fn a_note_the_cli_cancelled_goes_with_the_next_input() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"sleep": 0.5},
        {"tool": "Grep", "input": {"pattern": "b"}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(TypedSecondUpdate {
            content: "",
            asked: Default::default(),
        }),
    );
    let run = start_with(task, runtime);
    // Suspended once the note went as the user's message.
    until(|| fake.logged("midturn").len() == 1).await;
    run.commands.send(UserCommand::SuspendTask).await.unwrap();
    let (task, _) = run.finish().await;
    assert_eq!(fake.logged("midturn").len(), 1);
    assert_eq!(task.pending_context_injections, [TYPED_NOTE]);
}

/// The script of a task whose first call's typed note the CLI reports
/// cancelled before the call's result, as when the turn stops while the
/// call's hooks run. The fake goes on with the turn all the same.
fn note_cancelled_before_its_result() -> Value {
    json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "cancel_notes": true},
        {"tool": "Grep", "input": {"pattern": "b"}},
        complete("Done"),
    ]])
}

/// Runs the task of `note_cancelled_before_its_result`, its first call
/// given the typed note with `content`.
async fn run_with_note_cancelled_before_its_result(content: &'static str) -> (Fake, Task) {
    let fake = Fake::new(note_cancelled_before_its_result());
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(TypedSecondUpdate {
            content,
            asked: Default::default(),
        }),
    );
    let (task, _) = start_with(task, runtime).finish().await;
    (fake, task)
}

/// The texts of `task`'s record that hold the typed note.
fn with_typed_note(task: &Task) -> Vec<String> {
    texts(task)
        .into_iter()
        .filter(|text| text.contains("Use tabs."))
        .collect()
}

/// A note the CLI reports cancelled before the result it went with never
/// reached the model. The record of that result leaves it out, and the
/// note waits where it was, so it goes again as the user's message with the
/// next call's hook, recorded once.
#[tokio::test]
async fn a_note_cancelled_before_its_result_is_left_out_of_the_record() {
    let (fake, task) = run_with_note_cancelled_before_its_result("").await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let sent = fake.logged("midturn");
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[1]["text"], TYPED_NOTE);
    let posts = fake.logged("post");
    assert!(
        posts
            .iter()
            .all(|post| !post.to_string().contains("Use tabs.")),
        "never in a hook"
    );
    let recorded = with_typed_note(&task);
    assert_eq!(recorded.len(), 1, "{recorded:?}");
    assert!(task.pending_context_injections.is_empty());
}

/// The rest of the update went in the call's hook, so the record of the
/// result keeps it, without the note.
#[tokio::test]
async fn a_note_cancelled_before_its_result_leaves_the_rest_of_its_update() {
    let (_, task) = run_with_note_cancelled_before_its_result("A message: skip vendor/.").await;
    let updates = texts(&task)
        .into_iter()
        .filter(|text| text.starts_with("# Workflow context update"))
        .collect::<Vec<_>>();
    assert!(updates[0].ends_with("skip vendor/."), "{updates:?}");
    assert!(!updates[0].contains("Use tabs."), "{updates:?}");
    assert_eq!(with_typed_note(&task).len(), 1, "{updates:?}");
}

/// The operator's note of `TYPED_NOTE`, marked as note 3, given at the
/// second boundary, as a workflow gives the user's answer.
struct MarkedTypedNote(std::sync::atomic::AtomicUsize);

impl crate::runtime::TurnContextSource for MarkedTypedNote {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        None
    }

    fn context_update(&self, _task_id: uuid::Uuid) -> Option<crate::runtime::TurnContext> {
        let asked = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (asked == 1).then(|| crate::runtime::TurnContext {
            user_content: TYPED_NOTE.into(),
            ids: vec!["note:3".into()],
            marks: [("notes".to_string(), 3)].into(),
            ..Default::default()
        })
    }
}

/// A note given with a call's result that the CLI took, but exited before
/// it reported the result: the task never keeps the note's mark without
/// the note, and once the session ends the note waits in the task again,
/// with its mark, for the next input. Once the result is reported with the
/// note, the task holds both.
#[tokio::test]
async fn a_note_the_cli_took_before_it_exited_goes_back_with_its_mark() {
    let fake = Fake::new(json!([
        [{"tool": "Grep", "input": {"pattern": "a"}, "crash": "post"}],
        [{"tool": "Grep", "input": {"pattern": "b"}}, complete("Done")],
    ]));
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let runtime =
        recording(&recorder).with_turn_context(Arc::new(MarkedTypedNote(Default::default())));
    let mut run = start_with(task, runtime);
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    let saved = saved_once(&recorder, |task| {
        !task.pending_context_injections.is_empty()
    })
    .await;
    run.handle.abort();
    assert_eq!(fake.logged("midturn").len(), 1, "the CLI took the note");
    let holds = |task: &Task| {
        task.pending_context_injections
            .iter()
            .any(|item| item.text.contains("Use tabs."))
            || texts(task).iter().any(|text| text.contains("Use tabs."))
    };
    let snapshots = recorder.snapshots();
    assert!(!snapshots.is_empty());
    for (task, _) in &snapshots {
        assert!(
            !task.context_marks.contains_key("notes") || holds(task),
            "{:?}",
            task.context_marks
        );
    }
    assert_eq!(saved.pending_context_injections, [TYPED_NOTE]);
    assert_eq!(saved.context_marks.get("notes"), Some(&3));
    // Reported with its result, the note and its mark are kept.
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_turn_context(Arc::new(MarkedTypedNote(Default::default())));
    let (task, _) = start_with(task, runtime).finish().await;
    assert!(holds(&task));
    assert_eq!(task.context_marks.get("notes"), Some(&3));
}

/// A marked note the CLI cancels goes back to the task with its mark, so a
/// task that had an earlier mark doesn't keep that one beside the note and
/// get the note again when a host recovers it.
#[tokio::test]
async fn a_cancelled_note_goes_back_with_its_mark() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"sleep": 0.5},
        {"tool": "Grep", "input": {"pattern": "b"}},
        complete("Done"),
    ]]));
    let mut task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    task.context_marks.insert("notes".into(), 2);
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_turn_context(Arc::new(MarkedTypedNote(Default::default())));
    let run = start_with(task, runtime);
    // Suspended once the note went as the user's message.
    until(|| fake.logged("midturn").len() == 1).await;
    run.commands.send(UserCommand::SuspendTask).await.unwrap();
    let (task, _) = run.finish().await;
    assert_eq!(fake.logged("midturn").len(), 1);
    assert_eq!(task.pending_context_injections, [TYPED_NOTE]);
    assert_eq!(task.context_marks.get("notes"), Some(&3));
}

/// The same when the CLI reports the note cancelled before the result it
/// went with: the task holds the note, with its mark, once it goes with the
/// next call's result.
#[tokio::test]
async fn a_note_cancelled_before_its_result_keeps_its_mark() {
    let fake = Fake::new(note_cancelled_before_its_result());
    let mut task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    task.context_marks.insert("notes".into(), 2);
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_turn_context(Arc::new(MarkedTypedNote(Default::default())));
    let (task, _) = start_with(task, runtime).finish().await;
    // Sent again with the next call's hook.
    assert_eq!(fake.logged("midturn").len(), 2);
    assert_eq!(task.context_marks.get("notes"), Some(&3));
    assert_eq!(with_typed_note(&task).len(), 1);
    assert!(task.pending_context_injections.is_empty());
}

/// A completed task continued with a new configuration resumes its session
/// with the new tools and system prompt, telling Claude Code not to reuse
/// the prompt it recorded.
#[tokio::test]
async fn a_reconfigured_task_resumes_its_session_with_the_new_prompt_and_tools() {
    let fake = Fake::new(json!([[complete("First")], [complete("Second")]]));
    let task = Task::new(config(&fake, &[]), "Start.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    let session = task.claude_code_session.clone().unwrap();
    let mut searching = config(&fake, &["Grep"]);
    searching.instructions = "Search with Grep from now on.".into();
    task.reconfigure(searching).unwrap();
    task.prepare_for_resumption_from_completed("Search.".into())
        .unwrap();
    let (task, _) = start(task).finish().await;
    assert!(matches!(&task.current_state, TaskState::Completed { summary } if summary == "Second"));

    let argv = |index: usize| -> Vec<String> {
        serde_json::from_value(fake.logged("argv")[index]["argv"].clone()).unwrap()
    };
    assert!(!argv(0).iter().any(|arg| arg == "--system-prompt-snapshot"));
    let resumed = argv(1);
    let joined = resumed.join(" ");
    assert!(joined.contains("--system-prompt-snapshot off"), "{joined}");
    assert!(joined.contains(&format!("--resume={session}")), "{joined}");
    assert!(joined.contains("--tools Grep"), "{joined}");
    assert!(joined.contains("Search with Grep from now on."), "{joined}");
    assert!(task.claude_code_prompt_changed);
}

/// A cc chat (`AgentToolConfig::chat`): no tools, its replies in text end
/// its task.
fn chat_config(fake: &Fake) -> crate::agent_config::AgentConfig {
    let mut config = config(fake, &[]);
    config.tool_config.available_tool_names.clear();
    config.tool_config.text_reply_ends_task = true;
    config.tool_config.chat = true;
    config
}

/// The conversation Claude Code had when each message reached it (the
/// fake's transcript before it).
fn histories(fake: &Fake) -> Vec<Vec<String>> {
    fake.logged("user")
        .iter()
        .map(|entry| serde_json::from_value(entry["history"].clone()).unwrap())
        .collect()
}

/// Cut back to the follow-up, the session resumes at the reply before it
/// and gets the new message in its place. A branch copies the session up
/// to there into its own, and the original goes on with all of it.
#[tokio::test]
async fn a_cut_resumes_the_session_before_the_message_and_a_branch_forks_it() {
    let fake = Fake::new(json!([
        [{"text": "Where to?"}],
        [{"text": "Lisbon is nice."}],
        [{"text": "Porto is nice."}],
        [{"text": "Faro too."}],
    ]));
    let task = Task::new(
        chat_config(&fake),
        "Plan a trip.".into(),
        fake.workspace(),
        None,
    );
    let (mut task, _) = start(task).finish().await;
    task.prepare_for_resumption_from_completed("Lisbon.".into())
        .unwrap();
    let (original, _) = start(task).finish().await;
    let session = original.claude_code_session.clone().unwrap();
    let reply = original.claude_code_entries[&1].clone();
    assert_eq!(texts(&original)[3], "Lisbon is nice.");

    // A branch at the follow-up, with another message in its place.
    let mut branch = original.branch(2).unwrap();
    branch
        .prepare_for_resumption_from_completed("Porto.".into())
        .unwrap();
    let (branch, _) = start(branch).finish().await;
    assert!(
        matches!(&branch.current_state, TaskState::Completed { summary } if summary == "Porto is nice.")
    );
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[2]["argv"].clone()).unwrap();
    let joined = argv.join(" ");
    let forked = branch.claude_code_session.clone().unwrap();
    assert!(
        joined.contains(&format!(
            "--resume={session} --fork-session --session-id={forked} --resume-session-at={reply}"
        )),
        "{joined}"
    );
    let history = &histories(&fake)[2];
    assert_eq!(history.len(), 2, "{history:?}");
    assert_eq!(history[1], "Where to?");
    assert!(branch.claude_code_fork_of.is_none());

    // The original, cut back to its follow-up: the same session, resumed
    // at the reply.
    let mut cut = original.clone();
    cut.cut_back(2).unwrap();
    cut.prepare_for_resumption_from_completed("Faro.".into())
        .unwrap();
    let (cut, _) = start(cut).finish().await;
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[3]["argv"].clone()).unwrap();
    let joined = argv.join(" ");
    assert!(
        joined.contains(&format!("--resume={session} --resume-session-at={reply}")),
        "{joined}"
    );
    assert_eq!(histories(&fake)[3].len(), 2);
    assert_eq!(
        texts(&cut)[1..],
        ["Where to?", texts(&cut)[2].as_str(), "Faro too."]
    );
    assert!(texts(&cut)[2].starts_with("Faro."));
}

/// The reply and the thinking stream as Claude Code streams them, and the
/// complete messages don't repeat them. Claude Code's retries are reported
/// as the runtime reports its own.
#[tokio::test]
async fn a_reply_streams_as_it_comes_and_retries_are_reported() {
    let fake = Fake::new(json!([[
        {"retry": {"attempt": 1, "max_retries": 10, "retry_delay_ms": 1500,
                   "error_status": 529, "error": "overloaded"}},
        {"retry": {"attempt": 2, "max_retries": 10, "retry_delay_ms": 3000,
                   "error_status": 429, "error": "rate_limit"}},
        {"thinking": "Let me think about it.", "text": "The answer is four, of course.",
         "stream": true},
    ]]));
    let task = Task::new(
        chat_config(&fake),
        "What is 2 + 2?".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "The answer is four, of course.")
    );
    let text: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::AssistantResponseChunk(chunk) => Some(chunk.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, ["The answer ", "is four, ", "of course."]);
    let thinking: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ReasoningChunk(chunk) => Some(chunk.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(thinking, ["Let me ", "think about ", "it."]);
    let retries: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RetryScheduled {
                attempt,
                delay_ms,
                rate_limit,
                ..
            } => Some((*attempt, *delay_ms, *rate_limit)),
            _ => None,
        })
        .collect();
    assert_eq!(retries, [(1, 1500, false), (2, 3000, true)]);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { error_message, requires_approval_to_retry: false, .. }
            if error_message.contains("overloaded, status 529") && error_message.contains("attempt 1 of 10")
    )));
}

/// A request that fails mid-response is reported as a message of Claude
/// Code's own. Its text reaches the host as a warning, apart from the reply
/// before it, and the conversation doesn't keep it as reply text.
#[tokio::test]
async fn a_failed_requests_message_is_a_warning_and_not_part_of_the_reply() {
    let note = "API Error: Connection lost mid-response. The response above may be incomplete.";
    let fake = Fake::new(json!([[
        {"text": "The answer is"},
        {"api_error": "unknown", "text": note},
    ]]));
    let task = Task::new(
        chat_config(&fake),
        "What is 2 + 2?".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    let mut events = run
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
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    let (task, rest) = run.finish().await;
    events.extend(rest);
    let warnings: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ServiceWarning { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(warnings, [note]);
    let reply: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::AssistantResponseChunk(chunk) => Some(chunk.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reply, ["The answer is"]);
    assert_eq!(texts(&task)[1..], ["The answer is"]);
}

/// Claude Code sends a response as one assistant message per block, each
/// with the response's id: its thinking, its text, then its call. Streamed,
/// each block's text comes once, as it streams, and the complete messages
/// don't repeat it. The conversation records each block's message: the
/// text, then the call.
#[tokio::test]
async fn a_response_streamed_block_by_block_comes_once() {
    let fake = Fake::new(json!([[
        {"thinking": "Let me look.", "stream": true, "closes": false},
        {"text": "Checking the files.", "stream": true, "continues": true, "closes": false},
        {"tool": "Grep", "input": {"pattern": "todo"}, "continues": true},
        complete("Found none"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let chunks = |reasoning: bool| -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ReasoningChunk(chunk) if reasoning => Some(chunk.clone()),
                AgentEvent::AssistantResponseChunk(chunk) if !reasoning => Some(chunk.clone()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(chunks(true), ["Let me ", "look."]);
    assert_eq!(chunks(false), ["Checking the ", "files."]);
    let at = task
        .conversation
        .messages
        .iter()
        .position(|message| message.role == arko_ai::Role::Assistant)
        .unwrap();
    let (reply, call) = (
        &task.conversation.messages[at],
        &task.conversation.messages[at + 1],
    );
    let text: String = reply
        .content
        .iter()
        .filter_map(arko_ai::Content::as_text)
        .collect();
    assert_eq!(text, "Checking the files.");
    assert_eq!(call.role, arko_ai::Role::Assistant);
    assert_eq!(call.tool_calls[0].name, "Grep");
}

/// A subagent Claude Code runs (its `Agent` tool) makes requests of its
/// own: each is reported with the call that started it and no sizes (its
/// conversation isn't the task's), its usage is marked as the subagent's
/// and compared with that subagent's previous call only, and its messages
/// stay out of the task's conversation.
#[tokio::test]
async fn a_subagent_s_requests_are_reported_as_its_own() {
    let fake = Fake::new(json!([[
        {"subagent": {"parent": "toolu_agent", "text": "Looked around.", "input_tokens": 300}},
        {"subagent": {"parent": "toolu_agent", "text": "Found it.", "input_tokens": 450}},
        {"text": "All done.", "input_tokens": 1000},
    ]]));
    let task = Task::new(chat_config(&fake), "Look.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let started: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted {
                subagent, sizes, ..
            } => Some((subagent.clone(), sizes.is_some())),
            _ => None,
        })
        .collect();
    let agent = Some("toolu_agent".to_string());
    assert_eq!(
        started,
        [(agent.clone(), false), (agent.clone(), false), (None, true)]
    );
    let calls: Vec<_> = task
        .model_call_usage_history
        .iter()
        .map(|call| {
            (
                call.subagent.clone(),
                call.input_tokens,
                call.input_delta_from_previous_call,
            )
        })
        .collect();
    assert_eq!(
        calls,
        [
            (agent.clone(), 300, None),
            (agent, 450, Some(150)),
            (None, 1000, None),
        ]
    );
    let texts = texts(&task);
    assert!(
        !texts.iter().any(|text| text.contains("Looked around")),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|text| text.contains("All done.")),
        "{texts:?}"
    );
}

/// A response that fails while it streams is retried by Claude Code: the
/// host is told to drop what it streamed before the retry is reported, and
/// the retry streams anew. Only the full reply is kept.
#[tokio::test]
async fn a_response_that_fails_while_it_streams_is_discarded() {
    let fake = Fake::new(json!([[
        {"stream_fail": "The ans"},
        {"retry": {"attempt": 1, "max_retries": 10, "retry_delay_ms": 500,
                   "error_status": 529, "error": "overloaded"}},
        {"text": "The answer is four.", "stream": true},
    ]]));
    let task = Task::new(
        chat_config(&fake),
        "What is 2 + 2?".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "The answer is four.")
    );
    let order: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted { .. } => Some("request".to_string()),
            AgentEvent::AssistantResponseChunk(chunk) => Some(format!("chunk {chunk}")),
            AgentEvent::StreamedOutputDiscarded {
                text_bytes,
                reasoning_bytes,
            } => Some(format!("discard {text_bytes} {reasoning_bytes}")),
            AgentEvent::RetryScheduled { .. } => Some("retry".to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        [
            "request",
            "chunk The ans",
            "discard 7 0",
            "retry",
            "request",
            "chunk The answer ",
            "chunk is four.",
        ]
    );
}

/// A retry series that ends in a failure ends with its turn: the next
/// series counts its time from its own first retry.
#[tokio::test]
async fn a_failed_retry_series_doesn_t_count_into_the_next() {
    let retry = |attempt: u32| {
        json!({"retry": {"attempt": attempt, "max_retries": 10, "retry_delay_ms": 10,
                         "error_status": 529, "error": "overloaded"}})
    };
    let fake = Fake::new(json!([
        [retry(1), {"sleep": 0.4}, {"fail": "API Error: 529 overloaded"}],
        [retry(1), {"text": "Done."}],
    ]));
    let task = Task::new(chat_config(&fake), "Go.".into(), fake.workspace(), None);
    let mut run = start(task);
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
    run.commands
        .send(UserCommand::ApproveErrorRecovery { message: None })
        .await
        .unwrap();
    let (_, events) = run.finish().await;
    let elapsed: Vec<u64> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RetryScheduled { elapsed_ms, .. } => Some(*elapsed_ms),
            _ => None,
        })
        .collect();
    assert_eq!(elapsed.len(), 1, "{elapsed:?}");
    assert!(elapsed[0] < 300, "{elapsed:?}");
}

/// When Claude Code gives up on a response's stream and asks for it again
/// without streaming, what the stream delivered goes as it stops. The
/// response that takes its place comes whole, with its call and without a
/// `message_start`, and its text stays when the next response starts.
#[tokio::test]
async fn text_of_a_response_asked_again_without_streaming_stays() {
    let fake = Fake::new(json!([[
        {"stream_abandoned": "Let me che"},
        {"text": "Let me check the notes.", "unstreamed": true},
        {"tool": "Read", "input": {"file_path": "notes.txt"}, "continues": true, "unstreamed": true},
        {"text": "The notes are empty.", "stream": true},
    ]]));
    let mut config = config(&fake, &["Read"]);
    config.tool_config.text_reply_ends_task = true;
    let task = Task::new(config, "Read the notes.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    let order: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted { .. } => Some("request".to_string()),
            AgentEvent::AssistantResponseChunk(chunk) => Some(format!("chunk {chunk}")),
            AgentEvent::StreamedOutputDiscarded { text_bytes, .. } => {
                Some(format!("discard {text_bytes}"))
            }
            AgentEvent::ToolResult { tool_name, .. } => Some(format!("result {tool_name}")),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        [
            "request",
            "chunk Let me che",
            "discard 10",
            "chunk Let me check the notes.",
            "result Read",
            "request",
            "chunk The notes ",
            "chunk are empty.",
        ]
    );
    let texts = texts(&task);
    assert!(
        texts.iter().any(|text| text == "Let me check the notes."),
        "{texts:?}"
    );
    assert!(!texts.iter().any(|text| text == "Let me che"), "{texts:?}");
}

/// A response that fails while it streams is discarded too when it ends
/// the turn, or when Claude Code starts another response without reporting
/// a retry: the host drops what it streamed before the turn's error, or
/// before the next response's text.
#[tokio::test]
async fn a_response_that_fails_while_it_streams_is_discarded_however_it_ends() {
    let order = |events: &[AgentEvent]| -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ModelRequestStarted { .. } => Some("request".to_string()),
                AgentEvent::AssistantResponseChunk(chunk) => Some(format!("chunk {chunk}")),
                AgentEvent::StreamedOutputDiscarded { text_bytes, .. } => {
                    Some(format!("discard {text_bytes}"))
                }
                AgentEvent::ErrorEncountered { .. } => Some("error".to_string()),
                _ => None,
            })
            .collect()
    };
    let fake = Fake::new(json!([[
        {"stream_fail": "The ans"},
        {"fail": "API Error: 529 overloaded"},
    ]]));
    let task = Task::new(
        chat_config(&fake),
        "What is 2 + 2?".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    let events = run
        .events_until(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            )
        })
        .await;
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    run.finish().await;
    let seen = order(&events);
    let discard = seen.iter().position(|kind| kind == "discard 7").unwrap();
    assert_eq!(seen[..discard], ["request", "chunk The ans"], "{seen:?}");

    let fake = Fake::new(json!([[
        {"stream_fail": "The ans"},
        {"text": "The answer is four.", "stream": true},
    ]]));
    let task = Task::new(
        chat_config(&fake),
        "What is 2 + 2?".into(),
        fake.workspace(),
        None,
    );
    let (_, events) = start(task).finish().await;
    assert_eq!(
        order(&events),
        [
            "request",
            "chunk The ans",
            "discard 7",
            "request",
            "chunk The answer ",
            "chunk is four.",
        ]
    );
}

/// A cache refresh pings a fork of the session that isn't saved: the ping
/// reads the conversation (and its cache), its usage comes back, and the
/// session's own conversation never sees it.
#[tokio::test]
async fn a_cache_refresh_pings_a_fork_that_isn_t_saved() {
    let fake = Fake::new(json!([
        [{"text": "Hi."}],
        [{"text": "ok", "input_tokens": 900, "cached": 850}],
        [{"text": "Again."}],
    ]));
    let task = Task::new(chat_config(&fake), "Hello.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    let session = task.claude_code_session.clone().unwrap();
    let refresh = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .refresh_prompt_cache(task.clone(), Arc::new(ToolRegistry::new()), None)
        .await
        .unwrap();
    assert_eq!(refresh.usage.input_tokens, 900);
    assert_eq!(refresh.usage.cached_tokens, Some(850));
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[1]["argv"].clone()).unwrap();
    let joined = argv.join(" ");
    assert!(
        joined.contains(&format!("--resume={session} --fork-session")),
        "{joined}"
    );
    assert!(joined.contains("--max-turns 1") && joined.contains("--no-session-persistence"));
    let pinged = &histories(&fake)[1];
    assert_eq!(pinged.last().map(String::as_str), Some("Hi."));

    task.prepare_for_resumption_from_completed("Again?".into())
        .unwrap();
    let (task, _) = start(task).finish().await;
    let history = &histories(&fake)[2];
    assert!(
        !history.iter().any(|entry| entry.contains("single word ok")),
        "{history:?}"
    );
    assert_eq!(texts(&task).last().map(String::as_str), Some("Again."));
}

/// A chat over the handover size hands over at its next message: the
/// report request goes alone first (a chat's turns end with a reply, so it
/// can't go during one), the summary replaces the conversation, and the
/// fresh session answers the message that was waiting.
#[tokio::test]
async fn a_chat_over_the_handover_size_hands_over_at_its_next_message() {
    let fake = Fake::new(json!([
        [{"text": "Answer one.", "input_tokens": 500}],
        [{"text": "Answer two.", "input_tokens": 1500}],
        [{"text": "SUMMARY: questions one and two, answered.", "input_tokens": 1600}],
        [{"text": "Answer three.", "input_tokens": 300}],
    ]));
    let mut config = chat_config(&fake);
    config.handover = Some(HandoverConfig::at_input_tokens(1000));
    let task = Task::new(config, "Question one.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    task.prepare_for_resumption_from_completed("Question two.".into())
        .unwrap();
    let (mut task, _) = start(task).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Answer two.")
    );
    assert!(task.handovers.is_empty());
    task.prepare_for_resumption_from_completed("Question three.".into())
        .unwrap();
    let (task, events) = start(task).finish().await;

    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Answer three."),
        "{:?}",
        task.current_state
    );
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(
        task.handovers[0].report,
        "SUMMARY: questions one and two, answered."
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { input_tokens: 1500 }))
    );
    let sent = user_texts(&fake);
    assert!(
        sent[2].starts_with(crate::handover::DEFAULT_CHAT_REPORT_REQUEST),
        "{}",
        sent[2]
    );
    assert!(!sent[2].contains("Question three."));
    assert!(sent[3].starts_with("Question three."), "{}", sent[3]);
    assert!(sent[3].contains("SUMMARY: questions one and two, answered."));
    // The fresh session is a new conversation.
    assert_eq!(histories(&fake)[3].len(), 0);
}

/// A conversation cut back while over the handover size still hands over:
/// the cut keeps the conversation's first request (under the size) as it
/// was.
#[tokio::test]
async fn a_cut_back_over_the_handover_size_still_hands_over() {
    let fake = Fake::new(json!([
        [{"text": "One.", "input_tokens": 500}],
        [{"text": "Two.", "input_tokens": 1100}],
        [
            {"tool": "Grep", "input": {"pattern": "x"}, "input_tokens": 1200},
            {"text": "REPORT: one, and two again."},
        ],
        [{"text": "Two, again.", "input_tokens": 300}],
    ]));
    let mut config = replying(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(1000));
    let task = Task::new(config, "First.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    task.prepare_for_resumption_from_completed("Second.".into())
        .unwrap();
    let (mut task, _) = start(task).finish().await;
    task.cut_back(2).unwrap();
    task.prepare_for_resumption_from_completed("Second, once more.".into())
        .unwrap();
    let (task, events) = start(task).finish().await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { input_tokens: 1200 }))
    );
    assert_eq!(task.handovers.len(), 1, "{:?}", task.current_state);
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Two, again.")
    );
}

/// Runs `task` until it completes, approving the recovery of a failed
/// session if it pauses for one.
async fn finish_through_a_failure(mut run: Run) -> (Task, Vec<AgentEvent>) {
    let mut events = run
        .events_until(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(
                    TaskState::Completed { .. } | TaskState::AwaitingErrorRecoveryApproval { .. }
                )
            )
        })
        .await;
    if matches!(
        events.last(),
        Some(AgentEvent::TaskStateChanged(
            TaskState::AwaitingErrorRecoveryApproval { .. }
        ))
    ) {
        run.commands
            .send(UserCommand::ApproveErrorRecovery { message: None })
            .await
            .unwrap();
    }
    let (task, more) = run.finish().await;
    events.extend(more);
    (task, events)
}

/// The launch after a cut back, saved before Claude Code has the edited
/// message, keeps the cut's point: Claude Code saves the cut only with that
/// message, so a launch that failed before it (the CLI exited, or the host
/// crashed) resumes at the same entry again and sends the edited message,
/// never the conversation from before the cut.
#[tokio::test]
async fn a_cut_holds_its_resume_point_until_the_session_has_the_message() {
    let fake = Fake::new(json!([
        [{"text": "One."}],
        [{"text": "Two."}],
        [{"exit_unrecorded": 3}],
        [{"text": "Two, edited."}],
    ]));
    let task = Task::new(chat_config(&fake), "First.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    task.prepare_for_resumption_from_completed("Second.".into())
        .unwrap();
    let (mut task, _) = start(task).finish().await;
    let reply = task.claude_code_entries[&1].clone();
    task.cut_back(2).unwrap();
    task.prepare_for_resumption_from_completed("Second, edited.".into())
        .unwrap();
    let recorder = Arc::new(Recorder::default());
    let (task, _) = finish_through_a_failure(start_with(task, recording(&recorder))).await;

    // The task saved before the failed launch still says where to resume.
    assert!(recorder.snapshots().iter().any(|(saved, durable)| {
        *durable
            && saved.claude_code_resume_at.as_deref() == Some(reply.as_str())
            && saved.claude_code_synced == 2
            && saved.conversation.messages.len() == 3
    }));
    // The launch that worked resumed at the reply and got the edit.
    let users = fake.logged("user");
    let last = users.last().unwrap();
    assert!(
        last["text"]
            .as_str()
            .unwrap()
            .starts_with("Second, edited."),
        "{last}"
    );
    let history: Vec<String> = serde_json::from_value(last["history"].clone()).unwrap();
    assert_eq!(history.len(), 2, "{history:?}");
    assert!(history[0].starts_with("First."));
    assert_eq!(history[1], "One.");
    let argv = fake.logged("argv");
    let argv: Vec<String> = serde_json::from_value(argv.last().unwrap()["argv"].clone()).unwrap();
    assert!(
        argv.contains(&format!("--resume-session-at={reply}")),
        "{argv:?}"
    );
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Two, edited.")
    );
    assert!(task.claude_code_resume_at.is_none());
}

/// A branch whose first launch failed before Claude Code saved its message
/// forks again, under a new ID (a failed attempt's ID may be taken), from
/// the same point of the original session.
#[tokio::test]
async fn a_branch_forks_again_after_a_failed_first_launch() {
    let fake = Fake::new(json!([
        [{"text": "One."}],
        [{"text": "Two."}],
        [{"exit_unrecorded": 3}],
        [{"text": "Plan B, then."}],
    ]));
    let task = Task::new(chat_config(&fake), "First.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    task.prepare_for_resumption_from_completed("Second.".into())
        .unwrap();
    let (original, _) = start(task).finish().await;
    let mut branch = original.branch(2).unwrap();
    branch
        .prepare_for_resumption_from_completed("Plan B.".into())
        .unwrap();
    let (branch, _) = finish_through_a_failure(start(branch)).await;
    assert!(
        matches!(&branch.current_state, TaskState::Completed { summary } if summary == "Plan B, then.")
    );
    let forks: Vec<String> = fake
        .logged("argv")
        .iter()
        .filter_map(|entry| {
            let argv: Vec<String> = serde_json::from_value(entry["argv"].clone()).unwrap();
            argv.iter()
                .any(|arg| arg == "--fork-session")
                .then(|| {
                    argv.iter()
                        .find_map(|arg| arg.strip_prefix("--session-id=").map(str::to_string))
                })
                .flatten()
        })
        .collect();
    assert_eq!(forks.len(), 2, "{forks:?}");
    assert_ne!(forks[0], forks[1]);
    assert_eq!(
        branch.claude_code_session.as_deref(),
        Some(forks[1].as_str())
    );
    assert!(branch.claude_code_fork_of.is_none());
    let history = histories(&fake).last().unwrap().clone();
    assert_eq!(history.len(), 2, "{history:?}");
    assert_eq!(history[1], "One.");
}

/// What a tool image sink is given for a call: its runtime ID, Claude
/// Code's ID, its name and its images' captions.
type Reported = (uuid::Uuid, Option<String>, String, Vec<String>);

/// Records what a tool image sink is given. It implements only
/// `tool_images_with_call_id`, which the runtime calls.
#[derive(Default)]
struct Images(std::sync::Mutex<Vec<Reported>>);

impl crate::tool::ToolImageSink for Images {
    fn tool_images(
        &self,
        _task_id: uuid::Uuid,
        _tool_call_id: uuid::Uuid,
        _tool_name: &str,
        _images: &[crate::tool::ToolImage],
    ) {
        unreachable!("the runtime gives the provider's call ID too");
    }

    fn tool_images_with_call_id(
        &self,
        _task_id: uuid::Uuid,
        tool_call_id: uuid::Uuid,
        provider_call_id: Option<&str>,
        tool_name: &str,
        images: &[crate::tool::ToolImage],
    ) {
        self.0.lock().unwrap().push((
            tool_call_id,
            provider_call_id.map(str::to_string),
            tool_name.to_string(),
            images
                .iter()
                .map(|image| image.description.clone())
                .collect(),
        ));
    }
}

/// The host's sink gets the images of an arko tool's call once (not again
/// when Claude Code reports the result) and those of Claude Code's own
/// `Read`, each by the call's ID in the events and Claude Code's own ID for
/// it, and captioned alike: the file's path and size.
#[tokio::test]
async fn the_host_s_sink_gets_arko_and_built_in_tool_images() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__read_file", "id": "toolu_arko_image", "input": {"path": "a.png"}},
        {"tool": "Read", "id": "toolu_built_in_image", "input": {"file_path": "b.png"}, "output": [
            {"type": "text", "text": "b.png"},
            {"type": "image", "source": {"type": "base64", "data": "cGl4ZWxz", "media_type": "image/png"}},
        ]},
        complete("Seen both"),
    ]]));
    std::fs::write(
        fake.workspace().join("a.png"),
        b"\x89PNG\r\n\x1a\nimage-fixture",
    )
    .unwrap();
    let task = Task::new(
        config(&fake, &["read_file", "Read"]),
        "Look at both.".into(),
        fake.workspace(),
        None,
    );
    let images = Arc::new(Images::default());
    let runtime =
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_tool_image_sink(images.clone());
    let (task, events) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let recorded = images.0.lock().unwrap().clone();
    let names: Vec<_> = recorded
        .iter()
        .map(|(_, call_id, name, captions)| (call_id.as_deref(), name.as_str(), captions.clone()))
        .collect();
    assert_eq!(
        names,
        [
            (
                Some("toolu_arko_image"),
                "read_file",
                vec!["a.png, 1 KB".to_string()]
            ),
            (
                Some("toolu_built_in_image"),
                "Read",
                vec!["b.png, 1 KB".to_string()]
            )
        ]
    );
    for (id, _, name, _) in &recorded {
        assert!(
            events.iter().any(|event| matches!(
                event,
                AgentEvent::ToolCallFormatted { tool_call_id, tool_name, .. }
                    if tool_call_id == id && tool_name == name
            )),
            "{name}"
        );
    }
}

fn pixels() -> arko_ai::Image {
    arko_ai::Image::new_base64("cGl4ZWxz", "image/png")
}

fn image_block() -> Value {
    json!({"type": "image", "source": {"type": "base64", "data": "cGl4ZWxz", "media_type": "image/png"}})
}

/// The task's images go with its prompt, in the first input's content
/// blocks.
#[tokio::test]
async fn the_task_s_images_go_with_its_first_input() {
    let fake = Fake::new(json!([[complete("Seen")]]));
    let mut task = Task::new(
        config(&fake, &[]),
        "Describe this.".into(),
        fake.workspace(),
        None,
    );
    task.initial_task_images = vec![pixels()];
    let (task, _) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let sent: Value = serde_json::from_str(&user_texts(&fake)[0]).unwrap();
    let blocks = sent
        .as_array()
        .expect("an input with an image is sent as blocks");
    assert_eq!(blocks.len(), 2, "{sent}");
    assert!(
        blocks[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Describe this."),
        "{sent}"
    );
    assert_eq!(blocks[1], image_block());
    assert!(task.initial_task_images.is_empty());
    assert_eq!(task.initial_images(), [pixels()]);
}

/// Context with an image, once the model has been asked for context
/// before.
struct SecondUpdateWithImage {
    asked: std::sync::atomic::AtomicUsize,
    /// The note is the part of the update the user typed.
    typed: bool,
}

impl crate::runtime::TurnContextSource for SecondUpdateWithImage {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        None
    }

    fn context_update(&self, _task_id: uuid::Uuid) -> Option<crate::runtime::TurnContext> {
        let asked = self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let note = String::from("A new note: look at this.");
        let (content, user_content) = if self.typed {
            (String::new(), note)
        } else {
            (note, String::new())
        };
        (asked == 1).then(|| crate::runtime::TurnContext {
            content,
            user_content,
            ids: vec!["note:3".into()],
            images: vec![crate::runtime::ContextImage::new(
                "The image of note 3:",
                pixels(),
            )],
            ..Default::default()
        })
    }
}

/// Context with images that arrives during a turn can't go whole with a
/// hook's text: its text goes at once, saying the image follows, and the
/// image waits for the turn's end and goes with the next input, after its
/// label.
#[tokio::test]
async fn context_with_images_sends_its_text_at_once_and_the_images_next() {
    let fake = Fake::new(json!([
        [{"tool": "Grep", "input": {"pattern": "todo"}}, {"text": "Looked."}],
        [complete("Seen")],
    ]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(SecondUpdateWithImage {
            asked: Default::default(),
            typed: false,
        }),
    );
    let (task, events) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let posts = fake.logged("post");
    let hooked = posts[0]["answer"]["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap_or_else(|| panic!("{}", posts[0]));
    assert!(
        hooked.contains("# Workflow context update\n\nA new note: look at this.\n\n(The image of this update can't come during a turn. It comes with your next input.)"),
        "{hooked}"
    );
    let sent: Value = serde_json::from_str(&user_texts(&fake)[1]).unwrap();
    let blocks = sent
        .as_array()
        .expect("an input with an image is sent as blocks");
    // The runtime's note on the open turn, then the update.
    assert!(
        blocks[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Your turn ended")
    );
    let blocks = &blocks[1..];
    let text = blocks[0]["text"].as_str().unwrap();
    assert!(
        text.contains(
            "# Workflow context update\n\nThe images an earlier update said would follow."
        ),
        "{text}"
    );
    assert!(!text.contains("look at this"), "{text}");
    assert_eq!(
        &blocks[1..],
        [
            json!({"type": "text", "text": "The image of note 3:"}),
            image_block()
        ]
    );
    assert!(task.pending_context_images.is_empty());
    // The note counts as given with its text.
    let injected: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ContextInjected { ids, .. } => Some(ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(injected[0], ["note:3"]);
}

/// The same for a note the user typed: its text goes at once as their
/// message, with the line saying the image follows, and the image goes with
/// the next input.
#[tokio::test]
async fn a_note_the_user_typed_with_an_image_sends_its_text_at_once_and_the_image_next() {
    let fake = Fake::new(json!([
        [{"tool": "Grep", "input": {"pattern": "todo"}}, {"text": "Looked."}],
        [complete("Seen")],
    ]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Find todos.".into(),
        fake.workspace(),
        None,
    );
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(
        Arc::new(SecondUpdateWithImage {
            asked: Default::default(),
            typed: true,
        }),
    );
    let (task, events) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let sent = fake.logged("midturn");
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert_eq!(sent[0]["origin"], json!({"kind": "human"}));
    assert_eq!(
        sent[0]["text"],
        "A new note: look at this.\n\n(The image of this update can't come during a turn. It comes with your next input.)"
    );
    let posts = fake.logged("post");
    assert!(posts[0]["answer"]["hookSpecificOutput"]["additionalContext"].is_null());
    let recorded = texts(&task)
        .into_iter()
        .filter(|text| text.contains("look at this"))
        .collect::<Vec<_>>();
    assert_eq!(recorded.len(), 1, "{recorded:?}");
    assert!(recorded[0].starts_with("# Workflow context update\n\nA new note"));
    let next: Value = serde_json::from_str(&user_texts(&fake)[1]).unwrap();
    let blocks = next
        .as_array()
        .expect("an input with an image is sent as blocks");
    assert!(
        blocks[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Your turn ended")
    );
    let blocks = &blocks[1..];
    assert!(
        blocks[0]["text"]
            .as_str()
            .unwrap()
            .contains("The images an earlier update said would follow."),
        "{next}"
    );
    assert_eq!(
        &blocks[1..],
        [
            json!({"type": "text", "text": "The image of note 3:"}),
            image_block()
        ]
    );
    assert!(task.pending_context_images.is_empty());
    let injected: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ContextInjected { ids, .. } => Some(ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(injected[0], ["note:3"]);
}

/// Context sent before the session starts goes with its first input.
#[tokio::test]
async fn context_sent_before_the_session_goes_with_its_first_input() {
    let fake = Fake::new(json!([[complete("Found them")]]));
    let task = Task::new(
        config(&fake, &[]),
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
    let (task, events) = run.finish().await;
    assert!(task.pending_context_injections.is_empty());
    let first = &user_texts(&fake)[0];
    assert!(first.starts_with("Find todos."), "{first}");
    assert!(
        first.contains("# Workflow context update\n\nMessage from agent 'lead': skip vendor/."),
        "{first}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ContextInjected { .. }))
    );
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

/// Gives the fresh conversation its configuration.
struct ConfigHook(AgentConfig);

impl crate::handover::HandoverHook for ConfigHook {
    fn handover_config(&self, _task: &Task) -> Option<AgentConfig> {
        Some(self.0.clone())
    }
}

/// The handover hook's configuration runs the fresh session: its tools and
/// a system prompt written from it, and the task keeps it.
#[tokio::test]
async fn a_handover_s_fresh_session_runs_on_the_hook_s_configuration() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "REPORT: searched a and b."},
        ],
        [complete("Searched c")],
    ]));
    let mut saved = config(&fake, &["Grep"]);
    saved.handover = Some(HandoverConfig::at_input_tokens(1000));
    let mut current = saved.clone();
    current.instructions = "Search with Grep and Read.".into();
    current.tool_config.available_tool_names = vec!["Grep".into(), "Read".into()];
    let task = Task::new(saved, "Search a, b and c.".into(), fake.workspace(), None);
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_handover_hook(Arc::new(ConfigHook(current)));
    let (task, _) = start_with(task, runtime).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Searched c"),
        "{:?}",
        task.current_state
    );
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.agent_config.instructions, "Search with Grep and Read.");
    let argv = |index: usize| -> String {
        serde_json::from_value::<Vec<String>>(fake.logged("argv")[index]["argv"].clone())
            .unwrap()
            .join(" ")
    };
    let (first, fresh) = (argv(0), argv(1));
    assert!(first.contains("--tools Grep "), "{first}");
    assert!(!first.contains("Search with Grep and Read."), "{first}");
    assert!(fresh.contains("--tools Grep,Read"), "{fresh}");
    assert!(fresh.contains("Search with Grep and Read."), "{fresh}");
}

/// The handover hook's configuration brings its skills to the fresh
/// session: its system prompt lists them and `load_skill` is served for
/// them, from one skill to another, from none to some, and from some to
/// none (`load_skill` is no longer served).
#[tokio::test]
async fn a_handover_s_fresh_session_brings_the_configuration_s_skills() {
    let skills_of = |name: &str| crate::skill::SkillsConfig {
        embedded: vec![crate::skill::EmbeddedSkill::new(
            name,
            format!("---\nname: {name}\ndescription: The {name} skill.\n---\nThe {name} body.\n"),
        )],
        ..Default::default()
    };
    for (before, after) in [
        (Some("alpha"), Some("beta")),
        (None, Some("beta")),
        (Some("alpha"), None),
    ] {
        let wanted = after.or(before).unwrap();
        let fake = Fake::new(json!([
            [
                {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
                {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
                {"text": "REPORT: searched a and b."},
            ],
            [
                {"tool": "mcp__arko__load_skill", "input": {"name": wanted}},
                complete("Done"),
            ],
        ]));
        let mut saved = config(&fake, &["Grep"]);
        saved.handover = Some(HandoverConfig::at_input_tokens(1000));
        // As a host gives it, before the runtime sets its skills up.
        let mut current = saved.clone();
        current.skills_config = after.map(skills_of);
        saved.skills_config = before.map(skills_of);
        let mut task = Task::new(saved, "Search a and b.".into(), fake.workspace(), None);
        let mut registry = ToolRegistry::new();
        let prompt = task.agent_config.skills_config.clone().map(|skills| {
            crate::skill::SkillManager::load(&skills, &task.current_working_directory)
                .unwrap()
                .setup(&mut registry, &mut task.agent_config)
        });
        let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
            .with_handover_hook(Arc::new(ConfigHook(current)));
        let (events, commands, handle) =
            runtime.run_task(task, Arc::new(registry), None, false, prompt, None, None);
        let (task, _) = Run {
            events,
            commands,
            handle,
        }
        .finish()
        .await;
        assert_eq!(task.handovers.len(), 1, "{before:?} to {after:?}");
        let fresh = fake.logged("argv")[1]["argv"].to_string();
        let served = fake.logged("tools")[1]["tools"].to_string();
        let posts = fake.logged("post");
        let loaded = posts
            .iter()
            .find(|post| post["tool"] == "mcp__arko__load_skill")
            .map(|post| post["output"].to_string())
            .unwrap_or_default();
        match after {
            Some(name) => {
                assert!(fresh.contains(&format!("The {name} skill.")), "{fresh}");
                assert!(served.contains("load_skill"), "{served}");
                assert!(loaded.contains(&format!("The {name} body.")), "{loaded}");
            }
            None => {
                assert!(!fresh.contains("Available Skills"), "{fresh}");
                assert!(!served.contains("load_skill"), "{served}");
                assert!(!loaded.contains("body."), "{loaded}");
            }
        }
        if let Some(name) = before.filter(|_| before != after) {
            assert!(!fresh.contains(&format!("The {name} skill.")), "{fresh}");
        }
    }
}

/// The handover hook's configuration keeps its own entry for `load_skill`
/// (approval asked for), with the skills the same or other, in place of the
/// auto-approval the runtime gave it.
#[tokio::test]
async fn a_handover_s_fresh_session_keeps_the_configuration_s_load_skill_entry() {
    let skills_of = |name: &str| crate::skill::SkillsConfig {
        embedded: vec![crate::skill::EmbeddedSkill::new(
            name,
            format!("---\nname: {name}\ndescription: The {name} skill.\n---\nThe {name} body.\n"),
        )],
        ..Default::default()
    };
    for after in ["alpha", "beta"] {
        let fake = Fake::new(json!([
            [
                {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
                {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
                {"text": "REPORT: searched a and b."},
            ],
            [complete("Done")],
        ]));
        let mut saved = config(&fake, &["Grep"]);
        saved.handover = Some(HandoverConfig::at_input_tokens(1000));
        saved.skills_config = Some(skills_of("alpha"));
        let mut task = Task::new(saved, "Search a and b.".into(), fake.workspace(), None);
        let mut registry = ToolRegistry::new();
        let skills = task.agent_config.skills_config.clone().unwrap();
        let prompt = crate::skill::SkillManager::load(&skills, &task.current_working_directory)
            .unwrap()
            .setup(&mut registry, &mut task.agent_config);
        let mut current = task.agent_config.clone();
        current.skills_config = Some(skills_of(after));
        current.tool_config.tool_specific_configs.insert(
            crate::skill::LOAD_SKILL_TOOL_NAME.into(),
            PerToolConfigOverride {
                auto_approve_tool_use: Some(false),
                ..Default::default()
            },
        );
        let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
            .with_handover_hook(Arc::new(ConfigHook(current)));
        let (events, commands, handle) = runtime.run_task(
            task,
            Arc::new(registry),
            None,
            false,
            Some(prompt),
            None,
            None,
        );
        let (task, _) = Run {
            events,
            commands,
            handle,
        }
        .finish()
        .await;
        assert_eq!(task.handovers.len(), 1, "{after}");
        let tools = &task.agent_config.tool_config;
        assert_eq!(
            tools.tool_specific_configs[crate::skill::LOAD_SKILL_TOOL_NAME].auto_approve_tool_use,
            Some(false),
            "{after}"
        );
        assert!(
            tools
                .available_tool_names
                .contains(&crate::skill::LOAD_SKILL_TOOL_NAME.to_string())
        );
    }
}

/// A report asked for alone goes into a copy of the task's session as its
/// next input, also for an agent that compacts in Claude Code (no
/// `/compact` is sent).
/// Calls are held for it, `complete_task` too, and the task completes with
/// the reply as its summary, without a handover.
#[tokio::test]
async fn a_report_asked_for_alone_is_written_in_the_session() {
    let fake = Fake::new(json!([
        [complete("First")],
        [
            {"tool": "Grep", "input": {"pattern": "a"}},
            complete("Not yet"),
            {"text": "REPORT: did the first part."},
        ],
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig {
        compact_in_claude_code: true,
        ..HandoverConfig::at_input_tokens(1)
    });
    let task = Task::new(config, "Start.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    let session = task.claude_code_session.clone().unwrap();
    task.request_report(None).unwrap();
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "REPORT: did the first part."),
        "{:?}",
        task.current_state
    );
    assert!(task.handovers.is_empty());
    assert!(task.handover_requested.is_none());
    // Written in a copy of the session, which the task goes on in.
    let fork = task.claude_code_session.clone().unwrap();
    assert_ne!(fork, session);
    assert!(task.claude_code_fork_of.is_none());
    let users = user_texts(&fake);
    assert_eq!(users.len(), 2, "{users:?}");
    assert!(
        fake.logged("user")[1]["history"]
            .to_string()
            .contains("Start."),
        "the copy holds the conversation"
    );
    assert!(
        users[1].contains(crate::handover::DEFAULT_TEXT_REPORT_ONLY_REQUEST),
        "{}",
        users[1]
    );
    // The first run's `complete_task`, then the two calls held.
    let posts = fake.logged("post");
    assert_eq!(posts.len(), 3, "{posts:?}");
    assert!(
        posts[1..].iter().all(|post| post["is_error"] == true
            && post["output"]
                .as_str()
                .unwrap()
                .contains("Reply with it as plain text")),
        "{posts:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ConversationCompacted { .. }))
    );
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[1]["argv"].clone()).unwrap();
    assert!(argv.contains(&format!("--resume={session}")), "{argv:?}");
    assert!(argv.contains(&"--fork-session".to_string()), "{argv:?}");
    assert!(argv.contains(&format!("--session-id={fork}")), "{argv:?}");
}

/// A report asked of a branch whose session isn't copied yet (it hasn't
/// run) is written in a copy of the session the branch comes from, at the
/// branch's point, and the branch goes on in it.
#[tokio::test]
async fn a_report_asked_of_a_branch_that_hasn_t_run_copies_the_original_session() {
    let fake = Fake::new(json!([
        [complete("First")],
        [{"text": "REPORT: the branch did the first part."}],
    ]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Start.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start(task).finish().await;
    let session = task.claude_code_session.clone().unwrap();
    let mut branch = task.branch(task.conversation.messages.len()).unwrap();
    assert_eq!(
        branch.claude_code_fork_of.as_deref(),
        Some(session.as_str())
    );
    branch.request_report(None).unwrap();
    assert_eq!(
        branch.claude_code_fork_of.as_deref(),
        Some(session.as_str())
    );
    let (branch, _) = start(branch).finish().await;
    assert!(
        matches!(&branch.current_state, TaskState::Completed { summary } if summary == "REPORT: the branch did the first part."),
        "{:?}",
        branch.current_state
    );
    let copy = branch.claude_code_session.clone().unwrap();
    assert_ne!(copy, session);
    assert!(branch.claude_code_fork_of.is_none());
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[1]["argv"].clone()).unwrap();
    assert!(argv.contains(&format!("--resume={session}")), "{argv:?}");
    assert!(argv.contains(&"--fork-session".to_string()), "{argv:?}");
    assert!(argv.contains(&format!("--session-id={copy}")), "{argv:?}");
    assert!(
        fake.logged("user")[1]["history"]
            .to_string()
            .contains("Start."),
        "the copy holds the conversation"
    );
}

/// The host's threshold replaces the configured one, larger or smaller.
#[tokio::test]
async fn the_hosts_threshold_moves_the_handover_both_ways() {
    let with_threshold = |tokens| {
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
            .with_handover_threshold(crate::handover::HandoverThreshold::new(Some(tokens)))
    };
    // Raised over the configured 1,000: no handover at 1,500.
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
        {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
        complete("Searched a and b"),
    ]]));
    let mut raised = config(&fake, &["Grep"]);
    raised.handover = Some(HandoverConfig::at_input_tokens(1000));
    let task = Task::new(raised, "Search a and b.".into(), fake.workspace(), None);
    let (task, events) = start_with(task, with_threshold(5000)).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert!(task.handovers.is_empty());
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { .. }))
    );

    // Lowered under the configured 1,000,000: a handover at 1,500.
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "REPORT: searched a and b; next, c."},
        ],
        [complete("Searched c")],
    ]));
    let mut lowered = config(&fake, &["Grep"]);
    lowered.handover = Some(HandoverConfig::at_input_tokens(1_000_000));
    let task = Task::new(lowered, "Search a, b and c.".into(), fake.workspace(), None);
    let (task, events) = start_with(task, with_threshold(1000)).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(task.handovers[0].input_tokens, 1500);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { input_tokens: 1500 }))
    );
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

/// The calls of Claude Code's own tools, as the fake CLI ran them (without
/// the task's `complete_task` calls).
fn executed_built_ins(fake: &Fake) -> Vec<Value> {
    fake.logged("post")
        .into_iter()
        .filter(|post| post["tool"] != "mcp__arko__complete_task")
        .collect()
}

/// A task that ran in the workspace and moved to a `project` directory.
async fn moved_task(fake: &Fake, config: AgentConfig) -> (Task, PathBuf) {
    let project = fake.dir.join("project");
    (moved_task_to(fake, config, project.clone()).await, project)
}

/// A task that ran in the workspace and moved to `project`.
async fn moved_task_to(fake: &Fake, config: AgentConfig, project: PathBuf) -> Task {
    let task = Task::new(config, "Start.".into(), fake.workspace(), None);
    let (mut task, _) = start(task).finish().await;
    std::fs::create_dir_all(&project).unwrap();
    task.move_to(project);
    task.prepare_for_resumption_from_completed("Look around.".into())
        .unwrap();
    assert!(task.claude_code_directory.is_some());
    task
}

/// The CLI runs where the session started after the task moved, so the
/// calls of its own tools are given the task's directory, the one the
/// command guard and the path policy judge them from.
#[tokio::test]
async fn a_moved_task_runs_claude_codes_tools_in_its_directory() {
    let fake = Fake::new(json!([]));
    // A path that is absolute names the same file wherever the CLI runs.
    let absolute = fake.dir.join("project").join("notes.txt");
    let script = json!([[complete("First")], [
        {"tool": "Bash", "input": {"command": "ls build", "description": "List"}},
        {"tool": "Glob", "input": {"pattern": "*.rs"}},
        {"tool": "Grep", "input": {"pattern": "main", "path": "src"}},
        {"tool": "Read", "input": {"file_path": "Cargo.toml"}},
        {"tool": "Read", "input": {"file_path": "notes.txt"}},
        {"tool": "Read", "input": {"file_path": absolute.display().to_string()}},
        complete("Looked"),
    ]]);
    std::fs::write(fake.dir.join("script.json"), script.to_string()).unwrap();
    let (task, project) = moved_task(&fake, config(&fake, &["Bash", "Glob", "Grep", "Read"])).await;
    let (task, events) = start(task).finish().await;
    assert!(matches!(&task.current_state, TaskState::Completed { summary } if summary == "Looked"));

    let calls = executed_built_ins(&fake);
    let path = |name: &str| project.join(name).display().to_string();
    let input = |index: usize| &calls[index]["input"];
    if cfg!(unix) {
        assert_eq!(
            input(0)["command"],
            format!("cd -- '{}' || exit 1\nls build", project.display())
        );
    } else {
        assert!(
            input(0)["command"]
                .as_str()
                .unwrap()
                .ends_with("\nls build")
        );
    }
    assert_eq!(input(0)["description"], "List");
    assert_eq!(input(1)["path"], project.display().to_string());
    assert_eq!(input(1)["pattern"], "*.rs");
    assert_eq!(input(2)["path"], path("src"));
    assert_eq!(input(3)["file_path"], path("Cargo.toml"));
    assert_eq!(input(4)["file_path"], path("notes.txt"));
    assert_eq!(input(5)["file_path"], absolute.display().to_string());

    // The host sees the calls as the model wrote them.
    let formatted: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallFormatted {
                tool_name,
                parameters,
                ..
            } if tool_name == "Bash" => Some(parameters["command"].clone()),
            _ => None,
        })
        .collect();
    assert_eq!(formatted, [json!("ls build")]);
}

/// Claude Code makes a file tool's relative path absolute from the directory
/// it runs in, so after a move the path arrives under the session's
/// directory. It becomes the task's, for the path policy, the host and the
/// CLI, unless it is inside the task's directory already. A path outside the
/// session's directory, also one that leaves it by `..`, stays as it is and
/// is refused.
#[tokio::test]
async fn a_moved_tasks_file_paths_in_the_session_directory_become_the_tasks() {
    let fake = Fake::new(json!([]));
    // The task moves into a directory under the one the session started in.
    let session = fake.workspace();
    let project = session.join("project");
    // Joined a component at a time, so the separators are the platform's
    // (`\` on Windows), as in the path the mapping builds.
    let path = |directory: &Path, name: &str| {
        name.split('/')
            .fold(directory.to_path_buf(), |joined, part| joined.join(part))
            .display()
            .to_string()
    };
    let elsewhere = path(&std::env::temp_dir(), "arko-elsewhere.txt");
    let up = path(&session, "../up.txt");
    let script = json!([[complete("First")], [
        {"tool": "Read", "input": {"file_path": path(&session, "notes.txt")}},
        {"tool": "NotebookEdit", "input": {"notebook_path": path(&session, "n.ipynb"), "new_source": "x"}},
        {"tool": "Write", "input": {"file_path": path(&project, "kept.txt"), "content": "c"}},
        {"tool": "MultiEdit", "input": {"file_path": path(&session, "m.txt"),
                                        "edits": [{"old_string": "a", "new_string": "b"}]}},
        {"tool": "Edit", "input": {"file_path": path(&session, "src/lib.rs"),
                                   "old_string": "a", "new_string": "b"}},
        {"tool": "Read", "input": {"file_path": elsewhere}},
        {"tool": "Read", "input": {"file_path": up}},
        complete("Done"),
    ]]);
    std::fs::write(fake.dir.join("script.json"), script.to_string()).unwrap();
    let mut config = config(
        &fake,
        &["Read", "NotebookEdit", "Write", "MultiEdit", "Edit"],
    );
    config.tool_config.default_out_of_workspace_path_policy = PathSafetyPolicy::AutoReject;
    config.tool_config.tool_specific_configs.insert(
        "Edit".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let task = moved_task_to(&fake, config, project.clone()).await;
    let mut run = start(task);
    // The call that waits for approval shows the file it runs on, which is
    // inside the workspace.
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval {
        tool_call_id,
        parameters,
        approval_reason,
        ..
    }) = events.last()
    else {
        panic!("{events:#?}");
    };
    assert_eq!(parameters["file_path"], path(&project, "src/lib.rs"));
    assert_eq!(approval_reason.path_outside_workspace, None);
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(&task.current_state, TaskState::Completed { summary } if summary == "Done"));

    let calls = executed_built_ins(&fake);
    assert_eq!(calls.len(), 7);
    for call in &calls[..5] {
        assert_eq!(call["is_error"], false, "{call}");
    }
    assert_eq!(calls[0]["input"]["file_path"], path(&project, "notes.txt"));
    assert_eq!(
        calls[1]["input"]["notebook_path"],
        path(&project, "n.ipynb")
    );
    assert_eq!(calls[2]["input"]["file_path"], path(&project, "kept.txt"));
    assert_eq!(calls[3]["input"]["file_path"], path(&project, "m.txt"));
    assert_eq!(calls[4]["input"]["file_path"], path(&project, "src/lib.rs"));
    // Outside the session's directory, also by `..`: refused as written.
    for (call, given) in calls[5..].iter().zip([&elsewhere, &up]) {
        assert_eq!(call["is_error"], true, "{call}");
        assert!(
            call["output"]
                .as_str()
                .unwrap()
                .contains("outside the workspace"),
            "{call}"
        );
        assert_eq!(call["input"]["file_path"], *given);
    }
}

/// A call that waits for approval runs in the task's directory too.
#[tokio::test]
async fn an_approved_call_of_a_moved_task_runs_in_its_directory() {
    let fake = Fake::new(json!([[complete("First")], [
        {"tool": "Grep", "input": {"pattern": "main"}},
        complete("Looked"),
    ]]));
    let mut config = config(&fake, &["Grep"]);
    config.tool_config.tool_specific_configs.insert(
        "Grep".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let (task, project) = moved_task(&fake, config).await;
    let mut run = start(task);
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval {
        tool_call_id,
        parameters,
        ..
    }) = events.last()
    else {
        panic!("{events:#?}");
    };
    // The request shows the call as the model wrote it.
    assert!(!parameters.contains_key("path"), "{parameters:?}");
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await
        .unwrap();
    run.finish().await;
    let calls = executed_built_ins(&fake);
    assert_eq!(calls[0]["input"]["path"], project.display().to_string());
}

/// A task that didn't move runs where the CLI does: its calls are left as
/// the model wrote them.
#[tokio::test]
async fn a_task_that_didnt_move_keeps_its_calls_as_written() {
    let fake = Fake::new(json!([]));
    let notes = fake.workspace().join("notes.txt").display().to_string();
    let script = json!([[
        {"tool": "Bash", "input": {"command": "ls"}},
        {"tool": "Glob", "input": {"pattern": "*.rs"}},
        {"tool": "Read", "input": {"file_path": notes}},
        complete("Looked"),
    ]]);
    std::fs::write(fake.dir.join("script.json"), script.to_string()).unwrap();
    let task = Task::new(
        config(&fake, &["Bash", "Glob", "Read"]),
        "Look.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start(task).finish().await;
    assert!(task.claude_code_directory.is_none());
    let calls = executed_built_ins(&fake);
    assert_eq!(calls[0]["input"], json!({"command": "ls"}));
    assert_eq!(calls[1]["input"], json!({"pattern": "*.rs"}));
    assert_eq!(calls[2]["input"], json!({"file_path": notes}));
    for pre in fake.logged("pre") {
        assert!(
            pre["answer"]["hookSpecificOutput"]
                .get("updatedInput")
                .is_none(),
            "{pre}"
        );
    }
}

#[test]
fn shell_commands_change_into_the_directory_first() {
    let directory = std::path::Path::new(if cfg!(windows) {
        r"C:\it's here"
    } else {
        "/it's here"
    });
    let input = json!({"command": "make", "timeout": 5});
    let bash = super::in_directory("Bash", &input, directory).unwrap();
    let expected = if cfg!(windows) {
        "cd -- 'C:/it'\\''s here' || exit 1\nmake"
    } else {
        "cd -- '/it'\\''s here' || exit 1\nmake"
    };
    assert_eq!(bash, json!({"command": expected, "timeout": 5}));
    // A quote in the directory can't end PowerShell's string early, curly ones included.
    let directory = std::path::Path::new(if cfg!(windows) {
        r"C:\it’s"
    } else {
        "/it’s"
    });
    let shell = super::in_directory("PowerShell", &input, directory).unwrap();
    let expected = if cfg!(windows) {
        "Set-Location -LiteralPath 'C:\\it’’s' -ErrorAction Stop\nmake"
    } else {
        "Set-Location -LiteralPath '/it’’s' -ErrorAction Stop\nmake"
    };
    assert_eq!(shell["command"], expected);
}

#[test]
fn only_calls_that_depend_on_the_directory_are_changed() {
    let directory = std::path::Path::new(if cfg!(windows) { r"C:\work" } else { "/work" });
    let change = |name: &str, input: Value| super::in_directory(name, &input, directory);
    let inside = |relative: &str| directory.join(relative).display().to_string();
    let itself = directory.display().to_string();
    assert_eq!(
        change("Glob", json!({"pattern": "*"})),
        Some(json!({"pattern": "*", "path": itself}))
    );
    assert_eq!(
        change("Grep", json!({"pattern": "a", "path": ""})),
        Some(json!({"pattern": "a", "path": itself}))
    );
    assert_eq!(
        change("Edit", json!({"file_path": "src/../lib.rs"})),
        Some(json!({"file_path": inside("lib.rs")}))
    );
    // Absolute paths, paths from the home directory, and tools that name no path.
    let absolute = inside("a.txt");
    assert_eq!(change("Read", json!({"file_path": absolute})), None);
    assert_eq!(change("Read", json!({"file_path": "~/a.txt"})), None);
    assert_eq!(change("Read", json!({})), None);
    assert_eq!(
        change("WebFetch", json!({"url": "https://example.com"})),
        None
    );
    assert_eq!(change("Bash", json!({})), None);
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

/// The plan usage events a run reported.
fn plan_usage_events(events: &[AgentEvent]) -> Vec<&arko_ai::PlanUsage> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::PlanUsageReported(usage) => Some(usage),
            _ => None,
        })
        .collect()
}

/// The event's flags about extra usage reach the plan usage, and Claude
/// Code's usage data (`get_usage`, asked once however many events come) adds
/// the amounts to it and to the events after it.
#[tokio::test]
async fn extra_usage_follows_the_event_and_the_usage_data() {
    let event = json!({
        "status": "rejected", "rateLimitType": "five_hour", "isUsingOverage": true,
        "resetsAt": 1_793_152_800_u64, "overageStatus": "allowed",
        "overageResetsAt": 1_793_152_800_u64, "unifiedWindows": windows(1.0, 0.4),
    });
    // The fake answers once the test has seen the first report, so the order
    // is the same on a slow machine.
    let fake = Fake::new(json!([[
        {"rate_limit": event},
        {"serve": 1, "gated": true},
        {"rate_limit": event},
        complete("done"),
    ]]));
    fake.answer_usage(json!({
        "session": {"total_cost_usd": 0.0, "total_api_duration_ms": 0, "total_duration_ms": 0,
                    "total_lines_added": 0, "total_lines_removed": 0, "model_usage": {}},
        "subscription_type": "max", "rate_limits_available": true,
        "rate_limits": {
            "five_hour": {"utilization": 100, "resets_at": "2026-10-23T12:00:00Z"},
            "extra_usage": {"is_enabled": true, "monthly_limit": 50000, "used_credits": 27140,
                            "utilization": 54, "currency": "USD", "disabled_reason": null},
        },
        "behaviors": null,
    }));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let mut run = start(task);
    let mut events = run
        .events_until(|event| matches!(event, AgentEvent::PlanUsageReported(_)))
        .await;
    std::fs::write(fake.gate(), "").unwrap();
    events.extend(run.events_until(|_| false).await);
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    let usage = plan_usage_events(&events);
    assert_eq!(usage.len(), 3, "{usage:#?}");
    let flags = usage[0].extra_usage.as_ref().unwrap();
    assert!(flags.in_use);
    assert_eq!(flags.status.as_deref(), Some("allowed"));
    assert_eq!(flags.resets_at, Some(1_793_152_800));
    assert_eq!(flags.monthly_limit, None);
    // The answer and the event after it have the amounts.
    for with_amounts in &usage[1..] {
        let extra = with_amounts.extra_usage.as_ref().unwrap();
        assert!(extra.in_use);
        assert_eq!(extra.enabled, Some(true));
        assert_eq!(extra.status.as_deref(), Some("allowed"));
        assert_eq!(extra.monthly_limit, Some(50_000.0));
        assert_eq!(extra.used_credits, Some(27_140.0));
        assert_eq!(extra.utilization, Some(54.0));
        assert_eq!(extra.currency.as_deref(), Some("USD"));
        assert_eq!(extra.decimal_places, Some(2));
    }
    assert_eq!(fake.logged("get_usage").len(), 1);
    assert_eq!(fake.logged("get_usage")[0]["skip_behaviors"], true);
}

/// A chat turn's event comes right before its result, so the amounts have to
/// be asked for at the turn's start. The answer, slow here, comes first and the
/// event has the amounts. The currency says where the decimal point is.
#[tokio::test]
async fn the_event_of_a_chat_turn_has_the_amounts_asked_at_its_start() {
    let fake = Fake::new(json!([[
        {"serve": 1, "after": 0.3},
        {"rate_limit": {"status": "allowed_warning", "rateLimitType": "five_hour",
                        "overageStatus": "allowed", "unifiedWindows": windows(0.5, 0.4)}},
        {"text": "hello"},
    ]]));
    fake.answer_usage(json!({
        "rate_limits": {"extra_usage": {"is_enabled": true, "monthly_limit": 500000,
                                        "used_credits": 271400, "utilization": 54,
                                        "currency": "JPY", "decimal_places": null}},
    }));
    let task = Task::new(chat_config(&fake), "Hello.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run.events_until(|_| false).await;
    run.finish().await;

    let usage = plan_usage_events(&events);
    assert_eq!(usage.len(), 2, "{usage:#?}");
    let extra = usage[0].extra_usage.as_ref().unwrap();
    assert_eq!(extra.used_credits, Some(271_400.0));
    assert_eq!(extra.currency.as_deref(), Some("JPY"));
    assert_eq!(extra.decimal_places, Some(0));
    assert_eq!(fake.logged("get_usage").len(), 2);
}

/// The amounts are asked for again as a turn ends, so the plan usage after
/// the turn has what the turn used.
#[tokio::test]
async fn the_amounts_are_asked_again_as_a_turn_ends() {
    let fake = Fake::new(json!([[
        {"serve": 1},
        {"rate_limit": {"status": "allowed_warning", "rateLimitType": "five_hour",
                        "overageStatus": "allowed", "unifiedWindows": windows(0.5, 0.4)}},
        {"text": "hello"},
    ]]));
    let used = |credits: u64| {
        json!({"rate_limits": {"extra_usage": {"is_enabled": true, "monthly_limit": 50000,
                                               "used_credits": credits, "currency": "USD"}}})
    };
    fake.answer_usage(json!([used(27_140), used(27_390)]));
    let task = Task::new(chat_config(&fake), "Hello.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run.events_until(|_| false).await;
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    let usage = plan_usage_events(&events);
    let used: Vec<_> = usage
        .iter()
        .map(|usage| usage.extra_usage.as_ref().unwrap().used_credits)
        .collect();
    assert_eq!(used, [Some(27_140.0), Some(27_390.0)], "{usage:#?}");
    assert_eq!(fake.logged("get_usage").len(), 2);
    // The last report comes before the state that ends the task.
    let position = |wanted: fn(&AgentEvent) -> bool| events.iter().rposition(wanted).unwrap();
    assert!(
        position(|event| matches!(event, AgentEvent::PlanUsageReported(_)))
            < position(|event| matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::Completed { .. })
            )),
        "{events:#?}"
    );
}

/// Claude Code's own decimal places are kept, and the others are two, or none
/// for the currencies it shows in whole units.
#[test]
fn the_decimal_places_of_the_amounts_follow_the_currency() {
    let read = |extra: Value| {
        super::extra_usage_read(&json!({"rate_limits": {"extra_usage": extra}}))
            .unwrap()
            .decimal_places
    };
    assert_eq!(
        read(json!({"currency": "USD", "decimal_places": 3})),
        Some(3)
    );
    assert_eq!(read(json!({"currency": "EUR"})), Some(2));
    for currency in ["JPY", "krw", "VND"] {
        assert_eq!(read(json!({"currency": currency})), Some(0));
    }
    assert_eq!(read(json!({"is_enabled": false, "currency": null})), None);
}

/// Without usage data (a CLI that doesn't know the request), the plan usage
/// has the event's flags. The event a real Claude Code wrote for an account
/// whose organization turned extra usage off, and an event of the usage
/// credits' own limit, which isn't a plan window.
#[tokio::test]
async fn extra_usage_follows_the_event_alone_without_usage_data() {
    let fake = Fake::new(json!([[
        {"rate_limit": {
            "status": "allowed_warning", "resetsAt": 1_793_152_800_u64,
            "rateLimitType": "seven_day", "utilization": 0.71, "overageStatus": "rejected",
            "overageDisabledReason": "org_level_disabled", "isUsingOverage": false,
            "unifiedWindows": {
                "five_hour": {"utilization": 0.09, "resetsAt": 1_792_764_000_u64},
                "seven_day": {"utilization": 0.71, "resetsAt": 1_793_152_800_u64},
            },
            "overageEnabled": false,
        }},
        {"serve": 1},
        {"rate_limit": {"status": "allowed_warning", "rateLimitType": "overage",
                        "utilization": 0.9, "overageStatus": "allowed_warning"}},
        complete("done"),
    ]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run.events_until(|_| false).await;
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    let usage = plan_usage_events(&events);
    assert_eq!(usage.len(), 2, "{usage:#?}");
    assert_eq!(
        usage[0].extra_usage,
        Some(arko_ai::ExtraUsage {
            enabled: Some(false),
            status: Some("rejected".into()),
            disabled_reason: Some("org_level_disabled".into()),
            ..Default::default()
        })
    );
    assert!((usage[0].secondary.as_ref().unwrap().used_percent - 71.0).abs() < 1e-9);
    let credits = usage[1];
    assert!(credits.primary.is_none() && credits.secondary.is_none());
    let extra = credits.extra_usage.as_ref().unwrap();
    assert_eq!(extra.status.as_deref(), Some("allowed_warning"));
    assert!((extra.utilization.unwrap() - 90.0).abs() < 1e-9);
    assert_eq!(extra.monthly_limit, None);
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
async fn the_cli_doesnt_update_itself() {
    // An update would swap the binary under the host's other sessions.
    let fake = Fake::new(json!([[complete("done")]]));
    let task = Task::new(config(&fake, &[]), "Do it.".into(), fake.workspace(), None);
    start(task).finish().await;
    assert_eq!(fake.logged("argv")[0]["env"]["DISABLE_AUTOUPDATER"], "1");
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
        // Claude Code's retry is reported first (an error needing no
        // decision), and the pause's error follows.
        let events = run
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
async fn the_long_cache_is_passed_on_as_the_hour_and_its_opposite_as_five_minutes() {
    for (long, ttl, minutes) in [(true, "1h", 60), (false, "5m", 5)] {
        let fake = Fake::new(json!([[complete("done")]]));
        let mut config = config(&fake, &[]);
        config.llm_config.generation_config.long_prompt_cache = Some(long);
        let task = Task::new(config, "Do it.".into(), fake.workspace(), None);
        let (task, _) = start(task).finish().await;
        let argv: Vec<String> =
            serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
        let settings = argv.iter().position(|arg| arg == "--settings").unwrap();
        assert!(
            argv[settings + 1].contains(&format!("\"promptCacheTtl\":\"{ttl}\"")),
            "{argv:?}"
        );
        assert_eq!(
            arko_ai::prompt_cache::prompt_cache_lifetime(
                &task.agent_config.llm_config.model,
                &task.agent_config.llm_config.generation_config,
                false
            ),
            arko_ai::PromptCacheLifetime::Known(Duration::from_secs(minutes * 60))
        );
    }
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

/// A report asked for alone of a task cancelled while a call ran doesn't
/// run the call again: the session hears its outcome is unknown, then the
/// report's request.
#[tokio::test]
async fn a_report_asked_of_a_cancelled_task_runs_no_open_call_again() {
    let fake = Fake::new(json!([
        [{
            "tool": "mcp__arko__read_file",
            "input": {"path": "a.txt"},
            "id": "toolu_read",
            "crash": "pre",
        }],
        [{"text": "REPORT: read nothing yet."}],
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
    let mut saved = recorder.latest(|task| {
        !task.running_tool_calls.is_empty() && task.current_state == TaskState::InProgress
    });
    saved.set_state(TaskState::Cancelled {
        reason: "The agent did not stop in time and was aborted.".into(),
    });
    saved.request_report(None).unwrap();
    let (task, resumed) = run_saved(&fake, &saved).await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "REPORT: read nothing yet."),
        "{:?}",
        task.current_state
    );
    assert!(resumed.contains("its outcome is unknown"), "{resumed}");
    assert!(!resumed.contains("CONTENTS-OF-A"), "{resumed}");
    assert!(
        resumed.contains(crate::handover::DEFAULT_TEXT_REPORT_ONLY_REQUEST),
        "{resumed}"
    );
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
    // A PNG signature and a 16×16 size before the rest.
    let bytes = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x10\0\0\0\x10completed-image-fixture";
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
    assert!(!call.output.as_ref().unwrap().contains(&**data));
    assert!(
        !format!("{call:?}").contains(&**data),
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
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let resumed = fake.logged("user")[1]["text"].as_str().unwrap().to_string();
    assert!(!resumed.contains("tool calls were open"), "{resumed}");
}

/// An agent that compacts at `tokens`, with a handover hook whose context
/// follows each compaction.
fn compacting(fake: &Fake, tools: &[&str], tokens: usize) -> AgentConfig {
    let mut config = config(fake, tools);
    config.handover = Some(HandoverConfig {
        compact_in_claude_code: true,
        ..HandoverConfig::at_input_tokens(tokens)
    });
    config
}

struct StandingNotes;

impl crate::handover::HandoverHook for StandingNotes {
    fn on_compaction(&self, _task: &Task) -> Option<String> {
        Some("# Notes on your task\n- Keep the tests green.".into())
    }
}

fn with_notes() -> AgentRuntime {
    AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_handover_hook(Arc::new(StandingNotes))
}

fn user_texts(fake: &Fake) -> Vec<String> {
    fake.logged("user")
        .iter()
        .map(|entry| entry["text"].as_str().unwrap().to_string())
        .collect()
}

/// At the threshold, mid-turn, the next call is held and the turn
/// interrupted (its next step doesn't run). `/compact` goes into the same
/// session, and the agent continues from Claude Code's summary, told of
/// the held call and given the hook's context again. No handover takes
/// place.
#[tokio::test]
async fn a_compaction_holds_the_next_call_and_continues_in_the_same_session() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "Not reached: the turn was interrupted."},
        ],
        {"compact": {"pre_tokens": 1520, "post_tokens": 300}},
        [
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 600},
            complete("Searched a and b"),
        ],
    ]));
    let task = Task::new(
        compacting(&fake, &["Grep"], 1000),
        "Search a and b.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start_with(task, with_notes()).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert!(task.handovers.is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationCompacted {
            input_tokens: 1520,
            compacted_tokens: Some(300)
        }
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { .. }))
    );
    // The call at the threshold was held, not run, and said so.
    let posts = fake.logged("post");
    assert_eq!(posts[1]["is_error"], true);
    let held = posts[1]["output"].as_str().unwrap();
    assert!(
        held.starts_with(
            "Not an error: arko held this Grep call because Claude Code compacts your conversation now."
        ),
        "{held}"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolRejected { tool_name, reason: Some(reason), .. }
            if tool_name == "Grep" && reason.starts_with("Held so that Claude Code can compact")
    )));
    // One session: `/compact` as it is, then the note that continues.
    assert_eq!(fake.logged("argv").len(), 1);
    let users = user_texts(&fake);
    assert_eq!(users.len(), 3, "{users:#?}");
    assert_eq!(users[1], "/compact");
    let note = &users[2];
    assert!(
        note.contains("arko held these tool calls of yours and ended your turn"),
        "{note}"
    );
    assert!(note.contains(r#"- `Grep` with {"pattern":"b"}"#), "{note}");
    assert!(
        note.contains("Claude Code compacted your conversation"),
        "{note}"
    );
    assert!(
        note.contains("# Workflow context update\n\n# Notes on your task\n- Keep the tests green."),
        "{note}"
    );
    // Claude Code's own compaction stays off.
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    let settings = argv.iter().position(|arg| arg == "--settings").unwrap();
    assert!(argv[settings + 1].contains("\"autoCompactEnabled\":false"));
    // The task keeps the whole conversation, and its usage after the
    // compaction counts as the current conversation's.
    let texts = texts(&task);
    assert!(texts.iter().any(|text| text.starts_with("Search a and b.")));
    assert!(!texts.iter().any(|text| text == "/compact"));
    assert!(!texts.iter().any(|text| text.starts_with("Not reached")));
    assert_eq!(task.conversation_usage_start, 2);
    assert_eq!(task.usage_history[2].input_tokens, 600);
}

/// After a compaction, the questions still open come back with the
/// agent's next input, after the hook's context.
#[tokio::test]
async fn a_compaction_restates_the_open_questions() {
    let fake = Fake::new(json!([
        [
            {"tool": "mcp__arko__ask_user", "input": {"questions": ["Tabs or spaces?"], "wait": false}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "Not reached: the turn was interrupted."},
        ],
        {"compact": {"pre_tokens": 1520, "post_tokens": 300}},
        [complete("Searched")],
    ]));
    let mut config = compacting(&fake, &["Grep", "ask_user"], 1000);
    config.tool_config.async_questions = true;
    let task = Task::new(config, "Search b.".into(), fake.workspace(), None);
    let (task, _) = start_with(task, with_notes()).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let users = user_texts(&fake);
    let note = users.last().unwrap();
    assert!(
        note.contains("# Notes on your task\n- Keep the tests green.\n\n# Your open questions"),
        "{note}"
    );
    assert!(note.contains("q1 Tabs or spaces?"), "{note}");
}

/// A host that has answered `q1` and keeps `q2` open.
struct KeepsQ2;

impl crate::handover::HandoverHook for KeepsQ2 {
    fn open_questions(&self, task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        Some(
            task.questions
                .open
                .iter()
                .filter(|question| question.id == "q2")
                .cloned()
                .collect(),
        )
    }
}

/// The host answers `q1` of a batch the agent waited for (an ask without
/// `wait` waits) and keeps `q2` open: the log settles `q1` alone, and the
/// compaction after it restates `q2`.
#[tokio::test]
async fn a_compaction_after_a_partial_answer_restates_the_rest() {
    let fake = Fake::new(json!([
        [{"tool": "mcp__arko__ask_user", "input": {"questions": ["Tabs or spaces?", "Push now?"]}, "input_tokens": 400}],
        [
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "Not reached: the turn was interrupted."},
        ],
        {"compact": {"pre_tokens": 1520, "post_tokens": 300}},
        [complete("Searched")],
    ]));
    let mut config = compacting(&fake, &["Grep", "ask_user"], 1000);
    config.tool_config.async_questions = true;
    let task = Task::new(config, "Search b.".into(), fake.workspace(), None);
    let mut run = start_with(
        task,
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
            .with_handover_hook(Arc::new(KeepsQ2)),
    );
    run.events_until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "The user answered q1.\nq1 Tabs or spaces? Spaces\nStill open: q2.".into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.questions.open.len(), 1);
    assert_eq!(task.questions.open[0].id, "q2");
    let users = user_texts(&fake);
    let note = users.last().unwrap();
    assert!(note.contains("# Your open questions"), "{note}");
    assert!(note.contains("q2 Push now?"), "{note}");
    assert!(!note.contains("q1 Tabs or spaces?"), "{note}");
}

/// A conversation whose last turn ended over the threshold compacts as
/// its next session starts: `/compact` first, then the task's new input.
#[tokio::test]
async fn a_compaction_due_as_a_session_starts_comes_first() {
    let fake = Fake::new(json!([
        [
            {"text": "Looking.", "input_tokens": 400},
            {"tool": "mcp__arko__complete_task", "input": {"summary": "First answer."},
             "input_tokens": 1500},
        ],
        {"compact": {"pre_tokens": 1510, "post_tokens": 200}},
        [complete("Second answer.")],
    ]));
    let task = Task::new(
        compacting(&fake, &[], 1000),
        "First question.".into(),
        fake.workspace(),
        None,
    );
    let (mut task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ConversationCompacted { .. })),
        "complete_task ends the turn before a compaction"
    );
    task.prepare_for_resumption_from_completed("Second question.".into())
        .unwrap();
    let (task, events) = start_with(task, with_notes()).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Second answer."),
        "{:?}",
        task.current_state
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationCompacted {
            input_tokens: 1510,
            ..
        }
    )));
    let users = user_texts(&fake);
    assert_eq!(users.len(), 3, "{users:#?}");
    assert_eq!(users[1], "/compact");
    // The new input continues the task: no note to continue is needed.
    assert!(users[2].starts_with("Second question."), "{}", users[2]);
    assert!(!users[2].contains("Claude Code compacted"), "{}", users[2]);
    assert!(users[2].contains("Keep the tests green."), "{}", users[2]);
    // The same session, resumed.
    let argvs = fake.logged("argv");
    let second: Vec<String> = serde_json::from_value(argvs[1]["argv"].clone()).unwrap();
    assert!(second.iter().any(|arg| arg.starts_with("--resume=")));
}

/// A host's request compacts an agent that compacts: sent before its next
/// run starts, before that run's input. Sent before anything, it has nothing
/// to compact.
#[tokio::test]
async fn a_host_request_compacts() {
    let fake = Fake::new(json!([
        [complete("First answer.")],
        {"compact": {"pre_tokens": 800, "post_tokens": 100}},
        [complete("Second answer.")],
    ]));
    let task = Task::new(
        compacting(&fake, &[], 100_000),
        "First question.".into(),
        fake.workspace(),
        None,
    );
    let run = start(task);
    run.commands
        .send(UserCommand::RequestHandover)
        .await
        .unwrap();
    let (mut task, events) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ConversationCompacted { .. }))
    );
    task.prepare_for_resumption_from_completed("Second question.".into())
        .unwrap();
    let run = start(task);
    run.commands
        .send(UserCommand::RequestHandover)
        .await
        .unwrap();
    let (task, events) = run.finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "Second answer.")
    );
    assert!(task.handovers.is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationCompacted {
            input_tokens: 800,
            ..
        }
    )));
    let users = user_texts(&fake);
    assert_eq!(users.len(), 3, "{users:#?}");
    assert_eq!(users[1], "/compact");
    assert!(users[2].starts_with("Second question."), "{}", users[2]);
}

/// When Claude Code doesn't compact (`/compact` fails, or this Claude
/// Code passes it to the model as a message), the agent hands over
/// instead: the report request follows, and a fresh session starts from
/// the report.
#[tokio::test]
async fn a_compaction_that_doesnt_happen_falls_back_to_a_handover() {
    for compaction in [
        json!({"compact": {"error": "Not enough messages to compact."}}),
        json!([{"text": "I don't know /compact."}]),
    ] {
        let fake = Fake::new(json!([
            [
                {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
                {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
                {"text": "Not reached: the turn was interrupted."},
            ],
            compaction,
            [{"text": "REPORT: searched a; next, b."}],
            [complete("Searched b")],
        ]));
        let task = Task::new(
            compacting(&fake, &["Grep"], 1000),
            "Search a and b.".into(),
            fake.workspace(),
            None,
        );
        let (task, events) = start(task).finish().await;
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{:?}",
            task.current_state
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ConversationCompacted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::HandoverRequested { .. }))
        );
        assert_eq!(task.handovers.len(), 1);
        assert_eq!(task.handovers[0].report, "REPORT: searched a; next, b.");
        let users = user_texts(&fake);
        assert_eq!(users[1], "/compact");
        assert!(
            users[2].contains("Reply with that report as plain text"),
            "{}",
            users[2]
        );
        assert!(users[3].contains(REPORT_HEADING));
        assert_eq!(fake.logged("argv").len(), 2, "a fresh session");
    }
}

/// A turn that fails over the model's input limit compacts at once.
#[tokio::test]
async fn going_over_the_input_limit_compacts() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"fail": "Prompt is too long", "terminal_reason": "prompt_too_long"},
        ],
        {"compact": {"pre_tokens": 1_000_100, "post_tokens": 900}},
        [complete("done")],
    ]));
    let task = Task::new(
        compacting(&fake, &["Grep"], 500_000),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationCompacted {
            input_tokens: 1_000_100,
            ..
        }
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
    );
    let users = user_texts(&fake);
    assert_eq!(users[1], "/compact");
    assert!(users[2].contains("Claude Code compacted your conversation"));
}

struct Replaced(std::sync::Mutex<Vec<arko_ai::Message>>);

impl crate::handover::HandoverHook for Replaced {
    fn on_handover(&self, task: &Task, _report: &str) -> crate::handover::HandoverSeed {
        *self.0.lock().unwrap() = task.conversation.messages.clone();
        crate::handover::HandoverSeed::for_task(task)
    }
}

/// A turn over the model's input limit as Claude Code 2.1.299 reports it
/// when it refuses the request itself: its own message ("Prompt is too
/// long", `invalid_request`), then an error result.
fn too_long() -> Value {
    json!({"api_error": "invalid_request", "text": "Prompt is too long",
           "terminal_reason": "blocking_limit"})
}

/// The step that took the conversation over the limit: a search whose
/// result is too large, after one whose result fit.
fn oversized_step() -> Value {
    json!([
        {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400, "output": "a is in lib.rs"},
        {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 500, "output": "HUGE RESULT"},
        too_long(),
    ])
}

fn argv(fake: &Fake, index: usize) -> Vec<String> {
    serde_json::from_value(fake.logged("argv")[index]["argv"].clone()).unwrap()
}

/// The conversation a user message went to in the fake, oldest first.
fn history(entry: &Value) -> Vec<String> {
    entry["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|text| text.as_str().unwrap().to_string())
        .collect()
}

/// An agent that hands over (not compacting) whose step takes the
/// conversation over the model's input limit hands over without that step:
/// the session resumes at the message that ended the last request Claude
/// Code answered (the first search's result), and the report is asked for
/// there, told that the step was left out. The fresh session starts from
/// the report.
#[tokio::test]
async fn going_over_the_input_limit_hands_over_without_the_step() {
    let fake = Fake::new(json!([
        oversized_step(),
        [{"text": "REPORT: a is in lib.rs; b needs a narrower search."}],
        [complete("Found b")],
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(100_000));
    let task = Task::new(config, "Find a and b.".into(), fake.workspace(), None);
    let replaced = Arc::new(Replaced(Default::default()));
    let runtime =
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_handover_hook(replaced.clone());
    let (task, events) = start_with(task, runtime).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(
        task.handovers[0].report,
        "REPORT: a is in lib.rs; b needs a narrower search."
    );
    // At the last answered request's input, reported as an error that
    // needs no decision.
    assert_eq!(task.handovers[0].input_tokens, 500);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered {
            requires_approval_to_retry: false,
            error_kind: Some(arko_ai::ErrorKind::ContextOverflow),
            ..
        }
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::HandoverRequested { input_tokens: 500 }))
    );
    // The same session resumed at the first search's result, then a fresh
    // one started.
    assert_eq!(fake.logged("argv").len(), 3);
    let session = argv(&fake, 0)
        .iter()
        .find_map(|arg| arg.strip_prefix("--session-id=").map(str::to_string))
        .unwrap();
    let resumed = argv(&fake, 1);
    assert!(
        resumed.contains(&format!("--resume={session}")),
        "{resumed:?}"
    );
    assert!(
        resumed
            .iter()
            .any(|arg| arg.starts_with("--resume-session-at=")),
        "{resumed:?}"
    );
    assert!(!argv(&fake, 2).contains(&format!("--session-id={session}")));
    // The report request went to the conversation without the step: it
    // ends with the first search's result.
    let users = fake.logged("user");
    let request = users[1]["text"].as_str().unwrap();
    assert!(
        request.starts_with(crate::handover::STEP_LEFT_OUT),
        "{request}"
    );
    assert!(request.contains("Reply with that report as plain text"));
    let before = history(&users[1]);
    assert_eq!(before.last().unwrap(), "a is in lib.rs", "{before:?}");
    assert!(
        !before.iter().any(|text| text.contains("HUGE RESULT")
            || text.contains("Prompt is too long")
            || text.contains("\"b\"")),
        "{before:?}"
    );
    assert!(users[2]["text"].as_str().unwrap().contains(REPORT_HEADING));
    // The replaced conversation, as the task kept it, had the first search
    // and not the second.
    let old = replaced.0.lock().unwrap().clone();
    let calls: Vec<&Value> = old
        .iter()
        .flat_map(|message| &message.tool_calls)
        .map(|call| &call.arguments)
        .collect();
    assert_eq!(calls, [&json!({"pattern": "a"})]);
    let results: Vec<&str> = old
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(arko_ai::Content::as_tool_result)
        .map(|result| result.output.as_str())
        .collect();
    assert_eq!(results, ["a is in lib.rs"]);
}

/// A step that starts a turn is left out back to the message arko sent. Its
/// calls are one response, the result of the first coming before the second
/// call, as Claude Code reports parallel calls.
#[tokio::test]
async fn a_step_that_starts_a_turn_is_left_out_back_to_its_message() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 500, "output": "a is in lib.rs"},
            {"tool": "Grep", "input": {"pattern": "b"}, "continues": true, "output": "HUGE RESULT"},
            too_long(),
        ],
        [{"text": "REPORT: b needs a narrower search."}],
        [complete("Found b")],
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(100_000));
    let task = Task::new(config, "Find b.".into(), fake.workspace(), None);
    let (task, _) = start(task).finish().await;
    assert_eq!(task.handovers.len(), 1);
    let users = fake.logged("user");
    let before = history(&users[1]);
    assert_eq!(before.len(), 1, "{before:?}");
    assert!(before[0].starts_with("Find b."), "{before:?}");
    assert!(
        users[1]["text"]
            .as_str()
            .unwrap()
            .starts_with(crate::handover::STEP_LEFT_OUT)
    );
}

/// A report request that goes over the limit too ends the run on the error:
/// a step is left out once per handover.
#[tokio::test]
async fn a_step_is_left_out_once() {
    let fake = Fake::new(json!([
        oversized_step(),
        [too_long()],
        [complete("Not reached")]
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(100_000));
    let task = Task::new(config, "Find a and b.".into(), fake.workspace(), None);
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
                error_kind: Some(arko_ai::ErrorKind::ContextOverflow),
                ..
            }
        ))
    ));
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    let (task, _) = run.finish().await;
    assert!(task.handovers.is_empty());
    assert_eq!(fake.logged("argv").len(), 2, "one resumed session, no more");
    assert_eq!(fake.logged("user").len(), 2);
}

/// When an agent that compacts goes over the limit and Claude Code then
/// doesn't compact, the report request it falls back to goes over the limit
/// as well, and the step is left out as above.
#[tokio::test]
async fn a_compaction_that_doesnt_happen_over_the_limit_hands_over_without_the_step() {
    let fake = Fake::new(json!([
        oversized_step(),
        {"compact": {"error": "Conversation too long."}},
        [too_long()],
        [{"text": "REPORT: a is in lib.rs; b needs a narrower search."}],
        [complete("Found b")],
    ]));
    let task = Task::new(
        compacting(&fake, &["Grep"], 100_000),
        "Find a and b.".into(),
        fake.workspace(),
        None,
    );
    let (task, _) = start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(task.handovers.len(), 1);
    let users = fake.logged("user");
    let texts: Vec<&str> = users
        .iter()
        .map(|entry| entry["text"].as_str().unwrap())
        .collect();
    assert_eq!(texts[1], "/compact");
    assert!(texts[2].contains("Reply with that report as plain text"));
    assert!(
        texts[3].starts_with(crate::handover::STEP_LEFT_OUT),
        "{}",
        texts[3]
    );
    assert_eq!(history(&users[3]).last().unwrap(), "a is in lib.rs");
    assert!(texts[4].contains(REPORT_HEADING));
    assert!(
        argv(&fake, 1)
            .iter()
            .any(|arg| arg.starts_with("--resume-session-at="))
    );
}

/// A compaction Claude Code makes on its own (its auto-compaction, on for
/// an agent without `handover`) is reported too.
#[tokio::test]
async fn claude_codes_own_compaction_is_reported() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 900},
        {"system": {"subtype": "compact_boundary",
                    "compact_metadata": {"trigger": "auto", "pre_tokens": 950, "post_tokens": 120}}},
        complete("done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ConversationCompacted {
            input_tokens: 950,
            compacted_tokens: Some(120)
        }
    )));
    assert_eq!(task.conversation_usage_start, 1);
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    assert!(!argv.iter().any(|arg| arg.contains("autoCompactEnabled")));
}

// `text_reply_ends_task`: a turn that ends with text is the final reply.

/// `config` for an agent whose reply in text ends the task.
fn replying(fake: &Fake, tools: &[&str]) -> AgentConfig {
    let mut config = config(fake, tools);
    config.tool_config.text_reply_ends_task = true;
    config
}

/// The summaries of the run's completions, as the host is told of them.
fn completions(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TaskStateChanged(TaskState::Completed { summary }) => Some(summary.clone()),
            _ => None,
        })
        .collect()
}

/// A turn that ends with text completes the task with the text as its
/// summary, reported as a `complete_task` completion is. `complete_task`
/// isn't served (nor offered as a built-in), and the system prompt doesn't
/// name it. A turn whose last response called a tool isn't a reply (the
/// fake reports its earlier text as the turn's result), and one that ends
/// without text gets a note that doesn't name `complete_task`.
#[tokio::test]
async fn a_turn_that_ends_with_text_completes_the_task() {
    let fake = Fake::new(json!([
        [
            {"text": "Looking.", "input_tokens": 100},
            {"tool": "Grep", "input": {"pattern": "a"}},
        ],
        [{"tool": "Grep", "input": {"pattern": "b"}}],
        [
            {"text": "Searching again."},
            {"tool": "Grep", "input": {"pattern": "c"}},
            {"text": "Found 3 files."},
        ],
    ]));
    let mut config = replying(&fake, &["Grep"]);
    config
        .tool_config
        .available_tool_names
        .push("complete_task".into());
    let task = Task::new(config, "Count the files.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;

    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "Found 3 files.".into()
        }
    );
    assert_eq!(completions(&events), ["Found 3 files."]);
    assert!(matches!(events.last(), Some(AgentEvent::TaskStreamEnd)));
    let users = user_texts(&fake);
    assert_eq!(users.len(), 3, "{users:#?}");
    for note in &users[1..] {
        assert!(note.starts_with(super::OPEN_TURN_NO_REPLY_NOTE), "{note}");
    }
    assert!(!super::OPEN_TURN_NO_REPLY_NOTE.contains("complete_task"));
    let served = &fake.logged("tools")[0]["tools"];
    assert!(
        !served
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "complete_task"),
        "{served}"
    );
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    let tools = argv.iter().position(|arg| arg == "--tools").unwrap();
    assert_eq!(argv[tools + 1], "Grep");
    let system = task.conversation.system_message.clone().unwrap();
    assert!(!system.contains("complete_task"), "{system}");
    assert!(system.contains("is your final reply"), "{system}");
}

/// Each request Claude Code makes is reported when its response starts,
/// with the conversation's size (its arko tools included) and the bytes
/// per token the previous request's reported input calibrated.
#[tokio::test]
async fn each_request_is_reported_when_its_response_starts() {
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
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let started: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted {
                model,
                subagent,
                sizes,
                bytes_per_token,
                ..
            } => Some((model.clone(), subagent.clone(), *sizes, *bytes_per_token)),
            _ => None,
        })
        .collect();
    assert_eq!(started.len(), 2, "{events:#?}");
    let (model, subagent, first, ratio) = &started[0];
    assert_eq!(model, "claude-opus-5-5");
    assert_eq!(subagent, &None);
    assert_eq!(*ratio, crate::runtime::DEFAULT_BYTES_PER_TOKEN);
    let first = first.unwrap();
    assert_eq!(
        first.system_prompt,
        task.conversation.system_message.as_ref().unwrap().len()
    );
    // complete_task, served as arko's tool.
    assert!(first.tool_definitions > 0);
    assert!(first.user_text > 0);
    let (_, _, second, ratio) = &started[1];
    assert_eq!(
        Some(*ratio),
        crate::runtime::calibrated_bytes_per_token(first.text_bytes(), 1200)
    );
    assert!(second.unwrap().assistant_text > 0);
    // Every call is the conversation's own.
    assert!(
        task.model_call_usage_history
            .iter()
            .all(|call| call.subagent.is_none())
    );
}

/// An agent offered nothing (no tools, and a reply in text ends the task)
/// runs with Claude Code's built-ins off and a system prompt that says
/// nothing about tools, and its reply ends the task.
#[tokio::test]
async fn an_agent_offered_no_tools_is_told_nothing_about_tools() {
    let fake = Fake::new(json!([[{"text": "Hi there.", "input_tokens": 100}]]));
    let task = Task::new(
        replying(&fake, &[]),
        "Hello.".into(),
        fake.workspace(),
        None,
    );
    let (task, events) = start(task).finish().await;

    assert_eq!(completions(&events), ["Hi there."]);
    let argv: Vec<String> = serde_json::from_value(fake.logged("argv")[0]["argv"].clone()).unwrap();
    let tools = argv.iter().position(|arg| arg == "--tools").unwrap();
    assert_eq!(argv[tools + 1], "");
    let served = &fake.logged("tools")[0]["tools"];
    assert_eq!(served, &json!([]));
    let system = task.conversation.system_message.clone().unwrap();
    assert!(!system.contains("## Tool"), "{system}");
    assert!(!system.contains("mcp__"), "{system}");
}

/// After `ask_user`, Claude Code is told to end its turn: that turn's text
/// is not the final reply, and the task waits for the answer. The reply
/// after it ends the task.
#[tokio::test]
async fn the_turn_after_ask_user_is_not_the_final_reply() {
    let fake = Fake::new(json!([
        [
            {"tool": "mcp__arko__ask_user", "input": {"questions": "Which file?"}},
            {"text": "I asked which file."},
        ],
        [{"text": "Edited main.rs."}],
    ]));
    let task = Task::new(
        replying(&fake, &["ask_user"]),
        "Edit a file.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    let seen = run
        .events_until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    assert!(completions(&seen).is_empty(), "{seen:#?}");
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "main.rs".into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, events) = run.finish().await;
    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "Edited main.rs.".into()
        }
    );
    assert_eq!(completions(&events), ["Edited main.rs."]);
}

/// A host that answers the questions `answers` names once the agent asked
/// them. Their note reaches the agent with the next call's result, and they
/// are no longer open.
struct AnswersMidTurn {
    answers: &'static [&'static str],
    asked: std::sync::Mutex<Vec<crate::questions::AskedQuestion>>,
    answered: std::sync::Mutex<Vec<String>>,
}

impl AnswersMidTurn {
    fn new(answers: &'static [&'static str]) -> Arc<Self> {
        Arc::new(Self {
            answers,
            asked: Default::default(),
            answered: Default::default(),
        })
    }
}

impl crate::handover::HandoverHook for AnswersMidTurn {
    fn open_questions(&self, _task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        let answered = self.answered.lock().unwrap();
        Some(
            self.asked
                .lock()
                .unwrap()
                .iter()
                .filter(|question| !answered.contains(&question.id))
                .cloned()
                .collect(),
        )
    }

    fn questions_asked(
        &self,
        _task: &Task,
        asked: &[crate::questions::AskedQuestion],
        _wait: bool,
    ) {
        self.asked.lock().unwrap().extend(asked.iter().cloned());
    }
}

impl crate::runtime::TurnContextSource for AnswersMidTurn {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        let mut answered = self.answered.lock().unwrap();
        if self.asked.lock().unwrap().is_empty() || !answered.is_empty() {
            return None;
        }
        answered.extend(self.answers.iter().map(|id| id.to_string()));
        Some(format!(
            "The user answered {}.",
            crate::questions::id_list(self.answers)
        ))
    }
}

/// The agent asks and waits, but goes on working in the same turn, and the
/// answer reaches it there as a note. When the turn ends, the task doesn't
/// wait for the questions answered meanwhile: a batch answered whole ends
/// the turn as any other turn does, and of one answered in part, only the
/// rest is waited for.
#[tokio::test]
async fn questions_answered_while_the_turn_goes_on_are_not_waited_for() {
    for (answers, waited) in [(&["q1", "q2"][..], &[][..]), (&["q1"][..], &["q2"][..])] {
        let fake = Fake::new(json!([
            [
                {"tool": "mcp__arko__ask_user", "input": {"questions": ["Tabs or spaces?", "Push now?"]}},
                {"tool": "Grep", "input": {"pattern": "a"}},
                {"tool": "Grep", "input": {"pattern": "b"}},
                {"text": "Formatted with spaces."},
            ],
            [{"text": "Pushed."}],
        ]));
        let mut config = replying(&fake, &["ask_user", "Grep"]);
        config.tool_config.async_questions = true;
        let task = Task::new(config, "Format the code.".into(), fake.workspace(), None);
        let host = AnswersMidTurn::new(answers);
        let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
            .with_handover_hook(host.clone())
            .with_turn_context(host.clone());
        let mut run = start_with(task, runtime);
        let seen = run
            .events_until(|event| {
                matches!(
                    event,
                    AgentEvent::UserQuestionsAsked { .. }
                        | AgentEvent::TaskStateChanged(TaskState::Completed { .. })
                )
            })
            .await;
        let note = format!("The user answered {}.", crate::questions::id_list(answers));
        assert!(
            fake.logged("post").iter().any(|post| {
                post["answer"]["hookSpecificOutput"]["additionalContext"]
                    .as_str()
                    .is_some_and(|context| context.contains(&note))
            }),
            "the note reached the agent in the turn"
        );
        let waits = seen
            .iter()
            .filter_map(|event| match event {
                AgentEvent::UserQuestionsAsked {
                    questions,
                    asked,
                    wait: true,
                } => Some((
                    questions.clone(),
                    asked
                        .iter()
                        .map(|question| question.id.clone())
                        .collect::<Vec<_>>(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        if waited.is_empty() {
            assert!(waits.is_empty(), "{waits:?}");
            let (task, events) = run.finish().await;
            assert_eq!(completions(&seen), ["Formatted with spaces."]);
            assert!(
                !seen.iter().chain(&events).any(|event| matches!(
                    event,
                    AgentEvent::TaskStateChanged(TaskState::AwaitingUserResponse { .. })
                )),
                "{seen:#?}"
            );
            assert_eq!(fake.logged("argv").len(), 1);
            assert!(task.questions.open.is_empty() && task.questions.waiting.is_empty());
            continue;
        }
        assert_eq!(
            waits,
            [("q2 Push now?".to_string(), vec!["q2".to_string()])]
        );
        assert!(seen.iter().any(|event| matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingUserResponse { questions })
                if questions == "q2 Push now?"
        )));
        assert!(completions(&seen).is_empty());
        host.answered.lock().unwrap().push("q2".into());
        run.commands
            .send(UserCommand::SubmitUserResponse {
                response: "The user answered q2.".into(),
                ids: Vec::new(),
            })
            .await
            .unwrap();
        let (task, events) = run.finish().await;
        assert_eq!(completions(&events), ["Pushed."]);
        assert!(task.questions.open.is_empty() && task.questions.waiting.is_empty());
    }
}

/// The handover report comes before the final reply: the text of the turn
/// the report was asked for in is the report, and the reply in the fresh
/// conversation ends the task.
#[tokio::test]
async fn a_handover_report_is_not_the_final_reply() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "REPORT: searched a and b; next, c."},
        ],
        [{"text": "Searched a, b and c."}],
    ]));
    let mut config = replying(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(1000));
    let task = Task::new(config, "Search a, b and c.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(
        task.handovers[0].report,
        "REPORT: searched a and b; next, c."
    );
    assert_eq!(completions(&events), ["Searched a, b and c."]);
}

/// A final reply whose request crossed the handover size ends the task:
/// the report asked for meanwhile hadn't reached the model, as a
/// `complete_task` call would end it.
#[tokio::test]
async fn a_reply_before_the_report_request_reaches_the_model_ends_the_task() {
    let fake = Fake::new(json!([[{"text": "The answer.", "input_tokens": 1500}]]));
    let mut config = replying(&fake, &[]);
    config.handover = Some(HandoverConfig::at_input_tokens(1000));
    let task = Task::new(config, "A question.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert_eq!(completions(&events), ["The answer."]);
    assert!(task.handovers.is_empty());
    assert_eq!(fake.logged("user").len(), 1);
}

/// `/compact` and the turn arko interrupted for it are not final replies;
/// the reply after the compaction ends the task. A reply whose request
/// crossed the size ends the task before compacting, as `complete_task`
/// does, and the next run compacts first.
#[tokio::test]
async fn a_compaction_is_not_the_final_reply() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}, "input_tokens": 400},
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 1500},
            {"text": "Not reached: the turn was interrupted."},
        ],
        {"compact": {"pre_tokens": 1520, "post_tokens": 300}},
        [
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 500},
            {"text": "First answer.", "input_tokens": 1600},
        ],
        {"compact": {"pre_tokens": 1610, "post_tokens": 200}},
        [{"text": "Second answer."}],
    ]));
    let mut config = replying(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig {
        compact_in_claude_code: true,
        ..HandoverConfig::at_input_tokens(1000)
    });
    let task = Task::new(config, "First question.".into(), fake.workspace(), None);
    let (mut task, events) = start(task).finish().await;
    assert_eq!(completions(&events), ["First answer."]);
    let compactions = |events: &[AgentEvent]| {
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ConversationCompacted { .. }))
            .count()
    };
    assert_eq!(compactions(&events), 1);
    assert_eq!(user_texts(&fake)[1], "/compact");

    task.prepare_for_resumption_from_completed("Second question.".into())
        .unwrap();
    let (task, events) = start(task).finish().await;
    assert_eq!(completions(&events), ["Second answer."]);
    assert_eq!(compactions(&events), 1);
    let users = user_texts(&fake);
    assert_eq!(users[3], "/compact", "{users:#?}");
    assert!(users[4].starts_with("Second question."), "{users:#?}");
    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "Second answer.".into()
        }
    );
}

/// A tool that says it started, then waits until context arrives for its
/// task.
#[derive(Debug, Default)]
struct WaitsForContext {
    started: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl crate::tool::Tool for WaitsForContext {
    fn definition(&self) -> crate::tool::ToolDefinition {
        crate::tool::ToolDefinition {
            name: "waits_for_context".into(),
            description: "Waits for context".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: crate::tool::ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &crate::tool::TaskContext<'_>,
        _: std::collections::HashMap<String, Value>,
    ) -> Result<String, crate::tool::ToolError> {
        self.started.notify_one();
        match tokio::time::timeout(Duration::from_secs(20), context.context_arrived.arrived()).await
        {
            Ok(()) => Ok("context arrived".into()),
            Err(_) => Ok("no context".into()),
        }
    }
}

/// Context sent while an arko tool call waits ends the wait, and goes to the
/// agent with that call's result.
#[tokio::test]
async fn context_sent_while_an_arko_call_waits_ends_the_wait() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__waits_for_context", "input": {}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["waits_for_context"]),
        "Wait for news.".into(),
        fake.workspace(),
        None,
    );
    let tool = Arc::new(WaitsForContext::default());
    let mut registry = ToolRegistry::new();
    registry.register(tool.clone());
    let (events, commands, handle) = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .run_task(task, Arc::new(registry), None, false, None, None, None);
    let run = Run {
        events,
        commands,
        handle,
    };
    tokio::time::timeout(Duration::from_secs(20), tool.started.notified())
        .await
        .expect("the call runs");
    run.commands
        .send(UserCommand::InjectContext {
            content: "The build finished.".into(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let posts = fake.logged("post");
    assert_eq!(posts[0]["output"], "context arrived");
    let context = posts[0]["answer"]["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(context.contains("The build finished."), "{context}");
}

/// Three items of 6,000 bytes that a source gives one at a time, as a
/// source that doesn't size its updates may.
struct Items {
    left: std::sync::Mutex<Vec<String>>,
}

impl crate::runtime::TurnContextSource for Items {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        let mut left = self.left.lock().unwrap();
        (!left.is_empty()).then(|| left.remove(0))
    }
}

/// Claude Code replaces the text of a hook over 10,000 characters with a
/// preview, so a hook carries at most 9,500 bytes and what doesn't fit
/// comes with the next result, in order.
#[tokio::test]
async fn context_over_the_hooks_limit_comes_with_the_next_results() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "Grep", "input": {"pattern": "b"}},
        {"tool": "Grep", "input": {"pattern": "c"}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let items = Arc::new(Items {
        // The first goes with the task's first input.
        left: std::sync::Mutex::new(
            ["", "a", "b", "c"]
                .map(|letter| letter.repeat(6_000))
                .to_vec(),
        ),
    });
    let runtime =
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_turn_context(items.clone());
    let (task, _) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let hooked: Vec<String> = fake
        .logged("post")
        .iter()
        .filter_map(|post| {
            post["answer"]["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(hooked.len(), 3, "{}", hooked.len());
    for (hook, letter) in hooked.iter().zip(["a", "b", "c"]) {
        assert!(hook.len() <= 9_500, "{}", hook.len());
        assert!(hook.contains(&letter.repeat(6_000)));
    }
    assert!(items.left.lock().unwrap().is_empty());
}

/// The same for context sent to the agent (messages, terminal updates): the
/// injections that don't fit wait for the next result, in order.
#[tokio::test]
async fn injected_context_over_the_hooks_limit_comes_with_the_next_results() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__waits_for_context", "input": {}},
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "Grep", "input": {"pattern": "b"}},
        {"tool": "Grep", "input": {"pattern": "c"}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["waits_for_context", "Grep"]),
        "Wait for news.".into(),
        fake.workspace(),
        None,
    );
    let tool = Arc::new(WaitsForContext::default());
    let mut registry = ToolRegistry::new();
    registry.register(tool.clone());
    let (events, commands, handle) = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .run_task(task, Arc::new(registry), None, false, None, None, None);
    let run = Run {
        events,
        commands,
        handle,
    };
    tokio::time::timeout(Duration::from_secs(20), tool.started.notified())
        .await
        .expect("the call runs");
    for letter in ["a", "b", "c"] {
        run.commands
            .send(UserCommand::InjectContext {
                content: letter.repeat(6_000),
            })
            .await
            .unwrap();
    }
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.pending_context_injections.is_empty());
    let hooked: Vec<String> = fake
        .logged("post")
        .iter()
        .filter_map(|post| {
            post["answer"]["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert!(hooked.iter().all(|hook| hook.len() <= 9_500));
    // Each reached the agent whole, in order, and none with another.
    let order = ["a", "b", "c"].map(|letter| {
        let whole = letter.repeat(6_000);
        let at: Vec<_> = hooked
            .iter()
            .enumerate()
            .filter(|(_, hook)| hook.contains(&whole))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(at.len(), 1, "{letter}: {at:?}");
        at[0]
    });
    assert!(order[0] < order[1] && order[1] < order[2], "{order:?}");
}

/// A source that gives its updates one at a time, the first with the task's
/// first input.
struct Updates(std::sync::Mutex<std::collections::VecDeque<crate::runtime::TurnContext>>);

impl crate::runtime::TurnContextSource for Updates {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        None
    }

    fn context_update(&self, _task_id: uuid::Uuid) -> Option<crate::runtime::TurnContext> {
        self.0.lock().unwrap().pop_front()
    }
}

/// A runtime whose source gives `updates` after the first input's, each
/// at a hook once the one before has gone, and saves snapshots in `recorder`.
fn with_updates(
    recorder: &Arc<Recorder>,
    updates: Vec<crate::runtime::TurnContext>,
) -> AgentRuntime {
    let mut all = std::collections::VecDeque::from(updates);
    all.push_front(Default::default());
    recording(recorder).with_turn_context(Arc::new(Updates(std::sync::Mutex::new(all))))
}

/// A 16 KiB note of numbered lines, longer than a hook holds.
fn long_note() -> String {
    let mut note = String::new();
    let mut line = 0;
    while note.len() < 16 * 1024 {
        line += 1;
        note.push_str(&format!("Line {line:04} of the long note.\n"));
    }
    note.trim_end().to_string()
}

fn noted(note: &str, typed: &str) -> crate::runtime::TurnContext {
    crate::runtime::TurnContext {
        content: note.into(),
        user_content: typed.into(),
        marks: [("notes".to_string(), 3)].into(),
        ..Default::default()
    }
}

const GOES_ON: &str = "\n(This update goes on in the next one.)";
const CONTINUED: &str = "(Continued from the previous update.)\n";

/// The PostToolUse hooks' context, in order.
fn hooked(fake: &Fake) -> Vec<String> {
    fake.logged("post")
        .iter()
        .filter_map(|post| {
            post["answer"]["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .map(str::to_string)
        })
        .collect()
}

/// The context updates `task` holds, recorded and then pending, as one text
/// with each split item's markers taken out.
fn held(task: &Task) -> String {
    let heading = "# Workflow context update\n\n";
    texts(task)
        .into_iter()
        .filter_map(|text| text.strip_prefix(heading).map(str::to_string))
        .chain(
            task.pending_context_injections
                .iter()
                .map(|item| crate::runtime::hook_budget::shown_from(item, item.given)),
        )
        .collect::<Vec<_>>()
        .join("\n\n")
        .replace(&format!("{GOES_ON}\n\n{CONTINUED}"), "")
}

/// The latest snapshot `pick` accepts, once the run has saved one.
async fn saved_once(recorder: &Recorder, pick: impl Fn(&Task) -> bool) -> Task {
    for _ in 0..200 {
        if let Some((task, _)) = recorder
            .snapshots()
            .into_iter()
            .rev()
            .find(|(task, _)| pick(task))
        {
            return task;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let pending = recorder
        .snapshots()
        .iter()
        .map(|(task, _)| {
            task.pending_context_injections
                .iter()
                .map(|item| item.text.len())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    panic!("no such snapshot, pending: {pending:?}");
}

async fn until_paused(run: &mut Run) {
    run.events_until(|event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
    })
    .await;
}

/// Claude Code exits after the hook with a long note's first part, before
/// it reports the call's result. The task still holds the whole note
/// pending, and the next input carries it once, whole, without markers.
#[tokio::test]
async fn a_part_never_acknowledged_leaves_its_note_whole_for_the_next_input() {
    let fake = Fake::new(json!([
        [{"tool": "Grep", "input": {"pattern": "a"}, "crash": "post"}],
        [complete("Done")],
    ]));
    let note = long_note();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let runtime = with_updates(&recorder, vec![noted(&note, "")]);
    let mut run = start_with(task, runtime);
    until_paused(&mut run).await;
    let hooks = hooked(&fake);
    assert!(hooks[0].len() <= 9_500 && hooks[0].ends_with(GOES_ON));
    let saved = recorder.latest(|_| true);
    assert_eq!(
        saved.pending_context_injections,
        std::slice::from_ref(&note)
    );
    assert_eq!(saved.context_marks.get("notes"), Some(&3));
    run.commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let input = fake.logged("user")[1]["text"].as_str().unwrap().to_string();
    assert_eq!(input.matches(&note).count(), 1);
    assert!(!input.contains(GOES_ON.trim()) && !input.contains(CONTINUED.trim()));
    assert!(task.pending_context_injections.is_empty());
    assert_eq!(texts(&task).concat().matches(&note).count(), 1);
}

/// Once the first part's call is reported, the conversation holds the part
/// with its marker and the queue starts with the rest, in one step: every
/// snapshot holds the note once or not at all, and never its mark without
/// it. Claude Code exiting with the rest in flight leaves the rest, and the
/// typed note sent with its hook, pending in order for the next input.
#[tokio::test]
async fn an_acknowledged_part_leaves_its_rest_and_the_note_behind_it_pending() {
    let fake = Fake::new(json!([
        [
            {"tool": "Grep", "input": {"pattern": "a"}},
            {"tool": "Grep", "input": {"pattern": "b"}, "crash": "post"},
        ],
        [complete("Done")],
    ]));
    let note = long_note();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let runtime = with_updates(&recorder, vec![noted(&note, TYPED_NOTE)]);
    let mut run = start_with(task, runtime);
    until_paused(&mut run).await;
    // The session's end puts the note in the queue.
    let saved = saved_once(&recorder, |task| task.pending_context_injections.len() == 2).await;
    let snapshots = recorder.snapshots();
    for (task, _) in &snapshots {
        let held = held(task);
        if task.context_marks.contains_key("notes") {
            assert_eq!(held, format!("{note}\n\n{TYPED_NOTE}"));
        } else {
            assert!(held.is_empty() || held == note, "{held}");
        }
    }
    // The first part's record and the rest pending.
    assert!(snapshots.iter().any(|(task, _)| {
        task.pending_context_injections
            .first()
            .is_some_and(|rest| rest.given > 0)
            && texts(task).iter().any(|text| text.ends_with(GOES_ON))
    }));
    assert_eq!(saved.pending_context_injections.len(), 2);
    assert!(saved.pending_context_injections[0].given > 0);
    assert_eq!(saved.pending_context_injections[1], TYPED_NOTE);
    assert_eq!(saved.context_marks.get("notes"), Some(&3));
    assert_eq!(fake.logged("midturn").len(), 1, "sent with the rest's hook");
    run.commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    let input = fake.logged("user")[1]["text"].as_str().unwrap().to_string();
    let rest = input.find(CONTINUED).unwrap();
    assert!(rest < input.find(TYPED_NOTE).unwrap());
    assert_eq!(held(&task), format!("{note}\n\n{TYPED_NOTE}"));
}

/// Two calls that run side by side get a hook each. The later call's
/// result comes first, and the hooks' texts join the conversation in the
/// order the hooks were given, after both results.
#[tokio::test]
async fn hooks_of_parallel_calls_reported_out_of_order_keep_their_order() {
    let fake = Fake::new(json!([[
        {"parallel": [
            {"tool": "Grep", "input": {"pattern": "a"}},
            {"tool": "Grep", "input": {"pattern": "b"}},
        ], "report": [1, 0]},
        complete("Done"),
    ]]));
    let (a, b) = ("a".repeat(6_000), "b".repeat(6_000));
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let runtime = with_updates(&recorder, vec![a.clone().into(), b.clone().into()]);
    let (task, _) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let hooks = hooked(&fake);
    assert!(hooks[0].contains(&a) && hooks[1].contains(&b));
    let texts = texts(&task);
    let at = |text: &str| texts.iter().position(|held| held.contains(text)).unwrap();
    assert_eq!(at(&b), at(&a) + 1);
    // Both calls' results come first, the later call's before.
    let messages = &task.conversation.messages;
    let results = |index: usize| {
        messages[index]
            .content
            .iter()
            .filter_map(|content| content.as_tool_result())
            .map(|result| result.call_id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        [results(at(&a) - 2), results(at(&a) - 1)].concat(),
        [
            Some("toolu_p0_1".to_string()),
            Some("toolu_p0_0".to_string())
        ]
    );
    assert!(task.pending_context_injections.is_empty());
}

/// A typed note behind a long note's rest isn't sent as the user's message
/// until a hook carries the rest, and it comes after the rest.
#[tokio::test]
async fn a_typed_note_waits_until_a_hook_carries_what_is_ahead_of_it() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "Grep", "input": {"pattern": "b"}},
        complete("Done"),
    ]]));
    let note = long_note();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let runtime = with_updates(&recorder, vec![noted(&note, TYPED_NOTE)]);
    let (task, _) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // Sent with the second call's hook: the fake logs it as it reads it,
    // which can be before it reads the hook's answer.
    let events = fake
        .log()
        .iter()
        .map(|entry| entry["event"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    let at = |event: &str| {
        events
            .iter()
            .enumerate()
            .filter(|(_, logged)| *logged == event)
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    };
    let sent = at("midturn");
    assert_eq!(sent.len(), 1);
    assert!(sent[0] > at("pre")[1], "{events:?}");
    let hooks = hooked(&fake);
    assert!(hooks[1].starts_with(&format!("# Workflow context update\n\n{CONTINUED}")));
    assert!(!hooks.concat().contains("Use tabs."));
    assert_eq!(held(&task), format!("{note}\n\n{TYPED_NOTE}"));
    assert_eq!(task.context_marks.get("notes"), Some(&3));
}

/// A message given with an approval waits in the queue, so it comes after
/// a long note's rest that was waiting before it.
#[tokio::test]
async fn an_approval_message_comes_after_the_rest_ahead_of_it() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        {"tool": "Bash", "input": {"command": "cargo test"}},
        complete("Tested"),
    ]]));
    let mut config = config(&fake, &["Grep", "Bash"]);
    config.tool_config.tool_specific_configs.insert(
        "Bash".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let note = long_note();
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(config, "Test.".into(), fake.workspace(), None);
    let mut run = start_with(task, with_updates(&recorder, vec![noted(&note, "")]));
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) = events.last() else {
        panic!("{events:#?}");
    };
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: None,
            message_to_agent: Some("Only the unit tests.".into()),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let hooks = hooked(&fake);
    assert!(hooks.iter().all(|hook| hook.len() <= 9_500));
    let rest = hooks[1].find(CONTINUED).unwrap();
    let message = hooks[1]
        .find("Message from user on approval:\nOnly the unit tests.")
        .unwrap();
    assert!(rest < message, "{}", hooks[1]);
    assert!(held(&task).starts_with(&note));
}

/// The host's output for a built-in, given on its approval, is the call's
/// result. One longer than a hook holds is cut in the permission answer,
/// and its rest comes first in the next hook.
#[tokio::test]
async fn a_long_output_for_a_built_in_is_cut_and_its_rest_comes_next() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "cargo test"}},
        {"tool": "Grep", "input": {"pattern": "a"}},
        complete("Tested"),
    ]]));
    let mut config = config(&fake, &["Grep", "Bash"]);
    config.tool_config.tool_specific_configs.insert(
        "Bash".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let output = long_note().replace("long note", "test run's output");
    let task = Task::new(config, "Test.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) = events.last() else {
        panic!("{events:#?}");
    };
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: Some(output.clone()),
            message_to_agent: None,
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let answer = fake.logged("permission")[0]["answer"]["message"]
        .as_str()
        .unwrap()
        .to_string();
    use crate::runtime::hook_budget::{DENIAL_GOES_ON, DENIAL_REST};
    assert!(
        answer.len() <= 9_500 && answer.ends_with(DENIAL_GOES_ON),
        "{}",
        answer.len()
    );
    // The rest, longer than a hook holds itself, takes the next two, under
    // its own words first.
    let hooks = hooked(&fake);
    assert_eq!(hooks.len(), 2);
    let mut parts = Vec::new();
    for (index, hook) in hooks.iter().enumerate() {
        let rest = hook.strip_prefix("# Workflow context update\n\n").unwrap();
        let start = if index == 0 { DENIAL_REST } else { CONTINUED };
        assert!(hook.len() <= 9_500 && rest.starts_with(start), "{index}");
        parts.push(rest.to_string());
    }
    parts[0] = parts[0].strip_prefix(DENIAL_REST).unwrap().to_string();
    let rejoined = format!(
        "{}{}",
        answer.strip_suffix(DENIAL_GOES_ON).unwrap(),
        parts
            .join("\n\n")
            .replace(&format!("{GOES_ON}\n\n{CONTINUED}"), "")
    );
    assert!(
        rejoined == output,
        "{} of {} bytes",
        rejoined.len(),
        output.len()
    );
    assert!(task.pending_context_injections.is_empty());
}

/// The CLI completes the user's message before it reports the call's
/// result, so the note is done as its hook is acknowledged, and joins the
/// same record. (The order the other way round, a verdict after the result,
/// is `a_note_the_user_typed_goes_as_their_message_and_the_mail_in_the_hook`.)
#[tokio::test]
async fn a_note_completed_before_its_result_joins_its_hooks_record() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}, "complete_notes": true},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let recorder = Arc::new(Recorder::default());
    let runtime = with_updates(
        &recorder,
        vec![noted("A message: skip vendor/.", TYPED_NOTE)],
    );
    let (task, events) = start_with(task, runtime).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(fake.logged("midturn").len(), 1);
    let updates = texts(&task)
        .into_iter()
        .filter(|text| text.starts_with("# Workflow context update"))
        .collect::<Vec<_>>();
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert!(updates[0].ends_with(&format!("skip vendor/.\n\n{TYPED_NOTE}")));
    assert!(task.pending_context_injections.is_empty());
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ContextInjected { .. }))
            .count(),
        1
    );
}

/// An update's content and the part the user typed each carry their own
/// ids: the task records each with its item, and each is reported as its
/// item reaches the model.
#[tokio::test]
async fn an_updates_ids_go_with_their_own_items() {
    let fake = Fake::new(json!([[
        {"tool": "Grep", "input": {"pattern": "a"}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["Grep"]),
        "Search.".into(),
        fake.workspace(),
        None,
    );
    let recorder = Arc::new(Recorder::default());
    let update = crate::runtime::TurnContext {
        content: "A message: skip vendor/.".into(),
        user_content: TYPED_NOTE.into(),
        ids: vec!["mail:1".into()],
        user_ids: vec!["note:2".into()],
        ..Default::default()
    };
    let (task, events) = start_with(task, with_updates(&recorder, vec![update]))
        .finish()
        .await;
    let injected: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ContextInjected { ids, .. } => Some(ids.clone()),
            _ => None,
        })
        .filter(|ids| !ids.is_empty())
        .collect();
    assert_eq!(
        injected,
        [vec!["mail:1".to_string()], vec!["note:2".to_string()]]
    );
    // Each record holds its message's items with their ids.
    let texts = texts(&task);
    let recorded: Vec<(String, Vec<String>)> = task
        .context_records
        .iter()
        .flat_map(|record| {
            record
                .parts
                .iter()
                .map(|part| (texts[record.message].clone(), part.host_ids.clone()))
        })
        .collect();
    assert!(
        recorded
            .iter()
            .any(|(text, ids)| text.contains("skip vendor/") && ids == &["mail:1"])
    );
    assert!(
        recorded
            .iter()
            .any(|(text, ids)| text.contains("Use tabs.") && ids == &["note:2"])
    );
}

/// A snapshot sink that takes a while over a durable save, and notes, at
/// each durable save holding a denial's rest, how many permission answers
/// the fake CLI had received by then.
struct SlowSaves {
    log: PathBuf,
    answered: std::sync::Mutex<Vec<usize>>,
}

#[async_trait::async_trait]
impl crate::snapshot::TaskSnapshotSink for SlowSaves {
    async fn task_changed(&self, task: &Task, durable: bool) -> Result<(), String> {
        let rest = task
            .pending_context_injections
            .iter()
            .any(|item| matches!(item.kind, crate::task::PendingKind::DenialRest { .. }));
        if durable && rest {
            tokio::time::sleep(Duration::from_millis(300)).await;
            let answers = std::fs::read_to_string(&self.log)
                .unwrap_or_default()
                .lines()
                .filter(|line| line.contains(r#""event": "permission""#))
                .count();
            self.answered.lock().unwrap().push(answers);
        }
        Ok(())
    }
}

/// The host's output for a built-in, too long for its answer, can't be
/// given again, so the task is saved with its rest before the answer goes
/// out: no answer has reached the CLI when that save ends.
#[tokio::test]
async fn a_denials_rest_is_saved_before_the_denial_goes_out() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "cargo test"}},
        {"tool": "Grep", "input": {"pattern": "a"}},
        complete("Tested"),
    ]]));
    let mut config = config(&fake, &["Grep", "Bash"]);
    config.tool_config.tool_specific_configs.insert(
        "Bash".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let sink = Arc::new(SlowSaves {
        log: fake.dir.join("log.jsonl"),
        answered: Default::default(),
    });
    let task = Task::new(config, "Test.".into(), fake.workspace(), None);
    let runtime =
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_task_snapshots(sink.clone());
    let mut run = start_with(task, runtime);
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) = events.last() else {
        panic!("{events:#?}");
    };
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: Some(long_note()),
            message_to_agent: None,
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let answered = sink.answered.lock().unwrap().clone();
    assert_eq!(answered.first(), Some(&0), "{answered:?}");
}

/// A long note's first part goes in a hook, and its rest with the next
/// input after the open turn, as a context message of its own after the
/// runtime's note. The next turn goes over the input limit, so the step is
/// left out and the report asked for: the context messages the task
/// recorded stay, their records following them, so every record points at
/// the context it holds.
#[tokio::test]
async fn a_step_left_out_keeps_the_context_the_task_recorded() {
    let fake = Fake::new(json!([
        [{"tool": "Grep", "input": {"pattern": "a"}}, {"text": "Reading."}],
        [
            {"tool": "Grep", "input": {"pattern": "b"}, "input_tokens": 400, "output": "b is here"},
            {"tool": "Grep", "input": {"pattern": "c"}, "input_tokens": 500, "output": "HUGE RESULT"},
            too_long(),
        ],
        [{"text": "REPORT: b is here."}],
        [complete("Found them")],
    ]));
    let mut config = config(&fake, &["Grep"]);
    config.handover = Some(HandoverConfig::at_input_tokens(100_000));
    let task = Task::new(config, "Search.".into(), fake.workspace(), None);
    let recorder = Arc::new(Recorder::default());
    let note = long_note();
    let updates = vec![
        noted(&note, ""),
        // (With the next input.)
        Default::default(),
        "Mail M for the agent.".to_string().into(),
        "Mail P for the agent.".to_string().into(),
    ];
    let (task, _) = start_with(task, with_updates(&recorder, updates))
        .finish()
        .await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(task.handovers.len(), 1);
    // The rest went as a message of its own after the runtime's note.
    let before =
        recorder.latest(|task| task.handover_requested.is_none() && task.handovers.is_empty());
    let texts_before = texts(&before);
    let rest = texts_before
        .iter()
        .position(|text| text.starts_with(&format!("# Workflow context update\n\n{CONTINUED}")))
        .expect("the rest's message");
    assert!(texts_before[rest - 1].starts_with("Your turn ended"));
    assert!(before.runtime_messages.contains(&(rest - 1)));
    assert!(!before.runtime_messages.contains(&rest));
    // With the step left out, before the fresh conversation: every record
    // points at a context message, and they hold all the context.
    let left_out = recorder.latest(|task| task.handover_requested.is_some());
    let texts = texts(&left_out);
    let mut held = Vec::new();
    for record in &left_out.context_records {
        let text = &texts[record.message];
        if record.message > 0 {
            assert!(text.starts_with("# Workflow context update"), "{text}");
            held.push(
                text.strip_prefix("# Workflow context update\n\n")
                    .unwrap()
                    .to_string(),
            );
        }
    }
    let held = held
        .join("\n\n")
        .replace(&format!("{GOES_ON}\n\n{CONTINUED}"), "");
    for expected in [
        note.as_str(),
        "Mail M for the agent.",
        "Mail P for the agent.",
    ] {
        assert!(
            held.contains(expected),
            "{}",
            &expected[..20.min(expected.len())]
        );
    }
    // The fresh conversation holds nothing of the old one's context.
    assert!(
        task.context_records
            .iter()
            .all(|record| record.message < task.conversation.messages.len())
    );
}

/// A tool that returns whether its call says the task runs in Claude Code.
#[derive(Debug)]
struct SaysRuntime;

#[async_trait::async_trait]
impl crate::tool::Tool for SaysRuntime {
    fn definition(&self) -> crate::tool::ToolDefinition {
        crate::tool::ToolDefinition {
            name: "says_runtime".into(),
            description: "Says where the task runs".into(),
            ..WaitsForContext::default().definition()
        }
    }

    async fn execute(
        &self,
        context: &crate::tool::TaskContext<'_>,
        _: std::collections::HashMap<String, Value>,
    ) -> Result<String, crate::tool::ToolError> {
        Ok(context.runs_in_claude_code.to_string())
    }
}

/// The arko tools of a Claude Code agent see that its task runs in Claude
/// Code.
#[tokio::test]
async fn an_arko_call_of_a_claude_code_agent_sees_that_it_runs_in_claude_code() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__says_runtime", "input": {}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["says_runtime"]),
        "Say where you run.".into(),
        fake.workspace(),
        None,
    );
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(SaysRuntime));
    let (events, commands, handle) = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .run_task(task, Arc::new(registry), None, false, None, None, None);
    let run = Run {
        events,
        commands,
        handle,
    };
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(fake.logged("post")[0]["output"], "true");
}

/// An arko tool that runs until the test releases it.
#[derive(Debug, Default)]
struct Slow {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl crate::tool::Tool for Slow {
    fn definition(&self) -> crate::tool::ToolDefinition {
        crate::tool::ToolDefinition {
            name: "slow".into(),
            description: "Runs until released".into(),
            ..WaitsForContext::default().definition()
        }
    }

    fn background_mode(&self) -> crate::tool::ToolBackground {
        crate::tool::ToolBackground::Detach
    }

    async fn execute(
        &self,
        _: &crate::tool::TaskContext<'_>,
        _: std::collections::HashMap<String, Value>,
    ) -> Result<String, crate::tool::ToolError> {
        self.started.notify_one();
        self.release.notified().await;
        Ok("slow done".into())
    }
}

/// A Claude Code agent's arko call that moves answers Claude Code's MCP
/// request at once with the moved result, and its real result reaches the
/// agent as a note with the next call's result.
#[tokio::test]
async fn a_moved_arko_call_answers_claude_code_at_once() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__slow", "input": {}},
        {"tool": "mcp__arko__waits_for_context", "input": {}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["slow", "waits_for_context"]),
        "Index the repository.".into(),
        fake.workspace(),
        None,
    );
    let task_id = task.id;
    let jobs = crate::terminal::TerminalManager::new(crate::terminal::TerminalConfig::new(
        fake.dir.join("jobs"),
    ))
    .unwrap();
    let slow = Arc::new(Slow::default());
    let waits = Arc::new(WaitsForContext::default());
    let mut registry = ToolRegistry::new();
    registry.register(slow.clone());
    registry.register(waits.clone());
    let (events, commands, handle) = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_background_jobs(jobs.clone())
        .run_task(task, Arc::new(registry), None, false, None, None, None);
    let run = Run {
        events,
        commands,
        handle,
    };
    tokio::time::timeout(Duration::from_secs(20), slow.started.notified())
        .await
        .expect("the call runs");
    let calls = jobs.foreground_calls();
    assert_eq!(
        calls
            .iter()
            .map(|call| (call.owner, call.tool_name.as_str()))
            .collect::<Vec<_>>(),
        [(task_id, "slow")]
    );
    assert_eq!(jobs.move_to_background(task_id, None), 1);
    tokio::time::timeout(Duration::from_secs(20), waits.started.notified())
        .await
        .expect("Claude Code went on");
    let job = jobs.jobs()[0].job_id;
    slow.release.notify_one();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let posts = fake.logged("post");
    let moved = posts[0]["output"].as_str().unwrap();
    assert!(
        moved.starts_with(&format!("{}{job}, ", crate::terminal::IN_THE_BACKGROUND)),
        "{moved}"
    );
    assert_eq!(posts[0]["is_error"], false);
    assert_eq!(posts[1]["output"], "context arrived");
    let context = posts[1]["answer"]["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(
        context.contains(&format!(
            "Job {job} ended. It ran your slow call, which moved to the background"
        )) && context.contains("slow done"),
        "{context}"
    );
}

/// A monitor a Claude Code agent starts through arko's tool host alerts it
/// as moved calls send notes: its alert ends the agent's wait, and goes to
/// the CLI with that call's PostToolUse hook.
#[tokio::test]
async fn a_monitor_started_in_claude_code_alerts_the_agent() {
    let python = std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let command =
        format!("{python} -c \"print('server ready', flush=True); import time; time.sleep(30)\"");
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__monitor_start", "input": {"command": command, "pattern": "ready"}},
        {"tool": "mcp__arko__waits_for_context", "input": {}},
        complete("Done"),
    ]]));
    let task = Task::new(
        config(&fake, &["monitor_start", "waits_for_context"]),
        "Start the dev server.".into(),
        fake.workspace(),
        None,
    );
    let jobs = crate::terminal::TerminalManager::new(crate::terminal::TerminalConfig::new(
        fake.dir.join("jobs"),
    ))
    .unwrap();
    jobs.set_monitor_limits(crate::terminal::MonitorLimits {
        gather: Duration::from_millis(100),
        start_wait: Duration::ZERO,
        ..crate::terminal::MonitorLimits::default()
    })
    .unwrap();
    let waits = Arc::new(WaitsForContext::default());
    let mut registry = ToolRegistry::new();
    jobs.register_tools(&mut registry);
    registry.register(waits.clone());
    let (events, commands, handle) = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_background_jobs(jobs.clone())
        .run_task(task, Arc::new(registry), None, false, None, None, None);
    let run = Run {
        events,
        commands,
        handle,
    };
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let posts = fake.logged("post");
    let started = posts[0]["output"].as_str().unwrap();
    assert!(started.starts_with("Monitor "), "{started}");
    let id = started.split(' ').nth(1).unwrap();
    assert_eq!(posts[1]["output"], "context arrived");
    let context = posts[1]["answer"]["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(
        context.contains(&format!(
            "Monitor alert. Job {id} (`{command}`, lines that match `ready`) printed 1 matching line since "
        )) && context.contains("\n  server ready\n"),
        "{context}"
    );
    jobs.shutdown().await;
}

/// A host's record of questions with an allocator another run already took
/// numbers from: the next ask is numbered from 43.
#[derive(Default)]
struct SharedNumbers {
    asked: std::sync::Mutex<Vec<crate::questions::AskedQuestion>>,
}

impl crate::handover::HandoverHook for SharedNumbers {
    fn open_questions(&self, _task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        Some(self.asked.lock().unwrap().clone())
    }

    fn question_numbers(&self, _task: &Task, _count: u32) -> Option<u32> {
        Some(43 + self.asked.lock().unwrap().len() as u32)
    }

    fn questions_asked(
        &self,
        _task: &Task,
        asked: &[crate::questions::AskedQuestion],
        _wait: bool,
    ) {
        self.asked.lock().unwrap().extend(asked.iter().cloned());
    }
}

/// An `ask_user` call whose input the runtime can't read, with the result
/// the host supplied on approval, asks that result as one question. It is
/// numbered by the host's allocator and recorded with the host before the
/// question's event comes, as any ask is, and it waits, as such a call
/// always did. The compaction right after it restates it.
#[tokio::test]
async fn an_ask_with_a_supplied_result_is_numbered_and_recorded_by_the_host() {
    let fake = Fake::new(json!([
        [
            {"tool": "mcp__arko__ask_user", "input": {"questions": 42}, "input_tokens": 400},
            {"text": "Waiting for the answer.", "input_tokens": 1500},
        ],
        {"compact": {"pre_tokens": 1510, "post_tokens": 200}},
        [complete("Answered")],
    ]));
    let mut config = compacting(&fake, &["ask_user"], 1000);
    config.tool_config.async_questions = true;
    config.tool_config.tool_specific_configs.insert(
        "ask_user".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let task = Task::new(config, "Pick a branch.".into(), fake.workspace(), None);
    let host = Arc::new(SharedNumbers::default());
    let mut run = start_with(
        task,
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_handover_hook(host.clone()),
    );
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) = events.last() else {
        panic!("{events:#?}");
    };
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: Some("Which branch?".into()),
            message_to_agent: None,
        })
        .await
        .unwrap();
    let events = run
        .events_until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    // Recorded with the host before the event came.
    let recorded = host.asked.lock().unwrap().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].id, "q43");
    assert_eq!(recorded[0].question.question, "Which branch?");
    assert!(matches!(
        events.last(),
        Some(AgentEvent::UserQuestionsAsked { asked, wait: true, .. })
            if asked.len() == 1 && asked[0].id == "q43"
    ));
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "main".into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let users = user_texts(&fake);
    assert!(users.iter().any(|text| text == "/compact"), "{users:#?}");
    let last = users.last().unwrap();
    assert!(
        last.contains("# Your open questions") && last.contains("q43 Which branch?"),
        "{last}"
    );
}

/// The MCP bridge reaches a Claude Code agent as `mcp__arko__mcp`, served
/// with its schema, and its calls reach the session's servers. A call of a
/// tool the server marks read-only is recorded without waiting for the
/// save, and a call that writes waits.
#[tokio::test]
async fn a_claude_code_agent_reaches_mcp_servers_through_the_bridge() {
    let python = std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let stub = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp_stub_server.py");
    let hub = crate::mcp::McpHub::new(vec![
        crate::mcp::McpServerConfig::new("stub", python).with_args([stub.display().to_string()]),
    ]);
    hub.start_all(Duration::from_secs(30)).await;
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__mcp", "input": {"action": "servers"}, "id": "toolu_servers"},
        {
            "tool": "mcp__arko__mcp",
            "input": {"action": "call", "server": "stub", "tool": "echo", "arguments": {"text": "hi"}},
            "id": "toolu_echo",
        },
        {
            "tool": "mcp__arko__mcp",
            "input": {"action": "call", "server": "stub", "tool": "write_note", "arguments": {"text": "n"}},
            "id": "toolu_note",
        },
        complete("done"),
    ]]));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(crate::mcp::McpBridgeTool::worker(
        hub.clone(),
        Arc::new(crate::mcp::AllServers),
    )));
    let recorder = Arc::new(Recorder::default());
    let task = Task::new(
        config(&fake, &["mcp"]),
        "Use the server.".into(),
        fake.workspace(),
        None,
    );
    let (events, commands, handle) =
        recording(&recorder).run_task(task, Arc::new(registry), None, false, None, None, None);
    let (task, _) = Run {
        events,
        commands,
        handle,
    }
    .finish()
    .await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );

    let served = &fake.logged("tools")[0]["tools"];
    let mcp = served
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "mcp")
        .expect("the bridge is served");
    assert_eq!(
        mcp["inputSchema"]["properties"]["action"]["enum"],
        json!(["servers", "tools", "call", "start"])
    );
    let posts = fake.logged("post");
    assert_eq!(posts[0]["tool"], "mcp__arko__mcp");
    assert_eq!(posts[0]["output"], "stub  running, 2 tools");
    assert_eq!(posts[1]["output"], "echo: hi");
    assert_eq!(posts[2]["output"], "echo: n");
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
    assert!(!first_report("toolu_servers"));
    assert!(!first_report("toolu_echo"));
    assert!(first_report("toolu_note"));
    hub.shutdown().await;
}
