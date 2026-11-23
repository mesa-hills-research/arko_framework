//! Runs the Antigravity runtime against a scripted CLI and its MCP process.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arko_ai::{BillingKind, Image, LLMClient, Model, SubscriptionPlan, antigravity::HelperCommand};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::{AgentRuntime, UserCommand, event::AgentEvent};
use crate::{
    AgentConfig, AgentLLMConfig, AgentToolConfig, AntigravitySettings, ArkoAgentsError, Task,
    TaskState, ToolRegistry,
    agent_config::{PathSafetyPolicy, PerToolConfigOverride},
    handover::HandoverConfig,
};

// The app's end-to-end test uses the Rust helper. This one connects the fake
// CLI to the same host-side relay without building a separate executable.
const HELPER: &str = r#"
import json, os, socket, sys, threading

verb, path = sys.argv[1:]
if verb != 'relay':
    with open(path, 'a') as marker:
        marker.write(verb + '\n')
    if verb == 'gate':
        print('{"decision":"deny"}')
    sys.exit(0)

host, port = json.load(open(path))['address'].rsplit(':', 1)
with socket.create_connection((host, int(port))) as conn:
    hello = {'token': os.environ['ARKO_AGY_RELAY_TOKEN']}
    conn.sendall((json.dumps(hello) + '\n').encode())

    def send():
        for line in sys.stdin.buffer:
            conn.sendall(line)
            marker = os.environ.get('FAKE_AGY_MCP_CANCEL_MARKER')
            if marker and json.loads(line).get('method') == 'notifications/cancelled':
                with open(marker, 'w'):
                    pass
        conn.shutdown(socket.SHUT_WR)

    threading.Thread(target=send, daemon=True).start()
    while data := conn.recv(65536):
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
"#;

struct Fake {
    dir: PathBuf,
}

impl Fake {
    fn new(script: Value) -> Self {
        Self::in_dir(std::env::temp_dir(), script)
    }

    fn in_dir(parent: PathBuf, script: Value) -> Self {
        let dir = parent.join(format!("arko-fake-agy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("work")).unwrap();
        std::fs::write(dir.join("script.json"), script.to_string()).unwrap();
        Self { dir }
    }

    fn task(&self, tools: AgentToolConfig) -> Task {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_agy_agent.py");
        let python = std::env::var("ARKO_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
        let config = AgentConfig::builder(
            "agy-agent".into(),
            "Antigravity agent".into(),
            AgentLLMConfig {
                model: Model::antigravity(Model::google_gemini_4_neon()),
                generation_config: Default::default(),
            },
        )
        .role_description("You check examples.".into())
        .instructions("Read the example. Report what you found.".into())
        .tool_config(tools)
        .antigravity(AntigravitySettings {
            cli_path: Some(fixture),
            profile_root: Some(self.dir.join("profiles")),
            helper_command: Some(HelperCommand {
                program: python.into(),
                args: vec!["-c".into(), HELPER.into()],
            }),
            env: [
                ("FAKE_AGY_SCRIPT", "script.json"),
                ("FAKE_AGY_LOG", "wire.jsonl"),
                ("FAKE_AGY_STATE", "state"),
            ]
            .into_iter()
            .map(|(key, file)| (key.into(), self.dir.join(file).display().to_string()))
            .collect(),
            thought_level: Some("xhigh".into()),
        })
        .build()
        .unwrap();
        Task::new(
            config,
            "Do the example task.".into(),
            self.dir.join("work"),
            None,
        )
    }

    fn script(&self, script: &Value) {
        std::fs::write(self.dir.join("script.json"), script.to_string()).unwrap();
    }

    fn hold_usage(&self, task: &mut Task) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let proxy = self.dir.join("agy-with-held-usage");
        let started = self.dir.join("usage-started");
        let release = self.dir.join("usage-release");
        std::fs::write(
            &proxy,
            r#"#!/usr/bin/env python3
import os, pathlib, sys, time
original = os.environ['FAKE_AGY_ORIGINAL']
if sys.argv[1] == '-p':
    pathlib.Path(os.environ['FAKE_AGY_USAGE_STARTED']).touch()
    while not pathlib.Path(os.environ['FAKE_AGY_USAGE_RELEASE']).exists():
        time.sleep(0.01)
os.execv(original, [original, *sys.argv[1:]])
"#,
        )
        .unwrap();
        std::fs::set_permissions(&proxy, std::fs::Permissions::from_mode(0o700)).unwrap();
        let settings = task.agent_config.antigravity.as_mut().unwrap();
        let original = settings.cli_path.replace(proxy).unwrap();
        for (key, path) in [
            ("FAKE_AGY_ORIGINAL", &original),
            ("FAKE_AGY_USAGE_STARTED", &started),
            ("FAKE_AGY_USAGE_RELEASE", &release),
        ] {
            settings.env.insert(key.into(), path.display().to_string());
        }
        (started, release)
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.dir.join("wire.jsonl")).unwrap_or_default()
    }

    fn logged(&self, name: &str) -> Vec<Value> {
        self.log()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|entry| entry["event"] == name)
            .collect()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

struct Run {
    events: mpsc::Receiver<AgentEvent>,
    commands: mpsc::Sender<UserCommand>,
    handle: JoinHandle<Result<Task, ArkoAgentsError>>,
}

impl Run {
    fn start(task: Task) -> Self {
        Self::with_runtime(task, AgentRuntime::new(Arc::new(LLMClient::new())))
    }

    fn with_runtime(task: Task, runtime: AgentRuntime) -> Self {
        let (events, commands, handle) = runtime.run_task(
            task,
            Arc::new(ToolRegistry::new()),
            None,
            false,
            None,
            None,
            None,
        );
        Self {
            events,
            commands,
            handle,
        }
    }

    async fn until(&mut self, stop: impl Fn(&AgentEvent) -> bool) -> Vec<AgentEvent> {
        let mut seen = Vec::new();
        loop {
            let next = tokio::time::timeout(Duration::from_secs(15), self.events.recv())
                .await
                .unwrap_or_else(|_| {
                    panic!("No Antigravity event in time. Events so far: {seen:#?}")
                });
            let Some(event) = next else { return seen };
            let done = stop(&event);
            seen.push(event);
            if done {
                return seen;
            }
        }
    }

    async fn finish(mut self) -> (Task, Vec<AgentEvent>) {
        let events = self.until(|_| false).await;
        let task = self.handle.await.unwrap().unwrap();
        (task, events)
    }
}

fn chat() -> AgentToolConfig {
    AgentToolConfig {
        chat: true,
        ..AgentToolConfig::default()
    }
}

fn tools(names: &[&str]) -> AgentToolConfig {
    AgentToolConfig {
        default_auto_approve_tool_use: true,
        available_tool_names: names.iter().map(|name| name.to_string()).collect(),
        ..AgentToolConfig::default()
    }
}

fn complete(summary: &str) -> Value {
    json!({"tool": "complete_task", "input": {"summary": summary}})
}

#[tokio::test]
async fn antigravity_streams_the_reply_and_reports_plan_usage() {
    let fake = Fake::new(json!([[
        {"thought": "Check the example.", "usage": {"inputTokens": 100, "outputTokens": 10, "thoughtTokens": 4}},
        {"wait_for": "note"},
        {"text": "The example works."},
        {"update": {"sessionUpdate": "usage_update", "used": 92, "size": 12345}}
    ]]));
    let mut task = fake.task(chat());
    task.initial_task_images = vec![Image::from_bytes(b"example image", "image/png")];
    let mut run = Run::start(task);
    let mut events = run
        .until(|event| matches!(event, AgentEvent::PlanUsageReported(_)))
        .await;
    run.commands
        .send(UserCommand::InjectContext {
            content: "Continue the example.".into(),
        })
        .await
        .unwrap();
    let (task, rest) = run.finish().await;
    events.extend(rest);
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "The example works."),
        "{}",
        fake.log()
    );
    assert!(events.iter().any(
        |event| matches!(event, AgentEvent::ReasoningChunk(text) if text == "Check the example.")
    ));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::PlanUsageReported(usage)
        if usage.plan == Some(SubscriptionPlan::Antigravity) && !usage.antigravity_groups.is_empty())));
    assert_eq!(task.usage_history.len(), 1);
    assert_eq!(task.usage_history[0].billing, BillingKind::Subscription);
    assert_eq!(task.usage_history[0].billed(), 0.0);
    assert_eq!(task.model_call_usage_history[0].provider, "antigravity");
    assert_eq!(
        task.model_call_usage_history[0].max_input_tokens,
        Some(12345)
    );
    assert!(fake.log().contains("image/png"));
}

#[tokio::test]
async fn antigravity_lets_arko_approve_calls_and_send_messages() {
    let fake = Fake::new(json!([[
        {"tool": "read_file", "input": {"path": "example.txt"}},
        {"tool": "message_user", "input": {"message": "The file was read."}},
        complete("Done")
    ]]));
    let mut cfg = tools(&["read_file", "message_user"]);
    cfg.tool_specific_configs.insert(
        "read_file".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    let task = fake.task(cfg);
    std::fs::write(
        task.current_working_directory.join("example.txt"),
        "File content from Arko.",
    )
    .unwrap();
    let mut run = Run::start(task);
    let events = run
        .until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval {
        tool_call_id,
        tool_name,
        ..
    }) = events.last()
    else {
        panic!("{events:?}")
    };
    assert_eq!(tool_name, "read_file");
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: *tool_call_id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await
        .unwrap();
    let (task, events) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Done"),
        "{}",
        fake.log()
    );
    assert!(events.iter().any(|event| matches!(event, AgentEvent::ToolResult { output, .. } if output.contains("File content from Arko."))));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::MessageToUser { message, .. } if message == "The file was read.")));
    assert!(
        task.conversation
            .messages
            .iter()
            .any(|message| message.content.iter().any(|content| content
                .as_tool_result()
                .is_some_and(|result| result.output.contains("File content from Arko."))))
    );
}

#[tokio::test]
async fn antigravity_keeps_the_workspace_boundary() {
    let fake = Fake::new(json!([[
        {"tool": "read_file", "input": {"path": "../outside.txt"}},
        complete("Done")
    ]]));
    std::fs::write(fake.dir.join("outside.txt"), "Not in the workspace.").unwrap();
    let mut cfg = tools(&["read_file"]);
    cfg.default_out_of_workspace_path_policy = PathSafetyPolicy::AutoReject;
    let (task, events) = Run::start(fake.task(cfg)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::ToolRejected { reason: Some(reason), .. } if reason.contains("outside the workspace"))));
    assert!(!fake.log().contains("Not in the workspace."));
}

#[tokio::test]
async fn antigravity_resumes_its_session_after_a_blocking_question() {
    let fake = Fake::new(json!([
        [{"tool": "ask_user", "input": {"questions": "Which file?"}}],
        [complete("Used the answer")]
    ]));
    let mut run = Run::start(fake.task(tools(&["ask_user"])));
    let events = run
        .until(|event| matches!(event, AgentEvent::UserQuestionsAsked { .. }))
        .await;
    assert!(
        matches!(
            events.last(),
            Some(AgentEvent::UserQuestionsAsked { wait: true, .. })
        ),
        "{events:?}"
    );
    run.commands
        .send(UserCommand::SubmitUserResponse {
            response: "Use the example file.".into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Used the answer"),
        "{}",
        fake.log()
    );
    let prompts = fake.logged("prompt");
    assert_eq!(prompts.len(), 2, "{}", fake.log());
    assert_eq!(prompts[1]["sessionId"], prompts[0]["sessionId"]);
    assert_eq!(fake.logged("session_resume").len(), 1);
    let answer = prompts[1]["text"].as_str().unwrap();
    assert!(answer.contains("Use the example file."), "{answer}");
    assert!(!answer.contains("Do the example task."), "{answer}");
}

#[tokio::test]
async fn antigravity_asks_without_waiting() {
    let fake = Fake::new(json!([[
        {"tool": "ask_user", "input": {"questions": "Which file?", "wait": false}},
        complete("Kept going")
    ]]));
    let mut cfg = tools(&["ask_user"]);
    cfg.async_questions = true;
    let (task, events) = Run::start(fake.task(cfg)).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::UserQuestionsAsked { wait: false, .. }))
    );
}

struct DurableSnapshots(mpsc::UnboundedSender<String>);

#[async_trait::async_trait]
impl crate::snapshot::TaskSnapshotSink for DurableSnapshots {
    async fn task_changed(&self, task: &Task, durable: bool) -> Result<(), String> {
        if durable {
            self.0.send(serde_json::to_string(task).unwrap()).unwrap();
        }
        Ok(())
    }
}

#[tokio::test]
async fn antigravity_saves_a_note_before_confirming_it_went_to_the_running_turn() {
    const NOTE: &str = "A note about the example.";
    let fake = Fake::new(json!([[]]));
    let release = fake.dir.join("finish-turn");
    fake.script(&json!([[
        {"thought": "Waiting for a note.", "usage": {"inputTokens": 200, "outputTokens": 10}},
        {"wait_for": "note"},
        {"wait_file": release},
        {"text": "Got it.", "usage": {"inputTokens": 150, "outputTokens": 5, "cachedReadTokens": 75}}
    ]]));
    let (capture, mut snapshots) = mpsc::unbounded_channel();
    let runtime = AgentRuntime::new(Arc::new(LLMClient::new()))
        .with_task_snapshots(Arc::new(DurableSnapshots(capture)));
    let mut run = Run::with_runtime(fake.task(chat()), runtime);
    run.until(|event| matches!(event, AgentEvent::ReasoningChunk(_)))
        .await;
    run.commands
        .send(UserCommand::InjectContext {
            content: NOTE.into(),
        })
        .await
        .unwrap();
    let events = run
        .until(|event| matches!(event, AgentEvent::ContextInjected { .. }))
        .await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::ContextInjected { .. })
    ));
    let saved = std::iter::from_fn(|| snapshots.try_recv().ok()).any(|saved| saved.contains(NOTE));
    assert!(
        saved,
        "the note wasn't in a durable snapshot when it was acknowledged"
    );
    std::fs::write(release, "finish").unwrap();
    let (task, _) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Got it."),
        "{}",
        fake.log()
    );
    assert_eq!(task.usage_history.len(), 1);
    assert_eq!(task.usage_history[0].input_tokens, 425);
    assert_eq!(task.usage_history[0].output_tokens, 15);
    assert!(fake.log().contains(NOTE));
}

#[tokio::test]
async fn antigravity_cancels_the_turn_and_records_its_usage() {
    let fake = Fake::new(json!([[
        {"thought": "Waiting for cancel."}, {"wait_for": "cancel"}
    ]]));
    let mut run = Run::start(fake.task(chat()));
    run.until(|event| matches!(event, AgentEvent::ReasoningChunk(_)))
        .await;
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert_eq!(
        task.model_call_usage_history[0].stop_reason.as_deref(),
        Some("cancelled")
    );
}

#[tokio::test]
async fn antigravity_uses_a_fresh_session_after_a_failed_turn() {
    let fake = Fake::new(json!([
        [{"fail": "UPSTREAM (code 500): Try again"}], [{"text": "Recovered."}]
    ]));
    let (task, events) = Run::start(fake.task(chat())).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Recovered."),
        "{}",
        fake.log()
    );
    assert_eq!(task.usage_history.len(), 1);
    assert!(events.iter().any(|event| matches!(event, AgentEvent::ServiceWarning { text } if text.contains("Retrying in a new session"))));
}

#[tokio::test]
async fn antigravity_refuses_a_fallback_before_the_first_prompt() {
    let fake = Fake::new(json!({"fallback": true, "turns": [[{"text": "Must not run."}]]}));
    let (task, _) = Run::start(fake.task(chat())).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Failed { reason, .. } if reason.contains("didn't load Arko's agent")),
        "{}",
        fake.log()
    );
    assert!(!fake.log().contains("Do the example task."));
}

#[tokio::test]
async fn antigravity_cancels_if_an_unexpected_tool_is_reported() {
    let fake = Fake::new(json!([[
        {"update": {"sessionUpdate": "tool_call", "name": "run_command", "toolCallId": "example", "status": "in_progress"}},
        {"wait_for": "cancel"}
    ]]));
    let (task, events) = Run::start(fake.task(chat())).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Failed { reason, .. } if reason.contains("run_command")),
        "{}",
        fake.log()
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFormatted { .. }))
    );
}

#[tokio::test]
async fn antigravity_hands_over_from_its_context_count() {
    let fake = Fake::new(json!([
        [
            {"update": {"sessionUpdate": "usage_update", "used": 100, "size": 10000}},
            {"update": {"sessionUpdate": "usage_update", "used": 2000, "size": 10000}},
            {"text": "Ready for the handover."}
        ],
        [{"text": "Report on the example."}],
        [complete("Done after handover")]
    ]));
    let mut task = fake.task(tools(&[]));
    task.agent_config.handover = Some(HandoverConfig::at_context_fraction(0.1));
    let (task, events) = Run::start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Done after handover"),
        "{}",
        fake.log()
    );
    assert_eq!(task.handovers.len(), 1);
    assert!(events.iter().any(|event| matches!(event, AgentEvent::ConversationHandedOver { report, input_tokens: 2000, .. } if report == "Report on the example.")));
    assert!(fake.log().contains("Report on the example."));
}

#[tokio::test]
async fn antigravity_returns_an_error_if_the_new_session_also_fails() {
    let fake = Fake::new(json!([
        [{"fail": "First failure"}], [{"fail": "Second failure"}], [{"text": "Must not run"}]
    ]));
    let mut run = Run::start(fake.task(chat()));
    let events = run
        .until(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
        .await;
    assert!(events.iter().any(|event| matches!(event, AgentEvent::ErrorEncountered { error_message, .. } if error_message.contains("Second failure"))));
    run.commands
        .send(UserCommand::RejectErrorRecovery { message: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Failed { reason, .. } if reason.contains("Second failure"))
    );
    assert_eq!(fake.logged("prompt").len(), 2);
    let launches = fake.logged("session_new");
    assert_eq!(launches.len(), 2);
    assert_ne!(launches[0]["pid"], launches[1]["pid"]);
}

#[tokio::test]
async fn antigravity_allows_the_inert_built_ins_but_does_not_record_them_as_arko_calls() {
    let fake = Fake::new(json!([[
        {"builtin": "manage_task", "input": {"Action": "list"}},
        {"builtin": "list_resources"}, {"builtin": "read_resource"},
        {"text": "No resources or background tasks."}
    ]]));
    let (task, events) = Run::start(fake.task(chat())).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFormatted { .. }))
    );
}

#[tokio::test]
async fn antigravity_checks_the_fallback_hooks_during_the_turn() {
    let fake =
        Fake::new(json!({"fallback": "silent", "turns": [[{"text": "From the default agent"}]]}));
    let (task, _) = Run::start(fake.task(chat())).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Failed { reason, .. } if reason.contains("didn't load Arko's agent")),
        "{}",
        fake.log()
    );
    assert_eq!(fake.logged("hook").len(), 1);
}

#[tokio::test]
async fn antigravity_keeps_images_from_tool_results() {
    let fake = Fake::new(json!([[
        {"tool": "read_file", "input": {"path": "example.png"}},
        complete("Saw the image")
    ]]));
    let task = fake.task(tools(&["read_file"]));
    image::RgbImage::from_pixel(1, 1, image::Rgb([38, 84, 17]))
        .save(task.current_working_directory.join("example.png"))
        .unwrap();
    let (task, _) = Run::start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        fake.logged("tool_result")
            .iter()
            .any(|entry| entry.to_string().contains("image/png")),
        "{}",
        fake.log()
    );
    assert!(
        task.conversation
            .messages
            .iter()
            .any(|message| message.content.iter().any(|content| content
                .as_tool_result()
                .is_some_and(|result| result.images.len() == 1)))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn antigravity_runs_share_one_usage_read_even_when_it_fails() {
    for fails in [false, true] {
        let turns = vec![vec![json!({"sleep": 0.2}), json!({"text": "Done"})]; 5];
        let script = if fails {
            json!({"turns": turns, "usage_fail": {"exit": 1, "stderr": "No usage"}})
        } else {
            json!({"turns": turns})
        };
        let fake = Fake::new(script);
        let run = || Run::start(fake.task(chat())).finish();
        let all = tokio::time::timeout(
            Duration::from_secs(15),
            futures::future::join_all((0..4).map(|_| run())),
        )
        .await
        .unwrap();
        let last = run().await;
        for (task, _) in all.into_iter().chain([last]) {
            assert!(
                matches!(task.current_state, TaskState::Completed { summary } if summary == "Done")
            );
        }
        assert_eq!(fake.logged("usage_command").len(), 1, "{}", fake.log());
        let launches = fake.logged("session_new");
        let pids: std::collections::HashSet<String> = launches
            .iter()
            .map(|launch| launch["pid"].to_string())
            .collect();
        assert_eq!(pids.len(), 5, "{}", fake.log());
    }
}

#[tokio::test]
async fn antigravity_only_selects_a_thought_level_the_model_offers() {
    let fake = Fake::new(json!({
        "models": [{"value": "gemini-4-neon", "name": "Gemini 4 Neon", "levels": ["medium", "high"]}],
        "turns": [[{"text": "Done"}]]
    }));
    let (task, _) = Run::start(fake.task(chat())).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{}",
        fake.log()
    );
    assert_eq!(fake.logged("prompt")[0]["model"], "gemini-4-neon-high");
}

/// The task's first turn, then a follow-up in a later run, with the task
/// saved and loaded between them as a host does across a restart.
async fn follow_up(fake: &Fake, mut task: Task, follow_up: &str) -> Task {
    let (task_after, _) = Run::start(task).finish().await;
    task = serde_json::from_value(serde_json::to_value(task_after).unwrap()).unwrap();
    task.prepare_for_resumption_from_completed(follow_up.into())
        .unwrap();
    let (task, _) = Run::start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}\n{}",
        task.current_state,
        fake.log()
    );
    task
}

#[tokio::test]
async fn antigravity_follow_up_resumes_its_session() {
    use std::os::unix::fs::PermissionsExt;
    let fake = Fake::new(json!([
        [{"text": "An example answer."}], [{"text": "The follow-up answer."}]
    ]));
    let task = follow_up(&fake, fake.task(chat()), "One more example, please.").await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "The follow-up answer.")
    );
    let prompts = fake.logged("prompt");
    assert_eq!(prompts.len(), 2);
    assert_eq!(fake.logged("session_new").len(), 1, "{}", fake.log());
    let resumed = fake.logged("session_resume");
    assert_eq!(resumed.len(), 1, "{}", fake.log());
    assert_eq!(resumed[0]["params"]["sessionId"], prompts[0]["sessionId"]);
    assert_eq!(prompts[1]["sessionId"], prompts[0]["sessionId"]);
    // Only the new message goes, on the task's model and level again.
    let sent = prompts[1]["text"].as_str().unwrap();
    assert!(sent.contains("One more example, please."), "{sent}");
    assert!(!sent.contains("An example answer."), "{sent}");
    assert_eq!(prompts[1]["model"], prompts[0]["model"]);
    // The profile that holds the session stays, for the user alone.
    let profile = fake.dir.join("profiles").join(task.id.to_string());
    let stored = profile.join(format!(
        "antigravity-cli/conversations/{}.json",
        prompts[0]["sessionId"].as_str().unwrap()
    ));
    assert!(stored.is_file(), "{}", stored.display());
    let mode = std::fs::metadata(&profile).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o700);
}

struct HoldFirstDurableSnapshot(mpsc::UnboundedSender<Vec<u8>>);

#[async_trait::async_trait]
impl crate::snapshot::TaskSnapshotSink for HoldFirstDurableSnapshot {
    async fn task_changed(&self, task: &Task, durable: bool) -> Result<(), String> {
        if durable {
            self.0.send(serde_json::to_vec(task).unwrap()).unwrap();
            std::future::pending::<()>().await;
        }
        Ok(())
    }
}

#[tokio::test]
async fn antigravity_recovers_a_follow_up_from_its_pre_send_snapshot() {
    const FOLLOW_UP: &str = "Check the other example.";
    let fake = Fake::new(json!([[{"text": "Ready."}], [{"text": "Checked."}]]));
    let (mut task, _) = Run::start(fake.task(chat())).finish().await;
    task.prepare_for_resumption_from_completed(FOLLOW_UP.into())
        .unwrap();
    let (captured, mut checkpoint) = mpsc::unbounded_channel();
    let runtime = AgentRuntime::new(Arc::new(LLMClient::new()))
        .with_task_snapshots(Arc::new(HoldFirstDurableSnapshot(captured)));
    let (_events, _commands, handle) = runtime.run_task(
        task,
        Arc::new(ToolRegistry::new()),
        None,
        false,
        None,
        None,
        None,
    );
    let snapshot = tokio::time::timeout(Duration::from_secs(15), checkpoint.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&snapshot).contains(FOLLOW_UP));
    assert_eq!(fake.logged("session_resume").len(), 1, "{}", fake.log());
    assert_eq!(fake.logged("prompt").len(), 1, "{}", fake.log());
    handle.abort();
    assert!(handle.await.unwrap_err().is_cancelled());

    let saved: Task = serde_json::from_slice(&snapshot).unwrap();
    let (finished, _) = Run::start(saved).finish().await;
    assert!(matches!(
        finished.current_state,
        TaskState::Completed { .. }
    ));
    let prompts = fake.logged("prompt");
    assert_eq!(prompts.len(), 2, "{}", fake.log());
    assert!(
        prompts[1]["text"].as_str().unwrap().contains(FOLLOW_UP),
        "{prompts:#?}"
    );
}

#[tokio::test]
async fn antigravity_opens_a_relative_profile_root_from_another_working_directory() {
    let fake = Fake::in_dir(
        std::env::current_dir().unwrap(),
        json!([[{"text": "Ready."}], [{"text": "Checked."}]]),
    );
    let relative = PathBuf::from(fake.dir.file_name().unwrap()).join("profiles");
    let mut task = fake.task(chat());
    task.agent_config.antigravity.as_mut().unwrap().profile_root = Some(relative);
    let task = follow_up(&fake, task, "Check the other example.").await;
    assert!(fake.dir.join("profiles").join(task.id.to_string()).is_dir());
    assert_eq!(fake.logged("session_resume").len(), 1, "{}", fake.log());
}

#[tokio::test]
async fn antigravity_keeps_a_saved_session_until_the_user_signs_in_again() {
    let fake = Fake::new(json!([[{"text": "Ready."}], [{"text": "Checked."}]]));
    let (mut task, _) = Run::start(fake.task(chat())).finish().await;
    let saved = task.antigravity_session.as_ref().unwrap();
    let stored = fake
        .dir
        .join("profiles")
        .join(task.id.to_string())
        .join(format!("antigravity-cli/conversations/{}.json", saved.id));
    assert!(stored.is_file());
    fake.script(&json!({"signed_out": true}));
    task.prepare_for_resumption_from_completed("Check the other example.".into())
        .unwrap();
    let mut run = Run::start(task);
    let events = run
        .until(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
        .await;
    assert!(
        matches!(events.last(), Some(AgentEvent::ErrorEncountered {
        error_message, requires_approval_to_retry: true, ..
    }) if error_message == "Run `agy` to sign in."),
        "{events:?}"
    );
    assert!(
        stored.is_file(),
        "the signed-out CLI shouldn't discard the conversation"
    );
    fake.script(&json!([[{"text": "Ready."}], [{"text": "Checked."}]]));
    run.commands
        .send(UserCommand::ApproveErrorRecovery { message: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let prompts = fake.logged("prompt");
    assert_eq!(prompts[0]["sessionId"], prompts[1]["sessionId"]);
    assert!(
        prompts[1]["text"]
            .as_str()
            .unwrap()
            .contains("Check the other example.")
    );
}

#[tokio::test]
async fn antigravity_seeds_only_when_resume_fails() {
    let fake = Fake::new(json!({"resume": "refuse", "turns": [
        [{"text": "An example answer."}], [{"text": "The follow-up answer."}]
    ]}));
    follow_up(&fake, fake.task(chat()), "One more example, please.").await;
    assert_eq!(
        fake.logged("session_resume_refused").len(),
        1,
        "{}",
        fake.log()
    );
    assert_eq!(fake.logged("session_new").len(), 2);
    let prompts = fake.logged("prompt");
    assert_ne!(prompts[1]["sessionId"], prompts[0]["sessionId"]);
    let seed = prompts[1]["text"].as_str().unwrap();
    assert!(seed.contains("# The conversation so far"), "{seed}");
    assert!(seed.contains("An example answer."), "{seed}");
    assert!(
        seed.ends_with("# The user's new message\n\nOne more example, please."),
        "{seed}"
    );
    let text_blocks = prompts[1]["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "text")
        .count();
    assert_eq!(text_blocks, 1);
}

/// A new session gets each call's result under the call, long output cut,
/// and the user's new message alone at the end.
#[tokio::test]
async fn antigravity_seed_puts_results_under_calls_and_the_new_message_last() {
    let fake = Fake::new(json!([
        [{"tool": "read_file", "input": {"path": "example.txt"}}, complete("Read the example.")],
        [complete("Answered the new question.")]
    ]));
    let mut task = fake.task(tools(&["read_file"]));
    // Without a profile root each run starts a new session.
    task.agent_config.antigravity.as_mut().unwrap().profile_root = None;
    let long = format!(
        "Example first line.\n{}\nExample last line.\n",
        "x".repeat(6000)
    );
    std::fs::write(task.current_working_directory.join("example.txt"), long).unwrap();
    follow_up(&fake, task, "Is the example still there?").await;
    let prompts = fake.logged("prompt");
    assert_eq!(prompts.len(), 2, "{}", fake.log());
    let seed = prompts[1]["text"].as_str().unwrap();
    let at = |text: &str| {
        seed.find(text)
            .unwrap_or_else(|| panic!("{text:?} isn't in the seed:\n{seed}"))
    };
    // The task, then the call with its result right under it.
    assert!(at("Do the example task.") < at("Called `read_file`"));
    let result = &seed[at("Called `read_file`")..at("Example last line.")];
    assert!(result.contains("It returned:"), "{result}");
    assert!(result.contains("Example first line."), "{result}");
    assert!(!result.contains("\n## "), "{result}");
    assert!(seed.contains("characters left out"), "{seed}");
    assert!(!seed.contains("\n## user"), "{seed}");
    assert!(!seed.contains("continue the unfinished work"), "{seed}");
    assert!(
        seed.ends_with("# The user's new message\n\nIs the example still there?"),
        "{seed}"
    );
}

#[tokio::test]
async fn antigravity_reports_the_context_not_the_turn_sum() {
    let step = |n: u32| json!({"thought": format!("Step {n}."), "usage": {"inputTokens": 100000, "outputTokens": 10}});
    let fake = Fake::new(json!([[
        step(1), step(2), step(3), step(4), step(5), {"text": "Done."}
    ]]));
    let (task, events) = Run::start(fake.task(chat())).finish().await;
    // Billing counts every request of the turn.
    assert_eq!(task.usage_history.len(), 1);
    assert_eq!(task.usage_history[0].input_tokens, 500_000);
    // The conversation is as large as agy said it is.
    assert_eq!(task.current_conversation_context_size(), Some(100_010));
    assert_eq!(task.current_context_size(), Some(100_010));
    let call = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ModelCallUsageReported(call) => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(call.input_tokens, 500_000);
    let call = serde_json::to_value(call).unwrap();
    assert_eq!(call["context_tokens"], 100_010, "{call}");
    let latest = serde_json::to_value(&task.latest_request).unwrap();
    assert_eq!(latest["input_tokens"], 100_010, "{latest}");
}

/// A handover's fresh conversation starts a new session in a new profile,
/// and the old session goes.
#[tokio::test]
async fn antigravity_hands_over_into_a_new_profile() {
    let fake = Fake::new(json!([
        [
            {"update": {"sessionUpdate": "usage_update", "used": 100, "size": 10000}},
            {"update": {"sessionUpdate": "usage_update", "used": 2000, "size": 10000}},
            {"text": "Ready for the handover."}
        ],
        [{"text": "Report on the example."}],
        [complete("Done after handover")]
    ]));
    let mut task = fake.task(tools(&[]));
    task.agent_config.handover = Some(HandoverConfig::at_context_fraction(0.1));
    let (task, _) = Run::start(task).finish().await;
    assert_eq!(task.handovers.len(), 1, "{}", fake.log());
    let prompts = fake.logged("prompt");
    let (old, new) = (
        prompts[0]["sessionId"].as_str().unwrap(),
        prompts.last().unwrap()["sessionId"].as_str().unwrap(),
    );
    assert_ne!(old, new);
    let conversations = fake
        .dir
        .join("profiles")
        .join(task.id.to_string())
        .join("antigravity-cli/conversations");
    assert!(conversations.join(format!("{new}.json")).is_file());
    assert!(!conversations.join(format!("{old}.json")).exists());
    // The fresh conversation starts from the report, not the old record.
    let fresh = prompts.last().unwrap()["text"].as_str().unwrap();
    assert!(!fresh.contains("# The conversation so far"), "{fresh}");
}

#[tokio::test]
async fn antigravity_starts_a_new_session_when_the_prompt_changed() {
    let fake = Fake::new(json!([
        [{"text": "An example answer."}], [{"text": "The follow-up answer."}]
    ]));
    let (task, _) = Run::start(fake.task(chat())).finish().await;
    let mut task: Task = serde_json::from_value(serde_json::to_value(task).unwrap()).unwrap();
    let mut config = task.agent_config.clone();
    config.instructions = "Read the example twice.".into();
    task.reconfigure(config).unwrap();
    task.prepare_for_resumption_from_completed("One more example, please.".into())
        .unwrap();
    let (_, _) = Run::start(task).finish().await;
    assert!(fake.logged("session_resume").is_empty(), "{}", fake.log());
    assert_eq!(fake.logged("session_new").len(), 2);
    let seed = fake.logged("prompt")[1]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(seed.contains("# The conversation so far"), "{seed}");
    assert!(seed.contains("An example answer."), "{seed}");
}

#[tokio::test]
async fn antigravity_starts_a_new_session_when_the_profile_is_gone() {
    let fake = Fake::new(json!([
        [{"text": "An example answer."}], [{"text": "The follow-up answer."}]
    ]));
    let (task, _) = Run::start(fake.task(chat())).finish().await;
    let mut task: Task = serde_json::from_value(serde_json::to_value(task).unwrap()).unwrap();
    std::fs::remove_dir_all(fake.dir.join("profiles").join(task.id.to_string())).unwrap();
    task.prepare_for_resumption_from_completed("One more example, please.".into())
        .unwrap();
    let (task, _) = Run::start(task).finish().await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "The follow-up answer."),
        "{}",
        fake.log()
    );
    assert_eq!(fake.logged("session_new").len(), 2);
    let seed = fake.logged("prompt")[1]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(seed.contains("An example answer."), "{seed}");
}

/// Two conversations at once each keep their own profile and session.
#[tokio::test]
async fn antigravity_conversations_at_once_resume_their_own_sessions() {
    let fake = Fake::new(json!([
        {"match": "First example", "steps": [{"sleep": 0.2}, {"text": "First answer."}]},
        {"match": "Second example", "steps": [{"sleep": 0.2}, {"text": "Second answer."}]},
        {"match": "More of the first", "steps": [{"text": "First again."}]},
        {"match": "More of the second", "steps": [{"text": "Second again."}]}
    ]));
    let task = |prompt: &str| {
        let mut task = fake.task(chat());
        task.initial_task_prompt = prompt.into();
        task
    };
    let (first, second) = tokio::join!(
        follow_up(&fake, task("First example."), "More of the first."),
        follow_up(&fake, task("Second example."), "More of the second.")
    );
    assert!(
        matches!(&first.current_state, TaskState::Completed { summary } if summary == "First again.")
    );
    assert!(
        matches!(&second.current_state, TaskState::Completed { summary } if summary == "Second again.")
    );
    let prompts = fake.logged("prompt");
    let session = |text: &str| {
        prompts
            .iter()
            .find(|prompt| prompt["text"].as_str().unwrap().contains(text))
            .unwrap_or_else(|| panic!("no prompt with {text:?}: {prompts:#?}"))["sessionId"]
            .clone()
    };
    assert_eq!(session("First example."), session("More of the first."));
    assert_eq!(session("Second example."), session("More of the second."));
    assert_ne!(session("First example."), session("Second example."));
    assert_eq!(fake.logged("session_resume").len(), 2);
}

#[tokio::test]
async fn antigravity_reads_call_arguments_only_from_mcp() {
    let contents = "An example beginning with a result label.\n{}\n";
    let fake = Fake::new(json!([[
        {"tool": "write_file", "input": {"path": "example.txt", "content": contents}},
        complete("Wrote the file")
    ]]));
    let task = fake.task(tools(&["write_file"]));
    let path = task.current_working_directory.join("example.txt");
    let (task, events) = Run::start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(std::fs::read_to_string(path).unwrap(), contents);
    let args = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolCallFormatted {
                tool_name,
                parameters,
                ..
            } if tool_name == "write_file" => Some(parameters),
            _ => None,
        })
        .unwrap();
    assert_eq!(args["content"], contents);
    assert!(!args.contains_key("toolSummary"));
}

#[tokio::test]
async fn antigravity_does_not_refresh_the_prompt_cache() {
    let fake = Fake::new(json!([]));
    let runtime = AgentRuntime::new(Arc::new(LLMClient::new()));
    let error = runtime
        .refresh_prompt_cache(fake.task(chat()), Arc::new(ToolRegistry::new()), None)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        ArkoAgentsError::LLM(arko_ai::Error::UnsupportedFeature(_))
    ));
    assert!(fake.log().is_empty());
}

#[tokio::test]
async fn antigravity_finishes_without_waiting_for_a_plan_usage_read() {
    let fake = Fake::new(json!([]));
    let mut task = fake.task(chat());
    let (started, release) = fake.hold_usage(&mut task);
    fake.script(&json!([[{"wait_file": started}, {"text": "Done"}]]));
    let result = tokio::time::timeout(Duration::from_secs(2), Run::start(task).finish()).await;
    std::fs::write(release, "").unwrap();
    let (task, events) = result.expect("a turn must not wait for plan usage");
    assert!(matches!(task.current_state, TaskState::Completed { summary } if summary == "Done"));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::PlanUsageReported(_)))
    );
}

fn streamed(events: &[AgentEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::AssistantResponseChunk(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Two chunks of the model's reply that both start with a warning sign.
#[tokio::test]
async fn two_chunks_that_start_with_a_warning_sign_both_arrive() {
    let fake = Fake::new(json!([[
        {"text": "⚠ The first check failed.\n"},
        {"text": "⚠ The second check failed.\n"},
        {"text": "Both need a look."}
    ]]));
    let (task, events) = Run::start(fake.task(chat())).finish().await;
    let expected = "⚠ The first check failed.\n⚠ The second check failed.\nBoth need a look.";
    assert_eq!(streamed(&events), expected, "{}", fake.log());
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == expected),
        "{:?}",
        task.current_state
    );
}

/// Text that starts with a warning sign, then a tool call.
#[tokio::test]
async fn a_chunk_with_a_warning_sign_stays_before_the_call_after_it() {
    let fake = Fake::new(json!([[
        {"text": "⚠ The example file may be stale."},
        {"tool": "read_file", "input": {"path": "example.txt"}},
        complete("It was fine.")
    ]]));
    let task = fake.task(tools(&["read_file"]));
    std::fs::write(
        task.current_working_directory.join("example.txt"),
        "Example.",
    )
    .unwrap();
    let (task, events) = Run::start(task).finish().await;
    let chunk = events.iter().position(
        |event| matches!(event, AgentEvent::AssistantResponseChunk(text) if text.contains("stale")),
    );
    let call = events.iter().position(|event| {
        matches!(event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "read_file")
    });
    let text_at = task.conversation.messages.iter().position(|message| {
        message
            .content
            .iter()
            .any(|content| content.as_text().is_some_and(|text| text.contains("stale")))
    });
    let call_at = task.conversation.messages.iter().position(|message| {
        message
            .tool_calls
            .iter()
            .any(|call| call.name == "read_file")
    });
    assert!(
        chunk.is_some() && chunk < call && text_at.is_some() && text_at < call_at,
        "event: chunk {chunk:?}, call {call:?}; transcript: text {text_at:?}, call {call_at:?}"
    );
}

/// Control: the same text without the warning sign.
#[tokio::test]
async fn a_plain_chunk_stays_before_the_call_after_it() {
    let fake = Fake::new(json!([[
        {"text": "The example file may be stale."},
        {"tool": "read_file", "input": {"path": "example.txt"}},
        complete("It was fine.")
    ]]));
    let task = fake.task(tools(&["read_file"]));
    std::fs::write(
        task.current_working_directory.join("example.txt"),
        "Example.",
    )
    .unwrap();
    let (task, events) = Run::start(task).finish().await;
    let chunk = events.iter().position(
        |event| matches!(event, AgentEvent::AssistantResponseChunk(text) if text.contains("stale")),
    );
    let call = events.iter().position(|event| {
        matches!(event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "read_file")
    });
    let text_at = task.conversation.messages.iter().position(|message| {
        message
            .content
            .iter()
            .any(|content| content.as_text().is_some_and(|text| text.contains("stale")))
    });
    let call_at = task.conversation.messages.iter().position(|message| {
        message
            .tool_calls
            .iter()
            .any(|call| call.name == "read_file")
    });
    assert!(
        chunk.is_some() && chunk < call && text_at.is_some() && text_at < call_at,
        "event: chunk {chunk:?}, call {call:?}; transcript: text {text_at:?}, call {call_at:?}"
    );
}

/// A task resumed with an open call, on Antigravity.
#[tokio::test]
async fn the_recovery_note_on_antigravity_names_no_other_cli() {
    let fake = Fake::new(json!([[{"text": "Noted."}]]));
    let mut task = fake.task(chat());
    task.running_tool_calls.push(crate::task::RunningToolCall {
        tool_call_id: uuid::Uuid::new_v4(),
        call_id: Some("call-1".into()),
        tool_name: "read_file".into(),
        parameters: [("path".to_string(), json!("example.txt"))]
            .into_iter()
            .collect(),
        custom_output: None,
        message_to_agent: None,
        built_in: false,
        output: Some("Example content.".into()),
        output_images: Vec::new(),
    });
    let (_task, _) = Run::start(task).finish().await;
    let prompts = fake.logged("prompt");
    let text = prompts[0]["text"].as_str().unwrap_or_default().to_string();
    assert!(text.contains("tool calls were open"), "{text}");
    assert!(!text.contains("Claude Code"), "{text}");
}

/// A chat on Antigravity that ended its turn over the handover size, then
/// gets the user's next message.
#[tokio::test]
async fn a_chat_report_on_antigravity_sees_the_conversation() {
    let fake = Fake::new(json!([
        [
            {"thought": "Read the question.", "usage": {"inputTokens": 1000, "outputTokens": 10}},
            {"text": "The first answer mentions the blue example.",
             "usage": {"inputTokens": 299990, "outputTokens": 10}}
        ],
        [{"text": "Report of the chat."}],
        [{"text": "The second answer."}],
        [{"text": "A spare answer."}]
    ]));
    let mut task = fake.task(chat());
    task.agent_config.handover = Some(HandoverConfig::at_context_fraction(0.1));
    let (task, _) = Run::start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    let mut task: Task = serde_json::from_value(serde_json::to_value(task).unwrap()).unwrap();
    task.prepare_for_resumption_from_completed("Another question about it.".into())
        .unwrap();
    // The user asks for a handover with the next message.
    let run = Run::start(task);
    run.commands
        .send(UserCommand::RequestHandover)
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert_eq!(task.handovers.len(), 1);
    let prompts = fake.logged("prompt");
    assert_eq!(prompts.len(), 3, "{prompts:#?}");
    let report_prompt = &prompts[1];
    assert!(
        report_prompt["text"]
            .as_str()
            .unwrap()
            .contains("Reply with that summary")
    );
    // The session that holds the conversation writes the report.
    assert_eq!(
        report_prompt["sessionId"], prompts[0]["sessionId"],
        "the report request went to a new session without the conversation: {prompts:#?}"
    );
    assert_eq!(fake.logged("session_resume").len(), 1);
}

/// Antigravity can't start a session (signed out), as against a Claude Code
/// CLI that can't start.
#[tokio::test]
async fn a_session_that_cant_start_can_be_retried() {
    let fake = Fake::new(json!({"signed_out": true, "turns": [[{"text": "Done"}]]}));
    let mut run = Run::start(fake.task(chat()));
    let events = run
        .until(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
        .await;
    assert!(
        matches!(events.last(), Some(AgentEvent::ErrorEncountered {
        error_message, requires_approval_to_retry: true, ..
    }) if error_message == "Run `agy` to sign in."),
        "{events:?}"
    );
    fake.script(&json!([[{"text": "Done after sign-in."}]]));
    run.commands
        .send(UserCommand::ApproveErrorRecovery { message: None })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Done after sign-in.")
    );
}

/// A tool that runs until the test releases it.
#[derive(Debug)]
struct NonInterruptibleTool {
    started: Arc<tokio::sync::Notify>,
    released: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl crate::tool::Tool for NonInterruptibleTool {
    fn definition(&self) -> crate::tool::ToolDefinition {
        crate::tool::ToolDefinition {
            name: "slow".into(),
            description: "Waits for permission to finish.".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: crate::tool::ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    fn interruptible(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        _: &crate::tool::TaskContext<'_>,
        _: std::collections::HashMap<String, Value>,
    ) -> Result<String, crate::tool::ToolError> {
        self.started.notify_one();
        self.released.notified().await;
        Ok("Finished the call.".into())
    }
}

/// The user cancels while that tool's call runs.
#[tokio::test]
async fn a_call_running_at_a_cancel_gets_an_outcome() {
    let fake = Fake::new(json!([[
        {"tool": "slow", "input": {}},
        complete("Done")
    ]]));
    let cancel_marker = fake.dir.join("mcp-cancelled");
    let mut task = fake.task(tools(&["slow"]));
    task.agent_config.antigravity.as_mut().unwrap().env.insert(
        "FAKE_AGY_MCP_CANCEL_MARKER".into(),
        cancel_marker.display().to_string(),
    );
    let started = Arc::new(tokio::sync::Notify::new());
    let released = Arc::new(tokio::sync::Notify::new());
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(NonInterruptibleTool {
        started: started.clone(),
        released: released.clone(),
    }));
    let runtime = AgentRuntime::new(Arc::new(LLMClient::new()));
    let (events, commands, handle) =
        runtime.run_task(task, Arc::new(registry), None, false, None, None, None);
    let mut run = Run {
        events,
        commands,
        handle,
    };
    let seen = run
        .until(|event| matches!(event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "slow"))
        .await;
    let Some(AgentEvent::ToolCallFormatted { tool_call_id, .. }) = seen.last() else {
        panic!()
    };
    let id = *tool_call_id;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !cancel_marker.is_file() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(!run.handle.is_finished());
    released.notify_one();
    let (task, events) = run.finish().await;
    let settled = events.iter().any(|event| {
        matches!(event, AgentEvent::ToolResult { tool_call_id, output, .. }
        if *tool_call_id == id && output == "Finished the call.")
    });
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert!(task.running_tool_calls.is_empty());
    assert!(
        settled,
        "the call has no outcome after the cancel: {events:#?}"
    );
}

/// A shell that runs nothing, for the live eval.
#[derive(Debug)]
struct NoShell;

#[async_trait::async_trait]
impl crate::tool::Tool for NoShell {
    fn definition(&self) -> crate::tool::ToolDefinition {
        crate::tool::ToolDefinition {
            name: "shell".into(),
            description: "Runs a shell command in the workspace and returns its output.".into(),
            output_based_example: None,
            parameters: vec![crate::tool::ToolParameterDefinition {
                name: "command".into(),
                description: "The command.".into(),
                param_type: crate::tool::ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
                enum_values: None,
            }],
            parallel_safe: false,
            tool_type_details: crate::tool::ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _: &crate::tool::TaskContext<'_>,
        _: std::collections::HashMap<String, Value>,
    ) -> Result<String, crate::tool::ToolError> {
        Ok("exit code 0".into())
    }
}

/// A conversation of about 250 KB like the one that went wrong: the task, a
/// box status check that said no box is up, long checks, replies, and a
/// request about the box that got no answer.
fn long_record(task: &mut Task) {
    use arko_ai::{Content, Message, ToolCall, ToolResult};
    let messages = &mut task.conversation.messages;
    let call = |messages: &mut Vec<Message>, n: usize, command: &str, output: &str| {
        let call: ToolCall = serde_json::from_value(json!({
            "id": format!("call-{n}"), "name": "shell", "arguments": {"command": command}
        }))
        .unwrap();
        let mut asked = Message::new_assistant("");
        asked.tool_calls.push(call.clone());
        messages.push(asked);
        messages.push(Message::new_user_with_contents(vec![Content::ToolResult(
            ToolResult::from_tool_call(&call, output),
        )]));
    };
    messages.push(Message::new_user(
        "Feature stamping is committed on feature/stamp at 1a2b3c4d in ~/work/example. Check whether it is merged into release-0.9 and installed as the example command.",
    ));
    let filler = |label: &str| {
        (0..90)
            .map(|line| format!("{label} {line:03}: commit 1a2b3c4d example log line with some detail about the build\n"))
            .collect::<String>()
    };
    let mut n = 0;
    for label in ["git log", "git branch", "ls -la", "strings example"] {
        n += 1;
        call(messages, n, label, &filler(label));
    }
    messages.push(Message::new_assistant(
        "No to both. feature/stamp is not merged into release-0.9, and the installed example command is older.",
    ));
    messages.push(Message::new_user("Please do both"));
    n += 1;
    call(
        messages,
        n,
        "boxctl status",
        "no box is up\nwatch: inactive\n",
    );
    for label in [
        "git merge --ff-only 1a2b3c4d",
        "cargo build --release",
        "install -m 755",
        "example --version",
        "df -h /",
        "git reflog",
    ] {
        for _ in 0..4 {
            n += 1;
            call(messages, n, label, &filler(label));
        }
    }
    messages.push(Message::new_assistant(
        "Done. release-0.9 is fast-forwarded to 1a2b3c4d, and the example command is built and installed.",
    ));
    messages.push(Message::new_user("Alright bring down the build box please"));
    for label in ["strings example", "git reflog", "example --version"] {
        for _ in 0..4 {
            n += 1;
            call(messages, n, label, &filler(label));
        }
    }
    messages.push(Message::new_assistant("Both are done and verified."));
    task.claude_code_synced = messages.len();
    task.set_state(TaskState::Completed {
        summary: "Both are done and verified.".into(),
    });
}

/// Live, on the installed `agy` signed in: Flash at low thought level gets
/// the record above in a new session and the question that went unanswered.
/// Run it with `cargo test -p arko_agents --lib live_antigravity -- --ignored`.
#[tokio::test]
#[ignore = "runs the installed Antigravity CLI on the user's plan"]
async fn live_antigravity_answers_the_new_message_after_a_long_record() {
    let fake = Fake::new(json!([]));
    let mut task = fake.task(AgentToolConfig {
        text_reply_ends_task: true,
        ..tools(&["shell"])
    });
    task.agent_config.llm_config.model = Model::antigravity(Model::google_gemini_3_8_flash());
    let settings = task.agent_config.antigravity.as_mut().unwrap();
    settings.cli_path = None;
    settings.env.clear();
    settings.thought_level = Some("low".into());
    long_record(&mut task);
    let record: usize = task
        .conversation
        .messages
        .iter()
        .map(|message| serde_json::to_string(message).unwrap().len())
        .sum();
    assert!(record > 200_000, "{record}");
    task.prepare_for_resumption_from_completed("Is the build box down?".into())
        .unwrap();
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(NoShell));
    let runtime = AgentRuntime::new(Arc::new(LLMClient::new()));
    let (events, commands, handle) =
        runtime.run_task(task, Arc::new(registry), None, false, None, None, None);
    let run = Run {
        events,
        commands,
        handle,
    };
    let (task, events) = tokio::time::timeout(Duration::from_secs(600), run.finish())
        .await
        .expect("the live turn took over ten minutes");
    let commands: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallFormatted { parameters, .. } => Some(
                parameters["command"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ),
            _ => None,
        })
        .collect();
    let TaskState::Completed { summary } = &task.current_state else {
        panic!("{:?}", task.current_state)
    };
    eprintln!("reply: {summary}\ncommands: {commands:?}");
    assert!(summary.to_lowercase().contains("box"), "{summary}");
    assert!(
        !commands
            .iter()
            .any(|command| command.contains("git") || command.contains("cargo")),
        "it redid the old work: {commands:?}"
    );
}
