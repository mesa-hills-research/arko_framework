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
        let dir = std::env::temp_dir().join(format!("arko-fake-agy-{}", uuid::Uuid::new_v4()));
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
        let runtime = AgentRuntime::new(Arc::new(LLMClient::new()));
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
        {"text": "The example works."},
        {"update": {"sessionUpdate": "usage_update", "used": 92, "size": 12345}}
    ]]));
    let mut task = fake.task(chat());
    task.initial_task_images = vec![Image::from_bytes(b"example image", "image/png")];
    let (task, events) = Run::start(task).finish().await;
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
async fn antigravity_seeds_a_new_session_after_a_blocking_question() {
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
    assert!(
        fake.log()
            .contains("Continue from this record of the conversation")
    );
    assert!(fake.log().contains("Use the example file."));
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

#[tokio::test]
async fn antigravity_puts_a_note_into_the_running_turn() {
    let fake = Fake::new(json!([[
        {"thought": "Waiting for a note.", "usage": {"inputTokens": 200, "outputTokens": 10}},
        {"wait_for": "note"},
        {"text": "Got it.", "usage": {"inputTokens": 150, "outputTokens": 5, "cachedReadTokens": 75}}
    ]]));
    let mut run = Run::start(fake.task(chat()));
    run.until(|event| matches!(event, AgentEvent::ReasoningChunk(_)))
        .await;
    run.commands
        .send(UserCommand::InjectContext {
            content: "A note about the example.".into(),
        })
        .await
        .unwrap();
    let (task, events) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "Got it."),
        "{}",
        fake.log()
    );
    assert_eq!(task.usage_history.len(), 1);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ContextInjected { .. }))
    );
    assert_eq!(task.usage_history[0].input_tokens, 425);
    assert_eq!(task.usage_history[0].output_tokens, 15);
    assert!(fake.log().contains("A note about the example."));
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
        matches!(task.current_state, TaskState::Failed { reason, .. } if reason.contains("did not load Arko's agent")),
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
        matches!(task.current_state, TaskState::Failed { reason, .. } if reason.contains("fallback hook")),
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
        let turns = vec![vec![json!({"text": "Done"})]; 5];
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
        let (task, events) = run().await;
        for (task, events) in all.into_iter().chain([(task, events)]) {
            assert!(
                matches!(task.current_state, TaskState::Completed { summary } if summary == "Done")
            );
            assert_eq!(
                events
                    .iter()
                    .any(|event| matches!(event, AgentEvent::PlanUsageReported(_))),
                !fails
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

#[tokio::test]
async fn antigravity_resumes_from_the_saved_transcript() {
    let fake = Fake::new(json!([
        [{"text": "An example answer."}], [{"text": "The follow-up answer."}]
    ]));
    let (task, _) = Run::start(fake.task(chat())).finish().await;
    let mut task: Task = serde_json::from_value(serde_json::to_value(task).unwrap()).unwrap();
    assert!(task.agent_config.runs_in_antigravity());
    task.prepare_for_resumption_from_completed("One more example, please.".into())
        .unwrap();
    let (task, _) = Run::start(task).finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { summary } if summary == "The follow-up answer.")
    );
    let prompts = fake.logged("prompt");
    assert_eq!(prompts.len(), 2);
    assert_ne!(prompts[0]["sessionId"], prompts[1]["sessionId"]);
    let replay = prompts[1]["text"].as_str().unwrap();
    assert!(replay.contains("An example answer."), "{replay}");
    assert!(replay.contains("One more example, please."), "{replay}");
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
