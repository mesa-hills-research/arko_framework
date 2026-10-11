//! The overflow guard, the opt-in result caps, the predictive handover
//! check and the bounded overflow recovery, in arko's own runtime.

use super::*;
use crate::agent_config::ToolResultLimits;
use crate::handover::HandoverConfig;
use crate::tool::{ToolError, ToolTypeDetails, builtins::read_file};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// A tool that returns `bytes` bytes of numbered lines.
#[derive(Debug)]
struct DumpTool;

#[async_trait]
impl Tool for DumpTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "dump".into(),
            description: "Prints a lot.".into(),
            parameters: vec![],
            output_based_example: None,
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }
    async fn execute(
        &self,
        _: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let bytes = params.get("bytes").and_then(Value::as_u64).unwrap_or(10) as usize;
        Ok(lines(bytes))
    }
}

/// About `bytes` bytes of numbered lines: "line 1", "line 2", ...
fn lines(bytes: usize) -> String {
    let mut text = String::new();
    let mut n = 1;
    while text.len() < bytes {
        text.push_str(&format!("line {n}\n"));
        n += 1;
    }
    text.truncate(bytes);
    text
}

/// An agent on a model with an input limit of `limit` tokens, offered
/// `names` (and `complete_task`), working in `directory`.
fn harness(
    limit: usize,
    names: &[&str],
    directory: &std::path::Path,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let mut config = super::tests::test_config();
    config.llm_config.model = ai::Model {
        max_input_tokens: limit,
        ..ai::Model::openai_gpt_5_6_luna()
    };
    config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    config.tool_config.default_auto_approve_tool_use = true;
    config.tool_config.available_tool_names = names.iter().map(|name| name.to_string()).collect();
    let mut task = Task::new(
        config,
        "Read the data.".into(),
        directory.to_path_buf(),
        None,
    );
    task.set_state(TaskState::InProgress);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(DumpTool));
    let (events, receiver) = mpsc::channel(500);
    let (sender, commands) = mpsc::channel(10);
    (
        AgentLoop::new(
            task,
            Arc::new(ai::LLMClient::new()),
            Arc::new(registry),
            events,
            commands,
            None,
            false,
            None,
            None,
            None,
        ),
        receiver,
        sender,
    )
}

/// A response with `input_tokens` of input that calls `calls` (id, tool,
/// arguments). A small input makes the task's calibrated ratio its
/// largest, 4 bytes per token, so the estimates below are predictable.
fn respond(calls: &[(&str, &str, Value)], input_tokens: usize) -> ScriptedStreamResult {
    scripted("", calls, input_tokens, 10)
}

/// A `handover_report` call with `text`.
fn report(text: &str, input_tokens: usize) -> ScriptedStreamResult {
    respond(
        &[(
            "report",
            HANDOVER_REPORT_TOOL_NAME,
            json!({ "report": text }),
        )],
        input_tokens,
    )
}

fn scripted(
    text: &str,
    calls: &[(&str, &str, Value)],
    input_tokens: usize,
    output_tokens: usize,
) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant(text);
    for (id, name, arguments) in calls {
        message.tool_calls.push(ai::ToolCall {
            id: Some(id.to_string()),
            name: name.to_string(),
            arguments: arguments.clone(),
            thought_signature: None,
            provider_id: None,
            invalid_arguments: None,
        });
    }
    let stop_reason = if message.has_tool_calls() {
        ai::StopReason::ToolUse
    } else {
        ai::StopReason::EndTurn
    };
    let text = message_text(&message);
    let end = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(input_tokens, output_tokens),
        model_call_usage: ai::ModelCallUsage::new(
            "results".into(),
            "test".into(),
            "test".into(),
            input_tokens,
            output_tokens,
            1000,
        ),
        stop_reason,
        refusal: None,
        openai_turn_state: None,
    }));
    let mut events = Vec::new();
    if !text.is_empty() {
        events.push(Ok(ai::StreamEvent::TextChunk(text)));
    }
    events.push(Ok(end));
    Ok(futures::stream::iter(events).boxed())
}

fn complete() -> ScriptedStreamResult {
    respond(
        &[("done", "complete_task", json!({"summary": "done"}))],
        100,
    )
}

/// The provider's answer to a request over the model's input limit
/// (OpenAI's `context_length_exceeded`).
fn overflow() -> ScriptedStreamResult {
    Err(ai::Error::ProviderError {
        message: r#"OpenAI request failed: {"error":{"message":"Your input exceeds the context window of this model. Please adjust your input and try again.","type":"invalid_request_error","param":"input","code":"context_length_exceeded"}}"#.into(),
        status: Some(400),
        code: Some("context_length_exceeded".into()),
        retry_after: None,
    })
}

fn push(agent: &mut AgentLoop, responses: Vec<ScriptedStreamResult>) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend(responses);
}

async fn run(agent: AgentLoop, events: &mut mpsc::Receiver<AgentEvent>) -> (Task, Vec<AgentEvent>) {
    let outcome = timeout(Duration::from_secs(10), agent.run()).await;
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    let task = outcome
        .unwrap_or_else(|error| panic!("{error}: {received:#?}"))
        .unwrap();
    (task, received)
}

/// Each tool result in the conversation, by call id.
fn results(task: &Task) -> HashMap<String, String> {
    task.conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_tool_result)
        .map(|result| (result.call_id.clone().unwrap(), result.output.clone()))
        .collect()
}

fn shortened(events: &[AgentEvent]) -> Vec<(Option<String>, ShortenReason)> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolResultShortened {
                call_id, reason, ..
            } => Some((call_id.clone(), *reason)),
            _ => None,
        })
        .collect()
}

fn workspace() -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!("arko-results-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A turn whose results would certainly put the next request over the
/// model's input limit: the largest is shortened to a head-and-tail preview with a
/// notice, only as far as needed, and the parallel results that fit are
/// left exactly as they were.
#[tokio::test]
async fn the_guard_shortens_the_largest_results_only_as_far_as_needed() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(20_000, &["dump"], &directory);
    push(
        &mut agent,
        vec![
            respond(
                &[
                    ("huge", "dump", json!({"bytes": 200_000})),
                    ("medium", "dump", json!({"bytes": 30_000})),
                    ("small", "dump", json!({"bytes": 300})),
                ],
                100,
            ),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let results = results(&task);
    // 20,000 tokens, less the 4,000-token margin: the huge result (about
    // 50,000 tokens) keeps a 4,000-token preview, about 16,000 bytes.
    let huge = &results["huge"];
    assert!(huge.len() > 15_000 && huge.len() < 17_500, "{}", huge.len());
    assert!(huge.starts_with("line 1\nline 2\n"), "{huge:.100}");
    assert!(
        huge.contains(" lines ("),
        "the marker between head and tail"
    );
    assert!(
        huge.contains("[Shortened: this result is about 50,000 tokens (195 KB, "),
        "{}",
        &huge[huge.len() - 400..]
    );
    assert!(huge.ends_with("Ask for less: a smaller file, a narrower query, or fewer results.]"));
    assert!(!huge.contains("terminal") && !huge.contains("start_line"));
    // The others fit once the largest is shortened.
    assert_eq!(results["medium"], lines(30_000));
    assert_eq!(results["small"], lines(300));
    assert_eq!(
        shortened(&events),
        [(Some("huge".to_string()), ShortenReason::Overflow)]
    );
    // The event log keeps the whole output.
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolResult { output, .. } if output.len() == 200_000
    )));
    std::fs::remove_dir_all(directory).unwrap();
}

/// A task saved after its reply asked for a tool is estimated from its
/// saved usage: its latest request is the conversation before the reply,
/// and the turn's results are guarded against what that request reported.
#[tokio::test]
async fn a_saved_request_estimates_the_next_one() {
    let directory = workspace();
    let (mut agent, _events, _commands) = harness(20_000, &["dump"], &directory);
    agent.task.conversation.add_user_message("Read the data.");
    agent.task.conversation.messages.push(call_reply("data"));
    agent
        .task
        .usage_history
        .push(ai::UsageInfo::new(15_000, 10));
    let mark = agent.saved_request_mark().unwrap();
    assert_eq!(mark.messages, 1);
    agent.answered_request = Some(mark);
    // 15,000 of 20,000 tokens used: a result of 60,000 bytes (at least
    // 6,000 tokens) certainly doesn't fit.
    agent
        .task
        .pending_native_tool_results
        .push(ai::ToolResult::new(
            Some("data".into()),
            "dump",
            &lines(60_000),
        ));
    agent.limit_tool_results().await.unwrap();
    let output = &agent.task.pending_native_tool_results[0].output;
    assert!(
        output.contains("[Shortened: this result is about"),
        "{output}"
    );
    assert!(output.len() < 10_000, "{}", output.len());
    // A fresh task has no saved request.
    let (mut fresh, _events, _commands) = harness(20_000, &["dump"], &directory);
    assert!(fresh.saved_request_mark().is_none());
    std::fs::remove_dir_all(directory).unwrap();
}

/// A reply calling `dump`, with call id `id`.
fn call_reply(id: &str) -> ai::Message {
    let mut reply = ai::Message::new_assistant("");
    reply.tool_calls.push(ai::ToolCall {
        id: Some(id.into()),
        name: "dump".into(),
        arguments: json!({}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    });
    reply
}

/// A task run again (saved with a turn's results added, before its next
/// request) estimates that request from its saved usage, as the run that
/// made it would: the saved request reported 15,000 tokens for few bytes
/// (images, replayed reasoning), so the 30,000 bytes of results added
/// since take the conversation over the handover threshold.
#[tokio::test]
async fn a_task_run_again_estimates_from_its_saved_usage() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(25_000));
    agent.task.conversation.add_user_message("Read the data.");
    agent.task.conversation.messages.push(call_reply("data"));
    agent
        .task
        .conversation
        .add_tool_results(vec![ai::ToolResult::new(
            Some("data".into()),
            "dump",
            &lines(30_000),
        )]);
    agent
        .task
        .usage_history
        .push(ai::UsageInfo::new(15_000, 10));
    push(
        &mut agent,
        vec![report("REPORT: read the data.", 16_000), complete()],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        events.iter().any(|event| matches!(
            event,
            AgentEvent::HandoverRequested {
                input_tokens: 15_000
            }
        )),
        "{events:#?}"
    );
    assert_eq!(task.handovers.len(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}

/// Results that may fit are sent as they are, however full the context:
/// a small result in a nearly full context (it was once replaced by a
/// larger notice), and a large file that fits by the provider's count
/// though a calibrated estimate says it doesn't (365,873 bytes of Rust,
/// about 76,000 o200k tokens, after a request of 184,944 tokens, under the
/// 272,000-token limit of ChatGPT OAuth).
#[tokio::test]
async fn results_that_may_fit_are_sent_whole() {
    let directory = workspace();
    for (input, small, large) in [(267_000, 40, 3_000), (184_944, 300, 365_873)] {
        let (mut agent, mut events, _commands) = harness(272_000, &["dump"], &directory);
        push(
            &mut agent,
            vec![
                respond(
                    &[
                        ("small", "dump", json!({"bytes": small})),
                        ("large", "dump", json!({"bytes": large})),
                    ],
                    input,
                ),
                complete(),
            ],
        );
        let (task, events) = run(agent, &mut events).await;
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        let results = results(&task);
        assert_eq!(results["small"], lines(small));
        assert_eq!(results["large"], lines(large));
        assert!(shortened(&events).is_empty(), "{input}");
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// The handover check counts the model's reply (its output tokens, which
/// include reasoning the next request replays), not only the results.
#[tokio::test]
async fn the_handover_check_counts_the_models_reply() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(10_000));
    push(
        &mut agent,
        vec![
            scripted("", &[("data", "dump", json!({"bytes": 300}))], 5_000, 6_000),
            report("REPORT: read the data.", 11_000),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::HandoverRequested {
            input_tokens: 5_000
        }
    )));
    assert_eq!(task.handovers.len(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}

/// Line ranges passed to `read_file` while they are off are ignored: the
/// call reads the whole file. On, they read only those lines.
#[tokio::test]
async fn read_file_line_ranges_passed_while_off_are_ignored() {
    let directory = workspace();
    std::fs::write(directory.join("data.txt"), lines(200)).unwrap();
    for ranges in [false, true] {
        let (mut agent, mut events, _commands) = harness(1_000_000, &["read_file"], &directory);
        agent.task.agent_config.tool_config.read_file_line_ranges = ranges;
        push(
            &mut agent,
            vec![
                respond(
                    &[(
                        "read",
                        "read_file",
                        json!({"path": "data.txt", "start_line": 5, "end_line": 6}),
                    )],
                    100,
                ),
                complete(),
            ],
        );
        let (task, _) = run(agent, &mut events).await;
        let read = &results(&task)["read"];
        if ranges {
            assert!(read.starts_with("5 | line 5\n6 | line 6\n"), "{read}");
            assert!(!read.contains("1 | line 1\n"), "{read}");
        } else {
            assert!(read.starts_with("1 | line 1\n"), "{read}");
            assert!(read.contains("20 | line 20\n"), "{read}");
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// Recovery shortens the largest result added since the last answered
/// request, not the newest.
#[tokio::test]
async fn overflow_recovery_shortens_the_largest_new_result() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    push(
        &mut agent,
        vec![
            respond(
                &[
                    ("big", "dump", json!({"bytes": 400_000})),
                    ("newest", "dump", json!({"bytes": 40_000})),
                ],
                100,
            ),
            overflow(),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let results = results(&task);
    assert!(results["big"].contains("[Shortened:"));
    assert_eq!(results["newest"], lines(40_000));
    assert_eq!(
        shortened(&events),
        [(Some("big".to_string()), ShortenReason::OverflowRecovery)]
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// "Twice in a row": an answered request starts the count again, so a
/// later turn's overflow gets its own two resends.
#[tokio::test]
async fn the_overflow_count_starts_again_after_an_answer() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    // Recovery that runs out interrupts the task, rather than pausing it.
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    push(
        &mut agent,
        vec![
            respond(&[("first", "dump", json!({"bytes": 400_000}))], 100),
            overflow(),
            respond(&[("second", "dump", json!({"bytes": 400_000}))], 100),
            overflow(),
            overflow(),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(shortened(&events).len(), 3);
    std::fs::remove_dir_all(directory).unwrap();
}

/// Overflow resends are not ordinary retry attempts: a failure after them
/// still has the policy's attempts.
#[tokio::test]
async fn overflow_resends_dont_use_retry_attempts() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenInformAgent { max_attempts: 2 };
    agent.task.agent_config.llm_retry_config.initial_delay_ms = 1;
    agent.task.agent_config.llm_retry_config.max_delay_ms = 1;
    push(
        &mut agent,
        vec![
            respond(&[("huge", "dump", json!({"bytes": 400_000}))], 100),
            overflow(),
            Err(super::retry_tests::provider_failure(500, None)),
            complete(),
        ],
    );
    let (task, _) = run(agent, &mut events).await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// With the guard off, every result goes into the conversation whole, as
/// before the guard existed.
#[tokio::test]
async fn with_the_guard_off_results_go_in_whole() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(20_000, &["dump"], &directory);
    agent.task.agent_config.tool_config.result_limits =
        ToolResultLimits::default().without_overflow_guard();
    push(
        &mut agent,
        vec![
            respond(&[("huge", "dump", json!({"bytes": 200_000}))], 100),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert_eq!(results(&task)["huge"], lines(200_000));
    assert!(shortened(&events).is_empty());
    std::fs::remove_dir_all(directory).unwrap();
}

/// The notice names terminal commands on the file when the agent has a
/// terminal, read_file's line ranges when they are on and it has none, and
/// otherwise asks for less. read_file's description points to the terminal
/// for large files when a terminal is offered.
#[tokio::test]
async fn the_notice_points_to_the_terminal_or_to_line_ranges_or_asks_for_less() {
    let directory = workspace();
    std::fs::write(directory.join("big.log"), lines(200_000)).unwrap();
    for (names, ranges, expected, absent) in [
        (
            &["read_file", "terminal_start"][..],
            false,
            "Look at the parts you need with the terminal, for example: wc -l big.log; grep -n 'pattern' big.log; head -n 100 big.log; sed -n '5000,5400p' big.log; or a short script.]",
            "start_line",
        ),
        (
            &["read_file"][..],
            true,
            r#"Read the other parts with read_file's start_line and end_line, for example {"path": "big.log", "start_line": 5000, "end_line": 5400}.]"#,
            "terminal",
        ),
        (
            &["read_file"][..],
            false,
            "Ask for less: a smaller file, a narrower query, or fewer results.]",
            "start_line",
        ),
    ] {
        let (mut agent, mut events, _commands) = harness(20_000, names, &directory);
        agent.task.agent_config.tool_config.read_file_line_ranges = ranges;
        let model = agent.task.conversation.model.clone();
        let offered = agent.build_native_tool_config(&model, None);
        let description = &offered
            .tools
            .iter()
            .find(|tool| tool.name == "read_file")
            .unwrap()
            .description;
        assert_eq!(
            description.contains(read_file::TERMINAL_HINT),
            names.contains(&"terminal_start"),
            "{description}"
        );
        assert_eq!(description.contains("start_line"), ranges, "{description}");
        push(
            &mut agent,
            vec![
                respond(&[("read", "read_file", json!({"path": "big.log"}))], 100),
                complete(),
            ],
        );
        let (task, _) = run(agent, &mut events).await;
        let read = &results(&task)["read"];
        assert!(read.starts_with("1 | line 1\n"), "{read:.100}");
        assert!(read.ends_with(expected), "{}", &read[read.len() - 500..]);
        let notice = &read[read.rfind("[Shortened:").unwrap()..];
        assert!(!notice.contains(absent), "{notice}");
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// Opt-in caps: the agent's cap applies to every tool unless a tool's own
/// setting replaces it (null for none), whatever the input limit.
#[tokio::test]
async fn caps_shorten_results_over_them_with_the_tools_own_setting_first() {
    let directory = workspace();
    std::fs::write(directory.join("data.txt"), lines(40_000)).unwrap();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump", "read_file"], &directory);
    let tool_config = &mut agent.task.agent_config.tool_config;
    tool_config.result_limits = ToolResultLimits::default().max_result_tokens(1_000);
    tool_config
        .tool_specific_configs
        .entry("read_file".into())
        .or_default()
        .max_result_tokens = Some(None);
    push(
        &mut agent,
        vec![
            respond(
                &[
                    ("dump", "dump", json!({"bytes": 40_000})),
                    ("read", "read_file", json!({"path": "data.txt"})),
                ],
                100,
            ),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    let results = results(&task);
    let dump = &results["dump"];
    assert!(dump.len() < 5_000, "{}", dump.len());
    assert!(
        dump.contains("over the host's limit of 1,000 tokens per tool result."),
        "{dump}"
    );
    assert!(!results["read"].contains("[Shortened:"));
    assert_eq!(
        shortened(&events),
        [(Some("dump".to_string()), ShortenReason::ResultCap)]
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// The handover check estimates the request about to be sent: a turn
/// whose results take the conversation over the threshold hands over
/// before that request, though the latest request was under it. A turn
/// with small results doesn't.
#[tokio::test]
async fn the_handover_check_counts_the_turns_results() {
    let directory = workspace();
    for (bytes, handover) in [(60_000, true), (300, false)] {
        let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
        agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(10_000));
        let mut responses = vec![respond(&[("data", "dump", json!({"bytes": bytes}))], 5_000)];
        if handover {
            responses.push(report("REPORT: read the data.", 6_000));
        }
        responses.push(complete());
        push(&mut agent, responses);
        let (task, events) = run(agent, &mut events).await;
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        let requested: Vec<usize> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::HandoverRequested { input_tokens } => Some(*input_tokens),
                _ => None,
            })
            .collect();
        if handover {
            // Recorded with the latest request's reported size, as before.
            assert_eq!(requested, [5_000]);
            assert_eq!(task.handovers.len(), 1);
            assert_eq!(task.handovers[0].report, "REPORT: read the data.");
        } else {
            assert!(requested.is_empty(), "{requested:?}");
            assert!(task.handovers.is_empty());
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// The provider still finds the request over the model's input limit: the
/// largest result added since the last answered request is shortened and
/// the request sent again, twice at most (a preview, then the notice
/// alone); then the task pauses on the error, its kind recorded, and a
/// rejected recovery fails it with that kind.
#[tokio::test]
async fn an_overflow_is_recovered_from_twice_at_most_then_fails_with_its_kind() {
    let directory = workspace();
    let (mut agent, mut events, commands) = harness(1_000_000, &["dump"], &directory);
    push(
        &mut agent,
        vec![
            respond(&[("huge", "dump", json!({"bytes": 400_000}))], 100),
            overflow(),
            overflow(),
            overflow(),
        ],
    );
    let runner = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        seen.push(event.clone());
        if let AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
            cause,
            error_kind,
            ..
        }) = &event
        {
            assert_eq!(cause, &ErrorRecoveryCause::NonRetryable);
            assert_eq!(error_kind, &Some(ai::ErrorKind::ContextOverflow));
            commands
                .send(UserCommand::RejectErrorRecovery { message: None })
                .await
                .unwrap();
            break;
        }
    }
    let task = timeout(Duration::from_secs(10), runner)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        matches!(
            task.current_state,
            TaskState::Failed {
                error_kind: Some(ai::ErrorKind::ContextOverflow),
                ..
            }
        ),
        "{:?}",
        task.current_state
    );
    assert_eq!(
        shortened(&seen),
        [
            (Some("huge".to_string()), ShortenReason::OverflowRecovery),
            (Some("huge".to_string()), ShortenReason::OverflowRecovery),
        ]
    );
    // Each resend is reported as an error that needs no decision, with its
    // kind.
    let resends = seen
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ErrorEncountered {
                    requires_approval_to_retry: false,
                    error_kind: Some(ai::ErrorKind::ContextOverflow),
                    ..
                }
            )
        })
        .count();
    assert_eq!(resends, 2);
    // The second attempt left the notice alone.
    let huge = &results(&task)["huge"];
    assert!(
        huge.starts_with("[Shortened: this result is about"),
        "{huge}"
    );
    assert!(huge.ends_with("Ask for less: a smaller file, a narrower query, or fewer results.]"));
    std::fs::remove_dir_all(directory).unwrap();
}

/// One shortening is enough when the provider then accepts the request.
/// With the guard off, the overflow is not recovered from: the task pauses
/// at once, with the error's kind.
#[tokio::test]
async fn an_overflow_recovered_once_continues_and_none_without_the_guard() {
    let directory = workspace();
    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    push(
        &mut agent,
        vec![
            respond(&[("huge", "dump", json!({"bytes": 400_000}))], 100),
            overflow(),
            complete(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let huge = &results(&task)["huge"];
    // A 16,000-token preview of the 1,000,000-token limit, at 4 bytes a
    // token.
    assert!(huge.len() > 60_000 && huge.len() < 70_000, "{}", huge.len());
    assert_eq!(
        shortened(&events),
        [(Some("huge".to_string()), ShortenReason::OverflowRecovery)]
    );

    let (mut agent, mut events, _commands) = harness(1_000_000, &["dump"], &directory);
    agent.task.agent_config.tool_config.result_limits =
        ToolResultLimits::default().without_overflow_guard();
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    push(
        &mut agent,
        vec![
            respond(&[("huge", "dump", json!({"bytes": 400_000}))], 100),
            overflow(),
        ],
    );
    let (task, events) = run(agent, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert_eq!(results(&task)["huge"], lines(400_000));
    assert!(shortened(&events).is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered {
            error_kind: Some(ai::ErrorKind::ContextOverflow),
            ..
        }
    )));
    std::fs::remove_dir_all(directory).unwrap();
}
