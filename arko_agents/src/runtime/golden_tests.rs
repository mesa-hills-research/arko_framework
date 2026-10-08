//! Golden scenarios of the agent loop, for paths the other tests reach
//! loosely or not at all. Each runs the loop on scripted responses and
//! commands, and compares its trace (see
//! [`loop_trace`](super::loop_trace)) and the conversation each run leaves
//! with `loop_golden/<test>.trace`. `ARKO_LOOP_GOLDEN=bless` writes the
//! files instead, and prints them as `@@golden <test> <line>` lines for a
//! run on another machine. A blessed file records what the loop does now:
//! review it by hand.

use super::long_run_tests::turn_with_plan_usage;
use super::loop_trace::{self, TestTrace};
use super::retry_tests::{model_refusal, refused_response};
use super::tool_call_tests::{complete, config, native_call, text_turn, turn};
use super::*;
use crate::agent_config::{
    CacheMonitorConfig, MonitoringConfig, PerToolConfigOverride, PlanUsageThreshold,
};
use crate::task::{RefusalResolution, RunningToolCall};
use crate::tool::{ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;
use futures::StreamExt;
use std::io::Write as _;
use std::path::PathBuf;

/// Every scenario's workspace: a path that is never created (no tool here
/// touches the file system), written the same on every platform.
const WORKSPACE: &str = "/arko-golden-workspace";

/// The longest string a golden conversation keeps whole.
const LONGEST_MESSAGE_STRING: usize = 4_000;

/// A scenario's expected output: the trace and conversation of each run.
struct Golden {
    name: &'static str,
    trace: Arc<TestTrace>,
    output: Vec<String>,
    /// The trace lines already in `output`.
    taken: usize,
}

impl Golden {
    /// Traces the loops this test creates from now on.
    fn start(name: &'static str) -> Self {
        Self {
            name,
            trace: loop_trace::record(),
            output: Vec::new(),
            taken: 0,
        }
    }

    /// A run ended: its trace, then the conversation its task holds.
    fn run_ended(&mut self, task: &Task) {
        let lines = self.trace.lines();
        self.output.extend(lines[self.taken..].iter().cloned());
        self.taken = lines.len();
        let system = task
            .conversation
            .system_message
            .as_deref()
            .unwrap_or_default();
        self.output.push(format!(
            "--- conversation: {} messages, system prompt {:016x}",
            task.conversation.messages.len(),
            loop_trace::fnv1a(system.as_bytes())
        ));
        for message in &task.conversation.messages {
            let message = serde_json::to_value(message).unwrap();
            self.output.push(format!(
                "M {}",
                self.trace.normalized(message, LONGEST_MESSAGE_STRING)
            ));
        }
    }

    /// Compares the output with the scenario's file, or writes the file.
    fn check(self) {
        assert_eq!(
            self.trace.lines().len(),
            self.taken,
            "every run ends with run_ended"
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/runtime/loop_golden")
            .join(format!("{}.trace", self.name));
        let actual = self.output.join("\n") + "\n";
        if std::env::var("ARKO_LOOP_GOLDEN").is_ok_and(|mode| mode == "bless") {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &actual).unwrap();
            self.print();
            return;
        }
        let expected = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        if expected == actual {
            return;
        }
        self.print();
        let expected: Vec<&str> = expected.lines().collect();
        let line = expected
            .iter()
            .zip(&self.output)
            .position(|(expected, actual)| expected != actual)
            .unwrap_or(expected.len().min(self.output.len()));
        panic!(
            "{} differs from {} at line {}:\nexpected: {}\nactual:   {}\n(ARKO_LOOP_GOLDEN=bless rewrites the file)",
            self.name,
            path.display(),
            line + 1,
            expected.get(line).unwrap_or(&"<end>"),
            self.output.get(line).map_or("<end>", String::as_str)
        );
    }

    /// The output as `@@golden` lines, past libtest's capture.
    fn print(&self) {
        let mut stdout = std::io::stdout().lock();
        for line in &self.output {
            let _ = writeln!(stdout, "@@golden {} {line}", self.name);
        }
    }
}

/// A tool whose calls answer at once, `<name> ran <what>`, unless `what`
/// asks otherwise: `fail` fails, `big` answers 400,000 bytes, `hang` runs
/// until the call is dropped, and `wait` runs until the task's stop signal
/// fires. A `Resume` tool's recovered call answers `<name> resumed <what>`.
#[derive(Debug, Clone)]
struct Step {
    name: &'static str,
    parallel_safe: bool,
    interruptible: bool,
    recovery: ToolRecovery,
}

impl Step {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            parallel_safe: false,
            interruptible: true,
            recovery: ToolRecovery::InDoubt,
        }
    }

    fn parallel_safe(self) -> Self {
        Self {
            parallel_safe: true,
            ..self
        }
    }

    fn not_interruptible(self) -> Self {
        Self {
            interruptible: false,
            ..self
        }
    }

    fn recovery(self, recovery: ToolRecovery) -> Self {
        Self { recovery, ..self }
    }
}

#[async_trait]
impl Tool for Step {
    fn definition(&self) -> ToolDefinition {
        let parameter = |name: &str, is_path: bool| ToolParameterDefinition {
            name: name.into(),
            description: format!("The {name}"),
            param_type: ToolParameterType::String,
            required: false,
            is_path,
            flag: None,
            default_value: None,
        };
        ToolDefinition {
            name: self.name.into(),
            description: format!("Does the {} step", self.name),
            output_based_example: None,
            parameters: vec![parameter("what", false), parameter("path", true)],
            parallel_safe: self.parallel_safe,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    fn interruptible(&self) -> bool {
        self.interruptible
    }

    fn recovery_mode(&self) -> ToolRecovery {
        self.recovery
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let what = params
            .get("what")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match what {
            "fail" => Err(ToolError::ExecutionFailed(format!(
                "{} could not do it",
                self.name
            ))),
            "big" => Ok((1..)
                .map(|line| format!("line {line}\n"))
                .flat_map(|line| line.into_bytes())
                .take(400_000)
                .map(char::from)
                .collect()),
            "hang" => std::future::pending().await,
            "wait" => {
                context.stop.stopping().await;
                Ok(format!(
                    "{} wound down: {}",
                    self.name,
                    context.stop.reason().unwrap_or_default()
                ))
            }
            _ if context.recovering && self.recovery == ToolRecovery::Resume => {
                Ok(format!("{} resumed {what}", self.name))
            }
            _ => Ok(format!("{} ran {what}", self.name)),
        }
    }
}

fn registry(tools: Vec<Step>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for tool in tools {
        registry.register(Arc::new(tool));
    }
    registry
}

/// An auto-approving agent offered `tools` and `complete_task`, whose
/// retries wait exact delays.
fn golden_config(mode: ToolCallingMode, tools: &[&str]) -> AgentConfig {
    let mut config = config(mode, tools);
    config.llm_retry_config.jitter = false;
    config
}

/// Calls to `tool` wait for approval.
fn needs_approval(config: &mut AgentConfig, tool: &str) {
    config.tool_config.tool_specific_configs.insert(
        tool.into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
}

fn new_task(config: AgentConfig) -> Task {
    Task::new(config, "Do the task".into(), PathBuf::from(WORKSPACE), None)
}

fn golden_loop(
    task: Task,
    tools: Vec<Step>,
    simplified_tool_events: bool,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    golden_loop_with(task, registry(tools), simplified_tool_events)
}

fn golden_loop_with(
    task: Task,
    registry: ToolRegistry,
    simplified_tool_events: bool,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let (events, receiver) = mpsc::channel(500);
    let (sender, commands) = mpsc::channel(10);
    let agent = AgentLoop::new(
        task,
        Arc::new(ai::LLMClient::new()),
        Arc::new(registry),
        events,
        commands,
        None,
        simplified_tool_events,
        None,
        None,
        None,
    );
    (agent, receiver, sender)
}

/// The task as a host would load it again: through its saved form.
fn reloaded(task: &Task) -> Task {
    serde_json::from_str(&serde_json::to_string(task).unwrap()).unwrap()
}

fn script(agent: &mut AgentLoop, responses: impl IntoIterator<Item = ScriptedStreamResult>) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend(responses);
}

/// Runs `agent` until it ends, sending the commands `react` answers each
/// event with, in order, and returns its task.
async fn drive(
    agent: AgentLoop,
    mut events: mpsc::Receiver<AgentEvent>,
    commands: mpsc::Sender<UserCommand>,
    mut react: impl FnMut(&AgentEvent) -> Vec<UserCommand>,
) -> Task {
    let run = tokio::spawn(agent.run());
    while let Some(event) = timeout(Duration::from_secs(3_600), events.recv())
        .await
        .expect("the run stalled")
    {
        for command in react(&event) {
            // The run may have ended already.
            let _ = commands.send(command).await;
        }
    }
    run.await.unwrap().unwrap()
}

/// Runs `agent` until it ends, closing its command channel (as a host that
/// stopped) at the first event `close` accepts, and returns its task.
async fn drive_closing(
    agent: AgentLoop,
    mut events: mpsc::Receiver<AgentEvent>,
    commands: mpsc::Sender<UserCommand>,
    close: impl Fn(&AgentEvent) -> bool,
) -> Task {
    let run = tokio::spawn(agent.run());
    let mut commands = Some(commands);
    while let Some(event) = timeout(Duration::from_secs(3_600), events.recv())
        .await
        .expect("the run stalled")
    {
        if close(&event) {
            drop(commands.take());
        }
    }
    run.await.unwrap().unwrap()
}

fn no_commands(_: &AgentEvent) -> Vec<UserCommand> {
    Vec::new()
}

fn step_call(id: &str, tool: &str, what: &str) -> ai::ToolCall {
    native_call(id, tool, serde_json::json!({ "what": what }))
}

fn stream(events: Vec<ai::StreamEvent>) -> ScriptedStreamResult {
    Ok(futures::stream::iter(events.into_iter().map(Ok)).boxed())
}

/// A native turn calling `calls`, with `usage`.
fn usage_turn(calls: Vec<ai::ToolCall>, usage: ai::UsageInfo) -> ScriptedStreamResult {
    plan_usage_turn(calls, usage, None)
}

/// A native turn calling `calls`, with `usage` and the plan usage its
/// provider reported.
fn plan_usage_turn(
    calls: Vec<ai::ToolCall>,
    usage: ai::UsageInfo,
    plan_usage: Option<ai::PlanUsage>,
) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls = calls;
    let mut model_call_usage =
        ai::ModelCallUsage::new("golden".into(), "test".into(), "test".into(), 0, 0, 1000);
    model_call_usage.plan_usage = plan_usage;
    stream(vec![ai::StreamEvent::End(Box::new(
        ai::GenerationResponse {
            message,
            usage,
            model_call_usage,
            stop_reason: ai::StopReason::ToolUse,
            refusal: None,
            openai_turn_state: None,
        },
    ))])
}

/// A turn of text without tool calls, streamed.
fn reply(text: &str) -> ScriptedStreamResult {
    stream(vec![
        ai::StreamEvent::TextChunk(text.into()),
        ai::StreamEvent::End(Box::new(ai::GenerationResponse {
            message: ai::Message::new_assistant(text),
            usage: ai::UsageInfo::new(0, 0),
            model_call_usage: ai::ModelCallUsage::new(
                "golden".into(),
                "test".into(),
                "test".into(),
                0,
                0,
                1000,
            ),
            stop_reason: ai::StopReason::EndTurn,
            refusal: None,
            openai_turn_state: None,
        })),
    ])
}

fn approve() -> ToolApprovalDecision {
    ToolApprovalDecision::Approve {
        custom_tool_output: None,
        message_to_agent: None,
    }
}

fn approve_all(batch_id: Uuid, pending: &[PendingToolCallInfo]) -> UserCommand {
    UserCommand::SubmitBatchToolApprovals {
        batch_id,
        decisions: pending
            .iter()
            .map(|call| (call.tool_call_id, approve()))
            .collect(),
    }
}

fn is_pause(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::ErrorEncountered {
            requires_approval_to_retry: true,
            ..
        }
    )
}

/// A native turn [read, gate, write, read]: the first read runs at once,
/// the gate awaits approval, and the write and the read after it are held
/// (the write waits for every earlier call, and the read for the earlier
/// calls that aren't parallel-safe). Once the gate is approved it runs,
/// then the held calls, in call order, and the turn's results go into one
/// message in call order.
#[tokio::test(start_paused = true)]
async fn native_held_calls_run_after_the_approval() {
    let mut golden = Golden::start("native_held_calls_run_after_the_approval");
    let mut config = golden_config(ToolCallingMode::Native, &["read", "gate", "write"]);
    needs_approval(&mut config, "gate");
    let tools = vec![
        Step::new("read").parallel_safe(),
        Step::new("gate"),
        Step::new("write"),
    ];
    let (mut agent, events, commands) = golden_loop(new_task(config), tools, true);
    script(
        &mut agent,
        [
            turn(vec![
                step_call("r0", "read", "zero"),
                step_call("g1", "gate", "one"),
                step_call("w2", "write", "two"),
                step_call("r3", "read", "three"),
            ]),
            complete(),
        ],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::BatchToolCallAwaitingApproval {
            batch_id,
            pending_calls,
        } => vec![approve_all(*batch_id, pending_calls)],
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// A native batch decided call by call: approved with a message to the
/// agent, approved with the host's own output (the call doesn't run),
/// rejected with a reason, and left without a decision.
#[tokio::test(start_paused = true)]
async fn native_batch_decisions_approve_reject_and_leave_out() {
    let mut golden = Golden::start("native_batch_decisions_approve_reject_and_leave_out");
    let mut config = golden_config(ToolCallingMode::Native, &["gate"]);
    needs_approval(&mut config, "gate");
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("gate")], true);
    script(
        &mut agent,
        [
            turn(vec![
                step_call("g1", "gate", "one"),
                step_call("g2", "gate", "two"),
                step_call("g3", "gate", "three"),
                step_call("g4", "gate", "four"),
            ]),
            complete(),
        ],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::BatchToolCallAwaitingApproval {
            batch_id,
            pending_calls,
        } => {
            let decisions = [
                ToolApprovalDecision::Approve {
                    custom_tool_output: None,
                    message_to_agent: Some("Go ahead.".into()),
                },
                ToolApprovalDecision::Approve {
                    custom_tool_output: Some("The host's own output.".into()),
                    message_to_agent: None,
                },
                ToolApprovalDecision::Reject {
                    reason: Some("Not this one.".into()),
                },
            ];
            vec![UserCommand::SubmitBatchToolApprovals {
                batch_id: *batch_id,
                decisions: pending_calls
                    .iter()
                    .map(|call| call.tool_call_id)
                    .zip(decisions)
                    .collect(),
            }]
        }
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// An output-based batch under `ToolErrorPolicy::PauseAndAwaitApproval`
/// (with the granular parser events): the first call's path is outside
/// the workspace and rejected, the second awaits approval, and once
/// approved it fails. The task pauses once, for both errors, and the
/// results wait with the pause until it is continued.
#[tokio::test(start_paused = true)]
async fn output_based_batch_pauses_once_for_a_path_rejection_and_a_failure() {
    let mut golden =
        Golden::start("output_based_batch_pauses_once_for_a_path_rejection_and_a_failure");
    let mut config = golden_config(ToolCallingMode::OutputBased, &["check"]);
    needs_approval(&mut config, "check");
    config.error_recovery_config.on_tool_execution_error = ToolErrorPolicy::PauseAndAwaitApproval;
    let (mut agent, events, commands) = golden_loop(
        new_task(config),
        vec![Step::new("check").parallel_safe()],
        false,
    );
    script(
        &mut agent,
        [
            text_turn(
                "Checking both.\n<batched_tool_call>\n<check>\n<path>../outside.txt</path>\n</check>\n<check>\n<path>inside.txt</path>\n<what>fail</what>\n</check>\n</batched_tool_call>",
            ),
            text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
        ],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::BatchToolCallAwaitingApproval {
            batch_id,
            pending_calls,
        } => vec![approve_all(*batch_id, pending_calls)],
        event if is_pause(event) => vec![UserCommand::ContinueAfterError { note: None }],
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// An output-based call, then `complete_task`: the task completes without
/// a `TaskStateChanged(Completed)` event (native mode emits one; question
/// Q1 of the loop refactor).
#[tokio::test(start_paused = true)]
async fn output_based_complete_task() {
    let mut golden = Golden::start("output_based_complete_task");
    let config = golden_config(ToolCallingMode::OutputBased, &["step"]);
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], false);
    script(
        &mut agent,
        [
            text_turn("First the step.\n<step>\n<what>one</what>\n</step>"),
            text_turn(
                "All steps are done.\n<complete_task>\n<summary>All done.</summary>\n</complete_task>",
            ),
        ],
    );
    let task = drive(agent, events, commands, no_commands).await;
    golden.run_ended(&task);
    golden.check();
}

/// A host's handover request: the report request, the report turn and the
/// fresh conversation. Then the next response is suspended mid-stream
/// with context arriving just before, its partial text kept, and the
/// suspended task runs again to completion.
#[tokio::test(start_paused = true)]
async fn a_handover_then_a_suspension_mid_stream_then_a_resume() {
    let mut golden = Golden::start("a_handover_then_a_suspension_mid_stream_then_a_resume");
    let config = golden_config(ToolCallingMode::Native, &["step"]);
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], true);
    commands.try_send(UserCommand::RequestHandover).unwrap();
    script(
        &mut agent,
        [
            reply("Report: nothing is done yet; next, do the step."),
            Ok(
                futures::stream::iter([Ok(ai::StreamEvent::TextChunk("Working on it".into()))])
                    .chain(futures::stream::pending())
                    .boxed(),
            ),
        ],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::AssistantResponseChunk(text) if text == "Working on it" => vec![
            UserCommand::InjectContext {
                content: "Sent during the stream.".into(),
            },
            UserCommand::SuspendTask,
        ],
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);

    let (mut agent, events, commands) = golden_loop(reloaded(&task), vec![Step::new("step")], true);
    script(&mut agent, [complete()]);
    let task = drive(agent, events, commands, no_commands).await;
    golden.run_ended(&task);
    golden.check();
}

/// `CancelTask` while a call that isn't interruptible runs, with context
/// sent before and after it: the call runs to its end (its stop signal
/// fires), the cancellation then ends the task and the call after it is
/// answered as not run; the context taken before the cancellation is kept,
/// and the context sent after it is never taken.
#[tokio::test(start_paused = true)]
async fn a_cancellation_waits_for_a_call_that_is_not_interruptible() {
    let mut golden = Golden::start("a_cancellation_waits_for_a_call_that_is_not_interruptible");
    let config = golden_config(ToolCallingMode::Native, &["winds_down", "step"]);
    let tools = vec![
        Step::new("winds_down").not_interruptible(),
        Step::new("step"),
    ];
    let (mut agent, events, commands) = golden_loop(new_task(config), tools, true);
    script(
        &mut agent,
        [turn(vec![
            step_call("w", "winds_down", "wait"),
            step_call("s", "step", "after"),
        ])],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "step" => vec![
            UserCommand::InjectContext {
                content: "Sent before the cancellation.".into(),
            },
            UserCommand::CancelTask,
            UserCommand::InjectContext {
                content: "Sent after the cancellation.".into(),
            },
        ],
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// `SuspendTask` while a call runs: the call is stopped, the call after it
/// is answered as not run, the turn's results go into the conversation,
/// and the loop reports the task's state as it stops. The task then runs
/// again to completion.
#[tokio::test(start_paused = true)]
async fn a_suspension_during_tool_calls_reports_the_state() {
    let mut golden = Golden::start("a_suspension_during_tool_calls_reports_the_state");
    let config = golden_config(ToolCallingMode::Native, &["hangs", "step"]);
    let tools = || vec![Step::new("hangs"), Step::new("step")];
    let (mut agent, events, commands) = golden_loop(new_task(config), tools(), true);
    script(
        &mut agent,
        [turn(vec![
            step_call("h", "hangs", "hang"),
            step_call("s", "step", "after"),
        ])],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "step" => {
            vec![UserCommand::SuspendTask]
        }
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);

    let (mut agent, events, commands) = golden_loop(reloaded(&task), tools(), true);
    script(&mut agent, [complete()]);
    let task = drive(agent, events, commands, no_commands).await;
    golden.run_ended(&task);
    golden.check();
}

/// Three error-recovery decisions: `ApproveErrorRecovery` on a cache
/// monitor's pause without a message (no note reaches the model) and with
/// one (the message does), then `ContinueAfterError` with a note while a
/// native batch's results wait with a pause on its tool error (the note
/// goes beside the results).
#[tokio::test(start_paused = true)]
async fn error_recovery_decisions_after_a_monitor_pause_and_a_tool_error() {
    let mut golden =
        Golden::start("error_recovery_decisions_after_a_monitor_pause_and_a_tool_error");
    for message in [None, Some("The cache is cold after a deploy; keep going.")] {
        let mut config = golden_config(ToolCallingMode::Native, &["step"]);
        config.monitoring = MonitoringConfig {
            cache: Some(CacheMonitorConfig {
                warmup_calls: 1,
                window_calls: 2,
                warn_below: Some(0.8),
                pause_below: Some(0.5),
            }),
            ..Default::default()
        };
        let (mut agent, events, commands) =
            golden_loop(new_task(config), vec![Step::new("step")], true);
        let cached =
            |input: usize, cached: usize| ai::UsageInfo::new(input, 10).with_cached_tokens(cached);
        script(
            &mut agent,
            [
                usage_turn(vec![step_call("a", "step", "one")], cached(1000, 0)),
                usage_turn(vec![step_call("b", "step", "two")], cached(2000, 900)),
                usage_turn(vec![step_call("c", "step", "three")], cached(3000, 200)),
                complete(),
            ],
        );
        let task = drive(agent, events, commands, |event| match event {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. }) => {
                vec![UserCommand::ApproveErrorRecovery {
                    message: message.map(str::to_string),
                }]
            }
            _ => Vec::new(),
        })
        .await;
        golden.run_ended(&task);
    }

    let mut config = golden_config(ToolCallingMode::Native, &["step"]);
    config.error_recovery_config.on_tool_execution_error = ToolErrorPolicy::PauseAndAwaitApproval;
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], true);
    script(
        &mut agent,
        [
            turn(vec![
                step_call("a", "step", "one"),
                step_call("b", "step", "fail"),
            ]),
            complete(),
        ],
    );
    let task = drive(agent, events, commands, |event| {
        if is_pause(event) {
            vec![UserCommand::ContinueAfterError {
                note: Some("Skip the failing step and finish.".into()),
            }]
        } else {
            Vec::new()
        }
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// A refused task prompt: a `ResolveRefusal` that can't be applied is
/// reported and the wait goes on, and one that rephrases the prompt
/// continues the task.
#[tokio::test(start_paused = true)]
async fn refusal_resolutions_that_fail_and_that_work() {
    let mut golden = Golden::start("refusal_resolutions_that_fail_and_that_work");
    let config = golden_config(ToolCallingMode::Native, &["step"]);
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], true);
    script(
        &mut agent,
        [
            stream(vec![refused_response(
                ai::Message::new_assistant(""),
                model_refusal(),
            )]),
            complete(),
        ],
    );
    let mut pauses = 0;
    let task = drive(agent, events, commands, |event| {
        if !is_pause(event) {
            return Vec::new();
        }
        pauses += 1;
        let changes = if pauses == 1 {
            vec![RefusalResolution::RemoveUserMessage { message_index: 7 }]
        } else {
            vec![RefusalResolution::RephraseUserMessage {
                message_index: 0,
                text: "Please do the step.".into(),
            }]
        };
        vec![UserCommand::ResolveRefusal { changes }]
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// Commands that don't fit the wait: an `ApproveToolUse` or a batch
/// decision for another call or batch is ignored without an event, while a
/// `RejectToolUse` for another call and a command of another kind report
/// the state again. The right decisions then end the waits.
#[tokio::test(start_paused = true)]
async fn mismatched_decisions_are_ignored_or_answered_with_the_state() {
    let mut golden = Golden::start("mismatched_decisions_are_ignored_or_answered_with_the_state");
    let mut config = golden_config(ToolCallingMode::Native, &["gate"]);
    needs_approval(&mut config, "gate");
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("gate")], true);
    script(
        &mut agent,
        [
            turn(vec![step_call("g1", "gate", "one")]),
            turn(vec![
                step_call("g2", "gate", "two"),
                step_call("g3", "gate", "three"),
            ]),
            complete(),
        ],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => vec![
            UserCommand::ApproveToolUse {
                tool_call_id: Uuid::new_v4(),
                custom_tool_output: None,
                message_to_agent: None,
            },
            UserCommand::RejectToolUse {
                tool_call_id: Uuid::new_v4(),
                reason: Some("Wrong call.".into()),
            },
            UserCommand::SubmitUserResponse {
                response: "An answer nobody asked for.".into(),
            },
            UserCommand::ApproveToolUse {
                tool_call_id: *tool_call_id,
                custom_tool_output: None,
                message_to_agent: None,
            },
        ],
        AgentEvent::BatchToolCallAwaitingApproval {
            batch_id,
            pending_calls,
        } => vec![
            approve_all(Uuid::new_v4(), pending_calls),
            approve_all(*batch_id, pending_calls),
        ],
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// A request over the model's input limit is sent again twice, each time
/// with the largest new tool result shortened, then the task pauses with
/// the error's kind and the host rejects recovery. Then a plan-usage
/// slow-down, with context sent during its wait joining the request after
/// it.
#[tokio::test(start_paused = true)]
async fn overflow_recovery_then_a_plan_usage_slow_down() {
    let mut golden = Golden::start("overflow_recovery_then_a_plan_usage_slow_down");
    let mut config = golden_config(ToolCallingMode::Native, &["step"]);
    config.llm_config.model = ai::Model {
        max_input_tokens: 1_000_000,
        ..ai::Model::openai_gpt_5_6_luna()
    };
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], true);
    let overflow = || -> ScriptedStreamResult {
        Err(ai::Error::ProviderError {
            message: "Your input exceeds the context window of this model.".into(),
            status: Some(400),
            code: Some("context_length_exceeded".into()),
            retry_after: None,
        })
    };
    script(
        &mut agent,
        [
            usage_turn(
                vec![step_call("huge", "step", "big")],
                ai::UsageInfo::new(100, 10),
            ),
            overflow(),
            overflow(),
            overflow(),
        ],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. }) => {
            vec![UserCommand::RejectErrorRecovery { message: None }]
        }
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);

    let mut config = golden_config(ToolCallingMode::Native, &["step"]);
    config.monitoring = MonitoringConfig {
        plan_usage: Some(PlanUsageThreshold {
            percent: 90.0,
            action: PlanUsageAction::SlowDown { delay_ms: 10_000 },
        }),
        ..Default::default()
    };
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], true);
    script(
        &mut agent,
        [stream(vec![turn_with_plan_usage(95.0)]), complete()],
    );
    let task = drive(agent, events, commands, |event| match event {
        AgentEvent::PlanUsageThresholdReached { .. } => vec![UserCommand::InjectContext {
            content: "Sent during the slow-down.".into(),
        }],
        _ => Vec::new(),
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

fn running(call_id: Option<&str>, tool: &str, what: &str) -> RunningToolCall {
    RunningToolCall {
        tool_call_id: Uuid::new_v4(),
        call_id: call_id.map(str::to_string),
        tool_name: tool.into(),
        parameters: HashMap::from([("what".to_string(), Value::from(what))]),
        custom_output: None,
        message_to_agent: None,
        built_in: false,
        output: None,
    }
}

/// Tasks saved while their calls ran (a host that stopped), run again.
/// Native: an answered call keeps its result, an in-doubt call is
/// answered as restarted, rerun and resumed calls run again, a call that
/// waited for approval and one that never started are answered as not
/// reached; the saved approval wait (as an older version left it) no longer
/// waits. Output-based: the running call runs again, and the turn's
/// results go out with the restart note.
#[tokio::test(start_paused = true)]
async fn tasks_saved_while_calls_ran_settle_them_when_run_again() {
    let mut golden = Golden::start("tasks_saved_while_calls_ran_settle_them_when_run_again");
    let tools = || {
        vec![
            Step::new("writer"),
            Step::new("reader").recovery(ToolRecovery::Rerun),
            Step::new("resumer").recovery(ToolRecovery::Resume),
            Step::new("gate"),
        ]
    };
    let mut config = golden_config(
        ToolCallingMode::Native,
        &["writer", "reader", "resumer", "gate"],
    );
    needs_approval(&mut config, "gate");
    let mut task = new_task(config);
    let answered = step_call("answered", "reader", "zero");
    let gate = step_call("gate", "gate", "g");
    let mut assistant = ai::Message::new_assistant("");
    assistant.tool_calls = vec![
        answered.clone(),
        step_call("write", "writer", "w"),
        step_call("read", "reader", "r"),
        step_call("resume", "resumer", "child"),
        gate.clone(),
        step_call("unreached", "writer", "u"),
    ];
    task.conversation.system_message = Some("The system prompt.".into());
    task.conversation.add_user_message("Do the task");
    task.user_message_count = 1;
    task.conversation.messages.push(assistant);
    task.pending_native_tool_results = vec![ai::ToolResult::from_tool_call(&answered, "kept")];
    task.running_tool_calls = vec![
        running(Some("write"), "writer", "w"),
        running(Some("read"), "reader", "r"),
        running(Some("resume"), "resumer", "child"),
    ];
    let gate_id = Uuid::new_v4();
    task.pending_native_tool_calls.insert(gate_id, gate);
    task.set_state(TaskState::AwaitingBatchToolApproval {
        batch_id: Uuid::new_v4(),
        pending_calls: vec![PendingToolCallInfo {
            tool_call_id: gate_id,
            tool_name: "gate".into(),
            parameters: HashMap::from([("what".to_string(), Value::from("g"))]),
            tool_description: "Does the gate step".into(),
            approval_reason: Default::default(),
        }],
    });
    let (mut agent, events, commands) = golden_loop(reloaded(&task), tools(), true);
    script(&mut agent, [complete()]);
    let task = drive(agent, events, commands, no_commands).await;
    golden.run_ended(&task);

    let config = golden_config(ToolCallingMode::OutputBased, &["writer", "reader"]);
    let mut task = new_task(config);
    task.conversation.system_message = Some("The system prompt.".into());
    task.conversation.add_user_message("Do the task");
    task.user_message_count = 1;
    task.conversation.add_assistant_message(
        "<batched_tool_call>\n<writer>\n<what>w</what>\n</writer>\n<reader>\n<what>r</what>\n</reader>\n</batched_tool_call>",
    );
    task.pending_text_tool_results =
        vec!["Tool writer finished with the following result:\nwriter ran w".into()];
    task.running_tool_calls = vec![running(None, "reader", "r")];
    task.set_state(TaskState::InProgress);
    let (mut agent, events, commands) = golden_loop(reloaded(&task), tools(), false);
    script(
        &mut agent,
        [text_turn(
            "<complete_task>\n<summary>done</summary>\n</complete_task>",
        )],
    );
    let task = drive(agent, events, commands, no_commands).await;
    golden.run_ended(&task);
    golden.check();
}

/// Streams that end without a final response: with partial text and an
/// announced call (the call is rejected, the text kept, and the turn
/// answered as one without a tool call); with nothing (retried as a
/// truncated stream); and with a streamed refusal (handled as a refusal;
/// continuing asks the model to go on from its kept turn).
#[tokio::test(start_paused = true)]
async fn streams_that_end_without_a_final_response() {
    let mut golden = Golden::start("streams_that_end_without_a_final_response");
    let config = || golden_config(ToolCallingMode::Native, &["step"]);
    let partial = stream(vec![
        ai::StreamEvent::TextChunk("Partial answer".into()),
        ai::StreamEvent::ToolCallStart {
            index: 0,
            name: "step".into(),
        },
    ]);
    let refused = stream(vec![
        ai::StreamEvent::TextChunk("I can't do".into()),
        ai::StreamEvent::Refusal(model_refusal()),
    ]);
    for first in [partial, stream(Vec::new()), refused] {
        let (mut agent, events, commands) =
            golden_loop(new_task(config()), vec![Step::new("step")], true);
        script(&mut agent, [first, complete()]);
        let task = drive(agent, events, commands, |event| {
            if is_pause(event) {
                vec![UserCommand::ContinueAfterError { note: None }]
            } else {
                Vec::new()
            }
        })
        .await;
        golden.run_ended(&task);
    }
    golden.check();
}

/// A response stopped mid-stream, after partial text and an announced
/// call: by `CancelTask` (the text is kept, the call rejected, and the loop
/// ends at once, without another pass) and by a closed command channel (the
/// task is interrupted, and the loop ends after one more pass start).
#[tokio::test(start_paused = true)]
async fn stops_during_a_stream() {
    let mut golden = Golden::start("stops_during_a_stream");
    for close in [false, true] {
        let config = golden_config(ToolCallingMode::Native, &["step"]);
        let (mut agent, events, commands) =
            golden_loop(new_task(config), vec![Step::new("step")], true);
        script(
            &mut agent,
            [Ok(futures::stream::iter([
                Ok(ai::StreamEvent::TextChunk("Starting the step".into())),
                Ok(ai::StreamEvent::ToolCallStart {
                    index: 0,
                    name: "step".into(),
                }),
            ])
            .chain(futures::stream::pending())
            .boxed())],
        );
        let announced = |event: &AgentEvent| matches!(event, AgentEvent::ToolCallInitiated { .. });
        let task = if close {
            drive_closing(agent, events, commands, announced).await
        } else {
            drive(agent, events, commands, |event| {
                if announced(event) {
                    vec![UserCommand::CancelTask]
                } else {
                    Vec::new()
                }
            })
            .await
        };
        golden.run_ended(&task);
    }
    golden.check();
}

/// Context once, from the turn context source.
struct ContextOnce(std::sync::atomic::AtomicBool);

impl crate::runtime::TurnContextSource for ContextOnce {
    fn turn_context(&self, _task_id: Uuid) -> Option<String> {
        (!self.0.swap(true, std::sync::atomic::Ordering::SeqCst))
            .then(|| "Context from the source.".to_string())
    }
}

/// The turn boundary of a task run again whose conversation ends on the
/// model's text (a stream suspended mid-way), with saved context, a turn
/// context source and the file tree in diff mode: the saved context and the
/// source's go out as one update, then the file tree's full snapshot, so
/// the request ends on a user turn and needs no request to continue.
#[tokio::test(start_paused = true)]
async fn a_turn_boundary_adds_context_then_the_file_tree() {
    let mut golden = Golden::start("a_turn_boundary_adds_context_then_the_file_tree");
    let workspace = std::env::temp_dir().join(format!("arko-golden-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(workspace.join("notes.txt"), "notes").unwrap();
    std::fs::write(workspace.join("plan.md"), "plan").unwrap();
    golden
        .trace
        .mask(&workspace.to_string_lossy(), "<workspace>");
    let mut config = golden_config(ToolCallingMode::Native, &["step"]);
    let details = &mut config.environment_details_config;
    details.file_tree.include = true;
    details.file_tree_mode = crate::agent_config::FileTreeMode::Diff;
    let mut task = Task::new(config, "Do the task".into(), workspace.clone(), None);
    task.conversation.system_message = Some("The system prompt.".into());
    task.conversation.add_user_message("Do the task");
    task.user_message_count = 1;
    task.conversation
        .add_assistant_message("I was partway through the step");
    task.pending_context_injections = vec!["Saved context.".into()];
    task.set_state(TaskState::InProgress);
    let (mut agent, events, commands) = golden_loop(reloaded(&task), vec![Step::new("step")], true);
    agent.turn_context = Some(Arc::new(ContextOnce(Default::default())));
    script(&mut agent, [complete()]);
    let task = drive(agent, events, commands, no_commands).await;
    golden.run_ended(&task);
    std::fs::remove_dir_all(workspace).unwrap();
    golden.check();
}

/// One response that crosses the plan-usage threshold (`Pause`) and the
/// cache monitor's pause threshold: the boundary pauses for plan usage
/// first, then, once that is approved, for the cache.
#[tokio::test(start_paused = true)]
async fn plan_usage_pauses_before_the_cache_monitor() {
    let mut golden = Golden::start("plan_usage_pauses_before_the_cache_monitor");
    let mut config = golden_config(ToolCallingMode::Native, &["step"]);
    config.monitoring = MonitoringConfig {
        plan_usage: Some(PlanUsageThreshold {
            percent: 90.0,
            action: PlanUsageAction::Pause,
        }),
        cache: Some(CacheMonitorConfig {
            warmup_calls: 1,
            window_calls: 1,
            warn_below: None,
            pause_below: Some(0.5),
        }),
    };
    let (mut agent, events, commands) =
        golden_loop(new_task(config), vec![Step::new("step")], true);
    let plan = ai::PlanUsage {
        plan_type: Some("pro".into()),
        primary: Some(ai::PlanUsageWindow {
            used_percent: 95.0,
            window_minutes: Some(300),
            resets_at: Some(4_102_444_800),
            reset_after_seconds: None,
        }),
        ..Default::default()
    };
    script(
        &mut agent,
        [
            usage_turn(
                vec![step_call("a", "step", "one")],
                ai::UsageInfo::new(1000, 10).with_cached_tokens(0),
            ),
            plan_usage_turn(
                vec![step_call("b", "step", "two")],
                ai::UsageInfo::new(2000, 10).with_cached_tokens(0),
                Some(plan),
            ),
            complete(),
        ],
    );
    let task = drive(agent, events, commands, |event| {
        if is_pause(event) {
            vec![UserCommand::ApproveErrorRecovery { message: None }]
        } else {
            Vec::new()
        }
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}

/// A tool that adds what it observes of the conversation to the trace.
struct Observer(Arc<TestTrace>);

impl std::fmt::Debug for Observer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Observer")
    }
}

#[async_trait]
impl Tool for Observer {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "observer".into(),
            ..Step::new("observer").definition()
        }
    }

    fn reset_context(&self, _task_id: Uuid) {
        self.0.note("observer reset");
    }

    fn observe_context(&self, _task_id: Uuid, message: &ai::Message) {
        self.0.note(&format!(
            "observer saw {:?}: {}",
            message.role,
            message_text(message)
        ));
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        Ok("observed".into())
    }
}

/// Holds the first request.
struct HoldOnce(std::sync::atomic::AtomicBool);

impl crate::runtime::TurnContextSource for HoldOnce {
    fn turn_context(&self, _task_id: Uuid) -> Option<String> {
        None
    }

    fn hold_request(&self, _task_id: Uuid) -> Option<String> {
        (!self.0.swap(true, std::sync::atomic::Ordering::SeqCst))
            .then(|| "Held by the host.".to_string())
    }
}

/// The host holds the first request: the task pauses before the tools
/// observe the conversation, and they observe it once the request is
/// released, right before it is made.
#[tokio::test(start_paused = true)]
async fn tools_observe_the_conversation_after_a_held_request() {
    let mut golden = Golden::start("tools_observe_the_conversation_after_a_held_request");
    let config = golden_config(ToolCallingMode::Native, &["observer"]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Observer(golden.trace.clone())));
    let (mut agent, events, commands) = golden_loop_with(new_task(config), registry, true);
    agent.turn_context = Some(Arc::new(HoldOnce(Default::default())));
    script(&mut agent, [complete()]);
    let task = drive(agent, events, commands, |event| {
        if is_pause(event) {
            vec![UserCommand::ContinueAfterError { note: None }]
        } else {
            Vec::new()
        }
    })
    .await;
    golden.run_ended(&task);
    golden.check();
}
