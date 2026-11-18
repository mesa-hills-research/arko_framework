//! Calls a tool makes from inside its own call ([`NestedCalls`]). Each test
//! fails if a nested call skips the gate: the call would run, or wait, or
//! leave no events.

use super::tool_call_tests::{agent, complete, config, drain, native_call, turn, wait_for, workspace};
use super::*;
use crate::agent_config::PerToolConfigOverride;
use crate::runtime::event::{NESTED_OUTPUT_BYTES, NestedCallOutcome};
use crate::tool::{
    NestedCalls, NestedOutcome, Tool, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use std::sync::Mutex;

/// What the tools of a test record, in order.
#[derive(Debug, Default)]
struct Log(Mutex<Vec<String>>);

impl Log {
    fn push(&self, entry: impl Into<String>) {
        self.0.lock().unwrap().push(entry.into());
    }

    fn entries(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

/// How a [`Target`] call ends.
#[derive(Debug, Clone)]
enum Ends {
    Returns(String),
    Fails,
    Refuses,
    Waits(Duration),
}

/// A tool with a path parameter that logs its calls.
#[derive(Debug)]
struct Target {
    name: &'static str,
    ends: Ends,
    log: Arc<Log>,
}

/// Logs that a call was dropped before it ended.
struct DroppedWhileRunning<'a>(&'a Log, &'a str, bool);

impl Drop for DroppedWhileRunning<'_> {
    fn drop(&mut self) {
        if self.2 {
            self.0.push(format!("dropped {}", self.1));
        }
    }
}

#[async_trait]
impl Tool for Target {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.into(),
            description: "Logs its calls".into(),
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
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        assert!(context.nested.is_none(), "a nested call has no gate");
        self.log.push(format!("start {}", self.name));
        let mut running = DroppedWhileRunning(&self.log, self.name, true);
        let result = match &self.ends {
            Ends::Returns(text) => Ok(text.clone()),
            Ends::Fails => Err(ToolError::ExecutionFailed("it broke".into())),
            Ends::Refuses => Err(ToolError::Refused("the guard said no".into())),
            Ends::Waits(time) => {
                tokio::time::sleep(*time).await;
                Ok("waited".into())
            }
        };
        running.2 = false;
        self.log.push(format!("end {}", self.name));
        result
    }
}

/// A tool that calls the tools its `calls` argument lists (`[name,
/// arguments]` pairs) from inside its call, one after another, or all at
/// once with `together`, and returns what each got.
#[derive(Debug)]
struct Caller {
    log: Arc<Log>,
}

#[async_trait]
impl Tool for Caller {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "caller".into(),
            description: "Calls tools".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let nested = context.nested.clone().expect("a call of the model's has a gate");
        assert_eq!(Some(nested.outer_tool_call_id()), context.tool_call_id);
        let calls: Vec<(String, HashMap<String, Value>)> =
            serde_json::from_value(params["calls"].clone()).unwrap();
        let outcomes = if params.get("together") == Some(&Value::Bool(true)) {
            futures::future::join_all(calls.iter().map(|(name, args)| {
                let nested = nested.clone();
                async move { nested.call(name, args.clone()).await }
            }))
            .await
        } else {
            let mut outcomes = Vec::new();
            for (name, args) in &calls {
                outcomes.push(nested.call(name, args.clone()).await);
            }
            outcomes
        };
        let described: Vec<String> = outcomes.iter().map(describe).collect();
        for line in &described {
            self.log.push(line.clone());
        }
        Ok(described.join("\n"))
    }
}

fn describe(outcome: &NestedOutcome) -> String {
    match outcome {
        NestedOutcome::Finished(output) => format!("finished: {}", output.text),
        NestedOutcome::Failed(text) => format!("failed: {text}"),
        NestedOutcome::Refused(text) => format!("refused: {text}"),
        NestedOutcome::Halted(text) => format!("halted: {text}"),
    }
}

/// A registry with the caller and `targets`.
fn registry(log: &Arc<Log>, targets: &[(&'static str, Ends)]) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Caller { log: log.clone() }));
    for (name, ends) in targets {
        registry.register(Arc::new(Target {
            name,
            ends: ends.clone(),
            log: log.clone(),
        }));
    }
    registry
}

fn returns(text: &str) -> Ends {
    Ends::Returns(text.into())
}

fn caller_call(calls: Value) -> ai::ToolCall {
    native_call("outer", "caller", serde_json::json!({ "calls": calls }))
}

/// Runs one `caller` call of `calls` and then `complete_task`, and
/// returns the task and its events.
async fn run_caller(
    config: crate::agent_config::AgentConfig,
    registry: ToolRegistry,
    middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    calls: Value,
) -> (Task, Vec<AgentEvent>) {
    let workspace = workspace();
    let (mut agent, mut events, _commands) = agent(config, registry, &workspace);
    agent.execution_middleware = middleware;
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([turn(vec![caller_call(calls)]), complete()]);
    // (Long enough for the tests whose clock is paused.)
    let task = timeout(Duration::from_secs(4 * 3600), agent.run())
        .await
        .unwrap()
        .unwrap();
    std::fs::remove_dir_all(workspace).unwrap();
    (task, drain(&mut events))
}

fn outer_result(task: &Task) -> String {
    super::tool_call_tests::result_for(&task.conversation, "outer")[0]
        .output
        .clone()
}

fn tools() -> Vec<&'static str> {
    vec!["caller", "a", "b"]
}

/// A middleware that logs each call's `before` and `after`, and decides
/// `decision` for calls of `target`.
struct Decides {
    target: &'static str,
    decision: ToolExecutionDecision,
    log: Arc<Log>,
}

#[async_trait]
impl ToolExecutionMiddleware for Decides {
    async fn before(
        &self,
        _: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        self.log.push(format!("before {}", call.tool_name));
        Ok(if call.tool_name == self.target {
            self.decision.clone()
        } else {
            ToolExecutionDecision::Continue
        })
    }

    async fn after(
        &self,
        _: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
        _: ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError> {
        self.log.push(format!("after {}", call.tool_name));
        Ok(())
    }
}

#[tokio::test]
async fn nested_calls_pass_the_middleware_and_emit_their_own_events() {
    let log = Arc::new(Log::default());
    let middleware = Decides {
        target: "b",
        decision: ToolExecutionDecision::Reject {
            message_to_agent: "Refused: b is off limits.".into(),
        },
        log: log.clone(),
    };
    let (task, events) = run_caller(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[("a", returns("a's text")), ("b", returns("b's text"))]),
        Some(Arc::new(middleware)),
        serde_json::json!([["a", {}], ["b", {}]]),
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(
        log.entries(),
        [
            "before caller",
            "before a",
            "start a",
            "end a",
            "after a",
            "before b",
            "finished: a's text",
            "refused: Refused: b is off limits.",
            "after caller",
            "before complete_task",
            "after complete_task",
        ]
    );
    assert_eq!(
        outer_result(&task),
        "finished: a's text\nrefused: Refused: b is off limits."
    );
    // The model's call is the outer one alone, and each nested call has a
    // start and an end that name it.
    let outer = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolCallFormatted {
                tool_call_id,
                tool_name,
                ..
            } if tool_name == "caller" => Some(*tool_call_id),
            _ => None,
        })
        .unwrap();
    let formatted: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallFormatted { tool_name, .. }
            | AgentEvent::ToolResult { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        formatted,
        ["caller", "caller", "complete_task", "complete_task"]
    );
    let nested: Vec<(Uuid, String, Option<NestedCallOutcome>)> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::NestedToolCallStarted {
                outer_tool_call_id,
                tool_name,
                ..
            } => Some((*outer_tool_call_id, format!("start {tool_name}"), None)),
            AgentEvent::NestedToolCallFinished {
                outer_tool_call_id,
                tool_name,
                outcome,
                ..
            } => Some((*outer_tool_call_id, format!("end {tool_name}"), Some(*outcome))),
            _ => None,
        })
        .collect();
    assert_eq!(
        nested,
        [
            (outer, "start a".into(), None),
            (outer, "end a".into(), Some(NestedCallOutcome::Finished)),
            (outer, "start b".into(), None),
            (outer, "end b".into(), Some(NestedCallOutcome::Refused)),
        ]
    );
    // A nested call's failure isn't a tool error of the turn's.
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolRejected { .. }))
    );
}

#[tokio::test]
async fn a_nested_call_that_needs_approval_is_refused_without_waiting() {
    let log = Arc::new(Log::default());
    let mut config = config(ToolCallingMode::Native, &tools());
    config.tool_config.tool_specific_configs.insert(
        "b".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    let (task, events) = run_caller(
        config,
        registry(&log, &[("a", returns("a ran")), ("b", returns("b ran"))]),
        None,
        serde_json::json!([["b", {}], ["a", {}]]),
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let result = outer_result(&task);
    assert!(
        result.starts_with("refused: This call of b needs the user's approval")
            && result.contains("Call b directly instead.")
            && result.ends_with("finished: a ran"),
        "{result}"
    );
    assert!(!log.entries().contains(&"start b".to_string()));
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallAwaitingApproval { .. }
            | AgentEvent::BatchToolCallAwaitingApproval { .. }
            | AgentEvent::TaskStateChanged(TaskState::AwaitingToolApproval { .. })
    )));
}

#[tokio::test]
async fn a_nested_calls_paths_follow_the_path_policy() {
    use crate::agent_config::PathSafetyPolicy;
    for (policy, expected) in [
        (
            PathSafetyPolicy::AutoReject,
            "refused: Path for parameter 'path' is outside the workspace.",
        ),
        (
            PathSafetyPolicy::RequireApproval,
            "refused: This call of a needs the user's approval",
        ),
        (PathSafetyPolicy::AutoApprove, "finished: a ran"),
    ] {
        let log = Arc::new(Log::default());
        let mut config = config(ToolCallingMode::Native, &tools());
        config.tool_config.default_out_of_workspace_path_policy = policy.clone();
        let (task, _) = run_caller(
            config,
            registry(&log, &[("a", returns("a ran"))]),
            None,
            serde_json::json!([["a", {"path": "../outside.txt"}], ["a", {"path": "inside.txt"}]]),
        )
        .await;
        let result = outer_result(&task);
        assert!(result.starts_with(expected), "{policy:?}: {result}");
        assert!(result.ends_with("finished: a ran"), "{policy:?}: {result}");
        let runs = log.entries().iter().filter(|entry| *entry == "start a").count();
        let expected_runs = if policy == PathSafetyPolicy::AutoApprove { 2 } else { 1 };
        assert_eq!(runs, expected_runs, "{policy:?}");
    }
}

#[tokio::test]
async fn the_turns_own_tools_the_caller_and_unknown_tools_are_refused() {
    let log = Arc::new(Log::default());
    let (task, _) = run_caller(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[]),
        None,
        serde_json::json!([
            ["complete_task", {"summary": "done early"}],
            ["ask_user", {"questions": "Why?"}],
            ["close_questions", {"ids": "q1"}],
            ["handover_report", {"report": "r"}],
            ["caller", {"calls": []}],
            ["missing", {}],
        ]),
    )
    .await;
    let result = outer_result(&task);
    let lines: Vec<&str> = result.lines().collect();
    assert_eq!(lines.len(), 6, "{result}");
    for (line, name) in lines.iter().zip([
        "complete_task",
        "ask_user",
        "close_questions",
        "handover_report",
        "caller",
    ]) {
        assert_eq!(
            *line,
            format!("refused: {name} can't be called from inside a call of caller.")
        );
    }
    assert_eq!(lines[5], "refused: Tool 'missing' is not available.");
    // The task went on to its own complete_task.
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let summary = serde_json::to_string(&task.current_state).unwrap();
    assert!(!summary.contains("done early"), "{summary}");
}

#[tokio::test]
async fn a_middleware_that_fails_or_cancels_at_a_nested_call_stops_the_task_at_the_outer_call() {
    for cancel in [false, true] {
        let log = Arc::new(Log::default());
        let decision = if cancel {
            ToolExecutionDecision::CancelTask {
                reason: "Run cancelled.".into(),
            }
        } else {
            ToolExecutionDecision::FailTask {
                reason: "Budget exhausted.".into(),
            }
        };
        let (task, events) = run_caller(
            config(ToolCallingMode::Native, &tools()),
            registry(&log, &[("a", returns("a ran")), ("b", returns("b ran"))]),
            Some(Arc::new(Decides {
                target: "b",
                decision,
                log: log.clone(),
            })),
            serde_json::json!([["b", {}], ["a", {}]]),
        )
        .await;
        let reason = if cancel {
            "Run cancelled."
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
                cause: None,
                error_kind: None,
            }
        };
        assert_eq!(task.current_state, expected);
        // The call after the stop didn't run, and the caller heard of both.
        assert!(!log.entries().contains(&"start a".to_string()));
        let caller_heard: Vec<String> = log
            .entries()
            .into_iter()
            .filter(|entry| entry.starts_with("halted"))
            .collect();
        assert_eq!(caller_heard, vec![format!("halted: {reason}"); 2]);
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { tool_name, reason: Some(why), .. }
                if tool_name == "caller" && why == reason
        )));
    }
}

#[tokio::test]
async fn a_middleware_that_pauses_at_a_nested_call_pauses_the_task_with_the_outer_result() {
    let log = Arc::new(Log::default());
    let workspace = workspace();
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[("b", returns("b ran"))]),
        &workspace,
    );
    agent.execution_middleware = Some(Arc::new(Decides {
        target: "b",
        decision: ToolExecutionDecision::PauseTask {
            message_to_agent: "Not decided; paused for the host.".into(),
            reason: "The check could not run.".into(),
        },
        log: log.clone(),
    }));
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![caller_call(serde_json::json!([["b", {}]]))]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    let (error, cause) = wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
            error_message,
            cause,
            ..
        }) => Some((error_message.clone(), cause.clone())),
        _ => None,
    })
    .await;
    assert_eq!(error, "The check could not run.");
    assert_eq!(cause, ErrorRecoveryCause::Middleware);
    assert!(!log.entries().contains(&"start b".to_string()));
    commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // The outer call ran, and the model reads its result.
    assert_eq!(
        outer_result(&task),
        "halted: Not decided; paused for the host."
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test]
async fn nested_calls_run_one_at_a_time_in_the_order_made() {
    let log = Arc::new(Log::default());
    let wait = Ends::Waits(Duration::from_millis(50));
    let workspace = workspace();
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[("a", wait.clone()), ("b", wait)]),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "outer",
            "caller",
            serde_json::json!({"calls": [["a", {}], ["b", {}], ["a", {}]], "together": true}),
        )]),
        complete(),
    ]);
    timeout(Duration::from_secs(20), agent.run())
        .await
        .unwrap()
        .unwrap();
    let runs: Vec<String> = log
        .entries()
        .into_iter()
        .filter(|entry| entry.starts_with("start") || entry.starts_with("end"))
        .collect();
    assert_eq!(runs, ["start a", "end a", "start b", "end b", "start a", "end a"]);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_nested_call_keeps_its_tools_time_limit() {
    let log = Arc::new(Log::default());
    let mut config = config(ToolCallingMode::Native, &tools());
    config.tool_config.tool_specific_configs.insert(
        "a".into(),
        PerToolConfigOverride {
            tool_timeout_seconds: Some(Some(60)),
            ..Default::default()
        },
    );
    let (task, _) = run_caller(
        config,
        registry(&log, &[("a", Ends::Waits(Duration::from_secs(3600)))]),
        None,
        serde_json::json!([["a", {}]]),
    )
    .await;
    let result = outer_result(&task);
    assert!(
        result.starts_with("failed: The tool did not finish within the host's time limit of 1 minute"),
        "{result}"
    );
    assert_eq!(log.entries()[..2], ["start a", "dropped a"]);
}

#[tokio::test]
async fn a_refusal_inside_the_tool_is_a_refusal_and_a_failure_a_failure() {
    let log = Arc::new(Log::default());
    let (task, _) = run_caller(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[("a", Ends::Refuses), ("b", Ends::Fails)]),
        None,
        serde_json::json!([["a", {}], ["b", {}]]),
    )
    .await;
    assert_eq!(
        outer_result(&task),
        "refused: the guard said no\nfailed: Execution failed: it broke"
    );
}

#[tokio::test]
async fn a_nested_calls_end_event_holds_4_kb_of_its_result_and_its_length() {
    let log = Arc::new(Log::default());
    // Two-byte characters, so 4 KB falls inside none.
    let long = "é".repeat(3000);
    let (_, events) = run_caller(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[("a", returns(&long))]),
        None,
        serde_json::json!([["a", {}]]),
    )
    .await;
    let (output, bytes) = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::NestedToolCallFinished {
                output,
                output_bytes,
                ..
            } => Some((output.clone(), *output_bytes)),
            _ => None,
        })
        .unwrap();
    assert_eq!(bytes, 6000);
    assert_eq!(output.len(), NESTED_OUTPUT_BYTES);
    assert!(long.starts_with(&output));
}

#[tokio::test]
async fn a_stop_drops_a_nested_call_that_runs() {
    let log = Arc::new(Log::default());
    let workspace = workspace();
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &tools()),
        registry(&log, &[("a", Ends::Waits(Duration::from_secs(3600)))]),
        &workspace,
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![caller_call(serde_json::json!([["a", {}]]))]));
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::NestedToolCallStarted { .. }).then_some(())
    })
    .await;
    while !log.entries().contains(&"start a".to_string()) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert_eq!(log.entries(), ["start a", "dropped a"]);
    seen.extend(drain(&mut events));
    assert!(seen.iter().any(|event| matches!(
        event,
        AgentEvent::NestedToolCallFinished {
            outcome: NestedCallOutcome::Stopped,
            ..
        }
    )));
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A context made outside a runtime has no gate.
#[test]
fn a_context_made_by_hand_has_no_nested_calls() {
    let context = TaskContext::new(Uuid::new_v4(), "agent", "", std::path::Path::new("/"));
    let nested: Option<&NestedCalls> = context.nested.as_ref();
    assert!(nested.is_none());
}
