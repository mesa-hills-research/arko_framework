//! Calls a tool makes from inside its own call ([`NestedCalls`]). Each test
//! fails if a nested call skips the gate: the call would run, or wait, or
//! leave no events.

use super::tool_call_tests::{
    agent, complete, config, drain, native_call, turn, wait_for, workspace,
};
use super::*;
use crate::agent_config::PerToolConfigOverride;
use crate::runtime::event::{NESTED_OUTPUT_BYTES, NestedCallOutcome};
use crate::tool::{
    ContextSource, NestedCalls, NestedOutcome, NestedState, Tool, ToolError,
    ToolParameterDefinition, ToolParameterType, ToolTypeDetails,
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
/// once with `together`, and returns what each got. Its description names
/// the tools the agent reaches only through it.
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

    fn offered_definition(
        &self,
        tool_config: &crate::agent_config::AgentToolConfig,
    ) -> ToolDefinition {
        let mut definition = self.definition();
        if !tool_config.nested_only_tool_names.is_empty() {
            definition.description = format!(
                "Calls tools, such as {}",
                tool_config.nested_only_tool_names.join(" and ")
            );
        }
        definition
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let nested = context
            .nested
            .clone()
            .expect("a call of the model's has a gate");
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
        if params.get("records") == Some(&Value::Bool(true)) {
            for record in nested.records() {
                self.log
                    .push(format!("record {} {:?}", record.tool_name, record.state));
            }
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
        registry(
            &log,
            &[("a", returns("a's text")), ("b", returns("b's text"))],
        ),
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
            } => Some((
                *outer_tool_call_id,
                format!("end {tool_name}"),
                Some(*outcome),
            )),
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
        let runs = log
            .entries()
            .iter()
            .filter(|entry| *entry == "start a")
            .count();
        let expected_runs = if policy == PathSafetyPolicy::AutoApprove {
            2
        } else {
            1
        };
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

/// A tool reached only from inside a call isn't offered, and a direct call
/// of it is answered as unavailable, while the gate runs it. A call of it
/// that would wait for approval is refused without telling the agent to
/// call it directly. The tool that reaches it describes it.
#[tokio::test]
async fn a_nested_only_tool_runs_only_from_inside_a_call() {
    let log = Arc::new(Log::default());
    let mut config = config(ToolCallingMode::Native, &["caller"]);
    config.tool_config.nested_only_tool_names = vec!["a".into(), "b".into()];
    config.tool_config.tool_specific_configs.insert(
        "b".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    assert!(config.validate().is_ok());
    let workspace = workspace();
    let (mut agent, mut events, _commands) = agent(
        config,
        registry(
            &log,
            &[
                ("a", returns("a ran")),
                ("b", returns("b ran")),
                ("c", returns("c ran")),
            ],
        ),
        &workspace,
    );
    let offered: Vec<(String, String)> = agent
        .build_native_tool_config(&agent.task.conversation.model, None)
        .tools
        .into_iter()
        .map(|tool| (tool.name, tool.description))
        .collect();
    assert_eq!(offered.len(), 2, "{offered:?}");
    assert_eq!(
        offered[0],
        (
            "caller".to_string(),
            "Calls tools, such as a and b".to_string()
        )
    );
    assert_eq!(offered[1].0, "complete_task");
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call("direct", "a", serde_json::json!({}))]),
        turn(vec![caller_call(serde_json::json!([
            ["a", {}],
            ["b", {}],
            ["c", {}]
        ]))]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    std::fs::remove_dir_all(workspace).unwrap();
    let events = drain(&mut events);
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let direct = &super::tool_call_tests::result_for(&task.conversation, "direct")[0];
    assert!(
        direct.is_error && direct.output.contains("Tool 'a' is not available."),
        "{direct:?}"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::UnavailableToolAttempted { tool_name } if tool_name == "a"
    )));
    assert_eq!(
        outer_result(&task),
        "finished: a ran\n\
         refused: This call of b needs the user's approval, which a call from inside caller can't wait for.\n\
         refused: Tool 'c' is not available."
    );
    assert_eq!(
        log.entries()
            .iter()
            .filter(|entry| entry.starts_with("start "))
            .collect::<Vec<_>>(),
        ["start a"]
    );
}

/// A configuration refuses a tool reached only from inside a call that it
/// also offers, that no nested call may call, or that a chat lists.
#[test]
fn a_nested_only_tool_is_neither_offered_nor_the_turns_own() {
    let refusal = |change: &dyn Fn(&mut crate::agent_config::AgentConfig)| {
        let mut config = config(ToolCallingMode::Native, &["caller"]);
        change(&mut config);
        config.validate().err().map(|error| error.to_string())
    };
    assert_eq!(refusal(&|_| {}), None);
    let offered = refusal(&|config| {
        config.tool_config.nested_only_tool_names = vec!["caller".into()];
    })
    .unwrap();
    assert!(
        offered.contains(
            "nested_only_tool_names lists 'caller', but available_tool_names lists it too"
        ),
        "{offered}"
    );
    let own = refusal(&|config| {
        config.tool_config.nested_only_tool_names = vec!["ask_user".into()];
    })
    .unwrap();
    assert!(
        own.contains("nested_only_tool_names lists 'ask_user', but no call made from inside a call may call it"),
        "{own}"
    );
    let chat = refusal(&|config| {
        config.tool_config = crate::agent_config::AgentToolConfig {
            nested_only_tool_names: vec!["a".into()],
            ..crate::agent_config::AgentToolConfig::chat()
        };
    })
    .unwrap();
    assert!(
        chat.contains("a chat has no tools, but nested_only_tool_names lists 'a'"),
        "{chat}"
    );
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
    assert_eq!(
        runs,
        ["start a", "end a", "start b", "end b", "start a", "end a"]
    );
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
        result.starts_with(
            "failed: The tool did not finish within the host's time limit of 1 minute"
        ),
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

/// A middleware that fails after each call of `target`.
struct FailsAfter {
    target: &'static str,
}

#[async_trait]
impl ToolExecutionMiddleware for FailsAfter {
    async fn before(
        &self,
        _: &ToolExecutionContext<'_>,
        _: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        Ok(ToolExecutionDecision::Continue)
    }

    async fn after(
        &self,
        _: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
        _: ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError> {
        if call.tool_name == self.target {
            return Err(ToolError::ExecutionFailed("the check broke".into()));
        }
        Ok(())
    }
}

/// A call that ran before the hook after it failed is recorded as having
/// run, and one refused or halted before as not, so a caller can say what
/// its calls may have changed.
#[tokio::test]
async fn the_records_say_which_calls_ran() {
    let log = Arc::new(Log::default());
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &tools());
    config.tool_config.tool_specific_configs.insert(
        "b".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    let (mut agent, mut events, commands) = agent(
        config,
        registry(&log, &[("a", returns("a ran")), ("b", returns("b ran"))]),
        &workspace,
    );
    agent.execution_middleware = Some(Arc::new(FailsAfter { target: "a" }));
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "outer",
            "caller",
            serde_json::json!({"calls": [["b", {}], ["a", {}], ["a", {}]], "records": true}),
        )]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
        .then_some(())
    })
    .await;
    commands
        .send(UserCommand::ContinueAfterError { note: None })
        .await
        .unwrap();
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let records: Vec<String> = log
        .entries()
        .into_iter()
        .filter(|entry| entry.starts_with("record"))
        .collect();
    // b needed approval, so its tool never started, and the second a came
    // after the halt.
    assert_eq!(records, ["record a Halted"]);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Calls made outside a loop, whose events go to `events`.
fn nested_calls(registry: ToolRegistry, events: mpsc::Sender<AgentEvent>) -> NestedCalls {
    nested_calls_with(registry, events, None)
}

/// [`nested_calls`] with `middleware`.
fn nested_calls_with(
    registry: ToolRegistry,
    events: mpsc::Sender<AgentEvent>,
    middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
) -> NestedCalls {
    let config = config(ToolCallingMode::Native, &tools());
    let gate = Arc::new(super::super::nested::Gate {
        registry: Arc::new(registry),
        tool_config: config.tool_config.clone(),
        events,
    });
    let environment = Arc::new(RunEnvironment {
        task_id: Uuid::new_v4(),
        agent_name: "agent".into(),
        initial_task_prompt: String::new(),
        working_directory: std::env::temp_dir(),
        middleware,
        stop: crate::stop::StopSignal::never(),
        model: config.llm_config.model.clone(),
        image_sink: None,
        image_room: None,
        context: ContextSource::new(),
        turn_context: None,
        #[cfg(any(unix, windows))]
        jobs: None,
        terminal_tools: false,
        gate: Some(gate.clone()),
        prompt_cache: Default::default(),
    });
    NestedCalls::new(
        environment,
        gate,
        Uuid::new_v4(),
        "caller",
        crate::stop::StopSignal::never(),
    )
}

/// A call stopped while it runs, or as it delivers its end, with no room
/// in the event channel, still ends there once there is room, and its
/// record says how far it got.
#[tokio::test]
async fn a_stopped_call_owes_its_end_when_the_events_are_full() {
    let log = Arc::new(Log::default());
    let (events, mut received) = mpsc::channel(1);
    let calls = nested_calls(
        registry(
            &log,
            &[
                ("a", Ends::Waits(Duration::from_secs(3600))),
                ("b", returns("b ran")),
            ],
        ),
        events,
    );
    let ended = |event: Option<AgentEvent>| match event {
        Some(AgentEvent::NestedToolCallFinished {
            tool_name, outcome, ..
        }) => (tool_name, outcome),
        other => panic!("{other:?}"),
    };
    // Stopped while it runs, its start filling the channel.
    let running = tokio::spawn({
        let calls = calls.clone();
        async move { calls.call("a", HashMap::new()).await }
    });
    while !log.entries().contains(&"start a".to_string()) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    running.abort();
    let _ = running.await;
    assert!(matches!(
        received.recv().await,
        Some(AgentEvent::NestedToolCallStarted { .. })
    ));
    let end = timeout(Duration::from_secs(5), received.recv())
        .await
        .unwrap();
    assert_eq!(ended(end), ("a".into(), NestedCallOutcome::Stopped));
    // Stopped as it waits to deliver its end.
    let delivering = tokio::spawn({
        let calls = calls.clone();
        async move { calls.call("b", HashMap::new()).await }
    });
    while !log.entries().contains(&"end b".to_string()) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    delivering.abort();
    let _ = delivering.await;
    assert!(matches!(
        received.recv().await,
        Some(AgentEvent::NestedToolCallStarted { .. })
    ));
    let end = timeout(Duration::from_secs(5), received.recv())
        .await
        .unwrap();
    assert_eq!(ended(end), ("b".into(), NestedCallOutcome::Finished));
    let states: Vec<(String, NestedState)> = calls
        .records()
        .into_iter()
        .map(|record| (record.tool_name, record.state))
        .collect();
    assert_eq!(
        states,
        [
            ("a".into(), NestedState::Running),
            ("b".into(), NestedState::Ran { failed: false }),
        ]
    );
}

/// A middleware whose `before` waits until its call is dropped.
struct Waits;

#[async_trait]
impl ToolExecutionMiddleware for Waits {
    async fn before(
        &self,
        _: &ToolExecutionContext<'_>,
        _: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        std::future::pending().await
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

/// A call stopped while the middleware's `before` judges it never started
/// its tool, so it has no record.
#[tokio::test]
async fn a_call_stopped_before_its_tool_starts_has_no_record() {
    let log = Arc::new(Log::default());
    let (events, mut received) = mpsc::channel(10);
    let calls = nested_calls_with(
        registry(&log, &[("a", returns("a ran"))]),
        events,
        Some(Arc::new(Waits)),
    );
    let judged = tokio::spawn({
        let calls = calls.clone();
        async move { calls.call("a", HashMap::new()).await }
    });
    assert!(matches!(
        received.recv().await,
        Some(AgentEvent::NestedToolCallStarted { .. })
    ));
    tokio::time::sleep(Duration::from_millis(20)).await;
    judged.abort();
    let _ = judged.await;
    assert!(calls.records().is_empty());
    assert!(log.entries().is_empty());
}

/// A flood of calls with large arguments leaves each start event small, so
/// a subscriber that falls behind holds little of them.
#[tokio::test]
async fn a_start_event_holds_its_arguments_cut() {
    let log = Arc::new(Log::default());
    let (events, mut received) = mpsc::channel(1000);
    let calls = nested_calls(registry(&log, &[("a", returns("a ran"))]), events);
    let big = "é".repeat(64 * 1024);
    for _ in 0..20 {
        let mut params = HashMap::from([
            ("content".to_string(), Value::String(big.clone())),
            ("list".to_string(), serde_json::json!([big.clone()])),
            ("n".to_string(), serde_json::json!(1)),
        ]);
        // Arguments the tool doesn't take, many of them, and one whose name
        // is long.
        for extra in 0..40 {
            params.insert(format!("x{extra:02}"), Value::String(big.clone()));
        }
        params.insert(
            format!("a{}", "n".repeat(1024 * 1024)),
            serde_json::json!(0),
        );
        calls.call("a", params).await;
    }
    drop(calls);
    let mut most = 0;
    while let Ok(event) = received.try_recv() {
        if let AgentEvent::NestedToolCallStarted {
            parameters,
            parameters_bytes,
            ..
        } = &event
        {
            let content = parameters["content"].as_str().unwrap();
            assert!(
                content.ends_with(" more bytes left out]"),
                "{}",
                &content[content.len() - 40..]
            );
            assert_eq!(parameters["n"], serde_json::json!(1));
            assert!(
                parameters["(left out)"]
                    .as_str()
                    .unwrap()
                    .ends_with(" more arguments")
            );
            assert!(*parameters_bytes > 40 * 128 * 1024);
            assert!(
                parameters
                    .keys()
                    .any(|name| name.starts_with("annn") && name.ends_with(" more bytes left out]"))
            );
        }
        most = most.max(serde_json::to_string(&event).unwrap().len());
    }
    assert!(most < 6 * NESTED_OUTPUT_BYTES, "an event of {most} bytes");
    // The size counted is the arguments' JSON.
    let params = HashMap::from([("say".to_string(), Value::String("a \"quote\"".into()))]);
    assert_eq!(
        super::super::nested::json_bytes(&params),
        serde_json::to_string(&params).unwrap().len()
    );
}
