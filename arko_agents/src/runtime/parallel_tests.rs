//! Several calls in one response, the one-call-per-turn switch, the
//! host-side tool time limit, and stopping running calls.

use super::tool_call_tests::{
    agent, assert_calls_paired, complete, config, drain, native_call, result_for, turn, workspace,
};
use super::*;
use crate::agent_config::PerToolConfigOverride;
use crate::tool::{Tool, ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// What probes record: each call's start and end, and how many ran at once.
#[derive(Debug, Default)]
struct Record {
    log: Mutex<Vec<String>>,
    active: AtomicUsize,
    most: AtomicUsize,
}

/// A tool that takes `delay` and records when its calls run.
#[derive(Debug)]
struct Probe {
    name: &'static str,
    parallel_safe: bool,
    delay: Duration,
    uses_host_timeout: bool,
    record: Arc<Record>,
}

impl Probe {
    fn new(name: &'static str, parallel_safe: bool, record: &Arc<Record>) -> Self {
        Self {
            name,
            parallel_safe,
            delay: Duration::from_millis(60),
            uses_host_timeout: true,
            record: record.clone(),
        }
    }
}

#[async_trait]
impl Tool for Probe {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.into(),
            description: "Records its runs".into(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: "label".into(),
                description: "A label".into(),
                param_type: ToolParameterType::String,
                required: false,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: self.parallel_safe,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    fn uses_host_timeout(&self) -> bool {
        self.uses_host_timeout
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let label = params
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.record
            .log
            .lock()
            .unwrap()
            .push(format!("start {label}"));
        let active = self.record.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.record.most.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.record.active.fetch_sub(1, Ordering::SeqCst);
        self.record.log.lock().unwrap().push(format!("end {label}"));
        Ok(format!("{label} done"))
    }
}

fn probe_call(id: &str, tool: &str) -> ai::ToolCall {
    native_call(id, tool, serde_json::json!({"label": id}))
}

fn probes(record: &Arc<Record>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Probe::new("probe_read", true, record)));
    registry.register(Arc::new(Probe::new("probe_write", false, record)));
    registry
}

/// Reads, a write, a read, a write: a turn whose calls mix parallel-safe
/// tools and others.
fn mixed_turn() -> ScriptedStreamResult {
    turn(vec![
        probe_call("r1", "probe_read"),
        probe_call("r2", "probe_read"),
        probe_call("w1", "probe_write"),
        probe_call("r3", "probe_read"),
        probe_call("w2", "probe_write"),
    ])
}

fn position(log: &[String], entry: &str) -> usize {
    log.iter()
        .position(|logged| logged == entry)
        .unwrap_or_else(|| panic!("{entry} missing from {log:?}"))
}

/// A response's calls all run and each gets its result: parallel-safe
/// neighbours at the same time, any other call alone, in call order.
#[tokio::test]
async fn every_call_of_a_response_runs_parallel_safe_ones_together() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["probe_read", "probe_write"]),
        probes(&record),
        &workspace,
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([mixed_turn(), complete()]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    for id in ["r1", "r2", "w1", "r3", "w2"] {
        let result = &result_for(&task.conversation, id)[0];
        assert_eq!(result.output, format!("{id} done"));
        assert!(!result.is_error);
    }
    // The results are in call order.
    let order: Vec<_> = task.conversation.messages[2]
        .content
        .iter()
        .filter_map(ai::Content::as_tool_result)
        .map(|result| result.call_id.clone().unwrap())
        .collect();
    assert_eq!(order, ["r1", "r2", "w1", "r3", "w2"]);
    let log = record.log.lock().unwrap().clone();
    // The two reads overlap; each write runs alone, between its neighbours.
    assert!(
        position(&log, "start r2") < position(&log, "end r1"),
        "{log:?}"
    );
    for (before, after) in [
        ("end r1", "start w1"),
        ("end r2", "start w1"),
        ("end w1", "start r3"),
        ("end r3", "start w2"),
    ] {
        assert!(position(&log, before) < position(&log, after), "{log:?}");
    }
    assert_eq!(record.most.load(Ordering::SeqCst), 2);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A call waiting for approval counts as not finished, as the calls held
/// behind it do: a call that isn't parallel-safe waits for every earlier
/// call, and a parallel-safe one for every earlier call that isn't, also
/// across a suspension and a save. Parallel-safe calls before it, or behind
/// only parallel-safe calls, run at once. Every call gets one result.
#[tokio::test]
async fn calls_after_a_call_awaiting_approval_keep_call_order() {
    for approve in [true, false] {
        let workspace = workspace();
        let record = Arc::new(Record::default());
        let mut registry = probes(&record);
        registry.register(Arc::new(Probe::new("probe_gate", false, &record)));
        let mut config = config(
            ToolCallingMode::Native,
            &["probe_gate", "probe_write", "probe_read"],
        );
        config.tool_config.tool_specific_configs.insert(
            "probe_gate".into(),
            PerToolConfigOverride {
                auto_approve_tool_use: Some(false),
                ..Default::default()
            },
        );
        let (mut agent, mut events, commands) = agent(config, registry, &workspace);
        // A read before the gate runs at once; the write and the read after
        // the gate wait for it.
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(turn(vec![
                probe_call("r0", "probe_read"),
                probe_call("g1", "probe_gate"),
                probe_call("w1", "probe_write"),
                probe_call("r1", "probe_read"),
            ]));
        let run = tokio::spawn(agent.run());
        let (batch_id, pending) = loop {
            let event = timeout(Duration::from_secs(10), events.recv())
                .await
                .unwrap()
                .unwrap();
            if let AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } = event
            {
                break (batch_id, pending_calls);
            }
        };
        assert_eq!(pending.len(), 1);
        let log = record.log.lock().unwrap().clone();
        assert_eq!(log, ["start r0", "end r0"]);

        // Suspended and saved while waiting, then run again.
        commands.send(UserCommand::SuspendTask).await.unwrap();
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let saved: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
        let mut registry = probes(&record);
        registry.register(Arc::new(Probe::new("probe_gate", false, &record)));
        let (mut resumed, _events, commands) =
            super::tool_call_tests::agent_for(saved, registry, Arc::new(ai::LLMClient::new()));
        resumed
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(complete());
        let decision = if approve {
            ToolApprovalDecision::Approve {
                custom_tool_output: None,
                message_to_agent: None,
            }
        } else {
            ToolApprovalDecision::Reject {
                reason: Some("Not now.".into()),
            }
        };
        commands
            .send(UserCommand::SubmitBatchToolApprovals {
                batch_id,
                decisions: HashMap::from([(pending[0].tool_call_id, decision)]),
            })
            .await
            .unwrap();
        let task = timeout(Duration::from_secs(10), resumed.run())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert_calls_paired(&task.conversation);
        let log = record.log.lock().unwrap().clone();
        if approve {
            assert_eq!(
                log,
                [
                    "start r0", "end r0", "start g1", "end g1", "start w1", "end w1", "start r1",
                    "end r1"
                ]
            );
            assert_eq!(result_for(&task.conversation, "g1")[0].output, "g1 done");
        } else {
            assert_eq!(
                log,
                [
                    "start r0", "end r0", "start w1", "end w1", "start r1", "end r1"
                ]
            );
            assert_eq!(result_for(&task.conversation, "g1")[0].output, "Not now.");
        }
        for id in ["r0", "w1", "r1"] {
            assert_eq!(
                result_for(&task.conversation, id)[0].output,
                format!("{id} done")
            );
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// `[write needing approval, read]`: the read waits for the write to be
/// decided and to run, or to be rejected.
#[tokio::test]
async fn a_read_after_a_write_awaiting_approval_waits_for_it() {
    for approve in [true, false] {
        let workspace = workspace();
        let record = Arc::new(Record::default());
        let mut config = config(ToolCallingMode::Native, &["probe_write", "probe_read"]);
        config.tool_config.tool_specific_configs.insert(
            "probe_write".into(),
            PerToolConfigOverride {
                auto_approve_tool_use: Some(false),
                ..Default::default()
            },
        );
        let (mut agent, mut events, commands) = agent(config, probes(&record), &workspace);
        agent.scripted_responses.get_mut().unwrap().extend([
            turn(vec![
                probe_call("w1", "probe_write"),
                probe_call("r1", "probe_read"),
            ]),
            complete(),
        ]);
        let run = tokio::spawn(agent.run());
        let (batch_id, pending) = loop {
            let event = timeout(Duration::from_secs(10), events.recv())
                .await
                .unwrap()
                .unwrap();
            if let AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } = event
            {
                break (batch_id, pending_calls);
            }
        };
        assert!(record.log.lock().unwrap().is_empty(), "the read waits");
        let decision = if approve {
            ToolApprovalDecision::Approve {
                custom_tool_output: None,
                message_to_agent: None,
            }
        } else {
            ToolApprovalDecision::Reject { reason: None }
        };
        commands
            .send(UserCommand::SubmitBatchToolApprovals {
                batch_id,
                decisions: HashMap::from([(pending[0].tool_call_id, decision)]),
            })
            .await
            .unwrap();
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert_calls_paired(&task.conversation);
        let log = record.log.lock().unwrap().clone();
        if approve {
            assert_eq!(log, ["start w1", "end w1", "start r1", "end r1"]);
        } else {
            assert_eq!(log, ["start r1", "end r1"]);
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// Without approvals too, a read after a write waits for it: `[read, write,
/// read]` runs read, write, read.
#[tokio::test]
async fn read_write_read_runs_in_call_order() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let (mut agent, _events, _commands) = agent(
        config(ToolCallingMode::Native, &["probe_read", "probe_write"]),
        probes(&record),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![
            probe_call("r1", "probe_read"),
            probe_call("w1", "probe_write"),
            probe_call("r2", "probe_read"),
        ]),
        complete(),
    ]);
    timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        record.log.lock().unwrap().clone(),
        [
            "start r1", "end r1", "start w1", "end w1", "start r2", "end r2"
        ]
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// `complete_task` after an earlier call of its turn was rejected or failed
/// doesn't complete the task: its result says why, and the agent goes on.
#[tokio::test]
async fn complete_task_after_a_failed_or_rejected_call_does_not_complete() {
    for reject in [false, true] {
        let workspace = workspace();
        let record = Arc::new(Record::default());
        let mut config = config(ToolCallingMode::Native, &["probe_write", "read_file"]);
        if reject {
            config.tool_config.tool_specific_configs.insert(
                "probe_write".into(),
                PerToolConfigOverride {
                    auto_approve_tool_use: Some(false),
                    ..Default::default()
                },
            );
        }
        let (mut agent, mut events, commands) = agent(config, probes(&record), &workspace);
        let first = if reject {
            probe_call("w1", "probe_write")
        } else {
            // Fails: the file doesn't exist.
            native_call(
                "w1",
                "read_file",
                serde_json::json!({"path": "missing.txt"}),
            )
        };
        agent.scripted_responses.get_mut().unwrap().extend([
            turn(vec![
                first,
                native_call(
                    "done",
                    "complete_task",
                    serde_json::json!({"summary": "too early"}),
                ),
            ]),
            complete(),
        ]);
        let run = tokio::spawn(agent.run());
        if reject {
            let (batch_id, pending) = loop {
                let event = timeout(Duration::from_secs(10), events.recv())
                    .await
                    .unwrap()
                    .unwrap();
                if let AgentEvent::BatchToolCallAwaitingApproval {
                    batch_id,
                    pending_calls,
                } = event
                {
                    break (batch_id, pending_calls);
                }
            };
            commands
                .send(UserCommand::SubmitBatchToolApprovals {
                    batch_id,
                    decisions: HashMap::from([(
                        pending[0].tool_call_id,
                        ToolApprovalDecision::Reject { reason: None },
                    )]),
                })
                .await
                .unwrap();
        }
        let task = timeout(Duration::from_secs(10), run)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_calls_paired(&task.conversation);
        let early = &result_for(&task.conversation, "done")[0];
        assert_eq!(early.output, COMPLETION_AFTER_FAILURE);
        // The agent went on and completed with its next turn.
        assert_eq!(
            task.current_state,
            TaskState::Completed {
                summary: "done".into()
            }
        );
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// `complete_task` that would need approval, after a call that already
/// failed, is answered at once instead of asking the host, and a call held
/// behind it then runs.
#[tokio::test]
async fn complete_task_after_a_known_failure_is_not_put_to_approval() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let mut config = config(ToolCallingMode::Native, &["probe_write", "read_file"]);
    config.tool_config.tool_specific_configs.insert(
        COMPLETE_TASK_TOOL_NAME.into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    let (mut agent, mut events, commands) = agent(config, probes(&record), &workspace);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![
            native_call(
                "r1",
                "read_file",
                serde_json::json!({"path": "missing.txt"}),
            ),
            native_call(
                "early",
                "complete_task",
                serde_json::json!({"summary": "too early"}),
            ),
            probe_call("w1", "probe_write"),
        ]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let mut batches = 0;
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        match event {
            AgentEvent::BatchToolCallAwaitingApproval { .. } => batches += 1,
            // The next turn's complete_task is asked about as usual.
            AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => {
                commands
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await
                    .unwrap();
            }
            AgentEvent::TaskStreamEnd => break,
            _ => {}
        }
    }
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        batches, 0,
        "the first turn's completion is not put to approval"
    );
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "early")[0].output,
        COMPLETION_AFTER_FAILURE
    );
    assert_eq!(result_for(&task.conversation, "w1")[0].output, "w1 done");
    assert_eq!(
        task.current_state,
        TaskState::Completed {
            summary: "done".into()
        }
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A call held behind a call awaiting approval gets its one result when the
/// wait ends another way: a cancellation answers both as not run.
#[tokio::test]
async fn a_held_call_is_answered_when_the_approval_wait_is_cancelled() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let mut registry = probes(&record);
    registry.register(Arc::new(Probe::new("probe_gate", false, &record)));
    let mut config = config(ToolCallingMode::Native, &["probe_gate", "probe_write"]);
    config.tool_config.tool_specific_configs.insert(
        "probe_gate".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    let (mut agent, mut events, commands) = agent(config, registry, &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![
            probe_call("g1", "probe_gate"),
            probe_call("w1", "probe_write"),
        ]));
    let run = tokio::spawn(agent.run());
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        if matches!(event, AgentEvent::BatchToolCallAwaitingApproval { .. }) {
            break;
        }
    }
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert_calls_paired(&task.conversation);
    for id in ["g1", "w1"] {
        let result = &result_for(&task.conversation, id)[0];
        assert_eq!(result.output, TASK_CANCELLED_CALL);
    }
    assert!(record.log.lock().unwrap().is_empty());
    assert!(task.pending_native_tool_calls.is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A per-tool override makes a tool parallel-safe or not.
#[tokio::test]
async fn a_per_tool_override_decides_parallel_safety() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let mut config = config(ToolCallingMode::Native, &["probe_read", "probe_write"]);
    config.tool_config.tool_specific_configs.insert(
        "probe_read".into(),
        PerToolConfigOverride {
            parallel_safe: Some(false),
            ..Default::default()
        },
    );
    let (mut agent, _events, _commands) = agent(config, probes(&record), &workspace);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![
            probe_call("r1", "probe_read"),
            probe_call("r2", "probe_read"),
        ]),
        complete(),
    ]);
    timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.most.load(Ordering::SeqCst), 1);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// One call per turn: the request asks for it where the provider can (not
/// Gemini), the prompt says so, and a response's calls still all run, one
/// at a time in order. `parallel_tool_calls: false` in the model's
/// configured tool settings means the same.
#[tokio::test]
async fn one_call_per_turn_disables_parallel_calls_and_runs_calls_one_at_a_time() {
    for configured in [false, true] {
        let workspace = workspace();
        let record = Arc::new(Record::default());
        let mut config = config(ToolCallingMode::Native, &["probe_read", "probe_write"]);
        if configured {
            config.llm_config.generation_config.tool_config =
                Some(ai::ToolConfig::new(vec![]).with_parallel_tool_calls(false));
        } else {
            config.tool_config.one_tool_call_per_turn = true;
        }
        let (mut agent, _events, _commands) = agent(config, probes(&record), &workspace);
        let openai = agent.task.conversation.model.clone();
        let mut gemini = openai.clone();
        gemini.provider = ai::Provider::Google;
        let configured_tools = agent
            .task
            .agent_config
            .llm_config
            .generation_config
            .tool_config
            .clone();
        assert_eq!(
            agent
                .build_native_tool_config(&openai, configured_tools.as_ref())
                .parallel_tool_calls,
            Some(false)
        );
        assert_eq!(
            agent
                .build_native_tool_config(&gemini, configured_tools.as_ref())
                .parallel_tool_calls,
            None
        );
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .extend([mixed_turn(), complete()]);
        let task = timeout(Duration::from_secs(10), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert_calls_paired(&task.conversation);
        assert!(
            task.conversation
                .system_message
                .as_deref()
                .unwrap()
                .contains("Call at most one tool per response.")
        );
        assert_eq!(record.most.load(Ordering::SeqCst), 1);
        let log = record.log.lock().unwrap().clone();
        let starts: Vec<_> = log
            .iter()
            .filter_map(|entry| entry.strip_prefix("start "))
            .collect();
        assert_eq!(starts, ["r1", "r2", "w1", "r3", "w2"]);
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// A configured tool choice is sent with the agent's tools, and by default
/// tool use is optional: no tool choice is sent, and the prompt doesn't
/// demand a tool call in every response.
#[test]
fn native_requests_honour_a_configured_tool_choice() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["read_file"]);
    let (agent_default, _events, _commands) =
        agent(config.clone(), ToolRegistry::new(), &workspace);
    let request = agent_default.build_generation_config();
    let tools = request.tool_config.unwrap();
    assert!(tools.tool_choice.is_none());
    assert!(tools.parallel_tool_calls.is_none());
    config.llm_config.generation_config.tool_config = Some(
        ai::ToolConfig::new(vec![ai::ToolDefinition::new(
            "elsewhere",
            "Not the agent's",
            serde_json::json!({"type": "object"}),
        )])
        .with_tool_choice(ai::ToolChoice::Required),
    );
    let (agent_forced, _events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    let tools = agent_forced.build_generation_config().tool_config.unwrap();
    assert!(matches!(tools.tool_choice, Some(ai::ToolChoice::Required)));
    let names: Vec<_> = tools.tools.iter().map(|tool| tool.name.as_str()).collect();
    assert_eq!(names, ["read_file", "complete_task"]);

    let parsing = agent_default.task.agent_config.get_tool_parsing_config();
    let prompt = PromptBuilder::new(&agent_default.task.agent_config, vec![], &parsing, None)
        .build_system_prompt(&agent_default.task)
        .unwrap();
    assert!(!prompt.contains("MUST"), "{prompt}");
    assert!(prompt.contains("You can call several tools in one response"));
    assert!(prompt.contains("The task ends only when you call `complete_task`"));
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A call that runs past the host's time limit is stopped; its result says
/// so and the tool-error policy applies (by default the agent is told and
/// the task goes on). Exempt tools have no limit unless their own setting
/// gives one, and `None` turns the limit off.
#[tokio::test(start_paused = true)]
async fn a_call_past_the_host_time_limit_is_stopped_and_reported() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let slow = |name: &'static str, exempt: bool| Probe {
        name,
        parallel_safe: false,
        delay: Duration::from_secs(30 * 60),
        uses_host_timeout: !exempt,
        record: record.clone(),
    };
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(slow("slow", false)));
    registry.register(Arc::new(slow("slow_exempt", true)));
    registry.register(Arc::new(slow("slow_exempt_limited", true)));
    let mut config = config(
        ToolCallingMode::Native,
        &["slow", "slow_exempt", "slow_exempt_limited"],
    );
    assert_eq!(config.tool_config.tool_timeout_seconds, Some(20 * 60));
    config.tool_config.tool_specific_configs.insert(
        "slow_exempt_limited".into(),
        PerToolConfigOverride {
            tool_timeout_seconds: Some(Some(60)),
            ..Default::default()
        },
    );
    let (mut agent, mut events, _commands) = agent(config, registry, &workspace);
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![
            probe_call("limited", "slow"),
            probe_call("exempt", "slow_exempt"),
            probe_call("own_limit", "slow_exempt_limited"),
        ]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(4 * 3600), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    let limited = &result_for(&task.conversation, "limited")[0];
    assert!(limited.is_error);
    assert!(
        limited
            .output
            .contains("did not finish within the host's time limit of 20 minutes"),
        "{}",
        limited.output
    );
    assert_eq!(
        result_for(&task.conversation, "exempt")[0].output,
        "exempt done"
    );
    let own = &result_for(&task.conversation, "own_limit")[0];
    assert!(
        own.output.contains("time limit of 1 minute"),
        "{}",
        own.output
    );
    let errors = drain(&mut events)
        .into_iter()
        .filter(|event| {
            matches!(event, AgentEvent::ErrorEncountered { error_message, requires_approval_to_retry: false, .. } if error_message.contains("time limit"))
        })
        .count();
    assert_eq!(errors, 2);

    // Without a limit the call finishes.
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(slow("slow", false)));
    let mut config = super::tool_call_tests::config(ToolCallingMode::Native, &["slow"]);
    config.tool_config.tool_timeout_seconds = None;
    let (mut agent, _events, _commands) =
        super::tool_call_tests::agent(config, registry, &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([turn(vec![probe_call("unlimited", "slow")]), complete()]);
    let task = timeout(Duration::from_secs(4 * 3600), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result_for(&task.conversation, "unlimited")[0].output,
        "unlimited done"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A tool that waits until its task is stopped.
#[derive(Debug)]
struct Waits;

#[async_trait]
impl Tool for Waits {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "waits".into(),
            description: "Waits".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        std::future::pending().await
    }
}

/// `CancelTask` and `SuspendTask` stop a call that is running: the call is
/// answered as stopped, the calls after it as not run, and context sent
/// while it ran is kept. A suspended task can run again.
#[tokio::test]
async fn cancelling_or_suspending_stops_a_running_call() {
    for suspend in [false, true] {
        let workspace = workspace();
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Waits));
        let (mut agent, mut events, commands) = agent(
            config(ToolCallingMode::Native, &["waits", "read_file"]),
            registry,
            &workspace,
        );
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(turn(vec![
                native_call("wait", "waits", serde_json::json!({})),
                native_call("after", "read_file", serde_json::json!({"path": "x"})),
            ]));
        let run = tokio::spawn(agent.run());
        loop {
            let event = timeout(Duration::from_secs(10), events.recv())
                .await
                .unwrap()
                .unwrap();
            if matches!(&event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "read_file")
            {
                break;
            }
        }
        commands
            .send(UserCommand::InjectContext {
                content: "Sent while the call ran.".into(),
            })
            .await
            .unwrap();
        commands
            .send(if suspend {
                UserCommand::SuspendTask
            } else {
                UserCommand::CancelTask
            })
            .await
            .unwrap();
        let task = timeout(Duration::from_secs(10), run)
            .await
            .expect("the running call is stopped")
            .unwrap()
            .unwrap();
        let seen = drain(&mut events);
        assert_calls_paired(&task.conversation);
        let stopped = &result_for(&task.conversation, "wait")[0];
        let after = &result_for(&task.conversation, "after")[0];
        assert!(stopped.is_error && after.is_error);
        if suspend {
            assert_eq!(task.current_state, TaskState::InProgress);
            assert_eq!(stopped.output, STOPPED_BY_SUSPEND);
            assert_eq!(after.output, TASK_SUSPENDED_CALL);
            assert!(
                seen.iter().any(|event| matches!(
                    event,
                    AgentEvent::TaskStateChanged(TaskState::InProgress)
                ))
            );
        } else {
            assert_eq!(
                task.current_state,
                TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.into()
                }
            );
            assert_eq!(stopped.output, STOPPED_BY_CANCEL);
            assert_eq!(after.output, TASK_CANCELLED_CALL);
        }
        // Each call has an outcome event.
        assert_eq!(
            seen.iter()
                .filter(|event| matches!(event, AgentEvent::ToolRejected { .. }))
                .count(),
            2
        );
        // The context sent while the call ran is not lost.
        assert!(task.conversation.messages.iter().any(|message| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_text)
                .any(|text| text.contains("Sent while the call ran."))
        }));
        if suspend {
            // The suspended task runs again.
            let mut registry = ToolRegistry::new();
            registry.register(Arc::new(Waits));
            let (mut resumed, _events, _commands) =
                super::tool_call_tests::agent_for(task, registry, Arc::new(ai::LLMClient::new()));
            resumed
                .scripted_responses
                .get_mut()
                .unwrap()
                .push_back(complete());
            let task = timeout(Duration::from_secs(10), resumed.run())
                .await
                .unwrap()
                .unwrap();
            assert!(matches!(task.current_state, TaskState::Completed { .. }));
            assert_calls_paired(&task.conversation);
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// A parallel-safe tool that waits until its task is stopped.
#[derive(Debug)]
struct WaitsTogether;

#[async_trait]
impl Tool for WaitsTogether {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "waits_together".into(),
            parallel_safe: true,
            ..Waits.definition()
        }
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        std::future::pending().await
    }
}

/// When a stop ends a group of parallel calls, a call that had already
/// finished keeps its result; only the calls still running are stopped.
#[tokio::test]
async fn a_call_that_finished_before_a_stop_keeps_its_result() {
    let workspace = workspace();
    let record = Arc::new(Record::default());
    let mut registry = probes(&record);
    registry.register(Arc::new(WaitsTogether));
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &["probe_read", "waits_together"]),
        registry,
        &workspace,
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![
            probe_call("quick", "probe_read"),
            native_call("stuck", "waits_together", serde_json::json!({})),
        ]));
    let run = tokio::spawn(agent.run());
    // The quick read finishes (it records its end) before the stop.
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        if matches!(&event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "waits_together")
        {
            break;
        }
    }
    timeout(Duration::from_secs(10), async {
        while !record
            .log
            .lock()
            .unwrap()
            .iter()
            .any(|entry| entry == "end quick")
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "quick")[0].output,
        "quick done"
    );
    assert_eq!(
        result_for(&task.conversation, "stuck")[0].output,
        STOPPED_BY_CANCEL
    );
    let seen = drain(&mut events);
    assert!(seen.iter().any(|event| matches!(
        event,
        AgentEvent::ToolResult { tool_name, .. } if tool_name == "probe_read"
    )));
    // The cancellation is reported once.
    assert_eq!(
        seen.iter()
            .filter(|event| matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::Cancelled { .. })
            ))
            .count(),
        1
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A cancellation while an approved call runs is reported once, and when a
/// call of the stopped group had already failed the task, the stopped calls
/// say the task ended rather than that it was cancelled.
#[tokio::test]
async fn stops_are_reported_once_and_worded_for_how_the_task_ended() {
    let workspace = workspace();
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Waits));
    let mut config = config(ToolCallingMode::Native, &["waits"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, registry, &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![native_call(
            "wait",
            "waits",
            serde_json::json!({}),
        )]));
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        if let AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } = &event {
            commands
                .send(UserCommand::ApproveToolUse {
                    tool_call_id: *tool_call_id,
                    custom_tool_output: None,
                    message_to_agent: None,
                })
                .await
                .unwrap();
            break;
        }
        seen.push(event);
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    seen.extend(drain(&mut events));
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert_calls_paired(&task.conversation);
    assert_eq!(
        seen.iter()
            .filter(|event| matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::Cancelled { .. })
            ))
            .count(),
        1,
        "{seen:#?}"
    );
    std::fs::remove_dir_all(workspace).unwrap();

    // A call of the group failed the task (FailTask) as the cancel arrived.
    assert_eq!(
        stopped_call_reason(
            ToolsStopped::Cancelled,
            &TaskState::Failed {
                reason: "tool error".into(),
                cause: None,
            }
        ),
        STOPPED_TASK_ENDED
    );
    assert_eq!(
        stopped_call_reason(ToolsStopped::Suspended, &TaskState::InProgress),
        STOPPED_BY_SUSPEND
    );
    assert_eq!(
        stopped_call_reason(
            ToolsStopped::Cancelled,
            &TaskState::Cancelled {
                reason: CANCELLED_BY_USER.into()
            }
        ),
        STOPPED_BY_CANCEL
    );
}

/// A tool that winds its own work down when the task stops.
#[derive(Debug)]
struct WindsDown(Arc<tokio::sync::Notify>);

#[async_trait]
impl Tool for WindsDown {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "winds_down".into(),
            description: "Winds down".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    fn interruptible(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.0.notified().await;
        Ok("wound down".into())
    }
}

/// A call that isn't interruptible runs to its end when the task is
/// cancelled; the cancellation then takes effect, and later calls don't run.
#[tokio::test]
async fn a_cancellation_waits_for_a_call_that_is_not_interruptible() {
    let workspace = workspace();
    let release = Arc::new(tokio::sync::Notify::new());
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(WindsDown(release.clone())));
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &["winds_down", "read_file"]),
        registry,
        &workspace,
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(turn(vec![
            native_call("wind", "winds_down", serde_json::json!({})),
            native_call("after", "read_file", serde_json::json!({"path": "x"})),
        ]));
    let run = tokio::spawn(agent.run());
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        if matches!(&event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "read_file")
        {
            break;
        }
    }
    commands.send(UserCommand::CancelTask).await.unwrap();
    // The call is still running.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!run.is_finished());
    release.notify_one();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        task.current_state,
        TaskState::Cancelled {
            reason: CANCELLED_BY_USER.into()
        }
    );
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "wind")[0].output,
        "wound down"
    );
    assert_eq!(
        result_for(&task.conversation, "after")[0].output,
        TASK_CANCELLED_CALL
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// In output-based mode too, a cancellation stops a running call.
#[tokio::test]
async fn cancelling_stops_a_running_output_based_call() {
    let workspace = workspace();
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Waits));
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::OutputBased, &["waits"]),
        registry,
        &workspace,
    );
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(super::tool_call_tests::text_turn("<waits>\n</waits>"));
    let run = tokio::spawn(agent.run());
    loop {
        let event = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
            .unwrap();
        if matches!(&event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "waits")
        {
            break;
        }
    }
    commands.send(UserCommand::CancelTask).await.unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .expect("the running call is stopped")
        .unwrap()
        .unwrap();
    assert_eq!(
        task.current_state,
        TaskState::Cancelled {
            reason: CANCELLED_BY_USER.into()
        }
    );
    let seen = drain(&mut events);
    assert!(seen.iter().any(|event| matches!(
        event,
        AgentEvent::ToolRejected { reason: Some(reason), .. } if reason == STOPPED_BY_CANCEL
    )));
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Configurations saved with the removed batching settings still load, and
/// none of them makes calls run at the same time: `allow_batched_tool_calls`
/// is ignored, a tool's `allow_batched_calls: false` keeps it one at a time
/// and `true` is ignored. Tool time limits round-trip, `null` included.
#[test]
fn old_batching_settings_load_and_time_limits_round_trip() {
    let json = serde_json::json!({
        "available_tool_names": ["read_file", "write_file"],
        "allow_batched_tool_calls": false,
        "tool_specific_configs": {
            "write_file": {"allow_batched_calls": true, "auto_approve_tool_use": true},
            "read_file": {"allow_batched_calls": false}
        },
        "tool_calling_mode": "Native"
    });
    let loaded: crate::agent_config::AgentToolConfig = serde_json::from_value(json).unwrap();
    assert!(!loaded.one_tool_call_per_turn);
    assert_eq!(loaded.tool_timeout_seconds, Some(20 * 60));
    let write = &loaded.tool_specific_configs["write_file"];
    assert_eq!(write.parallel_safe, None);
    assert_eq!(write.auto_approve_tool_use, Some(true));
    assert_eq!(
        loaded.tool_specific_configs["read_file"].parallel_safe,
        Some(false)
    );
    let saved = serde_json::to_value(&loaded).unwrap();
    assert!(saved.get("allow_batched_tool_calls").is_none(), "{saved}");
    assert_eq!(
        saved["tool_specific_configs"]["write_file"],
        serde_json::json!({"auto_approve_tool_use": true})
    );
    assert_eq!(
        saved["tool_specific_configs"]["read_file"],
        serde_json::json!({"parallel_safe": false})
    );
    assert_eq!(saved["tool_timeout_seconds"], 1200);

    let yaml = "available_tool_names: [read_file]\ntool_timeout_seconds: null\ntool_specific_configs:\n  read_file:\n    tool_timeout_seconds: null\n  write_file:\n    tool_timeout_seconds: 90\n";
    let loaded: crate::agent_config::AgentToolConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(loaded.tool_timeout_seconds, None);
    assert_eq!(
        loaded.tool_specific_configs["read_file"].tool_timeout_seconds,
        Some(None)
    );
    assert_eq!(
        loaded.tool_specific_configs["write_file"].tool_timeout_seconds,
        Some(Some(90))
    );
    let again: crate::agent_config::AgentToolConfig =
        serde_json::from_value(serde_json::to_value(&loaded).unwrap()).unwrap();
    assert_eq!(again.tool_timeout_seconds, None);
    assert_eq!(
        again.tool_specific_configs["read_file"].tool_timeout_seconds,
        Some(None)
    );
    assert_eq!(
        again.tool_timeout("read_file", true),
        None,
        "an explicit null turns the limit off"
    );
    assert_eq!(
        again.tool_timeout("write_file", false),
        Some(Duration::from_secs(90))
    );

    // A saved tool definition's batching flag doesn't make it parallel-safe.
    let definition: ToolDefinition = serde_json::from_value(serde_json::json!({
        "name": "old", "description": "d", "parameters": [],
        "supports_batched_calls": true, "tool_type_details": "DeveloperDefined"
    }))
    .unwrap();
    assert!(!definition.parallel_safe);

    // A zero time limit is rejected.
    let mut config = super::tests::test_config();
    config.tool_config.tool_timeout_seconds = Some(0);
    assert!(config.validate().is_err());
}

/// In output-based mode the batch syntax is described whenever an offered
/// tool can be batched (a parallel-safe one), and not under one call per
/// turn.
#[test]
fn output_based_batching_follows_parallel_safety() {
    for one_call in [false, true] {
        let mut config = config(ToolCallingMode::OutputBased, &["read_file", "write_file"]);
        config.tool_config.one_tool_call_per_turn = one_call;
        let registry = ToolRegistry::new();
        let definitions: Vec<_> = ["read_file", "write_file"]
            .iter()
            .map(|name| registry.get(name).unwrap().definition())
            .collect();
        let task = Task::new(config.clone(), "task".into(), std::env::temp_dir(), None);
        let parsing = config.get_tool_parsing_config();
        let prompt = PromptBuilder::new(&config, definitions.iter().collect(), &parsing, None)
            .build_system_prompt(&task)
            .unwrap();
        assert_eq!(
            prompt.contains("<batched_tool_call>"),
            !one_call,
            "{prompt}"
        );
        assert_eq!(
            prompt.matches("Supports batched calls.").count(),
            usize::from(!one_call),
            "{prompt}"
        );
    }
}
