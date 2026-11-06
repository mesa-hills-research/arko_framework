//! The Claude Code runtime's waits for the host's decisions. A wait for an
//! answer or a recovery decision comes before any session, so its tests run
//! on paused time and start no CLI. A call's wait for approval runs in a
//! session, against the fake CLI.

use super::*;
use crate::runtime::HostDecisions;
use crate::task::ErrorRecoveryCause;
use tokio::time::Instant as TokioInstant;

const TIMEOUT: u64 = 100;

/// A Claude Code task waiting for error-recovery approval, with a
/// `TIMEOUT` command timeout (or `timeout`).
fn waiting_for_decision(fake: &Fake, decisions: HostDecisions, timeout: Option<u64>) -> Run {
    let mut config = config(fake, &[]);
    config.user_command_timeout_seconds = Some(timeout.unwrap_or(TIMEOUT));
    let mut task = Task::new(config, "Do the task.".into(), fake.workspace(), None);
    task.set_state(TaskState::AwaitingErrorRecoveryApproval {
        error_message: "synthetic outage".into(),
        cause: ErrorRecoveryCause::RetriesExhausted {
            retry_after_ms: None,
        },
        error_kind: None,
    });
    let runtime =
        AgentRuntime::new(Arc::new(arko_ai::LLMClient::new())).with_host_decisions(decisions);
    start_with(task, runtime)
}

/// Sends `script`'s commands at their times (seconds from `start`), then
/// keeps the channel open.
fn send_at(
    commands: mpsc::Sender<UserCommand>,
    start: TokioInstant,
    script: Vec<(u64, UserCommand)>,
) {
    tokio::spawn(async move {
        for (at, command) in script {
            tokio::time::sleep_until(start + Duration::from_secs(at)).await;
            let _ = commands.send(command).await;
        }
        std::future::pending::<()>().await;
    });
}

/// The run's task, once it ends, which must be before `by` seconds from
/// `start`.
async fn ended_by(run: Run, start: TokioInstant, by: u64) -> Task {
    let Run {
        mut events,
        commands,
        handle,
    } = run;
    drop(commands);
    tokio::spawn(async move { while events.recv().await.is_some() {} });
    tokio::time::timeout_at(start + Duration::from_secs(by), handle)
        .await
        .expect("the run was still waiting")
        .unwrap()
        .unwrap()
}

fn timed_out() -> TaskState {
    TaskState::Interrupted {
        reason: "No decision arrived before the command timeout.".into(),
    }
}

/// Context injected during the wait (as a workflow's message bus sends it)
/// doesn't move its deadline.
#[tokio::test(start_paused = true)]
async fn a_decision_wait_keeps_its_deadline_through_injected_context() {
    let fake = Fake::new(json!([]));
    let run = waiting_for_decision(&fake, HostDecisions::default(), None);
    let start = TokioInstant::now();
    send_at(
        run.commands.clone(),
        start,
        vec![(
            60,
            UserCommand::InjectContext {
                content: "A message from another agent.".into(),
            },
        )],
    );
    let task = ended_by(run, start, TIMEOUT + 1).await;
    assert_eq!(task.current_state, timed_out());
    assert_eq!(start.elapsed(), Duration::from_secs(TIMEOUT));
}

/// Commands the wait ignores (another call's approval, a refusal
/// resolution it can't apply, an answer while no question is open) don't
/// move its deadline either.
#[tokio::test(start_paused = true)]
async fn commands_a_decision_wait_ignores_do_not_extend_it() {
    let fake = Fake::new(json!([]));
    let run = waiting_for_decision(&fake, HostDecisions::default(), None);
    let start = TokioInstant::now();
    send_at(
        run.commands.clone(),
        start,
        vec![
            (
                40,
                UserCommand::ApproveToolUse {
                    tool_call_id: uuid::Uuid::new_v4(),
                    custom_tool_output: None,
                    message_to_agent: None,
                },
            ),
            (
                60,
                UserCommand::ResolveRefusal {
                    changes: vec![crate::task::RefusalResolution::RemoveUserMessage {
                        message_index: 0,
                    }],
                },
            ),
            (
                80,
                UserCommand::SubmitUserResponse {
                    response: "Not asked.".into(),
                    ids: Vec::new(),
                },
            ),
        ],
    );
    let task = ended_by(run, start, TIMEOUT + 1).await;
    assert_eq!(task.current_state, timed_out());
    assert_eq!(start.elapsed(), Duration::from_secs(TIMEOUT));
}

/// A wait the host says it is deciding has no deadline, however long the
/// decision takes.
#[tokio::test(start_paused = true)]
async fn a_decision_wait_the_host_is_deciding_is_not_timed() {
    let fake = Fake::new(json!([]));
    let run = waiting_for_decision(&fake, HostDecisions::default(), None);
    let start = TokioInstant::now();
    send_at(
        run.commands.clone(),
        start,
        vec![
            (0, UserCommand::HostDeciding),
            (5 * TIMEOUT, UserCommand::CancelTask),
        ],
    );
    let task = ended_by(run, start, 5 * TIMEOUT + 1).await;
    assert!(
        matches!(task.current_state, TaskState::Cancelled { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(start.elapsed(), Duration::from_secs(5 * TIMEOUT));
}

/// Set during a wait (as `AgentHandle::wait` does for a spawned agent),
/// `HostDecisions` stops the wait's timeout.
#[tokio::test(start_paused = true)]
async fn host_decisions_set_during_a_decision_wait_stop_its_timeout() {
    let fake = Fake::new(json!([]));
    let decisions = HostDecisions::new(false);
    let run = waiting_for_decision(&fake, decisions.clone(), None);
    let start = TokioInstant::now();
    tokio::spawn(async move {
        tokio::time::sleep_until(start + Duration::from_secs(TIMEOUT / 2)).await;
        decisions.set(true);
    });
    send_at(
        run.commands.clone(),
        start,
        vec![(3 * TIMEOUT, UserCommand::CancelTask)],
    );
    let task = ended_by(run, start, 3 * TIMEOUT + 1).await;
    assert!(
        matches!(task.current_state, TaskState::Cancelled { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(start.elapsed(), Duration::from_secs(3 * TIMEOUT));
}

/// A timeout too long for the clock (saved, or set by mistake) leaves the
/// wait without one.
#[tokio::test(start_paused = true)]
async fn a_command_timeout_too_long_for_the_clock_leaves_the_wait_untimed() {
    let fake = Fake::new(json!([]));
    let run = waiting_for_decision(&fake, HostDecisions::default(), Some(u64::MAX));
    let start = TokioInstant::now();
    send_at(
        run.commands.clone(),
        start,
        vec![(5 * TIMEOUT, UserCommand::CancelTask)],
    );
    let task = ended_by(run, start, 5 * TIMEOUT + 1).await;
    assert!(
        matches!(task.current_state, TaskState::Cancelled { .. }),
        "{:?}",
        task.current_state
    );
}

/// `Bash`, which needs approval, for an agent of `fake`.
fn approving(fake: &Fake) -> crate::agent_config::AgentConfig {
    let mut config = config(fake, &["Bash"]);
    config.tool_config.tool_specific_configs.insert(
        "Bash".into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..PerToolConfigOverride::default()
        },
    );
    config
}

/// The ID of the call the last event says awaits approval.
fn awaited(events: &[AgentEvent]) -> uuid::Uuid {
    match events.last() {
        Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) => *tool_call_id,
        _ => panic!("{events:#?}"),
    }
}

fn awaits_approval(event: &AgentEvent) -> bool {
    matches!(event, AgentEvent::ToolCallAwaitingApproval { .. })
}

/// A decision that comes after the host cancelled the task, before the
/// turn ended, leaves it cancelled, and the call doesn't run.
#[tokio::test]
async fn a_decision_after_a_cancel_leaves_the_task_cancelled() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "cargo test"}},
        complete("Tested"),
    ]]));
    let task = Task::new(
        approving(&fake),
        "Run the tests.".into(),
        fake.workspace(),
        None,
    );
    let mut run = start(task);
    let call = awaited(&run.events_until(awaits_approval).await);
    run.commands.send(UserCommand::CancelTask).await.unwrap();
    run.commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: call,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Cancelled { .. }),
        "{:?}",
        task.current_state
    );
    let permissions = fake.logged("permission");
    assert_eq!(
        permissions[0]["answer"]["behavior"], "deny",
        "{permissions:#?}"
    );
    assert!(
        fake.logged("post")
            .iter()
            .all(|post| post["output"] != "builtin Bash ran"),
        "the call ran"
    );
}

/// Set while a call waits for approval (as `AgentHandle::wait` does for a
/// spawned agent), `HostDecisions` stops the approval's timeout. Real
/// time, with the shortest timeout there is.
#[tokio::test]
async fn host_decisions_set_during_an_approval_wait_stop_its_timeout() {
    let fake = Fake::new(json!([[
        {"tool": "Bash", "input": {"command": "cargo test"}},
        complete("Tested"),
    ]]));
    let mut config = approving(&fake);
    config.user_command_timeout_seconds = Some(1);
    let task = Task::new(config, "Run the tests.".into(), fake.workspace(), None);
    let decisions = HostDecisions::new(false);
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()))
        .with_host_decisions(decisions.clone());
    let mut run = start_with(task, runtime);
    let call = awaited(&run.events_until(awaits_approval).await);
    decisions.set(true);
    tokio::time::sleep(Duration::from_millis(2500)).await;
    // A run whose approval timed out may have finished already.
    let _ = run
        .commands
        .send(UserCommand::ApproveToolUse {
            tool_call_id: call,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .await;
    let (task, _) = run.finish().await;
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    let permissions = fake.logged("permission");
    assert_eq!(
        permissions[0]["answer"]["behavior"], "allow",
        "{permissions:#?}"
    );
}
