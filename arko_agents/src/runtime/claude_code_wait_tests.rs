//! The Claude Code runtime's waits for the host's decisions, with the
//! opt-in `user_command_timeout_seconds`. A wait for an answer or a
//! recovery decision comes before any session, so these run on paused time
//! and start no CLI.

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
