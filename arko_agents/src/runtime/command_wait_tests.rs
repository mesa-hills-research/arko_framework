//! The opt-in `user_command_timeout_seconds`: one deadline per wait for a
//! command, set when the wait starts. (Paused time: the waits are exact.)

use super::tool_call_tests::{agent, complete, config, result_for, workspace};
use super::*;
use crate::agent_config::PerToolConfigOverride;
use tokio::time::Instant as TokioInstant;

const TIMEOUT: u64 = 100;

/// An agent with a `TIMEOUT` command timeout whose `complete_task` needs
/// approval.
fn approving_config() -> AgentConfig {
    let mut config = config(ToolCallingMode::Native, &[]);
    config.user_command_timeout_seconds = Some(TIMEOUT);
    config.tool_config.default_auto_approve_tool_use = false;
    config.tool_config.tool_specific_configs.insert(
        COMPLETE_TASK_TOOL_NAME.into(),
        PerToolConfigOverride {
            auto_approve_tool_use: Some(false),
            ..Default::default()
        },
    );
    config
}

/// An agent waiting for error-recovery approval; approved, it completes
/// (its `complete_task` needs approval when `approving`).
fn waiting_for_recovery(
    approving: bool,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
    std::path::PathBuf,
) {
    let workspace = workspace();
    let config = if approving {
        approving_config()
    } else {
        let mut config = config(ToolCallingMode::Native, &[]);
        config.user_command_timeout_seconds = Some(TIMEOUT);
        config
    };
    let (mut agent, events, commands) = agent(config, ToolRegistry::new(), &workspace);
    let task = &mut agent.task;
    task.conversation.system_message = Some("system".into());
    task.conversation.add_user_message("Do the task");
    task.set_state(TaskState::AwaitingErrorRecoveryApproval {
        error_message: "synthetic outage".into(),
        cause: ErrorRecoveryCause::RetriesExhausted {
            retry_after_ms: None,
        },
        error_kind: None,
    });
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    (agent, events, commands, workspace)
}

/// The ID of the next call awaiting approval.
async fn awaited_call(events: &mut mpsc::Receiver<AgentEvent>) -> Uuid {
    loop {
        if let Some(AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. }) = events.recv().await
        {
            return tool_call_id;
        }
    }
}

async fn sleep_until(start: TokioInstant, seconds: u64) {
    tokio::time::sleep_until(start + Duration::from_secs(seconds)).await;
}

/// Context injected during the wait (as a workflow's message bus does)
/// doesn't restart its deadline: the wait ends at its deadline, and a
/// decision after it is too late.
#[tokio::test(start_paused = true)]
async fn injected_context_does_not_extend_the_wait() {
    let (agent, _events, commands, workspace) = waiting_for_recovery(false);
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        sleep_until(start, 60).await;
        let _ = commands
            .send(UserCommand::InjectContext {
                content: "A message from another agent.".into(),
            })
            .await;
        sleep_until(start, 120).await;
        let _ = commands
            .send(UserCommand::ApproveErrorRecovery { message: None })
            .await;
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    // The reason names the state the task waited in.
    assert_eq!(
        task.current_state,
        TaskState::Interrupted {
            reason: format!(
                "Timeout ({TIMEOUT}s) waiting for user command in state AwaitingErrorRecoveryApproval."
            ),
        }
    );
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(TIMEOUT),
        "{:?}",
        task.current_state
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Commands the wait ignores (another call's approval or rejection, an
/// answer while no question is open) don't restart its deadline either.
#[tokio::test(start_paused = true)]
async fn ignored_commands_do_not_extend_the_wait() {
    let workspace = workspace();
    let (mut agent, mut events, commands) =
        agent(approving_config(), ToolRegistry::new(), &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(complete());
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        let call = awaited_call(&mut events).await;
        sleep_until(start, 40).await;
        let _ = commands
            .send(UserCommand::ApproveToolUse {
                tool_call_id: Uuid::new_v4(),
                custom_tool_output: None,
                message_to_agent: None,
            })
            .await;
        sleep_until(start, 60).await;
        let _ = commands
            .send(UserCommand::RejectToolUse {
                tool_call_id: Uuid::new_v4(),
                reason: None,
            })
            .await;
        sleep_until(start, 80).await;
        let _ = commands
            .send(UserCommand::SubmitUserResponse {
                response: "Not asked.".into(),
                ids: Vec::new(),
            })
            .await;
        sleep_until(start, 120).await;
        let _ = commands
            .send(UserCommand::ApproveToolUse {
                tool_call_id: call,
                custom_tool_output: None,
                message_to_agent: None,
            })
            .await;
        // Keeps the channels open.
        while events.recv().await.is_some() {}
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Interrupted { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(TIMEOUT),
        "{:?}",
        task.current_state
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A refusal resolution the task can't apply (here, it has no refused
/// input) is reported and the wait goes on: it doesn't restart the
/// deadline either.
#[tokio::test(start_paused = true)]
async fn a_refusal_resolution_that_fails_does_not_extend_the_wait() {
    let (agent, _events, commands, workspace) = waiting_for_recovery(false);
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        sleep_until(start, 60).await;
        let _ = commands
            .send(UserCommand::ResolveRefusal {
                changes: vec![crate::task::RefusalResolution::RemoveUserMessage {
                    message_index: 0,
                }],
            })
            .await;
        sleep_until(start, 120).await;
        let _ = commands
            .send(UserCommand::ApproveErrorRecovery { message: None })
            .await;
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Interrupted { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(TIMEOUT),
        "{:?}",
        task.current_state
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A decision ends the wait; a later wait has a deadline of its own, a
/// full timeout from when it starts.
#[tokio::test(start_paused = true)]
async fn a_decision_ends_the_wait_and_the_next_wait_has_its_own_deadline() {
    let (agent, mut events, commands, workspace) = waiting_for_recovery(true);
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        sleep_until(start, 90).await;
        let _ = commands
            .send(UserCommand::ContinueAfterError { note: None })
            .await;
        // `complete_task` now waits for approval, from 90 s.
        let call = awaited_call(&mut events).await;
        sleep_until(start, 180).await;
        let _ = commands
            .send(UserCommand::ApproveToolUse {
                tool_call_id: call,
                custom_tool_output: None,
                message_to_agent: None,
            })
            .await;
        // Keeps the channels open.
        while events.recv().await.is_some() {}
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(
        start.elapsed(),
        Duration::from_secs(180),
        "{:?}",
        task.current_state
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A wait the host says it is deciding (`HostDeciding`, as a workflow does
/// for each wait of its agents) has no deadline, however long the decision
/// takes; the next wait is timed again.
#[tokio::test(start_paused = true)]
async fn a_wait_the_host_is_deciding_is_not_timed() {
    let (agent, mut events, commands, workspace) = waiting_for_recovery(true);
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        let _ = commands.send(UserCommand::HostDeciding).await;
        sleep_until(start, 5 * TIMEOUT).await;
        let _ = commands
            .send(UserCommand::ContinueAfterError { note: None })
            .await;
        // `complete_task` now waits for approval, and nothing answers.
        awaited_call(&mut events).await;
        // Keeps the channels open.
        while events.recv().await.is_some() {}
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Interrupted { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(start.elapsed(), Duration::from_secs(6 * TIMEOUT));
    assert!(
        result_for(&task.conversation, "done")[0]
            .output
            .contains(&format!("Human approval timed out after {TIMEOUT} seconds")),
        "{:?}",
        task.conversation.messages.last()
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A saved task holding a zero timeout (which ended every wait at once,
/// before validation rejected it) still loads; running it fails at start
/// with the validation error, until the value is changed.
#[tokio::test(start_paused = true)]
async fn a_saved_task_with_a_zero_timeout_loads_but_does_not_start() {
    let (mut agent, _events, _commands, workspace) = waiting_for_recovery(false);
    let mut saved = serde_json::to_value(&agent.task).unwrap();
    saved["agent_config"]["user_command_timeout_seconds"] = serde_json::json!(0);
    agent.task = serde_json::from_value(saved).unwrap();
    let error = agent.run().await.unwrap_err().to_string();
    assert!(
        error.contains("user_command_timeout_seconds must be at least 1"),
        "{error}"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// While the run's `HostDecisions` is set (as a workflow sets it for the
/// runs it drives), no wait is timed; it isn't saved with the task, so the
/// same task run without it is timed as configured.
#[tokio::test(start_paused = true)]
async fn a_run_whose_host_decides_its_waits_is_not_timed() {
    let (mut agent, _events, commands, workspace) = waiting_for_recovery(false);
    agent.host_decisions = super::super::HostDecisions::new(true);
    let saved = serde_json::to_value(&agent.task).unwrap();
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        sleep_until(start, 5 * TIMEOUT).await;
        let _ = commands
            .send(UserCommand::ApproveErrorRecovery { message: None })
            .await;
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(start.elapsed(), Duration::from_secs(5 * TIMEOUT));

    let (mut agent, _events, _commands, workspace_again) = waiting_for_recovery(false);
    agent.task = serde_json::from_value(saved).unwrap();
    let start = TokioInstant::now();
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Interrupted { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(start.elapsed(), Duration::from_secs(TIMEOUT));
    std::fs::remove_dir_all(workspace).unwrap();
    std::fs::remove_dir_all(workspace_again).unwrap();
}

/// Set during a wait (as `AgentHandle::wait` does for a spawned agent), the
/// wait is no longer timed: its deadline passes without interrupting it.
#[tokio::test(start_paused = true)]
async fn host_decisions_set_during_a_wait_stop_its_timeout() {
    let (mut agent, _events, commands, workspace) = waiting_for_recovery(false);
    let decisions = super::super::HostDecisions::new(false);
    agent.host_decisions = decisions.clone();
    let start = TokioInstant::now();
    let _script = tokio::spawn(async move {
        sleep_until(start, TIMEOUT / 2).await;
        decisions.set(true);
        sleep_until(start, 3 * TIMEOUT).await;
        let _ = commands
            .send(UserCommand::ApproveErrorRecovery { message: None })
            .await;
        std::future::pending::<()>().await;
    });
    let task = agent.run().await.unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    assert_eq!(start.elapsed(), Duration::from_secs(3 * TIMEOUT));
    std::fs::remove_dir_all(workspace).unwrap();
}
