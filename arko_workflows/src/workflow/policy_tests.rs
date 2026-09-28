//! Approval and error-recovery policies, pauses and their resumption, the
//! hooks around them, and messages across a pause.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arko_agents::runtime::command::ToolApprovalDecision;
use arko_agents::{
    AgentConfig, ErrorRecoveryCause, LLMErrorPolicy, PerToolConfigOverride, TaskContext, Tool,
    ToolDefinition, ToolError, ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::*;
use crate::hooks::{HookPoint, HookResult};
use crate::policy::{
    AgentPause, AgentResume, ApprovalPolicy, ErrorRecoveryPolicy, RecoveryWaits,
    ToolApprovalHandler, ToolApprovalOutcome, ToolApprovalRequest,
};
use crate::test_support::{
    ProbeTool, Reply, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;

/// Counts the hooks fired at each point.
#[derive(Clone, Default)]
struct HookCounts(Arc<parking_lot::Mutex<HashMap<String, usize>>>);

impl HookCounts {
    fn register(&self, mut builder: WorkflowBuilder) -> WorkflowBuilder {
        for point in [
            HookPoint::OnAgentPaused,
            HookPoint::OnWorkflowFailed,
            HookPoint::OnUserInputRequired,
            HookPoint::PostAgentFailure,
        ] {
            let counts = self.0.clone();
            let name = format!("{point:?}");
            builder = builder.hook(
                point,
                Arc::new(move |_| {
                    *counts.lock().entry(name.clone()).or_default() += 1;
                    HookResult::Continue
                }),
            );
        }
        builder
    }

    fn get(&self, point: HookPoint) -> usize {
        self.0
            .lock()
            .get(&format!("{point:?}"))
            .copied()
            .unwrap_or(0)
    }
}

/// An agent whose configuration asks for approval of every tool but
/// `complete_task` and `auto_approved`.
fn approving_agent(auto_approved: &[&str]) -> AgentConfig {
    let mut config = agent_config("Friendly Coder");
    config.tool_config.default_auto_approve_tool_use = false;
    for tool in auto_approved.iter().chain(&["complete_task"]) {
        config.tool_config.tool_specific_configs.insert(
            tool.to_string(),
            PerToolConfigOverride {
                auto_approve_tool_use: Some(true),
                ..Default::default()
            },
        );
    }
    config
}

/// An agent that pauses for error recovery at the first failed request.
fn failing_once_agent() -> AgentConfig {
    let mut config = agent_config("Friendly Coder");
    config.llm_error_policy = LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 1 };
    config
}

/// Automatic recovery's waits, too short to matter.
fn instant() -> RecoveryWaits {
    RecoveryWaits {
        initial: Duration::from_millis(1),
        max: Duration::from_millis(1),
        reset_margin: Duration::ZERO,
        max_jitter: Duration::ZERO,
    }
}

fn builder(
    server: &ResponsesServer,
    workspace: &Workspace,
    config: AgentConfig,
) -> WorkflowBuilder {
    Workflow::builder()
        .name("policy-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", config)
        .recovery_waits(instant())
}

fn status(code: u16) -> Reply {
    Reply::Status {
        status: code,
        retry_after: None,
        body: json!({"error": {"message": format!("synthetic {code}"), "type": "server_error"}})
            .to_string(),
    }
}

/// A server answering request `index` with `reply(index)`.
async fn scripted(reply: impl Fn(usize) -> Reply + Send + Sync + 'static) -> ResponsesServer {
    ResponsesServer::start_async(move |index, _| {
        let reply = reply(index);
        Box::pin(async move { Some(reply) })
    })
    .await
}

/// A probe call, then completion.
fn probe_then_done(index: usize) -> Reply {
    if index == 0 { probe_call() } else { done() }
}

/// A failure with `code`, then completion.
fn failing_then_done(code: u16) -> impl Fn(usize) -> Reply + Send + Sync + 'static {
    move |index| if index == 0 { status(code) } else { done() }
}

fn probe_call() -> Reply {
    Reply::Stream(tool_call_turn("p", "probe", json!({})))
}

fn done() -> Reply {
    Reply::Stream(complete_turn("done", "done"))
}

fn input(body: &str) -> Value {
    serde_json::from_str::<Value>(body).unwrap()["input"].clone()
}

/// The user text of a request body.
fn user_text(body: &str) -> String {
    let mut text = Vec::new();
    for item in input(body).as_array().unwrap() {
        if item["role"] != "user" {
            continue;
        }
        match &item["content"] {
            Value::String(content) => text.push(content.clone()),
            Value::Array(parts) => text.extend(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str().map(str::to_string)),
            ),
            _ => {}
        }
    }
    text.join("\n")
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .expect("the workflow call returned")
}

/// `ApproveAll` is the default: a call the agent's configuration marks for
/// approval runs, and the tool hooks get why it needed approval.
#[tokio::test]
async fn approve_all_is_the_default_and_hooks_get_the_approval_reason() {
    let workspace = Workspace::new();
    let server = scripted(probe_then_done).await;
    let probe = ProbeTool::new("probe", false);
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let workflow = builder(&server, &workspace, approving_agent(&[]))
        .global_tool(probe.clone())
        .hook(HookPoint::PreToolExecution, {
            let seen = seen.clone();
            Arc::new(move |hook| {
                seen.lock()
                    .push((hook.tool_name.clone(), hook.approval_reason.clone()));
                HookResult::Continue
            })
        })
        .build()
        .unwrap();
    let result = within(workflow.run_agent("coder", "Probe.")).await.unwrap();
    assert_eq!(result.summary, "done");
    assert_eq!(probe.calls(), 1);
    let seen = seen.lock().clone();
    let probe_reason = seen
        .iter()
        .find(|(tool, _)| tool.as_deref() == Some("probe"))
        .and_then(|(_, reason)| reason.clone())
        .expect("the hook saw the probe's approval reason");
    assert!(probe_reason.tool_not_auto_approved);
    // complete_task is auto-approved: no reason.
    assert!(
        seen.iter()
            .any(|(tool, reason)| tool.as_deref() == Some("complete_task") && reason.is_none())
    );
    // Warned about once.
    assert_eq!(workflow.context().approval_warnings.lock().len(), 1);
    assert!(workflow.context().approval_reasons.lock().is_empty());
}

/// `RespectAgentConfig` pauses the agent for its caller instead of running
/// the call; `OnAgentPaused` fires once and `OnWorkflowFailed` doesn't.
/// Resuming with a decision runs (or rejects) the call.
#[tokio::test]
async fn respect_agent_config_pauses_for_the_caller_and_resume_decides() {
    for approve in [true, false] {
        let workspace = Workspace::new();
        let server = scripted(probe_then_done).await;
        let probe = ProbeTool::new("probe", false);
        let hooks = HookCounts::default();
        let workflow = hooks
            .register(builder(&server, &workspace, approving_agent(&[])))
            .global_tool(probe.clone())
            .approval_policy(ApprovalPolicy::RespectAgentConfig)
            .build()
            .unwrap();
        let error = within(workflow.run_agent("coder", "Probe."))
            .await
            .unwrap_err();
        let WorkflowError::AgentPaused {
            task_id,
            pause: AgentPause::ToolApproval { calls },
            ..
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_name, "probe");
        assert!(calls[0].approval_reason.tool_not_auto_approved);
        assert_eq!(probe.calls(), 0);
        assert_eq!(hooks.get(HookPoint::OnAgentPaused), 1);
        assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 0);
        let paused = workflow.agent_result(task_id).unwrap();
        assert_eq!(paused.status, AgentRunStatus::Paused);

        let decision = if approve {
            ToolApprovalDecision::Approve {
                custom_tool_output: None,
                message_to_agent: None,
            }
        } else {
            ToolApprovalDecision::Reject {
                reason: Some("Not today.".into()),
            }
        };
        let resumed = within(workflow.resume_agent(
            &paused,
            AgentResume::ToolApprovals(HashMap::from([(calls[0].tool_call_id, decision)])),
        ))
        .await
        .unwrap();
        assert_eq!(resumed.status, AgentRunStatus::Completed);
        assert_eq!(resumed.agent_run_id, task_id);
        assert_eq!(probe.calls(), usize::from(approve));
        assert_eq!(server.requests(), 2);
        if !approve {
            assert!(server.bodies()[1].contains("Not today."));
        }
        assert_eq!(hooks.get(HookPoint::OnAgentPaused), 1);
        assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 0);
    }
}

/// Decides every call with its outcome and records the requests.
struct Deciding(
    ToolApprovalOutcome,
    parking_lot::Mutex<Vec<ToolApprovalRequest>>,
);

#[async_trait]
impl ToolApprovalHandler for Deciding {
    async fn decide(&self, request: ToolApprovalRequest) -> ToolApprovalOutcome {
        self.1.lock().push(request);
        self.0.clone()
    }
}

/// A handler decides each call the configuration marks for approval, as it
/// is asked; its `Pause` pauses the agent for the caller.
#[tokio::test]
async fn an_approval_handler_decides_each_call() {
    for outcome in [
        ToolApprovalOutcome::Approve,
        ToolApprovalOutcome::Reject {
            reason: Some("Not now.".into()),
        },
        ToolApprovalOutcome::Pause,
    ] {
        let workspace = Workspace::new();
        let server = scripted(probe_then_done).await;
        let probe = ProbeTool::new("probe", false);
        let handler = Arc::new(Deciding(outcome.clone(), Default::default()));
        let workflow = builder(&server, &workspace, approving_agent(&[]))
            .global_tool(probe.clone())
            .approval_policy(ApprovalPolicy::Handler(handler.clone()))
            .build()
            .unwrap();
        let result = within(workflow.run_agent("coder", "Probe.")).await;
        let requests = handler.1.lock().clone();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].agent_name, "coder");
        assert_eq!(requests[0].tool_name, "probe");
        assert!(requests[0].reason.tool_not_auto_approved);
        match outcome {
            ToolApprovalOutcome::Approve => {
                assert!(result.is_ok(), "{result:?}");
                assert_eq!(probe.calls(), 1);
            }
            ToolApprovalOutcome::Reject { .. } => {
                assert!(result.is_ok(), "{result:?}");
                assert_eq!(probe.calls(), 0);
                assert!(server.bodies()[1].contains("Not now."));
            }
            ToolApprovalOutcome::Pause => {
                assert!(matches!(
                    result,
                    Err(WorkflowError::AgentPaused {
                        pause: AgentPause::ToolApproval { .. },
                        ..
                    })
                ));
                assert_eq!(probe.calls(), 0);
            }
        }
    }
}

/// `Automatic` (the default) sends a request whose retries ran out again,
/// unchanged, with nothing added to the conversation.
#[tokio::test]
async fn automatic_recovery_sends_the_request_again_after_retries_run_out() {
    let workspace = Workspace::new();
    let server = scripted(failing_then_done(503)).await;
    let hooks = HookCounts::default();
    let workflow = hooks
        .register(builder(&server, &workspace, failing_once_agent()))
        .build()
        .unwrap();
    let result = within(workflow.run_agent("coder", "Write it.")).await;
    assert_eq!(result.unwrap().summary, "done");
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2);
    assert_eq!(input(&bodies[0]), input(&bodies[1]));
    assert!(!user_text(&bodies[1]).contains("approved"));
    assert_eq!(hooks.get(HookPoint::OnAgentPaused), 0);
}

/// `Automatic` fails a run clearly on an error retrying can't fix, without
/// sending it again; `OnWorkflowFailed` fires once.
#[tokio::test]
async fn automatic_recovery_fails_clearly_on_an_error_retrying_cant_fix() {
    let workspace = Workspace::new();
    let server = scripted(|_| status(401)).await;
    let hooks = HookCounts::default();
    let workflow = hooks
        .register(builder(&server, &workspace, agent_config("Friendly Coder")))
        .build()
        .unwrap();
    let error = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap_err();
    let WorkflowError::Agent(arko_agents::ArkoAgentsError::TaskExecution(reason)) = &error else {
        panic!("{error:?}");
    };
    assert!(
        reason.contains(
            "can't succeed without a change to the request, a credential, or the configuration"
        ),
        "{reason}"
    );
    assert_eq!(server.requests(), 1);
    assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 1);
    assert_eq!(hooks.get(HookPoint::PostAgentFailure), 1);
    assert_eq!(
        workflow.result().agent_results[0].status,
        AgentRunStatus::Failed
    );
}

/// `Limited { max }` recovers `max` times in a drive, then fails the run.
#[tokio::test]
async fn limited_recovery_stops_after_its_count() {
    let workspace = Workspace::new();
    let server = scripted(|_| status(503)).await;
    let workflow = builder(&server, &workspace, failing_once_agent())
        .error_recovery_policy(ErrorRecoveryPolicy::Limited { max: 1 })
        .build()
        .unwrap();
    let error = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("error recovery limit (1 automatic recoveries) was reached"),
        "{error}"
    );
    assert_eq!(server.requests(), 2);
}

/// `ReturnToCaller` pauses the agent at an error-recovery pause; resuming
/// continues (the request is sent again unchanged) or fails it.
#[tokio::test]
async fn return_to_caller_pauses_at_recovery_and_resume_continues_or_fails() {
    for continue_after in [true, false] {
        let workspace = Workspace::new();
        let server = scripted(failing_then_done(503)).await;
        let hooks = HookCounts::default();
        let workflow = hooks
            .register(builder(&server, &workspace, failing_once_agent()))
            .error_recovery_policy(ErrorRecoveryPolicy::ReturnToCaller)
            .build()
            .unwrap();
        let error = within(workflow.run_agent("coder", "Write it."))
            .await
            .unwrap_err();
        let WorkflowError::AgentPaused {
            task_id,
            pause:
                AgentPause::ErrorRecovery {
                    cause: ErrorRecoveryCause::RetriesExhausted { .. },
                    ..
                },
            ..
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(hooks.get(HookPoint::OnAgentPaused), 1);
        assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 0);
        let paused = workflow.agent_result(task_id).unwrap();
        // A decision for another kind of pause is refused.
        assert!(matches!(
            workflow
                .resume_agent(&paused, AgentResume::ToolApprovals(HashMap::new()))
                .await,
            Err(WorkflowError::BuilderInvalid(_))
        ));
        let resume = if continue_after {
            AgentResume::ContinueAfterError { note: None }
        } else {
            AgentResume::FailAfterError {
                reason: Some("Given up by the operator.".into()),
            }
        };
        let failed_before = hooks.get(HookPoint::OnWorkflowFailed);
        let resumed = within(workflow.resume_agent(&paused, resume)).await;
        if continue_after {
            assert_eq!(resumed.unwrap().summary, "done");
            let bodies = server.bodies();
            assert_eq!(input(&bodies[0]), input(&bodies[1]));
        } else {
            let error = resumed.unwrap_err();
            assert!(
                error.to_string().contains("Given up by the operator."),
                "{error}"
            );
            assert_eq!(server.requests(), 1);
            assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), failed_before + 1);
        }
        // Only a paused run is resumed.
        let recorded = workflow.agent_result(task_id).unwrap();
        assert!(matches!(
            workflow
                .resume_agent(&recorded, AgentResume::ContinueAfterError { note: None })
                .await,
            Err(WorkflowError::BuilderInvalid(_))
        ));
    }
}

/// An agent that pauses in a parallel run is not a failure: the others run
/// on, and its result comes back with theirs, to be continued. A failure is
/// returned, with `OnWorkflowFailed` once per call.
#[tokio::test]
async fn one_agent_pausing_in_a_parallel_run_is_not_a_failure() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        let text = user_text(body);
        Some(if text.contains("Ask.") && !text.contains("parser.rs") {
            tool_call_turn("q", "ask_user", json!({"questions": "Which file?"}))
        } else {
            complete_turn("done", "done")
        })
    })
    .await;
    let mut asking = agent_config("Friendly Lead");
    asking
        .tool_config
        .available_tool_names
        .push("ask_user".into());
    let hooks = HookCounts::default();
    let workflow = hooks
        .register(
            Workflow::builder()
                .name("parallel-pause-test")
                .workspace(&workspace.0)
                .llm_client(Arc::new(server.client()))
                .agent("lead", asking)
                .agent("worker", agent_config("Friendly Worker")),
        )
        .build()
        .unwrap();
    let results = within(workflow.run_agents_parallel(vec![
        ("lead", "Ask.".to_string()),
        ("worker", "Work.".to_string()),
    ]))
    .await
    .unwrap();
    assert_eq!(
        results
            .iter()
            .map(|result| (result.agent_name.as_str(), result.status.clone()))
            .collect::<Vec<_>>(),
        [
            ("lead", AgentRunStatus::WaitingForInput),
            ("worker", AgentRunStatus::Completed),
        ]
    );
    assert_eq!(hooks.get(HookPoint::OnUserInputRequired), 1);
    assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 0);
    let answered = within(workflow.continue_agent(&results[0], "parser.rs"))
        .await
        .unwrap();
    assert_eq!(answered.status, AgentRunStatus::Completed);
    // A run call returning the pause isn't a failure either.
    let paused = within(workflow.run_agent("lead", "Ask.")).await;
    assert!(matches!(
        paused,
        Err(WorkflowError::UserInputRequired { .. })
    ));
    assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 0);
}

/// Two failures in one parallel call fire `OnWorkflowFailed` once.
#[tokio::test]
async fn workflow_failed_fires_once_per_call() {
    let workspace = Workspace::new();
    let server = scripted(|_| status(401)).await;
    let hooks = HookCounts::default();
    let workflow = hooks
        .register(
            Workflow::builder()
                .name("parallel-failure-test")
                .workspace(&workspace.0)
                .llm_client(Arc::new(server.client()))
                .agent("lead", agent_config("Friendly Lead"))
                .agent("worker", agent_config("Friendly Worker")),
        )
        .build()
        .unwrap();
    let result = within(workflow.run_agents_parallel(vec![
        ("lead", "One.".to_string()),
        ("worker", "Two.".to_string()),
    ]))
    .await;
    assert!(result.is_err());
    assert_eq!(hooks.get(HookPoint::PostAgentFailure), 2);
    assert_eq!(hooks.get(HookPoint::OnWorkflowFailed), 1);
}

/// Sends a message from `worker` to the calling `lead` run, and returns once
/// it is in the run's command channel, so the runtime adds it at the turn
/// boundary after this call.
#[derive(Debug, Default)]
struct Relay(std::sync::OnceLock<Arc<crate::context::SharedContext>>);

#[async_trait]
impl Tool for Relay {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "relay".to_string(),
            description: "Relays a message.".to_string(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        task: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let context = self.0.get().expect("the workflow is built");
        context
            .messages
            .send("worker", None, "lead", "Check the parser.".into(), vec![])
            .await
            .map_err(ToolError::ExecutionFailed)?;
        while !context
            .messages
            .unseen_messages("lead", task.task_id)
            .forwarded
        {
            tokio::task::yield_now().await;
        }
        Ok("relayed".to_string())
    }
}

/// Continuing a run that waited for input presents no message its
/// conversation already holds: one added at a turn boundary before the
/// question is not in the continuation prompt, and the run's completion
/// acknowledges it. (The continuation once presented it again.)
#[tokio::test]
async fn continuing_after_waiting_for_input_presents_no_message_twice() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn("r", "relay", json!({})),
            1 => tool_call_turn("q", "ask_user", json!({"questions": "Which file?"})),
            _ => complete_turn("done", "done"),
        })
    })
    .await;
    let mut lead = agent_config("Friendly Lead");
    lead.tool_config
        .available_tool_names
        .push("ask_user".into());
    let relay = Arc::new(Relay::default());
    let workflow = Workflow::builder()
        .name("message-pause-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", lead)
        .agent_tool("lead", relay.clone())
        .build()
        .unwrap();
    relay.0.set(workflow.context().clone()).unwrap();
    let error = within(workflow.run_agent("lead", "Lead the work."))
        .await
        .unwrap_err();
    let WorkflowError::UserInputRequired { task_id, .. } = error else {
        panic!("{error:?}");
    };
    // Delivered at the boundary after the relay.
    assert_eq!(
        user_text(&server.bodies()[1])
            .matches("Check the parser.")
            .count(),
        1
    );
    let waiting = workflow.agent_result(task_id).unwrap();
    assert!(waiting.task.pending_context_injections.is_empty());
    assert_eq!(
        workflow.context().messages.pending("lead")[0].held_by,
        [task_id]
    );
    let answered = within(workflow.continue_agent(&waiting, "parser.rs"))
        .await
        .unwrap();
    assert_eq!(answered.status, AgentRunStatus::Completed);
    let last = user_text(&server.bodies()[2]);
    assert_eq!(last.matches("Check the parser.").count(), 1, "{last}");
    assert!(!last.contains("--- Message "), "{last}");
    assert!(workflow.context().messages.pending("lead").is_empty());
}

/// Resuming a paused run delivers no message it already holds, and a
/// message sent while it was paused arrives once.
#[tokio::test]
async fn resuming_a_paused_run_delivers_no_message_twice() {
    let workspace = Workspace::new();
    let server = scripted(|index| match index {
        0 => Reply::Stream(tool_call_turn("r", "relay", json!({}))),
        1 => probe_call(),
        _ => done(),
    })
    .await;
    let relay = Arc::new(Relay::default());
    let workflow = Workflow::builder()
        .name("message-resume-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", approving_agent(&["relay"]))
        .agent_tool("lead", relay.clone())
        .global_tool(ProbeTool::new("probe", false))
        .approval_policy(ApprovalPolicy::RespectAgentConfig)
        .build()
        .unwrap();
    relay.0.set(workflow.context().clone()).unwrap();
    let error = within(workflow.run_agent("lead", "Lead the work."))
        .await
        .unwrap_err();
    let WorkflowError::AgentPaused {
        task_id,
        pause: AgentPause::ToolApproval { calls },
        ..
    } = error
    else {
        panic!("{error:?}");
    };
    let later = workflow
        .context()
        .messages
        .send(
            "worker",
            None,
            "lead",
            "Also check the lexer.".into(),
            vec![],
        )
        .await
        .unwrap();
    let paused = workflow.agent_result(task_id).unwrap();
    let resumed = within(workflow.resume_agent(
        &paused,
        AgentResume::ToolApprovals(HashMap::from([(
            calls[0].tool_call_id,
            ToolApprovalDecision::Approve {
                custom_tool_output: None,
                message_to_agent: None,
            },
        )])),
    ))
    .await
    .unwrap();
    assert_eq!(resumed.status, AgentRunStatus::Completed);
    let last = user_text(&server.bodies()[2]);
    assert_eq!(last.matches("Check the parser.").count(), 1, "{last}");
    assert_eq!(last.matches(&later.content).count(), 1, "{last}");
    assert!(workflow.context().messages.pending("lead").is_empty());
}

/// A message a run took from its channel but hadn't added when it stopped
/// (`Task::pending_context_injections`) reaches it once when it is
/// continued: from the task, not again in the prompt.
#[tokio::test]
async fn a_message_taken_but_not_added_reaches_the_continued_run_once() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn("q", "ask_user", json!({"questions": "Which file?"})),
            _ => complete_turn("done", "done"),
        })
    })
    .await;
    let mut lead = agent_config("Friendly Lead");
    lead.tool_config
        .available_tool_names
        .push("ask_user".into());
    let workflow = Workflow::builder()
        .name("message-taken-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", lead)
        .build()
        .unwrap();
    let error = within(workflow.run_agent("lead", "Lead the work."))
        .await
        .unwrap_err();
    let WorkflowError::UserInputRequired { task_id, .. } = error else {
        panic!("{error:?}");
    };
    // As when the runtime takes a message just as the task suspends: the
    // task keeps it, and finalization records that it holds it.
    let context = workflow.context();
    let message = context
        .messages
        .send("worker", None, "lead", "Check the parser.".into(), vec![])
        .await
        .unwrap();
    let mut waiting = workflow.agent_result(task_id).unwrap();
    waiting.task.pending_context_injections = vec![crate::messaging::injection(&message)];
    context.messages.settle_run(
        "lead",
        task_id,
        &waiting.task.conversation,
        &[],
        &waiting.task.pending_context_injections,
        false,
    );
    assert!(context.messages.pending_for("lead", task_id).is_empty());
    let answered = within(workflow.continue_agent(&waiting, "parser.rs"))
        .await
        .unwrap();
    assert_eq!(answered.status, AgentRunStatus::Completed);
    let last = user_text(&server.bodies()[1]);
    assert_eq!(last.matches("Check the parser.").count(), 1, "{last}");
    assert!(context.messages.pending("lead").is_empty());
}

/// A server answering by prompt: `Fail.` with a 401 (the run fails),
/// `Hold.` never (until `release` is notified), anything else completing.
async fn by_prompt(release: Arc<tokio::sync::Notify>) -> ResponsesServer {
    ResponsesServer::start_async(move |_, body| {
        let release = release.clone();
        Box::pin(async move {
            let text = user_text(&body);
            if text.contains("Fail.") {
                Some(status(401))
            } else if text.contains("Hold.") {
                release.notified().await;
                Some(done())
            } else {
                Some(done())
            }
        })
    })
    .await
}

fn two_agents(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("parallel-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", agent_config("Friendly Lead"))
        .agent("worker", agent_config("Friendly Worker"))
}

/// With `max_parallel_agents(1)` the second agent starts only when the
/// first has finished (it is not running meanwhile); results keep input
/// order.
#[tokio::test]
async fn a_concurrency_cap_starts_agents_as_places_free_up() {
    let workspace = Workspace::new();
    let release = Arc::new(tokio::sync::Notify::new());
    let server = by_prompt(release.clone()).await;
    let workflow = Arc::new(
        two_agents(&server, &workspace)
            .max_parallel_agents(1)
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("lead", "Hold.".to_string()),
                    ("worker", "Work.".to_string()),
                ])
                .await
        }
    });
    server.wait_for_requests(1).await;
    assert_eq!(
        workflow.agent_status("worker"),
        Some(crate::messaging::registry::AgentStatus::NotStarted)
    );
    release.notify_one();
    let results = within(run).await.unwrap().unwrap();
    assert_eq!(
        results
            .iter()
            .map(|result| result.agent_name.as_str())
            .collect::<Vec<_>>(),
        ["lead", "worker"]
    );
    assert_eq!(server.requests(), 2);
    // Zero is refused.
    assert!(matches!(
        two_agents(&server, &workspace)
            .max_parallel_agents(0)
            .build(),
        Err(WorkflowError::BuilderInvalid(_))
    ));
}

/// With `fail_fast(true)` the first failure stops the other agents (a
/// running one is cancelled, one not started never starts), and the call
/// returns that failure.
#[tokio::test]
async fn fail_fast_stops_the_other_agents_at_the_first_failure() {
    for cap in [None, Some(1)] {
        let workspace = Workspace::new();
        let server = by_prompt(Arc::new(tokio::sync::Notify::new())).await;
        let mut builder = two_agents(&server, &workspace).fail_fast(true);
        if let Some(cap) = cap {
            builder = builder.max_parallel_agents(cap);
        }
        let workflow = builder.build().unwrap();
        let (first, second) = if cap.is_some() {
            // The failing agent takes the only place.
            (("lead", "Fail."), ("worker", "Hold."))
        } else {
            (("worker", "Hold."), ("lead", "Fail."))
        };
        let error = within(workflow.run_agents_parallel(vec![
            (first.0, first.1.to_string()),
            (second.0, second.1.to_string()),
        ]))
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("can't succeed without a change"),
            "{error}"
        );
        let held = workflow
            .result()
            .agent_results
            .into_iter()
            .find(|result| result.agent_name == "worker");
        match cap {
            None => {
                let held = held.expect("the held agent is recorded");
                assert_eq!(held.status, AgentRunStatus::Cancelled);
                assert!(
                    held.error.as_deref().unwrap().contains("fail_fast"),
                    "{:?}",
                    held.error
                );
            }
            // It never started.
            Some(_) => assert!(held.is_none()),
        }
    }
}

/// An agent running inside another agent's tool call (here a completion
/// verifier) can't be paused for the caller: under `RespectAgentConfig` a
/// call it needs approved (even its own `complete_task`, which a default
/// configuration doesn't auto-approve) stops it with a reason, so the review
/// fails, a bounded number of times, and the call returns `VerifierFailed`.
/// No paused result is left behind, and the verifier isn't asked again and
/// again. The same holds for its question without a user input handler.
#[tokio::test]
async fn a_nested_run_that_needs_a_decision_is_stopped_not_paused() {
    for question in [false, true] {
        let workspace = Workspace::new();
        let server = ResponsesServer::start(move |index, body| {
            Some(if body.contains("You are a completion verifier") {
                if question {
                    tool_call_turn(
                        &format!("q{index}"),
                        "ask_user",
                        json!({"questions": "Which file?"}),
                    )
                } else {
                    complete_turn(&format!("v{index}"), "ACCEPTED")
                }
            } else {
                complete_turn(&format!("c{index}"), "done")
            })
        })
        .await;
        let mut reviewer = agent_config("Friendly Reviewer");
        // A default configuration: every call needs approval, complete_task
        // included (ask_user is exempt).
        reviewer.tool_config.default_auto_approve_tool_use = false;
        reviewer
            .tool_config
            .available_tool_names
            .push("ask_user".into());
        let hooks = HookCounts::default();
        let workflow = hooks
            .register(
                Workflow::builder()
                    .name("nested-approval-test")
                    .workspace(&workspace.0)
                    .llm_client(Arc::new(server.client()))
                    .agent("coder", approving_agent(&[]))
                    .agent("reviewer", reviewer)
                    .completion_verifier(
                        crate::verification::CompletionVerifierConfig::correctness(
                            "check", "reviewer",
                        )
                        .target("coder")
                        .review_retries(1),
                    )
                    .approval_policy(ApprovalPolicy::RespectAgentConfig)
                    .recovery_waits(instant()),
            )
            .build()
            .unwrap();
        let error = within(workflow.run_agent("coder", "Write it."))
            .await
            .unwrap_err();
        assert!(
            matches!(&error, WorkflowError::VerifierFailed { reason, .. } if reason.contains("can't be paused")),
            "{error:?}"
        );
        // Two reviews (one retry), one verifier request each.
        let verifier_requests = server
            .bodies()
            .iter()
            .filter(|body| body.contains("You are a completion verifier"))
            .count();
        assert_eq!(verifier_requests, 2, "question: {question}");
        assert_eq!(hooks.get(HookPoint::OnAgentPaused), 0);
        assert_eq!(hooks.get(HookPoint::OnUserInputRequired), 0);
        let results = workflow.result().agent_results;
        assert!(results.iter().all(|result| !result.status.is_paused()));
        let reviews = results
            .iter()
            .filter(|result| result.agent_name == "reviewer")
            .collect::<Vec<_>>();
        assert_eq!(reviews.len(), 2);
        assert!(reviews.iter().all(|review| {
            review.status == AgentRunStatus::Cancelled
                && review
                    .error
                    .as_deref()
                    .is_some_and(|reason| reason.contains("can't be paused"))
        }));
    }
}

/// A run waiting for input in a checkpoint saved before messages recorded
/// who holds them still gets no message twice when continued: what its
/// conversation holds is recorded again first.
#[tokio::test]
async fn a_waiting_run_from_an_older_checkpoint_gets_no_message_twice() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn("r", "relay", json!({})),
            1 => tool_call_turn("q", "ask_user", json!({"questions": "Which file?"})),
            _ => complete_turn("done", "done"),
        })
    })
    .await;
    let mut lead = agent_config("Friendly Lead");
    lead.tool_config
        .available_tool_names
        .push("ask_user".into());
    let run_dir = workspace.0.join(".run");
    let build = |resume: Option<&std::path::Path>| {
        let relay = Arc::new(Relay::default());
        let mut builder = Workflow::builder()
            .name("legacy-message-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("lead", lead.clone())
            .agent_tool("lead", relay.clone())
            .persistence(crate::persistence::PersistenceConfig::new(&run_dir));
        if let Some(path) = resume {
            builder = builder.resume_from(path);
        }
        let workflow = builder.build().unwrap();
        relay.0.set(workflow.context().clone()).unwrap();
        workflow
    };
    let first = build(None);
    let error = within(first.run_agent("lead", "Lead the work."))
        .await
        .unwrap_err();
    let WorkflowError::UserInputRequired { task_id, .. } = error else {
        panic!("{error:?}");
    };
    drop(first);
    // As an older version saved it: no held_by.
    let name = std::fs::read_to_string(run_dir.join("latest-checkpoint")).unwrap();
    let path = run_dir.join(name.trim());
    let mut checkpoint: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for message in checkpoint["message_queues"]["lead"].as_array_mut().unwrap() {
        message.as_object_mut().unwrap().remove("held_by");
    }
    std::fs::write(&path, checkpoint.to_string()).unwrap();

    let resumed = build(Some(&path));
    let waiting = resumed.agent_result(task_id).unwrap();
    within(resumed.continue_agent(&waiting, "parser.rs"))
        .await
        .unwrap();
    let last = user_text(server.bodies().last().unwrap());
    assert_eq!(last.matches("Check the parser.").count(), 1, "{last}");
    assert!(resumed.context().messages.pending("lead").is_empty());
}

/// `Automatic` waits out a plan usage limit until its reported reset, then
/// sends the request again unchanged.
#[tokio::test]
async fn automatic_recovery_waits_out_a_usage_limit() {
    let workspace = Workspace::new();
    let server = scripted(|index| {
        if index == 0 {
            Reply::Status {
                status: 429,
                retry_after: Some(1),
                body: json!({"error": {"message": "limit reached", "type": "rate_limit_error"}})
                    .to_string(),
            }
        } else {
            done()
        }
    })
    .await;
    let mut config = agent_config("Friendly Coder");
    // Any provider delay counts as a usage limit here.
    config.llm_retry_config.rate_limit_retry = Some(arko_agents::LLMRateLimitRetryConfig {
        initial_delay_ms: 1,
        max_delay_ms: 1,
        max_wait_ms: 1,
    });
    let workflow = builder(&server, &workspace, config).build().unwrap();
    let causes = pause_causes(&workflow);
    let started = std::time::Instant::now();
    let result = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap();
    assert_eq!(result.summary, "done");
    // The reset was a second away.
    assert!(started.elapsed() >= Duration::from_millis(500));
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2);
    assert_eq!(input(&bodies[0]), input(&bodies[1]));
    assert!(matches!(
        seen(&causes).await.as_slice(),
        [ErrorRecoveryCause::UsageLimit {
            resets_at_unix_ms: Some(_)
        }]
    ));
}

/// The causes of the error-recovery pauses the workflow's agents report,
/// as they are broadcast.
fn pause_causes(workflow: &Workflow) -> Arc<parking_lot::Mutex<Vec<ErrorRecoveryCause>>> {
    let causes = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut events = workflow.subscribe_events();
    tokio::spawn({
        let causes = causes.clone();
        async move {
            while let Ok(event) = events.recv().await {
                if let arko_agents::AgentEvent::TaskStateChanged(
                    arko_agents::TaskState::AwaitingErrorRecoveryApproval { cause, .. },
                ) = event.event
                {
                    causes.lock().push(cause);
                }
            }
        }
    });
    causes
}

/// The causes collected so far, once there is at least one.
async fn seen(causes: &parking_lot::Mutex<Vec<ErrorRecoveryCause>>) -> Vec<ErrorRecoveryCause> {
    within(async {
        loop {
            let seen = causes.lock().clone();
            if !seen.is_empty() {
                return seen;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
}

/// `Automatic` continues at once after a pause on tool errors (the agent's
/// opt-in `ToolErrorPolicy::PauseAndAwaitApproval`); the call's result
/// reports the error, and nothing says a user approved.
#[tokio::test]
async fn automatic_recovery_continues_after_tool_errors() {
    let workspace = Workspace::new();
    let server = scripted(|index| {
        if index == 0 {
            Reply::Stream(tool_call_turn(
                "r",
                "read_file",
                json!({"path": "missing.txt"}),
            ))
        } else {
            done()
        }
    })
    .await;
    let mut config = agent_config("Friendly Coder");
    config
        .tool_config
        .available_tool_names
        .push("read_file".into());
    config.error_recovery_config.on_tool_execution_error =
        arko_agents::ToolErrorPolicy::PauseAndAwaitApproval;
    let workflow = builder(&server, &workspace, config).build().unwrap();
    let causes = pause_causes(&workflow);
    let result = within(workflow.run_agent("coder", "Read it."))
        .await
        .unwrap();
    assert_eq!(result.summary, "done");
    assert_eq!(seen(&causes).await, [ErrorRecoveryCause::ToolError]);
    let second = &server.bodies()[1];
    assert!(second.contains("missing.txt"), "{second}");
    assert!(!user_text(second).contains("approved"), "{second}");
}
