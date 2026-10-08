use super::*;
use crate::agent_config::{AgentLLMConfig, AgentToolConfig};
use arko_ai::{GenerationConfig, Model};

/// Test helpers that run native calls outside a turn.
impl AgentLoop {
    /// Runs one auto-approved native call (see
    /// [`run_native_calls`](Self::run_native_calls)) and returns its result.
    #[cfg(test)]
    pub(super) async fn execute_native_tool(
        &mut self,
        ai_tool_call: &ai::ToolCall,
        internal_id: Uuid,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        let tool = self
            .tool_registry
            .get(&ai_tool_call.name)
            .ok_or_else(|| ArkoAgentsError::ToolNotFound(ai_tool_call.name.clone()))?;
        let params = Self::tool_call_args_to_map(ai_tool_call);
        let run = NativeRun {
            index: 0,
            call: ai_tool_call.clone(),
            run: self.tool_run(internal_id, &ai_tool_call.name, tool, params, None),
            message_to_agent: None,
        };
        self.run_native_calls_for_result(run, |_| true).await
    }

    /// Runs one native call and takes its result out of the task.
    #[cfg(test)]
    async fn run_native_calls_for_result(
        &mut self,
        run: NativeRun,
        keep_going: fn(&TaskState) -> bool,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        let mut results = NativeResults::new(&self.task);
        self.run_native_calls(vec![run], keep_going, Default::default(), &mut results)
            .await?;
        let recorded = results.positions.len();
        results.finish(&mut self.task);
        Ok(if recorded > 0 {
            self.task.pending_native_tool_results.pop()
        } else {
            None
        })
    }

    /// Runs one approved native call (see
    /// [`run_native_calls`](Self::run_native_calls)) and returns its result.
    #[cfg(test)]
    pub(super) async fn execute_native_tool_for_approval(
        &mut self,
        ai_call: &ai::ToolCall,
        internal_id: Uuid,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        custom_output: Option<String>,
        message_to_agent: Option<String>,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        match self.approved_native_run(
            0,
            ai_call,
            internal_id,
            tool_name,
            parameters,
            custom_output,
            message_to_agent,
        ) {
            Ok(run) => {
                self.run_native_calls_for_result(run, |state| !state.is_terminal())
                    .await
            }
            Err(()) => self.unavailable_native_tool(ai_call).await.map(Some),
        }
    }
}

pub(super) fn test_config() -> AgentConfig {
    AgentConfig::builder(
        "batch-test".to_string(),
        "Batch Test".to_string(),
        AgentLLMConfig {
            model: Model::openai_gpt_5_6_luna(),
            generation_config: GenerationConfig::default(),
        },
    )
    .role_description("test".to_string())
    .instructions("test".to_string())
    .tool_config(AgentToolConfig::default())
    .build()
    .unwrap()
}

/// Tells the task about something once, at its first turn boundary.
struct TellOnce(std::sync::atomic::AtomicBool);

impl super::super::TurnContextSource for TellOnce {
    fn turn_context(&self, _task_id: Uuid) -> Option<String> {
        (!self.0.swap(true, std::sync::atomic::Ordering::SeqCst))
            .then(|| "Agent `b` is now running.".to_string())
    }
}

/// A turn context source's content reaches the conversation before the
/// next model request, as a workflow context update, once.
#[tokio::test]
async fn turn_context_is_added_before_the_next_request() {
    use futures::StreamExt;
    let (mut agent, _events, _commands) = super::retry_tests::harness();
    let tool_config = &mut agent.task.agent_config.tool_config;
    tool_config.tool_calling_mode = ToolCallingMode::Native;
    tool_config.default_auto_approve_tool_use = true;
    agent.turn_context = Some(Arc::new(TellOnce(Default::default())));
    let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(turn.boxed()));
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    let updates = task
        .conversation
        .messages
        .iter()
        .filter(|message| message.role == ai::Role::User)
        .filter_map(|message| message.content.iter().find_map(ai::Content::as_text))
        .filter(|text| text.starts_with("# Workflow context update\n\nAgent `b` is now running."))
        .count();
    assert_eq!(updates, 1);
}

/// Holds every request while its flag is set.
struct HoldWhile(Arc<std::sync::atomic::AtomicBool>);

impl super::super::TurnContextSource for HoldWhile {
    fn turn_context(&self, _task_id: Uuid) -> Option<String> {
        None
    }

    fn hold_request(&self, _task_id: Uuid) -> Option<String> {
        self.0
            .load(std::sync::atomic::Ordering::SeqCst)
            .then(|| "Budget exceeded.".to_string())
    }
}

/// A held request isn't made: the task pauses with the
/// `BudgetExceeded` cause, whatever its LLM error policy, and
/// continuing asks again; once released, the request is made without
/// an error note.
#[tokio::test]
async fn a_held_request_pauses_the_task_until_it_is_released() {
    use futures::StreamExt;
    let (mut agent, mut events, commands) = super::retry_tests::harness();
    agent.task.agent_config.llm_error_policy = LLMErrorPolicy::InformAgentAndContinue;
    let tool_config = &mut agent.task.agent_config.tool_config;
    tool_config.tool_calling_mode = ToolCallingMode::Native;
    tool_config.default_auto_approve_tool_use = true;
    let hold = Arc::new(std::sync::atomic::AtomicBool::new(true));
    agent.turn_context = Some(Arc::new(HoldWhile(hold.clone())));
    let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(turn.boxed()));
    let run = tokio::spawn(agent.run());
    let mut pauses = 0;
    while let Some(event) = timeout(Duration::from_secs(10), events.recv())
        .await
        .unwrap()
    {
        match event {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                cause,
                ..
            }) => assert_eq!(cause, ErrorRecoveryCause::BudgetExceeded),
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } => {
                pauses += 1;
                if pauses == 2 {
                    hold.store(false, std::sync::atomic::Ordering::SeqCst);
                }
                commands
                    .send(UserCommand::ContinueAfterError { note: None })
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
    assert_eq!(pauses, 2);
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(!task.monitor_pause);
    let notes = task
        .conversation
        .messages
        .iter()
        .filter_map(|message| message.content.iter().find_map(ai::Content::as_text))
        .filter(|text| text.contains("Budget exceeded") || text.contains("error"))
        .count();
    assert_eq!(notes, 0, "no note reaches the model");
}

/// Every context injection queued before a turn boundary reaches the
/// next request, in one workflow context update, not one per request.
#[tokio::test]
async fn every_queued_injection_reaches_the_next_request() {
    use futures::StreamExt;
    let (mut agent, _events, commands) = super::retry_tests::harness();
    let tool_config = &mut agent.task.agent_config.tool_config;
    tool_config.tool_calling_mode = ToolCallingMode::Native;
    tool_config.default_auto_approve_tool_use = true;
    for content in ["First message.", "Second message."] {
        commands
            .try_send(UserCommand::InjectContext {
                content: content.into(),
            })
            .unwrap();
    }
    let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(turn.boxed()));
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    let messages = &task.conversation.messages;
    let update = messages
        .iter()
        .position(|message| {
            message.role == ai::Role::User
                && message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .any(|text| {
                        text.starts_with(
                            "# Workflow context update\n\nFirst message.\n\nSecond message.",
                        )
                    })
        })
        .expect("one update carries both injections");
    let answer = messages
        .iter()
        .position(|message| message.role == ai::Role::Assistant)
        .unwrap();
    assert!(update < answer, "the update precedes the only request");
}

/// Context taken from the channel together with a suspension stays with
/// the suspended task.
#[tokio::test]
async fn an_injection_taken_with_a_suspension_is_kept() {
    let (agent, _events, commands) = super::retry_tests::harness();
    commands
        .try_send(UserCommand::InjectContext {
            content: "Late note.".into(),
        })
        .unwrap();
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    let kept = task.conversation.messages.iter().any(|message| {
        message
            .content
            .iter()
            .filter_map(ai::Content::as_text)
            .any(|text| text.starts_with("# Workflow context update\n\nLate note."))
    });
    assert!(kept, "{:?}", task.conversation.messages);
    assert!(
        task.conversation
            .messages
            .iter()
            .all(|message| message.role != ai::Role::Assistant),
        "no request was made"
    );
}

#[test]
fn retry_events_saved_before_the_rate_limit_flag_still_load() {
    let event: AgentEvent = serde_json::from_value(serde_json::json!({
        "RetryScheduled": {"attempt": 2, "delay_ms": 1000, "elapsed_ms": 50}
    }))
    .unwrap();
    assert_eq!(
        event,
        AgentEvent::RetryScheduled {
            attempt: 2,
            delay_ms: 1000,
            elapsed_ms: 50,
            rate_limit: false,
        }
    );
}

/// Cancels the task at every call.
struct CancelEveryCall;

#[async_trait::async_trait]
impl ToolExecutionMiddleware for CancelEveryCall {
    async fn before(
        &self,
        _context: &ToolExecutionContext<'_>,
        _call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, crate::tool::ToolError> {
        Ok(ToolExecutionDecision::CancelTask {
            reason: "Run cancelled.".into(),
        })
    }

    async fn after(
        &self,
        _context: &ToolExecutionContext<'_>,
        _call: &ToolCallContext<'_>,
        _outcome: ToolExecutionOutcome<'_>,
    ) -> Result<(), crate::tool::ToolError> {
        panic!("a cancelled call does not run");
    }
}

/// A middleware's cancel decision rejects the call and ends the task as
/// cancelled, for an auto-approved call and for an approved one.
#[tokio::test]
async fn a_middleware_cancels_the_task_at_a_tool_call() {
    use futures::StreamExt;
    for approval in [false, true] {
        let (mut agent, mut events, commands) = super::retry_tests::harness();
        let tool_config = &mut agent.task.agent_config.tool_config;
        tool_config.tool_calling_mode = ToolCallingMode::Native;
        tool_config.default_auto_approve_tool_use = !approval;
        agent.execution_middleware = Some(Arc::new(CancelEveryCall));
        let turn = futures::stream::iter([Ok(super::retry_tests::complete_response())]);
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(turn.boxed()));
        let runner = tokio::spawn(agent.run());
        let mut received = Vec::new();
        while let Some(event) = timeout(Duration::from_secs(10), events.recv())
            .await
            .unwrap()
        {
            if let AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } = &event {
                commands
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id: *tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await
                    .unwrap();
            }
            received.push(event);
        }
        let task = runner.await.unwrap().unwrap();
        assert_eq!(
            task.current_state,
            TaskState::Cancelled {
                reason: "Run cancelled.".into()
            }
        );
        assert!(received.iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { reason: Some(reason), .. } if reason == "Run cancelled."
        )));
        assert!(
            !received
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolResult { .. }))
        );
    }
}
