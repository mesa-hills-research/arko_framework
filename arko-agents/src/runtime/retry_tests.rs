use super::*;
use crate::agent_config::{AgentLLMConfig, LLMCapacityRetryConfig, LLMRetryConfig};

pub(super) fn harness() -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let config = AgentConfig::builder(
        "retry-test".into(),
        "Retry test".into(),
        AgentLLMConfig {
            model: ai::Model::openai_gpt_5_6_luna(),
            generation_config: Default::default(),
        },
    )
    .role_description("test".into())
    .instructions("test".into())
    .llm_error_policy(LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 10 })
    .llm_retry_config(LLMRetryConfig {
        jitter: false,
        ..Default::default()
    })
    .build()
    .unwrap();
    let task = Task::new(config, "test".into(), std::env::temp_dir(), None);
    let (events, rx) = mpsc::channel(100);
    let (tx, commands) = mpsc::channel(10);
    (
        AgentLoop::new(
            task,
            Arc::new(ai::LLMClient::new()),
            Arc::new(ToolRegistry::new()),
            events,
            commands,
            None,
            false,
            None,
            None,
            None,
        ),
        rx,
        tx,
    )
}

pub(super) fn provider_failure(status: u16, delay: Option<Duration>) -> ai::Error {
    ai::Error::ProviderError {
        message: "synthetic outage".into(),
        status: Some(status),
        code: None,
        retry_after: delay,
    }
}

pub(super) fn capacity_failure(delay: Option<Duration>) -> ai::Error {
    ai::Error::ProviderError {
        message: "synthetic capacity outage".into(),
        status: None,
        code: Some("server_is_overloaded".into()),
        retry_after: delay,
    }
}

#[tokio::test(start_paused = true)]
async fn capacity_wait_survives_hours_without_prompt_growth_or_a_request_deadline() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
    let original_messages = agent.task.conversation.messages.len();
    let started = tokio::time::Instant::now();
    for _ in 0..20 {
        agent
            .run_llm_attempt::<()>(async { Err(capacity_failure(None)) })
            .await
            .unwrap();
        assert!(matches!(agent.task.current_state, TaskState::InProgress));
    }
    assert!(started.elapsed() > Duration::from_secs(3600));
    assert_eq!(agent.capacity_retry_attempts, 20);
    assert_eq!(agent.task.conversation.messages.len(), original_messages);
    assert!(agent.recovery_deadline().is_none());
    let result = agent
        .run_llm_attempt(async {
            tokio_sleep(Duration::from_secs(600)).await;
            Ok("recovered")
        })
        .await
        .unwrap();
    assert_eq!(result, Some("recovered"));
    let mut delays = Vec::new();
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::RetryScheduled { delay_ms, .. } => delays.push(delay_ms),
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } => panic!("capacity exhausted"),
            _ => {}
        }
    }
    assert_eq!(&delays[..5], &[30_000, 60_000, 120_000, 240_000, 300_000]);
    assert!(delays[5..].iter().all(|d| *d == 300_000));
}

#[tokio::test(start_paused = true)]
async fn capacity_wait_honors_server_hints_and_remains_cancellable_and_suspendable() {
    for suspend in [false, true] {
        for in_backoff in [false, true] {
            let (mut agent, mut events, commands) = harness();
            agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
            if !in_backoff {
                let started = tokio::time::Instant::now();
                agent
                    .run_llm_attempt::<()>(async {
                        Err(capacity_failure(Some(Duration::from_secs(600))))
                    })
                    .await
                    .unwrap();
                assert_eq!(started.elapsed(), Duration::from_secs(600));
            }
            let started = tokio::time::Instant::now();
            let sender = tokio::spawn(async move {
                commands
                    .send(UserCommand::InjectContext {
                        content: "peer progress".into(),
                    })
                    .await
                    .unwrap();
                tokio_sleep(Duration::from_millis(10)).await;
                commands
                    .send(if suspend {
                        UserCommand::SuspendTask
                    } else {
                        UserCommand::CancelTask
                    })
                    .await
                    .unwrap();
            });
            if in_backoff {
                agent
                    .run_llm_attempt::<()>(async {
                        Err(capacity_failure(Some(Duration::from_secs(600))))
                    })
                    .await
                    .unwrap();
            } else {
                agent
                    .run_llm_attempt(std::future::pending::<Result<(), ai::Error>>())
                    .await
                    .unwrap();
            }
            sender.await.unwrap();
            assert_eq!(started.elapsed(), Duration::from_millis(10));
            assert!(agent.stop_requested);
            assert_eq!(
                matches!(agent.task.current_state, TaskState::Cancelled { .. }),
                !suspend
            );
            while let Ok(event) = events.try_recv() {
                assert!(!matches!(
                    event,
                    AgentEvent::ErrorEncountered {
                        requires_approval_to_retry: true,
                        ..
                    }
                ));
            }
        }
    }
}

#[tokio::test(start_paused = true)]
async fn capacity_wait_does_not_disable_permanent_errors_or_other_retry_limits() {
    for permanent in [false, true] {
        let (mut agent, _events, _commands) = harness();
        agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
        agent.task.agent_config.llm_error_policy =
            LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 2 };
        agent
            .run_llm_attempt::<()>(async { Err(capacity_failure(Some(Duration::from_secs(3600)))) })
            .await
            .unwrap();
        let started = tokio::time::Instant::now();
        agent
            .run_llm_attempt::<()>(async {
                Err(provider_failure(if permanent { 401 } else { 503 }, None))
            })
            .await
            .unwrap();
        if !permanent {
            assert_eq!(started.elapsed(), Duration::from_millis(250));
            assert!(matches!(agent.task.current_state, TaskState::InProgress));
            agent
                .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
                .await
                .unwrap();
        } else {
            assert_eq!(started.elapsed(), Duration::ZERO);
        }
        assert!(matches!(
            agent.task.current_state,
            TaskState::AwaitingErrorRecoveryApproval { .. }
        ));
        // The capacity request counted toward neither ordinary limit.
        assert_eq!(agent.indefinite_interactions, 1);
        assert!(agent.ordinary_attempts() <= 2);
    }
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
    agent.task.agent_config.llm_error_policy = LLMErrorPolicy::PauseAndAwaitApproval;
    agent
        .run_llm_attempt::<()>(async { Err(capacity_failure(None)) })
        .await
        .unwrap();
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
    agent
        .run_llm_attempt::<()>(async { Err(capacity_failure(Some(Duration::MAX))) })
        .await
        .unwrap();
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
}

#[tokio::test(start_paused = true)]
async fn temporary_failure_recovers_in_the_same_task_and_honors_server_delay() {
    let (mut agent, mut events, _commands) = harness();
    let id = agent.task.id;
    let started = tokio::time::Instant::now();
    assert!(
        agent
            .run_llm_attempt::<()>(async {
                Err(provider_failure(429, Some(Duration::from_secs(60))))
            })
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(started.elapsed(), Duration::from_secs(60));
    assert_eq!(
        agent
            .run_llm_attempt(async { Ok("recovered") })
            .await
            .unwrap(),
        Some("recovered")
    );
    assert_eq!(agent.task.id, id);
    assert_eq!(agent.llm_interaction_attempts, 2);
    let mut scheduled = 0;
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::RetryScheduled { delay_ms, .. } => {
                assert_eq!(delay_ms, 60_000);
                scheduled += 1;
            }
            AgentEvent::ToolCallInitiated { .. }
            | AgentEvent::ToolResult { .. }
            | AgentEvent::UsageReported(_) => {
                panic!("failed request must not execute tools or report usage")
            }
            _ => {}
        }
    }
    assert_eq!(scheduled, 1);
}

#[tokio::test(start_paused = true)]
async fn stream_capacity_error_retries_without_requesting_configuration_changes() {
    let (mut agent, mut events, _commands) = harness();
    let started = tokio::time::Instant::now();
    agent
        .run_llm_attempt::<()>(async {
            Err(ai::Error::ProviderError {
                message: "Our servers are currently overloaded. Please try again later.".into(),
                status: None,
                code: Some("server_is_overloaded".into()),
                retry_after: None,
            })
        })
        .await
        .unwrap();
    // Retried by the default rate-limit retry: 1 s without a server delay.
    assert_eq!(started.elapsed(), Duration::from_secs(1));
    assert_eq!(
        agent
            .run_llm_attempt(async { Ok("recovered") })
            .await
            .unwrap(),
        Some("recovered")
    );
    let mut scheduled = 0;
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::RetryScheduled { .. } => scheduled += 1,
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } => {
                panic!("temporary capacity errors must use bounded automatic recovery")
            }
            _ => {}
        }
    }
    assert_eq!(scheduled, 1);
}

#[tokio::test(start_paused = true)]
async fn temporary_failures_exhaust_at_ten_attempts_and_can_be_suspended() {
    let (mut agent, mut events, commands) = harness();
    let started = tokio::time::Instant::now();
    for _ in 0..10 {
        assert!(
            agent
                .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
                .await
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(started.elapsed(), Duration::from_millis(127_750));
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
    let mut retries = 0;
    let mut pauses = 0;
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::RetryScheduled { .. } => retries += 1,
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } => pauses += 1,
            _ => {}
        }
    }
    assert_eq!((retries, pauses), (9, 1));
    commands.send(UserCommand::SuspendTask).await.unwrap();
    let task = timeout(Duration::from_secs(1), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
}

#[tokio::test(start_paused = true)]
async fn permanent_failures_pause_immediately_and_inform_policy_returns_to_host() {
    for error in [
        ai::Error::MissingApiKey("test".into()),
        provider_failure(400, None),
        provider_failure(401, None),
    ] {
        let (mut agent, mut events, _commands) = harness();
        let started = tokio::time::Instant::now();
        agent
            .run_llm_attempt::<()>(async { Err(error) })
            .await
            .unwrap();
        assert_eq!(started.elapsed(), Duration::ZERO);
        assert!(matches!(
            agent.task.current_state,
            TaskState::AwaitingErrorRecoveryApproval { .. }
        ));
        while let Ok(event) = events.try_recv() {
            assert!(!matches!(event, AgentEvent::RetryScheduled { .. }));
        }
    }
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenInformAgent { max_attempts: 1 };
    // Run the real runtime with missing provider configuration: it must
    // return the preserved task to the host instead of spinning Interrupted.
    let task = timeout(Duration::from_secs(1), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
}

#[tokio::test(start_paused = true)]
async fn excessive_server_delay_stops_without_retrying_early() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.max_elapsed_ms = 30_000;
    // Ordinary recovery (the default rate-limit retry would wait the 60 s).
    agent.task.agent_config.llm_retry_config.rate_limit_retry = None;
    let started = tokio::time::Instant::now();
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(429, Some(Duration::from_secs(60)))) })
        .await
        .unwrap();
    assert_eq!(started.elapsed(), Duration::ZERO);
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
}

#[tokio::test(start_paused = true)]
async fn elapsed_budget_interrupts_a_hung_retry_and_includes_request_time() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.max_elapsed_ms = 1000;
    let started = tokio::time::Instant::now();
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
        .await
        .unwrap();
    agent
        .run_llm_attempt(std::future::pending::<Result<(), ai::Error>>())
        .await
        .unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(1));
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
}

#[tokio::test(start_paused = true)]
async fn cancel_and_suspend_interrupt_backoff_and_pending_requests() {
    for suspend in [false, true] {
        for in_backoff in [false, true] {
            let (mut agent, _events, commands) = harness();
            let started = tokio::time::Instant::now();
            let sender = tokio::spawn(async move {
                tokio_sleep(Duration::from_millis(10)).await;
                commands
                    .send(if suspend {
                        UserCommand::SuspendTask
                    } else {
                        UserCommand::CancelTask
                    })
                    .await
                    .unwrap();
            });
            if in_backoff {
                agent
                    .run_llm_attempt::<()>(async {
                        Err(provider_failure(503, Some(Duration::from_secs(60))))
                    })
                    .await
                    .unwrap();
            } else {
                agent
                    .run_llm_attempt(std::future::pending::<Result<(), ai::Error>>())
                    .await
                    .unwrap();
            }
            sender.await.unwrap();
            assert_eq!(started.elapsed(), Duration::from_millis(10));
            assert!(agent.stop_requested);
            assert_eq!(
                matches!(agent.task.current_state, TaskState::Cancelled { .. }),
                !suspend
            );
        }
    }
}

#[tokio::test(start_paused = true)]
async fn peer_context_does_not_reset_recovery_limits() {
    let (mut agent, _events, commands) = harness();
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 2 };
    commands
        .send(UserCommand::InjectContext {
            content: "peer update".into(),
        })
        .await
        .unwrap();
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
        .await
        .unwrap();
    let started = agent.recovery_started;
    agent.flush_context_injections().unwrap();
    assert_eq!(agent.recovery_started, started);
    assert_eq!(agent.llm_interaction_attempts, 1);
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
        .await
        .unwrap();
    assert!(matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
}

#[tokio::test(start_paused = true)]
async fn expired_oauth_without_refresh_credentials_uses_recovery_instead_of_losing_task() {
    let (mut agent, mut events, commands) = harness();
    // Synthetic non-JWT access token is treated as expired. No refresh token
    // means no HTTP request can be made by this test.
    agent.llm_client = Arc::new(ai::LLMClient::new().with_openai_chatgpt_oauth(
        "synthetic-expired-token",
        None,
        "test-account",
    ));
    let runner = tokio::spawn(agent.run());
    loop {
        if let AgentEvent::ErrorEncountered {
            requires_approval_to_retry,
            error_message,
            ..
        } = events.recv().await.unwrap()
        {
            assert!(requires_approval_to_retry);
            assert!(error_message.contains("No refresh_token"));
            break;
        }
    }
    commands.send(UserCommand::SuspendTask).await.unwrap();
    let task = timeout(Duration::from_secs(1), runner)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
    assert!(task.usage_history.is_empty());
}

#[test]
fn retry_configuration_defaults_round_trip_and_rejects_invalid_limits() {
    let (agent, _events, _commands) = harness();
    let mut json = serde_json::to_value(&agent.task.agent_config).unwrap();
    json.as_object_mut().unwrap().remove("llm_retry_config");
    let config: AgentConfig = serde_json::from_value(json).unwrap();
    assert_eq!(config.llm_retry_config, LLMRetryConfig::default());
    let invalid = LLMRetryConfig {
        max_elapsed_ms: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
    let invalid = LLMRetryConfig {
        initial_delay_ms: 1000,
        max_delay_ms: 250,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
    let parsed: LLMRetryConfig = serde_json::from_str("{\"max_elapsed_ms\":60000}").unwrap();
    assert_eq!(parsed.initial_delay_ms, 250);
    assert_eq!(parsed.max_elapsed_ms, 60_000);
    assert!(parsed.capacity_retry.is_none());
    let parsed: LLMRetryConfig = serde_json::from_str(r#"{"capacity_retry":{}}"#).unwrap();
    assert_eq!(
        parsed.capacity_retry,
        Some(LLMCapacityRetryConfig::default())
    );
    assert_eq!(
        serde_json::from_value::<LLMRetryConfig>(serde_json::to_value(&parsed).unwrap()).unwrap(),
        parsed
    );
    for (initial_delay_ms, max_delay_ms) in [(0, 300_000), (300_000, 30_000)] {
        let invalid = LLMRetryConfig {
            capacity_retry: Some(LLMCapacityRetryConfig {
                initial_delay_ms,
                max_delay_ms,
            }),
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}

#[tokio::test(start_paused = true)]
async fn refresh_failure_consumes_the_shared_budget_before_model_request() {
    let (mut agent, _events, _commands) = harness();
    let model_calls = std::sync::atomic::AtomicUsize::new(0);
    let started = tokio::time::Instant::now();
    // The production operation performs refresh before creating a stream;
    // inject a transient refresh error at that boundary, then a success.
    for refresh_succeeds in [false, true] {
        agent
            .run_llm_attempt(async {
                if !refresh_succeeds {
                    return Err(provider_failure(503, Some(Duration::from_secs(2))));
                }
                model_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            })
            .await
            .unwrap();
    }
    assert_eq!(started.elapsed(), Duration::from_secs(2));
    assert_eq!(model_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(agent.llm_interaction_attempts, 2);
}

pub(super) fn complete_response() -> ai::StreamEvent {
    let mut message = ai::Message::new_assistant("Finished after recovery.");
    message.tool_calls.push(ai::ToolCall {
        id: Some("complete-once".into()),
        name: "complete_task".into(),
        arguments: serde_json::json!({"summary":"done"}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    });
    ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new(
            "test-call".into(),
            "test".into(),
            "test".into(),
            0,
            0,
            1000,
        ),
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }))
}

#[tokio::test(start_paused = true)]
async fn full_runtime_recovers_and_executes_the_completed_tool_only_once() {
    use futures::StreamExt;
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Err(provider_failure(503, None)));
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
    let started = tokio::time::Instant::now();
    let task = timeout(Duration::from_secs(1), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(started.elapsed(), Duration::from_millis(250));
    assert_eq!(task.usage_history.len(), 1);
    let mut results = 0;
    let mut retries = 0;
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::ToolResult { tool_name, .. } => {
                assert_eq!(tool_name, "complete_task");
                results += 1;
            }
            AgentEvent::RetryScheduled { .. } => retries += 1,
            _ => {}
        }
    }
    assert_eq!((results, retries), (1, 1));
}

#[tokio::test(start_paused = true)]
async fn full_runtime_capacity_recovers_after_old_limits_and_completes_once() {
    use futures::StreamExt;
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.capacity_retry = Some(Default::default());
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    for i in 0..12 {
        let response = if i % 2 == 0 {
            Err(capacity_failure(None))
        } else {
            Ok(futures::stream::iter([Err(capacity_failure(None))]).boxed())
        };
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(response);
    }
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(complete_response())]).boxed()));
    let started = tokio::time::Instant::now();
    let task = timeout(Duration::from_secs(3600), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(started.elapsed() > Duration::from_secs(300));
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.usage_history.len(), 1);
    let (mut retries, mut tools) = (0, 0);
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::RetryScheduled { .. } => retries += 1,
            AgentEvent::ToolResult { tool_name, .. } => {
                assert_eq!(tool_name, "complete_task");
                tools += 1;
            }
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } => panic!("capacity exhausted"),
            _ => {}
        }
    }
    assert_eq!((retries, tools), (12, 1));
}

#[tokio::test(start_paused = true)]
async fn full_runtime_bounds_a_retry_stream_and_preserves_partial_text() {
    use futures::StreamExt;
    let (mut agent, mut events, commands) = harness();
    agent.task.agent_config.llm_retry_config.max_elapsed_ms = 1000;
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Err(provider_failure(503, None)));
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(ai::StreamEvent::TextChunk(
            "partial work".into(),
        ))])
        .chain(futures::stream::pending())
        .boxed()));
    let started = tokio::time::Instant::now();
    let runner = tokio::spawn(agent.run());
    timeout(Duration::from_secs(2), async {
        loop {
            match events.recv().await.unwrap() {
                AgentEvent::ErrorEncountered {
                    requires_approval_to_retry: true,
                    error_message,
                    ..
                } => {
                    assert!(error_message.contains("budget exhausted during stream"));
                    break;
                }
                AgentEvent::ToolResult { .. } | AgentEvent::UsageReported(_) => {
                    panic!("incomplete stream must not execute tools or invent usage")
                }
                _ => {}
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(1));
    commands.send(UserCommand::SuspendTask).await.unwrap();
    let task = timeout(Duration::from_secs(1), runner)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(task.conversation.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|content| content.as_text() == Some("partial work"))
    }));
}

#[tokio::test(start_paused = true)]
async fn classifier_refusals_are_not_kept_or_retried() {
    use futures::StreamExt;
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    let messages_before = agent.task.conversation.messages.len();
    let mut refusal = ai::Refusal::new(ai::RefusalSource::SafetyClassifier);
    refusal.partial_output = Some("partial".into());
    let refused = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message: ai::Message::new_assistant(""),
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new(
            "refused".into(),
            "test".into(),
            "test".into(),
            0,
            0,
            1000,
        ),
        stop_reason: ai::StopReason::Refusal,
        refusal: Some(refusal),
        openai_turn_state: None,
    }));
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(refused)]).boxed()));
    let task = timeout(Duration::from_secs(1), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(&task.current_state, TaskState::Interrupted { reason } if reason.contains("refused")),
        "{:?}",
        task.current_state
    );
    assert!(
        !task.conversation.messages[messages_before..]
            .iter()
            .any(|message| message.role == ai::Role::Assistant),
        "the refused turn must not be added to history"
    );
    assert_eq!(task.usage_history.len(), 1, "refused calls are still accounted for");
    let mut reported = 0;
    while let Ok(event) = events.try_recv() {
        assert!(!matches!(event, AgentEvent::RetryScheduled { .. }));
        if let AgentEvent::RefusalReceived(refusal) = event {
            assert_eq!(refusal.source, ai::RefusalSource::SafetyClassifier);
            reported += 1;
        }
    }
    assert_eq!(reported, 1);
}

/// Runs one scripted stream to the end under a policy that returns to the host.
pub(super) async fn run_refused_turn(
    mode: ToolCallingMode,
    stream: Vec<ai::StreamEvent>,
) -> (Task, usize, Vec<AgentEvent>) {
    use futures::StreamExt;
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    agent.task.agent_config.tool_config.tool_calling_mode = mode;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let messages_before = agent.task.conversation.messages.len();
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter(stream.into_iter().map(Ok)).boxed()));
    let task = timeout(Duration::from_secs(1), agent.run())
        .await
        .unwrap()
        .unwrap();
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    (task, messages_before, received)
}

pub(super) fn model_refusal() -> ai::Refusal {
    let mut refusal = ai::Refusal::new(ai::RefusalSource::Model);
    refusal.explanation = Some("I can't help with that.".into());
    refusal
}

pub(super) fn refused_response(message: ai::Message, refusal: ai::Refusal) -> ai::StreamEvent {
    ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new(
            "refused".into(),
            "test".into(),
            "test".into(),
            0,
            0,
            1000,
        ),
        stop_reason: ai::StopReason::Refusal,
        refusal: Some(refusal),
        openai_turn_state: None,
    }))
}

/// A model's own refusal is kept in history, reported once as a refusal and
/// handled as a non-retryable error rather than a missing tool call.
fn assert_model_refusal_reported(task: &Task, messages_before: usize, events: &[AgentEvent]) {
    assert!(
        matches!(&task.current_state, TaskState::Interrupted { reason } if reason.contains("I can't help with that.")),
        "{:?}",
        task.current_state
    );
    let added: Vec<_> = task.conversation.messages[messages_before..]
        .iter()
        .filter(|message| message.role == ai::Role::Assistant)
        .collect();
    assert_eq!(added.len(), 1, "the refused turn is kept once");
    let text: String = added[0].content.iter().filter_map(ai::Content::as_text).collect();
    assert_eq!(text, "I can't help with that.");
    assert!(added[0].tool_calls.is_empty(), "calls of a refused turn are not kept");
    assert_eq!(task.usage_history.len(), 1);
    let refusals: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RefusalReceived(refusal) => Some(refusal),
            _ => None,
        })
        .collect();
    assert_eq!(refusals, [&model_refusal()]);
    for event in events {
        match event {
            AgentEvent::RetryScheduled { .. } => panic!("a refusal must not be retried"),
            AgentEvent::ToolResult { .. } => panic!("a refused turn's tools must not run"),
            AgentEvent::ErrorEncountered { error_message, .. } => {
                assert!(!error_message.contains("tool call"), "{error_message}");
                assert!(error_message.contains("refused"), "{error_message}");
            }
            _ => {}
        }
    }
}

#[tokio::test(start_paused = true)]
async fn native_mode_model_refusals_are_kept_reported_and_not_retried() {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(ai::ToolCall {
        id: Some("refused-call".into()),
        name: "complete_task".into(),
        arguments: serde_json::json!({"summary":"done"}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    });
    let (task, messages_before, events) = run_refused_turn(
        ToolCallingMode::Native,
        vec![
            ai::StreamEvent::Refusal(model_refusal()),
            refused_response(message, model_refusal()),
        ],
    )
    .await;
    assert_model_refusal_reported(&task, messages_before, &events);
}

/// An OpenAI response keeps its output items for replay. A refused turn the
/// agent keeps without its tool calls must replay those items without the
/// dropped call (a call with no result is rejected by the API), and the
/// turn keeps the response's routing token.
#[tokio::test(start_paused = true)]
async fn a_kept_openai_refusal_replays_its_items_without_the_dropped_calls() {
    use serde_json::json;
    let reasoning = json!({"type":"reasoning", "id":"rs_1", "summary":[], "encrypted_content":"enc_1"});
    let refusal_message = json!({"type":"message", "id":"msg_1", "role":"assistant",
        "content":[{"type":"refusal", "refusal":"I can't help with that."}]});
    let call = json!({"type":"function_call", "id":"fc_1", "call_id":"refused-call",
        "name":"complete_task", "arguments":"{\"summary\":\"done\"}"});
    let mut message = ai::Message::new_assistant("");
    message.id = Some("msg_1".into());
    message.reasoning_artifact = Some(ai::ReasoningArtifact::OpenAIOutputItems {
        items: vec![reasoning.clone(), refusal_message.clone(), call],
    });
    message.tool_calls.push(ai::ToolCall {
        id: Some("refused-call".into()),
        name: "complete_task".into(),
        arguments: json!({"summary":"done"}),
        thought_signature: None,
        provider_id: Some("fc_1".into()),
        invalid_arguments: None,
    });
    let ai::StreamEvent::End(mut response) = refused_response(message, model_refusal()) else {
        unreachable!()
    };
    response.openai_turn_state = Some("turn-1".into());
    let (task, messages_before, events) = run_refused_turn(
        ToolCallingMode::Native,
        vec![ai::StreamEvent::Refusal(model_refusal()), ai::StreamEvent::End(response)],
    )
    .await;
    assert_model_refusal_reported(&task, messages_before, &events);
    assert_eq!(task.conversation.current_openai_turn_state(), Some("turn-1"));

    // The trace records the next request before it is sent; nothing listens
    // on the port, so the call itself fails. Real sockets run on the real
    // clock: a paused one would jump to the next timer while they wait.
    tokio::time::resume();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let dir = std::env::temp_dir().join(format!("arko-agents-refusal-replay-{}", Uuid::new_v4()));
    let client = ai::LLMClient::new()
        .with_request_trace(ai::RequestTrace::new(&dir))
        .with_openai_client(
            ai::providers::openai::OpenAIClient::new("test")
                .with_base_url(format!("http://127.0.0.1:{port}")),
        );
    assert!(client.generate_text(&task.conversation, Default::default()).await.is_err());
    let request_file = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap()
        .join("0001-request.json");
    let request: Value = serde_json::from_str(&std::fs::read_to_string(request_file).unwrap()).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let input = request["body"]["input"].as_array().unwrap();
    assert!(!input.iter().any(|item| item["type"] == "function_call"), "{input:?}");
    let replayed: Vec<&Value> = input
        .iter()
        .filter(|item| item["type"] == "reasoning" || item["role"] == "assistant")
        .collect();
    assert_eq!(replayed, [&reasoning, &refusal_message]);
}

#[tokio::test(start_paused = true)]
async fn text_mode_model_refusals_are_kept_reported_and_not_retried() {
    let (task, messages_before, events) = run_refused_turn(
        ToolCallingMode::OutputBased,
        vec![
            ai::StreamEvent::Refusal(model_refusal()),
            refused_response(ai::Message::new_assistant(""), model_refusal()),
        ],
    )
    .await;
    assert_model_refusal_reported(&task, messages_before, &events);
}

#[tokio::test(start_paused = true)]
async fn a_streamed_model_refusal_without_a_final_response_is_still_a_refusal() {
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let (task, messages_before, events) =
            run_refused_turn(mode, vec![ai::StreamEvent::Refusal(model_refusal())]).await;
        assert!(
            matches!(&task.current_state, TaskState::Interrupted { reason } if reason.contains("refused")),
            "{:?}",
            task.current_state
        );
        let kept: Vec<String> = task.conversation.messages[messages_before..]
            .iter()
            .filter(|message| message.role == ai::Role::Assistant)
            .map(|message| message.content.iter().filter_map(ai::Content::as_text).collect())
            .collect();
        assert_eq!(kept, ["I can't help with that."]);
        assert!(events.iter().any(|event| matches!(event, AgentEvent::RefusalReceived(_))));
        assert!(!events.iter().any(|event| matches!(event, AgentEvent::RetryScheduled { .. })));
    }
}

#[tokio::test(start_paused = true)]
async fn tool_calls_with_invalid_json_arguments_are_rejected_not_run() {
    use futures::StreamExt;
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(ai::ToolCall {
        id: Some("broken".into()),
        name: "complete_task".into(),
        arguments: serde_json::Value::Null,
        thought_signature: None,
        provider_id: None,
        invalid_arguments: Some("{\"summary\":".into()),
    });
    let broken = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new(
            "broken".into(),
            "test".into(),
            "test".into(),
            0,
            0,
            1000,
        ),
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }));
    for response in [broken, complete_response()] {
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(futures::stream::iter([Ok(response)]).boxed()));
    }
    let task = timeout(Duration::from_secs(1), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let (mut rejected, mut results) = (0, 0);
    while let Ok(event) = events.try_recv() {
        match event {
            AgentEvent::ToolRejected { reason: Some(reason), .. } => {
                assert!(reason.contains("not valid JSON"), "{reason}");
                rejected += 1;
            }
            AgentEvent::ToolResult { .. } => results += 1,
            _ => {}
        }
    }
    assert_eq!((rejected, results), (1, 1));
    assert!(task.conversation.messages.iter().any(|message| {
        message.content.iter().any(|content| {
            content
                .as_tool_result()
                .is_some_and(|result| result.is_error && result.call_id.as_deref() == Some("broken"))
        })
    }));
}

/// A refusal delivered as an error (Gemini generation-blocked codes), before
/// or after the stream starts, is reported like other refusals; text
/// streamed before it is not kept.
#[tokio::test(start_paused = true)]
async fn refusals_reported_as_errors_are_reported_and_drop_partial_text() {
    use futures::StreamExt;
    let blocked = || {
        let mut refusal = ai::Refusal::new(ai::RefusalSource::ContentFilter);
        refusal.category = Some("recitation".into());
        refusal.partial_output = Some("Once upon".into());
        ai::Error::Refusal(Box::new(refusal))
    };
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        for mid_stream in [true, false] {
            let (mut agent, mut events, _commands) = harness();
            agent.task.agent_config.llm_error_policy =
                LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
            agent.task.agent_config.tool_config.tool_calling_mode = mode.clone();
            let messages_before = agent.task.conversation.messages.len();
            let response = if mid_stream {
                Ok(futures::stream::iter([
                    Ok(ai::StreamEvent::TextChunk("Once upon".into())),
                    Err(blocked()),
                ])
                .boxed())
            } else {
                Err(blocked())
            };
            agent.scripted_responses.get_mut().unwrap().push_back(response);
            let task = timeout(Duration::from_secs(1), agent.run())
                .await
                .unwrap()
                .unwrap();
            assert!(
                matches!(&task.current_state, TaskState::Interrupted { reason } if reason.contains("refused")),
                "{:?}",
                task.current_state
            );
            assert!(
                !task.conversation.messages[messages_before..]
                    .iter()
                    .any(|message| message.role == ai::Role::Assistant),
                "the refused output must not be added to history"
            );
            let mut reported = 0;
            while let Ok(event) = events.try_recv() {
                assert!(!matches!(event, AgentEvent::RetryScheduled { .. }));
                if let AgentEvent::RefusalReceived(refusal) = event {
                    assert_eq!(refusal.category.as_deref(), Some("recitation"));
                    reported += 1;
                }
            }
            assert_eq!(reported, 1, "{mode:?} mid_stream={mid_stream}");
        }
    }
}
