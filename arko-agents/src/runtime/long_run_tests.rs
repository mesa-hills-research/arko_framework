//! Long-running runs: rate limits retried after the provider's delay by
//! default, plan usage limits (pause, or opt-in wait until reset), and plan
//! usage reports with an opt-in threshold.

use super::retry_tests::{complete_response, harness, provider_failure};
use super::*;
use crate::agent_config::{
    LLMRateLimitRetryConfig, LLMRetryConfig, MonitoringConfig, PlanUsageThreshold,
};
use futures::StreamExt;

fn events_of(events: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    received
}

fn retry_delays(events: &[AgentEvent]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RetryScheduled { delay_ms, .. } => Some(*delay_ms),
            _ => None,
        })
        .collect()
}

fn usage_limit(reset_in: Option<Duration>) -> ai::Error {
    ai::Error::ProviderError {
        message: "The usage limit has been reached".into(),
        status: Some(429),
        code: Some("usage_limit_reached".into()),
        retry_after: reset_in,
    }
}

fn paused(agent: &AgentLoop) -> bool {
    matches!(
        agent.task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    )
}

#[tokio::test(start_paused = true)]
async fn rate_limits_follow_the_server_delay_by_default_outside_the_recovery_limits() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.max_elapsed_ms = 30_000;
    agent.task.agent_config.llm_error_policy =
        LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 3 };
    let messages = agent.task.conversation.messages.len();
    let started = tokio::time::Instant::now();
    for _ in 0..12 {
        agent
            .run_llm_attempt::<()>(async { Err(provider_failure(429, Some(Duration::from_secs(36)))) })
            .await
            .unwrap();
        assert!(matches!(agent.task.current_state, TaskState::InProgress));
    }
    // Twelve waits of the server's 36 s, past both the 3-attempt limit and
    // the 30 s elapsed budget, and nothing added to the conversation.
    assert_eq!(started.elapsed(), Duration::from_secs(12 * 36));
    assert_eq!(agent.task.conversation.messages.len(), messages);
    // Ordinary recovery is untouched: the next ordinary failure is its first.
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(503, None)) })
        .await
        .unwrap();
    assert_eq!(agent.ordinary_attempts(), 1);
    let events = events_of(&mut events);
    let delays = retry_delays(&events);
    assert_eq!(delays.len(), 13);
    assert!(delays[..12].iter().all(|delay| *delay == 36_000), "{delays:?}");
    assert_eq!(delays[12], 250);
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { requires_approval_to_retry: true, .. }
    )));
}

#[tokio::test(start_paused = true)]
async fn rate_limit_waits_back_off_without_a_hint_and_stay_cancellable() {
    let (mut agent, mut events, commands) = harness();
    for _ in 0..3 {
        agent
            .run_llm_attempt::<()>(async { Err(provider_failure(429, None)) })
            .await
            .unwrap();
    }
    assert_eq!(retry_delays(&events_of(&mut events)), [1_000, 2_000, 4_000]);
    let sender = tokio::spawn(async move {
        tokio_sleep(Duration::from_secs(1)).await;
        commands.send(UserCommand::CancelTask).await.unwrap();
    });
    let started = tokio::time::Instant::now();
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(429, Some(Duration::from_secs(600)))) })
        .await
        .unwrap();
    sender.await.unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(1));
    assert!(matches!(agent.task.current_state, TaskState::Failed { .. }));
}

#[tokio::test(start_paused = true)]
async fn usage_limits_pause_by_default_and_report_the_reset() {
    for error in [
        usage_limit(Some(Duration::from_secs(3 * 3600))),
        // A rate limit resetting hours away counts as a usage limit.
        provider_failure(429, Some(Duration::from_secs(2 * 3600))),
    ] {
        let (mut agent, mut events, _commands) = harness();
        let before = unix_millis_now();
        let started = tokio::time::Instant::now();
        agent.run_llm_attempt::<()>(async { Err(error) }).await.unwrap();
        assert_eq!(started.elapsed(), Duration::ZERO);
        assert!(paused(&agent));
        let events = events_of(&mut events);
        let reached: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::UsageLimitReached {
                    resets_at_unix_ms,
                    waiting_until_unix_ms,
                    ..
                } => Some((*resets_at_unix_ms, *waiting_until_unix_ms)),
                _ => None,
            })
            .collect();
        assert_eq!(reached.len(), 1);
        let (resets_at, waiting_until) = reached[0];
        assert!(resets_at.unwrap() >= before + 2 * 3600 * 1000);
        assert_eq!(waiting_until, None);
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ErrorEncountered { requires_approval_to_retry: true, error_message, .. }
                if error_message.contains("Usage limit reached")
        )));
        assert!(retry_delays(&events).is_empty());
    }
}

#[tokio::test(start_paused = true)]
async fn usage_limits_can_wait_until_the_reset() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.usage_limit_policy =
        UsageLimitPolicy::WaitUntilReset {
            max_wait_ms: 6 * 3600 * 1000,
        };
    let messages = agent.task.conversation.messages.len();
    let before = unix_millis_now();
    let started = tokio::time::Instant::now();
    agent
        .run_llm_attempt::<()>(async { Err(usage_limit(Some(Duration::from_secs(3 * 3600)))) })
        .await
        .unwrap();
    // The reset plus a one-second margin (no jitter in this harness).
    assert_eq!(started.elapsed(), Duration::from_secs(3 * 3600 + 1));
    assert!(matches!(agent.task.current_state, TaskState::InProgress));
    assert_eq!(agent.task.conversation.messages.len(), messages);
    let events = events_of(&mut events);
    let waiting_until = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::UsageLimitReached {
                waiting_until_unix_ms,
                ..
            } => *waiting_until_unix_ms,
            _ => None,
        })
        .unwrap();
    assert!(waiting_until >= before + (3 * 3600 + 1) * 1000);
    assert_eq!(agent.ordinary_attempts(), 1, "the wait is not an ordinary attempt");

    // A reset further off than max_wait, or none reported, pauses.
    for reset_in in [Some(Duration::from_secs(7 * 3600)), None] {
        let (mut agent, _events, _commands) = harness();
        agent.task.agent_config.llm_retry_config.usage_limit_policy =
            UsageLimitPolicy::WaitUntilReset {
                max_wait_ms: 6 * 3600 * 1000,
            };
        agent
            .run_llm_attempt::<()>(async { Err(usage_limit(reset_in)) })
            .await
            .unwrap();
        assert!(paused(&agent));
    }
}

#[tokio::test(start_paused = true)]
async fn a_usage_limit_wait_can_be_suspended() {
    let (mut agent, _events, commands) = harness();
    agent.task.agent_config.llm_retry_config.usage_limit_policy =
        UsageLimitPolicy::WaitUntilReset {
            max_wait_ms: 6 * 3600 * 1000,
        };
    let sender = tokio::spawn(async move {
        tokio_sleep(Duration::from_secs(60)).await;
        commands.send(UserCommand::SuspendTask).await.unwrap();
    });
    let started = tokio::time::Instant::now();
    agent
        .run_llm_attempt::<()>(async { Err(usage_limit(Some(Duration::from_secs(3600)))) })
        .await
        .unwrap();
    sender.await.unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(60));
    assert!(agent.stop_requested);
}

/// A native turn calling a tool the registry doesn't have, so the task
/// continues, reporting `used_percent` of the primary window.
fn turn_with_plan_usage(used_percent: f64) -> ai::StreamEvent {
    turn_with_plan_window(used_percent, Some(4_102_444_800))
}

fn turn_with_plan_window(used_percent: f64, resets_at: Option<u64>) -> ai::StreamEvent {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(ai::ToolCall {
        id: Some("call-missing".into()),
        name: "missing_tool".into(),
        arguments: serde_json::json!({}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    });
    let usage = ai::PlanUsage {
        plan_type: Some("pro".into()),
        primary: Some(ai::PlanUsageWindow {
            used_percent,
            window_minutes: Some(300),
            resets_at,
            reset_after_seconds: None,
        }),
        ..Default::default()
    };
    ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new("plan".into(), "openai".into(), "m".into(), 0, 0, 1000)
            .with_plan_usage(Some(usage)),
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }))
}

fn script(agent: &mut AgentLoop, turns: Vec<ai::StreamEvent>) {
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent.task.agent_config.tool_config.default_auto_approve_tool_use = true;
    for turn in turns {
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(futures::stream::iter([Ok(turn)]).boxed()));
    }
}

#[tokio::test(start_paused = true)]
async fn plan_usage_is_reported_and_nothing_acts_on_it_by_default() {
    let (mut agent, mut events, _commands) = harness();
    script(&mut agent, vec![turn_with_plan_usage(99.0), complete_response()]);
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let events = events_of(&mut events);
    let reported: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::PlanUsageReported(usage) => Some(usage.primary.as_ref().unwrap().used_percent),
            _ => None,
        })
        .collect();
    assert_eq!(reported, [99.0]);
    assert!(!events.iter().any(|event| matches!(event, AgentEvent::PlanUsageThresholdReached { .. })));
}

#[tokio::test(start_paused = true)]
async fn a_plan_usage_threshold_pauses_once_per_window() {
    let (mut agent, mut events, commands) = harness();
    agent.task.agent_config.monitoring = MonitoringConfig {
        plan_usage: Some(PlanUsageThreshold {
            percent: 90.0,
            action: PlanUsageAction::Pause,
        }),
        ..Default::default()
    };
    script(
        &mut agent,
        vec![turn_with_plan_usage(95.0), turn_with_plan_usage(96.0), complete_response()],
    );
    let runner = tokio::spawn(agent.run());
    let mut paused = 0;
    loop {
        match timeout(Duration::from_secs(60), events.recv()).await.unwrap().unwrap() {
            AgentEvent::PlanUsageThresholdReached {
                window,
                used_percent,
                threshold_percent,
                ..
            } => {
                assert_eq!((window.as_str(), used_percent, threshold_percent), ("primary", 95.0, 90.0));
            }
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { error_message }) => {
                assert!(error_message.contains("Plan usage reached 95.0%"), "{error_message}");
                paused += 1;
                commands
                    .send(UserCommand::ApproveErrorRecovery { message: None })
                    .await
                    .unwrap();
            }
            AgentEvent::TaskStreamEnd => break,
            _ => {}
        }
    }
    let task = runner.await.unwrap().unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(paused, 1, "the same window and reset pause once");
    // A monitor pause is not a model error: approving it tells the model
    // nothing about one.
    assert!(!task.conversation.messages.iter().any(|message| message
        .content
        .iter()
        .filter_map(ai::Content::as_text)
        .any(|text| text.contains("resulted in an error"))));
    assert!(!task.monitor_pause);
}

/// The reported window is saved with the task: a resumed task doesn't pause
/// for it again, and a report whose window has since reset is ignored.
#[tokio::test(start_paused = true)]
async fn a_plan_usage_threshold_pause_is_not_repeated_after_a_resume() {
    let threshold = MonitoringConfig {
        plan_usage: Some(PlanUsageThreshold {
            percent: 90.0,
            action: PlanUsageAction::Pause,
        }),
        ..Default::default()
    };
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.monitoring = threshold.clone();
    agent.task.agent_config.llm_error_policy = LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    script(&mut agent, vec![turn_with_plan_usage(95.0)]);
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert_eq!(task.plan_usage_threshold_reported.as_ref().unwrap().window, "primary");
    let task: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();

    let (mut agent, mut events, _commands) = harness();
    agent.task = task;
    script(&mut agent, vec![complete_response()]);
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let received = events_of(&mut events);
    assert!(!received.iter().any(|event| matches!(event, AgentEvent::PlanUsageThresholdReached { .. })));

    // A window that has already reset doesn't count.
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.monitoring = threshold;
    script(&mut agent, vec![turn_with_plan_window(99.0, Some(1_000)), complete_response()]);
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(!events_of(&mut events).iter().any(|event| matches!(event, AgentEvent::PlanUsageThresholdReached { .. })));
}

/// Rate limits without a delay are waited out with backoff, but not forever:
/// after `max_wait_ms` of such waits they are handled as a usage limit.
#[tokio::test(start_paused = true)]
async fn rate_limits_without_a_delay_hand_over_to_the_usage_limit_policy() {
    let (mut agent, mut events, _commands) = harness();
    let started = tokio::time::Instant::now();
    for _ in 0..100 {
        agent
            .run_llm_attempt::<()>(async { Err(provider_failure(429, None)) })
            .await
            .unwrap();
        if paused(&agent) {
            break;
        }
    }
    assert!(paused(&agent));
    let waited = started.elapsed();
    assert!(waited >= Duration::from_secs(15 * 60) && waited < Duration::from_secs(17 * 60), "{waited:?}");
    assert!(events_of(&mut events).iter().any(|event| matches!(
        event,
        AgentEvent::UsageLimitReached { resets_at_unix_ms: None, waiting_until_unix_ms: None, .. }
    )));
}

#[tokio::test(start_paused = true)]
async fn a_usage_limit_wait_must_fit_max_wait_with_its_margin_and_jitter() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.llm_retry_config.usage_limit_policy =
        UsageLimitPolicy::WaitUntilReset { max_wait_ms: 3600 * 1000 };
    // The reset fits, but the margin and jitter after it might not.
    agent
        .run_llm_attempt::<()>(async { Err(usage_limit(Some(Duration::from_secs(3600 - 10)))) })
        .await
        .unwrap();
    assert!(paused(&agent));
}

#[tokio::test(start_paused = true)]
async fn a_plan_usage_threshold_can_slow_requests_down_instead() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.monitoring = MonitoringConfig {
        plan_usage: Some(PlanUsageThreshold {
            percent: 90.0,
            action: PlanUsageAction::SlowDown { delay_ms: 10_000 },
        }),
        ..Default::default()
    };
    script(
        &mut agent,
        vec![turn_with_plan_usage(95.0), turn_with_plan_usage(96.0), complete_response()],
    );
    let started = tokio::time::Instant::now();
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // Two requests followed reports over the threshold.
    assert_eq!(started.elapsed(), Duration::from_secs(20));
    let reached = events_of(&mut events)
        .into_iter()
        .filter(|event| matches!(event, AgentEvent::PlanUsageThresholdReached { .. }))
        .count();
    assert_eq!(reached, 1);
}

#[test]
fn new_retry_and_monitoring_settings_default_on_load_and_round_trip() {
    // Configs saved before these settings existed.
    let old: LLMRetryConfig = serde_json::from_str(r#"{"max_elapsed_ms":60000}"#).unwrap();
    assert_eq!(old.rate_limit_retry, Some(LLMRateLimitRetryConfig::default()));
    assert_eq!(old.usage_limit_policy, UsageLimitPolicy::Pause);
    let (agent, _events, _commands) = harness();
    let mut saved = serde_json::to_value(&agent.task.agent_config).unwrap();
    assert!(saved.get("monitoring").is_none(), "nothing to save by default");
    saved["llm_retry_config"]
        .as_object_mut()
        .unwrap()
        .retain(|key, _| key == "max_elapsed_ms");
    let loaded: AgentConfig = serde_json::from_value(saved).unwrap();
    assert!(loaded.llm_retry_config.rate_limit_retry.is_some());
    assert!(loaded.monitoring.is_empty());

    // Opting out and in round-trips.
    let custom: LLMRetryConfig = serde_json::from_str(
        r#"{"rate_limit_retry":null,"usage_limit_policy":{"WaitUntilReset":{"max_wait_ms":3600000}}}"#,
    )
    .unwrap();
    assert_eq!(custom.rate_limit_retry, None);
    assert_eq!(
        custom.usage_limit_policy,
        UsageLimitPolicy::WaitUntilReset { max_wait_ms: 3_600_000 }
    );
    assert_eq!(
        serde_json::from_value::<LLMRetryConfig>(serde_json::to_value(&custom).unwrap()).unwrap(),
        custom
    );
    let monitoring: MonitoringConfig =
        serde_json::from_str(r#"{"plan_usage":{"percent":90}}"#).unwrap();
    assert_eq!(monitoring.plan_usage.as_ref().unwrap().action, PlanUsageAction::Pause);
    monitoring.validate().unwrap();
    for percent in [0.0, 100.5, f64::NAN] {
        let invalid = MonitoringConfig {
            plan_usage: Some(PlanUsageThreshold { percent, action: PlanUsageAction::Pause }),
            ..Default::default()
        };
        assert!(invalid.validate().is_err(), "{percent}");
    }
    let invalid = LLMRetryConfig {
        rate_limit_retry: Some(LLMRateLimitRetryConfig {
            initial_delay_ms: 0,
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

/// Rate limits with a delay count toward the same per-episode budget: short
/// delays that keep coming hand over to the usage-limit policy too.
#[tokio::test(start_paused = true)]
async fn repeated_short_rate_limit_delays_also_hand_over_to_the_usage_limit_policy() {
    let (mut agent, mut events, _commands) = harness();
    let started = tokio::time::Instant::now();
    for _ in 0..100 {
        agent
            .run_llm_attempt::<()>(async { Err(provider_failure(429, Some(Duration::from_secs(60)))) })
            .await
            .unwrap();
        if paused(&agent) {
            break;
        }
    }
    assert!(paused(&agent));
    // Fifteen minutes of 60-second waits, then the usage-limit policy.
    assert_eq!(started.elapsed(), Duration::from_secs(15 * 60));
    assert!(events_of(&mut events)
        .iter()
        .any(|event| matches!(event, AgentEvent::UsageLimitReached { .. })));
}

/// The budget counts the time actually waited: a wait cut short by a
/// command counts only up to then.
#[tokio::test(start_paused = true)]
async fn a_rate_limit_wait_cut_short_counts_the_time_waited() {
    let (mut agent, _events, commands) = harness();
    let sender = tokio::spawn(async move {
        tokio_sleep(Duration::from_secs(10)).await;
        commands.send(UserCommand::SuspendTask).await.unwrap();
    });
    agent
        .run_llm_attempt::<()>(async { Err(provider_failure(429, Some(Duration::from_secs(600)))) })
        .await
        .unwrap();
    sender.await.unwrap();
    assert!(agent.stop_requested);
    assert_eq!(agent.rate_limit_waited, Duration::from_secs(10));
}
