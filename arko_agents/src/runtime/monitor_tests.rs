//! Model-reroute events and the opt-in prompt-cache monitor.

use super::retry_tests::{complete_response, harness};
use super::*;
use crate::agent_config::{CacheMonitorConfig, MonitoringConfig};
use futures::StreamExt;

/// A native turn calling a tool the registry doesn't have, so the task
/// continues, with the given usage.
fn turn(usage: ai::UsageInfo, call: ai::ModelCallUsage) -> ai::StreamEvent {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(ai::ToolCall {
        id: Some("call-missing".into()),
        name: "missing_tool".into(),
        arguments: serde_json::json!({}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    });
    ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage,
        model_call_usage: call,
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }))
}

fn call(model: &str) -> ai::ModelCallUsage {
    ai::ModelCallUsage::new("c".into(), "openai".into(), model.into(), 0, 0, 1000)
}

fn cached(input: usize, cached: usize) -> ai::UsageInfo {
    ai::UsageInfo::new(input, 10).with_cached_tokens(cached)
}

fn script(agent: &mut AgentLoop, turns: Vec<ai::StreamEvent>) {
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent
        .task
        .agent_config
        .tool_config
        .default_auto_approve_tool_use = true;
    for turn in turns {
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(futures::stream::iter([Ok(turn)]).boxed()));
    }
}

fn events_of(events: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    received
}

#[tokio::test(start_paused = true)]
async fn calls_served_by_another_model_are_reported_as_reroutes() {
    let (mut agent, mut events, _commands) = harness();
    let mut routed = call("anthropic/claude-sonnet-4.5");
    routed.provider = "openrouter".into();
    routed.served_model = Some("anthropic/claude-sonnet-4.5".into());
    routed.requested_model = Some("openrouter/auto".into());
    let rerouted = call("gpt-5.6-luna").with_served_model(Some("gpt-5.2".into()), true);
    script(
        &mut agent,
        vec![
            turn(ai::UsageInfo::new(1, 1), routed),
            turn(ai::UsageInfo::new(1, 1), rerouted),
            turn(ai::UsageInfo::new(1, 1), call("gpt-5.6-luna")),
            complete_response(),
        ],
    );
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let reroutes: Vec<(String, String, String)> = events_of(&mut events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRerouted {
                provider,
                requested_model,
                served_model,
            } => Some((provider, requested_model, served_model)),
            _ => None,
        })
        .collect();
    assert_eq!(
        reroutes,
        [
            (
                "openrouter".into(),
                "openrouter/auto".into(),
                "anthropic/claude-sonnet-4.5".into()
            ),
            // Without a requested model on the usage, the task's model.
            ("openai".into(), "gpt-5.6-luna".into(), "gpt-5.2".into()),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn the_cache_monitor_is_off_by_default() {
    let (mut agent, mut events, _commands) = harness();
    script(
        &mut agent,
        vec![
            turn(cached(1000, 0), call("m")),
            turn(cached(2000, 0), call("m")),
            turn(cached(3000, 0), call("m")),
            complete_response(),
        ],
    );
    timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(!events_of(&mut events).iter().any(|event| matches!(
        event,
        AgentEvent::CacheEfficiencyReported { .. } | AgentEvent::CacheEfficiencyLow { .. }
    )));
}

#[tokio::test(start_paused = true)]
async fn the_cache_monitor_reports_warns_and_pauses_once_after_warm_up() {
    let (mut agent, mut events, commands) = harness();
    agent.task.agent_config.monitoring = MonitoringConfig {
        cache: Some(CacheMonitorConfig {
            warmup_calls: 1,
            window_calls: 2,
            warn_below: Some(0.8),
            pause_below: Some(0.5),
        }),
        ..Default::default()
    };
    script(
        &mut agent,
        vec![
            // Warm-up: not judged.
            turn(cached(1000, 0), call("m")),
            // 900 of 1000 possible: fine.
            turn(cached(2000, 900), call("m")),
            // 200 of 2000: the two-call window is 1100 / 3000.
            turn(cached(3000, 200), call("m")),
            // Still low: no second warning or pause.
            turn(cached(4000, 300), call("m")),
            complete_response(),
        ],
    );
    let runner = tokio::spawn(agent.run());
    let mut received = Vec::new();
    loop {
        let event = timeout(Duration::from_secs(60), events.recv())
            .await
            .unwrap()
            .unwrap();
        if let AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
            error_message,
            cause,
            ..
        }) = &event
        {
            assert!(
                error_message.contains("Prompt cache efficiency fell to 36.7%"),
                "{error_message}"
            );
            assert_eq!(*cause, crate::task::ErrorRecoveryCause::Monitor);
            commands
                .send(UserCommand::ApproveErrorRecovery { message: None })
                .await
                .unwrap();
        }
        let end = matches!(event, AgentEvent::TaskStreamEnd);
        received.push(event);
        if end {
            break;
        }
    }
    let task = runner.await.unwrap().unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let reported: Vec<(f64, usize)> = received
        .iter()
        .filter_map(|event| match event {
            AgentEvent::CacheEfficiencyReported {
                rolling_efficiency,
                calls,
                ..
            } => Some((*rolling_efficiency, *calls)),
            _ => None,
        })
        .collect();
    assert_eq!(reported.len(), 3, "{reported:?}");
    assert_eq!(reported[0], (0.9, 1));
    let low: Vec<bool> = received
        .iter()
        .filter_map(|event| match event {
            AgentEvent::CacheEfficiencyLow { pausing, .. } => Some(*pausing),
            _ => None,
        })
        .collect();
    assert_eq!(low, [false, true]);
    let pauses = received
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            )
        })
        .count();
    assert_eq!(pauses, 1);
    // Not a model error: approving adds no error note to the conversation.
    assert!(!task.conversation.messages.iter().any(|message| {
        message
            .content
            .iter()
            .filter_map(ai::Content::as_text)
            .any(|text| text.contains("resulted in an error"))
    }));
}

#[test]
fn cache_monitor_settings_load_with_defaults_and_are_validated() {
    let config: MonitoringConfig = serde_json::from_str(r#"{"cache":{}}"#).unwrap();
    assert_eq!(config.cache, Some(CacheMonitorConfig::default()));
    config.validate().unwrap();
    for invalid in [
        CacheMonitorConfig {
            window_calls: 0,
            ..Default::default()
        },
        CacheMonitorConfig {
            warn_below: Some(1.5),
            ..Default::default()
        },
        CacheMonitorConfig {
            pause_below: Some(-0.1),
            ..Default::default()
        },
    ] {
        let config = MonitoringConfig {
            cache: Some(invalid),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}

/// Calls to the refusal fallback model use another prompt cache: they are
/// not judged, and the next configured call is compared with the configured
/// model's previous call.
#[tokio::test(start_paused = true)]
async fn the_cache_monitor_skips_fallback_calls() {
    let (mut agent, mut events, _commands) = harness();
    agent.task.agent_config.monitoring = MonitoringConfig {
        cache: Some(CacheMonitorConfig {
            warmup_calls: 1,
            window_calls: 1,
            warn_below: Some(0.5),
            pause_below: None,
        }),
        ..Default::default()
    };
    agent.task.agent_config.refusal_fallback = Some(crate::agent_config::AgentLLMConfig {
        model: ai::Model::openai_gpt_6_sol(),
        generation_config: Default::default(),
    });
    let ai::StreamEvent::End(mut refused) = super::retry_tests::refused_response(
        ai::Message::new_assistant(""),
        super::retry_tests::model_refusal(),
    ) else {
        unreachable!()
    };
    refused.usage = cached(2000, 900);
    script(
        &mut agent,
        vec![
            turn(cached(1000, 0), call("m")),
            ai::StreamEvent::End(refused),
            // The fallback's call reads nothing from the configured cache.
            turn(cached(50_000, 0), call("gpt-6-sol")),
            turn(cached(3000, 1800), call("m")),
            complete_response(),
        ],
    );
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.fallback_calls, [2]);
    let received = events_of(&mut events);
    let calls: Vec<f64> = received
        .iter()
        .filter_map(|event| match event {
            AgentEvent::CacheEfficiencyReported {
                call_efficiency, ..
            } => Some(*call_efficiency),
            _ => None,
        })
        .collect();
    assert_eq!(calls, [0.9, 0.9]);
    assert!(
        !received
            .iter()
            .any(|event| matches!(event, AgentEvent::CacheEfficiencyLow { .. }))
    );
}
