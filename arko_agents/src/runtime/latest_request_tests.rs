//! The task's latest request (`Task::latest_request`) in arko's own runtime:
//! recorded with each answered request's send time and usage, in the same
//! step as its usage, and none after a handover until the fresh
//! conversation's first request is answered.

use super::*;
use crate::handover::HandoverConfig;
use crate::snapshot::{TaskSnapshotSink, TaskSnapshots};
use crate::task::RequestRecord;
use async_trait::async_trait;
use futures::StreamExt;

/// A response with `message`, whose request had `usage`.
fn answered(message: ai::Message, usage: ai::UsageInfo) -> ScriptedStreamResult {
    let stop_reason = if message.has_tool_calls() {
        ai::StopReason::ToolUse
    } else {
        ai::StopReason::EndTurn
    };
    let input = usage.input_tokens;
    let response = ai::GenerationResponse {
        message,
        usage,
        model_call_usage: ai::ModelCallUsage::new(
            "call".into(),
            "test".into(),
            "test".into(),
            input,
            5,
            1_000_000,
        ),
        stop_reason,
        refusal: None,
        openai_turn_state: None,
    };
    Ok(futures::stream::iter([Ok(ai::StreamEvent::End(Box::new(response)))]).boxed())
}

/// A turn calling `name` (with its name as its id), with `input` tokens of
/// input.
fn calling(name: &str, arguments: Value, input: usize) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    let mut call = super::image_tests::call(name, name);
    call.arguments = arguments;
    message.tool_calls.push(call);
    answered(message, ai::UsageInfo::new(input, 5))
}

fn completing(input: usize) -> ScriptedStreamResult {
    calling(
        COMPLETE_TASK_TOOL_NAME,
        serde_json::json!({"summary": "done"}),
        input,
    )
}

/// Keeps every snapshot it is given.
#[derive(Default)]
struct Snapshots(std::sync::Mutex<Vec<Task>>);

#[async_trait]
impl TaskSnapshotSink for Snapshots {
    async fn task_changed(&self, task: &Task, _durable: bool) -> Result<(), String> {
        self.0.lock().unwrap().push(task.clone());
        Ok(())
    }
}

fn native(agent: &mut AgentLoop) {
    let tools = &mut agent.task.agent_config.tool_config;
    tools.tool_calling_mode = ToolCallingMode::Native;
    tools.default_auto_approve_tool_use = true;
}

fn recorded(agent: &mut AgentLoop) -> Arc<Snapshots> {
    let snapshots = Arc::new(Snapshots::default());
    agent.snapshots = TaskSnapshots::new(Some(snapshots.clone() as Arc<dyn TaskSnapshotSink>));
    snapshots
}

/// The input of `record`'s request, if it has one.
fn input(record: &RequestRecord) -> Option<usize> {
    record.latest().map(|request| request.input_tokens)
}

/// The answered request is the latest, from when its own attempt was sent:
/// a failed attempt before it (retried after 250 ms) isn't, and the
/// request's usage gives its input and cache counts.
#[tokio::test]
async fn the_answered_request_is_the_latest_from_its_own_attempt() {
    let (mut agent, _events, _commands) = super::retry_tests::harness();
    native(&mut agent);
    let usage = ai::UsageInfo::new(52_000, 5)
        .with_cached_tokens(50_000)
        .with_cache_write_tokens(2_000);
    let mut message = ai::Message::new_assistant("");
    message.tool_calls.push(super::tool_call_tests::native_call(
        "done",
        COMPLETE_TASK_TOOL_NAME,
        serde_json::json!({"summary": "done"}),
    ));
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(Err(super::retry_tests::provider_failure(503, None)));
    scripted.push_back(answered(message, usage));
    let started = chrono::Utc::now();
    let task = timeout(Duration::from_secs(30), agent.run())
        .await
        .unwrap()
        .unwrap();
    let ended = chrono::Utc::now();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let request = task.latest_request.latest().expect("a request").clone();
    assert_eq!(
        (
            request.input_tokens,
            request.cached_tokens,
            request.cache_write_tokens
        ),
        (52_000, 50_000, 2_000)
    );
    assert!(
        request.sent_at - started >= chrono::Duration::milliseconds(250),
        "{} after the start",
        request.sent_at - started
    );
    assert!(request.sent_at <= ended);
    assert_eq!(request.cache_lifetime(), None, "the configuration sets it");
}

/// A request to the refusal fallback's model has another cache, so the
/// refused request to the configured model stays the latest.
#[tokio::test(start_paused = true)]
async fn a_fallback_request_leaves_the_latest_request() {
    let (mut agent, _events, _commands) = super::retry_tests::harness();
    native(&mut agent);
    agent.task.agent_config.refusal_fallback = Some(crate::agent_config::AgentLLMConfig {
        model: ai::Model::openai_gpt_6_sol(),
        generation_config: Default::default(),
    });
    let mut refused = super::retry_tests::refused_response(
        ai::Message::new_assistant(""),
        super::retry_tests::model_refusal(),
    );
    if let ai::StreamEvent::End(response) = &mut refused {
        response.usage = ai::UsageInfo::new(10, 0);
    }
    let scripted = agent.scripted_responses.get_mut().unwrap();
    scripted.push_back(Ok(futures::stream::iter([Ok(refused)]).boxed()));
    scripted.push_back(completing(20));
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.fallback_calls, [1]);
    assert_eq!(input(&task.latest_request), Some(10));
}

/// Every snapshot, any of which a crash could leave, holds the request
/// whose usage it last recorded, so the record never runs ahead of the
/// conversation or lags behind it.
#[tokio::test]
async fn every_snapshot_holds_the_request_of_its_latest_usage() {
    let (mut agent, mut events, _commands) = super::image_tests::harness(ToolCallingMode::Native);
    let snapshots = recorded(&mut agent);
    agent.scripted_responses.get_mut().unwrap().extend([
        calling("plain_text", serde_json::json!({}), 1_000),
        calling("plain_text", serde_json::json!({}), 1_500),
        completing(2_000),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    super::tool_call_tests::drain(&mut events);
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let snapshots = snapshots.0.lock().unwrap().clone();
    let mut seen = Vec::new();
    for snapshot in &snapshots {
        let size = snapshot.current_conversation_context_size();
        assert_eq!(input(&snapshot.latest_request), size);
        if seen.last() != Some(&size) {
            seen.push(size);
        }
    }
    assert_eq!(seen, [None, Some(1_000), Some(1_500), Some(2_000)]);
    assert_eq!(input(&task.latest_request), Some(2_000));
}

/// A handover replaces the conversation, so the task has no request until
/// the fresh conversation's first one is answered. A task saved in between
/// (stopped, say) has none.
#[tokio::test]
async fn a_handover_leaves_no_request_until_the_fresh_conversation_s_first() {
    let (mut agent, mut events, _commands) = super::image_tests::harness(ToolCallingMode::Native);
    agent.task.agent_config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    let snapshots = recorded(&mut agent);
    agent.scripted_responses.get_mut().unwrap().extend([
        // Under the threshold, then over it (a conversation that starts
        // over it isn't handed over).
        calling("plain_text", serde_json::json!({}), 500),
        calling("plain_text", serde_json::json!({}), 1_500),
        calling(
            HANDOVER_REPORT_TOOL_NAME,
            serde_json::json!({"report": "Read it. Next: finish."}),
            1_600,
        ),
        calling("plain_text", serde_json::json!({}), 200),
        completing(250),
    ]);
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    super::tool_call_tests::drain(&mut events);
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    let snapshots = snapshots.0.lock().unwrap().clone();
    let handed_over = snapshots
        .iter()
        .find(|snapshot| snapshot.handovers.len() == 1)
        .expect("a snapshot after the handover");
    assert_eq!(handed_over.latest_request, RequestRecord::NoRequest);
    assert_eq!(input(&task.latest_request), Some(250));
}

/// A crash leaves the task as its last save had it. A snapshot taken after
/// the first request, with every later save lost, restores with that
/// request as its latest, sent when the request was announced, and with
/// the conversation that request answered. Run again, it goes on and its
/// next request takes over.
#[tokio::test]
async fn a_task_restored_from_an_earlier_save_keeps_that_save_s_request() {
    let (mut agent, mut events, _commands) = super::image_tests::harness(ToolCallingMode::Native);
    let snapshots = recorded(&mut agent);
    agent.scripted_responses.get_mut().unwrap().extend([
        calling("plain_text", serde_json::json!({}), 1_000),
        calling("plain_text", serde_json::json!({}), 1_500),
        completing(2_000),
    ]);
    timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    let sent: Vec<_> = super::tool_call_tests::drain(&mut events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ModelRequestStarted { sent_at, .. } => sent_at,
            _ => None,
        })
        .collect();
    assert_eq!(sent.len(), 3);
    // The last save that has the first request as its latest, as a crash
    // after it would leave it on disk.
    let saved = snapshots
        .0
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|snapshot| input(&snapshot.latest_request) == Some(1_000))
        .map(|snapshot| serde_json::to_string(snapshot).unwrap())
        .unwrap();
    let restored: Task = serde_json::from_str(&saved).unwrap();
    let request = restored.latest_request.latest().unwrap();
    assert_eq!(
        request.sent_at.timestamp_millis(),
        sent[0].timestamp_millis()
    );
    assert_eq!(restored.current_conversation_context_size(), Some(1_000));
    assert_eq!(restored.usage_history.len(), 1);
    let answered = restored
        .conversation
        .messages
        .iter()
        .filter(|message| message.role == ai::Role::Assistant)
        .count();
    assert_eq!(answered, 1, "the conversation of the first request");

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(super::image_tests::TextTool));
    let (mut again, mut events, _commands) =
        super::tool_call_tests::agent_for(restored, registry, Arc::new(ai::LLMClient::new()));
    again.scripted_responses.get_mut().unwrap().extend([
        calling("plain_text", serde_json::json!({}), 1_600),
        completing(2_100),
    ]);
    let task = timeout(Duration::from_secs(5), again.run())
        .await
        .unwrap()
        .unwrap();
    super::tool_call_tests::drain(&mut events);
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let last = task.latest_request.latest().unwrap();
    assert_eq!(last.input_tokens, 2_100);
    assert!(last.sent_at > sent[0]);
}
