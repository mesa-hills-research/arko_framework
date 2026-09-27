//! Refusal recovery: the refused input is reported, the host changes it and
//! the task continues; the opt-in fallback model; announced calls and
//! output-based markup of refused turns.

use super::retry_tests::{
    complete_response, harness, model_refusal, provider_failure, refused_response,
    run_refused_turn,
};
use super::*;
use crate::agent_config::{AgentLLMConfig, EnvironmentDetailsConfig};
use crate::task::{RefusalResolution, RefusedInput, RefusedMessage, RefusedToolResult};
use futures::StreamExt;

fn native(agent: &mut AgentLoop) {
    agent.task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
    agent.task.agent_config.tool_config.default_auto_approve_tool_use = true;
}

fn push(agent: &mut AgentLoop, turn: ai::StreamEvent) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Ok(futures::stream::iter([Ok(turn)]).boxed()));
}

fn refused() -> ai::StreamEvent {
    refused_response(ai::Message::new_assistant(""), model_refusal())
}

/// A native turn calling a tool the registry doesn't have: its error result
/// is the next input.
fn missing_tool_turn() -> ai::StreamEvent {
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
        usage: ai::UsageInfo::new(0, 0),
        model_call_usage: ai::ModelCallUsage::new("t".into(), "openai".into(), "m".into(), 0, 0, 1000),
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }))
}

fn host(message_index: usize) -> RefusedMessage {
    RefusedMessage {
        message_index,
        runtime: false,
        tool_results: vec![],
        images: 0,
    }
}

/// Runs the agent, answering each refusal pause with the next list of
/// changes (a failed one leaves the task waiting for the next), and
/// returns the task and every event.
async fn run_resolving(
    agent: AgentLoop,
    mut events: mpsc::Receiver<AgentEvent>,
    commands: mpsc::Sender<UserCommand>,
    resolutions: Vec<Vec<RefusalResolution>>,
) -> (Task, Vec<AgentEvent>) {
    let runner = tokio::spawn(agent.run());
    let mut resolutions = resolutions.into_iter();
    let mut received = Vec::new();
    loop {
        let event = timeout(Duration::from_secs(60), events.recv())
            .await
            .unwrap()
            .unwrap();
        received.push(event.clone());
        match event {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
            | AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                ..
            } if received
                .iter()
                .any(|event| matches!(event, AgentEvent::RefusedInput(_))) =>
            {
                if let Some(changes) = resolutions.next() {
                    commands
                        .send(UserCommand::ResolveRefusal { changes })
                        .await
                        .unwrap();
                }
            }
            AgentEvent::TaskStreamEnd => break,
            _ => {}
        }
    }
    (runner.await.unwrap().unwrap(), received)
}

fn refused_inputs(events: &[AgentEvent]) -> Vec<RefusedInput> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RefusedInput(input) => Some(input.clone()),
            _ => None,
        })
        .collect()
}

fn texts(message: &ai::Message) -> String {
    message.content.iter().filter_map(ai::Content::as_text).collect()
}

fn events_of(events: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut received = Vec::new();
    while let Ok(event) = events.try_recv() {
        received.push(event);
    }
    received
}

#[tokio::test(start_paused = true)]
async fn a_refused_task_prompt_is_reported_and_can_be_rephrased() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(
        agent,
        events,
        commands,
        vec![vec![RefusalResolution::RephraseUserMessage {
            message_index: 0,
            text: "A rephrased task.".into(),
        }]],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }), "{:?}", task.current_state);
    assert_eq!(refused_inputs(&events), [RefusedInput { messages: vec![host(0)] }]);
    // The refused turn is gone and the rephrased prompt was answered.
    let messages = &task.conversation.messages;
    assert_eq!(texts(&messages[0]), "A rephrased task.");
    assert_eq!(task.initial_task_prompt, "A rephrased task.");
    assert_eq!(texts(&messages[1]), "Finished after recovery.");
    assert!(!messages.iter().any(|message| texts(message).contains("can't help")));
    assert!(task.refused_input.is_none());
    assert!(!events.iter().any(|event| matches!(event, AgentEvent::RetryScheduled { .. })));
}

/// The runtime's own messages (here a file-tree snapshot added after the
/// prompt) are part of the refused input but marked, and can't be changed.
#[tokio::test(start_paused = true)]
async fn runtime_messages_are_reported_apart_from_the_host_input() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    let workspace = std::env::temp_dir().join(format!("arko-refusal-tree-{}", Uuid::new_v4()));
    std::fs::create_dir(&workspace).unwrap();
    std::fs::write(workspace.join("notes.txt"), "notes").unwrap();
    agent.task.current_working_directory = workspace.clone();
    agent.task.agent_config.environment_details_config = EnvironmentDetailsConfig {
        file_tree: crate::agent_config::BuiltInEnvironmentDetail {
            include: true,
            ..Default::default()
        },
        ..Default::default()
    };
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(
        agent,
        events,
        commands,
        vec![
            vec![RefusalResolution::RemoveUserMessage { message_index: 1 }],
            vec![RefusalResolution::RephraseUserMessage {
                message_index: 0,
                text: "Another task.".into(),
            }],
        ],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let input = &refused_inputs(&events)[0];
    assert_eq!(input.messages.len(), 2);
    assert!(!input.messages[0].runtime);
    assert!(input.messages[1].runtime, "the file-tree snapshot");
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { error_message, .. } if error_message.contains("added by the runtime")
    )));
    assert!(texts(&task.conversation.messages[1]).contains("# File tree"));
    assert!(task.runtime_messages.contains(&1));
    std::fs::remove_dir_all(&workspace).ok();
}

#[tokio::test(start_paused = true)]
async fn refused_tool_results_can_be_redacted_and_an_invalid_resolution_keeps_waiting() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    push(&mut agent, missing_tool_turn());
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(
        agent,
        events,
        commands,
        vec![
            // Tool results can't be removed like a user message, and a
            // failed list of changes applies none of them.
            vec![
                RefusalResolution::RedactToolResults {
                    call_ids: vec!["call-missing".into()],
                    replacement: "[too early]".into(),
                },
                RefusalResolution::RemoveUserMessage { message_index: 2 },
            ],
            vec![RefusalResolution::RedactToolResults {
                call_ids: vec!["call-missing".into()],
                replacement: "[output removed]".into(),
            }],
        ],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }), "{:?}", task.current_state);
    assert_eq!(
        refused_inputs(&events),
        [RefusedInput {
            messages: vec![RefusedMessage {
                message_index: 2,
                runtime: false,
                tool_results: vec![RefusedToolResult {
                    call_id: Some("call-missing".into()),
                    tool_name: "missing_tool".into(),
                }],
                images: 0,
            }],
        }]
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ErrorEncountered { error_message, .. }
            if error_message.contains("Could not resolve the refusal") && error_message.contains("redact")
    )));
    let result = task.conversation.messages[2].content[0].as_tool_result().unwrap();
    assert_eq!(
        (result.call_id.as_deref(), result.output.as_str()),
        (Some("call-missing"), "[output removed]")
    );
    assert_eq!(texts(&task.conversation.messages[3]), "Finished after recovery.");
}

/// Under an informing error policy the refused task is returned to the
/// host, which resolves the refusal on the task and runs it again.
#[tokio::test(start_paused = true)]
async fn a_returned_task_can_be_resolved_and_run_again() {
    let (task, _, events) = run_refused_turn(ToolCallingMode::Native, vec![refused()]).await;
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert_eq!(refused_inputs(&events), [RefusedInput { messages: vec![host(0)] }]);
    assert_eq!(task.refused_input, Some(RefusedInput { messages: vec![host(0)] }));
    // Saved and loaded, the refused input survives.
    let mut task: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
    // The prompt is the only user message: it can be rephrased, not removed.
    assert!(task.resolve_refusal(vec![RefusalResolution::RemoveUserMessage { message_index: 0 }]).is_err());
    assert!(
        task.resolve_refusal(vec![RefusalResolution::RedactToolResults {
            call_ids: vec!["x".into()],
            replacement: String::new(),
        }])
        .is_err()
    );
    task.resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
        message_index: 0,
        text: "Try this instead.".into(),
    }])
    .unwrap();
    assert!(task.refused_input.is_none());
    assert!(
        task.resolve_refusal(vec![RefusalResolution::RemoveUserMessage { message_index: 0 }]).is_err(),
        "resolved once"
    );

    let (mut agent, _events, _commands) = harness();
    agent.task = task;
    native(&mut agent);
    push(&mut agent, complete_response());
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(texts(&task.conversation.messages[0]), "Try this instead.");

    // Tasks saved before refused inputs existed load without one.
    let mut saved = serde_json::to_value(&task).unwrap();
    for field in ["refused_input", "runtime_messages", "fallback_calls", "plan_usage_threshold_reported", "monitor_pause"] {
        saved.as_object_mut().unwrap().remove(field);
    }
    let loaded: Task = serde_json::from_value(saved).unwrap();
    assert!(loaded.refused_input.is_none() && loaded.runtime_messages.is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_refused_user_message_after_earlier_turns_can_be_removed() {
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    agent.task.conversation.system_message = Some("resumed".into());
    agent.task.conversation.add_user_message("first task");
    agent.task.conversation.add_assistant_message("done");
    agent.task.conversation.add_user_message("objectionable request");
    agent.task.user_message_count = 2;
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let (task, events) = run_resolving(
        agent,
        events,
        commands,
        vec![vec![RefusalResolution::RemoveUserMessage { message_index: 2 }]],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(refused_inputs(&events), [RefusedInput { messages: vec![host(2)] }]);
    let texts: Vec<String> = task.conversation.messages.iter().map(texts).collect();
    assert!(!texts.iter().any(|text| text.contains("objectionable")), "{texts:?}");
    // The conversation ended on an assistant turn, so the model was asked
    // to continue.
    assert!(texts[2].contains("Continue the task"), "{texts:?}");
    assert!(task.runtime_messages.contains(&2));
}

/// Tools that track what the model saw start over after the input changes.
#[tokio::test(start_paused = true)]
async fn tools_observe_the_changed_history_from_the_start() {
    #[derive(Debug, Default)]
    struct Observer {
        resets: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl crate::tool::Tool for Observer {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "observer".into(),
                description: "Observes".into(),
                parameters: vec![],
                output_based_example: None,
                supports_batched_calls: false,
                tool_type_details: crate::tool::ToolTypeDetails::DeveloperDefined,
                raw_json_schema_override: None,
            }
        }
        fn reset_context(&self, _: Uuid) {
            self.resets.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        async fn execute(
            &self,
            _: &TaskContext<'_>,
            _: HashMap<String, Value>,
        ) -> Result<String, crate::tool::ToolError> {
            Ok(String::new())
        }
    }
    let (mut agent, events, commands) = harness();
    native(&mut agent);
    let observer = Arc::new(Observer::default());
    let mut registry = ToolRegistry::new();
    registry.register(observer.clone());
    agent.tool_registry = Arc::new(registry);
    push(&mut agent, refused());
    push(&mut agent, complete_response());
    let resets_before = observer.resets.load(std::sync::atomic::Ordering::SeqCst);
    let (task, _) = run_resolving(
        agent,
        events,
        commands,
        vec![vec![RefusalResolution::RephraseUserMessage {
            message_index: 0,
            text: "Changed.".into(),
        }]],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(observer.resets.load(std::sync::atomic::Ordering::SeqCst) > resets_before);
}

fn fallback_config(model: ai::Model) -> Option<AgentLLMConfig> {
    Some(AgentLLMConfig {
        model,
        generation_config: Default::default(),
    })
}

#[tokio::test(start_paused = true)]
async fn an_opt_in_fallback_model_answers_a_refused_turn_once() {
    let (mut agent, mut events, _commands) = harness();
    native(&mut agent);
    agent.task.agent_config.refusal_fallback = fallback_config(ai::Model::openai_gpt_6_sol());
    let requested = Arc::clone(&agent.requested_models);
    push(&mut agent, refused());
    push(&mut agent, missing_tool_turn());
    push(&mut agent, complete_response());
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    // The refused turn went to the fallback once; later turns use the
    // configured model again.
    assert_eq!(*requested.lock().unwrap(), ["gpt-5.6-luna", "gpt-6-sol", "gpt-5.6-luna"]);
    assert!(!task.conversation.messages.iter().any(|message| texts(message).contains("can't help")));
    assert!(task.refused_input.is_none(), "the fallback's turn answered the input");
    assert_eq!(task.fallback_calls, [1]);
    let received = events_of(&mut events);
    let order: Vec<&str> = received
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RefusalReceived(_) => Some("refusal"),
            AgentEvent::RefusedInput(_) => Some("input"),
            AgentEvent::RefusalFallback { model_id } => {
                assert_eq!(model_id, "gpt-6-sol");
                Some("fallback")
            }
            _ => None,
        })
        .collect();
    assert_eq!(order, ["refusal", "input", "fallback"]);
}

#[tokio::test(start_paused = true)]
async fn a_refusal_from_the_fallback_model_stops_the_task() {
    let (mut agent, mut events, _commands) = harness();
    native(&mut agent);
    agent.task.agent_config.llm_error_policy = LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
    agent.task.agent_config.refusal_fallback = fallback_config(ai::Model::openai_gpt_6_sol());
    let requested = Arc::clone(&agent.requested_models);
    push(&mut agent, refused());
    push(&mut agent, refused());
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    assert_eq!(*requested.lock().unwrap(), ["gpt-5.6-luna", "gpt-6-sol"]);
    let received = events_of(&mut events);
    assert_eq!(refused_inputs(&received).len(), 2);
    assert_eq!(task.refused_input, Some(RefusedInput { messages: vec![host(0)] }));
}

/// A fallback request that fails leaves the refusal standing and
/// resolvable; approving recovery resends to the configured model.
#[tokio::test(start_paused = true)]
async fn a_failed_fallback_request_leaves_the_refusal_resolvable() {
    let (mut agent, mut events, commands) = harness();
    native(&mut agent);
    agent.task.agent_config.refusal_fallback = fallback_config(ai::Model::openai_gpt_6_sol());
    let requested = Arc::clone(&agent.requested_models);
    push(&mut agent, refused());
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(Err(provider_failure(401, None)));
    push(&mut agent, complete_response());
    let runner = tokio::spawn(agent.run());
    loop {
        match timeout(Duration::from_secs(60), events.recv()).await.unwrap().unwrap() {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { error_message }) => {
                assert!(error_message.contains("refused"), "{error_message}");
                assert!(error_message.contains("fallback model could not answer"), "{error_message}");
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
    assert_eq!(*requested.lock().unwrap(), ["gpt-5.6-luna", "gpt-6-sol", "gpt-5.6-luna"]);
}

#[test]
fn a_fallback_on_another_provider_is_rejected() {
    let (agent, _events, _commands) = harness();
    let mut config = agent.task.agent_config.clone();
    config.refusal_fallback = fallback_config(ai::Model::anthropic_claude_opus_5_5());
    let error = config.validate().unwrap_err();
    assert!(error.to_string().contains("refusal_fallback"), "{error}");
    let built = AgentConfig::builder(
        "x".into(),
        "x".into(),
        AgentLLMConfig {
            model: ai::Model::openai_gpt_5_6_luna(),
            generation_config: Default::default(),
        },
    )
    .role_description("r".into())
    .instructions("i".into())
    .refusal_fallback(fallback_config(ai::Model::anthropic_claude_opus_5_5()).unwrap())
    .build();
    assert!(built.is_err());
}

#[tokio::test(start_paused = true)]
async fn the_runtime_refuses_to_start_with_a_cross_provider_fallback() {
    let (mut agent, _events, _commands) = harness();
    agent.task.agent_config.refusal_fallback = fallback_config(ai::Model::anthropic_claude_opus_5_5());
    assert!(agent.run().await.is_err());
}

/// Low (a): calls a native stream announced before the refusal get
/// `ToolRejected("turn refused")`.
#[tokio::test(start_paused = true)]
async fn calls_announced_before_a_native_refusal_are_rejected() {
    let (task, _, events) = run_refused_turn(
        ToolCallingMode::Native,
        vec![
            ai::StreamEvent::ToolCallStart {
                index: 0,
                name: "complete_task".into(),
            },
            ai::StreamEvent::Refusal(model_refusal()),
            refused(),
        ],
    )
    .await;
    assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
    let initiated: Vec<Uuid> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallInitiated { tool_call_id, .. } => Some(*tool_call_id),
            _ => None,
        })
        .collect();
    let rejected: Vec<(Uuid, Option<String>)> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolRejected {
                tool_call_id,
                reason,
                ..
            } => Some((*tool_call_id, reason.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(initiated.len(), 1);
    assert_eq!(rejected, [(initiated[0], Some("turn refused".to_string()))]);
}

/// Calls announced while streaming that the final response doesn't hold
/// (for example cut off at the token limit) get an outcome too.
#[tokio::test(start_paused = true)]
async fn announced_calls_missing_from_a_completed_response_are_rejected() {
    let (mut agent, mut events, _commands) = harness();
    native(&mut agent);
    agent.scripted_responses.get_mut().unwrap().push_back(Ok(futures::stream::iter([
        Ok(ai::StreamEvent::ToolCallStart { index: 0, name: "complete_task".into() }),
        Ok(ai::StreamEvent::ToolCallStart { index: 1, name: "read_file".into() }),
        Ok(complete_response()),
    ])
    .boxed()));
    let task = timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let received = events_of(&mut events);
    let rejected: Vec<String> = received
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolRejected { tool_name, reason: Some(reason), .. } if reason == UNDELIVERED_CALL => {
                Some(tool_name.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(rejected, ["read_file"]);
}

/// Low (b): an output-based refused turn keeps only the refusal
/// explanation, or without one its text minus the tool-call markup that
/// was never run; OpenAI message items holding that markup are not replayed.
#[tokio::test(start_paused = true)]
async fn output_based_refusals_keep_no_tool_call_markup() {
    let markup = "Sure. <complete_task><summary>done</summary></complete_task>";
    for explanation in [Some("I can't help with that."), None] {
        let mut refusal = model_refusal();
        refusal.explanation = explanation.map(String::from);
        let mut message = ai::Message::new_assistant(markup);
        message.reasoning_artifact = Some(ai::ReasoningArtifact::OpenAIOutputItems {
            items: vec![
                serde_json::json!({"type":"reasoning", "id":"rs_1", "summary":[], "encrypted_content":"enc"}),
                serde_json::json!({"type":"message", "id":"msg_1", "role":"assistant",
                    "content":[{"type":"output_text", "text": markup}]}),
            ],
        });
        let (task, messages_before, events) = run_refused_turn(
            ToolCallingMode::OutputBased,
            vec![
                ai::StreamEvent::TextChunk(markup.into()),
                refused_response(message, refusal),
            ],
        )
        .await;
        assert!(matches!(task.current_state, TaskState::Interrupted { .. }));
        let kept: Vec<&ai::Message> = task.conversation.messages[messages_before..]
            .iter()
            .filter(|message| message.role == ai::Role::Assistant)
            .collect();
        assert_eq!(kept.len(), 1);
        let expected = explanation.unwrap_or("Sure. ");
        assert_eq!(texts(kept[0]), expected);
        let Some(ai::ReasoningArtifact::OpenAIOutputItems { items }) = &kept[0].reasoning_artifact else {
            panic!("the reasoning item is kept");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "reasoning");
        assert!(!events.iter().any(|event| matches!(event, AgentEvent::ToolResult { .. })));
    }
}

// ---------------------------------------------------------------------------
// The fallback through the real OpenAI request builders, against a local
// server speaking the Responses streaming protocol.

type Captured = Arc<std::sync::Mutex<Vec<(String, Value)>>>;

/// Serves one streamed response per request, in order, recording each
/// request's headers and JSON body.
async fn responses_server(replies: Vec<(String, Option<&'static str>)>) -> (String, Captured) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let captured: Captured = Arc::default();
    let record = Arc::clone(&captured);
    tokio::spawn(async move {
        for (body, turn_state) in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (headers, request) = loop {
                let mut chunk = [0; 8192];
                let n = socket.read(&mut chunk).await.unwrap();
                bytes.extend_from_slice(&chunk[..n]);
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
                    let length: usize = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|value| value.trim().parse().unwrap())
                        })
                        .unwrap();
                    if bytes.len() >= end + 4 + length {
                        let request: Value =
                            serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                        break (headers.to_ascii_lowercase(), request);
                    }
                }
            };
            record.lock().unwrap().push((headers, request));
            let extra = turn_state
                .map(|token| format!("x-codex-turn-state: {token}\r\n"))
                .unwrap_or_default();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (url, captured)
}

/// A streamed Responses turn delivering `items`.
fn stream_of(model: &str, items: Vec<Value>) -> String {
    let mut events: Vec<Value> = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| serde_json::json!({"type": "response.output_item.done", "output_index": index, "item": item}))
        .collect();
    events.push(serde_json::json!({"type": "response.completed", "response": {
        "id": "resp_test", "object": "response", "created_at": 0, "status": "completed", "model": model,
        "output": [], "usage": {"input_tokens": 10, "output_tokens": 5}, "metadata": {},
    }}));
    events
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

fn call_item(id: &str, name: &str, arguments: &str) -> Value {
    serde_json::json!({"type": "function_call", "id": format!("fc_{id}"), "call_id": format!("call_{id}"),
        "name": name, "arguments": arguments, "status": "completed"})
}

/// B1: over ChatGPT OAuth, a same-provider fallback request doesn't echo the
/// configured model's turn-state token, the configured model's later
/// request keeps its own token, and the fallback's reasoning items are
/// replayed only between models of one Codex compaction family.
#[tokio::test]
async fn a_fallback_turn_replays_correctly_through_the_openai_request_builders() {
    for (configured, keeps_reasoning) in [
        (ai::Model::openai_gpt_5_6_luna(), true),
        (ai::Model::openai_gpt_5_5(), false),
    ] {
        let refusal = serde_json::json!({"type": "message", "id": "msg_refused", "role": "assistant",
            "status": "completed", "content": [{"type": "refusal", "refusal": "I can't help with that."}]});
        let reasoning = serde_json::json!({"type": "reasoning", "id": "rs_fallback", "summary": [],
            "encrypted_content": "enc_fallback"});
        let (url, captured) = responses_server(vec![
            (stream_of(&configured.id, vec![refusal]), Some("turn-configured")),
            (
                stream_of("gpt-6-sol", vec![reasoning, call_item("1", "missing_tool", "{}")]),
                Some("turn-fallback"),
            ),
            (
                stream_of(&configured.id, vec![call_item("2", "complete_task", "{\"summary\":\"done\"}")]),
                None,
            ),
        ])
        .await;
        // A JWT-shaped access token expiring in 2100 (`{"exp":4102444800}`).
        let claims = "eyJleHAiOjQxMDI0NDQ4MDB9";
        let client = ai::LLMClient::new().with_openai_client(
            ai::providers::openai::OpenAIClient::new_chatgpt_oauth(
                &format!("header.{claims}.signature"),
                None,
                "test-account",
            )
            .with_base_url(url),
        );
        let (mut agent, _events, _commands) = harness();
        native(&mut agent);
        agent.llm_client = Arc::new(client);
        agent.task.agent_config.llm_config.model = configured.clone();
        agent.task.conversation.model = configured.clone();
        agent.task.agent_config.refusal_fallback = fallback_config(ai::Model::openai_gpt_6_sol());
        let task = timeout(Duration::from_secs(30), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }), "{:?}", task.current_state);

        let requests = captured.lock().unwrap().clone();
        assert_eq!(requests.len(), 3);
        let models: Vec<&str> = requests.iter().map(|(_, body)| body["model"].as_str().unwrap()).collect();
        // (ChatGPT requests name gpt-5.5 without its snapshot date.)
        assert!(configured.id.starts_with(models[0]), "{models:?}");
        assert_eq!(models[1..], ["gpt-6-sol", models[0]]);
        assert!(!requests[1].0.contains("x-codex-turn-state"), "{}", requests[1].0);
        assert!(requests[2].0.contains("x-codex-turn-state: turn-configured"), "{}", requests[2].0);
        // The last request replays the fallback's call and its result, and
        // its reasoning only within the compaction family.
        let input = requests[2].1["input"].as_array().unwrap();
        assert!(input.iter().any(|item| item["type"] == "function_call" && item["call_id"] == "call_1"));
        assert!(input.iter().any(|item| item["type"] == "function_call_output" && item["call_id"] == "call_1"));
        let replayed_reasoning = input
            .iter()
            .any(|item| item["type"] == "reasoning" && item["encrypted_content"] == "enc_fallback");
        assert_eq!(replayed_reasoning, keeps_reasoning, "{}", configured.id);
        assert!(!input.iter().any(|item| item["id"] == "msg_refused"), "the refused turn is not kept");
    }
}
