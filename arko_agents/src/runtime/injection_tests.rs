//! Context sent with `InjectContext` is never lost once the runtime has
//! taken it from the command channel: it goes into the conversation where a
//! user message can go, or is saved with the task
//! (`Task::pending_context_injections`) and added at the first safe turn
//! boundary after the task runs again, in the order sent, once. A native
//! call always gets its result before anything else follows it.

use super::long_run_tests::turn_with_plan_usage;
use super::retry_tests::{
    complete_response, harness, model_refusal, provider_failure, refused_response,
};
use super::tool_call_tests::{
    agent, agent_for, assert_calls_paired, complete, config, native_call, result_for, turn,
    wait_for, workspace,
};
use super::*;
use crate::agent_config::{AgentLLMConfig, MonitoringConfig, PlanUsageThreshold};
use crate::task::RefusalResolution;
use crate::tool::{Tool, ToolTypeDetails};
use futures::StreamExt;

const UPDATE: &str = "# Workflow context update\n\n";

/// The conversation's context updates: where each is, and the context it
/// carries (without the sections the runtime appends to a user message).
fn updates(conversation: &ai::Conversation) -> Vec<(usize, String)> {
    conversation
        .messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == ai::Role::User)
        .filter_map(|(index, message)| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_text)
                .find_map(|text| text.strip_prefix(UPDATE))
                .map(|update| (index, update.split("\n\n# ").next().unwrap().to_string()))
        })
        .collect()
}

/// The context of each update, in order.
fn update_texts(conversation: &ai::Conversation) -> Vec<String> {
    updates(conversation)
        .into_iter()
        .map(|(_, text)| text)
        .collect()
}

fn text(message: &ai::Message) -> String {
    message
        .content
        .iter()
        .filter_map(ai::Content::as_text)
        .collect()
}

fn last_assistant(conversation: &ai::Conversation) -> usize {
    conversation
        .messages
        .iter()
        .rposition(|message| message.role == ai::Role::Assistant)
        .unwrap()
}

/// Every native turn's calls are answered by the message after it, or the
/// turn is the conversation's last message and still waits for its
/// results: nothing comes between a call and its result.
fn assert_nothing_between_calls_and_results(conversation: &ai::Conversation) {
    let mut conversation = conversation.clone();
    if conversation
        .messages
        .last()
        .is_some_and(|message| message.role == ai::Role::Assistant && message.has_tool_calls())
    {
        conversation.messages.pop();
    }
    assert_calls_paired(&conversation);
}

fn inject(commands: &mpsc::Sender<UserCommand>, content: &str) {
    commands
        .try_send(UserCommand::InjectContext {
            content: content.into(),
        })
        .unwrap();
}

fn native(agent: &mut AgentLoop) {
    let tool_config = &mut agent.task.agent_config.tool_config;
    tool_config.tool_calling_mode = ToolCallingMode::Native;
    tool_config.default_auto_approve_tool_use = true;
}

fn push(agent: &mut AgentLoop, turn: ScriptedStreamResult) {
    agent.scripted_responses.get_mut().unwrap().push_back(turn);
}

fn scripted(event: ai::StreamEvent) -> ScriptedStreamResult {
    Ok(futures::stream::iter([Ok(event)]).boxed())
}

async fn finish(agent: AgentLoop) -> Task {
    timeout(Duration::from_secs(60), agent.run())
        .await
        .unwrap()
        .unwrap()
}

async fn joined(run: tokio::task::JoinHandle<Result<Task, ArkoAgentsError>>) -> Task {
    timeout(Duration::from_secs(60), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
}

/// The task as saved and loaded again, in a new runtime loop.
fn resumed(
    task: Task,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
    agent_for(task, ToolRegistry::new(), Arc::new(ai::LLMClient::new()))
}

/// Every event until the run ends.
async fn rest(events: &mut mpsc::Receiver<AgentEvent>, seen: &mut Vec<AgentEvent>) {
    while let Some(event) = timeout(Duration::from_secs(60), events.recv())
        .await
        .unwrap()
    {
        seen.push(event);
    }
}

fn states(events: &[AgentEvent]) -> Vec<TaskState> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TaskStateChanged(state) => Some(state.clone()),
            _ => None,
        })
        .collect()
}

/// L1: a task suspended at a turn boundary while a refusal fallback is
/// pending keeps the context the fallback holds back, saved with it. Run
/// again, the fallback answers the refused input alone; the saved context
/// follows, once, before context sent after the resume.
#[tokio::test(start_paused = true)]
async fn context_held_for_a_pending_fallback_survives_a_suspension() {
    let (mut agent, _events, commands) = harness();
    native(&mut agent);
    agent.task.agent_config.refusal_fallback = Some(AgentLLMConfig {
        model: ai::Model::openai_gpt_6_sol(),
        generation_config: Default::default(),
    });
    agent.task.conversation.system_message = Some("resumed".into());
    agent.task.conversation.add_user_message("task");
    agent.task.fallback_for = Some(model_refusal());
    inject(&commands, "First note.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = finish(agent).await;
    assert_eq!(task.current_state, TaskState::InProgress);
    assert!(task.fallback_for.is_some());
    assert!(update_texts(&task.conversation).is_empty(), "held back");
    assert_eq!(task.pending_context_injections, ["First note."]);

    let (mut agent, _events, commands) = resumed(task);
    let requested = Arc::clone(&agent.requested_models);
    inject(&commands, "Second note.");
    push(
        &mut agent,
        turn(vec![native_call(
            "missing",
            "missing_tool",
            serde_json::json!({}),
        )]),
    );
    push(&mut agent, complete());
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(*requested.lock().unwrap(), ["gpt-6-sol", "gpt-5.6-luna"]);
    assert_calls_paired(&task.conversation);
    let fallback_turn = task
        .conversation
        .messages
        .iter()
        .position(|message| message.role == ai::Role::Assistant)
        .unwrap();
    let updates = updates(&task.conversation);
    assert_eq!(
        updates,
        [(fallback_turn + 2, "First note.\n\nSecond note.".to_string())],
        "{:#?}",
        task.conversation.messages
    );
    assert!(updates[0].0 < last_assistant(&task.conversation));
    assert!(task.pending_context_injections.is_empty());
}

/// Sends context through the task's command channel as it runs, then drops
/// the channel's last sender: a host that shuts down right after sending.
#[derive(Debug)]
struct SendsThenCloses(std::sync::Mutex<Option<mpsc::Sender<UserCommand>>>);

#[async_trait::async_trait]
impl Tool for SendsThenCloses {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "notify".into(),
            description: "Sends a note, then closes the channel".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, crate::tool::ToolError> {
        let sender = self.0.lock().unwrap().take().expect("called once");
        sender
            .try_send(UserCommand::InjectContext {
                content: "Last words.".into(),
            })
            .unwrap();
        Ok("sent".into())
    }
}

/// A task returned interrupted on a refusal, by the configured model or,
/// after it, by the fallback, keeps the context taken meanwhile out of the
/// conversation, so the refused input can still be changed
/// (`Task::resolve_refusal` needs it to be the last input). Resolved and run
/// again, the task gets the context after the changed input, once.
#[tokio::test(start_paused = true)]
async fn context_taken_before_a_refusal_leaves_it_resolvable() {
    for fallback in [false, true] {
        let (mut agent, _events, commands) = harness();
        native(&mut agent);
        agent.task.agent_config.llm_error_policy =
            LLMErrorPolicy::RetryThenInformAgent { max_attempts: 3 };
        if fallback {
            agent.task.agent_config.refusal_fallback = Some(AgentLLMConfig {
                model: ai::Model::openai_gpt_6_sol(),
                generation_config: Default::default(),
            });
        }
        let refused = || refused_response(ai::Message::new_assistant(""), model_refusal());
        let slow_refusal = futures::stream::once(async move {
            tokio_sleep(Duration::from_secs(1)).await;
            Ok(refused())
        })
        .boxed();
        push(&mut agent, Ok(slow_refusal));
        if fallback {
            push(&mut agent, scripted(refused()));
        }
        let run = tokio::spawn(agent.run());
        tokio_sleep(Duration::from_millis(500)).await;
        inject(&commands, "Peer note.");
        let mut task = joined(run).await;
        assert!(
            matches!(task.current_state, TaskState::Interrupted { .. }),
            "{fallback}: {:?}",
            task.current_state
        );
        assert!(task.fallback_for.is_none());
        assert!(update_texts(&task.conversation).is_empty(), "{fallback}");
        assert_eq!(task.pending_context_injections, ["Peer note."]);
        task.resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
            message_index: 0,
            text: "Rephrased.".into(),
        }])
        .unwrap();

        let (mut agent, _events, _commands) = resumed(task);
        push(&mut agent, scripted(complete_response()));
        let task = finish(agent).await;
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert!(text(&task.conversation.messages[0]).starts_with("Rephrased."));
        assert_eq!(
            updates(&task.conversation),
            [(1, "Peer note.".to_string())],
            "{fallback}: {:#?}",
            task.conversation.messages
        );
        assert_eq!(last_assistant(&task.conversation), 2);
        assert!(task.pending_context_injections.is_empty());
    }
}

/// As the loop stops, context stays with the task while any of the last
/// turn's results is not in the conversation, even in a state that awaits
/// nothing; once they all are, it is added.
#[test]
fn context_waits_for_every_result_of_the_last_turn() {
    let call = native_call("read", "read_file", serde_json::json!({}));
    let mut open_turn = ai::Message::new_assistant("");
    open_turn.tool_calls = vec![call.clone()];
    let result = ai::ToolResult::from_tool_call(&call, "out");
    for case in [
        "open turn",
        "native result",
        "native call",
        "text result",
        "none",
    ] {
        let (mut agent, _events, _commands) = harness();
        let task = &mut agent.task;
        task.set_state(TaskState::Interrupted {
            reason: "stopped".into(),
        });
        task.conversation.add_user_message("task");
        match case {
            "open turn" => task.conversation.messages.push(open_turn.clone()),
            "native result" => task.pending_native_tool_results = vec![result.clone()],
            "native call" => {
                task.pending_native_tool_calls
                    .insert(Uuid::new_v4(), call.clone());
            }
            "text result" => task.pending_text_tool_results = vec!["out".into()],
            _ => {}
        }
        task.pending_context_injections = vec!["Note.".into()];
        let messages = task.conversation.messages.len();
        agent.keep_taken_context().unwrap();
        if case == "none" {
            assert_eq!(update_texts(&agent.task.conversation), ["Note."]);
            assert!(agent.task.pending_context_injections.is_empty());
        } else {
            assert_eq!(agent.task.conversation.messages.len(), messages, "{case}");
            assert_eq!(agent.task.pending_context_injections, ["Note."], "{case}");
        }
    }
}

/// L2: when the host's side of the channel closes while the loop takes the
/// queued commands, the context already taken is added to the interrupted
/// task's conversation, after the results of the turn before.
#[tokio::test]
async fn a_disconnect_while_taking_commands_keeps_the_context_taken() {
    let workspace = workspace();
    let notify = Arc::new(SendsThenCloses(std::sync::Mutex::new(None)));
    let mut registry = ToolRegistry::new();
    registry.register(notify.clone());
    let (mut agent, _events, commands) = agent(
        config(ToolCallingMode::Native, &["notify"]),
        registry,
        &workspace,
    );
    // The tool holds the only sender.
    *notify.0.lock().unwrap() = Some(commands);
    let requested = Arc::clone(&agent.requested_models);
    push(
        &mut agent,
        turn(vec![native_call("n", "notify", serde_json::json!({}))]),
    );
    push(&mut agent, complete());
    let task = finish(agent).await;
    assert_eq!(
        task.current_state,
        TaskState::Interrupted {
            reason: "Command channel disconnected.".into()
        }
    );
    assert_eq!(requested.lock().unwrap().len(), 1);
    assert_calls_paired(&task.conversation);
    let results = task
        .conversation
        .messages
        .iter()
        .position(|message| message.content.iter().any(|c| c.as_tool_result().is_some()))
        .unwrap();
    assert_eq!(
        updates(&task.conversation),
        [(results + 1, "Last words.".to_string())]
    );
    assert!(task.pending_context_injections.is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// L3: context queued ahead of a cancel is recorded in the cancelled task's
/// conversation, at a turn boundary and during an approval wait (after the
/// results of the calls it waited on), and the cancel still makes no
/// request.
#[tokio::test]
async fn context_queued_ahead_of_a_cancel_is_recorded() {
    let (at_boundary, _events, commands) = harness();
    let requested = Arc::clone(&at_boundary.requested_models);
    inject(&commands, "Before the cancel.");
    commands.try_send(UserCommand::CancelTask).unwrap();
    let task = finish(at_boundary).await;
    assert_eq!(
        task.current_state,
        TaskState::Cancelled {
            reason: CANCELLED_BY_USER.into()
        }
    );
    assert_eq!(update_texts(&task.conversation), ["Before the cancel."]);
    assert!(requested.lock().unwrap().is_empty(), "no request was made");
    assert!(task.pending_context_injections.is_empty());

    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["read_file"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    let requested = Arc::clone(&agent.requested_models);
    push(
        &mut agent,
        turn(vec![native_call(
            "read",
            "read_file",
            serde_json::json!({"path": "a.txt"}),
        )]),
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }).then_some(())
    })
    .await;
    inject(&commands, "While it waited.");
    commands.try_send(UserCommand::CancelTask).unwrap();
    let task = joined(run).await;
    assert!(matches!(task.current_state, TaskState::Cancelled { .. }));
    assert_eq!(requested.lock().unwrap().len(), 1);
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "read")[0].output,
        TASK_CANCELLED_CALL
    );
    let results = task
        .conversation
        .messages
        .iter()
        .position(|message| message.content.iter().any(|c| c.as_tool_result().is_some()))
        .unwrap();
    assert_eq!(
        updates(&task.conversation),
        [(results + 1, "While it waited.".to_string())]
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// L4: a suspension during a stream keeps the partial output, then the
/// context taken during the stream (no call is open: the calls it announced
/// are not in the conversation), and reports the state (N3). Run again, the
/// context is sent once.
#[tokio::test]
async fn a_suspension_during_a_stream_adds_the_context_taken_after_the_partial_output() {
    let (mut agent, mut events, commands) = harness();
    native(&mut agent);
    let stream = futures::stream::iter([
        Ok(ai::StreamEvent::TextChunk("Partial answer".into())),
        Ok(ai::StreamEvent::ToolCallStart {
            index: 0,
            name: "read_file".into(),
        }),
    ])
    .chain(futures::stream::pending())
    .boxed();
    push(&mut agent, Ok(stream));
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::ToolCallInitiated { .. }).then_some(())
    })
    .await;
    inject(&commands, "Mid-stream note.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = joined(run).await;
    rest(&mut events, &mut seen).await;
    assert_eq!(task.current_state, TaskState::InProgress);
    let messages = &task.conversation.messages;
    let partial = messages.len() - 2;
    assert_eq!(messages[partial].role, ai::Role::Assistant);
    assert!(!messages[partial].has_tool_calls());
    assert_eq!(text(&messages[partial]), "Partial answer");
    assert_eq!(
        updates(&task.conversation),
        [(partial + 1, "Mid-stream note.".to_string())]
    );
    assert!(task.pending_context_injections.is_empty());
    // The announced call was rejected, then the suspension reported.
    let rejected = seen
        .iter()
        .position(|event| matches!(event, AgentEvent::ToolRejected { reason: Some(reason), .. } if reason == INTERRUPTED_CALL))
        .unwrap();
    let suspended = seen
        .iter()
        .rposition(|event| matches!(event, AgentEvent::TaskStateChanged(TaskState::InProgress)))
        .unwrap();
    assert!(suspended > rejected, "{seen:#?}");

    let (mut agent, _events, _commands) = resumed(task);
    push(&mut agent, scripted(complete_response()));
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    let updates = updates(&task.conversation);
    assert_eq!(updates.len(), 1);
    assert!(updates[0].0 < last_assistant(&task.conversation));
}

/// L4: a suspension during a retry wait adds the context taken during the
/// wait to the conversation, and a run again sends it once.
#[tokio::test(start_paused = true)]
async fn a_suspension_during_a_retry_wait_adds_the_context_taken() {
    let (mut agent, mut events, commands) = harness();
    native(&mut agent);
    push(&mut agent, Err(provider_failure(500, None)));
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::RetryScheduled { .. }).then_some(())
    })
    .await;
    inject(&commands, "Retry note.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = joined(run).await;
    assert_eq!(task.current_state, TaskState::InProgress);
    let last = task.conversation.messages.len() - 1;
    assert_eq!(
        updates(&task.conversation),
        [(last, "Retry note.".to_string())]
    );
    assert!(task.pending_context_injections.is_empty());

    let (mut agent, _events, _commands) = resumed(task);
    push(&mut agent, scripted(complete_response()));
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(update_texts(&task.conversation), ["Retry note."]);
}

/// L4: in every awaiting state, context taken while the task waits is saved
/// with the suspended task, not added: the answer the task waits for (such
/// as a turn's tool results) comes first.
#[tokio::test]
async fn context_taken_in_an_awaiting_state_is_saved_with_the_suspended_task() {
    let workspace = workspace();
    let pending = |name: &str| PendingToolCallInfo {
        tool_call_id: Uuid::new_v4(),
        tool_name: name.into(),
        parameters: HashMap::new(),
        tool_description: String::new(),
        approval_reason: Default::default(),
    };
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        for state in [
            TaskState::AwaitingToolApproval {
                tool_call_id: Uuid::new_v4(),
                tool_name: "read_file".into(),
                parameters: HashMap::new(),
                tool_description: String::new(),
                approval_reason: Default::default(),
            },
            TaskState::AwaitingBatchToolApproval {
                batch_id: Uuid::new_v4(),
                pending_calls: vec![pending("read_file"), pending("list_files")],
            },
            TaskState::AwaitingUserResponse {
                questions: "Which file?".into(),
            },
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "failed".into(),
                cause: Default::default(),
            },
        ] {
            let mut config = config(mode.clone(), &["read_file", "list_files"]);
            config.tool_config.default_auto_approve_tool_use = false;
            let (mut agent, _events, commands) = agent(config, ToolRegistry::new(), &workspace);
            agent.task.set_state(state.clone());
            inject(&commands, "First note.");
            inject(&commands, "Second note.");
            commands.try_send(UserCommand::SuspendTask).unwrap();
            let task = finish(agent).await;
            assert_eq!(task.current_state, state, "{mode:?}");
            assert_eq!(
                task.pending_context_injections,
                ["First note.", "Second note."],
                "{mode:?} {state:?}"
            );
            assert!(update_texts(&task.conversation).is_empty());
        }
    }
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Where a message with tool results is in the conversation.
fn results_message(conversation: &ai::Conversation, call_id: &str) -> usize {
    conversation
        .messages
        .iter()
        .position(|message| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_tool_result)
                .any(|result| result.call_id.as_deref() == Some(call_id))
        })
        .unwrap()
}

/// L4: context taken while a native turn awaits approval is saved with the
/// suspended task, and after a save, a load and the approval it goes into
/// the conversation right after the turn's results, before context sent
/// later, once.
#[tokio::test]
async fn context_taken_during_an_approval_wait_follows_the_results_after_a_resume() {
    let workspace = workspace();
    std::fs::write(workspace.join("a.txt"), "A-CONTENTS").unwrap();
    let mut config = config(ToolCallingMode::Native, &["read_file"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    push(
        &mut agent,
        turn(vec![native_call(
            "read",
            "read_file",
            serde_json::json!({"path": "a.txt"}),
        )]),
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    let tool_call_id = wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => Some(*tool_call_id),
        _ => None,
    })
    .await;
    inject(&commands, "While it waited.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = joined(run).await;
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingToolApproval { .. }
    ));
    assert_eq!(task.pending_context_injections, ["While it waited."]);
    assert!(task.conversation.messages.last().unwrap().has_tool_calls());
    assert_nothing_between_calls_and_results(&task.conversation);

    let path = workspace.join("task.json");
    task.save_to_path(&path).unwrap();
    let task = Task::load_from_path(&path).unwrap();
    let (mut agent, _events, commands) =
        agent_for(task, ToolRegistry::new(), Arc::new(ai::LLMClient::new()));
    push(&mut agent, complete());
    commands
        .try_send(UserCommand::ApproveToolUse {
            tool_call_id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .unwrap();
    inject(&commands, "After the resume.");
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    assert!(
        result_for(&task.conversation, "read")[0]
            .output
            .contains("A-CONTENTS")
    );
    let results = results_message(&task.conversation, "read");
    assert_eq!(
        updates(&task.conversation),
        [(
            results + 1,
            "While it waited.\n\nAfter the resume.".to_string()
        )]
    );
    assert!(results + 1 < last_assistant(&task.conversation));
    assert!(task.pending_context_injections.is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// L4: context taken while the task waits for the answer to `ask_user` is
/// saved with it; answered through the task, the context follows the answer.
#[tokio::test]
async fn context_taken_while_a_question_waits_follows_the_answer() {
    let workspace = workspace();
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &["ask_user"]),
        ToolRegistry::new(),
        &workspace,
    );
    push(
        &mut agent,
        turn(vec![native_call(
            "ask",
            "ask_user",
            serde_json::json!({"questions": "Which file?"}),
        )]),
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::UserQuestionsAsked { .. }).then_some(())
    })
    .await;
    inject(&commands, "While you wait.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let mut task = joined(run).await;
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingUserResponse { .. }
    ));
    assert_eq!(task.pending_context_injections, ["While you wait."]);
    assert_nothing_between_calls_and_results(&task.conversation);

    task.prepare_for_resumption_with_user_response("README.md".into())
        .unwrap();
    let (mut agent, _events, _commands) = resumed(task);
    push(&mut agent, complete());
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    // The auto-approved question's result went in as it was asked; the
    // answer follows it.
    let answer = task
        .conversation
        .messages
        .iter()
        .position(|message| text(message).contains("Response: README.md"))
        .unwrap();
    assert!(answer > results_message(&task.conversation, "ask"));
    assert_eq!(
        updates(&task.conversation),
        [(answer + 1, "While you wait.".to_string())]
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// L4: context taken while a native turn's tool error pauses the task (its
/// results wait with it) is saved with it, and follows the results once
/// recovery is approved.
#[tokio::test]
async fn context_taken_during_a_tool_error_pause_follows_the_results() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["read_file"]);
    config.error_recovery_config.on_tool_execution_error =
        crate::agent_config::ToolErrorPolicy::PauseAndAwaitApproval;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    push(
        &mut agent,
        turn(vec![native_call(
            "missing",
            "read_file",
            serde_json::json!({"path": "missing.txt"}),
        )]),
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval { .. })
        )
        .then_some(())
    })
    .await;
    inject(&commands, "During the pause.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = joined(run).await;
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingErrorRecoveryApproval { .. }
    ));
    assert!(!task.pending_native_tool_results.is_empty());
    assert_eq!(task.pending_context_injections, ["During the pause."]);
    assert_nothing_between_calls_and_results(&task.conversation);

    let (mut agent, _events, commands) = resumed(task);
    push(&mut agent, complete());
    commands
        .try_send(UserCommand::ApproveErrorRecovery { message: None })
        .unwrap();
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    let results = results_message(&task.conversation, "missing");
    assert_eq!(
        updates(&task.conversation),
        [(results + 1, "During the pause.".to_string())]
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

fn slow_down(agent: &mut AgentLoop) {
    agent.task.agent_config.monitoring = MonitoringConfig {
        plan_usage: Some(PlanUsageThreshold {
            percent: 90.0,
            action: PlanUsageAction::SlowDown { delay_ms: 10_000 },
        }),
        ..Default::default()
    };
    native(agent);
    push(agent, scripted(turn_with_plan_usage(95.0)));
    push(agent, scripted(complete_response()));
}

/// L4: context sent while the plan-usage slow-down waits before a request
/// joins that request, not the one after; a suspension during the wait
/// keeps it.
#[tokio::test(start_paused = true)]
async fn context_sent_during_a_slow_down_joins_the_request_after_it() {
    let (mut agent, mut events, commands) = harness();
    slow_down(&mut agent);
    let started = tokio::time::Instant::now();
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::PlanUsageThresholdReached { .. }).then_some(())
    })
    .await;
    inject(&commands, "During the wait.");
    let task = joined(run).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(started.elapsed(), Duration::from_secs(10));
    assert_calls_paired(&task.conversation);
    let updates = updates(&task.conversation);
    assert_eq!(updates.len(), 1, "{:#?}", task.conversation.messages);
    assert_eq!(updates[0].1, "During the wait.");
    assert_eq!(updates[0].0 + 1, last_assistant(&task.conversation));

    let (mut agent, mut events, commands) = harness();
    slow_down(&mut agent);
    let requested = Arc::clone(&agent.requested_models);
    let run = tokio::spawn(agent.run());
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::PlanUsageThresholdReached { .. }).then_some(())
    })
    .await;
    inject(&commands, "During the wait.");
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = joined(run).await;
    assert_eq!(task.current_state, TaskState::InProgress);
    assert_eq!(requested.lock().unwrap().len(), 1);
    assert_calls_paired(&task.conversation);
    assert_eq!(update_texts(&task.conversation), ["During the wait."]);
    assert!(task.pending_context_injections.is_empty());
}

/// N3: a suspension taken at a turn boundary reports the task's state, as a
/// suspension anywhere else does.
#[tokio::test]
async fn a_suspension_at_a_turn_boundary_reports_the_state() {
    let (agent, mut events, commands) = harness();
    commands.try_send(UserCommand::SuspendTask).unwrap();
    let task = finish(agent).await;
    assert_eq!(task.current_state, TaskState::InProgress);
    let mut seen = Vec::new();
    rest(&mut events, &mut seen).await;
    // The start, then the suspension.
    assert_eq!(
        states(&seen),
        [TaskState::InProgress, TaskState::InProgress]
    );
    assert!(matches!(
        seen.as_slice(),
        [
            ..,
            AgentEvent::TaskStateChanged(TaskState::InProgress),
            AgentEvent::TaskStreamEnd
        ]
    ));
}

/// Context sent with the approval of a call, as a workflow's driver
/// approves every call, reaches the request right after the call: the loop
/// takes the queued commands after an approved call too.
#[tokio::test]
async fn context_sent_with_an_approval_reaches_the_very_next_request() {
    let workspace = workspace();
    std::fs::write(workspace.join("a.txt"), "A-CONTENTS").unwrap();
    let mut config = config(ToolCallingMode::Native, &["read_file"]);
    config.tool_config.default_auto_approve_tool_use = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    push(
        &mut agent,
        turn(vec![native_call(
            "read",
            "read_file",
            serde_json::json!({"path": "a.txt"}),
        )]),
    );
    push(&mut agent, complete());
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    let tool_call_id = wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => Some(*tool_call_id),
        _ => None,
    })
    .await;
    commands
        .try_send(UserCommand::ApproveToolUse {
            tool_call_id,
            custom_tool_output: None,
            message_to_agent: None,
        })
        .unwrap();
    inject(&commands, "Sent with the approval.");
    let task = joined(run).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    let results = results_message(&task.conversation, "read");
    assert_eq!(
        updates(&task.conversation),
        [(results + 1, "Sent with the approval.".to_string())]
    );
    assert_eq!(results + 2, last_assistant(&task.conversation));
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Context taken during a task's final turn is not added to the completed
/// task's conversation, which the model never read after it; the task keeps
/// it, and a continuation sends it once.
#[tokio::test(start_paused = true)]
async fn context_taken_in_the_final_turn_waits_for_a_continuation() {
    let (mut agent, _events, commands) = harness();
    native(&mut agent);
    let final_turn = futures::stream::once(async {
        tokio_sleep(Duration::from_secs(1)).await;
        Ok(complete_response())
    })
    .boxed();
    push(&mut agent, Ok(final_turn));
    let run = tokio::spawn(agent.run());
    tokio_sleep(Duration::from_millis(500)).await;
    inject(&commands, "Too late.");
    let mut task = joined(run).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(update_texts(&task.conversation).is_empty());
    assert_eq!(task.pending_context_injections, ["Too late."]);

    task.prepare_for_resumption_from_completed("Carry on.".into())
        .unwrap();
    let (mut agent, _events, _commands) = resumed(task);
    push(&mut agent, scripted(complete_response()));
    let task = finish(agent).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_calls_paired(&task.conversation);
    let carry_on = task
        .conversation
        .messages
        .iter()
        .position(|message| text(message) == "Carry on.")
        .unwrap();
    assert_eq!(
        updates(&task.conversation),
        [(carry_on + 1, "Too late.".to_string())]
    );
    assert!(task.pending_context_injections.is_empty());
}
