//! Messages to the user without ending the turn: the built-in
//! `message_user` tool, a host's own tool under its name, and the opt-in
//! `text_reply_as_message`.

use super::image_tests::response;
use super::tool_call_tests::{
    agent, complete, config, drain, native_call, result_for, text_turn, turn, workspace,
};
use super::*;
use crate::tool::builtins::message_user::MESSAGE_USER_RESULT;
use crate::tool::{ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;

/// The messages the run sent the user, with their call IDs.
fn messages(events: &[AgentEvent]) -> Vec<(String, Option<Uuid>)> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageToUser {
                message,
                tool_call_id,
            } => Some((message.clone(), *tool_call_id)),
            _ => None,
        })
        .collect()
}

/// The ID of the run's first call to `name`, from its `ToolCallFormatted`.
fn call_id(events: &[AgentEvent], name: &str) -> Uuid {
    events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolCallFormatted {
                tool_call_id,
                tool_name,
                ..
            } if tool_name == name => Some(*tool_call_id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no call to {name}: {events:#?}"))
}

fn position(events: &[AgentEvent], pick: impl Fn(&AgentEvent) -> bool) -> usize {
    events
        .iter()
        .position(pick)
        .unwrap_or_else(|| panic!("event not found: {events:#?}"))
}

fn retries(events: &[AgentEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, AgentEvent::RetryScheduled { .. }))
        .count()
}

/// The text of each message of the conversation.
fn texts(task: &Task) -> Vec<String> {
    task.conversation
        .messages
        .iter()
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_text)
                .collect()
        })
        .collect()
}

/// In either tool calling mode, `message_user` sends its message to the
/// host between the call's `ToolCallFormatted` and its `ToolResult`, and
/// the task carries on without a state change; the model's result says the
/// message was sent.
#[tokio::test]
async fn message_user_sends_the_message_and_the_task_carries_on() {
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let workspace = workspace();
        let native = mode == ToolCallingMode::Native;
        let (mut agent, mut events, _commands) = agent(
            config(mode.clone(), &["message_user"]),
            ToolRegistry::new(),
            &workspace,
        );
        let (message, done) = if native {
            (
                turn(vec![native_call(
                    "msg",
                    "message_user",
                    serde_json::json!({"message": "Halfway there."}),
                )]),
                complete(),
            )
        } else {
            (
                text_turn("<message_user>\n<message>Halfway there.</message>\n</message_user>"),
                text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
            )
        };
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .extend([message, done]);
        let task = timeout(Duration::from_secs(10), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{mode:?}: {:?}",
            task.current_state
        );
        let events = drain(&mut events);
        let id = call_id(&events, "message_user");
        assert_eq!(
            messages(&events),
            [("Halfway there.".to_string(), Some(id))],
            "{mode:?}"
        );
        let sent = position(&events, |event| {
            matches!(event, AgentEvent::MessageToUser { .. })
        });
        let formatted = position(
            &events,
            |event| matches!(event, AgentEvent::ToolCallFormatted { tool_call_id, .. } if *tool_call_id == id),
        );
        let result = position(&events, |event| {
            matches!(event, AgentEvent::ToolResult { tool_call_id, output, .. }
                if *tool_call_id == id && output == MESSAGE_USER_RESULT)
        });
        assert!(formatted < sent && sent < result, "{mode:?}: {events:#?}");
        // The only state change is the completion.
        let states: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TaskStateChanged(state) => Some(state.clone()),
                _ => None,
            })
            .collect();
        assert!(
            states
                .iter()
                .all(|state| matches!(state, TaskState::InProgress | TaskState::Completed { .. })),
            "{mode:?}: {states:?}"
        );
        if native {
            let results = result_for(&task.conversation, "msg");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].output, MESSAGE_USER_RESULT);
            assert!(!results[0].is_error);
        } else {
            assert!(
                texts(&task)
                    .iter()
                    .any(|text| text.contains(MESSAGE_USER_RESULT)),
                "{:#?}",
                task.conversation.messages
            );
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// An empty message is a tool error (handled under the tool-error policy,
/// which by default tells the agent and continues): nothing is sent.
#[tokio::test]
async fn an_empty_message_is_not_sent() {
    let workspace = workspace();
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::Native, &["message_user"]),
        ToolRegistry::new(),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "msg",
            "message_user",
            serde_json::json!({"message": "  "}),
        )]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(messages(&drain(&mut events)).is_empty());
    assert!(result_for(&task.conversation, "msg")[0].is_error);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A host's own `message_user`, as hosts registered before arko had one.
#[derive(Debug)]
struct HostMessageUser;

#[async_trait]
impl Tool for HostMessageUser {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "message_user".into(),
            description: "The host's own message tool".into(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: "message".into(),
                description: "The message".into(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        Ok("Shown by the host.".into())
    }
}

/// A host's tool registered as `message_user` replaces the built-in: the
/// model is offered the host's definition, its calls run the host's tool,
/// and the runtime sends no `MessageToUser` for them.
#[tokio::test]
async fn a_hosts_own_message_user_replaces_the_built_in() {
    let workspace = workspace();
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(HostMessageUser));
    assert_eq!(
        registry
            .get("message_user")
            .unwrap()
            .definition()
            .description,
        "The host's own message tool"
    );
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::Native, &["message_user"]),
        registry,
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "msg",
            "message_user",
            serde_json::json!({"message": "Halfway there."}),
        )]),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(messages(&drain(&mut events)).is_empty());
    assert_eq!(
        result_for(&task.conversation, "msg")[0].output,
        "Shown by the host."
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// The built-in is registered but offered only when listed.
#[test]
fn message_user_is_a_built_in_offered_when_listed() {
    let registry = ToolRegistry::new();
    let tool = registry.get("message_user").unwrap();
    let definition = tool.definition();
    assert_eq!(definition.tool_type_details, ToolTypeDetails::BuiltIn);
    assert!(definition.parallel_safe);
    assert_eq!(definition.parameters.len(), 1);
    assert_eq!(definition.parameters[0].name, "message");
    assert!(definition.parameters[0].required);
    assert_eq!(tool.recovery_mode(), ToolRecovery::Rerun);
    let unlisted = config(ToolCallingMode::Native, &["read_file"]);
    assert!(
        !unlisted
            .tool_config
            .offered_tool_names()
            .contains("message_user")
    );
    // It follows the approval default like any tool.
    let mut gated = config(ToolCallingMode::Native, &["message_user"]);
    gated.tool_config.default_auto_approve_tool_use = false;
    assert!(!gated.should_auto_approve_tool("message_user"));
}

/// By default a reply in plain text is a response the runtime can't use
/// (see `a_response_without_a_tool_call_gets_a_note`); nothing is sent to
/// the user, and the system prompt doesn't say text reaches the user.
#[tokio::test]
async fn text_replies_are_not_messages_by_default() {
    let workspace = workspace();
    let (mut agent, mut events, _commands) = agent(
        config(ToolCallingMode::Native, &[]),
        ToolRegistry::new(),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        response(ai::Message::new_assistant("Halfway there.")),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let events = drain(&mut events);
    assert!(messages(&events).is_empty());
    assert_eq!(retries(&events), 1);
    assert_eq!(texts(&task)[2], NO_NATIVE_TOOL_CALL);
    let system = task.conversation.system_message.clone().unwrap();
    assert!(
        system.contains("A response without a tool call does not end the task."),
        "{system}"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// With `text_reply_as_message`, a reply in plain text is sent to the user
/// as `MessageToUser` (without a call ID), and the run carries on after a
/// note, without an error. Another reply in plain text right after it is a
/// response without a tool call, as with the setting off; after a tool
/// call, a reply in plain text is a message again.
#[tokio::test]
async fn text_replies_become_messages_when_the_agent_opts_in() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["message_user"]);
    config.tool_config.text_reply_as_message = true;
    let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    agent.scripted_responses.get_mut().unwrap().extend([
        response(ai::Message::new_assistant("First reply.")),
        response(ai::Message::new_assistant("Second reply.")),
        turn(vec![native_call(
            "msg",
            "message_user",
            serde_json::json!({"message": "Through the tool."}),
        )]),
        response(ai::Message::new_assistant("After the call.")),
        complete(),
    ]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(task.current_state, TaskState::Completed { .. }),
        "{:?}",
        task.current_state
    );
    let events = drain(&mut events);
    let id = call_id(&events, "message_user");
    assert_eq!(
        messages(&events),
        [
            ("First reply.".to_string(), None),
            ("Through the tool.".to_string(), Some(id)),
            ("After the call.".to_string(), None),
        ]
    );
    // Only the second reply went through recovery.
    assert_eq!(retries(&events), 1);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ErrorEncountered { .. }))
            .count(),
        1
    );
    let texts = texts(&task);
    assert_eq!(
        &texts[1..6],
        [
            "First reply.",
            TEXT_REPLY_SENT,
            "Second reply.",
            NO_NATIVE_TOOL_CALL,
            ""
        ]
    );
    assert_eq!(&texts[7..9], ["After the call.", TEXT_REPLY_SENT]);
    for index in [2, 4, 8] {
        assert!(task.runtime_messages.contains(&index), "{index}");
    }
    let system = task.conversation.system_message.clone().unwrap();
    assert!(
        system.contains(
            "A reply in plain text, without a tool call, is sent to the user as a message"
        ),
        "{system}"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A response with neither text nor a tool call has nothing to send: it is
/// a response without a tool call even with the setting on.
#[tokio::test]
async fn an_empty_reply_is_not_a_message() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &[]);
    config.tool_config.text_reply_as_message = true;
    let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([response(ai::Message::new_assistant(" \n")), complete()]);
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let events = drain(&mut events);
    assert!(messages(&events).is_empty());
    assert_eq!(retries(&events), 1);
    std::fs::remove_dir_all(workspace).unwrap();
}

/// The event round-trips, and one saved without a call ID loads.
#[test]
fn the_event_round_trips() {
    let id = Uuid::new_v4();
    for event in [
        AgentEvent::MessageToUser {
            message: "Halfway there.".into(),
            tool_call_id: Some(id),
        },
        AgentEvent::MessageToUser {
            message: "A reply.".into(),
            tool_call_id: None,
        },
    ] {
        let saved = serde_json::to_value(&event).unwrap();
        assert_eq!(serde_json::from_value::<AgentEvent>(saved).unwrap(), event);
    }
    let saved = serde_json::to_value(AgentEvent::MessageToUser {
        message: "A reply.".into(),
        tool_call_id: None,
    })
    .unwrap();
    assert_eq!(
        saved,
        serde_json::json!({"MessageToUser": {"message": "A reply."}})
    );
}
