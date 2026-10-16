//! A task continued with a new configuration (`Task::reconfigure`): a chat
//! that becomes an agent offers its tools under a new system prompt, and an
//! agent that becomes a chat sends its tool history as text.

use super::tool_call_tests::agent_for;
use super::tool_less_tests::{chat_agent, push, replying_agent, run};
use super::*;

fn chat_config() -> AgentConfig {
    let mut config = super::tests::test_config();
    config.tool_config = crate::agent_config::AgentToolConfig {
        tool_calling_mode: config.tool_config.tool_calling_mode,
        ..crate::agent_config::AgentToolConfig::chat()
    };
    config
}

fn agent_config(tools: &[&str]) -> AgentConfig {
    let mut config = chat_config();
    config.tool_config.chat = false;
    config.tool_config.available_tool_names = tools.iter().map(|tool| tool.to_string()).collect();
    config
}

fn registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(super::image_tests::TextTool));
    registry
}

/// A chat continued as an agent: the same conversation and prompt cache
/// key, its tools offered, and a system prompt that says how to use them.
#[tokio::test]
async fn a_chat_continued_as_an_agent_offers_its_tools_under_a_new_prompt() {
    let (mut agent, mut events, _commands) = chat_agent();
    push(&mut agent, &["Hi."]);
    let (mut task, _) = run(agent, &mut events).await;
    let chat_prompt = task.conversation.system_message.clone().unwrap();
    let cache_key = task.conversation.prompt_cache_key.clone();
    assert!(!chat_prompt.contains("## Tool Usage"));

    task.reconfigure(agent_config(&["plain_text"])).unwrap();
    assert!(task.reconfigured);
    assert!(task.conversation.system_message.is_none());
    task.prepare_for_resumption_from_completed("Now use your tool.".into())
        .unwrap();
    let (mut agent, mut events, _commands) =
        agent_for(task, registry(), Arc::new(ai::LLMClient::new()));
    assert!(agent.offers_tools());
    let (_, config) = agent.next_request();
    assert_eq!(config.tool_config.unwrap().tools.len(), 1);
    push(&mut agent, &["Done."]);
    let (task, _) = run(agent, &mut events).await;

    let prompt = task.conversation.system_message.unwrap();
    assert!(prompt.contains("## Tool Usage"), "{prompt}");
    assert_eq!(task.conversation.prompt_cache_key, cache_key);
    let texts: Vec<_> = task
        .conversation
        .messages
        .iter()
        .map(message_text)
        .collect();
    assert!(texts[0].starts_with("Hello."), "{texts:?}");
    assert_eq!(texts[1], "Hi.");
    assert!(texts[2].starts_with("Now use your tool."), "{texts:?}");
}

/// An agent continued as a chat: its requests offer no tools, and the
/// conversation they send holds its earlier calls and results as text
/// (the task's own conversation keeps them).
#[tokio::test]
async fn an_agent_continued_as_a_chat_sends_its_tool_history_as_text() {
    let (agent, _events, _commands) = replying_agent(&["plain_text"]);
    let mut task = agent.task;
    let call = super::image_tests::call("call_1", "plain_text");
    let mut reply = ai::Message::new_assistant("Checking.");
    reply.tool_calls = vec![call.clone()];
    task.conversation.add_user_message("Hello.");
    task.conversation.messages.push(reply);
    task.conversation
        .add_tool_results(vec![ai::ToolResult::from_tool_call(&call, "plain output")]);
    task.conversation
        .add_assistant_message("It said plain output.");
    task.set_state(TaskState::Completed {
        summary: "It said plain output.".into(),
    });

    task.reconfigure(chat_config()).unwrap();
    task.prepare_for_resumption_from_completed("Thanks. Now just talk.".into())
        .unwrap();
    let (mut agent, _events, _commands) =
        agent_for(task, registry(), Arc::new(ai::LLMClient::new()));
    assert!(!agent.offers_tools());
    let (sent, config) = agent.next_request();
    assert!(config.tool_config.is_none());
    assert!(sent.messages.iter().all(|message| {
        message.tool_calls.is_empty()
            && message
                .content
                .iter()
                .all(|content| content.as_tool_result().is_none())
    }));
    let texts: Vec<_> = sent.messages.iter().map(message_text).collect();
    assert!(
        texts[1].contains("[Earlier tool call call_1: plain_text, arguments {}]"),
        "{texts:?}"
    );
    assert!(
        texts[2].contains(
            "[Result of tool call call_1 (plain_text):]\n> plain output\n[End of the result of call_1]"
        ),
        "{texts:?}"
    );
    assert_eq!(agent.task.conversation.messages[1].tool_calls.len(), 1);
}

/// Only between turns, and only with the same model.
#[test]
fn a_task_is_reconfigured_between_turns_with_its_model() {
    let mut task = Task::new(chat_config(), "Hello.".into(), std::env::temp_dir(), None);
    task.set_state(TaskState::InProgress);
    let error = task.reconfigure(agent_config(&["plain_text"])).unwrap_err();
    assert!(error.to_string().contains("only between turns"), "{error}");

    task.set_state(TaskState::AwaitingUserResponse {
        questions: "Which?".into(),
    });
    let mut other = agent_config(&["plain_text"]);
    other.llm_config.model = ai::Model::anthropic_claude_opus_5_5();
    let error = task.reconfigure(other).unwrap_err();
    assert!(error.to_string().contains("keeps its model"), "{error}");
    assert!(!task.reconfigured);

    task.claude_code_session = Some("session".into());
    task.reconfigure(agent_config(&["plain_text"])).unwrap();
    assert!(task.reconfigured && task.claude_code_prompt_changed);
    assert_eq!(
        task.agent_config.tool_config.available_tool_names,
        ["plain_text"]
    );
    // Saved with the task.
    let saved = serde_json::to_value(&task).unwrap();
    assert_eq!(saved["reconfigured"], true);
    assert_eq!(saved["claude_code_prompt_changed"], true);
    let fresh = serde_json::to_value(Task::new(
        chat_config(),
        "Hi.".into(),
        std::env::temp_dir(),
        None,
    ))
    .unwrap();
    assert!(fresh.get("reconfigured").is_none());
    assert!(fresh.get("claude_code_prompt_changed").is_none());
}

/// A cut back past the first message keeps the conversation's first call
/// as it was: a conversation that went over the handover size and was cut
/// back (an edit, a regenerate) still hands over once its next call is
/// over the size again. Its size is unknown until that call.
#[tokio::test]
async fn a_cut_back_over_the_handover_size_still_hands_over() {
    let (agent, _events, _commands) = replying_agent(&["plain_text"]);
    let mut task = agent.task;
    task.agent_config.handover = Some(crate::handover::HandoverConfig::at_input_tokens(1000));
    let usage = |input| ai::UsageInfo::new(input, 10);
    task.conversation.add_user_message("First.");
    task.record_usage(usage(500));
    task.conversation.add_assistant_message("One.");
    task.conversation.add_user_message("Second.");
    task.record_usage(usage(1100));
    task.conversation.add_assistant_message("Two.");
    task.set_state(TaskState::Completed {
        summary: "Two.".into(),
    });

    task.cut_back(2).unwrap();
    assert_eq!(task.conversation_usage_start, 0);
    assert_eq!(task.current_conversation_context_size(), None);
    task.prepare_for_resumption_from_completed("Second, again.".into())
        .unwrap();
    // The next call is over the size: the handover starts before the one
    // after it.
    task.record_usage(usage(1200));
    assert_eq!(task.current_conversation_context_size(), Some(1200));
    task.conversation.add_assistant_message("Two, again.");
    task.conversation.add_user_message("Third.");
    let (mut agent, _events, _commands) =
        agent_for(task, registry(), Arc::new(ai::LLMClient::new()));
    agent.start_handover_if_due().await.unwrap();
    assert!(agent.task.handover_requested.is_some());
}

/// A reconfiguration starts the cache monitor's window over: the new
/// prefix's full cache write isn't held against the conversation.
#[test]
fn a_reconfiguration_starts_the_cache_window_over() {
    let mut task = Task::new(chat_config(), "Hi.".into(), std::env::temp_dir(), None);
    task.record_usage(ai::UsageInfo::new(100, 10));
    task.set_state(TaskState::Completed {
        summary: "Hello.".into(),
    });
    assert_eq!(task.cache_monitor_start(), 0);
    task.reconfigure(agent_config(&["plain_text"])).unwrap();
    assert_eq!(task.cache_monitor_start(), 1);
    assert_eq!(task.conversation_usage_start, 0);
}

/// A chat is what says so: `AgentToolConfig::chat()` is offered nothing and
/// ends its task with text; an agent that merely lists no tools isn't one.
/// A chat can't list tools, run output-based, or send replies as messages.
#[test]
fn a_chat_is_explicit() {
    use crate::agent_config::{AgentToolConfig, ToolCallingMode};
    let chat = AgentToolConfig::chat();
    assert!(chat.chat && chat.offered_tool_names().is_empty() && chat.ends_task_with_text());
    let quiet = AgentToolConfig {
        available_tool_names: Vec::new(),
        text_reply_ends_task: true,
        ..AgentToolConfig::default()
    };
    assert!(!quiet.chat);
    let mut config = chat_config();
    config.validate().unwrap();
    let refused = |change: fn(&mut AgentToolConfig)| {
        let mut config = chat_config();
        change(&mut config.tool_config);
        config.validate().unwrap_err().to_string()
    };
    assert!(
        refused(|tools| tools.available_tool_names = vec!["read_file".into()])
            .contains("offered no tools")
    );
    assert!(
        refused(|tools| tools.tool_calling_mode = ToolCallingMode::OutputBased)
            .contains("native mode")
    );
    assert!(refused(|tools| tools.text_reply_as_message = true).contains("text_reply_as_message"));
    // Saved without the field unless it is on, and older configurations
    // load without it.
    config.tool_config.chat = false;
    let saved = serde_json::to_value(&config.tool_config).unwrap();
    assert!(saved.get("chat").is_none(), "{saved}");
    let loaded: AgentToolConfig = serde_json::from_value(saved).unwrap();
    assert!(!loaded.chat);
}
