//! Calls that wait while context arrives for their task.

use super::tool_call_tests::{
    agent, assert_calls_paired, complete, config, native_call, result_for, turn, wait_for,
    workspace,
};
use super::*;
use crate::tool::{Tool, ToolError, ToolTypeDetails};
use async_trait::async_trait;

/// A tool that waits until context arrives for its task, or `limit`.
#[derive(Debug)]
struct WaitsForContext {
    limit: Duration,
}

#[async_trait]
impl Tool for WaitsForContext {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "waits_for_context".into(),
            description: "Waits for context".into(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        match timeout(self.limit, context.context_arrived.arrived()).await {
            Ok(()) => Ok("context arrived".into()),
            Err(_) => Ok("no context".into()),
        }
    }
}

fn waits_for_context(limit: Duration) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(WaitsForContext { limit }));
    registry
}

/// Context sent while a call waits ends the wait, and reaches the model with
/// the call's result.
#[tokio::test]
async fn context_sent_while_a_call_waits_ends_the_wait() {
    let workspace = workspace();
    let (mut agent, mut events, commands) = agent(
        config(ToolCallingMode::Native, &["waits_for_context"]),
        waits_for_context(Duration::from_secs(60)),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "wait",
            "waits_for_context",
            serde_json::json!({}),
        )]),
        complete(),
    ]);
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    wait_for(&mut events, &mut seen, |event| {
        matches!(event, AgentEvent::ToolCallFormatted { tool_name, .. } if tool_name == "waits_for_context")
            .then_some(())
    })
    .await;
    commands
        .send(UserCommand::InjectContext {
            content: "The build finished.".into(),
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(10), run)
        .await
        .expect("the wait ends when the context arrives")
        .unwrap()
        .unwrap();
    assert_calls_paired(&task.conversation);
    assert_eq!(
        result_for(&task.conversation, "wait")[0].output,
        "context arrived"
    );
    let results = task
        .conversation
        .messages
        .iter()
        .position(|message| result_for_message(message, "wait"))
        .unwrap();
    // The context follows the result, before the next request.
    assert!(
        task.conversation.messages[results..]
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(ai::Content::as_text)
            .any(|text| text.contains("The build finished."))
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

fn result_for_message(message: &ai::Message, call_id: &str) -> bool {
    message
        .content
        .iter()
        .filter_map(ai::Content::as_tool_result)
        .any(|result| result.call_id.as_deref() == Some(call_id))
}

/// Context taken before a call starts is in the conversation already, so it
/// doesn't end the call's wait.
#[tokio::test]
async fn context_taken_before_a_call_doesnt_end_its_wait() {
    let workspace = workspace();
    let (mut agent, _events, commands) = agent(
        config(ToolCallingMode::Native, &["waits_for_context"]),
        waits_for_context(Duration::from_millis(300)),
        &workspace,
    );
    agent.scripted_responses.get_mut().unwrap().extend([
        turn(vec![native_call(
            "wait",
            "waits_for_context",
            serde_json::json!({}),
        )]),
        complete(),
    ]);
    commands
        .send(UserCommand::InjectContext {
            content: "Sent before the call.".into(),
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result_for(&task.conversation, "wait")[0].output,
        "no context"
    );
    std::fs::remove_dir_all(workspace).unwrap();
}
