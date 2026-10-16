//! Editing a message (`ContinueOptions::cut_back_to`) and branching a run
//! (`Task::branch`, `Workflow::import_agent_run`).

use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::notes::Note;
use crate::test_support::{ResponsesServer, Workspace, agent_config, text_turn};

/// A chat: its reply in text ends its task.
fn chat() -> AgentConfig {
    let mut config = agent_config("Main");
    config.tool_config.text_reply_ends_task = true;
    config
}

async fn server() -> ResponsesServer {
    ResponsesServer::start(|index, _| {
        Some(text_turn(&format!("t{index}"), &format!("Reply {index}.")))
    })
    .await
}

fn workflow(server: &ResponsesServer, workspace: &Workspace) -> Workflow {
    Workflow::builder()
        .name("cut-back-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", chat())
        .build()
        .unwrap()
}

/// The texts of a request's input, in order.
fn input_texts(body: &str) -> Vec<String> {
    let body: Value = serde_json::from_str(body).unwrap();
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| match &item["content"] {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        })
        .collect()
}

/// An edited message: the run goes on from just before it, with the new
/// text in its place and the messages after it gone.
#[tokio::test]
async fn an_edited_message_replaces_it_and_what_came_after() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let second = workflow
        .continue_agent(&first, "Tell me a joke.")
        .await
        .unwrap();
    assert_eq!(second.task.user_turns()[1].index, 2);

    let edited = workflow
        .continue_agent_with(
            &second,
            "Tell me a riddle.",
            ContinueOptions::new().cut_back_to(2),
        )
        .await
        .unwrap();
    assert_eq!(edited.agent_run_id, first.agent_run_id);
    let sent = input_texts(server.bodies().last().unwrap());
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert!(sent[0].starts_with("Hello."));
    assert_eq!(sent[1], "Reply 0.");
    assert!(sent[2].starts_with("Tell me a riddle."));
    assert!(!sent.iter().any(|text| text.contains("joke")));
    assert_eq!(edited.task.conversation.messages.len(), 4);

    // Not a message the host sent: refused, nothing sent.
    let requests = server.requests();
    let error = workflow
        .continue_agent_with(&edited, "x", ContinueOptions::new().cut_back_to(1))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("not a message the host sent")),
        "{error}"
    );
    assert_eq!(server.requests(), requests);
}

/// Cut back to the first message, the run starts over from the new prompt
/// under the same run ID.
#[tokio::test]
async fn an_edited_first_message_starts_the_run_over() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let again = workflow
        .continue_agent_with(
            &first,
            "Good evening.",
            ContinueOptions::new().cut_back_to(0),
        )
        .await
        .unwrap();
    assert_eq!(again.agent_run_id, first.agent_run_id);
    let sent = input_texts(server.bodies().last().unwrap());
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert!(sent[0].starts_with("Good evening."));
}

/// The notes the removed messages gave the run are given again, except the
/// one-time notes, which went with them.
#[tokio::test]
async fn a_cut_gives_standing_notes_again_but_not_one_time_ones() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.post_note(Note::new("Be brief.")).unwrap();
    workflow
        .post_note(Note::new("Typed meanwhile.").once())
        .unwrap();
    let second = workflow.continue_agent(&first, "Go on.").await.unwrap();
    let delivered = input_texts(server.bodies().last().unwrap()).join("\n");
    assert!(delivered.contains("Be brief.") && delivered.contains("Typed meanwhile."));

    workflow
        .continue_agent_with(
            &second,
            "Go on, please.",
            ContinueOptions::new().cut_back_to(2),
        )
        .await
        .unwrap();
    let sent = input_texts(server.bodies().last().unwrap()).join("\n");
    assert_eq!(sent.matches("Be brief.").count(), 1, "{sent}");
    assert!(!sent.contains("Typed meanwhile."), "{sent}");
    assert!(!sent.contains("Go on.\n"), "{sent}");
}

/// A branch is a run of its own: imported, it continues from the copied
/// conversation, and the original goes on with all of its own.
#[tokio::test]
async fn a_branch_goes_on_apart_from_the_original() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let original = workflow.continue_agent(&first, "Plan A.").await.unwrap();

    let branch = workflow
        .import_agent_run("main", original.task.branch(2).unwrap())
        .unwrap();
    assert_ne!(branch.agent_run_id, original.agent_run_id);
    workflow.continue_agent(&branch, "Plan B.").await.unwrap();
    let sent = input_texts(server.bodies().last().unwrap());
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert!(sent[2].starts_with("Plan B."));

    workflow
        .continue_agent(&original, "And then?")
        .await
        .unwrap();
    let sent = input_texts(server.bodies().last().unwrap()).join("\n");
    assert!(
        sent.contains("Plan A.") && !sent.contains("Plan B."),
        "{sent}"
    );

    // What can't be imported.
    let error = |result: Result<AgentResult, WorkflowError>| result.unwrap_err().to_string();
    assert!(error(workflow.import_agent_run("main", original.task.clone())).contains("already"));
    assert!(matches!(
        workflow.import_agent_run("ghost", original.task.branch(2).unwrap()),
        Err(WorkflowError::AgentNotFound(_))
    ));
    let mut running = original.task.branch(2).unwrap();
    running.set_state(arko_agents::TaskState::InProgress);
    assert!(error(workflow.import_agent_run("main", running)).contains("only a completed task"));
}

/// A cut back is reported as the run goes on: `ConversationCutBack` names
/// the message and how many went, before the run's own events.
#[tokio::test]
async fn a_cut_back_is_reported_in_the_events() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let second = workflow
        .continue_agent(&first, "Tell me a joke.")
        .await
        .unwrap();
    let mut events = workflow.subscribe_events();
    workflow
        .continue_agent_with(&second, "A riddle.", ContinueOptions::new().cut_back_to(2))
        .await
        .unwrap();
    let mut seen = Vec::new();
    while let Ok(event) = events.try_recv() {
        seen.push(event.event);
    }
    let cut = seen
        .iter()
        .position(|event| {
            matches!(
                event,
                arko_agents::AgentEvent::ConversationCutBack {
                    message_index: 2,
                    removed_messages: 2
                }
            )
        })
        .unwrap_or_else(|| panic!("{seen:#?}"));
    assert_eq!(cut, 0, "before the run's own events");
}

/// A run cut back no longer holds what it read in the removed messages: it
/// isn't told what changed since then (it reads again what it needs).
#[tokio::test]
async fn a_cut_back_forgets_what_the_removed_messages_read() {
    use crate::freshness::{ActorId, ContextFreshnessConfig, ResourceId};
    use crate::test_support::{complete_turn, tool_call_turn};
    let workspace = Workspace::new();
    std::fs::write(workspace.0.join("a.txt"), "first").unwrap();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn("r", "read_file", serde_json::json!({"path": "a.txt"})),
            _ => complete_turn(&format!("done{index}"), "done"),
        })
    })
    .await;
    let mut reader = agent_config("Reader");
    reader
        .tool_config
        .available_tool_names
        .push("read_file".into());
    let workflow = Workflow::builder()
        .name("cut-back-freshness-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("reader", reader)
        .context_freshness(ContextFreshnessConfig::default())
        .build()
        .unwrap();
    let first = workflow.run_agent("reader", "Read a.txt.").await.unwrap();
    let file = arko_agents::utils::resolve_path("a.txt", &workflow.context().workspace).unwrap();
    workflow
        .mark_resource_dirty(
            ActorId::new(uuid::Uuid::new_v4(), "writer"),
            ResourceId::File(file),
        )
        .await
        .unwrap();
    workflow
        .continue_agent_with(&first, "Start over.", ContinueOptions::new().cut_back_to(0))
        .await
        .unwrap();
    let last = server.bodies().last().unwrap().clone();
    assert!(last.contains("Start over."), "{last}");
    assert!(!last.contains("Changed since you last read them"), "{last}");
}

/// A run can go on in another directory: it works there from then
/// on, and its prompt is written again. A Claude Code run with a session
/// can't move (Claude Code finds a session only from its directory).
#[tokio::test]
async fn a_run_goes_on_in_another_directory() {
    let workspace = Workspace::new();
    std::fs::create_dir(workspace.0.join("project")).unwrap();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let moved = workflow
        .continue_agent_with(
            &first,
            "Now the project.",
            ContinueOptions::new().workspace("project"),
        )
        .await
        .unwrap();
    assert_eq!(
        moved.task.current_working_directory,
        std::path::absolute(workspace.0.join("project")).unwrap()
    );
    let refused = workflow
        .continue_agent_with(&moved, "x", ContinueOptions::new().workspace("missing"))
        .await
        .unwrap_err();
    assert!(refused.to_string().contains("does not exist"), "{refused}");

    let mut in_claude_code = moved.clone();
    in_claude_code.task.agent_config.llm_config.model =
        arko_ai::Model::claude_code(arko_ai::Model::anthropic_claude_opus_5_5());
    in_claude_code.task.claude_code_session = Some("s1".into());
    let refused = workflow
        .continue_agent_with(&in_claude_code, "x", ContinueOptions::new().workspace("."))
        .await
        .unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("stays in the directory its session started in"),
        "{refused}"
    );
}
