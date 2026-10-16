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
