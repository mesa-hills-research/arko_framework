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
    assert_eq!(second.task().unwrap().user_turns()[1].index, 2);

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
    assert_eq!(edited.task().unwrap().conversation.messages.len(), 4);

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
        .import_agent_run("main", original.task().unwrap().branch(2).unwrap())
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
    assert!(
        error(workflow.import_agent_run("main", original.task().unwrap().clone()))
            .contains("already")
    );
    assert!(matches!(
        workflow.import_agent_run("ghost", original.task().unwrap().branch(2).unwrap()),
        Err(WorkflowError::AgentNotFound(_))
    ));
    let mut running = original.task().unwrap().branch(2).unwrap();
    running.set_state(arko_agents::TaskState::InProgress);
    assert!(error(workflow.import_agent_run("main", running)).contains("only a completed task"));
}

/// A branch at the first message has no conversation: imported and
/// continued cut back to its first message, it starts over from the new
/// prompt as a run of its own, and the original keeps its conversation.
#[tokio::test]
async fn a_branch_at_the_first_message_starts_over_from_the_next() {
    let workspace = Workspace::new();
    let server = server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let original = workflow.continue_agent(&first, "Plan A.").await.unwrap();

    let branch = workflow
        .import_agent_run("main", original.task().unwrap().branch(0).unwrap())
        .unwrap();
    assert!(branch.task().unwrap().conversation.messages.is_empty());
    let answered = workflow
        .continue_agent_with(
            &branch,
            "Good evening.",
            ContinueOptions::new().cut_back_to(0),
        )
        .await
        .unwrap();
    assert_eq!(answered.agent_run_id, branch.agent_run_id);
    let sent = input_texts(server.bodies().last().unwrap());
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert!(sent[0].starts_with("Good evening."));
    assert_eq!(answered.task().unwrap().conversation.messages.len(), 2);
    assert_eq!(original.task().unwrap().conversation.messages.len(), 4);
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
/// on, and its prompt is written again.
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
        moved.task().unwrap().current_working_directory,
        std::path::absolute(workspace.0.join("project")).unwrap()
    );
    let refused = workflow
        .continue_agent_with(&moved, "x", ContinueOptions::new().workspace("missing"))
        .await
        .unwrap_err();
    assert!(refused.to_string().contains("does not exist"), "{refused}");
}

/// A Claude Code run moves too: its tools work in the new directory, while
/// the CLI resumes its session in the directory the session started in,
/// where Claude Code finds it, with the new directory open to it and the
/// new prompt.
#[tokio::test]
async fn a_claude_code_run_moves_while_its_session_stays() {
    use super::recovery_tests::FakeClaude;
    let fake = FakeClaude::new(serde_json::json!([
        [{"tool": "mcp__arko__complete_task", "input": {"summary": "Hello."}}],
        [
            {"tool": "mcp__arko__read_file", "input": {"path": "only-here.txt"}},
            {"tool": "mcp__arko__complete_task", "input": {"summary": "Read it."}},
        ],
    ]));
    let workspace = Workspace::new();
    let project = workspace.0.join("project");
    std::fs::create_dir(&project).unwrap();
    std::fs::write(project.join("only-here.txt"), "the project's own file").unwrap();
    let mut agent = fake.agent();
    agent.tool_config.available_tool_names = vec!["read_file".into()];
    let workflow = Workflow::builder()
        .name("cut-back-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(arko_ai::LLMClient::new()))
        .agent("main", agent)
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let moved = workflow
        .continue_agent_with(
            &first,
            "Now read the project's file.",
            ContinueOptions::new().workspace("project"),
        )
        .await
        .unwrap();
    assert_eq!(moved.summary, "Read it.");
    let project = std::path::absolute(&project).unwrap();
    assert_eq!(moved.task().unwrap().current_working_directory, project);
    assert!(
        moved
            .task()
            .unwrap()
            .conversation
            .messages
            .iter()
            .any(|message| {
                message.content.iter().any(|content| {
                    matches!(content, arko_ai::Content::ToolResult(result)
                if result.output.contains("the project's own file"))
                })
            })
    );
    let launches = fake.logged("argv");
    let canonical = |path: &str| std::fs::canonicalize(path).unwrap();
    let original = std::fs::canonicalize(&workspace.0).unwrap();
    assert_eq!(canonical(launches[0]["cwd"].as_str().unwrap()), original);
    assert_eq!(canonical(launches[1]["cwd"].as_str().unwrap()), original);
    let argv: Vec<String> = serde_json::from_value(launches[1]["argv"].clone()).unwrap();
    let after = |flag: &str| {
        argv.iter()
            .position(|arg| arg == flag)
            .map(|at| argv[at + 1].clone())
    };
    assert!(
        argv.iter().any(|arg| arg.starts_with("--resume=")),
        "{argv:?}"
    );
    assert_eq!(after("--add-dir"), Some(project.display().to_string()));
    assert_eq!(after("--system-prompt-snapshot").as_deref(), Some("off"));
}
