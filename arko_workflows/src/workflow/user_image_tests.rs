//! A user's images with a run's prompt and a continuation's, and the images
//! an agent's model can't take, refused before anything is sent.

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::test_support::{ResponsesServer, Workspace, agent_config, complete_turn};

fn png(data: &str) -> arko_ai::Image {
    arko_ai::Image::new_base64(data, "image/png")
}

/// The content of a request body's last user message.
fn last_user_content(body: &str) -> Value {
    let body: Value = serde_json::from_str(body).unwrap();
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|item| item["role"] == "user")
        .map(|item| item["content"].clone())
        .unwrap()
}

/// `content` is a text starting with `text`, then the image `data`.
fn assert_text_then_image(content: &Value, text: &str, data: &str) {
    let parts = content.as_array().unwrap_or_else(|| panic!("{content}"));
    assert_eq!(parts.len(), 2, "{content}");
    assert_eq!(parts[0]["type"], "input_text");
    assert!(
        parts[0]["text"].as_str().unwrap().starts_with(text),
        "{content}"
    );
    assert_eq!(
        parts[1],
        json!({"type": "input_image", "image_url": format!("data:image/png;base64,{data}")})
    );
}

async fn done_server() -> ResponsesServer {
    ResponsesServer::start(|index, _| Some(complete_turn(&format!("c{index}"), "done"))).await
}

fn workflow(server: &ResponsesServer, workspace: &Workspace) -> Workflow {
    let mut blind = agent_config("Blind");
    blind.llm_config.model.supports_image_input = false;
    let mut claude = agent_config("Claude");
    claude.llm_config.model = arko_ai::Model::anthropic_claude_opus_5_5();
    Workflow::builder()
        .name("user-image-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("worker", agent_config("Worker"))
        .agent("blind", blind)
        .agent("claude", claude)
        .build()
        .unwrap()
}

/// A run's images follow its prompt in its first message; a continuation's
/// follow its prompt in the message that continues it.
#[tokio::test]
async fn a_run_and_its_continuation_send_their_images_with_their_prompts() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let workflow = workflow(&server, &workspace);
    let first = workflow
        .run_agent_with(
            "worker",
            "Describe this.",
            RunOptions::new().images([png("Zmlyc3Q=")]),
        )
        .await
        .unwrap();
    assert_eq!(first.task.initial_task_images, [png("Zmlyc3Q=")]);
    assert_text_then_image(
        &last_user_content(&server.bodies()[0]),
        "Describe this.",
        "Zmlyc3Q=",
    );

    workflow
        .continue_agent_with(
            &first,
            "And this one?",
            ContinueOptions::new().images([png("c2Vjb25k")]),
        )
        .await
        .unwrap();
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2);
    assert_text_then_image(&last_user_content(&bodies[1]), "And this one?", "c2Vjb25k");
    // The first message still holds its image.
    assert!(bodies[1].contains("Zmlyc3Q="));
}

/// Images the agent's model can't take are refused before a request is
/// made, with the reason: a model without image input, and an image by URL
/// for a model that takes images as data only.
#[tokio::test]
async fn images_a_model_can_t_take_are_refused_before_anything_is_sent() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let workflow = workflow(&server, &workspace);
    let refused = |result: Result<AgentResult, WorkflowError>| match result {
        Err(WorkflowError::BuilderInvalid(message)) => message,
        other => panic!("{other:?}"),
    };
    let message = refused(
        workflow
            .run_agent_with("blind", "Look.", RunOptions::new().images([png("eA==")]))
            .await,
    );
    assert!(
        message.contains(
            "agent 'blind' can't be sent the image: model 'gpt-5.6-luna' doesn't take images"
        ),
        "{message}"
    );
    let url = arko_ai::Image::new_url("https://example.com/a.png");
    let message = refused(
        workflow
            .run_agent_with(
                "claude",
                "Look.",
                RunOptions::new().images([png("eA=="), url]),
            )
            .await,
    );
    assert!(
        message.contains(
            "can't be sent image 2: model 'claude-opus-5-5' takes images as data, not by URL"
        ),
        "{message}"
    );
    assert_eq!(server.requests(), 0);

    let first = workflow.run_agent("blind", "Hello.").await.unwrap();
    let message = refused(
        workflow
            .continue_agent_with(
                &first,
                "Look.",
                ContinueOptions::new().images([png("eA==")]),
            )
            .await,
    );
    assert!(message.contains("doesn't take images"), "{message}");
    assert_eq!(server.requests(), 1);
}

/// Records what the workflow's tool image sink is given.
#[derive(Default)]
struct Images(std::sync::Mutex<Vec<(uuid::Uuid, uuid::Uuid, String)>>);

impl arko_agents::ToolImageSink for Images {
    fn tool_images(
        &self,
        task_id: uuid::Uuid,
        tool_call_id: uuid::Uuid,
        tool_name: &str,
        images: &[arko_agents::ToolImage],
    ) {
        assert_eq!(images.len(), 1);
        self.0
            .lock()
            .unwrap()
            .push((task_id, tool_call_id, tool_name.to_string()));
    }
}

/// The workflow's sink gets the image an agent's `read_file` call shows,
/// by its run and the call's ID in the run's events.
#[tokio::test]
async fn the_workflow_s_sink_gets_the_images_tools_return() {
    use crate::test_support::tool_call_turn;
    let workspace = Workspace::new();
    std::fs::write(
        workspace.0.join("a.png"),
        b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x01\0\0\0\x01\x08\x02\0\0\0",
    )
    .unwrap();
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            0 => tool_call_turn("r0", "read_file", json!({"path": "a.png"})),
            _ => complete_turn("c1", "done"),
        })
    })
    .await;
    let mut reader = agent_config("Reader");
    reader.tool_config.available_tool_names = vec!["read_file".into()];
    let images = Arc::new(Images::default());
    let workflow = Workflow::builder()
        .name("tool-image-sink-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("reader", reader)
        .tool_image_sink(images.clone())
        .build()
        .unwrap();
    let mut events = workflow.subscribe_events();
    let result = workflow
        .run_agent("reader", "Look at a.png.")
        .await
        .unwrap();
    let recorded = images.0.lock().unwrap().clone();
    let [(run, call, name)] = recorded.as_slice() else {
        panic!("{recorded:?}");
    };
    assert_eq!((*run, name.as_str()), (result.agent_run_id, "read_file"));
    let mut formatted = false;
    while let Ok(event) = events.try_recv() {
        if let arko_agents::AgentEvent::ToolCallFormatted { tool_call_id, .. } = event.event {
            formatted |= tool_call_id == *call;
        }
    }
    assert!(formatted);
}
