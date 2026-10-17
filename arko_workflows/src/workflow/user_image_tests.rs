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

/// A run's images follow its prompt in its first message, and a
/// continuation's follow its prompt in the message that continues it.
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
    // Held once, in the first message.
    assert!(first.task.initial_task_images.is_empty());
    assert_eq!(first.task.initial_images(), [png("Zmlyc3Q=")]);
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

/// A PNG of `width` × `height` pixels, of one color.
fn sized_png(width: u32, height: u32) -> arko_ai::Image {
    let pixels = image::RgbaImage::from_pixel(width, height, image::Rgba([200, 40, 40, 255]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(pixels)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    arko_ai::Image::from_bytes(&bytes, "image/png")
}

/// An image the agent's model can't take as it is is fitted before it is
/// sent: here a BMP, which OpenAI doesn't take, becomes a PNG.
#[tokio::test]
async fn an_image_the_model_cant_take_as_it_is_is_fitted_before_it_is_sent() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let workflow = workflow(&server, &workspace);
    let mut bmp = Vec::new();
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(40, 30, image::Rgb([9, 9, 9])))
        .write_to(&mut std::io::Cursor::new(&mut bmp), image::ImageFormat::Bmp)
        .unwrap();
    let result = workflow
        .run_agent_with(
            "worker",
            "Describe this.",
            RunOptions::new().images([arko_ai::Image::from_bytes(&bmp, "image/bmp")]),
        )
        .await
        .unwrap();
    let content = last_user_content(&server.bodies()[0]);
    let url = content[1]["image_url"].as_str().unwrap();
    let data = url.strip_prefix("data:image/png;base64,").unwrap();
    assert_eq!(png(data).dimensions(), Some((40, 30)));
    assert_eq!(result.task.initial_images()[0].dimensions(), Some((40, 30)));
}

/// A conversation that holds as many images, or as much image data, as a
/// request to its model takes is sent no more: every later request would
/// fail. The refusal says how to make room.
#[tokio::test]
async fn a_conversation_full_of_images_takes_no_more() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let workflow = workflow(&server, &workspace);
    let mut claude = agent_config("Claude");
    claude.llm_config.model = arko_ai::Model::anthropic_claude_opus_5_5();
    let held = |images: Vec<arko_ai::Image>| {
        let mut task =
            arko_agents::Task::new(claude.clone(), "Look.".into(), workspace.0.clone(), None);
        task.conversation.add_user_message_with_contents(
            images.into_iter().map(arko_ai::Content::Image).collect(),
        );
        task.conversation.add_assistant_message("Seen.");
        task.set_state(arko_agents::TaskState::Completed {
            summary: "Seen.".into(),
        });
        workflow.import_agent_run("claude", task).unwrap()
    };
    let more = || ContinueOptions::new().images([png("eA==")]);

    let full = held(vec![png("eA=="); 100]);
    let error = workflow
        .continue_agent_with(&full, "And this?", more())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("agent 'claude' can't be sent another image. Its conversation holds 100 images, and model 'claude-opus-5-5' takes at most 100 in a request (cut the conversation back or hand it over to make room)"),
        "{error}"
    );

    let heavy = held(vec![png(&"A".repeat(24 * 1024 * 1024))]);
    let error = workflow
        .continue_agent_with(&heavy, "And this?", more())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Its conversation holds 24.0 MB, and a request to model 'claude-opus-5-5' takes at most 24.0 MB"),
        "{error}"
    );
    assert!(server.bodies().is_empty());
}

/// A session whose image file was deleted or damaged after it was saved
/// still resumes: each such image becomes a line saying it is gone, in the
/// message that held it, and the run reports them as it goes on.
#[tokio::test]
async fn a_session_resumes_without_the_image_files_it_lost() {
    for damage in ["deleted", "truncated"] {
        let workspace = Workspace::new();
        let server = done_server().await;
        let run = workspace.0.join(".run");
        let persisted = |run: &std::path::Path| {
            Workflow::builder()
                .name("user-image-test")
                .workspace(&workspace.0)
                .llm_client(Arc::new(server.client()))
                .agent("worker", agent_config("Worker"))
                .persistence(crate::persistence::PersistenceConfig::new(run))
        };
        let first = persisted(&run).build().unwrap();
        let worker = first
            .run_agent_with(
                "worker",
                "Describe this.",
                RunOptions::new().images([sized_png(40, 30)]),
            )
            .await
            .unwrap();
        first.checkpoint().await.unwrap();
        drop(first);
        // A copy of the run directory, as a partial copy of the data would
        // be, its image then lost.
        let copy = workspace.0.join(".copy");
        copy_directory(&run, &copy);
        let run = copy;
        let file = std::fs::read_dir(run.join("images"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        match damage {
            "deleted" => std::fs::remove_file(&file).unwrap(),
            _ => std::fs::write(&file, &std::fs::read(&file).unwrap()[..20]).unwrap(),
        }

        let latest = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
        let resumed = persisted(&run)
            .resume_from(run.join(latest.trim()))
            .build()
            .unwrap();
        let restored = resumed.agent_result(worker.agent_run_id).unwrap();
        let first_message = &restored.task.conversation.messages[0];
        assert!(
            first_message.content.iter().any(|content| matches!(
                content,
                arko_ai::Content::Text(text) if text.starts_with("(image no longer available: ")
            )),
            "{damage}: {first_message:?}"
        );
        assert_eq!(first_message.image_count(), 0, "{damage}");
        let mut events = resumed.subscribe_events_lossless();
        resumed.continue_agent(&restored, "And now?").await.unwrap();
        let mut reported = Vec::new();
        while let Some(event) = events.try_recv() {
            if let arko_agents::AgentEvent::ImagesUnavailable { reasons } = event.event {
                reported.extend(reasons);
            }
        }
        assert_eq!(reported.len(), 1, "{damage}: {reported:?}");
        let expected = if damage == "deleted" {
            "could not read image"
        } else {
            "is damaged"
        };
        assert!(reported[0].contains(expected), "{damage}: {reported:?}");
    }
}

/// Copies directory `from` to `to`, with what it holds.
fn copy_directory(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_directory(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}
