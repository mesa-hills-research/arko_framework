use super::*;
use crate::tool::{Tool, ToolError, ToolTypeDetails};
use async_trait::async_trait;
use futures::StreamExt;

const IMAGE_DATA: &str = "cGl4ZWxzLWZpeHR1cmU=";

#[derive(Debug)]
pub(super) struct ImageTool;
#[async_trait]
impl Tool for ImageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "view_image".into(),
            description: "Fixture image tool".into(),
            parameters: vec![],
            output_based_example: None,
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
        panic!("runtime must use structured tool execution")
    }
    async fn execute_with_output(
        &self,
        _: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        if params.get("fail").and_then(Value::as_bool) == Some(true) {
            return Err(ToolError::ExecutionFailed("bad image".into()));
        }
        Ok(ToolOutput {
            text: "Viewed fixture.png, 12 × 8".into(),
            images: vec![ToolImage {
                description: "fixture.png, 12 × 8".into(),
                image: ai::Image::new_base64(IMAGE_DATA, "image/png"),
            }],
        })
    }
}

#[derive(Debug)]
pub(super) struct TextTool;
#[async_trait]
impl Tool for TextTool {
    fn definition(&self) -> ToolDefinition {
        let mut definition = ImageTool.definition();
        definition.name = "plain_text".into();
        definition
    }
    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        Ok("Legacy text output".into())
    }
}

pub(super) fn harness(
    mode: ToolCallingMode,
) -> (
    AgentLoop,
    mpsc::Receiver<AgentEvent>,
    mpsc::Sender<UserCommand>,
) {
    let mut config = super::tests::test_config();
    config.tool_config.tool_calling_mode = mode;
    config.tool_config.default_auto_approve_tool_use = true;
    config.tool_config.available_tool_names = vec![
        "view_image".into(),
        "plain_text".into(),
        "complete_task".into(),
    ];
    let mut task = Task::new(
        config,
        "Inspect fixture image".into(),
        std::env::temp_dir(),
        None,
    );
    task.set_state(TaskState::InProgress);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ImageTool));
    registry.register(Arc::new(TextTool));
    let (events, receiver) = mpsc::channel(200);
    let (sender, commands) = mpsc::channel(10);
    (
        AgentLoop::new(
            task,
            Arc::new(ai::LLMClient::new()),
            Arc::new(registry),
            events,
            commands,
            None,
            false,
            None,
            None,
            None,
        ),
        receiver,
        sender,
    )
}

pub(super) fn call(id: &str, name: &str) -> ai::ToolCall {
    ai::ToolCall {
        id: Some(id.into()),
        name: name.into(),
        arguments: serde_json::json!({}),
        thought_signature: None,
        provider_id: None,
        invalid_arguments: None,
    }
}

fn assert_image(content: &ai::Content) {
    assert!(
        matches!(content, ai::Content::Image(ai::Image::Base64 { data, mime_type }) if data == IMAGE_DATA && mime_type == "image/png")
    );
}

fn assert_result_image(result: &ai::ToolResult) {
    assert_eq!(result.images.len(), 1, "{result:?}");
    assert_eq!(result.images[0].label, "fixture.png, 12 × 8");
    assert!(
        matches!(&result.images[0].image, ai::Image::Base64 { data, mime_type } if data == IMAGE_DATA && mime_type == "image/png")
    );
}

/// A native tool's images go inside its own result, labelled with the
/// tool's description (arko_ai adds the "Image from tool ..." prefix where a
/// model takes images only after the results). They wait with the results
/// for the rest of the turn, survive saving, and never reach events.
#[tokio::test]
async fn a_native_tools_images_go_inside_its_own_result() {
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::Native);
    for (id, name) in [
        ("one", "view_image"),
        ("two", "plain_text"),
        ("three", "view_image"),
    ] {
        let result = agent
            .execute_native_tool(&call(id, name), Uuid::new_v4())
            .await
            .unwrap()
            .unwrap();
        agent.task.pending_native_tool_results.push(result);
    }
    let pending_id = Uuid::new_v4();
    agent
        .task
        .pending_native_tool_calls
        .insert(pending_id, call("four", "plain_text"));
    assert!(agent.task.pending_tool_images.is_empty());
    assert!(agent.task.conversation.messages.is_empty());
    let path = std::env::temp_dir().join(format!("arko-image-state-{}.json", Uuid::new_v4()));
    agent.task.save_to_path(&path).unwrap();
    let restored = Task::load_from_path(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let (mut resumed, _, _commands2) = harness(ToolCallingMode::Native);
    resumed.task = restored;
    assert_eq!(
        resumed.task.pending_native_tool_calls[&pending_id]
            .id
            .as_deref(),
        Some("four")
    );
    resumed
        .flush_native_results_to_conversation(Some("Review both images."))
        .await
        .unwrap();
    let message = &resumed.task.conversation.messages[0];
    let contents = &message.content;
    assert_eq!(contents.len(), 4, "{contents:?}");
    for (index, expected) in ["one", "two", "three"].into_iter().enumerate() {
        assert_eq!(
            contents[index].as_tool_result().unwrap().call_id.as_deref(),
            Some(expected)
        );
    }
    assert_result_image(contents[0].as_tool_result().unwrap());
    assert!(contents[1].as_tool_result().unwrap().images.is_empty());
    assert_result_image(contents[2].as_tool_result().unwrap());
    assert_eq!(contents[3].as_text(), Some("Review both images."));
    assert_eq!(message.image_count(), 2);
    assert_eq!(resumed.task.user_message_count, 1);
    let stored = serde_json::to_string(&resumed.task).unwrap();
    let restored: Task = serde_json::from_str(&stored).unwrap();
    assert_result_image(
        restored.conversation.messages[0].content[0]
            .as_tool_result()
            .unwrap(),
    );
    while let Ok(event) = events.try_recv() {
        assert!(!serde_json::to_string(&event).unwrap().contains(IMAGE_DATA));
    }
}

/// A task saved mid-turn by an earlier version, with tool images waiting
/// beside the results (`pending_tool_images`), still loads and sends them as
/// it did: after the results and their text, each after its description.
#[tokio::test]
async fn an_older_task_with_images_beside_its_results_still_sends_them() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent
        .task
        .pending_native_tool_results
        .push(ai::ToolResult::from_tool_call(
            &call("one", "view_image"),
            "Viewed fixture.png, 12 × 8",
        ));
    let mut saved = serde_json::to_value(&agent.task).unwrap();
    saved["pending_tool_images"] = serde_json::json!([{
        "description": "Image from tool view_image, call one: fixture.png, 12 × 8",
        "image": serde_json::to_value(ai::Image::new_base64(IMAGE_DATA, "image/png")).unwrap(),
    }]);
    let (mut resumed, _, _commands2) = harness(ToolCallingMode::Native);
    resumed.task = serde_json::from_value(saved).unwrap();
    assert_eq!(resumed.task.pending_tool_images.len(), 1);
    resumed
        .flush_native_results_to_conversation(Some("Review it."))
        .await
        .unwrap();
    let contents = &resumed.task.conversation.messages[0].content;
    assert_eq!(contents.len(), 4, "{contents:?}");
    assert!(contents[0].as_tool_result().unwrap().images.is_empty());
    assert_eq!(contents[1].as_text(), Some("Review it."));
    assert_eq!(
        contents[2].as_text(),
        Some("Image from tool view_image, call one: fixture.png, 12 × 8")
    );
    assert_image(&contents[3]);
    assert!(resumed.task.pending_tool_images.is_empty());
}

#[tokio::test]
async fn text_mode_approval_errors_and_legacy_tools_preserve_their_semantics() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::OutputBased);
    let text = agent
        .execute_text_call(
            "view_image",
            HashMap::new(),
            Uuid::new_v4(),
            None,
            None,
            false,
        )
        .await
        .unwrap();
    assert!(text.contains("fixture.png"));
    assert!(!text.contains(IMAGE_DATA));
    agent.add_text_tool_results_to_conversation(&text);
    assert_image(&agent.task.conversation.messages[0].content[2]);
    let text = agent
        .execute_text_call(
            "plain_text",
            HashMap::new(),
            Uuid::new_v4(),
            None,
            None,
            false,
        )
        .await
        .unwrap();
    assert!(text.contains("Legacy text output"));
    assert!(agent.task.pending_tool_images.is_empty());
    let failed = agent
        .execute_native_tool(
            &ai::ToolCall {
                arguments: serde_json::json!({"fail":true}),
                ..call("bad", "view_image")
            },
            Uuid::new_v4(),
        )
        .await
        .unwrap()
        .unwrap();
    assert!(failed.is_error);
    assert!(failed.images.is_empty());
    assert!(agent.task.pending_tool_images.is_empty());
    // An approved call's images go inside its result too.
    let result = agent
        .execute_native_tool_for_approval(
            &call("approved", "view_image"),
            Uuid::new_v4(),
            "view_image",
            HashMap::new(),
            None,
            Some("Approved view".into()),
        )
        .await
        .unwrap()
        .unwrap();
    assert!(result.output.contains("Approved view"));
    assert_result_image(&result);
    assert!(agent.task.pending_tool_images.is_empty());
    agent.task.pending_native_tool_results.push(result);
    agent
        .flush_native_results_to_conversation(None)
        .await
        .unwrap();
    // Output the host supplies has no images.
    let custom = agent
        .execute_native_tool_for_approval(
            &call("custom", "view_image"),
            Uuid::new_v4(),
            "view_image",
            HashMap::new(),
            Some("Caller provided text".into()),
            None,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(custom.images.is_empty());
    assert!(agent.task.pending_tool_images.is_empty());
}

#[tokio::test]
async fn a_model_without_image_input_gets_the_tools_text_and_an_omission_note() {
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::Native);
    agent.task.conversation.model.supports_image_input = false;
    let result = agent
        .execute_native_tool(&call("image", "view_image"), Uuid::new_v4())
        .await
        .unwrap()
        .unwrap();
    assert!(!result.is_error);
    assert!(result.images.is_empty());
    assert!(result.output.contains("Viewed fixture.png"));
    assert!(result.output.contains("doesn't take images"));
    while let Ok(event) = events.try_recv() {
        assert!(!serde_json::to_string(&event).unwrap().contains(IMAGE_DATA));
    }

    let (mut agent, _events, _commands) = harness(ToolCallingMode::OutputBased);
    agent.task.conversation.model.supports_image_input = false;
    let text = agent
        .execute_text_call(
            "view_image",
            HashMap::new(),
            Uuid::new_v4(),
            None,
            None,
            false,
        )
        .await
        .unwrap();
    assert!(text.contains("doesn't take images"));
    assert!(agent.task.pending_tool_images.is_empty());
}

/// A PNG of `width` × `height` pixels.
fn png(width: u32, height: u32) -> ai::Image {
    let pixels = image::RgbaImage::from_pixel(width, height, image::Rgba([40, 90, 200, 255]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(pixels)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    ai::Image::from_bytes(&bytes, "image/png")
}

/// Images over the model's limits are shrunk to fit them. One that can't be
/// read to shrink it, or that the model can't take (a URL on Anthropic), is
/// left out with a line saying why, without losing the others or the text.
#[test]
fn oversized_tool_images_are_shrunk_or_left_out_without_losing_the_rest() {
    let mut output = ToolOutput {
        text: "Four screenshots".into(),
        images: vec![
            ToolImage {
                description: "small".into(),
                image: png(20, 10),
            },
            ToolImage {
                description: "wide".into(),
                image: png(3_000, 30),
            },
            ToolImage {
                description: "unreadable".into(),
                image: ai::Image::from_bytes(&vec![0; 6 * 1024 * 1024], "image/png"),
            },
            ToolImage {
                description: "link".into(),
                image: ai::Image::new_url("https://example.test/image.png"),
            },
        ],
    };
    output.admit_images(&ai::Model::anthropic_claude_opus_5_5());
    let kept: Vec<_> = output
        .images
        .iter()
        .map(|image| (image.description.as_str(), image.image.dimensions()))
        .collect();
    assert_eq!(
        kept,
        [("small", Some((20, 10))), ("wide", Some((2_000, 20)))]
    );
    assert!(output.text.starts_with("Four screenshots"));
    assert!(
        output
            .text
            .contains("image unreadable, left out: its 8.0 MB are over the 5.0 MB"),
        "{}",
        output.text
    );
    assert!(
        output
            .text
            .contains("image link, left out: model 'claude-opus-5-5' takes images as data")
    );
}

/// Past the images, or the image data, a request takes, a tool's images are
/// left out, saying how to make room.
#[test]
fn tool_images_past_the_request_count_are_left_out() {
    let mut output = ToolOutput {
        text: "Two screenshots".into(),
        images: vec![
            ToolImage::base64("first", IMAGE_DATA.into(), "image/png"),
            ToolImage::base64("second", IMAGE_DATA.into(), "image/png"),
        ],
    };
    let room = crate::tool::ImageRoom::new(1, usize::MAX);
    output.admit_images_within(&super::tests::test_config().llm_config.model, &room);
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.images[0].description, "first");
    assert!(
        output
            .text
            .contains("image second, which was left out because the conversation holds as many images as a request to this model takes (500)"),
        "{}",
        output.text
    );

    // And past the image data a request takes: the bytes count too.
    let mut output = ToolOutput {
        text: "Two screenshots".into(),
        images: vec![
            ToolImage::base64("first", IMAGE_DATA.into(), "image/png"),
            ToolImage::base64("second", IMAGE_DATA.into(), "image/png"),
        ],
    };
    let room = crate::tool::ImageRoom::new(10, IMAGE_DATA.len() + 5);
    output.admit_images_within(&super::tests::test_config().llm_config.model, &room);
    assert_eq!(output.images.len(), 1);
    assert!(
        output
            .text
            .contains("image second, which was left out because the conversation holds as much image data as a request to this model takes (40 MB)"),
        "{}",
        output.text
    );
}

struct ReplaceText;
#[async_trait]
impl crate::tool::ToolResultInterceptor for ReplaceText {
    async fn intercept(
        &self,
        _: &str,
        _: &str,
        _: &HashMap<String, Value>,
        _: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        Ok(InterceptedResult::Replaced("Short metadata".into()))
    }
}

#[tokio::test]
async fn interception_changes_text_without_dropping_images_and_old_tasks_still_load() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.interceptor_config = Some(InterceptorConfig {
        interceptor: Arc::new(ReplaceText),
        failure_policy: Default::default(),
    });
    let result = agent
        .execute_native_tool(&call("image", "view_image"), Uuid::new_v4())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.output, "Short metadata");
    assert_result_image(&result);
    let mut old = serde_json::to_value(&agent.task).unwrap();
    for name in [
        "pending_tool_images",
        "pending_native_tool_calls",
        "pending_native_tool_results",
    ] {
        old.as_object_mut().unwrap().remove(name);
    }
    let restored: Task = serde_json::from_value(old).unwrap();
    assert!(restored.pending_tool_images.is_empty());
    assert!(restored.pending_native_tool_results.is_empty());
    assert!(restored.pending_native_tool_calls.is_empty());
}

pub(super) fn response(message: ai::Message) -> ScriptedStreamResult {
    let stop_reason = if message.has_tool_calls() {
        ai::StopReason::ToolUse
    } else {
        ai::StopReason::EndTurn
    };
    Ok(
        futures::stream::iter([Ok(ai::StreamEvent::End(Box::new(ai::GenerationResponse {
            message,
            usage: ai::UsageInfo::new(0, 0),
            model_call_usage: ai::ModelCallUsage::new(
                "fixture".into(),
                "test".into(),
                "test".into(),
                0,
                0,
                1000,
            ),
            stop_reason,
            refusal: None,
            openai_turn_state: None,
        })))])
        .boxed(),
    )
}

#[tokio::test]
async fn full_native_loop_sends_image_content_before_the_next_model_turn() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    let mut view = ai::Message::new_assistant("Inspecting the image.");
    view.tool_calls.push(call("image", "view_image"));
    let mut done = ai::Message::new_assistant("Reviewed.");
    done.tool_calls.push(ai::ToolCall {
        arguments: serde_json::json!({"summary":"done"}),
        ..call("done", "complete_task")
    });
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([response(view), response(done)]);
    let task = timeout(Duration::from_secs(2), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let index = task
        .conversation
        .messages
        .iter()
        .position(ai::Message::has_images)
        .unwrap();
    assert!(
        task.conversation.messages[index + 1]
            .tool_calls
            .iter()
            .any(|call| call.name == "complete_task")
    );
    assert_result_image(
        task.conversation.messages[index].content[0]
            .as_tool_result()
            .unwrap(),
    );
}

#[tokio::test]
async fn native_user_answer_is_kept_after_ask_user_results_were_already_flushed() {
    let (mut agent, mut events, commands) = harness(ToolCallingMode::Native);
    agent.task.current_state = TaskState::AwaitingUserResponse {
        questions: "Remove the image browser?".into(),
    };
    let ask = call("question", "ask_user");
    agent
        .task
        .conversation
        .add_tool_results(vec![ai::ToolResult::from_tool_call(
            &ask,
            "Remove the image browser?",
        )]);
    let mut done = ai::Message::new_assistant("Acknowledged.");
    done.tool_calls.push(ai::ToolCall {
        arguments: serde_json::json!({"summary":"done"}),
        ..call("done", "complete_task")
    });
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(response(done));
    commands
        .send(UserCommand::SubmitUserResponse {
            response: "Remove the entire image browser.".into(),
        })
        .await
        .unwrap();
    let task = timeout(Duration::from_secs(2), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(task.conversation.messages[1].content.iter().any(|content| {
        content
            .as_text()
            .is_some_and(|text| text.contains("Remove the entire image browser."))
    }));
    while events.try_recv().is_ok() {}
}

#[test]
fn google_gif_results_are_described_without_blocking_other_images() {
    let mut model = super::tests::test_config().llm_config.model;
    model.provider = ai::Provider::Google;
    model.supports_image_input = true;
    let mut output = ToolOutput {
        text: "Screenshots".into(),
        images: vec![
            ToolImage::base64("animation", "R0lGODlh".into(), "image/gif"),
            ToolImage::base64("still", IMAGE_DATA.into(), "image/png"),
        ],
    };
    let mut other_provider = model.clone();
    other_provider.provider = ai::Provider::Anthropic;
    let mut accepted = output.clone();
    accepted.admit_images(&other_provider);
    assert_eq!(accepted.images.len(), 2);
    output.admit_images(&model);
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.images[0].description, "still");
    assert!(output.text.starts_with("Screenshots"));
    assert!(
        output
            .text
            .contains("doesn't take GIF images. Convert it to PNG or JPEG")
    );
}

/// A sink's report: the task, the call, the tool and the images.
type Reported = (Uuid, Uuid, String, Vec<ToolImage>);

/// Records what a tool image sink is given.
#[derive(Default)]
pub(super) struct ImageRecorder(pub std::sync::Mutex<Vec<Reported>>);

impl crate::tool::ToolImageSink for ImageRecorder {
    fn tool_images(
        &self,
        task_id: Uuid,
        tool_call_id: Uuid,
        tool_name: &str,
        images: &[ToolImage],
    ) {
        self.0.lock().unwrap().push((
            task_id,
            tool_call_id,
            tool_name.to_string(),
            images.to_vec(),
        ));
    }
}

/// The host's sink gets a call's images keyed by the call's runtime ID, as
/// the tool returned them: also when the model can't take them and its
/// result leaves them out. A call without images isn't reported.
#[tokio::test]
async fn the_host_s_sink_gets_each_call_s_images() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.task.conversation.model.supports_image_input = false;
    let recorder = Arc::new(ImageRecorder::default());
    agent.image_sink = Some(recorder.clone());
    let (seen, plain) = (Uuid::new_v4(), Uuid::new_v4());
    let result = agent
        .execute_native_tool(&call("one", "view_image"), seen)
        .await
        .unwrap()
        .unwrap();
    assert!(result.images.is_empty(), "the model takes no images");
    agent
        .execute_native_tool(&call("two", "plain_text"), plain)
        .await
        .unwrap()
        .unwrap();
    let recorded = recorder.0.lock().unwrap();
    let [(task_id, tool_call_id, name, images)] = recorded.as_slice() else {
        panic!("{}", recorded.len());
    };
    assert_eq!((*task_id, *tool_call_id), (agent.task.id, seen));
    assert_eq!(name, "view_image");
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].description, "fixture.png, 12 × 8");
}

/// Records the provider's call IDs a tool image sink is given. It implements
/// only `tool_images_with_call_id`, which the runtime calls.
#[derive(Default)]
struct CallIdRecorder(std::sync::Mutex<Vec<(Uuid, Option<String>, String)>>);

impl crate::tool::ToolImageSink for CallIdRecorder {
    fn tool_images(&self, _: Uuid, _: Uuid, _: &str, _: &[ToolImage]) {
        unreachable!("the runtime gives the provider's call ID too");
    }

    fn tool_images_with_call_id(
        &self,
        _task_id: Uuid,
        tool_call_id: Uuid,
        provider_call_id: Option<&str>,
        tool_name: &str,
        _images: &[ToolImage],
    ) {
        self.0.lock().unwrap().push((
            tool_call_id,
            provider_call_id.map(str::to_string),
            tool_name.to_string(),
        ));
    }
}

/// A sink that implements `tool_images_with_call_id` gets the ID the model's
/// provider gave the call, next to the runtime's. A sink that implements only
/// `tool_images` keeps working (see the test above).
#[tokio::test]
async fn the_host_s_sink_gets_the_providers_call_id() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    let recorder = Arc::new(CallIdRecorder::default());
    agent.image_sink = Some(recorder.clone());
    let runtime_id = Uuid::new_v4();
    agent
        .execute_native_tool(&call("call_abc", "view_image"), runtime_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *recorder.0.lock().unwrap(),
        [(
            runtime_id,
            Some("call_abc".to_string()),
            "view_image".to_string()
        )]
    );
}
