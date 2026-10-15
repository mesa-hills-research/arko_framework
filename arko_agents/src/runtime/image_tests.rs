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
    assert!(result.output.contains("this model doesn't take images"));
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
    assert!(text.contains("this model doesn't take images"));
    assert!(agent.task.pending_tool_images.is_empty());
}

#[test]
fn oversized_tool_images_are_left_out_without_losing_other_images_or_text() {
    use crate::tool::MAX_TOOL_IMAGE_BYTES;

    let mut output = ToolOutput {
        text: "Two screenshots".into(),
        images: vec![
            ToolImage {
                description: "at limit".into(),
                image: ai::Image::from_bytes(&vec![0; MAX_TOOL_IMAGE_BYTES], "image/png"),
            },
            ToolImage {
                description: "over limit".into(),
                image: ai::Image::from_bytes(&vec![0; MAX_TOOL_IMAGE_BYTES + 1], "image/png"),
            },
            ToolImage {
                description: "link".into(),
                image: ai::Image::new_url("https://example.test/image.png"),
            },
        ],
    };
    output.admit_images(&super::tests::test_config().llm_config.model);
    assert_eq!(output.images.len(), 2);
    assert_eq!(output.images[0].description, "at limit");
    assert_eq!(output.images[1].description, "link");
    assert!(output.text.starts_with("Two screenshots"));
    assert!(output.text.contains("over limit, left out"));
    assert_eq!(output.images[0].decoded_len(), Some(MAX_TOOL_IMAGE_BYTES));
    assert_eq!(output.images[1].decoded_len(), None);
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
    assert!(output.text.contains("Google models don't take GIF images"));
}
