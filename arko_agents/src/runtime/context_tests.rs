use super::image_tests::{call, harness, response};
use super::*;
use crate::agent_config::{EnvironmentDetailPlacement, FileTreeMode};
use crate::tool::{Tool, ToolError, ToolTypeDetails};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("arko-context-test-{}", Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("original.txt"), "original").unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Default)]
struct ContextTool {
    resets: AtomicUsize,
    observations: AtomicUsize,
    observed_results: AtomicUsize,
}
#[async_trait]
impl Tool for ContextTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_fixture".into(),
            description: "Update a fixture".into(),
            parameters: vec![],
            output_based_example: None,
            supports_batched_calls: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }
    fn allow_result_interception(&self) -> bool {
        false
    }
    fn reset_context(&self, _: Uuid) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
    fn observe_context(&self, _: Uuid, message: &ai::Message) {
        self.observations.fetch_add(1, Ordering::SeqCst);
        for content in &message.content {
            let delivered = match content {
                ai::Content::ToolResult(result) => {
                    result.name == "context_fixture" && result.output == "exact fixture result"
                }
                ai::Content::Text(text) => text.contains("exact fixture result"),
                _ => false,
            };
            if delivered {
                self.observed_results.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        std::fs::write(
            context.current_working_directory.join("created.txt"),
            "created",
        )
        .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        Ok("exact fixture result".into())
    }
}

struct Replace;
#[async_trait]
impl crate::tool::ToolResultInterceptor for Replace {
    async fn intercept(
        &self,
        _: &str,
        _: &str,
        _: &HashMap<String, Value>,
        _: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        Ok(InterceptedResult::Replaced("distilled".into()))
    }
}

fn install(agent: &mut AgentLoop, fixture: &Fixture) -> Arc<ContextTool> {
    let tool = Arc::new(ContextTool::default());
    let mut registry = ToolRegistry::new();
    registry.register(tool.clone());
    agent.tool_registry = Arc::new(registry);
    agent.task.current_working_directory = fixture.0.clone();
    agent
        .task
        .agent_config
        .tool_config
        .available_tool_names
        .push("context_fixture".into());
    agent
        .task
        .agent_config
        .environment_details_config
        .file_tree
        .include = true;
    agent.interceptor_config = Some(InterceptorConfig {
        interceptor: Arc::new(Replace),
        failure_policy: Default::default(),
    });
    tool
}

#[tokio::test]
async fn exact_results_bypass_distillation_and_are_observed_only_after_delivery_in_both_modes() {
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let fixture = Fixture::new();
        let (mut agent, _events, _commands) = harness(mode.clone());
        let tool = install(&mut agent, &fixture);
        let text_result = if mode == ToolCallingMode::Native {
            let result = agent
                .execute_native_tool(&call("one", "context_fixture"), Uuid::new_v4())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result.output, "exact fixture result");
            agent.task.pending_native_tool_results.push(result);
            String::new()
        } else {
            agent
                .execute_tool_and_handle_result(
                    "context_fixture",
                    HashMap::new(),
                    Uuid::new_v4(),
                    None,
                    None,
                )
                .await
                .unwrap()
        };
        agent.observe_committed_tool_context();
        assert_eq!(tool.observed_results.load(Ordering::SeqCst), 0);
        if mode == ToolCallingMode::Native {
            agent.flush_native_results_to_conversation(None);
        } else {
            assert!(text_result.contains("exact fixture result"));
            agent.add_text_tool_results_to_conversation(&text_result);
        }
        agent.observe_committed_tool_context();
        assert_eq!(tool.observed_results.load(Ordering::SeqCst), 1);
        agent.observe_committed_tool_context();
        assert_eq!(tool.observed_results.load(Ordering::SeqCst), 1);
        agent.task.conversation.messages.clear();
        agent.observe_committed_tool_context();
        assert_eq!(tool.resets.load(Ordering::SeqCst), 1);
        // Ordinary tools still use the configured interceptor.
        assert_eq!(
            agent
                .apply_tool_result_interception("ordinary", "raw", &HashMap::new())
                .await,
            ("distilled".into(), true)
        );
        assert_eq!(
            agent
                .apply_tool_result_interception("list_files", "full tree", &HashMap::new())
                .await,
            ("full tree".into(), false)
        );
    }
}

#[tokio::test]
async fn native_model_boundary_includes_initial_tree_and_changes_before_the_next_response() {
    let fixture = Fixture::new();
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    let tool = install(&mut agent, &fixture);
    let mut update = ai::Message::new_assistant("Update the fixture.");
    update.tool_calls.push(call("update", "context_fixture"));
    let mut done = ai::Message::new_assistant("Done.");
    done.tool_calls.push(ai::ToolCall {
        arguments: serde_json::json!({"summary":"done"}),
        ..call("done", "complete_task")
    });
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([response(update), response(done)]);
    let task = timeout(Duration::from_secs(2), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    let messages = task
        .conversation
        .messages
        .iter()
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(ai::Content::as_text)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>();
    let full = messages
        .iter()
        .position(|text| text.contains("Full snapshot:"))
        .unwrap();
    let change = messages
        .iter()
        .position(|text| text.contains("Added to tree: \"created.txt\""))
        .unwrap();
    assert!(full < change);
    assert!(messages[full].contains("original.txt"));
    assert!(!messages[change].contains("original.txt"));
    assert!(
        task.conversation.messages[change + 1]
            .tool_calls
            .iter()
            .any(|call| call.name == "complete_task")
    );
    assert_eq!(tool.observed_results.load(Ordering::SeqCst), 1);
    // A newly resumed runtime invalidates the tool's delivery cache.
    let registry = {
        let mut registry = ToolRegistry::new();
        registry.register(tool.clone());
        Arc::new(registry)
    };
    let (events, _receiver) = mpsc::channel(100);
    let (_sender, commands) = mpsc::channel(10);
    let mut resumed = AgentLoop::new(
        task,
        Arc::new(ai::LLMClient::new()),
        registry,
        events,
        commands,
        None,
        false,
        None,
        None,
        None,
    );
    assert_eq!(tool.resets.load(Ordering::SeqCst), 1);
    resumed.refresh_file_tree_context();
    assert!(
        resumed
            .task
            .conversation
            .messages
            .last()
            .unwrap()
            .content
            .iter()
            .filter_map(ai::Content::as_text)
            .any(|text| text.contains("Full snapshot:"))
    );
}

#[test]
fn file_tree_errors_preserve_baselines_and_full_mode_preserves_placement() {
    let fixture = Fixture::new();
    let (mut agent, _events, _commands) = harness(ToolCallingMode::OutputBased);
    install(&mut agent, &fixture);
    agent.refresh_file_tree_context();
    assert_eq!(agent.task.conversation.messages.len(), 1);
    agent.refresh_file_tree_context();
    assert_eq!(agent.task.conversation.messages.len(), 1);
    agent.task.current_working_directory = fixture.0.join("missing");
    agent.refresh_file_tree_context();
    assert_eq!(agent.task.conversation.messages.len(), 2);
    assert!(agent.file_tree_baseline.is_some());
    agent.refresh_file_tree_context();
    assert_eq!(agent.task.conversation.messages.len(), 2);
    agent.task.current_working_directory = fixture.0.clone();
    std::fs::remove_file(fixture.0.join("original.txt")).unwrap();
    agent.refresh_file_tree_context();
    assert!(
        agent.task.conversation.messages.last().unwrap().content[0]
            .as_text()
            .unwrap()
            .contains("Removed from tree: \"original.txt\"")
    );
    std::fs::write(fixture.0.join("visible.txt"), "x").unwrap();
    for placement in [
        EnvironmentDetailPlacement::SystemPrompt,
        EnvironmentDetailPlacement::InitialTaskPrompt,
        EnvironmentDetailPlacement::EachUserMessage,
    ] {
        agent
            .task
            .agent_config
            .environment_details_config
            .file_tree_mode = FileTreeMode::Full;
        agent
            .task
            .agent_config
            .environment_details_config
            .file_tree
            .placement = placement.clone();
        let config = &agent.task.agent_config;
        let builder = PromptBuilder::new(config, vec![], &agent.parsing_config, None);
        let system = builder.build_system_prompt(&agent.task).unwrap();
        let initial = builder
            .format_initial_task_prompt(&agent.task, "start")
            .unwrap();
        let user = builder
            .augment_user_message_with_env_details_and_reminder(&agent.task, "next", 2)
            .unwrap();
        assert_eq!(
            system.contains("visible.txt"),
            placement == EnvironmentDetailPlacement::SystemPrompt
        );
        assert_eq!(
            initial.contains("visible.txt"),
            placement != EnvironmentDetailPlacement::SystemPrompt
        );
        assert_eq!(
            user.contains("visible.txt"),
            placement == EnvironmentDetailPlacement::EachUserMessage
        );
    }
    let before = agent.task.conversation.messages.len();
    agent.refresh_file_tree_context();
    assert_eq!(agent.task.conversation.messages.len(), before);
    let legacy: crate::agent_config::EnvironmentDetailsConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(legacy.file_tree_mode, FileTreeMode::Diff);
}

#[tokio::test]
async fn text_model_boundary_includes_file_changes_and_observes_the_exact_result() {
    use futures::StreamExt;
    let fixture = Fixture::new();
    let (mut agent, mut events, _commands) = harness(ToolCallingMode::OutputBased);
    let tool = install(&mut agent, &fixture);
    for text in [
        "<context_fixture></context_fixture>",
        "<complete_task><summary>done</summary></complete_task>",
    ] {
        let end = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
            message: ai::Message::new_assistant(text),
            usage: ai::UsageInfo::new(0, 0),
            model_call_usage: ai::ModelCallUsage::new(
                "fixture".into(),
                "test".into(),
                "test".into(),
                0,
                0,
                1000,
            ),
            stop_reason: ai::StopReason::EndTurn,
            refusal: None,
            openai_turn_state: None,
        }));
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .push_back(Ok(futures::stream::iter([
                Ok(ai::StreamEvent::TextChunk(text.into())),
                Ok(end),
            ])
            .boxed()));
    }
    let outcome = timeout(Duration::from_secs(2), agent.run()).await;
    let mut emitted = Vec::new();
    while let Ok(event) = events.try_recv() {
        emitted.push(event);
    }
    let task = outcome
        .unwrap_or_else(|error| panic!("{error}: {emitted:#?}"))
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(tool.observed_results.load(Ordering::SeqCst), 1);
    let trees = task
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_text)
        .filter(|text| text.starts_with("# File tree"))
        .collect::<Vec<_>>();
    assert_eq!(trees.len(), 2);
    assert!(trees[0].contains("Full snapshot:"));
    assert!(trees[1].contains("Added to tree: \"created.txt\""));
    assert!(!trees[1].contains("original.txt"));
}

/// A dynamic environment detail whose provider was not set again (a saved
/// task) stops the task at its start with an error naming the detail and
/// its provider, instead of sending the model an error string.
#[tokio::test]
async fn a_dynamic_detail_without_its_provider_fails_the_task_at_start() {
    use crate::agent_config::DeveloperDefinableEnvironmentDetail;
    let detail = |placement| {
        DeveloperDefinableEnvironmentDetail::new_dynamic_placeholder(
            "Build status".into(),
            "ci_status".into(),
            placement,
        )
    };
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.task.developer_env_details = Some(vec![detail(EnvironmentDetailPlacement::SystemPrompt)]);
    let error = agent.run().await.unwrap_err().to_string();
    assert!(
        error.contains("'Build status'") && error.contains("'ci_status'"),
        "{error}"
    );

    // Once the prompts are written, only details sent with every user
    // message are still needed.
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.task.conversation.system_message = Some("Written at the start.".into());
    agent
        .task
        .conversation
        .add_user_message("Inspect fixture image");
    agent.task.developer_env_details = Some(vec![detail(EnvironmentDetailPlacement::SystemPrompt)]);
    let mut done = ai::Message::new_assistant("");
    done.tool_calls.push(ai::ToolCall {
        arguments: serde_json::json!({"summary": "done"}),
        ..call("done", "complete_task")
    });
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .push_back(response(done));
    let task = timeout(Duration::from_secs(5), agent.run())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(task.current_state, TaskState::Completed { .. }));

    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.task.conversation.system_message = Some("Written at the start.".into());
    agent.task.developer_env_details =
        Some(vec![detail(EnvironmentDetailPlacement::EachUserMessage)]);
    assert!(agent.run().await.is_err());
}

/// The date and time in the system prompt is labelled as the task's start;
/// elsewhere it is the current one.
#[test]
fn the_system_prompt_datetime_is_labelled_as_the_task_start() {
    for (placement, label) in [
        (
            EnvironmentDetailPlacement::SystemPrompt,
            "Task Start Datetime (UTC):",
        ),
        (
            EnvironmentDetailPlacement::InitialTaskPrompt,
            "Current Datetime (UTC):",
        ),
    ] {
        let mut config = super::tests::test_config();
        config.environment_details_config.current_datetime =
            crate::agent_config::BuiltInEnvironmentDetail {
                include: true,
                placement: placement.clone(),
                depth: None,
            };
        let task = Task::new(config.clone(), "task".into(), std::env::temp_dir(), None);
        let parsing = config.get_tool_parsing_config();
        let builder = PromptBuilder::new(&config, vec![], &parsing, None);
        let prompt = if placement == EnvironmentDetailPlacement::SystemPrompt {
            builder.build_system_prompt(&task).unwrap()
        } else {
            builder.format_initial_task_prompt(&task, "task").unwrap()
        };
        assert!(prompt.contains(label), "{prompt}");
        assert_eq!(prompt.matches("Datetime (UTC)").count(), 1, "{prompt}");
    }
}
