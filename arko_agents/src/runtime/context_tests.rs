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
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
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
                .execute_text_call(
                    "context_fixture",
                    HashMap::new(),
                    Uuid::new_v4(),
                    None,
                    None,
                    false,
                )
                .await
                .unwrap()
        };
        agent.observe_committed_tool_context();
        assert_eq!(tool.observed_results.load(Ordering::SeqCst), 0);
        if mode == ToolCallingMode::Native {
            agent
                .flush_native_results_to_conversation(None)
                .await
                .unwrap();
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
        // Ordinary tools still use the configured interceptor; tools that
        // opt out don't.
        let registry = ToolRegistry::new();
        let read_file = registry.get("read_file").unwrap();
        let list_files = registry.get("list_files").unwrap();
        assert!(
            agent
                .interception_for("read_file", read_file.as_ref())
                .is_some()
        );
        assert!(
            agent
                .interception_for("list_files", list_files.as_ref())
                .is_none()
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

/// In native mode the environment details and reminder due with a user
/// message go with tool results too, as text after the results in the same
/// turn (they used to reach only the runtime's own notes). Without such
/// settings a tool-results message holds only the results.
#[tokio::test]
async fn native_tool_results_carry_the_details_and_reminder_due() {
    use crate::agent_config::{
        DeveloperDefinableEnvironmentDetail, ReminderInjectionFrequency, UserReminder,
    };
    let turn = |id: &str| {
        let mut message = ai::Message::new_assistant("");
        message.tool_calls.push(call(id, "plain_text"));
        response(message)
    };
    let done = || {
        let mut message = ai::Message::new_assistant("");
        message.tool_calls.push(ai::ToolCall {
            arguments: serde_json::json!({"summary": "done"}),
            ..call("done", "complete_task")
        });
        response(message)
    };
    for frequency in [
        None,
        Some(ReminderInjectionFrequency::EveryUserMessage),
        Some(ReminderInjectionFrequency::EveryNUserMessages(2)),
    ] {
        let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
        if let Some(frequency) = frequency.clone() {
            agent.task.agent_config.user_reminder =
                Some(UserReminder::new("Stay in scope.".into(), frequency));
            agent.task.developer_env_details =
                Some(vec![DeveloperDefinableEnvironmentDetail::new_static(
                    "Build".into(),
                    "green".into(),
                    EnvironmentDetailPlacement::EachUserMessage,
                )]);
        }
        agent
            .scripted_responses
            .get_mut()
            .unwrap()
            .extend([turn("one"), turn("two"), done()]);
        let task = timeout(Duration::from_secs(10), agent.run())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        // The tool-results messages of the two turns, in order.
        let beside: Vec<Vec<&str>> = task
            .conversation
            .messages
            .iter()
            .filter(|message| {
                message
                    .content
                    .first()
                    .is_some_and(|content| content.as_tool_result().is_some())
            })
            .map(|message| {
                message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .collect()
            })
            .take(2)
            .collect();
        assert_eq!(beside.len(), 2);
        let details = "# Current Environment Information (updated this turn)\nBuild:\ngreen\n";
        match frequency {
            None => assert!(beside.iter().all(Vec::is_empty), "{beside:?}"),
            Some(ReminderInjectionFrequency::EveryUserMessage) => {
                for texts in &beside {
                    assert_eq!(
                        texts,
                        &[format!("{details}\n\n# Reminder\nStay in scope.").as_str()]
                    );
                }
            }
            Some(_) => {
                // The task prompt is user message 1: the first results are
                // message 2, the second message 3, when every second one
                // after the prompt gets the reminder.
                assert_eq!(beside[0], [details]);
                assert_eq!(
                    beside[1],
                    [format!("{details}\n\n# Reminder\nStay in scope.").as_str()]
                );
            }
        }
    }
}

/// A tool registered as `load_skill`.
#[derive(Debug)]
struct Skill;

#[async_trait]
impl Tool for Skill {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: crate::skill::LOAD_SKILL_TOOL_NAME.into(),
            description: "Loads a skill".into(),
            parameters: vec![],
            output_based_example: None,
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::BuiltIn,
            read_only: false,
            raw_json_schema_override: None,
        }
    }
    async fn execute(
        &self,
        _: &TaskContext<'_>,
        _: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        Ok("Exact skill instructions.".into())
    }
}

struct Fails;
#[async_trait]
impl crate::tool::ToolResultInterceptor for Fails {
    async fn intercept(
        &self,
        _: &str,
        _: &str,
        _: &HashMap<String, Value>,
        _: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        Err(ToolError::ExecutionFailed("distiller unavailable".into()))
    }
}

/// `load_skill` results are sent as written unless the skills
/// configuration opts in to interception, which then uses its own failure
/// policy; `InformAgent` gives the full result with a note.
#[tokio::test]
async fn load_skill_results_are_intercepted_only_when_opted_in() {
    use crate::tool::InterceptionFailurePolicy;
    let run = |opt_in: Option<InterceptionFailurePolicy>,
               interceptor: Arc<dyn crate::tool::ToolResultInterceptor>| async move {
        let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Skill));
        agent.tool_registry = Arc::new(registry);
        agent.task.agent_config.skills_config = Some(crate::skill::SkillsConfig {
            load_skill_interception: opt_in,
            ..Default::default()
        });
        agent.interceptor_config = Some(InterceptorConfig {
            interceptor,
            failure_policy: InterceptionFailurePolicy::UseRawResult,
        });
        agent
            .execute_native_tool(
                &call("skill", crate::skill::LOAD_SKILL_TOOL_NAME),
                Uuid::new_v4(),
            )
            .await
            .unwrap()
            .unwrap()
            .output
    };
    assert_eq!(
        run(None, Arc::new(Replace)).await,
        "Exact skill instructions."
    );
    assert_eq!(
        run(
            Some(InterceptionFailurePolicy::UseRawResult),
            Arc::new(Replace)
        )
        .await,
        "distilled"
    );
    assert_eq!(
        run(
            Some(InterceptionFailurePolicy::UseRawResult),
            Arc::new(Fails)
        )
        .await,
        "Exact skill instructions."
    );
    let informed = run(
        Some(InterceptionFailurePolicy::RetryThenInformAgent { max_attempts: 2 }),
        Arc::new(Fails),
    )
    .await;
    assert!(
        informed.starts_with("Exact skill instructions.\n\n[This is the full result")
            && informed.contains("distiller unavailable"),
        "{informed}"
    );
}

/// Context injected after tool results doesn't repeat the environment
/// details the results message already carried in the same turn.
#[tokio::test]
async fn a_context_update_after_results_does_not_repeat_the_details() {
    use crate::agent_config::DeveloperDefinableEnvironmentDetail;
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.task.developer_env_details = Some(vec![DeveloperDefinableEnvironmentDetail::new_static(
        "Build".into(),
        "green".into(),
        EnvironmentDetailPlacement::EachUserMessage,
    )]);
    let mut first = ai::Message::new_assistant("");
    first.tool_calls.push(call("one", "plain_text"));
    let mut done = ai::Message::new_assistant("");
    done.tool_calls.push(ai::ToolCall {
        arguments: serde_json::json!({"summary": "done"}),
        ..call("done", "complete_task")
    });
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend([response(first), response(done)]);
    // Given at the boundary after the first turn's results.
    agent.turn_context = Some(Arc::new(AfterFirst(Default::default())));
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    let texts: Vec<&str> = task
        .conversation
        .messages
        .iter()
        .filter(|message| message.role == ai::Role::User)
        .flat_map(|message| message.content.iter().filter_map(ai::Content::as_text))
        .collect();
    let update = texts
        .iter()
        .find(|text| text.starts_with("# Workflow context update"))
        .expect("the update was added");
    assert!(!update.contains("Build:"), "{update}");
    // The results message before it carried the details.
    let results_index = task
        .conversation
        .messages
        .iter()
        .position(|message| {
            message
                .content
                .first()
                .is_some_and(|content| content.as_tool_result().is_some())
        })
        .unwrap();
    let beside: String = task.conversation.messages[results_index]
        .content
        .iter()
        .filter_map(ai::Content::as_text)
        .collect();
    assert!(beside.contains("Build:\ngreen"), "{beside}");
}

/// Gives context from the second boundary on, once.
struct AfterFirst(std::sync::atomic::AtomicUsize);

impl super::super::TurnContextSource for AfterFirst {
    fn turn_context(&self, _task_id: Uuid) -> Option<String> {
        (self.0.fetch_add(1, Ordering::SeqCst) == 1).then(|| "Agent `b` finished.".to_string())
    }
}

/// A context update that follows a user message without this turn's
/// details (the prompt of a continued task) carries them itself.
#[tokio::test]
async fn a_context_update_after_a_continuation_prompt_carries_the_details() {
    use crate::agent_config::DeveloperDefinableEnvironmentDetail;
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    agent.task.developer_env_details = Some(vec![DeveloperDefinableEnvironmentDetail::new_static(
        "Build".into(),
        "green".into(),
        EnvironmentDetailPlacement::EachUserMessage,
    )]);
    agent.task.conversation.system_message = Some("Written earlier.".into());
    agent
        .task
        .conversation
        .add_user_message("Continue with the next part.");
    agent.task.pending_context_injections = vec!["Agent `b` finished.".into()];
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
    let task = timeout(Duration::from_secs(10), agent.run())
        .await
        .unwrap()
        .unwrap();
    let update = task
        .conversation
        .messages
        .iter()
        .flat_map(|message| message.content.iter().filter_map(ai::Content::as_text))
        .find(|text| text.starts_with("# Workflow context update"))
        .expect("the update was added");
    assert!(update.contains("Build:\ngreen"), "{update}");
}

/// A saved task holding a reminder every zero messages (never injected,
/// before validation rejected it) still loads; running it fails at start
/// with the validation error, until the reminder is changed or unset.
#[tokio::test(start_paused = true)]
async fn a_saved_task_with_a_reminder_every_zero_messages_does_not_start() {
    let (mut agent, _events, _commands) = harness(ToolCallingMode::Native);
    let mut saved = serde_json::to_value(&agent.task).unwrap();
    saved["agent_config"]["user_reminder"] =
        serde_json::json!({"content": "Stay in scope.", "frequency": {"EveryNUserMessages": 0}});
    agent.task = serde_json::from_value(saved).unwrap();
    let error = agent.run().await.unwrap_err().to_string();
    assert!(
        error.contains("EveryNUserMessages needs n of at least 1"),
        "{error}"
    );
}
