//! `wait_for_message`: who is offered it, how long it waits, what ends a
//! wait, and delivery of the awaited message at the next turn boundary.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arko_agents::runtime::command::UserCommand;
use arko_agents::{TaskContext, Tool, ToolError, TurnContextSource};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::*;
use crate::messaging::registry::WaitReason;
use crate::messaging::tools::{
    ActivityNotices, LIST_RECIPIENTS_TOOL_NAME, SEND_MESSAGE_TOOL_NAME, WAIT_FOR_MESSAGE_TOOL_NAME,
    WaitForMessageTool,
};
use crate::messaging::wait::{
    DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT, MAX_WAIT_FOR_MESSAGE_TIMEOUT, WaitForMessageTimeout,
};
use crate::test_support::{
    ProbeTool, Reply, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;

const WAITING: AgentStatus = AgentStatus::Waiting(WaitReason::AwaitingMessage);

fn agent(name: &str) -> AgentConfig {
    let mut config = agent_config(&format!("Friendly {name}"));
    config.role_description = format!("{name} role");
    config
}

fn builder_with(client: LLMClient, workspace: &Workspace, agents: &[&str]) -> WorkflowBuilder {
    let mut builder = Workflow::builder()
        .name("wait-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(client));
    for name in agents {
        builder = builder.agent(*name, agent(name));
    }
    builder
}

fn builder(workspace: &Workspace, agents: &[&str]) -> WorkflowBuilder {
    builder_with(LLMClient::new(), workspace, agents)
}

/// `agent`'s workflow tool `name`, when it is offered; its prepared
/// configuration must agree.
fn tool(workflow: &Workflow, agent: &str, name: &str) -> Option<Arc<dyn Tool>> {
    let context = workflow.context();
    let registry = workflow.engine.build_tool_registry(agent, context, 0);
    let config = workflow.engine.prepare_agent_config(
        agent,
        context.agent_configs.get(agent).unwrap(),
        context,
    );
    let listed = config
        .tool_config
        .available_tool_names
        .iter()
        .any(|listed| listed == name);
    let tool = registry.get(name);
    assert_eq!(tool.is_some(), listed, "{agent}: {name}");
    tool
}

fn offered(workflow: &Workflow, agent: &str, name: &str) -> bool {
    tool(workflow, agent, name).is_some()
}

/// A run of `agent` as the engine starts one: marked running, and its
/// command channel registered for message delivery and stops.
struct Run {
    id: uuid::Uuid,
    commands: mpsc::Receiver<UserCommand>,
    _sender: mpsc::Sender<UserCommand>,
}

fn run(workflow: &Workflow, agent: &str) -> Run {
    let context = workflow.context();
    let id = uuid::Uuid::new_v4();
    context.registry.start(&[(id, agent)]);
    let (sender, commands) = mpsc::channel(16);
    context
        .messages
        .register_run(agent, id, sender.clone(), &[]);
    context.run_control.register(id, &sender);
    Run {
        id,
        commands,
        _sender: sender,
    }
}

async fn call(
    workflow: &Workflow,
    agent: &str,
    run: uuid::Uuid,
    name: &str,
    params: &[(&str, Value)],
) -> Result<String, ToolError> {
    let context = workflow.context();
    let task_context = TaskContext::new(run, agent, "", &context.workspace);
    let params = params
        .iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect::<HashMap<_, _>>();
    tool(workflow, agent, name)
        .unwrap()
        .execute(&task_context, params)
        .await
}

async fn wait(workflow: &Workflow, agent: &str, run: uuid::Uuid) -> String {
    call(workflow, agent, run, WAIT_FOR_MESSAGE_TOOL_NAME, &[])
        .await
        .unwrap()
}

async fn send(workflow: &Workflow, from: &Run, from_agent: &str, to: &str, content: &str) {
    call(
        workflow,
        from_agent,
        from.id,
        SEND_MESSAGE_TOOL_NAME,
        &[("to", json!(to)), ("content", json!(content))],
    )
    .await
    .unwrap();
}

/// Waits (in a test, by polling) until `agent` has `status`.
async fn until_status(workflow: &Workflow, agent: &str, status: AgentStatus) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while workflow.agent_status(agent) != Some(status.clone()) {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{agent} never became {status:?}"));
}

/// A context update as the runtime commits it.
fn update(content: &str) -> arko_ai::Message {
    arko_ai::Message::new_user(&format!("# Workflow context update\n\n{content}"))
}

#[test]
fn the_wait_is_offered_to_agents_that_can_receive() {
    let workspace = Workspace::new();
    // One agent, or messaging off: no agent is offered it, and the name is
    // free for a host tool.
    for workflow in [
        builder(&workspace, &["coder"]),
        builder(&workspace, &["coder", "reviewer"]).disable_messaging(),
    ] {
        let workflow = workflow
            .global_tool(ProbeTool::new(WAIT_FOR_MESSAGE_TOOL_NAME, false))
            .build()
            .unwrap();
        for agent in workflow.context().agent_configs.keys() {
            let description = tool(&workflow, agent, WAIT_FOR_MESSAGE_TOOL_NAME)
                .unwrap()
                .definition()
                .description;
            assert_eq!(description, "Test probe.");
        }
    }

    // The lead may message only the worker; the worker no one; the auditor
    // everyone. So the auditor messages the lead and the worker, the lead
    // the worker, and no one the auditor.
    let agents = ["lead", "worker", "auditor"];
    let policy = MessagingPolicy::all()
        .can_message("lead", ["worker"])
        .cannot_message("worker");
    let workflow = builder(&workspace, &agents)
        .messaging(policy.clone())
        .build()
        .unwrap();
    let context = workflow.context();
    assert_eq!(context.message_senders("lead"), ["auditor"]);
    assert_eq!(context.message_senders("worker"), ["auditor", "lead"]);
    assert!(context.message_senders("auditor").is_empty());
    for (agent, sends, receives) in [
        ("lead", true, true),
        ("worker", false, true),
        ("auditor", true, false),
    ] {
        assert_eq!(
            offered(&workflow, agent, SEND_MESSAGE_TOOL_NAME),
            sends,
            "{agent}"
        );
        assert_eq!(
            offered(&workflow, agent, LIST_RECIPIENTS_TOOL_NAME),
            sends,
            "{agent}"
        );
        assert_eq!(
            offered(&workflow, agent, WAIT_FOR_MESSAGE_TOOL_NAME),
            receives,
            "{agent}"
        );
    }

    // The definition is static: the same for every agent, naming no agent
    // and no time.
    let definition = |agent: &str| {
        let tool = tool(&workflow, agent, WAIT_FOR_MESSAGE_TOOL_NAME).unwrap();
        let definition = tool.definition();
        assert!(!definition.parallel_safe);
        serde_json::to_string(&definition).unwrap()
    };
    let lead = definition("lead");
    assert_eq!(lead, definition("worker"));
    for agent in agents {
        assert!(!lead.contains(agent), "{lead}");
    }
    assert!(!lead.chars().any(|c| c.is_ascii_digit()), "{lead}");

    // An agent that isn't offered it may have a tool of that name; one that
    // is may not, and nor may a global tool.
    let auditor = builder(&workspace, &agents)
        .messaging(policy.clone())
        .agent_tool("auditor", ProbeTool::new(WAIT_FOR_MESSAGE_TOOL_NAME, false))
        .build()
        .unwrap();
    assert!(offered(&auditor, "auditor", WAIT_FOR_MESSAGE_TOOL_NAME));
    assert!(matches!(
        builder(&workspace, &agents)
            .messaging(policy.clone())
            .agent_tool("worker", ProbeTool::new(WAIT_FOR_MESSAGE_TOOL_NAME, false))
            .build(),
        Err(WorkflowError::BuilderInvalid(message)) if message.contains("reserved for agent 'worker'")
    ));
    assert!(matches!(
        builder(&workspace, &agents)
            .messaging(policy)
            .global_tool(ProbeTool::new(WAIT_FOR_MESSAGE_TOOL_NAME, false))
            .build(),
        Err(WorkflowError::BuilderInvalid(message)) if message.contains("reserved")
    ));
}

#[test]
fn wait_limits_are_checked_at_build() {
    let workspace = Workspace::new();
    let agents = ["lead", "worker"];
    for invalid in [
        Duration::ZERO,
        MAX_WAIT_FOR_MESSAGE_TIMEOUT + Duration::from_secs(1),
    ] {
        for built in [
            builder(&workspace, &agents)
                .wait_for_message_timeout(invalid)
                .build(),
            builder(&workspace, &agents)
                .agent_wait_for_message_timeout("lead", invalid)
                .build(),
        ] {
            assert!(matches!(
                built,
                Err(WorkflowError::BuilderInvalid(message)) if message.contains("wait_for_message timeout")
            ));
        }
    }
    assert!(matches!(
        builder(&workspace, &agents)
            .agent_wait_for_message_timeout("nobody", WaitForMessageTimeout::Auto)
            .build(),
        Err(WorkflowError::BuilderInvalid(message)) if message.contains("unknown agent 'nobody'")
    ));
    for valid in [
        WaitForMessageTimeout::Auto,
        Duration::from_millis(1).into(),
        MAX_WAIT_FOR_MESSAGE_TIMEOUT.into(),
    ] {
        builder(&workspace, &agents)
            .wait_for_message_timeout(valid)
            .agent_wait_for_message_timeout("worker", valid)
            .build()
            .unwrap();
    }
}

/// The automatic limit follows each agent's model, settings and client
/// auth; a fixed limit, the workflow's or the agent's own, replaces it.
#[tokio::test]
async fn automatic_limits_follow_each_agents_prompt_cache() {
    let workspace = Workspace::new();
    let with_model = |name: &str, model: arko_ai::Model, config: arko_ai::GenerationConfig| {
        let mut agent = agent(name);
        agent.llm_config.model = model;
        agent.llm_config.generation_config = config;
        (name.to_string(), agent)
    };
    let default = arko_ai::GenerationConfig::default;
    let agents = [
        with_model("openai", arko_ai::Model::openai_gpt_6_astra(), default()),
        with_model(
            "older",
            arko_ai::Model::openai_gpt_5_5(),
            default().with_prompt_caching(false),
        ),
        with_model(
            "claude",
            arko_ai::Model::anthropic_claude_opus_5_5(),
            default(),
        ),
        with_model(
            "claude_hour",
            arko_ai::Model::anthropic_claude_opus_5_5(),
            default().with_anthropic_cache_ttl(arko_ai::AnthropicCacheTtl::OneHour),
        ),
        with_model(
            "claude_off",
            arko_ai::Model::anthropic_claude_opus_5_5(),
            default().with_prompt_caching(false),
        ),
        with_model("gemini", arko_ai::Model::google_gemini_2_5_pro(), default()),
        with_model(
            "fixed",
            arko_ai::Model::anthropic_claude_opus_5_5(),
            default(),
        ),
        // The host's lifetime replaces the provider's.
        with_model(
            "hosted",
            arko_ai::Model::openai_gpt_6_astra(),
            default().with_prompt_cache_lifetime(Duration::from_secs(600)),
        ),
    ];
    let minutes = |minutes: u64| Duration::from_secs(minutes * 60);
    let clients = [
        (LLMClient::new().with_openai_api_key("sk-test"), false),
        (
            LLMClient::new().with_openai_chatgpt_oauth("access", None, "account"),
            true,
        ),
    ];
    for (client, oauth) in clients {
        let client = Arc::new(client);
        let mut builder = Workflow::builder()
            .name("wait-test")
            .workspace(&workspace.0)
            .llm_client(client.clone())
            .agent_wait_for_message_timeout("fixed", Duration::from_secs(42));
        for (name, config) in &agents {
            builder = builder.agent(name.clone(), config.clone());
        }
        let workflow = builder.build().unwrap();
        let limit = |agent: &str| {
            WaitForMessageTool::new(
                workflow.context().clone(),
                agent.to_string(),
                client.clone(),
            )
        };
        // OpenAI is a conservative 5 minutes with an API key, and 30
        // minutes over ChatGPT OAuth, whose cache ignores the caching switch.
        let openai = if oauth {
            minutes(27)
        } else {
            Duration::from_secs(270)
        };
        for (agent, expected) in [
            ("openai", openai),
            ("older", openai),
            ("claude", Duration::from_secs(270)),
            ("claude_hour", minutes(54)),
            ("claude_off", DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT),
            ("gemini", DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT),
            ("fixed", Duration::from_secs(42)),
            ("hosted", minutes(9)),
        ] {
            assert_eq!(limit(agent).limit().await, expected, "{agent}");
        }
    }

    // A workflow-wide fixed limit applies to agents without their own.
    let client = Arc::new(LLMClient::new());
    let workflow = Workflow::builder()
        .name("wait-test")
        .workspace(&workspace.0)
        .llm_client(client.clone())
        .agent("claude", agents[2].1.clone())
        .agent("fixed", agents[6].1.clone())
        .wait_for_message_timeout(Duration::from_secs(600))
        .agent_wait_for_message_timeout("fixed", WaitForMessageTimeout::Auto)
        .build()
        .unwrap();
    let limit = |agent: &str| {
        WaitForMessageTool::new(
            workflow.context().clone(),
            agent.to_string(),
            client.clone(),
        )
    };
    assert_eq!(limit("claude").limit().await, Duration::from_secs(600));
    assert_eq!(limit("fixed").limit().await, Duration::from_secs(270));
}

/// An automatic limit counts from the run's last model request, when its
/// prompt cache was last used; a fixed limit doesn't.
#[tokio::test(start_paused = true)]
async fn an_automatic_limit_counts_from_the_last_model_request() {
    let workspace = Workspace::new();
    let client = Arc::new(LLMClient::new());
    for (timeout, expected) in [
        (WaitForMessageTimeout::Auto, "No messages after 170 s."),
        (Duration::from_secs(200).into(), "No messages after 200 s."),
    ] {
        let workflow = builder(&workspace, &["lead", "worker"])
            .wait_for_message_timeout(timeout)
            .build()
            .unwrap();
        let context = workflow.context();
        let lead = run(&workflow, "lead");
        let tool = WaitForMessageTool::new(context.clone(), "lead".to_string(), client.clone());
        // The runtime reports the conversation's new messages, then the model
        // takes 100 s to answer with the call.
        tool.observe_context(lead.id, &arko_ai::Message::new_user("Go."));
        tokio::time::advance(Duration::from_secs(100)).await;
        let task_context = TaskContext::new(lead.id, "lead", "", &context.workspace);
        let started = tokio::time::Instant::now();
        let result = tool.execute(&task_context, HashMap::new()).await.unwrap();
        assert_eq!(result, expected);
        let waited = started.elapsed().as_secs();
        assert_eq!(
            waited,
            if timeout == WaitForMessageTimeout::Auto {
                170
            } else {
                200
            }
        );
    }
}

/// However long ago the run's last model request was, an automatic limit
/// leaves a wait a second.
#[tokio::test(start_paused = true)]
async fn an_automatic_limit_leaves_a_wait_at_least_a_second() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"]).build().unwrap();
    let context = workflow.context();
    let lead = run(&workflow, "lead");
    let tool = WaitForMessageTool::new(
        context.clone(),
        "lead".to_string(),
        Arc::new(LLMClient::new()),
    );
    // The limit is 270 s; the model took longer than that to call the tool.
    tool.observe_context(lead.id, &arko_ai::Message::new_user("Go."));
    tokio::time::advance(Duration::from_secs(300)).await;
    let task_context = TaskContext::new(lead.id, "lead", "", &context.workspace);
    let started = tokio::time::Instant::now();
    let result = tool.execute(&task_context, HashMap::new()).await.unwrap();
    assert_eq!(result, "No messages after 1 s.");
    assert_eq!(started.elapsed(), Duration::from_secs(1));
}

/// A wait that found a message not yet forwarded (the run's command channel
/// is full), which the conversation then turns out to hold, waits on to its
/// limit without spinning: its wait for the forward ends with the message.
#[tokio::test(start_paused = true)]
async fn a_wait_whose_message_is_seen_before_its_forward_does_not_spin() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let context = workflow.context().clone();
    let lead = uuid::Uuid::new_v4();
    context.registry.start(&[(lead, "lead")]);
    let (sender, _commands) = mpsc::channel(1);
    sender.try_send(UserCommand::SuspendTask).unwrap();
    context
        .messages
        .register_run("lead", lead, sender.clone(), &[]);
    context.run_control.register(lead, &sender);
    let worker = run(&workflow, "worker");
    send(&workflow, &worker, "worker", "lead", "Seen already.").await;
    let payload = crate::messaging::injection(&context.messages.pending("lead")[0]);

    let tool = tool(&workflow, "lead", WAIT_FOR_MESSAGE_TOOL_NAME).unwrap();
    let polls = Arc::new(AtomicUsize::new(0));
    let waiting = {
        let (tool, polls, cwd) = (tool.clone(), polls.clone(), context.workspace.clone());
        let mut wait = Box::pin(async move {
            let task_context = TaskContext::new(lead, "lead", "", &cwd);
            tool.execute(&task_context, HashMap::new()).await
        });
        tokio::spawn(std::future::poll_fn(move |cx| {
            polls.fetch_add(1, Ordering::SeqCst);
            wait.as_mut().poll(cx)
        }))
    };
    // The wait found the message and waits a second at most for its forward.
    while polls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    tool.observe_context(lead, &update(&payload));
    let before = polls.load(Ordering::SeqCst);
    for _ in 0..40 {
        tokio::time::advance(Duration::from_secs(1)).await;
    }
    let result = tokio::time::timeout(Duration::from_secs(1), waiting)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.unwrap(), "No messages after 30 s.");
    // Woken when the forward deadline passed and at the limit; spinning on
    // the passed deadline, it would be polled again at every step.
    let polls = polls.load(Ordering::SeqCst) - before;
    assert!(polls <= 4, "polled {polls} times");
}

/// A message sent while an agent waits ends the wait at once, is in the
/// agent's command channel by then (so the runtime adds it at its next turn
/// boundary), and others see the agent waiting meanwhile.
#[tokio::test]
async fn a_message_ends_the_wait_at_once_and_is_ready_for_the_next_turn() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let mut lead = run(&workflow, "lead");
    let worker = run(&workflow, "worker");
    let started = Instant::now();
    let (result, ()) = tokio::join!(wait(&workflow, "lead", lead.id), async {
        until_status(&workflow, "lead", WAITING).await;
        let listed = call(
            &workflow,
            "worker",
            worker.id,
            LIST_RECIPIENTS_TOOL_NAME,
            &[],
        )
        .await
        .unwrap();
        assert_eq!(listed, "- `lead` (lead role): waiting (awaiting_message)");
        send(&workflow, &worker, "worker", "lead", "The review is done.").await;
    });
    assert_eq!(result, "Message from `worker` waiting.");
    assert!(started.elapsed() < Duration::from_secs(5));
    let Ok(UserCommand::InjectContext { content }) = lead.commands.try_recv() else {
        panic!("the message is in the channel");
    };
    assert!(content.contains("The review is done."), "{content}");
    assert_eq!(workflow.agent_status("lead"), Some(AgentStatus::Running));
}

/// Without messages a wait ends at its limit. The model may ask for a
/// shorter wait, never a longer one.
#[tokio::test]
async fn a_wait_ends_at_its_limit_which_the_model_can_only_shorten() {
    let workspace = Workspace::new();
    let limited = |limit: Duration| {
        builder(&workspace, &["lead", "worker"])
            .wait_for_message_timeout(limit)
            .build()
            .unwrap()
    };
    let workflow = limited(Duration::from_millis(300));
    let lead = run(&workflow, "lead");
    let started = Instant::now();
    assert_eq!(
        wait(&workflow, "lead", lead.id).await,
        "No messages after 0.3 s."
    );
    let waited = started.elapsed();
    assert!(waited >= Duration::from_millis(300), "{waited:?}");
    let started = Instant::now();
    let result = call(
        &workflow,
        "lead",
        lead.id,
        WAIT_FOR_MESSAGE_TOOL_NAME,
        &[("timeout_seconds", json!("3600"))],
    )
    .await
    .unwrap();
    assert_eq!(result, "No messages after 0.3 s.");
    let waited = started.elapsed();
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    assert_eq!(workflow.agent_status("lead"), Some(AgentStatus::Running));

    let workflow = limited(Duration::from_secs(30));
    let lead = run(&workflow, "lead");
    let started = Instant::now();
    let result = call(
        &workflow,
        "lead",
        lead.id,
        WAIT_FOR_MESSAGE_TOOL_NAME,
        &[("timeout_seconds", json!("0.2"))],
    )
    .await
    .unwrap();
    assert_eq!(result, "No messages after 0.2 s.");
    assert!(started.elapsed() < Duration::from_secs(5));
    for invalid in [json!("0"), json!("-3"), json!("soon")] {
        assert!(matches!(
            call(
                &workflow,
                "lead",
                lead.id,
                WAIT_FOR_MESSAGE_TOOL_NAME,
                &[("timeout_seconds", invalid)]
            )
            .await,
            Err(ToolError::ParameterValidation { .. })
        ));
    }
}

/// A start or finish of an agent the waiter works with ends the wait; the
/// start notice still comes at the next turn boundary. Other agents'
/// starts don't end it.
#[tokio::test]
async fn a_start_or_finish_ends_the_wait_and_start_notices_still_come() {
    let workspace = Workspace::new();
    let policy = MessagingPolicy::all()
        .can_message("lead", ["worker"])
        .can_message("worker", ["lead"])
        .cannot_message("auditor");
    let workflow = builder(&workspace, &["lead", "worker", "auditor"])
        .messaging(policy)
        .wait_for_message_timeout(Duration::from_secs(30))
        .agent_wait_for_message_timeout("worker", Duration::from_millis(300))
        .build()
        .unwrap();
    let context = workflow.context();
    let lead = run(&workflow, "lead");
    // What the lead's initial prompt told it.
    context
        .registry
        .roster(lead.id, &["worker".to_string()], false);
    let (result, worker) = tokio::join!(wait(&workflow, "lead", lead.id), async {
        until_status(&workflow, "lead", WAITING).await;
        run(&workflow, "worker")
    });
    assert_eq!(result, "`worker` started.");
    let notices = ActivityNotices {
        context: context.clone(),
        agent_name: "lead".to_string(),
        review: false,
    };
    let notice = notices.turn_context(lead.id).unwrap();
    assert!(
        notice.starts_with("Agent `worker` (role: worker role) is now running."),
        "{notice}"
    );

    let (result, ()) = tokio::join!(wait(&workflow, "lead", lead.id), async {
        until_status(&workflow, "lead", WAITING).await;
        context
            .registry
            .finish(worker.id, AgentRunStatus::Completed);
    });
    assert_eq!(result, "`worker` finished (completed).");

    // The auditor is no peer of the worker's.
    let worker = run(&workflow, "worker");
    let (result, ()) = tokio::join!(wait(&workflow, "worker", worker.id), async {
        until_status(&workflow, "worker", WAITING).await;
        run(&workflow, "auditor");
    });
    assert_eq!(result, "No messages after 0.3 s.");
}

/// A stop of the workflow, or of the waiting run, ends even a long wait at
/// once; a wait whose task is aborted leaves the agent running.
#[tokio::test]
async fn stops_end_a_long_wait_at_once() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_secs(3_600))
        .build()
        .unwrap();
    let context = workflow.context();

    let lead = run(&workflow, "lead");
    tokio::select! {
        _ = wait(&workflow, "lead", lead.id) => panic!("the wait ended"),
        () = until_status(&workflow, "lead", WAITING) => {}
    }
    assert_eq!(workflow.agent_status("lead"), Some(AgentStatus::Running));

    // The workflow winds this run down (over budget, for example).
    let (result, stopped) = tokio::join!(wait(&workflow, "lead", lead.id), async {
        until_status(&workflow, "lead", WAITING).await;
        context.run_control.stop_agent(lead.id);
        Instant::now()
    });
    assert_eq!(result, "Run stopping.");
    assert!(stopped.elapsed() < Duration::from_secs(1));
    context.registry.finish(lead.id, AgentRunStatus::Cancelled);

    let lead = run(&workflow, "lead");
    let (result, cancelled) = tokio::join!(wait(&workflow, "lead", lead.id), async {
        until_status(&workflow, "lead", WAITING).await;
        assert!(workflow.cancel());
        Instant::now()
    });
    assert_eq!(result, "Workflow stopping.");
    assert!(cancelled.elapsed() < Duration::from_secs(1));
}

/// A graceful stop's wrap-up ends a long wait at once, through the call's
/// stop signal: the agent is told to finish at its next turn boundary.
#[tokio::test]
async fn a_wrap_up_ends_a_long_wait_through_the_stop_signal() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_secs(3_600))
        .build()
        .unwrap();
    let context = workflow.context();
    let lead = run(&workflow, "lead");
    let task_context = TaskContext::new(lead.id, "lead", "", &context.workspace)
        .with_stop(context.run_control.tools_signal());
    let tool = tool(&workflow, "lead", WAIT_FOR_MESSAGE_TOOL_NAME).unwrap();
    // Bounded: a wait that ignored the signal would last the hour.
    let wait = tokio::time::timeout(
        Duration::from_secs(10),
        tool.execute(&task_context, HashMap::new()),
    );
    let (result, stopped) = tokio::join!(wait, async {
        until_status(&workflow, "lead", WAITING).await;
        assert!(workflow.stop(
            crate::StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(3_600))
        ));
        Instant::now()
    });
    let result = result.expect("the wait returns on the stop signal");
    assert_eq!(result.unwrap(), "Workflow stopping.");
    assert!(stopped.elapsed() < Duration::from_secs(1));
    assert!(workflow.is_stopping(), "the run is still wrapping up");
    assert!(
        context
            .run_control
            .wrap_up_notice(lead.id)
            .is_some_and(|notice| notice.starts_with("The workflow is stopping: deploying a fix."))
    );
}

/// Only agent messages the conversation lacks end a wait as messages: not one
/// it holds already (still queued until the run completes), nor context
/// notices, which end it through its context signal once the runtime has
/// them.
#[tokio::test]
async fn only_undelivered_agent_messages_end_a_wait() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_millis(300))
        .build()
        .unwrap();
    let context = workflow.context();
    let mut lead = run(&workflow, "lead");
    let worker = run(&workflow, "worker");
    send(&workflow, &worker, "worker", "lead", "First finding.").await;
    let started = Instant::now();
    assert_eq!(
        wait(&workflow, "lead", lead.id).await,
        "Message from `worker` waiting."
    );
    assert!(started.elapsed() < Duration::from_millis(250));

    // The runtime adds it at the turn boundary and tells the tools.
    let Ok(UserCommand::InjectContext { content }) = lead.commands.try_recv() else {
        panic!("the message is in the channel");
    };
    tool(&workflow, "lead", WAIT_FOR_MESSAGE_TOOL_NAME)
        .unwrap()
        .observe_context(lead.id, &update(&content));
    assert_eq!(context.messages.pending("lead").len(), 1);
    let (result, ()) = tokio::join!(wait(&workflow, "lead", lead.id), async {
        until_status(&workflow, "lead", WAITING).await;
        assert!(
            context
                .messages
                .inject_task(lead.id, "src/lib.rs changed".into())
                .await
        );
    });
    assert_eq!(result, "No messages after 0.3 s.");
}

/// The text of every user item of a request body.
fn user_text(body: &str) -> String {
    let body: Value = serde_json::from_str(body).unwrap();
    let mut text = Vec::new();
    for item in body["input"].as_array().unwrap() {
        if item["role"] != "user" {
            continue;
        }
        match &item["content"] {
            Value::String(content) => text.push(content.clone()),
            Value::Array(parts) => text.extend(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str().map(str::to_string)),
            ),
            _ => {}
        }
    }
    text.join("\n")
}

/// The tool calls a request body replays, in order: name and output.
fn tool_results(body: &str) -> Vec<(String, String)> {
    let body: Value = serde_json::from_str(body).unwrap();
    let items = body["input"].as_array().unwrap();
    let names = items
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| (item["call_id"].clone(), item["name"].as_str().unwrap()))
        .collect::<Vec<_>>();
    items
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .map(|item| {
            let name = names
                .iter()
                .find(|(id, _)| *id == item["call_id"])
                .map_or("?", |(_, name)| name);
            let output = match &item["output"] {
                Value::String(output) => output.clone(),
                other => other.to_string(),
            };
            (name.to_string(), output)
        })
        .collect()
}

/// In native tool mode, against a scripted model: the lead asks the worker
/// for a review and waits; the worker waits for work, then replies. Each
/// wait ends when the message arrives, and the message is in the very next
/// request, once, and acknowledged when its run completes.
#[tokio::test]
async fn a_lead_and_a_worker_wait_for_each_other_in_native_mode() {
    lead_and_worker_wait_for_each_other(true).await;
}

/// The same with every tool call approved by the workflow's driver (the
/// agents' default): the message still reaches the very next request.
#[tokio::test]
async fn approved_waits_deliver_the_message_with_the_next_request() {
    lead_and_worker_wait_for_each_other(false).await;
}

async fn lead_and_worker_wait_for_each_other(auto_approve: bool) {
    let workspace = Workspace::new();
    const REQUEST: &str = "Please review the parser.";
    const REPLY: &str = "The parser looks fine.";
    let worker_asked = Arc::new(tokio::sync::Notify::new());
    let lead_waits = Arc::new(tokio::sync::Notify::new());
    let server = ResponsesServer::start_async({
        let (worker_asked, lead_waits) = (worker_asked.clone(), lead_waits.clone());
        move |_, body| {
            let (worker_asked, lead_waits) = (worker_asked.clone(), lead_waits.clone());
            Box::pin(async move {
                let text = user_text(&body);
                let results = tool_results(&body);
                let turn = results.len();
                let sent = results
                    .iter()
                    .any(|(name, _)| name == SEND_MESSAGE_TOOL_NAME);
                let wait = |id: String| tool_call_turn(&id, WAIT_FOR_MESSAGE_TOOL_NAME, json!({}));
                let reply = if text.contains("Lead the work.") {
                    let id = format!("lead{turn}");
                    if !sent {
                        // Ask once the worker runs its first request.
                        worker_asked.notified().await;
                        tool_call_turn(
                            &id,
                            SEND_MESSAGE_TOOL_NAME,
                            json!({"to": "worker", "content": REQUEST}),
                        )
                    } else if text.contains(REPLY) {
                        complete_turn(&id, "reviewed")
                    } else {
                        lead_waits.notify_one();
                        wait(id)
                    }
                } else if text.contains("Review the work.") {
                    let id = format!("worker{turn}");
                    if turn == 0 {
                        worker_asked.notify_one();
                    }
                    if sent {
                        complete_turn(&id, "replied")
                    } else if text.contains(REQUEST) {
                        // Reply once the lead is waiting for it.
                        lead_waits.notified().await;
                        tool_call_turn(
                            &id,
                            SEND_MESSAGE_TOOL_NAME,
                            json!({"to": "lead", "content": REPLY}),
                        )
                    } else {
                        wait(id)
                    }
                } else {
                    return None;
                };
                Some(Reply::Stream(reply))
            })
        }
    })
    .await;
    let mut builder = Workflow::builder()
        .name("wait-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .wait_for_message_timeout(Duration::from_secs(20));
    for name in ["lead", "worker"] {
        let mut config = agent(name);
        config.tool_config.default_auto_approve_tool_use = auto_approve;
        builder = builder.agent(name, config);
    }
    let workflow = builder.build().unwrap();
    let results = tokio::time::timeout(
        Duration::from_secs(15),
        workflow.run_agents_parallel(vec![
            ("lead", "Lead the work.".to_string()),
            ("worker", "Review the work.".to_string()),
        ]),
    )
    .await
    .expect("no wait ran to its limit")
    .unwrap();
    assert!(
        results
            .iter()
            .all(|result| result.status == AgentRunStatus::Completed)
    );

    let bodies = server.bodies();
    for (prompt, from, content) in [
        ("Lead the work.", "worker", REPLY),
        ("Review the work.", "lead", REQUEST),
    ] {
        let bodies = bodies
            .iter()
            .filter(|body| user_text(body).contains(prompt))
            .collect::<Vec<_>>();
        // The request after the wait that found the message carries it.
        let after_wait = bodies
            .iter()
            .find(|body| {
                tool_results(body).last().is_some_and(|(name, output)| {
                    name == WAIT_FOR_MESSAGE_TOOL_NAME
                        && output.contains(&format!("Message from `{from}` waiting."))
                })
            })
            .unwrap_or_else(|| panic!("{prompt}: a wait found the message"));
        assert_eq!(
            user_text(after_wait).matches(content).count(),
            1,
            "{prompt}"
        );
        // Delivered once for good.
        assert_eq!(
            user_text(bodies.last().unwrap()).matches(content).count(),
            1,
            "{prompt}"
        );
    }
    // Acknowledged when the runs completed.
    assert!(workflow.context().messages.pending("lead").is_empty());
    assert!(workflow.context().messages.pending("worker").is_empty());
}

/// Cancelling the workflow while an agent waits (with an hour's limit) ends
/// the run at once, cancelled.
#[tokio::test]
async fn a_cancel_during_a_long_wait_ends_the_run_at_once() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        (user_text(body).contains("Lead the work.") && tool_results(body).is_empty())
            .then(|| tool_call_turn("lead0", WAIT_FOR_MESSAGE_TOOL_NAME, json!({})))
    })
    .await;
    let workflow = builder_with(server.client(), &workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_secs(3_600))
        .build()
        .unwrap();
    let worker = workflow
        .spawn_agent("worker", "Review the work.")
        .await
        .unwrap();
    let lead = workflow
        .spawn_agent("lead", "Lead the work.")
        .await
        .unwrap();
    until_status(&workflow, "lead", WAITING).await;
    let cancelled = Instant::now();
    assert!(workflow.cancel());
    let result = tokio::time::timeout(Duration::from_secs(5), lead.wait())
        .await
        .unwrap();
    let latency = cancelled.elapsed();
    assert!(
        matches!(result, Err(WorkflowError::Cancelled)),
        "{result:?}"
    );
    assert!(latency < Duration::from_secs(1), "{latency:?}");
    let recorded = workflow
        .result()
        .agent_results
        .into_iter()
        .find(|result| result.agent_name == "lead")
        .unwrap();
    assert_eq!(recorded.status, AgentRunStatus::Cancelled);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker.wait()).await;
}

/// A wait with `task_context`, bounded.
async fn wait_with(tool: &dyn Tool, task_context: &TaskContext<'_>) -> String {
    tokio::time::timeout(
        Duration::from_secs(10),
        tool.execute(task_context, HashMap::new()),
    )
    .await
    .expect("the wait ends")
    .unwrap()
}

/// Moving the wait to the background ends it, and so does context that
/// arrives for the run. A message that arrives as context is still named.
#[tokio::test]
async fn a_move_or_arriving_context_ends_a_wait() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"])
        .wait_for_message_timeout(Duration::from_secs(3_600))
        .build()
        .unwrap();
    let context = workflow.context();
    let lead = run(&workflow, "lead");
    let tool = tool(&workflow, "lead", WAIT_FOR_MESSAGE_TOOL_NAME).unwrap();
    assert_eq!(tool.background_mode(), arko_agents::ToolBackground::Own);
    let moves = arko_agents::MoveSource::new();
    let mut task_context = TaskContext::new(lead.id, "lead", "", &context.workspace);
    task_context.moved = moves.signal();
    let (result, ()) = tokio::join!(wait_with(tool.as_ref(), &task_context), async {
        until_status(&workflow, "lead", WAITING).await;
        moves.fire();
    });
    assert_eq!(
        result,
        "The user ended this wait so you can go on with other work. Messages from other agents reach you as notes as they come."
    );

    let arrivals = arko_agents::ContextSource::new();
    let mut task_context = TaskContext::new(lead.id, "lead", "", &context.workspace);
    task_context.context_arrived = arrivals.signal();
    let (result, ()) = tokio::join!(wait_with(tool.as_ref(), &task_context), async {
        until_status(&workflow, "lead", WAITING).await;
        arrivals.notify();
    });
    assert_eq!(
        result,
        "An update for you arrived while you waited. It follows. Wait again if you still need to."
    );

    let worker = run(&workflow, "worker");
    let mut task_context = TaskContext::new(lead.id, "lead", "", &context.workspace);
    task_context.context_arrived = arrivals.signal();
    let (result, ()) = tokio::join!(wait_with(tool.as_ref(), &task_context), async {
        until_status(&workflow, "lead", WAITING).await;
        send(&workflow, &worker, "worker", "lead", "Found it.").await;
        arrivals.notify();
    });
    assert_eq!(result, "Message from `worker` waiting.");
}
