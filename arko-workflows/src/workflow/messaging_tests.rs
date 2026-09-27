//! Which agents are offered the messaging tools, what they learn about the
//! agents they may message, and when.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arko_agents::{AgentEvent, TaskContext, Tool, ToolError};
use serde_json::{Value, json};

use super::*;
use crate::messaging::registry::{AgentStatus, WaitReason};
use crate::messaging::tools::{LIST_RECIPIENTS_TOOL_NAME, SEND_MESSAGE_TOOL_NAME};
use crate::test_support::{
    ProbeTool, Reply, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;

const ROSTER: &str = "# Agents you can message";

fn agent(name: &str) -> arko_agents::AgentConfig {
    let mut config = agent_config(&format!("Friendly {name}"));
    config.role_description = format!("{name} role");
    config
}

fn builder_with(client: LLMClient, workspace: &Workspace, agents: &[&str]) -> WorkflowBuilder {
    let mut builder = Workflow::builder()
        .name("messaging-test")
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

/// The workflow's messaging tools for `agent`, when it is offered them; the
/// agent's prepared configuration must agree.
fn messaging_tools(workflow: &Workflow, agent: &str) -> Option<(Arc<dyn Tool>, Arc<dyn Tool>)> {
    let context = workflow.context();
    let registry = workflow.engine.build_tool_registry(agent, context, 0);
    let config = workflow.engine.prepare_agent_config(
        agent,
        context.agent_configs.get(agent).unwrap(),
        context,
    );
    let listed = |name: &str| {
        config
            .tool_config
            .available_tool_names
            .iter()
            .any(|listed| listed == name)
    };
    let send = registry.get(SEND_MESSAGE_TOOL_NAME);
    let list = registry.get(LIST_RECIPIENTS_TOOL_NAME);
    assert_eq!(send.is_some(), listed(SEND_MESSAGE_TOOL_NAME), "{agent}");
    assert_eq!(list.is_some(), listed(LIST_RECIPIENTS_TOOL_NAME), "{agent}");
    assert_eq!(send.is_some(), list.is_some(), "{agent}");
    send.zip(list)
}

fn task_context(workspace: &Workspace) -> TaskContext<'_> {
    TaskContext {
        task_id: uuid::Uuid::new_v4(),
        agent_name: "Friendly sender",
        initial_task_prompt: "coordinate",
        current_working_directory: &workspace.0,
    }
}

async fn send(
    workflow: &Workflow,
    from: &str,
    to: &str,
    workspace: &Workspace,
) -> Result<String, ToolError> {
    let (tool, _) = messaging_tools(workflow, from).unwrap();
    let params = HashMap::from([
        ("to".to_string(), json!(to)),
        ("content".to_string(), json!("hello")),
    ]);
    tool.execute(&task_context(workspace), params).await
}

async fn list(workflow: &Workflow, agent: &str, workspace: &Workspace) -> String {
    let (_, tool) = messaging_tools(workflow, agent).unwrap();
    tool.execute(&task_context(workspace), HashMap::new())
        .await
        .unwrap()
}

/// Marks a run of `agent` running, as the engine does when it starts one.
fn start(workflow: &Workflow, agent: &str) -> uuid::Uuid {
    let run = uuid::Uuid::new_v4();
    workflow.context().registry.start(&[(run, agent)]);
    run
}

/// The user text of a request body, for checking what an agent was told.
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

/// The probe tool results a request body carries.
fn tool_outputs(body: &str) -> usize {
    let body: Value = serde_json::from_str(body).unwrap();
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| {
            item["type"] == "function_call_output"
                && item["call_id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with("call_p"))
        })
        .count()
}

fn tools(body: &str) -> Value {
    serde_json::from_str::<Value>(body).unwrap()["tools"].clone()
}

fn is_request_of(body: &str, prompt: &str) -> bool {
    user_text(body).contains(prompt)
}

#[test]
fn a_single_agent_is_offered_no_messaging_tools_and_the_names_are_free() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["coder"]).build().unwrap();
    assert!(messaging_tools(&workflow, "coder").is_none());

    // Host tools may take the names when no agent is offered the workflow's.
    let workflow = builder(&workspace, &["coder"])
        .global_tool(ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
        .global_tool(ProbeTool::new(LIST_RECIPIENTS_TOOL_NAME, false))
        .build()
        .unwrap();
    let registry = workflow
        .engine
        .build_tool_registry("coder", workflow.context(), 0);
    for name in [SEND_MESSAGE_TOOL_NAME, LIST_RECIPIENTS_TOOL_NAME] {
        assert_eq!(
            registry.get(name).unwrap().definition().description,
            "Test probe."
        );
    }
}

#[test]
fn disabled_messaging_offers_neither_tool() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["coder", "reviewer"])
        .disable_messaging()
        .global_tool(ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
        .build()
        .unwrap();
    assert!(workflow.context().message_recipients("coder").is_empty());
    let registry = workflow
        .engine
        .build_tool_registry("reviewer", workflow.context(), 0);
    assert!(registry.get(LIST_RECIPIENTS_TOOL_NAME).is_none());
    assert_eq!(
        registry
            .get(SEND_MESSAGE_TOOL_NAME)
            .unwrap()
            .definition()
            .description,
        "Test probe."
    );
}

/// The tool definitions name no agent, so they are the same for every agent
/// and don't change as agents start.
#[test]
fn the_messaging_tools_are_static() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"]).build().unwrap();
    let definitions = |agent: &str| {
        let (send, list) = messaging_tools(&workflow, agent).unwrap();
        serde_json::to_string(&(send.definition(), list.definition())).unwrap()
    };
    let before = definitions("lead");
    assert!(!before.contains("worker") && !before.contains("lead"));
    start(&workflow, "worker");
    assert_eq!(definitions("lead"), before);
    assert_eq!(definitions("worker"), before);
    // The name is the workflow's while an agent is offered the tool.
    for name in [SEND_MESSAGE_TOOL_NAME, LIST_RECIPIENTS_TOOL_NAME] {
        assert!(matches!(
            builder(&workspace, &["lead", "worker"])
                .global_tool(ProbeTool::new(name, false))
                .build(),
            Err(WorkflowError::BuilderInvalid(message)) if message.contains("reserved")
        ));
    }
}

/// `list_recipients` reports each permitted recipient, never the agent
/// itself, through a run: not started, running, waiting, finished.
#[tokio::test]
async fn list_recipients_reports_statuses_through_a_run() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["lead", "worker"]).build().unwrap();
    assert_eq!(
        list(&workflow, "lead", &workspace).await,
        "- `worker` (worker role): not_started"
    );
    let run = start(&workflow, "worker");
    assert_eq!(
        list(&workflow, "lead", &workspace).await,
        "- `worker` (worker role): running"
    );
    workflow.context().registry.observe(
        run,
        &AgentEvent::RetryScheduled {
            attempt: 1,
            delay_ms: 60_000,
            elapsed_ms: 0,
            rate_limit: true,
        },
    );
    let waiting = list(&workflow, "lead", &workspace).await;
    assert!(
        waiting.starts_with("- `worker` (worker role): waiting (rate_limited until "),
        "{waiting}"
    );
    workflow.pause().unwrap();
    assert_eq!(
        list(&workflow, "lead", &workspace).await,
        "- `worker` (worker role): waiting (paused)"
    );
    workflow.resume().unwrap();
    workflow
        .context()
        .registry
        .finish(run, AgentRunStatus::Failed);
    assert_eq!(
        list(&workflow, "lead", &workspace).await,
        "- `worker` (worker role): finished (failed)"
    );
    assert_eq!(
        workflow.agent_status("worker"),
        Some(AgentStatus::Finished(AgentRunStatus::Failed))
    );
    assert_eq!(workflow.agent_status("nobody"), None);
}

/// Statuses as a real run goes through them: not started, running, waiting
/// out a provider rate limit, finished.
#[tokio::test]
async fn statuses_follow_a_rate_limited_run() {
    let workspace = Workspace::new();
    let respond = Arc::new(tokio::sync::Notify::new());
    let server = ResponsesServer::start_async({
        let respond = respond.clone();
        move |index, _| {
            let respond = respond.clone();
            Box::pin(async move {
                Some(if index == 0 {
                    respond.notified().await;
                    Reply::Status {
                        status: 429,
                        retry_after: Some(2),
                        body: json!({"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}}).to_string(),
                    }
                } else {
                    Reply::Stream(complete_turn("done", "worked"))
                })
            })
        }
    })
    .await;
    let workflow = builder_with(server.client(), &workspace, &["lead", "worker"])
        .build()
        .unwrap();
    start(&workflow, "lead");
    let status = || async { list(&workflow, "lead", &workspace).await };
    assert_eq!(status().await, "- `worker` (worker role): not_started");
    let worker = workflow
        .spawn_agent("worker", "Do the work.")
        .await
        .unwrap();
    assert_eq!(status().await, "- `worker` (worker role): running");
    let wait = tokio::spawn(worker.wait());
    server.wait_for_requests(1).await;
    respond.notify_one();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !status()
            .await
            .starts_with("- `worker` (worker role): waiting (rate_limited until ")
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the worker waits out the rate limit");
    let result = tokio::time::timeout(Duration::from_secs(20), wait)
        .await
        .unwrap()
        .unwrap();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        status().await,
        "- `worker` (worker role): finished (completed)"
    );
}

/// A restricted sender sees and reaches only its recipients; an
/// unrestricted one reaches every other agent; one permitted no one isn't
/// offered the tools; not-started and finished recipients are refused,
/// waiting ones reached. Agents started through workflow services get their
/// tools from the same registry and follow the same policy.
#[tokio::test]
async fn a_policy_and_the_recipient_status_decide_who_can_be_messaged() {
    let workspace = Workspace::new();
    let policy = MessagingPolicy::all()
        .can_message("worker_a", ["lead"])
        .cannot_message("auditor");
    let workflow = builder(&workspace, &["lead", "worker_a", "worker_b", "auditor"])
        .messaging(policy.clone())
        // An agent not offered the workflow's tools may have its own.
        .agent_tool("auditor", ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
        .build()
        .unwrap();
    let context = workflow.context();
    assert_eq!(context.message_recipients("worker_a"), ["lead"]);
    assert_eq!(
        context.message_recipients("lead"),
        ["auditor", "worker_a", "worker_b"]
    );
    assert_eq!(
        context.message_recipients("worker_b"),
        ["auditor", "lead", "worker_a"]
    );
    assert!(context.message_recipients("auditor").is_empty());
    let auditor = workflow.engine.build_tool_registry("auditor", context, 0);
    assert!(auditor.get(LIST_RECIPIENTS_TOOL_NAME).is_none());
    assert_eq!(
        list(&workflow, "worker_a", &workspace).await,
        "- `lead` (lead role): not_started"
    );

    // Not permitted, unknown, or itself: refused without naming others.
    for to in ["worker_b", "auditor", "nobody", "worker_a"] {
        let error = send(&workflow, "worker_a", to, &workspace)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("The agents you can message: lead."),
            "{error}"
        );
        assert!(!error.contains("worker_b") || to == "worker_b", "{error}");
    }
    let not_started = send(&workflow, "worker_a", "lead", &workspace)
        .await
        .unwrap_err()
        .to_string();
    assert!(not_started.contains("has not started"), "{not_started}");

    let lead = start(&workflow, "lead");
    let sent = send(&workflow, "worker_a", "lead", &workspace)
        .await
        .unwrap();
    assert!(sent.ends_with("sent to agent 'lead'."), "{sent}");
    context.registry.observe(
        lead,
        &AgentEvent::UsageLimitReached {
            resets_at_unix_ms: None,
            waiting_until_unix_ms: None,
            error_message: "limit".into(),
        },
    );
    let sent = send(&workflow, "worker_a", "lead", &workspace)
        .await
        .unwrap();
    assert!(sent.contains("which is waiting (usage_limit)"), "{sent}");
    context.registry.finish(lead, AgentRunStatus::Completed);
    let finished = send(&workflow, "worker_a", "lead", &workspace)
        .await
        .unwrap_err()
        .to_string();
    assert!(finished.contains("is finished (completed)"), "{finished}");
    assert_eq!(context.messages.pending("lead").len(), 2);

    for invalid in [
        MessagingPolicy::all().can_message("worker_a", ["Friendly lead"]),
        MessagingPolicy::all().cannot_message("editor"),
    ] {
        assert!(matches!(
            builder(&workspace, &["lead", "worker_a"]).messaging(invalid).build(),
            Err(WorkflowError::BuilderInvalid(message)) if message.contains("unknown agent")
        ));
    }
    assert!(matches!(
        builder(&workspace, &["lead", "worker_a", "worker_b", "auditor"])
            .messaging(policy)
            .agent_tool("worker_a", ProbeTool::new(LIST_RECIPIENTS_TOOL_NAME, false))
            .build(),
        Err(WorkflowError::BuilderInvalid(message)) if message.contains("reserved")
    ));
}

#[tokio::test]
async fn the_bus_refuses_a_message_to_its_sender() {
    let bus = crate::messaging::MessageBus::new(["lead".to_string(), "worker".to_string()]);
    let error = bus
        .send("lead", None, "lead", "hello".into(), Vec::new())
        .await
        .unwrap_err();
    assert_eq!(error, "'lead' is not a permitted recipient");
    assert!(bus.pending("lead").is_empty());
}

/// The initial prompt lists the permitted recipients, with their roles and
/// statuses, once; a continuation adds no list, and neither does a
/// completion verifier's prompt, nor the target prompt it quotes.
#[tokio::test]
async fn only_the_initial_prompt_lists_the_recipients() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        let text = user_text(body);
        if text.contains("You are a completion verifier") {
            Some(complete_turn("verdict", "ACCEPTED"))
        } else if text.contains("Draft it.") {
            Some(complete_turn("done", "drafted"))
        } else {
            // The lead keeps running.
            None
        }
    })
    .await;
    let workflow = Arc::new(
        builder_with(
            server.client(),
            &workspace,
            &["lead", "writer", "reviewer", "idle"],
        )
        .messaging(MessagingPolicy::all().can_message("writer", ["lead", "idle"]))
        .completion_verifier(
            crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
                .target("writer"),
        )
        .build()
        .unwrap(),
    );
    let lead = workflow
        .spawn_agent("lead", "Lead the work.")
        .await
        .unwrap();
    server.wait_for_requests(1).await;

    let result = workflow.run_agent("writer", "Draft it.").await.unwrap();
    let continued = workflow.continue_agent(&result, "Draft it again.").await;
    assert!(continued.is_ok(), "{continued:?}");

    let bodies = server.bodies();
    let writer = bodies
        .iter()
        .filter(|body| {
            is_request_of(body, "Draft it.")
                && !is_request_of(body, "You are a completion verifier")
        })
        .collect::<Vec<_>>();
    let first = user_text(writer[0]);
    assert_eq!(first.matches(ROSTER).count(), 1, "{first}");
    assert!(first.contains("- `lead` (lead role): running"), "{first}");
    assert!(
        first.contains("- `idle` (idle role): not_started"),
        "{first}"
    );
    assert!(!first.contains("reviewer role"), "{first}");
    // The continuation's request: still the one list, from the initial prompt.
    let last = user_text(writer.last().unwrap());
    assert!(last.contains("Draft it again."), "{last}");
    assert_eq!(last.matches(ROSTER).count(), 1, "{last}");
    let verifier = bodies
        .iter()
        .map(|body| user_text(body))
        .filter(|text| text.contains("You are a completion verifier"))
        .collect::<Vec<_>>();
    assert!(!verifier.is_empty());
    for text in verifier {
        assert!(!text.contains(ROSTER), "{text}");
        assert!(text.contains("Draft it."), "{text}");
    }
    drop(lead);
}

/// An agent learns once that a recipient started mid-task, including one
/// started dynamically through workflow services, in its next user message;
/// before that it can't message it. Its tool definitions don't change.
#[tokio::test]
async fn a_running_agent_is_told_once_when_a_recipient_starts() {
    let workspace = Workspace::new();
    let started = Arc::new(tokio::sync::Notify::new());
    let server = ResponsesServer::start_async({
        let started = started.clone();
        move |_, body| {
            let started = started.clone();
            Box::pin(async move {
                if !user_text(&body).contains("Lead the work.") {
                    return None; // The worker keeps running.
                }
                let turns = tool_outputs(&body);
                if turns == 0 {
                    // Answer the lead's first request once the worker runs.
                    started.notified().await;
                }
                (turns < 2).then(|| {
                    Reply::Stream(tool_call_turn(&format!("p{turns}"), "probe", json!({})))
                })
            })
        }
    })
    .await;
    let workflow = Arc::new(
        builder_with(server.client(), &workspace, &["lead", "worker"])
            .agent_tool("lead", ProbeTool::new("probe", false))
            .workflow_services(crate::services::WorkflowServicesConfig::default())
            .build()
            .unwrap(),
    );
    let lead = workflow
        .spawn_agent("lead", "Lead the work.")
        .await
        .unwrap();
    server.wait_for_requests(1).await;
    let refused = send(&workflow, "lead", "worker", &workspace)
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains("has not started"), "{refused}");

    // A dynamic start, as run_workflow_agent makes.
    let services = workflow.context().workflow_services.get().unwrap().clone();
    let worker =
        tokio::spawn(async move { services.run_agent("lead", "worker", "Work on it.").await });
    server.wait_for_requests(2).await;
    started.notify_one();
    // The lead's three requests: before the notice, with it, and after.
    server.wait_for_requests(4).await;
    let lead_bodies = server
        .bodies()
        .into_iter()
        .filter(|body| is_request_of(body, "Lead the work."))
        .collect::<Vec<_>>();
    assert_eq!(lead_bodies.len(), 3);
    let notice =
        "Agent `worker` (role: worker role) is now running. You can message it with send_message.";
    let counts = lead_bodies
        .iter()
        .map(|body| user_text(body).matches(notice).count())
        .collect::<Vec<_>>();
    assert_eq!(counts, [0, 1, 1]);
    assert!(user_text(&lead_bodies[0]).contains("- `worker` (worker role): not_started"));
    assert!(
        lead_bodies
            .iter()
            .all(|body| tools(body) == tools(&lead_bodies[0]))
    );
    send(&workflow, "lead", "worker", &workspace).await.unwrap();

    workflow.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(10), lead.wait()).await;
    let _ = tokio::time::timeout(Duration::from_secs(10), worker).await;
}

/// Agents started together each see the others as running from the start,
/// and get no start notice.
#[tokio::test]
async fn agents_started_together_see_each_other_running() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        (tool_outputs(body) == 0).then(|| tool_call_turn("p", "probe", json!({})))
    })
    .await;
    let workflow = Arc::new(
        builder_with(server.client(), &workspace, &["lead", "worker"])
            .global_tool(ProbeTool::new("probe", false))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("lead", "Lead the work.".to_string()),
                    ("worker", "Do the work.".to_string()),
                ])
                .await
        }
    });
    server.wait_for_requests(4).await;
    for (prompt, other) in [("Lead the work.", "worker"), ("Do the work.", "lead")] {
        let bodies = server
            .bodies()
            .into_iter()
            .filter(|body| is_request_of(body, prompt))
            .collect::<Vec<_>>();
        assert_eq!(bodies.len(), 2, "{prompt}");
        assert!(
            user_text(&bodies[0]).contains(&format!("- `{other}` ({other} role): running")),
            "{prompt}"
        );
        assert!(
            bodies
                .iter()
                .all(|body| !user_text(body).contains("is now running")),
            "{prompt}"
        );
    }
    workflow.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(10), run).await;
}

/// Resuming restores statuses and what each run was told: a continued run
/// learns once of a recipient that started again after the resume, and
/// nothing twice.
#[tokio::test]
async fn a_resumed_run_learns_of_new_starts_once() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        let text = user_text(body);
        if !text.contains("Lead the work.") {
            None
        } else if text.contains("Carry on.") {
            Some(if tool_outputs(body) == 0 {
                tool_call_turn("p", "probe", json!({}))
            } else {
                complete_turn("again", "carried on")
            })
        } else {
            Some(complete_turn("done", "led"))
        }
    })
    .await;
    let persisted = |server: &ResponsesServer| {
        builder_with(server.client(), &workspace, &["lead", "worker", "idle"])
            .global_tool(ProbeTool::new("probe", false))
            .persistence(crate::persistence::PersistenceConfig::new(
                workspace.0.join(".run"),
            ))
    };
    let first = Arc::new(persisted(&server).build().unwrap());
    let worker = first.spawn_agent("worker", "Do the work.").await.unwrap();
    server.wait_for_requests(1).await;
    let lead = first.run_agent("lead", "Lead the work.").await.unwrap();
    assert!(user_text(&server.bodies()[1]).contains("- `worker` (worker role): running"));
    let checkpoint_dir = workspace.0.join(".run");
    let name = std::fs::read_to_string(checkpoint_dir.join("latest-checkpoint")).unwrap();
    let checkpoint = checkpoint_dir.join(name.trim());
    let state: crate::persistence::state::WorkflowState =
        serde_json::from_str(&std::fs::read_to_string(&checkpoint).unwrap()).unwrap();
    assert!(state.agent_registry.is_some());
    first.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(10), worker.wait()).await;
    drop(first);

    let resumed = Arc::new(persisted(&server).resume_from(&checkpoint).build().unwrap());
    assert_eq!(
        resumed.agent_status("lead"),
        Some(AgentStatus::Finished(AgentRunStatus::Completed))
    );
    // The worker was running when the checkpoint was saved.
    assert_eq!(
        resumed.agent_status("worker"),
        Some(AgentStatus::Finished(AgentRunStatus::Interrupted))
    );
    let worker = resumed.spawn_agent("worker", "Do the work.").await.unwrap();
    let before = server.requests();
    server.wait_for_requests(before + 1).await;
    let lead = resumed
        .result()
        .agent_results
        .into_iter()
        .find(|result| result.agent_run_id == lead.agent_run_id)
        .unwrap();
    let continued = resumed.continue_agent(&lead, "Carry on.").await;
    assert!(continued.is_ok(), "{continued:?}");
    let lead_bodies = server
        .bodies()
        .into_iter()
        .filter(|body| is_request_of(body, "Carry on."))
        .collect::<Vec<_>>();
    assert_eq!(lead_bodies.len(), 2);
    let notice = "Agent `worker` (role: worker role) is now running.";
    for body in &lead_bodies {
        assert_eq!(user_text(body).matches(notice).count(), 1);
        assert!(!user_text(body).contains("Agent `idle`"));
    }
    resumed.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(10), worker.wait()).await;
}

/// Message queues checkpointed by an early version (messages without ids,
/// sender task ids or attachments, a queue for an agent no longer
/// registered, and no agent registry) load on resume.
#[test]
fn message_queues_from_an_early_checkpoint_load_on_resume() {
    let workspace = Workspace::new();
    let run = workspace.0.join(".run");
    std::fs::create_dir_all(&run).unwrap();
    let checkpoint = run.join("checkpoint-legacy.json");
    let message = |from: &str, to: &str| json!({"from": from, "to": to, "content": "hello", "timestamp": "2026-09-01T12:00:00Z"});
    std::fs::write(
        &checkpoint,
        json!({
            "workflow_name": "messaging-test",
            "checkpoint_id": uuid::Uuid::new_v4(),
            "timestamp": "2026-09-01T12:00:00Z",
            "memory_snapshot": [],
            "message_queues": {
                "reviewer": [message("coder", "reviewer")],
                "retired": [message("coder", "retired")]
            },
            "budget_spent": 0.0
        })
        .to_string(),
    )
    .unwrap();
    let workflow = builder(&workspace, &["coder", "reviewer"])
        .persistence(crate::persistence::PersistenceConfig::new(&run))
        .resume_from(&checkpoint)
        .build()
        .unwrap();
    let pending = workflow.context().messages.pending("reviewer");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].from, "coder");
    assert_eq!(workflow.context().messages.snapshot().len(), 2);
    assert_eq!(workflow.context().message_recipients("reviewer"), ["coder"]);
    assert_eq!(
        workflow.agent_status("coder"),
        Some(AgentStatus::NotStarted)
    );
    let _ = WaitReason::Paused;
}
