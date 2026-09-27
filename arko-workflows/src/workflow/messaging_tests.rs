//! Which agents are offered `send_message`, and whom they may message.

use std::collections::HashMap;
use std::sync::Arc;

use arko_agents::{TaskContext, Tool, ToolError};
use serde_json::json;

use super::*;
use crate::messaging::tools::SEND_MESSAGE_TOOL_NAME;
use crate::test_support::{ProbeTool, Workspace, agent_config};

fn builder(workspace: &Workspace, agents: &[&str]) -> WorkflowBuilder {
    let mut builder = Workflow::builder()
        .name("messaging-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(LLMClient::new()));
    for agent in agents {
        builder = builder.agent(*agent, agent_config(&format!("Friendly {agent}")));
    }
    builder
}

/// The workflow's `send_message` tool for `agent`, if it is offered one;
/// the agent's prepared configuration must agree.
fn send_message(workflow: &Workflow, agent: &str) -> Option<Arc<dyn Tool>> {
    let context = workflow.context();
    let registry = workflow.engine.build_tool_registry(agent, context);
    let config = workflow.engine.prepare_agent_config(
        agent,
        context.agent_configs.get(agent).unwrap(),
        context,
    );
    let tool = registry.get(SEND_MESSAGE_TOOL_NAME);
    assert_eq!(
        tool.is_some(),
        config
            .tool_config
            .available_tool_names
            .iter()
            .any(|name| name == SEND_MESSAGE_TOOL_NAME),
        "{agent}"
    );
    tool
}

fn recipients(workflow: &Workflow, agent: &str) -> String {
    let description = send_message(workflow, agent)
        .unwrap()
        .definition()
        .description;
    description
        .split("Valid recipients: ")
        .nth(1)
        .unwrap()
        .trim_end_matches('.')
        .to_string()
}

async fn send(
    workflow: &Workflow,
    from: &str,
    to: &str,
    workspace: &Workspace,
) -> Result<String, ToolError> {
    let tool = send_message(workflow, from).unwrap();
    let context = TaskContext {
        task_id: uuid::Uuid::new_v4(),
        agent_name: "Friendly sender",
        initial_task_prompt: "coordinate",
        current_working_directory: &workspace.0,
    };
    let params = HashMap::from([
        ("to".to_string(), json!(to)),
        ("content".to_string(), json!("hello")),
    ]);
    tool.execute(&context, params).await
}

#[test]
fn a_single_agent_is_not_offered_send_message_and_the_name_is_free() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["coder"]).build().unwrap();
    assert!(send_message(&workflow, "coder").is_none());

    // A host tool may take the name when no agent is offered the workflow's.
    let own = ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false);
    let workflow = builder(&workspace, &["coder"])
        .global_tool(own)
        .build()
        .unwrap();
    let registry = workflow
        .engine
        .build_tool_registry("coder", workflow.context());
    assert_eq!(
        registry
            .get(SEND_MESSAGE_TOOL_NAME)
            .unwrap()
            .definition()
            .description,
        "Test probe."
    );
}

#[tokio::test]
async fn each_agent_may_message_every_other_agent_but_itself_by_default() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["coder", "reviewer", "writer"])
        .build()
        .unwrap();
    assert_eq!(recipients(&workflow, "coder"), "reviewer, writer");
    assert_eq!(recipients(&workflow, "reviewer"), "coder, writer");
    assert!(matches!(
        send(&workflow, "coder", "coder", &workspace).await,
        Err(ToolError::ParameterValidation { .. })
    ));
    send(&workflow, "coder", "writer", &workspace)
        .await
        .unwrap();
    assert_eq!(workflow.context().messages.pending("writer").len(), 1);
    // The name is the workflow's while an agent is offered it.
    assert!(matches!(
        builder(&workspace, &["coder", "reviewer"])
            .global_tool(ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
            .build(),
        Err(WorkflowError::BuilderInvalid(message)) if message.contains("reserved")
    ));
}

#[test]
fn disabled_messaging_offers_no_agent_send_message() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace, &["coder", "reviewer"])
        .disable_messaging()
        .global_tool(ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
        .build()
        .unwrap();
    assert!(workflow.context().message_recipients("coder").is_empty());
    let registry = workflow
        .engine
        .build_tool_registry("reviewer", workflow.context());
    assert_eq!(
        registry
            .get(SEND_MESSAGE_TOOL_NAME)
            .unwrap()
            .definition()
            .description,
        "Test probe."
    );
}

/// A restricted sender sees and reaches only its recipients; an
/// unrestricted one reaches every other agent; one permitted no one isn't
/// offered the tool. Agents started through workflow services get their
/// tools from the same registry and follow the same policy.
#[tokio::test]
async fn a_policy_limits_whom_each_agent_sees_and_reaches() {
    let workspace = Workspace::new();
    let policy = MessagingPolicy::all()
        .can_message("worker_a", ["lead"])
        .cannot_message("auditor");
    let workflow = builder(&workspace, &["lead", "worker_a", "worker_b", "auditor"])
        .messaging(policy.clone())
        // An agent not offered the workflow's tool may have its own of that name.
        .agent_tool("auditor", ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
        .build()
        .unwrap();
    assert_eq!(recipients(&workflow, "worker_a"), "lead");
    assert_eq!(recipients(&workflow, "lead"), "auditor, worker_a, worker_b");
    assert_eq!(recipients(&workflow, "worker_b"), "auditor, lead, worker_a");
    assert!(workflow.context().message_recipients("auditor").is_empty());

    for to in ["worker_b", "auditor", "worker_a", "nobody"] {
        let error = send(&workflow, "worker_a", to, &workspace)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("valid recipients: lead"),
            "{error}"
        );
    }
    assert!(workflow.context().messages.pending("worker_b").is_empty());
    send(&workflow, "worker_a", "lead", &workspace)
        .await
        .unwrap();
    send(&workflow, "lead", "worker_b", &workspace)
        .await
        .unwrap();
    assert_eq!(workflow.context().messages.pending("lead").len(), 1);
    assert_eq!(workflow.context().messages.pending("worker_b").len(), 1);

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
            .agent_tool("worker_a", ProbeTool::new(SEND_MESSAGE_TOOL_NAME, false))
            .build(),
        Err(WorkflowError::BuilderInvalid(message)) if message.contains("reserved")
    ));
}

/// Message queues checkpointed by an early version (messages without ids,
/// sender task ids or attachments, and a queue for an agent no longer
/// registered) load on resume.
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
    assert_eq!(recipients(&workflow, "reviewer"), "coder");
}
