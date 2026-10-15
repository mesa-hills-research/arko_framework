//! What `build` requires and checks, and the tools each agent is offered.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arko_agents::TaskContext;

use super::*;
use crate::memory::tools::{
    LIST_MEMORIES_TOOL_NAME, READ_MEMORY_TOOL_NAME, WRITE_MEMORY_TOOL_NAME,
};
use crate::memory::{AccessScope, MemoryTier, WriteMode};
use crate::test_support::{Workspace, agent_config};
use crate::verification::CompletionVerifierConfig;
use crate::voting::{BallotConfig, TieBreak, VoteOption};

fn builder(workspace: &std::path::Path, agents: &[&str]) -> WorkflowBuilder {
    let mut builder = Workflow::builder()
        .name("build-test")
        .workspace(workspace)
        .llm_client(Arc::new(LLMClient::new()));
    for name in agents {
        builder = builder.agent(*name, agent_config(&format!("Friendly {name}")));
    }
    builder
}

fn invalid(result: Result<Workflow, WorkflowError>) -> String {
    match result {
        Err(WorkflowError::BuilderInvalid(message)) => message,
        Err(error) => panic!("another error: {error}"),
        Ok(_) => panic!("the workflow built"),
    }
}

fn tier(name: &str, readers: AccessScope, writers: AccessScope) -> MemoryTier {
    MemoryTier {
        name: name.into(),
        content: String::new(),
        readers,
        writers,
        write_mode: WriteMode::AppendOnly,
        entries: Vec::new(),
        max_bytes: None,
    }
}

fn only(agents: &[&str]) -> AccessScope {
    AccessScope::Only(agents.iter().map(|agent| agent.to_string()).collect())
}

#[test]
fn resume_config_comparison_ignores_workflow_tools_and_model_snapshots() {
    let current = agent_config("Friendly agent");
    let mut saved = current.clone();
    saved
        .tool_config
        .available_tool_names
        .push("workflow_tool".into());
    let mut workflow_tools = HashSet::from(["workflow_tool".to_string()]);
    let current_json = super::normalized_resume_agent_config(&current, &workflow_tools).unwrap();
    let saved_json = super::normalized_resume_agent_config(&saved, &workflow_tools).unwrap();
    assert_eq!(current_json, saved_json);

    workflow_tools.clear();
    saved.instructions.push_str(" Changed.");
    let current_json = super::normalized_resume_agent_config(&current, &workflow_tools).unwrap();
    let saved_json = super::normalized_resume_agent_config(&saved, &workflow_tools).unwrap();
    assert_ne!(current_json, saved_json);
}

/// The tools of `agent`'s registry, which its prepared configuration also
/// lists.
fn offered(workflow: &Workflow, agent: &str, tools: &[&str]) -> Vec<bool> {
    let context = workflow.context();
    let registry = workflow.engine.build_tool_registry(agent, context, 0);
    let config = workflow.engine.prepare_agent_config(
        agent,
        context.agent_configs.get(agent).unwrap(),
        context,
    );
    tools
        .iter()
        .map(|tool| {
            let registered = registry.get(tool).is_some();
            let listed = config
                .tool_config
                .available_tool_names
                .iter()
                .any(|name| name == tool);
            assert_eq!(registered, listed, "{agent}: {tool}");
            registered
        })
        .collect()
}

/// A verifier must name the agents it verifies: one without a target fails
/// to build, and one with targets gates only them (never its own agent).
#[test]
fn a_verifier_needs_an_explicit_target() {
    let workspace = Workspace::new();
    let message = invalid(
        builder(&workspace.0, &["coder", "reviewer", "planner"])
            .completion_verifier(CompletionVerifierConfig::correctness("check", "reviewer"))
            .build(),
    );
    assert!(
        message.contains("completion verifier 'check' has no target"),
        "{message}"
    );

    let verifier = CompletionVerifierConfig::correctness("check", "reviewer")
        .target("coder")
        .target("reviewer");
    assert!(verifier.applies_to("coder"));
    assert!(!verifier.applies_to("planner"));
    assert!(!verifier.applies_to("reviewer"), "never its own agent");
}

/// A verifier reviews `complete_task` calls, and gives its verdict with
/// one: an agent it reviews, and its own agent, can't end their tasks with a
/// reply in text. Other agents can.
#[test]
fn text_endings_are_refused_where_a_verifier_reviews_or_runs() {
    let workspace = Workspace::new();
    let replying = || {
        let mut config = agent_config("Friendly replier");
        config.tool_config.text_reply_ends_task = true;
        config
    };
    let verifier = || CompletionVerifierConfig::correctness("check", "reviewer").target("coder");
    let message = invalid(
        builder(&workspace.0, &["reviewer", "planner"])
            .agent("coder", replying())
            .completion_verifier(verifier())
            .build(),
    );
    assert!(
        message.contains("agent 'coder' ends its task with a reply in text")
            && message.contains("completion verifier 'check' can't review"),
        "{message}"
    );
    let message = invalid(
        builder(&workspace.0, &["coder", "planner"])
            .agent("reviewer", replying())
            .completion_verifier(verifier())
            .build(),
    );
    assert!(
        message.contains("completion verifier 'check' runs agent 'reviewer'"),
        "{message}"
    );
    builder(&workspace.0, &["coder", "reviewer"])
        .agent("planner", replying())
        .completion_verifier(verifier())
        .build()
        .unwrap();
}

/// Memory tools follow each agent's access: `read_memory` for an agent that
/// can read a tier, `write_memory` for one that can write one,
/// `list_memories` for either, none for an agent without access; and
/// `list_memories` names only the tiers the agent can access.
#[tokio::test]
async fn memory_tools_follow_each_agents_access() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace.0, &["reader", "writer", "outsider"])
        .memory(
            crate::memory::MemoryConfig::new()
                .tier(tier(
                    "notes",
                    only(&["reader", "writer"]),
                    only(&["writer"]),
                ))
                .tier(tier("drafts", only(&["writer"]), only(&["writer"])))
                .tier(tier("inbox", AccessScope::None, only(&["reader"]))),
        )
        .build()
        .unwrap();
    let tools = [
        READ_MEMORY_TOOL_NAME,
        WRITE_MEMORY_TOOL_NAME,
        LIST_MEMORIES_TOOL_NAME,
    ];
    assert_eq!(offered(&workflow, "reader", &tools), [true, true, true]);
    assert_eq!(offered(&workflow, "writer", &tools), [true, true, true]);
    assert_eq!(
        offered(&workflow, "outsider", &tools),
        [false, false, false]
    );

    let list = |agent: &str| {
        let registry = workflow
            .engine
            .build_tool_registry(agent, workflow.context(), 0);
        registry.get(LIST_MEMORIES_TOOL_NAME).unwrap()
    };
    let task = TaskContext::new(
        uuid::Uuid::new_v4(),
        "Friendly reader",
        "read",
        &workspace.0,
    );
    let listed = list("reader").execute(&task, HashMap::new()).await.unwrap();
    assert_eq!(
        listed,
        "Available memory tiers:\n- inbox (write-only)\n- notes (read-only)\n"
    );
    assert!(!listed.contains("drafts"), "{listed}");

    // A read-only agent isn't offered write_memory.
    let workflow = builder(&workspace.0, &["reader"])
        .memory(crate::memory::MemoryConfig::new().tier(tier(
            "notes",
            AccessScope::All,
            AccessScope::None,
        )))
        .build()
        .unwrap();
    assert_eq!(offered(&workflow, "reader", &tools), [true, false, true]);
}

/// Every agent-name list is checked at build: memory access sets, workflow
/// services callers and targets, and voters. Tier names must be unique.
#[test]
fn agent_name_lists_are_validated_at_build() {
    let workspace = Workspace::new();
    let cases = [
        (
            builder(&workspace.0, &["coder"]).memory(
                crate::memory::MemoryConfig::new().tier(tier(
                    "notes",
                    only(&["coder", "cdoer"]),
                    AccessScope::All,
                )),
            ),
            "memory tier 'notes' readers name unknown agent 'cdoer'",
        ),
        (
            builder(&workspace.0, &["coder"]).memory(
                crate::memory::MemoryConfig::new().tier(tier(
                    "notes",
                    AccessScope::All,
                    only(&["reviewer"]),
                )),
            ),
            "memory tier 'notes' writers name unknown agent 'reviewer'",
        ),
        (
            builder(&workspace.0, &["coder"]).memory(
                crate::memory::MemoryConfig::new()
                    .global_tier("notes")
                    .global_tier("notes"),
            ),
            "duplicate memory tier 'notes'",
        ),
        (
            builder(&workspace.0, &["coder"]).enable_voting(["coder", "critic"]),
            "voting names unknown agent 'critic'",
        ),
        (
            builder(&workspace.0, &["coder"]).workflow_services(
                crate::services::WorkflowServicesConfig::default().allow_caller("orchestrator"),
            ),
            "workflow services caller 'orchestrator' is not a registered agent",
        ),
    ];
    for (builder, expected) in cases {
        let message = invalid(builder.build());
        assert!(message.contains(expected), "{message}");
    }
}

/// `cast_vote` is offered to the named voters only, and a ballot's
/// eligible voters must be among them (with none named, every voter is).
#[test]
fn cast_vote_is_offered_to_voters_only() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace.0, &["one", "two", "clerk"])
        .enable_voting(["one", "two"])
        .build()
        .unwrap();
    let vote = [crate::voting::CAST_VOTE_TOOL_NAME];
    assert_eq!(offered(&workflow, "one", &vote), [true]);
    assert_eq!(offered(&workflow, "clerk", &vote), [false]);
    let ballot = |eligible: &[&str]| BallotConfig {
        question: "Which?".into(),
        options: vec![
            VoteOption {
                id: "a".into(),
                description: "A".into(),
            },
            VoteOption {
                id: "b".into(),
                description: "B".into(),
            },
        ],
        eligible_voters: eligible.iter().map(|voter| voter.to_string()).collect(),
        quorum: 1,
        consensus_threshold: 0.5,
        weights: HashMap::new(),
        tie_break: TieBreak::RejectTie,
    };
    assert!(workflow.begin_ballot(ballot(&["one"])).is_ok());
    assert!(workflow.begin_ballot(ballot(&[])).is_ok());
    assert!(matches!(
        workflow.begin_ballot(ballot(&["one", "clerk"])),
        Err(WorkflowError::Deliberation(message)) if message.contains("'clerk' is not offered cast_vote")
    ));
}

/// The workspace is stored absolute, with `.` and `..` resolved, without
/// resolving symbolic links or producing a Windows verbatim path.
#[test]
fn the_workspace_is_normalized_to_an_absolute_path() {
    let workspace = Workspace::new();
    std::fs::create_dir_all(workspace.0.join("project")).unwrap();
    let dotted = workspace
        .0
        .join("project")
        .join("..")
        .join(".")
        .join("project");
    let workflow = builder(&dotted, &["coder"]).build().unwrap();
    assert_eq!(workflow.context().workspace, workspace.0.join("project"));

    let current = builder(std::path::Path::new("."), &["coder"])
        .build()
        .unwrap();
    let workspace_path = &current.context().workspace;
    assert!(workspace_path.is_absolute());
    assert_eq!(*workspace_path, std::env::current_dir().unwrap());
    assert!(!workspace_path.to_string_lossy().starts_with(r"\\?\"));

    let missing = invalid(builder(&workspace.0.join("missing"), &["coder"]).build());
    assert!(
        missing.contains("does not exist or is not a directory"),
        "{missing}"
    );
}

/// How each workflow tool declares itself to the agent runtime: read-only
/// tools may run beside other parallel-safe calls; tools whose order
/// matters (a message, a write, a vote, a delegated run) or that wait
/// don't. The waiting and delegating tools opt out of the host tool time
/// limit, and a delegated run isn't interrupted mid-call (the workflow
/// winds the delegated agent down itself).
#[test]
fn workflow_tools_declare_how_they_run() {
    let workspace = Workspace::new();
    let workflow = builder(&workspace.0, &["lead", "worker"])
        .memory(crate::memory::MemoryConfig::new().global_tier("notes"))
        .workflow_services(
            crate::services::WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("worker"),
        )
        .enable_voting(["lead"])
        .build()
        .unwrap();
    let registry = workflow
        .engine
        .build_tool_registry("lead", workflow.context(), 0);
    for (tool, parallel_safe, host_timeout, interruptible) in [
        (READ_MEMORY_TOOL_NAME, true, true, true),
        (LIST_MEMORIES_TOOL_NAME, true, true, true),
        (WRITE_MEMORY_TOOL_NAME, false, true, true),
        ("list_recipients", true, true, true),
        ("send_message", false, true, true),
        ("wait_for_message", false, false, true),
        ("run_workflow_agent", false, false, false),
        ("cast_vote", false, true, true),
    ] {
        let tool_ref = registry.get(tool).unwrap_or_else(|| panic!("{tool}"));
        assert_eq!(
            (
                tool_ref.definition().parallel_safe,
                tool_ref.uses_host_timeout(),
                tool_ref.interruptible()
            ),
            (parallel_safe, host_timeout, interruptible),
            "{tool}"
        );
    }
}
