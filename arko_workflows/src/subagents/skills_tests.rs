//! Skills given in memory reach the runs of a workflow, and the runs its
//! agents start, from the configuration of the agent each run is of.

use std::sync::Arc;

use arko_agents::skill::{EmbeddedSkill, SkillsConfig};
use serde_json::json;

use super::tests::{bodies_of, offered, result_of, serve, start, wait, within};
use super::*;
use crate::services::WorkflowServicesConfig;
use crate::test_support::{Workspace, agent_config};
use crate::{Workflow, WorkflowBuilder};

/// A skills configuration with an embedded skill that is always in the
/// instructions, `always`, and one loaded on demand, `on_demand`.
fn skills(always: &str, on_demand: &str) -> SkillsConfig {
    let skill = |name: &str, extra: &str| {
        EmbeddedSkill::new(
            name,
            format!(
                "---\nname: {name}\ndescription: The {name} skill.\n{extra}---\nThe {name} text.\n"
            ),
        )
    };
    SkillsConfig {
        embedded: vec![skill(always, "auto_load: true\n"), skill(on_demand, "")],
        ..Default::default()
    }
}

fn builder(
    server: &crate::test_support::ResponsesServer,
    workspace: &Workspace,
) -> WorkflowBuilder {
    let mut lead = agent_config("Lead");
    lead.skills_config = Some(skills("lead-always", "lead-review"));
    let mut worker = agent_config("Worker");
    worker.skills_config = Some(skills("worker-always", "worker-review"));
    Workflow::builder()
        .name("skills-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", lead)
        .agent("worker", worker)
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("worker"),
        )
        .subagents(SubagentsConfig::default())
}

fn load(name: &str) -> (&'static str, serde_json::Value) {
    ("load_skill", json!({ "name": name }))
}

/// A run has the embedded skills of its agent's configuration, the
/// auto-loaded ones in its instructions and the others through `load_skill`.
/// A run continued from its saved task has them too, and a subagent has those
/// of its own agent, not its lead's.
#[tokio::test]
async fn a_run_and_its_subagent_have_the_embedded_skills_of_their_agents() {
    let server = serve(vec![
        (
            "lead",
            vec![load("lead-review"), start("Review it."), wait()],
        ),
        ("r1", vec![load("worker-review")]),
    ])
    .await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = within(workflow.run_agent("lead", "Delegate the review."))
        .await
        .unwrap();
    assert_eq!(first.summary, "lead done");

    let lead = bodies_of(&server, "lead");
    assert!(lead[0].contains("BEGIN SKILL: lead-always"), "{}", lead[0]);
    assert!(lead[0].contains("The lead-always text."), "{}", lead[0]);
    assert!(offered(&lead[0]).contains(&"load_skill".to_string()));
    assert!(lead[0].contains("**lead-review**: The lead-review skill."));
    assert_eq!(
        result_of(&lead[1], "lead", 0),
        "--- BEGIN SKILL: lead-review ---\nThe lead-review text.\n--- END SKILL: lead-review ---"
    );
    assert!(!lead[0].contains("worker-always"), "{}", lead[0]);

    let worker = bodies_of(&server, "r1");
    assert!(
        worker[0].contains("The worker-always text."),
        "{}",
        worker[0]
    );
    assert!(offered(&worker[0]).contains(&"load_skill".to_string()));
    assert_eq!(
        result_of(&worker[1], "r1", 0),
        "--- BEGIN SKILL: worker-review ---\nThe worker-review text.\n--- END SKILL: worker-review ---"
    );
    assert!(!worker[0].contains("lead-always"), "{}", worker[0]);
    assert!(!worker[0].contains("lead-review"), "{}", worker[0]);

    // Continued from the task the run saved, the lead still has them.
    workflow.continue_agent(&first, "Once more.").await.unwrap();
    let last = server
        .bodies()
        .into_iter()
        .rfind(|body| body.contains("Once more."))
        .unwrap();
    assert!(last.contains("The lead-always text."), "{last}");
    assert!(offered(&last).contains(&"load_skill".to_string()));
}
