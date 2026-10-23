//! Skills given in memory reach the runs of a workflow, and the runs its
//! agents start, from the configuration of the agent each run is of. A saved
//! run takes the ones its agent is registered with now.

use std::sync::Arc;

use arko_agents::skill::{EmbeddedSkill, SkillsConfig};
use serde_json::{Value, json};

use super::tests::{bodies_of, offered, result_of, role, serve, start, wait, within};
use super::*;
use crate::persistence::PersistenceConfig;
use crate::services::WorkflowServicesConfig;
use crate::test_support::{
    Reply, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::{Workflow, WorkflowBuilder};

/// An embedded skill `name` whose text is "The `name` `version`."
fn skill(name: &str, always: bool, version: &str) -> EmbeddedSkill {
    let extra = if always { "auto_load: true\n" } else { "" };
    EmbeddedSkill::new(
        name,
        format!(
            "---\nname: {name}\ndescription: The {name} skill.\n{extra}---\nThe {name} {version}.\n"
        ),
    )
}

/// A skills configuration with an embedded skill that is always in the
/// instructions, `always`, and one loaded on demand, `on_demand`.
fn skills(always: &str, on_demand: &str) -> SkillsConfig {
    SkillsConfig {
        embedded: vec![skill(always, true, "text"), skill(on_demand, false, "text")],
        ..Default::default()
    }
}

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    builder_with(
        server,
        workspace,
        Some(skills("lead-always", "lead-review")),
        Some(skills("worker-always", "worker-review")),
    )
}

/// A workflow where `lead` and `worker` have the given skills
/// configurations.
fn builder_with(
    server: &ResponsesServer,
    workspace: &Workspace,
    lead_skills: Option<SkillsConfig>,
    worker_skills: Option<SkillsConfig>,
) -> WorkflowBuilder {
    let mut lead = agent_config("Lead");
    lead.skills_config = lead_skills;
    let mut worker = agent_config("Worker");
    worker.skills_config = worker_skills;
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
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
}

fn load(name: &str) -> (&'static str, Value) {
    ("load_skill", json!({ "name": name }))
}

/// The result a request carries for the call `call_id` of the server's script.
fn output_of(body: &str, call_id: &str) -> String {
    let body: Value = serde_json::from_str(body).unwrap();
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| {
            item["type"] == "function_call_output" && item["call_id"] == format!("call_{call_id}")
        })
        .and_then(|item| item["output"].as_str())
        .unwrap_or_default()
        .to_string()
}

fn latest_checkpoint(workspace: &Workspace) -> std::path::PathBuf {
    let run = workspace.0.join(".run");
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    run.join(name.trim())
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

/// A server where the lead completes at once the first time, and after
/// "Go on." loads the skills `review`, `added` and `removed`, starts a
/// worker and waits for it. The worker loads `worker-review`.
async fn serve_after_a_resume() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        let sent = |id: &str| body.contains(&format!("call_{id}"));
        let (name, arguments) = if role(body) == "r1" {
            if sent("w1") {
                return Some(complete_turn("w2", "worker done"));
            }
            ("w1", load("worker-review"))
        } else if !body.contains("Go on.") {
            return Some(complete_turn("first", "lead first"));
        } else if !sent("c1") {
            ("c1", load("review"))
        } else if !sent("c2") {
            ("c2", load("added"))
        } else if !sent("c3") {
            ("c3", load("removed"))
        } else if !sent("c4") {
            ("c4", start("Review it."))
        } else if !sent("c5") {
            ("c5", wait())
        } else {
            return Some(complete_turn("last", "lead done"));
        };
        Some(tool_call_turn(name, arguments.0, arguments.1))
    })
    .await
}

/// A run saved while its agent had some embedded skills resumes when the
/// agent is registered with others (a changed text, one removed and one
/// added), and goes on with the skills it is registered with now. A
/// subagent it starts has them too. The saved run doesn't hold their text.
#[tokio::test]
async fn a_resumed_run_takes_the_embedded_skills_its_agent_is_registered_with() {
    let server = serve_after_a_resume().await;
    let workspace = Workspace::new();
    let before = SkillsConfig {
        embedded: vec![
            skill("always", true, "version one"),
            skill("review", false, "version one"),
            skill("removed", false, "version one"),
        ],
        ..Default::default()
    };
    let workflow = builder_with(&server, &workspace, Some(before), None)
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "Hello.")).await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);
    let saved = std::fs::read_to_string(latest_checkpoint(&workspace)).unwrap();
    assert!(
        !saved.contains("version one"),
        "the checkpoint holds the text"
    );

    let after = SkillsConfig {
        embedded: vec![
            skill("always", true, "version two"),
            skill("review", false, "version two"),
            skill("added", false, "version two"),
        ],
        ..Default::default()
    };
    let resumed = builder_with(
        &server,
        &workspace,
        Some(after),
        Some(skills("worker-always", "worker-review")),
    )
    .resume_from(latest_checkpoint(&workspace))
    .build()
    .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    let result = within(resumed.continue_agent(&restored, "Go on."))
        .await
        .unwrap();
    assert_eq!(result.summary, "lead done");

    let last = bodies_of(&server, "lead").pop().unwrap();
    assert!(offered(&last).contains(&"load_skill".to_string()));
    assert_eq!(
        output_of(&last, "c1"),
        "--- BEGIN SKILL: review ---\nThe review version two.\n--- END SKILL: review ---"
    );
    assert_eq!(
        output_of(&last, "c2"),
        "--- BEGIN SKILL: added ---\nThe added version two.\n--- END SKILL: added ---"
    );
    assert!(
        output_of(&last, "c3").starts_with("Skill 'removed' not found. Available skills:"),
        "{}",
        output_of(&last, "c3")
    );
    // The run keeps the system prompt it started with, so the skill in it
    // and the list of skills stay as they were. `load_skill` has the new text.
    assert!(last.contains("The always version one."), "{last}");
    assert!(!last.contains("The always version two."), "{last}");
    // A subagent started after the resume has its agent's skills.
    let worker = bodies_of(&server, "r1");
    assert!(
        worker[0].contains("The worker-always text."),
        "{}",
        worker[0]
    );
    assert_eq!(
        output_of(&worker[1], "w1"),
        "--- BEGIN SKILL: worker-review ---\nThe worker-review text.\n--- END SKILL: worker-review ---"
    );
}

/// A run saved by an agent that had no skills configuration resumes when
/// the agent is registered with embedded skills, and has them.
#[tokio::test]
async fn a_run_saved_without_skills_resumes_with_the_embedded_ones() {
    let server = serve_after_a_resume().await;
    let workspace = Workspace::new();
    let workflow = builder_with(&server, &workspace, None, None)
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "Hello.")).await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let after = SkillsConfig {
        embedded: vec![skill("review", false, "version two")],
        ..Default::default()
    };
    let resumed = builder_with(&server, &workspace, Some(after), None)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    within(resumed.continue_agent(&restored, "Go on."))
        .await
        .unwrap();
    let last = bodies_of(&server, "lead").pop().unwrap();
    assert_eq!(
        output_of(&last, "c1"),
        "--- BEGIN SKILL: review ---\nThe review version two.\n--- END SKILL: review ---"
    );

    // A change to the other fields of the skills configuration still
    // refuses the resume.
    let folders = SkillsConfig {
        global_sources: vec![arko_agents::skill::SkillSource {
            path: "/elsewhere".into(),
            include: None,
            exclude: None,
        }],
        ..Default::default()
    };
    let error = builder_with(&server, &workspace, Some(folders), None)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("skills_config"), "{error}");
}

/// A server where the lead completes unless it is told "Go on." (then it
/// loads `review`) or "Load it two." (then it loads `review` again).
async fn serve_loads() -> ResponsesServer {
    ResponsesServer::start(|_, body| {
        Some(
            if body.contains("Load it two.") && !body.contains("call_c2") {
                tool_call_turn("c2", "load_skill", json!({ "name": "review" }))
            } else if body.contains("Go on.") && !body.contains("call_c1") {
                tool_call_turn("c1", "load_skill", json!({ "name": "review" }))
            } else {
                complete_turn("done", "lead done")
            },
        )
    })
    .await
}

fn review(version: &str) -> Option<SkillsConfig> {
    Some(SkillsConfig {
        embedded: vec![skill("review", false, version)],
        ..Default::default()
    })
}

/// A run the host reconfigured with embedded skills keeps the ones it was
/// given while the process lives, and has the registered agent's after a
/// save and a resume, since a saved configuration doesn't hold the text.
#[tokio::test]
async fn a_reconfigured_run_has_embedded_skills_across_a_save_and_a_resume() {
    let server = serve_loads().await;
    let workspace = Workspace::new();
    let workflow = builder_with(&server, &workspace, review("registered"), None)
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "Hello.")).await.unwrap();
    let mut given = agent_config("Lead");
    given.skills_config = review("given");
    let second = within(workflow.continue_agent_with(
        &first,
        "Reconfigured.",
        crate::ContinueOptions::new().reconfigure(given),
    ))
    .await
    .unwrap();
    assert!(second.task.reconfigured);

    // The skills the host gave stay, and the registered agent's don't replace them.
    within(workflow.continue_agent(&second, "Go on."))
        .await
        .unwrap();
    let last = bodies_of(&server, "lead").pop().unwrap();
    assert!(
        output_of(&last, "c1").contains("The review given."),
        "{}",
        output_of(&last, "c1")
    );
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    // After the resume, the run has the registered agent's skills, and
    // `load_skill` is still offered.
    let resumed = builder_with(&server, &workspace, review("registered"), None)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    assert!(restored.task.reconfigured);
    within(resumed.continue_agent(&restored, "Load it two."))
        .await
        .unwrap();
    let last = bodies_of(&server, "lead").pop().unwrap();
    assert!(offered(&last).contains(&"load_skill".to_string()));
    assert!(
        output_of(&last, "c2").contains("The review registered."),
        "{}",
        output_of(&last, "c2")
    );
}

/// A run the host reconfigured without a skills configuration stays
/// without one after a save and a resume, though its agent is registered
/// with embedded skills.
#[tokio::test]
async fn a_reconfigured_run_without_skills_stays_without_them_after_a_resume() {
    let server = serve_loads().await;
    let workspace = Workspace::new();
    let workflow = builder_with(&server, &workspace, review("registered"), None)
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "Hello.")).await.unwrap();
    let mut given = agent_config("Lead");
    given.skills_config = None;
    within(workflow.continue_agent_with(
        &first,
        "Reconfigured.",
        crate::ContinueOptions::new().reconfigure(given),
    ))
    .await
    .unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let resumed = builder_with(&server, &workspace, review("registered"), None)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    within(resumed.continue_agent(&restored, "Load it two."))
        .await
        .unwrap();
    let requests = bodies_of(&server, "lead");
    assert!(
        !offered(requests.last().unwrap()).contains(&"load_skill".to_string()),
        "{}",
        requests.last().unwrap()
    );
}

/// The prompt cache refresh of a run loaded from a save offers the
/// registered agent's skills, as the run's next request will.
#[tokio::test]
async fn a_refresh_of_a_saved_run_offers_the_registered_skills() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start_async(|index, _| {
        let reply = if index == 0 {
            Reply::Stream(complete_turn("first", "lead first"))
        } else {
            Reply::Status {
                status: 200,
                retry_after: None,
                body: json!({
                    "id": "resp_r", "object": "response", "created_at": 0, "status": "completed",
                    "model": "gpt-5.6-luna",
                    "output": [{"type": "message", "id": "m1", "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
                    "usage": {"input_tokens": 1200, "output_tokens": 1},
                })
                .to_string(),
            }
        };
        Box::pin(async move { Some(reply) })
    })
    .await;
    let workflow = builder_with(&server, &workspace, review("registered"), None)
        .build()
        .unwrap();
    let first = within(workflow.run_agent("lead", "Hello.")).await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let resumed = builder_with(&server, &workspace, review("registered"), None)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    within(resumed.refresh_prompt_cache(&restored))
        .await
        .unwrap();
    let refresh = server.bodies().pop().unwrap();
    assert!(
        offered(&refresh).contains(&"load_skill".to_string()),
        "{refresh}"
    );
}

/// A paused run whose task came from a save (it has no embedded skills)
/// goes on with its agent's registered ones when it is resumed.
#[tokio::test]
async fn a_resumed_pause_takes_the_registered_embedded_skills() {
    use crate::policy::{AgentResume, ErrorRecoveryPolicy};
    let workspace = Workspace::new();
    let server = ResponsesServer::start_async(|index, body| {
        let reply = if index == 0 {
            Reply::Status {
                status: 503,
                retry_after: None,
                body: json!({"error": {"message": "synthetic 503", "type": "server_error"}})
                    .to_string(),
            }
        } else if !body.contains("call_c1") {
            Reply::Stream(tool_call_turn(
                "c1",
                "load_skill",
                json!({ "name": "review" }),
            ))
        } else {
            Reply::Stream(complete_turn("done", "lead done"))
        };
        let reply = Some(reply);
        Box::pin(async move { reply })
    })
    .await;
    let mut lead = agent_config("Lead");
    lead.llm_error_policy =
        arko_agents::LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 1 };
    lead.skills_config = review("registered");
    let workflow = Workflow::builder()
        .name("skills-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", lead)
        .error_recovery_policy(ErrorRecoveryPolicy::ReturnToCaller)
        .build()
        .unwrap();
    let error = within(workflow.run_agent("lead", "Hello."))
        .await
        .unwrap_err();
    let crate::WorkflowError::AgentPaused { task_id, .. } = error else {
        panic!("{error:?}");
    };
    // As if loaded from a save, which leaves the embedded skills out.
    let mut paused = workflow.agent_result(task_id).unwrap();
    paused.task = serde_json::from_value(serde_json::to_value(&paused.task).unwrap()).unwrap();
    assert!(
        paused
            .task
            .agent_config
            .skills_config
            .as_ref()
            .unwrap()
            .embedded
            .is_empty()
    );
    within(workflow.resume_agent(&paused, AgentResume::ContinueAfterError { note: None }))
        .await
        .unwrap();
    let bodies = server.bodies();
    assert!(
        offered(&bodies[1]).contains(&"load_skill".to_string()),
        "{}",
        bodies[1]
    );
    assert!(
        output_of(bodies.last().unwrap(), "c1").contains("The review registered."),
        "{}",
        bodies.last().unwrap()
    );
}

/// The host replaces an agent's skills while a subagent of it runs. That
/// subagent keeps its instructions and its `load_skill`, and the lead's
/// conversation keeps its own. A subagent that starts after the change, and a
/// new conversation of the lead, have the new skills.
#[tokio::test]
async fn skills_set_while_runs_go_reach_only_the_runs_that_start_after() {
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let server = ResponsesServer::start_async({
        let gate = gate.clone();
        move |_, body| {
            let gate = gate.clone();
            Box::pin(async move {
                let sent = |id: &str| body.contains(&format!("call_{id}"));
                let (name, arguments) = match role(&body).as_str() {
                    "r1" if !sent("w1") => {
                        // The first subagent's first request waits for the change.
                        gate.acquire().await.unwrap().forget();
                        ("w1", load("worker-old-review"))
                    }
                    "r2" if !sent("w2") => ("w2", load("worker-new-review")),
                    "r1" | "r2" => return Some(Reply::Stream(complete_turn("w", "worker done"))),
                    _ if body.contains("Hello again.") => {
                        return Some(Reply::Stream(complete_turn("again", "lead again")));
                    }
                    _ if body.contains("Go on.") && !sent("b1") => ("b1", start("Review again.")),
                    _ if body.contains("Go on.") && !sent("b2") => ("b2", wait()),
                    _ if !body.contains("Go on.") && !sent("a1") => ("a1", start("Review it.")),
                    _ if !body.contains("Go on.") && !sent("a2") => ("a2", wait()),
                    _ => return Some(Reply::Stream(complete_turn("lead", "lead done"))),
                };
                Some(Reply::Stream(tool_call_turn(
                    name,
                    arguments.0,
                    arguments.1,
                )))
            })
        }
    })
    .await;
    let workspace = Workspace::new();
    let workflow = builder_with(
        &server,
        &workspace,
        Some(skills("lead-old", "lead-old-review")),
        Some(skills("worker-old", "worker-old-review")),
    )
    .build()
    .unwrap();
    let host = async {
        super::tests::until(|| !bodies_of(&server, "r1").is_empty()).await;
        workflow
            .set_agent_skills("lead", Some(skills("lead-new", "lead-new-review")))
            .unwrap();
        workflow
            .set_agent_skills("worker", Some(skills("worker-new", "worker-new-review")))
            .unwrap();
        gate.add_permits(1);
    };
    let (first, ()) = tokio::join!(within(workflow.run_agent("lead", "Hello.")), host);
    within(workflow.continue_agent(&first.unwrap(), "Go on."))
        .await
        .unwrap();
    within(workflow.run_agent("lead", "Hello again."))
        .await
        .unwrap();
    assert!(matches!(
        workflow.set_agent_skills("nobody", None),
        Err(crate::WorkflowError::AgentNotFound(_))
    ));

    let old = bodies_of(&server, "r1");
    assert!(old.iter().all(|body| body.contains("The worker-old text.")));
    assert!(old.iter().all(|body| !body.contains("worker-new")));
    assert!(output_of(&old[1], "w1").contains("The worker-old-review text."));
    let new = bodies_of(&server, "r2");
    assert!(new.iter().all(|body| body.contains("The worker-new text.")));
    assert!(new.iter().all(|body| !body.contains("worker-old")));
    assert!(output_of(&new[1], "w2").contains("The worker-new-review text."));

    // The lead's conversation keeps its instructions, a new one has the new ones.
    let bodies = server.bodies();
    let continued = bodies.iter().rfind(|body| body.contains("Go on.")).unwrap();
    assert!(continued.contains("The lead-old text."), "{continued}");
    assert!(!continued.contains("lead-new"), "{continued}");
    let again = bodies
        .iter()
        .find(|body| body.contains("Hello again."))
        .unwrap();
    assert!(again.contains("The lead-new text."), "{again}");
    assert!(!again.contains("lead-old"), "{again}");
}
