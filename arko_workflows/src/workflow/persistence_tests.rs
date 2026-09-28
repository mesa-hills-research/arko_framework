//! Checkpoints that refer to task files, retention, resuming older
//! checkpoints, and the event log during runs.

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::persistence::PersistenceConfig;
use crate::persistence::state::WorkflowState;
use crate::test_support::{ResponsesServer, Workspace, agent_config, complete_turn};
use crate::workflow::result::AgentRunStatus;

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("persistence-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
}

fn run_dir(workspace: &Workspace) -> std::path::PathBuf {
    workspace.0.join(".run")
}

fn latest(workspace: &Workspace) -> (std::path::PathBuf, Value) {
    let run = run_dir(workspace);
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    let path = run.join(name.trim());
    let json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    (path, json)
}

fn checkpoint_files(workspace: &Workspace) -> Vec<String> {
    let mut names = std::fs::read_dir(run_dir(workspace))
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("checkpoint-") && name.ends_with(".json"))
        .collect::<Vec<_>>();
    names.sort();
    names
}

async fn done_server() -> ResponsesServer {
    ResponsesServer::start(|index, _| Some(complete_turn(&format!("c{index}"), "done"))).await
}

/// A checkpoint refers to each run's task file instead of holding the task
/// (the conversation is stored once), stays pretty-printed, and a resume
/// reads the tasks back from the files.
#[tokio::test]
async fn checkpoints_refer_to_task_files_and_resume_reads_them() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let config = || PersistenceConfig::new(run_dir(&workspace));
    let workflow = builder(&server, &workspace)
        .persistence(config())
        .build()
        .unwrap();
    let result = workflow.run_agent("coder", "Write it.").await.unwrap();
    let (path, checkpoint) = latest(&workspace);
    assert!(checkpoint.get("agent_results").is_none(), "{checkpoint}");
    let runs = checkpoint["agent_runs"].as_array().unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(
        runs[0]["task_file"],
        format!("agents/{}.json", result.agent_run_id)
    );
    assert!(
        !checkpoint.to_string().contains("Write it."),
        "no conversation"
    );
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("\n  \"workflow_name\""),
        "pretty-printed"
    );
    drop(workflow);

    let resumed = builder(&server, &workspace)
        .persistence(config())
        .resume_from(&path)
        .build()
        .unwrap();
    let restored = resumed.agent_result(result.agent_run_id).unwrap();
    assert_eq!(restored.status, AgentRunStatus::Completed);
    assert_eq!(
        restored.task.conversation.messages.len(),
        result.task.conversation.messages.len()
    );
    // A missing task file is a clear error.
    std::fs::remove_file(run_dir(&workspace).join(format!("agents/{}.json", result.agent_run_id)))
        .unwrap();
    let error = builder(&server, &workspace)
        .persistence(config())
        .resume_from(&path)
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(&error, WorkflowError::Persistence(message) if message.contains("task file")),
        "{error}"
    );
}

/// A checkpoint written before task files were referenced (holding the
/// results with their tasks) still resumes; the next checkpoint writes the
/// task files and refers to them.
#[tokio::test]
async fn a_checkpoint_holding_its_tasks_still_resumes() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let run = run_dir(&workspace);
    std::fs::create_dir_all(&run).unwrap();
    let run_id = uuid::Uuid::new_v4();
    std::fs::write(run.join("workflow-run-id"), run_id.to_string()).unwrap();
    let task = arko_agents::Task::new(
        agent_config("Friendly Coder"),
        "Write it.".into(),
        workspace.0.clone(),
        None,
    );
    let mut task_json = serde_json::to_value(&task).unwrap();
    task_json["current_state"] = json!({"Completed": {"summary": "done"}});
    let legacy = json!({
        "workflow_run_id": run_id,
        "workflow_name": "persistence-test",
        "checkpoint_id": uuid::Uuid::new_v4(),
        "timestamp": "2026-09-01T12:00:00Z",
        "memory_snapshot": [],
        "message_queues": {},
        "budget_spent": 0.0,
        "agent_results": [{
            "agent_run_id": task.id,
            "agent_name": "coder",
            "status": "Completed",
            "summary": "done",
            "error": null,
            "task": task_json
        }]
    });
    let path = run.join("checkpoint-legacy.json");
    std::fs::write(&path, serde_json::to_string_pretty(&legacy).unwrap()).unwrap();
    let workflow = builder(&server, &workspace)
        .persistence(PersistenceConfig::new(&run))
        .resume_from(&path)
        .build()
        .unwrap();
    assert_eq!(
        workflow.agent_result(task.id).unwrap().status,
        AgentRunStatus::Completed
    );
    workflow.checkpoint().await.unwrap();
    let (_, checkpoint) = latest(&workspace);
    assert_eq!(
        checkpoint["agent_runs"][0]["task_file"],
        format!("agents/{}.json", task.id)
    );
    assert!(run.join(format!("agents/{}.json", task.id)).is_file());
    let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
    assert_eq!(
        state.agent_results_in(&run).unwrap()[0].agent_run_id,
        task.id
    );
}

/// Every checkpoint is kept by default; `retain_checkpoints(n)` keeps the
/// newest `n`, including those of an earlier process.
#[tokio::test]
async fn checkpoints_are_kept_unless_retention_is_set() {
    for retain in [None, Some(2)] {
        let workspace = Workspace::new();
        let server = done_server().await;
        // An earlier process's checkpoint.
        std::fs::create_dir_all(run_dir(&workspace)).unwrap();
        std::fs::write(run_dir(&workspace).join("checkpoint-earlier.json"), "{}").unwrap();
        let mut config = PersistenceConfig::new(run_dir(&workspace));
        if let Some(retain) = retain {
            config = config.retain_checkpoints(retain);
        }
        let workflow = builder(&server, &workspace)
            .persistence(config)
            .build()
            .unwrap();
        for _ in 0..3 {
            workflow.checkpoint().await.unwrap();
        }
        let files = checkpoint_files(&workspace);
        match retain {
            None => assert_eq!(files.len(), 4, "{files:?}"),
            Some(retain) => {
                assert_eq!(files.len(), retain, "{files:?}");
                assert!(!files.contains(&"checkpoint-earlier.json".to_string()));
                // The latest is among them.
                let (path, _) = latest(&workspace);
                assert!(path.is_file());
            }
        }
    }
}

/// An event log that can't be written doesn't stop the agents: the run
/// completes, and the events are written once writing works again.
#[tokio::test]
async fn an_event_log_that_cant_be_written_doesnt_stop_the_run() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let log = run_dir(&workspace).join("events.jsonl");
    std::fs::create_dir_all(&log).unwrap();
    let workflow = builder(&server, &workspace)
        .persistence(PersistenceConfig::new(run_dir(&workspace)).auto_checkpoint(false))
        .build()
        .unwrap();
    let result = workflow.run_agent("coder", "Write it.").await.unwrap();
    assert_eq!(result.summary, "done");
    std::fs::remove_dir(&log).unwrap();
    workflow.checkpoint().await.unwrap();
    let events = std::fs::read_to_string(&log).unwrap();
    assert!(
        events.contains(r#""Completed":{"summary":"done"}"#),
        "{events}"
    );
}
