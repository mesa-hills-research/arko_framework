//! Checkpoints that refer to task versions, retention, task states that
//! can't be saved, resuming older checkpoints, and the event log during
//! runs.

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::persistence::state::WorkflowState;
use crate::persistence::{CheckpointRetention, PersistenceConfig};
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

/// The task version files of task `id`, relative to the run directory.
fn versions(workspace: &Workspace, id: uuid::Uuid) -> Vec<String> {
    let mut names = std::fs::read_dir(run_dir(workspace).join(format!("agents/{id}")))
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| format!("agents/{id}/{}", entry.file_name().to_string_lossy()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn task_file_of(checkpoint: &Value) -> String {
    checkpoint["agent_runs"][0]["task_file"]
        .as_str()
        .unwrap()
        .to_string()
}

/// A checkpoint refers to the version of each run's task it recorded
/// instead of holding the task (the conversation is stored once), stays
/// pretty-printed, and a resume reads the tasks back from the files.
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
    let task_file = task_file_of(&checkpoint);
    assert!(
        task_file.starts_with(&format!("agents/{}/", result.agent_run_id)),
        "{task_file}"
    );
    assert_eq!(
        versions(&workspace, result.agent_run_id),
        std::slice::from_ref(&task_file)
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
    std::fs::remove_file(run_dir(&workspace).join(&task_file)).unwrap();
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
    assert_eq!(versions(&workspace, task.id), [task_file_of(&checkpoint)]);
    let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
    assert_eq!(
        state.agent_results_in(&run).unwrap()[0].agent_run_id,
        task.id
    );
}

/// Every checkpoint is kept by default; `keep_last_checkpoints(n)` keeps the
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
            config = config.keep_last_checkpoints(retain);
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

/// Each checkpoint refers to the task versions it saw: a task that runs on
/// gets a new version, and an older checkpoint still reads the task as it
/// was then. (Checkpoints once shared one task file, which held its latest
/// state.)
#[tokio::test]
async fn an_older_checkpoint_reads_the_task_as_it_was() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let config = || PersistenceConfig::new(run_dir(&workspace));
    let workflow = builder(&server, &workspace)
        .persistence(config())
        .build()
        .unwrap();
    let first = workflow.run_agent("coder", "Write it.").await.unwrap();
    let (first_checkpoint, first_json) = latest(&workspace);
    let second = workflow
        .continue_agent(&first, "Now test it.")
        .await
        .unwrap();
    let (second_checkpoint, second_json) = latest(&workspace);
    assert_ne!(first_checkpoint, second_checkpoint);
    assert_ne!(task_file_of(&first_json), task_file_of(&second_json));
    assert_eq!(versions(&workspace, first.agent_run_id).len(), 2);
    drop(workflow);

    for (checkpoint, expected) in [(&first_checkpoint, &first), (&second_checkpoint, &second)] {
        let resumed = builder(&server, &workspace)
            .persistence(config())
            .resume_from(checkpoint)
            .build()
            .unwrap();
        let restored = resumed.agent_result(first.agent_run_id).unwrap();
        assert_eq!(
            restored.task.conversation.messages.len(),
            expected.task.conversation.messages.len()
        );
        assert_eq!(
            serde_json::to_string(&restored.task.conversation.messages)
                .unwrap()
                .contains("Now test it."),
            std::ptr::eq(expected, &second)
        );
    }
}

/// A task that didn't change is written once: every checkpoint refers to
/// the same version, in this process and after a resume.
#[tokio::test]
async fn an_unchanged_task_is_written_once() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let config = || PersistenceConfig::new(run_dir(&workspace));
    let workflow = builder(&server, &workspace)
        .persistence(config())
        .build()
        .unwrap();
    let result = workflow.run_agent("coder", "Write it.").await.unwrap();
    let saved = versions(&workspace, result.agent_run_id);
    assert_eq!(saved.len(), 1);
    let modified = || {
        std::fs::metadata(run_dir(&workspace).join(&saved[0]))
            .unwrap()
            .modified()
            .unwrap()
    };
    let written = modified();
    for _ in 0..2 {
        workflow.checkpoint().await.unwrap();
        assert_eq!(task_file_of(&latest(&workspace).1), saved[0]);
    }
    let (path, _) = latest(&workspace);
    drop(workflow);
    let resumed = builder(&server, &workspace)
        .persistence(config())
        .resume_from(&path)
        .build()
        .unwrap();
    resumed.checkpoint().await.unwrap();
    assert_eq!(task_file_of(&latest(&workspace).1), saved[0]);
    assert_eq!(versions(&workspace, result.agent_run_id), saved);
    assert_eq!(modified(), written, "not written again");
}

/// With retention, the task versions only deleted checkpoints referred to
/// are deleted too; those a kept checkpoint refers to stay, and so do the
/// task files of older versions (`agents/<id>.json`).
#[tokio::test]
async fn retention_deletes_only_task_versions_no_kept_checkpoint_refers_to() {
    for retain in [1, 2] {
        let workspace = Workspace::new();
        let server = done_server().await;
        std::fs::create_dir_all(run_dir(&workspace).join("agents")).unwrap();
        let legacy = run_dir(&workspace).join(format!("agents/{}.json", uuid::Uuid::new_v4()));
        std::fs::write(&legacy, "{}").unwrap();
        let workflow = builder(&server, &workspace)
            .persistence(PersistenceConfig::new(run_dir(&workspace)).keep_last_checkpoints(retain))
            .build()
            .unwrap();
        let first = workflow.run_agent("coder", "Write it.").await.unwrap();
        let first_version = task_file_of(&latest(&workspace).1);
        workflow
            .continue_agent(&first, "Now test it.")
            .await
            .unwrap();
        let second_version = task_file_of(&latest(&workspace).1);
        let kept = versions(&workspace, first.agent_run_id);
        if retain == 1 {
            assert_eq!(kept, [second_version], "the first checkpoint is gone");
        } else {
            let mut both = vec![first_version, second_version];
            both.sort();
            assert_eq!(kept, both);
        }
        assert!(legacy.is_file());
    }
}

/// A task state that can't be saved doesn't fail the run: the checkpoint
/// is still written, refers to the task's last saved version, and records
/// that its latest state is missing; a resume from it says so and restores
/// the task as last saved. The next checkpoint saves the latest state. A
/// task never saved is left out of the checkpoint (and not restored).
#[tokio::test]
async fn a_task_state_that_cant_be_saved_is_retried_and_reported() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let config = || PersistenceConfig::new(run_dir(&workspace));
    let workflow = builder(&server, &workspace)
        .persistence(config())
        .build()
        .unwrap();
    let failing = || {
        workflow
            .engine
            .persistence()
            .unwrap()
            .fail_task_writes
            .clone()
    };
    let first = workflow.run_agent("coder", "Write it.").await.unwrap();
    let saved = task_file_of(&latest(&workspace).1);

    failing().store(true, std::sync::atomic::Ordering::SeqCst);
    let second = workflow
        .continue_agent(&first, "Now test it.")
        .await
        .expect("the run doesn't fail");
    assert_eq!(second.status, AgentRunStatus::Completed);
    let (stale_path, checkpoint) = latest(&workspace);
    assert_eq!(task_file_of(&checkpoint), saved, "the last saved version");
    let missing = &checkpoint["missing_task_states"][0];
    assert_eq!(missing["agent_run_id"], json!(first.agent_run_id));
    assert_eq!(missing["status"], "Completed");
    assert_eq!(missing["last_saved"], json!(saved));
    assert!(
        missing["reason"]
            .as_str()
            .unwrap()
            .contains("injected task write failure")
    );
    let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
    let (results, warnings) = state
        .agent_results_with_warnings(&run_dir(&workspace))
        .unwrap();
    assert_eq!(
        results[0].task.conversation.messages.len(),
        first.task.conversation.messages.len(),
        "restored as last saved"
    );
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains(&format!(
            "The checkpoint lacks the latest state of agent 'coder' ({})",
            first.agent_run_id
        )),
        "{warnings:?}"
    );
    assert!(
        warnings[0].contains("resumes as last saved"),
        "{warnings:?}"
    );

    // Writing works again: the next checkpoint saves the latest state.
    failing().store(false, std::sync::atomic::Ordering::SeqCst);
    workflow.checkpoint().await.unwrap();
    let (_, checkpoint) = latest(&workspace);
    assert!(checkpoint.get("missing_task_states").is_none());
    assert_ne!(task_file_of(&checkpoint), saved);
    assert_eq!(versions(&workspace, first.agent_run_id).len(), 2);

    // A task never saved is left out, and not restored.
    failing().store(true, std::sync::atomic::Ordering::SeqCst);
    let unsaved = workflow.run_agent("coder", "Write another.").await.unwrap();
    let (path, checkpoint) = latest(&workspace);
    assert_eq!(checkpoint["agent_runs"].as_array().unwrap().len(), 1);
    assert_eq!(
        checkpoint["missing_task_states"][0]["agent_run_id"],
        json!(unsaved.agent_run_id)
    );
    assert!(
        checkpoint["missing_task_states"][0]
            .get("last_saved")
            .is_none()
    );
    let state: WorkflowState = serde_json::from_value(checkpoint).unwrap();
    let (results, warnings) = state
        .agent_results_with_warnings(&run_dir(&workspace))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("so it is not restored")),
        "{warnings:?}"
    );
    drop(workflow);
    // Resuming from such a checkpoint works.
    for checkpoint in [&stale_path, &path] {
        builder(&server, &workspace)
            .persistence(config())
            .resume_from(checkpoint)
            .build()
            .unwrap();
    }
}

/// Waits, up to a generous limit, until `done` holds.
async fn until(mut done: impl FnMut() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while !done() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the condition came true");
}

/// A run call dropped while its finished run waits to be saved (behind
/// another write) still records the result, and saves it once writing goes
/// on. (The result was once recorded only after the save, so dropping the
/// call then lost it: `agent_result` found nothing to continue.)
#[tokio::test]
async fn a_result_is_recorded_when_its_call_is_dropped_while_saving() {
    use crate::messaging::registry::AgentStatus;
    let workspace = Workspace::new();
    let server = done_server().await;
    let workflow = Arc::new(
        builder(&server, &workspace)
            // Without the task journal: its save before a tool call runs
            // would wait for the held writes.
            .persistence(PersistenceConfig::new(run_dir(&workspace)).task_journal(false))
            .build()
            .unwrap(),
    );
    let held = workflow.engine.persistence().unwrap().hold_writes().await;
    let call = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    // The run ended; its call waits to save the result.
    until(|| {
        matches!(
            workflow.agent_status("coder"),
            Some(AgentStatus::Finished(_))
        )
    })
    .await;
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    drop(held);
    until(|| !workflow.result().agent_results.is_empty()).await;
    let result = workflow.result().agent_results[0].clone();
    assert_eq!(result.status, AgentRunStatus::Completed);
    until(|| !versions(&workspace, result.agent_run_id).is_empty()).await;
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

/// Resuming keeps memory content by tier name under the configuration's
/// access; a checkpoint tier the configuration lacks stops the build unless
/// it is dropped or renamed.
#[tokio::test]
async fn resumed_memory_follows_the_configuration() {
    use crate::memory::{AccessScope, MemoryConfig, MemoryTier, WriteMode};
    let workspace = Workspace::new();
    let server = done_server().await;
    let config = || PersistenceConfig::new(run_dir(&workspace));
    let first = builder(&server, &workspace)
        .persistence(config())
        .memory(MemoryConfig::new().global_tier("notes").global_tier("old"))
        .build()
        .unwrap();
    first
        .context()
        .memory
        .write("coder", "notes", "kept")
        .unwrap();
    first
        .context()
        .memory
        .write("coder", "old", "archived")
        .unwrap();
    first.checkpoint().await.unwrap();
    let (path, _) = latest(&workspace);
    drop(first);

    let read_only = || {
        MemoryConfig::new().tier(MemoryTier {
            name: "notes".into(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::None,
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: None,
        })
    };
    let refused = builder(&server, &workspace)
        .persistence(config())
        .memory(read_only())
        .resume_from(&path)
        .build()
        .err()
        .unwrap();
    assert!(
        refused
            .to_string()
            .contains("memory tier 'old' of the checkpoint is not configured"),
        "{refused}"
    );

    let resumed = builder(&server, &workspace)
        .persistence(config())
        .memory(read_only())
        .drop_memory_tier_on_resume("old")
        .resume_from(&path)
        .build()
        .unwrap();
    let memory = &resumed.context().memory;
    assert_eq!(memory.read("coder", "notes").unwrap(), "kept");
    assert!(
        memory.write("coder", "notes", "more").is_err(),
        "writers from the configuration"
    );
    // The tools follow the restored access too.
    let registry = resumed
        .engine
        .build_tool_registry("coder", resumed.context(), 0);
    assert!(registry.get("read_memory").is_some());
    assert!(registry.get("write_memory").is_none());
}

/// Keeping the last zero checkpoints fails the build, so no checkpoint is
/// deleted. (A zero count on the former public `Option<usize>` field
/// deleted every checkpoint in the directory after each save: the one just
/// written and every earlier process's, with their task versions.) Every
/// checkpoint is kept by default.
#[tokio::test]
async fn keeping_the_last_zero_checkpoints_fails_the_build() {
    let workspace = Workspace::new();
    let server = done_server().await;
    let config = PersistenceConfig::new(run_dir(&workspace));
    assert_eq!(config.retain_checkpoints, CheckpointRetention::KeepAll);
    let error = builder(&server, &workspace)
        .persistence(config.clone().keep_last_checkpoints(0))
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message)
            if message == "keep_last_checkpoints needs at least 1. Every checkpoint is kept by default"),
        "{error}"
    );
    // A later valid count replaces the rejected one.
    let config = config.keep_last_checkpoints(0).keep_last_checkpoints(1);
    assert_eq!(
        config.retain_checkpoints,
        CheckpointRetention::KeepLast(std::num::NonZeroUsize::MIN)
    );
    assert!(
        builder(&server, &workspace)
            .persistence(config)
            .build()
            .is_ok()
    );
}
