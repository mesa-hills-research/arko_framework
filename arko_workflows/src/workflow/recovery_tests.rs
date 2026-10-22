//! The task journal: agent runs saved as they run, and recovered after the
//! workflow stops while they run (a crash, simulated by freezing the run
//! directory and dropping the workflow).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::budget::BudgetExceededPolicy;
use crate::persistence::state::WorkflowState;
use crate::persistence::{LIVE_CHECKPOINT_FILE, PersistenceConfig};
use crate::services::{RUN_WORKFLOW_AGENT_TOOL_NAME, WorkflowServicesConfig};
use crate::test_support::{
    ProbeTool, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::workflow::result::AgentRunStatus;

/// The result an in-doubt call gets when its task is recovered.
const RESTARTED: &str = "restarted while this call was running";

const VERIFIER: &str = "You are a completion verifier";

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("recovery-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent("reviewer", agent_config("Friendly Reviewer"))
}

fn run_dir(workspace: &Workspace) -> std::path::PathBuf {
    workspace.0.join(".run")
}

fn config(workspace: &Workspace) -> PersistenceConfig {
    PersistenceConfig::new(run_dir(workspace))
}

/// The checkpoint `latest-checkpoint` points to.
fn latest(workspace: &Workspace) -> std::path::PathBuf {
    let run = run_dir(workspace);
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    run.join(name.trim())
}

fn read_state(path: &std::path::Path) -> WorkflowState {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn live_file(workspace: &Workspace, id: uuid::Uuid) -> std::path::PathBuf {
    run_dir(workspace).join(format!("agents/{id}/live.json"))
}

fn versioned_checkpoints(workspace: &Workspace) -> Vec<std::path::PathBuf> {
    let mut paths = std::fs::read_dir(run_dir(workspace))
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.starts_with("checkpoint-") && name.ends_with(".json")
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(20), future)
        .await
        .expect("the workflow call returned")
}

/// Waits, up to a generous limit, until `done` holds.
async fn until(mut done: impl FnMut() -> bool) {
    within(async {
        while !done() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
}

/// Stops `workflow` as a crash would: from now on nothing reaches its run
/// directory, and its runs are dropped. The agent runtimes' own tasks may
/// run on (blocked in a tool or a held request): a test must not count on
/// what they do after the crash, such as the server's request count while
/// one of them could still send.
async fn crash<T>(workflow: Arc<Workflow>, runs: Vec<tokio::task::JoinHandle<T>>) {
    workflow.engine.persistence().unwrap().simulate_crash();
    for run in runs {
        run.abort();
        let _ = run.await;
    }
    drop(workflow);
}

/// Whether `latest-checkpoint` names the live checkpoint and a live file it
/// refers to records a running call.
fn running_call_saved(workspace: &Workspace) -> bool {
    let run = run_dir(workspace);
    std::fs::read_to_string(run.join("latest-checkpoint"))
        .is_ok_and(|name| name == LIVE_CHECKPOINT_FILE)
        && std::fs::read_to_string(run.join(LIVE_CHECKPOINT_FILE))
            .ok()
            .and_then(|json| serde_json::from_str::<WorkflowState>(&json).ok())
            .is_some_and(|state| {
                state.agent_runs.iter().any(|record| {
                    std::fs::read_to_string(run.join(&record.task_file))
                        .is_ok_and(|task| task.contains("running_tool_calls"))
                })
            })
}

/// The number of user messages a request sends.
fn user_messages(body: &str) -> usize {
    body.matches(r#""role":"user""#).count()
}

/// Runs `coder` with `probe` (blocking) in a workflow that crashes while
/// the probe runs; returns the checkpoint to resume from.
async fn crash_mid_tool(
    server: &ResponsesServer,
    workspace: &Workspace,
    probe: Arc<ProbeTool>,
) -> std::path::PathBuf {
    let workflow = Arc::new(
        builder(server, workspace)
            .global_tool(probe.clone())
            .persistence(config(workspace))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    // The call runs only once the task and a checkpoint listing it are
    // saved (a `Rerun` call doesn't wait for the save).
    within(probe.started.notified()).await;
    if probe.recovery == arko_agents::ToolRecovery::Rerun {
        until(|| running_call_saved(workspace)).await;
    }
    assert!(running_call_saved(workspace));
    let path = latest(workspace);
    assert_eq!(path.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
    crash(workflow, vec![run]).await;
    path
}

/// A run that crashed while its tool call ran comes back interrupted with
/// its task, and `recover_agent` continues it where it was: the call (in
/// doubt by default) is answered without running again, no message is
/// added, and the run keeps its identity. Once it ends, its live file is
/// gone and the latest checkpoint records its result; each model call is
/// charged once.
#[tokio::test]
async fn a_run_that_crashed_mid_tool_is_recovered_where_it_was() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains(RESTARTED) {
            complete_turn("done", "done")
        } else {
            tool_call_turn("probe", "probe", json!({}))
        })
    })
    .await;
    let path = crash_mid_tool(&server, &workspace, ProbeTool::new("probe", true)).await;

    let probe = ProbeTool::new("probe", false);
    let resumed = builder(&server, &workspace)
        .global_tool(probe.clone())
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let interrupted = resumed.interrupted_runs();
    assert_eq!(interrupted.len(), 1);
    let saved = &interrupted[0];
    assert_eq!(saved.agent_name, "coder");
    assert_eq!(saved.status, AgentRunStatus::Interrupted);
    assert_eq!(saved.task.running_tool_calls.len(), 1);
    assert_eq!(saved.task.running_tool_calls[0].tool_name, "probe");
    assert!(live_file(&workspace, saved.agent_run_id).is_file());

    let result = within(resumed.recover_agent(saved)).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.summary, "done");
    assert_eq!(result.agent_run_id, saved.agent_run_id);
    assert_eq!(probe.calls(), 0, "an in-doubt call doesn't run again");
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2, "not started again");
    assert!(bodies[1].contains(RESTARTED), "{}", bodies[1]);
    assert_eq!(user_messages(&bodies[1]), user_messages(&bodies[0]));
    assert!(resumed.interrupted_runs().is_empty());

    assert!(!live_file(&workspace, result.agent_run_id).exists());
    let state = read_state(&latest(&workspace));
    let record = state
        .agent_runs
        .iter()
        .find(|record| record.agent_run_id == result.agent_run_id)
        .unwrap();
    assert_eq!(record.status, AgentRunStatus::Completed);
    assert!(record.usage_charged.is_none());
    assert_eq!(
        resumed.cost_ledger().entries.len(),
        result.task.usage_history.len(),
        "each model call is charged once"
    );

    // Only an interrupted run can be recovered.
    let error = resumed.recover_agent(&result).await.unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.starts_with("only an interrupted agent run can be recovered")),
        "{error}"
    );
}

/// A call of a tool that declares `Rerun` runs again when its task is
/// recovered.
#[tokio::test]
async fn a_rerun_tool_runs_again_on_recovery() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains("probed") {
            complete_turn("done", "done")
        } else {
            tool_call_turn("probe", "probe", json!({}))
        })
    })
    .await;
    let rerun = arko_agents::ToolRecovery::Rerun;
    let path = crash_mid_tool(
        &server,
        &workspace,
        ProbeTool::recovered_by("probe", true, rerun),
    )
    .await;
    let probe = ProbeTool::recovered_by("probe", false, rerun);
    let resumed = builder(&server, &workspace)
        .global_tool(probe.clone())
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    let result = within(resumed.recover_agent(&saved)).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(probe.calls(), 1);
    assert_eq!(server.requests(), 2);
}

/// A run being recovered is not listed as interrupted and can't be
/// recovered twice at once.
#[tokio::test]
async fn a_run_is_recovered_once_at_a_time() {
    let workspace = Workspace::new();
    // After the first request, every request waits.
    let server = ResponsesServer::start(|index, _| {
        (index == 0).then(|| tool_call_turn("probe", "probe", json!({})))
    })
    .await;
    let path = crash_mid_tool(&server, &workspace, ProbeTool::new("probe", true)).await;
    let resumed = Arc::new(
        builder(&server, &workspace)
            .global_tool(ProbeTool::new("probe", false))
            .persistence(config(&workspace))
            .resume_from(&path)
            .build()
            .unwrap(),
    );
    let saved = resumed.interrupted_runs().remove(0);
    let recovering = tokio::spawn({
        let (resumed, saved) = (resumed.clone(), saved.clone());
        async move { resumed.recover_agent(&saved).await }
    });
    server.wait_for_requests(2).await;
    assert!(resumed.interrupted_runs().is_empty());
    let error = resumed.recover_agent(&saved).await.unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("is running already")),
        "{error}"
    );
    recovering.abort();
}

/// Resuming an interrupted run with a changed agent configuration fails
/// the build, as for a paused run.
#[tokio::test]
async fn an_interrupted_run_needs_its_configuration() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|_, _| Some(tool_call_turn("probe", "probe", json!({})))).await;
    let path = crash_mid_tool(&server, &workspace, ProbeTool::new("probe", true)).await;
    let mut changed = agent_config("Friendly Coder");
    changed.instructions = "other instructions".to_string();
    changed.tool_config.tool_timeout_seconds = Some(30);
    let error = Workflow::builder()
        .name("recovery-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", changed)
        .agent("reviewer", agent_config("Friendly Reviewer"))
        .global_tool(ProbeTool::new("probe", false))
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .err()
        .unwrap();
    // The error names the fields that changed.
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("agent 'coder' configuration changed since its checkpoint (instructions, tool_config.tool_timeout_seconds changed). Use the original configuration")),
        "{error}"
    );
}

/// The changed fields are named by their paths, objects field by field and
/// arrays as a whole, a long list cut short.
#[test]
fn changed_configuration_fields_are_named_by_their_paths() {
    let saved = json!({
        "instructions": "a",
        "tool_config": {"available_tool_names": ["read_file"], "tool_timeout_seconds": 1200},
        "handover": {"at_input_tokens": 100},
        "claude_code": null,
    });
    let current = json!({
        "instructions": "b",
        "tool_config": {"available_tool_names": ["read_file", "terminal_start"], "tool_timeout_seconds": 1200},
        "handover": {"at_input_tokens": 100, "report_request": "Report."},
        "claude_code": {"env": {"KEY": "1"}},
    });
    let mut changed = Vec::new();
    super::changed_fields(&saved, &current, "", &mut changed);
    assert_eq!(
        changed,
        [
            "claude_code",
            "handover.report_request",
            "instructions",
            "tool_config.available_tool_names",
        ]
    );
    let many: Vec<String> = (0..12).map(|n| format!("f{n}")).collect();
    assert_eq!(
        super::describe_changed_fields(&many),
        "f0, f1, f2, f3, f4, f5, f6, f7, f8, f9 and 2 more"
    );
}

/// Usage a running agent reported that the checkpoint's ledger lacks (the
/// workflow hadn't processed the report yet) is charged on resume, once.
#[tokio::test]
async fn usage_the_ledger_lacked_is_charged_on_resume() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|_, _| Some(tool_call_turn("probe", "probe", json!({})))).await;
    let path = crash_mid_tool(&server, &workspace, ProbeTool::new("probe", true)).await;
    // As if the crash came before the workflow charged the model call.
    let mut checkpoint: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    checkpoint["cost_ledger"] = json!({"entries": []});
    checkpoint["budget_spent"] = json!(0.0);
    let record = checkpoint["agent_runs"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|record| record["status"] == "Interrupted")
        .unwrap();
    assert_eq!(record["usage_charged"], 1);
    record["usage_charged"] = json!(0);
    std::fs::write(&path, checkpoint.to_string()).unwrap();

    let resumed = builder(&server, &workspace)
        .global_tool(ProbeTool::new("probe", false))
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    assert_eq!(saved.task.usage_history.len(), 1);
    assert_eq!(resumed.cost_ledger().entries.len(), 1);
}

/// Without the task journal nothing is saved while agents run.
#[tokio::test]
async fn without_the_task_journal_running_agents_are_not_saved() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|_, _| Some(tool_call_turn("probe", "probe", json!({})))).await;
    let probe = ProbeTool::new("probe", true);
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(probe.clone())
            .persistence(config(&workspace).task_journal(false))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    within(probe.started.notified()).await;
    let run_dir = run_dir(&workspace);
    assert!(!run_dir.join(LIVE_CHECKPOINT_FILE).exists());
    assert!(!run_dir.join("latest-checkpoint").exists());
    assert!(!run_dir.join("agents").exists());
    crash(workflow, vec![run]).await;
}

/// The task journal's files are compact JSON unless `pretty_json` is set;
/// checkpoints stay pretty. Once a run ends its live file is deleted, the
/// latest checkpoint is a versioned one, and the live checkpoint still
/// resumes (it refers to no deleted file).
#[tokio::test]
async fn journal_files_are_compact_unless_pretty_and_go_when_runs_end() {
    for pretty in [false, true] {
        let workspace = Workspace::new();
        let release = Arc::new(tokio::sync::Notify::new());
        let server = ResponsesServer::start_async({
            let release = release.clone();
            move |index, _| {
                let release = release.clone();
                Box::pin(async move {
                    if index == 0 {
                        Some(crate::test_support::Reply::Stream(tool_call_turn(
                            "probe",
                            "probe",
                            json!({}),
                        )))
                    } else {
                        // Held until the files are checked.
                        release.notified().await;
                        Some(crate::test_support::Reply::Stream(complete_turn(
                            "done", "done",
                        )))
                    }
                })
            }
        })
        .await;
        let workflow = Arc::new(
            builder(&server, &workspace)
                .global_tool(ProbeTool::new("probe", false))
                .persistence(config(&workspace).pretty_json(pretty))
                .build()
                .unwrap(),
        );
        let run = tokio::spawn({
            let workflow = workflow.clone();
            async move { workflow.run_agent("coder", "Write it.").await }
        });
        server.wait_for_requests(2).await;
        let live_checkpoint = run_dir(&workspace).join(LIVE_CHECKPOINT_FILE);
        let state = read_state(&live_checkpoint);
        let running = &state.agent_runs[0];
        assert_eq!(running.status, AgentRunStatus::Interrupted);
        let live = run_dir(&workspace).join(&running.task_file);
        assert_eq!(live, live_file(&workspace, running.agent_run_id));
        for file in [&live, &live_checkpoint] {
            let json = std::fs::read_to_string(file).unwrap();
            assert_eq!(json.contains("\n  \""), pretty, "{}", file.display());
        }

        release.notify_one();
        let result = within(run).await.unwrap().unwrap();
        assert!(!live_file(&workspace, result.agent_run_id).exists());
        let latest = latest(&workspace);
        assert_ne!(latest.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
        assert!(
            std::fs::read_to_string(&latest)
                .unwrap()
                .contains("\n  \"workflow_name\""),
            "checkpoints stay pretty"
        );
        drop(workflow);
        let resumed = builder(&server, &workspace)
            .global_tool(ProbeTool::new("probe", false))
            .persistence(config(&workspace))
            .resume_from(&live_checkpoint)
            .build()
            .unwrap();
        assert_eq!(
            resumed.agent_result(result.agent_run_id).unwrap().status,
            AgentRunStatus::Completed
        );
    }
}

/// A server for delegation: `coder` (prompt "Delegate.") runs `reviewer`
/// (prompt "Block."), which calls `block`. Once a call has a result, each
/// completes. While `hold_reviewer` is set, the reviewer's requests wait.
async fn delegation_server(hold_reviewer: Arc<AtomicBool>) -> ResponsesServer {
    ResponsesServer::start(move |_, body| {
        let answered = body.contains("function_call_output");
        if body.contains("Delegate") {
            Some(if answered {
                complete_turn("coder-done", "coder done")
            } else {
                let prompt = if body.contains("Delegate B.") {
                    "Block B."
                } else {
                    "Block."
                };
                tool_call_turn(
                    "delegate",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "reviewer", "prompt": prompt}),
                )
            })
        } else if hold_reviewer.load(Ordering::SeqCst) {
            None
        } else if answered {
            Some(complete_turn("reviewer-done", "reviewed"))
        } else {
            Some(tool_call_turn("block", "block", json!({})))
        }
    })
    .await
}

fn delegating(
    server: &ResponsesServer,
    workspace: &Workspace,
    block: Arc<ProbeTool>,
    max_active_calls: u32,
) -> WorkflowBuilder {
    builder(server, workspace)
        .agent("planner", agent_config("Friendly Planner"))
        .agent_tool("reviewer", block)
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("coder")
                .allow_caller("planner")
                .allow_target("reviewer")
                .max_active_calls(max_active_calls),
        )
        .persistence(config(workspace))
}

/// Crashes while `coder`'s delegated reviewer is stuck in `block`; returns
/// the checkpoint to resume from.
async fn crash_mid_delegation(
    server: &ResponsesServer,
    workspace: &Workspace,
) -> std::path::PathBuf {
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(
        delegating(server, workspace, block.clone(), 4)
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Delegate.").await }
    });
    within(block.started.notified()).await;
    let path = latest(workspace);
    assert_eq!(path.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
    crash(workflow, vec![run]).await;
    path
}

/// Recovery of a run at `depth`, as its parent's call recovers it.
fn nested(depth: u32) -> engine::Recovery<'static> {
    engine::Recovery {
        depth,
        wait: true,
        ..engine::Recovery::default()
    }
}

fn result_of(workflow: &Workflow, agent: &str) -> AgentResult {
    workflow
        .result()
        .agent_results
        .into_iter()
        .rfind(|result| result.agent_name == agent)
        .unwrap()
}

/// A crash while a delegated agent runs: only the delegating run is
/// listed; recovering it recovers the delegated run (same identity, in
/// doubt about its stuck call), whose result becomes the delegating call's
/// result. Neither starts again.
#[tokio::test]
async fn a_delegated_run_is_recovered_through_its_parent() {
    let workspace = Workspace::new();
    let server = delegation_server(Arc::default()).await;
    let path = crash_mid_delegation(&server, &workspace).await;
    assert_eq!(server.requests(), 2);

    let block = ProbeTool::new("block", false);
    let resumed = delegating(&server, &workspace, block.clone(), 4)
        .resume_from(&path)
        .build()
        .unwrap();
    let interrupted = resumed.interrupted_runs();
    assert_eq!(interrupted.len(), 1);
    let coder = &interrupted[0];
    assert_eq!(coder.agent_name, "coder");
    let reviewer = result_of(&resumed, "reviewer");
    assert_eq!(reviewer.status, AgentRunStatus::Interrupted);
    let parent = reviewer.task.parent.unwrap();
    assert_eq!(parent.task_id, coder.agent_run_id);
    assert_eq!(
        parent.tool_call_id,
        coder.task.running_tool_calls[0].tool_call_id
    );
    let error = resumed.recover_agent(&reviewer).await.unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("Recover that run")),
        "{error}"
    );

    let result = within(resumed.recover_agent(coder)).await.unwrap();
    assert_eq!(result.summary, "coder done");
    assert_eq!(result.agent_run_id, coder.agent_run_id);
    let recovered = result_of(&resumed, "reviewer");
    assert_eq!(recovered.agent_run_id, reviewer.agent_run_id);
    assert_eq!(recovered.status, AgentRunStatus::Completed);
    assert_eq!(block.calls(), 0, "in doubt");
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 4, "neither started again");
    assert!(bodies[2].contains(RESTARTED), "{}", bodies[2]);
    assert!(bodies[3].contains("reviewed"), "{}", bodies[3]);
    for id in [coder.agent_run_id, reviewer.agent_run_id] {
        assert!(!live_file(&workspace, id).exists());
    }
}

/// A delegated run that completed before the crash, while its parent's
/// call result was not saved yet, is not run again: recovering the parent
/// returns its result.
#[tokio::test]
async fn a_delegated_run_that_completed_is_not_run_again() {
    let workspace = Workspace::new();
    let server = delegation_server(Arc::default()).await;
    let path = crash_mid_delegation(&server, &workspace).await;

    // The reviewer completes (recovered alone), then the workflow stops
    // before its parent runs on.
    let resumed = Arc::new(
        delegating(&server, &workspace, ProbeTool::new("block", false), 4)
            .resume_from(&path)
            .build()
            .unwrap(),
    );
    let reviewer = result_of(&resumed, "reviewer");
    let completed = within(
        resumed
            .engine
            .recover_nested(&reviewer, resumed.context(), nested(1)),
    )
    .await
    .unwrap();
    assert_eq!(completed.status, AgentRunStatus::Completed);
    assert_eq!(server.requests(), 3);
    let path = latest(&workspace);
    crash(resumed, Vec::<tokio::task::JoinHandle<()>>::new()).await;

    let resumed = delegating(&server, &workspace, ProbeTool::new("block", false), 4)
        .resume_from(&path)
        .build()
        .unwrap();
    assert_eq!(
        result_of(&resumed, "reviewer").status,
        AgentRunStatus::Completed
    );
    let coder = resumed.interrupted_runs().remove(0);
    let result = within(resumed.recover_agent(&coder)).await.unwrap();
    assert_eq!(result.summary, "coder done");
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 4, "the reviewer didn't run again");
    assert!(bodies[3].contains("reviewed"), "{}", bodies[3]);
}

/// A delegated run the crash came before any save of (it ran no tool: runs
/// are saved before their first) runs when its parent is recovered.
#[tokio::test]
async fn a_delegated_run_never_saved_runs_when_its_parent_is_recovered() {
    let workspace = Workspace::new();
    let hold_reviewer = Arc::new(AtomicBool::new(true));
    let server = delegation_server(hold_reviewer.clone()).await;
    // The crash comes as the reviewer starts.
    let switch = Arc::new(std::sync::OnceLock::<Arc<AtomicBool>>::new());
    let workflow = Arc::new(
        delegating(&server, &workspace, ProbeTool::new("block", true), 4)
            .hook(crate::hooks::HookPoint::PreAgentSpawn, {
                let switch = switch.clone();
                Arc::new(move |hook| {
                    if hook.agent_name.as_deref() == Some("reviewer") {
                        switch.get().unwrap().store(true, Ordering::SeqCst);
                    }
                    crate::hooks::HookResult::Continue
                })
            })
            .build()
            .unwrap(),
    );
    switch
        .set(workflow.engine.persistence().unwrap().crash_switch())
        .unwrap();
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Delegate.").await }
    });
    server.wait_for_requests(2).await;
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    hold_reviewer.store(false, Ordering::SeqCst);
    let resumed = delegating(&server, &workspace, ProbeTool::new("block", false), 4)
        .resume_from(&path)
        .build()
        .unwrap();
    assert!(
        resumed
            .result()
            .agent_results
            .iter()
            .all(|result| result.agent_name != "reviewer"),
        "never saved"
    );
    let coder = resumed.interrupted_runs().remove(0);
    let result = within(resumed.recover_agent(&coder)).await.unwrap();
    assert_eq!(result.summary, "coder done");
    let reviewer = result_of(&resumed, "reviewer");
    assert_eq!(reviewer.status, AgentRunStatus::Completed);
    assert_eq!(reviewer.task.parent.unwrap().task_id, coder.agent_run_id);
}

/// Parallel runs that crashed are recovered together, each with its
/// delegated run, and recovered delegations aren't refused by the
/// active-call limit (they were admitted before the crash).
#[tokio::test]
async fn parallel_runs_and_their_delegations_are_recovered_together() {
    let workspace = Workspace::new();
    let server = delegation_server(Arc::default()).await;
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(
        delegating(&server, &workspace, block.clone(), 2)
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("coder", "Delegate A.".to_string()),
                    ("planner", "Delegate B.".to_string()),
                ])
                .await
        }
    });
    until(|| block.calls() == 2).await;
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let resumed = Arc::new(
        delegating(&server, &workspace, ProbeTool::new("block", false), 1)
            .resume_from(&path)
            .build()
            .unwrap(),
    );
    let interrupted = resumed.interrupted_runs();
    let mut names = interrupted
        .iter()
        .map(|result| result.agent_name.as_str())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["coder", "planner"]);
    let (first, second) = within(async {
        tokio::join!(
            resumed.recover_agent(&interrupted[0]),
            resumed.recover_agent(&interrupted[1])
        )
    })
    .await;
    assert_eq!(first.unwrap().summary, "coder done");
    assert_eq!(second.unwrap().summary, "coder done");
    let reviewers = resumed
        .result()
        .agent_results
        .into_iter()
        .filter(|result| result.agent_name == "reviewer")
        .collect::<Vec<_>>();
    assert_eq!(reviewers.len(), 2);
    assert!(
        reviewers
            .iter()
            .all(|result| result.status == AgentRunStatus::Completed)
    );
    assert_eq!(server.requests(), 8, "nothing started again");
}

/// A completion verifier reviewing when the workflow stopped is superseded:
/// recovering its target reviews the completion again with a new review,
/// and the stopped review is recorded cancelled.
#[tokio::test]
async fn a_review_interrupted_by_a_crash_is_done_again() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains(VERIFIER) {
            if body.contains("function_call_output") {
                complete_turn("verdict", "ACCEPTED")
            } else {
                tool_call_turn("block", "block", json!({}))
            }
        } else {
            complete_turn("done", "done")
        })
    })
    .await;
    let verifying = |block: Arc<ProbeTool>| {
        builder(&server, &workspace)
            .agent_tool("reviewer", block)
            .completion_verifier(
                crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
                    .target("coder"),
            )
            .persistence(config(&workspace))
    };
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(verifying(block.clone()).build().unwrap());
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    within(block.started.notified()).await;
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let resumed = verifying(ProbeTool::new("block", false))
        .resume_from(&path)
        .build()
        .unwrap();
    let interrupted = resumed.interrupted_runs();
    assert_eq!(interrupted.len(), 1, "the verifier is not listed");
    let stopped_review = result_of(&resumed, "reviewer");
    assert_eq!(stopped_review.status, AgentRunStatus::Interrupted);

    let result = within(resumed.recover_agent(&interrupted[0]))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    let reviews = resumed
        .result()
        .agent_results
        .into_iter()
        .filter(|result| result.agent_name == "reviewer")
        .collect::<Vec<_>>();
    assert_eq!(reviews.len(), 2);
    let superseded = reviews
        .iter()
        .find(|review| review.agent_run_id == stopped_review.agent_run_id)
        .unwrap();
    assert_eq!(superseded.status, AgentRunStatus::Cancelled);
    assert!(
        superseded
            .error
            .as_deref()
            .unwrap()
            .starts_with("Superseded"),
        "{:?}",
        superseded.error
    );
    let records = resumed.context().verification_records.read().clone();
    assert_eq!(records.len(), 1);
    assert_ne!(records[0].verifier_task_id, stopped_review.agent_run_id);
    assert!(resumed.interrupted_runs().is_empty());
    // A superseded review is not reopened: the new review replaced it.
    let error = resumed.reopen_agent(superseded, "").await.unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("superseded")),
        "{error}"
    );
}

/// A run saved while the workflow's approval handler decided its call
/// comes back waiting, and recovering it has the call decided again (here
/// approved by the default policy): it runs.
#[tokio::test]
async fn an_approval_wait_is_decided_again_on_recovery() {
    struct Undecided;
    #[async_trait::async_trait]
    impl crate::policy::ToolApprovalHandler for Undecided {
        async fn decide(
            &self,
            _request: crate::policy::ToolApprovalRequest,
        ) -> crate::policy::ToolApprovalOutcome {
            std::future::pending().await
        }
    }
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains("probed") {
            complete_turn("done", "done")
        } else {
            tool_call_turn("probe", "probe", json!({}))
        })
    })
    .await;
    let mut coder = agent_config("Friendly Coder");
    coder.tool_config.default_auto_approve_tool_use = false;
    coder.tool_config.tool_specific_configs.insert(
        "complete_task".to_string(),
        arko_agents::PerToolConfigOverride {
            auto_approve_tool_use: Some(true),
            ..Default::default()
        },
    );
    let approving = |probe: Arc<ProbeTool>| {
        Workflow::builder()
            .name("recovery-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("coder", coder.clone())
            .global_tool(probe)
            .persistence(config(&workspace))
    };
    let workflow = Arc::new(
        approving(ProbeTool::new("probe", false))
            .approval_policy(crate::policy::ApprovalPolicy::Handler(Arc::new(Undecided)))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    until(|| {
        let saved = std::fs::read_dir(run_dir(&workspace).join("agents"))
            .into_iter()
            .flatten()
            .flatten()
            .any(|task| {
                std::fs::read_to_string(task.path().join("live.json"))
                    .is_ok_and(|json| json.contains("AwaitingToolApproval"))
            });
        saved
            && std::fs::read_to_string(run_dir(&workspace).join("latest-checkpoint"))
                .is_ok_and(|name| name == LIVE_CHECKPOINT_FILE)
    })
    .await;
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let probe = ProbeTool::new("probe", false);
    let resumed = approving(probe.clone()).resume_from(&path).build().unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    assert!(matches!(
        saved.task.current_state,
        arko_agents::TaskState::AwaitingToolApproval { .. }
    ));
    let result = within(resumed.recover_agent(&saved)).await.unwrap();
    assert_eq!(result.summary, "done");
    assert_eq!(probe.calls(), 1);
}

/// While one run runs and another has finished, a versioned checkpoint
/// records the running one with a version of its task (never its live
/// file), and retention keeps what the live checkpoint refers to: the
/// running task's live file and the versions it names. Resuming from the
/// versioned checkpoint recovers the running run.
#[tokio::test]
async fn versioned_checkpoints_record_running_runs_and_retention_keeps_live_files() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if !body.contains("Probe.") {
            complete_turn("done", "done")
        } else if body.contains("function_call_output") {
            complete_turn("probed", "probe done")
        } else {
            tool_call_turn("probe", "probe", json!({}))
        })
    })
    .await;
    let retaining = |probe: Arc<ProbeTool>| {
        builder(&server, &workspace)
            .agent_tool("reviewer", probe)
            .persistence(config(&workspace).keep_last_checkpoints(1))
    };
    let block = ProbeTool::new("probe", true);
    let workflow = Arc::new(retaining(block.clone()).build().unwrap());
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move {
            workflow
                .run_agents_parallel(vec![
                    ("coder", "Write it.".to_string()),
                    ("reviewer", "Probe.".to_string()),
                ])
                .await
        }
    });
    within(block.started.notified()).await;
    until(|| workflow.result().agent_results.len() == 1).await;
    let coder = workflow.result().agent_results[0].clone();
    assert_eq!(coder.agent_name, "coder");
    workflow.checkpoint().await.unwrap();
    workflow.checkpoint().await.unwrap();
    let checkpoints = versioned_checkpoints(&workspace);
    assert_eq!(checkpoints.len(), 1);
    let versioned = read_state(&checkpoints[0]);
    let running = versioned
        .agent_runs
        .iter()
        .find(|record| record.agent_name == "reviewer")
        .unwrap();
    assert_eq!(running.status, AgentRunStatus::Interrupted);
    assert!(!running.task_file.ends_with("live.json"));
    assert!(run_dir(&workspace).join(&running.task_file).is_file());
    assert!(live_file(&workspace, running.agent_run_id).is_file());
    let live = read_state(&run_dir(&workspace).join(LIVE_CHECKPOINT_FILE));
    for record in &live.agent_runs {
        assert!(
            run_dir(&workspace).join(&record.task_file).is_file(),
            "{}",
            record.task_file
        );
    }
    let path = checkpoints[0].clone();
    crash(workflow, vec![run]).await;

    let resumed = retaining(ProbeTool::new("probe", false))
        .resume_from(&path)
        .build()
        .unwrap();
    assert_eq!(
        resumed.agent_result(coder.agent_run_id).unwrap().status,
        AgentRunStatus::Completed
    );
    let reviewer = resumed.interrupted_runs().remove(0);
    assert_eq!(reviewer.agent_name, "reviewer");
    let result = within(resumed.recover_agent(&reviewer)).await.unwrap();
    assert_eq!(result.summary, "probe done");
}

/// A server whose first request calls `probe` and whose later requests
/// complete; while `hold` is set, the later requests wait for it.
async fn probe_then_complete(hold: Option<Arc<tokio::sync::Notify>>) -> ResponsesServer {
    ResponsesServer::start_async(move |index, _| {
        let hold = hold.clone();
        Box::pin(async move {
            if index == 0 {
                return Some(crate::test_support::Reply::Stream(tool_call_turn(
                    "probe",
                    "probe",
                    json!({}),
                )));
            }
            if let Some(hold) = hold {
                hold.notified().await;
            }
            Some(crate::test_support::Reply::Stream(complete_turn(
                "done", "done",
            )))
        })
    })
    .await
}

/// Without auto-checkpoints, `latest-checkpoint` still names the newest
/// state once a run ends: the live checkpoint recording its end, not a
/// versioned checkpoint written while it ran (which records it running).
#[tokio::test]
async fn latest_names_the_newest_checkpoint_without_auto_checkpoints() {
    let workspace = Workspace::new();
    let release = Arc::new(tokio::sync::Notify::new());
    let server = probe_then_complete(Some(release.clone())).await;
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(ProbeTool::new("probe", false))
            .persistence(config(&workspace).auto_checkpoint(false))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    server.wait_for_requests(2).await;
    workflow.checkpoint().await.unwrap();
    let versioned = latest(&workspace);
    assert_ne!(versioned.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
    assert!(
        read_state(&versioned)
            .agent_runs
            .iter()
            .any(|record| record.status == AgentRunStatus::Interrupted)
    );

    release.notify_one();
    let result = within(run).await.unwrap().unwrap();
    let path = latest(&workspace);
    assert_eq!(path.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
    let state = read_state(&path);
    assert_eq!(state.agent_runs.len(), 1);
    assert_eq!(state.agent_runs[0].status, AgentRunStatus::Completed);
    assert!(!live_file(&workspace, result.agent_run_id).exists());
    drop(workflow);
    let resumed = builder(&server, &workspace)
        .global_tool(ProbeTool::new("probe", false))
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    assert!(resumed.interrupted_runs().is_empty());
    assert_eq!(
        resumed.agent_result(result.agent_run_id).unwrap().status,
        AgentRunStatus::Completed
    );
}

/// Runs `coder` (a probe call, then completion) in a workflow that crashes
/// as the run's end reaches `point`; returns the run's result and the
/// checkpoint to resume from.
async fn crash_as_the_run_ends(
    server: &ResponsesServer,
    workspace: &Workspace,
    point: crate::persistence::CrashPoint,
) -> (AgentResult, std::path::PathBuf) {
    let workflow = Arc::new(
        builder(server, workspace)
            .global_tool(ProbeTool::new("probe", false))
            .persistence(config(workspace))
            .build()
            .unwrap(),
    );
    workflow.engine.persistence().unwrap().crash_at(point);
    let result = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    let path = latest(workspace);
    crash(workflow, Vec::<tokio::task::JoinHandle<()>>::new()).await;
    (result, path)
}

/// A crash after a run's result is recorded (its task version written) and
/// before any checkpoint records it: the run comes back interrupted, and
/// recovering it finishes it without running its calls again; each model
/// call is charged once.
#[tokio::test]
async fn a_crash_between_recording_a_result_and_its_checkpoint_is_recovered() {
    let workspace = Workspace::new();
    let server = probe_then_complete(None).await;
    let (ended, path) = crash_as_the_run_ends(
        &server,
        &workspace,
        crate::persistence::CrashPoint::ResultRecorded,
    )
    .await;
    assert_eq!(path.file_name().unwrap(), LIVE_CHECKPOINT_FILE);

    let probe = ProbeTool::new("probe", false);
    let resumed = builder(&server, &workspace)
        .global_tool(probe.clone())
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    assert_eq!(saved.agent_run_id, ended.agent_run_id);
    let result = within(resumed.recover_agent(&saved)).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.summary, "done");
    assert_eq!(probe.calls(), 0);
    assert!(server.requests() <= 3, "{}", server.requests());
    assert_eq!(
        resumed.cost_ledger().entries.len(),
        result.task.usage_history.len()
    );
    assert!(!live_file(&workspace, result.agent_run_id).exists());
    assert!(resumed.interrupted_runs().is_empty());
}

/// A crash after the live checkpoint recorded a run's end, before the
/// versioned checkpoint after it: `latest-checkpoint` names the live
/// checkpoint, which records the run completed.
#[tokio::test]
async fn a_crash_before_the_checkpoint_after_a_run_keeps_its_end() {
    let workspace = Workspace::new();
    let server = probe_then_complete(None).await;
    let (ended, path) = crash_as_the_run_ends(
        &server,
        &workspace,
        crate::persistence::CrashPoint::LiveRunReleased,
    )
    .await;
    assert_eq!(path.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
    let resumed = builder(&server, &workspace)
        .global_tool(ProbeTool::new("probe", false))
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    assert!(resumed.interrupted_runs().is_empty());
    assert_eq!(
        resumed.agent_result(ended.agent_run_id).unwrap().status,
        AgentRunStatus::Completed
    );
    assert!(!live_file(&workspace, ended.agent_run_id).exists());
}

/// A run restored from its live file stays in every checkpoint while its
/// task can't be saved as a version: checkpoints refer to the live file
/// (which is kept), and a resume from them still recovers it.
#[tokio::test]
async fn a_restored_run_stays_recorded_while_it_cannot_be_versioned() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains(RESTARTED) {
            complete_turn("done", "done")
        } else {
            tool_call_turn("probe", "probe", json!({}))
        })
    })
    .await;
    let path = crash_mid_tool(&server, &workspace, ProbeTool::new("probe", true)).await;
    let resumed = builder(&server, &workspace)
        .global_tool(ProbeTool::new("probe", false))
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    let persistence = resumed.engine.persistence().unwrap();
    persistence.fail_task_writes.store(true, Ordering::SeqCst);
    resumed.checkpoint().await.unwrap();
    let checkpoint = latest(&workspace);
    assert_ne!(checkpoint.file_name().unwrap(), LIVE_CHECKPOINT_FILE);
    let state = read_state(&checkpoint);
    let record = state
        .agent_runs
        .iter()
        .find(|record| record.agent_run_id == saved.agent_run_id)
        .expect("still recorded");
    assert_eq!(record.status, AgentRunStatus::Interrupted);
    assert!(record.task_file.ends_with("live.json"));
    assert_eq!(state.missing_task_states.len(), 1);
    assert!(live_file(&workspace, saved.agent_run_id).is_file());
    drop(resumed);

    let resumed = builder(&server, &workspace)
        .global_tool(ProbeTool::new("probe", false))
        .persistence(config(&workspace))
        .resume_from(&checkpoint)
        .build()
        .unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    let result = within(resumed.recover_agent(&saved)).await.unwrap();
    assert_eq!(result.summary, "done");
}

/// Recovering a run whose delegation can't resume (the tool isn't offered
/// any more) supersedes the delegated run and the run that one delegated
/// to in turn: none is left interrupted.
#[tokio::test]
async fn superseding_a_delegated_run_supersedes_its_own_delegations() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        let answered = body.contains("function_call_output");
        Some(if body.contains("Delegate.") {
            if answered {
                complete_turn("coder-done", "coder done")
            } else {
                tool_call_turn(
                    "delegate",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "planner", "prompt": "Relay."}),
                )
            }
        } else if body.contains("Relay.") {
            if answered {
                complete_turn("planner-done", "planner done")
            } else {
                tool_call_turn(
                    "relay",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "reviewer", "prompt": "Block."}),
                )
            }
        } else if answered {
            complete_turn("reviewer-done", "reviewed")
        } else {
            tool_call_turn("block", "block", json!({}))
        })
    })
    .await;
    let chained = |block: Arc<ProbeTool>, coder_calls: bool| {
        let mut services = WorkflowServicesConfig::default()
            .allow_caller("planner")
            .allow_target("planner")
            .allow_target("reviewer");
        if coder_calls {
            services = services.allow_caller("coder");
        }
        builder(&server, &workspace)
            .agent("planner", agent_config("Friendly Planner"))
            .agent_tool("reviewer", block)
            .workflow_services(services)
            .persistence(config(&workspace))
    };
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(chained(block.clone(), true).build().unwrap());
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Delegate.").await }
    });
    within(block.started.notified()).await;
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let resumed = chained(ProbeTool::new("block", false), false)
        .resume_from(&path)
        .build()
        .unwrap();
    for agent in ["planner", "reviewer"] {
        assert_eq!(
            result_of(&resumed, agent).status,
            AgentRunStatus::Interrupted
        );
    }
    let coder = resumed.interrupted_runs().remove(0);
    assert_eq!(coder.agent_name, "coder");
    let result = within(resumed.recover_agent(&coder)).await.unwrap();
    assert_eq!(result.summary, "coder done");
    for agent in ["planner", "reviewer"] {
        let superseded = result_of(&resumed, agent);
        assert_eq!(superseded.status, AgentRunStatus::Cancelled, "{agent}");
    }
    assert!(
        resumed
            .result()
            .agent_results
            .iter()
            .all(|result| result.status != AgentRunStatus::Interrupted)
    );
}

/// A group of calls recovery simply runs again (`Rerun`) doesn't wait for
/// the task to be saved: the call runs while every write is held.
#[tokio::test]
async fn a_rerun_call_does_not_wait_for_the_save() {
    let workspace = Workspace::new();
    let server = probe_then_complete(None).await;
    let rerun = ProbeTool::recovered_by("probe", false, arko_agents::ToolRecovery::Rerun);
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(rerun.clone())
            .persistence(config(&workspace))
            .build()
            .unwrap(),
    );
    let held = workflow.engine.persistence().unwrap().hold_writes().await;
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    within(rerun.started.notified()).await;
    drop(held);
    let result = within(run).await.unwrap().unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
}

/// The live checkpoint is written again only when a resume would read
/// something new from it: not for a running task's progress alone (its
/// live file has that), but for a cost it can't tell from the tasks.
#[tokio::test]
async fn the_live_checkpoint_is_rewritten_only_when_the_state_changes() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|_, _| Some(tool_call_turn("probe", "probe", json!({})))).await;
    let probe = ProbeTool::new("probe", true);
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(probe.clone())
            .persistence(config(&workspace))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    within(probe.started.notified()).await;
    let persistence = workflow.engine.persistence().unwrap().clone();
    let context = workflow.context();
    let live = run_dir(&workspace).join(LIVE_CHECKPOINT_FILE);
    let written = || read_state(&live).checkpoint_id;
    persistence
        .write_live_checkpoint(context, false)
        .await
        .unwrap();
    let before = written();
    persistence
        .write_live_checkpoint(context, true)
        .await
        .unwrap();
    assert_eq!(written(), before, "nothing changed");
    context.budget.add_manual_cost(0.5, "search", None);
    persistence
        .write_live_checkpoint(context, false)
        .await
        .unwrap();
    assert_ne!(written(), before);
    assert_eq!(read_state(&live).cost_ledger.entries.len(), 2);
    crash(workflow, vec![run]).await;
}

/// The scripted fake Claude Code CLI of arko_agents' tests, in a fresh
/// directory, playing `script`.
pub(super) struct FakeClaude(pub(super) std::path::PathBuf);

impl FakeClaude {
    pub(super) fn new(script: Value) -> Self {
        let dir = std::env::temp_dir().join(format!("arko-wf-fake-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("script.json"), script.to_string()).unwrap();
        Self(dir)
    }

    pub(super) fn agent(&self) -> arko_agents::AgentConfig {
        let python = std::env::var("ARKO_TEST_PYTHON")
            .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
        let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../arko_agents/tests/fixtures/fake_claude_agent.py");
        let path = |name: &str| self.0.join(name).display().to_string();
        let mut config = agent_config("Friendly Coder");
        config.llm_config.model =
            arko_ai::Model::claude_code(arko_ai::Model::anthropic_claude_opus_5_5());
        config.claude_code = Some(arko_agents::ClaudeCodeSettings {
            cli_path: Some(python.into()),
            cli_args: vec![script.display().to_string()],
            env: std::collections::BTreeMap::from([
                ("FAKE_CLAUDE_SCRIPT".to_string(), path("script.json")),
                ("FAKE_CLAUDE_LOG".to_string(), path("log.jsonl")),
                ("FAKE_CLAUDE_STATE".to_string(), path("state")),
            ]),
            ..Default::default()
        });
        config
    }

    /// The fake's log entries of kind `event`.
    pub(super) fn logged(&self, event: &str) -> Vec<Value> {
        std::fs::read_to_string(self.0.join("log.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|entry| entry["event"] == event)
            .collect()
    }
}

impl Drop for FakeClaude {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A Claude Code agent whose CLI can't start fails under `Automatic`, as an
/// agent of arko's own runtime does for an error retrying can't fix: the
/// runtime reports the pause's cause before the error, so the policy sees
/// it (it saw an unknown cause before, and returned the agent paused).
#[tokio::test]
async fn a_claude_code_agent_that_cant_start_fails_with_its_cause() {
    let workspace = Workspace::new();
    let fake = FakeClaude::new(json!([]));
    let mut config = fake.agent();
    let claude_code = config.claude_code.as_mut().unwrap();
    claude_code.cli_args = vec!["-c".into(), "import sys; sys.exit(1)".into()];
    let workflow = Workflow::builder()
        .name("cant-start")
        .workspace(&workspace.0)
        .llm_client(Arc::new(arko_ai::LLMClient::new()))
        .agent("coder", config)
        .build()
        .unwrap();
    let error = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        workflow.run_agent("coder", "Write it."),
    )
    .await
    .expect("the run returned")
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("can't succeed without a change to the request, a credential"),
        "{error}"
    );
    let result = &workflow.result().agent_results[0];
    assert_eq!(result.status, AgentRunStatus::Failed);
    assert_eq!(
        result.failure_cause(),
        Some(&arko_agents::ErrorRecoveryCause::NonRetryable)
    );
    assert!(
        result
            .error
            .as_deref()
            .unwrap()
            .contains("Claude Code could not start")
    );
}

/// A Claude Code agent that isn't signed in (or whose login expired) fails
/// under `Automatic` with that cause: Claude Code starts, and reports it
/// on the first turn, as `authentication_failed` (it was retried for ever,
/// as a session failure).
#[tokio::test]
async fn a_claude_code_agent_that_cant_sign_in_fails_with_its_cause() {
    let workspace = Workspace::new();
    let fake = FakeClaude::new(json!([[
        {"api_error": "authentication_failed", "text": "Not logged in · Please run /login", "status": null},
    ]]));
    let workflow = Workflow::builder()
        .name("cant-sign-in")
        .workspace(&workspace.0)
        .llm_client(Arc::new(arko_ai::LLMClient::new()))
        .agent("coder", fake.agent())
        .build()
        .unwrap();
    let error = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("a credential"), "{error}");
    let result = &workflow.result().agent_results[0];
    assert_eq!(
        result.failure_cause(),
        Some(&arko_agents::ErrorRecoveryCause::NonRetryable)
    );
    assert!(
        result.error.as_deref().unwrap().contains("Not logged in"),
        "{:?}",
        result.error
    );
    assert_eq!(fake.logged("argv").len(), 1, "not retried");
}

/// A Claude Code agent's refused input can't be changed (Claude Code keeps
/// its own conversation): `ResolveRefusal` is refused at once, and the
/// agent stays paused for another decision.
#[tokio::test]
async fn a_claude_code_refusal_cant_be_resolved_by_changing_its_input() {
    let workspace = Workspace::new();
    let fake = FakeClaude::new(json!([[
        {"system": {"subtype": "model_refusal", "api_refusal_category": "cyber",
                    "api_refusal_explanation": "The request was refused."}},
        {"text": "No."},
    ]]));
    let workflow = Workflow::builder()
        .name("cc-refusal")
        .workspace(&workspace.0)
        .llm_client(Arc::new(arko_ai::LLMClient::new()))
        .agent("coder", fake.agent())
        .error_recovery_policy(crate::policy::ErrorRecoveryPolicy::ReturnToCaller)
        .build()
        .unwrap();
    let error = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap_err();
    let WorkflowError::AgentPaused {
        task_id,
        pause: crate::policy::AgentPause::ErrorRecovery { cause, .. },
        ..
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(cause, arko_agents::ErrorRecoveryCause::Refusal);
    let paused = workflow.agent_result(task_id).unwrap();
    let change = arko_agents::RefusalResolution::RemoveUserMessage { message_index: 0 };
    let refused = within(workflow.resume_agent(
        &paused,
        crate::policy::AgentResume::ResolveRefusal(vec![change]),
    ))
    .await
    .unwrap_err();
    assert!(
        matches!(&refused, WorkflowError::BuilderInvalid(reason) if reason.contains("Claude Code")),
        "{refused:?}"
    );
    let failed = within(workflow.resume_agent(
        &paused,
        crate::policy::AgentResume::FailAfterError { reason: None },
    ))
    .await
    .unwrap_err();
    assert!(failed.to_string().contains("rejected"), "{failed}");
}

/// A tool whose every call fails.
#[derive(Debug)]
struct FailingTool;

#[async_trait::async_trait]
impl arko_agents::Tool for FailingTool {
    fn definition(&self) -> arko_agents::ToolDefinition {
        arko_agents::ToolDefinition {
            name: "boom".to_string(),
            description: "Fails.".to_string(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: arko_agents::ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &arko_agents::TaskContext<'_>,
        _params: std::collections::HashMap<String, Value>,
    ) -> Result<String, arko_agents::ToolError> {
        Err(arko_agents::ToolError::ExecutionFailed("boom".into()))
    }
}

/// A Claude Code agent's pause after its turn (here a tool error under
/// `ToolErrorPolicy::PauseAndAwaitApproval`) is decided by the workflow:
/// `Automatic` continues it. (Claude Code reported such a pause by its
/// state alone, and the run call waited for ever.) The error is reported
/// once, by the pause, naming its call, as arko's own runtime reports it
/// (it was reported first on its own too).
#[tokio::test]
async fn a_claude_code_agents_tool_error_pause_is_continued() {
    use arko_agents::ToolErrorPolicy;
    let boom = json!({"tool": "mcp__arko__boom", "input": {}, "id": "toolu_boom"});
    let complete = json!({"tool": "mcp__arko__complete_task", "input": {"summary": "done"}});
    for (policy, script, turns, reports) in [
        // One turn: the error is the call's result, and the model goes on.
        (
            ToolErrorPolicy::InformAgentAndContinue,
            json!([[boom, complete]]),
            1,
            (false, true),
        ),
        (
            ToolErrorPolicy::PauseAndAwaitApproval,
            json!([[boom], [complete]]),
            2,
            (true, true),
        ),
    ] {
        let workspace = Workspace::new();
        let fake = FakeClaude::new(script);
        let mut config = fake.agent();
        config.error_recovery_config.on_tool_execution_error = policy.clone();
        let workflow = Workflow::builder()
            .name("tool-error")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("coder", config)
            .global_tool(Arc::new(FailingTool))
            .build()
            .unwrap();
        let mut events = workflow.subscribe_events();
        let result = within(workflow.run_agent("coder", "Boom it."))
            .await
            .unwrap();
        assert_eq!(result.summary, "done");
        assert_eq!(fake.logged("user").len(), turns, "{policy:?}");
        // (Needs a decision, names its call.)
        let mut errors = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let arko_agents::runtime::event::AgentEvent::ErrorEncountered {
                requires_approval_to_retry,
                tool_call_id,
                ..
            } = event.event
            {
                errors.push((requires_approval_to_retry, tool_call_id.is_some()));
            }
        }
        assert_eq!(errors, [reports], "{policy:?}");
    }
}

/// A Claude Code agent whose completion its verifier can't review is
/// returned as `VerifierFailed` (the pause was reported by its state alone,
/// and the run call waited for ever), and continued with `continue_agent`
/// it is reviewed again.
#[tokio::test]
async fn a_claude_code_agent_its_verifier_cant_review_is_returned_and_continued() {
    let workspace = Workspace::new();
    let complete = json!([{"tool": "mcp__arko__complete_task", "input": {"summary": "done"}}]);
    let fake = FakeClaude::new(json!([complete, complete]));
    let reviews = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server = ResponsesServer::start({
        let reviews = reviews.clone();
        move |index, _| {
            // Unusable at first, then accepted.
            let verdict = if reviews.fetch_add(1, Ordering::SeqCst) == 0 {
                "Looks good to me."
            } else {
                "ACCEPTED"
            };
            Some(complete_turn(&format!("v{index}"), verdict))
        }
    })
    .await;
    let workflow = Workflow::builder()
        .name("cc-verifier")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", fake.agent())
        .agent("reviewer", agent_config("Friendly Reviewer"))
        .completion_verifier(
            crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
                .target("coder")
                .format_retries(0)
                .review_retries(0),
        )
        .build()
        .unwrap();
    let error = within(workflow.run_agent("coder", "Write it."))
        .await
        .unwrap_err();
    let WorkflowError::VerifierFailed { task_id, .. } = error else {
        panic!("{error:?}");
    };
    let paused = workflow.agent_result(task_id).unwrap();
    assert!(matches!(
        paused.task.current_state,
        arko_agents::TaskState::AwaitingErrorRecoveryApproval {
            cause: arko_agents::ErrorRecoveryCause::Middleware,
            ..
        }
    ));
    let result = within(workflow.continue_agent(&paused, "Have it reviewed again."))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(reviews.load(Ordering::SeqCst), 2);
}

/// A Claude Code agent running a tool call when the workflow crashed is
/// recovered like any other: its session resumes (`--resume`), and its
/// first message says the call's outcome is unknown.
#[tokio::test]
async fn a_claude_code_agent_is_recovered_after_a_crash() {
    let workspace = Workspace::new();
    let fake = FakeClaude::new(json!([
        [{"tool": "mcp__arko__probe", "input": {}, "id": "toolu_probe"}],
        [{"tool": "mcp__arko__complete_task", "input": {"summary": "recovered"}}],
    ]));
    let server = ResponsesServer::start(|_, _| None).await;
    let claude_code = |probe: Arc<ProbeTool>| {
        Workflow::builder()
            .name("recovery-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("coder", fake.agent())
            .global_tool(probe)
            .persistence(config(&workspace))
    };
    let probe = ProbeTool::new("probe", true);
    let workflow = Arc::new(claude_code(probe.clone()).build().unwrap());
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Probe it.").await }
    });
    within(probe.started.notified()).await;
    assert!(running_call_saved(&workspace));
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let resumed = claude_code(ProbeTool::new("probe", false))
        .resume_from(&path)
        .build()
        .unwrap();
    let saved = resumed.interrupted_runs().remove(0);
    assert!(saved.task.claude_code_session.is_some());
    assert_eq!(saved.task.running_tool_calls.len(), 1);
    let sent_before = fake.logged("user").len();
    let result = within(resumed.recover_agent(&saved)).await.unwrap();
    assert_eq!(
        result.status,
        AgentRunStatus::Completed,
        "{:?}",
        result.error
    );
    assert_eq!(result.summary, "recovered");
    assert_eq!(result.agent_run_id, saved.agent_run_id);
    let first = fake.logged("user")[sent_before]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        first.contains("`mcp__arko__probe` (toolu_probe)"),
        "{first}"
    );
    assert!(first.contains("its outcome is unknown"), "{first}");
    let argv = fake.logged("argv").last().unwrap()["argv"].to_string();
    assert!(argv.contains("--resume="), "{argv}");
    assert_eq!(server.requests(), 0);
}

/// The agent run a `run_workflow_agent` result names as resumable
/// (`resume_run "<id>"`), in a request body (JSON, quotes escaped).
fn named_run(body: &str) -> Option<uuid::Uuid> {
    let start = body.rfind("resume_run \\\"")?;
    let rest = &body[start..];
    rest.char_indices()
        .find_map(|(index, _)| rest.get(index..index + 36)?.parse().ok())
}

/// The number of tool results a request sends.
fn outputs(body: &str) -> usize {
    body.matches("function_call_output").count()
}

/// A server for orphaned delegations. `coder` ("Delegate.") runs
/// `reviewer` ("Block."); once its call failed naming the reviewer's run,
/// it resumes that run once ("Finish the review."), or, while
/// `hang_after_note` is set, its request waits; then it completes.
/// `planner` ("Resume the review.") resumes the run in `run` with the same
/// message, then completes. `reviewer` calls `block`; resumed with the
/// message, it calls `block` once more (`block-again`); it completes once
/// a call has a result.
async fn orphan_server(
    run: Arc<parking_lot::Mutex<Option<uuid::Uuid>>>,
    hang_after_note: Arc<AtomicBool>,
) -> ResponsesServer {
    ResponsesServer::start(move |_, body| {
        let resume = |run: uuid::Uuid, call: &str| {
            tool_call_turn(
                call,
                RUN_WORKFLOW_AGENT_TOOL_NAME,
                json!({"agent": "reviewer", "prompt": "Finish the review.", "resume_run": run.to_string()}),
            )
        };
        if body.contains("Delegate.") {
            match (outputs(body), named_run(body)) {
                (0, _) => Some(tool_call_turn(
                    "delegate",
                    RUN_WORKFLOW_AGENT_TOOL_NAME,
                    json!({"agent": "reviewer", "prompt": "Block."}),
                )),
                (1, Some(_)) if hang_after_note.load(Ordering::SeqCst) => None,
                (1, Some(named)) => Some(resume(named, "resume")),
                _ => Some(complete_turn("coder-done", "coder done")),
            }
        } else if body.contains("Resume the review.") {
            Some(if outputs(body) == 0 {
                resume(run.lock().unwrap(), "resume")
            } else {
                complete_turn("planner-done", "planner done")
            })
        } else if body.contains("Finish the review.") && !body.contains("call_block-again") {
            Some(tool_call_turn("block-again", "block", json!({})))
        } else if outputs(body) > 0 {
            Some(complete_turn("reviewer-done", "reviewed"))
        } else {
            Some(tool_call_turn("block", "block", json!({})))
        }
    })
    .await
}

/// A crash while `coder`'s delegated reviewer runs, then a resume where the
/// reviewer's agent budget is spent: recovering the coder's call fails, and
/// its result names the reviewer's run and says the coder can resume it.
/// The coder tries (`resume_run`), is refused by the budget again, and
/// completes: the reviewer is orphaned, listed for the host with its parent
/// link. Returns the checkpoint after that and the reviewer's result.
async fn orphan_delegation(
    server: &ResponsesServer,
    workspace: &Workspace,
) -> (std::path::PathBuf, AgentResult) {
    let path = crash_mid_delegation(server, workspace).await;
    let resumed = delegating(server, workspace, ProbeTool::new("block", false), 4)
        // Refused at admission: the recovery of the delegated run fails.
        .budget_exceeded_policy(BudgetExceededPolicy::Cancel)
        .agent_budget("reviewer", 1e-12)
        .resume_from(&path)
        .build()
        .unwrap();
    let interrupted = resumed.interrupted_runs();
    assert_eq!(
        interrupted.len(),
        1,
        "the reviewer is its parent's to resume"
    );
    let reviewer = result_of(&resumed, "reviewer");
    let result = within(resumed.recover_agent(&interrupted[0]))
        .await
        .unwrap();
    assert_eq!(result.summary, "coder done");

    let id = reviewer.agent_run_id.to_string();
    let bodies = server.bodies();
    let coder = |results| {
        bodies
            .iter()
            .find(|body| body.contains("Delegate.") && outputs(body) == results)
            .unwrap()
    };
    let told = coder(1);
    assert!(told.contains("exceeded its budget"), "{told}");
    assert!(told.contains(&format!(
        "Agent run {id} of agent 'reviewer' was interrupted"
    )));
    assert!(told.contains("You can resume it"), "{told}");
    assert_eq!(named_run(told), Some(reviewer.agent_run_id));
    let refused = coder(2);
    assert!(
        refused.matches("exceeded its budget").count() >= 2,
        "{refused}"
    );

    let orphans = resumed.interrupted_runs();
    assert_eq!(orphans.len(), 1);
    assert_eq!(orphans[0].agent_run_id, reviewer.agent_run_id);
    assert_eq!(orphans[0].task.parent, reviewer.task.parent);
    let error = resumed.recover_agent(&orphans[0]).await.unwrap_err();
    assert!(
        matches!(error, WorkflowError::AgentBudgetExceeded { .. }),
        "{error}"
    );
    let path = latest(workspace);
    drop(resumed);
    (path, reviewer)
}

/// A delegated run whose recovery through its parent's call fails is named
/// in the call's result and left interrupted; once its parent completed,
/// it is orphaned: listed by `interrupted_runs` (with its parent link) and
/// recovered by the host, which releases the link. It runs once.
#[tokio::test]
async fn a_delegation_that_cannot_be_recovered_is_named_and_left_to_the_host() {
    let workspace = Workspace::new();
    let server = orphan_server(Arc::default(), Arc::default()).await;
    let (path, reviewer) = orphan_delegation(&server, &workspace).await;

    let block = ProbeTool::new("block", false);
    let resumed = delegating(&server, &workspace, block.clone(), 4)
        .resume_from(&path)
        .build()
        .unwrap();
    let orphans = resumed.interrupted_runs();
    assert_eq!(orphans.len(), 1);
    assert_eq!(orphans[0].task.parent, reviewer.task.parent);
    let result = within(resumed.recover_agent(&orphans[0])).await.unwrap();
    assert_eq!(result.summary, "reviewed");
    assert_eq!(result.agent_run_id, reviewer.agent_run_id);
    assert_eq!(result.task.parent, None, "the host's own now");
    assert_eq!(block.calls(), 0, "in doubt");
    assert!(resumed.interrupted_runs().is_empty());
}

/// Another agent resumes an orphaned run with `resume_run`: the run
/// continues its own task with the caller's message, and its parent link
/// points at the resuming call, so that after a crash only the caller is
/// listed and recovering it recovers the run in that call.
#[tokio::test]
async fn an_agent_resumes_an_orphaned_run_and_a_later_crash_finds_it_there() {
    let workspace = Workspace::new();
    let run = Arc::new(parking_lot::Mutex::new(None));
    let server = orphan_server(run.clone(), Arc::default()).await;
    let (path, reviewer) = orphan_delegation(&server, &workspace).await;
    *run.lock() = Some(reviewer.agent_run_id);

    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(
        delegating(&server, &workspace, block.clone(), 4)
            .resume_from(&path)
            .build()
            .unwrap(),
    );
    let planner = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("planner", "Resume the review.").await }
    });
    within(block.started.notified()).await;
    let resumed_body = server
        .bodies()
        .into_iter()
        .rfind(|body| body.contains("Finish the review.") && !body.contains("Resume the review."))
        .unwrap();
    assert!(resumed_body.contains(RESTARTED), "{resumed_body}");
    assert!(
        resumed_body.contains("Workflow context update"),
        "{resumed_body}"
    );
    let saved: Value = serde_json::from_str(
        &std::fs::read_to_string(live_file(&workspace, reviewer.agent_run_id)).unwrap(),
    )
    .unwrap();
    let link: arko_agents::TaskParent = serde_json::from_value(saved["parent"].clone()).unwrap();
    assert_ne!(Some(link), reviewer.task.parent);
    assert!(running_call_saved(&workspace));
    let path = latest(&workspace);
    crash(workflow, vec![planner]).await;

    let block = ProbeTool::new("block", false);
    let resumed = delegating(&server, &workspace, block.clone(), 4)
        .resume_from(&path)
        .build()
        .unwrap();
    let interrupted = resumed.interrupted_runs();
    assert_eq!(interrupted.len(), 1);
    assert_eq!(interrupted[0].agent_name, "planner");
    assert_eq!(link.task_id, interrupted[0].agent_run_id);
    assert_eq!(
        resumed
            .agent_result(reviewer.agent_run_id)
            .unwrap()
            .task
            .parent,
        Some(link)
    );
    let result = within(resumed.recover_agent(&interrupted[0]))
        .await
        .unwrap();
    assert_eq!(result.summary, "planner done");
    let recovered = resumed.agent_result(reviewer.agent_run_id).unwrap();
    assert_eq!(recovered.status, AgentRunStatus::Completed);
    assert_eq!(recovered.task.parent, Some(link));
    assert_eq!(block.calls(), 0, "in doubt");
    assert!(resumed.interrupted_runs().is_empty());
}

/// A run whose parent's call was answered (it failed, naming the run)
/// before the parent stopped is not superseded when the parent is
/// recovered: the parent resumes it (`resume_run`), in a new call its
/// link then points at. Meanwhile it is listed as orphaned.
#[tokio::test]
async fn a_run_whose_call_was_answered_stays_for_its_parent_to_resume() {
    let workspace = Workspace::new();
    let hang = Arc::new(AtomicBool::new(true));
    let server = orphan_server(Arc::default(), hang.clone()).await;
    let path = crash_mid_delegation(&server, &workspace).await;
    let workflow = Arc::new(
        delegating(&server, &workspace, ProbeTool::new("block", false), 4)
            // Refused at admission: the recovery of the delegated run fails.
            .budget_exceeded_policy(BudgetExceededPolicy::Cancel)
            .agent_budget("reviewer", 1e-12)
            .resume_from(&path)
            .build()
            .unwrap(),
    );
    let reviewer = result_of(&workflow, "reviewer");
    let coder = workflow.interrupted_runs().remove(0);
    let recovering = tokio::spawn({
        let (workflow, coder) = (workflow.clone(), coder.clone());
        async move { workflow.recover_agent(&coder).await }
    });
    // The coder's task is saved with the failed call's result naming the
    // reviewer (its request waits).
    until(|| {
        std::fs::read_to_string(live_file(&workspace, coder.agent_run_id)).is_ok_and(|task| {
            !task.contains("running_tool_calls") && task.contains("You can resume it")
        }) && std::fs::read_to_string(latest(&workspace))
            .is_ok_and(|state| state.contains(&coder.agent_run_id.to_string()))
    })
    .await;
    let path = latest(&workspace);
    crash(workflow, vec![recovering]).await;

    hang.store(false, Ordering::SeqCst);
    let resumed = delegating(&server, &workspace, ProbeTool::new("block", false), 4)
        .resume_from(&path)
        .build()
        .unwrap();
    let mut listed = resumed
        .interrupted_runs()
        .into_iter()
        .map(|run| run.agent_name)
        .collect::<Vec<_>>();
    listed.sort();
    assert_eq!(listed, ["coder", "reviewer"]);
    let result = within(resumed.recover_agent(&coder)).await.unwrap();
    assert_eq!(result.summary, "coder done");
    let resumed_reviewer = resumed.agent_result(reviewer.agent_run_id).unwrap();
    assert_eq!(resumed_reviewer.status, AgentRunStatus::Completed);
    let link = resumed_reviewer.task.parent.unwrap();
    assert_eq!(link.task_id, coder.agent_run_id);
    assert_ne!(Some(link), reviewer.task.parent, "the resuming call");
    assert!(resumed.interrupted_runs().is_empty());
    // A host recovering the runs it listed before gets the outcome of the
    // run its parent resumed meanwhile.
    let requests = server.requests();
    let outcome = within(resumed.recover_agent(&reviewer)).await.unwrap();
    assert_eq!(outcome.summary, "reviewed");
    assert_eq!(server.requests(), requests, "not run again");
}

async fn refusal(
    services: &crate::services::WorkflowServices,
    caller: &str,
    caller_run: uuid::Uuid,
    agent: &str,
    run: uuid::Uuid,
) -> String {
    services
        .resume_run(
            caller,
            caller_run,
            agent,
            run,
            "Finish the review.",
            1,
            None,
        )
        .await
        .unwrap_err()
        .to_string()
}

/// `resume_run` resumes a run of the named agent that the caller's run
/// started (or an orphaned one), within the active-call limit; anything
/// else is refused with the reason. Resumed, the run continues its own
/// task with the message, inside the new call.
#[tokio::test]
async fn resume_run_resumes_only_a_run_its_caller_may_resume() {
    let workspace = Workspace::new();
    let server = delegation_server(Arc::default()).await;
    let path = crash_mid_delegation(&server, &workspace).await;
    let resumed = builder(&server, &workspace)
        .agent("planner", agent_config("Friendly Planner"))
        .agent_tool("reviewer", ProbeTool::new("block", false))
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("coder")
                .allow_caller("planner")
                .allow_target("reviewer")
                .allow_target("coder")
                .max_active_calls(1),
        )
        .persistence(config(&workspace))
        .resume_from(&path)
        .build()
        .unwrap();
    let services = resumed.context().workflow_services.get().unwrap().clone();
    let coder = resumed.interrupted_runs().remove(0);
    let reviewer = result_of(&resumed, "reviewer");
    let (someone, unknown) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());

    let cases = [
        ("planner", someone, "reviewer", unknown, "no such agent run"),
        (
            "planner",
            someone,
            "coder",
            reviewer.agent_run_id,
            "it is a run of agent 'reviewer', not 'coder'",
        ),
        (
            "planner",
            someone,
            "coder",
            coder.agent_run_id,
            "run_workflow_agent didn't start it",
        ),
        (
            "planner",
            someone,
            "reviewer",
            reviewer.agent_run_id,
            "can still resume it",
        ),
        (
            "reviewer",
            someone,
            "coder",
            coder.agent_run_id,
            "not authorized",
        ),
    ];
    for (caller, caller_run, agent, run, expected) in cases {
        let error = refusal(&services, caller, caller_run, agent, run).await;
        assert!(error.contains(expected), "{expected}: {error}");
    }
    // A run that failed can't be resumed (a cancelled one is reopened: see
    // `budget_tests`).
    let mut cancelled = reviewer.task.clone();
    cancelled.id = uuid::Uuid::new_v4();
    cancelled.set_state(arko_agents::TaskState::Failed {
        reason: "stopped".to_string(),
        cause: None,
        error_kind: None,
    });
    let cancelled_id = cancelled.id;
    crate::workflow::result::upsert_result(
        &mut resumed.context().agent_results.write(),
        AgentResult::of_task("reviewer".to_string(), cancelled),
    );
    let error = refusal(
        &services,
        "coder",
        coder.agent_run_id,
        "reviewer",
        cancelled_id,
    )
    .await;
    assert!(error.contains("it ended failed (stopped)"), "{error}");
    // Nor can a run another replaced after a restart (superseded).
    let mut superseded = reviewer.task.clone();
    superseded.id = uuid::Uuid::new_v4();
    superseded.set_state(arko_agents::TaskState::Cancelled {
        reason: crate::workflow::engine::SUPERSEDED_AFTER_RESTART.to_string(),
    });
    let superseded_id = superseded.id;
    crate::workflow::result::upsert_result(
        &mut resumed.context().agent_results.write(),
        AgentResult::of_task("reviewer".to_string(), superseded),
    );
    let error = refusal(
        &services,
        "coder",
        coder.agent_run_id,
        "reviewer",
        superseded_id,
    )
    .await;
    assert!(error.contains("superseded it"), "{error}");
    // It counts against the active-call limit.
    let held = services.hold_call();
    let error = refusal(
        &services,
        "coder",
        coder.agent_run_id,
        "reviewer",
        reviewer.agent_run_id,
    )
    .await;
    assert!(error.contains("active-call limit (1) reached"), "{error}");
    drop(held);
    assert_eq!(server.requests(), 2, "nothing ran");

    let call = arko_agents::TaskParent {
        task_id: coder.agent_run_id,
        tool_call_id: uuid::Uuid::new_v4(),
    };
    let result = within(services.resume_run(
        "coder",
        coder.agent_run_id,
        "reviewer",
        reviewer.agent_run_id,
        "Finish the review.",
        1,
        Some(call),
    ))
    .await
    .unwrap();
    assert_eq!(result.summary, "reviewed");
    assert_eq!(result.agent_run_id, reviewer.agent_run_id);
    assert_eq!(result.task.parent, Some(call));
    let body = server.bodies().pop().unwrap();
    assert!(body.contains("Finish the review."), "{body}");
    assert!(body.contains(RESTARTED), "{body}");
}

/// A delegated run already being recovered when its parent's call
/// recovers it is waited for, not run twice: the call returns the outcome
/// that recovery recorded. The host can't recover it meanwhile.
#[tokio::test]
async fn a_delegated_run_being_recovered_is_waited_for_not_run_twice() {
    let workspace = Workspace::new();
    let release = Arc::new(tokio::sync::Notify::new());
    let server = ResponsesServer::start_async({
        let release = release.clone();
        move |_, body| {
            let release = release.clone();
            Box::pin(async move {
                let answered = outputs(&body) > 0;
                Some(crate::test_support::Reply::Stream(
                    if body.contains("Delegate") {
                        if answered {
                            complete_turn("coder-done", "coder done")
                        } else {
                            tool_call_turn(
                                "delegate",
                                RUN_WORKFLOW_AGENT_TOOL_NAME,
                                json!({"agent": "reviewer", "prompt": "Block."}),
                            )
                        }
                    } else if answered {
                        release.notified().await;
                        complete_turn("reviewer-done", "reviewed")
                    } else {
                        tool_call_turn("block", "block", json!({}))
                    },
                ))
            })
        }
    })
    .await;
    let path = crash_mid_delegation(&server, &workspace).await;
    let resumed = Arc::new(
        delegating(&server, &workspace, ProbeTool::new("block", false), 4)
            .resume_from(&path)
            .build()
            .unwrap(),
    );
    let reviewer = result_of(&resumed, "reviewer");
    let coder = resumed.interrupted_runs().remove(0);
    let first = tokio::spawn({
        let (resumed, reviewer) = (resumed.clone(), reviewer.clone());
        async move {
            resumed
                .engine
                .recover_nested(&reviewer, resumed.context(), nested(1))
                .await
        }
    });
    server.wait_for_requests(3).await;
    let second = tokio::spawn({
        let (resumed, coder) = (resumed.clone(), coder.clone());
        async move { resumed.recover_agent(&coder).await }
    });
    until(|| resumed.context().registry.is_running(coder.agent_run_id)).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(server.requests(), 3, "the coder's call waits");
    let error = resumed.recover_agent(&reviewer).await.unwrap_err();
    assert!(error.to_string().contains("which is running"), "{error}");

    release.notify_one();
    let first = within(first).await.unwrap().unwrap();
    assert_eq!(first.summary, "reviewed");
    let second = within(second).await.unwrap().unwrap();
    assert_eq!(second.summary, "coder done");
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 4, "the reviewer ran once");
    assert!(bodies[3].contains("reviewed"), "{}", bodies[3]);
}

/// A Claude Code agent whose delegation can't be recovered is told, in
/// the first message of its resumed session, which run it ran and that it
/// can resume it; once it completed, the run is orphaned for the host.
#[tokio::test]
async fn a_claude_code_parent_is_told_which_delegated_run_it_can_resume() {
    let workspace = Workspace::new();
    let fake = FakeClaude::new(json!([
        [{"tool": "mcp__arko__run_workflow_agent", "input": {"agent": "reviewer", "prompt": "Block."}, "id": "toolu_delegate"}],
        [{"tool": "mcp__arko__complete_task", "input": {"summary": "recovered"}}],
    ]));
    let server = delegation_server(Arc::default()).await;
    let claude_code = |block: Arc<ProbeTool>| {
        Workflow::builder()
            .name("recovery-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("coder", fake.agent())
            .agent("reviewer", agent_config("Friendly Reviewer"))
            .agent_tool("reviewer", block)
            .workflow_services(
                WorkflowServicesConfig::default()
                    .allow_caller("coder")
                    .allow_target("reviewer"),
            )
            .persistence(config(&workspace))
    };
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(claude_code(block.clone()).build().unwrap());
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Delegate.").await }
    });
    within(block.started.notified()).await;
    assert!(running_call_saved(&workspace));
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let resumed = claude_code(ProbeTool::new("block", false))
        // Refused at admission: the recovery of the delegated run fails.
        .budget_exceeded_policy(BudgetExceededPolicy::Cancel)
        .agent_budget("reviewer", 1e-12)
        .resume_from(&path)
        .build()
        .unwrap();
    let coder = resumed.interrupted_runs().remove(0);
    assert_eq!(coder.agent_name, "coder");
    let reviewer = result_of(&resumed, "reviewer");
    let sent_before = fake.logged("user").len();
    let result = within(resumed.recover_agent(&coder)).await.unwrap();
    assert_eq!(result.summary, "recovered", "{:?}", result.error);
    let first = fake.logged("user")[sent_before]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        first.contains("`mcp__arko__run_workflow_agent` (toolu_delegate)"),
        "{first}"
    );
    assert!(
        first.contains(&format!(
            "Agent run {} of agent 'reviewer' was interrupted",
            reviewer.agent_run_id
        )),
        "{first}"
    );
    assert!(
        first.contains(&format!("resume_run \"{}\"", reviewer.agent_run_id)),
        "{first}"
    );
    let orphans = resumed.interrupted_runs();
    assert_eq!(orphans.len(), 1);
    assert_eq!(orphans[0].agent_run_id, reviewer.agent_run_id);
}

/// A completion verifier's review interrupted by a crash is superseded when
/// its target is recovered, never run again: resuming doesn't check its
/// agent's configuration, which may have changed.
#[tokio::test]
async fn a_review_nobody_resumes_does_not_need_its_configuration() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|_, body| {
        Some(if body.contains(VERIFIER) {
            if outputs(body) > 0 {
                complete_turn("verdict", "ACCEPTED")
            } else {
                tool_call_turn("block", "block", json!({}))
            }
        } else {
            complete_turn("done", "done")
        })
    })
    .await;
    let verifying = |block: Arc<ProbeTool>, reviewer: arko_agents::AgentConfig| {
        Workflow::builder()
            .name("recovery-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("coder", agent_config("Friendly Coder"))
            .agent("reviewer", reviewer)
            .agent_tool("reviewer", block)
            .completion_verifier(
                crate::verification::CompletionVerifierConfig::correctness("check", "reviewer")
                    .target("coder"),
            )
            .persistence(config(&workspace))
    };
    let block = ProbeTool::new("block", true);
    let workflow = Arc::new(
        verifying(block.clone(), agent_config("Friendly Reviewer"))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    within(block.started.notified()).await;
    let path = latest(&workspace);
    crash(workflow, vec![run]).await;

    let mut changed = agent_config("Friendly Reviewer");
    changed.instructions = "stricter instructions".to_string();
    let resumed = verifying(ProbeTool::new("block", false), changed)
        .resume_from(&path)
        .build()
        .unwrap();
    let stopped_review = result_of(&resumed, "reviewer");
    assert_eq!(stopped_review.status, AgentRunStatus::Interrupted);
    let coder = resumed.interrupted_runs().remove(0);
    let result = within(resumed.recover_agent(&coder)).await.unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(
        resumed
            .agent_result(stopped_review.agent_run_id)
            .unwrap()
            .status,
        AgentRunStatus::Cancelled
    );
}

/// The images of a task the journal already holds are not copied again at
/// each safe point: the latest snapshot is refreshed in place, to equal the
/// task, and keeps its images. With `--nocapture` the test prints what a
/// safe point costs against a clone of the task (about 15 MB of images).
#[tokio::test]
async fn the_journal_keeps_the_images_it_holds_at_each_safe_point() {
    use crate::persistence::TaskJournal;
    use arko_agents::TaskSnapshotSink as _;

    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|_, _| Some(tool_call_turn("probe", "probe", json!({})))).await;
    let probe = ProbeTool::new("probe", true);
    let workflow = Arc::new(
        builder(&server, &workspace)
            .global_tool(probe.clone())
            .persistence(config(&workspace))
            .build()
            .unwrap(),
    );
    let run = tokio::spawn({
        let workflow = workflow.clone();
        async move { workflow.run_agent("coder", "Write it.").await }
    });
    // The run waits in its tool, so these reports are the only ones.
    within(probe.started.notified()).await;
    let manager = workflow.engine.persistence().unwrap().clone();
    let journal = TaskJournal::new(manager.clone(), workflow.context.clone());
    let id = std::fs::read_dir(run_dir(&workspace).join("agents"))
        .unwrap()
        .flatten()
        .find_map(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .parse::<uuid::Uuid>()
                .ok()
        })
        .unwrap();
    let mut task = manager.latest_live_task(id).unwrap();

    // About 15 MB of screenshots (ten, a step each).
    let screenshot = |fill: u8| {
        let mut bytes = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x10\0\0\0\x10".to_vec();
        bytes.resize(1_100_000, fill);
        arko_ai::Image::from_bytes(&bytes, "image/png")
    };
    for fill in 0..10 {
        task.conversation
            .add_user_message_with_image(&format!("Screenshot {fill}."), screenshot(fill));
        task.conversation
            .add_assistant_message(&format!("Seen {fill}."));
    }
    // A durable report returns once the snapshot is written, and the writer
    // has let go of it.
    journal.task_changed(&task, true).await.unwrap();
    let held = manager.live_image_addresses(id);
    assert_eq!(held.len(), 10);

    // Safe points that add only text: the snapshot keeps its images.
    let (mut cloning, mut reporting) = (Duration::ZERO, Duration::ZERO);
    for step in 0..5 {
        task.conversation
            .add_user_message(&format!("Carry on {step}."));
        task.conversation
            .add_assistant_message(&format!("Carrying on {step}."));
        let started = std::time::Instant::now();
        let copy = task.clone();
        cloning += started.elapsed();
        drop(copy);
        let started = std::time::Instant::now();
        journal.task_changed(&task, false).await.unwrap();
        reporting += started.elapsed();
        // The writer has let go of it again.
        journal.task_changed(&task, true).await.unwrap();
        assert_eq!(manager.live_image_addresses(id), held);
    }
    let megabytes = task
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .map(|content| match content {
            arko_ai::Content::Image(arko_ai::Image::Base64 { data, .. }) => data.len(),
            _ => 0,
        })
        .sum::<usize>()
        >> 20;
    println!(
        "a safe point of a task with {megabytes} MB of images: {:?} to clone it, {:?} to report it",
        cloning / 5,
        reporting / 5,
    );

    // A step that adds an image keeps the others and holds the new one.
    task.conversation
        .add_user_message_with_image("Screenshot 10.", screenshot(10));
    journal.task_changed(&task, true).await.unwrap();
    let addresses = manager.live_image_addresses(id);
    assert_eq!(addresses.len(), 11);
    assert_eq!(addresses[..10], held[..]);

    // What the journal holds is the task, as a clone of it would be.
    let saved = manager.latest_live_task(id).unwrap();
    assert_eq!(
        serde_json::to_value(&saved).unwrap(),
        serde_json::to_value(&task).unwrap()
    );
    crash(workflow, vec![run]).await;
}
