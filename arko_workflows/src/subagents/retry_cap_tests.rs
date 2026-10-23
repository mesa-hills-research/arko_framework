//! `SubagentsConfig::max_failed_requests`, the cap on how many model requests
//! of a subagent fail in a row.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arko_agents::LLMErrorPolicy;
use serde_json::json;

use super::tests::{
    Noting, bodies_of, builder_with, result_of, role, start, turn, until, wait, within,
};
use super::*;
use crate::policy::RecoveryWaits;
use crate::test_support::{Reply, ResponsesServer, Workspace, agent_config, tool_call_turn};
use crate::workflow::result::AgentRunStatus;

fn failure() -> Reply {
    Reply::Status {
        status: 503,
        retry_after: None,
        body: json!({"error": {"message": "synthetic 503", "type": "server_error"}}).to_string(),
    }
}

fn overloaded() -> Reply {
    Reply::Status {
        status: 529,
        retry_after: None,
        body: json!({"error": {"message": "synthetic overload", "type": "overloaded_error"}})
            .to_string(),
    }
}

/// A server that fails a worker's model request with a 503 unless `answers`
/// has an answer for it, as the number of the request among the worker's.
/// The lead starts one worker and waits for it twice.
async fn flaky(
    answers: Vec<(&'static str, usize, Reply)>,
    calls: Arc<AtomicUsize>,
) -> ResponsesServer {
    flaky_with(failure, answers, calls).await
}

/// [`flaky`], failing with `failure`.
async fn flaky_with(
    fail: fn() -> Reply,
    answers: Vec<(&'static str, usize, Reply)>,
    calls: Arc<AtomicUsize>,
) -> ResponsesServer {
    let answers = Arc::new(parking_lot::Mutex::new(answers));
    ResponsesServer::start_async(move |_, body| {
        let (role, turn) = (role(&body), turn(&body));
        let reply = if role == "lead" {
            let step = [start("Count the files."), wait(), wait()]
                .into_iter()
                .nth(turn);
            match step {
                Some((tool, arguments)) => {
                    Reply::Stream(tool_call_turn(&format!("lead_{turn}"), tool, arguments))
                }
                None => Reply::Stream(tool_call_turn(
                    "lead_done",
                    "complete_task",
                    json!({ "summary": "lead done" }),
                )),
            }
        } else {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            let mut answers = answers.lock();
            match answers
                .iter()
                .position(|(of, index, _)| *of == role && *index == call)
            {
                Some(found) => answers.remove(found).2,
                None => fail(),
            }
        };
        Box::pin(async move { Some(reply) })
    })
    .await
}

/// An agent that retries a failed request twice shortly, then pauses.
fn quick(name: &str) -> AgentConfig {
    let mut config = agent_config(name);
    config.llm_error_policy = LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 3 };
    config.llm_retry_config.initial_delay_ms = 50;
    config.llm_retry_config.max_delay_ms = 50;
    config.llm_retry_config.jitter = false;
    config
}

fn instant() -> RecoveryWaits {
    RecoveryWaits {
        initial: Duration::from_millis(1),
        max: Duration::from_millis(1),
        reset_margin: Duration::ZERO,
        max_jitter: Duration::ZERO,
    }
}

/// A worker whose requests keep failing fails the moment the cap is reached,
/// counting the agent's own retries, and a model response starts the count
/// again. Here the cap is 4 and a round is 3 requests. The first round ends
/// in a pause (3 failed), a response resets, the next round ends in a pause
/// (3 failed), and the first request after it is the fourth in a row.
#[tokio::test]
async fn a_worker_fails_after_its_failed_requests_in_a_row() {
    let calls = Arc::new(AtomicUsize::new(0));
    let progress = Reply::Stream(tool_call_turn(
        "r1_0",
        "message_agent",
        json!({ "to": "lead", "message": "Still counting." }),
    ));
    let server = flaky(vec![("r1", 3, progress)], calls.clone()).await;
    let workspace = Workspace::new();
    let workflow = builder_with(
        &server,
        &workspace,
        SubagentsConfig::default().max_failed_requests(4),
        quick("Worker"),
    )
    .recovery_waits(instant())
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate the count."))
        .await
        .unwrap();
    until(|| workflow.subagents()[0].state == SubagentState::Failed).await;

    // The cancel may land after a ninth request went out.
    let calls = calls.load(Ordering::SeqCst);
    assert!((8..=9).contains(&calls), "{calls}");
    let error = workflow.subagents()[0].error.clone().unwrap();
    assert!(
        error.contains("It stopped after 4 failed requests in a row."),
        "{error}"
    );
}

/// A worker that waits out overloaded errors never pauses, since the agent
/// retries those on its own. The cap fails it at the third failed request,
/// and its run ends failed, as at a pause, so its lead isn't offered to
/// continue it. The host's line ends its news.
#[tokio::test]
async fn a_worker_waiting_out_overloaded_errors_fails_at_the_cap() {
    let calls = Arc::new(AtomicUsize::new(0));
    let server = flaky_with(overloaded, vec![], calls.clone()).await;
    let workspace = Workspace::new();
    let mut worker = quick("Worker");
    worker.llm_retry_config.rate_limit_retry = Some(arko_agents::LLMRateLimitRetryConfig {
        initial_delay_ms: 20,
        max_delay_ms: 20,
        max_wait_ms: 15 * 60 * 1000,
    });
    let workflow = builder_with(
        &server,
        &workspace,
        SubagentsConfig::default()
            .max_failed_requests(3)
            .hook(Arc::new(Noting)),
        worker,
    )
    .recovery_waits(instant())
    .build()
    .unwrap();
    within(workflow.run_agent("lead", "Delegate the count."))
        .await
        .unwrap();
    until(|| workflow.subagents()[0].state == SubagentState::Failed).await;

    let error = workflow.subagents()[0].error.clone().unwrap();
    assert!(
        error.contains("It stopped after 3 failed requests in a row."),
        "{error}"
    );
    let run = workflow
        .agent_result(workflow.subagents()[0].run_id)
        .unwrap();
    assert_eq!(run.status, AgentRunStatus::Failed);
    let lead = bodies_of(&server, "lead");
    let told = result_of(&lead[2], "lead", 1);
    assert!(
        told.contains("stopped with an error:") && told.contains("Its run can't go on"),
        "{told}"
    );
    assert!(!told.contains("message_agent continues it"), "{told}");
    assert!(
        told.contains("start a new one if the work is still needed.\n\nr1 used 12 tokens (Failed, its result)."),
        "{told}"
    );
}

/// The host sets the cap while a worker, under none, keeps failing. The worker
/// is past it already, so it fails at its next failed request.
#[tokio::test]
async fn a_cap_the_host_sets_while_a_worker_runs_applies_to_it() {
    let calls = Arc::new(AtomicUsize::new(0));
    let server = flaky(vec![], calls.clone()).await;
    let workspace = Workspace::new();
    let workflow = builder_with(
        &server,
        &workspace,
        SubagentsConfig::default(),
        quick("Worker"),
    )
    .recovery_waits(instant())
    .build()
    .unwrap();
    let host = async {
        until(|| calls.load(Ordering::SeqCst) >= 7).await;
        assert_eq!(workflow.subagents()[0].state, SubagentState::Running);
        workflow.set_max_failed_requests(Some(2));
        until(|| workflow.subagents()[0].state == SubagentState::Failed).await;
    };
    let (lead, ()) = tokio::join!(
        within(workflow.run_agent("lead", "Delegate the count.")),
        host
    );
    lead.unwrap();

    let error = workflow.subagents()[0].error.clone().unwrap();
    let failed = error
        .split("It stopped after ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|count| count.parse::<u32>().ok());
    assert!(failed.is_some_and(|failed| failed >= 6), "{error}");
}

/// The cap is for workers. The run the host drives recovers as often as its
/// policy says.
#[tokio::test]
async fn the_cap_leaves_the_main_agent_alone() {
    let calls = Arc::new(AtomicUsize::new(0));
    let server = ResponsesServer::start_async({
        let calls = calls.clone();
        move |_, _| {
            let reply = if calls.fetch_add(1, Ordering::SeqCst) < 6 {
                failure()
            } else {
                Reply::Stream(tool_call_turn(
                    "lead_done",
                    "complete_task",
                    json!({ "summary": "lead done" }),
                ))
            };
            Box::pin(async move { Some(reply) })
        }
    })
    .await;
    let workspace = Workspace::new();
    let workflow = crate::Workflow::builder()
        .name("subagents-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", quick("Lead"))
        .agent("worker", agent_config("Worker"))
        .workflow_services(
            crate::services::WorkflowServicesConfig::default()
                .allow_caller("lead")
                .allow_target("worker"),
        )
        .subagents(SubagentsConfig::default().max_failed_requests(1))
        .recovery_waits(instant())
        .build()
        .unwrap();
    let result = within(workflow.run_agent("lead", "Say hello."))
        .await
        .unwrap();
    assert_eq!(result.summary, "lead done");
    assert_eq!(calls.load(Ordering::SeqCst), 7);
}
