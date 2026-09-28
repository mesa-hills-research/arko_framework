//! Completion verification through real runs: re-review with prior
//! findings, and failed reviews.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::test_support::{ResponsesServer, Workspace, agent_config, complete_turn};
use crate::verification::CompletionVerifierConfig;
use crate::workflow::result::AgentRunStatus;

const VERIFIER: &str = "You are a completion verifier";

fn builder(
    server: &ResponsesServer,
    workspace: &Workspace,
    verifier: CompletionVerifierConfig,
) -> WorkflowBuilder {
    Workflow::builder()
        .name("verification-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("coder", agent_config("Friendly Coder"))
        .agent("reviewer", agent_config("Friendly Reviewer"))
        .completion_verifier(verifier)
}

/// A server where the coder always completes and the verifier answers
/// its `n`th request (from 0) with `verdict(n)`.
async fn server(verdict: fn(usize) -> &'static str) -> ResponsesServer {
    let reviews = Arc::new(AtomicUsize::new(0));
    ResponsesServer::start(move |index, body| {
        Some(if body.contains(VERIFIER) {
            let review = reviews.fetch_add(1, Ordering::SeqCst);
            complete_turn(&format!("v{index}"), verdict(review))
        } else {
            complete_turn(&format!("c{index}"), "done")
        })
    })
    .await
}

fn verifier_bodies(server: &ResponsesServer) -> Vec<String> {
    server
        .bodies()
        .into_iter()
        .filter(|body| body.contains(VERIFIER))
        .collect()
}

/// A new review of a task after a rejection is given the prior verdict and
/// findings by default; with `include_prior_findings(false)` it is a first
/// review again.
#[tokio::test]
async fn a_re_review_is_given_the_prior_findings_unless_opted_out() {
    for include in [true, false] {
        let workspace = Workspace::new();
        let server = server(|review| {
            if review == 0 {
                "REJECTED_MINOR\nFix the parser."
            } else {
                "ACCEPTED"
            }
        })
        .await;
        let workflow = builder(
            &server,
            &workspace,
            CompletionVerifierConfig::correctness("check", "reviewer")
                .target("coder")
                .include_prior_findings(include),
        )
        .build()
        .unwrap();
        let result = workflow.run_agent("coder", "Write it.").await.unwrap();
        assert_eq!(result.status, AgentRunStatus::Completed);
        let reviews = verifier_bodies(&server);
        assert_eq!(reviews.len(), 2);
        assert!(reviews[0].contains("This is the first review pass"));
        let second = &reviews[1];
        assert_eq!(second.contains("This is a re-review"), include, "{second}");
        assert_eq!(second.contains("Fix the parser."), include, "{second}");
        assert_eq!(second.contains("This is the first review pass"), !include);
    }
}

/// A review that fails (here, a verdict that never follows the contract)
/// doesn't fail the task: the completion is rejected so the agent calls
/// `complete_task` again, up to `max_review_retries`; then the task is
/// paused, neither accepted nor rejected, and the call returns
/// `VerifierFailed`. The verdict is asked for again `max_format_retries`
/// times per review.
#[tokio::test]
async fn a_failed_review_is_retried_then_reported_as_verifier_failed() {
    for (format_retries, review_retries) in [(1, 1), (0, 0)] {
        let workspace = Workspace::new();
        let server = server(|_| "Looks good to me.").await;
        let hooks_failed = Arc::new(AtomicUsize::new(0));
        let workflow = builder(
            &server,
            &workspace,
            CompletionVerifierConfig::correctness("check", "reviewer")
                .target("coder")
                .format_retries(format_retries)
                .review_retries(review_retries),
        )
        .hook(crate::hooks::HookPoint::PostAgentFailure, {
            let hooks_failed = hooks_failed.clone();
            Arc::new(move |hook| {
                if hook.agent_name.as_deref() == Some("coder") {
                    hooks_failed.fetch_add(1, Ordering::SeqCst);
                }
                crate::hooks::HookResult::Continue
            })
        })
        // A pause is not a failure: neither hook fires.
        .hook(crate::hooks::HookPoint::OnWorkflowFailed, {
            let hooks_failed = hooks_failed.clone();
            Arc::new(move |_| {
                hooks_failed.fetch_add(1, Ordering::SeqCst);
                crate::hooks::HookResult::Continue
            })
        })
        .build()
        .unwrap();
        let error = workflow.run_agent("coder", "Write it.").await.unwrap_err();
        let WorkflowError::VerifierFailed {
            verifier,
            agent,
            task_id,
            reason,
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!((verifier.as_str(), agent.as_str()), ("check", "coder"));
        assert!(reason.contains("unusable"), "{reason}");
        let coder = workflow.agent_result(task_id).unwrap();
        assert_eq!(coder.status, AgentRunStatus::Paused);
        assert!(matches!(
            coder.task.current_state,
            arko_agents::TaskState::AwaitingErrorRecoveryApproval {
                cause: arko_agents::ErrorRecoveryCause::Middleware,
                ..
            }
        ));
        assert_eq!(hooks_failed.load(Ordering::SeqCst), 0, "not failed");
        let reviews = (review_retries as usize + 1) * (format_retries as usize + 1);
        assert_eq!(verifier_bodies(&server).len(), reviews);
        assert_eq!(
            server.requests() - reviews,
            review_retries as usize + 1,
            "the coder's completions"
        );
        if review_retries > 0 {
            let retried = server
                .bodies()
                .into_iter()
                .filter(|body| !body.contains(VERIFIER))
                .nth(1)
                .unwrap();
            assert!(
                retried.contains("could not review this completion"),
                "{retried}"
            );
        }
    }
}

/// A task whose completion a verifier could not review is left paused,
/// not cancelled: `continue_agent` gives it the prompt as its next message,
/// and its next completion is reviewed again (with the full count of
/// retries). (The task was once cancelled, and could not be continued.)
#[tokio::test]
async fn a_task_a_verifier_could_not_review_is_continued_with_continue_agent() {
    let workspace = Workspace::new();
    // Unusable until the third review.
    let server = server(|review| {
        if review < 2 {
            "Looks good to me."
        } else {
            "ACCEPTED"
        }
    })
    .await;
    let workflow = builder(
        &server,
        &workspace,
        CompletionVerifierConfig::correctness("check", "reviewer")
            .target("coder")
            .format_retries(0)
            .review_retries(0),
    )
    .build()
    .unwrap();
    let error = workflow.run_agent("coder", "Write it.").await.unwrap_err();
    let WorkflowError::VerifierFailed { task_id, .. } = error else {
        panic!("{error:?}");
    };
    let paused = workflow.agent_result(task_id).unwrap();
    assert_eq!(paused.status, AgentRunStatus::Paused);

    // Continued, it completes again; the review fails again (the count
    // started over), and the task is paused again.
    let error = workflow
        .continue_agent(&paused, "The verifier is back; complete again.")
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkflowError::VerifierFailed { task_id: id, .. } if id == task_id),
        "{error:?}"
    );
    let coder_bodies = || {
        server
            .bodies()
            .into_iter()
            .filter(|body| !body.contains(VERIFIER))
            .collect::<Vec<_>>()
    };
    let continued = coder_bodies().last().unwrap().clone();
    assert!(
        continued.contains("Nothing was decided about your work; the task is paused"),
        "{continued}"
    );
    assert!(
        continued.contains("The verifier is back; complete again."),
        "{continued}"
    );

    let paused = workflow.agent_result(task_id).unwrap();
    let result = workflow
        .continue_agent(&paused, "Complete once more.")
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    assert_eq!(result.agent_run_id, task_id);
    assert_eq!(verifier_bodies(&server).len(), 3);
    assert_eq!(coder_bodies().len(), 3);
}

/// In a parallel run, a task whose completion its verifier could not
/// review is a pause, not a failure: its paused result comes back with the
/// others, and `fail_fast` doesn't stop them.
#[tokio::test]
async fn a_verifier_failure_in_a_parallel_run_is_a_pause() {
    let workspace = Workspace::new();
    let server = server(|_| "Looks good to me.").await;
    let workflow = builder(
        &server,
        &workspace,
        CompletionVerifierConfig::correctness("check", "reviewer")
            .target("coder")
            .format_retries(0)
            .review_retries(0),
    )
    .agent(
        "writer",
        crate::test_support::agent_config("Friendly Writer"),
    )
    .fail_fast(true)
    .build()
    .unwrap();
    let results = workflow
        .run_agents_parallel(vec![
            ("coder", "Write it.".to_string()),
            ("writer", "Write the docs.".to_string()),
        ])
        .await
        .unwrap();
    assert_eq!(results[0].agent_name, "coder");
    assert_eq!(results[0].status, AgentRunStatus::Paused);
    assert_eq!(results[1].status, AgentRunStatus::Completed);
}
