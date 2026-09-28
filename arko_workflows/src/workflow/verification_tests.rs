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
/// stopped and the call returns `VerifierFailed`. The verdict is asked for
/// again `max_format_retries` times per review.
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
        assert_eq!(coder.status, AgentRunStatus::Cancelled);
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
