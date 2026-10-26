//! Questions an agent asks without waiting: the user input handler's answer
//! reaches the agent in its mailbox, and a handover asks the handler which
//! questions are still open.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::test_support::{
    ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
};
use crate::user_input::{UserInputHandler, UserInputRequest};
use crate::workflow::result::AgentRunStatus;

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .expect("in time")
}

/// Answers every question "Spaces.", and says only `q2` is still open.
struct Spaces;

#[async_trait]
impl UserInputHandler for Spaces {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String> {
        assert!(!request.wait);
        Ok(format!("{} Spaces.", request.asked[0].id))
    }

    fn open_questions(
        &self,
        _agent_run_id: uuid::Uuid,
    ) -> Option<Vec<arko_agents::questions::AskedQuestion>> {
        Some(vec![arko_agents::questions::AskedQuestion {
            id: "q2".into(),
            question: arko_agents::questions::Question {
                question: "B?".into(),
                ..Default::default()
            },
        }])
    }
}

fn asking_agent() -> arko_agents::AgentConfig {
    let mut config = agent_config("Asking Lead");
    config
        .tool_config
        .available_tool_names
        .push("ask_user".into());
    config.tool_config.async_questions = true;
    config
}

/// A server whose agent asks, completes "early" before the answer came, or
/// "answered" once a request shows `answer`.
async fn asking_server(answer: &'static str) -> ResponsesServer {
    ResponsesServer::start(move |_, body| {
        Some(if body.contains(answer) {
            complete_turn("done", "answered")
        } else if body.contains("Asked q1.") {
            complete_turn("early", "early")
        } else {
            tool_call_turn(
                "ask",
                "ask_user",
                json!({"questions": [{"question": "Tabs or spaces?", "choices": ["Tabs", "Spaces"]}]}),
            )
        })
    })
    .await
}

/// Runs the asking agent until it has read `answer`: at once, or after the
/// answer reached its mailbox and woke it (its run had ended).
async fn run_until_answered(workflow: &Workflow, answer: &str) {
    let result = within(workflow.run_agent("lead", "Format the code."))
        .await
        .unwrap();
    assert_eq!(result.status, AgentRunStatus::Completed);
    if result.summary == "answered" {
        return;
    }
    within(async {
        while !workflow
            .pending_news(result.agent_run_id)
            .iter()
            .any(|item| item.text().contains(answer))
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    let continued = within(workflow.continue_agent(&result, "Go on."))
        .await
        .unwrap();
    assert_eq!(continued.summary, "answered");
}

/// Without `wait`, the agent goes on, the handler's answer (its `respond`,
/// by default) reaches it as mail, and the run never waits for input.
#[tokio::test]
async fn an_answer_to_a_question_asked_without_waiting_comes_as_mail() {
    let workspace = Workspace::new();
    let server = asking_server("q1 Spaces.").await;
    let workflow = Workflow::builder()
        .name("async-question-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", asking_agent())
        .user_input_handler(Arc::new(Spaces))
        .build()
        .unwrap();
    run_until_answered(&workflow, "q1 Spaces.").await;
    let bodies = server.bodies();
    assert!(
        bodies
            .iter()
            .any(|body| body.contains("About your question q1:")),
        "{bodies:#?}"
    );
}

/// Without a user input handler, the agent is told no one can answer.
#[tokio::test]
async fn without_a_handler_the_agent_hears_no_one_can_answer() {
    let workspace = Workspace::new();
    let server = asking_server("No one can answer questions").await;
    let workflow = Workflow::builder()
        .name("async-question-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", asking_agent())
        .build()
        .unwrap();
    run_until_answered(&workflow, "No one can answer questions").await;
}

/// A handover restates the questions the handler says are open, or, when
/// it doesn't say, those the task's log holds.
#[tokio::test]
async fn a_handover_restates_the_questions_the_handler_has_open() {
    use arko_agents::handover::HandoverHook;
    for handler in [true, false] {
        let workspace = Workspace::new();
        let server = asking_server("never").await;
        let mut builder = Workflow::builder()
            .name("async-question-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("lead", asking_agent());
        if handler {
            builder = builder.user_input_handler(Arc::new(Spaces));
        }
        let workflow = builder.build().unwrap();
        let hook = crate::workflow::engine::WorkflowHandover {
            context: workflow.context().clone(),
            agent_name: "lead".to_string(),
            prompt_message_ids: Vec::new(),
            review: false,
        };
        let mut task = arko_agents::Task::new(
            asking_agent(),
            "Format the code.".into(),
            workspace.0.clone(),
            None,
        );
        task.questions.ask(
            ["A?", "B?", "C?"]
                .into_iter()
                .map(|question| arko_agents::questions::Question {
                    question: question.to_string(),
                    ..Default::default()
                })
                .collect(),
        );
        let open = hook
            .open_questions(&task)
            .unwrap_or_else(|| task.questions.open.clone())
            .into_iter()
            .map(|question| question.id)
            .collect::<Vec<_>>();
        if handler {
            assert_eq!(open, ["q2"]);
        } else {
            assert_eq!(open, ["q1", "q2", "q3"]);
        }
    }
}

/// Holds every answer until `release` sends, then answers "q1 Spaces.".
struct Held {
    release: tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

#[async_trait]
impl UserInputHandler for Held {
    async fn respond(&self, _request: UserInputRequest) -> Result<String, String> {
        let release = self.release.lock().await.take();
        if let Some(release) = release {
            let _ = release.await;
        }
        Ok("q1 Spaces.".to_string())
    }
}

/// An answer that comes after the asking run ended wakes it: the host
/// hears `NewsWaiting` about the run itself, and the run it continues
/// reads the answer.
#[tokio::test]
async fn a_late_answer_tells_the_host_its_run_has_news() {
    let workspace = Workspace::new();
    let server = asking_server("q1 Spaces.").await;
    let (release, held) = tokio::sync::oneshot::channel();
    let workflow = Workflow::builder()
        .name("async-question-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", asking_agent())
        .user_input_handler(Arc::new(Held {
            release: tokio::sync::Mutex::new(Some(held)),
        }))
        .build()
        .unwrap();
    let mut news = workflow.subscribe_subagents();
    let result = within(workflow.run_agent("lead", "Format the code."))
        .await
        .unwrap();
    assert_eq!(result.summary, "early");
    release.send(()).unwrap();
    let event = within(async {
        loop {
            let event = news.recv().await.unwrap();
            if matches!(
                event.kind,
                crate::subagents::SubagentEventKind::NewsWaiting { .. }
            ) {
                return event;
            }
        }
    })
    .await;
    assert_eq!(event.run_id, result.agent_run_id);
    assert_eq!(event.caller_run_id, result.agent_run_id);
    assert_eq!(event.agent_name, "lead");
    let continued = within(workflow.continue_agent(&result, "Go on."))
        .await
        .unwrap();
    assert_eq!(continued.summary, "answered");
}

/// Says the host has seen questions up to `q5` of the lead's runs.
struct SeenFive;

#[async_trait]
impl UserInputHandler for SeenFive {
    async fn respond(&self, _request: UserInputRequest) -> Result<String, String> {
        Ok(String::new())
    }

    async fn asked(&self, _request: UserInputRequest) -> Result<Option<String>, String> {
        Ok(None)
    }

    fn last_question_number(&self, agent_name: &str) -> u32 {
        if agent_name == "lead" { 5 } else { 0 }
    }
}

/// A run numbers its questions after the last one the host has seen, so
/// ids stay unique across the host's runs.
#[tokio::test]
async fn questions_are_numbered_after_those_the_host_has_seen() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(move |_, body| {
        Some(if body.contains("Asked q6.") {
            complete_turn("done", "asked")
        } else {
            tool_call_turn("ask", "ask_user", json!({"questions": ["Tabs or spaces?"]}))
        })
    })
    .await;
    let workflow = Workflow::builder()
        .name("async-question-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("lead", asking_agent())
        .user_input_handler(Arc::new(SeenFive))
        .build()
        .unwrap();
    let result = within(workflow.run_agent("lead", "Format the code."))
        .await
        .unwrap();
    assert_eq!(result.summary, "asked");
    let task = result.task().unwrap();
    assert_eq!(task.questions.open[0].id, "q6");
}
