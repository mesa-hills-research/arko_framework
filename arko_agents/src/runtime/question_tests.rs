//! `ask_user` with numbered questions, answers that come later, `wait`,
//! `close_questions`, and the open questions a handover restates.

use super::tool_call_tests::{agent, complete, config, native_call, text_turn, turn, workspace};
use super::*;
use crate::handover::HandoverConfig;
use crate::questions::OPEN_QUESTIONS_HEADING;
use futures::StreamExt;

fn push(agent: &mut AgentLoop, responses: Vec<ScriptedStreamResult>) {
    agent
        .scripted_responses
        .get_mut()
        .unwrap()
        .extend(responses);
}

/// A native response with `calls` and `input_tokens` of input.
fn with_input(calls: Vec<ai::ToolCall>, input_tokens: usize) -> ScriptedStreamResult {
    let mut message = ai::Message::new_assistant("");
    message.tool_calls = calls;
    let end = ai::StreamEvent::End(Box::new(ai::GenerationResponse {
        message,
        usage: ai::UsageInfo::new(input_tokens, 10),
        model_call_usage: ai::ModelCallUsage::new(
            "questions".into(),
            "test".into(),
            "test".into(),
            input_tokens,
            10,
            1000,
        ),
        stop_reason: ai::StopReason::ToolUse,
        refusal: None,
        openai_turn_state: None,
    }));
    Ok(futures::stream::iter(vec![Ok(end)]).boxed())
}

fn asks(events: &[AgentEvent]) -> Vec<(Vec<String>, bool)> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::UserQuestionsAsked { asked, wait, .. } => Some((
                asked.iter().map(|question| question.id.clone()).collect(),
                *wait,
            )),
            _ => None,
        })
        .collect()
}

async fn finish(
    run: tokio::task::JoinHandle<Result<Task, crate::error::ArkoAgentsError>>,
    events: &mut mpsc::Receiver<AgentEvent>,
) -> (Task, Vec<AgentEvent>) {
    let task = timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let mut seen = Vec::new();
    while let Ok(event) = events.try_recv() {
        seen.push(event);
    }
    (task, seen)
}

/// The conversation as JSON, tool results included.
fn conversation_text(task: &Task) -> String {
    serde_json::to_string(&task.conversation).unwrap()
}

/// With `async_questions` and `wait: false`, `ask_user` returns at once
/// with the questions' ids, which go on across calls, and the task never
/// waits. In both tool
/// calling modes, with the list and with one question as text.
#[tokio::test]
async fn an_ask_returns_at_once_with_ids_that_go_on() {
    for mode in [ToolCallingMode::Native, ToolCallingMode::OutputBased] {
        let workspace = workspace();
        let mut config = config(mode.clone(), &["ask_user"]);
        config.tool_config.async_questions = true;
        let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
        let responses = if mode == ToolCallingMode::Native {
            vec![
                turn(vec![native_call(
                    "ask",
                    "ask_user",
                    serde_json::json!({"questions": [
                        {"question": "Tabs or spaces?", "choices": ["Tabs", "Spaces"], "default": "Spaces"},
                        {"question": "Push now?"}
                    ], "wait": false}),
                )]),
                turn(vec![native_call(
                    "again",
                    "ask_user",
                    serde_json::json!({"questions": "Which branch?", "wait": false}),
                )]),
                complete(),
            ]
        } else {
            vec![
                text_turn(
                    "<ask_user>\n<questions>[{\"question\": \"Tabs or spaces?\", \"choices\": [\"Tabs\", \"Spaces\"]}, \"Push now?\"]</questions>\n<wait>false</wait>\n</ask_user>",
                ),
                text_turn(
                    "<ask_user>\n<questions>Which branch?</questions>\n<wait>false</wait>\n</ask_user>",
                ),
                text_turn("<complete_task>\n<summary>done</summary>\n</complete_task>"),
            ]
        };
        push(&mut agent, responses);
        let run = tokio::spawn(agent.run());
        let (task, seen) = finish(run, &mut events).await;
        assert!(
            matches!(task.current_state, TaskState::Completed { .. }),
            "{mode:?}"
        );
        assert_eq!(
            asks(&seen),
            [
                (vec!["q1".to_string(), "q2".to_string()], false),
                (vec!["q3".to_string()], false)
            ],
            "{mode:?}"
        );
        assert!(
            !seen.iter().any(|event| matches!(
                event,
                AgentEvent::TaskStateChanged(TaskState::AwaitingUserResponse { .. })
            )),
            "{mode:?}"
        );
        let text = conversation_text(&task);
        assert!(
            text.contains(
                "Asked q1 and q2. Their answers reach you as a note. Until then, do only work that no answer could change."
            ),
            "{mode:?}: {text}"
        );
        assert!(
            text.contains("Asked q3. Its answer reaches you as a note"),
            "{mode:?}: {text}"
        );
        assert_eq!(
            task.questions
                .open
                .iter()
                .map(|question| question.id.as_str())
                .collect::<Vec<_>>(),
            ["q1", "q2", "q3"]
        );
        assert_eq!(task.questions.open[0].question.choices, ["Tabs", "Spaces"]);
        // The log is saved with the task.
        let saved: Task = serde_json::from_value(serde_json::to_value(&task).unwrap()).unwrap();
        assert_eq!(saved.questions, task.questions);
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// A task saved and resumed keeps its questions: the next ask goes on
/// from their ids, and those still open stay open.
#[tokio::test]
async fn the_ids_go_on_after_a_resume() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.async_questions = true;
    let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    push(
        &mut agent,
        vec![
            turn(vec![native_call(
                "ask",
                "ask_user",
                serde_json::json!({"questions": ["A?", "B?"], "wait": false}),
            )]),
            complete(),
        ],
    );
    let run = tokio::spawn(agent.run());
    let (task, _) = finish(run, &mut events).await;
    let mut saved: Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
    assert_eq!(saved.questions.asked, 2);
    saved
        .prepare_for_resumption_from_completed("Go on.".into())
        .unwrap();
    let (mut resumed, mut events, _commands) = super::tool_call_tests::agent_for(
        saved,
        ToolRegistry::new(),
        Arc::new(ai::LLMClient::new()),
    );
    push(
        &mut resumed,
        vec![
            turn(vec![native_call(
                "again",
                "ask_user",
                serde_json::json!({"questions": ["C?"], "wait": false}),
            )]),
            complete(),
        ],
    );
    let run = tokio::spawn(resumed.run());
    let (task, seen) = finish(run, &mut events).await;
    assert_eq!(
        asks(&seen),
        [(vec!["q3".to_string()], false)],
        "{:?}\n{seen:#?}",
        task.current_state
    );
    assert_eq!(
        task.questions
            .open
            .iter()
            .map(|question| question.id.as_str())
            .collect::<Vec<_>>(),
        ["q1", "q2", "q3"]
    );
    std::fs::remove_dir_all(workspace).unwrap();
}

/// An ask holds the task until the answer, which reaches the agent as the
/// host wrote it and settles the questions: with `wait: true`, without
/// `wait` (it defaults to true), and always without `async_questions`.
#[tokio::test]
async fn wait_holds_the_task_until_the_answer() {
    let which = serde_json::json!([{"question": "Which file?"}]);
    for (async_questions, params) in [
        (true, serde_json::json!({"questions": which, "wait": true})),
        (true, serde_json::json!({"questions": which})),
        (false, serde_json::json!({"questions": which})),
    ] {
        let workspace = workspace();
        let mut config = config(ToolCallingMode::Native, &["ask_user"]);
        config.tool_config.async_questions = async_questions;
        let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
        push(
            &mut agent,
            vec![
                turn(vec![native_call("ask", "ask_user", params)]),
                complete(),
            ],
        );
        let run = tokio::spawn(agent.run());
        let mut seen = Vec::new();
        let questions =
            super::tool_call_tests::wait_for(&mut events, &mut seen, |event| match event {
                AgentEvent::UserQuestionsAsked {
                    questions, wait, ..
                } => Some((questions.clone(), *wait)),
                _ => None,
            })
            .await;
        assert_eq!(questions, ("q1 Which file?".to_string(), true));
        commands
            .send(UserCommand::SubmitUserResponse {
                response: "The user answered q1.\nq1 Which file? README.md".into(),
                ids: Vec::new(),
            })
            .await
            .unwrap();
        let (task, _) = finish(run, &mut events).await;
        assert!(matches!(task.current_state, TaskState::Completed { .. }));
        assert!(task.questions.open.is_empty() && task.questions.waiting.is_empty());
        let text = conversation_text(&task);
        assert!(text.contains("q1 Which file? README.md"), "{text}");
        // The answer of a host that answers later comes as it is.
        assert_eq!(
            text.contains("You asked the following questions"),
            !async_questions,
            "{text}"
        );
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// A report asked for alone of a task waiting for its answer: the asking
/// call keeps its one result, ahead of the request, and the task completes
/// with the report, its question still waiting in its log.
#[tokio::test]
async fn a_report_from_a_task_waiting_for_its_answer_keeps_each_result_once() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.async_questions = false;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    let which = serde_json::json!({"questions": [{"question": "Which file?"}]});
    push(
        &mut agent,
        vec![turn(vec![native_call("ask", "ask_user", which)])],
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    super::tool_call_tests::wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::UserQuestionsAsked { .. } => Some(()),
        _ => None,
    })
    .await;
    commands.send(UserCommand::SuspendTask).await.unwrap();
    let (mut task, _) = finish(run, &mut events).await;
    assert!(matches!(
        task.current_state,
        TaskState::AwaitingUserResponse { .. }
    ));

    task.request_report(None).unwrap();
    let (mut agent, mut events, _commands) = super::tool_call_tests::agent_for(
        task,
        ToolRegistry::new(),
        Arc::new(ai::LLMClient::new()),
    );
    push(
        &mut agent,
        vec![super::image_tests::response(ai::Message::new_assistant(
            "REPORT: asked which file.",
        ))],
    );
    let run = tokio::spawn(agent.run());
    let (task, _) = finish(run, &mut events).await;
    assert!(
        matches!(&task.current_state, TaskState::Completed { summary } if summary == "REPORT: asked which file."),
        "{:?}",
        task.current_state
    );
    let results = task
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_tool_result)
        .filter(|result| result.call_id.as_deref() == Some("ask"))
        .count();
    assert_eq!(results, 1, "{}", conversation_text(&task));
    assert!(!task.questions.waiting.is_empty());
    std::fs::remove_dir_all(workspace).unwrap();
}

/// `close_questions` closes the questions the task asked and reports them,
/// and says which ids it never asked.
#[tokio::test]
async fn close_questions_settles_what_the_task_asked() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user", "close_questions"]);
    config.tool_config.async_questions = true;
    let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    push(
        &mut agent,
        vec![
            turn(vec![native_call(
                "ask",
                "ask_user",
                serde_json::json!({"questions": ["A?", "B?"], "wait": false}),
            )]),
            turn(vec![native_call(
                "close",
                "close_questions",
                serde_json::json!({"ids": ["q1", "q9"], "why": "Found it in the README"}),
            )]),
            complete(),
        ],
    );
    let run = tokio::spawn(agent.run());
    let (task, seen) = finish(run, &mut events).await;
    assert!(seen.iter().any(|event| matches!(
        event,
        AgentEvent::QuestionsClosed { ids, why: Some(why) }
            if ids == &["q1"] && why == "Found it in the README"
    )));
    assert_eq!(task.questions.open.len(), 1);
    assert_eq!(task.questions.open[0].id, "q2");
    let text = conversation_text(&task);
    assert!(text.contains("Closed q1. You never asked q9."), "{text}");
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A hook that says only `q2` is still open (the host answered `q1`).
struct HostAnswered;

impl HandoverHook for HostAnswered {
    fn open_questions(&self, task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        Some(
            task.questions
                .open
                .iter()
                .filter(|question| question.id == "q2")
                .cloned()
                .collect(),
        )
    }
}

/// The host has `q2` open, with its text, whatever the task's log says.
struct HostKeepsQ2;

impl HandoverHook for HostKeepsQ2 {
    fn open_questions(&self, _task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        Some(vec![crate::questions::AskedQuestion {
            id: "q2".into(),
            question: crate::questions::Question {
                question: "Push now?".into(),
                ..Default::default()
            },
        }])
    }
}

/// The host answers only `q1` of a batch the agent waited for: the answer
/// settles `q1` alone, so `q2` stays open in the task's log, a checkpoint
/// keeps it, and the handover after it restates it.
#[tokio::test]
async fn a_partial_answer_to_a_waited_batch_leaves_the_rest_open() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.async_questions = true;
    config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    agent.handover_hook = Some(Arc::new(HostKeepsQ2));
    push(
        &mut agent,
        vec![
            with_input(
                vec![native_call(
                    "ask",
                    "ask_user",
                    serde_json::json!({"questions": ["Tabs or spaces?", "Push now?"], "wait": true}),
                )],
                500,
            ),
            // Over the threshold: the next request asks for the report.
            with_input(
                vec![native_call(
                    "list",
                    "ask_user",
                    serde_json::json!({"questions": ["Which branch?"], "wait": false}),
                )],
                1_500,
            ),
            with_input(
                vec![native_call(
                    "report",
                    HANDOVER_REPORT_TOOL_NAME,
                    serde_json::json!({"report": "Used spaces."}),
                )],
                1_600,
            ),
            complete(),
        ],
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    super::tool_call_tests::wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::UserQuestionsAsked { wait: true, .. } => Some(()),
        _ => None,
    })
    .await;
    commands
        .send(UserCommand::SubmitUserResponse {
            response: "The user answered q1.\nq1 Tabs or spaces? Spaces\nStill open: q2.".into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, _) = finish(run, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    let saved: Task = serde_json::from_value(serde_json::to_value(&task).unwrap()).unwrap();
    assert_eq!(
        saved
            .questions
            .open
            .iter()
            .map(|question| question.id.as_str())
            .collect::<Vec<_>>(),
        ["q2", "q3"],
        "the log settled q1 alone"
    );
    assert!(saved.questions.waiting.is_empty());
    let first = message_text(&task.conversation.messages[0]);
    assert!(first.contains("q2 Push now?"), "{first}");
    assert!(!first.contains("q1 Tabs or spaces?"), "{first}");
    std::fs::remove_dir_all(workspace).unwrap();
}

/// The fresh conversation of a handover restates the questions still open,
/// as the task's log has them, or as the host's hook says.
#[tokio::test]
async fn a_handover_restates_the_open_questions() {
    for hook in [false, true] {
        let workspace = workspace();
        let mut config = config(ToolCallingMode::Native, &["ask_user"]);
        config.tool_config.async_questions = true;
        config.handover = Some(HandoverConfig::at_input_tokens(1_000));
        let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
        if hook {
            agent.handover_hook = Some(Arc::new(HostAnswered));
        }
        push(
            &mut agent,
            vec![
                with_input(
                    vec![native_call(
                        "ask",
                        "ask_user",
                        serde_json::json!({"questions": [
                            {"question": "Tabs or spaces?"},
                            {"question": "Push now?", "default": "not yet"}
                        ], "wait": false}),
                    )],
                    500,
                ),
                // Over the threshold: the next request asks for the report.
                with_input(
                    vec![native_call(
                        "again",
                        "ask_user",
                        serde_json::json!({"questions": ["Which branch?"], "wait": false}),
                    )],
                    1_500,
                ),
                with_input(
                    vec![native_call(
                        "report",
                        HANDOVER_REPORT_TOOL_NAME,
                        serde_json::json!({"report": "Asked three questions."}),
                    )],
                    1_600,
                ),
                complete(),
            ],
        );
        let run = tokio::spawn(agent.run());
        let (task, _) = finish(run, &mut events).await;
        assert_eq!(task.handovers.len(), 1, "hook {hook}");
        let first = message_text(&task.conversation.messages[0]);
        assert!(first.contains(OPEN_QUESTIONS_HEADING), "{first}");
        assert!(
            first.contains("q2 Push now?\n   Default: not yet"),
            "{first}"
        );
        assert_eq!(first.contains("q1 Tabs or spaces?"), !hook, "{first}");
        assert_eq!(first.contains("q3 Which branch?"), !hook, "{first}");
        std::fs::remove_dir_all(workspace).unwrap();
    }
}

/// A host's record of the questions: it numbers them from 41 and keeps
/// those asked open.
#[derive(Default)]
struct HostBook {
    asked: std::sync::Mutex<Vec<crate::questions::AskedQuestion>>,
}

impl HandoverHook for HostBook {
    fn open_questions(&self, _task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        Some(self.asked.lock().unwrap().clone())
    }

    fn question_numbers(&self, _task: &Task, _count: u32) -> Option<u32> {
        Some(41 + self.asked.lock().unwrap().len() as u32)
    }

    fn questions_asked(
        &self,
        _task: &Task,
        asked: &[crate::questions::AskedQuestion],
        _wait: bool,
    ) {
        self.asked.lock().unwrap().extend(asked.iter().cloned());
    }
}

/// The host numbers the questions and hears of them as they are recorded,
/// so a handover right after an ask restates it, once.
#[tokio::test]
async fn a_handover_right_after_an_ask_restates_it_once() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.async_questions = true;
    config.handover = Some(HandoverConfig::at_input_tokens(1_000));
    let (mut agent, mut events, _commands) = agent(config, ToolRegistry::new(), &workspace);
    agent.handover_hook = Some(Arc::new(HostBook::default()));
    push(
        &mut agent,
        vec![
            with_input(
                vec![native_call(
                    "first",
                    "ask_user",
                    serde_json::json!({"questions": ["Which branch?"], "wait": false}),
                )],
                500,
            ),
            // Over the threshold: the next request asks for the report.
            with_input(
                vec![native_call(
                    "ask",
                    "ask_user",
                    serde_json::json!({"questions": ["Tabs or spaces?"], "wait": false}),
                )],
                1_500,
            ),
            with_input(
                vec![native_call(
                    "report",
                    HANDOVER_REPORT_TOOL_NAME,
                    serde_json::json!({"report": "Asked a question."}),
                )],
                1_600,
            ),
            complete(),
        ],
    );
    let run = tokio::spawn(agent.run());
    let (task, seen) = finish(run, &mut events).await;
    assert_eq!(task.handovers.len(), 1);
    assert_eq!(
        asks(&seen),
        [
            (vec!["q41".to_string()], false),
            (vec!["q42".to_string()], false)
        ]
    );
    let first = message_text(&task.conversation.messages[0]);
    assert_eq!(first.matches("q42 Tabs or spaces?").count(), 1, "{first}");
    assert_eq!(first.matches("q41 Which branch?").count(), 1, "{first}");
    std::fs::remove_dir_all(workspace).unwrap();
}

/// A host that answers the agent's questions as it hears of them, before
/// the task waits for the answer. Their note goes on the command channel at
/// once, and they are no longer open.
struct AnswersAtOnce {
    commands: mpsc::Sender<UserCommand>,
    answered: std::sync::Mutex<Vec<String>>,
}

impl HandoverHook for AnswersAtOnce {
    fn open_questions(&self, task: &Task) -> Option<Vec<crate::questions::AskedQuestion>> {
        let answered = self.answered.lock().unwrap();
        Some(
            task.questions
                .open
                .iter()
                .filter(|question| !answered.contains(&question.id))
                .cloned()
                .collect(),
        )
    }

    fn questions_asked(
        &self,
        _task: &Task,
        asked: &[crate::questions::AskedQuestion],
        _wait: bool,
    ) {
        let ids = asked
            .iter()
            .map(|question| question.id.clone())
            .collect::<Vec<_>>();
        self.commands
            .try_send(UserCommand::InjectContext {
                content: format!("The user answered {}.", crate::questions::id_list(&ids)),
            })
            .unwrap();
        self.answered.lock().unwrap().extend(ids);
    }
}

/// An answer that lands before the task waits for it isn't read early or
/// lost. The host's reply to the wait comes first and the answer's note
/// after it, both in the request that follows, and the question is
/// settled.
#[tokio::test]
async fn an_answer_that_lands_before_the_wait_follows_the_reply() {
    const REPLIED: &str = "The user replied. What they said is in the note that follows.";
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.async_questions = true;
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    agent.handover_hook = Some(Arc::new(AnswersAtOnce {
        commands: commands.clone(),
        answered: Default::default(),
    }));
    push(
        &mut agent,
        vec![
            turn(vec![native_call(
                "ask",
                "ask_user",
                serde_json::json!({"questions": ["Tabs or spaces?"]}),
            )]),
            complete(),
        ],
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    super::tool_call_tests::wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::UserQuestionsAsked { wait: true, .. } => Some(()),
        _ => None,
    })
    .await;
    commands
        .send(UserCommand::SubmitUserResponse {
            response: REPLIED.into(),
            ids: Vec::new(),
        })
        .await
        .unwrap();
    let (task, _) = finish(run, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(task.questions.open.is_empty() && task.questions.waiting.is_empty());
    assert!(task.pending_context_injections.is_empty());
    let texts = task
        .conversation
        .messages
        .iter()
        .map(message_text)
        .collect::<Vec<_>>();
    let joined = texts.join("\n---\n");
    let note = "The user answered q1.";
    assert_eq!(joined.matches(note).count(), 1, "{joined}");
    assert!(
        joined.find(REPLIED).unwrap() < joined.find(note).unwrap(),
        "{joined}"
    );
    let noted = texts.iter().position(|text| text.contains(note)).unwrap();
    let done = task
        .conversation
        .messages
        .iter()
        .position(|message| {
            message
                .tool_calls
                .iter()
                .any(|call| call.name == "complete_task")
        })
        .unwrap();
    assert!(noted < done, "the note went with the request: {texts:#?}");
    std::fs::remove_dir_all(workspace).unwrap();
}

/// Keeps the task as the handover found it.
#[derive(Default)]
struct HandedOver(std::sync::Mutex<Option<Task>>);

impl crate::handover::HandoverHook for HandedOver {
    fn on_handover(&self, task: &Task, _report: &str) -> crate::handover::HandoverSeed {
        *self.0.lock().unwrap() = Some(task.clone());
        crate::handover::HandoverSeed::for_task(task)
    }
}

/// An answer that holds what the host delivers is in the step a request
/// over the model's input limit leaves out. The answer stays with its
/// record, and the asking call goes with its result, so every result left
/// has its call. (`Task::leave_out_step`'s test has the answer beside the
/// results.)
#[tokio::test]
async fn a_step_left_out_keeps_an_answer_without_the_result_beside_it() {
    let workspace = workspace();
    let mut config = config(ToolCallingMode::Native, &["ask_user"]);
    config.tool_config.async_questions = false;
    config.handover = Some(HandoverConfig::at_input_tokens(900_000));
    let (mut agent, mut events, commands) = agent(config, ToolRegistry::new(), &workspace);
    let handed_over = Arc::new(HandedOver::default());
    agent.handover_hook = Some(handed_over.clone());
    let which = serde_json::json!({"questions": [{"question": "Which file?"}]});
    push(
        &mut agent,
        vec![
            with_input(vec![native_call("ask", "ask_user", which)], 100),
            Err(ai::Error::ProviderError {
                message: "Your input exceeds the context window of this model.".into(),
                status: Some(400),
                code: Some("context_length_exceeded".into()),
                retry_after: None,
            }),
            with_input(
                vec![native_call(
                    "report",
                    HANDOVER_REPORT_TOOL_NAME,
                    serde_json::json!({"report": "Asked which file."}),
                )],
                120,
            ),
            complete(),
        ],
    );
    let run = tokio::spawn(agent.run());
    let mut seen = Vec::new();
    super::tool_call_tests::wait_for(&mut events, &mut seen, |event| match event {
        AgentEvent::UserQuestionsAsked { .. } => Some(()),
        _ => None,
    })
    .await;
    commands
        .send(UserCommand::SubmitUserResponse {
            response: "The user answered q1.\nq1 Which file? README.md".into(),
            ids: vec!["message:answer".into()],
        })
        .await
        .unwrap();
    let (task, _) = finish(run, &mut events).await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert_eq!(task.handovers.len(), 1);
    // The report covers the answer, which the fresh conversation records.
    assert_eq!(
        task.holding("message:answer"),
        crate::task::Holding::Recorded
    );
    let old = handed_over.0.lock().unwrap().clone().unwrap();
    let calls = old
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.tool_calls)
        .filter_map(|call| call.id.clone())
        .collect::<Vec<_>>();
    assert!(!calls.contains(&"ask".to_string()), "{calls:?}");
    for result in old
        .conversation
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_tool_result)
    {
        assert!(
            result
                .call_id
                .as_ref()
                .is_some_and(|call| calls.contains(call)),
            "a result without its call: {result:?}"
        );
    }
    let answer = old
        .conversation
        .messages
        .iter()
        .position(|message| message_text(message).contains("q1 Which file? README.md"))
        .expect("the answer stays");
    assert_eq!(old.conversation.messages[answer].role, ai::Role::User);
    assert_eq!(
        old.holding("message:answer"),
        crate::task::Holding::Recorded
    );
    assert!(
        old.context_records
            .iter()
            .any(|record| record.message == answer)
    );
    std::fs::remove_dir_all(workspace).unwrap();
}
