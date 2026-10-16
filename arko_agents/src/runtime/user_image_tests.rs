//! A user's images: with the task's prompt in its first message, and with a
//! context update, each after its label.

use super::tool_less_tests::{chat_agent, push, run};
use super::*;
use crate::runtime::{ContextImage, TurnContext, TurnContextSource};

fn pixels() -> ai::Image {
    ai::Image::new_base64("cGl4ZWxz", "image/png")
}

fn images(message: &ai::Message) -> Vec<&ai::Image> {
    message
        .content
        .iter()
        .filter_map(ai::Content::as_image)
        .collect()
}

/// The task's images follow its prompt in its first message, and the
/// request is made with them.
#[tokio::test]
async fn the_first_message_holds_the_task_s_images_after_its_prompt() {
    let (mut agent, mut events, _commands) = chat_agent();
    agent.task.initial_task_images = vec![pixels(), pixels()];
    push(&mut agent, &["Two charts."]);
    let (task, _) = run(agent, &mut events).await;

    let first = &task.conversation.messages[0];
    assert!(message_text(first).starts_with("Hello."));
    assert!(matches!(&first.content[0], ai::Content::Text(_)));
    assert_eq!(images(first), [&pixels(), &pixels()]);
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
}

/// A source that gives one context update, at its first call.
struct Once(std::sync::Mutex<Option<TurnContext>>);

impl TurnContextSource for Once {
    fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
        None
    }

    fn context_update(&self, _task_id: uuid::Uuid) -> Option<TurnContext> {
        self.0.lock().unwrap().take()
    }
}

/// A context update's images follow its text, each after its label, in
/// the same message; one with images and no text is added too.
#[tokio::test]
async fn a_context_update_s_images_follow_its_text_each_after_its_label() {
    for content in ["A new note on your task.", ""] {
        let (mut agent, mut events, _commands) = chat_agent();
        agent.turn_context = Some(Arc::new(Once(std::sync::Mutex::new(Some(TurnContext {
            content: content.into(),
            ids: vec!["note:3".into()],
            images: vec![ContextImage::new("The image of note 3:", pixels())],
        })))));
        push(&mut agent, &["Seen."]);
        let (task, events) = run(agent, &mut events).await;

        let update = &task.conversation.messages[1];
        let ai::Content::Text(text) = &update.content[0] else {
            panic!("{update:#?}");
        };
        assert!(text.starts_with("# Workflow context update"), "{text}");
        assert!(text.contains(content));
        assert!(
            matches!(&update.content[1..], [ai::Content::Text(label), ai::Content::Image(image)]
                if label == "The image of note 3:" && *image == pixels()),
            "{update:#?}"
        );
        assert!(task.pending_context_images.is_empty());
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ContextInjected { ids, .. } if ids == &["note:3"]
        )));
    }
}

/// A continuation's images follow its text; so do an answer's.
#[test]
fn a_continuation_and_an_answer_hold_their_images_after_their_text() {
    let mut task = Task::new(
        super::tests::test_config(),
        "Hello.".into(),
        std::env::temp_dir(),
        None,
    );
    task.set_state(TaskState::Completed {
        summary: "Hi.".into(),
    });
    task.prepare_for_resumption_from_completed_with_images("Look at this.".into(), vec![pixels()])
        .unwrap();
    let last = task.conversation.messages.last().unwrap();
    assert!(message_text(last).starts_with("Look at this."));
    assert!(matches!(last.content.last(), Some(ai::Content::Image(_))));

    task.set_state(TaskState::AwaitingUserResponse {
        questions: "Which one?".into(),
    });
    task.prepare_for_resumption_with_user_response_and_images("This one.".into(), vec![pixels()])
        .unwrap();
    let last = task.conversation.messages.last().unwrap();
    assert!(message_text(last).contains("Response: This one."));
    assert_eq!(images(last), [&pixels()]);
}

/// A task saved without images loads; the images are saved with it.
#[test]
fn a_task_s_images_are_saved_with_it() {
    let mut task = Task::new(
        super::tests::test_config(),
        "Hello.".into(),
        std::env::temp_dir(),
        None,
    );
    let json = serde_json::to_value(&task).unwrap();
    assert!(json.get("initial_task_images").is_none());
    assert!(json.get("pending_context_images").is_none());
    task.initial_task_images = vec![pixels()];
    task.pending_context_injections = vec!["Later.".into()];
    task.pending_context_images = vec![ContextImage::new("Image 1:", pixels())];
    let loaded: Task = serde_json::from_value(serde_json::to_value(&task).unwrap()).unwrap();
    assert_eq!(loaded.initial_task_images, [pixels()]);
    assert_eq!(
        loaded.pending_context_images,
        [ContextImage::new("Image 1:", pixels())]
    );
}
