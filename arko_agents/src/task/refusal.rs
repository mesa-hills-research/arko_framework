//! Refused inputs and how the host changes them. See [`Task::resolve_refusal`].

use super::Task;
use crate::error::ArkoAgentsError;
use arko_ai::{Content, Message, Role};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The input a refused turn answered, with every user message after the last
/// assistant turn, in order. Besides the host's own input (the task prompt,
/// a user response, a context update, tool results) the runtime adds some
/// (a file-tree update, an error or recovery note, a request to continue).
/// Those are marked [`runtime`](RefusedMessage::runtime) and can't be
/// changed. Reported with `AgentEvent::RefusedInput` and kept in
/// [`Task::refused_input`] until the refusal is resolved, recovery is
/// approved, or a later turn completes.
///
/// Approving recovery without a change (`ApproveErrorRecovery`) accepts the
/// input as it is. The refused turn stays in history followed by the
/// runtime's recovery note, so if the model refuses again, the reported
/// input is only that note, which can't be changed. Change the input with
/// `ResolveRefusal` instead of approving when it should be changed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RefusedInput {
    #[serde(default)]
    pub messages: Vec<RefusedMessage>,
}

/// One message of a [`RefusedInput`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RefusedMessage {
    /// Index in `Task::conversation.messages`.
    pub message_index: usize,
    /// Added by the runtime rather than taken from the host or a tool.
    pub runtime: bool,
    /// The message holds tool results with a note the runtime added beside
    /// them (a recovery note). Its text can't be rephrased, while its tool
    /// results and images can still be changed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub runtime_text: bool,
    /// Native tool results in the message, by provider call id. (In
    /// output-based tool mode, results are text in the message instead.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<RefusedToolResult>,
    /// Images in the message, such as tool screenshots, including its own and those
    /// inside its tool results.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub images: usize,
}

fn is_zero(count: &usize) -> bool {
    *count == 0
}

/// A tool result in a [`RefusedMessage`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RefusedToolResult {
    /// The provider's call id, which [`RefusalResolution::RedactToolResults`] takes.
    pub call_id: Option<String>,
    pub tool_name: String,
}

/// A change to a refused input. See [`Task::resolve_refusal`] and
/// `UserCommand::ResolveRefusal`. Changes name messages by their
/// [`RefusedMessage::message_index`] and never apply to runtime messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RefusalResolution {
    /// Replace the text of a message. Sections the runtime appended to it
    /// (`# Initial Environment Information`, `# Current Environment
    /// Information (updated this turn)`, `# Reminder`) are kept after the new
    /// text. Images and tool results are kept. In a message holding tool
    /// results this is the text beside them (such as a native `ask_user`
    /// answer), not an image description. The text must not be empty.
    /// Rephrasing the task prompt applies the prompt's `output_based_only`
    /// markers as the prompt builder does, and also
    /// updates [`Task::initial_task_prompt`].
    RephraseUserMessage { message_index: usize, text: String },
    /// Remove a message without tool results. At least one user message not
    /// added by the runtime must remain in the conversation, so the task
    /// prompt can't be removed when it is the only one.
    RemoveUserMessage { message_index: usize },
    /// Remove a message's images, including those inside its tool results (which keep
    /// their text) and its own, with the descriptions the runtime added
    /// before tool images sent beside the results.
    RemoveImages { message_index: usize },
    /// Replace the output text of tool results by provider call id, keeping
    /// each call paired with its result. Images inside the results stay.
    /// Remove them with `RemoveImages`.
    RedactToolResults {
        call_ids: Vec<String>,
        replacement: String,
    },
}

/// Runtime-added sections at the end of a message's text.
const ENVIRONMENT_SECTIONS: [&str; 2] = [
    "\n\n# Initial Environment Information\n",
    "\n\n# Current Environment Information (updated this turn)\n",
];

/// Where the runtime-added sections at the end of `text` start. Beside
/// native tool results the sections can be the whole text, without the
/// blank line before them.
pub(super) fn runtime_sections_start(text: &str, reminder: Option<&str>) -> usize {
    let mut end = text.len();
    if let Some(reminder) = reminder {
        let section = format!("\n\n# Reminder\n{reminder}");
        if text.ends_with(&section) {
            end -= section.len();
        } else if text == format!("# Reminder\n{reminder}") {
            return 0;
        }
    }
    ENVIRONMENT_SECTIONS
        .iter()
        .filter_map(|marker| text[..end].rfind(marker))
        .max()
        .or_else(|| {
            ENVIRONMENT_SECTIONS
                .iter()
                .any(|marker| text.starts_with(&marker[2..]))
                .then_some(0)
        })
        .unwrap_or(end)
}

/// A text part describing the image after it (added by the runtime for tool
/// images).
fn is_image_description(content: &[Content], index: usize) -> bool {
    content[index]
        .as_text()
        .is_some_and(|text| text.starts_with("Image from tool "))
        && content.get(index + 1).and_then(Content::as_image).is_some()
}

fn rephrase(message: &mut Message, text: &str, reminder: Option<&str>) {
    let target = (0..message.content.len()).find(|&index| {
        message.content[index].as_text().is_some() && !is_image_description(&message.content, index)
    });
    match target {
        Some(index) => {
            let old = message.content[index].as_text().unwrap_or_default();
            let kept = &old[runtime_sections_start(old, reminder)..];
            let separator = if kept.is_empty() || kept.starts_with("\n\n") {
                ""
            } else {
                "\n\n"
            };
            message.content[index] = Content::Text(format!("{text}{separator}{kept}"));
        }
        None => {
            let after_results = message
                .content
                .iter()
                .rposition(|content| content.as_tool_result().is_some())
                .map_or(0, |index| index + 1);
            message
                .content
                .insert(after_results, Content::Text(text.to_string()));
        }
    }
}

fn remove_images(message: &mut Message) {
    // The descriptions first: they are recognized by the image after them.
    let content = std::mem::take(&mut message.content);
    message.content = content
        .iter()
        .enumerate()
        .filter(|(index, _)| !is_image_description(&content, *index))
        .map(|(_, part)| part.clone())
        .collect();
    message.remove_images();
}

impl Task {
    /// The user messages after the last assistant turn: the input the next
    /// turn answers.
    pub(crate) fn current_input(&self) -> Option<RefusedInput> {
        self.input_of(&self.conversation.messages)
    }

    /// The user messages after the last assistant turn in `messages` (a
    /// prefix of the conversation).
    fn input_of(&self, messages: &[Message]) -> Option<RefusedInput> {
        let start = messages
            .iter()
            .rposition(|message| message.role == Role::Assistant)
            .map_or(0, |index| index + 1);
        let messages: Vec<RefusedMessage> = (start..messages.len())
            .filter(|&index| messages[index].role == Role::User)
            .map(|index| RefusedMessage {
                message_index: index,
                runtime: self.runtime_messages.contains(&index),
                runtime_text: self.runtime_text_messages.contains(&index),
                tool_results: messages[index]
                    .content
                    .iter()
                    .filter_map(Content::as_tool_result)
                    .map(|result| RefusedToolResult {
                        call_id: result.call_id.clone(),
                        tool_name: result.name.clone(),
                    })
                    .collect(),
                images: messages[index].image_count(),
            })
            .collect();
        (!messages.is_empty()).then_some(RefusedInput { messages })
    }

    /// Marks the conversation's last message as added by the runtime.
    pub(crate) fn mark_last_message_runtime(&mut self) {
        if let Some(index) = self.conversation.messages.len().checked_sub(1) {
            self.runtime_messages.insert(index);
        }
    }

    /// Marks the text beside the tool results of the conversation's last
    /// message as added by the runtime.
    pub(crate) fn mark_last_message_runtime_text(&mut self) {
        if let Some(index) = self.conversation.messages.len().checked_sub(1) {
            self.runtime_text_messages.insert(index);
        }
    }

    /// Changes the input of the latest refused turn ([`Task::refused_input`])
    /// and removes the refused turn, so that the task can continue with a
    /// request the provider has not refused. All changes are checked before
    /// any is applied. The refused input must still be the conversation's
    /// last input (followed only by the refused turn, if it was kept). Used
    /// by the runtime for `UserCommand::ResolveRefusal`. Call it directly on
    /// a task that was returned interrupted, before running it again.
    pub fn resolve_refusal(
        &mut self,
        changes: Vec<RefusalResolution>,
    ) -> Result<(), ArkoAgentsError> {
        let invalid = |message: String| ArkoAgentsError::InvalidTaskState(message);
        let input = self
            .refused_input
            .clone()
            .ok_or_else(|| invalid("The task has no refused input to resolve.".into()))?;
        if input.messages.is_empty() {
            return Err(invalid(
                "The refused input holds no messages to change.".into(),
            ));
        }
        if changes.is_empty() {
            return Err(invalid(
                "Give at least one change to the refused input.".into(),
            ));
        }
        if self.current_refused_input() != Some(input.clone()) {
            return Err(invalid(
                "The conversation changed since the refusal: the refused input is no longer its last input.".into(),
            ));
        }
        let last = input
            .messages
            .last()
            .map_or(0, |message| message.message_index);
        let find = |index: usize| {
            input
                .messages
                .iter()
                .find(|message| message.message_index == index)
                .ok_or_else(|| {
                    invalid(format!("Message {index} is not part of the refused input."))
                })
        };
        let mut removed = BTreeSet::new();
        let mut changed = BTreeSet::new();
        for change in &changes {
            let index = match change {
                RefusalResolution::RephraseUserMessage { message_index, .. }
                | RefusalResolution::RemoveUserMessage { message_index }
                | RefusalResolution::RemoveImages { message_index } => *message_index,
                RefusalResolution::RedactToolResults { call_ids, .. } => {
                    if call_ids.is_empty() {
                        return Err(invalid("Name at least one tool result to redact.".into()));
                    }
                    for call_id in call_ids {
                        let known = input.messages.iter().any(|message| {
                            message
                                .tool_results
                                .iter()
                                .any(|result| result.call_id.as_deref() == Some(call_id))
                        });
                        if !known {
                            return Err(invalid(format!(
                                "Call {call_id} is not among the refused tool results."
                            )));
                        }
                    }
                    continue;
                }
            };
            let message = find(index)?;
            if message.runtime {
                return Err(invalid(format!(
                    "Message {index} was added by the runtime; it can't be changed."
                )));
            }
            match change {
                RefusalResolution::RemoveUserMessage { .. } => {
                    if !message.tool_results.is_empty() {
                        return Err(invalid(format!(
                            "Message {index} holds tool results; redact them instead of removing it."
                        )));
                    }
                    removed.insert(index);
                }
                RefusalResolution::RemoveImages { .. } if message.images == 0 => {
                    return Err(invalid(format!("Message {index} has no images.")));
                }
                RefusalResolution::RephraseUserMessage { text, .. } => {
                    if message.runtime_text {
                        return Err(invalid(format!(
                            "The text of message {index} was added by the runtime; it can't be rephrased."
                        )));
                    }
                    if text.trim().is_empty() {
                        return Err(invalid("A rephrased message needs text.".into()));
                    }
                    changed.insert(index);
                }
                _ => {
                    changed.insert(index);
                }
            }
        }
        if !removed.is_disjoint(&changed) {
            return Err(invalid("A removed message can't also be changed.".into()));
        }
        // The runtime's own messages don't count: without a message from
        // the host the model has nothing to work on.
        let users_left = self.conversation.messages[..=last]
            .iter()
            .enumerate()
            .filter(|(index, message)| {
                message.role == Role::User
                    && !removed.contains(index)
                    && !self.runtime_messages.contains(index)
            })
            .count();
        if users_left == 0 {
            return Err(invalid(
                "Removing these messages would leave no user message from the host; rephrase instead.".into(),
            ));
        }

        let mut conversation = self.conversation.clone();
        conversation.messages.truncate(last + 1);
        let reminder = self
            .agent_config
            .user_reminder
            .as_ref()
            .map(|reminder| reminder.content.as_str());
        let task_prompt = conversation
            .messages
            .iter()
            .position(|message| message.role == Role::User);
        let mut initial_task_prompt = None;
        for change in &changes {
            match change {
                RefusalResolution::RephraseUserMessage {
                    message_index,
                    text,
                } => {
                    if task_prompt == Some(*message_index) {
                        // As the prompt builder formats the task prompt.
                        let processed = if self.agent_config.is_native_tool_calling() {
                            crate::parsing::strip_output_based_only_blocks(text)
                        } else {
                            crate::parsing::remove_output_based_only_markers(text)
                        };
                        rephrase(
                            &mut conversation.messages[*message_index],
                            &processed,
                            reminder,
                        );
                        initial_task_prompt = Some(text.clone());
                    } else {
                        rephrase(&mut conversation.messages[*message_index], text, reminder);
                    }
                }
                RefusalResolution::RemoveImages { message_index } => {
                    remove_images(&mut conversation.messages[*message_index]);
                }
                RefusalResolution::RedactToolResults {
                    call_ids,
                    replacement,
                } => {
                    for call_id in call_ids {
                        conversation
                            .redact_tool_result(call_id, replacement)
                            .map_err(|error| invalid(error.to_string()))?;
                    }
                }
                RefusalResolution::RemoveUserMessage { .. } => {}
            }
        }
        let kept = |marks: &BTreeSet<usize>| -> BTreeSet<usize> {
            marks
                .iter()
                .copied()
                .filter(|index| *index <= last)
                .collect()
        };
        let mut runtime_messages = kept(&self.runtime_messages);
        let mut runtime_text_messages = kept(&self.runtime_text_messages);
        for &index in removed.iter().rev() {
            conversation.messages.remove(index);
            let renumber = |marks: BTreeSet<usize>| -> BTreeSet<usize> {
                marks
                    .into_iter()
                    .filter(|marked| *marked != index)
                    .map(|marked| if marked > index { marked - 1 } else { marked })
                    .collect()
            };
            runtime_messages = renumber(runtime_messages);
            runtime_text_messages = renumber(runtime_text_messages);
        }

        self.conversation = conversation;
        self.runtime_messages = runtime_messages;
        self.runtime_text_messages = runtime_text_messages;
        if let Some(prompt) = initial_task_prompt {
            self.initial_task_prompt = prompt;
        }
        self.user_message_count = self.user_message_count.saturating_sub(removed.len());
        self.refused_input = None;
        Ok(())
    }

    /// The conversation's current input, if it still is the refused input:
    /// followed by nothing, or by one assistant turn without tool calls (the
    /// kept refusal).
    fn current_refused_input(&self) -> Option<RefusedInput> {
        let messages = &self.conversation.messages;
        let end = match messages.last() {
            Some(message) if message.role == Role::Assistant => {
                if !message.tool_calls.is_empty() {
                    return None;
                }
                messages.len() - 1
            }
            _ => messages.len(),
        };
        self.input_of(&messages[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentLLMConfig, UserReminder};
    use arko_ai::{Image, ToolCall, ToolResult};

    fn task() -> Task {
        let config = crate::agent_config::AgentConfig::builder(
            "refusal".into(),
            "Refusal".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .user_reminder(UserReminder::initial_only("Stay on task.".into()))
        .build()
        .unwrap();
        Task::new(config, "old task".into(), std::env::temp_dir(), None)
    }

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: Some(id.into()),
            name: "screenshot".into(),
            arguments: serde_json::json!({}),
            thought_signature: None,
            provider_id: None,
            invalid_arguments: None,
        }
    }

    /// A prompt with runtime sections, a turn calling a tool, and its
    /// result with an ask_user-style answer and a screenshot, then a runtime
    /// note. The last two are the refused input.
    fn refused_task() -> Task {
        let mut task = task();
        task.conversation.add_user_message(
            "old task\n\n# Initial Environment Information\ncwd: /work\n\n# Reminder\nStay on task.",
        );
        let mut turn = Message::new_assistant("");
        turn.tool_calls.push(call("call_1"));
        task.conversation.messages.push(turn);
        task.conversation.add_user_message_with_contents(vec![
            Content::ToolResult(ToolResult::from_tool_call(&call("call_1"), "secret output")),
            Content::Text(
                "Answer: yes\n\n# Current Environment Information (updated this turn)\ncwd: /work"
                    .into(),
            ),
            Content::Text("Image from tool screenshot, call call_1: the screen".into()),
            Content::Image(Image::new_url("https://example.com/screen.png")),
        ]);
        task.conversation
            .add_user_message("Continue the task from where your previous response ended.");
        task.mark_last_message_runtime();
        task.user_message_count = 3;
        task.refused_input = task.current_input();
        task.conversation
            .add_assistant_message("I can't help with that.");
        task
    }

    #[test]
    fn the_refused_input_lists_every_message_since_the_last_assistant_turn() {
        let task = refused_task();
        let input = task.refused_input.clone().unwrap();
        let indices: Vec<(usize, bool)> = input
            .messages
            .iter()
            .map(|message| (message.message_index, message.runtime))
            .collect();
        assert_eq!(indices, [(2, false), (3, true)]);
        assert_eq!(
            input.messages[0].tool_results[0].call_id.as_deref(),
            Some("call_1")
        );
        assert_eq!(input.messages[0].images, 1);
    }

    #[test]
    fn rephrasing_keeps_the_sections_the_runtime_added() {
        let mut task = task();
        task.conversation.add_user_message(
            "old task\n\n# Initial Environment Information\ncwd: /work\n\n# Reminder\nStay on task.",
        );
        task.refused_input = task.current_input();
        task.resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
            message_index: 0,
            text: "new task".into(),
        }])
        .unwrap();
        assert_eq!(
            task.conversation.messages[0].content[0].as_text(),
            Some(
                "new task\n\n# Initial Environment Information\ncwd: /work\n\n# Reminder\nStay on task."
            )
        );
        assert_eq!(task.initial_task_prompt, "new task");
    }

    /// Beside native tool results the runtime's sections can be the whole
    /// text. Rephrasing puts the new text before them.
    #[test]
    fn rephrasing_beside_results_keeps_sections_that_are_the_whole_text() {
        let mut task = task();
        task.conversation.add_user_message("task");
        let mut turn = Message::new_assistant("");
        turn.tool_calls.push(call("call_1"));
        task.conversation.messages.push(turn);
        task.conversation.add_user_message_with_contents(vec![
            Content::ToolResult(ToolResult::from_tool_call(&call("call_1"), "output")),
            Content::Text(
                "# Current Environment Information (updated this turn)\ncwd: /work\n\n# Reminder\nStay on task."
                    .into(),
            ),
        ]);
        task.refused_input = task.current_input();
        task.resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
            message_index: 2,
            text: "Answer: no".into(),
        }])
        .unwrap();
        assert_eq!(
            task.conversation.messages[2].content[1].as_text(),
            Some(
                "Answer: no\n\n# Current Environment Information (updated this turn)\ncwd: /work\n\n# Reminder\nStay on task."
            )
        );
    }

    #[test]
    fn a_tool_results_message_can_have_its_text_and_images_changed() {
        let mut task = refused_task();
        task.resolve_refusal(vec![
            RefusalResolution::RephraseUserMessage {
                message_index: 2,
                text: "Answer: no".into(),
            },
            RefusalResolution::RemoveImages { message_index: 2 },
            RefusalResolution::RedactToolResults {
                call_ids: vec!["call_1".into()],
                replacement: "[removed]".into(),
            },
        ])
        .unwrap();
        let content = &task.conversation.messages[2].content;
        assert_eq!(content.len(), 2, "{content:?}");
        assert_eq!(content[0].as_tool_result().unwrap().output, "[removed]");
        assert_eq!(
            content[1].as_text(),
            Some("Answer: no\n\n# Current Environment Information (updated this turn)\ncwd: /work")
        );
        // The refused turn is gone; the runtime note stays.
        assert_eq!(task.conversation.messages.len(), 4);
        assert!(task.runtime_messages.contains(&3));
        assert!(task.refused_input.is_none());
    }

    /// Images inside tool results count as the message's images.
    /// `RemoveImages` removes them (keeping each result's text), and
    /// redaction replaces only the text.
    #[test]
    fn images_inside_tool_results_are_counted_and_removed() {
        use arko_ai::ToolResultImage;
        let result_with_image = || {
            ToolResult::from_tool_call(&call("call_1"), "secret output").with_images(vec![
                ToolResultImage::new("the screen", Image::new_url("https://example.com/s.png")),
            ])
        };
        let refused = |task: &mut Task| {
            task.conversation.add_user_message("task");
            let mut turn = Message::new_assistant("");
            turn.tool_calls.push(call("call_1"));
            task.conversation.messages.push(turn);
            task.conversation
                .add_user_message_with_contents(vec![Content::ToolResult(result_with_image())]);
            task.refused_input = task.current_input();
        };
        let mut task = task();
        refused(&mut task);
        assert_eq!(task.refused_input.as_ref().unwrap().messages[0].images, 1);
        task.resolve_refusal(vec![RefusalResolution::RedactToolResults {
            call_ids: vec!["call_1".into()],
            replacement: "[removed]".into(),
        }])
        .unwrap();
        let result = task.conversation.messages[2].content[0]
            .as_tool_result()
            .unwrap();
        assert_eq!(result.output, "[removed]");
        assert_eq!(result.images.len(), 1, "redaction keeps the image");

        let mut task = self::task();
        refused(&mut task);
        task.resolve_refusal(vec![RefusalResolution::RemoveImages { message_index: 2 }])
            .unwrap();
        let result = task.conversation.messages[2].content[0]
            .as_tool_result()
            .unwrap();
        assert_eq!(result.output, "secret output");
        assert!(result.images.is_empty());
        assert_eq!(task.conversation.messages[2].content.len(), 1);
    }

    #[test]
    fn invalid_or_stale_changes_apply_nothing() {
        let original = refused_task();
        for changes in [
            // The runtime's note can't be changed, so nothing is redacted.
            vec![
                RefusalResolution::RedactToolResults {
                    call_ids: vec!["call_1".into()],
                    replacement: "[removed]".into(),
                },
                RefusalResolution::RephraseUserMessage {
                    message_index: 3,
                    text: "x".into(),
                },
            ],
            vec![RefusalResolution::RemoveUserMessage { message_index: 2 }],
            vec![RefusalResolution::RemoveImages { message_index: 3 }],
            vec![RefusalResolution::RephraseUserMessage {
                message_index: 0,
                text: "not part of the input".into(),
            }],
            vec![],
        ] {
            let mut task = original.clone();
            assert!(
                task.resolve_refusal(changes.clone()).is_err(),
                "{changes:?}"
            );
            assert_eq!(
                serde_json::to_value(&task.conversation).unwrap(),
                serde_json::to_value(&original.conversation).unwrap()
            );
            assert!(task.refused_input.is_some());
        }
        // Input added after the refusal makes the recorded one stale.
        let mut task = original.clone();
        task.conversation.add_user_message("a new message");
        assert!(
            task.resolve_refusal(vec![RefusalResolution::RedactToolResults {
                call_ids: vec!["call_1".into()],
                replacement: "[removed]".into(),
            }])
            .is_err()
        );
        let result = task.conversation.messages[2].content[0]
            .as_tool_result()
            .unwrap();
        assert_eq!(result.output, "secret output");
    }

    #[test]
    fn removing_a_message_renumbers_runtime_messages_and_counts() {
        let mut task = task();
        task.conversation.add_user_message("first");
        task.conversation.add_assistant_message("ok");
        task.conversation.add_user_message("objectionable");
        task.conversation
            .add_user_message("# File tree (max depth 3)\n...");
        task.mark_last_message_runtime();
        task.user_message_count = 3;
        task.refused_input = task.current_input();
        task.resolve_refusal(vec![RefusalResolution::RemoveUserMessage {
            message_index: 2,
        }])
        .unwrap();
        assert_eq!(task.conversation.messages.len(), 3);
        assert_eq!(task.runtime_messages, BTreeSet::from([2]));
        assert_eq!(task.user_message_count, 2);
    }

    #[test]
    fn the_task_prompt_stays_when_only_runtime_messages_would_remain() {
        let mut task = task();
        task.conversation.add_user_message("the task");
        task.conversation
            .add_user_message("# File tree (max depth 3)\n...");
        task.mark_last_message_runtime();
        task.refused_input = task.current_input();
        let error = task
            .resolve_refusal(vec![RefusalResolution::RemoveUserMessage {
                message_index: 0,
            }])
            .unwrap_err();
        assert!(
            error.to_string().contains("no user message from the host"),
            "{error}"
        );
        assert_eq!(task.conversation.messages.len(), 2);
    }

    #[test]
    fn rephrasing_needs_text_and_formats_the_task_prompt_like_the_prompt_builder() {
        let mut task = task();
        task.agent_config.tool_config.tool_calling_mode =
            crate::agent_config::ToolCallingMode::Native;
        task.conversation.add_user_message("old task");
        task.refused_input = task.current_input();
        assert!(
            task.resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
                message_index: 0,
                text: "  ".into(),
            }])
            .is_err()
        );
        let text = "{{#output_based_only}}Use XML tags.{{/output_based_only}}Do the task.";
        task.resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
            message_index: 0,
            text: text.into(),
        }])
        .unwrap();
        assert_eq!(
            task.conversation.messages[0].content[0].as_text(),
            Some("Do the task.")
        );
        assert_eq!(task.initial_task_prompt, text);
    }

    #[test]
    fn a_runtime_note_beside_tool_results_is_not_rephrased() {
        let mut task = task();
        task.conversation.add_user_message("task");
        let mut turn = Message::new_assistant("");
        turn.tool_calls.push(call("call_1"));
        task.conversation.messages.push(turn);
        task.conversation.add_user_message_with_contents(vec![
            Content::ToolResult(ToolResult::from_tool_call(&call("call_1"), "output")),
            Content::Text("The previous attempt resulted in an error: ...".into()),
        ]);
        task.mark_last_message_runtime_text();
        task.refused_input = task.current_input();
        assert!(task.refused_input.as_ref().unwrap().messages[0].runtime_text);
        let error = task
            .resolve_refusal(vec![RefusalResolution::RephraseUserMessage {
                message_index: 2,
                text: "x".into(),
            }])
            .unwrap_err();
        assert!(
            error.to_string().contains("added by the runtime"),
            "{error}"
        );
        task.resolve_refusal(vec![RefusalResolution::RedactToolResults {
            call_ids: vec!["call_1".into()],
            replacement: "[removed]".into(),
        }])
        .unwrap();
    }

    #[test]
    fn a_saved_input_without_messages_loads_and_is_rejected_cleanly() {
        let input: RefusedInput = serde_json::from_str("{}").unwrap();
        assert!(input.messages.is_empty());
        let mut task = task();
        task.conversation.add_user_message("task");
        task.refused_input = Some(input);
        let error = task
            .resolve_refusal(vec![RefusalResolution::RemoveImages { message_index: 0 }])
            .unwrap_err();
        assert!(error.to_string().contains("no messages"), "{error}");
    }
}
