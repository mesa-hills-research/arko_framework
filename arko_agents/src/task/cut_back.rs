//! Cutting a task's conversation back to just before a message the host
//! sent (to edit it or send it again), and branching a copy of it.

use arko_ai::{Content, Message, Role};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Task, TaskState};
use crate::error::ArkoAgentsError;

/// What a message the host sent to a task was; see [`Task::user_turns`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserTurnKind {
    /// The task's prompt, the conversation's first message.
    Prompt,
    /// A later message: a follow-up, or an answer to the agent's question.
    Message,
    /// A `# Workflow context update`: notes, messages and other context the
    /// host added while the task ran.
    ContextUpdate,
}

/// A message the host sent to a task; see [`Task::user_turns`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserTurn {
    /// Its index in the task's conversation, which [`Task::cut_back`] and
    /// [`Task::branch`] take.
    pub index: usize,
    pub kind: UserTurnKind,
    /// Its text, without the environment details and reminder the runtime
    /// added after it (a workflow's own additions, such as messages from
    /// other agents, stay).
    pub text: String,
    /// How many images it holds.
    pub images: usize,
}

const CONTEXT_UPDATE: &str = "# Workflow context update";

impl Task {
    /// The messages of the current conversation the host sent, oldest
    /// first: the task's prompt, later messages and context updates. The
    /// runtime's own messages (requests to continue, notes on errors) and
    /// tool results are left out. Messages before a handover are not in
    /// the conversation any more.
    pub fn user_turns(&self) -> Vec<UserTurn> {
        self.conversation
            .messages
            .iter()
            .enumerate()
            .filter(|(index, message)| self.is_user_turn(*index, message))
            .map(|(index, message)| {
                let text = self.host_text(message);
                let kind = if text.starts_with(CONTEXT_UPDATE) {
                    UserTurnKind::ContextUpdate
                } else if index == 0 {
                    UserTurnKind::Prompt
                } else {
                    UserTurnKind::Message
                };
                UserTurn {
                    index,
                    kind,
                    text,
                    images: message
                        .content
                        .iter()
                        .filter(|c| c.as_image().is_some())
                        .count(),
                }
            })
            .collect()
    }

    /// The text of a message the host sent, without the environment details
    /// and reminder the runtime added after it.
    pub(crate) fn host_text(&self, message: &Message) -> String {
        let reminder = self
            .agent_config
            .user_reminder
            .as_ref()
            .map(|reminder| reminder.content.as_str());
        let text = message
            .content
            .iter()
            .filter_map(Content::as_text)
            .next()
            .unwrap_or_default();
        text[..super::refusal::runtime_sections_start(text, reminder)].to_string()
    }

    fn is_user_turn(&self, index: usize, message: &Message) -> bool {
        message.role == Role::User
            && !self.runtime_messages.contains(&index)
            && message
                .content
                .iter()
                .all(|content| content.as_tool_result().is_none())
    }

    /// Cuts the conversation back to just before message `index`, one of
    /// [`user_turns`](Self::user_turns), to send it again or a changed one
    /// in its place. Everything from it on goes: the later messages, the
    /// calls and results waiting with the task, context taken and not yet
    /// added, a refusal or handover under way. The task's usage stays, and
    /// its context size is unknown until its next request. The task is then
    /// `Completed` (with the last reply kept as its summary), so a follow-up
    /// continues it
    /// ([`prepare_for_resumption_from_completed`](Self::prepare_for_resumption_from_completed)).
    /// Cut back to its first message (`0`), it is `Pending` with no
    /// conversation, and runs from [`initial_task_prompt`](Self::initial_task_prompt)
    /// (which the host replaces) like a new task.
    ///
    /// A Claude Code task's session resumes at the last entry before the
    /// cut, and the host's messages after that entry go with the next input
    /// ([`claude_code_resume_at`](Self::claude_code_resume_at)). A cut that
    /// would leave the model's work without its entry (a task recorded
    /// before entries were, or a compaction in between) is refused with
    /// `InvalidTaskState`, as is an index that isn't a user turn. The host
    /// cuts back a task its runtime isn't running.
    pub fn cut_back(&mut self, index: usize) -> Result<(), ArkoAgentsError> {
        let Some(message) = self.conversation.messages.get(index) else {
            return Err(ArkoAgentsError::InvalidTaskState(format!(
                "message {index} is past the conversation's end ({} messages)",
                self.conversation.messages.len()
            )));
        };
        if !self.is_user_turn(index, message) {
            return Err(ArkoAgentsError::InvalidTaskState(format!(
                "message {index} is not a message the host sent (see Task::user_turns)"
            )));
        }
        self.cut_at(index)
    }

    /// A copy of the task, as a new task (its own ID, no usage, no parent)
    /// whose conversation is cut back to just before message `index`, as
    /// [`cut_back`](Self::cut_back) does, or holds all of it with `index`
    /// the conversation's length. The original stays as it is. A Claude Code
    /// task's copy forks the session when it first runs
    /// ([`claude_code_fork_of`](Self::claude_code_fork_of)), so both
    /// conversations go on apart. Index `0` is refused: a new run of the
    /// agent starts from a prompt.
    pub fn branch(&self, index: usize) -> Result<Task, ArkoAgentsError> {
        if index == 0 {
            return Err(ArkoAgentsError::InvalidTaskState(
                "a branch keeps at least the task's first message; start a new task instead"
                    .to_string(),
            ));
        }
        let mut branch = self.clone();
        branch.id = Uuid::new_v4();
        branch.parent = None;
        branch.start_time = chrono::Utc::now();
        branch.usage_history.clear();
        branch.model_call_usage_history.clear();
        branch.fallback_calls.clear();
        branch.plan_usage_threshold_reported = None;
        branch.monitor_pause = false;
        if index == self.conversation.messages.len() {
            branch.cut_at(index)?;
        } else {
            branch.cut_back(index)?;
        }
        branch.conversation_usage_start = 0;
        branch.conversation_model_call_start = 0;
        branch.cache_window_start = 0;
        // A task still to fork its session (a branch that hasn't run) is
        // branched from the same session.
        if let Some(own) = branch.claude_code_session.take() {
            let parent = branch.claude_code_fork_of.take().unwrap_or(own);
            branch.claude_code_fork_of = Some(parent);
            branch.claude_code_session = Some(Uuid::new_v4().to_string());
        }
        Ok(branch)
    }

    fn cut_at(&mut self, index: usize) -> Result<(), ArkoAgentsError> {
        // First, as it can refuse: nothing changes then.
        self.cut_claude_code_session(index)?;
        let messages = &mut self.conversation.messages;
        messages.truncate(index);
        self.conversation.openai_turn_state = None;
        self.runtime_messages.retain(|kept| *kept < index);
        self.runtime_text_messages.retain(|kept| *kept < index);
        self.refused_input = None;
        self.fallback_for = None;
        self.pending_native_tool_results.clear();
        self.pending_native_tool_calls.clear();
        self.pending_tool_images.clear();
        self.pending_text_tool_results.clear();
        self.deferred_tool_errors.clear();
        self.running_tool_calls.clear();
        self.pending_context_injections.clear();
        self.pending_context_images.clear();
        self.handover_requested = None;
        if index == 0 {
            // A new conversation: its usage starts here.
            self.conversation_usage_start = self.usage_history.len();
            self.conversation_model_call_start = self.model_call_usage_history.len();
            self.context_size_unknown = false;
        } else {
            // The same conversation, shorter: its first call (which decides
            // whether it can hand over) stays, and its size is unknown until
            // the next call.
            self.context_size_unknown = true;
        }
        self.user_message_count = self
            .conversation
            .messages
            .iter()
            .filter(|message| message.role == Role::User)
            .count();
        if index == 0 {
            self.conversation.system_message = None;
            self.set_state(TaskState::Pending);
        } else {
            let summary = match self.conversation.messages.last() {
                Some(message) if message.role == Role::Assistant => message
                    .content
                    .iter()
                    .filter_map(Content::as_text)
                    .collect(),
                _ => String::new(),
            };
            self.set_state(TaskState::Completed { summary });
        }
        Ok(())
    }

    /// Where a cut back to just before message `index` leaves the Claude
    /// Code session: resumed at the last entry before it, with the host's
    /// messages after that entry sent again; or, with no entry before it,
    /// a new session.
    fn cut_claude_code_session(&mut self, index: usize) -> Result<(), ArkoAgentsError> {
        if self.claude_code_session.is_none() {
            self.claude_code_synced = self.claude_code_synced.min(index);
            self.claude_code_entries.retain(|kept, _| *kept < index);
            return Ok(());
        }
        let entry = self
            .claude_code_entries
            .range(..index)
            .next_back()
            .map(|(kept, uuid)| (*kept, uuid.clone()));
        let from = entry.as_ref().map_or(0, |(kept, _)| kept + 1);
        let resent = &self.conversation.messages[from..index];
        if !resent
            .iter()
            .enumerate()
            .all(|(offset, message)| self.is_user_turn(from + offset, message))
        {
            return Err(ArkoAgentsError::InvalidTaskState(format!(
                "the Claude Code session can't be cut back to message {index}: its transcript entry for message {} is not known (recorded before entries were, or before a compaction)",
                from + resent
                    .iter()
                    .position(|message| message.role != Role::User)
                    .unwrap_or(0)
            )));
        }
        match entry {
            Some((_, uuid)) => self.claude_code_resume_at = Some(uuid),
            None => {
                // Nothing of the session stays: the next run starts one.
                self.claude_code_session = None;
                self.claude_code_resume_at = None;
                self.claude_code_fork_of = None;
                self.claude_code_prompt_changed = false;
            }
        }
        self.claude_code_entries.retain(|kept, _| *kept < index);
        self.claude_code_synced = from;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentConfig, AgentLLMConfig};
    use arko_ai::{Image, ToolCall, ToolResult};

    fn task() -> Task {
        let config = AgentConfig::builder(
            "chat".into(),
            "Chat".into(),
            AgentLLMConfig {
                model: arko_ai::Model::anthropic_claude_opus_5_5(),
                generation_config: Default::default(),
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .build()
        .unwrap();
        Task::new(config, "Plan a trip.".into(), "/w".into(), None)
    }

    fn call() -> ToolCall {
        serde_json::from_value(serde_json::json!({"id": "c1", "name": "lookup", "arguments": {}}))
            .unwrap()
    }

    /// A conversation of two turns: the prompt (with environment details),
    /// a reply; a follow-up with an image, a call, its result, a context
    /// update, a reply; and a runtime note.
    fn conversation(task: &mut Task) {
        let messages = &mut task.conversation;
        messages.add_user_message("Plan a trip.\n\n# Initial Environment Information\ncwd: /w");
        messages.add_assistant_message("Where to?");
        messages.add_user_message_with_contents(vec![
            Content::Text("Lisbon, like this.".into()),
            Content::Image(Image::new_base64("cGl4", "image/png")),
        ]);
        let mut reply = Message::new_assistant("Looking.");
        reply.tool_calls = vec![call()];
        messages.messages.push(reply);
        messages.add_tool_results(vec![ToolResult::from_tool_call(&call(), "sunny")]);
        messages.add_user_message("# Workflow context update\n\nA note: in May.");
        messages.add_assistant_message("Lisbon in May is sunny.");
        messages.add_user_message("Continue.");
        task.runtime_messages.insert(7);
        task.set_state(TaskState::Completed {
            summary: "Lisbon in May is sunny.".into(),
        });
    }

    #[test]
    fn the_host_s_messages_are_its_user_turns() {
        let mut task = task();
        conversation(&mut task);
        let turns = task.user_turns();
        let summary: Vec<_> = turns
            .iter()
            .map(|turn| (turn.index, turn.kind, turn.text.as_str(), turn.images))
            .collect();
        assert_eq!(
            summary,
            [
                (0, UserTurnKind::Prompt, "Plan a trip.", 0),
                (2, UserTurnKind::Message, "Lisbon, like this.", 1),
                (
                    5,
                    UserTurnKind::ContextUpdate,
                    "# Workflow context update\n\nA note: in May.",
                    0
                ),
            ]
        );
    }

    /// The cut leaves what came before the message, and the task ready for
    /// a follow-up; the usage stays, the context size starts over.
    #[test]
    fn a_cut_leaves_the_conversation_before_the_message() {
        let mut task = task();
        conversation(&mut task);
        task.usage_history.push(Default::default());
        task.pending_context_injections.push("Later.".into());
        task.handover_requested = Some(crate::handover::HandoverRequest::new(10));
        task.cut_back(2).unwrap();
        assert_eq!(task.conversation.messages.len(), 2);
        assert_eq!(
            task.current_state,
            TaskState::Completed {
                summary: "Where to?".into()
            }
        );
        assert!(task.runtime_messages.is_empty());
        assert!(task.pending_context_injections.is_empty());
        assert!(task.handover_requested.is_none());
        assert_eq!(task.usage_history.len(), 1);
        assert_eq!(task.current_conversation_context_size(), None);
        assert_eq!(task.user_message_count, 1);
        task.prepare_for_resumption_from_completed("Porto instead.".into())
            .unwrap();
        assert_eq!(task.conversation.messages.len(), 3);

        // A context update in the middle of a turn: its results stay.
        let mut task = self::task();
        conversation(&mut task);
        task.cut_back(5).unwrap();
        assert_eq!(task.conversation.messages.len(), 5);
        assert_eq!(
            task.current_state,
            TaskState::Completed {
                summary: String::new()
            }
        );
    }

    #[test]
    fn only_a_user_turn_is_cut_back_to() {
        let mut task = task();
        conversation(&mut task);
        for index in [1, 4, 7, 8, 9] {
            let error = task.clone().cut_back(index).unwrap_err().to_string();
            assert!(
                error.contains("not a message the host sent") || error.contains("past"),
                "{index}: {error}"
            );
        }
        // The first message: a fresh start from the task's prompt.
        task.cut_back(0).unwrap();
        assert!(task.conversation.messages.is_empty());
        assert!(task.conversation.system_message.is_none());
        assert_eq!(task.current_state, TaskState::Pending);
    }

    /// A Claude Code session resumes at the entry before the cut, and the
    /// host's messages after that entry go again; without the model's
    /// entries, the cut is refused.
    #[test]
    fn a_claude_code_session_resumes_at_the_entry_before_the_cut() {
        let mut task = task();
        conversation(&mut task);
        task.claude_code_session = Some("s1".into());
        task.claude_code_synced = 8;
        for (index, uuid) in [
            (0, "u0"),
            (1, "a1"),
            (2, "u2"),
            (3, "a3"),
            (4, "r4"),
            (6, "a6"),
        ] {
            task.claude_code_entries.insert(index, uuid.into());
        }
        let mut cut = task.clone();
        cut.cut_back(5).unwrap();
        assert_eq!(cut.claude_code_resume_at.as_deref(), Some("r4"));
        assert_eq!(cut.claude_code_synced, 5);
        assert_eq!(cut.claude_code_entries.len(), 5);

        // The follow-up was sent with a note added before it went out: both
        // go again.
        let mut bundled = task.clone();
        bundled.claude_code_entries.remove(&2);
        bundled.cut_back(2).unwrap();
        assert_eq!(bundled.claude_code_resume_at.as_deref(), Some("a1"));
        assert_eq!(bundled.claude_code_synced, 2);

        // The first message: a new session.
        let mut fresh = task.clone();
        fresh.cut_back(0).unwrap();
        assert!(fresh.claude_code_session.is_none());

        // The model's reply has no entry: refused, nothing changed.
        let mut unknown = task.clone();
        unknown.claude_code_entries.remove(&3);
        unknown.claude_code_entries.remove(&4);
        let error = unknown.cut_back(5).unwrap_err().to_string();
        assert!(error.contains("transcript entry"), "{error}");
        assert_eq!(unknown.conversation.messages.len(), 8);
    }

    /// A branch is a new task with the conversation up to the message, no
    /// usage, and a Claude Code session forked from the original's.
    #[test]
    fn a_branch_is_a_new_task_that_forks_the_session() {
        let mut task = task();
        conversation(&mut task);
        task.usage_history.push(Default::default());
        task.claude_code_session = Some("s1".into());
        task.claude_code_entries.insert(1, "a1".into());
        task.claude_code_entries.insert(6, "a6".into());
        let branch = task.branch(2).unwrap();
        assert_ne!(branch.id, task.id);
        assert_eq!(branch.conversation.messages.len(), 2);
        assert!(branch.usage_history.is_empty());
        assert_eq!(branch.claude_code_fork_of.as_deref(), Some("s1"));
        assert_ne!(branch.claude_code_session.as_deref(), Some("s1"));
        assert_eq!(branch.claude_code_resume_at.as_deref(), Some("a1"));
        assert_eq!(
            branch.conversation.prompt_cache_key,
            task.conversation.prompt_cache_key
        );
        // The original is unchanged.
        assert_eq!(task.conversation.messages.len(), 8);

        // All of it: the conversation's length.
        let whole = task.branch(8);
        assert!(whole.is_err(), "the runtime's note has no entry");
        task.conversation.messages.pop();
        task.runtime_messages.clear();
        let whole = task.branch(7).unwrap();
        assert_eq!(whole.conversation.messages.len(), 7);
        assert_eq!(whole.claude_code_resume_at.as_deref(), Some("a6"));
        // A branch of a branch that hasn't run forks the same session.
        let again = whole.branch(7).unwrap();
        assert_eq!(again.claude_code_fork_of.as_deref(), Some("s1"));

        assert!(task.branch(0).is_err());
    }
}
