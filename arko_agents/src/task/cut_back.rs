//! Cutting a task's conversation back to just before a message the host
//! sent (to edit it or send it again), and branching a copy of it.

use arko_ai::{Content, Message, Role};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Task, TaskState};
use crate::agent_config::ToolCallingMode;
use crate::error::ArkoAgentsError;

/// The result of a call that was running when the task stopped (its host
/// crashed, say), settled as `ToolRecovery::InDoubt` when the task runs
/// again, or answered so in a branch of the task.
pub(crate) const RESTARTED_CALL: &str = "The task was restarted while this call was running (its host stopped), so it may have run partly or fully. Check its effects before calling it again.";

/// The result of a call of a turn that ended before the call ran.
pub(crate) const UNREACHED_CALL: &str = "The turn ended before this call ran. It was not run.";

/// What a message the host sent to a task was. See [`Task::user_turns`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserTurnKind {
    /// The task's prompt, the conversation's first message.
    Prompt,
    /// A later message, either a follow-up or an answer to the agent's question.
    Message,
    /// A `# Workflow context update` with notes, messages and other context the
    /// host added while the task ran.
    ContextUpdate,
}

/// A message the host sent to a task. See [`Task::user_turns`].
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

pub(crate) const CONTEXT_UPDATE: &str = "# Workflow context update";

impl Task {
    /// The messages of the current conversation the host sent, oldest
    /// first, including the task's prompt, later messages and context updates. The
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
    /// in its place. Everything from it on goes, which is the later
    /// messages, the calls and results waiting with the task, context taken
    /// and not yet added, and a refusal or handover under way. The task's usage stays, and
    /// its context size is unknown until its next request. The task is then
    /// `Completed` (with the last reply kept as its summary), so a follow-up
    /// continues it
    /// ([`prepare_for_resumption_from_completed`](Self::prepare_for_resumption_from_completed)).
    /// Cut back to its first message (`0`), it is `Pending` with no
    /// conversation, and runs from [`initial_task_prompt`](Self::initial_task_prompt)
    /// (which the host replaces) like a new task. A task with no
    /// conversation (a [`branch`](Self::branch) at its first message) is cut
    /// back to `0` too.
    ///
    /// A Claude Code task's session resumes at the last entry before the
    /// cut, and the host's messages after that entry go with the next input
    /// ([`claude_code_resume_at`](Self::claude_code_resume_at)). A cut that
    /// would leave the model's work without its entry (a task recorded
    /// before entries were, or a compaction in between) is refused with
    /// `InvalidTaskState`, as is an index that isn't a user turn. The host
    /// cuts back a task its runtime isn't running.
    pub fn cut_back(&mut self, index: usize) -> Result<(), ArkoAgentsError> {
        if index == 0 && self.conversation.messages.is_empty() {
            return self.cut_at(0);
        }
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
    /// conversations go on apart. A whole copy of a task stopped while its
    /// last turn's calls ran answers each of them, so the copy's next
    /// request pairs every call with a result, as providers require.
    ///
    /// At its first message (`0`), the copy has no conversation but keeps
    /// the rest, such as its [`handovers`](Self::handovers), whose latest
    /// report a host may start the branch's conversation from. It is
    /// `Completed` with an empty summary, for the host to import, and the
    /// host continues it cut back to its first message
    /// (`ContinueOptions::cut_back_to(0)` in `arko_workflows`), which
    /// starts it over from the host's message as a new task starts.
    pub fn branch(&self, index: usize) -> Result<Task, ArkoAgentsError> {
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
            // A task stopped while its last turn's calls ran: each call gets
            // a result, as providers require of the next request.
            branch.answer_open_calls();
            branch.cut_at(branch.conversation.messages.len())?;
        } else {
            branch.cut_back(index)?;
        }
        if index == 0 {
            branch.set_state(TaskState::Completed {
                summary: String::new(),
            });
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

    /// Answers the native calls of the conversation's last turn that have
    /// no result yet. A result the task holds for one goes in. A call that
    /// was running reads as one a restarted task settles
    /// ([`RESTARTED_CALL`]), and one that never ran as not run
    /// ([`UNREACHED_CALL`]). A Claude Code task's session keeps its own
    /// record of the turn.
    fn answer_open_calls(&mut self) {
        if self.agent_config.tool_config.tool_calling_mode != ToolCallingMode::Native
            || self.agent_config.runs_in_claude_code()
        {
            return;
        }
        let Some(turn) =
            self.conversation.messages.last().filter(|message| {
                message.role == Role::Assistant && !message.tool_calls.is_empty()
            })
        else {
            return;
        };
        let mut results = Vec::new();
        for call in &turn.tool_calls {
            let Some(id) = &call.id else {
                continue;
            };
            let held = self
                .pending_native_tool_results
                .iter()
                .find(|result| result.call_id.as_ref() == Some(id));
            let running = self
                .running_tool_calls
                .iter()
                .any(|running| running.call_id.as_ref() == Some(id));
            results.push(match held {
                Some(result) => result.clone(),
                None if running => arko_ai::ToolResult::error_from_tool_call(call, RESTARTED_CALL),
                None => arko_ai::ToolResult::error_from_tool_call(call, UNREACHED_CALL),
            });
        }
        if !results.is_empty() {
            self.pending_native_tool_results = results;
            self.add_pending_tool_results(None);
        }
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
        // What it took may have gone with the messages cut, so a host reads
        // its conversation again.
        self.context_marks.clear();
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

    /// Has the task's Claude Code session resume at the end of the task's
    /// conversation as it is now, without what the session received or
    /// produced after that point, such as a turn a host undid by putting
    /// back an earlier copy of the task. The next launch resumes the session
    /// at the transcript entry of the last message the task holds an entry
    /// for, and the host's messages after it are sent again, as after
    /// [`cut_back`](Self::cut_back). Nothing changes for a task without a
    /// session. Returns `InvalidTaskState` when a message after that entry
    /// isn't the host's.
    pub fn rewind_claude_code_session(&mut self) -> Result<(), ArkoAgentsError> {
        self.cut_claude_code_session(self.conversation.messages.len())
    }

    /// Where a cut back to just before message `index` leaves the Claude
    /// Code session: resumed at the last entry before it, with the host's
    /// messages after that entry sent again, or, with no entry before it,
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
                "the Claude Code session can't be cut back to message {index}. Its transcript entry for message {} is not known (recorded before entries were, or before a compaction)",
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
                self.claude_code_directory = None;
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
    use std::path::PathBuf;

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

    /// A conversation of two turns and a runtime note. The first turn is the
    /// prompt (with environment details) and a reply. The second is a
    /// follow-up with an image, a call, its result, a context update and a
    /// reply.
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
    /// a follow-up. The usage stays, and the context size starts over.
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
    /// host's messages after that entry go again. Without the model's
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

    /// A whole branch of a task stopped while its last turn's calls ran
    /// answers each of them: with the result the task holds, as a call a
    /// restarted task settles, or as one that never ran.
    #[test]
    fn a_branch_answers_the_calls_its_last_turn_left_open() {
        let mut task = task();
        task.agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
        let call = |id: &str| -> ToolCall {
            serde_json::from_value(serde_json::json!({"id": id, "name": "lookup", "arguments": {}}))
                .unwrap()
        };
        task.conversation.add_user_message("Look it up.");
        let mut turn = Message::new_assistant("Looking.");
        turn.tool_calls = vec![call("held"), call("running"), call("unreached")];
        task.conversation.messages.push(turn);
        task.pending_native_tool_results
            .push(ToolResult::from_tool_call(&call("held"), "found it"));
        task.running_tool_calls.push(crate::task::RunningToolCall {
            tool_call_id: Uuid::new_v4(),
            call_id: Some("running".into()),
            tool_name: "lookup".into(),
            parameters: Default::default(),
            custom_output: None,
            message_to_agent: None,
            built_in: false,
            output: None,
            output_images: Vec::new(),
        });
        task.set_state(TaskState::InProgress);
        let branch = task.branch(2).unwrap();
        assert_eq!(branch.conversation.messages.len(), 3);
        let results: Vec<(String, String)> = branch.conversation.messages[2]
            .content
            .iter()
            .filter_map(Content::as_tool_result)
            .map(|result| (result.call_id.clone().unwrap(), result.output.clone()))
            .collect();
        assert_eq!(
            results,
            [
                ("held".to_string(), "found it".to_string()),
                ("running".to_string(), RESTARTED_CALL.to_string()),
                ("unreached".to_string(), UNREACHED_CALL.to_string()),
            ]
        );
        assert!(branch.pending_native_tool_results.is_empty());
        assert!(branch.running_tool_calls.is_empty());
        // The original is as it was.
        assert_eq!(task.conversation.messages.len(), 2);
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

        // At the first message: no conversation, and the rest kept.
        task.handovers.push(crate::handover::HandoverRecord {
            at: chrono::Utc::now(),
            input_tokens: 500,
            replaced_messages: 4,
            report: "The trip so far.".into(),
        });
        let first = task.branch(0).unwrap();
        assert!(first.conversation.messages.is_empty());
        assert!(first.conversation.system_message.is_none());
        assert!(
            matches!(&first.current_state, TaskState::Completed { summary } if summary.is_empty())
        );
        assert_eq!(first.handovers, task.handovers);
        assert_eq!(first.claude_code_session, None);
        assert_eq!(first.claude_code_fork_of, None);
        assert_eq!(first.claude_code_resume_at, None);
        // Cut back to its first message, it starts over.
        let mut again = first.clone();
        again.cut_back(0).unwrap();
        assert!(matches!(again.current_state, TaskState::Pending));
        assert!(again.cut_back(1).is_err());
    }

    /// A task moves to another directory with its prompt written again. A
    /// Claude Code session stays in the directory it started in, and gets
    /// the new prompt. Moving back there needs no other directory.
    #[test]
    fn a_moved_task_keeps_its_claude_code_session_where_it_started() {
        let mut plain = task();
        plain.conversation.system_message = Some("prompt".into());
        plain.move_to("/elsewhere".into());
        assert_eq!(plain.current_working_directory, PathBuf::from("/elsewhere"));
        assert!(plain.conversation.system_message.is_none());
        assert_eq!(plain.claude_code_directory, None);

        let mut session = task();
        session.claude_code_session = Some("s1".into());
        session.move_to("/elsewhere".into());
        assert_eq!(session.claude_code_directory, Some(PathBuf::from("/w")));
        assert!(session.claude_code_prompt_changed);
        session.move_to("/third".into());
        assert_eq!(session.claude_code_directory, Some(PathBuf::from("/w")));
        session.move_to("/w".into());
        assert_eq!(session.claude_code_directory, None);
    }
}
