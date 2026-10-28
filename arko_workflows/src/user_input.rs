//! Optional human-in-the-loop routing for an agent's `ask_user` tool.

use async_trait::async_trait;

pub use arko_agents::questions::AskedQuestion;

/// Questions an agent asked, for the workflow's [`UserInputHandler`].
#[derive(Debug, Clone)]
pub struct UserInputRequest {
    pub workflow_run_id: uuid::Uuid,
    pub agent_run_id: uuid::Uuid,
    pub agent_name: String,
    /// The questions as text, with their ids, choices and defaults
    /// ([`arko_agents::questions::questions_text`]).
    pub questions: String,
    /// The questions, each with its id in the agent's task (`q1`, `q2`, ...).
    pub asked: Vec<AskedQuestion>,
    /// The agent waits for the answer. Without it (an agent with
    /// [`async_questions`](arko_agents::AgentToolConfig::async_questions)
    /// that passed `wait: false`), the agent goes on, and the answers reach
    /// it later.
    pub wait: bool,
}

#[async_trait]
pub trait UserInputHandler: Send + Sync {
    /// Answers questions the agent waits for. The answer is the agent's
    /// next input. An agent with `async_questions` gets it as it is, so
    /// word it as a note
    /// ([`answer_note`](arko_agents::questions::answer_note)).
    async fn respond(&self, request: UserInputRequest) -> Result<String, String>;

    /// Takes questions the agent asked without waiting for the answer
    /// (`request.wait` is false). It runs in the background, while the agent
    /// goes on. An answer it returns reaches the agent in its mailbox, as a
    /// subagent's report does: at its next turn boundary, or, for a run that
    /// ended, when the host continues it (the workflow sends
    /// [`NewsWaiting`](crate::subagents::SubagentEventKind::NewsWaiting)).
    /// The default answers with [`respond`](Self::respond).
    ///
    /// A host that shows questions and gathers answers over time returns
    /// `Ok(None)` and delivers each answer itself, as a note or the run's
    /// next message, worded with [`arko_agents::questions`]. It then says
    /// which questions are still open in
    /// [`open_questions`](Self::open_questions).
    async fn asked(&self, request: UserInputRequest) -> Result<Option<String>, String> {
        self.respond(request).await.map(Some)
    }

    /// The questions of agent run `agent_run_id` that are still open, which
    /// its fresh conversation restates after a handover (and its
    /// conversation after Claude Code compacted it), or `None` (the default)
    /// to go by the task's own record
    /// ([`QuestionLog`](arko_agents::questions::QuestionLog)), which knows
    /// only of answers given while the agent waited and the questions it
    /// closed. An answer to questions the agent waited for
    /// ([`respond`](Self::respond)) leaves those this returns open, so a
    /// host may answer some of them. An agent in Claude Code waits for an
    /// answer only when the turn that asked ends, and doesn't wait for the
    /// questions this leaves out by then, such as those answered while the
    /// turn went on.
    fn open_questions(&self, agent_run_id: uuid::Uuid) -> Option<Vec<AskedQuestion>> {
        let _ = agent_run_id;
        None
    }

    /// The number of the first of `count` questions agent run
    /// `agent_run_id` (of agent `agent_name`) asks now: they are `q{n}`,
    /// `q{n + 1}`, ... A host that gives out the numbers itself, one ask at
    /// a time, sees every id once across its runs, even runs at the same
    /// time, so an answer can't reach another run's question. `None` (the
    /// default) numbers each task's questions on from its own count. Asked
    /// only for the runs whose questions the handler takes (not a subagent
    /// whose questions go to its lead).
    fn question_numbers(
        &self,
        agent_run_id: uuid::Uuid,
        agent_name: &str,
        count: u32,
    ) -> Option<u32> {
        let _ = (agent_run_id, agent_name, count);
        None
    }

    /// Agent run `agent_run_id` (of agent `agent_name`) asked `asked`
    /// (`wait`: it waits for the answer). Called as its runtime records
    /// them, before the agent goes on and before [`asked`](Self::asked) or
    /// [`respond`](Self::respond) hears of them, so that
    /// [`open_questions`](Self::open_questions) has them for a handover or a
    /// compaction that comes right after. The default does nothing.
    fn questions_recorded(
        &self,
        agent_run_id: uuid::Uuid,
        agent_name: &str,
        asked: &[AskedQuestion],
        wait: bool,
    ) {
        let _ = (agent_run_id, agent_name, asked, wait);
    }
}
