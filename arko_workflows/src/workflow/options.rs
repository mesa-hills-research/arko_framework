//! Options for a single agent run.

use std::path::{Path, PathBuf};

use crate::error::WorkflowError;

/// Options for one agent run, for [`Workflow::run_agent_with`] and
/// [`Workflow::spawn_agent_with`]. `RunOptions::default()` runs as
/// [`Workflow::run_agent`] does.
///
/// [`Workflow::run_agent_with`]: crate::Workflow::run_agent_with
/// [`Workflow::spawn_agent_with`]: crate::Workflow::spawn_agent_with
/// [`Workflow::run_agent`]: crate::Workflow::run_agent
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RunOptions {
    /// The run's working directory, instead of the workflow's workspace. The
    /// directory its shell, terminal and file tools work in and that its path
    /// checks allow (the task's
    /// [`current_working_directory`](arko_agents::Task::current_working_directory)),
    /// and where its skills load from. Saved with the task, so a continued,
    /// resumed or recovered run keeps it. A relative path is resolved against
    /// the workflow's workspace. The directory must exist.
    pub workspace: Option<PathBuf>,
    /// Images that go with the prompt, after its text in the run's first
    /// message (the task's
    /// [`initial_task_images`](arko_agents::Task::initial_task_images)). A
    /// fresh conversation after a handover starts with them again. An image
    /// the agent's model can't take is refused (`WorkflowError::BuilderInvalid`
    /// with [`arko_ai::Model::image_refusal`]'s reason), before the run
    /// starts.
    pub images: Vec<arko_ai::Image>,
    /// The configuration the run starts with, in place of its agent's
    /// registered one (see [`RunOptions::config`]).
    pub config: Option<arko_agents::AgentConfig>,
}

// By hand, since a configuration has no `PartialEq` of its own: two are equal
// when they save the same.
impl PartialEq for RunOptions {
    fn eq(&self, other: &Self) -> bool {
        let saved = |config: &Option<arko_agents::AgentConfig>| {
            config
                .as_ref()
                .map(|config| serde_json::to_value(config).ok())
        };
        self.workspace == other.workspace
            && self.images == other.images
            && saved(&self.config) == saved(&other.config)
    }
}

impl Eq for RunOptions {}

impl RunOptions {
    /// Options that change nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts the run with `config` in place of its agent's registered
    /// configuration, without rebuilding the workflow, such as another
    /// model, provider or runtime (an API model, a ChatGPT sign-in, Claude
    /// Code), or other instructions or tools. The workflow adds its tools to `config`
    /// as it does to the registered one, and the agent's name, its notes,
    /// messages, subagents and budget stay the agent's. Images go by
    /// `config`'s model. The task keeps the configuration, so the run is
    /// continued, handed over and resumed from a checkpoint with it. A resume
    /// doesn't compare it with the agent's registered one, as for
    /// [`ContinueOptions::reconfigure`] (the task is marked
    /// [`reconfigured`](arko_agents::Task::reconfigured)). The tools it
    /// lists must be registered for the agent (`resume_only_agent_tool`
    /// counts), or they aren't offered.
    pub fn config(mut self, config: arko_agents::AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Runs the agent in `workspace`. See [`RunOptions::workspace`].
    pub fn workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }

    /// Sends `images` with the prompt. See [`RunOptions::images`].
    pub fn images(mut self, images: impl IntoIterator<Item = arko_ai::Image>) -> Self {
        self.images = images.into_iter().collect();
        self
    }
}

/// Options for continuing an agent run, for
/// [`Workflow::continue_agent_with`] and [`Workflow::restart_with`].
/// `ContinueOptions::default()` continues as [`Workflow::continue_agent`]
/// does.
///
/// [`Workflow::continue_agent_with`]: crate::Workflow::continue_agent_with
/// [`Workflow::restart_with`]: crate::Workflow::restart_with
/// [`Workflow::continue_agent`]: crate::Workflow::continue_agent
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ContinueOptions {
    /// Images that go with the prompt, after its text in the same message.
    /// An image the agent's model can't take is refused
    /// (`WorkflowError::BuilderInvalid` with
    /// [`arko_ai::Model::image_refusal`]'s reason), before the run goes on.
    pub images: Vec<arko_ai::Image>,
    /// The configuration the run continues with, in place of the one saved
    /// in its task (see [`ContinueOptions::reconfigure`]).
    pub config: Option<arko_agents::AgentConfig>,
    /// The message the run's conversation is cut back to first, by its
    /// index (see [`ContinueOptions::cut_back_to`]).
    pub cut_back_to: Option<usize>,
    /// The working directory the run goes on in (see
    /// [`ContinueOptions::workspace`]).
    pub workspace: Option<PathBuf>,
}

impl ContinueOptions {
    /// Options that change nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sends `images` with the prompt. See [`ContinueOptions::images`].
    pub fn images(mut self, images: impl IntoIterator<Item = arko_ai::Image>) -> Self {
        self.images = images.into_iter().collect();
        self
    }

    /// Continues the run with `config` in place of the configuration saved
    /// in its task, such as a switch between a chat (no tools) and an agent
    /// (tools), with the same conversation, its prompt cache key and run ID,
    /// under `config`'s tools and a system message written from `config`.
    /// The workflow adds its tools to `config` as it does for a new run
    /// (none to a chat). The model must stay the same. The new tools and
    /// system message make the next request write the provider's prompt
    /// cache afresh. A Claude Code agent's session resumes with the new
    /// flags and prompt. The task keeps the configuration from then on, also
    /// across a resume from a checkpoint, which doesn't compare it with the
    /// agent's registered one. Only [`Workflow::continue_agent_with`] and
    /// a continued run's [`Workflow::restart_with`] take it
    /// (`BuilderInvalid` otherwise).
    ///
    /// [`Workflow::continue_agent_with`]: crate::Workflow::continue_agent_with
    /// [`Workflow::restart_with`]: crate::Workflow::restart_with
    pub fn reconfigure(mut self, config: arko_agents::AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Cuts the run's conversation back to just before message `index`,
    /// one of its task's [`user_turns`](arko_agents::Task::user_turns), and
    /// sends the prompt in its place, as an edited message or the same one
    /// again (regenerate). The messages from `index` on go, with what waited
    /// with them (calls, results, a question, a pause). See
    /// [`Task::cut_back`](arko_agents::Task::cut_back). Index `0` starts the
    /// conversation over from the prompt, as a new run's (with the
    /// workflow's notes and list of recipients), under the same run ID. A
    /// run that isn't running can be cut back whatever its status, whether
    /// completed, waiting for input, paused, cancelled, failed or
    /// interrupted. The notes the removed messages gave it are given again,
    /// except one-time notes, which went with them. Messages from other
    /// agents in them are not delivered again. A Claude Code agent's session
    /// resumes at the reply before the cut. Only
    /// [`Workflow::continue_agent_with`] takes it.
    ///
    /// [`Workflow::continue_agent_with`]: crate::Workflow::continue_agent_with
    pub fn cut_back_to(mut self, index: usize) -> Self {
        self.cut_back_to = Some(index);
        self
    }

    /// Continues the run in `workspace` instead of its working directory,
    /// such as a chat that becomes an agent working on a folder. Its tools
    /// work there, its path checks allow it, and skills load from it, as
    /// [`RunOptions::workspace`] sets for a new run (resolved the same way,
    /// against the workflow's workspace, and it must exist). A run that
    /// moves gets its system prompt written again, since the prompt may
    /// state the directory. A Claude Code agent's conversation stays in the
    /// directory its session started in (Claude Code finds a session only
    /// from that directory), so moving one with a session is refused
    /// (`BuilderInvalid`). Only [`Workflow::continue_agent_with`] takes it.
    ///
    /// [`Workflow::continue_agent_with`]: crate::Workflow::continue_agent_with
    pub fn workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }
}

/// `workspace` as a run's working directory: resolved against the
/// workflow's `base` when relative, made absolute with `.` and `..` resolved
/// lexically (as the workflow's own workspace is), and required to be an
/// existing directory.
pub(crate) fn run_workspace(base: &Path, workspace: &Path) -> Result<PathBuf, WorkflowError> {
    let joined = if workspace.is_absolute() {
        workspace.to_path_buf()
    } else {
        base.join(workspace)
    };
    let absolute = std::path::absolute(&joined).map_err(|error| {
        WorkflowError::BuilderInvalid(format!(
            "workspace '{}' can't be made absolute: {error}",
            workspace.display()
        ))
    })?;
    let normalized = arko_agents::utils::normalize_path(&absolute);
    if !normalized.is_dir() {
        return Err(WorkflowError::BuilderInvalid(format!(
            "workspace '{}' does not exist or is not a directory",
            normalized.display()
        )));
    }
    Ok(normalized)
}
