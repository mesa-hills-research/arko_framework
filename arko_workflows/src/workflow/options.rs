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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RunOptions {
    /// The run's working directory, instead of the workflow's workspace: the
    /// directory its shell, terminal and file tools work in and that its path
    /// checks allow (the task's
    /// [`current_working_directory`](arko_agents::Task::current_working_directory)),
    /// and where its skills load from. Saved with the task, so a continued,
    /// resumed or recovered run keeps it. A relative path is resolved against
    /// the workflow's workspace; the directory must exist.
    pub workspace: Option<PathBuf>,
}

impl RunOptions {
    /// Options that change nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs the agent in `workspace`; see [`RunOptions::workspace`].
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
