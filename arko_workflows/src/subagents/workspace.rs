//! Isolated workspaces for subagents through [`WorkspaceProvider`] and
//! [`GitWorktreeProvider`], which gives each subagent a git worktree.

use std::path::{Path, PathBuf};

use arko_ai::process::NoConsoleWindow;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// What a workspace is created for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceRequest {
    pub workflow_run_id: Uuid,
    /// The subagent's run id and short id (such as `r1`), and its agent.
    pub run_id: Uuid,
    pub handle: String,
    pub agent: String,
    /// The run that starts it.
    pub caller_run_id: Uuid,
    /// The caller's working directory, which the workspace stands in for.
    pub base: PathBuf,
}

/// An isolated workspace a [`WorkspaceProvider`] created. Saved with the
/// subagent's record, so that its later runs work there and
/// [`WorkspaceProvider::remove`] gets it back after a resume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsolatedWorkspace {
    /// The subagent's working directory.
    pub path: PathBuf,
    /// The workspace's top directory, when not `path` (a worktree whose
    /// subdirectory stands in for the caller's directory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<PathBuf>,
    /// The repository it was made from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<PathBuf>,
    /// Its branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// The commit it was made from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_commit: Option<String>,
    /// What the subagent is told about where it works, such as how to hand
    /// its changes back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl IsolatedWorkspace {
    /// A workspace at `path`, with nothing else recorded.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            root: None,
            repository: None,
            branch: None,
            base_commit: None,
            description: None,
        }
    }
}

/// Creates and removes the isolated workspaces of subagents started with
/// `start_agent`'s `isolated`, using a git worktree ([`GitWorktreeProvider`]), a
/// copy, a container's mount. Set with
/// [`SubagentsConfig::workspace_provider`](super::SubagentsConfig::workspace_provider).
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Creates the workspace for `request`'s subagent. The directory it
    /// returns ([`IsolatedWorkspace::path`]) must exist. An error is the
    /// caller's `start_agent` error.
    async fn create(&self, request: &WorkspaceRequest) -> Result<IsolatedWorkspace, String>;

    /// Removes `workspace`, after its subagent stopped (`stop_agent`'s
    /// `remove_workspace`). An error is the caller's `stop_agent` error, and
    /// the subagent keeps its workspace.
    async fn remove(&self, workspace: &IsolatedWorkspace) -> Result<(), String>;
}

/// Gives each subagent a git worktree of the repository its caller works
/// in, with a checkout of the current commit (`HEAD`) at `<folder>/<id>`, on a
/// new branch `<prefix><id>` (default prefix `arko/`), with the subagent
/// working in the same subdirectory its caller works in. `<id>` is the
/// subagent's short id (`r1`), with a suffix (`r1-2`) when an earlier
/// workflow run left that folder or branch. Worktrees are made one at a
/// time (git locks the repository's files). Removing one keeps its branch,
/// and git refuses while it has changes that aren't committed.
#[derive(Debug)]
pub struct GitWorktreeProvider {
    folder: PathBuf,
    branch_prefix: String,
    lock: tokio::sync::Mutex<()>,
}

impl GitWorktreeProvider {
    /// Worktrees in `folder` (created when needed).
    pub fn new(folder: impl Into<PathBuf>) -> Self {
        Self {
            folder: folder.into(),
            branch_prefix: "arko/".to_string(),
            lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Branches named `<prefix><id>`.
    pub fn branch_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.branch_prefix = prefix.into();
        self
    }
}

async fn git(directory: &Path, args: &[&str]) -> Result<String, String> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .no_console_window()
        .output()
        .await
        .map_err(|error| format!("could not run git: {error}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let error = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(if error.is_empty() {
            format!("git {} failed", args.join(" "))
        } else {
            error
        })
    }
}

#[async_trait]
impl WorkspaceProvider for GitWorktreeProvider {
    async fn create(&self, request: &WorkspaceRequest) -> Result<IsolatedWorkspace, String> {
        let base = &request.base;
        let repository = git(base, &["rev-parse", "--show-toplevel"])
            .await
            .map(PathBuf::from)
            .map_err(|_| {
                format!(
                    "an isolated workspace is a git worktree, and {} isn't in a git repository",
                    base.display()
                )
            })?;
        let commit = git(base, &["rev-parse", "--verify", "HEAD"])
            .await
            .map_err(|_| "the repository has no commit to make a worktree from".to_string())?;
        std::fs::create_dir_all(&self.folder)
            .map_err(|error| format!("could not create {}: {error}", self.folder.display()))?;
        let root;
        let branch;
        {
            let _one_at_a_time = self.lock.lock().await;
            // A free name: an earlier workflow run may have left `r1`.
            let mut name = request.handle.clone();
            let mut n = 1;
            loop {
                let taken_branch = git(
                    &repository,
                    &[
                        "rev-parse",
                        "--verify",
                        "--quiet",
                        &format!("refs/heads/{}{name}", self.branch_prefix),
                    ],
                )
                .await
                .is_ok();
                if !taken_branch && !self.folder.join(&name).exists() {
                    break;
                }
                n += 1;
                name = format!("{}-{n}", request.handle);
            }
            root = self.folder.join(&name);
            branch = format!("{}{name}", self.branch_prefix);
            let target = root.to_string_lossy().to_string();
            git(
                &repository,
                &["worktree", "add", "-b", &branch, &target, &commit],
            )
            .await
            .map_err(|error| format!("could not make the worktree: {error}"))?;
        }
        // The caller's place in the repository, in the worktree.
        let relative = std::fs::canonicalize(base)
            .ok()
            .zip(std::fs::canonicalize(&repository).ok())
            .and_then(|(base, repository)| {
                base.strip_prefix(&repository).ok().map(Path::to_path_buf)
            })
            .unwrap_or_default();
        let path = root.join(relative);
        let path = if path.is_dir() { path } else { root.clone() };
        let short = &commit[..commit.len().min(10)];
        let description = format!(
            "You work in `{}`: a git worktree of the repository, on the branch `{branch}`, made from commit {short}. Your lead and the other agents work in other checkouts, so your changes reach them when your branch is merged. Commit your changes on this branch before you report (don't switch branches, rebase or push), and name your commits in your report. Changes not committed in the original checkout aren't in this one.",
            path.display()
        );
        Ok(IsolatedWorkspace {
            root: (path != root).then_some(root),
            path,
            repository: Some(repository),
            branch: Some(branch),
            base_commit: Some(commit),
            description: Some(description),
        })
    }

    async fn remove(&self, workspace: &IsolatedWorkspace) -> Result<(), String> {
        let root = workspace.root.as_ref().unwrap_or(&workspace.path);
        let repository = workspace.repository.as_ref().unwrap_or(root);
        let target = root.to_string_lossy().to_string();
        let _one_at_a_time = self.lock.lock().await;
        git(repository, &["worktree", "remove", &target])
            .await
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "the worktree wasn't removed: {error}. Commit or discard its changes first (in {})",
                    root.display()
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_git(directory: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// A worktree is made at the caller's place in the repository, on its
    /// own branch, and removed again with its branch kept.
    #[tokio::test]
    async fn a_worktree_is_made_at_the_callers_place_and_removed() {
        let workspace = crate::test_support::Workspace::new();
        let repository = workspace.0.join("repo");
        std::fs::create_dir_all(repository.join("sub")).unwrap();
        run_git(&repository, &["init", "-q"]);
        std::fs::write(repository.join("sub/file.txt"), "hello").unwrap();
        run_git(&repository, &["add", "."]);
        run_git(
            &repository,
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-q",
                "-m",
                "first",
            ],
        );
        let provider =
            GitWorktreeProvider::new(workspace.0.join("worktrees")).branch_prefix("test/");
        let request = WorkspaceRequest {
            workflow_run_id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            handle: "r1".to_string(),
            agent: "coder".to_string(),
            caller_run_id: Uuid::new_v4(),
            base: repository.join("sub"),
        };
        let created = provider.create(&request).await.unwrap();
        assert_eq!(created.path, workspace.0.join("worktrees/r1/sub"));
        assert_eq!(created.root, Some(workspace.0.join("worktrees/r1")));
        assert_eq!(created.branch.as_deref(), Some("test/r1"));
        assert!(created.path.join("file.txt").is_file());
        assert!(created.description.as_deref().unwrap().contains("test/r1"));
        provider.remove(&created).await.unwrap();
        assert!(!workspace.0.join("worktrees/r1").exists());
        assert_eq!(
            run_git(&repository, &["branch", "--list", "test/r1"]),
            "test/r1"
        );
        // A later workflow run's `r1` gets a free name: the branch stayed.
        let again = provider.create(&request).await.unwrap();
        assert_eq!(again.root, Some(workspace.0.join("worktrees/r1-2")));
        assert_eq!(again.branch.as_deref(), Some("test/r1-2"));
        provider.remove(&again).await.unwrap();

        // Outside a repository, the start fails with the reason.
        let outside = workspace.0.join("plain");
        std::fs::create_dir_all(&outside).unwrap();
        let error = provider
            .create(&WorkspaceRequest {
                base: outside,
                handle: "r2".to_string(),
                ..request
            })
            .await
            .unwrap_err();
        assert!(error.contains("isn't in a git repository"), "{error}");
    }
}
