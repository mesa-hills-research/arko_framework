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
    /// The repository it was made from. This is the main repository, also
    /// when the caller worked in a worktree of it.
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
/// time (git locks the repository's files). A caller that works in a
/// worktree of its own (a subagent that starts subagents) gets a worktree
/// made from its checkout, recorded against the main repository, so that
/// either can be removed first. Removing one keeps its branch, and git
/// refuses while it has changes that aren't committed. The git commands it
/// runs ignore the git variables of the process (`GIT_DIR`, `GIT_INDEX_FILE`
/// and the like, which a hook or `git rebase -x` sets), so a host started
/// from one works on the directories it is given.
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

/// The variables by which a process that git started (a hook, or
/// `git rebase -x`) points the git commands under it at its own repository.
/// The provider's git commands drop them, so that they work on the directory
/// they are given.
const GIT_ENVIRONMENT: [&str; 8] = [
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_PREFIX",
    "GIT_NAMESPACE",
];

async fn git(directory: &Path, args: &[&str]) -> Result<String, String> {
    let mut command = tokio::process::Command::new("git");
    command
        .arg("-C")
        .arg(directory)
        .args(args)
        .no_console_window();
    for name in GIT_ENVIRONMENT {
        command.env_remove(name);
    }
    let output = command
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

/// The repository's main working tree (the directory of a bare repository),
/// which `directory` is in, or is a linked worktree of.
async fn main_worktree(directory: &Path) -> Option<PathBuf> {
    let listed = git(directory, &["worktree", "list", "--porcelain"])
        .await
        .ok()?;
    let first = listed.lines().next()?;
    first.strip_prefix("worktree ").map(PathBuf::from)
}

#[async_trait]
impl WorkspaceProvider for GitWorktreeProvider {
    async fn create(&self, request: &WorkspaceRequest) -> Result<IsolatedWorkspace, String> {
        let base = &request.base;
        let top = git(base, &["rev-parse", "--show-toplevel"])
            .await
            .map(PathBuf::from)
            .map_err(|_| {
                format!(
                    "an isolated workspace is a git worktree, and {} isn't in a git repository",
                    base.display()
                )
            })?;
        // The repository is the main one, also when the caller works in a
        // linked worktree (a subagent's own), so that the new worktree can
        // be removed after the caller's.
        let repository = main_worktree(base).await.unwrap_or_else(|| top.clone());
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
        // The caller's place in its checkout, in the worktree.
        let relative = std::fs::canonicalize(base)
            .ok()
            .zip(std::fs::canonicalize(&top).ok())
            .and_then(|(base, top)| base.strip_prefix(&top).ok().map(Path::to_path_buf))
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
        let mut command = std::process::Command::new("git");
        command.arg("-C").arg(directory).args(args);
        for name in GIT_ENVIRONMENT {
            command.env_remove(name);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// Commits what is staged in `directory` as a tester, also when
    /// nothing is.
    fn commit(directory: &Path, message: &str) {
        run_git(
            directory,
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                message,
            ],
        );
    }

    /// A repository with one commit, holding `sub/file.txt`.
    fn repository_with_a_commit(workspace: &crate::test_support::Workspace) -> PathBuf {
        let repository = workspace.0.join("repo");
        std::fs::create_dir_all(repository.join("sub")).unwrap();
        run_git(&repository, &["init", "-q"]);
        std::fs::write(repository.join("sub/file.txt"), "hello").unwrap();
        run_git(&repository, &["add", "."]);
        commit(&repository, "first");
        repository
    }

    fn request_from(base: PathBuf, handle: &str) -> WorkspaceRequest {
        WorkspaceRequest {
            workflow_run_id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            handle: handle.to_string(),
            agent: "coder".to_string(),
            caller_run_id: Uuid::new_v4(),
            base,
        }
    }

    /// What the child process of the next test is told, so that it runs the
    /// part of the test that is for it.
    const CHILD: &str = "ARKO_TEST_GIT_ENVIRONMENT_CHILD";

    /// Sets git's variables on `command` as a hook or `git rebase -x` sets
    /// them, all pointing at the repository `decoy`.
    fn point_git_at(command: &mut std::process::Command, decoy: &Path) {
        let git_dir = decoy.join(".git");
        for name in GIT_ENVIRONMENT {
            let value = match name {
                "GIT_DIR" | "GIT_COMMON_DIR" => git_dir.clone(),
                "GIT_INDEX_FILE" => git_dir.join("index"),
                "GIT_WORK_TREE" => decoy.to_path_buf(),
                "GIT_OBJECT_DIRECTORY" | "GIT_ALTERNATE_OBJECT_DIRECTORIES" => {
                    git_dir.join("objects")
                }
                "GIT_PREFIX" => PathBuf::from("sub/"),
                _ => PathBuf::from("decoy"),
            };
            command.env(name, value);
        }
    }

    /// Everything about `repository` that a git command aimed at it by
    /// mistake would change.
    fn state_of(repository: &Path) -> (Vec<String>, Vec<u8>) {
        let git = |args: &[&str]| run_git(repository, args);
        (
            vec![
                git(&["for-each-ref"]),
                git(&["worktree", "list", "--porcelain"]),
                git(&["rev-parse", "HEAD"]),
                git(&["status", "--porcelain"]),
            ],
            std::fs::read(repository.join(".git/index")).unwrap(),
        )
    }

    /// The git commands the provider and these helpers run work on the
    /// directory they are given, whatever git's variables of the process
    /// say. The process here is a child of the test, started with the
    /// variables pointing at a decoy repository, and the decoy stays as it was.
    #[test]
    fn git_commands_ignore_the_git_variables_of_the_process() {
        assert!(
            GIT_ENVIRONMENT.contains(&"GIT_DIR") && GIT_ENVIRONMENT.contains(&"GIT_INDEX_FILE")
        );
        let workspace = crate::test_support::Workspace::new();
        let decoy = workspace.0.join("decoy");
        std::fs::create_dir_all(&decoy).unwrap();
        run_git(&decoy, &["init", "-q"]);
        std::fs::write(decoy.join("decoy.txt"), "decoy").unwrap();
        run_git(&decoy, &["add", "."]);
        commit(&decoy, "decoy");
        let before = state_of(&decoy);

        let module = module_path!().split_once("::").unwrap().1;
        let test = format!("{module}::the_child_of_the_git_variables_test");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([test.as_str(), "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1");
        point_git_at(&mut child, &decoy);
        let output = child.output().unwrap();
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{report}");
        assert!(
            report.contains("1 passed"),
            "the child test didn't run: {report}"
        );
        assert!(state_of(&decoy) == before, "the decoy repository changed");
    }

    /// The part of the test above that runs with git's variables set, in a
    /// process of its own. Alone it does nothing.
    #[tokio::test]
    async fn the_child_of_the_git_variables_test() {
        if std::env::var_os(CHILD).is_none() {
            return;
        }
        assert!(std::env::var_os("GIT_DIR").is_some());
        let workspace = crate::test_support::Workspace::new();
        let repository = repository_with_a_commit(&workspace);
        let provider =
            GitWorktreeProvider::new(workspace.0.join("worktrees")).branch_prefix("test/");
        let first = provider
            .create(&request_from(repository.join("sub"), "r1"))
            .await
            .unwrap();
        let second = provider
            .create(&request_from(first.path.clone(), "r2"))
            .await
            .unwrap();
        let listed = run_git(&repository, &["worktree", "list", "--porcelain"]);
        assert!(listed.contains("r1") && listed.contains("r2"), "{listed}");
        provider.remove(&first).await.unwrap();
        provider.remove(&second).await.unwrap();
        assert!(!run_git(&repository, &["worktree", "list"]).contains("r1"));
    }

    /// A worktree made from a subagent's own worktree belongs to the main
    /// repository, so it can still be removed once the first is gone, and
    /// it starts from the commit the first has checked out.
    #[tokio::test]
    async fn a_worktree_of_a_worktree_is_removed_after_its_parent() {
        let workspace = crate::test_support::Workspace::new();
        let repository = repository_with_a_commit(&workspace);
        let provider =
            GitWorktreeProvider::new(workspace.0.join("worktrees")).branch_prefix("test/");
        let first = provider
            .create(&request_from(repository.join("sub"), "r1"))
            .await
            .unwrap();
        commit(&first.path, "second");
        let second = provider
            .create(&request_from(first.path.clone(), "r2"))
            .await
            .unwrap();
        assert_eq!(second.root, Some(workspace.0.join("worktrees/r2")));
        assert_eq!(
            second.base_commit.as_deref(),
            Some(run_git(&first.path, &["rev-parse", "HEAD"]).as_str())
        );
        assert!(second.path.join("file.txt").is_file());

        provider.remove(&first).await.unwrap();
        provider.remove(&second).await.unwrap();
        assert!(!workspace.0.join("worktrees/r1").exists());
        assert!(!workspace.0.join("worktrees/r2").exists());
        let main = std::fs::canonicalize(&repository).unwrap();
        for created in [&first, &second] {
            let recorded = created.repository.as_ref().unwrap();
            assert_eq!(std::fs::canonicalize(recorded).unwrap(), main);
        }
    }

    /// A worktree is made at the caller's place in the repository, on its
    /// own branch, and removed again with its branch kept.
    #[tokio::test]
    async fn a_worktree_is_made_at_the_callers_place_and_removed() {
        let workspace = crate::test_support::Workspace::new();
        let repository = repository_with_a_commit(&workspace);
        let provider =
            GitWorktreeProvider::new(workspace.0.join("worktrees")).branch_prefix("test/");
        let request = request_from(repository.join("sub"), "r1");
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
