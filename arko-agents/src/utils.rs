use crate::error::ArkoAgentsError;
use chrono::{DateTime, Utc};
use std::env;
use std::path::{Component, Path, PathBuf};

/// Gets the current working directory as a string.
pub fn get_current_working_directory_string() -> Result<String, ArkoAgentsError> {
    env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(ArkoAgentsError::Io)
}

/// Gets the operating system name.
pub fn get_operating_system() -> String {
    env::consts::OS.to_string()
}

/// Gets the current datetime in UTC as an RFC3339 string.
pub fn get_current_datetime_utc() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.to_rfc3339()
}

/// Normalizes a path, resolving `.` and `..` components without accessing the filesystem.
/// This is crucial for handling paths that may not exist yet (e.g., for `write_file`).
///
/// Symlinks are not followed, and a `..` with nothing left to remove (at the
/// root, or leading a relative path) is dropped. No shell expansion is done;
/// see [`is_path_in_workspace`] for that.
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Normal(c) => ret.push(c),
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::Prefix(..) => unreachable!(),
        }
    }
    ret
}

/// Checks if a given `path_to_check_str` resolves to a location within the `workspace_path`.
///
/// This function is platform-agnostic and handles various path formats including relative paths,
/// parent traversals (`..`), and shell variables (`~`, `$HOME`, etc.).
///
/// # Arguments
/// * `path_to_check_str`: The path string to validate.
/// * `workspace_path`: The absolute, canonical path of the agent's workspace.
/// * `task_cwd`: The current working directory of the task, used as a base for resolving relative paths.
///
/// # Returns
/// `Ok(true)` if the path is safely within the workspace, `Ok(false)` otherwise.
/// `Err` if shell expansion fails.
pub fn is_path_in_workspace(
    path_to_check_str: &str,
    workspace_path: &Path,
    task_cwd: &Path,
) -> Result<bool, ArkoAgentsError> {
    let expanded_path_str = match shellexpand::full(path_to_check_str) {
        Ok(p) => p.into_owned(),
        Err(e) => {
            return Err(ArkoAgentsError::Configuration(format!(
                "Failed to expand path '{}': {}",
                path_to_check_str, e
            )));
        }
    };
    let path_to_check = Path::new(&expanded_path_str);

    let absolute_path_to_check = if path_to_check.is_absolute() {
        path_to_check.to_path_buf()
    } else {
        task_cwd.join(path_to_check)
    };

    let normalized_path = normalize_path(&absolute_path_to_check);

    Ok(normalized_path.starts_with(workspace_path))
}

/// Generates a file tree string for the given root path.
///
/// # Arguments
/// * `root_path`: The root directory to generate the tree for.
/// * `max_depth`: Optional maximum depth to traverse.
///
/// # Returns
/// A string representation of the file tree.
pub fn get_file_tree(
    root_path: &Path,
    max_depth: Option<usize>,
) -> Result<String, ArkoAgentsError> {
    crate::file_tree::FileTreeSnapshot::capture(root_path, max_depth)
        .map(|snapshot| snapshot.full())
}
