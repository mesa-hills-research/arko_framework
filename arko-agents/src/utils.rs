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
/// root, or leading a relative path) is dropped, so normalize absolute paths
/// (as [`resolve_path`] does). No shell expansion is done.
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

/// Resolves a tool's path parameter against a task's working directory: the
/// path the built-in file tools (`read_file`, `write_file`, `apply_diff`,
/// `list_files`) open, and the path [`is_path_in_workspace`] checks.
///
/// - Surrounding whitespace is trimmed.
/// - The path is taken literally: `~` and `$VAR` are not expanded (`~/notes`
///   names a directory called `~` in the working directory), and `\`
///   separates components only on Windows.
/// - A relative path is joined to the working directory, which is itself
///   made absolute against the process's current directory when it is
///   relative (such as `./project`).
/// - `.` and `..` are then resolved lexically with [`normalize_path`];
///   symbolic links are not followed.
///
/// Fails when the path contains a NUL byte, or when a relative working
/// directory needs the process's current directory and it is unavailable.
pub fn resolve_path(path: &str, working_directory: &Path) -> std::io::Result<PathBuf> {
    if path.contains('\0') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the path contains a NUL byte",
        ));
    }
    Ok(normalize_path(
        &absolute_working_directory(working_directory)?.join(path.trim()),
    ))
}

fn absolute_working_directory(working_directory: &Path) -> std::io::Result<PathBuf> {
    Ok(if working_directory.is_absolute() {
        working_directory.to_path_buf()
    } else {
        env::current_dir()?.join(working_directory)
    })
}

/// Whether `path` resolves (with [`resolve_path`]) inside `workspace`, the
/// task's working directory. The check is lexical, like the resolution: a
/// symbolic link inside the workspace that points outside it is not
/// detected.
pub fn is_path_in_workspace(path: &str, workspace: &Path) -> Result<bool, ArkoAgentsError> {
    let root = normalize_path(&absolute_working_directory(workspace)?);
    Ok(resolve_path(path, workspace)?.starts_with(root))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn paths_resolve_literally_against_the_working_directory() {
        let workspace = Path::new("/workspace/project");
        let resolve = |path: &str| resolve_path(path, workspace).unwrap();
        assert_eq!(resolve("src/./lib.rs"), workspace.join("src/lib.rs"));
        assert_eq!(resolve(" src/lib.rs\n"), workspace.join("src/lib.rs"));
        assert_eq!(resolve("/etc/hosts"), PathBuf::from("/etc/hosts"));
        assert_eq!(resolve("."), workspace);
        // No shell expansion, and `\` is an ordinary character on Unix.
        assert_eq!(resolve("~/notes"), workspace.join("~/notes"));
        assert_eq!(resolve("$HOME/notes"), workspace.join("$HOME/notes"));
        assert_eq!(resolve(r"a\..\..\x"), workspace.join(r"a\..\..\x"));
        assert!(resolve_path("a\0b", workspace).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn the_workspace_check_judges_the_path_the_tools_open() {
        let workspace = Path::new("/workspace/project");
        for (path, inside) in [
            ("src/lib.rs", true),
            ("src/../../project/lib.rs", true),
            ("../outside", false),
            (" ../outside", false),
            ("/workspace/project-other/x", false),
            // `$PWD`, `$HOME` and `~` are directory names, so two `..`
            // leave the workspace whatever the variables hold.
            ("$PWD/../../outside", false),
            ("$HOME/..", true),
            ("~/../../outside", false),
            ("~/../inside", true),
        ] {
            assert_eq!(
                is_path_in_workspace(path, workspace).unwrap(),
                inside,
                "{path}"
            );
            assert_eq!(
                resolve_path(path, workspace)
                    .unwrap()
                    .starts_with(workspace),
                inside,
                "{path}"
            );
        }
    }

    #[test]
    fn relative_working_directories_are_resolved_against_the_current_directory() {
        let current = env::current_dir().unwrap();
        for workspace in ["./ws", "ws", "ws/"] {
            let workspace = Path::new(workspace);
            assert_eq!(
                resolve_path("a.txt", workspace).unwrap(),
                current.join("ws").join("a.txt")
            );
            assert!(is_path_in_workspace("a.txt", workspace).unwrap());
            assert!(is_path_in_workspace("./sub/../a.txt", workspace).unwrap());
            assert!(!is_path_in_workspace("../a.txt", workspace).unwrap());
            // Lexically, `ws/../../ws/x` would drop the leading `..` and stay
            // in `ws`; it names `../ws/x`, outside.
            assert!(!is_path_in_workspace("../../ws/x", workspace).unwrap());
        }
    }
}
