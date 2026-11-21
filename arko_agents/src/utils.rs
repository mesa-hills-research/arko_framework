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

/// Resolves a tool's path parameter against a task's working directory.
/// Returns the path the built-in file tools (`read_file`, `write_file`, `edit_file`,
/// `apply_diff`, `list_files`) open. Use [`is_path_in_workspace`] for admission.
///
/// - Surrounding whitespace is trimmed.
/// - The path is taken literally, so `~` and `$VAR` are not expanded (`~/notes`
///   names a directory called `~` in the working directory).
/// - A relative path is joined to the working directory, and the result is
///   made absolute with [`std::path::absolute`] against the process's
///   current directory, so a relative working directory (such as
///   `./project`) works.
/// - On Windows, that is the resolution Win32 applies when a file is opened.
///   `\` and `/` both separate components, `.` and `..` are resolved, a
///   drive-relative path (`C:dir`) uses that drive's current directory, and
///   trailing dots and spaces are dropped. A verbatim path (`\\?\C:\...`) is
///   kept as written. On Unix, `\` is an ordinary character.
/// - `.` and `..` are then resolved lexically with [`normalize_path`].
///   Symbolic links are not followed. On macOS, `/var` is a link to
///   `/private/var`, so a path written through one of them is not inside a
///   working directory written through the other.
///
/// Fails when the path contains a NUL byte, or when a relative path needs
/// the process's current directory and it is unavailable.
pub fn resolve_path(path: &str, working_directory: &Path) -> std::io::Result<PathBuf> {
    if path.contains('\0') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the path contains a NUL byte",
        ));
    }
    Ok(normalize_path(&absolute(
        &working_directory.join(path.trim()),
    )?))
}

/// A path [`resolve_path`] returned, relative to `working_directory` when it
/// is inside it (`./sub/file`, or `.` for the directory itself), else as it
/// is. The relative form holds no `..` and starts with `.`, so a program run
/// in the working directory opens the same file and can't read the path as
/// an option. On Windows it also avoids handing programs a verbatim `\\?\`
/// path, which many of them mishandle.
pub(crate) fn relative_to_working_directory(resolved: &Path, working_directory: &Path) -> PathBuf {
    let Ok(root) = absolute(working_directory).map(|root| normalize_path(&root)) else {
        return resolved.to_path_buf();
    };
    match resolved.strip_prefix(&root) {
        Ok(rest) if rest.as_os_str().is_empty() => PathBuf::from("."),
        Ok(rest) => Path::new(".").join(rest),
        Err(_) => resolved.to_path_buf(),
    }
}

/// [`std::path::absolute`], with an empty path meaning the current
/// directory.
fn absolute(path: &Path) -> std::io::Result<PathBuf> {
    if path.as_os_str().is_empty() {
        std::path::absolute(".")
    } else {
        std::path::absolute(path)
    }
}

/// Whether the path the filesystem would open is inside the task's workspace.
/// Resolves symlinks before checking containment. For a new path, the deepest
/// existing parent must resolve. Once the remaining components are known to
/// be missing, check their lexical resolution too.
/// Checks both the path as supplied to a CLI and the one [`resolve_path`]
/// gives a built-in tool, since the tools do not all handle whitespace or
/// `..` alike. The filesystem may still change before the tool opens the path.
///
/// A path that starts with `~` (`~` alone, or `~/`, and on Windows `~\` too)
/// is always outside. [`resolve_path`] takes it as a directory name, but
/// Claude Code's search tools and programs a tool starts may expand it to the
/// home directory.
pub fn is_path_in_workspace(path: &str, workspace: &Path) -> Result<bool, ArkoAgentsError> {
    Ok(path_in_workspace(path, workspace)?.is_some())
}

/// The part of `path` below `workspace`, when [`is_path_in_workspace`] finds
/// it inside (empty for the workspace itself), else `None`. It holds no `..`.
pub(crate) fn path_in_workspace(
    path: &str,
    workspace: &Path,
) -> Result<Option<PathBuf>, ArkoAgentsError> {
    if starts_with_home(path) {
        return Ok(None);
    }
    let normalized = resolve_path(path, workspace)?;
    let supplied = absolute(&workspace.join(path))?;
    let workspace = absolute(workspace)?;
    let root = effective_path(&workspace)?;
    let actual = effective_path(&supplied)?;
    let Ok(relative) = actual.strip_prefix(&root) else {
        if normalized.starts_with(normalize_path(&workspace)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the path follows a symbolic link outside the workspace",
            )
            .into());
        }
        return Ok(None);
    };
    if supplied != normalized && !effective_path(&normalized)?.starts_with(&root) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the path has more than one workspace resolution",
        )
        .into());
    }
    Ok(Some(relative.to_path_buf()))
}

fn effective_path(path: &Path) -> std::io::Result<PathBuf> {
    use std::io::ErrorKind;
    let mut prefix = path;
    let mut missing = Vec::new();
    loop {
        match std::fs::canonicalize(prefix) {
            Ok(mut resolved) => {
                if !missing.is_empty() && !std::fs::metadata(&resolved)?.is_dir() {
                    return Err(ErrorKind::NotADirectory.into());
                }
                let fold = missing.contains(&Component::ParentDir);
                resolved.extend(missing.into_iter().rev());
                if fold {
                    // Some tools normalize paths before opening them. Check
                    // the names exposed by folding this missing suffix too.
                    return effective_path(&normalize_path(&resolved));
                }
                return Ok(resolved);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                // A dangling link occupies the name even when canonicalize
                // can't open its target. Treating it like a missing entry
                // would remove the link from the very check meant to find it.
                match std::fs::symlink_metadata(prefix) {
                    Err(missing_entry) if missing_entry.kind() == ErrorKind::NotFound => {}
                    Err(inaccessible) => return Err(inaccessible),
                    Ok(_) => return Err(error),
                }
                let Some(component @ (Component::Normal(_) | Component::ParentDir)) =
                    prefix.components().next_back()
                else {
                    return Err(error);
                };
                missing.push(component);
                prefix = prefix.parent().ok_or(error)?;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Whether the first component of `path` is `~`.
fn starts_with_home(path: &str) -> bool {
    let path = path.trim();
    path == "~" || path.starts_with("~/") || (cfg!(windows) && path.starts_with("~\\"))
}

/// Generates a file tree string for the given root path.
///
/// # Arguments
/// * `root_path` is the root directory to generate the tree for.
/// * `max_depth` sets the optional maximum depth to traverse.
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

/// Replaces the file at `path` with `bytes` in one step, creating its parent
/// directory. The bytes go to a new temporary file in the same directory,
/// are flushed to disk through that file's own write handle, and the file is
/// then renamed over `path`. A crash or a full disk during the save leaves
/// the previous file whole. (Windows can flush only a handle opened for
/// writing, which is why the flush uses the handle that wrote.) On failure
/// the temporary file is removed.
pub fn write_file_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let (directory, temporary) = temporary_beside(path)?;
    std::fs::create_dir_all(directory)?;
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

/// The directory of `path`, and a new temporary file's path in it.
pub(crate) fn temporary_beside(path: &Path) -> std::io::Result<(&Path, PathBuf)> {
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} does not name a file", path.display()),
        )
    })?;
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut temporary_name = std::ffi::OsString::from(".");
    temporary_name.push(name);
    temporary_name.push(format!(".{}.tmp", uuid::Uuid::new_v4().simple()));
    Ok((directory, directory.join(temporary_name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The save replaces the file rather than rewriting it in place. A
    /// reader of the old file (here a hard link to it) keeps the old
    /// content whole, and no temporary file is left behind.
    #[test]
    fn atomic_writes_replace_the_file_whole() {
        let directory = std::env::temp_dir().join(format!("arko-atomic-{}", uuid::Uuid::new_v4()));
        let path = directory.join("nested").join("task.json");
        write_file_atomically(&path, b"old").unwrap();
        let reader = directory.join("reader.json");
        std::fs::hard_link(&path, &reader).unwrap();
        write_file_atomically(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(std::fs::read(&reader).unwrap(), b"old");
        let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, [std::ffi::OsString::from("task.json")]);

        // A save that fails (the path is a directory) leaves nothing behind.
        let blocked = directory.join("blocked");
        std::fs::create_dir_all(blocked.join("inside")).unwrap();
        assert!(write_file_atomically(&blocked, b"x").is_err());
        let names: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().all(|name| !name.ends_with(".tmp")),
            "{names:?}"
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

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
        // Programs run in the working directory get paths inside it
        // relative to it.
        let relative = |path: &str| relative_to_working_directory(&resolve(path), workspace);
        assert_eq!(relative("src/../lib.rs"), PathBuf::from("./lib.rs"));
        assert_eq!(relative("-n"), PathBuf::from("./-n"));
        assert_eq!(relative("."), PathBuf::from("."));
        assert_eq!(relative("../other"), PathBuf::from("/workspace/other"));
    }

    #[cfg(unix)]
    #[test]
    fn the_workspace_check_follows_links_before_parent_components() {
        use std::os::unix::fs::symlink;
        let root =
            std::env::temp_dir().join(format!("arko-workspace-policy-{}", uuid::Uuid::new_v4()));
        let workspace = root.join("workspace");
        let outside = root.join("outside");
        std::fs::create_dir_all(workspace.join("sub")).unwrap();
        std::fs::create_dir_all(outside.join("deep")).unwrap();
        std::fs::write(outside.join("existing"), "outside").unwrap();
        symlink(outside.join("deep"), workspace.join("out")).unwrap();
        symlink(outside.join("deep"), workspace.join("spaced ")).unwrap();
        symlink(workspace.join("sub"), workspace.join("in")).unwrap();
        symlink(outside.join("missing"), workspace.join("dangling")).unwrap();
        for path in ["out/new", "out/../existing", "out/../new", "spaced "] {
            assert!(is_path_in_workspace(path, &workspace).is_err(), "{path}");
        }
        assert!(!is_path_in_workspace("../outside/more", &workspace).unwrap());
        for path in [
            "in/new",
            "in/../new",
            "sub/nonexistent/file",
            "missing/../file",
        ] {
            assert!(is_path_in_workspace(path, &workspace).unwrap(), "{path}");
        }
        for path in ["dangling", "dangling/sub/new", "missing/../dangling"] {
            assert!(is_path_in_workspace(path, &workspace).is_err(), "{path}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    /// `~` as the first component may be the home directory to Claude Code's
    /// search tools and to programs, so it is outside though `resolve_path`
    /// takes it as a directory name. Elsewhere it is an ordinary name.
    #[test]
    fn a_path_starting_with_tilde_is_outside_the_workspace() {
        let workspace = std::env::temp_dir().join("arko-tilde-workspace");
        let mut outside = vec!["~", " ~\n", "~/notes", "~/../notes"];
        if cfg!(windows) {
            outside.push(r"~\notes");
        }
        for path in outside {
            assert!(!is_path_in_workspace(path, &workspace).unwrap(), "{path}");
            assert!(
                resolve_path(path, &workspace)
                    .unwrap()
                    .starts_with(&workspace),
                "{path}"
            );
        }
        for path in ["~user/notes", "~notes", "src/~/notes", "src/~"] {
            assert!(is_path_in_workspace(path, &workspace).unwrap(), "{path}");
        }
    }

    /// The part below the workspace follows the rule of `is_path_in_workspace`.
    #[test]
    fn the_part_of_a_path_below_the_workspace() {
        let workspace =
            std::env::temp_dir().join(format!("arko-below-workspace-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        let below = |path: &str| path_in_workspace(path, &workspace).unwrap();
        assert_eq!(
            below("src/./lib.rs"),
            Some(PathBuf::from("src").join("lib.rs"))
        );
        let absolute = workspace.join("src").join("..").join("a.txt");
        assert_eq!(
            below(&absolute.to_string_lossy()),
            Some(PathBuf::from("a.txt"))
        );
        assert_eq!(below("."), Some(PathBuf::new()));
        for outside in ["../a.txt", "src/../../a.txt", "~/a.txt"] {
            assert_eq!(below(outside), None, "{outside}");
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn a_windows_workspace_matches_its_canonical_spelling() {
        let workspace =
            std::env::temp_dir().join(format!("arko-workspace-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        let canonical = std::fs::canonicalize(&workspace).unwrap();
        for path in [workspace.join(r"src\lib.rs"), canonical.join(r"src\lib.rs")] {
            assert_eq!(
                path_in_workspace(path.to_str().unwrap(), &workspace).unwrap(),
                Some(PathBuf::from(r"src\lib.rs"))
            );
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_resolve_as_windows_opens_them() {
        let workspace = Path::new(r"C:\Work\Project");
        let resolve = |path: &str| resolve_path(path, workspace).unwrap();
        let file = PathBuf::from(r"C:\Work\Project\src\lib.rs");
        // `\` and `/` both separate components, mixed too.
        assert_eq!(resolve(r"src\lib.rs"), file);
        assert_eq!(resolve("src/lib.rs"), file);
        assert_eq!(resolve(r"src/sub\..\lib.rs"), file);
        assert_eq!(resolve(r" C:\Work\Project\src\lib.rs "), file);
        // `..` above the drive root stays at the root.
        assert_eq!(resolve(r"..\..\..\..\x"), PathBuf::from(r"C:\x"));
        // A root-relative path is on the working directory's drive.
        assert_eq!(resolve(r"\x"), PathBuf::from(r"C:\x"));
        // Win32 drops a trailing dot, as when the file is opened.
        assert_eq!(resolve(r"src\lib.rs."), file);
        // Programs run in the working directory get paths inside it
        // relative to it, never in verbatim form.
        let relative = |path: &str| relative_to_working_directory(&resolve(path), workspace);
        assert_eq!(relative("src/lib.rs"), PathBuf::from(r".\src\lib.rs"));
        assert_eq!(relative(r"..\x"), PathBuf::from(r"C:\Work\x"));
        let verbatim = Path::new(r"\\?\C:\Work\Project");
        assert_eq!(
            relative_to_working_directory(&resolve_path("src/lib.rs", verbatim).unwrap(), verbatim),
            PathBuf::from(r".\src\lib.rs")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_unc_and_drive_relative_working_directories() {
        let share = Path::new(r"\\Server\Share\work");
        assert_eq!(
            resolve_path("x", share).unwrap(),
            PathBuf::from(r"\\Server\Share\work\x")
        );
        // A drive-relative working directory uses that drive's current
        // directory, as Windows does.
        let drive_relative = Path::new("C:ws");
        let absolute = std::path::absolute(drive_relative).unwrap();
        assert!(absolute.is_absolute());
        assert_eq!(
            resolve_path("x", drive_relative).unwrap(),
            absolute.join("x")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_aliases_resolve_to_the_same_workspace() {
        // `/var`, where temporary directories live, links to `/private/var`.
        assert_eq!(
            std::fs::canonicalize("/var").unwrap(),
            PathBuf::from("/private/var")
        );
        let written = Path::new("/var/folders");
        let canonical = Path::new("/private/var/folders");
        assert!(is_path_in_workspace("x", written).unwrap());
        assert!(is_path_in_workspace("/var/folders/x", written).unwrap());
        assert!(is_path_in_workspace("/private/var/folders/x", written).unwrap());
        assert!(is_path_in_workspace("/var/folders/x", canonical).unwrap());
        assert!(is_path_in_workspace("/private/var/folders/x", canonical).unwrap());
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
            assert!(!is_path_in_workspace("../../ws/x", workspace).unwrap());
        }
    }
}
