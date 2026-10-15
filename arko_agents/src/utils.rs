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
/// path the built-in file tools (`read_file`, `write_file`, `edit_file`,
/// `apply_diff`, `list_files`) open, and the path [`is_path_in_workspace`] checks.
///
/// - Surrounding whitespace is trimmed.
/// - The path is taken literally: `~` and `$VAR` are not expanded (`~/notes`
///   names a directory called `~` in the working directory).
/// - A relative path is joined to the working directory, and the result is
///   made absolute with [`std::path::absolute`] against the process's
///   current directory, so a relative working directory (such as
///   `./project`) works.
/// - On Windows, that is the resolution Win32 applies when a file is opened:
///   `\` and `/` both separate components, `.` and `..` are resolved, a
///   drive-relative path (`C:dir`) uses that drive's current directory, and
///   trailing dots and spaces are dropped. A verbatim path (`\\?\C:\...`) is
///   kept as written. On Unix, `\` is an ordinary character.
/// - `.` and `..` are then resolved lexically with [`normalize_path`].
///   Symbolic links are not followed: on macOS, `/var` is a link to
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
/// an option; on Windows it also avoids handing programs a verbatim `\\?\`
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

/// Whether `path` resolves (with [`resolve_path`]) inside `workspace`, the
/// task's working directory, compared component by component. The check is
/// lexical, like the resolution: a symbolic link inside the workspace that
/// points outside it is not detected.
///
/// On Windows, whose file systems usually ignore case, components are
/// compared without regard to ASCII case (`c:\work\x` is inside `C:\Work`),
/// and a verbatim prefix matches its ordinary form (`\\?\C:\` and `C:\`,
/// `\\?\UNC\server\share` and `\\server\share`). A directory with case
/// sensitivity turned on is still compared without regard to case. On other
/// platforms the comparison is exact, also on macOS file systems that
/// ignore case: a path spelled in another case is judged outside.
pub fn is_path_in_workspace(path: &str, workspace: &Path) -> Result<bool, ArkoAgentsError> {
    let root = normalize_path(&absolute(workspace)?);
    let resolved = resolve_path(path, workspace)?;
    let mut components = resolved.components();
    Ok(root.components().all(|root_component| {
        components
            .next()
            .is_some_and(|c| same_component(c, root_component))
    }))
}

#[cfg(not(windows))]
fn same_component(a: Component<'_>, b: Component<'_>) -> bool {
    a == b
}

#[cfg(windows)]
fn same_component(a: Component<'_>, b: Component<'_>) -> bool {
    use std::path::Prefix::{Disk, UNC, VerbatimDisk, VerbatimUNC};
    let same_text = |a: &std::ffi::OsStr, b: &std::ffi::OsStr| {
        a.as_encoded_bytes()
            .eq_ignore_ascii_case(b.as_encoded_bytes())
    };
    match (a, b) {
        (Component::Prefix(a), Component::Prefix(b)) => match (a.kind(), b.kind()) {
            (Disk(a) | VerbatimDisk(a), Disk(b) | VerbatimDisk(b)) => a.eq_ignore_ascii_case(&b),
            (
                UNC(server_a, share_a) | VerbatimUNC(server_a, share_a),
                UNC(server_b, share_b) | VerbatimUNC(server_b, share_b),
            ) => same_text(server_a, server_b) && same_text(share_a, share_b),
            _ => same_text(a.as_os_str(), b.as_os_str()),
        },
        (Component::Normal(a), Component::Normal(b)) => same_text(a, b),
        (a, b) => a == b,
    }
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

/// Replaces the file at `path` with `bytes` in one step, creating its parent
/// directory: the bytes go to a new temporary file in the same directory,
/// are flushed to disk through that file's own write handle, and the file is
/// then renamed over `path`. A crash or a full disk during the save leaves
/// the previous file whole. (Windows can flush only a handle opened for
/// writing, which is why the flush uses the handle that wrote.) On failure
/// the temporary file is removed.
pub fn write_file_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
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
    std::fs::create_dir_all(directory)?;
    let mut temporary_name = std::ffi::OsString::from(".");
    temporary_name.push(name);
    temporary_name.push(format!(".{}.tmp", uuid::Uuid::new_v4().simple()));
    let temporary = directory.join(temporary_name);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The save replaces the file rather than rewriting it in place: a
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

    #[cfg(windows)]
    #[test]
    fn windows_paths_resolve_and_compare_as_windows_opens_them() {
        let workspace = Path::new(r"C:\Work\Project");
        let resolve = |path: &str| resolve_path(path, workspace).unwrap();
        let inside = |path: &str| is_path_in_workspace(path, workspace).unwrap();
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
        for (path, expected) in [
            (r"src\lib.rs", true),
            // Drive letters and names compare without regard to case.
            (r"c:\work\project\x", true),
            (r"C:\WORK\PROJECT\x", true),
            (r"C:/Work/Project/x", true),
            // A verbatim prefix matches its ordinary form.
            (r"\\?\C:\Work\Project\x", true),
            (r"\\?\c:\work\project\x", true),
            (r"C:\Work\Project\..\..\Work\Project\x", true),
            (r"C:\Work\Project2\x", false),
            (r"C:\Work\x", false),
            (r"..\x", false),
            (r"..\..\..\..\x", false),
            (r"\x", false),
            // Another drive, absolute or drive-relative.
            (r"D:\Work\Project\x", false),
            ("D:x", false),
            // Win32 leaves a verbatim path as written; its `..` is resolved
            // lexically and still leaves.
            (r"\\?\C:\Work\Project\..\x", false),
        ] {
            assert_eq!(inside(path), expected, "{path}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_unc_and_drive_relative_working_directories() {
        let share = Path::new(r"\\Server\Share\work");
        for (path, expected) in [
            ("x", true),
            (r"sub/x", true),
            (r"\\server\share\WORK\x", true),
            (r"\\?\UNC\server\share\work\x", true),
            (r"\\server\other\work\x", false),
            (r"\\other\share\work\x", false),
            // The root of a UNC working directory is its share's root.
            (r"\x", false),
            (r"..\x", false),
            (r"..\..\..\x", false),
        ] {
            assert_eq!(
                is_path_in_workspace(path, share).unwrap(),
                expected,
                "{path}"
            );
        }
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
        assert!(is_path_in_workspace("x", drive_relative).unwrap());
        assert!(!is_path_in_workspace(r"..\x", drive_relative).unwrap());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_symlinked_directories_are_compared_as_written() {
        // `/var`, where temporary directories live, links to `/private/var`.
        assert_eq!(
            std::fs::canonicalize("/var").unwrap(),
            PathBuf::from("/private/var")
        );
        let written = Path::new("/var/folders");
        let canonical = Path::new("/private/var/folders");
        assert!(is_path_in_workspace("x", written).unwrap());
        assert!(is_path_in_workspace("/var/folders/x", written).unwrap());
        // Symlinks are not followed: the other spelling is outside.
        assert!(!is_path_in_workspace("/private/var/folders/x", written).unwrap());
        assert!(!is_path_in_workspace("/var/folders/x", canonical).unwrap());
        assert!(is_path_in_workspace("/private/var/folders/x", canonical).unwrap());
        // Names compare exactly, though the file system may ignore case.
        assert!(!is_path_in_workspace("/private/var/FOLDERS/x", canonical).unwrap());
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
