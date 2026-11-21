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
///   Symbolic links are not followed here, so `link/..` is the working
///   directory. [`is_path_in_workspace`] resolves links.
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

/// Whether the path a tool would open is inside the task's workspace, when
/// the check runs.
///
/// The workspace and the path are compared in canonical form, with symbolic
/// links resolved (on Windows also junctions, case and short names). So a
/// link to the workspace, its canonical form and the form it was given in
/// all get the same answer. For a path that doesn't exist yet, its deepest
/// existing parent is resolved. A trailing separator or a final `.` doesn't
/// hide a link, so `link/` is judged as `link`.
///
/// Tools don't all read a path alike. [`resolve_path`] trims it and resolves
/// `..` by name, while a program may take it untrimmed, or leave `..` to the
/// operating system, which follows a link before going up. Each of these
/// readings is judged, and the path is inside only when all of them are.
///
/// It fails, so the call is refused under every path policy, for a path that
/// enters the workspace and leaves it through a link (`link/x` or
/// `link/../x`, where `link` points outside), for one that some readings
/// find inside and others outside, and for one through a link whose target
/// is missing.
///
/// The check holds nothing open, so a link created or changed between the
/// check and the tool's use of the path isn't seen.
///
/// A path that starts with `~` (`~` alone, or `~/`, and on Windows `~\` too)
/// is always outside. [`resolve_path`] takes it as a directory name, but
/// Claude Code's search tools and programs a tool starts may expand it to the
/// home directory.
pub fn is_path_in_workspace(path: &str, workspace: &Path) -> Result<bool, ArkoAgentsError> {
    Ok(path_in_workspace(path, workspace)?.is_some())
}

/// The part of `path` below `workspace`, when [`is_path_in_workspace`] finds
/// it inside (empty for the workspace itself), else `None`. It is the rest of
/// the path as [`resolve_path`] spells it, after the point where the path
/// enters the workspace, so a link below that point keeps its name. It holds
/// no `..`.
pub(crate) fn path_in_workspace(
    path: &str,
    workspace: &Path,
) -> Result<Option<PathBuf>, ArkoAgentsError> {
    const ESCAPE: &str = "the path follows a symbolic link outside the workspace";
    const AMBIGUOUS: &str =
        "the path is inside the workspace or outside it, depending on how a tool reads it";
    if starts_with_home(path) {
        return Ok(None);
    }
    let workspace = absolute(workspace)?;
    // Each pair is the path with `..` resolved by name, then with `..` left
    // to the operating system. Arko's file tools trim the path and resolve
    // `..` by name (`resolve_path`). A program may do neither.
    let opened = resolve_path(path, &workspace)?;
    let untrimmed = absolute(&workspace.join(path))?;
    let readings = [
        (opened, absolute(&workspace.join(path.trim()))?),
        (normalize_path(&untrimmed), untrimmed),
    ];
    let root = effective_path(&workspace)?;
    // Every reading is resolved, so a link that fails to resolve in any of
    // them refuses the path.
    let mut inside = [[false; 2]; 2];
    for ((by_name, by_system), inside) in readings.iter().zip(&mut inside) {
        *inside = [
            effective_path(by_name)?.starts_with(&root),
            effective_path(by_system)?.starts_with(&root),
        ];
    }
    if inside.iter().flatten().all(|inside| *inside) {
        return entry(&readings[0].0, &root)?
            .map(Some)
            .ok_or_else(|| refusal(AMBIGUOUS));
    }
    // A reading by name that enters the workspace has left it through a
    // link when it, or the same path read by the operating system, ends
    // outside.
    for ((by_name, _), inside) in readings.iter().zip(inside) {
        if inside != [true, true] && entry(by_name, &root)?.is_some() {
            return Err(refusal(ESCAPE));
        }
    }
    if inside.iter().flatten().any(|inside| *inside) {
        return Err(refusal(AMBIGUOUS));
    }
    Ok(None)
}

fn refusal(reason: &'static str) -> ArkoAgentsError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, reason).into()
}

/// Where the absolute, `..`-free `path` enters the workspace whose canonical
/// form is `root`, as a path below `root`. The first ancestor of `path`, from
/// the file system's root down, that resolves inside `root` gives its place
/// there, and the rest of `path` follows as written. `None` when no ancestor
/// resolves inside. After that point the path can leave the workspace only
/// through a link.
fn entry(path: &Path, root: &Path) -> std::io::Result<Option<PathBuf>> {
    let ancestors: Vec<&Path> = path.ancestors().collect();
    for ancestor in ancestors.into_iter().rev() {
        let resolved = effective_path(ancestor)?;
        let Ok(place) = resolved.strip_prefix(root) else {
            continue;
        };
        let mut below = place.to_path_buf();
        below.extend(path.strip_prefix(ancestor).unwrap_or(Path::new("")));
        return Ok(Some(below));
    }
    Ok(None)
}

/// `path` as the file system resolves it, with every symbolic link followed.
/// For a path that doesn't exist yet, its deepest existing parent is
/// resolved and the missing names are appended. When those hold `..`, the
/// result is resolved again, as for a tool that creates the missing
/// directories first.
///
/// The probes use the path rebuilt from its components, so they look up the
/// names that `Path::parent` and `Path::components` give. With a trailing
/// separator, the file system follows a final link even for
/// `symlink_metadata`, and a dangling link written `link/` would look like a
/// missing name. A name that exists only as a link whose target is missing
/// is refused.
fn effective_path(path: &Path) -> std::io::Result<PathBuf> {
    use std::io::ErrorKind;
    let path: PathBuf = path.components().collect();
    let mut prefix = path.as_path();
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
                    Ok(_) => {
                        return Err(std::io::Error::new(
                            ErrorKind::InvalidInput,
                            "the path goes through a symbolic link whose target doesn't exist",
                        ));
                    }
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

    #[cfg(unix)]
    #[derive(Debug, PartialEq)]
    enum Verdict {
        Inside(PathBuf),
        Outside,
        Refused(String),
    }

    /// A workspace `real` with directories `sub` and `sub/deeper`, a link
    /// `alias` to it, and a directory `outside` beside it holding `deep`.
    #[cfg(unix)]
    struct Linked {
        root: PathBuf,
    }

    #[cfg(unix)]
    impl Linked {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("arko-linked-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(root.join("real/sub/deeper")).unwrap();
            std::fs::create_dir_all(root.join("outside/deep")).unwrap();
            std::os::unix::fs::symlink("real", root.join("alias")).unwrap();
            Linked { root }
        }

        /// A link at `name` in the workspace, to `target` as written.
        fn link(&self, name: &str, target: &str) {
            std::os::unix::fs::symlink(target, self.root.join("real").join(name)).unwrap();
        }

        /// The verdict on `path`, relative to the workspace. It must be the
        /// same with the workspace given as `real` and as `alias`, and with
        /// `path` given relative and absolute through either of them.
        fn verdict(&self, path: &str) -> Verdict {
            let spellings = [self.root.join("real"), self.root.join("alias")];
            let mut verdicts = Vec::new();
            for workspace in &spellings {
                let mut given = vec![path.to_string()];
                for base in &spellings {
                    given.push(format!("{}/{path}", base.display()));
                }
                for path in given {
                    let verdict = match path_in_workspace(&path, workspace) {
                        Ok(Some(below)) => Verdict::Inside(below),
                        Ok(None) => Verdict::Outside,
                        Err(error) => Verdict::Refused(error.to_string()),
                    };
                    verdicts.push((path, workspace.clone(), verdict));
                }
            }
            for (given, workspace, verdict) in &verdicts {
                assert_eq!(
                    verdict,
                    &verdicts[0].2,
                    "{given} in {}",
                    workspace.display()
                );
            }
            verdicts.remove(0).2
        }

        fn refused(&self, path: &str) -> String {
            match self.verdict(path) {
                Verdict::Refused(reason) => reason,
                other => panic!("{path}: {other:?}"),
            }
        }

        fn inside(&self, path: &str) -> PathBuf {
            match self.verdict(path) {
                Verdict::Inside(below) => below,
                other => panic!("{path}: {other:?}"),
            }
        }
    }

    #[cfg(unix)]
    impl Drop for Linked {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// With a trailing `/` the file system follows a final link even for
    /// `symlink_metadata`, so a dangling link looked like a missing name. A
    /// tool that drops the `/` then writes through the link.
    #[cfg(unix)]
    #[test]
    fn a_trailing_separator_does_not_hide_a_dangling_link() {
        let linked = Linked::new();
        linked.link("dangling", "../outside/new-file");
        linked.link("chained", "dangling");
        linked.link("inner", "sub/missing");
        for path in [
            "dangling",
            "dangling/",
            "dangling//",
            "dangling/.",
            "dangling/./",
            "./dangling/",
            "sub/../dangling/",
            "chained/",
            "dangling/new/",
            "inner/",
        ] {
            let reason = linked.refused(path);
            assert!(reason.contains("target doesn't exist"), "{path}: {reason}");
        }
        assert!(!linked.root.join("outside/new-file").exists());
    }

    /// A link out of the workspace, to a directory or a file, is refused
    /// whether the path names the workspace through a link to it or by its
    /// canonical form, and whichever of them the workspace was given as.
    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_workspace_is_refused_in_every_spelling_of_it() {
        let linked = Linked::new();
        std::fs::write(linked.root.join("outside/secret.txt"), "").unwrap();
        std::fs::write(linked.root.join("real/sub/notes.txt"), "").unwrap();
        linked.link("out", "../outside/deep");
        linked.link("in", "sub");
        linked.link("secret", "../outside/secret.txt");
        linked.link("notes", "sub/notes.txt");
        for path in [
            "out",
            "out/",
            "out/new",
            "out/../new",
            "in/../out/x",
            "secret",
            "secret/",
            "in/../secret",
        ] {
            let reason = linked.refused(path);
            assert!(reason.contains("symbolic link outside"), "{path}: {reason}");
        }
        assert_eq!(linked.verdict("../outside/x"), Verdict::Outside);
        assert_eq!(linked.verdict("../outside/deep/x"), Verdict::Outside);
        assert_eq!(linked.verdict("../outside/secret.txt"), Verdict::Outside);
        assert_eq!(linked.inside("."), PathBuf::new());
        assert_eq!(linked.inside("in/x"), PathBuf::from("in/x"));
        assert_eq!(linked.inside("notes"), PathBuf::from("notes"));
    }

    /// Links that lead to links, and relative targets that go up with `..`,
    /// are followed to their end before the path's own `..` applies.
    #[cfg(unix)]
    #[test]
    fn link_chains_and_relative_links_with_parent_components() {
        let linked = Linked::new();
        linked.link("b", "../outside/deep");
        linked.link("a", "b");
        linked.link("c", "a");
        linked.link("up", "..");
        linked.link("here", ".");
        linked.link("back", "../real/sub/deeper");
        linked.link("d1", "d2");
        linked.link("d2", "../outside/missing");
        // `c/../x` is `real/x` by name, and `outside/x` by the file system.
        // `here/../x` is `real/x` by name, and `x` beside `real` by the file
        // system.
        for path in ["c", "c/x", "c/", "c/../x", "up/outside/x", "here/../x"] {
            let reason = linked.refused(path);
            assert!(reason.contains("symbolic link outside"), "{path}: {reason}");
        }
        for path in ["d1", "d1/", "d1/x"] {
            let reason = linked.refused(path);
            assert!(reason.contains("target doesn't exist"), "{path}: {reason}");
        }
        // Outside both by name and by the file system.
        assert_eq!(linked.verdict("a/../../x"), Verdict::Outside);
        assert_eq!(linked.inside("back/new"), PathBuf::from("back/new"));
        // `real/new` by name, and `real/sub/new` by the file system.
        assert_eq!(linked.inside("back/../new"), PathBuf::from("new"));
        assert_eq!(
            linked.inside("up/real/sub/new"),
            PathBuf::from("up/real/sub/new")
        );
    }

    /// A new file below a link to a directory is judged by where the link
    /// leads, and keeps the link's name in the part below the workspace.
    #[cfg(unix)]
    #[test]
    fn a_new_file_under_a_linked_directory() {
        let linked = Linked::new();
        linked.link("in", "sub");
        linked.link("out", "../outside/deep");
        let outside = linked.root.join("outside").display().to_string();
        linked.link("absolute", &outside);
        for path in ["in/new.txt", "in/new/deeper.txt", "in/new/"] {
            assert_eq!(
                linked.inside(path),
                PathBuf::from(path.trim_end_matches('/')),
                "{path}"
            );
        }
        for path in ["out/new.txt", "out/new/deeper.txt", "absolute/new.txt"] {
            let reason = linked.refused(path);
            assert!(reason.contains("symbolic link outside"), "{path}: {reason}");
        }
        // A file below a file is no new file.
        std::fs::write(linked.root.join("real/sub/file"), "").unwrap();
        assert!(matches!(linked.verdict("in/file/new"), Verdict::Refused(_)));
    }

    /// The workspace may be a link, or sit below one. Paths inside, outside
    /// and leaving through a link get the same answers as for its target. A
    /// workspace that is a dangling link refuses every path.
    #[cfg(unix)]
    #[test]
    fn a_workspace_that_is_a_link() {
        use std::os::unix::fs::symlink;
        let linked = Linked::new();
        linked.link("out", "../outside/deep");
        symlink(".", linked.root.join("here")).unwrap();
        let below_link = linked.root.join("here/alias");
        let canonical = std::fs::canonicalize(linked.root.join("real")).unwrap();
        for (path, expected) in [
            ("sub/x".to_string(), Some(PathBuf::from("sub/x"))),
            (
                format!("{}/sub/x", canonical.display()),
                Some(PathBuf::from("sub/x")),
            ),
            ("../outside/x".to_string(), None),
            (format!("{}/x", linked.root.join("outside").display()), None),
        ] {
            assert_eq!(
                path_in_workspace(&path, &below_link).unwrap(),
                expected,
                "{path}"
            );
        }
        for path in [
            "out/x".to_string(),
            format!("{}/out/x", canonical.display()),
        ] {
            assert!(path_in_workspace(&path, &below_link).is_err(), "{path}");
        }
        symlink("missing", linked.root.join("gone")).unwrap();
        for path in ["x", "../outside/x"] {
            assert!(
                path_in_workspace(path, &linked.root.join("gone")).is_err(),
                "{path}"
            );
        }
    }

    /// A program that resolves `..` by name without trimming the path reads
    /// `" x/../evil "` as `evil ` in the workspace, while trimming reads
    /// `evil` and leaving `..` to the file system reads `sub/evil `.
    #[cfg(unix)]
    #[test]
    fn every_reading_of_a_path_is_judged() {
        let linked = Linked::new();
        linked.link(" x", "sub/deeper");
        linked.link("evil ", "../outside/deep");
        let workspace = linked.root.join("real");
        for path in [" x/../evil ", "sub/deeper/../../ x/../evil "] {
            let reason = path_in_workspace(path, &workspace).unwrap_err().to_string();
            assert!(reason.contains("symbolic link outside"), "{path}: {reason}");
        }
        // Trimmed it leaves the workspace, untrimmed it names ` ..`.
        let reason = path_in_workspace(" ../outside", &workspace)
            .unwrap_err()
            .to_string();
        assert!(
            reason.contains("depending on how a tool reads it"),
            "{reason}"
        );
        assert_eq!(
            path_in_workspace(" sub/x ", &workspace).unwrap(),
            Some(PathBuf::from("sub/x"))
        );
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

    /// The workspace as written, in its canonical `\\?\` form and in upper
    /// case are one directory. Either separator, a trailing one included,
    /// gives the same part below it.
    #[cfg(windows)]
    #[test]
    fn a_windows_workspace_matches_every_spelling_of_it() {
        let workspace =
            std::env::temp_dir().join(format!("arko-workspace-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        let canonical = std::fs::canonicalize(&workspace).unwrap();
        let upper = PathBuf::from(workspace.to_string_lossy().to_uppercase());
        let spellings = [&workspace, &canonical, &upper];
        let below = PathBuf::from(r"src\lib.rs");
        for written in spellings {
            let mut paths: Vec<String> = [
                r"src\lib.rs",
                "src/lib.rs",
                r"src\lib.rs\",
                "src/lib.rs/",
                r"src\x\..\lib.rs",
            ]
            .map(String::from)
            .into();
            for base in spellings {
                paths.push(format!(r"{}\src\lib.rs", base.display()));
                paths.push(format!(r"{}\src\lib.rs\", base.display()));
            }
            for path in &paths {
                assert_eq!(
                    path_in_workspace(path, written).unwrap(),
                    Some(below.clone()),
                    "{path} in {}",
                    written.display()
                );
            }
            assert_eq!(
                path_in_workspace(".", written).unwrap(),
                Some(PathBuf::new())
            );
            assert_eq!(path_in_workspace(r"..\x", written).unwrap(), None);
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
