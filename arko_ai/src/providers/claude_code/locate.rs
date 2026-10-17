//! Finds the program to run for `claude`.
//!
//! On Windows, npm installs Claude Code as a `claude.cmd` launcher script.
//! `std::process::Command` searches `PATH` only for `.exe` files, and a
//! launcher script would pass every argument through `cmd.exe`, whose quoting
//! breaks JSON and multi-line prompts. So the script is followed to the
//! package entry next to it, the native `bin/claude.exe` of newer releases,
//! or `cli.js` (run with `node`) of older ones. Native installs are an `.exe`
//! already. Elsewhere the name is used as given.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Launcher-script extensions that can't be run without a shell.
const LAUNCHER_EXTENSIONS: [&str; 3] = ["cmd", "bat", "ps1"];

/// Package entries relative to the directory of npm's launcher script.
const PACKAGE_ENTRIES: [&[&str]; 2] = [
    &[
        "node_modules",
        "@anthropic-ai",
        "claude-code",
        "bin",
        "claude.exe",
    ],
    &["node_modules", "@anthropic-ai", "claude-code", "cli.js"],
];

/// A program and the arguments that come before the client's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Launch {
    pub program: PathBuf,
    pub prefix_args: Vec<String>,
}

/// What `resolve` needs from the machine, so it can be tested anywhere.
pub(super) struct Environment<'a> {
    pub windows: bool,
    pub path: Option<&'a OsStr>,
    pub path_extensions: Option<&'a str>,
    pub is_file: &'a dyn Fn(&Path) -> bool,
}

impl Environment<'_> {
    fn with_host<T>(f: impl FnOnce(&Environment) -> T) -> T {
        let path = std::env::var_os("PATH");
        let path_extensions = std::env::var("PATHEXT").ok();
        let is_file = |path: &Path| path.is_file();
        f(&Environment {
            windows: cfg!(windows),
            path: path.as_deref(),
            path_extensions: path_extensions.as_deref(),
            is_file: &is_file,
        })
    }
}

/// The launch for `program` (default `claude`) on this machine.
pub(super) fn launch(program: &Path) -> Launch {
    Environment::with_host(|environment| resolve(program, environment))
}

pub(super) fn resolve(program: &Path, environment: &Environment) -> Launch {
    let direct = Launch {
        program: program.to_path_buf(),
        prefix_args: Vec::new(),
    };
    if !environment.windows {
        return direct;
    }
    let Some(found) = find_on_path(program, environment) else {
        return direct;
    };
    let is_launcher = found
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            LAUNCHER_EXTENSIONS
                .iter()
                .any(|launcher| extension.eq_ignore_ascii_case(launcher))
        });
    if !is_launcher {
        return Launch {
            program: found,
            prefix_args: Vec::new(),
        };
    }
    let directory = found.parent().unwrap_or(Path::new(""));
    for entry in PACKAGE_ENTRIES {
        let candidate = entry
            .iter()
            .fold(directory.to_path_buf(), |path, part| path.join(part));
        if (environment.is_file)(&candidate) {
            return if candidate.extension() == Some(OsStr::new("js")) {
                Launch {
                    program: PathBuf::from("node"),
                    prefix_args: vec![candidate.display().to_string()],
                }
            } else {
                Launch {
                    program: candidate,
                    prefix_args: Vec::new(),
                }
            };
        }
    }
    log::warn!(
        "{} is a launcher script with no Claude Code package next to it. Running it directly",
        found.display()
    );
    direct
}

/// The file `program` names, looked up in `PATH` with Windows `PATHEXT`
/// extensions when it has no directory part.
fn find_on_path(program: &Path, environment: &Environment) -> Option<PathBuf> {
    let extensions: Vec<String> = environment
        .path_extensions
        .unwrap_or(".COM;.EXE;.BAT;.CMD")
        .split(';')
        .filter(|extension| !extension.is_empty())
        .map(str::to_ascii_lowercase)
        .chain([".ps1".to_string()])
        .collect();
    let candidates = |base: &Path| -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        if base.extension().is_some() {
            candidates.push(base.to_path_buf());
        }
        for extension in &extensions {
            let mut name = base.as_os_str().to_os_string();
            name.push(extension);
            candidates.push(PathBuf::from(name));
        }
        candidates
    };
    let has_directory = program
        .parent()
        .is_some_and(|parent| !parent.as_os_str().is_empty());
    if has_directory || program.is_absolute() {
        return candidates(program)
            .into_iter()
            .find(|candidate| (environment.is_file)(candidate));
    }
    let path = environment.path?;
    // `PATH` uses `;` on Windows; split by hand so tests can build one on
    // any host.
    path.to_string_lossy()
        .split(';')
        .filter(|directory| !directory.is_empty())
        .flat_map(|directory| candidates(&Path::new(directory).join(program)))
        .find(|candidate| (environment.is_file)(candidate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve_with(program: &str, files: &[&str]) -> Launch {
        let files: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
        let is_file = |path: &Path| files.iter().any(|file| file == path);
        let path = OsStr::new(r"C:\Windows\system32;C:\Users\luke\AppData\Roaming\npm");
        resolve(
            Path::new(program),
            &Environment {
                windows: true,
                path: Some(path),
                path_extensions: Some(".COM;.EXE;.BAT;.CMD"),
                is_file: &is_file,
            },
        )
    }

    // Paths are joined with the host's separator, so the fake files are
    // built the same way.
    fn npm(parts: &[&str]) -> String {
        parts
            .iter()
            .fold(
                PathBuf::from(r"C:\Users\luke\AppData\Roaming\npm"),
                |path, part| path.join(part),
            )
            .display()
            .to_string()
    }

    #[test]
    fn an_npm_launcher_resolves_to_the_native_binary() {
        let shim = npm(&["claude.cmd"]);
        let binary = npm(&[
            "node_modules",
            "@anthropic-ai",
            "claude-code",
            "bin",
            "claude.exe",
        ]);
        let launch = resolve_with("claude", &[&shim, &binary]);
        assert_eq!(launch.program, PathBuf::from(&binary));
        assert!(launch.prefix_args.is_empty());
    }

    #[test]
    fn an_older_npm_package_runs_its_script_with_node() {
        let shim = npm(&["claude.cmd"]);
        let script = npm(&["node_modules", "@anthropic-ai", "claude-code", "cli.js"]);
        let launch = resolve_with("claude", &[&shim, &script]);
        assert_eq!(launch.program, PathBuf::from("node"));
        assert_eq!(launch.prefix_args, vec![script]);
    }

    #[test]
    fn a_native_install_is_used_as_found() {
        let native = npm(&["claude.exe"]);
        let launch = resolve_with("claude", &[&native]);
        assert_eq!(launch.program, PathBuf::from(&native));
    }

    #[test]
    fn an_explicit_launcher_path_is_followed_too() {
        let shim = npm(&["claude.cmd"]);
        let binary = npm(&[
            "node_modules",
            "@anthropic-ai",
            "claude-code",
            "bin",
            "claude.exe",
        ]);
        let launch = resolve_with(&shim, &[&shim, &binary]);
        assert_eq!(launch.program, PathBuf::from(&binary));
    }

    #[test]
    fn unresolved_names_and_other_platforms_are_left_alone() {
        assert_eq!(resolve_with("claude", &[]).program, PathBuf::from("claude"));
        let unix = resolve(
            Path::new("claude"),
            &Environment {
                windows: false,
                path: None,
                path_extensions: None,
                is_file: &|_| true,
            },
        );
        assert_eq!(unix.program, PathBuf::from("claude"));
    }
}
