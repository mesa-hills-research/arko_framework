//! Workflow configuration and portable discovery metadata.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedPlatform {
    pub os: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
}

impl SupportedPlatform {
    pub fn new(os: impl Into<String>) -> Self {
        Self {
            os: os.into(),
            arch: None,
        }
    }

    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.arch = Some(arch.into());
        self
    }

    pub fn matches_host(&self) -> bool {
        self.os.eq_ignore_ascii_case(std::env::consts::OS)
            && self
                .arch
                .as_deref()
                .is_none_or(|arch| arch.eq_ignore_ascii_case(std::env::consts::ARCH))
    }
}

/// Something outside the workflow it needs or uses: a program it runs, a
/// service or a login. [`WorkflowMetadata::missing_dependencies`] checks the
/// ones that name an `executable`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowDependency {
    pub name: String,
    pub description: String,
    /// The program to look for: a name found on `PATH` (on Windows, with or
    /// without a `PATHEXT` extension), or a path to a file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_hint: Option<String>,
    /// Only on this OS (`std::env::consts::OS`, any case); any when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    /// Only on this architecture (`std::env::consts::ARCH`, any case); any
    /// when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// The dependency helps but isn't required: the workflow works without
    /// it. `false` (the default): required.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
}

impl WorkflowDependency {
    /// A required dependency on any host, with no executable to check.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            executable: None,
            install_hint: None,
            os: None,
            arch: None,
            optional: false,
        }
    }

    /// The program to look for.
    pub fn executable(mut self, executable: impl Into<String>) -> Self {
        self.executable = Some(executable.into());
        self
    }

    /// How to install it, for the user.
    pub fn install_hint(mut self, hint: impl Into<String>) -> Self {
        self.install_hint = Some(hint.into());
        self
    }

    /// Only on this OS.
    pub fn os(mut self, os: impl Into<String>) -> Self {
        self.os = Some(os.into());
        self
    }

    /// Only on this architecture.
    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.arch = Some(arch.into());
        self
    }

    /// Marks it optional: the workflow works without it.
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Whether it applies on a host with `os` and `arch`.
    fn applies_to(&self, os: &str, arch: &str) -> bool {
        let matches = |wanted: &Option<String>, host: &str| {
            wanted
                .as_deref()
                .is_none_or(|wanted| wanted.eq_ignore_ascii_case(host))
        };
        matches(&self.os, os) && matches(&self.arch, arch)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowMetadata {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub supported_platforms: Vec<SupportedPlatform>,
    /// The workflow's dependencies, required and
    /// [optional](WorkflowDependency::optional).
    #[serde(default)]
    pub required_dependencies: Vec<WorkflowDependency>,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl WorkflowMetadata {
    /// The dependencies this host lacks: those that apply to it (their `os`
    /// and `arch` match `std::env::consts::{OS, ARCH}`, or are unset) whose
    /// `executable` isn't found, required and optional alike. A dependency
    /// without an `executable` is never reported. Nothing is printed or
    /// installed; the workflow decides what to do, for example stop before
    /// any model call when a required one is missing and warn about the
    /// optional ones.
    pub fn missing_dependencies(&self) -> Vec<&WorkflowDependency> {
        let extensions = if cfg!(windows) {
            let extensions = std::env::var("PATHEXT").unwrap_or_default();
            let extensions: Vec<String> = extensions
                .split(';')
                .filter(|extension| !extension.is_empty())
                .map(str::to_string)
                .collect();
            if extensions.is_empty() {
                [".COM", ".EXE", ".BAT", ".CMD"]
                    .map(str::to_string)
                    .to_vec()
            } else {
                extensions
            }
        } else {
            Vec::new()
        };
        self.missing_on(
            std::env::consts::OS,
            std::env::consts::ARCH,
            std::env::var_os("PATH").as_deref(),
            &extensions,
        )
    }

    /// [`missing_dependencies`](Self::missing_dependencies) on a host with
    /// `os`, `arch`, the search path `path`, and executable `extensions`.
    fn missing_on(
        &self,
        os: &str,
        arch: &str,
        path: Option<&OsStr>,
        extensions: &[String],
    ) -> Vec<&WorkflowDependency> {
        self.required_dependencies
            .iter()
            .filter(|dependency| dependency.applies_to(os, arch))
            .filter(|dependency| {
                dependency
                    .executable
                    .as_deref()
                    .is_some_and(|executable| !find_executable(executable, path, extensions))
            })
            .collect()
    }
}

/// Whether `executable` is found: as a file when it is a path, otherwise in
/// a directory of `path`. Each of `extensions` (Windows' `PATHEXT`) is tried
/// after the name as given. On Unix, the file must be executable.
fn find_executable(executable: &str, path: Option<&OsStr>, extensions: &[String]) -> bool {
    let found = |candidate: &Path| {
        std::iter::once(candidate.to_path_buf())
            .chain(extensions.iter().map(|extension| {
                let mut name = candidate.as_os_str().to_owned();
                name.push(extension);
                PathBuf::from(name)
            }))
            .any(|file| is_executable(&file))
    };
    if executable.contains(['/', '\\']) {
        return found(Path::new(executable));
    }
    path.is_some_and(|path| std::env::split_paths(path).any(|dir| found(&dir.join(executable))))
}

fn is_executable(file: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(file) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        metadata.is_file()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub name: String,
    pub workspace: PathBuf,
    #[serde(default)]
    pub metadata: WorkflowMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_optional_dependency_round_trips_and_old_json_is_required() {
        let old: WorkflowDependency = serde_json::from_str(
            r#"{"name": "shell", "description": "Runs checks", "executable": "bash"}"#,
        )
        .unwrap();
        assert_eq!(
            old,
            WorkflowDependency::new("shell", "Runs checks").executable("bash")
        );
        assert!(!old.optional);
        assert!(!serde_json::to_string(&old).unwrap().contains("optional"));

        let search = WorkflowDependency::new("ripgrep", "Fast search")
            .executable("rg")
            .install_hint("apt install ripgrep")
            .os("linux")
            .optional();
        let json = serde_json::to_value(&search).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "name": "ripgrep",
                "description": "Fast search",
                "executable": "rg",
                "install_hint": "apt install ripgrep",
                "os": "linux",
                "optional": true
            })
        );
        assert_eq!(
            serde_json::from_value::<WorkflowDependency>(json).unwrap(),
            search
        );
    }

    #[test]
    fn the_check_reports_missing_executables_that_apply_to_the_host() {
        let dir = std::env::temp_dir().join(format!("arko-dependencies-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["present-tool", "plain-file", "windows-tool.exe"] {
            std::fs::write(dir.join(name), "").unwrap();
        }
        #[cfg(unix)]
        for name in ["present-tool", "windows-tool.exe"] {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(0o755))
                .unwrap();
        }
        let path = std::env::join_paths([dir.join("elsewhere"), dir.clone()]).unwrap();
        let dependency = |name: &str, executable: &str| {
            WorkflowDependency::new(name, "test").executable(executable)
        };
        let metadata = WorkflowMetadata {
            required_dependencies: vec![
                dependency("present", "present-tool"),
                dependency("absent", "absent-tool"),
                dependency("absent but optional", "absent-tool").optional(),
                dependency("absent elsewhere", "absent-tool").os("plan9"),
                dependency("absent on another arch", "absent-tool").arch("riscv64"),
                dependency("on this host", "absent-tool")
                    .os("LINUX")
                    .arch("x86_64"),
                WorkflowDependency::new("a login", "not a program"),
                dependency("by path", &dir.join("present-tool").display().to_string()),
                dependency(
                    "missing by path",
                    &dir.join("absent-tool").display().to_string(),
                ),
                dependency("with an extension", "windows-tool"),
            ],
            ..WorkflowMetadata::default()
        };
        let missing = |extensions: &[String]| {
            metadata
                .missing_on("linux", "x86_64", Some(&path), extensions)
                .into_iter()
                .map(|dependency| dependency.name.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            missing(&[]),
            [
                "absent",
                "absent but optional",
                "on this host",
                "missing by path",
                "with an extension"
            ]
        );
        assert_eq!(
            missing(&[".exe".to_string()]),
            [
                "absent",
                "absent but optional",
                "on this host",
                "missing by path"
            ]
        );
        // Without a search path, only a path can be found.
        assert_eq!(metadata.missing_on("linux", "x86_64", None, &[]).len(), 6);
        #[cfg(unix)]
        assert!(!find_executable("plain-file", Some(&path), &[]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_host_check_looks_on_path() {
        let metadata = WorkflowMetadata {
            required_dependencies: vec![
                WorkflowDependency::new("missing", "test").executable("arko-no-such-program"),
                WorkflowDependency::new("elsewhere", "test")
                    .executable("arko-no-such-program")
                    .os("plan9"),
            ],
            ..WorkflowMetadata::default()
        };
        let missing = metadata.missing_dependencies();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].name, "missing");
    }
}
