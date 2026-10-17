//! The new crate's starting files, and the checks the forge runs on it.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// A crate name from `name`: lowercase letters, digits, `-` and `_`,
/// starting with a letter.
pub fn crate_name(name: &str) -> Option<String> {
    let name: String = name
        .trim()
        .chars()
        .map(|c| match c {
            'A'..='Z' => c.to_ascii_lowercase(),
            'a'..='z' | '0'..='9' | '-' | '_' => c,
            _ => '-',
        })
        .collect();
    let name = name.trim_matches(|c| c == '-' || c == '_').to_string();
    name.starts_with(|c: char| c.is_ascii_lowercase())
        .then_some(name)
}

/// Writes the crate's starting files into `dir`, which must be missing or
/// empty.
pub fn scaffold(dir: &Path, name: &str, arko: &Path, use_case: &str) -> io::Result<()> {
    if dir.exists() && std::fs::read_dir(dir)?.next().is_some() {
        return Err(io::Error::other(format!(
            "{} is not empty. Choose a new output directory, or pass --resume to continue a run",
            dir.display()
        )));
    }
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(dir.join("Cargo.toml"), manifest(name, arko))?;
    // The checkout's lock file pins the shared dependencies to the versions
    // arko is built and tested with, and lets builds reuse its artifacts.
    let lock = arko.join("Cargo.lock");
    if lock.is_file() {
        std::fs::copy(lock, dir.join("Cargo.lock"))?;
    }
    std::fs::write(
        dir.join("src/main.rs"),
        "//! Placeholder written by the forge; the architect replaces it.\n\nfn main() {}\n",
    )?;
    std::fs::write(dir.join(".gitignore"), "/target\n")?;
    std::fs::write(
        dir.join("USE_CASE.md"),
        format!("# Use case\n\n{use_case}\n"),
    )?;
    Ok(())
}

/// The package name in `dir`'s `Cargo.toml`.
pub fn package_name(dir: &Path) -> Option<String> {
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).ok()?;
    let mut in_package = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == "name"
        {
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

/// Whether `dir`'s `Cargo.toml` depends on the arko crates in `arko` by
/// path.
pub fn depends_on(dir: &Path, arko: &Path) -> bool {
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
    let workflows = arko
        .join("arko_workflows")
        .to_string_lossy()
        .replace('\\', "/");
    manifest.replace('\\', "/").contains(&workflows)
}

/// Adds a requested change to the end of an existing crate's `USE_CASE.md`,
/// which keeps the original use case above it.
pub fn add_change(dir: &Path, change: &str) -> io::Result<()> {
    let path = dir.join("USE_CASE.md");
    let mut text = std::fs::read_to_string(&path).unwrap_or_else(|_| "# Use case\n".into());
    text = format!(
        "{}\n\n## Change requested {}\n\n{}\n",
        text.trim_end(),
        chrono::Utc::now().format("%Y-%m-%d"),
        change.trim()
    );
    std::fs::write(path, text)
}

fn manifest(name: &str, arko: &Path) -> String {
    let path = |krate: &str| {
        // TOML strings take forward slashes on every platform.
        arko.join(krate).to_string_lossy().replace('\\', "/")
    };
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"
rust-version = "1.89"
publish = false

# Its own workspace, even when it sits inside another one.
[workspace]

[dependencies]
arko_ai = {{ path = "{ai}" }}
arko_agents = {{ path = "{agents}" }}
arko_workflows = {{ path = "{workflows}" }}
async-trait = "0.1.79"
chrono = {{ version = "0.4.38", features = ["serde"] }}
clap = {{ version = "4.5", features = ["derive"] }}
env_logger = "0.10"
log = "0.4.21"
serde = {{ version = "1.0.197", features = ["derive"] }}
serde_json = "1.0.115"
tokio = {{ version = "1.37.0", features = ["full"] }}
uuid = {{ version = "1.8.0", features = ["v4", "serde"] }}
"#,
        ai = path("arko_ai"),
        agents = path("arko_agents"),
        workflows = path("arko_workflows"),
    )
}

/// One failed check: what ran, and the end of its output.
pub struct CheckFailure {
    pub check: String,
    pub output: String,
}

/// The checks every forged crate must pass: it builds, its tests pass, its
/// CLI starts, and its docs exist.
pub async fn check(dir: &Path, target_dir: &Path) -> Result<Vec<String>, CheckFailure> {
    let mut passed = Vec::new();
    for doc in ["README.md", "DESIGN.md"] {
        let written = std::fs::read_to_string(dir.join(doc))
            .map(|text| !text.trim().is_empty())
            .unwrap_or(false);
        if !written {
            return Err(CheckFailure {
                check: format!("{doc} exists"),
                output: format!("{doc} is missing or empty."),
            });
        }
    }
    for args in [
        &["build", "--all-targets"][..],
        &["test", "--quiet"],
        &["run", "--quiet", "--", "--help"],
    ] {
        let label = format!("cargo {}", args.join(" "));
        let output = tokio::time::timeout(
            Duration::from_secs(30 * 60),
            tokio::process::Command::new("cargo")
                .args(args)
                .current_dir(dir)
                .env("CARGO_TARGET_DIR", target_dir)
                .kill_on_drop(true)
                .output(),
        )
        .await;
        let output = match output {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                return Err(CheckFailure {
                    check: label,
                    output: format!("could not run cargo: {error}"),
                });
            }
            Err(_) => {
                return Err(CheckFailure {
                    check: label,
                    output: "timed out after 30 minutes".into(),
                });
            }
        };
        if !output.status.success() {
            let text = format!(
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return Err(CheckFailure {
                check: label,
                output: tail(&text, 12_000),
            });
        }
        passed.push(label);
    }
    Ok(passed)
}

/// The last `limit` bytes of `text`, on a character boundary.
pub fn tail(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut start = text.len() - limit;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &text[start..])
}

/// The directory runs are kept in, `~/.arko/forge`.
pub fn forge_home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".arko")
        .join("forge")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_names_are_cleaned() {
        assert_eq!(crate_name("Daily Digest").as_deref(), Some("daily-digest"));
        assert_eq!(crate_name("  news_bot! ").as_deref(), Some("news_bot"));
        assert_eq!(crate_name("42"), None);
    }

    #[test]
    fn the_scaffold_writes_a_standalone_crate() {
        let dir = std::env::temp_dir().join(format!("forge-scaffold-{}", uuid::Uuid::new_v4()));
        let arko = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        scaffold(&dir, "demo", arko, "Summarise my inbox.").unwrap();
        let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        assert!(manifest.contains("name = \"demo\""));
        assert!(manifest.contains("[workspace]"));
        assert!(manifest.contains("arko_workflows"));
        assert!(dir.join("Cargo.lock").is_file());
        assert!(
            std::fs::read_to_string(dir.join("USE_CASE.md"))
                .unwrap()
                .contains("Summarise my inbox.")
        );
        // Not over existing work.
        assert!(scaffold(&dir, "demo", arko, "again").is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_existing_crate_takes_a_change() {
        let dir = std::env::temp_dir().join(format!("forge-update-{}", uuid::Uuid::new_v4()));
        let arko = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        scaffold(&dir, "longcoder", arko, "Run coding work for hours.").unwrap();
        assert_eq!(package_name(&dir).as_deref(), Some("longcoder"));
        assert!(depends_on(&dir, arko));
        assert!(!depends_on(&dir, Path::new("/elsewhere/arko")));
        add_change(&dir, "Add a deliberation mode.").unwrap();
        let use_case = std::fs::read_to_string(dir.join("USE_CASE.md")).unwrap();
        let (original, change) = use_case.split_once("## Change requested").unwrap();
        assert!(original.contains("Run coding work for hours."));
        assert!(change.contains("Add a deliberation mode."));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tails_keep_the_end() {
        assert_eq!(tail("abcdef", 3), "…def");
        assert_eq!(tail("abc", 10), "abc");
    }
}
