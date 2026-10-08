//! The repositories: the pinned manifest, the output directory's layout,
//! and `prepare` (checkouts, read-only, and the arko_index indexes, timed).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::BenchResult;

/// The default manifest: seven repositories in seven languages.
pub const DEFAULT_MANIFEST: &str = include_str!("repos.json");

/// A repository pinned to a tag or commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoSpec {
    pub id: String,
    /// Where to fetch it (a URL, or a local repository or bundle).
    pub url: String,
    /// The tag (or commit) to check out.
    pub rev: String,
    /// The commit `rev` must resolve to; checked after the fetch.
    #[serde(default)]
    pub commit: Option<String>,
    /// The main language: rust, c, cpp, go, java, python, typescript, ...
    pub language: String,
    #[serde(default)]
    pub description: String,
}

pub fn load_manifest(path: Option<&Path>) -> BenchResult<Vec<RepoSpec>> {
    let text = match path {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?,
        None => DEFAULT_MANIFEST.to_string(),
    };
    let repos: Vec<RepoSpec> = serde_json::from_str(&text)?;
    let mut seen = std::collections::HashSet::new();
    for repo in &repos {
        if repo.id.is_empty()
            || !repo
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(format!("bad repository id {:?}", repo.id).into());
        }
        if !seen.insert(repo.id.clone()) {
            return Err(format!("repository {} is listed twice", repo.id).into());
        }
    }
    Ok(repos)
}

/// Where everything lives under `--out`.
#[derive(Debug, Clone)]
pub struct Layout {
    pub out: PathBuf,
}

impl Layout {
    pub fn new(out: &Path) -> BenchResult<Self> {
        std::fs::create_dir_all(out)?;
        Ok(Self {
            out: std::path::absolute(out)?,
        })
    }

    pub fn repo_dir(&self, id: &str) -> PathBuf {
        self.out.join("repos").join(id)
    }

    pub fn index_dir(&self, id: &str) -> PathBuf {
        self.out.join("indexes").join(id)
    }

    pub fn dense_dir(&self, id: &str) -> PathBuf {
        self.out.join("indexes").join(format!("{id}.dense"))
    }

    pub fn bin_dir(&self, arm: crate::Arm) -> PathBuf {
        self.out.join("bin").join(format!("arm-{}", arm.letter()))
    }

    pub fn prepared_path(&self) -> PathBuf {
        self.out.join("prepare.json")
    }

    pub fn tasks_dir(&self) -> PathBuf {
        self.out.join("tasks")
    }

    pub fn default_tasks(&self) -> PathBuf {
        self.out.join("tasks.json")
    }

    pub fn run_dir(&self, name: &str) -> PathBuf {
        self.out.join("runs").join(name)
    }

    pub fn scratch_dir(&self) -> PathBuf {
        self.out.join("scratch")
    }
}

/// A prepared repository, as `prepare.json` records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedRepo {
    pub spec: RepoSpec,
    pub checkout: PathBuf,
    pub index: PathBuf,
    /// The commit checked out.
    pub commit: String,
    /// Files (outside `.git`), their bytes, and files per language.
    pub files: usize,
    pub bytes: u64,
    pub files_by_language: BTreeMap<String, usize>,
    /// The arko_index build: wall time and its summary lines.
    pub index_build_ms: u64,
    pub index_summary: String,
    /// SHA-256 of the arko_index CLI binary that built the index.
    pub index_cli_sha256: String,
    /// Arm C's dense index, when built.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense: Option<crate::dense::DenseInfo>,
}

pub fn load_prepared(layout: &Layout) -> BenchResult<BTreeMap<String, PreparedRepo>> {
    let path = layout.prepared_path();
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn save_prepared(
    layout: &Layout,
    prepared: &BTreeMap<String, PreparedRepo>,
) -> BenchResult<()> {
    crate::write_atomic(
        &layout.prepared_path(),
        serde_json::to_string_pretty(prepared)?.as_bytes(),
    )
}

/// The arko_index CLI: `--index-cli`, else `arko_index_cli` next to this
/// binary (both are cargo examples, so a shared target directory puts them
/// side by side).
pub fn find_index_cli(explicit: Option<&Path>) -> BenchResult<PathBuf> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(std::path::absolute(path)?);
        }
        return Err(format!("--index-cli {} is not a file", path.display()).into());
    }
    let exe = std::env::current_exe()?;
    if let Some(dir) = exe.parent() {
        let name = if cfg!(windows) {
            "arko_index_cli.exe"
        } else {
            "arko_index_cli"
        };
        let sibling = dir.join(name);
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    Err("the arko_index CLI was not found next to this binary: build it with `cargo build --release -p arko_index --example arko_index_cli` (into the same target directory), or pass --index-cli".into())
}

pub fn sha256_file(path: &Path) -> BenchResult<String> {
    let bytes = std::fs::read(path)?;
    Ok(hex(&Sha256::digest(&bytes)))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Options of `prepare`.
pub struct PrepareOptions<'a> {
    pub layout: &'a Layout,
    pub repos: Vec<RepoSpec>,
    /// Repository id -> local source overriding its URL.
    pub sources: HashMap<String, String>,
    pub index_cli: PathBuf,
    /// Fetch and index again even when prepared.
    pub force: bool,
    pub dense: Option<crate::dense::DenseOptions>,
}

pub async fn prepare(options: PrepareOptions<'_>) -> BenchResult<BTreeMap<String, PreparedRepo>> {
    let layout = options.layout;
    let _lock = crate::ledger::DirLock::acquire(&layout.out.join("prepare"), "a prepare")?;
    let mut busy: Vec<PathBuf> = vec![layout.tasks_dir()];
    if let Ok(runs) = std::fs::read_dir(layout.out.join("runs")) {
        busy.extend(runs.flatten().map(|entry| entry.path()));
    }
    for dir in busy {
        if let Some(pid) = crate::ledger::DirLock::holder(&dir) {
            return Err(format!(
                "{} is in use by process {pid}: prepare changes the checkouts and indexes it uses; wait for it, or stop it",
                dir.display()
            )
            .into());
        }
    }
    let mut prepared = load_prepared(layout)?;
    let cli_sha = sha256_file(&options.index_cli)?;
    for spec in &options.repos {
        let source = options
            .sources
            .get(&spec.id)
            .cloned()
            .unwrap_or_else(|| spec.url.clone());
        let checkout = layout.repo_dir(&spec.id);
        let index = layout.index_dir(&spec.id);
        let existing = prepared.get(&spec.id).cloned();
        let checkout_current = !options.force
            && existing
                .as_ref()
                .is_some_and(|repo| repo.spec.rev == spec.rev && checkout.join(".git").exists());
        let commit = if checkout_current {
            let commit = existing
                .as_ref()
                .map(|repo| repo.commit.clone())
                .unwrap_or_default();
            eprintln!("{}: checkout of {} present ({commit})", spec.id, spec.rev);
            commit
        } else {
            eprintln!("{}: fetching {} from {source}", spec.id, spec.rev);
            checkout_repo(&checkout, &source, &spec.rev)?
        };
        if let Some(expected) = &spec.commit
            && !commit.starts_with(expected.as_str())
            && !expected.starts_with(commit.as_str())
        {
            return Err(format!(
                "{}: {} resolved to {commit}, but the manifest pins {expected}",
                spec.id, spec.rev
            )
            .into());
        }
        let (files, bytes, files_by_language) = tree_stats(&checkout)?;
        let index_current = checkout_current
            && existing.as_ref().is_some_and(|repo| {
                repo.index_cli_sha256 == cli_sha && index.join("CURRENT2").exists()
            });
        let (index_build_ms, index_summary) = if index_current {
            let repo = existing.as_ref().expect("checked above");
            eprintln!("{}: index present", spec.id);
            (repo.index_build_ms, repo.index_summary.clone())
        } else {
            build_index(&options.index_cli, &checkout, &index, &spec.id)?
        };
        let mut dense = existing
            .as_ref()
            .and_then(|repo| repo.dense.clone())
            .filter(|_| index_current);
        if let Some(dense_options) = &options.dense
            && (dense.is_none() || options.force)
        {
            let dense_dir = layout.dense_dir(&spec.id);
            set_writable(&dense_dir, true);
            dense = Some(
                crate::dense::build(
                    dense_options,
                    &options.index_cli,
                    &checkout,
                    &dense_dir,
                    &spec.id,
                )
                .await?,
            );
            set_writable(&dense_dir, false);
        }
        set_writable(&checkout, false);
        set_writable(&index, false);
        let repo = PreparedRepo {
            spec: spec.clone(),
            checkout: checkout.clone(),
            index: index.clone(),
            commit,
            files,
            bytes,
            files_by_language,
            index_build_ms,
            index_summary,
            index_cli_sha256: cli_sha.clone(),
            dense,
        };
        prepared.insert(spec.id.clone(), repo);
        save_prepared(layout, &prepared)?;
    }
    Ok(prepared)
}

/// Fetches `rev` from `source` into a fresh `dir` (depth 1) and checks it
/// out; returns the commit.
fn checkout_repo(dir: &Path, source: &str, rev: &str) -> BenchResult<String> {
    if dir.exists() {
        set_writable(dir, true);
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    git(dir, &["init", "-q"])?;
    // A local source is fetched by path: `git fetch` then reads it as a
    // repository (or a bundle).
    git(dir, &["fetch", "-q", "--depth", "1", source, rev])?;
    git(
        dir,
        &[
            "-c",
            "advice.detachedHead=false",
            "checkout",
            "-q",
            "--detach",
            "FETCH_HEAD",
        ],
    )?;
    let commit = git(dir, &["rev-parse", "HEAD"])?;
    Ok(commit.trim().to_string())
}

fn git(dir: &Path, args: &[&str]) -> BenchResult<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|error| format!("running git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed in {}: {}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Builds the arko_index index of `checkout` into `index` with the CLI,
/// timed. Returns the wall time and the CLI's summary lines.
fn build_index(cli: &Path, checkout: &Path, index: &Path, id: &str) -> BenchResult<(u64, String)> {
    if index.exists() {
        set_writable(index, true);
        std::fs::remove_dir_all(index)?;
    }
    if let Some(parent) = index.parent() {
        std::fs::create_dir_all(parent)?;
    }
    eprintln!("{id}: building the arko_index index");
    let started = Instant::now();
    let output = Command::new(cli)
        .arg("index2")
        .arg(checkout)
        .arg(index)
        .output()
        .map_err(|error| format!("running {}: {error}", cli.display()))?;
    let elapsed = started.elapsed().as_millis() as u64;
    let summary = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .trim()
    .to_string();
    if !output.status.success() {
        return Err(format!("{id}: arko_index index2 failed: {summary}").into());
    }
    eprintln!(
        "{id}: indexed in {:.1} s: {summary}",
        elapsed as f64 / 1000.0
    );
    Ok((elapsed, summary))
}

/// Files, bytes and files per language under `root`, outside `.git`.
pub fn tree_stats(root: &Path) -> BenchResult<(usize, u64, BTreeMap<String, usize>)> {
    let mut files = 0;
    let mut bytes = 0;
    let mut by_language = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if entry.file_name() != ".git" {
                    stack.push(entry.path());
                }
            } else if file_type.is_file() {
                files += 1;
                bytes += entry.metadata()?.len();
                if let Some(language) = language_of(&entry.path()) {
                    *by_language.entry(language.to_string()).or_insert(0) += 1;
                }
            }
        }
    }
    Ok((files, bytes, by_language))
}

/// The language arko_index parses a file as, by its extension.
pub fn language_of(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match extension.as_str() {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "js" | "mjs" | "cjs" | "jsx" => "javascript",
        "ts" | "mts" | "cts" | "tsx" => "typescript",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => "cpp",
        _ => return None,
    })
}

/// Makes `path` and everything below it read-only, or writable again (for
/// the owner). The agents' checkouts are read-only, so no agent can change
/// what another reads. Errors are ignored: a missing path has nothing to do.
pub fn set_writable(path: &Path, writable: bool) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fn walk(path: &Path, writable: bool) {
            let Ok(metadata) = std::fs::symlink_metadata(path) else {
                return;
            };
            if metadata.file_type().is_symlink() {
                return;
            }
            let mode = metadata.permissions().mode();
            let new_mode = if writable {
                mode | 0o200
            } else {
                mode & !0o222
            };
            // A directory is made writable before its entries and read-only
            // after them.
            if metadata.is_dir() {
                if writable {
                    let _ =
                        std::fs::set_permissions(path, std::fs::Permissions::from_mode(new_mode));
                }
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        walk(&entry.path(), writable);
                    }
                }
                if !writable {
                    let _ =
                        std::fs::set_permissions(path, std::fs::Permissions::from_mode(new_mode));
                }
            } else {
                let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(new_mode));
            }
        }
        walk(path, writable);
    }
    #[cfg(not(unix))]
    {
        let _ = (path, writable);
    }
}

/// The programs an arm may not have: stubs of these names, first on its
/// `PATH`, answer "command not found", whatever else the machine has
/// installed.
const INDEX_PROGRAMS: [&str; 3] = ["arko_index", "arko_index_cli", "arko_index_dense"];

/// Sets up each arm's tool directory, first on its shell's `PATH`: arm A
/// gets stubs for every arko_index program; arm B the arko_index CLI as
/// `arko_index` and a stub for `arko_index_dense`; arm C also
/// `arko_index_dense`, which asks this process's dense search service.
/// Only the directories of `arms` are touched (a command sets up the arms
/// it runs, and leaves those of another process alone), and files are
/// only replaced (atomically) when they differ.
pub fn setup_arm_bins(layout: &Layout, arms: &[crate::Arm], index_cli: &Path) -> BenchResult<()> {
    let stub =
        |name: &str| format!("#!/bin/sh\necho \"bash: {name}: command not found\" >&2\nexit 127\n");
    for &arm in arms {
        let dir = layout.bin_dir(arm);
        std::fs::create_dir_all(&dir)?;
        let stubs: &[&str] = match arm {
            crate::Arm::A => &INDEX_PROGRAMS,
            crate::Arm::B => &INDEX_PROGRAMS[1..],
            crate::Arm::C => &INDEX_PROGRAMS[1..2],
        };
        for name in stubs {
            ensure_script(&dir.join(name), &stub(name))?;
        }
        if arm == crate::Arm::A {
            continue;
        }
        ensure_link(index_cli, &dir.join("arko_index"))?;
        if arm == crate::Arm::C {
            let exe = std::env::current_exe()?;
            ensure_script(
                &dir.join("arko_index_dense"),
                &format!(
                    "#!/bin/sh\nexec '{}' dense-query \"$@\"\n",
                    exe.display().to_string().replace('\'', "'\\''")
                ),
            )?;
        }
    }
    Ok(())
}

/// Writes an executable script, through a temporary file and a rename,
/// unless it already holds `text`.
fn ensure_script(path: &Path, text: &str) -> BenchResult<()> {
    if std::fs::read_to_string(path).is_ok_and(|current| current == text) {
        return Ok(());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, text)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(())
}

/// Points `link` at `target` (a symbolic link; a copy on Windows), through
/// a temporary link and a rename, unless it already does.
fn ensure_link(target: &Path, link: &Path) -> BenchResult<()> {
    #[cfg(unix)]
    {
        if std::fs::read_link(link).is_ok_and(|current| current == target) {
            return Ok(());
        }
        let temporary = link.with_extension(format!("tmp-{}", std::process::id()));
        let _ = std::fs::remove_file(&temporary);
        std::os::unix::fs::symlink(target, &temporary)?;
        std::fs::rename(&temporary, link)?;
    }
    #[cfg(not(unix))]
    std::fs::copy(target, link.with_extension("exe")).map(|_| ())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_manifest_pins_seven_repositories_in_seven_languages() {
        let repos = load_manifest(None).unwrap();
        assert_eq!(repos.len(), 7);
        let languages: std::collections::BTreeSet<_> =
            repos.iter().map(|repo| repo.language.as_str()).collect();
        assert_eq!(languages.len(), 7);
        assert!(repos.iter().all(|repo| repo.commit.is_some()));
    }

    #[cfg(unix)]
    #[test]
    fn read_only_trees_can_be_made_writable_again() {
        let root = crate::test_dir("readonly");
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("a/b/f.txt"), "x").unwrap();
        set_writable(&root, false);
        // Permissions don't bind root.
        let root_user = std::env::var("USER").is_ok_and(|user| user == "root");
        if !root_user {
            assert!(std::fs::write(root.join("a/b/f.txt"), "y").is_err());
            assert!(std::fs::write(root.join("a/new.txt"), "y").is_err());
        }
        set_writable(&root, true);
        std::fs::write(root.join("a/b/f.txt"), "y").unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn tree_stats_skip_git_and_count_languages() {
        let root = crate::test_dir("stats");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join(".git/HEAD"), "ref").unwrap();
        std::fs::write(root.join("src/a.rs"), "fn a() {}").unwrap();
        std::fs::write(root.join("src/b.h"), "int b;").unwrap();
        std::fs::write(root.join("README"), "hi").unwrap();
        let (files, bytes, languages) = tree_stats(&root).unwrap();
        assert_eq!(files, 3);
        assert_eq!(bytes, 9 + 6 + 2);
        assert_eq!(languages.get("rust"), Some(&1));
        assert_eq!(languages.get("c"), Some(&1));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
