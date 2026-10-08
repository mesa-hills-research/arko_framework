//! Variant arms: arms with arm B's tools whose index was built, whose search
//! results render, or whose tool section reads, another way, for A/B tests of
//! arko_index options and prompts on the same tasks and solvers.
//!
//! A variants file defines them; `prepare-variants` builds one index per variant
//! and repository (`indexes/<repo>@<variant>/`), or reuses another arm's index
//! (`base`), and records them in `variants.json`; `run --arms B,NAME,...` runs
//! them. In a variant arm,
//! `arko_index` is the CLI itself, or a script that adds the variant's
//! `search2` flags. When the index embeds, `search2` goes through a service in
//! the benchmark process that runs the CLI with the embedding key, so the key
//! never reaches an agent's shell (like arm C's `arko_index_dense`).

// The search service is Unix-only; elsewhere its parts are unused.
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::repos::{Layout, PreparedRepo, set_writable};
use crate::{Arm, BenchResult};

/// The variable that holds the search service's socket in a variant arm's shell.
pub const SOCKET_VARIABLE: &str = "ARKO_INDEX_SEARCH_SOCKET";

/// One variant arm, as a variants file defines it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariantSpec {
    /// The arm's name: 1 to 32 letters, digits, `-` or `_`, other than A, B, C.
    pub name: String,
    /// What the variant tests, for the report.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Another arm whose index this one uses instead of building its own: `B`,
    /// `C` (B's index and arm C's `arko_index_dense`), or a variant with an
    /// index of its own, as that arm's index is when a run starts. `index`
    /// stays empty then, and `search`, `prompt` and `tools_text` stay this
    /// variant's. For example `{"name": "B-guided", "base": "B", "prompt":
    /// "guided"}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    /// `arko_index index2` flags for the variant's index, such as
    /// `["--lexical-scope"]` or `["--embed", "voyage-code-4", "--embed-text",
    /// "source"]`. Empty: the default index (arm B's, built by this CLI).
    #[serde(default)]
    pub index: Vec<String>,
    /// `search2` flags put before the solver's own on every search, such as
    /// `["--text"]` or `["--mode", "lexical"]`.
    #[serde(default)]
    pub search: Vec<String>,
    /// The solver's tool section: `standard` (the default: arm B's, its hybrid
    /// version when the index embeds, arm C's with `base` C) or `guided` (which
    /// tool to use when, with worked examples: `prompts::TOOLS_GUIDED`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// A tool paragraph of your own instead (`{rg}` becomes the note on
    /// ripgrep), for render variants whose output the standard paragraphs don't
    /// describe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_text: Option<String>,
}

impl VariantSpec {
    /// Whether the index flags ask for embeddings.
    pub fn embeds(&self) -> bool {
        self.index
            .iter()
            .any(|flag| ["--embed", "--embed-config"].contains(&flag_name(flag)))
    }

    /// Whether the arm has arm C's `arko_index_dense` (`base` C).
    pub fn dense_tool(&self) -> bool {
        self.base.as_deref() == Some("C")
    }

    /// The prompt's name: `standard`, `guided`, or `own text` (`tools_text`).
    pub fn prompt_name(&self) -> &str {
        if self.tools_text.is_some() {
            "own text"
        } else {
            self.prompt.as_deref().unwrap_or("standard")
        }
    }

    /// The solver's tool section (before `{rg}` is replaced), for an index that
    /// embeds (`hybrid`) or not.
    pub fn tools_paragraph(&self, hybrid: bool) -> BenchResult<String> {
        use crate::prompts;
        if let Some(text) = &self.tools_text {
            return Ok(text.clone());
        }
        match (self.prompt_name(), hybrid, self.dense_tool()) {
            ("standard", _, true) => {
                Ok(format!("{}{}", prompts::TOOLS_INDEX, prompts::TOOLS_DENSE))
            }
            ("standard", true, false) => Ok(prompts::TOOLS_INDEX_HYBRID.to_string()),
            ("standard", false, false) => Ok(prompts::TOOLS_INDEX.to_string()),
            ("guided", hybrid, dense) => Ok(prompts::guided_tools(hybrid, dense)),
            (other, ..) => Err(format!(
                "variant {}: prompt {other:?}: standard or guided (or give tools_text)",
                self.name
            )
            .into()),
        }
    }

    fn validate(&self) -> BenchResult<()> {
        match Arm::parse(&self.name)? {
            Arm::Variant(_) => {}
            _ => {
                return Err(format!(
                    "variant {:?}: A, B and C are the built-in arms; choose another name",
                    self.name
                )
                .into());
            }
        }
        for flag in &self.index {
            if ["--report", "--help", "-h"].contains(&flag_name(flag)) {
                return Err(
                    format!("variant {}: index flag {flag} is not allowed", self.name).into(),
                );
            }
        }
        for flag in &self.search {
            if flag == "--" {
                return Err(format!("variant {}: search flags can't hold --", self.name).into());
            }
        }
        if let Some(base) = &self.base {
            if !self.index.is_empty() {
                return Err(format!(
                    "variant {}: a variant with a base uses its base's index: leave out `index`",
                    self.name
                )
                .into());
            }
            let builtin = matches!(Arm::parse(base)?, Arm::A | Arm::B | Arm::C);
            if base == &self.name || base == "A" || (builtin && base != "B" && base != "C") {
                return Err(
                    format!("variant {}: base {base:?}: B, C or a variant", self.name).into(),
                );
            }
        }
        self.tools_paragraph(self.embeds())?;
        Ok(())
    }

    /// The `index2` flags of a build: the variant's, with the offline hashing
    /// embedder for `--mock-embeddings`.
    pub fn build_flags(&self, mock_embeddings: bool) -> Vec<String> {
        let mut flags = Vec::new();
        let mut iter = self.index.iter();
        while let Some(flag) = iter.next() {
            match flag.split_once('=') {
                Some(("--embed", _)) if mock_embeddings => flags.push("--embed=hash".into()),
                None if flag == "--embed" && mock_embeddings => {
                    iter.next();
                    flags.extend(["--embed".to_string(), "hash".to_string()]);
                }
                _ => flags.push(flag.clone()),
            }
        }
        flags
    }
}

/// `--flag=value` and `--flag` alike: the flag's name.
fn flag_name(flag: &str) -> &str {
    flag.split_once('=').map_or(flag, |(name, _)| name)
}

/// Reads a variants file: a JSON list of [`VariantSpec`]s.
pub fn load_specs(path: &Path) -> BenchResult<Vec<VariantSpec>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("reading {}: {error}", path.display()))?;
    parse_specs(&text).map_err(|error| format!("{}: {error}", path.display()).into())
}

/// The variants of a variants file's text, checked.
pub fn parse_specs(text: &str) -> BenchResult<Vec<VariantSpec>> {
    let specs: Vec<VariantSpec> = serde_json::from_str(text)?;
    let mut seen = std::collections::HashSet::new();
    for spec in &specs {
        spec.validate()?;
        if !seen.insert(spec.name.to_ascii_lowercase()) {
            return Err(format!("variant {} is defined twice", spec.name).into());
        }
    }
    for spec in &specs {
        if let Some(base) = &spec.base
            && let Some(target) = specs.iter().find(|other| &other.name == base)
            && target.base.is_some()
        {
            return Err(format!(
                "variant {}: base {base} has a base itself; name the variant with the index",
                spec.name
            )
            .into());
        }
    }
    Ok(specs)
}

/// One variant's index of one repository, as `variants.json` records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantIndex {
    pub dir: PathBuf,
    /// The commit indexed.
    pub commit: String,
    /// The `index2` flags it was built with.
    pub flags: Vec<String>,
    /// SHA-256 of the arko_index CLI that built it.
    pub cli_sha256: String,
    /// Whether it has embeddings (its `dense.json`).
    pub dense: bool,
    pub build_ms: u64,
    /// The CLI's stderr: counts, embedding tokens, the report summary.
    pub summary: String,
}

/// A prepared variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedVariant {
    pub spec: VariantSpec,
    /// Built with the offline hashing embedder (`--mock-embeddings`).
    #[serde(default)]
    pub mock_embeddings: bool,
    pub repos: BTreeMap<String, VariantIndex>,
}

impl PreparedVariant {
    /// Whether its indexes have embeddings (any of them).
    pub fn dense(&self) -> bool {
        self.repos.values().any(|index| index.dense)
    }

    /// The solver's tool section (before `{rg}` is replaced).
    pub fn tools_paragraph(&self) -> BenchResult<String> {
        self.spec.tools_paragraph(self.dense())
    }
}

pub fn load_prepared(layout: &Layout) -> BenchResult<BTreeMap<String, PreparedVariant>> {
    let path = layout.variants_path();
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&text)?)
}

fn save_prepared(layout: &Layout, prepared: &BTreeMap<String, PreparedVariant>) -> BenchResult<()> {
    crate::write_atomic(
        &layout.variants_path(),
        serde_json::to_string_pretty(prepared)?.as_bytes(),
    )
}

/// Options of `prepare-variants`.
pub struct PrepareVariants<'a> {
    pub layout: &'a Layout,
    pub specs: Vec<VariantSpec>,
    pub repos: &'a BTreeMap<String, PreparedRepo>,
    pub index_cli: PathBuf,
    pub mock_embeddings: bool,
    pub force: bool,
}

/// Builds every variant's index of every repository that isn't built with the
/// same flags, CLI and commit yet. An index directory is kept across rebuilds,
/// so its embedding cache saves the tokens of chunks embedded before.
pub fn prepare(options: PrepareVariants<'_>) -> BenchResult<BTreeMap<String, PreparedVariant>> {
    let layout = options.layout;
    let _lock = crate::ledger::DirLock::acquire(&layout.variants_lock_dir(), "a prepare-variants")?;
    if let Some(pid) = crate::ledger::DirLock::holder(&layout.out.join("prepare")) {
        return Err(format!("`prepare` is running (pid {pid}); wait for it").into());
    }
    let cli_sha = crate::repos::sha256_file(&options.index_cli)?;
    let mut prepared = load_prepared(layout)?;
    // Variants with an index of their own first, then those that reuse one.
    let (own, based): (Vec<&VariantSpec>, Vec<&VariantSpec>) =
        options.specs.iter().partition(|spec| spec.base.is_none());
    // The indexes to build: those not built with the same flags, CLI and commit
    // yet. A definition's other fields (description, search flags, prompt) leave
    // its indexes as they are.
    let mut stale = std::collections::BTreeSet::new();
    for spec in &own {
        let flags = spec.build_flags(options.mock_embeddings);
        let existing = prepared.get(&spec.name);
        for (id, repo) in options.repos {
            let dir = layout.variant_index_dir(id, &spec.name);
            let current = existing
                .and_then(|variant| variant.repos.get(id))
                .is_some_and(|index| {
                    index.dir == dir
                        && index.flags == flags
                        && index.cli_sha256 == cli_sha
                        && index.commit == repo.commit
                        && dir.join("CURRENT2").exists()
                });
            if !current || options.force {
                stale.insert(dir);
            }
        }
    }
    // A running run that reads one of these indexes (a variant's own, or one
    // based on it) must end first. Other runs go on beside the builds.
    if !stale.is_empty()
        && let Ok(runs) = std::fs::read_dir(layout.out.join("runs"))
    {
        for run in runs.flatten() {
            let Some(pid) = crate::ledger::DirLock::holder(&run.path()) else {
                continue;
            };
            let reads_these =
                crate::runner::RunManifest::load(&run.path()).map_or(true, |manifest| {
                    manifest
                        .variants
                        .values()
                        .flat_map(|variant| variant.repos.values())
                        .any(|index| stale.contains(&index.dir))
                });
            if reads_these {
                return Err(format!(
                    "{} is in use by process {pid} and reads indexes this would rebuild; wait for it, or stop it",
                    run.path().display()
                )
                .into());
            }
        }
    }
    for spec in own {
        let flags = spec.build_flags(options.mock_embeddings);
        let mut variant = PreparedVariant {
            spec: spec.clone(),
            mock_embeddings: options.mock_embeddings,
            repos: prepared
                .get(&spec.name)
                .map(|existing| existing.repos.clone())
                .unwrap_or_default(),
        };
        for (id, repo) in options.repos {
            let dir = layout.variant_index_dir(id, &spec.name);
            if !stale.contains(&dir) {
                eprintln!("{}@{}: index present", id, spec.name);
                continue;
            }
            let index = build(
                &options.index_cli,
                &cli_sha,
                &flags,
                repo,
                &dir,
                id,
                &spec.name,
            )?;
            variant.repos.insert(id.clone(), index);
            prepared.insert(spec.name.clone(), variant.clone());
            save_prepared(layout, &prepared)?;
        }
        prepared.insert(spec.name.clone(), variant);
        save_prepared(layout, &prepared)?;
    }
    for spec in based {
        let base = spec.base.as_deref().unwrap_or_default();
        let mut repos = BTreeMap::new();
        for (id, repo) in options.repos {
            let index = match base {
                "B" | "C" => {
                    if base == "C" && repo.dense.is_none() {
                        return Err(format!(
                            "variant {}: arm C's dense index of {id} is missing; run `prepare --dense`",
                            spec.name
                        )
                        .into());
                    }
                    arm_index(base, repo)
                }
                variant => prepared
                    .get(variant)
                    .and_then(|prepared| prepared.repos.get(id))
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "variant {}: base {variant} has no index of {id}; prepare {variant} first",
                            spec.name
                        )
                    })?,
            };
            repos.insert(id.clone(), index);
        }
        eprintln!("{}: uses {base}'s indexes", spec.name);
        let mut variant = prepared
            .get(&spec.name)
            .filter(|existing| existing.spec == *spec)
            .cloned()
            .unwrap_or(PreparedVariant {
                spec: spec.clone(),
                mock_embeddings: options.mock_embeddings,
                repos: BTreeMap::new(),
            });
        variant.repos.extend(repos);
        prepared.insert(spec.name.clone(), variant);
        save_prepared(layout, &prepared)?;
    }
    Ok(prepared)
}

/// Arm B's index of `repo`, for a variant based on arm `base` (`B` or `C`).
fn arm_index(base: &str, repo: &PreparedRepo) -> VariantIndex {
    VariantIndex {
        dir: repo.index.clone(),
        commit: repo.commit.clone(),
        flags: Vec::new(),
        cli_sha256: repo.index_cli_sha256.clone(),
        dense: false,
        build_ms: repo.index_build_ms,
        summary: format!("arm {base}'s index"),
    }
}

/// Builds one variant index with the CLI, timed.
fn build(
    cli: &Path,
    cli_sha: &str,
    flags: &[String],
    repo: &PreparedRepo,
    dir: &Path,
    id: &str,
    name: &str,
) -> BenchResult<VariantIndex> {
    std::fs::create_dir_all(dir)?;
    set_writable(dir, true);
    // The index's saved embedding settings go: the CLI would otherwise keep the
    // ones flags leave out (an earlier definition's cap, quantization or text),
    // so each build is what its flags say. The embedding cache stays.
    match std::fs::remove_file(dir.join("dense.json")) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    eprintln!(
        "{id}@{name}: building the index: index2 {}",
        flags.join(" ")
    );
    let started = Instant::now();
    let output = std::process::Command::new(cli)
        .arg("index2")
        .args(flags)
        .arg("--")
        .arg(&repo.checkout)
        .arg(dir)
        .output()
        .map_err(|error| format!("running {}: {error}", cli.display()))?;
    let build_ms = started.elapsed().as_millis() as u64;
    let summary = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let dense = dir.join("dense.json").exists();
    set_writable(dir, false);
    if !output.status.success() {
        return Err(format!("{id}@{name}: arko_index index2 failed: {summary}").into());
    }
    // The counts, the embedding summary and the report summary.
    let lines: Vec<&str> = summary
        .lines()
        .filter(|line| {
            line.starts_with("indexed ")
                || line.contains(": embeddings: ")
                || line.contains("token cap")
                || line.contains(" files indexed")
        })
        .collect();
    eprintln!(
        "{id}@{name}: indexed in {:.1} s: {}",
        build_ms as f64 / 1000.0,
        lines.join(" | ")
    );
    Ok(VariantIndex {
        dir: dir.to_path_buf(),
        commit: repo.commit.clone(),
        flags: flags.to_vec(),
        cli_sha256: cli_sha.to_string(),
        dense,
        build_ms,
        summary,
    })
}

/// The variants among `arms`, each with an index of every repository in
/// `task_repos` of the commit `repos` has checked out. A variant based on
/// another arm gets that arm's indexes as they are now.
pub fn for_arms(
    layout: &Layout,
    arms: &[Arm],
    repos: &BTreeMap<String, PreparedRepo>,
    task_repos: &std::collections::BTreeSet<String>,
) -> BenchResult<BTreeMap<String, PreparedVariant>> {
    let names: Vec<&str> = arms
        .iter()
        .filter_map(|arm| match arm {
            Arm::Variant(name) => Some(*name),
            _ => None,
        })
        .collect();
    if names.is_empty() {
        return Ok(BTreeMap::new());
    }
    let prepared = load_prepared(layout)?;
    let mut out = BTreeMap::new();
    for name in names {
        let mut variant = prepared
            .get(name)
            .ok_or_else(|| {
                format!("arm {name} is not a prepared variant: run `prepare-variants` with a variants file that defines it")
            })?
            .clone();
        // variants.json holds a copy of the base's records from when this
        // variant was prepared: the base may have been rebuilt since.
        if let Some(base) = variant.spec.base.clone() {
            variant.repos = match base.as_str() {
                "B" | "C" => repos
                    .iter()
                    .filter(|(_, repo)| base == "B" || repo.dense.is_some())
                    .map(|(id, repo)| (id.clone(), arm_index(&base, repo)))
                    .collect(),
                other => prepared
                    .get(other)
                    .ok_or_else(|| {
                        format!(
                            "variant {name}: its base {other} is not prepared: run `prepare-variants` for it"
                        )
                    })?
                    .repos
                    .clone(),
            };
        }
        let whose = variant.spec.base.as_deref().unwrap_or(name);
        let missing: Vec<&str> = task_repos
            .iter()
            .filter(|repo| !variant.repos.contains_key(*repo))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            let fix = match whose {
                "B" => "run `prepare` for them".to_string(),
                "C" => "run `prepare --dense` for them".to_string(),
                _ => format!("run `prepare-variants` for {whose}"),
            };
            return Err(format!(
                "variant {name} has no index of {}: {fix}",
                missing.join(", ")
            )
            .into());
        }
        for id in task_repos {
            let (Some(index), Some(repo)) = (variant.repos.get(id), repos.get(id)) else {
                continue;
            };
            if index.commit != repo.commit {
                let short = |commit: &str| commit[..commit.len().min(10)].to_string();
                return Err(format!(
                    "variant {name}'s index of {id} is of commit {}, but the checkout is of {}: run `prepare-variants` for {whose} again",
                    short(&index.commit),
                    short(&repo.commit)
                )
                .into());
            }
        }
        out.insert(name.to_string(), variant);
    }
    Ok(out)
}

/// Refuses variants whose searches embed their queries with Voyage when
/// `VOYAGE_API_KEY` is unset, as arm C does: without it every `search2` of
/// theirs would fail.
pub fn check_embedding_key(
    variants: &BTreeMap<String, PreparedVariant>,
    task_repos: &std::collections::BTreeSet<String>,
) -> BenchResult<()> {
    let key_set = std::env::var_os("VOYAGE_API_KEY")
        .is_some_and(|key| !key.to_string_lossy().trim().is_empty());
    match voyage_variant(variants, task_repos)? {
        Some((name, id)) if !key_set => Err(format!(
            "variant {name}'s index of {id} embeds with Voyage: its searches need VOYAGE_API_KEY (or prepare-variants --mock-embeddings for a dry run)"
        )
        .into()),
        _ => Ok(()),
    }
}

/// The first variant, and repository, whose index of a task repository embeds
/// with Voyage.
fn voyage_variant(
    variants: &BTreeMap<String, PreparedVariant>,
    task_repos: &std::collections::BTreeSet<String>,
) -> BenchResult<Option<(String, String)>> {
    for (name, variant) in variants {
        for (id, index) in &variant.repos {
            if index.dense
                && task_repos.contains(id)
                && embedding_provider(&index.dir)?.as_deref() == Some("voyage")
            {
                return Ok(Some((name.clone(), id.clone())));
            }
        }
    }
    Ok(None)
}

/// The embedding provider of an index's current generation (`voyage`, `hash`),
/// or `None` when it has no embeddings.
fn embedding_provider(dir: &Path) -> BenchResult<Option<String>> {
    let generation = std::fs::read_to_string(dir.join("CURRENT2"))
        .map_err(|error| format!("{}: {error}", dir.join("CURRENT2").display()))?;
    let path = dir.join(generation.trim()).join("dense.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("{}: {error}", path.display()).into()),
    };
    let info: serde_json::Value = serde_json::from_slice(&bytes)?;
    Ok(info["embedder"]["provider"].as_str().map(str::to_string))
}

/// The `arko_index` a variant arm's shell runs: the CLI itself, or a script
/// that adds the variant's `search2` flags, or one that sends `search2` to the
/// run's search service (when the index embeds).
pub fn arko_index_program(
    variant: &PreparedVariant,
    name: &str,
    index_cli: &Path,
) -> BenchResult<Option<String>> {
    let quote = |text: &str| format!("'{}'", text.replace('\'', "'\\''"));
    let cli = quote(&index_cli.display().to_string());
    if variant.dense() {
        let exe = std::env::current_exe()?;
        return Ok(Some(format!(
            "#!/bin/sh\nif [ \"$1\" = search2 ]; then exec {} index-search {} \"$@\"; fi\nexec {cli} \"$@\"\n",
            quote(&exe.display().to_string()),
            quote(name),
        )));
    }
    if variant.spec.search.is_empty() {
        return Ok(None);
    }
    let flags: Vec<String> = variant.spec.search.iter().map(|flag| quote(flag)).collect();
    Ok(Some(format!(
        "#!/bin/sh\nif [ \"$1\" = search2 ]; then shift; exec {cli} search2 {} \"$@\"; fi\nexec {cli} \"$@\"\n",
        flags.join(" ")
    )))
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchRequest {
    variant: String,
    /// The arguments after `search2`.
    args: Vec<String>,
    cwd: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchReply {
    stdout: String,
    stderr: String,
    code: i32,
}

/// The output kept of one search, per stream.
const MAX_REPLY_BYTES: usize = 4 << 20;

/// The run's search service for variant arms whose index embeds: runs
/// `arko_index search2` with the variant's flags in this process's environment,
/// which holds the embedding key.
pub struct SearchService {
    socket: PathBuf,
    task: tokio::task::JoinHandle<()>,
}

impl SearchService {
    #[cfg(unix)]
    pub async fn start(
        layout: &Layout,
        variants: &BTreeMap<String, PreparedVariant>,
        index_cli: &Path,
        timeout: Duration,
    ) -> BenchResult<Self> {
        let mut socket = layout
            .scratch_dir()
            .join(format!("index-search-{}.sock", std::process::id()));
        // Unix socket paths are limited to about 100 bytes.
        if socket.as_os_str().len() > 100 {
            socket = std::env::temp_dir().join(format!(
                "index-benchmark-search-{}.sock",
                std::process::id()
            ));
        }
        if let Some(parent) = socket.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _ = std::fs::remove_file(&socket);
        let listener = tokio::net::UnixListener::bind(&socket)?;
        let flags: BTreeMap<String, Vec<String>> = variants
            .iter()
            .filter(|(_, variant)| variant.dense())
            .map(|(name, variant)| (name.clone(), variant.spec.search.clone()))
            .collect();
        let cli = index_cli.to_path_buf();
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let flags = flags.clone();
                let cli = cli.clone();
                tokio::spawn(async move {
                    let _ = serve(stream, &flags, &cli, timeout).await;
                });
            }
        });
        Ok(Self { socket, task })
    }

    #[cfg(not(unix))]
    pub async fn start(
        _layout: &Layout,
        _variants: &BTreeMap<String, PreparedVariant>,
        _index_cli: &Path,
        _timeout: Duration,
    ) -> BenchResult<Self> {
        Err("variant arms whose index embeds need a Unix host".into())
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub fn stop(self) {
        self.task.abort();
        let _ = std::fs::remove_file(&self.socket);
    }
}

#[cfg(unix)]
async fn serve(
    stream: tokio::net::UnixStream,
    flags: &BTreeMap<String, Vec<String>>,
    cli: &Path,
    timeout: Duration,
) -> BenchResult<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let (reader, mut writer) = stream.into_split();
    let mut line = String::new();
    BufReader::new(reader).read_line(&mut line).await?;
    let reply = match search(&line, flags, cli, timeout).await {
        Ok(reply) => reply,
        Err(error) => SearchReply {
            stdout: String::new(),
            stderr: format!("arko_index: {error}\n"),
            code: 1,
        },
    };
    let mut text = serde_json::to_string(&reply)?;
    text.push('\n');
    writer.write_all(text.as_bytes()).await?;
    writer.shutdown().await?;
    Ok(())
}

#[cfg(unix)]
async fn search(
    line: &str,
    flags: &BTreeMap<String, Vec<String>>,
    cli: &Path,
    timeout: Duration,
) -> BenchResult<SearchReply> {
    let request: SearchRequest = serde_json::from_str(line)?;
    let flags = flags
        .get(&request.variant)
        .ok_or_else(|| format!("no search service for variant {}", request.variant))?;
    let child = tokio::process::Command::new(cli)
        .arg("search2")
        .args(flags)
        .args(&request.args)
        .current_dir(&request.cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| format!("search2 took longer than {} s", timeout.as_secs()))??;
    let text = |bytes: &[u8]| {
        let bytes = &bytes[..bytes.len().min(MAX_REPLY_BYTES)];
        String::from_utf8_lossy(bytes).into_owned()
    };
    Ok(SearchReply {
        stdout: text(&output.stdout),
        stderr: text(&output.stderr),
        code: output.status.code().unwrap_or(1),
    })
}

/// `index-search VARIANT search2 ARGS...`: a variant arm's `arko_index search2`,
/// answered by the running benchmark's search service. Returns the exit code.
pub async fn search_command(variant: &str, args: &[String]) -> BenchResult<i32> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let socket = std::env::var(SOCKET_VARIABLE).map_err(|_| {
            format!("{SOCKET_VARIABLE} is not set: this arko_index runs inside the benchmark")
        })?;
        let args = match args.split_first() {
            Some((first, rest)) if first == "search2" => rest.to_vec(),
            _ => args.to_vec(),
        };
        let mut stream = tokio::net::UnixStream::connect(&socket).await?;
        let request = serde_json::to_string(&SearchRequest {
            variant: variant.to_string(),
            args,
            cwd: std::env::current_dir()?,
        })?;
        stream.write_all(format!("{request}\n").as_bytes()).await?;
        let mut reply = String::new();
        stream.read_to_string(&mut reply).await?;
        let reply: SearchReply = serde_json::from_str(reply.trim())?;
        std::io::stdout().write_all(reply.stdout.as_bytes())?;
        std::io::stderr().write_all(reply.stderr.as_bytes())?;
        Ok(reply.code)
    }
    #[cfg(not(unix))]
    {
        let _ = (variant, args);
        Err("variant arms whose index embeds need a Unix host".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, index: &[&str], search: &[&str]) -> VariantSpec {
        VariantSpec {
            name: name.into(),
            description: String::new(),
            base: None,
            index: index.iter().map(|flag| flag.to_string()).collect(),
            search: search.iter().map(|flag| flag.to_string()).collect(),
            prompt: None,
            tools_text: None,
        }
    }

    #[test]
    fn specs_validate_and_choose_their_prompt() {
        let lexical = spec("scope", &["--lexical-scope"], &[]);
        assert!(!lexical.embeds());
        assert_eq!(
            lexical.tools_paragraph(false).unwrap(),
            crate::prompts::TOOLS_INDEX
        );
        assert_eq!(lexical.build_flags(true), ["--lexical-scope"]);
        let dense = spec(
            "src",
            &["--embed", "voyage-code-4", "--embed-text", "source"],
            &[],
        );
        assert!(dense.embeds());
        assert_eq!(
            dense.tools_paragraph(true).unwrap(),
            crate::prompts::TOOLS_INDEX_HYBRID
        );
        assert_eq!(
            dense.build_flags(true),
            ["--embed", "hash", "--embed-text", "source"]
        );
        assert_eq!(
            spec("x", &["--embed=voyage-code-4"], &[]).build_flags(true),
            ["--embed=hash"]
        );
        assert!(spec("B", &[], &[]).validate().is_err());
        assert!(spec("a b", &[], &[]).validate().is_err());
        assert!(spec("r", &["--report", "x"], &[]).validate().is_err());
        let mut bad_prompt = spec("p", &[], &[]);
        bad_prompt.prompt = Some("other".into());
        assert!(bad_prompt.validate().is_err());
        // Prompt variants over another arm's index.
        let based = |base: &str, prompt: &str| VariantSpec {
            base: Some(base.into()),
            prompt: Some(prompt.into()),
            ..spec("g", &[], &[])
        };
        assert!(based("B", "guided").validate().is_ok());
        assert!(based("hyb-scope", "guided").validate().is_ok());
        assert!(based("A", "guided").validate().is_err());
        assert!(based("b", "guided").validate().is_err());
        assert!(based("g", "guided").validate().is_err());
        let with_index = VariantSpec {
            index: vec!["--lexical-scope".into()],
            ..based("B", "guided")
        };
        assert!(with_index.validate().is_err());
        let guided_b = based("B", "guided").tools_paragraph(false).unwrap();
        assert_eq!(guided_b, crate::prompts::guided_tools(false, false));
        assert!(guided_b.contains("arko_index search2") && !guided_b.contains("arko_index_dense"));
        assert!(guided_b.contains("\"lexical_rank\":1") && guided_b.contains("{rg}"));
        for placeholder in ["{ranking}", "{ranks}", "{dense_use}", "{dense_example}"] {
            assert!(!guided_b.contains(placeholder), "{placeholder}");
        }
        let guided_c = based("C", "guided").tools_paragraph(false).unwrap();
        assert!(guided_c.contains("$ arko_index_dense '"), "{guided_c}");
        let standard_c = based("C", "standard").tools_paragraph(false).unwrap();
        assert_eq!(
            standard_c,
            format!(
                "{}{}",
                crate::prompts::TOOLS_INDEX,
                crate::prompts::TOOLS_DENSE
            )
        );
        assert!(
            based("hyb-scope", "guided")
                .tools_paragraph(true)
                .unwrap()
                .contains("dense_rank")
        );
        // A base can't have a base itself.
        assert!(
            parse_specs(
                r#"[{"name": "x", "base": "B"}, {"name": "y", "base": "x", "prompt": "guided"}]"#
            )
            .is_err()
        );
        let dir = crate::test_dir("variants-file");
        let path = dir.join("variants.json");
        std::fs::write(
            &path,
            r#"[{"name": "scope", "index": ["--lexical-scope"]}, {"name": "Scope"}]"#,
        )
        .unwrap();
        assert!(load_specs(&path).is_err(), "names differ only in case");
        std::fs::write(&path, r#"[{"name": "scope", "indexes": []}]"#).unwrap();
        assert!(load_specs(&path).is_err(), "unknown field");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_example_variants_file_is_valid() {
        let specs = parse_specs(include_str!("variant-arms.json")).unwrap();
        let names: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "lex",
                "lex-scope",
                "lex-scope-desc",
                "hyb-src",
                "hyb-path",
                "hyb-scope",
                "hyb-scope-desc",
                "B-guided",
                "C-guided",
                "lex-guided",
                "hyb-scope-guided"
            ]
        );
        assert_eq!(specs.iter().filter(|spec| spec.embeds()).count(), 4);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.prompt_name() == "guided")
                .count(),
            4
        );
    }

    #[test]
    fn programs_add_search_flags_or_go_through_the_service() {
        let mut variant = PreparedVariant {
            spec: spec("plain", &[], &[]),
            mock_embeddings: false,
            repos: BTreeMap::new(),
        };
        let cli = Path::new("/opt/arko index/cli");
        assert_eq!(arko_index_program(&variant, "plain", cli).unwrap(), None);
        variant.spec = spec("text", &[], &["--text", "--limit", "5"]);
        let script = arko_index_program(&variant, "text", cli).unwrap().unwrap();
        assert!(
            script.contains("exec '/opt/arko index/cli' search2 '--text' '--limit' '5' \"$@\""),
            "{script}"
        );
        variant.repos.insert(
            "demo".into(),
            VariantIndex {
                dir: PathBuf::new(),
                commit: String::new(),
                flags: Vec::new(),
                cli_sha256: String::new(),
                dense: true,
                build_ms: 0,
                summary: String::new(),
            },
        );
        let script = arko_index_program(&variant, "text", cli).unwrap().unwrap();
        assert!(script.contains("index-search 'text' \"$@\""), "{script}");
    }

    #[cfg(unix)]
    #[test]
    fn prepare_rebuilds_what_changed_and_runs_use_the_current_indexes() {
        use std::collections::BTreeSet;
        use std::os::unix::fs::PermissionsExt;
        let dir = crate::test_dir("prepare-variants");
        let layout = Layout::new(&dir.join("out")).unwrap();
        // A fake CLI: `index2 FLAGS -- CHECKOUT DIR` logs its flags and
        // publishes a generation.
        let log = dir.join("builds.log");
        let cli = dir.join("fake_cli");
        std::fs::write(
            &cli,
            format!(
                "#!/bin/sh\necho \"$*\" >> '{}'\nfor last; do :; done\nmkdir -p \"$last/g\"\necho g > \"$last/CURRENT2\"\n",
                log.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        let builds = || {
            std::fs::read_to_string(&log)
                .unwrap_or_default()
                .lines()
                .count()
        };
        let mut repos = BTreeMap::from([("demo".to_string(), crate::mock::demo_repo(&layout.out))]);
        let specs = |lex_index: &str, lex_description: &str| {
            parse_specs(&format!(
                r#"[{{"name": "lex", "description": "{lex_description}", "index": [{lex_index}]}},
                    {{"name": "scope", "index": ["--lexical-scope"]}},
                    {{"name": "lex-guided", "base": "lex", "prompt": "guided"}},
                    {{"name": "B-guided", "base": "B", "prompt": "guided"}}]"#
            ))
            .unwrap()
        };
        let prepare = |specs: Vec<VariantSpec>, repos: &BTreeMap<String, PreparedRepo>| {
            super::prepare(PrepareVariants {
                layout: &layout,
                specs,
                repos,
                index_cli: cli.clone(),
                mock_embeddings: false,
                force: false,
            })
        };
        let only = |specs: Vec<VariantSpec>, name: &str| -> Vec<VariantSpec> {
            specs.into_iter().filter(|spec| spec.name == name).collect()
        };
        let task_repos = BTreeSet::from(["demo".to_string()]);
        let for_arms = |name: &'static str, repos: &BTreeMap<String, PreparedRepo>| {
            super::for_arms(&layout, &[Arm::Variant(name)], repos, &task_repos)
        };
        prepare(specs("", "baseline"), &repos).unwrap();
        assert_eq!(builds(), 2, "lex and scope");
        // A new description builds nothing; new flags rebuild.
        prepare(specs("", "reworded"), &repos).unwrap();
        assert_eq!(builds(), 2);
        prepare(
            only(specs("\"--lexical-scope\"", "reworded"), "lex"),
            &repos,
        )
        .unwrap();
        assert_eq!(builds(), 3);
        // lex-guided, prepared before, gets lex's index as it is now.
        let guided = for_arms("lex-guided", &repos).unwrap();
        let index = &guided["lex-guided"].repos["demo"];
        assert_eq!(index.flags, ["--lexical-scope"]);
        assert_eq!(index.dir, layout.variant_index_dir("demo", "lex"));
        assert_eq!(
            load_prepared(&layout).unwrap()["lex-guided"].repos["demo"].flags,
            Vec::<String>::new(),
            "variants.json keeps the copy from when lex-guided was prepared"
        );

        // A running run of lex-guided reads lex's index: rebuilding it waits for
        // the run. Other variants build beside it, and so do based ones.
        let run_dir = layout.run_dir("guided");
        std::fs::create_dir_all(&run_dir).unwrap();
        let manifest = crate::runner::RunManifest {
            name: "guided".into(),
            created_at: chrono::Utc::now(),
            model: crate::agents::ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap(),
            mock: true,
            shell_timeout_seconds: 120,
            shell_max_output_bytes: 16384,
            shell_max_calls: 80,
            max_input_tokens: 1000,
            attempt_timeout_seconds: None,
            grader_shell: true,
            index_cli_sha256: String::new(),
            index_cli: cli.clone(),
            arms: vec![Arm::Variant("lex-guided")],
            samples: 1,
            tasks_file: "tasks.json".into(),
            sessions: Vec::new(),
            variants: guided,
        };
        std::fs::write(
            run_dir.join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("LOCK"),
            std::os::unix::process::parent_id().to_string(),
        )
        .unwrap();
        let error = prepare(only(specs("", "reworded"), "lex"), &repos).unwrap_err();
        assert!(error.to_string().contains("in use"), "{error}");
        prepare(only(specs("\"--lexical-scope\"", "again"), "lex"), &repos).unwrap();
        prepare(only(specs("", ""), "lex-guided"), &repos).unwrap();
        let mut scope = only(specs("", ""), "scope");
        scope[0].index = vec![
            "--lexical-scope".into(),
            "--max-bytes".into(),
            "2048".into(),
        ];
        prepare(scope, &repos).unwrap();
        assert_eq!(builds(), 4, "scope rebuilt beside the run");
        std::fs::remove_file(run_dir.join("LOCK")).unwrap();

        // After `prepare` checks out another commit, a run refuses the variants
        // whose indexes are of the old one, until they are prepared again.
        repos.get_mut("demo").unwrap().commit = "def".into();
        for name in ["lex", "lex-guided"] {
            let error = for_arms(name, &repos).unwrap_err().to_string();
            assert!(
                error.contains("commit abc, but the checkout is of def")
                    && error.contains("`prepare-variants` for lex again"),
                "{error}"
            );
        }
        let b_guided = for_arms("B-guided", &repos).unwrap();
        assert_eq!(b_guided["B-guided"].repos["demo"].commit, "def");
        assert_eq!(b_guided["B-guided"].repos["demo"].dir, repos["demo"].index);
        prepare(only(specs("\"--lexical-scope\"", "again"), "lex"), &repos).unwrap();
        assert_eq!(builds(), 5);
        assert_eq!(
            for_arms("lex-guided", &repos).unwrap()["lex-guided"].repos["demo"].commit,
            "def"
        );
        set_writable(&layout.out, true);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn voyage_indexes_are_found_from_their_generation() {
        let dir = crate::test_dir("embedding-provider");
        let index = |name: &str, provider: Option<&str>| {
            let state = dir.join(name);
            std::fs::create_dir_all(state.join("index2-1")).unwrap();
            std::fs::write(state.join("CURRENT2"), "index2-1\n").unwrap();
            if let Some(provider) = provider {
                std::fs::write(
                    state.join("index2-1/dense.json"),
                    format!(r#"{{"embedder": {{"provider": "{provider}", "model": "m"}}}}"#),
                )
                .unwrap();
            }
            VariantIndex {
                dir: state,
                commit: String::new(),
                flags: Vec::new(),
                cli_sha256: String::new(),
                dense: provider.is_some(),
                build_ms: 0,
                summary: String::new(),
            }
        };
        assert_eq!(
            embedding_provider(&index("lexical", None).dir).unwrap(),
            None
        );
        let variant = |index: VariantIndex| PreparedVariant {
            spec: spec("v", &[], &[]),
            mock_embeddings: false,
            repos: BTreeMap::from([("demo".to_string(), index)]),
        };
        let demo = std::collections::BTreeSet::from(["demo".to_string()]);
        let hash = BTreeMap::from([("hash".to_string(), variant(index("hash", Some("hash"))))]);
        assert_eq!(voyage_variant(&hash, &demo).unwrap(), None);
        let voyage = BTreeMap::from([(
            "voyage".to_string(),
            variant(index("voyage", Some("voyage"))),
        )]);
        assert_eq!(
            voyage_variant(&voyage, &demo).unwrap(),
            Some(("voyage".to_string(), "demo".to_string()))
        );
        assert_eq!(voyage_variant(&voyage, &Default::default()).unwrap(), None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_search_service_runs_the_cli_with_the_variants_flags() {
        use std::os::unix::fs::PermissionsExt;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let dir = crate::test_dir("search-service");
        let layout = Layout::new(&dir.join("out")).unwrap();
        let cli = dir.join("fake_cli");
        std::fs::write(
            &cli,
            "#!/bin/sh\necho \"args: $*\"\npwd\necho note >&2\nexit 3\n",
        )
        .unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut variant = PreparedVariant {
            spec: spec("dense", &["--embed", "hash"], &["--mode", "hybrid"]),
            mock_embeddings: true,
            repos: BTreeMap::new(),
        };
        variant.repos.insert(
            "demo".into(),
            VariantIndex {
                dir: PathBuf::new(),
                commit: String::new(),
                flags: Vec::new(),
                cli_sha256: String::new(),
                dense: true,
                build_ms: 0,
                summary: String::new(),
            },
        );
        let variants = BTreeMap::from([("dense".to_string(), variant)]);
        let service = SearchService::start(&layout, &variants, &cli, Duration::from_secs(5))
            .await
            .unwrap();
        let ask = async |variant: &str| -> SearchReply {
            let mut stream = tokio::net::UnixStream::connect(service.socket())
                .await
                .unwrap();
            let request = serde_json::to_string(&SearchRequest {
                variant: variant.into(),
                args: vec!["STATE".into(), "retry delay".into()],
                cwd: dir.clone(),
            })
            .unwrap();
            stream
                .write_all(format!("{request}\n").as_bytes())
                .await
                .unwrap();
            let mut reply = String::new();
            stream.read_to_string(&mut reply).await.unwrap();
            serde_json::from_str(reply.trim()).unwrap()
        };
        let reply = ask("dense").await;
        assert_eq!(
            reply.stdout.lines().next(),
            Some("args: search2 --mode hybrid STATE retry delay")
        );
        assert!(reply.stdout.contains(dir.to_str().unwrap()), "{reply:?}");
        assert_eq!((reply.stderr.as_str(), reply.code), ("note\n", 3));
        let unknown = ask("other").await;
        assert_eq!(unknown.code, 1);
        assert!(unknown.stderr.contains("no search service"), "{unknown:?}");
        service.stop();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
