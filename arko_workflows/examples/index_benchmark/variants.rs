//! Variant arms: arms with arm B's tools whose index was built, or whose search
//! results render, with other arko_index options, for A/B tests of those options on
//! the same tasks and solvers.
//!
//! A variants file defines them; `prepare-variants` builds one index per variant
//! and repository (`indexes/<repo>@<variant>/`) and records them in
//! `variants.json`; `run --arms B,NAME,...` runs them. In a variant arm,
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
    /// `arko_index index2` flags for the variant's index, such as
    /// `["--lexical-scope"]` or `["--embed", "voyage-code-4", "--embed-text",
    /// "source"]`. Empty: the default index (arm B's, built by this CLI).
    #[serde(default)]
    pub index: Vec<String>,
    /// `search2` flags put before the solver's own on every search, such as
    /// `["--text"]` or `["--mode", "lexical"]`.
    #[serde(default)]
    pub search: Vec<String>,
    /// The solver's tool paragraph: `index` (arm B's) or `hybrid` (arm B's,
    /// saying that search2 also ranks by meaning). Default: `hybrid` when the
    /// index embeds, else `index`.
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

    /// The solver's tool paragraph (before `{rg}` is replaced).
    pub fn tools_paragraph(&self) -> BenchResult<String> {
        if let Some(text) = &self.tools_text {
            return Ok(text.clone());
        }
        let prompt =
            self.prompt
                .as_deref()
                .unwrap_or(if self.embeds() { "hybrid" } else { "index" });
        match prompt {
            "index" => Ok(crate::prompts::TOOLS_INDEX.to_string()),
            "hybrid" => Ok(crate::prompts::TOOLS_INDEX_HYBRID.to_string()),
            other => Err(format!(
                "variant {}: prompt {other:?}: index or hybrid (or give tools_text)",
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
        self.tools_paragraph()?;
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
    if let Ok(runs) = std::fs::read_dir(layout.out.join("runs")) {
        for run in runs.flatten() {
            if let Some(pid) = crate::ledger::DirLock::holder(&run.path()) {
                return Err(format!(
                    "{} is in use by process {pid}: a run may be reading these indexes; wait for it, or stop it",
                    run.path().display()
                )
                .into());
            }
        }
    }
    let cli_sha = crate::repos::sha256_file(&options.index_cli)?;
    let mut prepared = load_prepared(layout)?;
    for spec in &options.specs {
        let flags = spec.build_flags(options.mock_embeddings);
        let mut variant = match prepared.get(&spec.name) {
            Some(existing) if existing.spec == *spec => existing.clone(),
            _ => PreparedVariant {
                spec: spec.clone(),
                mock_embeddings: options.mock_embeddings,
                repos: BTreeMap::new(),
            },
        };
        variant.mock_embeddings = options.mock_embeddings;
        for (id, repo) in options.repos {
            let dir = layout.variant_index_dir(id, &spec.name);
            let current = variant.repos.get(id).is_some_and(|index| {
                index.flags == flags
                    && index.cli_sha256 == cli_sha
                    && index.commit == repo.commit
                    && dir.join("CURRENT2").exists()
            });
            if current && !options.force {
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
    Ok(prepared)
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

/// The variants among `arms`, each prepared for every repository in `repos`.
pub fn for_arms(
    layout: &Layout,
    arms: &[Arm],
    repos: &std::collections::BTreeSet<String>,
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
        let variant = prepared.get(name).ok_or_else(|| {
            format!("arm {name} is not a prepared variant: run `prepare-variants` with a variants file that defines it")
        })?;
        let missing: Vec<&str> = repos
            .iter()
            .filter(|repo| !variant.repos.contains_key(*repo))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "variant {name} has no index of {}: run `prepare-variants` for them",
                missing.join(", ")
            )
            .into());
        }
        out.insert(name.to_string(), variant.clone());
    }
    Ok(out)
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
            lexical.tools_paragraph().unwrap(),
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
            dense.tools_paragraph().unwrap(),
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
                "hyb-scope-desc"
            ]
        );
        assert_eq!(specs.iter().filter(|spec| spec.embeds()).count(), 4);
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
