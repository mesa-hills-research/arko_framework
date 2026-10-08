//! The arko_index agent benchmark: does giving coding agents arko_index make
//! them better or cheaper at real code-understanding tasks than a terminal
//! alone?
//!
//! Every agent runs in an arko workflow on the same model and settings
//! (`gpt-6-astra:xhigh`, Fast mode, ChatGPT OAuth). Solvers answer questions
//! about pinned repositories in one of three arms: (A) a terminal; (B) a
//! terminal with the arko_index CLI and a prebuilt index; (C, opt-in) B
//! plus dense search. A grader that doesn't know the arm scores each answer
//! against a verified key, beside a deterministic check of its locations.
//! Variant arms (`prepare-variants`) compare arko_index options on the same
//! tasks. See README.md.

mod agents;
mod answer;
mod dense;
mod generate;
mod ledger;
mod mock;
mod prompts;
mod report;
mod repos;
mod runner;
mod shell;
mod tasks;
mod variants;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};

pub type BenchResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A benchmark arm: what the solver has besides its terminal. Serialized as its
/// name (`"A"`, `"B"`, `"C"`, or a variant's name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arm {
    /// A terminal only.
    A,
    /// A terminal and the arko_index CLI with a prebuilt index.
    B,
    /// B plus dense (embedding) search.
    C,
    /// A variant arm (`prepare-variants`): B's tools over an index built, or
    /// searched, with other arko_index options. Holds the variant's name.
    Variant(&'static str),
}

impl Arm {
    pub const ALL: [Arm; 3] = [Arm::A, Arm::B, Arm::C];

    pub fn name(self) -> &'static str {
        match self {
            Arm::A => "A",
            Arm::B => "B",
            Arm::C => "C",
            Arm::Variant(name) => name,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Arm::A => "terminal",
            Arm::B => "terminal + arko_index",
            Arm::C => "terminal + arko_index + dense",
            Arm::Variant(_) => "terminal + arko_index variant",
        }
    }

    /// The arm named `name`: A, B or C (any case), else a variant. A variant's
    /// name is 1 to 32 letters, digits, `-` or `_`.
    pub fn parse(name: &str) -> BenchResult<Arm> {
        let name = name.trim();
        Ok(match name.to_ascii_uppercase().as_str() {
            "A" => Arm::A,
            "B" => Arm::B,
            "C" => Arm::C,
            _ => {
                if name.is_empty()
                    || name.len() > 32
                    || !name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                {
                    return Err(format!(
                        "arm {name:?}: A, B, C or a variant's name (1 to 32 letters, digits, - or _)"
                    )
                    .into());
                }
                Arm::Variant(intern(name))
            }
        })
    }

    pub fn parse_list(text: &str) -> BenchResult<Vec<Arm>> {
        let mut arms = Vec::new();
        for part in text.split(',').filter(|part| !part.trim().is_empty()) {
            let arm = Arm::parse(part)?;
            if !arms.contains(&arm) {
                arms.push(arm);
            }
        }
        if arms.is_empty() {
            return Err("no arms given".into());
        }
        arms.sort();
        Ok(arms)
    }
}

/// A variant's name, kept for the rest of the process (a handful per run).
fn intern(name: &str) -> &'static str {
    static NAMES: std::sync::OnceLock<
        parking_lot::Mutex<std::collections::BTreeSet<&'static str>>,
    > = std::sync::OnceLock::new();
    let mut names = NAMES.get_or_init(Default::default).lock();
    if let Some(known) = names.get(name) {
        return known;
    }
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.insert(name);
    name
}

impl std::fmt::Display for Arm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for Arm {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Arm {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Arm::parse(&name).map_err(serde::de::Error::custom)
    }
}

/// Writes `bytes` to `path` through a temporary file and a rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> BenchResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension().and_then(|e| e.to_str()).unwrap_or(""),
        std::process::id()
    ));
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(())
}

/// A fresh directory for a test, under the system's temporary directory.
#[cfg(test)]
pub fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "index-benchmark-test-{name}-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, Parser)]
#[command(
    name = "index_benchmark",
    about = "Benchmark coding agents with and without arko_index on code-understanding questions"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fetch the pinned repositories, make them read-only, and build their
    /// arko_index indexes (timed, outside the benchmark).
    Prepare(PrepareArgs),
    /// Build the indexes of variant arms from a variants file: arm B's tools
    /// over an index built, or searched, with other arko_index options. Runs
    /// beside `generate`.
    PrepareVariants(PrepareVariantsArgs),
    /// Write questions with a writer agent, check each with a verifier agent,
    /// and assemble the task set from the seed tasks and the questions both
    /// agree on.
    Generate(GenerateArgs),
    /// Assemble the task set again from the seed tasks and the verified
    /// questions, without running agents.
    Assemble(AssembleArgs),
    /// Check a task set against the checkouts and print its counts.
    Validate(ValidateArgs),
    /// Run the benchmark: every task in every arm, graded as answers come
    /// in. Resumable; stops gracefully on Ctrl-C, SIGTERM or a STOP file.
    Run(RunArgs),
    /// Write the report of a run (report.md, report.json, attempts.csv).
    Report(ReportArgs),
    /// Arm C's dense search, as `arko_index_dense` runs it in the agents'
    /// shells (talks to the running benchmark).
    #[command(hide = true)]
    DenseQuery(DenseQueryArgs),
    /// A variant arm's `arko_index search2` when its index embeds, as the
    /// agents' shells run it (talks to the running benchmark).
    #[command(hide = true)]
    IndexSearch(IndexSearchArgs),
}

#[derive(Debug, Args)]
struct OutArgs {
    /// The benchmark's directory: checkouts, indexes, tasks and runs.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct ModelArgs {
    /// The model every agent runs on, `id` or `id:effort`.
    #[arg(long, default_value = "gpt-6-astra:xhigh")]
    model: String,
    /// Standard processing instead of Fast mode.
    #[arg(long)]
    no_fast: bool,
    /// The Codex home whose `auth.json` holds the ChatGPT login (default:
    /// `$CODEX_HOME`, else `~/.codex`). Repeat it for failover between
    /// ChatGPT accounts, first preferred.
    #[arg(long)]
    codex_home: Vec<PathBuf>,
    /// Dry run: a local mock model answers instead (no model calls). Tool
    /// calls run for real.
    #[arg(long)]
    mock: bool,
    /// The mock's reasoning tokens per model call, for its token estimates.
    #[arg(long, default_value_t = 1500)]
    mock_reasoning_tokens: usize,
    /// Shell commands the mock's solvers run before answering (searches
    /// and reads in turn), for its token estimates.
    #[arg(long, default_value_t = 12)]
    mock_steps: usize,
    /// Milliseconds each mock reply takes (to rehearse stops and resumes).
    #[arg(long, default_value_t = 0)]
    mock_delay_ms: u64,
}

#[derive(Debug, Args)]
struct PrepareArgs {
    #[command(flatten)]
    out: OutArgs,
    /// A repository manifest (default: the built-in one, repos.json).
    #[arg(long)]
    repos: Option<PathBuf>,
    /// Only these repository ids (comma-separated).
    #[arg(long)]
    only: Option<String>,
    /// Fetch a repository from elsewhere: `ID=PATH_OR_URL` (repeatable),
    /// for example `--source arko=/home/me/arko_framework`.
    #[arg(long)]
    source: Vec<String>,
    /// The arko_index CLI (default: `arko_index_cli` next to this binary).
    #[arg(long)]
    index_cli: Option<PathBuf>,
    /// Fetch and index again even when prepared.
    #[arg(long)]
    force: bool,
    /// Also build arm C's dense index (Voyage `voyage-code-4`; needs
    /// VOYAGE_API_KEY, or --mock-embeddings).
    #[arg(long)]
    dense: bool,
    /// Dense index with an offline hashing embedder instead of Voyage (for
    /// dry runs).
    #[arg(long)]
    mock_embeddings: bool,
    #[arg(long, default_value = "voyage-code-4")]
    dense_model: String,
    #[arg(long, default_value_t = 1024)]
    dense_dimensions: usize,
}

#[derive(Debug, Args)]
struct GenerateArgs {
    #[command(flatten)]
    out: OutArgs,
    #[command(flatten)]
    model: ModelArgs,
    /// Only these repository ids (comma-separated).
    #[arg(long)]
    only: Option<String>,
    /// Only these kinds (comma-separated: definition, callers, behaviour,
    /// impact).
    #[arg(long)]
    kinds: Option<String>,
    /// Questions wanted per repository and kind in the task set.
    #[arg(long, default_value_t = 9)]
    per_kind: usize,
    /// Questions written per question wanted (verification drops some).
    #[arg(long, default_value_t = 1.5)]
    oversample: f64,
    /// Agents running at once.
    #[arg(long, default_value_t = 24)]
    concurrency: usize,
    /// The task set to write (default: OUT/tasks.json).
    #[arg(long)]
    tasks: Option<PathBuf>,
    /// The writer's and verifier's tools: `terminal` (arm A's) or `index`
    /// (arm B's, with arko_index). Terminal by default, so the questions
    /// aren't shaped by the tool under test.
    #[arg(long, default_value = "terminal")]
    writer_tools: String,
    /// The arko_index CLI (default: `arko_index_cli` next to this binary).
    #[arg(long)]
    index_cli: Option<PathBuf>,
    #[command(flatten)]
    shell: ShellArgs,
}

#[derive(Debug, Args)]
struct AssembleArgs {
    #[command(flatten)]
    out: OutArgs,
    #[arg(long, default_value_t = 9)]
    per_kind: usize,
    #[arg(long)]
    tasks: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    #[command(flatten)]
    out: OutArgs,
    /// The task set (default: OUT/tasks.json; `seed` for the built-in seed
    /// tasks).
    #[arg(long)]
    tasks: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
struct ShellArgs {
    /// Output of one shell command returned to an agent, in bytes.
    #[arg(long, default_value_t = 16_384)]
    max_output_bytes: usize,
    /// Seconds a shell command may run.
    #[arg(long, default_value_t = 120)]
    command_timeout: u64,
    /// Shell commands per agent run; after that the agent is asked to
    /// answer.
    #[arg(long, default_value_t = 80)]
    max_shell_calls: usize,
}

impl ShellArgs {
    fn settings(&self) -> shell::ShellSettings {
        shell::ShellSettings {
            timeout: Duration::from_secs(self.command_timeout.max(1)),
            max_output_bytes: self.max_output_bytes.max(1024),
            max_calls: self.max_shell_calls.max(1),
        }
    }
}

#[derive(Debug, Args)]
struct RunArgs {
    #[command(flatten)]
    out: OutArgs,
    #[command(flatten)]
    model: ModelArgs,
    /// The run's name: its state is OUT/runs/NAME, and running again with
    /// the same name resumes it.
    #[arg(long, default_value = "main")]
    name: String,
    /// The task set (default: OUT/tasks.json; `seed` for the built-in seed
    /// tasks).
    #[arg(long)]
    tasks: Option<PathBuf>,
    /// The arms: A (terminal), B (+ arko_index), C (+ dense; needs
    /// `prepare --dense`), or variant names (`prepare-variants`).
    #[arg(long, default_value = "A,B")]
    arms: String,
    /// Runs of each task in each arm.
    #[arg(long, default_value_t = 1)]
    samples: u32,
    /// Agents running at once (solvers and graders).
    #[arg(long, default_value_t = 24)]
    concurrency: usize,
    /// Only these repository ids (comma-separated).
    #[arg(long)]
    only_repos: Option<String>,
    /// Only these kinds (comma-separated).
    #[arg(long)]
    only_kinds: Option<String>,
    /// Only the first N tasks (in the run's shuffled order), for a pilot.
    #[arg(long)]
    limit: Option<usize>,
    #[command(flatten)]
    shell: ShellArgs,
    /// Input tokens (all calls, cached included) after which an agent run is
    /// stopped and counted as failed.
    #[arg(long, default_value_t = 8_000_000)]
    max_input_tokens: usize,
    /// The arko_index CLI (default: `arko_index_cli` next to this binary).
    #[arg(long)]
    index_cli: Option<PathBuf>,
    /// Minutes an agent run may work (waits for usage limits excluded)
    /// before it is stopped and counted as failed; 0 for no limit.
    #[arg(long, default_value_t = 60)]
    attempt_timeout_minutes: u64,
    /// Attempts per task and arm that end for reasons outside the agent (a
    /// stop, a lost login) before it is given up.
    #[arg(long, default_value_t = 3)]
    max_attempts: u32,
    /// Don't grade (grade later with `--grade-only`).
    #[arg(long)]
    no_grade: bool,
    /// Only grade answers not graded yet.
    #[arg(long)]
    grade_only: bool,
    /// Graders judge from the key alone, without a shell to check the code.
    #[arg(long)]
    grader_without_shell: bool,
    /// Seed of the task order.
    #[arg(long, default_value_t = 20261007)]
    seed: u64,
}

#[derive(Debug, Args)]
struct ReportArgs {
    #[command(flatten)]
    out: OutArgs,
    #[arg(long, default_value = "main")]
    name: String,
    /// The task set (default: the one the run used).
    #[arg(long)]
    tasks: Option<PathBuf>,
    /// Bootstrap resamples for the confidence intervals.
    #[arg(long, default_value_t = 4000)]
    resamples: usize,
}

#[derive(Debug, Args)]
struct PrepareVariantsArgs {
    #[command(flatten)]
    out: OutArgs,
    /// The variants file: a JSON list of variants (see README.md).
    #[arg(long)]
    variants: PathBuf,
    /// Only these variants (comma-separated names).
    #[arg(long)]
    names: Option<String>,
    /// Only these repository ids (comma-separated).
    #[arg(long)]
    only: Option<String>,
    /// The arko_index CLI (default: `arko_index_cli` next to this binary).
    #[arg(long)]
    index_cli: Option<PathBuf>,
    /// Embed with the offline hashing embedder instead of the variants' models
    /// (for dry runs).
    #[arg(long)]
    mock_embeddings: bool,
    /// Build again even when built with the same flags, CLI and commit.
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
struct IndexSearchArgs {
    /// The variant arm.
    variant: String,
    /// `search2` and its arguments.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

#[derive(Debug, Args)]
struct DenseQueryArgs {
    /// Results to print.
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// The query.
    #[arg(required = true, trailing_var_arg = true)]
    query: Vec<String>,
}

fn split_list(text: &Option<String>) -> Option<Vec<String>> {
    text.as_ref().map(|text| {
        text.split(',')
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect()
    })
}

fn parse_kinds(text: &Option<String>) -> BenchResult<Option<Vec<tasks::TaskKind>>> {
    let Some(list) = split_list(text) else {
        return Ok(None);
    };
    list.iter()
        .map(|kind| {
            tasks::TaskKind::parse(kind).ok_or_else(|| format!("unknown kind {kind}").into())
        })
        .collect::<BenchResult<Vec<_>>>()
        .map(Some)
}

/// The task set at `path`, `seed` for the built-in seed tasks.
fn load_task_set(
    layout: &repos::Layout,
    path: &Option<PathBuf>,
) -> BenchResult<(Vec<tasks::BenchTask>, PathBuf)> {
    match path {
        Some(path) if path.as_os_str() == "seed" => Ok((
            tasks::parse_tasks(generate::SEED_TASKS)?,
            PathBuf::from("seed"),
        )),
        Some(path) => Ok((tasks::load_tasks(path)?, path.clone())),
        None => {
            let path = layout.default_tasks();
            if !path.exists() {
                return Err(format!(
                    "{} doesn't exist: run `generate` (or `assemble`), or pass --tasks seed for the seed tasks",
                    path.display()
                )
                .into());
            }
            Ok((tasks::load_tasks(&path)?, path))
        }
    }
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let cli = Cli::parse();
    if let Err(error) = run(cli).await {
        eprintln!("index_benchmark: {error}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> BenchResult<()> {
    match cli.command {
        Command::Prepare(args) => prepare(args).await,
        Command::PrepareVariants(args) => prepare_variants(args),
        Command::Generate(args) => generate_command(args).await,
        Command::Assemble(args) => {
            let layout = repos::Layout::new(&args.out.out)?;
            let path = args.tasks.unwrap_or_else(|| layout.default_tasks());
            let tasks = generate::assemble(&layout, args.per_kind, &path)?;
            print_counts(&layout, &tasks)?;
            Ok(())
        }
        Command::Validate(args) => {
            let layout = repos::Layout::new(&args.out.out)?;
            let (tasks, path) = load_task_set(&layout, &args.tasks)?;
            let prepared = repos::load_prepared(&layout)?;
            let roots: HashMap<String, PathBuf> = prepared
                .iter()
                .map(|(id, repo)| (id.clone(), repo.checkout.clone()))
                .collect();
            let issues = tasks::validate_against_checkout(&tasks, &roots);
            println!("{}: {} tasks", path.display(), tasks.len());
            print_counts(&layout, &tasks)?;
            for issue in &issues {
                println!("{}: {}", issue.task, issue.message);
            }
            if issues.is_empty() {
                println!("every location is in its checkout");
                Ok(())
            } else {
                Err(format!("{} problems", issues.len()).into())
            }
        }
        Command::Run(args) => run_command(args).await,
        Command::Report(args) => {
            let layout = repos::Layout::new(&args.out.out)?;
            let run_dir = layout.run_dir(&args.name);
            let manifest = runner::RunManifest::load(&run_dir)?;
            let tasks_path = args.tasks.or_else(|| Some(manifest.tasks_file.clone()));
            let (tasks, _) = load_task_set(&layout, &tasks_path)?;
            let prepared = repos::load_prepared(&layout)?;
            let markdown = report::write(&run_dir, &manifest, &tasks, &prepared, args.resamples)?;
            println!("{markdown}");
            Ok(())
        }
        Command::DenseQuery(args) => dense::query_command(&args.query.join(" "), args.limit).await,
        Command::IndexSearch(args) => {
            let code = variants::search_command(&args.variant, &args.args).await?;
            std::process::exit(code);
        }
    }
}

fn print_counts(layout: &repos::Layout, tasks: &[tasks::BenchTask]) -> BenchResult<()> {
    let prepared = repos::load_prepared(layout)?;
    let mut languages: HashMap<String, String> = repos::load_manifest(None)?
        .into_iter()
        .map(|repo| (repo.id, repo.language))
        .collect();
    for (id, repo) in prepared {
        languages.insert(id, repo.spec.language);
    }
    println!("{}", tasks::count_table(tasks, &languages));
    Ok(())
}

fn prepare_variants(args: PrepareVariantsArgs) -> BenchResult<()> {
    let layout = repos::Layout::new(&args.out.out)?;
    let mut specs = variants::load_specs(&args.variants)?;
    if let Some(names) = split_list(&args.names) {
        for name in &names {
            if !specs.iter().any(|spec| &spec.name == name) {
                return Err(
                    format!("{} defines no variant {name}", args.variants.display()).into(),
                );
            }
        }
        specs.retain(|spec| names.contains(&spec.name));
    }
    let mut prepared = repos::load_prepared(&layout)?;
    if prepared.is_empty() {
        return Err("no prepared repositories: run `prepare` first".into());
    }
    if let Some(only) = split_list(&args.only) {
        for id in &only {
            if !prepared.contains_key(id) {
                return Err(format!("repository {id} is not prepared").into());
            }
        }
        prepared.retain(|id, _| only.contains(id));
    }
    let index_cli = repos::find_index_cli(args.index_cli.as_deref())?;
    let built = variants::prepare(variants::PrepareVariants {
        layout: &layout,
        specs: specs.clone(),
        repos: &prepared,
        index_cli,
        mock_embeddings: args.mock_embeddings,
        force: args.force,
    })?;
    println!("| variant | repo | index2 flags | dense | build (s) |\n|---|---|---|---|---|");
    for spec in &specs {
        let Some(variant) = built.get(&spec.name) else {
            continue;
        };
        for (id, index) in &variant.repos {
            if !prepared.contains_key(id) {
                continue;
            }
            println!(
                "| {} | {id} | {} | {} | {:.1} |",
                spec.name,
                index.flags.join(" "),
                if index.dense { "yes" } else { "-" },
                index.build_ms as f64 / 1000.0
            );
        }
    }
    Ok(())
}

async fn prepare(args: PrepareArgs) -> BenchResult<()> {
    let layout = repos::Layout::new(&args.out.out)?;
    let mut manifest = repos::load_manifest(args.repos.as_deref())?;
    if let Some(only) = split_list(&args.only) {
        for id in &only {
            if !manifest.iter().any(|repo| &repo.id == id) {
                return Err(format!("unknown repository {id}").into());
            }
        }
        manifest.retain(|repo| only.contains(&repo.id));
    }
    let mut sources = HashMap::new();
    for source in &args.source {
        let (id, location) = source
            .split_once('=')
            .ok_or_else(|| format!("--source {source}: expected ID=PATH_OR_URL"))?;
        sources.insert(id.to_string(), location.to_string());
    }
    let index_cli = repos::find_index_cli(args.index_cli.as_deref())?;
    let dense = args.dense.then(|| dense::DenseOptions {
        mock: args.mock_embeddings,
        model: args.dense_model.clone(),
        dimensions: args.dense_dimensions,
    });
    let prepared = repos::prepare(repos::PrepareOptions {
        layout: &layout,
        repos: manifest,
        sources,
        index_cli,
        force: args.force,
        dense,
    })
    .await?;
    println!("| repo | commit | files | MB | index build (s) | dense |\n|---|---|---|---|---|---|");
    for (id, repo) in &prepared {
        println!(
            "| {id} | {} | {} | {:.1} | {:.1} | {} |",
            &repo.commit[..repo.commit.len().min(10)],
            repo.files,
            repo.bytes as f64 / 1e6,
            repo.index_build_ms as f64 / 1000.0,
            repo.dense
                .as_ref()
                .map_or("-".to_string(), |dense| format!("{} chunks", dense.chunks))
        );
    }
    Ok(())
}

/// The mock model, when `--mock` is given: its URL for the client.
async fn start_mock(args: &ModelArgs) -> BenchResult<Option<mock::MockServer>> {
    if !args.mock {
        return Ok(None);
    }
    Ok(Some(
        mock::MockServer::start(mock::MockSettings {
            reasoning_tokens: args.mock_reasoning_tokens,
            solver_steps: args.mock_steps,
            delay: Duration::from_millis(args.mock_delay_ms),
        })
        .await?,
    ))
}

async fn generate_command(args: GenerateArgs) -> BenchResult<()> {
    let layout = repos::Layout::new(&args.out.out)?;
    let model = agents::ModelSettings::parse(&args.model.model, !args.model.no_fast)?;
    let mock = start_mock(&args.model).await?;
    let client =
        agents::build_client(&args.model.codex_home, mock.as_ref().map(|mock| mock.url()))?;
    let prepared = repos::load_prepared(&layout)?;
    let mut repo_ids: Vec<String> = prepared.keys().cloned().collect();
    if let Some(only) = split_list(&args.only) {
        repo_ids.retain(|id| only.contains(id));
    }
    if repo_ids.is_empty() {
        return Err("no prepared repositories: run `prepare` first".into());
    }
    let kinds = parse_kinds(&args.kinds)?.unwrap_or_else(|| tasks::TaskKind::ALL.to_vec());
    let writer_arm = match args.writer_tools.as_str() {
        "terminal" => Arm::A,
        "index" => Arm::B,
        other => return Err(format!("--writer-tools {other}: terminal or index").into()),
    };
    let _lock = ledger::DirLock::acquire(&layout.tasks_dir(), "a `generate`")?;
    if let Some(pid) = ledger::DirLock::holder(&layout.out.join("prepare")) {
        return Err(format!("`prepare` is running (pid {pid}); wait for it").into());
    }
    if writer_arm == Arm::B {
        let index_cli = repos::find_index_cli(args.index_cli.as_deref())?;
        repos::setup_arm_bins(&layout, &[Arm::B], &index_cli, &Default::default())?;
    } else {
        // Arm A's stubs need no CLI; the path is not used.
        repos::setup_arm_bins(&layout, &[Arm::A], Path::new(""), &Default::default())?;
    }
    let path = args.tasks.clone().unwrap_or_else(|| layout.default_tasks());
    let tasks = generate::generate(generate::GenerateOptions {
        layout: layout.clone(),
        prepared,
        repo_ids,
        kinds,
        per_kind: args.per_kind,
        oversample: args.oversample,
        concurrency: args.concurrency.max(1),
        model,
        client,
        shell: args.shell.settings(),
        tasks_path: path,
        mock: args.model.mock,
        writer_arm,
    })
    .await?;
    print_counts(&layout, &tasks)?;
    if let Some(mock) = mock {
        mock.shutdown();
    }
    Ok(())
}

async fn run_command(args: RunArgs) -> BenchResult<()> {
    let layout = repos::Layout::new(&args.out.out)?;
    let arms = Arm::parse_list(&args.arms)?;
    let model = agents::ModelSettings::parse(&args.model.model, !args.model.no_fast)?;
    let (mut task_set, tasks_path) = load_task_set(&layout, &args.tasks)?;
    if let Some(repos) = split_list(&args.only_repos) {
        task_set.retain(|task| repos.contains(&task.repo));
    }
    if let Some(kinds) = parse_kinds(&args.only_kinds)? {
        task_set.retain(|task| kinds.contains(&task.kind));
    }
    let prepared = repos::load_prepared(&layout)?;
    let roots: HashMap<String, PathBuf> = prepared
        .iter()
        .map(|(id, repo)| (id.clone(), repo.checkout.clone()))
        .collect();
    let issues = tasks::validate_against_checkout(&task_set, &roots);
    if !issues.is_empty() {
        for issue in issues.iter().take(20) {
            eprintln!("{}: {}", issue.task, issue.message);
        }
        return Err(format!(
            "{} problems in the task set (run `validate` for all of them)",
            issues.len()
        )
        .into());
    }
    let task_repos: std::collections::BTreeSet<String> =
        task_set.iter().map(|task| task.repo.clone()).collect();
    let variants = variants::for_arms(&layout, &arms, &prepared, &task_repos)?;
    // Arm C, and variants based on it, search with arm C's dense index.
    if arms.contains(&Arm::C) || variants.values().any(|variant| variant.spec.dense_tool()) {
        let missing: Vec<&str> = task_set
            .iter()
            .filter(|task| {
                prepared
                    .get(&task.repo)
                    .is_none_or(|repo| repo.dense.is_none())
            })
            .map(|task| task.repo.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "arm C needs a dense index; run `prepare --dense` for: {}",
                missing.join(", ")
            )
            .into());
        }
    }
    let _lock =
        ledger::DirLock::acquire(&layout.run_dir(&args.name), &format!("run {}", args.name))?;
    if let Some(pid) = ledger::DirLock::holder(&layout.out.join("prepare")) {
        return Err(format!("`prepare` is running (pid {pid}); wait for it").into());
    }
    if !variants.is_empty()
        && let Some(pid) = ledger::DirLock::holder(&layout.variants_lock_dir())
    {
        return Err(format!("`prepare-variants` is running (pid {pid}); wait for it").into());
    }
    let index_cli = repos::find_index_cli(args.index_cli.as_deref())?;
    let index_cli_sha256 = repos::sha256_file(&index_cli)?;
    for task in &task_set {
        if let Some(repo) = prepared.get(&task.repo)
            && repo.index_cli_sha256 != index_cli_sha256
            && arms.iter().any(|arm| matches!(arm, Arm::B | Arm::C))
        {
            eprintln!(
                "warning: the index of {} was built by another arko_index CLI than {}; run `prepare` again to rebuild it",
                task.repo,
                index_cli.display()
            );
            break;
        }
    }
    for (name, variant) in &variants {
        if variant
            .repos
            .iter()
            .any(|(id, index)| task_repos.contains(id) && index.cli_sha256 != index_cli_sha256)
        {
            eprintln!(
                "warning: variant {name}'s indexes were built by another arko_index CLI than {}; run `prepare-variants` again to rebuild them",
                index_cli.display()
            );
        }
    }
    let mut bin_arms = arms.clone();
    if !bin_arms.contains(&Arm::A) {
        // The graders' shells are arm A's.
        bin_arms.push(Arm::A);
    }
    repos::setup_arm_bins(&layout, &bin_arms, &index_cli, &variants)?;
    let mock = start_mock(&args.model).await?;
    let client =
        agents::build_client(&args.model.codex_home, mock.as_ref().map(|mock| mock.url()))?;
    let settings = runner::RunSettings {
        name: args.name.clone(),
        arms,
        samples: args.samples.max(1),
        concurrency: args.concurrency.max(1),
        model,
        shell: args.shell.settings(),
        max_input_tokens: args.max_input_tokens,
        attempt_timeout: (args.attempt_timeout_minutes > 0)
            .then(|| Duration::from_secs(args.attempt_timeout_minutes * 60)),
        max_attempts: args.max_attempts.max(1),
        grade: !args.no_grade,
        grade_only: args.grade_only,
        grader_shell: !args.grader_without_shell,
        seed: args.seed,
        limit: args.limit,
        mock: args.model.mock,
        tasks_file: tasks_path,
        index_cli,
        index_cli_sha256,
        variants,
    };
    let summary = runner::run(runner::RunInputs {
        layout: layout.clone(),
        settings,
        tasks: task_set,
        prepared,
        client,
    })
    .await?;
    println!("{summary}");
    if let Some(mock) = mock {
        mock.shutdown();
    }
    Ok(())
}
