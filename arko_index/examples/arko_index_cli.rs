//! Run from the workspace root; see arko_index/README.md.
use arko_ai::{EmbeddingClient, EmbeddingModelDetails, EmbeddingProviderIdentifier, LLMClient};
use arko_index::{
    backend::{
        ArkoBackend, Backend, Limits, ModelReply, ModelRequest, VectorReply, embedding_limits,
    },
    *,
};
use clap::{Parser, Subcommand};
use std::{path::PathBuf, sync::Arc, time::Duration};
#[path = "support/search_judge.rs"]
mod search_judge;
use search_judge::JudgeArgs;
type AnyResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    /// A Codex auth.json (access_token, refresh_token and account_id at the top
    /// level or in a tokens object). Every token refresh reads it again first,
    /// and rotated tokens are saved to it at once.
    #[arg(long)]
    oauth_file: Option<PathBuf>,
    #[arg(long, default_value_t = 6)]
    batch_count: usize,
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
    /// Opt-in: cut a chunk-model or judge response off after this many
    /// seconds without data. Off by default: models can reason silently for
    /// a long time (GPT Pro models for over an hour), so a short value cuts
    /// healthy responses off. Make it generous.
    #[arg(long)]
    stream_idle_timeout_secs: Option<u64>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Build {
        root: PathBuf,
        /// The index name; defaults to the root directory's name.
        #[arg(long)]
        name: Option<String>,
    },
    Update {
        index: String,
    },
    Resume {
        job: String,
    },
    Status {
        job: String,
    },
    List,
    Search {
        index: String,
        query: String,
        #[arg(long, conflicts_with = "semantic")]
        lexical: bool,
        /// Use vector-only retrieval instead of hybrid retrieval.
        #[arg(long)]
        semantic: bool,
        /// Hide source fragments shorter than this many UTF-8 bytes; 0 keeps all.
        #[arg(long, default_value_t = 40)]
        min_chunk_bytes: usize,
        /// Return this many complete results from the final combined ranking.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// The most affected paths a rebuild notice lists.
        #[arg(long, default_value_t = 100)]
        rebuild_file_limit: usize,
        #[command(flatten)]
        judging: JudgeArgs,
    },
    Read {
        index: String,
        generation: String,
        ids: Vec<String>,
    },
    Watch {
        index: String,
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
        /// Seconds between full scans that catch missed file-system events;
        /// 0 scans only at start and after events.
        #[arg(long, default_value_t = 30)]
        reconcile_secs: u64,
    },
}
impl Command {
    /// Whether the command sends model or embedding requests, or counts
    /// tokens, and so needs the provider backend.
    fn needs_backend(&self) -> bool {
        !matches!(self, Self::Status { .. } | Self::List | Self::Read { .. })
    }
}
/// The backend of commands that make no provider requests: `list`, `status`
/// and `read` need neither credentials nor a tokenizer.
struct NoProviders {
    profile: EmbeddingProfile,
    limits: Limits,
}
impl NoProviders {
    fn new(profile: EmbeddingProfile) -> Self {
        let limits =
            EmbeddingModelDetails::get(EmbeddingProviderIdentifier::Voyage, &profile.model)
                .ok()
                .and_then(|details| embedding_limits(&details).ok())
                .unwrap_or(Limits {
                    item_tokens: 32_000,
                    batch_tokens: 120_000,
                    batch_items: 1_000,
                });
        Self { profile, limits }
    }
}
fn no_providers() -> Error {
    Error::Invalid("this command makes no provider requests".into())
}
#[async_trait::async_trait]
impl Backend for NoProviders {
    async fn chunk(&self, _: ModelRequest) -> Result<ModelReply> {
        Err(no_providers())
    }
    async fn embed(&self, _: Vec<String>, _: bool) -> Result<VectorReply> {
        Err(no_providers())
    }
    fn tokens(&self, _: &str) -> Result<usize> {
        Err(no_providers())
    }
    fn limits(&self) -> Limits {
        self.limits
    }
    fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }
    fn output_tokenizer(&self) -> String {
        format!(
            "voyageai/{}@{}",
            self.profile.model, self.profile.tokenizer_revision
        )
    }
}
/// The default index name: the root directory's name.
fn default_name(root: &std::path::Path) -> String {
    std::fs::canonicalize(root)
        .ok()
        .and_then(|root| {
            root.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "index".into())
}
/// Credentials from an OAuth file when given, kept in step with the file,
/// else from the environment.
fn load_client(path: Option<PathBuf>) -> AnyResult<LLMClient> {
    match path {
        Some(path) => Ok(arko_ai::CodexAuthFile::load(path)?.client()?),
        None => Ok(LLMClient::from_env()?),
    }
}
async fn observe(index: &ArkoIndex, job: &str) -> AnyResult<()> {
    let mut cursor = 0;
    loop {
        for event in index.events(job, cursor, 100).await? {
            cursor = event.sequence;
            println!("{}", serde_json::to_string(&event)?);
        }
        let status = index.status(job).await?;
        match status.phase.as_str() {
            "ready" => return Ok(()),
            "paused" | "cancelled" | "failed" | "blocked" => {
                return Err(
                    format!("{}: {}", status.phase, status.problem.unwrap_or_default()).into(),
                );
            }
            _ => {}
        }
        tokio::select! {_=tokio::time::sleep(Duration::from_millis(250))=>{},_=tokio::signal::ctrl_c()=>{index.pause(job).await?;}}
    }
}
#[tokio::main]
async fn main() -> AnyResult<()> {
    // Warnings from every crate, and the index's progress (such as the
    // updates `watch` starts); RUST_LOG overrides it.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,arko_index=info"),
    )
    .init();
    let args = Args::parse();
    let config = Config {
        batch_count: args.batch_count,
        concurrency: args.concurrency,
        stream_idle_timeout_ms: args
            .stream_idle_timeout_secs
            .map(|seconds| seconds.saturating_mul(1000)),
        ..Default::default()
    };
    // Only commands that call providers or count tokens load credentials and
    // the tokenizer.
    let (client, backend): (Option<LLMClient>, Arc<dyn Backend>) = if args.command.needs_backend() {
        let client = load_client(args.oauth_file.clone())?;
        let backend = ArkoBackend::new(
            client.clone(),
            EmbeddingClient::from_env(),
            config.embedding.clone(),
        )
        .await?;
        (Some(client), Arc::new(backend))
    } else {
        (None, Arc::new(NoProviders::new(config.embedding.clone())))
    };
    let index = ArkoIndex::open(args.database, backend, config).await?;
    match args.command {
        Command::Build { root, name } => {
            let name = name.unwrap_or_else(|| default_name(&root));
            let job = index.start_build(BuildSpec::new(name, root)).await?;
            println!("{}", serde_json::to_string(&job)?);
            observe(&index, &job.job).await?;
        }
        Command::Update { index: id } => {
            let job = index.start_update(&id).await?;
            println!("{}", serde_json::to_string(&job)?);
            observe(&index, &job.job).await?;
        }
        Command::Resume { job } => {
            index.resume(&job).await?;
            observe(&index, &job).await?;
        }
        Command::Status { job } => println!(
            "{}",
            serde_json::to_string_pretty(&index.status(&job).await?)?
        ),
        Command::List => println!("{}", serde_json::to_string_pretty(&index.list().await?)?),
        Command::Search {
            index: id,
            query,
            lexical,
            semantic,
            min_chunk_bytes,
            top_k,
            rebuild_file_limit,
            judging,
        } => {
            let options = SearchOptions {
                mode: if lexical {
                    SearchMode::Lexical
                } else if semantic {
                    SearchMode::Semantic
                } else {
                    SearchMode::Hybrid
                },
                rebuild_file_limit: Some(rebuild_file_limit),
                min_chunk_bytes,
                top_k: Some(top_k),
                ..Default::default()
            };
            let result = if judging.judge {
                let client = client.as_ref().ok_or("search needs a client")?;
                let mut judge = judging.build_judge(client, args.oauth_file.as_deref())?;
                if let Some(seconds) = args.stream_idle_timeout_secs {
                    judge = judge.with_stream_idle_timeout(Duration::from_secs(seconds));
                }
                index
                    .search_with_judge(&id, &query, options, &judge, judging.options())
                    .await
            } else {
                index.search(&id, &query, options).await
            };
            println!("{}", serde_json::to_string_pretty(&result?)?);
        }
        Command::Read {
            index: id,
            generation,
            ids,
        } => println!(
            "{}",
            serde_json::to_string_pretty(&index.read_chunks(&id, &generation, &ids, 0).await?)?
        ),
        Command::Watch {
            index: id,
            debounce_ms,
            reconcile_secs,
        } => {
            let reconcile = (reconcile_secs > 0).then(|| Duration::from_secs(reconcile_secs));
            let watcher = index
                .watch(
                    &id,
                    WatchOptions::new(Duration::from_millis(debounce_ms)).with_reconcile(reconcile),
                )
                .await?;
            // Each watcher event as a JSON line: scans, and the updates it
            // starts, resumes or leaves paused or blocked.
            let mut events = watcher.subscribe();
            let printer = tokio::spawn(async move {
                loop {
                    match events.recv().await {
                        Ok(event) => {
                            if let Ok(line) = serde_json::to_string(&event) {
                                println!("{line}");
                            }
                        }
                        // Events missed while printing fell behind.
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            });
            let _ = tokio::signal::ctrl_c().await;
            watcher.stop().await?;
            printer.abort();
        }
    }
    // An update job can still be running (watch starts them); a token
    // refresh in flight holds the credentials until its new tokens are
    // saved, so wait for it before the runtime shuts down.
    if let Some(client) = client {
        client.openai_chatgpt_oauth_credentials().await;
    }
    Ok(())
}
