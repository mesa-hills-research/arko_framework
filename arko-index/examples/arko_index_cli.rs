//! Run from the workspace root; see arko-index/README.md.
use arko_ai::{EmbeddingClient, LLMClient};
use arko_index::{backend::ArkoBackend, *};
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
    /// User-owned OAuth JSON, with access_token, refresh_token and account_id,
    /// either at the top level or in a tokens object. Rotated tokens are saved here.
    #[arg(long)]
    oauth_file: Option<PathBuf>,
    #[arg(long, default_value_t = 6)]
    batch_count: usize,
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Build {
        root: PathBuf,
        #[arg(long, default_value = "source")]
        name: String,
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
        #[arg(long)]
        rebuild_file_limit: Option<usize>,
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
    },
}
/// A Codex `auth.json`; rotated tokens are written back with `sync`.
type OAuthFile = arko_ai::CodexAuthFile;
/// Credentials from an OAuth file when given, else from the environment.
fn clients(path: Option<PathBuf>) -> AnyResult<(LLMClient, Option<OAuthFile>)> {
    if let Some(path) = path {
        let file = OAuthFile::load(path)?;
        return Ok((file.client()?, Some(file)));
    }
    Ok((LLMClient::from_env(), None))
}
async fn observe(
    index: &ArkoIndex,
    job: &str,
    client: &LLMClient,
    oauth: &mut Option<OAuthFile>,
) -> AnyResult<()> {
    let mut cursor = 0;
    loop {
        for event in index.events(job, cursor, 100).await? {
            cursor = event.sequence;
            println!("{}", serde_json::to_string(&event)?);
        }
        if let Some(oauth) = oauth {
            oauth.sync(client).await?;
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
    let args = Args::parse();
    let (client, mut oauth) = clients(args.oauth_file.clone())?;
    let embeddings = EmbeddingClient::from_env();
    let config = Config {
        batch_count: args.batch_count,
        concurrency: args.concurrency,
        ..Default::default()
    };
    let backend =
        Arc::new(ArkoBackend::new(client.clone(), embeddings, config.embedding.clone()).await?);
    let index = ArkoIndex::open(args.database, backend, config).await?;
    match args.command {
        Command::Build { root, name } => {
            let job = index.start_build(BuildSpec::new(name, root)).await?;
            println!("{}", serde_json::to_string(&job)?);
            observe(&index, &job.job, &client, &mut oauth).await?;
        }
        Command::Update { index: id } => {
            let job = index.start_update(&id).await?;
            println!("{}", serde_json::to_string(&job)?);
            observe(&index, &job.job, &client, &mut oauth).await?;
        }
        Command::Resume { job } => {
            index.resume(&job).await?;
            observe(&index, &job, &client, &mut oauth).await?;
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
                rebuild_file_limit,
                min_chunk_bytes,
                top_k: Some(top_k),
                ..Default::default()
            };
            let result = if judging.judge {
                let mut session = judging.session(&client, args.oauth_file.as_deref())?;
                let result = index
                    .search_with_judge(&id, &query, options, &session.judge, judging.options())
                    .await;
                // Preserve rotated credentials even when selection fails.
                session.persist().await?;
                result
            } else {
                index.search(&id, &query, options).await
            };
            if let Some(oauth) = &mut oauth {
                oauth.sync(&client).await?;
            }
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
        } => {
            let watcher = index
                .watch(
                    &id,
                    Duration::from_millis(debounce_ms),
                    Duration::from_secs(30),
                )
                .await?;
            loop {
                tokio::select! {_=tokio::signal::ctrl_c()=>break,_=tokio::time::sleep(Duration::from_secs(1))=>{if let Some(oauth)=&mut oauth{oauth.sync(&client).await?;}}}
            }
            watcher.stop().await?;
        }
    }
    if let Some(oauth) = &mut oauth {
        oauth.sync(&client).await?;
    }
    Ok(())
}
