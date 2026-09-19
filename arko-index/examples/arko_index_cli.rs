//! Run from the workspace root; see arko-index/README.md.
use arko_ai::{EmbeddingClient, LLMClient};
use arko_index::{backend::ArkoBackend, *};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Duration};
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
        #[arg(long)]
        lexical: bool,
        #[arg(long)]
        rebuild_file_limit: Option<usize>,
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
struct OAuthFile {
    path: PathBuf,
    original: Vec<u8>,
}
fn clients(path: Option<PathBuf>) -> AnyResult<(LLMClient, Option<OAuthFile>)> {
    if let Some(path) = path {
        let original = std::fs::read(&path)?;
        let value: Value = serde_json::from_slice(&original)?;
        let tokens = value.get("tokens").unwrap_or(&value);
        let access = tokens["access_token"]
            .as_str()
            .ok_or("OAuth file needs access_token")?;
        let account = tokens["account_id"]
            .as_str()
            .ok_or("OAuth file needs account_id")?;
        let client = LLMClient::new().with_openai_chatgpt_oauth_rate_limit_retries(
            access,
            tokens["refresh_token"].as_str(),
            account,
            false,
        );
        return Ok((client, Some(OAuthFile { path, original })));
    }
    if let Ok(access) = std::env::var("OPENAI_OAUTH_ACCESS_TOKEN") {
        let account = std::env::var("OPENAI_OAUTH_ACCOUNT_ID")?;
        let refresh = std::env::var("OPENAI_OAUTH_REFRESH_TOKEN").ok();
        return Ok((
            LLMClient::new().with_openai_chatgpt_oauth_rate_limit_retries(
                &access,
                refresh.as_deref(),
                &account,
                false,
            ),
            None,
        ));
    }
    Ok((
        match std::env::var("OPENAI_API_KEY") {
            Ok(key) => LLMClient::new().with_openai_api_key(&key),
            Err(_) => LLMClient::new(),
        },
        None,
    ))
}
impl OAuthFile {
    async fn persist(&mut self, client: &LLMClient) -> AnyResult<()> {
        let Some(tokens) = client.openai_chatgpt_oauth_credentials().await else {
            return Ok(());
        };
        let mut value: Value = serde_json::from_slice(&self.original)?;
        let object = if value.get("tokens").is_some() {
            value.get_mut("tokens").unwrap()
        } else {
            &mut value
        };
        if object["access_token"].as_str() == Some(&tokens.access_token)
            && object["refresh_token"].as_str() == tokens.refresh_token.as_deref()
        {
            return Ok(());
        }
        object["access_token"] = json!(tokens.access_token);
        object["refresh_token"] = json!(tokens.refresh_token);
        object["account_id"] = json!(tokens.account_id);
        if std::fs::read(&self.path)? != self.original {
            return Err(
                "OAuth file changed externally; refusing to overwrite newer credentials".into(),
            );
        }
        let bytes = serde_json::to_vec_pretty(&value)?;
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        use std::io::Write;
        temporary.write_all(&bytes)?;
        temporary.as_file().sync_all()?;
        temporary.persist(&self.path)?;
        self.original = bytes;
        Ok(())
    }
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
            oauth.persist(client).await?;
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
    let (client, mut oauth) = clients(args.oauth_file)?;
    let embeddings = match std::env::var("VOYAGE_API_KEY") {
        Ok(key) => EmbeddingClient::new().with_voyage_api_key(&key),
        Err(_) => EmbeddingClient::new(),
    };
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
            rebuild_file_limit,
        } => {
            let options = SearchOptions {
                mode: if lexical {
                    SearchMode::Lexical
                } else {
                    SearchMode::Hybrid
                },
                rebuild_file_limit,
                ..Default::default()
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&index.search(&id, &query, options).await?)?
            );
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
                tokio::select! {_=tokio::signal::ctrl_c()=>break,_=tokio::time::sleep(Duration::from_secs(1))=>{if let Some(oauth)=&mut oauth{oauth.persist(&client).await?;}}}
            }
            watcher.stop().await?;
        }
    }
    if let Some(oauth) = &mut oauth {
        oauth.persist(&client).await?;
    }
    Ok(())
}
