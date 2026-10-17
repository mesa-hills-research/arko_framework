//! Arm C's dense search (opt-in): every arko_index chunk embedded with
//! Voyage (`prepare --dense`), and a query service in the benchmark process
//! that the agents' `arko_index_dense` command asks over a Unix socket, so
//! the Voyage key never reaches an agent's shell.
//!
//! arko_index itself is lexical only (dense search is planned for it); this
//! arm measures what dense retrieval over the same chunks would add.

// The query service is Unix-only; elsewhere its parts are unused.
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::BenchResult;
use crate::repos::PreparedRepo;

/// The variables `arko_index_dense` reads: the service's socket and the
/// repository it searches.
pub const SOCKET_VARIABLE: &str = "ARKO_DENSE_SOCKET";
pub const REPO_VARIABLE: &str = "ARKO_DENSE_REPO";

#[derive(Debug, Clone)]
pub struct DenseOptions {
    /// An offline hashing embedder instead of Voyage.
    pub mock: bool,
    pub model: String,
    pub dimensions: usize,
}

/// A built dense index, as `prepare.json` records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseInfo {
    pub dir: PathBuf,
    pub model: String,
    pub dimensions: usize,
    pub mock: bool,
    pub chunks: usize,
    /// Embedding tokens billed (Voyage's count; estimated for the mock).
    pub tokens: usize,
    pub build_ms: u64,
}

/// The metadata of one chunk, as search results print it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub path: String,
    pub start_line: u64,
    pub end_line: u64,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub symbol_path: String,
    #[serde(default)]
    pub language: String,
    #[serde(skip_serializing, default)]
    pub start_byte: usize,
    #[serde(skip_serializing, default)]
    pub end_byte: usize,
}

/// The offline embedder: words and identifier parts hashed into signed
/// buckets, normalized. Deterministic; for dry runs and tests only.
pub fn mock_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0f32; dimensions.max(1)];
    for token in crate::dense::tokens(text) {
        let hash = fnv(token.as_bytes());
        let bucket = (hash % vector.len() as u64) as usize;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        vector[bucket] += sign;
    }
    normalize(&mut vector);
    vector
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Lowercase words and identifier parts (snake_case and camelCase split).
fn tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let mut part = String::new();
        let mut previous_lower = false;
        for c in word.chars() {
            if c.is_uppercase() && previous_lower && !part.is_empty() {
                tokens.push(part.to_lowercase());
                part.clear();
            }
            previous_lower = c.is_lowercase() || c.is_ascii_digit();
            part.push(c);
        }
        if !part.is_empty() {
            tokens.push(part.to_lowercase());
        }
    }
    tokens
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector {
            *x /= norm;
        }
    }
}

/// Embeds `texts` with Voyage, or the mock.
async fn embed(
    client: &Option<arko_ai::EmbeddingClient>,
    model: &str,
    dimensions: usize,
    texts: Vec<String>,
    query: bool,
) -> BenchResult<(Vec<Vec<f32>>, usize)> {
    let Some(client) = client else {
        let tokens = texts.iter().map(|text| text.len().div_ceil(4)).sum();
        return Ok((
            texts
                .iter()
                .map(|text| mock_embedding(text, dimensions))
                .collect(),
            tokens,
        ));
    };
    let config = arko_ai::EmbeddingConfig::new(arko_ai::EmbeddingProviderIdentifier::Voyage, model)
        .with_output_dimensionality(dimensions)
        .with_voyage_input_type(if query {
            arko_ai::VoyageInputType::Query
        } else {
            arko_ai::VoyageInputType::Document
        });
    let mut last_error = String::new();
    for attempt in 0..6u32 {
        match client.embed(texts.clone(), &config).await {
            Ok(response) => {
                let mut embeddings = response.embeddings;
                for vector in &mut embeddings {
                    normalize(vector);
                }
                return Ok((embeddings, response.usage.input_tokens));
            }
            Err(error) => {
                last_error = error.to_string();
                tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt))).await;
            }
        }
    }
    Err(format!("Voyage embedding failed: {last_error}").into())
}

fn voyage_client(mock: bool) -> BenchResult<Option<arko_ai::EmbeddingClient>> {
    if mock {
        return Ok(None);
    }
    match std::env::var("VOYAGE_API_KEY") {
        Ok(key) if !key.trim().is_empty() => Ok(Some(
            arko_ai::EmbeddingClient::new().with_voyage_api_key(key.trim()),
        )),
        _ => Err("dense search needs VOYAGE_API_KEY (or --mock-embeddings for a dry run)".into()),
    }
}

/// Embeds every chunk arko_index makes of `checkout` into `dir`.
pub async fn build(
    options: &DenseOptions,
    index_cli: &Path,
    checkout: &Path,
    dir: &Path,
    id: &str,
) -> BenchResult<DenseInfo> {
    let client = voyage_client(options.mock)?;
    let started = Instant::now();
    let output = tokio::process::Command::new(index_cli)
        .arg("chunk")
        .arg(checkout)
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!(
            "{id}: arko_index chunk failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let mut chunks: Vec<ChunkMeta> = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(chunk) = serde_json::from_str::<ChunkMeta>(line) {
            chunks.push(chunk);
        }
    }
    eprintln!(
        "{id}: embedding {} chunks with {}",
        chunks.len(),
        if options.mock {
            "the mock embedder"
        } else {
            &options.model
        }
    );
    let mut files: HashMap<String, Vec<u8>> = HashMap::new();
    let mut vectors: Vec<f32> = Vec::with_capacity(chunks.len() * options.dimensions);
    let mut tokens = 0;
    for (batch_index, batch) in chunks.chunks(64).enumerate() {
        let mut texts = Vec::with_capacity(batch.len());
        for chunk in batch {
            if !files.contains_key(&chunk.path) {
                let bytes = std::fs::read(checkout.join(&chunk.path)).unwrap_or_default();
                files.insert(chunk.path.clone(), bytes);
            }
            let bytes = &files[&chunk.path];
            let end = chunk.end_byte.min(bytes.len());
            let start = chunk.start_byte.min(end);
            let text = String::from_utf8_lossy(&bytes[start..end]);
            texts.push(format!("{}\n{}", chunk.path, text));
        }
        let (embeddings, used) =
            embed(&client, &options.model, options.dimensions, texts, false).await?;
        tokens += used;
        for embedding in embeddings {
            if embedding.len() != options.dimensions {
                return Err(format!(
                    "{id}: an embedding has {} dimensions, not {}",
                    embedding.len(),
                    options.dimensions
                )
                .into());
            }
            vectors.extend(embedding);
        }
        if batch_index % 50 == 49 {
            eprintln!(
                "{id}: {} of {} chunks embedded",
                (batch_index + 1) * 64,
                chunks.len()
            );
        }
    }
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let mut meta = String::new();
    for chunk in &chunks {
        meta.push_str(&serde_json::to_string(chunk)?);
        meta.push('\n');
    }
    std::fs::write(dir.join("meta.jsonl"), meta)?;
    let bytes: Vec<u8> = vectors.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join("vectors.f32"), bytes)?;
    let info = DenseInfo {
        dir: dir.to_path_buf(),
        model: if options.mock {
            "mock-hashing".into()
        } else {
            options.model.clone()
        },
        dimensions: options.dimensions,
        mock: options.mock,
        chunks: chunks.len(),
        tokens,
        build_ms: started.elapsed().as_millis() as u64,
    };
    std::fs::write(dir.join("info.json"), serde_json::to_string_pretty(&info)?)?;
    eprintln!(
        "{id}: dense index of {} chunks, {tokens} embedding tokens, {:.1} s",
        info.chunks,
        info.build_ms as f64 / 1000.0
    );
    Ok(info)
}

/// A dense index in memory.
pub struct DenseIndex {
    info: DenseInfo,
    meta: Vec<ChunkMeta>,
    vectors: Vec<f32>,
}

impl DenseIndex {
    pub fn load(info: &DenseInfo) -> BenchResult<Self> {
        let meta: Vec<ChunkMeta> = std::fs::read_to_string(info.dir.join("meta.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        let bytes = std::fs::read(info.dir.join("vectors.f32"))?;
        let vectors: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if vectors.len() != meta.len() * info.dimensions {
            return Err(format!(
                "{}: {} vectors for {} chunks",
                info.dir.display(),
                vectors.len() / info.dimensions.max(1),
                meta.len()
            )
            .into());
        }
        Ok(Self {
            info: info.clone(),
            meta,
            vectors,
        })
    }

    /// The `limit` chunks most similar to `query` (normalized).
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<(f32, &ChunkMeta)> {
        let dimensions = self.info.dimensions;
        let mut scored: Vec<(f32, usize)> = self
            .vectors
            .chunks_exact(dimensions)
            .enumerate()
            .map(|(index, vector)| {
                let score: f32 = vector.iter().zip(query).map(|(a, b)| a * b).sum();
                (score, index)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored
            .into_iter()
            .take(limit)
            .map(|(score, index)| (score, &self.meta[index]))
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct QueryRequest {
    repo: String,
    query: String,
    limit: usize,
}

/// The query service of a run with arm C.
pub struct DenseService {
    socket: PathBuf,
    task: tokio::task::JoinHandle<()>,
}

impl DenseService {
    #[cfg(unix)]
    pub async fn start(
        run_dir: &Path,
        prepared: &std::collections::BTreeMap<String, PreparedRepo>,
        mock: bool,
    ) -> BenchResult<Self> {
        let mut socket = run_dir.join("dense.sock");
        // Unix socket paths are limited to about 100 bytes.
        if socket.as_os_str().len() > 100 {
            socket = std::env::temp_dir()
                .join(format!("index-benchmark-dense-{}.sock", std::process::id()));
        }
        let _ = std::fs::remove_file(&socket);
        let listener = tokio::net::UnixListener::bind(&socket)?;
        let infos: HashMap<String, DenseInfo> = prepared
            .iter()
            .filter_map(|(id, repo)| repo.dense.clone().map(|dense| (id.clone(), dense)))
            .collect();
        let any_real = infos.values().any(|info| !info.mock);
        let client = Arc::new(voyage_client(mock || !any_real)?);
        let indexes: Arc<tokio::sync::Mutex<HashMap<String, Arc<DenseIndex>>>> = Arc::default();
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let infos = infos.clone();
                let client = client.clone();
                let indexes = indexes.clone();
                tokio::spawn(async move {
                    let _ = serve(stream, &infos, &client, &indexes).await;
                });
            }
        });
        Ok(Self { socket, task })
    }

    #[cfg(not(unix))]
    pub async fn start(
        _run_dir: &Path,
        _prepared: &std::collections::BTreeMap<String, PreparedRepo>,
        _mock: bool,
    ) -> BenchResult<Self> {
        Err("arm C (dense search) needs a Unix host".into())
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
    infos: &HashMap<String, DenseInfo>,
    client: &Option<arko_ai::EmbeddingClient>,
    indexes: &tokio::sync::Mutex<HashMap<String, Arc<DenseIndex>>>,
) -> BenchResult<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let (reader, mut writer) = stream.into_split();
    let mut line = String::new();
    BufReader::new(reader).read_line(&mut line).await?;
    let reply = match answer(&line, infos, client, indexes).await {
        Ok(lines) => lines,
        Err(error) => format!("{}\n", serde_json::json!({ "error": error.to_string() })),
    };
    writer.write_all(reply.as_bytes()).await?;
    writer.shutdown().await?;
    Ok(())
}

#[cfg(unix)]
async fn answer(
    line: &str,
    infos: &HashMap<String, DenseInfo>,
    client: &Option<arko_ai::EmbeddingClient>,
    indexes: &tokio::sync::Mutex<HashMap<String, Arc<DenseIndex>>>,
) -> BenchResult<String> {
    let request: QueryRequest = serde_json::from_str(line)?;
    let info = infos
        .get(&request.repo)
        .ok_or_else(|| format!("no dense index for {}", request.repo))?;
    let index = {
        let mut indexes = indexes.lock().await;
        match indexes.get(&request.repo) {
            Some(index) => index.clone(),
            None => {
                let index = Arc::new(DenseIndex::load(info)?);
                indexes.insert(request.repo.clone(), index.clone());
                index
            }
        }
    };
    let client = if info.mock { &None } else { client };
    let (mut embeddings, _) = embed(
        client,
        &info.model,
        info.dimensions,
        vec![request.query.clone()],
        true,
    )
    .await?;
    let query = embeddings.pop().ok_or("no query embedding")?;
    let mut text = String::new();
    for (score, chunk) in index.search(&query, request.limit.clamp(1, 50)) {
        let mut value = serde_json::to_value(chunk)?;
        value["score"] = serde_json::json!((score * 1000.0).round() / 1000.0);
        text.push_str(&value.to_string());
        text.push('\n');
    }
    Ok(text)
}

/// `arko_index_dense QUERY`: asks the running benchmark's service.
pub async fn query_command(query: &str, limit: usize) -> BenchResult<()> {
    #[cfg(unix)]
    {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let socket = std::env::var(SOCKET_VARIABLE).map_err(|_| {
            format!("{SOCKET_VARIABLE} is not set. arko_index_dense runs inside the benchmark")
        })?;
        let repo =
            std::env::var(REPO_VARIABLE).map_err(|_| format!("{REPO_VARIABLE} is not set"))?;
        let mut stream = tokio::net::UnixStream::connect(&socket).await?;
        let request = serde_json::to_string(&QueryRequest {
            repo,
            query: query.to_string(),
            limit,
        })?;
        stream.write_all(format!("{request}\n").as_bytes()).await?;
        let mut reply = String::new();
        stream.read_to_string(&mut reply).await?;
        if let Some(first) = reply.lines().next()
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(first)
            && let Some(error) = value.get("error").and_then(|e| e.as_str())
        {
            return Err(format!("arko_index_dense: {error}").into());
        }
        print!("{reply}");
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (query, limit);
        Err("arko_index_dense needs a Unix host".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embeddings_rank_related_text_first() {
        let query = mock_embedding("retry backoff delay", 256);
        let related = mock_embedding("fn retryBackoffDelay(attempt: u32) -> Duration", 256);
        let unrelated = mock_embedding("struct HtmlParser { tokens: Vec<Token> }", 256);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&query, &related) > dot(&query, &unrelated));
        assert!((dot(&query, &query) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn splits_identifiers_into_parts() {
        assert_eq!(
            tokens("retryPolicy max_bytes HTTPServer"),
            vec!["retry", "policy", "max", "bytes", "httpserver"]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_service_answers_queries_over_its_socket() {
        let dir = crate::test_dir("dense");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        let chunks = [
            ("src/retry.rs", "fn retry_backoff_delay() {}"),
            ("src/html.rs", "struct HtmlParser {}"),
        ];
        let mut meta = String::new();
        let mut bytes = Vec::new();
        for (path, text) in chunks {
            meta.push_str(
                &serde_json::to_string(&ChunkMeta {
                    path: path.into(),
                    start_line: 1,
                    end_line: 1,
                    kind: "function_item".into(),
                    symbol_path: String::new(),
                    language: "rust".into(),
                    start_byte: 0,
                    end_byte: 0,
                })
                .unwrap(),
            );
            meta.push('\n');
            for x in mock_embedding(&format!("{path}\n{text}"), 64) {
                bytes.extend(x.to_le_bytes());
            }
        }
        std::fs::write(index_dir.join("meta.jsonl"), meta).unwrap();
        std::fs::write(index_dir.join("vectors.f32"), bytes).unwrap();
        let info = DenseInfo {
            dir: index_dir,
            model: "mock-hashing".into(),
            dimensions: 64,
            mock: true,
            chunks: 2,
            tokens: 0,
            build_ms: 0,
        };
        let repo = PreparedRepo {
            spec: crate::repos::RepoSpec {
                id: "demo".into(),
                url: String::new(),
                rev: "v1".into(),
                commit: None,
                language: "rust".into(),
                description: String::new(),
            },
            checkout: dir.clone(),
            index: dir.clone(),
            commit: "abc".into(),
            files: 0,
            bytes: 0,
            files_by_language: Default::default(),
            index_build_ms: 0,
            index_summary: String::new(),
            index_cli_sha256: String::new(),
            dense: Some(info),
        };
        let prepared = std::collections::BTreeMap::from([("demo".to_string(), repo)]);
        let service = DenseService::start(&dir, &prepared, true).await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::UnixStream::connect(service.socket())
            .await
            .unwrap();
        stream
            .write_all(b"{\"repo\":\"demo\",\"query\":\"retry backoff\",\"limit\":1}\n")
            .await
            .unwrap();
        let mut reply = String::new();
        stream.read_to_string(&mut reply).await.unwrap();
        assert_eq!(reply.lines().count(), 1, "{reply}");
        assert!(reply.contains("src/retry.rs"), "{reply}");
        service.stop();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
