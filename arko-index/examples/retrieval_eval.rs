//! Compare existing semantic/hybrid search without rebuilding or changing ranking.
//! Usage: retrieval_eval DATABASE INDEX_ID QUERIES_JSON OUTPUT_DIRECTORY
use arko_ai::{EmbeddingClient, LLMClient};
use arko_index::{ArkoIndex, Config, SearchMode, SearchOptions, backend::ArkoBackend};
use serde::Deserialize;
use serde_json::json;
use std::{path::PathBuf, sync::Arc, time::Instant};

#[derive(Deserialize)]
struct Query {
    id: String,
    query: String,
    path_prefix: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: retrieval_eval DATABASE INDEX_ID QUERIES_JSON OUTPUT_DIRECTORY".into());
    }
    let queries: Vec<Query> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&output)?;
    let config = Config {
        // Cache the existing corpus matrix; this changes reuse, not candidates/ranking.
        vector_cache_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    let embeddings = EmbeddingClient::new().with_voyage_api_key(&std::env::var("VOYAGE_API_KEY")?);
    // Search never invokes Luna, so no LLM/OAuth credentials are needed.
    let backend =
        Arc::new(ArkoBackend::new(LLMClient::new(), embeddings, config.embedding.clone()).await?);
    let index = ArkoIndex::open(&args[0], backend, config).await?;
    for (i, query) in queries.iter().enumerate() {
        // Alternate order to avoid always giving one mode a warm query/vector cache.
        let modes = if i % 2 == 0 {
            [SearchMode::Semantic, SearchMode::Hybrid]
        } else {
            [SearchMode::Hybrid, SearchMode::Semantic]
        };
        for mode in modes {
            let label = match mode {
                SearchMode::Semantic => "semantic",
                _ => "hybrid",
            };
            let path = output.join(format!("{}.{label}.json", query.id));
            let start = Instant::now();
            let page = index
                .search(
                    &args[1],
                    &query.query,
                    SearchOptions {
                        mode,
                        top_k: Some(10),
                        min_chunk_bytes: 40,
                        path_prefix: query.path_prefix.clone(),
                        ..Default::default()
                    },
                )
                .await?;
            let result = json!({
                "query_id": query.id,
                "query": query.query,
                "path_prefix": query.path_prefix,
                "elapsed_ms": start.elapsed().as_millis(),
                "page": page
            });
            std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
            println!(
                "{}",
                json!({"query_id": query.id, "mode": label,
                "hits": result["page"]["hits"].as_array().unwrap().len(),
                "elapsed_ms": result["elapsed_ms"]})
            );
        }
    }
    Ok(())
}
