# Arko Index

A Rust library that indexes source code and documentation for retrieval. A language model
(GPT-5.6 Luna by default) chooses chunk boundaries along complete logical units, Voyage
embeds the chunks, and a local SQLite database serves lexical, semantic and hybrid search
that returns the complete text of every matching chunk.

The [reference](../docs/arko_index-reference.md) documents file selection, chunk
validation, configuration, job events, embeddings, search options, usage accounting, the
LLM judge, and the example CLI in full.

## Installation

```toml
[dependencies]
arko_index = { path = "../arko_index" }
arko_ai = { path = "../arko_ai" }
tokio = { version = "1.37", features = ["full"] }
```

## Build and search

```rust
use arko_ai::{EmbeddingClient, LLMClient};
use arko_index::{ArkoIndex, BuildSpec, Config, SearchOptions, backend::ArkoBackend};
use std::{sync::Arc, time::Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LLMClient::new().with_openai_chatgpt_oauth(
        &std::env::var("OPENAI_OAUTH_ACCESS_TOKEN")?,
        std::env::var("OPENAI_OAUTH_REFRESH_TOKEN").ok().as_deref(),
        &std::env::var("OPENAI_OAUTH_ACCOUNT_ID")?,
    );
    let config = Config::default();
    let backend = Arc::new(ArkoBackend::new(
        client.clone(),
        EmbeddingClient::new().with_voyage_api_key(&std::env::var("VOYAGE_API_KEY")?),
        config.embedding.clone(),
    ).await?);
    let index = ArkoIndex::open("/path/outside/source/index.db", backend, config).await?;

    // start_build returns at once; the job runs in the background.
    let job = index.start_build(BuildSpec::new("my-project", "/path/to/source")).await?;
    loop {
        let status = index.status(&job.job).await?;
        match status.phase.as_str() {
            "ready" => break,
            "paused" | "blocked" | "cancelled" | "failed" => {
                return Err(status.problem.unwrap_or(status.phase).into());
            }
            _ => tokio::time::sleep(Duration::from_secs(1)).await,
        }
    }

    let page = index.search(&job.index, "how are retries scheduled?", SearchOptions::default()).await?;
    for hit in &page.hits {
        // hit.text is the complete chunk; the hit also carries its byte and line ranges.
        println!("{}:{}-{}\n{}", hit.path, hit.first_line, hit.last_line, hit.text);
    }
    Ok(())
}
```

Each `start_build` creates a new index with its own ID and returns a `JobHandle` (`index`
and `job` IDs). `events(job, after, limit)` returns a job's progress events in pages, and
`search` keeps answering from the last complete generation while an update runs.

Credentials stay with the caller's `LLMClient`; the index database stores none. The client
above uses ChatGPT OAuth; `LLMClient::new().with_openai_api_key(key)` uses an OpenAI API
key instead. ChatGPT OAuth credentials rotate on refresh, and this client keeps them in
memory only: keep a clone of the `LLMClient` and persist them from
`openai_chatgpt_oauth_credentials()`. A client built from a Codex `auth.json`
(`arko_ai::CodexAuthFile`) saves them to the file itself. ChatGPT OAuth calls are billed to
the subscription (see [Usage and cost](../docs/arko_index-reference.md#usage-and-cost)).

## Key concepts

- **File selection.** A scan applies git ignore rules, even outside a repository, never
  enters directories such as `.git`, `target` and `node_modules`, and indexes common
  source, configuration and documentation files unless `include` and `exclude` globs say
  otherwise. See [Which files are indexed](../docs/arko_index-reference.md#which-files-are-indexed).
- **Chunk boundaries.** The chunk model partitions each file into coherent chunks with no
  size limit, keeping functions, impl blocks, type definitions and documentation sections
  whole. Requests are limited to 10% of the chunk model's input limit by default (27,200
  `o200k_base` tokens for GPT-5.6 Luna over ChatGPT OAuth); a larger file is split into
  windows at headings, items or blank lines, or skipped or blocked if configured. See
  [Request budget and large files](../docs/arko_index-reference.md#request-budget-and-large-files). Every cut is checked against the source, and in Rust files that parse and in
  Markdown files also against guarded units such as functions, statements, code blocks
  and list items; invalid cuts are repaired. A file's chunks cover it exactly, and source
  files are never modified. See [Chunking and validation](../docs/arko_index-reference.md#chunking-and-validation).
- **Jobs.** Builds and updates run in the background through `snapshotting`, `chunking`
  and `embedding` to `ready`, and can be paused, resumed from their checkpoints, or
  cancelled. Transient provider failures are retried with backoff. See
  [Jobs, scheduling and retries](../docs/arko_index-reference.md#jobs-scheduling-and-retries).
- **Generations and updates.** A successful job publishes a generation in one
  transaction, and search reads the active generation, so an update that fails, pauses or
  is cancelled leaves the previous one in place. Updates rescan the source with the saved
  spec and reuse the boundaries of unchanged files and the vectors of text embedded
  before; `watch` starts updates when the source changes, resumes updates interrupted by a
  stopped process, and reports updates it leaves paused or blocked. See
  [Storage and updates](../docs/arko_index-reference.md#storage-and-updates) and
  [Watching for changes](../docs/arko_index-reference.md#watching-for-changes).
- **Search.** Hybrid search combines SQLite FTS5 (BM25) and exact cosine-similarity
  rankings with reciprocal rank fusion; lexical and semantic modes use one of them. Path,
  literal-text and size filters apply before ranking, and results page with cursors.
  `search_with_judge` asks a model to filter and rank a candidate pool. See
  [Search](../docs/arko_index-reference.md#search) and
  [LLM judge](../docs/arko_index-reference.md#llm-judge).
- **Usage and cost.** `status(job).usage` sums the token usage and cost of every
  chunk-model and embedding response a job received, retries and rejected responses
  included. See [Usage and cost](../docs/arko_index-reference.md#usage-and-cost).
- **Storage.** An index database is a single SQLite file, locked so that one `ArkoIndex`
  serves it at a time. Keep it outside the indexed directory. Unchanged chunks are stored
  once and shared across generations; `delete` removes data no remaining index uses, and
  pruning older generations is opt-in (`Config::keep_generations`). Databases of the first
  schema version are migrated when opened. See
  [Storage and updates](../docs/arko_index-reference.md#storage-and-updates).

## Example CLI

The `arko_index_cli` example builds, updates, watches and searches indexes. From the
workspace root:

```bash
cargo run -p arko_index --example arko_index_cli -- \
  --database /path/to/index.db --oauth-file /path/to/auth.json \
  build /path/to/project --name project

cargo run -p arko_index --example arko_index_cli -- \
  --database /path/to/index.db search INDEX_ID 'retry scheduling'
```

`VOYAGE_API_KEY` must be set. OpenAI credentials come from `--oauth-file` (a Codex
`auth.json`) or from the environment. See [Example CLI](../docs/arko_index-reference.md#example-cli)
for every command and option, including the LLM judge.

## Requirements and testing

Building requires Rust 1.89 or later and a C compiler (SQLite is compiled from source).
The index needs a Tokio runtime and a filesystem that supports file locking for its
database.

The tests use a deterministic in-process backend, so they make no provider calls and need
no credentials:

```bash
cargo test -p arko_index --locked
```

CI runs them on Linux, macOS and Windows.

## License

MIT OR Apache-2.0
