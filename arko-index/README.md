# arko-index

A portable Rust library for indexing source code and documentation with Luna-selected boundaries, Voyage embeddings, and local SQLite retrieval. It accepts OpenAI OAuth or API-key clients through `arko-ai`.

## Build and search

```rust,no_run
use arko_ai::{LLMClient, EmbeddingClient};
use arko_index::{ArkoIndex, Config, BuildSpec, SearchOptions, backend::ArkoBackend};
use std::sync::Arc;
# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = LLMClient::new().with_openai_chatgpt_oauth_rate_limit_retries(
    &std::env::var("OPENAI_OAUTH_ACCESS_TOKEN")?,
    std::env::var("OPENAI_OAUTH_REFRESH_TOKEN").ok().as_deref(),
    &std::env::var("OPENAI_OAUTH_ACCOUNT_ID")?,
    false, // arko-index schedules transient request retries
);
let config = Config::default();
let backend = Arc::new(ArkoBackend::new(
    client.clone(),
    EmbeddingClient::new().with_voyage_api_key(&std::env::var("VOYAGE_API_KEY")?),
    config.embedding.clone(),
).await?);
let index = ArkoIndex::open("/path/outside/source/index.db", backend, config).await?;
let job = index.start_build(BuildSpec::new("my-project", "/path/to/source")).await?;
// start_build returns promptly. Observe status/events until phase == "ready".
let status = index.status(&job.job).await?;
let events = index.events(&job.job, 0, 100).await?;
# let _ = (status, events);
// Once ready:
let page = index.search(&job.index, "how are retries scheduled?", SearchOptions::default()).await?;
for hit in &page.hits {
    // hit.text is the entire source chunk, with exact byte and line ranges.
    println!("{}:{} {}", hit.path, hit.first_line, hit.text);
}
# Ok(()) }
```

The host retains its `LLMClient` clone and can persist refreshed OAuth credentials using `openai_chatgpt_oauth_credentials()`. Credentials never enter index metadata or request artifacts. `with_openai_api_key()` also works. OAuth usage reported as $0 is valid; it is not treated as an authentication failure.

## Runnable example

From the framework root:

```bash
cargo run -p arko-index --example arko_index_cli -- \
  --database /path/to/index.db --oauth-file /path/to/oauth.json \
  build /path/to/project --name project

cargo run -p arko-index --example arko_index_cli -- \
  --database /path/to/index.db search INDEX_ID 'retry scheduling'
```

Set `VOYAGE_API_KEY` in the process environment. OpenAI credentials can come from `--oauth-file`, `OPENAI_OAUTH_ACCESS_TOKEN` / `OPENAI_OAUTH_REFRESH_TOKEN` / `OPENAI_OAUTH_ACCOUNT_ID`, or `OPENAI_API_KEY`. The example does not automatically discover private credential files.

The OAuth JSON can have `access_token`, `refresh_token`, and `account_id` at its top level or under `tokens`. Use a file owned by the calling application. The example atomically persists rotated tokens and refuses to overwrite external changes. With environment-only OAuth, the host is responsible for saving rotated refresh tokens.

Other commands: `update INDEX_ID`, `resume JOB_ID`, `status JOB_ID`, `list`, `read INDEX_ID GENERATION CHUNK_ID...`, and `watch INDEX_ID --debounce-ms 500`. Build/update stream durable events. Ctrl+C requests a graceful pause. The library supports search/status while its own background job runs; a database lock prevents a second service from incorrectly recovering another process's active jobs.

## Source accuracy and scheduling

Chunking follows complete logical components, with no preferred or maximum chunk size. Functions, methods, coherent impl/class/type definitions and their comments stay together regardless of length. Documentation follows topic sections and complete examples. Whole files are presented to Luna with LF-normalized snapshots and absolute, padded line numbers. Source files are untouched; the model returns internal boundaries and exact suffixes, and all stored text comes from the snapshots. There is no arbitrary anchor-length cap.

Rust parser guards protect functions, impls, traits, structs, enums, modules and statements regardless of size. Markdown guards protect paragraphs, headings with content, examples and list items. Other languages or temporarily invalid syntax are marked `unchecked`. If the source has no legal internal boundary, its single chunk is established deterministically without asking Luna to select a boundary.

Repairs receive the complete source and all legal candidates inside each disputed interval. They can select multiple boundaries in order or return an empty array to merge the disputed intervals, preserving all source. Snapshot/revision validation prevents stale repairs. There is no repair-attempt cap, response token budget, request byte budget or model generation timeout. Source validity is required before publication; non-retryable provider errors remain explicit resumable failures.

Initial and repair prompts share guidance to group import/include blocks, related module declarations, re-exports and constants, and adjacent helpers with the behavior they support. Documentation introductions, examples and the instructions needed to use them stay together. Candidate IDs are possible split positions, not a checklist; semantic grouping has no numeric size threshold. Rust guards additionally protect each complete import statement, including multiline import trees and attributes, using its existing `syn` AST span. Cross-statement and documentation grouping is model-guided; it is not a claim that every language or semantic relationship is deterministically checked. Changes to either prompt invalidate old boundary reuse on the next update.

Six files per initial request and two simultaneously active Luna requests are scheduling settings (`batch_count` and `concurrency`). A finished slot is immediately refilled; initial batches have priority over repairs and retries. Embedding concurrency is separately configurable. Classified transient provider failures retry without an attempt cap, with an increasing wait after each failure (about one additional second plus jitter), honoring longer provider retry timing. There is no configured maximum wait or attempt count. Pause drains active calls into checkpoints; cancellation drops local request futures. `request_attempt` events record starts, failures, retry waits and completions as they happen.

Only Voyage's actual API token and item constraints require payload splitting or embedding-bin limits. The default profile is `voyage-4`, 1,024-dimensional float32, with tokenizer `voyageai/voyage-4@44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9`. Token counts use the actual submitted source text with the selected cached tokenizer. Arbitrary safety reserves and chunk-size targets are absent. API truncation is disabled.

If a semantic chunk exceeds Voyage's per-input token allowance, subdivision prefers whole components that fit—for example, complete methods from an oversized impl. Only units that themselves exceed that allowance can be split internally; such pieces are labeled `forced_split` and preserve exact UTF-8 provenance. All files finish chunking and validation before missing embedding payloads are deduplicated and packed into API-compliant bins. Vectors are checked for count, dimension, finite values and nonzero norm, then normalized. Provider response indexes restore input order.

SQLite keeps snapshots, chunk ranges, vectors, checkpoints and request artifacts. Generations publish atomically. A failed update retains the previous complete generation. Boundary reuse requires both the same normalized source and the current chunking-policy identity; changing the prompt/structural policy triggers re-evaluation even for unchanged files. Renames reuse content/vector identities. Historical request and repair findings are recorded in [REPAIR_FINDINGS.md](REPAIR_FINDINGS.md).

## Watching and rebuild notices

Watchers debounce until a quiet period, then compare final normalized hashes. Repeated writes and atomic-save event sequences become one net modification. Create/delete pairs that leave no file disappear. A unique deleted/added content match becomes a move; ambiguous duplicates remain explicit additions/deletions. `changes(job)` returns the coalesced plan. Periodic scans recover missed events without bypassing an active debounce interval. Jobs are finite; edits during a build enter a later update.

Search includes a rebuild notice with **all affected paths by default**. `SearchOptions::rebuild_file_limit` explicitly opts into truncating that list. Its token count is informational; no response budget suppresses it. While initial change detection runs, `determining_changes` is true. Paused/blocked updates remain visible with their phase.

## Complete retrieval

`search()` returns **complete source text in `SearchHit.text`**, with snapshot hash, exact byte/line range, structure status, score type and duplicate-location count. There are no previews or response token budgets. All ranked hits are returned by default; `SearchOptions.top_k: Some(n)` explicitly requests a page of `n` complete hits, with a generation-pinned cursor for the rest. There is no hard maximum on that caller-selected count, no hidden candidate-pool cap and no silent query-term truncation.

Hybrid retrieval combines SQLite FTS5 with exact cosine ranking. `path_prefix` and `literal` filter candidates before scoring. Every eligible candidate is scored before optional paging. Cached query embeddings and the vector-cache memory setting affect reuse/storage strategy only; larger matrices are scored from streamed database rows with identical candidate coverage. `candidate_count` and `omitted_hits` make caller-requested paging explicit. `search_tokens` reports the tokens in returned source text using the named `tokenizer`; it does not control output.

`read_chunks(index, generation, ids, neighbors)` remains available for fetching named chunks from a particular generation or requesting adjacent chunks. It has no arbitrary maximum of 100 IDs or 10 neighbors. Search already includes the full text, so a second read is unnecessary unless the caller wants different chunks or neighboring context.

Migration: `preview`, `preview_end_byte`, `preview_tokens`, `result_token_budget` and the old config fields `small_file_chars`, `max_repair_rounds`, `request_byte_budget` have been removed. Use `text` and the full chunk's existing byte range. `top_k` is now optional and defaults to `None`; `candidate_limit` became `candidate_count`, and the informational tokenizer field is `tokenizer`. Existing complete index generations remain readable; updating them adopts the new policy instead of silently reusing their old boundaries.

## Validation and boundaries

```bash
cargo test -p arko-index --locked
cargo test --workspace --lib --locked
cargo check --workspace --all-targets --locked
```

Tests cover whole large functions/impls/examples, full search results beyond the former response/count limits, queries beyond the former 64-term cutoff, unlimited targeted repair and transient retry, live attempt events, changed-policy invalidation, Voyage-only subdivision, exact reconstruction, eager scheduling, pause/cancel, generation isolation, malformed vectors, caching and atomic-save debounce. The CI matrix runs Linux, Windows and macOS. This checkout was executed on Linux; other OS results require their CI runners.

Requires a modern Rust toolchain with edition 2024 and portable file locking (`File::try_lock`). Legacy `arko_index` databases have incompatible provenance and are rejected; build a new database. Use one service instance per database. Historical generations/artifacts are retained until explicit deletion; automatic retention/garbage collection and ANN/reranking remain separate work. Additional language grammars and retrieval-quality tuning need measured follow-up trials.
