# Arko Index

A Rust library that indexes source code and documentation for retrieval. A language model
(GPT-5.6 Luna by default) chooses chunk boundaries along complete logical units, Voyage
embeds the chunks, and a local SQLite database serves lexical, semantic and hybrid search
that returns the complete text of every matching chunk.

## Build and search

```rust,no_run
use arko_ai::{EmbeddingClient, LLMClient};
use arko_index::{ArkoIndex, BuildSpec, Config, SearchOptions, backend::ArkoBackend};
use std::{sync::Arc, time::Duration};
# async fn example() -> Result<(), Box<dyn std::error::Error>> {
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
# Ok(()) }
```

The index runs its jobs as Tokio tasks, so use it inside a Tokio runtime. Each
`start_build` creates a new index with its own ID. `events(job, after, limit)` streams a
job's progress (see [Jobs, scheduling and retries](#jobs-scheduling-and-retries)), and
`search` keeps answering from the last complete generation while an update runs.

Credentials stay with the caller's `LLMClient`; the index database stores none. The client
above uses ChatGPT OAuth; `LLMClient::new().with_openai_api_key(key)` uses an OpenAI API
key instead. With ChatGPT OAuth, keep a clone of the `LLMClient` and persist rotated
credentials from `openai_chatgpt_oauth_credentials()`. ChatGPT OAuth calls are billed to
the subscription (see [Usage and cost](#usage-and-cost)).

## Chunk model

`ArkoBackend::new` selects boundaries with GPT-5.6 Luna at high reasoning effort, with
reasoning summaries and low verbosity (`backend::default_chunk_config()`).
`with_chunk_model` selects another `arko-ai` model and its generation settings; the client
passed to `ArkoBackend::new` must hold credentials for that model's provider:

```rust,no_run
use arko_ai::{GenerationConfig, Model, OpenAIEffort, ReasoningConfig};
# fn example(backend: arko_index::backend::ArkoBackend) {
let backend = backend.with_chunk_model(
    Model::openai_gpt_6_luna(),
    GenerationConfig::default().with_reasoning(ReasoningConfig::openai(OpenAIEffort::Medium, true)),
);
# let _ = backend; }
```

The backend owns the tool configuration: the boundary tool replaces any `tool_config` in
the settings, the call is forced where the model allows it, and parallel tool calls are
disabled where the provider supports that. Every reply must be exactly one call to that
tool in a completed turn (`StopReason::EndTurn` or `StopReason::ToolUse`):

- A refusal, or a reply stopped by a token or context limit, fails the request.
- A missing, extra or malformed tool call is retried as a transient failure.

Each job's `snapshot` event records the chunk `model` and `reasoning`. Boundaries are
validated against the source in the same way for every model, so after a model change an
update still reuses the boundaries of unchanged files. To re-chunk everything with the new
model, build a new index.

## Which files are indexed

A scan walks `BuildSpec::root` without following symbolic links and includes hidden files.
Ignore rules apply as in git, even outside a repository: `.gitignore` and `.ignore` files
(including those in the root's parent directories), `.git/info/exclude` and the global git
excludes file.

Directories named in `BuildSpec::skip_directories` are never entered, at any depth,
whatever the ignore files say. The default, `DEFAULT_SKIPPED_DIRECTORIES`, is `.git`,
`target`, `node_modules`, `.runtime`, `.cache`, `__pycache__`, `.venv` and `dist`. Names
match exactly; they are not patterns. Replace the list with `with_skip_directories`, for
example to index authored documentation under a `dist/` directory. An empty list skips
nothing, so keep `.git` in a custom list.

`include` and `exclude` hold glob patterns matched against the whole root-relative path,
with `/` separators. `*` also matches `/`, so `*.md` matches Markdown files at any depth.

```rust,no_run
# use arko_index::BuildSpec;
let spec = BuildSpec {
    include: vec!["docs/**/*.md".into(), "src/**/*.rs".into()],
    exclude: vec!["docs/archive/**".into()],
    ..BuildSpec::new("docs-and-code", "/path/to/project")
};
# let _ = spec;
```

A file that matches an `exclude` pattern is skipped. With `include` patterns, exactly the
files matching one of them are indexed, whatever their extension. Without them, a file is
indexed when its name is `Dockerfile`, `Makefile`, `CMakeLists.txt`, `Justfile`,
`LICENSE`, `README` or `AGENTS.md`, or when its extension (in any case) is one of `rs`,
`py`, `pyi`, `js`, `jsx`, `ts`, `tsx`, `mjs`, `cjs`, `go`, `c`, `h`, `cc`, `cpp`, `hpp`,
`cs`, `java`, `kt`, `swift`, `rb`, `php`, `lua`, `sh`, `ps1`, `sql`, `ex`, `exs`, `erl`,
`hs`, `scala`, `vue`, `svelte`, `html`, `css`, `scss`, `md`, `mdx`, `rst`, `txt`, `adoc`,
`toml`, `yaml`, `yml`, `json`, `xml`, `proto`, `graphql` or `tf`.

Files that are not valid UTF-8 or that contain NUL bytes are skipped; a path that is not
valid UTF-8 stops the scan with an error. `source::scan` reports symbolic links and
skipped files with a reason in `Manifest::excluded`, and each job's `snapshot` event
counts them; skipped directories are not reported. The spec is saved with the index, so
updates and the watcher scan with the same settings.

## Example CLI

The `arko_index_cli` example builds, updates, watches and searches indexes. From the
workspace root:

```bash
cargo run -p arko-index --example arko_index_cli -- \
  --database /path/to/index.db --oauth-file /path/to/auth.json \
  build /path/to/project --name project

cargo run -p arko-index --example arko_index_cli -- \
  --database /path/to/index.db search INDEX_ID 'retry scheduling'
```

| Command | Action |
|---|---|
| `build ROOT [--name NAME]` | Build a new index (name defaults to `source`) and stream its events. |
| `update INDEX_ID` | Start an update and stream its events. |
| `resume JOB_ID` | Resume a paused, blocked or failed job and stream its events. |
| `status JOB_ID` | Print the job's status, including usage. |
| `list` | List indexes and their active generations. |
| `search INDEX_ID QUERY` | Search; see the options below and [LLM judge](#llm-judge). |
| `read INDEX_ID GENERATION CHUNK_ID...` | Print chunks from a generation (the `generation` of a search result). |
| `watch INDEX_ID [--debounce-ms MS]` | Start updates when the source changes (500 ms debounce, a reconciliation scan every 30 s) until Ctrl+C. |

Options before the command: `--database PATH` (required), `--oauth-file PATH`,
`--batch-count N` (default 6) and `--concurrency N` (default 2). Resuming a job requires
the `--batch-count` and `--concurrency` values it started with.

`build`, `update` and `resume` print events as JSON lines until the job is ready; Ctrl+C
pauses the job so that `resume` can continue it later. The database lock admits one
process at a time, so other commands on the same database fail while one of these, or
`watch`, is running.

`search` returns the top 10 results by default (`--top-k N`) and hides chunks shorter than
40 bytes (`--min-chunk-bytes 0` includes them). `--lexical` selects full-text retrieval
only and `--semantic` vector retrieval only; otherwise search is hybrid.
`--rebuild-file-limit N` caps the paths listed in a rebuild notice.

`VOYAGE_API_KEY` must be set. OpenAI credentials come from `--oauth-file`, or else from the
environment: `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (with an optional
`OPENAI_OAUTH_REFRESH_TOKEN`) for ChatGPT OAuth, or `OPENAI_API_KEY`. The OAuth file uses
the Codex `auth.json` layout, with `access_token`, `refresh_token` and `account_id` at the
top level or under `tokens`. Rotated tokens are written back atomically (the previous
contents are kept in `<file>.bak`), and credentials that another process wrote to the file
are loaded rather than overwritten; give the example a file that no other application
refreshes. Rotated tokens from environment credentials are not saved.

## Chunking and validation

The chunk model partitions each file into coherent groups with no preferred or maximum
size. Complete functions, methods, impl blocks, classes and type definitions stay
together with their comments, whatever their length; documentation keeps each topic
section and worked example together with the introduction and instructions needed to use
it. The guidance also asks the model to keep import blocks, related module declarations,
re-exports and constants, and adjacent helpers with the code they support. That grouping
is model-guided; the structural guards below are what the index enforces.

Each initial request carries up to `batch_count` complete files, normalized to LF line
endings and shown with absolute line numbers. Because files are sent whole, the model's
context window must hold each batch; lower `batch_count` for very large files. The model
answers with cuts: an absolute line number and an exact suffix of the text ending on that
line. A cut is valid when the suffix matches, the line is not blank, the cuts are in
strictly increasing order, and the cut does not fall inside a guarded unit:

| Files | `structure` | Guarded units |
|---|---|---|
| `.rs` files that parse with `syn` | `rust-guarded` | Functions and methods, impl blocks, traits, structs, enums, modules, `use` declarations (including multiline trees), statements and string literals, each with its attributes and doc comments. |
| `.md` and `.mdx` files | `markdown-guarded` | Paragraphs, code blocks, list items, headings, and each heading together with the block that follows it. |
| Other files, and Rust files that do not parse | `unchecked` | None; cuts are checked against the line text only. |

Invalid cuts are repaired without changing accepted ones. If there is no legal boundary
between the neighboring accepted cuts, the disputed cuts are removed without a model
request (a `deterministic_repair` event). Otherwise a repair request sends the complete
file with the legal boundary IDs of each disputed region, and the model returns an ordered
selection of boundaries, or `[]` to merge the region into one chunk. Repairs continue
until every cut is valid, and a repair applies only to the snapshot and cut set it was
requested for. A file with no legal internal boundary becomes a single chunk without a
model request (counted in `JobStatus::bypassed`); empty and whitespace-only files produce
no chunks.

Source files are never modified. All stored text comes from the normalized snapshot, and
a file's chunks cover it exactly, in order, with no gaps or overlaps.

The chunking policy (the prompts and a policy version) has an identity, recorded as
`chunking_policy` in each `snapshot` event. When it changes, the next update re-chunks
every file.

## Jobs, scheduling and retries

`start_build(spec)` and `start_update(index)` return a `JobHandle` (`index` and `job`
IDs) at once and run the job in the background. `status(job)` returns a `JobStatus` with
the phase, the last `problem`, file counts (`files`, `validated`, `bypassed`,
`unchecked`), the requests in flight and the job's usage. A job moves through
`snapshotting`, `chunking` and `embedding` to `ready`, or stops as:

- `paused`: after `pause(job)`, or when its process stopped mid-run (detected when the
  database is next opened);
- `blocked`: after a provider error, an invalid response, or with files whose boundaries
  still need repair;
- `cancelled`: after `cancel(job)`;
- `failed`: after a local error, such as an I/O or database failure.

`resume(job)` continues a paused, blocked or failed job from its checkpoints. It requires
the `Config` the job started with, and fails if the chunking policy has changed since; in
that case, cancel the job and start a new update. An index has at most one unfinished
job: `start_update` fails while a build or update of the index is running, paused or
blocked.

| `Config` field | Default | Meaning |
|---|---|---|
| `batch_count` | 6 | Files per initial chunk-model request. |
| `concurrency` | 2 | Chunk-model requests in flight, shared by all jobs of the `ArkoIndex`. |
| `embedding_concurrency` | 2 | Embedding requests in flight, including search query embeddings. |
| `embedding` | `voyage-4`, 1,024 dimensions | Embedding profile; see [Embeddings](#embeddings). |
| `vector_cache_bytes` | 64 MiB | Memory for caching search vectors; see [Search](#search). |
| `max_transient_attempts` | 10 | Transient failures allowed per request before it fails. |

`batch_count`, both concurrency limits and `max_transient_attempts` must be positive. A
finished request slot is refilled immediately, and initial chunk requests take priority
over repairs and retries.

Transient provider failures (rate limits, timeouts, server errors, truncated streams and
malformed tool calls) are retried with jittered exponential backoff. The backoff step
starts at 1 s and doubles up to 60 s; each wait is drawn between half and all of the
step, and is never shorter than a retry delay the provider asks for. After
`max_transient_attempts` transient failures, the request fails with its last error, a
`retries_exhausted` attempt event is recorded, and the job becomes `blocked`. Explicit
capacity signals (overloaded or unavailable) are retried on the same schedule without
consuming attempts, with no limit on the total wait. Other errors, such as authentication
failures, invalid requests and refusals, are not retried.

`pause(job)` stops queued requests and ends retry waits at once, while requests already in
flight finish and are checkpointed. `cancel(job)` also drops in-flight requests. A job
that is being cancelled cannot be paused.

`events(job, after, limit)` returns up to `limit` events with a sequence number greater
than `after`; pass the last sequence seen to continue. Each `Event` has a `kind` and a
JSON `detail`:

| Kind | Detail |
|---|---|
| `phase` | The new `phase` and `problem`. |
| `snapshot` | Counts of scanned, excluded and affected files; the manifest hash; `chunking_policy` and `policy_changed`; the chunk `model` and `reasoning`. |
| `request_attempt` | `artifact` (request ID), `repair`, and `event`: `phase` (`started`, `failed`, `retry_wait`, `retries_exhausted` or `completed`), `attempt`, `unix_ms` and a phase-specific `detail`, such as `error` and `usage` for a failure or `delay_ms` and `capacity` for a retry wait. |
| `request_finished` | `artifact`, `repair` and the `usage` of the accepted reply. |
| `deterministic_repair` | Cuts removed without a model request. |
| `pause_requested` | A pause was requested. |
| `embedding_plan` | Embedding `payloads`, `reused` vectors and request `bins`. |
| `embedded` | An embedding request's `billed_tokens`, `usage` and `rejected_usage`. |
| `embedding_rejected` | An embedding response whose vectors failed validation, with its `error` and usage. |
| `embedding_failed` | A failed embedding request's `error` and `rejected_usage`. |

`request_artifact(job, id)`, where `id` is an event's `artifact`, returns the saved
chunk-model request with its response or error and its attempt events.

## Embeddings

The default `EmbeddingProfile` is Voyage `voyage-4` with 1,024 float32 dimensions,
counting tokens with the `voyageai/voyage-4` tokenizer at the pinned revision
`44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9`. `ArkoBackend::new` loads that tokenizer from
the Hugging Face Hub. It accepts Voyage models whose per-input, per-request token and
per-request item limits are known (not `voyage-code-4`), at a dimension the model
supports; for `voyage-4` the limits are 32,000 tokens per input, and 320,000 tokens and
1,000 inputs per request. `ArkoIndex::open` requires the backend's profile to equal
`Config::embedding`.

Embedding starts once every file has valid boundaries. Token counts use the exact
submitted text, and Voyage truncation is disabled. A chunk is split only when it exceeds
the per-input token limit. The split falls at the last line boundary that fits and does
not cut through a guarded unit (such as a method or statement) that fits on its own; if
there is none, at the middle of the remaining text, on a character boundary. Pieces cut
this way have the structure `forced_split`, and together they reproduce the chunk exactly.

Payloads are deduplicated by content, vectors already stored in the database are reused,
and the rest are packed into requests within Voyage's limits. Each response must contain
one finite, nonzero vector of the configured dimension per input; vectors are normalized
to unit length. An invalid response is recorded as `embedding_rejected` and blocks the
job, leaving the previous generation in place.

## Storage and updates

An index database is a single SQLite file. Opening it takes an exclusive lock on
`<database>.lock`, so one `ArkoIndex` serves a database at a time: a second open fails, as
does an open on a filesystem without file locking. Run jobs, status and search through
that one instance. Keep the database outside the indexed directory, so that database
writes do not wake the watcher.

A successful job publishes a generation, identified by the job ID. Its chunks, the index's
active generation and the `ready` phase are committed in one transaction. Search uses the
active generation, and an update that fails, pauses or is cancelled leaves the previous
one in place. `IndexInfo::generation` names the active generation, and
`SearchPage::fresh_at_publication` tells whether the source still matched the snapshot
when the generation was published.

`start_update(index)` rescans the source with the saved spec and reuses earlier work:

- A file whose normalized content and extension match a file of the active generation
  reuses its validated boundaries, including after a rename, as long as the chunking policy
  and the embedding profile are unchanged.
- Vectors are stored by content, so a chunk whose text was embedded before with the same
  profile reuses its vector.

`changes(job)` returns the update's file changes: `Added`, `Modified`, `Deleted`, and
`Moved` when exactly one deleted file and one added file have the same content.

Generations, events and request artifacts are kept until the index is removed with
`delete(index)`, which fails while a job is running. `read_chunks` reads any complete
generation.

A database created with an older, incompatible schema is rejected when opened; build a new
database. Complete generations in a compatible database remain searchable, and the next
update re-chunks every file if the chunking policy has changed.

## Usage and cost

`status(job).usage` is a `JobUsage`: the token usage and cost (`arko_ai::UsageInfo`) of
every chunk-model and embedding response the job received, including retries, summed
across pauses, resumes and restarts. Use it to count index spend against a budget. Query
embeddings made by `search` belong to no job.

| Member | Meaning |
|---|---|
| `billed()` | What the account was charged. ChatGPT OAuth calls are covered by the subscription, so they add zero here, while their `total_cost` shows the list-price estimate. |
| `total()` | The summed usage of chunking and embedding; `None` when no call reported usage. |
| `chunking`, `embedding` | Summed usage of each kind of call, accepted and rejected responses alike. |
| `chunk_calls`, `embedding_calls` | Accepted replies. |
| `rejected_calls` | Responses the backend rejected, once per attempt, whether or not the request was retried. |
| `unreported_calls` | Accepted replies whose backend reported no usage; their tokens and cost are not included. |

Rejected responses count because the provider bills them. `ArkoBackend` rejects refusals,
replies stopped by a token limit, and replies without exactly one well-formed call to the
boundary tool. Failures without a response (connection errors, provider error statuses,
streams cut off mid-response) and cancelled calls carry no usage and are not included,
although a provider may bill some of them.

Each usage is saved in the same transaction as the one event that carries it, so a job's
events add up to its totals, even after a crash:

- `request_finished`: `detail.usage`, from an accepted chunk reply.
- `request_attempt` in phase `failed`: `detail.event.detail.usage`, from a rejected chunk
  response. It is saved when the attempt is recorded, before any retry wait.
- `embedded` and `embedding_rejected`: `detail.usage` and `detail.rejected_usage`.
- `embedding_failed`: `detail.rejected_usage`.

Sum events rather than request artifacts: `request_artifact` repeats a rejected chunk
response's usage in `attempts[].detail.usage` and, once a retry is accepted, in
`response.usage.retry_failures[].usage`.

Custom backends report usage in `ModelReply::usage_info` and `VectorReply::usage_info`, and
attach a rejected response's usage to their error with `Error::with_usage`. Rejected
embedding attempts have no events of their own; their usage is saved with the request's
final event. Non-finite costs count as zero.

## Watching for changes

`watch(index, debounce, reconcile)` watches the index root and returns a `WatchHandle`;
`stop().await` ends it. Both intervals must be positive.

Every file-system event restarts the `debounce` timer. When it expires, the watcher scans
the source with the saved spec, compares the normalized content of every file with the
active generation, and starts an update if anything changed. Repeated writes and
atomic-save sequences therefore become one modification, and a file created and deleted
in between leaves no change. Every `reconcile` interval, a scan also runs to catch missed
events, after any active debounce period.

While a build or update of the index is running, paused or blocked, the watcher starts
no other; it checks again after each debounce period. Edits made during an update are
picked up by the next one. Scan and update errors are logged with the `log` crate, and the
watcher keeps running.

## Search

`search(index, query, options)` returns a `SearchPage` from the active generation; before
the first build completes, it returns `Error::Blocked`. `SearchOptions::mode` selects the
retrieval:

- `Hybrid` (default) combines the lexical and semantic rankings with reciprocal rank
  fusion; `score_type` is `rrf`.
- `Lexical` uses SQLite FTS5 (BM25) over each chunk's path, text and identifier parts, so
  `retry` matches `retryDelay`; a chunk matches when it contains any query term.
  `score_type` is `negative_bm25`.
- `Semantic` ranks by exact cosine similarity between the query embedding and every
  candidate vector; `score_type` is `cosine`.

| `SearchOptions` field | Default | Effect |
|---|---|---|
| `mode` | `Hybrid` | Retrieval mode. |
| `top_k` | `None` | `None` returns every eligible chunk; `Some(n)` returns a page of `n` hits with a cursor for the rest. `Some(0)` is an error; there is no upper bound. In hybrid mode, it applies to the combined ranking. |
| `min_chunk_bytes` | 0 | Skips chunks shorter than this many bytes. |
| `path_prefix` | `None` | Keeps chunks whose root-relative path starts with this string. |
| `literal` | `None` | Keeps chunks that contain this exact, case-sensitive text. |
| `cursor` | `None` | Continues from a previous page's `next`. |
| `rebuild_file_limit` | `None` | Caps the paths listed in the rebuild notice. |

Filters apply before ranking in every mode, every remaining chunk is scored, and every
query term is used. `min_chunk_bytes` affects search only; `read_chunks` still returns
short chunks.

Each `SearchHit` carries the complete chunk `text`, its `path`, `snapshot` hash,
`start_byte` and `end_byte`, `first_line` and `last_line`, `structure`, `score` and
`score_type`. A chunk whose text appears at several locations is returned once, and
`other_locations` counts the others.

`candidate_count` is the number of eligible results, and `omitted_hits` the number after
this page. `next` is a cursor pinned to the page's generation, so later pages come from
the same generation even after an update publishes a new one. A cursor works only with the
same query, `mode`, `top_k`, `min_chunk_bytes`, `path_prefix` and `literal`; otherwise
search returns `Error::Stale`.

`search_tokens` counts the tokens of the returned text with the tokenizer named in
`tokenizer`, and `rebuild_tokens` those of the rebuild notice, for callers that budget
context; neither limits the response.

While an update is running, paused or blocked, `rebuild` describes it: the job, its phase
and every affected path, unless `rebuild_file_limit` caps the list (`omitted_files` counts
the rest). `determining_changes` is true until the update has scanned the source. Results
still come from the last complete generation.

Semantic and hybrid searches embed the query with the configured Voyage profile; a query
over the per-input token limit is rejected. Query embeddings are cached in memory. The
vectors of the most recent generation and filter set are cached when they fit in
`vector_cache_bytes`; larger sets are streamed from the database, with identical results.

`read_chunks(index, generation, ids, neighbors)` returns the named chunks of a complete
generation, plus up to `neighbors` adjacent chunks on each side within the same file,
ordered by path and position. Search results already contain the full text; use
`read_chunks` for surrounding context or for chunks of a particular generation.

## LLM judge

`search_with_judge` retrieves a candidate pool with a normal search, asks a judge to
filter and rank the candidates, and then applies `top_k`. It works with lexical, semantic
and hybrid retrieval; `search` never calls a judge.

```rust,no_run
use arko_ai::{GenerationConfig, LLMClient, Model, OpenAIEffort, ReasoningConfig};
use arko_index::{ArkoIndex, JudgeOptions, LlmSearchJudge, SearchOptions};
# async fn example(index: &ArkoIndex, id: &str, client: LLMClient) -> arko_index::Result<()> {
// The client holds the judge's credentials (API key or ChatGPT OAuth).
let judge = LlmSearchJudge::new(client.clone(), Model::openai_gpt_5_6_luna())
    .with_generation_config(GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, false)))
    .with_instructions("Prefer practical implementation examples.");
let page = index.search_with_judge(id, "how do I retry a failed workflow step?",
    SearchOptions { top_k: Some(10), min_chunk_bytes: 40, ..Default::default() },
    &judge, JudgeOptions { candidate_limit: 20, max_attempts: 2 },
).await?;
# let _ = page;
# Ok(()) }
```

- The judge receives the complete text of the top `candidate_limit` chunks (default 20),
  after the search filters. `candidate_limit` must be positive and at least `top_k`;
  `top_k: None` returns every selected candidate.
- `top_k` is a maximum: the judge may select fewer candidates, including none.
- The judge selects candidate IDs through a tool call, so it cannot change source text,
  paths, ranges or scores. It must account for every candidate exactly once, either
  selected or excluded with a nonempty reason.
- Hits keep their original retrieval scores; their order is the judge's ranking.
- The built-in instructions treat source text as untrusted data. They ask the model to
  exclude metadata or import-only, navigation-only and unexpanded include-only passages,
  and to keep useful alternatives and supporting evidence.
- The judge gets up to `max_attempts` attempts (default 2) to return a valid selection; a
  refused or incomplete reply counts as invalid. Provider errors are returned at once.
  When the attempts run out, the search returns the error rather than unjudged results.
- Candidates are sent in full, so choose a model and `candidate_limit` whose context
  window fits them. The judge applies no timeout of its own.

`page.judge` is a `JudgeReport` with the judge's name, the attempts used, the examined and
unexamined candidate counts, the number selected and the number selected beyond `top_k`,
and every exclusion with its reason. `candidate_count` still counts every eligible result,
and `omitted_hits` every one missing from the response. Judged searches neither
accept nor return cursors, because each judgment covers one candidate pool; use `search`
for paging.

`LlmSearchJudge::new(client, model)` accepts any `arko-ai` model and uses its default
generation settings; set reasoning and other settings with `with_generation_config`, and
add selection preferences with `with_instructions`. Like the chunk model, the judge owns
the tool configuration: where a model cannot be forced to call the tool (for example,
Anthropic models with manual extended thinking), it uses automatic tool choice and
validates the reply in the same way. Credentials and generation settings stay with the
client and the judge, never in the index database. To use another provider, transport or
selection policy, implement the `SearchJudge` trait.

### Judge options in the example CLI

```bash
cargo run -p arko-index --example arko_index_cli -- \
  --database /path/to/index.db --oauth-file /path/to/auth.json \
  search INDEX_ID 'retry scheduling' --judge \
  --judge-model gpt-5.6-luna --judge-effort high \
  --judge-candidates 20 --top-k 10
```

`--judge` enables the judge, and every other judge option requires it. When enabled, the
judge defaults to GPT-5.6 Luna at high reasoning effort, with 20 candidates and two
attempts.

| Option | Behavior |
|---|---|
| `--judge-provider openai\|anthropic` | Judge provider; `openai` by default. |
| `--judge-model ID` | Model ID; defaults to `gpt-5.6-luna` for OpenAI and is required for Anthropic. |
| `--judge-effort LEVEL` | OpenAI reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. Requires an OpenAI reasoning model and overrides `reasoning` in `--judge-config`. |
| `--judge-candidates N` | Candidate pool size (default 20); must be at least `--top-k`. |
| `--judge-attempts N` | Attempts for invalid selections (default 2). |
| `--judge-auth inherit` | Default: reuse the global OpenAI client and its ChatGPT OAuth or API-key credentials. OpenAI judges only. |
| `--judge-auth api-key` | Read `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, independently of the global credentials. |
| `--judge-api-key-env NAME` | Read the API key from another environment variable; requires `--judge-auth api-key`. Keys are never passed as argument values. |
| `--judge-auth oauth --judge-oauth-file PATH` | ChatGPT OAuth from `PATH`, or from `--oauth-file` when no judge file is given. Rotated tokens are saved even if selection fails; the same file as `--oauth-file` reuses the global client. |
| `--judge-config PATH` | JSON with `GenerationConfig` fields (from `arko-ai`); omitted fields keep their defaults. A `reasoning` field, even `null`, replaces the high-effort default and must match the judge's provider. Unknown fields and `tool_config` are rejected. |
| `--judge-instructions-file PATH` | Append selection preferences from a UTF-8 text file; ID validation still applies. |

For example, these search options run an Anthropic judge with adaptive thinking at high
effort, on a model that supports it:

```bash
--judge --judge-provider anthropic --judge-model MODEL_ID \
  --judge-auth api-key --judge-config settings.json
```

```json
{"reasoning": {"AnthropicAdaptive": {"effort": "high"}}}
```

The file can also set `max_tokens`, `prompt_caching` and other generation settings; the
judge sets no output limit of its own.

## Requirements and testing

Building requires Rust 1.89 or later and a C compiler (SQLite is compiled from source).
The index needs a Tokio runtime and a filesystem that supports file locking for its
database.

The tests use a deterministic in-process backend, so they make no provider calls and need
no credentials:

```bash
cargo test -p arko-index --locked
```

CI runs them on Linux, macOS and Windows.
