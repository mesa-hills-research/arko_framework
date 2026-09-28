# Arko Index reference

Reference for `arko_index` 0.2.0: how the chunk model is called and its boundaries
validated, which files are indexed, jobs and their events, embeddings, storage and
updates, usage and cost accounting, watching, search, the LLM judge, and the example
CLI. The [README](../arko_index/README.md) gives an overview and a build-and-search
example.

- [Chunk model](#chunk-model)
- [Which files are indexed](#which-files-are-indexed)
- [Chunking and validation](#chunking-and-validation)
- [Jobs, scheduling and retries](#jobs-scheduling-and-retries)
- [Embeddings](#embeddings)
- [Storage and updates](#storage-and-updates)
- [Usage and cost](#usage-and-cost)
- [Watching for changes](#watching-for-changes)
- [Search](#search)
- [LLM judge](#llm-judge)
- [Example CLI](#example-cli)

## Chunk model

`ArkoBackend::new` selects boundaries with GPT-5.6 Luna at high reasoning effort, with
reasoning summaries and low verbosity (`backend::default_chunk_config()`).
`with_chunk_model` selects another `arko_ai` model and its generation settings; the client
passed to `ArkoBackend::new` must hold credentials for that model's provider:

```rust
use arko_ai::{GenerationConfig, Model, OpenAIEffort, ReasoningConfig};

let backend = backend.with_chunk_model(
    Model::openai_gpt_6_luna(),
    GenerationConfig::default().with_reasoning(ReasoningConfig::openai(OpenAIEffort::Medium, true)),
);
```

The backend owns the tool configuration: the boundary tool replaces any `tool_config` in
the settings, the call is forced where the model allows it, and parallel tool calls are
disabled where the provider supports that. Every reply must be exactly one call to that
tool in a completed turn (`StopReason::EndTurn` or `StopReason::ToolUse`):

- A refusal, or a reply stopped by a token or context limit, fails the request.
- A missing, extra or malformed tool call is retried as a transient failure.

Every request of a job, initial requests and repairs, before and after a resume, carries
the job's ID as its prompt-cache key (`ModelRequest::cache_key`), and puts the shared
instructions first, so the provider can reuse their cached prefix across the job. Over
ChatGPT OAuth the key is also the `session-id` header the backend routes cache affinity
by. Caching follows the model's provider; see the `arko_ai` README.

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

```rust
use arko_index::BuildSpec;

let spec = BuildSpec {
    include: vec!["docs/**/*.md".into(), "src/**/*.rs".into()],
    exclude: vec!["docs/archive/**".into()],
    ..BuildSpec::new("docs-and-code", "/path/to/project")
};
```

A file that matches an `exclude` pattern is skipped. With `include` patterns, exactly the
files matching one of them are indexed, whatever their extension. Without them, a file is
indexed when its name is `Dockerfile`, `Makefile`, `CMakeLists.txt`, `Justfile`,
`LICENSE`, `README` or `AGENTS.md`, or when its extension (in any case) is one of `rs`,
`py`, `pyi`, `js`, `jsx`, `ts`, `tsx`, `mjs`, `cjs`, `go`, `c`, `h`, `cc`, `cpp`, `hpp`,
`cs`, `java`, `kt`, `swift`, `rb`, `php`, `lua`, `sh`, `ps1`, `sql`, `ex`, `exs`, `erl`,
`hs`, `scala`, `vue`, `svelte`, `html`, `css`, `scss`, `md`, `mdx`, `rst`, `txt`, `adoc`,
`toml`, `yaml`, `yml`, `json`, `xml`, `proto`, `graphql` or `tf`.

Files larger than `BuildSpec::max_file_bytes` (default 16 MiB, `DEFAULT_MAX_FILE_BYTES`)
are skipped without being read, with the reason "too large (over max_file_bytes)". Files
that are not valid UTF-8 or that contain NUL bytes are skipped. An entry the scan can't
read does not stop it: a directory or file without read permission ("unreadable:
permission denied"), a file removed while the scan runs ("removed during the scan"), an
ignore file with invalid rules, and a file whose path is not valid UTF-8 ("path is not
valid UTF-8", reported with the invalid bytes replaced) are skipped with that reason.
`source::scan` reports symbolic links and skipped files with a reason in
`Manifest::excluded`, and each job's `snapshot` event counts them; directories skipped by
name or by ignore rules are not reported. The spec is saved with the index, so
updates and the watcher scan with the same settings.

## Chunking and validation

The chunk model partitions each file into coherent groups with no preferred or maximum
size. Complete functions, methods, impl blocks, classes and type definitions stay
together with their comments, whatever their length; documentation keeps each topic
section and worked example together with the introduction and instructions needed to use
it. The guidance also asks the model to keep import blocks, related module declarations,
re-exports and constants, and adjacent helpers with the code they support. That grouping
is model-guided; the structural guards below are what the index enforces.

Each initial request carries up to `batch_count` files, normalized to LF line endings and
shown with absolute line numbers, within the chunk request budget below. The model
answers with cuts: an absolute line number and an exact suffix of the text ending on that
line. A cut is valid when the suffix matches, the line is not blank, the cuts are in
strictly increasing order, and the cut does not fall inside a guarded unit:

| Files | `structure` | Guarded units |
|---|---|---|
| `.rs` files that parse with `syn` | `rust-guarded` | Functions and methods, impl blocks, traits, structs, enums, modules, `use` declarations (including multiline trees), statements and string literals, each with its attributes and doc comments. |
| `.md` and `.mdx` files | `markdown-guarded` | Paragraphs, code blocks, list items, headings, and each heading together with the block that follows it. |
| Other files, and Rust files that do not parse | `unchecked` | None; cuts are checked against the line text only. |

### Request budget and large files

`BuildSpec::chunking` (`ChunkingOptions`) sizes chunk-model requests. A request's size is
its rendered file content, each file's header line and numbered rows as the model sees
them; the fixed instructions and tool schema (under 1,000 tokens) come on top.

- **What is measured.** Chunk models of the OpenAI provider are measured in `o200k_base`
  tokens, the encoding GPT-4o and later, GPT-5.x and GPT-6 use. It is built into the
  crate, so counting needs no download, and special-token text such as `<|endoftext|>`
  counts as ordinary text, as the API counts it. Other chunk models are measured in bytes.
  Lines longer than 128 KiB count as their byte length, which is never less than their
  token count.
- **Default budget.** 10% of the chunk model's input limit on the client's auth path
  (`LLMClient::max_input_tokens`): 92,200 tokens for GPT-5.6 and GPT-6 with an API key,
  27,200 over ChatGPT OAuth (whose limit is 272,000), and 4 bytes per token of that share
  for models measured in bytes (about 400 KB for a 1M-token Claude model). A model whose
  limit is unknown, such as an OpenRouter model built without the catalog or a custom
  backend, is assumed to accept 128,000 tokens (`budget::FALLBACK_MAX_INPUT_TOKENS`), so
  its default is 51,200 bytes; the job logs a warning. A custom `Backend` reports its
  model's tokenizer and input limit with `Backend::chunk_input`.
- **Explicit limits.** `max_request_tokens` sets the token limit (for models measured in
  bytes, it applies as 4 bytes per token), and `max_request_bytes` a byte limit; when both
  are set, both apply.
- **Batches.** Initial requests are filled with up to `batch_count` files as long as their
  combined size fits the budget.
- **Resumes.** A job resolves its budget when it scans the source, records it as
  `chunk_budget` in its `snapshot` event, and keeps it when resumed, even if the client's
  auth path has changed since.

A file whose rendering exceeds the budget is handled by `ChunkingOptions::oversize`
(`OversizePolicy`):

| Policy | Effect |
|---|---|
| `Split` (default) | The file is split into windows that fit, each sent as its own part of a request. Boundaries are tried from strongest to weakest: Markdown headings (level 1, then level 2, and so on), then top-level blocks, never inside a fenced code block; in Rust files that parse, top-level items, then members of impls, traits and modules, then statements of function bodies; in other files, lines after a blank line and lines at the smallest indentation, at the outermost bracket depth first; then single lines. Boundaries that cut no guarded unit are preferred. Pieces are of similar size, and a small remainder, such as a lone closing brace, joins its neighbor. A window edge is also a chunk edge. A single line that exceeds the budget on its own becomes one chunk without a model request. |
| `Skip` | The file is left out of the index. The job's exclusions list it with the reason "too large (over the chunk request budget)", and `JobStatus::skipped` counts it. |
| `Block` | The job is blocked before any model request, with a problem that names the files. |

A file that is split or skipped gets an `oversize` event with its size, the budget, the
action taken, the number of windows and any single lines made into their own chunks.
A file with no legal internal boundary becomes one chunk without a model request (counted
in `JobStatus::bypassed`), so `Split` leaves it whole; `Skip` and `Block` apply to it too
when it exceeds the budget. Empty and whitespace-only files produce no chunks.

### Repairs

Invalid cuts are repaired without changing accepted ones. If there is no legal boundary
between the neighboring accepted cuts, the disputed cuts are removed without a model
request (a `deterministic_repair` event). Otherwise a repair request sends the window's
rows (the complete file when it is one window) with the legal boundary IDs of each
disputed region, and the model returns an ordered selection of boundaries, or `[]` to
merge the region into one chunk. A repair applies only to the snapshot and cut set it was
requested for.

A window gets up to `ChunkingOptions::max_repair_rounds` repair requests (5 by default;
`None` repairs until every cut is valid). When its cuts are still invalid after that,
`ChunkingOptions::repair_limit` (`RepairLimitPolicy`) decides:

- `Merge` (default): the invalid cuts are dropped, merging each disputed region into its
  neighboring chunk, which is always valid; accepted cuts stay, and a
  `deterministic_repair` event records the merge.
- `Block`: the file stays unfinished and the job is blocked, naming it; resuming the job
  allows as many repairs again.

Source files are never modified. All stored text comes from the normalized snapshot, and
a file's chunks cover it exactly, in order, with no gaps or overlaps.

The chunking policy (the prompts and a policy version) has an identity, recorded as
`chunking_policy` in each `snapshot` event. When it changes, the next update re-chunks
every file.

## Jobs, scheduling and retries

`start_build(spec)` and `start_update(index)` return a `JobHandle` (`index` and `job`
IDs) at once and run the job in the background. `status(job)` returns a `JobStatus` with
the phase, the last `problem`, file counts (`files`, `validated`, `bypassed`,
`unchecked`, `skipped`), the requests in flight and the job's usage. A job moves through
`snapshotting`, `chunking` and `embedding` to `ready`, or stops as:

- `paused`: after `pause(job)`, when a plan usage limit was reached, or when its process
  stopped mid-run (detected when the database is next opened, with the problem "process
  interrupted; resume persisted work", or "paused" if a pause had been requested);
- `blocked`: after a provider error, an invalid response, or with files whose boundaries
  still need repair. When only transient provider failures that ran out of attempts
  blocked it, its `phase` event has `retryable: true`;
- `cancelled`: after `cancel(job)`;
- `failed`: after a local error, such as an I/O or database failure.

`resume(job)` continues a paused, blocked or failed job from its checkpoints. Settings
that shape the stored output must match the job's: its `Config::embedding` profile, and
its chunking policy and request budget, which the job saved when it scanned the source
(the budget is reused as saved). Every other `Config` field is a runtime setting and is
taken from the current configuration; a change is logged, recorded as a
`settings_changed` event and saved with the job. If the embedding profile differs, resume
fails; if the chunking policy has changed, or a newer generation of the index was
published since the job started, resume fails too. In these cases, cancel the job and
start a new update. An index has at most one unfinished job: `start_update` fails while a
build or update of the index is running, paused or blocked.

| `Config` field | Default | Meaning |
|---|---|---|
| `batch_count` | 6 | Most files per initial chunk-model request, within the request budget. |
| `concurrency` | 2 | Chunk-model requests in flight, shared by all jobs of the `ArkoIndex`. |
| `embedding_concurrency` | 2 | Embedding requests in flight, including search query embeddings. |
| `embedding` | `voyage-4`, 1,024 dimensions | Embedding profile; see [Embeddings](#embeddings). |
| `vector_cache_bytes` | 64 MiB | Memory for caching search vectors; see [Search](#search). |
| `max_transient_attempts` | 10 | Transient failures allowed per request before it fails. |
| `max_capacity_wait_ms` | 15 minutes | Longest total wait per request for capacity signals; `None` waits as long as the provider is overloaded. |
| `query_priority` | `false` | Opt-in: dispatch search query embeddings before queued build embeddings. |
| `query_max_transient_attempts` | `None` | Opt-in: a smaller transient attempt budget for search query embeddings; `None` uses `max_transient_attempts`. |
| `embedding_batch_tokens` | `None` | Opt-in: fewer tokens per embedding request than the model allows; see [Embeddings](#embeddings). |
| `embedding_batch_items` | `None` | Opt-in: fewer inputs per embedding request than the model allows. |
| `keep_generations` | `None` | Opt-in: prune complete generations beyond the newest `n`; see [Storage and updates](#storage-and-updates). |
| `synchronous` | `Full` | SQLite `synchronous` mode; `Normal` is an opt-in; see [Storage and updates](#storage-and-updates). |
| `stream_idle_timeout_ms` | `None` | Opt-in: the longest gap between data on a chunk-model response before it is cut off; see below. |

`batch_count`, both concurrency limits and `max_transient_attempts` must be positive, and
`query_max_transient_attempts` too when set. A finished request slot is refilled
immediately, and initial chunk requests take priority over repairs and retries.

Transient provider failures (rate limits, timeouts, server errors, truncated streams and
malformed tool calls) are retried with jittered exponential backoff. The backoff step
starts at 1 s and doubles up to 60 s; each wait is drawn between half and all of the
step, and is never shorter than a retry delay the provider asks for. After
`max_transient_attempts` transient failures, the request fails with its last error, a
`retries_exhausted` attempt event is recorded, and the job becomes `blocked`. Explicit
capacity signals (overloaded or unavailable) are retried on the same schedule without
consuming attempts, until one request has waited `max_capacity_wait_ms` (15 minutes by
default) for them in all; then it fails the same way. Other errors, such as
authentication failures, invalid requests and refusals, are not retried.

Chunk-model requests have no timeout by default: a response is waited for as long as its
connection stays open. This is deliberate. Models can reason silently for a long time,
GPT Pro models for over an hour, and a stream may carry nothing until the reasoning ends.
The opt-in `Config::stream_idle_timeout_ms` cuts a response off once no data has arrived
for that long (arko_ai's `LLMClient::with_stream_idle_timeout`, which `ArkoBackend`
applies to each chunk request). The cut is a truncated stream, a transient failure that
is retried, and the provider may bill the work that was cut off. A value shorter than the
longest silent stretch of the chunk model cuts healthy responses off, so make it generous
(for example two hours, 7,200,000). Only the gaps between data count; there is no total
deadline. It is a runtime setting: a resumed job takes it from the current configuration.
`LlmSearchJudge::with_stream_idle_timeout` does the same for the judge.

A plan usage limit (ChatGPT `usage_limit_reached`) is not retried: the request records a
`usage_limit` attempt event, the job's queued requests stop before another one is sent,
requests already in flight finish, and the job becomes `paused` with a problem such as "usage limit reached; resets at
2026-09-28T14:05:00Z (in about 3h 12m). Resume the job after the reset." followed by the
provider's error. The watcher leaves such a job for the host; `resume(job)` continues it.

`pause(job)` stops queued requests and ends retry waits at once, while requests already in
flight finish and are checkpointed. `cancel(job)` also drops in-flight requests. A job
that is being cancelled cannot be paused.

`events(job, after, limit)` returns up to `limit` events with a sequence number greater
than `after`; pass the last sequence seen to continue. Each `Event` has a `kind` and a
JSON `detail`:

| Kind | Detail |
|---|---|
| `phase` | The new `phase` and `problem`, and `retryable: true` for a job blocked only by transient provider failures. |
| `snapshot` | Counts of scanned, excluded and affected files; the manifest hash; `chunking_policy` and `policy_changed`; the chunk `model` and `reasoning`; the resolved `chunk_budget` (`tokenizer`, `max_tokens`, `max_bytes`, the model's `max_input_tokens`, whether it was known, and the `oversize` policy); the number of `oversize_files` and `skipped` files. |
| `oversize` | A file over the chunk request budget that was split or skipped: its `path`, rendered `bytes` and `tokens`, the `budget`, the `action` (`split` or `skipped`), its `windows` and the `forced_rows` made into their own chunks. Under `Block` no events are written; the problem names the files. |
| `request_attempt` | `artifact` (request ID), `repair`, and `event`: `phase` (`started`, `failed`, `retry_wait`, `retries_exhausted`, `usage_limit` or `completed`), `attempt`, `unix_ms` and a phase-specific `detail`, such as `error` and `usage` for a failure or `delay_ms` and `capacity` for a retry wait. |
| `request_finished` | `artifact`, `repair` and the `usage` of the accepted reply. |
| `deterministic_repair` | Cuts removed without a model request. |
| `pause_requested` | A pause was requested. |
| `pruned` | Generations pruned after this job published (`generations`) and the unreferenced data removed (`removed`). |
| `settings_changed` | A resume with other runtime settings: `changes`, each with the `setting` and its `from` and `to` values. |
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
the Hugging Face Hub. It accepts the Voyage models of the `arko_ai` registry at a
dimension the model supports, with their published limits: for `voyage-4`, 32,000 tokens
per input, and 320,000 tokens and 1,000 inputs per request. Voyage publishes no
per-request token limit for `voyage-code-4`, the code retrieval model; requests to it are
limited to 120,000 tokens (`backend::UNPUBLISHED_BATCH_TOKENS`), the smallest per-request
limit Voyage publishes for any current model. `ArkoIndex::open` requires the backend's
profile to equal `Config::embedding`.

`Config::embedding_batch_tokens` and `Config::embedding_batch_items` lower the tokens and
inputs of one embedding request, for example to stay within an account's rate limit. They
can't raise the model's limits, and the token setting can't be below the per-input limit;
`ArkoIndex::open` rejects such values.

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
when the generation was published; it is false when that rescan failed, which doesn't
stop the publication.

`start_update(index)` rescans the source with the saved spec and reuses earlier work:

- A file whose normalized content and extension match a file of the active generation
  reuses its validated boundaries, including after a rename, as long as the chunking policy
  and the embedding profile are unchanged. Files the newest cancelled or failed job of the
  index finished chunking are reused the same way, so cancelling a blocked build and
  starting an update pays only for the files that were not finished.
- Vectors are stored by content, so a chunk whose text was embedded before with the same
  profile reuses its vector.

`changes(job)` returns the update's file changes: `Added`, `Modified`, `Deleted`, and
`Moved` when exactly one deleted file and one added file have the same content.

Storage is shared where content is the same. Snapshots are stored once per normalized
content, vectors once per embedded text, and a chunk's full-text entry once per location
and content, so generations that keep a file unchanged share its chunk texts; each
generation adds only its chunk and file-state rows.

Generations, events and request artifacts are kept until the index is removed with
`delete(index)`, which fails while a job is running. `read_chunks` reads any complete
generation. `delete` also removes the data no remaining index uses: chunk texts no chunk
uses, snapshots no file state or chunk uses, and, when no job of any index is unfinished,
vectors no chunk uses (an unfinished job may rely on vectors it has not published yet).
Data another index shares is kept.

`Config::keep_generations` is an opt-in retention setting. With `Some(n)`, each publish
prunes the index's complete generations beyond the newest `n`: their chunks, file states
and request artifacts are removed, the job keeps its events and usage with the phase
`pruned` (and a `phase` event), unreferenced data is removed as by `delete`, and the
publishing job records a `pruned` event listing the generations and what was removed.
Pruning is part of the publishing transaction, so once a job is `ready` the older
generations are pruned; if pruning fails, they are kept and a warning is logged.
`read_chunks` and cursors for a pruned generation return `Error::Stale`. The default,
`None`, keeps every generation.

`Config::synchronous` sets SQLite's `synchronous` mode. `Full` (the default) syncs every
commit to disk. `Normal`, an opt-in, syncs at WAL checkpoints: fewer disk syncs while jobs
record progress, at the risk of losing the last commits (never corrupting the database) on
power loss or an operating-system crash; jobs resume from their checkpoints.

A database of the first schema version is migrated when opened, in one transaction that
leaves it unchanged if anything fails: each generation's full-text copies of its chunks
become shared chunk texts, and everything else is kept as it was. The migration does not
shrink the file; SQLite reuses the freed pages. A database with an unknown schema version
is rejected; build a new database. Complete generations remain searchable, and the next
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

`watch(index, options)` watches the index root and returns a `WatchHandle`;
`stop().await` ends it, and `subscribe()` receives its `WatchEvent`s. `WatchOptions`:

| Field | Default | Meaning |
|---|---|---|
| `debounce` | 500 ms | How long the source must be quiet after a file-system event before it is scanned. |
| `reconcile` | 30 s | Interval of full scans that catch events the file system did not report; `None` scans only at start and after events. |
| `resume_interrupted` | `true` | Resume an update whose process stopped mid-run. |
| `retry_blocked_after` | `None` | Opt-in: resume an update blocked by transient provider failures this long after the watcher found it blocked. |

Every relevant file-system event restarts the `debounce` timer. When it expires, the
watcher scans the source with the saved spec, compares the normalized content of every
file with the active generation, and starts an update if anything changed. Repeated
writes and atomic-save sequences therefore become one modification, and a file created
and deleted in between leaves no change. The watcher also scans once at start and every
`reconcile` interval, after any active debounce period.

Events under directories the scan does not enter are ignored: directories named in
`skip_directories` (such as `target` or `node_modules`) and directories excluded by ignore
rules, as found by the last scan. Activity there, such as a build or a package install,
causes no scan.

Updates stay serialized. While a build or update of the index is running, the watcher
checks again after each debounce period, so edits made during an update are picked up by
the next one. An unfinished update found paused or blocked is handled as follows:

- **Interrupted** (paused because its process stopped mid-run, detected when the database
  was opened): resumed, unless `resume_interrupted` is off.
- **Blocked by transient provider failures** (rate limits, server errors or capacity
  waits that ran out of attempts; its `phase` event has `retryable: true`): resumed after
  `retry_blocked_after` when that is set.
- **Otherwise** (paused with `pause(job)` or by a usage limit, blocked for another
  reason, or one the watcher failed to resume): left for the host, and no other update
  starts until it is resumed or cancelled. Without periodic scans (`reconcile: None`) the
  watcher still looks at a held update every twenty debounce periods, so edits made in
  the meantime are indexed once the host resumes or cancels it.

| `WatchEvent` | Meaning |
|---|---|
| `Scanned { files, changes }` | A scan finished; `changes` files differ from the active generation. |
| `UpdateStarted { job }` | The watcher started an update. |
| `Resumed { job, reason }` | The watcher resumed an update; `reason` is `interrupted` or `blocked`. |
| `Held { job, phase, problem }` | A paused or blocked update the watcher leaves alone; reported once per job, phase and problem. |
| `Failed { error }` | A scan, update or resume failed; the watcher keeps running. |

Events are also logged with the `log` crate: started and resumed updates at info level,
held updates and failures as warnings. A subscriber that falls more than 64 events behind
misses the oldest.

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
| `top_k` | `Some(10)` | A page of `n` hits with a cursor for the rest; `None` returns every eligible chunk. `Some(0)` is an error; there is no upper bound. In hybrid mode, it applies to the combined ranking. |
| `min_chunk_bytes` | 40 | Skips chunks shorter than this many bytes; 0 keeps every chunk. |
| `path_prefix` | `None` | Keeps chunks whose root-relative path starts with this string. |
| `literal` | `None` | Keeps chunks that contain this exact, case-sensitive text. |
| `cursor` | `None` | Continues from a previous page's `next`. |
| `rebuild_file_limit` | `Some(100)` | Caps the paths listed in the rebuild notice; `None` lists every one. |

Filters apply before ranking in every mode, every remaining chunk is scored, and every
query term is used. Ranking reads only the ranking rows from the database; chunk text and
locations are loaded for the returned page alone. `min_chunk_bytes` affects search only;
`read_chunks` still returns short chunks.

Each `SearchHit` carries the complete chunk `text`, its `path`, `snapshot` hash,
`start_byte` and `end_byte`, `first_line` and `last_line`, `structure`, `score` and
`score_type`. A chunk whose text appears at several locations is returned once, and
`other_locations` counts the others.

`may_be_stale` marks a hit whose file may have changed since it was indexed: an unfinished
update (running, paused or blocked) lists the file as affected, or the file under the
index root no longer matches the chunk's snapshot, because it changed, was removed, can't
be read or now exceeds `max_file_bytes`. Search reads the files of the returned hits to
check; when the index root is not accessible, only unfinished updates mark hits. The
text of a marked hit is still the indexed text; read the file for its current content.

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
over the per-input token limit is rejected. Query embeddings share the embedding request
slots with running jobs and wait behind queued build embeddings; `Config::query_priority`
dispatches them first, and `Config::query_max_transient_attempts` gives them a smaller
retry budget. They return `Error::Invalid` when the generation was embedded with another
profile (model, dimensions or tokenizer revision), since its vectors can't be compared
with the query's; lexical search still works, and an update publishes a generation with
the configured profile. Query embeddings are cached in memory. The vectors of the most
recent generation and filter set are cached when they fit in `vector_cache_bytes`; larger
sets are streamed from the database, with identical results.

`read_chunks(index, generation, ids, neighbors)` returns the named chunks of a complete
generation, plus up to `neighbors` adjacent chunks on each side within the same file,
ordered by path and position. Search results already contain the full text; use
`read_chunks` for surrounding context or for chunks of a particular generation.

## LLM judge

`search_with_judge` retrieves a candidate pool with a normal search, asks a judge to
filter and rank the candidates, and then applies `top_k`. It works with lexical, semantic
and hybrid retrieval; `search` never calls a judge.

```rust
use arko_ai::{GenerationConfig, Model, OpenAIEffort, ReasoningConfig};
use arko_index::{JudgeOptions, LlmSearchJudge, SearchOptions};

// The client holds the judge's credentials (API key or ChatGPT OAuth).
let judge = LlmSearchJudge::new(client.clone(), Model::openai_gpt_5_6_luna())
    .with_generation_config(GenerationConfig::default()
        .with_reasoning(ReasoningConfig::openai(OpenAIEffort::High, false)))
    .with_instructions("Prefer practical implementation examples.");
let page = index.search_with_judge(&index_id, "how do I retry a failed workflow step?",
    SearchOptions { top_k: Some(10), min_chunk_bytes: 40, ..Default::default() },
    &judge, JudgeOptions { candidate_limit: 20, max_attempts: 2, ..Default::default() },
).await?;
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
  refused or incomplete reply counts as invalid. Transient provider failures (rate limits,
  server errors, capacity) are retried with the index's backoff, never sooner than a
  delay the provider asks for, until `max_transient_attempts` (default 3) have failed.
  A provider asking for a wait longer than 60 seconds (`MAX_JUDGE_RETRY_WAIT`) fails the
  search instead. Other provider errors, plan usage limits included, are returned at once.
  When the
  attempts run out, the search returns the error rather than unjudged results.
- Candidates are sent in full, so choose a model and `candidate_limit` whose context
  window fits them. The judge applies no timeout of its own.

`page.judge` is a `JudgeReport` with the judge's name, the attempts used (judge calls,
retries included), the summed `usage` of the judge's responses (rejected replies included,
when the judge reports usage), the examined and unexamined candidate counts, the number
selected and the number selected beyond `top_k`, and every exclusion with its reason. The
judge's usage is not part of any job's usage. `candidate_count` still counts every
eligible result, and `omitted_hits` every one missing from the response. Judged searches
neither accept nor return cursors, because each judgment covers one candidate pool; use
`search` for paging.

`LlmSearchJudge::new(client, model)` accepts any `arko_ai` model and uses its default
generation settings; set reasoning and other settings with `with_generation_config`, and
add selection preferences with `with_instructions` (appended to the built-in instructions
under "Additional retrieval preferences"; nothing is appended without them). Like the
chunk model, the judge owns the tool configuration: where a model cannot be forced to call
the tool (for example, Anthropic models with manual extended thinking), it uses automatic
tool choice and validates the reply in the same way. Credentials and generation settings
stay with the client and the judge, never in the index database. To use another provider,
transport or selection policy, implement the `SearchJudge` trait; implement
`SearchJudge::select_with_usage` as well to report usage.

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

| Command | Action |
|---|---|
| `build ROOT [--name NAME]` | Build a new index (the name defaults to the root directory's name) and stream its events. |
| `update INDEX_ID` | Start an update and stream its events. |
| `resume JOB_ID` | Resume a paused, blocked or failed job and stream its events. |
| `status JOB_ID` | Print the job's status, including usage. |
| `list` | List indexes and their active generations. |
| `search INDEX_ID QUERY` | Search; see the options below and [Judge options](#judge-options). |
| `read INDEX_ID GENERATION CHUNK_ID...` | Print chunks from a generation (the `generation` of a search result). |
| `watch INDEX_ID [--debounce-ms MS] [--reconcile-secs S]` | Keep the index up to date until Ctrl+C (500 ms debounce; a reconciliation scan every 30 s, or only at start and after events with `--reconcile-secs 0`), printing each watcher event as a JSON line. |

Options before the command: `--database PATH` (required), `--oauth-file PATH`,
`--batch-count N` (default 6), `--concurrency N` (default 2), and the opt-in
`--stream-idle-timeout-secs S` (`Config::stream_idle_timeout_ms`, also applied to the
judge; off by default, and if set, make it generous: see
[Jobs, scheduling and retries](#jobs-scheduling-and-retries)). A resumed job uses the
values given to `resume`.

`build`, `update` and `resume` print events as JSON lines until the job is ready; Ctrl+C
pauses the job so that `resume` can continue it later. `watch` prints the watcher's events
(`scanned`, `update_started`, `resumed`, `held`, `failed`). Log messages go to standard
error: warnings from every crate and the index's progress at info level, such as the
updates the watcher starts; set `RUST_LOG` to change that. The database lock admits one
process at a time, so other commands on the same database fail while one of these, or
`watch`, is running.

`search` returns the top 10 results by default (`--top-k N`) and hides chunks shorter than
40 bytes (`--min-chunk-bytes 0` includes them), as the library defaults do. `--lexical`
selects full-text retrieval only and `--semantic` vector retrieval only; otherwise search
is hybrid. `--rebuild-file-limit N` caps the paths listed in a rebuild notice (default
100).

`list`, `status` and `read` make no provider requests, so they need no credentials and
load no tokenizer. The other commands load the Voyage tokenizer; those that embed need
`VOYAGE_API_KEY`, and those that chunk or judge need OpenAI credentials, from
`--oauth-file`, or else from the environment: `OPENAI_OAUTH_ACCESS_TOKEN` and
`OPENAI_OAUTH_ACCOUNT_ID` (with an optional `OPENAI_OAUTH_REFRESH_TOKEN`) for ChatGPT
OAuth, or `OPENAI_API_KEY`. The OAuth file uses the Codex `auth.json` layout, with
`access_token`, `refresh_token` and `account_id` at the top level or under `tokens`, and
can be the one Codex uses: every token refresh reads the file again first, adopting tokens
another process refreshed in the meantime, and rotated tokens are saved to it at once,
atomically (the previous contents are kept in `<file>.bak`); see [ChatGPT OAuth
credentials](../arko_ai/README.md#chatgpt-oauth-credentials). Rotated tokens from
environment credentials are not saved.

### Judge options

```bash
cargo run -p arko_index --example arko_index_cli -- \
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
| `--judge-attempts N` | Attempts for invalid selections (default 2); transient provider errors are retried separately, up to three failures. |
| `--judge-auth inherit` | Default: reuse the global OpenAI client and its ChatGPT OAuth or API-key credentials. OpenAI judges only. |
| `--judge-auth api-key` | Read `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, independently of the global credentials. |
| `--judge-api-key-env NAME` | Read the API key from another environment variable; requires `--judge-auth api-key`. Keys are never passed as argument values. |
| `--judge-auth oauth --judge-oauth-file PATH` | ChatGPT OAuth from `PATH`, or from `--oauth-file` when no judge file is given, kept in step with the file like `--oauth-file`; the same file as `--oauth-file` reuses the global client. |
| `--judge-config PATH` | JSON with `GenerationConfig` fields (from `arko_ai`); omitted fields keep their defaults. A `reasoning` field, even `null`, replaces the high-effort default and must match the judge's provider. Unknown fields and `tool_config` are rejected. |
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
