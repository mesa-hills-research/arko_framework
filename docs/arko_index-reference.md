# arko_index reference

arko_index partitions source trees into syntax-aware chunks with tree-sitter, builds
offline BM25 indexes over them, optionally embeds the chunks for dense and hybrid
search, and keeps chunk snapshots current with durable watchers. The library makes no
network calls and depends on no other arko crate: embeddings come from a caller's
`dense::Embedder` (its examples adapt arko_ai's Voyage client, a dev-dependency). The
[crate README](../arko_index/README.md) explains the behaviour (which files are indexed,
how boundaries and IDs are chosen, context headers, dense and hybrid search, the
watchers' durability); this page lists the API, the CLI, the environment variables,
the defaults and the on-disk state.

## Modules

| Module | What it holds | Code |
|---|---|---|
| `arko_index::bench_v1` | Chunking and the polling watcher: `Options`, `SourceSpan`, `Chunk`, `FileMetadata`, `FileChunks`, `CHUNK_FORMAT`, `Delta`, `Error`, `Watcher`, `chunk_str`, `chunk_str_with_metadata`, `chunk_file`, `chunk_file_with_metadata`, `chunk_dir`, `chunk_dir_each`, `chunk_dir_map`, `open_watch`, the `_with` variants; re-exports `FileRules`, `Report`, `SkipReason`, `Skipped`. | `arko_index/src/bench_v1.rs`, `arko_index/src/watch.rs`, `arko_index/src/syntax.rs`, `arko_index/src/languages.rs`, `arko_index/src/rules/`, `arko_index/src/headers.rs`, `arko_index/src/front_matter.rs` |
| `arko_index::search` | The first index format (one JSON file): `index`, `index_with`, `search`, `outline`. | `arko_index/src/search.rs` |
| `arko_index::index2` | The disk-resident index: `index`, `index_with`, `build`, `BuildOptions`, `search`, `dense_info`, `BuildStats`. | `arko_index/src/index2.rs` |
| `arko_index::dense` | Embeddings: `Embedder`, `ModelDescriptor`, `EmbedLimits`, `Embedding`, `InputKind`, `HashEmbedder`, `embed_query`, `render`, `RENDER_VERSION`, `DenseConfig`, `DenseBuild`, `EmbedProgress`, `EmbedStats`, `DenseIndexInfo`. | `arko_index/src/dense/` |
| `arko_index::vector` | Vector storage: `VectorStore`, `BruteForceStore`, `Quantization`, `Filter`, `VectorMeta`, `VectorHit`. | `arko_index/src/vector.rs` |
| `arko_index::hybrid` | Lexical, dense and hybrid search: `search`, `SearchRequest`, `Mode`, `Fusion`, `fuse`, `Fused`, `SearchHit`, `SearchResults`. | `arko_index/src/hybrid.rs` |
| `arko_index::watch2` | Linux only: `Watcher2`, the inotify watcher. | `arko_index/src/watch2.rs` |
| crate root | `FileRules`, `Report`, `SkipReason`, `Skipped`, `threads()`, and on Unix `bench_scan`, `bench_scan_with`. | `arko_index/src/lib.rs`, `arko_index/src/files.rs`, `arko_index/src/par.rs`, `arko_index/src/scan.rs` |

Errors are `bench_v1::Error`, a boxed `std::error::Error + Send + Sync`.

## Chunking

| Item | What it does |
|---|---|
| `Options { max_bytes, front_matter_description }` | `max_bytes`: the source-byte bound of a chunk. Default 4,096; values below 4 are rejected. A rendered chunk (path, newline, source) is also capped at 24,000 bytes. `front_matter_description` (default `false`): Markdown scopes start with the front matter's title and description instead of the title alone. |
| `chunk_str(path, text, options)` | Chunks one text under the given path (kept verbatim; the extension or file name selects the grammar). |
| `chunk_str_with_metadata(path, text, options)` | The same, as `FileChunks { metadata, chunks }`: a Markdown file's metadata region (`FileMetadata`) with its chunks. |
| `chunk_file(path, options)`, `chunk_file_with_metadata(path, options)` | Reads and chunks one file under its file name. A binary or non-UTF-8 file gives no chunks. |
| `chunk_dir(root, options)`, `chunk_dir_with(root, options, rules)` | Every file the rules select, in path order; `_with` also returns the `Report`. |
| `chunk_dir_each[_with](root, options, [rules,] sink)` | Streams each file's chunks to `sink` in path order, chunking on `threads()` workers. |
| `chunk_dir_map[_with](root, options, [rules,] threads, map, sink)` | Like `chunk_dir_each`, applying `map` to each file's `FileChunks` on the worker (the CLI renders JSON there). Memory is bounded by a reorder window of source bytes. |

Languages: 50, from extensions and well-known file names (`Makefile`,
`CMakeLists.txt`, `PKGBUILD`, ...), listed with each language's rules in
[arko_index languages](arko_index-languages.md). Anything else is `text`, split by the
bounded fallback.

**`Chunk` fields** (one JSON object per line in CLI output, in this order): `id`
(SHA-256 based, from path plus trimmed content and an occurrence ordinal), `path`
(relative, slash-separated), `start_byte`, `end_byte` (half-open UTF-8 byte range),
`start_line`, `end_line` (1-based, inclusive), `language`, `symbol_path` (enclosing
symbols), `kind` (the syntax kind, `function_family` for a merged pair, or a fallback
kind), `token_estimate` (a conservative byte upper bound of the path, a newline and
the source), `scope` (the context: enclosing definitions such as
`impl Display for Point › fn fmt`, or a Markdown heading chain; written only when not
empty). `Chunk::header()` gives `path:start_line-end_line · scope`.

**`FileMetadata` fields** (a JSON line before the file's chunks in CLI output, format
3): `file` (the path), `metadata_range` (a `SourceSpan` from byte 0: `start_byte`,
`end_byte`, `start_line`, `end_line`), `front_matter` (the raw block, delimiters
included; absent when empty), `title` and `description` (absent when the front matter
has none). Only Markdown files with front matter or MDX import/export lines at the top
have one. Every byte of a file is in exactly one chunk or in this region.

**Chunk format versions** (`bench_v1::CHUNK_FORMAT`, now 3): format 1 is the ten fields
up to `token_estimate`; format 2 appends `scope`. Boundaries and IDs are the same in
both. Format 3 adds `FileMetadata` lines to the `chunk` output and leaves the chunk
fields as in format 2. A reader of either treats a missing `scope` as empty; a chunk without a scope
serializes identically in both. Watcher states and index documents written before
format 2 load (their chunks have no scope); a watcher reports the chunks that gain one
as `changed`, once. The README's [Context headers](../arko_index/README.md#context-headers)
lists the labels per language.

**`Delta` fields**: `sequence` (monotone), `added` (chunks), `removed` (IDs), `changed`
(chunks whose ID stayed but whose bytes, range or lines changed). Applying a delta as
sets to the previous snapshot gives the next snapshot.

## File selection

`FileRules` (`#[non_exhaustive]`; set fields on `FileRules::default()`):
`respect_ignore` (default `false`), `exclude` and `include` (gitignore-syntax patterns,
default empty). `FileRules::validate` rejects an invalid pattern before any work.

`Report` (`#[non_exhaustive]`): `rules`, `files` (indexed count), `counts` (per
`SkipReason`), `skipped` (`Skipped { path, reason, detail }`), `ignored` (at most
`IGNORED_LIST_CAP` = 1,000 entries), `excluded_by_pattern`, `not_included`,
`ignored_by_gitignore`, `ignored_by_ignore_file`, `ignored_by_git_info_exclude`,
`warnings`. `summary()`, `notable(limit)`, `worth_printing()` and `ignored_total()`
produce what the CLI prints.

`SkipReason`: `binary`, `not-utf8`, `name-not-utf8`, `unreadable`, `symlink`,
`special`, `git-dir`, `git-file`, `cachedir-tag`, `state-dir`, and with
`respect_ignore` `gitignore`, `ignore-file`, `git-info-exclude`.
`is_default_exclusion()` is true for `git-dir`, `git-file`, `cachedir-tag` and
`state-dir`. The README's "What gets indexed" section has the rules and their
precedence.

## Search

| Item | What it does |
|---|---|
| `search::index[_with](root, state_dir, options, [rules])` | Chunks the tree and writes `index.json`; returns the chunk count (and the `Report`). |
| `search::search(state_dir, query)` | Loads `index.json`; returns up to ten chunks, best first. |
| `search::outline(chunks)` | Groups chunks by path for a human-readable outline. |
| `index2::index[_with](root, state_dir, options, threads, mem_budget, [rules])` | Builds a new generation with postings memory bounded by `mem_budget` bytes, lexical only; returns `BuildStats { docs, terms, runs, report, dense }`. |
| `index2::build(root, state_dir, options, &BuildOptions)` | The same with `BuildOptions { threads, mem_budget, rules, lexical_scope, dense }` (`#[non_exhaustive]`; `Default`, `with_*`). `lexical_scope` (default `false`) also indexes the words of each chunk's scope for lexical search. With `dense: Some(DenseBuild)` the chunks are embedded too (below). |
| `index2::search(state_dir, query)` | Reads only the query terms' postings of the current generation; same results as `search::search` on the same tree (unless the index was built with `lexical_scope`). |
| `index2::dense_info(state_dir)` | The current generation's `DenseIndexInfo` (model, render version, embedded text, quantization, chunks, chunks with vectors), or `None`. |

Scoring: BM25 with k1 = 1.2 and b = 0.75 over lowercase words plus identifier parts
(snake_case and camelCase components); ties by path, then start byte.

## Dense and hybrid search

| Item | What it does |
|---|---|
| `dense::Embedder` | The provider interface (`Send + Sync`): `descriptor()`, `limits()`, `count_tokens(texts)`, `embed(texts, InputKind)` returning `Embedding { vectors, tokens }` (tokens billed; 0 means "use the counted tokens"). |
| `dense::ModelDescriptor` | `provider`, `model`, `dimensions`, `input_types`, `revision` (optional); `new`, `with_input_types`, `with_revision`, `key()` (16 hex digits naming the cache directory), `display()`. Part of the embedding cache key. |
| `dense::EmbedLimits::new(max_items, max_request_tokens, max_input_tokens)` | Batching limits. A text over a limit is not embedded (`EmbedStats::too_long`). |
| `dense::HashEmbedder::new(dimensions)` | Offline, deterministic feature hashing of the lexical terms (provider `hash`, model `hash-v1`); for tests and examples, not semantic. |
| `dense::render(chunk, chunk_text)`, `RENDER_VERSION` | The embedded text, version 1: `path`, `\n`, the scope and `\n` when there is one, the source trimmed. |
| `dense::render_with(chunk, chunk_text, EmbedText)`, `EmbedText` | The embedded text with the parts `EmbedText` names: `Source` (the trimmed source alone), `PathSource` (path line and source) or `PathScopeSource` (the default, `render`'s text). `as_str` and `FromStr` use `source`, `path-source`, `path-scope-source`. |
| `dense::embed_query(embedder, query)` | The query's vector and billed tokens, for `SearchRequest::query_vector`. |
| `dense::DenseConfig` | The per-index settings in `STATE_DIR/dense.json`: `version` (1), `embedder` (a `ModelDescriptor`), `render_version`, `quantization`, `max_tokens_per_run`, `embed_text` (`PathScopeSource` when absent); `new`, `with_*`, `load(state_dir)`, `read(path)`, `save(state_dir)`, `remove(state_dir)`. Saved by every build with embeddings. |
| `dense::DenseBuild` | One build's embedding: `embedder`, `quantization` (`F32`), `max_tokens` (`None`), `embed_text` (`PathScopeSource`), `progress` (called after each request with `EmbedProgress { texts, total, tokens, requests }`); `new`, `from_config`, `with_*`. |
| `dense::EmbedStats` | `BuildStats::dense`: `embedder`, `render_version`, `embed_text`, `docs`, `embedded_docs`, `texts` (distinct), `cached`, `new`, `requests`, `tokens` (billed), `max_tokens`, `cap_reached`, `not_embedded`, `not_embedded_tokens` (counted), `too_long`, `cache_records`. |
| `vector::VectorStore` | `dimensions`, `len`, `is_empty`, `insert(id, vector, VectorMeta { path, language })` (replaces), `delete(id)`, `search(query, k, &Filter) -> Vec<VectorHit { id, score }>` (cosine; best first, then id ascending), `persist`, `open(dir)`. |
| `vector::BruteForceStore::create(dir, dimensions, Quantization)` | The exact store; `Quantization::Int8` adds an 8-bit copy scanned first, with the best `max(4k, 40)` rescored exactly. |
| `vector::Filter` | `path_prefix` (a string prefix of the relative path), `language` (the chunk's `language`); `with_path_prefix`, `with_language`, `matches`, `is_empty`. |
| `hybrid::search(state_dir, &SearchRequest, Option<&dyn Embedder>)` | One generation, read consistently; returns `SearchResults { hits, mode, query_tokens, dense }`. A search that loses a race with a build's switch starts over on the new generation, embedding the query once. |
| `hybrid::SearchRequest::new(query)` | `mode` (`Lexical`), `limit` (10), `candidates` (50, hybrid), `fusion` (RRF), `filter` (none), `query_vector` (none); `with_*`. |
| `hybrid::SearchHit` | The chunk (flattened in JSON), `header`, `score` (BM25, cosine or fused), `lexical_rank`, `dense_rank` (1-based). |
| `hybrid::Fusion` | `Rrf { k: 60, lexical_weight: 1, dense_weight: 1 }` (default): `Σ weight / (k + rank)`; `Linear { lexical_weight }`: `w · minmax(lexical) + (1 − w) · minmax(dense)`. Ties by document id. |
| `hybrid::fuse(lexical, dense, &Fusion, limit)` | The fusion on its own, over `(id, score)` lists. |

Dense and hybrid modes need the generation to have embeddings and an embedder whose
descriptor equals the generation's (or a `query_vector`); filters apply to both
sides. See the README's [Dense and hybrid search](../arko_index/README.md#dense-and-hybrid-search)
for the cache, the cap and the accounting.

## Watchers

| Item | What it does |
|---|---|
| `open_watch[_with](root, state_dir, options, [rules])` | Opens (or creates) a polling watcher's state and reconciles with the tree. Any platform. |
| `Watcher::next_change(timeout)` | The next pending delta, or `None` when the tree is quiet. |
| `Watcher::ack(sequence)` | Durable, cumulative acknowledgement; a future sequence is rejected. |
| `Watcher::snapshot()`, `flush()`, `report()`, `rules_changed()` | The current chunks, persisting the state, the current `Report`, and whether the open applied changed rules. |
| `watch2::Watcher2::open[_with](root, state_dir, options, [rules])` | Linux: the inotify watcher. `next_change`, `next_change_to(timeout, writer)` (streams the delta's JSON line), `ack`, `report`, `rules_changed`, `file_count`, and `stats` (timings). |

## Command line

`arko_index_cli` is an example of the crate:
`cargo run --release -p arko_index --example arko_index_cli -- COMMAND ...`.

| Command | Output |
|---|---|
| `chunk ROOT` (directory or file) | Chunk JSON lines, and a `FileMetadata` line before a Markdown file's chunks when it has front matter. |
| `outline ROOT` | `path:start-end<TAB>kind<TAB>symbol_path` lines; a metadata region is `path:start-end<TAB>metadata<TAB>title`. |
| `index ROOT STATE_DIR` | stderr `indexed N chunks` and the report summary. |
| `search STATE_DIR QUERY...` | Up to ten chunk JSON lines (the words are joined with spaces). |
| `index2 ROOT STATE_DIR` | stderr `indexed N chunks, T terms, R runs` and the report summary; with embeddings also a progress line per request and the token summary. Flags: `--lexical-scope` (`BuildOptions::lexical_scope`, this build only), `--embed MODEL` (`voyage-code-4`, `voyage:MODEL`, `hash`), `--dimensions N`, `--embed-text source\|path-source\|path-scope-source`, `--embed-config FILE`, `--quantize f32\|int8`, `--max-embed-tokens N\|none`, `--no-embed`; without them, the saved `dense.json` applies. |
| `search2 STATE_DIR QUERY...` | Up to `--limit` (10) JSON lines: the chunk's fields plus `header`, `score`, `lexical_rank`, `dense_rank`; `--text` prints `rank. header  [mode score; ranks]`. `--mode lexical\|dense\|hybrid` (default hybrid when the index has embeddings, else lexical), `--candidates N`, `--fusion rrf\|linear`, `--rrf-k K`, `--lexical-weight W`, `--dense-weight W`, `--path-prefix P`, `--language L`. Dense and hybrid print the query's tokens on stderr. |
| `watch ROOT STATE_DIR` | Delta JSON lines, each acknowledged after it is flushed; stops on SIGINT or SIGTERM. |
| `watch2 ROOT STATE_DIR` | Linux. As `watch`; stderr also gets a JSON line with file count and timings at start and per delta. |
| `bench-scan ROOT` | Unix. One JSON line of scan timings. |
| `--help`, `-h` | Usage on stderr. |

File selection flags (commands that walk a tree): `--respect-ignore`,
`--exclude PATTERN`, `--include PATTERN` (repeatable), `--report FILE`; `--opt=value`
works, `--` ends the flags. Those commands also take `--front-matter-description`
(`Options::front_matter_description`). Errors print `arko_index: <message>` and exit with status 1.
stderr lines from the report start with `arko_index: `.

`dense_live_check` is a second example: it builds an index with Voyage embeddings of
a small corpus and runs the retrieval evaluation in the three modes, printing ranks,
scores and tokens (`--dry-run` only counts tokens). It spends Voyage tokens; see the
file's header and the README.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ARKO_INDEX_THREADS` | The CPUs the process may use (`available_parallelism`, which honours affinity masks and cgroup quotas) | Worker threads for chunking, `index2` and the watchers' parallel passes (`threads()`). |
| `ARKO_INDEX_WINDOW_MB` | 32 MB of source per thread | Reorder window of the ordered parallel map: how far workers may run ahead of the output. |
| `ARKO_INDEX_INDEX_MEM_MB` | 256 | CLI only: `index2`'s postings memory budget before spilling to runs. |
| `ARKO_INDEX_STATS` | Unset | When set, `index2` prints per-phase timings to stderr. |
| `MALLOC_MMAP_THRESHOLD_` | Unset: the CLI sets 1 MB | glibc's own variable; when set, the CLI leaves the threshold to glibc. |
| `VOYAGE_API_KEY` | Unset | Examples only (through arko_ai's `EmbeddingClient::from_env`): the key for embedding with Voyage. The library reads no credentials. |

## Features

arko_index has no cargo features. It parses with mhr_tree_sitter, a pure-Rust port of
tree-sitter 0.25.10 and 54 grammars, so the library needs no C compiler. The port pins
the grammar versions: an upgrade changes chunk boundaries and IDs. See
[Parsing](../arko_index/README.md#parsing). The examples and tests depend on arko_ai
(dev-dependency, for the Voyage adapter), whose dependencies need a C compiler.

## Defaults

| Setting | Default |
|---|---|
| `Options::max_bytes` | 4,096 source bytes; the rendered cap is 24,000 bytes |
| `Options::front_matter_description` | `false`: a Markdown scope's root is the front matter's title alone (or the first `# H1`) |
| `BuildOptions::lexical_scope` (CLI `index2 --lexical-scope`) | `false`: lexical search indexes each chunk's path and source, not its scope |
| `FileRules::respect_ignore` | `false`: ignore files are not read; gitignored files are indexed |
| `FileRules::exclude`, `include` | Empty |
| Built-in exclusions | `.git` directories and files, `CACHEDIR.TAG` directories (not the root), state directories |
| Non-UTF-8, binary, unreadable files | Skipped and reported; the run continues |
| Symbolic links | Not followed; reported |
| Search results | At most ten (`SearchRequest::limit` 10) |
| Polling watcher debounce | 75 ms of stable inventory |
| Threads | All available CPUs |
| Embeddings (`BuildOptions::dense`) | `None`: indexes are lexical; dense is opt-in per index |
| Embedding model (CLI `--embed`) | None until chosen; then the saved `dense.json`. The CLI's Voyage models default to the model's dimensions (voyage-code-4: 1,024) with input types on |
| Token cap (`DenseBuild::max_tokens`, `DenseConfig::max_tokens_per_run`) | `None`: no cap |
| Embedded text (`DenseBuild::embed_text`, `DenseConfig::embed_text`, CLI `--embed-text`) | `PathScopeSource`: the path, the scope, then the source |
| Vector storage (`Quantization`) | `F32`; `Int8` rescores `max(4k, 40)` candidates |
| Search mode (`SearchRequest::mode`) | `Lexical`; the CLI's `search2` uses hybrid when the index has embeddings |
| Hybrid candidates (`SearchRequest::candidates`) | 50 per side |
| Fusion (`Fusion::default`) | Reciprocal rank fusion, k = 60, both weights 1 (`Linear`: lexical weight 0.5 in the CLI) |
| Filters (`Filter::default`) | None |
| Render version (`RENDER_VERSION`) | 1: path, scope, trimmed source |
| Embedding cache compaction | When records exceed twice the live texts plus 1,024 |
| Voyage adapter (examples) | 1,000 texts and 320,000 tokens per request for voyage-code-4; 4 retries of transient failures |

## On-disk state

A state directory holds the `.arko-index-state` marker (the walk excludes any
directory that holds it) and `report.json` (the last walk's report).

| Writer | Files |
|---|---|
| `search::index` | `index.json` (version 1), `.index.lock` |
| `index2::index`, `index2::build` | `CURRENT2` (the name of the current generation), `index2-<stamp>-<pid>/` with `meta.json` (version 2), `lens.dat`, `docs.off`, `docs.dat`, `terms.dat`, `terms.idx`, `post.dat`, `files.dat` (each file's first document and path, for filters); with embeddings also `dense.json` (`DenseIndexInfo`, version 1) and `dense/` (`slots.dat` with ids, paths, languages, int8 scales and a SHA-256 checksum; `vectors.f32`; `vectors.i8` with int8). Older generations are removed after a switch. |
| `index2::build` with embeddings | `dense.json` (`DenseConfig`, version 1: the model and settings later builds follow) and `embed-cache/<model key>/` (`cache.json` with the full descriptor and the epoch, `keys.<epoch>` with 32-byte text hashes, `vectors.<epoch>` with the vectors as returned, appended and fsynced per request). |
| `open_watch` | `watch.json` (version 1: snapshot, fingerprints, sequence, acknowledgement, pending deltas, rules, and the engine-qualified parse cache) |
| `Watcher2` | `watch2.snap` (version 3), `watch2.recs.N`, `watch2.wal` |

Writers take an advisory lock on the state (the marker, or `.index.lock`), so a second
writer fails fast. Commits write a temporary file, fsync it, rename it and fsync the
directory (on Unix).

**Compatibility.** arko_index is unpublished, and its on-disk data has no compatibility
guarantee between versions: an index or watcher state from an earlier build may be
refused (`watch2` refuses a version 2 snapshot and asks for a rebuild) or rebuilt. A
watcher's parse cache is reused only when the engine digest (the chunking sources and
the locked tree-sitter versions) matches; otherwise the files are parsed again,
without spurious deltas, except once for chunk format 2: chunks that gain a `scope`
are delivered as `changed`. The engine digest also changes with
`Options::front_matter_description`. An index2 generation from before this version loads and
searches lexically; filters need its `files.dat` (rebuild to filter). Saved files
with embeddings carry versions (`dense.json` 1, the generation's `dense.json` 1,
`slots.dat` 1, `cache.json` 1) and new fields get serde defaults. The SQLite
databases of the earlier, model-chunked arko_index are not readable by this crate.

## Platforms

Linux runs everything. macOS has everything but `watch2`. Windows compiles: the
directory fsync is skipped, and `watch2` and `bench-scan` are unavailable.
