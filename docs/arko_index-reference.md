# arko_index reference

arko_index partitions source trees into syntax-aware chunks with tree-sitter, builds
offline BM25 indexes over them, and keeps chunk snapshots current with durable
watchers. It makes no network calls and depends on no other arko crate. The
[crate README](../arko_index/README.md) explains the behaviour (which files are indexed,
how boundaries and IDs are chosen, the watchers' durability); this page lists the API,
the CLI, the environment variables, the defaults and the on-disk state.

## Modules

| Module | What it holds | Code |
|---|---|---|
| `arko_index::bench_v1` | Chunking and the polling watcher: `Options`, `SourceSpan`, `Chunk`, `Delta`, `Error`, `Watcher`, `chunk_str`, `chunk_file`, `chunk_dir`, `chunk_dir_each`, `chunk_dir_map`, `open_watch`, the `_with` variants; re-exports `FileRules`, `Report`, `SkipReason`, `Skipped`. | `arko_index/src/bench_v1.rs`, `arko_index/src/watch.rs` |
| `arko_index::search` | The first index format (one JSON file): `index`, `index_with`, `search`, `outline`. | `arko_index/src/search.rs` |
| `arko_index::index2` | The disk-resident index: `index`, `index_with`, `search`, `BuildStats`. | `arko_index/src/index2.rs` |
| `arko_index::watch2` | Linux only: `Watcher2`, the inotify watcher. | `arko_index/src/watch2.rs` |
| crate root | `FileRules`, `Report`, `SkipReason`, `Skipped`, `threads()`, and on Unix `bench_scan`, `bench_scan_with`. | `arko_index/src/lib.rs`, `arko_index/src/files.rs`, `arko_index/src/par.rs`, `arko_index/src/scan.rs` |

Errors are `bench_v1::Error`, a boxed `std::error::Error + Send + Sync`.

## Chunking

| Item | What it does |
|---|---|
| `Options { max_bytes }` | The source-byte bound of a chunk. Default 4,096; values below 4 are rejected. A rendered chunk (path, newline, source) is also capped at 24,000 bytes. |
| `chunk_str(path, text, options)` | Chunks one text under the given path (kept verbatim; the extension selects the grammar). |
| `chunk_file(path, options)` | Reads and chunks one file under its file name. A binary or non-UTF-8 file gives no chunks. |
| `chunk_dir(root, options)`, `chunk_dir_with(root, options, rules)` | Every file the rules select, in path order; `_with` also returns the `Report`. |
| `chunk_dir_each[_with](root, options, [rules,] sink)` | Streams each file's chunks to `sink` in path order, chunking on `threads()` workers. |
| `chunk_dir_map[_with](root, options, [rules,] threads, map, sink)` | Like `chunk_dir_each`, applying `map` on the worker (the CLI renders JSON there). Memory is bounded by a reorder window of source bytes. |

Languages by extension: `rs` Rust; `py` Python; `js`, `jsx` JavaScript; `ts`, `tsx`
TypeScript (TSX grammar for `tsx`); `go` Go; `java` Java; `c`, `h` C (`.h` parsed with
the C++ grammar); `cc`, `cpp`, `hpp` C++. Anything else is `text`, split by the bounded
fallback.

**`Chunk` fields** (one JSON object per line in CLI output, in this order): `id`
(SHA-256 based, from path plus trimmed content and an occurrence ordinal), `path`
(relative, slash-separated), `start_byte`, `end_byte` (half-open UTF-8 byte range),
`start_line`, `end_line` (1-based, inclusive), `language`, `symbol_path` (enclosing
symbols), `kind` (the syntax kind, `function_family` for a merged pair, or a fallback
kind), `token_estimate` (a conservative byte upper bound of the rendered input).

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
| `index2::index[_with](root, state_dir, options, threads, mem_budget, [rules])` | Builds a new generation with postings memory bounded by `mem_budget` bytes; returns `BuildStats { docs, terms, runs, report }`. |
| `index2::search(state_dir, query)` | Reads only the query terms' postings of the current generation; same results as `search::search` on the same tree. |

Scoring: BM25 with k1 = 1.2 and b = 0.75 over lowercase words plus identifier parts
(snake_case and camelCase components); ties by path, then start byte.

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
| `chunk ROOT` (directory or file) | Chunk JSON lines. |
| `outline ROOT` | `path:start-end<TAB>kind<TAB>symbol_path` lines. |
| `index ROOT STATE_DIR` | stderr `indexed N chunks` and the report summary. |
| `search STATE_DIR QUERY...` | Up to ten chunk JSON lines (the words are joined with spaces). |
| `index2 ROOT STATE_DIR` | stderr `indexed N chunks, T terms, R runs` and the report summary. |
| `search2 STATE_DIR QUERY...` | As `search`. |
| `watch ROOT STATE_DIR` | Delta JSON lines, each acknowledged after it is flushed; stops on SIGINT or SIGTERM. |
| `watch2 ROOT STATE_DIR` | Linux. As `watch`; stderr also gets a JSON line with file count and timings at start and per delta. |
| `bench-scan ROOT` | Unix. One JSON line of scan timings. |
| `--help`, `-h` | Usage on stderr. |

File selection flags (commands that walk a tree): `--respect-ignore`,
`--exclude PATTERN`, `--include PATTERN` (repeatable), `--report FILE`; `--opt=value`
works, `--` ends the flags. Errors print `arko_index: <message>` and exit with status 1.
stderr lines from the report start with `arko_index: `.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ARKO_INDEX_THREADS` | The CPUs the process may use (`available_parallelism`, which honours affinity masks and cgroup quotas) | Worker threads for chunking, `index2` and the watchers' parallel passes (`threads()`). |
| `ARKO_INDEX_WINDOW_MB` | 32 MB of source per thread | Reorder window of the ordered parallel map: how far workers may run ahead of the output. |
| `ARKO_INDEX_INDEX_MEM_MB` | 256 | CLI only: `index2`'s postings memory budget before spilling to runs. |
| `ARKO_INDEX_STATS` | Unset | When set, `index2` prints per-phase timings to stderr. |
| `MALLOC_MMAP_THRESHOLD_` | Unset: the CLI sets 1 MB | glibc's own variable; when set, the CLI leaves the threshold to glibc. |

## Features

arko_index has no cargo features. It parses with mhr_tree_sitter, a pure-Rust port of
tree-sitter 0.25.10 and its grammars, so a build needs no C compiler. The port pins the
grammar versions: an upgrade changes chunk boundaries and IDs. See
[Parsing](../arko_index/README.md#parsing).

## Defaults

| Setting | Default |
|---|---|
| `Options::max_bytes` | 4,096 source bytes; the rendered cap is 24,000 bytes |
| `FileRules::respect_ignore` | `false`: ignore files are not read; gitignored files are indexed |
| `FileRules::exclude`, `include` | Empty |
| Built-in exclusions | `.git` directories and files, `CACHEDIR.TAG` directories (not the root), state directories |
| Non-UTF-8, binary, unreadable files | Skipped and reported; the run continues |
| Symbolic links | Not followed; reported |
| Search results | At most ten |
| Polling watcher debounce | 75 ms of stable inventory |
| Threads | All available CPUs |

## On-disk state

A state directory holds the `.arko-index-state` marker (the walk excludes any
directory that holds it) and `report.json` (the last walk's report).

| Writer | Files |
|---|---|
| `search::index` | `index.json` (version 1), `.index.lock` |
| `index2::index` | `CURRENT2` (the name of the current generation), `index2-<stamp>-<pid>/` with `meta.json` (version 2), `lens.dat`, `docs.off`, `docs.dat`, `terms.dat`, `terms.idx`, `post.dat`. Older generations are removed after a switch. |
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
without spurious deltas. The SQLite databases of the earlier, model-chunked arko_index
are not readable by this crate.

## Platforms

Linux runs everything. macOS has everything but `watch2`. Windows compiles: the
directory fsync is skipped, and `watch2` and `bench-scan` are unavailable.
