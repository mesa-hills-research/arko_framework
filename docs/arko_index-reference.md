# arko_index reference

arko_index partitions source trees into syntax-aware chunks with tree-sitter, builds
offline BM25 indexes over them, optionally embeds the chunks for dense and hybrid
search, and keeps chunk snapshots current with durable watchers. The library makes no
network calls and depends on no other arko crate: embeddings come from a caller's
`dense::Embedder` (its examples adapt arko_ai's Voyage client, a dev-dependency). The
[crate README](../arko_index/README.md) is the overview. This page describes the
behaviour (which files are indexed, how boundaries and IDs are chosen, context headers,
dense and hybrid search, the watchers' durability) and lists the API, the CLI, the
environment variables, the defaults and the on-disk state.

## Modules

| Module | What it holds | Code |
|---|---|---|
| `arko_index::bench_v1` | Chunking and the polling watcher: `Options`, `SourceSpan`, `Chunk`, `FileMetadata`, `FileChunks`, `CHUNK_FORMAT`, `Delta`, `Error`, `Watcher`, `chunk_str`, `chunk_str_with_metadata`, `chunk_file`, `chunk_file_with_metadata`, `chunk_dir`, `chunk_dir_each`, `chunk_dir_map`, `open_watch`, the `_with` variants. It re-exports `FileRules`, `Report`, `SkipReason`, `Skipped`. | `arko_index/src/bench_v1.rs`, `arko_index/src/watch.rs`, `arko_index/src/syntax.rs`, `arko_index/src/languages.rs`, `arko_index/src/rules/`, `arko_index/src/headers.rs`, `arko_index/src/front_matter.rs` |
| `arko_index::search` | The first index format (one JSON file): `index`, `index_with`, `search`, `outline`. | `arko_index/src/search.rs` |
| `arko_index::index2` | The disk-resident index: `index`, `index_with`, `build`, `BuildOptions`, `search`, `dense_info`, `BuildStats`, and its embedding: `embed`, `status`, `IndexStatus`. | `arko_index/src/index2.rs` |
| `arko_index::dense` | Embeddings: `Embedder`, `ModelDescriptor`, `EmbedLimits`, `Embedding`, `InputKind`, `HashEmbedder`, `embed_query`, `render`, `render_with`, `EmbedText`, `RENDER_VERSION`, `DenseConfig`, `DenseBuild`, `EmbedProgress`, `EmbedStats`, `EmbedState`, `DenseIndexInfo`. | `arko_index/src/dense/` |
| `arko_index::vector` | Vector storage: `VectorStore`, `CaprockStore`, `StoreConfig`, `Quantization`, `VectorIndex`, `Filter`, `VectorId`, `VectorMeta`, `VectorRecord`, `VectorHit`. | `arko_index/src/vector.rs` |
| `arko_index::hybrid` | Lexical, dense and hybrid search: `search`, `SearchRequest`, `Mode`, `Fusion`, `fuse`, `Fused`, `SearchHit`, `SearchResults`. | `arko_index/src/hybrid.rs` |
| `arko_index::watch2` | Linux only: `Watcher2`, the inotify watcher. | `arko_index/src/watch2.rs` |
| crate root | `FileRules`, `Report`, `SkipReason`, `Skipped`, `threads()`, and on Unix `bench_scan`, `bench_scan_with`. | `arko_index/src/lib.rs`, `arko_index/src/files.rs`, `arko_index/src/par.rs`, `arko_index/src/scan.rs` |

Errors are `bench_v1::Error`, a boxed `std::error::Error + Send + Sync`.

## Chunking

| Item | What it does |
|---|---|
| `Options { max_bytes, front_matter_description }` | `max_bytes`: the source-byte bound of a chunk. Default 4,096, and values below 4 are rejected. A rendered chunk (path, newline, source) is also capped at 24,000 bytes. `front_matter_description` (default `false`): Markdown scopes start with the front matter's title and description instead of the title alone. |
| `chunk_str(path, text, options)` | Chunks one text under the given path (kept verbatim, and the extension or file name selects the grammar). |
| `chunk_str_with_metadata(path, text, options)` | The same, as `FileChunks { metadata, chunks }`: a Markdown file's metadata region (`FileMetadata`) with its chunks. |
| `chunk_file(path, options)`, `chunk_file_with_metadata(path, options)` | Reads and chunks one file under its file name. A binary or non-UTF-8 file gives no chunks. |
| `chunk_dir(root, options)`, `chunk_dir_with(root, options, rules)` | Every file the rules select, in path order. `_with` also returns the `Report`. |
| `chunk_dir_each[_with](root, options, [rules,] threads, sink)` | Streams each file's chunks to `sink` in path order, chunking on `threads` workers. A panic in `sink` returns the error `panicked: MESSAGE`. |
| `chunk_dir_map[_with](root, options, [rules,] threads, map, sink)` | Like `chunk_dir_each`, applying `map` to each file's `FileChunks` on the worker (the CLI renders JSON there). Memory is bounded by a reorder window of source bytes. A panic in `map` or `sink` returns the error `panicked: MESSAGE`. |

```rust
use arko_index::bench_v1::{chunk_str, Options};
use std::path::Path;
let chunks = chunk_str(Path::new("sample.rs"), "fn hello() {}\n", &Options::default())?;
```

Languages: 50, from extensions and well-known file names (`Makefile`,
`CMakeLists.txt`, `PKGBUILD`, ...), listed with each language's rules in
[arko_index languages](arko_index-languages.md). Anything else is `text`, split by the
bounded fallback.

**Paths and coverage.** For directory input, paths are slash-separated and relative.
`chunk_file` uses the file name, and `chunk_str` keeps the supplied UTF-8 path verbatim.
Literal Unix backslashes are filename characters, not separators. A directory walk skips
and reports undecodable names instead of letting them collide with another path, and
`chunk_file` and `chunk_str` return an error for one. Empty files produce no chunks.
Every included nonempty text byte, including whitespace, is in exactly one chunk, or in a
Markdown file's metadata region (its front matter), which `chunk_str_with_metadata` and
`chunk_file_with_metadata` return with the chunks.

A result is metadata, not a copy of the source: read the half-open UTF-8 byte range
`start_byte..end_byte` of `ROOT/path` to see the exact text. Lines are inclusive and
1-based, and a final newline belongs to its line.

**`Chunk` fields** (one JSON object per line in CLI output, in this order): `id`
(SHA-256 based, from path plus trimmed content and an occurrence ordinal), `path`
(relative, slash-separated), `start_byte`, `end_byte` (half-open UTF-8 byte range),
`start_line`, `end_line` (1-based, inclusive), `language`, `symbol_path` (enclosing
symbols), `kind` (the syntax kind, `function_family` for a merged pair, or a fallback
kind), `token_estimate` (a conservative byte upper bound of the path, a newline and
the source), `scope` (the context: enclosing definitions such as
`impl Display for Point › fn fmt`, or a Markdown heading chain, written only when not
empty). `Chunk::header()` gives `path:start_line-end_line · scope`.

**`FileMetadata` fields** (a JSON line before the file's chunks in CLI output, format
3): `file` (the path), `metadata_range` (a `SourceSpan` from byte 0: `start_byte`,
`end_byte`, `start_line`, `end_line`), `front_matter` (the raw block, delimiters
included, absent when empty), `title` and `description` (absent when the front matter
has none). Only Markdown files with front matter or MDX import/export lines at the top
have one. Every byte of a file is in exactly one chunk or in this region.

**`Delta` fields**: `sequence` (monotone), `added` (chunks), `removed` (IDs), `changed`
(chunks whose ID stayed but whose bytes or any other field changed: range, lines, scope,
symbol path). Applying a delta as sets to the previous snapshot gives the next snapshot.

### Boundaries and identities

* Default `max_bytes=4096` is a strict source bound. Values below 4 are rejected
  because one UTF-8 scalar can occupy four bytes. Every range endpoint is on a UTF-8
  boundary.
* A separate 24,000 **rendered-byte** safety cap includes the entire path and
  newline, even for arbitrarily large `max_bytes`. Pathological paths with no room
  for a scalar are rejected, not silently truncated. `token_estimate` is a
  conservative byte upper bound including the path, not an exact tokenizer count.
* Function, method and type boundaries are preferred. Comments and attributes attach
  forward. Namespace, class and impl bodies and nested object-call wrappers expose
  member boundaries. Oversized nodes split recursively, and adjacent non-definition
  siblings are packed within the same syntax scope.
* Exactly two adjacent short callable definitions may form a `function_family` if
  they are true AST siblings and share an overload name or a nontrivial first word in
  compound identifiers (`render_plain` / `render_debug`). The pair is at most 1,200
  bytes and each constituent supplies at least a third of its nonwhitespace bytes.
  Identical duplicates stay separate, and long same-family runs are not greedily
  paired. Family chunks are not merged again as tiny definitions.
* Sustained runs (eight or more) of tiny same-kind definitions can be combined into
  coherent groups up to 768 bytes. Content-derived cut points resynchronize after an
  insertion, unlike fixed-count windows. Ordinary short files and substantial
  unrelated functions stay independent.
* The C++ grammar parses `.h` files, whose `language` field stays `c`. If C/C++
  parsing has errors, generic annotation macros are replaced with spaces in a
  temporary parse buffer, and that tree is kept only if ERROR coverage decreases.
  Returned ranges always refer to the unmodified source.
* A deterministic work budget stops a parse after 100 × min(1,000 + one per 32 bytes of
  the file, 50,000) parser operations, at most 5 million, and the file is then split by
  the bounded fallback. The budget counts operations, not time, so the result is the same
  on every machine. Some inputs still take long: deeply nested unclosed XML or Svelte
  tags can take minutes on one file. A file whose grammar panics, or whose scanner the
  runtime stops because it keeps advancing at the end of the input, is split by the
  bounded fallback too, and a walk lists it in the report's `warnings` (see [File
  selection](#file-selection)). A file with any physical line over 65,536 bytes is split
  entirely by the bounded fallback. Single-child AST wrapper descent is iterative and
  depth-bounded. Missing parses, unsupported formats, deep recovery trees and oversized
  atomic literals also use the fallback, which prefers complete lines or whitespace, then
  a UTF-8 boundary. An atom or string larger than the bound is split inside: exact
  coverage and the size bound take precedence over keeping such a token whole.
* IDs hash the path plus the trimmed chunk content, with a per-content occurrence
  ordinal. Absolute offsets are excluded, so unrelated edits do not change the IDs of
  unchanged units. Identical duplicates get distinct IDs. Renames change IDs. Edits
  to a merged unit can change that unit's membership and identity, and nearby
  content-defined groups resynchronize. Source offsets are always current.

### Context headers

A chunk's `scope` is shown with a search result as its header,
`path:start_line-end_line · scope`. For code it is the chain of definitions that enclose
the chunk, from the syntax tree, the chunk's own definition included. A chunk merged from
several definitions (a function family, a run of tiny definitions) shows the enclosing
scope and the members joined with ` | `, and text packed between definitions shows the
enclosing scope. In Markdown (`.md`, `.markdown`, `.mdx`) it is the document's title,
then the heading chain in effect where the chunk starts. Other text has no scope. The
labels of each language, the original nine included, and the Markdown rules are in
[arko_index languages](arko_index-languages.md).

Labels longer than 200 bytes end in `…`, and a scope over 1,000 bytes keeps its innermost
elements after `… › `. `symbol_path` is unchanged: names only (`Point::fmt`), without
modules or keywords. Headers are metadata: they never change chunk boundaries or IDs.

**Chunk format versions** (`bench_v1::CHUNK_FORMAT`, now 3). Format 1 had the ten fields
up to `token_estimate`. The `scope` field arrived in format 2, written only when it is not
empty, so a chunk without one serializes exactly as before, and a reader treats a missing
`scope` as empty. Boundaries and IDs are the same in both. Format 3 adds a `FileMetadata`
line before a Markdown file's chunks in the `chunk` output, recording its metadata region
(front matter), with a `file` key where chunks have `id`. It leaves the chunk fields as in
format 2. Watcher states and index documents written before format 2 load (their chunks
have no scope). Because the field is new, a watcher state written before format 2
reconciles on its first open with a one-time `changed` delta for every chunk that has a
scope (same IDs, bytes and lines).

## File selection

The same rules apply to every command that walks a tree and to the library
(`chunk_dir`, `chunk_dir_each`, `chunk_dir_map`, `open_watch`, `search::index`,
`index2::index`, `Watcher2::open`, and their `_with` variants).

**Defaults.** Every regular file under the root is indexed, including hidden files
and files that `.gitignore` lists, except:

* `.git` directories, and `.git` files (the one-line `gitdir:` pointer of a submodule
  or linked worktree)
* directories that hold a valid `CACHEDIR.TAG`: a file whose first 43 bytes are
  `Signature: 8a477f597d28d172789f06886806bc55` ([Cache Directory Tagging
  Specification](https://bford.info/cachedir/)). Cargo writes one into `target/`, and
  other build tools and caches do the same. A directory named `target` without the
  tag (Linux's `drivers/target`) is indexed like any other. The root itself is
  indexed even when it is tagged, and an `--include` pattern that names a tagged
  directory brings it back (below).
* the tool's own state directories inside the root: the `STATE_DIR` given to
  `index`/`index2`/`watch`/`watch2`, and any directory that holds the
  `.arko-index-state` marker those commands create

No other name is excluded. Symbolic links are never followed: a link is reported, and
a target inside the root is indexed under its own path. Ignore files are read only when
you opt in, and ignore files in directories **above** the root never apply.

**Files that cannot be indexed are skipped and reported.** Each one is listed in the
report (below) with its reason, and the run continues:

| Reason | Meaning |
|---|---|
| `binary` | the file contains a NUL byte |
| `not-utf8` | no NUL byte, but not valid UTF-8 (Latin-1 text, for example) |
| `name-not-utf8` | the file or directory name is not UTF-8 (chunk paths are UTF-8), and a directory is skipped whole |
| `unreadable` | reading the file or listing the directory failed (`detail` has the error) |
| `symlink` | a symbolic link (not followed) |
| `special` | a FIFO, socket or device |
| `git-dir`, `git-file`, `cachedir-tag`, `state-dir` | the default exclusions above (a directory's contents are not visited) |

A missing or unreadable root is still an error.

**Options.**

| CLI (repeatable where it takes a pattern) | Library: `FileRules` field | Effect |
|---|---|---|
| `--respect-ignore` | `respect_ignore: true` | Honour `.gitignore` and `.ignore` files in the root and its subdirectories and `.git/info/exclude` of repositories in the tree, with Git's syntax and precedence (deeper files override shallower ones, `!` re-includes, `.ignore` overrides `.gitignore`, which overrides `.git/info/exclude`), also outside a Git checkout. |
| `--exclude PATTERN` | `exclude` | A file or directory matching a pattern is not indexed (a directory is not entered). |
| `--include PATTERN` | `include` | When given, only files matching a pattern, directly or through one of their directories, are indexed. A pattern that matches a `CACHEDIR.TAG` directory itself brings it back. |
| `--report FILE` | (returned) | Also write the full report as JSON to `FILE`. |

Patterns use gitignore syntax relative to the root: `vendor/` (any directory named
`vendor`), `/build/` (only at the root), `*.min.js`, `docs/**/*.svg`, and `!keep.md`
to re-include within the same list. An `--exclude` always wins. An `--include`
pattern that matches a tagged directory itself (`target/`, `/build/cache/`, or a
catch-all such as `*`) brings that directory back, while one that only matches files
inside it (`*.rs`) does not. `.git` and the state directories always stay excluded. An
invalid pattern is an error before any work. `--opt=value` works too, and `--` ends
the options. In Rust:

```rust
use arko_index::bench_v1::{chunk_dir_with, FileRules, Options};
let mut rules = FileRules::default(); // non_exhaustive: set fields on a default
rules.respect_ignore = true;
rules.exclude.push("vendor/".into());
let (chunks, report) = chunk_dir_with(std::path::Path::new("."), &Options::default(), &rules)?;
eprintln!("{}", report.summary());
```

*Which ignore files apply with `--respect-ignore`.* `.gitignore` and `.ignore` files in
the root and below, and `.git/info/exclude` of every repository whose `.git` directory
is at or below the root (it applies to everything under that repository's directory,
nested repositories included). They never include ignore files above the root (indexing
a subdirectory of a repository doesn't apply the repository's top-level `.gitignore`:
index the repository root, or repeat those patterns with `--exclude`), the global Git
excludes file (`core.excludesFile`, `~/.config/git/ignore`), or the `info/exclude` of a
repository whose `.git` is a file (a submodule or linked worktree), which lives in a git
directory elsewhere.

*Precedence.* An entry is attributed to the first rule that removes it: ignore files
(when honoured), symlink, special file, `.git` (directory or file), state directory,
`--exclude`, `--include` (files only), name not UTF-8, `CACHEDIR.TAG` (unless an
`--include` pattern names the directory), then, when the file is read: `unreadable`,
`binary`, `not-utf8`.

**The report.** Everything that is not indexed is listed in `skipped` with its reason,
except what your own rules remove. Ignore-file removals are counted per source
(`ignored_by_gitignore`, `ignored_by_ignore_file`, `ignored_by_git_info_exclude`, a
directory counting once) and listed in `ignored` (sorted, at most 1,000 entries, while
the counts are complete). `--exclude` and `--include` removals are counted
(`excluded_by_pattern`, `not_included`).

* `chunk` and `outline` print one summary line to stderr only when something other
  than the built-in exclusions (`.git`, `CACHEDIR.TAG` directories, state directories)
  was skipped, a warning occurred, or nothing was indexed. Then up to 20 entries per
  reason for every reason except `binary`, `symlink`, `git-dir`, `git-file` and
  `state-dir`, which are only counted, and each warning as `arko_index: warning: …`.
  The summary line ends with `; N warnings` when there are any. `--report FILE` is
  always written, complete:

  ```text
  arko_index: 90419 files indexed; 87 skipped (4 binary, 2 not-utf8, 81 symlink)
  arko_index: skipped (not-utf8): arch/m68k/hp300/hp300map.map
  arko_index: skipped (not-utf8): drivers/tty/vt/defkeymap.map
  ```
* `index` and `index2` always print the summary line and those entries after their own
  line, and write the full report to `STATE_DIR/report.json` after the index commits.
* `watch` and `watch2` print it at start, print a new summary line whenever the counts
  change, and keep `STATE_DIR/report.json` current (replaced atomically whenever the
  report changes, by `watch2` right after the batch's delta is delivered, and not
  fsynced, because every start rebuilds it).
* `bench-scan` takes the rules but writes no report.
* Library: `chunk_dir_with` and `chunk_dir_each_with` return a `Report`,
  `search::index_with` returns `(chunk count, Report)`, `index2::index_with` returns it in
  `BuildStats::report`, and `Watcher::report()` / `Watcher2::report()` give the
  current one. `Report::summary()` and `Report::notable(limit)` produce the CLI lines.

`report.json` (paths are relative and slash-separated, directories end with `/`, and
bytes that are not UTF-8 appear as `\xHH`):

```json
{
  "rules": { "respect_ignore": false, "exclude": [], "include": [] },
  "files": 90419,
  "counts": { "binary": 4, "not-utf8": 2, "symlink": 81 },
  "skipped": [
    { "path": "arch/m68k/hp300/hp300map.map", "reason": "not-utf8" },
    { "path": "secret.key", "reason": "unreadable", "detail": "Permission denied (os error 13)" }
  ],
  "excluded_by_pattern": 0,
  "not_included": 0,
  "ignored_by_gitignore": 0,
  "ignored_by_ignore_file": 0,
  "ignored_by_git_info_exclude": 0
}
```

With `--respect-ignore`, `ignored` lists entries such as
`{ "path": "target/", "reason": "gitignore" }` (reasons `gitignore`, `ignore-file`,
`git-info-exclude`). `warnings` (present only when non-empty) lists problems that
skipped nothing: an invalid line in an ignore file, or a file whose parser failed and
which was chunked as plain text:

```text
PATH: the LANG parser panicked (MESSAGE), chunked as plain text
PATH: the LANG parser stopped (a scanner never returned), chunked as plain text
```

Parser warnings come from `chunk_dir*` walks, `index2` builds (`report.json` and
`BuildStats::report`) and both watchers' `report()`. A watcher keeps one in memory
only, while the file is unchanged in that process. `chunk_str`, `chunk_file` and
`search::index` chunk such a file as text without a warning.

**The API.** `FileRules` (`#[non_exhaustive]`, set fields on `FileRules::default()`):
`respect_ignore` (default `false`), `exclude` and `include` (gitignore-syntax patterns,
default empty). `FileRules::validate` rejects an invalid pattern before any work.

`Report` (`#[non_exhaustive]`): `rules`, `files` (indexed count), `counts` (per
`SkipReason`), `skipped` (`Skipped { path, reason, detail }`), `ignored` (at most 1,000
entries), `excluded_by_pattern`, `not_included`, `ignored_by_gitignore`,
`ignored_by_ignore_file`, `ignored_by_git_info_exclude`, `warnings`. `summary()`,
`notable(limit)`, `worth_printing()` and `ignored_total()` produce what the CLI prints.

`SkipReason`: `binary`, `not-utf8`, `name-not-utf8`, `unreadable`, `symlink`,
`special`, `git-dir`, `git-file`, `cachedir-tag`, `state-dir`, and with
`respect_ignore` `gitignore`, `ignore-file`, `git-info-exclude`.
`is_default_exclusion()` is true for `git-dir`, `git-file`, `cachedir-tag` and
`state-dir`.

**Watchers and rule changes.**

* `watch` re-walks with the rules on every poll, so editing an ignore file or
  `.git/info/exclude` (when honoured) or adding or removing a `CACHEDIR.TAG` produces
  a delta like any other change.
* `watch2` runs a full reconciliation on any event for a `CACHEDIR.TAG` and, when
  ignore files are honoured, for a `.gitignore` or `.ignore`, or for `exclude` in a
  repository's `.git/info` (watched for that file only. A `.git/info` directory created
  inside an existing `.git` is picked up by the next full reconciliation). An
  excluded, tagged directory keeps a watch on itself only, to see its tag disappear.
  File events re-list the affected directories by walking from the root along their
  path, so every rule (including ignore files on the way down) applies exactly as in a
  full walk. Its `excluded_by_pattern`, `not_included`, `ignored_by_*` counts and the
  `ignored` list are refreshed by full reconciliations.
* **Restarting with different rules reconciles.** Both watchers record the rules in
  their state. When a state was written with different rules, the reconciliation at
  open applies the new rules like any change made while the watcher was down: the
  first delta removes the chunks of files the new rules exclude and adds those they
  include. `rules_changed()` reports it, and the CLI prints a note. Pass the same
  options on every start to avoid this.

## Search

| Item | What it does |
|---|---|
| `search::index[_with](root, state_dir, options, [rules])` | Chunks the tree and writes `index.json`. Returns the chunk count (and the `Report`). |
| `search::search(state_dir, query)` | Loads `index.json`. Returns up to ten chunks, best first. |
| `search::outline(chunks)` | Groups chunks by path for a human-readable outline. |
| `index2::index[_with](root, state_dir, options, threads, mem_budget, [rules])` | Builds a new generation with postings memory bounded by `mem_budget` bytes, lexical only. Returns `BuildStats { docs, terms, runs, report, dense }`. |
| `index2::build(root, state_dir, options, &BuildOptions)` | The same with `BuildOptions { threads, mem_budget, rules, lexical_scope, dense }` (`#[non_exhaustive]`, `Default`, `with_*`). `lexical_scope` (default `false`) also indexes the words of each chunk's scope for lexical search. With `dense: Some(DenseBuild)` the generation is published for lexical search first, and its chunks are then embedded (below): before `build` returns, or later by `index2::embed` with `DenseBuild::embed_later`. |
| `index2::embed(state_dir, &DenseBuild)` | An embedding run: embeds what the current generation lacks and stores the vectors request by request, with the generation's model and settings. Returns its `EmbedStats`. |
| `index2::search(state_dir, query)` | Reads only the query terms' postings of the current generation. Same results as `search::search` on the same tree (unless the index was built with `lexical_scope`). |
| `index2::dense_info(state_dir)` | The current generation's `DenseIndexInfo` (format version, model, render version, embedded text, quantization, vector index, vector store directory, chunks, chunks with vectors so far, `state`, and `store_config()`), or `None`. Like `hybrid::search`, it starts over when a build switches generations during the read, so the CLI's `search2` picks its default mode from the current generation. |
| `index2::status(state_dir)` | `IndexStatus { generation, docs, dense, embedding, progress }`: the current generation, its chunks, its `DenseIndexInfo`, whether an embedding run is working now, and the latest run's `EmbedProgress` on this generation. |

`search` and `search2` score with BM25 (k1 = 1.2, b = 0.75) over lowercase words plus
identifier components (snake_case parts and camelCase humps), so `retryPolicy` matches
`retry policy`. Ties break by path, then start byte. At most ten hits are returned
(`search2 --limit` and the library's `hybrid::search` choose another number).

`index2` writes a new generation directory and publishes it by atomically rewriting
`CURRENT2`, so readers keep the previous generation until the switch. A build that fails
before publishing removes the directory it was writing. Its build memory is bounded:
postings accumulate up to a budget (the CLI's default is 256 MB,
`ARKO_INDEX_INDEX_MEM_MB`) and then spill to sorted runs that are merged. A build with
embeddings reads the vector store to count the vectors its generation already has, and
an embedding run opens it for writing: each loads its codes (1 KB per chunk with int8
codes at 1,024 dimensions, 4 KB with f32). The watchers maintain chunk snapshots: rerun
`index2` to refresh search (with embeddings, the run embeds only the chunks whose text
changed), or give the CLI's watchers `--index` to rebuild an index as the tree changes.

## Dense and hybrid search

Dense search finds chunks by the similarity of embeddings, so a question in words finds
code that uses other words. Hybrid search fuses both. **Dense search is opt-in**: an index
is lexical unless a build is asked to embed, and embedding calls a provider that bills for
it. A build publishes its generation for lexical search before anything is embedded, and
dense coverage fills in while an embedding run works ([Embedding
runs](#embedding-runs)). The [retrieval evaluation](arko_index-verification.md#retrieval-evaluation)
compares the three modes.

**Embedders.** The library defines the interface, `dense::Embedder`:
`descriptor()` names the vectors (provider, model, dimensions, whether documents and
queries use different input types, an optional revision), `limits()` gives the request
limits, `count_tokens()` counts (or bounds) tokens, and `embed(texts, Document | Query)`
returns the vectors and the tokens billed. Local models will implement the same trait.
The crate ships `dense::HashEmbedder`, an offline, deterministic feature-hashing embedder
for tests and trying things out (it knows nothing of meaning). The examples'
`support/voyage.rs` adapts arko_ai's Voyage client (its HTTP, error classification and
tokenizer): documents are embedded with `input_type: document`, queries with `query`,
the dimensions always sent, `truncation: false`, requests packed to at most 1,000 texts
and 320,000 tokens (voyage-code-4's measured limit), and transient failures retried four
times with backoff.

**What is embedded** (render version 1, `dense::render`): the path, the scope on its
own line when there is one, then the chunk's source without leading and trailing
whitespace. Line numbers are left out, so an edit above a chunk does not change its
text. `--embed-text source` embeds the source alone and `path-source` the path and the
source (`dense::EmbedText`, saved in `dense.json`).

**The model is chosen per index and saved.** A build with embeddings writes
`STATE_DIR/dense.json` (`dense::DenseConfig`): provider, model, dimensions,
input-type use, render version, vector codes and index, the token cap and the embedded text.
The CLI's later builds follow it when no `--embed` is given (a library caller passes
`DenseBuild::from_config` with `DenseConfig::load`), so a rebuild after a watcher reported
changes uses the same model unasked. `--no-embed` (or `DenseConfig::remove`) turns
embeddings off for the index. Each generation records the model and the coverage of its
own vectors (`dense.json` in the generation), and dense search embeds the query with that
model and refuses an embedder of another.

**Embeddings are cached** by (model descriptor, SHA-256 of the rendered text) in
`STATE_DIR/embed-cache/<model key>/`, across generations. A run embeds only the
texts the cache lacks, each distinct text once, and a model change (another model, other
dimensions or input types) embeds everything with the new model in its own cache
directory, so switching back costs nothing. Vectors are appended and fsynced after
every request, so tokens paid for are kept even when a run fails, stops or is killed.
When the model's cache holds more than twice as many records as the index has distinct
texts, plus 1,024, a build compacts the model's cache after publishing, unless a run is
embedding then (a later build does it). Other models' caches stay until their
directories are deleted.

**Token accounting and the cap.** A run reports, in `dense::EmbedStats` (returned by
`index2::embed`, and in `BuildStats::dense` by a build that embeds before it returns,
printed by the CLI), the chunks, distinct texts, cache hits, texts embedded, requests and
tokens billed (Voyage's figure). A per-run cap (`DenseBuild::max_tokens`,
`--max-embed-tokens N`, saved with the index, no cap by default) stops embedding before
a request would take the run's billed tokens past it. The generation stays searchable,
the chunks without vectors are found by lexical search only, and `EmbedStats` counts
them and their tokens (`not_embedded`, `not_embedded_tokens`), what finishing would
take. A later run with a higher cap embeds only those. Every run has a cap of its own,
the runs that take over a newer generation included. `--max-embed-tokens 0` sends
nothing and reports the full count. A search reports `SearchResults::query_tokens`, the
tokens billed to embed its query (0 for lexical search, or when the caller passes a
vector it embedded with `dense::embed_query`), for callers that charge queries to a
budget.

**Vectors** are kept by [Caprock](https://github.com/mesa-hills-research/caprock),
a pure, safe Rust vector store, in `STATE_DIR/vectors-<digest>/`, one store per model and
settings (`vector::StoreConfig::dir_name`). Embedding runs update it in place. A chunk's
vector id is a digest of its path, its embedded text's hash and its rank among the file's
chunks with the same text, so an unchanged chunk keeps its vector whatever its document
id, and the run after a rebuild adds the vectors of new and edited chunks, removes those
of chunks that are gone, and leaves the rest (`EmbedStats::vectors_added`,
`vectors_removed`). Each update commits atomically, and a crash keeps the last commit. A dense search scans the int8
codes of every vector (1 byte per dimension, held in memory) and rescores each
segment's best `max(4 × k, 40)` candidates with the full-precision vectors on disk, so
scores are exact cosine similarities (measurements below). `--quantize f32`
(`Quantization::F32`) keeps exact f32 codes instead, at 4 bytes per dimension.
`--vector-index graph` (`VectorIndex::Graph`) adds a Vamana graph to each segment of
20,000 vectors or more, which answers in about a millisecond at some cost in recall and
build time, for very large indexes or high query rates. Equal similarities order by
vector id.

The store can hold vectors the current generation lacks: those of chunks an older
generation had, until the run for the current one removes them, which is the first thing
it does. A search skips them, fetching as many more as the store holds beyond the
generation's, and starts over when a build switched generations under it. Stores of
other models or settings are removed by the run for a generation of another model, and
by a build without embeddings (renamed to `vectors-removed-...` first, so a removal that
stops halfway leaves nothing a run opens). A run replaces a store it cannot use (damaged,
missing a file, or in a format this version cannot read) with a new one and fills it
from the embedding cache. A search refuses a store that holds fewer vectors than the
coverage recorded for its generation (a replacement made outside a run, or one a run
stopped filling before it recorded the new coverage) and asks for a rebuild.

**Hybrid fusion.** `hybrid::search` takes the best 50 chunks (`candidates`) from each
side and fuses the rankings with reciprocal rank fusion by default: a chunk scores
`Σ weight / (60 + rank)` over the lists that contain it, ranks from 1, both weights 1,
so a chunk both sides rank well comes first and neither score scale matters.
`Fusion::Linear` (`--fusion linear --lexical-weight W`) instead adds `W` times the
min-max normalized BM25 score and `1 − W` times the normalized similarity. Ties order
by document id. Path-prefix and language filters (`--path-prefix`, `--language`)
apply to both sides. A path prefix names a folder or a file and matches whole path
components: `src/net` finds `src/net/retry.rs` and `src/net`, but not `src/network.rs`.

| Item | What it does |
|---|---|
| `dense::Embedder` | The provider interface (`Send + Sync`): `descriptor()`, `limits()`, `count_tokens(texts)`, `embed(texts, InputKind)` returning `Embedding { vectors, tokens }` (tokens billed, where 0 means "use the counted tokens"). |
| `dense::ModelDescriptor` | `provider`, `model`, `dimensions`, `input_types`, `revision` (optional), with `new`, `with_input_types`, `with_revision`, `key()` (16 hex digits naming the cache directory), `display()`. Part of the embedding cache key. |
| `dense::EmbedLimits::new(max_items, max_request_tokens, max_input_tokens)` | Batching limits. A text over a limit is not embedded (`EmbedStats::too_long`). |
| `dense::HashEmbedder::new(dimensions)` | Offline, deterministic, non-semantic feature hashing of the lexical terms (provider `hash`, model `hash-v1`), for tests and examples. |
| `dense::render(chunk, chunk_text)`, `RENDER_VERSION` | The embedded text, version 1: `path`, `\n`, the scope and `\n` when there is one, the source trimmed. |
| `dense::render_with(chunk, chunk_text, EmbedText)`, `EmbedText` | The embedded text with the parts `EmbedText` names: `Source` (the trimmed source alone), `PathSource` (path line and source) or `PathScopeSource` (the default, `render`'s text). `as_str` and `FromStr` use `source`, `path-source`, `path-scope-source`. |
| `dense::embed_query(embedder, query)` | The query's vector and billed tokens, for `SearchRequest::query_vector`. |
| `dense::DenseConfig` | The per-index settings in `STATE_DIR/dense.json`: `version` (1), `embedder` (a `ModelDescriptor`), `render_version`, `quantization`, `vector_index` (`Flat` when absent), `max_tokens_per_run`, `embed_text` (`PathScopeSource` when absent), with `new`, `with_*`, `load(state_dir)`, `read(path)`, `save(state_dir)`, `remove(state_dir)`. Saved by every build with embeddings. |
| `dense::DenseBuild` | One build's or run's embedding: `embedder`, `quantization` (`Int8`), `vector_index` (`Flat`), `max_tokens` (`None`), `embed_text` (`PathScopeSource`), `progress` (called after each request with `EmbedProgress { texts, total, tokens, requests, embedded_docs, docs }`), `embed_later` (`false`: `build` embeds before it returns), `stop` (`None`: when the flag is set, the run stops after its current request), with `new`, `from_config`, `with_*`. `index2::embed` uses the embedder, the cap, `progress` and `stop`, and takes the other settings from the generation. |
| `dense::EmbedStats` | `BuildStats::dense` and `index2::embed`'s result: `embedder`, `render_version`, `embed_text`, `docs`, `embedded_docs`, `texts` (distinct), `cached`, `new`, `requests`, `tokens` (billed), `max_tokens`, `cap_reached`, `not_embedded`, `not_embedded_tokens` (counted), `too_long`, `cache_records`, `vectors_added`, `vectors_removed` (the vector store's update), `store_created`, `superseded` (a newer generation was published), `stopped`. A build with `embed_later` reports what it published: `embedded_docs` chunks with a vector already, and `texts - cached` texts to embed. |
| `dense::EmbedState` | How far a generation's embedding has come (`DenseIndexInfo::state`): `Embedding`, `Complete` (the default, and what a generation written before runs existed reads as), `CapReached`. |
| `vector::VectorStore` | `dimensions`, `len`, `is_empty`, `upsert(&[VectorRecord { id, vector, meta: VectorMeta { path, language } }])` (replaces, a batch all or nothing), `delete(&[id])`, `ids()` (ascending), `search(query, k, &Filter) -> Vec<VectorHit { id, score }>` (cosine, best first, then id ascending), `persist` (atomic). Ids are `VectorId` (`u128`). |
| `vector::CaprockStore` | The store, on Caprock: `create(dir, &StoreConfig)`, `open_for(dir, &StoreConfig)` (for writing, an error for another model or settings), `open(dir)` (read-only, the last persisted state), `matches(&StoreConfig)`, `compact()`. One writer per directory, any number of readers. A search returns at most 1,000 hits (Caprock's limit). |
| `vector::StoreConfig` | `model` (a `ModelDescriptor`), `text_format` (how the texts were rendered), `quantization`, `index`, with `new`, `with_quantization`, `with_index`, `dir_name()` (`vectors-` and 16 hex digits of a digest of the model and settings). |
| `vector::Quantization` | The codes kept in memory: `Int8` (default, 1 byte per dimension, each segment's best `max(4k, 40)` candidates rescored with the full-precision vectors) or `F32` (exact). |
| `vector::VectorIndex` | `Flat` (default, a scan of every code) or `Graph` (a Vamana graph on each segment of 20,000 vectors or more, searched approximately). |
| `vector::Filter` | `path_prefix` (a folder or file, matched by whole path components: `src/net` matches `src/net/retry.rs`, not `src/network.rs`), `language` (the chunk's `language`), with `with_path_prefix`, `with_language`, `matches`, `is_empty`. |
| `hybrid::search(state_dir, &SearchRequest, Option<&dyn Embedder>)` | One generation, read consistently. Returns `SearchResults { hits, mode, query_tokens, dense }`, where `dense` holds the generation's coverage. A search that loses a race with a build's switch starts over on the new generation, embedding the query once. |
| `hybrid::SearchRequest::new(query)` | `mode` (`Lexical`), `limit` (10), `candidates` (50, hybrid), `fusion` (RRF), `filter` (none), `query_vector` (none), with `with_*`. |
| `hybrid::SearchHit` | The chunk (flattened in JSON), `header`, `score` (BM25, cosine or fused), `lexical_rank`, `dense_rank` (1-based). |
| `hybrid::Fusion` | `Rrf { k: 60, lexical_weight: 1, dense_weight: 1 }` (default): `Σ weight / (k + rank)`. `Linear { lexical_weight }`: `w · minmax(lexical) + (1 − w) · minmax(dense)`. Ties by document id. |
| `hybrid::fuse(lexical, dense, &Fusion, limit)` | The fusion on its own, over `(id, score)` lists. |

Dense and hybrid modes need the generation to have embeddings and an embedder whose
descriptor equals the generation's (or a `query_vector`), and filters apply to both
sides. The dense side returns at most 1,000 candidates (Caprock's `MAX_K`), so
hybrid `candidates` above 1,000 are capped there.

### Embedding runs

A build with embeddings writes and publishes its lexical generation first, so lexical
search works as soon as the tree is chunked, and records each chunk's vector id and
text hash with it. The vectors come from an embedding run: `index2::build` runs one
after publishing and returns when it ends, and with `DenseBuild::embed_later` it returns
at once and leaves the run to `index2::embed`, called on any thread or in another
process (the CLI's `index2` starts one in the background). A run works on the
current generation:

1. It removes the store's vectors of chunks the generation lacks (the old texts of
   edited chunks, deleted files) and the stores of other models or settings.
2. It adds the vectors the embedding cache holds for chunks without one, committing
   every 16,384 vectors.
3. It embeds the remaining texts, request by request, from the texts the build spooled
   (`embed.spool` in the generation). Each request's vectors are appended to the cache,
   added to the store and committed, and then the coverage is recorded in
   `STATE_DIR/embed-progress.json`.

A run ends when every chunk has a vector (texts over the model's input limit stay
without one), at its token cap, when `DenseBuild::stop` is set (`EmbedStats::stopped`), on
an error, or when a newer generation is published (`EmbedStats::superseded`), which it
checks after every request. What it embedded stays in the cache, so the next run, for
the newer generation or after a crash, a stop or an error, embeds only the texts left and
pays for none twice. One run works at a time, under `STATE_DIR/.embed.lock`: a run waits
for the one in progress to end, which a newer generation makes it do after its current
request. Builds take `.index.lock` and don't wait for runs, except a build without
embeddings, which removes the vector stores once a run in progress has stopped.

When the run of `index2::build` fails (the provider is down, say), `build` returns the
error, and the generation it published stays searchable with the vectors stored so far.
A host that rebuilds as a watcher reports changes calls `index2::embed` again whenever a
run returns `superseded`, as the CLI's watchers do with `--index`.

**Searching during a run.** Dense and hybrid search use the vectors stored so far, with
exact scores, and find the other chunks by lexical search only. `--mode dense` searches
those vectors too, and finds nothing before the first request's vectors are stored (the
query is not embedded then). `SearchResults::dense` (`DenseIndexInfo`) tells how far the
coverage goes: `embedded` of `docs` chunks have a vector (at least: the run records it
after each commit), and `state` is `Embedding` while a run adds vectors or after one
stopped before finishing, `CapReached` after a run stopped at its cap, or `Complete`. The
CLI's `search2` prints it under the results, as in
`arko_index: dense: 4210 of 20000 chunks embedded (embedding continues in the background)`.
A search maps the store's ids through its own generation's vector keys, and an id names
a path and an embedded text, so a vector is only ever returned for the text it embeds:
an edited chunk is found densely again once its new text is embedded. `index2::status`
gives the current generation, its coverage, the latest run's `EmbedProgress` (texts
embedded and to embed, tokens billed, requests) and whether a run is working now.

## Vector store measurements

Measured through `vector::CaprockStore` on mhr_vector d7eb201, Caprock's earlier name,
whose search kernels are those of the pinned Caprock commit. One thread, a warm page
cache, a VMware VM with 6 cores of an Intel i9-13980HX (AVX2) and 22 GB of RAM.
Each store was built from vectors in memory (upserts in batches of 1,024, then
`compact` and `persist`, as a build does for a new store), reopened read-only, and
searched for the top 10 with no filter. Recall@10 is against an exact f64 cosine
ranking of every vector. Latencies are the medians of three runs (2,000, 1,000 and
1,000 queries on the Voyage sets, 500, 300 and 300 on the synthetic set). Memory is
the resident memory the open store added, disk is the store's directory.

| Vectors (1,024 dimensions) | Store | Build | Disk | Memory | Open | Median | p99 | Recall@10 |
|---|---|---|---|---|---|---|---|---|
| 34,601 documentation chunks (voyage-4) | int8, flat (default) | 1.2 s | 172 MB | 38 MB | 0.02 s | 2.2 ms | 2.8 ms | 1.000 |
| | f32, flat | 0.8 s | 138 MB | 139 MB | 0.11 s | 4.8 ms | 5.5 ms | 1.000 |
| | int8, graph | 15 s | 180 MB | 46 MB | 0.04 s | 0.41 ms | 1.0 ms | 0.999 |
| 199,999 code chunks (voyage-code-4) | int8, flat (default) | 6.1 s | 987 MB | 207 MB | 0.13 s | 13.5 ms | 18.8 ms | 1.000 |
| | f32, flat | 4.5 s | 791 MB | 793 MB | 0.60 s | 28.2 ms | 30.9 ms | 1.000 |
| | int8, graph | 192 s | 1,035 MB | 256 MB | 0.24 s | 1.6 ms | 3.0 ms | 0.973 |
| 1,000,000 synthetic | int8, flat (default) | 130 s | 4,933 MB | 1,033 MB | 0.65 s | 67.0 ms | 87.0 ms | 1.000 |

The recall of the Voyage sets is over 2,000 queries each (doc-comment sentences for
code, the Cloudflare evaluation questions and page headings for documentation), that
of the synthetic set over 500. The sets are Caprock's `docs-voyage4-200k` and
`code-voyagecode4-200k` (see its [benchmarks](https://github.com/mesa-hills-research/caprock/blob/main/docs/benchmarks.md)) and
its clustered generator (1,000 clusters, spread 0.35, seed 1). Builds write the whole store and
are bound by the disk, so their times vary with other work on the machine: earlier
builds of the code set's int8 store took from 6 to 37 s. The synthetic build includes
generating the vectors.

Int8 rescoring depth, recall@10 over the same 2,000 queries of each set (the default
is `max(4k, 40)`, so 40 at k = 10):

| Candidates rescored per segment | 10 | 20 | 32 | 40 | 64 | 100 |
|---|---|---|---|---|---|---|
| Documentation chunks | 0.99665 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Code chunks | 0.9954 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Updating the 199,999-vector store as a build does (open it for writing, upsert 1,000
vectors, delete 1,000, persist) took 0.13 s to open (it loads the codes), 6 ms to
apply and 17 ms to persist. A search runs on one thread, and a one-off search (the
CLI's `search2`) pays the open time first.

## Watchers

| Item | What it does |
|---|---|
| `open_watch[_with](root, state_dir, options, [rules])` | Opens (or creates) a polling watcher's state and reconciles with the tree. Any platform. |
| `Watcher::next_change(timeout)` | The next pending delta, or `None` when the tree is quiet. |
| `Watcher::ack(sequence)` | Durable, cumulative acknowledgement. A future sequence is rejected. |
| `Watcher::snapshot()`, `flush()`, `report()`, `rules_changed()` | The current chunks, persisting the state, the current `Report`, and whether the open applied changed rules. |
| `watch2::Watcher2::open[_with](root, state_dir, options, [rules])` | Linux: the inotify watcher. `next_change`, `next_change_to(timeout, writer)` (streams the delta's JSON line), `ack`, `report`, `rules_changed`, `file_count`, and `stats` (timings). |

`open_watch(ROOT, STATE, &options)` holds an OS advisory single-writer lock. State
contains the complete snapshot, exact per-chunk fingerprints, a monotone sequence, the
cumulative acknowledgement and pending deltas. A transaction writes a temporary file
through a buffered writer, flushes, fsyncs, atomically renames and fsyncs the
directory. Snapshot and outbox commit together **before** delivery. Pre-rename
failures do not advance in-memory state and can be retried. If the rename succeeds but
the directory sync fails, the on-disk outcome is uncertain: all further operations on
that `Watcher` return an error. **Drop and reopen it** to recover the actual journal
before proceeding. A killed writer releases the OS lock automatically.

Reconciliation on open detects changes made while the watcher was down. Polling hashes
file contents, so same-mtime and same-size edits are not missed, and unchanged files
reuse cached chunks and exact-byte fingerprints. Reopening reuses this cache only when
the source SHA-256, the options and an engine digest (the crate version, the chunking
source files and the locked tree-sitter versions) match. A 75 ms stable-inventory
debounce survives short `next_change` calls. Each pending delta is returned once per open
instance, and unacknowledged deltas replay in order after a restart. `ack(sequence)` is
durable and cumulative, and a future sequence is rejected. `next_change(timeout)` returns
`None` when quiet. `flush` persists the committed snapshot and journal.

`watch2` (`Watcher2`, Linux) keeps the same guarantees with different storage: a
checkpoint (`watch2.snap`) holding the file table and undelivered deltas, an
append-only record store with each file version's chunks, and a write-ahead journal
(`watch2.wal`) appended and fdatasync'ed before a delta is delivered. inotify events
and metadata reconciliation (size, mtime, ctime, inode, device, with the racy-git rule)
decide which files to read, so a commit costs in proportion to the change, not the
repository.

A change that keeps a chunk's ID but alters its bytes or any other field (range, lines,
scope, symbol path) emits `changed`, and a content-ID change emits remove plus add. Exact
fingerprints detect same-sized trailing-whitespace changes that trimmed IDs alone would
miss, including while the watcher was down. Applying deltas as sets transforms the prior
snapshot into the next complete snapshot.

The CLI flushes stdout before acknowledging a delivered line. SIGINT or SIGTERM stops a
watcher after its current batch, and a second signal ends it at once: it prints
`arko_index: stopped by a second signal` and exits with status 130 (a delta not yet
acknowledged replays on the next start). A broken output pipe leaves the delta
unacknowledged. Library callers should acknowledge only after their own durable
processing and deduplicate by sequence: delivery is at-least-once across crashes.

## Command line

`arko_index_cli` is an example of the crate:
`cargo run --release -p arko_index --example arko_index_cli -- COMMAND ...`.

| Command | What it does and prints |
|---|---|
| `chunk ROOT` (directory or file) | Every chunk of a directory (or of one file) as one JSON line, in path order, and a `FileMetadata` line before a Markdown file's chunks when it has front matter. |
| `outline ROOT` | The same partition as `path:start-end<TAB>kind<TAB>symbol_path` lines. A metadata region is `path:start-end<TAB>metadata<TAB>title`. |
| `index2 ROOT STATE_DIR` | Builds and publishes the disk-resident BM25 index. stderr gets `indexed N chunks, T terms, R runs` and the report summary. With embeddings (`--embed`, or the saved `dense.json` when no `--embed`, `--embed-config` or `--no-embed` is given), it then says how many chunks have vectors and how many texts are left to embed, starts `embed STATE_DIR` as a detached process of the same executable (its output in `STATE_DIR/embed.log`), and returns: lexical search works at once, and dense coverage fills in. `--wait-embeddings` embeds in this process instead, printing a progress line per request, the token summary and the vector store's update, and returns when the run ends. Flags below. |
| `embed STATE_DIR` | An embedding run in this process: embeds what the current generation lacks, with the model saved with the index, printing a progress line per request and the summary. `--max-embed-tokens N\|none` replaces the saved cap for this run. SIGINT or SIGTERM stops it after the current request (a second signal at once). It waits for a run in progress to end first. |
| `status STATE_DIR` | One JSON line, `index2::status`: the current generation, its chunks, its embeddings with their coverage and `state`, whether a run is embedding (`embedding`), and the latest run's progress. stderr gets a summary. |
| `search2 STATE_DIR QUERY...` | The best hits, up to `--limit` (10), as JSON lines: the chunk's fields plus `header`, `score`, `lexical_rank`, `dense_rank`. `--text` prints one line per hit, `rank. header  [mode score; ranks]`. A lexical search reads only the postings of the query's terms. Dense and hybrid print the query's tokens on stderr, and the coverage while some chunks have no vector. Flags below. |
| `index ROOT STATE_DIR` | The first index format: one JSON file that a search loads whole. stderr gets `indexed N chunks` and the report summary. |
| `search STATE_DIR QUERY...` | Up to ten chunk JSON lines from `index`'s file (the words are joined with spaces). Same tokens, scores and ranking as a lexical `search2` of an `index2` built without `--lexical-scope`. |
| `watch ROOT STATE_DIR` | Any platform: a polling watcher, which hashes the whole tree on every poll. Streams deltas as JSON lines, each acknowledged after it is flushed. Stops after the current batch on SIGINT or SIGTERM, and at once on a second signal (status 130). `--index INDEX_STATE_DIR` also keeps that index current (below). |
| `watch2 ROOT STATE_DIR` | Linux: inotify plus metadata reconciliation, otherwise as `watch`. stderr also gets a JSON line with the file count and timings at start, and one with the sequence and timings per delta. |
| `bench-scan ROOT` | Unix: one JSON line of timings of the scan primitives a watcher pass needs. |
| `--help`, `-h` | Usage on stderr. |

`chunk`, `outline`, `index`, `index2`, `watch`, `watch2` and `bench-scan` take the file
selection flags: `--respect-ignore`, `--exclude PATTERN`, `--include PATTERN`
(repeatable) and `--report FILE` (which `bench-scan` ignores). `--opt=value` works, and
`--` ends the flags. Those commands also take `--front-matter-description`
(`Options::front_matter_description`). `index2` and `search2` also take the flags below.

`watch` and `watch2` take `--index INDEX_STATE_DIR`: they then also keep that index2
state current. When a poll finds the tree quiet after deltas, they rebuild it with their
file selection flags and the embedding model saved there (choose one with
`index2 --embed` first), and a thread of the watcher embeds each new generation, a run
after each rebuild, continuing a run a rebuild superseded. stderr gets a line per rebuild,
`arko_index: index rebuilt: N chunks; dense: E have vectors, T distinct texts to embed`,
and a progress line per embedding request. On a start with an existing index the thread
continues its embedding, and a new index is built at the first quiet poll. A signal stops
the embedding after its current request.

The stdout of `chunk`, `search`, `search2` (without `--text`), `watch` and `watch2`
carries only JSON lines, and diagnostics, token counts included, go to stderr. Errors
print `arko_index: <message>` and exit with status 1. stderr lines from the report start
with `arko_index: `. Index files are durable and can be searched after a restart without
the source tree. An index writer takes an advisory lock, so a concurrent writer fails
fast while readers keep using the last published snapshot. The state directory must not
equal or contain the source root. Putting it inside the root is supported, and the walk
skips it.

Worker threads default to the CPUs the process may use (`ARKO_INDEX_THREADS`
overrides), and chunk output is identical at any thread count. The CLI (not the library)
sets glibc's mmap threshold to 1 MB unless `MALLOC_MMAP_THRESHOLD_` is set, which lowers
peak memory.

| `index2` flag | Effect |
|---|---|
| `--embed MODEL` | Embed with `MODEL`: a Voyage model (`voyage-code-4`), `voyage:MODEL`, or `hash` (offline, for trying things out). Saved in `dense.json`. Needs `VOYAGE_API_KEY` for Voyage. |
| `--dimensions N` | Vector dimensions (voyage-code-4: 256, 512, 1024 by default, 2048). |
| `--embed-text source\|path-source\|path-scope-source` | What each chunk's embedded text holds besides its source (default `path-scope-source`). Saved in `dense.json`. |
| `--embed-config FILE` | Take the settings from a file in the format of `dense.json`. `--dimensions` applies after it, checked against the file's model. With `--embed` it is an error (`--embed and --embed-config both choose the model: give one`). |
| `--quantize f32\|int8` | Vector codes held in memory: `int8` (default) or `f32`, exact at 4 bytes per dimension. Saved in `dense.json`. |
| `--vector-index flat\|graph` | How dense search finds candidates: `flat` (default) or `graph`, which adds Vamana graphs for very large indexes. Saved in `dense.json`. |
| `--max-embed-tokens N\|none` | The token cap of each embedding run (default none). Saved in `dense.json`. |
| `--no-embed` | Build lexically and remove `dense.json` (the cache stays). |
| `--wait-embeddings` | Embed in this process after publishing, and return when the run ends (by default a background process embeds). |
| `--lexical-scope` | Also index the words of each chunk's scope for lexical search (`BuildOptions::lexical_scope`, this build only). |

Without these flags, `index2` follows the saved `dense.json` if there is one. For
example, with the offline `hash` embedder on a 285-file tree, `index2` returns once
lexical search is ready:

```text
$ $CLI index2 --embed hash ./tree ./state
arko_index: embedding with hash/hash-v1 (256 dimensions), text path-scope-source
indexed 7412 chunks, 15730 terms, 6 runs
arko_index: 285 files indexed
arko_index: lexical search is ready; dense: 0 of 7412 chunks have vectors, 7412 of 7412 distinct texts to embed (0 in the cache)
arko_index: embedding in the background (process 2477566); `status ./state` shows the progress, ./state/embed.log the run's output
$ $CLI status ./state > /dev/null
arko_index: index2-1791518349020582000-2477525: 7412 chunks; dense (hash/hash-v1 (256 dimensions)): 0 of 7412 chunks have vectors (embedding continues in the background)
```

With a cap and `--wait-embeddings`, it prints a progress line per request and the run's
summary:

```text
$ $CLI index2 --embed hash --max-embed-tokens 300000 --wait-embeddings ./tree ./state
arko_index: embedding with hash/hash-v1 (256 dimensions), text path-scope-source, at most 300000 tokens per run
indexed 7412 chunks, 15730 terms, 6 runs
arko_index: 285 files indexed
arko_index: lexical search is ready; dense: 0 of 7412 chunks have vectors, 7412 of 7412 distinct texts to embed (0 in the cache)
arko_index: embedded 256/7412 texts, 44395 tokens, 1 requests; 256 of 7412 chunks have vectors
...
arko_index: embeddings: hash/hash-v1 (256 dimensions); 7412 chunks, 7412 distinct texts: 0 from the cache, 1756 embedded now in 7 requests, 299823 tokens
arko_index: stopped at the token cap (300000): 5656 texts (776862 counted tokens) not embedded; their chunks are found by lexical search only. Run `embed STATE_DIR` with a higher --max-embed-tokens (or none) to finish; embedded texts come from the cache
arko_index: 1756 of 7412 chunks have vectors; the cache holds 1756 vectors for this model
arko_index: vector store (new): 1756 vectors added, 0 removed
```

`embed --max-embed-tokens none ./state` then reported `1756 from the cache, 5656 embedded
now in 23 requests, 776862 tokens`.

| `search2` flag | Effect |
|---|---|
| `--mode lexical\|dense\|hybrid` | Default: `hybrid` when the index has embeddings, else `lexical`. |
| `--limit N`, `--candidates N` | Results (10) and hybrid candidates per side (50). |
| `--fusion rrf\|linear`, `--rrf-k K`, `--lexical-weight W`, `--dense-weight W` | Fusion (RRF, k 60, weights 1, linear weight 0.5). |
| `--path-prefix P`, `--language L` | Filters. A path prefix is a folder or a file, matched by whole components. |
| `--text` | One line per hit: rank, header, mode, score and ranks. |

```text
$ $CLI search2 --text --limit 2 ./index-state effective_max_input_tokens input limit
 1. arko_ai/src/client.rs:467-476 · impl LLMClient › fn max_input_tokens  [hybrid 0.0325; lexical #2, dense #1]
 2. arko_ai/src/input_limit.rs:10-27 · fn effective_max_input_tokens  [hybrid 0.0320; lexical #3, dense #2]
arko_index: query embedded with hash/hash-v1 (256 dimensions): 7 tokens
```

`dense_live_check` is a second example: it builds an index with Voyage embeddings of
a small corpus and runs the retrieval evaluation in the three modes, printing ranks,
scores and tokens (`--dry-run` only counts tokens). It spends Voyage tokens. See the
file's header and the [retrieval evaluation](arko_index-verification.md#retrieval-evaluation).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ARKO_INDEX_THREADS` | The CPUs the process may use (`available_parallelism`, which honours affinity masks and cgroup quotas) | Worker threads for chunking, `index2` and `watch2`'s parallel passes (`threads()`). |
| `ARKO_INDEX_WINDOW_MB` | 32 MB of source per thread | Reorder window of the ordered parallel map: how far workers may run ahead of the output. |
| `ARKO_INDEX_INDEX_MEM_MB` | 256 | CLI only: `index2`'s postings memory budget before spilling to runs. |
| `ARKO_INDEX_STATS` | Unset | When set, `index2` prints per-phase timings to stderr. |
| `MALLOC_MMAP_THRESHOLD_` | Unset: the CLI sets 1 MB | glibc's own variable. When it is set, the CLI leaves the threshold to glibc. |
| `VOYAGE_API_KEY` | Unset | Examples only (through arko_ai's `EmbeddingClient::from_env`): the key for embedding with Voyage. The library reads no credentials. |

## Parsing

Parsing uses [Mesquite](https://github.com/mesa-hills-research/mesquite)
(a Git dependency pinned to commit a94f0d8 in `Cargo.toml`), a pure, safe Rust port of
tree-sitter 0.25.10 and 54 grammars, so the library needs no C compiler. Its libraries keep
tree-sitter's names (`tree_sitter`, `tree_sitter_rust`, ...). The port pins the grammar
versions: an upgrade changes chunk boundaries and IDs. The grammars of the original nine
languages are the versions this crate used with the C library: tree-sitter-rust 0.24.2,
-python 0.25.0, -typescript 0.23.2 (TypeScript and TSX), -javascript 0.25.0, -go 0.25.0,
-java 0.23.5, -c 0.24.2 and -cpp 0.23.4. Mesquite fixes the Kotlin scanner's endless
loop on an annotation at the end of a file and the Markdown scanner's overflow at 255
nested blocks. Its `Parser::parse` returns no tree when a lexer or scanner keeps advancing
at the end of the input (2^20 times in one token), a guard the C library lacks.
[arko_index verification](arko_index-verification.md#the-pure-rust-parser)
has the checks of the switch.

Dense search stores its vectors with [Caprock](https://github.com/mesa-hills-research/caprock) (a Git dependency pinned to
commit 5ec18cc in `Cargo.toml`), also pure, safe Rust.

arko_index has no cargo features. The examples use arko_ai for Voyage, a dev-dependency
whose dependencies need a C compiler, so building the tests and examples needs one.

## Defaults

| Setting | Default |
|---|---|
| `Options::max_bytes` | 4,096 source bytes, and the rendered cap is 24,000 bytes |
| `Options::front_matter_description` | `false`: a Markdown scope's root is the front matter's title alone (or the first level-1 heading) |
| `BuildOptions::lexical_scope` (CLI `index2 --lexical-scope`) | `false`: lexical search indexes each chunk's path and source, not its scope |
| `FileRules::respect_ignore` | `false`: ignore files are not read, so gitignored files are indexed |
| `FileRules::exclude`, `include` | Empty |
| Built-in exclusions | `.git` directories and files, `CACHEDIR.TAG` directories (not the root), state directories |
| Non-UTF-8, binary, unreadable files | Skipped and reported, and the run continues |
| Symbolic links | Not followed, and reported |
| Search results | At most ten (`SearchRequest::limit` 10) |
| Polling watcher debounce | 75 ms of stable inventory |
| Threads | All available CPUs (chunking, `index2`, `watch2`) |
| Embeddings (`BuildOptions::dense`) | `None`: indexes are lexical, and dense is opt-in per index |
| When a build embeds (`DenseBuild::embed_later`) | `false`: `build` publishes the generation, then runs the embedding before it returns. The CLI's `index2` returns once lexical search is ready and embeds in a background process (`--wait-embeddings` waits) |
| Embedding model (CLI `--embed`) | None until chosen, then the saved `dense.json`. The CLI's Voyage models default to the model's dimensions (voyage-code-4: 1,024) with input types on |
| Token cap (`DenseBuild::max_tokens`, `DenseConfig::max_tokens_per_run`) | `None`: no cap. A cap applies to each embedding run |
| Embedded text (`DenseBuild::embed_text`, `DenseConfig::embed_text`, CLI `--embed-text`) | `PathScopeSource`: the path, the scope, then the source |
| Vector codes (`Quantization`, CLI `--quantize`) | `Int8`, with each segment's best `max(4k, 40)` candidates rescored at full precision, and `F32` is exact |
| Vector search (`VectorIndex`, CLI `--vector-index`) | `Flat` scans every code, and `Graph` adds Vamana graphs |
| Search mode (`SearchRequest::mode`) | `Lexical`, and the CLI's `search2` uses hybrid when the index has embeddings |
| Hybrid candidates (`SearchRequest::candidates`) | 50 per side |
| Fusion (`Fusion::default`) | Reciprocal rank fusion, k = 60, both weights 1 (`Linear`: lexical weight 0.5 in the CLI) |
| Filters (`Filter::default`) | None |
| Render version (`RENDER_VERSION`) | 1: path, scope, trimmed source |
| Embedding cache compaction | When the model's cache holds more than twice the index's distinct texts plus 1,024 records |
| Voyage adapter (examples) | 1,000 texts and 320,000 tokens per request for voyage-code-4, and 4 retries of transient failures |

## On-disk state

A state directory holds the `.arko-index-state` marker (the walk excludes any
directory that holds it) and `report.json` (the last walk's report).

| Writer | Files |
|---|---|
| `search::index` | `index.json` (version 1), `.index.lock` |
| `index2::index`, `index2::build` | `CURRENT2` (the name of the current generation), `.index.lock`, `index2-<stamp>-<pid>/` with `meta.json` (version 2, with `dense`: whether the generation has embeddings, `false` when absent), `lens.dat`, `docs.off`, `docs.dat`, `terms.dat`, `terms.idx`, `post.dat`, `files.dat` (each file's first document and path, for filters), and with embeddings also `dense.json` (`DenseIndexInfo`, version 2, naming the vector store, with the coverage when published), `vkeys.dat` (per document, by vector id: the 128-bit id big-endian, then the document as u32), `vhash.dat` (per document, in order: its embedded text's 32-byte hash) and `embed.spool` (the texts the cache lacked, removed when every one is embedded). Older generations are removed after a switch, or by a later build when one cannot be removed then. |
| `index2::build` with embeddings, `index2::embed` | `vectors-<digest>/`, the vector store (Caprock's files: `CURRENT`, manifests, segments, tombstones, `LOCK`), updated in place by embedding runs and shared by the generations. Also `dense.json` (`DenseConfig`, version 1: the model and settings the CLI's later builds follow), `embed-cache/<model key>/` (`cache.json` with the full descriptor and the epoch, `keys.<epoch>` with 32-byte text hashes, `vectors.<epoch>` with the vectors as returned, appended and fsynced per request), `embed-progress.json` (version 1: the generation the latest run embeds, its `state` and `EmbedProgress`, replaced after each commit of the store) and `.embed.lock`. The CLI's background run writes `embed.log`. |
| `open_watch` | `watch.json` (version 1: snapshot, fingerprints, sequence, acknowledgement, pending deltas, rules, and the engine-qualified parse cache) |
| `Watcher2` | `watch2.snap` (version 3), `watch2.recs.N`, `watch2.wal` |

Writers take an advisory lock on the state (the marker, or `.index.lock`), so a second
writer fails fast. An embedding run takes `.embed.lock` and waits for one in progress.
Commits write a temporary file, fsync it, rename it and fsync the directory (on Unix).

**Compatibility.** arko_index is unpublished, and its on-disk data has no compatibility
guarantee between versions: an index or watcher state from an earlier build may be
refused (`watch2` refuses a version 2 snapshot and asks for a rebuild) or rebuilt. A
watcher's parse cache is reused only when the engine digest (the crate version, the
chunking sources and the locked tree-sitter versions) and `max_bytes` match. Otherwise
the files are parsed again, without spurious deltas, except once for chunk format 2:
chunks that gain a `scope` are delivered as `changed`. The engine digest also changes
with `Options::front_matter_description`, and with the Mesquite commit (the move
to a94f0d8 rebuilds the watchers' cached chunks once). An index2 generation from before
this version loads and searches lexically, and filters need its `files.dat` (rebuild to
filter). Saved files with embeddings carry versions (`dense.json` 1, the generation's
`dense.json` 2, `cache.json` 1, and Caprock's own format versions) and new fields get
serde defaults. A generation from before embedding runs (no `vhash.dat`) is searched as
before, its `state` read as `Complete`, and `index2::embed` asks for a rebuild. A
generation whose vectors an earlier version stored (its `dense.json` is version 1)
searches lexically with `--mode lexical`, and dense and hybrid search
(the CLI's default) ask for a rebuild before embedding the query. The rebuild takes the
vectors from the embedding cache. A build replaces a vector store it cannot use
(damaged, missing a file, or in a format this version cannot read) with a new one, from
the embedding cache. The stores of arko_index versions that used mhr_vector, Caprock's
earlier name, are in such a format: dense search asks for a rebuild, and the first build
replaces them. The SQLite
databases of the earlier, model-chunked arko_index are not readable by this crate.

## Platforms

Linux is the primary platform: every command and test runs there. macOS has everything
but `watch2` (inotify). On Windows the crate compiles, the watchers' directory fsync is
skipped (Windows cannot open a directory as a file), and `bench-scan` and `watch2` are
unavailable. The CI tests the crate on macOS and only compiles it on Windows. The
identity checks and measurements behind these docs were made on Linux.

## Trade-offs

* `watch` reads the whole tree on every poll (cached parsing, not incremental
  filesystem I/O), and `watch2` is the incremental alternative on Linux.
* `search` loads the entire JSON index, and `search2` reads only the query terms'
  postings.
* Dense search is opt-in and costs tokens to build and per query. A search holds every
  vector's int8 codes in memory (1 KB per chunk at 1,024 dimensions) and scans them
  all on one thread, so its time grows with the index: `--vector-index graph` is the
  alternative for millions of chunks. A one-off search (the CLI's `search2`) also
  loads the codes first. An embedding run holds about 300 bytes per chunk in memory
  (text hashes, vector keys and the cache index) besides the store's codes, and the
  texts to embed are spooled to disk.
* Until a run has embedded a generation, dense and hybrid search cover only the chunks
  with a vector. The coverage a search reports is what the run recorded after its last
  commit. A build counts the vectors its generation already has when it publishes, and
  one that overlaps the first step of the previous generation's run can count a few
  vectors that run then removes: the generation's own run corrects the count.
* C/C++ recovery is heuristic, not a C preprocessor. Tiny merged definitions can lose
  retrieval concentration.
* The first `watch2` delta of a large tree is one JSON line holding every chunk.
