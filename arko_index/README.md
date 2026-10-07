# arko_index

Deterministic, syntax-aware partitioning of source repositories into chunks, offline
BM25 code search, and durable watchers that turn file changes into chunk deltas. Eight
code families are parsed with tree-sitter (Rust, Python, JavaScript, TypeScript
including TSX, Go, Java, C, C++); other UTF-8 text is split by a strictly bounded
fallback. Everything runs locally: no network, no credentials, no model calls.

The crate builds the tree-sitter C runtime and grammars, so it needs a C compiler.
Linux is the primary platform; see [Platforms](#platforms). The
[arko_index reference](../docs/arko_index-reference.md) lists the API, the CLI, the
environment variables and the on-disk state.

## Command line

The CLI ships as the `arko_index_cli` example:

```sh
cargo build --release -p arko_index --example arko_index_cli
CLI=target/release/examples/arko_index_cli
$CLI chunk ./repo > chunks.jsonl
$CLI index2 ./repo ./index-state           # disk-resident BM25 index
$CLI search2 ./index-state 'exponential backoff retry'
$CLI outline ./repo                        # path:start-end, kind, enclosing symbols
$CLI watch2 ./repo ./watch-state           # JSONL durable delta stream (Linux)
# Choosing files (any command that walks a tree): see "What gets indexed"
$CLI chunk --respect-ignore --exclude 'vendor/' --report skipped.json ./repo
```

| Command | What it does |
|---|---|
| `chunk ROOT` | Prints every chunk of a directory (or of one file) as one JSON line, in path order. |
| `outline ROOT` | The same partition as `path:start-end<TAB>kind<TAB>symbol_path` lines. |
| `index2 ROOT STATE_DIR`, `search2 STATE_DIR QUERY...` | Builds the disk-resident BM25 index; prints the best hits (at most ten) as chunk JSON lines. A search reads only the postings of the query's terms. |
| `index ROOT STATE_DIR`, `search STATE_DIR QUERY...` | The first index format: one JSON file that a search loads whole. Same tokens, scores and ranking as `index2`/`search2`. |
| `watch2 ROOT STATE_DIR` | Linux: inotify plus metadata reconciliation; streams deltas as JSON lines and acknowledges each after a flushed line. |
| `watch ROOT STATE_DIR` | Any platform: a polling watcher with the same delta protocol, which hashes the whole tree on every poll. |
| `bench-scan ROOT` | Unix: prints timings of the scan primitives a watcher pass needs. |

`chunk`, `outline`, `index`, `index2`, `watch`, `watch2` and `bench-scan` accept the file
selection flags below. `chunk`, `search`, `search2`, `watch` and `watch2` stdout carries
only JSON lines; diagnostics go to stderr. Index files are durable and can be searched
after a restart without the source tree. An index writer takes an advisory lock, so a
concurrent writer fails fast while readers keep using the last published snapshot. The
state directory must not equal or contain the source root; putting it inside the root
is supported, and the walk skips it.

A result is metadata, not a copy of the source: read the half-open UTF-8 byte range
`start_byte..end_byte` of `ROOT/path` to see the exact text. Lines are inclusive and
1-based; a final newline belongs to its line.

Worker threads default to the CPUs the process may use (`ARKO_INDEX_THREADS` overrides);
chunk output is identical at any thread count. The CLI sets glibc's mmap threshold to
1 MB (unless `MALLOC_MMAP_THRESHOLD_` is set), which lowers peak memory; the library
does not.

## Library

`arko_index::bench_v1` holds the chunking and polling-watcher API (`Options`, `Chunk`,
`SourceSpan`, `Delta`, `Watcher`, `chunk_str`, `chunk_file`, `chunk_dir*`,
`open_watch*`); `search`, `index2` and (on Linux) `watch2` hold the indexes and the
inotify watcher.

```rust
use arko_index::bench_v1::{chunk_str, Options};
use std::path::Path;
let chunks = chunk_str(Path::new("sample.rs"), "fn hello() {}\n", &Options::default())?;
# Ok::<(), arko_index::bench_v1::Error>(())
```

For directory input, paths are slash-separated and relative. `chunk_file` uses the file
name; `chunk_str` keeps the supplied UTF-8 path verbatim. Literal Unix backslashes are
filename characters, not separators. A directory walk skips and reports undecodable
names instead of letting them collide with another path; `chunk_file` and `chunk_str`
return an error for one. Empty files produce no chunks. Every included nonempty text
byte, including whitespace, is in exactly one chunk.

## What gets indexed

The same rules apply to every command that walks a tree and to the library
(`chunk_dir`, `chunk_dir_each`, `chunk_dir_map`, `open_watch`, `search::index`,
`index2::index`, `Watcher2::open`, and their `_with` variants).

**Defaults.** Every regular file under the root is indexed, including hidden files
and files that `.gitignore` lists, except:

* `.git` directories, and `.git` files (the one-line `gitdir:` pointer of a submodule
  or linked worktree);
* directories that hold a valid `CACHEDIR.TAG`: a file whose first 43 bytes are
  `Signature: 8a477f597d28d172789f06886806bc55` ([Cache Directory Tagging
  Specification](https://bford.info/cachedir/)). Cargo writes one into `target/`;
  other build tools and caches do the same. A directory named `target` without the
  tag (Linux's `drivers/target`) is indexed like any other. The root itself is
  indexed even when it is tagged, and an `--include` pattern that names a tagged
  directory brings it back (below);
* the tool's own state directories inside the root: the `STATE_DIR` given to
  `index`/`index2`/`watch`/`watch2`, and any directory that holds the
  `.arko-index-state` marker those commands create.

No other name is excluded. Symbolic links are never followed: a link is reported, and
a target inside the root is indexed under its own path. Ignore files are not read
unless you opt in, and ignore files in directories **above** the root never apply.

**Files that cannot be indexed are skipped, never silently.** Each one is listed in
the report (below) with its reason, and the run continues:

| Reason | Meaning |
|---|---|
| `binary` | the file contains a NUL byte |
| `not-utf8` | no NUL byte, but not valid UTF-8 (Latin-1 text, for example) |
| `name-not-utf8` | the file or directory name is not UTF-8 (chunk paths are UTF-8); a directory is skipped whole |
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
catch-all such as `*`) brings that directory back; one that only matches files inside
it (`*.rs`) does not. `.git` and the state directories always stay excluded. An
invalid pattern is an error before any work. `--opt=value` works too, and `--` ends
the options. In Rust:

```rust
use arko_index::bench_v1::{chunk_dir_with, FileRules, Options};
let mut rules = FileRules::default(); // non_exhaustive: set fields on a default
rules.respect_ignore = true;
rules.exclude.push("vendor/".into());
let (chunks, report) = chunk_dir_with(std::path::Path::new("."), &Options::default(), &rules)?;
eprintln!("{}", report.summary());
# Ok::<(), arko_index::bench_v1::Error>(())
```

*Which ignore files apply with `--respect-ignore`.* `.gitignore` and `.ignore` files in
the root and below, and `.git/info/exclude` of every repository whose `.git` directory
is at or below the root (it applies to everything under that repository's directory,
nested repositories included). Never: ignore files above the root (indexing a
subdirectory of a repository does not apply the repository's top-level `.gitignore`;
index the repository root, or repeat those patterns with `--exclude`); the global Git
excludes file (`core.excludesFile`, `~/.config/git/ignore`); the `info/exclude` of a
repository whose `.git` is a file (submodule, linked worktree), which lives in a git
directory elsewhere.

*Precedence.* An entry is attributed to the first rule that removes it: ignore files
(when honoured), symlink, special file, `.git` (directory or file), state directory,
`--exclude`, `--include` (files only), name not UTF-8, `CACHEDIR.TAG` (unless an
`--include` pattern names the directory); then, when the file is read: `unreadable`,
`binary`, `not-utf8`.

**The report.** Everything that is not indexed is listed in `skipped` with its reason,
except what your own rules remove. Ignore-file removals are counted per source
(`ignored_by_gitignore`, `ignored_by_ignore_file`, `ignored_by_git_info_exclude`; a
directory counts once) and listed in `ignored` (sorted, at most 1,000 entries; the
counts are complete). `--exclude` and `--include` removals are counted
(`excluded_by_pattern`, `not_included`).

* `chunk` and `outline` print one summary line to stderr only when something other
  than the built-in exclusions (`.git`, `CACHEDIR.TAG` directories, state directories)
  was skipped, a warning occurred, or nothing was indexed; your own rules alone do not
  trigger it. Then up to 20 entries per reason for every reason except `binary`,
  `symlink`, `git-dir`, `git-file` and `state-dir`, which are only counted.
  `--report FILE` is always written, complete:

  ```text
  arko_index: 90419 files indexed; 87 skipped (4 binary, 2 not-utf8, 81 symlink)
  arko_index: skipped (not-utf8): arch/m68k/hp300/hp300map.map
  arko_index: skipped (not-utf8): drivers/tty/vt/defkeymap.map
  ```
* `index` and `index2` print the same after their own line and write the full report
  to `STATE_DIR/report.json` after the index commits.
* `watch` and `watch2` print it at start, print a new summary line whenever the counts
  change, and keep `STATE_DIR/report.json` current (replaced atomically whenever the
  report changes, by `watch2` right after the batch's delta is delivered; not fsynced,
  because every start rebuilds it).
* Library: `chunk_dir_with` and `chunk_dir_each_with` return a `Report`,
  `search::index_with` returns `(chunks, Report)`, `index2::index_with` returns it in
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
skipped nothing, such as an invalid line in an ignore file.

**Watchers and rule changes.**

* `watch` re-walks with the rules on every poll, so editing an ignore file or
  `.git/info/exclude` (when honoured) or adding or removing a `CACHEDIR.TAG` produces
  a delta like any other change.
* `watch2` runs a full reconciliation on any event for a `CACHEDIR.TAG` and, when
  ignore files are honoured, for a `.gitignore` or `.ignore`, or for `exclude` in a
  repository's `.git/info` (watched for that file only; a `.git/info` directory created
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

## Boundaries and identities

* Default `max_bytes=4096` is a strict source bound; values below 4 are rejected
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
  Identical duplicates stay separate; long same-family runs are not greedily paired.
  Family chunks are not merged again as tiny definitions.
* Sustained runs (eight or more) of tiny same-kind definitions can be combined into
  coherent groups up to 768 bytes. Content-derived cut points resynchronize after an
  insertion, unlike fixed-count windows. Ordinary short files and substantial
  unrelated functions stay independent.
* The C++ grammar parses `.h` files, whose `language` field stays `c`. If C/C++
  parsing has errors, generic annotation macros are replaced with spaces in a
  temporary parse buffer, and that tree is kept only if ERROR coverage decreases.
  Returned ranges always refer to the unmodified source.
* A deterministic parser-work budget and a 65,536-byte maximum physical source line
  prevent adversarial lexer and parser stalls; giant lines use the bounded fallback.
  Single-child AST wrapper descent is iterative and depth-bounded. Missing parses,
  unsupported formats, deep recovery trees and oversized atomic literals also use the
  fallback, which prefers complete lines or whitespace, then a UTF-8 boundary. An
  atom or string larger than the bound is split inside: exact coverage and the size
  bound take precedence over keeping such a token whole.
* IDs hash the path plus the trimmed chunk content, with a per-content occurrence
  ordinal. Absolute offsets are excluded, so unrelated edits do not change the IDs of
  unchanged units. Identical duplicates get distinct IDs. Renames change IDs. Edits
  to a merged unit can change that unit's membership and identity; nearby
  content-defined groups resynchronize. Source offsets are always current.

## Search

`search` and `search2` score with BM25 (k1 = 1.2, b = 0.75) over lowercase words plus
identifier components (snake_case parts and camelCase humps), so `retryPolicy` matches
`retry policy`. Ties break by path, then start byte; at most ten hits are returned.
`index2` writes a new generation directory and publishes it by atomically rewriting
`CURRENT2`, so readers keep the previous generation until the switch. Its build memory
is bounded: postings accumulate up to a budget (the CLI's default is 256 MB,
`ARKO_INDEX_INDEX_MEM_MB`) and then spill to sorted runs that are merged. The watchers
maintain chunk snapshots, not an index: rerun `index2` to refresh search.

## Durable watching

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
file contents, so same-mtime and same-size edits are not missed; unchanged files reuse
cached chunks and exact-byte fingerprints. Reopening reuses this cache only when the
source SHA-256, the options and an engine digest (the chunking source files and the
locked tree-sitter versions) match. A 75 ms stable-inventory debounce survives short
`next_change` calls. Each pending delta is returned once per open instance;
unacknowledged deltas replay in order after a restart. `ack(sequence)` is durable and
cumulative; a future sequence is rejected. `next_change(timeout)` returns `None` when
quiet. `flush` persists the committed snapshot and journal.

`watch2` (`Watcher2`, Linux) keeps the same guarantees with different storage: a
checkpoint (`watch2.snap`) holding the file table and undelivered deltas, an
append-only record store with each file version's chunks, and a write-ahead journal
(`watch2.wal`) appended and fdatasync'ed before a delta is delivered. inotify events
and metadata reconciliation (size, mtime, ctime, inode, device, with the racy-git rule)
decide which files to read, so a commit costs in proportion to the change, not the
repository.

An ID-preserving byte, range or line change emits `changed`; a content-ID change emits
remove plus add. Exact fingerprints detect same-sized trailing-whitespace changes that
trimmed IDs alone would miss, including while the watcher was down. Applying deltas as
sets transforms the prior snapshot into the next complete snapshot.

The CLI flushes stdout before acknowledging a delivered line and handles SIGINT and
SIGTERM. A broken output pipe leaves the delta unacknowledged. Library callers should
acknowledge only after their own durable processing and deduplicate by sequence:
delivery is at-least-once across crashes, not exactly-once.

## Platforms

Linux is the primary platform: every command and test runs there. On macOS everything
except `watch2` (inotify) is available. On Windows the crate compiles and the watchers'
directory fsync is skipped (Windows cannot open a directory as a file); `bench-scan` and
`watch2` are unavailable. The CI builds and tests the crate on macOS and Windows; the
identity checks and measurements behind this README were made on Linux.

## Validation tools

```sh
cargo test -p arko_index --locked
python3 arko_index/tools/validate_cli.py target/release/examples/arko_index_cli
python3 arko_index/tools/audit_coverage.py CORPUS chunks.jsonl
```

`validate_cli.py` drives the CLI end to end: repeated `index`, hidden, ignored and state
paths, best-first `search`, and `watch` through SIGKILL restarts, burst edits, SIGTERM
and a broken pipe. `audit_coverage.py` checks a `chunk` output against a corpus without
using the crate's code: every UTF-8 file covered byte for byte in order, unique IDs,
line numbers, languages and the size bounds. It reads every file under the corpus, so
use it on trees without `.git` or `CACHEDIR.TAG` directories.

## Trade-offs

* `watch` reads the whole tree on every poll (cached parsing, not incremental
  filesystem I/O); `watch2` is the incremental alternative on Linux.
* `search` loads the entire JSON index; `search2` reads only the query terms' postings.
* Search is lexical (BM25 and identifier decomposition); there are no embeddings.
* C/C++ recovery is heuristic, not a C preprocessor. Tiny merged definitions can lose
  retrieval concentration.
* The first `watch2` delta of a large tree is one JSON line holding every chunk.

## Origin

arko_index was produced by an autonomous research run (as `ar_chunker`), then hardened
for repository scale and for the file-inclusion rules above. It replaced an earlier
arko_index that chunked with a language model, embedded with Voyage and searched a
SQLite database; the new crate cannot read that crate's databases.
