# arko_index

Deterministic, syntax-aware partitioning of source repositories into chunks, offline
BM25 code search, optional dense (embedding) and hybrid search, and durable watchers
that turn file changes into chunk deltas. Code, prose, markup and data in 50 languages
are parsed with tree-sitter and chunked along their structure (see
[Languages](#languages)); other UTF-8 text is split by a strictly bounded fallback.
Every chunk carries a context header: its enclosing definitions
(`impl Display for Point › fn fmt`) or, in Markdown, the document's title and heading
chain. Chunking, lexical search and watching run locally: no
network, no credentials, no model calls. Dense search is opt-in: it embeds through an
[`Embedder`](#dense-and-hybrid-search) the caller supplies (the CLI uses Voyage through
arko_ai), and the library itself never makes a network call.

Parsing uses mhr_tree_sitter, a pure-Rust port of tree-sitter, so the library needs no
C compiler; see [Parsing](#parsing). (Its examples and tests use arko_ai for Voyage,
whose dependencies do.) Linux is the primary platform; see [Platforms](#platforms).
The [arko_index reference](../docs/arko_index-reference.md) lists the API, the CLI,
the environment variables and the on-disk state.

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
# Dense and hybrid search (opt-in; spends Voyage tokens): see "Dense and hybrid search"
export VOYAGE_API_KEY=...
$CLI index2 --embed voyage-code-4 --max-embed-tokens 5000000 ./repo ./index-state
$CLI index2 ./repo ./index-state           # later builds reuse the saved model and cache
$CLI search2 --text ./index-state 'where is the input limit applied'   # hybrid
$CLI search2 --mode lexical --language rust ./index-state 'RetryPolicy'
```

| Command | What it does |
|---|---|
| `chunk ROOT` | Prints every chunk of a directory (or of one file) as one JSON line, in path order. |
| `outline ROOT` | The same partition as `path:start-end<TAB>kind<TAB>symbol_path` lines. |
| `index2 ROOT STATE_DIR`, `search2 STATE_DIR QUERY...` | Builds the disk-resident BM25 index (with `--embed`, also embeddings); prints the best hits (ten by default) as JSON lines: the chunk's fields plus `header` and `score` (and `lexical_rank`, `dense_rank`), or with `--text` one line per hit. A lexical search reads only the postings of the query's terms. |
| `index ROOT STATE_DIR`, `search STATE_DIR QUERY...` | The first index format: one JSON file that a search loads whole. Same tokens, scores and ranking as `index2`/`search2`. |
| `watch2 ROOT STATE_DIR` | Linux: inotify plus metadata reconciliation; streams deltas as JSON lines and acknowledges each after a flushed line. |
| `watch ROOT STATE_DIR` | Any platform: a polling watcher with the same delta protocol, which hashes the whole tree on every poll. |
| `bench-scan ROOT` | Unix: prints timings of the scan primitives a watcher pass needs. |

`chunk`, `outline`, `index`, `index2`, `watch`, `watch2` and `bench-scan` accept the file
selection flags below; `index2` and `search2` also take the flags in
[Dense and hybrid search](#dense-and-hybrid-search). `chunk`, `search`, `search2`
(without `--text`), `watch` and `watch2` stdout carries only JSON lines; diagnostics,
token counts included, go to stderr. Index files are durable and can be searched
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
inotify watcher; `dense` (the `Embedder` interface, the per-index config), `vector`
(the `VectorStore` trait and the exact store) and `hybrid` (lexical, dense and hybrid
search) hold dense search.

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
byte, including whitespace, is in exactly one chunk, or in a Markdown file's metadata
region (its front matter), which `chunk_str_with_metadata` and
`chunk_file_with_metadata` return with the chunks.

## Languages

Bash, C, C#, C++, CMake, CSS, CUDA, Dart, DTD, Elixir, Elm, embedded templates (ERB,
EJS), Erlang, F#, Fortran, GLSL, Go, Haskell, HCL (Terraform), HTML, Java, JavaScript,
JSON, Julia, Kotlin, Lua, Make, Markdown and MDX, Objective-C, OCaml, Odin, Perl, PHP,
PowerShell, Protocol Buffers, Python, R, Ruby, Rust, Scala, Solidity, SQL, Svelte,
Swift, TOML, TypeScript and TSX, Verilog, XML, YAML and Zig.

Each grammar has a table of rules: which node kinds are definitions, which containers
expose their members, what attaches to what, and how scopes are labelled. Code splits
into its definitions, prose into sections, markup into elements and data into keys or
tables. A Markdown file's front matter (and MDX's leading `import`/`export` lines) is
file metadata: the chunks cover the rest, and its title roots every chunk's scope.
[arko_index languages](../docs/arko_index-languages.md) lists the extensions and file
names of each language, how ambiguous ones are settled (`.h`, `.m`, `.v`, `.pl`) and
every language's rules.

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

## Context headers

Every chunk has a `scope`, shown with a search result as its header,
`path:start_line-end_line · scope`:

```text
src/geometry.rs:27-34 · mod geometry › impl fmt::Display for Point › fn fmt
tools/retry.py:22-32 · class RetryPolicy › async def run
src/pool.ts:5-17 · namespace Storage
include/router.h:18-26 · namespace net › namespace http › class Router › function find
docs/guide.md:14-21 · # Guide › ## Install › ### Linux
```

* **Code**: the chain of definitions that enclose the chunk, from the syntax tree, the
  chunk's own definition included: `mod`, `impl Trait for Type`, `trait`, `struct`,
  `enum`, `fn`, `macro_rules!` (Rust); `class`, `def`, `async def` (Python, through
  decorators); `namespace`, `module`, `class`, `interface`, `enum`, `type`,
  `function` (also an arrow function bound to a name), `method` (TypeScript and
  JavaScript, through `export`); `func`, `func (r *T) Name`, `type` (Go); `class`,
  `interface`, `enum`, `record`, `method`, `constructor` (Java); `namespace`, `class`,
  `struct`, `enum`, `function` (C and C++, through templates). A chunk merged from
  several definitions (a function family, a run of tiny definitions) shows the
  enclosing scope and the members joined with ` | `; text packed between definitions
  shows the enclosing scope.
* **Markdown** (`.md`, `.markdown`, `.mdx`): the document's title (the front matter's
  `title`, or the first `# H1`), then the heading chain in effect where the chunk
  starts (`Guide › ## Install › ### Linux`), from ATX and setext headings. Fenced code
  blocks and the front matter hold no headings. A chunk that starts before the first
  heading but contains one takes that heading's chain.
* **Other languages**: the labels their rules give, from `defmodule Shop.Cart › def add`
  (Elixir) to `jobs › lint` (YAML) and `<body> › <section id="summary">` (HTML), listed
  in [arko_index languages](../docs/arko_index-languages.md).
* Other text has no scope.

Labels longer than 200 bytes end in `…`; a scope over 1,000 bytes keeps its innermost
elements after `… › `. `symbol_path` is unchanged: names only (`Point::fmt`), without
modules or keywords.

**Headers are metadata.** They never change chunk boundaries or IDs. The `scope` field
is chunk JSON **format 2** (`bench_v1::CHUNK_FORMAT`): format 1 had the ten fields up
to `token_estimate`; format 2 appends `scope`, written only when it is not empty, so a
chunk without one serializes exactly as before. Readers treat a missing `scope` as
empty. Because the field is new, a watcher state written before format 2 reconciles
on its first open with a one-time `changed` delta for every chunk that has a scope
(same IDs, bytes and lines); nothing is added or removed. Format 3 adds a line before
a Markdown file's chunks that records its metadata region (front matter), with a `file`
key where chunks have `id`. The chunks themselves are unchanged.

## Search

`search` and `search2` score with BM25 (k1 = 1.2, b = 0.75) over lowercase words plus
identifier components (snake_case parts and camelCase humps), so `retryPolicy` matches
`retry policy`. Ties break by path, then start byte; at most ten hits are returned
(`search2 --limit` and the library's `hybrid::search` choose another number).
`index2` writes a new generation directory and publishes it by atomically rewriting
`CURRENT2`, so readers keep the previous generation until the switch. Its build memory
is bounded: postings accumulate up to a budget (the CLI's default is 256 MB,
`ARKO_INDEX_INDEX_MEM_MB`) and then spill to sorted runs that are merged. The watchers
maintain chunk snapshots, not an index: rerun `index2` to refresh search (with
embeddings, a rerun embeds only the chunks whose text changed).

## Dense and hybrid search

Dense search finds chunks by the similarity of embeddings, so a question in words
finds code that uses other words; on the arko retrieval evaluation, dense
voyage-code-4 found 9 of 9 definitions by depth questions where BM25 found 4, and 5
of 6 behaviour questions against 4, while BM25 found 5 of 7 callers against 2. Hybrid
search fuses both. **Dense search is opt-in**: an index is lexical unless a build is
asked to embed, and embedding calls a provider that bills for it.

**Embedders.** The library defines the interface, `dense::Embedder`, and makes no
network calls: `descriptor()` names the vectors (provider, model, dimensions, whether
documents and queries use different input types, an optional revision),
`limits()` gives the request limits, `count_tokens()` counts (or bounds) tokens, and
`embed(texts, Document | Query)` returns the vectors and the tokens billed. Local
models will implement the same trait. The crate ships `dense::HashEmbedder`, an
offline, deterministic feature-hashing embedder for tests and trying things out (it
knows nothing of meaning). The examples' `support/voyage.rs` adapts arko_ai's Voyage
client (its HTTP, error classification and tokenizer): documents are embedded with
`input_type: document`, queries with `query`, the dimensions always sent,
`truncation: false`, requests packed to at most 1,000 texts and 320,000 tokens
(voyage-code-4's measured limit), and transient failures retried four times with
backoff.

**What is embedded** (render version 1, `dense::render`): the path, the scope on its
own line when there is one, then the chunk's source without leading and trailing
whitespace. Line numbers are left out, so an edit above a chunk does not change its
text.

**The model is chosen per index and saved.** A build with embeddings writes
`STATE_DIR/dense.json` (`dense::DenseConfig`): provider, model, dimensions,
input-type use, render version, vector quantization and the token cap. Later builds
read it (the CLI does when no `--embed` is given), so a rebuild after a watcher
reported changes uses the same model unasked. `--no-embed` (or `DenseConfig::remove`)
turns embeddings off for the index. Each generation records the model and the
coverage of its own vectors (`dense.json` in the generation); dense search embeds the
query with that model and refuses an embedder of another.

**Embeddings are cached** by (model descriptor, SHA-256 of the rendered text) in
`STATE_DIR/embed-cache/<model key>/`, across generations. A build embeds only the
texts the cache lacks, each distinct text once: an unchanged chunk is never embedded
twice, a changed one is embedded again, and a model change (another model, other
dimensions or input types) embeds everything with the new model in its own cache
directory, so switching back costs nothing. Vectors are appended and fsynced after
every request, so tokens paid for are kept even when a build fails or stops. When
records of texts no longer in the index outnumber the live ones (over twice as many,
plus 1,024), the build compacts the model's cache; other models' caches stay until
their directories are deleted.

**Token accounting and the cap.** A build reports, in `BuildStats::dense`
(`dense::EmbedStats`; the CLI prints it), the chunks, distinct texts, cache hits,
texts embedded, requests and tokens billed (Voyage's figure). A per-run cap
(`DenseBuild::max_tokens`, `--max-embed-tokens N`, saved with the index; no cap by
default) stops embedding before a request would take the run's billed tokens past it:
the build still completes and publishes, the chunks without vectors are found by
lexical search only, and the report lists them with their counted tokens, what
finishing would take. A rerun with a higher cap embeds only those. `--max-embed-tokens
0` sends nothing and reports the full count. A search reports
`SearchResults::query_tokens`, the tokens billed to embed its query (0 for lexical
search, or when the caller passes a vector it embedded with `dense::embed_query`), for
callers that charge queries to a budget.

**Vectors** are stored per generation, published with it by the same atomic
`CURRENT2` switch, behind `vector::VectorStore` (insert, delete, top-k search with an
optional path-prefix or language filter, persist, open). `vector::BruteForceStore` is
exact: every search reads all vectors, in 1 MB blocks, so memory stays at about 12
bytes per vector; scores are cosine similarities; equal scores order by document id.
`--quantize int8` (`Quantization::Int8`) adds an 8-bit copy that a search scans first,
a quarter of the bytes, then rescores the best `4 × k` (at least 40) exactly. An
approximate store can replace it behind the same trait.

**Hybrid fusion.** `hybrid::search` takes the best 50 chunks (`candidates`) from each
side and fuses the rankings with reciprocal rank fusion by default: a chunk scores
`Σ weight / (60 + rank)` over the lists that contain it, ranks from 1, both weights 1,
so a chunk both sides rank well comes first and neither score scale matters.
`Fusion::Linear` (`--fusion linear --lexical-weight W`) instead adds `W` times the
min-max normalized BM25 score and `1 − W` times the normalized similarity. Ties order
by document id. Path-prefix and language filters (`--path-prefix`, `--language`)
apply to both sides.

**Command line.**

| `index2` flag | Effect |
|---|---|
| `--embed MODEL` | Embed with `MODEL`: a Voyage model (`voyage-code-4`), `voyage:MODEL`, or `hash` (offline, for trying things out). Saved in `dense.json`. Needs `VOYAGE_API_KEY` for Voyage. |
| `--dimensions N` | Vector dimensions (voyage-code-4: 256, 512, 1024 by default, 2048). |
| `--embed-config FILE` | Take the settings from a file in the format of `dense.json`. |
| `--quantize f32\|int8` | Vector storage (default `f32`). |
| `--max-embed-tokens N\|none` | The per-build token cap (default none). |
| `--no-embed` | Build lexically and remove `dense.json` (the cache stays). |

Without these flags, `index2` follows the saved `dense.json` if there is one. It
prints progress per request and a summary on stderr; for example, with the offline
`hash` embedder and a cap, on a 265-file tree:

```text
arko_index: embedding with hash/hash-v1 (256 dimensions), at most 300000 tokens
arko_index: embedded 256/7018 texts, 57437 tokens, 1 requests
...
indexed 7018 chunks, 16000 terms, 6 runs
arko_index: embeddings: hash/hash-v1 (256 dimensions); 7018 chunks, 7018 distinct texts: 0 from the cache, 1916 embedded now in 8 requests, 299997 tokens
arko_index: stopped at the token cap (300000): 5102 texts (784381 counted tokens) not embedded; their chunks are found by lexical search only. Rerun with a higher --max-embed-tokens (or none) to finish; embedded texts come from the cache
arko_index: 1916 of 7018 chunks have vectors; the cache holds 1916 vectors for this model
```

A rerun with `--max-embed-tokens none` then reported `1916 from the cache, 5102
embedded now in 20 requests, 784381 tokens`, and one more run `7018 from the cache, 0
embedded now`.

| `search2` flag | Effect |
|---|---|
| `--mode lexical\|dense\|hybrid` | Default: `hybrid` when the index has embeddings, else `lexical`. |
| `--limit N`, `--candidates N` | Results (10) and hybrid candidates per side (50). |
| `--fusion rrf\|linear`, `--rrf-k K`, `--lexical-weight W`, `--dense-weight W` | Fusion (RRF, k 60, weights 1; linear weight 0.5). |
| `--path-prefix P`, `--language L` | Filters. |
| `--text` | One line per hit: rank, header, mode, score and ranks. |

```text
$ $CLI search2 --text --limit 2 ./index-state effective_max_input_tokens input limit
 1. arko_ai/src/client.rs:467-476 · impl LLMClient › fn max_input_tokens  [hybrid 0.0325; lexical #2, dense #1]
 2. arko_ai/src/input_limit.rs:10-27 · fn effective_max_input_tokens  [hybrid 0.0320; lexical #3, dense #2]
arko_index: query embedded with hash/hash-v1 (256 dimensions): 7 tokens
```

**Live check.** `examples/dense_live_check.rs` builds an index with Voyage
embeddings of the arko snapshot the retrieval evaluation was written against (by
default `~/Desktop/auto-research-forge/port-check/eval/arko-snapshot`), runs the
evaluation questions (`breadth-depth-arko.json`, `behaviour-arko.json`) in lexical,
dense and hybrid modes, and prints each question's rank, the scores and the tokens
used. `--dry-run` only counts the tokens (no key needed); a rerun reuses its state
and spends only query tokens. See the file's header for its flags.

```sh
cargo run --release -p arko_index --example dense_live_check -- --dry-run
VOYAGE_API_KEY=... cargo run --release -p arko_index --example dense_live_check
```

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

## Parsing

Parsing uses [mhr_tree_sitter](https://github.com/mesa-hills-research/mhr_tree_sitter)
(a Git dependency pinned to one commit in `Cargo.toml`), a pure, safe Rust port of
tree-sitter 0.25.10 and 54 grammars, made by
an AI agent swarm and differentially tested against the C library. Its libraries keep
tree-sitter's names (`tree_sitter`, `tree_sitter_rust`, ...). The grammars of the
original eight languages are the versions this crate used with the C library:
tree-sitter-rust 0.24.2, -python 0.25.0, -typescript 0.23.2 (TypeScript and TSX),
-javascript 0.25.0, -go 0.25.0, -java 0.23.5, -c 0.24.2 and -cpp 0.23.4. Their chunks
stayed byte-identical when the other grammars came: 21,924 files of five local corpora
(this repository, a research repository's sources, a GPUI workspace, the openai/codex
repository, and CPython and Go sources) gave the same output as before. No C compiler is
needed.

Before the switch, the crate used the C library and an optional GCC profile-guided
build of the grammars. The switch was checked on eight trees: a snapshot of this
repository, a research repository (17.6 GB of text, gitignored runs included), the
openai/codex repository, CPython and Go sources, a mixed research corpus, and
TypeScript, Python and Rust repositories. In total 237,312 files, 18.9 GB of text and
6.5 million chunks. Each tree's `chunk` output was byte-identical across the C build,
the profile-guided C build and the port, in three runs each at six threads, and also
at one thread for five of the trees.

<!-- TODO: speed and memory against the C library, measured on dedicated cores. -->

## Validation tools

```sh
cargo test -p arko_index --locked
python3 arko_index/tools/validate_cli.py target/release/examples/arko_index_cli
python3 arko_index/tools/audit_coverage.py CORPUS chunks.jsonl
python3 arko_index/scripts/language_check.py --cli CLI [--baseline OLD_CLI] CORPUS...
```

`validate_cli.py` drives the CLI end to end: repeated `index`, hidden, ignored and state
paths, best-first `search`, and `watch` through SIGKILL restarts, burst edits, SIGTERM
and a broken pipe. `audit_coverage.py` checks a `chunk` output against a corpus without
using the crate's code: every UTF-8 file covered byte for byte in order, unique IDs,
line numbers, languages and the size bounds. It reads every file under the corpus, so
use it on trees without `.git` or `CACHEDIR.TAG` directories. `language_check.py`
checks whole corpora language by language, and with `--baseline` compares the original
languages' output with an older build (see
[Checking whole corpora](../docs/arko_index-languages.md#checking-whole-corpora)).

## Trade-offs

* `watch` reads the whole tree on every poll (cached parsing, not incremental
  filesystem I/O); `watch2` is the incremental alternative on Linux.
* `search` loads the entire JSON index; `search2` reads only the query terms' postings.
* Dense search is opt-in and costs tokens to build and per query. Its vector store is
  exact: a search reads every vector (about 29 MB for 7,000 chunks at 1,024
  dimensions, a quarter of that with `int8`), which suits repositories, not corpora of
  millions of chunks; an approximate store can replace it behind `VectorStore`.
  Embedding holds about 100 bytes per chunk in memory (text hashes and the cache
  index); texts to embed are spooled to disk.
* C/C++ recovery is heuristic, not a C preprocessor. Tiny merged definitions can lose
  retrieval concentration.
* The first `watch2` delta of a large tree is one JSON line holding every chunk.

## Origin

arko_index was produced by an autonomous research run (as `ar_chunker`), then hardened
for repository scale and for the file-inclusion rules above. It replaced an earlier
arko_index that chunked with a language model, embedded with Voyage and searched a
SQLite database; the new crate cannot read that crate's databases.
