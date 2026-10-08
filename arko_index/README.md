# arko_index

arko_index splits source repositories into chunks along their syntax, searches them with
BM25, and keeps the chunks current with durable watchers that turn file changes into chunk
deltas. Code, prose, markup and data in 50 languages are parsed with tree-sitter and chunked
along their structure, and other UTF-8 text is split by a strictly bounded fallback. Every
chunk carries a context header: its enclosing definitions (`impl Display for Point › fn fmt`)
or, in Markdown, the document's title and heading chain. Chunking, lexical search and
watching run entirely on your machine.

Dense (embedding) and hybrid search are opt-in. They embed through a `dense::Embedder` the
caller supplies (the CLI uses Voyage through arko_ai), and the library itself never makes a
network call.

It is built for coding agents and the tools around them. A result names an exact byte and
line range of the source, the same tree and options always give the same chunks and IDs,
and a watcher reports each change to the tree as chunk deltas.

## Search quality

On 19 retrieval questions about a snapshot of this repository, the expected code was in the
top ten results this often (dense with voyage-code-4):

| Mode | Depth (9) | Breadth: definitions (4) | Breadth: callers (7) | Behaviour (6) |
|---|---|---|---|---|
| Lexical (BM25) | 4 | 4 | 5 | 4 |
| Dense | 9 | 4 | 2 | 4 |
| Hybrid | 9 | 4 | 3 | 4 |

Dense search finds code that uses other words than the question, and lexical search finds
callers by name. The [retrieval evaluation](../docs/arko_index-verification.md#retrieval-evaluation)
has the questions, the method and an earlier run.

## Install

arko_index is a crate of this workspace:

```toml
[dependencies]
arko_index = { path = "../arko_index" }
```

The library needs no C compiler: it parses with
[mhr_tree_sitter](https://github.com/mesa-hills-research/mhr_tree_sitter), a pure-Rust port
of tree-sitter. Its examples use arko_ai for Voyage, a dev-dependency whose dependencies need
one, so building the tests and examples does too. Linux is the primary platform. macOS has
everything but `watch2`, and Windows compiles (see [Platforms](../docs/arko_index-reference.md#platforms)).

## Usage

The CLI ships as the `arko_index_cli` example:

```sh
cargo build --release -p arko_index --example arko_index_cli
CLI=target/release/examples/arko_index_cli
$CLI chunk ./repo > chunks.jsonl
$CLI index2 ./repo ./index-state           # disk-resident BM25 index
$CLI search2 ./index-state 'exponential backoff retry'
$CLI outline ./repo                        # path:start-end, kind, enclosing symbols
$CLI watch2 ./repo ./watch-state           # JSONL durable delta stream (Linux)
# Choosing files (any command that walks a tree)
$CLI chunk --respect-ignore --exclude 'vendor/' --report skipped.json ./repo
# Dense and hybrid search (opt-in, spends Voyage tokens)
export VOYAGE_API_KEY=...
$CLI index2 --embed voyage-code-4 --max-embed-tokens 5000000 ./repo ./index-state
$CLI index2 ./repo ./index-state           # later builds reuse the saved model and cache
$CLI search2 --text ./index-state 'where is the input limit applied'   # hybrid
$CLI search2 --mode lexical --language rust ./index-state 'RetryPolicy'
```

In Rust:

```rust
use arko_index::bench_v1::{chunk_str, Options};
use std::path::Path;
let chunks = chunk_str(Path::new("sample.rs"), "fn hello() {}\n", &Options::default())?;
```

Every regular file under the root is indexed, hidden and gitignored files included, except
`.git`, directories tagged with `CACHEDIR.TAG` (Cargo's `target/`, for one) and the tool's
own state directories. `--respect-ignore` honours `.gitignore`, `.ignore` and
`.git/info/exclude`, `--exclude` and `--include` take gitignore patterns, and `--report FILE`
writes everything that was skipped and why. Files that can't be indexed (binary, not UTF-8,
unreadable) are skipped and reported, and the run goes on. The library takes the same rules
as `FileRules`. See [File selection](../docs/arko_index-reference.md#file-selection).

## Context headers

A chunk's `scope` names where it sits: the chain of definitions that enclose it, from the
syntax tree, or in Markdown the document's title and heading chain. A search result shows it
as its header, `path:start_line-end_line · scope`:

```text
src/geometry.rs:27-34 · mod geometry › impl fmt::Display for Point › fn fmt
tools/retry.py:22-32 · class RetryPolicy › async def run
src/pool.ts:5-17 · namespace Storage
include/router.h:18-26 · namespace net › namespace http › class Router › function find
docs/guide.md:14-21 · Guide › ## Install › ### Linux
```

Headers are metadata: they never change chunk boundaries or IDs. Each language's labels are
in [arko_index languages](../docs/arko_index-languages.md), and the limits and formats in
[Context headers](../docs/arko_index-reference.md#context-headers).

## Watchers

`watch` (any platform) and `watch2` (Linux, inotify) stream chunk deltas as JSON lines. A
delta is committed to disk before it is delivered and replays until it is acknowledged, so a
consumer that acknowledges only after its own durable processing sees every change at least
once across crashes. See [Watchers](../docs/arko_index-reference.md#watchers).

## Languages

Bash, C, C#, C++, CMake, CSS, CUDA, Dart, DTD, Elixir, Elm, embedded templates (ERB,
EJS), Erlang, F#, Fortran, GLSL, Go, Haskell, HCL (Terraform), HTML, Java, JavaScript,
JSON, Julia, Kotlin, Lua, Make, Markdown and MDX, Objective-C, OCaml, Odin, Perl, PHP,
PowerShell, Protocol Buffers, Python, R, Ruby, Rust, Scala, Solidity, SQL, Svelte,
Swift, TOML, TypeScript and TSX, Verilog, XML, YAML and Zig.

Code splits into its definitions, prose into sections, markup into elements and data into
keys or tables. [arko_index languages](../docs/arko_index-languages.md) lists the extensions
and file names of each language, how ambiguous ones are settled (`.h`, `.m`, `.v`, `.pl`)
and every language's rules.

## Parsing

Parsing uses [mhr_tree_sitter](https://github.com/mesa-hills-research/mhr_tree_sitter), a
pure, safe Rust port of tree-sitter 0.25.10 and 54 grammars. When this crate moved to it
from the C library, its `chunk` output stayed byte-identical on eight trees: 237,312 files,
18.9 GB of text and 6.5 million chunks. [Verification](../docs/arko_index-verification.md)
has the details.

<!-- TODO: speed and memory against the C library, measured on dedicated cores. -->

## Documentation

- [arko_index reference](../docs/arko_index-reference.md): the API, the CLI, file selection,
  chunking, context headers, search, dense and hybrid search, the watchers, environment
  variables, defaults, on-disk state, platforms and trade-offs.
- [arko_index languages](../docs/arko_index-languages.md): which grammar parses a file, the
  rules of each language, and Markdown front matter.
- [arko_index verification](../docs/arko_index-verification.md): the tests and validation
  tools, the checks of the parser switch, and the retrieval evaluation.

## Origin

arko_index was produced by an autonomous research run (as `ar_chunker`), then hardened
for repository scale and for the file-inclusion rules in the reference. It replaced an
earlier arko_index that chunked with a language model, embedded with Voyage and searched a
SQLite database. The new crate cannot read that crate's databases.
