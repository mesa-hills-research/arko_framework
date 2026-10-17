# arko_index

arko_index splits source repositories into chunks along their syntax, searches them with
BM25, and keeps the chunks current with durable watchers that turn file changes into chunk
deltas. Code, prose, markup and data in 53 languages are parsed with tree-sitter and chunked
along their structure, and other UTF-8 text is split by a strictly bounded fallback. Every
chunk carries a context header with its enclosing definitions (`impl Display for Point › fn fmt`)
or, in Markdown, the document's title and heading chain. Chunking, lexical search and
watching run entirely on your machine.

Dense (embedding) and hybrid search are opt-in. They embed through a `dense::Embedder` the
caller supplies (the CLI uses Voyage through arko_ai) or with a code-embedding model run on
the CPU ([local models](#local-models)), and the library itself never makes a network call.
Lexical search works as soon as a build has chunked the tree, and dense results fill in while
the chunks are embedded, however long a slow model takes. The vectors
live in [Caprock](https://github.com/mesa-hills-research/caprock), a pure-Rust store
updated in place as they arrive. A search scans int8 codes held in memory and rescores the
best candidates at full precision. On 199,999 code chunks it took 13.5 ms per query and
returned the exact top ten
([measurements](../docs/arko_index-reference.md#vector-store-measurements)).

It is built for coding agents and the tools around them. A result names an exact byte and
line range of the source, the same tree and options always give the same chunks and IDs,
and a watcher reports each change to the tree as chunk deltas.

## Search quality

On 19 retrieval questions about a snapshot of this repository, the expected code was in the
top ten results this often (dense with voyage-code-4):

| Mode | Depth (9) | Breadth, definitions (4) | Breadth, callers (7) | Behaviour (6) |
|---|---|---|---|---|
| Lexical (BM25) | 4 | 4 | 5 | 4 |
| Dense | 9 | 4 | 2 | 4 |
| Hybrid | 9 | 4 | 3 | 4 |

Dense search finds code that uses other words than the question, and lexical search finds
callers by name. The [retrieval evaluation](../docs/arko_index-evaluation.md#retrieval-evaluation)
describes the questions and the method.

## Install

The crate isn't on crates.io yet, so add it from this repository:

```toml
[dependencies]
arko_index = { git = "https://github.com/mesa-hills-research/arko_framework" }
```

The library needs no C compiler, because it parses with
[Mesquite](https://github.com/mesa-hills-research/mesquite), a pure-Rust port of
tree-sitter, and stores vectors with Caprock, also pure Rust. Its examples use arko_ai
for Voyage, a dev-dependency whose dependencies need one, so building the tests and examples
does too. Linux is the primary platform. macOS has
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
$CLI status ./index-state                  # embedding goes on in the background: progress
$CLI index2 ./repo ./index-state           # later builds reuse the saved model and cache
$CLI search2 --text ./index-state 'where is the input limit applied'   # hybrid
$CLI search2 --mode lexical --language rust ./index-state 'RetryPolicy'
$CLI search2 --path-prefix src/net ./index-state 'retry'   # one folder (whole path components)
# Dense search with a local model: no key, nothing billed (build with --features local-embed)
$CLI index2 --embed local ./repo ./index-state
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

A chunk's `scope` names where it sits, as the chain of definitions that enclose it, from the
syntax tree, or in Markdown the document's title and heading chain. A search result shows it
as its header, `path:start_line-end_line · scope`:

```text
src/geometry.rs:27-34 · mod geometry › impl fmt::Display for Point › fn fmt
tools/retry.py:22-32 · class RetryPolicy › async def run
src/pool.ts:5-17 · namespace Storage
include/router.h:18-26 · namespace net › namespace http › class Router › function find
docs/guide.md:14-21 · Guide › ## Install › ### Linux
```

Headers are metadata and never change chunk boundaries or IDs. Each language's labels are
in [arko_index languages](../docs/arko_index-languages.md), and the limits and formats in
[Context headers](../docs/arko_index-reference.md#context-headers).

## Local models

With the `local-embed` feature, `dense::local::LocalEmbedder` runs a code-embedding model on
the CPU, in pure Rust, so dense search needs no key and costs nothing per token.
`--embed local` picks nomic-ai/CodeRankEmbed (137M parameters, MIT) and runs most of its
arithmetic in int8: about 1.4 times as fast as f32 (`local:f32`), with vectors within a
cosine similarity of 0.998 of the f32 ones. The first build downloads the pinned
files (548 MB) into the Hugging Face cache. On the retrieval questions above it finds what
voyage-code-4 finds. On 6 cores of a laptop's i9-13980HX it embeds about 7.6 chunks a
second, so the 7,521 chunks of this repository's evaluation snapshot took about 16 minutes.
That happens once, because the cache keeps the vectors, and lexical search works while they
fill in.
With `local-embed-cuda` (NVIDIA, through CUDA) or `local-embed-gpu` (Vulkan, Metal or
DirectX 12, through wgpu) the same models run on a GPU, in f32, f16 or bf16
(`--embed local:f16:gpu`). [arko_index local embeddings](../docs/arko_index-local-embeddings.md)
has the models, the GPUs, the measurements and the evaluation.

## Watchers

`watch` (any platform) and `watch2` (Linux, inotify) stream chunk deltas as JSON lines. A
delta is committed to disk before it is delivered and replays until it is acknowledged, so a
consumer that acknowledges only after its own durable processing sees every change at least
once across crashes. See [Watchers](../docs/arko_index-reference.md#watchers).

## Languages

Astro, Bash, C, C#, C++, CMake, CSS, CUDA, Dart, diffs and patches, DTD, Elixir, Elm,
embedded templates (ERB, EJS), Erlang, F#, Fortran, GLSL, Go, GraphQL, Haskell, HCL
(Terraform), HTML, Java, JavaScript, JSON, Julia, Kotlin, Lua, Make, Markdown and MDX,
Objective-C, OCaml, Odin, Perl, PHP, PowerShell, Protocol Buffers, Python, R, Ruby, Rust,
Scala, Solidity, SQL, Svelte, Swift, TOML, TypeScript and TSX, Verilog, XML, YAML and Zig.

Code splits into its definitions, prose into sections, markup into elements, data into
keys or tables, and diffs into file sections and hunks.
[arko_index languages](../docs/arko_index-languages.md) lists the extensions and file
names of each language, how ambiguous ones are settled (`.h`, `.m`, `.v`, `.pl`) and
every language's rules.

## Parsing

Parsing uses [Mesquite](https://github.com/mesa-hills-research/mesquite), a pure, safe
Rust port of tree-sitter 0.25.10 and 58 grammars, and its chunks are the same as those of
the C tree-sitter library, which the crate used before.

## Documentation

- The [arko_index reference](../docs/arko_index-reference.md) covers the API, the CLI, file
  selection, chunking, context headers, search, dense and hybrid search, the watchers, environment
  variables, defaults, on-disk state, platforms and trade-offs.
- [arko_index languages](../docs/arko_index-languages.md) says which grammar parses a file,
  gives the rules of each language, and describes Markdown front matter.
- [arko_index tests and evaluation](../docs/arko_index-evaluation.md) gives the test commands
  and validation tools, and the retrieval evaluation behind the table above.
- [arko_index local embeddings](../docs/arko_index-local-embeddings.md) describes the local
  models, how they run, their speed and their retrieval quality.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT), at your option.
