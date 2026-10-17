# arko_index tests and evaluation

How to test arko_index, and the retrieval evaluation behind the search-quality table in the
[crate README](../arko_index/README.md).

## Tests and tools

```sh
cargo test -p arko_index --locked
python3 arko_index/tools/validate_cli.py target/release/examples/arko_index_cli
python3 arko_index/tools/audit_coverage.py CORPUS chunks.jsonl
python3 arko_index/scripts/language_check.py --cli CLI [--baseline OLD_CLI] CORPUS...
```

`validate_cli.py` drives the CLI end to end, with repeated `index`, hidden, ignored and state
paths, best-first `search`, and `watch` through SIGKILL restarts, burst edits, SIGTERM and a
broken pipe. `audit_coverage.py` checks a `chunk` output against a corpus without using the
crate's code. It checks that every UTF-8 file is covered byte for byte in order, and the
unique IDs, line numbers, languages and size bounds. It reads every file under the corpus, so
use it on trees without `.git` or `CACHEDIR.TAG` directories. `language_check.py` checks
whole corpora language by language, and with `--baseline` compares the original languages'
output with an older build (see
[Checking whole corpora](arko_index-languages.md#checking-whole-corpora)). The tests of each
language are described in [arko_index languages](arko_index-languages.md#checking).

## The pure-Rust parser

arko_index parses with [Mesquite](https://github.com/mesa-hills-research/mesquite), a pure,
safe Rust port of tree-sitter 0.25.10 and 58 grammars (see
[Parsing](arko_index-reference.md#parsing)). It replaced the C library, and the `chunk`
output stayed the same on eight source trees, 237,312 files in all.

## Retrieval evaluation

`examples/dense_live_check.rs` builds an index with Voyage embeddings of a corpus, runs
evaluation questions in lexical, dense and hybrid modes, and prints each question's rank, the
scores and the tokens used. The questions are two JSON files (`breadth-depth-arko.json` and
`behaviour-arko.json`) about a snapshot of this repository. Neither the snapshot nor the
question files are in the repository, so point `--corpus` and `--eval-dir` at your own.
`--dry-run` only counts the tokens (no key needed), and a rerun reuses its state and spends
only query tokens. The file's header lists every flag.

```sh
cargo run --release -p arko_index --example dense_live_check -- --corpus DIR --eval-dir DIR --dry-run
VOYAGE_API_KEY=... cargo run --release -p arko_index --example dense_live_check -- --corpus DIR --eval-dir DIR
```

The 19 questions come in three kinds. Depth questions (9) ask where a given enum variant
is defined. Breadth questions (4) ask for a definition and its callers (7 callers in
all). Behaviour questions (6) describe in words what some code does. A question counts as
found when the expected chunk is in the top ten results.

The run behind the README's table used voyage-code-4 at 1,024 dimensions with the default
embedded text (path, scope and source), on 7,018 chunks. Building the index took 1,790,527
tokens and the queries 161, with reciprocal rank fusion at k = 60 and 50 candidates per side.

| Mode | Depth (9) | Breadth, definitions (4) | Breadth, callers (7) | Behaviour (6) |
|---|---|---|---|---|
| Lexical (BM25) | 4 | 4 | 5 | 4 |
| Dense | 9 | 4 | 2 | 4 |
| Hybrid | 9 | 4 | 3 | 4 |

Embedding only the path and the source, without the context header, found the same with
lexical search, and with dense search one behaviour question more (9, 4, 2 and 5).
