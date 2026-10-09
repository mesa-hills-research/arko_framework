# arko_index verification

How arko_index's output is checked, and the measurements behind the
[crate README](../arko_index/README.md): the tests and validation tools, the checks of the
switch to the pure-Rust parser, and the retrieval evaluation of lexical, dense and hybrid
search.

## Tests and tools

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
[Checking whole corpora](arko_index-languages.md#checking-whole-corpora)). The tests of
each language are described in [arko_index languages](arko_index-languages.md#checking).

## The pure-Rust parser

arko_index parses with [Mesquite](https://github.com/mesa-hills-research/mesquite),
a pure, safe Rust port of tree-sitter 0.25.10 and 54 grammars (see
[Parsing](arko_index-reference.md#parsing)).

Before the switch, the crate used the C library and an optional GCC profile-guided
build of the grammars. The switch was checked on eight trees: a snapshot of this
repository, a research repository (17.6 GB of text, gitignored runs included), the
openai/codex repository, CPython and Go sources, a mixed research corpus, and
TypeScript, Python and Rust repositories. In total 237,312 files, 18.9 GB of text and
6.5 million chunks. Each tree's `chunk` output was byte-identical across the C build,
the profile-guided C build and the port, in three runs each at six threads, and also
at one thread for five of the trees.

The original nine languages' chunks also stayed byte-identical when the other grammars
came: 21,924 files of five local corpora (this repository, a research repository's
sources, a GPUI workspace, the openai/codex repository, and CPython and Go sources) gave
the same output as before.

## Retrieval evaluation

`examples/dense_live_check.rs` builds an index with Voyage embeddings of the arko
snapshot the retrieval evaluation was written against (by default
`~/Desktop/auto-research-forge/port-check/eval/arko-snapshot`), runs the evaluation
questions (`breadth-depth-arko.json`, `behaviour-arko.json`) in lexical, dense and hybrid
modes, and prints each question's rank, the scores and the tokens used. `--dry-run` only
counts the tokens (no key needed), and a rerun reuses its state and spends only query
tokens. See the file's header for its flags.

```sh
cargo run --release -p arko_index --example dense_live_check -- --dry-run
VOYAGE_API_KEY=... cargo run --release -p arko_index --example dense_live_check
```

The 19 questions come in three kinds. Depth questions (9) ask where a given enum variant
is defined. Breadth questions (4) ask for a definition and its callers (7 callers in
all). Behaviour questions (6) describe in words what some code does. A question counts as
found when the expected chunk is in the top ten results.

The 2026-10-07 run: voyage-code-4 at 1,024 dimensions with the default embedded text
(path, scope and source), 7,018 chunks, 1,790,527 tokens to build the index and 161 for
the queries, reciprocal rank fusion with k = 60 and 50 candidates per side.

| Mode | Depth (9) | Breadth: definitions (4) | Breadth: callers (7) | Behaviour (6) |
|---|---|---|---|---|
| Lexical (BM25) | 4 | 4 | 5 | 4 |
| Dense | 9 | 4 | 2 | 4 |
| Hybrid | 9 | 4 | 3 | 4 |

An earlier run (2026-10-04), before context headers, embedded the path and the source.
It found the same with lexical search (4, 4, 5 and 4), and with dense search 9, 4, 2 and
5: one behaviour question more than the 2026-10-07 run.
