# The arko_index agent benchmark

Does giving a coding agent arko_index make it better or cheaper at understanding code than a
terminal alone? This example answers it by measurement: agents answer code-understanding
questions about real repositories, pinned to tags, in two (or three) arms that differ only in
their tools, and a grader that doesn't know the arm scores every answer against a verified key.

| Arm | The solver has |
|---|---|
| A | a terminal (a `shell` tool in a read-only checkout; `rg`, `grep`, `sed`, `git grep`, ...) |
| B | the same terminal, plus the arko_index CLI as `arko_index`, with an index of the checkout built beforehand |
| C (opt-in) | B, plus dense search over the same chunks (`arko_index_dense`, Voyage embeddings) |
| variants (opt-in) | B's tools over an index built, or searched, with other arko_index options ([Variant arms](#variant-arms)) |

Every agent (the solvers of every arm, the grader, the question writer and the verifier) runs
in an arko workflow on the same model and settings: `gpt-6-astra` at `xhigh` effort, Fast mode,
over a ChatGPT login (Codex's `auth.json`). The arms' prompts are the same except for the
paragraph that describes the extra tools, and their shells are the same except for the programs
first on `PATH` (arm A gets stubs that answer "command not found" for every arko_index program).

## The tasks

Four kinds of question, each with an answer key of required items:

| Kind | Asks | Key |
|---|---|---|
| definition | where something is defined (by name, or by what it does) | its location |
| callers | who calls a function or method (in a stated scope) | every call site |
| behaviour | how a feature works | 3 to 6 facts, and the functions or files that implement it |
| impact | what a change would affect | the places affected, and why |

Items are locations (`path:start-end`), files, or facts. Locations and files are also checked
deterministically against the answer's `LOCATIONS:` (3 lines of slack); some keys list known
wrong answers (a same-named variant of another enum, say), and citing one first fails the check.

The task set (`OUT/tasks.json`) is assembled from:

- **Seed tasks** (built in, `seed-tasks.json`): the earlier arko question sets
  (`behaviour-arko.json`, and the depth and breadth sets of `breadth-depth-arko.json`)
  converted to tasks, every key rechecked by hand at the pinned arko commit: 15 definition and
  4 callers tasks.
- **Generated tasks**: a writer agent writes questions per repository and kind, in runs of at most
  four, each run pointed at other directories. Its `submit_question` tool rejects keys whose
  locations aren't in the checkout, duplicates, and questions that name the answer's file. A
  separate verifier agent then answers each question blind, gets the key, rechecks every
  difference in the code, and returns `agree`, `disagree` or `ambiguous`. Only questions it
  agrees with are kept, up to `--per-kind` per repository and kind (the seeds count toward it).
  `generate` keeps writing while the verified questions and those still being verified (at the
  expected yield, one in `--oversample`) fall short, up to enough writer runs for twice the
  questions wanted per repository and kind. With the defaults (9 per kind) that is up to
  258 tasks: 9 per repository and kind, and all 15 of arko's seed definition questions (seeds
  are always kept).
- The writer and the verifier have arm A's tools, a terminal (`--writer-tools index` gives them
  arko_index too), so the questions aren't shaped by the tool under test: a writer that found
  its code with arko_index would tend to describe it in the words that find it there.

The repositories (`repos.json`; each checked out at its tag, depth 1, and made read-only):

| Id | Repository | Tag | Language | Files | MB |
|---|---|---|---|---|---|
| arko | mesa-hills-research/arko_framework (private: `--source arko=PATH`) | commit 8f8569a | Rust | 294 | 7.2 |
| redis | redis/redis | 7.2.5 | C | 1,583 | 15.5 |
| leveldb | google/leveldb | 1.23 | C++ | 152 | 1.0 |
| caddy | caddyserver/caddy | v2.8.4 | Go | 471 | 2.7 |
| jsoup | jhy/jsoup | jsoup-1.18.1 | Java | 245 | 2.3 |
| scrapy | scrapy/scrapy | 2.11.2 | Python | 532 | 4.2 |
| vite | vitejs/vite | v5.4.10 | TypeScript | 2,125 | 13.3 |

## Requirements

Linux (arm C needs a Unix host), `git`, `bash`, and network access for `prepare`. `rg`
(ripgrep) is recommended: both arms are told it is there when it is. A ChatGPT login: run
`codex login` (with `CODEX_HOME` set if you keep several), and pass that home with
`--codex-home` (default `$CODEX_HOME`, else `~/.codex`). Arm C needs `VOYAGE_API_KEY` in the
benchmark's environment (never in the agents' shells: their commands get no variable that looks
like a credential). Don't run it as root: the read-only checkouts are what keep concurrent agents
from changing each other's code.

With two or more ChatGPT accounts, repeat `--codex-home` (one login per home, first preferred):
a request an account's usage limit refuses goes to the next account, and requests move back
once the first is usable again (arko's `ChatGptAccounts`).

## Running it

Build the benchmark and the arko_index CLI into the same target directory (the benchmark finds
`arko_index_cli` next to itself; or pass `--index-cli` to `prepare`, `generate` and `run`):

```sh
cargo build --release -p arko_index --example arko_index_cli
cargo build --release -p arko_workflows --example index_benchmark
B=target/release/examples/index_benchmark
OUT=~/index-benchmark
```

1. Fetch the repositories and build their indexes (untimed; the build times are recorded and
   reported):

   ```sh
   $B prepare --out $OUT --source arko=/path/to/arko_framework
   # Arm C too: VOYAGE_API_KEY=... $B prepare --out $OUT --source arko=... --dense
   ```

2. Generate and verify the questions, and assemble the task set:

   ```sh
   $B generate --out $OUT --codex-home ~/.codex --concurrency 24
   $B validate --out $OUT                  # counts, and every key location checked again
   ```

   `generate` is resumable: run it again after a stop, or with a higher `--per-kind` or
   `--oversample`, and it writes and verifies only what is missing. It says which repositories
   and kinds fell short. `assemble --per-kind N` rebuilds `tasks.json` from what was verified,
   without agents.

3. Run the benchmark (start with a pilot), then read the report:

   ```sh
   $B run --out $OUT --codex-home ~/.codex --name pilot --limit 20
   $B run --out $OUT --codex-home ~/.codex --name main --concurrency 24
   $B report --out $OUT --name main          # also written at the end of every run
   ```

   `--arms A,B,C` adds arm C; `--samples 3` runs each task three times per arm (the intervals
   then resample tasks with all their samples). Every agent run counts toward the plan's usage
   limits; when one is reached, the runs wait for the reset (arko's automatic recovery) and the
   wait doesn't count toward their time limit.

Stopping and resuming: Ctrl-C, `SIGTERM`, or `touch $OUT/runs/NAME/STOP` start no new agent
runs and let the running ones finish; a second signal cancels them (they run again on resume,
without using up an attempt); a third exits. Running the same command again resumes: every
finished solver run and grading is recorded as it ends (`attempts.jsonl`, `grades.jsonl`), and
only the rest is done. A lost login (or a billing or permission error) stops the whole run, or
`generate`, and the agent runs it failed are run again on resume without using up an attempt;
other runs that end outside the agent's control are run again up to `--max-attempts` (3). A
resumed run must keep its model, shell settings and limits; it may add arms, samples and tasks.
One process at a time works on a run, on `generate`, or on `prepare` (a `LOCK` file with its
process id). `prepare` refuses to start while a run or `generate` is working, since it replaces
checkouts and indexes, and they refuse to start while it works.

## Metrics

Per task, arm and sample (`attempts.jsonl`, `attempts.csv`): the outcome (answered, failed,
over a limit), the answer, model calls, tool and shell calls (and how many ran arko_index:
commands whose program is `arko_index`, not ones that merely mention it), input tokens (and the
cached part), output tokens (and reasoning), the largest context, wall time and working time
(wall time less waits for usage limits and retries; the report's time is working time), the API
list price of the tokens (ChatGPT OAuth bills nothing per call; this is the cost the same tokens
would have at API prices, in Fast mode), and the transcript.

`report.md` (and `report.json`) gives, per arm and per kind, and per repository:

- **pass**: the grader's verdict `correct`, with a Wilson 95% interval (a bootstrap over tasks
  with `--samples` above 1); a run that didn't answer fails;
- **items**: the share of the key's items the grader found met, with a bootstrap 95% interval
  over tasks;
- **check**: the deterministic check of locations and files (a cited range more than three times
  as long as the key's, plus 30 lines, meets nothing, so citing whole files doesn't pay);
- tokens, calls, wall time and cost, with intervals, and cost per pass;
- **paired differences** between arms on the same tasks (B − A, ...), with bootstrap intervals,
  and how many tasks each arm did better on;
- the **cost/quality frontier**: per kind, the arms no other arm beats on both cost and item
  score (`frontier.csv` for plotting);
- how often the grader and the deterministic check agree, and what was left ungraded;
- runs whose commands named the benchmark's own files outside their checkout (`tasks.json`,
  the records, transcripts, other directories of `--out`): the solver's shell can reach them, and
  the prompt tells every arm to stay in its checkout; check those transcripts by hand.

The grader sees the question, the key and the answer, never the arm: lines of the answer that
describe a search (`$ARKO_INDEX_STATE`, `arko_index_dense`, or a search command such as
`arko_index search2`, `rg -n`, `grep -r` or `git grep` beside words like "I ran" or "searched")
are removed first, in every arm, and gradings run interleaved with the solvers. It has a shell in the checkout to
check claims the key doesn't settle (`--grader-without-shell` to judge from the key alone).

## Limits

| Flag | Default | |
|---|---|---|
| `--concurrency` | 24 | Agent runs at once (solvers and graders). |
| `--max-output-bytes` | 16,384 | Output of one shell command returned to the agent (head and tail). |
| `--command-timeout` | 120 s | Per shell command. |
| `--max-shell-calls` | 80 | After that the agent is asked to answer; ten more and the run is stopped. |
| `--max-input-tokens` | 8,000,000 | All calls, cached included; over it the run is stopped and fails. |
| `--attempt-timeout-minutes` | 60 | Working time (waits for usage limits excluded); 0 for none. |

Runs stopped by a limit count as failures (scored 0) in every arm alike, so a limit that binds
often mixes cost into correctness: check the "over limit" counts of the pilot. Input grows with
every call (each request carries the whole conversation), so 8M tokens is roughly 45 to 65 model
calls of a long investigation (at 3,000 to 8,000 tokens of growth per call).

## Dry run

`--mock` replaces the model with a local server that plays every role from a script (no model
calls): solvers run `--mock-steps` (12) shell commands, searching and reading in turn (arm A
with `rg`, arms B and C with `arko_index search2` and `arko_index_dense`), then answer; the
writer submits definition questions from a `grep` (or `arko_index outline`) of one file; the
verifier and grader return verdicts. The
commands run for real, so a dry run exercises the checkouts, indexes, the arms' tools, grading,
resume and the report, and measures prompt and tool-output sizes. Its token counts are
estimates: 4 bytes per token, each call's earlier reasoning carried back in its input, and
`--mock-reasoning-tokens` (1,500) of reasoning per call. `prepare --dense --mock-embeddings`
builds arm C's index with an offline hashing embedder.

```sh
$B prepare --out $OUT/dry --source arko=/path/to/arko_framework --dense --mock-embeddings
$B generate --out $OUT/dry --mock --per-kind 4
$B run --out $OUT/dry --mock --name dry --arms A,B,C
```

## Variant arms

Variant arms compare arko_index options on the same tasks with the same solvers. Each one
has arm B's tools over its own index, built with other `index2` flags, and can add
`search2` flags to every search. A variants file defines them, and `variant-arms.json` here
defines the first set:

| Arm | What changes |
|---|---|
| `lex` | Nothing: the default lexical index, built by this CLI (the baseline for the others) |
| `lex-scope` | Lexical search also indexes each chunk's scope (`--lexical-scope`) |
| `lex-scope-desc` | As `lex-scope`, with Markdown scopes rooted at the front matter's title and description |
| `hyb-src`, `hyb-path`, `hyb-scope` | Hybrid search, the embedding seeing the source, path and source, or path, scope and source |
| `hyb-scope-desc` | As `hyb-scope`, with Markdown scopes rooted at title and description |

A variant's fields:

| Field | |
|---|---|
| `name` | The arm's name in `--arms`, records and reports: letters, digits, `-` and `_` |
| `description` | For the report |
| `index` | `index2` flags, such as `["--lexical-scope"]` or `["--embed", "voyage-code-4", "--embed-text", "source"]` |
| `search` | `search2` flags put before the solver's own, such as `["--text"]` or `["--mode", "lexical"]` |
| `prompt` | The tools paragraph: `index` (arm B's) or `hybrid` (says search2 also ranks by meaning). Default: `hybrid` when the index embeds |
| `tools_text` | A tools paragraph of your own, for search flags whose output the standard ones don't describe |

```sh
# Builds every variant's index of every prepared repository (beside a running `generate`)
VOYAGE_API_KEY=... $B prepare-variants --out $OUT --variants variant-arms.json
$B prepare-variants --out $OUT --variants variant-arms.json --names lex,lex-scope --only redis
# Runs them with the built-in arms
VOYAGE_API_KEY=... $B run --out $OUT --codex-home ~/.codex --name variants --arms B,lex,lex-scope,hyb-scope
```

`prepare-variants` builds `indexes/<id>@<name>/` with `arko_index_cli index2 FLAGS`, and
rebuilds one only when its flags, the CLI or the commit changed (`--force` rebuilds all).
The index directory stays across rebuilds, so its embedding cache keeps the vectors
already paid for. Variants that embed need `VOYAGE_API_KEY` in the benchmark's environment
for `prepare-variants` and for `run`. In their shells, `arko_index search2` goes through
a service in the benchmark process that runs the CLI with the key, the way arm C's
`arko_index_dense` does. `--max-embed-tokens 0` in a variant's flags counts the tokens a
build would embed and embeds nothing.

A resumed run keeps each variant's definition and indexes, and may add variants. Later
variants, such as chunk sizes or breadcrumbs inside large definitions, are new `index2`
flags in a variants file: eight arms make a 2×2×2 factorial.

Dry run, with the offline hashing embedder in place of Voyage:

```sh
$B prepare-variants --out $OUT/dry --variants variant-arms.json --mock-embeddings
$B run --out $OUT/dry --mock --name dry-variants --arms A,B,lex,lex-scope,hyb-src,hyb-scope
```

## Layout of `--out`

```text
repos/<id>/               read-only checkouts
indexes/<id>/             arko_index index2 state (and <id>.dense/ for arm C)
indexes/<id>@<variant>/   a variant arm's index
prepare.json              commits, sizes, index build times, the CLI's hash
variants.json             the variants' definitions, index flags, builds and CLI hashes
bin/arm-<arm>/            the programs first on each arm's PATH
tasks/                    candidates, verifications and writer runs (JSON lines), transcripts
tasks.json                the task set
runs/<name>/              manifest.json, attempts.jsonl, grades.jsonl, transcripts/,
                          report.md, report.json, attempts.csv, frontier.csv, progress.json
scratch/                  each agent run's TMPDIR, removed when it ends
```

`runs/<name>/`, `tasks/`, `prepare/` and `variants/` hold a `LOCK` file while a process works on them.
