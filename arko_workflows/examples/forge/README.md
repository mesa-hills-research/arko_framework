# The forge

A workflow that builds workflows. Describe what you want in your own words, and the forge
designs and writes an `arko_workflows` binary crate that does it, checks that it builds, passes
its tests and starts, and hands it to you with a README and a design note.

```sh
cargo run -p arko_workflows --example forge -- \
    --output ../morning-digest \
    --use-case "Every morning, read the RSS feeds in feeds.txt, pick the five stories most
                relevant to Rust and AI agents, and write a short digest to digest.md.
                It runs unattended on a VM."
```

Without `--use-case` or `--use-case-file`, it asks for the use case at the terminal.

## How it works

- **The architect** (Claude Opus 5.5 at xhigh effort by default) owns the result. It reads the
  framework's guidance (the workflow-creation skill, the design guides, the reference map and the
  examples), designs the smallest workflow that fits, writes the crate, and runs `cargo build`,
  `cargo test` and `--help` as it goes. It writes `README.md` and `DESIGN.md` for the new
  workflow. Its conversation is handed over at `--handover-at` input tokens, so long builds keep
  going in a fresh conversation.
- **Explorers** (Claude Sonnet 5.5, on by default) answer the architect's questions about the
  framework with exact signatures and file references, through `run_workflow_agent`. Up to
  `--max-active-calls` (4) run at once. `--no-explorers` turns them off to use less.
- **The forge's checks**: after the architect finishes, the forge itself checks that
  `README.md` and `DESIGN.md` exist, and that `cargo build --all-targets`, `cargo test` and
  `cargo run -- --help` pass. A failure goes back to the architect, in the same conversation, up to
  `--max-repairs` (3) times.
- **Lessons**: the architect can record lessons with `record_lesson` (API pitfalls, patterns that
  worked). They're kept in `~/.arko/forge/lessons.md` and given to the architect at the start of
  every run, so later runs don't repeat earlier mistakes. `--no-lessons` turns this off.
- **Trial runs** (off by default): with `--trial-runs N`, the architect may run the new workflow for
  real, up to N times and `--trial-minutes` each, and fix what the output shows. Each run uses real
  model usage.

The new crate depends on this checkout by path, has its own `[workspace]`, starts from the
checkout's `Cargo.lock`, and builds into the checkout's `target` directory (`--target-dir`), so
the arko crates compile once.

## Involvement

How much the forge involves you while it builds (`--involvement`):

| Level | What happens |
|---|---|
| `clarify` (default) | Once the architect knows your use case and its options, it asks one batch of questions, each with a suggested default. Then it works alone, asking again only when it's blocked on a decision that's yours. |
| `none` | It never asks; it chooses what fits and lists every assumption in `DESIGN.md`. |

This is separate from how autonomous the **new** workflow is: a set-and-forget VM job, one that
asks questions at the start, or one you supervise throughout. That's part of your use case; say
which you want, or the architect chooses and explains it.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--use-case TEXT` / `--use-case-file PATH` | asked at the terminal | What the workflow should do. |
| `--output DIR` | required | The new crate's directory: new or empty. |
| `--name NAME` | the directory's name | The crate's name. |
| `--involvement none\|clarify` | `clarify` | See above. |
| `--backend claude-code\|api` | `claude-code` | Run the agents in the installed Claude Code CLI (plan usage), or through the Anthropic API (`ANTHROPIC_API_KEY`, billed per token). |
| `--architect-model`, `--architect-effort` | `claude-opus-5-5`, `xhigh` | The architect's Anthropic model and effort. |
| `--explorer-model`, `--explorer-effort` | `claude-sonnet-5-5`, `high` | The explorers'. |
| `--no-explorers` | off | No explorers. |
| `--max-active-calls N` | 4 | Explorers running at once. |
| `--trial-runs N`, `--trial-minutes M` | 0, 20 | Live trial runs allowed, and the longest each may take. |
| `--handover-at TOKENS` | 300000 | The architect's handover threshold. |
| `--max-repairs N` | 3 | Failed checks sent back to the architect. |
| `--lessons PATH`, `--no-lessons` | `~/.arko/forge/lessons.md` | The lessons file. |
| `--arko PATH` | this checkout | The arko checkout the crate depends on and the agents read. |
| `--target-dir PATH` | the checkout's `target` | The shared Cargo target directory. |
| `--state-dir PATH` | `~/.arko/forge/runs/<name>-<time>` | Checkpoints, the event log, trial logs and the report. |
| `--resume STATE_DIR` | | Continue an interrupted run with its saved settings. |
| `--budget DOLLARS` | none | Pause and stop at this spend (billed usage: the API backend, or Claude Code under an API key); with `--resume`, replaces the saved budget. |
| `--verbose` | off | Print every agent's tool calls and text. |

## What you get

- The crate in `--output`: the workflow's source, `README.md` (how to run it), `DESIGN.md` (why
  it's built the way it is), and `USE_CASE.md` (your request).
- In the state directory: `report.md` (the architect's summary, the checks, and usage per agent),
  `events.jsonl` (every event of the run), `trials/` (full logs of trial runs), and checkpoints.

## Resuming

If the forge stops (a crash, a closed terminal, a reboot), run it again with
`--resume <state dir>`; no other flags are needed, because the run's settings and the architect's
instructions are saved there when it starts. A finished architect task goes straight to the
checks. An interrupted one continues where it stopped: the running agents are saved as they run,
so the architect resumes its conversation (and any explorer it was waiting for resumes too),
told that a tool call it was running when the forge stopped may or may not have finished. If it
can't be recovered, a new architect run continues from the work already in the crate, which
keeps `DESIGN.md` current for exactly this. An explorer run that the architect didn't resume
itself is recovered after it. Usage limits and provider errors don't stop the forge: they're
waited out and retried.

With `--budget`, reaching the budget doesn't cancel anything: each agent pauses before its next
model request (a tool call it is running finishes first), and once every agent is paused the
forge saves them and exits with status 2, printing how to continue. Resume with a higher budget
and the agents carry on from their pauses:

```text
forge --resume <state dir> --budget <dollars>
```

Spending can go past the budget by the requests already in flight when it is reached.
