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

To change a workflow that exists (one the forge built, or any arko workflow crate), give its
directory with `--update` and describe the change as the use case:

```sh
cargo run -p arko_workflows --example forge -- \
    --update ../longcoder --use-case-file change.md
```

## Requirements

| Program | | What for |
|---|---|---|
| `cargo` | required | Building, testing and running the workflow crate (install Rust with [rustup](https://rustup.rs)). |
| `bash` | required on Linux and macOS | The shell the forge's `run_shell` and `trial_run` tools run commands in (on Windows, `cmd.exe`). |
| `claude`, signed in | required when an agent runs in Claude Code (the default `--backend claude-code` with Claude models) | The Claude Code CLI the agents run in. Run `claude` once to sign in. |
| `rg` (ripgrep) | optional | The fast code search the agents run in their shells; without it they use slower tools such as `grep`. `apt install ripgrep`, `brew install ripgrep` or `winget install BurntSushi.ripgrep.MSVC`. |

The forge checks these when it starts, and again on `--resume`: a missing required program stops
it before anything runs (exit status 1), naming the program and how to install it; a missing
optional one is a warning. Accounts and keys aren't part of this check: `--backend api` needs
`ANTHROPIC_API_KEY`; OpenAI models (in the deliberation presets) a ChatGPT login (Codex's
`auth.json`, or `OPENAI_OAUTH_ACCESS_TOKEN` with `OPENAI_OAUTH_ACCOUNT_ID`) or `OPENAI_API_KEY`;
Google models `GEMINI_API_KEY`.

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
  model usage. Trials run as separate processes, so their actual spend isn't visible to the forge:
  with a `--trial-budget` or `--budget`, each run reserves a cap in full, and the architect must
  pass the same cap to the workflow's own budget flag, which enforces it.

The new crate depends on this checkout by path, has its own `[workspace]`, starts from the
checkout's `Cargo.lock`, and builds into the checkout's `target` directory (`--target-dir`), so
the arko crates compile once.

With `--update DIR`, the forge works on the crate in `DIR` instead: the change is added to the end
of its `USE_CASE.md` (the original use case stays above it), and the architect reads the crate's
README, DESIGN.md, decision records and code before it designs the change. The crate keeps its own
`Cargo.lock`; the forge warns when its arko dependencies point somewhere other than `--arko`.

## What the agents may run

Version control and publishing are yours: the forge leaves commits, branches and the index to you,
and you review and commit the crate yourself. Every agent's shell (Claude Code's `Bash`, or the
forge's `run_shell` on the API backend) passes arko's command guard with these command rules
(`tools::command_rules`):

- **git** only reads: `git --version`, `status`, `log`, `diff`, `show`, `rev-parse`, `rev-list`,
  `merge-base`, `cat-file`, `ls-files`, `ls-tree`, `blame`, `describe`, `shortlog`, `grep`,
  `branch` (bare, `-a`, `-r`, `-v`, `-vv`, `-a -v`, `-a -vv`, `--list …` or `--show-current`:
  `git branch <name>` would create one), `tag` (bare, or `--list …`), `stash list`, `reflog`
  (bare, or `reflog show …`), `show-ref`, `for-each-ref`, `worktree list`, `submodule status`,
  `check-ignore`, `help`, `remote -v`, `config --get` and `config --list`. Any other git command
  is refused, and so are `--output` (which writes a file: read the output from the command
  instead) and `git grep -O`/`--open-files-in-pager` (which run a program). Global options other
  than `-C`, `--no-pager`, `-P` and `--no-optional-locks` are refused too.
- **gh** only reads: `gh --version`, `pr view`, `pr list`, `pr diff`, `pr checks`, `pr status`,
  `issue view`, `issue list`, `repo view`, `auth status`, `run list` and `run view`.
- **cargo** doesn't `publish`, `login`, `logout`, `owner` or `yank` (as its subcommand, after any
  `+toolchain` or one option such as `--locked`); everything else runs, `cargo run -- publish`
  and `cargo test login` included.
- **Spikes** may also use git fully inside the spikes' scratch folder (`deliberations/` in the
  state directory), to test claims about git in toy repositories.

The rules match the commands the shell will run, wherever they are in the line (`cd sub && git
commit`, `bash -c '…'`, `$(…)`, `xargs`), not mentions: `ls /usr/share/doc/git` and
`grep -rn git` run. The guard also refuses destructive commands on paths outside the crate, as it
does for any arko agent. A refused command reaches the agent with the reason, and the forge
prints it (`✗ Bash: Refused: …`, or as an error for `run_shell`). The architect, and each
deliberation agent with a shell, is told that a refusal is the policy working: not to work around
it, and to report it if it blocked them.

## Deliberation

With `--deliberate light` or `--deliberate full`, the decisions that are expensive to reverse once
work builds on them (one-way doors: the workflow's topology, its autonomy and recovery design, its
shared state, its model choices) aren't made in one turn of the architect. The architect frames
each one (the question, the constraints, the evidence so far, and the criteria a good answer
meets) with `request_deliberation` and ends its turn, and the forge runs it:

1. **Diverge.** Proposers draft designs independently, in separate conversations, on different
   models. Each states its assumptions, its risks, and what evidence would prove it wrong.
2. **Scrutinize.** A critic reviews each proposal. Every finding carries a severity, its
   evidence, and a realistic trigger: the circumstance in which the problem actually occurs.
   Critics are told what a useful finding is, and that style preferences and failures without a
   plausible cause aren't findings.
3. **Rebut.** Each proposer answers the findings on its design, in its own conversation: it
   accepts a finding and revises the design, accepts it in part (revising for that part and
   contesting the rest), or contests it as incorrect, not applicable, out of scope, or
   implausible. Each part is taken only in its own stage: once its design has been reviewed, a
   proposer revises it through its answer, not by submitting it again. With two rounds, each
   critic then maintains or withdraws what was contested.
4. **Test.** The contested findings that matter (critical or major, the most severe first) become
   spikes: a small experiment in a scratch crate that depends on the arko checkout (a snippet that
   compiles, the framework's offline tests). Spikes make no model calls.
5. **Converge.** A judge (or a panel, by majority, or the architect itself) rules on each
   contested finding, picks one design, and grafts on specific fixes from the others. Judges don't
   see which model wrote which proposal.
6. **Record.** The decision record goes to the crate's `decisions/` folder (the decision, the
   chosen design, the alternatives and why they lost, the findings upheld and dismissed, the spike
   results) and back to the architect, which builds on it and summarises it in DESIGN.md. If the
   checks or trial runs keep failing because of a recorded decision, the architect may reopen it
   once.

When the architect frames several decisions at once, they are deliberated one after another in
the order it framed them, so each can build on the records before it, and the architect continues
only once all are settled (when it is the judge, it rules on each in turn first). Everything else
is a two-way door, which the architect decides and moves on from. The ledger of
every stage (proposals, findings, answers, spikes and rulings) is `deliberations.json` in the state
directory. Each part is saved the moment its agent submits it, so a resumed forge continues a
deliberation where it stopped and never runs a finished stage again.

| | `light` | `full` |
|---|---|---|
| Proposers | 2: Opus 5.5 and GPT-6.1 Sol | 4: Opus 5.5, GPT-6.1 Sol, Fable 5.1, GPT-6 Astra |
| Critique rounds | 1 | 2 |
| Spikes per decision | 0 | up to 3 |
| Who rules | the architect | a judge on Fable 5.1 |
| Which decisions | the top-level architecture, as one decision | every one-way door, up to 4 |

Every model runs at xhigh effort. Each proposal's critic runs on the next proposer's model, so a
design is reviewed by a different model than the one that wrote it. OpenAI models run through the
ChatGPT login when there is one (Codex's `auth.json`, or `OPENAI_OAUTH_ACCESS_TOKEN` with
`OPENAI_OAUTH_ACCOUNT_ID`), else `OPENAI_API_KEY`; without either, the
presets use Opus 5.5 and Fable 5.1 only. Every setting can be changed with the flags below, and a
model is written `id` or `id:effort`.

A deliberation uses real model usage: in `full`, each decision is about a dozen agent runs. Compare
runs of the same use case with and without it before making it a habit: it earns its place when it
removes more rework than it costs.

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
| `--use-case TEXT` / `--use-case-file PATH` | asked at the terminal | What the workflow should do (with `--update`, the change to make). |
| `--output DIR` | required | The new crate's directory: new or empty. |
| `--update DIR` | | Change the existing crate in `DIR` instead of building a new one. |
| `--name NAME` | the directory's name | The crate's name. |
| `--involvement none\|clarify` | `clarify` | See above. |
| `--backend claude-code\|api` | `claude-code` | Run the agents in the installed Claude Code CLI (plan usage), or through the Anthropic API (`ANTHROPIC_API_KEY`, billed per token). |
| `--architect-model`, `--architect-effort` | `claude-opus-5-5`, `xhigh` | The architect's model and effort. |
| `--explorer-model`, `--explorer-effort` | `claude-sonnet-5-5`, `high` | The explorers'. |
| `--no-explorers` | off | No explorers. |
| `--max-active-calls N` | 4 | Explorers running at once. |
| `--trial-runs N`, `--trial-minutes M` | 0, 20 | Live trial runs allowed, and the longest each may take. |
| `--trial-budget DOLLARS` | none | The most all trial runs together may spend. Each run reserves its cap in full (the workflow's own budget flag enforces it); without this, trials draw on `--budget`, and their reservations count toward it. |
| `--handover-at TOKENS` | 300000 | The architect's handover threshold. |
| `--max-repairs N` | 3 | Failed checks sent back to the architect. |
| `--lessons PATH`, `--no-lessons` | `~/.arko/forge/lessons.md` | The lessons file. |
| `--arko PATH` | this checkout | The arko checkout the crate depends on and the agents read. |
| `--target-dir PATH` | the checkout's `target` | The shared Cargo target directory. |
| `--state-dir PATH` | `~/.arko/forge/runs/<name>-<time>` | Checkpoints, the event log, trial logs and the report. |
| `--resume STATE_DIR` | | Continue an interrupted run with its saved settings. |
| `--budget DOLLARS` | none | Pause and stop at this spend (billed usage: the API backend, or Claude Code with `--metered`); with `--resume`, replaces the saved budget. |
| `--plan-budget-5h POINTS` | none | Pause and stop once the run has used this many percentage points of a subscription plan's five-hour usage window, counted from its level when the run started (Claude Code, ChatGPT sign-in). Each plan's window is counted on its own; other use of the same plan during the run counts too. With `--resume`, replaces the saved limit. |
| `--plan-budget-7d POINTS` | none | The same for each plan's seven-day window. |
| `--claude-code-budget-5h`, `--claude-code-budget-7d POINTS` | none | The Claude Code plan's own limits, replacing the two above for that plan. |
| `--chatgpt-budget-5h`, `--chatgpt-budget-7d POINTS` | none | The ChatGPT plan's own limits. |
| `--metered` | off | The Claude Code CLI is billed per token (an API key, cloud credits, a host-managed provider): count its list-price estimate as spend, so costs and `--budget` apply. |
| `--deliberate light\|full` | off | Deliberate on one-way-door decisions (see [Deliberation](#deliberation)); the flags below override the preset. |
| `--proposers N` | 2 / 4 | Proposers per decision. |
| `--proposer-models MODELS` | see the table | Comma-separated, used in turn. |
| `--critic-models MODELS` | the next proposer's | The critics', in proposal order. |
| `--critique-rounds 1\|2` | 1 / 2 | 2 adds the critics' replies to the rebuttals. |
| `--spikes N` | 0 / 3 | The most spikes per decision. |
| `--spike-model MODEL` | the architect's | The model that runs spikes. |
| `--judges lead\|MODELS` | `lead` / `claude-fable-5-1` | `lead`: the architect rules; several models vote as a panel. |
| `--gate top\|one-way` | `top` / `one-way` | Only the top-level architecture, or every one-way door. |
| `--max-decisions N` | 1 / 4 | Decisions deliberated per run (reopening one doesn't count). |
| `--verbose` | off | Print every agent's tool calls and text. |

## What you get

- The crate in `--output`: the workflow's source, `README.md` (how to run it), `DESIGN.md` (why
  it's built the way it is), `USE_CASE.md` (your request), and with deliberation, the decision
  records in `decisions/`.
- In the state directory: `report.md` (the architect's summary, the decisions deliberated, the
  checks, and usage per agent), `events.jsonl` (every event of the run), `trials/` (full logs of
  trial runs), `deliberations.json` and `deliberations/` (every stage of each deliberation, and the
  spikes' scratch crates), and checkpoints.

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
waited out and retried (a Claude Code session that fails, too: it is resumed after a wait).

An agent whose model request needs a new credential or a configuration change (an expired or
invalidated login, Claude Code not signed in or its login expired, a Claude Code CLI that can't
start, a bad key) gets one fresh start per process:
the architect is started again, continuing from the crate; the architect may ask an explorer
again; a deliberation agent's part starts again. If it fails the same way again, the forge saves
and exits with status 3, naming the agent and its model. Sign in again (`claude` for Claude Code,
`codex login` for ChatGPT) and resume with `forge --resume <state dir>`; the agent starts afresh.

With `--budget`, reaching the budget doesn't cancel anything: each agent pauses before its next
model request (a tool call it is running finishes first), and once every agent is paused the
forge saves them and exits with status 2, printing how to continue. Resume with a higher budget
and the agents carry on from their pauses:

```text
forge --resume <state dir> --budget <dollars>
```

The plan budgets work the same way for subscription plans, in percentage points of their usage
windows used since the run started. A run with agents on both Claude Code and the ChatGPT login
counts each plan's windows apart, and any limit reached stops it: for example
`--claude-code-budget-5h 85 --chatgpt-budget-7d 25 --budget 35`. The dollar budget counts only
what is billed per token, so subscription usage never reaches it. The stop message names the plan,
the window and the flag to raise, and the run continues with a higher limit:

```text
forge --resume <state dir> --claude-code-budget-5h <points>
```

Spending can go past a budget by the requests already in flight when it is reached.
