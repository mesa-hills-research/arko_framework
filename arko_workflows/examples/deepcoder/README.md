# DeepCoder example

A sequential coding workflow built from `arko_workflows` primitives. An orchestrator
splits a specification into small coder tasks, every coder completion is checked
by verifier agents, and a final verifier gates the end of the run.

```text
orchestrator
  -> run_workflow_agent(researcher)   (optional, bounded repository questions)
  -> run_workflow_agent(coder)
       -> correctness verifier
       -> efficiency verifier (minor repairs only)
  -> repeat for each dependency-ordered slice
  -> final verifier
  -> complete
```

The orchestrator runs slices one at a time. Each accepted slice becomes the
verified base for the next, and small slices keep individual agent tasks
manageable. The example uses no graph or parallel scheduler.

Each verification gate starts with a fresh, comprehensive review. When a gate
requests a repair, the same verifier conversation continues with its previous
verdict and a bounded summary of the tracked file and memory changes, so it can
reuse unchanged evidence and run focused checks. The final gate runs
automatically, because workflow-service target restrictions allow the orchestrator to
call only the coder and the researcher, never a verifier. Verifiers may make several
read-only calls in one response, which run together. `run_shell` isn't parallel-safe, so
it runs only after the calls before it have finished.

## Run

The example uses OpenAI models only. Credentials are tried in this order:

1. ChatGPT OAuth environment variables (the refresh token is optional):

   ```bash
   export OPENAI_OAUTH_ACCESS_TOKEN=...
   export OPENAI_OAUTH_REFRESH_TOKEN=...
   export OPENAI_OAUTH_ACCOUNT_ID=...
   ```

2. A Codex login at `$CODEX_HOME/auth.json` or `~/.codex/auth.json`, shared with
   Codex. Every token refresh reads the file again first, adopting tokens Codex
   refreshed in the meantime, and rotated tokens are saved to it at once (see
   [ChatGPT OAuth credentials](../../../docs/arko_ai-guide.md#chatgpt-oauth-credentials)).
3. An OpenAI API key:

   ```bash
   export OPENAI_API_KEY=...
   ```

Then run:

```bash
cargo run -p arko_workflows --example deepcoder -- \
  --workspace /absolute/path/to/project \
  --spec /absolute/path/to/specification.md
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--budget` | `50` | Workflow budget in dollars. |
| `--max-minor-corrections` | `20` | Minor repairs the correctness verifier may request per coder task. |
| `--max-efficiency-corrections` | `3` | Repairs the efficiency verifier may request per coder task. |
| `--max-final-corrections` | `4` | Minor repairs the final verifier may request. |
| `--shell-timeout-seconds` | `1800` | Default and maximum `run_shell` timeout. |
| `--shell-output-bytes` | `65536` | Maximum `run_shell` output returned to an agent. |
| `--state-dir PATH` | new directory | Run directory for checkpoints and logs. Must be outside the workspace. |
| `--resume PATH` | none | Resume from a checkpoint file or a run directory. |
| `--ephemeral` | off | Disable checkpoints and run artifacts. Cannot be combined with `--resume`. |
| `--dangerously-allow-all-commands` | off | Turn off the `run_shell` command guard, so every command runs as written. |

Budgets count billed cost. ChatGPT OAuth calls are billed at $0, so `--budget`
and the reported total cost only constrain API-key runs.

With persistence enabled, each run gets a uniquely named directory under
`.arko-deepcoder-runs` next to the workspace. `--resume` accepts either a
checkpoint file or a run directory containing a `latest-checkpoint` pointer. A
small input manifest records the canonical workspace path and the specification
hash, so a run cannot be resumed against different inputs by accident. A state
directory inside the workspace is rejected.

## Shared memory

| Tier | Writer | Contents |
| --- | --- | --- |
| `requirements` | none (read-only) | The specification, visible to every role. |
| `decisions` | orchestrator | Durable architecture and product decisions. |
| `implementation_log` | coder | Changed files and the checks run. |
| `verification_notes` | verifiers | Findings shared between verifiers. |

Every tier has a size limit. The workflow library checkpoints memory, agent tasks,
verification records, messages, freshness state, event logs, and costs.

## Shell tool and isolation

The example's `run_shell` tool runs local commands in the workspace with a
timeout and a cap on returned output, and only the coder and verifier roles
receive it. It removes common token, key, and secret variables from the child
environment. It refuses commands that would delete, move, overwrite or reformat a
protected path (the filesystem root, the home directory or its parents, system
directories, devices) or a path outside the workspace, using the
[terminal command guard](../../../docs/interactive-terminals.md#command-guard).
`--dangerously-allow-all-commands` turns that off. It isn't a sandbox for untrusted code. Role prompts tell verifiers
to stay read-only and coders to avoid deployment or publication, but shell
commands can reach anything the host process can.

The example has no isolated workspace copies or rollback-protected merges. Run it
in a disposable checkout, or add an application-specific isolation and merge layer
when untrusted changes, concurrent coders, or transactional rollback matter. Shell
output is truncated deterministically rather than distilled by a model, so
verifier evidence is never rewritten.
