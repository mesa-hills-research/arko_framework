# DeepCoder example

This is a generic, code-defined adaptation of the archived `deepcoder-v2` workflow. It demonstrates
how the reusable `arko_workflows` primitives replace application-specific orchestration code:

```text
orchestrator
  -> run_workflow_agent(coder)
       -> correctness verifier
       -> efficiency verifier (minor-only)
  -> repeat for dependency-ordered slices
  -> final verifier
  -> complete
```

The orchestrator stays sequential intentionally. Decomposition keeps individual agent tasks small,
and one accepted slice becomes the verified base for the next. The example does not add a graph or
parallel scheduler.

Each gate starts with a fresh comprehensive verifier. If a later gate requests a repair, the same
verifier conversation resumes with its prior verdict and a bounded tracked file/memory delta, so it
can reuse unchanged evidence and run focused checks. The final release gate is automatic: workflow
service target restrictions prevent the orchestrator from directly invoking verifier roles. Native
mixed-tool batches are enabled only for verifier reads; any batch containing a tool not marked
batch-safe is rejected without executing any call in that batch.

## Run

Set either an OpenAI API key:

```bash
export OPENAI_API_KEY=...
```

The example prefers ChatGPT OAuth and automatically discovers `$CODEX_HOME/auth.json` or
`$HOME/.codex/auth.json`. Explicit OAuth environment values can also be used:

```bash
export OPENAI_OAUTH_ACCESS_TOKEN=...
export OPENAI_OAUTH_REFRESH_TOKEN=...
export OPENAI_OAUTH_ACCOUNT_ID=...
```

Then run:

```bash
cargo run -p arko-workflows --example deepcoder -- \
  --workspace /absolute/path/to/project \
  --spec /absolute/path/to/specification.md
```

Useful controls:

```text
--budget 50
--max-minor-corrections 20
--max-efficiency-corrections 3
--max-final-corrections 4
--shell-timeout-seconds 1800
--shell-output-bytes 65536
--state-dir /path/outside/the/workspace
--resume /run/directory/or/checkpoint.json
--ephemeral
```

With persistence enabled, the default run directory is a uniquely named sibling under
`.arko-deepcoder-runs`. Resume accepts either a checkpoint or a run directory containing the
`latest-checkpoint` pointer. A small input manifest binds the run to the canonical workspace and
specification hash, preventing accidental resume against different inputs. Persistent state is
rejected if it is inside the mutable workspace.

## Shared memory

- `requirements`: immutable specification visible to every role.
- `decisions`: orchestrator-owned durable architectural/product decisions.
- `implementation_log`: coder-owned changed-file and check evidence.
- `verification_notes`: verifier-owned shared findings.

Each tier has an explicit writer policy and size bound. Memory, agent tasks, verification records,
messages, freshness state, event logs, and costs are checkpointed by the workflow library.

## Important boundary

The example's `run_shell` tool is intentionally small and cooperative. It runs local commands in the
workspace, caps returned output, applies a timeout, and is exposed only to coder/verifier roles. It
removes common token/key/secret variables from child environments. It is not a hostile-code sandbox.
Although role prompts require verifiers to remain read-only and coders to avoid
deployment/publication, shell commands can technically access anything available to the host
process.

Unlike the specialized archived harness, this example does not create VWS copies or perform
rollback-protected merges. Run it in a disposable checkout or add an application-specific isolated
workspace/merge layer when untrusted changes, concurrent coders, or transactional rollback are part
of the threat model. Deterministic shell output is capped rather than LLM-distilled so verifier
evidence is not silently rewritten.
