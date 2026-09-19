# Accurate, efficient context

Context tooling must preserve the evidence needed for a decision. Use deterministic
changes for mutable state and structured reports; use model distillation selectively
for bulk material. Keep authoritative inputs and expandable evidence available.

## Memory reads

`read_memory` returns a full, versioned snapshot on the first read in an agent
runtime. Later reads return exact changes or an unchanged result. `full: true`
requests a fresh snapshot. Append-only updates return the exact appended text;
editable memory returns a line replacement including deletions. The tool returns
a full snapshot when that is smaller than the change record.

Responses are JSON records with `tier`, `version`, and `kind`. Full and append
records carry `content`. A replacement carries `base_version`, `start_line`
(zero-based), `delete_lines`, and replacement `content`. Lines retain their newline
terminators, so applying the change reconstructs the current text byte for byte.
Append records also identify `base_version`. An unchanged record identifies the
current version. Versions count the tier's accepted writes.

Baselines belong to a task and tier, and advance only after the exact result enters
that task's conversation. Reads waiting for batch completion do not advance them.
Restarting or resuming a runtime resets its baselines; the next read is full.
Applications replacing/compacting a conversation should start a new runtime.
Custom runners can use `Tool::reset_context(task_id)` and
`Tool::observe_context(task_id, message)` for the same lifecycle. Observations
must contain the actual message sent to the model. Without observations, reads
conservatively return full snapshots. These synchronous callbacks only track
delivery; they do not perform model calls.
When `max_chars` truncates the content, it returns an explicitly partial trailing
window and does not establish or advance a baseline. Exact memory results bypass result
interceptors, including distillers.

The tool response changed from raw text to JSON. Applications parsing tool output
must handle the record kinds; direct `MemoryStore::read` and the Python host's
`read_memory` retain their existing full/window behavior. This protocol belongs
to an agent's conversation, not a process-wide memory cache.

## File-tree context

When file-tree environment context is enabled, `file_tree_mode: "diff"` is the
default. Before the first model call, the runtime injects a complete tree within
the configured depth and ignore rules. Before subsequent calls it sends only
visible additions, removals, type changes, and identifiable moves. An unchanged
tree produces no message. This applies to native and text-based tool calling.

Diff mode sends tree context in user messages at model-call boundaries, independent
of the legacy placement setting. Set `file_tree_mode: "full"` to preserve full-tree
rendering at the configured placement. An agent can use `list_files` for an explicit
full listing at any time. Restart/resume supplies a new full baseline.
Rust struct literals for `EnvironmentDetailsConfig` now need `file_tree_mode`
or `..Default::default()`; saved configurations without the field use diff mode.

Tree entries are sorted. The scanner does not follow symlinked directories or read
file contents. On Unix, move matching uses unambiguous filesystem identity and
metadata; uncertain matches and platforms without that identity show removal and
addition. Tree changes describe the visible inventory, not file-content edits.
Ignore/depth changes can also change that inventory. A failed scan does not replace
the last successful baseline. The runtime reports the scan failure and retries.

## Distillation

Distillers are opt-in through `WorkflowBuilder::distiller`. Configure the model,
reasoning, target agents/tools, minimum input characters, maximum output tokens,
and instructions. The threshold applies to Unicode characters. Empty target sets
apply to all eligible agents/tools; select targets for predictable use.

```rust,ignore
use arko_ai::{Model, OpenAIEffort, ReasoningConfig};
use arko_workflows::prelude::*;

let distiller = DistillationConfig::new("bulk-research", Model::openai_gpt_5_6_luna())
    .reasoning(ReasoningConfig::openai(OpenAIEffort::High, false))
    .target_agent("researcher")
    .target_tool("search_documents")
    .min_chars(12_000)
    .max_output_tokens(2_000)
    .instructions("Preserve exact findings, qualifications, and source references. Remove irrelevant repetition.");
// Register with Workflow::builder().distiller(distiller).
```

The existing defaults remain 8,000 input characters and 4,000 output tokens;
omitted reasoning uses the provider's default. Reasoning must match the selected
provider and the output limit must be positive. Distillers see the task prompt,
tool name, parameters, and raw result. Their output is model-authored, not a
deterministic accuracy guarantee. Keep source evidence accessible through tools.
Interception events and the `PostToolResultInterception` hook expose original and
effective text; use a host hook to retain source artifacts when needed. Configured
interceptor-error policies still fall back to raw output. Bound huge outputs in the producing tool
when raw fallback would be unsuitable.

## Shell tools and background jobs

`arko-agents::terminal` supplies reusable interactive background tools for Linux,
macOS, and Windows.
See [interactive terminals](interactive-terminals.md) for the separate detection
experiment, program request/reply protocol, inferred prompt notices, and setup.

Workflows offering shell tools should provide an immediate-return background mode
for long commands. Keep quick reads convenient; use a configurable short foreground
wait and return a running job handle when that wait expires. A wait limit and an
execution deadline are separate settings. Background jobs must be owned by the
host runtime so they survive the return of the tool call.

Use this contract when designing a shell tool:

- Start returns a stable job ID, status, working directory, and log/artifact paths.
  Configure concurrency limits and execution deadlines for the workflow's workload.
- Completion sends a compact notice containing job ID, terminal status, exit code,
  relevant diagnostic excerpts, and log/artifact references. Preserve failure and
  cancellation details. Report truncation and provide bounded range or cursor
  reads for logs; repeated progress reads should return only new output.
- The host monitors jobs and delivers notices through `SharedContext.messages`
  (`MessageBus::send` for a named recipient, or `inject_task` for an active task).
  An idle owner needs a host continuation policy; a queued message alone does not
  start a model run. Avoid agent polling loops or a separate polling agent.
- Persist job ownership and notification IDs for workflows that support resume.
  Checkpoint queued delivery before acknowledging it; deduplicate replayed notices
  by job/event ID. Reconcile interrupted jobs against recorded state and artifacts
  before retrying commands with side effects.
- Provide status, bounded wait, log read, and cancellation operations. Document
  pause/shutdown behavior and which shell state persists between invocations.
  Explicitly terminate owned processes when cancellation requires it.

The local `arko-research` workflow demonstrates this pattern in `src/jobs.rs`,
`src/tools.rs` (`lab_start_job` / `lab_shell(background=true)`), and
`src/research.rs::notify_jobs`. Its test
`immediate_jobs_inject_success_and_failure_without_polling_model` checks compact
completion delivery. These are reference implementations in that workflow;
the reusable terminal manager can provide transport and job handling for new hosts.
Hosts still select the process permissions and lifecycle appropriate to their workflow.

Reusable agent guidance:

> Start long shell commands in background mode and retain their job IDs. Continue
> useful independent work while they run. Completion notifications will identify
> the outcome and where to retrieve details. If the next step depends on a job,
> use the host's bounded wait or yield mechanism. Inspect progress only when it
> informs a decision, and cancel work that is no longer needed. Save substantial
> output and checkpoints to artifacts. Keep model turns purposeful; avoid filler
> turns whose only purpose is to keep a prompt cache warm.

## Review and prompting

For each workflow, define authoritative inputs, ownership, freshness, required
evidence, and retrieval paths before implementation. Prefer the simplest design
that supports accurate decisions. Explain useful changes without repeating known
facts. Evaluate context optimizations against exact reconstruction/coverage tests
and representative agent behavior; fewer characters alone do not establish quality.
