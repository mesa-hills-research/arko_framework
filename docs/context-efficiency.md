# Context management

Arko keeps an agent's context accurate and compact with four mechanisms:
versioned memory reads, file-tree change notices, opt-in tool-result distillation,
and background shell jobs with completion notices. State that changes during a run
is delivered as exact, deterministic changes. Model-written summaries are reserved
for bulk tool output, and only where a workflow opts in by naming the agents and
tools to distill. The original output stays in the event log and interception
events; an agent that needs the exact text of a distilled result (for example a
distilled `read_file`) can't get it by calling the tool again.

## Memory reads

In an agent runtime, the `read_memory` tool from
[arko_workflows](../arko_workflows/README.md) returns a full, versioned snapshot on
the first read of a tier. Later reads return the exact change since the last
snapshot the agent received, or an unchanged record. `full: true` requests a fresh
snapshot. When the new text extends the previous text, as appends do, the change
is the exact appended text; otherwise it is a single line replacement, which can
include deletions. When a full snapshot would be no larger than the change, the
tool returns the snapshot instead.

Responses are JSON records with `tier`, `version`, and `kind`:

| `kind` | Meaning |
| --- | --- |
| `full` | `content` is the complete current text. |
| `append` | Append `content` to the text at `base_version`. |
| `replace` | At `base_version`, replace `delete_lines` lines starting at zero-based `start_line` with `content`. |
| `unchanged` | The text is unchanged; adopt the returned `version`. |
| `partial` | `max_chars` truncated the text: `content` is the trailing window of `total_chars` characters. |

Lines keep their newline terminators, so applying a change reproduces the current
text byte for byte. A version counts the tier's accepted writes. A `partial`
response is not a complete snapshot and does not establish a baseline. The
[API index](../arko_workflows/INDEX.md#memory) documents the full record contract.

Baselines are kept per task and tier, and advance only once the exact result has
entered that task's conversation; reads still waiting on a tool batch do not
advance them. Restarting or resuming a runtime resets its baselines, so the next
read is full. An application that replaces or compacts a conversation should start
a new runtime. Custom runners reproduce this lifecycle with
`Tool::reset_context(task_id)` and `Tool::observe_context(task_id, message)`,
passing the actual messages sent to the model. Without observations, reads
conservatively return full snapshots. Both callbacks are synchronous and only
track delivery.

Memory read results bypass result interceptors, including distillers. The
versioned protocol belongs to the agent tool: `MemoryStore::read` and the Python
`read_memory` method return plain text.

## File-tree context

When file-tree environment details are enabled in `EnvironmentDetailsConfig`,
`file_tree_mode` defaults to `diff`. Before the first model call the runtime sends
the complete tree, within the configured depth (3 when unset) and ignore rules.
Before later calls it sends only visible additions, removals, type changes, and
identifiable moves; an unchanged tree adds no message. This works with native and
text-based tool calling. Configurations without `file_tree_mode` use diff mode.

Diff mode sends tree context as user messages at model-call boundaries, regardless
of the file tree's placement setting. `file_tree_mode: "full"` renders the full
tree at the configured placement instead. Agents can call `list_files` for a full
listing at any time, and a restarted or resumed runtime starts from a new full
snapshot.

Tree entries are sorted. The scanner does not follow symlinked directories or read
file contents, so the notices describe the visible inventory, not edits to file
contents. On Unix, moves are detected from filesystem identity and metadata when
the match is unambiguous; other platforms, and uncertain matches, report a removal
and an addition. Changing ignore rules or depth also changes the visible
inventory. A failed scan keeps the last successful baseline; the runtime reports
the failure and scans again before the next call.

## Tool-result distillation

Distillers are opt-in and registered with `WorkflowBuilder::distiller`. Each one
selects its model, reasoning, target agents and tools, minimum input size (in
Unicode characters), maximum output tokens, and instructions. Target agents are
the names the agents are registered under with `WorkflowBuilder::agent`, not
their `AgentConfig::name`, and `build` rejects a target that is not a registered
agent. A distiller needs at least one target agent and one target tool: `build`
rejects one without. Any tool whose results may be intercepted can be a target,
`read_file` included, for a large file an agent wants specific details from; the
agent then works from the summary, so target it only where that suits.

```rust,ignore
use arko_ai::{Model, OpenAIEffort, ReasoningConfig};
use arko_workflows::prelude::*;

let distiller = DistillationConfig::new("bulk-research", Model::openai_gpt_5_6_luna())
    .reasoning(ReasoningConfig::openai(OpenAIEffort::High, false))
    .target_agent("researcher") // The registration name used below.
    .target_tool("search_documents")
    .min_chars(12_000)
    .max_output_tokens(16_000) // Room for high-effort reasoning plus the summary.
    .instructions("Preserve exact findings, qualifications, and source references. Remove irrelevant repetition.");

let workflow = Workflow::builder()
    // Name, workspace, and client as usual.
    .agent("researcher", researcher_config)
    .distiller(distiller)
    .build()?;
```

Defaults are 8,000 input characters and, without a set output limit, 16,000 output
tokens (or the model's output limit, when smaller) where the provider takes a limit.
Without an explicit reasoning setting, the provider's default applies. The reasoning
configuration must match the model's provider, and the output limit must be
positive. Prompt caching is turned off for distillation requests where the provider
allows it, since each is a one-off; on ChatGPT OAuth, which caches implicitly and
rejects an output limit, neither setting is sent unless the limit is set
explicitly.

Reasoning tokens count against the output limit with OpenAI API keys and on
Anthropic, and some models reason by default (for example GPT-5.5 and later, and
Claude Opus 5.5). Leave room for reasoning as well as the summary: a limit that is
too small produces truncated summaries, which are billed but not used.

A distiller sees the task prompt, the tool name and parameters, and the raw
result. Its output is model-written, not a guaranteed-accurate transformation, so
keep source evidence reachable through tools. Interception events and the
`PostToolResultInterception` hook expose both the original and the delivered text;
use a hook to retain source artifacts when needed.

Only a finished summary replaces the tool result, and every call is billed:

- A refusal, or a summary cut off by the output limit, keeps the raw result
  without a retry, because a retry would most likely end the same way.
- Any other early stop (including an unreported stop reason) or an empty summary
  is an interceptor error. The workflow's interception failure policy may retry it
  before falling back to the raw result (resuming at the distiller when it follows
  other interceptors).

Once the workflow run is stopped, cancelled or aborted by a hook, distillers make
no further requests and leave tool output unchanged.

If a raw fallback would be unsuitable for very large outputs, bound the output in
the tool that produces it. `complete_task` and `ask_user` results are never
distilled, and `load_skill` results only when the agent's skills configuration
opts in (`SkillsConfig::load_skill_interception` in arko_agents). The
[API index](../arko_workflows/INDEX.md#tool-result-distillation) documents
configuration fields, records, and cost accounting.

## Background shell jobs

The [interactive terminal tools](interactive-terminals.md) in
`arko_agents::terminal` run commands as background jobs on Linux, macOS, and
Windows, accept input while they run, and notify the owning agent on completion.

A workflow that provides its own shell tool should follow the same pattern: long
commands start in the background and return a job handle immediately, while quick
commands stay convenient through a short, configurable foreground wait that
returns the handle when it expires. The foreground wait and the execution deadline
are separate settings. The host runtime owns background jobs, so they outlive the
tool call that started them. A complete design provides:

- **Start**: returns a stable job ID, status, working directory, and log or
  artifact paths. Concurrency limits and execution deadlines are configurable.
- **Completion notices**: a compact notice with the job ID, final status, exit
  code, relevant diagnostic excerpts, and log or artifact references, preserving
  failure and cancellation details. Truncation is reported, logs support bounded
  range or cursor reads, and repeated progress reads return only new output.
- **Delivery**: the host monitors jobs and delivers notices through
  `SharedContext.messages` (`MessageBus::send` for a named recipient, or
  `inject_task` for an active task). A queued message does not start a model run
  on its own, so an idle owner needs a host continuation policy. Agents should not
  poll in a loop, and no separate polling agent is needed.
- **Resume**: workflows that support resume persist job ownership and
  notification IDs, checkpoint queued deliveries before acknowledging them,
  deduplicate replayed notices by job or event ID, and reconcile interrupted jobs
  against recorded state and artifacts before retrying commands with side effects.
- **Control**: status, bounded wait, log reads, and cancellation, with documented
  pause and shutdown behavior, a statement of which shell state persists between
  invocations, and explicit termination of owned processes when cancellation
  requires it.

Reusable agent instructions:

> Start long shell commands in background mode and keep their job IDs. Continue
> useful independent work while they run. Completion notifications identify the
> outcome and where to retrieve details. If the next step depends on a job, use the
> host's bounded wait or yield mechanism. Inspect progress only when it informs a
> decision, and cancel work that is no longer needed. Save substantial output and
> checkpoints to artifacts. Keep model turns purposeful; avoid turns whose only
> purpose is to keep a prompt cache warm.
