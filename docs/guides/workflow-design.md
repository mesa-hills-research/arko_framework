# General Workflow Design Guide for Arko Framework

A concise guide to designing reliable, efficient workflows with arko_framework. It covers workflow structure and the decisions that matter before implementation. For agent prompts, tools, and model selection, see the [general agent design guide](agent-design.md). For API details, defaults, and complete signatures, see the [`arko_workflows` API index](../../arko_workflows/INDEX.md) and the [reference map](../reference-map.md). The [complete workflow example](../../arko_workflows/examples/complete_workflow.rs) shows how the pieces fit together in a runnable workflow.

This guide offers ideas, not rules. A simpler approach is usually better than an elaborate one: start with the simplest workflow that could meet the goal, and add agents, stages, or controls only when it falls short. Aim for workflows that run autonomously.

A workflow built with `arko_workflows` is always a binary. It never means a wrapper around `arko_agents`, such as an agent configuration run through `arko_agents_cli`, unless that is explicitly requested: `arko_workflows` exists for more comprehensive, autonomous agent workflows.

## Start with the outcome

Define the result before choosing agents or communication tools. Write down the deliverable, the source material, what counts as complete, and how a reviewer can verify it. Include the conditions that require human input, the conditions that should stop the run, and the conditions that can be retried safely.

A workflow should have one clear owner for each decision and artifact. If two agents can both change the same output, define how their changes are merged or which result wins. If the workflow only needs one agent to perform a task, use one agent; coordination adds latency, context, and failure modes.

Treat the workflow as a sequence of observable state changes rather than a collection of prompts. Each stage should have an input, an action, an output, and a clear handoff to the next stage. This makes it easier to run stages in parallel, resume after interruption, and diagnose incomplete work.

## Choose the topology

Start with the smallest topology that meets the requirements. `arko_workflows` supports ordinary Rust or Python control flow around sequential, parallel, and background agent execution; the framework does not require a separate graph or scheduler configuration.

- **Sequential stages** work well when one result is required before the next stage can begin. Use them for extract-then-transform or draft-then-review flows.
- **Parallel stages** work well when agents can work independently on the same input. Use `run_agents_parallel` when the work can be divided without shared writes or ordering requirements. Set a parallelism limit when provider capacity, memory, or cost requires one.
- **Background agents** are useful for work that can continue while the host handles another part of the run. Define how the host observes completion and what happens if the parent run stops.
- **Delegated stages** let one agent call another through the bounded `run_workflow_agent` service. Enable delegation only for callers and targets that have a concrete reason to use it, and keep nested work bounded.
- **Human-in-the-loop stages** pause when the workflow needs a decision, clarification, or approval. Make the question specific enough that the host can answer it without reconstructing the entire run.

Separate independent discovery from dependent decisions. For example, several investigators can inspect different sources in parallel, then one writer can combine their findings, and a reviewer can verify the finished artifact.

## Give each agent a bounded role

Give each agent a responsibility that can be described in one or two sentences. A useful role names the input it reads, the artifact or decision it owns, and the signal that tells it to finish. Avoid roles such as “manage the entire project” when the workflow can divide the work into discovery, implementation, and review.

Keep agent configuration close to the role it serves. Provide only the tools, memory tiers, recipients, and model capacity the role needs; agents tend to use whatever they are offered or told about, whether or not the task needs it. Narrow configuration makes failures easier to attribute and reduces the amount of irrelevant context each agent must process.

When several agents perform the same kind of work, give them independent inputs or explicit comparison tasks. If they all edit the same file or memory entry without a merge rule, the workflow is relying on timing rather than design.

## Design shared state deliberately

Use shared memory for durable, structured context that several agents need to read or update. Use messaging for a timely handoff, notification, or question. Use the workspace for artifacts that must be inspected by tools or retained outside the conversation. The same fact may appear in more than one place, but define which location is authoritative.

### Memory

Create a separate memory tier for each kind of shared state with a different audience or retention rule. Examples include requirements, research findings, decisions, and verification notes. Configure readers and writers explicitly. Give write access only to the agents that own a tier's content.

Choose a write mode that matches the data. Append-only history is useful for findings and audit notes; replacement is useful for a current plan or status document. Set a size limit when unbounded growth would crowd out the context needed to finish the task.

On resume, configure the same tier names or explicitly map or drop changed tiers.

### Messaging

Use messages to tell another agent that work is ready, to pass a small instruction, or to ask for a response. A message should state what changed, where the durable result is stored, and what the recipient should do next. Do not use a message as the only copy of a requirement or result that must survive a checkpoint.

Start with the narrowest `MessagingPolicy` that supports the topology. One-way communication is useful when a reviewer should report to an owner without receiving work back. Allow two-way communication when agents need to clarify or coordinate. An agent can wait for a message, but the workflow should define a timeout or a continuation path for a missing response. Agents are told when a recipient finishes, so one waiting on a reply learns when none can come. If the workflow continues or reruns agents, `keep_messages_for_finished()` lets a late reply reach the next run instead of being refused.

Give long-running agents, such as a lead that coordinates workers for hours, a conversation handover (`AgentConfig::handover`) rather than a restart: the agent continues in the same run from its own report, so its identity, messages and budget carry over.

### Operator notes

Use a note (`Workflow::post_note`) when the person running the workflow, the host program, or an observer needs to steer agents that are already running: a correction ("the helper's bug is fixed; counts after 14:00 are real"), a workaround found meanwhile, a new priority. Running agents get it at their next turn boundary and later runs in their initial prompt, so it is the right tool when guidance must outlive the current run. A message is the agents' own channel, delivered to one run and done; memory is shared state agents read when they choose to; notes are pushed to every run of their audience, and standing notes stay in later prompts: a newer note takes precedence where they differ, but the older one stays visible, so state what changed. A note that is really a document belongs in a file or a memory tier, with a short note pointing to it.

Notes are also the supported way to change instructions while a workflow runs or across a resume. Changing an agent's configuration (its instructions or tools, for example) makes a resume refuse the tasks it saved; a note changes what the agents do without touching the configuration. Keep notes few and specific: each one stays in every later prompt (the newest first, capped), and a newer note takes precedence where they differ, so state the change rather than repeating the whole plan. One-time notes (`Note::once`) suit something that matters now, such as "re-read section 3, it changed"; standing notes (the default) suit anything later runs must know.

A note is delivered in the user role, labelled with its source, as a refinement of the task: agents are told it doesn't change their role's rules. It carries the authority of whoever can post it, so treat posting as part of the workflow's trust boundary. Agents have no tool to post notes, but with the opt-in control folder (`WorkflowBuilder::control_folder`), whoever can write `<run dir>/control/` can post as any source: keep the run directory, and any `note` command the binary offers, out of the agents' workspace and sandboxes. Label an observer agent's notes as its own (`NoteSource::Agent`), so the recipients weigh them as another agent's view rather than the operator's.

Keep independent roles independent. A completion verifier gets a note only when the note names it, and judges the completion without the target's notes: when a note changes what counts as done ("skip the appendix"), address it to the verifier too, or it may keep rejecting completions that follow the note. Mark other roles that must not be steered, such as a checker or an independent replicator, with `notes_by_name_only`. Since an agent that receives a note can pass it on in a message, restrict whom such a role can hear from with the `MessagingPolicy`; the workflow logs a warning when a note can reach such an agent that way.

### Freshness

When agents can read and write the same files, memory tiers, or other resources concurrently, enable context-freshness tracking. A freshness conflict means an agent is about to act on data that another actor changed after it read it. Decide whether the agent should reread, merge, retry, or stop for review.

## Plan tools and context

List the tools each role needs before building the workflow. Global tools are appropriate for capabilities every agent should share; agent-local tools are better when a capability is sensitive, expensive, or relevant to only one role. Keep tool names and descriptions stable because prompts, freshness mappings, interceptors, and distillers can target them by name.

Use tool-result interceptors when every call of a tool needs a deterministic transformation, such as redaction or normalization. Use distillation when selected large results should be condensed before they reach an agent. Distillation is an additional model call and an additional cost, so target only the tools or agents that need it. If a transformation must withhold data when it fails, configure its failure behavior so the raw result is not delivered accidentally.

Give each stage the smallest useful context, preserve important state in memory or files, and ask agents to record concise findings for the next stage.

Plan for large tool results. Every tool returns its whole output, and one large file or query result can be more than a model takes in a request. The overflow guard (on by default) shortens a turn's results only when they would certainly overflow the model's input limit, with a notice telling the agent how to get the rest, so a workflow doesn't need its own read cap. Give roles that meet large files a terminal and tell them to read in parts. Where a role should never pull large outputs, or one tool returns too much, set `ToolResultLimits::max_result_tokens` or that tool's `max_result_tokens`. A distiller skips an output larger than its own model takes, so a distiller meant for very large outputs needs a model with a large input limit.

## Define completion and review

An agent completes when it has produced the artifact or decision required by its role and recorded a useful summary. A successful `complete_task` call is a handoff signal, not proof that the artifact is correct.

Use a completion verifier when a result must satisfy a concrete contract. A verifier can inspect the target's files and memory, accept the completion, or reject it with repair context that sends the target back to work. Give the verifier a narrow checklist and access to the evidence it needs.

Keep verification separate from creation when independence matters: the reviewer checks the finished state and reports specific failures rather than editing the output itself.

For workflows that need consensus rather than a single review, use structured voting with explicit options, eligible voters, quorum, weights, and tie behavior. Give voters well-defined options to choose from.

## Set boundaries for tools and failures

The workflow's defaults are autonomous: every tool call is approved (`ApprovalPolicy::ApproveAll`), and errors are recovered automatically without an attempt limit (`ErrorRecoveryPolicy::Automatic`). Keep them for set-and-forget workflows; they are the only settings that let a run finish without a person. Don't switch to `Limited`, `ReturnToCaller`, `RespectAgentConfig`, or an approval handler out of caution; use them only when a person is meant to supervise the run.

Use hooks for lifecycle policy and observation: budget warnings, audit events, pre-tool checks, post-tool recording, freshness conflicts, and cancellation. A hook can stop a run but can't undo work already done, so run checks before the side effect and record the reason for any abort.

Route questions through a `UserInputHandler` when the host can answer them asynchronously. Without a handler, the run returns `UserInputRequired`; continue the paused agent with the answer rather than starting a second run that loses its conversation.

Decide what each kind of failure means for the run. A failed run records the kind of error that failed it (`AgentResult::failure_kind`, `arko_ai::ErrorKind`): a lost login (`Authentication`) or a credit wall (`Billing`) will fail every agent on that credential, so a set-and-forget host should stop the run and ask for a new sign-in; a request too large for the model (`ContextOverflow`) or an invalid request fails only that run. Decide by the kind, never by matching error text.

Give cancellation a deliberate path. A cancel handle can stop the workflow from another task, and hooks can abort it when a policy is violated. Configure a grace period that lets agents record results without allowing a stuck tool call to hold the run indefinitely. Report cancellation distinctly from failure so a caller can decide whether resuming is appropriate.

## Budget the workflow

Set a workflow budget when a run has a known cost ceiling, and set agent budgets when one role must not consume the whole allowance. Add a warning threshold before the hard limit and use the cost ledger to explain where spending went. Include manual charges and distillation costs in the estimate.

Choose models and reasoning effort per role. The [model-choice guidance](agent-design.md#model-choice) recommends Opus 5.5 or GPT-6.1 Sol for most work, stronger tag-in models for a different perspective, Astra 6 for 3D design paired with Fable 5.1 for 3D interaction, GPT-6.1 Sol (or, less often, Sonnet 5.5) for codebase-exploration subagents, and GPT-6 Luna for high-volume transformations.

## Make side effects durable

Use persistence when a workflow must survive process loss, operator interruption, or a long pause. Checkpoints retain the workflow identity, finalized agent results, memory, pending messages, operator notes, cost ledger, verification and distillation records, freshness state, and task versions. Configure a run directory outside source control and decide how many checkpoints to retain.

Design resume behavior before enabling checkpoints. `resume_from` restores state and configuration from the new builder; it does not replay the original Rust or Python control flow and it does not automatically continue agents. The host must inspect restored results and explicitly run the next stage or continue a paused agent. A checkpoint can also record cancellation, so decide whether the host should keep a stopped run stopped before resuming it.

Wrap external side effects in durable steps. A durable step records its input, attempt, and outcome so a retry does not repeat work. If an operation can finish after the process loses contact with it, provide a resolver for an in-doubt result or require an operator to mark it completed, failed, or safe to retry.

Keep saved state compatible with earlier versions. Add serde defaults for new fields, continue reading old checkpoint and task formats, and avoid changing the meaning of an existing workflow name or memory tier without an explicit migration path.

Resume with the configuration the run was saved with. `resume_from` refuses a run whose agent configuration changed, and its error names the changed fields (`instructions`, `tool_config.available_tool_names`, ...), so you can restore them or register the changed configuration under another agent name. Changes outside the agent configurations (the workflow's Rust code, tool implementations and descriptions, hooks, policies, budgets) don't stop a resume.

## Observe and test the run

Subscribe to workflow events or register hooks when the host needs progress, audit records, or live status. Report agent results, waiting states, pauses, costs, and cancellation causes separately. A final summary should identify incomplete work and the evidence used to accept completed work.

Build and validate the workflow before making provider calls. The complete workflow example supports a dry run that builds the workflow and writes a checkpoint without credentials or network access. Use that path to catch missing agents, invalid policies, unknown targets, and persistence errors early. Workflow runs are expensive, so gauge the user's budget before live test runs (the smallest real run, an interruption and resume, a verifier rejection, a missing human answer, a tool failure, or a budget boundary) or iterating on a workflow against tests or benchmarks. When the user authorizes extensive testing and refinement, go ahead. At any level of testing, start simple and change the workflow only where the results show it's necessary.

## Workflow checklist

- The deliverable, owner, completion condition, and reviewer are explicit.
- The topology uses the fewest agents and stages that meet the requirements.
- Parallel work has independent inputs or a defined merge rule.
- Each memory tier has an owner, access policy, write mode, and size limit where needed.
- Messages carry handoff information, while durable facts live in memory or files.
- A run someone steers while it runs takes operator notes instead of configuration changes, and independent roles get only the notes that name them.
- Shared resources have freshness tracking or an explicit conflict policy.
- Each agent has only the tools and context required for its role. Roles that meet large files read them through the terminal; caps on tool results are set only where a role needs a tighter budget than the overflow guard gives.
- Completion verification checks the artifact, not only the agent's summary.
- Set-and-forget workflows keep `ApproveAll` and `Automatic`; cancellation, budgets, and human input are configured for the deployment, and a lost login (`failure_kind()` is `Authentication`) stops the run rather than failing every agent in turn.
- External side effects use durable steps and define in-doubt resolution.
- Checkpoints and old saved data remain loadable, and resume behavior is explicit.
- A dry run passes. Live test runs and benchmark iterations fit the budget the user has authorized.
