---
name: workflow-creation
description: Design or modify arko_workflows workflows when a task involves workflow topology, agent coordination, shared state, verification, persistence, or lifecycle behavior.
---

# Workflow creation

Use this skill when creating or changing a workflow in `arko_framework`.

A workflow built with `arko_workflows` is always a binary. It never means a wrapper around `arko_agents`, such as an agent configuration run through `arko_agents_cli`, unless that is explicitly requested: `arko_workflows` exists for more comprehensive, autonomous agent workflows.

## Read the relevant guidance

The design guides offer ideas, not rules.

- Read the [general workflow design guide](../../docs/guides/workflow-design.md) before choosing the topology or shared-state design.
- Read the [general agent design guide](../../docs/guides/agent-design.md) when changing agent roles, prompts, tools, models, or reasoning effort.
- Consult the [`arko_workflows` API index](../../arko_workflows/INDEX.md) for exact signatures, defaults, and persistence behavior.
- Use the [complete workflow example](../../arko_workflows/examples/complete_workflow.rs) as the reference for a workflow that combines memory, messaging, verification, distillation, checkpoints, and resume.

## Design the workflow

1. Define the deliverable, source of truth, completion condition, reviewer, human-input points, and safe retry boundaries.
2. Choose the smallest topology that fits: sequential stages for dependencies, parallel stages for independent work, bounded delegation for materially useful subtasks, and human pauses for decisions the workflow cannot make safely.
3. Give each agent one bounded role with explicit inputs, owned outputs, tools, memory tiers, and permitted recipients.
4. Use memory for durable shared state, messages for handoffs and questions, and files for artifacts. Assign every shared resource an owner and an access policy.
5. Enable freshness tracking when agents can concurrently read and write the same files, memory tiers, or other resources. Define whether a conflict causes a reread, merge, retry, or review.
6. Add completion verification when a successful `complete_task` call is not enough evidence that the artifact is correct. Keep reviewers independent when that matters.
7. Configure cancellation, budgets, and human input for the deployment. For set-and-forget workflows, keep the default `ApprovalPolicy::ApproveAll` and `ErrorRecoveryPolicy::Automatic`: they are the only settings that let a run finish without a person. Decide by the error's kind, never its text: a run that failed with `AgentResult::failure_kind()` `Some(ErrorKind::Authentication)` (a lost login) will be followed by every agent on that credential, so stop the run, save it, and ask for a new sign-in; a `ContextOverflow` or `InvalidRequest` failure is that run's alone.
8. Use persistence for runs that must survive interruption. Wrap external side effects in durable steps and define how in-doubt outcomes are resolved. Preserve compatibility with earlier checkpoints and task files.

Start with the simplest approach that could work, and build up only when it falls short. Anything you mention in an agent's prompt or offer as a tool or parameter steers the agent toward using it, even when the task doesn't need it, so decide carefully what each agent sees and phrase optional things as optional. Keep the workflow control flow explicit, and add agents, shared write access, retries, and expensive model calls only when each has a concrete role in the outcome. Prefer the model and reasoning recommendations in the model-choice section of the agent guide; leave `model.max_output_tokens` at the model maximum unless a clear limit is required.

## Design patterns

These patterns are optional. Use one only when the workflow has the coordination problem it solves; many workflows need none of them.

### Orchestrator and delegated specialists

Give one lead agent ownership of decomposition, delegation, synthesis, and the final decision. Let it spawn or call bounded specialist agents for independent subtasks, then require each specialist to return evidence, an artifact location, and unresolved questions. `run_workflow_agent` is the framework's bounded delegation service; use explicit caller and target allowlists and a concurrency limit. If the host owns scheduling, use the workflow execution methods instead.

Fan out only as far as the task benefits from independent work. A research workflow might let a lead start up to eight researchers on demand, while a smaller task might need only two. Keep specialist inputs separate when independent discovery matters, and define whether specialists may message the lead, one another, or both.

### Verifier-driven repair

Place a verifier on the completion boundary when an agent's `complete_task` summary is not enough evidence. Have the verifier classify findings as accepted, minor, or major:

- A minor finding sends focused feedback back to the offending agent for repair. Continue the same agent task so it keeps its context and repair history.
- A major finding sends the verifier's findings to a fresh run of the offending agent. Restarting from a clean task prevents a deeply flawed approach from anchoring the repair, while the findings and required evidence remain available in the new prompt or memory.
- Continue the same verifier across reviews. Enable `reuse_conversation(true)` when it should retain its prior reasoning on the same target task, and keep prior findings or repair deltas available when a new verifier run is required. Conversation reuse is keyed to the target task ID, so a major repair that starts a fresh target task must carry the verifier's earlier findings through persistent memory or a seeded review prompt.

Set a total minor-repair cap for the workflow, commonly 10–15 across all targets and verifier passes. `CompletionVerifierConfig::minor_rejection_cap` is keyed per target task and verifier, so enforce a workflow-wide cap in the host or orchestrator when that is the intended limit. When the cap is reached, stop cycling and escalate or finish with an explicit incomplete result.

### Verifier feedback to the orchestrator

Send every verifier finding to the orchestrator as well as to the agent that must repair it. The orchestrator should incorporate recurring findings into the instructions, task decomposition, or acceptance criteria it uses for future specialists. Store durable findings in a dedicated memory tier so later sessions can avoid repeating known mistakes.

### Independent research and replication

For research tasks, let the lead delegate independent questions to researchers and preserve separate evidence until synthesis. Researchers can message the lead and one another to compare or correct findings. Record agreed conclusions separately and keep each researcher's evidence. Independent prompts and separate outputs make counterexamples more likely to be rediscovered instead of copied from the first report.

### Handover reports for long runs

For agents that approach a context or input-token limit, set `AgentConfig::handover` with a threshold appropriate to the model: `HandoverConfig::at_context_fraction(0.25)`, or `at_input_tokens(200_000)` for a long research run. At the threshold the runtime asks the agent for a report covering the task, completed work and evidence, decisions, open questions, what it is waiting for and from whom, and the next steps. During that turn it rejects every tool call except `complete_task`, then continues the same task in a fresh conversation that starts from the task and the report. The workflow keeps the agent's identity, delivered messages, and budget, and refreshes its recipient list; nothing needs to be checkpointed or restarted by hand. Set `report_request` to ask for workflow-specific content, and use `UserCommand::RequestHandover` to trigger one from the host. Save durable results in memory or files as usual; the report is for the agent's own continuation.

## Choose the autonomy level

Decide how much supervision the workflow is meant to require before configuring its policies and tools. Default to fully autonomous; choose supervision only when a person is meant to take part in the run, not out of caution.

- A fully autonomous workflow should be set and forget. Keep `ApproveAll` and `Automatic`, and leave agents on a pausing `llm_error_policy` (the default) so the workflow recovers their errors instead of the run ending interrupted. Define budget limits, cancellation behavior, and the complete toolset in advance. Omit `ask_user` unless a real handler can answer it automatically; an unanswered question turns a supposedly autonomous run into a pause.
- A supervised workflow can pause for approvals, error decisions, and questions. Give the host a clear continuation path and make each question specific enough to answer without reconstructing the run.
- A remotely supervised workflow can remain autonomous while sending status, failure, or decision requests through an explicit integration such as a Telegram bot or another API service.

Choose tools according to that level. `ask_user`, approval requests, error-recovery pauses, and external notification tools should be deliberate parts of the interaction contract. Configure what the workflow may do without a person, what it must report, and what it must stop for.

## Implement and verify

- Inspect existing workflow code and saved-data formats before changing them.
- Rely on the overflow guard (`ToolResultLimits`, on by default) for tool results too large for the model: don't write your own read cap in a hook. Give agents that meet large files a terminal and tell them to read in parts (`wc -l`, `grep -n`, `sed -n 'a,bp'`); set `max_result_tokens` only where a role or a tool needs a tighter budget. `read_file` line ranges are opt-in, for agents without a terminal.
- Resume with the build and agent configurations the run was saved with. A refused resume names the changed configuration fields; restore them, or register the changed configuration under another agent name.
- Use the framework’s actual builder and execution APIs; do not invent a separate scheduler or graph layer.
- Declare the external programs the workflow runs in its `WorkflowMetadata`, required and optional, with install hints, and check them at startup with `missing_dependencies()`: stop before any model call when a required one is missing, and warn about optional ones.
- Run `cargo fmt --all -- --check` and the repository’s required locked checks after implementation.
- Exercise a dry run before provider calls when the workflow supports one.
- Workflow runs are expensive. Gauge the user's budget before running live tests of the pause, error, verification-rejection, cancellation, budget, persistence, and resume paths, or iterating on the workflow against tests or benchmarks; when the user authorizes extensive testing and refinement, go ahead. At any level of testing, start simple and change the workflow only where the results show it's necessary.
- Report incomplete work, pauses, cancellation causes, costs, and the evidence used to accept a result.
