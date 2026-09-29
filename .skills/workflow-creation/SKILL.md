---
name: workflow-creation
description: Design or modify arko_workflows workflows when a task involves workflow topology, agent coordination, shared state, verification, persistence, or lifecycle behavior.
---

# Workflow creation

Use this skill when creating or changing a workflow in `arko_framework`.

## Read the relevant guidance

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
7. Configure approvals, error recovery, cancellation, budgets, and human input for the deployment. The workflow defaults approve tool calls and recover errors automatically, so do not treat those defaults as a safety policy.
8. Use persistence for runs that must survive interruption. Wrap external side effects in durable steps and define how in-doubt outcomes are resolved. Preserve compatibility with earlier checkpoints and task files.

Keep the workflow control flow explicit. Do not add agents, shared write access, retries, or expensive model calls without a concrete role in the outcome. Prefer the model and reasoning recommendations in the model-choice section of the agent guide; leave `model.max_output_tokens` at the model maximum unless a clear limit is required.

## Design patterns

Use these as patterns to choose from, not as requirements for every workflow. Combine them only when each pattern solves a real coordination problem.

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

Send every verifier finding to the orchestrator as well as to the agent that must repair it. The orchestrator should incorporate recurring findings into the instructions, task decomposition, or acceptance criteria it uses for future specialists. Store durable findings in a dedicated memory tier so later sessions can avoid repeating known mistakes; messages alone are not a persistent learning record.

### Independent research and replication

For research tasks, let the lead delegate independent questions to researchers and preserve separate evidence until synthesis. Researchers can message the lead and one another when comparison or correction is useful, but shared conclusions should be marked as conclusions rather than silently replacing each researcher's evidence. Independent prompts and separate outputs make counterexamples more likely to be rediscovered instead of copied from the first report.

### Continuation reports for long runs

For agents that approach a context or input-token limit, track cumulative usage from the workflow's usage events and choose a threshold appropriate to the model. A 200,000-input-token threshold is a useful example for a long research run. At the threshold, inject a continuation instruction for the next turn: require a concise progress report covering completed work, evidence, open questions, and the next steps, then require `complete_task`.

Enforce the continuation turn at the tool boundary rather than relying on the prompt alone. During that turn, reject every tool call except a dedicated continuation-report operation and `complete_task`; save the report in memory or a file, checkpoint the run, and resume from the report in a fresh context. This preserves progress without allowing the agent to spend the continuation turn on more exploration.

## Choose the autonomy level

Decide how much supervision the workflow is meant to require before configuring its policies and tools.

- A fully autonomous workflow should be set and forget. Define its error-recovery policy, approval policy, retry and budget limits, cancellation behavior, and complete toolset in advance. Omit `ask_user` unless a real handler can answer it automatically; an unanswered question turns a supposedly autonomous run into a pause.
- A supervised workflow can pause for approvals, error decisions, and questions. Give the host a clear continuation path and make each question specific enough to answer without reconstructing the run.
- A remotely supervised workflow can remain autonomous while sending status, failure, or decision requests through an explicit integration such as a Telegram bot or another API service. Notifications do not replace durable state or a defined recovery decision.

Choose tools according to that level. `ask_user`, approval requests, error-recovery pauses, and external notification tools should be deliberate parts of the interaction contract. Do not rely on permissive defaults to define autonomy: configure what the workflow may do without a person, what it must report, and what it must stop for.

## Implement and verify

- Inspect existing workflow code and saved-data formats before changing them.
- Use the framework’s actual builder and execution APIs; do not invent a separate scheduler or graph layer.
- Run `cargo fmt --all -- --check` and the repository’s required locked checks after implementation.
- Exercise a dry run before provider calls when the workflow supports one.
- Test the important pause, error, verification-rejection, cancellation, budget, persistence, and resume paths for the workflow’s deployment.
- Report incomplete work, pauses, cancellation causes, costs, and the evidence used to accept a result.
