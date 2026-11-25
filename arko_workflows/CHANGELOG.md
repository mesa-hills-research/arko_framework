# Changelog

## 0.11.0 (2026-11-25)

### Waits end for notes and mail

- A note posted for a running agent, and mail that wakes it (a worker's report, a message from
  its lead), end the agent's running waits (`terminal_control wait`, `wait_for_message`), in the
  native loop and in Claude Code. Before, they waited for the next turn boundary, up to 270 s.
- `wait_for_agents` with `runs` still ignores mail from other runs.
- Every other wait of the run ends the same way. `UserInputRequest::context_arrived` and
  `ToolApprovalRequest::context_arrived` fire when context arrives while the handler waits, or at
  once for context that came before, so the handler can return and let the agent read it. A
  subagent's question to its lead ends its wait too, and the subagent reads what arrived.

### Starting subagents

- `SubagentHook::prepare_start`, async, is called for a live `start_agent` call with a `StartCall`
  (the call's IDs, its stop signal and its context signal), so a host can wait for the user's
  approval and stop waiting on a stop or a note. `StartOutcome::NotStarted(text)` gives the text as
  the call's result without starting the run or failing the call. The default calls `prepare`,
  which a resume and a handover still call. `start_agent` no longer ends at the host's tool
  timeout, since its hook may wait for a person.
- `start_agent` takes `workspace` together with `isolated: true`, and makes the isolated
  workspace from the given one after its checks. Both together were refused.
- `SubagentsConfig::message_parameters` (`message_parameter`) adds the host's own parameters to
  `message_agent`, as `start_parameters` does for `start_agent`.

### Terminal jobs

- On the third signal `stop_on_signals` also kills the process groups of the terminal jobs still
  running (`arko_agents::kill_terminal_jobs`), which outlived the exit before.

### Antigravity

- A `PlanBudget`'s limits leave out the Antigravity plan, whose model groups have windows of their
  own.

### Workspaces through links

- `start_agent`'s `workspace` is judged as arko_agents' workspace check judges a path, with
  symbolic links resolved. Before, it was compared by name only, so a link inside the caller's
  directory could start a run outside it with `workspaces_anywhere` off. Such a workspace is now
  refused with the reason.

### Tools reached only from inside a call

- An agent tool listed in the agent's `nested_only_tool_names` isn't offered to its runs. A saved
  run needs its nested-only tools registered to resume, as it does its offered ones, and the
  freshness configuration and distillers may name them.

### Breaking changes

- arko_workflows takes arko_agents 0.10, whose types its public API holds. Its agents get
  arko_agents 0.10's `apply_diff`, which applies every block of a call or none.
- `UserInputRequest` and `ToolApprovalRequest` have a new public field, `context_arrived`, and
  `SubagentsConfig` has `message_parameters`, so struct literals break.

### Dependencies

- Takes arko_ai 0.4, whose types its public API holds.

## 0.10.1 (2026-11-18)

### Waits inside the prompt cache

- `wait_for_message` and `wait_for_agents` take their deadline from the call's
  `TaskContext::prompt_cache`, as `terminal_control wait` does, so every wait of a request ends at
  the same moment. `wait_for_message` no longer dates the agent's requests itself.
- With a known cache lifetime a `WaitForMessageTimeout::Fixed` limit only shortens the wait, so no
  limit runs past the cache. An exhausted cache leaves no wait, with no one-second minimum, and a
  message still being forwarded at the deadline no longer extends it.
- `WaitForMessageTimeout::resolve` gives `Auto` a known lifetime's `longest_wait`, at most 24 hours,
  with no minimum.

### Dependencies

- Takes arko_ai 0.3.1 and arko_agents 0.9.1.

## 0.10.0 (2026-11-18)

### Breaking changes

- arko_workflows takes arko_ai 0.3 and arko_agents 0.9, whose types its public API holds.
