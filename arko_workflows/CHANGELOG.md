# Changelog

## 0.11.0 (2026-11-18)

### Waits end for notes and mail

- A note posted for a running agent, and mail that wakes it (a worker's report, a message from
  its lead), end the agent's running waits (`terminal_control wait`, `wait_for_message`), in the
  native loop and in Claude Code. Before, they waited for the next turn boundary, up to 270 s.
- `wait_for_agents` with `runs` still ignores mail from other runs.

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

### Dependencies

- Takes arko_ai 0.3.2.

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
