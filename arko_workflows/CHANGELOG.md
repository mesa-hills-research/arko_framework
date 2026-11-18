# Changelog

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
