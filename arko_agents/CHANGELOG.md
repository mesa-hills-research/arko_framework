# Changelog

## 0.9.0 (2026-11-18)

### Nested calls

- A tool can call the agent's other tools from inside its own call, through `TaskContext::nested`,
  in the native loop and in the Claude Code runtime. A call made through `NestedCalls` gets the
  checks a call of the model's gets. The task's configuration must offer the tool, its paths pass
  the path policy, and the execution middleware sees it before and after. A call that would wait
  for approval is refused at once, and `complete_task`, `ask_user`, `close_questions`,
  `handover_report` and the calling tool can't be called this way.
- Nested calls run one at a time, in the order made, under their tool's time limit and the outer
  call's stop signal. A stop or pause the middleware decides at a nested call is the outer call's.
- Each nested call emits `AgentEvent::NestedToolCallStarted` and `NestedToolCallFinished`, which
  name the outer call. The start holds the arguments cut at 4 KB each (names at 256 bytes, at most
  16 KB in all) with their whole size as JSON, and the end holds at most 4 KB of the result. Every
  start has its end, also when the call is stopped while the event channel is full.
- `NestedCalls::records` lists the calls whose tools started, each as far as it got, so a caller
  that was stopped or timed out can say what its calls may have changed.
- A check inside a tool that refuses a call, such as the command guard, can return
  `ToolError::Refused`. It reads as `ExecutionFailed` does.
- `AgentToolConfig::auto_approves` and `path_safety_policy` answer for a tool name, as the loop
  does.

### Models

- Reasoning replays between GPT-6.2 and the GPT-5.6 and GPT-6 models, which share Codex's
  compaction family.

### Breaking changes

- arko_agents takes arko_ai 0.3, whose types its public API holds.
- `AgentEvent` has the variants `NestedToolCallStarted` and `NestedToolCallFinished`, and
  `ToolError` has `Refused`, so an exhaustive `match` on either needs arms for them.
