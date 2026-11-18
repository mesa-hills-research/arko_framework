# Changelog

## Unreleased

### apply_diff

- `apply_diff` applies every block of a call or none. A block not found, found more than once
  or overlapping another, and a diff that doesn't parse, leave the file byte for byte as it was.
  The error names each failing block and why, with `edit_file`'s hints and the first line that
  differs. Before, it wrote the blocks that matched and left the file half changed.
- A block's search lines match whole lines exactly, anywhere in the file, as it was before the
  call. The fuzzy search near `:start_line:`, the stripping of copied line numbers and the
  re-indenting of replacement lines are gone. Blocks need no `:start_line:` and `-------` lines,
  and blocks that still have them parse.
- `replace_all` replaces every occurrence of each block's search lines.
- A new file takes the file's place in one step, through a symbolic link, with its owner, group,
  permissions, ACL and other properties, so an error or a crash leaves the old file whole. A file
  with a property the new file can't keep, such as a second hard link, is refused unchanged, and
  so is any property not known to be kept. A byte-order mark, CRLF line breaks and a missing
  last line break are kept.
- Its result shows each block's new lines, as `edit_file`'s does, and carries a
  `ToolReceipt::Edit` covering every block.
- Its description is a few sentences and one example.
- `write_file`'s description points to `apply_diff` when the agent is offered it
  (`write_file::APPLY_DIFF_HINT`).

### Breaking changes

- An `apply_diff` call with a block that fails is an error and writes nothing.
- A line of the file that is itself a marker is written with a backslash before it
  (`\=======`). The `AD_ESC::` prefix is gone.

## 0.9.1 (2026-11-18)

### Waits inside the prompt cache

- `terminal_control wait` lasts `wait_ms` (1 second without it), up to the longest wait that ends
  before the agent's prompt cache lapses, counted from the request that made the call. It used to
  end after 30 seconds whatever `wait_ms` said. A longer request is cut to that, and a wait that
  then runs out has a `note` that says so, with the time the cache lapses when its lifetime is
  known. New output, the job's end or an input request, context that reaches the agent and a stop
  end a wait sooner.
- `TaskContext::prompt_cache` is the call's `PromptCacheWindow`, the task's cache lifetime and the
  anchor of the request that made the call, a moment no later than its use of the cache, on the
  monotonic clock. `PromptCacheWindow::deadline` is the same for every wait of a request, and
  `cap` shortens a wait to it. A known lifetime with no anchor, as for a task loaded from a save,
  leaves no wait. An unknown lifetime, or caching off, gives 4 min 30 s from the start of the
  wait.
- The window uses the conversation's switch for caching, so a conversation with caching off gets
  the fixed ceiling and not an hour. A call that follows a request to the refusal fallback's model
  gets that model's lifetime.
- In Claude Code the lifetime is the one the run asked the CLI for (`promptCacheTtl`), else the
  one the cache writes show, else 5 minutes. The anchor is the latest hand-off to the CLI that must
  come before its next main request, which is the turn's input or the hook, permission request or
  result of a call the main conversation announced. A slow or unstreamed reply no longer makes a
  request look newer than it is.
- `LatestRequest::sent_at` is that anchor, and the new `LatestRequest::sent_instant` is the same
  moment on the monotonic clock, set with `with_sent_instant`. It isn't saved.
- `TerminalManager::wait` no longer cuts its duration to 30 seconds, so the caller sets the bound.
  It returns when the job's revision or state changes or the duration ends, and an update that
  changes neither, such as part of an input declaration, doesn't end it.

### Dependencies

- Takes arko_ai 0.3.1.

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
