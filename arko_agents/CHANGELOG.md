# Changelog

## 0.10.0 (2026-11-18)

### Workspace paths

- The workspace check resolves symbolic links before it decides. Before, it compared the path's
  text only, so a link could take a read or write outside the workspace. The working directory
  and the path are compared in canonical form, so a link to the working directory, its canonical
  form and the spelling it was given in all get the same answer. For a path that doesn't exist
  yet, its deepest existing parent is resolved.
- A path that enters the workspace and leaves it through a link, such as `link/x` or `link/../x`
  where `link` points outside, is refused under every path policy, however it's spelled. So is a
  path through a link whose target doesn't exist, also when it ends in a separator (`link/`).
- The check judges each way a tool may read the path, trimmed or not, with `..` resolved by name
  from the working directory as given or from its canonical form, or by the operating system. A
  path that some readings find inside and others outside is refused. A path from `~` still counts
  as outside, and is refused when a link named `~` in the workspace leads outside.
- On Windows junctions are followed as links are. A `\\?\` path holding `/` and a name with `:`
  (a stream of a file) are refused, since tools read them differently.
- The check judges a path when the call is admitted, so a link created or changed before the tool
  opens the path isn't seen. It covers `is_path` parameters and the path inputs of Claude Code's
  file and search tools. Parameters not marked `is_path`, paths inside other arguments, a shell's
  working directory and what a program does with its paths are outside it.

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
- The edited text goes to a new file that takes the old one's place in one step, so an error or a
  crash leaves the old file whole. The new file has the old one's owner, group, permissions, ACL
  and other properties before any content is written. A symbolic link is followed and stays a
  link. A byte-order mark, CRLF line breaks and a missing last line break are kept.
- Its result shows each block's new lines, as `edit_file`'s does, and carries a
  `ToolReceipt::Edit` covering every block.
- Its description is a few sentences and one example.
- `write_file`'s description points to `apply_diff` when the agent is offered it
  (`write_file::APPLY_DIFF_HINT`).

### Tools reached only from inside a call

- `AgentToolConfig::nested_only_tool_names` lists tools an agent reaches only through
  `NestedCalls`, such as code that calls them. They aren't offered to the model, in either
  runtime, and a direct call of one is answered as unavailable. The nested calls' gate allows
  them, and they are reset and bound with the task as offered tools are. `runnable_tool_names`
  gives both kinds, and `is_nested_only` tells them apart. `AgentConfig::validate` refuses a
  name listed in `available_tool_names` too, one of `NEVER_NESTED`, and any for a chat.
- `Tool::offered_definition(tool_config)` gives the definition an agent with that configuration
  is offered, by default `definition()`, so a tool that calls nested-only tools can describe
  them. `AgentToolConfig::offered_tool_definition` shows another tool's definition as such an
  agent would see it, with the built-in tools' changes.
- The gate's refusal of a call that would wait for approval tells the agent to call the tool
  directly only when it can.
- `write_file`'s hint names `apply_diff` or `edit_file` also when the agent reaches that tool
  only from inside a call.

### Waits end for notes

- A call's `context_arrived` also fires while the task's turn context source holds context for
  its next update that should end a wait, such as a workflow's note. Before,
  `terminal_control wait` ran its whole time while such a note waited.
- `TurnContextSource` gains `has_waking_update` and `update_changed`, both defaulted, and
  `ContextSignal` gains `with_updates` and `without_updates`.
- The signal holds the source weakly, so a call moved to the background doesn't keep the
  terminal manager alive through it.

### ask_user

- Its description says to repeat an unanswered question in the same words. It said to ask the
  question again, which read as a new `ask_user` call while the question was still open.

### Breaking changes

- An `apply_diff` call with a block that fails is an error and writes nothing.
- `apply_diff` refuses a file, and leaves it as it was, when the new file can't keep one of its
  properties or the file has one not known to be kept. Among them are a second hard link, the
  setuid or setgid bit, an owner other than the process's user, an ACL on macOS, and encryption,
  compression or named streams on Windows. On Unix systems other than Linux and macOS it refuses
  every file. The error says why and points to `write_file` or the shell. `edit_file` still
  writes these files in place.
- `apply_diff` refuses an empty file (`write_file` gives it content) and a file with a NUL byte.
  It refuses a pipe or a device without reading it.
- A line of the file that is itself a marker is written with a backslash before it
  (`\=======`). The `AD_ESC::` prefix is gone.
- `AgentToolConfig` has a new public field, `nested_only_tool_names`.
- `ToolError::ExecutionFailed` and `ToolError::Refused` read as their message alone. The loop
  already says the tool failed, so a failed call read "Tool 'read_file' execution failed:
  Execution failed: …" to the model, in the events and in saved conversations. It now reads
  "Tool 'read_file' execution failed: …".

### Dependencies

- Takes arko_ai 0.3.2.

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
