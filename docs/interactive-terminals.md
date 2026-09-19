# Interactive background terminals

`arko-agents::terminal` provides opt-in terminal tools for Linux, macOS, and
Windows 10 1809 / Server 2019 or later.
Jobs return immediately and can accept input while running. The host forwards
input/completion notices through the agent runtime's existing context channel.

## What the separate experiment established

The standalone Linux [PTY probe](../arko-agents/tests/fixtures/interactive_probe.py)
uses Python's standard library and real child processes, independently of the
Rust implementation. The 2026-09-13 Linux run covered 14 scenarios:

| Cases | Observation |
| --- | --- |
| Y/N without newline; stderr; split writes; ANSI color; raw key; `/dev/tty` | Input delivered successfully in all six. Prompt rules recognized them. |
| Silent input; an unflushed prompt | Input delivered successfully, but output-based detection missed both. |
| Quiet computation; quiet sleep | No question was present. |
| A log resembling a Y/N question | Prompt matching produced a false positive. |
| A background reader with other work continuing | A reading thread did not imply the job needed an answer. |
| Main thread reading while another thread completes the job | The main thread had the same `wait_woken` state as real questions. |
| That same threaded case printing a prompt-like log | Combining a matching prompt with a terminal wait still produced a false positive. |

The PTY was writable and its drained output queue was empty in every case.
These observations are not a universal input-wait detector. The implementation
therefore distinguishes application declarations from inferred prompt matches.
It does not use silence, process sleep state, or descriptor writability as proof
that a question needs an answer.

Reproduce the isolated probe with:

```sh
python3 arko-agents/tests/fixtures/interactive_probe.py
```

## Input contract

A snapshot has an execution `state` and a separate optional `input` notice:

- `input: null`: no current input request is known. Manual input remains possible.
- `input.source.kind: "program"`: the application explicitly declared a request.
  Its `request_id` accompanies the answer on the dedicated reply channel.
- `input.source.kind: "prompt_rule"`: a configured rule matched terminal output.
  This means **possible input needed**, and can have false positives or misses.

For every reply, supply the latest snapshot's `revision`. Output, new questions,
withdrawals, and accepted inputs advance the revision. The host rejects stale or
duplicate submissions against changes it has observed. For ordinary terminals,
the program can still change between observation and the actual write; revision
checking does not make an arbitrary terminal transaction atomic. Program-mode
receivers must check the reply's request ID before applying its answer.

Responses contain bounded recent output and byte cursors. Pass the previous
`output.end` as `after_output` to read only new bytes. `available_end` indicates
whether more bytes remain. `gap` reports that an earlier range left the bounded
buffer. Invalid or split UTF-8 has an exact `raw_hex` representation alongside
its display text. `prompt_truncated` flags a shortened inferred prompt.
`output.log` retains combined output in arrival order up to `max_log_bytes`;
`log_truncated` reports that limit. In PTY mode, stdout/stderr share a terminal. Windows ConPTY emits a rendered
terminal stream; use pipes when the original stdout/stderr bytes matter.
The prompt matcher handles common line prompts and ANSI decoration, not a full
screen terminal UI.

## Registering the tools

Share one manager across the four tools. Its directory must belong to this host;
the manager takes an exclusive lock and keeps artifacts in per-job subdirectories.

```rust,ignore
use arko_agents::terminal::{TerminalConfig, TerminalManager, TerminalTool};
use arko_agents::ToolRegistry;

let manager = TerminalManager::new(TerminalConfig::new(workspace.join(".terminal-jobs")))?;
let mut registry = ToolRegistry::new();
manager.register_tools(&mut registry);
// For a plain AgentRuntime, add TerminalTool::NAMES to the agent's available tools.
```

For `arko-workflows`, add the tools to the selected agent:

```rust,ignore
for tool in TerminalTool::all(manager.clone()) {
    builder = builder.agent_tool("builder", std::sync::Arc::new(tool));
}
```

`AgentRuntime::run_task` binds their weak context sender automatically through
`Tool::bind_runtime`. Custom hosts can call `manager.bind_runtime(task_id,
sender.downgrade())`. Jobs belong to a task ID; another task cannot inspect, answer,
or cancel them through these tools. The existing approval/middleware path applies
and exact terminal tool results bypass distillers.

| Tool | Behavior |
| --- | --- |
| `terminal_start` | Execute `command` in the task's working directory; return a job ID immediately. `terminal` selects PTY or pipes; `detect_prompts` toggles inferred notices; `timeout_ms` sets an execution deadline. |
| `terminal_status` | Read state and a bounded output window, optionally using `after_output`. |
| `terminal_input` | Deliver `text` using the latest `revision`. For ordinary line prompts send `\n` on Unix/pipes or `\r` on Windows PTYs; for raw keys send the key. Program requests receive a JSON reply with their request ID. |
| `terminal_control` | `list`, `wait`, `cancel`, or `forget`. Waits are bounded to 30 seconds and do not terminate a job. Forget removes a completed job from the catalog while retaining its artifacts. |

Integer tool parameters are encoded as strings for both native and text-based
calling. For example, `{"job_id":"…","revision":"3","text":"Y\n"}`.

The manager's defaults are four concurrent jobs, 64 retained job records, 64 KiB
of buffered output per job, 4 MiB of logs, 2,000-byte excerpts, and a 150 ms prompt
settling interval. Configure limits, the shell path and arguments, a default execution deadline,
and named regex rules through `TerminalConfig`. No execution deadline is imposed
unless configured. The default rules cover Y/N, password/passphrase, and common
“press a key” prompts. These tools execute with the host's process permissions;
terminal transport itself does not create a filesystem/network sandbox.

## Explicit requests from programs or adapters

The child receives two local file paths in environment variables:

- `ARKO_INPUT_REQUEST_PATH`: append newline-delimited UTF-8 JSON requests.
- `ARKO_INPUT_REPLY_PATH`: tail newline-delimited UTF-8 JSON replies.

A program can request input without printing to stdout. Both transports use the
same append-only file protocol. Each stream has one writer; append complete lines
and flush them. Readers retain partial lines and treat EOF as "no new bytes yet".
The host checks incrementally every 20 ms without a model call or network listener.
Use a host-owned directory on a local filesystem.

This example uses the small [`Replies` reader](../arko-agents/tests/fixtures/control_client.py)
to retain partial lines while waiting:

```python
import json, os, uuid
from control_client import Replies

replies = Replies()
request_id = str(uuid.uuid4())
with open(os.environ["ARKO_INPUT_REQUEST_PATH"], "a", encoding="utf-8") as requests:
    requests.write(json.dumps({
        "type": "input_request", "request_id": request_id,
        "prompt": "Continue with the selected option?"
    }) + "\n")
    requests.flush()

while True:
    reply = json.loads(replies.readline())
    if reply["request_id"] == request_id:
        answer = reply["text"]
        break
```

Declare one outstanding request at a time. Use a new ID for each question;
repeated IDs are deduplicated for the job's lifetime. To withdraw a question, send
`{"type":"input_cancelled","request_id":"…"}`. Check reply IDs even when using
withdrawals, since a reply can cross a cancellation in transit. Program-mode
answers are appended to the reply stream. Ordinary commands continue to
use stdin/PTY input. Stdout containing similar JSON cannot create a program notice.

Control frames are limited to 8 KiB, request IDs to 128 bytes, prompts to 4 KiB,
distinct requests to 1,024 per job, and each control stream to 8 MiB.
Control files retain the requests and replies alongside the output log. Never
truncate or replace an active stream. Malformed or conflicting declarations stop
the job with a diagnostic. A failed/partial input write also stops the job because
blindly retrying could duplicate an answer.

## Notifications and lifecycle

Input requests and completion produce context messages. One pending delivery per
job waits for channel capacity independently of the process. Superseded questions
are coalesced into the latest state. Unchanged prompts and routine output do not
produce repeated notifications; status reads remain available while output grows.
Rebinding an active manager to a resumed runtime can replay its latest notices;
use job ID/revision to identify updates.

Call `manager.shutdown().await` to cancel owned processes and wait for their
cleanup. Shutdown prevents new starts. Dropping the manager requests cancellation;
its directory stays locked while workers finish. A wait timeout is independent
of an execution deadline. PTY input supports control characters according to the
program's terminal settings; this first version does not provide terminal resize
or a pipe-stdin close operation.

On a host restart, saved running jobs become `interrupted`, and pending input is
cleared. Saved PIDs are never used to reattach, kill, or rerun a command. Job records
and logs are retained; the live output buffer and notification queue are not
checkpointed. A resumed host should list/reconcile jobs before relying on them.
The host controls continuation of an idle/completed agent; context delivery does
not itself start a new model run.

## Platform-specific behavior

| Area | Linux / macOS | Windows |
| --- | --- | --- |
| Interactive transport | Native PTY | Native ConPTY |
| Default shell | `/bin/sh -c` | `cmd.exe /D /S /C` |
| Enter for ordinary terminal input | `\n` | `\r` |
| Cancellation | Owned process group | Windows Job Object, assigned during process creation |
| Pipe mode | Piped stdin/stdout/stderr | Piped stdin/stdout/stderr |
| Declaration protocol | Append-only local files | Same protocol |
| Directory ownership | Standard library exclusive file lock | Same API |

The Windows backend follows Microsoft's [ConPTY session lifecycle](https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session).
Its [job-list process attribute](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute)
places the child in its Job Object at creation, avoiding an assignment race.
Input writes and ConPTY shutdown run independently of output reads so a full pipe
cannot block the async executor. The Unix backend uses nonblocking descriptors.

Shell strings still use the configured shell's syntax. A workflow providing shell
tools should include the host OS/shell in its small, stable context. Use portable
programs or explicit platform branches when authoring commands. The library
requires Rust 1.89+ for standard-library cross-platform file locking.

## Verification

The Rust integration tests use real shell/Python processes and exercise both
transports, multiple prompt styles, request/reply IDs, repeated and withdrawn
questions, stale and duplicate inputs, bounded/non-UTF-8 output, ownership,
backpressure, cancellation/deadlines, process-group cleanup, restart records,
and automatic runtime binding.

```sh
cargo test -p arko-agents --lib terminal:: --offline --locked
cargo run -p arko-agents --example interactive_terminal --offline --locked
```

The example runs a fixed Y/N fixture and supplies its predetermined test answer;
it requires no model call. Workflow prompts should instruct agents to inspect the
actual question, follow their task's authorization, use the current revision,
retrieve missing output, and preserve the distinction between program declarations
and inferred prompts.

Validation recorded on 2026-09-13:

- Linux: the standalone 14-scenario probe, 17 terminal integration tests, all 142
  framework library tests, and the interactive example passed locally.
- Windows x86-64 GNU and macOS ARM64: the actual platform I/O source compile-checks
  using an isolated harness with the workspace's Tokio version. This checks the
  new backend without requiring cross-compilers for unrelated native libraries.
- Native Windows/macOS interaction tests are prepared in
  [the CI matrix](../.github/workflows/terminals.yml); they have not been run here.
  This host only runs Linux. A whole-workspace Windows cross-check stopped at the
  existing `ring` dependency because the MinGW C compiler is not installed.

Python 3 is required for fixtures and the example; `ARKO_TEST_PYTHON` can select
its executable. The runtime library itself does not require Python.
