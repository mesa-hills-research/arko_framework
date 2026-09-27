# Interactive terminals

`arko-agents::terminal` provides opt-in terminal tools for Linux, macOS, and
Windows 10 1809 / Windows Server 2019 or later. A command starts as a background
job and the tool call returns immediately. The job can accept input while it runs,
and the host delivers input requests and completion notices to the owning agent
through the agent runtime's context channel.

## Registering the tools

Share one `TerminalManager` across the four tools. The manager's directory must
belong to this host: the manager takes an exclusive lock on it and keeps each
job's artifacts in a per-job subdirectory.

```rust,ignore
use arko_agents::terminal::{TerminalConfig, TerminalManager, TerminalTool};
use arko_agents::ToolRegistry;

let manager = TerminalManager::new(TerminalConfig::new(workspace.join(".terminal-jobs")))?;
let mut registry = ToolRegistry::new();
manager.register_tools(&mut registry);
// For a plain AgentRuntime, add TerminalTool::NAMES to the agent's available tools.
```

With `arko-workflows`, add the tools to the selected agent:

```rust,ignore
for tool in TerminalTool::all(manager.clone()) {
    builder = builder.agent_tool("builder", std::sync::Arc::new(tool));
}
```

`AgentRuntime::run_task` binds the tools the agent is offered to the task's context
channel through `Tool::bind_runtime`. Custom hosts call `manager.bind_runtime(task_id,
sender.downgrade())` themselves. Jobs belong to the task that started them;
another task cannot inspect, answer, or cancel them through these tools. The
runtime's approval and middleware path applies to terminal tools as to any other
tool, and their results bypass tool-result distillers so output reaches the agent
unchanged.

| Tool | Behavior |
| --- | --- |
| `terminal_start` | Run `command` in the task's working directory and return a job ID immediately. `terminal` selects PTY or pipes; `detect_prompts` toggles inferred input notices; `timeout_ms` sets an execution deadline. |
| `terminal_status` | Read the job state and a bounded output window, optionally starting at `after_output`. |
| `terminal_input` | Deliver `text`, citing the latest `revision`. For an ordinary line prompt send `\n` on Unix and in pipe mode, or `\r` in a Windows PTY; for a raw key send the key alone. A program request receives a JSON reply carrying its request ID. |
| `terminal_control` | `list`, `wait`, `cancel`, or `forget`. A wait ends after at most 30 seconds and does not stop the job. `forget` removes a finished job from the catalog and keeps its artifacts. |

Integer parameters are passed as strings in both native and text-based tool
calling, for example `{"job_id":"…","revision":"3","text":"Y\n"}`.

## Configuration

`TerminalConfig::new(directory)` sets these defaults:

| Setting | Default |
| --- | --- |
| Concurrently running jobs (`max_running`) | 4 |
| Retained job records (`max_jobs`) | 64 |
| Buffered output per job (`output_buffer_bytes`) | 64 KiB |
| Log file size per job (`max_log_bytes`) | 4 MiB |
| Output window per status read or notice (`excerpt_bytes`) | 2,000 bytes |
| Output quiet period before prompt matching (`prompt_quiet_ms`) | 150 ms |
| Execution deadline (`default_timeout`) | None |
| Shell (`shell`, `shell_args`) | `/bin/sh -c` on Unix, `cmd.exe /D /S /C` on Windows |
| Prompt rules (`prompt_rules`) | Y/N, password/passphrase, and "press a key" prompts |

Prompt rules are named regular expressions. Jobs run with the host process's
permissions; the terminal tools do not provide a filesystem or network sandbox.

## Input requests

A snapshot has an execution `state` and an optional `input` notice:

- `input: null`: no input request is known. Input can still be sent at any time.
- `input.source.kind: "program"`: the program explicitly declared a request
  (see [Program input requests](#program-input-requests)). The reply carries its
  `request_id`.
- `input.source.kind: "prompt_rule"`: a configured rule matched the terminal
  output. This means input is *possibly* needed; rules can miss prompts and can
  match output that is not a question.

Every `terminal_input` call supplies the `revision` of the latest snapshot. Output,
new or withdrawn requests, and accepted input advance the revision, and the host
rejects stale or duplicate submissions. For ordinary terminal input, the program
can still change between the observation and the write: revision checks do not
make a terminal interaction atomic. Programs using explicit requests must check
the request ID of each reply before applying it.

### Why prompt detection is advisory

Whether a process is waiting for terminal input cannot be determined reliably
from outside it. The standalone Linux probe
[`interactive_probe.py`](../arko-agents/tests/fixtures/interactive_probe.py)
(Python standard library, real child processes) runs 14 scenarios against a PTY:

- Prompt matching recognizes Y/N prompts without a newline, prompts on stderr,
  split writes, ANSI-colored prompts, raw-key reads, and prompts written to
  `/dev/tty`.
- It misses silent input reads and prompts left in an unflushed buffer.
- It matches log output that only resembles a question, including while another
  thread finishes the job.
- Silence, a process sleeping in a read, and PTY writability look the same for a
  real question, a background reader thread, and ordinary computation.

The terminal tools therefore report program declarations and inferred prompt
matches as distinct sources, and never treat silence, process sleep state, or
descriptor writability as evidence that a question needs an answer. The probe
runs independently of the Rust implementation:

```sh
python3 arko-agents/tests/fixtures/interactive_probe.py
```

## Reading output

Responses contain a bounded window of recent output with byte offsets. Pass the
previous `output.end` as `after_output` to read only new bytes; `available_end` is
the end of the buffered output, so more bytes remain when it is greater than
`end`. `gap` reports that part of the requested range has already left the
buffer. When the window contains invalid or split UTF-8, `raw_hex` holds the exact
bytes alongside the display text. `prompt_truncated` marks a shortened inferred
prompt.

Each job's `output.log` (at `log_path`) keeps the combined output in arrival order
up to `max_log_bytes`; `log_truncated` reports that the limit was reached. In PTY
mode stdout and stderr share one terminal, and Windows ConPTY emits a rendered
terminal stream; use pipe mode when the original stdout and stderr bytes matter.
The prompt matcher handles line prompts and ANSI decoration, not full-screen
terminal interfaces.

## Program input requests

A program, or an adapter wrapping it, can declare an input request without
printing anything. The child process receives two file paths in its environment:

- `ARKO_INPUT_REQUEST_PATH`: append newline-delimited UTF-8 JSON requests.
- `ARKO_INPUT_REPLY_PATH`: read newline-delimited UTF-8 JSON replies.

Both terminal modes use this append-only file protocol. Each stream has a single
writer, which appends complete lines and flushes them. Readers keep partial lines
and treat end-of-file as "no new bytes yet". The host polls the request stream
every 20 ms, without a model call or a network listener. Keep the manager's
directory on a local filesystem.

This example uses the [`Replies` reader](../arko-agents/tests/fixtures/control_client.py)
from the test fixtures, which retains partial lines while waiting:

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

Declare one outstanding request at a time and use a new ID for each question;
repeated IDs are deduplicated for the lifetime of the job. To withdraw a question,
send `{"type":"input_cancelled","request_id":"…"}`. Check reply IDs even when
withdrawing requests, because a reply can cross a cancellation. Answers to
program requests go to the reply stream; ordinary commands keep receiving input
through stdin or the PTY. JSON printed to stdout never creates a program request.

Limits: 8 KiB per request line, 128 bytes per request ID, 4 KiB per prompt, 1,024
distinct requests per job, and 8 MiB per control stream. The request and reply
files are kept with the output log; never truncate or replace an active stream.
A malformed or conflicting declaration stops the job with a diagnostic. A failed
or partial input write also stops the job, because retrying it could deliver an
answer twice.

## Notifications and lifecycle

Input requests and job completion produce context messages for the owning task.
Each job has at most one pending delivery, which waits for channel capacity
without blocking the job. A newer question replaces an undelivered older one.
Unchanged prompts and routine output do not produce further notifications; status
reads remain available while output grows. Rebinding an active manager to a
resumed runtime can replay the latest notices; use the job ID and revision to
recognize updates. A notice does not start a model run by itself: the host decides
whether an idle or completed agent continues.

`manager.shutdown().await` cancels owned processes, waits for their cleanup, and
prevents new starts. Dropping the manager requests cancellation; the directory
stays locked until the workers finish. A wait timeout is independent of a job's
execution deadline. PTY input handles control characters according to the
program's terminal settings. Terminal resizing and closing stdin in pipe mode are
not supported.

After a host restart, saved running jobs are marked `interrupted` and pending
input is cleared. Saved process IDs are never used to reattach to, kill, or rerun
a command. Job records and logs persist; the live output buffer and notification
queue do not. A resumed host should list and reconcile jobs before relying on
them.

## Platform behavior

| Area | Linux / macOS | Windows |
| --- | --- | --- |
| Interactive transport | PTY | ConPTY |
| Default shell | `/bin/sh -c` | `cmd.exe /D /S /C` |
| Enter for ordinary terminal input | `\n` | `\r` |
| Cancellation | Owned process group | Job Object, assigned at process creation |
| Pipe mode | Piped stdin, stdout, and stderr | Piped stdin, stdout, and stderr |
| Program input requests | Append-only local files | Same protocol |
| Directory ownership | Standard-library exclusive file lock | Same API |

The Windows backend follows Microsoft's
[ConPTY session lifecycle](https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session)
and uses the
[job-list process attribute](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute)
to place the child in its Job Object at creation, avoiding an assignment race.
Input writes and ConPTY shutdown run independently of output reads, so a full pipe
cannot block the async executor. The Unix backend uses nonblocking descriptors.

Commands use the configured shell's syntax. Workflows that offer shell tools
should include the host OS and shell in the agent's context and write commands
with portable programs or explicit platform branches.

## Prompting agents

Agent instructions for terminal work should ask the agent to read the actual
question before answering, stay within its task's authorization, cite the current
revision, fetch missing output before deciding, and treat inferred prompts as
possible rather than certain input requests.

## Testing

The integration tests run real shell and Python processes. They cover both
transports, several prompt styles, request and reply IDs, repeated and withdrawn
questions, stale and duplicate input, bounded and non-UTF-8 output, ownership,
backpressure, cancellation and deadlines, process-group cleanup, restart records,
and automatic runtime binding. CI runs them on Linux, macOS, and Windows.

```sh
cargo test -p arko-agents --lib terminal:: --locked
cargo run -p arko-agents --example interactive_terminal --locked
```

The example answers a fixed Y/N fixture with a predetermined answer and needs no
model or credentials. The fixtures and the example require Python 3;
`ARKO_TEST_PYTHON` selects the interpreter. The library itself does not require
Python.
