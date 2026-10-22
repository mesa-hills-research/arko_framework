# Interactive terminals

`arko_agents::terminal` provides opt-in terminal tools for Linux, macOS, and
Windows 10 1809 / Windows Server 2019 or later. A command starts as a background
job and the tool call returns immediately. The job can accept input while it runs,
and the host delivers input requests and completion notices to the owning agent
through the agent runtime's context channel. A [monitor](#monitors) watches a
command's output, or a check run again and again, and alerts its agent of what
matters.

## Registering the tools

Share one `TerminalManager` across the terminal and monitor tools. The manager's
directory must belong to this host, because the manager takes an exclusive lock on
it and keeps each job's artifacts in a per-job subdirectory.

```rust,ignore
use arko_agents::terminal::{TerminalConfig, TerminalManager, TerminalTool};
use arko_agents::ToolRegistry;

let manager = TerminalManager::new(TerminalConfig::new(workspace.join(".terminal-jobs")))?;
let mut registry = ToolRegistry::new();
manager.register_tools(&mut registry);
// For a plain AgentRuntime, add TerminalTool::NAMES (and MonitorTool::NAMES for
// monitors) to the agent's available tools.
```

With `arko_workflows`, add the tools to the selected agent:

```rust,ignore
for tool in TerminalTool::all(manager.clone()) {
    builder = builder.agent_tool("builder", std::sync::Arc::new(tool));
}
```

`AgentRuntime::run_task` binds the tools the agent is offered to the task's
context channel through `Tool::bind_runtime`. Custom hosts call
`manager.bind_runtime(task_id, sender.downgrade())` themselves. Jobs belong to
the task that started them. Another task cannot inspect, answer, or cancel them
through these tools. The runtime's approval and middleware path applies to
terminal tools as to any other tool, and their results bypass tool-result
distillers so output reaches the agent unchanged. The agent's host-side tool time
limit does not apply to them (a job runs in the background with its own deadline,
and a wait ends after at most 30 seconds), unless a per-tool setting gives one.

| Tool | Behavior |
| --- | --- |
| `terminal_start` | Run `command` in the task's working directory and return a job ID immediately. `terminal` selects PTY or pipes. `detect_prompts` toggles inferred input notices. `timeout_ms` sets an execution deadline. |
| `terminal_status` | Read the job state and a bounded output window, optionally starting at `after_output`. |
| `terminal_input` | Deliver `text`, citing the latest `revision`. For an ordinary line prompt send `\n` on Unix and in pipe mode, or `\r` in a Windows PTY. For a raw key send the key alone. A program request receives a JSON reply carrying its request ID. |
| `terminal_control` | `list`, `wait`, `cancel`, or `forget`. A wait ends after at most 30 seconds and does not stop the job. `forget` removes a finished job from the catalog and keeps its artifacts. |

Integer parameters are passed as strings in both native and text-based tool
calling, for example `{"job_id":"…","revision":"3","text":"Y\n"}`.

Each update reaches its owner once. An update that finds no runtime bound (the
agent's turn has ended) waits for the next runtime the owner binds, such as the
one of its next turn. One sent while the agent makes the last request of its
turn waits with its task (`Task::pending_context_injections`) for the next turn.

## Held jobs

A host that runs a command to completion itself, such as a shell tool, can hand
the command to the agent while it runs. `StartOptions { held: true, .. }` starts
a held job. It sends no terminal updates and `terminal_control list` leaves it
out, while the host reads it with `status` and `wait` as usual.
`TerminalManager::release(owner, job_id)` moves it to the background. From then
on it is listed, and its updates go to its owner, starting with an event it
already has. A job that finished before the release stays held, and `release`
returns `TerminalError::Finished`, as the host has its result.
`StartOptions::shell_args` gives one job its own arguments before the command,
for example a non-interactive mode for commands run to completion.

`TerminalManager::subscribe_finished()` reports each job that finishes in the
background, with its owner, ID and state, so that a host can start a turn of an
idle agent for it.

`TerminalManager::background_jobs()` lists the jobs running in the background,
of every owner, with each one's command and start time, so that a host can say
what stops before it exits. `StartOptions::label` sets the command shown, for a
host that wraps the command it runs.

`TerminalManager::close(known)` readies the manager for the host's exit. If a
job runs in the background that isn't in `known` (the jobs the host's user
agreed to stop), it stays open and returns the jobs running. Otherwise it
closes, so no job starts and none moves to the background (`release` fails), so
`shutdown` stops no background job the user didn't know of. The check and the
closing are one step under the jobs' lock, so a job that moves to the
background at the same moment is either listed or refused.

`TerminalManager::jobs()` lists the same jobs as `JobInfo`, with what each runs
(`JobKind::Command` or `JobKind::Call { tool_name }`) and the call that made it
(`tool_call_id`, the call that started the command or the call that moved).
`has_news(owner)` says whether an owner has an update or a monitor's news that
went to no runtime because none was bound, also news that waits for its next
turn and should wake no one. `has_alerts(owner)` counts only the news that
should wake an idle agent, so a host can start a turn for it. That's a job's
update (unless the user stopped the job or the owner's run ended), and a
monitor's alert, its warning while the user counts as active, or its command
exiting by itself.

`stop_for_user(owner, job_id)` stops a job for the host's user. Its update says
the user stopped it (`error` is `stopped by the user`), and `subscribe_finished`
doesn't report it, so it waits for the owner's next turn instead of waking an
idle agent. `cancel` is the owner's own stop.

## Moved tool calls

The manager also runs tool calls that moved to the background. While a call of a
tool that can move runs in the foreground, `hold_call(owner, tool_call_id,
tool_name, label)` lists it. A runtime given the manager
(`AgentRuntime::with_background_jobs`) does this for the calls of tools whose
`Tool::background_mode` isn't `Never`, once the execution middleware lets them
run. `foreground_calls()` lists the calls that can move, oldest first, and
`watch_foreground_calls()` changes as that list does, for a host that shows
which calls can move. `move_to_background(owner, Some(tool_call_id))` moves one
call, and `None` moves all of an owner's calls. It returns how many moved, and a
manager closed for its host's exit moves none. The `HeldCall` guard the call
holds sees the move (`is_moved`, `moved().await`, `signal()`), and dropping it
takes the call off the list. `set_background_after(Some(after))` makes the calls
of `Detach` tools move by themselves after running that long, read with
`background_after()`. The default is `None`, never.

`adopt_call(owner, tool_call_id, tool_name, label, stop, call)` runs the rest of
a moved call, a future of `CallOutput { text, is_error, images }`
(`CallOutput::text`, `CallOutput::error`, `with_images`), as a job. The runtimes
call it for a `Detach` call that moved, and answer the call with the job's ID.
The job is listed with the background jobs, `terminal_status` and
`terminal_control wait` read it, and `terminal_input` is refused, since a call
takes no input. When the call ends, its owner gets a note in place of the
terminal update.

```text
Job 4d2c1a9b-… ended. It ran your index_create call, which moved to the background 4 min ago. Its result follows.

Indexed 1,204 files in 3 indexes.
```

An error says `ended with an error` and `The error follows.` A result longer
than `CALL_NOTE_CHARS` (30,000 characters) is cut to its first and last parts,
with a line saying how much was left out, and the job's `output.log` holds all
of it. Images the call returned are saved in the job's directory as
`image-1.png` and so on, and the note says where they are. The job ends
`succeeded`, or `failed` for an error or a call that panicked.

Stopping the job (`terminal_control cancel`, `stop_for_user`, `shutdown`) fires
`stop`, which the runtime gives the call as its `TaskContext::stop`, and the
call has 3 seconds to return before its future is dropped. The job ends
`cancelled`, and the note says who stopped it and gives what the call returned
as it stopped, if it did. `stop_owner_watchers(owner)` stops an owner's moved
calls for a host whose agent run ended (a worker that finished), since their
notes would reach no one. The owner's commands keep running. A workflow given
the manager (`WorkflowBuilder::background_jobs`) does this for its subagents.
Neither this nor `stop_for_user` is reported to `subscribe_finished`.

A moved call's job is saved like any other, with its tool and call, and after a
restart it is an interrupted record, never run again. `interrupted_at_open()`
lists the jobs that were running in the background when the last process using
the directory stopped, or that its `shutdown` (or dropping the manager) stopped,
so a host can tell their owners. Each open lists a job once, and held jobs are
left out.

## Monitors

A monitor watches a command for its agent and alerts it, so the agent doesn't
poll. It is a job of the manager, listed with `JobKind::Monitor(MonitorInfo {
trigger, expires_at, alerts })`, and `terminal_status` reads its output.

- `MonitorOptions::lines(pattern)` runs the command in a terminal (a PTY) until
  it exits or the monitor ends, and each output line that matches `pattern`, a
  regular expression, is an alert. With `None` every line is. Lines end at `\n`
  or `\r`, so each step of a progress bar is one. They are matched without their
  escape sequences, cut at 500 bytes, and a line without its end counts once
  output has been quiet for 2 seconds. Patterns compile with a size limit, and
  the `regex` crate matches in linear time.
- `MonitorOptions::check(every_seconds, when)` runs the command as a check every
  10 to 600 seconds, with pipes and closed input. Runs never overlap. A run is
  stopped, with the processes it started, at its interval or after 60 seconds,
  whichever comes first, and one that runs over counts as a failure. The owner is
  alerted when the result changes as `CheckChange` says (`Success` when it starts
  succeeding, `Failure` when it starts failing, `Change` for either, `Output` when
  the first 4 KB of its output change), never while it stays the same. Each run
  adds a line such as `[10:44:30 UTC] exit 0, 0.1 s` and its output to the job's
  output.

`start_monitor(context, command, options)` starts one for the context's task,
after the command guard checks the command. `with_expiry` sets how long it lasts
(30 minutes by default and at most) and `with_label` what lists show instead of
the command. It returns `MonitorStarted { job_id, expires_at, summary }`, where
`summary` is what the `monitor_start` tool returns. A monitor of lines waits a
second for a command that exits at once, and a check runs its first check before
it returns (waiting up to 10 seconds), so the start gives the state that changes
count from.

```text
Monitor 9a1b6c2e-… started. It runs `curl -sf localhost:3000/health` every 15 s and alerts you when it starts succeeding, until 11:42 UTC (30 min). Its first check failed with exit code 7 in 0.1 s, with no output.
```

### The outbox

Each owner has one outbox for all its monitors. News that comes after a quiet
minute waits 2 seconds for more, then goes, and the owner gets at most one note a
minute, however many monitors it has. Each monitor's part of a note shows up to
20 lines and 2,000 bytes, counts the rest, and says where its output has them.

```text
Monitor alert. Job 7f3e9c1a-… (`npm run dev`, lines that match `error|warn`) printed 15 matching lines since 10:41:05 UTC.
  [ERROR] Failed to compile src/app.tsx
  [ERROR] Cannot find module './util'
  warn: 2 deprecations
12 more matching lines are in its output. terminal_status with after_output 18234 reads them.
It ends in 22 min.
```

A check that changed several times since its last alert is one alert that says
how often and how it is now. A monitor that matched more than 200 lines in 5
minutes adds "A narrower pattern saves turns."

The notes go to the owner's bound runtime as `InjectContext`, as terminal
updates do, so they end a `wait_for_agents` or `wait_for_message` through
`TaskContext::context_arrived`. With no runtime bound (the agent is idle), the
news waits for the owner's next runtime, `has_news(owner)` says so (and
`has_alerts(owner)` for news that should wake it), and news that should wake
the owner is announced once on `subscribe_alerts()`, as a
`JobAlert { owner, job_id, kind }`. At most one alert goes to an owner a minute.
A host starts a turn of the idle agent for it, and the runtime that turn binds
gets the news at once. A note that reached the runtime during the agent's last
request waits in the task (`Task::pending_context_injections`) for its next
turn, and announces nothing. `terminal::note_wakes(note)` says what such a note
should wake the agent for (an alert, or a warning before its monitor ends), so
the host can start that turn itself.

| News | While the agent works | When it's idle |
| --- | --- | --- |
| Matching lines, a check's change | A note | `JobAlert` (`AlertKind::Lines`, `Check`) |
| The warning before the end | A note | `JobAlert` (`AlertKind::EndsSoon`), while the user counts as active |
| The command exited by itself | A note | `subscribe_finished`, as for other jobs |
| It expired, or its owner or the user stopped it | A note | With the next turn, no alert |

### Ending and rearming

A monitor ends when its time is up, which stops its command (its job ends
`timed_out` with the error `expired`), when its command exits, or when it is
stopped (`cancel`, `stop_for_user`, `stop_owner_watchers`, `shutdown`). A monitor
armed for at least 10 minutes warns its owner 5 minutes before its end, and
`rearm(owner, job_id, lasts)` keeps it for up to 30 minutes from now and arms the
warning again.

```text
Monitor 7f3e9c1a-… (`npm run dev`) ends in 5 min, at 11:42 UTC. monitor_control rearm keeps it going for up to 30 min from now. If you no longer need it, let it end.
```

A host can tell the manager two things that shape the warnings.

- `set_prompt_cache_expiry(owner, Some(at))` says when the agent's prompt cache
  expires (its latest model request, or a refresh, plus the cache's lifetime). A
  warning that would come after that comes a minute before it instead and says
  how long is left, so the turn it starts reads the conversation from the cache.
  While a runtime bound for the owner is open (the agent works), the warning
  keeps its usual time, since the agent reads it only after its call. Monitors
  plan their warnings again each time the host says, and as a runtime binds.
- `set_user_active_at(at)` says when the user last wrote. Once the user hasn't
  written for `set_warning_wakes_within` (2 hours by default,
  `DEFAULT_WARNINGS_WAKE_WITHIN`, `None` for always), warnings no longer wake an
  idle agent and go with its next turn. Until a host says, the user counts as
  having written when the manager was made.

`set_monitor_limits(MonitorLimits)` changes the limits for monitors started from
then on.

| Limit (`MonitorLimits`) | Default |
| --- | --- |
| Running monitors per owner, and of all owners (`per_owner`, `in_all`) | 5 and 20 |
| How long a monitor lasts at most (`longest`) | 30 minutes |
| Warning (`warning_before`, for monitors armed for at least `warned_from`) | 5 minutes before the end, from 10 minutes |
| Early warning before the prompt cache expires (`cache_margin`) | 1 minute |
| Least time between notes or alerts, and the wait for more (`alert_interval`, `gather`) | 1 minute and 2 seconds |
| Lines and bytes of a monitor's news per note (`lines_per_note`, `bytes_per_note`) | 20 and 2,000 |
| Bytes kept of a line, and the quiet before a partial line counts (`line_bytes`, `partial_line_after`) | 500 bytes and 2 seconds |
| A line monitor's wait for a command that exits at once (`start_wait`) | 1 second |
| A check's interval, its runs, its output compared, its first result (`shortest_check`, `longest_check`, `check_run_limit`, `check_output_bytes`, `first_check_wait`) | 10 to 600 seconds, 60 seconds, 4,096 bytes, 10 seconds |
| A pattern's length (`pattern_chars`) | 1,000 characters |
| Matching lines that make a pattern noisy (`noisy_lines`, `noisy_window`) | 200 in 5 minutes |

A monitor is saved like any other job, with what it watches, its end and its
count of alerts. After a restart a monitor that was running is an interrupted
record, listed by `interrupted_at_open()`, and it never starts again, since its
command may start a server or have other effects. `stop_owner_watchers(owner)`
stops an owner's monitors with its moved calls. On Windows a monitor's command
must be one line, as `cmd.exe` takes one.

### The monitor tools

`MonitorTool::all(manager)` gives the agent `monitor_start` and
`monitor_control` (`MonitorTool::NAMES`), which `register_tools` registers with
the terminal tools. As with those, an agent is offered them only when its
allowlist names them.

| Tool | Behavior |
| --- | --- |
| `monitor_start` | Watch `command`'s lines that match `pattern`, or with `every_seconds` run it as a check and alert as `alert_when` says (`success`, `failure`, `change` or `output`), for `expires_minutes` (1 to 30). Returns what the monitor watches until when, with its output so far or its first check's result. |
| `monitor_control` | `list` the agent's monitors (with `job_id`, `command`, `watching`, `alerts` and `ends_in_minutes`), `rearm` one for `minutes` from now (30 by default), or `stop` one. |

A Claude Code agent's monitors run in the runtime's manager like the native
loop's. Their notes reach it with its next tool call's result (the
`PostToolUse` hook) or at the end of its turn.

## Configuration

`TerminalConfig::new(directory)` sets these defaults:

| Setting | Default |
| --- | --- |
| Buffered output per job (`output_buffer_bytes`) | 64 KiB |
| Log file size per job (`max_log_bytes`) | 4 MiB |
| Output window per status read or notice (`excerpt_bytes`) | 2,000 bytes |
| Output quiet period before prompt matching (`prompt_quiet_ms`) | 150 ms |
| Execution deadline (`default_timeout`) | None |
| Shell (`shell`, `shell_args`) | `/bin/sh -c` on Unix, `cmd.exe /D /S /C` on Windows |
| Prompt rules (`prompt_rules`) | Y/N, password/passphrase, and "press a key" prompts |
| Quiet environment for PTY jobs (`quiet_terminal_environment`) | On |
| Command guard (`command_guard`) | On, so drastic commands are refused (see [Command guard](#command-guard)) |

PTY jobs run with the host's environment plus `TERM=dumb`, `NO_COLOR=1`,
`PAGER=cat`, `GIT_PAGER=cat`, and `LANG`, `LC_CTYPE`, and `LC_ALL` set to a UTF-8
locale, `C.UTF-8` (`en_US.UTF-8` on macOS, which may lack `C.UTF-8`)
(`QUIET_TERMINAL_ENVIRONMENT`), on every platform. Programs that see a
terminal then print plain text instead of colours, progress redraws, and pagers,
which would otherwise fill the output window. Full-screen programs such as
editors, `top`, and `less` refuse to run or degrade under `TERM=dumb`. Set
`quiet_terminal_environment: false` to give PTY jobs the host's environment
unchanged. Pipe jobs always get it unchanged.

Prompt rules are named regular expressions. The number of jobs is not limited,
running or finished. Jobs run with the host process's permissions. The terminal
tools do not provide a filesystem or network sandbox.

## Command guard

`TerminalConfig::command_guard` refuses commands that would delete, move,
overwrite, or reformat something by accident. It checks `terminal_start`
commands, and `terminal_input` text too, since that can be a command typed into
a shell. A refused command starts no job. The tool returns an error that names the
path and why, so the agent can correct the command.

| Target | Default | Setting that allows it |
| --- | --- | --- |
| The working directory and everything below it, and the inside of temporary directories | Allowed | — |
| Other paths, such as `~/Documents` or a sibling repository | Refused | `allow_outside_workspace: true`, or the path (and what is below it) in `allowed_paths` |
| Protected paths, such as the filesystem root, drive roots and network-share roots, the home directory, its parents and other users' home directories, top-level directories (`/usr`, `/opt`, `D:\data`), system directories (`/etc`, `/var`, `/System`, `C:\Windows`, `C:\Program Files`), mounted volumes (`/Volumes/*`, `/mnt/*`, `/media/*`), and devices | Refused | Only `dangerously_allow_all_commands: true`, which turns the guard off |

A working directory right under a root, such as `D:\RoboEnv` or `/data`, holds its
files like any other working directory. One that is itself a protected place, such
as the home directory or `C:\Windows`, allows nothing inside it.

The guard reads the command in the shell's language, chosen from `shell`
(`ShellDialect::from_shell`, for POSIX shells, `cmd.exe`, or PowerShell). It follows
what the shell will do rather than matching command names:

- It finds the paths a command acts on, for `rm`, `rmdir`, `shred`, `find -delete` and
  `-exec rm`, `git clean -f`, `rsync --delete`, `mv` sources, `truncate`,
  `dd of=`, `tee`, recursive `chmod`/`chown`, disk tools (`mkfs`, `wipefs`, `fdisk`,
  `diskutil erase…`), `>` redirections, and on Windows `rd`, `del`, `move`,
  `format`, `diskpart`, `robocopy /MIR`, `icacls /t`, `Remove-Item` and its
  aliases, `Move-Item`, `Set-Content`, `Format-Volume`, `Clear-Disk`, and
  `[IO.Directory]::Delete`.
- A POSIX redirect to `/dev/null`, `/dev/stderr`, `/dev/fd/N` and the other standard
  pseudo files writes to no disk, and neither does one to `/dev/tcp/HOST/PORT` or
  `/dev/udp/HOST/PORT`, for which bash opens a network connection (`exec 3<>/dev/tcp/…`,
  `echo hi > /dev/tcp/…`, `&>`). Only that exact form counts, with `..` resolved first,
  so `/dev/tcp/../sda` is `/dev/sda` and is refused, as is `/dev/tcp/HOST` without a
  port, and a host or port only known when the command runs. Paths in `cmd.exe` and
  PowerShell are judged as before.
- It resolves `~`, variables from the job's environment and from assignments
  earlier in the command, `cd` and `Set-Location` (an unset `cd $DIR` goes home),
  `..`, and wildcards. An unset variable expands to nothing, as in the shell, so
  `rm -rf "$OUT"/*` with `OUT` unset is refused as a delete of `/*`. `${OUT:?}`
  makes the shell stop instead, and the guard allows it. A `cd` or assignment in
  a POSIX subshell lasts only there, in `( … )`, in a background list (`… &`),
  and in a pipeline's member other than the last, be it a command, a subshell, a
  `{ … }` group or a loop. One in a pipeline's last member leaves the directory
  or variable only known when the command runs, because bash runs that member in a
  subshell and zsh in the shell itself. A `{ … }` group, loop or `if` run in the
  shell itself keeps its `cd`.
- It follows wrappers to the command they run (`sudo` and `env` with their
  `NAME=value` assignments, `nice`, `timeout`, `setsid`, `flock`, `taskset`,
  `unshare`, `strace`, `ltrace`, `chronic`, `watch -x`), judging it from the
  directory `env -C`, `sudo -D` or `unshare -w` gives it. It follows `xargs` and
  pipelines (`find … | xargs rm`, `Get-ChildItem … | Remove-Item`, also with
  the listing in a subshell or `{ … }` group), and checks `sh -c`, `bash -c`,
  `eval`, `watch`, `su -c`, `script -c`, `flock … -c`, `cmd /c`, and
  `powershell -Command` and `-EncodedCommand` text as commands.
- In PowerShell, a path on a drive that holds no files (`Env:`, `Variable:`,
  `Function:`, `Alias:`) isn't checked, so `Remove-Item Env:\GH_TOKEN` runs. The
  registry and the certificate store (`HKCU:`, `HKLM:`, `Registry::…`, `Cert:`) and
  drives a session defines count as outside the working directory. `Temp:` is the
  temporary directory, and a `FileSystem::` path is the path it names. After a
  `cd` into a drive that holds no files, relative paths count as outside.
- `$(pwd)` is the working directory and `$(mktemp)` a new temporary path. A path
  it can't know in advance (other `$(…)` output, or a variable a sourced script
  may set) counts as outside the working directory. If that value could be empty
  and the rest of the path starts at the root, the root path is checked as well.

It guards against accidents, not against a determined agent. A script, Makefile,
or test the command runs is not inspected, and neither is text a command decodes
and pipes to a shell. Harmless commands are not refused because of what they
print, so `echo "rm -rf /"` and a heredoc containing it run.

```rust,ignore
let mut config = TerminalConfig::new(workspace.join(".terminal-jobs"));
config.command_guard.allowed_paths.push(home.join(".cache/my-app"));
// For hosts that want no guard at all:
// config.command_guard.dangerously_allow_all_commands = true;
```

`TerminalManager::set_allow_outside_workspace(allow)` changes
`allow_outside_workspace` while the manager runs, for hosts that let the user
turn outside access on and off during a session. It applies from the next
`terminal_start` command or `terminal_input` text, and
`allows_outside_workspace()` reads the current value.

`CommandGuard::check(command, dialect, working_directory)` runs the same check
for other tools that run shell commands. The `deepcoder` example's `run_shell`
uses it, with a `--dangerously-allow-all-commands` flag.

### Command rules

`CommandGuard::rules` allows or refuses programs and their arguments. There are
none by default. Each `CommandRule` has an action (`RuleAction::Allow` or
`Deny`), a pattern, and optionally a directory it applies `within` and a
`reason` for the model:

```rust,ignore
use arko_agents::terminal::CommandRule;

config.command_guard.rules = vec![
    // git only reads...
    CommandRule::allow("git status *"),
    CommandRule::allow("git log *"),
    CommandRule::allow("git diff *"),
    CommandRule::deny("git * --output* *"),
    // ...except in the scratch folder.
    CommandRule::allow("git *").within("scratch"),
    CommandRule::deny("cargo * publish *").because("Publishing is the user's."),
];
```

- A pattern is words separated by spaces. The first is the
  program, compared with the base name of the command's program, ignoring case
  and `.exe` (`/usr/bin/git` and `GIT.EXE` are `git`). Each other word matches
  one argument, in order. `*` alone matches any number of arguments, none
  included. `*` inside a word matches any text in that argument (`--output*`
  matches `--output=log.txt`). Any other word matches itself exactly, after the
  shell's quoting is removed. A pattern matches the whole command, so `git log`
  matches only `git log` with no arguments, and `git * --output*` only a command
  whose last argument starts with `--output`. Variables the guard can resolve
  are resolved, and an unquoted one splits into words as the shell splits it.
  An argument only known when the command runs (`$(…)` output) matches only `*`.
- Before matching, git's global options `-C <dir>`, `--no-pager`, `-P`
  and `--no-optional-locks` are skipped, so `git log *` matches
  `git -C repo --no-pager log`. Its other global options (`-c`, `--git-dir`,
  `--work-tree`, …) stay, so `git log *` doesn't match `git -c core.pager=… log`.
- A command that a `Deny` rule matches is refused. Otherwise, if
  its program has `Allow` rules, it runs only if one of them matches. If not,
  the rules allow it. The path checks apply either way.
- A rule with `within` applies only to a command run in that
  directory or below it, following `cd` and git's `-C` in the same command line
  (the guard checks one line at a time, from the job's working directory, and a
  Claude Code agent's `Bash` calls each start there too, since arko starts the
  CLI with `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1`). A `cd` in a subshell
  lasts only there, so after `(cd scratch && cargo test); git commit -m x` the
  commit runs in the working directory. A relative `within` is relative to the
  job's working directory. A command whose directory isn't known
  is never within one, as after `cd` to a value only known when it runs, with
  `env -C` or `unshare -w`, or for git with `--git-dir`, `--work-tree`, or one
  of `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_COMMON_DIR`,
  `GIT_OBJECT_DIRECTORY` and `GIT_ALTERNATE_OBJECT_DIRECTORIES` set in the
  environment or the command line (also through `env`, `sudo` or `strace -E`),
  whatever `-C` follows (git then works on that repository, work tree, index or
  object store). An `Allow` rule
  with `within` still puts its program on an allow-list everywhere, so with the
  rules above, git runs fully in `scratch` and only reads elsewhere.
- The rules apply to every command the guard finds in the line. That means each
  pipeline and list item, the command behind wrappers such as `sudo`, `env`, `nice`,
  `timeout`, `nohup` and the others above, commands in `sh -c`, `bash -c`, `eval`,
  `watch`, `su -c`, `script -c`, `cmd /c` and `powershell -Command`, commands in
  `$(…)` and backticks (in PowerShell, `$(…)`, `@(…)` and `(…)`), and the commands
  `xargs` and `find -exec` run. Words that only mention a program are not commands, so under
  a git allow-list `ls /usr/share/doc/git`, `grep -rn git .`, `echo git push`
  and a heredoc that contains `git push` run. In PowerShell, a statement that
  starts with a quoted or computed word runs a command only after `&`. Without
  it (`$files | Measure-Object`) it is an expression.
- While there are rules, a command whose
  program is only known when it runs (`$(which git) push`, `& $tool push`), or
  shell text that is (`bash -c "$(cat script.sh)"`, `eval "$(…)"`), is refused,
  since the rules can't check it. Without rules these run as before.
- Deny rules guard against accidents. They match the words the guard reads,
  so an argument only known when the command runs doesn't match a deny rule's
  literal words, so `cargo publish *` doesn't refuse `cargo $(printf publish)`, or
  `sh -c 'cargo "$1"' _ publish`. An allow-list is the stronger form, because such an
  argument matches only `*`, so with `cargo build *` and `cargo test *` allowed,
  both are refused.
- A refusal names the command and the rule, or for an allow-list
  the patterns allowed for that program, followed by the rule's `reason`
  (without one, a note to do the work another way). Its `target` is the
  command, and `protected` is false. `dangerously_allow_all_commands` turns the
  rules off too.

Like the rest of the guard, rules guard against accidents. A script, Makefile or
program a command runs can still run anything. A Claude Code agent's `Bash` and
`PowerShell` get the same rules from `ClaudeCodeSettings::command_rules`.

## Input requests

A snapshot has an execution `state` and an optional `input` notice:

- `input: null` means no input request is known. Input can still be sent at any time.
- `input.source.kind: "program"` means the program explicitly declared a request
  (see [Program input requests](#program-input-requests)). The reply carries its
  `request_id`.
- `input.source.kind: "prompt_rule"` means a configured rule matched the terminal
  output. This means input is *possibly* needed. Rules can miss prompts and can
  match output that is not a question.

Every `terminal_input` call supplies the `revision` of the latest snapshot. Output,
new or withdrawn requests, and accepted input advance the revision, and the host
rejects stale or duplicate submissions. For ordinary terminal input, the program
can still change between the observation and the write, because revision checks don't
make a terminal interaction atomic. Programs using explicit requests must check
the request ID of each reply before applying it.

### Why prompt detection is advisory

Whether a process is waiting for terminal input cannot be determined reliably
from outside it. The standalone Linux probe
[`interactive_probe.py`](../arko_agents/tests/fixtures/interactive_probe.py)
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
python3 arko_agents/tests/fixtures/interactive_probe.py
```

## Reading output

Responses contain a bounded window of recent output with byte offsets. Pass the
previous `output.end` as `after_output` to read only new bytes. `available_end` is
the end of the buffered output, so more bytes remain when it is greater than
`end`. `gap` reports that part of the requested range has already left the
buffer. When the window contains invalid or split UTF-8, `raw_hex` holds the exact
bytes alongside the display text. `prompt_truncated` marks a shortened inferred
prompt.

Each job's `output.log` (at `log_path`) keeps the combined output in arrival order
up to `max_log_bytes`. `log_truncated` reports that the limit was reached. In PTY
mode stdout and stderr share one terminal, and Windows ConPTY emits a rendered
terminal stream. Use pipe mode when the original stdout and stderr bytes matter.
On Windows, the text a PTY job's status and notices show the agent has its
escape sequences removed, since ConPTY adds cursor moves, window titles, and
colour resets whatever the program prints. Cursor-forward moves become the
spaces they stand for. The byte offsets (`start`, `end`, `available_end`,
`after_output`), `raw_hex`, and `output.log` keep the raw output.
The prompt matcher handles line prompts and ANSI decoration, not full-screen
terminal interfaces.

Because a response holds a bounded window and the full output stays in `output.log`,
the terminal is the way to read large files and outputs, with `wc -l` for the size,
`grep -n` to find what is needed, `head` or `sed -n 'a,bp'` for a range, or a short
script that prints only what matters. Printing a whole large file costs one window,
never the context. An agent offered both `read_file` and these tools sees a sentence in
`read_file`'s description pointing it here for large or generated files, and when a
result of another tool is too large for the model's context, arko_agents' overflow
guard names these commands in the notice of the shortened result (see
[Large tool results](context-efficiency.md#large-tool-results)).

## Program input requests

A program, or an adapter wrapping it, can declare an input request without
printing anything. The child process receives two file paths in its environment:

- `ARKO_INPUT_REQUEST_PATH` is where it appends newline-delimited UTF-8 JSON requests.
- `ARKO_INPUT_REPLY_PATH` is where it reads newline-delimited UTF-8 JSON replies.

Both terminal modes use this append-only file protocol. Each stream has a single
writer, which appends complete lines and flushes them. Readers keep partial lines
and treat end-of-file as "no new bytes yet". The host polls the request stream
every 20 ms, without a model call or a network listener. Keep the manager's
directory on a local filesystem.

This example uses the [`Replies` reader](../arko_agents/tests/fixtures/control_client.py)
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

Declare one outstanding request at a time and use a new ID for each question.
Repeated IDs are deduplicated for the lifetime of the job. To withdraw a question,
send `{"type":"input_cancelled","request_id":"…"}`. Check reply IDs even when
withdrawing requests, because a reply can cross a cancellation. Answers to
program requests go to the reply stream. Ordinary commands keep receiving input
through stdin or the PTY. JSON printed to stdout never creates a program request.

The limits are 8 KiB per request line, 128 bytes per request ID, 4 KiB per prompt, 1,024
distinct requests per job, and 8 MiB per control stream. The request and reply
files are kept with the output log. Never truncate or replace an active stream.
A malformed or conflicting declaration stops the job with a diagnostic. A failed
or partial input write also stops the job, because retrying it could deliver an
answer twice.

## Notifications and lifecycle

Input requests and job completion produce context messages for the owning task.
Each job has at most one pending delivery, which waits for channel capacity
without blocking the job. A newer question replaces an undelivered older one.
Unchanged prompts and routine output do not produce further notifications. Status
reads remain available while output grows. Rebinding an active manager to a
resumed runtime can replay the latest notices. Use the job ID and revision to
recognize updates. A notice doesn't start a model run by itself, and the host decides
whether an idle or completed agent continues.

`manager.shutdown().await` cancels owned processes, waits for their cleanup (up to
3 seconds, and 4 for a moved call), and prevents new starts. The next open lists the
background jobs it stopped with `interrupted_at_open()`. Dropping the manager requests cancellation. The directory
stays locked until the workers finish. A wait timeout is independent of a job's
execution deadline. PTY input handles control characters according to the
program's terminal settings. Terminal resizing and closing stdin in pipe mode are
not supported.

After a host restart, saved running jobs are marked `interrupted` and pending
input is cleared. Saved process IDs are never used to reattach to, kill, or rerun
a command. Job records and logs persist. The live output buffer and notification
queue do not. A resumed host should list and reconcile jobs before relying on
them.

## Platform behavior

| Area | Linux / macOS | Windows |
| --- | --- | --- |
| Interactive transport | PTY | ConPTY |
| Default shell | `/bin/sh -c` | `cmd.exe /D /S /C` |
| Enter for ordinary terminal input | `\n` | `\r` |
| Cancellation | Owned process group | Job Object, assigned at process creation |
| Pipe mode | Piped stdin, stdout, and stderr | Piped stdin, stdout, and stderr, in a console with no window |
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
possible rather than certain input requests. Where an agent may meet large files
(data, logs, generated code), ask it to inspect them with the terminal in parts
(`wc -l`, `grep -n`, `sed -n '100,200p'`) rather than reading them whole.

## Testing

The integration tests run real shell and Python processes. They cover both
transports, several prompt styles, request and reply IDs, repeated and withdrawn
questions, stale and duplicate input, bounded and non-UTF-8 output, ownership,
backpressure, cancellation and deadlines, process-group cleanup, restart records,
and automatic runtime binding, and monitors' alerts, checks, warnings and ends.
CI runs them on Linux, macOS, and Windows. The monitors' timing tests run on
tokio's paused clock.

```sh
cargo test -p arko_agents --lib terminal:: --locked
cargo run -p arko_agents --example interactive_terminal --locked
```

The example answers a fixed Y/N fixture with a predetermined answer and needs no
model or credentials. The fixtures and the example require Python 3.
`ARKO_TEST_PYTHON` selects the interpreter. The library itself does not require
Python.
