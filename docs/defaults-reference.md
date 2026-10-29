# Defaults to know

This page collects the defaults that most change what agents do. The
[reference map](reference-map.md) lists every public setting with its default and
omitted-field behaviour, the opt-ins, persistence compatibility, the code location, and
links to the crate guides.

These defaults change what an agent sees of its tool results, in arko's own runtime:

- The overflow guard is on (`ToolResultLimits::overflow_guard`). It shortens a turn's tool
  results only when the next request would certainly be over the model's input limit
  (a lower bound on it, counting what the turn added at 10 bytes per token, is over the
  limit), and if the
  provider still finds a request too long it shortens the largest new result and resends,
  at most twice. See [Tool result size](reference-map.md#tool-result-size-arko_agents).
- Caps are opt-in, so `ToolResultLimits::max_result_tokens` and
  `PerToolConfigOverride::max_result_tokens` are `None`. So are `read_file` line ranges
  (`AgentToolConfig::read_file_line_ranges`, `false`).
- A native reply in plain text, without a tool call, goes through error recovery with a
  note asking for a tool call. `AgentToolConfig::text_reply_as_message` (`false`) sends it
  to the user as a message instead. The built-in `message_user` tool, offered when listed,
  sends a message either way.
- `AgentToolConfig::text_reply_ends_task` (`false`) makes that reply the agent's final
  reply instead. It completes the task with the text as the summary, and the agent isn't
  offered `complete_task`. It works in native mode and on Claude Code models, and is
  refused together with `text_reply_as_message` and on agents a completion verifier
  reviews or runs on. See [Tools](reference-map.md#tools-arko_agents).
- `AgentToolConfig::chat` (`false`) makes the agent a chat, with no tools. Its reply in text ends
  its task, a workflow adds none of its own tools (messaging, memory, subagents, registered
  tools) and no list of recipients, and a handover asks for a summary in text. Only an agent
  set this way is a chat, and one that merely lists no tools keeps the workflow's tools.
- Error kinds (`arko_ai::ErrorKind`) are recorded with pauses and failures, and stop
  nothing by themselves.
- No call moves to the background unless the runtime has a terminal manager
  (`AgentRuntime::with_background_jobs`, `WorkflowBuilder::background_jobs`) and the
  host moves it. Calls move by themselves only after the manager's
  `set_background_after`, `None` by default. A moved call keeps its host time limit. See
  [Moving calls to the background](arko_agents-guide.md#moving-calls-to-the-background).
- Monitors (`monitor_start`, `TerminalManager::start_monitor`) are offered only when an
  agent's allowlist names their tools. An agent gets at most one note of monitor news a
  minute, however many monitors it has, and a monitor lasts at most 30 minutes unless the
  agent rearms it. Its warning before the end wakes an idle agent, before its prompt
  cache expires if the host says when that is, and only for 2 hours after the user last
  wrote (`set_warning_wakes_within`). See [Monitors](interactive-terminals.md#monitors).
- An `McpHub` runs its own process of each MCP server (`McpServerConfig::shared`,
  `false`), gives a server 60 s to start and list its tools, and lets workers reach it
  through the `mcp` tool (`McpServerConfig::workers`, `true`). See
  [The MCP hub and the mcp tool](arko_agents-guide.md#the-mcp-hub-and-the-mcp-tool).
- Skills are listed for every model and every agent, unless a skill's `models` or
  `workers: false` says otherwise. `SkillsConfig::model_names` is empty (no model filter),
  `subagent` is `false` (a workflow's subagent runs still never load a `workers: false`
  skill), `only` is `None` and `offer_when_empty` is `false`, so a conversation that starts
  with no skill has no `load_skill`. `load_skill` reads the skill sources at each call, and
  a missing `SkillSource` directory warns unless it is `optional`. See
  [Skills](arko_agents-guide.md#skills).

A client signs in to one ChatGPT account unless `LLMClient::with_chatgpt_accounts`
installs an ordered list. The list fails over only when the backend refuses an account (a usage limit, or
credits or quota run out, while credits keep an account serving), checks the accounts above the
serving one at most every 5 minutes, and never runs in the background. An account follows its
Codex file to another ChatGPT account only after `with_file_following`, and `set_accounts` edits
the list in place. See
[Client and credentials](reference-map.md#client-and-credentials-arko_ai).

Operator notes and graceful stops have these defaults:

- Operator notes do nothing until a note is posted, and the control folder through which
  another process posts them is off (`WorkflowBuilder::control_folder`). Completion
  verifiers' reviews get only the notes that name them, and no agent has a reply hint
  (`WorkflowBuilder::notes_reply_hint`). See
  [Builder defaults](reference-map.md#builder-defaults-arko_workflows).
- A stop (`Workflow::stop`) has no wrap-up unless its `StopRequest` asks for one, and
  nothing is timed until a stop is under way. A `StopAwareCommand`'s graces (10 s, then
  5 s after `SIGTERM`) start only then. See
  [Stopping tool calls](reference-map.md#stopping-tool-calls-arko_agents).
- A stop of one agent run (`Workflow::stop_run`, `stop_agent`) takes the same request,
  without a wrap-up unless it asks for one, and an intent with no default
  (`RunStopIntent::Retire` or `Relaunch`). A stop request in the control folder
  (`notes::control::request_stop`) defaults to an operator's stop without a wrap-up that
  retires the runs. The workflow reads the requests every second while agents run, and
  only those written after it was built. `runctl stop` asks for a 600-second wrap-up.

Defaults of background subagents (`WorkflowBuilder::subagents`, off until set), detailed in
[Background subagents](reference-map.md#background-subagents-arko_workflows):

- At most 4 run at once (`max_running`), and starts over that wait in a queue.
- Subagents start no subagents (`max_depth` 1), and callers keep `run_workflow_agent`.
  With a higher `max_depth`, every subagent of a caller agent under that depth gets the
  tools that start runs, unless the `SubagentHook`'s `offers_start_tools` says no.
- Stopping a subagent stops the subagents it started and theirs in turn, with the same
  wrap-up, and a subagent that ends stops the ones it still has (a two-minute wrap-up).
- A subagent can't report while its own subagents run. Its `complete_task` is refused up
  to five times in a row, then accepted and the subagents are stopped. While it waits for
  them in `wait_for_agents` it holds no `max_running` place.
- A resume stops a subagent whose own lead won't run again instead of relaunching it.
- A lead may link its subagents as peers, who may then send each other 20 messages
  between two of the lead's (`max_peer_messages`). The lead gets no copies unless
  `copy_peer_messages_to_lead` is set.
- A subagent's `ask_user` goes to its lead (`questions_to_lead`), and waits for the answer.
- `ask_user` waits for the answer. `AgentToolConfig::async_questions` lets the agent pass
  `wait: false` to go on without it, for a host that delivers answers later.
- A subagent recovers from failed model requests without a limit, like the run the host
  drives. `max_failed_requests` makes it fail once that many requests have failed in a row.
- A `workspace` must be inside the caller's directory (`workspaces_anywhere` lifts this),
  and `isolated` is offered only with a `WorkspaceProvider`.
- Workflow-service calls over `max_active_calls` (4) fail at once unless
  `queue_active_calls` is set.

arko_index has these defaults (see [Defaults (arko_index)](reference-map.md#defaults-arko_index)):

- Every regular file under the root is indexed, hidden and gitignored files included.
  Ignore files are read only with `FileRules::respect_ignore` (`--respect-ignore`), and
  never above the root. `.git` directories and files, `CACHEDIR.TAG` directories
  and the tool's state directories are always skipped.
- Binary, non-UTF-8 and unreadable files are skipped and listed in the report, and the
  walk goes on.
- Chunking, `index2` builds and the inotify watcher (`watch2`) use every available CPU
  (`ARKO_INDEX_THREADS` overrides), with identical output at any thread count. The JSON
  index (`search::index`) and the polling watcher use one thread.
- Dense (embedding) search is opt-in, and an index is lexical unless a build sets
  `BuildOptions::dense` (CLI `index2 --embed MODEL`). The library makes no network
  calls and reads no credentials. The examples embed with Voyage through arko_ai
  (`VOYAGE_API_KEY`), or with a local model on the CPU (feature `local-embed`, off by
  default, where `--embed local` picks nomic-ai/CodeRankEmbed, runs it in int8 (`local:f32`
  for full precision) and downloads it into the Hugging Face cache on first use). GPUs
  need `local-embed-cuda` or `local-embed-gpu` (off). The device is `auto`, so int8 stays
  on the CPU, f32 takes an NVIDIA GPU through CUDA or a discrete GPU through wgpu when
  the build and machine have one, and a GPU runs f32 unless f16 or bf16 is named. Once chosen, the model is saved with the index
  (`STATE_DIR/dense.json`), and the CLI's later `index2` builds follow it (a library
  caller passes `DenseBuild::from_config` with `DenseConfig::load`). Embeddings are
  cached by model and text, so a rebuild embeds only changed chunks. The embedded text is the path, the
  scope and the source (`EmbedText::PathScopeSource`, CLI `--embed-text`).
- A build publishes its generation for lexical search before it embeds. The library's
  `index2::build` then embeds before it returns (`DenseBuild::embed_later` leaves it to
  `index2::embed`), and the CLI's `index2` returns at once and embeds in a background
  process (`--wait-embeddings` waits). That process ends, after its current request, when
  the program that ran `index2` ends, its session leader on Unix or its parent on Windows
  (`--owner-pid PID` names another owner, `0` none). Dense and hybrid search use the
  vectors stored so far and report the coverage. A newer generation takes over from a run in progress,
  and what a run embedded comes from the cache in the next one.
- There's no token cap by default (`DenseBuild::max_tokens`, `--max-embed-tokens`), so set
  one for unattended builds. The cap applies to each embedding run. At the cap a run stops,
  the generation stays searchable, and the run reports what is left. Runs and searches
  report the tokens billed.
- Searches return ten results. The library's `SearchRequest` is lexical by default,
  and the CLI's `search2` is hybrid when the index has embeddings. Hybrid fusion is
  reciprocal rank fusion (k = 60, equal weights) over 50 candidates per side.
- Dense search scans every vector's int8 codes (1 byte per dimension, held in memory)
  and rescores the best candidates with the full-precision vectors, so scores are
  exact (`Quantization::Int8`, `VectorIndex::Flat`). F32 codes (`--quantize f32`,
  exact, 4 bytes per dimension) and Vamana graphs (`--vector-index graph`, for very
  large indexes) are opt-in. Embedding runs update the vector store in place, adding and
  removing only changed chunks' vectors.
- Chunks carry a context header (`scope`, the enclosing definitions or Markdown headings),
  added in chunk JSON format 2 without changing boundaries or IDs. Lexical search indexes a
  chunk's path and source, and its scope only with `BuildOptions::lexical_scope`
  (`index2 --lexical-scope`).
- 53 languages are parsed and chunked along their structure, and anything else is split
  by the bounded fallback. A Markdown file's front matter is file metadata (chunk format 3
  records it), and its title, without the description, roots every chunk's scope
  (`Options::front_matter_description` adds the description). See
  [arko_index languages](arko_index-languages.md).
