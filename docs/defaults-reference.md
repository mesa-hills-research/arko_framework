# Defaults and reference map

The complete defaults reference is now part of the [arko_framework reference map](reference-map.md).
It covers every public setting, its default and omitted-field behavior, opt-ins, persistence
compatibility, code location, and links to the detailed crate guides.

Defaults that change what an agent sees of its tool results, in arko's own runtime:

- The overflow guard is on (`ToolResultLimits::overflow_guard`). It shortens a turn's tool
  results only when the next request would certainly be over the model's input limit
  (a lower bound on it, counting what the turn added at 10 bytes per token, is over the
  limit), and if the
  provider still finds a request too long it shortens the largest new result and resends,
  at most twice. See [Tool result size](reference-map.md#tool-result-size-arko_agents).
- Caps are opt-in: `ToolResultLimits::max_result_tokens` and
  `PerToolConfigOverride::max_result_tokens` are `None`. So are `read_file` line ranges
  (`AgentToolConfig::read_file_line_ranges`, `false`).
- Error kinds (`arko_ai::ErrorKind`) are recorded with pauses and failures, and stop
  nothing by themselves.

ChatGPT sign-ins: one account unless `LLMClient::with_chatgpt_accounts` installs an ordered
list. The list fails over only when the backend refuses an account (a usage limit, or
credits or quota run out, while credits keep an account serving), checks the accounts above the
serving one at most every 5 minutes, and never runs in the background. See
[Client and credentials](reference-map.md#client-and-credentials-arko_ai).

Defaults of operator notes and graceful stops:

- Operator notes do nothing until a note is posted, and the control folder through which
  another process posts them is off (`WorkflowBuilder::control_folder`). Completion
  verifiers' reviews get only the notes that name them. See
  [Builder defaults](reference-map.md#builder-defaults-arko_workflows).
- A stop (`Workflow::stop`) has no wrap-up unless its `StopRequest` asks for one, and
  nothing is timed until a stop is under way: a `StopAwareCommand`'s graces (10 s, then
  5 s after `SIGTERM`) start only then. See
  [Stopping tool calls](reference-map.md#stopping-tool-calls-arko_agents).
- A stop of one agent run (`Workflow::stop_run`, `stop_agent`) takes the same request,
  without a wrap-up unless it asks for one, and an intent with no default
  (`RunStopIntent::Retire` or `Relaunch`). A stop request in the control folder
  (`notes::control::request_stop`) defaults to an operator's stop without a wrap-up that
  retires the runs. The workflow reads the requests every second while agents run, and
  only those written after it was built. `runctl stop` asks for a 600-second wrap-up.

Defaults of arko_index (see [Defaults (arko_index)](reference-map.md#defaults-arko_index)):

- Every regular file under the root is indexed, hidden and gitignored files included.
  Ignore files are read only with `FileRules::respect_ignore` (`--respect-ignore`), and
  never above the root. Built in: `.git` directories and files, `CACHEDIR.TAG` directories
  and the tool's state directories are skipped.
- Binary, non-UTF-8 and unreadable files are skipped and listed in the report, and the
  walk goes on.
- Chunking, `index2` builds and the inotify watcher (`watch2`) use every available CPU
  (`ARKO_INDEX_THREADS` overrides), with identical output at any thread count. The JSON
  index (`search::index`) and the polling watcher use one thread.
- Dense (embedding) search is opt-in: an index is lexical unless a build sets
  `BuildOptions::dense` (CLI `index2 --embed MODEL`). The library makes no network
  calls and reads no credentials. The examples embed with Voyage through arko_ai
  (`VOYAGE_API_KEY`), or with a local model on the CPU (feature `local-embed`, off by
  default: `--embed local` picks nomic-ai/CodeRankEmbed and downloads it into the
  Hugging Face cache on first use). Once chosen, the model is saved with the index
  (`STATE_DIR/dense.json`), and the CLI's later `index2` builds follow it (a library
  caller passes `DenseBuild::from_config` with `DenseConfig::load`). Embeddings are
  cached by model and text, so a rebuild embeds only changed chunks. The embedded text is the path, the
  scope and the source (`EmbedText::PathScopeSource`, CLI `--embed-text`).
- A build publishes its generation for lexical search before it embeds. The library's
  `index2::build` then embeds before it returns (`DenseBuild::embed_later` leaves it to
  `index2::embed`), and the CLI's `index2` returns at once and embeds in a background
  process (`--wait-embeddings` waits). That process ends, after its current request, when
  the program that ran `index2` ends: its session leader on Unix, its parent on Windows
  (`--owner-pid PID` names another owner, `0` none). Dense and hybrid search use the
  vectors stored so far and report the coverage. A newer generation takes over from a run in progress,
  and what a run embedded comes from the cache in the next one.
- No token cap by default (`DenseBuild::max_tokens`, `--max-embed-tokens`): set one
  for unattended builds. The cap applies to each embedding run. At the cap a run stops,
  the generation stays searchable, and the run reports what is left. Runs and searches
  report the tokens billed.
- Searches return ten results. The library's `SearchRequest` is lexical by default,
  and the CLI's `search2` is hybrid when the index has embeddings. Hybrid fusion is
  reciprocal rank fusion (k = 60, equal weights) over 50 candidates per side.
- Dense search scans every vector's int8 codes (1 byte per dimension, held in memory)
  and rescores the best candidates with the full-precision vectors, so scores are
  exact (`Quantization::Int8`, `VectorIndex::Flat`). F32 codes (`--quantize f32`,
  exact, 4 bytes per dimension) and Vamana graphs (`--vector-index graph`, for very
  large indexes) are opt-in. Embedding runs update the vector store in place: only
  changed chunks' vectors are added and removed.
- Chunks carry a context header (`scope`: enclosing definitions or Markdown headings),
  added in chunk JSON format 2 without changing boundaries or IDs. Lexical search indexes a
  chunk's path and source, and its scope only with `BuildOptions::lexical_scope`
  (`index2 --lexical-scope`).
- 50 languages are parsed and chunked along their structure, and anything else is split
  by the bounded fallback. A Markdown file's front matter is file metadata (chunk format 3
  records it), and its title, without the description, roots every chunk's scope
  (`Options::front_matter_description` adds the description). See
  [arko_index languages](arko_index-languages.md).
