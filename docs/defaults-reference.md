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

Defaults of operator notes and graceful stops:

- Operator notes do nothing until a note is posted, and the control folder through which
  another process posts them is off (`WorkflowBuilder::control_folder`). Completion
  verifiers' reviews get only the notes that name them. See
  [Builder defaults](reference-map.md#builder-defaults-arko_workflows).
- A stop (`Workflow::stop`) has no wrap-up unless its `StopRequest` asks for one, and
  nothing is timed until a stop is under way: a `StopAwareCommand`'s graces (10 s, then
  5 s after `SIGTERM`) start only then. See
  [Stopping tool calls](reference-map.md#stopping-tool-calls-arko_agents).

Defaults of arko_index (see [Defaults (arko_index)](reference-map.md#defaults-arko_index)):

- Every regular file under the root is indexed, hidden and gitignored files included.
  Ignore files are read only with `FileRules::respect_ignore` (`--respect-ignore`), and
  never above the root. Built in: `.git` and `CACHEDIR.TAG` directories and the tool's
  state directories are skipped.
- Binary, non-UTF-8 and unreadable files are skipped and listed in the report; a walk
  never stops for them.
- Chunking, indexing and watching use every available CPU (`ARKO_INDEX_THREADS`
  overrides), with identical output at any thread count.
- Dense (embedding) search is opt-in: an index is lexical unless a build sets
  `BuildOptions::dense` (CLI `index2 --embed MODEL`). The library makes no network
  calls and reads no credentials; the examples embed with Voyage through arko_ai
  (`VOYAGE_API_KEY`). Once chosen, the model is saved with the index
  (`STATE_DIR/dense.json`) and later builds follow it; embeddings are cached by model
  and text, so a rebuild embeds only changed chunks.
- No token cap by default (`DenseBuild::max_tokens`, `--max-embed-tokens`): set one
  for unattended builds. At the cap a build stops embedding, still publishes, and
  reports what is left. Builds and searches report the tokens billed.
- Searches return ten results. The library's `SearchRequest` is lexical by default;
  the CLI's `search2` is hybrid when the index has embeddings. Hybrid fusion is
  reciprocal rank fusion (k = 60, equal weights) over 50 candidates per side.
  Vectors are stored as f32 (`Quantization::Int8` opt-in).
- Chunks carry a context header (`scope`: enclosing definitions or Markdown headings),
  chunk JSON format 2; boundaries and IDs are unchanged.
