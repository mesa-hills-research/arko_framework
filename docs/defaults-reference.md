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
credits or quota run out; credits keep an account serving), checks the accounts above the
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
