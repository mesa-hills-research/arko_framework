use crate::claude_code::ClaudeCodeSettings;
use crate::error::ArkoAgentsError;
use crate::handover::HandoverConfig;
use crate::parsing::ToolParsingConfig;
use crate::skill::SkillsConfig;
use crate::task::Task;
use arko_ai::{
    GenerationConfig as LLMGenerationConfig, Model as LLMModel, Provider as LLMProvider,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Specifies where environment details should be placed in prompts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum EnvironmentDetailPlacement {
    #[default]
    SystemPrompt,
    InitialTaskPrompt,
    EachUserMessage,
}

/// Configuration for a built-in environment detail. A field omitted in
/// JSON or YAML takes its [`Default`] value (`include: false`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BuiltInEnvironmentDetail {
    #[serde(default)]
    pub include: bool,
    #[serde(default)]
    pub placement: EnvironmentDetailPlacement,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
}

/// How enabled file-tree context is delivered to the agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileTreeMode {
    /// Full first snapshot, then visible changes at model-call boundaries.
    #[default]
    Diff,
    /// Full tree at the configured environment-detail placement.
    Full,
}

/// Configuration for all built-in environment details.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentDetailsConfig {
    #[serde(default)]
    pub current_working_directory: BuiltInEnvironmentDetail,
    #[serde(default)]
    pub file_tree: BuiltInEnvironmentDetail,
    #[serde(default)]
    pub file_tree_mode: FileTreeMode,
    #[serde(default)]
    pub current_datetime: BuiltInEnvironmentDetail,
    #[serde(default)]
    pub operating_system: BuiltInEnvironmentDetail,
}

type DynProviderFn = Arc<dyn Fn(&Task) -> Result<String, String> + Send + Sync>;

/// Provider for dynamic content in environment details.
#[derive(Clone)]
pub enum DynamicContentProvider {
    Static(String),
    Dynamic(DynProviderFn),
}

impl std::fmt::Debug for DynamicContentProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynamicContentProvider::Static(s) => f.debug_tuple("Static").field(s).finish(),
            DynamicContentProvider::Dynamic(_) => f.debug_tuple("Dynamic").field(&"<Fn>").finish(),
        }
    }
}

/// A developer-defined environment detail that can be injected into prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeveloperDefinableEnvironmentDetail {
    pub name: String,
    pub placement: EnvironmentDetailPlacement,

    #[serde(skip)]
    pub content_provider: Option<DynamicContentProvider>,

    #[serde(alias = "content", skip_serializing_if = "Option::is_none")]
    pub static_content_for_serde: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_provider_name_for_serde: Option<String>,
}

impl DeveloperDefinableEnvironmentDetail {
    /// Creates a new static environment detail.
    pub fn new_static(
        name: String,
        content: String,
        placement: EnvironmentDetailPlacement,
    ) -> Self {
        Self {
            name,
            placement,
            content_provider: Some(DynamicContentProvider::Static(content.clone())),
            static_content_for_serde: Some(content),
            dynamic_provider_name_for_serde: None,
        }
    }

    /// Why this detail has no content to give, if it has none: a dynamic
    /// detail loaded from a saved task or configuration whose provider was
    /// not set again (providers are not saved), or a detail with neither
    /// content nor a provider. The runtime refuses to start a task with such
    /// a detail.
    pub fn missing_content_error(&self) -> Option<String> {
        if self.content_provider.is_some() || self.static_content_for_serde.is_some() {
            return None;
        }
        Some(match &self.dynamic_provider_name_for_serde {
            Some(provider) => format!(
                "Environment detail '{}' needs its dynamic content provider '{}', which was not set: providers are not saved with a task or configuration, so set `content_provider` again before running the task.",
                self.name, provider
            ),
            None => format!(
                "Environment detail '{}' has no content: give it static content or a content provider.",
                self.name
            ),
        })
    }

    /// Creates a placeholder for a dynamic environment detail that will be rehydrated by the application.
    pub fn new_dynamic_placeholder(
        name: String,
        provider_name: String,
        placement: EnvironmentDetailPlacement,
    ) -> Self {
        Self {
            name,
            placement,
            content_provider: None,
            static_content_for_serde: None,
            dynamic_provider_name_for_serde: Some(provider_name),
        }
    }
}

/// Policy for handling paths that are outside the workspace directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum PathSafetyPolicy {
    #[serde(rename = "AutoApprove")]
    AutoApprove,
    #[serde(rename = "RequireApproval")]
    RequireApproval,
    #[default]
    #[serde(rename = "AutoReject")]
    AutoReject,
}

/// Per-tool configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(from = "PerToolConfigOverrideRepr")]
pub struct PerToolConfigOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_approve_tool_use: Option<bool>,
    /// Whether this tool's calls may run at the same time as other
    /// parallel-safe calls of a response, instead of the tool's own
    /// [`ToolDefinition::parallel_safe`](crate::tool::ToolDefinition::parallel_safe);
    /// see [`AgentToolConfig`]. A configuration saved with the former
    /// `allow_batched_calls` (which let a tool's calls share a response,
    /// run one after another) loads `false` here and ignores `true`, so no
    /// old setting makes calls run at the same time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_safe: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_of_workspace_path_policy: Option<PathSafetyPolicy>,
    /// This tool's host-side time limit instead of
    /// [`AgentToolConfig::tool_timeout_seconds`]: `Some(Some(seconds))`, or
    /// `Some(None)` (JSON `null`) for none. It applies to the tools that are
    /// otherwise exempt, such as the terminal tools, too. `None` (the field
    /// left out) uses the agent's setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_timeout_seconds: Option<Option<u64>>,
}

/// The serialized form of [`PerToolConfigOverride`], which also reads a
/// setting of older versions.
#[derive(Deserialize)]
struct PerToolConfigOverrideRepr {
    #[serde(default)]
    auto_approve_tool_use: Option<bool>,
    #[serde(default)]
    parallel_safe: Option<bool>,
    #[serde(default)]
    allow_batched_calls: Option<bool>,
    #[serde(default)]
    out_of_workspace_path_policy: Option<PathSafetyPolicy>,
    #[serde(default, deserialize_with = "deserialize_explicit_option")]
    tool_timeout_seconds: Option<Option<u64>>,
}

impl From<PerToolConfigOverrideRepr> for PerToolConfigOverride {
    fn from(repr: PerToolConfigOverrideRepr) -> Self {
        Self {
            auto_approve_tool_use: repr.auto_approve_tool_use,
            parallel_safe: repr
                .parallel_safe
                .or(repr.allow_batched_calls.filter(|allowed| !allowed)),
            out_of_workspace_path_policy: repr.out_of_workspace_path_policy,
            tool_timeout_seconds: repr.tool_timeout_seconds,
        }
    }
}

/// A field that can be left out (`None`), set to `null` (`Some(None)`), or
/// set to a value (`Some(Some(value))`).
fn deserialize_explicit_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// Specifies how tool calls are communicated between the agent and the LLM.
///
/// # Native (default)
///
/// Tool definitions are sent via the LLM API's native tool calling feature.
/// The model responds with structured tool call objects rather than text
/// markup, which is what current models are trained for.
///
/// Key behavioral differences from output-based mode:
/// - One response can call several *different* tools (output-based mode
///   restricts batches to the same tool).
/// - XML usage examples are not sent to the model; only the tool description
///   and JSON Schema parameters are provided.
/// - Content wrapped in `{{#output_based_only}}...{{/output_based_only}}`
///   markers in `instructions` or `role_description` is stripped.
///
/// # Output-based (legacy)
///
/// The model outputs tool calls as XML-like structured text which the
/// framework parses. It works with models without native tool calling.
///
/// A configuration that doesn't name a mode gets `Native`. The configuration
/// snapshot saved with a [`Task`] always names its mode; one saved before
/// the mode was recorded ran output-based, and loads that way.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolCallingMode {
    OutputBased,
    #[default]
    Native,
}

thread_local! {
    /// Set while a task's configuration snapshot is deserialized; see
    /// [`deserialize_task_agent_config`].
    static LOADING_TASK_SNAPSHOT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The tool calling mode of a configuration loaded without one: `Native`,
/// or `OutputBased` in a saved task's snapshot, which older versions (whose
/// default it was) wrote without the mode.
fn loaded_tool_calling_mode() -> ToolCallingMode {
    if LOADING_TASK_SNAPSHOT.with(std::cell::Cell::get) {
        ToolCallingMode::OutputBased
    } else {
        ToolCallingMode::Native
    }
}

/// The tool configuration of a configuration loaded without one; see
/// [`loaded_tool_calling_mode`].
fn loaded_tool_config() -> AgentToolConfig {
    AgentToolConfig {
        tool_calling_mode: loaded_tool_calling_mode(),
        ..AgentToolConfig::default()
    }
}

/// Deserializes [`Task::agent_config`]: a snapshot without a tool calling
/// mode ran in output-based mode, the default when it was saved, and keeps
/// it.
pub(crate) fn deserialize_task_agent_config<'de, D>(
    deserializer: D,
) -> Result<AgentConfig, D::Error>
where
    D: Deserializer<'de>,
{
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOADING_TASK_SNAPSHOT.with(|loading| loading.set(self.0));
        }
    }
    let _restore = Restore(LOADING_TASK_SNAPSHOT.with(|loading| loading.replace(true)));
    AgentConfig::deserialize(deserializer)
}

/// Configuration for agent tool usage.
///
/// # Several calls in one response
///
/// A response's tool calls all run, and each gets exactly one result. They
/// run in call order as under a fair read/write lock, as Codex runs them: a
/// parallel-safe call starts once every earlier call that isn't
/// parallel-safe has finished, and any other call once every earlier call
/// has finished. So `[read, read]` run together, and `[read, write, read]`
/// runs the first read, then the write, then the second read. A tool is
/// parallel-safe when its definition says
/// so ([`ToolDefinition::parallel_safe`](crate::tool::ToolDefinition::parallel_safe);
/// the built-in reads are, the built-in writes are not, and an MCP tool is
/// when its server marks it `readOnlyHint`) unless its
/// [`PerToolConfigOverride::parallel_safe`] says otherwise.
/// [`one_tool_call_per_turn`](Self::one_tool_call_per_turn) asks for one
/// call per response instead.
///
/// Approvals keep that order: a call awaiting approval counts as not
/// finished, and so does a call held behind one. A call that must wait for
/// one is held (saved with the task) until the approvals are decided, and
/// then runs with the approved calls, in call order under the same rule;
/// the calls after a rejected call run all the same. If the wait ends
/// without a decision, a held call's result says an earlier call was still
/// waiting for approval. Calls of the response after one that ends the turn
/// (`complete_task`, `ask_user`, or a tool error that fails the task) are
/// not run, and their results say so; among the calls that run once
/// approvals are decided, only one that ends the task stops the others.
///
/// `complete_task` after an earlier call of its turn was rejected or failed
/// doesn't complete the task: the model chose to finish before it saw those
/// results, so its result says the task was not completed and asks it to
/// review the results and call `complete_task` again if the task is done.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "AgentToolConfigRepr")]
pub struct AgentToolConfig {
    /// The tools the agent is offered, besides `complete_task`, which is
    /// always offered. The runtime offers and runs only these (those the
    /// task's registry holds): a native call to any other tool gets an error
    /// result and `AgentEvent::UnavailableToolAttempted`, and output-based
    /// parsing recognizes only these tools' tags.
    pub available_tool_names: Vec<String>,
    /// Ask for one tool call per response (off by default). In native mode
    /// the request disables parallel tool calls where the provider can
    /// (OpenAI `parallel_tool_calls: false`, Anthropic
    /// `disable_parallel_tool_use`, OpenRouter `parallel_tool_calls: false`);
    /// Gemini can't, so a Gemini response's calls run one at a time instead,
    /// each after the one before, as every provider's do under this switch.
    /// The native system prompt asks for one call per response. In
    /// output-based mode no tool can be batched.
    #[serde(default)]
    pub one_tool_call_per_turn: bool,
    /// Whether calls run without approval, unless a tool's
    /// `tool_specific_configs` entry says otherwise. `ask_user` runs without
    /// approval either way, unless its own entry sets
    /// `auto_approve_tool_use: Some(false)`.
    #[serde(default = "default_auto_approve_tool_use_false")]
    pub default_auto_approve_tool_use: bool,
    #[serde(default)]
    pub default_out_of_workspace_path_policy: PathSafetyPolicy,
    #[serde(default)]
    pub tool_specific_configs: HashMap<String, PerToolConfigOverride>,
    /// How tool calls are communicated with the LLM. Defaults to `Native`;
    /// see [`ToolCallingMode`].
    #[serde(default = "loaded_tool_calling_mode")]
    pub tool_calling_mode: ToolCallingMode,
    /// The host-side time limit of a tool call, in seconds: 20 minutes
    /// ([`DEFAULT_TOOL_TIMEOUT_SECONDS`]) by default, `None` (JSON `null`)
    /// for none. It is never shown to the model as a parameter. A call that
    /// runs longer is stopped (a command it started is killed), and its
    /// result tells the model it did not finish within the host's time
    /// limit; the tool-error policy then applies as to any failed call
    /// (`InformAgentAndContinue` by default). Tools with their own wait and
    /// job handling are exempt ([`Tool::uses_host_timeout`](crate::tool::Tool::uses_host_timeout)),
    /// such as the terminal tools. A tool's
    /// [`PerToolConfigOverride::tool_timeout_seconds`] replaces it for that
    /// tool. Output supplied with an approval (`custom_tool_output`) runs
    /// nothing, so no limit applies.
    #[serde(default = "default_tool_timeout_seconds")]
    pub tool_timeout_seconds: Option<u64>,
}

/// The default [`AgentToolConfig::tool_timeout_seconds`]: 20 minutes.
pub const DEFAULT_TOOL_TIMEOUT_SECONDS: u64 = 20 * 60;

fn default_tool_timeout_seconds() -> Option<u64> {
    Some(DEFAULT_TOOL_TIMEOUT_SECONDS)
}

/// The serialized form of [`AgentToolConfig`], which also reads a setting of
/// older versions.
#[derive(Deserialize)]
struct AgentToolConfigRepr {
    available_tool_names: Vec<String>,
    #[serde(default)]
    one_tool_call_per_turn: bool,
    #[serde(default = "default_auto_approve_tool_use_false")]
    default_auto_approve_tool_use: bool,
    #[serde(default)]
    default_out_of_workspace_path_policy: PathSafetyPolicy,
    #[serde(default)]
    tool_specific_configs: HashMap<String, PerToolConfigOverride>,
    #[serde(default = "loaded_tool_calling_mode")]
    tool_calling_mode: ToolCallingMode,
    #[serde(default = "default_tool_timeout_seconds")]
    tool_timeout_seconds: Option<u64>,
    /// Removed: every call of a response runs now. Read only to warn when a
    /// configuration had turned batching off.
    #[serde(default)]
    allow_batched_tool_calls: Option<bool>,
}

impl From<AgentToolConfigRepr> for AgentToolConfig {
    fn from(repr: AgentToolConfigRepr) -> Self {
        if repr.allow_batched_tool_calls == Some(false) {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                log::warn!(
                    "Agent configurations no longer have `allow_batched_tool_calls`; `false` is ignored. Every tool call of a response now runs, calls to parallel-safe tools at the same time; set `one_tool_call_per_turn: true` to ask for one call per response."
                );
            });
        }
        Self {
            available_tool_names: repr.available_tool_names,
            one_tool_call_per_turn: repr.one_tool_call_per_turn,
            default_auto_approve_tool_use: repr.default_auto_approve_tool_use,
            default_out_of_workspace_path_policy: repr.default_out_of_workspace_path_policy,
            tool_specific_configs: repr.tool_specific_configs,
            tool_calling_mode: repr.tool_calling_mode,
            tool_timeout_seconds: repr.tool_timeout_seconds,
        }
    }
}

fn default_auto_approve_tool_use_false() -> bool {
    false
}

impl AgentToolConfig {
    /// The names of the tools the agent is offered and can run:
    /// `available_tool_names` and `complete_task`.
    pub fn offered_tool_names(&self) -> HashSet<String> {
        self.available_tool_names
            .iter()
            .cloned()
            .chain([crate::tool::builtins::COMPLETE_TASK_TOOL_NAME.to_string()])
            .collect()
    }

    /// The host-side time limit of a call to `tool_name`: its
    /// [`PerToolConfigOverride::tool_timeout_seconds`], or else
    /// [`tool_timeout_seconds`](Self::tool_timeout_seconds) unless the tool
    /// is exempt (`uses_host_timeout` false).
    pub fn tool_timeout(
        &self,
        tool_name: &str,
        uses_host_timeout: bool,
    ) -> Option<std::time::Duration> {
        let seconds = match self
            .tool_specific_configs
            .get(tool_name)
            .and_then(|config| config.tool_timeout_seconds)
        {
            Some(explicit) => explicit,
            None if uses_host_timeout => self.tool_timeout_seconds,
            None => None,
        };
        seconds.map(std::time::Duration::from_secs)
    }

    fn validate(&self) -> Result<(), ArkoAgentsError> {
        let invalid = |seconds: Option<u64>| {
            seconds.is_some_and(|seconds| {
                seconds == 0
                    || seconds
                        .checked_mul(1000)
                        .is_none_or(|milliseconds| !representable_ms(milliseconds))
            })
        };
        if invalid(self.tool_timeout_seconds)
            || self
                .tool_specific_configs
                .values()
                .any(|config| invalid(config.tool_timeout_seconds.flatten()))
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "AgentToolConfig".into(),
                message: "Tool timeouts must be positive and representable; use None (null) for no timeout.".into(),
            });
        }
        Ok(())
    }
}

impl Default for AgentToolConfig {
    fn default() -> Self {
        Self {
            available_tool_names: vec![crate::tool::builtins::COMPLETE_TASK_TOOL_NAME.to_string()],
            one_tool_call_per_turn: false,
            default_auto_approve_tool_use: false,
            default_out_of_workspace_path_policy: PathSafetyPolicy::default(),
            tool_specific_configs: HashMap::new(),
            tool_calling_mode: ToolCallingMode::default(),
            tool_timeout_seconds: default_tool_timeout_seconds(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct AgentLLMConfigSerdeHelper {
    provider: LLMProvider,
    model_id: String,
    /// Snapshot of the model description. Loading prefers the registry entry
    /// for `model_id` and falls back to this snapshot for ids the registry
    /// doesn't know (for example models added after the config was saved).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<LLMModel>,
    #[serde(default)]
    generation_config: LLMGenerationConfig,
}

/// Configuration for the LLM used by an agent.
#[derive(Debug, Clone)]
pub struct AgentLLMConfig {
    pub model: LLMModel,
    pub generation_config: LLMGenerationConfig,
}

impl Serialize for AgentLLMConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let helper = AgentLLMConfigSerdeHelper {
            provider: self.model.provider.clone(),
            model_id: self.model.id.clone(),
            model: Some(self.model.clone()),
            generation_config: self.generation_config.clone(),
        };
        helper.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AgentLLMConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = AgentLLMConfigSerdeHelper::deserialize(deserializer)?;
        let snapshot = helper
            .model
            .filter(|model| model.provider == helper.provider && model.id == helper.model_id);
        let model = match (
            LLMModel::get(helper.provider.clone(), &helper.model_id),
            snapshot,
        ) {
            (Ok(model), _) => model,
            (Err(_), Some(snapshot)) => snapshot,
            // OpenRouter slugs are dynamic: a hand-written config without a
            // snapshot gets a model without catalog metadata.
            (Err(_), None) if helper.provider == LLMProvider::OpenRouter => {
                LLMModel::openrouter(helper.model_id.clone())
            }
            (Err(e), None) => {
                return Err(serde::de::Error::custom(format!(
                    "Failed to get model '{}' for provider '{:?}': {}",
                    helper.model_id, helper.provider, e
                )));
            }
        };
        Ok(AgentLLMConfig {
            model,
            generation_config: helper.generation_config,
        })
    }
}

/// Policy for handling tool execution errors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolErrorPolicy {
    #[default]
    InformAgentAndContinue,
    PauseAndAwaitApproval,
    FailTask,
}

/// Policy for handling LLM errors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LLMErrorPolicy {
    InformAgentAndContinue,
    PauseAndAwaitApproval,
    RetryThenInformAgent { max_attempts: u32 },
    RetryThenPauseAndAwaitApproval { max_attempts: u32 },
}

impl Default for LLMErrorPolicy {
    /// Ten attempts (the first request and nine retries), then a pause. With
    /// the default [`LLMRetryConfig`] timing the retries back off from about
    /// 2 seconds to the 64-second cap, so the 5-minute elapsed budget, not
    /// the attempt count, usually ends recovery from an outage.
    fn default() -> Self {
        LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 10 }
    }
}

/// Timing limits shared by model-call and proactive token-refresh recovery.
/// The elapsed budget starts at the first failure of an episode and limits
/// the waits between attempts and the decision to make another: a failure
/// after it is spent ends recovery. The budget doesn't count the time a
/// response the provider accepted runs (it resumes if the response fails).
/// It never cuts off a request in flight,
/// which runs until it completes, fails, or the host cancels or suspends the
/// task (the opt-in `stream_idle_timeout_ms` is the only limit on a quiet
/// stream). Explicit capacity errors may opt into a separate indefinite wait
/// under RetryThen* policies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct LLMRetryConfig {
    pub initial_delay_ms: u64,
    /// Caps exponential backoff. A server Retry-After may exceed this cap,
    /// but must still fit the elapsed recovery budget.
    pub max_delay_ms: u64,
    pub max_elapsed_ms: u64,
    /// Equal jitter: choose a delay between half and all of the backoff cap.
    pub jitter: bool,
    /// None retains bounded recovery. Some retries explicit capacity errors until
    /// success or cancellation, without consuming the ordinary recovery budget.
    pub capacity_retry: Option<LLMCapacityRetryConfig>,
    /// On by default: short rate limits (HTTP 429, `rate_limit_exceeded`,
    /// `slow_down`) and, unless `capacity_retry` is set, capacity errors
    /// (`server_is_overloaded`, ...) are retried after the provider's delay
    /// (with jitter), until success or cancellation, without consuming the
    /// ordinary recovery budget or adding to the conversation. `None` (JSON
    /// `null`) handles them with ordinary bounded recovery.
    pub rate_limit_retry: Option<LLMRateLimitRetryConfig>,
    /// What happens when a plan usage limit is reached (ChatGPT
    /// `usage_limit_reached`, or a rate limit whose reset is further off than
    /// [`LLMRateLimitRetryConfig::max_wait_ms`]). Pauses by default.
    pub usage_limit_policy: UsageLimitPolicy,
    /// Opt-in: cut a streamed model response off once no data has arrived
    /// for this long (arko_ai's
    /// [`LLMClient::with_stream_idle_timeout`](arko_ai::LLMClient::with_stream_idle_timeout),
    /// applied to this agent's requests). `None`, the default, waits as long
    /// as the connection stays open, and that is deliberate: models can
    /// reason silently for a long time (GPT Pro models for over an hour),
    /// and a stream may carry nothing until the reasoning ends. A value
    /// shorter than that cuts healthy responses off; the cut surfaces as a
    /// truncated stream, which ordinary recovery sends again, paying for the
    /// work again. If you set one, make it generous (for example two hours).
    /// Only the gaps between data count: there is no total deadline. Never
    /// sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_idle_timeout_ms: Option<u64>,
}

impl Default for LLMRetryConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 2_000,
            max_delay_ms: 64_000,
            max_elapsed_ms: 300_000,
            jitter: true,
            capacity_retry: None,
            rate_limit_retry: Some(LLMRateLimitRetryConfig::default()),
            usage_limit_policy: UsageLimitPolicy::default(),
            stream_idle_timeout_ms: None,
        }
    }
}

/// Recovery from short rate limits and capacity errors; see
/// [`LLMRetryConfig::rate_limit_retry`]. The provider's delay (`Retry-After`,
/// rate-limit reset headers, a delay in the error body) is a minimum, with
/// jitter added; without one the delay backs off exponentially.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct LLMRateLimitRetryConfig {
    /// First backoff delay when the provider gives none.
    pub initial_delay_ms: u64,
    /// Cap of that backoff.
    pub max_delay_ms: u64,
    /// The longest provider delay followed as a short rate limit. A rate
    /// limit asking for a longer wait is handled as a usage limit
    /// ([`LLMRetryConfig::usage_limit_policy`]); a capacity error asking for
    /// one gets ordinary bounded recovery.
    pub max_wait_ms: u64,
}

impl Default for LLMRateLimitRetryConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 1_000,
            max_delay_ms: 60_000,
            max_wait_ms: 15 * 60 * 1000,
        }
    }
}

impl LLMRateLimitRetryConfig {
    fn validate(&self) -> Result<(), ArkoAgentsError> {
        if self.initial_delay_ms == 0
            || self.max_delay_ms < self.initial_delay_ms
            || !representable_ms(self.max_delay_ms)
            || !representable_ms(self.max_wait_ms)
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "LLMRateLimitRetryConfig".into(),
                message: "Rate-limit retry delays must be positive, ordered, and representable."
                    .into(),
            });
        }
        Ok(())
    }
}

/// What the runtime does when a plan usage limit is reached; see
/// [`LLMRetryConfig::usage_limit_policy`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum UsageLimitPolicy {
    /// Stop retrying and return control like exhausted recovery: the task
    /// awaits error-recovery approval under the pausing error policies, or
    /// is interrupted (returned to the host) otherwise.
    #[default]
    Pause,
    /// Opt-in: wait until the limit resets (plus a 1-second margin and up to
    /// 30 seconds of jitter) and retry unchanged, when the provider says
    /// when that is and the whole wait, margin and jitter included, fits in
    /// `max_wait_ms`; otherwise pause. Applies under every error policy
    /// (`PauseAndAwaitApproval` and `InformAgentAndContinue` too). The wait
    /// is cancellable and reported with `AgentEvent::UsageLimitReached`.
    ///
    /// A wait longer than the model's prompt-cache lifetime (see
    /// [`PlanUsageAction`]) makes the retried request read the whole context
    /// uncached, once; it can't be helped once the limit is reached. For an
    /// agent running in Claude Code, the limit is the one Claude Code reports
    /// reached, and the task resumes its session after the wait.
    WaitUntilReset { max_wait_ms: u64 },
}

fn representable_ms(milliseconds: u64) -> bool {
    std::time::Instant::now()
        .checked_add(std::time::Duration::from_millis(milliseconds))
        .is_some()
}

impl LLMRetryConfig {
    pub fn validate(&self) -> Result<(), ArkoAgentsError> {
        if let Some(capacity) = &self.capacity_retry {
            capacity.validate()?;
        }
        if let Some(rate_limit) = &self.rate_limit_retry {
            rate_limit.validate()?;
        }
        if self
            .stream_idle_timeout_ms
            .is_some_and(|timeout| timeout == 0 || !representable_ms(timeout))
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "LLMRetryConfig".into(),
                message: "stream_idle_timeout_ms must be positive and representable.".into(),
            });
        }
        if let UsageLimitPolicy::WaitUntilReset { max_wait_ms } = self.usage_limit_policy
            && !representable_ms(max_wait_ms)
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "UsageLimitPolicy".into(),
                message: "WaitUntilReset max_wait_ms must be representable.".into(),
            });
        }
        if self.initial_delay_ms == 0
            || self.max_delay_ms < self.initial_delay_ms
            || self.max_elapsed_ms == 0
            || std::time::Instant::now()
                .checked_add(std::time::Duration::from_millis(self.max_elapsed_ms))
                .is_none()
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "LLMRetryConfig".into(),
                message: "Retry delays must be positive, max_delay_ms must cover initial_delay_ms, and max_elapsed_ms must be positive and representable.".into(),
            });
        }
        Ok(())
    }
}

/// Backoff for opt-in indefinite capacity recovery. Server hints remain minimums.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct LLMCapacityRetryConfig {
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for LLMCapacityRetryConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 30_000,
            max_delay_ms: 300_000,
        }
    }
}

impl LLMCapacityRetryConfig {
    fn validate(&self) -> Result<(), ArkoAgentsError> {
        if self.initial_delay_ms == 0
            || self.max_delay_ms < self.initial_delay_ms
            || std::time::Instant::now()
                .checked_add(std::time::Duration::from_millis(self.max_delay_ms))
                .is_none()
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "LLMCapacityRetryConfig".into(),
                message: "Capacity retry delays must be positive, ordered, and representable."
                    .into(),
            });
        }
        Ok(())
    }
}

/// Opt-in monitors of model calls. Nothing is monitored by default.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MonitoringConfig {
    /// Act when subscription plan usage (reported by the ChatGPT backend on
    /// ChatGPT OAuth calls) reaches a percentage of a usage-limit window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_usage: Option<PlanUsageThreshold>,
    /// Report the conversation's prompt-cache efficiency after each model
    /// call, and warn or pause when it stays low.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMonitorConfig>,
}

/// See [`MonitoringConfig::cache`]. A call's cache efficiency is the input
/// it read from the prompt cache divided by the most it could have read:
/// the smaller of the previous call's input and its own (the shared prefix
/// can't be longer). Above 100% means the provider also cached earlier
/// output. The first `warmup_calls` calls of the conversation, and calls
/// whose provider reports no cache figure, are not judged. The rolling
/// efficiency sums the last `window_calls` judged calls; the thresholds
/// apply once that many have been judged, and each triggers once until the
/// efficiency recovers above it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CacheMonitorConfig {
    pub warmup_calls: usize,
    pub window_calls: usize,
    /// Emit `AgentEvent::CacheEfficiencyLow` below this (0 to 1).
    pub warn_below: Option<f64>,
    /// Also pause, before the next request, below this (0 to 1), like
    /// exhausted recovery. Off by default.
    pub pause_below: Option<f64>,
}

impl Default for CacheMonitorConfig {
    fn default() -> Self {
        Self {
            warmup_calls: 2,
            window_calls: 5,
            warn_below: Some(0.5),
            pause_below: None,
        }
    }
}

impl MonitoringConfig {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn validate(&self) -> Result<(), ArkoAgentsError> {
        if let Some(threshold) = &self.plan_usage {
            let delay_ok = match threshold.action {
                PlanUsageAction::SlowDown { delay_ms } => representable_ms(delay_ms),
                PlanUsageAction::Pause => true,
            };
            let percent_ok = threshold.percent > 0.0 && threshold.percent <= 100.0;
            if !(percent_ok && delay_ok) {
                return Err(ArkoAgentsError::BuilderError {
                    struct_name: "PlanUsageThreshold".into(),
                    message: "percent must be in (0, 100] and a slow-down delay representable."
                        .into(),
                });
            }
        }
        if let Some(cache) = &self.cache {
            let fraction =
                |value: Option<f64>| value.is_none_or(|value| (0.0..=1.0).contains(&value));
            if cache.window_calls == 0
                || !fraction(cache.warn_below)
                || !fraction(cache.pause_below)
            {
                return Err(ArkoAgentsError::BuilderError {
                    struct_name: "CacheMonitorConfig".into(),
                    message: "window_calls must be positive and thresholds between 0 and 1.".into(),
                });
            }
        }
        Ok(())
    }
}

/// See [`MonitoringConfig::plan_usage`]. Checked before each model request
/// against the most recent report: when the most-used window is at or above
/// `percent`, `AgentEvent::PlanUsageThresholdReached` is emitted (once per
/// window and reset) and `action` applies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanUsageThreshold {
    pub percent: f64,
    #[serde(default)]
    pub action: PlanUsageAction,
}

/// What a [`PlanUsageThreshold`] does.
///
/// Neither action saves usage: the task's requests count the same against
/// the plan whenever they are made. What differs is whether the task keeps
/// going as the limit nears. Choose with the window's reset in mind (the
/// threshold event carries it) and the prompt cache: a request made after
/// the cache has expired reads the whole context again as uncached input,
/// which uses more of the plan than cached input, more than any wait saves. [`prompt_cache_lifetime`](arko_ai::prompt_cache::prompt_cache_lifetime)
/// gives the model's lifetime: 5 minutes for Anthropic by default (1 hour
/// with `AnthropicCacheTtl::OneHour`), a conservative 5 minutes for OpenAI;
/// for a Claude Code model, `anthropic_cache_ttl` when set (Claude Code's
/// `promptCacheTtl`), otherwise Claude Code's choice: 1 hour on a Claude
/// subscription within its usage limits, 5 minutes on an API key. Claude
/// Code's subagents (its `Agent` tool) and helper requests cache for 5
/// minutes whatever the main conversation's TTL, so a stop of more than
/// about 4½ minutes loses their caches even when the main conversation's
/// lasts an hour.
///
/// - The reset is far off: [`Pause`](Self::Pause), so the rest of the
///   window isn't spent before someone decides. The request that resumes
///   after the cache expired reads the context uncached, once.
/// - The reset is close: let the task run, with
///   [`UsageLimitPolicy::WaitUntilReset`] to wait out the limit if it is
///   reached first (again, one uncached read once the wait ends), or
///   [`SlowDown`](Self::SlowDown) to stretch what is left until the reset.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum PlanUsageAction {
    /// Pause once per window and reset, like exhausted recovery (awaiting
    /// error-recovery approval, or interrupted under the informing error
    /// policies). Suits a threshold reached with the window's reset far off.
    #[default]
    Pause,
    /// Wait `delay_ms` (cancellable) before every model request while usage
    /// stays at or above the threshold. It spends the same usage over more
    /// time, so it only helps when the reset comes before the rest of the
    /// window runs out: pace the task to reach the reset without stopping.
    ///
    /// Keep `delay_ms` well under the model's prompt-cache lifetime (see
    /// [`PlanUsageAction`]): a delay that reaches it makes every request after
    /// it miss the cache, which uses the window up faster than no delay at
    /// all. The runtime logs a warning, once per run, for a delay within 30
    /// seconds of the lifetime; for a Claude Code agent offered the `Agent`
    /// or `Task` tool, of its subagents' 5 minutes. For an agent
    /// running in Claude Code, each tool call is held instead (and the
    /// request that follows it with it), since Claude Code makes the
    /// requests.
    SlowDown { delay_ms: u64 },
}

/// Configuration for error recovery behavior.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ErrorRecoveryConfig {
    #[serde(default)]
    pub on_tool_execution_error: ToolErrorPolicy,
}

/// Specifies how frequently a user reminder should be injected into the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ReminderInjectionFrequency {
    /// Inject the reminder only into the initial task prompt (the first user message).
    #[default]
    InitialTaskPromptOnly,
    /// Inject the reminder into every user message.
    EveryUserMessage,
    /// Inject the reminder into the initial task prompt and every Nth user
    /// message after it. N is at least 1 ([`AgentConfig::validate`]); to
    /// turn the reminder off, leave [`AgentConfig::user_reminder`] unset.
    EveryNUserMessages(usize),
}

/// A user-provided reminder to reinforce instructions that the agent may stray from.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserReminder {
    /// The reminder content to inject.
    pub content: String,
    /// How frequently the reminder should be injected.
    #[serde(default)]
    pub frequency: ReminderInjectionFrequency,
}

impl UserReminder {
    /// Creates a new user reminder with the given content and frequency.
    pub fn new(content: String, frequency: ReminderInjectionFrequency) -> Self {
        Self { content, frequency }
    }

    /// Creates a reminder that is only injected into the initial task prompt.
    pub fn initial_only(content: String) -> Self {
        Self {
            content,
            frequency: ReminderInjectionFrequency::InitialTaskPromptOnly,
        }
    }

    /// Creates a reminder that is injected into every user message.
    pub fn every_message(content: String) -> Self {
        Self {
            content,
            frequency: ReminderInjectionFrequency::EveryUserMessage,
        }
    }

    /// Creates a reminder that is injected every N user messages.
    pub fn every_n_messages(content: String, n: usize) -> Self {
        Self {
            content,
            frequency: ReminderInjectionFrequency::EveryNUserMessages(n),
        }
    }
}

/// The main configuration for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    pub role_description: String,
    pub instructions: String,
    pub llm_config: AgentLLMConfig,
    #[serde(default = "loaded_tool_config")]
    pub tool_config: AgentToolConfig,
    #[serde(default)]
    pub environment_details_config: EnvironmentDetailsConfig,
    /// Replaces the description of `complete_task`'s `summary` parameter,
    /// in both tool calling modes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complete_task_summary_prompt_override: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parsing_config_override: Option<ToolParsingConfig>,
    #[serde(default)]
    pub error_recovery_config: ErrorRecoveryConfig,
    #[serde(default)]
    pub llm_error_policy: LLMErrorPolicy,
    #[serde(default)]
    pub llm_retry_config: LLMRetryConfig,
    /// Opt-in (default `None`, no limit): a task awaiting a command (an
    /// approval, an answer, or error recovery) that has no decision this
    /// many seconds after the wait started is returned `Interrupted`, and
    /// calls awaiting approval get a result saying so. At least 1
    /// ([`validate`](Self::validate)). The deadline is set
    /// once per wait: `InjectContext` and commands the wait ignores don't
    /// move it, and a decision ends the wait (the next wait gets its own).
    /// A wait the host says it is deciding
    /// ([`UserCommand::HostDeciding`](crate::runtime::command::UserCommand::HostDeciding))
    /// isn't timed, nor is any wait of a run whose host decides all of them
    /// ([`HostDecisions`](crate::runtime::HostDecisions), which a workflow
    /// sets for the runs it drives; it isn't saved with the task).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_command_timeout_seconds: Option<u64>,
    /// Optional user-provided reminder to reinforce instructions the agent may stray from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_reminder: Option<UserReminder>,
    /// Optional skills configuration. When present, enables on-demand skill
    /// loading via the `load_skill` tool and/or auto-loaded skills in the
    /// system prompt. When absent, skills are completely disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills_config: Option<SkillsConfig>,
    /// Opt-in monitors of model calls; see [`MonitoringConfig`].
    #[serde(default, skip_serializing_if = "MonitoringConfig::is_empty")]
    pub monitoring: MonitoringConfig,
    /// Opt-in: a model (with its generation settings) that a refused turn is
    /// sent to once, unchanged, before the refusal stops the task. The
    /// refused turn is not kept; the fallback's response continues the task,
    /// and later turns use `llm_config` again. If the fallback also refuses,
    /// the refusal stands; a fallback request that fails otherwise is
    /// recovered from by its own error (retried to the fallback model, or
    /// paused with that error's cause and the fallback still pending). Off
    /// by default.
    ///
    /// The fallback must use the configured model's provider ([`validate`](AgentConfig::validate)
    /// rejects others): providers can't replay each other's assistant turns.
    /// Its `tool_config` is sent with its requests (in native mode, its
    /// `tool_choice` and `parallel_tool_calls`), but the calls of its
    /// response run under the agent's settings: one at a time only when
    /// [`AgentToolConfig::one_tool_call_per_turn`] or the configured model's
    /// `tool_config` asks for it.
    /// Its ChatGPT turn-state token is kept apart from the configured
    /// model's, and OpenAI reasoning is replayed between the two models only
    /// when both are in one Codex compaction family (the GPT-5.6 and GPT-6
    /// models of `Provider::OpenAI`); otherwise it is dropped, with the ids of
    /// that response's other items. Anthropic thinking is kept (a model that
    /// can't read another model's thinking ignores it). Replaying Gemini
    /// thought signatures and OpenRouter `reasoning_details` to another model
    /// of the same provider is unverified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal_fallback: Option<AgentLLMConfig>,
    /// Opt-in: once the context reaches a threshold, the agent writes a
    /// report and continues from it in a fresh conversation, in the same
    /// task; see [`crate::handover`]. Off by default. A host can ask for a
    /// handover with [`UserCommand::RequestHandover`](crate::runtime::command::UserCommand::RequestHandover)
    /// whether or not this is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handover: Option<HandoverConfig>,
    /// How the Claude Code CLI runs, for an agent whose model is a Claude
    /// Code model; see [`crate::claude_code`]. `None`: the defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code: Option<ClaudeCodeSettings>,
}

impl AgentConfig {
    /// Creates a new builder for `AgentConfig`.
    pub fn builder(id: String, name: String, llm_config: AgentLLMConfig) -> AgentConfigBuilder {
        AgentConfigBuilder::new(id, name, llm_config)
    }

    /// Loads an `AgentConfig` from a JSON file.
    pub fn load_from_path(path: &Path) -> Result<Self, ArkoAgentsError> {
        let file_content =
            std::fs::read_to_string(path).map_err(|e| ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            })?;
        serde_json::from_str(&file_content).map_err(ArkoAgentsError::Json)
    }

    /// Saves the `AgentConfig` to a JSON file, replacing it in one step (see
    /// [`write_file_atomically`](crate::utils::write_file_atomically)).
    pub fn save_to_path(&self, path: &Path) -> Result<(), ArkoAgentsError> {
        let json_string = serde_json::to_string_pretty(self)?;
        crate::utils::write_file_atomically(path, json_string.as_bytes()).map_err(|e| {
            ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            }
        })
    }

    /// Determines if tool use should be auto-approved for a given tool: its
    /// `tool_specific_configs` override, or else
    /// `default_auto_approve_tool_use`. `ask_user` is auto-approved unless
    /// its override says otherwise, since the question already waits for an
    /// answer.
    pub fn should_auto_approve_tool(&self, tool_name: &str) -> bool {
        self.tool_config
            .tool_specific_configs
            .get(tool_name)
            .and_then(|override_cfg| override_cfg.auto_approve_tool_use)
            .unwrap_or(
                self.tool_config.default_auto_approve_tool_use
                    || tool_name == crate::tool::builtins::ASK_USER_TOOL_NAME,
            )
    }

    /// Gets the path safety policy for a given tool.
    pub fn get_path_safety_policy(&self, tool_name: &str) -> PathSafetyPolicy {
        self.tool_config
            .tool_specific_configs
            .get(tool_name)
            .and_then(|override_cfg| override_cfg.out_of_workspace_path_policy.clone())
            .unwrap_or_else(|| {
                self.tool_config
                    .default_out_of_workspace_path_policy
                    .clone()
            })
    }

    /// Whether calls to a tool may run at the same time as other
    /// parallel-safe calls of a response: its
    /// [`PerToolConfigOverride::parallel_safe`], or else its definition's
    /// [`parallel_safe`](crate::tool::ToolDefinition::parallel_safe). See
    /// [`AgentToolConfig`].
    pub fn is_parallel_safe(&self, tool_definition: &crate::tool::ToolDefinition) -> bool {
        self.tool_config
            .tool_specific_configs
            .get(&tool_definition.name)
            .and_then(|override_cfg| override_cfg.parallel_safe)
            .unwrap_or(tool_definition.parallel_safe)
    }

    /// Whether, in output-based mode, a response may hold a batch of calls
    /// to this tool: a parallel-safe tool, unless the agent asks for one
    /// call per response.
    pub fn allows_batched_calls_for_tool(
        &self,
        tool_definition: &crate::tool::ToolDefinition,
    ) -> bool {
        !self.asks_one_tool_call_per_turn() && self.is_parallel_safe(tool_definition)
    }

    /// Whether the agent asks for one tool call per response:
    /// [`AgentToolConfig::one_tool_call_per_turn`], or `parallel_tool_calls:
    /// Some(false)` in the model's configured
    /// [`tool_config`](arko_ai::GenerationConfig::tool_config), which means the
    /// same.
    pub fn asks_one_tool_call_per_turn(&self) -> bool {
        self.tool_config.one_tool_call_per_turn
            || self
                .llm_config
                .generation_config
                .tool_config
                .as_ref()
                .and_then(|tool_config| tool_config.parallel_tool_calls)
                == Some(false)
    }

    /// A tool's definition as the agent is shown it, in either tool calling
    /// mode: with [`complete_task_summary_prompt_override`](Self::complete_task_summary_prompt_override)
    /// as `complete_task`'s `summary` description.
    pub(crate) fn offered_tool_definition(
        &self,
        mut definition: crate::tool::ToolDefinition,
    ) -> crate::tool::ToolDefinition {
        if definition.name == crate::tool::builtins::COMPLETE_TASK_TOOL_NAME
            && let Some(summary) = &self.complete_task_summary_prompt_override
        {
            for parameter in &mut definition.parameters {
                if parameter.name
                    == crate::tool::builtins::complete_task::COMPLETE_TASK_SUMMARY_PARAM_NAME
                {
                    parameter.description = summary.clone();
                }
            }
        }
        definition
    }

    /// Gets the tool parsing configuration, using the override if present.
    pub fn get_tool_parsing_config(&self) -> ToolParsingConfig {
        self.parsing_config_override.clone().unwrap_or_default()
    }

    /// Returns true if this agent config uses native tool calling.
    pub fn is_native_tool_calling(&self) -> bool {
        self.tool_config.tool_calling_mode == ToolCallingMode::Native
    }

    /// Checks the settings the builder checks: retry timing, monitors, a
    /// command timeout and reminder frequency of at least 1, a forced tool
    /// choice only where the model takes it
    /// ([`Model::supports_forced_tool_choice`](arko_ai::Model::supports_forced_tool_choice),
    /// for the configured model and the refusal fallback), and a refusal
    /// fallback on the configured model's provider. The runtime checks them
    /// again when a task starts, since configs can be built or loaded
    /// without the builder (a saved task holding a rejected value loads,
    /// and running it fails with this error until the value is changed).
    pub fn validate(&self) -> Result<(), ArkoAgentsError> {
        self.llm_retry_config.validate()?;
        self.monitoring.validate()?;
        if let Some(handover) = &self.handover {
            handover.validate()?;
        }
        self.tool_config.validate()?;
        if self.user_command_timeout_seconds == Some(0) {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "AgentConfig".into(),
                message: "user_command_timeout_seconds must be at least 1 (a timeout of 0 ends every wait at once); leave it unset (None) to wait for commands without a limit.".into(),
            });
        }
        if let Some(reminder) = &self.user_reminder
            && reminder.frequency == ReminderInjectionFrequency::EveryNUserMessages(0)
        {
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "UserReminder".into(),
                message: "EveryNUserMessages needs n of at least 1 (with 0 the reminder is never injected); to turn the reminder off, leave user_reminder unset (None).".into(),
            });
        }
        for (setting, llm) in std::iter::once(("llm_config", &self.llm_config)).chain(
            self.refusal_fallback
                .iter()
                .map(|f| ("refusal_fallback", f)),
        ) {
            let config = &llm.generation_config;
            if let Some(choice @ (arko_ai::ToolChoice::Required | arko_ai::ToolChoice::Specific(_))) =
                config
                    .tool_config
                    .as_ref()
                    .and_then(|tools| tools.tool_choice.as_ref())
                && !llm.model.supports_forced_tool_choice(config)
            {
                return Err(ArkoAgentsError::BuilderError {
                    struct_name: "AgentConfig".into(),
                    message: format!(
                        "{setting}.generation_config.tool_config.tool_choice forces a tool ({choice:?}), which model '{}' doesn't accept with this generation config (Model::supports_forced_tool_choice); use Auto or None.",
                        llm.model.id
                    ),
                });
            }
        }
        self.validate_claude_code()?;
        if let Some(fallback) = &self.refusal_fallback
            && fallback.model.provider != self.llm_config.model.provider
        {
            // Providers can't replay each other's assistant turns (reasoning
            // artifacts are provider-specific), so a cross-provider fallback
            // would break every later request.
            return Err(ArkoAgentsError::BuilderError {
                struct_name: "AgentConfig".into(),
                message: format!(
                    "refusal_fallback must use the configured model's provider ({:?}), not {:?}.",
                    self.llm_config.model.provider, fallback.model.provider
                ),
            });
        }
        Ok(())
    }
}

impl AgentConfig {
    /// Whether the agent's model runs in Claude Code
    /// ([`Provider::ClaudeCode`](arko_ai::Provider::ClaudeCode)).
    pub fn runs_in_claude_code(&self) -> bool {
        self.llm_config.model.provider == arko_ai::Provider::ClaudeCode
    }

    /// The Claude Code rules of [`validate`](Self::validate): built-in tools
    /// only for Claude Code models, and for those, only the settings the
    /// Claude Code runtime applies.
    fn validate_claude_code(&self) -> Result<(), ArkoAgentsError> {
        let error = |message: String| ArkoAgentsError::BuilderError {
            struct_name: "AgentConfig".into(),
            message,
        };
        let tools = &self.tool_config;
        if !self.runs_in_claude_code() {
            if let Some(tool) = tools
                .available_tool_names
                .iter()
                .find(|name| crate::claude_code::is_built_in_tool(name))
            {
                return Err(error(format!(
                    "'{tool}' is a Claude Code built-in tool, offered only to agents whose model is a Claude Code model (Model::claude_code); model '{}' is {:?}.",
                    self.llm_config.model.id, self.llm_config.model.provider
                )));
            }
            if self.claude_code.is_some() {
                return Err(error(format!(
                    "claude_code settings apply only to Claude Code models; model '{}' is {:?}.",
                    self.llm_config.model.id, self.llm_config.model.provider
                )));
            }
            return Ok(());
        }
        if tools.tool_calling_mode != ToolCallingMode::Native {
            return Err(error(
                "a Claude Code agent calls tools natively; set tool_calling_mode to Native.".into(),
            ));
        }
        if let Some(tool) = tools
            .available_tool_names
            .iter()
            .find(|name| crate::claude_code::UNSUPPORTED_BUILT_IN_TOOLS.contains(&name.as_str()))
        {
            return Err(error(format!(
                "Claude Code's '{tool}' can't be offered: it would bypass the host (ask the user with ask_user instead)."
            )));
        }
        for (tool, config) in &tools.tool_specific_configs {
            if !crate::claude_code::is_built_in_tool(tool) {
                continue;
            }
            if config.tool_timeout_seconds.is_some() {
                return Err(error(format!(
                    "tool_timeout_seconds can't apply to Claude Code's '{tool}': Claude Code runs its built-in tools, and arko can't stop one call."
                )));
            }
            if config.parallel_safe.is_some() {
                return Err(error(format!(
                    "parallel_safe can't apply to Claude Code's '{tool}': Claude Code decides how its built-in tools run."
                )));
            }
        }
        Ok(())
    }
}

/// Builder for constructing `AgentConfig` instances.
#[derive(Default)]
pub struct AgentConfigBuilder {
    id: Option<String>,
    name: Option<String>,
    role_description: Option<String>,
    instructions: Option<String>,
    llm_config: Option<AgentLLMConfig>,
    tool_config: Option<AgentToolConfig>,
    environment_details_config: Option<EnvironmentDetailsConfig>,
    complete_task_summary_prompt_override: Option<String>,
    parsing_config_override: Option<ToolParsingConfig>,
    error_recovery_config: Option<ErrorRecoveryConfig>,
    llm_error_policy: Option<LLMErrorPolicy>,
    llm_retry_config: Option<LLMRetryConfig>,
    user_command_timeout_seconds: Option<u64>,
    user_reminder: Option<UserReminder>,
    skills_config: Option<SkillsConfig>,
    monitoring: Option<MonitoringConfig>,
    refusal_fallback: Option<AgentLLMConfig>,
    handover: Option<HandoverConfig>,
    claude_code: Option<ClaudeCodeSettings>,
}

impl AgentConfigBuilder {
    /// Creates a new builder with required fields.
    pub fn new(id: String, name: String, llm_config: AgentLLMConfig) -> Self {
        Self {
            id: Some(id),
            name: Some(name),
            llm_config: Some(llm_config),
            ..Default::default()
        }
    }

    /// Sets the role description for the agent.
    pub fn role_description(mut self, role_description: String) -> Self {
        self.role_description = Some(role_description);
        self
    }

    /// Sets the instructions for the agent.
    pub fn instructions(mut self, instructions: String) -> Self {
        self.instructions = Some(instructions);
        self
    }

    /// Sets the tool configuration.
    pub fn tool_config(mut self, tool_config: AgentToolConfig) -> Self {
        self.tool_config = Some(tool_config);
        self
    }

    /// Sets the environment details configuration.
    pub fn environment_details_config(mut self, config: EnvironmentDetailsConfig) -> Self {
        self.environment_details_config = Some(config);
        self
    }

    /// Sets the description of `complete_task`'s `summary` parameter; see
    /// [`AgentConfig::complete_task_summary_prompt_override`].
    pub fn complete_task_summary_prompt_override(mut self, prompt: String) -> Self {
        self.complete_task_summary_prompt_override = Some(prompt);
        self
    }

    /// Sets the parsing configuration override.
    pub fn parsing_config_override(mut self, config: ToolParsingConfig) -> Self {
        self.parsing_config_override = Some(config);
        self
    }

    /// Sets the error recovery configuration.
    pub fn error_recovery_config(mut self, config: ErrorRecoveryConfig) -> Self {
        self.error_recovery_config = Some(config);
        self
    }

    /// Sets the LLM error policy.
    pub fn llm_error_policy(mut self, policy: LLMErrorPolicy) -> Self {
        self.llm_error_policy = Some(policy);
        self
    }

    /// Sets bounded retry timing for model calls and token refresh.
    pub fn llm_retry_config(mut self, config: LLMRetryConfig) -> Self {
        self.llm_retry_config = Some(config);
        self
    }

    /// Sets the opt-in command timeout in seconds; see
    /// [`AgentConfig::user_command_timeout_seconds`].
    pub fn user_command_timeout_seconds(mut self, timeout: Option<u64>) -> Self {
        self.user_command_timeout_seconds = timeout;
        self
    }

    /// Sets the user-provided reminder.
    pub fn user_reminder(mut self, reminder: UserReminder) -> Self {
        self.user_reminder = Some(reminder);
        self
    }

    /// Sets the skills configuration.
    pub fn skills_config(mut self, config: SkillsConfig) -> Self {
        self.skills_config = Some(config);
        self
    }

    /// Sets the opt-in model-call monitors.
    pub fn monitoring(mut self, config: MonitoringConfig) -> Self {
        self.monitoring = Some(config);
        self
    }

    /// Sets the opt-in model a refused turn is sent to once; see
    /// [`AgentConfig::refusal_fallback`].
    pub fn refusal_fallback(mut self, config: AgentLLMConfig) -> Self {
        self.refusal_fallback = Some(config);
        self
    }

    /// Sets the opt-in conversation handover; see [`AgentConfig::handover`].
    pub fn handover(mut self, config: HandoverConfig) -> Self {
        self.handover = Some(config);
        self
    }

    /// Sets how the Claude Code CLI runs, for a Claude Code model; see
    /// [`AgentConfig::claude_code`].
    pub fn claude_code(mut self, settings: ClaudeCodeSettings) -> Self {
        self.claude_code = Some(settings);
        self
    }

    /// Builds the `AgentConfig`, returning an error if required fields are missing.
    pub fn build(self) -> Result<AgentConfig, ArkoAgentsError> {
        let llm_retry_config = self.llm_retry_config.unwrap_or_default();
        let monitoring = self.monitoring.unwrap_or_default();
        let config = AgentConfig {
            id: self.id.ok_or_else(|| ArkoAgentsError::BuilderError {
                struct_name: "AgentConfig".into(),
                message: "ID is required".into(),
            })?,
            name: self.name.ok_or_else(|| ArkoAgentsError::BuilderError {
                struct_name: "AgentConfig".into(),
                message: "Name is required".into(),
            })?,
            role_description: self.role_description.ok_or_else(|| {
                ArkoAgentsError::BuilderError {
                    struct_name: "AgentConfig".into(),
                    message: "Role description is required".into(),
                }
            })?,
            instructions: self
                .instructions
                .ok_or_else(|| ArkoAgentsError::BuilderError {
                    struct_name: "AgentConfig".into(),
                    message: "Instructions are required".into(),
                })?,
            llm_config: self
                .llm_config
                .ok_or_else(|| ArkoAgentsError::BuilderError {
                    struct_name: "AgentConfig".into(),
                    message: "LLM config is required".into(),
                })?,
            tool_config: self.tool_config.unwrap_or_default(),
            environment_details_config: self.environment_details_config.unwrap_or_default(),
            complete_task_summary_prompt_override: self.complete_task_summary_prompt_override,
            parsing_config_override: self.parsing_config_override,
            error_recovery_config: self.error_recovery_config.unwrap_or_default(),
            llm_error_policy: self.llm_error_policy.unwrap_or_default(),
            llm_retry_config,
            user_command_timeout_seconds: self.user_command_timeout_seconds,
            user_reminder: self.user_reminder,
            skills_config: self.skills_config,
            monitoring,
            refusal_fallback: self.refusal_fallback,
            handover: self.handover,
            claude_code: self.claude_code,
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod llm_config_serde_tests {
    use super::*;

    #[test]
    fn registry_models_round_trip_and_the_registry_wins_over_a_stale_snapshot() {
        let config = AgentLLMConfig {
            model: LLMModel::openai_gpt_6_astra(),
            generation_config: LLMGenerationConfig::default().with_max_tokens(1234),
        };
        let mut saved = serde_json::to_value(&config).unwrap();
        assert_eq!(saved["model_id"], "gpt-6-astra");
        assert_eq!(saved["model"]["id"], "gpt-6-astra");
        // A snapshot with outdated data is ignored when the registry knows the id.
        saved["model"]["max_input_tokens"] = serde_json::json!(1);
        let loaded: AgentLLMConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(
            loaded.model.max_input_tokens,
            LLMModel::openai_gpt_6_astra().max_input_tokens
        );
        assert_eq!(loaded.generation_config.max_tokens, Some(1234));
    }

    #[test]
    fn unknown_model_ids_load_from_their_snapshot() {
        let mut model = LLMModel::openai_gpt_6_luna();
        model.id = "gpt-future".into();
        let config = AgentLLMConfig {
            model: model.clone(),
            generation_config: LLMGenerationConfig::default(),
        };
        let saved = serde_json::to_string(&config).unwrap();
        let loaded: AgentLLMConfig = serde_json::from_str(&saved).unwrap();
        assert_eq!(loaded.model.id, "gpt-future");
        assert_eq!(loaded.model.max_input_tokens, model.max_input_tokens);

        // Without a snapshot an unknown id still fails clearly.
        let bare = serde_json::json!({"provider": "OpenAI", "model_id": "gpt-future"});
        let error = serde_json::from_value::<AgentLLMConfig>(bare).unwrap_err();
        assert!(error.to_string().contains("gpt-future"), "{error}");
    }

    #[test]
    fn minimal_hand_written_configs_load_with_default_generation_settings() {
        let yaml = "provider: Anthropic\nmodel_id: claude-opus-5-5\n";
        let loaded: AgentLLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(loaded.model.id, "claude-opus-5-5");
        assert!(loaded.generation_config.max_tokens.is_none());
        let partial =
            "provider: OpenAI\nmodel_id: gpt-6-sol\ngeneration_config:\n  max_tokens: 900\n";
        let loaded: AgentLLMConfig = serde_yaml::from_str(partial).unwrap();
        assert_eq!(loaded.generation_config.max_tokens, Some(900));
    }

    #[test]
    fn openrouter_configs_load_with_or_without_a_catalog_snapshot() {
        let yaml = "provider: OpenRouter\nmodel_id: anthropic/claude-sonnet-5\ngeneration_config:\n  reasoning: !OpenRouter\n    effort: high\n  openrouter:\n    provider:\n      zdr: true\n";
        let loaded: AgentLLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(loaded.model.provider, LLMProvider::OpenRouter);
        assert_eq!(loaded.model.id, "anthropic/claude-sonnet-5");
        assert!(loaded.model.openrouter.is_none());
        assert!(matches!(
            loaded.generation_config.reasoning,
            Some(arko_ai::ReasoningConfig::OpenRouter(ref reasoning))
                if reasoning.effort == Some(arko_ai::ReasoningEffort::High)
        ));
        let zdr = loaded
            .generation_config
            .openrouter
            .as_ref()
            .and_then(|o| o.provider.as_ref())
            .and_then(|p| p.zdr);
        assert_eq!(zdr, Some(true));

        // A saved snapshot keeps catalog metadata.
        let mut model = LLMModel::openrouter("openai/gpt-6-sol");
        model.max_input_tokens = 1_050_000;
        let config = AgentLLMConfig {
            model: model.clone(),
            generation_config: LLMGenerationConfig::default(),
        };
        let loaded: AgentLLMConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(loaded.model.max_input_tokens, 1_050_000);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    fn config(model: LLMModel, generation_config: LLMGenerationConfig) -> AgentConfig {
        AgentConfig::builder(
            "validation-test".into(),
            "Validation test".into(),
            AgentLLMConfig {
                model,
                generation_config,
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .build()
        .unwrap()
    }

    fn luna() -> AgentConfig {
        config(
            LLMModel::openai_gpt_5_6_luna(),
            LLMGenerationConfig::default(),
        )
    }

    /// A zero command timeout (which ended every wait at once) is rejected,
    /// by the builder too; a saved task holding one still loads.
    #[test]
    fn a_zero_command_timeout_is_rejected() {
        let mut config = luna();
        config.user_command_timeout_seconds = Some(0);
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains("user_command_timeout_seconds must be at least 1")
                && error.contains("unset (None)"),
            "{error}"
        );
        config.user_command_timeout_seconds = Some(1);
        assert!(config.validate().is_ok());
        let built = AgentConfig::builder(
            "x".into(),
            "x".into(),
            AgentLLMConfig {
                model: LLMModel::openai_gpt_5_6_luna(),
                generation_config: LLMGenerationConfig::default(),
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .user_command_timeout_seconds(Some(0))
        .build();
        assert!(built.is_err());

        let mut saved =
            serde_json::to_value(Task::new(luna(), "Do it.".into(), ".".into(), None)).unwrap();
        saved["agent_config"]["user_command_timeout_seconds"] = serde_json::json!(0);
        let task: Task = serde_json::from_value(saved).unwrap();
        assert_eq!(task.agent_config.user_command_timeout_seconds, Some(0));
        assert!(task.agent_config.validate().is_err());
    }

    /// A reminder every zero messages (which was never injected, not even
    /// into the task prompt) is rejected, pointing at unsetting the
    /// reminder; a saved task holding one still loads.
    #[test]
    fn a_reminder_every_zero_messages_is_rejected() {
        let mut config = luna();
        config.user_reminder = Some(UserReminder::every_n_messages("Mind the tests.".into(), 0));
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains("EveryNUserMessages needs n of at least 1")
                && error.contains("leave user_reminder unset"),
            "{error}"
        );
        config.user_reminder = Some(UserReminder::every_n_messages("Mind the tests.".into(), 1));
        assert!(config.validate().is_ok());

        let mut saved =
            serde_json::to_value(Task::new(luna(), "Do it.".into(), ".".into(), None)).unwrap();
        saved["agent_config"]["user_reminder"] = serde_json::json!({"content": "Mind the tests.", "frequency": {"EveryNUserMessages": 0}});
        let task: Task = serde_json::from_value(saved).unwrap();
        assert_eq!(
            task.agent_config.user_reminder.as_ref().unwrap().frequency,
            ReminderInjectionFrequency::EveryNUserMessages(0)
        );
        assert!(task.agent_config.validate().is_err());
    }

    /// An omitted `include` takes the default, `false` (it was required).
    #[test]
    fn an_omitted_include_takes_the_default() {
        let detail =
            serde_json::from_str::<BuiltInEnvironmentDetail>(r#"{"placement": "EachUserMessage"}"#)
                .unwrap();
        assert!(!detail.include);
        assert_eq!(
            detail.placement,
            EnvironmentDetailPlacement::EachUserMessage
        );
        let details: EnvironmentDetailsConfig =
            serde_yaml::from_str("file_tree:\n  depth: 2\ncurrent_datetime:\n  include: true\n")
                .unwrap();
        assert!(!details.file_tree.include);
        assert_eq!(details.file_tree.depth, Some(2));
        assert!(details.current_datetime.include);
    }

    /// A forced tool choice the configured model (or the refusal fallback)
    /// doesn't take fails validation, so the builder and a task start
    /// report it before any request; others pass.
    #[test]
    fn a_forced_tool_choice_the_model_rejects_fails_validation() {
        let with_choice = |choice: arko_ai::ToolChoice| {
            LLMGenerationConfig::default()
                .with_tool_config(arko_ai::ToolConfig::new(vec![]).with_tool_choice(choice))
        };
        for model in [
            LLMModel::openrouter("anthropic/claude-opus-5.5"),
            LLMModel::anthropic_claude_fable_5_1(),
        ] {
            for choice in [
                arko_ai::ToolChoice::Required,
                arko_ai::ToolChoice::Specific("lookup".into()),
            ] {
                let mut config = luna();
                config.llm_config = AgentLLMConfig {
                    model: model.clone(),
                    generation_config: with_choice(choice.clone()),
                };
                let error = config.validate().unwrap_err().to_string();
                assert!(
                    error.contains("llm_config.generation_config.tool_config.tool_choice")
                        && error.contains(&model.id),
                    "{error}"
                );
                config.llm_config.generation_config = with_choice(arko_ai::ToolChoice::Auto);
                assert!(config.validate().is_ok());
            }
        }
        let mut config = luna();
        config.llm_config.generation_config = with_choice(arko_ai::ToolChoice::Required);
        assert!(config.validate().is_ok(), "GPT takes a forced choice");
        let mut fallback = config.llm_config.clone();
        fallback.model = LLMModel::openai_gpt_6_sol();
        config.refusal_fallback = Some(fallback);
        assert!(config.validate().is_ok());

        let mut config = self::config(
            LLMModel::anthropic_claude_sonnet_4_5(),
            LLMGenerationConfig::default(),
        );
        let mut fallback = config.llm_config.clone();
        fallback.model = LLMModel::anthropic_claude_opus_5_5();
        fallback.generation_config = with_choice(arko_ai::ToolChoice::Required);
        config.refusal_fallback = Some(fallback);
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("refusal_fallback"), "{error}");
    }
}
