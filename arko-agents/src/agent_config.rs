use crate::error::ArkoAgentsError;
use crate::parsing::ToolParsingConfig;
use crate::skill::SkillsConfig;
use crate::task::Task;
use arko_ai::{
    GenerationConfig as LLMGenerationConfig, Model as LLMModel, Provider as LLMProvider,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
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

/// Configuration for a built-in environment detail.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BuiltInEnvironmentDetail {
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
pub struct PerToolConfigOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_approve_tool_use: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_batched_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_of_workspace_path_policy: Option<PathSafetyPolicy>,
}

/// Specifies how tool calls are communicated between the agent and the LLM.
///
/// # Output-Based (default)
///
/// The model outputs tool calls as XML-like structured text which the framework
/// parses. This is the traditional approach and works with all models.
///
/// # Native
///
/// Tool definitions are sent via the LLM API's native tool calling feature.
/// The model responds with structured tool call objects rather than text markup.
/// Recommended for newer models (GPT-5+, Claude Opus 4+, Gemini 2.5+) whose
/// training is centered around native tool use.
///
/// Key behavioral differences in native mode:
/// - Parallel tool calls may involve *different* tools (output-based restricts
///   batches to the same tool).
/// - XML usage examples are not sent to the model; only the tool description
///   and JSON Schema parameters are provided.
/// - Content wrapped in `{{#output_based_only}}...{{/output_based_only}}`
///   markers in `instructions` or `role_description` is stripped.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolCallingMode {
    #[default]
    OutputBased,
    Native,
}

/// Configuration for agent tool usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolConfig {
    pub available_tool_names: Vec<String>,
    #[serde(default)]
    pub allow_batched_tool_calls: bool,
    #[serde(default = "default_auto_approve_tool_use_false")]
    pub default_auto_approve_tool_use: bool,
    #[serde(default)]
    pub default_out_of_workspace_path_policy: PathSafetyPolicy,
    #[serde(default)]
    pub tool_specific_configs: HashMap<String, PerToolConfigOverride>,
    /// How tool calls are communicated with the LLM.
    /// Defaults to `OutputBased` (XML-like syntax parsed from model output).
    #[serde(default)]
    pub tool_calling_mode: ToolCallingMode,
}

fn default_auto_approve_tool_use_false() -> bool {
    false
}

impl Default for AgentToolConfig {
    fn default() -> Self {
        Self {
            available_tool_names: vec![crate::tool::builtins::COMPLETE_TASK_TOOL_NAME.to_string()],
            allow_batched_tool_calls: false,
            default_auto_approve_tool_use: false,
            default_out_of_workspace_path_policy: PathSafetyPolicy::default(),
            tool_specific_configs: HashMap::new(),
            tool_calling_mode: ToolCallingMode::default(),
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
        let model = match (LLMModel::get(helper.provider.clone(), &helper.model_id), snapshot) {
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
    fn default() -> Self {
        LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 3 }
    }
}

/// Timing limits shared by model-call and proactive token-refresh recovery.
/// The elapsed budget starts at the first failure and includes subsequent requests.
/// Explicit capacity errors may opt into a separate indefinite wait under RetryThen* policies.
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
}

impl Default for LLMRetryConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 250,
            max_delay_ms: 64_000,
            max_elapsed_ms: 300_000,
            jitter: true,
            capacity_retry: None,
            rate_limit_retry: Some(LLMRateLimitRetryConfig::default()),
            usage_limit_policy: UsageLimitPolicy::default(),
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
                message: "Rate-limit retry delays must be positive, ordered, and representable.".into(),
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
                message: "Capacity retry delays must be positive, ordered, and representable.".into(),
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
            if !(threshold.percent > 0.0 && threshold.percent <= 100.0) || !delay_ok {
                return Err(ArkoAgentsError::BuilderError {
                    struct_name: "PlanUsageThreshold".into(),
                    message: "percent must be in (0, 100] and a slow-down delay representable.".into(),
                });
            }
        }
        if let Some(cache) = &self.cache {
            let fraction = |value: Option<f64>| value.is_none_or(|value| (0.0..=1.0).contains(&value));
            if cache.window_calls == 0 || !fraction(cache.warn_below) || !fraction(cache.pause_below) {
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum PlanUsageAction {
    /// Pause once per window and reset, like exhausted recovery (awaiting
    /// error-recovery approval, or interrupted under the informing error
    /// policies).
    #[default]
    Pause,
    /// Wait `delay_ms` (cancellable) before every model request while usage
    /// stays at or above the threshold.
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
    /// Inject the reminder every N user messages after the initial task prompt.
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
    #[serde(default)]
    pub tool_config: AgentToolConfig,
    #[serde(default)]
    pub environment_details_config: EnvironmentDetailsConfig,
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
    /// and later turns use `llm_config` again. If the fallback request fails
    /// or also refuses, the refusal stands. Off by default.
    ///
    /// The fallback must use the configured model's provider ([`validate`](AgentConfig::validate)
    /// rejects others): providers can't replay each other's assistant turns.
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

    /// Saves the `AgentConfig` to a JSON file.
    pub fn save_to_path(&self, path: &Path) -> Result<(), ArkoAgentsError> {
        let json_string = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json_string).map_err(|e| ArkoAgentsError::FileOperation {
            path: path.to_path_buf(),
            source: e,
        })
    }

    /// Determines if tool use should be auto-approved for a given tool.
    pub fn should_auto_approve_tool(&self, tool_name: &str) -> bool {
        self.tool_config
            .tool_specific_configs
            .get(tool_name)
            .and_then(|override_cfg| override_cfg.auto_approve_tool_use)
            .unwrap_or(self.tool_config.default_auto_approve_tool_use)
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

    /// Determines if batched calls are allowed for a given tool definition.
    pub fn allows_batched_calls_for_tool(
        &self,
        tool_definition: &crate::tool::ToolDefinition,
    ) -> bool {
        if !tool_definition.supports_batched_calls {
            return false;
        }
        if let Some(override_cfg) = self
            .tool_config
            .tool_specific_configs
            .get(&tool_definition.name)
            && let Some(allow_batched) = override_cfg.allow_batched_calls
        {
            return allow_batched;
        }
        self.tool_config.allow_batched_tool_calls
    }

    /// Gets the tool parsing configuration, using the override if present.
    pub fn get_tool_parsing_config(&self) -> ToolParsingConfig {
        self.parsing_config_override.clone().unwrap_or_default()
    }

    /// Returns true if this agent config uses native tool calling.
    pub fn is_native_tool_calling(&self) -> bool {
        self.tool_config.tool_calling_mode == ToolCallingMode::Native
    }

    /// Checks the settings the builder checks: retry timing, monitors, and
    /// a refusal fallback on the configured model's provider. The runtime
    /// checks them again when a task starts, since configs can be built or
    /// loaded without the builder.
    pub fn validate(&self) -> Result<(), ArkoAgentsError> {
        self.llm_retry_config.validate()?;
        self.monitoring.validate()?;
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

    /// Sets the override for the complete_task summary prompt.
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

    /// Sets the user command timeout in seconds.
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
        assert_eq!(loaded.model.max_input_tokens, LLMModel::openai_gpt_6_astra().max_input_tokens);
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
        let partial = "provider: OpenAI\nmodel_id: gpt-6-sol\ngeneration_config:\n  max_tokens: 900\n";
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
        let zdr = loaded.generation_config.openrouter.as_ref().and_then(|o| o.provider.as_ref()).and_then(|p| p.zdr);
        assert_eq!(zdr, Some(true));

        // A saved snapshot keeps catalog metadata.
        let mut model = LLMModel::openrouter("openai/gpt-6-sol");
        model.max_input_tokens = 1_050_000;
        let config = AgentLLMConfig {
            model: model.clone(),
            generation_config: LLMGenerationConfig::default(),
        };
        let loaded: AgentLLMConfig = serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(loaded.model.max_input_tokens, 1_050_000);
    }
}
