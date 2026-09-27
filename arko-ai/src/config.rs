use crate::reasoning::{OpenAIReasoningMode, ReasoningConfig};
use crate::tool::ToolConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationConfig {
    /// Sent only when `temperature_specified` is set; otherwise the model's
    /// default applies. Google's Interactions API no longer accepts it.
    pub temperature: f64,
    pub temperature_specified: bool,
    pub max_tokens: Option<usize>,
    pub verbosity: Option<Verbosity>,
    pub reasoning: Option<ReasoningConfig>,
    pub openai_reasoning_mode: Option<OpenAIReasoningMode>,
    /// Controls Anthropic thinking independently from the selected reasoning mode.
    ///
    /// `None` preserves existing behavior and uses the model's normal default.
    /// A supplied Anthropic reasoning configuration still enables thinking.
    ///
    /// `Some(true)` explicitly enables thinking. When no Anthropic reasoning
    /// configuration is supplied, adaptive thinking is selected for models that
    /// support it.
    ///
    /// `Some(false)` explicitly sends `thinking.type = "disabled"`. This is
    /// rejected for models such as Claude Fable 5 where adaptive thinking is
    /// always enabled.
    pub anthropic_thinking_enabled: Option<bool>,
    /// Controls prompt caching when the provider exposes a convenient
    /// request-level mechanism.
    ///
    /// `None` inherits [`Conversation::enable_cache`], which defaults to
    /// enabled.
    ///
    /// For Anthropic, the resolved value controls ephemeral prompt caching.
    ///
    /// For GPT-5.6 models, enabled selects implicit prompt caching and disabled
    /// selects explicit-only mode without breakpoints, preventing prompt-cache
    /// reads and writes. Older OpenAI models retain their automatic provider
    /// behavior because they do not accept GPT-5.6 prompt-cache options.
    ///
    /// Gemini implicit caching remains provider-managed and cannot be disabled
    /// through this setting.
    ///
    /// [`Conversation::enable_cache`]: crate::conversation::Conversation::enable_cache
    pub prompt_caching: Option<bool>,
    /// Requests summarized Anthropic thinking text when adaptive thinking is enabled.
    ///
    /// Supported by Claude Fable 5, Claude Opus 4.8, Claude Opus 4.7, and
    /// Claude Sonnet 5. `None` and `Some(false)` keep the provider default.
    pub anthropic_summarized_thinking: Option<bool>,
    /// Enables the opt-in 1M context beta for Anthropic models that still require it.
    ///
    /// Currently this is used for Claude Sonnet 4 and Claude Sonnet 4.5.
    ///
    /// Newer supported Claude models include the full 1M context window at
    /// standard pricing. For those models this flag is accepted for
    /// compatibility but does not add a beta header or alter pricing.
    pub long_context: Option<bool>,
    /// Tool calling configuration. When set, tool definitions are sent
    /// with the request and the model may respond with tool calls instead
    /// of or in addition to text.
    pub tool_config: Option<ToolConfig>,
    /// Enables interleaved thinking for supported Anthropic models.
    ///
    /// When enabled, Claude can produce thinking blocks after receiving tool
    /// results, allowing it to reason about intermediate results before
    /// deciding what to do next.
    ///
    /// `None` uses the model's normal behavior. Models with automatic
    /// interleaved thinking enable it whenever adaptive thinking is active.
    ///
    /// `Some(true)` explicitly enables the feature and requires supported
    /// Anthropic reasoning.
    ///
    /// `Some(false)` explicitly disables the feature. This is rejected when
    /// the selected model automatically requires interleaved thinking with its
    /// active adaptive-thinking mode.
    pub interleaved_thinking: Option<bool>,
}

impl GenerationConfig {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            temperature_specified: true,
            max_tokens: None,
            verbosity: None,
            reasoning: None,
            openai_reasoning_mode: None,
            anthropic_thinking_enabled: None,
            prompt_caching: None,
            anthropic_summarized_thinking: None,
            long_context: None,
            tool_config: None,
            interleaved_thinking: None,
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self.temperature_specified = true;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_openai_reasoning_mode(
        mut self,
        mode: OpenAIReasoningMode,
    ) -> Self {
        self.openai_reasoning_mode = Some(mode);
        self
    }

    /// Explicitly enables or disables Anthropic thinking.
    ///
    /// See [`anthropic_thinking_enabled`](Self::anthropic_thinking_enabled)
    /// for model-specific behavior.
    pub fn with_anthropic_thinking_enabled(mut self, enabled: bool) -> Self {
        self.anthropic_thinking_enabled = Some(enabled);
        self
    }

    /// Enables or disables supported provider prompt caching.
    ///
    /// See [`prompt_caching`](Self::prompt_caching) for provider-specific
    /// behavior. When not specified, the conversation-level setting is used.
    pub fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = Some(enabled);
        self
    }

    /// Requests summarized Anthropic thinking text when adaptive thinking is enabled.
    pub fn with_anthropic_summarized_thinking(mut self, enabled: bool) -> Self {
        self.anthropic_summarized_thinking = Some(enabled);
        self
    }

    /// Enables the opt-in extended context beta for Anthropic models that still require it.
    pub fn with_long_context(mut self, enabled: bool) -> Self {
        self.long_context = Some(enabled);
        self
    }

    pub fn with_tool_config(mut self, tool_config: ToolConfig) -> Self {
        self.tool_config = Some(tool_config);
        self
    }

    /// Enables or disables interleaved thinking for supported Anthropic models.
    pub fn with_interleaved_thinking(mut self, enabled: bool) -> Self {
        self.interleaved_thinking = Some(enabled);
        self
    }
}