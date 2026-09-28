use crate::model::Provider;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Defines the effort level for OpenAI reasoning models.
///
/// GPT-5.6 supports `None`, `Low`, `Medium`, `High`, `XHigh`, and `Max`.
/// Earlier models support provider-specific subsets that are validated when
/// the request is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIReasoningMode {
    Standard,
    Pro,
}

/// Defines the effort level for Anthropic reasoning.
///
/// `Low`, `Medium`, and `High` are available on all supported effort models.
///
/// `XHigh` is available on Claude Fable 5, Claude Opus 4.8, Claude Opus 4.7,
/// and Claude Sonnet 5.
///
/// `Max` is available on Claude Fable 5, Claude Opus 4.8, Claude Opus 4.7,
/// Claude Opus 4.6, Claude Sonnet 5, and Claude Sonnet 4.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicReasoningEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// Defines the thinking level for Gemini 3.x reasoning models.
///
/// Gemini 3.5 Flash supports `Minimal`, `Low`, `Medium`, and `High`, with
/// `Medium` as its provider default.
///
/// Supported values are validated against the selected model when a request
/// is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeminiThinkingLevel {
    #[serde(rename = "MINIMAL")]
    Minimal,
    #[serde(rename = "LOW")]
    Low,
    #[serde(rename = "MEDIUM")]
    Medium,
    #[serde(rename = "HIGH")]
    High,
}

impl GeminiThinkingLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            GeminiThinkingLevel::Minimal => "MINIMAL",
            GeminiThinkingLevel::Low => "LOW",
            GeminiThinkingLevel::Medium => "MEDIUM",
            GeminiThinkingLevel::High => "HIGH",
        }
    }
}

/// Provider-agnostic reasoning effort, from no reasoning to the most.
///
/// Used by [`OpenRouterReasoning`] and by mid-conversation effort updates
/// ([`Conversation::add_reasoning_effort_update`]). OpenRouter translates
/// each level to the served model's vocabulary and clamps unsupported levels
/// to the nearest supported one, except `None`, which models with mandatory
/// reasoning reject.
///
/// [`Conversation::add_reasoning_effort_update`]: crate::conversation::Conversation::add_reasoning_effort_update
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    /// The wire value (`"xhigh"`, `"none"`, ...).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl From<OpenAIEffort> for ReasoningEffort {
    fn from(effort: OpenAIEffort) -> Self {
        match effort {
            OpenAIEffort::None => Self::None,
            OpenAIEffort::Minimal => Self::Minimal,
            OpenAIEffort::Low => Self::Low,
            OpenAIEffort::Medium => Self::Medium,
            OpenAIEffort::High => Self::High,
            OpenAIEffort::XHigh => Self::XHigh,
            OpenAIEffort::Max => Self::Max,
        }
    }
}

impl From<AnthropicReasoningEffort> for ReasoningEffort {
    fn from(effort: AnthropicReasoningEffort) -> Self {
        match effort {
            AnthropicReasoningEffort::Low => Self::Low,
            AnthropicReasoningEffort::Medium => Self::Medium,
            AnthropicReasoningEffort::High => Self::High,
            AnthropicReasoningEffort::XHigh => Self::XHigh,
            AnthropicReasoningEffort::Max => Self::Max,
        }
    }
}

impl From<GeminiThinkingLevel> for ReasoningEffort {
    fn from(level: GeminiThinkingLevel) -> Self {
        match level {
            GeminiThinkingLevel::Minimal => Self::Minimal,
            GeminiThinkingLevel::Low => Self::Low,
            GeminiThinkingLevel::Medium => Self::Medium,
            GeminiThinkingLevel::High => Self::High,
        }
    }
}

/// OpenRouter `reasoning.summary` verbosity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// OpenRouter `reasoning.context`: which replayed reasoning the model may use
/// (OpenAI GPT-5.6 and newer only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningContext {
    Auto,
    AllTurns,
    CurrentTurn,
}

/// The OpenRouter Chat Completions `reasoning` object. Serializes to the wire
/// form with unset fields omitted, so the model's defaults apply (catalog
/// `reasoning.default_enabled` / `default_effort`).
/// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
///
/// `effort` and `max_tokens` are mutually exclusive; requests setting both
/// are rejected before sending. `exclude: true` hides reasoning from the
/// response (it is still generated and billed); whether it also withholds the
/// `reasoning_details` needed to continue tool calls is undocumented, so avoid
/// it in tool loops.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Reasoning token budget (Anthropic-style); at least 1024 on Claude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ReasoningContext>,
    /// `pro` routes to the model's pro variant (OpenAI GPT-5.6 and newer).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<OpenAIReasoningMode>,
}

impl OpenRouterReasoning {
    /// Reasoning at a given effort.
    pub fn effort(effort: ReasoningEffort) -> Self {
        Self {
            effort: Some(effort),
            ..Self::default()
        }
    }

    /// Reasoning with a token budget.
    pub fn max_tokens(max_tokens: usize) -> Self {
        Self {
            max_tokens: Some(max_tokens),
            ..Self::default()
        }
    }

    /// Reasoning on, with the model's default effort (`enabled: true`).
    pub fn enabled() -> Self {
        Self {
            enabled: Some(true),
            ..Self::default()
        }
    }

    /// Reasoning off (`enabled: false`); rejected by models whose reasoning is
    /// mandatory.
    pub fn disabled() -> Self {
        Self {
            enabled: Some(false),
            ..Self::default()
        }
    }

    pub fn with_exclude(mut self, exclude: bool) -> Self {
        self.exclude = Some(exclude);
        self
    }

    pub fn with_summary(mut self, summary: ReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }

    pub fn with_context(mut self, context: ReasoningContext) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_mode(mut self, mode: OpenAIReasoningMode) -> Self {
        self.mode = Some(mode);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningConfig {
    OpenAI {
        effort: OpenAIEffort,
        include_summaries: bool,
    },
    Anthropic {
        budget_tokens: usize,
        effort: Option<AnthropicReasoningEffort>,
    },
    /// Anthropic adaptive thinking where the model determines its own
    /// reasoning budget.
    ///
    /// Supported on Claude Fable 5, Claude Opus 4.8, Claude Opus 4.7,
    /// Claude Opus 4.6, Claude Sonnet 5, and Claude Sonnet 4.6.
    ///
    /// Claude Fable 5 always uses adaptive thinking. Claude Sonnet 5 also
    /// enables adaptive thinking by default when this configuration is absent.
    AnthropicAdaptive {
        effort: Option<AnthropicReasoningEffort>,
    },
    Google {
        thinking_budget: isize,
        include_summaries: bool,
    },
    /// Gemini 3.x thinking-level-based reasoning configuration.
    ///
    /// Supported levels are validated against the selected model when the
    /// request is built.
    Gemini {
        thinking_level: GeminiThinkingLevel,
        include_summaries: bool,
    },
    /// OpenRouter's unified `reasoning` object. The only reasoning
    /// configuration OpenRouter models accept; convert another provider's
    /// configuration explicitly with [`ReasoningConfig::to_openrouter`].
    OpenRouter(OpenRouterReasoning),
}

impl ReasoningConfig {
    /// An OpenRouter reasoning configuration.
    pub fn openrouter(reasoning: OpenRouterReasoning) -> Self {
        ReasoningConfig::OpenRouter(reasoning)
    }

    /// OpenRouter reasoning at a given effort.
    pub fn openrouter_effort(effort: ReasoningEffort) -> Self {
        ReasoningConfig::OpenRouter(OpenRouterReasoning::effort(effort))
    }

    /// The closest OpenRouter equivalent of this configuration. Nothing is
    /// translated implicitly: OpenRouter models drop other variants (with a
    /// warning), and this conversion fails where no equivalent exists.
    ///
    /// - `OpenAI`: the effort, plus `summary: "auto"` when summaries were requested.
    /// - `Anthropic`: the budget as `max_tokens`, or `enabled: false` for a zero
    ///   budget. A budget combined with an effort fails: OpenRouter accepts
    ///   one or the other.
    /// - `AnthropicAdaptive`: the effort, or `enabled: true` without one.
    /// - `Google`: `-1` (dynamic) becomes `enabled: true`, `0` `enabled: false`
    ///   and a positive budget `max_tokens`.
    /// - `Gemini`: the thinking level as an effort.
    ///
    /// `include_summaries: false` on the Google variants is not carried over:
    /// OpenRouter returns reasoning whenever the model produces it, and its
    /// only switch (`exclude`) may also withhold the signatures needed to
    /// continue tool calls.
    pub fn to_openrouter(&self) -> Result<OpenRouterReasoning, crate::error::Error> {
        let invalid = |message: &str| Err(crate::error::Error::InvalidRequest(message.to_string()));
        Ok(match self {
            ReasoningConfig::OpenAI {
                effort,
                include_summaries,
            } => OpenRouterReasoning {
                effort: Some((*effort).into()),
                summary: include_summaries.then_some(ReasoningSummary::Auto),
                ..OpenRouterReasoning::default()
            },
            ReasoningConfig::Anthropic {
                budget_tokens,
                effort: None,
            } => {
                if *budget_tokens > 0 {
                    OpenRouterReasoning::max_tokens(*budget_tokens)
                } else {
                    OpenRouterReasoning::disabled()
                }
            }
            ReasoningConfig::Anthropic {
                effort: Some(_), ..
            } => {
                return invalid(
                    "OpenRouter reasoning takes either an effort or a token budget, not both; use OpenRouterReasoning::effort or ::max_tokens.",
                );
            }
            ReasoningConfig::AnthropicAdaptive { effort } => match effort {
                Some(effort) => OpenRouterReasoning::effort((*effort).into()),
                None => OpenRouterReasoning::enabled(),
            },
            ReasoningConfig::Google {
                thinking_budget, ..
            } => match *thinking_budget {
                -1 => OpenRouterReasoning::enabled(),
                0 => OpenRouterReasoning::disabled(),
                budget if budget > 0 => OpenRouterReasoning::max_tokens(budget as usize),
                _ => {
                    return invalid(
                        "A negative thinking budget other than -1 (dynamic) has no OpenRouter equivalent.",
                    );
                }
            },
            ReasoningConfig::Gemini { thinking_level, .. } => {
                OpenRouterReasoning::effort((*thinking_level).into())
            }
            ReasoningConfig::OpenRouter(reasoning) => reasoning.clone(),
        })
    }

    pub fn openai(effort: OpenAIEffort, include_summaries: bool) -> Self {
        ReasoningConfig::OpenAI {
            effort,
            include_summaries,
        }
    }

    pub fn anthropic(budget_tokens: usize) -> Self {
        ReasoningConfig::Anthropic {
            budget_tokens,
            effort: None,
        }
    }

    pub fn anthropic_with_effort(budget_tokens: usize, effort: AnthropicReasoningEffort) -> Self {
        ReasoningConfig::Anthropic {
            budget_tokens,
            effort: Some(effort),
        }
    }

    /// Creates an Anthropic adaptive-thinking configuration.
    pub fn anthropic_adaptive() -> Self {
        ReasoningConfig::AnthropicAdaptive { effort: None }
    }

    /// Creates an Anthropic adaptive-thinking configuration with an effort level.
    pub fn anthropic_adaptive_with_effort(effort: AnthropicReasoningEffort) -> Self {
        ReasoningConfig::AnthropicAdaptive {
            effort: Some(effort),
        }
    }

    pub fn google(thinking_budget: isize, include_summaries: bool) -> Self {
        ReasoningConfig::Google {
            thinking_budget,
            include_summaries,
        }
    }

    /// Creates a Gemini 3.x thinking-level configuration.
    pub fn gemini(thinking_level: GeminiThinkingLevel, include_summaries: bool) -> Self {
        ReasoningConfig::Gemini {
            thinking_level,
            include_summaries,
        }
    }

    pub fn is_enabled(&self) -> bool {
        match self {
            ReasoningConfig::OpenAI { .. } => true,
            ReasoningConfig::Anthropic { budget_tokens, .. } => *budget_tokens > 0,
            ReasoningConfig::AnthropicAdaptive { .. } => true,
            ReasoningConfig::Google {
                thinking_budget, ..
            } => *thinking_budget != 0,
            ReasoningConfig::Gemini { .. } => true,
            ReasoningConfig::OpenRouter(reasoning) => {
                reasoning.enabled != Some(false) && reasoning.effort != Some(ReasoningEffort::None)
            }
        }
    }

    pub fn provider(&self) -> Provider {
        match self {
            ReasoningConfig::OpenAI { .. } => Provider::OpenAI,
            ReasoningConfig::Anthropic { .. } => Provider::Anthropic,
            ReasoningConfig::AnthropicAdaptive { .. } => Provider::Anthropic,
            ReasoningConfig::Google { .. } => Provider::Google,
            ReasoningConfig::Gemini { .. } => Provider::Google,
            ReasoningConfig::OpenRouter(_) => Provider::OpenRouter,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnthropicReasoningFragment {
    Thinking { thinking: String, signature: String },
    RedactedThinking { data: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleReasoningFragment {
    pub summary: String,
    pub signature: String,
}

/// One OpenAI Responses `reasoning` output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIReasoningItem {
    pub encrypted_content: String,
    pub summaries: Option<Vec<String>>,
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "ReasoningArtifactRepr")]
pub enum ReasoningArtifact {
    /// Every reasoning item of one OpenAI response, in output order. All of
    /// them are replayed, ahead of the turn's function calls.
    ///
    /// Written by earlier versions; responses now keep
    /// [`OpenAIOutputItems`](Self::OpenAIOutputItems). Conversations saved
    /// with this form still replay as before: the reasoning items, then the
    /// message's function calls, then its text.
    OpenAIEncryptedReasoning { items: Vec<OpenAIReasoningItem> },
    /// The replayable output items of one OpenAI Responses call, in output
    /// order and exactly as the provider returned them, minus `status`:
    /// `reasoning` items that carry `encrypted_content`, `message` items
    /// (with their `phase`, if any) and finished `function_call` items.
    ///
    /// Later requests replay them verbatim and in this order, for example
    /// `reasoning`, `message` (commentary), `function_call`, followed by the
    /// call's `function_call_output`. That keeps the prompt the server
    /// renders the same from request to request (prompt caching) and lets
    /// reasoning carry across turns with `store: false`. A `function_call`
    /// item is replayed only while the message still holds that call (by
    /// `call_id`), so every replayed call has an output. When the items hold
    /// no `message`, the message's own text (for example the explanation an
    /// agent keeps for a refused turn) is replayed after them.
    OpenAIOutputItems { items: Vec<Value> },
    AnthropicReasoning {
        fragments: Vec<AnthropicReasoningFragment>,
    },
    GoogleReasoning {
        fragments: Vec<GoogleReasoningFragment>,
    },
    /// OpenRouter `reasoning_details` items, kept verbatim and in order. They
    /// are sent back unmodified on the assistant turn: they carry the
    /// signatures and encrypted reasoning that Claude, Gemini and OpenAI models
    /// need to continue after tool calls. Items are opaque JSON; encrypted
    /// items (`reasoning.encrypted`) hold no readable text.
    OpenRouter { details: Vec<serde_json::Value> },
}

// Conversations saved before multiple OpenAI reasoning items were kept hold
// a single item's fields directly under `OpenAIEncryptedReasoning`.
#[derive(Deserialize)]
#[allow(clippy::enum_variant_names)] // Names must match the serialized tags.
enum ReasoningArtifactRepr {
    OpenAIEncryptedReasoning(OpenAIReasoningRepr),
    OpenAIOutputItems {
        items: Vec<Value>,
    },
    AnthropicReasoning {
        fragments: Vec<AnthropicReasoningFragment>,
    },
    GoogleReasoning {
        fragments: Vec<GoogleReasoningFragment>,
    },
    OpenRouter {
        details: Vec<serde_json::Value>,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAIReasoningRepr {
    Items { items: Vec<OpenAIReasoningItem> },
    Single(OpenAIReasoningItem),
}

impl From<ReasoningArtifactRepr> for ReasoningArtifact {
    fn from(repr: ReasoningArtifactRepr) -> Self {
        match repr {
            ReasoningArtifactRepr::OpenAIEncryptedReasoning(OpenAIReasoningRepr::Items {
                items,
            }) => Self::OpenAIEncryptedReasoning { items },
            ReasoningArtifactRepr::OpenAIEncryptedReasoning(OpenAIReasoningRepr::Single(item)) => {
                Self::OpenAIEncryptedReasoning { items: vec![item] }
            }
            ReasoningArtifactRepr::OpenAIOutputItems { items } => Self::OpenAIOutputItems { items },
            ReasoningArtifactRepr::AnthropicReasoning { fragments } => {
                Self::AnthropicReasoning { fragments }
            }
            ReasoningArtifactRepr::GoogleReasoning { fragments } => {
                Self::GoogleReasoning { fragments }
            }
            ReasoningArtifactRepr::OpenRouter { details } => Self::OpenRouter { details },
        }
    }
}

impl ReasoningArtifact {
    pub fn new_openai_encrypted(encrypted_content: &str, summaries: Option<Vec<String>>) -> Self {
        ReasoningArtifact::OpenAIEncryptedReasoning {
            items: vec![OpenAIReasoningItem {
                encrypted_content: encrypted_content.to_string(),
                summaries,
                id: None,
            }],
        }
    }

    pub fn new_anthropic_thinking(thinking: &str, signature: &str) -> Self {
        ReasoningArtifact::AnthropicReasoning {
            fragments: vec![AnthropicReasoningFragment::Thinking {
                thinking: thinking.to_string(),
                signature: signature.to_string(),
            }],
        }
    }

    pub fn new_anthropic_redacted(data: &str) -> Self {
        ReasoningArtifact::AnthropicReasoning {
            fragments: vec![AnthropicReasoningFragment::RedactedThinking {
                data: data.to_string(),
            }],
        }
    }

    pub fn new_anthropic_mixed(fragments: Vec<AnthropicReasoningFragment>) -> Self {
        ReasoningArtifact::AnthropicReasoning { fragments }
    }

    pub fn new_google_reasoning(fragments: Vec<GoogleReasoningFragment>) -> Self {
        ReasoningArtifact::GoogleReasoning { fragments }
    }

    /// OpenRouter `reasoning_details`, kept verbatim.
    pub fn new_openrouter(details: Vec<serde_json::Value>) -> Self {
        ReasoningArtifact::OpenRouter { details }
    }

    pub fn provider(&self) -> Provider {
        match self {
            ReasoningArtifact::OpenAIEncryptedReasoning { .. }
            | ReasoningArtifact::OpenAIOutputItems { .. } => Provider::OpenAI,
            ReasoningArtifact::AnthropicReasoning { .. } => Provider::Anthropic,
            ReasoningArtifact::GoogleReasoning { .. } => Provider::Google,
            ReasoningArtifact::OpenRouter { .. } => Provider::OpenRouter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_efforts_use_wire_names_and_convert_from_provider_levels() {
        use ReasoningEffort::*;
        for effort in [None, Minimal, Low, Medium, High, XHigh, Max] {
            assert_eq!(serde_json::to_value(effort).unwrap(), effort.as_str());
        }
        assert_eq!(ReasoningEffort::from(OpenAIEffort::None), None);
        assert_eq!(ReasoningEffort::from(OpenAIEffort::XHigh), XHigh);
        assert_eq!(ReasoningEffort::from(AnthropicReasoningEffort::Max), Max);
        assert_eq!(ReasoningEffort::from(GeminiThinkingLevel::Minimal), Minimal);
    }

    #[test]
    fn openai_artifacts_saved_with_a_single_item_still_load() {
        let saved = serde_json::json!({"OpenAIEncryptedReasoning": {
            "encrypted_content": "enc", "summaries": ["s"], "id": "rs_1"
        }});
        let ReasoningArtifact::OpenAIEncryptedReasoning { items } =
            serde_json::from_value(saved).unwrap()
        else {
            panic!("expected OpenAI reasoning");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id.as_deref(), Some("rs_1"));

        let artifact = ReasoningArtifact::OpenAIEncryptedReasoning {
            items: vec![items[0].clone(), items[0].clone()],
        };
        let round_trip: ReasoningArtifact =
            serde_json::from_value(serde_json::to_value(&artifact).unwrap()).unwrap();
        assert!(
            matches!(round_trip, ReasoningArtifact::OpenAIEncryptedReasoning { items } if items.len() == 2)
        );

        let items = serde_json::json!([
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "phase": "commentary", "content": []},
        ]);
        let artifact = ReasoningArtifact::OpenAIOutputItems {
            items: items.as_array().unwrap().clone(),
        };
        let round_trip: ReasoningArtifact =
            serde_json::from_value(serde_json::to_value(&artifact).unwrap()).unwrap();
        assert!(
            matches!(&round_trip, ReasoningArtifact::OpenAIOutputItems { items: kept } if Value::from(kept.clone()) == items)
        );
        assert_eq!(round_trip.provider(), Provider::OpenAI);

        let anthropic = ReasoningArtifact::new_anthropic_redacted("data");
        let round_trip: ReasoningArtifact =
            serde_json::from_value(serde_json::to_value(&anthropic).unwrap()).unwrap();
        assert_eq!(round_trip.provider(), Provider::Anthropic);
    }

    /// Artifacts load through `ReasoningArtifactRepr`, so a variant missing
    /// there still compiles but fails to load. Every variant must round-trip.
    #[test]
    fn every_reasoning_artifact_variant_round_trips() {
        // Exhaustive on purpose: a new variant fails to compile here until it
        // gets a sample below.
        fn tag(artifact: &ReasoningArtifact) -> &'static str {
            match artifact {
                ReasoningArtifact::OpenAIEncryptedReasoning { .. } => "OpenAIEncryptedReasoning",
                ReasoningArtifact::OpenAIOutputItems { .. } => "OpenAIOutputItems",
                ReasoningArtifact::AnthropicReasoning { .. } => "AnthropicReasoning",
                ReasoningArtifact::GoogleReasoning { .. } => "GoogleReasoning",
                ReasoningArtifact::OpenRouter { .. } => "OpenRouter",
            }
        }
        let samples = [
            ReasoningArtifact::new_openai_encrypted("enc", Some(vec!["s".into()])),
            ReasoningArtifact::OpenAIOutputItems {
                items: vec![
                    serde_json::json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []}),
                ],
            },
            ReasoningArtifact::new_anthropic_mixed(vec![
                AnthropicReasoningFragment::Thinking {
                    thinking: "t".into(),
                    signature: "sig".into(),
                },
                AnthropicReasoningFragment::RedactedThinking {
                    data: "data".into(),
                },
            ]),
            ReasoningArtifact::new_google_reasoning(vec![GoogleReasoningFragment {
                summary: "s".into(),
                signature: "sig".into(),
            }]),
            ReasoningArtifact::new_openrouter(vec![
                serde_json::json!({"type": "reasoning.encrypted", "data": "enc"}),
            ]),
        ];
        let tags: std::collections::BTreeSet<_> = samples.iter().map(tag).collect();
        assert_eq!(tags.len(), samples.len(), "one sample per variant");
        for artifact in &samples {
            let saved = serde_json::to_value(artifact).unwrap();
            let loaded: ReasoningArtifact = serde_json::from_value(saved.clone())
                .unwrap_or_else(|error| panic!("{} did not load: {error}", tag(artifact)));
            assert_eq!(tag(&loaded), tag(artifact));
            assert_eq!(serde_json::to_value(&loaded).unwrap(), saved);
        }
    }
}
