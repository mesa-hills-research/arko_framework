use crate::model::Provider;
use serde::{Deserialize, Serialize};

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
}

impl ReasoningConfig {
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

    pub fn anthropic_with_effort(
        budget_tokens: usize,
        effort: AnthropicReasoningEffort,
    ) -> Self {
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
        }
    }

    pub fn provider(&self) -> Provider {
        match self {
            ReasoningConfig::OpenAI { .. } => Provider::OpenAI,
            ReasoningConfig::Anthropic { .. } => Provider::Anthropic,
            ReasoningConfig::AnthropicAdaptive { .. } => Provider::Anthropic,
            ReasoningConfig::Google { .. } => Provider::Google,
            ReasoningConfig::Gemini { .. } => Provider::Google,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnthropicReasoningFragment {
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
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
    OpenAIEncryptedReasoning {
        items: Vec<OpenAIReasoningItem>,
    },
    AnthropicReasoning {
        fragments: Vec<AnthropicReasoningFragment>,
    },
    GoogleReasoning {
        fragments: Vec<GoogleReasoningFragment>,
    },
}

// Conversations saved before multiple OpenAI reasoning items were kept hold
// a single item's fields directly under `OpenAIEncryptedReasoning`.
#[derive(Deserialize)]
#[allow(clippy::enum_variant_names)] // Names must match the serialized tags.
enum ReasoningArtifactRepr {
    OpenAIEncryptedReasoning(OpenAIReasoningRepr),
    AnthropicReasoning {
        fragments: Vec<AnthropicReasoningFragment>,
    },
    GoogleReasoning {
        fragments: Vec<GoogleReasoningFragment>,
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
            ReasoningArtifactRepr::AnthropicReasoning { fragments } => {
                Self::AnthropicReasoning { fragments }
            }
            ReasoningArtifactRepr::GoogleReasoning { fragments } => {
                Self::GoogleReasoning { fragments }
            }
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

    pub fn provider(&self) -> Provider {
        match self {
            ReasoningArtifact::OpenAIEncryptedReasoning { .. } => Provider::OpenAI,
            ReasoningArtifact::AnthropicReasoning { .. } => Provider::Anthropic,
            ReasoningArtifact::GoogleReasoning { .. } => Provider::Google,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(matches!(round_trip, ReasoningArtifact::OpenAIEncryptedReasoning { items } if items.len() == 2));

        let anthropic = ReasoningArtifact::new_anthropic_redacted("data");
        let round_trip: ReasoningArtifact =
            serde_json::from_value(serde_json::to_value(&anthropic).unwrap()).unwrap();
        assert_eq!(round_trip.provider(), Provider::Anthropic);
    }
}