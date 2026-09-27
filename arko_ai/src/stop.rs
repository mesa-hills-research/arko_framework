//! Provider-agnostic reasons a generation ended, and refusals.
//!
//! Each provider maps its own vocabulary onto [`StopReason`]. The provider's
//! raw value is still recorded in [`ModelCallUsage::stop_reason`] and
//! [`ModelCallUsage::stop_details`] for telemetry.
//!
//! [`ModelCallUsage::stop_reason`]: crate::usage::ModelCallUsage::stop_reason
//! [`ModelCallUsage::stop_details`]: crate::usage::ModelCallUsage::stop_details

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Why the model stopped generating, normalized across providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model finished its turn without requesting tools.
    EndTurn,
    /// The model stopped to have the client run one or more tool calls.
    ToolUse,
    /// Output hit the requested or model output-token limit. Tool calls that
    /// were still being written when the limit was reached are dropped.
    MaxTokens,
    /// A caller-supplied stop sequence was generated.
    StopSequence,
    /// The request or response was refused; see
    /// [`GenerationResponse::refusal`](crate::GenerationResponse::refusal).
    Refusal,
    /// The provider paused a long-running turn that the caller may continue
    /// (Anthropic `pause_turn`).
    PauseTurn,
    /// Generation stopped because the model's context window was exhausted.
    ContextWindowExceeded,
    /// A provider value with no normalized equivalent, kept verbatim.
    Other(String),
}

impl StopReason {
    /// Whether the output was cut short by a token or context limit.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::MaxTokens | Self::ContextWindowExceeded)
    }

    /// Placeholder for responses serialized before stop reasons existed.
    pub(crate) fn unknown() -> Self {
        Self::Other("unknown".to_string())
    }
}

/// Who declined to produce the requested output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalSource {
    /// A provider safety classifier stopped generation (Anthropic
    /// `stop_reason: "refusal"`). Anthropic requires the refused turn to be
    /// removed from history before continuing, so the response message is
    /// left empty and any text generated before the refusal is moved to
    /// [`Refusal::partial_output`].
    SafetyClassifier,
    /// The model declined in its own words (OpenAI structured `refusal`
    /// content). The response is otherwise a normal assistant turn; the
    /// refusal text is in [`Refusal::explanation`], not in the message text.
    Model,
    /// A content filter around the model cut the output (OpenAI
    /// `incomplete_details.reason: "content_filter"`). Like
    /// [`SafetyClassifier`](Self::SafetyClassifier), partial output is moved
    /// out of the message.
    ContentFilter,
}

impl RefusalSource {
    /// Whether the refused turn may remain in conversation history.
    pub fn keeps_turn(&self) -> bool {
        matches!(self, Self::Model)
    }
}

/// Details of a refusal, carried by both
/// [`GenerationResponse::refusal`](crate::GenerationResponse::refusal) and
/// [`StreamEvent::Refusal`](crate::StreamEvent::Refusal).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Refusal {
    pub source: RefusalSource,
    /// Provider policy category, when supplied (for example Anthropic
    /// `stop_details.category`). May be absent even for classifier refusals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Human-readable explanation suitable for display. Its wording is not
    /// stable; don't parse it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    /// Text generated before the refusal. It is incomplete and was removed
    /// from the message so it isn't sent back as history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_output: Option<String>,
    /// The provider's raw refusal data (for example Anthropic `stop_details`).
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub provider_details: Value,
}

impl Refusal {
    pub fn new(source: RefusalSource) -> Self {
        Self {
            source,
            category: None,
            explanation: None,
            partial_output: None,
            provider_details: Value::Null,
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let source = match self.source {
            RefusalSource::SafetyClassifier => "safety classifier",
            RefusalSource::Model => "model",
            RefusalSource::ContentFilter => "content filter",
        };
        write!(f, "refused by {source}")?;
        if let Some(category) = &self.category {
            write!(f, " ({category})")?;
        }
        if let Some(explanation) = &self.explanation {
            write!(f, ": {explanation}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reasons_round_trip_including_unknown_values() {
        for reason in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::Other("compaction".into()),
        ] {
            let value = serde_json::to_value(&reason).unwrap();
            assert_eq!(serde_json::from_value::<StopReason>(value).unwrap(), reason);
        }
        assert_eq!(serde_json::to_value(StopReason::ToolUse).unwrap(), "tool_use");
    }

    #[test]
    fn refusal_display_and_serialization_omit_absent_fields() {
        let mut refusal = Refusal::new(RefusalSource::SafetyClassifier);
        refusal.category = Some("cyber".into());
        refusal.explanation = Some("Declined.".into());
        assert_eq!(refusal.to_string(), "refused by safety classifier (cyber): Declined.");
        let value = serde_json::to_value(Refusal::new(RefusalSource::Model)).unwrap();
        assert_eq!(value, serde_json::json!({"source": "model"}));
        assert!(!RefusalSource::SafetyClassifier.keeps_turn());
        assert!(RefusalSource::Model.keeps_turn());
    }
}
