use crate::embedding_provider_identifier::EmbeddingProviderIdentifier;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoyageInputType {
    Query,
    Document,
}

impl VoyageInputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VoyageInputType::Query => "query",
            VoyageInputType::Document => "document",
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProviderIdentifier,
    pub model_id: String,
    pub output_dimensionality: Option<usize>,
    pub voyage_input_type: Option<VoyageInputType>,
    /// If `None`, the provider default is used (false for our implementation).
    pub voyage_truncation: Option<bool>,
    /// Hugging Face access token for tokenizers that require authentication.
    pub hf_token: Option<String>,
    /// Pin token accounting to a Hugging Face revision.
    #[serde(default)]
    pub tokenizer_revision: Option<String>,
}

impl EmbeddingConfig {
    pub fn new(provider: EmbeddingProviderIdentifier, model_id: &str) -> Self {
        Self {
            provider,
            model_id: model_id.to_string(),
            output_dimensionality: None,
            voyage_input_type: None,
            voyage_truncation: None,
            hf_token: None,
            tokenizer_revision: None,
        }
    }

    pub fn with_output_dimensionality(mut self, dim: usize) -> Self {
        self.output_dimensionality = Some(dim);
        self
    }

    pub fn with_voyage_input_type(mut self, input_type: VoyageInputType) -> Self {
        self.voyage_input_type = Some(input_type);
        self
    }

    pub fn with_voyage_truncation(mut self, truncation: bool) -> Self {
        self.voyage_truncation = Some(truncation);
        self
    }

    pub fn with_tokenizer_revision(mut self, revision: impl Into<String>) -> Self {
        self.tokenizer_revision = Some(revision.into());
        self
    }

    pub fn with_hf_token(mut self, token: String) -> Self {
        self.hf_token = Some(token);
        self
    }
}

// The Hugging Face token never appears in `Debug` output.
impl std::fmt::Debug for EmbeddingConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddingConfig")
            .field("provider", &self.provider)
            .field("model_id", &self.model_id)
            .field("output_dimensionality", &self.output_dimensionality)
            .field("voyage_input_type", &self.voyage_input_type)
            .field("voyage_truncation", &self.voyage_truncation)
            .field("hf_token", &self.hf_token.as_ref().map(|_| "<redacted>"))
            .field("tokenizer_revision", &self.tokenizer_revision)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_redacts_the_hugging_face_token() {
        let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-4")
            .with_hf_token("hf_synthetic_secret".into());
        let debug = format!("{config:?}");
        assert!(
            debug.contains("voyage-4") && debug.contains("<redacted>"),
            "{debug}"
        );
        assert!(!debug.contains("hf_synthetic_secret"), "{debug}");
    }
}
