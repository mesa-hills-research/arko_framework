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
    /// Voyage's `truncation` flag. `None` sends `false` (Voyage's own
    /// default is `true`). arko never truncates silently: an input over the
    /// model's per-input token limit fails locally with
    /// `Error::TokenLimitExceeded` before any request, whatever this is set
    /// to, so no input over the limit reaches Voyage to be truncated.
    pub voyage_truncation: Option<bool>,
    /// Hugging Face access token for tokenizers that require authentication.
    /// Never serialized, so a saved config doesn't hold the secret (`Debug`
    /// redacts it too): after loading a config, set it again with
    /// [`with_hf_token`](Self::with_hf_token), from wherever the application
    /// keeps it (for example an environment variable it reads). A config
    /// saved by an earlier version that holds it still loads it.
    #[serde(skip_serializing)]
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

    /// The token is never written when a config is serialized (it was, in
    /// plain text); a loaded config has none until it is set again, and a
    /// config an earlier version saved with one still loads it.
    #[test]
    fn the_hugging_face_token_is_never_serialized() {
        let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-4")
            .with_hf_token("hf_synthetic_secret".into())
            .with_tokenizer_revision("main");
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("hf_synthetic_secret"), "{json}");
        assert!(!json.contains("hf_token"), "{json}");
        let loaded: EmbeddingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.hf_token, None);
        assert_eq!(loaded.tokenizer_revision.as_deref(), Some("main"));
        let loaded = loaded.with_hf_token("hf_synthetic_secret".into());
        assert_eq!(loaded.hf_token.as_deref(), Some("hf_synthetic_secret"));

        let older = r#"{"provider":"Voyage","model_id":"voyage-4","output_dimensionality":null,"voyage_input_type":null,"voyage_truncation":null,"hf_token":"hf_synthetic_secret","tokenizer_revision":null}"#;
        let loaded: EmbeddingConfig = serde_json::from_str(older).unwrap();
        assert_eq!(loaded.hf_token.as_deref(), Some("hf_synthetic_secret"));
    }

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
