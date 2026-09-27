use crate::embedding_provider_identifier::EmbeddingProviderIdentifier;
use crate::error::Error;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelDetails {
    pub provider: EmbeddingProviderIdentifier,
    pub id: String,
    /// The ID used in API calls, e.g., "voyage-4".
    pub api_id: String,
    pub max_input_tokens_per_item: usize,
    /// Maximum total tokens allowed in a single batch, when the provider
    /// publishes one. `None` means no known local limit; the provider may
    /// still enforce a limit.
    pub max_batch_total_tokens: Option<usize>,
    /// Maximum number of items allowed in a single batch, when the provider
    /// enforces one.
    pub max_batch_size: Option<usize>,
    /// Hugging Face tokenizer model name used for local token counting.
    pub tokenizer_id: String,
    pub requires_hf_token: bool,
    pub cost_per_million_input_tokens: f64,
    pub default_dimensions: usize,
    pub supported_dimensions: Vec<usize>,
}

impl EmbeddingModelDetails {
    /// General-purpose embeddings optimized for retrieval quality.
    pub fn voyage_4_large() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-4-large".to_string(),
            api_id: "voyage-4-large".to_string(),
            max_input_tokens_per_item: 32_000,
            max_batch_total_tokens: Some(120_000),
            max_batch_size: Some(1_000),
            tokenizer_id: "voyageai/voyage-4-large".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.12,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    /// General-purpose embeddings balancing quality, latency, and cost.
    pub fn voyage_4() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-4".to_string(),
            api_id: "voyage-4".to_string(),
            max_input_tokens_per_item: 32_000,
            max_batch_total_tokens: Some(320_000),
            max_batch_size: Some(1_000),
            tokenizer_id: "voyageai/voyage-4".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.06,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    /// General-purpose embeddings optimized for latency and cost.
    pub fn voyage_4_lite() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-4-lite".to_string(),
            api_id: "voyage-4-lite".to_string(),
            max_input_tokens_per_item: 32_000,
            max_batch_total_tokens: Some(1_000_000),
            max_batch_size: Some(1_000),
            tokenizer_id: "voyageai/voyage-4-lite".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.02,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    /// Embeddings optimized for code retrieval.
    ///
    /// Voyage does not currently publish an aggregate request-token limit
    /// for this model. Per-input and item-count limits are checked locally;
    /// aggregate token limits are left to the provider.
    pub fn voyage_code_4() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-code-4".to_string(),
            api_id: "voyage-code-4".to_string(),
            max_input_tokens_per_item: 32_000,
            max_batch_total_tokens: None,
            max_batch_size: Some(1_000),
            tokenizer_id: "voyageai/voyage-code-4".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.12,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    pub fn voyage_3_5() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-3.5".to_string(),
            api_id: "voyage-3.5".to_string(),
            max_input_tokens_per_item: 32000,
            max_batch_total_tokens: Some(320_000),
            max_batch_size: Some(1000),
            tokenizer_id: "voyageai/voyage-3.5".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.06,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    pub fn voyage_3_large() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-3-large".to_string(),
            api_id: "voyage-3-large".to_string(),
            max_input_tokens_per_item: 32000,
            max_batch_total_tokens: Some(120_000),
            max_batch_size: Some(1000),
            tokenizer_id: "voyageai/voyage-3-large".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.18,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    pub fn voyage_3_5_lite() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-3.5-lite".to_string(),
            api_id: "voyage-3.5-lite".to_string(),
            max_input_tokens_per_item: 32000,
            max_batch_total_tokens: Some(1_000_000),
            max_batch_size: Some(1000),
            tokenizer_id: "voyageai/voyage-3.5-lite".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.02,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    pub fn voyage_code_3() -> Self {
        Self {
            provider: EmbeddingProviderIdentifier::Voyage,
            id: "voyage-code-3".to_string(),
            api_id: "voyage-code-3".to_string(),
            max_input_tokens_per_item: 32000,
            max_batch_total_tokens: Some(120_000),
            max_batch_size: Some(1000),
            tokenizer_id: "voyageai/voyage-code-3".to_string(),
            requires_hf_token: false,
            cost_per_million_input_tokens: 0.18,
            default_dimensions: 1024,
            supported_dimensions: vec![256, 512, 1024, 2048],
        }
    }

    pub fn get(provider: EmbeddingProviderIdentifier, model_id: &str) -> Result<Self, Error> {
        match provider {
            EmbeddingProviderIdentifier::Voyage => match model_id {
                "voyage-4-large" => Ok(Self::voyage_4_large()),
                "voyage-4" => Ok(Self::voyage_4()),
                "voyage-4-lite" => Ok(Self::voyage_4_lite()),
                "voyage-code-4" => Ok(Self::voyage_code_4()),
                "voyage-3.5" => Ok(Self::voyage_3_5()),
                "voyage-3-large" => Ok(Self::voyage_3_large()),
                "voyage-3.5-lite" => Ok(Self::voyage_3_5_lite()),
                "voyage-code-3" => Ok(Self::voyage_code_3()),
                _ => Err(Error::ModelNotSupported(format!(
                    "Voyage embedding model '{}' not supported.",
                    model_id
                ))),
            },
        }
    }
}
