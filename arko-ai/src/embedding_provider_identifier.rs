use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifies the provider used for text embeddings.
///
/// Voyage AI is the sole supported embeddings provider. Google Gemini
/// embeddings were removed when the Google provider migrated to the
/// Interactions API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingProviderIdentifier {
    Voyage,
}

impl fmt::Display for EmbeddingProviderIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingProviderIdentifier::Voyage => write!(f, "Voyage AI"),
        }
    }
}