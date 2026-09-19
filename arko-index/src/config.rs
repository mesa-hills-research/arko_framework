use crate::{Result, error::invalid};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub batch_count: usize,
    pub concurrency: usize,
    pub embedding_concurrency: usize,
    pub embedding: EmbeddingProfile,
    pub vector_cache_bytes: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            batch_count: 6,
            concurrency: 2,
            embedding_concurrency: 2,
            embedding: EmbeddingProfile::default(),
            vector_cache_bytes: 64 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EmbeddingProfile {
    pub model: String,
    pub dimensions: usize,
    pub tokenizer_revision: String,
}
impl Default for EmbeddingProfile {
    fn default() -> Self {
        Self {
            model: "voyage-4".into(),
            dimensions: 1024,
            tokenizer_revision: "44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9".into(),
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.batch_count == 0 || self.concurrency == 0 || self.embedding_concurrency == 0 {
            return Err(invalid(
                "batch count and request concurrency must be positive",
            ));
        }
        if self.embedding.tokenizer_revision.is_empty() {
            return Err(invalid("pin a tokenizer revision"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildSpec {
    pub name: String,
    pub root: PathBuf,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}
impl BuildSpec {
    pub fn new(name: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            root: root.into(),
            include: vec![],
            exclude: vec![],
        }
    }
}
