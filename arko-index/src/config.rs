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
    /// Attempts of one provider request that may fail with transient errors
    /// (rate limits, timeouts, server errors, truncated streams, malformed
    /// tool calls) before the request fails with the last error. Explicit
    /// capacity signals (overloaded / unavailable) keep waiting without
    /// consuming attempts; pause and cancel interrupt any wait.
    pub max_transient_attempts: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            batch_count: 6,
            concurrency: 2,
            embedding_concurrency: 2,
            embedding: EmbeddingProfile::default(),
            vector_cache_bytes: 64 * 1024 * 1024,
            max_transient_attempts: 10,
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
        if self.max_transient_attempts == 0 {
            return Err(invalid("max_transient_attempts must be positive"));
        }
        if self.embedding.tokenizer_revision.is_empty() {
            return Err(invalid("pin a tokenizer revision"));
        }
        Ok(())
    }
}
/// Directory names a scan never enters by default, at any depth, whether or
/// not an ignore file mentions them. See [`BuildSpec::skip_directories`].
pub const DEFAULT_SKIPPED_DIRECTORIES: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".runtime",
    ".cache",
    "__pycache__",
    ".venv",
    "dist",
];
fn default_skipped_directories() -> Vec<String> {
    DEFAULT_SKIPPED_DIRECTORIES
        .iter()
        .map(|name| (*name).to_owned())
        .collect()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildSpec {
    pub name: String,
    pub root: PathBuf,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Directory names (exact, not patterns) never entered, at any depth.
    /// Defaults to [`DEFAULT_SKIPPED_DIRECTORIES`], also for specs saved
    /// before this field existed; an empty list skips none (keep `.git` in a
    /// custom list). Ignore files and `exclude` patterns apply as well.
    #[serde(default = "default_skipped_directories")]
    pub skip_directories: Vec<String>,
}
impl BuildSpec {
    pub fn new(name: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            root: root.into(),
            include: vec![],
            exclude: vec![],
            skip_directories: default_skipped_directories(),
        }
    }
    /// Replaces [`BuildSpec::skip_directories`].
    pub fn with_skip_directories(
        mut self,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.skip_directories = names.into_iter().map(Into::into).collect();
        self
    }
}
