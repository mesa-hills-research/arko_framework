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
    /// capacity signals (overloaded / unavailable) wait without consuming
    /// attempts, up to `max_capacity_wait_ms`. A plan usage limit is not
    /// retried: the job pauses. Pause and cancel interrupt any wait.
    pub max_transient_attempts: usize,
    /// The longest one request may spend waiting out explicit capacity
    /// signals (overloaded / unavailable), in milliseconds, before it fails
    /// with the last error: 15 minutes by default. `None` waits as long as
    /// the provider reports it is overloaded.
    pub max_capacity_wait_ms: Option<u64>,
    /// Opt-in: dispatch search query embeddings before queued build
    /// embeddings. Off by default: a query waits its turn behind a running
    /// update's embedding requests.
    pub query_priority: bool,
    /// Opt-in: the transient failures a search query embedding may have
    /// before the search fails, instead of `max_transient_attempts`.
    pub query_max_transient_attempts: Option<usize>,
    /// Opt-in retention: keep the newest `n` complete generations of each
    /// index and prune older ones after each publish (their chunks, file
    /// states and request artifacts; the job, its events and usage stay,
    /// with the phase `pruned`), then remove data nothing references. `None`
    /// (the default) keeps every generation until the index is deleted.
    pub keep_generations: Option<usize>,
    /// How SQLite syncs commits to disk. [`Synchronous::Full`] by default.
    pub synchronous: Synchronous,
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
            max_capacity_wait_ms: Some(DEFAULT_MAX_CAPACITY_WAIT_MS),
            query_priority: false,
            query_max_transient_attempts: None,
            keep_generations: None,
            synchronous: Synchronous::Full,
        }
    }
}
/// The default [`Config::max_capacity_wait_ms`]: 15 minutes.
pub const DEFAULT_MAX_CAPACITY_WAIT_MS: u64 = 15 * 60 * 1000;
/// SQLite's `synchronous` setting for the index database
/// ([`Config::synchronous`]).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Synchronous {
    /// Every commit is synced to disk before it returns: nothing committed
    /// is lost on power loss.
    #[default]
    Full,
    /// Opt-in: commits are synced at WAL checkpoints, so fewer disk syncs
    /// run while jobs record their progress. Power loss or an operating
    /// system crash can lose the last commits (never corrupting the
    /// database); jobs resume from their checkpoints.
    Normal,
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
        if self.query_max_transient_attempts == Some(0) {
            return Err(invalid(
                "query_max_transient_attempts must be positive when set",
            ));
        }
        if self.keep_generations == Some(0) {
            return Err(invalid("keep_generations must be positive when set"));
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
/// Files larger than this many bytes are never read by default; see
/// [`BuildSpec::max_file_bytes`].
pub const DEFAULT_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
fn default_max_file_bytes() -> u64 {
    DEFAULT_MAX_FILE_BYTES
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
    /// Files larger than this are excluded without being read, with the
    /// reason "too large". [`DEFAULT_MAX_FILE_BYTES`] (16 MiB) by default,
    /// also for specs saved before this field existed.
    #[serde(default = "default_max_file_bytes")]
    pub max_file_bytes: u64,
    /// How chunk-model requests are sized, and what happens to files that
    /// exceed that size.
    #[serde(default)]
    pub chunking: ChunkingOptions,
}
impl BuildSpec {
    pub fn new(name: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            root: root.into(),
            include: vec![],
            exclude: vec![],
            skip_directories: default_skipped_directories(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            chunking: ChunkingOptions::default(),
        }
    }
    /// Replaces [`BuildSpec::chunking`].
    pub fn with_chunking(mut self, chunking: ChunkingOptions) -> Self {
        self.chunking = chunking;
        self
    }
    /// Replaces [`BuildSpec::max_file_bytes`].
    pub fn with_max_file_bytes(mut self, bytes: u64) -> Self {
        self.max_file_bytes = bytes;
        self
    }
    pub fn validate(&self) -> Result<()> {
        if self.max_file_bytes == 0 {
            return Err(invalid("max_file_bytes must be positive"));
        }
        self.chunking.validate()
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

/// How chunk-model requests are sized ([`BuildSpec::chunking`]).
///
/// A request's size is its rendered file content: each file's header line
/// and its numbered rows, as the chunk model sees them. The fixed
/// instructions and tool schema, under 1,000 tokens, come on top. Chunk
/// models of the OpenAI provider are measured in `o200k_base` tokens, the
/// encoding GPT-5.x and GPT-6 use; other models are measured in bytes.
///
/// By default a request may take 10% of the chunk model's input limit on the
/// client's auth path ([`arko_ai::LLMClient::max_input_tokens`]): 92,200
/// tokens for GPT-5.6 and GPT-6 with an API key, 27,200 over ChatGPT OAuth,
/// and 4 bytes per token of that share for models measured in bytes (about
/// 400 KB for a 1M-token Claude model). A model whose limit is unknown is
/// assumed to accept [`FALLBACK_MAX_INPUT_TOKENS`](crate::budget::FALLBACK_MAX_INPUT_TOKENS).
/// Initial requests hold up to [`Config::batch_count`] files within the same
/// budget. A job resolves its budget when it scans the source and keeps it
/// through resumes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ChunkingOptions {
    /// The most `o200k_base` tokens one request may hold, for chunk models
    /// measured in tokens; `None` uses the default share. For models
    /// measured in bytes it applies as 4 bytes per token unless
    /// `max_request_bytes` is set.
    pub max_request_tokens: Option<usize>,
    /// The most bytes one request may hold. `None` uses the default for
    /// models measured in bytes, and no byte limit for models measured in
    /// tokens. When both limits are set, both apply.
    pub max_request_bytes: Option<usize>,
    /// What happens to a file that doesn't fit in one request.
    pub oversize: OversizePolicy,
    /// Repair requests per window before `repair_limit` applies: 5 by
    /// default. `None` repairs until every cut is valid.
    pub max_repair_rounds: Option<usize>,
    /// What happens to a window whose cuts are still invalid after
    /// `max_repair_rounds` repairs.
    pub repair_limit: RepairLimitPolicy,
}
impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            max_request_tokens: None,
            max_request_bytes: None,
            oversize: OversizePolicy::Split,
            max_repair_rounds: Some(5),
            repair_limit: RepairLimitPolicy::Merge,
        }
    }
}
/// What happens to a window whose cuts are still invalid when its repair
/// requests run out ([`ChunkingOptions::repair_limit`]).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RepairLimitPolicy {
    /// Drop the invalid cuts, merging each disputed region into the
    /// neighboring chunk, which is always valid; accepted cuts stay. A
    /// `deterministic_repair` event records it.
    #[default]
    Merge,
    /// Leave the window unfinished, so the job is blocked naming the file;
    /// resuming the job allows as many repairs again.
    Block,
}
impl ChunkingOptions {
    pub fn validate(&self) -> Result<()> {
        if self.max_request_tokens == Some(0) || self.max_request_bytes == Some(0) {
            return Err(invalid("chunk request limits must be positive"));
        }
        if self.max_repair_rounds == Some(0) {
            return Err(invalid("max_repair_rounds must be positive when set"));
        }
        Ok(())
    }
}
/// What happens to a file that doesn't fit in one chunk request
/// ([`ChunkingOptions::oversize`]).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OversizePolicy {
    /// Split the file into windows that fit, at the strongest boundaries
    /// available: Markdown headings (level 1 first), then top-level blocks;
    /// in Rust, items, then members of impls, traits and modules, then
    /// statements; in other files, blank lines and lines at the outermost
    /// bracket depth and indentation, then deeper ones; then single lines.
    /// Pieces are of similar size, and boundaries that cut through a guarded
    /// unit are used only when no other fits. A window edge is also a chunk
    /// edge. A single line that exceeds the budget becomes one chunk without
    /// a model request (an `oversize` event lists it).
    #[default]
    Split,
    /// Leave the file out of the index, as an exclusion with the reason "too
    /// large (over the chunk request budget)".
    Skip,
    /// Block the job before any model request, naming the files.
    Block,
}
