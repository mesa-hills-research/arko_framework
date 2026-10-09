//! Dense (embedding) search: the [`Embedder`] interface that providers implement,
//! the [`ModelDescriptor`] that names the vectors an embedder produces, the per-index
//! [`DenseConfig`], the text that is embedded for a chunk ([`render`]), and the
//! statistics of an embedding run ([`EmbedStats`]).
//!
//! arko_index makes no network calls itself: an [`Embedder`] is supplied by the
//! caller. The crate ships [`HashEmbedder`] (offline, deterministic, not semantic: for
//! tests and examples); the `arko_index_cli` example adapts arko_ai's Voyage client.
//! An index is built with embeddings by [`index2::build`](crate::index2::build) with
//! [`BuildOptions::dense`](crate::index2::BuildOptions::dense) set, and searched by
//! [`hybrid::search`](crate::hybrid::search). A build publishes its generation for
//! lexical search before it embeds, and [`index2::embed`](crate::index2::embed) adds the
//! vectors a generation lacks, so dense coverage fills in while the index is in use.
pub(crate) mod cache;
pub(crate) mod run;

use crate::{
    bench_v1::Chunk,
    bench_v1::Error,
    fsio::atomic_write,
    vector::{Quantization, StoreConfig, VectorIndex},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

/// The version of [`render`]'s output. Embeddings are cached by the hash of the
/// rendered text, so a new version re-embeds every chunk.
pub const RENDER_VERSION: u32 = 1;

/// The text embedded for a chunk (render version 1) with the default
/// [`EmbedText::PathScopeSource`]: the path, the scope on its own line when there is
/// one, then the chunk's source with leading and trailing whitespace removed:
///
/// ```text
/// src/net/retry.rs
/// impl RetryPolicy › fn backoff
/// pub fn backoff(&self, attempt: u32) -> Duration { ... }
/// ```
///
/// Line numbers are left out on purpose: an edit above a chunk moves its lines but
/// leaves this text, and so its cached embedding, unchanged. `chunk_text` is the
/// chunk's source (`file[start_byte..end_byte]`).
pub fn render(chunk: &Chunk, chunk_text: &str) -> String {
    render_with(chunk, chunk_text, EmbedText::default())
}

/// [`render`] with the parts `embed_text` chooses: [`EmbedText::Source`] is the
/// trimmed source alone, [`EmbedText::PathSource`] the path line and the source.
pub fn render_with(chunk: &Chunk, chunk_text: &str, embed_text: EmbedText) -> String {
    let body = chunk_text.trim();
    let mut out = String::with_capacity(chunk.path.len() + chunk.scope.len() + body.len() + 2);
    if embed_text != EmbedText::Source {
        out.push_str(&chunk.path);
        out.push('\n');
    }
    if embed_text == EmbedText::PathScopeSource && !chunk.scope.is_empty() {
        out.push_str(&chunk.scope);
        out.push('\n');
    }
    out.push_str(body);
    out
}

/// What the embedded text of a chunk holds besides its source ([`render_with`]).
/// Saved with the index ([`DenseConfig::embed_text`]). Embeddings are cached by the
/// hash of the rendered text, so each choice has its own cached vectors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbedText {
    /// The chunk's source only.
    Source,
    /// The path on its own line, then the source.
    PathSource,
    /// The path, the scope on its own line when there is one, then the source.
    #[default]
    PathScopeSource,
}
impl EmbedText {
    /// The name the CLI and `dense.json` use: `source`, `path-source` or
    /// `path-scope-source`.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedText::Source => "source",
            EmbedText::PathSource => "path-source",
            EmbedText::PathScopeSource => "path-scope-source",
        }
    }
}
impl std::str::FromStr for EmbedText {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "source" => EmbedText::Source,
            "path-source" => EmbedText::PathSource,
            "path-scope-source" => EmbedText::PathScopeSource,
            _ => {
                return Err(format!(
                    "embedded text {s:?}: use source, path-source or path-scope-source"
                )
                .into());
            }
        })
    }
}

/// The cache key of a rendered text: its SHA-256.
pub(crate) fn text_hash(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

/// Names how chunks were rendered for embedding (part of a vector store's model):
/// `arko_index/render-1/path-scope-source`.
pub(crate) fn text_format(render_version: u32, embed_text: EmbedText) -> String {
    format!("arko_index/render-{render_version}/{}", embed_text.as_str())
}

/// Whether a text is embedded as a document (a chunk) or as a search query. Some
/// models (Voyage's `input_type`) embed them differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Document,
    Query,
}

/// Names the vectors an embedder produces. Two embedders with equal descriptors must
/// produce interchangeable vectors: the descriptor is part of the embedding cache key,
/// so changing any field re-embeds everything.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelDescriptor {
    /// `voyage`, `hash`, and later local providers.
    pub provider: String,
    /// The provider's model name, such as `voyage-code-4`.
    pub model: String,
    /// The length of every vector.
    pub dimensions: usize,
    /// Whether documents and queries are embedded with different input types.
    pub input_types: bool,
    /// A model revision or weights digest, for providers whose model name does not
    /// pin the weights (local models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}
impl ModelDescriptor {
    pub fn new(provider: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            dimensions,
            input_types: false,
            revision: None,
        }
    }
    pub fn with_input_types(mut self, input_types: bool) -> Self {
        self.input_types = input_types;
        self
    }
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }
    /// A short stable key: the first 16 hex digits of the SHA-256 of the descriptor's
    /// JSON. Names the descriptor's directory in the embedding cache.
    pub fn key(&self) -> String {
        let json = serde_json::to_vec(self).expect("descriptor serializes");
        let digest = Sha256::digest(&json);
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }
    /// `provider/model` and the dimensions, for messages.
    pub fn display(&self) -> String {
        format!(
            "{}/{} ({} dimensions)",
            self.provider, self.model, self.dimensions
        )
    }
}

/// An embedder's request limits, which batching respects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbedLimits {
    /// Texts per request.
    pub max_items: usize,
    /// Total tokens per request (by [`Embedder::count_tokens`]).
    pub max_request_tokens: u64,
    /// Tokens per text. A longer text is not embedded (counted in
    /// [`EmbedStats::too_long`]); its chunk stays searchable lexically.
    pub max_input_tokens: u64,
}
impl EmbedLimits {
    pub fn new(max_items: usize, max_request_tokens: u64, max_input_tokens: u64) -> Self {
        Self {
            max_items: max_items.max(1),
            max_request_tokens,
            max_input_tokens,
        }
    }
}

/// The result of one embedding request.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    /// One vector per input text, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Tokens the provider billed for the request (0 when it does not say; the
    /// counted tokens are then used for accounting).
    pub tokens: u64,
}

/// A text embedding model. Implementations make the requests; arko_index batches
/// texts within [`limits`](Self::limits), stops at the caller's token cap, caches
/// the vectors and stores them. Calls come from one thread at a time.
///
/// `embed` should not retry by itself in a way that bills twice without saying so:
/// the tokens it returns are what the run accounts.
pub trait Embedder: Send + Sync {
    /// Names the vectors this embedder produces.
    fn descriptor(&self) -> ModelDescriptor;
    /// Request limits.
    fn limits(&self) -> EmbedLimits;
    /// The token count of each text as the provider will bill it, or an upper bound.
    /// Used to pack requests and to stop at the token cap before a request is sent.
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error>;
    /// Embeds `texts` (at most `limits().max_items`, within the token limits).
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error>;
}

/// Embeds a search query: its vector and the tokens billed.
pub fn embed_query(embedder: &dyn Embedder, query: &str) -> Result<(Vec<f32>, u64), Error> {
    let mut e = embedder.embed(&[query], InputKind::Query)?;
    if e.vectors.len() != 1 {
        return Err(format!(
            "the embedder returned {} vectors for 1 query",
            e.vectors.len()
        )
        .into());
    }
    let tokens = if e.tokens > 0 {
        e.tokens
    } else {
        embedder
            .count_tokens(&[query])?
            .first()
            .copied()
            .unwrap_or(0)
    };
    Ok((e.vectors.remove(0), tokens))
}

/// An offline, deterministic embedder for tests and examples: feature hashing of the
/// lexical search terms (words and identifier parts), with a sign per term. Texts
/// that share terms get similar vectors; it knows nothing of meaning. Descriptor:
/// provider `hash`, model `hash-v1`. One token per term, at least one per text.
#[derive(Clone, Debug)]
pub struct HashEmbedder {
    dims: usize,
}
impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dims: dimensions.max(2),
        }
    }
    fn vector(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dims];
        for term in crate::search::tokens(text) {
            let h = Sha256::digest(term.as_bytes());
            let i = u64::from_le_bytes(h[..8].try_into().unwrap()) % self.dims as u64;
            v[i as usize] += if h[8] & 1 == 0 { 1.0 } else { -1.0 };
        }
        if v.iter().all(|x| *x == 0.0) {
            v[0] = 1.0;
        }
        v
    }
}
impl Embedder for HashEmbedder {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor::new("hash", "hash-v1", self.dims)
    }
    fn limits(&self) -> EmbedLimits {
        EmbedLimits::new(256, 1 << 40, 1 << 40)
    }
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        Ok(texts
            .iter()
            .map(|t| (crate::search::tokens(t).len() as u64).max(1))
            .collect())
    }
    fn embed(&self, texts: &[&str], _kind: InputKind) -> Result<Embedding, Error> {
        let tokens = self.count_tokens(texts)?.iter().sum();
        Ok(Embedding {
            vectors: texts.iter().map(|t| self.vector(t)).collect(),
            tokens,
        })
    }
}

/// The embedding settings saved with an index (`STATE_DIR/dense.json`): which model
/// embeds it and how. Written by a build with embeddings, so later builds (after a
/// watcher reported changes, for example) can use the same model without being told
/// again; see [`DenseConfig::load`]. Removing it makes later builds lexical only, in
/// callers that follow it (the CLI does).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DenseConfig {
    /// The file format version (1).
    #[serde(default = "one")]
    pub version: u32,
    /// The model: provider, model name, dimensions and input-type use.
    pub embedder: ModelDescriptor,
    /// The [`render`] version the last build embedded with.
    #[serde(default = "render_version")]
    pub render_version: u32,
    #[serde(default)]
    pub quantization: Quantization,
    /// How dense search finds candidates (default: a flat scan).
    #[serde(default)]
    pub vector_index: VectorIndex,
    /// The token cap of each build (`None`: no cap).
    #[serde(default)]
    pub max_tokens_per_run: Option<u64>,
    /// What each chunk's embedded text holds besides its source (default: the path
    /// and the scope).
    #[serde(default)]
    pub embed_text: EmbedText,
}
fn one() -> u32 {
    1
}
fn render_version() -> u32 {
    RENDER_VERSION
}
impl DenseConfig {
    /// The file in the state directory.
    pub const FILE: &'static str = "dense.json";

    pub fn new(embedder: ModelDescriptor) -> Self {
        Self {
            version: 1,
            embedder,
            render_version: RENDER_VERSION,
            quantization: Quantization::default(),
            vector_index: VectorIndex::default(),
            max_tokens_per_run: None,
            embed_text: EmbedText::default(),
        }
    }
    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }
    pub fn with_vector_index(mut self, vector_index: VectorIndex) -> Self {
        self.vector_index = vector_index;
        self
    }
    pub fn with_max_tokens_per_run(mut self, cap: Option<u64>) -> Self {
        self.max_tokens_per_run = cap;
        self
    }
    pub fn with_embed_text(mut self, embed_text: EmbedText) -> Self {
        self.embed_text = embed_text;
        self
    }
    /// The saved config, or `None` when the index has none (lexical only).
    pub fn load(state_dir: &Path) -> Result<Option<Self>, Error> {
        match std::fs::read(state_dir.join(Self::FILE)) {
            Ok(bytes) => Self::parse(&bytes, &state_dir.join(Self::FILE)).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    /// Reads a config file anywhere (the format of `dense.json`). `render_version`
    /// is informational: a build always renders with [`RENDER_VERSION`] and records it.
    pub fn read(path: &Path) -> Result<Self, Error> {
        Self::parse(&std::fs::read(path)?, path)
    }
    fn parse(bytes: &[u8], path: &Path) -> Result<Self, Error> {
        let config: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        if config.version != 1 {
            return Err(format!(
                "{}: unsupported dense config version {}",
                path.display(),
                config.version
            )
            .into());
        }
        Ok(config)
    }
    /// Replaces the saved config atomically.
    pub fn save(&self, state_dir: &Path) -> Result<(), Error> {
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        atomic_write(state_dir, Self::FILE, &bytes)
    }
    /// Removes the saved config; returns whether there was one. The embedding cache
    /// stays (`STATE_DIR/embed-cache/`).
    pub fn remove(state_dir: &Path) -> Result<bool, Error> {
        match std::fs::remove_file(state_dir.join(Self::FILE)) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

/// Embedding settings of one build ([`BuildOptions::dense`](crate::index2::BuildOptions::dense))
/// or one embedding run ([`index2::embed`](crate::index2::embed), which takes the
/// quantization, vector index and embedded text from the generation it embeds).
#[non_exhaustive]
pub struct DenseBuild<'a> {
    pub embedder: &'a dyn Embedder,
    pub quantization: Quantization,
    pub vector_index: VectorIndex,
    /// Stop embedding before a request would bring this run's tokens over the cap.
    /// `None`: no cap.
    pub max_tokens: Option<u64>,
    /// What each chunk's embedded text holds besides its source.
    pub embed_text: EmbedText,
    /// Called after each embedding request.
    pub progress: Option<&'a dyn Fn(&EmbedProgress)>,
    /// Return from [`index2::build`](crate::index2::build) once the generation is
    /// published, leaving the embedding to [`index2::embed`](crate::index2::embed).
    /// `false` (the default): the build embeds after publishing, before it returns.
    pub embed_later: bool,
    /// When it is stopped, the run stops after its current request and records the
    /// reason. The vectors paid for are kept, and a later run continues.
    pub stop: Option<&'a StopSignal>,
}
impl<'a> DenseBuild<'a> {
    pub fn new(embedder: &'a dyn Embedder) -> Self {
        Self {
            embedder,
            quantization: Quantization::default(),
            vector_index: VectorIndex::default(),
            max_tokens: None,
            embed_text: EmbedText::default(),
            progress: None,
            embed_later: false,
            stop: None,
        }
    }
    /// The settings of a saved config (quantization, vector index, cap and embedded
    /// text) for this embedder.
    pub fn from_config(embedder: &'a dyn Embedder, config: &DenseConfig) -> Self {
        Self {
            max_tokens: config.max_tokens_per_run,
            ..Self::new(embedder)
                .with_quantization(config.quantization)
                .with_vector_index(config.vector_index)
                .with_embed_text(config.embed_text)
        }
    }
    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }
    pub fn with_vector_index(mut self, vector_index: VectorIndex) -> Self {
        self.vector_index = vector_index;
        self
    }
    pub fn with_max_tokens(mut self, cap: Option<u64>) -> Self {
        self.max_tokens = cap;
        self
    }
    pub fn with_embed_text(mut self, embed_text: EmbedText) -> Self {
        self.embed_text = embed_text;
        self
    }
    pub fn with_progress(mut self, progress: &'a dyn Fn(&EmbedProgress)) -> Self {
        self.progress = Some(progress);
        self
    }
    pub fn with_embed_later(mut self, embed_later: bool) -> Self {
        self.embed_later = embed_later;
        self
    }
    pub fn with_stop(mut self, stop: &'a StopSignal) -> Self {
        self.stop = Some(stop);
        self
    }
}

/// Asks an embedding run to stop after its current request ([`DenseBuild::stop`]), with
/// a reason the run records for [`index2::status`](crate::index2::status). Shared
/// between threads: a signal handler, or a thread watching the program that started
/// the run.
#[derive(Debug, Default)]
pub struct StopSignal {
    stopped: AtomicBool,
    reason: Mutex<Option<String>>,
}
impl StopSignal {
    pub fn new() -> Self {
        Self::default()
    }
    /// Asks for the stop. Returns `false` when it was already asked for, whose reason is
    /// kept.
    pub fn stop(&self, reason: impl Into<String>) -> bool {
        let mut kept = self.reason.lock().unwrap_or_else(|e| e.into_inner());
        if self.stopped.swap(true, Ordering::SeqCst) {
            return false;
        }
        *kept = Some(reason.into());
        true
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }
    /// Why the stop was asked for.
    pub fn reason(&self) -> Option<String> {
        self.reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Progress of an embedding run, after each request.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbedProgress {
    /// Texts embedded so far by this run.
    pub texts: u64,
    /// Texts this run has to embed (not in the cache).
    pub total: u64,
    /// Tokens billed so far by this run.
    pub tokens: u64,
    pub requests: u64,
    /// Chunks of the generation with a vector in the store now.
    #[serde(default)]
    pub embedded_docs: u64,
    /// Chunks in the generation.
    #[serde(default)]
    pub docs: u64,
}

/// What an index build's embedding did ([`BuildStats::dense`](crate::index2::BuildStats::dense)),
/// or one embedding run ([`index2::embed`](crate::index2::embed)). A build with
/// [`DenseBuild::embed_later`] reports the generation it published: its chunks, those
/// with a vector already, and its texts, of which `texts - cached` are left to embed.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct EmbedStats {
    pub embedder: ModelDescriptor,
    pub render_version: u32,
    pub embed_text: EmbedText,
    /// Chunks in the index.
    pub docs: u64,
    /// Chunks with a vector (the others are found by lexical search only).
    pub embedded_docs: u64,
    /// Distinct rendered texts (identical chunks share one embedding).
    pub texts: u64,
    /// Distinct texts found in the cache, so not embedded again.
    pub cached: u64,
    /// Texts embedded by this run.
    pub new: u64,
    /// Embedding requests sent.
    pub requests: u64,
    /// Tokens billed by this run (the provider's figure, else the counted tokens).
    pub tokens: u64,
    pub max_tokens: Option<u64>,
    /// Whether embedding stopped at the cap.
    pub cap_reached: bool,
    /// Texts left without a vector: stopped by the cap, or too long.
    pub not_embedded: u64,
    /// The counted tokens of those texts: what embedding them would take.
    pub not_embedded_tokens: u64,
    /// Texts over the embedder's per-input limit.
    pub too_long: u64,
    /// Vectors in the model's cache after the build (after compaction, if any).
    pub cache_records: u64,
    /// Vectors this build added to the vector store, and removed from it: the chunks
    /// whose embedded text or path changed (all of them when the store is new).
    pub vectors_added: u64,
    pub vectors_removed: u64,
    /// Whether this build created the vector store: the first build with this model
    /// and these settings, or the store was unusable (damaged, or another format).
    pub store_created: bool,
    /// The run stopped because a newer generation was published: the run for that
    /// generation takes over, with what this one embedded in the cache.
    pub superseded: bool,
    /// The run stopped when [`DenseBuild::stop`] was stopped. A later run continues.
    pub stopped: bool,
}
impl EmbedStats {
    pub(crate) fn new(
        embedder: ModelDescriptor,
        embed_text: EmbedText,
        max_tokens: Option<u64>,
    ) -> Self {
        Self {
            embedder,
            render_version: RENDER_VERSION,
            embed_text,
            docs: 0,
            embedded_docs: 0,
            texts: 0,
            cached: 0,
            new: 0,
            requests: 0,
            tokens: 0,
            max_tokens,
            cap_reached: false,
            not_embedded: 0,
            not_embedded_tokens: 0,
            too_long: 0,
            cache_records: 0,
            vectors_added: 0,
            vectors_removed: 0,
            store_created: false,
            superseded: false,
            stopped: false,
        }
    }
}

/// How far the embedding of a generation has come ([`DenseIndexInfo::state`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbedState {
    /// Vectors are still being added: a run is embedding the generation, or one
    /// stopped before it finished and [`index2::embed`](crate::index2::embed) continues.
    Embedding,
    /// Every chunk that can have a vector has one. Texts over the model's input limit
    /// have none.
    #[default]
    Complete,
    /// A run stopped at its token cap. A run with a higher cap, or none, continues.
    CapReached,
}
impl EmbedState {
    /// `embedding`, `complete` or `cap-reached`.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedState::Embedding => "embedding",
            EmbedState::Complete => "complete",
            EmbedState::CapReached => "cap-reached",
        }
    }
}

/// What a generation's embeddings are (`dense.json` in the generation directory).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DenseIndexInfo {
    /// The format version: 2. Generations of version 1 (earlier arko_index versions)
    /// kept their own vectors, and are searched lexically only.
    pub version: u32,
    pub embedder: ModelDescriptor,
    pub render_version: u32,
    #[serde(default)]
    pub embed_text: EmbedText,
    pub quantization: Quantization,
    #[serde(default)]
    pub vector_index: VectorIndex,
    /// The vector store's directory in the state directory (`vectors-...`).
    #[serde(default)]
    pub store: String,
    /// Chunks in the generation.
    pub docs: u64,
    /// Chunks with a vector in the store, at least: it grows while the generation is
    /// embedded.
    pub embedded: u64,
    /// How far the embedding has come. Generations whose `dense.json` lacks it were
    /// embedded before they were published: `Complete`.
    #[serde(default)]
    pub state: EmbedState,
}
impl DenseIndexInfo {
    /// The format version this crate writes and searches.
    pub const VERSION: u32 = 2;

    /// Whether every chunk has a vector.
    pub fn complete(&self) -> bool {
        self.embedded == self.docs
    }

    /// The model and settings of the generation's vector store.
    pub fn store_config(&self) -> StoreConfig {
        StoreConfig::new(
            self.embedder.clone(),
            text_format(self.render_version, self.embed_text),
        )
        .with_quantization(self.quantization)
        .with_index(self.vector_index)
    }
}
