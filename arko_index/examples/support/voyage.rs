//! A Voyage AI [`Embedder`] for arko_index, on arko_ai's tested Voyage client (HTTP,
//! error classification, local token counting with the model's tokenizer). It does
//! not re-implement any of that: it maps arko_index's synchronous interface onto
//! `arko_ai::EmbeddingClient`, retries what arko_ai classifies as transient, and
//! reports the tokens Voyage billed.
//!
//! The file depends only on arko_ai, arko_index and tokio, so the arko integration
//! (the code-search tool) can move it, unchanged, into the crate that hosts the tool.
use arko_ai::{
    BackoffPolicy, EmbeddingClient, EmbeddingConfig, EmbeddingModelDetails,
    EmbeddingProviderIdentifier, RetryClass, TokenCounter, VoyageInputType,
};
use arko_index::{
    bench_v1::Error,
    dense::{EmbedLimits, Embedder, Embedding, InputKind, ModelDescriptor},
};
use std::time::Duration;

/// Voyage's total tokens per request, where arko_ai's model table has none:
/// voyage-code-4 answered "The max allowed tokens per submitted batch is 320000"
/// (2026-10-04, docs/voyage-api-notes.md). Other models without a figure get the
/// smallest documented limit.
fn request_token_limit(details: &EmbeddingModelDetails) -> u64 {
    match (details.max_batch_total_tokens, details.id.as_str()) {
        (Some(n), _) => n as u64,
        (None, "voyage-code-4") => 320_000,
        (None, _) => 120_000,
    }
}

/// Embeds with a Voyage model. Documents use `input_type: document` and queries
/// `input_type: query` unless input types are off; the output dimension is always
/// sent; `truncation` is false (arko_ai's default), so no text is cut silently.
pub struct VoyageEmbedder {
    client: EmbeddingClient,
    details: EmbeddingModelDetails,
    dimensions: usize,
    input_types: bool,
    counter: TokenCounter,
    runtime: tokio::runtime::Runtime,
    max_retries: u32,
    backoff: BackoffPolicy,
}

impl VoyageEmbedder {
    /// With the key in `VOYAGE_API_KEY` (read by arko_ai's `EmbeddingClient::from_env`).
    pub fn from_env(
        model: &str,
        dimensions: Option<usize>,
        input_types: bool,
    ) -> Result<Self, Error> {
        if std::env::var("VOYAGE_API_KEY").map_or(true, |k| k.trim().is_empty()) {
            return Err("set VOYAGE_API_KEY to embed with Voyage".into());
        }
        Self::new(EmbeddingClient::from_env(), model, dimensions, input_types)
    }

    /// With a configured client. Loads the model's tokenizer (from Hugging Face on
    /// first use; arko_ai caches it).
    pub fn new(
        client: EmbeddingClient,
        model: &str,
        dimensions: Option<usize>,
        input_types: bool,
    ) -> Result<Self, Error> {
        let details = EmbeddingModelDetails::get(EmbeddingProviderIdentifier::Voyage, model)?;
        let dimensions = dimensions.unwrap_or(details.default_dimensions);
        if !details.supported_dimensions.contains(&dimensions) {
            return Err(format!(
                "{model} supports {:?} dimensions, not {dimensions}",
                details.supported_dimensions
            )
            .into());
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, model);
        let counter = runtime.block_on(client.token_counter(&config))?;
        Ok(Self {
            client,
            details,
            dimensions,
            input_types,
            counter,
            runtime,
            max_retries: 4,
            backoff: BackoffPolicy::new(Duration::from_secs(2), Duration::from_secs(60)),
        })
    }

    /// Retries of a request arko_ai classifies as transient (rate limit, server
    /// error, lost connection) or capacity (502-504); default 4. Voyage does not
    /// document whether failed requests are billed; a retried request whose first
    /// attempt Voyage processed but whose answer was lost is billed twice and
    /// accounted once.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    fn config(&self, kind: InputKind) -> EmbeddingConfig {
        let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, &self.details.id)
            .with_output_dimensionality(self.dimensions);
        if !self.input_types {
            return config;
        }
        config.with_voyage_input_type(match kind {
            InputKind::Document => VoyageInputType::Document,
            InputKind::Query => VoyageInputType::Query,
        })
    }
}

impl Embedder for VoyageEmbedder {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor::new("voyage", self.details.id.clone(), self.dimensions)
            .with_input_types(self.input_types)
    }

    fn limits(&self) -> EmbedLimits {
        EmbedLimits::new(
            self.details.max_batch_size.unwrap_or(1000),
            request_token_limit(&self.details),
            self.details.max_input_tokens_per_item as u64,
        )
    }

    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        texts
            .iter()
            .map(|t| Ok(self.counter.count(t)? as u64))
            .collect()
    }

    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        let config = self.config(kind);
        let texts: Vec<String> = texts.iter().map(|t| (*t).to_owned()).collect();
        let mut attempt = 0;
        loop {
            match self
                .runtime
                .block_on(self.client.embed(texts.clone(), &config))
            {
                Ok(response) => {
                    return Ok(Embedding {
                        vectors: response.embeddings,
                        tokens: response.usage.input_tokens as u64,
                    });
                }
                Err(e)
                    if attempt < self.max_retries
                        && matches!(
                            e.retry_class(),
                            RetryClass::Transient | RetryClass::Capacity
                        ) =>
                {
                    attempt += 1;
                    std::thread::sleep(self.backoff.next_delay(attempt, e.retry_after()));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}
