//! Local embedding models (feature `local-embed`): [`LocalEmbedder`] runs a small
//! code-embedding model on the CPU, in pure Rust, behind the same [`Embedder`]
//! interface as a provider, so dense and hybrid search work with no API and no
//! per-token cost.
//!
//! [`MODELS`] lists the models it runs, each pinned to a Hugging Face commit with the
//! size and SHA-256 of every file. The library reads the files from a directory
//! ([`LocalEmbedder::load`]) and checks them. Fetching them is the caller's: the
//! examples download into the Hugging Face cache (`examples/support/local.rs`).
//!
//! A batch of texts is tokenized with the model's own tokenizer (truncated at
//! [`LocalModel::max_tokens`]), run through the encoder packed together without
//! padding, pooled ([`Pooling`]) and normalized to unit length. Work is spread over a
//! dedicated thread pool ([`LocalEmbedder::with_threads`]; by default
//! [`threads`](crate::threads)).
mod model;
mod safetensors;

pub use model::{Architecture, Pooling};

use super::{EmbedLimits, Embedder, Embedding, InputKind, ModelDescriptor};
use crate::bench_v1::Error;
use model::{Config, Encoder};
use safetensors::Tensors;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokenizers::{Tokenizer, TruncationParams};

/// One file of a model: its name in the repository, its size and SHA-256 (hex).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// A model the local embedder runs. Every model has `config.json`,
/// `tokenizer.json` and `model.safetensors`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LocalModel {
    /// The short name: the repository's model name, such as `CodeRankEmbed`.
    pub name: &'static str,
    /// The Hugging Face repository, such as `nomic-ai/CodeRankEmbed`.
    pub repo: &'static str,
    /// The pinned commit of the repository.
    pub revision: &'static str,
    pub dimensions: usize,
    /// Tokens per text, special tokens included. Longer texts are truncated.
    pub max_tokens: usize,
    /// Prepended to a search query.
    pub query_prefix: &'static str,
    /// Prepended to a document.
    pub document_prefix: &'static str,
    /// The licence of the weights (SPDX).
    pub licence: &'static str,
    pub architecture: Architecture,
    pub pooling: Pooling,
    pub files: &'static [ModelFile],
}

/// `nomic-ai/CodeRankEmbed`: 137M parameters, 768 dimensions, MIT. A code retriever
/// trained from Arctic-Embed-M-Long on 21M code pairs. Queries take the prefix
/// `Represent this query for searching relevant code: `.
pub const CODE_RANK_EMBED: LocalModel = LocalModel {
    name: "CodeRankEmbed",
    repo: "nomic-ai/CodeRankEmbed",
    revision: "3c4b60807d71f79b43f3c4363786d9493691f8b1",
    dimensions: 768,
    max_tokens: 8192,
    query_prefix: "Represent this query for searching relevant code: ",
    document_prefix: "",
    licence: "MIT",
    architecture: Architecture::NomicBert,
    pooling: Pooling::Cls,
    files: &[
        ModelFile {
            name: "config.json",
            size: 1525,
            sha256: "5ff856a41d0f53ef2d74520627d464bd75c2efd8f26f381bd528654895c29b6c",
        },
        ModelFile {
            name: "tokenizer.json",
            size: 711_649,
            sha256: "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
        },
        ModelFile {
            name: "model.safetensors",
            size: 546_938_168,
            sha256: "827529bcd58aef0d9082e66eeff7e7d53a02f62bd005f841a26b3d3e2fb17ebe",
        },
    ],
};

/// `jinaai/jina-embeddings-v2-base-code`: 161M parameters, 768 dimensions,
/// Apache-2.0. A code-and-English BERT with ALiBi positions (8,192 tokens) and a
/// tokenizer trained on code. No query or document prefixes.
pub const JINA_CODE_V2: LocalModel = LocalModel {
    name: "jina-embeddings-v2-base-code",
    repo: "jinaai/jina-embeddings-v2-base-code",
    revision: "516f4baf13dec4ddddda8631e019b5737c8bc250",
    dimensions: 768,
    max_tokens: 8192,
    query_prefix: "",
    document_prefix: "",
    licence: "Apache-2.0",
    architecture: Architecture::JinaBert,
    pooling: Pooling::Mean,
    files: &[
        ModelFile {
            name: "config.json",
            size: 1216,
            sha256: "e426aa684c7f9a95c5f020aa855faf93a24f065f5fad0c9e17b124670cabdea6",
        },
        ModelFile {
            name: "tokenizer.json",
            size: 2_561_316,
            sha256: "b01c78a902aa4facb2f47f95449f48e2f7bbfea5d2472ee2f6ce92323c6f86e5",
        },
        ModelFile {
            name: "model.safetensors",
            size: 321_767_312,
            sha256: "8b53bfd4ae2cd586004a6ca4a16551b630a2a1b1d655ff1ee9be1286a1781c5b",
        },
    ],
};

/// The models the local embedder runs; the first is [`DEFAULT_MODEL`].
pub const MODELS: &[LocalModel] = &[CODE_RANK_EMBED, JINA_CODE_V2];

/// The model `--embed local` picks when none is named.
pub const DEFAULT_MODEL: &LocalModel = &MODELS[0];

/// The model with this short name or repository (case-insensitive).
pub fn find(name: &str) -> Option<&'static LocalModel> {
    MODELS
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(name) || m.repo.eq_ignore_ascii_case(name))
}

impl LocalModel {
    /// Provider `local`, the repository as the model, the pinned commit as the
    /// revision, and input types when queries and documents get different prefixes.
    pub fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor::new("local", self.repo, self.dimensions)
            .with_input_types(self.query_prefix != self.document_prefix)
            .with_revision(self.revision)
    }

    /// The model that produces the vectors `descriptor` names.
    pub fn for_descriptor(descriptor: &ModelDescriptor) -> Result<&'static LocalModel, Error> {
        let model = MODELS
            .iter()
            .find(|m| descriptor.provider == "local" && m.repo == descriptor.model)
            .ok_or_else(|| format!("no local model makes {}", descriptor.display()))?;
        if &model.descriptor() != descriptor {
            return Err(format!(
                "{} was embedded with another revision or settings of {} ({:?}); this build runs revision {}",
                descriptor.display(),
                model.repo,
                descriptor.revision,
                model.revision
            )
            .into());
        }
        Ok(model)
    }

    /// The total size of the model's files.
    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// Texts per [`embed`](Embedder::embed) call ([`EmbedLimits::max_items`]).
const MAX_ITEMS: usize = 256;

/// The default tokens per forward pass ([`LocalEmbedder::with_batch_tokens`]).
pub const DEFAULT_BATCH_TOKENS: usize = 8192;

/// Runs a [`LocalModel`] on the CPU. See the [module docs](self).
pub struct LocalEmbedder {
    model: &'static LocalModel,
    encoder: Encoder,
    tokenizer: Tokenizer,
    pool: rayon::ThreadPool,
    threads: usize,
    batch_tokens: usize,
}

impl LocalEmbedder {
    /// Loads `model` from the files in `dir` (named as in [`LocalModel::files`]),
    /// after checking each one's size and SHA-256.
    pub fn load(model: &'static LocalModel, dir: &Path) -> Result<Self, Error> {
        let read = |name: &str| -> Result<Vec<u8>, Error> {
            let file = model
                .files
                .iter()
                .find(|f| f.name == name)
                .ok_or_else(|| format!("{} lists no {name}", model.repo))?;
            let path = dir.join(name);
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let digest: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if bytes.len() as u64 != file.size || digest != file.sha256 {
                return Err(format!(
                    "{}: not the pinned file of {} at {} (size or SHA-256 differs); delete it to download it again",
                    path.display(),
                    model.repo,
                    model.revision
                )
                .into());
            }
            Ok(bytes)
        };
        let config = Config::parse(model.architecture, &read("config.json")?)?;
        let tokenizer = Tokenizer::from_bytes(read("tokenizer.json")?)
            .map_err(|e| format!("{}: tokenizer.json: {e}", model.repo))?;
        let tensors = Tensors::parse(read("model.safetensors")?)?;
        let encoder = Encoder::load(config, &tensors)?;
        Self::new(model, encoder, tokenizer)
    }

    fn new(
        model: &'static LocalModel,
        encoder: Encoder,
        mut tokenizer: Tokenizer,
    ) -> Result<Self, Error> {
        if encoder.hidden() != model.dimensions {
            return Err(format!(
                "{}: the encoder has {} dimensions, not {}",
                model.repo,
                encoder.hidden(),
                model.dimensions
            )
            .into());
        }
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: model.max_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|e| format!("{}: truncation: {e}", model.repo))?;
        let threads = crate::threads();
        Ok(Self {
            model,
            encoder,
            tokenizer,
            pool: pool(threads)?,
            threads,
            batch_tokens: DEFAULT_BATCH_TOKENS,
        })
    }

    /// Threads for tokenizing and inference (default: [`threads`](crate::threads)).
    pub fn with_threads(mut self, threads: usize) -> Result<Self, Error> {
        self.threads = threads.max(1);
        self.pool = pool(self.threads)?;
        Ok(self)
    }

    /// Tokens per forward pass (default [`DEFAULT_BATCH_TOKENS`]). Larger batches keep
    /// more threads busy and take more memory: about 12 KB per token for a model with
    /// 768 dimensions.
    pub fn with_batch_tokens(mut self, tokens: usize) -> Self {
        self.batch_tokens = tokens.max(1);
        self
    }

    pub fn model(&self) -> &'static LocalModel {
        self.model
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Token ids of each text with the prefix of `kind`, special tokens included,
    /// truncated at the model's limit.
    fn encode(&self, texts: &[&str], kind: InputKind) -> Result<Vec<Vec<u32>>, Error> {
        let prefix = match kind {
            InputKind::Query => self.model.query_prefix,
            InputKind::Document => self.model.document_prefix,
        };
        let inputs: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let encodings = self
            .pool
            .install(|| self.tokenizer.encode_batch_fast(inputs, true))
            .map_err(|e| format!("tokenizing for {}: {e}", self.model.repo))?;
        let ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
        if ids.iter().any(Vec::is_empty) {
            return Err(format!("{}: a text produced no tokens", self.model.repo).into());
        }
        Ok(ids)
    }

    /// One forward pass over `batch`; each text's unit vector.
    fn run(&self, batch: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Error> {
        let lens: Vec<usize> = batch.iter().map(Vec::len).collect();
        let ids = batch.concat();
        let h = self.encoder.hidden();
        let states = self.pool.install(|| self.encoder.forward(&ids, &lens))?;
        let mut out = Vec::with_capacity(batch.len());
        let mut start = 0;
        for len in lens {
            let rows = &states[start * h..(start + len) * h];
            let mut v = match self.model.pooling {
                Pooling::Cls => rows[..h].to_vec(),
                Pooling::Mean => {
                    let mut sum = vec![0f32; h];
                    for row in rows.chunks_exact(h) {
                        for (s, x) in sum.iter_mut().zip(row) {
                            *s += x;
                        }
                    }
                    sum.iter_mut().for_each(|s| *s /= len as f32);
                    sum
                }
            };
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            out.push(v);
            start += len;
        }
        Ok(out)
    }
}

fn pool(threads: usize) -> Result<rayon::ThreadPool, Error> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .thread_name(|i| format!("arko-embed-{i}"))
        .build()
        .map_err(|e| format!("starting the embedding threads: {e}").into())
}

impl Embedder for LocalEmbedder {
    fn descriptor(&self) -> ModelDescriptor {
        self.model.descriptor()
    }

    /// Up to 256 texts per call, at least the model's own limit of tokens per call;
    /// no text is too long, since longer texts are truncated.
    fn limits(&self) -> EmbedLimits {
        EmbedLimits::new(
            MAX_ITEMS,
            self.batch_tokens.max(self.model.max_tokens) as u64,
            self.model.max_tokens as u64,
        )
    }

    /// Tokens as documents, after truncation.
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        Ok(self
            .encode(texts, InputKind::Document)?
            .iter()
            .map(|ids| ids.len() as u64)
            .collect())
    }

    /// Returns the tokens processed as `tokens` (nothing is billed).
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        let ids = self.encode(texts, kind)?;
        let mut vectors = Vec::with_capacity(ids.len());
        let mut tokens = 0u64;
        // Forward passes of at most `batch_tokens` (one text may exceed it alone).
        let mut i = 0;
        while i < ids.len() {
            let (mut j, mut sum) = (i, 0);
            while j < ids.len() && (j == i || sum + ids[j].len() <= self.batch_tokens) {
                sum += ids[j].len();
                j += 1;
            }
            vectors.extend(self.run(&ids[i..j])?);
            tokens += sum as u64;
            i = j;
        }
        Ok(Embedding { vectors, tokens })
    }
}

#[cfg(test)]
mod tests;
