//! Local embedding models (feature `local-embed`): [`LocalEmbedder`] runs a small
//! code-embedding model on this machine, in pure Rust, behind the same [`Embedder`]
//! interface as a provider, so dense and hybrid search work with no API and no
//! per-token cost.
//!
//! [`MODELS`] lists the models it runs, each pinned to a Hugging Face commit with the
//! size and SHA-256 of every file. The library reads the files from a directory
//! ([`LocalEmbedder::load`]) and checks them. Fetching them is the caller's: the
//! examples download into the Hugging Face cache (`examples/support/local.rs`).
//!
//! A batch of texts is tokenized with the model's own tokenizer, run through the
//! encoder, pooled ([`Pooling`]) and normalized to unit length. A text longer than
//! [`LocalModel::max_tokens`] is embedded in windows the model takes, each with its
//! special tokens, and its vector is their mean weighted by the text tokens each holds,
//! at unit length. On the CPU the texts run packed together without padding, the linear
//! layers in int8 by default ([`Precision`]), spread over a dedicated thread pool
//! ([`LocalEmbedder::with_threads`], by default [`threads`](crate::threads)). With the
//! features `local-embed-gpu` (wgpu: Vulkan, Metal, DirectX 12) and `local-embed-cuda`
//! the same models run on a GPU ([`Device`]), in `f32`, `f16` or `bf16`, in padded
//! batches of texts of similar length. Both devices split and combine long texts the
//! same way.
#[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
mod gpu;
mod int8;
mod model;
mod safetensors;
mod window;

pub use model::{Architecture, Pooling};

use super::{EmbedLimits, Embedder, Embedding, InputKind, ModelDescriptor};
use crate::bench_v1::Error;
use model::{Config, Encoder};
use safetensors::Tensors;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokenizers::Tokenizer;

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
    /// Tokens per text the model takes, special tokens included. Longer texts are
    /// embedded in windows of this many tokens.
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

/// The arithmetic a local model runs in. On the CPU: `int8` (the default) or `f32`.
/// On a GPU (features `local-embed-gpu` and `local-embed-cuda`): `f32`, `f16` or
/// `bf16` (CUDA only). The precision is part of the vectors' identity
/// ([`LocalModel::descriptor`]), so an index never mixes vectors of two precisions. The
/// device is not: `f32` vectors from the CPU and from a GPU differ only by rounding, and
/// one index takes both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// CPU only. The linear layers after attention (the output projection and
    /// feed-forward layers, which hold most of the weights and work) run on integers:
    /// weights rotated by a block Hadamard matrix and quantized per output channel to
    /// 7 bits at load, activations rotated the same way and quantized per token to 8
    /// bits at run time, products summed exactly in 32-bit integers. The query, key and
    /// value projection, attention, embeddings, layer norms, pooling and normalization
    /// run in `f32`.
    #[default]
    Int8,
    /// 32-bit floats, as the model was trained.
    F32,
    /// GPU only: every matrix product (the linear layers and attention's two) takes
    /// 16-bit floats (IEEE half precision), with the weights stored in it. Layer norms,
    /// softmax, residual sums, pooling and normalization run in `f32`.
    F16,
    /// CUDA only: as [`F16`](Self::F16) with bfloat16, which has `f32`'s range and
    /// fewer mantissa bits.
    Bf16,
}

impl Precision {
    /// Every precision, the default first.
    pub const ALL: [Precision; 4] = [
        Precision::Int8,
        Precision::F32,
        Precision::F16,
        Precision::Bf16,
    ];

    /// `int8`, `f32`, `f16` or `bf16`.
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Int8 => "int8",
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
        }
    }

    /// Whether the CPU runs this precision (`int8` and `f32`).
    pub fn runs_on_cpu(self) -> bool {
        matches!(self, Precision::Int8 | Precision::F32)
    }

    /// Whether a GPU runs this precision (`f32`, `f16`, and `bf16` with CUDA).
    pub fn runs_on_gpu(self) -> bool {
        !matches!(self, Precision::Int8)
    }

    /// The suffix of the descriptor's revision: none for `f32`, the weights as
    /// published.
    fn suffix(self) -> &'static str {
        match self {
            Precision::Int8 => "+int8",
            Precision::F32 => "",
            Precision::F16 => "+f16",
            Precision::Bf16 => "+bf16",
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Precision {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        Precision::ALL
            .into_iter()
            .find(|p| p.as_str().eq_ignore_ascii_case(s))
            .ok_or_else(|| format!("unknown precision {s:?}: use int8, f32, f16 or bf16").into())
    }
}

/// Where a local model runs ([`LocalEmbedder::load_on`], [`Placement::find`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// The CPU for `int8`. For `f32`: CUDA when this build has it (feature
    /// `local-embed-cuda`) and the machine an NVIDIA GPU with its driver and NVRTC,
    /// else a discrete GPU through wgpu (feature `local-embed-gpu`), else the CPU. For
    /// `f16` and `bf16`, as [`Gpu`](Self::Gpu).
    #[default]
    Auto,
    /// The CPU (`int8` and `f32`).
    Cpu,
    /// CUDA when available, else any GPU wgpu finds, discrete ones first.
    Gpu,
    /// An NVIDIA GPU through CUDA (feature `local-embed-cuda`): the first one the
    /// driver lists (`CUDA_VISIBLE_DEVICES` picks another).
    Cuda,
    /// A GPU through wgpu (feature `local-embed-gpu`): Vulkan, Metal or DirectX 12,
    /// discrete GPUs first, then integrated and virtual ones, then wgpu's CPU adapters.
    Wgpu,
}

impl Device {
    /// Every device, the default first.
    pub const ALL: [Device; 5] = [
        Device::Auto,
        Device::Cpu,
        Device::Gpu,
        Device::Cuda,
        Device::Wgpu,
    ];

    /// `auto`, `cpu`, `gpu`, `cuda` or `wgpu`.
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
            Device::Cuda => "cuda",
            Device::Wgpu => "wgpu",
        }
    }

    /// The precision a model runs in when none is named: `int8` on the CPU (and
    /// [`Auto`](Self::Auto)), `f32` on a GPU.
    pub fn default_precision(self) -> Precision {
        match self {
            Device::Auto | Device::Cpu => Precision::Int8,
            Device::Gpu | Device::Cuda | Device::Wgpu => Precision::F32,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Device {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        Device::ALL
            .into_iter()
            .find(|d| d.as_str().eq_ignore_ascii_case(s))
            .ok_or_else(|| format!("unknown device {s:?}: use auto, cpu, gpu, cuda or wgpu").into())
    }
}

/// The models the local embedder runs. The first is [`DEFAULT_MODEL`].
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
    /// revision (followed by `+int8`, `+f16` or `+bf16` for those precisions), and
    /// input types when queries and documents get different prefixes.
    pub fn descriptor(&self, precision: Precision) -> ModelDescriptor {
        ModelDescriptor::new("local", self.repo, self.dimensions)
            .with_input_types(self.query_prefix != self.document_prefix)
            .with_revision(format!("{}{}", self.revision, precision.suffix()))
    }

    /// The model and precision that produce the vectors `descriptor` names.
    pub fn for_descriptor(
        descriptor: &ModelDescriptor,
    ) -> Result<(&'static LocalModel, Precision), Error> {
        let model = MODELS
            .iter()
            .find(|m| descriptor.provider == "local" && m.repo == descriptor.model)
            .ok_or_else(|| format!("no local model makes {}", descriptor.display()))?;
        Precision::ALL
            .into_iter()
            .find(|&p| &model.descriptor(p) == descriptor)
            .map(|p| (model, p))
            .ok_or_else(|| {
                format!(
                    "{} was embedded with another revision or settings of {} ({:?}); this build runs revision {}",
                    descriptor.display(),
                    model.repo,
                    descriptor.revision,
                    model.revision
                )
                .into()
            })
    }

    /// The total size of the model's files.
    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// The limits of a [`LocalEmbedder`] of this model on the CPU with the default
    /// batch size, known without loading it. [`Placement::limits`] gives them for any
    /// device.
    pub fn limits(&self) -> EmbedLimits {
        limits(self, false, DEFAULT_BATCH_TOKENS)
    }
}

/// Texts per [`embed`](Embedder::embed) call ([`EmbedLimits::max_items`]) on the CPU.
const MAX_ITEMS: usize = 256;

/// Texts per [`embed`](Embedder::embed) call on a GPU: enough for batches of similar
/// lengths.
const GPU_MAX_ITEMS: usize = 1024;

/// Forward passes' worth of tokens per [`embed`](Embedder::embed) call on a GPU.
const GPU_REQUEST_PASSES: usize = 8;

/// The request limits with `batch_tokens` per forward pass: on the CPU up to 256 texts
/// and the larger of `batch_tokens` and the model's input per request, on a GPU up to
/// 1,024 texts and the larger of eight passes and the model's input. No text is too
/// long: a longer one goes in a request of its own and is embedded in windows.
fn limits(model: &LocalModel, on_gpu: bool, batch_tokens: usize) -> EmbedLimits {
    let (items, tokens) = if on_gpu {
        (
            GPU_MAX_ITEMS,
            GPU_REQUEST_PASSES.saturating_mul(batch_tokens),
        )
    } else {
        (MAX_ITEMS, batch_tokens)
    };
    EmbedLimits::new(items, tokens.max(model.max_tokens) as u64, u64::MAX)
}

/// The default tokens per forward pass on the CPU ([`LocalEmbedder::with_batch_tokens`]).
pub const DEFAULT_BATCH_TOKENS: usize = 8192;

/// The default tokens per forward pass on a GPU, padding included
/// ([`LocalEmbedder::with_batch_tokens`]).
pub const DEFAULT_GPU_BATCH_TOKENS: usize = 4096;

/// Runs a [`LocalModel`] on the CPU or a GPU. See the [module docs](self).
pub struct LocalEmbedder {
    model: &'static LocalModel,
    precision: Precision,
    engine: Engine,
    tokenizer: Tokenizer,
    /// The query and document prefixes' tokens, which every window of a long text
    /// repeats.
    query_prefix: Vec<u32>,
    document_prefix: Vec<u32>,
    pool: rayon::ThreadPool,
    threads: usize,
    batch_tokens: usize,
    stats: std::sync::Mutex<BatchStats>,
}

enum Engine {
    Cpu(Encoder),
    #[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
    Gpu(gpu::Gpu),
}

/// The forward passes an embedder ran since it was loaded
/// ([`LocalEmbedder::batch_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct BatchStats {
    pub passes: u64,
    /// The sequences the passes ran: texts, or windows of longer texts.
    pub texts: u64,
    /// The sequences' tokens.
    pub tokens: u64,
    /// The tokens the passes computed: on a GPU the sequences' tokens and their
    /// padding, on the CPU (which packs them without padding) the same as `tokens`.
    pub padded_tokens: u64,
}

impl LocalEmbedder {
    /// Loads `model` from the files in `dir` (named as in [`LocalModel::files`]),
    /// after checking each one's size and SHA-256, to run on the CPU in the default
    /// [`Precision`] (int8).
    pub fn load(model: &'static LocalModel, dir: &Path) -> Result<Self, Error> {
        Self::load_with_precision(model, dir, Precision::default())
    }

    /// [`load`](Self::load) on the CPU in `precision` (`int8` or `f32`).
    pub fn load_with_precision(
        model: &'static LocalModel,
        dir: &Path,
        precision: Precision,
    ) -> Result<Self, Error> {
        Self::load_on(model, dir, precision, Device::Cpu)
    }

    /// [`load`](Self::load) on `device` in `precision`. [`Device::Auto`] picks a GPU
    /// for the precisions a GPU runs, when this build and machine have one, and the CPU
    /// otherwise ([`device`](Self::device) says which).
    pub fn load_on(
        model: &'static LocalModel,
        dir: &Path,
        precision: Precision,
        device: Device,
    ) -> Result<Self, Error> {
        Self::load_placed(model, dir, precision, &Placement::find(device, precision)?)
    }

    /// [`load`](Self::load) on the device `placement` found for `precision`. A caller
    /// that loads the model on its first use finds the placement first, which is quick
    /// and reads no model, and reports its [`limits`](Placement::limits) until then.
    pub fn load_placed(
        model: &'static LocalModel,
        dir: &Path,
        precision: Precision,
        placement: &Placement,
    ) -> Result<Self, Error> {
        // `Option<Infallible>`, which is `Copy`, in a build without GPUs.
        #[allow(clippy::clone_on_copy)]
        let gpu = placement.gpu.clone();
        Self::load_choice(model, dir, precision, gpu)
    }

    /// [`load`](Self::load) on the CPU (`None`) or a GPU found before.
    fn load_choice(
        model: &'static LocalModel,
        dir: &Path,
        precision: Precision,
        gpu: Option<GpuChoice>,
    ) -> Result<Self, Error> {
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
        // Quantizing the weights runs on the embedder's threads.
        let threads = crate::threads();
        let pool = pool(threads)?;
        let engine = match gpu {
            None => Engine::Cpu(pool.install(|| Encoder::load(config, &tensors, precision))?),
            #[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
            Some(choice) => Engine::Gpu(gpu::Gpu::load(
                choice,
                config,
                model.pooling,
                &tensors,
                precision,
            )?),
        };
        drop(tensors);
        Self::new(model, precision, engine, tokenizer, pool, threads)
    }

    fn new(
        model: &'static LocalModel,
        precision: Precision,
        engine: Engine,
        mut tokenizer: Tokenizer,
        pool: rayon::ThreadPool,
        threads: usize,
    ) -> Result<Self, Error> {
        let (hidden, batch_tokens) = match &engine {
            Engine::Cpu(encoder) => (encoder.hidden(), DEFAULT_BATCH_TOKENS),
            #[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
            Engine::Gpu(g) => (g.hidden(), DEFAULT_GPU_BATCH_TOKENS),
        };
        if hidden != model.dimensions {
            return Err(format!(
                "{}: the encoder has {hidden} dimensions, not {}",
                model.repo, model.dimensions
            )
            .into());
        }
        tokenizer.with_padding(None);
        // Whole texts: a `tokenizer.json` may set a truncation (CodeRankEmbed's, 512).
        tokenizer
            .with_truncation(None)
            .map_err(|e| format!("{}: truncation: {e}", model.repo))?;
        let prefix = |text: &str| -> Result<Vec<u32>, Error> {
            Ok(tokenizer
                .encode_fast(text, false)
                .map_err(|e| format!("tokenizing for {}: {e}", model.repo))?
                .get_ids()
                .to_vec())
        };
        let (query_prefix, document_prefix) =
            (prefix(model.query_prefix)?, prefix(model.document_prefix)?);
        Ok(Self {
            model,
            precision,
            engine,
            tokenizer,
            query_prefix,
            document_prefix,
            pool,
            threads,
            batch_tokens,
            stats: Default::default(),
        })
    }

    /// Threads for tokenizing and, on the CPU, inference (default:
    /// [`threads`](crate::threads)).
    pub fn with_threads(mut self, threads: usize) -> Result<Self, Error> {
        self.threads = threads.max(1);
        self.pool = pool(self.threads)?;
        Ok(self)
    }

    /// Tokens per forward pass. On the CPU (default [`DEFAULT_BATCH_TOKENS`]) larger
    /// batches keep more threads busy and take more memory: about 58 KB per token for a
    /// model with 768 dimensions. On a GPU (default [`DEFAULT_GPU_BATCH_TOKENS`]) the
    /// count includes padding, and memory grows with the batch and with its texts'
    /// lengths (attention).
    pub fn with_batch_tokens(mut self, tokens: usize) -> Self {
        self.batch_tokens = tokens.max(1);
        self
    }

    pub fn model(&self) -> &'static LocalModel {
        self.model
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn batch_tokens(&self) -> usize {
        self.batch_tokens
    }

    /// Where the model runs, for messages: `CPU`, `CUDA: NAME` or
    /// `wgpu: NAME (API, type)`.
    pub fn device(&self) -> &str {
        match &self.engine {
            Engine::Cpu(_) => "CPU",
            #[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
            Engine::Gpu(g) => g.name(),
        }
    }

    /// Whether the model runs on a GPU.
    pub fn on_gpu(&self) -> bool {
        !matches!(self.engine, Engine::Cpu(_))
    }

    /// The forward passes since the embedder was loaded.
    pub fn batch_stats(&self) -> BatchStats {
        *self.stats.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn count_passes(&self, passes: u64, seqs: &[&[u32]], padded_tokens: u64) {
        let mut s = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        s.passes += passes;
        s.texts += seqs.len() as u64;
        s.tokens += seqs.iter().map(|t| t.len() as u64).sum::<u64>();
        s.padded_tokens += padded_tokens;
    }

    /// Each text with the prefix of `kind`, tokenized whole: its token ids, special
    /// tokens included, and how many special tokens lead and trail.
    fn encode(&self, texts: &[&str], kind: InputKind) -> Result<Vec<Encoded>, Error> {
        let prefix = match kind {
            InputKind::Query => self.model.query_prefix,
            InputKind::Document => self.model.document_prefix,
        };
        let inputs: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let encodings = self
            .pool
            .install(|| self.tokenizer.encode_batch_fast(inputs, true))
            .map_err(|e| format!("tokenizing for {}: {e}", self.model.repo))?;
        encodings
            .iter()
            .map(|e| {
                let special = e.get_special_tokens_mask();
                let lead = special.iter().take_while(|&&s| s == 1).count();
                let tail = special.iter().rev().take_while(|&&s| s == 1).count();
                if e.get_ids().is_empty() {
                    return Err(format!("{}: a text produced no tokens", self.model.repo).into());
                }
                Ok(Encoded {
                    ids: e.get_ids().to_vec(),
                    lead: lead.min(e.len()),
                    tail: tail.min(e.len() - lead.min(e.len())),
                })
            })
            .collect()
    }

    /// The unit vector of each sequence (a text or one window of it), in order, on
    /// the CPU or the GPU.
    fn run_all(&self, seqs: &[&[u32]]) -> Result<Vec<Vec<f32>>, Error> {
        match &self.engine {
            Engine::Cpu(encoder) => self.run_cpu(encoder, seqs),
            #[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
            Engine::Gpu(g) => {
                let (vectors, plan) = g.embed(seqs, self.batch_tokens)?;
                self.count_passes(plan.batches.len() as u64, seqs, plan.padded_tokens());
                Ok(vectors)
            }
        }
    }

    /// [`run_all`](Self::run_all) on the CPU: forward passes of at most `batch_tokens`
    /// (one sequence may exceed it alone).
    fn run_cpu(&self, encoder: &Encoder, seqs: &[&[u32]]) -> Result<Vec<Vec<f32>>, Error> {
        let mut vectors = Vec::with_capacity(seqs.len());
        let mut i = 0;
        while i < seqs.len() {
            let (mut j, mut sum) = (i, 0);
            while j < seqs.len() && (j == i || sum + seqs[j].len() <= self.batch_tokens) {
                sum += seqs[j].len();
                j += 1;
            }
            vectors.extend(self.run(encoder, &seqs[i..j])?);
            i = j;
        }
        Ok(vectors)
    }

    /// One forward pass over `batch` on the CPU. Each sequence's unit vector.
    fn run(&self, encoder: &Encoder, batch: &[&[u32]]) -> Result<Vec<Vec<f32>>, Error> {
        let lens: Vec<usize> = batch.iter().map(|ids| ids.len()).collect();
        let ids = batch.concat();
        let h = encoder.hidden();
        let states = self.pool.install(|| encoder.forward(&ids, &lens))?;
        self.count_passes(1, batch, ids.len() as u64);
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
        self.model.descriptor(self.precision)
    }

    /// On the CPU up to 256 texts and the larger of the batch size and the model's
    /// input per call, on a GPU up to 1,024 texts and the larger of eight batches and
    /// the model's input. No text is too long: a longer one is embedded in windows.
    fn limits(&self) -> EmbedLimits {
        limits(self.model, self.on_gpu(), self.batch_tokens)
    }

    /// Tokens as documents, special tokens included, of the whole text.
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        Ok(self
            .encode(texts, InputKind::Document)?
            .iter()
            .map(|e| e.ids.len() as u64)
            .collect())
    }

    /// One vector per text. Returns the tokens processed as `tokens`, the special
    /// tokens and prefixes that windows repeat included (nothing is billed).
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        let prefix = match kind {
            InputKind::Query => &self.query_prefix,
            InputKind::Document => &self.document_prefix,
        };
        // Every text's windows, in order: (text, window).
        let mut windows = Vec::new();
        for (t, e) in self.encode(texts, kind)?.into_iter().enumerate() {
            for w in window::split(&e.ids, e.lead, e.tail, prefix, self.model.max_tokens)? {
                windows.push((t, w));
            }
        }
        let seqs: Vec<&[u32]> = windows.iter().map(|(_, w)| &w.ids[..]).collect();
        let tokens = seqs.iter().map(|s| s.len() as u64).sum();
        let window_vectors = self.run_all(&seqs)?;
        if window_vectors.len() != windows.len() {
            return Err(format!("{}: the encoder lost vectors", self.model.repo).into());
        }
        let mut vectors = Vec::with_capacity(texts.len());
        let mut at = 0;
        while at < windows.len() {
            let text = windows[at].0;
            let end = at + windows[at..].iter().take_while(|(t, _)| *t == text).count();
            let weights: Vec<usize> = windows[at..end].iter().map(|(_, w)| w.weight).collect();
            vectors.push(window::combine(&window_vectors[at..end], &weights));
            at = end;
        }
        Ok(Embedding { vectors, tokens })
    }
}

/// A tokenized text: its ids and how many special tokens lead and trail them.
struct Encoded {
    ids: Vec<u32>,
    lead: usize,
    tail: usize,
}

#[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
type GpuChoice = gpu::Choice;
#[cfg(not(any(feature = "local-embed-gpu", feature = "local-embed-cuda")))]
type GpuChoice = std::convert::Infallible;

/// The GPU `device` takes `precision` to, or `None` for the CPU.
#[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
fn choose_gpu(device: Device, precision: Precision) -> Result<Option<GpuChoice>, Error> {
    match device {
        Device::Cpu => cpu_only(precision).map(|()| None),
        Device::Auto if precision == Precision::Int8 => Ok(None),
        _ if precision == Precision::Int8 => {
            Err("int8 runs on the CPU: a GPU runs f32, f16 or bf16".into())
        }
        _ => gpu::choose(device, precision),
    }
}

#[cfg(not(any(feature = "local-embed-gpu", feature = "local-embed-cuda")))]
fn choose_gpu(device: Device, precision: Precision) -> Result<Option<GpuChoice>, Error> {
    match device {
        Device::Cpu | Device::Auto => cpu_only(precision).map(|()| None),
        _ => Err(format!(
            "the {device} device needs arko_index built with the local-embed-gpu or local-embed-cuda feature"
        )
        .into()),
    }
}

/// Checks that the CPU runs `precision`.
fn cpu_only(precision: Precision) -> Result<(), Error> {
    if precision.runs_on_cpu() {
        Ok(())
    } else {
        Err(format!(
            "{precision} runs on a GPU (features local-embed-gpu and local-embed-cuda): the CPU runs int8 and f32"
        )
        .into())
    }
}

/// The device a model will run on, found from a [`Device`] choice and a precision
/// without loading the model ([`LocalEmbedder::load_placed`]).
#[derive(Clone)]
pub struct Placement {
    gpu: Option<GpuChoice>,
}

impl Placement {
    /// Where `device` runs `precision` on this machine: a GPU when one is named or
    /// [`Device::Auto`] finds one, else the CPU. Errors when the named device can't,
    /// or when `precision` needs a GPU and there is none. The CPU is found without
    /// looking for a GPU: `int8` on [`Device::Auto`] or anything on [`Device::Cpu`].
    pub fn find(device: Device, precision: Precision) -> Result<Self, Error> {
        Ok(Self {
            gpu: choose_gpu(device, precision)?,
        })
    }

    /// The CPU.
    pub fn cpu() -> Self {
        Self { gpu: None }
    }

    pub fn on_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// The device, for messages: `CPU`, `CUDA: NAME` or `wgpu: NAME (API, type)`.
    pub fn name(&self) -> String {
        #[cfg(any(feature = "local-embed-gpu", feature = "local-embed-cuda"))]
        if let Some(choice) = &self.gpu {
            return choice.name();
        }
        "CPU".into()
    }

    /// The request limits a model loaded here reports with its default batch tokens.
    pub fn limits(&self, model: &LocalModel) -> EmbedLimits {
        let tokens = if self.on_gpu() {
            DEFAULT_GPU_BATCH_TOKENS
        } else {
            DEFAULT_BATCH_TOKENS
        };
        limits(model, self.on_gpu(), tokens)
    }
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Placement").field(&self.name()).finish()
    }
}

#[cfg(test)]
mod tests;
