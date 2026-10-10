//! Local embedding models for the examples (`--features local-embed`): fetches a
//! model's pinned files into the Hugging Face cache once, then loads them with
//! arko_index's [`LocalEmbedder`], which checks every file's size and SHA-256.
//!
//! [`LazyEmbedder`] loads the model on its first use: a process that only records
//! which model an index uses, such as `index2` before it hands the embedding to a
//! background run, never reads the model's 548 MB. On a GPU it finds the device when
//! it is made, which reads no model either.
//!
//! The file depends only on arko_index and hf-hub, so a host of the code-search tool
//! can take it unchanged, as with `voyage.rs`.
use arko_index::{
    bench_v1::Error,
    dense::{
        EmbedLimits, Embedder, Embedding, InputKind, ModelDescriptor,
        local::{Device, LocalEmbedder, LocalModel, Placement, Precision},
    },
};
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::{path::PathBuf, sync::OnceLock};

/// The directory of `model`'s files in the Hugging Face cache (`$HF_HOME/hub`, by
/// default `~/.cache/huggingface/hub`), downloading the files it lacks.
pub fn model_dir(model: &LocalModel) -> Result<PathBuf, Error> {
    let repo = Repo::with_revision(
        model.repo.to_owned(),
        RepoType::Model,
        model.revision.to_owned(),
    );
    let cached = Cache::from_env().repo(repo.clone());
    let missing: u64 = model
        .files
        .iter()
        .filter(|f| cached.get(f.name).is_none())
        .map(|f| f.size)
        .sum();
    if missing > 0 {
        eprintln!(
            "arko_index: downloading {} at {} ({} MB) into the Hugging Face cache",
            model.repo,
            &model.revision[..12],
            missing.div_ceil(1 << 20)
        );
    }
    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| format!("Hugging Face: {e}"))?
        .repo(repo);
    let mut dir = None;
    for file in model.files {
        let path = api
            .get(file.name)
            .map_err(|e| format!("downloading {} from {}: {e}", file.name, model.repo))?;
        dir = path.parent().map(PathBuf::from);
    }
    dir.ok_or_else(|| format!("{} lists no files", model.repo).into())
}

/// `model` in `precision` on the CPU, loaded from the cache (downloaded first if need
/// be), with `ARKO_INDEX_THREADS` threads (default: every CPU this process may use).
pub fn embedder(model: &'static LocalModel, precision: Precision) -> Result<LocalEmbedder, Error> {
    LocalEmbedder::load_with_precision(model, &model_dir(model)?, precision)
}

/// [`embedder`] on the device `placement` found, saying so when it is a GPU.
pub fn embedder_placed(
    model: &'static LocalModel,
    precision: Precision,
    placement: &Placement,
) -> Result<LocalEmbedder, Error> {
    let e = LocalEmbedder::load_placed(model, &model_dir(model)?, precision, placement)?;
    if e.on_gpu() {
        eprintln!(
            "arko_index: {} in {precision} on {}",
            model.repo,
            e.device()
        );
    }
    Ok(e)
}

/// How a [`LazyEmbedder`] loads its model.
type Loader = Box<dyn Fn() -> Result<LocalEmbedder, Error> + Send + Sync>;

/// A local model loaded on first use. Its descriptor and limits come from the catalog
/// and the device, so asking for them loads nothing. The first token count or embedding
/// downloads the model if need be and loads it, and a failure is kept and returned from
/// then on.
pub struct LazyEmbedder {
    model: &'static LocalModel,
    precision: Precision,
    limits: EmbedLimits,
    load: Loader,
    loaded: OnceLock<Result<LocalEmbedder, String>>,
}

impl LazyEmbedder {
    /// `model` in `precision` on the CPU, loaded by [`embedder`] on first use.
    pub fn new(model: &'static LocalModel, precision: Precision) -> Self {
        Self::with_loader(model, precision, move || embedder(model, precision))
    }

    /// `model` in `precision` on `device`, loaded by [`embedder_placed`] on first use.
    /// The device is found now ([`Placement::find`]), so a device this build or
    /// machine lacks is an error here, and the limits are those of the device found.
    pub fn on(
        model: &'static LocalModel,
        precision: Precision,
        device: Device,
    ) -> Result<Self, Error> {
        let placement = Placement::find(device, precision)?;
        let limits = placement.limits(model);
        let mut lazy = Self::with_loader(model, precision, move || {
            embedder_placed(model, precision, &placement)
        });
        lazy.limits = limits;
        Ok(lazy)
    }

    /// `model` in `precision` on the CPU, loaded by `load` on first use.
    pub fn with_loader(
        model: &'static LocalModel,
        precision: Precision,
        load: impl Fn() -> Result<LocalEmbedder, Error> + Send + Sync + 'static,
    ) -> Self {
        Self {
            model,
            precision,
            limits: model.limits(),
            load: Box::new(load),
            loaded: OnceLock::new(),
        }
    }

    /// Whether the model was loaded, or its loading tried.
    pub fn is_loaded(&self) -> bool {
        self.loaded.get().is_some()
    }

    fn embedder(&self) -> Result<&LocalEmbedder, Error> {
        self.loaded
            .get_or_init(|| (self.load)().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| e.clone().into())
    }
}

impl Embedder for LazyEmbedder {
    fn descriptor(&self) -> ModelDescriptor {
        self.model.descriptor(self.precision)
    }

    fn limits(&self) -> EmbedLimits {
        self.limits
    }

    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        self.embedder()?.count_tokens(texts)
    }

    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        self.embedder()?.embed(texts, kind)
    }
}
