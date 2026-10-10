//! Embedders for the examples: Voyage through arko_ai, local models (with
//! `--features local-embed`), and arko_index's offline `HashEmbedder`, and the
//! processes of the CLI's background embedding (`process`). Shared by
//! `arko_index_cli`, `dense_live_check` and `local_embed_bench`.
#![allow(dead_code)]
#[cfg(feature = "local-embed")]
pub mod local;
pub mod process;
pub mod voyage;

use arko_ai::{EmbeddingModelDetails, EmbeddingProviderIdentifier};
use arko_index::{
    bench_v1::Error,
    dense::{EmbedLimits, Embedder, Embedding, HashEmbedder, InputKind, ModelDescriptor},
};

/// The default dimensions of the offline hash embedder.
pub const HASH_DIMENSIONS: usize = 256;

/// The model an `--embed` value names: `voyage-code-4` (a bare Voyage model name),
/// `voyage:MODEL`, `local`, `local:MODEL`, `local:PRECISION` or
/// `local:MODEL:PRECISION` (a model run on this machine, in `int8` by default or
/// `f32`; needs `--features local-embed`), or `hash` / `hash:hash-v1` (offline; for
/// trying things out without a key). `dimensions` overrides the model's default.
pub fn descriptor_for(spec: &str, dimensions: Option<usize>) -> Result<ModelDescriptor, Error> {
    let (provider, model) = match spec.split_once(':') {
        Some((p, m)) => (p, m),
        None if spec == "hash" => ("hash", "hash-v1"),
        None if spec == "local" => ("local", ""),
        None if spec.starts_with("voyage") => ("voyage", spec),
        None => {
            return Err(format!(
                "unknown embedding model {spec:?}: use a Voyage model (voyage-code-4), voyage:MODEL, local, local:MODEL[:PRECISION] or hash"
            )
            .into());
        }
    };
    match provider {
        "voyage" => {
            let details = EmbeddingModelDetails::get(EmbeddingProviderIdentifier::Voyage, model)?;
            let dims = dimensions.unwrap_or(details.default_dimensions);
            if !details.supported_dimensions.contains(&dims) {
                return Err(format!(
                    "{model} supports {:?} dimensions, not {dims}",
                    details.supported_dimensions
                )
                .into());
            }
            Ok(ModelDescriptor::new("voyage", model, dims).with_input_types(true))
        }
        "hash" if model == "hash-v1" => Ok(ModelDescriptor::new(
            "hash",
            "hash-v1",
            dimensions.unwrap_or(HASH_DIMENSIONS),
        )),
        "local" => local_descriptor(model, dimensions),
        _ => Err(format!("unknown embedding model {spec:?}").into()),
    }
}

/// A local model by name or repository (`""`: the default model), optionally
/// followed by `:int8` or `:f32` (or a precision alone).
#[cfg(feature = "local-embed")]
fn local_descriptor(spec: &str, dimensions: Option<usize>) -> Result<ModelDescriptor, Error> {
    use arko_index::dense::local::{self, Precision};
    let (model, precision) = match spec.rsplit_once(':') {
        Some((model, precision)) => (model, precision.parse()?),
        None => match spec.parse::<Precision>() {
            Ok(precision) => ("", precision),
            Err(_) => (spec, Precision::default()),
        },
    };
    let m = if model.is_empty() {
        local::DEFAULT_MODEL
    } else {
        local::find(model).ok_or_else(|| {
            let names: Vec<&str> = local::MODELS.iter().map(|m| m.name).collect();
            format!(
                "unknown local model {model:?}: use one of {}",
                names.join(", ")
            )
        })?
    };
    if let Some(d) = dimensions.filter(|&d| d != m.dimensions) {
        return Err(format!("{} has {} dimensions, not {d}", m.name, m.dimensions).into());
    }
    Ok(m.descriptor(precision))
}

#[cfg(not(feature = "local-embed"))]
fn local_descriptor(_: &str, _: Option<usize>) -> Result<ModelDescriptor, Error> {
    Err(NO_LOCAL.into())
}

const NO_LOCAL: &str = "local models need arko_index built with --features local-embed";

/// An embedder that produces the vectors `descriptor` names (Voyage needs
/// `VOYAGE_API_KEY`; a local model loads on first use, downloaded if need be).
pub fn embedder_for(descriptor: &ModelDescriptor) -> Result<Box<dyn Embedder>, Error> {
    let embedder: Box<dyn Embedder> = match descriptor.provider.as_str() {
        "voyage" => Box::new(voyage::VoyageEmbedder::from_env(
            &descriptor.model,
            Some(descriptor.dimensions),
            descriptor.input_types,
        )?),
        "hash" => Box::new(PacedHash::new(descriptor.dimensions)),
        #[cfg(feature = "local-embed")]
        "local" => {
            let (model, precision) =
                arko_index::dense::local::LocalModel::for_descriptor(descriptor)?;
            Box::new(local::LazyEmbedder::new(model, precision))
        }
        #[cfg(not(feature = "local-embed"))]
        "local" => return Err(NO_LOCAL.into()),
        p => return Err(format!("unknown embedding provider {p:?}").into()),
    };
    if &embedder.descriptor() != descriptor {
        return Err(format!("cannot make an embedder for {}", descriptor.display()).into());
    }
    Ok(embedder)
}

/// The offline hash embedder, waiting `ARKO_INDEX_HASH_DELAY_MS` per document request
/// when that is set: background embedding at the pace of a real model, offline.
struct PacedHash {
    inner: HashEmbedder,
    delay: std::time::Duration,
}
impl PacedHash {
    fn new(dimensions: usize) -> Self {
        let millis = std::env::var("ARKO_INDEX_HASH_DELAY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        Self {
            inner: HashEmbedder::new(dimensions),
            delay: std::time::Duration::from_millis(millis),
        }
    }
}
impl Embedder for PacedHash {
    fn descriptor(&self) -> ModelDescriptor {
        self.inner.descriptor()
    }
    fn limits(&self) -> EmbedLimits {
        self.inner.limits()
    }
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        self.inner.count_tokens(texts)
    }
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        if kind == InputKind::Document && !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        self.inner.embed(texts, kind)
    }
}
