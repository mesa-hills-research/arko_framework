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
/// `voyage:MODEL`, `local[:MODEL][:PRECISION][:DEVICE]` (a model run on this machine;
/// needs `--features local-embed`: in `int8` by default or `f32` on the CPU, and in
/// `f32` by default, `f16` or `bf16` on a GPU with `local-embed-gpu` or
/// `local-embed-cuda`; the device `auto`, `cpu`, `gpu`, `cuda` or `wgpu` is
/// [`set_device`] for this process), or `hash` / `hash:hash-v1` (offline; for
/// trying things out without a key). `dimensions` overrides the model's default.
pub fn descriptor_for(spec: &str, dimensions: Option<usize>) -> Result<ModelDescriptor, Error> {
    let (provider, model) = match spec.split_once(':') {
        Some((p, m)) => (p, m),
        None if spec == "hash" => ("hash", "hash-v1"),
        None if spec == "local" => ("local", ""),
        None if spec.starts_with("voyage") => ("voyage", spec),
        None => {
            return Err(format!(
                "unknown embedding model {spec:?}. Use a Voyage model (voyage-code-4), voyage:MODEL, local, local:MODEL[:PRECISION] or hash"
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

/// A local model by name or repository (`""`: the default model), a precision and a
/// device, each optional, separated by `:`.
#[cfg(feature = "local-embed")]
fn local_descriptor(spec: &str, dimensions: Option<usize>) -> Result<ModelDescriptor, Error> {
    use arko_index::dense::local::{self, Device, Precision};
    let (mut model, mut precision, mut device) = ("", None, None);
    for part in spec.split(':').filter(|p| !p.is_empty()) {
        if let Ok(p) = part.parse::<Precision>() {
            precision = Some(p);
        } else if let Ok(d) = part.parse::<Device>() {
            device = Some(d);
        } else if model.is_empty() {
            model = part;
        } else {
            return Err(format!("{part:?} in local:{spec} is no precision or device").into());
        }
    }
    if let Some(d) = device {
        set_device(d.as_str());
    }
    let precision = precision.unwrap_or(device.unwrap_or_default().default_precision());
    let m = if model.is_empty() {
        local::DEFAULT_MODEL
    } else {
        local::find(model).ok_or_else(|| {
            let names: Vec<&str> = local::MODELS.iter().map(|m| m.name).collect();
            format!(
                "unknown local model {model:?}. Use one of {}",
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

/// The device a local model runs on in this process: one an `--embed` value named
/// ([`set_device`]), else `ARKO_INDEX_EMBED_DEVICE`, else `auto`.
static DEVICE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Sets the device of local models for this process (and the background embedding
/// it starts: [`named_device`]).
pub fn set_device(device: &str) {
    *DEVICE.lock().unwrap() = Some(device.to_owned());
}

/// The device an `--embed` value named in this process, if one did.
pub fn named_device() -> Option<String> {
    DEVICE.lock().unwrap().clone()
}

#[cfg(feature = "local-embed")]
fn local_device() -> Result<arko_index::dense::local::Device, Error> {
    match named_device().or_else(|| std::env::var("ARKO_INDEX_EMBED_DEVICE").ok()) {
        Some(d) if !d.trim().is_empty() => d.trim().parse(),
        _ => Ok(Default::default()),
    }
}

/// [`embedder_for`] a search's queries: a local model runs on the CPU when the CPU
/// runs its precision, since one query takes milliseconds there and a GPU takes
/// seconds to start. The model loads on first use, as with [`embedder_for`].
pub fn query_embedder_for(descriptor: &ModelDescriptor) -> Result<Box<dyn Embedder>, Error> {
    #[cfg(feature = "local-embed")]
    if descriptor.provider == "local" {
        let (model, precision) = arko_index::dense::local::LocalModel::for_descriptor(descriptor)?;
        let device = if precision.runs_on_cpu() {
            arko_index::dense::local::Device::Cpu
        } else {
            local_device()?
        };
        return Ok(Box::new(local::LazyEmbedder::on(model, precision, device)?));
    }
    embedder_for(descriptor)
}

/// An embedder that produces the vectors `descriptor` names (Voyage needs
/// `VOYAGE_API_KEY`; a local model loads on first use, downloaded if need be, on the
/// device [`local_device`] names).
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
            Box::new(local::LazyEmbedder::on(model, precision, local_device()?)?)
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
