//! Embedders for the examples: Voyage through arko_ai, and arko_index's offline
//! `HashEmbedder`. Shared by `arko_index_cli` and `dense_live_check`.
#![allow(dead_code)]
pub mod voyage;

use arko_ai::{EmbeddingModelDetails, EmbeddingProviderIdentifier};
use arko_index::{
    bench_v1::Error,
    dense::{Embedder, HashEmbedder, ModelDescriptor},
};

/// The default dimensions of the offline hash embedder.
pub const HASH_DIMENSIONS: usize = 256;

/// The model an `--embed` value names: `voyage-code-4` (a bare Voyage model name),
/// `voyage:MODEL`, or `hash` / `hash:hash-v1` (offline; for trying things out without
/// a key). `dimensions` overrides the model's default.
pub fn descriptor_for(spec: &str, dimensions: Option<usize>) -> Result<ModelDescriptor, Error> {
    let (provider, model) = match spec.split_once(':') {
        Some((p, m)) => (p, m),
        None if spec == "hash" => ("hash", "hash-v1"),
        None if spec.starts_with("voyage") => ("voyage", spec),
        None => {
            return Err(format!(
                "unknown embedding model {spec:?}: use a Voyage model (voyage-code-4), voyage:MODEL or hash"
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
        _ => Err(format!("unknown embedding model {spec:?}").into()),
    }
}

/// An embedder that produces the vectors `descriptor` names (Voyage needs
/// `VOYAGE_API_KEY`).
pub fn embedder_for(descriptor: &ModelDescriptor) -> Result<Box<dyn Embedder>, Error> {
    let embedder: Box<dyn Embedder> = match descriptor.provider.as_str() {
        "voyage" => Box::new(voyage::VoyageEmbedder::from_env(
            &descriptor.model,
            Some(descriptor.dimensions),
            descriptor.input_types,
        )?),
        "hash" => Box::new(HashEmbedder::new(descriptor.dimensions)),
        p => return Err(format!("unknown embedding provider {p:?}").into()),
    };
    if &embedder.descriptor() != descriptor {
        return Err(format!("cannot make an embedder for {}", descriptor.display()).into());
    }
    Ok(embedder)
}
