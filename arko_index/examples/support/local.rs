//! Local embedding models for the examples (`--features local-embed`): fetches a
//! model's pinned files into the Hugging Face cache once, then loads them with
//! arko_index's [`LocalEmbedder`], which checks every file's size and SHA-256.
//!
//! The file depends only on arko_index and hf-hub, so a host of the code-search tool
//! can take it unchanged, as with `voyage.rs`.
use arko_index::{
    bench_v1::Error,
    dense::local::{LocalEmbedder, LocalModel},
};
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::path::PathBuf;

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

/// `model`, loaded from the cache (downloaded first if need be), with
/// `ARKO_INDEX_THREADS` threads (default: every CPU this process may use).
pub fn embedder(model: &'static LocalModel) -> Result<LocalEmbedder, Error> {
    LocalEmbedder::load(model, &model_dir(model)?)
}
