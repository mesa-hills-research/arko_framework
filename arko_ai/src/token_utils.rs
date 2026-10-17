use crate::{embedding_model::EmbeddingModelDetails, error::Error};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
use tokenizers::{FromPretrainedParameters, Tokenizer};

type Slot = Arc<Mutex<Option<Arc<Tokenizer>>>>;
static TOKENIZERS: OnceLock<Mutex<HashMap<(String, String), Slot>>> = OnceLock::new();

/// Counts raw text tokens for one tokenizer repository/revision, without padding,
/// truncation, or special tokens. Clones share the tokenizer that embedding requests
/// with the same repository and revision use. Obtain one with
/// [`EmbeddingClient::token_counter`](crate::EmbeddingClient::token_counter).
#[derive(Clone)]
pub struct TokenCounter {
    tokenizer: Arc<Tokenizer>,
}

impl TokenCounter {
    pub(crate) fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self { tokenizer }
    }

    pub fn count(&self, text: &str) -> Result<usize, Error> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.len())
            .map_err(|e| Error::TokenizerError(e.to_string()))
    }
}

impl std::fmt::Debug for TokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCounter").finish_non_exhaustive()
    }
}

/// Loads one reusable, untruncated tokenizer per repository/revision. Failed downloads
/// remain retryable. Concurrent loads of the same revision share one initialization.
/// Call from a blocking worker when the tokenizer is not already cached.
pub(crate) fn load_tokenizer(
    id: &str,
    revision: Option<&str>,
    hf_token: Option<&str>,
    requires_hf_token: bool,
) -> Result<Arc<Tokenizer>, Error> {
    if requires_hf_token && hf_token.is_none() {
        return Err(Error::MissingHuggingFaceToken(id.into()));
    }
    let revision = revision.unwrap_or("main");
    let slot = {
        let mut cache = TOKENIZERS
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| Error::TokenizerError("tokenizer cache lock poisoned".into()))?;
        cache
            .entry((id.into(), revision.into()))
            .or_default()
            .clone()
    };
    let mut loaded = slot
        .lock()
        .map_err(|_| Error::TokenizerError("tokenizer initialization lock poisoned".into()))?;
    if let Some(tokenizer) = &*loaded {
        return Ok(tokenizer.clone());
    }
    if let Some(error) = cache_location_error(
        std::env::var_os("HF_HOME").as_deref(),
        std::env::home_dir().as_deref(),
    ) {
        return Err(error);
    }
    let parameters = FromPretrainedParameters {
        revision: revision.into(),
        token: hf_token.map(str::to_owned),
        ..Default::default()
    };
    let mut tokenizer = without_panics(|| Tokenizer::from_pretrained(id, Some(parameters)))?
        .map_err(|e| load_error(id, revision, e))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| Error::TokenizerError(e.to_string()))?;
    tokenizer.with_padding(None);
    let tokenizer = Arc::new(tokenizer);
    *loaded = Some(tokenizer.clone());
    Ok(tokenizer)
}

/// Why a tokenizer failed to load. Tokenizers come from the Hugging Face Hub
/// on first use and are then read from its cache with no network request, so
/// a failed download says what to do without network access to the Hub. A
/// Hub that answers 404 has no such repository or revision (a pinned
/// revision the repository doesn't have), which is said plainly. Other
/// failures (an invalid id, a damaged cached file) are reported as they are.
fn load_error(id: &str, revision: &str, error: impl std::fmt::Display) -> Error {
    let error = error.to_string();
    if is_download_failure(&error) && error.contains("status code 404") {
        return Error::TokenizerError(format!(
            "Failed to load the tokenizer {id}@{revision}: the Hugging Face Hub has no tokenizer at this revision of {id} ({error}). Check the model name and the tokenizer revision (a commit, branch or tag of {id}); without a pinned revision the repository's current version is loaded."
        ));
    }
    if !is_download_failure(&error) {
        return Error::TokenizerError(format!(
            "Failed to load the tokenizer {id}@{revision}: {error}"
        ));
    }
    Error::TokenizerError(format!(
        "Failed to download the tokenizer {id}@{revision} from the Hugging Face Hub: {error}. Tokens are counted locally with this tokenizer before a request is sent, so it is downloaded once from the Hub (huggingface.co) and then read from the Hugging Face cache. Without network access to the Hub, set HF_ENDPOINT to a mirror, or copy the cached tokenizer from a machine that has it into the cache (HF_HOME, by default ~/.cache/huggingface)."
    ))
}

/// Whether a load error is the Hub client's for a failed download: the
/// request failed (`request error: ...`, which names the URL and the
/// network failure), was retried to its limit, or got an answer without the
/// headers a download needs. These are the messages of hf-hub's `ApiError`,
/// which the tokenizers crate passes on as they are.
fn is_download_failure(error: &str) -> bool {
    ["request error:", "Too many retries", "Header "]
        .iter()
        .any(|prefix| error.starts_with(prefix))
}

/// The Hugging Face cache lives in `HF_HOME`, else in the home directory.
/// With neither, the Hub client can't place it (it would panic).
fn cache_location_error(
    hf_home: Option<&std::ffi::OsStr>,
    home: Option<&std::path::Path>,
) -> Option<Error> {
    (hf_home.is_none() && home.is_none()).then(|| {
        Error::TokenizerError(
            "Cannot load a tokenizer: there is no home directory for the Hugging Face cache. Set HF_HOME to a writable directory."
                .into(),
        )
    })
}

/// Runs `load`, turning a panic in the Hub client (such as a cache directory
/// that can't be found) into an error.
fn without_panics<T>(load: impl FnOnce() -> T) -> Result<T, Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(load)).map_err(|panic| {
        let reason = panic
            .downcast_ref::<&str>()
            .map(|reason| reason.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown cause".into());
        Error::TokenizerError(format!(
            "Loading a tokenizer failed ({reason}). If no home directory is available, set HF_HOME to a writable directory for the Hugging Face cache."
        ))
    })
}

/// Counts raw text tokens without padding, truncation, or special tokens.
pub fn count_tokens<I, T>(
    id: &str,
    texts: I,
    hf_token: Option<&str>,
    requires_hf_token: bool,
) -> Result<Vec<usize>, Error>
where
    I: IntoIterator<Item = T>,
    T: AsRef<str>,
{
    count_tokens_with_revision(id, texts, hf_token, requires_hf_token, None)
}

pub fn count_tokens_with_revision<I, T>(
    id: &str,
    texts: I,
    hf_token: Option<&str>,
    requires_hf_token: bool,
    revision: Option<&str>,
) -> Result<Vec<usize>, Error>
where
    I: IntoIterator<Item = T>,
    T: AsRef<str>,
{
    let tokenizer = load_tokenizer(id, revision, hf_token, requires_hf_token)?;
    texts
        .into_iter()
        .map(|text| {
            tokenizer
                .encode(text.as_ref(), false)
                .map(|e| e.len())
                .map_err(|e| Error::TokenizerError(e.to_string()))
        })
        .collect()
}

pub fn count_tokens_for_text(
    model: &EmbeddingModelDetails,
    text: &str,
    hf_token: Option<&str>,
) -> Result<usize, Error> {
    Ok(count_tokens(
        &model.tokenizer_id,
        [text],
        hf_token,
        model.requires_hf_token,
    )?[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // A word-level tokenizer whose post-processor would prepend [CLS] if special
    // tokens were added, so the count shows they are not.
    fn tokenizer() -> Arc<Tokenizer> {
        Arc::new(Tokenizer::from_str(r#"{
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "decoder": null,
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [0], "tokens": ["[CLS]"]}}},
            "model": {"type": "WordLevel", "vocab": {"[CLS]": 0, "a": 1, "b": 2, "[UNK]": 3}, "unk_token": "[UNK]"}
        }"#).unwrap())
    }

    /// A failed download says where tokenizers come from and what to set
    /// without network access. Other failures are reported plainly.
    #[test]
    fn only_download_failures_get_network_advice() {
        let download = load_error(
            "voyageai/voyage-4",
            "main",
            "request error: https://huggingface.co/voyageai/voyage-4/resolve/main/tokenizer.json: Dns Failed: resolve dns name 'huggingface.co:443'",
        )
        .to_string();
        for expected in [
            "voyageai/voyage-4@main",
            "huggingface.co",
            "HF_ENDPOINT",
            "HF_HOME",
        ] {
            assert!(download.contains(expected), "{expected}: {download}");
        }
        assert!(
            load_error(
                "voyageai/voyage-4",
                "main",
                "Too many retries: request error: x"
            )
            .to_string()
            .contains("HF_ENDPOINT")
        );

        // Rejected before any network request: no network advice.
        let Err(Error::TokenizerError(message)) =
            load_tokenizer("not a repository!", Some("main"), None, false)
        else {
            panic!("expected a tokenizer error");
        };
        assert!(message.contains("not a repository!@main"), "{message}");
        assert!(!message.contains("HF_ENDPOINT"), "{message}");
        let corrupt = load_error(
            "voyageai/voyage-4",
            "main",
            "expected value at line 1 column 1",
        )
        .to_string();
        assert!(!corrupt.contains("HF_ENDPOINT"), "{corrupt}");

        // A revision the repository doesn't have: the Hub answers 404.
        let missing = load_error(
            "voyageai/voyage-4-lite",
            "44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9",
            "request error: https://huggingface.co/voyageai/voyage-4-lite/resolve/44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9/tokenizer.json: status code 404",
        )
        .to_string();
        assert!(
            missing.contains("voyageai/voyage-4-lite@44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9")
                && missing.contains("no tokenizer at this revision")
                && !missing.contains("HF_ENDPOINT"),
            "{missing}"
        );
    }

    #[test]
    fn a_missing_home_directory_is_an_error_not_a_panic() {
        let home = std::path::Path::new("/home/user");
        let hf_home = std::ffi::OsStr::new("/cache");
        assert!(cache_location_error(None, Some(home)).is_none());
        assert!(cache_location_error(Some(hf_home), None).is_none());
        let error = cache_location_error(None, None).unwrap();
        assert!(error.to_string().contains("HF_HOME"), "{error}");

        let error =
            without_panics(|| -> () { panic!("Cache directory cannot be found") }).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Cache directory cannot be found"),
            "{error}"
        );
        assert_eq!(without_panics(|| 7).unwrap(), 7);
    }

    #[test]
    fn token_counter_counts_raw_tokens_without_special_tokens() {
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.encode("a b", true).unwrap().len(), 3);
        let counter = TokenCounter::new(tokenizer);
        assert_eq!(counter.count("a b").unwrap(), 2);
        assert_eq!(counter.count("a b a unknown").unwrap(), 4);
        assert_eq!(counter.clone().count("").unwrap(), 0);
    }
}
