use crate::{embedding_model::EmbeddingModelDetails, error::Error};
use std::{collections::HashMap, sync::{Arc, Mutex, OnceLock}};
use tokenizers::{Tokenizer, FromPretrainedParameters};

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
        self.tokenizer.encode(text, false)
            .map(|e| e.len()).map_err(|e| Error::TokenizerError(e.to_string()))
    }
}

impl std::fmt::Debug for TokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCounter").finish_non_exhaustive()
    }
}

/// Loads one reusable, untruncated tokenizer per repository/revision. Failed downloads
/// remain retryable; concurrent loads of the same revision share one initialization.
/// Call from a blocking worker when the tokenizer is not already cached.
pub(crate) fn load_tokenizer(
    id: &str, revision: Option<&str>, hf_token: Option<&str>, requires_hf_token: bool,
) -> Result<Arc<Tokenizer>, Error> {
    if requires_hf_token && hf_token.is_none() {
        return Err(Error::MissingHuggingFaceToken(id.into()));
    }
    let revision = revision.unwrap_or("main");
    let slot = {
        let mut cache = TOKENIZERS.get_or_init(Default::default).lock()
            .map_err(|_| Error::TokenizerError("tokenizer cache lock poisoned".into()))?;
        cache.entry((id.into(), revision.into())).or_default().clone()
    };
    let mut loaded = slot.lock()
        .map_err(|_| Error::TokenizerError("tokenizer initialization lock poisoned".into()))?;
    if let Some(tokenizer) = &*loaded { return Ok(tokenizer.clone()); }
    let mut tokenizer = Tokenizer::from_pretrained(id, Some(FromPretrainedParameters {
        revision: revision.into(), token: hf_token.map(str::to_owned), ..Default::default()
    })).map_err(|e| Error::TokenizerError(format!("Failed to load {id}@{revision}: {e}")))?;
    tokenizer.with_truncation(None).map_err(|e| Error::TokenizerError(e.to_string()))?;
    tokenizer.with_padding(None);
    let tokenizer = Arc::new(tokenizer);
    *loaded = Some(tokenizer.clone());
    Ok(tokenizer)
}

/// Counts raw text tokens without padding, truncation, or special tokens.
pub fn count_tokens<I, T>(id: &str, texts: I, hf_token: Option<&str>, requires_hf_token: bool)
    -> Result<Vec<usize>, Error>
where I: IntoIterator<Item=T>, T: AsRef<str> {
    count_tokens_with_revision(id, texts, hf_token, requires_hf_token, None)
}

pub fn count_tokens_with_revision<I, T>(
    id: &str, texts: I, hf_token: Option<&str>, requires_hf_token: bool, revision: Option<&str>,
) -> Result<Vec<usize>, Error>
where I: IntoIterator<Item=T>, T: AsRef<str> {
    let tokenizer = load_tokenizer(id, revision, hf_token, requires_hf_token)?;
    texts.into_iter().map(|text| tokenizer.encode(text.as_ref(), false)
        .map(|e| e.len()).map_err(|e| Error::TokenizerError(e.to_string()))).collect()
}

pub fn count_tokens_for_text(model: &EmbeddingModelDetails, text: &str, hf_token: Option<&str>)
    -> Result<usize, Error> {
    Ok(count_tokens(&model.tokenizer_id, [text], hf_token, model.requires_hf_token)?[0])
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
