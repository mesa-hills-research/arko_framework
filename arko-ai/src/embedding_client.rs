use crate::{
    embedding_config::EmbeddingConfig,
    embedding_model::EmbeddingModelDetails,
    embedding_provider_identifier::EmbeddingProviderIdentifier,
    error::Error,
    providers::{EmbeddingProviderTrait, EmbeddingResponse, voyage::VoyageClient},
    token_utils,
};

/// Client for generating text embeddings.
///
/// Voyage AI is the supported embeddings provider.
pub struct EmbeddingClient {
    voyage_provider: Option<VoyageClient>,
}

impl EmbeddingClient {
    pub fn new() -> Self {
        Self {
            voyage_provider: None,
        }
    }

    pub fn with_voyage_api_key(mut self, key: &str) -> Self {
        self.voyage_provider = Some(VoyageClient::new(key));
        self
    }

    /// Installs a preconfigured Voyage client (for example one with
    /// [`VoyageClient::with_base_url`]).
    pub fn with_voyage_client(mut self, client: VoyageClient) -> Self {
        self.voyage_provider = Some(client);
        self
    }

    async fn get_model_details(
        &self,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingModelDetails, Error> {
        EmbeddingModelDetails::get(config.provider, &config.model_id)
    }

    /// Embeds a list of texts using the specified configuration.
    pub async fn embed(
        &self,
        texts: Vec<String>,
        config: &EmbeddingConfig,
    ) -> Result<EmbeddingResponse, Error> {
        if texts.is_empty() {
            return Err(Error::InvalidRequest("Input texts cannot be empty.".to_string()));
        }

        let model_details = self.get_model_details(config).await?;

        if let Some(dim) = config.output_dimensionality
            && !model_details.supported_dimensions.contains(&dim) {
                return Err(Error::InvalidDimension(format!(
                    "Dimension {} not supported by model {}. Supported: {:?}.",
                    dim, model_details.id, model_details.supported_dimensions
                )));
            }

        let tokenizer_id = model_details.tokenizer_id.clone();
        let hf_token = config.hf_token.clone();
        let revision = config.tokenizer_revision.clone();
        let requires_hf_token = model_details.requires_hf_token;
        let (texts, counts) = tokio::task::spawn_blocking(move || {
            let counts = token_utils::count_tokens_with_revision(
                &tokenizer_id, texts.iter(), hf_token.as_deref(), requires_hf_token,
                revision.as_deref(),
            )?;
            Ok::<_, Error>((texts, counts))
        }).await.map_err(|e| Error::TokenizerError(format!("tokenizer worker failed: {e}")))??;
        let mut total_tokens_for_batch = 0;
        for (i, &token_count) in counts.iter().enumerate() {
            if token_count > model_details.max_input_tokens_per_item {
                return Err(Error::TokenLimitExceeded(format!(
                    "Text at index {} ({} tokens) exceeds model's per-item limit of {}.",
                    i, token_count, model_details.max_input_tokens_per_item
                )));
            }
            total_tokens_for_batch += token_count;
        }

        if let Some(max_batch_tokens) = model_details.max_batch_total_tokens
            && total_tokens_for_batch > max_batch_tokens {
                return Err(Error::TokenLimitExceeded(format!(
                    "Total tokens in batch ({} tokens) exceed model's batch limit of {}.",
                    total_tokens_for_batch, max_batch_tokens
                )));
            }
        if let Some(max_batch_size) = model_details.max_batch_size
            && texts.len() > max_batch_size {
                return Err(Error::BatchSizeLimitExceeded(format!(
                    "Number of texts in batch ({}) exceeds model's batch size limit of {}.",
                    texts.len(), max_batch_size
                )));
            }

        match config.provider {
            EmbeddingProviderIdentifier::Voyage => {
                let provider = self
                    .voyage_provider
                    .as_ref()
                    .ok_or_else(|| Error::MissingApiKey("Voyage AI".to_string()))?;
                provider.embed_texts(&texts, config, &model_details).await
            }
        }
    }

    /// Loads the model's tokenizer on a blocking worker, honoring the configured revision
    /// and Hugging Face token. The counter shares the tokenizer [`embed`](Self::embed)
    /// uses for the same configuration, so repeated counts need no further loading.
    pub async fn token_counter(&self, config: &EmbeddingConfig) -> Result<token_utils::TokenCounter, Error> {
        let model_details = self.get_model_details(config).await?;
        let hf_token = config.hf_token.clone();
        let revision = config.tokenizer_revision.clone();
        let tokenizer = tokio::task::spawn_blocking(move || {
            token_utils::load_tokenizer(
                &model_details.tokenizer_id, revision.as_deref(), hf_token.as_deref(),
                model_details.requires_hf_token,
            )
        }).await.map_err(|e| Error::TokenizerError(format!("tokenizer worker failed: {e}")))??;
        Ok(token_utils::TokenCounter::new(tokenizer))
    }

    /// Counts tokens for a single text according to the specified model configuration.
    /// Uses the batch-aware tokenizer helper so tokenizer initialization can be shared if callers
    /// move to batch-counting. For a single text this is functionally equivalent but avoids
    /// duplicating two different tokenization code paths.
    pub fn count_tokens(
        &self,
        text: &str,
        config: &EmbeddingConfig,
    ) -> Result<usize, Error> {
        let model_details = EmbeddingModelDetails::get(config.provider, &config.model_id)?;
        let counts = token_utils::count_tokens_with_revision(
            &model_details.tokenizer_id,
            std::iter::once(text),
            config.hf_token.as_deref(),
            model_details.requires_hf_token,
            config.tokenizer_revision.as_deref(),
        )?;
        Ok(counts.into_iter().next().unwrap_or(0))
    }

    /// Calculates the estimated cost for embedding a list of texts.
    /// This involves tokenizing all texts but does not make API calls to the embedding provider.
    pub fn get_embedding_cost(
        &self,
        texts: &[String],
        config: &EmbeddingConfig,
    ) -> Result<f64, Error> {
        if texts.is_empty() {
            return Ok(0.0);
        }
        let model_details = EmbeddingModelDetails::get(config.provider, &config.model_id)?;

        if model_details.cost_per_million_input_tokens == 0.0 {
            return Ok(0.0);
        }

        let counts = token_utils::count_tokens_with_revision(
            &model_details.tokenizer_id,
            texts.iter(),
            config.hf_token.as_deref(),
            model_details.requires_hf_token,
            config.tokenizer_revision.as_deref(),
        )?;
        let total_tokens: usize = counts.iter().sum();

        let cost = (total_tokens as f64 / 1_000_000.0) * model_details.cost_per_million_input_tokens;
        Ok(cost)
    }

    /// Calculates the estimated cost for embedding based on a pre-calculated total number of tokens.
    /// This is useful when tokens have been summed up from multiple sources before estimating cost.
    pub fn get_embedding_cost_for_token_count(
        &self,
        total_tokens: usize,
        config: &EmbeddingConfig,
    ) -> Result<f64, Error> {
        if total_tokens == 0 {
            return Ok(0.0);
        }
        let model_details = EmbeddingModelDetails::get(config.provider, &config.model_id)?;

        if model_details.cost_per_million_input_tokens == 0.0 {
            return Ok(0.0);
        }

        let cost = (total_tokens as f64 / 1_000_000.0) * model_details.cost_per_million_input_tokens;
        Ok(cost)
    }
}

impl Default for EmbeddingClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "downloads a Hugging Face tokenizer"]
    async fn token_counter_matches_batch_counting_for_the_same_revision() {
        let revision = "44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9";
        let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-4")
            .with_tokenizer_revision(revision);
        let counter = EmbeddingClient::new().token_counter(&config).await.unwrap();
        let text = "fn main() { println!(\"Grüße, 世界\"); }";
        let expected = token_utils::count_tokens_with_revision(
            "voyageai/voyage-4", [text], None, false, Some(revision),
        ).unwrap()[0];
        assert!(expected > 0);
        assert_eq!(counter.count(text).unwrap(), expected);
    }
}