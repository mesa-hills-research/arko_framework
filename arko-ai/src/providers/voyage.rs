// src/providers/voyage.rs
use crate::{
    embedding_config::{EmbeddingConfig, VoyageInputType},
    embedding_model::EmbeddingModelDetails,
    error::Error,
    providers::{EmbeddingProviderTrait, EmbeddingResponse as ArkoEmbeddingResponse},
    usage::UsageInfo,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value; // For dynamic input field

#[derive(Clone)]
pub struct VoyageClient {
    api_key: String,
    http_client: reqwest::Client,
}

impl VoyageClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http_client: reqwest::Client::new(),
        }
    }
}

// Voyage API specific structs
mod api {
    use super::*;
    #[derive(Serialize, Debug)]
    pub struct VoyageEmbeddingRequest<'a> {
        pub input: Value, // Can be String or Vec<String>, handled by serde_json::to_value
        pub model: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input_type: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub truncation: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub output_dimension: Option<usize>,
        // output_dtype and encoding_format default to float and null (array) respectively
    }

    #[derive(Deserialize, Debug)]
    pub struct VoyageEmbeddingResponse {
        pub data: Vec<VoyageEmbeddingObject>,
        // model: String, // We already know the model
        pub usage: VoyageUsage,
    }

    #[derive(Deserialize, Debug)]
    pub struct VoyageEmbeddingObject {
        pub embedding: Vec<f32>,
        pub index: usize,
        // object: String,
    }

    #[derive(Deserialize, Debug)]
    pub struct VoyageUsage {
        pub total_tokens: usize,
    }
}

#[async_trait]
impl EmbeddingProviderTrait for VoyageClient {
    async fn embed_texts(
        &self,
        texts: &[String],
        config: &EmbeddingConfig,
        model_details: &EmbeddingModelDetails,
    ) -> Result<ArkoEmbeddingResponse, Error> {
        let url = "https://api.voyageai.com/v1/embeddings";

        // Voyage API takes a single string or an array of strings.
        // If texts has one element, pass it as a string, otherwise as an array.
        let input_value = if texts.len() == 1 {
            serde_json::to_value(&texts[0])?
        } else {
            serde_json::to_value(texts)?
        };
        
        let request_payload = api::VoyageEmbeddingRequest {
            input: input_value,
            model: &model_details.api_id,
            input_type: config.voyage_input_type.as_ref().map(VoyageInputType::as_str),
            truncation: config.voyage_truncation.or(Some(false)), // Default to false if not specified
            output_dimension: config.output_dimensionality,
        };

        let response = self.http_client
            .post(url)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_payload)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::from_response("Voyage AI", response).await);
        }

        let response_data: api::VoyageEmbeddingResponse = response.json().await?;
        
        let embeddings = ordered_embeddings(response_data.data, texts.len(),
            config.output_dimensionality.unwrap_or(model_details.default_dimensions))?;

        let usage = UsageInfo::for_embedding(
            response_data.usage.total_tokens,
            model_details.cost_per_million_input_tokens,
        );

        Ok(ArkoEmbeddingResponse { embeddings, usage })
    }
}
fn ordered_embeddings(data: Vec<api::VoyageEmbeddingObject>, count: usize, dimensions: usize)
    -> Result<Vec<Vec<f32>>, Error> {
    let invalid = || Error::InvalidRequest("Malformed Voyage response: missing/duplicate index or invalid vector".into());
    if data.len() != count { return Err(invalid()); }
    let mut ordered = vec![None; count];
    for item in data {
        if item.index >= count || ordered[item.index].is_some() || item.embedding.len() != dimensions
            || item.embedding.iter().any(|v| !v.is_finite())
            || !item.embedding.iter().any(|&v| v != 0.0) { return Err(invalid()); }
        ordered[item.index] = Some(item.embedding);
    }
    ordered.into_iter().map(|v| v.ok_or_else(invalid)).collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn item(index: usize, embedding: Vec<f32>) -> api::VoyageEmbeddingObject { api::VoyageEmbeddingObject { index, embedding } }
    #[test]
    fn response_indexes_restore_input_order() {
        assert_eq!(ordered_embeddings(vec![item(1,vec![2.0]),item(0,vec![1.0])],2,1).unwrap(),vec![vec![1.0],vec![2.0]]);
    }
    #[test]
    fn malformed_responses_are_rejected() {
        for data in [vec![],vec![item(2,vec![1.0])],vec![item(0,vec![0.0])],vec![item(0,vec![f32::NAN])],vec![item(0,vec![1.0,2.0])]] {
            assert!(ordered_embeddings(data,1,1).is_err());
        }
        assert!(ordered_embeddings(vec![item(0,vec![1.0]),item(0,vec![2.0])],2,1).is_err());
    }
}
