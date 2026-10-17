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

/// Default embeddings API base URL. Requests go to `{base}/embeddings`.
const VOYAGE_BASE_URL: &str = "https://api.voyageai.com/v1";

#[derive(Clone)]
pub struct VoyageClient {
    api_key: String,
    http_client: reqwest::Client,
    base_url: String,
}

impl VoyageClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http_client: super::default_http_client(),
            base_url: VOYAGE_BASE_URL.to_string(),
        }
    }

    /// Sends requests to `{base_url}/embeddings` instead of
    /// `https://api.voyageai.com/v1/embeddings`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
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
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));

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
            input_type: config
                .voyage_input_type
                .as_ref()
                .map(VoyageInputType::as_str),
            truncation: config.voyage_truncation.or(Some(false)), // Default to false if not specified
            output_dimension: config.output_dimensionality,
        };

        let response = self
            .http_client
            .post(url)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_payload)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(voyage_error(
                Error::from_response("Voyage AI", response).await,
            ));
        }

        // A body cut off in transfer stays a transport error; one that isn't
        // the documented object is a malformed response.
        let body = response.bytes().await?;
        let response_data: api::VoyageEmbeddingResponse =
            serde_json::from_slice(&body).map_err(malformed)?;

        let embeddings = ordered_embeddings(
            response_data.data,
            texts.len(),
            config
                .output_dimensionality
                .unwrap_or(model_details.default_dimensions),
        )?;

        let usage = UsageInfo::for_embedding(
            response_data.usage.total_tokens,
            model_details.cost_per_million_input_tokens,
        );

        Ok(ArkoEmbeddingResponse { embeddings, usage })
    }
}

/// Voyage's errors classify by HTTP status as Voyage documents them
/// (https://docs.voyageai.com/docs/error-codes): 400, 401 and 403 are
/// permanent, 429 is a rate limit and 500 a transient server error. 502, 503
/// and 504 are "Service Unavailable": "Our servers are currently
/// experiencing high traffic", an explicit capacity signal, so they take the
/// `service_unavailable` code, which classifies as
/// [`RetryClass::Capacity`](crate::RetryClass::Capacity). The body (for 4XX,
/// `{"detail": "<message>"}`) stays in the message.
///
/// The 400s that name a limit take the kinds arko's own limit checks give the
/// same requests, as Voyage answered them (2026-10-04). A request over the
/// model's total tokens per request carries an `error_code`
/// (`TOO_MANY_TOKENS_IN_BATCH`, undocumented), which becomes the error's
/// code: [`ErrorKind::ContextOverflow`](crate::ErrorKind::ContextOverflow).
/// The others carry only a `detail`: an input over the model's context
/// window ("...does not fit into the model's context window of 32000
/// tokens...") is `ContextOverflow`, and a batch over the item limit ("The
/// batch size limit is 1000...") is
/// [`ErrorKind::RequestTooLarge`](crate::ErrorKind::RequestTooLarge).
fn voyage_error(mut error: Error) -> Error {
    if let Error::ProviderError {
        status: Some(status),
        code,
        message,
        ..
    } = &mut error
        && code.is_none()
    {
        *code = error_code(message).or(match *status {
            502..=504 => Some("service_unavailable".into()),
            400 if message.contains("does not fit into the model's context window") => {
                Some("context_length_exceeded".into())
            }
            400 if message.contains("The batch size limit is") => Some("request_too_large".into()),
            _ => None,
        });
    }
    error
}

/// The `error_code` of a Voyage error body (`Voyage AI: {...}` in the
/// message), when it has one.
fn error_code(message: &str) -> Option<String> {
    let body = message.strip_prefix("Voyage AI: ")?;
    serde_json::from_str::<Value>(body)
        .ok()?
        .get("error_code")?
        .as_str()
        .map(str::to_owned)
}

/// A successful response that isn't the documented embeddings object. Voyage
/// answered, and may have billed, the request, so this is not the caller's
/// invalid request, and it is permanent rather than resent automatically: an
/// [`Error::JsonError`] of kind [`ErrorKind::Unknown`](crate::ErrorKind::Unknown).
fn malformed(detail: impl std::fmt::Display) -> Error {
    Error::JsonError(<serde_json::Error as serde::de::Error>::custom(format!(
        "Malformed Voyage AI embeddings response: {detail}"
    )))
}

fn ordered_embeddings(
    data: Vec<api::VoyageEmbeddingObject>,
    count: usize,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>, Error> {
    let invalid = || malformed("missing/duplicate index or invalid vector");
    if data.len() != count {
        return Err(invalid());
    }
    let mut ordered = vec![None; count];
    for item in data {
        if item.index >= count
            || ordered[item.index].is_some()
            || item.embedding.len() != dimensions
            || item.embedding.iter().any(|v| !v.is_finite())
            || !item.embedding.iter().any(|&v| v != 0.0)
        {
            return Err(invalid());
        }
        ordered[item.index] = Some(item.embedding);
    }
    ordered.into_iter().map(|v| v.ok_or_else(invalid)).collect()
}
#[cfg(test)]
mod mock_server_tests;
#[cfg(test)]
mod tests {
    use super::*;
    fn item(index: usize, embedding: Vec<f32>) -> api::VoyageEmbeddingObject {
        api::VoyageEmbeddingObject { index, embedding }
    }
    #[test]
    fn response_indexes_restore_input_order() {
        assert_eq!(
            ordered_embeddings(vec![item(1, vec![2.0]), item(0, vec![1.0])], 2, 1).unwrap(),
            vec![vec![1.0], vec![2.0]]
        );
    }
    #[test]
    fn malformed_responses_are_rejected() {
        for data in [
            vec![],
            vec![item(2, vec![1.0])],
            vec![item(0, vec![0.0])],
            vec![item(0, vec![f32::NAN])],
            vec![item(0, vec![1.0, 2.0])],
        ] {
            assert!(ordered_embeddings(data, 1, 1).is_err());
        }
        assert!(ordered_embeddings(vec![item(0, vec![1.0]), item(0, vec![2.0])], 2, 1).is_err());
    }
}
