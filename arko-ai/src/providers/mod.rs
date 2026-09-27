pub mod anthropic;
pub mod google;
pub mod openai;
pub mod voyage;

use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::Message,
    usage::{ModelCallUsage, UsageInfo},
};
use futures::stream::BoxStream;
use rustls_platform_verifier::BuilderVerifierExt;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

// rustls with the ring provider, passed explicitly so nothing process-wide is
// installed, and the operating system's trust store. HTTP/1.1 only, as before.
fn tls_config() -> rustls::ClientConfig {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let versions = || {
        rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .expect("ring supports rustls' default protocol versions")
    };
    let mut config = match versions().with_platform_verifier() {
        Ok(builder) => builder.with_no_client_auth(),
        // Keep client construction infallible, as with the previous native
        // TLS backend: HTTPS requests then fail certificate verification.
        Err(error) => {
            log::error!("arko_ai: system CA certificates are unavailable ({error}); HTTPS requests will fail");
            versions()
                .with_root_certificates(rustls::RootCertStore::empty())
                .with_no_client_auth()
        }
    };
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    config
}

// No total request deadline: a connected generation may take arbitrarily long.
// Keep connection establishment and idle connection-pool cleanup bounded.
pub(crate) fn http_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .tls_backend_preconfigured(tls_config())
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
}

pub(crate) fn default_http_client() -> reqwest::Client {
    http_client_builder()
        .build()
        .expect("Failed to construct the arko_ai HTTP client")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub message: Message,
    pub usage: UsageInfo,
    pub model_call_usage: ModelCallUsage,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    ReasoningChunk(String),
    TextChunk(String),
    ToolCallStart {
        index: usize,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        delta: String,
    },
    End(Box<GenerationResponse>),
}

#[async_trait::async_trait]
pub trait Provider {
    async fn generate_text(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, Error>;

    async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error>;
}

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub usage: UsageInfo,
}

#[async_trait::async_trait]
pub trait EmbeddingProviderTrait {
    async fn embed_texts(
        &self,
        texts: &[String],
        config: &crate::embedding_config::EmbeddingConfig,
        model_details: &crate::embedding_model::EmbeddingModelDetails,
    ) -> Result<EmbeddingResponse, Error>;
}
#[cfg(test)]
mod transport_tests {
    use super::*;
    use tokio::io::{AsyncReadExt,AsyncWriteExt};
    #[tokio::test]
    async fn connected_response_can_outlive_thirty_minutes() {
        let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address=listener.local_addr().unwrap();
        let (release,ready)=tokio::sync::oneshot::channel();
        let server=tokio::spawn(async move {
            let(mut socket,_)=listener.accept().await.unwrap();let mut bytes=[0;4096];assert!(socket.read(&mut bytes).await.unwrap()>0);
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n").await.unwrap();
            ready.await.unwrap();socket.write_all(b"ok").await.unwrap();
        });
        let response=default_http_client().get(format!("http://{address}/")).send().await.unwrap();
        tokio::time::pause();
        let body=tokio::spawn(response.text());tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(1_801)).await;tokio::task::yield_now().await;
        assert!(!body.is_finished(),"connected response was cut off by a total request deadline");
        tokio::time::resume();release.send(()).unwrap();assert_eq!(body.await.unwrap().unwrap(),"ok");server.await.unwrap();
    }
    // Network checks for the TLS setup: `cargo test -p arko-ai -- --ignored tls_`.
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn tls_verifies_public_certificates_over_http_1_1() {
        let response=default_http_client().get("https://api.openai.com/v1/models").send().await.unwrap();
        assert_eq!(response.status(),reqwest::StatusCode::UNAUTHORIZED);
        assert_eq!(response.version(),reqwest::Version::HTTP_11);
    }
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn tls_rejects_an_expired_certificate() {
        let error=default_http_client().get("https://expired.badssl.com/").send().await.unwrap_err();
        assert!(error.is_connect(),"{error:?}");
    }
}
