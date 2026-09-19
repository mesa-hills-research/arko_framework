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
use serde::{Deserialize, Serialize};
use std::time::Duration;

// No total request deadline: a connected generation may take arbitrarily long.
// Keep connection establishment and idle connection-pool cleanup bounded.
pub(crate) fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
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
            let(mut socket,_)=listener.accept().await.unwrap();let mut bytes=[0;4096];socket.read(&mut bytes).await.unwrap();
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
}
