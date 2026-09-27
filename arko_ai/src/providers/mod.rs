pub mod anthropic;
pub mod google;
pub mod openai;
pub mod openrouter;
pub mod voyage;

use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    message::{Content, Message},
    stop::{Refusal, StopReason},
    tool::ToolCall,
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

/// A [`reqwest::ClientBuilder`] with the setup this crate's providers use:
/// rustls with the *ring* crypto provider passed explicitly (nothing is
/// installed process-wide), the operating system's trust store, HTTP/1.1, a
/// 30-second connect timeout and no total request deadline. Use it, or
/// [`http_client`], to build a client passed to this crate, such as to
/// [`refresh_chatgpt_oauth_token`](openai::refresh_chatgpt_oauth_token).
///
/// This crate builds reqwest without a default crypto provider, so
/// `reqwest::Client::new()` and `reqwest::Client::builder().build()` panic
/// unless the application installs a process-wide rustls `CryptoProvider` or
/// enables one through reqwest's own features. The TLS setup here is complete:
/// reqwest's TLS options (extra root certificates, TLS versions, disabling
/// certificate verification) do not apply to this builder.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    // No total request deadline: a connected generation may take arbitrarily
    // long. Keep connection establishment and idle connection-pool cleanup
    // bounded.
    reqwest::Client::builder()
        .tls_backend_preconfigured(tls_config())
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
}

/// A client built from [`http_client_builder`], as this crate's providers use.
pub fn http_client() -> reqwest::Client {
    http_client_builder()
        .build()
        .expect("Failed to construct the arko_ai HTTP client")
}

pub(crate) fn default_http_client() -> reqwest::Client {
    http_client()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub message: Message,
    pub usage: UsageInfo,
    pub model_call_usage: ModelCallUsage,
    /// Why generation stopped, normalized across providers.
    #[serde(default = "StopReason::unknown")]
    pub stop_reason: StopReason,
    /// Present when the model or a filter refused. For refusals whose
    /// source doesn't keep the turn ([`RefusalSource::keeps_turn`]), `message`
    /// is an empty assistant message with no tool calls or reasoning.
    ///
    /// [`RefusalSource::keeps_turn`]: crate::stop::RefusalSource::keeps_turn
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Refusal>,
    /// Sticky-routing token the ChatGPT backend returned for this call in
    /// the `x-codex-turn-state` response header.
    /// [`Conversation::add_assistant_message_from_response`] keeps it so the
    /// rest of the turn echoes it; see [`Conversation::openai_turn_state`].
    ///
    /// [`Conversation::add_assistant_message_from_response`]: crate::Conversation::add_assistant_message_from_response
    /// [`Conversation::openai_turn_state`]: crate::Conversation::openai_turn_state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_turn_state: Option<String>,
}

impl GenerationResponse {
    /// The model the provider reports having served; see
    /// [`ModelCallUsage::served_model`].
    pub fn served_model(&self) -> Option<&str> {
        self.model_call_usage.served_model.as_deref()
    }

    /// Whether the provider served a different model than the one
    /// requested; see [`ModelCallUsage::served_model_changed`].
    pub fn rerouted(&self) -> bool {
        self.model_call_usage.served_model_changed()
    }

    /// The concatenated text content of the assistant message.
    pub fn text(&self) -> String {
        self.message
            .content
            .iter()
            .filter_map(Content::as_text)
            .collect()
    }

    /// The single call to `name` in a response that was forced to call it.
    ///
    /// Fails with [`Error::Refusal`] for refusals, with
    /// [`Error::UnexpectedToolCalls`] when the response has no call, several
    /// calls, or a different tool (its `stop_reason` shows whether a token
    /// limit cut the call off), and with [`Error::InvalidToolArguments`] when
    /// the call's arguments are not valid JSON.
    pub fn single_tool_call(&self, name: &str) -> Result<&ToolCall, Error> {
        if let Some(refusal) = &self.refusal {
            return Err(Error::Refusal(Box::new(refusal.clone())));
        }
        match self.message.tool_calls.as_slice() {
            [call] if call.name == name => match &call.invalid_arguments {
                Some(raw) => Err(Error::InvalidToolArguments {
                    name: call.name.clone(),
                    raw: raw.clone(),
                }),
                None => Ok(call),
            },
            calls => Err(Error::UnexpectedToolCalls {
                expected: name.to_string(),
                stop_reason: self.stop_reason.clone(),
                found: calls.iter().map(|call| call.name.clone()).collect(),
            }),
        }
    }
}

/// Rejects content the model can't take before anything is sent: images for
/// models without image input, and mid-conversation configuration updates for
/// providers that can't send them.
pub(crate) fn ensure_supported_inputs(conversation: &Conversation) -> Result<(), Error> {
    if conversation.model.provider != crate::model::Provider::OpenRouter
        && conversation.messages.iter().any(Message::is_configuration_update)
    {
        return Err(Error::UnsupportedFeature(format!(
            "Mid-conversation reasoning-effort updates are supported only for OpenRouter models, not {:?} model '{}'. Remove the update or change the request-level reasoning configuration.",
            conversation.model.provider, conversation.model.id
        )));
    }
    let has_images = conversation
        .messages
        .iter()
        .any(|message| message.content.iter().any(|content| content.as_image().is_some()));
    if has_images && !conversation.model.supports_image_input {
        return Err(Error::UnsupportedFeature(format!(
            "Model '{}' does not accept image input.",
            conversation.model.id
        )));
    }
    Ok(())
}

/// Removes a refused turn's output from `message` so it can't be replayed,
/// returning any text generated before the refusal.
pub(crate) fn take_refused_output(message: &mut Message) -> Option<String> {
    let text: String = message
        .content
        .iter()
        .filter_map(Content::as_text)
        .collect();
    message.content = vec![Content::Text(String::new())];
    message.tool_calls.clear();
    message.reasoning_artifact = None;
    (!text.is_empty()).then_some(text)
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    ReasoningChunk(String),
    TextChunk(String),
    /// Emitted once, just before [`End`](Self::End), when the response was
    /// refused. Text already streamed for a refusal whose source doesn't
    /// keep the turn should be discarded by the consumer.
    Refusal(Refusal),
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

/// A client for one LLM provider's API. Named `ProviderClient` to keep it
/// distinct from the [`Provider`](crate::model::Provider) enum that identifies
/// a model's provider.
#[async_trait::async_trait]
pub trait ProviderClient {
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
mod response_tests {
    use super::*;
    use crate::stop::RefusalSource;

    fn response(calls: &[&str], stop_reason: StopReason, refusal: Option<Refusal>) -> GenerationResponse {
        let mut message = Message::new_assistant("text");
        for name in calls {
            message.tool_calls.push(ToolCall::from_argument_text(None, name.to_string(), "{}"));
        }
        GenerationResponse {
            message,
            usage: UsageInfo::new(0, 0),
            model_call_usage: ModelCallUsage::new("id".into(), "test".into(), "test".into(), 0, 0, 0),
            stop_reason,
            refusal,
            openai_turn_state: None,
        }
    }

    #[test]
    fn single_tool_call_distinguishes_refusals_limits_and_wrong_calls() {
        assert!(response(&["pick"], StopReason::ToolUse, None).single_tool_call("pick").is_ok());
        for (calls, stop) in [
            (vec![], StopReason::EndTurn),
            (vec!["other"], StopReason::ToolUse),
            (vec!["pick", "pick"], StopReason::ToolUse),
        ] {
            let error = response(&calls, stop, None).single_tool_call("pick").unwrap_err();
            assert!(matches!(error, Error::UnexpectedToolCalls { .. }));
            assert!(error.is_retryable());
        }
        let error = response(&[], StopReason::MaxTokens, None).single_tool_call("pick").unwrap_err();
        assert!(!error.is_retryable(), "a token limit needs a configuration change");
        let refused = response(&["pick"], StopReason::Refusal, Some(Refusal::new(RefusalSource::Model)));
        let error = refused.single_tool_call("pick").unwrap_err();
        assert!(matches!(error, Error::Refusal(_)));
        assert!(!error.is_retryable());
        assert_eq!(refused.text(), "text");
    }

    #[test]
    fn responses_saved_before_stop_reasons_still_load() {
        let mut value = serde_json::to_value(response(&[], StopReason::EndTurn, None)).unwrap();
        value.as_object_mut().unwrap().remove("stop_reason");
        let loaded: GenerationResponse = serde_json::from_value(value).unwrap();
        assert_eq!(loaded.stop_reason, StopReason::Other("unknown".into()));
        assert!(loaded.refusal.is_none());
    }
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
    // Network checks for the TLS setup: `cargo test -p arko_ai -- --ignored tls_`.
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
        // The failure must be certificate validation, not DNS or a reset.
        let mut source:Option<&(dyn std::error::Error+'static)>=Some(&error);
        let mut expired=false;
        while let Some(current)=source {
            if matches!(current.downcast_ref::<rustls::Error>(),Some(rustls::Error::InvalidCertificate(rustls::CertificateError::Expired|rustls::CertificateError::ExpiredContext{..}))) {expired=true;break;}
            // io::Error::source() skips its payload (the wrapped rustls error); descend into it instead.
            source=match current.downcast_ref::<std::io::Error>().and_then(std::io::Error::get_ref) {
                Some(inner)=>Some(inner as &(dyn std::error::Error+'static)),
                None=>current.source(),
            };
        }
        assert!(expired,"expected an expired-certificate error in the source chain: {error:?}");
    }
}
