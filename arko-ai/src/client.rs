use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    model::{Model, Provider},
    providers::{self, GenerationResponse, Provider as ProviderTrait, StreamEvent},
};
use futures::stream::BoxStream;

#[derive(Clone)]
pub struct LLMClient {
    openai_client: Option<providers::openai::OpenAIClient>,
    anthropic_client: Option<providers::anthropic::AnthropicClient>,
    google_client: Option<providers::google::GoogleClient>,
}

impl LLMClient {
    pub fn new() -> Self {
        Self {
            openai_client: None,
            anthropic_client: None,
            google_client: None,
        }
    }

    pub fn with_openai_api_key(mut self, key: &str) -> Self {
        self.openai_client = Some(providers::openai::OpenAIClient::new(key));
        self
    }

    /// Configures the OpenAI provider to use ChatGPT OAuth authentication.
    ///
    /// Requests will be sent to the Codex endpoint at
    /// `chatgpt.com/backend-api/codex/responses` using the provided OAuth
    /// credentials instead of the standard API key endpoint.
    ///
    /// # Token refresh
    ///
    /// Access tokens expire periodically (~1 hour). When a `refresh_token`
    /// is provided, call [`refresh_openai_oauth_token`](Self::refresh_openai_oauth_token)
    /// to obtain a fresh access token. Use [`is_openai_token_expired`](Self::is_openai_token_expired)
    /// to check proactively. The `arko_agents` runtime performs this check
    /// automatically before each LLM call.
    /// If the server rejects an OAuth request with HTTP 401 `token_expired`
    /// before that expiry, the provider refreshes once and resends the request.
    /// Other authentication failures and a second rejection return to the caller.
    pub fn with_openai_chatgpt_oauth(
        self,
        access_token: &str,
        refresh_token: Option<&str>,
        account_id: &str,
    ) -> Self {
        self.with_openai_chatgpt_oauth_rate_limit_retries(
            access_token,
            refresh_token,
            account_id,
            false,
        )
    }

    /// Configures ChatGPT OAuth with an explicit OAuth rate-limit policy.
    ///
    /// When `retry_rate_limits` is true, HTTP 429 responses from the OAuth
    /// token endpoint wait for the advertised reset when available and
    /// otherwise use bounded exponential backoff. Retrying continues until
    /// the refresh succeeds or a different error is returned.
    pub fn with_openai_chatgpt_oauth_rate_limit_retries(
        mut self,
        access_token: &str,
        refresh_token: Option<&str>,
        account_id: &str,
        retry_rate_limits: bool,
    ) -> Self {
        self.openai_client = Some(
            providers::openai::OpenAIClient::new_chatgpt_oauth(
                access_token,
                refresh_token,
                account_id,
            )
            .with_oauth_rate_limit_retries(retry_rate_limits),
        );
        self
    }

    /// Configures the OpenAI provider with a pre-constructed [`OpenAIAuth`]
    /// value, allowing full control over the authentication method.
    ///
    /// [`OpenAIAuth`]: crate::providers::openai::OpenAIAuth
    pub fn with_openai_auth(mut self, auth: providers::openai::OpenAIAuth) -> Self {
        self.openai_client = Some(providers::openai::OpenAIClient::with_auth(auth));
        self
    }

    /// Replaces the OpenAI authentication credentials in-place.
    ///
    /// Because all clones of `LLMClient` share the same internal
    /// credential slot (via `Arc<RwLock<OpenAIAuth>>`), this update is
    /// visible to every clone immediately. Requests already in flight
    /// are unaffected — they snapshot credentials at the start of each
    /// call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingApiKey`] if no OpenAI client has been
    /// configured on this `LLMClient`.
    pub async fn update_openai_auth(
        &self,
        auth: providers::openai::OpenAIAuth,
    ) -> Result<(), Error> {
        match &self.openai_client {
            Some(client) => {
                client.update_auth(auth).await;
                Ok(())
            }
            None => Err(Error::MissingApiKey("OpenAI".to_string())),
        }
    }

    /// Returns `true` if the OpenAI provider is configured with ChatGPT
    /// OAuth authentication.
    ///
    /// Returns `false` if no OpenAI client is configured or if it uses
    /// API key auth.
    pub async fn is_openai_chatgpt_oauth(&self) -> bool {
        match &self.openai_client {
            Some(client) => client.is_chatgpt_oauth().await,
            None => false,
        }
    }

    /// Returns a snapshot of the active ChatGPT OAuth credentials.
    ///
    /// This permits a long-running host to persist rotated refresh tokens
    /// without exposing provider internals. API-key and unconfigured clients
    /// return `None`.
    pub async fn openai_chatgpt_oauth_credentials(
        &self,
    ) -> Option<providers::openai::ChatGptOAuthCredentials> {
        match &self.openai_client {
            Some(client) => client.chatgpt_oauth_credentials().await,
            None => None,
        }
    }

    /// Returns `true` if the current OpenAI OAuth access token is expired
    /// or will expire within the given margin.
    ///
    /// Returns `false` if no OpenAI client is configured, or if it uses
    /// API key auth. Returns `true` if the token cannot be parsed
    /// (conservative — triggers refresh).
    pub async fn is_openai_token_expired(&self, margin: std::time::Duration) -> bool {
        match &self.openai_client {
            Some(client) => client.is_oauth_token_expired(margin).await,
            None => false,
        }
    }

    /// Refreshes the OpenAI ChatGPT OAuth access token using the stored
    /// refresh token and the client's internal HTTP client.
    ///
    /// This method does **not** require the caller to supply a
    /// `reqwest::Client`, making it suitable for crates that do not
    /// depend on reqwest (such as `arko_agents`).
    ///
    /// On success, the new credentials are written to the shared credential
    /// slot. All clones of this `LLMClient` will use the new access token
    /// for subsequent requests.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingApiKey`] if no OpenAI client is configured.
    /// - [`Error::AuthenticationFailed`] if no refresh token is stored.
    /// - [`Error::InvalidRequest`] if the client uses API key auth.
    /// - [`Error::ProviderError`] if the token endpoint returns an error.
    pub async fn refresh_openai_oauth_token(
        &self,
    ) -> Result<providers::openai::OAuthRefreshResult, Error> {
        match &self.openai_client {
            Some(client) => client.refresh_oauth_token().await,
            None => Err(Error::MissingApiKey("OpenAI".to_string())),
        }
    }

    /// Refresh once, bypassing provider-level retries so an agent can enforce
    /// its shared recovery deadline and respond to cancellation.
    pub async fn refresh_openai_oauth_token_once(
        &self,
    ) -> Result<providers::openai::OAuthRefreshResult, Error> {
        match &self.openai_client {
            Some(client) => client.refresh_oauth_token_once().await,
            None => Err(Error::MissingApiKey("OpenAI".to_string())),
        }
    }

    pub fn with_anthropic_api_key(mut self, key: &str) -> Self {
        self.anthropic_client = Some(providers::anthropic::AnthropicClient::new(key));
        self
    }

    pub fn with_google_api_key(mut self, key: &str) -> Self {
        self.google_client = Some(providers::google::GoogleClient::new(key));
        self
    }

    pub fn new_conversation(&self, model: Model) -> Conversation {
        Conversation::new(model)
    }

    fn get_provider(&self, provider: &Provider) -> Result<&dyn ProviderTrait, Error> {
        match provider {
            Provider::OpenAI => self
                .openai_client
                .as_ref()
                .map(|c| c as &dyn ProviderTrait)
                .ok_or_else(|| Error::MissingApiKey("OpenAI".to_string())),
            Provider::Anthropic => self
                .anthropic_client
                .as_ref()
                .map(|c| c as &dyn ProviderTrait)
                .ok_or_else(|| Error::MissingApiKey("Anthropic".to_string())),
            Provider::Google => self
                .google_client
                .as_ref()
                .map(|c| c as &dyn ProviderTrait)
                .ok_or_else(|| Error::MissingApiKey("Google".to_string())),
        }
    }

    pub async fn generate_text(
        &self,
        conversation: &Conversation,
        config: GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        let provider = self.get_provider(&conversation.model.provider)?;
        provider.generate_text(conversation, &config).await
    }

    /// Generate a streamed completion from the conversation.
    ///
    /// Returns a boxed stream of StreamEvent results that can be consumed
    /// to get incremental completion chunks (reasoning, text) and the final
    /// complete response.
    pub async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let provider = self.get_provider(&conversation.model.provider)?;
        provider.generate_stream(conversation, &config).await
    }
}

impl Default for LLMClient {
    fn default() -> Self {
        Self::new()
    }
}