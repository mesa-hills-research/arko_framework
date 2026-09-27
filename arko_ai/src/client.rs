use crate::{
    config::GenerationConfig,
    conversation::Conversation,
    error::Error,
    model::{Model, Provider},
    providers::{self, GenerationResponse, ProviderClient, StreamEvent},
    trace::RequestTrace,
};
use futures::stream::BoxStream;
use std::sync::Arc;

#[derive(Clone)]
pub struct LLMClient {
    openai_client: Option<providers::openai::OpenAIClient>,
    anthropic_client: Option<providers::anthropic::AnthropicClient>,
    google_client: Option<providers::google::GoogleClient>,
    openrouter_client: Option<providers::openrouter::OpenRouterClient>,
    request_trace: Option<RequestTrace>,
    openai_request_hook: Option<Arc<dyn providers::openai::OpenAIRequestHook>>,
    openai_originator: Option<String>,
    openai_user_agent: Option<String>,
}

impl LLMClient {
    pub fn new() -> Self {
        Self {
            openai_client: None,
            anthropic_client: None,
            google_client: None,
            openrouter_client: None,
            request_trace: None,
            openai_request_hook: None,
            openai_originator: None,
            openai_user_agent: None,
        }
    }

    /// Configures every provider whose credentials are in the environment.
    /// Empty values are ignored.
    ///
    /// - OpenAI: ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and
    ///   `OPENAI_OAUTH_ACCOUNT_ID` (plus optional `OPENAI_OAUTH_REFRESH_TOKEN`)
    ///   when both are set; otherwise `OPENAI_API_KEY`. For credentials in a
    ///   Codex `auth.json`, use [`CodexAuthFile`](crate::codex_auth::CodexAuthFile).
    /// - Anthropic: `ANTHROPIC_API_KEY`.
    /// - Google: `GOOGLE_API_KEY`, else `GEMINI_API_KEY`.
    /// - OpenRouter: `OPENROUTER_API_KEY`.
    ///
    /// The library reads no environment variables unless this is called.
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let var = |name: &str| lookup(name).filter(|value| !value.trim().is_empty());
        let mut client = Self::new();
        if let (Some(access_token), Some(account_id)) = (
            var("OPENAI_OAUTH_ACCESS_TOKEN"),
            var("OPENAI_OAUTH_ACCOUNT_ID"),
        ) {
            let refresh_token = var("OPENAI_OAUTH_REFRESH_TOKEN");
            client = client.with_openai_chatgpt_oauth(
                &access_token,
                refresh_token.as_deref(),
                &account_id,
            );
        } else if let Some(key) = var("OPENAI_API_KEY") {
            client = client.with_openai_api_key(&key);
        }
        if let Some(key) = var("ANTHROPIC_API_KEY") {
            client = client.with_anthropic_api_key(&key);
        }
        if let Some(key) = var("GOOGLE_API_KEY").or_else(|| var("GEMINI_API_KEY")) {
            client = client.with_google_api_key(&key);
        }
        if let Some(key) = var("OPENROUTER_API_KEY") {
            client = client.with_openrouter_api_key(&key);
        }
        client
    }

    /// Whether a provider client is configured for `provider`.
    pub fn has_provider(&self, provider: &Provider) -> bool {
        self.get_provider(provider).is_ok()
    }

    /// Sends `originator: <originator>` with every OpenAI request, naming
    /// your application to the ChatGPT backend. Nothing is sent by default.
    /// Use your own application's name, never another client's. Applies to
    /// the OpenAI provider whether configured before or after this call.
    pub fn with_openai_originator(mut self, originator: impl Into<String>) -> Self {
        self.openai_originator = Some(originator.into());
        self.openai_client = self
            .openai_client
            .take()
            .map(|client| self.configure_openai(client));
        self
    }

    /// Sends `User-Agent: <user_agent>` with every OpenAI request. Nothing is
    /// sent by default. Applies to the OpenAI provider whether configured
    /// before or after this call.
    pub fn with_openai_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.openai_user_agent = Some(user_agent.into());
        self.openai_client = self
            .openai_client
            .take()
            .map(|client| self.configure_openai(client));
        self
    }

    /// Writes every model request and response to `trace` for debugging;
    /// see [`RequestTrace`]. Applies to providers configured before or after
    /// this call. Currently implemented for the OpenAI provider.
    pub fn with_request_trace(mut self, trace: RequestTrace) -> Self {
        self.request_trace = Some(trace);
        self.openai_client = self
            .openai_client
            .take()
            .map(|client| self.configure_openai(client));
        self
    }

    /// Installs a hook that can adjust each OpenAI request; see
    /// [`OpenAIRequestHook`](providers::openai::OpenAIRequestHook).
    pub fn with_openai_request_hook(
        mut self,
        hook: Arc<dyn providers::openai::OpenAIRequestHook>,
    ) -> Self {
        self.openai_request_hook = Some(hook);
        self.openai_client = self
            .openai_client
            .take()
            .map(|client| self.configure_openai(client));
        self
    }

    /// Applies the OpenAI settings made on this `LLMClient`. Settings left
    /// unset keep the OpenAI client's own, so a client installed with
    /// [`with_openai_client`](Self::with_openai_client) keeps its trace,
    /// hook, originator and User-Agent.
    fn configure_openai(
        &self,
        mut client: providers::openai::OpenAIClient,
    ) -> providers::openai::OpenAIClient {
        if self.request_trace.is_some() {
            client = client.with_request_trace(self.request_trace.clone());
        }
        if self.openai_request_hook.is_some() {
            client = client.with_request_hook(self.openai_request_hook.clone());
        }
        if self.openai_originator.is_some() {
            client = client.with_originator(self.openai_originator.clone());
        }
        if self.openai_user_agent.is_some() {
            client = client.with_user_agent(self.openai_user_agent.clone());
        }
        client
    }

    pub fn with_openai_api_key(mut self, key: &str) -> Self {
        self.openai_client = Some(self.configure_openai(providers::openai::OpenAIClient::new(key)));
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
    /// to obtain a fresh access token, or
    /// [`refresh_openai_oauth_token_if_expiring`](Self::refresh_openai_oauth_token_if_expiring)
    /// to refresh proactively. The `arko_agents` runtime refreshes
    /// automatically before each LLM call.
    /// If the server rejects an OAuth request with HTTP 401 `token_expired`
    /// before that expiry, the provider refreshes once and resends the request.
    /// Other authentication failures and a second rejection return to the
    /// caller (with a credential store, such as a Codex `auth.json`, any 401
    /// first re-reads the store, and newer tokens there get one resend).
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
    /// otherwise use bounded exponential backoff, for up to 8 retries; see
    /// [`OpenAIClient::with_oauth_rate_limit_retries`](providers::openai::OpenAIClient::with_oauth_rate_limit_retries).
    pub fn with_openai_chatgpt_oauth_rate_limit_retries(
        mut self,
        access_token: &str,
        refresh_token: Option<&str>,
        account_id: &str,
        retry_rate_limits: bool,
    ) -> Self {
        self.openai_client = Some(
            self.configure_openai(
                providers::openai::OpenAIClient::new_chatgpt_oauth(
                    access_token,
                    refresh_token,
                    account_id,
                )
                .with_oauth_rate_limit_retries(retry_rate_limits),
            ),
        );
        self
    }

    /// Configures the OpenAI provider with the ChatGPT OAuth credentials in
    /// a Codex `auth.json`, kept in step with the file: every token refresh
    /// first re-reads it (adopting tokens Codex refreshed, for the same
    /// account only) and rotated tokens are written back at once. See
    /// [`CodexAuthFile::openai_client`](crate::codex_auth::CodexAuthFile::openai_client)
    /// and [`CodexAuthStore`](crate::codex_auth::CodexAuthStore).
    pub fn with_codex_auth_file(
        mut self,
        file: &crate::codex_auth::CodexAuthFile,
    ) -> Result<Self, Error> {
        self.openai_client = Some(self.configure_openai(file.openai_client()?));
        Ok(self)
    }

    /// Configures the OpenAI provider with a pre-constructed [`OpenAIAuth`]
    /// value, allowing full control over the authentication method.
    ///
    /// [`OpenAIAuth`]: crate::providers::openai::OpenAIAuth
    pub fn with_openai_auth(mut self, auth: providers::openai::OpenAIAuth) -> Self {
        self.openai_client =
            Some(self.configure_openai(providers::openai::OpenAIClient::with_auth(auth)));
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
    ///
    /// A refresh in flight on this client (or a clone) is waited for: it
    /// holds the credentials until its new tokens are installed and stored.
    /// A refresh finishes even when its caller is dropped, but a Tokio
    /// runtime shutting down cancels it, and tokens it had not stored yet are
    /// lost (the previous refresh token no longer works). A host that must
    /// not lose them stops starting refreshes (finishes or cancels its agent
    /// runs) and then awaits this before shutting the runtime down; clients
    /// built separately (even from one Codex `auth.json`) are awaited one by
    /// one.
    pub async fn openai_chatgpt_oauth_credentials(
        &self,
    ) -> Option<providers::openai::ChatGptOAuthCredentials> {
        match &self.openai_client {
            Some(client) => client.chatgpt_oauth_credentials().await,
            None => None,
        }
    }

    /// How long the provider keeps the prompt cache of this client's requests
    /// for `model` with `config` after their last use, for a conversation
    /// with caching enabled (the default); see
    /// [`prompt_cache_lifetime`](crate::prompt_cache::prompt_cache_lifetime).
    /// For OpenAI models it depends on the client's OpenAI auth: API key or
    /// ChatGPT OAuth.
    pub async fn prompt_cache_lifetime(
        &self,
        model: &Model,
        config: &GenerationConfig,
    ) -> crate::prompt_cache::PromptCacheLifetime {
        let chatgpt_oauth =
            model.provider == Provider::OpenAI && self.is_openai_chatgpt_oauth().await;
        crate::prompt_cache::prompt_cache_lifetime(model, config, chatgpt_oauth)
    }

    /// Replaces the ChatGPT OAuth credentials only if the client still holds
    /// `expected`; returns whether it did.
    pub(crate) async fn replace_openai_chatgpt_credentials(
        &self,
        expected: &providers::openai::ChatGptOAuthCredentials,
        new: providers::openai::ChatGptOAuthCredentials,
    ) -> bool {
        match &self.openai_client {
            Some(client) => client.replace_chatgpt_credentials(expected, new).await,
            None => false,
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
    /// for subsequent requests. A call that waits while another refresh
    /// replaces the credentials returns those instead of refreshing again.
    ///
    /// Once the refresh request is sent, the refresh completes even if this
    /// future is dropped, though not through a runtime shutdown (see
    /// [`openai_chatgpt_oauth_credentials`](Self::openai_chatgpt_oauth_credentials)).
    /// The request times out after 60 seconds with a retryable error.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingApiKey`] if no OpenAI client is configured.
    /// - [`Error::AuthenticationFailed`] if no refresh token is stored.
    /// - [`Error::InvalidRequest`] if the client uses API key auth.
    /// - [`Error::ProviderError`] if the token endpoint returns an error.
    /// - [`Error::HttpError`] if the request fails or times out.
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

    /// Refreshes the OpenAI ChatGPT OAuth credentials if the access token
    /// expires within `margin`, with at most one refresh request (no
    /// provider-level retries, like
    /// [`refresh_openai_oauth_token_once`](Self::refresh_openai_oauth_token_once)).
    ///
    /// Use this instead of checking [`is_openai_token_expired`](Self::is_openai_token_expired)
    /// and then refreshing: two tasks sharing a client could both see an
    /// expiring token and rotate it twice. Here the decision is made under
    /// the credential lock, after re-reading the credential store (such as a
    /// Codex `auth.json`), so concurrent calls refresh once.
    ///
    /// Returns `Ok(Some(_))` when this call installed new credentials
    /// (refreshed, or adopted from the credential store), and `Ok(None)` when
    /// the credentials don't expire within `margin` (including ones another
    /// caller just refreshed) and for API-key and unconfigured clients. See
    /// [`OpenAIClient::refresh_oauth_token_if_expiring`](providers::openai::OpenAIClient::refresh_oauth_token_if_expiring).
    pub async fn refresh_openai_oauth_token_if_expiring(
        &self,
        margin: std::time::Duration,
    ) -> Result<Option<providers::openai::OAuthRefreshResult>, Error> {
        match &self.openai_client {
            Some(client) => client.refresh_oauth_token_if_expiring(margin).await,
            None => Ok(None),
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

    /// Configures OpenRouter with an API key and default options. See
    /// [`with_openrouter_client`](Self::with_openrouter_client) for a client
    /// with default options, attribution or a different base URL.
    pub fn with_openrouter_api_key(mut self, key: &str) -> Self {
        self.openrouter_client = Some(providers::openrouter::OpenRouterClient::new(key));
        self
    }

    /// Installs a preconfigured OpenRouter client.
    pub fn with_openrouter_client(
        mut self,
        client: providers::openrouter::OpenRouterClient,
    ) -> Self {
        self.openrouter_client = Some(client);
        self
    }

    /// The configured OpenRouter client, for catalog access
    /// ([`fetch_models`](providers::openrouter::OpenRouterClient::fetch_models),
    /// [`model`](providers::openrouter::OpenRouterClient::model)).
    pub fn openrouter(&self) -> Option<&providers::openrouter::OpenRouterClient> {
        self.openrouter_client.as_ref()
    }

    pub fn new_conversation(&self, model: Model) -> Conversation {
        Conversation::new(model)
    }

    /// Installs a preconfigured OpenAI client (for example one with
    /// [`with_base_url`](providers::openai::OpenAIClient::with_base_url)).
    pub fn with_openai_client(mut self, client: providers::openai::OpenAIClient) -> Self {
        self.openai_client = Some(self.configure_openai(client));
        self
    }

    /// Installs a preconfigured Anthropic client.
    pub fn with_anthropic_client(mut self, client: providers::anthropic::AnthropicClient) -> Self {
        self.anthropic_client = Some(client);
        self
    }

    /// Installs a preconfigured Google client.
    pub fn with_google_client(mut self, client: providers::google::GoogleClient) -> Self {
        self.google_client = Some(client);
        self
    }

    fn get_provider(&self, provider: &Provider) -> Result<&dyn ProviderClient, Error> {
        match provider {
            Provider::OpenAI => self
                .openai_client
                .as_ref()
                .map(|c| c as &dyn ProviderClient)
                .ok_or_else(|| Error::MissingApiKey("OpenAI".to_string())),
            Provider::Anthropic => self
                .anthropic_client
                .as_ref()
                .map(|c| c as &dyn ProviderClient)
                .ok_or_else(|| Error::MissingApiKey("Anthropic".to_string())),
            Provider::Google => self
                .google_client
                .as_ref()
                .map(|c| c as &dyn ProviderClient)
                .ok_or_else(|| Error::MissingApiKey("Google".to_string())),
            Provider::OpenRouter => self
                .openrouter_client
                .as_ref()
                .map(|c| c as &dyn ProviderClient)
                .ok_or_else(|| Error::MissingApiKey("OpenRouter".to_string())),
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
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn client(vars: &[(&str, &str)]) -> LLMClient {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        LLMClient::from_lookup(|name| vars.get(name).cloned())
    }

    #[tokio::test]
    async fn from_env_configures_the_providers_it_finds() {
        let empty = client(&[]);
        for provider in [
            Provider::OpenAI,
            Provider::Anthropic,
            Provider::Google,
            Provider::OpenRouter,
        ] {
            assert!(!empty.has_provider(&provider));
        }

        let keys = client(&[
            ("OPENAI_API_KEY", "sk-test"),
            ("ANTHROPIC_API_KEY", "ak-test"),
            ("GEMINI_API_KEY", "g-test"),
            ("OPENROUTER_API_KEY", "or-test"),
        ]);
        assert!(keys.has_provider(&Provider::OpenAI));
        assert!(!keys.is_openai_chatgpt_oauth().await);
        assert!(keys.has_provider(&Provider::Anthropic));
        assert!(keys.has_provider(&Provider::Google));
        assert!(keys.has_provider(&Provider::OpenRouter));
        assert!(keys.openrouter().is_some());

        let oauth = client(&[
            ("OPENAI_OAUTH_ACCESS_TOKEN", "access"),
            ("OPENAI_OAUTH_ACCOUNT_ID", "account"),
            ("OPENAI_API_KEY", "sk-test"),
            ("ANTHROPIC_API_KEY", "  "),
        ]);
        assert!(oauth.is_openai_chatgpt_oauth().await);
        let credentials = oauth.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(credentials.account_id, "account");
        assert!(credentials.refresh_token.is_none());
        assert!(
            !oauth.has_provider(&Provider::Anthropic),
            "blank values are ignored"
        );

        // An incomplete OAuth pair falls back to the API key.
        let partial = client(&[
            ("OPENAI_OAUTH_ACCESS_TOKEN", "access"),
            ("OPENAI_API_KEY", "sk"),
        ]);
        assert!(!partial.is_openai_chatgpt_oauth().await);
        assert!(partial.has_provider(&Provider::OpenAI));
    }

    /// The client's OpenAI auth decides whether turning caching off
    /// disables it: the ChatGPT OAuth backend caches regardless.
    #[tokio::test]
    async fn prompt_cache_lifetime_reads_the_openai_auth() {
        use crate::prompt_cache::{OPENAI_PROMPT_CACHE_LIFETIME, PromptCacheLifetime};
        let off = GenerationConfig::default().with_prompt_caching(false);
        let model = Model::openai_gpt_6_astra();
        let key = client(&[("OPENAI_API_KEY", "sk-test")]);
        assert_eq!(
            key.prompt_cache_lifetime(&model, &GenerationConfig::default())
                .await,
            PromptCacheLifetime::Known(OPENAI_PROMPT_CACHE_LIFETIME)
        );
        assert_eq!(
            key.prompt_cache_lifetime(&model, &off).await,
            PromptCacheLifetime::Disabled
        );
        let oauth = client(&[
            ("OPENAI_OAUTH_ACCESS_TOKEN", "access"),
            ("OPENAI_OAUTH_ACCOUNT_ID", "account"),
        ]);
        assert_eq!(
            oauth.prompt_cache_lifetime(&model, &off).await,
            PromptCacheLifetime::Known(OPENAI_PROMPT_CACHE_LIFETIME)
        );
        // Other providers don't depend on it, nor on being configured.
        assert_eq!(
            oauth
                .prompt_cache_lifetime(&Model::google_gemini_2_5_pro(), &off)
                .await,
            PromptCacheLifetime::Unknown
        );
    }
}
