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
    /// Set instead of `openai_client` by
    /// [`with_chatgpt_accounts`](Self::with_chatgpt_accounts).
    openai_accounts: Option<providers::openai::ChatGptAccounts>,
    anthropic_client: Option<providers::anthropic::AnthropicClient>,
    google_client: Option<providers::google::GoogleClient>,
    openrouter_client: Option<providers::openrouter::OpenRouterClient>,
    request_trace: Option<RequestTrace>,
    openai_request_hook: Option<Arc<dyn providers::openai::OpenAIRequestHook>>,
    openai_originator: Option<String>,
    openai_user_agent: Option<String>,
    stream_idle_timeout: Option<std::time::Duration>,
    /// The provider clients set while the client is in use, shared by every
    /// clone (see [`set_anthropic_client`](Self::set_anthropic_client)).
    set_clients: Arc<std::sync::RwLock<SetClients>>,
}

/// Provider clients set with [`LLMClient::set_openai_client`] and the like.
/// Each comes before the provider's client the `LLMClient` was built with.
#[derive(Default)]
struct SetClients {
    openai: Option<providers::openai::OpenAIClient>,
    anthropic: Option<providers::anthropic::AnthropicClient>,
    google: Option<providers::google::GoogleClient>,
    openrouter: Option<providers::openrouter::OpenRouterClient>,
}

impl LLMClient {
    pub fn new() -> Self {
        Self {
            openai_client: None,
            openai_accounts: None,
            anthropic_client: None,
            google_client: None,
            openrouter_client: None,
            request_trace: None,
            openai_request_hook: None,
            openai_originator: None,
            openai_user_agent: None,
            stream_idle_timeout: None,
            set_clients: Arc::default(),
        }
    }

    /// Configures every provider whose credentials are in the environment.
    /// Empty values are ignored.
    ///
    /// - OpenAI: ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and
    ///   `OPENAI_OAUTH_ACCOUNT_ID` (plus optional `OPENAI_OAUTH_REFRESH_TOKEN`)
    ///   when both are set; otherwise `OPENAI_API_KEY`. Setting only some of
    ///   the `OPENAI_OAUTH_*` variables is an error
    ///   ([`Error::AuthenticationFailed`]) rather than a silent switch to
    ///   API-key billing.
    /// - Anthropic: `ANTHROPIC_API_KEY`.
    /// - Google: `GOOGLE_API_KEY`, else `GEMINI_API_KEY`.
    /// - OpenRouter: `OPENROUTER_API_KEY`.
    ///
    /// ChatGPT OAuth refresh tokens rotate: a refresh makes the refresh token
    /// in the environment stop working. When `OPENAI_OAUTH_REFRESH_TOKEN` is
    /// the one the Codex `auth.json` holds (`$CODEX_HOME/auth.json`, else
    /// `~/.codex/auth.json`, compared by SHA-256 fingerprint, for the same
    /// account), the client uses that file as its credential store instead,
    /// as [`with_codex_auth_file`](Self::with_codex_auth_file) does, so
    /// rotated tokens are written back and Codex stays signed in (logged at
    /// info level). Otherwise rotated tokens are kept in memory only, and the
    /// first rotation logs a warning: save them from
    /// [`openai_chatgpt_oauth_credentials`](Self::openai_chatgpt_oauth_credentials),
    /// or load credentials with [`CodexAuthFile`](crate::codex_auth::CodexAuthFile)
    /// and [`with_codex_auth_file`](Self::with_codex_auth_file). Without a
    /// refresh token the access token is used until the server rejects it.
    ///
    /// The library reads no environment variables unless this is called; the
    /// Codex `auth.json` is read only when `OPENAI_OAUTH_REFRESH_TOKEN` is set.
    pub fn from_env() -> Result<Self, Error> {
        Self::from_env_with(|name| std::env::var(name).ok())
    }

    /// [`from_env`](Self::from_env), reading each variable (and, to locate
    /// the Codex `auth.json`, `CODEX_HOME`, `HOME` and `USERPROFILE`) through
    /// `lookup`. A lookup that returns `None` for some names hides them: an
    /// application that takes ChatGPT OAuth credentials from a file can hide
    /// the `OPENAI_OAUTH_*` variables so they are not read at all.
    pub fn from_env_with(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, Error> {
        let var = |name: &str| lookup(name).filter(|value| !value.trim().is_empty());
        let mut client = Self::new();
        let oauth = [
            "OPENAI_OAUTH_ACCESS_TOKEN",
            "OPENAI_OAUTH_ACCOUNT_ID",
            "OPENAI_OAUTH_REFRESH_TOKEN",
        ]
        .map(|name| (name, var(name)));
        match &oauth {
            [
                (_, Some(access_token)),
                (_, Some(account_id)),
                (_, refresh_token),
            ] => {
                client = client.with_chatgpt_oauth_from_env(
                    access_token,
                    refresh_token.as_deref(),
                    account_id,
                    &lookup,
                )?;
            }
            [(_, None), (_, None), (_, None)] => {
                if let Some(key) = var("OPENAI_API_KEY") {
                    client = client.with_openai_api_key(&key);
                }
            }
            partial => {
                let set: Vec<&str> = partial
                    .iter()
                    .filter(|(_, value)| value.is_some())
                    .map(|(name, _)| *name)
                    .collect();
                return Err(Error::AuthenticationFailed(format!(
                    "ChatGPT OAuth needs both OPENAI_OAUTH_ACCESS_TOKEN and OPENAI_OAUTH_ACCOUNT_ID, but only {} {} set. Set both, or unset the OPENAI_OAUTH_* variables to use OPENAI_API_KEY.",
                    set.join(" and "),
                    if set.len() == 1 { "is" } else { "are" }
                )));
            }
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
        Ok(client)
    }

    /// ChatGPT OAuth credentials from the environment: kept in step with the
    /// Codex `auth.json` when it holds the same refresh token, otherwise in
    /// memory with a warning at the first rotation.
    fn with_chatgpt_oauth_from_env(
        self,
        access_token: &str,
        refresh_token: Option<&str>,
        account_id: &str,
        lookup: &impl Fn(&str) -> Option<String>,
    ) -> Result<Self, Error> {
        if let Some(refresh_token) = refresh_token
            && let Some(file) = codex_file_holding(refresh_token, account_id, lookup)
        {
            log::info!(
                "arko_ai: OPENAI_OAUTH_REFRESH_TOKEN is the refresh token in {}; using that file for the ChatGPT OAuth credentials, so rotated tokens are saved to it and Codex stays signed in",
                file.path().display()
            );
            return self.with_codex_auth_file(&file);
        }
        let client = providers::openai::OpenAIClient::new_chatgpt_oauth(
            access_token,
            refresh_token,
            account_id,
        );
        let client = if refresh_token.is_some() {
            client.with_unsaved_rotation_warning()
        } else {
            client
        };
        Ok(self.with_openai_client(client))
    }

    /// Whether a provider client is configured for `provider`, by a builder
    /// or set while the client is in use.
    pub fn has_provider(&self, provider: &Provider) -> bool {
        let set = self.read_set_clients();
        let is_set = match provider {
            Provider::OpenAI => set.openai.is_some(),
            Provider::Anthropic => set.anthropic.is_some(),
            Provider::Google => set.google.is_some(),
            Provider::OpenRouter => set.openrouter.is_some(),
            Provider::ClaudeCode => false,
        };
        is_set || self.get_provider(provider).is_ok()
    }

    /// Sends `originator: <originator>` with every OpenAI request, naming
    /// your application to the ChatGPT backend. Nothing is sent by default.
    /// Use your own application's name, never another client's. Applies to
    /// the OpenAI provider whether configured before or after this call.
    pub fn with_openai_originator(mut self, originator: impl Into<String>) -> Self {
        self.openai_originator = Some(originator.into());
        self.reconfigure_openai();
        self
    }

    /// Sends `User-Agent: <user_agent>` with every OpenAI request. Nothing is
    /// sent by default. Applies to the OpenAI provider whether configured
    /// before or after this call.
    pub fn with_openai_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.openai_user_agent = Some(user_agent.into());
        self.reconfigure_openai();
        self
    }

    /// Writes every model request and response to `trace` for debugging;
    /// see [`RequestTrace`]. Applies to providers configured before or after
    /// this call. Currently implemented for the OpenAI provider only: the
    /// first request to another provider logs a warning (once per provider)
    /// that it is not traced.
    pub fn with_request_trace(mut self, trace: RequestTrace) -> Self {
        self.request_trace = Some(trace);
        self.reconfigure_openai();
        self
    }

    /// Cuts a streamed response off once no data has arrived for `timeout`,
    /// on every provider, whether configured before or after this call.
    /// Off by default, and deliberately so: there is no timeout on a
    /// response while its connection stays open.
    ///
    /// Models can reason for a long time without sending anything: GPT Pro
    /// models can think for over an hour, and a stream may carry no event
    /// until the reasoning ends. A timeout shorter than that cuts healthy
    /// responses off; they then fail with [`Error::StreamTruncated`], which
    /// is transient, so a caller that retries (such as the arko_agents
    /// runtime) sends them again, paying for the work again. If you set one,
    /// make it generous, well above the longest silent stretch the models you
    /// use can have (for example two hours).
    ///
    /// Only the time between data on a streamed response counts, from its
    /// first byte of body on: there is no total deadline, the wait for the
    /// response to start (its status and headers) is not covered, and
    /// responses that don't stream (`generate_text`, except over ChatGPT
    /// OAuth, whose backend always streams) are unaffected. Connecting stays
    /// bounded by its own 30-second timeout. The setting is never sent to
    /// the model.
    pub fn with_stream_idle_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.stream_idle_timeout = Some(timeout);
        self.reconfigure_openai();
        self.anthropic_client = self
            .anthropic_client
            .take()
            .map(|client| client.with_stream_idle_timeout(Some(timeout)));
        self.google_client = self
            .google_client
            .take()
            .map(|client| client.with_stream_idle_timeout(Some(timeout)));
        self.openrouter_client = self
            .openrouter_client
            .take()
            .map(|client| client.with_stream_idle_timeout(Some(timeout)));
        self
    }

    /// Applies the stream idle timeout set on this `LLMClient`, if any, to a
    /// provider client; otherwise the client keeps its own.
    fn idle<T>(&self, client: T, apply: impl FnOnce(T, Option<std::time::Duration>) -> T) -> T {
        match self.stream_idle_timeout {
            Some(timeout) => apply(client, Some(timeout)),
            None => client,
        }
    }

    /// Installs a hook that can adjust each OpenAI request; see
    /// [`OpenAIRequestHook`](providers::openai::OpenAIRequestHook).
    pub fn with_openai_request_hook(
        mut self,
        hook: Arc<dyn providers::openai::OpenAIRequestHook>,
    ) -> Self {
        self.openai_request_hook = Some(hook);
        self.reconfigure_openai();
        self
    }

    /// Applies the OpenAI settings made on this `LLMClient` to the OpenAI
    /// client, or to every account's client of a ChatGPT account list.
    fn reconfigure_openai(&mut self) {
        if let Some(client) = self.openai_client.take() {
            self.openai_client = Some(self.configure_openai(client));
        }
        if let Some(accounts) = self.openai_accounts.take() {
            self.openai_accounts =
                Some(accounts.map_clients(|client| self.configure_openai(client)));
        }
    }

    /// The OpenAI client the OAuth methods act on: one set with
    /// [`set_openai_client`](Self::set_openai_client), else the configured
    /// one, or, with a ChatGPT account list, the account the next request
    /// would go to. (Clones share the credentials.)
    fn openai(&self) -> Option<providers::openai::OpenAIClient> {
        if let Some(client) = self.set_openai() {
            return Some(client);
        }
        match &self.openai_accounts {
            Some(accounts) => Some(accounts.preferred_client().clone()),
            None => self.openai_client.clone(),
        }
    }

    /// Whether OpenAI requests go to the ChatGPT backend with an account's
    /// OAuth login, which takes no output limit
    /// ([`with_openai_chatgpt_oauth`](Self::with_openai_chatgpt_oauth),
    /// [`with_chatgpt_accounts`](Self::with_chatgpt_accounts)). `false`
    /// without an OpenAI client.
    pub async fn openai_uses_chatgpt_oauth(&self) -> bool {
        match self.openai() {
            Some(client) => client.is_chatgpt_oauth().await,
            None => false,
        }
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
        self.idle(
            client,
            providers::openai::OpenAIClient::with_stream_idle_timeout,
        )
    }

    pub fn with_openai_api_key(mut self, key: &str) -> Self {
        self.openai_accounts = None;
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
    /// token endpoint to an explicit
    /// [`refresh_openai_oauth_token`](Self::refresh_openai_oauth_token) call
    /// wait for the advertised reset when available and otherwise use
    /// bounded exponential backoff, for up to 8 retries; see
    /// [`OpenAIClient::with_oauth_rate_limit_retries`](providers::openai::OpenAIClient::with_oauth_rate_limit_retries).
    /// The refreshes requests make on their own, including those of the
    /// arko_agents runtime, never wait: their 429 goes to the agent's own
    /// rate-limit recovery.
    pub fn with_openai_chatgpt_oauth_rate_limit_retries(
        mut self,
        access_token: &str,
        refresh_token: Option<&str>,
        account_id: &str,
        retry_rate_limits: bool,
    ) -> Self {
        self.openai_accounts = None;
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
        self.openai_accounts = None;
        self.openai_client = Some(self.configure_openai(file.openai_client()?));
        Ok(self)
    }

    /// Configures the OpenAI provider with a pre-constructed [`OpenAIAuth`]
    /// value, allowing full control over the authentication method.
    ///
    /// [`OpenAIAuth`]: crate::providers::openai::OpenAIAuth
    pub fn with_openai_auth(mut self, auth: providers::openai::OpenAIAuth) -> Self {
        self.openai_accounts = None;
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
    /// configured on this `LLMClient`, and [`Error::InvalidRequest`] when the
    /// OpenAI provider has a list of ChatGPT accounts (update an account's
    /// own client instead).
    pub async fn update_openai_auth(
        &self,
        auth: providers::openai::OpenAIAuth,
    ) -> Result<(), Error> {
        if let Some(client) = self.set_openai() {
            client.update_auth(auth).await;
            return Ok(());
        }
        if self.openai_accounts.is_some() {
            return Err(Error::InvalidRequest(
                "The OpenAI provider has several ChatGPT accounts; update an account's own client (ChatGptAccount::client)".into(),
            ));
        }
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
        match self.openai() {
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
        match self.openai() {
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

    /// The most input tokens this client's requests to `model` may carry;
    /// see [`effective_max_input_tokens`](crate::input_limit::effective_max_input_tokens).
    /// For OpenAI models it depends on the client's OpenAI auth: API key or
    /// ChatGPT OAuth.
    pub async fn max_input_tokens(&self, model: &Model) -> Option<usize> {
        let chatgpt_oauth =
            model.provider == Provider::OpenAI && self.is_openai_chatgpt_oauth().await;
        crate::input_limit::effective_max_input_tokens(model, chatgpt_oauth)
    }

    /// Replaces the ChatGPT OAuth credentials only if the client still holds
    /// `expected`; returns whether it did.
    pub(crate) async fn replace_openai_chatgpt_credentials(
        &self,
        expected: &providers::openai::ChatGptOAuthCredentials,
        new: providers::openai::ChatGptOAuthCredentials,
    ) -> bool {
        match self.openai() {
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
        match self.openai() {
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
        match self.openai() {
            Some(client) => client.refresh_oauth_token().await,
            None => Err(Error::MissingApiKey("OpenAI".to_string())),
        }
    }

    /// Refresh once, bypassing provider-level retries so an agent can enforce
    /// its shared recovery deadline and respond to cancellation.
    pub async fn refresh_openai_oauth_token_once(
        &self,
    ) -> Result<providers::openai::OAuthRefreshResult, Error> {
        match self.openai() {
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
    /// caller just refreshed) and for API-key and unconfigured clients.
    /// Credentials without a refresh token are used until the server rejects
    /// them: this returns `Ok(None)` for them, and the rejected request fails
    /// with [`Error::AuthenticationFailed`]. See
    /// [`OpenAIClient::refresh_oauth_token_if_expiring`](providers::openai::OpenAIClient::refresh_oauth_token_if_expiring).
    pub async fn refresh_openai_oauth_token_if_expiring(
        &self,
        margin: std::time::Duration,
    ) -> Result<Option<providers::openai::OAuthRefreshResult>, Error> {
        match self.openai() {
            Some(client) => client.refresh_oauth_token_if_expiring(margin).await,
            None => Ok(None),
        }
    }

    pub fn with_anthropic_api_key(self, key: &str) -> Self {
        self.with_anthropic_client(providers::anthropic::AnthropicClient::new(key))
    }

    pub fn with_google_api_key(self, key: &str) -> Self {
        self.with_google_client(providers::google::GoogleClient::new(key))
    }

    /// Configures OpenRouter with an API key and default options. See
    /// [`with_openrouter_client`](Self::with_openrouter_client) for a client
    /// with default options, attribution or a different base URL.
    pub fn with_openrouter_api_key(self, key: &str) -> Self {
        self.with_openrouter_client(providers::openrouter::OpenRouterClient::new(key))
    }

    /// Installs a preconfigured OpenRouter client.
    pub fn with_openrouter_client(
        mut self,
        client: providers::openrouter::OpenRouterClient,
    ) -> Self {
        self.openrouter_client = Some(self.idle(
            client,
            providers::openrouter::OpenRouterClient::with_stream_idle_timeout,
        ));
        self
    }

    /// The OpenRouter client the `LLMClient` was built with, for catalog
    /// access ([`fetch_models`](providers::openrouter::OpenRouterClient::fetch_models),
    /// [`model`](providers::openrouter::OpenRouterClient::model)).
    pub fn openrouter(&self) -> Option<&providers::openrouter::OpenRouterClient> {
        self.openrouter_client.as_ref()
    }

    pub fn new_conversation(&self, model: Model) -> Conversation {
        Conversation::new(model)
    }

    /// Configures the OpenAI provider with an ordered list of ChatGPT OAuth
    /// accounts, first preferred: a request an account's usage limit (or
    /// credits) refuses goes to the next account at once, and requests move
    /// back as soon as a higher account is usable again. See
    /// [`ChatGptAccounts`](providers::openai::ChatGptAccounts). The OpenAI
    /// settings made on this `LLMClient` (originator, User-Agent, trace,
    /// hook, idle timeout) apply to every account.
    ///
    /// The OAuth methods ([`refresh_openai_oauth_token_if_expiring`](Self::refresh_openai_oauth_token_if_expiring),
    /// [`openai_chatgpt_oauth_credentials`](Self::openai_chatgpt_oauth_credentials)
    /// and the others) act on the account the next request would go to;
    /// each account refreshes its own credentials before it is used, and
    /// with a Codex `auth.json` keeps its file in step.
    pub fn with_chatgpt_accounts(mut self, accounts: providers::openai::ChatGptAccounts) -> Self {
        self.openai_client = None;
        self.openai_accounts = Some(accounts.map_clients(|client| self.configure_openai(client)));
        self
    }

    /// Configures the OpenAI provider with the Codex `auth.json` files of
    /// several ChatGPT accounts, labelled, first preferred; see
    /// [`with_chatgpt_accounts`](Self::with_chatgpt_accounts) and
    /// [`ChatGptAccounts::from_codex_auth_files`](providers::openai::ChatGptAccounts::from_codex_auth_files).
    pub fn with_codex_auth_files<'a>(
        self,
        files: impl IntoIterator<Item = (&'a str, &'a crate::codex_auth::CodexAuthFile)>,
    ) -> Result<Self, Error> {
        let accounts = providers::openai::ChatGptAccounts::from_codex_auth_files(files)?;
        Ok(self.with_chatgpt_accounts(accounts))
    }

    /// The ChatGPT account list installed with
    /// [`with_chatgpt_accounts`](Self::with_chatgpt_accounts), for its
    /// statuses, [`check_now`](providers::openai::ChatGptAccounts::check_now)
    /// and [`mark_usable`](providers::openai::ChatGptAccounts::mark_usable).
    /// Clones of this `LLMClient` share it.
    pub fn chatgpt_accounts(&self) -> Option<&providers::openai::ChatGptAccounts> {
        self.openai_accounts.as_ref()
    }

    /// Installs a preconfigured OpenAI client (for example one with
    /// [`with_base_url`](providers::openai::OpenAIClient::with_base_url)).
    pub fn with_openai_client(mut self, client: providers::openai::OpenAIClient) -> Self {
        self.openai_accounts = None;
        self.openai_client = Some(self.configure_openai(client));
        self
    }

    /// Installs a preconfigured Anthropic client.
    pub fn with_anthropic_client(mut self, client: providers::anthropic::AnthropicClient) -> Self {
        self.anthropic_client = Some(self.idle(
            client,
            providers::anthropic::AnthropicClient::with_stream_idle_timeout,
        ));
        self
    }

    /// Installs a preconfigured Google client.
    pub fn with_google_client(mut self, client: providers::google::GoogleClient) -> Self {
        self.google_client = Some(self.idle(
            client,
            providers::google::GoogleClient::with_stream_idle_timeout,
        ));
        self
    }

    /// Sets the OpenAI client of this `LLMClient` and of every clone of it,
    /// made before or after: for an application that takes credentials while
    /// it runs. It comes before the OpenAI client (or the ChatGPT accounts)
    /// the client was built with, and replaces one set before. Requests that
    /// start afterwards use it, also those of conversations and agent runs
    /// already going on. Requests in flight finish with the client they
    /// started with.
    ///
    /// Each clone's OpenAI settings (trace, hook, `originator`,
    /// `User-Agent`, the stream idle timeout) are applied to it where they are
    /// set, as [`with_openai_client`](Self::with_openai_client) applies them.
    pub fn set_openai_client(&self, client: providers::openai::OpenAIClient) {
        self.write_set_clients().openai = Some(client);
    }

    /// Sets the Anthropic client of this `LLMClient` and of every clone of
    /// it, made before or after, as
    /// [`set_openai_client`](Self::set_openai_client) sets OpenAI's. Each
    /// clone's stream idle timeout applies to it where one is set.
    pub fn set_anthropic_client(&self, client: providers::anthropic::AnthropicClient) {
        self.write_set_clients().anthropic = Some(client);
    }

    /// Sets the Google client of this `LLMClient` and of every clone of it,
    /// made before or after, as [`set_openai_client`](Self::set_openai_client)
    /// sets OpenAI's. Each clone's stream idle timeout applies to it where
    /// one is set.
    pub fn set_google_client(&self, client: providers::google::GoogleClient) {
        self.write_set_clients().google = Some(client);
    }

    /// Sets the OpenRouter client of this `LLMClient` and of every clone of
    /// it, made before or after, as
    /// [`set_openai_client`](Self::set_openai_client) sets OpenAI's. Each
    /// clone's stream idle timeout applies to it where one is set.
    pub fn set_openrouter_client(&self, client: providers::openrouter::OpenRouterClient) {
        self.write_set_clients().openrouter = Some(client);
    }

    fn write_set_clients(&self) -> std::sync::RwLockWriteGuard<'_, SetClients> {
        self.set_clients
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn read_set_clients(&self) -> std::sync::RwLockReadGuard<'_, SetClients> {
        self.set_clients
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The OpenAI client set with [`set_openai_client`](Self::set_openai_client),
    /// with this client's OpenAI settings.
    fn set_openai(&self) -> Option<providers::openai::OpenAIClient> {
        let client = self.read_set_clients().openai.clone()?;
        Some(self.configure_openai(client))
    }

    /// The client set for `provider` while the client is in use, with this
    /// client's settings.
    fn set_client(&self, provider: &Provider) -> Option<Box<dyn ProviderClient + Send + Sync>> {
        match provider {
            Provider::OpenAI => Some(Box::new(self.set_openai()?)),
            Provider::Anthropic => {
                let client = self.read_set_clients().anthropic.clone()?;
                Some(Box::new(self.idle(
                    client,
                    providers::anthropic::AnthropicClient::with_stream_idle_timeout,
                )))
            }
            Provider::Google => {
                let client = self.read_set_clients().google.clone()?;
                Some(Box::new(self.idle(
                    client,
                    providers::google::GoogleClient::with_stream_idle_timeout,
                )))
            }
            Provider::OpenRouter => {
                let client = self.read_set_clients().openrouter.clone()?;
                Some(Box::new(self.idle(
                    client,
                    providers::openrouter::OpenRouterClient::with_stream_idle_timeout,
                )))
            }
            Provider::ClaudeCode => None,
        }
    }

    fn get_provider(&self, provider: &Provider) -> Result<&dyn ProviderClient, Error> {
        match provider {
            Provider::OpenAI => self
                .openai_accounts
                .as_ref()
                .map(|accounts| accounts as &dyn ProviderClient)
                .or_else(|| self.openai_client.as_ref().map(|c| c as &dyn ProviderClient))
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
            Provider::ClaudeCode => Err(Error::ModelNotSupported(
                "Claude Code models run in a Claude Code session (arko_ai::claude_code), which arko_agents drives for an agent; they take no LLMClient requests.".to_string(),
            )),
        }
    }

    /// Request tracing covers the OpenAI provider only; says so, once per
    /// provider, when a traced client sends to another.
    fn note_untraced(&self, provider: &Provider) {
        if self.request_trace.is_some() && *provider != Provider::OpenAI {
            crate::notice::warn_once(&format!("trace:{provider:?}"), || {
                format!(
                    "arko_ai: request tracing covers OpenAI requests only; {provider:?} requests are not traced."
                )
            });
        }
    }

    pub async fn generate_text(
        &self,
        conversation: &Conversation,
        config: GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        if let Some(set) = self.set_client(&conversation.model.provider) {
            self.note_untraced(&conversation.model.provider);
            return set.generate_text(conversation, &config).await;
        }
        let provider = self.get_provider(&conversation.model.provider)?;
        self.note_untraced(&conversation.model.provider);
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
        if let Some(set) = self.set_client(&conversation.model.provider) {
            self.note_untraced(&conversation.model.provider);
            return set.generate_stream(conversation, &config).await;
        }
        let provider = self.get_provider(&conversation.model.provider)?;
        self.note_untraced(&conversation.model.provider);
        provider.generate_stream(conversation, &config).await
    }
}

/// The Codex `auth.json` (located through `lookup`, as
/// [`CodexAuthFile::default_path`] locates it) when it holds `refresh_token`
/// for `account_id`. Tokens are compared by SHA-256 fingerprint; a missing or
/// unreadable file is no match. Nothing about the tokens is logged.
///
/// [`CodexAuthFile::default_path`]: crate::codex_auth::CodexAuthFile::default_path
fn codex_file_holding(
    refresh_token: &str,
    account_id: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Option<crate::codex_auth::CodexAuthFile> {
    use sha2::{Digest, Sha256};
    let path = crate::codex_auth::CodexAuthFile::default_path_with(|name| {
        lookup(name).map(std::ffi::OsString::from)
    })?;
    if !path.is_file() {
        return None;
    }
    let file = match crate::codex_auth::CodexAuthFile::load(&path) {
        Ok(file) => file,
        Err(error) => {
            log::debug!(
                "arko_ai: not comparing OPENAI_OAUTH_REFRESH_TOKEN with {}: {error}",
                path.display()
            );
            return None;
        }
    };
    let stored = file.credentials().ok()?;
    let fingerprint = |token: &str| Sha256::digest(token.as_bytes());
    let same_token = stored
        .refresh_token
        .as_deref()
        .is_some_and(|stored| fingerprint(stored) == fingerprint(refresh_token));
    (same_token && stored.account_id == account_id).then_some(file)
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

    fn try_client(vars: &[(&str, &str)]) -> Result<LLMClient, Error> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        LLMClient::from_env_with(|name| vars.get(name).cloned())
    }

    fn client(vars: &[(&str, &str)]) -> LLMClient {
        try_client(vars).unwrap()
    }

    /// Whether OpenAI requests go to the ChatGPT backend, which takes no
    /// output limit.
    #[tokio::test]
    async fn the_client_tells_a_chatgpt_login_from_an_api_key() {
        assert!(!LLMClient::new().openai_uses_chatgpt_oauth().await);
        let api = LLMClient::new().with_openai_api_key("sk-test");
        assert!(!api.openai_uses_chatgpt_oauth().await);
        let oauth = LLMClient::new().with_openai_chatgpt_oauth("token", None, "account");
        assert!(oauth.openai_uses_chatgpt_oauth().await);
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
    }

    /// Part of the OAuth variables is a mistake, not a reason to bill the API
    /// key instead.
    #[test]
    fn a_partial_oauth_configuration_is_an_error() {
        for (set, named) in [
            (
                vec![("OPENAI_OAUTH_ACCESS_TOKEN", "access")],
                "OPENAI_OAUTH_ACCESS_TOKEN is set",
            ),
            (
                vec![("OPENAI_OAUTH_ACCOUNT_ID", "account")],
                "OPENAI_OAUTH_ACCOUNT_ID is set",
            ),
            (
                vec![
                    ("OPENAI_OAUTH_ACCESS_TOKEN", "access"),
                    ("OPENAI_OAUTH_REFRESH_TOKEN", "refresh"),
                ],
                "OPENAI_OAUTH_ACCESS_TOKEN and OPENAI_OAUTH_REFRESH_TOKEN",
            ),
            (
                vec![("OPENAI_OAUTH_REFRESH_TOKEN", "refresh")],
                "OPENAI_OAUTH_REFRESH_TOKEN is set",
            ),
        ] {
            let mut vars = set.clone();
            vars.push(("OPENAI_API_KEY", "sk"));
            let error = try_client(&vars).err().unwrap();
            assert!(matches!(error, Error::AuthenticationFailed(_)), "{error}");
            let message = error.to_string();
            assert!(message.contains(named), "{message}");
            for (_, value) in set {
                assert!(
                    !message.contains(value),
                    "values are never shown: {message}"
                );
            }
        }
        // Blank values count as unset.
        let key = client(&[("OPENAI_OAUTH_ACCESS_TOKEN", " "), ("OPENAI_API_KEY", "sk")]);
        assert!(key.has_provider(&Provider::OpenAI));
    }

    fn write_codex_auth(dir: &std::path::Path, refresh_token: &str, account_id: &str) {
        std::fs::write(
            dir.join("auth.json"),
            serde_json::json!({
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "file-access",
                    "refresh_token": refresh_token,
                    "account_id": account_id,
                }
            })
            .to_string(),
        )
        .unwrap();
    }

    fn openai(client: &LLMClient) -> &providers::openai::OpenAIClient {
        client.openai_client.as_ref().unwrap()
    }

    /// An environment refresh token that is Codex's own is kept in step with
    /// its auth.json, so a rotation can't sign Codex out.
    #[tokio::test]
    async fn an_environment_refresh_token_codex_holds_uses_its_auth_file() {
        let codex_home = tempfile::tempdir().unwrap();
        write_codex_auth(codex_home.path(), "shared-refresh", "account");
        let home = codex_home.path().to_str().unwrap();
        let oauth = |refresh: &str, account: &str| {
            client(&[
                ("CODEX_HOME", home),
                ("OPENAI_OAUTH_ACCESS_TOKEN", "env-access"),
                ("OPENAI_OAUTH_ACCOUNT_ID", account),
                ("OPENAI_OAUTH_REFRESH_TOKEN", refresh),
            ])
        };

        let shared = oauth("shared-refresh", "account");
        assert!(openai(&shared).has_credential_store());
        assert!(!openai(&shared).warns_unsaved_rotation());
        let credentials = shared.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(credentials.access_token, "file-access");

        // Another token, or the same token for another account: memory only,
        // with a warning at the first rotation.
        for other in [
            oauth("other-refresh", "account"),
            oauth("shared-refresh", "other"),
        ] {
            assert!(!openai(&other).has_credential_store());
            assert!(openai(&other).warns_unsaved_rotation());
            let credentials = other.openai_chatgpt_oauth_credentials().await.unwrap();
            assert_eq!(credentials.access_token, "env-access");
        }

        // Without a refresh token there is nothing to rotate or compare.
        let no_refresh = client(&[
            ("CODEX_HOME", home),
            ("OPENAI_OAUTH_ACCESS_TOKEN", "env-access"),
            ("OPENAI_OAUTH_ACCOUNT_ID", "account"),
        ]);
        assert!(!openai(&no_refresh).has_credential_store());
        assert!(!openai(&no_refresh).warns_unsaved_rotation());

        // A missing or unreadable file is no match; ~/.codex is found through
        // HOME when CODEX_HOME is unset.
        let empty = tempfile::tempdir().unwrap();
        std::fs::create_dir(empty.path().join(".codex")).unwrap();
        std::fs::write(empty.path().join(".codex").join("auth.json"), "not json").unwrap();
        let unreadable = client(&[
            ("HOME", empty.path().to_str().unwrap()),
            ("OPENAI_OAUTH_ACCESS_TOKEN", "env-access"),
            ("OPENAI_OAUTH_ACCOUNT_ID", "account"),
            ("OPENAI_OAUTH_REFRESH_TOKEN", "shared-refresh"),
        ]);
        assert!(!openai(&unreadable).has_credential_store());
        let home_dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(home_dir.path().join(".codex")).unwrap();
        write_codex_auth(&home_dir.path().join(".codex"), "shared-refresh", "account");
        let found = client(&[
            ("HOME", home_dir.path().to_str().unwrap()),
            ("OPENAI_OAUTH_ACCESS_TOKEN", "env-access"),
            ("OPENAI_OAUTH_ACCOUNT_ID", "account"),
            ("OPENAI_OAUTH_REFRESH_TOKEN", "shared-refresh"),
        ]);
        assert!(openai(&found).has_credential_store());
    }

    /// Only OpenAI requests are traced; a traced request to another provider
    /// says so once per provider.
    #[tokio::test]
    async fn traced_requests_to_other_providers_warn_that_they_are_not_traced() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let dir = tempfile::tempdir().unwrap();
        // A local server that answers every request with a 500, so the
        // requests fail at once.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let failing = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut chunk = [0; 8192];
                    while let Ok(n @ 1..) = socket.read(&mut chunk).await {
                        request.extend_from_slice(&chunk[..n]);
                        let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                            continue;
                        };
                        let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let length: usize = head
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .map_or(0, |value| value.trim().parse().unwrap_or(0));
                        if request.len() >= end + 4 + length {
                            break;
                        }
                    }
                    let _ = socket
                        .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                        .await;
                });
            }
        });
        let untraced = LLMClient::new()
            .with_anthropic_client(
                providers::anthropic::AnthropicClient::new("test").with_base_url(&failing),
            )
            .with_openai_client(
                providers::openai::OpenAIClient::new("test").with_base_url(&failing),
            );
        let mut claude = Conversation::new(Model::anthropic_claude_opus_5_5());
        claude.add_user_message("Hi.");
        let mut gpt = Conversation::new(Model::openai_gpt_6_astra());
        gpt.add_user_message("Hi.");

        assert!(
            untraced
                .generate_text(&claude, GenerationConfig::default())
                .await
                .is_err()
        );
        assert!(!crate::notice::warned("trace:Anthropic"));

        let traced = untraced.with_request_trace(RequestTrace::new(dir.path()));
        assert!(
            traced
                .generate_text(&gpt, GenerationConfig::default())
                .await
                .is_err()
        );
        assert!(!crate::notice::warned("trace:OpenAI"));
        assert!(
            traced
                .generate_stream(&claude, GenerationConfig::default())
                .await
                .is_err()
        );
        assert!(crate::notice::warned("trace:Anthropic"));
    }

    /// A provider client set while the client is in use reaches every clone,
    /// made before or after, from their next request: it adds the provider,
    /// and replaces the one the client was built with and one set before.
    #[tokio::test]
    async fn a_provider_client_set_in_use_reaches_every_clone() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // A local server that keeps each request's key and answers 500.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let keys = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen = keys.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let seen = seen.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut chunk = [0; 8192];
                    while let Ok(n @ 1..) = socket.read(&mut chunk).await {
                        request.extend_from_slice(&chunk[..n]);
                        let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                            continue;
                        };
                        let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let length: usize = head
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .map_or(0, |value| value.trim().parse().unwrap_or(0));
                        if request.len() >= end + 4 + length {
                            let key = head
                                .lines()
                                .find_map(|line| {
                                    line.strip_prefix("x-api-key:")
                                        .or_else(|| line.strip_prefix("authorization: bearer"))
                                })
                                .unwrap_or_default();
                            seen.lock().unwrap().push(key.trim().to_string());
                            break;
                        }
                    }
                    let _ = socket
                        .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                        .await;
                });
            }
        });
        let anthropic =
            |key: &str| providers::anthropic::AnthropicClient::new(key).with_base_url(&url);
        let mut claude = Conversation::new(Model::anthropic_claude_opus_5_5());
        claude.add_user_message("Hi.");
        let last_key = || keys.lock().unwrap().last().cloned();

        let client = LLMClient::new();
        let before = Arc::new(client.clone());
        assert!(!before.has_provider(&Provider::Anthropic));
        client.set_anthropic_client(anthropic("first"));
        let after = client.clone().with_openai_api_key("sk-other");
        for clone in [&client, &*before, &after] {
            assert!(clone.has_provider(&Provider::Anthropic));
            assert!(
                clone
                    .generate_text(&claude, GenerationConfig::default())
                    .await
                    .is_err()
            );
            assert_eq!(last_key().as_deref(), Some("first"));
        }
        // It comes before the client built with, and a later one replaces it.
        let built = LLMClient::new().with_anthropic_client(anthropic("built"));
        assert!(
            built
                .generate_stream(&claude, GenerationConfig::default())
                .await
                .is_err()
        );
        assert_eq!(last_key().as_deref(), Some("built"));
        built.set_anthropic_client(anthropic("first"));
        before.set_anthropic_client(anthropic("second"));
        for clone in [&built, &client] {
            assert!(
                clone
                    .generate_stream(&claude, GenerationConfig::default())
                    .await
                    .is_err()
            );
        }
        assert_eq!(
            *keys.lock().unwrap(),
            ["first", "first", "first", "built", "first", "second"]
        );

        // An OpenAI client set over a ChatGPT login is what the OAuth methods
        // see too.
        let oauth = client_with_oauth();
        assert!(oauth.is_openai_chatgpt_oauth().await);
        oauth.set_openai_client(providers::openai::OpenAIClient::new("sk-set").with_base_url(&url));
        assert!(!oauth.is_openai_chatgpt_oauth().await);
        assert!(oauth.openai_chatgpt_oauth_credentials().await.is_none());
        let mut gpt = Conversation::new(Model::openai_gpt_6_astra());
        gpt.add_user_message("Hi.");
        assert!(
            oauth
                .generate_text(&gpt, GenerationConfig::default())
                .await
                .is_err()
        );
        assert_eq!(last_key().as_deref(), Some("sk-set"));
    }

    fn client_with_oauth() -> LLMClient {
        client(&[
            ("OPENAI_OAUTH_ACCESS_TOKEN", "access"),
            ("OPENAI_OAUTH_ACCOUNT_ID", "account"),
        ])
    }

    /// The client's OpenAI auth decides the lifetime (5 minutes with an API
    /// key, 30 over ChatGPT OAuth) and whether turning caching off disables
    /// it: the ChatGPT OAuth backend caches regardless.
    #[tokio::test]
    async fn prompt_cache_lifetime_reads_the_openai_auth() {
        use crate::prompt_cache::{
            CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME, OPENAI_PROMPT_CACHE_LIFETIME, PromptCacheLifetime,
        };
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
        for config in [&GenerationConfig::default(), &off] {
            assert_eq!(
                oauth.prompt_cache_lifetime(&model, config).await,
                PromptCacheLifetime::Known(CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME)
            );
        }
        // Other providers don't depend on it, nor on being configured.
        assert_eq!(
            oauth
                .prompt_cache_lifetime(&Model::google_gemini_2_5_pro(), &off)
                .await,
            PromptCacheLifetime::Unknown
        );
    }
}
