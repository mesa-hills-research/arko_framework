use super::ChatGptOAuthCredentials;
use crate::error::Error;
use std::sync::Arc;

/// Storage for ChatGPT OAuth credentials shared with other programs, such as
/// the Codex CLI's `auth.json` ([`CodexAuthStore`](crate::codex_auth::CodexAuthStore)).
///
/// ChatGPT refresh tokens rotate: once a refresh succeeds, the previous
/// refresh token stops working, and the OAuth server treats a second use of
/// it as a reused token (Codex then asks the user to sign in again). So a
/// client with a store installed (see
/// [`OpenAIClient::with_credential_store`](super::OpenAIClient::with_credential_store))
/// consults it around every refresh, proactive or after an HTTP 401:
///
/// 1. [`before_refresh`](Self::before_refresh) re-reads the store. When
///    another program already refreshed, its tokens are adopted and no
///    refresh request is made.
/// 2. After a refresh, [`save_refreshed`](Self::save_refreshed) stores the
///    rotated tokens at once.
/// 3. When the refresh request fails, the store is read again: if another
///    program rotated the tokens in the meantime, they are adopted.
///
/// The client holds its credential lock, and the store's
/// [`refresh_lock`](Self::refresh_lock) when it has one, across these calls,
/// so neither clones of one client nor clients sharing the lock refresh
/// concurrently. The calls run to completion even when the caller of the
/// refresh is dropped. Implementations may block briefly (a small file read
/// or write); they must never log or return tokens or account ids in error
/// messages.
pub trait ChatGptOAuthStore: Send + Sync {
    /// Called before a refresh with the credentials the client holds (and is
    /// about to refresh). Returns newer credentials for the same account to
    /// use instead of refreshing, `None` to go ahead with the refresh, or an
    /// error to stop it: a refresh must not proceed when the store cannot be
    /// read, since the stored tokens may already have rotated.
    fn before_refresh(
        &self,
        current: &ChatGptOAuthCredentials,
    ) -> Result<Option<ChatGptOAuthCredentials>, Error>;

    /// Called right after a refresh with the rotated credentials and the
    /// refresh response's `id_token`, if any. The previous refresh token no
    /// longer works, so these must be stored now. An error is returned to the
    /// caller of the refresh; the client keeps (and uses) the new
    /// credentials either way.
    fn save_refreshed(
        &self,
        refreshed: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<(), Error>;

    /// A lock shared by every store of the same credentials in this process
    /// (for a Codex `auth.json`, every [`CodexAuthStore`](crate::codex_auth::CodexAuthStore)
    /// of the same file), held from [`before_refresh`](Self::before_refresh)
    /// to [`save_refreshed`](Self::save_refreshed): clients with different
    /// stores of the same credentials then never refresh concurrently. A
    /// client takes its own credential lock first, then this one. `None`
    /// (the default) when no other store shares the credentials.
    fn refresh_lock(&self) -> Option<Arc<tokio::sync::Mutex<()>>> {
        None
    }
}
