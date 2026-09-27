//! Reading and safely updating a Codex CLI `auth.json` (ChatGPT OAuth).
//!
//! ChatGPT OAuth refresh tokens rotate: once a refresh succeeds, the previous
//! refresh token stops working. When arko refreshes credentials loaded from the
//! file Codex also uses, the new tokens must be written back, or Codex is left
//! with a dead refresh token; and before refreshing, the file must be read
//! again, since Codex may already have refreshed (a second use of a rotated
//! refresh token makes Codex ask the user to sign in again).
//!
//! [`CodexAuthFile::openai_client`] and [`CodexAuthFile::client`] build a
//! client that does both on every refresh (through a [`CodexAuthStore`]).
//! [`CodexAuthFile`] loads the file, writes rotated tokens back atomically
//! (after a backup), and never overwrites tokens that another process (such
//! as Codex) wrote in the meantime.

use crate::{
    client::LLMClient,
    error::Error,
    providers::openai::{ChatGptOAuthCredentials, ChatGptOAuthStore, OpenAIClient},
};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Result of [`CodexAuthFile::persist`] and [`CodexAuthFile::sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexAuthSync {
    /// The file and the credentials already agree.
    Unchanged,
    /// Rotated credentials were written to the file.
    Saved,
    /// The file held newer credentials (written by another process), which the
    /// client now uses.
    Reloaded,
}

/// A Codex CLI `auth.json`. Both layouts are accepted: tokens nested under
/// `"tokens"` (what Codex writes) and top-level `access_token` /
/// `refresh_token` / `account_id` fields.
#[derive(Clone)]
pub struct CodexAuthFile {
    path: PathBuf,
    /// The file's bytes as last read or written by this value.
    snapshot: Vec<u8>,
}

// The snapshot holds tokens and possibly an API key: show only the path.
impl std::fmt::Debug for CodexAuthFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CodexAuthFile")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// How [`CodexAuthFile::reconcile`] brought the file and a client's
/// credentials together.
enum Reconciled {
    Unchanged,
    Saved,
    /// The file holds newer credentials; `bytes` become the snapshot once
    /// they are adopted.
    Newer {
        credentials: ChatGptOAuthCredentials,
        bytes: Vec<u8>,
    },
}

/// How often [`CodexAuthFile::persist_tokens`] starts over when the file
/// changes between reading it and replacing it.
const PERSIST_ATTEMPTS: usize = 3;

#[cfg(test)]
type RenameHook = Box<dyn FnMut(&Path)>;

#[cfg(test)]
thread_local! {
    /// Runs just before `persist_tokens` re-reads the file ahead of the
    /// rename, so tests can change the file in that window.
    static BEFORE_RENAME: std::cell::RefCell<Option<RenameHook>> =
        const { std::cell::RefCell::new(None) };
}

impl CodexAuthFile {
    /// `$CODEX_HOME/auth.json`, else `~/.codex/auth.json`. Reads environment
    /// variables only when called.
    pub fn default_path() -> Option<PathBuf> {
        if let Some(codex_home) = std::env::var_os("CODEX_HOME").filter(|value| !value.is_empty()) {
            return Some(PathBuf::from(codex_home).join("auth.json"));
        }
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .filter(|value| !value.is_empty())?;
        Some(PathBuf::from(home).join(".codex").join("auth.json"))
    }

    /// Loads the file at [`default_path`](Self::default_path).
    pub fn load_default() -> Result<Self, Error> {
        let path = Self::default_path().ok_or_else(|| {
            Error::AuthenticationFailed("No home directory to locate ~/.codex/auth.json".into())
        })?;
        Self::load(path)
    }

    pub fn load(path: impl Into<PathBuf>) -> Result<Self, Error> {
        let path = path.into();
        let snapshot = std::fs::read(&path)?;
        let file = Self { path, snapshot };
        file.credentials()?;
        Ok(file)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The ChatGPT OAuth credentials in the file.
    pub fn credentials(&self) -> Result<ChatGptOAuthCredentials, Error> {
        parse_credentials(&self.snapshot, &self.path)
    }

    /// An OpenAI client using the file's ChatGPT OAuth credentials and kept
    /// in step with the file: every token refresh first re-reads the file
    /// (adopting tokens Codex refreshed, for the same account only), and
    /// rotated tokens are written back as soon as a refresh succeeds. See
    /// [`CodexAuthStore`].
    pub fn openai_client(&self) -> Result<OpenAIClient, Error> {
        let credentials = self.credentials()?;
        Ok(OpenAIClient::new_chatgpt_oauth(
            &credentials.access_token,
            credentials.refresh_token.as_deref(),
            &credentials.account_id,
        )
        .with_credential_store(Some(Arc::new(CodexAuthStore::new(self.clone())))))
    }

    /// An [`LLMClient`] using [`openai_client`](Self::openai_client).
    pub fn client(&self) -> Result<LLMClient, Error> {
        Ok(LLMClient::new().with_openai_client(self.openai_client()?))
    }

    /// Writes `credentials` to the file, keeping every other field and the
    /// file's layout. The previous contents go to `auth.json.bak` and the new
    /// contents are written atomically (temporary file, then rename).
    ///
    /// Fails without writing when the file changed since it was loaded or last
    /// written and now holds different credentials; newer credentials written by
    /// another process are never overwritten. A change that leaves the
    /// credentials as loaded (formatting or other fields) is not a rotation:
    /// `credentials` are then written onto the file's current contents.
    pub fn persist(
        &mut self,
        credentials: &ChatGptOAuthCredentials,
    ) -> Result<CodexAuthSync, Error> {
        self.persist_tokens(credentials, None)
    }

    /// [`persist`](Self::persist), also writing the `id_token` a refresh
    /// returned (Codex reads the account's plan and email from it). Without
    /// one, the file keeps its `id_token`.
    ///
    /// The file is read again just before it is replaced; if it changed in
    /// between, the checks start over with its new contents.
    pub fn persist_tokens(
        &mut self,
        credentials: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<CodexAuthSync, Error> {
        for _ in 0..PERSIST_ATTEMPTS {
            let current = std::fs::read(&self.path)?;
            if current != self.snapshot {
                let in_file = parse_credentials(&current, &self.path)?;
                if in_file == *credentials {
                    self.snapshot = current;
                    return Ok(CodexAuthSync::Unchanged);
                }
                if in_file != self.credentials()? {
                    return Err(Error::AuthenticationFailed(format!(
                        "{} changed since it was loaded; refusing to overwrite newer credentials",
                        self.path.display()
                    )));
                }
                self.snapshot = current.clone();
            } else if self.credentials()? == *credentials {
                return Ok(CodexAuthSync::Unchanged);
            }

            let bytes = self.contents_with(credentials, id_token)?;
            let target = std::fs::canonicalize(&self.path)?;
            let permissions = std::fs::metadata(&target)?.permissions();
            let backup = with_suffix(&target, ".bak");
            write_file(&backup, &self.snapshot, &permissions)?;
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos());
            let temporary = with_suffix(&target, &format!(".tmp-{}-{nanos}", std::process::id()));
            // The contents are synced before the rename, so the new name never
            // refers to a partly written file.
            write_temporary(&temporary, &permissions, |file| file.write_all(&bytes))?;
            // Narrow the window in which another process's write could be
            // lost: the file must still hold what the new contents are based
            // on. (Codex takes no lock on auth.json, so the window cannot be
            // closed entirely.)
            #[cfg(test)]
            BEFORE_RENAME.with(|hook| {
                if let Some(hook) = hook.borrow_mut().as_mut() {
                    hook(&target);
                }
            });
            match std::fs::read(&target) {
                Ok(latest) if latest == current => {}
                Ok(_) => {
                    let _ = std::fs::remove_file(&temporary);
                    continue;
                }
                Err(error) => {
                    let _ = std::fs::remove_file(&temporary);
                    return Err(error.into());
                }
            }
            if let Err(error) = std::fs::rename(&temporary, &target) {
                let _ = std::fs::remove_file(&temporary);
                return Err(error.into());
            }
            self.snapshot = bytes;
            // The file is replaced; make the rename itself durable. A failure here
            // is only logged: the new contents are already in place.
            if let Err(error) = sync_parent_directory(&target) {
                log::warn!(
                    "arko_ai: could not sync the directory of {}: {error}",
                    target.display()
                );
            }
            return Ok(CodexAuthSync::Saved);
        }
        Err(Error::AuthenticationFailed(format!(
            "{} kept changing while rotated credentials were being saved; not overwriting it",
            self.path.display()
        )))
    }

    /// Writes `credentials` to a new file next to this one,
    /// `auth.json.recovered-<unix seconds>`, in this file's format, and
    /// returns its path. On Unix only the owner can read it (mode 0600 from
    /// creation). An existing file is never replaced.
    ///
    /// For when [`persist`](Self::persist) or [`sync`](Self::sync) fails
    /// after a refresh: the previous refresh token no longer works, so the
    /// rotated credentials must not be lost with the process. Move the
    /// recovery file over `auth.json` (or sign in again) before Codex next
    /// refreshes.
    pub fn write_recovery(&self, credentials: &ChatGptOAuthCredentials) -> Result<PathBuf, Error> {
        self.write_recovery_tokens(credentials, None)
    }

    /// [`write_recovery`](Self::write_recovery) with the `id_token` a refresh
    /// returned.
    pub fn write_recovery_tokens(
        &self,
        credentials: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<PathBuf, Error> {
        let bytes = self.contents_with(credentials, id_token)?;
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        let base = with_suffix(&self.path, &format!(".recovered-{seconds}"));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        for attempt in 0..100 {
            let path = if attempt == 0 {
                base.clone()
            } else {
                with_suffix(&base, &format!("-{attempt}"))
            };
            let mut file = match options.open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            };
            let written = file.write_all(&bytes).and_then(|()| file.sync_all());
            if let Err(error) = written {
                // Close first: Windows cannot remove an open file.
                drop(file);
                let _ = std::fs::remove_file(&path);
                return Err(error.into());
            }
            if let Err(error) = sync_parent_directory(&path) {
                log::warn!(
                    "arko_ai: could not sync the directory of {}: {error}",
                    path.display()
                );
            }
            return Ok(path);
        }
        Err(Error::IoError(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} and 99 numbered variants already exist", base.display()),
        )))
    }

    /// This file's contents (as last read or written) with `credentials` (and
    /// `id_token`, when given) in place of its tokens, keeping every other
    /// field and the layout.
    fn contents_with(
        &self,
        credentials: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<Vec<u8>, Error> {
        let mut value: Value = serde_json::from_slice(&self.snapshot)?;
        let nested = value.get("tokens").is_some_and(Value::is_object);
        let tokens = if nested {
            &mut value["tokens"]
        } else {
            &mut value
        };
        tokens["access_token"] = Value::String(credentials.access_token.clone());
        if let Some(refresh_token) = &credentials.refresh_token {
            tokens["refresh_token"] = Value::String(refresh_token.clone());
        }
        if let Some(id_token) = id_token.filter(|token| !token.is_empty()) {
            tokens["id_token"] = Value::String(id_token.to_string());
        }
        tokens["account_id"] = Value::String(credentials.account_id.clone());
        if nested || value.get("last_refresh").is_some() {
            value["last_refresh"] = Value::String(rfc3339_utc(SystemTime::now()));
        }
        let mut bytes = serde_json::to_vec_pretty(&value)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Brings the file and a client's credentials together: saves
    /// credentials the client rotated, or reports newer credentials another
    /// process wrote. Only credentials of the same account are ever saved or
    /// reported; both having rotated is an error, since only one rotation can
    /// be valid.
    fn reconcile(&mut self, in_client: &ChatGptOAuthCredentials) -> Result<Reconciled, Error> {
        let known = self.credentials()?;
        let current = std::fs::read(&self.path)?;
        let in_file = parse_credentials(&current, &self.path)?;
        if in_file.account_id != in_client.account_id {
            return Err(Error::AuthenticationFailed(format!(
                "{} holds a different ChatGPT account than this client (signed out, or signed in to another account); sign in again",
                self.path.display()
            )));
        }
        if current != self.snapshot {
            if in_file == *in_client {
                self.snapshot = current;
                return Ok(Reconciled::Unchanged);
            }
            if in_file != known {
                if *in_client == known {
                    return Ok(Reconciled::Newer {
                        credentials: in_file,
                        bytes: current,
                    });
                }
                return Err(Error::AuthenticationFailed(format!(
                    "Both {} and this client rotated ChatGPT OAuth credentials; sign in again",
                    self.path.display()
                )));
            }
            // Only formatting or other fields changed: not a rotation.
            self.snapshot = current;
        }
        if *in_client == known {
            return Ok(Reconciled::Unchanged);
        }
        self.persist(in_client)?;
        Ok(Reconciled::Saved)
    }

    /// Reconciles the file with `client`'s ChatGPT OAuth credentials: writes
    /// credentials the client rotated, or loads newer credentials another
    /// process wrote into the client. Fails when both changed, since only one
    /// rotation can be valid, and when the file holds a different account
    /// (nothing is then written or loaded). Clients without ChatGPT OAuth are
    /// left alone.
    ///
    /// A client from [`client`](Self::client) or
    /// [`openai_client`](Self::openai_client) keeps the file in step on its
    /// own; `sync` is for clients configured some other way.
    pub async fn sync(&mut self, client: &LLMClient) -> Result<CodexAuthSync, Error> {
        let Some(in_client) = client.openai_chatgpt_oauth_credentials().await else {
            return Ok(CodexAuthSync::Unchanged);
        };
        match self.reconcile(&in_client)? {
            Reconciled::Unchanged => Ok(CodexAuthSync::Unchanged),
            Reconciled::Saved => Ok(CodexAuthSync::Saved),
            Reconciled::Newer { credentials, bytes } => {
                // Only replace what was read: a refresh that finished in
                // the meantime must not be undone.
                if client
                    .replace_openai_chatgpt_credentials(&in_client, credentials.clone())
                    .await
                {
                    self.snapshot = bytes;
                    return Ok(CodexAuthSync::Reloaded);
                }
                if client.openai_chatgpt_oauth_credentials().await.as_ref() == Some(&credentials) {
                    self.snapshot = bytes;
                    return Ok(CodexAuthSync::Unchanged);
                }
                Err(Error::AuthenticationFailed(format!(
                    "Both {} and this client rotated ChatGPT OAuth credentials; sign in again",
                    self.path.display()
                )))
            }
        }
    }
}

/// A [`CodexAuthFile`] as the [`ChatGptOAuthStore`] of an OpenAI client:
/// before every refresh the file is read again, and tokens Codex refreshed
/// are used instead of refreshing (for the same account only; a file now
/// holding another account stops the refresh). Rotated tokens, with the
/// refresh's `id_token`, are written back as soon as a refresh succeeds
/// (backup, atomic replace, never over newer tokens).
///
/// When rotated tokens cannot be written back, they go to a private
/// recovery file next to `auth.json` (see
/// [`CodexAuthFile::write_recovery`]) and the refresh returns an error
/// naming it; the client keeps using the new tokens, and the next refresh
/// tries to save them again first.
pub struct CodexAuthStore {
    file: Mutex<CodexAuthFile>,
}

impl CodexAuthStore {
    pub fn new(file: CodexAuthFile) -> Self {
        Self {
            file: Mutex::new(file),
        }
    }

    /// The `auth.json` this store keeps in step.
    pub fn path(&self) -> PathBuf {
        self.lock().path.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CodexAuthFile> {
        // Every update leaves the file state consistent, so a holder that
        // panicked leaves nothing half-done.
        self.file.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl std::fmt::Debug for CodexAuthStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CodexAuthStore")
            .field("path", &self.path())
            .finish_non_exhaustive()
    }
}

impl ChatGptOAuthStore for CodexAuthStore {
    fn before_refresh(
        &self,
        current: &ChatGptOAuthCredentials,
    ) -> Result<Option<ChatGptOAuthCredentials>, Error> {
        let mut file = self.lock();
        match file.reconcile(current)? {
            Reconciled::Unchanged | Reconciled::Saved => Ok(None),
            Reconciled::Newer { credentials, bytes } => {
                file.snapshot = bytes;
                Ok(Some(credentials))
            }
        }
    }

    fn save_refreshed(
        &self,
        refreshed: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<(), Error> {
        let mut file = self.lock();
        let error = match file.persist_tokens(refreshed, id_token) {
            Ok(_) => return Ok(()),
            Err(error) => error,
        };
        let path = file.path.display().to_string();
        let message = match file.write_recovery_tokens(refreshed, id_token) {
            Ok(recovery) => format!(
                "refreshed ChatGPT OAuth tokens could not be saved to {path} ({error}); they are in {} (readable only by you): put it in place of {path}, or sign in to Codex again, before Codex next refreshes",
                recovery.display()
            ),
            Err(recovery_error) => format!(
                "refreshed ChatGPT OAuth tokens could not be saved to {path} ({error}) nor to a recovery file ({recovery_error}); sign in to Codex again"
            ),
        };
        log::error!("arko_ai: {message}");
        Err(Error::AuthenticationFailed(message))
    }
}

fn parse_credentials(bytes: &[u8], path: &Path) -> Result<ChatGptOAuthCredentials, Error> {
    let value: Value = serde_json::from_slice(bytes)?;
    let tokens = value
        .get("tokens")
        .filter(|tokens| tokens.is_object())
        .unwrap_or(&value);
    let field = |name: &str| {
        tokens
            .get(name)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    let missing = |name: &str| {
        Error::AuthenticationFailed(format!("{} has no ChatGPT OAuth {name}", path.display()))
    };
    Ok(ChatGptOAuthCredentials {
        access_token: field("access_token").ok_or_else(|| missing("access_token"))?,
        refresh_token: field("refresh_token"),
        account_id: field("account_id").ok_or_else(|| missing("account_id"))?,
    })
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn write_file(path: &Path, bytes: &[u8], permissions: &std::fs::Permissions) -> Result<(), Error> {
    let mut file = std::fs::File::create(path)?;
    fill_file(&mut file, permissions, |file| file.write_all(bytes))?;
    Ok(())
}

/// Creates `path` and writes it with `write`, removing it again when anything
/// fails after it was created.
fn write_temporary(
    path: &Path,
    permissions: &std::fs::Permissions,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> Result<(), Error> {
    let mut file = std::fs::File::create(path)?;
    let result = fill_file(&mut file, permissions, write);
    if result.is_err() {
        // Close first: Windows cannot remove an open file.
        drop(file);
        let _ = std::fs::remove_file(path);
    }
    result.map_err(Error::from)
}

fn fill_file(
    file: &mut std::fs::File,
    permissions: &std::fs::Permissions,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    // Credentials: match the original file's permissions before writing.
    file.set_permissions(permissions.clone())?;
    write(file)?;
    file.sync_all()
}

/// Syncs the directory containing `path`, so a rename into it survives a
/// crash. Windows cannot open a directory as a file; there this does nothing.
fn sync_parent_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// `YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`, the format Codex writes for `last_refresh`.
fn rfc3339_utc(at: SystemTime) -> String {
    let elapsed = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = elapsed.as_secs() as i64;
    let (days, time) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        time / 3_600,
        time % 3_600 / 60,
        time % 60,
        elapsed.subsec_nanos()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAIAuth;
    use std::time::Duration;

    const NESTED: &str = r#"{
  "OPENAI_API_KEY": null,
  "auth_mode": "chatgpt",
  "last_refresh": "2026-09-21T06:33:44.750634434Z",
  "tokens": {
    "access_token": "access-1",
    "account_id": "account-1",
    "id_token": "id-1",
    "refresh_token": "refresh-1"
  }
}"#;

    fn temp_file(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        std::fs::write(&path, contents).unwrap();
        (dir, path)
    }

    fn rotated() -> ChatGptOAuthCredentials {
        ChatGptOAuthCredentials {
            access_token: "access-2".into(),
            refresh_token: Some("refresh-2".into()),
            account_id: "account-1".into(),
        }
    }

    #[test]
    fn both_layouts_load() {
        let (_dir, path) = temp_file(NESTED);
        let credentials = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        assert_eq!(credentials.access_token, "access-1");
        assert_eq!(credentials.refresh_token.as_deref(), Some("refresh-1"));
        assert_eq!(credentials.account_id, "account-1");

        let (_dir, path) =
            temp_file(r#"{"access_token":"a","refresh_token":"r","account_id":"acc"}"#);
        let credentials = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        assert_eq!(
            (
                credentials.access_token.as_str(),
                credentials.account_id.as_str()
            ),
            ("a", "acc")
        );

        let (_dir, path) = temp_file(r#"{"tokens":{"access_token":"a"}}"#);
        assert!(matches!(
            CodexAuthFile::load(&path),
            Err(Error::AuthenticationFailed(_))
        ));
    }

    #[test]
    fn persist_backs_up_keeps_other_fields_and_permissions() {
        let (_dir, path) = temp_file(NESTED);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let mut file = CodexAuthFile::load(&path).unwrap();
        assert_eq!(
            file.persist(&file.credentials().unwrap()).unwrap(),
            CodexAuthSync::Unchanged
        );
        assert_eq!(file.persist(&rotated()).unwrap(), CodexAuthSync::Saved);

        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["access_token"], "access-2");
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");
        assert_eq!(saved["tokens"]["id_token"], "id-1");
        assert_eq!(saved["auth_mode"], "chatgpt");
        assert!(saved["OPENAI_API_KEY"].is_null());
        assert_ne!(saved["last_refresh"], "2026-09-21T06:33:44.750634434Z");
        let backup = std::fs::read_to_string(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(backup, NESTED);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        let leftovers = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".tmp-")
            })
            .count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn persist_never_overwrites_newer_tokens_written_elsewhere() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let external = NESTED.replace("refresh-1", "refresh-from-codex");
        std::fs::write(&path, &external).unwrap();
        assert!(matches!(
            file.persist(&rotated()),
            Err(Error::AuthenticationFailed(_))
        ));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), external);
    }

    /// `NESTED` rewritten by another program: same credentials, different
    /// formatting and an extra field.
    fn reformatted() -> String {
        let mut value: Value = serde_json::from_str(NESTED).unwrap();
        value["written_by"] = Value::String("another program".into());
        serde_json::to_string(&value).unwrap()
    }

    #[test]
    fn persist_writes_onto_a_file_changed_without_a_rotation() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let external = reformatted();
        std::fs::write(&path, &external).unwrap();

        assert_eq!(file.persist(&rotated()).unwrap(), CodexAuthSync::Saved);
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["access_token"], "access-2");
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");
        // Merged onto the current contents, not the loaded ones.
        assert_eq!(saved["written_by"], "another program");
        let backup = std::fs::read_to_string(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(backup, external);
        assert_eq!(file.persist(&rotated()).unwrap(), CodexAuthSync::Unchanged);
    }

    #[tokio::test]
    async fn sync_saves_a_client_rotation_onto_a_file_changed_without_a_rotation() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client = file.client().unwrap();
        std::fs::write(&path, reformatted()).unwrap();
        // Nothing rotated: the file is left as it is.
        assert_eq!(file.sync(&client).await.unwrap(), CodexAuthSync::Unchanged);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), reformatted());

        std::fs::write(&path, NESTED).unwrap();
        client
            .update_openai_auth(OpenAIAuth::ChatGptOAuth(rotated()))
            .await
            .unwrap();
        assert_eq!(file.sync(&client).await.unwrap(), CodexAuthSync::Saved);
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );
    }

    #[test]
    fn recovery_files_are_private_new_files_next_to_the_original() {
        let (dir, path) = temp_file(NESTED);
        let file = CodexAuthFile::load(&path).unwrap();
        let first = file.write_recovery(&rotated()).unwrap();
        let second = file.write_recovery(&rotated()).unwrap();
        assert_ne!(first, second, "an existing recovery file is never replaced");
        for recovery in [&first, &second] {
            assert_eq!(recovery.parent(), Some(dir.path()));
            let name = recovery.file_name().unwrap().to_string_lossy();
            assert!(name.starts_with("auth.json.recovered-"), "{name}");
            assert_eq!(
                CodexAuthFile::load(recovery).unwrap().credentials().unwrap(),
                rotated()
            );
            let saved: Value = serde_json::from_slice(&std::fs::read(recovery).unwrap()).unwrap();
            assert_eq!(saved["tokens"]["id_token"], "id-1");
            assert_eq!(saved["auth_mode"], "chatgpt");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = std::fs::metadata(recovery).unwrap().permissions().mode() & 0o777;
                assert_eq!(mode, 0o600);
            }
        }
        // The original is untouched.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);
    }

    #[test]
    fn a_failed_temporary_write_leaves_no_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json.tmp-test");
        let permissions = std::fs::metadata(dir.path()).unwrap().permissions();
        let result = write_temporary(&path, &permissions, |file| {
            file.write_all(b"{\"tokens\":")?;
            Err(std::io::Error::other("disk full"))
        });
        assert!(matches!(result, Err(Error::IoError(_))));
        assert!(!path.exists());
    }

    #[test]
    fn debug_output_shows_the_path_but_no_file_contents() {
        let contents = NESTED.replace(
            r#""OPENAI_API_KEY": null"#,
            r#""OPENAI_API_KEY": "sk-synthetic""#,
        );
        let (_dir, path) = temp_file(&contents);
        let file = CodexAuthFile::load(&path).unwrap();
        let debug = format!("{file:?}");
        assert!(debug.contains("auth.json"), "{debug}");
        for secret in ["access-1", "refresh-1", "id-1", "sk-synthetic"] {
            assert!(!debug.contains(secret), "{debug}");
        }
    }

    #[tokio::test]
    async fn sync_saves_client_rotations_and_adopts_external_ones() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client = file.client().unwrap();
        assert_eq!(file.sync(&client).await.unwrap(), CodexAuthSync::Unchanged);

        // The client rotated: the file is updated.
        client
            .update_openai_auth(OpenAIAuth::ChatGptOAuth(rotated()))
            .await
            .unwrap();
        assert_eq!(file.sync(&client).await.unwrap(), CodexAuthSync::Saved);
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );

        // Codex rotated: the client adopts the file's tokens.
        let external = std::fs::read_to_string(&path)
            .unwrap()
            .replace("refresh-2", "refresh-3");
        std::fs::write(&path, &external).unwrap();
        assert_eq!(file.sync(&client).await.unwrap(), CodexAuthSync::Reloaded);
        let adopted = client.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(adopted.refresh_token.as_deref(), Some("refresh-3"));

        // Both rotated: nothing is written and the conflict is reported.
        client
            .update_openai_auth(OpenAIAuth::ChatGptOAuth(ChatGptOAuthCredentials {
                refresh_token: Some("refresh-4".into()),
                ..rotated()
            }))
            .await
            .unwrap();
        let external = external.replace("refresh-3", "refresh-5");
        std::fs::write(&path, &external).unwrap();
        assert!(file.sync(&client).await.is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), external);
    }

    #[test]
    fn persisting_a_refresh_writes_its_id_token_and_keeps_the_old_one_otherwise() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        assert_eq!(
            file.persist_tokens(&rotated(), Some("id-2")).unwrap(),
            CodexAuthSync::Saved
        );
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["id_token"], "id-2");
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");

        let third = ChatGptOAuthCredentials {
            refresh_token: Some("refresh-3".into()),
            ..rotated()
        };
        file.persist_tokens(&third, None).unwrap();
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["id_token"], "id-2");

        // The flat layout gets the id token at the top level.
        let (_dir, path) =
            temp_file(r#"{"access_token":"a","refresh_token":"r","account_id":"account-1"}"#);
        let mut file = CodexAuthFile::load(&path).unwrap();
        file.persist_tokens(&rotated(), Some("id-2")).unwrap();
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["id_token"], "id-2");
        assert!(saved.get("tokens").is_none());
    }

    fn set_before_rename(hook: Option<RenameHook>) {
        BEFORE_RENAME.with(|slot| *slot.borrow_mut() = hook);
    }

    fn temporary_files(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".tmp-")
            })
            .count()
    }

    #[test]
    fn a_rotation_written_just_before_the_rename_is_not_overwritten() {
        let (dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let external = NESTED.replace("refresh-1", "refresh-from-codex");
        let written = external.clone();
        set_before_rename(Some(Box::new(move |target| {
            std::fs::write(target, &written).unwrap();
        })));
        let result = file.persist_tokens(&rotated(), Some("id-2"));
        set_before_rename(None);
        assert!(matches!(result, Err(Error::AuthenticationFailed(_))), "{result:?}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), external);
        assert_eq!(temporary_files(dir.path()), 0);
    }

    #[test]
    fn a_change_without_a_rotation_just_before_the_rename_is_merged() {
        let (dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let mut once = Some(reformatted());
        set_before_rename(Some(Box::new(move |target| {
            if let Some(contents) = once.take() {
                std::fs::write(target, contents).unwrap();
            }
        })));
        let result = file.persist_tokens(&rotated(), None);
        set_before_rename(None);
        assert_eq!(result.unwrap(), CodexAuthSync::Saved);
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");
        assert_eq!(saved["written_by"], "another program");
        assert_eq!(temporary_files(dir.path()), 0);
    }

    fn other_account() -> String {
        NESTED
            .replace("account-1", "account-2")
            .replace("refresh-1", "refresh-other")
    }

    fn codex_refreshed() -> String {
        NESTED
            .replace("access-1", "access-codex")
            .replace("refresh-1", "refresh-codex")
    }

    #[tokio::test]
    async fn sync_never_adopts_or_saves_another_account() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client =
            LLMClient::new().with_openai_chatgpt_oauth("access-1", Some("refresh-1"), "account-1");
        // Codex signed in to another account: the client keeps its own.
        std::fs::write(&path, other_account()).unwrap();
        let error = file.sync(&client).await.unwrap_err();
        assert!(error.to_string().contains("different ChatGPT account"), "{error}");
        let credentials = client.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(credentials.account_id, "account-1");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), other_account());

        // A client holding another account never writes it over the file.
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client =
            LLMClient::new().with_openai_chatgpt_oauth("access-9", Some("refresh-9"), "account-2");
        assert!(file.sync(&client).await.is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);
    }

    #[test]
    fn the_store_adopts_codex_rotations_saves_unsaved_ones_and_refuses_other_accounts() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        assert_eq!(store.before_refresh(&loaded).unwrap(), None);

        // Codex refreshed: its tokens are used instead of refreshing, once.
        std::fs::write(&path, codex_refreshed()).unwrap();
        let newer = store.before_refresh(&loaded).unwrap().unwrap();
        assert_eq!(newer.access_token, "access-codex");
        assert_eq!(newer.refresh_token.as_deref(), Some("refresh-codex"));
        assert_eq!(store.before_refresh(&newer).unwrap(), None);

        // A client holding a rotation that was never saved saves it first.
        let unsaved = ChatGptOAuthCredentials {
            access_token: "access-unsaved".into(),
            refresh_token: Some("refresh-unsaved".into()),
            account_id: "account-1".into(),
        };
        assert_eq!(store.before_refresh(&unsaved).unwrap(), None);
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            unsaved
        );

        // Another account in the file stops the refresh and changes nothing.
        std::fs::write(&path, other_account()).unwrap();
        let error = store.before_refresh(&unsaved).unwrap_err();
        assert!(matches!(error, Error::AuthenticationFailed(_)));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), other_account());
        // A file that can't be read stops it too.
        std::fs::write(&path, "{not json").unwrap();
        assert!(store.before_refresh(&unsaved).is_err());
    }

    #[test]
    fn the_store_saves_refreshes_and_keeps_unsaveable_ones_in_a_recovery_file() {
        let (dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        store.save_refreshed(&rotated(), Some("id-2")).unwrap();
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["id_token"], "id-2");

        // Codex wrote a different rotation meanwhile: nothing is overwritten
        // and the refreshed tokens go to a recovery file.
        let external = std::fs::read_to_string(&path)
            .unwrap()
            .replace("refresh-2", "refresh-codex");
        std::fs::write(&path, &external).unwrap();
        let refreshed = ChatGptOAuthCredentials {
            access_token: "access-3".into(),
            refresh_token: Some("refresh-3".into()),
            account_id: "account-1".into(),
        };
        let error = store
            .save_refreshed(&refreshed, Some("id-3"))
            .unwrap_err()
            .to_string();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), external);
        for secret in ["access-3", "refresh-3", "id-3", "account-1", "refresh-codex"] {
            assert!(!error.contains(secret), "{error}");
        }
        let recovery = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.to_string_lossy().contains(".recovered-"))
            .unwrap();
        assert!(error.contains(&recovery.display().to_string()), "{error}");
        assert_eq!(
            CodexAuthFile::load(&recovery).unwrap().credentials().unwrap(),
            refreshed
        );
        let debug = format!("{store:?}");
        assert!(debug.contains("auth.json") && !debug.contains("refresh"), "{debug}");
    }

    #[tokio::test]
    async fn clients_from_the_file_carry_the_store() {
        let (_dir, path) = temp_file(NESTED);
        let file = CodexAuthFile::load(&path).unwrap();
        let client = file.client().unwrap();
        // Codex refreshed; a refresh adopts its tokens without a request (no
        // token endpoint is reachable from this test).
        std::fs::write(&path, codex_refreshed()).unwrap();
        let result = client.refresh_openai_oauth_token_once().await.unwrap();
        assert_eq!(result.access_token, "access-codex");
        let credentials = client.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(credentials.refresh_token.as_deref(), Some("refresh-codex"));
        let via_builder = LLMClient::new().with_codex_auth_file(&file).unwrap();
        assert!(via_builder.is_openai_chatgpt_oauth().await);
    }

    #[test]
    fn timestamps_use_codex_format() {
        // A multiple of 100 ns: Windows `SystemTime` has 100 ns resolution.
        let at = UNIX_EPOCH + Duration::new(1_790_577_224, 750_634_400);
        assert_eq!(rfc3339_utc(at), "2026-09-28T06:33:44.750634400Z");
        assert_eq!(rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00.000000000Z");
    }
}
