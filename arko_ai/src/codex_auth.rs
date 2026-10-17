//! Reading and safely updating a Codex CLI `auth.json` (ChatGPT OAuth).
//!
//! ChatGPT OAuth refresh tokens rotate. Once a refresh succeeds, the previous
//! refresh token stops working. When arko refreshes credentials loaded from the
//! file Codex also uses, the new tokens must be written back, or Codex is left
//! with a dead refresh token. And before refreshing, the file must be read
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
    providers::openai::{ChatGptOAuthCredentials, ChatGptOAuthStore, OpenAIClient, spent_tokens},
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
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

/// A Codex CLI `auth.json`. Accepts both tokens nested under
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
    /// The file holds newer credentials. `bytes` become the snapshot once
    /// they are adopted.
    Newer {
        credentials: ChatGptOAuthCredentials,
        bytes: Vec<u8>,
    },
}

/// How often [`CodexAuthFile::persist_tokens`] starts over when the file
/// changes between reading it and replacing it.
const PERSIST_ATTEMPTS: usize = 3;

/// How often the file is read before a JSON error is believed: Codex writes
/// `auth.json` in place (truncate, then write), so a read can see it empty or
/// half written. The reads are [`SETTLE_DELAY`] apart.
const SETTLE_READS: usize = 5;
const SETTLE_DELAY: std::time::Duration = std::time::Duration::from_millis(30);

#[cfg(test)]
type RenameHook = Box<dyn FnMut(&Path)>;

#[cfg(test)]
type ReadHook = Box<dyn FnMut(&Path)>;

#[cfg(test)]
thread_local! {
    /// Runs just before `persist_tokens` re-reads the file ahead of the
    /// rename, so tests can change the file in that window.
    static BEFORE_RENAME: std::cell::RefCell<Option<RenameHook>> =
        const { std::cell::RefCell::new(None) };
    /// Runs before each read of `read_credentials`, so tests can write the
    /// file as Codex does.
    static BEFORE_READ: std::cell::RefCell<Option<ReadHook>> =
        const { std::cell::RefCell::new(None) };
}

impl CodexAuthFile {
    /// `$CODEX_HOME/auth.json`, else `~/.codex/auth.json`. Reads environment
    /// variables only when called.
    pub fn default_path() -> Option<PathBuf> {
        Self::default_path_with(|name| std::env::var_os(name))
    }

    /// [`default_path`](Self::default_path), reading variables with `var`.
    pub(crate) fn default_path_with(
        var: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Option<PathBuf> {
        if let Some(codex_home) = var("CODEX_HOME").filter(|value| !value.is_empty()) {
            return Some(PathBuf::from(codex_home).join("auth.json"));
        }
        let home = var("HOME")
            .or_else(|| var("USERPROFILE"))
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

    /// Loads the file at `path`. A relative path is made absolute against
    /// the current directory now, so that a later change of directory
    /// doesn't move the file this value (and its clients) read and write.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, Error> {
        let path = std::path::absolute(path.into())?;
        let (snapshot, _) = read_credentials(&path)?;
        Ok(Self { path, snapshot })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The ChatGPT OAuth credentials in the file.
    pub fn credentials(&self) -> Result<ChatGptOAuthCredentials, Error> {
        parse_credentials(&self.snapshot, &self.path)
    }

    /// An OpenAI client using the file's ChatGPT OAuth credentials and kept
    /// in step with the file. Every token refresh first re-reads the file
    /// (adopting tokens Codex refreshed, for the same account only), and
    /// rotated tokens are written back as soon as a refresh succeeds.
    /// Clients made from the same file in this process refresh one at a
    /// time. See [`CodexAuthStore`].
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
    /// written and now holds different credentials. Newer credentials written by
    /// another process are never overwritten. A change that leaves the
    /// credentials as loaded (formatting or other fields) is not a rotation,
    /// so `credentials` are then written onto the file's current contents.
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
    /// The file is read again just before it is replaced. If it changed in
    /// between, the checks start over with its new contents.
    pub fn persist_tokens(
        &mut self,
        credentials: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<CodexAuthSync, Error> {
        for _ in 0..PERSIST_ATTEMPTS {
            let (current, in_file) = read_credentials(&self.path)?;
            if current != self.snapshot {
                if in_file == *credentials {
                    self.snapshot = current;
                    return Ok(CodexAuthSync::Unchanged);
                }
                if in_file != self.credentials()? {
                    return Err(Error::AuthenticationFailed(format!(
                        "{} changed since it was loaded. Refusing to overwrite newer credentials",
                        self.path.display()
                    )));
                }
                self.snapshot = current.clone();
            } else if in_file == *credentials {
                return Ok(CodexAuthSync::Unchanged);
            }

            let bytes = self.contents_with(credentials, id_token)?;
            let target = std::fs::canonicalize(&self.path)?;
            let permissions = std::fs::metadata(&target)?.permissions();
            let backup = with_suffix(&target, ".bak");
            write_backup(&backup, &self.snapshot, &permissions)?;
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
            "{} kept changing while rotated credentials were being saved. Not overwriting it",
            self.path.display()
        )))
    }

    /// Writes `credentials` to a new file next to this one,
    /// `auth.json.recovered-<unix seconds>`, in this file's format, and
    /// returns its path. On Unix only the owner can read it (mode 0600 from
    /// creation). An existing file is never replaced.
    ///
    /// For when [`persist`](Self::persist) or [`sync`](Self::sync) fails
    /// after a refresh. The previous refresh token no longer works, so the
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
    /// credentials the client rotated (with `id_token`, the one their refresh
    /// returned, when known), or reports newer credentials another process
    /// wrote. Only credentials of the same account are ever saved or
    /// reported. Both having rotated is an error, since only one rotation can
    /// be valid.
    /// Credentials holding a refresh token this process knows is spent are
    /// neither reported nor saved.
    fn reconcile(
        &mut self,
        in_client: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<Reconciled, Error> {
        let known = self.credentials()?;
        let (current, in_file) = read_credentials(&self.path)?;
        if in_file.account_id != in_client.account_id {
            return Err(different_account(&self.path));
        }
        if current != self.snapshot {
            if in_file == *in_client {
                self.snapshot = current;
                return Ok(Reconciled::Unchanged);
            }
            if in_file != known {
                if *in_client == known {
                    if spent_tokens::credentials_spent(&in_file) {
                        return Err(Error::AuthenticationFailed(format!(
                            "{} holds a ChatGPT OAuth refresh token that {SPENT}. Not loading it (using it again would sign you out)",
                            self.path.display()
                        )));
                    }
                    return Ok(Reconciled::Newer {
                        credentials: in_file,
                        bytes: current,
                    });
                }
                return Err(Error::AuthenticationFailed(format!(
                    "Both {} and this client rotated ChatGPT OAuth credentials. Sign in again",
                    self.path.display()
                )));
            }
            // Only formatting or other fields changed: not a rotation.
            self.snapshot = current;
        }
        if *in_client == known {
            return Ok(Reconciled::Unchanged);
        }
        if spent_tokens::credentials_spent(in_client) {
            return Err(Error::AuthenticationFailed(format!(
                "this client holds a ChatGPT OAuth refresh token that {SPENT}. Not saving it to {}",
                self.path.display()
            )));
        }
        self.persist_tokens(in_client, id_token)?;
        Ok(Reconciled::Saved)
    }

    /// Reconciles the file with `client`'s ChatGPT OAuth credentials. Writes
    /// credentials the client rotated, or loads newer credentials another
    /// process wrote into the client. Fails when both changed, since only one
    /// rotation can be valid, when the file holds a different account, and
    /// when the credentials to write or load hold a refresh token this
    /// process knows is spent (nothing is then written or loaded). Clients
    /// without ChatGPT OAuth are left alone. Credentials a client of this
    /// file rotated but couldn't save are written with their refresh's
    /// `id_token`.
    ///
    /// A client from [`client`](Self::client) or
    /// [`openai_client`](Self::openai_client) keeps the file in step on its
    /// own. `sync` is for clients configured some other way.
    pub async fn sync(&mut self, client: &LLMClient) -> Result<CodexAuthSync, Error> {
        let Some(in_client) = client.openai_chatgpt_oauth_credentials().await else {
            return Ok(CodexAuthSync::Unchanged);
        };
        // A rotation a store of this file couldn't save: saved here with the
        // `id_token` its refresh returned.
        let shared = shared_file(&self.path);
        let rotation = shared
            .unsaved(&in_client.account_id)
            .filter(|rotation| rotation.credentials == in_client);
        let id_token = rotation
            .as_ref()
            .and_then(|rotation| rotation.id_token.as_deref());
        match self.reconcile(&in_client, id_token)? {
            Reconciled::Unchanged => Ok(CodexAuthSync::Unchanged),
            Reconciled::Saved => {
                if let Some(rotation) = &rotation {
                    shared.done_with(rotation);
                }
                Ok(CodexAuthSync::Saved)
            }
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
                    "Both {} and this client rotated ChatGPT OAuth credentials. Sign in again",
                    self.path.display()
                )))
            }
        }
    }
}

/// A [`CodexAuthFile`] as the [`ChatGptOAuthStore`] of an OpenAI client.
/// Before every refresh the file is read again, and tokens Codex refreshed
/// are used instead of refreshing (for the same account only, a file now
/// holding another account stops the refresh). Rotated tokens, with the
/// refresh's `id_token`, are written back as soon as a refresh succeeds
/// (backup, atomic replace, never over newer tokens).
///
/// This process never sends a refresh token it knows is spent, and neither
/// the client nor its store adopts or saves credentials holding one. A token
/// is spent once a refresh with it returned a new one, or the server refused
/// it as reused, expired or revoked. It is then recorded (as a digest, in
/// one record for the whole process), which the rows below and the client
/// check before every refresh request. A refresh whose answer was lost may
/// have spent a token without the record knowing (the next refresh
/// sends it, the server refuses it, and then it is recorded), and a token
/// spent by an earlier run of the process or by another one looks live. An
/// old `auth.json` restored from a backup is then adopted like new
/// credentials, and drops an unsaved rotation of its account (which stays in
/// its recovery file).
///
/// When rotated tokens can't be written back, they go to a private
/// recovery file next to `auth.json` (see [`CodexAuthFile::write_recovery`])
/// and the refresh returns an error naming it. The client keeps using the
/// new tokens. Every store of the file in this process then knows of this
/// *unsaved rotation* (the latest one of each account). It is kept (also
/// when no store of the file is left) until the file holds live credentials
/// of its account, either the rotation itself (saved by a store, or put in
/// place from the recovery file), or others (the user signed in again). Only
/// the client that made it holds its tokens. They are never handed to another
/// client before they are in the file.
///
/// Before a refresh of credentials `C`, the first row matching what the file
/// holds applies (an unsaved rotation of another account than `C`'s plays no
/// part):
///
/// | The file holds | Then |
/// |---|---|
/// | nothing readable (deleted by `codex logout`, say), or another account | The refresh stops with that error, which names the recovery file of an unsaved rotation of `C`'s account. |
/// | spent credentials, and a rotation is unsaved | The rotation is saved over them, and the row for the rotation applies. While it can't be saved, every refresh stops with an error naming its recovery file (so the client holding it doesn't rotate it again before it is saved). |
/// | spent credentials, and no rotation is unsaved | `C` is saved over them and refreshed. If `C` is spent too (or has no refresh token), the refresh stops (sign in again). |
/// | `C` | `C` is refreshed. |
/// | other live credentials | They are adopted instead of refreshing. Codex refreshed, the user signed in again, or `C` was never in the file (credentials the host installed, say, they are not saved over live ones). |
///
/// Every store of the same file (by canonical path) in this process shares
/// one [`refresh_lock`](ChatGptOAuthStore::refresh_lock), so clients built
/// separately from one `auth.json` (each [`CodexAuthFile::openai_client`]
/// call makes its own store) refresh one at a time. The second finds the
/// first one's tokens in the file instead of reusing the rotated refresh
/// token. Each store keeps its own record of the file as last read.
pub struct CodexAuthStore {
    file: Mutex<CodexAuthFile>,
    shared: Arc<SharedFile>,
}

/// What every [`CodexAuthStore`] of one `auth.json` in this process shares.
struct SharedFile {
    /// The file's canonical path: its key in [`SHARED_FILES`] and for its
    /// spent tokens.
    key: PathBuf,
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
    /// Rotated credentials a store could not save: the latest of each
    /// account, by account id.
    unsaved: Mutex<BTreeMap<String, UnsavedRotation>>,
}

#[derive(Clone)]
struct UnsavedRotation {
    credentials: ChatGptOAuthCredentials,
    /// The refresh's `id_token`, saved with the credentials.
    id_token: Option<String>,
    /// The recovery file the rotation was written to, if any.
    recovery: Option<PathBuf>,
}

impl SharedFile {
    fn unsaved_slot(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, UnsavedRotation>> {
        // Each update inserts or removes a whole entry: valid after any panic.
        self.unsaved
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The unsaved rotation of `account_id`, if any.
    fn unsaved(&self, account_id: &str) -> Option<UnsavedRotation> {
        self.unsaved_slot().get(account_id).cloned()
    }

    /// Records `rotation` as its account's unsaved rotation, in place of an
    /// earlier one, and keeps this state for the life of the process until
    /// it is done with, so that a client made from the file later, when no
    /// store of it is left, still finds it.
    fn keep_unsaved(self: &Arc<Self>, rotation: UnsavedRotation) {
        let mut unsaved = self.unsaved_slot();
        unsaved.insert(rotation.credentials.account_id.clone(), rotation);
        // Under the slot's lock, so keeping always follows the slot.
        self.keep(true);
    }

    /// Forgets `rotation` (unless another replaced it meanwhile).
    fn done_with(self: &Arc<Self>, rotation: &UnsavedRotation) {
        let mut unsaved = self.unsaved_slot();
        let account_id = &rotation.credentials.account_id;
        if unsaved
            .get(account_id)
            .is_some_and(|unsaved| unsaved.credentials == rotation.credentials)
        {
            unsaved.remove(account_id);
            self.keep(!unsaved.is_empty());
        }
    }

    fn keep(self: &Arc<Self>, kept: bool) {
        let mut files = shared_files();
        // By pointer: an entry of a dropped state for the same path may
        // still be listed.
        if let Some(entry) = files
            .iter_mut()
            .find(|entry| std::ptr::eq(entry.shared.as_ptr(), Arc::as_ptr(self)))
        {
            entry.kept = kept.then(|| self.clone());
        }
    }
}

/// An `auth.json` in [`SHARED_FILES`].
struct SharedEntry {
    key: PathBuf,
    shared: Weak<SharedFile>,
    /// Set while the file has an unsaved rotation. See
    /// [`SharedFile::keep_unsaved`].
    kept: Option<Arc<SharedFile>>,
}

fn shared_files() -> std::sync::MutexGuard<'static, Vec<SharedEntry>> {
    // The list is valid after any panic: it only ever gains or drops whole
    // entries, or sets a field.
    SHARED_FILES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The shared state of the `auth.json` files with a [`CodexAuthStore`] in
/// this process, by canonical path. An entry lives as long as a store uses
/// it, or while the file has an unsaved rotation. Dead entries are dropped
/// when a store is made. (Locks: a store's own lock, then a file's unsaved
/// rotation, then this list, never the other way round.)
static SHARED_FILES: Mutex<Vec<SharedEntry>> = Mutex::new(Vec::new());

/// The key of the `auth.json` at `path` in this process: its canonical path
/// (its absolute path while it doesn't exist).
fn file_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path)
        .or_else(|_| std::path::absolute(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

fn shared_file(path: &Path) -> Arc<SharedFile> {
    let key = file_key(path);
    let mut files = shared_files();
    files.retain(|entry| entry.shared.strong_count() > 0);
    if let Some(shared) = files
        .iter()
        .find(|entry| entry.key == key)
        .and_then(|entry| entry.shared.upgrade())
    {
        return shared;
    }
    let shared = Arc::new(SharedFile {
        key: key.clone(),
        refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        unsaved: Mutex::new(BTreeMap::new()),
    });
    files.push(SharedEntry {
        key,
        shared: Arc::downgrade(&shared),
        kept: None,
    });
    shared
}

/// Why a spent refresh token can't be used, for error messages.
const SPENT: &str = "can't be used again (this process exchanged it for a new one, or the server refused it as reused, expired or revoked)";

/// The error for credentials holding a refresh token this process knows is
/// spent: they are neither sent nor adopted. With the credentials'
/// `account_id`, names the recovery file of that account's unsaved rotation
/// of an `auth.json`, when there is one.
pub(crate) fn spent_refresh_token_error(account_id: Option<&str>) -> Error {
    let files: Vec<Arc<SharedFile>> = shared_files()
        .iter()
        .filter_map(|entry| entry.shared.upgrade())
        .collect();
    let unsaved = account_id.and_then(|account_id| {
        files.iter().find_map(|shared| {
            let rotation = shared.unsaved(account_id)?;
            Some((shared.key.clone(), rotation.recovery))
        })
    });
    let refused = format!(
        "Not using a ChatGPT OAuth refresh token that {SPENT}. Using it again would sign you out"
    );
    Error::AuthenticationFailed(match unsaved {
        Some((path, Some(recovery))) => format!(
            "{refused}. The newest tokens for {path} could not be saved there. They are in {} ({RECOVERY_FILE_PRIVACY}). Put it in place of {path}, or sign in to Codex again, before Codex next refreshes",
            recovery.display(),
            path = path.display(),
        ),
        Some((path, None)) => format!(
            "{refused}. The newest tokens for {} could not be saved. Sign in to Codex again",
            path.display()
        ),
        None => format!("{refused}. Use the tokens that refresh returned, or sign in again"),
    })
}

impl CodexAuthStore {
    pub fn new(file: CodexAuthFile) -> Self {
        Self {
            shared: shared_file(&file.path),
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
        self.file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn different_account(path: &Path) -> Error {
    Error::AuthenticationFailed(format!(
        "{} holds a different ChatGPT account than this client (signed out, or signed in to another account). Sign in again",
        path.display()
    ))
}

/// `error`, which stops a refresh while `rotation` of the client's account
/// is unsaved, with where the rotation is. (No advice to put it in place: the
/// file may hold another account, or be gone, on purpose.)
fn with_unsaved_rotation(error: Error, path: &Path, rotation: &UnsavedRotation) -> Error {
    let path = path.display();
    Error::AuthenticationFailed(match &rotation.recovery {
        Some(recovery) => format!(
            "{error}. Refreshed ChatGPT OAuth tokens of this account that could not be saved to {path} are in {} ({RECOVERY_FILE_PRIVACY})",
            recovery.display()
        ),
        None => format!(
            "{error}. Refreshed ChatGPT OAuth tokens of this account could not be saved to {path}, nor to a recovery file"
        ),
    })
}

/// The error for refreshed tokens that could not be saved to `path` because
/// of `error`, naming the recovery file they went to, if any.
fn unsaved_error(path: &Path, error: &Error, recovery: Option<&Path>) -> Error {
    let path = path.display();
    Error::AuthenticationFailed(match recovery {
        Some(recovery) => format!(
            "refreshed ChatGPT OAuth tokens could not be saved to {path} ({error}). They are in {} ({RECOVERY_FILE_PRIVACY}). Put it in place of {path}, or sign in to Codex again, before Codex next refreshes",
            recovery.display()
        ),
        None => format!(
            "refreshed ChatGPT OAuth tokens could not be saved to {path} ({error}). Sign in to Codex again"
        ),
    })
}

/// Where the recovery file message says the file's privacy stands.
#[cfg(unix)]
const RECOVERY_FILE_PRIVACY: &str = "readable only by you";
#[cfg(not(unix))]
const RECOVERY_FILE_PRIVACY: &str = "keep this credential file private";

impl std::fmt::Debug for CodexAuthStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CodexAuthStore")
            .field("path", &self.path())
            .finish_non_exhaustive()
    }
}

impl ChatGptOAuthStore for CodexAuthStore {
    /// See the table in the [`CodexAuthStore`] docs.
    fn before_refresh(
        &self,
        current: &ChatGptOAuthCredentials,
    ) -> Result<Option<ChatGptOAuthCredentials>, Error> {
        let mut file = self.lock();
        let path = file.path.clone();
        let spent = spent_tokens::credentials_spent;
        let unsaved = self.shared.unsaved(&current.account_id);
        // While a rotation is unsaved, whatever stops the refresh names its
        // recovery file; the rotation is kept.
        let stop = |error: Error| match &unsaved {
            Some(rotation) => with_unsaved_rotation(error, &path, rotation),
            None => error,
        };
        let (bytes, mut in_file) = read_credentials(&path).map_err(stop)?;
        if in_file.account_id != current.account_id {
            return Err(stop(different_account(&path)));
        }
        file.snapshot = bytes;
        if let Some(rotation) = unsaved {
            if spent(&in_file) {
                file.persist_tokens(&rotation.credentials, rotation.id_token.as_deref())
                    .map_err(|error| unsaved_error(&path, &error, rotation.recovery.as_deref()))?;
                in_file = rotation.credentials.clone();
            }
            // The file holds the rotation now, or other live credentials.
            self.shared.done_with(&rotation);
        }
        if spent(&in_file) {
            // Dead credentials in the file: this client's replace them, if
            // they can still be refreshed.
            if current.refresh_token.is_none() || spent(current) {
                return Err(Error::AuthenticationFailed(format!(
                    "{} holds a ChatGPT OAuth refresh token that {SPENT}, and this client holds no newer one (using it again would sign you out). Sign in to Codex again",
                    path.display()
                )));
            }
            file.persist_tokens(current, None)?;
            return Ok(None);
        }
        if in_file == *current {
            return Ok(None);
        }
        Ok(Some(in_file))
    }

    fn save_refreshed(
        &self,
        refreshed: &ChatGptOAuthCredentials,
        id_token: Option<&str>,
    ) -> Result<(), Error> {
        let mut file = self.lock();
        // The refresh replaced the credentials `before_refresh` let through:
        // the file's, as this store last read or saved them.
        if let Ok(replaced) = file.credentials()
            && let Some(spent) = replaced.refresh_token.as_deref()
            && refreshed.refresh_token.as_deref() != Some(spent)
        {
            spent_tokens::record(spent);
        }
        let error = match file.persist_tokens(refreshed, id_token) {
            Ok(_) => return Ok(()),
            Err(error) => error,
        };
        let (recovery, failure) = match file.write_recovery_tokens(refreshed, id_token) {
            Ok(recovery) => {
                let failure = unsaved_error(&file.path, &error, Some(&recovery));
                (Some(recovery), failure)
            }
            Err(recovery_error) => (
                None,
                Error::AuthenticationFailed(format!(
                    "refreshed ChatGPT OAuth tokens could not be saved to {} ({error}) nor to a recovery file ({recovery_error}). Sign in to Codex again",
                    file.path.display()
                )),
            ),
        };
        self.shared.keep_unsaved(UnsavedRotation {
            credentials: refreshed.clone(),
            id_token: id_token
                .filter(|id_token| !id_token.is_empty())
                .map(str::to_string),
            recovery,
        });
        log::error!("arko_ai: {failure}");
        Err(failure)
    }

    fn refresh_lock(&self) -> Option<Arc<tokio::sync::Mutex<()>>> {
        Some(self.shared.refresh_lock.clone())
    }
}

/// Reads the file and the credentials in it. A file that is not valid JSON
/// is read again, up to [`SETTLE_READS`] times, before the error is
/// returned: Codex writes `auth.json` in place (truncate, then write), so a
/// read can land between the two.
///
/// The waits between reads block the calling thread, which may be an async
/// runtime's (the [`ChatGptOAuthStore`] calls are synchronous and run inside
/// a refresh). They are bounded like the file I/O around them: they only
/// follow a read that could not be parsed, and add up to at most
/// `(SETTLE_READS - 1) * SETTLE_DELAY` (120 ms) per call.
/// (`tokio::task::block_in_place` would not work on a current-thread
/// runtime.)
fn read_credentials(path: &Path) -> Result<(Vec<u8>, ChatGptOAuthCredentials), Error> {
    let mut reads = 0;
    loop {
        #[cfg(test)]
        BEFORE_READ.with(|hook| {
            if let Some(hook) = hook.borrow_mut().as_mut() {
                hook(path);
            }
        });
        let bytes = std::fs::read(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::IoError(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("No Codex login at {}. Run `codex login`", path.display()),
                ))
            } else {
                Error::IoError(error)
            }
        })?;
        reads += 1;
        match parse_credentials(&bytes, path) {
            Ok(credentials) => return Ok((bytes, credentials)),
            // A short, bounded blocking wait; see above.
            Err(Error::JsonError(_)) if reads < SETTLE_READS => std::thread::sleep(SETTLE_DELAY),
            Err(error) => return Err(error),
        }
    }
}

fn parse_credentials(bytes: &[u8], path: &Path) -> Result<ChatGptOAuthCredentials, Error> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| {
        Error::JsonError(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Could not parse Codex login at {}: {error}", path.display()),
        )))
    })?;
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

/// Creates the file `path` for credentials, failing if it exists. On Unix
/// only the owner can read it from the start (mode 0600), so no other user
/// can open it before `fill_file` gives it the original file's permissions.
fn create_private(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)
}

/// Replaces the backup `path` with `bytes`. The previous backup is removed
/// rather than rewritten, since a file's creation mode applies only to new
/// files.
fn write_backup(
    path: &Path,
    bytes: &[u8],
    permissions: &std::fs::Permissions,
) -> Result<(), Error> {
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
        _ => {}
    }
    let mut file = create_private(path)?;
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
    let mut file = create_private(path)?;
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
/// crash. Windows cannot open a directory as a file. There this does nothing.
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
                CodexAuthFile::load(recovery)
                    .unwrap()
                    .credentials()
                    .unwrap(),
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
        assert!(
            matches!(result, Err(Error::AuthenticationFailed(_))),
            "{result:?}"
        );
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
        assert!(
            error.to_string().contains("different ChatGPT account"),
            "{error}"
        );
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
    fn the_store_adopts_codex_rotations_and_refuses_other_accounts() {
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

        // Another account in the file stops the refresh and changes nothing.
        std::fs::write(&path, other_account()).unwrap();
        let error = store.before_refresh(&newer).unwrap_err();
        assert!(matches!(error, Error::AuthenticationFailed(_)));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), other_account());
        // A file that can't be read stops it too.
        std::fs::write(&path, "{not json").unwrap();
        assert!(store.before_refresh(&newer).is_err());
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
        for secret in [
            "access-3",
            "refresh-3",
            "id-3",
            "account-1",
            "refresh-codex",
        ] {
            assert!(!error.contains(secret), "{error}");
        }
        let recovery = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.to_string_lossy().contains(".recovered-"))
            .unwrap();
        assert!(error.contains(&recovery.display().to_string()), "{error}");
        assert_eq!(
            CodexAuthFile::load(&recovery)
                .unwrap()
                .credentials()
                .unwrap(),
            refreshed
        );
        let debug = format!("{store:?}");
        assert!(
            debug.contains("auth.json") && !debug.contains("refresh"),
            "{debug}"
        );
    }

    #[test]
    fn stores_of_one_file_share_a_refresh_lock() {
        let (dir, path) = temp_file(NESTED);
        let first = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        // Another spelling of the same path.
        let second = CodexAuthStore::new(
            CodexAuthFile::load(dir.path().join(".").join("auth.json")).unwrap(),
        );
        let lock = first.refresh_lock().unwrap();
        assert!(Arc::ptr_eq(&lock, &second.refresh_lock().unwrap()));
        assert!(Arc::ptr_eq(&first.shared, &second.shared));
        let (_other_dir, other_path) = temp_file(NESTED);
        let other = CodexAuthStore::new(CodexAuthFile::load(&other_path).unwrap());
        assert!(!Arc::ptr_eq(&lock, &other.refresh_lock().unwrap()));

        // An entry goes once no store uses it.
        let registered = |path: &Path| {
            let key = std::fs::canonicalize(path).unwrap();
            SHARED_FILES
                .lock()
                .unwrap()
                .iter()
                .any(|entry| entry.key == key)
        };
        assert!(registered(&other_path));
        drop(other);
        let _ = shared_file(&path);
        assert!(!registered(&other_path));
        assert!(registered(&path));
    }

    fn set_before_read(hook: Option<ReadHook>) {
        BEFORE_READ.with(|slot| *slot.borrow_mut() = hook);
    }

    /// Codex writes `auth.json` in place (truncate, then write): a read can
    /// see it empty or half written.
    #[test]
    fn a_file_codex_is_still_writing_is_read_again() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let mut file = CodexAuthFile::load(&path).unwrap();
        let loaded = file.credentials().unwrap();
        for finished in [codex_refreshed(), NESTED.to_string()] {
            let persisting = finished == NESTED;
            let reads = std::rc::Rc::new(std::cell::Cell::new(0));
            let counted = reads.clone();
            set_before_read(Some(Box::new(move |target| {
                counted.set(counted.get() + 1);
                match counted.get() {
                    1 => std::fs::write(target, "").unwrap(),
                    2 => std::fs::write(target, &finished[..finished.len() / 2]).unwrap(),
                    3 => std::fs::write(target, &finished).unwrap(),
                    _ => {}
                }
            })));
            if persisting {
                // Persisting reads the file first too.
                let result = file.persist_tokens(&rotated(), None);
                set_before_read(None);
                assert_eq!(result.unwrap(), CodexAuthSync::Saved);
                assert_eq!(
                    CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
                    rotated()
                );
            } else {
                let result = store.before_refresh(&loaded);
                set_before_read(None);
                assert_eq!(result.unwrap().unwrap().access_token, "access-codex");
            }
            assert_eq!(reads.get(), 3);
        }

        // A file that stays invalid fails after a few reads.
        let reads = std::rc::Rc::new(std::cell::Cell::new(0));
        let counted = reads.clone();
        set_before_read(Some(Box::new(move |_| counted.set(counted.get() + 1))));
        std::fs::write(&path, "{\"tokens\":").unwrap();
        let result = CodexAuthFile::load(&path);
        set_before_read(None);
        assert!(matches!(result, Err(Error::JsonError(_))), "{result:?}");
        assert_eq!(reads.get(), SETTLE_READS);
    }

    #[test]
    fn a_rotation_saved_after_a_failed_save_keeps_its_id_token() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        // A directory where the backup goes makes the save fail.
        let backup = with_suffix(&path, ".bak");
        std::fs::create_dir(&backup).unwrap();
        assert!(store.save_refreshed(&rotated(), Some("id-2")).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);
        std::fs::remove_dir(&backup).unwrap();

        // The next refresh first saves the rotation, with the refresh's id_token.
        assert_eq!(store.before_refresh(&rotated()).unwrap(), None);
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");
        assert_eq!(saved["tokens"]["id_token"], "id-2");
    }

    /// A rotation that another store of the file couldn't save is handed to
    /// no one: while the file holds the spent tokens and can't be written,
    /// every refresh stops with an error naming the recovery file (the client
    /// holding the rotation doesn't rotate it again either). Once the file
    /// can be written, the next refresh saves it and the others take it.
    #[test]
    fn every_store_of_the_file_knows_of_an_unsaved_rotation_until_it_is_saved() {
        let (_dir, path) = temp_file(NESTED);
        let first = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let second = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        let saving_error = fail_save(&first, &path).to_string();
        assert!(saving_error.contains(".recovered-"), "{saving_error}");

        for (store, held) in [
            (&second, &loaded),
            (&first, &rotated()),
            (&second, &rotated()),
        ] {
            let error = store.before_refresh(held).unwrap_err().to_string();
            assert!(error.contains(".recovered-"), "{error}");
        }
        // A client of another account is told about its own account only.
        let other = ChatGptOAuthCredentials {
            account_id: "account-2".into(),
            ..loaded.clone()
        };
        let error = second.before_refresh(&other).unwrap_err().to_string();
        assert!(
            error.contains("different ChatGPT account") && !error.contains(".recovered-"),
            "{error}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);

        std::fs::remove_dir(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(second.before_refresh(&loaded).unwrap(), Some(rotated()));
        assert!(first.shared.unsaved_slot().is_empty());
        assert_eq!(first.before_refresh(&rotated()).unwrap(), None);
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");
        assert_eq!(saved["tokens"]["id_token"], "id-2");
    }

    /// Codex wrote a signed-in account's tokens where `NESTED` was.
    fn signed_in_again() -> String {
        NESTED
            .replace("access-1", "access-signin")
            .replace("refresh-1", "refresh-signin")
    }

    /// Fails `store`'s save of `rotated()` (a directory where the backup
    /// goes). Returns the error.
    fn fail_save(store: &CodexAuthStore, path: &Path) -> Error {
        std::fs::create_dir(with_suffix(path, ".bak")).unwrap();
        store.save_refreshed(&rotated(), Some("id-2")).unwrap_err()
    }

    #[test]
    fn the_client_holding_an_unsaved_rotation_takes_a_new_sign_in() {
        let (_dir, path) = temp_file(NESTED);
        let holder = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let other = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        fail_save(&holder, &path);
        // The user signed in again: the rotation, never saved, is dropped,
        // and every client takes the new credentials instead of refreshing.
        std::fs::write(&path, signed_in_again()).unwrap();
        let adopted = holder.before_refresh(&rotated()).unwrap().unwrap();
        assert_eq!(adopted.refresh_token.as_deref(), Some("refresh-signin"));
        assert!(holder.shared.unsaved_slot().is_empty());
        assert_eq!(
            other.before_refresh(&loaded).unwrap(),
            Some(adopted.clone())
        );
        assert_eq!(holder.before_refresh(&adopted).unwrap(), None);

        // Another account signed in is refused, and the rotation is kept.
        let (_dir, path) = temp_file(NESTED);
        let holder = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        fail_save(&holder, &path);
        std::fs::write(&path, other_account()).unwrap();
        let error = holder.before_refresh(&rotated()).unwrap_err().to_string();
        assert!(
            error.contains("different ChatGPT account") && error.contains(".recovered-"),
            "{error}"
        );
        assert!(!error.contains("put it in place"), "{error}");
        assert!(!holder.shared.unsaved_slot().is_empty());
    }

    #[test]
    fn an_unsaved_rotation_outlives_every_store_of_the_file() {
        let (_dir, path) = temp_file(NESTED);
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        let first = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        fail_save(&first, &path);
        let key = first.shared.key.clone();
        drop(first);
        // A client made from the file later doesn't refresh the spent
        // tokens; once the file can be written, it saves the rotation and
        // takes it.
        let later = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let error = later.before_refresh(&loaded).unwrap_err().to_string();
        assert!(error.contains(".recovered-"), "{error}");
        std::fs::remove_dir(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(later.before_refresh(&loaded).unwrap(), Some(rotated()));
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );
        // Once saved, the state is no longer kept.
        let kept = SHARED_FILES
            .lock()
            .unwrap()
            .iter()
            .any(|entry| entry.key == key && entry.kept.is_some());
        assert!(!kept);
    }

    /// `codex logout` deleted the file: every refresh stops, naming the
    /// recovery file, and the rotation is kept until the user signs in again.
    #[test]
    fn an_unreadable_file_stops_the_refresh_even_with_an_unsaved_rotation() {
        let (_dir, path) = temp_file(NESTED);
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        let first = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let second = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        fail_save(&first, &path);
        std::fs::remove_file(&path).unwrap();
        for (store, held) in [(&second, &loaded), (&first, &rotated())] {
            let error = store.before_refresh(held).unwrap_err().to_string();
            assert!(
                error.contains(".recovered-") && !error.contains("put it in place"),
                "{error}"
            );
        }
        assert!(!first.shared.unsaved_slot().is_empty());
        std::fs::write(&path, signed_in_again()).unwrap();
        let adopted = second.before_refresh(&loaded).unwrap().unwrap();
        assert_eq!(adopted.refresh_token.as_deref(), Some("refresh-signin"));
        assert_eq!(first.before_refresh(&rotated()).unwrap(), Some(adopted));
    }

    /// Credentials a client holds that were never in the file (installed by
    /// its host, say) give way to the file's live ones, also after another
    /// store's unsaved rotation is saved. They are never saved over them.
    #[test]
    fn credentials_the_file_never_held_give_way_to_its_live_ones() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        let host = ChatGptOAuthCredentials {
            access_token: "access-host".into(),
            refresh_token: Some("refresh-host".into()),
            account_id: "account-1".into(),
        };
        assert_eq!(store.before_refresh(&host).unwrap(), Some(loaded));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);

        let holder = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        fail_save(&holder, &path);
        let error = store.before_refresh(&host).unwrap_err().to_string();
        assert!(error.contains(".recovered-"), "{error}");
        std::fs::remove_dir(with_suffix(&path, ".bak")).unwrap();
        for _ in 0..2 {
            assert_eq!(store.before_refresh(&host).unwrap(), Some(rotated()));
        }
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );
    }

    /// B1: a client holding credentials older than the file's, both spent by
    /// another client's refreshes, is never given the file's spent ones.
    #[test]
    fn a_client_holding_older_credentials_never_takes_spent_ones_from_the_file() {
        let (_dir, path) = temp_file(NESTED);
        let first = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let idle = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        // The first client refreshes twice: saved, then not.
        first.save_refreshed(&rotated(), None).unwrap();
        let third = ChatGptOAuthCredentials {
            access_token: "access-3".into(),
            refresh_token: Some("refresh-3".into()),
            account_id: "account-1".into(),
        };
        // A directory in place of the first save's backup fails the second.
        std::fs::remove_file(with_suffix(&path, ".bak")).unwrap();
        std::fs::create_dir(with_suffix(&path, ".bak")).unwrap();
        assert!(first.save_refreshed(&third, None).is_err());
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );

        let error = idle.before_refresh(&loaded).unwrap_err().to_string();
        assert!(error.contains(".recovered-"), "{error}");
        std::fs::remove_dir(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(idle.before_refresh(&loaded).unwrap(), Some(third.clone()));
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            third
        );
    }

    /// A file put back to spent credentials (an old backup restored, say)
    /// gets the live credentials of the next client to refresh. A client
    /// holding only spent ones stops.
    #[test]
    fn a_file_holding_spent_credentials_gets_a_client_s_live_ones() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let other = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        store.save_refreshed(&rotated(), Some("id-2")).unwrap();
        std::fs::write(&path, NESTED).unwrap();
        let without_refresh_token = ChatGptOAuthCredentials {
            refresh_token: None,
            ..rotated()
        };
        for held in [&loaded, &without_refresh_token] {
            let error = other.before_refresh(held).unwrap_err();
            assert!(matches!(error, Error::AuthenticationFailed(_)), "{error}");
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);
        assert_eq!(store.before_refresh(&rotated()).unwrap(), None);
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            rotated()
        );
        assert_eq!(other.before_refresh(&loaded).unwrap(), Some(rotated()));
    }

    /// Each account keeps its own unsaved rotation: one made for another
    /// account while the file held it doesn't replace the first.
    #[test]
    fn unsaved_rotations_are_kept_per_account() {
        let (_dir, path) = temp_file(NESTED);
        let holder = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let idle = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        fail_save(&holder, &path);
        // Codex signed in to another account, whose client can't save its
        // refresh either.
        std::fs::write(&path, other_account()).unwrap();
        let other = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let other_rotated = ChatGptOAuthCredentials {
            access_token: "access-other-2".into(),
            refresh_token: Some("refresh-other-2".into()),
            account_id: "account-2".into(),
        };
        assert!(other.save_refreshed(&other_rotated, None).is_err());
        let first_recovery = {
            let unsaved = holder.shared.unsaved_slot();
            assert_eq!(unsaved.len(), 2);
            assert_eq!(unsaved["account-2"].credentials, other_rotated);
            unsaved["account-1"].recovery.clone().unwrap()
        };
        let error = holder.before_refresh(&rotated()).unwrap_err().to_string();
        assert!(
            error.contains(&format!("{} (", first_recovery.display())),
            "{error}"
        );

        // The file is back to the first account's spent credentials: its
        // rotation is saved and taken; the other account's is still kept.
        std::fs::write(&path, NESTED).unwrap();
        std::fs::remove_dir(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(idle.before_refresh(&loaded).unwrap(), Some(rotated()));
        let unsaved = holder.shared.unsaved_slot();
        assert_eq!(unsaved.keys().collect::<Vec<_>>(), ["account-2"]);
    }

    /// A token spent anywhere in the process (by a client without a store,
    /// say) counts: the store never offers credentials holding it, and
    /// replaces them in the file with the client's live ones.
    #[test]
    fn the_store_knows_of_tokens_spent_anywhere_in_the_process() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let host = ChatGptOAuthCredentials {
            access_token: "access-host".into(),
            refresh_token: Some("refresh-host".into()),
            account_id: "account-1".into(),
        };
        let loaded = CodexAuthFile::load(&path).unwrap().credentials().unwrap();
        assert_eq!(store.before_refresh(&host).unwrap(), Some(loaded.clone()));
        spent_tokens::record("refresh-1");
        assert_eq!(store.before_refresh(&host).unwrap(), None);
        assert_eq!(
            CodexAuthFile::load(&path).unwrap().credentials().unwrap(),
            host
        );
        // A client whose own credentials are spent too stops.
        std::fs::write(&path, NESTED).unwrap();
        let error = store.before_refresh(&loaded).unwrap_err();
        assert!(error.to_string().contains("can't be used again"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);
    }

    /// `sync` on a client of the file whose rotation couldn't be saved
    /// writes it with the `id_token` its refresh returned.
    #[tokio::test]
    async fn sync_saves_an_unsaved_rotation_with_its_id_token() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client = file.client().unwrap();
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        fail_save(&store, &path);
        client
            .update_openai_auth(OpenAIAuth::ChatGptOAuth(rotated()))
            .await
            .unwrap();
        std::fs::remove_dir(with_suffix(&path, ".bak")).unwrap();
        assert_eq!(file.sync(&client).await.unwrap(), CodexAuthSync::Saved);
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"]["refresh_token"], "refresh-2");
        assert_eq!(saved["tokens"]["id_token"], "id-2");
        assert!(store.shared.unsaved_slot().is_empty());
    }

    /// A relative path is made absolute when the file is loaded, so a later
    /// change of directory doesn't move the file its clients use.
    #[cfg(unix)]
    #[test]
    fn a_relative_path_is_made_absolute_when_loaded() {
        let (_dir, path) = temp_file(NESTED);
        let cwd = std::env::current_dir().unwrap();
        let mut relative = PathBuf::new();
        for _ in cwd.components().skip(1) {
            relative.push("..");
        }
        relative.push(path.strip_prefix("/").unwrap());
        assert!(relative.is_relative());
        let file = CodexAuthFile::load(&relative).unwrap();
        assert!(file.path().is_absolute(), "{}", file.path().display());
        assert_eq!(
            std::fs::canonicalize(file.path()).unwrap(),
            std::fs::canonicalize(&path).unwrap()
        );
    }

    /// The unsaved rotation keeps its own file state alive, even while an
    /// entry of a dropped state for the same path is still listed.
    #[test]
    fn an_unsaved_rotation_keeps_its_own_state_alive() {
        let (_dir, path) = temp_file(NESTED);
        let store = CodexAuthStore::new(CodexAuthFile::load(&path).unwrap());
        let key = store.shared.key.clone();
        let dropped = Arc::new(SharedFile {
            key: key.clone(),
            refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
            unsaved: Mutex::new(BTreeMap::new()),
        });
        shared_files().insert(
            0,
            SharedEntry {
                key: key.clone(),
                shared: Arc::downgrade(&dropped),
                kept: None,
            },
        );
        drop(dropped);
        fail_save(&store, &path);
        let files = shared_files();
        let kept: Vec<_> = files
            .iter()
            .filter(|entry| entry.key == key && entry.kept.is_some())
            .collect();
        assert_eq!(kept.len(), 1);
        assert!(std::ptr::eq(
            kept[0].shared.as_ptr(),
            Arc::as_ptr(&store.shared)
        ));
    }

    /// `sync` never loads or saves credentials holding a refresh token this
    /// process already exchanged for a new one.
    #[tokio::test]
    async fn sync_never_loads_or_saves_spent_credentials() {
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client =
            LLMClient::new().with_openai_chatgpt_oauth("access-1", Some("refresh-1"), "account-1");
        spent_tokens::record("refresh-codex");
        std::fs::write(&path, codex_refreshed()).unwrap();
        let error = file.sync(&client).await.unwrap_err();
        assert!(error.to_string().contains("can't be used again"), "{error}");
        let held = client.openai_chatgpt_oauth_credentials().await.unwrap();
        assert_eq!(held.refresh_token.as_deref(), Some("refresh-1"));

        // A client holding spent credentials: they are not saved.
        let (_dir, path) = temp_file(NESTED);
        let mut file = CodexAuthFile::load(&path).unwrap();
        let client = LLMClient::new().with_openai_chatgpt_oauth(
            "access-9",
            Some("sync-spent-refresh"),
            "account-1",
        );
        spent_tokens::record("sync-spent-refresh");
        let error = file.sync(&client).await.unwrap_err();
        assert!(error.to_string().contains("can't be used again"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), NESTED);
    }

    #[cfg(unix)]
    #[test]
    fn backups_and_temporary_files_are_created_private() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        let (dir, path) = temp_file(NESTED);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        // An earlier backup readable by everyone is replaced, not reused.
        let backup = with_suffix(&path, ".bak");
        std::fs::write(&backup, "old backup").unwrap();
        std::fs::set_permissions(&backup, std::fs::Permissions::from_mode(0o644)).unwrap();
        let mut file = CodexAuthFile::load(&path).unwrap();
        assert_eq!(file.persist(&rotated()).unwrap(), CodexAuthSync::Saved);
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), NESTED);
        assert_eq!(mode(&backup), 0o600);
        assert_eq!(mode(&path), 0o600);

        let created = dir.path().join("created");
        drop(create_private(&created).unwrap());
        assert_eq!(mode(&created), 0o600);
        assert!(
            create_private(&created).is_err(),
            "never opens an existing file"
        );
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
