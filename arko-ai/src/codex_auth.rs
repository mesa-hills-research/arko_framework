//! Reading and safely updating a Codex CLI `auth.json` (ChatGPT OAuth).
//!
//! ChatGPT OAuth refresh tokens rotate: once a refresh succeeds, the previous
//! refresh token stops working. When arko refreshes credentials loaded from the
//! file Codex also uses, the new tokens must be written back, or Codex is left
//! with a dead refresh token. [`CodexAuthFile`] loads the file, writes rotated
//! tokens back atomically (after a backup), and never overwrites tokens that
//! another process (such as Codex) wrote in the meantime.

use crate::{
    client::LLMClient,
    error::Error,
    providers::openai::{ChatGptOAuthCredentials, OpenAIAuth},
};
use serde_json::Value;
use std::path::{Path, PathBuf};
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
#[derive(Debug, Clone)]
pub struct CodexAuthFile {
    path: PathBuf,
    /// The file's bytes as last read or written by this value.
    snapshot: Vec<u8>,
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

    /// An [`LLMClient`] using the file's ChatGPT OAuth credentials.
    pub fn client(&self) -> Result<LLMClient, Error> {
        let credentials = self.credentials()?;
        Ok(LLMClient::new().with_openai_chatgpt_oauth(
            &credentials.access_token,
            credentials.refresh_token.as_deref(),
            &credentials.account_id,
        ))
    }

    /// Writes `credentials` to the file, keeping every other field and the
    /// file's layout. The previous contents go to `auth.json.bak` and the new
    /// contents are written atomically (temporary file, then rename).
    ///
    /// Fails without writing when the file changed since it was loaded or last
    /// written and now holds different credentials; newer credentials written by
    /// another process are never overwritten.
    pub fn persist(
        &mut self,
        credentials: &ChatGptOAuthCredentials,
    ) -> Result<CodexAuthSync, Error> {
        let current = std::fs::read(&self.path)?;
        if current != self.snapshot {
            if parse_credentials(&current, &self.path)? == *credentials {
                self.snapshot = current;
                return Ok(CodexAuthSync::Unchanged);
            }
            return Err(Error::AuthenticationFailed(format!(
                "{} changed since it was loaded; refusing to overwrite newer credentials",
                self.path.display()
            )));
        }
        if self.credentials()? == *credentials {
            return Ok(CodexAuthSync::Unchanged);
        }

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
        tokens["account_id"] = Value::String(credentials.account_id.clone());
        if nested || value.get("last_refresh").is_some() {
            value["last_refresh"] = Value::String(rfc3339_utc(SystemTime::now()));
        }
        let mut bytes = serde_json::to_vec_pretty(&value)?;
        bytes.push(b'\n');

        let target = std::fs::canonicalize(&self.path)?;
        let permissions = std::fs::metadata(&target)?.permissions();
        let backup = with_suffix(&target, ".bak");
        write_file(&backup, &self.snapshot, &permissions)?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let temporary = with_suffix(&target, &format!(".tmp-{}-{nanos}", std::process::id()));
        write_file(&temporary, &bytes, &permissions)?;
        if let Err(error) = std::fs::rename(&temporary, &target) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.into());
        }
        self.snapshot = bytes;
        Ok(CodexAuthSync::Saved)
    }

    /// Reconciles the file with `client`'s ChatGPT OAuth credentials: writes
    /// credentials the client rotated, or loads newer credentials another
    /// process wrote into the client. Fails when both changed, since only one
    /// rotation can be valid. Clients without ChatGPT OAuth are left alone.
    pub async fn sync(&mut self, client: &LLMClient) -> Result<CodexAuthSync, Error> {
        let Some(in_client) = client.openai_chatgpt_oauth_credentials().await else {
            return Ok(CodexAuthSync::Unchanged);
        };
        let known = self.credentials()?;
        let current = std::fs::read(&self.path)?;
        if current != self.snapshot {
            let in_file = parse_credentials(&current, &self.path)?;
            if in_file == in_client {
                self.snapshot = current;
                return Ok(CodexAuthSync::Unchanged);
            }
            if in_client == known {
                client
                    .update_openai_auth(OpenAIAuth::ChatGptOAuth(in_file))
                    .await?;
                self.snapshot = current;
                return Ok(CodexAuthSync::Reloaded);
            }
            return Err(Error::AuthenticationFailed(format!(
                "Both {} and this client rotated ChatGPT OAuth credentials; sign in again",
                self.path.display()
            )));
        }
        if in_client == known {
            return Ok(CodexAuthSync::Unchanged);
        }
        self.persist(&in_client)
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
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    // Credentials: match the original file's permissions before writing.
    file.set_permissions(permissions.clone())?;
    file.write_all(bytes)?;
    file.sync_all()?;
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
    fn timestamps_use_codex_format() {
        let at = UNIX_EPOCH + Duration::new(1_790_577_224, 750_634_434);
        assert_eq!(rfc3339_utc(at), "2026-09-28T06:33:44.750634434Z");
        assert_eq!(rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00.000000000Z");
    }
}
