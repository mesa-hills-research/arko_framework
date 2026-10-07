//! Failover between ChatGPT accounts: an ordered list of ChatGPT OAuth
//! sign-ins (for example two Codex `auth.json` files) behind one OpenAI
//! provider. Requests go to the first account that can serve them; one that
//! reaches its usage limit is set aside until it resets, or until a status
//! check finds it usable again, and the request goes to the next account at
//! once.
//!
//! What the ChatGPT backend does, from openai/codex (`codex-rs`, ac9b5b8):
//!
//! - A used-up plan answers HTTP 429 with `error.type: "usage_limit_reached"`
//!   and the reset time in `resets_at` (`codex-api/src/api_bridge.rs`), and
//!   credit or spend walls with codes such as `credit_balance_exhausted` or
//!   `usage_not_included`. Codex stops the turn on either; no retry helps.
//! - Credits need no opt-in: Codex sends no flag, and counts an account
//!   whose credits are left (or unlimited) as usable when its plan's windows
//!   are used up (`tui/src/chatwidget/backend_banners.rs`,
//!   `update_backend_banner`). So an account with credits keeps serving
//!   until the backend refuses it.
//! - The usage endpoint, `GET https://chatgpt.com/backend-api/wham/usage`
//!   with the bearer token and `ChatGPT-Account-Id`
//!   (`backend-client/src/client/rate_limit_resets.rs`,
//!   `get_rate_limit_status`), reports `rate_limit.allowed`, the windows,
//!   the credits, spend control and the reached-limit type; a manual usage
//!   reset shows there at once.
//! - The `x-codex-turn-state` routing token belongs to the account that
//!   issued it: Codex drops it when the signed-in account changes
//!   (`core/src/client.rs`, the `owner_changed` reset). Codex has no
//!   evidence either way on whether one account accepts another's encrypted
//!   reasoning (it never switches accounts within a conversation and handles
//!   no encrypted-content error), so a switch takes the safe path here: a
//!   request replays only the reasoning the serving account produced.

use super::{OpenAIAuth, OpenAIClient};
use crate::{
    codex_auth::CodexAuthFile,
    config::GenerationConfig,
    conversation::Conversation,
    error::{Error, ErrorKind},
    providers::{GenerationResponse, ProviderClient, StreamEvent},
    reasoning::ReasoningArtifact,
    usage::{PlanCredits, PlanUsage, PlanUsageWindow},
};
use futures::{
    StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, SystemTime},
};

/// The ChatGPT backend's usage endpoint, as Codex reads it.
pub const CHATGPT_USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";

/// How often an exhausted account ranked above the one serving is checked
/// on the usage endpoint, by default.
pub const DEFAULT_STATUS_CHECK_INTERVAL: Duration = Duration::from_secs(5 * 60);

/// An account's access token is refreshed before use when it expires within
/// this margin (the arko_agents runtime's margin).
const REFRESH_MARGIN: Duration = Duration::from_secs(5 * 60);

/// How long one usage-endpoint request may take.
const USAGE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// The most reasoning items and turn-state tokens whose account is
/// remembered (about 40 MB); past it the record starts over.
const MAX_ORIGINS: usize = 1 << 20;

/// One ChatGPT OAuth account of a [`ChatGptAccounts`] list: a label (used in
/// events, logs and [`ModelCallUsage::account`](crate::usage::ModelCallUsage::account);
/// never a secret) and the client holding its credentials, which refreshes
/// them on its own (with a Codex `auth.json`, kept in step with the file).
#[derive(Clone)]
pub struct ChatGptAccount {
    label: String,
    client: OpenAIClient,
}

impl fmt::Debug for ChatGptAccount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatGptAccount")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl ChatGptAccount {
    /// An account using `client`, which must use ChatGPT OAuth.
    pub fn new(label: impl Into<String>, client: OpenAIClient) -> Self {
        Self {
            label: label.into(),
            client,
        }
    }

    /// An account using the credentials in a Codex `auth.json`, kept in step
    /// with the file (see [`CodexAuthFile::openai_client`]).
    pub fn from_codex_auth_file(
        label: impl Into<String>,
        file: &CodexAuthFile,
    ) -> Result<Self, Error> {
        Ok(Self::new(label, file.openai_client()?))
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn client(&self) -> &OpenAIClient {
        &self.client
    }
}

/// Something that happened to the accounts of a [`ChatGptAccounts`] list,
/// passed to its [observer](ChatGptAccounts::with_observer) and logged.
/// `Display` writes a sentence fit for a notification; no event holds a
/// secret.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChatGptAccountEvent {
    /// The backend refused `account`: its usage limit (or its credits) ran
    /// out. It is set aside until `resets_at`, when known, or until a status
    /// check finds it usable.
    Exhausted {
        account: String,
        resets_at: Option<SystemTime>,
        /// The error's code, such as `usage_limit_reached`.
        code: Option<String>,
    },
    /// Requests now go to `to` instead of `from`.
    Switched {
        from: String,
        to: String,
        reason: SwitchReason,
    },
    /// An exhausted account serves requests again: the backend accepted a
    /// request to it, or it was [marked usable](ChatGptAccounts::mark_usable).
    Usable {
        account: String,
        reason: UsableReason,
    },
    /// Every account is exhausted: requests fail with a usage-limit error
    /// until the earliest known reset (`resets_at`), unless an account is
    /// found usable sooner.
    AllExhausted { resets_at: Option<SystemTime> },
    /// `account`'s responses report a plan window used up while it has
    /// credits: the backend is drawing on them.
    UsingCredits {
        account: String,
        balance: Option<String>,
    },
}

/// Why requests moved to another account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchReason {
    /// The account serving them was exhausted.
    Exhausted,
    /// An account ranked higher is usable again.
    Restored,
}

/// How an exhausted account was found usable. Except for `MarkedUsable`,
/// the account is tried by the next request and reported usable once the
/// backend accepts a request again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsableReason {
    /// The reset time the backend gave has passed.
    ResetTimePassed,
    /// The usage endpoint allowed requests (for example after a manual
    /// usage reset).
    StatusCheck,
    /// A usage check couldn't tell (the endpoint failed or didn't say), so
    /// a request tried the account.
    Retried,
    /// [`ChatGptAccounts::mark_usable`] was called.
    MarkedUsable,
}

/// "in 2h 5m", "now", or "at an unknown time".
fn reset_phrase(resets_at: Option<SystemTime>) -> String {
    let Some(resets_at) = resets_at else {
        return "at an unknown time".into();
    };
    let seconds = resets_at
        .duration_since(SystemTime::now())
        .map_or(0, |left| left.as_secs());
    match (seconds / 3600, seconds % 3600 / 60) {
        (0, 0) => "now".into(),
        (0, minutes) => format!("in {minutes}m"),
        (hours, minutes) => format!("in {hours}h {minutes}m"),
    }
}

impl fmt::Display for ChatGptAccountEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted {
                account,
                resets_at,
                code,
            } => write!(
                formatter,
                "ChatGPT account `{account}` reached its usage limit ({}); it resets {}",
                code.as_deref().unwrap_or("no code"),
                reset_phrase(*resets_at)
            ),
            Self::Switched { from, to, reason } => match reason {
                SwitchReason::Exhausted => write!(
                    formatter,
                    "ChatGPT requests moved from `{from}` (exhausted) to `{to}`"
                ),
                SwitchReason::Restored => write!(
                    formatter,
                    "ChatGPT requests moved back from `{from}` to `{to}`, which is usable again"
                ),
            },
            Self::Usable { account, reason } => write!(
                formatter,
                "ChatGPT account `{account}` is usable again ({})",
                match reason {
                    UsableReason::ResetTimePassed => "its reset time passed",
                    UsableReason::StatusCheck => "the usage endpoint allows requests",
                    UsableReason::Retried => "a request to it was accepted",
                    UsableReason::MarkedUsable => "marked usable",
                }
            ),
            Self::AllExhausted { resets_at } => write!(
                formatter,
                "Every ChatGPT account reached its usage limit; the earliest resets {}",
                reset_phrase(*resets_at)
            ),
            Self::UsingCredits { account, balance } => write!(
                formatter,
                "ChatGPT account `{account}` used up a plan window and is drawing on its credits (balance {})",
                balance.as_deref().unwrap_or("unknown")
            ),
        }
    }
}

/// What the ChatGPT backend's usage endpoint ([`CHATGPT_USAGE_URL`])
/// reported for an account. Fields the response lacks are `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatGptUsageStatus {
    /// `plan_type`, such as `pro`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_type: Option<String>,
    /// `rate_limit.allowed`: the plan's own usage lets requests through.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed: Option<bool>,
    /// `rate_limit.limit_reached`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_reached: Option<bool>,
    /// `rate_limit.primary_window`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<PlanUsageWindow>,
    /// `rate_limit.secondary_window`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary: Option<PlanUsageWindow>,
    /// `credits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits: Option<PlanCredits>,
    /// `spend_control.reached`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spend_control_reached: Option<bool>,
    /// `rate_limit_reached_type.type`, such as
    /// `workspace_owner_credits_depleted`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_reached_type: Option<String>,
    /// `rate_limit_reset_credits.available_count`: the manual usage resets
    /// the account has left.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_credits_available: Option<u64>,
}

impl ChatGptUsageStatus {
    /// Reads a usage-endpoint response.
    pub fn from_json(value: &Value) -> Self {
        let rate_limit = value.get("rate_limit").filter(|limit| limit.is_object());
        let window = |name: &str| {
            let window = rate_limit?.get(name)?;
            Some(PlanUsageWindow {
                used_percent: window.get("used_percent")?.as_f64()?,
                window_minutes: window
                    .get("limit_window_seconds")
                    .and_then(Value::as_u64)
                    .filter(|seconds| *seconds > 0)
                    .map(|seconds| seconds.div_ceil(60)),
                resets_at: window
                    .get("reset_at")
                    .and_then(Value::as_u64)
                    .filter(|at| *at > 0),
                reset_after_seconds: window.get("reset_after_seconds").and_then(Value::as_u64),
            })
        };
        let credits = value.get("credits").and_then(|credits| {
            Some(PlanCredits {
                has_credits: credits.get("has_credits")?.as_bool()?,
                unlimited: credits
                    .get("unlimited")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                balance: credits.get("balance").and_then(|balance| match balance {
                    Value::String(text) => Some(text.clone()),
                    Value::Number(number) => Some(number.to_string()),
                    _ => None,
                }),
            })
        });
        let reached_type = value.get("rate_limit_reached_type");
        Self {
            plan_type: value
                .get("plan_type")
                .and_then(Value::as_str)
                .map(String::from),
            allowed: rate_limit
                .and_then(|limit| limit.get("allowed"))
                .and_then(Value::as_bool),
            limit_reached: rate_limit
                .and_then(|limit| limit.get("limit_reached"))
                .and_then(Value::as_bool),
            primary: window("primary_window"),
            secondary: window("secondary_window"),
            credits,
            spend_control_reached: value
                .pointer("/spend_control/reached")
                .and_then(Value::as_bool),
            limit_reached_type: reached_type
                .and_then(|kind| kind.get("type").or(Some(kind)))
                .and_then(Value::as_str)
                .map(String::from),
            reset_credits_available: value
                .pointer("/rate_limit_reset_credits/available_count")
                .and_then(Value::as_u64),
        }
    }

    /// Whether the account can serve requests, as Codex decides that its
    /// ordinary usage has recovered: the plan allows requests or credits are
    /// left, no spend control is reached and no limit type is reported.
    /// `None` when the response doesn't say (no `rate_limit.allowed`).
    pub fn usable(&self) -> Option<bool> {
        let allowed = self.allowed?;
        let credits = self.credits.as_ref().is_some_and(PlanCredits::usable);
        Some(
            (allowed || credits)
                && self.spend_control_reached != Some(true)
                && self.limit_reached_type.is_none(),
        )
    }

    /// The latest reset time of a window that is used up.
    fn exhausted_until(&self) -> Option<SystemTime> {
        [&self.primary, &self.secondary]
            .into_iter()
            .flatten()
            .filter(|window| window.used_percent >= 100.0)
            .filter_map(|window| window.resets_at)
            .max()
            .map(|at| SystemTime::UNIX_EPOCH + Duration::from_secs(at))
    }
}

/// One account of [`ChatGptAccounts::statuses`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChatGptAccountStatus {
    pub label: String,
    /// The last request went to this account.
    pub active: bool,
    /// Set aside after the backend refused it; see
    /// [`ChatGptAccountEvent::Exhausted`].
    pub exhausted: bool,
    /// Set aside, but the next request tries it (with why it may be usable
    /// again): it is taken back once the backend accepts a request.
    pub retry: Option<UsableReason>,
    pub exhausted_since: Option<SystemTime>,
    /// When the backend said the limit resets, if it did.
    pub resets_at: Option<SystemTime>,
    /// The code of the error that set it aside.
    pub error_code: Option<String>,
    /// The plan usage its last response reported.
    pub plan_usage: Option<PlanUsage>,
    /// The last usage-endpoint report, and when it came.
    pub usage: Option<ChatGptUsageStatus>,
    pub checked_at: Option<SystemTime>,
}

impl ChatGptAccountStatus {
    /// Whether its last response reported a plan window used up while it
    /// has credits: the backend is drawing on them.
    pub fn using_credits(&self) -> bool {
        self.plan_usage.as_ref().is_some_and(using_credits)
    }
}

fn using_credits(usage: &PlanUsage) -> bool {
    usage.credits.as_ref().is_some_and(PlanCredits::usable)
        && [&usage.primary, &usage.secondary]
            .into_iter()
            .flatten()
            .any(|window| window.used_percent >= 100.0)
}

/// An account set aside.
#[derive(Debug, Clone)]
struct Block {
    since: SystemTime,
    until: Option<SystemTime>,
    next_check: SystemTime,
    code: Option<String>,
    /// Set when the account may be usable again (its reset time passed, or
    /// a check allowed it or couldn't tell): the next request tries it. The
    /// account is taken back, with [`ChatGptAccountEvent::Usable`] for this
    /// reason, once a request to it is accepted; a refusal sets it aside
    /// again quietly. So a check that allows an account the backend still
    /// refuses (a limit the check doesn't show) notifies nobody.
    retry: Option<UsableReason>,
}

#[derive(Debug, Default)]
struct AccountState {
    block: Option<Block>,
    plan_usage: Option<PlanUsage>,
    using_credits: bool,
    usage: Option<ChatGptUsageStatus>,
    checked_at: Option<SystemTime>,
}

impl AccountState {
    fn takes_requests(&self) -> bool {
        self.block
            .as_ref()
            .is_none_or(|block| block.retry.is_some())
    }
}

#[derive(Debug)]
struct PoolState {
    accounts: Vec<AccountState>,
    /// The account the last request went to.
    active: Option<usize>,
    /// The account each reasoning item (by `encrypted_content`) and
    /// turn-state token came from, by fingerprint. Ones not here are taken
    /// to be the first account's.
    origins: HashMap<u64, usize>,
    /// Accounts that served a response in this process.
    served: BTreeSet<usize>,
    /// [`ChatGptAccountEvent::AllExhausted`] was sent and no account has
    /// served since.
    all_exhausted_sent: bool,
}

fn fingerprint(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Errors after which an account can't serve requests until its limit
/// resets or its credits are refilled: a plan usage limit, and the credit,
/// spend and quota walls Codex stops on (`QuotaExceeded`,
/// `UsageNotIncluded`).
fn exhausts_account(error: &Error) -> bool {
    error.is_usage_limit() || error.kind() == ErrorKind::Billing
}

fn error_code(error: &Error) -> Option<String> {
    match error {
        Error::ProviderError { code, .. } => code.clone(),
        _ => None,
    }
}

type Observer = Arc<dyn Fn(&ChatGptAccountEvent) + Send + Sync>;

/// An ordered list of ChatGPT OAuth accounts for the OpenAI provider, first
/// preferred; install it with
/// [`LLMClient::with_chatgpt_accounts`](crate::LLMClient::with_chatgpt_accounts).
///
/// - Each request goes to the first account not set aside. An account is
///   set aside only when the backend refuses it: a usage limit
///   (`usage_limit_reached`) or a credit, spend or quota wall. An account
///   with credits keeps serving on them, as in Codex.
/// - A refused request goes to the next account at once (nothing was
///   generated), with a [`ChatGptAccountEvent`]. When every account is set
///   aside, the request fails with a usage-limit error whose
///   [`retry_after`](Error::retry_after) is the earliest known reset, as a
///   single account's would.
/// - Requests move back to a higher account as soon as it is usable. The
///   next request tries it once its reset time has passed, or once the
///   usage endpoint (checked every
///   [`status check interval`](Self::with_status_check_interval) while a
///   lower account serves, or on [`check_now`](Self::check_now)) allows
///   requests, as after a manual usage reset, or can't tell. It is taken
///   back when the backend accepts that request, and a refusal sets it
///   aside again without a new event (so a limit the check doesn't show
///   notifies nobody). [`mark_usable`](Self::mark_usable) takes it back at
///   once.
/// - A request replays only the encrypted reasoning, and the
///   `x-codex-turn-state` token, of the account serving it; see the module
///   documentation. Reasoning from before this process (a resumed
///   conversation) counts as the first account's.
/// - [`ModelCallUsage::account`](crate::usage::ModelCallUsage::account)
///   names the account that served each call.
///
/// Clones share the accounts and their state.
#[derive(Clone)]
pub struct ChatGptAccounts {
    accounts: Arc<[ChatGptAccount]>,
    state: Arc<Mutex<PoolState>>,
    /// Held while status checks run, so concurrent requests check once.
    check_lock: Arc<tokio::sync::Mutex<()>>,
    status_check_interval: Duration,
    usage_url: String,
    observer: Option<Observer>,
    #[cfg(test)]
    clock: Option<Arc<Mutex<SystemTime>>>,
}

impl fmt::Debug for ChatGptAccounts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatGptAccounts")
            .field("accounts", &self.accounts)
            .field("status_check_interval", &self.status_check_interval)
            .finish_non_exhaustive()
    }
}

impl ChatGptAccounts {
    /// The accounts in order of preference. Fails with
    /// [`Error::InvalidRequest`] for an empty list, an empty or repeated
    /// label, an account using an API key, or two labels for one ChatGPT
    /// account.
    pub fn new(accounts: impl IntoIterator<Item = ChatGptAccount>) -> Result<Self, Error> {
        let accounts: Vec<ChatGptAccount> = accounts.into_iter().collect();
        if accounts.is_empty() {
            return Err(Error::InvalidRequest(
                "ChatGptAccounts needs at least one account".into(),
            ));
        }
        let mut labels = BTreeSet::new();
        let mut account_ids: HashMap<String, &str> = HashMap::new();
        for account in &accounts {
            if account.label.trim().is_empty() || !labels.insert(account.label.as_str()) {
                return Err(Error::InvalidRequest(format!(
                    "ChatGPT account labels must be distinct and non-empty; `{}` is not",
                    account.label
                )));
            }
            // Credentials being refreshed right now are not checked.
            let Ok(auth) = account.client.auth.try_read() else {
                continue;
            };
            match &*auth {
                OpenAIAuth::ApiKey(_) => {
                    return Err(Error::InvalidRequest(format!(
                        "ChatGPT account `{}` uses an API key; every account must use ChatGPT OAuth",
                        account.label
                    )));
                }
                OpenAIAuth::ChatGptOAuth(credentials) => {
                    if let Some(other) =
                        account_ids.insert(credentials.account_id.clone(), &account.label)
                    {
                        return Err(Error::InvalidRequest(format!(
                            "ChatGPT accounts `{other}` and `{}` are the same account",
                            account.label
                        )));
                    }
                }
            }
        }
        let state = PoolState {
            accounts: accounts.iter().map(|_| AccountState::default()).collect(),
            active: None,
            origins: HashMap::new(),
            served: BTreeSet::new(),
            all_exhausted_sent: false,
        };
        Ok(Self {
            accounts: accounts.into(),
            state: Arc::new(Mutex::new(state)),
            check_lock: Arc::new(tokio::sync::Mutex::new(())),
            status_check_interval: DEFAULT_STATUS_CHECK_INTERVAL,
            usage_url: CHATGPT_USAGE_URL.into(),
            observer: None,
            #[cfg(test)]
            clock: None,
        })
    }

    /// The accounts of Codex `auth.json` files, labelled, in order of
    /// preference; see [`ChatGptAccount::from_codex_auth_file`].
    pub fn from_codex_auth_files<'a>(
        files: impl IntoIterator<Item = (&'a str, &'a CodexAuthFile)>,
    ) -> Result<Self, Error> {
        let accounts = files
            .into_iter()
            .map(|(label, file)| ChatGptAccount::from_codex_auth_file(label, file))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(accounts)
    }

    /// How often an exhausted account ranked above the one serving is
    /// checked on the usage endpoint (default 5 minutes). Checks happen as
    /// requests are made, never in the background.
    pub fn with_status_check_interval(mut self, interval: Duration) -> Self {
        self.status_check_interval = interval;
        self
    }

    /// Reads account usage from `url` instead of [`CHATGPT_USAGE_URL`], for
    /// example a proxy or a test server.
    pub fn with_usage_url(mut self, url: impl Into<String>) -> Self {
        self.usage_url = url.into();
        self
    }

    /// Calls `observer` with every [`ChatGptAccountEvent`] (they are logged
    /// either way), for example to tell an operator that the first account
    /// is exhausted. It is called after the state changed, from the task
    /// making the request, and must not block for long.
    pub fn with_observer(
        mut self,
        observer: impl Fn(&ChatGptAccountEvent) + Send + Sync + 'static,
    ) -> Self {
        self.observer = Some(Arc::new(observer));
        self
    }

    /// Applies `configure` to every account's client (the `LLMClient`'s
    /// OpenAI settings). The clients keep their credentials.
    pub(crate) fn map_clients(mut self, configure: impl Fn(OpenAIClient) -> OpenAIClient) -> Self {
        self.accounts = self
            .accounts
            .iter()
            .map(|account| ChatGptAccount {
                label: account.label.clone(),
                client: configure(account.client.clone()),
            })
            .collect();
        self
    }

    pub fn accounts(&self) -> &[ChatGptAccount] {
        &self.accounts
    }

    /// The label of the account the last request went to.
    pub fn active_account(&self) -> Option<String> {
        let state = self.lock();
        state.active.map(|index| self.accounts[index].label.clone())
    }

    /// Every account's state, in order of preference.
    pub fn statuses(&self) -> Vec<ChatGptAccountStatus> {
        let state = self.lock();
        self.statuses_of(&state)
    }

    fn statuses_of(&self, state: &PoolState) -> Vec<ChatGptAccountStatus> {
        self.accounts
            .iter()
            .zip(&state.accounts)
            .enumerate()
            .map(|(index, (account, account_state))| {
                let block = account_state.block.as_ref();
                ChatGptAccountStatus {
                    label: account.label.clone(),
                    active: state.active == Some(index),
                    exhausted: block.is_some(),
                    retry: block.and_then(|block| block.retry),
                    exhausted_since: block.map(|block| block.since),
                    resets_at: block.and_then(|block| block.until),
                    error_code: block.and_then(|block| block.code.clone()),
                    plan_usage: account_state.plan_usage.clone(),
                    usage: account_state.usage.clone(),
                    checked_at: account_state.checked_at,
                }
            })
            .collect()
    }

    /// Checks every exhausted account on the usage endpoint now (for
    /// example once the operator has pressed a usage reset), and returns the
    /// accounts' states. The next request tries the first account a check
    /// allowed (or couldn't decide on; see [`ChatGptAccountStatus::retry`]),
    /// which is taken back once the backend accepts it. Waits for a check
    /// under way.
    pub async fn check_now(&self) -> Vec<ChatGptAccountStatus> {
        let _checking = self.check_lock.lock().await;
        let exhausted: Vec<usize> = {
            let state = self.lock();
            (0..self.accounts.len())
                .filter(|index| state.accounts[*index].block.is_some())
                .collect()
        };
        self.check(&exhausted).await;
        self.statuses()
    }

    /// Reads the usage of the account labelled `label` from the usage
    /// endpoint, without changing whether it is set aside (the report shows
    /// in its [status](Self::statuses)). Fails with
    /// [`Error::InvalidRequest`] for an unknown label, and with the
    /// endpoint's error.
    pub async fn read_usage(&self, label: &str) -> Result<ChatGptUsageStatus, Error> {
        let index = self.index_of(label)?;
        let status = fetch_usage(&self.accounts[index].client, &self.usage_url).await?;
        let now = self.now();
        let mut state = self.lock();
        state.accounts[index].usage = Some(status.clone());
        state.accounts[index].checked_at = Some(now);
        Ok(status)
    }

    fn index_of(&self, label: &str) -> Result<usize, Error> {
        self.accounts
            .iter()
            .position(|account| account.label == label)
            .ok_or_else(|| Error::InvalidRequest(format!("no ChatGPT account `{label}`")))
    }

    /// Takes the account labelled `label` back now, without a check (for
    /// example when the operator says its usage was reset). Fails with
    /// [`Error::InvalidRequest`] for an unknown label.
    pub fn mark_usable(&self, label: &str) -> Result<(), Error> {
        let index = self.index_of(label)?;
        let mut events = Vec::new();
        {
            let mut state = self.lock();
            if state.accounts[index].block.take().is_some() {
                events.push(ChatGptAccountEvent::Usable {
                    account: label.into(),
                    reason: UsableReason::MarkedUsable,
                });
            }
        }
        self.emit(events);
        Ok(())
    }

    /// The client of the account the next request would go to (the first
    /// usable one, else the first): what the `LLMClient`'s OAuth methods
    /// (refreshing, reading credentials) act on.
    pub(crate) fn preferred_client(&self) -> &OpenAIClient {
        let now = self.now();
        let state = self.lock();
        let index = (0..self.accounts.len())
            .find(|index| {
                let account = &state.accounts[*index];
                account.takes_requests()
                    || account
                        .block
                        .as_ref()
                        .and_then(|block| block.until)
                        .is_some_and(|until| until <= now)
            })
            .unwrap_or(0);
        &self.accounts[index].client
    }

    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn now(&self) -> SystemTime {
        #[cfg(test)]
        if let Some(clock) = &self.clock {
            return *clock.lock().unwrap();
        }
        SystemTime::now()
    }

    fn emit(&self, events: Vec<ChatGptAccountEvent>) {
        for event in events {
            match &event {
                ChatGptAccountEvent::Usable { .. } | ChatGptAccountEvent::UsingCredits { .. } => {
                    log::info!("arko_ai: {event}")
                }
                _ => log::warn!("arko_ai: {event}"),
            }
            if let Some(observer) = &self.observer {
                observer(&event);
            }
        }
    }

    /// Lets the next request try accounts whose reset time has passed.
    fn expire(state: &mut PoolState, now: SystemTime) {
        for account in &mut state.accounts {
            if let Some(block) = account.block.as_mut()
                && block.retry.is_none()
                && block.until.is_some_and(|until| until <= now)
            {
                block.retry = Some(UsableReason::ResetTimePassed);
            }
        }
    }

    /// Runs the status checks that are due: exhausted accounts ranked above
    /// the first usable one whose check time has come. Skipped while
    /// another request is checking.
    async fn run_due_checks(&self) {
        let now = self.now();
        let due: Vec<usize> = {
            let mut state = self.lock();
            Self::expire(&mut state, now);
            let first_usable = (0..self.accounts.len())
                .find(|index| state.accounts[*index].takes_requests())
                .unwrap_or(self.accounts.len());
            (0..first_usable)
                .filter(|index| {
                    state.accounts[*index]
                        .block
                        .as_ref()
                        .is_some_and(|block| block.retry.is_none() && block.next_check <= now)
                })
                .collect()
        };
        if due.is_empty() {
            return;
        }
        let Ok(_checking) = self.check_lock.try_lock() else {
            return;
        };
        self.check(&due).await;
    }

    /// Checks `indices` on the usage endpoint, one at a time. An account a
    /// check allows, or can't decide on, is tried by the next request.
    async fn check(&self, indices: &[usize]) {
        for &index in indices {
            let account = &self.accounts[index];
            let result = fetch_usage(&account.client, &self.usage_url).await;
            let now = self.now();
            {
                let mut state = self.lock();
                let account_state = &mut state.accounts[index];
                let usable = match result {
                    Ok(status) => {
                        let usable = status.usable();
                        if usable == Some(false)
                            && let Some(block) = account_state.block.as_mut()
                            && let Some(until) = status.exhausted_until().filter(|at| *at > now)
                        {
                            block.until = Some(until);
                        }
                        account_state.usage = Some(status);
                        account_state.checked_at = Some(now);
                        usable
                    }
                    Err(error) => {
                        log::info!(
                            "arko_ai: the usage of ChatGPT account `{}` could not be read ({error}); the next request tries it",
                            account.label
                        );
                        None
                    }
                };
                if let Some(block) = account_state.block.as_mut() {
                    block.next_check = now + self.status_check_interval;
                    block.retry = match usable {
                        Some(true) => Some(UsableReason::StatusCheck),
                        Some(false) => None,
                        None => Some(UsableReason::Retried),
                    };
                }
            }
        }
    }

    /// The first account to try that `tried` doesn't hold.
    fn choose(&self, tried: &BTreeSet<usize>) -> Option<usize> {
        let mut state = self.lock();
        Self::expire(&mut state, self.now());
        (0..self.accounts.len())
            .find(|index| !tried.contains(index) && state.accounts[*index].takes_requests())
    }

    /// Sets `index` aside after `error`.
    fn exhaust(&self, index: usize, error: &Error) {
        let now = self.now();
        let mut events = Vec::new();
        {
            let mut state = self.lock();
            let account = &mut state.accounts[index];
            let previous = account.block.take();
            let until = error
                .retry_after()
                .map(|left| now + left)
                .or_else(|| previous.as_ref().and_then(|block| block.until))
                .filter(|until| *until > now);
            let code = error_code(error);
            // A retry that was refused again: nothing new to report.
            if !previous.as_ref().is_some_and(|block| block.retry.is_some()) {
                events.push(ChatGptAccountEvent::Exhausted {
                    account: self.accounts[index].label.clone(),
                    resets_at: until,
                    code: code.clone(),
                });
            }
            account.block = Some(Block {
                since: previous.as_ref().map_or(now, |block| block.since),
                until,
                next_check: now + self.status_check_interval,
                code,
                retry: None,
            });
        }
        self.emit(events);
    }

    /// The error of a request no account can serve: a usage-limit error
    /// retryable at the earliest known reset; without one, the last
    /// account's own error when there is one.
    fn all_exhausted(&self, last_error: Option<Error>) -> Error {
        let now = self.now();
        let mut events = Vec::new();
        let (earliest, summary) = {
            let mut state = self.lock();
            let earliest = state
                .accounts
                .iter()
                .filter_map(|account| account.block.as_ref()?.until)
                .min();
            if !state.all_exhausted_sent {
                state.all_exhausted_sent = true;
                events.push(ChatGptAccountEvent::AllExhausted {
                    resets_at: earliest,
                });
            }
            let summary = self
                .accounts
                .iter()
                .zip(&state.accounts)
                .map(|(account, account_state)| {
                    let until = account_state.block.as_ref().and_then(|block| block.until);
                    format!("`{}` resets {}", account.label, reset_phrase(until))
                })
                .collect::<Vec<_>>()
                .join(", ");
            (earliest, summary)
        };
        self.emit(events);
        match (earliest, last_error) {
            (None, Some(error)) => error,
            (earliest, _) => Error::ProviderError {
                message: format!("Every ChatGPT account reached its usage limit ({summary})"),
                status: Some(429),
                code: Some("usage_limit_reached".into()),
                retry_after: earliest.map(|at| {
                    at.duration_since(now)
                        .unwrap_or_default()
                        .max(Duration::from_secs(1))
                }),
            },
        }
    }

    /// Records that `index` accepted a request, taking it back if it was
    /// set aside.
    fn serving(&self, index: usize) {
        let mut events = Vec::new();
        {
            let mut state = self.lock();
            if let Some(block) = state.accounts[index].block.take() {
                events.push(ChatGptAccountEvent::Usable {
                    account: self.accounts[index].label.clone(),
                    reason: block.retry.unwrap_or(UsableReason::Retried),
                });
            }
            // Before the first request, the first account is the one in use.
            let previous = state.active.replace(index).unwrap_or(0);
            if previous != index {
                events.push(ChatGptAccountEvent::Switched {
                    from: self.accounts[previous].label.clone(),
                    to: self.accounts[index].label.clone(),
                    reason: if index < previous {
                        SwitchReason::Restored
                    } else {
                        SwitchReason::Exhausted
                    },
                });
            }
        }
        self.emit(events);
    }

    /// Records a response `index` served, and labels it.
    fn served(&self, index: usize, response: &mut GenerationResponse) {
        let label = &self.accounts[index].label;
        response.model_call_usage.account = Some(label.clone());
        let mut events = Vec::new();
        {
            let mut state = self.lock();
            state.served.insert(index);
            state.all_exhausted_sent = false;
            if let Some(usage) = &response.model_call_usage.plan_usage {
                let account = &mut state.accounts[index];
                let on_credits = using_credits(usage);
                if on_credits && !account.using_credits {
                    events.push(ChatGptAccountEvent::UsingCredits {
                        account: label.clone(),
                        balance: usage
                            .credits
                            .as_ref()
                            .and_then(|credits| credits.balance.clone()),
                    });
                }
                account.using_credits = on_credits;
                account.plan_usage = Some(usage.clone());
            }
            if self.accounts.len() > 1 {
                if state.origins.len() > MAX_ORIGINS {
                    log::info!(
                        "arko_ai: forgetting which ChatGPT account produced older reasoning items"
                    );
                    state.origins.clear();
                }
                let mut record = |text: &str| {
                    state.origins.insert(fingerprint(text), index);
                };
                for text in encrypted_reasoning(&response.message.reasoning_artifact) {
                    record(text);
                }
                if let Some(token) = &response.openai_turn_state {
                    record(token);
                }
            }
        }
        self.emit(events);
    }

    /// `conversation` as `index` may receive it: without the encrypted
    /// reasoning, and the turn-state token, of other accounts. Borrowed when
    /// nothing is removed.
    fn prepare<'a>(&self, conversation: &'a Conversation, index: usize) -> Cow<'a, Conversation> {
        if self.accounts.len() < 2 {
            return Cow::Borrowed(conversation);
        }
        let state = self.lock();
        // Unknown items count as the first account's.
        if index == 0 && state.served.iter().all(|served| *served == 0) {
            return Cow::Borrowed(conversation);
        }
        let foreign =
            |text: &str| state.origins.get(&fingerprint(text)).copied().unwrap_or(0) != index;
        let foreign_turn_state = conversation
            .current_openai_turn_state()
            .is_some_and(&foreign);
        let foreign_reasoning = conversation
            .messages
            .iter()
            .any(|message| encrypted_reasoning(&message.reasoning_artifact).any(&foreign));
        if !foreign_turn_state && !foreign_reasoning {
            return Cow::Borrowed(conversation);
        }
        let mut prepared = conversation.clone();
        if foreign_turn_state {
            prepared.openai_turn_state = None;
        }
        for message in &mut prepared.messages {
            match &mut message.reasoning_artifact {
                Some(ReasoningArtifact::OpenAIOutputItems { items }) => items.retain(|item| {
                    !(item.get("type").and_then(Value::as_str) == Some("reasoning")
                        && item
                            .get("encrypted_content")
                            .and_then(Value::as_str)
                            .is_some_and(&foreign))
                }),
                Some(ReasoningArtifact::OpenAIEncryptedReasoning { items }) => {
                    items.retain(|item| !foreign(&item.encrypted_content))
                }
                _ => {}
            }
        }
        Cow::Owned(prepared)
    }

    /// Whether `index` is set aside and being tried again.
    fn retrying(&self, index: usize) -> bool {
        self.lock().accounts[index]
            .block
            .as_ref()
            .is_some_and(|block| block.retry.is_some())
    }

    /// Trying `index` again failed for another reason than its limit: it
    /// stays set aside until a check allows it.
    fn retry_failed(&self, index: usize, error: &Error) {
        log::info!(
            "arko_ai: trying ChatGPT account `{}` again failed ({error}); it stays set aside",
            self.accounts[index].label
        );
        let now = self.now();
        if let Some(block) = self.lock().accounts[index].block.as_mut() {
            block.retry = None;
            block.next_check = now + self.status_check_interval;
            block.until = block.until.filter(|until| *until > now);
        }
    }

    /// Opens a stream on the first account not in `tried` that accepts the
    /// request.
    fn open<'a>(
        &'a self,
        conversation: &'a Conversation,
        config: &'a GenerationConfig,
        mut tried: BTreeSet<usize>,
    ) -> BoxFuture<'a, Result<BoxStream<'static, Result<StreamEvent, Error>>, Error>> {
        Box::pin(async move {
            self.run_due_checks().await;
            let mut last_error = None;
            loop {
                let Some(index) = self.choose(&tried) else {
                    return Err(self.all_exhausted(last_error));
                };
                tried.insert(index);
                let retrying = self.retrying(index);
                let client = &self.accounts[index].client;
                let opened = async {
                    client
                        .refresh_oauth_token_if_expiring(REFRESH_MARGIN)
                        .await?;
                    let prepared = self.prepare(conversation, index);
                    client.generate_stream(&prepared, config).await
                }
                .await;
                match opened {
                    Ok(stream) => {
                        self.serving(index);
                        return Ok(self.watch(stream, index, tried, conversation, config));
                    }
                    Err(error) if exhausts_account(&error) => {
                        self.exhaust(index, &error);
                        last_error = Some(error);
                    }
                    // Trying an account set aside is a guess; another
                    // account serves when it fails.
                    Err(error) if retrying => self.retry_failed(index, &error),
                    Err(error) => return Err(error),
                }
            }
        })
    }

    /// Passes `index`'s stream on, labelling its response. A refusal before
    /// anything was streamed moves the request to the next account; one
    /// after it fails the stream with a transient error when another
    /// account is usable (so the retry goes there), else with the refusal.
    fn watch(
        &self,
        inner: BoxStream<'static, Result<StreamEvent, Error>>,
        index: usize,
        tried: BTreeSet<usize>,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> BoxStream<'static, Result<StreamEvent, Error>> {
        // The request, kept for moving it to another account mid-stream.
        let request = Arc::new((self.clone(), conversation.clone(), config.clone(), tried));
        stream::unfold(
            WatchState::Streaming {
                inner,
                delivered: false,
            },
            move |watch| {
                let request = request.clone();
                async move {
                    let (pool, conversation, config, tried) = &*request;
                    match watch {
                        WatchState::Done => None,
                        WatchState::Moved(mut next) => {
                            let item = next.next().await?;
                            Some((item, WatchState::Moved(next)))
                        }
                        WatchState::Streaming {
                            mut inner,
                            delivered,
                        } => match inner.next().await? {
                            Ok(StreamEvent::End(mut response)) => {
                                pool.served(index, &mut response);
                                Some((Ok(StreamEvent::End(response)), WatchState::Done))
                            }
                            Ok(event) => Some((
                                Ok(event),
                                WatchState::Streaming {
                                    inner,
                                    delivered: true,
                                },
                            )),
                            Err(error) if exhausts_account(&error) => {
                                pool.exhaust(index, &error);
                                if !delivered {
                                    return match pool.open(conversation, config, tried.clone()).await {
                                        Ok(mut next) => {
                                            let item = next.next().await?;
                                            Some((item, WatchState::Moved(next)))
                                        }
                                        Err(error) => Some((Err(error), WatchState::Done)),
                                    };
                                }
                                let next_usable = pool.choose(tried);
                                let error = match next_usable {
                                    Some(next) => Error::StreamTruncated(format!(
                                        "ChatGPT account `{}` was refused during the response; a retry goes to `{}` ({error})",
                                        pool.accounts[index].label, pool.accounts[next].label
                                    )),
                                    None => error,
                                };
                                Some((Err(error), WatchState::Done))
                            }
                            Err(error) => Some((Err(error), WatchState::Done)),
                        },
                    }
                }
            },
        )
        .boxed()
    }
}

enum WatchState {
    Streaming {
        inner: BoxStream<'static, Result<StreamEvent, Error>>,
        /// An event was passed on.
        delivered: bool,
    },
    /// The request moved to another account; its stream is labelled there.
    Moved(BoxStream<'static, Result<StreamEvent, Error>>),
    Done,
}

/// The `encrypted_content` of the reasoning items in `artifact`.
fn encrypted_reasoning(
    artifact: &Option<ReasoningArtifact>,
) -> Box<dyn Iterator<Item = &str> + '_> {
    match artifact {
        Some(ReasoningArtifact::OpenAIOutputItems { items }) => Box::new(
            items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
                .filter_map(|item| item.get("encrypted_content").and_then(Value::as_str)),
        ),
        Some(ReasoningArtifact::OpenAIEncryptedReasoning { items }) => {
            Box::new(items.iter().map(|item| item.encrypted_content.as_str()))
        }
        _ => Box::new(std::iter::empty()),
    }
}

/// Reads `client`'s account usage from `url` with its credentials (refreshed
/// first when they expire soon), sending the headers Codex's backend client
/// sends: the bearer token, `ChatGPT-Account-Id`, and the client's
/// `User-Agent` when one is configured.
async fn fetch_usage(client: &OpenAIClient, url: &str) -> Result<ChatGptUsageStatus, Error> {
    client
        .refresh_oauth_token_if_expiring(REFRESH_MARGIN)
        .await?;
    let auth = client.snapshot_auth().await;
    if !auth.is_chatgpt_oauth() {
        return Err(Error::InvalidRequest(
            "account usage needs ChatGPT OAuth credentials".into(),
        ));
    }
    let mut request = OpenAIClient::apply_auth_headers(
        &auth,
        client.http_client.get(url).timeout(USAGE_REQUEST_TIMEOUT),
    );
    if let Some(user_agent) = &client.user_agent {
        request = request.header(reqwest::header::USER_AGENT, user_agent);
    }
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = super::retry_after_from_headers(&response);
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable error body>".into());
        return Err(super::response_error(
            "ChatGPT usage request failed",
            status,
            retry_after,
            &body,
        ));
    }
    let value: Value = response.json().await?;
    Ok(ChatGptUsageStatus::from_json(&value))
}

#[async_trait::async_trait]
impl ProviderClient for ChatGptAccounts {
    async fn generate_text(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, Error> {
        // The ChatGPT backend always streams; collect the stream.
        let mut stream = self.open(conversation, config, BTreeSet::new()).await?;
        while let Some(event) = stream.next().await {
            if let StreamEvent::End(response) = event? {
                return Ok(*response);
            }
        }
        Err(Error::StreamTruncated(
            "OpenAI stream ended before the response completed".to_string(),
        ))
    }

    async fn generate_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        self.open(conversation, config, BTreeSet::new()).await
    }
}

#[cfg(test)]
#[path = "accounts_tests.rs"]
mod tests;
