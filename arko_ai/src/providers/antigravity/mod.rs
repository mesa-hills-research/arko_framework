//! Models that run through the installed Antigravity CLI on the user's plan.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The host executable and the arguments before `relay`, `gate` or `tripwire`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelperCommand {
    pub program: PathBuf,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

/// The usage windows Antigravity reports for a set of models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageGroup {
    pub name: String,
    pub description: String,
    pub buckets: Vec<UsageBucket>,
}

/// One window in the Antigravity plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageBucket {
    pub id: String,
    pub name: String,
    pub window: String,
    pub remaining_fraction: f64,
    pub reset_time: String,
}

/// The lowest version Arko supports.
pub const MINIMUM_VERSION: &str = "1.5.14";

/// The installed CLI and the models available on its sign-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignIn {
    pub cli_path: PathBuf,
    pub version: String,
    pub tier: Option<String>,
    pub models: Vec<AvailableModel>,
}

/// A model family and the thought levels the CLI offers for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableModel {
    pub id: String,
    pub name: String,
    pub thought_levels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SignInProblem {
    #[error("The Antigravity CLI was not found.")]
    Missing,
    #[error("Antigravity {found} is too old. Install {MINIMUM_VERSION} or later.")]
    Outdated { found: String },
    #[error("Sign in by running `agy` once.")]
    SignedOut,
    #[error("Could not check the Antigravity CLI. {0}")]
    Failed(String),
}

pub fn find_agy() -> Option<PathBuf> {
    None
}

pub async fn check_sign_in(_cli: Option<&Path>) -> Result<SignIn, SignInProblem> {
    Err(SignInProblem::Missing)
}

pub async fn run_relay(_path: &Path) -> io::Result<()> {
    unavailable()
}

pub fn run_gate(_path: &Path) -> io::Result<()> {
    unavailable()
}

pub fn run_tripwire(_path: &Path) -> io::Result<()> {
    unavailable()
}

fn unavailable() -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Antigravity is not available yet.",
    ))
}
