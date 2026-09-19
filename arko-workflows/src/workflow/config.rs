//! Workflow configuration and portable discovery metadata.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedPlatform {
    pub os: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
}

impl SupportedPlatform {
    pub fn new(os: impl Into<String>) -> Self {
        Self {
            os: os.into(),
            arch: None,
        }
    }

    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.arch = Some(arch.into());
        self
    }

    pub fn matches_host(&self) -> bool {
        self.os.eq_ignore_ascii_case(std::env::consts::OS)
            && self
                .arch
                .as_deref()
                .is_none_or(|arch| arch.eq_ignore_ascii_case(std::env::consts::ARCH))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowDependency {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowMetadata {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub supported_platforms: Vec<SupportedPlatform>,
    #[serde(default)]
    pub required_dependencies: Vec<WorkflowDependency>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub name: String,
    pub workspace: PathBuf,
    #[serde(default)]
    pub metadata: WorkflowMetadata,
}
