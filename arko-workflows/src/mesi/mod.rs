//! MESI-inspired file conflict tracking for parallel agents.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Record of an agent reading a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReadRecord {
    pub agent: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Record of an agent writing a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWriteRecord {
    pub agent: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// State tracked per file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileState {
    pub readers: Vec<FileReadRecord>,
    pub writers: Vec<FileWriteRecord>,
}

/// The action to take when a conflict is detected.
#[derive(Debug, Clone)]
pub enum ConflictAction {
    Allow,
    Block {
        conflicting_writes: Vec<FileWriteRecord>,
    },
    NotifyAndAllow {
        conflicting_writes: Vec<FileWriteRecord>,
    },
}

/// Configuration for MESI file tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MesiConfig {
    /// If true, block writes that conflict with stale reads.
    pub block_on_conflict: bool,
    /// Tool names considered write-capable.
    pub write_tools: HashSet<String>,
    /// Tool names considered read-capable.
    pub read_tools: HashSet<String>,
}

impl Default for MesiConfig {
    fn default() -> Self {
        Self {
            block_on_conflict: true,
            write_tools: ["write_file".into(), "apply_diff".into()].into(),
            read_tools: ["read_file".into()].into(),
        }
    }
}

/// Tracks file reads and writes across parallel agents.
pub struct FileTracker {
    files: RwLock<HashMap<PathBuf, FileState>>,
    config: MesiConfig,
}

impl FileTracker {
    pub fn new() -> Self {
        Self::with_config(MesiConfig::default())
    }

    pub fn with_config(config: MesiConfig) -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Record that an agent read a file.
    pub fn register_read(&self, agent: &str, path: &Path) {
        let mut files = self.files.write();
        let state = files.entry(path.to_path_buf()).or_default();
        state.readers.push(FileReadRecord {
            agent: agent.to_string(),
            timestamp: chrono::Utc::now(),
        });
    }

    /// Record that an agent wrote a file.
    pub fn register_write(&self, agent: &str, path: &Path) {
        let mut files = self.files.write();
        let state = files.entry(path.to_path_buf()).or_default();
        state.writers.push(FileWriteRecord {
            agent: agent.to_string(),
            timestamp: chrono::Utc::now(),
        });
    }

    /// Check if an agent writing to this file would conflict with stale reads.
    pub fn check_write_conflict(&self, agent: &str, path: &PathBuf) -> ConflictAction {
        let files = self.files.read();
        let Some(state) = files.get(path) else {
            return ConflictAction::Allow;
        };

        let my_last_read = state
            .readers
            .iter()
            .filter(|r| r.agent == agent)
            .map(|r| r.timestamp)
            .max();

        let Some(my_read_time) = my_last_read else {
            return ConflictAction::Allow;
        };

        let conflicting: Vec<FileWriteRecord> = state
            .writers
            .iter()
            .filter(|w| w.agent != agent && w.timestamp > my_read_time)
            .cloned()
            .collect();

        if conflicting.is_empty() {
            ConflictAction::Allow
        } else if self.config.block_on_conflict {
            ConflictAction::Block {
                conflicting_writes: conflicting,
            }
        } else {
            ConflictAction::NotifyAndAllow {
                conflicting_writes: conflicting,
            }
        }
    }

    pub fn is_write_tool(&self, tool_name: &str) -> bool {
        self.config.write_tools.contains(tool_name)
    }

    pub fn is_read_tool(&self, tool_name: &str) -> bool {
        self.config.read_tools.contains(tool_name)
    }
}

impl Default for FileTracker {
    fn default() -> Self {
        Self::new()
    }
}
