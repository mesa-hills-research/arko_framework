//! Tiered, access-controlled memory system for workflows.

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::WorkflowError;

mod reads;
pub mod tools;

/// Who can access a memory tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessScope {
    /// All agents in the workflow.
    All,
    /// Only the named agents.
    Only(HashSet<String>),
    /// No agents.
    None,
}

impl AccessScope {
    /// Returns true if the named agent is permitted.
    pub fn allows(&self, agent_name: &str) -> bool {
        match self {
            AccessScope::All => true,
            AccessScope::Only(set) => set.contains(agent_name),
            AccessScope::None => false,
        }
    }
}

/// How content in a memory tier can be modified.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum WriteMode {
    /// New content is appended to existing content.
    #[default]
    AppendOnly,
    /// Content can be replaced entirely.
    FullEdit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: uuid::Uuid,
    pub author: String,
    #[serde(default)]
    pub task_id: Option<uuid::Uuid>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content: String,
}

/// A single memory tier with content and access policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTier {
    pub name: String,
    pub content: String,
    pub readers: AccessScope,
    pub writers: AccessScope,
    pub write_mode: WriteMode,
    #[serde(default)]
    pub entries: Vec<MemoryEntry>,
    #[serde(default)]
    pub max_bytes: Option<usize>,
}

/// The complete memory store. Each tier is independently lockable.
#[derive(Default)]
pub struct MemoryStore {
    tiers: HashMap<String, RwLock<MemoryTier>>,
}

impl MemoryStore {
    /// Creates a new memory store from the given tiers.
    pub fn new(tiers: Vec<MemoryTier>) -> Self {
        let map = tiers
            .into_iter()
            .map(|t| (t.name.clone(), RwLock::new(t)))
            .collect();
        Self { tiers: map }
    }

    /// Returns true if no memory tiers are configured.
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Read the content of a memory tier.
    pub fn read(&self, agent_name: &str, tier_name: &str) -> Result<String, WorkflowError> {
        self.read_versioned(agent_name, tier_name)
            .map(|(_, content)| content)
    }

    /// Read a tier and its write version under the same lock.
    pub fn read_versioned(
        &self,
        agent_name: &str,
        tier_name: &str,
    ) -> Result<(usize, String), WorkflowError> {
        let tier_lock = self
            .tiers
            .get(tier_name)
            .ok_or_else(|| WorkflowError::MemoryTierNotFound(tier_name.to_string()))?;
        let tier = tier_lock.read();
        if !tier.readers.allows(agent_name) {
            return Err(WorkflowError::MemoryAccessDenied {
                agent: agent_name.to_string(),
                tier: tier_name.to_string(),
                action: "read".to_string(),
            });
        }
        Ok((tier.entries.len(), tier.content.clone()))
    }

    /// Write content to a memory tier.
    pub fn write(
        &self,
        agent_name: &str,
        tier_name: &str,
        content: &str,
    ) -> Result<(), WorkflowError> {
        self.write_with_context(agent_name, None, tier_name, content)
    }

    pub fn write_with_context(
        &self,
        agent_name: &str,
        task_id: Option<uuid::Uuid>,
        tier_name: &str,
        content: &str,
    ) -> Result<(), WorkflowError> {
        let tier_lock = self
            .tiers
            .get(tier_name)
            .ok_or_else(|| WorkflowError::MemoryTierNotFound(tier_name.to_string()))?;
        let mut tier = tier_lock.write();
        if !tier.writers.allows(agent_name) {
            return Err(WorkflowError::MemoryAccessDenied {
                agent: agent_name.to_string(),
                tier: tier_name.to_string(),
                action: "write".to_string(),
            });
        }
        let separator_bytes = usize::from(
            matches!(tier.write_mode, WriteMode::AppendOnly) && !tier.content.is_empty(),
        );
        let resulting_bytes = match tier.write_mode {
            WriteMode::AppendOnly => tier
                .content
                .len()
                .saturating_add(separator_bytes)
                .saturating_add(content.len()),
            WriteMode::FullEdit => content.len(),
        };
        if tier.max_bytes.is_some_and(|limit| resulting_bytes > limit) {
            return Err(WorkflowError::MemoryLimitExceeded {
                tier: tier_name.to_string(),
                attempted: resulting_bytes,
                limit: tier.max_bytes.unwrap_or(0),
            });
        }
        match tier.write_mode {
            WriteMode::AppendOnly => {
                if !tier.content.is_empty() {
                    tier.content.push('\n');
                }
                tier.content.push_str(content);
            }
            WriteMode::FullEdit => {
                tier.content = content.to_string();
            }
        }
        tier.entries.push(MemoryEntry {
            id: uuid::Uuid::new_v4(),
            author: agent_name.to_string(),
            task_id,
            timestamp: chrono::Utc::now(),
            content: content.to_string(),
        });
        Ok(())
    }

    pub fn read_window(
        &self,
        agent_name: &str,
        tier_name: &str,
        max_chars: Option<usize>,
    ) -> Result<String, WorkflowError> {
        let content = self.read(agent_name, tier_name)?;
        let Some(max_chars) = max_chars else {
            return Ok(content);
        };
        let count = content.chars().count();
        if count <= max_chars {
            return Ok(content);
        }
        let tail = content.chars().skip(count - max_chars).collect::<String>();
        Ok(format!(
            "[showing final {max_chars} of {count} characters]\n{tail}"
        ))
    }

    pub fn entries(
        &self,
        agent_name: &str,
        tier_name: &str,
    ) -> Result<Vec<MemoryEntry>, WorkflowError> {
        let tier_lock = self
            .tiers
            .get(tier_name)
            .ok_or_else(|| WorkflowError::MemoryTierNotFound(tier_name.to_string()))?;
        let tier = tier_lock.read();
        if !tier.readers.allows(agent_name) {
            return Err(WorkflowError::MemoryAccessDenied {
                agent: agent_name.to_string(),
                tier: tier_name.to_string(),
                action: "read".to_string(),
            });
        }
        Ok(tier.entries.clone())
    }

    /// Returns (tier_name, can_read, can_write) for each tier accessible to the agent.
    pub fn list_accessible(&self, agent_name: &str) -> Vec<(String, bool, bool)> {
        self.tiers
            .iter()
            .map(|(name, lock)| {
                let tier = lock.read();
                (
                    name.clone(),
                    tier.readers.allows(agent_name),
                    tier.writers.allows(agent_name),
                )
            })
            .collect()
    }

    /// Snapshot all tiers for persistence.
    pub fn snapshot(&self) -> Vec<MemoryTier> {
        self.tiers
            .values()
            .map(|lock| lock.read().clone())
            .collect()
    }
}

/// Builder for configuring the memory system.
#[derive(Default)]
pub struct MemoryConfig {
    tiers: Vec<MemoryTier>,
}

impl MemoryConfig {
    /// Creates a new empty memory configuration builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a custom memory tier.
    pub fn tier(mut self, tier: MemoryTier) -> Self {
        self.tiers.push(tier);
        self
    }

    /// Adds a global tier readable and writable by all agents.
    pub fn global_tier(self, name: impl Into<String>) -> Self {
        self.tier(MemoryTier {
            name: name.into(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::All,
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: None,
        })
    }

    /// Adds a tier readable by all but writable only by specific agents.
    pub fn broadcast_tier(
        self,
        name: impl Into<String>,
        writers: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.tier(MemoryTier {
            name: name.into(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::Only(writers.into_iter().map(|s| s.into()).collect()),
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: None,
        })
    }

    pub fn global_tier_with_limit(self, name: impl Into<String>, max_bytes: usize) -> Self {
        self.tier(MemoryTier {
            name: name.into(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::All,
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: Some(max_bytes),
        })
    }

    /// Builds the MemoryStore from this configuration.
    pub fn build(self) -> MemoryStore {
        MemoryStore::new(self.tiers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_limits_entries_and_windows_are_enforced() {
        let store = MemoryConfig::new()
            .global_tier_with_limit("notes", 10)
            .build();
        let task_id = uuid::Uuid::new_v4();
        store
            .write_with_context("coder", Some(task_id), "notes", "12345")
            .unwrap();
        store
            .write_with_context("coder", Some(task_id), "notes", "6789")
            .unwrap();
        assert!(matches!(
            store.write("coder", "notes", "x"),
            Err(WorkflowError::MemoryLimitExceeded { .. })
        ));
        assert!(
            store
                .read_window("coder", "notes", Some(4))
                .unwrap()
                .ends_with("6789")
        );
        let entries = store.entries("coder", "notes").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].task_id, Some(task_id));
    }
}
