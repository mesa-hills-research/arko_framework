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

/// One write to a tier, with its text: what
/// [`MemoryStore::entries`] returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: uuid::Uuid,
    pub author: String,
    #[serde(default)]
    pub task_id: Option<uuid::Uuid>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content: String,
}

/// One write to a tier, as the tier stores it. Its text is stored once: as
/// a range of the tier's `content` while it is there (every write of an
/// append-only tier, and the latest of a full-edit tier), otherwise here
/// (an earlier version of a full-edit tier).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryWrite {
    pub id: uuid::Uuid,
    pub author: String,
    #[serde(default)]
    pub task_id: Option<uuid::Uuid>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// The text, when it is not in the tier's content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Where the text is in the tier's content: byte offsets, start
    /// inclusive and end exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(usize, usize)>,
}

/// A single memory tier with content and access policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTier {
    pub name: String,
    pub content: String,
    pub readers: AccessScope,
    pub writers: AccessScope,
    pub write_mode: WriteMode,
    /// The write history, oldest first; see [`MemoryWrite`].
    #[serde(default)]
    pub entries: Vec<MemoryWrite>,
    /// The most bytes the content may hold; `None` (the default) is no
    /// limit.
    #[serde(default)]
    pub max_bytes: Option<usize>,
}

impl MemoryTier {
    /// The text of `write`, one of this tier's entries.
    pub fn text_of<'a>(&'a self, write: &'a MemoryWrite) -> &'a str {
        match (&write.content, write.range) {
            (Some(content), _) => content,
            (None, Some((start, end))) => self.content.get(start..end).unwrap_or_default(),
            (None, None) => "",
        }
    }

    /// The write history with each write's text.
    pub fn history(&self) -> Vec<MemoryEntry> {
        self.entries
            .iter()
            .map(|write| MemoryEntry {
                id: write.id,
                author: write.author.clone(),
                task_id: write.task_id,
                timestamp: write.timestamp,
                content: self.text_of(write).to_string(),
            })
            .collect()
    }

    /// Stores each write's text once: a text that is (still) in the content
    /// becomes a range of it. A tier saved before (whose entries each held
    /// their text) is converted: an append-only tier's writes from the
    /// end of its content back, a full-edit tier's latest write.
    fn store_texts_once(&mut self) {
        match self.write_mode {
            WriteMode::AppendOnly => {
                let mut end = self.content.len();
                for write in self.entries.iter_mut().rev() {
                    let Some(text) = write.content.as_deref() else {
                        match write.range {
                            // Already a range: the ones before end at its
                            // start.
                            Some((start, _)) => {
                                end = start.saturating_sub(1);
                                continue;
                            }
                            None => break,
                        }
                    };
                    if !self.content.is_char_boundary(end) || !self.content[..end].ends_with(text) {
                        break;
                    }
                    let start = end - text.len();
                    write.range = Some((start, end));
                    write.content = None;
                    // The newline an append puts before each later write.
                    match start.checked_sub(1) {
                        Some(before) if self.content.as_bytes()[before] == b'\n' => end = before,
                        _ => break,
                    }
                }
            }
            WriteMode::FullEdit => {
                if let Some(write) = self.entries.last_mut()
                    && write.content.as_deref() == Some(self.content.as_str())
                {
                    write.range = Some((0, self.content.len()));
                    write.content = None;
                }
            }
        }
    }

    /// Stores every write's text itself (before the write mode changes on
    /// resume, since ranges follow the write mode).
    fn hold_texts(&mut self) {
        for index in 0..self.entries.len() {
            if self.entries[index].content.is_none() {
                let text = self.text_of(&self.entries[index]).to_string();
                let write = &mut self.entries[index];
                write.content = Some(text);
                write.range = None;
            }
        }
    }
}

fn same_mode(left: &WriteMode, right: &WriteMode) -> bool {
    matches!(
        (left, right),
        (WriteMode::AppendOnly, WriteMode::AppendOnly) | (WriteMode::FullEdit, WriteMode::FullEdit)
    )
}

/// The complete memory store. Each tier is independently lockable.
#[derive(Default)]
pub struct MemoryStore {
    tiers: HashMap<String, RwLock<MemoryTier>>,
}

impl MemoryStore {
    /// Creates a new memory store from the given tiers. Each write's text is
    /// stored once (see [`MemoryWrite`]); tiers saved before are converted.
    pub fn new(tiers: Vec<MemoryTier>) -> Self {
        let map = tiers
            .into_iter()
            .map(|mut tier| {
                tier.store_texts_once();
                (tier.name.clone(), RwLock::new(tier))
            })
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
        let range = match tier.write_mode {
            WriteMode::AppendOnly => {
                if !tier.content.is_empty() {
                    tier.content.push('\n');
                }
                let start = tier.content.len();
                tier.content.push_str(content);
                (start, tier.content.len())
            }
            WriteMode::FullEdit => {
                // The version replaced keeps its own text.
                let replaced = std::mem::take(&mut tier.content);
                if let Some(latest) = tier
                    .entries
                    .last_mut()
                    .filter(|latest| latest.content.is_none())
                {
                    latest.content = Some(
                        latest
                            .range
                            .and_then(|(start, end)| replaced.get(start..end))
                            .unwrap_or_default()
                            .to_string(),
                    );
                    latest.range = None;
                }
                tier.content = content.to_string();
                (0, tier.content.len())
            }
        };
        tier.entries.push(MemoryWrite {
            id: uuid::Uuid::new_v4(),
            author: agent_name.to_string(),
            task_id,
            timestamp: chrono::Utc::now(),
            content: None,
            range: Some(range),
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
        Ok(tier.history())
    }

    /// `(tier_name, can_read, can_write)` for each tier the agent can read
    /// or write, sorted by name; tiers it can't access are left out.
    pub fn list_accessible(&self, agent_name: &str) -> Vec<(String, bool, bool)> {
        let mut accessible = self
            .tiers
            .iter()
            .filter_map(|(name, lock)| {
                let tier = lock.read();
                let (read, write) = (
                    tier.readers.allows(agent_name),
                    tier.writers.allows(agent_name),
                );
                (read || write).then(|| (name.clone(), read, write))
            })
            .collect::<Vec<_>>();
        accessible.sort();
        accessible
    }

    /// Whether the agent can read any tier, and whether it can write any:
    /// which memory tools it is offered.
    pub fn access_of(&self, agent_name: &str) -> (bool, bool) {
        self.tiers
            .values()
            .fold((false, false), |(read, write), lock| {
                let tier = lock.read();
                (
                    read || tier.readers.allows(agent_name),
                    write || tier.writers.allows(agent_name),
                )
            })
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

    /// Adds a tier readable by all and writable only by `writers`, with a
    /// byte limit (opt-in; tiers have none by default).
    pub fn broadcast_tier_with_limit(
        self,
        name: impl Into<String>,
        writers: impl IntoIterator<Item = impl Into<String>>,
        max_bytes: usize,
    ) -> Self {
        self.tier(MemoryTier {
            name: name.into(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::Only(writers.into_iter().map(|s| s.into()).collect()),
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: Some(max_bytes),
        })
    }

    /// Adds a global tier with a byte limit (opt-in; tiers have none by
    /// default).
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

    /// The configured tiers.
    pub fn tiers(&self) -> &[MemoryTier] {
        &self.tiers
    }

    /// Tier names must be unique, and every agent an access set names must
    /// be registered.
    pub(crate) fn validate(&self, agents: &HashSet<String>) -> Result<(), WorkflowError> {
        let mut names = HashSet::new();
        for tier in &self.tiers {
            if !names.insert(tier.name.as_str()) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "duplicate memory tier '{}'",
                    tier.name
                )));
            }
            for (role, scope) in [("readers", &tier.readers), ("writers", &tier.writers)] {
                if let AccessScope::Only(named) = scope {
                    let mut named = named.iter().collect::<Vec<_>>();
                    named.sort();
                    if let Some(unknown) = named.into_iter().find(|agent| !agents.contains(*agent))
                    {
                        return Err(WorkflowError::BuilderInvalid(format!(
                            "memory tier '{}' {role} name unknown agent '{unknown}'",
                            tier.name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The store of a resumed run: the configured tiers, each with the
    /// content and history the checkpoint saved under its name (or under a
    /// name `renames` maps to it); access, limit and write mode come from
    /// this configuration. A configured tier the checkpoint lacks starts as
    /// configured. A checkpoint tier matching no configured tier is an error
    /// unless `dropped` names it (it is then discarded).
    pub(crate) fn restore(
        self,
        saved: Vec<MemoryTier>,
        renames: &HashMap<String, String>,
        dropped: &HashSet<String>,
    ) -> Result<MemoryStore, WorkflowError> {
        let mut tiers = self.tiers;
        let mut restored_from: HashMap<String, String> = HashMap::new();
        let mut saved = saved;
        saved.sort_by(|left, right| left.name.cmp(&right.name));
        for from in renames.keys() {
            if !saved.iter().any(|tier| &tier.name == from) {
                log::warn!(
                    "rename_memory_tier_on_resume: the checkpoint has no memory tier '{from}'"
                );
            }
        }
        for name in dropped {
            if !saved.iter().any(|tier| &tier.name == name) {
                log::warn!(
                    "drop_memory_tier_on_resume: the checkpoint has no memory tier '{name}'"
                );
            }
        }
        for mut old in saved {
            if dropped.contains(&old.name) {
                log::info!(
                    "Memory tier '{}' of the checkpoint is dropped on resume",
                    old.name
                );
                continue;
            }
            let target = renames.get(&old.name).unwrap_or(&old.name).clone();
            let Some(tier) = tiers.iter_mut().find(|tier| tier.name == target) else {
                return Err(WorkflowError::BuilderInvalid(
                    match renames.get(&old.name) {
                        Some(target) => format!(
                            "memory tier '{}' of the checkpoint is renamed to '{target}', which is not configured",
                            old.name
                        ),
                        None => format!(
                            "memory tier '{name}' of the checkpoint is not configured: configure it, map it to a configured tier with rename_memory_tier_on_resume(\"{name}\", ..), or discard it with drop_memory_tier_on_resume(\"{name}\")",
                            name = old.name
                        ),
                    },
                ));
            };
            if let Some(earlier) = restored_from.insert(target.clone(), old.name.clone()) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "memory tiers '{earlier}' and '{}' of the checkpoint both restore tier '{target}'",
                    old.name
                )));
            }
            if !same_mode(&old.write_mode, &tier.write_mode) {
                old.hold_texts();
            }
            if tier
                .max_bytes
                .is_some_and(|limit| old.content.len() > limit)
            {
                log::warn!(
                    "Memory tier '{target}' is restored with {} bytes, over its configured limit of {} bytes; writes that grow it fail until it is smaller",
                    old.content.len(),
                    tier.max_bytes.unwrap_or_default()
                );
            }
            tier.content = old.content;
            tier.entries = old.entries;
        }
        Ok(MemoryStore::new(tiers))
    }

    /// Builds the MemoryStore from this configuration.
    pub fn build(self) -> MemoryStore {
        MemoryStore::new(self.tiers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(store: &MemoryStore, name: &str) -> MemoryTier {
        store.tiers[name].read().clone()
    }

    /// Each write's text is stored once: in the tier's content for every
    /// append and the latest full edit, in the entry for an earlier
    /// version. The history reads the same either way.
    #[test]
    fn a_write_s_text_is_stored_once() {
        let store = MemoryConfig::new()
            .global_tier("log")
            .tier(MemoryTier {
                name: "plan".into(),
                content: String::new(),
                readers: AccessScope::All,
                writers: AccessScope::All,
                write_mode: WriteMode::FullEdit,
                entries: Vec::new(),
                max_bytes: None,
            })
            .build();
        for text in ["first", "second", "third"] {
            store.write("coder", "log", text).unwrap();
            store.write("coder", "plan", text).unwrap();
        }
        let log = tier(&store, "log");
        assert_eq!(log.content, "first\nsecond\nthird");
        assert!(log.entries.iter().all(|write| write.content.is_none()));
        assert_eq!(
            log.entries
                .iter()
                .map(|write| write.range)
                .collect::<Vec<_>>(),
            [Some((0, 5)), Some((6, 12)), Some((13, 18))]
        );
        let saved = serde_json::to_string(&log).unwrap();
        assert_eq!(saved.matches("second").count(), 1, "{saved}");

        let plan = tier(&store, "plan");
        assert_eq!(plan.content, "third");
        assert_eq!(
            plan.entries
                .iter()
                .map(|write| (write.content.as_deref(), write.range))
                .collect::<Vec<_>>(),
            [
                (Some("first"), None),
                (Some("second"), None),
                (None, Some((0, 5)))
            ]
        );
        for name in ["log", "plan"] {
            assert_eq!(
                store
                    .entries("coder", name)
                    .unwrap()
                    .into_iter()
                    .map(|entry| entry.content)
                    .collect::<Vec<_>>(),
                ["first", "second", "third"]
            );
        }
    }

    /// A tier saved before (each entry held its text, and the content held
    /// it again) loads with the texts stored once and the same history.
    #[test]
    fn a_tier_saved_with_its_texts_twice_loads_with_them_once() {
        let entry = |text: &str| {
            serde_json::json!({
                "id": uuid::Uuid::new_v4(),
                "author": "coder",
                "timestamp": "2026-09-01T12:00:00Z",
                "content": text
            })
        };
        let saved = |mode: &str, content: &str, texts: &[&str]| -> MemoryTier {
            serde_json::from_value(serde_json::json!({
                "name": "notes",
                "content": content,
                "readers": "All",
                "writers": "All",
                "write_mode": mode,
                "entries": texts.iter().map(|text| entry(text)).collect::<Vec<_>>()
            }))
            .unwrap()
        };
        // Initial content, then three appends (one multi-byte).
        let store = MemoryStore::new(vec![saved(
            "AppendOnly",
            "seed\nalpha\nbeta ü\ngamma",
            &["alpha", "beta ü", "gamma"],
        )]);
        let notes = tier(&store, "notes");
        assert!(notes.entries.iter().all(|write| write.content.is_none()));
        assert_eq!(
            store
                .entries("coder", "notes")
                .unwrap()
                .into_iter()
                .map(|entry| entry.content)
                .collect::<Vec<_>>(),
            ["alpha", "beta ü", "gamma"]
        );
        store.write("coder", "notes", "delta").unwrap();
        assert_eq!(store.entries("coder", "notes").unwrap()[3].content, "delta");

        let store = MemoryStore::new(vec![saved("FullEdit", "v2", &["v1", "v2"])]);
        let notes = tier(&store, "notes");
        assert_eq!(notes.entries[0].content.as_deref(), Some("v1"));
        assert_eq!(notes.entries[1].range, Some((0, 2)));
    }

    fn configured() -> MemoryConfig {
        MemoryConfig::new()
            .tier(MemoryTier {
                name: "notes".into(),
                content: String::new(),
                readers: AccessScope::Only(HashSet::from(["reviewer".to_string()])),
                writers: AccessScope::Only(HashSet::from(["reviewer".to_string()])),
                write_mode: WriteMode::AppendOnly,
                entries: Vec::new(),
                max_bytes: Some(1_000),
            })
            .global_tier("new")
    }

    fn checkpoint() -> Vec<MemoryTier> {
        let store = MemoryConfig::new()
            .global_tier("notes")
            .global_tier("old")
            .build();
        store.write("coder", "notes", "kept").unwrap();
        store.write("coder", "old", "archived").unwrap();
        store.snapshot()
    }

    /// On resume the configuration decides access and limits, the
    /// checkpoint supplies content and history by name, and a configured
    /// tier it lacks starts as configured.
    #[test]
    fn a_resume_takes_access_from_the_configuration_and_content_from_the_checkpoint() {
        let store = configured()
            .restore(
                checkpoint(),
                &HashMap::new(),
                &HashSet::from(["old".to_string()]),
            )
            .unwrap();
        let notes = tier(&store, "notes");
        assert_eq!(notes.content, "kept");
        assert_eq!(notes.max_bytes, Some(1_000));
        assert!(
            store.read("coder", "notes").is_err(),
            "access from the configuration"
        );
        assert_eq!(store.read("reviewer", "notes").unwrap(), "kept");
        assert_eq!(
            store.entries("reviewer", "notes").unwrap()[0].content,
            "kept"
        );
        assert_eq!(store.read("coder", "new").unwrap(), "");
        assert!(store.read("coder", "old").is_err(), "dropped");
    }

    /// A checkpoint tier matching no configured tier stops the resume,
    /// unless it is dropped or renamed to a configured tier; two tiers
    /// can't restore one.
    #[test]
    fn a_checkpoint_tier_without_a_configured_tier_is_refused_unless_mapped_or_dropped() {
        let error = configured()
            .restore(checkpoint(), &HashMap::new(), &HashSet::new())
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("memory tier 'old' of the checkpoint is not configured"),
            "{error}"
        );

        let renamed = configured()
            .restore(
                checkpoint(),
                &HashMap::from([("old".to_string(), "new".to_string())]),
                &HashSet::new(),
            )
            .unwrap();
        assert_eq!(renamed.read("coder", "new").unwrap(), "archived");

        let clash = configured()
            .restore(
                checkpoint(),
                &HashMap::from([("old".to_string(), "notes".to_string())]),
                &HashSet::new(),
            )
            .err()
            .unwrap();
        assert!(
            clash.to_string().contains("both restore tier 'notes'"),
            "{clash}"
        );

        let unconfigured = configured()
            .restore(
                checkpoint(),
                &HashMap::from([("old".to_string(), "missing".to_string())]),
                &HashSet::new(),
            )
            .err()
            .unwrap();
        assert!(
            unconfigured
                .to_string()
                .contains("renamed to 'missing', which is not configured"),
            "{unconfigured}"
        );
    }

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
