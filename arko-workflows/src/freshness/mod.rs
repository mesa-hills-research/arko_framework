//! Pragmatic stale-context prevention for files, directory listings, memory,
//! and workflow-defined resources.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceId {
    File(PathBuf),
    DirectoryTree(PathBuf),
    MemoryTier(String),
    Custom { namespace: String, key: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorId {
    pub task_id: Uuid,
    pub agent_name: String,
}

impl ActorId {
    pub fn new(task_id: Uuid, agent_name: impl Into<String>) -> Self {
        Self {
            task_id,
            agent_name: agent_name.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResourceEffect {
    ReadFile { parameter: String },
    ReadDirectoryTree { parameter: String },
    WriteFile { parameter: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFreshnessConfig {
    pub block_on_conflict: bool,
    pub tool_effects: HashMap<String, ToolResourceEffect>,
}

impl Default for ContextFreshnessConfig {
    fn default() -> Self {
        Self {
            block_on_conflict: true,
            tool_effects: HashMap::from([
                (
                    "read_file".to_string(),
                    ToolResourceEffect::ReadFile {
                        parameter: "path".to_string(),
                    },
                ),
                (
                    "list_files".to_string(),
                    ToolResourceEffect::ReadDirectoryTree {
                        parameter: "directory".to_string(),
                    },
                ),
                (
                    "write_file".to_string(),
                    ToolResourceEffect::WriteFile {
                        parameter: "path".to_string(),
                    },
                ),
                (
                    "apply_diff".to_string(),
                    ToolResourceEffect::WriteFile {
                        parameter: "path".to_string(),
                    },
                ),
            ]),
        }
    }
}

impl ContextFreshnessConfig {
    pub fn tool_effect(mut self, tool_name: impl Into<String>, effect: ToolResourceEffect) -> Self {
        self.tool_effects.insert(tool_name.into(), effect);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResourceState {
    version: u64,
    last_writer: Option<ActorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObservedRead {
    version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessConflict {
    pub resource: ResourceId,
    pub observed_version: u64,
    pub current_version: u64,
    pub last_writer: Option<ActorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FreshnessSnapshot {
    #[serde(with = "resource_state_map")]
    resources: HashMap<ResourceId, ResourceState>,
    #[serde(with = "observed_read_map")]
    reads: HashMap<ActorId, HashMap<ResourceId, ObservedRead>>,
    #[serde(default, with = "actor_resource_map")]
    writes: HashMap<ActorId, HashSet<ResourceId>>,
}

mod resource_state_map {
    use super::*;
    use serde::{Deserializer, Serializer};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Representation {
        Pairs(Vec<(ResourceId, ResourceState)>),
        Legacy(HashMap<ResourceId, ResourceState>),
    }

    pub fn serialize<S>(
        value: &HashMap<ResourceId, ResourceState>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.iter().collect::<Vec<_>>().serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<HashMap<ResourceId, ResourceState>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match Representation::deserialize(deserializer)? {
            Representation::Pairs(pairs) => pairs.into_iter().collect(),
            Representation::Legacy(map) => map,
        })
    }
}

mod observed_read_map {
    use super::*;
    use serde::{Deserializer, Serializer};

    type Pairs = Vec<(ActorId, Vec<(ResourceId, ObservedRead)>)>;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Representation {
        Pairs(Pairs),
        Legacy(HashMap<ActorId, HashMap<ResourceId, ObservedRead>>),
    }

    pub fn serialize<S>(
        value: &HashMap<ActorId, HashMap<ResourceId, ObservedRead>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .iter()
            .map(|(actor, reads)| (actor, reads.iter().collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<HashMap<ActorId, HashMap<ResourceId, ObservedRead>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match Representation::deserialize(deserializer)? {
            Representation::Pairs(pairs) => pairs
                .into_iter()
                .map(|(actor, reads)| (actor, reads.into_iter().collect()))
                .collect(),
            Representation::Legacy(map) => map,
        })
    }
}

mod actor_resource_map {
    use super::*;
    use serde::{Deserializer, Serializer};

    type Pairs = Vec<(ActorId, Vec<ResourceId>)>;

    pub fn serialize<S>(
        value: &HashMap<ActorId, HashSet<ResourceId>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .iter()
            .map(|(actor, resources)| (actor, resources.iter().collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<HashMap<ActorId, HashSet<ResourceId>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Pairs::deserialize(deserializer)?
            .into_iter()
            .map(|(actor, resources)| (actor, resources.into_iter().collect()))
            .collect())
    }
}

pub struct ResourceTracker {
    workspace: PathBuf,
    config: ContextFreshnessConfig,
    state: RwLock<FreshnessSnapshot>,
}

impl ResourceTracker {
    pub fn new(workspace: PathBuf, config: ContextFreshnessConfig) -> Self {
        Self {
            workspace,
            config,
            state: RwLock::new(FreshnessSnapshot::default()),
        }
    }

    pub fn from_snapshot(
        workspace: PathBuf,
        config: ContextFreshnessConfig,
        snapshot: FreshnessSnapshot,
    ) -> Self {
        Self {
            workspace,
            config,
            state: RwLock::new(snapshot),
        }
    }

    pub fn snapshot(&self) -> FreshnessSnapshot {
        self.state.read().clone()
    }

    pub fn config(&self) -> &ContextFreshnessConfig {
        &self.config
    }

    pub fn resource_for_tool(
        &self,
        tool_name: &str,
        parameters: &HashMap<String, Value>,
    ) -> Option<(ResourceId, bool)> {
        match self.config.tool_effects.get(tool_name)? {
            ToolResourceEffect::ReadFile { parameter } => parameters
                .get(parameter)
                .and_then(Value::as_str)
                .map(|path| (ResourceId::File(self.normalize_path(path)), false)),
            ToolResourceEffect::ReadDirectoryTree { parameter } => parameters
                .get(parameter)
                .and_then(Value::as_str)
                .map(|path| (ResourceId::DirectoryTree(self.normalize_path(path)), false)),
            ToolResourceEffect::WriteFile { parameter } => parameters
                .get(parameter)
                .and_then(Value::as_str)
                .map(|path| (ResourceId::File(self.normalize_path(path)), true)),
        }
    }

    pub fn record_read(&self, actor: &ActorId, resource: ResourceId) {
        let mut state = self.state.write();
        let version = state
            .resources
            .entry(resource.clone())
            .or_insert(ResourceState {
                version: 0,
                last_writer: None,
            })
            .version;
        state
            .reads
            .entry(actor.clone())
            .or_default()
            .insert(resource, ObservedRead { version });
    }

    pub(crate) fn observed_resources(&self, actor: &ActorId) -> Vec<ResourceId> {
        self.state
            .read()
            .reads
            .get(actor)
            .map(|reads| reads.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub(crate) fn written_resources(&self, actor: &ActorId) -> Vec<ResourceId> {
        self.state
            .read()
            .writes
            .get(actor)
            .map(|writes| writes.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub(crate) fn all_written_resources(&self) -> Vec<ResourceId> {
        self.state
            .read()
            .writes
            .values()
            .flat_map(|writes| writes.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn check_write(&self, actor: &ActorId, resource: &ResourceId) -> Vec<FreshnessConflict> {
        let state = self.state.read();
        let Some(reads) = state.reads.get(actor) else {
            return Vec::new();
        };
        reads
            .iter()
            .filter(|(observed, _)| resources_overlap_for_write(observed, resource))
            .filter_map(|(observed, read)| {
                let current = state.resources.get(observed)?;
                (current.version > read.version
                    && current
                        .last_writer
                        .as_ref()
                        .is_some_and(|writer| writer != actor))
                .then(|| FreshnessConflict {
                    resource: observed.clone(),
                    observed_version: read.version,
                    current_version: current.version,
                    last_writer: current.last_writer.clone(),
                })
            })
            .collect()
    }

    /// Record a successful mutation and return readers whose observed context
    /// has just become stale.
    pub fn record_write(&self, actor: &ActorId, resource: ResourceId) -> Vec<ActorId> {
        let mut state = self.state.write();
        state
            .writes
            .entry(actor.clone())
            .or_default()
            .insert(resource.clone());
        let mut affected = vec![resource.clone()];
        if let ResourceId::File(path) = &resource {
            affected.extend(
                state
                    .resources
                    .keys()
                    .filter_map(|candidate| match candidate {
                        ResourceId::DirectoryTree(directory) if path.starts_with(directory) => {
                            Some(candidate.clone())
                        }
                        _ => None,
                    }),
            );
        }

        for affected_resource in &affected {
            let entry = state
                .resources
                .entry(affected_resource.clone())
                .or_insert(ResourceState {
                    version: 0,
                    last_writer: None,
                });
            entry.version = entry.version.saturating_add(1);
            entry.last_writer = Some(actor.clone());
        }

        state
            .reads
            .iter()
            .filter(|(reader, reads)| {
                *reader != actor
                    && reads.keys().any(|read| {
                        affected
                            .iter()
                            .any(|changed| resources_overlap_for_write(read, changed))
                    })
            })
            .map(|(reader, _)| reader.clone())
            .collect()
    }

    pub fn clear_actor(&self, actor: &ActorId) {
        self.state.write().reads.remove(actor);
    }

    fn normalize_path(&self, input: &str) -> PathBuf {
        let path = Path::new(input);
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace.join(path)
        };
        lexical_normalize(&joined)
    }
}

fn resources_overlap_for_write(observed: &ResourceId, written: &ResourceId) -> bool {
    match (observed, written) {
        (ResourceId::File(left), ResourceId::File(right)) => left == right,
        (ResourceId::DirectoryTree(directory), ResourceId::File(file)) => {
            file.starts_with(directory)
        }
        (left, right) => left == right,
    }
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut output = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                output.pop();
            }
            other => output.push(other.as_os_str()),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_named_task_instances_are_distinct_and_conflict() {
        let tracker = ResourceTracker::new(
            PathBuf::from("/workspace"),
            ContextFreshnessConfig::default(),
        );
        let reader = ActorId::new(Uuid::new_v4(), "coder");
        let writer = ActorId::new(Uuid::new_v4(), "coder");
        let resource = ResourceId::File(PathBuf::from("/workspace/src/lib.rs"));
        tracker.record_read(&reader, resource.clone());
        tracker.record_write(&writer, resource.clone());
        assert_eq!(tracker.check_write(&reader, &resource).len(), 1);
    }

    #[test]
    fn file_creation_invalidates_directory_tree_read() {
        let tracker = ResourceTracker::new(
            PathBuf::from("/workspace"),
            ContextFreshnessConfig::default(),
        );
        let reader = ActorId::new(Uuid::new_v4(), "planner");
        let writer = ActorId::new(Uuid::new_v4(), "coder");
        tracker.record_read(
            &reader,
            ResourceId::DirectoryTree(PathBuf::from("/workspace/src")),
        );
        let stale = tracker.record_write(
            &writer,
            ResourceId::File(PathBuf::from("/workspace/src/new.rs")),
        );
        assert_eq!(stale, vec![reader]);
    }

    #[test]
    fn populated_snapshot_round_trips_through_json() {
        let workspace = PathBuf::from("/workspace");
        let tracker = ResourceTracker::new(workspace.clone(), ContextFreshnessConfig::default());
        let reader = ActorId::new(Uuid::new_v4(), "reviewer");
        let writer = ActorId::new(Uuid::new_v4(), "coder");
        let resource = ResourceId::File(workspace.join("src/lib.rs"));
        tracker.record_read(&reader, resource.clone());
        tracker.record_write(&writer, resource.clone());

        let json = serde_json::to_string(&tracker.snapshot()).unwrap();
        let snapshot: FreshnessSnapshot = serde_json::from_str(&json).unwrap();
        let restored =
            ResourceTracker::from_snapshot(workspace, ContextFreshnessConfig::default(), snapshot);

        let conflicts = restored.check_write(&reader, &resource);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].last_writer.as_ref(), Some(&writer));
        assert_eq!(restored.written_resources(&writer), vec![resource]);
    }

    #[test]
    fn legacy_empty_json_snapshot_still_deserializes() {
        let snapshot: FreshnessSnapshot =
            serde_json::from_str(r#"{"resources":{},"reads":{}}"#).unwrap();
        assert!(snapshot.resources.is_empty());
        assert!(snapshot.reads.is_empty());
        assert!(snapshot.writes.is_empty());
    }
}
