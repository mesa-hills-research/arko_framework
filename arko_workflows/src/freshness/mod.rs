//! Pragmatic stale-context prevention for files, directory listings, memory,
//! and workflow-defined resources.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

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

/// A field omitted in JSON or YAML takes its [`Default`] value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
                    "edit_file".to_string(),
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
                // Claude Code's file tools, for agents whose model runs in
                // Claude Code (arko_agents::claude_code).
                (
                    "Read".to_string(),
                    ToolResourceEffect::ReadFile {
                        parameter: "file_path".to_string(),
                    },
                ),
                (
                    "Write".to_string(),
                    ToolResourceEffect::WriteFile {
                        parameter: "file_path".to_string(),
                    },
                ),
                (
                    "Edit".to_string(),
                    ToolResourceEffect::WriteFile {
                        parameter: "file_path".to_string(),
                    },
                ),
                (
                    "MultiEdit".to_string(),
                    ToolResourceEffect::WriteFile {
                        parameter: "file_path".to_string(),
                    },
                ),
                (
                    "NotebookEdit".to_string(),
                    ToolResourceEffect::WriteFile {
                        parameter: "notebook_path".to_string(),
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
    /// Actors whose task has stopped but can be continued. Their reads are
    /// kept, but no write notifies them. Not saved, since after a resume no task
    /// runs, so every actor of the snapshot starts here.
    finished: RwLock<HashSet<ActorId>>,
}

impl ResourceTracker {
    pub fn new(workspace: PathBuf, config: ContextFreshnessConfig) -> Self {
        Self {
            workspace,
            config,
            state: RwLock::new(FreshnessSnapshot::default()),
            finished: RwLock::new(HashSet::new()),
        }
    }

    /// A tracker restored from a checkpoint. No task runs after a resume, so
    /// every actor the snapshot recorded counts as finished (see
    /// [`finish_actor`](Self::finish_actor)) until its task is continued.
    pub fn from_snapshot(
        workspace: PathBuf,
        config: ContextFreshnessConfig,
        snapshot: FreshnessSnapshot,
    ) -> Self {
        let finished = snapshot.reads.keys().cloned().collect();
        Self {
            workspace,
            config,
            state: RwLock::new(snapshot),
            finished: RwLock::new(finished),
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
        self.resource_for_tool_in(tool_name, parameters, &self.workspace)
    }

    /// [`resource_for_tool`](Self::resource_for_tool) for a call made by a
    /// run working in `working_directory` (the task's working directory, which the
    /// workflow's workspace, or the run's own, see
    /// [`RunOptions::workspace`](crate::workflow::RunOptions::workspace)). A
    /// path parameter's key is the path the built-in file tools
    /// (`read_file`, `write_file`, `edit_file`, `apply_diff`, `list_files`)
    /// open, resolved against `working_directory` with arko_agents'
    /// [`resolve_path`](arko_agents::utils::resolve_path), which the
    /// runtime's workspace check also uses. `None` for a path that can't be
    /// resolved, which no tool opens.
    pub fn resource_for_tool_in(
        &self,
        tool_name: &str,
        parameters: &HashMap<String, Value>,
        working_directory: &std::path::Path,
    ) -> Option<(ResourceId, bool)> {
        let resolve = |path: &str| arko_agents::utils::resolve_path(path, working_directory).ok();
        match self.config.tool_effects.get(tool_name)? {
            ToolResourceEffect::ReadFile { parameter } => parameters
                .get(parameter)
                .and_then(Value::as_str)
                .and_then(resolve)
                .map(|path| (ResourceId::File(path), false)),
            ToolResourceEffect::ReadDirectoryTree { parameter } => parameters
                .get(parameter)
                .and_then(Value::as_str)
                .and_then(resolve)
                .map(|path| (ResourceId::DirectoryTree(path), false)),
            ToolResourceEffect::WriteFile { parameter } => parameters
                .get(parameter)
                .and_then(Value::as_str)
                .and_then(resolve)
                .map(|path| (ResourceId::File(path), true)),
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

    /// Record a successful mutation and return the readers whose observed
    /// context has just become stale and who are to be told, excluding actors
    /// whose task has finished (their reads are kept for a continuation,
    /// which is told what changed, see [`resume_actor`](Self::resume_actor)).
    pub fn record_write(&self, actor: &ActorId, resource: ResourceId) -> Vec<ActorId> {
        let finished = self.finished.read();
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
                    && !finished.contains(*reader)
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
        self.finished.write().remove(actor);
    }

    /// Records that `actor`'s task stopped. A task that can be `continued`
    /// (completed, waiting for input, or paused) keeps its reads, so a
    /// continuation still sees what changed, but no longer gets freshness
    /// notices or fires conflict hooks. The reads of any other task are
    /// forgotten.
    pub fn finish_actor(&self, actor: &ActorId, continued: bool) {
        if continued {
            self.finished.write().insert(actor.clone());
        } else {
            self.clear_actor(actor);
        }
    }

    /// Records that `actor`'s finished task runs again, and returns the
    /// resources it read that another actor changed since, sorted.
    pub fn resume_actor(&self, actor: &ActorId) -> Vec<ResourceId> {
        self.finished.write().remove(actor);
        self.changed_since_read(actor)
    }

    /// The resources `actor` read that another actor changed since, sorted.
    pub fn changed_since_read(&self, actor: &ActorId) -> Vec<ResourceId> {
        let state = self.state.read();
        let Some(reads) = state.reads.get(actor) else {
            return Vec::new();
        };
        let mut changed = reads
            .iter()
            .filter(|(resource, read)| {
                state.resources.get(*resource).is_some_and(|current| {
                    current.version > read.version
                        && current
                            .last_writer
                            .as_ref()
                            .is_some_and(|writer| writer != actor)
                })
            })
            .map(|(resource, _)| resource.clone())
            .collect::<Vec<_>>();
        changed.sort_by_key(resource_sort_key);
        changed
    }

    /// The message a continued task gets about `changed` resources it had
    /// read, or `None` when there are none.
    pub(crate) fn changed_notice(&self, changed: &[ResourceId]) -> Option<String> {
        if changed.is_empty() {
            return None;
        }
        let lines = changed
            .iter()
            .map(|resource| format!("- {}", self.describe(resource)))
            .collect::<Vec<_>>()
            .join("\n");
        Some(format!(
            "Changed since you last read them (another agent wrote them after your read):\n{lines}\nRe-read any of these before you edit it or rely on its content."
        ))
    }

    /// A resource as an agent is told about it: a path relative to the
    /// workspace when inside it.
    fn describe(&self, resource: &ResourceId) -> String {
        let path = |path: &std::path::Path| {
            path.strip_prefix(&self.workspace)
                .unwrap_or(path)
                .display()
                .to_string()
        };
        match resource {
            ResourceId::File(file) => format!("file `{}`", path(file)),
            ResourceId::DirectoryTree(directory) => {
                format!("the listing of directory `{}`", path(directory))
            }
            ResourceId::MemoryTier(tier) => format!("memory tier `{tier}`"),
            ResourceId::Custom { namespace, key } => format!("`{namespace}:{key}`"),
        }
    }
}

fn resource_sort_key(resource: &ResourceId) -> (u8, String) {
    match resource {
        ResourceId::File(path) => (0, path.display().to_string()),
        ResourceId::DirectoryTree(path) => (1, path.display().to_string()),
        ResourceId::MemoryTier(tier) => (2, tier.clone()),
        ResourceId::Custom { namespace, key } => (3, format!("{namespace}:{key}")),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Fields omitted in JSON take the `Default` values (both were
    /// required). Given ones replace them whole.
    #[test]
    fn omitted_freshness_fields_take_the_defaults() {
        let defaults = ContextFreshnessConfig::default();
        let effects = |config: &ContextFreshnessConfig| {
            let mut names = config.tool_effects.keys().cloned().collect::<Vec<_>>();
            names.sort();
            names
        };
        let loaded: ContextFreshnessConfig =
            serde_json::from_str(r#"{"block_on_conflict": false}"#).unwrap();
        assert!(!loaded.block_on_conflict);
        assert_eq!(effects(&loaded), effects(&defaults));
        let empty: ContextFreshnessConfig = serde_json::from_str("{}").unwrap();
        assert!(empty.block_on_conflict);
        assert_eq!(effects(&empty), effects(&defaults));
        let custom: ContextFreshnessConfig = serde_json::from_str(
            r#"{"tool_effects": {"edit": {"WriteFile": {"parameter": "file"}}}}"#,
        )
        .unwrap();
        assert!(custom.block_on_conflict);
        assert_eq!(effects(&custom), ["edit"]);
        let saved = serde_json::to_string(&defaults).unwrap();
        let reloaded: ContextFreshnessConfig = serde_json::from_str(&saved).unwrap();
        assert_eq!(effects(&reloaded), effects(&defaults));
    }

    #[cfg(unix)]
    #[test]
    fn tool_paths_resolve_as_the_built_in_tools_open_them() {
        // The default effects: read_file, write_file, edit_file and
        // apply_diff on `path`, list_files on `directory`.
        let tracker = ResourceTracker::new(
            PathBuf::from("/workspace"),
            ContextFreshnessConfig::default(),
        );
        let key = |tool: &str, parameter: &str, path: &str| {
            let parameters = HashMap::from([(parameter.to_string(), Value::from(path))]);
            tracker.resource_for_tool(tool, &parameters).unwrap()
        };
        let file = |path: &str| match key("read_file", "path", path) {
            (ResourceId::File(path), false) => path,
            other => panic!("unexpected resource {other:?}"),
        };
        assert_eq!(file("src/./lib.rs"), PathBuf::from("/workspace/src/lib.rs"));
        assert_eq!(
            file("src/../Cargo.toml"),
            PathBuf::from("/workspace/Cargo.toml")
        );
        assert_eq!(file("/other/./a/../b"), PathBuf::from("/other/b"));
        assert_eq!(file("../../../../etc/hosts"), PathBuf::from("/etc/hosts"));
        // The tools trim the path first.
        assert_eq!(
            file(" src/lib.rs\n"),
            PathBuf::from("/workspace/src/lib.rs")
        );
        for tool in ["write_file", "edit_file", "apply_diff"] {
            assert_eq!(
                key(tool, "path", "\tsrc/lib.rs "),
                (
                    ResourceId::File(PathBuf::from("/workspace/src/lib.rs")),
                    true
                )
            );
        }
        // No shell expansion: the tools open the path as given.
        assert_eq!(file("~/notes"), PathBuf::from("/workspace/~/notes"));
        assert_eq!(file("$HOME/notes"), PathBuf::from("/workspace/$HOME/notes"));
        // A path no tool can open has no key.
        let nul = HashMap::from([("path".to_string(), Value::from("a\0b"))]);
        assert!(tracker.resource_for_tool("read_file", &nul).is_none());
        // Trailing slashes don't change the directory list_files lists, so
        // a later write inside it conflicts with the read.
        let directory = |path: &str| match key("list_files", "directory", path) {
            (ResourceId::DirectoryTree(path), false) => path,
            other => panic!("unexpected resource {other:?}"),
        };
        assert_eq!(
            directory(" src/core/ "),
            PathBuf::from("/workspace/src/core")
        );
        assert_eq!(
            directory("src/core//"),
            PathBuf::from("/workspace/src/core")
        );
        assert_eq!(directory("."), PathBuf::from("/workspace"));
        // `\` is an ordinary character on Unix, as for the tools.
        assert_eq!(
            directory(r"src\core"),
            PathBuf::from(r"/workspace/src\core")
        );
        assert!(resources_overlap_for_write(
            &ResourceId::DirectoryTree(directory("src/core")),
            &key("write_file", "path", "src/core/mod.rs").0,
        ));
    }

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

    /// A finished reader is not told of a write (no notice, no conflict
    /// hook), but its reads are kept: when its task is continued it learns
    /// what changed, and a write it attempts without reading again still
    /// conflicts. A task that can't be continued forgets its reads.
    #[test]
    fn a_finished_reader_is_told_what_changed_when_it_is_continued() {
        let workspace = PathBuf::from("/workspace");
        let tracker = ResourceTracker::new(workspace.clone(), ContextFreshnessConfig::default());
        let reader = ActorId::new(Uuid::new_v4(), "reviewer");
        let writer = ActorId::new(Uuid::new_v4(), "coder");
        let file = ResourceId::File(workspace.join("src/lib.rs"));
        let untouched = ResourceId::File(workspace.join("README.md"));
        let tier = ResourceId::MemoryTier("notes".into());
        for resource in [&file, &untouched, &tier] {
            tracker.record_read(&reader, resource.clone());
        }
        tracker.finish_actor(&reader, true);
        assert!(tracker.record_write(&writer, file.clone()).is_empty());
        assert!(tracker.record_write(&writer, tier.clone()).is_empty());
        // Its own writes don't count.
        tracker.record_write(&reader, untouched.clone());
        let changed = tracker.resume_actor(&reader);
        assert_eq!(changed, [file.clone(), tier.clone()]);
        let notice = tracker.changed_notice(&changed).unwrap();
        assert!(notice.contains("- file `src/lib.rs`"), "{notice}");
        assert!(notice.contains("- memory tier `notes`"), "{notice}");
        assert!(!notice.contains("README"), "{notice}");
        assert!(tracker.changed_notice(&[]).is_none());
        // Running again, it is told of writes and blocked until it rereads.
        assert!(!tracker.check_write(&reader, &file).is_empty());
        assert_eq!(
            tracker.record_write(&writer, file.clone()),
            std::slice::from_ref(&reader)
        );

        tracker.finish_actor(&reader, false);
        assert!(tracker.observed_resources(&reader).is_empty());
        assert!(tracker.resume_actor(&reader).is_empty());
    }

    /// After a resume no task runs: every actor of the checkpoint counts as
    /// finished until its task is continued.
    #[test]
    fn restored_readers_count_as_finished() {
        let workspace = PathBuf::from("/workspace");
        let tracker = ResourceTracker::new(workspace.clone(), ContextFreshnessConfig::default());
        let reader = ActorId::new(Uuid::new_v4(), "reviewer");
        let writer = ActorId::new(Uuid::new_v4(), "coder");
        let file = ResourceId::File(workspace.join("src/lib.rs"));
        tracker.record_read(&reader, file.clone());
        let restored = ResourceTracker::from_snapshot(
            workspace,
            ContextFreshnessConfig::default(),
            tracker.snapshot(),
        );
        assert!(restored.record_write(&writer, file.clone()).is_empty());
        assert_eq!(restored.resume_actor(&reader), [file]);
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
