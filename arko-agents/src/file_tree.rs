//! Deterministic visible file inventories and exact inventory changes.
use crate::error::ArkoAgentsError;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    File,
    Directory,
    Symlink,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Identity {
    device: u64,
    inode: u64,
    born_nanos: u128,
    kind: Kind,
}

#[derive(Debug, Clone)]
struct Entry {
    kind: Kind,
    identity: Option<Identity>,
}

#[derive(Debug, Clone)]
pub(crate) struct FileTreeSnapshot {
    root: PathBuf,
    depth: Option<usize>,
    entries: BTreeMap<String, Entry>,
}

fn identity(metadata: &std::fs::Metadata, kind: Kind) -> Option<Identity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let born_nanos = metadata
            .created()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos();
        Some(Identity {
            device: metadata.dev(),
            inode: metadata.ino(),
            born_nanos,
            kind,
        })
    }
    #[cfg(not(unix))]
    {
        let _ = (metadata, kind);
        None
    }
}

impl FileTreeSnapshot {
    pub(crate) fn capture(root: &Path, depth: Option<usize>) -> Result<Self, ArkoAgentsError> {
        let root = root.canonicalize()?;
        if !root.is_dir() {
            return Err(ArkoAgentsError::Runtime(format!(
                "File-tree root is not a directory: {}",
                root.display()
            )));
        }
        let mut walker = ignore::WalkBuilder::new(&root);
        walker
            .hidden(false)
            .parents(true)
            .git_global(true)
            .git_ignore(true)
            .require_git(false)
            .follow_links(false)
            .max_depth(depth)
            .filter_entry(|entry| entry.depth() == 0 || entry.file_name() != OsStr::new(".git"));
        let mut entries = BTreeMap::new();
        for entry in walker.build() {
            let entry = entry.map_err(ArkoAgentsError::IgnoreError)?;
            if let Some(error) = entry.error() {
                return Err(ArkoAgentsError::Runtime(format!(
                    "File-tree ignore rules could not be applied: {error}"
                )));
            }
            if entry.depth() == 0 {
                continue;
            }
            let path = entry.path();
            let relative = path
                .strip_prefix(&root)
                .map_err(|e| ArkoAgentsError::Runtime(e.to_string()))?;
            let name = relative
                .to_str()
                .ok_or_else(|| {
                    ArkoAgentsError::Runtime(format!(
                        "File-tree path is not valid UTF-8: {:?}",
                        relative
                    ))
                })?
                .replace(std::path::MAIN_SEPARATOR, "/");
            let metadata = std::fs::symlink_metadata(path)?;
            let kind = if metadata.is_symlink() {
                Kind::Symlink
            } else if metadata.is_dir() {
                Kind::Directory
            } else if metadata.is_file() {
                Kind::File
            } else {
                Kind::Other
            };
            entries.insert(
                name,
                Entry {
                    kind,
                    identity: identity(&metadata, kind),
                },
            );
        }
        Ok(Self {
            root,
            depth,
            entries,
        })
    }

    fn label(path: &str, entry: &Entry) -> String {
        let suffix = match entry.kind {
            Kind::Directory => "/",
            Kind::Symlink => "@",
            _ => "",
        };
        // JSON quoting keeps unusual names and newline characters unambiguous.
        serde_json::to_string(&format!("{path}{suffix}")).expect("string serialization")
    }

    pub(crate) fn full(&self) -> String {
        let mut output = format!(
            "{}/\n",
            self.root.file_name().unwrap_or_default().to_string_lossy()
        );
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| left.split('/').cmp(right.split('/')));
        for (path, entry) in entries {
            let indent = "  ".repeat(path.split('/').count());
            let name = path.rsplit('/').next().unwrap_or(path);
            output.push_str(&format!("{indent}{}\n", Self::label(name, entry)));
        }
        output
    }

    pub(crate) fn changes_from(&self, before: &Self) -> Option<String> {
        if self.root != before.root || self.depth != before.depth {
            return Some(format!(
                "Full snapshot (tree scope changed):\n{}",
                self.full()
            ));
        }
        let mut removed = before
            .entries
            .iter()
            .filter(|(p, e)| self.entries.get(*p).is_none_or(|n| n.kind != e.kind))
            .map(|(p, _)| p.clone())
            .collect::<BTreeSet<_>>();
        let mut added = self
            .entries
            .iter()
            .filter(|(p, e)| before.entries.get(*p).is_none_or(|n| n.kind != e.kind))
            .map(|(p, _)| p.clone())
            .collect::<BTreeSet<_>>();
        if removed.is_empty() && added.is_empty() {
            return None;
        }

        let identities = |entries: &BTreeMap<String, Entry>| {
            let mut map: HashMap<Identity, Vec<String>> = HashMap::new();
            for (path, entry) in entries {
                if let Some(id) = &entry.identity {
                    map.entry(id.clone()).or_default().push(path.clone());
                }
            }
            map
        };
        let old_ids = identities(&before.entries);
        let new_ids = identities(&self.entries);
        let mut moves = Vec::new();
        for old in removed.iter() {
            let Some(id) = &before.entries[old].identity else {
                continue;
            };
            let Some(new_paths) = new_ids.get(id) else {
                continue;
            };
            if old_ids[id].len() == 1 && new_paths.len() == 1 && added.contains(&new_paths[0]) {
                moves.push((old.clone(), new_paths[0].clone()));
            }
        }
        for (old, new) in &moves {
            removed.remove(old);
            added.remove(new);
        }
        let mut output = String::new();
        for (old, new) in &moves {
            let covered_by_parent = moves.iter().any(|(parent_old, parent_new)| {
                before.entries[parent_old].kind == Kind::Directory
                    && parent_old != old
                    && old
                        .strip_prefix(&format!("{parent_old}/"))
                        .zip(new.strip_prefix(&format!("{parent_new}/")))
                        .is_some_and(|(a, b)| a == b)
            });
            if !covered_by_parent {
                output.push_str(&format!(
                    "Moved: {} -> {}\n",
                    Self::label(old, &before.entries[old]),
                    Self::label(new, &self.entries[new])
                ));
            }
        }
        for path in removed {
            output.push_str(&format!(
                "Removed from tree: {}\n",
                Self::label(&path, &before.entries[&path])
            ));
        }
        for path in added {
            output.push_str(&format!(
                "Added to tree: {}\n",
                Self::label(&path, &self.entries[&path])
            ));
        }
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("arko-tree-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
        fn capture(&self) -> FileTreeSnapshot {
            FileTreeSnapshot::capture(&self.0, Some(3)).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn inventory_is_sorted_filtered_depth_limited_and_quiet_on_content_edits() {
        let f = Fixture::new();
        std::fs::write(f.0.join("z.txt"), "before").unwrap();
        std::fs::write(f.0.join("a.txt"), "a").unwrap();
        std::fs::create_dir(f.0.join("a")).unwrap();
        std::fs::write(f.0.join("a/child.txt"), "x").unwrap();
        std::fs::write(f.0.join(".gitignore"), "ignored/\n").unwrap();
        for dir in ["ignored", ".git", "one/two/three"] {
            std::fs::create_dir_all(f.0.join(dir)).unwrap();
        }
        std::fs::write(f.0.join("ignored/no.txt"), "x").unwrap();
        std::fs::write(f.0.join("one/two/three/deep.txt"), "x").unwrap();
        let before = f.capture();
        let full = before.full();
        assert!(full.find("a.txt").unwrap() < full.find("z.txt").unwrap());
        assert!(full.find("child.txt").unwrap() < full.find("a.txt").unwrap());
        assert!(
            !full.contains("no.txt") && !full.contains("deep.txt") && !full.contains("\".git/\"")
        );
        std::fs::write(f.0.join("z.txt"), "content edited").unwrap();
        assert!(f.capture().changes_from(&before).is_none());
        assert!(
            FileTreeSnapshot::capture(&f.0, Some(1))
                .unwrap()
                .changes_from(&before)
                .unwrap()
                .contains("scope changed")
        );
    }

    #[test]
    fn additions_removals_and_type_changes_have_exact_paths() {
        let f = Fixture::new();
        std::fs::write(f.0.join("old"), "x").unwrap();
        let before = f.capture();
        std::fs::remove_file(f.0.join("old")).unwrap();
        std::fs::create_dir(f.0.join("old")).unwrap();
        std::fs::write(f.0.join("old/new.txt"), "x").unwrap();
        let delta = f.capture().changes_from(&before).unwrap();
        assert!(delta.contains("Removed from tree: \"old\""));
        assert!(delta.contains("Added to tree: \"old/\""));
        assert!(delta.contains("Added to tree: \"old/new.txt\""));
    }

    #[test]
    fn invalid_ignore_rules_do_not_produce_an_authoritative_snapshot() {
        let fixture = Fixture::new();
        std::fs::write(fixture.0.join(".gitignore"), "[invalid\n").unwrap();
        let error = FileTreeSnapshot::capture(&fixture.0, Some(3)).unwrap_err();
        assert!(error.to_string().contains("ignore rules"));
    }

    #[test]
    fn move_matching_is_unambiguous_and_directory_moves_cover_children() {
        let root = PathBuf::from("/fixture");
        let entry = |inode, kind| Entry {
            kind,
            identity: Some(Identity {
                device: 1,
                inode,
                born_nanos: 123,
                kind,
            }),
        };
        let before = FileTreeSnapshot {
            root: root.clone(),
            depth: Some(3),
            entries: BTreeMap::from([
                ("old".into(), entry(1, Kind::Directory)),
                ("old/a".into(), entry(2, Kind::File)),
            ]),
        };
        let after = FileTreeSnapshot {
            root,
            depth: Some(3),
            entries: BTreeMap::from([
                ("new".into(), entry(1, Kind::Directory)),
                ("new/a".into(), entry(2, Kind::File)),
            ]),
        };
        assert_eq!(
            after.changes_from(&before).unwrap(),
            "Moved: \"old/\" -> \"new/\"\n"
        );
        let mut ambiguous = before.clone();
        ambiguous
            .entries
            .insert("alias".into(), entry(2, Kind::File));
        let delta = after.changes_from(&ambiguous).unwrap();
        assert!(delta.contains("Removed from tree: \"old/a\""));
        assert!(delta.contains("Added to tree: \"new/a\""));
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_not_followed_and_names_are_escaped() {
        let f = Fixture::new();
        let external = Fixture::new();
        std::fs::write(external.0.join("outside.txt"), "x").unwrap();
        std::os::unix::fs::symlink(&external.0, f.0.join("link")).unwrap();
        std::fs::write(f.0.join("line\nbreak"), "x").unwrap();
        let before = f.capture();
        assert!(!before.full().contains("outside.txt"));
        assert!(before.full().contains("line\\nbreak"));
        std::fs::rename(f.0.join("line\nbreak"), f.0.join("renamed")).unwrap();
        let delta = f.capture().changes_from(&before).unwrap();
        if before.entries["line\nbreak"].identity.is_some() {
            assert!(delta.contains("Moved: \"line\\nbreak\" -> \"renamed\""));
        } else {
            assert!(delta.contains("Removed from tree") && delta.contains("Added to tree"));
        }
    }
}
