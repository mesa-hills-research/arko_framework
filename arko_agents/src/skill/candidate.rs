use std::path::Path;

use super::loader::{self, Bundle, Problem};
use super::types::{LoadedSkill, SkillLoadError};
use crate::error::ArkoAgentsError;

/// Reads a skill that someone offers to save, as loading would read it, and
/// returns what loading makes of it. `path` is a skill folder (with a
/// `SKILL.md`), a `.md` file, or a `.skill` or `.zip` bundle.
///
/// The result has the front matter's `name`, `description`, `auto_load`,
/// `models` and `workers`, the body and the `.md` resources, so a skill that
/// passes is a skill that loads. It reads the one skill and doesn't
/// deduplicate or filter. A host still checks what it limits itself, such as
/// the form of the name, the number of files, or links in a folder.
///
/// A bundle is checked further, because its entries would be written to a
/// folder. An entry with an absolute path, a `..` or a link, a `.md` file
/// that isn't UTF-8 text, or more than 16 MB unpacked refuses the whole
/// bundle (see [`read_bundle`]).
pub fn read_candidate(path: impl AsRef<Path>) -> Result<LoadedSkill, SkillLoadError> {
    let path = path.as_ref();
    let text = path.to_string_lossy().to_string();
    let label = format!("candidate:{text}");
    let metadata = std::fs::metadata(path).map_err(|source| SkillLoadError::Io {
        path: text.clone(),
        source,
    })?;
    if metadata.is_dir() {
        return match loader::read_folder_skill(path, &label) {
            None => Err(SkillLoadError::MissingSkillMd { path: text }),
            Some(Ok(skill)) => Ok(skill),
            Some(Err(Problem::Invalid(error))) => Err(error),
            Some(Err(Problem::Unreadable(ArkoAgentsError::FileOperation { path, source }))) => {
                Err(SkillLoadError::Io {
                    path: path.to_string_lossy().to_string(),
                    source,
                })
            }
            Some(Err(Problem::Unreadable(error))) => Err(SkillLoadError::Io {
                path: text,
                source: std::io::Error::other(error.to_string()),
            }),
        };
    }
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.ends_with(".md") {
        loader::read_flat_skill(path, &label)
    } else if name.ends_with(".skill") || name.ends_with(".zip") {
        let data = std::fs::read(path).map_err(|source| SkillLoadError::Io {
            path: text.clone(),
            source,
        })?;
        loader::read_bundle_bytes(&data, &text, &label, Bundle::Candidate)
    } else {
        Err(SkillLoadError::NotASkill { path: text })
    }
}

/// Reads the bytes of a `.skill` or `.zip` bundle as [`read_candidate`] reads
/// a bundle file, with the `SKILL.md` at the root or in one top folder.
///
/// Refuses a bundle that has an entry with an absolute path, a `..` or a
/// link, a `.md` file that isn't UTF-8 text, or more than 16 MB unpacked.
///
/// ```
/// use arko_agents::skill::read_bundle;
///
/// // Not a zip archive.
/// assert!(read_bundle(b"---\nname: x\n---\n").is_err());
/// ```
pub fn read_bundle(bytes: &[u8]) -> Result<LoadedSkill, SkillLoadError> {
    loader::read_bundle_bytes(bytes, "<bundle>", "bundle", Bundle::Candidate)
}
