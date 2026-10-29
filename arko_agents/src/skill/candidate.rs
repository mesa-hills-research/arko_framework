use std::io::Write;
use std::path::{Path, PathBuf};

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
/// folder, so it is refused for the reasons [`read_bundle`] gives.
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
/// link, a `.md` file that isn't UTF-8 text, more than 16 MB unpacked, more
/// than 1,000 files, or a name or size that the local header and the central
/// directory give differently.
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

/// Writes the files of a `.skill` or `.zip` bundle to the new folder `dest`
/// and returns the skill, read as [`read_bundle`] reads it. Every file is
/// written, scripts and other files that aren't text included, and its
/// `SKILL.md` is at the root of the bundle or in one folder, which becomes
/// `dest`. Entries outside that folder are left out.
///
/// The bundle is checked in full before anything is written, by the checks of
/// [`read_bundle`] (an absolute path, a `..`, a link, text that isn't UTF-8,
/// more than 16 MB unpacked, more than 1,000 files or headers that disagree
/// refuse it). The files
/// are then written to a hidden folder beside `dest` and renamed to it, so a
/// failure leaves nothing at `dest`. Missing parent folders are made. `dest`
/// itself must not exist. Files are written without the executable bit,
/// whatever the bundle says.
///
/// ```
/// use std::io::Write;
/// use arko_agents::skill::unpack_bundle;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
/// let options = zip::write::SimpleFileOptions::default();
/// writer.start_file("SKILL.md", options)?;
/// writer.write_all(b"---\nname: tips\ndescription: Some tips.\n---\nBe brief.\n")?;
/// writer.start_file("scripts/run.sh", options)?;
/// writer.write_all(b"echo hi")?;
/// let bundle = writer.finish()?.into_inner();
///
/// let dest = std::env::temp_dir().join(format!("unpack-doc-{}", std::process::id())).join("tips");
/// let skill = unpack_bundle(&bundle, &dest)?;
/// assert_eq!(skill.name, "tips");
/// assert!(dest.join("scripts/run.sh").is_file());
/// # std::fs::remove_dir_all(dest.parent().unwrap())?;
/// # Ok(())
/// # }
/// ```
pub fn unpack_bundle(bytes: &[u8], dest: impl AsRef<Path>) -> Result<LoadedSkill, SkillLoadError> {
    let dest = dest.as_ref();
    let (skill, files) = loader::parse_bundle(bytes, "<bundle>", "bundle", Bundle::Unpack)?;
    let io_error = |path: &Path, source: std::io::Error| SkillLoadError::Io {
        path: path.to_string_lossy().to_string(),
        source,
    };
    if dest.symlink_metadata().is_ok() {
        let exists = std::io::Error::new(std::io::ErrorKind::AlreadyExists, "it exists already");
        return Err(io_error(dest, exists));
    }
    let name = dest.file_name().ok_or_else(|| {
        let none = std::io::Error::new(std::io::ErrorKind::InvalidInput, "it has no name");
        io_error(dest, none)
    })?;
    let parent = match dest.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(parent).map_err(|e| io_error(parent, e))?;

    let staging = parent.join(format!(
        ".{}.unpacking-{}",
        name.to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir(&staging).map_err(|e| io_error(&staging, e))?;
    let written = write_files(&staging, &files)
        .and_then(|()| std::fs::rename(&staging, dest).map_err(|e| io_error(dest, e)));
    if let Err(error) = written {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    Ok(skill)
}

/// Writes `files`, named from the folder `root`, into it.
fn write_files(root: &Path, files: &[(String, Vec<u8>)]) -> Result<(), SkillLoadError> {
    for (name, content) in files {
        let mut path = PathBuf::from(root);
        for part in name
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
        {
            // Checked already. This keeps a mistake from writing elsewhere.
            if part == ".." {
                return Err(SkillLoadError::NotASkill { path: name.clone() });
            }
            path.push(part);
        }
        let io_error = |source| SkillLoadError::Io {
            path: path.to_string_lossy().to_string(),
            source,
        };
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder).map_err(io_error)?;
        }
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o644);
        let mut file = options.open(&path).map_err(io_error)?;
        file.write_all(content).map_err(io_error)?;
    }
    Ok(())
}
