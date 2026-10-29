use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::config::{EmbeddedSkill, InvalidSkillPolicy, SkillSource, SkillsConfig};
use super::frontmatter::parse_frontmatter;
use super::types::{
    AutoLoadedSkillContent, LoadedSkill, SkillFrontmatter, SkillLoadError, SkillPromptContext,
    SkillScan, SkillSummary,
};
use crate::error::ArkoAgentsError;

/// Loads all skills from the provided SkillsConfig, resolving workspace paths
/// relative to `workspace_root`.
///
/// Returns a tuple of all loaded skills and pre-computed prompt data.
pub fn load_skills(
    config: &SkillsConfig,
    workspace_root: &Path,
) -> Result<(Vec<LoadedSkill>, SkillPromptContext), ArkoAgentsError> {
    let mut found = Vec::new();
    for (source, directory, label) in sources(config, workspace_root) {
        if !directory.is_dir() {
            if source.optional && !directory.exists() {
                continue;
            }
            match config.on_invalid_skill {
                InvalidSkillPolicy::Warn => {
                    log::warn!(
                        "Skills directory '{}' does not exist or is not a directory. Skipping.",
                        directory.display()
                    );
                    continue;
                }
                InvalidSkillPolicy::Error => {
                    return Err(ArkoAgentsError::Configuration(format!(
                        "Skills directory '{}' does not exist or is not a directory.",
                        directory.display()
                    )));
                }
            }
        }
        found.push((
            Origin::Folder(source),
            survey_or_unreadable(&directory, &label),
        ));
    }
    found.extend(embedded_origin(config));

    let mut skills = Vec::new();
    for settled in settle(found, config) {
        match settled.outcome {
            Outcome::Loaded(skill) => skills.push(skill),
            Outcome::Invalid(error) => handle_skill_error(error, &config.on_invalid_skill)?,
            Outcome::Unreadable(error) => match config.on_invalid_skill {
                InvalidSkillPolicy::Warn => {
                    log::warn!("{}", unreadable_warning(&error));
                }
                InvalidSkillPolicy::Error => return Err(error),
            },
            Outcome::LeftOut(..)
            | Outcome::Duplicate(..)
            | Outcome::Replaced(..)
            | Outcome::Filtered(..) => {}
        }
    }
    let prompt_context = build_prompt_context(&skills, config);

    Ok((skills, prompt_context))
}

/// Every skill file and folder in the sources of `config`, then its
/// embedded skills, with what [`load_skills`] makes of it, loaded (under its
/// name, or renamed) or skipped and why. Unlike loading, it never fails. A
/// directory that doesn't exist has nothing to list, and one that can't be
/// read is listed as a skipped entry.
pub fn scan(config: &SkillsConfig, workspace_root: &Path) -> Vec<SkillScan> {
    let mut found = Vec::new();
    for (source, directory, label) in sources(config, workspace_root) {
        if !directory.is_dir() {
            continue;
        }
        found.push((
            Origin::Folder(source),
            survey_or_unreadable(&directory, &label),
        ));
    }
    found.extend(embedded_origin(config));
    settle(found, config)
        .into_iter()
        .map(|settled| settled.scan(config))
        .collect()
}

/// The skill files and folders of the source directory `directory`, or,
/// when it can't be read, the directory itself as one that can't be read.
fn survey_or_unreadable(directory: &Path, source_label: &str) -> Vec<Found> {
    survey_source(directory, source_label).unwrap_or_else(|error| {
        vec![Found {
            path: directory.to_path_buf(),
            fallback_name: directory
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default(),
            skill: Err(Problem::Unreadable(error)),
            embedded: false,
        }]
    })
}

/// The `LoadedSkill::source_label` of an embedded skill.
const EMBEDDED_LABEL: &str = "embedded";

/// The embedded skills of `config` as one more source, after the folders,
/// each read. `None` without any.
fn embedded_origin(config: &SkillsConfig) -> Option<(Origin<'_>, Vec<Found>)> {
    if config.embedded.is_empty() {
        return None;
    }
    let found = config
        .embedded
        .iter()
        .map(|skill| Found {
            path: PathBuf::new(),
            fallback_name: skill.name.clone(),
            skill: read_embedded_skill(skill).map_err(Problem::Invalid),
            embedded: true,
        })
        .collect();
    Some((Origin::Embedded, found))
}

/// Where the skills of one source come from, in the order they count.
enum Origin<'a> {
    /// A source directory, with its lists.
    Folder(&'a SkillSource),
    /// The embedded skills of the configuration. A folder skill with the
    /// name of one replaces it.
    Embedded,
}

/// The sources of `config` in the order they count (the workspace's, then
/// the global ones), each with its directory and label.
fn sources<'a>(
    config: &'a SkillsConfig,
    workspace_root: &Path,
) -> Vec<(&'a SkillSource, PathBuf, String)> {
    let workspace = config.workspace_source.iter().map(|source| {
        (
            source,
            workspace_root.join(&source.path),
            format!("workspace:{}", source.path),
        )
    });
    let global = config.global_sources.iter().map(|source| {
        (
            source,
            PathBuf::from(&source.path),
            format!("global:{}", source.path),
        )
    });
    workspace.chain(global).collect()
}

/// A skill file or folder in a source, as read.
struct Found {
    /// Empty for an embedded skill.
    path: PathBuf,
    /// Its name when it can't be read: its folder's, or its file's without
    /// the extension, or an embedded skill's given name.
    fallback_name: String,
    skill: Result<LoadedSkill, Problem>,
    embedded: bool,
}

/// Why a skill file or folder can't be loaded. The policy skips or
/// refuses either.
pub(super) enum Problem {
    /// An invalid skill.
    Invalid(SkillLoadError),
    /// A file or folder that can't be read.
    Unreadable(ArkoAgentsError),
}

/// A skill file or folder, settled: what loading makes of it.
struct Settled {
    path: PathBuf,
    fallback_name: String,
    embedded: bool,
    outcome: Outcome,
}

enum Outcome {
    Loaded(LoadedSkill),
    Invalid(SkillLoadError),
    Unreadable(ArkoAgentsError),
    /// Left out by its source's list: `include` or `exclude`.
    LeftOut(LoadedSkill, &'static str),
    /// The same content as the skill loaded under this name.
    Duplicate(LoadedSkill, String),
    /// An embedded skill that the folder skill loaded under its name, from
    /// this path, replaces.
    Replaced(LoadedSkill, PathBuf),
    /// Left out by the configuration's `subagent` setting or `only` list,
    /// which apply after the names are settled.
    Filtered(LoadedSkill, &'static str),
}

impl Settled {
    fn scan(self, config: &SkillsConfig) -> SkillScan {
        let scan = |skill: &LoadedSkill, skipped: Option<String>| SkillScan {
            path: self.path.clone(),
            embedded: self.embedded,
            name: skill.name.clone(),
            description: skill.description.clone(),
            auto_load: skill.auto_load,
            models: skill.models.clone(),
            workers: skill.workers,
            listed: is_listed(skill, config),
            skipped,
        };
        let unread = |why: String| SkillScan {
            path: self.path.clone(),
            embedded: self.embedded,
            name: self.fallback_name.clone(),
            description: String::new(),
            auto_load: false,
            models: Vec::new(),
            workers: true,
            listed: false,
            skipped: Some(why),
        };
        match &self.outcome {
            Outcome::Loaded(skill) => scan(skill, None),
            Outcome::Invalid(error) => unread(error.to_string()),
            Outcome::Unreadable(error) => unread(error.to_string()),
            Outcome::LeftOut(skill, "include") => {
                scan(skill, Some("not in its source's include list".to_string()))
            }
            Outcome::LeftOut(skill, _) => {
                scan(skill, Some("in its source's exclude list".to_string()))
            }
            Outcome::Duplicate(skill, of) => scan(
                skill,
                Some(format!("the same as the skill '{of}', loaded already")),
            ),
            Outcome::Replaced(skill, by) => scan(
                skill,
                Some(format!(
                    "replaced by the skill '{}' in {}",
                    skill.name,
                    by.display()
                )),
            ),
            Outcome::Filtered(skill, "subagent") => scan(
                skill,
                Some("for the main agent only (workers: false)".into()),
            ),
            Outcome::Filtered(skill, _) => scan(skill, Some("not in the agent's only list".into())),
        }
    }
}

/// The skill's content is in the system prompt, so `load_skill` doesn't load
/// it: it loads automatically and is for the configuration's model.
pub(super) fn content_in_prompt(skill: &LoadedSkill, config: &SkillsConfig) -> bool {
    skill.auto_load && is_listed(skill, config)
}

/// The skill is for the configuration's model: its `models` match the
/// model's names, or there are no names to filter by. The prompt lists only
/// such skills, and `auto_load` applies only to them.
pub(super) fn is_listed(skill: &LoadedSkill, config: &SkillsConfig) -> bool {
    config.model_names.is_empty() || super::models_match(&skill.models, &config.model_names)
}

/// The skill files and folders in the source directory `directory`, in
/// the order of their names, each read. Hidden entries are passed over,
/// as are a folder without a `SKILL.md`, `README.md` and files that are
/// neither `.md` nor `.skill`.
fn survey_source(directory: &Path, source_label: &str) -> Result<Vec<Found>, ArkoAgentsError> {
    let entries = std::fs::read_dir(directory).map_err(|e| ArkoAgentsError::FileOperation {
        path: directory.to_path_buf(),
        source: e,
    })?;
    let mut entries =
        entries
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ArkoAgentsError::FileOperation {
                path: directory.to_path_buf(),
                source: e,
            })?;
    entries.sort_by_key(|entry| entry.file_name());

    let mut found = Vec::new();
    for entry in entries {
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();
        if file_name.starts_with('.') {
            continue;
        }
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(e) => {
                found.push(Found {
                    fallback_name: file_name,
                    skill: Err(Problem::Unreadable(ArkoAgentsError::FileOperation {
                        path: path.clone(),
                        source: e,
                    })),
                    path,
                    embedded: false,
                });
                continue;
            }
        };

        let skill = if file_type.is_dir() {
            match read_folder_skill(&path, source_label) {
                Some(skill) => skill,
                None => continue,
            }
        } else if file_type.is_file() && file_name.ends_with(".skill") {
            read_bundle_skill(&path, source_label).map_err(Problem::Invalid)
        } else if file_type.is_file() && file_name.ends_with(".md") && file_name != "README.md" {
            read_flat_skill(&path, source_label).map_err(Problem::Invalid)
        } else {
            continue;
        };
        let fallback_name = if file_type.is_dir() {
            file_name
        } else {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().to_string())
                .unwrap_or(file_name)
        };
        found.push(Found {
            path,
            fallback_name,
            skill,
            embedded: false,
        });
    }
    Ok(found)
}

/// What loading makes of the skills `found` in each source: each source's
/// lists leave some out, a skill with the content of one loaded already is
/// a duplicate, and one with the name of one loaded already is renamed
/// (`name-2`). Earlier sources come first. The embedded skills come last
/// and have the lowest priority. A folder skill's name replaces an embedded
/// skill with that name, which is left out instead of renamed. The
/// configuration's `subagent` setting and `only` list come last, so the
/// names are those the agent that isn't a worker has.
fn settle(found: Vec<(Origin<'_>, Vec<Found>)>, config: &SkillsConfig) -> Vec<Settled> {
    let mut settled = Vec::new();
    let mut name_registry: HashSet<String> = HashSet::new();
    // The path of each folder skill loaded, by the name it loaded under.
    let mut folder_skills: HashMap<String, PathBuf> = HashMap::new();
    let mut content_hashes: HashMap<u64, String> = HashMap::new();

    for (origin, found) in found {
        for Found {
            path,
            fallback_name,
            skill,
            embedded,
        } in found
        {
            let location = path.clone();
            let settle = |outcome| Settled {
                path,
                fallback_name,
                embedded,
                outcome,
            };
            let mut skill = match skill {
                Ok(skill) => skill,
                Err(Problem::Invalid(error)) => {
                    settled.push(settle(Outcome::Invalid(error)));
                    continue;
                }
                Err(Problem::Unreadable(error)) => {
                    settled.push(settle(Outcome::Unreadable(error)));
                    continue;
                }
            };
            if let Origin::Folder(source) = &origin
                && let Some(list) = left_out(&skill, source)
            {
                settled.push(settle(Outcome::LeftOut(skill, list)));
                continue;
            }
            if embedded && let Some(folder) = folder_skills.get(&skill.name) {
                log::debug!(
                    "Embedded skill '{}' is replaced by the skill of that name in {}.",
                    skill.name,
                    folder.display()
                );
                let folder = folder.clone();
                settled.push(settle(Outcome::Replaced(skill, folder)));
                continue;
            }

            let content_hash = compute_skill_content_hash(&skill);
            if let Some(existing_name) = content_hashes.get(&content_hash) {
                log::debug!(
                    "Skill '{}' from {} is a content duplicate of '{}'. Skipping.",
                    skill.original_name,
                    skill.source_label,
                    existing_name
                );
                let existing_name = existing_name.clone();
                settled.push(settle(Outcome::Duplicate(skill, existing_name)));
                continue;
            }

            if name_registry.contains(&skill.name) {
                let mut suffix = 2u32;
                let mut candidate = format!("{}-{}", skill.original_name, suffix);
                while name_registry.contains(&candidate) {
                    suffix += 1;
                    candidate = format!("{}-{}", skill.original_name, suffix);
                }
                log::warn!(
                    "Skill '{}' from {} has a name collision with an existing skill. Renamed to '{}'.",
                    skill.original_name,
                    skill.source_label,
                    candidate
                );
                skill.name = candidate;
            }

            name_registry.insert(skill.name.clone());
            if !embedded {
                folder_skills.insert(skill.name.clone(), location);
            }
            content_hashes.insert(content_hash, skill.name.clone());
            if config.subagent && !skill.workers {
                settled.push(settle(Outcome::Filtered(skill, "subagent")));
            } else if config
                .only
                .as_ref()
                .is_some_and(|only| !only.contains(&skill.name))
            {
                settled.push(settle(Outcome::Filtered(skill, "only")));
            } else {
                settled.push(settle(Outcome::Loaded(skill)));
            }
        }
    }

    settled
}

/// The list of `source` that leaves `skill` out, if one does: `include`
/// (applied first), or `exclude`.
fn left_out(skill: &LoadedSkill, source: &SkillSource) -> Option<&'static str> {
    let name = skill.original_name.as_str();
    if let Some(ref include_list) = source.include
        && !include_list.iter().any(|included| included == name)
    {
        return Some("include");
    }
    if let Some(ref exclude_list) = source.exclude
        && exclude_list.iter().any(|excluded| excluded == name)
    {
        return Some("exclude");
    }
    None
}

/// The folder skill in `dir_path`: `None` when the folder has no
/// `SKILL.md`.
pub(super) fn read_folder_skill(
    dir_path: &Path,
    source_label: &str,
) -> Option<Result<LoadedSkill, Problem>> {
    let skill_md_path = dir_path.join("SKILL.md");
    if !skill_md_path.exists() || !skill_md_path.is_file() {
        return None;
    }

    let content = match std::fs::read_to_string(&skill_md_path) {
        Ok(c) => c,
        Err(e) => {
            return Some(Err(Problem::Invalid(SkillLoadError::Io {
                path: skill_md_path.to_string_lossy().to_string(),
                source: e,
            })));
        }
    };

    let (frontmatter, body) = match parse_frontmatter(&content, &skill_md_path.to_string_lossy()) {
        Ok(result) => result,
        Err(e) => return Some(Err(Problem::Invalid(e))),
    };

    let dir_name = dir_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    if frontmatter.name != dir_name {
        log::warn!(
            "Skill '{}' has frontmatter name '{}', which does not match directory name '{}'. Using the frontmatter name.",
            dir_name,
            frontmatter.name,
            dir_name
        );
    }

    let mut resources = HashMap::new();
    if let Err(error) = collect_resources(dir_path, dir_path, &mut resources) {
        return Some(Err(Problem::Unreadable(error)));
    }

    Some(Ok(loaded(frontmatter, body, resources, source_label)))
}

fn collect_resources(
    base_dir: &Path,
    current_dir: &Path,
    resources: &mut HashMap<String, String>,
) -> Result<(), ArkoAgentsError> {
    let entries = std::fs::read_dir(current_dir).map_err(|e| ArkoAgentsError::FileOperation {
        path: current_dir.to_path_buf(),
        source: e,
    })?;

    for entry_result in entries {
        let entry = entry_result.map_err(|e| ArkoAgentsError::FileOperation {
            path: current_dir.to_path_buf(),
            source: e,
        })?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|e| ArkoAgentsError::FileOperation {
                path: path.clone(),
                source: e,
            })?;

        let entry_name = entry.file_name().to_string_lossy().to_string();

        if entry_name.starts_with('.') {
            continue;
        }

        if file_type.is_dir() {
            if entry_name == "scripts" {
                continue;
            }
            collect_resources(base_dir, &path, resources)?;
        } else if file_type.is_file() {
            if entry_name == "SKILL.md" {
                continue;
            }
            if !entry_name.ends_with(".md") {
                continue;
            }
            let relative = path
                .strip_prefix(base_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let file_content =
                std::fs::read_to_string(&path).map_err(|e| ArkoAgentsError::FileOperation {
                    path: path.clone(),
                    source: e,
                })?;
            resources.insert(relative, file_content);
        }
    }

    Ok(())
}

/// The bundle skill (a zip archive) in the file `file_path`.
fn read_bundle_skill(file_path: &Path, source_label: &str) -> Result<LoadedSkill, SkillLoadError> {
    let path = file_path.to_string_lossy();
    let data = std::fs::read(file_path).map_err(|e| SkillLoadError::Io {
        path: path.to_string(),
        source: e,
    })?;
    read_bundle_bytes(&data, &path, source_label, Bundle::Loading)
}

/// How far the entries of a bundle are checked.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Bundle {
    /// A bundle in a source directory. Its `.md` files that aren't UTF-8 text
    /// are passed over, and its entries are only read into memory.
    Loading,
    /// A bundle offered to be saved as a skill, whose entries would be
    /// written to a folder. An entry with an absolute path, a `..` or a link,
    /// a `.md` file that isn't UTF-8 text, or more than
    /// [`MAX_CANDIDATE_BYTES`] in all is refused.
    Candidate,
}

/// The most that the entries of a bundle offered as a skill may hold when
/// unpacked, so that a small file can't fill the memory.
const MAX_CANDIDATE_BYTES: u64 = 16 * 1024 * 1024;

/// The bundle skill in the zip archive `data`, named `path` in errors.
pub(super) fn read_bundle_bytes(
    data: &[u8],
    path: &str,
    source_label: &str,
    mode: Bundle,
) -> Result<LoadedSkill, SkillLoadError> {
    let zip_error = |message: String| SkillLoadError::ZipError {
        path: path.to_string(),
        message,
    };
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(data)).map_err(|e| zip_error(e.to_string()))?;

    let too_much = || {
        zip_error(format!(
            "The bundle holds more than {} MB unpacked.",
            MAX_CANDIDATE_BYTES / (1024 * 1024)
        ))
    };
    let mut entries: Vec<(String, String)> = Vec::new();
    // What the entries say they hold, and the text read from them.
    let mut unpacked = 0u64;
    let mut text_bytes = 0u64;

    for i in 0..archive.len() {
        let mut zip_file = archive
            .by_index(i)
            .map_err(|e| zip_error(format!("Failed to read zip entry {}: {}", i, e)))?;
        let name = zip_file
            .name()
            .map_err(|e| zip_error(format!("Failed to read the name of zip entry {}: {}", i, e)))?
            .replace('\\', "/");

        if mode == Bundle::Candidate {
            check_candidate_entry(&zip_file, &name).map_err(&zip_error)?;
            unpacked = unpacked.saturating_add(zip_file.size());
            if unpacked > MAX_CANDIDATE_BYTES {
                return Err(too_much());
            }
        }
        // Only the skill's text is read.
        if zip_file.is_dir() || !name.ends_with(".md") {
            continue;
        }

        let mut content = String::new();
        let read = match mode {
            Bundle::Loading => zip_file.read_to_string(&mut content),
            // However much an entry says it holds, no more than what is left
            // of the limit is read.
            Bundle::Candidate => (&mut zip_file)
                .take(MAX_CANDIDATE_BYTES - text_bytes + 1)
                .read_to_string(&mut content),
        };
        text_bytes += content.len() as u64;
        if text_bytes > MAX_CANDIDATE_BYTES && mode == Bundle::Candidate {
            return Err(too_much());
        }
        if let Err(e) = read {
            if mode == Bundle::Candidate {
                return Err(zip_error(format!("'{name}' is not UTF-8 text ({e}).")));
            }
            log::debug!(
                "Skipping non-UTF-8 zip entry '{}' in {:?}: {}",
                name,
                path,
                e
            );
            continue;
        }

        entries.push((name, content));
    }

    let root_prefix = determine_zip_root(&entries);

    let mut skill_md_content: Option<String> = None;
    let mut resources = HashMap::new();

    for (raw_name, content) in &entries {
        let relative = if raw_name.starts_with(&root_prefix) {
            raw_name[root_prefix.len()..].to_string()
        } else {
            raw_name.clone()
        };

        if relative.is_empty() || relative.ends_with('/') {
            continue;
        }

        if relative == "SKILL.md" {
            skill_md_content = Some(content.clone());
            continue;
        }

        let parts: Vec<&str> = relative.split('/').collect();
        if parts.contains(&"scripts") {
            continue;
        }

        resources.insert(relative, content.clone());
    }

    let skill_md = skill_md_content.ok_or_else(|| SkillLoadError::MissingSkillMd {
        path: path.to_string(),
    })?;

    let (frontmatter, body) = parse_frontmatter(&skill_md, path)?;

    Ok(loaded(frontmatter, body, resources, source_label))
}

/// Why an entry of a bundle offered as a skill can't be written to a skill's
/// folder, if it can't: its name leaves the folder, or it isn't a file or a
/// folder.
fn check_candidate_entry<R: Read + ?Sized>(
    entry: &zip::read::ZipFile<'_, R>,
    name: &str,
) -> Result<(), String> {
    let drive = name.as_bytes().get(1) == Some(&b':')
        && name.as_bytes().first().is_some_and(u8::is_ascii_alphabetic);
    if name.starts_with('/') || drive || name.contains('\0') {
        return Err(format!("'{name}' has an absolute path."));
    }
    if name.split('/').any(|part| part == "..") || entry.enclosed_name().is_none() {
        return Err(format!("'{name}' leaves the skill's folder."));
    }
    // The file type is in the mode's top bits: 0o10 a file, 0o04 a folder.
    if let Some(mode) = entry.unix_mode() {
        let kind = mode & 0o170000;
        if kind == 0o120000 {
            return Err(format!("'{name}' is a link."));
        }
        if kind != 0 && kind != 0o100000 && kind != 0o040000 {
            return Err(format!("'{name}' is neither a file nor a folder."));
        }
    }
    Ok(())
}

fn determine_zip_root(entries: &[(String, String)]) -> String {
    for (name, _) in entries {
        if name == "SKILL.md" {
            return String::new();
        }
        if name.ends_with("/SKILL.md") {
            return name[..name.len() - "SKILL.md".len()].to_string();
        }
    }
    String::new()
}

/// The flat skill (a markdown file) in the file `file_path`.
pub(super) fn read_flat_skill(
    file_path: &Path,
    source_label: &str,
) -> Result<LoadedSkill, SkillLoadError> {
    let content = std::fs::read_to_string(file_path).map_err(|e| SkillLoadError::Io {
        path: file_path.to_string_lossy().to_string(),
        source: e,
    })?;

    let (frontmatter, body) = parse_frontmatter(&content, &file_path.to_string_lossy())?;

    Ok(loaded(frontmatter, body, HashMap::new(), source_label))
}

/// The skill given in memory, read as a flat skill is, with no resources.
fn read_embedded_skill(embedded: &EmbeddedSkill) -> Result<LoadedSkill, SkillLoadError> {
    let (frontmatter, body) = parse_frontmatter(
        &embedded.content,
        &format!("embedded skill '{}'", embedded.name),
    )?;
    if frontmatter.name != embedded.name {
        log::warn!(
            "Embedded skill '{}' has frontmatter name '{}', which does not match its name. Using the frontmatter name.",
            embedded.name,
            frontmatter.name
        );
    }

    Ok(loaded(frontmatter, body, HashMap::new(), EMBEDDED_LABEL))
}

/// The skill that the parsed front matter, the body and the resources make.
fn loaded(
    frontmatter: SkillFrontmatter,
    content: String,
    resources: HashMap<String, String>,
    source_label: &str,
) -> LoadedSkill {
    LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        models: frontmatter.models,
        workers: frontmatter.workers,
        content,
        resources,
        source_label: source_label.to_string(),
    }
}

fn compute_skill_content_hash(skill: &LoadedSkill) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    skill.content.hash(&mut hasher);
    let mut sorted_keys: Vec<&String> = skill.resources.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        key.hash(&mut hasher);
        skill.resources[key].hash(&mut hasher);
    }
    hasher.finish()
}

/// What the prompt holds of the skills: the content of those that load
/// automatically and the line of each other, for the skills that are for
/// the configuration's model. The others are not in the prompt.
fn build_prompt_context(skills: &[LoadedSkill], config: &SkillsConfig) -> SkillPromptContext {
    let mut on_demand = Vec::new();
    let mut auto_loaded = Vec::new();

    for skill in skills.iter().filter(|skill| is_listed(skill, config)) {
        if skill.auto_load {
            auto_loaded.push(AutoLoadedSkillContent {
                name: skill.name.clone(),
                content: skill.content.clone(),
            });
        } else {
            on_demand.push(SkillSummary {
                name: skill.name.clone(),
                description: skill.description.clone(),
            });
        }
    }

    SkillPromptContext {
        on_demand_skills: on_demand,
        auto_loaded_skills: auto_loaded,
    }
}

fn unreadable_warning(error: &ArkoAgentsError) -> String {
    format!("Skipping a skill that can't be read ({error}).")
}

fn handle_skill_error(
    error: SkillLoadError,
    policy: &InvalidSkillPolicy,
) -> Result<(), ArkoAgentsError> {
    match policy {
        InvalidSkillPolicy::Warn => {
            log::warn!("Skipping invalid skill: {}", error);
            Ok(())
        }
        InvalidSkillPolicy::Error => Err(ArkoAgentsError::Skill(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreadable_skill_warnings_keep_the_error_in_parentheses() {
        let error = ArkoAgentsError::Configuration("unreadable skill file".into());
        assert_eq!(
            unreadable_warning(&error),
            format!("Skipping a skill that can't be read ({error}).")
        );
    }

    /// A temporary directory, removed when dropped.
    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("arko-skills-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn skill(name: &str, body: &str) -> String {
        format!("---\nname: {name}\ndescription: The {name} skill.\n---\n{body}\n")
    }

    fn global(path: &Path) -> SkillSource {
        SkillSource {
            path: path.display().to_string(),
            include: None,
            exclude: None,
            optional: false,
        }
    }

    /// The scan lists each skill file and folder with what loading makes
    /// of it, and what it lists as loaded is what loading loads.
    #[test]
    fn a_scan_lists_what_loading_makes_of_each_skill() {
        let temp = Temp::new();
        let workspace = temp.0.join("project");
        let first = workspace.join(".skills");
        let second = temp.0.join("global");
        write(&first.join("review/SKILL.md"), &skill("review", "Review."));
        write(&first.join("review/FORMS.md"), "Forms.");
        write(&first.join("flat.md"), &skill("flat", "Flat."));
        write(&first.join("broken/SKILL.md"), "No front matter.\n");
        write(&first.join("bundle.skill"), "not a zip");
        write(&first.join("README.md"), "About these skills.");
        write(&first.join(".hidden/SKILL.md"), &skill("hidden", "Hidden."));
        write(&first.join("notes/other.md"), "Not a skill folder.");
        write(&first.join("left/SKILL.md"), &skill("left", "Left."));
        // The same name, another skill: renamed. The same content: skipped.
        write(
            &second.join("review/SKILL.md"),
            &skill("review", "Another review."),
        );
        write(&second.join("copy/SKILL.md"), &skill("flat", "Flat."));
        let config = SkillsConfig {
            workspace_source: Some(SkillSource {
                path: ".skills".into(),
                include: None,
                exclude: Some(vec!["left".into()]),
                optional: false,
            }),
            global_sources: vec![global(&second), global(&temp.0.join("missing"))],
            embedded: Vec::new(),
            on_invalid_skill: InvalidSkillPolicy::Warn,
            load_skill_interception: None,
            ..Default::default()
        };

        let scanned = scan(&config, &workspace);
        let listed: Vec<(&Path, &str, Option<&str>)> = scanned
            .iter()
            .map(|skill| {
                (
                    skill.path.as_path(),
                    skill.name.as_str(),
                    skill.skipped.as_deref(),
                )
            })
            .collect();
        assert_eq!(
            listed
                .iter()
                .map(|(path, name, _)| (path.to_path_buf(), *name))
                .collect::<Vec<_>>(),
            [
                (first.join("broken"), "broken"),
                (first.join("bundle.skill"), "bundle"),
                (first.join("flat.md"), "flat"),
                (first.join("left"), "left"),
                (first.join("review"), "review"),
                (second.join("copy"), "flat"),
                (second.join("review"), "review-2"),
            ]
        );
        let skipped: Vec<Option<&str>> = listed.iter().map(|(_, _, skipped)| *skipped).collect();
        assert!(skipped[0].unwrap().contains("frontmatter"), "{skipped:?}");
        assert!(skipped[1].unwrap().contains("Zip"), "{skipped:?}");
        assert_eq!(skipped[2], None);
        assert_eq!(skipped[3], Some("in its source's exclude list"));
        assert_eq!(skipped[4], None);
        assert_eq!(
            skipped[5],
            Some("the same as the skill 'flat', loaded already")
        );
        assert_eq!(skipped[6], None);
        assert_eq!(scanned[2].description, "The flat skill.");

        // Loading loads what the scan lists as loaded.
        let (loaded, _) = load_skills(&config, &workspace).unwrap();
        let names: Vec<&str> = loaded.iter().map(|skill| skill.name.as_str()).collect();
        let scanned_names: Vec<&str> = scanned
            .iter()
            .filter(|skill| skill.skipped.is_none())
            .map(|skill| skill.name.as_str())
            .collect();
        assert_eq!(names, scanned_names);
        assert_eq!(names, ["flat", "review", "review-2"]);

        // Only what a source includes.
        let mut only = config.clone();
        only.workspace_source.as_mut().unwrap().include = Some(vec!["flat".into()]);
        let scanned = scan(&only, &workspace);
        let review = scanned
            .iter()
            .find(|skill| skill.path == first.join("review"))
            .unwrap();
        assert_eq!(
            review.skipped.as_deref(),
            Some("not in its source's include list")
        );

        // Where loading refuses an invalid skill, the scan still lists all.
        let mut strict = config.clone();
        strict.global_sources.pop();
        strict.on_invalid_skill = InvalidSkillPolicy::Error;
        assert!(load_skills(&strict, &workspace).is_err());
        assert_eq!(scan(&strict, &workspace).len(), 7);
    }

    /// A skill file or a source directory that can't be read is skipped,
    /// as an invalid skill is, and the scan lists it so. Loading under
    /// `Error` still refuses it.
    #[cfg(unix)]
    #[test]
    fn what_cant_be_read_is_skipped_and_listed() {
        use std::os::unix::fs::PermissionsExt;
        let temp = Temp::new();
        let first = temp.0.join("first");
        let closed = temp.0.join("closed");
        write(&first.join("ok/SKILL.md"), &skill("ok", "Fine."));
        write(&first.join("bad/SKILL.md"), &skill("bad", "Bad."));
        std::fs::write(first.join("bad/ref.md"), [0xff, 0xfe, 0x00]).unwrap();
        write(&closed.join("hidden/SKILL.md"), &skill("hidden", "Hidden."));
        std::fs::set_permissions(&closed, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::read_dir(&closed).is_ok() {
            // (Run as root, which reads it anyway.)
            std::fs::set_permissions(&closed, std::fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }
        let mut config = SkillsConfig {
            workspace_source: None,
            global_sources: vec![global(&first), global(&closed)],
            embedded: Vec::new(),
            on_invalid_skill: InvalidSkillPolicy::Warn,
            load_skill_interception: None,
            ..Default::default()
        };
        let (loaded, _) = load_skills(&config, &temp.0).unwrap();
        let names: Vec<&str> = loaded.iter().map(|skill| skill.name.as_str()).collect();
        assert_eq!(names, ["ok"]);
        let scanned = scan(&config, &temp.0);
        let listed: Vec<(&Path, &str, bool)> = scanned
            .iter()
            .map(|skill| {
                (
                    skill.path.as_path(),
                    skill.name.as_str(),
                    skill.skipped.is_some(),
                )
            })
            .collect();
        assert_eq!(
            listed,
            [
                (first.join("bad").as_path(), "bad", true),
                (first.join("ok").as_path(), "ok", false),
                (closed.as_path(), "closed", true),
            ]
        );
        config.on_invalid_skill = InvalidSkillPolicy::Error;
        assert!(load_skills(&config, &temp.0).is_err());
        std::fs::set_permissions(&closed, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}
