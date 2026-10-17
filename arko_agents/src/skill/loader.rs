use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::config::{InvalidSkillPolicy, SkillSource, SkillsConfig};
use super::frontmatter::parse_frontmatter;
use super::types::{
    AutoLoadedSkillContent, LoadedSkill, SkillLoadError, SkillPromptContext, SkillScan,
    SkillSummary,
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
        found.push((source, survey_source(&directory, &label)?));
    }

    let mut skills = Vec::new();
    for settled in settle(found) {
        match settled.outcome {
            Outcome::Loaded(skill) => skills.push(skill),
            Outcome::Invalid(error) => handle_skill_error(error, &config.on_invalid_skill)?,
            Outcome::Unreadable(error) => return Err(error),
            Outcome::LeftOut(..) | Outcome::Duplicate(..) => {}
        }
    }
    let prompt_context = build_prompt_context(&skills);

    Ok((skills, prompt_context))
}

/// Every skill file and folder in the sources of `config`, with what
/// [`load_skills`] makes of it: loaded (under its name, or renamed), or
/// skipped and why. Unlike loading, it doesn't fail: a directory that
/// doesn't exist has nothing to list, and one that can't be read is
/// logged and left out.
pub fn scan(config: &SkillsConfig, workspace_root: &Path) -> Vec<SkillScan> {
    let mut found = Vec::new();
    for (source, directory, label) in sources(config, workspace_root) {
        if !directory.is_dir() {
            continue;
        }
        match survey_source(&directory, &label) {
            Ok(list) => found.push((source, list)),
            Err(error) => log::warn!(
                "Skills directory '{}' could not be read: {error}",
                directory.display()
            ),
        }
    }
    settle(found).into_iter().map(Settled::scan).collect()
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
    path: PathBuf,
    /// Its name when it can't be read: its folder's, or its file's without
    /// the extension.
    fallback_name: String,
    skill: Result<LoadedSkill, Problem>,
}

/// Why a skill file or folder can't be loaded.
enum Problem {
    /// An invalid skill, which the policy skips or refuses.
    Invalid(SkillLoadError),
    /// A file that can't be read, which loading always fails on.
    Unreadable(ArkoAgentsError),
}

/// A skill file or folder, settled: what loading makes of it.
struct Settled {
    path: PathBuf,
    fallback_name: String,
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
}

impl Settled {
    fn scan(self) -> SkillScan {
        let scan = |skill: &LoadedSkill, skipped: Option<String>| SkillScan {
            path: self.path.clone(),
            name: skill.name.clone(),
            description: skill.description.clone(),
            auto_load: skill.auto_load,
            skipped,
        };
        let unread = |why: String| SkillScan {
            path: self.path.clone(),
            name: self.fallback_name.clone(),
            description: String::new(),
            auto_load: false,
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
        }
    }
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
        });
    }
    Ok(found)
}

/// What loading makes of the skills `found` in each source: each source's
/// lists leave some out, a skill with the content of one loaded already is
/// a duplicate, and one with the name of one loaded already is renamed
/// (`name-2`). Earlier sources come first.
fn settle(found: Vec<(&SkillSource, Vec<Found>)>) -> Vec<Settled> {
    let mut settled = Vec::new();
    let mut name_registry: HashSet<String> = HashSet::new();
    let mut content_hashes: HashMap<u64, String> = HashMap::new();

    for (source, found) in found {
        for Found {
            path,
            fallback_name,
            skill,
        } in found
        {
            let settle = |outcome| Settled {
                path,
                fallback_name,
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
            if let Some(list) = left_out(&skill, source) {
                settled.push(settle(Outcome::LeftOut(skill, list)));
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
            content_hashes.insert(content_hash, skill.name.clone());
            settled.push(settle(Outcome::Loaded(skill)));
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
fn read_folder_skill(dir_path: &Path, source_label: &str) -> Option<Result<LoadedSkill, Problem>> {
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
            "Skill '{}': frontmatter name '{}' does not match directory name '{}'. Using frontmatter name.",
            dir_name,
            frontmatter.name,
            dir_name
        );
    }

    let mut resources = HashMap::new();
    if let Err(error) = collect_resources(dir_path, dir_path, &mut resources) {
        return Some(Err(Problem::Unreadable(error)));
    }

    Some(Ok(LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        content: body,
        resources,
        source_label: source_label.to_string(),
    }))
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
    let data = match std::fs::read(file_path) {
        Ok(d) => d,
        Err(e) => {
            let err = SkillLoadError::Io {
                path: file_path.to_string_lossy().to_string(),
                source: e,
            };
            return Err(err);
        }
    };

    let cursor = std::io::Cursor::new(data);
    let mut archive = match zip::ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            let err = SkillLoadError::ZipError {
                path: file_path.to_string_lossy().to_string(),
                message: e.to_string(),
            };
            return Err(err);
        }
    };

    let mut entries: Vec<(String, String)> = Vec::new();

    for i in 0..archive.len() {
        let mut zip_file = match archive.by_index(i) {
            Ok(f) => f,
            Err(e) => {
                let err = SkillLoadError::ZipError {
                    path: file_path.to_string_lossy().to_string(),
                    message: format!("Failed to read zip entry {}: {}", i, e),
                };
                return Err(err);
            }
        };

        if zip_file.is_dir() {
            continue;
        }

        let name = match zip_file.name() {
            Ok(name) => name.replace('\\', "/"),
            Err(e) => {
                let err = SkillLoadError::ZipError {
                    path: file_path.to_string_lossy().to_string(),
                    message: format!("Failed to read the name of zip entry {}: {}", i, e),
                };
                return Err(err);
            }
        };
        let mut content = String::new();
        if let Err(e) = zip_file.read_to_string(&mut content) {
            log::debug!(
                "Skipping non-UTF-8 zip entry '{}' in {:?}: {}",
                name,
                file_path,
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

        if !relative.ends_with(".md") {
            continue;
        }

        let parts: Vec<&str> = relative.split('/').collect();
        if parts.contains(&"scripts") {
            continue;
        }

        resources.insert(relative, content.clone());
    }

    let skill_md = match skill_md_content {
        Some(c) => c,
        None => {
            let err = SkillLoadError::MissingSkillMd {
                path: file_path.to_string_lossy().to_string(),
            };
            return Err(err);
        }
    };

    let (frontmatter, body) = parse_frontmatter(&skill_md, &file_path.to_string_lossy())?;

    Ok(LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        content: body,
        resources,
        source_label: source_label.to_string(),
    })
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
fn read_flat_skill(file_path: &Path, source_label: &str) -> Result<LoadedSkill, SkillLoadError> {
    let content = std::fs::read_to_string(file_path).map_err(|e| SkillLoadError::Io {
        path: file_path.to_string_lossy().to_string(),
        source: e,
    })?;

    let (frontmatter, body) = parse_frontmatter(&content, &file_path.to_string_lossy())?;

    Ok(LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        content: body,
        resources: HashMap::new(),
        source_label: source_label.to_string(),
    })
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

fn build_prompt_context(skills: &[LoadedSkill]) -> SkillPromptContext {
    let mut on_demand = Vec::new();
    let mut auto_loaded = Vec::new();

    for skill in skills {
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
            }),
            global_sources: vec![global(&second), global(&temp.0.join("missing"))],
            on_invalid_skill: InvalidSkillPolicy::Warn,
            load_skill_interception: None,
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
}
