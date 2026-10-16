use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;

use super::config::{InvalidSkillPolicy, SkillSource, SkillsConfig};
use super::frontmatter::parse_frontmatter;
use super::types::{
    AutoLoadedSkillContent, LoadedSkill, SkillLoadError, SkillPromptContext, SkillSummary,
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
    let mut all_source_skills: Vec<Vec<LoadedSkill>> = Vec::new();

    if let Some(ref ws_source) = config.workspace_source {
        let resolved = workspace_root.join(&ws_source.path);
        let label = format!("workspace:{}", ws_source.path);
        let skills =
            load_skills_from_source(ws_source, &resolved, &label, &config.on_invalid_skill)?;
        all_source_skills.push(skills);
    }

    for global_source in &config.global_sources {
        let resolved = std::path::PathBuf::from(&global_source.path);
        let label = format!("global:{}", global_source.path);
        let skills =
            load_skills_from_source(global_source, &resolved, &label, &config.on_invalid_skill)?;
        all_source_skills.push(skills);
    }

    let deduplicated = deduplicate_skills(all_source_skills);
    let prompt_context = build_prompt_context(&deduplicated);

    Ok((deduplicated, prompt_context))
}

fn load_skills_from_source(
    source: &SkillSource,
    resolved_path: &Path,
    source_label: &str,
    policy: &InvalidSkillPolicy,
) -> Result<Vec<LoadedSkill>, ArkoAgentsError> {
    if !resolved_path.exists() || !resolved_path.is_dir() {
        match policy {
            InvalidSkillPolicy::Warn => {
                log::warn!(
                    "Skills directory '{}' does not exist or is not a directory. Skipping.",
                    resolved_path.display()
                );
                return Ok(Vec::new());
            }
            InvalidSkillPolicy::Error => {
                return Err(ArkoAgentsError::Configuration(format!(
                    "Skills directory '{}' does not exist or is not a directory.",
                    resolved_path.display()
                )));
            }
        }
    }

    let mut skills = Vec::new();

    let entries = std::fs::read_dir(resolved_path).map_err(|e| ArkoAgentsError::FileOperation {
        path: resolved_path.to_path_buf(),
        source: e,
    })?;

    for entry_result in entries {
        let entry = entry_result.map_err(|e| ArkoAgentsError::FileOperation {
            path: resolved_path.to_path_buf(),
            source: e,
        })?;
        let file_type = entry
            .file_type()
            .map_err(|e| ArkoAgentsError::FileOperation {
                path: entry.path(),
                source: e,
            })?;
        let file_name = entry.file_name().to_string_lossy().to_string();

        if file_name.starts_with('.') {
            continue;
        }

        if file_type.is_dir() {
            attempt_load_folder_skill(&entry.path(), source_label, policy, &mut skills)?;
        } else if file_type.is_file() {
            if file_name.ends_with(".skill") {
                attempt_load_bundle_skill(&entry.path(), source_label, policy, &mut skills)?;
            } else if file_name.ends_with(".md") && file_name != "README.md" {
                attempt_load_flat_skill(&entry.path(), source_label, policy, &mut skills)?;
            }
        }
    }

    let filtered = apply_filters(skills, source);
    Ok(filtered)
}

fn attempt_load_folder_skill(
    dir_path: &Path,
    source_label: &str,
    policy: &InvalidSkillPolicy,
    skills: &mut Vec<LoadedSkill>,
) -> Result<(), ArkoAgentsError> {
    let skill_md_path = dir_path.join("SKILL.md");
    if !skill_md_path.exists() || !skill_md_path.is_file() {
        return Ok(());
    }

    let content = match std::fs::read_to_string(&skill_md_path) {
        Ok(c) => c,
        Err(e) => {
            let err = SkillLoadError::Io {
                path: skill_md_path.to_string_lossy().to_string(),
                source: e,
            };
            return handle_skill_error(err, policy);
        }
    };

    let (frontmatter, body) = match parse_frontmatter(&content, &skill_md_path.to_string_lossy()) {
        Ok(result) => result,
        Err(e) => return handle_skill_error(e, policy),
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
    collect_resources(dir_path, dir_path, &mut resources)?;

    skills.push(LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        content: body,
        resources,
        source_label: source_label.to_string(),
    });

    Ok(())
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

fn attempt_load_bundle_skill(
    file_path: &Path,
    source_label: &str,
    policy: &InvalidSkillPolicy,
    skills: &mut Vec<LoadedSkill>,
) -> Result<(), ArkoAgentsError> {
    let data = match std::fs::read(file_path) {
        Ok(d) => d,
        Err(e) => {
            let err = SkillLoadError::Io {
                path: file_path.to_string_lossy().to_string(),
                source: e,
            };
            return handle_skill_error(err, policy);
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
            return handle_skill_error(err, policy);
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
                return handle_skill_error(err, policy);
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
                return handle_skill_error(err, policy);
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
            return handle_skill_error(err, policy);
        }
    };

    let (frontmatter, body) = match parse_frontmatter(&skill_md, &file_path.to_string_lossy()) {
        Ok(result) => result,
        Err(e) => return handle_skill_error(e, policy),
    };

    skills.push(LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        content: body,
        resources,
        source_label: source_label.to_string(),
    });

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

fn attempt_load_flat_skill(
    file_path: &Path,
    source_label: &str,
    policy: &InvalidSkillPolicy,
    skills: &mut Vec<LoadedSkill>,
) -> Result<(), ArkoAgentsError> {
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            let err = SkillLoadError::Io {
                path: file_path.to_string_lossy().to_string(),
                source: e,
            };
            return handle_skill_error(err, policy);
        }
    };

    let (frontmatter, body) = match parse_frontmatter(&content, &file_path.to_string_lossy()) {
        Ok(result) => result,
        Err(e) => return handle_skill_error(e, policy),
    };

    skills.push(LoadedSkill {
        name: frontmatter.name.clone(),
        original_name: frontmatter.name,
        description: frontmatter.description,
        auto_load: frontmatter.auto_load,
        content: body,
        resources: HashMap::new(),
        source_label: source_label.to_string(),
    });

    Ok(())
}

fn apply_filters(skills: Vec<LoadedSkill>, source: &SkillSource) -> Vec<LoadedSkill> {
    let mut filtered = skills;

    if let Some(ref include_list) = source.include {
        let include_set: HashSet<&str> = include_list.iter().map(|s| s.as_str()).collect();
        filtered.retain(|s| include_set.contains(s.original_name.as_str()));
    }

    if let Some(ref exclude_list) = source.exclude {
        let exclude_set: HashSet<&str> = exclude_list.iter().map(|s| s.as_str()).collect();
        filtered.retain(|s| !exclude_set.contains(s.original_name.as_str()));
    }

    filtered
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

fn deduplicate_skills(source_lists: Vec<Vec<LoadedSkill>>) -> Vec<LoadedSkill> {
    let mut final_skills: Vec<LoadedSkill> = Vec::new();
    let mut name_registry: HashSet<String> = HashSet::new();
    let mut content_hashes: HashMap<u64, String> = HashMap::new();

    for source_skills in source_lists {
        for mut skill in source_skills {
            let content_hash = compute_skill_content_hash(&skill);

            if let Some(existing_name) = content_hashes.get(&content_hash) {
                log::debug!(
                    "Skill '{}' from {} is a content duplicate of '{}'. Skipping.",
                    skill.original_name,
                    skill.source_label,
                    existing_name
                );
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
            final_skills.push(skill);
        }
    }

    final_skills
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
