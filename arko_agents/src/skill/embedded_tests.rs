//! Tests of skills given in memory.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{Value, json};

use super::*;
use crate::agent_config::{AgentConfig, AgentLLMConfig};
use crate::error::ArkoAgentsError;
use crate::parsing::PromptBuilder;
use crate::tool::{TaskContext, Tool, ToolRegistry};

/// The text of a `SKILL.md` for the skill `name`.
pub(super) fn text(name: &str, extra: &str, body: &str) -> String {
    format!("---\nname: {name}\ndescription: The {name} skill.\n{extra}---\n{body}\n")
}

pub(super) fn embedded(name: &str, extra: &str, body: &str) -> EmbeddedSkill {
    EmbeddedSkill::new(name, text(name, extra, body))
}

/// A temporary directory, removed when dropped.
pub(super) struct Temp(pub(super) std::path::PathBuf);

impl Temp {
    pub(super) fn new() -> Self {
        let path = std::env::temp_dir().join(format!("arko-embedded-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A folder skill `name` in the folder `folder`.
pub(super) fn folder_skill(folder: &Path, name: &str, body: &str) {
    let dir = folder.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("SKILL.md"), text(name, "", body)).unwrap();
}

pub(super) fn global(path: &Path) -> SkillSource {
    SkillSource {
        path: path.display().to_string(),
        include: None,
        exclude: None,
        optional: false,
    }
}

/// The result of `load_skill` for `name` (and `resource`).
async fn load_skill(manager: &SkillManager, name: &str, resource: Option<&str>) -> String {
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &workspace);
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("name".into(), name.into());
    if let Some(resource) = resource {
        params.insert("resource".into(), resource.into());
    }
    manager
        .create_tool()
        .execute(&context, params)
        .await
        .unwrap()
}

pub(super) fn agent_config() -> AgentConfig {
    AgentConfig::builder(
        "reviewer".into(),
        "Reviewer".into(),
        AgentLLMConfig {
            model: arko_ai::Model::openai_gpt_5_6_luna(),
            generation_config: Default::default(),
        },
    )
    .role_description("Reviews.".into())
    .instructions("Review.".into())
    .build()
    .unwrap()
}

/// An embedded skill is listed to the agent and `load_skill` returns it,
/// without a path or any resource.
#[tokio::test]
async fn an_embedded_skill_is_listed_and_loadable() {
    let config = SkillsConfig {
        embedded: vec![embedded("review", "", "Review things.")],
        ..Default::default()
    };
    let manager = SkillManager::load(&config, Path::new("/nowhere")).unwrap();
    assert!(manager.has_on_demand_skills());
    assert!(!manager.has_auto_loaded_skills());
    assert_eq!(manager.total_skill_count(), 1);
    let listed = &manager.prompt_context().on_demand_skills;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].name, "review");
    assert_eq!(listed[0].description, "The review skill.");

    assert_eq!(
        load_skill(&manager, "review", None).await,
        "--- BEGIN SKILL: review ---\nReview things.\n--- END SKILL: review ---"
    );
    // It has no resources, which the answer says.
    assert_eq!(
        load_skill(&manager, "review", Some("FORMS.md")).await,
        "Resource 'FORMS.md' not found in skill 'review'. This skill has no additional resources."
    );
    assert!(
        load_skill(&manager, "other", None)
            .await
            .contains("Skill 'other' not found. Available skills:\n- review")
    );

    // `setup` offers the tool for it, as it does for a folder skill.
    let mut registry = ToolRegistry::new();
    let mut agent = agent_config();
    manager.setup(&mut registry, &mut agent);
    assert!(registry.get(LOAD_SKILL_TOOL_NAME).is_some());
    assert!(
        agent
            .tool_config
            .available_tool_names
            .contains(&LOAD_SKILL_TOOL_NAME.to_string())
    );
}

/// An embedded skill with `auto_load` is in the system prompt, and the
/// others are listed by name and description.
#[test]
fn an_auto_loaded_embedded_skill_is_in_the_prompt() {
    let config = SkillsConfig {
        embedded: vec![
            embedded("style", "auto_load: true\n", "Write plainly."),
            embedded("review", "", "Review things."),
        ],
        ..Default::default()
    };
    let manager = SkillManager::load(&config, Path::new("/nowhere")).unwrap();
    assert!(manager.has_auto_loaded_skills());
    // Only the on-demand skill is loadable.
    assert_eq!(manager.total_skill_count(), 2);

    let mut registry = ToolRegistry::new();
    let mut agent = agent_config();
    let context = manager.setup(&mut registry, &mut agent);
    let parsing = agent.get_tool_parsing_config();
    let task = crate::Task::new(agent.clone(), "Go.".into(), "/w".into(), None);
    let prompt = PromptBuilder::new(&agent, vec![], &parsing, Some(&context))
        .build_system_prompt(&task)
        .unwrap();
    assert!(
        prompt.contains("--- BEGIN SKILL: style ---\nWrite plainly.\n--- END SKILL: style ---"),
        "{prompt}"
    );
    assert!(
        prompt.contains("- **review**: The review skill."),
        "{prompt}"
    );
    assert!(!prompt.contains("Review things."), "{prompt}");
}

/// A skill of the same name in a folder wins over the embedded one, from
/// whichever source it comes, and the embedded one is left out, not renamed.
#[tokio::test]
async fn a_folder_skill_replaces_the_embedded_skill_of_its_name() {
    let temp = Temp::new();
    let workspace = temp.0.join("project");
    let global_folder = temp.0.join("global");
    folder_skill(&workspace.join(".skills"), "review", "Your review.");
    folder_skill(&global_folder, "deploy", "Your deploy.");
    let config = SkillsConfig {
        workspace_source: Some(SkillSource {
            path: ".skills".into(),
            include: None,
            exclude: None,
            optional: false,
        }),
        global_sources: vec![global(&global_folder)],
        embedded: vec![
            embedded("review", "", "Built-in review."),
            embedded("deploy", "", "Built-in deploy."),
            embedded("style", "", "Built-in style."),
        ],
        ..Default::default()
    };
    let manager = SkillManager::load(&config, &workspace).unwrap();
    let names: Vec<&str> = manager
        .prompt_context()
        .on_demand_skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    // The folders' skills first, then the embedded one that stays.
    assert_eq!(names, ["review", "deploy", "style"]);
    assert_eq!(
        load_skill(&manager, "review", None).await,
        "--- BEGIN SKILL: review ---\nYour review.\n--- END SKILL: review ---"
    );
    assert_eq!(
        load_skill(&manager, "deploy", None).await,
        "--- BEGIN SKILL: deploy ---\nYour deploy.\n--- END SKILL: deploy ---"
    );
    assert_eq!(
        load_skill(&manager, "style", None).await,
        "--- BEGIN SKILL: style ---\nBuilt-in style.\n--- END SKILL: style ---"
    );
    // Nothing was renamed.
    assert!(
        load_skill(&manager, "review-2", None)
            .await
            .contains("not found")
    );

    // The scan says which are embedded, and which folder replaced one.
    let scanned = scan(&config, &workspace);
    let rows: Vec<(&str, bool, Option<String>)> = scanned
        .iter()
        .map(|skill| (skill.name.as_str(), skill.embedded, skill.skipped.clone()))
        .collect();
    assert_eq!(
        rows,
        [
            ("review", false, None),
            ("deploy", false, None),
            (
                "review",
                true,
                Some(format!(
                    "replaced by the skill 'review' in {}",
                    workspace.join(".skills").join("review").display()
                ))
            ),
            (
                "deploy",
                true,
                Some(format!(
                    "replaced by the skill 'deploy' in {}",
                    global_folder.join("deploy").display()
                ))
            ),
            ("style", true, None),
        ]
    );
    assert!(scanned[2].path.as_os_str().is_empty());
    assert_eq!(scanned[0].path, workspace.join(".skills").join("review"));
}

/// A folder skill that its source's list leaves out replaces nothing.
#[test]
fn a_folder_skill_left_out_does_not_replace_the_embedded_one() {
    let temp = Temp::new();
    folder_skill(&temp.0, "review", "Your review.");
    let config = SkillsConfig {
        global_sources: vec![SkillSource {
            path: temp.0.display().to_string(),
            include: None,
            exclude: Some(vec!["review".into()]),
            optional: false,
        }],
        embedded: vec![embedded("review", "", "Built-in review.")],
        ..Default::default()
    };
    let manager = SkillManager::load(&config, &temp.0).unwrap();
    assert_eq!(manager.prompt_context().on_demand_skills.len(), 1);
    let scanned = scan(&config, &temp.0);
    assert_eq!(scanned.len(), 2);
    assert!(scanned[0].skipped.is_some() && !scanned[0].embedded);
    assert!(scanned[1].skipped.is_none() && scanned[1].embedded);
}

/// Embedded skills among themselves go by the rules of folders. The same
/// content loads once, and another skill with the name is renamed.
#[test]
fn embedded_skills_with_the_same_name_follow_the_folder_rules() {
    let config = SkillsConfig {
        embedded: vec![
            embedded("review", "", "First."),
            embedded("review", "", "Second."),
            embedded("again", "", "First."),
        ],
        ..Default::default()
    };
    let (skills, _) = loader::load_skills(&config, Path::new("/nowhere")).unwrap();
    let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
    assert_eq!(names, ["review", "review-2"]);
    assert!(skills.iter().all(|skill| skill.source_label == "embedded"));
    assert!(skills.iter().all(|skill| skill.resources.is_empty()));
}

/// The front matter's name is the skill's name, as for a folder skill.
#[test]
fn the_front_matter_names_an_embedded_skill() {
    let config = SkillsConfig {
        embedded: vec![EmbeddedSkill::new(
            "file-name",
            text("front-matter-name", "", "Body."),
        )],
        ..Default::default()
    };
    let manager = SkillManager::load(&config, Path::new("/nowhere")).unwrap();
    assert_eq!(
        manager.prompt_context().on_demand_skills[0].name,
        "front-matter-name"
    );
}

/// An invalid embedded skill is skipped under `Warn` and refuses the load
/// under `Error`, which names it.
#[test]
fn an_invalid_embedded_skill_follows_the_invalid_skill_policy() {
    let mut config = SkillsConfig {
        embedded: vec![
            EmbeddedSkill::new("broken", "No front matter.\n"),
            embedded("fine", "", "Fine."),
            EmbeddedSkill::new("nameless", "---\ndescription: No name.\n---\nBody.\n"),
        ],
        ..Default::default()
    };
    let manager = SkillManager::load(&config, Path::new("/nowhere")).unwrap();
    let names: Vec<&str> = manager
        .prompt_context()
        .on_demand_skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect();
    assert_eq!(names, ["fine"]);
    // The scan lists all, with why the invalid ones are skipped.
    let scanned = scan(&config, Path::new("/nowhere"));
    assert_eq!(scanned.len(), 3);
    assert_eq!(scanned[0].name, "broken");
    assert!(scanned[0].embedded);
    assert!(
        scanned[0]
            .skipped
            .as_deref()
            .unwrap()
            .contains("embedded skill 'broken'"),
        "{:?}",
        scanned[0].skipped
    );
    assert!(scanned[1].skipped.is_none());
    assert_eq!(scanned[2].name, "nameless");
    assert!(scanned[2].skipped.is_some());

    config.on_invalid_skill = InvalidSkillPolicy::Error;
    let error = SkillManager::load(&config, Path::new("/nowhere"))
        .err()
        .unwrap();
    assert!(matches!(error, ArkoAgentsError::Skill(_)), "{error}");
    assert!(
        error.to_string().contains("embedded skill 'broken'"),
        "{error}"
    );
}

/// A configuration saved before embedded skills existed loads, and one
/// without any saves as before.
#[test]
fn a_saved_configuration_without_embedded_skills_still_loads() {
    let old = json!({
        "global_sources": [{ "path": "/skills" }],
        "on_invalid_skill": "Error"
    });
    let config: SkillsConfig = serde_json::from_value(old.clone()).unwrap();
    assert!(config.embedded.is_empty());
    assert_eq!(config.global_sources[0].path, "/skills");
    assert_eq!(config.on_invalid_skill, InvalidSkillPolicy::Error);
    // Saved again, it has no `embedded` key.
    let saved = serde_json::to_value(&config).unwrap();
    assert!(saved.get("embedded").is_none(), "{saved}");
    assert_eq!(
        serde_json::to_value(SkillsConfig::default()).unwrap(),
        json!({ "on_invalid_skill": "Warn" })
    );

    // The same inside an agent configuration saved before.
    let mut agent = serde_json::to_value(agent_config()).unwrap();
    agent["skills_config"] = old;
    let agent: AgentConfig = serde_json::from_value(agent).unwrap();
    assert!(agent.skills_config.unwrap().embedded.is_empty());
}

/// Embedded skills are program code. A saved configuration, and the task
/// that holds it, doesn't have them, and one that lists them still loads.
#[test]
fn embedded_skills_are_not_saved_with_a_configuration() {
    let config = SkillsConfig {
        embedded: vec![embedded("review", "auto_load: true\n", "Review things.")],
        global_sources: vec![global(Path::new("/skills"))],
        ..Default::default()
    };
    let saved = serde_json::to_value(&config).unwrap();
    assert!(saved.get("embedded").is_none(), "{saved}");
    let loaded: SkillsConfig = serde_json::from_value(saved).unwrap();
    assert!(loaded.embedded.is_empty());
    assert_eq!(loaded.global_sources[0].path, "/skills");

    // A configuration that lists them, such as a file someone wrote, loads.
    let written = json!({ "embedded": [{
        "name": "review",
        "content": text("review", "", "Review things."),
    }] });
    let loaded: SkillsConfig = serde_json::from_value(written).unwrap();
    assert_eq!(loaded.embedded, [embedded("review", "", "Review things.")]);

    // Inside an agent configuration, and so inside a saved task.
    let mut agent = agent_config();
    agent.skills_config = Some(config.clone());
    let task = crate::Task::new(agent, "Go.".into(), "/w".into(), None);
    assert_eq!(
        task.agent_config.skills_config.as_ref().unwrap().embedded,
        config.embedded
    );
    let task: crate::Task = serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
    let skills = task.agent_config.skills_config.unwrap();
    assert!(skills.embedded.is_empty());
    assert_eq!(skills.global_sources.len(), 1);
}
