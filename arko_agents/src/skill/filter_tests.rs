//! Tests of the model filter, the worker filters, the live `load_skill`,
//! optional sources and the readers of skills offered to be saved.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, Once};

use serde_json::{Value, json};
use zip::write::SimpleFileOptions;

use super::embedded_tests::{Temp, agent_config, embedded, global, text};
use super::*;
use crate::error::ArkoAgentsError;
use crate::parsing::PromptBuilder;
use crate::tool::{TaskContext, Tool, ToolRegistry};

/// A folder skill `name` whose front matter has the `extra` lines.
fn skill_with(folder: &Path, name: &str, extra: &str, body: &str) {
    let dir = folder.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("SKILL.md"), text(name, extra, body)).unwrap();
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| name.to_string()).collect()
}

/// `load_skill` as `tool` answers for `name`.
async fn call(tool: &LoadSkillTool, name: &str) -> String {
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &workspace);
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("name".into(), name.into());
    tool.execute(&context, params).await.unwrap()
}

fn listed(manager: &SkillManager) -> Vec<&str> {
    manager
        .prompt_context()
        .on_demand_skills
        .iter()
        .map(|skill| skill.name.as_str())
        .collect()
}

fn loaded_names(config: &SkillsConfig, workspace: &Path) -> Vec<String> {
    scan(config, workspace)
        .into_iter()
        .filter(|skill| skill.skipped.is_none())
        .map(|skill| skill.name)
        .collect()
}

/// The system prompt of an agent given the skills `manager` loaded.
fn prompt_of(manager: &SkillManager) -> String {
    let mut registry = ToolRegistry::new();
    let mut agent = agent_config();
    let context = manager.setup(&mut registry, &mut agent);
    let parsing = agent.get_tool_parsing_config();
    let task = crate::Task::new(agent.clone(), "Go.".into(), "/w".into(), None);
    PromptBuilder::new(&agent, vec![], &parsing, Some(&context))
        .build_system_prompt(&task)
        .unwrap()
}

// A configuration saved by 0.3.

#[test]
fn a_configuration_saved_by_0_3_loads_with_the_new_fields_off() {
    let saved = json!({
        "workspace_source": {"path": ".skills"},
        "global_sources": [{"path": "/g", "include": ["a"], "exclude": ["b"]}],
        "on_invalid_skill": "Error"
    });
    let config: SkillsConfig = serde_json::from_value(saved.clone()).unwrap();
    assert!(!config.workspace_source.as_ref().unwrap().optional);
    assert!(!config.global_sources[0].optional);
    assert!(config.model_names.is_empty());
    assert!(!config.subagent);
    assert_eq!(config.only, None);
    assert!(!config.offer_when_empty);
    // And it saves as it was, so a comparison with a saved one holds.
    assert_eq!(serde_json::to_value(&config).unwrap(), saved);
    assert_eq!(
        serde_json::to_value(SkillsConfig::default()).unwrap(),
        json!({"on_invalid_skill": "Warn"})
    );
}

#[test]
fn the_new_fields_are_saved_and_read_back() {
    let config = SkillsConfig {
        global_sources: vec![SkillSource {
            path: "/g".into(),
            include: None,
            exclude: None,
            optional: true,
        }],
        model_names: names(&["haiku", "cc:haiku"]),
        subagent: true,
        only: Some(names(&["a"])),
        offer_when_empty: true,
        ..Default::default()
    };
    let saved = serde_json::to_value(&config).unwrap();
    assert_eq!(saved["global_sources"][0]["optional"], true);
    assert_eq!(saved["model_names"], json!(["haiku", "cc:haiku"]));
    assert_eq!(saved["subagent"], true);
    assert_eq!(saved["only"], json!(["a"]));
    assert_eq!(saved["offer_when_empty"], true);
    let back: SkillsConfig = serde_json::from_value(saved).unwrap();
    assert!(back.global_sources[0].optional);
    assert_eq!(back.model_names, config.model_names);
    assert!(back.subagent);
    assert_eq!(back.only, config.only);
    assert!(back.offer_when_empty);
}

// The model filter.

#[tokio::test]
async fn a_skill_for_other_models_is_unlisted_and_loads_by_name() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(&folder, "any-model", "", "For all.");
    skill_with(&folder, "small", "models: [haiku, luna]\n", "For small.");
    skill_with(&folder, "pinned", "models: claude-opus-5-5\n", "For opus.");
    let config = |model_names: &[&str]| SkillsConfig {
        global_sources: vec![global(&folder)],
        model_names: names(model_names),
        ..Default::default()
    };

    // A Haiku agent.
    let haiku = config(&["claude-haiku-5-5", "haiku"]);
    let manager = SkillManager::load(&haiku, &temp.0).unwrap();
    assert_eq!(listed(&manager), ["any-model", "small"]);
    let prompt = prompt_of(&manager);
    assert!(prompt.contains("- **small**: The small skill."), "{prompt}");
    assert!(!prompt.contains("pinned"), "{prompt}");
    let tool = manager.create_tool();
    // Any skill loads by its exact name, matching or not.
    assert!(call(&tool, "pinned").await.contains("For opus."));
    assert!(call(&tool, "small").await.contains("For small."));
    // The list of an unknown name holds the skills for the model only.
    let missing = call(&tool, "other").await;
    assert_eq!(
        missing,
        "Skill 'other' not found. Available skills:\n- any-model\n- small"
    );

    // An Opus agent.
    let opus = config(&["claude-opus-5-5", "opus"]);
    let manager = SkillManager::load(&opus, &temp.0).unwrap();
    assert_eq!(listed(&manager), ["any-model", "pinned"]);
    assert!(manager.has_on_demand_skills());
    assert_eq!(manager.total_skill_count(), 3);
    assert!(
        call(&manager.create_tool(), "small")
            .await
            .contains("For small.")
    );

    // No names, no filter.
    let manager = SkillManager::load(&config(&[]), &temp.0).unwrap();
    assert_eq!(listed(&manager), ["any-model", "pinned", "small"]);
}

#[test]
fn the_scan_says_which_skills_are_listed() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(&folder, "a-any", "", "A.");
    skill_with(&folder, "b-small", "models: haiku\nworkers: false\n", "B.");
    std::fs::create_dir_all(folder.join("c-broken")).unwrap();
    std::fs::write(folder.join("c-broken/SKILL.md"), "No front matter.").unwrap();
    let config = SkillsConfig {
        global_sources: vec![global(&folder)],
        model_names: names(&["opus"]),
        ..Default::default()
    };
    let scanned = scan(&config, &temp.0);
    assert_eq!(scanned.len(), 3);
    assert!(scanned[0].listed);
    assert!(scanned[0].models.is_empty());
    assert!(scanned[0].workers);
    assert!(!scanned[1].listed);
    assert_eq!(scanned[1].models, ["haiku"]);
    assert!(!scanned[1].workers);
    assert_eq!(scanned[1].skipped, None);
    // One that can't be read isn't listed.
    assert!(!scanned[2].listed);
    assert!(scanned[2].skipped.is_some());

    let haiku = SkillsConfig {
        model_names: names(&["haiku"]),
        ..config
    };
    assert!(scan(&haiku, &temp.0)[1].listed);
}

#[tokio::test]
async fn auto_load_applies_only_on_a_matching_model() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(
        &folder,
        "plain-style",
        "auto_load: true\nmodels: haiku\n",
        "Write plainly.",
    );
    skill_with(&folder, "always", "auto_load: true\n", "Always on.");
    let config = |model_names: &[&str]| SkillsConfig {
        global_sources: vec![global(&folder)],
        model_names: names(model_names),
        ..Default::default()
    };

    // A match puts the content in the prompt, and `load_skill` has no more to give.
    let manager = SkillManager::load(&config(&["haiku"]), &temp.0).unwrap();
    let prompt = prompt_of(&manager);
    assert!(prompt.contains("Write plainly."), "{prompt}");
    assert!(prompt.contains("Always on."), "{prompt}");
    assert!(!manager.has_on_demand_skills());
    assert!(
        call(&manager.create_tool(), "plain-style")
            .await
            .starts_with("Skill 'plain-style' not found")
    );

    // No match leaves it out of the prompt, loads on demand and stays unlisted.
    let manager = SkillManager::load(&config(&["opus"]), &temp.0).unwrap();
    let prompt = prompt_of(&manager);
    assert!(!prompt.contains("Write plainly."), "{prompt}");
    assert!(!prompt.contains("plain-style"), "{prompt}");
    assert!(prompt.contains("Always on."), "{prompt}");
    assert!(manager.prompt_context().on_demand_skills.is_empty());
    assert!(manager.has_on_demand_skills());
    let tool = manager.create_tool();
    assert_eq!(
        call(&tool, "plain-style").await,
        "--- BEGIN SKILL: plain-style ---\nWrite plainly.\n--- END SKILL: plain-style ---"
    );
    // Its name isn't in the list of an unknown name either.
    assert_eq!(
        call(&tool, "other").await,
        "Skill 'other' not found. Available skills:\nNo on-demand skills are currently available."
    );

    // The tool registers for it, so an agent can load the skill by name.
    let mut registry = ToolRegistry::new();
    let mut agent = agent_config();
    manager.setup(&mut registry, &mut agent);
    assert!(registry.get(LOAD_SKILL_TOOL_NAME).is_some());
}

#[test]
fn the_tool_says_that_skills_added_since_the_list_load_by_name_too() {
    let tool = LoadSkillTool::new(SkillsConfig::default(), "/nowhere");
    let description = tool.definition().description;
    assert!(
        description.contains("Skills added since the conversation started"),
        "{description}"
    );
    assert!(!description.contains(';'), "{description}");
}

// The worker filters.

#[test]
fn only_leaves_the_other_folder_and_embedded_skills_out() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(&folder, "alpha", "", "Alpha.");
    skill_with(&folder, "beta", "", "Beta.");
    let config = SkillsConfig {
        global_sources: vec![global(&folder)],
        embedded: vec![
            embedded("gamma", "", "Gamma."),
            embedded("delta", "", "Delta."),
        ],
        only: Some(names(&["beta", "delta", "no-such-skill"])),
        ..Default::default()
    };
    let manager = SkillManager::load(&config, &temp.0).unwrap();
    assert_eq!(listed(&manager), ["beta", "delta"]);
    assert_eq!(manager.total_skill_count(), 2);
    let scanned = scan(&config, &temp.0);
    let skipped: Vec<(&str, Option<&str>)> = scanned
        .iter()
        .map(|skill| (skill.name.as_str(), skill.skipped.as_deref()))
        .collect();
    assert_eq!(
        skipped,
        [
            ("alpha", Some("not in the agent's only list")),
            ("beta", None),
            ("gamma", Some("not in the agent's only list")),
            ("delta", None),
        ]
    );

    // An empty list leaves everything out.
    let none = SkillsConfig {
        only: Some(Vec::new()),
        ..config.clone()
    };
    assert!(loaded_names(&none, &temp.0).is_empty());
    assert!(
        !SkillManager::load(&none, &temp.0)
            .unwrap()
            .has_on_demand_skills()
    );
    // No list leaves nothing out.
    let all = SkillsConfig {
        only: None,
        ..config
    };
    assert_eq!(loaded_names(&all, &temp.0).len(), 4);
}

/// A saved run's configuration has no embedded skills, and a resume gives
/// the registered ones again (`restore_embedded_skills` in arko_workflows),
/// which can't widen the list.
#[test]
fn only_still_holds_when_a_resume_gives_the_embedded_skills_again() {
    let registered = SkillsConfig {
        embedded: vec![
            embedded("gamma", "", "Gamma."),
            embedded("delta", "", "Delta."),
        ],
        ..Default::default()
    };
    let worker = SkillsConfig {
        only: Some(names(&["gamma"])),
        ..registered.clone()
    };
    let temp = Temp::new();
    assert_eq!(loaded_names(&worker, &temp.0), ["gamma"]);

    let saved = serde_json::to_value(&worker).unwrap();
    let mut resumed: SkillsConfig = serde_json::from_value(saved).unwrap();
    assert!(resumed.embedded.is_empty());
    assert_eq!(loaded_names(&resumed, &temp.0), Vec::<String>::new());
    resumed.embedded = registered.embedded.clone();
    assert_eq!(loaded_names(&resumed, &temp.0), ["gamma"]);
    let manager = SkillManager::load(&resumed, &temp.0).unwrap();
    assert_eq!(listed(&manager), ["gamma"]);
}

#[test]
fn subagent_leaves_out_the_skills_for_the_main_agent_only() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(&folder, "shared", "", "Shared.");
    skill_with(&folder, "private", "workers: false\n", "Private.");
    let config = |subagent| SkillsConfig {
        global_sources: vec![global(&folder)],
        embedded: vec![
            embedded("built-in-private", "workers: false\n", "Built-in private."),
            embedded("built-in", "workers: true\n", "Open."),
        ],
        subagent,
        ..Default::default()
    };
    // The main agent has all of them.
    assert_eq!(
        loaded_names(&config(false), &temp.0),
        ["private", "shared", "built-in-private", "built-in"]
    );
    // A worker has the others.
    let worker = config(true);
    assert_eq!(loaded_names(&worker, &temp.0), ["shared", "built-in"]);
    let manager = SkillManager::load(&worker, &temp.0).unwrap();
    assert_eq!(listed(&manager), ["shared", "built-in"]);
    let scanned = scan(&worker, &temp.0);
    assert_eq!(scanned[0].name, "private");
    assert_eq!(
        scanned[0].skipped.as_deref(),
        Some("for the main agent only (workers: false)")
    );
    assert!(!scanned[0].workers);
}

#[test]
fn the_worker_filters_keep_the_names_the_main_agent_has() {
    let temp = Temp::new();
    let first = temp.0.join("first");
    let second = temp.0.join("second");
    skill_with(&first, "notes", "workers: false\n", "The first.");
    skill_with(&second, "notes", "", "The second.");
    let config = |subagent| SkillsConfig {
        global_sources: vec![global(&first), global(&second)],
        subagent,
        ..Default::default()
    };
    // The main agent's second skill of the name is `notes-2`, and a worker's
    // is the same, so a list of names means the same skill to both.
    assert_eq!(loaded_names(&config(false), &temp.0), ["notes", "notes-2"]);
    assert_eq!(loaded_names(&config(true), &temp.0), ["notes-2"]);
    let only = SkillsConfig {
        only: Some(names(&["notes-2"])),
        ..config(false)
    };
    assert_eq!(loaded_names(&only, &temp.0), ["notes-2"]);
}

// The live `load_skill`.

#[tokio::test]
async fn load_skill_reads_the_sources_at_each_call() {
    let temp = Temp::new();
    let workspace = temp.0.join("project");
    let folder = workspace.join(".skills");
    skill_with(&folder, "existing", "", "Version one.");
    let config = SkillsConfig {
        workspace_source: Some(SkillSource {
            path: ".skills".into(),
            include: None,
            exclude: None,
            optional: false,
        }),
        ..Default::default()
    };
    let manager = SkillManager::load(&config, &workspace).unwrap();
    let tool = manager.create_tool();
    assert!(call(&tool, "existing").await.contains("Version one."));
    assert!(
        call(&tool, "later")
            .await
            .starts_with("Skill 'later' not found")
    );

    // A skill written after the run started.
    skill_with(&folder, "later", "", "Written later.");
    assert!(call(&tool, "later").await.contains("Written later."));
    // The list of an unknown name has it too.
    assert!(call(&tool, "other").await.ends_with("- existing\n- later"));
    // The system prompt keeps the list it started with.
    assert_eq!(listed(&manager), ["existing"]);

    // An edit.
    skill_with(&folder, "existing", "", "Version two.");
    let edited = call(&tool, "existing").await;
    assert!(edited.contains("Version two."), "{edited}");
    assert!(!edited.contains("Version one."), "{edited}");

    // A resource written later.
    std::fs::write(folder.join("later/FORMS.md"), "Form text.").unwrap();
    assert!(call(&tool, "later").await.contains("- FORMS.md"));

    // A removed skill.
    std::fs::remove_dir_all(folder.join("later")).unwrap();
    assert!(
        call(&tool, "later")
            .await
            .starts_with("Skill 'later' not found")
    );

    // A folder that didn't exist when the run started.
    std::fs::remove_dir_all(&folder).unwrap();
    assert!(
        call(&tool, "existing")
            .await
            .starts_with("Skill 'existing' not found")
    );
    skill_with(&folder, "existing", "", "Back again.");
    assert!(call(&tool, "existing").await.contains("Back again."));
}

#[tokio::test]
async fn load_skill_applies_the_filters_to_what_it_reads() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(&folder, "kept", "", "Kept.");
    let config = SkillsConfig {
        global_sources: vec![global(&folder)],
        only: Some(names(&["kept"])),
        ..Default::default()
    };
    let tool = SkillManager::load(&config, &temp.0).unwrap().create_tool();
    // Written later, but not among the agent's skills.
    skill_with(&folder, "other", "", "Other.");
    assert!(
        call(&tool, "other")
            .await
            .starts_with("Skill 'other' not found")
    );
    assert!(call(&tool, "kept").await.contains("Kept."));
}

#[tokio::test]
async fn load_skill_skips_a_skill_that_broke_after_the_run_started() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    skill_with(&folder, "good", "", "Good.");
    skill_with(&folder, "fragile", "", "Fragile.");
    let config = SkillsConfig {
        global_sources: vec![global(&folder)],
        on_invalid_skill: InvalidSkillPolicy::Error,
        ..Default::default()
    };
    let tool = SkillManager::load(&config, &temp.0).unwrap().create_tool();
    std::fs::write(folder.join("fragile/SKILL.md"), "Not a skill.").unwrap();
    assert!(call(&tool, "good").await.contains("Good."));
    assert!(
        call(&tool, "fragile")
            .await
            .starts_with("Skill 'fragile' not found")
    );
}

#[test]
fn offer_when_empty_registers_load_skill_without_skills() {
    let setup = |config: &SkillsConfig| {
        let manager = SkillManager::load(config, Path::new("/nowhere")).unwrap();
        let mut registry = ToolRegistry::new();
        let mut agent = agent_config();
        manager.setup(&mut registry, &mut agent);
        let offered = agent
            .tool_config
            .available_tool_names
            .contains(&LOAD_SKILL_TOOL_NAME.to_string());
        assert_eq!(registry.get(LOAD_SKILL_TOOL_NAME).is_some(), offered);
        (offered, agent)
    };
    let (offered, _) = setup(&SkillsConfig::default());
    assert!(!offered);
    let (offered, agent) = setup(&SkillsConfig {
        offer_when_empty: true,
        ..Default::default()
    });
    assert!(offered);
    assert_eq!(
        agent.tool_config.tool_specific_configs[LOAD_SKILL_TOOL_NAME].auto_approve_tool_use,
        Some(true)
    );
    // Also when the filters leave nothing.
    let (offered, _) = setup(&SkillsConfig {
        embedded: vec![embedded("a", "", "A.")],
        only: Some(Vec::new()),
        offer_when_empty: true,
        ..Default::default()
    });
    assert!(offered);
}

#[tokio::test]
async fn an_offered_load_skill_finds_the_first_skill_written_later() {
    let temp = Temp::new();
    let folder = temp.0.join("skills");
    let config = SkillsConfig {
        global_sources: vec![SkillSource {
            path: folder.display().to_string(),
            include: None,
            exclude: None,
            optional: true,
        }],
        offer_when_empty: true,
        ..Default::default()
    };
    let manager = SkillManager::load(&config, &temp.0).unwrap();
    let mut registry = ToolRegistry::new();
    let mut agent = agent_config();
    manager.setup(&mut registry, &mut agent);
    let tool = manager.create_tool();
    assert_eq!(
        call(&tool, "first").await,
        "Skill 'first' not found. Available skills:\nNo on-demand skills are currently available."
    );
    skill_with(&folder, "first", "", "The first.");
    assert!(call(&tool, "first").await.contains("The first."));
}

// Optional sources.

static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        LOG.lock().unwrap().push(record.args().to_string());
    }
    fn flush(&self) {}
}

/// The warnings and errors logged that mention `needle`, from now on.
fn logged(needle: &str) -> Vec<String> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Warn);
    });
    LOG.lock()
        .unwrap()
        .iter()
        .filter(|line| line.contains(needle))
        .cloned()
        .collect()
}

#[test]
fn a_missing_optional_source_is_empty_without_a_warning_under_either_policy() {
    // Installs the logger before anything runs.
    logged("");
    let temp = Temp::new();
    let missing = temp.0.join("not-made-yet");
    let source = |optional| SkillSource {
        path: missing.display().to_string(),
        include: None,
        exclude: None,
        optional,
    };
    for policy in [InvalidSkillPolicy::Warn, InvalidSkillPolicy::Error] {
        let config = SkillsConfig {
            global_sources: vec![source(true)],
            on_invalid_skill: policy,
            ..Default::default()
        };
        let manager = SkillManager::load(&config, &temp.0).unwrap();
        assert_eq!(manager.total_skill_count(), 0);
        assert!(scan(&config, &temp.0).is_empty());
    }
    assert_eq!(logged(&temp.0.display().to_string()), Vec::<String>::new());

    // A source that isn't optional still warns, or refuses under Error.
    let required = SkillsConfig {
        global_sources: vec![source(false)],
        ..Default::default()
    };
    SkillManager::load(&required, &temp.0).unwrap();
    assert_eq!(logged(&temp.0.display().to_string()).len(), 1);
    let strict = SkillsConfig {
        on_invalid_skill: InvalidSkillPolicy::Error,
        ..required
    };
    assert!(matches!(
        SkillManager::load(&strict, &temp.0),
        Err(ArkoAgentsError::Configuration(_))
    ));
}

#[test]
fn an_optional_source_that_is_a_file_is_still_a_problem() {
    let temp = Temp::new();
    let file = temp.0.join("file");
    std::fs::write(&file, "Not a folder.").unwrap();
    let config = SkillsConfig {
        global_sources: vec![SkillSource {
            path: file.display().to_string(),
            include: None,
            exclude: None,
            optional: true,
        }],
        on_invalid_skill: InvalidSkillPolicy::Error,
        ..Default::default()
    };
    assert!(SkillManager::load(&config, &temp.0).is_err());
}

#[test]
fn an_optional_source_is_read_once_it_exists() {
    let temp = Temp::new();
    let folder = temp.0.join("later");
    let config = SkillsConfig {
        global_sources: vec![SkillSource {
            path: folder.display().to_string(),
            include: None,
            exclude: None,
            optional: true,
        }],
        ..Default::default()
    };
    assert!(loaded_names(&config, &temp.0).is_empty());
    skill_with(&folder, "made", "", "Made.");
    assert_eq!(loaded_names(&config, &temp.0), ["made"]);
}

// Skills offered to be saved.

fn zip_of(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, content) in files {
        writer
            .start_file(*name, SimpleFileOptions::default())
            .unwrap();
        writer.write_all(content).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn skill_md(name: &str, extra: &str) -> String {
    text(name, extra, "Body.")
}

#[test]
fn a_candidate_folder_is_read_as_loading_reads_it() {
    let temp = Temp::new();
    let dir = temp.0.join("release-notes");
    std::fs::create_dir_all(dir.join("scripts")).unwrap();
    std::fs::create_dir_all(dir.join("reference")).unwrap();
    std::fs::write(
        dir.join("SKILL.md"),
        skill_md(
            "release-notes",
            "models: [haiku, \"gpt-6*\"]\nworkers: false\n",
        ),
    )
    .unwrap();
    std::fs::write(dir.join("reference/style.md"), "Style.").unwrap();
    std::fs::write(dir.join("scripts/run.sh"), "echo hi").unwrap();

    let skill = read_candidate(&dir).unwrap();
    assert_eq!(skill.name, "release-notes");
    assert_eq!(skill.description, "The release-notes skill.");
    assert!(!skill.auto_load);
    assert_eq!(skill.models, ["haiku", "gpt-6*"]);
    assert!(!skill.workers);
    assert_eq!(skill.content.trim(), "Body.");
    assert_eq!(
        skill.resources.keys().collect::<Vec<_>>(),
        ["reference/style.md"]
    );

    // What loading makes of it is the same.
    let config = SkillsConfig {
        global_sources: vec![global(&temp.0)],
        ..Default::default()
    };
    let (loaded, _) = loader::load_skills(&config, &temp.0).unwrap();
    assert_eq!(loaded[0].content, skill.content);
    assert_eq!(loaded[0].resources, skill.resources);
    assert_eq!(loaded[0].models, skill.models);

    // A folder without a SKILL.md, and a front matter that doesn't parse.
    let empty = temp.0.join("empty");
    std::fs::create_dir(&empty).unwrap();
    assert!(matches!(
        read_candidate(&empty),
        Err(SkillLoadError::MissingSkillMd { .. })
    ));
    std::fs::write(empty.join("SKILL.md"), "---\nname: [\n---\n").unwrap();
    assert!(matches!(
        read_candidate(&empty),
        Err(SkillLoadError::FrontmatterParseError { .. })
    ));
    // A resource that isn't UTF-8 text.
    std::fs::write(dir.join("reference/bad.md"), [0xff, 0xfe, 0x00]).unwrap();
    assert!(matches!(
        read_candidate(&dir),
        Err(SkillLoadError::Io { .. })
    ));
}

#[test]
fn a_candidate_file_is_a_markdown_file_or_a_bundle() {
    let temp = Temp::new();
    let md = temp.0.join("notes.md");
    std::fs::write(&md, skill_md("notes", "auto_load: true\n")).unwrap();
    let skill = read_candidate(&md).unwrap();
    assert_eq!(skill.name, "notes");
    assert!(skill.auto_load);
    assert!(skill.resources.is_empty());
    assert!(skill.workers);

    let bundle = zip_of(&[
        ("SKILL.md", skill_md("packed", "models: haiku\n").as_bytes()),
        ("FORMS.md", b"Forms."),
        ("scripts/run.sh", b"echo hi"),
        ("scripts/NOTES.md", b"Not a resource."),
    ]);
    for extension in ["skill", "zip"] {
        let path = temp.0.join(format!("packed.{extension}"));
        std::fs::write(&path, &bundle).unwrap();
        let skill = read_candidate(&path).unwrap();
        assert_eq!(skill.name, "packed");
        assert_eq!(skill.models, ["haiku"]);
        assert_eq!(skill.resources.keys().collect::<Vec<_>>(), ["FORMS.md"]);
    }
    // The bytes alone, with the files in one top folder.
    let nested = zip_of(&[
        ("packed/SKILL.md", skill_md("packed", "").as_bytes()),
        ("packed/reference/a.md", b"A."),
    ]);
    let skill = read_bundle(&nested).unwrap();
    assert_eq!(skill.name, "packed");
    assert_eq!(
        skill.resources.keys().collect::<Vec<_>>(),
        ["reference/a.md"]
    );

    // Another kind of file, a missing path, a bundle without a SKILL.md and
    // bytes that aren't a bundle.
    let text_file = temp.0.join("notes.txt");
    std::fs::write(&text_file, "text").unwrap();
    assert!(matches!(
        read_candidate(&text_file),
        Err(SkillLoadError::NotASkill { .. })
    ));
    assert!(matches!(
        read_candidate(temp.0.join("missing.md")),
        Err(SkillLoadError::Io { .. })
    ));
    assert!(matches!(
        read_bundle(&zip_of(&[("README.md", b"No skill.")])),
        Err(SkillLoadError::MissingSkillMd { .. })
    ));
    assert!(matches!(
        read_bundle(b"---\nname: x\n---\n"),
        Err(SkillLoadError::ZipError { .. })
    ));
}

#[test]
fn a_bundle_with_an_unsafe_entry_is_refused() {
    let skill = skill_md("packed", "");
    let with = |name: &str| zip_of(&[("SKILL.md", skill.as_bytes()), (name, b"Text.")]);
    for name in [
        "../outside.md",
        "reference/../../outside.md",
        "/etc/outside.md",
        "C:/outside.md",
        "..\\outside.md",
        "scripts/../../up.sh",
        "up/../x.md",
        "a/../b/c.md",
    ] {
        let error = read_bundle(&with(name)).unwrap_err();
        assert!(
            matches!(&error, SkillLoadError::ZipError { message, .. }
                if message.contains("absolute path") || message.contains("leaves the skill's folder")),
            "{name}: {error}"
        );
    }
    // Loading from a source folder reads such an entry into memory only and
    // doesn't refuse it.
    let temp = Temp::new();
    std::fs::write(temp.0.join("odd.skill"), with("../outside.md")).unwrap();
    let config = SkillsConfig {
        global_sources: vec![global(&temp.0)],
        ..Default::default()
    };
    assert_eq!(loaded_names(&config, &temp.0), ["packed"]);
}

#[test]
fn a_bundle_with_a_link_is_refused() {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer
        .start_file("SKILL.md", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(skill_md("packed", "").as_bytes()).unwrap();
    writer
        .add_symlink("link", "/etc/passwd", SimpleFileOptions::default())
        .unwrap();
    let bundle = writer.finish().unwrap().into_inner();
    let error = read_bundle(&bundle).unwrap_err();
    assert!(
        matches!(&error, SkillLoadError::ZipError { message, .. } if message.contains("is a link")),
        "{error}"
    );
    // The same for a link named like a resource.
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer
        .start_file("SKILL.md", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(skill_md("packed", "").as_bytes()).unwrap();
    writer
        .add_symlink("notes.md", "../../secret.md", SimpleFileOptions::default())
        .unwrap();
    assert!(read_bundle(&writer.finish().unwrap().into_inner()).is_err());
}

#[test]
fn a_bundle_with_text_that_is_not_utf8_or_too_much_of_it_is_refused() {
    let skill = skill_md("packed", "");
    let bad = zip_of(&[
        ("SKILL.md", skill.as_bytes()),
        ("a.md", &[0xff, 0xfe, 0x00]),
    ]);
    let error = read_bundle(&bad).unwrap_err();
    assert!(
        matches!(&error, SkillLoadError::ZipError { message, .. } if message.contains("a.md")),
        "{error}"
    );
    // Loading passes the file over, as it always did.
    let temp = Temp::new();
    std::fs::write(temp.0.join("bad.skill"), &bad).unwrap();
    let config = SkillsConfig {
        global_sources: vec![global(&temp.0)],
        ..Default::default()
    };
    let (loaded, _) = loader::load_skills(&config, &temp.0).unwrap();
    assert!(loaded[0].resources.is_empty());

    // A small file that unpacks to a lot.
    let zeros = vec![0u8; 17 * 1024 * 1024];
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer
        .start_file("SKILL.md", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(skill.as_bytes()).unwrap();
    writer
        .start_file(
            "big.dat",
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
        )
        .unwrap();
    writer.write_all(&zeros).unwrap();
    let bomb = writer.finish().unwrap().into_inner();
    assert!(bomb.len() < 100 * 1024);
    let error = read_bundle(&bomb).unwrap_err();
    assert!(
        matches!(&error, SkillLoadError::ZipError { message, .. } if message.contains("unpacked")),
        "{error}"
    );
}

// Unpacking a bundle.

/// The names of the entries under `dir`, with `/` between folders.
fn files_under(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = walkdir::WalkDir::new(dir)
        .min_depth(1)
        .into_iter()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| {
            entry
                .path()
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    found.sort();
    found
}

#[test]
fn unpacking_writes_every_file_and_returns_the_skill() {
    let temp = Temp::new();
    let executable = SimpleFileOptions::default().unix_permissions(0o755);
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, content) in [
        (
            "SKILL.md",
            skill_md("packed", "models: haiku\n").into_bytes(),
        ),
        ("reference/style.md", b"Style.".to_vec()),
        ("scripts/run.sh", b"#!/bin/sh\necho hi".to_vec()),
        ("data/table.bin", vec![0xff, 0x00, 0x01]),
        (".hidden", b"Hidden.".to_vec()),
    ] {
        writer.start_file(name, executable).unwrap();
        writer.write_all(&content).unwrap();
    }
    writer.add_directory("empty/", executable).unwrap();
    let bundle = writer.finish().unwrap().into_inner();

    // The parents are made, and `dest` is.
    let dest = temp.0.join("skills/nested/packed");
    let skill = unpack_bundle(&bundle, &dest).unwrap();
    assert_eq!(skill.name, "packed");
    assert_eq!(skill.models, ["haiku"]);
    assert_eq!(
        files_under(&dest),
        [
            ".hidden",
            "SKILL.md",
            "data/table.bin",
            "reference/style.md",
            "scripts/run.sh"
        ]
    );
    assert_eq!(
        std::fs::read(dest.join("data/table.bin")).unwrap(),
        [0xff, 0x00, 0x01]
    );
    assert_eq!(
        std::fs::read_to_string(dest.join("scripts/run.sh")).unwrap(),
        "#!/bin/sh\necho hi"
    );
    // Nothing else is left beside it.
    assert_eq!(
        std::fs::read_dir(dest.parent().unwrap()).unwrap().count(),
        1
    );

    // Loading the folder gives the skill that was returned.
    let from_folder = read_candidate(&dest).unwrap();
    assert_eq!(from_folder.content, skill.content);
    assert_eq!(from_folder.resources, skill.resources);
    assert_eq!(from_folder.models, skill.models);
    assert_eq!(
        skill.resources.keys().collect::<Vec<_>>(),
        ["reference/style.md"]
    );

    // The bundle's executable bit is not kept.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for file in files_under(&dest) {
            let mode = std::fs::metadata(dest.join(&file))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o111, 0, "{file} is executable");
        }
    }
}

#[test]
fn unpacking_a_bundle_with_one_top_folder_writes_that_folder_only() {
    let temp = Temp::new();
    let bundle = zip_of(&[
        ("packed/SKILL.md", skill_md("packed", "").as_bytes()),
        ("packed/reference/a.md", b"A."),
        ("packed/scripts/run.sh", b"echo hi"),
        ("__MACOSX/packed/._SKILL.md", b"Not part of it."),
        ("notes.md", b"Beside it."),
    ]);
    let dest = temp.0.join("out");
    let skill = unpack_bundle(&bundle, &dest).unwrap();
    assert_eq!(
        files_under(&dest),
        ["SKILL.md", "reference/a.md", "scripts/run.sh"]
    );
    // The entries outside the folder aren't resources either, as `read_bundle` says.
    assert_eq!(
        skill.resources.keys().collect::<Vec<_>>(),
        ["reference/a.md"]
    );
    assert_eq!(
        read_bundle(&bundle)
            .unwrap()
            .resources
            .keys()
            .collect::<Vec<_>>(),
        ["reference/a.md"]
    );
}

#[test]
fn a_bundle_that_is_refused_leaves_nothing_behind() {
    let temp = Temp::new();
    let skill = skill_md("packed", "");
    let parent = temp.0.join("skills");
    let dest = parent.join("packed");
    let with = |name: &str| zip_of(&[("SKILL.md", skill.as_bytes()), (name, b"Text.")]);

    let mut refused: Vec<(&str, Vec<u8>)> = vec![
        ("a ..", with("../outside.md")),
        ("a nested ..", with("reference/../../outside.md")),
        ("an absolute path", with("/etc/outside.md")),
        ("a drive", with("C:/outside.md")),
        ("no SKILL.md", zip_of(&[("README.md", b"No skill.")])),
        (
            "a bad front matter",
            zip_of(&[("SKILL.md", b"No front matter.")]),
        ),
        (
            "text that isn't UTF-8",
            zip_of(&[
                ("SKILL.md", skill.as_bytes()),
                ("a.md", &[0xff, 0xfe, 0x00]),
            ]),
        ),
        ("not a zip", b"just text".to_vec()),
    ];
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer
        .start_file("SKILL.md", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(skill.as_bytes()).unwrap();
    writer
        .add_symlink("link", "/etc/passwd", SimpleFileOptions::default())
        .unwrap();
    refused.push(("a link", writer.finish().unwrap().into_inner()));
    // Too many files.
    let many: Vec<(String, &[u8])> = (0..1001)
        .map(|i| (format!("f{i}.txt"), &b"x"[..]))
        .collect();
    let many: Vec<(&str, &[u8])> = std::iter::once(("SKILL.md", skill.as_bytes()))
        .chain(many.iter().map(|(name, data)| (name.as_str(), *data)))
        .collect();
    refused.push(("too many files", zip_of(&many)));
    // Too much text.
    let zeros = vec![0u8; 17 * 1024 * 1024];
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer
        .start_file("SKILL.md", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(skill.as_bytes()).unwrap();
    writer
        .start_file(
            "big.dat",
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
        )
        .unwrap();
    writer.write_all(&zeros).unwrap();
    refused.push(("too much", writer.finish().unwrap().into_inner()));

    for (why, bundle) in &refused {
        assert!(unpack_bundle(bundle, &dest).is_err(), "{why}");
        // Not even the parent folder.
        assert!(!parent.exists(), "{why}");
        assert!(!temp.0.join("outside.md").exists(), "{why}");
    }
}

#[test]
fn unpacking_into_a_folder_that_exists_changes_nothing() {
    let temp = Temp::new();
    let dest = temp.0.join("packed");
    std::fs::create_dir(&dest).unwrap();
    std::fs::write(dest.join("old.md"), "Old.").unwrap();
    let bundle = zip_of(&[("SKILL.md", skill_md("packed", "").as_bytes())]);
    let error = unpack_bundle(&bundle, &dest).unwrap_err();
    assert!(
        matches!(&error, SkillLoadError::Io { source, .. }
            if source.kind() == std::io::ErrorKind::AlreadyExists),
        "{error}"
    );
    assert_eq!(files_under(&dest), ["old.md"]);
    assert_eq!(std::fs::read_dir(&temp.0).unwrap().count(), 1);
}

#[test]
fn a_failed_write_removes_what_it_wrote() {
    let temp = Temp::new();
    // A file and a folder with the same name can't both be written.
    let bundle = zip_of(&[
        ("SKILL.md", skill_md("packed", "").as_bytes()),
        ("data", b"A file."),
        ("data/table.bin", b"In a folder."),
    ]);
    let dest = temp.0.join("skills/packed");
    let error = unpack_bundle(&bundle, &dest).unwrap_err();
    assert!(matches!(error, SkillLoadError::Io { .. }), "{error}");
    assert!(!dest.exists());
    // The folder beside it holds no hidden leftover.
    assert_eq!(
        std::fs::read_dir(dest.parent().unwrap()).unwrap().count(),
        0
    );
}

// An archive whose local headers and central directory disagree.

/// Where `needle` is in `bytes`, the first time.
fn find(bytes: &[u8], needle: &[u8]) -> usize {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .unwrap()
}

/// A stored archive of a skill and a `reference/a.md`, and where the local
/// header of the second entry starts.
fn stored_bundle() -> (Vec<u8>, usize) {
    let stored = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer.start_file("SKILL.md", stored).unwrap();
    writer.write_all(skill_md("packed", "").as_bytes()).unwrap();
    writer.start_file("reference/a.md", stored).unwrap();
    writer.write_all(b"Reference text.").unwrap();
    let bundle = writer.finish().unwrap().into_inner();
    // The first local header starts at 0.
    let second = bundle[4..]
        .windows(4)
        .position(|window| window == b"PK\x03\x04")
        .map(|at| at + 4)
        .unwrap();
    (bundle, second)
}

#[test]
fn a_bundle_whose_local_header_names_an_entry_otherwise_is_refused() {
    let temp = Temp::new();
    let (mut bundle, _) = stored_bundle();
    // The local header says `reference/b.md`, the central directory `reference/a.md`.
    let at = find(&bundle, b"reference/a.md");
    bundle[at + "reference/".len()] = b'b';
    let error = read_bundle(&bundle).unwrap_err();
    assert!(
        matches!(&error, SkillLoadError::ZipError { message, .. }
            if message.contains("don't match the central directory")),
        "{error}"
    );
    let dest = temp.0.join("skills/packed");
    assert!(unpack_bundle(&bundle, &dest).is_err());
    assert!(!temp.0.join("skills").exists());
    // An unchanged one is fine.
    let (bundle, _) = stored_bundle();
    assert!(read_bundle(&bundle).is_ok());
    assert!(unpack_bundle(&bundle, &dest).is_ok());
}

#[test]
fn a_bundle_whose_local_header_sizes_an_entry_otherwise_is_refused() {
    let temp = Temp::new();
    let (mut bundle, second) = stored_bundle();
    // The size of the second entry in its local header, a byte more than the
    // central directory says.
    let size_at = second + 22;
    bundle[size_at] += 1;
    let error = read_bundle(&bundle).unwrap_err();
    assert!(matches!(error, SkillLoadError::ZipError { .. }), "{error}");
    let dest = temp.0.join("packed");
    assert!(unpack_bundle(&bundle, &dest).is_err());
    assert!(!dest.exists());
}

#[test]
fn a_bundle_in_a_source_folder_with_mismatched_headers_is_skipped() {
    let temp = Temp::new();
    let (good, _) = stored_bundle();
    let mut odd = good.clone();
    let at = find(&odd, b"reference/a.md");
    odd[at + "reference/".len()] = b'b';
    std::fs::write(temp.0.join("good.skill"), &good).unwrap();
    std::fs::write(temp.0.join("odd.skill"), &odd).unwrap();
    let config = SkillsConfig {
        global_sources: vec![global(&temp.0)],
        ..Default::default()
    };
    // The odd one is skipped with a reason, and the other loads.
    let scanned = scan(&config, &temp.0);
    assert_eq!(scanned.len(), 2);
    assert_eq!(scanned[0].skipped, None);
    assert!(
        scanned[1]
            .skipped
            .as_deref()
            .is_some_and(|why| why.contains("don't match the central directory")),
        "{:?}",
        scanned[1].skipped
    );
    assert_eq!(
        SkillManager::load(&config, &temp.0)
            .unwrap()
            .total_skill_count(),
        1
    );
    // Under Error it is refused, as a broken skill is.
    let strict = SkillsConfig {
        on_invalid_skill: InvalidSkillPolicy::Error,
        ..config
    };
    assert!(matches!(
        SkillManager::load(&strict, &temp.0),
        Err(ArkoAgentsError::Skill(SkillLoadError::ZipError { .. }))
    ));
}

#[test]
fn a_streamed_bundle_with_data_descriptors_loads() {
    let mut writer = zip::ZipWriter::new_stream(Vec::new());
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    writer.start_file("SKILL.md", options).unwrap();
    writer.write_all(skill_md("packed", "").as_bytes()).unwrap();
    writer.start_file("reference/a.md", options).unwrap();
    writer.write_all(b"Reference text.").unwrap();
    let bundle = writer.finish().unwrap().get_ref().clone();
    // The sizes follow the data, so the local headers hold none.
    assert_ne!(bundle[6] & 0x08, 0, "no data descriptor");
    let skill = read_bundle(&bundle).unwrap();
    assert_eq!(skill.resources["reference/a.md"], "Reference text.");
    let temp = Temp::new();
    std::fs::write(temp.0.join("streamed.skill"), &bundle).unwrap();
    let config = SkillsConfig {
        global_sources: vec![global(&temp.0)],
        ..Default::default()
    };
    assert_eq!(loaded_names(&config, &temp.0), ["packed"]);
}
