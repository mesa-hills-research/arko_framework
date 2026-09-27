//! The agent configurations in `examples/` load and make sense.

use arko_agents::parsing::PromptBuilder;
use arko_agents::{AgentConfig, Task, ToolRegistry};
use std::path::Path;

#[test]
fn every_example_agent_configuration_loads_validates_and_builds_its_prompt() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let registry = ToolRegistry::new();
    let placeholder = regex::Regex::new(r"\{\{?[#/]?[A-Za-z_]+\}\}?").unwrap();
    let mut loaded = 0;
    for entry in std::fs::read_dir(&examples).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let name = path.display();
        let config = AgentConfig::load_from_path(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        config.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
        // A model the registry serves for its provider, not a saved copy.
        let model = &config.llm_config.model;
        arko_ai::Model::get(model.provider.clone(), &model.id)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        // Settings the model accepts: Claude models that always think
        // adaptively reject sampling settings, manual thinking budgets and
        // disabled thinking.
        let generation = &config.llm_config.generation_config;
        if model.omits_anthropic_sampling_parameters() {
            assert!(!generation.temperature_specified, "{name}");
        }
        if model.rejects_anthropic_manual_thinking() {
            assert!(
                !matches!(
                    generation.reasoning,
                    Some(arko_ai::ReasoningConfig::Anthropic { .. })
                ),
                "{name}"
            );
        }
        if model.always_uses_anthropic_adaptive_thinking() {
            assert_ne!(generation.anthropic_thinking_enabled, Some(false), "{name}");
        }
        // Every tool it names is built in.
        let definitions: Vec<_> = config
            .tool_config
            .available_tool_names
            .iter()
            .map(|tool| {
                registry
                    .get(tool)
                    .unwrap_or_else(|| panic!("{name}: no built-in tool '{tool}'"))
                    .definition()
            })
            .collect();
        // The system prompt substitutes everything in the instructions.
        let task = Task::new(
            config.clone(),
            "Add a README".into(),
            examples.clone(),
            None,
        );
        let parsing = config.get_tool_parsing_config();
        let prompt = PromptBuilder::new(&config, definitions.iter().collect(), &parsing, None)
            .build_system_prompt(&task)
            .unwrap();
        assert!(!placeholder.is_match(&prompt), "{name}: {prompt}");
        loaded += 1;
    }
    assert!(loaded >= 2, "found {loaded} example configurations");
}
