//! The model, the client, and the agents' configurations. Every agent (the
//! solvers of every arm, the grader, the writer and the verifier) runs on the
//! same model with the same settings.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arko_agents::{AgentConfig, AgentLLMConfig, AgentToolConfig, ToolCallingMode};
use arko_ai::{
    CodexAuthFile, GenerationConfig, LLMClient, Model, OpenAIEffort, OpenAIServiceTier,
    ReasoningConfig,
};
use serde::{Deserialize, Serialize};

use crate::BenchResult;

/// The model every agent runs on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    pub model_id: String,
    /// `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`.
    pub effort: String,
    /// Fast mode (`service_tier: priority`).
    pub fast: bool,
}

impl ModelSettings {
    /// Parses `id` or `id:effort` (default effort `xhigh`).
    pub fn parse(spec: &str, fast: bool) -> BenchResult<Self> {
        let (model_id, effort) = spec.split_once(':').unwrap_or((spec, "xhigh"));
        let settings = Self {
            model_id: model_id.trim().to_string(),
            effort: effort.trim().to_ascii_lowercase(),
            fast,
        };
        settings.model()?;
        settings.openai_effort()?;
        Ok(settings)
    }

    pub fn model(&self) -> BenchResult<Model> {
        let model = Model::find(&self.model_id)
            .ok_or_else(|| format!("unknown model {}", self.model_id))?;
        if model.provider != arko_ai::Provider::OpenAI {
            return Err(format!(
                "{} is not an OpenAI model. The benchmark runs on ChatGPT OAuth",
                self.model_id
            )
            .into());
        }
        Ok(model)
    }

    pub fn openai_effort(&self) -> BenchResult<OpenAIEffort> {
        Ok(match self.effort.as_str() {
            "none" => OpenAIEffort::None,
            "minimal" => OpenAIEffort::Minimal,
            "low" => OpenAIEffort::Low,
            "medium" => OpenAIEffort::Medium,
            "high" => OpenAIEffort::High,
            "xhigh" => OpenAIEffort::XHigh,
            "max" => OpenAIEffort::Max,
            other => return Err(format!("unknown effort {other}").into()),
        })
    }

    pub fn generation_config(&self) -> BenchResult<GenerationConfig> {
        let mut config = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(self.openai_effort()?, false))
            .with_prompt_caching(true);
        if self.fast {
            config = config.with_openai_service_tier(OpenAIServiceTier::Fast);
        }
        Ok(config)
    }

    /// The API list price of a call's tokens for this model and tier, in
    /// dollars: what the benchmark reports as cost. ChatGPT OAuth bills
    /// nothing per call; this is the API-equivalent cost.
    pub fn api_cost(&self, input: usize, cached: usize, output: usize) -> f64 {
        let Ok(model) = self.model() else {
            return 0.0;
        };
        let pricing = &model.pricing;
        let tier = if self.fast {
            arko_ai::pricing::ServiceTier::Fast
        } else {
            arko_ai::pricing::ServiceTier::Standard
        };
        let multiplier = pricing.tier_multiplier(Some(&tier), false);
        let cached = cached.min(input);
        let cached_price = pricing
            .cached_input_cost_per_million
            .unwrap_or(pricing.input_cost_per_million);
        ((input - cached) as f64 * pricing.input_cost_per_million
            + cached as f64 * cached_price
            + output as f64 * pricing.output_cost_per_million)
            / 1_000_000.0
            * multiplier
    }
}

/// The Codex home whose `auth.json` holds the ChatGPT login: `--codex-home`,
/// else `$CODEX_HOME`, else `~/.codex`.
pub fn codex_auth_path(codex_home: Option<&Path>) -> BenchResult<PathBuf> {
    if let Some(home) = codex_home {
        return Ok(home.join("auth.json"));
    }
    CodexAuthFile::default_path()
        .ok_or_else(|| "no home directory to find ~/.codex/auth.json. Pass --codex-home".into())
}

/// The client for every agent: ChatGPT OAuth from a Codex `auth.json`
/// (tokens are refreshed in step with the file), or the mock model.
///
/// Several `--codex-home`s become a list of ChatGPT accounts with failover,
/// first preferred (`LLMClient::with_codex_auth_files`): a request an
/// account's usage limit refuses goes to the next account, and requests move
/// back once the first is usable again.
pub fn build_client(
    codex_homes: &[PathBuf],
    mock_url: Option<&str>,
) -> BenchResult<Arc<LLMClient>> {
    if let Some(url) = mock_url {
        return Ok(Arc::new(LLMClient::new().with_openai_client(
            arko_ai::providers::openai::OpenAIClient::new("mock-key").with_base_url(url),
        )));
    }
    let load = |home: Option<&Path>| -> BenchResult<(String, CodexAuthFile)> {
        let path = codex_auth_path(home)?;
        let auth = CodexAuthFile::load(&path).map_err(|error| {
            format!(
                "the ChatGPT login at {} could not be read ({error}). Run `codex login` (with CODEX_HOME set to the Codex home you pass) first",
                path.display()
            )
        })?;
        Ok((account_label(&path), auth))
    };
    if codex_homes.len() <= 1 {
        let (_, auth) = load(codex_homes.first().map(PathBuf::as_path))?;
        return Ok(Arc::new(LLMClient::new().with_codex_auth_file(&auth)?));
    }
    let mut accounts = Vec::new();
    let mut labels = std::collections::HashSet::new();
    for (position, home) in codex_homes.iter().enumerate() {
        let (label, auth) = load(Some(home))?;
        // Labels name the accounts in failover events, so keep them distinct.
        let label = if labels.insert(label.clone()) {
            label
        } else {
            format!("{label}-{}", position + 1)
        };
        accounts.push((label, auth));
    }
    Ok(Arc::new(LLMClient::new().with_codex_auth_files(
        accounts.iter().map(|(label, auth)| (label.as_str(), auth)),
    )?))
}

/// An account's label: the name of its Codex home (`~/.codex-200` is
/// `codex-200`).
fn account_label(auth_path: &Path) -> String {
    auth_path
        .parent()
        .and_then(|dir| dir.file_name())
        .map(|name| name.to_string_lossy().trim_start_matches('.').to_string())
        .unwrap_or_else(|| auth_path.display().to_string())
}

/// An agent with the given role and instructions on `model`, running every
/// tool call without approval, in native tool calling.
pub fn agent_config(
    id: &str,
    name: &str,
    role: &str,
    instructions: &str,
    summary_prompt: &str,
    model: &ModelSettings,
    tool_timeout_seconds: u64,
) -> BenchResult<AgentConfig> {
    let llm_config = AgentLLMConfig {
        model: model.model()?,
        generation_config: model.generation_config()?,
    };
    let tool_config = AgentToolConfig {
        default_auto_approve_tool_use: true,
        tool_calling_mode: ToolCallingMode::Native,
        tool_timeout_seconds: Some(tool_timeout_seconds),
        ..AgentToolConfig::default()
    };
    Ok(
        AgentConfig::builder(id.to_string(), name.to_string(), llm_config)
            .role_description(role.trim().to_string())
            .instructions(instructions.trim().to_string())
            .tool_config(tool_config)
            .complete_task_summary_prompt_override(summary_prompt.to_string())
            .build()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_specs() {
        let settings = ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap();
        assert_eq!(settings.model_id, "gpt-6-astra");
        assert_eq!(settings.openai_effort().unwrap(), OpenAIEffort::XHigh);
        assert!(ModelSettings::parse("gpt-6-astra:extreme", true).is_err());
        assert!(ModelSettings::parse("no-such-model", true).is_err());
        assert!(ModelSettings::parse("claude-opus-5-5", true).is_err());
    }

    #[test]
    fn api_cost_applies_cache_and_fast_prices() {
        let fast = ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap();
        let standard = ModelSettings::parse("gpt-6-astra:xhigh", false).unwrap();
        // 1M input of which 0.5M cached, 0.1M output.
        let base = standard.api_cost(1_000_000, 500_000, 100_000);
        let pricing = standard.model().unwrap().pricing;
        let expected = (500_000.0 * pricing.input_cost_per_million
            + 500_000.0 * pricing.cached_input_cost_per_million.unwrap()
            + 100_000.0 * pricing.output_cost_per_million)
            / 1_000_000.0;
        assert!((base - expected).abs() < 1e-9);
        assert!(fast.api_cost(1_000_000, 500_000, 100_000) > base);
    }
}
