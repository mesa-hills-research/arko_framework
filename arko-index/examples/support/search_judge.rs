use super::{AnyResult, OAuthFile, clients};
use arko_ai::{
    GenerationConfig, LLMClient, Model, OpenAIEffort, Provider, ReasoningConfig, model::ModelType,
};
use arko_index::{JudgeOptions, LlmSearchJudge};
use clap::{Args, ValueEnum};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum JudgeProvider {
    Openai,
    Anthropic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum JudgeAuth {
    Inherit,
    ApiKey,
    Oauth,
}

#[derive(Args)]
pub struct JudgeArgs {
    /// Opt in to LLM filtering/reranking after retrieval; may return fewer results.
    #[arg(long)]
    pub judge: bool,
    #[arg(long, value_enum, default_value = "openai", requires = "judge")]
    judge_provider: JudgeProvider,
    /// Defaults to gpt-5.6-luna for OpenAI; required for other providers.
    #[arg(long, requires = "judge")]
    judge_model: Option<String>,
    /// OpenAI reasoning effort. Defaults to high unless --judge-config supplies it.
    #[arg(long, requires = "judge", value_parser = ["none", "minimal", "low", "medium", "high", "xhigh", "max"])]
    judge_effort: Option<String>,
    /// Number of retrieval candidates to judge before applying --top-k.
    #[arg(long, default_value_t = 20, requires = "judge")]
    judge_candidates: usize,
    /// Attempts for invalid tool selections; provider errors are returned directly.
    #[arg(long, default_value_t = 2, requires = "judge")]
    judge_attempts: usize,
    /// Inherit the global OpenAI client, use a provider API key, or use an OAuth file.
    #[arg(long, value_enum, default_value = "inherit", requires = "judge")]
    judge_auth: JudgeAuth,
    /// API-key environment variable name (default OPENAI_API_KEY / ANTHROPIC_API_KEY).
    #[arg(long, requires = "judge")]
    judge_api_key_env: Option<String>,
    /// Judge-specific OAuth file; requires --judge-auth oauth. Defaults to --oauth-file.
    #[arg(long, requires = "judge")]
    judge_oauth_file: Option<PathBuf>,
    /// JSON file with GenerationConfig fields; omitted fields keep defaults.
    #[arg(long, requires = "judge")]
    judge_config: Option<PathBuf>,
    /// Append task-specific selection preferences from a UTF-8 text file.
    #[arg(long, requires = "judge")]
    judge_instructions_file: Option<PathBuf>,
}

pub struct JudgeSession {
    pub judge: LlmSearchJudge,
    client: LLMClient,
    oauth: Option<OAuthFile>,
}
impl JudgeSession {
    pub async fn persist(&mut self) -> AnyResult<()> {
        if let Some(oauth) = &mut self.oauth {
            oauth.persist(&self.client).await?;
        }
        Ok(())
    }
}

fn generation_config(overrides: Option<Value>) -> AnyResult<GenerationConfig> {
    let mut value = serde_json::to_value(GenerationConfig::default())?;
    if let Some(overrides) = overrides {
        let fields = overrides
            .as_object()
            .ok_or("judge config must be a JSON object")?;
        for (key, field) in fields {
            if key == "tool_config" {
                return Err("judge owns tool_config; configure selection preferences with --judge-instructions-file".into());
            }
            if value.get(key).is_none() {
                return Err(format!("unknown judge configuration field: {key}").into());
            }
            value[key] = field.clone();
        }
        if fields.contains_key("temperature") && !fields.contains_key("temperature_specified") {
            value["temperature_specified"] = Value::Bool(true);
        }
    }
    Ok(serde_json::from_value(value)?)
}

impl JudgeArgs {
    pub fn options(&self) -> JudgeOptions {
        JudgeOptions {
            candidate_limit: self.judge_candidates,
            max_attempts: self.judge_attempts,
        }
    }

    fn model_and_config(&self) -> AnyResult<(Model, GenerationConfig)> {
        let provider = match self.judge_provider {
            JudgeProvider::Openai => Provider::OpenAI,
            JudgeProvider::Anthropic => Provider::Anthropic,
        };
        let id = match self.judge_model.as_deref() {
            Some(id) => id,
            None if provider == Provider::OpenAI => "gpt-5.6-luna",
            None => return Err("--judge-model is required for this provider".into()),
        };
        let model = Model::get(provider.clone(), id)?;
        let overrides = self
            .judge_config
            .as_ref()
            .map(|path| -> AnyResult<Value> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) })
            .transpose()?;
        let reasoning_specified = overrides
            .as_ref()
            .is_some_and(|v| v.get("reasoning").is_some());
        let mut config = generation_config(overrides)?;
        if let Some(effort) = &self.judge_effort {
            if provider != Provider::OpenAI || model.model_type != ModelType::Reasoning {
                return Err("--judge-effort requires an OpenAI reasoning model; use --judge-config for provider-specific settings".into());
            }
            config.reasoning = Some(ReasoningConfig::openai(
                serde_json::from_value(Value::String(effort.clone()))?,
                false,
            ));
        } else if !reasoning_specified
            && provider == Provider::OpenAI
            && model.model_type == ModelType::Reasoning
        {
            config.reasoning = Some(ReasoningConfig::openai(OpenAIEffort::High, false));
        }
        if config
            .reasoning
            .as_ref()
            .is_some_and(|r| r.provider() != provider)
        {
            return Err("judge reasoning configuration does not match the model provider".into());
        }
        Ok((model, config))
    }

    pub fn session(
        &self,
        inherited: &LLMClient,
        inherited_path: Option<&Path>,
    ) -> AnyResult<JudgeSession> {
        let (model, generation) = self.model_and_config()?;
        if self.judge_api_key_env.is_some() && self.judge_auth != JudgeAuth::ApiKey {
            return Err("--judge-api-key-env requires --judge-auth api-key".into());
        }
        if self.judge_oauth_file.is_some() && self.judge_auth != JudgeAuth::Oauth {
            return Err("--judge-oauth-file requires --judge-auth oauth".into());
        }
        let (client, oauth) = match self.judge_auth {
            JudgeAuth::Inherit => {
                if model.provider != Provider::OpenAI {
                    return Err("the global client uses OpenAI authentication; select --judge-auth api-key for another provider".into());
                }
                (inherited.clone(), None)
            }
            JudgeAuth::ApiKey => {
                let default_env = if model.provider == Provider::OpenAI {
                    "OPENAI_API_KEY"
                } else {
                    "ANTHROPIC_API_KEY"
                };
                let name = self.judge_api_key_env.as_deref().unwrap_or(default_env);
                let key = std::env::var(name)
                    .map_err(|_| format!("missing judge API-key environment variable: {name}"))?;
                if key.trim().is_empty() {
                    return Err(format!("empty judge API-key environment variable: {name}").into());
                }
                let client = if model.provider == Provider::OpenAI {
                    LLMClient::new().with_openai_api_key(&key)
                } else {
                    LLMClient::new().with_anthropic_api_key(&key)
                };
                (client, None)
            }
            JudgeAuth::Oauth => {
                if model.provider != Provider::OpenAI {
                    return Err("OAuth judge authentication requires OpenAI".into());
                }
                let path = self
                    .judge_oauth_file
                    .as_deref()
                    .or(inherited_path)
                    .ok_or("--judge-auth oauth requires --judge-oauth-file or --oauth-file")?;
                let same_file = inherited_path
                    .map(|global| -> std::io::Result<bool> {
                        Ok(std::fs::canonicalize(global)? == std::fs::canonicalize(path)?)
                    })
                    .transpose()?
                    .unwrap_or(false);
                if same_file {
                    (inherited.clone(), None)
                } else {
                    clients(Some(path.to_owned()))?
                }
            }
        };
        let instructions = self
            .judge_instructions_file
            .as_ref()
            .map(std::fs::read_to_string)
            .transpose()?
            .unwrap_or_default();
        let judge = LlmSearchJudge::new(client.clone(), model)
            .with_generation_config(generation)
            .with_instructions(instructions);
        Ok(JudgeSession {
            judge,
            client,
            oauth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    fn parse(extra: &[&str]) -> AnyResult<JudgeArgs> {
        let args = super::super::Args::try_parse_from(
            ["cli", "--database", "test.db", "search", "id", "query"]
                .into_iter()
                .chain(extra.iter().copied()),
        )?;
        match args.command {
            super::super::Command::Search { judging, .. } => Ok(judging),
            _ => unreachable!(),
        }
    }

    #[test]
    fn judging_is_opt_in_and_effort_is_configurable() {
        assert!(!parse(&[]).unwrap().judge);
        assert!(parse(&["--judge-model", "gpt-5.6-luna"]).is_err());
        let args = parse(&["--judge"]).unwrap();
        let (model, config) = args.model_and_config().unwrap();
        assert_eq!(model.id, "gpt-5.6-luna");
        assert!(matches!(
            config.reasoning,
            Some(ReasoningConfig::OpenAI {
                effort: OpenAIEffort::High,
                ..
            })
        ));
        let args = parse(&["--judge", "--judge-effort", "max"]).unwrap();
        assert!(matches!(
            args.model_and_config().unwrap().1.reasoning,
            Some(ReasoningConfig::OpenAI {
                effort: OpenAIEffort::Max,
                ..
            })
        ));
        assert!(parse(&["--lexical", "--semantic"]).is_err());
    }

    #[test]
    fn config_merges_defaults_and_rejects_mistakes() {
        let config = generation_config(Some(json!({"temperature":0.2,"max_tokens":4096}))).unwrap();
        assert!(config.temperature_specified);
        assert_eq!(config.max_tokens, Some(4096));
        assert!(generation_config(Some(json!({"temprature":0.2}))).is_err());
        assert!(generation_config(Some(json!({"tool_config":null}))).is_err());
        assert!(generation_config(Some(json!([]))).is_err());
    }

    #[test]
    fn auth_configuration_does_not_silently_fall_back() {
        for flags in [
            vec!["--judge", "--judge-auth", "oauth"],
            vec!["--judge", "--judge-api-key-env", "TEST_KEY"],
            vec!["--judge", "--judge-provider", "anthropic"],
        ] {
            assert!(
                parse(&flags)
                    .unwrap()
                    .session(&LLMClient::new(), None)
                    .is_err()
            );
        }
        assert!(
            parse(&["--judge"])
                .unwrap()
                .session(&LLMClient::new(), None)
                .is_ok()
        );
    }
}
