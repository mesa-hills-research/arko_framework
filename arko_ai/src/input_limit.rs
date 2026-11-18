//! How much input a model accepts on the auth path in use.
use crate::model::{Model, Provider};

/// The input limit, in tokens, of the ChatGPT OAuth (Codex) backend for most
/// models. Taken from the `context_window` its model catalog serves for
/// GPT-5.x and GPT-6 models. OpenAI does not document it. The API accepts more
/// (922,000 input tokens for GPT-5.6 and GPT-6). See
/// [`chatgpt_oauth_max_input_tokens`] for the models with more.
pub const CHATGPT_OAUTH_MAX_INPUT_TOKENS: usize = 272_000;

/// The ChatGPT OAuth input limit of the models whose Codex catalog entry has a
/// `context_window` of 500,000, which are GPT-6.2 Sol, GPT-6.1 Sol and
/// GPT-6.1 Terra (`codex-rs/models-manager/models.json` at openai/codex
/// 6c0cee50, which also raised GPT-6.1 Sol from 272,000). OpenAI's Codex docs
/// give the same default context for GPT-6.2 Sol and GPT-6.1 Sol.
pub const CHATGPT_OAUTH_LONG_MAX_INPUT_TOKENS: usize = 500_000;

/// The input limit of the ChatGPT OAuth backend for the OpenAI model `id`,
/// [`CHATGPT_OAUTH_LONG_MAX_INPUT_TOKENS`] for GPT-6.2 Sol, GPT-6.1 Sol and
/// GPT-6.1 Terra, else [`CHATGPT_OAUTH_MAX_INPUT_TOKENS`].
pub fn chatgpt_oauth_max_input_tokens(id: &str) -> usize {
    match id {
        "gpt-6.2-sol" | "gpt-6.1-sol" | "gpt-6.1-terra" => CHATGPT_OAUTH_LONG_MAX_INPUT_TOKENS,
        _ => CHATGPT_OAUTH_MAX_INPUT_TOKENS,
    }
}

/// The most input tokens `model` accepts on the given auth path. Returns
/// [`Model::max_input_tokens`], capped at [`chatgpt_oauth_max_input_tokens`]
/// for OpenAI models when `openai_chatgpt_oauth` is set. `None` when the
/// model's limit is unknown (`max_input_tokens` of 0, as for
/// [`Model::openrouter`] models built without the catalog).
///
/// [`LLMClient::max_input_tokens`](crate::client::LLMClient::max_input_tokens)
/// reads the auth path from the client.
pub fn effective_max_input_tokens(model: &Model, openai_chatgpt_oauth: bool) -> Option<usize> {
    match model.max_input_tokens {
        0 => None,
        limit if model.provider == Provider::OpenAI && openai_chatgpt_oauth => {
            Some(limit.min(chatgpt_oauth_max_input_tokens(&model.id)))
        }
        limit => Some(limit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LLMClient;

    #[test]
    fn chatgpt_oauth_caps_openai_models_only() {
        let luna = Model::openai_gpt_5_6_luna();
        assert_eq!(effective_max_input_tokens(&luna, false), Some(922_000));
        assert_eq!(effective_max_input_tokens(&luna, true), Some(272_000));
        let gpt_6 = Model::openai_gpt_6_luna();
        assert_eq!(effective_max_input_tokens(&gpt_6, true), Some(272_000));
        // A limit under the backend's stays as it is.
        let small = Model {
            max_input_tokens: 128_000,
            ..Model::openai_gpt_5_5()
        };
        assert_eq!(effective_max_input_tokens(&small, true), Some(128_000));
        // The flag is about OpenAI requests, and other providers keep theirs.
        let opus = Model::anthropic_claude_opus_5_5();
        assert_eq!(
            effective_max_input_tokens(&opus, true),
            Some(opus.max_input_tokens)
        );
        assert_eq!(
            effective_max_input_tokens(&Model::openrouter("openai/gpt-5.6-luna"), false),
            None
        );
    }

    /// Each OpenAI model's limit over ChatGPT OAuth follows the
    /// `context_window` of its Codex catalog entry
    /// (`codex-rs/models-manager/models.json` at openai/codex 6c0cee50).
    #[test]
    fn chatgpt_oauth_limits_follow_codex_catalog_context_windows() {
        for (id, limit) in [
            ("gpt-6.2-sol", 500_000),
            ("gpt-6.1-sol", 500_000),
            ("gpt-6.1-terra", 500_000),
            ("gpt-6.2-astra", 272_000),
            ("gpt-6-astra", 272_000),
            ("gpt-6-sol", 272_000),
            ("gpt-6-luna", 272_000),
            ("gpt-5.6-sol", 272_000),
            ("gpt-5.5-2026-04-23", 272_000),
        ] {
            let model = Model::find(id).unwrap_or_else(|| panic!("{id}"));
            assert_eq!(
                effective_max_input_tokens(&model, true),
                Some(limit),
                "{id}"
            );
            assert_eq!(chatgpt_oauth_max_input_tokens(id), limit, "{id}");
            assert_eq!(
                effective_max_input_tokens(&model, false),
                Some(model.max_input_tokens),
                "{id}"
            );
        }
    }

    #[tokio::test]
    async fn the_client_supplies_its_openai_auth_path() {
        let luna = Model::openai_gpt_5_6_luna();
        let api_key = LLMClient::new().with_openai_api_key("key");
        assert_eq!(api_key.max_input_tokens(&luna).await, Some(922_000));
        let oauth =
            LLMClient::new().with_openai_chatgpt_oauth("access", Some("refresh"), "account");
        assert_eq!(oauth.max_input_tokens(&luna).await, Some(272_000));
        let opus = Model::anthropic_claude_opus_5_5();
        assert_eq!(
            oauth.max_input_tokens(&opus).await,
            Some(opus.max_input_tokens)
        );
        // No OpenAI client configured, so the registry limit applies.
        assert_eq!(
            LLMClient::new().max_input_tokens(&luna).await,
            Some(922_000)
        );
    }
}
