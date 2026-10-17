//! How much input a model accepts on the auth path in use.
use crate::model::{Model, Provider};

/// The input limit, in tokens, of the ChatGPT OAuth (Codex) backend: the
/// `context_window` its model catalog serves for GPT-5.x and GPT-6 models.
/// OpenAI does not document it. The API accepts more (922,000 input tokens
/// for GPT-5.6 and GPT-6).
pub const CHATGPT_OAUTH_MAX_INPUT_TOKENS: usize = 272_000;

/// The most input tokens `model` accepts on the given auth path:
/// [`Model::max_input_tokens`], capped at [`CHATGPT_OAUTH_MAX_INPUT_TOKENS`]
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
            Some(limit.min(CHATGPT_OAUTH_MAX_INPUT_TOKENS))
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
        // The flag is about OpenAI requests; other providers keep theirs.
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
        // No OpenAI client configured: the registry limit.
        assert_eq!(
            LLMClient::new().max_input_tokens(&luna).await,
            Some(922_000)
        );
    }
}
