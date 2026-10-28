//! Chat Completions request building. Every optional field is sent only when
//! arko has a reason to send it, so OpenRouter and the serving provider apply
//! their own defaults otherwise.

use super::{
    OpenRouterClient,
    catalog::OpenRouterModelInfo,
    options::{
        AutoExactoPolicy, OpenRouterOptions, OpenRouterServiceTier, ProviderPreferences, SortBy,
    },
};
use crate::{
    config::{AnthropicCacheTtl, GenerationConfig, Verbosity},
    conversation::Conversation,
    error::Error,
    message::{Content, Image, Message, Role},
    model::{Model, Provider},
    reasoning::{
        OpenAIReasoningMode, OpenRouterReasoning, ReasoningArtifact, ReasoningConfig,
        ReasoningEffort,
    },
    tool::{ToolChoice, ToolConfig},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::borrow::Cow;

/// Anthropic beta that makes OpenRouter keep `strict` on Claude tools
/// (without it `strict` is stripped).
pub(crate) const STRUCTURED_OUTPUTS_BETA: &str = "structured-outputs-2025-11-13";
pub(crate) const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// OpenRouter's limit for `session_id`.
pub(crate) const MAX_SESSION_ID_CHARS: usize = 256;

/// A request body plus the headers it needs.
#[derive(Debug, Clone)]
pub(crate) struct PreparedRequest {
    pub(crate) body: ChatRequest,
    /// Values for the `x-anthropic-beta` header.
    pub(crate) anthropic_betas: Vec<&'static str>,
    /// The model with the best metadata available (its own, else the loaded
    /// catalog's), used for pricing and limits.
    pub(crate) model: Model,
}

/// `POST /api/v1/chat/completions` body. Deprecated or removed fields
/// (`usage`, `stream_options`, `transforms`) are never sent: usage is always
/// returned.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<OpenRouterReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) verbosity: Option<Verbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    pub(crate) session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) plugins: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<OpenRouterServiceTier>,
}

/// The model of a combined `model@preset/slug` reference
/// (`openai/gpt-6-luna@preset/email-copywriter` → `openai/gpt-6-luna`).
/// Other slugs, including a bare `@preset/slug`, are returned unchanged.
/// https://openrouter.ai/docs/guides/features/presets
pub(crate) fn without_preset(slug: &str) -> &str {
    match slug.split_once("@preset/") {
        Some((model, _)) if !model.is_empty() => model,
        _ => slug,
    }
}

/// The slug without a preset or variant suffix (`openai/gpt-6-luna:floor` →
/// `openai/gpt-6-luna`).
pub(crate) fn base_slug(slug: &str) -> &str {
    let slug = without_preset(slug);
    slug.split_once(':').map_or(slug, |(base, _)| base)
}

/// The author namespace (`anthropic` for `~anthropic/claude-sonnet-latest`).
pub(crate) fn vendor(slug: &str) -> Option<&str> {
    slug.trim_start_matches('~')
        .split_once('/')
        .map(|(vendor, _)| vendor)
}

/// Claude models, which need explicit `cache_control` to be cached.
pub(crate) fn is_claude_family(slug: &str) -> bool {
    vendor(slug) == Some("anthropic")
}

/// The Auto Router plugin id for this slug. Each router slug reads only its
/// own id. Settings under the other id are silently ignored.
pub(crate) fn auto_router_plugin_id(slug: &str) -> Option<&'static str> {
    match base_slug(slug) {
        "openrouter/auto" => Some("auto-router"),
        "openrouter/auto-beta" => Some("auto-beta-router"),
        _ => None,
    }
}

/// OpenAI pro variants (`openai/gpt-6-sol-pro`), which reject effort updates.
fn is_openai_pro_slug(slug: &str) -> bool {
    vendor(slug) == Some("openai") && base_slug(slug).ends_with("-pro")
}

/// Variants that are themselves a provider sort (`:exacto` sorts by quality,
/// `:nitro` by throughput, `:floor` by price).
fn is_sort_variant(slug: &str) -> bool {
    matches!(
        without_preset(slug)
            .rsplit_once(':')
            .map(|(_, variant)| variant),
        Some("exacto" | "nitro" | "floor")
    )
}

fn invalid(message: String) -> Error {
    Error::InvalidRequest(message)
}

/// Interleaved thinking on Claude: OpenRouter sends the beta that turns it
/// on, but has no way to turn it off. (On other models the setting is
/// dropped with the other Anthropic ones, see `without_foreign_settings`.)
fn check_interleaved_thinking(model: &Model, config: &GenerationConfig) -> Result<(), Error> {
    match config.interleaved_thinking {
        Some(false) if is_claude_family(&model.id) => Err(Error::UnsupportedFeature(format!(
            "OpenRouter can't disable interleaved thinking for '{}'. Leave GenerationConfig::interleaved_thinking unset.",
            model.id
        ))),
        _ => Ok(()),
    }
}

fn image_part(image: &Image) -> Value {
    let url = match image {
        Image::Url(url) => url.clone(),
        Image::Base64 { data, mime_type } => format!("data:{mime_type};base64,{data}"),
    };
    json!({"type": "image_url", "image_url": {"url": url}})
}

/// User content: a plain string for a single text part, otherwise parts.
fn user_content(contents: &[&Content]) -> Value {
    if let [Content::Text(text)] = contents {
        return Value::String(text.clone());
    }
    Value::Array(
        contents
            .iter()
            .filter_map(|content| match content {
                Content::Text(text) => Some(json!({"type": "text", "text": text})),
                Content::Image(image) => Some(image_part(image)),
                Content::ToolResult(_) => None,
            })
            .collect(),
    )
}

fn text_only(message: &Message, role: &str) -> Result<String, Error> {
    let mut text = String::new();
    for content in &message.content {
        match content {
            Content::Text(part) => text.push_str(part),
            _ => {
                return Err(invalid(format!(
                    "OpenRouter {role} messages carry only text. This one has images or tool results."
                )));
            }
        }
    }
    Ok(text)
}

fn assistant_message(message: &Message, raw_invalid_arguments: bool) -> Result<Value, Error> {
    let text = text_only(message, "assistant")?;
    let mut value = json!({"role": "assistant"});
    value["content"] = if text.is_empty() && !message.tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(text)
    };
    if !message.tool_calls.is_empty() {
        value["tool_calls"] = message
            .tool_calls
            .iter()
            .map(|call| {
                // Invalid arguments are replayed exactly as the model wrote
                // them to OpenAI models; other vendors need a JSON object
                // (Claude `tool_use.input`), so they get `{}` as with the
                // direct providers, and the tool result explains the error.
                let arguments = match &call.invalid_arguments {
                    Some(raw) if raw_invalid_arguments => raw.clone(),
                    _ => serde_json::to_string(&call.replay_arguments())
                        .unwrap_or_else(|_| "{}".into()),
                };
                json!({
                    "id": call.id.clone().unwrap_or_default(),
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                })
            })
            .collect();
    }
    match &message.reasoning_artifact {
        // Replayed verbatim and in order: Claude signatures and Gemini thought
        // signatures must match what the model produced.
        Some(ReasoningArtifact::OpenRouter { details }) if !details.is_empty() => {
            value["reasoning_details"] = Value::Array(details.clone());
        }
        Some(ReasoningArtifact::OpenRouter { .. }) | None => {}
        Some(other) => log::debug!(
            "arko_ai: dropping a {:?} reasoning artifact. OpenRouter replays only its own reasoning_details",
            other.provider()
        ),
    }
    Ok(value)
}

fn convert_messages(conversation: &Conversation) -> Result<Vec<Value>, Error> {
    let raw_invalid_arguments = vendor(&conversation.model.id) == Some("openai");
    let mut messages = Vec::new();
    if let Some(system) = &conversation.system_message {
        messages.push(json!({"role": "system", "content": system}));
    }
    let mut previous_was_update = false;
    for (position, message) in conversation.messages.iter().enumerate() {
        if let Some(update) = &message.configuration_update {
            if previous_was_update {
                return Err(invalid(
                    "Two configuration updates are adjacent. OpenRouter rejects adjacent updates. Keep only the latest.".into(),
                ));
            }
            previous_was_update = true;
            // An update belongs before a user turn; between a tool call and
            // its result it would break the call/result pairing.
            let before_tool_results = conversation.messages.get(position + 1).is_some_and(|next| {
                next.content
                    .iter()
                    .any(|content| content.as_tool_result().is_some())
            });
            if before_tool_results {
                return Err(invalid(
                    "A configuration update sits between tool calls and their results. Add it after the tool results, before the next user turn.".into(),
                ));
            }
            // The Chat Completions form: a content-less system message.
            messages.push(json!({
                "role": "system",
                "content": "",
                "configuration_update": {"reasoning": {"effort": update.reasoning_effort}},
            }));
            continue;
        }
        previous_was_update = false;
        match message.role {
            Role::System => {
                messages.push(json!({"role": "system", "content": text_only(message, "system")?}));
            }
            Role::Assistant => messages.push(assistant_message(message, raw_invalid_arguments)?),
            Role::User => {
                // Tool results become `tool` messages, followed by any other
                // content of the same turn as a user message.
                for content in &message.content {
                    if let Content::ToolResult(result) = content {
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": result.call_id.clone().unwrap_or_default(),
                            "content": result.output,
                        }));
                    }
                }
                // Result images follow the results (OpenRouter doesn't
                // document how each upstream takes images inside a tool
                // message).
                let rest = super::super::tool_images::content_after_results(message, false);
                let rest: Vec<&Content> = rest.iter().map(AsRef::as_ref).collect();
                let had_results = message
                    .content
                    .iter()
                    .any(|content| content.as_tool_result().is_some());
                if !rest.is_empty() || !had_results {
                    messages.push(json!({"role": "user", "content": user_content(&rest)}));
                }
            }
        }
    }
    Ok(messages)
}

fn tool_definitions(tool_config: &ToolConfig) -> Vec<Value> {
    tool_config
        .tools
        .iter()
        .map(|tool| {
            let mut function = json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            });
            if let Some(strict) = tool.strict {
                function["strict"] = Value::Bool(strict);
            }
            json!({"type": "function", "function": function})
        })
        .collect()
}

fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::None => json!("none"),
        ToolChoice::Specific(name) => json!({"type": "function", "function": {"name": name}}),
    }
}

/// Checks a parameter against the catalog's `supported_parameters`, so a
/// parameter the model doesn't take fails here instead of being dropped by
/// OpenRouter (`require_parameters` is off by default).
fn ensure_parameter(
    info: Option<&OpenRouterModelInfo>,
    model: &str,
    parameter: &str,
    any_of: &[&str],
) -> Result<(), Error> {
    match info {
        Some(info) if !any_of.iter().any(|name| info.supports_parameter(name)) => {
            Err(invalid(format!(
                "OpenRouter lists no `{parameter}` support for model '{model}', so it would be dropped silently."
            )))
        }
        _ => Ok(()),
    }
}

fn validate_reasoning(
    reasoning: &OpenRouterReasoning,
    info: Option<&OpenRouterModelInfo>,
    model: &str,
) -> Result<(), Error> {
    if reasoning.effort.is_some() && reasoning.max_tokens.is_some() {
        return Err(invalid(
            "OpenRouter reasoning takes either `effort` or `max_tokens`, not both.".into(),
        ));
    }
    ensure_parameter(info, model, "reasoning", &["reasoning"])?;
    let mandatory = info
        .and_then(|info| info.reasoning.as_ref())
        .is_some_and(|reasoning| reasoning.mandatory);
    if mandatory
        && (reasoning.effort == Some(ReasoningEffort::None) || reasoning.enabled == Some(false))
    {
        return Err(invalid(format!(
            "Reasoning is mandatory for model '{model}'. It rejects `effort: none` and `enabled: false`."
        )));
    }
    Ok(())
}

impl OpenRouterClient {
    /// Builds the request for `conversation`, validating it against the
    /// model's catalog metadata when available (the model's own, or the
    /// client's loaded catalog).
    pub(crate) fn build_request(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
    ) -> Result<PreparedRequest, Error> {
        let requested = &conversation.model;
        if requested.provider != Provider::OpenRouter {
            return Err(Error::ProviderNotSupported(format!(
                "Model '{}' is not an OpenRouter model.",
                requested.id
            )));
        }
        super::super::ensure_supported_inputs(conversation)?;
        let sanitized = super::super::without_foreign_settings(requested, config);
        let config: &GenerationConfig = &sanitized;
        check_interleaved_thinking(requested, config)?;
        let slug = requested.id.as_str();

        // The model's own catalog metadata, else the loaded catalog's.
        let catalog = self.catalog();
        if let Some(catalog) = &catalog
            && !slug.starts_with('@')
            && !catalog.contains(slug)
        {
            return Err(Error::ModelNotSupported(format!(
                "OpenRouter model '{slug}' is not in the loaded catalog ({:?}). Check the slug, or clear the catalog to send it unchecked.",
                catalog.source()
            )));
        }
        let from_catalog = match (&requested.openrouter, &catalog) {
            (None, Some(catalog)) => catalog.get(slug),
            _ => None,
        };
        let model: Cow<Model> = from_catalog.map_or(Cow::Borrowed(requested), Cow::Owned);
        let info = model.openrouter.as_ref();

        let has_images = conversation.messages.iter().any(Message::has_images);
        if has_images && !model.supports_image_input {
            return Err(Error::UnsupportedFeature(format!(
                "OpenRouter model '{slug}' does not accept image input."
            )));
        }

        let options = config.openrouter.as_ref().map_or_else(
            || self.default_options.clone(),
            |options| options.merged_over(&self.default_options),
        );
        let claude = is_claude_family(slug);
        let mut anthropic_betas = Vec::new();
        if config.interleaved_thinking == Some(true) {
            anthropic_betas.push(INTERLEAVED_THINKING_BETA);
        }

        let reasoning = match &config.reasoning {
            None => None,
            Some(ReasoningConfig::OpenRouter(reasoning)) => {
                validate_reasoning(reasoning, info, slug)?;
                // Claude needs room for the answer after the thinking budget.
                if let (true, Some(budget), Some(max_tokens)) =
                    (claude, reasoning.max_tokens, config.max_tokens)
                    && max_tokens <= budget
                {
                    return Err(invalid(format!(
                        "On Claude, max_tokens ({max_tokens}) must exceed the reasoning budget ({budget})."
                    )));
                }
                Some(reasoning.clone())
            }
            // Another provider's variant was dropped above.
            Some(_) => None,
        };

        if let Some(verbosity) = config.verbosity {
            ensure_parameter(info, slug, "verbosity", &["verbosity"])?;
            // On Claude, OpenRouter sends verbosity as output_config.effort,
            // which would override the reasoning effort.
            if claude && reasoning.as_ref().is_some_and(|r| r.effort.is_some()) {
                return Err(invalid(format!(
                    "On Claude, OpenRouter maps verbosity ({verbosity:?}) to output_config.effort, overriding reasoning.effort. Set only one."
                )));
            }
        }

        let temperature = config.temperature;
        if temperature.is_some() {
            ensure_parameter(info, slug, "temperature", &["temperature"])?;
        }
        if config.max_tokens.is_some() {
            ensure_parameter(
                info,
                slug,
                "max_tokens",
                &["max_tokens", "max_completion_tokens"],
            )?;
        }

        let (tools, tool_choice, parallel_tool_calls) = match config.offered_tools() {
            Some(tool_config) => {
                ensure_parameter(info, slug, "tools", &["tools"])?;
                if tool_config.tool_choice.is_some() {
                    ensure_parameter(info, slug, "tool_choice", &["tool_choice"])?;
                }
                // As the Anthropic provider does for the same Claude models.
                if matches!(
                    tool_config.tool_choice,
                    Some(ToolChoice::Required | ToolChoice::Specific(_))
                ) && !model.supports_forced_tool_choice(config)
                {
                    return Err(invalid(format!(
                        "Forced tool use is not supported for model '{slug}' with this configuration (a Claude model that rejects it, or manual thinking with a reasoning budget). Use Auto or None tool choice."
                    )));
                }
                if claude
                    && tool_config
                        .tools
                        .iter()
                        .any(|tool| tool.strict == Some(true))
                {
                    anthropic_betas.push(STRUCTURED_OUTPUTS_BETA);
                }
                (
                    Some(tool_definitions(tool_config)),
                    tool_config.tool_choice.as_ref().map(tool_choice),
                    tool_config.parallel_tool_calls,
                )
            }
            None => (None, None, None),
        };

        if conversation
            .messages
            .iter()
            .any(Message::is_configuration_update)
        {
            let pro_mode =
                reasoning.as_ref().and_then(|r| r.mode) == Some(OpenAIReasoningMode::Pro);
            if pro_mode || is_openai_pro_slug(slug) {
                return Err(invalid(format!(
                    "Mid-conversation effort updates can't be combined with pro reasoning mode or a -pro model ('{slug}')."
                )));
            }
            // "Claude accepts low, medium, high, xhigh, and max (minimal is
            // sent as low, none is rejected)."
            let to_none = conversation.messages.iter().any(|message| {
                message
                    .configuration_update
                    .as_ref()
                    .is_some_and(|update| update.reasoning_effort == ReasoningEffort::None)
            });
            if claude && to_none {
                return Err(invalid(format!(
                    "Claude models reject a mid-conversation effort update to `none` ('{slug}'). Use `minimal` (sent as `low`) or higher."
                )));
            }
        }
        let messages = convert_messages(conversation)?;

        let session_id = conversation.prompt_cache_key.clone();
        if session_id.chars().count() > MAX_SESSION_ID_CHARS {
            return Err(invalid(format!(
                "Conversation::prompt_cache_key is sent as OpenRouter's session_id, which is limited to {MAX_SESSION_ID_CHARS} characters."
            )));
        }

        let caching =
            config.prompt_caching.unwrap_or(conversation.enable_cache) && model.supports_cache;
        let cache_control = if caching && claude {
            let mut cache_control = json!({"type": "ephemeral"});
            if let Some(ttl) = config.claude_cache_ttl() {
                cache_control["ttl"] = json!(match ttl {
                    AnthropicCacheTtl::FiveMinutes => "5m",
                    AnthropicCacheTtl::OneHour => "1h",
                });
            }
            Some(cache_control)
        } else if config.anthropic_cache_ttl.is_some() {
            return Err(invalid(format!(
                "A cache TTL applies only to Claude models with prompt caching enabled, not '{slug}'."
            )));
        } else {
            None
        };
        let paid_cache_writes = match info {
            Some(_) => model
                .pricing
                .cache_write_cost_per_million
                .is_some_and(|price| price > 0.0),
            None => matches!(vendor(slug), Some("anthropic" | "openai")),
        };
        // With caching off, OpenAI models that bill cache writes (GPT-5.6 and
        // later) still write cache entries unless explicit mode leaves the
        // request without breakpoints. The catalog says which models bill
        // writes; without one, the registry's model of the same id does.
        let takes_cache_options = match info {
            Some(_) => paid_cache_writes,
            None => base_slug(slug)
                .strip_prefix("openai/")
                .and_then(|id| Model::get(Provider::OpenAI, id).ok())
                .is_some_and(|model| model.supports_openai_prompt_cache_options()),
        };
        let prompt_cache_options =
            (!caching && vendor(slug) == Some("openai") && takes_cache_options)
                .then(|| json!({"mode": "explicit"}));

        let provider =
            self.routing_preferences(&options, slug, tools.is_some(), caching, paid_cache_writes);
        let plugins = auto_router_plugin_id(slug)
            .and_then(|id| {
                options
                    .auto_router
                    .as_ref()
                    .and_then(|router| router.plugin(id))
            })
            .map(|plugin| vec![plugin]);

        Ok(PreparedRequest {
            body: ChatRequest {
                model: slug.to_string(),
                messages,
                stream: None,
                max_tokens: config.max_tokens,
                temperature,
                reasoning,
                verbosity: config.verbosity,
                tools,
                tool_choice,
                parallel_tool_calls,
                session_id,
                prompt_cache_key: caching.then(|| conversation.prompt_cache_key.clone()),
                prompt_cache_options,
                cache_control,
                provider,
                models: options.fallback_models.clone(),
                plugins,
                user: options.user.clone(),
                service_tier: options.service_tier,
            },
            anthropic_betas,
            model: model.into_owned(),
        })
    }

    /// The `provider` object: the merged preferences, Auto Router overrides
    /// for router slugs, and the Auto Exacto policy for tool requests.
    fn routing_preferences(
        &self,
        options: &OpenRouterOptions,
        slug: &str,
        has_tools: bool,
        caching: bool,
        paid_cache_writes: bool,
    ) -> Option<ProviderPreferences> {
        let mut provider = options.provider.clone().unwrap_or_default();
        if auto_router_plugin_id(slug).is_some()
            && let Some(overrides) = options
                .auto_router
                .as_ref()
                .and_then(|router| router.provider.as_ref())
        {
            provider = overrides.merged_over(&provider);
        }
        // An explicit sort, order or sorting variant already decides routing
        // (and turns Auto Exacto off, or in `order`'s case also sticky
        // routing).
        let explicit = provider.sort.is_some() || provider.order.is_some() || is_sort_variant(slug);
        let sort_by_price = has_tools
            && !explicit
            && match options.auto_exacto.unwrap_or_default() {
                AutoExactoPolicy::AllowAutoExacto => false,
                AutoExactoPolicy::DisableAutoExacto => true,
                AutoExactoPolicy::CacheAware => caching && paid_cache_writes,
            };
        if sort_by_price {
            provider.sort = Some(SortBy::Price.into());
        }
        (!provider.is_empty()).then_some(provider)
    }
}
