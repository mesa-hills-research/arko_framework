use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCallUsage {
    pub call_id: String,
    pub provider: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_usage_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_delta_from_previous_call: Option<isize>,
    /// Provider's response termination reason, such as Anthropic's `refusal`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Provider-supplied termination details, when available. No response text is added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<serde_json::Value>,
}

impl ModelCallUsage {
    pub fn new(
        call_id: String,
        provider: String,
        model: String,
        input_tokens: usize,
        output_tokens: usize,
        max_input_tokens: usize,
    ) -> Self {
        let context_usage_percent = if max_input_tokens > 0 {
            Some(input_tokens as f64 / max_input_tokens as f64 * 100.0)
        } else {
            None
        };

        Self {
            call_id,
            provider,
            model,
            input_tokens,
            output_tokens,
            max_input_tokens: Some(max_input_tokens),
            context_usage_percent,
            input_delta_from_previous_call: None,
            stop_reason: None,
            stop_details: None,
        }
    }

    pub fn with_input_delta(mut self, delta: isize) -> Self {
        self.input_delta_from_previous_call = Some(delta);
        self
    }

    pub fn with_stop_metadata(
        mut self,
        reason: Option<String>,
        details: Option<serde_json::Value>,
    ) -> Self {
        self.stop_reason = reason;
        self.stop_details = details.filter(|value| !value.is_null());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextEventType {
    System,
    UserMessage,
    AssistantMessage,
    AssistantToolCall,
    ToolResult,
    ToolDefinition,
    FileAttachment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextEvent {
    pub call_id: String,
    pub event_type: ContextEventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub char_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,
}

impl ContextEvent {
    pub fn new(call_id: String, event_type: ContextEventType) -> Self {
        Self {
            call_id,
            event_type,
            char_count: None,
            byte_count: None,
            token_count: None,
        }
    }

    pub fn for_tool_call(call_id: &str, serialized: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            event_type: ContextEventType::AssistantToolCall,
            char_count: Some(serialized.len()),
            byte_count: Some(serialized.len()),
            token_count: None,
        }
    }

    pub fn for_tool_result(call_id: &str, serialized: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            event_type: ContextEventType::ToolResult,
            char_count: Some(serialized.len()),
            byte_count: Some(serialized.len()),
            token_count: None,
        }
    }

    pub fn for_text(call_id: &str, event_type: ContextEventType, text: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            event_type,
            char_count: Some(text.len()),
            byte_count: Some(text.len()),
            token_count: None,
        }
    }

    pub fn with_token_count(mut self, token_count: usize) -> Self {
        self.token_count = Some(token_count);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageInfo {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub input_cost: f64,
    pub output_cost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_cost: Option<f64>,
    pub total_cost: f64,
}

impl UsageInfo {
    pub fn new(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens,
            input_cost: 0.0,
            output_cost: 0.0,
            cached_tokens: None,
            cache_read_cost: None,
            cache_write_tokens: None,
            cache_write_cost: None,
            total_cost: 0.0,
        }
    }

    pub fn with_cached_tokens(mut self, cached_tokens: usize) -> Self {
        self.cached_tokens = Some(cached_tokens);
        self
    }

    pub fn with_cache_write_tokens(mut self, cache_write_tokens: usize) -> Self {
        self.cache_write_tokens = Some(cache_write_tokens);
        self
    }

    pub fn calculate_cost(
        &mut self,
        model: &crate::model::Model,
        config: Option<&crate::config::GenerationConfig>,
    ) {
        self.cache_read_cost = None;
        self.cache_write_cost = None;

        let use_large_context_pricing = if model.provider == crate::model::Provider::Anthropic {
            let total_input = self.input_tokens
                + self.cached_tokens.unwrap_or(0)
                + self.cache_write_tokens.unwrap_or(0);

            config.and_then(|c| c.long_context).unwrap_or(false)
                && model.pricing.large_input_threshold.is_some()
                && total_input > model.pricing.large_input_threshold.unwrap()
        } else {
            model.pricing.large_input_threshold.is_some()
                && self.input_tokens > model.pricing.large_input_threshold.unwrap()
        };

        let non_cached_input_tokens = match model.provider {
            crate::model::Provider::Anthropic => self.input_tokens,
            crate::model::Provider::OpenAI => self
                .input_tokens
                .saturating_sub(self.cached_tokens.unwrap_or(0))
                .saturating_sub(self.cache_write_tokens.unwrap_or(0)),
            crate::model::Provider::Google => self
                .input_tokens
                .saturating_sub(self.cached_tokens.unwrap_or(0)),
        };

        let input_rate = if use_large_context_pricing {
            model
                .pricing
                .large_input_cost_per_million
                .unwrap_or(model.pricing.input_cost_per_million)
        } else {
            model.pricing.input_cost_per_million
        };

        self.input_cost = (non_cached_input_tokens as f64 / 1_000_000.0) * input_rate;

        if let Some(cached_tokens) = self.cached_tokens
            && let Some(default_rate) = model.pricing.cached_input_cost_per_million
        {
            let rate = if use_large_context_pricing {
                model
                    .pricing
                    .large_cached_input_cost_per_million
                    .unwrap_or(default_rate)
            } else {
                default_rate
            };

            self.cache_read_cost =
                Some((cached_tokens as f64 / 1_000_000.0) * rate);
        }

        if let Some(cache_write_tokens) = self.cache_write_tokens
            && let Some(default_rate) = model.pricing.cache_write_cost_per_million
        {
            let rate = if use_large_context_pricing {
                model
                    .pricing
                    .large_cache_write_cost_per_million
                    .unwrap_or(default_rate)
            } else {
                default_rate
            };

            self.cache_write_cost =
                Some((cache_write_tokens as f64 / 1_000_000.0) * rate);
        }

        let output_rate = if use_large_context_pricing {
            model
                .pricing
                .large_output_cost_per_million
                .unwrap_or(model.pricing.output_cost_per_million)
        } else {
            model.pricing.output_cost_per_million
        };

        self.output_cost = (self.output_tokens as f64 / 1_000_000.0) * output_rate;

        self.total_cost = self.input_cost
            + self.output_cost
            + self.cache_read_cost.unwrap_or(0.0)
            + self.cache_write_cost.unwrap_or(0.0);
    }

    pub fn for_embedding(input_tokens: usize, model_cost_per_million_input_tokens: f64) -> Self {
        let input_cost =
            (input_tokens as f64 / 1_000_000.0) * model_cost_per_million_input_tokens;

        Self {
            input_tokens,
            output_tokens: 0,
            input_cost,
            output_cost: 0.0,
            cached_tokens: None,
            cache_read_cost: None,
            cache_write_tokens: None,
            cache_write_cost: None,
            total_cost: input_cost,
        }
    }
}
