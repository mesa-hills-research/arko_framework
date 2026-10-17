use serde::{Deserialize, Serialize};

/// Usage metrics of one model call.
///
/// What the model fields hold, per provider:
///
/// | Provider | `model` | `served_model` | `requested_model` | `rerouted` |
/// |---|---|---|---|---|
/// | OpenAI | the response's `model` | `openai-model` header, else the response's `model`, else `response.created`'s | never set | a reported model differs from the requested one |
/// | OpenRouter | the model OpenRouter reported serving, else the requested slug | the model OpenRouter reported serving | the requested slug, when a different model served the call | never set |
/// | Anthropic | the response's `model` | never set | never set | never set |
/// | Google | the requested model id | never set | never set | never set |
///
/// [`served_model_changed`](Self::served_model_changed) is the
/// provider-independent test for a call served by another model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCallUsage {
    pub call_id: String,
    pub provider: String,
    /// The model that answered, as the provider names it. See the table
    /// above.
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
    /// The model that was requested, when the provider served a different one
    /// (OpenRouter model fallbacks, routers such as `openrouter/auto`, and
    /// `~…-latest` aliases). `model` is the one that served the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model: Option<String>,
    /// The upstream provider that served a routed call, when reported (for
    /// example OpenRouter's `provider`, such as `Anthropic` or `Google Vertex`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_provider: Option<String>,
    /// The model the provider reports having served, when it reports one.
    /// For OpenAI: the `openai-model` (or `x-openai-model`) response header,
    /// else the response's `model`, else the model in `response.created`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    /// Whether any model the provider reported for the call differs from
    /// the requested one (ignoring case and snapshot date suffixes), for
    /// example when the ChatGPT backend routes a request to a fallback model.
    /// Set by the OpenAI provider. OpenRouter reports a change through
    /// `requested_model`. [`served_model_changed`](Self::served_model_changed)
    /// covers both.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rerouted: bool,
    /// Subscription plan usage the provider reported with the call (the
    /// ChatGPT backend's `x-codex-*` headers on ChatGPT OAuth requests).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_usage: Option<PlanUsage>,
    /// The label of the ChatGPT account that served the call, when the
    /// OpenAI provider has several
    /// ([`ChatGptAccounts`](crate::providers::openai::ChatGptAccounts)).
    /// Never a token or account id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    /// The provider refused the reasoning summaries the request asked for,
    /// and the request was sent again without them: the OpenAI provider on
    /// a 400 saying the organization must be verified to generate reasoning
    /// summaries. The provider's client asks for none from then on.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reasoning_summaries_refused: bool,
    /// The subagent whose call this was, when the model runs its own
    /// subagents (Claude Code's `Agent` tool): the id of the tool call that
    /// started it. `None` for a call of the conversation itself, the one
    /// whose size and prompt cache a host shows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<String>,
}

/// How much of a subscription plan's usage limits is used, as the ChatGPT
/// backend reports it on every ChatGPT OAuth response (`x-codex-*` headers,
/// which Codex shows as its rate-limit status) and the Claude Code CLI in
/// its `rate_limit_event`s. Requests fail with a usage limit error
/// ([`RetryClass::UsageLimit`](crate::error::RetryClass::UsageLimit)) once a
/// window is used up, until it resets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanUsage {
    /// The subscription the report is of, set by the component that reads
    /// it. Records saved before it existed have none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<SubscriptionPlan>,
    /// The account's plan (`x-codex-plan-type`, such as `plus` or `pro`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_type: Option<String>,
    /// The limit currently applied (`x-codex-active-limit`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_limit: Option<String>,
    /// The primary window (`x-codex-primary-*`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<PlanUsageWindow>,
    /// The secondary window (`x-codex-secondary-*`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary: Option<PlanUsageWindow>,
    /// The account's credits (`x-codex-credits-*`), when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits: Option<PlanCredits>,
}

/// A ChatGPT account's usage credits, as the backend reports them in the
/// `x-codex-credits-*` response headers and on its usage endpoint (Codex's
/// `CreditsSnapshot`). Codex sends nothing to opt in to them, and counts an
/// account whose credits are left (or unlimited) as usable once its plan's
/// windows are used up.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanCredits {
    /// Credits are left (`x-codex-credits-has-credits`).
    pub has_credits: bool,
    /// The credits are unlimited (`x-codex-credits-unlimited`).
    #[serde(default)]
    pub unlimited: bool,
    /// The balance, as the backend writes it (`x-codex-credits-balance`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balance: Option<String>,
}

impl PlanCredits {
    /// Whether requests can go on when the plan's windows are used up:
    /// credits are left, or unlimited.
    pub fn usable(&self) -> bool {
        self.has_credits || self.unlimited
    }
}

/// The subscription a [`PlanUsage`] report is of. Each has its own usage
/// windows, with their own levels and reset times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SubscriptionPlan {
    /// A ChatGPT plan, as a ChatGPT OAuth login reports it (`x-codex-*`
    /// headers).
    #[serde(rename = "chatgpt")]
    ChatGpt,
    /// A Claude plan, as the Claude Code CLI's login reports it
    /// (`rate_limit_event`).
    #[serde(rename = "claude_code")]
    ClaudeCode,
}

impl SubscriptionPlan {
    pub const ALL: [SubscriptionPlan; 2] =
        [SubscriptionPlan::ChatGpt, SubscriptionPlan::ClaudeCode];

    /// `"chatgpt"` or `"claude_code"`.
    pub fn key(self) -> &'static str {
        match self {
            SubscriptionPlan::ChatGpt => "chatgpt",
            SubscriptionPlan::ClaudeCode => "claude_code",
        }
    }

    /// `"ChatGPT"` or `"Claude Code"`.
    pub fn label(self) -> &'static str {
        match self {
            SubscriptionPlan::ChatGpt => "ChatGPT",
            SubscriptionPlan::ClaudeCode => "Claude Code",
        }
    }
}

/// One usage-limit window of a [`PlanUsage`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanUsageWindow {
    /// Percentage of the window's limit used (`-used-percent`).
    pub used_percent: f64,
    /// The window's length in minutes (`-window-minutes`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_minutes: Option<u64>,
    /// When the window resets, in Unix seconds (`-reset-at`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resets_at: Option<u64>,
    /// Seconds until the window resets, as of the response
    /// (`-reset-after-seconds`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_after_seconds: Option<u64>,
}

impl PlanUsage {
    /// Reads the ChatGPT backend's `x-codex-*` plan usage headers, as a
    /// report of [`SubscriptionPlan::ChatGpt`]. `None` when the response
    /// has none.
    pub fn from_codex_headers(headers: &reqwest::header::HeaderMap) -> Option<Self> {
        let text = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
        };
        let number = |name: &str| {
            text(name)
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|n| n.is_finite())
        };
        let whole = |name: &str| number(name).filter(|n| *n >= 0.0).map(|n| n as u64);
        let window = |prefix: &str| {
            let used_percent = number(&format!("x-codex-{prefix}-used-percent"))?;
            let window = PlanUsageWindow {
                used_percent,
                window_minutes: whole(&format!("x-codex-{prefix}-window-minutes"))
                    .filter(|m| *m > 0),
                resets_at: whole(&format!("x-codex-{prefix}-reset-at")).filter(|t| *t > 0),
                reset_after_seconds: whole(&format!("x-codex-{prefix}-reset-after-seconds")),
            };
            // A window with nothing used, no length and no reset time is a
            // placeholder (the backend sends zeros for an unused secondary).
            (used_percent != 0.0 || window.window_minutes.is_some() || window.resets_at.is_some())
                .then_some(window)
        };
        let flag = |name: &str| match text(name)?.to_ascii_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        };
        // As Codex reads them (`parse_credits_snapshot` in
        // codex-rs/codex-api/src/rate_limits.rs): both flags must be there.
        let credits = flag("x-codex-credits-has-credits").and_then(|has_credits| {
            Some(PlanCredits {
                has_credits,
                unlimited: flag("x-codex-credits-unlimited")?,
                balance: text("x-codex-credits-balance").map(String::from),
            })
        });
        let usage = Self {
            plan: None,
            plan_type: text("x-codex-plan-type").map(String::from),
            active_limit: text("x-codex-active-limit").map(String::from),
            primary: window("primary"),
            secondary: window("secondary"),
            credits,
        };
        (usage != Self::default()).then_some(Self {
            plan: Some(SubscriptionPlan::ChatGpt),
            ..usage
        })
    }

    /// The window with the highest used percentage, named `"primary"` or
    /// `"secondary"`.
    pub fn most_used_window(&self) -> Option<(&'static str, &PlanUsageWindow)> {
        [
            ("primary", self.primary.as_ref()),
            ("secondary", self.secondary.as_ref()),
        ]
        .into_iter()
        .filter_map(|(name, window)| Some((name, window?)))
        .max_by(|a, b| a.1.used_percent.total_cmp(&b.1.used_percent))
    }
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
            requested_model: None,
            upstream_provider: None,
            served_model: None,
            rerouted: false,
            plan_usage: None,
            account: None,
            reasoning_summaries_refused: false,
            subagent: None,
        }
    }

    pub fn with_plan_usage(mut self, plan_usage: Option<PlanUsage>) -> Self {
        self.plan_usage = plan_usage;
        self
    }

    /// Whether the provider served a different model than the one requested:
    /// OpenRouter set `requested_model`, or OpenAI reported a reroute.
    pub fn served_model_changed(&self) -> bool {
        self.requested_model.is_some() || self.rerouted
    }

    pub fn with_served_model(mut self, served_model: Option<String>, rerouted: bool) -> Self {
        self.served_model = served_model;
        self.rerouted = rerouted;
        self
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

/// How a call is paid for, which decides [`UsageInfo::billed_cost`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BillingKind {
    /// Charged per token at list prices: the billed cost is the estimate.
    #[default]
    Metered,
    /// Covered by a subscription (ChatGPT OAuth): nothing is billed per call.
    /// The estimate shows what the call would cost at API list prices.
    Subscription,
    /// The provider reported the actual charge in [`UsageInfo::reported_cost`].
    ProviderReported,
    /// The provider does not charge for this call, although it reports its
    /// tokens (an Anthropic classifier refusal before any output in an
    /// unbilled category). The estimate shows what it would have cost.
    NotCharged,
    /// An aggregate of calls billed in different ways.
    Mixed,
}

/// Token usage and cost of one model call, or a sum of calls.
///
/// Token counts mean the same thing for every provider:
/// - `input_tokens` is the TOTAL input, including cache reads
///   (`cached_tokens`) and cache writes (`cache_write_tokens`). Usage records
///   saved before this definition stored only uncached input for Anthropic.
/// - `output_tokens` includes reasoning/thinking tokens (`reasoning_tokens`).
///
/// Costs are USD estimates at list prices, for the tier, residency and date
/// that applied. `total_cost` is their sum. `billed_cost` is what the account
/// is actually charged (see [`BillingKind`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageInfo {
    /// Total input tokens, including cache reads and writes.
    pub input_tokens: usize,
    /// Output tokens, including reasoning tokens.
    pub output_tokens: usize,
    /// Estimated cost of uncached, unwritten input.
    pub input_cost: f64,
    /// Estimated cost of output tokens.
    pub output_cost: f64,
    /// Input tokens read from the prompt cache (part of `input_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_cost: Option<f64>,
    /// Input tokens written to the prompt cache (part of `input_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<usize>,
    /// The part of `cache_write_tokens` written with a 1-hour TTL (Anthropic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_cost: Option<f64>,
    /// Reasoning/thinking tokens, when the provider reports them (part of
    /// `output_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<usize>,
    /// Processing tier used for pricing: the provider-reported tier when the
    /// response includes one, else the requested tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<crate::pricing::ServiceTier>,
    /// Data-residency region the provider reported (Anthropic `inference_geo`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Sum of the estimated costs above.
    pub total_cost: f64,
    /// Actual charge reported by the provider (OpenRouter `usage.cost`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_cost: Option<f64>,
    /// With bring-your-own-key routing, what the upstream provider charged
    /// your own provider account (OpenRouter
    /// `usage.cost_details.upstream_inference_cost`). `reported_cost` is then
    /// only OpenRouter's fee. Included in `billed_cost`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_cost: Option<f64>,
    #[serde(default)]
    pub billing: BillingKind,
    /// What the account was charged. Records saved before billing kinds
    /// existed have none. See [`UsageInfo::billed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billed_cost: Option<f64>,
}

impl UsageInfo {
    pub fn new(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens,
            ..Self::default()
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

    /// What the account was charged: the billed cost, or for records saved
    /// before billing kinds existed, `total_cost` (which was already zero for
    /// subscription calls then).
    pub fn billed(&self) -> f64 {
        self.billed_cost.unwrap_or(self.total_cost)
    }

    /// Estimates cost at list prices in effect now. See
    /// [`UsageInfo::calculate_cost_at`].
    pub fn calculate_cost(
        &mut self,
        model: &crate::model::Model,
        config: Option<&crate::config::GenerationConfig>,
    ) {
        self.calculate_cost_at(model, config, std::time::SystemTime::now());
    }

    /// Estimates cost at the list prices in effect at `at`, for the tier and
    /// residency that applied, and sets `billed_cost` from the billing kind.
    ///
    /// Rules:
    /// - Long-context rates apply when total input exceeds the model's
    ///   threshold. On Claude Sonnet 4 and 4.5, whose long context needed a
    ///   (now retired) beta, they apply only when `long_context` was requested.
    /// - The tier is the reported `service_tier`, else the requested OpenAI
    ///   tier, else standard. Unpriced tiers use standard rates.
    /// - Tier and residency multipliers stack and apply to every token class.
    /// - Token classes without their own price are billed at the input rate.
    pub fn calculate_cost_at(
        &mut self,
        model: &crate::model::Model,
        config: Option<&crate::config::GenerationConfig>,
        at: std::time::SystemTime,
    ) {
        let pricing = model.pricing.at(at);
        let base = &model.pricing;

        let long_context = pricing.large_input_threshold.is_some_and(|threshold| {
            self.input_tokens > threshold
                && (!model.had_retired_anthropic_long_context_beta()
                    || config.and_then(|c| c.long_context).unwrap_or(false))
        });
        let pick = |large: Option<f64>, standard: Option<f64>| {
            if long_context {
                large.or(standard)
            } else {
                standard
            }
        };

        if self.service_tier.is_none() && model.provider == crate::model::Provider::OpenAI {
            self.service_tier = config
                .and_then(|c| c.openai_service_tier)
                .map(|tier| tier.pricing_tier());
        }
        let tiers = if pricing.service_tiers.is_empty() {
            base
        } else {
            pricing
        };
        let residency = if pricing.residency.is_empty() {
            base
        } else {
            pricing
        };
        let multiplier = tiers.tier_multiplier(self.service_tier.as_ref(), long_context)
            * residency.residency_multiplier(self.inference_geo.as_deref());

        let input_rate = pick(
            pricing.large_input_cost_per_million,
            Some(pricing.input_cost_per_million),
        )
        .unwrap_or(pricing.input_cost_per_million);
        let rate = |price: Option<f64>| price.unwrap_or(input_rate) * multiplier;
        let cost = |tokens: usize, rate: f64| tokens as f64 / 1_000_000.0 * rate;

        let cached = self.cached_tokens.unwrap_or(0);
        let written = self.cache_write_tokens.unwrap_or(0);
        let written_1h = self.cache_write_1h_tokens.unwrap_or(0).min(written);
        let uncached = self
            .input_tokens
            .saturating_sub(cached)
            .saturating_sub(written);

        self.input_cost = cost(uncached, input_rate * multiplier);
        self.cache_read_cost = self.cached_tokens.map(|tokens| {
            cost(
                tokens,
                rate(pick(
                    pricing.large_cached_input_cost_per_million,
                    pricing.cached_input_cost_per_million,
                )),
            )
        });
        self.cache_write_cost = self.cache_write_tokens.map(|_| {
            cost(
                written - written_1h,
                rate(pick(
                    pricing.large_cache_write_cost_per_million,
                    pricing.cache_write_cost_per_million,
                )),
            ) + cost(
                written_1h,
                rate(pick(
                    pricing.large_cache_write_1h_cost_per_million,
                    pricing.cache_write_1h_cost_per_million,
                )),
            )
        });
        let output_rate = pick(
            pricing.large_output_cost_per_million,
            Some(pricing.output_cost_per_million),
        )
        .unwrap_or(pricing.output_cost_per_million);
        self.output_cost = cost(self.output_tokens, output_rate * multiplier);

        self.total_cost = self.input_cost
            + self.output_cost
            + self.cache_read_cost.unwrap_or(0.0)
            + self.cache_write_cost.unwrap_or(0.0);
        self.update_billed_cost();
    }

    /// Records a provider-reported charge. It takes precedence over the
    /// estimate for billing.
    pub fn with_reported_cost(mut self, cost: f64) -> Self {
        self.reported_cost = Some(cost);
        self.billing = BillingKind::ProviderReported;
        self.update_billed_cost();
        self
    }

    /// Marks the call as covered by a subscription (nothing billed per call).
    pub fn with_billing(mut self, billing: BillingKind) -> Self {
        self.billing = billing;
        self.update_billed_cost();
        self
    }

    fn update_billed_cost(&mut self) {
        self.billed_cost = match self.billing {
            BillingKind::Metered => Some(self.total_cost),
            BillingKind::Subscription | BillingKind::NotCharged => Some(0.0),
            BillingKind::ProviderReported => Some(
                self.reported_cost.unwrap_or(self.total_cost) + self.upstream_cost.unwrap_or(0.0),
            ),
            BillingKind::Mixed => self.billed_cost,
        };
    }

    pub fn for_embedding(input_tokens: usize, model_cost_per_million_input_tokens: f64) -> Self {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * model_cost_per_million_input_tokens;

        Self {
            input_tokens,
            input_cost,
            total_cost: input_cost,
            billed_cost: Some(input_cost),
            ..Self::default()
        }
    }
}

fn add_options<T: std::ops::Add<Output = T>>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    }
}

/// Token counts add saturating: absurd provider figures must not panic
/// (debug builds check overflow), for example inside a caller's lock.
fn add_counts(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (a, b) => a.or(b),
    }
}

impl std::ops::AddAssign<&UsageInfo> for UsageInfo {
    /// Adds another call's usage. Tier, region and billing kind are kept when
    /// both sides agree. Otherwise the tier and region become unknown and the
    /// billing kind becomes [`BillingKind::Mixed`].
    fn add_assign(&mut self, other: &UsageInfo) {
        let billed = self.billed() + other.billed();
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.input_cost += other.input_cost;
        self.output_cost += other.output_cost;
        self.cached_tokens = add_counts(self.cached_tokens, other.cached_tokens);
        self.cache_read_cost = add_options(self.cache_read_cost, other.cache_read_cost);
        self.cache_write_tokens = add_counts(self.cache_write_tokens, other.cache_write_tokens);
        self.cache_write_1h_tokens =
            add_counts(self.cache_write_1h_tokens, other.cache_write_1h_tokens);
        self.cache_write_cost = add_options(self.cache_write_cost, other.cache_write_cost);
        self.reasoning_tokens = add_counts(self.reasoning_tokens, other.reasoning_tokens);
        if self.service_tier != other.service_tier {
            self.service_tier = None;
        }
        if self.inference_geo != other.inference_geo {
            self.inference_geo = None;
        }
        self.total_cost += other.total_cost;
        self.reported_cost = add_options(self.reported_cost, other.reported_cost);
        self.upstream_cost = add_options(self.upstream_cost, other.upstream_cost);
        if self.billing != other.billing {
            self.billing = BillingKind::Mixed;
        }
        self.billed_cost = Some(billed);
    }
}

impl std::ops::AddAssign for UsageInfo {
    fn add_assign(&mut self, other: UsageInfo) {
        *self += &other;
    }
}

impl<'a> std::iter::Sum<&'a UsageInfo> for UsageInfo {
    fn sum<I: Iterator<Item = &'a UsageInfo>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            return Self::default();
        };
        let mut total = first.clone();
        total.billed_cost = Some(first.billed());
        for usage in iter {
            total += usage;
        }
        total
    }
}

impl std::iter::Sum for UsageInfo {
    fn sum<I: Iterator<Item = UsageInfo>>(iter: I) -> Self {
        let usages: Vec<_> = iter.collect();
        usages.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{GenerationConfig, OpenAIServiceTier},
        model::{Model, ModelPricing},
        pricing::{ScheduledPricing, ServiceTier},
    };
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-9
    }

    #[test]
    fn adding_absurd_token_counts_saturates() {
        let huge = UsageInfo {
            cached_tokens: Some(usize::MAX),
            cache_write_tokens: Some(usize::MAX),
            cache_write_1h_tokens: Some(usize::MAX),
            reasoning_tokens: Some(usize::MAX),
            ..UsageInfo::new(usize::MAX, usize::MAX)
        };
        let mut total = huge.clone();
        total += &huge;
        assert_eq!(
            (total.input_tokens, total.output_tokens),
            (usize::MAX, usize::MAX)
        );
        assert_eq!(total.cached_tokens, Some(usize::MAX));
        assert_eq!(total.cache_write_tokens, Some(usize::MAX));
        assert_eq!(total.cache_write_1h_tokens, Some(usize::MAX));
        assert_eq!(total.reasoning_tokens, Some(usize::MAX));
        let summed: UsageInfo = [huge.clone(), huge].iter().sum();
        assert_eq!(summed.input_tokens, usize::MAX);
    }

    #[test]
    fn codex_plan_usage_headers_parse_and_old_records_load() {
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in [
            ("x-codex-plan-type", "prolite"),
            ("x-codex-active-limit", "premium"),
            ("x-codex-primary-used-percent", "11"),
            ("x-codex-primary-window-minutes", "10080"),
            ("x-codex-primary-reset-at", "1791108021"),
            ("x-codex-primary-reset-after-seconds", "593594"),
            // An unused secondary window is sent as zeros and an empty reset.
            ("x-codex-secondary-used-percent", "0"),
            ("x-codex-secondary-window-minutes", "0"),
            ("x-codex-secondary-reset-at", ""),
            ("x-codex-secondary-reset-after-seconds", "0"),
        ] {
            headers.insert(name, value.parse().unwrap());
        }
        let usage = PlanUsage::from_codex_headers(&headers).unwrap();
        assert_eq!(usage.plan, Some(SubscriptionPlan::ChatGpt));
        assert_eq!(serde_json::to_value(&usage).unwrap()["plan"], "chatgpt");
        assert_eq!(usage.plan_type.as_deref(), Some("prolite"));
        assert_eq!(usage.active_limit.as_deref(), Some("premium"));
        assert_eq!(
            usage.primary,
            Some(PlanUsageWindow {
                used_percent: 11.0,
                window_minutes: Some(10080),
                resets_at: Some(1_791_108_021),
                reset_after_seconds: Some(593_594),
            })
        );
        assert_eq!(usage.secondary, None);
        assert_eq!(usage.credits, None);
        assert_eq!(usage.most_used_window().unwrap().0, "primary");

        // Credits, as Codex reads them: both flags, then the balance.
        let mut with_credits = headers.clone();
        for (name, value) in [
            ("x-codex-credits-has-credits", "True"),
            ("x-codex-credits-unlimited", "0"),
            ("x-codex-credits-balance", "34012.5"),
        ] {
            with_credits.insert(name, value.parse().unwrap());
        }
        let credits = PlanUsage::from_codex_headers(&with_credits)
            .unwrap()
            .credits
            .unwrap();
        assert_eq!(
            credits,
            PlanCredits {
                has_credits: true,
                unlimited: false,
                balance: Some("34012.5".into()),
            }
        );
        assert!(credits.usable());
        with_credits.remove("x-codex-credits-unlimited");
        assert_eq!(
            PlanUsage::from_codex_headers(&with_credits)
                .unwrap()
                .credits,
            None
        );
        assert_eq!(
            PlanUsage::from_codex_headers(&reqwest::header::HeaderMap::new()),
            None
        );

        let call = ModelCallUsage::new("c".into(), "openai".into(), "m".into(), 1, 1, 10)
            .with_plan_usage(Some(usage.clone()));
        let round_trip: ModelCallUsage =
            serde_json::from_str(&serde_json::to_string(&call).unwrap()).unwrap();
        assert_eq!(round_trip.plan_usage, Some(usage));
        // Records saved before plan usage existed load without it.
        let old =
            r#"{"call_id":"c","provider":"openai","model":"m","input_tokens":1,"output_tokens":1}"#;
        let old: ModelCallUsage = serde_json::from_str(old).unwrap();
        assert_eq!(old.plan_usage, None);
        assert!(!serde_json::to_string(&old).unwrap().contains("plan_usage"));
        // Reports saved before the plan was named load without it.
        let old: PlanUsage =
            serde_json::from_str(r#"{"plan_type":"pro","primary":{"used_percent":5.0}}"#).unwrap();
        assert_eq!(old.plan, None);
    }

    fn priced(model: &Model, usage: UsageInfo, config: Option<&GenerationConfig>) -> UsageInfo {
        let mut usage = usage;
        usage.calculate_cost(model, config);
        usage
    }

    #[test]
    fn served_model_changed_covers_openrouter_and_openai_reports() {
        let usage = ModelCallUsage::new("id".into(), "p".into(), "m".into(), 0, 0, 0);
        assert!(!usage.served_model_changed());
        let mut routed = usage.clone();
        routed.requested_model = Some("openrouter/auto".into());
        assert!(routed.served_model_changed());
        let rerouted = usage.with_served_model(Some("gpt-5.2".into()), true);
        assert!(rerouted.served_model_changed());
    }

    #[test]
    fn short_context_only_tiers_fall_back_to_standard_long_context_rates() {
        let fast = GenerationConfig::default().with_openai_service_tier(OpenAIServiceTier::Fast);
        let model = Model::openai_gpt_5_5();
        // Short context: Fast is 2.5x ($5 -> $12.50 input, $30 -> $75 output).
        let short = priced(&model, UsageInfo::new(100_000, 1_000_000), Some(&fast));
        assert!(close(short.total_cost, 0.1 * 12.5 + 75.0));
        // Above 272K, GPT-5.5 Fast mode is not offered: standard long-context rates.
        let long = priced(&model, UsageInfo::new(300_000, 1_000_000), Some(&fast));
        assert!(close(long.total_cost, 0.3 * 10.0 + 45.0));
        // GPT-6 Fast applies to long context too.
        let astra = priced(
            &Model::openai_gpt_6_astra(),
            UsageInfo::new(300_000, 0),
            Some(&fast),
        );
        assert!(close(astra.total_cost, 0.3 * 40.0));
    }

    #[test]
    fn token_classes_without_a_price_are_billed_at_the_input_rate() {
        // gpt-5.4-pro has no cached-input discount; above 272K input costs $60.
        let pro = priced(
            &Model::openai_gpt_5_4_pro(),
            UsageInfo::new(1_000_000, 0).with_cached_tokens(400_000),
            None,
        );
        assert!(close(pro.input_cost, 0.6 * 60.0));
        assert!(close(pro.cache_read_cost.unwrap(), 0.4 * 60.0));
        assert!(close(pro.total_cost, 60.0));
        let short_pro = priced(
            &Model::openai_gpt_5_4_pro(),
            UsageInfo::new(200_000, 0).with_cached_tokens(100_000),
            None,
        );
        assert!(close(short_pro.total_cost, 0.2 * 30.0));
        // Models before GPT-5.6 charge no cache-write premium: writes cost input.
        let writes = priced(
            &Model::openai_gpt_5_5(),
            UsageInfo::new(100_000, 0).with_cache_write_tokens(40_000),
            None,
        );
        assert!(close(writes.cache_write_cost.unwrap(), 0.04 * 5.0));
        assert!(close(writes.total_cost, 0.1 * 5.0));
    }

    #[test]
    fn anthropic_long_context_rates_need_the_long_context_request() {
        // The rates Sonnet 4.5 had while its (now retired) 1M beta existed.
        let mut model = Model::anthropic_claude_sonnet_4_5();
        model.pricing.large_input_threshold = Some(200_000);
        model.pricing.large_input_cost_per_million = Some(6.00);
        model.pricing.large_output_cost_per_million = Some(22.50);
        let usage = || UsageInfo::new(250_000, 0);
        assert!(close(priced(&model, usage(), None).total_cost, 0.25 * 3.0));
        let long = GenerationConfig {
            long_context: Some(true),
            ..GenerationConfig::default()
        };
        assert!(close(
            priced(&model, usage(), Some(&long)).total_cost,
            0.25 * 6.0
        ));
    }

    /// Claude Haiku 5.5 charges 5x on every token class for a prompt over
    /// 100,000 tokens, cached and written input included, with no
    /// long-context request.
    #[test]
    fn haiku_5_5_prices_follow_the_prompt_size() {
        let haiku = Model::anthropic_claude_haiku_5_5();
        // At the threshold: $0.10 input, $0.50 output.
        let at = priced(&haiku, UsageInfo::new(100_000, 1_000_000), None);
        assert!(close(at.total_cost, 0.1 * 0.10 + 0.50));
        // Over it, mostly from the cache: $0.50 input, $0.05 reads, $0.625
        // 5-minute and $1 1-hour writes, $2.50 output.
        let mut usage = UsageInfo::new(150_000, 1_000_000)
            .with_cached_tokens(100_000)
            .with_cache_write_tokens(40_000);
        usage.cache_write_1h_tokens = Some(10_000);
        let over = priced(&haiku, usage, None);
        assert!(close(over.input_cost, 0.01 * 0.50));
        assert!(close(over.cache_read_cost.unwrap(), 0.1 * 0.05));
        assert!(close(
            over.cache_write_cost.unwrap(),
            0.03 * 0.625 + 0.01 * 1.00
        ));
        assert!(close(over.output_cost, 2.50));
        // The same through Claude Code, and 1.1x with US-only inference.
        let claude_code = priced(
            &Model::claude_code(haiku.clone()),
            UsageInfo::new(150_000, 0),
            None,
        );
        assert!(close(claude_code.total_cost, 0.15 * 0.50));
        let mut us = UsageInfo::new(150_000, 0);
        us.inference_geo = Some("us".into());
        assert!(close(
            priced(&haiku, us, None).total_cost,
            0.15 * 0.50 * 1.1
        ));
    }

    #[test]
    fn scheduled_prices_follow_the_request_time() {
        let mut model = Model::openai_gpt_6_luna();
        model.pricing = ModelPricing {
            input_cost_per_million: 0.75,
            output_cost_per_million: 3.75,
            schedule: vec![ScheduledPricing::new(
                "2027-01-01",
                ModelPricing {
                    input_cost_per_million: 1.50,
                    output_cost_per_million: 7.50,
                    ..ModelPricing::default()
                },
            )],
            ..ModelPricing::default()
        };
        let boundary = UNIX_EPOCH + Duration::from_secs(1_798_761_600);
        let cost_at = |at: SystemTime| {
            let mut usage = UsageInfo::new(1_000_000, 1_000_000);
            usage.calculate_cost_at(&model, None, at);
            usage.total_cost
        };
        assert!(close(cost_at(boundary - Duration::from_secs(1)), 4.5));
        assert!(close(cost_at(boundary), 9.0));
    }

    #[test]
    fn billing_kinds_decide_the_billed_amount() {
        let model = Model::openai_gpt_6_luna();
        let metered = priced(&model, UsageInfo::new(1_000_000, 0), None);
        assert!(close(metered.billed(), metered.total_cost));
        let subscription = priced(
            &model,
            UsageInfo::new(1_000_000, 0).with_billing(BillingKind::Subscription),
            None,
        );
        assert!(subscription.total_cost > 0.0);
        assert_eq!(subscription.billed(), 0.0);
        let reported = priced(&model, UsageInfo::new(1_000_000, 0), None).with_reported_cost(0.42);
        assert_eq!(reported.billing, BillingKind::ProviderReported);
        assert!(close(reported.billed(), 0.42));
        // Records saved before billing kinds: billed = total_cost.
        let legacy: UsageInfo = serde_json::from_value(serde_json::json!({
            "input_tokens": 10, "output_tokens": 5, "input_cost": 0.5,
            "output_cost": 0.25, "total_cost": 0.75
        }))
        .unwrap();
        assert_eq!(legacy.billing, BillingKind::Metered);
        assert!(close(legacy.billed(), 0.75));
    }

    #[test]
    fn sums_keep_billed_amounts_and_mark_mixed_billing() {
        let model = Model::openai_gpt_6_luna();
        let mut metered = priced(
            &model,
            UsageInfo::new(1_000_000, 0).with_cached_tokens(10),
            None,
        );
        metered.reasoning_tokens = Some(7);
        metered.service_tier = Some(ServiceTier::Standard);
        let subscription = priced(
            &model,
            UsageInfo::new(2_000_000, 0).with_billing(BillingKind::Subscription),
            None,
        );
        let total: UsageInfo = [metered.clone(), subscription.clone()].iter().sum();
        assert_eq!(total.input_tokens, 3_000_000);
        assert_eq!(total.cached_tokens, Some(10));
        assert_eq!(total.reasoning_tokens, Some(7));
        assert_eq!(total.service_tier, None);
        assert_eq!(total.billing, BillingKind::Mixed);
        assert!(close(
            total.total_cost,
            metered.total_cost + subscription.total_cost
        ));
        assert!(close(total.billed(), metered.total_cost));
        let mut added = metered.clone();
        added += &metered;
        assert_eq!(added.billing, BillingKind::Metered);
        assert_eq!(added.service_tier, Some(ServiceTier::Standard));
        assert!(close(added.billed(), 2.0 * metered.total_cost));
        assert_eq!(
            std::iter::empty::<UsageInfo>().sum::<UsageInfo>(),
            UsageInfo::default()
        );
    }
}
