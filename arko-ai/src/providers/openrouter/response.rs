//! Response and stream parsing: text, refusals, tool calls, verbatim
//! `reasoning_details`, stop reasons, usage and cost.
//!
//! Unknown fields and enum values are ignored; OpenRouter adds them without
//! notice.

use super::openrouter_error;
use crate::{
    config::GenerationConfig,
    error::Error,
    message::{Content, Message, Role},
    model::Model,
    pricing::ServiceTier,
    providers::{GenerationResponse, StreamEvent, take_refused_output},
    reasoning::ReasoningArtifact,
    stop::{Refusal, RefusalSource, StopReason},
    tool::ToolCall,
    usage::{ModelCallUsage, UsageInfo},
};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use serde_json::{Map, Value, json};

/// What a response is checked against: the requested slug, the model's
/// pricing and limits, and what is known about the model it may be served by.
#[derive(Debug, Clone)]
pub(crate) struct ResponseContext {
    pub(crate) model: Model,
    pub(crate) config: GenerationConfig,
    requested: String,
    /// The catalog's canonical (dated) slug, when known.
    canonical: Option<String>,
    /// Whether model fallbacks were requested.
    fallbacks: bool,
}

impl ResponseContext {
    pub(crate) fn new(model: Model, config: GenerationConfig, fallbacks: bool) -> Self {
        Self {
            requested: model.id.clone(),
            canonical: model
                .openrouter
                .as_ref()
                .and_then(|info| info.canonical_slug.clone()),
            model,
            config,
            fallbacks,
        }
    }

    /// Whether `served` is a different model from the one requested.
    /// Responses name the model by its dated permaslug, whose name can differ
    /// from the requested slug's (`anthropic/claude-4.5-sonnet-20250929` for
    /// `anthropic/claude-sonnet-4.5`). With the catalog's canonical slug the
    /// comparison is exact; without it only changes arko can recognize count:
    /// routers, `~` aliases and presets, configured fallbacks, or another
    /// vendor. Variant suffixes and trailing dates are ignored.
    fn served_model_changed(&self, served: &str) -> bool {
        let served_slug = normalized_slug(served);
        let same = normalized_slug(&self.requested) == served_slug
            || self
                .canonical
                .as_deref()
                .is_some_and(|canonical| normalized_slug(canonical) == served_slug);
        if same {
            return false;
        }
        let requested = super::request::base_slug(&self.requested);
        self.canonical.is_some()
            || self.fallbacks
            || requested.starts_with("openrouter/")
            || requested.starts_with(['~', '@'])
            || super::request::vendor(served) != super::request::vendor(requested)
    }
}

/// A slug without variant suffix or trailing snapshot date
/// (`-20260630` or `-2026-06-30`).
fn normalized_slug(slug: &str) -> &str {
    let slug = super::request::base_slug(slug);
    let bytes = slug.as_bytes();
    let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    let n = bytes.len();
    if n > 9 && bytes[n - 9] == b'-' && digits(n - 8..n) {
        return &slug[..n - 9];
    }
    if n > 11
        && bytes[n - 11] == b'-'
        && bytes[n - 6] == b'-'
        && bytes[n - 3] == b'-'
        && digits(n - 10..n - 6)
        && digits(n - 5..n - 3)
        && digits(n - 2..n)
    {
        return &slug[..n - 11];
    }
    slug
}

fn str_field<'a>(value: &'a Value, name: &str) -> Option<&'a str> {
    value.get(name).and_then(Value::as_str)
}

fn non_empty(text: Option<&str>) -> Option<&str> {
    text.filter(|text| !text.is_empty())
}

fn has_error(value: &Value) -> bool {
    value.get("error").is_some_and(|error| !error.is_null())
}

/// A tool call being assembled.
#[derive(Debug, Default)]
struct PendingToolCall {
    index: u64,
    id: Option<String>,
    name: String,
    arguments: String,
    started: bool,
}

impl PendingToolCall {
    fn into_tool_call(self) -> ToolCall {
        ToolCall::from_argument_text(self.id, self.name, &self.arguments)
    }
}

fn arguments_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Assembles streamed `delta.tool_calls` by `index`, so interleaved calls
/// keep their own arguments. Ids are opaque.
#[derive(Debug, Default)]
struct ToolCallAssembler {
    calls: Vec<PendingToolCall>,
}

impl ToolCallAssembler {
    fn push(&mut self, delta: &Value) -> Vec<StreamEvent> {
        let id = non_empty(str_field(delta, "id"));
        let index = delta
            .get("index")
            .and_then(Value::as_u64)
            .or_else(|| {
                id.and_then(|id| {
                    self.calls
                        .iter()
                        .find(|call| call.id.as_deref() == Some(id))
                })
                .map(|call| call.index)
            })
            .or_else(|| {
                (id.is_none())
                    .then(|| self.calls.last().map(|call| call.index))
                    .flatten()
            })
            .unwrap_or_else(|| {
                self.calls
                    .iter()
                    .map(|call| call.index + 1)
                    .max()
                    .unwrap_or(0)
            });
        let position = match self.calls.iter().position(|call| call.index == index) {
            Some(position) => position,
            None => {
                self.calls.push(PendingToolCall {
                    index,
                    ..PendingToolCall::default()
                });
                self.calls.len() - 1
            }
        };
        let call = &mut self.calls[position];
        if call.id.is_none() {
            call.id = id.map(String::from);
        }
        let function = delta.get("function");
        if call.name.is_empty()
            && let Some(name) = non_empty(function.and_then(|f| str_field(f, "name")))
        {
            call.name = name.to_string();
        }
        let mut events = Vec::new();
        if !call.started && !call.name.is_empty() {
            call.started = true;
            events.push(StreamEvent::ToolCallStart {
                index: index as usize,
                name: call.name.clone(),
            });
            if !call.arguments.is_empty() {
                events.push(StreamEvent::ToolCallArgumentsDelta {
                    index: index as usize,
                    delta: call.arguments.clone(),
                });
            }
        }
        let arguments = arguments_text(function.and_then(|f| f.get("arguments")));
        if !arguments.is_empty() {
            call.arguments.push_str(&arguments);
            if call.started {
                events.push(StreamEvent::ToolCallArgumentsDelta {
                    index: index as usize,
                    delta: arguments,
                });
            }
        }
        events
    }

    fn finish(mut self) -> Vec<PendingToolCall> {
        self.calls.sort_by_key(|call| call.index);
        self.calls
    }
}

/// Merges streamed `reasoning_details` chunks into whole items. Chunks with
/// the same `index` and `type` are one item: `text`, `summary` and `data`
/// are concatenated and any other non-null field (signature, id, format)
/// is taken. OpenRouter documents only that chunks "concatenate in order";
/// this rule is to be confirmed against non-streaming output.
const REDACTED: &str = "[REDACTED]";

#[derive(Debug, Default)]
pub(crate) struct DetailsMerger {
    items: Vec<Value>,
}

impl DetailsMerger {
    pub(crate) fn merge(&mut self, item: &Value) {
        let Some(fields) = item.as_object() else {
            self.items.push(item.clone());
            return;
        };
        let index = fields.get("index").and_then(Value::as_u64);
        let kind = fields.get("type").and_then(Value::as_str);
        let target = match index {
            // A continuation chunk may omit `type`; the index alone decides.
            Some(index) => self.items.iter_mut().rev().find(|existing| {
                existing.get("index").and_then(Value::as_u64) == Some(index)
                    && (kind.is_none() || existing.get("type").and_then(Value::as_str) == kind)
            }),
            // Without an index, a chunk continues the previous item of the
            // same type and id.
            None => self.items.last_mut().filter(|last| {
                last.get("index").is_none()
                    && last.get("type").and_then(Value::as_str) == kind
                    && last.get("id").filter(|id| !id.is_null())
                        == fields.get("id").filter(|id| !id.is_null())
            }),
        };
        let Some(Value::Object(target)) = target else {
            self.items.push(item.clone());
            return;
        };
        for (key, value) in fields {
            match (key.as_str(), value) {
                ("text" | "summary" | "data", Value::String(piece)) => match target.get_mut(key) {
                    // Streams may show encrypted data as `[REDACTED]`; never
                    // splice the placeholder into real data.
                    Some(Value::String(existing)) if existing.as_str() == REDACTED => {
                        *existing = piece.clone();
                    }
                    Some(Value::String(_)) if piece.as_str() == REDACTED => {}
                    Some(Value::String(existing)) => existing.push_str(piece),
                    _ => {
                        target.insert(key.clone(), value.clone());
                    }
                },
                (_, Value::Null) => {
                    target.entry(key.clone()).or_insert(Value::Null);
                }
                _ => {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
    }

    pub(crate) fn finish(self) -> Vec<Value> {
        self.items
    }
}

/// Readable reasoning in detail items: `reasoning.text` text and
/// `reasoning.summary` summaries. Encrypted items are never shown.
fn readable_reasoning(items: &[Value]) -> String {
    items
        .iter()
        .filter_map(|item| match str_field(item, "type") {
            Some("reasoning.text") => str_field(item, "text"),
            Some("reasoning.summary") => str_field(item, "summary"),
            _ => None,
        })
        .collect()
}

/// Message text from a string or an array of content parts.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|part| str_field(part, "type").is_none_or(|kind| kind == "text"))
            .filter_map(|part| str_field(part, "text"))
            .collect(),
        _ => String::new(),
    }
}

/// Everything collected from a response or a stream.
#[derive(Debug, Default)]
struct Turn {
    id: Option<String>,
    served_model: Option<String>,
    upstream_provider: Option<String>,
    service_tier: Option<String>,
    text: String,
    refusal: String,
    reasoning_details: Vec<Value>,
    tool_calls: Vec<PendingToolCall>,
    finish_reason: Option<String>,
    native_finish_reason: Option<String>,
    usage: Option<Value>,
}

impl Turn {
    /// Records the response-level fields of a response or chunk.
    fn record_envelope(&mut self, value: &Value) {
        if self.id.is_none() {
            self.id = non_empty(str_field(value, "id")).map(String::from);
        }
        if let Some(model) = non_empty(str_field(value, "model")) {
            self.served_model = Some(model.to_string());
        }
        if let Some(provider) = non_empty(str_field(value, "provider")) {
            self.upstream_provider = Some(provider.to_string());
        }
        if let Some(tier) = non_empty(str_field(value, "service_tier")) {
            self.service_tier = Some(tier.to_string());
        }
        if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
            // Usage arrives once, in the final chunk; a repeat replaces it.
            self.usage = Some(usage.clone());
        }
    }

    /// The first finish reason wins: the usage chunk repeats it.
    fn record_finish(&mut self, choice: &Value) {
        if self.finish_reason.is_none() {
            self.finish_reason = non_empty(str_field(choice, "finish_reason")).map(String::from);
        }
        if self.native_finish_reason.is_none() {
            self.native_finish_reason =
                non_empty(str_field(choice, "native_finish_reason")).map(String::from);
        }
    }
}

/// The error for `finish_reason: "error"`, from the choice's `error` when
/// present.
fn finish_error(choice: &Value) -> Error {
    match choice.get("error").filter(|error| !error.is_null()) {
        Some(error) => openrouter_error(None, None, &json!({ "error": error }).to_string()),
        None => openrouter_error(
            None,
            None,
            r#"{"error":{"message":"OpenRouter reported finish_reason \"error\" without details","metadata":{"error_type":"server"}}}"#,
        ),
    }
}

/// Parses a non-streaming response body. A 200 response can still carry an
/// error: a top-level `error` without choices, or `choices[0].error` with
/// partial output.
pub(crate) fn parse_response(
    body: &str,
    context: &ResponseContext,
) -> Result<GenerationResponse, Error> {
    let value: Value = serde_json::from_str(body)?;
    if has_error(&value) {
        return Err(openrouter_error(None, None, body));
    }
    let choice = value
        .pointer("/choices/0")
        .ok_or_else(|| Error::ApiError(format!("OpenRouter response has no choices: {body}")))?;
    if has_error(choice) || str_field(choice, "finish_reason") == Some("error") {
        return Err(finish_error(choice));
    }
    let mut turn = Turn::default();
    turn.record_envelope(&value);
    turn.record_finish(choice);
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    turn.text = content_text(message.get("content"));
    turn.refusal = str_field(&message, "refusal")
        .unwrap_or_default()
        .to_string();
    turn.reasoning_details = message
        .get("reasoning_details")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    turn.tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(position, call)| {
                    let function = call.get("function");
                    PendingToolCall {
                        index: position as u64,
                        id: non_empty(str_field(call, "id")).map(String::from),
                        name: function
                            .and_then(|f| str_field(f, "name"))
                            .unwrap_or_default()
                            .to_string(),
                        arguments: arguments_text(function.and_then(|f| f.get("arguments"))),
                        started: true,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(finish(turn, context))
}

/// Parses a Chat Completions SSE byte stream. Comment lines (`:
/// OPENROUTER PROCESSING` keep-alives) are skipped by the SSE parser. `End`
/// is emitted once a finish reason was seen and the stream reached `[DONE]`
/// or delivered its usage chunk; a stream that ends earlier yields
/// [`Error::StreamTruncated`]. Error chunks (HTTP 200 already sent) become
/// errors, refusals among them [`Error::Refusal`].
pub(crate) fn stream_events<S, B, E>(
    bytes: S,
    context: ResponseContext,
) -> BoxStream<'static, Result<StreamEvent, Error>>
where
    S: futures::Stream<Item = Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let stream = try_stream! {
        let mut events = Box::pin(bytes.eventsource());
        let mut turn = Turn::default();
        let mut tool_calls = ToolCallAssembler::default();
        let mut details = DetailsMerger::default();
        let mut done = false;

        while let Some(event) = events.next().await {
            let event = event.map_err(|error| Error::ApiError(format!("SSE error: {error}")))?;
            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                done = true;
                break;
            }
            let chunk: Value = serde_json::from_str(data)?;
            if has_error(&chunk) {
                Err(openrouter_error(None, None, data))?;
            }
            turn.record_envelope(&chunk);
            // Debug and OpenAI-style usage chunks have no choices.
            let Some(choice) = chunk.pointer("/choices/0") else {
                continue;
            };
            if has_error(choice) || str_field(choice, "finish_reason") == Some("error") {
                Err(finish_error(choice))?;
            }
            let delta = choice.get("delta").cloned().unwrap_or(Value::Null);
            if let Some(text) = non_empty(str_field(&delta, "content")) {
                turn.text.push_str(text);
                yield StreamEvent::TextChunk(text.to_string());
            }
            // Refusal text is reported through `StreamEvent::Refusal`, never
            // as ordinary text.
            if let Some(refusal) = str_field(&delta, "refusal") {
                turn.refusal.push_str(refusal);
            }
            let reasoning = non_empty(str_field(&delta, "reasoning"));
            if let Some(items) = delta.get("reasoning_details").and_then(Value::as_array) {
                for item in items {
                    details.merge(item);
                }
                if reasoning.is_none() {
                    let readable = readable_reasoning(items);
                    if !readable.is_empty() {
                        yield StreamEvent::ReasoningChunk(readable);
                    }
                }
            }
            if let Some(reasoning) = reasoning {
                yield StreamEvent::ReasoningChunk(reasoning.to_string());
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    for event in tool_calls.push(call) {
                        yield event;
                    }
                }
            }
            turn.record_finish(choice);
        }

        if turn.finish_reason.is_none() || !(done || turn.usage.is_some()) {
            Err(Error::StreamTruncated(
                "OpenRouter stream ended before a finish reason and its usage".to_string(),
            ))?;
        }
        turn.reasoning_details = details.finish();
        turn.tool_calls = tool_calls.finish();
        let response = finish(turn, &context);
        if let Some(refusal) = &response.refusal {
            yield StreamEvent::Refusal(refusal.clone());
        }
        yield StreamEvent::End(Box::new(response));
    };
    stream.boxed()
}

/// Maps `finish_reason` (with `native_finish_reason` and `message.refusal`)
/// to a stop reason and refusal.
///
/// | finish_reason | result |
/// |---|---|
/// | `stop` | `EndTurn` (`ToolUse` with tool calls, `StopSequence` for a native `stop_sequence`) |
/// | `tool_calls` | `ToolUse` |
/// | `length` | `MaxTokens`; tool calls with unfinished (invalid JSON) arguments are dropped |
/// | `content_filter` | `Refusal`: `SafetyClassifier` for a native `refusal`, `Model` when `message.refusal` is set, else `ContentFilter` |
/// | other | `Other(raw)` |
///
/// A `message.refusal` with any other finish reason is a `Model` refusal.
fn stop_and_refusal(
    message: &mut Message,
    finish_reason: Option<&str>,
    native: Option<&str>,
    refusal_text: Option<&str>,
) -> (StopReason, Option<Refusal>) {
    let details = || {
        let mut details = Map::new();
        details.insert("finish_reason".into(), json!(finish_reason));
        if let Some(native) = native {
            details.insert("native_finish_reason".into(), json!(native));
        }
        Value::Object(details)
    };
    let refusal = |source: RefusalSource, message: &mut Message| {
        let mut refusal = Refusal::new(source);
        refusal.explanation = refusal_text.map(String::from);
        if !source.keeps_turn() {
            refusal.partial_output = take_refused_output(message);
        }
        refusal.provider_details = details();
        (StopReason::Refusal, Some(refusal))
    };
    match finish_reason {
        Some("content_filter") if native == Some("refusal") => {
            refusal(RefusalSource::SafetyClassifier, message)
        }
        Some("content_filter") if refusal_text.is_some() => refusal(RefusalSource::Model, message),
        Some("content_filter") => refusal(RefusalSource::ContentFilter, message),
        _ if refusal_text.is_some() => refusal(RefusalSource::Model, message),
        Some("tool_calls") => (StopReason::ToolUse, None),
        Some("stop") if native == Some("stop_sequence") => (StopReason::StopSequence, None),
        Some("stop") if !message.tool_calls.is_empty() => (StopReason::ToolUse, None),
        Some("stop") => (StopReason::EndTurn, None),
        Some("length") => {
            message
                .tool_calls
                .retain(|call| call.invalid_arguments.is_none());
            (StopReason::MaxTokens, None)
        }
        Some(other) => (StopReason::Other(other.to_string()), None),
        None => (StopReason::unknown(), None),
    }
}

/// Parsed `usage`. `prompt_tokens` is the total input (cache reads and
/// writes included) and `completion_tokens` includes reasoning.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ReportedUsage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) cached_tokens: Option<usize>,
    pub(crate) cache_write_tokens: Option<usize>,
    pub(crate) reasoning_tokens: Option<usize>,
    pub(crate) cost: Option<f64>,
    pub(crate) is_byok: bool,
    pub(crate) upstream_inference_cost: Option<f64>,
}

impl ReportedUsage {
    pub(crate) fn parse(usage: &Value) -> Self {
        let count = |pointer: &str| {
            usage
                .pointer(pointer)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
        };
        let amount = |pointer: &str| {
            usage.pointer(pointer).and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            })
        };
        Self {
            prompt_tokens: count("/prompt_tokens").unwrap_or(0),
            completion_tokens: count("/completion_tokens").unwrap_or(0),
            cached_tokens: count("/prompt_tokens_details/cached_tokens"),
            cache_write_tokens: count("/prompt_tokens_details/cache_write_tokens"),
            reasoning_tokens: count("/completion_tokens_details/reasoning_tokens"),
            cost: amount("/cost"),
            is_byok: usage
                .get("is_byok")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            upstream_inference_cost: amount("/cost_details/upstream_inference_cost"),
        }
    }

    /// `UsageInfo` for this usage. The reported `cost` is what was charged
    /// (`BillingKind::ProviderReported`); with BYOK it is OpenRouter's fee and
    /// the provider's own charge is `upstream_cost`, both billed. Only when
    /// no cost is reported is the call estimated from the model's prices.
    pub(crate) fn usage_info(
        &self,
        model: &Model,
        config: &GenerationConfig,
        service_tier: Option<&str>,
    ) -> UsageInfo {
        let mut usage = UsageInfo::new(self.prompt_tokens, self.completion_tokens);
        usage.cached_tokens = self.cached_tokens;
        usage.cache_write_tokens = self.cache_write_tokens;
        usage.reasoning_tokens = self.reasoning_tokens;
        usage.service_tier = service_tier.map(ServiceTier::from_openai);
        match self.cost {
            Some(cost) => {
                if self.is_byok {
                    usage.upstream_cost = self.upstream_inference_cost;
                }
                usage.with_reported_cost(cost)
            }
            None => {
                usage.calculate_cost(model, Some(config));
                usage
            }
        }
    }
}

fn finish(turn: Turn, context: &ResponseContext) -> GenerationResponse {
    let tool_calls = turn
        .tool_calls
        .into_iter()
        .map(PendingToolCall::into_tool_call)
        .collect();
    let mut message = Message {
        role: Role::Assistant,
        content: vec![Content::Text(turn.text)],
        reasoning_artifact: (!turn.reasoning_details.is_empty())
            .then(|| ReasoningArtifact::new_openrouter(turn.reasoning_details)),
        tool_calls,
        id: None,
        configuration_update: None,
    };
    let refusal_text = non_empty(Some(turn.refusal.as_str()));
    let (stop_reason, refusal) = stop_and_refusal(
        &mut message,
        turn.finish_reason.as_deref(),
        turn.native_finish_reason.as_deref(),
        refusal_text,
    );

    let reported = turn.usage.as_ref().map(ReportedUsage::parse);
    if reported.is_none() {
        log::warn!("arko_ai: OpenRouter response carried no usage; recording zero tokens");
    }
    let usage = reported.unwrap_or_default().usage_info(
        &context.model,
        &context.config,
        turn.service_tier.as_deref(),
    );

    let served = turn
        .served_model
        .unwrap_or_else(|| context.requested.clone());
    let stop_details = {
        let mut details = Map::new();
        if let Some(native) = &turn.native_finish_reason {
            details.insert("native_finish_reason".into(), json!(native));
        }
        if let Some(text) = refusal_text {
            details.insert("refusal".into(), json!(text));
        }
        (!details.is_empty()).then_some(Value::Object(details))
    };
    let mut model_call_usage = ModelCallUsage::new(
        turn.id.unwrap_or_default(),
        context.model.provider.as_telemetry_str().to_string(),
        served.clone(),
        usage.input_tokens,
        usage.output_tokens,
        context.model.max_input_tokens,
    )
    .with_stop_metadata(turn.finish_reason, stop_details);
    if context.served_model_changed(&served) {
        model_call_usage.requested_model = Some(context.requested.clone());
    }
    model_call_usage.upstream_provider = turn.upstream_provider;

    GenerationResponse {
        message,
        usage,
        model_call_usage,
        stop_reason,
        refusal,
    }
}
