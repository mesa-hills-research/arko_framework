//! The size of the next request and the tool results that would put it over
//! the model's input limit: the overflow guard, the opt-in caps, and the
//! recovery from a request the provider found too large.

use super::*;

/// The facts the notice of a shortened `result` gives: `text` is its text
/// as it stands (without a notice it already ends with), at `tokens`
/// tokens, under its opt-in `cap`.
fn notice_facts<'a>(
    result: &'a ai::ToolResult,
    paths: &'a HashMap<String, String>,
    reason: ShortenReason,
    text: &str,
    tokens: usize,
    cap: Option<usize>,
    tool_config: &crate::agent_config::AgentToolConfig,
) -> result_budget::NoticeFacts<'a> {
    result_budget::NoticeFacts {
        tool_name: &result.name,
        path: result
            .call_id
            .as_ref()
            .and_then(|id| paths.get(id))
            .map(String::as_str),
        reason,
        tokens,
        bytes: text.len(),
        lines: text.lines().count(),
        head_lines: 0,
        tail_lines: 0,
        cap,
        terminal: tool_config.offers_terminal(),
        line_ranges: tool_config.read_file_line_ranges,
    }
}

/// A request as it was sent: what its reported usage is compared with to
/// estimate the next request (see [`result_budget`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct RequestMark {
    /// The conversation's message count.
    pub(super) messages: usize,
    /// Its size in bytes ([`result_budget::request_bytes`]).
    pub(super) bytes: usize,
    /// Its images' estimated tokens ([`ai::RequestSizes::image_tokens`]),
    /// which the provider's input count includes and its text bytes don't.
    pub(super) image_tokens: usize,
    /// The conversation it belongs to ([`Task::conversation_usage_start`]).
    pub(super) conversation: usize,
}

/// How many times a request the provider found over the model's input
/// limit is sent again after shortening a tool result; see
/// [`AgentLoop::recover_from_overflow`].
const MAX_OVERFLOW_RECOVERIES: u32 = 2;

impl AgentLoop {
    /// The estimated input of the current conversation's next request, in
    /// tokens: the latest answered request's reported input, plus the bytes
    /// added since at the task's bytes-per-token ratio (see
    /// [`result_budget`]). Without such a request in this run (a task run
    /// again, or a fresh conversation), the whole request is estimated from
    /// its bytes, and the latest reported request with its response is the
    /// least it can be. What the handover check compares, and what the
    /// overflow guard sizes shortened results by.
    pub(super) fn estimated_request_tokens(&self) -> usize {
        self.request_estimate(false)
    }

    /// The fewest tokens the current conversation's next request can take:
    /// as [`estimated_request_tokens`](Self::estimated_request_tokens), but
    /// with every byte added since the latest answered request at
    /// [`result_budget::LEAST_BYTES_PER_TOKEN`] and the reply at its bytes
    /// alone. What the overflow guard decides by: it acts only when even
    /// this is over the model's input limit.
    fn least_request_tokens(&self) -> usize {
        self.request_estimate(true)
    }

    fn request_estimate(&self, least: bool) -> usize {
        let ratio = self.bytes_per_token;
        let messages = &self.task.conversation.messages;
        if let (Some(mark), Some(input)) = (
            self.answered_request,
            self.task.current_conversation_context_size(),
        ) && mark.conversation == self.task.conversation_usage_start
            && mark.messages <= messages.len()
        {
            let added = &messages[mark.messages..];
            let bytes = |assistant: bool| {
                added
                    .iter()
                    .filter(|message| (message.role == ai::Role::Assistant) == assistant)
                    .map(result_budget::message_bytes)
                    .sum::<usize>()
            };
            if least {
                return input + result_budget::least_tokens(bytes(true) + bytes(false));
            }
            // The model's reply counts at least its reported output, which
            // includes reasoning the next request may replay.
            let output = self
                .task
                .usage_history
                .last()
                .map_or(0, |usage| usage.output_tokens);
            let reply = output.max(result_budget::tokens(bytes(true), ratio));
            return input + reply + result_budget::tokens(bytes(false), ratio);
        }
        let whole =
            result_budget::request_bytes(&self.task.conversation, &self.build_generation_config());
        let latest = self.task.latest_conversation_usage();
        if least {
            return result_budget::least_tokens(whole)
                .max(latest.map_or(0, |usage| usage.input_tokens));
        }
        result_budget::tokens(whole, ratio)
            .max(latest.map_or(0, |usage| usage.input_tokens + usage.output_tokens))
    }

    /// For a task that runs again with requests in its current
    /// conversation: its latest request, as the conversation shows it (the
    /// messages before the model's latest reply), with the task's ratio
    /// calibrated from its reported input, so the next request is estimated
    /// as in the run that made it. `None` without such a request.
    pub(super) fn saved_request_mark(&mut self) -> Option<RequestMark> {
        let input = self.task.current_conversation_context_size()?;
        let conversation = &self.task.conversation;
        let reply = conversation
            .messages
            .iter()
            .rposition(|message| message.role == ai::Role::Assistant)?;
        let sizes = ai::RequestSizes::of_messages(
            conversation,
            &conversation.messages[..reply],
            self.build_generation_config().offered_tools(),
        );
        let bytes = sizes.text_bytes();
        if let Some(ratio) =
            result_budget::calibrated_ratio(bytes, input.saturating_sub(sizes.image_tokens))
        {
            self.bytes_per_token = ratio;
        }
        Some(RequestMark {
            messages: reply,
            bytes,
            image_tokens: sizes.image_tokens,
            conversation: self.task.conversation_usage_start,
        })
    }

    /// The paths the calls of the conversation's latest model turn passed
    /// to `read_file`, by call id: what a shortened result's notice names.
    fn read_file_paths(&self) -> HashMap<String, String> {
        self.task
            .conversation
            .messages
            .iter()
            .rev()
            .find(|message| message.role == ai::Role::Assistant && !message.tool_calls.is_empty())
            .into_iter()
            .flat_map(|message| &message.tool_calls)
            .filter(|call| call.name == crate::tool::builtins::READ_FILE_TOOL_NAME)
            .filter_map(|call| {
                let path = call.arguments.get("path")?.as_str()?;
                Some((call.id.clone()?, path.to_string()))
            })
            .collect()
    }

    /// The overflow guard and the opt-in caps
    /// ([`ToolResultLimits`](crate::agent_config::ToolResultLimits)) over
    /// the turn's native results, as they go into the conversation: after
    /// result interceptors and distillers, before the next request. A
    /// result over its cap, or among the largest when the turn's results
    /// would certainly put the next request over the model's input limit
    /// (see [`result_budget::plan`]), is shortened to its head and tail
    /// with a notice, and `ToolResultShortened` reports it; any other result
    /// is left alone.
    pub(super) async fn limit_tool_results(&mut self) -> Result<(), ArkoAgentsError> {
        if self.task.pending_native_tool_results.is_empty() {
            return Ok(());
        }
        let tool_config = self.task.agent_config.tool_config.clone();
        let caps: Vec<Option<usize>> = self
            .task
            .pending_native_tool_results
            .iter()
            .map(|result| tool_config.max_result_tokens(&result.name))
            .collect();
        let guard = tool_config.result_limits.overflow_guard;
        if !guard && caps.iter().all(Option::is_none) {
            return Ok(());
        }
        let ratio = self.bytes_per_token;
        let limit = match guard {
            true => {
                self.llm_client
                    .max_input_tokens(&self.task.conversation.model)
                    .await
            }
            false => None,
        };
        let room = limit.map(|limit| {
            let margin = tool_config
                .result_limits
                .margin_tokens
                .unwrap_or_else(|| result_budget::default_margin(limit));
            result_budget::Room {
                available: limit
                    .saturating_sub(margin)
                    .saturating_sub(self.estimated_request_tokens()),
                certain: limit.saturating_sub(self.least_request_tokens()),
                least_share: ratio / result_budget::LEAST_BYTES_PER_TOKEN,
                preview: result_budget::preview_tokens(limit),
            }
        });
        let sizes: Vec<usize> = self
            .task
            .pending_native_tool_results
            .iter()
            .map(|result| result_budget::tokens(result.output.len(), ratio))
            .collect();
        let paths = self.read_file_paths();
        let facts = |index: usize, reason: ShortenReason| {
            let result = &self.task.pending_native_tool_results[index];
            notice_facts(
                result,
                &paths,
                reason,
                &result.output,
                sizes[index],
                caps[index],
                &tool_config,
            )
        };
        let notices: Vec<usize> = (0..sizes.len())
            .map(|index| {
                result_budget::notice_tokens(&facts(index, ShortenReason::Overflow), ratio)
            })
            .collect();
        let cuts = result_budget::plan(&sizes, &notices, &caps, room);
        if cuts.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut shortened = Vec::new();
        for (index, cut) in cuts.into_iter().enumerate() {
            let Some(cut) = cut else {
                continue;
            };
            let result = &self.task.pending_native_tool_results[index];
            let (text, kept) =
                result_budget::shorten(&result.output, cut.keep, ratio, facts(index, cut.reason));
            log::warn!(
                "Task {}: the result of {} (about {} tokens) was shortened to about {kept} tokens ({:?}).",
                self.task.id,
                result.name,
                sizes[index],
                cut.reason
            );
            shortened.push((
                index,
                text,
                AgentEvent::ToolResultShortened {
                    call_id: result.call_id.clone(),
                    tool_name: result.name.clone(),
                    original_tokens: sizes[index],
                    kept_tokens: kept,
                    reason: cut.reason,
                },
            ));
        }
        for (index, text, event) in shortened {
            self.task.pending_native_tool_results[index].output = text;
            self.emit_event(event).await?;
        }
        Ok(())
    }

    /// The largest tool result added to the conversation since the
    /// latest answered request (or since the last message with results):
    /// its message, its place in it, and its size in bytes.
    fn largest_new_result(&self) -> Option<(usize, usize, usize)> {
        let messages = &self.task.conversation.messages;
        // The results added since the latest answered request: one of them
        // made this request too large.
        let start = self
            .answered_request
            .filter(|mark| {
                mark.conversation == self.task.conversation_usage_start
                    && mark.messages <= messages.len()
            })
            .map(|mark| mark.messages)
            .or_else(|| {
                messages.iter().rposition(|message| {
                    message
                        .content
                        .iter()
                        .any(|content| content.as_tool_result().is_some())
                })
            })
            .unwrap_or(messages.len());
        messages
            .iter()
            .enumerate()
            .skip(start)
            .flat_map(|(message, content)| {
                content
                    .content
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, content)| {
                        content
                            .as_tool_result()
                            .map(|result| (message, index, result.output.len()))
                    })
            })
            .max_by_key(|(_, _, bytes)| *bytes)
    }

    /// A request the provider found over the model's input limit
    /// ([`ai::ErrorKind::ContextOverflow`]): with the overflow guard on, in
    /// native tool calling, the largest tool result added since the latest
    /// answered request is shortened (to a preview, or one already that
    /// small to its notice alone) and the request is sent again, at most
    /// [`MAX_OVERFLOW_RECOVERIES`] times in a row. When that can't help
    /// (after those attempts, or with no result large enough to have caused
    /// it), an agent that hands over does so without the step that made the
    /// request too large; see
    /// [`hand_over_without_the_step`](Self::hand_over_without_the_step).
    /// Returns whether a request is sent again; otherwise recovery goes on
    /// as for any error that retrying can't fix, and the task stops with
    /// the error's kind.
    pub(super) async fn recover_from_overflow(
        &mut self,
        error: &ai::Error,
    ) -> Result<bool, ArkoAgentsError> {
        if error.kind() != ai::ErrorKind::ContextOverflow
            || !self
                .task
                .agent_config
                .tool_config
                .result_limits
                .overflow_guard
            || !self.is_native_mode()
        {
            return Ok(false);
        }
        if self.overflow_recoveries >= MAX_OVERFLOW_RECOVERIES {
            return self.hand_over_without_the_step(error).await;
        }
        let Some((message, index, _)) = self.largest_new_result() else {
            return self.hand_over_without_the_step(error).await;
        };
        let ratio = self.bytes_per_token;
        let preview = self
            .llm_client
            .max_input_tokens(&self.task.conversation.model)
            .await
            .map_or(
                result_budget::preview_tokens(0),
                result_budget::preview_tokens,
            );
        let paths = self.read_file_paths();
        let tool_config = self.task.agent_config.tool_config.clone();
        let ai::Content::ToolResult(result) =
            &self.task.conversation.messages[message].content[index]
        else {
            return Ok(false);
        };
        // A notice the result already ends with is replaced, not kept.
        let (output, shortened_before) = match result.output.rfind("\n\n[Shortened: ") {
            Some(notice) => (result.output[..notice].to_string(), true),
            None if result.output.starts_with("[Shortened: ") => (String::new(), true),
            None => (result.output.clone(), false),
        };
        let tokens = result_budget::tokens(output.len(), ratio);
        // Too small to be what made the request too large.
        if tokens < result_budget::MIN_KEPT_TOKENS {
            return self.hand_over_without_the_step(error).await;
        }
        // A preview first; one shortened before (by the guard, a cap or the
        // previous attempt) keeps its notice alone.
        let keep = if tokens > preview && !shortened_before {
            preview
        } else {
            0
        };
        let facts = notice_facts(
            result,
            &paths,
            ShortenReason::OverflowRecovery,
            &output,
            tokens,
            None,
            &tool_config,
        );
        let (text, kept) = result_budget::shorten(&output, keep, ratio, facts);
        let shortened = AgentEvent::ToolResultShortened {
            call_id: result.call_id.clone(),
            tool_name: result.name.clone(),
            original_tokens: tokens,
            kept_tokens: kept,
            reason: ShortenReason::OverflowRecovery,
        };
        let tool_name = result.name.clone();
        if let ai::Content::ToolResult(result) =
            &mut self.task.conversation.messages[message].content[index]
        {
            result.output = text;
        }
        self.overflow_recoveries += 1;
        // Sent again at once; not an ordinary recovery attempt.
        self.recovery.indefinite_interactions =
            self.recovery.indefinite_interactions.saturating_add(1);
        log::warn!(
            "Task {}: the request was over the model's input limit, so the result of {tool_name} was shortened and the request is sent again ({}/{MAX_OVERFLOW_RECOVERIES}).",
            self.task.id,
            self.overflow_recoveries
        );
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: format!(
                "The request was over the model's input limit, so the result of {tool_name} was shortened and the request is sent again ({}/{MAX_OVERFLOW_RECOVERIES}): {error}",
                self.overflow_recoveries
            ),
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: Some(ai::ErrorKind::ContextOverflow),
        })
        .await?;
        self.emit_event(shortened).await?;
        Ok(true)
    }

    /// The last step of [`recover_from_overflow`](Self::recover_from_overflow),
    /// for an agent that hands over (it has `handover`, or the host set a
    /// threshold): the step the request added since the latest answered
    /// request, which fit, is left out (the model's reply, its results, and
    /// the runtime's messages after them, such as a report request), and the
    /// report is asked for at the next turn boundary from the conversation
    /// as it was, with [`STEP_LEFT_OUT`](crate::handover::STEP_LEFT_OUT)
    /// before the request. Context updates added since stay: they hold what
    /// the host sent. Returns whether a request is sent again: not without
    /// a handover, without an answered request in this conversation, with a
    /// report already given, or with no reply or results to leave out (the
    /// report request itself was too large).
    async fn hand_over_without_the_step(
        &mut self,
        error: &ai::Error,
    ) -> Result<bool, ArkoAgentsError> {
        let hands_over =
            self.task.agent_config.handover.is_some() || self.handover_threshold.get().is_some();
        let reported = self
            .task
            .handover_requested
            .as_ref()
            .is_some_and(|request| request.report.is_some());
        let messages = &self.task.conversation.messages;
        let Some(start) = self
            .answered_request
            .filter(|mark| {
                mark.conversation == self.task.conversation_usage_start
                    && mark.messages <= messages.len()
            })
            .map(|mark| mark.messages)
            .filter(|_| hands_over && !reported)
        else {
            return Ok(false);
        };
        let in_step = |message: &ai::Message| {
            message.role == ai::Role::Assistant
                || message
                    .content
                    .iter()
                    .any(|content| content.as_tool_result().is_some())
        };
        if !messages[start..].iter().any(in_step) {
            return Ok(false);
        }
        let added = self.task.conversation.messages.split_off(start);
        let runtime = &self.task.runtime_messages;
        let kept: Vec<ai::Message> = added
            .into_iter()
            .enumerate()
            .filter(|(offset, message)| !in_step(message) && !runtime.contains(&(start + offset)))
            .map(|(_, message)| message)
            .collect();
        self.task.conversation.messages.extend(kept);
        // The messages kept after `start` are the host's, not the runtime's.
        self.task.runtime_messages.retain(|index| *index < start);
        self.task
            .runtime_text_messages
            .retain(|index| *index < start);
        if self.details_sent_with.is_some_and(|index| index >= start) {
            self.details_sent_with = None;
        }
        self.task.handover_requested = Some(HandoverRequest::new(
            self.task.current_conversation_context_size().unwrap_or(0),
        ));
        self.step_left_out = true;
        // Sent again at once; not an ordinary recovery attempt.
        self.recovery.indefinite_interactions =
            self.recovery.indefinite_interactions.saturating_add(1);
        log::warn!(
            "Task {}: the request was over the model's input limit even with its results shortened, so its latest step was left out and the agent hands over.",
            self.task.id
        );
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: format!(
                "The request was over the model's input limit even with its results shortened, so its latest step was left out and the agent hands over from the conversation before it: {error}"
            ),
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: Some(ai::ErrorKind::ContextOverflow),
        })
        .await?;
        Ok(true)
    }
}
