//! The model's response as it streams: its text and reasoning sent as they
//! come, the calls it announces, the commands that arrive meanwhile, and its
//! end, whose usage is recorded and reported.

use super::*;

/// The text of a message.
pub(super) fn message_text(message: &ai::Message) -> String {
    message
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect()
}

/// The text of an output-based turn without its tool-call markup.
fn parsed_plain_text(events: &[RawParserEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            RawParserEvent::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// A model's turn, streamed to its end, for the turn phase.
pub(super) struct StreamedTurn {
    /// The final response, which the conversation holds; `None` for a
    /// stream that ended without one, its partial text kept.
    pub(super) response: Option<ai::GenerationResponse>,
    /// The output-based parser's events (none in native mode).
    pub(super) parser_events: Vec<RawParserEvent>,
    /// The calls the stream announced (`ToolCallInitiated`), by index, with
    /// their runtime ids and names: the turn runs those its response holds
    /// and rejects the others.
    pub(super) announced: HashMap<usize, (Uuid, String)>,
}

/// How a pass's model request and its stream ended.
#[allow(
    clippy::large_enum_variant,
    reason = "returned once per turn and taken apart at once: a box would only add an allocation"
)]
pub(super) enum ModelTurn {
    /// With a turn whose calls the turn phase handles.
    Streamed(StreamedTurn),
    /// Otherwise: the pass ends (`NextPass`, or `Exit` for a task that
    /// ended).
    Ended(Flow),
}

impl AgentLoop {
    pub(super) async fn emit_throttled_text(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.text_buffer.is_empty()
            && (self.event_throttle_duration.is_zero()
                || self.last_text_emit_time.elapsed() >= self.event_throttle_duration)
        {
            self.emit_event(AgentEvent::AssistantResponseChunk(self.text_buffer.clone()))
                .await?;
            self.text_buffer.clear();
            self.last_text_emit_time = Instant::now();
        }
        Ok(())
    }

    pub(super) async fn flush_throttled_text(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.text_buffer.is_empty() {
            self.emit_event(AgentEvent::AssistantResponseChunk(self.text_buffer.clone()))
                .await?;
            self.text_buffer.clear();
        }
        Ok(())
    }

    async fn emit_throttled_reasoning(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.reasoning_buffer.is_empty()
            && (self.event_throttle_duration.is_zero()
                || self.last_reasoning_emit_time.elapsed() >= self.event_throttle_duration)
        {
            self.emit_event(AgentEvent::ReasoningChunk(self.reasoning_buffer.clone()))
                .await?;
            self.reasoning_buffer.clear();
            self.last_reasoning_emit_time = Instant::now();
        }
        Ok(())
    }

    pub(super) async fn flush_throttled_reasoning(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.reasoning_buffer.is_empty() {
            self.emit_event(AgentEvent::ReasoningChunk(self.reasoning_buffer.clone()))
                .await?;
            self.reasoning_buffer.clear();
        }
        Ok(())
    }

    /// Records model call usage on the task (computing the input delta)
    /// and emits the `ModelCallUsageReported` event.
    async fn record_and_emit_model_call_usage(
        &mut self,
        raw_usage: ai::ModelCallUsage,
    ) -> Result<(), ArkoAgentsError> {
        let usage_with_delta = self.task.record_model_call_usage(raw_usage);
        let usage = usage_with_delta.clone();
        self.emit_event(AgentEvent::ModelCallUsageReported(usage_with_delta))
            .await?;
        if let Some(plan_usage) = usage.plan_usage.clone() {
            self.emit_event(AgentEvent::PlanUsageReported(plan_usage))
                .await?;
        }
        if usage.served_model_changed() {
            let requested_model = usage.requested_model.clone().unwrap_or_else(|| {
                match self.task.agent_config.refusal_fallback.as_ref() {
                    Some(fallback) if self.current_request_is_fallback => fallback.model.id.clone(),
                    _ => self.task.conversation.model.id.clone(),
                }
            });
            self.emit_event(AgentEvent::ModelRerouted {
                provider: usage.provider.clone(),
                requested_model,
                served_model: usage.served_model.clone().unwrap_or(usage.model.clone()),
            })
            .await?;
        }
        if usage.reasoning_summaries_refused {
            // The run asks for none again, also once resumed.
            if let Some(ai::ReasoningConfig::OpenAI {
                include_summaries, ..
            }) = &mut self
                .task
                .agent_config
                .llm_config
                .generation_config
                .reasoning
            {
                *include_summaries = false;
            }
            self.emit_event(AgentEvent::ReasoningSummariesRefused {
                provider: usage.provider.clone(),
                model: usage.model.clone(),
            })
            .await?;
        }
        self.monitor_cache().await
    }

    /// Streams a response to its end, or until the host stops the task:
    /// its text and reasoning are sent as they come, the calls it
    /// announces are reported, and its end is accounted for (usage, the
    /// request estimate, refusals). Every outcome but a usable turn ends
    /// the pass.
    pub(super) async fn stream_response(
        &mut self,
        mut response_stream: futures::stream::BoxStream<
            'static,
            Result<ai::StreamEvent, ai::Error>,
        >,
    ) -> Result<ModelTurn, ArkoAgentsError> {
        let mut raw_events_for_turn: Vec<RawParserEvent> = Vec::new();
        let mut parser = if !self.is_native_mode() {
            Some(ToolCallParser::new(
                self.task.agent_config.tool_config.offered_tool_names(),
                self.parsing_config.clone(),
            ))
        } else {
            None
        };

        let mut turn_ended = false;
        let mut final_generation_response: Option<ai::GenerationResponse> = None;
        let mut streamed_text = String::new();
        let mut streamed_reasoning = 0;
        let mut native_stream_call_ids: HashMap<usize, (Uuid, String)> = HashMap::new();
        let mut streamed_refusal: Option<ai::Refusal> = None;

        // The stream runs until it ends or fails, or the host stops the
        // task; no recovery budget cuts it off (only arko_ai's opt-in
        // stream idle timeout can).
        loop {
            // The final response ends the loop as it arrives.
            debug_assert!(final_generation_response.is_none());
            tokio::select! {
                biased;
                maybe_cmd = self.command_receiver.recv() => {
                    if self.on_stream_command(maybe_cmd, &streamed_text, &mut native_stream_call_ids).await? {
                        turn_ended = true;
                        break;
                    }
                }

                stream_item = response_stream.try_next() => {
                    match stream_item {
                        Ok(Some(ai::StreamEvent::TextChunk(text_chunk))) => {
                            streamed_text.push_str(&text_chunk);
                            if self.is_native_mode() {
                                self.text_buffer.push_str(&text_chunk);
                                self.emit_throttled_text().await?;
                            } else {
                                raw_events_for_turn.extend(parser.as_mut().unwrap().ingest_chunk(&text_chunk));
                            }
                        }
                        Ok(Some(ai::StreamEvent::ReasoningChunk(reasoning))) => {
                            streamed_reasoning += reasoning.len();
                            self.reasoning_buffer.push_str(&reasoning);
                            self.emit_throttled_reasoning().await?;
                        }
                        Ok(Some(ai::StreamEvent::ToolCallStart { index, name })) => {
                            if self.is_native_mode() {
                                self.flush_throttled_text().await?;
                                let uuid = Uuid::new_v4();
                                native_stream_call_ids.insert(index, (uuid, name.clone()));
                                self.emit_event(AgentEvent::ToolCallInitiated {
                                    tool_call_id: uuid,
                                    tool_name: name,
                                }).await?;
                            }
                        }
                        Ok(Some(ai::StreamEvent::ToolCallArgumentsDelta { .. })) => {}
                        // Providers repeat the refusal on `End`; it is
                        // kept only for a stream that ends without one.
                        Ok(Some(ai::StreamEvent::Refusal(refusal))) => {
                            streamed_refusal = Some(refusal);
                        }
                        Ok(Some(ai::StreamEvent::End(boxed_generation_response))) => {
                            if self.on_response_end(*boxed_generation_response, &mut parser, &mut raw_events_for_turn, &mut native_stream_call_ids, &mut final_generation_response).await? {
                                turn_ended = true;
                            }
                            break;
                        }
                        Ok(None) => {
                            if let Some(ref mut p) = parser {
                                raw_events_for_turn.extend(p.finalize());
                            }
                            if let Some(refusal) = streamed_refusal.take() {
                                let partial = ai::Message::new_assistant(&streamed_text);
                                let plain_text = parser.is_some().then(|| parsed_plain_text(&raw_events_for_turn));
                                self.reject_announced_calls(&mut native_stream_call_ids, REFUSED_TURN_CALL).await?;
                                self.handle_refusal(refusal, partial, plain_text).await?;
                                turn_ended = true;
                                break;
                            }
                            // Without a final response no announced call runs.
                            self.reject_announced_calls(&mut native_stream_call_ids, FAILED_STREAM_CALL).await?;
                            if !streamed_text.is_empty() {
                                self.task.conversation.add_assistant_message(&streamed_text);
                            }
                            if streamed_text.is_empty() {
                                self.handle_llm_error(ai::Error::StreamTruncated("the response stream ended with no output".to_string()), None).await?;
                                turn_ended = true;
                            }
                            break;
                        }
                        // A refusal reported as an error (Gemini
                        // generation-blocked codes): its partial text is
                        // not kept.
                        Err(ai::Error::Refusal(refusal)) => {
                            self.discard_streamed(&streamed_text, streamed_reasoning).await?;
                            self.reject_announced_calls(&mut native_stream_call_ids, REFUSED_TURN_CALL).await?;
                            self.handle_refusal(*refusal, ai::Message::new_assistant(""), None).await?;
                            turn_ended = true;
                            break;
                        }
                        // The request is sent again unchanged, so what the
                        // failed stream delivered is not kept: it would end
                        // the retried request on an assistant turn (a
                        // prefill, which Claude 4.6 and later reject).
                        Err(e) => {
                            self.discard_streamed(&streamed_text, streamed_reasoning).await?;
                            self.reject_announced_calls(&mut native_stream_call_ids, FAILED_STREAM_CALL).await?;
                            self.handle_llm_error(e, None).await?;
                            turn_ended = true;
                            break;
                        }
                    }
                }
            }
        }
        self.flush_throttled_reasoning().await?;
        self.flush_throttled_text().await?;

        if turn_ended || !matches!(self.task.current_state, TaskState::InProgress) {
            return Ok(ModelTurn::Ended(self.end_of_pass()));
        }
        Ok(ModelTurn::Streamed(StreamedTurn {
            response: final_generation_response,
            parser_events: raw_events_for_turn,
            announced: native_stream_call_ids,
        }))
    }

    /// The response failed after streaming `text` and `reasoning` bytes of
    /// reasoning: what is still buffered isn't sent, and the host is told to
    /// drop what was ([`AgentEvent::StreamedOutputDiscarded`]).
    async fn discard_streamed(
        &mut self,
        text: &str,
        reasoning: usize,
    ) -> Result<(), ArkoAgentsError> {
        // Output-based text goes to the parser, not to the host, as it comes.
        let text_bytes = if self.is_native_mode() {
            text.len() - self.text_buffer.len()
        } else {
            0
        };
        let reasoning_bytes = reasoning - self.reasoning_buffer.len();
        self.text_buffer.clear();
        self.reasoning_buffer.clear();
        if text_bytes > 0 || reasoning_bytes > 0 {
            self.emit_event(AgentEvent::StreamedOutputDiscarded {
                text_bytes,
                reasoning_bytes,
            })
            .await?;
        }
        Ok(())
    }

    /// A command that arrived while a response streamed. `CancelTask`,
    /// `SuspendTask` or a closed channel stops the stream: the partial
    /// text is kept, the calls it announced are rejected, and `true` says
    /// the pass ends. Context and handover requests are taken; other
    /// commands are ignored.
    async fn on_stream_command(
        &mut self,
        maybe_cmd: Option<UserCommand>,
        streamed_text: &str,
        native_stream_call_ids: &mut HashMap<usize, (Uuid, String)>,
    ) -> Result<bool, ArkoAgentsError> {
        match maybe_cmd.map(|command| self.absorb_command(command)) {
            Some(Some(UserCommand::CancelTask)) => {
                self.keep_partial_output(streamed_text, native_stream_call_ids)
                    .await?;
                self.task.set_state(TaskState::Cancelled {
                    reason: CANCELLED_BY_USER.to_string(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                return Ok(true);
            }
            Some(Some(UserCommand::SuspendTask)) => {
                self.keep_partial_output(streamed_text, native_stream_call_ids)
                    .await?;
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                self.stop_requested = true;
                return Ok(true);
            }
            Some(_) => {}
            None => {
                self.keep_partial_output(streamed_text, native_stream_call_ids)
                    .await?;
                self.stop_requested = true;
                self.interrupt("Command channel disconnected during LLM streaming.".to_string())
                    .await?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// A stream the host stopped (before its final response): the text it
    /// delivered is kept as the turn, and the calls it announced are
    /// rejected.
    async fn keep_partial_output(
        &mut self,
        text: &str,
        native_stream_call_ids: &mut HashMap<usize, (Uuid, String)>,
    ) -> Result<(), ArkoAgentsError> {
        if !text.is_empty() {
            self.task.conversation.add_assistant_message(text);
        }
        self.reject_announced_calls(native_stream_call_ids, INTERRUPTED_CALL)
            .await
    }

    /// A response's `End`: its usage and the request it answered are
    /// recorded and reported, then a refused turn is handled as a refusal
    /// (`true`: the pass ends), and any other turn joins the conversation
    /// as the final response.
    async fn on_response_end(
        &mut self,
        mut generation_response: ai::GenerationResponse,
        parser: &mut Option<ToolCallParser>,
        raw_events_for_turn: &mut Vec<RawParserEvent>,
        native_stream_call_ids: &mut HashMap<usize, (Uuid, String)>,
        final_generation_response: &mut Option<ai::GenerationResponse>,
    ) -> Result<bool, ArkoAgentsError> {
        if self.current_request_is_fallback {
            self.prepare_fallback_response(&mut generation_response);
        }
        self.task.record_usage(generation_response.usage.clone());
        // The request was answered: its usage calibrates
        // the estimate of the next one.
        self.answered_request = self.sent_request.take();
        if let Some(ratio) = self.answered_request.and_then(|mark| {
            result_budget::calibrated_ratio(mark.bytes, generation_response.usage.input_tokens)
        }) {
            self.bytes_per_token = ratio;
        }
        self.overflow_recoveries = 0;
        self.emit_event(AgentEvent::UsageReported(generation_response.usage.clone()))
            .await?;
        self.record_and_emit_model_call_usage(generation_response.model_call_usage.clone())
            .await?;
        if generation_response.stop_reason == ai::StopReason::MaxTokens {
            let usage = &generation_response.model_call_usage;
            log::warn!(
                "Task {}: the response of {} ended at its output-token limit.",
                self.task.id,
                usage.model
            );
            self.emit_event(AgentEvent::OutputTokenLimitReached {
                model: usage
                    .served_model
                    .clone()
                    .unwrap_or_else(|| usage.model.clone()),
                output_tokens: generation_response.usage.output_tokens,
            })
            .await?;
        }
        // A refused turn is reported, never retried
        // unchanged, and never treated as a turn that
        // failed to call a tool.
        if let Some(refusal) = generation_response.refusal.clone() {
            // The call's routing token still belongs
            // to this turn, as for any other response.
            if let Some(token) = &generation_response.openai_turn_state {
                self.task.conversation.record_openai_turn_state(token);
            }
            if let Some(ref mut p) = *parser {
                raw_events_for_turn.extend(p.finalize());
            }
            let plain_text = parser
                .is_some()
                .then(|| parsed_plain_text(raw_events_for_turn));
            self.reject_announced_calls(native_stream_call_ids, REFUSED_TURN_CALL)
                .await?;
            self.handle_refusal(refusal, generation_response.message, plain_text)
                .await?;
            return Ok(true);
        }
        // A completed turn ends any fallback after a
        // refusal and answers any refused input.
        self.task.fallback_for = None;
        self.task.refused_input = None;
        // The provider answered: an outage is over, so
        // its elapsed budget no longer runs (a response
        // the runtime can't use still counts as an
        // attempt).
        self.recovery.recovery_started = None;
        self.recovery.paused_recovery_elapsed = None;
        self.task
            .conversation
            .add_assistant_message_from_response(generation_response.clone());
        *final_generation_response = Some(generation_response);
        if let Some(ref mut p) = *parser {
            raw_events_for_turn.extend(p.finalize());
        }
        Ok(false)
    }

    /// Reports calls a stream announced (`ToolCallInitiated`) that will not
    /// run because the turn ended without them, so every announced call gets
    /// an outcome.
    pub(super) async fn reject_announced_calls(
        &self,
        announced: &mut HashMap<usize, (Uuid, String)>,
        reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        let mut calls: Vec<_> = announced.drain().collect();
        calls.sort_by_key(|(index, _)| *index);
        for (_, (tool_call_id, tool_name)) in calls {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name,
                reason: Some(reason.to_string()),
            })
            .await?;
        }
        Ok(())
    }
}
