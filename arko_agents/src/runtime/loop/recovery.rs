//! Recovery from a failed model request under the error policy: the attempts
//! and their limits, the waits between them (backoff, capacity and rate-limit
//! waits, usage limits), and the pause or interruption that ends recovery.

use super::*;

/// A usage-limit wait retries this long after the reported reset, plus up
/// to [`USAGE_LIMIT_RESET_JITTER`].
const USAGE_LIMIT_RESET_MARGIN: Duration = Duration::from_secs(1);
const USAGE_LIMIT_RESET_JITTER: Duration = Duration::from_secs(30);

/// How recovery from a failed model request waits.
enum RecoveryWait {
    /// Bounded by the error policy's attempts and the elapsed budget; a
    /// message about the error is added before the retry.
    Ordinary,
    /// Opt-in indefinite capacity recovery (`capacity_retry`).
    Capacity(crate::agent_config::LLMCapacityRetryConfig),
    /// Default recovery from short rate limits and capacity errors
    /// (`rate_limit_retry`).
    RateLimit(crate::agent_config::LLMRateLimitRetryConfig),
    /// A plan usage limit, handled under `usage_limit_policy`.
    UsageLimit,
}

/// The attempts recovery from a failed request makes under `policy`.
fn recovery_max_attempts(policy: LLMErrorPolicy, model_output: bool) -> u32 {
    match policy {
        LLMErrorPolicy::RetryThenInformAgent { max_attempts }
        | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts } => max_attempts.max(1),
        // Responses the runtime can't use are asked again at once and
        // outside the elapsed budget, so they need an attempt limit.
        LLMErrorPolicy::InformAgentAndContinue if model_output => {
            MODEL_OUTPUT_ATTEMPTS_WITHOUT_LIMIT
        }
        LLMErrorPolicy::InformAgentAndContinue => u32::MAX,
        LLMErrorPolicy::PauseAndAwaitApproval => 1,
    }
}

/// What a failed request's recovery reports: the error, the attempt,
/// and how it waits.
fn recovery_message(
    wait: &RecoveryWait,
    attempt: u32,
    max_attempts: u32,
    error: &ai::Error,
    note: &Option<String>,
    fallback: bool,
) -> String {
    let message = match wait {
        RecoveryWait::Capacity(_) => format!(
            "LLM capacity error (attempt {attempt}; waiting until service recovers): {error}"
        ),
        RecoveryWait::RateLimit(_) => {
            let kind = if error.is_capacity_error() {
                "capacity error"
            } else {
                "rate limit"
            };
            format!("LLM {kind} (attempt {attempt}; retrying when the provider allows): {error}")
        }
        RecoveryWait::Ordinary | RecoveryWait::UsageLimit => match note {
            Some(note) => {
                format!("Model response not usable (attempt {attempt}/{max_attempts}): {note}")
            }
            None => format!("LLM error (attempt {attempt}/{max_attempts}): {error}"),
        },
    };
    if fallback {
        format!(
            "The refusal fallback model's request failed; it still has the refused input to answer. {message}"
        )
    } else {
        message
    }
}

/// Why recovery stops at this attempt, with the pause's cause: an error
/// that retrying can't fix, or the attempts used up (not for a wait that
/// doesn't count them).
fn recovery_stop(
    error: &ai::Error,
    indefinite: bool,
    attempt: u32,
    max_attempts: u32,
    model_output: bool,
) -> Option<(&'static str, ErrorRecoveryCause)> {
    if !error.is_retryable() {
        Some((
            "Failure requires a request, credential, or configuration change",
            if matches!(error, ai::Error::Refusal(_)) {
                ErrorRecoveryCause::Refusal
            } else {
                ErrorRecoveryCause::NonRetryable
            },
        ))
    } else if !indefinite && attempt >= max_attempts {
        Some((
            "Retry attempt limit exhausted",
            if model_output {
                ErrorRecoveryCause::UnusableResponses
            } else {
                ErrorRecoveryCause::RetriesExhausted {
                    retry_after_ms: error.retry_after().map(duration_millis),
                }
            },
        ))
    } else {
        None
    }
}

/// The delay before the next attempt: none for a response the runtime
/// can't use, otherwise the backoff (from the provider's delay when it
/// gives one).
fn retry_delay_for(
    wait: &RecoveryWait,
    config: &crate::agent_config::LLMRetryConfig,
    attempt: u32,
    error: &ai::Error,
    model_output: bool,
) -> Duration {
    // A rate limit's own delay is followed as given (plus jitter) rather
    // than grown by backoff; backoff applies when there is none.
    let backoff_attempt = match wait {
        RecoveryWait::RateLimit(_) if error.retry_after().is_some() => 1,
        _ => attempt,
    };
    if model_output {
        Duration::ZERO
    } else {
        super::super::retry::retry_delay(
            config,
            backoff_attempt,
            error.retry_after(),
            ai::retry::random_sample(),
        )
    }
}

pub(super) fn unix_millis_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, duration_millis)
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// `3h 12m`, `12m 5s` or `40s`.
fn describe_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (days, hours, minutes, rest) = (
        seconds / 86_400,
        seconds % 86_400 / 3_600,
        seconds % 3_600 / 60,
        seconds % 60,
    );
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {rest}s")
    } else {
        format!("{rest}s")
    }
}

/// The attempts a response the runtime can't use gets under
/// `LLMErrorPolicy::InformAgentAndContinue`, which has no attempt limit.
pub(super) const MODEL_OUTPUT_ATTEMPTS_WITHOUT_LIMIT: u32 = 10;

async fn wait_for_recovery_deadline(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending::<()>().await,
    }
}

impl AgentLoop {
    pub(super) fn reset_llm_recovery(&mut self) {
        self.llm_interaction_attempts = 0;
        self.recovery_started = None;
        self.capacity_recovery_started = None;
        self.capacity_retry_attempts = 0;
        self.indefinite_interactions = 0;
        self.paused_recovery_elapsed = None;
        self.rate_limit_waited = Duration::ZERO;
    }

    /// Ordinary recovery attempts in this episode: requests made, less those
    /// that ended in a wait that doesn't count against the ordinary limits.
    pub(super) fn ordinary_attempts(&self) -> u32 {
        self.llm_interaction_attempts
            .saturating_sub(self.indefinite_interactions)
            .max(1)
    }

    /// Stops the ordinary elapsed-recovery clock (and its deadline) while an
    /// indefinite wait lasts; see [`resume_ordinary_recovery_clock`](Self::resume_ordinary_recovery_clock).
    pub(super) fn pause_ordinary_recovery_clock(&mut self) {
        if let Some(started) = self.recovery_started.take() {
            self.paused_recovery_elapsed = Some(
                self.paused_recovery_elapsed
                    .unwrap_or_default()
                    .saturating_add(started.elapsed()),
            );
        }
    }

    /// Restarts a paused ordinary recovery clock with the time it had
    /// already used.
    fn resume_ordinary_recovery_clock(&mut self) {
        if self.recovery_started.is_none()
            && let Some(used) = self.paused_recovery_elapsed.take()
        {
            let now = tokio::time::Instant::now();
            self.recovery_started = Some(now.checked_sub(used).unwrap_or(now));
        }
    }

    pub(super) fn recovery_deadline(&self) -> Option<tokio::time::Instant> {
        self.recovery_started.map(|start| {
            start + Duration::from_millis(self.task.agent_config.llm_retry_config.max_elapsed_ms)
        })
    }

    /// Ends recovery: the task pauses for error-recovery approval, with
    /// `cause`, under the pausing error policies, or is interrupted.
    pub(super) async fn finish_llm_recovery(
        &mut self,
        message: String,
        tool_call_id: Option<Uuid>,
        cause: ErrorRecoveryCause,
        error_kind: Option<ai::ErrorKind>,
    ) -> Result<(), ArkoAgentsError> {
        // A pending fallback stays pending: continuing sends its request
        // again, and resolving the refusal drops it.
        self.task.monitor_pause = false;
        let pause = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::PauseAndAwaitApproval
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if pause {
            self.task
                .set_state(TaskState::AwaitingErrorRecoveryApproval {
                    error_message: message.clone(),
                    cause,
                    error_kind,
                });
        } else {
            // Exhaustion must return control to the host rather than starting
            // an unbounded second retry cycle under RetryThenInformAgent.
            self.task.set_state(TaskState::Interrupted {
                reason: message.clone(),
            });
            self.stop_requested = true;
        }
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: pause,
            tool_call_id,
            error_kind,
        })
        .await
    }

    /// How recovery from `error` waits.
    fn recovery_wait(&self, error: &ai::Error) -> RecoveryWait {
        if error.is_usage_limit() {
            return RecoveryWait::UsageLimit;
        }
        let retrying = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::RetryThenInformAgent { .. }
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if !retrying {
            return RecoveryWait::Ordinary;
        }
        let config = &self.task.agent_config.llm_retry_config;
        if error.is_capacity_error()
            && let Some(capacity) = &config.capacity_retry
        {
            return RecoveryWait::Capacity(capacity.clone());
        }
        if let Some(rate_limit) = &config.rate_limit_retry
            && (error.is_rate_limit() || error.is_capacity_error())
        {
            let max_wait = Duration::from_millis(rate_limit.max_wait_ms);
            // A rate limit asking for a longer wait, or rate limits that have
            // kept the task waiting this long in all (with or without
            // delays), are treated as a usage limit.
            let too_long = error.retry_after().is_some_and(|hint| hint > max_wait)
                || (error.is_rate_limit() && self.rate_limit_waited >= max_wait);
            if !too_long {
                return RecoveryWait::RateLimit(rate_limit.clone());
            }
            // A rate limit that resets this far off is a usage limit in all
            // but name.
            if error.is_rate_limit() {
                return RecoveryWait::UsageLimit;
            }
        }
        RecoveryWait::Ordinary
    }

    /// When a usage limit is waited out under
    /// [`UsageLimitPolicy::WaitUntilReset`], the delay before the retry.
    fn usage_limit_wait(&self, error: &ai::Error) -> Option<Duration> {
        let UsageLimitPolicy::WaitUntilReset { max_wait_ms } =
            self.task.agent_config.llm_retry_config.usage_limit_policy
        else {
            return None;
        };
        let reset_in = error.retry_after()?;
        // The whole wait, margin and jitter included, must fit max_wait.
        let longest = reset_in
            .saturating_add(USAGE_LIMIT_RESET_MARGIN)
            .saturating_add(USAGE_LIMIT_RESET_JITTER);
        if longest > Duration::from_millis(max_wait_ms) {
            return None;
        }
        // A little past the reset, spread so that agents sharing a plan do
        // not all retry at once.
        let jitter = if self.task.agent_config.llm_retry_config.jitter {
            ai::retry::random_sample() % (USAGE_LIMIT_RESET_JITTER.as_millis() as u64 + 1)
        } else {
            0
        };
        Some(
            reset_in
                .saturating_add(USAGE_LIMIT_RESET_MARGIN)
                .saturating_add(Duration::from_millis(jitter)),
        )
    }

    /// Recovers from a failed model request (a connection failure, a server
    /// error, a truncated stream, a refusal, ...): under the error policy, the
    /// request is sent again unchanged, without a note in the conversation,
    /// since the model had no part in the failure, or recovery ends.
    pub(super) async fn handle_llm_error(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        self.recover(error, tool_call_id, None).await
    }

    /// Recovers from a response the runtime can't use (no tool call, or an
    /// output-based call it can't read) under the error policy, like a
    /// failed request, but with `note` added to the conversation before the
    /// next request, so the model can do better. It isn't an outage: the
    /// request is made again at once, without backoff, and the elapsed
    /// recovery budget neither starts nor applies; the attempt limit bounds
    /// it.
    pub(super) async fn handle_model_output_failure(
        &mut self,
        note: String,
    ) -> Result<(), ArkoAgentsError> {
        self.recover(ai::Error::ApiError(note.clone()), None, Some(note))
            .await
    }

    /// The attempt a failed request is in its recovery, and when that
    /// recovery started. A wait that doesn't count against ordinary
    /// recovery (capacity, rate limits) stops its elapsed clock and counts
    /// on its own; otherwise the clock resumes (not for a response the
    /// runtime can't use, which isn't an outage).
    fn recovery_attempt(
        &mut self,
        indefinite: bool,
        model_output: bool,
    ) -> (u32, tokio::time::Instant) {
        if indefinite {
            // These waits neither consume nor reset ordinary recovery: its
            // elapsed clock stops while they last, and the requests they
            // make are not counted as its attempts.
            self.pause_ordinary_recovery_clock();
            self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
            let started = *self
                .capacity_recovery_started
                .get_or_insert_with(tokio::time::Instant::now);
            self.capacity_retry_attempts = self.capacity_retry_attempts.saturating_add(1);
            (self.capacity_retry_attempts, started)
        } else {
            // A response the runtime can't use isn't an outage: the elapsed
            // budget stays as it was (paused while the response ran).
            if !model_output {
                self.resume_ordinary_recovery_clock();
            }
            let started = if model_output {
                self.recovery_started
                    .unwrap_or_else(tokio::time::Instant::now)
            } else {
                *self
                    .recovery_started
                    .get_or_insert_with(tokio::time::Instant::now)
            };
            (self.ordinary_attempts(), started)
        }
    }

    async fn recover(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
        note: Option<String>,
    ) -> Result<(), ArkoAgentsError> {
        // A request over the model's input limit is made smaller, not sent
        // again unchanged.
        if note.is_none() && self.recover_from_overflow(&error).await? {
            return Ok(());
        }
        // Recorded with the pause or error that ends recovery.
        let error_kind = Some(error.kind()).filter(|kind| *kind != ai::ErrorKind::Unknown);
        // A failed fallback request (the refusal fallback answering a refused
        // turn) is recovered from like any request, by its own error: a
        // transient failure is retried, to the fallback model, and a pause
        // keeps the fallback pending, so continuing sends it again. Only the
        // fallback's own refusal (see `handle_refusal`) makes the refusal
        // stand.
        let fallback = self.task.fallback_for.is_some() && self.current_request_is_fallback;
        let mut config = self.task.agent_config.llm_retry_config.clone();
        let policy = self.task.agent_config.llm_error_policy.clone();
        let wait = self.recovery_wait(&error);
        if let RecoveryWait::UsageLimit = wait {
            return self.handle_usage_limit(error, tool_call_id).await;
        }
        let indefinite = !matches!(wait, RecoveryWait::Ordinary);
        let model_output = note.is_some();
        let (attempt, started) = self.recovery_attempt(indefinite, model_output);
        match &wait {
            RecoveryWait::Capacity(capacity) => {
                config.initial_delay_ms = capacity.initial_delay_ms;
                config.max_delay_ms = capacity.max_delay_ms;
            }
            RecoveryWait::RateLimit(rate_limit) => {
                config.initial_delay_ms = rate_limit.initial_delay_ms;
                config.max_delay_ms = rate_limit.max_delay_ms;
            }
            RecoveryWait::Ordinary | RecoveryWait::UsageLimit => {}
        }
        let elapsed = started.elapsed();
        let max_attempts = recovery_max_attempts(policy, model_output);
        let message = recovery_message(&wait, attempt, max_attempts, &error, &note, fallback);
        let stop = recovery_stop(&error, indefinite, attempt, max_attempts, model_output);
        if let Some((reason, cause)) = stop {
            return self
                .finish_llm_recovery(
                    format!("{reason}. {message}"),
                    tool_call_id,
                    cause,
                    error_kind,
                )
                .await;
        }
        let delay = retry_delay_for(&wait, &config, attempt, &error, model_output);
        if !indefinite
            && !model_output
            && elapsed.saturating_add(delay) >= Duration::from_millis(config.max_elapsed_ms)
        {
            return self
                .finish_llm_recovery(
                    format!("Next retry cannot fit the elapsed recovery budget. {message}"),
                    tool_call_id,
                    ErrorRecoveryCause::RetriesExhausted {
                        retry_after_ms: Some(duration_millis(delay)),
                    },
                    error_kind,
                )
                .await;
        }
        let Some(retry_at) = tokio::time::Instant::now().checked_add(delay) else {
            return self
                .finish_llm_recovery(
                    format!("Server retry delay cannot be represented safely. {message}"),
                    tool_call_id,
                    ErrorRecoveryCause::RetriesExhausted {
                        retry_after_ms: Some(duration_millis(delay)),
                    },
                    error_kind,
                )
                .await;
        };
        let counts_toward_budget =
            matches!(wait, RecoveryWait::RateLimit(_)) && error.is_rate_limit();
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: false,
            tool_call_id,
            error_kind,
        })
        .await?;
        self.emit_event(AgentEvent::RetryScheduled {
            attempt,
            delay_ms: delay.as_millis() as u64,
            elapsed_ms: elapsed.as_millis() as u64,
            rate_limit: error.is_rate_limit(),
        })
        .await?;
        if let Some(note) = &note {
            self.append_error_recovery_user_message(note)?;
            // Asked again at once, with no budget deadline to meet.
            return Ok(());
        }
        let waiting_since = tokio::time::Instant::now();
        let waited = self.wait_before_retry(retry_at, tool_call_id).await;
        if counts_toward_budget {
            // The time actually waited: a command may have cut it short.
            self.rate_limit_waited = self
                .rate_limit_waited
                .saturating_add(waiting_since.elapsed());
        }
        waited
    }

    /// Handles a plan usage limit under
    /// [`LLMRetryConfig::usage_limit_policy`](crate::agent_config::LLMRetryConfig::usage_limit_policy):
    /// waits for the reset and retries unchanged (opt-in), or stops
    /// retrying.
    async fn handle_usage_limit(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let now_ms = unix_millis_now();
        let reset_in = error.retry_after();
        let resets_at_unix_ms = reset_in.map(|delay| now_ms.saturating_add(duration_millis(delay)));
        let when = match reset_in {
            Some(delay) => format!("resets in about {}", describe_duration(delay)),
            None => "reset time not reported".to_string(),
        };
        let retry_at = self.usage_limit_wait(&error).and_then(|delay| {
            tokio::time::Instant::now()
                .checked_add(delay)
                .map(|at| (delay, at))
        });
        let Some((delay, retry_at)) = retry_at else {
            self.emit_event(AgentEvent::UsageLimitReached {
                resets_at_unix_ms,
                waiting_until_unix_ms: None,
                error_message: error.to_string(),
            })
            .await?;
            return self
                .finish_llm_recovery(
                    format!("Usage limit reached ({when}). {error}"),
                    tool_call_id,
                    ErrorRecoveryCause::UsageLimit { resets_at_unix_ms },
                    None,
                )
                .await;
        };
        // Like the other unchanged retries, the wait doesn't count against
        // ordinary recovery.
        self.pause_ordinary_recovery_clock();
        self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
        self.emit_event(AgentEvent::UsageLimitReached {
            resets_at_unix_ms,
            waiting_until_unix_ms: Some(now_ms.saturating_add(duration_millis(delay))),
            error_message: error.to_string(),
        })
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: format!(
                "Usage limit reached ({when}); waiting {} before retrying. {error}",
                describe_duration(delay)
            ),
            requires_approval_to_retry: false,
            tool_call_id,
            error_kind: None,
        })
        .await?;
        self.wait_before_retry(retry_at, tool_call_id).await
    }

    /// Waits until `retry_at` before a retry, handling commands; the task is
    /// then in progress again. An ordinary recovery deadline that passes
    /// first ends recovery: the elapsed budget limits the waits between
    /// attempts and the decision to make another, never a request in flight.
    async fn wait_before_retry(
        &mut self,
        retry_at: tokio::time::Instant,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let deadline = self.recovery_deadline();
        let sleep = tokio::time::sleep_until(retry_at);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                _ = wait_for_recovery_deadline(deadline) => {
                    let retry_after_ms = Some(duration_millis(retry_at.saturating_duration_since(tokio::time::Instant::now())));
                    return self.finish_llm_recovery("Elapsed model recovery budget exhausted during backoff.".into(), tool_call_id, ErrorRecoveryCause::RetriesExhausted { retry_after_ms }, None).await;
                }
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(()); }
                }
                _ = &mut sleep => {
                    self.task.set_state(TaskState::InProgress);
                    return Ok(());
                }
            }
        }
    }
}
