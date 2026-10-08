//! The opt-in monitors that pause or slow a task before its next request (the
//! prompt-cache monitor, the plan-usage threshold) and the host's hold on a
//! request.

use super::*;

impl AgentLoop {
    /// The opt-in cache monitor ([`MonitoringConfig::cache`](crate::agent_config::MonitoringConfig::cache)),
    /// after a call's usage is recorded.
    pub(super) async fn monitor_cache(&mut self) -> Result<(), ArkoAgentsError> {
        let Some(config) = self.task.agent_config.monitoring.cache.clone() else {
            return Ok(());
        };
        // Calls to the refusal fallback model use another cache: they are
        // not judged, and later calls compare with the configured model's.
        if self.task.fallback_calls.last()
            == Some(&(self.task.usage_history.len().saturating_sub(1)))
        {
            return Ok(());
        }
        let configured: Vec<ai::UsageInfo> = self
            .task
            .usage_history
            .iter()
            .enumerate()
            .skip(self.task.conversation_usage_start)
            .filter(|(index, _)| !self.task.fallback_calls.contains(index))
            .map(|(_, usage)| usage.clone())
            .collect();
        let Some(efficiency) = super::super::monitor::cache_efficiency(&configured, &config) else {
            return Ok(());
        };
        self.emit_event(AgentEvent::CacheEfficiencyReported {
            call_efficiency: efficiency.call,
            rolling_efficiency: efficiency.rolling,
            calls: efficiency.calls,
        })
        .await?;
        if efficiency.calls < config.window_calls {
            return Ok(());
        }
        let low =
            |threshold: Option<f64>| threshold.filter(|threshold| efficiency.rolling < *threshold);
        match low(config.warn_below) {
            Some(threshold) if !self.cache_warned => {
                self.cache_warned = true;
                self.emit_event(AgentEvent::CacheEfficiencyLow {
                    rolling_efficiency: efficiency.rolling,
                    threshold,
                    pausing: false,
                })
                .await?;
            }
            Some(_) => {}
            None => self.cache_warned = false,
        }
        match low(config.pause_below) {
            Some(threshold) if !self.cache_pause_episode => {
                self.cache_pause_episode = true;
                self.cache_pause_pending = Some(format!(
                    "Prompt cache efficiency fell to {:.1}% over the last {} calls (threshold {:.1}%).",
                    efficiency.rolling * 100.0,
                    efficiency.calls,
                    threshold * 100.0
                ));
                self.emit_event(AgentEvent::CacheEfficiencyLow {
                    rolling_efficiency: efficiency.rolling,
                    threshold,
                    pausing: true,
                })
                .await?;
            }
            Some(_) => {}
            None => self.cache_pause_episode = false,
        }
        Ok(())
    }

    /// Pauses for a monitor (like exhausted recovery), marking the pause so
    /// that approving it adds no error note to the conversation. A fallback
    /// still to be made survives the pause.
    pub(super) async fn finish_monitor_pause(
        &mut self,
        message: String,
    ) -> Result<(), ArkoAgentsError> {
        self.finish_llm_recovery(message, None, ErrorRecoveryCause::Monitor, None)
            .await?;
        self.task.monitor_pause = matches!(
            self.task.current_state,
            TaskState::AwaitingErrorRecoveryApproval { .. }
        );
        Ok(())
    }

    /// Holds the next model request for the host
    /// ([`TurnContextSource::hold_request`](super::super::TurnContextSource::hold_request)):
    /// the task pauses for error-recovery approval, whatever the error
    /// policy, marked as a monitor pause is, so that approving it adds no
    /// error note. A fallback still to be made survives the pause.
    pub(super) async fn hold_request(&mut self, reason: String) -> Result<(), ArkoAgentsError> {
        log::info!(
            "Task {}: the host holds the next request: {reason}",
            self.task.id
        );
        self.task
            .set_state(TaskState::AwaitingErrorRecoveryApproval {
                error_message: reason.clone(),
                cause: ErrorRecoveryCause::BudgetExceeded,
                error_kind: None,
            });
        self.task.monitor_pause = true;
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: reason,
            requires_approval_to_retry: true,
            tool_call_id: None,
            error_kind: None,
        })
        .await
    }

    /// Before a model request: applies the opt-in plan usage threshold
    /// ([`MonitoringConfig::plan_usage`](crate::agent_config::MonitoringConfig::plan_usage))
    /// to the most recent report. Returns whether to make the request.
    pub(super) async fn apply_plan_usage_threshold(&mut self) -> Result<bool, ArkoAgentsError> {
        let Some(threshold) = self.task.agent_config.monitoring.plan_usage.clone() else {
            return Ok(true);
        };
        let Some(usage) = self
            .task
            .model_call_usage_history
            .last()
            .and_then(|call| call.plan_usage.clone())
        else {
            return Ok(true);
        };
        let Some((window, reached)) = usage
            .most_used_window()
            .filter(|(_, window)| window.used_percent >= threshold.percent)
        else {
            return Ok(true);
        };
        // A report whose window has reset since (a task resumed later) is
        // no longer current.
        let now_seconds = unix_millis_now() / 1000;
        if reached
            .resets_at
            .is_some_and(|resets_at| resets_at <= now_seconds)
        {
            return Ok(true);
        }
        let mark = PlanUsageWindowMark {
            window: window.to_string(),
            resets_at: reached.resets_at,
        };
        let first = self.task.plan_usage_threshold_reported.as_ref() != Some(&mark);
        if first {
            self.task.plan_usage_threshold_reported = Some(mark);
            self.emit_event(AgentEvent::PlanUsageThresholdReached {
                window: window.to_string(),
                used_percent: reached.used_percent,
                threshold_percent: threshold.percent,
                resets_at_unix_ms: reached
                    .resets_at
                    .map(|seconds| seconds.saturating_mul(1000)),
                action: threshold.action.clone(),
            })
            .await?;
        }
        match threshold.action {
            PlanUsageAction::Pause if first => {
                let message = format!(
                    "Plan usage reached {:.1}% of the {window} usage-limit window (threshold {:.1}%).",
                    reached.used_percent, threshold.percent
                );
                self.finish_monitor_pause(message).await?;
                Ok(false)
            }
            PlanUsageAction::Pause => Ok(true),
            PlanUsageAction::SlowDown { delay_ms } => self.slow_down(delay_ms).await,
        }
    }

    /// The plan-usage slow-down: waits `delay_ms` before the request,
    /// taking commands (a stop ends the wait and the request), then takes
    /// the commands queued meanwhile as a turn boundary does, and the
    /// context sent meanwhile joins this request. Returns whether to make
    /// it.
    async fn slow_down(&mut self, delay_ms: u64) -> Result<bool, ArkoAgentsError> {
        if !std::mem::replace(&mut self.slow_down_cache_checked, true) {
            let lifetime = self
                .llm_client
                .prompt_cache_lifetime(
                    &self.task.agent_config.llm_config.model,
                    &self.task.agent_config.llm_config.generation_config,
                )
                .await;
            super::super::monitor::warn_if_slow_down_outlasts_cache(
                self.task.id,
                delay_ms,
                lifetime,
            );
        }
        let Some(until) = tokio::time::Instant::now().checked_add(Duration::from_millis(delay_ms))
        else {
            return Ok(true);
        };
        let sleep = tokio::time::sleep_until(until);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(false); }
                }
                _ = &mut sleep => break,
            }
        }
        // The wait came after the boundary: commands sent meanwhile
        // are taken as there, and context sent meanwhile joins this
        // request rather than the next, under the boundary's rule
        // (only a pending fallback holds it back).
        if self.take_queued_commands().await? {
            return Ok(false);
        }
        if self.task.fallback_for.is_none() {
            self.add_taken_context()?;
        }
        Ok(true)
    }
}
