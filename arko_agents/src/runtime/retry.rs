use crate::agent_config::LLMRetryConfig;
use arko_ai::BackoffPolicy;
use std::time::Duration;

/// Delay before recovery `attempt` under `config`, using the framework's
/// shared [`BackoffPolicy`]. `sample` is supplied separately so timing tests
/// need no random generator.
pub(super) fn retry_delay(
    config: &LLMRetryConfig,
    attempt: u32,
    server: Option<Duration>,
    sample: u64,
) -> Duration {
    BackoffPolicy {
        initial: Duration::from_millis(config.initial_delay_ms),
        max: Duration::from_millis(config.max_delay_ms),
        jitter: config.jitter,
    }
    .delay(attempt, server, sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_delay_saturates_and_jitter_stays_in_bounds() {
        let mut config = LLMRetryConfig {
            initial_delay_ms: 250,
            jitter: false,
            ..Default::default()
        };
        assert_eq!(retry_delay(&config, 1, None, 0), Duration::from_millis(250));
        assert_eq!(retry_delay(&config, 9, None, 0), Duration::from_secs(64));
        assert_eq!(
            retry_delay(&config, u32::MAX, None, 0),
            Duration::from_secs(64)
        );
        config.jitter = true;
        let delays: std::collections::HashSet<_> = (0..1000)
            .map(|n| retry_delay(&config, 4, None, n))
            .collect();
        assert!(delays.len() > 1);
        assert!(
            delays
                .iter()
                .all(|d| *d >= Duration::from_secs(1) && *d <= Duration::from_secs(2))
        );
        assert_eq!(
            retry_delay(&config, 4, Some(Duration::from_secs(60)), 0),
            Duration::from_secs(60)
        );
        let server_delays: std::collections::HashSet<_> = (0..1000)
            .map(|n| retry_delay(&config, 4, Some(Duration::from_secs(60)), n))
            .collect();
        assert!(server_delays.len() > 1);
        assert!(
            server_delays
                .iter()
                .all(|d| *d >= Duration::from_secs(60) && *d <= Duration::from_secs(61))
        );
        // A huge hint saturates instead of overflowing; the recovery loop
        // then fails the retry (checked deadline) rather than panicking.
        let huge = retry_delay(&config, 3, Some(Duration::from_secs(u64::MAX)), u64::MAX);
        assert!(huge >= Duration::from_secs(u64::MAX));
        assert!(tokio::time::Instant::now().checked_add(huge).is_none());
        // Hints parsed from responses are capped and schedule normally.
        let capped = retry_delay(
            &config,
            3,
            Some(arko_ai::retry::MAX_SERVER_DELAY_HINT),
            u64::MAX,
        );
        assert!(tokio::time::Instant::now().checked_add(capped).is_some());
    }
}
