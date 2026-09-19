use crate::agent_config::LLMRetryConfig;
use std::time::Duration;

/// `sample` is supplied separately so timing tests need no random generator.
pub(super) fn retry_delay(
    config: &LLMRetryConfig,
    attempt: u32,
    server: Option<Duration>,
    sample: u64,
) -> Duration {
    let multiplier = 1_u64
        .checked_shl(attempt.saturating_sub(1))
        .unwrap_or(u64::MAX);
    let cap = config
        .initial_delay_ms
        .saturating_mul(multiplier)
        .min(config.max_delay_ms);
    let millis = if config.jitter {
        let floor = cap.div_ceil(2);
        floor + sample % (cap - floor + 1)
    } else {
        cap
    };
    // Add nonnegative jitter to a server minimum too, so a shared reset
    // does not synchronize all agents. Never retry before that minimum.
    let server_jitter = if config.jitter {
        sample % (cap / 2 + 1)
    } else {
        0
    };
    let minimum = server
        .unwrap_or_default()
        .saturating_add(Duration::from_millis(server_jitter));
    Duration::from_millis(millis).max(minimum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_delay_saturates_and_jitter_stays_in_bounds() {
        let mut config = LLMRetryConfig {
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
    }
}
