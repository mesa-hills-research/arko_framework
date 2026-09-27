//! Prompt-cache efficiency of a conversation; see
//! [`CacheMonitorConfig`](crate::agent_config::CacheMonitorConfig).

use crate::agent_config::CacheMonitorConfig;
use arko_ai::UsageInfo;

/// Cache efficiency after the latest call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CacheEfficiency {
    /// The latest call's efficiency.
    pub call: f64,
    /// Over the last judged calls, up to the window.
    pub rolling: f64,
    /// Judged calls in `rolling`.
    pub calls: usize,
}

/// Cache efficiency after the last call in `history` (the conversation's
/// calls in order), or `None` when that call is not judged: during warm-up,
/// without a cache figure, or without input to compare.
pub(super) fn cache_efficiency(
    history: &[UsageInfo],
    config: &CacheMonitorConfig,
) -> Option<CacheEfficiency> {
    let first_judged = config.warmup_calls.max(1);
    // Tokens read from the cache and the most that could have been.
    let judge = |index: usize| {
        let cached = history[index].cached_tokens?;
        let ceiling = history[index - 1]
            .input_tokens
            .min(history[index].input_tokens);
        (ceiling > 0).then_some((cached, ceiling))
    };
    let last_index = history.len().checked_sub(1)?;
    if last_index < first_judged {
        return None;
    }
    let (call_cached, call_ceiling) = judge(last_index)?;
    let judged: Vec<(usize, usize)> = (first_judged..history.len()).filter_map(judge).collect();
    let window = &judged[judged.len().saturating_sub(config.window_calls.max(1))..];
    let (cached, ceiling) = window
        .iter()
        .fold((0_usize, 0_usize), |(cached, ceiling), (c, m)| {
            (cached.saturating_add(*c), ceiling.saturating_add(*m))
        });
    Some(CacheEfficiency {
        call: call_cached as f64 / call_ceiling as f64,
        rolling: cached as f64 / ceiling as f64,
        calls: window.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(input: usize, cached: Option<usize>) -> UsageInfo {
        UsageInfo {
            cached_tokens: cached,
            ..UsageInfo::new(input, 10)
        }
    }

    #[test]
    fn efficiency_compares_cache_reads_with_the_shared_prefix_after_warm_up() {
        let config = CacheMonitorConfig {
            warmup_calls: 2,
            window_calls: 3,
            ..Default::default()
        };
        let mut history = vec![call(1000, Some(0)), call(2000, Some(900))];
        // Warm-up: the first two calls are not judged.
        assert_eq!(cache_efficiency(&history[..1], &config), None);
        assert_eq!(cache_efficiency(&history, &config), None);
        // min(2000, 3000) = 2000 could have been cached; 1500 was.
        history.push(call(3000, Some(1500)));
        let efficiency = cache_efficiency(&history, &config).unwrap();
        assert_eq!((efficiency.call, efficiency.rolling, efficiency.calls), (0.75, 0.75, 1));
        // A shrinking context caps the ceiling at the smaller input.
        history.push(call(1000, Some(1000)));
        let efficiency = cache_efficiency(&history, &config).unwrap();
        assert_eq!(efficiency.call, 1.0);
        assert_eq!(efficiency.rolling, 2500.0 / 3000.0);
        // Calls without a cache figure are skipped, and the window rolls.
        history.push(call(1200, None));
        assert_eq!(cache_efficiency(&history, &config), None);
        history.push(call(1400, Some(0)));
        history.push(call(1600, Some(1400)));
        let efficiency = cache_efficiency(&history, &config).unwrap();
        assert_eq!(efficiency.calls, 3);
        assert_eq!(efficiency.rolling, (1000.0 + 0.0 + 1400.0) / (1000.0 + 1200.0 + 1400.0));
    }
}
