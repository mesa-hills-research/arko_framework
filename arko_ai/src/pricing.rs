//! Price variants that depend on how a request was processed.
//!
//! [`ModelPricing`] holds a model's standard list prices. The types here
//! describe documented modifiers for processing tiers ([`TierPricing`]), data
//! residency ([`ResidencyPricing`]) and dated price changes
//! ([`ScheduledPricing`]). [`UsageInfo::calculate_cost`] applies them using
//! the values a provider reports for the call (falling back to what was
//! requested), so the estimate reflects what was actually applied.
//!
//! [`UsageInfo::calculate_cost`]: crate::usage::UsageInfo::calculate_cost

use crate::model::ModelPricing;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// How a request was processed, when that affects its price.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    /// Standard processing (OpenAI `default`/`auto`, Anthropic `standard`).
    Standard,
    /// Lower-priority, discounted processing (OpenAI `flex`, Gemini `flex`).
    Flex,
    /// Premium low-latency processing through OpenAI Fast mode (formerly Priority,
    /// requested or echoed as `priority` or `fast`) and Anthropic fast mode
    /// (`speed: "fast"`).
    Fast,
    /// Anthropic's commitment-based priority tier and Gemini `priority`.
    Priority,
    /// OpenAI Scale tier (prepaid token units).
    Scale,
    /// Batch processing.
    Batch,
    /// A provider value with no normalized equivalent, kept verbatim.
    Other(String),
}

impl ServiceTier {
    /// Normalizes an OpenAI Responses `service_tier` value.
    /// <https://developers.openai.com/api/reference/resources/responses/methods/create>
    pub fn from_openai(value: &str) -> Self {
        match value {
            "default" | "auto" => Self::Standard,
            "flex" => Self::Flex,
            "priority" | "fast" => Self::Fast,
            "scale" => Self::Scale,
            other => Self::Other(other.to_string()),
        }
    }

    /// Normalizes Anthropic `usage.service_tier` / `usage.speed`. Fast mode is
    /// a speed option, reported separately from the service tier.
    /// <https://platform.claude.com/docs/en/build-with-claude/fast-mode>
    pub fn from_anthropic(service_tier: Option<&str>, speed: Option<&str>) -> Option<Self> {
        if speed == Some("fast") {
            return Some(Self::Fast);
        }
        service_tier.map(|tier| match tier {
            "standard" => Self::Standard,
            "priority" => Self::Priority,
            "batch" => Self::Batch,
            other => Self::Other(other.to_string()),
        })
    }
}

/// Price of a processing tier relative to standard rates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierPricing {
    pub tier: ServiceTier,
    /// Multiplier applied to every standard per-token rate (input, cached
    /// input, cache writes and output).
    pub multiplier: f64,
    /// Whether the tier also applies above the long-context threshold. When
    /// false, long-context requests are priced at standard long-context
    /// rates (for example OpenAI Fast mode on GPT-5.4/5.5).
    #[serde(default = "default_true")]
    pub applies_to_long_context: bool,
}

fn default_true() -> bool {
    true
}

impl TierPricing {
    pub fn new(tier: ServiceTier, multiplier: f64) -> Self {
        Self {
            tier,
            multiplier,
            applies_to_long_context: true,
        }
    }

    pub fn flex(multiplier: f64) -> Self {
        Self::new(ServiceTier::Flex, multiplier)
    }

    pub fn fast(multiplier: f64) -> Self {
        Self::new(ServiceTier::Fast, multiplier)
    }

    /// Long-context requests in this tier are priced at standard rates.
    pub fn short_context_only(mut self) -> Self {
        self.applies_to_long_context = false;
        self
    }
}

/// Multiplier for a data-residency option, matched against the region a
/// provider reports (Anthropic `usage.inference_geo`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidencyPricing {
    pub region: String,
    pub multiplier: f64,
}

impl ResidencyPricing {
    pub fn new(region: impl Into<String>, multiplier: f64) -> Self {
        Self {
            region: region.into(),
            multiplier,
        }
    }
}

/// Prices that take effect on a date. The base [`ModelPricing`] applies
/// before the earliest scheduled date.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledPricing {
    /// First day (UTC, `YYYY-MM-DD`) on which `pricing` applies.
    pub effective_from: String,
    /// Complete prices for the period. Empty tier or residency lists inherit
    /// the base pricing's lists.
    pub pricing: ModelPricing,
}

impl ScheduledPricing {
    pub fn new(effective_from: impl Into<String>, pricing: ModelPricing) -> Self {
        Self {
            effective_from: effective_from.into(),
            pricing,
        }
    }

    /// Unix seconds at 00:00 UTC on `effective_from`, if it is a valid date.
    pub fn effective_from_unix(&self) -> Option<i64> {
        parse_utc_date(&self.effective_from)
    }
}

impl ModelPricing {
    /// The prices in effect at `at`, from the latest scheduled period that has
    /// started, else the base prices.
    pub fn at(&self, at: SystemTime) -> &ModelPricing {
        let now = unix_seconds(at);
        self.schedule
            .iter()
            .filter_map(|period| Some((period.effective_from_unix()?, period)))
            .filter(|(start, _)| *start <= now)
            .max_by_key(|(start, _)| *start)
            .map_or(self, |(_, period)| &period.pricing)
    }

    /// Multiplier for `tier`. Standard, unknown and unpriced tiers (for
    /// example Anthropic priority, OpenAI scale) use standard rates.
    pub fn tier_multiplier(&self, tier: Option<&ServiceTier>, long_context: bool) -> f64 {
        let Some(tier) = tier else {
            return 1.0;
        };
        self.service_tiers
            .iter()
            .find(|pricing| &pricing.tier == tier)
            .filter(|pricing| !long_context || pricing.applies_to_long_context)
            .map_or(1.0, |pricing| pricing.multiplier)
    }

    /// Whether requests may select `tier` for this model.
    pub fn offers_tier(&self, tier: &ServiceTier) -> bool {
        self.service_tiers
            .iter()
            .any(|pricing| &pricing.tier == tier)
    }

    /// Multiplier for a reported residency region. Other regions (such as
    /// the default `global`) use standard rates.
    pub fn residency_multiplier(&self, region: Option<&str>) -> f64 {
        let Some(region) = region else {
            return 1.0;
        };
        self.residency
            .iter()
            .find(|pricing| pricing.region == region)
            .map_or(1.0, |pricing| pricing.multiplier)
    }
}

fn unix_seconds(at: SystemTime) -> i64 {
    match at.duration_since(UNIX_EPOCH) {
        Ok(elapsed) => i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX),
        Err(before) => -i64::try_from(before.duration().as_secs()).unwrap_or(i64::MAX),
    }
}

/// Parses `YYYY-MM-DD` into Unix seconds at 00:00 UTC.
fn parse_utc_date(value: &str) -> Option<i64> {
    let mut parts = value.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Days from civil (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at_date(date: &str) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(parse_utc_date(date).unwrap() as u64)
    }

    #[test]
    fn dates_parse_to_utc_midnight_and_reject_malformed_values() {
        assert_eq!(parse_utc_date("1970-01-01"), Some(0));
        assert_eq!(parse_utc_date("2027-01-01"), Some(1_798_761_600));
        assert_eq!(parse_utc_date("2024-02-29"), Some(1_709_164_800));
        for bad in [
            "2027-13-01",
            "2027-01",
            "2027-01-01-01",
            "soon",
            "2027-01-32",
        ] {
            assert_eq!(parse_utc_date(bad), None, "{bad}");
        }
    }

    #[test]
    fn schedules_switch_on_their_effective_date() {
        let pricing = ModelPricing {
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
        let before = at_date("2027-01-01") - Duration::from_secs(1);
        assert_eq!(pricing.at(before).input_cost_per_million, 0.75);
        assert_eq!(
            pricing.at(at_date("2027-01-01")).input_cost_per_million,
            1.50
        );
        assert_eq!(
            pricing.at(at_date("2030-06-01")).output_cost_per_million,
            7.50
        );
    }

    #[test]
    fn provider_tier_values_normalize() {
        assert_eq!(ServiceTier::from_openai("priority"), ServiceTier::Fast);
        assert_eq!(ServiceTier::from_openai("fast"), ServiceTier::Fast);
        assert_eq!(ServiceTier::from_openai("default"), ServiceTier::Standard);
        assert_eq!(ServiceTier::from_openai("flex"), ServiceTier::Flex);
        assert_eq!(
            ServiceTier::from_openai("ultrafast"),
            ServiceTier::Other("ultrafast".into())
        );
        assert_eq!(
            ServiceTier::from_anthropic(Some("standard"), Some("fast")),
            Some(ServiceTier::Fast)
        );
        assert_eq!(
            ServiceTier::from_anthropic(Some("priority"), None),
            Some(ServiceTier::Priority)
        );
        assert_eq!(ServiceTier::from_anthropic(None, None), None);
    }
}
