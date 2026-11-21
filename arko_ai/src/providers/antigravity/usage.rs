//! The plan's usage limits, as `agy -p /usage --output-format json` reports
//! them in about two seconds, with no model call.
//!
//! Models share limits in groups, such as "Gemini Models" and "Claude and GPT
//! models", and each group has a weekly window and often a five-hour one.

use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use super::locate::{Launch, RunError, Scratch};
use super::{UsageGroup, group_for_model};
use crate::usage::{PlanUsage, SubscriptionPlan};

/// How long `agy -p /usage` may take.
const USAGE_TIMEOUT: Duration = Duration::from_secs(30);

/// Reads the plan's usage groups. `env` is set on top of the inherited
/// environment. agy runs in an empty temporary folder, so no project's
/// settings or hooks load, and what it leaves running is killed once it
/// exits.
pub async fn plan_usage(cli: &Path, env: &[(String, String)]) -> Result<Vec<UsageGroup>, String> {
    let mut launch = Launch::new(cli);
    launch.env = env.to_vec();
    read_usage(&launch).await
}

async fn read_usage(launch: &Launch) -> Result<Vec<UsageGroup>, String> {
    let scratch = Scratch::create("usage")
        .await
        .map_err(|error| format!("Couldn't make a temporary folder. {error}"))?;
    let output = launch
        .output(
            &["-p", "/usage", "--output-format", "json"],
            Some(scratch.path()),
            USAGE_TIMEOUT,
        )
        .await
        .map_err(|error| match error {
            RunError::NotFound => "The Antigravity CLI was not found.".to_string(),
            RunError::Failed(message) => message,
        })?;
    drop(scratch);
    if !output.status.success() {
        return Err(output.failure("agy -p /usage"));
    }
    parse_usage(&output.stdout)
}

#[derive(Deserialize)]
struct Report {
    command: Command,
}

#[derive(Deserialize)]
struct Command {
    data: Data,
}

#[derive(Deserialize)]
struct Data {
    groups: Vec<UsageGroup>,
}

/// Reads the groups from what `agy -p /usage --output-format json` printed
/// (`command.data.groups`).
pub fn parse_usage(json: &str) -> Result<Vec<UsageGroup>, String> {
    serde_json::from_str::<Report>(json.trim())
        .map(|report| report.command.data.groups)
        .map_err(|error| format!("`agy -p /usage` printed something unexpected. {error}"))
}

/// A report of the Antigravity plan with all its groups. `active_limit`
/// names the group of the model shown as `display_name` (such as "Gemini 4
/// Neon"), when one has it.
pub fn plan_usage_report(groups: Vec<UsageGroup>, display_name: Option<&str>) -> PlanUsage {
    let active_limit = display_name
        .and_then(|name| group_for_model(&groups, name))
        .map(|group| group.name.clone());
    PlanUsage {
        plan: Some(SubscriptionPlan::Antigravity),
        active_limit,
        antigravity_groups: groups,
        ..PlanUsage::default()
    }
}

#[cfg(test)]
mod tests {
    use super::super::UsageBucket;
    use super::super::acp::tests::{Fake, timed};
    use super::super::locate::tests::fake_launch;
    use super::*;

    /// What agy 1.5.14 printed on a Google AI Pro plan.
    const SAMPLE: &str = r#"{"conversation_id":"","status":"SUCCESS","response":"Gemini Models\tWeekly Limit Remaining\t97%\t2026-11-28T19:38:35Z\nGemini Models\tFive Hour Limit Remaining\t91%\t2026-11-22T00:38:35Z\nClaude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-28T21:21:32Z\nClaude and GPT models\tFive Hour Limit Remaining\t100%\t2026-11-22T02:21:32Z\nGemini 4 Argon\tGemini 4 Argon Weekly Limit\t100%\t2026-11-28T21:21:32Z\n","duration_seconds":0,"num_turns":0,"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":0},"command":{"name":"usage","data":{"description":"Within each group, models share a weekly limit and a 5-hour limit. Quota is consumed proportionally to the cost of the tokens. Thus, limits will last longer with shorter tasks or using more cost-effective models. The 5-hour limit smooths out aggregate demand to fairly distribute global capacity across all users, while your weekly limit is tied directly to your individual tier.","groups":[{"name":"Gemini Models","description":"Models within this group: Gemini 4 Neon, Gemini 4 Hydrogen, Orchestrator (Neon), Gemini 3.8 Flash","buckets":[{"id":"gemini-weekly","name":"Weekly Limit Remaining","description":"You have used some of your weekly limit, it will fully refresh in 6 days, 22 hours.","window":"weekly","remaining_fraction":0.9698070883750916,"reset_time":"2026-11-28T19:38:35Z"},{"id":"gemini-5h","name":"Five Hour Limit Remaining","description":"You have used some of your 5-hour limit, it will fully refresh in 3 hours, 17 minutes.","window":"5h","remaining_fraction":0.9143246412277222,"reset_time":"2026-11-22T00:38:35Z"}]},{"name":"Claude and GPT models","description":"Models within this group: Claude Opus 5.5, Claude Sonnet 5.5, GPT-OSS 120B","buckets":[{"id":"3p-weekly","name":"Weekly Limit Remaining","window":"weekly","remaining_fraction":1,"reset_time":"2026-11-28T21:21:32Z"},{"id":"3p-5h","name":"Five Hour Limit Remaining","window":"5h","remaining_fraction":1,"reset_time":"2026-11-22T02:21:32Z"}]},{"name":"Gemini 4 Argon","description":"For paid Pro plans, Argon quota is separate. Higher Argon quota (included as part of the larger Gemini quota bucket) is available on Ultra plans.","buckets":[{"id":"gemini-argon-wtu-weekly","name":"Gemini 4 Argon Weekly Limit","window":"weekly","remaining_fraction":1,"reset_time":"2026-11-28T21:21:32Z"}]}]}}}"#;

    #[test]
    fn the_groups_are_read_with_their_windows() {
        let groups = parse_usage(SAMPLE).unwrap();
        let names: Vec<&str> = groups.iter().map(|group| group.name.as_str()).collect();
        assert_eq!(
            names,
            ["Gemini Models", "Claude and GPT models", "Gemini 4 Argon"]
        );
        assert_eq!(
            groups[0].buckets[1],
            UsageBucket {
                id: "gemini-5h".into(),
                name: "Five Hour Limit Remaining".into(),
                window: "5h".into(),
                remaining_fraction: 0.9143246412277222,
                reset_time: "2026-11-22T00:38:35Z".into(),
            }
        );
        assert_eq!(groups[1].buckets[0].remaining_fraction, 1.0);
        assert_eq!(groups[2].buckets.len(), 1);
        assert_eq!(groups[2].buckets[0].window, "weekly");
        assert!(parse_usage("Usage isn't available.").is_err());
        assert!(parse_usage(r#"{"status": "SUCCESS"}"#).is_err());
    }

    #[test]
    fn the_report_names_the_models_group() {
        let groups = parse_usage(SAMPLE).unwrap();
        let report = plan_usage_report(groups.clone(), Some("Claude Opus 5.5"));
        assert_eq!(report.plan, Some(SubscriptionPlan::Antigravity));
        assert_eq!(
            report.active_limit.as_deref(),
            Some("Claude and GPT models")
        );
        assert_eq!(report.antigravity_groups, groups);
        assert_eq!(report.primary, None);
        assert_eq!(report.secondary, None);
        assert_eq!(
            plan_usage_report(groups.clone(), Some("Gemini 4 Argon"))
                .active_limit
                .as_deref(),
            Some("Gemini 4 Argon")
        );
        assert_eq!(
            plan_usage_report(groups.clone(), Some("Gemini 4 Neon"))
                .active_limit
                .as_deref(),
            Some("Gemini Models")
        );
        assert_eq!(plan_usage_report(groups, None).active_limit, None);
    }

    #[tokio::test]
    async fn usage_runs_agy_in_an_empty_folder() {
        let fake = Fake::new();
        let sample = fake.dir.path().join("usage.json");
        std::fs::write(&sample, SAMPLE).unwrap();
        let mut launch = fake_launch(&fake);
        launch
            .env
            .push(("FAKE_AGY_USAGE_FILE".into(), sample.display().to_string()));
        let groups = timed(read_usage(&launch)).await.unwrap();
        assert_eq!(groups, parse_usage(SAMPLE).unwrap());
        let record = &fake.records()[0];
        assert_eq!(
            record["args"],
            serde_json::json!(["-p", "/usage", "--output-format", "json"])
        );
        assert_eq!(record["no_update"], "true");
        let cwd = Path::new(record["cwd"].as_str().unwrap());
        let folder = cwd.file_name().unwrap().to_string_lossy();
        assert!(folder.starts_with("agy-usage-"), "{}", cwd.display());

        launch.env.push(("FAKE_AGY_USAGE_EXIT".into(), "2".into()));
        let error = timed(read_usage(&launch)).await.unwrap_err();
        assert!(error.contains("agy -p /usage"), "{error}");
    }
}
