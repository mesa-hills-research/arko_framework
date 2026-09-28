//! Completion verification by a separately configured workflow agent.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::context::SharedContext;
use crate::freshness::{ActorId, ResourceId};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationKind {
    Correctness,
    /// Efficiency verifiers are advisory: a reported major issue is coerced
    /// to a minor correction and the gate is skipped after its repair cap.
    Efficiency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionVerifierConfig {
    pub name: String,
    pub verifier_agent: String,
    pub target_agents: HashSet<String>,
    pub kind: VerificationKind,
    pub instructions: String,
    pub max_minor_rejections: u32,
    pub max_major_rejections: u32,
    #[serde(default)]
    pub reuse_conversation: bool,
    #[serde(default)]
    pub repair_context: Option<RepairContextConfig>,
    /// A fresh re-review (without `reuse_conversation`) is given the prior
    /// verdict and findings for the same task. Default `true`.
    #[serde(default = "default_true")]
    pub include_prior_findings: bool,
    /// How many times a verdict that can't be parsed is asked for again in
    /// the contract's format before the review counts as failed. Default 1.
    #[serde(default = "default_format_retries")]
    pub max_format_retries: u32,
    /// How many failed reviews (the verifier's run failed, or its verdict
    /// stayed unusable) of one task are answered by asking the agent to call
    /// `complete_task` again, which starts a new review. The next failure
    /// stops the task and the run call returns
    /// [`WorkflowError::VerifierFailed`](crate::WorkflowError::VerifierFailed);
    /// the task itself isn't failed. Default 1.
    #[serde(default = "default_review_retries")]
    pub max_review_retries: u32,
}

fn default_true() -> bool {
    true
}

fn default_format_retries() -> u32 {
    1
}

fn default_review_retries() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepairContextConfig {
    /// Maximum repair-delta text injected into a continued verifier.
    pub max_bytes: usize,
}

impl CompletionVerifierConfig {
    pub fn correctness(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verifier_agent: verifier_agent.into(),
            target_agents: HashSet::new(),
            kind: VerificationKind::Correctness,
            instructions: "Check correctness, completeness, and maintainability. Inspect the actual workspace rather than trusting the candidate summary.".to_string(),
            max_minor_rejections: 3,
            max_major_rejections: 0,
            reuse_conversation: false,
            repair_context: None,
            include_prior_findings: true,
            max_format_retries: default_format_retries(),
            max_review_retries: default_review_retries(),
        }
    }

    pub fn efficiency(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verifier_agent: verifier_agent.into(),
            target_agents: HashSet::new(),
            kind: VerificationKind::Efficiency,
            instructions: "Look for avoidable repeated work, poor asymptotic behavior, unnecessary I/O or API calls, and needless resource use. Do not demand speculative optimization.".to_string(),
            max_minor_rejections: 2,
            max_major_rejections: 0,
            reuse_conversation: false,
            repair_context: None,
            include_prior_findings: true,
            max_format_retries: default_format_retries(),
            max_review_retries: default_review_retries(),
        }
    }

    /// Gate this agent's completions (by registration name). A verifier
    /// needs at least one target: `build` rejects one without.
    pub fn target(mut self, agent: impl Into<String>) -> Self {
        self.target_agents.insert(agent.into());
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    pub fn minor_rejection_cap(mut self, cap: u32) -> Self {
        self.max_minor_rejections = cap;
        self
    }

    pub fn major_rejection_cap(mut self, cap: u32) -> Self {
        self.max_major_rejections = cap;
        self
    }

    /// Continue the same verifier conversation when this target is checked
    /// again during its correction loop.
    pub fn reuse_conversation(mut self, enabled: bool) -> Self {
        self.reuse_conversation = enabled;
        self
    }

    /// Whether a fresh re-review is given the prior verdict and findings
    /// (default `true`); `false` makes every review without
    /// `reuse_conversation` an independent first review.
    pub fn include_prior_findings(mut self, enabled: bool) -> Self {
        self.include_prior_findings = enabled;
        self
    }

    /// How many times an unparseable verdict is asked for again (default 1).
    pub fn format_retries(mut self, retries: u32) -> Self {
        self.max_format_retries = retries;
        self
    }

    /// How many failed reviews of one task are retried by asking the agent
    /// to call `complete_task` again (default 1); see
    /// [`max_review_retries`](Self::max_review_retries).
    pub fn review_retries(mut self, retries: u32) -> Self {
        self.max_review_retries = retries;
        self
    }

    /// Inject a bounded file/memory delta when continuing a verifier.
    pub fn repair_context(mut self, max_bytes: usize) -> Self {
        self.repair_context = Some(RepairContextConfig {
            max_bytes: max_bytes.max(1_024),
        });
        self
    }

    /// Whether the verifier gates `agent`'s completions: one of its named
    /// targets (`build` requires at least one), never its own verifier
    /// agent.
    pub(crate) fn applies_to(&self, agent: &str) -> bool {
        agent != self.verifier_agent && self.target_agents.contains(agent)
    }

    pub(crate) fn efficiency_repair_budget_exhausted(&self, rejected: u32) -> bool {
        self.kind == VerificationKind::Efficiency && rejected >= self.max_minor_rejections
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerificationDecision {
    Approved,
    RejectedMinor,
    RejectedMajor,
}

/// Plain-text protocol completion verifiers use for their final summaries.
///
/// The decision occupies the first non-empty line. Everything after it is
/// opaque feedback, so verifier findings never need JSON or string escaping.
pub const VERIFIER_OUTPUT_CONTRACT: &str = r#"Return only plain text in this format:
REJECTED_MINOR
Specific actionable feedback must begin here and may continue across multiple lines.

The first non-empty line must be exactly one of:
ACCEPTED
REJECTED_MINOR
REJECTED_MAJOR

For ACCEPTED, feedback is optional. REJECTED_MINOR and REJECTED_MAJOR require non-empty actionable
feedback. Do not use JSON, Markdown, or put feedback on the decision line."#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub target_task_id: Uuid,
    pub target_agent: String,
    pub verifier_name: String,
    pub verifier_agent: String,
    pub verifier_task_id: Uuid,
    pub decision: VerificationDecision,
    pub feedback: String,
}

#[derive(Debug)]
pub(crate) struct VerifierResponse {
    pub decision: VerificationDecision,
    pub feedback: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RepairBaseline {
    files: HashMap<PathBuf, FileBaseline>,
    memory_entry_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct FileBaseline {
    exists: bool,
    hash: Option<String>,
    text: Option<String>,
}

impl FileBaseline {
    fn capture(path: &Path, max_text_bytes: usize) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => {
                let hash = Some(crate::digest::sha256_hex(&bytes));
                let text = (bytes.len() <= max_text_bytes)
                    .then(|| String::from_utf8(bytes).ok())
                    .flatten();
                Self {
                    exists: true,
                    hash,
                    text,
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Self {
                exists: false,
                hash: None,
                text: None,
            },
            Err(_) => Self {
                exists: true,
                hash: None,
                text: None,
            },
        }
    }
}

pub(crate) fn capture_repair_baseline(
    context: &SharedContext,
    target: &ActorId,
    verifier: &ActorId,
    max_text_bytes: usize,
) -> RepairBaseline {
    let Some(tracker) = &context.resource_tracker else {
        return RepairBaseline::default();
    };
    let mut resources = tracker.written_resources(target);
    resources.extend(tracker.observed_resources(verifier));
    resources.extend(tracker.all_written_resources());
    resources.sort_by_key(resource_label);
    resources.dedup();

    let mut baseline = RepairBaseline::default();
    for resource in resources {
        match resource {
            ResourceId::File(path) if path.starts_with(&context.workspace) => {
                baseline
                    .files
                    .entry(path.clone())
                    .or_insert_with(|| FileBaseline::capture(&path, max_text_bytes));
            }
            ResourceId::File(_) => {}
            ResourceId::MemoryTier(tier) => {
                if let Ok(entries) = context.memory.entries(&verifier.agent_name, &tier) {
                    baseline.memory_entry_counts.insert(tier, entries.len());
                }
            }
            ResourceId::DirectoryTree(_) | ResourceId::Custom { .. } => {}
        }
    }
    baseline
}

pub(crate) fn render_repair_delta(
    context: &SharedContext,
    verifier_agent: &str,
    baseline: &RepairBaseline,
    max_bytes: usize,
) -> String {
    let mut output = String::from(
        "Tracked changes since the prior verdict follow. This is a bounded navigation aid, not proof; inspect a full resource whenever the delta is truncated or ambiguous.\n",
    );
    let Some(tracker) = &context.resource_tracker else {
        output.push_str("Context freshness is disabled, so no repair delta is available.");
        return output;
    };

    let mut changed = false;
    let mut file_paths = baseline.files.keys().cloned().collect::<HashSet<_>>();
    let mut memory_tiers = baseline
        .memory_entry_counts
        .keys()
        .cloned()
        .collect::<HashSet<_>>();
    for resource in tracker.all_written_resources() {
        match resource {
            ResourceId::File(path) if path.starts_with(&context.workspace) => {
                file_paths.insert(path);
            }
            ResourceId::File(_) => {}
            ResourceId::MemoryTier(tier) => {
                memory_tiers.insert(tier);
            }
            ResourceId::DirectoryTree(_) | ResourceId::Custom { .. } => {}
        }
    }

    let mut file_paths = file_paths.into_iter().collect::<Vec<_>>();
    file_paths.sort();
    for path in file_paths {
        let before = baseline.files.get(&path).cloned().unwrap_or(FileBaseline {
            exists: false,
            hash: None,
            text: None,
        });
        let after = FileBaseline::capture(&path, max_bytes);
        if before.exists == after.exists && before.hash == after.hash {
            continue;
        }
        changed = true;
        let display = path
            .strip_prefix(&context.workspace)
            .unwrap_or(&path)
            .display()
            .to_string();
        let delta = render_file_delta(&display, &before, &after);
        if !append_bounded(&mut output, &format!("\n{delta}"), max_bytes) {
            return output;
        }
    }

    let mut memory_tiers = memory_tiers.into_iter().collect::<Vec<_>>();
    memory_tiers.sort();
    for tier in memory_tiers {
        let Ok(entries) = context.memory.entries(verifier_agent, &tier) else {
            continue;
        };
        let prior_count = baseline
            .memory_entry_counts
            .get(&tier)
            .copied()
            .unwrap_or_default();
        let additions = entries.iter().skip(prior_count).collect::<Vec<_>>();
        if additions.is_empty() {
            continue;
        }
        changed = true;
        if !append_bounded(
            &mut output,
            &format!("\nMemory tier `{tier}` received these new entries:\n"),
            max_bytes,
        ) {
            return output;
        }
        for entry in additions {
            if !append_bounded(
                &mut output,
                &format!("[author: {}]\n{}\n", entry.author, entry.content),
                max_bytes,
            ) {
                return output;
            }
        }
    }

    if !changed {
        let _ = append_bounded(
            &mut output,
            "\nNo changes made through tracked file or memory write tools were detected.",
            max_bytes,
        );
    }
    output
}

fn resource_label(resource: &ResourceId) -> String {
    match resource {
        ResourceId::File(path) => format!("0:{}", path.display()),
        ResourceId::MemoryTier(tier) => format!("1:{tier}"),
        ResourceId::DirectoryTree(path) => format!("2:{}", path.display()),
        ResourceId::Custom { namespace, key } => format!("3:{namespace}:{key}"),
    }
}

fn render_file_delta(path: &str, before: &FileBaseline, after: &FileBaseline) -> String {
    match (&before.text, &after.text, before.exists, after.exists) {
        (_, Some(after), false, true) => render_text_delta(path, "", after, true, false),
        (Some(before), _, true, false) => render_text_delta(path, before, "", false, true),
        (Some(before), Some(after), true, true) => {
            render_text_delta(path, before, after, false, false)
        }
        _ => format!(
            "Changed `{path}` (text diff unavailable; before sha256={}, after sha256={}).\n",
            before.hash.as_deref().unwrap_or("unavailable"),
            after.hash.as_deref().unwrap_or("unavailable"),
        ),
    }
}

fn render_text_delta(path: &str, before: &str, after: &str, added: bool, deleted: bool) -> String {
    let old_lines = before.lines().collect::<Vec<_>>();
    let new_lines = after.lines().collect::<Vec<_>>();
    let mut prefix = 0;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < old_lines.len().saturating_sub(prefix)
        && suffix < new_lines.len().saturating_sub(prefix)
        && old_lines[old_lines.len() - 1 - suffix] == new_lines[new_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let context_start = prefix.saturating_sub(3);
    let old_change_end = old_lines.len().saturating_sub(suffix);
    let new_change_end = new_lines.len().saturating_sub(suffix);
    let old_end = (old_change_end + 3).min(old_lines.len());
    let new_end = (new_change_end + 3).min(new_lines.len());
    let old_header = if added { "/dev/null" } else { path };
    let new_header = if deleted { "/dev/null" } else { path };
    let mut output = format!(
        "--- {old_header}\n+++ {new_header}\n@@ -{},{} +{},{} @@\n",
        context_start + 1,
        old_end.saturating_sub(context_start),
        context_start + 1,
        new_end.saturating_sub(context_start),
    );
    for line in &old_lines[context_start..prefix] {
        let _ = writeln!(output, " {line}");
    }
    for line in &old_lines[prefix..old_change_end] {
        let _ = writeln!(output, "-{line}");
    }
    for line in &new_lines[prefix..new_change_end] {
        let _ = writeln!(output, "+{line}");
    }
    let trailing_context = old_lines.len().saturating_sub(suffix);
    for line in &old_lines[trailing_context..old_end] {
        let _ = writeln!(output, " {line}");
    }
    output
}

fn append_bounded(output: &mut String, addition: &str, max_bytes: usize) -> bool {
    if output.len().saturating_add(addition.len()) <= max_bytes {
        output.push_str(addition);
        return true;
    }
    let marker = "\n[repair delta truncated; read the listed resources directly]\n";
    let available = max_bytes
        .saturating_sub(output.len())
        .saturating_sub(marker.len());
    let mut end = available.min(addition.len());
    while end > 0 && !addition.is_char_boundary(end) {
        end -= 1;
    }
    output.push_str(&addition[..end]);
    output.push_str(marker);
    false
}

pub(crate) fn parse_response(value: &str) -> Result<VerifierResponse, String> {
    let mut consumed = 0;

    for chunk in value.split_inclusive('\n') {
        let line = chunk.strip_suffix('\n').unwrap_or(chunk);
        let line = line.strip_suffix('\r').unwrap_or(line);

        if line.trim().is_empty() {
            consumed += chunk.len();
            continue;
        }

        let decision = match line.trim() {
            "ACCEPTED" => VerificationDecision::Approved,
            "REJECTED_MINOR" => VerificationDecision::RejectedMinor,
            "REJECTED_MAJOR" => VerificationDecision::RejectedMajor,
            other => {
                return Err(format!(
                    "first non-empty line must be ACCEPTED, REJECTED_MINOR, or REJECTED_MAJOR; got {other:?}"
                ));
            }
        };
        let feedback_start = consumed + chunk.len();
        let feedback = value[feedback_start..].trim().to_string();
        if decision != VerificationDecision::Approved && feedback.is_empty() {
            return Err(format!(
                "{} requires non-empty actionable feedback on the following line",
                line.trim()
            ));
        }
        return Ok(VerifierResponse { decision, feedback });
    }

    Err("verifier response was empty".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arko_agents::{AgentConfig, AgentLLMConfig};
    use arko_ai::{GenerationConfig, LLMClient, Model};

    use crate::freshness::ContextFreshnessConfig;
    use crate::memory::MemoryConfig;
    use crate::workflow::Workflow;

    fn test_agent(id: &str) -> AgentConfig {
        AgentConfig::builder(
            id.to_string(),
            id.to_string(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_luna(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test agent".to_string())
        .instructions("test instructions".to_string())
        .build()
        .unwrap()
    }

    #[test]
    fn efficiency_gate_stops_blocking_at_its_repair_cap() {
        let efficiency =
            CompletionVerifierConfig::efficiency("efficiency", "reviewer").minor_rejection_cap(3);
        assert!(!efficiency.efficiency_repair_budget_exhausted(2));
        assert!(efficiency.efficiency_repair_budget_exhausted(3));

        let correctness =
            CompletionVerifierConfig::correctness("correctness", "reviewer").minor_rejection_cap(3);
        assert!(!correctness.efficiency_repair_budget_exhausted(3));

        let sticky = CompletionVerifierConfig::correctness("sticky", "reviewer")
            .reuse_conversation(true)
            .repair_context(10);
        assert!(sticky.reuse_conversation);
        assert_eq!(sticky.repair_context.unwrap().max_bytes, 1_024);
    }

    /// A configuration saved before the re-review and retry settings loads
    /// with their defaults.
    #[test]
    fn a_configuration_saved_before_the_retry_settings_loads_with_defaults() {
        let mut saved = serde_json::to_value(
            CompletionVerifierConfig::correctness("check", "reviewer").target("coder"),
        )
        .unwrap();
        for field in [
            "include_prior_findings",
            "max_format_retries",
            "max_review_retries",
        ] {
            saved.as_object_mut().unwrap().remove(field);
        }
        let loaded: CompletionVerifierConfig = serde_json::from_value(saved).unwrap();
        assert!(loaded.include_prior_findings);
        assert_eq!(loaded.max_format_retries, 1);
        assert_eq!(loaded.max_review_retries, 1);
    }

    #[test]
    fn parses_decision_line_with_opaque_multiline_feedback() {
        let response = parse_response(
            r#"
REJECTED_MINOR
The value "\ud800": is an escaped lone surrogate.

Keep this feedback on multiple lines without JSON escaping."#,
        )
        .unwrap();

        assert_eq!(response.decision, VerificationDecision::RejectedMinor);
        assert_eq!(
            response.feedback,
            "The value \"\\ud800\": is an escaped lone surrogate.\n\nKeep this feedback on multiple lines without JSON escaping."
        );
    }

    #[test]
    fn accepts_a_decision_without_feedback() {
        let response = parse_response("\r\n  ACCEPTED  \r\n").unwrap();

        assert_eq!(response.decision, VerificationDecision::Approved);
        assert!(response.feedback.is_empty());
    }

    #[test]
    fn rejected_decisions_require_feedback() {
        assert_eq!(
            parse_response("REJECTED_MINOR\n").unwrap_err(),
            "REJECTED_MINOR requires non-empty actionable feedback on the following line"
        );
        assert!(parse_response("REJECTED_MAJOR\n   \r\n").is_err());
    }

    #[test]
    fn rejects_json_and_inline_feedback() {
        let json = r#"{"decision":"approved","feedback":"looks good"}"#;
        assert!(parse_response(json).is_err());
        assert!(parse_response("REJECTED_MINOR: fix the parser").is_err());
    }

    #[test]
    fn compact_text_delta_contains_only_the_changed_region_and_context() {
        let before = "one\ntwo\nthree\nfour\nfive\nsix\nseven\n";
        let after = "one\ntwo\nthree\nFOUR\nfive\nsix\nseven\n";

        let delta = render_text_delta("src/example.rs", before, after, false, false);

        assert!(delta.contains("--- src/example.rs"));
        assert!(delta.contains("-four"));
        assert!(delta.contains("+FOUR"));
        assert!(delta.contains(" seven"));
    }

    #[test]
    fn bounded_delta_truncation_preserves_utf8() {
        let mut output = String::from("prefix\n");
        assert!(!append_bounded(&mut output, &"🦀".repeat(1_000), 128));
        assert!(output.is_char_boundary(output.len()));
        assert!(output.contains("repair delta truncated"));
        assert!(output.len() <= 128);
    }

    #[test]
    fn repair_delta_combines_tracked_file_and_append_only_memory_changes() {
        let workspace =
            std::env::temp_dir().join(format!("arko-repair-delta-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&workspace).unwrap();
        let path = workspace.join("example.txt");
        std::fs::write(&path, "before\n").unwrap();
        let workflow = Workflow::builder()
            .name("repair-delta-test")
            .workspace(&workspace)
            .llm_client(Arc::new(LLMClient::new()))
            .agent("coder", test_agent("coder"))
            .agent("reviewer", test_agent("reviewer"))
            .memory(MemoryConfig::new().global_tier("implementation_log"))
            .context_freshness(ContextFreshnessConfig::default())
            .build()
            .unwrap();
        let target = ActorId::new(uuid::Uuid::new_v4(), "coder");
        let reviewer = ActorId::new(uuid::Uuid::new_v4(), "reviewer");
        let file = ResourceId::File(path.clone());
        let memory = ResourceId::MemoryTier("implementation_log".to_string());
        let tracker = workflow.context().resource_tracker.as_ref().unwrap();
        tracker.record_write(&target, file.clone());
        tracker.record_read(&reviewer, file.clone());
        workflow
            .context()
            .memory
            .write_with_context(
                "coder",
                Some(target.task_id),
                "implementation_log",
                "initial entry",
            )
            .unwrap();
        tracker.record_write(&target, memory.clone());
        tracker.record_read(&reviewer, memory.clone());
        let baseline = capture_repair_baseline(workflow.context(), &target, &reviewer, 16 * 1_024);

        std::fs::write(&path, "after\n").unwrap();
        tracker.record_write(&target, file);
        workflow
            .context()
            .memory
            .write_with_context(
                "coder",
                Some(target.task_id),
                "implementation_log",
                "repair entry",
            )
            .unwrap();
        tracker.record_write(&target, memory);

        let delta = render_repair_delta(workflow.context(), "reviewer", &baseline, 16 * 1_024);

        assert!(delta.contains("-before"));
        assert!(delta.contains("+after"));
        assert!(delta.contains("repair entry"));
        assert!(!delta.contains("initial entry"));
        std::fs::remove_dir_all(workspace).unwrap();
    }
}
