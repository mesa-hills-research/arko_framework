//! The report: correctness and cost per arm and kind with confidence
//! intervals, paired differences between arms, and the cost/quality
//! frontier.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;

use crate::answer::{DeterministicCheck, deterministic_check};
use crate::ledger::read_jsonl;
use crate::repos::PreparedRepo;
use crate::runner::{AttemptRecord, GradeRecord, Outcome, RunManifest, parsed_answer};
use crate::tasks::{BenchTask, ItemKind, TaskKind};
use crate::{Arm, BenchResult};

/// A small deterministic random number generator (SplitMix64), for the
/// task order and the bootstrap.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index below `n` (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// An estimate with a 95% confidence interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Estimate {
    pub value: f64,
    pub low: f64,
    pub high: f64,
    /// Observations behind it.
    pub n: usize,
}

impl Estimate {
    fn fmt(&self, digits: usize) -> String {
        if self.n == 0 {
            return "-".into();
        }
        format!(
            "{:.d$} [{:.d$}, {:.d$}]",
            self.value,
            self.low,
            self.high,
            d = digits
        )
    }

    fn fmt_percent(&self) -> String {
        if self.n == 0 {
            return "-".into();
        }
        format!(
            "{:.0}% [{:.0}, {:.0}]",
            self.value * 100.0,
            self.low * 100.0,
            self.high * 100.0
        )
    }
}

/// Wilson score interval for a proportion.
pub fn wilson(successes: usize, n: usize) -> Estimate {
    if n == 0 {
        return Estimate {
            value: 0.0,
            low: 0.0,
            high: 0.0,
            n: 0,
        };
    }
    let z = 1.959_963_984_540_054_f64;
    let n_f = n as f64;
    let p = successes as f64 / n_f;
    let denominator = 1.0 + z * z / n_f;
    let centre = (p + z * z / (2.0 * n_f)) / denominator;
    let half = z * ((p * (1.0 - p) + z * z / (4.0 * n_f)) / n_f).sqrt() / denominator;
    Estimate {
        value: p,
        low: (centre - half).max(0.0),
        high: (centre + half).min(1.0),
        n,
    }
}

/// The mean of clustered observations (one cluster per task, holding its
/// samples) with a percentile bootstrap interval that resamples clusters.
pub fn bootstrap_mean(clusters: &[Vec<f64>], resamples: usize, rng: &mut Rng) -> Estimate {
    let clusters: Vec<&Vec<f64>> = clusters
        .iter()
        .filter(|cluster| !cluster.is_empty())
        .collect();
    let n: usize = clusters.iter().map(|cluster| cluster.len()).sum();
    if n == 0 {
        return Estimate {
            value: 0.0,
            low: 0.0,
            high: 0.0,
            n: 0,
        };
    }
    let mean_of = |picked: &mut dyn Iterator<Item = &Vec<f64>>| {
        let (mut sum, mut count) = (0.0, 0usize);
        for cluster in picked {
            sum += cluster.iter().sum::<f64>();
            count += cluster.len();
        }
        sum / count as f64
    };
    let value = mean_of(&mut clusters.iter().copied());
    if clusters.len() < 2 || resamples == 0 {
        return Estimate {
            value,
            low: value,
            high: value,
            n,
        };
    }
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            let mut picked = (0..clusters.len()).map(|_| clusters[rng.below(clusters.len())]);
            mean_of(&mut picked)
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |q: f64| means[((q * (means.len() - 1) as f64).round() as usize).min(means.len() - 1)];
    Estimate {
        value,
        low: at(0.025),
        high: at(0.975),
        n,
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// One settled cell: the task, its final attempt, and the grading.
#[derive(Debug, Clone)]
pub struct Row<'a> {
    pub task: &'a BenchTask,
    pub attempt: &'a AttemptRecord,
    pub grade: Option<&'a GradeRecord>,
    pub det: Option<DeterministicCheck>,
}

impl Row<'_> {
    fn answered(&self) -> bool {
        self.attempt.outcome == Outcome::Answered
    }

    /// The grader's pass: false for a run that didn't answer; `None` for an
    /// answer not graded.
    pub fn pass(&self) -> Option<bool> {
        if !self.answered() {
            return Some(false);
        }
        self.grade.and_then(|grade| grade.pass)
    }

    pub fn item_score(&self) -> Option<f64> {
        if !self.answered() {
            return Some(0.0);
        }
        self.grade.and_then(|grade| grade.item_score)
    }

    /// The deterministic check: `None` when the key has nothing checkable.
    pub fn det_pass(&self) -> Option<bool> {
        let checkable = self
            .task
            .key
            .items
            .iter()
            .any(|item| matches!(item.kind, ItemKind::Location | ItemKind::File));
        if !checkable {
            return None;
        }
        Some(self.answered() && self.det.as_ref().is_some_and(|det| det.pass))
    }

    pub fn det_score(&self) -> Option<f64> {
        self.det_pass()?;
        if !self.answered() {
            return Some(0.0);
        }
        self.det.as_ref().and_then(|det| det.score)
    }
}

/// The settled cells of a run: for each task, arm and sample, the last
/// final attempt and its first successful grading.
pub fn rows<'a>(
    tasks: &'a [BenchTask],
    attempts: &'a [AttemptRecord],
    grades: &'a [GradeRecord],
) -> Vec<Row<'a>> {
    let tasks_by_id: HashMap<&str, &BenchTask> =
        tasks.iter().map(|task| (task.id.as_str(), task)).collect();
    let mut last: BTreeMap<(String, Arm, u32), &AttemptRecord> = BTreeMap::new();
    for attempt in attempts.iter().filter(|attempt| attempt.outcome.is_final()) {
        if tasks_by_id.contains_key(attempt.task_id.as_str()) {
            last.insert(
                (attempt.task_id.clone(), attempt.arm, attempt.sample),
                attempt,
            );
        }
    }
    let mut grade_of: HashMap<&str, &GradeRecord> = HashMap::new();
    for grade in grades.iter().filter(|grade| grade.ok) {
        grade_of.entry(grade.attempt_id.as_str()).or_insert(grade);
    }
    last.into_values()
        .map(|attempt| {
            let task = tasks_by_id[attempt.task_id.as_str()];
            let det = (attempt.outcome == Outcome::Answered)
                .then(|| deterministic_check(task, &parsed_answer(attempt)));
            Row {
                task,
                attempt,
                grade: grade_of.get(attempt.attempt_id.as_str()).copied(),
                det,
            }
        })
        .collect()
}

/// The figures of one group of cells (an arm, and a kind or a repository or
/// everything).
#[derive(Debug, Clone, Serialize)]
pub struct GroupStats {
    pub arm: Arm,
    pub group: String,
    pub cells: usize,
    pub tasks: usize,
    pub answered: usize,
    pub failed: usize,
    pub over_limit: usize,
    pub graded: usize,
    pub pass_rate: Estimate,
    pub item_score: Estimate,
    pub det_pass_rate: Estimate,
    pub input_tokens: Estimate,
    pub uncached_input_tokens: Estimate,
    pub output_tokens: Estimate,
    pub reasoning_tokens: f64,
    pub llm_calls: Estimate,
    pub shell_calls: Estimate,
    pub index_calls: f64,
    /// Share of cells whose agent ran arko_index at least once.
    pub index_adoption: f64,
    /// Working time: wall time less waits for usage limits and retries.
    pub work_seconds: Estimate,
    pub median_work_seconds: f64,
    pub api_cost: Estimate,
    /// Total cost divided by passes.
    pub cost_per_pass: Option<f64>,
}

fn clusters_of<'a>(rows: &[&Row<'a>], value: impl Fn(&Row<'a>) -> Option<f64>) -> Vec<Vec<f64>> {
    let mut by_task: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for row in rows {
        if let Some(v) = value(row) {
            by_task.entry(row.task.id.as_str()).or_default().push(v);
        }
    }
    by_task.into_values().collect()
}

/// A rate over cells with a 95% interval: Wilson's when every task has one
/// cell (or there is only one task), else a bootstrap that resamples tasks
/// with all their samples.
fn rate<'a>(
    rows: &[&Row<'a>],
    value: impl Fn(&Row<'a>) -> Option<bool>,
    resamples: usize,
    rng: &mut Rng,
) -> Estimate {
    let clusters = clusters_of(rows, |row| value(row).map(|pass| f64::from(u8::from(pass))));
    if clusters.len() < 2 || clusters.iter().all(|cluster| cluster.len() <= 1) {
        let values: Vec<f64> = clusters.into_iter().flatten().collect();
        let successes = values.iter().filter(|value| **value > 0.5).count();
        return wilson(successes, values.len());
    }
    bootstrap_mean(&clusters, resamples, rng)
}

pub fn group_stats(
    arm: Arm,
    group: &str,
    rows: &[&Row<'_>],
    resamples: usize,
    rng: &mut Rng,
) -> GroupStats {
    let passes: Vec<bool> = rows.iter().filter_map(|row| row.pass()).collect();
    let usage = |f: fn(&crate::runner::UsageSummary) -> f64| {
        clusters_of(rows, move |row| Some(f(&row.attempt.usage)))
    };
    let mut works: Vec<f64> = rows
        .iter()
        .map(|row| row.attempt.active_ms as f64 / 1000.0)
        .collect();
    let total_cost: f64 = rows.iter().map(|row| row.attempt.usage.api_cost).sum();
    let pass_count = passes.iter().filter(|pass| **pass).count();
    let n = rows.len().max(1) as f64;
    GroupStats {
        arm,
        group: group.to_string(),
        cells: rows.len(),
        tasks: rows
            .iter()
            .map(|row| row.task.id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        answered: rows.iter().filter(|row| row.answered()).count(),
        failed: rows
            .iter()
            .filter(|row| row.attempt.outcome == Outcome::Failed)
            .count(),
        over_limit: rows
            .iter()
            .filter(|row| row.attempt.outcome == Outcome::Limit)
            .count(),
        graded: rows
            .iter()
            .filter(|row| row.answered() && row.grade.is_some())
            .count(),
        pass_rate: rate(rows, Row::pass, resamples, rng),
        item_score: bootstrap_mean(&clusters_of(rows, Row::item_score), resamples, rng),
        det_pass_rate: rate(rows, Row::det_pass, resamples, rng),
        input_tokens: bootstrap_mean(&usage(|u| u.input_tokens as f64), resamples, rng),
        uncached_input_tokens: bootstrap_mean(
            &usage(|u| (u.input_tokens - u.cached_input_tokens.min(u.input_tokens)) as f64),
            resamples,
            rng,
        ),
        output_tokens: bootstrap_mean(&usage(|u| u.output_tokens as f64), resamples, rng),
        reasoning_tokens: rows
            .iter()
            .map(|row| row.attempt.usage.reasoning_tokens as f64)
            .sum::<f64>()
            / n,
        llm_calls: bootstrap_mean(&usage(|u| u.llm_calls as f64), resamples, rng),
        shell_calls: bootstrap_mean(&usage(|u| u.shell_calls as f64), resamples, rng),
        index_calls: rows
            .iter()
            .map(|row| (row.attempt.usage.index_calls + row.attempt.usage.dense_calls) as f64)
            .sum::<f64>()
            / n,
        index_adoption: rows
            .iter()
            .filter(|row| row.attempt.usage.index_calls + row.attempt.usage.dense_calls > 0)
            .count() as f64
            / n,
        work_seconds: bootstrap_mean(
            &clusters_of(rows, |row| Some(row.attempt.active_ms as f64 / 1000.0)),
            resamples,
            rng,
        ),
        median_work_seconds: median(&mut works),
        api_cost: bootstrap_mean(&usage(|u| u.api_cost), resamples, rng),
        cost_per_pass: (pass_count > 0).then(|| total_cost / pass_count as f64),
    }
}

/// Arm `second` minus arm `first` over the tasks both settled.
#[derive(Debug, Clone, Serialize)]
pub struct PairedStats {
    pub first: Arm,
    pub second: Arm,
    pub group: String,
    pub tasks: usize,
    pub pass_rate: Estimate,
    pub item_score: Estimate,
    pub det_pass_rate: Estimate,
    pub input_tokens: Estimate,
    pub output_tokens: Estimate,
    pub api_cost: Estimate,
    pub work_seconds: Estimate,
    /// Tasks where `second` scored higher, the same, lower (item score).
    pub wins: usize,
    pub ties: usize,
    pub losses: usize,
}

pub fn paired_stats(
    first: Arm,
    second: Arm,
    group: &str,
    rows: &[&Row<'_>],
    resamples: usize,
    rng: &mut Rng,
) -> PairedStats {
    // Per task and arm: the mean over samples of each figure.
    type Means = (Option<f64>, Option<f64>, Option<f64>, f64, f64, f64, f64);
    let mut per_task: BTreeMap<&str, HashMap<Arm, Vec<&Row<'_>>>> = BTreeMap::new();
    for row in rows {
        per_task
            .entry(row.task.id.as_str())
            .or_default()
            .entry(row.attempt.arm)
            .or_default()
            .push(row);
    }
    let means = |rows: &[&Row<'_>]| -> Means {
        let avg = |values: Vec<f64>| -> Option<f64> {
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        };
        (
            avg(rows
                .iter()
                .filter_map(|row| row.pass().map(|pass| if pass { 1.0 } else { 0.0 }))
                .collect()),
            avg(rows.iter().filter_map(|row| row.item_score()).collect()),
            avg(rows
                .iter()
                .filter_map(|row| row.det_pass().map(|pass| if pass { 1.0 } else { 0.0 }))
                .collect()),
            avg(rows
                .iter()
                .map(|row| row.attempt.usage.input_tokens as f64)
                .collect())
            .unwrap_or(0.0),
            avg(rows
                .iter()
                .map(|row| row.attempt.usage.output_tokens as f64)
                .collect())
            .unwrap_or(0.0),
            avg(rows.iter().map(|row| row.attempt.usage.api_cost).collect()).unwrap_or(0.0),
            avg(rows
                .iter()
                .map(|row| row.attempt.active_ms as f64 / 1000.0)
                .collect())
            .unwrap_or(0.0),
        )
    };
    let mut diffs: [Vec<Vec<f64>>; 7] = Default::default();
    let (mut wins, mut ties, mut losses, mut tasks) = (0, 0, 0, 0);
    for arms in per_task.values() {
        let (Some(a), Some(b)) = (arms.get(&first), arms.get(&second)) else {
            continue;
        };
        tasks += 1;
        let (a, b) = (means(a), means(b));
        let optional = [(a.0, b.0), (a.1, b.1), (a.2, b.2)];
        for (index, (x, y)) in optional.into_iter().enumerate() {
            if let (Some(x), Some(y)) = (x, y) {
                diffs[index].push(vec![y - x]);
            }
        }
        for (index, (x, y)) in [(a.3, b.3), (a.4, b.4), (a.5, b.5), (a.6, b.6)]
            .into_iter()
            .enumerate()
        {
            diffs[3 + index].push(vec![y - x]);
        }
        if let (Some(x), Some(y)) = (a.1, b.1) {
            if (y - x).abs() < 1e-9 {
                ties += 1;
            } else if y > x {
                wins += 1;
            } else {
                losses += 1;
            }
        }
    }
    let mut estimate = |index: usize| bootstrap_mean(&diffs[index], resamples, rng);
    PairedStats {
        first,
        second,
        group: group.to_string(),
        tasks,
        pass_rate: estimate(0),
        item_score: estimate(1),
        det_pass_rate: estimate(2),
        input_tokens: estimate(3),
        output_tokens: estimate(4),
        api_cost: estimate(5),
        work_seconds: estimate(6),
        wins,
        ties,
        losses,
    }
}

/// A point of the cost/quality frontier.
#[derive(Debug, Clone, Serialize)]
pub struct FrontierPoint {
    pub arm: Arm,
    pub group: String,
    pub api_cost: f64,
    pub total_tokens: f64,
    pub item_score: f64,
    pub pass_rate: f64,
    /// No other arm of the group is at most as costly and at least as good
    /// (item score), and better in one.
    pub on_frontier: bool,
}

pub fn frontier(stats: &[GroupStats]) -> Vec<FrontierPoint> {
    let mut points: Vec<FrontierPoint> = stats
        .iter()
        .filter(|stats| stats.cells > 0)
        .map(|stats| FrontierPoint {
            arm: stats.arm,
            group: stats.group.clone(),
            api_cost: stats.api_cost.value,
            total_tokens: stats.input_tokens.value + stats.output_tokens.value,
            item_score: stats.item_score.value,
            pass_rate: stats.pass_rate.value,
            on_frontier: true,
        })
        .collect();
    let snapshot = points.clone();
    for point in &mut points {
        point.on_frontier = !snapshot.iter().any(|other| {
            other.group == point.group
                && other.arm != point.arm
                && other.api_cost <= point.api_cost
                && other.item_score >= point.item_score
                && (other.api_cost < point.api_cost || other.item_score > point.item_score)
        });
    }
    points
}

/// How often the grader and the deterministic check agree on location
/// items.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Agreement {
    pub items: usize,
    pub agree: usize,
    pub grader_only: usize,
    pub check_only: usize,
}

fn agreement(rows: &[Row<'_>]) -> Agreement {
    let mut agreement = Agreement::default();
    for row in rows {
        let (Some(grade), true) = (row.grade, row.answered()) else {
            continue;
        };
        let Some(verdict) = &grade.verdict else {
            continue;
        };
        let answer = parsed_answer(row.attempt);
        for item in &row.task.key.items {
            if item.kind != ItemKind::Location {
                continue;
            }
            let Some(location) = &item.location else {
                continue;
            };
            let cited = if answer.locations.is_empty() {
                &answer.mentioned
            } else {
                &answer.locations
            };
            let check = cited.iter().any(|cited| {
                crate::answer::cited_matches(location, cited, crate::answer::LINE_TOLERANCE)
            });
            let grader = verdict
                .items
                .iter()
                .any(|judgment| judgment.id == item.id && judgment.met);
            agreement.items += 1;
            match (grader, check) {
                (true, true) | (false, false) => agreement.agree += 1,
                (true, false) => agreement.grader_only += 1,
                (false, true) => agreement.check_only += 1,
            }
        }
    }
    agreement
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub run: RunManifest,
    pub by_kind: Vec<GroupStats>,
    pub by_repo: Vec<GroupStats>,
    pub paired: Vec<PairedStats>,
    pub frontier: Vec<FrontierPoint>,
    pub agreement: Agreement,
    pub infra_attempts: usize,
    /// Settled cells whose agent's commands named the benchmark's own files
    /// outside the checkout, per arm (check their transcripts).
    pub looked_outside: BTreeMap<Arm, usize>,
    pub ungraded_answers: usize,
    pub failed_gradings: usize,
    pub grading_usage: crate::runner::UsageSummary,
    pub index_builds: BTreeMap<String, u64>,
}

/// Computes the report from the run's records.
pub fn compute(
    manifest: &RunManifest,
    tasks: &[BenchTask],
    attempts: &[AttemptRecord],
    grades: &[GradeRecord],
    prepared: &BTreeMap<String, PreparedRepo>,
    resamples: usize,
) -> Report {
    let rows = rows(tasks, attempts, grades);
    let mut rng = Rng::new(0x5EED);
    let arms: Vec<Arm> = rows
        .iter()
        .map(|row| row.attempt.arm)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // Every kind together ("all", `None`), then each kind.
    let groups = std::iter::once(None).chain(TaskKind::ALL.into_iter().map(Some));
    let mut by_kind = Vec::new();
    let mut paired = Vec::new();
    for kind in groups {
        let group = kind.map_or("all", TaskKind::as_str);
        let in_group: Vec<&Row<'_>> = rows
            .iter()
            .filter(|row| kind.is_none_or(|kind| row.task.kind == kind))
            .collect();
        if in_group.is_empty() {
            continue;
        }
        for arm in &arms {
            let arm_rows: Vec<&Row<'_>> = in_group
                .iter()
                .copied()
                .filter(|row| row.attempt.arm == *arm)
                .collect();
            by_kind.push(group_stats(*arm, group, &arm_rows, resamples, &mut rng));
        }
        for (i, first) in arms.iter().enumerate() {
            for second in &arms[i + 1..] {
                paired.push(paired_stats(
                    *first, *second, group, &in_group, resamples, &mut rng,
                ));
            }
        }
    }
    let mut by_repo = Vec::new();
    let repos: BTreeSet<&str> = rows.iter().map(|row| row.task.repo.as_str()).collect();
    for repo in repos {
        for arm in &arms {
            let arm_rows: Vec<&Row<'_>> = rows
                .iter()
                .filter(|row| row.task.repo == repo && row.attempt.arm == *arm)
                .collect();
            let language = prepared
                .get(repo)
                .map_or(String::new(), |repo| format!(" ({})", repo.spec.language));
            by_repo.push(group_stats(
                *arm,
                &format!("{repo}{language}"),
                &arm_rows,
                resamples,
                &mut rng,
            ));
        }
    }
    let frontier = frontier(&by_kind);
    let graded_ok: BTreeSet<&str> = grades
        .iter()
        .filter(|grade| grade.ok)
        .map(|grade| grade.attempt_id.as_str())
        .collect();
    let mut grading_usage = crate::runner::UsageSummary::default();
    for grade in grades {
        grading_usage.add(&grade.usage);
    }
    Report {
        generated_at: chrono::Utc::now(),
        run: manifest.clone(),
        agreement: agreement(&rows),
        ungraded_answers: rows
            .iter()
            .filter(|row| row.answered() && !graded_ok.contains(row.attempt.attempt_id.as_str()))
            .count(),
        failed_gradings: grades.iter().filter(|grade| !grade.ok).count(),
        looked_outside: rows.iter().filter(|row| row.attempt.looked_outside).fold(
            BTreeMap::new(),
            |mut counts, row| {
                *counts.entry(row.attempt.arm).or_insert(0) += 1;
                counts
            },
        ),
        infra_attempts: attempts
            .iter()
            .filter(|attempt| attempt.outcome == Outcome::Infra)
            .count(),
        grading_usage,
        index_builds: prepared
            .iter()
            .map(|(id, repo)| (id.clone(), repo.index_build_ms))
            .collect(),
        by_kind,
        by_repo,
        paired,
        frontier,
    }
}

fn tokens(value: f64) -> String {
    if value >= 1e6 {
        format!("{:.2}M", value / 1e6)
    } else if value >= 1e3 {
        format!("{:.0}k", value / 1e3)
    } else {
        format!("{value:.0}")
    }
}

fn tokens_estimate(estimate: &Estimate) -> String {
    if estimate.n == 0 {
        return "-".into();
    }
    format!(
        "{} [{}, {}]",
        tokens(estimate.value),
        tokens(estimate.low),
        tokens(estimate.high)
    )
}

fn signed_tokens(estimate: &Estimate) -> String {
    if estimate.n == 0 {
        return "-".into();
    }
    let s = |v: f64| {
        if v < 0.0 {
            format!("-{}", tokens(-v))
        } else {
            format!("+{}", tokens(v))
        }
    };
    format!(
        "{} [{}, {}]",
        s(estimate.value),
        s(estimate.low),
        s(estimate.high)
    )
}

fn signed_percent(estimate: &Estimate) -> String {
    if estimate.n == 0 {
        return "-".into();
    }
    format!(
        "{:+.1} pp [{:+.1}, {:+.1}]",
        estimate.value * 100.0,
        estimate.low * 100.0,
        estimate.high * 100.0
    )
}

/// The report as Markdown.
pub fn render(report: &Report) -> String {
    let mut text = String::new();
    let run = &report.run;
    let _ = writeln!(text, "# arko_index agent benchmark: run `{}`\n", run.name);
    let _ = writeln!(
        text,
        "Model `{}` at `{}` effort{}, every agent{}. Arms: {}. Shell output cap {} bytes, {} s per command, {} commands per run; runs stop at {} input tokens{}. Generated {}.\n",
        run.model.model_id,
        run.model.effort,
        if run.model.fast { ", Fast mode" } else { "" },
        if run.mock {
            " (MOCK MODEL: figures are not real results)"
        } else {
            ""
        },
        run.arms
            .iter()
            .map(|arm| format!("{arm} ({})", arm.label()))
            .collect::<Vec<_>>()
            .join(", "),
        run.shell_max_output_bytes,
        run.shell_timeout_seconds,
        run.shell_max_calls,
        run.max_input_tokens,
        run.attempt_timeout_seconds
            .map_or(String::new(), |seconds| format!(
                " or after {} min of work",
                seconds / 60
            )),
        report.generated_at.format("%Y-%m-%d %H:%M UTC"),
    );
    if !run.variants.is_empty() {
        let _ = writeln!(
            text,
            "Variant arms (arm B's tools over another index, or other search flags):\n\n| arm | index2 flags | search2 flags | tools | description |\n|---|---|---|---|---|"
        );
        for (name, variant) in &run.variants {
            let flags = variant
                .repos
                .values()
                .next()
                .map_or(variant.spec.index.join(" "), |index| index.flags.join(" "));
            let tools = if variant.spec.tools_text.is_some() {
                "own text".to_string()
            } else {
                variant.spec.prompt.clone().unwrap_or_else(|| {
                    if variant.spec.embeds() {
                        "hybrid".into()
                    } else {
                        "index".into()
                    }
                })
            };
            let _ = writeln!(
                text,
                "| {name} | `{}` | `{}` | {tools} | {} |",
                flags,
                variant.spec.search.join(" "),
                variant.spec.description.replace('|', "\\|"),
            );
        }
        text.push('\n');
    }
    let _ = writeln!(
        text,
        "Correctness: **pass** is the blind grader's verdict `correct` (a run that didn't answer fails); **items** is the share of the key's required items the grader found met (0 for no answer); **check** is the deterministic check of the key's locations and files. Intervals are 95%: Wilson for rates (a bootstrap over tasks when tasks have several samples), a bootstrap over tasks for means. Time is working time: wall time less waits for usage limits and retries. Cost is the API list price of the tokens (ChatGPT OAuth itself bills nothing per call).\n"
    );
    let _ = writeln!(text, "## Per arm and kind\n");
    let _ = writeln!(
        text,
        "| kind | arm | cells | answered | pass | items | check | input tok | uncached in | output tok | calls | shell | index use | work s (median) | cost $ | $/pass |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    );
    for stats in &report.by_kind {
        let _ = writeln!(
            text,
            "| {} | {} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} ({:.0}) | {} | {} |",
            stats.group,
            stats.arm,
            stats.cells,
            stats.answered,
            if stats.failed + stats.over_limit > 0 {
                format!(
                    " ({} failed, {} over limit)",
                    stats.failed, stats.over_limit
                )
            } else {
                String::new()
            },
            stats.pass_rate.fmt_percent(),
            stats.item_score.fmt(2),
            stats.det_pass_rate.fmt_percent(),
            tokens_estimate(&stats.input_tokens),
            tokens_estimate(&stats.uncached_input_tokens),
            tokens_estimate(&stats.output_tokens),
            stats.llm_calls.fmt(1),
            stats.shell_calls.fmt(1),
            if stats.arm == Arm::A {
                "-".to_string()
            } else {
                format!(
                    "{:.0}% ({:.1}/run)",
                    stats.index_adoption * 100.0,
                    stats.index_calls
                )
            },
            stats.work_seconds.fmt(0),
            stats.median_work_seconds,
            stats.api_cost.fmt(3),
            stats
                .cost_per_pass
                .map_or("-".into(), |cost| format!("{cost:.3}")),
        );
    }
    if !report.paired.is_empty() {
        let _ = writeln!(
            text,
            "\n## Paired differences (second arm minus first, same tasks)\n"
        );
        let _ = writeln!(
            text,
            "| kind | arms | tasks | pass | items | check | input tok | output tok | cost $ | work s | items better/same/worse |\n|---|---|---|---|---|---|---|---|---|---|---|"
        );
        for paired in &report.paired {
            let _ = writeln!(
                text,
                "| {} | {}→{} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} |",
                paired.group,
                paired.first,
                paired.second,
                paired.tasks,
                signed_percent(&paired.pass_rate),
                if paired.item_score.n == 0 {
                    "-".into()
                } else {
                    format!(
                        "{:+.3} [{:+.3}, {:+.3}]",
                        paired.item_score.value, paired.item_score.low, paired.item_score.high
                    )
                },
                signed_percent(&paired.det_pass_rate),
                signed_tokens(&paired.input_tokens),
                signed_tokens(&paired.output_tokens),
                if paired.api_cost.n == 0 {
                    "-".into()
                } else {
                    format!(
                        "{:+.3} [{:+.3}, {:+.3}]",
                        paired.api_cost.value, paired.api_cost.low, paired.api_cost.high
                    )
                },
                if paired.work_seconds.n == 0 {
                    "-".into()
                } else {
                    format!(
                        "{:+.0} [{:+.0}, {:+.0}]",
                        paired.work_seconds.value,
                        paired.work_seconds.low,
                        paired.work_seconds.high
                    )
                },
                paired.wins,
                paired.ties,
                paired.losses,
            );
        }
    }
    let _ = writeln!(text, "\n## Cost/quality frontier\n");
    let _ = writeln!(
        text,
        "Per group, an arm is on the frontier when no other arm costs at most as much with at least the item score (and is better in one). `frontier.csv` has the points for plotting.\n"
    );
    let _ = writeln!(
        text,
        "| kind | arm | cost $ | tokens | items | pass | frontier |\n|---|---|---|---|---|---|---|"
    );
    for point in &report.frontier {
        let _ = writeln!(
            text,
            "| {} | {} | {:.3} | {} | {:.2} | {:.0}% | {} |",
            point.group,
            point.arm,
            point.api_cost,
            tokens(point.total_tokens),
            point.item_score,
            point.pass_rate * 100.0,
            if point.on_frontier { "yes" } else { "" }
        );
    }
    if !report.by_repo.is_empty() {
        let _ = writeln!(text, "\n## Per repository\n");
        let _ = writeln!(
            text,
            "| repository | arm | cells | pass | items | check | input tok | output tok | cost $ |\n|---|---|---|---|---|---|---|---|---|"
        );
        for stats in &report.by_repo {
            let _ = writeln!(
                text,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                stats.group,
                stats.arm,
                stats.cells,
                stats.pass_rate.fmt_percent(),
                stats.item_score.fmt(2),
                stats.det_pass_rate.fmt_percent(),
                tokens_estimate(&stats.input_tokens),
                tokens_estimate(&stats.output_tokens),
                stats.api_cost.fmt(3),
            );
        }
    }
    let _ = writeln!(text, "\n## Checks and bookkeeping\n");
    let agreement = &report.agreement;
    if agreement.items > 0 {
        let _ = writeln!(
            text,
            "- Grader and deterministic check agree on {} of {} location items ({:.0}%); grader only: {}, check only: {}.",
            agreement.agree,
            agreement.items,
            agreement.agree as f64 * 100.0 / agreement.items as f64,
            agreement.grader_only,
            agreement.check_only
        );
    }
    let _ = writeln!(
        text,
        "- Answers not graded yet: {}; failed gradings: {}; runs that ended outside the agent's control (run again): {}.",
        report.ungraded_answers, report.failed_gradings, report.infra_attempts
    );
    if !report.looked_outside.is_empty() {
        let _ = writeln!(
            text,
            "- **Runs whose commands named the benchmark's own files** (task sets, records, transcripts) outside their checkout: {}. Their answers may come from a key or another arm's answer: check the transcripts (`looked_outside` in attempts.jsonl).",
            report
                .looked_outside
                .iter()
                .map(|(arm, count)| format!("arm {arm}: {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let _ = writeln!(
        text,
        "- Grading used {} input and {} output tokens (${:.2} at list price).",
        tokens(report.grading_usage.input_tokens as f64),
        tokens(report.grading_usage.output_tokens as f64),
        report.grading_usage.api_cost
    );
    if !report.index_builds.is_empty() {
        let _ = writeln!(
            text,
            "- Index builds (outside the timed runs): {}.",
            report
                .index_builds
                .iter()
                .map(|(repo, ms)| format!("{repo} {:.1} s", *ms as f64 / 1000.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    text
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// One line per settled cell.
pub fn attempts_csv(rows: &[Row<'_>], prepared: &BTreeMap<String, PreparedRepo>) -> String {
    let mut text = String::from(
        "task_id,repo,language,kind,arm,sample,outcome,pass,item_score,check_pass,check_score,wrong_claims,input_tokens,cached_input_tokens,output_tokens,reasoning_tokens,llm_calls,shell_calls,index_calls,dense_calls,wall_s,work_s,api_cost,looked_outside\n",
    );
    let option = |value: Option<f64>| value.map_or(String::new(), |v| format!("{v:.4}"));
    let flag = |value: Option<bool>| value.map_or(String::new(), |v| (v as u8).to_string());
    for row in rows {
        let usage = &row.attempt.usage;
        let _ = writeln!(
            text,
            "{},{},{},{},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.5},{}",
            csv_field(&row.task.id),
            csv_field(&row.task.repo),
            prepared
                .get(&row.task.repo)
                .map_or("", |repo| repo.spec.language.as_str()),
            row.task.kind,
            row.attempt.arm,
            row.attempt.sample,
            row.attempt.outcome,
            flag(row.pass()),
            option(row.item_score()),
            flag(row.det_pass()),
            option(row.det_score()),
            row.grade
                .and_then(|grade| grade.verdict.as_ref())
                .map_or(String::new(), |verdict| verdict
                    .wrong_claims
                    .len()
                    .to_string()),
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.llm_calls,
            usage.shell_calls,
            usage.index_calls,
            usage.dense_calls,
            row.attempt.wall_ms as f64 / 1000.0,
            row.attempt.active_ms as f64 / 1000.0,
            usage.api_cost,
            u8::from(row.attempt.looked_outside),
        );
    }
    text
}

fn frontier_csv(points: &[FrontierPoint]) -> String {
    let mut text =
        String::from("group,arm,api_cost,total_tokens,item_score,pass_rate,on_frontier\n");
    for point in points {
        let _ = writeln!(
            text,
            "{},{},{:.5},{:.0},{:.4},{:.4},{}",
            point.group,
            point.arm,
            point.api_cost,
            point.total_tokens,
            point.item_score,
            point.pass_rate,
            point.on_frontier
        );
    }
    text
}

/// Reads the run's records, writes `report.md`, `report.json`,
/// `attempts.csv` and `frontier.csv` into the run directory, and returns the
/// Markdown.
pub fn write(
    run_dir: &Path,
    manifest: &RunManifest,
    tasks: &[BenchTask],
    prepared: &BTreeMap<String, PreparedRepo>,
    resamples: usize,
) -> BenchResult<String> {
    let attempts: Vec<AttemptRecord> = read_jsonl(&run_dir.join("attempts.jsonl"))?;
    let grades: Vec<GradeRecord> = read_jsonl(&run_dir.join("grades.jsonl"))?;
    let report = compute(manifest, tasks, &attempts, &grades, prepared, resamples);
    let markdown = render(&report);
    crate::write_atomic(&run_dir.join("report.md"), markdown.as_bytes())?;
    crate::write_atomic(
        &run_dir.join("report.json"),
        serde_json::to_string_pretty(&report)?.as_bytes(),
    )?;
    let rows = rows(tasks, &attempts, &grades);
    crate::write_atomic(
        &run_dir.join("attempts.csv"),
        attempts_csv(&rows, prepared).as_bytes(),
    )?;
    crate::write_atomic(
        &run_dir.join("frontier.csv"),
        frontier_csv(&report.frontier).as_bytes(),
    )?;
    Ok(markdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer::{GraderVerdict, ItemJudgment};
    use crate::runner::UsageSummary;
    use crate::tasks::{AnswerKey, KeyItem, Location, Origin};

    fn task(id: &str, kind: TaskKind) -> BenchTask {
        BenchTask {
            id: id.into(),
            repo: "demo".into(),
            kind,
            question: "q".into(),
            scope: None,
            key: AnswerKey {
                answer: "a".into(),
                items: vec![
                    KeyItem {
                        id: "L1".into(),
                        kind: ItemKind::Location,
                        text: "it".into(),
                        location: Some(Location::new("src/a.rs", 10, 20)),
                    },
                    KeyItem {
                        id: "F1".into(),
                        kind: ItemKind::Fact,
                        text: "fact".into(),
                        location: None,
                    },
                ],
                distractors: vec![],
            },
            origin: Origin::default(),
            difficulty: None,
            verification: None,
        }
    }

    fn attempt(
        task: &str,
        arm: Arm,
        outcome: Outcome,
        answer: &str,
        input: usize,
    ) -> AttemptRecord {
        AttemptRecord {
            attempt_id: format!("{task}-{arm}"),
            task_id: task.into(),
            repo: "demo".into(),
            kind: TaskKind::Definition,
            arm,
            sample: 0,
            outcome,
            detail: None,
            failure_kind: None,
            answer: (outcome == Outcome::Answered).then(|| answer.to_string()),
            started_at: chrono::Utc::now(),
            wall_ms: 10_000,
            active_ms: 10_000,
            usage: UsageSummary {
                llm_calls: 5,
                input_tokens: input,
                output_tokens: input / 10,
                api_cost: input as f64 / 1e6,
                index_calls: usize::from(arm == Arm::B),
                ..UsageSummary::default()
            },
            transcript: None,
            looked_outside: false,
        }
    }

    fn grade(attempt: &AttemptRecord, location_met: bool, fact_met: bool) -> GradeRecord {
        let verdict = GraderVerdict {
            items: vec![
                ItemJudgment {
                    id: "L1".into(),
                    met: location_met,
                    note: String::new(),
                },
                ItemJudgment {
                    id: "F1".into(),
                    met: fact_met,
                    note: String::new(),
                },
            ],
            wrong_claims: vec![],
            verdict: if location_met && fact_met {
                "correct".into()
            } else {
                "partially_correct".into()
            },
            notes: String::new(),
        };
        GradeRecord {
            grade_id: format!("g-{}", attempt.attempt_id),
            attempt_id: attempt.attempt_id.clone(),
            task_id: attempt.task_id.clone(),
            ok: true,
            item_score: Some((u8::from(location_met) + u8::from(fact_met)) as f64 / 2.0),
            pass: Some(location_met && fact_met),
            verdict: Some(verdict),
            detail: None,
            scrubbed: false,
            started_at: chrono::Utc::now(),
            wall_ms: 1,
            usage: UsageSummary::default(),
            transcript: None,
            failure_kind: None,
        }
    }

    #[test]
    fn wilson_intervals_are_sane() {
        let estimate = wilson(8, 10);
        assert!((estimate.value - 0.8).abs() < 1e-9);
        assert!(estimate.low > 0.44 && estimate.low < 0.5, "{estimate:?}");
        assert!(estimate.high > 0.94 && estimate.high < 0.97, "{estimate:?}");
        let none = wilson(0, 20);
        assert_eq!(none.low, 0.0);
        assert!(none.high > 0.1 && none.high < 0.2);
        assert_eq!(wilson(0, 0).n, 0);
    }

    #[test]
    fn bootstrap_brackets_the_mean_and_resamples_clusters() {
        let mut rng = Rng::new(1);
        let clusters: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        let estimate = bootstrap_mean(&clusters, 2000, &mut rng);
        assert!((estimate.value - 24.5).abs() < 1e-9);
        assert!(estimate.low < 24.5 && estimate.high > 24.5);
        assert!(estimate.low > 18.0 && estimate.high < 31.0, "{estimate:?}");
        // One cluster: no interval to speak of.
        let single = bootstrap_mean(&[vec![1.0, 3.0]], 100, &mut rng);
        assert_eq!(
            (single.value, single.low, single.high, single.n),
            (2.0, 2.0, 2.0, 2)
        );
    }

    #[test]
    fn aggregates_grades_failures_and_paired_differences() {
        let tasks: Vec<BenchTask> = (0..4)
            .map(|i| task(&format!("t{i}"), TaskKind::Definition))
            .collect();
        let right = "ANSWER: here\nLOCATIONS:\nsrc/a.rs:12\n";
        let wrong = "ANSWER: there\nLOCATIONS:\nsrc/b.rs:12\n";
        let mut attempts = Vec::new();
        let mut grades = Vec::new();
        for (i, task) in tasks.iter().enumerate() {
            // Arm A: right on t0 and t1, wrong on t2, failed on t3.
            let a = match i {
                0 | 1 => attempt(&task.id, Arm::A, Outcome::Answered, right, 200_000),
                2 => attempt(&task.id, Arm::A, Outcome::Answered, wrong, 300_000),
                _ => attempt(&task.id, Arm::A, Outcome::Limit, "", 4_000_000),
            };
            // Arm B: right everywhere, with fewer tokens.
            let b = attempt(&task.id, Arm::B, Outcome::Answered, right, 100_000);
            if a.outcome == Outcome::Answered {
                grades.push(grade(&a, i < 2, i < 2));
            }
            grades.push(grade(&b, true, i != 3));
            attempts.push(a);
            attempts.push(b);
        }
        // An infra attempt and a failed grading don't count.
        attempts.push(AttemptRecord {
            attempt_id: "infra".into(),
            ..attempt("t0", Arm::A, Outcome::Infra, "", 5)
        });
        let manifest_settings = crate::runner::RunManifest {
            name: "test".into(),
            created_at: chrono::Utc::now(),
            model: crate::agents::ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap(),
            mock: true,
            shell_timeout_seconds: 120,
            shell_max_output_bytes: 16384,
            shell_max_calls: 80,
            max_input_tokens: 4_000_000,
            attempt_timeout_seconds: None,
            grader_shell: true,
            index_cli_sha256: String::new(),
            index_cli: std::path::PathBuf::new(),
            arms: vec![Arm::A, Arm::B],
            samples: 1,
            tasks_file: "tasks.json".into(),
            sessions: vec![],
            variants: Default::default(),
        };
        let report = compute(
            &manifest_settings,
            &tasks,
            &attempts,
            &grades,
            &BTreeMap::new(),
            500,
        );
        let find = |arm: Arm, group: &str| {
            report
                .by_kind
                .iter()
                .find(|stats| stats.arm == arm && stats.group == group)
                .unwrap()
                .clone()
        };
        let a = find(Arm::A, "all");
        assert_eq!((a.cells, a.answered, a.over_limit, a.graded), (4, 3, 1, 3));
        assert_eq!(a.pass_rate.n, 4);
        assert!((a.pass_rate.value - 0.5).abs() < 1e-9);
        // Item scores: 1, 1, 0, 0 (no answer).
        assert!((a.item_score.value - 0.5).abs() < 1e-9);
        // The check: right, right, wrong location, no answer.
        assert!((a.det_pass_rate.value - 0.5).abs() < 1e-9);
        assert!((a.input_tokens.value - 1_175_000.0).abs() < 1e-6);
        let b = find(Arm::B, "all");
        assert!((b.pass_rate.value - 0.75).abs() < 1e-9);
        assert!((b.item_score.value - 0.875).abs() < 1e-9);
        assert!((b.det_pass_rate.value - 1.0).abs() < 1e-9);
        assert!((b.index_adoption - 1.0).abs() < 1e-9);
        assert_eq!(find(Arm::B, "definition").cells, 4);

        let paired = report
            .paired
            .iter()
            .find(|paired| paired.group == "all")
            .unwrap();
        assert_eq!(
            (paired.first, paired.second, paired.tasks),
            (Arm::A, Arm::B, 4)
        );
        assert!((paired.pass_rate.value - 0.25).abs() < 1e-9);
        assert!((paired.item_score.value - 0.375).abs() < 1e-9);
        assert!((paired.input_tokens.value + 1_075_000.0).abs() < 1e-6);
        assert_eq!((paired.wins, paired.ties, paired.losses), (2, 2, 0));

        // B is cheaper and better: A is off the frontier.
        let on = |arm: Arm| {
            report
                .frontier
                .iter()
                .find(|point| point.arm == arm && point.group == "all")
                .unwrap()
                .on_frontier
        };
        assert!(on(Arm::B));
        assert!(!on(Arm::A));
        assert_eq!(report.infra_attempts, 1);
        assert_eq!(report.ungraded_answers, 0);
        // The grader met L1 exactly where the location was right.
        assert_eq!(report.agreement.items, 7);
        assert_eq!(report.agreement.agree, 7);

        let markdown = render(&report);
        assert!(markdown.contains("| all | A | 4 | 3 (0 failed, 1 over limit) | 50%"));
        assert!(markdown.contains("A→B"));
        let csv = attempts_csv(&rows(&tasks, &attempts, &grades), &BTreeMap::new());
        assert_eq!(csv.lines().count(), 9);
    }

    #[test]
    fn time_is_working_time_and_samples_widen_rates() {
        let tasks: Vec<BenchTask> = (0..6)
            .map(|i| task(&format!("t{i}"), TaskKind::Definition))
            .collect();
        let right = "LOCATIONS:\nsrc/a.rs:12\n";
        let mut attempts = Vec::new();
        let mut grades = Vec::new();
        for task in &tasks {
            for sample in 0..3 {
                let mut a = attempt(&task.id, Arm::A, Outcome::Answered, right, 1000);
                a.attempt_id = format!("{}-{sample}", task.id);
                a.sample = sample;
                // Five minutes of work; arm A also waited three hours for a
                // usage limit on t0.
                a.active_ms = 300_000;
                a.wall_ms = if task.id == "t0" { 11_100_000 } else { 300_000 };
                // Samples of a task agree: t0..t2 pass, t3..t5 fail.
                let pass = task.id < "t3".to_string();
                grades.push(grade(&a, pass, pass));
                attempts.push(a);
            }
        }
        let manifest = crate::runner::RunManifest {
            name: "test".into(),
            created_at: chrono::Utc::now(),
            model: crate::agents::ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap(),
            mock: true,
            shell_timeout_seconds: 120,
            shell_max_output_bytes: 16384,
            shell_max_calls: 80,
            max_input_tokens: 4_000_000,
            attempt_timeout_seconds: None,
            grader_shell: true,
            index_cli_sha256: String::new(),
            index_cli: std::path::PathBuf::new(),
            arms: vec![Arm::A],
            samples: 3,
            tasks_file: "tasks.json".into(),
            sessions: vec![],
            variants: Default::default(),
        };
        let report = compute(
            &manifest,
            &tasks,
            &attempts,
            &grades,
            &BTreeMap::new(),
            2000,
        );
        let all = report
            .by_kind
            .iter()
            .find(|stats| stats.group == "all")
            .unwrap();
        assert!((all.work_seconds.value - 300.0).abs() < 1e-9);
        assert!((all.median_work_seconds - 300.0).abs() < 1e-9);
        // 18 cells, but six independent tasks: the interval is that of six
        // tasks, much wider than Wilson's for 18 cells.
        assert!((all.pass_rate.value - 0.5).abs() < 1e-9);
        let cells = wilson(9, 18);
        assert!(
            all.pass_rate.high - all.pass_rate.low > (cells.high - cells.low) * 1.2,
            "{:?} vs {cells:?}",
            all.pass_rate
        );
    }

    #[test]
    fn rng_shuffles_deterministically() {
        let mut a: Vec<u32> = (0..20).collect();
        let mut b = a.clone();
        Rng::new(42).shuffle(&mut a);
        Rng::new(42).shuffle(&mut b);
        assert_eq!(a, b);
        assert_ne!(a, (0..20).collect::<Vec<_>>());
    }
}
