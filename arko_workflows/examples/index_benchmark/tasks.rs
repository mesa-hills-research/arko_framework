//! Benchmark tasks: code-understanding questions with verified answer keys.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::BenchResult;

/// What a question asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskKind {
    /// Where something is defined.
    Definition,
    /// Who calls a function or method.
    Callers,
    /// How a feature or mechanism works.
    Behaviour,
    /// What a change would affect.
    Impact,
}

impl TaskKind {
    pub const ALL: [TaskKind; 4] = [
        TaskKind::Definition,
        TaskKind::Callers,
        TaskKind::Behaviour,
        TaskKind::Impact,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TaskKind::Definition => "definition",
            TaskKind::Callers => "callers",
            TaskKind::Behaviour => "behaviour",
            TaskKind::Impact => "impact",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "definition" | "def" => Some(TaskKind::Definition),
            "callers" | "caller" => Some(TaskKind::Callers),
            "behaviour" | "behavior" => Some(TaskKind::Behaviour),
            "impact" => Some(TaskKind::Impact),
            _ => None,
        }
    }

    /// Whether an answer is a list whose wrong entries count against it.
    pub fn is_list(self) -> bool {
        matches!(self, TaskKind::Callers | TaskKind::Impact)
    }
}

impl fmt::Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A line range in a file of the checkout: 1-based, inclusive.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
}

impl Location {
    pub fn new(path: impl Into<String>, start_line: u32, end_line: u32) -> Self {
        Self {
            path: path.into(),
            start_line,
            end_line: end_line.max(start_line),
        }
    }

    /// Parses `path:line` or `path:start-end` (also `path#Lstart-Lend`,
    /// `path:line:column`, and a leading `./`).
    pub fn parse(text: &str) -> Option<Self> {
        let text = text
            .trim()
            .trim_matches(|c: char| c == '`' || c == '"' || c == '\'');
        let text = text.strip_prefix("./").unwrap_or(text);
        let (path, lines) = if let Some((path, lines)) = text.split_once('#') {
            (path, lines.replace('L', ""))
        } else {
            let (path, lines) = text.split_once(':')?;
            (path, lines.to_string())
        };
        let path = path.trim();
        if path.is_empty() || path.contains(char::is_whitespace) {
            return None;
        }
        let lines = lines.trim();
        let (start, end) = match lines.split_once('-') {
            Some((start, end)) => (leading_number(start)?, leading_number(end)),
            None => (leading_number(lines)?, None),
        };
        let end = end.unwrap_or(start);
        if start == 0 {
            return None;
        }
        Some(Location::new(path, start, end))
    }

    /// Whether `other` (a cited location) touches this range widened by
    /// `tolerance` lines on each side.
    pub fn matches(&self, other: &Location, tolerance: u32) -> bool {
        normalize_path(&self.path) == normalize_path(&other.path)
            && other.start_line <= self.end_line.saturating_add(tolerance)
            && other.end_line.saturating_add(tolerance) >= self.start_line
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start_line == self.end_line {
            write!(f, "{}:{}", self.path, self.start_line)
        } else {
            write!(f, "{}:{}-{}", self.path, self.start_line, self.end_line)
        }
    }
}

fn leading_number(text: &str) -> Option<u32> {
    let digits: String = text
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// A path as written in the checkout: slashes, no leading `./` or `/`.
pub fn normalize_path(path: &str) -> String {
    let path = path.trim().replace('\\', "/");
    let path = path.strip_prefix("./").unwrap_or(&path);
    path.trim_start_matches('/').to_string()
}

/// What a key item is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemKind {
    /// Code the answer must identify; checked deterministically too.
    Location,
    /// A file the answer must name; checked deterministically too.
    File,
    /// A statement the answer must make; checked by the grader only.
    Fact,
}

/// One required element of a correct answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyItem {
    pub id: String,
    pub kind: ItemKind,
    /// The item in words: what the location is, the fact, why the file.
    pub text: String,
    /// For location items, the range; for file items, the file (lines unused).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
}

/// The answer key of a task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerKey {
    /// A reference answer in prose.
    pub answer: String,
    pub items: Vec<KeyItem>,
    /// Plausible but wrong answers (for example a same-named item of
    /// another type). Citing one first fails the deterministic check.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distractors: Vec<KeyItem>,
}

impl AnswerKey {
    /// The key as the grader and the verifier read it.
    pub fn render(&self) -> String {
        let mut text = format!(
            "Reference answer: {}\nRequired items:\n",
            self.answer.trim()
        );
        for item in &self.items {
            text.push_str(&format!("[{}] {}\n", item.id, render_item(item)));
        }
        if !self.distractors.is_empty() {
            text.push_str(
                "Known wrong answers (giving one of these as the answer is a wrong claim):\n",
            );
            for item in &self.distractors {
                text.push_str(&format!("- {}\n", render_item(item)));
            }
        }
        text
    }
}

fn render_item(item: &KeyItem) -> String {
    match (item.kind, &item.location) {
        (ItemKind::Location, Some(location)) => format!("location {location}: {}", item.text),
        (ItemKind::File, Some(location)) => format!("file {}: {}", location.path, item.text),
        _ => format!("fact: {}", item.text),
    }
}

/// Where a task came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Origin {
    /// `seed` (converted from an earlier question set) or `generated`.
    pub source: String,
    /// The question set and id, or the writer run.
    #[serde(default)]
    pub reference: String,
}

/// One benchmark task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchTask {
    pub id: String,
    /// A repository id from the manifest.
    pub repo: String,
    pub kind: TaskKind,
    pub question: String,
    /// What counts, when it matters ("non-test code").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub key: AnswerKey,
    #[serde(default)]
    pub origin: Origin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<String>,
    /// The verifier's record, for generated tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<serde_json::Value>,
}

/// A task file: a JSON array, or a `{"tasks": [...]}` object, or JSON lines.
pub fn load_tasks(path: &Path) -> BenchResult<Vec<BenchTask>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("reading {}: {error}", path.display()))?;
    parse_tasks(&text).map_err(|error| format!("{}: {error}", path.display()).into())
}

pub fn parse_tasks(text: &str) -> BenchResult<Vec<BenchTask>> {
    let trimmed = text.trim_start();
    let tasks: Vec<BenchTask> = if trimmed.starts_with('[') {
        serde_json::from_str(text)?
    } else if trimmed.starts_with('{') && !looks_like_json_lines(text) {
        #[derive(Deserialize)]
        struct Wrapped {
            tasks: Vec<BenchTask>,
        }
        serde_json::from_str::<Wrapped>(text)?.tasks
    } else {
        let mut tasks = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            tasks.push(
                serde_json::from_str(line)
                    .map_err(|error| format!("line {}: {error}", index + 1))?,
            );
        }
        tasks
    };
    let mut seen = HashSet::new();
    for task in &tasks {
        if !seen.insert(task.id.as_str()) {
            return Err(format!("duplicate task id {}", task.id).into());
        }
        check_shape(task).map_err(|error| format!("task {}: {error}", task.id))?;
    }
    Ok(tasks)
}

fn looks_like_json_lines(text: &str) -> bool {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let first = lines.next();
    let second = lines.next();
    matches!((first, second), (Some(a), Some(b)) if a.trim_end().ends_with('}') && b.trim_start().starts_with('{'))
}

pub fn save_tasks(path: &Path, tasks: &[BenchTask]) -> BenchResult<()> {
    let text = serde_json::to_string_pretty(tasks)?;
    crate::write_atomic(path, text.as_bytes())
}

/// Structural checks that need no checkout.
pub fn check_shape(task: &BenchTask) -> Result<(), String> {
    if task.id.trim().is_empty() || task.repo.trim().is_empty() {
        return Err("empty id or repo".into());
    }
    if task.question.trim().is_empty() {
        return Err("empty question".into());
    }
    if task.key.items.is_empty() {
        return Err("the key has no items".into());
    }
    let mut ids = HashSet::new();
    for item in task.key.items.iter().chain(&task.key.distractors) {
        if !item.id.is_empty() && !ids.insert(item.id.as_str()) {
            return Err(format!("duplicate item id {}", item.id));
        }
        match item.kind {
            ItemKind::Location | ItemKind::File if item.location.is_none() => {
                return Err(format!("item {} has no location", item.id));
            }
            _ => {}
        }
    }
    let locations = task
        .key
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::Location)
        .count();
    let facts = task
        .key
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::Fact)
        .count();
    match task.kind {
        TaskKind::Definition if locations == 0 => Err("a definition key needs its location".into()),
        TaskKind::Callers if locations == 0 => Err("a callers key needs its call sites".into()),
        TaskKind::Behaviour if facts == 0 => Err("a behaviour key needs facts".into()),
        TaskKind::Impact if locations == 0 => Err("an impact key needs locations".into()),
        _ => Ok(()),
    }
}

/// A problem found by [`validate_against_checkout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub task: String,
    pub message: String,
}

/// Checks every location of `tasks` against the checkouts in `repos`
/// (repo id -> checkout root): the file exists and the lines are in it.
/// Tasks whose repo has no checkout are reported once each.
pub fn validate_against_checkout(
    tasks: &[BenchTask],
    repos: &HashMap<String, std::path::PathBuf>,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let mut line_counts: HashMap<std::path::PathBuf, Option<u32>> = HashMap::new();
    for task in tasks {
        let Some(root) = repos.get(&task.repo) else {
            issues.push(ValidationIssue {
                task: task.id.clone(),
                message: format!("repository {} is not prepared", task.repo),
            });
            continue;
        };
        for item in task.key.items.iter().chain(&task.key.distractors) {
            let Some(location) = &item.location else {
                continue;
            };
            if let Err(message) = check_location(root, location, &mut line_counts) {
                issues.push(ValidationIssue {
                    task: task.id.clone(),
                    message: format!("item {}: {message}", item.id),
                });
            }
        }
    }
    issues
}

/// Checks that `location` names a file of the checkout at `root` and lines
/// within it.
pub fn check_location(
    root: &Path,
    location: &Location,
    cache: &mut HashMap<std::path::PathBuf, Option<u32>>,
) -> Result<(), String> {
    let relative = normalize_path(&location.path);
    if relative.is_empty() || relative.split('/').any(|part| part == "..") {
        return Err(format!(
            "{} is not a path inside the repository",
            location.path
        ));
    }
    let path = root.join(&relative);
    let lines = *cache
        .entry(path.clone())
        .or_insert_with(|| std::fs::read(&path).ok().map(|bytes| count_lines(&bytes)));
    let Some(lines) = lines else {
        return Err(format!("{relative} does not exist"));
    };
    if location.start_line == 0 || location.end_line < location.start_line {
        return Err(format!("{location}: bad line range"));
    }
    if location.end_line > lines {
        return Err(format!("{location}: the file has {lines} lines"));
    }
    Ok(())
}

/// Lines in a file, counting a last line without a newline.
pub fn count_lines(bytes: &[u8]) -> u32 {
    let newlines = bytes.iter().filter(|&&byte| byte == b'\n').count() as u32;
    if bytes.last().is_some_and(|&byte| byte != b'\n') {
        newlines + 1
    } else {
        newlines
    }
}

/// Task counts by (repo, kind), for summaries.
pub fn counts(tasks: &[BenchTask]) -> BTreeMap<(String, TaskKind), usize> {
    let mut counts = BTreeMap::new();
    for task in tasks {
        *counts.entry((task.repo.clone(), task.kind)).or_insert(0) += 1;
    }
    counts
}

/// A table of task counts per repository and kind.
pub fn count_table(tasks: &[BenchTask], languages: &HashMap<String, String>) -> String {
    let counts = counts(tasks);
    let repos: Vec<String> = counts
        .keys()
        .map(|(repo, _)| repo.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut text = String::from(
        "| repo | language | definition | callers | behaviour | impact | total |\n|---|---|---|---|---|---|---|\n",
    );
    let mut totals = [0usize; 4];
    for repo in &repos {
        let mut row_total = 0;
        let mut cells = Vec::new();
        for (index, kind) in TaskKind::ALL.iter().enumerate() {
            let count = counts.get(&(repo.clone(), *kind)).copied().unwrap_or(0);
            totals[index] += count;
            row_total += count;
            cells.push(count.to_string());
        }
        text.push_str(&format!(
            "| {repo} | {} | {} | {row_total} |\n",
            languages.get(repo).map_or("?", String::as_str),
            cells.join(" | ")
        ));
    }
    text.push_str(&format!(
        "| total | | {} | {} | {} | {} | {} |\n",
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals.iter().sum::<usize>()
    ));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, kind: TaskKind, items: Vec<KeyItem>) -> BenchTask {
        BenchTask {
            id: id.into(),
            repo: "demo".into(),
            kind,
            question: "Where is it?".into(),
            scope: None,
            key: AnswerKey {
                answer: "Here.".into(),
                items,
                distractors: vec![],
            },
            origin: Origin::default(),
            difficulty: None,
            verification: None,
        }
    }

    fn location_item(id: &str, path: &str, start: u32, end: u32) -> KeyItem {
        KeyItem {
            id: id.into(),
            kind: ItemKind::Location,
            text: "the code".into(),
            location: Some(Location::new(path, start, end)),
        }
    }

    #[test]
    fn parses_locations_in_common_forms() {
        assert_eq!(
            Location::parse("src/a.rs:12"),
            Some(Location::new("src/a.rs", 12, 12))
        );
        assert_eq!(
            Location::parse("`./src/a.rs:12-40`"),
            Some(Location::new("src/a.rs", 12, 40))
        );
        assert_eq!(
            Location::parse("src/a.rs:12:5"),
            Some(Location::new("src/a.rs", 12, 12))
        );
        assert_eq!(
            Location::parse("src/a.rs#L3-L9"),
            Some(Location::new("src/a.rs", 3, 9))
        );
        assert_eq!(Location::parse("src/a.rs"), None);
        assert_eq!(Location::parse("src/a.rs:0"), None);
        assert_eq!(Location::parse("not a path:3"), None);
    }

    #[test]
    fn location_matching_uses_the_tolerance() {
        let key = Location::new("src/a.rs", 100, 120);
        assert!(key.matches(&Location::new("src/a.rs", 110, 110), 0));
        assert!(key.matches(&Location::new("./src/a.rs", 97, 98), 3));
        assert!(!key.matches(&Location::new("src/a.rs", 90, 95), 3));
        assert!(!key.matches(&Location::new("src/b.rs", 110, 110), 3));
        assert!(key.matches(&Location::new("src/a.rs", 1, 500), 0));
    }

    #[test]
    fn loads_arrays_wrapped_objects_and_json_lines() {
        let one = task(
            "t1",
            TaskKind::Definition,
            vec![location_item("L1", "a.rs", 1, 2)],
        );
        let two = task(
            "t2",
            TaskKind::Callers,
            vec![location_item("L1", "b.rs", 3, 3)],
        );
        let tasks = vec![one, two];
        let array = serde_json::to_string_pretty(&tasks).unwrap();
        assert_eq!(parse_tasks(&array).unwrap(), tasks);
        let wrapped = serde_json::json!({ "tasks": tasks }).to_string();
        assert_eq!(parse_tasks(&wrapped).unwrap(), tasks);
        let lines = tasks
            .iter()
            .map(|task| serde_json::to_string(task).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(parse_tasks(&lines).unwrap(), tasks);
    }

    #[test]
    fn rejects_duplicate_ids_and_keys_without_their_items() {
        let one = task(
            "t1",
            TaskKind::Definition,
            vec![location_item("L1", "a.rs", 1, 2)],
        );
        let text = serde_json::to_string(&vec![one.clone(), one]).unwrap();
        assert!(
            parse_tasks(&text)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        let facts_only = task(
            "t3",
            TaskKind::Callers,
            vec![KeyItem {
                id: "F1".into(),
                kind: ItemKind::Fact,
                text: "a fact".into(),
                location: None,
            }],
        );
        let text = serde_json::to_string(&vec![facts_only]).unwrap();
        assert!(
            parse_tasks(&text)
                .unwrap_err()
                .to_string()
                .contains("call sites")
        );
    }

    #[test]
    fn validates_locations_against_a_checkout() {
        let root = crate::test_dir("validate");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "fn a() {}\nfn b() {}\nfn c() {}").unwrap();
        let good = task(
            "good",
            TaskKind::Definition,
            vec![location_item("L1", "src/a.rs", 2, 3)],
        );
        let past_end = task(
            "past-end",
            TaskKind::Definition,
            vec![location_item("L1", "src/a.rs", 3, 4)],
        );
        let missing = task(
            "missing",
            TaskKind::Definition,
            vec![location_item("L1", "src/nope.rs", 1, 1)],
        );
        let mut other_repo = good.clone();
        other_repo.id = "other".into();
        other_repo.repo = "elsewhere".into();
        let repos = HashMap::from([("demo".to_string(), root.clone())]);
        let issues = validate_against_checkout(&[good, past_end, missing, other_repo], &repos);
        let by_task: Vec<&str> = issues.iter().map(|issue| issue.task.as_str()).collect();
        assert_eq!(by_task, vec!["past-end", "missing", "other"]);
        assert!(issues[0].message.contains("has 3 lines"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn counts_lines_with_and_without_a_final_newline() {
        assert_eq!(count_lines(b""), 0);
        assert_eq!(count_lines(b"a"), 1);
        assert_eq!(count_lines(b"a\n"), 1);
        assert_eq!(count_lines(b"a\nb"), 2);
    }
}
