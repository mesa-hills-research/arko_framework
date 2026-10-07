//! Answers: parsing what a solver returns, the deterministic checks, and
//! the grader's and verifier's JSON replies.

use serde::{Deserialize, Serialize};

use crate::tasks::{AnswerKey, BenchTask, ItemKind, Location, normalize_path};

/// Lines of slack when a cited location is compared with a key location.
pub const LINE_TOLERANCE: u32 = 3;

/// A solver's answer, split into its sections.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ParsedAnswer {
    /// The ANSWER section (or the whole text when there are no sections).
    pub answer: String,
    /// The LOCATIONS section's locations, in order.
    pub locations: Vec<Location>,
    /// Locations mentioned anywhere in the text, LOCATIONS first.
    pub mentioned: Vec<Location>,
}

/// Splits `text` into its ANSWER and LOCATIONS sections. Lenient: headings
/// may carry markdown (`**ANSWER:**`, `## Locations`), and locations may be
/// list items with descriptions after them.
pub fn parse_answer(text: &str) -> ParsedAnswer {
    let mut answer_lines = Vec::new();
    let mut location_lines = Vec::new();
    let mut section = None;
    for line in text.lines() {
        match heading(line) {
            Some(Heading::Answer(rest)) => {
                section = Some(Heading::Answer(""));
                if !rest.is_empty() {
                    answer_lines.push(rest.to_string());
                }
                continue;
            }
            Some(Heading::Locations(rest)) => {
                section = Some(Heading::Locations(""));
                if !rest.is_empty() {
                    location_lines.push(rest.to_string());
                }
                continue;
            }
            None => {}
        }
        match section {
            Some(Heading::Locations(_)) => location_lines.push(line.to_string()),
            _ => answer_lines.push(line.to_string()),
        }
    }
    let answer = answer_lines.join("\n").trim().to_string();
    let locations: Vec<Location> = location_lines
        .iter()
        .flat_map(|line| locations_in(line))
        .collect();
    let mut mentioned = locations.clone();
    for location in locations_in(&answer) {
        if !mentioned.contains(&location) {
            mentioned.push(location);
        }
    }
    ParsedAnswer {
        answer,
        locations,
        mentioned,
    }
}

enum Heading<'a> {
    Answer(&'a str),
    Locations(&'a str),
}

fn heading(line: &str) -> Option<Heading<'_>> {
    let stripped = line
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_start_matches("**")
        .trim();
    let upper = stripped.to_ascii_uppercase();
    for (name, is_answer) in [("ANSWER", true), ("LOCATIONS", false)] {
        if let Some(rest) = upper.strip_prefix(name) {
            let rest_original = &stripped[name.len()..];
            if !(rest.starts_with(':') || rest.trim_start_matches('*').trim().is_empty()) {
                continue;
            }
            let rest_original = rest_original
                .trim_start_matches('*')
                .trim_start_matches(':')
                .trim_start_matches('*')
                .trim();
            return Some(if is_answer {
                Heading::Answer(rest_original)
            } else {
                Heading::Locations(rest_original)
            });
        }
    }
    None
}

/// Every `path:line` or `path:start-end` in `text`. A path must contain a
/// dot or a slash, so prose such as `note: 3` is not taken for one.
pub fn locations_in(text: &str) -> Vec<Location> {
    let mut found = Vec::new();
    for raw in text.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '[' | ']' | '<' | '>' | '|' | '*')
    }) {
        let token = raw
            .trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | '.' | ':' | '-' | '—'))
            .trim();
        if token.is_empty() || token.starts_with("http") {
            continue;
        }
        let Some(location) = Location::parse(token) else {
            continue;
        };
        if !(location.path.contains('.') || location.path.contains('/')) {
            continue;
        }
        if !found.contains(&location) {
            found.push(location);
        }
    }
    found
}

/// The deterministic part of grading: the key's location and file items
/// checked against the answer's locations and text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicCheck {
    /// Location and file items in the key.
    pub items: usize,
    /// Of those, the ones the answer meets.
    pub met: usize,
    /// `met / items`, or `None` when the key has no checkable items.
    pub score: Option<f64>,
    /// For list questions: cited locations that match no key item.
    pub unmatched_locations: usize,
    /// For list questions: matched cited locations / cited locations.
    pub precision: Option<f64>,
    /// The first cited location is a known wrong answer.
    pub distractor_first: bool,
    /// Every checkable item met, and no distractor first.
    pub pass: bool,
}

/// Whether a cited location identifies a key location: they overlap (with
/// [`LINE_TOLERANCE`] lines of slack), and the citation is at most three
/// times as long as the key's range plus 30 lines, so citing a whole file
/// or a large chunk doesn't meet every item inside it.
pub fn cited_matches(key: &Location, cited: &Location, tolerance: u32) -> bool {
    let key_lines = key.end_line - key.start_line + 1;
    let cited_lines = cited.end_line - cited.start_line + 1;
    key.matches(cited, tolerance) && cited_lines <= key_lines * 3 + 30
}

/// Checks `answer` against the location and file items of `task`'s key.
pub fn deterministic_check(task: &BenchTask, answer: &ParsedAnswer) -> DeterministicCheck {
    check_key(&task.key, task.kind.is_list(), answer)
}

pub fn check_key(key: &AnswerKey, is_list: bool, answer: &ParsedAnswer) -> DeterministicCheck {
    let cited = if answer.locations.is_empty() {
        &answer.mentioned
    } else {
        &answer.locations
    };
    let text_paths: Vec<String> = answer
        .mentioned
        .iter()
        .map(|location| normalize_path(&location.path))
        .collect();
    let answer_text = answer.answer.replace('\\', "/");
    let mut items = 0;
    let mut met = 0;
    for item in &key.items {
        let Some(location) = &item.location else {
            continue;
        };
        match item.kind {
            ItemKind::Location => {
                items += 1;
                if cited
                    .iter()
                    .any(|cited| cited_matches(location, cited, LINE_TOLERANCE))
                {
                    met += 1;
                }
            }
            ItemKind::File => {
                items += 1;
                let path = normalize_path(&location.path);
                if text_paths.contains(&path) || answer_text.contains(&path) {
                    met += 1;
                }
            }
            ItemKind::Fact => {}
        }
    }
    let key_locations: Vec<&Location> = key
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::Location)
        .filter_map(|item| item.location.as_ref())
        .collect();
    let (unmatched_locations, precision) = if is_list && !cited.is_empty() {
        let matched = cited
            .iter()
            .filter(|cited| {
                key_locations
                    .iter()
                    .any(|location| cited_matches(location, cited, LINE_TOLERANCE))
            })
            .count();
        (
            cited.len() - matched,
            Some(matched as f64 / cited.len() as f64),
        )
    } else {
        (0, None)
    };
    let distractor_first = cited.first().is_some_and(|first| {
        key.distractors.iter().any(|item| {
            item.location
                .as_ref()
                .is_some_and(|location| cited_matches(location, first, 0))
        })
    });
    let score = (items > 0).then(|| met as f64 / items as f64);
    DeterministicCheck {
        items,
        met,
        score,
        unmatched_locations,
        precision,
        distractor_first,
        pass: items > 0 && met == items && !distractor_first,
    }
}

/// Removes descriptions of search commands from an answer before the
/// grader sees it, so the grader can't tell the arms apart. A line is
/// dropped when it names the arms' search tools in a way only a command can
/// (`$ARKO_INDEX_STATE`, `arko_index_dense`), or names a search command
/// (`arko_index search2`, `rg -n`, `grep -r`, `git grep`, ...) together with
/// words describing the search ("I ran", "searched", "using `"). Other
/// mentions stay, so answers about the code of these tools keep their
/// content. The same rule applies to every arm.
pub fn scrub_for_grader(text: &str) -> (String, bool) {
    const TOOL_ONLY: [&str; 2] = ["ARKO_INDEX_STATE", "arko_index_dense"];
    const COMMANDS: [&str; 7] = [
        "arko_index search2",
        "arko_index outline",
        "rg -",
        "`rg ",
        "grep -r",
        "grep -n",
        "git grep",
    ];
    const PROCESS: [&str; 12] = [
        "i ran",
        "i used",
        "i searched",
        "searched",
        "searching",
        "ran `",
        "running `",
        "using `",
        "with `",
        "via `",
        "found with",
        "found by",
    ];
    let mut scrubbed = false;
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| {
            let lower = line.to_lowercase();
            let drop = TOOL_ONLY.iter().any(|marker| line.contains(marker))
                || (COMMANDS.iter().any(|command| line.contains(command))
                    && PROCESS.iter().any(|word| lower.contains(word)));
            scrubbed |= drop;
            !drop
        })
        .collect();
    (lines.join("\n"), scrubbed)
}

/// The grader's judgment of one item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemJudgment {
    pub id: String,
    pub met: bool,
    #[serde(default)]
    pub note: String,
}

/// The grader's reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraderVerdict {
    pub items: Vec<ItemJudgment>,
    #[serde(default)]
    pub wrong_claims: Vec<String>,
    pub verdict: String,
    #[serde(default)]
    pub notes: String,
}

impl GraderVerdict {
    /// Items met / items in the key (items the grader skipped count as
    /// unmet).
    pub fn item_score(&self, key: &AnswerKey) -> f64 {
        if key.items.is_empty() {
            return 0.0;
        }
        let met = key
            .items
            .iter()
            .filter(|item| {
                self.items
                    .iter()
                    .any(|judgment| judgment.id == item.id && judgment.met)
            })
            .count();
        met as f64 / key.items.len() as f64
    }

    pub fn pass(&self) -> bool {
        self.verdict == "correct"
    }
}

/// Parses the grader's reply: the first JSON object in `text` with the
/// expected fields. Checks that every key item is judged.
pub fn parse_grader_verdict(text: &str, key: &AnswerKey) -> Result<GraderVerdict, String> {
    let mut verdict: GraderVerdict = first_json_object(text)?;
    verdict.verdict = verdict
        .verdict
        .trim()
        .to_ascii_lowercase()
        .replace(' ', "_");
    if !matches!(
        verdict.verdict.as_str(),
        "correct" | "partially_correct" | "incorrect"
    ) {
        return Err(format!("unknown verdict {:?}", verdict.verdict));
    }
    let missing: Vec<&str> = key
        .items
        .iter()
        .filter(|item| !verdict.items.iter().any(|judgment| judgment.id == item.id))
        .map(|item| item.id.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(format!("items not judged: {}", missing.join(", ")));
    }
    Ok(verdict)
}

/// The verifier's verdict on a key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierVerdict {
    pub verdict: String,
    #[serde(default)]
    pub problems: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

pub fn parse_verifier_verdict(text: &str) -> Result<VerifierVerdict, String> {
    let mut verdict: VerifierVerdict = first_json_object(text)?;
    verdict.verdict = verdict.verdict.trim().to_ascii_lowercase();
    if !matches!(verdict.verdict.as_str(), "agree" | "disagree" | "ambiguous") {
        return Err(format!("unknown verdict {:?}", verdict.verdict));
    }
    Ok(verdict)
}

/// The first JSON object in `text` that parses as `T` (models sometimes
/// wrap it in a code fence or add a sentence).
fn first_json_object<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, String> {
    let mut last_error = String::from("no JSON object in the reply");
    for (start, _) in text.match_indices('{') {
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for (offset, c) in text[start..].char_indices() {
            if in_string {
                match (escaped, c) {
                    (true, _) => escaped = false,
                    (false, '\\') => escaped = true,
                    (false, '"') => in_string = false,
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let candidate = &text[start..start + offset + 1];
                        match serde_json::from_str::<T>(candidate) {
                            Ok(value) => return Ok(value),
                            Err(error) => last_error = error.to_string(),
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{KeyItem, TaskKind};

    fn key(items: Vec<KeyItem>, distractors: Vec<KeyItem>) -> AnswerKey {
        AnswerKey {
            answer: "ref".into(),
            items,
            distractors,
        }
    }

    fn item(id: &str, kind: ItemKind, location: Option<Location>) -> KeyItem {
        KeyItem {
            id: id.into(),
            kind,
            text: "x".into(),
            location,
        }
    }

    #[test]
    fn parses_sections_with_markdown_headings() {
        let text = "**ANSWER:** It is `Foo::bar`, see src/foo.rs:10.\nMore prose.\n\n## Locations\n- `src/foo.rs:10-20` the method\n- src/lib.rs:3\n";
        let parsed = parse_answer(text);
        assert!(parsed.answer.starts_with("It is `Foo::bar`"));
        assert!(parsed.answer.contains("More prose."));
        assert_eq!(
            parsed.locations,
            vec![
                Location::new("src/foo.rs", 10, 20),
                Location::new("src/lib.rs", 3, 3)
            ]
        );
        assert_eq!(parsed.mentioned.len(), 3);
    }

    #[test]
    fn an_answer_without_sections_is_all_answer() {
        let parsed = parse_answer("Defined in a/b.py:7 as a class.");
        assert_eq!(parsed.answer, "Defined in a/b.py:7 as a class.");
        assert!(parsed.locations.is_empty());
        assert_eq!(parsed.mentioned, vec![Location::new("a/b.py", 7, 7)]);
    }

    #[test]
    fn prose_with_colons_is_not_a_location() {
        assert!(locations_in("note: 3 cases; step:2").is_empty());
        assert!(locations_in("https://example.com:8080/x").is_empty());
    }

    #[test]
    fn deterministic_check_scores_locations_files_and_distractors() {
        let key = key(
            vec![
                item(
                    "L1",
                    ItemKind::Location,
                    Some(Location::new("src/a.rs", 10, 30)),
                ),
                item(
                    "L2",
                    ItemKind::Location,
                    Some(Location::new("src/b.rs", 5, 5)),
                ),
                item("P1", ItemKind::File, Some(Location::new("src/c.rs", 1, 1))),
                item("F1", ItemKind::Fact, None),
            ],
            vec![item(
                "D1",
                ItemKind::Location,
                Some(Location::new("src/z.rs", 1, 9)),
            )],
        );
        let answer = parse_answer("ANSWER:\nUses src/c.rs.\nLOCATIONS:\nsrc/a.rs:12\nsrc/b.rs:9\n");
        let check = check_key(&key, false, &answer);
        assert_eq!((check.items, check.met), (3, 2));
        assert!(!check.pass);
        assert!((check.score.unwrap() - 2.0 / 3.0).abs() < 1e-9);

        let wrong_first =
            parse_answer("LOCATIONS:\nsrc/z.rs:4\nsrc/a.rs:12\nsrc/b.rs:5\nsrc/c.rs:1\n");
        let check = check_key(&key, false, &wrong_first);
        assert_eq!(check.met, 3);
        assert!(check.distractor_first);
        assert!(!check.pass);

        let right = parse_answer("LOCATIONS:\nsrc/a.rs:12\nsrc/b.rs:5\nsrc/c.rs:1\n");
        assert!(check_key(&key, false, &right).pass);
    }

    #[test]
    fn broad_citations_meet_no_item() {
        let key = key(
            vec![item(
                "L1",
                ItemKind::Location,
                Some(Location::new("src/a.rs", 100, 110)),
            )],
            vec![],
        );
        // The function itself, or a chunk around it, meets the item.
        assert!(check_key(&key, false, &parse_answer("LOCATIONS:\nsrc/a.rs:95-140\n")).pass);
        // A citation of the whole file doesn't.
        assert!(!check_key(&key, false, &parse_answer("LOCATIONS:\nsrc/a.rs:1-900\n")).pass);
    }

    #[test]
    fn list_questions_report_precision() {
        let key = key(
            vec![
                item("L1", ItemKind::Location, Some(Location::new("a.c", 10, 10))),
                item("L2", ItemKind::Location, Some(Location::new("b.c", 20, 20))),
            ],
            vec![],
        );
        let answer = parse_answer("LOCATIONS:\na.c:11\nb.c:20\nc.c:5\nd.c:6\n");
        let check = check_key(&key, true, &answer);
        assert_eq!(check.met, 2);
        assert_eq!(check.unmatched_locations, 2);
        assert_eq!(check.precision, Some(0.5));
        assert!(check.pass);
    }

    #[test]
    fn scrub_removes_search_tool_lines_only() {
        let (text, scrubbed) = scrub_for_grader(
            "ANSWER:\nI ran arko_index search2 \"$ARKO_INDEX_STATE\" 'x'.\nIt is in a.rs.",
        );
        assert!(scrubbed);
        assert_eq!(text, "ANSWER:\nIt is in a.rs.");
        // Describing the CLI's own code keeps the line.
        let about = "ANSWER:\n`arko_index search2` reads only the postings of the query's terms.";
        assert_eq!(scrub_for_grader(about), (about.to_string(), false));
        let (text, scrubbed) = scrub_for_grader("ANSWER:\nIt is in arko_index/src/lib.rs:3.");
        assert!(!scrubbed);
        assert!(text.contains("arko_index/src/lib.rs"));
        let (text, scrubbed) = scrub_for_grader("Found with `rg -n foo`.\nIt is foo.");
        assert!(scrubbed);
        assert_eq!(text, "It is foo.");
    }

    #[test]
    fn parses_grader_verdicts_and_requires_every_item() {
        let key = key(
            vec![
                item("L1", ItemKind::Location, Some(Location::new("a.rs", 1, 2))),
                item("F1", ItemKind::Fact, None),
            ],
            vec![],
        );
        let reply = "Here you go:\n```json\n{\"items\": [{\"id\": \"L1\", \"met\": true, \"note\": \"ok {braces}\"}, {\"id\": \"F1\", \"met\": false}], \"wrong_claims\": [], \"verdict\": \"Partially Correct\", \"notes\": \"n\"}\n```";
        let verdict = parse_grader_verdict(reply, &key).unwrap();
        assert_eq!(verdict.verdict, "partially_correct");
        assert!((verdict.item_score(&key) - 0.5).abs() < 1e-9);
        assert!(!verdict.pass());

        let missing = "{\"items\": [{\"id\": \"L1\", \"met\": true}], \"verdict\": \"correct\"}";
        assert!(
            parse_grader_verdict(missing, &key)
                .unwrap_err()
                .contains("F1")
        );
        assert!(parse_grader_verdict("no json", &key).is_err());
    }

    #[test]
    fn parses_verifier_verdicts() {
        let verdict = parse_verifier_verdict("{\"verdict\": \"Agree\", \"problems\": []}").unwrap();
        assert_eq!(verdict.verdict, "agree");
        assert!(parse_verifier_verdict("{\"verdict\": \"maybe\"}").is_err());
    }

    #[test]
    fn deterministic_check_uses_mentions_without_a_locations_section() {
        let task = crate::tasks::BenchTask {
            id: "t".into(),
            repo: "r".into(),
            kind: TaskKind::Definition,
            question: "q".into(),
            scope: None,
            key: key(
                vec![item(
                    "L1",
                    ItemKind::Location,
                    Some(Location::new("x/y.go", 40, 60)),
                )],
                vec![],
            ),
            origin: Default::default(),
            difficulty: None,
            verification: None,
        };
        let answer = parse_answer("It's the method at x/y.go:41.");
        assert!(deterministic_check(&task, &answer).pass);
    }
}
