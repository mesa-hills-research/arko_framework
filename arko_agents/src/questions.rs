//! The questions an agent asks the user with `ask_user`, and the texts that
//! carry them and their answers.
//!
//! Each question gets an id that is unique in its task (`q1`, `q2`, ...).
//! The task keeps them in its [`QuestionLog`], which a checkpoint saves.
//! With [`AgentToolConfig::async_questions`](crate::agent_config::AgentToolConfig::async_questions)
//! on, `ask_user` returns at once (unless the agent passes `wait: true`) and
//! the host delivers the answers later, as context such as a note. The
//! helpers here write those notes in one format: [`answer_note`],
//! [`dismissed_note`], [`message_answers`] and [`still_open`]. The agent
//! settles a question itself with `close_questions`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The most choices a question shows. More are cut.
pub const MAX_CHOICES: usize = 6;

/// A question the agent asks the user.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    /// The question, with why it matters.
    pub question: String,
    /// Suggested answers, at most [`MAX_CHOICES`]. The user may still answer
    /// in their own words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
    /// The user may pick several of the choices.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub multiple: bool,
    /// What the agent assumes if the user doesn't answer, and what "you
    /// decide" means.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

/// A question with its id in the task, as it was asked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskedQuestion {
    /// `q1`, `q2`, ... in the order the task asked them.
    pub id: String,
    #[serde(flatten)]
    pub question: Question,
}

/// The questions a task asked, saved with it.
///
/// `open` holds the questions that are open as far as the runtime knows.
/// It drops those the agent closed (`close_questions`), those answered while
/// the agent waited, and those a host settles with [`settle`](Self::settle).
/// A host that delivers answers later as context tracks them itself, and
/// tells a handover which are still open through
/// [`HandoverHook::open_questions`](crate::handover::HandoverHook::open_questions).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionLog {
    /// How many questions the task has asked. The next one is `q{asked + 1}`.
    #[serde(default)]
    pub asked: u32,
    /// The questions still open.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open: Vec<AskedQuestion>,
    /// The ids of the questions the agent waits for (`wait: true`, or a
    /// host that answers at once). The answer settles them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub waiting: Vec<String>,
}

impl QuestionLog {
    /// The task never asked.
    pub fn is_empty(&self) -> bool {
        self.asked == 0 && self.open.is_empty() && self.waiting.is_empty()
    }

    /// Records `questions` as asked and open, each with the next id.
    pub fn ask(&mut self, questions: Vec<Question>) -> Vec<AskedQuestion> {
        let asked = numbered(questions, self.asked);
        self.record(&asked, false);
        asked
    }

    /// Records questions numbered before (see [`ask`](Self::ask)) as asked
    /// and open, and `wait`ed for. Later questions are numbered after them.
    pub fn record(&mut self, asked: &[AskedQuestion], wait: bool) {
        let last = asked.iter().filter_map(|question| number_of(&question.id));
        self.asked = last.fold(self.asked, u32::max);
        self.open.extend(asked.iter().cloned());
        if wait {
            self.waiting = asked.iter().map(|question| question.id.clone()).collect();
        }
    }

    /// Whether the task asked a question with this id.
    pub fn was_asked(&self, id: &str) -> bool {
        was_asked(id, self.asked)
    }

    /// The open question `id`.
    pub fn get(&self, id: &str) -> Option<&AskedQuestion> {
        self.open.iter().find(|question| question.id == id)
    }

    /// Settles the questions `ids` (answered, dismissed or closed): they are
    /// no longer open. Returns those that were.
    pub fn settle<S: AsRef<str>>(&mut self, ids: &[S]) -> Vec<AskedQuestion> {
        let mut settled = Vec::new();
        self.open.retain(|question| {
            let found = ids.iter().any(|id| id.as_ref() == question.id);
            if found {
                settled.push(question.clone());
            }
            !found
        });
        self.waiting
            .retain(|id| !ids.iter().any(|settled| settled.as_ref() == id));
        settled
    }

    /// Settles the questions the agent waits for, and returns their ids.
    pub fn settle_waiting(&mut self) -> Vec<String> {
        let waiting = std::mem::take(&mut self.waiting);
        self.settle(&waiting);
        waiting
    }
}

/// `questions` with the ids after `last`, the highest number taken.
fn numbered(questions: Vec<Question>, last: u32) -> Vec<AskedQuestion> {
    (last + 1..)
        .zip(questions)
        .map(|(number, question)| AskedQuestion {
            id: format!("q{number}"),
            question,
        })
        .collect()
}

/// Whether `id` is one of the first `last` questions' ids.
fn was_asked(id: &str, last: u32) -> bool {
    number_of(id).is_some_and(|number| number >= 1 && number <= last)
}

/// The number of an id such as `q3` (also `Q3`, `3`).
fn number_of(id: &str) -> Option<u32> {
    let id = id.trim();
    let digits = id
        .strip_prefix('q')
        .or_else(|| id.strip_prefix('Q'))
        .unwrap_or(id);
    digits.parse().ok()
}

/// The id `q{n}` of an id the agent wrote as `Q3`, `3` or ` q3 `.
pub fn normalize_id(id: &str) -> Option<String> {
    number_of(id).map(|number| format!("q{number}"))
}

/// Reads `ask_user`'s `questions`: an array of questions (each an object
/// with `question` and optionally `choices`, `multiple` and `default`, or a
/// plain string), or one question as text (the form older configurations
/// and models use). Text that holds a JSON array, as an output-based call
/// carries it, is read as the array. Returns why it can't be read.
pub fn parse_questions(value: &Value) -> Result<Vec<Question>, String> {
    let questions = match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err("Questions cannot be empty.".to_string());
            }
            match serde_json::from_str::<Value>(trimmed) {
                Ok(array @ Value::Array(_)) => return parse_questions(&array),
                Ok(object @ Value::Object(_)) => vec![parse_question(&object)?],
                _ => vec![Question {
                    question: trimmed.to_string(),
                    ..Question::default()
                }],
            }
        }
        Value::Array(items) => items
            .iter()
            .map(parse_question)
            .collect::<Result<Vec<_>, _>>()?,
        Value::Object(_) => vec![parse_question(value)?],
        _ => return Err("Questions must be a list of questions.".to_string()),
    };
    if questions.is_empty() {
        return Err("Questions cannot be empty.".to_string());
    }
    Ok(questions)
}

fn parse_question(value: &Value) -> Result<Question, String> {
    let object = match value {
        Value::String(text) if !text.trim().is_empty() => {
            return Ok(Question {
                question: text.trim().to_string(),
                ..Question::default()
            });
        }
        Value::Object(object) => object,
        _ => return Err("Each question needs a `question` text.".to_string()),
    };
    let question = object
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or("Each question needs a `question` text.")?
        .to_string();
    let choices = match object.get("choices") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(text.trim().to_string()),
                Value::Number(number) => Some(number.to_string()),
                Value::Bool(flag) => Some(flag.to_string()),
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .take(MAX_CHOICES)
            .collect(),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Array(_)) => {
                let mut nested = object.clone();
                nested.insert(
                    "choices".to_string(),
                    serde_json::from_str(text).unwrap_or_default(),
                );
                return parse_question(&Value::Object(nested));
            }
            _ => vec![text.trim().to_string()]
                .into_iter()
                .filter(|text| !text.is_empty())
                .collect(),
        },
        Some(_) => return Err("A question's `choices` must be a list of texts.".to_string()),
    };
    let multiple = match object.get("multiple") {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::String(text)) => text.trim().eq_ignore_ascii_case("true"),
        _ => false,
    };
    let default = match object.get("default") {
        Some(Value::String(text)) => Some(text.trim().to_string()).filter(|text| !text.is_empty()),
        Some(Value::Number(number)) => Some(number.to_string()),
        Some(Value::Bool(flag)) => Some(flag.to_string()),
        _ => None,
    };
    Ok(Question {
        question,
        choices,
        multiple,
        default,
    })
}

/// Reads `ask_user`'s `wait`: a boolean, or `"true"` as text.
pub fn parse_wait(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::String(text)) => text.trim().eq_ignore_ascii_case("true"),
        _ => false,
    }
}

/// Reads `close_questions`' `ids`: an array of ids, or text such as
/// `"q3, q4"`. Each is normalized to `q{n}`. Returns why they can't be read.
pub fn parse_ids(value: &Value) -> Result<Vec<String>, String> {
    let raw: Vec<String> = match value {
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .collect(),
        Value::String(text) => match serde_json::from_str::<Value>(text.trim()) {
            Ok(array @ Value::Array(_)) => return parse_ids(&array),
            _ => text
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|part| !part.is_empty())
                .map(str::to_string)
                .collect(),
        },
        Value::Number(number) => vec![number.to_string()],
        _ => return Err("ids must list question ids, such as [\"q3\"].".to_string()),
    };
    let mut ids = Vec::new();
    for id in raw {
        let normalized = normalize_id(&id).ok_or_else(|| format!("'{id}' isn't a question id."))?;
        if !ids.contains(&normalized) {
            ids.push(normalized);
        }
    }
    if ids.is_empty() {
        return Err("ids must list question ids, such as [\"q3\"].".to_string());
    }
    Ok(ids)
}

/// Ids as a sentence lists them: `q3`, `q3 and q4`, `q3, q4 and q5`.
pub fn id_list<S: AsRef<str>>(ids: &[S]) -> String {
    match ids {
        [] => String::new(),
        [one] => one.as_ref().to_string(),
        [rest @ .., last] => format!(
            "{} and {}",
            rest.iter()
                .map(|id| id.as_ref())
                .collect::<Vec<_>>()
                .join(", "),
            last.as_ref()
        ),
    }
}

/// `ask_user`'s result when it returns at once: "Asked q3 and q4. Their
/// answers reach you as a note, so go on with your work."
pub fn asked_reply(asked: &[AskedQuestion]) -> String {
    let ids = asked.iter().map(|q| q.id.as_str()).collect::<Vec<_>>();
    if ids.len() == 1 {
        format!(
            "Asked {}. Its answer reaches you as a note, so go on with your work.",
            ids[0]
        )
    } else {
        format!(
            "Asked {}. Their answers reach you as a note, so go on with your work.",
            id_list(&ids)
        )
    }
}

/// The questions as text, one block each, with their ids, choices and
/// defaults: what a lead reads of a subagent's questions, and what a host
/// that only shows text shows.
///
/// ```text
/// q3 Tabs or spaces?
///    Choices: Tabs | Spaces
///    Default: Spaces
/// ```
pub fn questions_text(asked: &[AskedQuestion]) -> String {
    asked
        .iter()
        .map(question_block)
        .collect::<Vec<_>>()
        .join("\n")
}

fn question_block(asked: &AskedQuestion) -> String {
    let question = &asked.question;
    let mut block = format!("{} {}", asked.id, question.question.trim());
    if !question.choices.is_empty() {
        block.push_str(&format!("\n   Choices: {}", question.choices.join(" | ")));
        if question.multiple {
            block.push_str(" (any number)");
        }
    }
    if let Some(default) = &question.default {
        block.push_str(&format!("\n   Default: {default}"));
    }
    block
}

/// The user's answer to one question. Several parts may be set: chosen
/// choices and text of their own ("Other"). With `you_decide` the agent
/// goes on with its default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    /// The choices picked, as the question words them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
    /// An answer in the user's own words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The user leaves it to the agent.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub you_decide: bool,
}

impl Answer {
    /// Whether nothing is filled in.
    pub fn is_empty(&self) -> bool {
        !self.you_decide
            && self.choices.iter().all(|choice| choice.trim().is_empty())
            && self
                .text
                .as_deref()
                .is_none_or(|text| text.trim().is_empty())
    }

    /// The answer as the agent's note gives it.
    pub fn words(&self, question: Option<&Question>) -> String {
        if self.you_decide {
            return match question.and_then(|question| question.default.as_deref()) {
                Some(default) => format!("You decide (you said you'd assume: {default})"),
                None => "You decide".to_string(),
            };
        }
        let choices = self
            .choices
            .iter()
            .map(|choice| choice.trim())
            .filter(|choice| !choice.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        let text = self.text.as_deref().map(str::trim).unwrap_or_default();
        match (choices.is_empty(), text.is_empty()) {
            (_, true) => choices,
            (true, false) => text.to_string(),
            (false, false) => format!("{choices}. {text}"),
        }
    }
}

/// A question's id with the user's answer to it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionAnswer {
    pub id: String,
    #[serde(flatten)]
    pub answer: Answer,
}

impl QuestionAnswer {
    pub fn new(id: impl Into<String>, answer: Answer) -> Self {
        Self {
            id: id.into(),
            answer,
        }
    }
}

/// One submission of answers, some questions answered.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answers {
    /// The answers, in the order given.
    #[serde(default)]
    pub answers: Vec<QuestionAnswer>,
}

impl Answers {
    /// Whether nothing is filled in.
    pub fn is_empty(&self) -> bool {
        self.answers.iter().all(|given| given.answer.is_empty())
    }

    /// The ids answered, in order.
    pub fn ids(&self) -> Vec<String> {
        self.answers
            .iter()
            .filter(|given| !given.answer.is_empty())
            .map(|given| given.id.clone())
            .collect()
    }
}

/// The note that gives the agent one submission of answers. `asked` holds
/// the questions (for their text and defaults), `still_open` the ids that
/// stay open after it.
///
/// ```text
/// The user answered q3 and q4.
/// q3 Tabs or spaces? Spaces
/// q4 Push now? You decide (you said you'd assume: not yet)
/// Still open: q5.
/// ```
pub fn answer_note<S: AsRef<str>>(
    asked: &[AskedQuestion],
    answers: &Answers,
    still_open: &[S],
) -> String {
    let answered = answers
        .answers
        .iter()
        .filter(|given| !given.answer.is_empty())
        .collect::<Vec<_>>();
    let mut lines = Vec::new();
    let ids = answered
        .iter()
        .map(|given| given.id.as_str())
        .collect::<Vec<_>>();
    lines.push(format!("The user answered {}.", id_list(&ids)));
    for QuestionAnswer { id, answer } in answered {
        let question = asked
            .iter()
            .find(|question| &question.id == id)
            .map(|question| &question.question);
        let words = answer.words(question);
        lines.push(match question {
            Some(question) => format!("{id} {} {words}", question.question.trim()),
            None => format!("{id} {words}"),
        });
    }
    if !still_open.is_empty() {
        lines.push(format!("Still open: {}.", id_list(still_open)));
    }
    lines.join("\n")
}

/// The note that tells the agent the user dismissed questions: "The user
/// dismissed q5." The agent goes on with its defaults.
pub fn dismissed_note<S: AsRef<str>>(ids: &[S]) -> String {
    format!("The user dismissed {}.", id_list(ids))
}

/// The user's message labelled as the answer to `ids`: "(This message
/// answers q3 and q4.)" before the text.
pub fn message_answers<S: AsRef<str>>(ids: &[S], text: &str) -> String {
    if ids.is_empty() {
        return text.to_string();
    }
    format!("(This message answers {}.)\n{text}", id_list(ids))
}

/// The user's message with the questions still open named at its end:
/// "(Still open: q5.)".
pub fn still_open<S: AsRef<str>>(text: &str, ids: &[S]) -> String {
    if ids.is_empty() {
        return text.to_string();
    }
    format!("{text}\n\n(Still open: {}.)", id_list(ids))
}

/// The heading of [`open_section`].
pub const OPEN_QUESTIONS_HEADING: &str = "# Your open questions";

/// The section that restates the agent's open questions after a handover
/// or a compaction, or `None` without any.
pub fn open_section(open: &[AskedQuestion]) -> Option<String> {
    if open.is_empty() {
        return None;
    }
    Some(format!(
        "{OPEN_QUESTIONS_HEADING}\nYou asked the user these, and they haven't answered yet. Their answers reach you as notes. Close any that no longer matter with close_questions.\n{}",
        questions_text(open)
    ))
}

/// An `ask_user` call that asks `questions` (`wait`: the agent passed
/// `wait: true`): they are numbered after `last` (or the log's count, when
/// higher) and recorded in `log`. Returns them,
/// whether the agent waits for the answer (it always does without
/// `async_questions`), and the call's result for the agent.
pub(crate) fn take_ask(
    log: &mut QuestionLog,
    last: u32,
    questions: Vec<Question>,
    wait: bool,
    async_questions: bool,
) -> (Vec<AskedQuestion>, bool, String) {
    let (asked, wait, reply) = plan_ask(last.max(log.asked), questions, wait, async_questions);
    log.record(&asked, wait);
    (asked, wait, reply)
}

/// [`take_ask`] without recording the questions: they are numbered after
/// `last`, the highest number taken, and the caller records them with
/// [`QuestionLog::record`] once the call runs.
pub(crate) fn plan_ask(
    last: u32,
    questions: Vec<Question>,
    wait: bool,
    async_questions: bool,
) -> (Vec<AskedQuestion>, bool, String) {
    let wait = wait || !async_questions;
    let asked = numbered(questions, last);
    let reply = crate::tool::builtins::ask_user::call_result(&asked, wait);
    (asked, wait, reply)
}

/// A `close_questions` call that closes `ids`: those the task asked are
/// settled in `log`. Returns their ids and the call's result for the agent.
pub(crate) fn take_close(log: &mut QuestionLog, ids: Vec<String>) -> (Vec<String>, String) {
    let (known, reply) = plan_close(log.asked, ids);
    log.settle(&known);
    (known, reply)
}

/// [`take_close`] without settling them: those among the first `last`
/// questions are known.
pub(crate) fn plan_close(last: u32, ids: Vec<String>) -> (Vec<String>, String) {
    let (known, unknown): (Vec<String>, Vec<String>) =
        ids.into_iter().partition(|id| was_asked(id, last));
    let reply = crate::tool::builtins::close_questions::call_result(&known, &unknown);
    (known, reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn asked(id: &str, question: &str, default: Option<&str>) -> AskedQuestion {
        AskedQuestion {
            id: id.to_string(),
            question: Question {
                question: question.to_string(),
                default: default.map(str::to_string),
                ..Question::default()
            },
        }
    }

    #[test]
    fn questions_are_read_in_both_forms() {
        let structured = parse_questions(&json!([
            {"question": "Tabs or spaces?", "choices": ["Tabs", "Spaces"], "default": "Spaces"},
            {"question": "Which?", "choices": ["a", "b"], "multiple": true},
            "Push now?"
        ]))
        .unwrap();
        assert_eq!(structured.len(), 3);
        assert_eq!(structured[0].choices, ["Tabs", "Spaces"]);
        assert_eq!(structured[0].default.as_deref(), Some("Spaces"));
        assert!(structured[1].multiple);
        assert_eq!(structured[2].question, "Push now?");
        // Today's form: one question as text.
        let text = parse_questions(&json!("  Which file?\nAnd why?  ")).unwrap();
        assert_eq!(
            text,
            [Question {
                question: "Which file?\nAnd why?".into(),
                ..Question::default()
            }]
        );
        // An output-based call carries the array as text.
        let carried = parse_questions(&json!(
            r#"[{"question": "A?", "choices": "[\"x\", \"y\"]", "multiple": "true"}]"#
        ))
        .unwrap();
        assert_eq!(carried[0].choices, ["x", "y"]);
        assert!(carried[0].multiple);
        // At most six choices.
        let many =
            parse_questions(&json!([{"question": "N?", "choices": [1, 2, 3, 4, 5, 6, 7, 8]}]))
                .unwrap();
        assert_eq!(many[0].choices.len(), MAX_CHOICES);
        for bad in [json!(""), json!([]), json!([{"choices": ["a"]}]), json!(3)] {
            assert!(parse_questions(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn ids_go_on_across_asks_and_settle() {
        let mut log = QuestionLog::default();
        let first = log.ask(vec![
            Question {
                question: "A?".into(),
                ..Question::default()
            },
            Question {
                question: "B?".into(),
                ..Question::default()
            },
        ]);
        let second = log.ask(vec![Question {
            question: "C?".into(),
            ..Question::default()
        }]);
        assert_eq!(
            [&first[0].id, &first[1].id, &second[0].id],
            ["q1", "q2", "q3"]
        );
        assert!(log.was_asked("Q2") && !log.was_asked("q4"));
        assert_eq!(log.settle(&["q2", "q9"]).len(), 1);
        assert_eq!(
            log.open.iter().map(|q| q.id.as_str()).collect::<Vec<_>>(),
            ["q1", "q3"]
        );
        log.waiting = vec!["q3".into()];
        assert_eq!(log.settle_waiting(), ["q3"]);
        assert_eq!(log.open.len(), 1);
        let saved = serde_json::to_value(&log).unwrap();
        assert_eq!(serde_json::from_value::<QuestionLog>(saved).unwrap(), log);
        assert!(QuestionLog::default().is_empty());
    }

    #[test]
    fn close_ids_are_read_from_a_list_or_text() {
        assert_eq!(
            parse_ids(&json!(["q3", "Q4", 5, "q3"])).unwrap(),
            ["q3", "q4", "q5"]
        );
        assert_eq!(parse_ids(&json!("q3, q4")).unwrap(), ["q3", "q4"]);
        assert_eq!(parse_ids(&json!("[\"q7\"]")).unwrap(), ["q7"]);
        assert!(parse_ids(&json!("soon")).is_err());
        assert!(parse_ids(&json!([])).is_err());
    }

    #[test]
    fn the_notes_have_one_format() {
        let asked = [
            asked("q3", "Tabs or spaces?", None),
            asked("q4", "Push now?", Some("not yet")),
        ];
        assert_eq!(id_list(&["q3"]), "q3");
        assert_eq!(id_list(&["q3", "q4", "q5"]), "q3, q4 and q5");
        assert_eq!(
            asked_reply(&asked),
            "Asked q3 and q4. Their answers reach you as a note, so go on with your work."
        );
        assert_eq!(
            asked_reply(&asked[..1]),
            "Asked q3. Its answer reaches you as a note, so go on with your work."
        );
        let answers = Answers {
            answers: vec![
                QuestionAnswer::new(
                    "q3",
                    Answer {
                        choices: vec!["Spaces".into()],
                        ..Answer::default()
                    },
                ),
                QuestionAnswer::new(
                    "q4",
                    Answer {
                        you_decide: true,
                        ..Answer::default()
                    },
                ),
            ],
        };
        assert_eq!(
            answer_note(&asked, &answers, &["q5"]),
            "The user answered q3 and q4.\n\
             q3 Tabs or spaces? Spaces\n\
             q4 Push now? You decide (you said you'd assume: not yet)\n\
             Still open: q5."
        );
        let other = Answers {
            answers: vec![QuestionAnswer::new(
                "q3",
                Answer {
                    choices: vec!["Tabs".into()],
                    text: Some("but 2 wide".into()),
                    you_decide: false,
                },
            )],
        };
        assert_eq!(
            answer_note::<&str>(&asked, &other, &[]),
            "The user answered q3.\nq3 Tabs or spaces? Tabs. but 2 wide"
        );
        assert_eq!(dismissed_note(&["q5"]), "The user dismissed q5.");
        assert_eq!(
            message_answers(&["q3", "q4"], "Spaces, and not yet."),
            "(This message answers q3 and q4.)\nSpaces, and not yet."
        );
        assert_eq!(still_open("Hello", &["q5"]), "Hello\n\n(Still open: q5.)");
        assert_eq!(still_open::<&str>("Hello", &[]), "Hello");
    }

    #[test]
    fn the_text_form_keeps_the_structure() {
        let mut question = asked("q3", "Tabs or spaces?", Some("Spaces"));
        question.question.choices = vec!["Tabs".into(), "Spaces".into()];
        question.question.multiple = true;
        assert_eq!(
            questions_text(&[question.clone(), asked("q4", "Why?", None)]),
            "q3 Tabs or spaces?\n   Choices: Tabs | Spaces (any number)\n   Default: Spaces\nq4 Why?"
        );
        let section = open_section(&[question]).unwrap();
        assert!(section.starts_with(OPEN_QUESTIONS_HEADING), "{section}");
        assert!(section.contains("q3 Tabs or spaces?"), "{section}");
        assert_eq!(open_section(&[]), None);
    }
}
