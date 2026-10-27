use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::questions::{self, AskedQuestion, Question};
use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

pub const ASK_USER_TOOL_NAME: &str = "ask_user";
pub const ASK_USER_QUESTIONS_PARAM_NAME: &str = "questions";
/// `ask_user`'s parameter that says whether the agent waits for the answer,
/// offered with
/// [`AgentToolConfig::async_questions`](crate::agent_config::AgentToolConfig::async_questions)
/// on. It is true unless the agent passes false.
pub const ASK_USER_WAIT_PARAM_NAME: &str = "wait";

const DESCRIPTION: &str = "Asks the user and waits for their answer. Ask only what you can't \
    reasonably decide yourself, and put related questions in one call.";

const ASYNC_DESCRIPTION: &str = "Asks the user and, by default, waits for the answer. Ask only \
    what you can't reasonably decide yourself, and put related questions in one call. Most \
    questions need the answer before you go on: a preference, a design choice, the scope, which \
    model or account to use, and anything that writes, changes or deletes files, spends money or \
    starts workers. Never act on your default while such a question is open. Pass wait false \
    only when you have work that no answer could change, such as reading and investigating. \
    Then the call returns the questions' ids (q1, q2, ...) at once, the answers reach you later \
    as a note, and you do only that work until they come. When a question no longer matters \
    (you found the answer, or the plan changed), close it with close_questions. When the user \
    dismisses a question, go on with your default.";

/// What both descriptions say of a reply that isn't an answer.
const NON_ANSWER: &str = "A reply that doesn't answer a question (a new request, or words \
    about something else) leaves it unanswered. Respond to what the user said, ask the question \
    again in the same words, and end with a short reminder such as \"The question about the \
    branch is still waiting for your answer.\"";

const QUESTIONS_DESCRIPTION: &str = "The questions, as a list. Each has `question` (what you \
    need to know and why it matters) and, when they help, `choices` (up to 6 short suggested \
    answers, shown as buttons), `multiple` (true lets the user pick several choices) and \
    `default` (what you'll assume if the user doesn't answer, shown to them).";

const WAIT_DESCRIPTION: &str = "Optional. Default true, which holds you until the user \
    answers. Pass false only to go on with work that no answer could change while the question \
    is open.";

#[derive(Debug, Clone)]
pub struct AskUserTool;

impl AskUserTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AskUserTool {
    fn default() -> Self {
        Self::new()
    }
}

/// The JSON schema of one question, for native tool calls.
fn question_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What you need to know and why it matters."
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: up to 6 short suggested answers, shown as buttons. The user can still answer in their own words."
            },
            "multiple": {
                "type": "boolean",
                "description": "Optional: true lets the user pick several of the choices."
            },
            "default": {
                "type": "string",
                "description": "Optional: what you'll assume if the user doesn't answer. The user sees it."
            }
        },
        "required": ["question"]
    })
}

fn schema(wait: bool) -> Value {
    let mut properties = serde_json::Map::new();
    properties.insert(
        ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
        json!({
            "type": "array",
            "items": question_schema(),
            "description": QUESTIONS_DESCRIPTION,
        }),
    );
    if wait {
        properties.insert(
            ASK_USER_WAIT_PARAM_NAME.to_string(),
            json!({"type": "boolean", "description": WAIT_DESCRIPTION}),
        );
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": [ASK_USER_QUESTIONS_PARAM_NAME],
    })
}

/// `ask_user`'s definition as an agent with `tool_config` is shown it. With
/// [`async_questions`](crate::agent_config::AgentToolConfig::async_questions)
/// on, the built-in `ask_user` offers `wait`, whose false returns at once,
/// and its description says when to pass it. Any other definition is
/// returned unchanged.
pub fn offered_definition(
    mut definition: ToolDefinition,
    tool_config: &crate::agent_config::AgentToolConfig,
) -> ToolDefinition {
    if definition.name != ASK_USER_TOOL_NAME
        || !matches!(definition.tool_type_details, ToolTypeDetails::BuiltIn)
        || !tool_config.async_questions
    {
        return definition;
    }
    definition.description = format!("{ASYNC_DESCRIPTION} {NON_ANSWER}");
    definition.parameters.push(ToolParameterDefinition {
        name: ASK_USER_WAIT_PARAM_NAME.to_string(),
        description: WAIT_DESCRIPTION.to_string(),
        param_type: ToolParameterType::Bool,
        required: false,
        is_path: false,
        flag: None,
        default_value: None,
    });
    definition.raw_json_schema_override = Some(schema(true));
    definition
}

/// What an `ask_user` call asks, read from its result: the questions and
/// whether the agent waits (true unless it passed `wait: false`). The
/// built-in tool's result holds them as JSON. Any other text (a host's own
/// `ask_user`, or a result the host supplied on approval) is one question,
/// as `ask_user` took them before it took a list, and the agent waits.
pub fn read_call(result: &str) -> (Vec<Question>, bool) {
    if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(result.trim())
        && let Some(value) = object.get(ASK_USER_QUESTIONS_PARAM_NAME)
        && let Ok(asked) = questions::parse_questions(value)
    {
        return (
            asked,
            questions::parse_wait(object.get(ASK_USER_WAIT_PARAM_NAME)),
        );
    }
    let question = Question {
        question: result.trim().to_string(),
        ..Question::default()
    };
    (vec![question], true)
}

/// The result an `ask_user` call that asked `asked` returns to the agent:
/// the questions as text when the agent waits for the answer, else
/// [`questions::asked_reply`].
pub fn call_result(asked: &[AskedQuestion], wait: bool) -> String {
    if wait {
        questions::questions_text(asked)
    } else {
        questions::asked_reply(asked)
    }
}

#[async_trait]
impl Tool for AskUserTool {
    /// Asking again only returns the questions. A call running when its
    /// task stopped asks them again.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: ASK_USER_TOOL_NAME.to_string(),
            description: format!("{DESCRIPTION} {NON_ANSWER}"),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
                description: format!(
                    "{QUESTIONS_DESCRIPTION} Write the list as JSON, or one question as plain text."
                ),
                param_type: ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: Some(schema(false)),
        }
    }

    /// Checks the questions, and returns them with `wait` as JSON, which
    /// the runtime reads ([`read_call`]) to number them and ask.
    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let value = params.get(ASK_USER_QUESTIONS_PARAM_NAME).ok_or_else(|| {
            ToolError::MissingParameter {
                param_name: ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
            }
        })?;
        let asked = questions::parse_questions(value).map_err(|message| {
            ToolError::ParameterValidation {
                param_name: ASK_USER_QUESTIONS_PARAM_NAME.to_string(),
                message,
            }
        })?;
        let wait = questions::parse_wait(params.get(ASK_USER_WAIT_PARAM_NAME));
        Ok(json!({
            ASK_USER_QUESTIONS_PARAM_NAME: asked,
            ASK_USER_WAIT_PARAM_NAME: wait,
        })
        .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::AgentToolConfig;

    async fn run(params: Value) -> Result<String, ToolError> {
        let dir = std::env::temp_dir();
        let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &dir);
        AskUserTool
            .execute(&context, serde_json::from_value(params).unwrap())
            .await
    }

    #[tokio::test]
    async fn the_tool_takes_a_list_or_one_question_as_text() {
        let listed = run(json!({
            "questions": [{"question": "Tabs or spaces?", "choices": ["Tabs", "Spaces"]}],
            "wait": false
        }))
        .await
        .unwrap();
        let (asked, wait) = read_call(&listed);
        assert!(!wait);
        assert_eq!(asked[0].choices, ["Tabs", "Spaces"]);
        // Without `wait`, the agent waits.
        let text = run(json!({"questions": "Which file?"})).await.unwrap();
        let (asked, wait) = read_call(&text);
        assert!(wait);
        assert_eq!(asked[0].question, "Which file?");
        // A host's own result text is one question, which the agent waits for.
        assert_eq!(
            read_call("Why?"),
            (
                vec![Question {
                    question: "Why?".into(),
                    ..Question::default()
                }],
                true
            )
        );
        assert!(run(json!({"questions": "  "})).await.is_err());
        assert!(run(json!({})).await.is_err());
    }

    #[test]
    fn only_async_questions_offer_wait() {
        let mut config = AgentToolConfig::default();
        let plain = offered_definition(AskUserTool.definition(), &config);
        assert!(plain.parameters.iter().all(|p| p.name != "wait"));
        assert!(plain.description.contains("waits for their answer"));
        assert!(plain.to_json_schema()["properties"]["questions"]["type"] == "array");
        config.async_questions = true;
        let offered = offered_definition(AskUserTool.definition(), &config);
        assert!(offered.parameters.iter().any(|p| p.name == "wait"));
        assert!(offered.to_json_schema()["properties"]["wait"]["type"] == "boolean");
        assert!(
            offered
                .description
                .contains("by default, waits for the answer")
        );
        assert!(offered.description.contains("Pass wait false only"));
        assert!(WAIT_DESCRIPTION.starts_with("Optional. Default true"));
        // Both say what to do with a reply that isn't an answer.
        for description in [plain.description, offered.description] {
            assert!(description.ends_with(NON_ANSWER), "{description}");
        }
    }
}
