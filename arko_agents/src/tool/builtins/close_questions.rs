use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::questions;
use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

pub const CLOSE_QUESTIONS_TOOL_NAME: &str = "close_questions";
pub const CLOSE_QUESTIONS_IDS_PARAM_NAME: &str = "ids";
pub const CLOSE_QUESTIONS_WHY_PARAM_NAME: &str = "why";

const IDS_DESCRIPTION: &str = "The ids of the questions to close, such as [\"q3\", \"q4\"].";
const WHY_DESCRIPTION: &str = "Optional: why, in a few words the user sees (\"Found it in the \
    README\", \"No longer needed\").";

/// Closes questions the agent asked with `ask_user` that no longer need an
/// answer: it found the answer, or they no longer matter. The runtime
/// settles them in the task's [`QuestionLog`](crate::questions::QuestionLog)
/// and reports them with
/// [`AgentEvent::QuestionsClosed`](crate::AgentEvent::QuestionsClosed), so
/// the host stops showing them as open.
#[derive(Debug, Clone, Default)]
pub struct CloseQuestionsTool;

impl CloseQuestionsTool {
    pub fn new() -> Self {
        Self
    }
}

/// What a `close_questions` call closes, read from its result: the ids and
/// why. `None` for a result the built-in tool didn't write.
pub fn read_call(result: &str) -> Option<(Vec<String>, Option<String>)> {
    let Ok(Value::Object(object)) = serde_json::from_str::<Value>(result.trim()) else {
        return None;
    };
    let ids = questions::parse_ids(object.get(CLOSE_QUESTIONS_IDS_PARAM_NAME)?).ok()?;
    let why = object
        .get(CLOSE_QUESTIONS_WHY_PARAM_NAME)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|why| !why.is_empty())
        .map(str::to_string);
    Some((ids, why))
}

/// The result a `close_questions` call returns: the questions it closed,
/// and the ids the task never asked.
pub fn call_result(closed: &[String], unknown: &[String]) -> String {
    let mut parts = Vec::new();
    if !closed.is_empty() {
        parts.push(format!("Closed {}.", questions::id_list(closed)));
    }
    match unknown {
        [] => {}
        [one] => parts.push(format!("You never asked {one}.")),
        many => parts.push(format!("You never asked {}.", questions::id_list(many))),
    }
    parts.join(" ")
}

#[async_trait]
impl Tool for CloseQuestionsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: CLOSE_QUESTIONS_TOOL_NAME.to_string(),
            description: "Closes questions you asked with ask_user that no longer need an \
                answer, because you found the answer or they no longer matter. The user sees \
                them closed. Closing is never required."
                .to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: CLOSE_QUESTIONS_IDS_PARAM_NAME.to_string(),
                    description: IDS_DESCRIPTION.to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: CLOSE_QUESTIONS_WHY_PARAM_NAME.to_string(),
                    description: WHY_DESCRIPTION.to_string(),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            read_only: false,
            raw_json_schema_override: Some(json!({
                "type": "object",
                "properties": {
                    CLOSE_QUESTIONS_IDS_PARAM_NAME: {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": IDS_DESCRIPTION,
                    },
                    CLOSE_QUESTIONS_WHY_PARAM_NAME: {
                        "type": "string",
                        "description": WHY_DESCRIPTION,
                    },
                },
                "required": [CLOSE_QUESTIONS_IDS_PARAM_NAME],
            })),
        }
    }

    /// Checks the ids, and returns them with `why` as JSON, which the
    /// runtime reads ([`read_call`]) to close them.
    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let value = params.get(CLOSE_QUESTIONS_IDS_PARAM_NAME).ok_or_else(|| {
            ToolError::MissingParameter {
                param_name: CLOSE_QUESTIONS_IDS_PARAM_NAME.to_string(),
            }
        })?;
        let ids =
            questions::parse_ids(value).map_err(|message| ToolError::ParameterValidation {
                param_name: CLOSE_QUESTIONS_IDS_PARAM_NAME.to_string(),
                message,
            })?;
        let why = params
            .get(CLOSE_QUESTIONS_WHY_PARAM_NAME)
            .and_then(Value::as_str)
            .unwrap_or_default();
        Ok(json!({
            CLOSE_QUESTIONS_IDS_PARAM_NAME: ids,
            CLOSE_QUESTIONS_WHY_PARAM_NAME: why,
        })
        .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_tool_reads_ids_and_why() {
        let dir = std::env::temp_dir();
        let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &dir);
        let result = CloseQuestionsTool
            .execute(
                &context,
                serde_json::from_value(json!({"ids": ["q3", "Q4"], "why": " Found it "})).unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            read_call(&result),
            Some((vec!["q3".into(), "q4".into()], Some("Found it".into())))
        );
        assert!(
            CloseQuestionsTool
                .execute(
                    &context,
                    serde_json::from_value(json!({"ids": "soon"})).unwrap()
                )
                .await
                .is_err()
        );
        assert_eq!(
            call_result(&["q3".into()], &["q9".into()]),
            "Closed q3. You never asked q9."
        );
    }
}
