use super::types::ToolTypeDetails;
use crate::error::ArkoAgentsError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolParameterType {
    #[serde(rename = "string")]
    String,
    #[serde(rename = "bool")]
    Bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// `"true"` or `"false"`, in any case and with surrounding whitespace.
pub(crate) fn parse_bool_text(text: &str) -> Option<bool> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("true") {
        Some(true)
    } else if text.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolParameterDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub param_type: ToolParameterType,
    pub required: bool,
    /// The value is a path the runtime checks against the workspace before
    /// the call runs, as [`crate::utils::resolve_path`] resolves it: taken
    /// literally, without `~` or `$VAR` expansion. A tool with such a
    /// parameter should open the path that function returns, so the check
    /// judges the path the tool opens.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_path: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_value: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_based_example: Option<String>,
    pub parameters: Vec<ToolParameterDefinition>,
    /// Whether calls to this tool may run at the same time as other
    /// parallel-safe calls of the same response (native mode), and be
    /// batched in one output-based response. Set it only for tools whose
    /// calls don't affect each other, such as reads; calls that write, or
    /// whose order matters, run one at a time in the order the model made
    /// them. An agent's configuration can override it per tool; see
    /// [`AgentToolConfig`](crate::agent_config::AgentToolConfig). Definitions
    /// saved with the former `supports_batched_calls` load it here.
    #[serde(default, alias = "supports_batched_calls")]
    pub parallel_safe: bool,
    pub tool_type_details: ToolTypeDetails,
    /// When set, `to_json_schema()` and `to_arko_ai_tool_definition()` use
    /// this value verbatim instead of generating a schema from `parameters`.
    /// This is used by MCP tools whose input schemas may contain types
    /// (integer, number, array, nested objects) that the simplified
    /// `ToolParameterDefinition` model cannot represent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_json_schema_override: Option<Value>,
}

impl ToolDefinition {
    pub fn load_from_path(path: &Path) -> Result<Self, ArkoAgentsError> {
        let file_content =
            std::fs::read_to_string(path).map_err(|e| ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            })?;
        serde_json::from_str(&file_content).map_err(ArkoAgentsError::Json)
    }

    /// Saves the definition to a JSON file, replacing it in one step (see
    /// [`write_file_atomically`](crate::utils::write_file_atomically)).
    pub fn save_to_path(&self, path: &Path) -> Result<(), ArkoAgentsError> {
        let json_string = serde_json::to_string_pretty(self)?;
        crate::utils::write_file_atomically(path, json_string.as_bytes()).map_err(|e| {
            ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            }
        })
    }

    /// Gives text arguments their declared types. Output-based tool calls
    /// carry every value as a string: a `bool` parameter's `"true"` or
    /// `"false"` (in any case, surrounding whitespace ignored) becomes a
    /// JSON boolean. Other values are left as they are.
    pub fn coerce_text_arguments(&self, params: &mut std::collections::HashMap<String, Value>) {
        for def in &self.parameters {
            if def.param_type == ToolParameterType::Bool
                && let Some(value) = params.get_mut(&def.name)
                && let Some(boolean) = value.as_str().and_then(parse_bool_text)
            {
                *value = Value::Bool(boolean);
            }
        }
    }

    pub fn validate_params(
        &self,
        params: &std::collections::HashMap<String, Value>,
    ) -> Result<(), super::error::ToolError> {
        for def in &self.parameters {
            match params.get(&def.name) {
                Some(value) => match def.param_type {
                    ToolParameterType::String => {
                        if !value.is_string() {
                            return Err(super::error::ToolError::ParameterValidation {
                                param_name: def.name.clone(),
                                message: format!("Expected a string value, got {:?}", value),
                            });
                        }
                    }
                    ToolParameterType::Bool => {
                        if !value.is_boolean() {
                            return Err(super::error::ToolError::ParameterValidation {
                                param_name: def.name.clone(),
                                message: format!("Expected a boolean value, got {:?}", value),
                            });
                        }
                    }
                },
                None => {
                    if def.required && def.default_value.is_none() {
                        return Err(super::error::ToolError::MissingParameter {
                            param_name: def.name.clone(),
                        });
                    }
                }
            }
        }
        for param_name in params.keys() {
            if !self.parameters.iter().any(|def| &def.name == param_name) {
                return Err(super::error::ToolError::ParameterValidation {
                    param_name: param_name.clone(),
                    message: "Unknown parameter provided".to_string(),
                });
            }
        }
        Ok(())
    }

    pub fn to_json_schema(&self) -> serde_json::Value {
        if let Some(ref schema) = self.raw_json_schema_override {
            return schema.clone();
        }

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let type_str = match param.param_type {
                ToolParameterType::String => "string",
                ToolParameterType::Bool => "boolean",
            };
            let prop = serde_json::json!({
                "type": type_str,
                "description": param.description,
            });
            properties.insert(param.name.clone(), prop);
            if param.required {
                required.push(serde_json::Value::String(param.name.clone()));
            }
        }

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }

    pub fn to_arko_ai_tool_definition(&self) -> arko_ai::ToolDefinition {
        // `required: false` means omission in this API, not a required nullable field.
        // Responses may normalize an unspecified strict mode and make optional fields
        // mandatory. Preserve the schema exactly using arko_ai's existing primitive.
        arko_ai::ToolDefinition::new(&self.name, &self.description, self.to_json_schema())
            .with_strict(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn native_conversion_preserves_omitted_optional_parameters() {
        let definition = ToolDefinition {
            name: "read_reference".into(),
            description: "Read exact evidence".into(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "path".into(),
                    description: "Path".into(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "section".into(),
                    description: "Optional heading".into(),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        };
        let params = HashMap::from([("path".into(), serde_json::json!("design.md"))]);
        definition.validate_params(&params).unwrap();
        let native = definition.to_arko_ai_tool_definition();
        assert_eq!(native.strict, Some(false));
        assert_eq!(native.parameters["required"], serde_json::json!(["path"]));
        assert_eq!(native.parameters["properties"]["section"]["type"], "string");
        assert!(definition.validate_params(&HashMap::new()).is_err());
        let wire = serde_json::to_value(native).unwrap();
        assert_eq!(wire["strict"], false);
    }

    #[test]
    fn text_arguments_of_boolean_parameters_become_booleans() {
        let parameter = |name: &str, param_type| ToolParameterDefinition {
            name: name.into(),
            description: name.into(),
            param_type,
            required: false,
            is_path: false,
            flag: None,
            default_value: None,
        };
        let definition = ToolDefinition {
            name: "fixture".into(),
            description: "Fixture".into(),
            output_based_example: None,
            parameters: vec![
                parameter("a", ToolParameterType::Bool),
                parameter("b", ToolParameterType::Bool),
                parameter("c", ToolParameterType::Bool),
                parameter("d", ToolParameterType::Bool),
                parameter("text", ToolParameterType::String),
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        };
        let mut params = HashMap::from([
            ("a".to_string(), serde_json::json!(" TRUE\n")),
            ("b".to_string(), serde_json::json!("False")),
            ("c".to_string(), serde_json::json!("yes")),
            ("d".to_string(), serde_json::json!(true)),
            ("text".to_string(), serde_json::json!("true")),
        ]);
        definition.coerce_text_arguments(&mut params);
        assert_eq!(params["a"], serde_json::json!(true));
        assert_eq!(params["b"], serde_json::json!(false));
        // Other text is left for the tool to reject, and strings stay strings.
        assert_eq!(params["c"], serde_json::json!("yes"));
        assert_eq!(params["d"], serde_json::json!(true));
        assert_eq!(params["text"], serde_json::json!("true"));
    }
}
