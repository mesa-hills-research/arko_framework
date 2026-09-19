use serde::{Deserialize, Serialize};

/// Details about the type of tool and how it should be executed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolTypeDetails {
    BuiltIn,
    DeveloperDefined,
    EndUserDefined {
        /// The command template to execute, e.g., "my_script.sh {arg1} --flag {arg2}".
        /// The parameters for this command are defined in ToolDefinition.parameters.
        command_template: String,
    },
}
