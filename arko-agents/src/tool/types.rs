use serde::{Deserialize, Serialize};

/// Details about the type of tool and how it should be executed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolTypeDetails {
    BuiltIn,
    DeveloperDefined,
    EndUserDefined {
        /// The command to run, e.g. `"~/bin/my_script.sh --verbose"`: `~`
        /// and environment variables are expanded, and the result is split
        /// on whitespace (quotes are not interpreted) into the program and
        /// its first arguments. The template has no placeholders; the
        /// arguments from `ToolDefinition::parameters` follow it: string
        /// parameters without a `flag` as positional arguments, in order,
        /// then flagged string parameters as `flag value` and each true
        /// boolean parameter as its flag.
        command_template: String,
    },
}
