//! The checks that decide whether a call may run as it is, waits for
//! approval, or is refused, shared by the native loop's calls and the
//! calls a tool makes from inside its own call ([`super::nested`]), so the
//! two can't drift.

use crate::agent_config::{AgentToolConfig, PathSafetyPolicy};
use crate::tool::ToolDefinition;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// The workspace check of a call's `is_path` parameters.
pub(super) enum PathCheck {
    Inside,
    /// This parameter's path is outside the workspace.
    Outside(String),
    /// A path couldn't be judged. The call is rejected with this reason.
    Invalid(String),
}

/// A call's paths under the path-safety policy. See [`path_verdict`].
pub(super) enum PathVerdict {
    /// The paths don't stop the call.
    Allowed,
    /// This parameter's path is outside the workspace, and the policy asks
    /// for approval.
    NeedsApproval(String),
    /// This parameter's path is outside the workspace, and the policy
    /// rejects the call.
    Outside(String),
    /// A path couldn't be judged. The call is rejected with this reason.
    Invalid(String),
}

/// Checks a call's `is_path` parameters against `workspace`, as
/// [`crate::utils::is_path_in_workspace`] resolves them. A missing, null or
/// empty value is not checked. Any other non-string value is invalid, since
/// a tool could read it as a path the check never saw.
pub(super) fn check_path_parameters(
    definition: &ToolDefinition,
    params: &HashMap<String, Value>,
    workspace: &Path,
) -> PathCheck {
    for param_def in definition.parameters.iter().filter(|param| param.is_path) {
        let path = match params.get(&param_def.name) {
            None | Some(Value::Null) => continue,
            Some(Value::String(path)) if path.is_empty() => continue,
            Some(Value::String(path)) => path,
            Some(other) => {
                return PathCheck::Invalid(format!(
                    "Path validation failed for parameter '{}'. A path must be a string, not {other}.",
                    param_def.name
                ));
            }
        };
        match crate::utils::is_path_in_workspace(path, workspace) {
            Ok(true) => {}
            Ok(false) => return PathCheck::Outside(param_def.name.clone()),
            Err(error) => {
                return PathCheck::Invalid(format!(
                    "Path validation failed for parameter '{}': {error}",
                    param_def.name
                ));
            }
        }
    }
    PathCheck::Inside
}

/// A call's `is_path` parameters checked against `workspace` (see
/// [`check_path_parameters`]), and a path outside it judged under the
/// tool's path-safety policy in `tool_config`. Each caller answers a
/// rejection its own way.
pub(super) fn path_verdict(
    tool_config: &AgentToolConfig,
    tool_name: &str,
    definition: &ToolDefinition,
    params: &HashMap<String, Value>,
    workspace: &Path,
) -> PathVerdict {
    match check_path_parameters(definition, params, workspace) {
        PathCheck::Inside => PathVerdict::Allowed,
        PathCheck::Invalid(reason) => PathVerdict::Invalid(reason),
        PathCheck::Outside(param_name) => match tool_config.path_safety_policy(tool_name) {
            PathSafetyPolicy::AutoReject => PathVerdict::Outside(param_name),
            PathSafetyPolicy::RequireApproval => PathVerdict::NeedsApproval(param_name),
            PathSafetyPolicy::AutoApprove => PathVerdict::Allowed,
        },
    }
}
